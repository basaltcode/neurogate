from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from neurogate.config import AdhocResolveError, build_adhoc_provider
from neurogate.errors import ErrorCategory, classify
from neurogate.metrics import request_duration_seconds, requests_total
from neurogate.providers import (
    AudioGenerationResult,
    AudioSpeechResult,
    AudioTranscribeResult,
    EmbeddingResult,
    ImageGenerationResult,
    ModerationResult,
    Provider,
    ProviderCallResult,
    RerankResult,
    TranslationResult,
)
from neurogate.stats import RateTracker

log = logging.getLogger(__name__)

# Short default backoff — breaks the request-spam loop on community mirrors
# (libretranslate:fedilab) without long-banning a provider on a transient blip.
_DEFAULT_RATE_LIMIT_COOLDOWN_S = 60

# MyMemory returns the exact reset window in its error body.
_MYMEMORY_RESET_RE = re.compile(
    r"NEXT AVAILABLE IN\s+(\d+)\s+HOURS?\s+(\d+)\s+MINUTES?\s+(\d+)\s+SECONDS?",
    re.IGNORECASE,
)


def _parse_rate_limit_cooldown(provider_name: str, error_msg: str) -> int:
    """Cooldown duration in seconds for a rate-limited provider.

    If the upstream tells us when the window resets (mymemory embeds it in the
    body), use that. Otherwise fall back to a short default — long enough to
    stop the chain hammering the same 429-ing provider on every retry, short
    enough that a transient minute-rate spike clears on its own.
    """
    if provider_name.startswith("mymemory"):
        m = _MYMEMORY_RESET_RE.search(error_msg)
        if m:
            h, mi, s = (int(x) for x in m.groups())
            return h * 3600 + mi * 60 + s + 30  # +30s safety margin
    return _DEFAULT_RATE_LIMIT_COOLDOWN_S


class LLMRouter:
    """Try providers in chain order. Fallback decisions are driven by ErrorCategory:

    - RATE_LIMIT / SERVER_ERROR / TIMEOUT / EMPTY → try next provider
    - CONTEXT_EXCEEDED → try next, but also skip downstream providers with
      known-smaller context windows (the same request would overflow them too).
    - MODEL_DEAD → try next and log loudly (config needs updating).
    - OTHER (auth / bad request / unknown) → raise immediately, no fallback.

    Pre-flight skips — no network call, counted in metrics:
    - NO_TOOL_SUPPORT: provider can't do tools but tools are requested.
    - NO_REASONING_SUPPORT: caller requested reasoning but provider has none.
    - RATE_CAPPED: local RPM/RPD budget for this window is exhausted.
    """

    def __init__(
        self,
        chains: dict[str, list[Provider]],
        default: str,
        rate_tracker: RateTracker | None = None,
        providers_by_name: dict[str, Provider] | None = None,
    ) -> None:
        if not chains:
            raise ValueError("LLMRouter needs at least one chain")
        if default not in chains:
            raise ValueError(f"default chain {default!r} not in chains {list(chains)}")
        if any(not providers for providers in chains.values()):
            raise ValueError("every chain needs at least one provider")
        self._chains = chains
        self._default = default
        self._rate_tracker = rate_tracker
        if providers_by_name is None:
            providers_by_name = {}
            for providers in chains.values():
                for provider in providers:
                    providers_by_name.setdefault(provider.name, provider)
        self._providers_by_name = providers_by_name
        # Ad-hoc providers built on demand from `kind:model_id` strings.
        # Cached by full model string so each unique ad-hoc model gets one
        # HTTP client across requests. Cleared on hot reload.
        self._adhoc_cache: dict[str, Provider] = {}

    @property
    def chains(self) -> dict[str, list[str]]:
        return {name: [p.name for p in chain] for name, chain in self._chains.items()}

    @property
    def default_chain(self) -> str:
        return self._default

    def chain_names(self) -> list[str]:
        return list(self._chains)

    def provider_names(self) -> list[str]:
        return list(self._providers_by_name)

    def update_state(
        self,
        chains: dict[str, list[Provider]],
        default: str,
        providers_by_name: dict[str, Provider],
    ) -> None:
        """Atomically swap chains/default/providers in-place. Used by the chains
        editor's hot reload — endpoints that captured `router` in a closure see
        the new state on the next request without process restart.

        In-flight requests keep their already-resolved provider list (Python
        rebinds names, not the lists they pointed to), so they finish on the
        old shape — same semantics as a graceful rollout.
        """
        if not chains:
            raise ValueError("update_state needs at least one chain")
        if default not in chains:
            raise ValueError(f"default chain {default!r} not in chains {list(chains)}")
        if any(not providers for providers in chains.values()):
            raise ValueError("every chain needs at least one provider")
        self._chains = chains
        self._default = default
        self._providers_by_name = providers_by_name
        self._adhoc_cache.clear()

    def resolve_chain(self, chain_name: str | None) -> tuple[str, list[Provider]]:
        """Resolve target by name. A chain name expands to its provider list;
        a provider name expands to a single-provider chain (no fallback).

        If the value isn't a known chain/provider but looks like a `kind:model_id`
        ad-hoc reference (`openai:gpt-5-foo`, `groq/llama-99b`), we build a
        one-shot OpenAICompatProvider using a server-side env-key for the kind.
        Unknown kinds / missing env keys raise AdhocResolveError (mapped to 400
        in main.py). Anything else falls back to the default chain.
        """
        if chain_name:
            if chain_name in self._chains:
                return chain_name, self._chains[chain_name]
            provider = self._providers_by_name.get(chain_name)
            if provider is not None:
                return chain_name, [provider]
            cached = self._adhoc_cache.get(chain_name)
            if cached is not None:
                return chain_name, [cached]
            if ":" in chain_name or "/" in chain_name:
                # Looks like a kind-prefixed ad-hoc model — propagate build
                # errors (unknown kind, missing env key) as AdhocResolveError
                # instead of silently fall-back to default.
                provider = build_adhoc_provider(chain_name)
                self._adhoc_cache[chain_name] = provider
                self._providers_by_name[chain_name] = provider
                return chain_name, [provider]
        return self._default, self._chains[self._default]

    def _preflight_skip(
        self,
        provider: Provider,
        *,
        tools: list[dict[str, Any]] | None,
        min_context: int | None,
        wants_reasoning: bool = False,
        has_images: bool = False,
    ) -> ErrorCategory | None:
        """Return the category for the pre-flight skip, or None if the provider is eligible."""
        if tools and not getattr(provider, "supports_tools", True):
            return ErrorCategory.NO_TOOL_SUPPORT
        if wants_reasoning and not getattr(provider, "reasoning", False):
            return ErrorCategory.NO_REASONING_SUPPORT
        if has_images and not getattr(provider, "supports_vision", False):
            return ErrorCategory.NO_VISION_SUPPORT
        if min_context is not None:
            ctx = getattr(provider, "context_window", None)
            if ctx is not None and ctx <= min_context:
                return ErrorCategory.CONTEXT_EXCEEDED
        rpd = getattr(provider, "rpd", None)
        rpm = getattr(provider, "rpm", None)
        if self._rate_tracker is not None and not self._rate_tracker.is_available(
            provider.name, rpd=rpd, rpm=rpm
        ):
            return ErrorCategory.RATE_CAPPED
        return None

    @staticmethod
    def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
        """True if any message carries multimodal content (image_url block).

        OpenAI/Anthropic-style content arrays with `{"type": "image_url", ...}` или
        `{"type": "image", ...}` блоками. Используется в pre-flight для пропуска
        text-only провайдеров — иначе они отдают HTTP 400 "content must be a string"
        и роутер считает это non-retryable, обрывая цепочку.
        """
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype in ("image_url", "image", "input_image"):
                    return True
        return False

    @staticmethod
    def _wants_reasoning(request_extras: dict[str, Any] | None) -> bool:
        """True if the caller asked for thinking/reasoning. Skips providers that
        don't support it so the chain falls through to a reasoning-capable peer
        instead of 400-ing on `Unsupported parameter(s): reasoning`.

        Triggers on either OpenAI-style `reasoning_effort` (low/medium/high) or
        the `reasoning` object that claude-code-router forwards from Claude API
        thinking blocks. `reasoning_effort: "none"` is an explicit opt-out.
        """
        if not request_extras:
            return False
        effort = request_extras.get("reasoning_effort")
        if isinstance(effort, str) and effort and effort != "none":
            return True
        reasoning = request_extras.get("reasoning")
        if isinstance(reasoning, dict) and reasoning:
            return True
        if reasoning is True:
            return True
        return False

    def _handle_failure(
        self, provider: Provider, exc: BaseException, elapsed: float
    ) -> ErrorCategory:
        category = classify(exc)
        requests_total.labels(provider=provider.name, outcome=category.value).inc()
        request_duration_seconds.labels(
            provider=provider.name, outcome=category.value
        ).observe(elapsed)
        if category is ErrorCategory.MODEL_DEAD:
            log.error(
                "config-stale: provider %s reports model decommissioned — remove from config (%s)",
                provider.name,
                str(exc)[:200],
            )
        if category is ErrorCategory.RATE_LIMIT and self._rate_tracker is not None:
            cooldown_s = _parse_rate_limit_cooldown(provider.name, str(exc))
            until_ts = int(time.time()) + cooldown_s
            self._rate_tracker.set_cooldown(provider.name, until_ts=until_ts)
            log.warning(
                "rate-limited: %s cooldown %ss (%s)",
                provider.name,
                cooldown_s,
                str(exc)[:160],
            )
        # FreeTheAi daily-checkin lockout — one shared key gates ALL freetheai*
        # providers, so a single 403 daily_checkin_required means none of them
        # will work until manual /checkin in Discord. Propagate the cooldown to
        # every freetheai* provider so the chain stops wasting retries on them.
        # 60s window: balance between responsive recovery (after /checkin, full
        # restoration in ≤60s) and probe-noise during a stale-key window
        # (1 wasted retry per minute on the first chain-pass after expiry —
        # if still 403, breaker re-arms for another 60s).
        if (
            "daily_checkin_required" in str(exc)
            and provider.name.startswith(("freetheai:", "freetheai_image:"))
            and self._rate_tracker is not None
        ):
            until_ts = int(time.time()) + 60
            for sibling_name in self._providers_by_name:
                if sibling_name.startswith(("freetheai:", "freetheai_image:")):
                    self._rate_tracker.set_cooldown(sibling_name, until_ts=until_ts)
            log.warning(
                "freetheai daily_checkin_required — locking ALL freetheai* providers for 60s"
            )
        return category

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        chain_name: str | None = None,
    ) -> tuple[ProviderCallResult, str, str]:
        resolved_name, providers = self.resolve_chain(chain_name)
        # Chain `web` signals every provider to attach its native web-search tool
        # (google_search for Gemini, browser_search for Groq gpt-oss). OR `:online`
        # providers ignore the flag — their model-id suffix handles search server-side.
        web_search = resolved_name == "web"
        wants_reasoning = self._wants_reasoning(request_extras)
        has_images = self._messages_have_images(messages)
        last_exc: BaseException | None = None
        skipped: list[tuple[str, str]] = []
        # Updated when a CONTEXT_EXCEEDED failure tells us the request needs a bigger window.
        min_context: int | None = None

        for provider in providers:
            pre = self._preflight_skip(
                provider,
                tools=tools,
                min_context=min_context,
                wants_reasoning=wants_reasoning,
                has_images=has_images,
            )
            if pre is not None:
                skipped.append((provider.name, pre.value))
                requests_total.labels(provider=provider.name, outcome=pre.value).inc()
                log.info("skipping %s: %s", provider.name, pre.value)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await provider.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    request_extras=request_extras,
                    web_search=web_search,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                if category.skips_smaller_context:
                    ctx = getattr(provider, "context_window", None)
                    if ctx is not None:
                        min_context = max(min_context or 0, ctx)
                log.warning(
                    "provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            detail = ", ".join(f"{n}={r}" for n, r in skipped)
            raise RuntimeError(
                f"chain {resolved_name}: all providers skipped pre-flight: {detail}"
            )
        raise last_exc

    async def chat_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        request_extras: dict[str, Any] | None = None,
        chain_name: str | None = None,
    ) -> AsyncIterator[tuple[bytes, str, str]]:
        # Fallback semantics for streaming: once a provider yields its first chunk,
        # we commit to it — mid-stream failures surface as a truncated stream rather
        # than a transparent reconnect. Simpler and matches OpenAI's own behavior.
        resolved_name, providers = self.resolve_chain(chain_name)
        web_search = resolved_name == "web"
        wants_reasoning = self._wants_reasoning(request_extras)
        has_images = self._messages_have_images(messages)
        last_exc: BaseException | None = None
        skipped: list[tuple[str, str]] = []
        min_context: int | None = None

        for provider in providers:
            pre = self._preflight_skip(
                provider,
                tools=tools,
                min_context=min_context,
                wants_reasoning=wants_reasoning,
                has_images=has_images,
            )
            if pre is not None:
                skipped.append((provider.name, pre.value))
                requests_total.labels(provider=provider.name, outcome=pre.value).inc()
                log.info("skipping %s: %s", provider.name, pre.value)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)

            started = time.monotonic()
            stream = provider.chat_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                request_extras=request_extras,
                web_search=web_search,
            )
            try:
                first_chunk = await stream.__anext__()
            except StopAsyncIteration:
                exc = RuntimeError(f"{provider.name} empty response")
                elapsed = time.monotonic() - started
                self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                log.warning("provider %s produced no chunks, trying next", provider.name)
                continue
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "provider %s non-retryable stream error (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                if category.skips_smaller_context:
                    ctx = getattr(provider, "context_window", None)
                    if ctx is not None:
                        min_context = max(min_context or 0, ctx)
                log.warning(
                    "provider %s stream failed before first chunk (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("recovered via %s (stream)", provider.name)

            yield first_chunk, provider.name, resolved_name
            try:
                async for chunk in stream:
                    yield chunk, provider.name, resolved_name
            except Exception as exc:
                log.warning(
                    "provider %s mid-stream error: %s", provider.name, str(exc)[:200]
                )
            return

        if last_exc is None:
            detail = ", ".join(f"{n}={r}" for n, r in skipped)
            raise RuntimeError(
                f"chain {resolved_name}: all providers skipped pre-flight: {detail}"
            )
        raise last_exc

    async def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        mime_type: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float | None = None,
        chain_name: str | None = None,
    ) -> tuple[AudioTranscribeResult, str, str]:
        """Run audio-transcription fallback chain. Skips providers without
        `supports_audio=True` (they'd raise NotImplementedError). Otherwise the
        retry semantics match `chat()` — rate_limit/server/timeout → next provider."""
        resolved_name, providers = self.resolve_chain(chain_name)
        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            if not getattr(provider, "supports_audio", False):
                skipped.append(provider.name)
                log.info("transcribe: skipping %s (no audio support)", provider.name)
                continue

            rpd = getattr(provider, "rpd", None)
            rpm = getattr(provider, "rpm", None)
            if self._rate_tracker is not None and not self._rate_tracker.is_available(
                provider.name, rpd=rpd, rpm=rpm
            ):
                skipped.append(provider.name)
                requests_total.labels(
                    provider=provider.name, outcome=ErrorCategory.RATE_CAPPED.value
                ).inc()
                log.info("transcribe: skipping %s (rate_capped)", provider.name)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await provider.transcribe(
                    audio=audio,
                    filename=filename,
                    mime_type=mime_type,
                    language=language,
                    prompt=prompt,
                    response_format=response_format,
                    temperature=temperature,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "transcribe: provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "transcribe: provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("transcribe: recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no audio-capable providers (skipped: {skipped})"
            )
        raise last_exc

    async def generate_images(
        self,
        *,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
        chain_name: str | None = None,
    ) -> tuple[ImageGenerationResult, str, str]:
        """Run image-generation fallback chain. Skips providers without
        `supports_images=True`. Retry semantics match `transcribe()`."""
        resolved_name, providers = self.resolve_chain(chain_name)
        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            if not getattr(provider, "supports_images", False):
                skipped.append(provider.name)
                log.info("generate_images: skipping %s (no image support)", provider.name)
                continue

            rpd = getattr(provider, "rpd", None)
            rpm = getattr(provider, "rpm", None)
            if self._rate_tracker is not None and not self._rate_tracker.is_available(
                provider.name, rpd=rpd, rpm=rpm
            ):
                skipped.append(provider.name)
                requests_total.labels(
                    provider=provider.name, outcome=ErrorCategory.RATE_CAPPED.value
                ).inc()
                log.info("generate_images: skipping %s (rate_capped)", provider.name)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await provider.generate_images(
                    prompt=prompt,
                    n=n,
                    size=size,
                    response_format=response_format,
                    extra=extra,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "generate_images: provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "generate_images: provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("generate_images: recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no image-capable providers (skipped: {skipped})"
            )
        raise last_exc

    async def edit_images(
        self,
        *,
        image: str,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
        extra: dict[str, Any] | None = None,
        chain_name: str | None = None,
    ) -> tuple[ImageGenerationResult, str, str]:
        """Run image-edit fallback chain. Skips providers without
        `supports_image_edit=True` (most image providers are gen-only).
        """
        resolved_name, providers = self.resolve_chain(chain_name)
        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            if not getattr(provider, "supports_image_edit", False):
                skipped.append(provider.name)
                log.info("edit_images: skipping %s (no edit support)", provider.name)
                continue

            rpd = getattr(provider, "rpd", None)
            rpm = getattr(provider, "rpm", None)
            if self._rate_tracker is not None and not self._rate_tracker.is_available(
                provider.name, rpd=rpd, rpm=rpm
            ):
                skipped.append(provider.name)
                requests_total.labels(
                    provider=provider.name, outcome=ErrorCategory.RATE_CAPPED.value
                ).inc()
                log.info("edit_images: skipping %s (rate_capped)", provider.name)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await provider.edit_images(
                    image=image,
                    prompt=prompt,
                    n=n,
                    size=size,
                    response_format=response_format,
                    extra=extra,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "edit_images: provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "edit_images: provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("edit_images: recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no edit-capable providers (skipped: {skipped})"
            )
        raise last_exc

    async def generate_speech(
        self,
        *,
        input_text: str,
        voice: str | None = None,
        response_format: str = "mp3",
        speed: float = 1.0,
        extra: dict[str, Any] | None = None,
        chain_name: str | None = None,
    ) -> tuple[AudioSpeechResult, str, str]:
        """Run text-to-speech fallback chain. Skips providers without
        `supports_speech=True`. Retry semantics match `transcribe()` /
        `generate_images()`."""
        resolved_name, providers = self.resolve_chain(chain_name)
        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            if not getattr(provider, "supports_speech", False):
                skipped.append(provider.name)
                log.info("generate_speech: skipping %s (no speech support)", provider.name)
                continue

            rpd = getattr(provider, "rpd", None)
            rpm = getattr(provider, "rpm", None)
            if self._rate_tracker is not None and not self._rate_tracker.is_available(
                provider.name, rpd=rpd, rpm=rpm
            ):
                skipped.append(provider.name)
                requests_total.labels(
                    provider=provider.name, outcome=ErrorCategory.RATE_CAPPED.value
                ).inc()
                log.info("generate_speech: skipping %s (rate_capped)", provider.name)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await provider.generate_speech(
                    input_text=input_text,
                    voice=voice,
                    response_format=response_format,
                    speed=speed,
                    extra=extra,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "generate_speech: provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "generate_speech: provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("generate_speech: recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no speech-capable providers (skipped: {skipped})"
            )
        raise last_exc

    async def generate_sfx(
        self,
        *,
        prompt: str,
        duration_s: float | None = None,
        extra: dict[str, Any] | None = None,
        chain_name: str | None = None,
    ) -> tuple[AudioGenerationResult, str, str]:
        """Run text-to-audio (SFX/ambient) fallback chain. Skips providers without
        `supports_sfx=True`. Retry semantics match `generate_speech()`."""
        resolved_name, providers = self.resolve_chain(chain_name)
        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            if not getattr(provider, "supports_sfx", False):
                skipped.append(provider.name)
                log.info("generate_sfx: skipping %s (no sfx support)", provider.name)
                continue

            rpd = getattr(provider, "rpd", None)
            rpm = getattr(provider, "rpm", None)
            if self._rate_tracker is not None and not self._rate_tracker.is_available(
                provider.name, rpd=rpd, rpm=rpm
            ):
                skipped.append(provider.name)
                requests_total.labels(
                    provider=provider.name, outcome=ErrorCategory.RATE_CAPPED.value
                ).inc()
                log.info("generate_sfx: skipping %s (rate_capped)", provider.name)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await provider.generate_sfx(
                    prompt=prompt,
                    duration_s=duration_s,
                    extra=extra,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "generate_sfx: provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "generate_sfx: provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("generate_sfx: recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no sfx-capable providers (skipped: {skipped})"
            )
        raise last_exc

    async def embed(
        self,
        *,
        input_texts: list[str],
        extra: dict[str, Any] | None = None,
        chain_name: str | None = None,
    ) -> tuple[EmbeddingResult, str, str]:
        """Run embeddings fallback chain. Skips providers without
        `supports_embed=True`. Retry semantics match `generate_speech()`."""
        resolved_name, providers = self.resolve_chain(chain_name)
        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            if not getattr(provider, "supports_embed", False):
                skipped.append(provider.name)
                log.info("embed: skipping %s (no embed support)", provider.name)
                continue

            rpd = getattr(provider, "rpd", None)
            rpm = getattr(provider, "rpm", None)
            if self._rate_tracker is not None and not self._rate_tracker.is_available(
                provider.name, rpd=rpd, rpm=rpm
            ):
                skipped.append(provider.name)
                requests_total.labels(
                    provider=provider.name, outcome=ErrorCategory.RATE_CAPPED.value
                ).inc()
                log.info("embed: skipping %s (rate_capped)", provider.name)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await provider.embed(
                    input_texts=input_texts,
                    extra=extra,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "embed: provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "embed: provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("embed: recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no embed-capable providers (skipped: {skipped})"
            )
        raise last_exc

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = False,
        extra: dict[str, Any] | None = None,
        chain_name: str | None = None,
    ) -> tuple[RerankResult, str, str]:
        """Run rerank fallback chain. Скипает провайдеров без `supports_rerank=True`.
        Семантика retry/skip — как embed (rate-cap + non-retryable raise)."""
        resolved_name, providers = self.resolve_chain(chain_name)
        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            if not getattr(provider, "supports_rerank", False):
                skipped.append(provider.name)
                log.info("rerank: skipping %s (no rerank support)", provider.name)
                continue

            rpd = getattr(provider, "rpd", None)
            rpm = getattr(provider, "rpm", None)
            if self._rate_tracker is not None and not self._rate_tracker.is_available(
                provider.name, rpd=rpd, rpm=rpm
            ):
                skipped.append(provider.name)
                requests_total.labels(
                    provider=provider.name, outcome=ErrorCategory.RATE_CAPPED.value
                ).inc()
                log.info("rerank: skipping %s (rate_capped)", provider.name)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await provider.rerank(
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    return_documents=return_documents,
                    extra=extra,
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "rerank: provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "rerank: provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("rerank: recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no rerank-capable providers (skipped: {skipped})"
            )
        raise last_exc

    async def translate(
        self,
        *,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        chain_name: str | None = None,
    ) -> tuple[TranslationResult, str, str]:
        """Run translation fallback chain. Dedicated MT-провайдеры (LibreTranslate,
        MyMemory) используют свой translate(); LLM-провайдеры оборачиваются через
        chat() со standard translate-prompt.

        source_lang='auto' разрешает автоопределение. Провайдеры без auto-detect
        (MyMemory) скипаются на этом значении → следующий в цепочке.
        """
        resolved_name, providers = self.resolve_chain(chain_name)

        if resolved_name == "translate_adaptive":
            ru_involved = source_lang.lower() == "ru" or target_lang.lower() == "ru"
            head = "yandex:translate" if ru_involved else "cohere:aya-expanse-32b"
            providers = sorted(providers, key=lambda p: 0 if p.name == head else 1)

        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            # Rate-cap/tools-preflight — tools=None, min_context=None (translation
            # однократный вызов, контекст обычно не проблема).
            pre = self._preflight_skip(provider, tools=None, min_context=None)
            if pre is not None:
                skipped.append(provider.name)
                requests_total.labels(provider=provider.name, outcome=pre.value).inc()
                log.info("translate: skipping %s: %s", provider.name, pre.value)
                continue

            is_dedicated = getattr(provider, "supports_translation", False)
            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                if is_dedicated:
                    result = await provider.translate(
                        text=text,
                        target_lang=target_lang,
                        source_lang=source_lang,
                    )
                else:
                    result = await _translate_via_chat(
                        provider,
                        text=text,
                        target_lang=target_lang,
                        source_lang=source_lang,
                    )
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "translate: provider %s non-retryable (%s): %s",
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "translate: provider %s failed (%s), trying next",
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("translate: recovered via %s", provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no translation-capable providers (skipped: {skipped})"
            )
        raise last_exc

    async def moderate_text(
        self,
        *,
        input_texts: list[str],
        chain_name: str | None = None,
    ) -> tuple[ModerationResult, str, str]:
        """Run text-moderation fallback. Скипает провайдеров без
        `supports_moderation_text=True`. Семантика retry/skip — как embed."""
        return await self._moderate(
            chain_name=chain_name,
            require_attr="supports_moderation_text",
            label="moderate_text",
            call=lambda provider: provider.moderate_text(input_texts=input_texts),
        )

    async def moderate_image(
        self,
        *,
        images: list[str],
        context_text: str | None = None,
        chain_name: str | None = None,
    ) -> tuple[ModerationResult, str, str]:
        """Run image-moderation fallback. Скипает провайдеров без
        `supports_moderation_image=True` (Mistral text-only / Llama Guard 3-8B
        text-only / Prompt Guard text-only)."""
        return await self._moderate(
            chain_name=chain_name,
            require_attr="supports_moderation_image",
            label="moderate_image",
            call=lambda provider: provider.moderate_image(
                images=images, context_text=context_text
            ),
        )

    async def _moderate(
        self,
        *,
        chain_name: str | None,
        require_attr: str,
        label: str,
        call,
    ) -> tuple[ModerationResult, str, str]:
        resolved_name, providers = self.resolve_chain(chain_name)
        last_exc: BaseException | None = None
        skipped: list[str] = []

        for provider in providers:
            if not getattr(provider, require_attr, False):
                skipped.append(provider.name)
                log.info("%s: skipping %s (no %s)", label, provider.name, require_attr)
                continue

            pre = self._preflight_skip(provider, tools=None, min_context=None)
            if pre is not None:
                skipped.append(provider.name)
                requests_total.labels(provider=provider.name, outcome=pre.value).inc()
                log.info("%s: skipping %s: %s", label, provider.name, pre.value)
                continue

            if self._rate_tracker is not None:
                self._rate_tracker.record(provider.name)
            started = time.monotonic()
            try:
                result = await call(provider)
            except NotImplementedError as exc:
                # Провайдер заявил capability но реально модель не подходит — мягко скипаем.
                skipped.append(provider.name)
                log.warning("%s: %s NotImplementedError: %s", label, provider.name, exc)
                continue
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(provider, exc, elapsed)
                last_exc = exc
                if not category.retryable:
                    log.error(
                        "%s: provider %s non-retryable (%s): %s",
                        label,
                        provider.name,
                        category.value,
                        exc,
                    )
                    raise
                log.warning(
                    "%s: provider %s failed (%s), trying next",
                    label,
                    provider.name,
                    category.value,
                )
                continue

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=provider.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            if last_exc is not None:
                log.info("%s: recovered via %s", label, provider.name)
            return result, provider.name, resolved_name

        if last_exc is None:
            raise RuntimeError(
                f"chain {resolved_name}: no {require_attr} providers (skipped: {skipped})"
            )
        raise last_exc

    async def chat_moa(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_extras: dict[str, Any] | None = None,
        moa_chain: str = "moa",
        aggregator_chain: str = "reasoning_quality",
        proposer_timeout_s: float = 90.0,
        max_proposer_chars: int = 6000,
    ) -> tuple[ProviderCallResult, str, str, list[dict[str, Any]]]:
        """Mixture-of-Agents: fan out to all providers in `moa_chain` in parallel,
        then synthesize via `aggregator_chain` (which uses router.chat() with fallback).

        Returns (final_result, aggregator_provider_name, resolved_moa_name, proposals).
        Each proposal is a dict with: provider, text, prompt_tokens, completion_tokens,
        latency_ms, error (None on success).
        """
        resolved_moa, proposers = self.resolve_chain(moa_chain)
        if resolved_moa != moa_chain:
            raise RuntimeError(
                f"moa chain {moa_chain!r} not found (resolved to {resolved_moa!r})"
            )

        eligible: list[Provider] = []
        proposals: list[dict[str, Any]] = []
        for p in proposers:
            pre = self._preflight_skip(p, tools=None, min_context=None)
            if pre is not None:
                requests_total.labels(provider=p.name, outcome=pre.value).inc()
                proposals.append({
                    "provider": p.name,
                    "text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": 0,
                    "error": pre.value,
                })
                continue
            eligible.append(p)

        if not eligible:
            raise RuntimeError(
                f"moa chain {moa_chain}: no proposers eligible (all skipped pre-flight)"
            )

        async def _call(p: Provider) -> dict[str, Any]:
            if self._rate_tracker is not None:
                self._rate_tracker.record(p.name)
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    p.chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=None,
                        tool_choice=None,
                        request_extras=request_extras,
                    ),
                    timeout=proposer_timeout_s,
                )
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - started
                requests_total.labels(
                    provider=p.name, outcome=ErrorCategory.TIMEOUT.value
                ).inc()
                return {
                    "provider": p.name,
                    "text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": int(elapsed * 1000),
                    "error": "timeout",
                }
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(p, exc, elapsed)
                return {
                    "provider": p.name,
                    "text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": int(elapsed * 1000),
                    "error": category.value,
                }

            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=p.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=p.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            return {
                "provider": p.name,
                "text": result.text or "",
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms": int(elapsed * 1000),
                "error": None,
            }

        fan_out = await asyncio.gather(*(_call(p) for p in eligible))
        proposals.extend(fan_out)

        successes = [pr for pr in proposals if pr["error"] is None and pr["text"]]
        if not successes:
            raise RuntimeError(
                f"moa chain {moa_chain}: all {len(proposals)} proposers failed"
            )

        # Fewer than 2 successes — no synthesis needed, return the lone proposer as-is.
        if len(successes) < 2:
            sole = successes[0]
            result = ProviderCallResult(
                text=sole["text"],
                prompt_tokens=sole["prompt_tokens"],
                completion_tokens=sole["completion_tokens"],
                finish_reason="stop",
            )
            return result, sole["provider"], resolved_moa, proposals

        user_query = _extract_user_query(messages)

        # Per Together AI MoA paper: исключаем собственный ответ aggregator-а из
        # synthesis prompt, чтобы избежать self-bias (когда модель одновременно
        # proposer и aggregator — а пересечение moa/reasoning_quality неизбежно).
        # Перебираем aggregator-цепочку вручную: на каждом кандидате пересобираем
        # synthesis без его собственного proposal и вызываем его как single-provider
        # chain (router.chat с chain_name=p.name даёт ровно один провайдер, без
        # нежелательного fallback на default_chain).
        _, agg_providers = self.resolve_chain(aggregator_chain)
        last_exc: BaseException | None = None
        for agg_p in agg_providers:
            pre = self._preflight_skip(agg_p, tools=None, min_context=None)
            if pre is not None:
                continue
            filtered = [s for s in successes if s["provider"] != agg_p.name]
            if not filtered:
                continue
            synthesis_messages = _build_synthesis_messages(
                user_query=user_query,
                successes=filtered,
                max_chars=max_proposer_chars,
            )
            # Aggregator синтезирует 20+ мнений в структурированный ответ — ему нужно
            # больше токенов чем одному proposer-у. Plus Gemini тратит часть на thinking.
            # Пользовательский max_tokens остаётся для proposer-ов (им хватает краткого ответа).
            agg_max_tokens = max(max_tokens or 0, 2500)
            try:
                agg_result, agg_name, _ = await self.chat(
                    messages=synthesis_messages,
                    temperature=temperature,
                    max_tokens=agg_max_tokens,
                    tools=None,
                    tool_choice=None,
                    request_extras=request_extras,
                    chain_name=agg_p.name,
                )
                return agg_result, agg_name, resolved_moa, proposals
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"no aggregator in {aggregator_chain} could synthesize "
            f"(all {len(agg_providers)} candidates failed pre-flight)"
        )

    async def chat_deep_search(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_extras: dict[str, Any] | None = None,
        planner_chain: str = "reasoning_quality",
        searcher_chain: str = "web",
        synthesizer_chain: str = "reasoning_quality",
        critic_chain: str = "reasoning_quality",
        max_subquestions: int = 4,
        max_critic_rounds: int = 1,
        jina_enabled: bool = True,
    ) -> tuple[ProviderCallResult, str, dict[str, Any]]:
        """Deep Search orchestrator (plan → search → synthesize → critique → iterate).

        Reads the user question from last user message. All pipeline logic lives in
        deep_search.run_deep_search to keep router thin.
        """
        from neurogate.deep_search import run_deep_search

        user_query = _extract_user_query(messages)
        if not user_query:
            raise RuntimeError("deep_search: no user message found")

        return await run_deep_search(
            self,
            user_query,
            planner_chain=planner_chain,
            searcher_chain=searcher_chain,
            synthesizer_chain=synthesizer_chain,
            critic_chain=critic_chain,
            max_subquestions=max_subquestions,
            max_critic_rounds=max_critic_rounds,
            temperature=temperature,
            max_tokens=max_tokens,
            request_extras=request_extras,
            jina_enabled=jina_enabled,
        )

    async def chat_debate(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_extras: dict[str, Any] | None = None,
        debate_chain: str = "debate",
        aggregator_chain: str = "reasoning_quality",
        agents: int = 3,
        rounds: int = 2,
        agent_timeout_s: float = 120.0,
        max_agent_chars: int = 4000,
        event_emit: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> tuple[ProviderCallResult, str, str, list[list[dict[str, Any]]], list[str]]:
        """Multi-Agent Debate (Du et al. 2023, Liang et al. 2023): N разных моделей
        отвечают независимо, затем R-1 раундов inter-agent revision (каждый агент
        видит ответы остальных и переписывает свой). Финальный синтез — aggregator.

        Returns (final_result, aggregator_provider_name, resolved_debate_name,
        transcript, agent_names). transcript[round_idx][agent_idx] = dict с
        {round, agent_index, provider, text, prompt_tokens, completion_tokens,
        latency_ms, error}. agent_names — имена выбранных N агентов.

        Отличие от MoA: MoA — это ОДИН раунд + aggregator. Debate — это R раундов
        с обменом мнениями между агентами; ошибки/галлюцинации одного агента
        корректируются другими через критику в следующих раундах.
        """
        if agents < 2:
            raise ValueError(f"debate: agents must be ≥2, got {agents}")
        if rounds < 1:
            raise ValueError(f"debate: rounds must be ≥1, got {rounds}")

        resolved_debate, candidates = self.resolve_chain(debate_chain)
        if resolved_debate != debate_chain:
            raise RuntimeError(
                f"debate chain {debate_chain!r} not found (resolved to {resolved_debate!r})"
            )

        selected: list[Provider] = []
        for p in candidates:
            if len(selected) >= agents:
                break
            pre = self._preflight_skip(p, tools=None, min_context=None)
            if pre is not None:
                requests_total.labels(provider=p.name, outcome=pre.value).inc()
                log.info("debate: skipping %s pre-flight: %s", p.name, pre.value)
                continue
            selected.append(p)

        if len(selected) < 2:
            raise RuntimeError(
                f"debate chain {debate_chain}: need ≥2 eligible agents, got {len(selected)}"
            )

        agent_names = [p.name for p in selected]
        user_query = _extract_user_query(messages)
        log.info(
            "debate: chain=%s agents=%d rounds=%d aggregator=%s names=%s",
            resolved_debate, len(selected), rounds, aggregator_chain, agent_names,
        )

        async def _emit(ev: dict[str, Any]) -> None:
            if event_emit is None:
                return
            try:
                await event_emit(ev)
            except Exception:
                log.exception("debate: event_emit raised, dropping event")

        await _emit({
            "type": "meta",
            "agents": agent_names,
            "rounds": rounds,
            "aggregator_chain": aggregator_chain,
            "debate_chain": resolved_debate,
        })

        async def _call_agent(
            p: Provider,
            agent_idx: int,
            round_idx: int,
            agent_messages: list[dict[str, Any]],
        ) -> dict[str, Any]:
            if self._rate_tracker is not None:
                self._rate_tracker.record(p.name)
            # Уважаем provider.timeout как пол: slow-модели (qwen3.5-397b: 240s)
            # не должны false-падать по короткому debate-флору.
            provider_timeout = getattr(p, "timeout", None) or 0.0
            effective_timeout = max(agent_timeout_s, float(provider_timeout))
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    p.chat(
                        messages=agent_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=None,
                        tool_choice=None,
                        request_extras=request_extras,
                    ),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - started
                requests_total.labels(
                    provider=p.name, outcome=ErrorCategory.TIMEOUT.value
                ).inc()
                return {
                    "round": round_idx,
                    "agent_index": agent_idx,
                    "provider": p.name,
                    "text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": int(elapsed * 1000),
                    "error": "timeout",
                }
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(p, exc, elapsed)
                return {
                    "round": round_idx,
                    "agent_index": agent_idx,
                    "provider": p.name,
                    "text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": int(elapsed * 1000),
                    "error": category.value,
                }
            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=p.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=p.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            return {
                "round": round_idx,
                "agent_index": agent_idx,
                "provider": p.name,
                "text": result.text or "",
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms": int(elapsed * 1000),
                "error": None,
            }

        transcript: list[list[dict[str, Any]]] = []

        async def _run_round(
            r: int,
            coros: list[Awaitable[dict[str, Any]]],
        ) -> list[dict[str, Any]]:
            await _emit({"type": "round_start", "round": r})
            for idx, p in enumerate(selected):
                await _emit({
                    "type": "agent_start",
                    "round": r,
                    "agent_index": idx,
                    "provider": p.name,
                })
            tasks = [asyncio.create_task(c) for c in coros]
            results: list[dict[str, Any] | None] = [None] * len(tasks)
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                results[rec["agent_index"]] = rec
                await _emit({"type": "agent_done", **rec})
            await _emit({"type": "round_done", "round": r})
            return [r for r in results if r is not None]

        # Round 0: each agent answers independently with the original messages.
        round0 = await _run_round(
            0, [_call_agent(p, i, 0, messages) for i, p in enumerate(selected)]
        )
        transcript.append(round0)
        log.info(
            "debate: round=0 ok=%d/%d",
            sum(1 for a in round0 if a["error"] is None and a["text"]),
            len(round0),
        )

        # Rounds 1..rounds-1: each agent sees the *previous* round's answers from
        # the other agents and is asked to critique/revise its own answer.
        for r in range(1, rounds):
            prev_round = transcript[-1]

            def _revise_factory(p: Provider, idx: int, r: int = r,
                                prev_round: list[dict[str, Any]] = prev_round):
                async def _revise() -> dict[str, Any]:
                    others = [
                        a for j, a in enumerate(prev_round)
                        if j != idx and a["error"] is None and a["text"]
                    ]
                    # If everyone else failed, just re-emit our previous answer
                    # rather than asking a degenerate "revise based on nothing" prompt.
                    # Preserve prev_self["error"] verbatim — a successful prev_self
                    # must stay error=None so the survivor isn't false-failed and
                    # filtered out of the final-round successes.
                    if not others:
                        prev_self = prev_round[idx]
                        return {
                            "round": r,
                            "agent_index": idx,
                            "provider": p.name,
                            "text": prev_self["text"],
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "latency_ms": 0,
                            "error": prev_self["error"],
                        }
                    # Pre-flight: cooldown set in round 0 (e.g. 429 → 60s) must
                    # short-circuit round 1, otherwise we burn another 429 on the
                    # same provider. Selection at round 0 doesn't cover this since
                    # cooldown is set *after* the first failed call.
                    pre = self._preflight_skip(p, tools=None, min_context=None)
                    if pre is not None:
                        requests_total.labels(provider=p.name, outcome=pre.value).inc()
                        return {
                            "round": r,
                            "agent_index": idx,
                            "provider": p.name,
                            "text": "",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "latency_ms": 0,
                            "error": pre.value,
                        }
                    revise_messages = _build_debate_revision_messages(
                        user_query=user_query,
                        own_prev=prev_round[idx],
                        others=others,
                        max_chars=max_agent_chars,
                    )
                    return await _call_agent(p, idx, r, revise_messages)
                return _revise()

            round_r = await _run_round(
                r, [_revise_factory(p, i) for i, p in enumerate(selected)]
            )
            transcript.append(round_r)
            log.info(
                "debate: round=%d ok=%d/%d",
                r,
                sum(1 for a in round_r if a["error"] is None and a["text"]),
                len(round_r),
            )

        last_round = transcript[-1]
        successes = [a for a in last_round if a["error"] is None and a["text"]]
        if not successes:
            breakdown = ", ".join(
                f"{a['provider']}={a['error'] or 'empty'}" for a in last_round
            )
            raise RuntimeError(
                f"debate: all {len(last_round)} agents failed in final round "
                f"({breakdown})"
            )
        if len(successes) < 2:
            sole = successes[0]
            result = ProviderCallResult(
                text=sole["text"],
                prompt_tokens=sole["prompt_tokens"],
                completion_tokens=sole["completion_tokens"],
                finish_reason="stop",
            )
            await _emit({
                "type": "aggregator_done",
                "provider": sole["provider"],
                "text": sole["text"],
                "prompt_tokens": sole["prompt_tokens"],
                "completion_tokens": sole["completion_tokens"],
                "note": "single-survivor",
            })
            return result, sole["provider"], resolved_debate, transcript, agent_names

        synthesis_messages = _build_debate_synthesis_messages(
            user_query=user_query,
            final_answers=successes,
            rounds=rounds,
            max_chars=max_agent_chars,
        )
        agg_max_tokens = max(max_tokens or 0, 2500)

        # Aggregator-цепочка с self-bias защитой как в MoA: исключаем собственный
        # ответ aggregator-а (если он же был среди агентов) из synthesis prompt.
        _, agg_providers = self.resolve_chain(aggregator_chain)
        last_exc: BaseException | None = None
        for agg_p in agg_providers:
            pre = self._preflight_skip(agg_p, tools=None, min_context=None)
            if pre is not None:
                continue
            filtered = [s for s in successes if s["provider"] != agg_p.name]
            if not filtered:
                continue
            messages_for_this_agg = (
                synthesis_messages if len(filtered) == len(successes)
                else _build_debate_synthesis_messages(
                    user_query=user_query,
                    final_answers=filtered,
                    rounds=rounds,
                    max_chars=max_agent_chars,
                )
            )
            await _emit({"type": "aggregator_start", "provider": agg_p.name})
            try:
                agg_result, agg_name, _ = await self.chat(
                    messages=messages_for_this_agg,
                    temperature=temperature,
                    max_tokens=agg_max_tokens,
                    tools=None,
                    tool_choice=None,
                    request_extras=request_extras,
                    chain_name=agg_p.name,
                )
                await _emit({
                    "type": "aggregator_done",
                    "provider": agg_name,
                    "text": agg_result.text or "",
                    "prompt_tokens": agg_result.prompt_tokens,
                    "completion_tokens": agg_result.completion_tokens,
                })
                return agg_result, agg_name, resolved_debate, transcript, agent_names
            except Exception as exc:
                last_exc = exc
                await _emit({
                    "type": "aggregator_error",
                    "provider": agg_p.name,
                    "message": repr(exc),
                })
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"debate: no aggregator in {aggregator_chain} could synthesize "
            f"(all {len(agg_providers)} candidates failed pre-flight)"
        )

    async def chat_sc(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        request_extras: dict[str, Any] | None = None,
        sc_chain: str = "sc",
        aggregator_chain: str = "reasoning_quality",
        samples: int = 5,
        sample_timeout_s: float = 90.0,
        max_sample_chars: int = 6000,
    ) -> tuple[ProviderCallResult, str, str, list[dict[str, Any]], str]:
        """Self-Consistency (Wang et al., 2022): N сэмплов от ОДНОЙ модели + синтез.

        Returns (final_result, aggregator_provider_name, resolved_sc_chain,
        samples_list, base_provider_name). samples_list — dicts с
        {sample_index, provider, text, prompt_tokens, completion_tokens,
        latency_ms, error, temperature}.

        В отличие от MoA, self-bias НЕ проблема: aggregator консолидирует
        собственные сэмплы — это и есть смысл SC. Temperature для сэмплов
        повышается до 1.0 (или user-value, если задан > 0.7) чтобы получить
        разнообразие рассуждений.
        """
        if samples < 2:
            raise ValueError(f"sc: samples must be ≥2, got {samples}")

        resolved_sc, base_candidates = self.resolve_chain(sc_chain)
        if resolved_sc != sc_chain:
            raise RuntimeError(
                f"sc chain {sc_chain!r} not found (resolved to {resolved_sc!r})"
            )

        # Находим первого eligible base-провайдера. Fallback на следующих если первый
        # скипнут по pre-flight (rate-cap/tools/context).
        base: Provider | None = None
        for p in base_candidates:
            pre = self._preflight_skip(p, tools=None, min_context=None)
            if pre is None:
                base = p
                break
            requests_total.labels(provider=p.name, outcome=pre.value).inc()
            log.info("sc: skipping base %s: %s", p.name, pre.value)
        if base is None:
            raise RuntimeError(
                f"sc chain {sc_chain}: no base provider eligible pre-flight"
            )

        # Разнообразие — ключ к SC. Дефолтим на 1.0, но уважаем пользователя если он
        # явно задал более высокую температуру.
        sample_temperature = max(temperature or 0.0, 1.0) if temperature is None else max(temperature, 0.7)

        async def _one_sample(idx: int) -> dict[str, Any]:
            if self._rate_tracker is not None:
                self._rate_tracker.record(base.name)
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    base.chat(
                        messages=messages,
                        temperature=sample_temperature,
                        max_tokens=max_tokens,
                        tools=None,
                        tool_choice=None,
                        request_extras=request_extras,
                    ),
                    timeout=sample_timeout_s,
                )
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - started
                requests_total.labels(
                    provider=base.name, outcome=ErrorCategory.TIMEOUT.value
                ).inc()
                return {
                    "sample_index": idx,
                    "provider": base.name,
                    "text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": int(elapsed * 1000),
                    "error": "timeout",
                    "temperature": sample_temperature,
                }
            except Exception as exc:
                elapsed = time.monotonic() - started
                category = self._handle_failure(base, exc, elapsed)
                return {
                    "sample_index": idx,
                    "provider": base.name,
                    "text": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": int(elapsed * 1000),
                    "error": category.value,
                    "temperature": sample_temperature,
                }
            elapsed = time.monotonic() - started
            requests_total.labels(
                provider=base.name, outcome=ErrorCategory.SUCCESS.value
            ).inc()
            request_duration_seconds.labels(
                provider=base.name, outcome=ErrorCategory.SUCCESS.value
            ).observe(elapsed)
            return {
                "sample_index": idx,
                "provider": base.name,
                "text": result.text or "",
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms": int(elapsed * 1000),
                "error": None,
                "temperature": sample_temperature,
            }

        samples_list = await asyncio.gather(*(_one_sample(i) for i in range(samples)))

        successes = [s for s in samples_list if s["error"] is None and s["text"]]
        if not successes:
            raise RuntimeError(
                f"sc: all {samples} samples of {base.name} failed"
            )
        if len(successes) < 2:
            sole = successes[0]
            result = ProviderCallResult(
                text=sole["text"],
                prompt_tokens=sole["prompt_tokens"],
                completion_tokens=sole["completion_tokens"],
                finish_reason="stop",
            )
            return result, sole["provider"], resolved_sc, samples_list, base.name

        user_query = _extract_user_query(messages)
        synthesis_messages = _build_sc_synthesis_messages(
            user_query=user_query,
            samples=successes,
            base_name=base.name,
            max_chars=max_sample_chars,
        )
        agg_max_tokens = max(max_tokens or 0, 2500)

        # SC aggregator не исключает "свои" samples — они И ЕСТЬ собственные
        # сэмплы базовой модели, и консолидация их — смысл SC. Но всё равно
        # перебираем aggregator-цепочку с fallback на случай отказа первого.
        _, agg_providers = self.resolve_chain(aggregator_chain)
        last_exc: BaseException | None = None
        for agg_p in agg_providers:
            pre = self._preflight_skip(agg_p, tools=None, min_context=None)
            if pre is not None:
                continue
            try:
                agg_result, agg_name, _ = await self.chat(
                    messages=synthesis_messages,
                    temperature=temperature,
                    max_tokens=agg_max_tokens,
                    tools=None,
                    tool_choice=None,
                    request_extras=request_extras,
                    chain_name=agg_p.name,
                )
                return agg_result, agg_name, resolved_sc, samples_list, base.name
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"sc: no aggregator in {aggregator_chain} could consolidate samples"
        )


# ISO-639-1 коды → читаемое имя для translate-prompt. Добавляем только то что
# реально поддерживают MT-провайдеры (LibreTranslate ~30 langs, MyMemory 150+).
# Остальные коды улетают в prompt как есть — LLM справится.
_LANG_NAMES = {
    "en": "English", "ru": "Russian", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "tr": "Turkish",
    "pl": "Polish", "uk": "Ukrainian", "nl": "Dutch", "sv": "Swedish",
    "cs": "Czech", "he": "Hebrew", "hi": "Hindi", "vi": "Vietnamese",
    "id": "Indonesian", "th": "Thai", "el": "Greek", "fi": "Finnish",
    "da": "Danish", "no": "Norwegian", "ro": "Romanian", "hu": "Hungarian",
    "bg": "Bulgarian", "auto": "auto-detect source language",
}


async def _translate_via_chat(
    provider: Provider,
    *,
    text: str,
    target_lang: str,
    source_lang: str,
) -> TranslationResult:
    """Wrap LLM chat() into a translation call. Prompt инструктирует модель
    выдавать ТОЛЬКО перевод без пояснений/кавычек — иначе распарсить сложно."""
    tgt = _LANG_NAMES.get(target_lang, target_lang)
    if source_lang == "auto":
        sys_prompt = (
            f"You are a professional translator. Translate the user's text into {tgt}. "
            "Detect the source language automatically. Output ONLY the translated text — "
            "no explanations, no quotes, no source text, no language tags."
        )
    else:
        src = _LANG_NAMES.get(source_lang, source_lang)
        sys_prompt = (
            f"You are a professional translator. Translate the following {src} text into {tgt}. "
            "Output ONLY the translated text — no explanations, no quotes, no source text, no language tags."
        )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text},
    ]
    # temperature=0: перевод детерминирован, креативность вредит fidelity
    result = await provider.chat(
        messages=messages,
        temperature=0.0,
        max_tokens=max(256, len(text) * 3),  # грубая оценка: перевод редко длиннее x3
        tools=None,
        tool_choice=None,
        request_extras=None,
    )
    translated = (result.text or "").strip()
    if not translated:
        raise RuntimeError(f"{provider.name} empty translation")
    # Убираем возможные обёртки которые LLM любит добавлять (маркдаун-блоки, кавычки)
    translated = translated.strip('"\'`')
    if translated.startswith("```") and translated.endswith("```"):
        translated = translated[3:-3].strip()
    return TranslationResult(
        text=translated,
        target_lang=target_lang,
        source_lang=source_lang if source_lang != "auto" else None,
        provider_model=f"llm:{provider.name}",
        raw={"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens},
    )


def _extract_user_query(messages: list[dict[str, Any]]) -> str:
    """Pull the last user message as a plain string. Non-string content (vision
    parts) is best-effort stringified — moa mostly serves text-first prompts."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            parts.append(str(c.get("text", "")))
                return "\n".join(parts) or str(content)
            return str(content or "")
    return ""


def _build_synthesis_messages(
    *,
    user_query: str,
    successes: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    blocks: list[str] = []
    for i, pr in enumerate(successes, 1):
        text = pr["text"] or ""
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "\n…[обрезано]"
        blocks.append(f"[Модель {i}: {pr['provider']}]\n{text}")

    system = (
        "Ты — агрегатор ответов. Я задал один и тот же вопрос нескольким LLM. "
        "Синтезируй единый качественный ответ:\n"
        "1. Найди общее, что подтверждают несколько моделей — это скорее всего верно.\n"
        "2. При противоречиях выбери наиболее аргументированную версию.\n"
        "3. Дополни лучшими деталями из отдельных ответов.\n"
        "4. Игнорируй очевидные ошибки и галлюцинации одной модели.\n"
        "5. Ответь на языке исходного вопроса.\n"
        "Не упоминай, что ответ синтезирован из нескольких источников."
    )
    user = (
        f"Исходный вопрос:\n{user_query}\n\n"
        f"Ответы {len(successes)} моделей:\n\n"
        + "\n\n".join(blocks)
        + "\n\nДай финальный ответ."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_sc_synthesis_messages(
    *,
    user_query: str,
    samples: list[dict[str, Any]],
    base_name: str,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Synthesis prompt для Self-Consistency: N сэмплов одной модели + aggregator.

    Отличается от MoA-версии: все сэмплы от ОДНОЙ модели, поэтому консолидация —
    это поиск наиболее consistent линии рассуждения, а не выбор лучшего из разных.
    """
    blocks: list[str] = []
    for s in samples:
        text = s["text"] or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…[обрезано]"
        blocks.append(f"[Сэмпл {s['sample_index'] + 1}]\n{text}")

    system = (
        f"Ты — агрегатор ответов. Одна и та же модель ({base_name}) была опрошена "
        f"{len(samples)} раз с высокой temperature — получились разные рассуждения "
        "по одному вопросу. Твоя задача — консолидировать их в единый надёжный ответ:\n"
        "1. Найди консистентный ответ, который встречается в большинстве сэмплов — "
        "это скорее всего правильное рассуждение.\n"
        "2. Отбрось outliers — отдельные сэмплы, противоречащие большинству.\n"
        "3. Если сэмплы единодушны — просто изложи их общий ответ качественно.\n"
        "4. Если расходятся сильно — укажи обе версии, отдав приоритет более "
        "аргументированной.\n"
        "5. Ответь на языке исходного вопроса. Не упоминай, что ответ получен "
        "через self-consistency."
    )
    user = (
        f"Исходный вопрос:\n{user_query}\n\n"
        f"{len(samples)} сэмплов модели {base_name}:\n\n"
        + "\n\n".join(blocks)
        + "\n\nДай финальный ответ."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_debate_revision_messages(
    *,
    user_query: str,
    own_prev: dict[str, Any],
    others: list[dict[str, Any]],
    max_chars: int,
) -> list[dict[str, Any]]:
    """Round-N prompt: показываем агенту его собственный предыдущий ответ
    и ответы остальных, просим переосмыслить и выдать улучшенную версию."""
    own_text = own_prev.get("text") or ""
    if len(own_text) > max_chars:
        own_text = own_text[:max_chars] + "\n…[обрезано]"

    other_blocks: list[str] = []
    for i, a in enumerate(others, 1):
        text = a.get("text") or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…[обрезано]"
        other_blocks.append(f"[Коллега {i}: {a['provider']}]\n{text}")

    system = (
        "Ты — один из нескольких ИИ-агентов в дебатах. Каждый агент дал свой "
        "независимый ответ на вопрос. Сейчас ты видишь ответы коллег и свой "
        "предыдущий ответ. Твоя задача — критически их переосмыслить и выдать "
        "улучшенную версию своего ответа:\n"
        "1. Если у коллег есть аргументы сильнее твоих — учти их.\n"
        "2. Если ты заметил ошибки/галлюцинации в чужих ответах — игнорируй их и "
        "оставайся при своей точке зрения.\n"
        "3. Если все согласны — изложи общий ответ максимально качественно.\n"
        "4. Если есть разногласия — выбери наиболее обоснованную позицию и "
        "аргументируй её. Не поддавайся давлению большинства, если уверен в своей правоте.\n"
        "5. Ответь на языке вопроса. Не упоминай дебаты, коллег или раунды — "
        "только финальный ответ на исходный вопрос."
    )
    user = (
        f"Исходный вопрос:\n{user_query}\n\n"
        f"Твой предыдущий ответ:\n{own_text}\n\n"
        f"Ответы {len(others)} других агентов:\n\n"
        + "\n\n".join(other_blocks)
        + "\n\nДай свою улучшенную версию ответа."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_debate_synthesis_messages(
    *,
    user_query: str,
    final_answers: list[dict[str, Any]],
    rounds: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Финальный synthesis: агрегатор берёт ответы агентов из ПОСЛЕДНЕГО раунда
    дебатов (агенты уже видели чужие ответы и переписали свои с учётом критики)."""
    blocks: list[str] = []
    for i, a in enumerate(final_answers, 1):
        text = a.get("text") or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…[обрезано]"
        blocks.append(f"[Агент {i}: {a['provider']}]\n{text}")

    if rounds > 1:
        round_note = (
            f"(прошло {rounds} раунд(ов), агенты обменялись мнениями и переписали "
            "свои ответы с учётом критики коллег). Перед тобой — их финальные позиции."
        )
    else:
        round_note = (
            "(агенты отвечали независимо, без обмена мнениями). "
            "Перед тобой — их позиции."
        )
    system = (
        f"Ты — финальный арбитр в дебатах {len(final_answers)} ИИ-агентов "
        f"{round_note} "
        "Синтезируй единый ответ:\n"
        "1. Если агенты сошлись — изложи общий ответ качественно.\n"
        "2. При остаточных разногласиях — выбери наиболее аргументированную версию.\n"
        "3. Игнорируй очевидные ошибки и галлюцинации одного агента.\n"
        "4. Ответь на языке исходного вопроса.\n"
        "Не упоминай дебаты, агентов или раунды — только финальный ответ."
    )
    user = (
        f"Исходный вопрос:\n{user_query}\n\n"
        f"Финальные ответы {len(final_answers)} агентов:\n\n"
        + "\n\n".join(blocks)
        + "\n\nДай финальный ответ."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
