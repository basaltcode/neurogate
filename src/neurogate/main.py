from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from neurogate.anthropic_compat import (
    MessagesRequest,
    request_to_openai,
    result_to_anthropic,
    translate_stream,
)
from neurogate.audit import audit_loop, run_audit
from neurogate.health import health_loop, run_health_report
from neurogate.auto_route import classify_intent
from neurogate.config import (
    AdhocResolveError,
    ChainConfig,
    SkippedProvider,
    build_adhoc_provider,
    load_config,
    rewrite_chains_in_yaml,
)
from neurogate.metrics import registry
from neurogate.router import LLMRouter
from neurogate.stats import RateTracker
from neurogate.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    ModelInfo,
    ModelList,
)

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# Который клиент шлёт запрос — устанавливается middleware из заголовка
# `X-Client-Name`, читается _log_call. Информационная метка (вариант B плана #15):
# клиент сам себя называет, нельзя отозвать «по одному боту» — но даёт разрез
# «кто жрёт квоту» без миграции токенов.
_current_client: ContextVar[str | None] = ContextVar("neurogate_client", default=None)

_CLIENT_NAME_RE = re.compile(r"[^a-z0-9_./\-]")


def _sanitize_client_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = raw.strip().lower()[:64]
    name = _CLIENT_NAME_RE.sub("", name)
    return name or None

# Fields we handle explicitly on /v1/chat/completions — everything else at the
# top level is treated as a provider pass-through (prompt_cache_key, response_format,
# seed, reasoning_effort, etc.).
_KNOWN_REQUEST_FIELDS = frozenset({
    "model", "messages", "temperature", "top_p",
    "max_tokens", "max_completion_tokens", "stream",
    "stop", "presence_penalty", "frequency_penalty", "n",
    "user", "tools", "tool_choice", "parallel_tool_calls",
})

# Cross-provider reasoning knob. Mapped per-provider inside each Provider.
# "none" disables thinking where possible (Gemini budget=0, Z.AI disabled).
# "low"/"medium"/"high" — OpenAI-standard levels; forwarded to OpenAI-compat
# reasoning models, converted to thinking_budget for Gemini, enables Z.AI.
_VALID_REASONING_EFFORT = frozenset({"none", "low", "medium", "high"})


def _collect_request_extras(req: ChatCompletionRequest) -> dict | None:
    dumped = req.model_dump(exclude_none=True)
    extras = {k: v for k, v in dumped.items() if k not in _KNOWN_REQUEST_FIELDS}
    # claude-code-router translates Claude API thinking → OpenAI `reasoning`.
    # We canonicalize to `reasoning_effort` so downstream provider adapters
    # have a single knob to read (and so we don't leak the raw `reasoning`
    # field to OpenAI-compat providers that 400 on unknown params).
    raw_reasoning = extras.pop("reasoning", None)
    if raw_reasoning is not None and "reasoning_effort" not in extras:
        mapped = _reasoning_to_effort(raw_reasoning)
        if mapped is not None:
            extras["reasoning_effort"] = mapped
    re_effort = extras.get("reasoning_effort")
    if re_effort is not None and re_effort not in _VALID_REASONING_EFFORT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid reasoning_effort: {re_effort!r}. "
                f"Must be one of {sorted(_VALID_REASONING_EFFORT)}."
            ),
        )
    return extras or None


def _reasoning_to_effort(reasoning: Any) -> str | None:
    """Translate a `reasoning` payload into our canonical effort string.

    Recognizes:
    - Anthropic-style `{"type": "enabled", "budget_tokens": N}` (forwarded by
      claude-code-router from Claude API thinking blocks)
    - OpenAI-new-style `{"effort": "low|medium|high"}`
    - Bare strings (`"high"`) and bools (`true` → "medium")
    Anything else returns None and the field is dropped silently.
    """
    if isinstance(reasoning, str):
        return reasoning if reasoning in _VALID_REASONING_EFFORT else None
    if isinstance(reasoning, bool):
        return "medium" if reasoning else "none"
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort in _VALID_REASONING_EFFORT:
            return effort
        rtype = reasoning.get("type")
        if rtype == "disabled":
            return "none"
        budget = reasoning.get("budget_tokens")
        if rtype == "enabled" or isinstance(budget, int):
            if isinstance(budget, int):
                if budget <= 0:
                    return "none"
                if budget < 4096:
                    return "low"
                if budget < 16384:
                    return "medium"
                return "high"
            return "medium"
    return None


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _print_startup_banner(cfg: ChainConfig, host: str, port: int) -> None:
    """Print a human-readable summary on startup so a fresh-clone user immediately
    understands what's running and which providers are missing keys."""
    active = len(cfg.all_providers)
    skipped_count = len(cfg.skipped)
    chain_names = list(cfg.chains.keys())
    bar = "═" * 60
    print(bar)
    print(" llmgate v0.1.0")
    print(f" active providers: {active}")
    if skipped_count:
        print(f" skipped (missing key/field): {skipped_count}")
    print(f" chains: {', '.join(chain_names)}  (default: {cfg.default})")
    print(f" dashboard: http://{host}:{port}/dashboard")
    print(bar)
    if skipped_count:
        from neurogate.config import _missing_env_vars  # local import to avoid cycle

        grouped = _missing_env_vars(cfg.skipped)
        if grouped:
            print("To enable more providers, set in .env:")
            for env_var, providers in grouped[:10]:
                preview = ", ".join(providers[:3])
                if len(providers) > 3:
                    preview += f", +{len(providers) - 3} more"
                print(f"  {env_var:<24} → {preview}")
            if len(grouped) > 10:
                print(f"  ... and {len(grouped) - 10} more env vars (see startup log)")
            print("See docs/providers-setup.md for how to get each key.")
            print(bar)


def create_app() -> FastAPI:
    _configure_logging()
    config_path = Path(os.getenv("NEUROGATE_CONFIG", "config.yaml"))
    cfg = load_config(config_path)

    stats_path = Path(os.getenv("NEUROGATE_STATS_DB", "stats.db"))
    rate_tracker = RateTracker(stats_path)

    providers_by_name = {p.name: p for p in cfg.all_providers}
    router = LLMRouter(
        cfg.chains,
        default=cfg.default,
        rate_tracker=rate_tracker,
        providers_by_name=providers_by_name,
    )
    for name, chain in router.chains.items():
        log.info("chain %s: %s", name, " → ".join(chain))

    _print_startup_banner(
        cfg,
        host=os.getenv("NEUROGATE_HOST", "127.0.0.1"),
        port=int(os.getenv("NEUROGATE_PORT", "8765")),
    )

    api_token = os.getenv("NEUROGATE_API_TOKEN", "").strip()
    virtual_model = os.getenv("NEUROGATE_VIRTUAL_MODEL", "auto")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [
            asyncio.create_task(audit_loop(app)),
            asyncio.create_task(health_loop(app)),
        ]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title="neurogate", version="0.1.0", lifespan=lifespan)
    app.state.router = router
    app.state.rate_tracker = rate_tracker
    app.state.api_token = api_token
    app.state.virtual_model = virtual_model
    app.state.providers = cfg.all_providers

    @app.middleware("http")
    async def _capture_client(request: Request, call_next):
        # Не делаем reset — SSE-генераторы дотягивают _log_call после возврата
        # из middleware. Каждый request — свой asyncio Task, contextvar не
        # «протекает» между запросами.
        _current_client.set(_sanitize_client_name(request.headers.get("x-client-name")))
        return await call_next(request)

    def _authorize(authorization: str | None) -> None:
        if not api_token:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        supplied = authorization[len("Bearer ") :].strip()
        if supplied != api_token:
            raise HTTPException(status_code=401, detail="invalid bearer token")

    def auth_dep(authorization: str | None = Header(default=None)) -> None:
        _authorize(authorization)

    def _pick_chain(model_field: str | None) -> str:
        if not model_field:
            return router.default_chain
        if model_field in router.chains or model_field in providers_by_name:
            return model_field
        # Ad-hoc model strings (`openai:gpt-5-foo`, `groq/llama-99b`) — let the
        # router try to build a one-shot provider via build_adhoc_provider.
        # Any other unknown value silently falls back to default (старое поведение).
        if ":" in model_field or "/" in model_field:
            return model_field
        return router.default_chain

    def _log_call(
        *,
        endpoint: str,
        outcome: str,
        client_model: str | None = None,
        chain_requested: str | None = None,
        chain_resolved: str | None = None,
        provider: str | None = None,
        started: float | None = None,
        result: Any = None,
        stream: bool = False,
        error: BaseException | None = None,
        extra: dict | None = None,
    ) -> None:
        """Best-effort sink for /v1/calls dashboard. Never raises."""
        duration_ms = (
            int((time.monotonic() - started) * 1000) if started is not None else None
        )
        prompt_tokens = completion_tokens = total_tokens = cached_tokens = None
        if result is not None:
            pt = getattr(result, "prompt_tokens", None)
            ct = getattr(result, "completion_tokens", None)
            cht = getattr(result, "cached_tokens", None)
            prompt_tokens = pt if pt else None
            completion_tokens = ct if ct else None
            cached_tokens = cht if cht else None
            if pt is not None or ct is not None:
                total_tokens = (pt or 0) + (ct or 0) or None
        err_type = err_msg = None
        if error is not None:
            err_type = type(error).__name__
            err_msg = str(error)
        rate_tracker.record_call(
            endpoint=endpoint,
            outcome=outcome,
            client_model=client_model,
            chain_requested=chain_requested,
            chain_resolved=chain_resolved,
            provider=provider,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            stream=stream,
            error_type=err_type,
            error_msg=err_msg,
            extra=extra,
            client=_current_client.get(),
        )

    @app.get("/health")
    async def health() -> dict:
        # Unauth: plain liveness, no config leakage.
        return {"ok": True}

    @app.get("/v1/health")
    async def health_detail(_: None = Depends(auth_dep)) -> dict:
        return {
            "ok": True,
            "default_chain": router.default_chain,
            "chains": router.chains,
        }

    @app.get("/metrics")
    async def metrics(_: None = Depends(auth_dep)) -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/dashboard")
    async def dashboard() -> FileResponse:
        return FileResponse(_STATIC_DIR / "dashboard.html", media_type="text/html")

    @app.get("/static/made-in-basalt.svg")
    async def made_in_basalt() -> FileResponse:
        return FileResponse(_STATIC_DIR / "made-in-basalt.svg", media_type="image/svg+xml")

    @app.get("/v1/metrics.json")
    async def metrics_json(_: None = Depends(auth_dep)) -> dict:
        counters: dict[str, dict[str, float]] = {}
        latency_count: dict[str, float] = {}
        latency_sum: dict[str, float] = {}
        for family in registry.collect():
            if family.name == "neurogate_requests":
                for sample in family.samples:
                    if sample.name != "neurogate_requests_total":
                        continue
                    provider = sample.labels.get("provider", "?")
                    outcome = sample.labels.get("outcome", "?")
                    counters.setdefault(provider, {})[outcome] = sample.value
            elif family.name == "neurogate_request_duration_seconds":
                for sample in family.samples:
                    provider = sample.labels.get("provider", "?")
                    if sample.name.endswith("_count"):
                        latency_count[provider] = latency_count.get(provider, 0.0) + sample.value
                    elif sample.name.endswith("_sum"):
                        latency_sum[provider] = latency_sum.get(provider, 0.0) + sample.value
        latency: dict[str, dict[str, float]] = {}
        for provider, count in latency_count.items():
            total = latency_sum.get(provider, 0.0)
            latency[provider] = {
                "count": count,
                "sum_s": total,
                "avg_s": (total / count) if count > 0 else 0.0,
            }
        return {"counters": counters, "latency": latency}

    @app.get("/v1/stats")
    async def get_stats(_: None = Depends(auth_dep)) -> dict:
        usage = []
        for p in cfg.all_providers:
            caps = {"rpd": getattr(p, "rpd", None), "rpm": getattr(p, "rpm", None)}
            row = {
                "provider": p.name,
                "caps": caps,
                "quality": getattr(p, "quality", None),
                "latency_s": getattr(p, "latency_s", None),
                "ru": getattr(p, "ru", None),
                "reasoning": bool(getattr(p, "reasoning", False)),
                "quota_limited": bool(getattr(p, "quota_limited", False)),
                **rate_tracker.usage(p.name),
                "reset": rate_tracker.reset_info(
                    p.name, rpd=caps["rpd"], rpm=caps["rpm"]
                ),
            }
            usage.append(row)
        return {
            "default_chain": router.default_chain,
            "chains": router.chains,
            "usage": usage,
        }

    @app.get("/v1/chains/edit")
    async def get_chains_for_edit(_: None = Depends(auth_dep)) -> dict:
        """Snapshot for the dashboard's chain editor: current chain → providers
        mapping, the pool of all known provider names (for drag-source list),
        and the default chain. Read-only — mutation goes through PUT /v1/chains."""
        return {
            "chains": router.chains,
            "providers": [p.name for p in cfg.all_providers],
            "default": router.default_chain,
        }

    @app.put("/v1/chains")
    async def update_chains(request: Request, _: None = Depends(auth_dep)) -> dict:
        """Replace the `chains:` and `default_chain:` sections of config.yaml,
        then hot-reload the router so the next request hits the new shape.
        Provider pool stays as-is (rebuilding HTTP clients is unnecessary —
        we just rewire which chains reference which providers)."""
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        raw_chains = payload.get("chains")
        new_default = payload.get("default")
        if not isinstance(raw_chains, dict) or not raw_chains:
            raise HTTPException(
                status_code=400,
                detail="`chains` must be a non-empty object {name: [provider, ...]}",
            )
        if not isinstance(new_default, str) or not new_default:
            raise HTTPException(status_code=400, detail="`default` must be a non-empty string")

        # Validate: chain names sane, providers exist, no collision with provider names.
        provider_pool = {p.name: p for p in cfg.all_providers}
        normalized: dict[str, list[str]] = {}
        for name, plist in raw_chains.items():
            if not isinstance(name, str) or not name.strip():
                raise HTTPException(status_code=400, detail=f"invalid chain name: {name!r}")
            cname = name.strip()
            if cname in provider_pool:
                raise HTTPException(
                    status_code=400,
                    detail=f"chain name {cname!r} collides with an existing provider name",
                )
            if not isinstance(plist, list) or not plist:
                raise HTTPException(
                    status_code=400,
                    detail=f"chain {cname!r}: provider list must be non-empty",
                )
            seen: set[str] = set()
            ordered: list[str] = []
            for pname in plist:
                if not isinstance(pname, str):
                    raise HTTPException(
                        status_code=400,
                        detail=f"chain {cname!r}: provider names must be strings",
                    )
                if pname not in provider_pool:
                    # Ad-hoc kind:model_id — собираем on the fly из env-ключа.
                    # Регистрируем в provider_pool + cfg.all_providers, чтобы
                    # последующая resolve-стадия и round-trip через load_config
                    # увидели его как обычного члена пула.
                    try:
                        adhoc = build_adhoc_provider(pname)
                    except AdhocResolveError as exc:
                        raise HTTPException(
                            status_code=400,
                            detail=f"chain {cname!r}: {exc}",
                        ) from exc
                    provider_pool[adhoc.name] = adhoc
                    if not any(p.name == adhoc.name for p in cfg.all_providers):
                        cfg.all_providers.append(adhoc)
                if pname in seen:
                    continue  # silently dedupe — same provider twice in one chain is meaningless
                seen.add(pname)
                ordered.append(pname)
            normalized[cname] = ordered

        if new_default not in normalized:
            raise HTTPException(
                status_code=400,
                detail=f"default chain {new_default!r} not present in submitted chains",
            )

        # Persist to YAML first (round-trips through load_config — surfaces breakage
        # before we touch in-memory state). On success, rebuild chains using the
        # existing provider pool (no HTTP-client churn) and atomically swap.
        try:
            backup_path = rewrite_chains_in_yaml(config_path, normalized, new_default)
        except Exception as exc:
            log.exception("chains rewrite failed")
            raise HTTPException(
                status_code=500, detail=f"failed to write config: {exc!r}"
            ) from exc

        new_chains_resolved: dict[str, list] = {
            name: [provider_pool[p] for p in plist] for name, plist in normalized.items()
        }
        new_pbn = {p.name: p for p in cfg.all_providers}
        router.update_state(new_chains_resolved, new_default, new_pbn)
        # Mutate cfg in-place so closures that captured `cfg` see the new state.
        cfg.chains = new_chains_resolved
        cfg.default = new_default

        log.info("chains updated via /v1/chains: %d chains, default=%s, backup=%s",
                 len(normalized), new_default, backup_path)
        return {
            "ok": True,
            "chains": router.chains,
            "default": router.default_chain,
            "backup": str(backup_path),
        }

    @app.get("/v1/calls")
    async def list_calls(
        request: Request,
        _: None = Depends(auth_dep),
        limit: int = 200,
        since: int | None = None,
        endpoint: str | None = None,
        chain: str | None = None,
        provider: str | None = None,
        outcome: str | None = None,
        client: str | None = None,
    ) -> dict:
        """Recent inbound calls — what client asked, which chain resolved, which
        provider answered, latency, tokens. Powers the «Вызовы» dashboard tab."""
        calls = rate_tracker.list_calls(
            limit=limit,
            since=since,
            endpoint=endpoint,
            chain=chain,
            provider=provider,
            outcome=outcome,
            client=client,
        )
        # Summary windows: last hour and last 24h help the dashboard render
        # "X calls / Y errors / Z tokens" headline numbers without re-aggregating
        # in the browser over a potentially-truncated `calls` list.
        now = int(time.time())
        return {
            "calls": calls,
            "summary": {
                "last_1h": rate_tracker.calls_summary(since=now - 3600),
                "last_24h": rate_tracker.calls_summary(since=now - 86400),
            },
        }

    @app.get("/v1/audit")
    async def list_audit_reports(_: None = Depends(auth_dep)) -> dict:
        audits = rate_tracker.list_audits(limit=60)
        return {
            "audits": [{"date": d, "created_at": ts} for d, ts in audits],
        }

    @app.get("/v1/audit/{date}")
    async def get_audit_report(date: str, _: None = Depends(auth_dep)) -> dict:
        md = rate_tracker.get_audit(date)
        if md is None:
            raise HTTPException(status_code=404, detail="audit not found")
        return {"date": date, "markdown": md}

    @app.post("/v1/audit/run")
    async def trigger_audit_run(
        request: Request,
        chain: str | None = None,
        _: None = Depends(auth_dep),
    ) -> dict:
        # `chain` — имя цепочки (chat/code/...) или конкретного провайдера.
        # Дефолт задан в audit.py → _DEFAULT_AUDIT_CHAIN=code.
        kwargs: dict[str, Any] = {}
        if chain:
            kwargs["chain_name"] = chain
        try:
            date, provider = await run_audit(request.app, **kwargs)
        except Exception as exc:
            log.exception("manual audit run failed")
            raise HTTPException(status_code=502, detail=f"audit failed: {exc!r}") from exc
        return {"ok": True, "date": date, "provider": provider}

    @app.post("/v1/health/run")
    async def trigger_health_run(
        request: Request,
        _: None = Depends(auth_dep),
    ) -> dict:
        try:
            await run_health_report(request.app)
        except Exception as exc:
            log.exception("manual health run failed")
            raise HTTPException(status_code=502, detail=f"health failed: {exc!r}") from exc
        return {"ok": True}

    @app.get("/v1/models", response_model=ModelList)
    async def list_models(_: None = Depends(auth_dep)) -> ModelList:
        virtual_ids = [virtual_model, *router.chain_names()]
        seen: set[str] = set()
        models: list[ModelInfo] = []
        for mid in virtual_ids:
            if mid in seen:
                continue
            seen.add(mid)
            models.append(ModelInfo(id=mid))
        for p in cfg.all_providers:
            if p.name in seen:
                continue
            seen.add(p.name)
            models.append(ModelInfo(id=p.name))
        return ModelList(data=models)

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        _authorize(authorization)
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        max_tokens = req.max_tokens if req.max_tokens is not None else req.max_completion_tokens
        if req.model and req.model == request.app.state.virtual_model:
            chain_name, auto_reason = classify_intent(
                messages=messages,
                available_chains=set(router.chain_names()),
                default=router.default_chain,
            )
            log.info("auto-router: picked chain=%s reason=%s", chain_name, auto_reason)
        else:
            chain_name = _pick_chain(req.model)
        # Eagerly resolve chain so ad-hoc `kind:model_id` surfaces a clean 400
        # ("OPENAI_API_KEY not set") instead of a 502 from the chat fallback path.
        # Idempotent: cached after first build.
        try:
            request.app.state.router.resolve_chain(chain_name)
        except AdhocResolveError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Forward any non-standard top-level fields (prompt_cache_key, response_format,
        # seed, reasoning_effort, logit_bias, service_tier, …) to OpenAI-compat providers.
        request_extras = _collect_request_extras(req)

        # MoA (Mixture of Agents): parallel fan-out to all proposers, then aggregator-синтез.
        # Stream и tools не поддерживаются в первой итерации — proposals собираются
        # целиком перед синтезом, а tool-calls через ансамбль требуют отдельного дизайна.
        if chain_name == "moa":
            if req.stream:
                raise HTTPException(
                    status_code=400,
                    detail="stream=true не поддерживается с model=moa в первой итерации",
                )
            if req.tools:
                raise HTTPException(
                    status_code=400,
                    detail="tools не поддерживаются с model=moa",
                )
            aggregator_chain = request.query_params.get("aggregator") or "reasoning_quality"
            started = time.monotonic()
            try:
                result, agg_name, resolved_moa, proposals = (
                    await request.app.state.router.chat_moa(
                        messages=messages,
                        temperature=req.temperature,
                        max_tokens=max_tokens,
                        request_extras=request_extras,
                        moa_chain=chain_name,
                        aggregator_chain=aggregator_chain,
                    )
                )
            except Exception as exc:
                log.exception("moa failed (aggregator=%s)", aggregator_chain)
                _log_call(
                    endpoint="moa",
                    outcome="error",
                    client_model=req.model,
                    chain_requested=chain_name,
                    started=started,
                    error=exc,
                    extra={"aggregator_chain": aggregator_chain},
                )
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"moa failed (aggregator {aggregator_chain}): {exc!r}",
                            "type": "moa_failed",
                        }
                    },
                )
            _log_call(
                endpoint="moa",
                outcome="success",
                client_model=req.model,
                chain_requested=chain_name,
                chain_resolved=resolved_moa,
                provider=agg_name,
                started=started,
                result=result,
                extra={
                    "aggregator_chain": aggregator_chain,
                    "proposer_count": len(proposals),
                    "proposer_success": sum(1 for p in proposals if p["error"] is None),
                    "proposers": [p.get("provider") for p in proposals],
                },
            )
            message = ChatMessage(
                role="assistant",
                content=result.text or None,
                tool_calls=result.tool_calls,
            )
            resp = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=agg_name,
                choices=[
                    ChatCompletionChoice(
                        message=message,
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens,
                    cached_tokens=result.cached_tokens or None,
                ),
                provider=agg_name,
            )
            resp_dict = resp.model_dump(exclude_none=True)
            resp_dict["chain"] = resolved_moa
            resp_dict["moa"] = {
                "proposals": proposals,
                "aggregator_chain": aggregator_chain,
                "aggregator_provider": agg_name,
                "proposer_count": len(proposals),
                "proposer_success": sum(1 for p in proposals if p["error"] is None),
            }
            return resp_dict

        # Deep Search: plan → search → synthesize → critique → (iterate once if gaps).
        # Тяжёлая операция (15-40s), non-streaming. Returns финальный markdown +
        # subquestions/sources/trace в response.deep_search.
        if chain_name == "deep_search":
            if req.stream:
                raise HTTPException(
                    status_code=400,
                    detail="stream=true не поддерживается с model=deep_search",
                )
            if req.tools:
                raise HTTPException(
                    status_code=400,
                    detail="tools не поддерживаются с model=deep_search",
                )
            qp = request.query_params
            planner_chain = qp.get("planner") or "reasoning_quality"
            searcher_chain = qp.get("searcher") or "web"
            synthesizer_chain = qp.get("synth") or qp.get("synthesizer") or "reasoning_quality"
            critic_chain = qp.get("critic") or "reasoning_quality"
            try:
                max_subq = int(qp.get("max_subq") or 4)
                rounds = int(qp.get("rounds") or 1)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="max_subq / rounds must be integers"
                )
            max_subq = max(1, min(max_subq, 6))
            rounds = max(0, min(rounds, 2))
            jina_raw = (qp.get("jina") or "").strip().lower()
            jina_enabled = jina_raw not in ("0", "false", "no", "off")
            started = time.monotonic()
            try:
                result, synth_provider, meta = (
                    await request.app.state.router.chat_deep_search(
                        messages=messages,
                        temperature=req.temperature,
                        max_tokens=max_tokens,
                        request_extras=request_extras,
                        planner_chain=planner_chain,
                        searcher_chain=searcher_chain,
                        synthesizer_chain=synthesizer_chain,
                        critic_chain=critic_chain,
                        max_subquestions=max_subq,
                        max_critic_rounds=rounds,
                        jina_enabled=jina_enabled,
                    )
                )
            except Exception as exc:
                log.exception("deep_search failed")
                _log_call(
                    endpoint="deep_search",
                    outcome="error",
                    client_model=req.model,
                    chain_requested=chain_name,
                    started=started,
                    error=exc,
                    extra={
                        "planner_chain": planner_chain,
                        "searcher_chain": searcher_chain,
                        "synthesizer_chain": synthesizer_chain,
                        "critic_chain": critic_chain,
                    },
                )
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"deep_search failed: {exc!r}",
                            "type": "deep_search_failed",
                        }
                    },
                )
            _log_call(
                endpoint="deep_search",
                outcome="success",
                client_model=req.model,
                chain_requested=chain_name,
                chain_resolved="deep_search",
                provider=synth_provider,
                started=started,
                result=result,
                extra={
                    "planner_chain": planner_chain,
                    "searcher_chain": searcher_chain,
                    "synthesizer_chain": synthesizer_chain,
                    "critic_chain": critic_chain,
                    "subquestions": (meta or {}).get("subquestions"),
                    "sources": len((meta or {}).get("sources") or []) if isinstance(meta, dict) else None,
                },
            )
            message = ChatMessage(
                role="assistant",
                content=result.text or None,
                tool_calls=result.tool_calls,
            )
            resp = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=synth_provider,
                choices=[
                    ChatCompletionChoice(
                        message=message,
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens,
                    cached_tokens=result.cached_tokens or None,
                ),
                provider=synth_provider,
            )
            resp_dict = resp.model_dump(exclude_none=True)
            resp_dict["chain"] = "deep_search"
            resp_dict["deep_search"] = meta
            return resp_dict

        # Multi-Agent Debate: N разных моделей × R раундов inter-agent revision.
        # Отличается от MoA: после первого раунда каждый агент видит ответы
        # коллег и переписывает свой с учётом критики. Финальный synth — aggregator.
        if chain_name == "debate":
            if req.tools:
                raise HTTPException(
                    status_code=400,
                    detail="tools не поддерживаются с model=debate",
                )
            qp = request.query_params
            aggregator_chain = qp.get("aggregator") or "reasoning_quality"
            try:
                agents_n = int(qp.get("agents") or 3)
                rounds_n = int(qp.get("rounds") or 2)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="agents / rounds must be integers"
                )
            if agents_n < 2 or agents_n > 6:
                raise HTTPException(
                    status_code=400, detail="agents must be between 2 and 6"
                )
            if rounds_n < 1 or rounds_n > 4:
                raise HTTPException(
                    status_code=400, detail="rounds must be between 1 and 4"
                )

            if req.stream:
                async def debate_event_stream():
                    queue: asyncio.Queue[dict | None] = asyncio.Queue()

                    async def emit(ev: dict) -> None:
                        await queue.put(ev)

                    started = time.monotonic()
                    runner_state: dict[str, Any] = {
                        "outcome": "success",
                        "exc": None,
                        "agg_provider": None,
                    }

                    async def runner():
                        try:
                            ret = await request.app.state.router.chat_debate(
                                messages=messages,
                                temperature=req.temperature,
                                max_tokens=max_tokens,
                                request_extras=request_extras,
                                debate_chain=chain_name,
                                aggregator_chain=aggregator_chain,
                                agents=agents_n,
                                rounds=rounds_n,
                                event_emit=emit,
                            )
                            try:
                                runner_state["agg_provider"] = ret[1]
                            except (TypeError, IndexError):
                                pass
                        except Exception as exc:
                            log.exception("debate stream failed (aggregator=%s)", aggregator_chain)
                            runner_state["outcome"] = "error"
                            runner_state["exc"] = exc
                            await queue.put({"type": "error", "message": repr(exc)})
                        finally:
                            _log_call(
                                endpoint="debate",
                                outcome=str(runner_state["outcome"]),
                                client_model=req.model,
                                chain_requested=chain_name,
                                chain_resolved="debate",
                                provider=runner_state.get("agg_provider"),
                                started=started,
                                stream=True,
                                error=runner_state.get("exc"),
                                extra={
                                    "aggregator_chain": aggregator_chain,
                                    "agents": agents_n,
                                    "rounds": rounds_n,
                                },
                            )
                            await queue.put(None)

                    task = asyncio.create_task(runner())
                    try:
                        while True:
                            ev = await queue.get()
                            if ev is None:
                                break
                            yield f"event: debate\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()
                        yield b"data: [DONE]\n\n"
                    finally:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, Exception):
                                pass

                return StreamingResponse(
                    debate_event_stream(),
                    media_type="text/event-stream",
                )

            started = time.monotonic()
            try:
                result, agg_name, resolved_debate, transcript, agent_names = (
                    await request.app.state.router.chat_debate(
                        messages=messages,
                        temperature=req.temperature,
                        max_tokens=max_tokens,
                        request_extras=request_extras,
                        debate_chain=chain_name,
                        aggregator_chain=aggregator_chain,
                        agents=agents_n,
                        rounds=rounds_n,
                    )
                )
            except Exception as exc:
                log.exception("debate failed (aggregator=%s)", aggregator_chain)
                _log_call(
                    endpoint="debate",
                    outcome="error",
                    client_model=req.model,
                    chain_requested=chain_name,
                    started=started,
                    error=exc,
                    extra={
                        "aggregator_chain": aggregator_chain,
                        "agents": agents_n,
                        "rounds": rounds_n,
                    },
                )
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"debate failed (aggregator {aggregator_chain}): {exc!r}",
                            "type": "debate_failed",
                        }
                    },
                )
            _log_call(
                endpoint="debate",
                outcome="success",
                client_model=req.model,
                chain_requested=chain_name,
                chain_resolved=resolved_debate,
                provider=agg_name,
                started=started,
                result=result,
                extra={
                    "aggregator_chain": aggregator_chain,
                    "agents": agent_names,
                    "agent_count": len(agent_names),
                    "rounds": len(transcript),
                    "final_round_success": sum(
                        1 for a in (transcript[-1] if transcript else []) if a["error"] is None
                    ),
                },
            )
            message = ChatMessage(
                role="assistant",
                content=result.text or None,
                tool_calls=result.tool_calls,
            )
            resp = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=agg_name,
                choices=[
                    ChatCompletionChoice(
                        message=message,
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens,
                    cached_tokens=result.cached_tokens or None,
                ),
                provider=agg_name,
            )
            resp_dict = resp.model_dump(exclude_none=True)
            resp_dict["chain"] = resolved_debate
            last_round = transcript[-1] if transcript else []
            resp_dict["debate"] = {
                "agents": agent_names,
                "rounds": len(transcript),
                "transcript": transcript,
                "aggregator_chain": aggregator_chain,
                "aggregator_provider": agg_name,
                "agent_count": len(agent_names),
                "final_round_success": sum(
                    1 for a in last_round if a["error"] is None
                ),
            }
            return resp_dict

        # SC (Self-Consistency): N сэмплов от ОДНОЙ модели + aggregator-синтез.
        # Отличается от MoA: base-модель одна (первый eligible из цепочки `sc`),
        # разнообразие — за счёт высокой temperature (≥1.0).
        if chain_name == "sc":
            if req.stream:
                raise HTTPException(
                    status_code=400,
                    detail="stream=true не поддерживается с model=sc",
                )
            if req.tools:
                raise HTTPException(
                    status_code=400,
                    detail="tools не поддерживаются с model=sc",
                )
            aggregator_chain = request.query_params.get("aggregator") or "reasoning_quality"
            try:
                samples_n = int(request.query_params.get("samples") or 5)
            except ValueError:
                raise HTTPException(status_code=400, detail="samples must be integer")
            if samples_n < 2 or samples_n > 20:
                raise HTTPException(
                    status_code=400, detail="samples must be between 2 and 20"
                )
            started = time.monotonic()
            try:
                result, agg_name, resolved_sc, samples_list, base_name = (
                    await request.app.state.router.chat_sc(
                        messages=messages,
                        temperature=req.temperature,
                        max_tokens=max_tokens,
                        request_extras=request_extras,
                        sc_chain=chain_name,
                        aggregator_chain=aggregator_chain,
                        samples=samples_n,
                    )
                )
            except Exception as exc:
                log.exception("sc failed (aggregator=%s)", aggregator_chain)
                _log_call(
                    endpoint="sc",
                    outcome="error",
                    client_model=req.model,
                    chain_requested=chain_name,
                    started=started,
                    error=exc,
                    extra={"aggregator_chain": aggregator_chain, "samples": samples_n},
                )
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": f"sc failed (aggregator {aggregator_chain}): {exc!r}",
                            "type": "sc_failed",
                        }
                    },
                )
            _log_call(
                endpoint="sc",
                outcome="success",
                client_model=req.model,
                chain_requested=chain_name,
                chain_resolved=resolved_sc,
                provider=agg_name,
                started=started,
                result=result,
                extra={
                    "aggregator_chain": aggregator_chain,
                    "base_provider": base_name,
                    "sample_count": len(samples_list),
                    "sample_success": sum(1 for s in samples_list if s["error"] is None),
                },
            )
            message = ChatMessage(
                role="assistant",
                content=result.text or None,
                tool_calls=result.tool_calls,
            )
            resp = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=agg_name,
                choices=[
                    ChatCompletionChoice(
                        message=message,
                        finish_reason=result.finish_reason,
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens,
                    cached_tokens=result.cached_tokens or None,
                ),
                provider=agg_name,
            )
            resp_dict = resp.model_dump(exclude_none=True)
            resp_dict["chain"] = resolved_sc
            resp_dict["sc"] = {
                "samples": samples_list,
                "base_provider": base_name,
                "aggregator_chain": aggregator_chain,
                "aggregator_provider": agg_name,
                "sample_count": len(samples_list),
                "sample_success": sum(1 for s in samples_list if s["error"] is None),
            }
            return resp_dict

        if req.stream:
            async def sse_body():
                started = time.monotonic()
                used_provider: str | None = None
                used_chain: str | None = None
                logged = False
                try:
                    first = True
                    async for chunk, _p, _c in request.app.state.router.chat_stream(
                        messages=messages,
                        temperature=req.temperature,
                        max_tokens=max_tokens,
                        tools=req.tools,
                        tool_choice=req.tool_choice,
                        request_extras=request_extras,
                        chain_name=chain_name,
                    ):
                        used_provider = _p
                        used_chain = _c
                        if first:
                            meta = json.dumps({"provider": _p, "chain": _c})
                            yield f"event: neurogate\ndata: {meta}\n\n".encode()
                            first = False
                        yield chunk
                    _log_call(
                        endpoint="chat",
                        outcome="success",
                        client_model=req.model,
                        chain_requested=chain_name,
                        chain_resolved=used_chain,
                        provider=used_provider,
                        started=started,
                        stream=True,
                    )
                    logged = True
                except Exception as exc:
                    log.exception("all providers failed (chain=%s, stream)", chain_name)
                    _log_call(
                        endpoint="chat",
                        outcome="error",
                        client_model=req.model,
                        chain_requested=chain_name,
                        chain_resolved=used_chain,
                        provider=used_provider,
                        started=started,
                        stream=True,
                        error=exc,
                    )
                    logged = True
                    err_payload = json.dumps(
                        {
                            "error": {
                                "message": f"all providers failed on chain {chain_name}: {exc!r}",
                                "type": "upstream_exhausted",
                            }
                        }
                    )
                    yield f"data: {err_payload}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                finally:
                    if not logged:
                        # Client disconnect mid-stream: still record what we know.
                        _log_call(
                            endpoint="chat",
                            outcome="cancelled",
                            client_model=req.model,
                            chain_requested=chain_name,
                            chain_resolved=used_chain,
                            provider=used_provider,
                            started=started,
                            stream=True,
                        )

            return StreamingResponse(
                sse_body(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.chat(
                messages=messages,
                temperature=req.temperature,
                max_tokens=max_tokens,
                tools=req.tools,
                tool_choice=req.tool_choice,
                request_extras=request_extras,
                chain_name=chain_name,
            )
        except Exception as exc:
            log.exception("all providers failed (chain=%s)", chain_name)
            _log_call(
                endpoint="chat",
                outcome="error",
                client_model=req.model,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"tools": bool(req.tools)} if req.tools else None,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"all providers failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )

        _log_call(
            endpoint="chat",
            outcome="success",
            client_model=req.model,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            result=result,
            extra={
                "finish_reason": result.finish_reason,
                "tool_calls": len(result.tool_calls) if result.tool_calls else 0,
            },
        )
        message = ChatMessage(
            role="assistant",
            content=result.text or None,
            tool_calls=result.tool_calls,
        )
        resp = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=used_name,
            choices=[
                ChatCompletionChoice(
                    message=message,
                    finish_reason=result.finish_reason,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
                cached_tokens=result.cached_tokens or None,
            ),
            provider=used_name,
        )
        resp_dict = resp.model_dump(exclude_none=True)
        resp_dict["chain"] = used_chain
        return resp_dict

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("audio"),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float | None = Form(None),
        authorization: str | None = Header(default=None),
    ):
        """OpenAI-compatible speech-to-text. `model` selects chain/provider
        (default "audio"). Accepts multipart/form-data with a file field."""
        _authorize(authorization)
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="empty audio file")
        mime_type = file.content_type or "application/octet-stream"
        filename = file.filename or "audio"
        chain_name = _pick_chain(model)
        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.transcribe(
                audio=audio_bytes,
                filename=filename,
                mime_type=mime_type,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("transcription failed (chain=%s)", chain_name)
            _log_call(
                endpoint="transcribe",
                outcome="error",
                client_model=model,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"bytes": len(audio_bytes), "mime": mime_type},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"transcription failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="transcribe",
            outcome="success",
            client_model=model,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "bytes": len(audio_bytes),
                "mime": mime_type,
                "language": result.language,
                "duration_s": result.duration_s,
            },
        )

        # OpenAI-compat: "text"/"srt"/"vtt" → plain body; "json"/"verbose_json" → JSON.
        if response_format in ("text", "srt", "vtt"):
            return PlainTextResponse(result.text)
        payload: dict[str, Any] = {"text": result.text}
        if result.duration_s is not None:
            payload["duration"] = result.duration_s
        if result.language:
            payload["language"] = result.language
        payload["provider"] = used_name
        payload["chain"] = used_chain
        return payload

    @app.post("/v1/audio/speech")
    async def audio_speech(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """OpenAI-compatible text→speech. JSON body: {input, voice, model,
        response_format, speed}. `model` selects chain/provider (default "tts").
        Returns raw audio bytes (audio/mpeg for mp3). Unknown keys pass through
        as provider extras (e.g. `volume`, `pitch` for Edge TTS)."""
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        input_text = body.get("input")
        if not input_text or not isinstance(input_text, str):
            raise HTTPException(status_code=400, detail="`input` (non-empty string) is required")

        voice = body.get("voice")
        if voice is not None and not isinstance(voice, str):
            raise HTTPException(status_code=400, detail="`voice` must be a string")

        response_format = body.get("response_format", "mp3")
        if response_format not in ("mp3", "opus", "aac", "flac", "wav", "pcm"):
            raise HTTPException(
                status_code=400,
                detail="`response_format` must be one of mp3/opus/aac/flac/wav/pcm",
            )

        speed_raw = body.get("speed", 1.0)
        try:
            speed = float(speed_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="`speed` must be numeric") from None
        if not (0.25 <= speed <= 4.0):
            raise HTTPException(status_code=400, detail="`speed` must be between 0.25 and 4.0")

        model_field = body.get("model") or "tts"
        chain_name = _pick_chain(model_field)

        known = {"input", "voice", "response_format", "speed", "model"}
        extra = {k: v for k, v in body.items() if k not in known}

        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.generate_speech(
                input_text=input_text,
                voice=voice,
                response_format=response_format,
                speed=speed,
                extra=extra or None,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("tts failed (chain=%s)", chain_name)
            _log_call(
                endpoint="speech",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"voice": voice, "format": response_format, "input_len": len(input_text)},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"tts failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="speech",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "voice": result.voice or voice,
                "format": response_format,
                "input_len": len(input_text),
                "audio_bytes": len(result.audio) if result.audio else 0,
            },
        )

        headers = {
            "X-Neurogate-Provider": used_name,
            "X-Neurogate-Chain": used_chain,
        }
        if result.voice:
            headers["X-Neurogate-Voice"] = result.voice
        return Response(
            content=result.audio,
            media_type=result.content_type,
            headers=headers,
        )

    @app.post("/v1/audio/sfx")
    async def audio_sfx(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """Text→sound effect / ambient. JSON body: {prompt, duration?, model?, ...}.
        `model` selects chain/provider (default "sfx"). `duration` clamped to 1-30s.
        Returns raw audio bytes (audio/wav by default). Designed for game SFX and
        ambient loops, not music or speech — see /v1/audio/speech for TTS."""
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        prompt = body.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="`prompt` (non-empty string) is required")

        duration_raw = body.get("duration", body.get("duration_s"))
        duration_s: float | None = None
        if duration_raw is not None:
            try:
                duration_s = float(duration_raw)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="`duration` must be numeric") from None
            if not (1.0 <= duration_s <= 30.0):
                raise HTTPException(status_code=400, detail="`duration` must be between 1 and 30 seconds")

        model_field = body.get("model") or "sfx"
        chain_name = _pick_chain(model_field)

        known = {"prompt", "duration", "duration_s", "model"}
        extra = {k: v for k, v in body.items() if k not in known}

        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.generate_sfx(
                prompt=prompt,
                duration_s=duration_s,
                extra=extra or None,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("sfx failed (chain=%s)", chain_name)
            _log_call(
                endpoint="sfx",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"duration_s": duration_s, "prompt_len": len(prompt)},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"sfx failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="sfx",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "duration_s": result.duration_s,
                "prompt_len": len(prompt),
                "audio_bytes": len(result.audio) if result.audio else 0,
            },
        )

        headers = {
            "X-Neurogate-Provider": used_name,
            "X-Neurogate-Chain": used_chain,
        }
        if result.duration_s is not None:
            headers["X-Neurogate-Duration"] = f"{result.duration_s:.2f}"
        return Response(
            content=result.audio,
            media_type=result.content_type,
            headers=headers,
        )

    @app.post("/v1/embeddings")
    async def embeddings(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """OpenAI-compatible embeddings. JSON body: {input, model, ...extras}.
        `input` — string или list[str]. `model` выбирает chain/provider
        (по умолчанию `embed`; для кода — `embed_code`).

        Pass-through extras уходят провайдеру в `extra`:
        - Voyage: `input_type` ('document'|'query'), `output_dimension`
        - Jina v3+: `task` ('retrieval.query'|'retrieval.passage'|...), `dimensions`
        - Cohere: `input_type` ('search_document'|'search_query'|...), `dimensions`
        - Gemini: `task_type` ('RETRIEVAL_QUERY'|...), `dimensions` (MRL)

        Response: OpenAI-shape {object, data:[{object,embedding,index}], model, usage,
        provider, chain}.
        """
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        raw_input = body.get("input")
        if raw_input is None:
            raise HTTPException(status_code=400, detail="`input` is required (string or list[str])")
        if isinstance(raw_input, str):
            if not raw_input:
                raise HTTPException(status_code=400, detail="`input` must be non-empty")
            input_texts = [raw_input]
        elif isinstance(raw_input, list):
            if not raw_input:
                raise HTTPException(status_code=400, detail="`input` list must be non-empty")
            input_texts = []
            for i, item in enumerate(raw_input):
                if not isinstance(item, str) or not item:
                    raise HTTPException(
                        status_code=400,
                        detail=f"`input[{i}]` must be a non-empty string",
                    )
                input_texts.append(item)
        else:
            raise HTTPException(status_code=400, detail="`input` must be a string or list of strings")

        model_field = body.get("model") or "embed"
        chain_name = _pick_chain(model_field)

        # Pass-through extras (everything outside OpenAI core shape).
        known = {"input", "model", "encoding_format", "user", "dimensions"}
        extra: dict[str, Any] = {k: v for k, v in body.items() if k not in known}
        # `dimensions` стандартный OpenAI-knob для MRL truncation — кидаем в extras
        # под обоими именами, провайдер выберет нужное.
        if "dimensions" in body and body["dimensions"] is not None:
            extra["dimensions"] = body["dimensions"]

        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.embed(
                input_texts=input_texts,
                extra=extra or None,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("embeddings failed (chain=%s)", chain_name)
            _log_call(
                endpoint="embed",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"inputs": len(input_texts)},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"embeddings failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="embed",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "inputs": len(input_texts),
                "dim": result.dim,
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
            },
        )

        data_items = [
            {"object": "embedding", "embedding": vec, "index": i}
            for i, vec in enumerate(result.vectors)
        ]
        return {
            "object": "list",
            "data": data_items,
            "model": result.model or used_name,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "total_tokens": result.prompt_tokens,
            },
            "provider": used_name,
            "chain": used_chain,
        }

    @app.post("/v1/rerank")
    async def rerank(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """Cross-encoder reranker. JSON body: {query, documents, model?, top_n?,
        return_documents?, ...extras}. Шейп — Jina/Cohere-совместимый (POST /rerank).

        `model` выбирает chain/provider (default `rerank`). `documents` — list[str].
        `top_n` ограничивает количество результатов (Voyage: top_k нормализуется
        автоматически). Pass-through extras уходят провайдеру через `extra`.

        Response: {model, results:[{index, relevance_score, document?}], usage:
        {total_tokens}, provider, chain}. `index` указывает на позицию во входном
        списке; список отсортирован по убыванию `relevance_score`.
        """
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        query = body.get("query")
        if not isinstance(query, str) or not query:
            raise HTTPException(status_code=400, detail="`query` is required (non-empty string)")

        raw_docs = body.get("documents")
        if not isinstance(raw_docs, list) or not raw_docs:
            raise HTTPException(
                status_code=400, detail="`documents` is required (non-empty list of strings)"
            )
        documents: list[str] = []
        for i, item in enumerate(raw_docs):
            # Cohere-style {"text": "..."} тоже принимаем для совместимости.
            if isinstance(item, str):
                if not item:
                    raise HTTPException(
                        status_code=400, detail=f"`documents[{i}]` must be non-empty"
                    )
                documents.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"]:
                documents.append(item["text"])
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"`documents[{i}]` must be a non-empty string or {{text: str}}",
                )

        top_n_raw = body.get("top_n") or body.get("top_k")
        top_n: int | None = None
        if top_n_raw is not None:
            try:
                top_n = int(top_n_raw)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="`top_n` must be an integer") from exc
            if top_n <= 0:
                raise HTTPException(status_code=400, detail="`top_n` must be positive")

        return_documents = bool(body.get("return_documents", False))
        model_field = body.get("model") or "rerank"
        chain_name = _pick_chain(model_field)

        known = {"query", "documents", "model", "top_n", "top_k", "return_documents"}
        extra: dict[str, Any] = {k: v for k, v in body.items() if k not in known}

        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.rerank(
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=return_documents,
                extra=extra or None,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("rerank failed (chain=%s)", chain_name)
            _log_call(
                endpoint="rerank",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"docs": len(documents), "top_n": top_n},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"rerank failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="rerank",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "docs": len(documents),
                "top_n": top_n,
                "results": len(result.results) if result.results else 0,
                "model": result.model,
            },
        )

        return {
            "model": result.model or used_name,
            "results": result.results,
            "usage": {"total_tokens": result.total_tokens},
            "provider": used_name,
            "chain": used_chain,
        }

    @app.post("/v1/images/generations")
    async def images_generations(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """OpenAI-compatible text→image. JSON body: {prompt, model, n, size,
        response_format}. `model` selects chain/provider (default "image_gen").
        Unknown top-level keys are forwarded to the provider as `extra` (e.g.
        `negative_prompt`, `enhance`, `safe`, `transparent` for Pollinations)."""
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        prompt = body.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="`prompt` (non-empty string) is required")

        model_field = body.get("model") or "image_gen"
        n_raw = body.get("n", 1)
        try:
            n = int(n_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="`n` must be an integer") from None
        n = max(1, min(n, 4))

        size = body.get("size")
        if size is not None and not isinstance(size, str):
            raise HTTPException(status_code=400, detail="`size` must be a string like '1024x1024'")

        response_format = body.get("response_format", "b64_json")
        if response_format not in ("b64_json", "url"):
            raise HTTPException(
                status_code=400,
                detail="`response_format` must be 'b64_json' or 'url'",
            )

        # Pass-through extras (anything not in the OpenAI core shape). Kept as a
        # dict[str, Any]; provider stringifies scalar values.
        known = {"prompt", "model", "n", "size", "response_format"}
        extra = {k: v for k, v in body.items() if k not in known}

        chain_name = _pick_chain(model_field)
        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.generate_images(
                prompt=prompt,
                n=n,
                size=size,
                response_format=response_format,
                extra=extra or None,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("image generation failed (chain=%s)", chain_name)
            _log_call(
                endpoint="image",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"n": n, "size": size, "prompt_len": len(prompt)},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"image generation failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="image",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "n": n,
                "size": size,
                "prompt_len": len(prompt),
                "model": result.model,
                "images": len(result.images) if result.images else 0,
            },
        )

        return {
            "created": int(time.time()),
            "data": result.images,
            "model": result.model or used_name,
            "provider": used_name,
            "chain": used_chain,
        }

    @app.post("/v1/images/edits")
    async def images_edits(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """OpenAI-compatible image edit. JSON body: {prompt, image, model?, n?,
        size?, response_format?}. `image` must be a `data:image/<fmt>;base64,...`
        URI (raw base64 also accepted — we'll wrap it). `model` selects chain
        (default `image_edit`). Multipart upload not supported in this build —
        send b64 in JSON.
        """
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        prompt = body.get("prompt")
        if not prompt or not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="`prompt` (non-empty string) is required")

        image = body.get("image")
        if not image or not isinstance(image, str):
            raise HTTPException(
                status_code=400,
                detail="`image` is required: data URI 'data:image/png;base64,...' or raw base64",
            )
        # Auto-wrap raw base64 (no data: prefix) — assume PNG.
        if not image.startswith("data:image/"):
            image = "data:image/png;base64," + image.split(",", 1)[-1]

        model_field = body.get("model") or "image_edit"
        n_raw = body.get("n", 1)
        try:
            n = int(n_raw)
        except Exception:
            raise HTTPException(status_code=400, detail="`n` must be an integer") from None
        n = max(1, min(n, 4))

        size = body.get("size")
        if size is not None and not isinstance(size, str):
            raise HTTPException(status_code=400, detail="`size` must be a string like '1024x1024'")

        response_format = body.get("response_format", "b64_json")
        if response_format not in ("b64_json", "url"):
            raise HTTPException(
                status_code=400,
                detail="`response_format` must be 'b64_json' or 'url'",
            )

        known = {"prompt", "image", "model", "n", "size", "response_format"}
        extra = {k: v for k, v in body.items() if k not in known}

        chain_name = _pick_chain(model_field)
        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.edit_images(
                image=image,
                prompt=prompt,
                n=n,
                size=size,
                response_format=response_format,
                extra=extra or None,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("image edit failed (chain=%s)", chain_name)
            _log_call(
                endpoint="image_edit",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"n": n, "size": size, "prompt_len": len(prompt)},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"image edit failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="image_edit",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "n": n,
                "size": size,
                "prompt_len": len(prompt),
                "model": result.model,
                "images": len(result.images) if result.images else 0,
            },
        )

        return {
            "created": int(time.time()),
            "data": result.images,
            "model": result.model or used_name,
            "provider": used_name,
            "chain": used_chain,
        }

    @app.post("/v1/translate")
    async def translate(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """Translation endpoint. JSON body: {text, target_lang, source_lang?, model?}.
        `source_lang` defaults to 'auto' (работает только для провайдеров с
        автодетектом — LibreTranslate и LLM; MyMemory на auto скипается).
        `model` выбирает цепочку (по умолчанию 'translation').

        Response: {text, source_lang, target_lang, provider, chain, provider_model}.
        """
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        text = body.get("text")
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="`text` (non-empty string) is required")

        target_lang = body.get("target_lang") or body.get("target")
        if not target_lang or not isinstance(target_lang, str):
            raise HTTPException(
                status_code=400,
                detail="`target_lang` (ISO-639-1, e.g. 'en') is required",
            )

        source_lang = body.get("source_lang") or body.get("source") or "auto"
        if not isinstance(source_lang, str):
            raise HTTPException(status_code=400, detail="`source_lang` must be a string")

        model_field = body.get("model") or "translation"
        chain_name = _pick_chain(model_field)

        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.translate(
                text=text,
                target_lang=target_lang,
                source_lang=source_lang,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("translate failed (chain=%s)", chain_name)
            _log_call(
                endpoint="translate",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "text_len": len(text),
                },
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"translate failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="translate",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "source_lang": result.source_lang,
                "target_lang": result.target_lang,
                "text_len": len(text),
                "provider_model": result.provider_model,
            },
        )

        return {
            "text": result.text,
            "source_lang": result.source_lang,
            "target_lang": result.target_lang,
            "provider": used_name,
            "chain": used_chain,
            "provider_model": result.provider_model,
        }

    @app.post("/v1/moderations")
    async def moderations(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """OpenAI-compat text moderation. JSON body: {input, model?}.
        `input` — string или list[string]. `model` выбирает chain (default `moderation`).

        Response: {id, model, results: [{flagged, categories, category_scores, ...}],
        provider, chain}.
        """
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        raw_input = body.get("input")
        if raw_input is None:
            raise HTTPException(status_code=400, detail="`input` (string or list of strings) is required")
        if isinstance(raw_input, str):
            input_texts = [raw_input]
        elif isinstance(raw_input, list):
            input_texts = []
            for i, item in enumerate(raw_input):
                if not isinstance(item, str):
                    raise HTTPException(
                        status_code=400,
                        detail=f"`input[{i}]` must be a string for /v1/moderations (text-only)",
                    )
                input_texts.append(item)
        else:
            raise HTTPException(
                status_code=400, detail="`input` must be a string or list of strings"
            )
        if not input_texts:
            raise HTTPException(status_code=400, detail="`input` must not be empty")

        model_field = body.get("model") or "moderation"
        chain_name = _pick_chain(model_field)

        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.moderate_text(
                input_texts=input_texts,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("moderations failed (chain=%s)", chain_name)
            _log_call(
                endpoint="moderation",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"inputs": len(input_texts)},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"moderations failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="moderation",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "inputs": len(input_texts),
                "model": result.model,
                "flagged": sum(1 for r in (result.results or []) if r.get("flagged")),
            },
        )

        return {
            "id": f"modr-{uuid.uuid4().hex[:24]}",
            "model": result.model or used_name,
            "results": result.results,
            "provider": used_name,
            "chain": used_chain,
        }

    @app.post("/v1/moderations/images")
    async def moderations_images(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """Image moderation. JSON body: {input, model?, context_text?}.
        `input` — URL ИЛИ data:image/...;base64,... ИЛИ list таких. Default chain
        `moderation_image`. `context_text` — необязательный текстовый якорь, который
        Llama Guard 4 / omni-moderation учитывают при классификации (например
        промпт пользователя, который привёл к изображению).
        """
        _authorize(authorization)
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        raw_input = body.get("input") or body.get("image") or body.get("images")
        if raw_input is None:
            raise HTTPException(
                status_code=400,
                detail="`input` (image URL/data-URI или list) is required",
            )
        if isinstance(raw_input, str):
            images = [raw_input]
        elif isinstance(raw_input, list):
            images = []
            for i, item in enumerate(raw_input):
                if isinstance(item, str):
                    images.append(item)
                elif isinstance(item, dict):
                    img_url = item.get("image_url")
                    if isinstance(img_url, dict):
                        url = img_url.get("url")
                    else:
                        url = item.get("url") or item.get("b64_json")
                    if not url or not isinstance(url, str):
                        raise HTTPException(
                            status_code=400,
                            detail=f"`input[{i}]` missing image url/b64",
                        )
                    images.append(url)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"`input[{i}]` must be a string or {{image_url|url|b64_json}}",
                    )
        else:
            raise HTTPException(
                status_code=400,
                detail="`input` must be a string, list of strings, or list of OpenAI image parts",
            )
        if not images:
            raise HTTPException(status_code=400, detail="`input` must not be empty")

        context_text = body.get("context_text") or body.get("context")
        if context_text is not None and not isinstance(context_text, str):
            raise HTTPException(status_code=400, detail="`context_text` must be a string")

        model_field = body.get("model") or "moderation_image"
        chain_name = _pick_chain(model_field)

        started = time.monotonic()
        try:
            result, used_name, used_chain = await request.app.state.router.moderate_image(
                images=images,
                context_text=context_text,
                chain_name=chain_name,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("moderations/images failed (chain=%s)", chain_name)
            _log_call(
                endpoint="moderation_image",
                outcome="error",
                client_model=model_field,
                chain_requested=chain_name,
                started=started,
                error=exc,
                extra={"images": len(images)},
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"moderations/images failed on chain {chain_name}: {exc!r}",
                        "type": "upstream_exhausted",
                    }
                },
            )
        _log_call(
            endpoint="moderation_image",
            outcome="success",
            client_model=model_field,
            chain_requested=chain_name,
            chain_resolved=used_chain,
            provider=used_name,
            started=started,
            extra={
                "images": len(images),
                "model": result.model,
                "flagged": sum(1 for r in (result.results or []) if r.get("flagged")),
            },
        )

        return {
            "id": f"modr-img-{uuid.uuid4().hex[:24]}",
            "model": result.model or used_name,
            "results": result.results,
            "provider": used_name,
            "chain": used_chain,
        }

    @app.post("/v1/messages")
    async def messages(
        req: MessagesRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        """Anthropic Messages API compatibility.

        Point any Anthropic client (Claude Code, @anthropic-ai/sdk) at this server
        via `ANTHROPIC_BASE_URL` — the request is translated to OpenAI shape,
        dispatched through the same provider chain, and the response is converted back.
        """
        _authorize(authorization)
        openai_args = request_to_openai(req)
        chain_name = _pick_chain(req.model)

        if req.stream:
            async def sse_body():
                started = time.monotonic()
                used_provider: str | None = None
                used_chain: str | None = None
                logged = False

                async def openai_byte_stream():
                    nonlocal used_provider, used_chain
                    async for chunk, _p, _c in request.app.state.router.chat_stream(
                        messages=openai_args["messages"],
                        temperature=openai_args["temperature"],
                        max_tokens=openai_args["max_tokens"],
                        tools=openai_args["tools"],
                        tool_choice=openai_args["tool_choice"],
                        chain_name=chain_name,
                    ):
                        used_provider = _p
                        used_chain = _c
                        yield chunk

                try:
                    async for ev in translate_stream(
                        openai_byte_stream(), model=req.model
                    ):
                        yield ev
                    _log_call(
                        endpoint="messages",
                        outcome="success",
                        client_model=req.model,
                        chain_requested=chain_name,
                        chain_resolved=used_chain,
                        provider=used_provider,
                        started=started,
                        stream=True,
                    )
                    logged = True
                except Exception as exc:
                    log.exception(
                        "all providers failed on /v1/messages (chain=%s, stream)", chain_name
                    )
                    _log_call(
                        endpoint="messages",
                        outcome="error",
                        client_model=req.model,
                        chain_requested=chain_name,
                        chain_resolved=used_chain,
                        provider=used_provider,
                        started=started,
                        stream=True,
                        error=exc,
                    )
                    logged = True
                    err = json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "type": "upstream_exhausted",
                                "message": f"all providers failed on chain {chain_name}: {exc!r}",
                            },
                        }
                    )
                    yield f"event: error\ndata: {err}\n\n".encode()
                finally:
                    if not logged:
                        _log_call(
                            endpoint="messages",
                            outcome="cancelled",
                            client_model=req.model,
                            chain_requested=chain_name,
                            chain_resolved=used_chain,
                            provider=used_provider,
                            started=started,
                            stream=True,
                        )

            return StreamingResponse(
                sse_body(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        started = time.monotonic()
        try:
            result, used_name, used_chain_msg = await request.app.state.router.chat(
                messages=openai_args["messages"],
                temperature=openai_args["temperature"],
                max_tokens=openai_args["max_tokens"],
                tools=openai_args["tools"],
                tool_choice=openai_args["tool_choice"],
                chain_name=chain_name,
            )
        except Exception as exc:
            log.exception("all providers failed on /v1/messages (chain=%s)", chain_name)
            _log_call(
                endpoint="messages",
                outcome="error",
                client_model=req.model,
                chain_requested=chain_name,
                started=started,
                error=exc,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "type": "error",
                    "error": {
                        "type": "upstream_exhausted",
                        "message": f"all providers failed on chain {chain_name}: {exc!r}",
                    },
                },
            )

        _log_call(
            endpoint="messages",
            outcome="success",
            client_model=req.model,
            chain_requested=chain_name,
            chain_resolved=used_chain_msg,
            provider=used_name,
            started=started,
            result=result,
            extra={"finish_reason": result.finish_reason},
        )
        return result_to_anthropic(
            text=result.text or "",
            tool_calls=result.tool_calls,
            finish_reason=result.finish_reason,
            model=used_name,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
        )

    return app


def run() -> None:
    import uvicorn

    host = os.getenv("NEUROGATE_HOST", "127.0.0.1")
    port = int(os.getenv("NEUROGATE_PORT", "8765"))
    uvicorn.run("neurogate.main:create_app", host=host, port=port, factory=True, log_level="info")


app = None  # lazy; create_app() is the factory uvicorn imports


if __name__ == "__main__":
    run()
