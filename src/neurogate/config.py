from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from neurogate.providers import (
    AIHordeImageProvider,
    CloudflareEmbedProvider,
    CloudflareImageProvider,
    CohereChatProvider,
    CohereEmbedProvider,
    CohereTranslateProvider,
    EdgeTTSProvider,
    FreeTheAiImageProvider,
    GeminiEmbedProvider,
    GeminiImageProvider,
    GeminiProvider,
    GigaChatImageProvider,
    GigaChatProvider,
    GroqWhisperProvider,
    HFSpaceAudioProvider,
    LibreTranslateProvider,
    LlamaGuardProvider,
    MistralModerationProvider,
    MyMemoryProvider,
    OpenAICompatProvider,
    OpenAIEmbedProvider,
    OpenAIModerationProvider,
    PollinationsImageProvider,
    Provider,
    RerankProvider,
    TogetherImageProvider,
    YandexARTImageProvider,
    YandexTranslateProvider,
)

# Kinds that don't need an API key:
#   edge_tts — anonymous WebSocket to speech.platform.bing.com
#   libretranslate — most public mirrors (fedilab, self-hosted) accept no auth
#   mymemory — 5000 chars/day anonymous, 50000 с contact email (не ключом)
#   aihorde — anonymous через apikey="0000000000" (community-distributed inference)
_NO_API_KEY_KINDS = {"edge_tts", "libretranslate", "mymemory", "aihorde", "hf_space_audio", "ovhcloud"}

# Kinds that don't have a model concept (single-purpose services). `model` поле
# в yaml для них опциональное — у провайдера свой дефолт или оно не применимо.
#   edge_tts — default voice instead of model
#   libretranslate / mymemory / yandex_translate — один MT-движок, не выбирается
_NO_MODEL_KINDS = _NO_API_KEY_KINDS | {"yandex_translate", "hf_space_audio"}

log = logging.getLogger(__name__)


PROVIDER_KIND_DEFAULTS = {
    "gemini": {"base_url": None},
    "openai": {"base_url": "https://api.openai.com/v1"},
    "groq": {"base_url": "https://api.groq.com/openai/v1"},
    "cerebras": {"base_url": "https://api.cerebras.ai/v1"},
    "sambanova": {"base_url": "https://api.sambanova.ai/v1"},
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1"},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "extra_headers": {"HTTP-Referer": "https://github.com/neurogate", "X-Title": "neurogate"},
    },
    "zai": {
        "base_url": "https://api.z.ai/api/paas/v4",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    "cloudflare": {"base_url": None},  # needs account_id
    "github": {"base_url": "https://models.github.ai/inference"},
    # HuggingFace Inference Providers Router (OpenAI-compat). Free token = ~$0.10/mo
    # shared monthly credit across paid sub-providers; PRO раз в 20× больше.
    # Модель указывается как "<repo>:<provider>" (e.g. deepseek-ai/DeepSeek-V3.2-Exp:novita).
    "huggingface": {"base_url": "https://router.huggingface.co/v1"},
    # FreeTheAi (api.freetheai.xyz) — OpenAI-compat, ~16k aliases (cat/*, bbg/*, fth/*, or/*…).
    # Лимиты: 10 RPM, 1 concurrent на ключ. Ключ требует daily Discord /checkin —
    # без чек-ина все запросы возвращают HTTP 403 daily_checkin_required. Поэтому
    # включён в роутер только как kind, в общие chains не подмешиваем.
    "freetheai": {"base_url": "https://api.freetheai.xyz/v1"},
    # Same FreeTheAi key, image-gen endpoint. URL responses (vhr/*) are downloaded
    # and re-encoded as b64 inside the provider — the upstream URLs expire fast.
    "freetheai_image": {"base_url": "https://api.freetheai.xyz/v1"},
    "gigachat": {"base_url": "https://gigachat.devices.sberbank.ru/api/v1"},
    "gigachat_image": {"base_url": "https://gigachat.devices.sberbank.ru/api/v1"},
    "yandex_foundation": {"base_url": "https://llm.api.cloud.yandex.net/v1"},
    "mistral": {"base_url": "https://api.mistral.ai/v1"},
    # DeepSeek direct API. OpenAI-compat. 5M токенов кредитов на регистрацию,
    # дальше PAYG ($0.28/$0.42 за 1M на V3.2). Off-peak 50–75% скидка 16:30–00:30 UTC.
    # Prefix caching (cache hit 4–10× дешевле miss) — структурируй промпты так, чтобы
    # стабильный префикс шёл первым. V4-Flash в превью с 1M context.
    "deepseek": {"base_url": "https://api.deepseek.com/v1"},
    # Alibaba DashScope International (Qwen). OpenAI-compat compatible-mode.
    # Регистрация через Alibaba Cloud International, 1M токенов бесплатно на модель.
    # qwen3.6-plus / qwen3.6-max-preview / qwen3-coder-480b — флагманы 1M-context.
    "dashscope": {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"},
    # OVHcloud AI Endpoints — anonymous free tier без ключа и регистрации (2 RPM на IP
    # на модель). 40+ open-моделей в EU. Идеален как deep fallback last-resort.
    # OpenAICompatProvider пропускает Authorization header при пустом api_key.
    "ovhcloud": {"base_url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"},
    "groq_whisper": {"base_url": "https://api.groq.com/openai/v1"},
    "pollinations_image": {"base_url": "https://image.pollinations.ai"},
    "gemini_image": {"base_url": None},
    "cloudflare_image": {"base_url": None},  # needs account_id
    "together_image": {"base_url": "https://api.together.xyz/v1"},
    "yandex_art": {"base_url": "https://llm.api.cloud.yandex.net/foundationModels/v1"},
    "aihorde": {"base_url": "https://aihorde.net/api/v2"},
    # Pollinations legacy text API — OpenAI-compat, anonymous tier = openai-fast only.
    # Referer обязателен для tier recognition (иначе всегда anonymous, вне зависимости от Bearer).
    "pollinations": {
        "base_url": "https://text.pollinations.ai/v1",
        "extra_headers": {"Referer": "neurogate"},
    },
    # LibreTranslate-compat public instance. libretranslate.com/.de стали платными
    # в начале 2026; fedilab — живой community-mirror (2026-04-23 probe).
    "libretranslate": {"base_url": "https://translate.fedilab.app"},
    "mymemory": {"base_url": "https://api.mymemory.translated.net"},
    "cohere": {"base_url": "https://api.cohere.com/v2"},
    "cohere_chat": {"base_url": "https://api.cohere.com/v2"},
    "yandex_translate": {"base_url": "https://translate.api.cloud.yandex.net"},
    # Embeddings: OpenAI-compat, разные base_url. Cohere/Gemini/Cloudflare —
    # отдельные классы (свой шейп).
    "voyage_embed": {"base_url": "https://api.voyageai.com/v1"},
    "jina_embed": {"base_url": "https://api.jina.ai/v1"},
    "mistral_embed": {"base_url": "https://api.mistral.ai/v1"},
    "nvidia_embed": {"base_url": "https://integrate.api.nvidia.com/v1"},
    "github_embed": {"base_url": "https://models.github.ai/inference"},
    "openai_embed": {"base_url": "https://api.openai.com/v1"},
    "cohere_embed": {"base_url": "https://api.cohere.com/v2"},
    "gemini_embed": {"base_url": None},
    "cloudflare_embed": {"base_url": None},  # needs account_id
    # Reranker: разные хосты, единый класс RerankProvider. Все три — POST /rerank,
    # Bearer auth, request почти идентичен (top_n vs top_k у Voyage, results vs data).
    "jina_rerank": {"base_url": "https://api.jina.ai/v1"},
    "cohere_rerank": {"base_url": "https://api.cohere.com/v2"},
    "voyage_rerank": {"base_url": "https://api.voyageai.com/v1"},
    # Moderation: разные шейпы (OpenAI/Mistral — native /v1/moderations; Llama Guard —
    # chat-completions wrapper). Llama Guard разворачивается на любой OpenAI-compat
    # базе через `kind: llama_guard` + `base_url:` (Groq/Together/OpenRouter/CF).
    "openai_moderation": {"base_url": "https://api.openai.com/v1"},
    "mistral_moderation": {"base_url": "https://api.mistral.ai/v1"},
    "llama_guard": {"base_url": None},  # обязательное base_url через config
}


@dataclass
class SkippedProvider:
    """A provider entry that didn't make it into the runtime config.

    `env_var` is the env-variable name the user can set to enable it (None если
    провайдер требует чего-то другого, например `space_id` для hf_space_audio).
    """

    name: str
    kind: str
    reason: str  # "no_api_key" | "no_model" | "missing_field"
    env_var: str | None = None


@dataclass
class ChainConfig:
    """Resolved chains + providers from config.yaml.

    Mutable: PUT /v1/chains rewrites the YAML and pokes new chains/default
    into the in-memory cfg + router без рестарта. Endpoints, захватившие
    `cfg` в замыкании при create_app, видят актуальную картину после reload.
    """

    chains: dict[str, list[Provider]]
    default: str
    all_providers: list[Provider]
    skipped: list[SkippedProvider] = field(default_factory=list)


# Ad-hoc model resolution. Юзер шлёт `model: "openai:gpt-5-foo"` — мы видим, что
# такого ни в chains, ни в providers нет, парсим префикс как kind, забираем
# дефолтный env-ключ и собираем OpenAICompatProvider на лету. Никакого UI/storage
# для ключа: единственный источник правды — серверный env.
#
# Сюда попадают только kinds, которые работают на чистом OpenAICompatProvider без
# дополнительной конфигурации (folder_id, account_id, custom auth schemes).
# gemini/gigachat/yandex_foundation/cloudflare/llama_guard — отдельные классы и
# требуют доп. полей, ad-hoc для них не делаем.
_ADHOC_DEFAULT_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "github": "GITHUB_MODELS_TOKEN",
    "huggingface": "HF_TOKEN",
    "zai": "ZAI_API_KEY",
    "pollinations": "POLLINATIONS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "freetheai": "FREETHEAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    # ovhcloud — anonymous, no env key. Excluded from ad-hoc resolution because
    # ad-hoc requires an env var to be set (see build_adhoc_provider). Configure
    # via providers: yaml entry with kind: ovhcloud (no api_key_env).
}


class AdhocResolveError(LookupError):
    """Raised when an ad-hoc `kind:model_id` cannot be built (unknown kind,
    missing env key, etc.). main.py catches and returns 400 with the message."""


def build_adhoc_provider(model_string: str) -> Provider:
    """Build a one-shot Provider from a `kind:model_id` (or `kind/model_id`) string.

    Returns an OpenAICompatProvider if the kind is supported and an env-key is
    set on the server; otherwise raises AdhocResolveError with a human-readable
    reason. The caller is responsible for caching to avoid rebuilding HTTP clients.
    """
    if not model_string:
        raise AdhocResolveError("empty model string")
    sep_idx = -1
    for ch in (":", "/"):
        idx = model_string.find(ch)
        if idx > 0 and (sep_idx == -1 or idx < sep_idx):
            sep_idx = idx
    if sep_idx <= 0:
        raise AdhocResolveError(
            f"model {model_string!r} is not a known chain/provider and has no "
            f"`kind:model_id` prefix — cannot resolve as ad-hoc"
        )
    kind = model_string[:sep_idx].lower()
    model_id = model_string[sep_idx + 1 :]
    if not model_id:
        raise AdhocResolveError(f"model {model_string!r}: empty model_id after kind prefix")

    env_var = _ADHOC_DEFAULT_API_KEY_ENV.get(kind)
    if env_var is None:
        supported = ", ".join(sorted(_ADHOC_DEFAULT_API_KEY_ENV))
        raise AdhocResolveError(
            f"kind {kind!r} is not supported for ad-hoc models (need: one of {supported})"
        )
    api_key = os.getenv(env_var, "").strip()
    if not api_key:
        raise AdhocResolveError(
            f"kind {kind!r} requires {env_var} on the server, which is not set"
        )

    defaults = PROVIDER_KIND_DEFAULTS.get(kind, {})
    base_url = defaults.get("base_url")
    if not base_url:
        raise AdhocResolveError(
            f"kind {kind!r} has no default base_url — ad-hoc not supported"
        )
    extra_headers = dict(defaults.get("extra_headers", {})) or None
    extra_body = dict(defaults.get("extra_body", {})) or None

    return OpenAICompatProvider(
        name=model_string,
        kind=kind,
        base_url=base_url,
        api_key=api_key,
        model=model_id,
        extra_headers=extra_headers,
        extra_body=extra_body,
        timeout=60.0,
    )


def _resolve_api_key(entry: dict[str, Any]) -> str:
    if "api_key" in entry and entry["api_key"]:
        return str(entry["api_key"])
    env_name = entry.get("api_key_env")
    if env_name:
        val = os.getenv(env_name)
        if val:
            return val
    return ""


def _resolve_pool_keys(entry: dict[str, Any]) -> list[str]:
    """Multi-key pool: return ordered, deduped list of keys from `api_keys` (inline list
    or CSV string) and/or `api_keys_env` (CSV in env var). Returns [] if neither field is set."""
    keys: list[str] = []
    raw = entry.get("api_keys")
    if isinstance(raw, list):
        keys.extend(str(k).strip() for k in raw if str(k).strip())
    elif isinstance(raw, str) and raw.strip():
        keys.extend(k.strip() for k in raw.split(",") if k.strip())
    env_name = entry.get("api_keys_env")
    if env_name:
        val = os.getenv(env_name, "")
        if val:
            keys.extend(k.strip() for k in val.split(",") if k.strip())
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _expand_pool_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """If the entry declares a multi-key pool with ≥2 keys, return N copies — one per
    key — with `name#1/#2/...` suffixes and inline `api_key` set. Otherwise return the
    entry unchanged in a single-element list (preserves single `api_key_env` path)."""
    keys = _resolve_pool_keys(entry)
    if len(keys) < 2:
        return [entry]
    base = entry.get("name") or f"{entry.get('kind','?')}:{entry.get('model','?')}"
    out: list[dict[str, Any]] = []
    for i, k in enumerate(keys, start=1):
        clone = dict(entry)
        clone.pop("api_keys", None)
        clone.pop("api_keys_env", None)
        clone.pop("api_key_env", None)
        clone["api_key"] = k
        clone["name"] = f"{base}#{i}"
        out.append(clone)
    return out


def _build_provider(
    entry: dict[str, Any],
    skipped: list[SkippedProvider] | None = None,
) -> Provider | None:
    name = entry.get("name") or f"{entry.get('kind','?')}:{entry.get('model','?')}"
    kind = (entry.get("kind") or "openai").lower()
    api_key = _resolve_api_key(entry)

    if not api_key and kind not in _NO_API_KEY_KINDS:
        env_var = entry.get("api_key_env") or entry.get("api_keys_env")
        log.warning("skipping %s: no api key (set %s in env)", name, env_var)
        if skipped is not None:
            skipped.append(SkippedProvider(name, kind, "no_api_key", env_var))
        return None

    defaults = PROVIDER_KIND_DEFAULTS.get(kind, {})
    base_url = entry.get("base_url") or defaults.get("base_url")
    extra_headers = {**defaults.get("extra_headers", {}), **(entry.get("extra_headers") or {})}
    extra_body = {**defaults.get("extra_body", {}), **(entry.get("extra_body") or {})}
    model = entry.get("model")
    if not model and kind not in _NO_MODEL_KINDS:
        log.warning("skipping %s: model not set", name)
        if skipped is not None:
            skipped.append(SkippedProvider(name, kind, "no_model", None))
        return None

    rpd = entry.get("rpd")
    rpm = entry.get("rpm")
    rpd = int(rpd) if rpd else None
    rpm = int(rpm) if rpm else None
    context_window = entry.get("context_window")
    context_window = int(context_window) if context_window else None
    max_output_tokens = entry.get("max_output_tokens")
    max_output_tokens = int(max_output_tokens) if max_output_tokens else None
    quality = entry.get("quality")
    quality = int(quality) if quality is not None else None
    latency_s = entry.get("latency_s")
    latency_s = float(latency_s) if latency_s is not None else None
    ru = entry.get("ru")
    ru = int(ru) if ru is not None else None
    reasoning = bool(entry.get("reasoning", False))

    if kind == "gemini":
        return GeminiProvider(
            name,
            api_key=api_key,
            model=model,
            rpd=rpd,
            rpm=rpm,
            context_window=context_window,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
            reasoning=reasoning,
        )

    if kind == "groq_whisper":
        return GroqWhisperProvider(
            name=name,
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.groq.com/openai/v1",
            timeout=float(entry.get("timeout", 120.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "edge_tts":
        return EdgeTTSProvider(
            name=name,
            model=model or "edge-tts",
            default_voice=str(entry.get("default_voice") or "en-US-AriaNeural"),
            timeout=float(entry.get("timeout", 60.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "hf_space_audio":
        space_id = entry.get("space_id")
        if not space_id:
            log.warning("skipping %s: hf_space_audio requires space_id", name)
            return None
        prompt_field = entry.get("prompt_field", "prompt")
        if prompt_field is False:
            prompt_field = None
        duration_field = entry.get("duration_field")
        if duration_field is False:
            duration_field = None
        return HFSpaceAudioProvider(
            name=name,
            space_id=str(space_id),
            api_name=str(entry.get("api_name") or "/predict"),
            prompt_field=prompt_field,
            duration_field=duration_field,
            default_duration_s=float(entry.get("default_duration_s", 10.0)),
            hf_token=api_key or None,
            timeout=float(entry.get("timeout", 180.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "libretranslate":
        return LibreTranslateProvider(
            name=name,
            base_url=base_url or "https://translate.fedilab.app",
            api_key=api_key,  # пустая строка если ключа нет — это OK для fedilab
            timeout=float(entry.get("timeout", 20.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "yandex_translate":
        folder_id = entry.get("folder_id") or os.getenv("YANDEX_FOLDER_ID", "")
        if not folder_id:
            log.warning("skipping %s: YANDEX_FOLDER_ID missing", name)
            return None
        return YandexTranslateProvider(
            name=name,
            api_key=api_key,
            folder_id=folder_id,
            base_url=base_url or "https://translate.api.cloud.yandex.net",
            timeout=float(entry.get("timeout", 30.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "cohere_chat":
        return CohereChatProvider(
            name=name,
            api_key=api_key,
            model=model or "command-r-08-2024",
            base_url=base_url or "https://api.cohere.com/v2",
            timeout=float(entry.get("timeout", 60.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "cohere":
        return CohereTranslateProvider(
            name=name,
            api_key=api_key,
            model=model or "c4ai-aya-expanse-32b",
            base_url=base_url or "https://api.cohere.com/v2",
            timeout=float(entry.get("timeout", 60.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    # === Embeddings ===
    # Универсальный OpenAI-compat для Voyage/Jina/Mistral/NVIDIA/GitHub/OpenAI.
    # Шейп идентичный — отличаются только base_url + опциональные extra_body
    # (Voyage `input_type`, Jina `task`/`normalized`, MRL `dimensions`).
    if kind in ("voyage_embed", "jina_embed", "mistral_embed", "nvidia_embed",
                "github_embed", "openai_embed"):
        dim_raw = entry.get("dim")
        dim = int(dim_raw) if dim_raw is not None else None
        return OpenAIEmbedProvider(
            name=name,
            kind=kind,
            base_url=base_url or "",
            api_key=api_key,
            model=model,
            extra_headers=extra_headers or None,
            extra_body=extra_body or None,
            timeout=float(entry.get("timeout", 60.0)),
            dim=dim,
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
            quota_limited=bool(entry.get("quota_limited", False)),
        )

    if kind == "cohere_embed":
        dim_raw = entry.get("dim")
        dim = int(dim_raw) if dim_raw is not None else None
        return CohereEmbedProvider(
            name=name,
            api_key=api_key,
            model=model or "embed-multilingual-v3.0",
            base_url=base_url or "https://api.cohere.com/v2",
            timeout=float(entry.get("timeout", 60.0)),
            dim=dim,
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
            quota_limited=bool(entry.get("quota_limited", True)),
        )

    if kind == "gemini_embed":
        dim_raw = entry.get("dim")
        dim = int(dim_raw) if dim_raw is not None else None
        return GeminiEmbedProvider(
            name=name,
            api_key=api_key,
            model=model or "gemini-embedding-001",
            dim=dim,
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "cloudflare_embed":
        account_id = entry.get("account_id") or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            log.warning("skipping %s: cloudflare account_id missing", name)
            return None
        dim_raw = entry.get("dim")
        dim = int(dim_raw) if dim_raw is not None else None
        return CloudflareEmbedProvider(
            name=name,
            api_key=api_key,
            account_id=account_id,
            model=model or "@cf/baai/bge-m3",
            timeout=float(entry.get("timeout", 60.0)),
            dim=dim,
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind in ("jina_rerank", "cohere_rerank", "voyage_rerank"):
        return RerankProvider(
            name=name,
            kind=kind,
            base_url=base_url or "",
            api_key=api_key,
            model=model,
            extra_headers=extra_headers or None,
            extra_body=extra_body or None,
            timeout=float(entry.get("timeout", 60.0)),
            max_documents=int(entry["max_documents"]) if entry.get("max_documents") else None,
            max_query_chars=int(entry["max_query_chars"]) if entry.get("max_query_chars") else None,
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
            quota_limited=bool(entry.get("quota_limited", False)),
        )

    if kind == "mymemory":
        return MyMemoryProvider(
            name=name,
            base_url=base_url or "https://api.mymemory.translated.net",
            contact_email=str(entry.get("contact_email") or os.getenv("MYMEMORY_CONTACT_EMAIL", "")),
            timeout=float(entry.get("timeout", 15.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "openai_moderation":
        return OpenAIModerationProvider(
            name=name,
            api_key=api_key,
            model=model or "omni-moderation-latest",
            base_url=base_url or "https://api.openai.com/v1",
            timeout=float(entry.get("timeout", 30.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "mistral_moderation":
        return MistralModerationProvider(
            name=name,
            api_key=api_key,
            model=model or "mistral-moderation-latest",
            base_url=base_url or "https://api.mistral.ai/v1",
            timeout=float(entry.get("timeout", 30.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "llama_guard":
        # Поддерживаем два способа задать base_url:
        #   1. явно через `base_url:` в YAML (для Together / любого custom OpenAI-compat),
        #   2. через `host: groq|together|openrouter|cloudflare` (sugar для популярных).
        host = (entry.get("host") or "").lower()
        host_urls = {
            "groq": "https://api.groq.com/openai/v1",
            "together": "https://api.together.xyz/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        resolved_base = base_url or host_urls.get(host)
        if host == "cloudflare":
            account_id = entry.get("account_id") or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
            if not account_id:
                log.warning("skipping %s: cloudflare account_id missing", name)
                return None
            resolved_base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        if not resolved_base:
            log.warning("skipping %s: llama_guard needs base_url or host", name)
            return None
        # OpenRouter требует Referer/X-Title (как у обычного openrouter kind).
        lg_extra_headers = dict(extra_headers)
        if host == "openrouter":
            lg_extra_headers.setdefault("HTTP-Referer", "https://github.com/neurogate")
            lg_extra_headers.setdefault("X-Title", "neurogate")
        return LlamaGuardProvider(
            name=name,
            kind=f"llama_guard:{host or 'custom'}",
            base_url=resolved_base,
            api_key=api_key,
            model=model,
            multimodal=bool(entry.get("multimodal", False)),
            extra_headers=lg_extra_headers or None,
            timeout=float(entry.get("timeout", 30.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
            quota_limited=bool(entry.get("quota_limited", False)),
        )

    if kind == "pollinations_image":
        return PollinationsImageProvider(
            name=name,
            api_key=api_key,
            model=model,
            base_url=base_url or "https://image.pollinations.ai",
            referer=str(entry.get("referer") or "neurogate"),
            timeout=float(entry.get("timeout", 60.0)),
            min_interval_s=float(entry.get("min_interval_s", 5.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "gemini_image":
        return GeminiImageProvider(
            name=name,
            api_key=api_key,
            model=model,
            timeout=float(entry.get("timeout", 90.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "cloudflare_image":
        account_id = entry.get("account_id") or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            log.warning("skipping %s: cloudflare account_id missing", name)
            return None
        steps = entry.get("steps")
        steps = int(steps) if steps is not None else None
        return CloudflareImageProvider(
            name=name,
            api_key=api_key,
            account_id=account_id,
            model=model,
            timeout=float(entry.get("timeout", 60.0)),
            steps=steps,
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "yandex_art":
        folder_id = entry.get("folder_id") or os.getenv("YANDEX_FOLDER_ID", "")
        if not folder_id:
            log.warning("skipping %s: YANDEX_FOLDER_ID missing", name)
            return None
        return YandexARTImageProvider(
            name=name,
            api_key=api_key,
            folder_id=folder_id,
            model=model or "yandex-art/latest",
            base_url=base_url or "https://llm.api.cloud.yandex.net/foundationModels/v1",
            timeout=float(entry.get("timeout", 120.0)),
            max_wait_s=float(entry.get("max_wait_s", 90.0)),
            poll_interval_s=float(entry.get("poll_interval_s", 2.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
            quota_limited=bool(entry.get("quota_limited", True)),
        )

    if kind == "freetheai_image":
        return FreeTheAiImageProvider(
            name=name,
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.freetheai.xyz/v1",
            timeout=float(entry.get("timeout", 90.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "together_image":
        steps = entry.get("steps")
        steps = int(steps) if steps is not None else None
        return TogetherImageProvider(
            name=name,
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.together.xyz/v1",
            timeout=float(entry.get("timeout", 60.0)),
            steps=steps,
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "aihorde":
        # Anonymous apikey работает без регистрации (low priority).
        # Если задан AIHORDE_API_KEY — поднимет приоритет в очереди.
        ahkey = api_key or os.getenv("AIHORDE_API_KEY") or "0000000000"
        steps = entry.get("steps")
        steps = int(steps) if steps is not None else 20
        return AIHordeImageProvider(
            name=name,
            api_key=ahkey,
            model=model or "stable_diffusion",
            base_url=base_url or "https://aihorde.net/api/v2",
            client_agent=str(entry.get("client_agent")
                             or "neurogate:0.1:https://github.com/neurogate"),
            steps=steps,
            sampler_name=str(entry.get("sampler_name") or "k_euler"),
            cfg_scale=float(entry.get("cfg_scale", 7.0)),
            max_wait_s=float(entry.get("max_wait_s", 180.0)),
            poll_interval_s=float(entry.get("poll_interval_s", 3.0)),
            timeout=float(entry.get("timeout", 240.0)),
            rpd=rpd,
            rpm=rpm,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
        )

    if kind == "yandex_foundation":
        # Yandex AI Studio /v1/chat/completions — OpenAI-compat, но:
        #   - Authorization: Api-Key <key> (НЕ Bearer)
        #   - model URI: gpt://<folder_id>/<short_model>/<version>
        # Покрывается грантом 6 000 ₽ до 22.10.2026 (Yandex AI Studio).
        folder_id = entry.get("folder_id") or os.getenv("YANDEX_FOLDER_ID", "")
        if not folder_id:
            log.warning("skipping %s: YANDEX_FOLDER_ID missing", name)
            return None
        full_model = f"gpt://{folder_id}/{model}" if not model.startswith("gpt://") else model
        return OpenAICompatProvider(
            name=name,
            kind="yandex_foundation",
            base_url=base_url or "https://llm.api.cloud.yandex.net/v1",
            api_key=api_key,
            model=full_model,
            extra_headers=extra_headers or None,
            extra_body=extra_body or None,
            auth_scheme="Api-Key",
            timeout=float(entry.get("timeout", 60.0)),
            rpd=rpd,
            rpm=rpm,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
            reasoning=reasoning,
            quota_limited=bool(entry.get("quota_limited", True)),
        )

    if kind in ("gigachat", "gigachat_image"):
        # Russian Trusted Root CA не лежит в стандартных трастсторах — verify_ssl
        # по умолчанию False. Можно подложить путь к PEM через verify_ssl: <path>.
        verify_raw = entry.get("verify_ssl", False)
        verify_ssl: bool | str = verify_raw if isinstance(verify_raw, str) else bool(verify_raw)
        cls = GigaChatImageProvider if kind == "gigachat_image" else GigaChatProvider
        return cls(
            name=name,
            auth_key=api_key,
            model=model,
            scope=str(entry.get("scope") or "GIGACHAT_API_PERS"),
            base_url=base_url or "https://gigachat.devices.sberbank.ru/api/v1",
            timeout=float(entry.get("timeout", 60.0)),
            verify_ssl=verify_ssl,
            rpd=rpd,
            rpm=rpm,
            context_window=context_window,
            quality=quality,
            latency_s=latency_s,
            ru=ru,
            reasoning=reasoning,
            quota_limited=bool(entry.get("quota_limited", True)),
        )

    if kind == "cloudflare":
        account_id = entry.get("account_id") or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if not account_id:
            log.warning("skipping %s: cloudflare account_id missing", name)
            return None
        base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

    if not base_url:
        log.warning("skipping %s: base_url resolution failed", name)
        return None

    return OpenAICompatProvider(
        name=name,
        kind=kind,
        base_url=base_url,
        api_key=api_key,
        model=model,
        extra_headers=extra_headers or None,
        extra_body=extra_body or None,
        timeout=float(entry.get("timeout", 60.0)),
        rpd=rpd,
        rpm=rpm,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        quality=quality,
        latency_s=latency_s,
        ru=ru,
        reasoning=reasoning,
        quota_limited=bool(entry.get("quota_limited", False)),
    )


def load_config(config_path: Path | str) -> ChainConfig:
    """Load providers + chains from YAML.

    Two supported shapes:
    - New: `providers: [...]` + `chains: {name: [provider_name, ...]}` + `default_chain: <name>`
    - Legacy: `providers: [...]` only → collapses to single chain "auto".
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_providers = data.get("providers") or []
    if not isinstance(raw_providers, list):
        raise ValueError("config.providers must be a list")

    by_name: dict[str, Provider] = {}
    ordered: list[Provider] = []
    skipped: list[SkippedProvider] = []
    pool_aliases: dict[str, list[str]] = {}  # base name → expanded names (for chain refs)
    for entry in raw_providers:
        sub_entries = _expand_pool_entry(entry)
        if len(sub_entries) > 1:
            base = entry.get("name") or f"{entry.get('kind','?')}:{entry.get('model','?')}"
            pool_aliases[base] = [s["name"] for s in sub_entries]
        for sub in sub_entries:
            p = _build_provider(sub, skipped=skipped)
            if p is None:
                continue
            if sub.get("tools") is False:
                p.supports_tools = False
            if sub.get("vision") is True:
                p.supports_vision = True
            if p.name in by_name:
                log.warning("duplicate provider name %s — keeping first", p.name)
                continue
            by_name[p.name] = p
            ordered.append(p)

    if not ordered:
        raise RuntimeError(_format_no_providers_error(skipped))

    raw_chains = data.get("chains")
    if raw_chains is None:
        chains = {"auto": list(ordered)}
        default = "auto"
        return ChainConfig(
            chains=chains, default=default, all_providers=ordered, skipped=skipped
        )

    if not isinstance(raw_chains, dict) or not raw_chains:
        raise ValueError("config.chains must be a non-empty mapping of chain_name → [provider_name, ...]")

    chains: dict[str, list[Provider]] = {}
    empty_chains: list[tuple[str, list[str]]] = []  # for the final summary
    for chain_name, provider_names in raw_chains.items():
        if not isinstance(provider_names, list):
            raise ValueError(f"chains.{chain_name} must be a list of provider names")
        resolved: list[Provider] = []
        chain_missing: list[str] = []
        for pname in provider_names:
            # Pool alias: chain refs use the base name; expand to the per-key siblings here.
            if pname in pool_aliases:
                expanded_resolved = [
                    by_name[exp] for exp in pool_aliases[pname] if exp in by_name
                ]
                if expanded_resolved:
                    resolved.extend(expanded_resolved)
                else:
                    chain_missing.append(pname)
                continue
            provider = by_name.get(pname)
            if provider is None:
                # Ad-hoc kind:model_id — построить on the fly из env-ключа.
                # Имя провайдера === строка из YAML, чтобы round-trip был стабильным.
                try:
                    provider = build_adhoc_provider(pname)
                except AdhocResolveError as exc:
                    log.warning(
                        "chain %s references unknown/unavailable provider %s — skipping (%s)",
                        chain_name,
                        pname,
                        exc,
                    )
                    chain_missing.append(pname)
                    continue
                by_name[provider.name] = provider
                ordered.append(provider)
            resolved.append(provider)
        if not resolved:
            # Don't crash — drop the chain and let the rest of the gateway start.
            # If a client requests this chain, router falls back to default.
            log.warning(
                "chain '%s' has no resolvable providers — skipping it (set env vars to enable)",
                chain_name,
            )
            empty_chains.append((chain_name, chain_missing))
            continue
        chains[chain_name] = resolved

    if not chains:
        raise RuntimeError(_format_all_chains_empty_error(empty_chains, skipped))

    default = data.get("default_chain") or next(iter(chains))
    if default not in chains:
        # The configured default chain is among the dropped ones — fall back to
        # the first surviving chain. Telling the user beats crashing.
        log.warning(
            "default_chain=%r is empty; falling back to %r", default, next(iter(chains))
        )
        default = next(iter(chains))

    return ChainConfig(chains=chains, default=default, all_providers=ordered, skipped=skipped)


def load_providers(config_path: Path) -> list[Provider]:
    """Legacy entrypoint. Returns the flat provider list from the default chain."""
    cfg = load_config(config_path)
    return cfg.chains[cfg.default]


def _missing_env_vars(skipped: list[SkippedProvider]) -> list[tuple[str, list[str]]]:
    """Group skipped providers by env_var → list of provider names. Returns ordered
    by frequency (most-needed env-var first). Providers without env_var are skipped."""
    by_env: dict[str, list[str]] = {}
    for sp in skipped:
        if sp.reason != "no_api_key" or not sp.env_var:
            continue
        by_env.setdefault(sp.env_var, []).append(sp.name)
    return sorted(by_env.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _format_no_providers_error(skipped: list[SkippedProvider]) -> str:
    """Friendly error when every provider in config.yaml was skipped — usually
    means a fresh clone with no env vars set yet."""
    lines = [
        "No providers successfully built — every provider in config.yaml was skipped.",
        "",
        "  Reason: no API keys found in environment.",
        "  Fix: set at least one of these env vars in .env:",
    ]
    grouped = _missing_env_vars(skipped)
    if grouped:
        for env_var, providers in grouped[:8]:
            lines.append(f"    {env_var}  → enables {len(providers)} provider(s)")
        if len(grouped) > 8:
            lines.append(f"    ... and {len(grouped) - 8} more (see startup log)")
    else:
        lines.append("    GROQ_API_KEY (free, fastest — see docs/providers-setup.md#groq)")
        lines.append("    GEMINI_API_KEY, OPENROUTER_API_KEY, ...")
    lines.append("")
    lines.append("  See docs/providers-setup.md for how to get each key. Then restart.")
    return "\n".join(lines)


def _format_all_chains_empty_error(
    empty_chains: list[tuple[str, list[str]]], skipped: list[SkippedProvider]
) -> str:
    """Friendly error when *every* chain ended up empty — usually means no API
    keys are set at all. Tells the user the most-impactful env vars to set."""
    lines = [
        "Every chain has zero resolvable providers — server can't start.",
        "",
        "  Set at least one of these env vars in .env to bring chains back online:",
    ]
    grouped = _missing_env_vars(skipped)
    if grouped:
        for env_var, providers in grouped[:8]:
            lines.append(f"    {env_var}  → enables {len(providers)} provider(s)")
        if len(grouped) > 8:
            lines.append(f"    ... and {len(grouped) - 8} more (see startup log)")
    else:
        lines.append("    GROQ_API_KEY (free, fastest — see docs/providers-setup.md#groq)")
    lines.append("")
    lines.append("  Affected chains:")
    for chain_name, missing in empty_chains[:10]:
        lines.append(f"    {chain_name}: {len(missing)} member(s) skipped")
    lines.append("")
    lines.append("  See docs/providers-setup.md for how to get each key.")
    return "\n".join(lines)


def _format_chains_block(chains: dict[str, list[str]]) -> str:
    """Render `{chain_name: [provider_name, ...]}` as the indented YAML body
    that lives under `chains:`. Provider names are emitted as-is — they came
    from a config that already loaded, so quoting isn't needed in practice."""
    lines: list[str] = []
    for name, providers in chains.items():
        lines.append(f"  {name}:")
        for p in providers:
            lines.append(f"    - {p}")
    return "\n".join(lines) + "\n"


# Top-level YAML key on its own line (column 0). Used as the "next section"
# anchor so we replace exactly the chains block without eating trailing comments
# that belong to the next section.
_TOP_LEVEL_KEY = re.compile(r"(?m)^[A-Za-z_]\w*\s*:")


def rewrite_chains_in_yaml(
    config_path: Path | str,
    chains: dict[str, list[str]],
    default: str,
) -> Path:
    """Atomically replace the `chains:` block and `default_chain:` line in the
    YAML, preserving comments and the `providers:` section. Backs up the prior
    file to `<config>.bak`, writes a tempfile, then `os.replace`s into place.

    Returns the backup path so callers can surface it. Raises ValueError if the
    rewrite produces YAML that fails to round-trip back into a valid ChainConfig
    — in that case the original file is left untouched.
    """
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")

    # default_chain: single-line top-level key.
    new_default_line = f"default_chain: {default}\n"
    default_pat = re.compile(r"(?m)^default_chain\s*:.*\n?")
    if default_pat.search(text):
        text = default_pat.sub(new_default_line, text, count=1)
    else:
        text = text.rstrip() + "\n\n" + new_default_line

    # chains: multiline block — replace from the `chains:` line through the
    # last indented child, stopping at the next top-level key or EOF.
    new_block = "chains:\n" + _format_chains_block(chains) + "\n"
    chains_pat = re.compile(
        r"(?ms)^chains\s*:\s*\n(?:[ \t].*\n|\s*\n)*"
    )
    m = chains_pat.search(text)
    if m:
        text = text[: m.start()] + new_block + text[m.end() :]
    else:
        text = text.rstrip() + "\n\n" + new_block

    # Round-trip: parse the rewritten YAML back through load_config to surface
    # any breakage before we touch the on-disk file.
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    try:
        load_config(tmp_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    if config_path.exists():
        # Use replace so a concurrent reader sees either old or new, never partial.
        backup_path.write_bytes(config_path.read_bytes())
    os.replace(tmp_path, config_path)
    return backup_path
