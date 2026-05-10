from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING
from zoneinfo import ZoneInfo

import httpx

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)

# Gemini Free Tier daily quota resets at 00:00 Pacific Time (handles PDT/PST).
# We run 3 min after reset so fresh quota is guaranteed.
_PT = ZoneInfo("America/Los_Angeles")
_RUN_HOUR = 0
_RUN_MINUTE = 3

# Per-kind /v1/models discovery. We iterate through configured providers,
# dedupe by (kind, base_url), and fetch each unique endpoint once.
# - path: appended to the provider's base_url; absolute URL if it starts with "http".
# - auth: "bearer" → use the provider's api_key as Bearer; None → public.
# Kinds not in this map (cloudflare, *_image, *_translate, edge_tts, …) are skipped.
_KIND_MODELS_FETCH: dict[str, dict[str, Any]] = {
    "openrouter":   {"path": "/models", "auth": None},
    "openai":       {"path": "/models", "auth": "bearer"},
    "groq":         {"path": "/models", "auth": "bearer"},
    "cerebras":     {"path": "/models", "auth": "bearer"},
    "sambanova":    {"path": "/models", "auth": "bearer"},
    "nvidia":       {"path": "/models", "auth": "bearer"},
    "mistral":      {"path": "/models", "auth": "bearer"},
    "zai":          {"path": "/models", "auth": "bearer"},
    "github":       {"path": "https://models.github.ai/catalog/models", "auth": "bearer"},
    # Pollinations: base_url is text.pollinations.ai/v1 (chat endpoint).
    # The model listing lives one level up at /models — chat completions
    # interpret /v1/models as a prompt rather than serving the catalog.
    "pollinations": {"path": "https://text.pollinations.ai/models", "auth": None},
}
# Gemini lists models on a separate host with key in the query string.
_GEMINI_MODELS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=200"
)


def seconds_until_next_run(now: datetime | None = None) -> float:
    """Seconds from `now` until the next 00:03 Pacific Time. DST-safe via zoneinfo."""
    now = now or datetime.now(_PT)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_PT)
    target = now.replace(hour=_RUN_HOUR, minute=_RUN_MINUTE, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _known_providers_markdown(providers: list[Any]) -> str:
    rows = ["| provider:model | rpd | rpm | quality |", "|---|---|---|---|"]
    for p in providers:
        rows.append(
            f"| `{p.name}` "
            f"| {getattr(p, 'rpd', None) or '—'} "
            f"| {getattr(p, 'rpm', None) or '—'} "
            f"| {getattr(p, 'quality', None) or '—'} |"
        )
    return "\n".join(rows)


def _discover_model_endpoints(
    providers: list[Any],
) -> list[tuple[str, str, str | None]]:
    """Walk configured providers and produce a deduped list of (label, url, bearer_key)
    triples to fetch. Multiple providers sharing the same kind+base_url collapse
    to a single fetch."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str | None]] = []
    for p in providers:
        kind = getattr(p, "_kind", None)
        base_url = getattr(p, "_base_url", None)
        api_key = getattr(p, "_api_key", None)

        # OpenAI-compat family.
        if kind in _KIND_MODELS_FETCH and base_url:
            cfg = _KIND_MODELS_FETCH[kind]
            url = cfg["path"] if cfg["path"].startswith("http") else f"{base_url}{cfg['path']}"
            key = api_key if cfg["auth"] == "bearer" else None
            dedupe = (kind, url)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            out.append((kind, url, key))
            continue

        # Gemini — separate host, key in query string.
        if api_key and getattr(p, "_client", None) is not None and "genai" in type(p._client).__module__:
            url = _GEMINI_MODELS_URL.format(key=api_key)
            dedupe = ("gemini", url)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            out.append(("gemini", url, None))
    return out


async def _fetch_live_models(
    providers: list[Any], timeout: float = 15.0
) -> dict[str, Any]:
    """Fetch /v1/models from each unique provider endpoint in parallel.
    Returns {kind: parsed_json}. Failed fetches are skipped silently."""
    targets = _discover_model_endpoints(providers)

    async def fetch_one(
        client: httpx.AsyncClient, label: str, url: str, key: str | None
    ) -> tuple[str, Any]:
        try:
            headers = {"Authorization": f"Bearer {key}"} if key else None
            resp = await client.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return label, resp.json()
        except Exception as exc:
            log.warning("audit: failed to fetch %s: %s", label, str(exc)[:200])
            return label, None

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(fetch_one(client, lbl, url, key) for lbl, url, key in targets)
        )
    return {label: data for label, data in results if data is not None}


# Keywords that flag a post as "might be about a free LLM API". We over-match on
# purpose — Gemini re-filters in the prompt — but cut posts that mention none of
# these to save tokens.
_RELEVANCE_KEYWORDS = re.compile(
    r"\b(free|api|llm|gemini|claude|gpt|mistral|grok|qwen|llama|deepseek|nemotron|"
    r"cerebras|groq|sambanova|cloudflare|openrouter|nvidia nim|token|rate limit|rpd|rpm|"
    r"бесплат|квот|токен)\b",
    re.IGNORECASE,
)


def _is_relevant(*texts: str) -> bool:
    return any(_RELEVANCE_KEYWORDS.search(t or "") for t in texts)


async def _fetch_hn_news(timeout: float = 10.0, hours: int = 48) -> list[dict[str, Any]]:
    """HN Algolia search over multiple queries — narrow query 'free LLM API' returns
    0 hits in practice, so we run several broader queries and dedupe by objectID."""
    since = int(time.time()) - hours * 3600
    queries = ["free API", "LLM API", "new model", "OpenRouter"]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    base = (
        "https://hn.algolia.com/api/v1/search?"
        "tags=story&numericFilters=created_at_i>{since}&hitsPerPage=20&query={q}"
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for q in queries:
                resp = await client.get(base.format(since=since, q=q.replace(" ", "+")))
                resp.raise_for_status()
                for h in (resp.json().get("hits") or []):
                    oid = h.get("objectID") or ""
                    if oid in seen:
                        continue
                    title = h.get("title") or ""
                    if not _is_relevant(title):
                        continue
                    seen.add(oid)
                    out.append({
                        "title": title,
                        "url": h.get("url") or f"https://news.ycombinator.com/item?id={oid}",
                        "points": h.get("points") or 0,
                        "comments": h.get("num_comments") or 0,
                    })
    except Exception as exc:
        log.warning("audit: HN fetch failed: %s", str(exc)[:200])
    out.sort(key=lambda x: x["points"], reverse=True)
    return out[:15]


async def _fetch_reddit_news(timeout: float = 10.0) -> list[dict[str, Any]]:
    """Pull r/LocalLLaMA new posts. r/artificial blocks programmatic UAs → skipped."""
    subs = ["LocalLLaMA"]
    headers = {"User-Agent": "neurogate-audit/0.1 (github.com/neurogate)"}
    cutoff = time.time() - 48 * 3600
    out: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        for sub in subs:
            try:
                resp = await client.get(f"https://www.reddit.com/r/{sub}/new.json?limit=50")
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.warning("audit: reddit %s fetch failed: %s", sub, str(exc)[:200])
                continue
            for c in (data.get("data") or {}).get("children") or []:
                d = c.get("data") or {}
                created = d.get("created_utc") or 0
                if created < cutoff:
                    continue
                title = d.get("title") or ""
                body = (d.get("selftext") or "")[:300]
                if not _is_relevant(title, body):
                    continue
                out.append({
                    "sub": sub,
                    "title": title,
                    "body": body,
                    "url": f"https://reddit.com{d.get('permalink', '')}",
                    "score": d.get("score") or 0,
                })
    # sort by score, top 20
    out.sort(key=lambda x: x.get("score", 0), reverse=True)
    return out[:20]


def _extract_model_list(data: Any) -> list[dict[str, Any]]:
    """Best-effort extraction of model dicts from a /models response.
    Handles OpenAI-compat ({'data': [...]}), Gemini ({'models': [...]}),
    and bare-list responses."""
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("data"), list):
        return [m for m in data["data"] if isinstance(m, dict)]
    if isinstance(data.get("models"), list):
        # Gemini: name="models/gemini-2.0-flash" → flatten to id="gemini-2.0-flash"
        return [
            {"id": (m.get("name") or "").removeprefix("models/"), **m}
            for m in data["models"] if isinstance(m, dict)
        ]
    return []


def _summarize_live(live: dict[str, Any]) -> str:
    parts: list[str] = []
    for name, data in live.items():
        models = _extract_model_list(data)
        if not models:
            continue
        if name == "openrouter":
            free = [
                m for m in models
                if (m.get("id", "").endswith(":free")
                    or (m.get("pricing") or {}).get("prompt") == "0")
            ]
            sample = [
                f"- `{m.get('id')}` — ctx {m.get('context_length', '?')}"
                for m in free[:80]
            ]
            parts.append(f"### OpenRouter (free models, {len(free)} total)\n" + "\n".join(sample))
            continue
        ids = sorted({
            (m.get("id") or m.get("name") or m.get("model") or "").strip()
            for m in models
        } - {""})
        preview = "\n".join(f"- `{i}`" for i in ids[:60])
        extra = f"\n_+{len(ids) - 60} ещё_" if len(ids) > 60 else ""
        parts.append(f"### {name} ({len(ids)} моделей)\n{preview}{extra}")
    return "\n\n".join(parts) or "_нет данных_"


def _summarize_news(hn: list[dict[str, Any]], reddit: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if hn:
        lines = [f"- [{h['title']}]({h['url']}) · {h['points']}↑ {h['comments']}💬" for h in hn]
        parts.append("### Hacker News (last 48h, filtered)\n" + "\n".join(lines))
    if reddit:
        lines = []
        for r in reddit:
            snippet = f" — {r['body'][:180]}" if r.get("body") else ""
            lines.append(f"- r/{r['sub']}: [{r['title']}]({r['url']}) · {r['score']}↑{snippet}")
        parts.append("### Reddit (r/LocalLLaMA + r/artificial, last 48h, filtered)\n" + "\n".join(lines))
    return "\n\n".join(parts) or "_нет релевантных постов за 48ч_"


def _build_prompt(known_md: str, live_md: str, news_md: str) -> str:
    return f"""Ты аудитор бесплатных AI API. Твоя задача — раз в сутки находить новые бесплатные \
модели и провайдеров, которые ещё не подключены к моему прокси neurogate.

## Мои текущие провайдеры (уже подключены)

{known_md}

## Live-снапшот публичных каталогов моделей

{live_md}

## Свежие обсуждения за 48ч (HN + Reddit, prefiltered по ключевым словам)

{news_md}

## Задача

Сравни известные провайдеры с live-снапшотом **И** с постами в новостях. Выпиши только \
**бесплатные** новинки в следующем формате (markdown). Для каждой находки из новостей — \
ссылку на источник обязательно.

## Новые модели у существующих провайдеров
- **provider:model-id** — квота (если известна), почему интересно

## Новые бесплатные провайдеры
Из постов за 48ч + твоей памяти (с меткой какой источник). Формат:
- **Название** — URL, free tier условия, контекст/квота. [ref](url-из-новостей-если-есть)

## Изменения квот / depreciated
- Что поменялось у уже подключённых

## Рекомендации
1-3 строки — что стоит добавить в конфиг в первую очередь.

---

**Важные правила фильтрации:**
- Игнорируй посты про платные API (OpenAI, Anthropic и тд) — только FREE TIER.
- Игнорируй релизы weights-моделей без hosted API (HuggingFace drops, GGUF-кванты).
- Игнорируй drama / отзывы / мемы / «сравнения» — только анонсы API.
- Если пост не про новый API — не упоминай вообще.
- Если реально нового нет — ответь одной строкой «Нового нет».

Отчёт на русском, коротко, без воды. Не выдумывай — только то, что подтверждено источником \
из live-снапшота или постами выше. Дата сейчас: {datetime.now(_PT).strftime('%Y-%m-%d')}.
"""


def _strip_trailer(markdown: str) -> str:
    """Drop the '_Сгенерировано X_' footer so comparison / filtering isn't thrown off."""
    return markdown.rsplit("\n---\n", 1)[0] if "\n---\n" in markdown else markdown


def _has_findings(markdown: str) -> bool:
    """Heuristic: does this report actually contain new-item bullets?

    The prompt instructs Gemini to use `- **name** — …` for every finding, and
    `Если реально нового нет — так и напиши одной строкой` otherwise. So a body
    with zero bold-prefixed bullets is effectively empty.
    """
    for line in _strip_trailer(markdown).splitlines():
        if line.lstrip().startswith("- **"):
            return True
    return False


async def _notify_telegram(date: str, provider: str, markdown: str) -> None:
    """Send a Telegram push if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set
    and the report contains at least one new-item bullet. No-op otherwise."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    if not _has_findings(markdown):
        log.info("audit: telegram skipped for %s (no findings)", date)
        return
    # Telegram message limit is 4096 chars — keep headroom for headers.
    preview = _strip_trailer(markdown).strip()[:3500]
    dashboard = os.getenv("NEUROGATE_PUBLIC_URL", "").rstrip("/")
    link = f"\n\n🔗 {dashboard}/dashboard" if dashboard else ""
    text = f"🤖 *neurogate audit {date}* via `{provider}`\n\n{preview}{link}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
            )
            if resp.status_code >= 400:
                log.warning(
                    "telegram notify HTTP %d: %s", resp.status_code, resp.text[:200]
                )
                return
        log.info("audit: telegram notified for %s", date)
    except Exception as exc:
        log.warning("telegram notify failed: %s", exc)


_DEFAULT_AUDIT_CHAIN = "code"


async def run_audit(
    app: "FastAPI", *, chain_name: str = _DEFAULT_AUDIT_CHAIN
) -> tuple[str, str]:
    """Generate a daily audit report and save it. `chain_name` can be any chain
    name ('code', 'chat', 'quality', 'latency') or a specific provider name.
    Returns (date, provider_used)."""
    providers = app.state.providers
    known_md = _known_providers_markdown(providers)
    # Fetch all three sources in parallel — they're independent network calls.
    live, hn, reddit = await asyncio.gather(
        _fetch_live_models(providers),
        _fetch_hn_news(),
        _fetch_reddit_news(),
    )
    live_md = _summarize_live(live)
    news_md = _summarize_news(hn, reddit)
    log.info("audit: context sources — live:%d HN:%d reddit:%d", len(live), len(hn), len(reddit))
    prompt = _build_prompt(known_md, live_md, news_md)

    result, used_provider, used_chain = await app.state.router.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=3000,
        chain_name=chain_name,
    )

    date_str = datetime.now(_PT).strftime("%Y-%m-%d")
    md = (result.text or "").strip() + (
        f"\n\n---\n_Сгенерировано {used_provider} (chain: {used_chain}), "
        f"{datetime.now(_PT).strftime('%Y-%m-%d %H:%M %Z')}_"
    )
    app.state.rate_tracker.save_audit(date_str, md)
    log.info("audit: saved report for %s via %s", date_str, used_provider)
    await _notify_telegram(date_str, used_provider, md)
    return date_str, used_provider


async def audit_loop(app: "FastAPI") -> None:
    """Background task: sleep until 00:03 PT, run audit, repeat. Cancelled on shutdown."""
    log.info("audit: loop started, next run in %.0fs", seconds_until_next_run())
    while True:
        try:
            delay = seconds_until_next_run()
            await asyncio.sleep(delay)
            await run_audit(app)
        except asyncio.CancelledError:
            log.info("audit: loop cancelled")
            raise
        except Exception:
            log.exception("audit: run failed, sleeping 1h and retrying")
            await asyncio.sleep(3600)
