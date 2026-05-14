"""Deep Search — multi-step research agent (plan → search → synthesize → critique → iterate).

Pipeline (MVP, без отдельного reader-step):
1. Planner раскладывает вопрос на N подзапросов (JSON list).
2. Searcher × N (parallel) — каждый подзапрос через цепочку `web` (gemini google_search
   или OpenRouter :online). Эти провайдеры уже делают поиск+summary сами.
3. Synthesizer собирает все findings + исходный вопрос в финальный ответ с
   numbered-цитатами [1][2] и секцией "Источники:".
4. Critic проверяет draft на пробелы → возвращает список недостающих тем или "ok".
5. Если критик нашёл пробелы → ещё 1 round (доп. search + re-synthesize). Hard cap 2.

URL-ы извлекаются regex-ом из ответов searcher-а и хранятся в `sources[]` для
финального списка. Отдельного fetch+read шага нет — это MVP; см. plans.md §11a
про будущий Brave/DDG бэкенд для получения сырых URL-ов.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import httpx

from neurogate.providers import ProviderCallResult

if TYPE_CHECKING:
    from neurogate.router import LLMRouter

log = logging.getLogger(__name__)

# URL-extraction regex — совпадает с http(s):// до первого whitespace или
# закрывающей скобки/кавычки. Trailing `.`, `,`, `;`, `)` стрипаем отдельно.
_URL_RX = re.compile(r"https?://[^\s)\]\"'<>]+", re.UNICODE)
_TRAILING_PUNCT = ".,;:!?)"


def _extract_urls(text: str, limit: int = 10) -> list[str]:
    """Вытаскивает уникальные URL-ы из свободного текста. Preserves order, dedupes."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _URL_RX.finditer(text):
        url = m.group(0).rstrip(_TRAILING_PUNCT)
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= limit:
            break
    return out


async def _fetch_jina_pages(
    urls: list[str],
    timeout_s: float = 15.0,
    max_chars_per_page: int = 6000,
) -> dict[str, str]:
    """Fetch clean markdown for each URL via r.jina.ai/<URL>. Free endpoint, no auth.

    Returns {url: markdown}. URLs that fail are silently omitted (caller falls
    back to the searcher-LLM's summary for that URL).
    """
    if not urls:
        return {}

    async def fetch_one(client: httpx.AsyncClient, url: str) -> tuple[str, str | None]:
        try:
            resp = await client.get(
                f"https://r.jina.ai/{url}",
                timeout=timeout_s,
                headers={"Accept": "text/markdown", "X-Return-Format": "markdown"},
                follow_redirects=True,
            )
            resp.raise_for_status()
        except Exception as exc:
            log.info("jina: fetch failed %s: %s", url, str(exc)[:120])
            return url, None
        body = (resp.text or "").strip()
        if not body:
            return url, None
        if len(body) > max_chars_per_page:
            body = body[:max_chars_per_page] + "\n…[обрезано Jina Reader]"
        return url, body

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(fetch_one(client, u) for u in urls),
            return_exceptions=False,
        )
    return {u: c for u, c in results if c}


async def _enrich_with_jina(
    findings: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    *,
    max_urls_per_finding: int = 3,
    max_urls_total: int = 12,
    timeout_s: float = 15.0,
) -> None:
    """Fetch full page content via Jina Reader for top URLs in each finding.

    Mutates findings in-place: adds f["full_pages"] = {url: markdown}.
    Bounded to avoid fan-out explosion on long URL lists.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for f in findings:
        if f.get("error"):
            continue
        for url in f.get("urls", [])[:max_urls_per_finding]:
            if url in seen:
                continue
            seen.add(url)
            candidates.append(url)
            if len(candidates) >= max_urls_total:
                break
        if len(candidates) >= max_urls_total:
            break

    started = time.monotonic()
    pages = await _fetch_jina_pages(candidates, timeout_s=timeout_s)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    for f in findings:
        f["full_pages"] = {
            url: pages[url] for url in f.get("urls", []) if url in pages
        }

    trace.append({
        "step": "jina_enrich",
        "latency_ms": elapsed_ms,
        "urls_tried": len(candidates),
        "urls_fetched": len(pages),
    })


def _parse_json_list(text: str) -> list[str] | None:
    """Parse a JSON array of strings from LLM output. Handles markdown-fenced blocks
    (```json ... ```). Returns None if parse fails — caller decides fallback."""
    if not text:
        return None
    s = text.strip()
    # Strip markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    # Try to find the first `[...]` in the string if the LLM added prose around it
    first_bracket = s.find("[")
    last_bracket = s.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        s = s[first_bracket : last_bracket + 1]
    try:
        parsed = json.loads(s)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    out = []
    for item in parsed:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out or None


async def _plan(
    router: "LLMRouter",
    user_query: str,
    planner_chain: str,
    trace: list[dict[str, Any]],
    max_subqs: int,
    request_extras: dict[str, Any] | None,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    prompt = (
        f"Пользователь задал исследовательский вопрос. Раздели его на {max_subqs} "
        f"конкретных подзапросов, которые вместе покроют тему и которые можно искать "
        f"в вебе независимо.\n\n"
        f"Вопрос пользователя:\n{user_query}\n\n"
        f"Правила:\n"
        f"- Максимум {max_subqs} подзапросов.\n"
        f"- Каждый подзапрос — полноценная фраза для поисковика на языке вопроса.\n"
        f"- Подзапросы должны быть достаточно разные, чтобы дополнять друг друга.\n"
        f"- Если вопрос простой и не требует декомпозиции — верни 1 элемент.\n\n"
        f"Верни ТОЛЬКО JSON-массив строк, без markdown-форматирования и комментариев. "
        f"Пример формата: [\"первый подзапрос\", \"второй подзапрос\"]"
    )
    started = time.monotonic()
    result, provider, _ = await router.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
        chain_name=planner_chain,
        request_extras=request_extras,
        exclude=exclude,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    subqs = _parse_json_list(result.text or "")
    if not subqs:
        log.warning(
            "deep_search: planner %s returned unparseable JSON, using raw query as single subq",
            provider,
        )
        subqs = [user_query]
    subqs = subqs[:max_subqs]

    trace.append({
        "step": "plan",
        "provider": provider,
        "latency_ms": elapsed_ms,
        "subquestions": subqs,
    })
    return subqs


async def _search_one(
    router: "LLMRouter",
    subq: str,
    searcher_chain: str,
    request_extras: dict[str, Any] | None,
    timeout_s: float,
    exclude: Iterable[str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result, provider, _ = await asyncio.wait_for(
            router.chat(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Найди в вебе актуальную информацию по теме:\n{subq}\n\n"
                            f"Дай развёрнутый фактический ответ с конкретными числами/датами/"
                            f"именами где уместно. Обязательно укажи URL-источники в тексте "
                            f"(как обычные ссылки, например: 'согласно TechCrunch "
                            f"(https://example.com), ...'). Если доступ к вебу недоступен — "
                            f"ответь на основе собственных знаний и явно отметь это."
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=1500,
                chain_name=searcher_chain,
                request_extras=request_extras,
                exclude=exclude,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        return {
            "subq": subq,
            "provider": None,
            "text": "",
            "urls": [],
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": "timeout",
        }
    except Exception as exc:
        return {
            "subq": subq,
            "provider": None,
            "text": "",
            "urls": [],
            "latency_ms": int((time.monotonic() - started) * 1000),
            "error": str(exc)[:300],
        }
    elapsed_ms = int((time.monotonic() - started) * 1000)
    text = result.text or ""
    return {
        "subq": subq,
        "provider": provider,
        "text": text,
        "urls": _extract_urls(text, limit=5),
        "latency_ms": elapsed_ms,
        "error": None,
    }


async def _parallel_search(
    router: "LLMRouter",
    subqs: list[str],
    searcher_chain: str,
    trace: list[dict[str, Any]],
    request_extras: dict[str, Any] | None,
    timeout_s: float,
    exclude: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    findings = await asyncio.gather(
        *(_search_one(router, q, searcher_chain, request_extras, timeout_s, exclude=exclude) for q in subqs)
    )
    for f in findings:
        trace.append({
            "step": "search",
            "subq": f["subq"],
            "provider": f["provider"],
            "latency_ms": f["latency_ms"],
            "urls_found": len(f["urls"]),
            "chars": len(f["text"]),
            "error": f["error"],
        })
    return findings


def _build_sources(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe URLs across all findings, assign sequential ids for [N] citations."""
    sources: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for f in findings:
        for url in f["urls"]:
            if url in seen:
                continue
            seen[url] = len(sources) + 1
            sources.append({
                "id": len(sources) + 1,
                "url": url,
                "subq": f["subq"],
            })
    return sources


def _build_synthesis_messages(
    user_query: str,
    findings: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    max_finding_chars: int = 4000,
) -> list[dict[str, Any]]:
    findings_blocks: list[str] = []
    for i, f in enumerate(findings, 1):
        if f["error"]:
            continue
        summary = f["text"]
        if len(summary) > max_finding_chars:
            summary = summary[:max_finding_chars] + "\n…[обрезано]"
        header = f"### Подзапрос {i}: {f['subq']}\n(источник-модель: {f['provider']})"

        full_pages: dict[str, str] = f.get("full_pages") or {}
        if full_pages:
            pages_blocks = [
                f"**Полный текст {url}:**\n{content}"
                for url, content in full_pages.items()
            ]
            findings_blocks.append(
                f"{header}\n[обогащено Jina Reader: {len(full_pages)} стр.]\n\n"
                f"**Summary от searcher:**\n{summary}\n\n"
                + "\n\n".join(pages_blocks)
            )
        else:
            findings_blocks.append(f"{header}\n\n{summary}")
    sources_block = "\n".join(
        f"[{s['id']}] {s['url']}" for s in sources
    ) if sources else "(URL-источники не найдены в ответах searcher-а)"

    system = (
        "Ты — аналитик-синтезатор deep-research агента. Тебе дан вопрос пользователя "
        "и результаты поиска по нескольким подзапросам. Твоя задача — собрать "
        "финальный структурированный ответ.\n\n"
        "Правила:\n"
        "1. Пиши в markdown. Можно использовать заголовки (##), списки, таблицы.\n"
        "2. Каждое фактическое утверждение подтверждай numbered-цитатой вида [1], "
        "[2] — номера берутся из списка источников ниже. Если источник без URL — "
        "не цитируй.\n"
        "3. Не копируй дословно findings — синтезируй, сопоставляй, отмечай "
        "противоречия между источниками если они есть.\n"
        "4. В конце ответа обязательная секция `## Источники` со списком всех "
        "использованных `[N]` с URL-ами.\n"
        "5. Ответь на языке вопроса пользователя.\n"
        "6. Если findings недостаточны для уверенного ответа — честно скажи что "
        "неясно / требует доп. поиска. Не фантазируй."
    )
    user = (
        f"Вопрос пользователя:\n{user_query}\n\n"
        f"Результаты поиска:\n\n"
        + "\n\n".join(findings_blocks)
        + f"\n\nСписок найденных URL-источников (используй эти номера для [N]-цитат):\n"
        + sources_block
        + "\n\nДай финальный ответ с цитатами."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def _synthesize(
    router: "LLMRouter",
    user_query: str,
    findings: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    synthesizer_chain: str,
    trace: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float | None,
    request_extras: dict[str, Any] | None,
    iteration: int,
    exclude: Iterable[str] | None = None,
) -> tuple[ProviderCallResult, str]:
    messages = _build_synthesis_messages(user_query, findings, sources)
    agg_max_tokens = max(max_tokens or 0, 2500)
    started = time.monotonic()
    result, provider, _ = await router.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=agg_max_tokens,
        chain_name=synthesizer_chain,
        request_extras=request_extras,
        exclude=exclude,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    trace.append({
        "step": "synthesize",
        "provider": provider,
        "latency_ms": elapsed_ms,
        "iteration": iteration,
        "draft_chars": len(result.text or ""),
    })
    return result, provider


async def _critique(
    router: "LLMRouter",
    user_query: str,
    draft: str,
    critic_chain: str,
    trace: list[dict[str, Any]],
    request_extras: dict[str, Any] | None,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Ask critic to identify gaps in draft. Returns list of additional search queries
    to address the gaps, or [] if draft is complete."""
    prompt = (
        f"Ты — критик research-ответа. Тебе даны вопрос пользователя и первый "
        f"драфт ответа. Найди критичные пробелы: факты которые не освещены, "
        f"заявления без обоснования, противоречия, устаревшая информация.\n\n"
        f"Вопрос пользователя:\n{user_query}\n\n"
        f"Драфт ответа:\n{draft}\n\n"
        f"Если драфт полный и качественный — верни пустой JSON-массив [].\n"
        f"Если есть пробелы — верни JSON-массив из 1-3 дополнительных поисковых "
        f"запросов, которые заполнят эти пробелы. Пример: "
        f'["что известно о X после 2025", "статистика Y за последний год"].\n\n'
        f"Верни ТОЛЬКО JSON-массив, без markdown и комментариев."
    )
    started = time.monotonic()
    result, provider, _ = await router.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
        chain_name=critic_chain,
        request_extras=request_extras,
        exclude=exclude,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    gaps = _parse_json_list(result.text or "") or []
    gaps = gaps[:3]  # hard cap — избегаем ballooning
    trace.append({
        "step": "critique",
        "provider": provider,
        "latency_ms": elapsed_ms,
        "gaps_found": len(gaps),
        "gaps": gaps,
    })
    return gaps


async def run_deep_search(
    router: "LLMRouter",
    user_query: str,
    *,
    planner_chain: str = "reasoning_quality",
    searcher_chain: str = "web",
    synthesizer_chain: str = "reasoning_quality",
    critic_chain: str = "reasoning_quality",
    max_subquestions: int = 4,
    max_critic_rounds: int = 1,
    search_timeout_s: float = 90.0,
    temperature: float | None = None,
    max_tokens: int | None = None,
    request_extras: dict[str, Any] | None = None,
    jina_enabled: bool = True,
    jina_timeout_s: float = 15.0,
    jina_max_urls: int = 12,
    exclude: Iterable[str] | None = None,
) -> tuple[ProviderCallResult, str, dict[str, Any]]:
    """Полный pipeline deep_search. Возвращает (result, synthesizer_provider, metadata).

    metadata:
        subquestions: list[str] — из Plan-шага
        sources: list[{id, url, subq}] — все уникальные URL из findings
        trace: list[dict] — каждый шаг с провайдером и latency
        iterations: int — сколько раз прошли synthesize (1 без critic-раунда, 2 с)
    """
    trace: list[dict[str, Any]] = []

    # Step 1: Plan
    subqs = await _plan(
        router, user_query, planner_chain, trace,
        max_subqs=max_subquestions,
        request_extras=request_extras,
        exclude=exclude,
    )

    # Step 2: Parallel Search
    findings = await _parallel_search(
        router, subqs, searcher_chain, trace,
        request_extras=request_extras,
        timeout_s=search_timeout_s,
        exclude=exclude,
    )

    # Minimum viable findings check
    successful_findings = [f for f in findings if f["error"] is None and f["text"]]
    if not successful_findings:
        raise RuntimeError(
            f"deep_search: all {len(findings)} searches failed (no findings to synthesize)"
        )

    # Step 2.5: Enrich findings with full page content via Jina Reader
    # (free endpoint, r.jina.ai/<URL> → clean markdown). Failures are silent —
    # synthesizer falls back to searcher-LLM summary for missing pages.
    if jina_enabled:
        await _enrich_with_jina(
            findings, trace,
            max_urls_total=jina_max_urls,
            timeout_s=jina_timeout_s,
        )

    # Step 3: Synthesize draft
    sources = _build_sources(findings)
    draft_result, synth_provider = await _synthesize(
        router, user_query, findings, sources, synthesizer_chain, trace,
        max_tokens=max_tokens, temperature=temperature,
        request_extras=request_extras, iteration=1,
        exclude=exclude,
    )
    iterations = 1

    # Step 4-5: Critic loop (hard cap max_critic_rounds additional rounds)
    for round_n in range(1, max_critic_rounds + 1):
        gaps = await _critique(
            router, user_query, draft_result.text or "", critic_chain, trace,
            request_extras=request_extras,
            exclude=exclude,
        )
        if not gaps:
            break
        # Additional search round on the gaps
        extra_findings = await _parallel_search(
            router, gaps, searcher_chain, trace,
            request_extras=request_extras,
            timeout_s=search_timeout_s,
            exclude=exclude,
        )
        if jina_enabled:
            await _enrich_with_jina(
                extra_findings, trace,
                max_urls_total=jina_max_urls,
                timeout_s=jina_timeout_s,
            )
        findings.extend(extra_findings)
        sources = _build_sources(findings)
        # Re-synthesize with all findings combined
        draft_result, synth_provider = await _synthesize(
            router, user_query, findings, sources, synthesizer_chain, trace,
            max_tokens=max_tokens, temperature=temperature,
            request_extras=request_extras, iteration=round_n + 1,
            exclude=exclude,
        )
        iterations += 1

    metadata: dict[str, Any] = {
        "subquestions": subqs,
        "sources": sources,
        "trace": trace,
        "iterations": iterations,
        "planner_chain": planner_chain,
        "searcher_chain": searcher_chain,
        "synthesizer_chain": synthesizer_chain,
        "critic_chain": critic_chain,
        "jina_enabled": jina_enabled,
    }
    return draft_result, synth_provider, metadata
