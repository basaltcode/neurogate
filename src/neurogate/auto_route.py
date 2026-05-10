"""Heuristic intent classifier for `model: "auto"`.

Picks one of the existing fallback chains based on regex/marker matches over
the last user message. No network calls, no LLM. If nothing matches or the
proposed chain isn't configured, falls back to the router's default chain.

Excluded by design (heuristic must NEVER pick these — explicit opt-in only):
- `paid` — Opus / paid-tier models, must require explicit consent.
- `moa` / `sc` / `deep_search` / `debate` — heavy ensembles that burn provider quota.
- `translation*` / `audio` / `tts` / `image_gen` / `embed*` / `rerank` /
  `moderation*` — non-chat endpoints that wouldn't make sense for a
  /v1/chat/completions request.
"""
from __future__ import annotations

import re
from typing import Any

EXCLUDED: frozenset[str] = frozenset({
    "paid",
    "moa", "sc", "deep_search", "debate",
    "translation", "translate_adaptive",
    "audio", "tts",
    "image_gen",
    "embed", "embed_code",
    "rerank",
    "moderation", "moderation_image", "moderation_jailbreak", "moderation_ru",
})

_CODE_RE = re.compile(
    r"```|"
    r"\b(?:def|function|class|import|return|const|let|var|async|await|public|private)\s|"
    r"console\.log|print\(|"
    r"\bTraceback\b|\bSegmentation fault\b|"
    r"\bError:|Exception:|"
    r"\.(?:py|ts|tsx|js|jsx|go|rs|java|cpp|cs|rb|php|swift|kt|sql|sh)\b|"
    r"\b(?:npm|pip|cargo|go mod|yarn|pnpm)\s|"
    r"напиши\s+(?:функц|код|метод|класс|скрипт|программ)|"
    r"(?:fix|debug|refactor|implement|review)\s+(?:this|my|the)\s",
    re.IGNORECASE,
)

_WEB_RE = re.compile(
    r"https?://"
    # full English words (both word boundaries)
    r"|\b(?:search\s+the\s+web|google|news|today|breaking|"
    r"what'?s\s+(?:new|the\s+latest)|what\s+happened)\b"
    # English/Russian stems — left boundary only, optional suffix.
    # `\b` on the right is unreliable across cyrillic suffixes (новости, погода).
    r"|\b(?:найди|поищи|загугли|новост|сегодн|погод)\w*"
    r"|\bкурс\s+(?:валют|доллар|евро|биткоин|btc|eth)",
    re.IGNORECASE,
)

_REASONING_RE = re.compile(
    r"\b(?:докажи|обоснуй|"
    r"step\s+by\s+step|reason\s+through|chain\s+of\s+thought|think\s+step|"
    r"прикинь\s+по\s+шагам|поэтапно|пошагово|"
    r"reason\s+carefully|let'?s\s+think)\b",
    re.IGNORECASE,
)


def _has_image_part(messages: list[dict[str, Any]]) -> bool:
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(c.get("text", ""))
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            return "\n".join(parts)
        return str(content or "")
    return ""


def classify_intent(
    messages: list[dict[str, Any]],
    available_chains: set[str],
    default: str,
) -> tuple[str, str]:
    """Return (chain_name, reason). The chain is guaranteed to be present in
    `available_chains` and never one of EXCLUDED. Falls back to `default` if
    no rule matched or the matched chain is unavailable."""

    def _ok(name: str) -> bool:
        return name in available_chains and name not in EXCLUDED

    if _has_image_part(messages) and _ok("image"):
        return "image", "image_url-in-content"

    text = _last_user_text(messages)

    if _WEB_RE.search(text) and _ok("web"):
        return "web", "web-keyword-or-url"

    if _CODE_RE.search(text) and _ok("code"):
        return "code", "code-marker"

    if len(text) > 800 and _REASONING_RE.search(text) and _ok("reasoning_quality"):
        return "reasoning_quality", "long-reasoning-prompt"

    return default, "fallback-default"
