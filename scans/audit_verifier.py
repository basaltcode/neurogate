"""Stage 2 verifier: dedup + smoke-test + identity-probe.

Три последовательных gate:
1. dedup     — кандидат уже есть в config.yaml.
2. smoke     — реальный POST с max_tokens=10. HTTP != 200 → rejected.
3. identity  — короткий «what model are you?» промпт. Если major-version в ответе
               не соответствует claimed — rejected_identity. Это ловит
               silent fallback (NVIDIA NIM возвращал HTTP 200 на выдуманный
               deepseek-v4-flash и роутил на DeepSeek-V2.1 — 2026-04-29).

LLM-консенсус не используется — все три gate detеrministic.

Usage:
    uv run python scans/audit_verifier.py scans/audit-YYYY-MM-DD-candidates.json
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # CI/Actions: env vars set directly, .env file not present

from neurogate.config import _build_provider  # noqa: E402

CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class Verdict:
    name: str
    kind: str
    model: str
    # confirmed | confirmed_unverified | rejected_dedup | rejected_http
    # | rejected_identity | rejected_no_key | error
    status: str
    http_code: int | None
    detail: str
    identity_reply: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "status": self.status,
            "http_code": self.http_code,
            "detail": self.detail,
            "identity_reply": self.identity_reply,
        }


def _load_existing() -> tuple[set[str], set[tuple[str, str]]]:
    """Returns (existing names, existing (kind, model) pairs)."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    names: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for entry in cfg.get("providers", []) or []:
        if not isinstance(entry, dict):
            continue
        n = entry.get("name")
        k = entry.get("kind")
        m = entry.get("model")
        if n:
            names.add(n)
        if k and m:
            pairs.add((k, m))
    return names, pairs


_HTTP_RE = re.compile(r"HTTP (\d{3})")

# Известные family-токены для identity-probe. Если в claimed `name` встречается
# одно из них, ожидаем то же слово в self-report модели. Список можно расширять.
_FAMILIES = (
    "deepseek", "grok", "gemma", "glm", "qwen", "llama", "gpt", "mistral",
    "claude", "gemini", "phi", "ling", "lfm", "liquid", "nemotron", "kimi",
    "minimax", "hunyuan", "yi", "sonnet", "opus", "haiku", "scout", "maverick",
    "o1", "o3", "o4", "command", "cohere",
)

_VER_RE = re.compile(r"\d+(?:\.\d+)?")


def _extract_http_code(exc: BaseException) -> int | None:
    msg = str(exc) or repr(exc)
    m = _HTTP_RE.search(msg)
    return int(m.group(1)) if m else None


def _claimed_family_version(name: str) -> tuple[str | None, str | None]:
    slug = name.split(":", 1)[1] if ":" in name else name
    slug = slug.lower()
    fam = next((f for f in _FAMILIES if f in slug), None)
    m = _VER_RE.search(slug)
    return fam, (m.group(0) if m else None)


def _identity_match(name: str, reply: str) -> str:
    """Returns: 'match' | 'mismatch' | 'ambiguous'."""
    fam, ver = _claimed_family_version(name)
    r = reply.lower()
    if not fam or fam not in r or ver is None:
        return "ambiguous"
    reply_majors = {v.split(".")[0] for v in _VER_RE.findall(r)}
    if not reply_majors:
        return "ambiguous"
    return "match" if ver.split(".")[0] in reply_majors else "mismatch"


def _build(entry: dict):
    try:
        # generous timeout — NVIDIA NIM может молотить >60s даже на max_tokens=10
        return _build_provider({**entry, "rpm": 999, "rpd": 999, "timeout": 240}), None
    except Exception as exc:
        return None, f"build_provider failed: {exc!r}"


async def _call(provider, prompt: str, max_tokens: int) -> tuple[int | None, str]:
    try:
        result = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        msg = str(exc) or repr(exc)
        return _extract_http_code(exc), msg[:400]
    return 200, (getattr(result, "text", "") or "").strip()


_IDENTITY_PROMPT = (
    "What is your exact model name and version? "
    "Answer in one short sentence. Do not roleplay or pretend."
)


async def verify_one(
    entry: dict, names: set[str], pairs: set[tuple[str, str]]
) -> Verdict:
    name = entry["name"]
    kind = entry["kind"]
    model = entry["model"]

    if name in names or (kind, model) in pairs:
        return Verdict(name, kind, model, "rejected_dedup", None,
                       "already in config.yaml")

    if not os.getenv(entry.get("api_key_env", "")):
        return Verdict(name, kind, model, "rejected_no_key", None,
                       f"missing env {entry.get('api_key_env')}")

    provider, build_err = _build(entry)
    if provider is None:
        return Verdict(name, kind, model, "error", None, build_err or "build failed")

    # Gate 2: smoke
    smoke_code, smoke_detail = await _call(provider, "hi", max_tokens=10)
    if smoke_code != 200:
        if smoke_code is None:
            return Verdict(name, kind, model, "error", None, smoke_detail)
        return Verdict(name, kind, model, "rejected_http", smoke_code, smoke_detail)

    # Gate 3: identity probe
    id_code, id_text = await _call(provider, _IDENTITY_PROMPT, max_tokens=120)
    if id_code != 200:
        # smoke прошёл, identity упал — странно, помечаем как unverified
        return Verdict(name, kind, model, "confirmed_unverified", 200,
                       f"smoke ok; identity call failed: {id_text[:120]}",
                       identity_reply="")

    match = _identity_match(name, id_text)
    if match == "mismatch":
        return Verdict(name, kind, model, "rejected_identity", 200,
                       f"identity mismatch (claimed vs reported)",
                       identity_reply=id_text[:300])
    if match == "ambiguous":
        return Verdict(name, kind, model, "confirmed_unverified", 200,
                       "identity ambiguous — manual review",
                       identity_reply=id_text[:300])
    return Verdict(name, kind, model, "confirmed", 200,
                   f"smoke ok + identity match",
                   identity_reply=id_text[:300])


async def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    candidates_path = Path(sys.argv[1])
    candidates = json.loads(candidates_path.read_text())

    names, pairs = _load_existing()
    print(f"loaded {len(names)} existing names, {len(pairs)} (kind,model) pairs",
          file=sys.stderr)

    verdicts = await asyncio.gather(
        *(verify_one(c, names, pairs) for c in candidates)
    )

    out = [v.to_dict() for v in verdicts]
    out_path = candidates_path.with_name(
        candidates_path.stem.replace("-candidates", "-verified") + ".json"
    )
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1

    print(f"\nResults → {out_path}")
    for st in ("confirmed", "confirmed_unverified", "rejected_dedup",
               "rejected_identity", "rejected_http", "rejected_no_key", "error"):
        if st in counts:
            print(f"  {st:22s} {counts[st]}")
    print()
    for v in verdicts:
        line = f"  [{v.status:20s}] {v.name:55s} {v.http_code or '-':>3} {v.detail[:80]}"
        print(line)
        if v.identity_reply:
            print(f"    identity: {v.identity_reply[:140]!r}")


if __name__ == "__main__":
    asyncio.run(main())
