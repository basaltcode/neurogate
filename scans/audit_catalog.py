"""Stage 1 (catalog mode): poll vendor /v1/models, diff vs config.yaml,
emit new candidates as JSON ready for audit_verifier.py.

Никаких LLM в Stage 1 — берём то, что вендор реально публикует, и сравниваем
с тем, что у нас уже есть. Это убирает галлюцинации Stage 1 как класс.

Usage:
    uv run python scans/audit_catalog.py [--kinds groq,openrouter,...] [--free-only]
    # then:
    uv run python scans/audit_verifier.py scans/audit-YYYY-MM-DD-candidates.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv  # noqa: F401
    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # CI/Actions: env vars set directly, .env file not present

CONFIG_PATH = ROOT / "config.yaml"
SNAPSHOT_PATH = ROOT / "scans" / "catalog-snapshot.json"
BLOCKLIST_PATH = ROOT / "scans" / "paid_models_blocklist.yaml"


@dataclass(frozen=True)
class Vendor:
    kind: str
    base_url: str
    api_key_env: str
    auth_scheme: str = "Bearer"
    extra_headers: dict | None = None


# OpenAI-compat /v1/models. Gemini/Cohere/HF/Cloudflare/GitHub имеют
# нестандартные шейпы — добавим отдельно по необходимости.
VENDORS: list[Vendor] = [
    Vendor("groq",       "https://api.groq.com/openai/v1",       "GROQ_API_KEY"),
    Vendor("openrouter", "https://openrouter.ai/api/v1",         "OPENROUTER_API_KEY",
           extra_headers={"HTTP-Referer": "https://github.com/llmgate", "X-Title": "llmgate"}),
    Vendor("cerebras",   "https://api.cerebras.ai/v1",           "CEREBRAS_API_KEY"),
    Vendor("sambanova",  "https://api.sambanova.ai/v1",          "SAMBANOVA_API_KEY"),
    Vendor("nvidia",     "https://integrate.api.nvidia.com/v1",  "NVIDIA_API_KEY"),
    Vendor("mistral",    "https://api.mistral.ai/v1",            "MISTRAL_API_KEY"),
    Vendor("zai",        "https://api.z.ai/api/paas/v4",         "ZAI_API_KEY"),
]


def _load_existing_pairs() -> set[tuple[str, str]]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    pairs: set[tuple[str, str]] = set()
    for entry in cfg.get("providers", []) or []:
        if not isinstance(entry, dict):
            continue
        k, m = entry.get("kind"), entry.get("model")
        if k and m:
            pairs.add((k, m))
    return pairs


async def _fetch_models(vendor: Vendor) -> tuple[list[dict], str]:
    """Returns (model_entries, status_note). Each entry is the raw vendor dict
    (id + optional pricing/context_length/etc)."""
    key = os.getenv(vendor.api_key_env)
    if not key:
        return [], f"no {vendor.api_key_env}"
    headers = {
        "Authorization": f"{vendor.auth_scheme} {key}",
        "Accept": "application/json",
    }
    if vendor.extra_headers:
        headers.update(vendor.extra_headers)
    url = f"{vendor.base_url}/models"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return [], f"HTTP {resp.status_code}: {resp.text[:120]}"
        data = resp.json()
    except Exception as exc:
        return [], f"exception: {exc!r}"

    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return [], f"unexpected shape: {type(data).__name__}"
    entries = [it for it in items if isinstance(it, dict) and it.get("id")]
    return entries, f"ok ({len(entries)} models)"


def _load_blocklist() -> dict[str, list[str]]:
    if not BLOCKLIST_PATH.exists():
        return {}
    data = yaml.safe_load(BLOCKLIST_PATH.read_text()) or {}
    return {k: list(v or []) for k, v in data.items()}


def _is_free_for_us(kind: str, entry: dict, blocklist: dict[str, list[str]]) -> bool:
    """Returns True if model is safe to add to free-tier catalog candidates.

    Per-vendor logic:
    - openrouter: pricing.prompt and pricing.completion both == "0".
      Plus blocklist substrings (e.g. ":online", ":nitro" — these add cost).
    - mistral, zai: blocklist substrings (no public pricing endpoint).
    - groq, cerebras, sambanova: True. У них нет paid-only моделей,
      все доступны на free-tier с rate-limits.
    - nvidia: True для совместимости, **но** все NIM модели billable из единого
      $1000-starter-credit пула — это не free vs paid проблема, а balance-monitor.
      См. open question #9 в docs/features/auto-audit-and-add-models.md.
    """
    model_id = str(entry.get("id", ""))
    low = model_id.lower()

    for sub in blocklist.get(kind, []):
        if sub.lower() in low:
            return False

    if kind == "openrouter":
        pricing = entry.get("pricing") or {}
        prompt = str(pricing.get("prompt", "0"))
        completion = str(pricing.get("completion", "0"))
        try:
            return float(prompt) == 0.0 and float(completion) == 0.0
        except (ValueError, TypeError):
            return False

    return True


# Подстроки для отсева не-чат моделей. Каталоги возвращают всё подряд,
# а Stage 2 умеет только /chat/completions.
_NON_CHAT_SUBSTRINGS = (
    "whisper", "tts", "stt", "transcrib", "audio",
    "embed", "embedding", "rerank", "reranker",
    "guard", "moderation", "shield", "promptguard",
    "image", "flux", "stable-diffusion", "cosmos", "riva", "edify",
    "vision-only", "ocr", "speech",
    "/orpheus", "playai-",
    # nvidia-специфичные не-чат:
    "nemotron-parse", "retrieval-qa", "nv-embedqa", "nv-rerankqa",
    "esm", "evo2", "rfdiffusion", "alphafold", "parakeet", "canary",
    "video-",
)


def _looks_like_chat(model_id: str) -> bool:
    low = model_id.lower()
    return not any(sub in low for sub in _NON_CHAT_SUBSTRINGS)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", help="comma-separated subset",
                    default=",".join(v.kind for v in VENDORS))
    ap.add_argument("--free-only", action="store_true",
                    help="filter to free models: openrouter pricing.prompt/completion == 0, "
                         "plus paid_models_blocklist.yaml substrings for mistral/zai/openrouter")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "scans" / f"audit-{date.today()}-candidates.json")
    args = ap.parse_args()

    selected_kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    vendors = [v for v in VENDORS if v.kind in selected_kinds]

    existing = _load_existing_pairs()
    blocklist = _load_blocklist()
    print(f"existing (kind,model) pairs in config: {len(existing)}", file=sys.stderr)
    print(f"blocklist vendors: {sorted(blocklist.keys())}", file=sys.stderr)

    prev_snapshot: dict[str, list[str]] = {}
    if SNAPSHOT_PATH.exists():
        prev_snapshot = json.loads(SNAPSHOT_PATH.read_text())
        print(f"prev snapshot: {SNAPSHOT_PATH.name} "
              f"({sum(len(v) for v in prev_snapshot.values())} models across "
              f"{len(prev_snapshot)} vendors)", file=sys.stderr)
        bootstrap = False
    else:
        print("no prev snapshot → BOOTSTRAP mode (will save snapshot, emit empty candidates)",
              file=sys.stderr)
        bootstrap = True

    results = await asyncio.gather(*(_fetch_models(v) for v in vendors))

    new_candidates: list[dict] = []
    new_snapshot: dict[str, list[str]] = {}
    for vendor, (entries, note) in zip(vendors, results):
        if not entries:
            print(f"  {vendor.kind:11s} {note}", file=sys.stderr)
            if vendor.kind in prev_snapshot:
                new_snapshot[vendor.kind] = prev_snapshot[vendor.kind]
            continue
        ids = [str(e["id"]) for e in entries]
        new_snapshot[vendor.kind] = ids

        prev_ids = set(prev_snapshot.get(vendor.kind, []))
        skipped_existing = skipped_paid = skipped_nonchat = skipped_seen = 0
        added = 0
        for entry in entries:
            model_id = str(entry["id"])
            if (vendor.kind, model_id) in existing:
                skipped_existing += 1
                continue
            if not _looks_like_chat(model_id):
                skipped_nonchat += 1
                continue
            if args.free_only and not _is_free_for_us(vendor.kind, entry, blocklist):
                skipped_paid += 1
                continue
            if not bootstrap and model_id in prev_ids:
                skipped_seen += 1
                continue
            new_candidates.append({
                "name": f"{vendor.kind}:{model_id}",
                "kind": vendor.kind,
                "model": model_id,
                "api_key_env": vendor.api_key_env,
            })
            added += 1
        print(
            f"  {vendor.kind:11s} {note:38s} "
            f"new={added} skip_exist={skipped_existing} "
            f"skip_nonchat={skipped_nonchat} skip_paid={skipped_paid} "
            f"skip_seen={skipped_seen}",
            file=sys.stderr,
        )

    SNAPSHOT_PATH.write_text(json.dumps(new_snapshot, indent=2, sort_keys=True))
    print(f"\nsnapshot → {SNAPSHOT_PATH}", file=sys.stderr)

    if bootstrap:
        new_candidates = []
        print("BOOTSTRAP: candidates list emitted as empty. "
              "Re-run later — only delta vs this snapshot will be picked up.",
              file=sys.stderr)

    args.out.write_text(json.dumps(new_candidates, indent=2, ensure_ascii=False))
    print(f"wrote {len(new_candidates)} candidates → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
