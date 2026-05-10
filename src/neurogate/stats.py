from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

# When each provider's daily quota actually rolls over. Derived from provider docs
# and verified via `x-ratelimit-reset-*` headers where available — see apicount.md.
#   utc_midnight — hard reset at 00:00 UTC
#   pt_midnight  — hard reset at 00:00 America/Los_Angeles (handles PDT/PST)
#   rolling_24h  — first request in the window ages out 24h after it happened
#   monthly      — 1st of next month UTC (NVIDIA credits)
#   none         — no daily cap at all
_KIND_RESET_POLICY = {
    "gemini": "pt_midnight",
    "groq": "utc_midnight",
    "github": "utc_midnight",
    "openrouter": "utc_midnight",
    "cerebras": "utc_midnight",
    "sambanova": "rolling_24h",
    "cloudflare": "rolling_24h",
    "mistral": "none",
    "nvidia": "monthly",
    "zai": "none",
    "openai": "utc_midnight",
}


def _kind_from_name(provider: str) -> str:
    return provider.split(":", 1)[0] if ":" in provider else provider


def _next_utc_midnight(now: int) -> int:
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    nxt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(nxt.timestamp())


def _next_pt_midnight(now: int) -> int:
    pt = datetime.fromtimestamp(now, tz=ZoneInfo("America/Los_Angeles"))
    nxt = (pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(nxt.timestamp())


def _next_utc_month(now: int) -> int:
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    year, month = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
    return int(
        datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    )


class RateTracker:
    """Per-provider request log that survives restarts.

    Stores a row per attempt in `rate_events(provider, ts, kind)` and answers
    "is provider X still under its RPM/RPD cap right now?" via rolling-window
    counts (last 60s / last 24h). Rolling windows are slightly pessimistic vs
    Pacific-midnight resets, but safe: worst case we skip for an extra hour
    after reset and fall back to the next link in the chain.
    """

    _WINDOW_RPM = 60
    _WINDOW_RPD = 86400
    _RETAIN = 25 * 3600
    # call_events keeps per-call audit trail for the dashboard "Вызовы" tab.
    # Larger window than rate_events because operators want to see what was
    # called over the last few days, not just the rolling RPD bucket.
    _CALL_RETAIN = 7 * 86400

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS rate_events ("
            "provider TEXT NOT NULL, "
            "ts INTEGER NOT NULL, "
            "kind TEXT NOT NULL DEFAULT 'request')"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rate_events_provider_ts "
            "ON rate_events(provider, ts)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS audit_reports ("
            "date TEXT PRIMARY KEY, "
            "markdown TEXT NOT NULL, "
            "created_at INTEGER NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS call_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts INTEGER NOT NULL, "
            "endpoint TEXT NOT NULL, "
            "client_model TEXT, "
            "chain_requested TEXT, "
            "chain_resolved TEXT, "
            "provider TEXT, "
            "outcome TEXT NOT NULL, "
            "duration_ms INTEGER, "
            "prompt_tokens INTEGER, "
            "completion_tokens INTEGER, "
            "total_tokens INTEGER, "
            "cached_tokens INTEGER, "
            "stream INTEGER NOT NULL DEFAULT 0, "
            "error_type TEXT, "
            "error_msg TEXT, "
            "extra TEXT)"
        )
        # Add `client` to existing DBs (SQLite has no IF NOT EXISTS for ALTER).
        cur = self._conn.execute("PRAGMA table_info(call_events)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "client" not in existing_cols:
            self._conn.execute("ALTER TABLE call_events ADD COLUMN client TEXT")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_events_ts ON call_events(ts DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_events_client_ts ON call_events(client, ts DESC)"
        )
        self._conn.execute(
            "DELETE FROM rate_events WHERE ts < ?",
            (int(time.time()) - self._RETAIN,),
        )
        self._conn.execute(
            "DELETE FROM call_events WHERE ts < ?",
            (int(time.time()) - self._CALL_RETAIN,),
        )
        self._writes_since_prune = 0
        self._call_writes_since_prune = 0
        # In-memory cooldowns set after a hard rate-limit (429 / quota-exhausted).
        # Provider stays unavailable until ts. On process restart we retry once and
        # re-cooldown if the upstream is still rate-limited — acceptable.
        self._cooldowns: dict[str, int] = {}

    def set_cooldown(self, provider: str, *, until_ts: int) -> None:
        prev = self._cooldowns.get(provider)
        if prev is None or until_ts > prev:
            self._cooldowns[provider] = until_ts

    def cooldown_until(self, provider: str) -> int | None:
        ts = self._cooldowns.get(provider)
        if ts is None:
            return None
        if ts <= int(time.time()):
            self._cooldowns.pop(provider, None)
            return None
        return ts

    def record(self, provider: str, kind: str = "request") -> None:
        self._conn.execute(
            "INSERT INTO rate_events(provider, ts, kind) VALUES(?, ?, ?)",
            (provider, int(time.time()), kind),
        )
        self._writes_since_prune += 1
        if self._writes_since_prune >= 500:
            self._conn.execute(
                "DELETE FROM rate_events WHERE ts < ?",
                (int(time.time()) - self._RETAIN,),
            )
            self._writes_since_prune = 0

    def is_available(
        self, provider: str, *, rpd: int | None, rpm: int | None,
        ignore_cooldown: bool = False,
    ) -> bool:
        if not ignore_cooldown and self.cooldown_until(provider) is not None:
            return False
        if not rpd and not rpm:
            return True
        now = int(time.time())
        if rpm:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM rate_events WHERE provider = ? AND ts > ?",
                (provider, now - self._WINDOW_RPM),
            )
            if cur.fetchone()[0] >= rpm:
                return False
        if rpd:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM rate_events WHERE provider = ? AND ts > ?",
                (provider, now - self._WINDOW_RPD),
            )
            if cur.fetchone()[0] >= rpd:
                return False
        return True

    def usage(self, provider: str) -> dict[str, int]:
        now = int(time.time())
        cur = self._conn.execute(
            "SELECT "
            "COALESCE(SUM(CASE WHEN ts > ? THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN ts > ? THEN 1 ELSE 0 END), 0) "
            "FROM rate_events WHERE provider = ?",
            (now - self._WINDOW_RPM, now - self._WINDOW_RPD, provider),
        )
        row = cur.fetchone()
        return {"last_1m": int(row[0]), "last_24h": int(row[1])}

    def reset_info(
        self, provider: str, *, rpd: int | None, rpm: int | None
    ) -> dict[str, int | str | None]:
        """When do this provider's RPM/RPD windows next free up a slot?

        Returns unix timestamps (seconds). RPM is always rolling-60s across all
        providers; RPD depends on the provider kind's documented reset policy.
        """
        kind = _kind_from_name(provider)
        policy = _KIND_RESET_POLICY.get(kind, "utc_midnight")
        now = int(time.time())
        rpm_reset_at: int | None = None
        rpd_reset_at: int | None = None

        if rpm:
            cur = self._conn.execute(
                "SELECT MIN(ts) FROM rate_events WHERE provider = ? AND ts > ?",
                (provider, now - self._WINDOW_RPM),
            )
            oldest = cur.fetchone()[0]
            if oldest is not None:
                rpm_reset_at = int(oldest) + self._WINDOW_RPM

        if rpd:
            if policy == "utc_midnight":
                rpd_reset_at = _next_utc_midnight(now)
            elif policy == "pt_midnight":
                rpd_reset_at = _next_pt_midnight(now)
            elif policy == "monthly":
                rpd_reset_at = _next_utc_month(now)
            elif policy == "rolling_24h":
                cur = self._conn.execute(
                    "SELECT MIN(ts) FROM rate_events WHERE provider = ? AND ts > ?",
                    (provider, now - self._WINDOW_RPD),
                )
                oldest = cur.fetchone()[0]
                if oldest is not None:
                    rpd_reset_at = int(oldest) + self._WINDOW_RPD

        return {
            "policy": policy,
            "rpm_reset_at": rpm_reset_at,
            "rpd_reset_at": rpd_reset_at,
        }

    def record_call(
        self,
        *,
        endpoint: str,
        outcome: str,
        client_model: str | None = None,
        chain_requested: str | None = None,
        chain_resolved: str | None = None,
        provider: str | None = None,
        duration_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        cached_tokens: int | None = None,
        stream: bool = False,
        error_type: str | None = None,
        error_msg: str | None = None,
        extra: dict[str, Any] | None = None,
        client: str | None = None,
    ) -> None:
        """Record a single user-facing call to call_events.

        Distinct from `record()` (which logs every individual provider attempt
        for rate-limit accounting): this captures one row per inbound API call,
        so the dashboard can show "client asked X, chain Y resolved, provider Z
        answered, took Nms, Mtok in / Ktok out". Best-effort — never raises.
        """
        try:
            extra_json = json.dumps(extra, ensure_ascii=False) if extra else None
            if error_msg and len(error_msg) > 500:
                error_msg = error_msg[:500]
            self._conn.execute(
                "INSERT INTO call_events("
                "ts, endpoint, client_model, chain_requested, chain_resolved, "
                "provider, outcome, duration_ms, prompt_tokens, completion_tokens, "
                "total_tokens, cached_tokens, stream, error_type, error_msg, extra, client"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(time.time()),
                    endpoint,
                    client_model,
                    chain_requested,
                    chain_resolved,
                    provider,
                    outcome,
                    duration_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    cached_tokens,
                    1 if stream else 0,
                    error_type,
                    error_msg,
                    extra_json,
                    client,
                ),
            )
            self._call_writes_since_prune += 1
            if self._call_writes_since_prune >= 500:
                self._conn.execute(
                    "DELETE FROM call_events WHERE ts < ?",
                    (int(time.time()) - self._CALL_RETAIN,),
                )
                self._call_writes_since_prune = 0
        except Exception:
            log.exception("record_call failed (endpoint=%s, outcome=%s)", endpoint, outcome)

    def list_calls(
        self,
        *,
        limit: int = 200,
        since: int | None = None,
        endpoint: str | None = None,
        chain: str | None = None,
        provider: str | None = None,
        outcome: str | None = None,
        client: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, ts, endpoint, client_model, chain_requested, chain_resolved, "
            "provider, outcome, duration_ms, prompt_tokens, completion_tokens, "
            "total_tokens, cached_tokens, stream, error_type, error_msg, extra, client "
            "FROM call_events WHERE 1=1"
        )
        args: list[Any] = []
        if since is not None:
            sql += " AND ts >= ?"
            args.append(int(since))
        if endpoint:
            sql += " AND endpoint = ?"
            args.append(endpoint)
        if chain:
            sql += " AND (chain_requested = ? OR chain_resolved = ?)"
            args.extend([chain, chain])
        if provider:
            sql += " AND provider = ?"
            args.append(provider)
        if outcome:
            sql += " AND outcome = ?"
            args.append(outcome)
        if client:
            if client == "anon":
                sql += " AND (client IS NULL OR client = '')"
            else:
                sql += " AND client = ?"
                args.append(client)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(int(limit), 2000)))
        cur = self._conn.execute(sql, args)
        rows: list[dict[str, Any]] = []
        for r in cur.fetchall():
            extra_raw = r[16]
            extra: Any = None
            if extra_raw:
                try:
                    extra = json.loads(extra_raw)
                except Exception:
                    extra = extra_raw
            rows.append({
                "id": int(r[0]),
                "ts": int(r[1]),
                "endpoint": r[2],
                "client_model": r[3],
                "chain_requested": r[4],
                "chain_resolved": r[5],
                "provider": r[6],
                "outcome": r[7],
                "duration_ms": r[8],
                "prompt_tokens": r[9],
                "completion_tokens": r[10],
                "total_tokens": r[11],
                "cached_tokens": r[12],
                "stream": bool(r[13]),
                "error_type": r[14],
                "error_msg": r[15],
                "extra": extra,
                "client": r[17],
            })
        return rows

    def calls_summary(self, *, since: int | None = None) -> dict[str, Any]:
        """Aggregate counts/durations/tokens for the dashboard summary band."""
        sql = "SELECT COUNT(*), COALESCE(SUM(duration_ms), 0), COALESCE(SUM(total_tokens), 0) FROM call_events"
        args: list[Any] = []
        if since is not None:
            sql += " WHERE ts >= ?"
            args.append(int(since))
        cur = self._conn.execute(sql, args)
        row = cur.fetchone() or (0, 0, 0)
        total = int(row[0] or 0)
        sum_ms = int(row[1] or 0)
        sum_tok = int(row[2] or 0)

        breakdown_sql = (
            "SELECT outcome, COUNT(*) FROM call_events"
            + (" WHERE ts >= ?" if since is not None else "")
            + " GROUP BY outcome"
        )
        cur2 = self._conn.execute(breakdown_sql, args)
        outcomes: dict[str, int] = {str(r[0]): int(r[1]) for r in cur2.fetchall()}

        client_sql = (
            "SELECT COALESCE(NULLIF(client, ''), 'anon') AS c, COUNT(*) "
            "FROM call_events"
            + (" WHERE ts >= ?" if since is not None else "")
            + " GROUP BY c ORDER BY 2 DESC"
        )
        cur3 = self._conn.execute(client_sql, args)
        clients: dict[str, int] = {str(r[0]): int(r[1]) for r in cur3.fetchall()}
        return {
            "total": total,
            "duration_ms_sum": sum_ms,
            "total_tokens_sum": sum_tok,
            "outcomes": outcomes,
            "clients": clients,
        }

    def save_audit(self, date: str, markdown: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO audit_reports(date, markdown, created_at) "
            "VALUES(?, ?, ?)",
            (date, markdown, int(time.time())),
        )

    def get_audit(self, date: str) -> str | None:
        cur = self._conn.execute(
            "SELECT markdown FROM audit_reports WHERE date = ?", (date,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def list_audits(self, limit: int = 30) -> list[tuple[str, int]]:
        cur = self._conn.execute(
            "SELECT date, created_at FROM audit_reports ORDER BY date DESC LIMIT ?",
            (limit,),
        )
        return [(str(r[0]), int(r[1])) for r in cur.fetchall()]

    def count_calls_since(self, ts: int) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM call_events WHERE ts >= ?", (int(ts),)
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
