"""Daily server health heartbeat → Telegram.

Runs in a background asyncio task on the same schedule as `audit_loop` (00:03 PT).
Always sends — silence means the bot or server is down. Anomalies (high RAM, full
disk, service inactive, errors in journalctl) are flagged with ⚠️/🔴 markers in
the message header so they can't be missed at a glance."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING
from zoneinfo import ZoneInfo

import httpx
import psutil

from .audit import seconds_until_next_run

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)

_PT = ZoneInfo("America/Los_Angeles")
_DISK_PATH = "/"
# Systemd unit name. На проде сервис называется `llmgate.service`; локальный
# unit может быть `llmgate.service`. Можно переопределить через ENV.
_SERVICE_UNIT = os.getenv("LLMGATE_SYSTEMD_UNIT", "llmgate")


def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _gb(b: int) -> str:
    return f"{b / (1024 ** 3):.1f}"


def _mb(b: int) -> int:
    return b // (1024 * 1024)


async def _systemctl_show() -> dict[str, str]:
    """Fetch service state via `systemctl show`. {} on non-Linux."""
    if not shutil.which("systemctl"):
        return {}
    try:
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "show", _SERVICE_UNIT,
            "-p", "ActiveState", "-p", "NRestarts", "-p", "MainPID",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
    except Exception:
        return {}
    info: dict[str, str] = {}
    for line in out.decode("utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


async def _journalctl_errors() -> tuple[int, list[str]]:
    """(count, last_3_lines) of err+ from the service over last 24h. (0, []) on non-Linux."""
    if not shutil.which("journalctl"):
        return 0, []
    try:
        proc = await asyncio.create_subprocess_exec(
            "journalctl", "-u", _SERVICE_UNIT, "-p", "err",
            "--since", "24 hours ago", "--no-pager", "-q",
            "-o", "short-iso",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except Exception:
        return 0, []
    lines = [ln for ln in out.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    return len(lines), lines[-3:]


def _top_mem(n: int = 3) -> list[tuple[str, int]]:
    procs: list[tuple[str, int]] = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            mi = p.info.get("memory_info")
            if mi is None:
                continue
            procs.append((p.info.get("name") or "?", int(mi.rss)))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procs.sort(key=lambda x: x[1], reverse=True)
    return procs[:n]


def _collect_sync(app: "FastAPI") -> dict[str, Any]:
    """psutil + rate_tracker. cpu_percent(interval=1.0) blocks ~1s — we run this
    in a thread pool from the async caller."""
    return {
        "uptime": time.time() - psutil.boot_time(),
        "load": psutil.getloadavg(),
        "cpus": psutil.cpu_count(logical=True) or 1,
        "cpu_pct": psutil.cpu_percent(interval=1.0),
        "mem": psutil.virtual_memory(),
        "swap": psutil.swap_memory(),
        "disk": psutil.disk_usage(_DISK_PATH),
        "self_rss": psutil.Process().memory_info().rss,
        "top_mem": _top_mem(3),
        "api_24h": app.state.rate_tracker.count_calls_since(int(time.time()) - 86400),
    }


async def collect_health(app: "FastAPI") -> dict[str, Any]:
    data = await asyncio.to_thread(_collect_sync, app)
    data["systemd"] = await _systemctl_show()
    err_count, err_tail = await _journalctl_errors()
    data["err_count"] = err_count
    data["err_tail"] = err_tail
    return data


def _classify(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (warns, crits) lists of subsystem names breaching thresholds."""
    warns: list[str] = []
    crits: list[str] = []

    if data["mem"].percent >= 90:
        crits.append("ram")
    elif data["mem"].percent >= 70:
        warns.append("ram")

    if data["disk"].percent >= 90:
        crits.append("disk")
    elif data["disk"].percent >= 75:
        warns.append("disk")

    load1 = data["load"][0]
    cpus = data["cpus"]
    if load1 >= cpus * 1.5:
        crits.append("load")
    elif load1 >= cpus * 0.7:
        warns.append("load")

    # Linux ядро держит "хвост" swap из неактивных страниц даже при свободной RAM.
    # На multi-tenant VPS swap всегда ненулевой, поэтому warn только при существенном
    # использовании — иначе heartbeat = вечный фолз-флаг.
    if data["swap"].total > 0:
        if data["swap"].percent >= 75:
            crits.append("swap")
        elif data["swap"].percent >= 25:
            warns.append("swap")

    sysctl = data["systemd"]
    if sysctl and sysctl.get("ActiveState") != "active":
        crits.append("service")

    if int(sysctl.get("NRestarts", "0") or "0") > 0:
        warns.append("restarts")

    if data["err_count"] >= 10:
        crits.append("errors")
    elif data["err_count"] >= 1:
        warns.append("errors")

    return warns, crits


def format_report(date: str, data: dict[str, Any]) -> str:
    warns, crits = _classify(data)

    def mp(key: str) -> str:
        """Marker prefix: '🔴 ' / '⚠️ ' / '' depending on classification."""
        if key in crits:
            return "🔴 "
        if key in warns:
            return "⚠️ "
        return ""

    if crits:
        header = f"🔴 {', '.join(crits)}"
    elif warns:
        header = f"⚠️ {', '.join(warns)}"
    else:
        header = "✅ all good"

    mem, swap, disk = data["mem"], data["swap"], data["disk"]
    load1, load5, load15 = data["load"]
    cpus = data["cpus"]
    sysctl = data["systemd"]

    out: list[str] = [
        f"🖥️ llmgate health {date} — {header}",
        "",
        f"uptime: {_fmt_uptime(data['uptime'])}",
        f"{mp('load')}load: {load1:.2f} / {load5:.2f} / {load15:.2f} ({cpus} cpu)",
        (
            f"cpu: {data['cpu_pct']:.0f}%"
            f"  ·  {mp('ram')}ram: {_gb(mem.used)}/{_gb(mem.total)} GB ({mem.percent:.0f}%)"
            f"  ·  {mp('swap')}swap: {_gb(swap.used)}/{_gb(swap.total)} GB"
        ),
        f"{mp('disk')}disk /: {_gb(disk.used)}/{_gb(disk.total)} GB ({disk.percent:.0f}%)",
        "",
    ]

    svc_bits: list[str] = []
    if sysctl:
        active = sysctl.get("ActiveState", "?")
        svc_bits.append(f"{mp('service')}{active}")
        n_restarts = int(sysctl.get("NRestarts", "0") or "0")
        svc_bits.append(f"{mp('restarts')}{n_restarts} restarts")
    svc_bits.append(f"{_mb(data['self_rss'])}MB RSS")
    out.append("llmgate: " + "  ·  ".join(svc_bits))
    out.append(f"api 24h: {data['api_24h']} calls")
    out.append(f"{mp('errors')}errors 24h: {data['err_count']}")
    for ln in data["err_tail"]:
        out.append(f"  {ln[:140]}")
    out.append("")

    top_str = "  ·  ".join(f"{n} {_mb(b)}MB" for n, b in data["top_mem"])
    out.append(f"top mem: {top_str}")
    return "\n".join(out)


async def _notify_telegram(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        log.info("health: telegram skipped (no token/chat_id)")
        return
    payload = text if len(text) <= 3900 else text[:3900] + "\n…"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": payload,
                    "disable_web_page_preview": True,
                },
            )
            if resp.status_code >= 400:
                log.warning("health telegram HTTP %d: %s", resp.status_code, resp.text[:200])
                return
        log.info("health: telegram notified")
    except Exception as exc:
        log.warning("health telegram failed: %s", exc)


async def run_health_report(app: "FastAPI") -> None:
    data = await collect_health(app)
    date = datetime.now(_PT).strftime("%Y-%m-%d")
    report = format_report(date, data)
    log.info("health: report generated for %s", date)
    await _notify_telegram(report)


async def health_loop(app: "FastAPI") -> None:
    log.info("health: loop started, next run in %.0fs", seconds_until_next_run())
    while True:
        try:
            await asyncio.sleep(seconds_until_next_run())
            await run_health_report(app)
        except asyncio.CancelledError:
            log.info("health: loop cancelled")
            raise
        except Exception:
            log.exception("health: run failed, sleeping 1h and retrying")
            await asyncio.sleep(3600)
