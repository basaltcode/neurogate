#!/usr/bin/env bash
# Weekly audit: vendor catalog diff → smoke + identity verification.
# Запуск: вручную или через `crontab -e` (см. ниже).
# НЕ деплоит, НЕ мержит, НЕ редактирует chains: — только candidates.json + verified.json.
#
# crontab пример (понедельник 10:00 MSK = 07:00 UTC):
#   0 7 * * 1 cd /Users/niko/Desktop/llmgate && bash scans/run_weekly_audit.sh >> scans/cron.log 2>&1

set -euo pipefail

cd "$(dirname "$0")/.."

DATE=$(date +%F)
CANDIDATES="scans/audit-${DATE}-candidates.json"
VERIFIED="scans/audit-${DATE}-verified.json"

echo "=== $(date -u +%FT%TZ) audit ${DATE} ==="

# Stage 1: catalog diff (no LLM, no hallucinations possible)
uv run python scans/audit_catalog.py --free-only --out "${CANDIDATES}"

NUM=$(python3 -c "import json,sys; print(len(json.load(open('${CANDIDATES}'))))")
if [[ "${NUM}" == "0" ]]; then
    echo "no new candidates — done."
    exit 0
fi

# Stage 2: dedup + smoke + identity-probe
uv run python scans/audit_verifier.py "${CANDIDATES}"

CONFIRMED=$(python3 -c "import json; d=json.load(open('${VERIFIED}')); print(sum(1 for v in d if v['status']=='confirmed'))")
UNVERIFIED=$(python3 -c "import json; d=json.load(open('${VERIFIED}')); print(sum(1 for v in d if v['status']=='confirmed_unverified'))")

echo
echo "summary: ${CONFIRMED} confirmed, ${UNVERIFIED} confirmed_unverified (manual review)"
echo "files:"
echo "  - ${CANDIDATES}"
echo "  - ${VERIFIED}"
echo
echo "Next steps (manual):"
echo "  1. Eyeball ${VERIFIED}"
echo "  2. For each confirmed/confirmed_unverified — add provider entry to config.yaml"
echo "  3. Run fetchtest/bench_latency.py + tests/ru_bench.py for the new entries"
echo "  4. Open PR. Никаких auto-merge / auto-deploy."
