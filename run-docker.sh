#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 python trading/main.py"
    exit 1
fi

CMD_DESC="$*"
FAILURE_LOG="$SCRIPT_DIR/logs/session_failures.log"

# C.5: operator kill switch. `touch HALT` stops cron sessions cold; `rm HALT`
# resumes. Checked before the EXIT trap is installed, so a halt neither tears
# down containers nor fires the failure alert below — a deliberate halt is not
# a failure and exits 0. In-container twin: ALGO_TRADING_HALTED (v2/session.py),
# which also covers `task session` and manual runs that bypass this script.
# See docs/runbook-recovery.md "Halt / Resume".
if [ -f "$SCRIPT_DIR/HALT" ]; then
    echo "[$(date -Is)] HALT sentinel present — skipping: $CMD_DESC"
    exit 0
fi

# Failure alerting: every failure mode used to be silent — the EXIT trap
# tore the containers down and cron discarded the nonzero exit. On failure
# we now append to logs/session_failures.log and, when ALGO_ALERT_WEBHOOK_URL
# is set (Slack/Discord/ntfy-style JSON webhook), POST a short alert.
notify_failure() {
    local status="$1"
    local msg="[$(date -Is)] run-docker.sh failed (exit ${status}): ${CMD_DESC}"
    mkdir -p "$SCRIPT_DIR/logs"
    echo "$msg" | tee -a "$FAILURE_LOG" >&2
    if [ -n "${ALGO_ALERT_WEBHOOK_URL:-}" ]; then
        curl -fsS -m 10 -X POST \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"Pinchy: session command failed (exit ${status}): ${CMD_DESC}\"}" \
            "$ALGO_ALERT_WEBHOOK_URL" >/dev/null \
            || echo "[$(date -Is)] alert webhook POST failed" | tee -a "$FAILURE_LOG" >&2
    fi
}

cleanup() {
    local status=$?
    echo "Stopping containers..."
    docker compose down
    if [ "$status" -ne 0 ]; then
        notify_failure "$status"
    fi
    exit "$status"
}

trap cleanup EXIT

echo "Starting containers..."
docker compose up -d

echo "Waiting for services..."
sleep 5

echo "Running: $@"
docker compose exec -T "$@"
