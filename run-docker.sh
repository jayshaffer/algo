#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <instance> <service> <command> [args...]"
    echo "Example: $0 live trading python -m v2.session"
    exit 1
fi

INSTANCE="$1"
shift
ENV_FILE="$SCRIPT_DIR/instances/$INSTANCE.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "No such instance: instances/$INSTANCE.env" >&2
    exit 2
fi
# Project-scoped compose invocation: `down` below must only ever touch THIS
# instance's stack — a bare `docker compose down` would take whichever
# project matched the directory name, including another instance's.
COMPOSE=(docker compose -p "pinchy-$INSTANCE" --env-file "$ENV_FILE")

CMD_DESC="[$INSTANCE] $*"
FAILURE_LOG="$SCRIPT_DIR/logs/session_failures.log"

# C.5: operator kill switch, two levels. Global `touch HALT` stops every
# instance; `touch instances/<name>.HALT` stops just that one. Checked before
# the EXIT trap is installed, so a halt neither tears down containers nor
# fires the failure alert below — a deliberate halt is not a failure and
# exits 0. In-container twin: ALGO_TRADING_HALTED (v2/session.py).
# See docs/runbook-recovery.md "Halt / Resume".
if [ -f "$SCRIPT_DIR/HALT" ]; then
    echo "[$(date -Is)] HALT sentinel present — skipping: $CMD_DESC"
    exit 0
fi
if [ -f "$SCRIPT_DIR/instances/$INSTANCE.HALT" ]; then
    echo "[$(date -Is)] instances/$INSTANCE.HALT present — skipping: $CMD_DESC"
    exit 0
fi

# Failure alerting: every failure mode used to be silent — the EXIT trap
# tore the containers down and cron discarded the nonzero exit. On failure
# we now append to logs/session_failures.log and, when ALGO_ALERT_WEBHOOK_URL
# is set (Slack/Discord/ntfy-style JSON webhook), POST a short alert.
#
# cron-wrap.sh owns alerting (plus the dead-man's-switch ping) for every
# scheduled job and sets ALGO_CRON_WRAPPED so we don't double-report. This
# fallback stays for direct/manual invocations: the audit's recurring lesson is
# that the un-wrapped path is exactly the one that quietly stops being covered.
notify_failure() {
    local status="$1"
    [ -z "${ALGO_CRON_WRAPPED:-}" ] || return 0
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

# `set -e` is still in force inside the trap, so the teardown must not be able
# to abort the handler before it reports: a `docker compose down` that fails
# (daemon restarting, network already gone) would otherwise kill the shell and
# swallow the real exit status along with the alert. Report first, tear down
# with an explicit `|| true`.
cleanup() {
    local status=$?
    if [ "$status" -ne 0 ]; then
        notify_failure "$status"
    fi
    echo "Stopping containers..."
    "${COMPOSE[@]}" down || echo "[$(date -Is)] docker compose down failed" >&2
    exit "$status"
}

trap cleanup EXIT

echo "Starting containers..."
"${COMPOSE[@]}" up -d

echo "Waiting for services..."
sleep 5

echo "Running: $@"
"${COMPOSE[@]}" exec -T "$@"
