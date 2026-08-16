#!/bin/bash
#
# cron-wrap.sh <label> <command> [args...]
#
# The single cron entry point for every scheduled job in this repo. It exists
# because the 2026-08-13 audit found the system had been dead for two months
# with nothing noticing (finding 0.3): the prod path's alerting lived inside
# run-docker.sh, the paper cron went straight to `task paper:session` and so had
# no failure handling at all, and nothing anywhere could tell "host powered off
# for three weeks" apart from "all fine".
#
# Responsibilities, in order:
#   1. HALT sentinel  — a deliberate halt exits 0 and does not alert, but DOES
#                       still send the liveness ping (see below).
#   2. Liveness ping  — ALGO_HEARTBEAT_URL, healthchecks.io-style. This is the
#                       dead-man's switch: the monitor pages when a ping does
#                       NOT arrive, which is the only way "the host is off" or
#                       "cron is not firing" becomes visible. A halted system is
#                       still a live host, so a halt pings success.
#   3. Failure alert  — logs/session_failures.log + ALGO_ALERT_WEBHOOK_URL POST
#                       + a /fail ping.
#
# Host-side configuration: cron runs with a near-empty environment, and the
# compose env_file only ever reaches containers. Both vars below are read HERE,
# on the host, so they live in .env.host (gitignored; see .env.host.example).
#
#   ALGO_ALERT_WEBHOOK_URL          JSON webhook, POSTed on failure.
#   ALGO_HEARTBEAT_URL              default dead-man's-switch check URL.
#   ALGO_HEARTBEAT_URL_<LABEL>      per-job override; <LABEL> is the label
#                                   argument uppercased with - turned into _.
#                                   Each job wants its own check because each
#                                   has its own schedule — one shared URL means
#                                   any single job pinging keeps a dead job green.
#
# Exits with the wrapped command's exit status.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --ignore-halt is for jobs that protect the system rather than trade with it
# (backups, health checks). HALT means "do not trade", not "do not protect the
# data" — a hiatus is precisely when an unnoticed backup gap would hurt most.
IGNORE_HALT=
if [ "${1:-}" = "--ignore-halt" ]; then
    IGNORE_HALT=1
    shift
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 [--ignore-halt] <label> <command> [args...]" >&2
    echo "Example: $0 paper-session task paper:session" >&2
    exit 2
fi

LABEL="$1"
shift
CMD_DESC="$*"
FAILURE_LOG="$SCRIPT_DIR/logs/session_failures.log"

# Host-side config. Not an error if absent — every var it holds is optional.
if [ -f "$SCRIPT_DIR/.env.host" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$SCRIPT_DIR/.env.host"
    set +a
fi

log() {
    mkdir -p "$SCRIPT_DIR/logs"
    echo "[$(date -Is)] cron-wrap[$LABEL] $*" | tee -a "$FAILURE_LOG" >&2
}

# Per-job override wins over the shared default.
heartbeat_url() {
    local key
    key="ALGO_HEARTBEAT_URL_$(echo "$LABEL" | tr '[:lower:]-' '[:upper:]_')"
    local url="${!key:-${ALGO_HEARTBEAT_URL:-}}"
    echo "$url"
}

# Best-effort by design: a monitoring endpoint being down must never turn a
# successful trading session into a failed one, so every failure here is logged
# and swallowed.
ping_heartbeat() {
    local suffix="${1:-}"
    local base
    base="$(heartbeat_url)"
    [ -n "$base" ] || return 0
    curl -fsS -m 10 --retry 3 -o /dev/null "${base}${suffix}" \
        || log "heartbeat ping failed (${suffix:-ok})"
}

notify_failure() {
    local status="$1"
    log "FAILED (exit ${status}): ${CMD_DESC}"
    if [ -n "${ALGO_ALERT_WEBHOOK_URL:-}" ]; then
        curl -fsS -m 10 -X POST \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"Pinchy: ${LABEL} failed (exit ${status}): ${CMD_DESC}\"}" \
            "$ALGO_ALERT_WEBHOOK_URL" >/dev/null \
            || log "alert webhook POST failed"
    fi
}

# A halt is deliberate: exit 0, do not alert. It still pings, because the point
# of the dead-man's switch is "is this host alive and is cron firing", and the
# answer during a hiatus is yes. Losing that distinction is how a halt starts
# looking like an outage again.
if [ -z "$IGNORE_HALT" ] && [ -f "$SCRIPT_DIR/HALT" ]; then
    echo "[$(date -Is)] cron-wrap[$LABEL] HALT sentinel present — skipping: $CMD_DESC"
    ping_heartbeat
    exit 0
fi

ping_heartbeat /start

# Tells run-docker.sh that alerting is already owned up here, so a wrapped prod
# session logs and POSTs once rather than twice. Unwrapped invocations of
# run-docker.sh keep their own alerting — the whole point of this audit is that
# the fallback path is the one that silently stops working.
export ALGO_CRON_WRAPPED=1

"$@"
status=$?

if [ "$status" -ne 0 ]; then
    notify_failure "$status"
    ping_heartbeat "/${status}"
else
    ping_heartbeat
fi

exit "$status"
