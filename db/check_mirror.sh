#!/bin/bash
#
# Verifies that db/init/ and db/migrations/ describe the same schema.
#
# db/init/ only ever runs on a FRESH Postgres volume, so long-lived prod/paper
# volumes get their changes from db/migrations/ instead. The two are maintained
# by hand and have silently diverged three times (audit 3.3, 2.2): CI seeds from
# db/init/ alone, so it has been testing against a schema that is structurally
# different from production, and a DR restore onto a fresh volume comes up
# missing tables the strategist reads.
#
# The invariant: every migration is idempotent, so applying all of them on top
# of a fully init-seeded database must be a no-op. If the schema changes, some
# migration carries a change that db/init/ never learned about.
#
# Usage: db/check_mirror.sh [psql-connection-args...]
#   CI:    db/check_mirror.sh -h localhost -U algo
#   local: db/check_mirror.sh -h localhost -p 5432 -U algo
#
# Needs CREATEDB on the connecting role; works on a scratch database so it
# never touches trading data.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH_DB="schema_mirror_check_$$"
PSQL_ARGS=("$@")

psql_run() { psql "${PSQL_ARGS[@]}" -v ON_ERROR_STOP=1 "$@"; }

cleanup() {
    psql_run -d postgres -q -c "DROP DATABASE IF EXISTS \"$SCRATCH_DB\";" >/dev/null 2>&1
}
trap cleanup EXIT

echo "==> creating scratch database $SCRATCH_DB"
psql_run -d postgres -q -c "CREATE DATABASE \"$SCRATCH_DB\";" || exit 1

echo "==> applying db/init/*.sql (what a fresh volume gets)"
for f in "$REPO_DIR"/db/init/*.sql; do
    psql_run -d "$SCRATCH_DB" -q -f "$f" >/dev/null || {
        echo "FAIL: db/init/$(basename "$f") did not apply cleanly" >&2
        exit 1
    }
done

before="$(mktemp)"
after="$(mktemp)"
dump_schema() {
    # \restrict / \unrestrict carry a random nonce that differs per pg_dump
    # invocation; stripping them is the only normalization applied, so a real
    # schema difference cannot hide behind it.
    pg_dump "${PSQL_ARGS[@]}" -d "$SCRATCH_DB" --schema-only --no-owner --no-privileges \
        | grep -vE '^\\(un)?restrict '
}

dump_schema > "$before" || exit 1

echo "==> applying db/migrations/*.sql on top (what a long-lived volume gets)"
for f in "$REPO_DIR"/db/migrations/*.sql; do
    # A migration whose objects were later dropped cannot be replayed against a
    # current schema. Such files opt out with a `-- mirror-check: skip` line
    # carrying the reason; they stay in the directory as applied history.
    if grep -q '^-- mirror-check: skip' "$f"; then
        echo "    skipping $(basename "$f") (opted out)"
        continue
    fi
    psql_run -d "$SCRATCH_DB" -q -f "$f" >/dev/null || {
        echo "FAIL: db/migrations/$(basename "$f") is not idempotent — it errored" >&2
        echo "      when applied to a database already seeded from db/init/." >&2
        exit 1
    }
done

dump_schema > "$after" || exit 1

if diff -u "$before" "$after" > /tmp/schema_mirror_diff.$$ 2>&1; then
    echo "==> OK: db/init/ and db/migrations/ agree"
    rm -f "$before" "$after" /tmp/schema_mirror_diff.$$
    exit 0
fi

echo "" >&2
echo "FAIL: applying db/migrations/ changed a schema built from db/init/." >&2
echo "" >&2
echo "Something in db/migrations/ has no db/init/ counterpart. CI and any" >&2
echo "fresh volume are missing it. Add the equivalent db/init/NNN_*.sql file." >&2
echo "" >&2
cat /tmp/schema_mirror_diff.$$ >&2
rm -f "$before" "$after" /tmp/schema_mirror_diff.$$
exit 1
