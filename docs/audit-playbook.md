# Audit Playbook

This file is read by the Claude Code /loop session on each 24h tick. It is the single source of truth for what gets audited. Edit this file to change the audit; no code change required.

> **Spec:** `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`

## How to read this file

When invoked from the /loop:

1. Execute every entry under **Deterministic checks** in order.
2. Run the **Ideation pass** per its instructions.
3. For every finding (deterministic or ideation), apply the **Filing rules**.
4. Then perform **Phase B (Execution)** per the spec.
5. Stop and wait for the next interval.

## Environment

Pick the docker service based on each check's declared `env`:
- `prod` -> `docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"`
- `paper` -> `docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"`
- `both` -> run twice (once each env). File env in the ticket title prefix: `[audit:<category>:<env>] <Title>`.

The `$POSTGRES_USER` and `$POSTGRES_DB` come from `.env` and `.env.paper`; they're already in the trading container's environment.

## Deterministic checks

(filled in by Step 2 of Task 2)

## Ideation pass

(filled in by Step 1 of Task 3)

## Filing rules

(filled in by Step 2 of Task 3)

## Phase B: Execution

(filled in by Step 3 of Task 3)
