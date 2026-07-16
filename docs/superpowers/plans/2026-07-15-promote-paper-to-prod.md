# Promote Paper Stack to Primary Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the paper pipeline the primary (default, unprefixed) stack: paper DB volume behind `db:5432`, paper Alpaca account in `.env`, public dashboard published again (relabeled as paper trading), single daily cron via `run-docker.sh`, old live-trading DB archived and dormant.

**Architecture:** Pure config/ops swap plus one small code change (paper banner in the rendered dashboard shell). No data copy: `docker-compose.yml` re-points its volume to the existing paper volume (`algo_postgres_data_paper` — both compose files share project name `algo`, so the name resolves identically). The paper overlay, `.env.paper`, and `paper:*` Taskfile targets are retired.

**Tech Stack:** docker compose, go-task (Taskfile), cron, psql/pg_dump, Python (dashboard page renderer), pytest in the trading container.

**Spec:** `docs/superpowers/specs/2026-07-15-promote-paper-to-prod-design.md`

**Spec correction discovered during planning:** `.env.paper` does NOT set `ALGO_EXECUTOR_MODEL` — there is no Sonnet pilot to carry over. The promoted executor stays at the default (`claude-haiku-4-5-20251001`). The spec's Decisions table is wrong on this one point; do not add the knob.

**Facts established during planning (do not re-derive):**
- Both env files use `POSTGRES_USER=algo`, `POSTGRES_DB=trading`, and the same password — the paper volume accepts the same credentials. Verified via `DATABASE_URL` in both files.
- `.env.paper`'s `DATABASE_URL` host is `db-paper`; the promoted stack's service is `db`, so the host must change.
- The public dashboard's `index.html` is rendered at publish time by `v2/dashboard_publish.py` (see comment at v2/dashboard_publish.py:1500 "replaces the old static index.html copy"); the stale `index.html` in the site repo (`DASHBOARD_REPO_PATH`) is a dead artifact. All page chrome comes from `_render_page_shell` in `v2/dashboard_pages.py` (title template line ~305, logo line ~271).
- `run-docker.sh` traps EXIT → `docker compose down`, so the promoted stack is ephemeral between sessions, same as old prod. The nightly backup must bring `db` up itself and must not tear the stack down if a session is mid-flight.
- Old `.env` contains retired social vars (`TWITTER_*`, `BLUESKY_*`, `ALGO_ENABLE_TRADE_POSTS`) — drop them. Keep audit/Jira vars and `FINNHUB_API_KEY` as-is (carried, not validated, to avoid breaking the audit loop or legacy v1 imports).
- Prod `db` container may already be running (left up by the 2026-07-15 audit).

---

### Task 1: Archive the retired live-trading DB and preserve live credentials

**Files:**
- Modify: `.gitignore` (add `backups/`)
- Create (untracked): `backups/prod-live-final-2026-07-15.dump`, `.env.live-retired`

- [ ] **Step 1: Confirm the OLD prod db container is up and still points at the live volume**

Run: `cd /home/jay/dev/algo && docker compose up -d db && docker compose exec -T db psql -U algo -d trading -tAc "SELECT max(session_date) FROM sessions;"`
Expected: `2026-06-15` (the live DB's last session). If you see a different date, STOP — you are looking at the wrong volume; do not proceed to the swap until this is understood.

- [ ] **Step 2: Add backups/ to .gitignore**

Check first: `grep -n "backups" .gitignore || true`. If absent, append:

```
# DB archives (pg_dump output) — never commit
backups/
```

- [ ] **Step 3: Dump the live DB**

```bash
mkdir -p backups
docker compose exec -T db pg_dump -U algo -Fc trading > backups/prod-live-final-2026-07-15.dump
ls -la backups/
```

Expected: dump file present, size on the order of MBs (not 0 bytes).

- [ ] **Step 4: Verify the dump is restorable-shaped**

Run: `docker compose exec -T db pg_restore --list < backups/prod-live-final-2026-07-15.dump | head -20`
Expected: a table-of-contents listing including `TABLE decisions`, `TABLE theses`, etc. Errors = bad dump, redo Step 3.

- [ ] **Step 5: Copy the dump off the WSL VHD**

Discover the Windows user dir, then copy:

```bash
ls /mnt/c/Users/
mkdir -p "/mnt/c/Users/<WINDOWS_USER>/pinchy-backups"
cp backups/prod-live-final-2026-07-15.dump "/mnt/c/Users/<WINDOWS_USER>/pinchy-backups/"
ls -la "/mnt/c/Users/<WINDOWS_USER>/pinchy-backups/"
```

Record the chosen path — Task 8 writes it into CLAUDE.md.

- [ ] **Step 6: Preserve live credentials and stop the old stack**

```bash
cp .env .env.live-retired
grep -q "^\.env" .gitignore && echo "env files already ignored"
docker compose down
```

Expected: `.env.live-retired` exists; `docker ps` shows no algo containers.

- [ ] **Step 7: Commit the .gitignore change**

```bash
git add .gitignore
git commit -m "chore: ignore backups/ (DB archive directory)"
```

---

### Task 2: Swap compose + env to the paper account and volume

**Files:**
- Modify: `docker-compose.yml:9` (volume mapping), `:50` (volume declaration)
- Modify: `.env` (rebuilt from `.env.paper` + carried-over vars)
- Delete: `docker-compose.paper.yml`, `.env.paper`

- [ ] **Step 1: Re-point the db volume in docker-compose.yml**

In the `db` service, change:

```yaml
    volumes:
      - postgres_data:/var/lib/postgresql/data
```
to:
```yaml
    volumes:
      - postgres_data_paper:/var/lib/postgresql/data
```

and at the bottom, change the declaration:

```yaml
volumes:
  postgres_data_paper:
```

(The old `postgres_data` declaration is removed; the dormant volume `algo_postgres_data` continues to exist outside compose management.)

- [ ] **Step 2: Rebuild .env**

Construct the new `.env` as: **all of `.env.paper`** with the `DATABASE_URL` host fixed (`@db-paper:5432` → `@db:5432`), **plus** these lines copied verbatim from `.env.live-retired`:

```
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_PAGES_PROJECT=...
DASHBOARD_REPO_PATH=...
DASHBOARD_URL=...        # old .env had this line twice — keep exactly one
ANTHROPIC_API_KEY=...    # prefer the .env.paper value if they differ
FINNHUB_API_KEY=...
ALGO_AUDIT_FILE_JIRA=...
JIRA_BASE_URL=...
JIRA_EMAIL=...
JIRA_API_TOKEN=...
```

Explicitly **dropped** (retired social stack): `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET`, `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`, `ALGO_ENABLE_TRADE_POSTS`.

Sanity-check the result (no secrets printed):

```bash
grep -E '^(ALPACA_BASE_URL|ALPACA_PAPER|POSTGRES_USER|POSTGRES_DB)=' .env
grep -c 'DASHBOARD_URL' .env
grep 'DATABASE_URL' .env | grep -o '@[a-z-]*:'
```

Expected: `ALPACA_BASE_URL=https://paper-api.alpaca.markets`, `ALPACA_PAPER=true`, `POSTGRES_USER=algo`, `POSTGRES_DB=trading`, DASHBOARD_URL count `1`, DATABASE_URL host `@db:`.

- [ ] **Step 3: Retire the paper overlay and env file**

```bash
git rm docker-compose.paper.yml
rm .env.paper
```

(`.env.paper` is gitignored/untracked — plain `rm`; git history is not involved for it, which is why `.env.live-retired` was preserved in Task 1.)

- [ ] **Step 4: Validate compose renders**

Run: `docker compose config --quiet && echo OK`
Expected: `OK` (no references to db-paper or .env.paper remain in the default stack).

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml docker-compose.paper.yml
git commit -m "feat: promote paper stack to primary — db volume + env swap"
```

---

### Task 3: Bring up the promoted stack, verify identity, apply migrations

**Files:** none (verification + DB state)

- [ ] **Step 1: Start the promoted db and prove it's the paper history**

```bash
docker compose up -d db
sleep 5
docker compose exec -T db psql -U algo -d trading -tAc "SELECT max(session_date) FROM sessions;"
docker compose exec -T db psql -U algo -d trading -tAc "SELECT count(*) FROM decisions;"
```

Expected: max session date ≈ `2026-06-26` (paper's last session — NOT 2026-06-15, which would mean the volume swap failed and you're on the live volume; STOP if so). Note both numbers.

- [ ] **Step 2: Check migration drift on the promoted volume**

```bash
docker compose exec -T db psql -U algo -d trading -tAc "SELECT filename FROM schema_migrations ORDER BY filename;"
ls db/migrations/
```

Expected: some tail of `db/migrations/` missing from the table (paper volume has its own drift history).

- [ ] **Step 3: Apply migrations**

Run: `task db:migrate`
Expected: `==> applying ...` lines for each missing file, no errors.

- [ ] **Step 4: Verify migrations landed**

Run: `docker compose exec -T db psql -U algo -d trading -tAc "SELECT count(*) FROM schema_migrations;" && docker compose exec -T db psql -U algo -d trading -tAc "SELECT price_per_mtok_input FROM model_pricing WHERE model LIKE 'claude-fable%';"`
Expected: migration count == number of files in `db/migrations/`; a fable-5 pricing row exists. (If the pricing table/columns differ, adapt the query — the point is the fable-5 row.)

- [ ] **Step 5: Executor cross-check passes with the new env**

Run: `docker compose up -d trading && docker compose exec -T trading python -c "import v2.executor; print('cross-check OK')"`
Expected: `cross-check OK` (module-load ALPACA_PAPER/URL validation passes).

---

### Task 4: Taskfile — remove paper targets, add db:backup

**Files:**
- Modify: `Taskfile.yml`

- [ ] **Step 1: Remove all paper targets**

Delete these tasks entirely: `paper:up`, `paper:down`, `paper:session`, `paper:session:dry-run`, `paper:dashboard`, `paper:backfill:decision`, `paper:db:migrate` (Taskfile.yml lines ~112-145 and ~225-242). Also delete the `# Paper trading` section header comment.

- [ ] **Step 2: Add db:backup target**

Add under the Database section. It must self-manage the db container and must NOT tear down a running session:

```yaml
  db:backup:
    desc: pg_dump the primary DB to backups/ and copy off-VHD; safe to run while a session is live
    cmds:
      - |
        set -e
        TRADING_WAS_UP=$(docker compose ps --status running --format '{{.Name}}' | grep -c trading || true)
        docker compose up -d db
        sleep 3
        mkdir -p backups
        STAMP=$(date +%F)
        docker compose exec -T db pg_dump -U algo -Fc trading > "backups/primary-${STAMP}.dump"
        ls -la "backups/primary-${STAMP}.dump"
        if [ -d "{{.OFFSITE_DIR}}" ]; then
          cp "backups/primary-${STAMP}.dump" "{{.OFFSITE_DIR}}/"
        else
          echo "WARN: offsite dir {{.OFFSITE_DIR}} not found; dump kept locally only"
        fi
        # keep the 14 most recent local dumps
        ls -t backups/primary-*.dump | tail -n +15 | xargs -r rm --
        if [ "$TRADING_WAS_UP" = "0" ]; then
          docker compose stop db
        fi
    vars:
      OFFSITE_DIR: /mnt/c/Users/<WINDOWS_USER>/pinchy-backups
```

Replace `<WINDOWS_USER>` with the directory discovered in Task 1 Step 5.

- [ ] **Step 3: Verify Taskfile parses and backup works end-to-end**

```bash
task --list
task db:backup
ls backups/
```

Expected: `--list` shows no `paper:*` targets and shows `db:backup`; a `primary-<today>.dump` exists locally and in the offsite dir; db container stopped afterwards (nothing else was running).

- [ ] **Step 4: Commit**

```bash
git add Taskfile.yml
git commit -m "feat: retire paper:* Taskfile targets; add db:backup with offsite copy"
```

---

### Task 5: Paper-trading banner on the public dashboard (TDD)

**Files:**
- Modify: `v2/dashboard_pages.py` (`_render_page_shell`, ~line 324)
- Test: `tests/v2/test_dashboard_pages.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_dashboard_pages.py` (match the file's existing import style for `_render_page_shell` or the public render functions — if `_render_page_shell` isn't imported directly in existing tests, assert via `render_mistakes_page`, which wraps it):

```python
def test_page_shell_carries_paper_trading_banner():
    from v2.dashboard_pages import render_mistakes_page

    html = render_mistakes_page(closed_losers=[], retired_rules=[], base_url="https://example.com")
    assert "paper trading" in html.lower()
    assert "no real money" in html.lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `docker compose exec -T trading python -m pytest tests/v2/test_dashboard_pages.py::test_page_shell_carries_paper_trading_banner -v`
Expected: FAIL (`assert 'paper trading' in ...`).

- [ ] **Step 3: Add the banner to `_render_page_shell`**

In `v2/dashboard_pages.py`, in the shell template immediately after the site header/nav block (near the `'<a class="logo" href="/">⌬ Pinchy</a>'` line ~271, inside the rendered body), insert a banner element:

```python
_PAPER_BANNER = (
    '<div class="paper-banner" role="note">'
    '📄 Paper trading — simulated orders on a paper account, no real money at risk.'
    '</div>'
)
```

and include `_PAPER_BANNER` in the shell output directly below the header. Follow the file's existing string-template idiom (it uses `$title`-style templates around line 305 — add the banner as a literal in the template, not a substitution, since it appears on every page).

Add minimal styling in the shell's inline CSS (or `public_dashboard/styles.css` if that's where page chrome styles live — check where `.logo` is styled and put `.paper-banner` next to it):

```css
.paper-banner {
  background: #fef3c7;
  color: #92400e;
  padding: 6px 16px;
  font-size: 0.85rem;
  text-align: center;
}
@media (prefers-color-scheme: dark) {
  .paper-banner { background: #3a2e08; color: #fbbf24; }
}
```

- [ ] **Step 4: Run the test to verify it passes, plus the file's whole suite**

Run: `docker compose exec -T trading python -m pytest tests/v2/test_dashboard_pages.py -v`
Expected: new test PASSES; if any existing snapshot-ish tests assert on full shell HTML, update them to expect the banner (that's a legitimate behavior change, not test fudging).

- [ ] **Step 5: Commit**

```bash
git add v2/dashboard_pages.py tests/v2/test_dashboard_pages.py public_dashboard/styles.css
git commit -m "feat: label public dashboard as paper trading (banner in page shell)"
```

---

### Task 6: Rewrite the crontab (single pipeline + backup + alert var)

**Files:**
- Modify: `crontab`

- [ ] **Step 1: Rewrite the repo crontab file**

Replace the full contents of `crontab` with:

```
# Pinchy Crontab — single (paper-account) pipeline since 2026-07-15
# Install with: crontab /home/jay/dev/algo/crontab
# All times are MST (America/Denver, UTC-7)
# NOTE: cron only runs if the WSL cron service is started (Windows Task
# Scheduler runs start-wsl-cron.bat at boot).

PATH=/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin

# TODO: set a real webhook URL (Slack/Discord/ntfy) to activate failure alerts
ALGO_ALERT_WEBHOOK_URL=

# Daily session (1 PM MST / 3 PM ET, Mon-Fri) — paper account
# Stage 0: backfill + attribution | 0.5: supervisor | 1: news pipeline
# 2: strategist | 3: executor | 4: reflection | 5: public dashboard publish
0 13 * * 1-5 /home/jay/dev/algo/run-docker.sh trading python -m v2.session

# Weekly deep learning analysis (5 AM MST / 7 AM ET, Sunday)
0 5 * * 0 /home/jay/dev/algo/run-docker.sh trading python -m v2.learn --days 60

# Nightly DB backup (8 PM MST, daily) — local dump + off-VHD copy
0 20 * * * cd /home/jay/dev/algo && task db:backup >> logs/backup.log 2>&1
```

(The old 12:30 `task paper:session` line is gone — that pipeline IS now the 13:00 line. The 13:00 line is deliberately active: promotion ends the hiatus for paper trading.)

- [ ] **Step 2: Install and verify**

```bash
crontab /home/jay/dev/algo/crontab
diff <(crontab -l) /home/jay/dev/algo/crontab && echo "IN SYNC"
```

Expected: `IN SYNC` — installed state matches the repo for the first time since the hiatus edit.

- [ ] **Step 3: Commit**

```bash
git add crontab
git commit -m "feat: single-pipeline crontab — daily paper session, nightly backup, alert var stub"
```

---

### Task 7: Sweep remaining paper-stack references

**Files:**
- Modify: whatever the sweep finds (expected: `docs/audit-playbook.md` Environment section; possibly `.claude/commands/audit-*.md`, `run-docker.sh` comments, README)

- [ ] **Step 1: Find every live reference**

```bash
grep -rn --exclude-dir=.git --exclude-dir=logs_paper --exclude-dir=docs/superpowers \
  -e 'docker-compose\.paper' -e '\.env\.paper' -e 'db-paper' -e 'trading-paper' \
  -e 'paper:session' -e 'paper:up' -e 'paper:db:migrate' -e 'logs_paper' . \
  | grep -v -e '^\./docs/audits/' -e 'CLAUDE\.md'
```

(CLAUDE.md is Task 8; historical audit reports and specs stay as-is — they document the past.)

- [ ] **Step 2: Fix the audit playbook's Environment section**

In `docs/audit-playbook.md` (lines ~21-28): the `paper`/`both` env definitions reference `db-paper` and the paper overlay, which no longer exist. Rewrite the section to a single environment:

```markdown
## Environment

All checks run against the single primary database:
`docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"`

(The paper/prod split was retired 2026-07-15 when the paper pipeline was
promoted to primary; the live-trading DB is archived and dormant. Checks
formerly marked `env: both` or `env: paper` now run once against the
primary DB. Ticket title prefixes drop the `:<env>` segment.)
```

Leave each check's `env:` line in place (they're now interpreted per the note) — do NOT hand-edit 30 check entries.

- [ ] **Step 3: Fix any other hits from Step 1**

For each remaining hit, apply the same principle: active tooling gets updated to the single pipeline; historical docs/specs get left alone. `tests/` hits, if any, get updated and re-run.

- [ ] **Step 4: Run the full test suite**

Run: `docker compose exec -T trading python -m pytest tests/ -q`
Expected: pass (≈1,965 tests). Any failure traced to this branch's changes gets fixed before proceeding.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: retire paper-stack references in audit playbook and tooling"
```

---

### Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite the "Pipelines: Paper vs Prod" section**

Replace the whole section (heading + table + paragraph) with:

```markdown
## Pipeline

Since 2026-07-15 there is a single pipeline, trading a **paper** Alpaca
account (the live-money account was liquidated and its history archived —
see below). It uses the default compose file, `.env`, `db` (Postgres
`:5432`), `dashboard` (`:3000`), and `./logs`. `ALPACA_PAPER=true` is the
normal state of `.env`; the module-load cross-check still enforces
key/URL/flag agreement.

**Retired live-trading stack (2026-02 → 2026-06):** its history lives in
the dormant docker volume `algo_postgres_data` and in
`backups/prod-live-final-2026-07-15.dump` (offsite copy under
`/mnt/c/Users/<WINDOWS_USER>/pinchy-backups/`). The live Alpaca/Cloudflare
credentials are preserved in `.env.live-retired` (gitignored). The
`docker-compose.paper.yml` overlay and `paper:*` Taskfile targets were
removed; resurrect them from git history if a second tier is ever needed.

**Backups:** `task db:backup` dumps the primary DB to `backups/` and
copies it off-VHD; cron runs it nightly at 8 PM MST.
```

Fill in the actual `<WINDOWS_USER>` path from Task 1.

- [ ] **Step 2: Sweep the rest of CLAUDE.md**

Update every other paper/prod mention: the `paper:*` command references in the migration-convention paragraph (now just `task db:migrate`), the `.env.paper` mentions under Environment Variables (including the `ALGO_EXECUTOR_MODEL` example — the knob still exists but there is no `.env.paper`; note it defaults to Haiku), the dashboard row in the session-stage table (add "labeled paper trading"), and the paper-runs paragraph under Codebase Layout.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md — single paper-account pipeline, archive + backup notes"
```

---

### Task 9: End-to-end verification + PR

**Files:** none

- [ ] **Step 1: Cold-start the promoted stack**

```bash
docker compose down
docker compose up -d
sleep 8
docker compose ps
```

Expected: `db`, `trading`, `dashboard` running (no `-paper` suffixes).

- [ ] **Step 2: Verify the trading container sees the right world**

```bash
docker compose exec -T trading python -c "
import os, v2.executor
from v2.db import get_cursor  # adjust import if the cursor helper lives elsewhere (grep 'def get_cursor')
print('paper flag:', os.environ['ALPACA_PAPER'])
with get_cursor() as cur:
    cur.execute('SELECT max(session_date) FROM sessions')
    print('db max session:', cur.fetchone()[0])
"
```

Expected: `paper flag: true`, `db max session: 2026-06-26`-ish (the paper history).

- [ ] **Step 3: Lint + full suite one last time**

```bash
task lint
docker compose exec -T trading python -m pytest tests/ -q
```

Expected: both clean.

- [ ] **Step 4: Tear down to resting state**

Run: `docker compose down`
Expected: no algo containers; next cron fire brings everything up via run-docker.sh.

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin promote-paper-to-prod
gh pr create --title "Promote paper stack to primary pipeline" --body "$(cat <<'EOF'
Live account liquidated (hiatus, 2026-07-15); the paper pipeline becomes the
primary, unprefixed stack per the approved design spec.

- db volume re-pointed to the paper volume (no data copy); live DB archived
  (pg_dump local + off-VHD) and dormant
- .env now carries the paper Alpaca account + Cloudflare publishing creds;
  .env.paper and docker-compose.paper.yml retired
- public dashboard publishes again, labeled paper trading (banner in shell)
- crontab: one daily session line via run-docker.sh, nightly db:backup,
  ALGO_ALERT_WEBHOOK_URL stub; installed == repo again
- Taskfile paper:* targets removed; db:backup added
- audit playbook + CLAUDE.md updated for the single pipeline

Spec: docs/superpowers/specs/2026-07-15-promote-paper-to-prod-design.md
Plan: docs/superpowers/plans/2026-07-15-promote-paper-to-prod.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Deliberately NOT in this plan:** a `--dry-run` smoke session (audit finding A.11: dry-run persists phantom decision rows into the learning DB). The first real session runs on the next market day's cron — watch it via `docker compose logs -f trading` or `logs/session.log`. The first publish will put paper data on the public dashboard; if you want to eyeball it first, run `task dashboard:publish` manually after reviewing locally.
