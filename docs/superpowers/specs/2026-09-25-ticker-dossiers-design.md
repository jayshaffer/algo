# Ticker Dossiers — Design Spec

**Date:** 2026-09-25
**Status:** Draft, pending operator review. The decisions log (section 2)
was agreed in conversation and section 3 was presented; sections 4–10 are
proposed defaults, with **[review]** marking choices made without discussion.
**Scope:** Phase 1 of fundamental deep research — a per-ticker research
artifact ("dossier") built by a separate job, consumed by the strategist
for thesis validation and confirmation. Ideation (screening for new
tickers) is a separate, later spec.

## 1. Why

The strategist is told to "apply your knowledge of company fundamentals"
and has only `web_search`. Whatever fundamentals it uses are unstored,
unreproducible and unmeasured. Every signal the system attributes today
is news-derived. This spec adds a fundamental signal that is stored,
dated, point-in-time correct, and scoreable by attribution.

Known risks this design is shaped around:

- **Horizon mismatch.** Fundamentals move quarterly; the system churns
  daily (GOOGL reversed 11 times in 22 days). Dossiers only pay off if
  they slow the churn, not if they decorate it. The evaluation (section 7)
  measures this rather than assuming it.
- **Economics.** Lifetime gross P&L is about −$35 against $150–250 of API
  spend. Dossiers are built on events, not daily, capped per run, and are
  intended to run on the subscription backend.
- **Advisory gates get argued past.** The confirmation gate is advisory
  (operator decision), so the design records dossier state at decision
  time in code, independent of what the strategist chooses to cite.

## 2. Decisions log (agreed 2026-09-25)

| # | Decision | Rejected alternatives |
|---|---|---|
| D1 | Uses: thesis validation + confirmation now; ideation later, own spec | All three at once |
| D2 | Backend: Claude subscription via headless `claude -p`; builder is backend-agnostic | API-only; hybrid Haiku/Sonnet |
| D3 | Confirmation gate is **advisory**; code records dossier state at decision time | Hard block; override-with-citation |
| D4 | Hard data from SEC EDGAR only (+ `web_search` in investigation); adapter allows a paid source later | EDGAR + FMP/Finnhub; web-only |
| D5 | Pre-built dossiers + request queue; new tickers get researched before the *next* session | Synchronous deep dive inside the strategist loop |
| D6 | Separate cron job under `cron-wrap.sh`, not a session stage | Stage 1.5 |
| D7 | Packet first, then a bounded investigation; `trajectory`/`valuation` computed by code | One-shot packet call; free-roaming agent |

## 3. Components and data flow

```
06:00 weekdays: cron-wrap.sh --instance paper paper-dossiers task dossiers INSTANCE=paper
  └─ python -m v2.dossiers.run
       1. targets.select()   held ∪ active-thesis tickers that are stale,
                             ∪ pending dossier_requests; capped at N
       2. per ticker (failures isolated):
            a. edgar.fetch    companyfacts, submissions index, filing text   (disk-cached)
            b. packet.build   trends, valuation percentile, insiders, excerpts,
                              code-computed trajectory + valuation
            c. researcher.run packet + tools → dossier JSON (backend: cli | api)
            d. schema.validate → INSERT dossiers; resolve request rows
       3. dossier_runs row: status, per-ticker results, usage

12:30 paper session (existing):
  strategist pre-seeded context  ← "Dossiers" section (one line per ticker)
  tools: get_dossier(ticker), request_dossier(ticker, reason)
  INSERT theses / buy decisions  → code writes dossier_links snapshot
```

**[review] Schedule is 06:00, not 11:30 as first discussed.** Filings land
after the close and overnight, so 06:00 MST (08:00 ET) catches them before
the open. It also keeps the job out of the pilot's 10:30–13:30 window,
where the strategist pilot draws on the same subscription usage.

### Modules (`v2/dossiers/`)

| Module | Responsibility | LLM | Depends on |
|---|---|---|---|
| `edgar.py` | SEC client: ticker→CIK map, `companyfacts`, submissions index, fetching a filing section. 10 req/s limit, `User-Agent` from `ALGO_SEC_USER_AGENT`, on-disk cache under `LOGS_DIR/edgar-cache` | no | `requests` |
| `packet.py` | Raw facts → typed `Packet` dataclass. Computes `trajectory` and `valuation` | no | `edgar`, `market_data` (prices) |
| `targets.py` | Which tickers this run, in priority order | no | DB |
| `researcher.py` | `ResearchBackend` protocol; `ClaudeCliBackend`, `ApiBackend`; prompt assembly; one repair retry on invalid JSON | yes | backends |
| `schema.py` | Dossier JSON schema, validation, verdict enums | no | — |
| `store.py` | All dossier SQL (insert, latest-per-ticker, requests, links, runs) | no | `get_cursor` |
| `run.py` | Orchestration, per-ticker isolation, exit status | no | all above |

`edgar.py` exposes a small `FundamentalsSource` interface so a paid
provider (estimates, revisions) can be added later without touching
`packet.py`'s consumers.

## 4. The dossier

### 4.1 Packet (code, deterministic)

- Last 12 quarters + 5 fiscal years: revenue, gross/operating margin, FCF,
  total debt, cash, diluted shares — from `companyfacts`, each value
  carrying its `filed` date. **Only facts with `filed <= as_of` are used**
  (point-in-time correctness; no look-ahead in later analysis).
- Derived: YoY and QoQ growth, margin deltas, net-debt/FCF, share count
  change (dilution/buybacks).
- Valuation: trailing P/S, P/FCF, EV/EBIT from price × diluted shares,
  plus each metric's percentile against the ticker's own 5-year history.
- Filings: index of the last 180 days (10-K, 10-Q, 8-K with item codes,
  Form 4 net insider buys/sells).
- Excerpts: Risk Factors + MD&A from the latest 10-K/10-Q, truncated to
  a fixed token budget **[review: 12k tokens]**.

### 4.2 Code-computed verdict fields

- `trajectory`: `improving | flat | deteriorating | unknown` from TTM
  revenue growth and operating-margin delta vs. one year ago, by fixed
  thresholds in `packet.py` **[review: thresholds set in the plan with
  tests; e.g. improving = revenue growth > 5% and margin delta ≥ 0]**.
- `valuation`: `cheap | fair | rich | unknown` from the median percentile
  of the three multiples (< 25 cheap, > 75 rich).

`unknown` when the data is insufficient (recent IPO, no filer, negative
denominators). ETFs and non-SEC filers get no dossier; `targets.py`
records them as `no_filer` and does not retry for 30 days.

### 4.3 Investigation (LLM)

The researcher gets the packet and must, in order:

1. Write 2–4 `open_questions` the packet raises.
2. Investigate within budget **[review: max 10 turns, max 5 web searches]**
   using `read_filing_section(accession, section)` and `web_search`.
3. Emit the dossier JSON.

### 4.4 Dossier JSON (validated by `schema.py`)

```json
{
  "ticker": "NVDA",
  "as_of": "2026-09-25",
  "quality": "strong | ok | weak",
  "red_flags": [{"flag": "...", "evidence": "<citation id>"}],
  "open_questions": [{"question": "...", "finding": "...", "evidence": ["<citation id>"]}],
  "bull_case": "...",
  "bear_case": "...",
  "what_breaks_it": "...",
  "summary": "<= 400 words, markdown",
  "citations": [{"id": "c1", "kind": "packet | filing | url", "ref": "..."}]
}
```

Every `red_flags[].evidence` and `open_questions[].evidence` must resolve
to a `citations[].id`; unresolved references fail validation. `trajectory`
and `valuation` are **not** in the LLM output — `run.py` merges them from
the packet so the LLM cannot override them.

## 5. Storage

New tables, each with a `db/init/041..044_*.sql` file and a
`db/migrations/017..020_*.sql` mirror (idempotent, headers name their
counterpart, per the mirror convention):

- **`dossiers`**: `id`, `ticker`, `as_of`, `built_at`, `trajectory`,
  `valuation`, `quality`, `red_flags jsonb`, `body jsonb` (full validated
  JSON), `packet jsonb`, `filings_basis text[]` (accession numbers),
  `backend`, `model`, `cli_version`, `usage jsonb`, `run_id`.
  Index `(ticker, built_at desc)`.
- **`dossier_requests`**: `id`, `ticker`, `reason`, `requested_at`,
  `session_id`, `status` (`pending | done | failed | no_filer`),
  `dossier_id`, `resolved_at`. Partial unique index on `ticker` where
  `status = 'pending'` (dedupe).
- **`dossier_runs`**: `id`, `started_at`, `finished_at`, `status`
  (`completed | partial | failed`), `backend`, `targets jsonb`,
  `results jsonb`, `error`.
- **`dossier_links`**: `id`, `entity_type` (`thesis | decision`),
  `entity_id`, `ticker`, `dossier_id` (nullable — "no dossier existed"
  is a recorded state), `quality`, `trajectory`, `valuation`,
  `red_flag_count`, `dossier_age_days`, `recorded_at`.

Links are **not** written to `decision_signals`. That table records what
the LLM cited; a link records what was known regardless of citation.
Mixing them would change the meaning of existing attribution rows.

## 6. Integration with the session

### 6.1 Target selection and staleness

A ticker is stale and gets rebuilt when any of:

- no dossier exists;
- a 10-K, 10-Q, or 8-K item 2.02 (results) filed after the latest
  dossier's `filings_basis`;
- the latest dossier is older than **[review: 30 days]**.

Priority: held positions → active theses → queue (oldest first). Cap
`ALGO_DOSSIER_MAX_PER_RUN` **[review: default 5]**. Queue rows over the
cap stay `pending` for the next run.

### 6.2 Strategist

- New pre-seeded context section `Dossiers`, added to
  `_build_pre_seeded_context` in `v2/ideation_claude.py`: one line per
  held/thesis ticker — `TICKER quality/trajectory/valuation, N red flags,
  age Nd` — or `no dossier`.
- New tools in the strategist registry (`v2/tools.py`), which the pilot's
  MCP server will expose automatically: `get_dossier(ticker)` returns the
  latest dossier's summary, red flags and code fields;
  `request_dossier(ticker, reason)` inserts a queue row (idempotent).
- Prompt addition (advisory, per D3): when opening a thesis or buying,
  consult the dossier; if none exists, request one.

### 6.3 Snapshot recording

`insert_thesis` and the buy path of `insert_decision` in
`v2/database/trading_db.py` call `store.record_link(...)` after their
insert commits, on a separate cursor. A link failure is logged and
swallowed; it never rolls back or blocks the thesis or decision — the
money path must not depend on the research subsystem.

## 7. Evaluation

A `dossier_attribution` report (new function in `v2/attribution.py`,
surfaced on the dashboard) groups closed decisions by the link's
`quality`, `trajectory` and `valuation` at entry, including
`no dossier`, and reports count, win rate, and mean `outcome_7d` / `outcome_30d` per
bucket (the columns `backfill.py` already fills).

It answers the question behind D3: do trades opened against a negative
dossier do worse? **Sample size is the binding constraint**: paper opens
few positions per week, so expect around 8–12 weeks before any bucket
means anything. Promotion of the gate (to override-with-citation or hard
block) is a later decision made on this report, not in this spec.

## 8. Failure handling

| Failure | Behavior |
|---|---|
| EDGAR error / timeout for a ticker | Ticker `failed` in run results; others continue |
| Invalid dossier JSON | One repair retry with the validation errors; then `failed` |
| Backend `is_error`, `[auth]`, or `[usage_limit]` | Stop the run (don't hammer); remaining tickers stay pending; run `partial` |
| All targets failed (targets > 0) | Run `failed`, exit nonzero → cron-wrap alert |
| Some succeeded | Run `partial`, exit 0 |
| Job didn't run | Session uses existing dossiers; age is visible in context |

`ClaudeCliBackend` follows the pilot's verified recipe: never `--bare`,
`env -i HOME PATH`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`,
`--setting-sources ""`, `--system-prompt`, scratch cwd, and success is
read from `is_error` in the JSON, never the exit code.

## 9. Prerequisites and rollout

1. **Backend.** `ClaudeCliBackend` reuses the isolated runner from the
   strategist pilot (`docs/superpowers/plans/2026-09-22-strategist-pilot.md`).
   The dossier cron is not installed until that runner is merged, or the
   operator opts into `ALGO_DOSSIER_BACKEND=api` with a budget.
2. **Config.** `ALGO_SEC_USER_AGENT` (operator supplies contact email),
   `ALGO_DOSSIER_BACKEND` (`cli | api`), `ALGO_DOSSIER_MODEL`
   **[review: default `claude-sonnet-5`]**, `ALGO_DOSSIER_MAX_PER_RUN`.
3. **Paper only.** `live` stays halted and gets no dossier job. Sharing
   dossiers across instances is deferred until live returns.

## 10. Testing

- `edgar.py`: recorded `companyfacts` / submissions fixtures (2–3 real
  tickers, trimmed), mocked HTTP; never touches the network.
- `packet.py`: pure functions; point-in-time filter tested with facts
  filed after `as_of`; threshold boundaries for `trajectory`/`valuation`.
- `schema.py`: citation resolution, enum validation, code fields rejected
  if present in LLM output.
- `researcher.py`: fake backend; repair-retry path; `is_error` handling.
- `run.py`: per-ticker isolation, run status/exit code matrix from
  section 8.
- `store.py` / session integration: new DB-touching import sites added to
  `_SESSION_DB_PATCH_TARGETS`; any new session-reachable network call added
  to `_NETWORK_PATCH_TARGETS` in `tests/v2/conftest.py`.
- Schema mirror: `tests/test_schema_mirror.py` and `db/check_mirror.sh`
  must pass with the four new table pairs.

## 11. Out of scope

- Ideation / universe screening (next spec, after section 7 shows signal).
- Enforced confirmation gate.
- Paid fundamentals data (estimates, revisions, surprises).
- Live instance; cross-instance dossier sharing.
- Dashboard pages beyond the attribution report and a dossier list.
