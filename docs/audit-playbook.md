# Audit Playbook

This file is read by the Claude Code /loop session on each 24h tick. It is the single source of truth for what gets audited. Edit this file to change the audit; no code change required.

> **Spec:** `docs/superpowers/specs/2026-05-12-audit-loop-mcp-design.md`

## How to read this file

This file is the catalog of audit checks. The orchestration — filing rules, the per-invocation cap, and Phase B execution — lives in the slash commands:

- `.claude/commands/audit-discover.md` — Phase A: read this catalog, run the checks, file findings as Jira tickets (5-create cap)
- `.claude/commands/audit-execute.md` — Phase B: work approved/apply audit tickets from Jira (2-per-invocation cap)
- `.claude/commands/audit-tick.md` — runs both phases in sequence; this is what the `/loop` invokes
- `.claude/commands/audit-rehearse.md` — dry-run Phase A with writes stubbed

When this catalog is being consumed by `/audit-discover` (or its rehearsal):

1. Execute every entry under **Deterministic checks** in order.
2. Run the **Ideation pass** per its instructions.

## Environment

Pick the docker service based on each check's declared `env`:
- `prod` -> `docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"`
- `paper` -> `docker compose -f docker-compose.yml -f docker-compose.paper.yml exec -T db-paper psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"`
- `both` -> run twice (once each env). File env in the ticket title prefix: `[audit:<category>:<env>] <Title>`.

The `$POSTGRES_USER` and `$POSTGRES_DB` come from `.env` and `.env.paper`; they're already in the trading container's environment.

## Deterministic checks

### ORPHAN_FK_NEWS_SIGNAL

- **env:** both
- **severity:** warn
- **category:** integrity
- **worktype:** db
- **topic_slug:** orphan-fk-news-signal
- **title_template:** "{n} orphan news_signal reference(s) in decision_signals"
- **sql:**
  ```sql
  SELECT DISTINCT signal_id FROM decision_signals
  WHERE signal_type='news_signal'
    AND signal_id NOT IN (SELECT id FROM news_signals);
  ```
- **finding_when:** "rows returned"
- **body_template:** "`decision_signals` rows reference `news_signal` ids that no longer exist. Filtered downstream but pollute schema. Auto-fix deletes the orphaned rows. Affected signal_ids: {signal_id_list}."
- **suggested_fix:**
  - **sql:** `DELETE FROM decision_signals WHERE signal_type = 'news_signal' AND signal_id IN (SELECT signal_id FROM decision_signals WHERE signal_type='news_signal' AND signal_id NOT IN (SELECT id FROM news_signals));`
  - **dry_run_probe:** `SELECT count(*) FROM decision_signals WHERE signal_type = 'news_signal' AND signal_id NOT IN (SELECT id FROM news_signals);`

### ORPHAN_FK_MACRO_SIGNAL

- **env:** both
- **severity:** warn
- **category:** integrity
- **worktype:** db
- **topic_slug:** orphan-fk-macro-signal
- **title_template:** "{n} orphan macro_signal reference(s) in decision_signals"
- **sql:**
  ```sql
  SELECT DISTINCT signal_id FROM decision_signals
  WHERE signal_type='macro_signal'
    AND signal_id NOT IN (SELECT id FROM macro_signals);
  ```
- **finding_when:** "rows returned"
- **body_template:** "`decision_signals` rows reference `macro_signal` ids that no longer exist. Filtered downstream but pollute schema. Auto-fix deletes the orphaned rows. Affected signal_ids: {signal_id_list}."
- **suggested_fix:**
  - **sql:** `DELETE FROM decision_signals WHERE signal_type = 'macro_signal' AND signal_id IN (SELECT signal_id FROM decision_signals WHERE signal_type='macro_signal' AND signal_id NOT IN (SELECT id FROM macro_signals));`
  - **dry_run_probe:** `SELECT count(*) FROM decision_signals WHERE signal_type = 'macro_signal' AND signal_id NOT IN (SELECT id FROM macro_signals);`

### ORPHAN_FK_THESIS

- **env:** both
- **severity:** warn
- **category:** integrity
- **worktype:** db
- **topic_slug:** orphan-fk-thesis
- **title_template:** "{n} orphan thesis reference(s) in decision_signals"
- **sql:**
  ```sql
  SELECT DISTINCT signal_id FROM decision_signals
  WHERE signal_type='thesis'
    AND signal_id NOT IN (SELECT id FROM theses);
  ```
- **finding_when:** "rows returned"
- **body_template:** "`decision_signals` rows reference `thesis` ids that no longer exist. Filtered downstream but pollute schema. Auto-fix deletes the orphaned rows. Affected signal_ids: {signal_id_list}."
- **suggested_fix:**
  - **sql:** `DELETE FROM decision_signals WHERE signal_type = 'thesis' AND signal_id IN (SELECT signal_id FROM decision_signals WHERE signal_type='thesis' AND signal_id NOT IN (SELECT id FROM theses));`
  - **dry_run_probe:** `SELECT count(*) FROM decision_signals WHERE signal_type = 'thesis' AND signal_id NOT IN (SELECT id FROM theses);`

### BACKFILL_GAP_7D

- **env:** both
- **severity:** warn
- **category:** integrity
- **worktype:** code
- **topic_slug:** backfill-gap-7d
- **title_template:** "{n} decision(s) missing 7d outcome/benchmark backfill"
- **sql:**
  ```sql
  SELECT id FROM decisions
   WHERE action IN ('buy','sell')
     AND date <= (now()::date - INTERVAL '7 days')::date
     AND (outcome_7d IS NULL OR benchmark_7d IS NULL)
   ORDER BY id;
  ```
- **finding_when:** "rows returned"
- **body_template:** "Decisions older than 7 trading days have NULL outcome_7d or benchmark_7d. Auto-fix invokes `backfill_decision_outcomes` for each. Affected decision_ids: {decision_ids}. Note: severity escalates to critical when count > 25 (warn otherwise); fix path is the same."
- **suggested_fix:** "For each decision_id in the finding, run `docker compose exec -T trading python -m v2.backfill --decision-id <ID>`. If this recurs frequently, promote to a Taskfile target."

### BACKFILL_GAP_30D

- **env:** both
- **severity:** warn
- **category:** integrity
- **worktype:** code
- **topic_slug:** backfill-gap-30d
- **title_template:** "{n} decision(s) missing 30d outcome/benchmark backfill"
- **sql:**
  ```sql
  SELECT id FROM decisions
   WHERE action IN ('buy','sell')
     AND date <= (now()::date - INTERVAL '30 days')::date
     AND (outcome_30d IS NULL OR benchmark_30d IS NULL)
   ORDER BY id;
  ```
- **finding_when:** "rows returned"
- **body_template:** "Decisions older than 30 trading days have NULL outcome_30d or benchmark_30d. Auto-fix invokes `backfill_decision_outcomes` for each. Affected decision_ids: {decision_ids}. Note: severity escalates to critical when count > 25 (warn otherwise); fix path is the same."
- **suggested_fix:** "For each decision_id in the finding, run `docker compose exec -T trading python -m v2.backfill --decision-id <ID>`. If this recurs frequently, promote to a Taskfile target."

### INVALID_ATTRIBUTION_CATEGORY

- **env:** both
- **severity:** critical
- **category:** integrity
- **worktype:** code
- **topic_slug:** invalid-attribution-category
- **title_template:** "{n} invalid attribution category value(s)"
- **sql:**
  ```sql
  SELECT DISTINCT category FROM signal_attribution;
  ```
- **finding_when:** "any returned category is NOT one of the meta-marker literals (`signal_gap`, `rule_gate`, `thesis`) AND does not have a `news_signal:<VALID_TICKER_CATEGORIES>` / `macro_signal:<VALID_MACRO_CATEGORIES>` shape"
- **body_template:** "`signal_attribution` contains categories outside the classifier's valid enums. Returned categories: {categories}. Compare against the classifier's VALID_TICKER_CATEGORIES / VALID_MACRO_CATEGORIES enums (in `v2/classifier.py`) and file a ticket for any unexpected entries. Allowed shapes: literal `thesis`, `signal_gap`, or `rule_gate` (the latter two are deliberate observability markers emitted by `v2/trader.py`); `news_signal:<VALID_TICKER_CATEGORIES>`; `macro_signal:<VALID_MACRO_CATEGORIES>`. Anything else implies a classifier regression or a direct DB write."
- **suggested_fix:** "Inspect `v2/classifier.py` for recent changes to VALID_TICKER_CATEGORIES / VALID_MACRO_CATEGORIES. If a category was renamed/removed, write a migration to remap the offending `signal_attribution` rows; if a regression introduced a new category that was never registered, restore the enum entry and re-classify."

### SNAPSHOT_GAP

- **env:** prod
- **severity:** warn
- **category:** integrity
- **worktype:** code
- **topic_slug:** snapshot-gap
- **title_template:** "{n} trading day(s) missing account_snapshot in last 30d"
- **sql:**
  ```sql
  WITH d AS (
      SELECT generate_series(now()::date - 30, now()::date - 1, '1 day')::date AS day
  )
  SELECT d.day FROM d
  LEFT JOIN account_snapshots a ON a.date=d.day
  WHERE a.date IS NULL;
  ```
- **finding_when:** "any returned day is a trading day (per NYSE calendar) with no snapshot row"
- **body_template:** "`account_snapshots` has gaps on trading days, breaking equity-curve and daily-snapshot dashboards. Raw missing dates from SQL: {missing_dates}. Filter against NYSE trading-day calendar (`v2/market_calendar.py`) and investigate snapshot stage. Common cause: Alpaca historical retrieval failed silently or the stage was skipped."
- **suggested_fix:** "In `v2/dashboard_publish.py` (or wherever `account_snapshots` is written), audit the daily snapshot stage for silent failures. Consider adding a retry around the Alpaca call and raising on persistent failure rather than continuing."

### DECISION_EQUITY_DRIFT

- **env:** prod
- **severity:** critical
- **category:** integrity
- **worktype:** code
- **topic_slug:** decision-equity-drift
- **title_template:** "{n} decision(s) with account_equity drifted from snapshot"
- **sql:**
  ```sql
  SELECT d.id,
         d.account_equity AS decision_equity,
         a.portfolio_value AS snapshot_equity,
         (d.account_equity - a.portfolio_value) AS delta
  FROM decisions d
  JOIN account_snapshots a ON a.date = d.date
  WHERE d.action IN ('buy','sell')
    AND d.date > now()::date - 60
    AND ABS(COALESCE(d.account_equity, 0) - a.portfolio_value) > 100;
  ```
- **finding_when:** "rows returned"
- **body_template:** "Decisions in last 60 days have `account_equity` differing from same-day `account_snapshots.portfolio_value` by > $100. Suggests stale snapshot logging (P0.4 regression). Max delta observed: ${max_delta}. Affected decision_ids: {decision_ids}."
- **suggested_fix:** "Audit the stage that stamps `decisions.account_equity` (likely `v2/trader.py` or `v2/executor.py`). It should read the live Alpaca account equity at decision time, not a cached or earlier-stage value. Verify the call site against the snapshot stage in `v2/session.py`."

### ATTRIBUTION_COVERAGE_LOW

- **env:** prod
- **severity:** warn
- **category:** coverage
- **worktype:** code
- **topic_slug:** attribution-coverage-low
- **title_template:** "Only {n} attribution categories with n_30d>=3"
- **sql:**
  ```sql
  SELECT category, COALESCE(sample_size_30d,0) AS sample_size_30d
  FROM signal_attribution;
  ```
- **finding_when:** "count of rows with sample_size_30d >= 3 is < 5"
- **body_template:** "`signal_attribution` has fewer than the expected number of populated categories. Threshold: >= 5 with sample_size_30d >= 3. Qualifying categories: {qualifying_categories}. All categories observed: {all_categories}. Likely cause: signal_refs wiring (strategist -> playbook -> executor) has regressed and decisions are landing without signal citations."
- **suggested_fix:** "Trace the signal_refs path end-to-end: `v2/ideation_claude.py` (strategist create_thesis/add_signal_refs), playbook write in `v2/tools.py`, and executor decision recording in `v2/agent.py` / `v2/trader.py`. Reference the 2026-05-02 wiring fix; any new regression should be reverted or repaired."

### STAGE_FAILURE_RATE

- **env:** prod
- **severity:** warn
- **category:** health
- **worktype:** code
- **topic_slug:** stage-failure-rate
- **title_template:** "{n} stage(s) with failure rate >= 20% in last 30d"
- **sql:**
  ```sql
  SELECT stage_name,
         COUNT(*) FILTER (WHERE status='completed') AS completed,
         COUNT(*) FILTER (WHERE status='failed') AS failed
  FROM session_stages
  WHERE started_at > now() - interval '30 days'
  GROUP BY stage_name;
  ```
- **finding_when:** "any stage with (completed+failed) >= 3 AND failed/(completed+failed) >= 0.20"
- **body_template:** "Stages with elevated failure rates (last 30d). See per-stage rates: {stages}. Escalates to critical (vs warn) if any single stage rate >= 0.50. Investigate stage-specific exceptions in the trading container logs."
- **suggested_fix:** "Identify the failing stage from the finding and inspect recent logs: `docker compose logs trading | grep <stage>`. Common causes: upstream API errors (Alpaca, Anthropic), DB connection drops, prompt regressions. Fix the underlying bug or add a retry where appropriate."

### STAGE_RUNNING_STALE

- **env:** prod
- **severity:** warn
- **category:** health
- **worktype:** code
- **topic_slug:** stage-running-stale
- **title_template:** "{n} session_stage row(s) stuck in 'running' >24h"
- **sql:**
  ```sql
  SELECT id, stage_name FROM session_stages
  WHERE status='running' AND started_at < now() - interval '24 hours';
  ```
- **finding_when:** "rows returned"
- **body_template:** "session_stages rows never marked completed/failed; orphan from interrupted runs. Affected: {stage_ids} ({stage_names}). Usually indicates the trading container was killed mid-stage and didn't unwind cleanly."
- **suggested_fix:** "Audit the stage completion path in `v2/session.py` — wrap stage execution in a try/finally that marks the row failed on uncaught exception. As a one-off cleanup, the human can manually `UPDATE session_stages SET status='failed' WHERE id IN (...)` after confirming the runs are truly dead."

### COST_TREND_SPIKE

- **env:** prod
- **severity:** info
- **category:** cost
- **worktype:** code
- **topic_slug:** cost-trend-spike
- **title_template:** "{n} stage(s) with token usage >=2x prior 7-day window"
- **sql:**
  ```sql
  WITH recent AS (
      SELECT stage_name,
             SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)
                +COALESCE(cache_creation_tokens,0)
                +COALESCE(cache_read_tokens,0)) AS tok
      FROM session_stages
      WHERE started_at > now() - interval '7 days' AND status='completed'
      GROUP BY stage_name
  ),
  prior AS (
      SELECT stage_name,
             SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)
                +COALESCE(cache_creation_tokens,0)
                +COALESCE(cache_read_tokens,0)) AS tok
      FROM session_stages
      WHERE started_at > now() - interval '14 days'
        AND started_at <= now() - interval '7 days'
        AND status='completed'
      GROUP BY stage_name
  )
  SELECT COALESCE(r.stage_name, p.stage_name) AS stage_name,
         COALESCE(r.tok, 0) AS recent_tok,
         COALESCE(p.tok, 0) AS prior_tok
  FROM recent r FULL OUTER JOIN prior p ON r.stage_name = p.stage_name;
  ```
- **finding_when:** "any stage with prior_tok > 0 AND recent_tok >= 2 * prior_tok"
- **body_template:** "Per-stage 7-day-rolling token totals doubled vs. prior 7-day window. Affected: {stages}. Likely cause: prompt growth or cache regression (a moved/removed `ephemeral` breakpoint silently doubles cost)."
- **suggested_fix:** "Bisect recent commits to the affected stage's module(s) for prompt or cache-breakpoint changes. Restore the ephemeral cache breakpoint or trim the prompt as appropriate. Cross-check against CACHE_HIT_RATIO_DEGRADATION findings — they often co-fire."

### DECISIONS_NO_SIGNAL_REFS

- **env:** prod
- **severity:** critical
- **category:** quality
- **worktype:** code
- **topic_slug:** decisions-no-signal-refs
- **title_template:** "{genuinely_missing} of {total} recent buy/sell decisions derived from ideation theses have no signal_refs"
- **sql:**
  ```sql
  -- Summary check (run first):
  WITH classified AS (
    SELECT
      d.id,
      CASE
        WHEN ds.decision_id IS NOT NULL                       THEN 'has_refs'
        WHEN COALESCE(d.is_off_playbook, false)               THEN 'excluded_off_playbook'
        WHEN d.playbook_action_id IS NULL OR pa.thesis_id IS NULL
                                                              THEN 'excluded_no_thesis'
        WHEN t.source = 'adoption'                            THEN 'excluded_adoption'
        ELSE 'genuinely_missing'
      END AS bucket
    FROM decisions d
    LEFT JOIN (SELECT DISTINCT decision_id FROM decision_signals) ds
      ON ds.decision_id = d.id
    LEFT JOIN playbook_actions pa ON pa.id = d.playbook_action_id
    LEFT JOIN theses t            ON t.id  = pa.thesis_id
    WHERE d.action IN ('buy','sell')
      AND d.date > now()::date - 30
  )
  SELECT
    COUNT(*)                                                 AS total,
    COUNT(*) FILTER (WHERE bucket='has_refs')                AS has_refs,
    COUNT(*) FILTER (WHERE bucket='excluded_off_playbook')   AS excluded_off_playbook,
    COUNT(*) FILTER (WHERE bucket='excluded_no_thesis')      AS excluded_no_thesis,
    COUNT(*) FILTER (WHERE bucket='excluded_adoption')       AS excluded_adoption,
    COUNT(*) FILTER (WHERE bucket='genuinely_missing')       AS genuinely_missing
  FROM classified;

  -- If genuinely_missing > 0, run this for evidence:
  SELECT d.id
    FROM decisions d
    LEFT JOIN (SELECT DISTINCT decision_id FROM decision_signals) ds
      ON ds.decision_id = d.id
    LEFT JOIN playbook_actions pa ON pa.id = d.playbook_action_id
    LEFT JOIN theses t            ON t.id  = pa.thesis_id
   WHERE d.action IN ('buy','sell')
     AND d.date > now()::date - 30
     AND ds.decision_id IS NULL
     AND COALESCE(d.is_off_playbook, false) = false
     AND d.playbook_action_id IS NOT NULL
     AND pa.thesis_id IS NOT NULL
     AND COALESCE(t.source, '') <> 'adoption'
   ORDER BY d.id;
  ```
- **finding_when:** "summary.genuinely_missing > 0"
- **body_template:** "Strategist created theses without citing signals via `signal_refs`/`add_signal_refs`, so executor decisions derived from them have nothing to attribute. Summary: total={total}, has_refs={has_refs}, excluded_off_playbook={excluded_off_playbook}, excluded_no_thesis={excluded_no_thesis}, excluded_adoption={excluded_adoption}, genuinely_missing={genuinely_missing}. Affected decision_ids: {decision_ids}. Escalates to critical when genuinely_missing/total > 0.10 (warn otherwise). Fix in v2/ideation_claude.py prompt or the create_thesis / update_thesis tool-call validation."
- **suggested_fix:** "In `v2/ideation_claude.py`, strengthen the strategist's prompt to require `signal_refs` on every new thesis. Alternatively, add validation in `v2/tools.py` create_thesis/update_thesis handlers to reject calls without signal_refs (when the thesis is not adoption-sourced)."

### DECISIONS_OFF_PLAYBOOK_NO_SIGNAL_REFS

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** decisions-off-playbook-no-signal-refs
- **title_template:** "{missing} of {total} off-playbook buy/sell decisions in last 30d have no signal_refs"
- **sql:**
  ```sql
  WITH off_playbook AS (
    SELECT
      d.id,
      ds.decision_id IS NULL AS missing
    FROM decisions d
    LEFT JOIN (SELECT DISTINCT decision_id FROM decision_signals) ds
      ON ds.decision_id = d.id
    WHERE d.action IN ('buy','sell')
      AND d.date > now()::date - 30
      AND COALESCE(d.is_off_playbook, false) = true
  )
  SELECT
    COUNT(*)                        AS total,
    COUNT(*) FILTER (WHERE missing) AS missing,
    array_agg(id ORDER BY id) FILTER (WHERE missing) AS missing_ids
  FROM off_playbook;
  ```
- **finding_when:** "total >= 3 AND missing >= 1"
- **body_template:** "Off-playbook decisions are pure executor judgment calls — the strategist did not approve them, so the executor must justify with `signal_refs`. The existing `DECISIONS_NO_SIGNAL_REFS` check explicitly *excludes* the off-playbook bucket, so a gap here is invisible to that check. Decisions with no `decision_signals` rows mean attribution has nothing to score against and the executor's reasoning is unverifiable post-hoc. total={total}, missing={missing}, missing_ids={missing_ids}. Escalates to critical when missing/total >= 0.50."
- **suggested_fix:** "Inspect each missing_id's `reasoning` in `decisions`. If the executor cited signals informally in prose but didn't structure them via the `signal_refs` JSON field on its response, harden the executor's prompt in `v2/agent.py` to require `signal_refs` whenever `is_off_playbook=true`. If the `signal_refs` field was populated by the LLM but got stripped (e.g., by `validate_signal_refs` in `v2/agent.py`), check the warn-level log for the stripping reason — orphaned IDs or unknown types — and either repair the source data or relax the validation. Going off-playbook without a signal trail makes attribution impossible and forfeits the learning loop."

### THESES_NO_SIGNAL_REFS

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** theses-no-signal-refs
- **title_template:** "{missing} of {total} recent ideation theses have no signal_refs"
- **sql:**
  ```sql
  -- Summary check (run first):
  SELECT
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE ts.thesis_id IS NULL) AS missing
  FROM theses t
  LEFT JOIN (SELECT DISTINCT thesis_id FROM thesis_signals) ts
    ON ts.thesis_id = t.id
  WHERE t.created_at > now() - interval '30 days'
    AND COALESCE(t.source, '') <> 'adoption';

  -- If total > 0 AND missing/total > 0.25, run this for evidence:
  SELECT t.id FROM theses t
  LEFT JOIN (SELECT DISTINCT thesis_id FROM thesis_signals) ts
    ON ts.thesis_id = t.id
  WHERE t.created_at > now() - interval '30 days'
    AND COALESCE(t.source, '') <> 'adoption'
    AND ts.thesis_id IS NULL
  ORDER BY t.id;
  ```
- **finding_when:** "total > 0 AND missing/total > 0.25"
- **body_template:** "Strategist is creating theses without citing signals, violating Rule #6 from the 2026-05-02 wiring fix. Without citations, downstream attribution receives nothing to score. Adopted theses are excluded by design. total={total}, missing={missing}. Offending thesis_ids: {thesis_ids}."
- **suggested_fix:** "In `v2/ideation_claude.py`, reinforce Rule #6 in the strategist prompt or harden `create_thesis`/`add_signal_refs` tool validation in `v2/tools.py` to refuse thesis creation without at least one signal_ref."

### STRATEGIST_NOT_USING_REVERSAL_TOOL

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** strategist-not-using-reversal-tool
- **title_template:** "Strategist not consulting prior playbooks despite round-trip evidence"
- **sql:**
  ```sql
  WITH recent_sessions AS (
      SELECT DISTINCT session_id
      FROM agent_events
      WHERE session_id IS NOT NULL
        AND occurred_at > now() - interval '14 days'
      ORDER BY session_id DESC
      LIMIT 3
  )
  SELECT s.session_id,
         SUM(CASE
               WHEN e.event_type='evidence_shown'
                AND e.payload->>'evidence_kind'='round_trips'
                AND COALESCE(jsonb_array_length(e.payload->'items'), 0) > 0
               THEN 1 ELSE 0 END) AS reversal_evidence,
         SUM(CASE
               WHEN e.event_type='tool_invocation'
                AND e.payload->>'tool_name'='get_recent_playbooks'
                AND e.stage_name='ideation'
               THEN 1 ELSE 0 END) AS lookup_calls
  FROM recent_sessions s
  LEFT JOIN agent_events e ON e.session_id = s.session_id
  GROUP BY s.session_id
  ORDER BY s.session_id DESC;
  ```
- **finding_when:** "all 3 most-recent sessions in last 14d have reversal_evidence >= 1 AND lookup_calls == 0"
- **body_template:** "3 consecutive sessions showed round-trip evidence to reflection, but the ideation stage never called `get_recent_playbooks` to compare against prior plays. The reversal-lookup tool exists for this exact case. Sessions: {sessions}."
- **suggested_fix:** "In `v2/ideation_claude.py`, update the strategist prompt to require a `get_recent_playbooks` call when reversal/round-trip evidence is present. Consider adding the rule as a system-prompt invariant rather than relying on memos."

### REFLECTION_INERT_ON_ROUND_TRIPS

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** reflection-inert-on-round-trips
- **title_template:** "Reflection not acting on round-trip evidence"
- **sql:**
  ```sql
  WITH recent_sessions AS (
      SELECT DISTINCT session_id
      FROM agent_events
      WHERE session_id IS NOT NULL
        AND occurred_at > now() - interval '21 days'
      ORDER BY session_id DESC
      LIMIT 5
  )
  SELECT s.session_id,
         SUM(CASE
               WHEN e.event_type='evidence_shown'
                AND e.payload->>'evidence_kind'='round_trips'
                AND COALESCE(jsonb_array_length(e.payload->'items'), 0) > 0
               THEN 1 ELSE 0 END) AS reversal_evidence,
         SUM(CASE
               WHEN e.event_type='tool_invocation'
                AND e.stage_name='reflection'
                AND e.payload->>'tool_name' IN ('propose_rule', 'retire_rule')
               THEN 1 ELSE 0 END) AS rule_actions
  FROM recent_sessions s
  LEFT JOIN agent_events e ON e.session_id = s.session_id
  GROUP BY s.session_id
  ORDER BY s.session_id DESC;
  ```
- **finding_when:** "all 5 most-recent sessions in last 21d have reversal_evidence >= 1 AND rule_actions == 0"
- **body_template:** "5 consecutive sessions surfaced round-trip evidence to reflection, but no rules were proposed or retired in any of them. Either the rules already cover the pattern (and reflection should retire dead rules), or reflection is ignoring the signal. Sessions: {sessions}."
- **suggested_fix:** "In `v2/strategy.py`, audit the reflection prompt and tool surface to ensure round-trip evidence is surfaced prominently and that propose_rule/retire_rule are expected outputs when evidence persists across sessions."

### TELEMETRY_INERT_WINDOW

- **env:** both
- **severity:** critical
- **category:** audit_gap
- **worktype:** code
- **topic_slug:** telemetry-inert-window
- **title_template:** "agent_events recorded zero events in last 7d despite {stages_completed_7d} completed stage(s)"
- **sql:**
  ```sql
  WITH activity AS (
    SELECT
      (SELECT COUNT(*) FROM session_stages
         WHERE status = 'completed'
           AND completed_at > now() - interval '7 days') AS stages_completed_7d,
      (SELECT COUNT(*) FROM agent_events
         WHERE occurred_at > now() - interval '7 days') AS events_recorded_7d
  )
  SELECT stages_completed_7d, events_recorded_7d
  FROM activity
  WHERE stages_completed_7d > 0
    AND events_recorded_7d = 0;
  ```
- **finding_when:** "rows returned"
- **body_template:** "`agent_events` recorded zero rows in the last 7 days even though {stages_completed_7d} session stage(s) completed successfully. This silently disables the entire telemetry-based audit catalog — TOOL_ERROR_RATE, RISK_BLOCK_*, IDEATION_TOOL_DROUGHT, EXECUTOR_*, CLASSIFIER_ERROR_RATE, AGENT_CALL_*, LOOP_*, CACHE_HIT_RATIO_DEGRADATION, COST_TREND_SPIKE, STRATEGIST_NOT_USING_REVERSAL_TOOL, REFLECTION_INERT_ON_ROUND_TRIPS — because each gate threshold (`n >= 5`, `total >= 10`, etc.) is never satisfied. The audit becomes effectively blind to quality/health/cost while still appearing to run."
- **suggested_fix:** "`v2/telemetry.py::record_event` wraps the INSERT in a broad `except Exception: logger.exception(...)` (intentional — telemetry must never break a session). Inspect the trading container logs for `\"Failed to record agent_event; continuing\"` over the last 7 days; the accompanying psycopg2 exception will name the cause (most likely schema drift in `agent_events`, a missing index, or an FK violation against `sessions`). Confirm with `\\d agent_events` against the affected DB vs `db/init/026_agent_events.sql`. If the table is missing or partial, re-apply the init script (or add a migration under `db/migrations/`). Do NOT tighten the swallowing except clause without explicit sign-off — that invariant is load-bearing for session reliability."

### TOOL_ERROR_RATE

- **env:** prod
- **severity:** warn
- **category:** health
- **worktype:** code
- **topic_slug:** tool-error-rate
- **title_template:** "{n} tool(s) with error rate >= 20% in last 7d"
- **sql:**
  ```sql
  SELECT payload->>'tool_name' AS tool_name,
         COUNT(*) AS n,
         COUNT(*) FILTER (
             WHERE COALESCE((payload->>'success')::boolean, true) = false
         ) AS errors
  FROM agent_events
  WHERE event_type = 'tool_invocation'
    AND occurred_at > now() - interval '7 days'
  GROUP BY 1;
  ```
- **finding_when:** "any tool with n >= 5 AND errors/n >= 0.20"
- **body_template:** "Tool handlers are raising exceptions. Per-tool rates: {tools}. Escalates to critical (vs warn) if any single tool rate >= 0.50."
- **suggested_fix:** "Inspect the offending tool handler in `v2/tools.py` for the named tool. Common causes: argument-validation bugs, DB schema mismatch, or upstream API errors. Add try/except with logging or fix the underlying bug."

### RISK_BLOCK_HOTSPOT

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** risk-block-hotspot
- **title_template:** "{n} ticker(s) blocked by sector-cap >= 3x in 7d"
- **sql:**
  ```sql
  SELECT payload->>'ticker' AS ticker, COUNT(*) AS n
  FROM agent_events
  WHERE event_type = 'risk_block'
    AND occurred_at > now() - interval '7 days'
  GROUP BY 1
  ORDER BY n DESC;
  ```
- **finding_when:** "any ticker with n >= 3 risk_block events in 7d"
- **body_template:** "The same ticker is being repeatedly proposed and blocked by the sector-cap gate. The strategist isn't reacting to the rejection - review whether the cap or the thesis needs to change. Tickers: {tickers}."
- **suggested_fix:** "Either tighten the strategist prompt in `v2/ideation_claude.py` to respect sector-cap rejections (advisory text already exists), or revisit the cap configuration in `v2/risk.py`. If the ticker is genuinely justified despite the cap, document the override path."

### RISK_BLOCK_BURST

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** risk-block-burst
- **title_template:** "{n} day(s) with >= 5 sector-cap rejections"
- **sql:**
  ```sql
  SELECT date_trunc('day', occurred_at)::date AS d, COUNT(*) AS n
  FROM agent_events
  WHERE event_type = 'risk_block'
    AND occurred_at > now() - interval '14 days'
  GROUP BY 1
  ORDER BY d DESC;
  ```
- **finding_when:** "any single day in last 14d has >= 5 risk_block events"
- **body_template:** "A single session generated 5+ sector-cap rejections. Either the strategist ignored advisory text, or the cap is wrong for current portfolio shape. Dates: {dates}."
- **suggested_fix:** "Inspect the dated session's ideation log to see what the strategist was attempting. If advisory text is being ignored, harden the prompt in `v2/ideation_claude.py`. If the cap is genuinely wrong for current portfolio shape, adjust `v2/risk.py` sector-cap config."

### IDEATION_TOOL_DROUGHT

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** ideation-tool-drought
- **title_template:** "{n} expected ideation tool(s) unused in last 7d"
- **sql:**
  ```sql
  -- Session count gate (run first):
  SELECT COUNT(DISTINCT session_id) AS n_sessions
  FROM agent_events
  WHERE event_type = 'tool_invocation'
    AND stage_name = 'ideation'
    AND session_id IS NOT NULL
    AND occurred_at > now() - interval '7 days';

  -- If n_sessions >= 3, run per-tool counts:
  SELECT payload->>'tool_name' AS tool_name, COUNT(*) AS n
  FROM agent_events
  WHERE event_type = 'tool_invocation'
    AND stage_name = 'ideation'
    AND occurred_at > now() - interval '7 days'
  GROUP BY 1;
  ```
- **finding_when:** ">= 3 distinct ideation sessions in last 7d AND any tool in EXPECTED_IDEATION_TOOLS (`get_portfolio_state`, `get_active_theses`, `get_news_signals`, `get_signal_attribution`, `get_recent_playbooks`, `write_playbook`) has zero invocations"
- **body_template:** "One or more tools in the expected ideation toolset went unused across the last 7 days of ideation sessions. Either prompt drift dropped them or the loop is silently skipping them - both are regressions. Missing: {missing_tools}. Observed counts: {observed_counts}."
- **suggested_fix:** "In `v2/ideation_claude.py`, confirm each tool in `EXPECTED_IDEATION_TOOLS` (`get_portfolio_state`, `get_active_theses`, `get_news_signals`, `get_signal_attribution`, `get_recent_playbooks`, `write_playbook`) is still registered and referenced in the strategist prompt. Check for recent commits that may have dropped a tool definition."

### EXECUTOR_TRUNCATION_RATE

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** executor-truncation-rate
- **title_template:** "Executor truncated on {truncated}/{total} calls (last 14d)"
- **sql:**
  ```sql
  SELECT COUNT(*) AS total,
         COUNT(*) FILTER (
           WHERE payload->>'stop_reason' = 'max_tokens'
         ) AS truncated
  FROM agent_events
  WHERE event_type = 'agent_call'
    AND payload->>'purpose' = 'executor'
    AND occurred_at > now() - interval '14 days';
  ```
- **finding_when:** "total >= 5 AND truncated > 0 AND truncated/total >= 0.10"
- **body_template:** "Executor responses hit max_tokens. Input context may be too large; check ExecutorInput field sizes. Or model max_tokens budget needs raising. total={total}, truncated={truncated}, rate={rate}. Escalates to critical when rate >= 0.25 (warn otherwise)."
- **suggested_fix:** "In `v2/executor.py`, raise the default for `ALGO_EXECUTOR_MAX_TOKENS` (currently 8192) to 16384, or trim ExecutorInput context fields in `v2/context.py`. Update the docs block in `CLAUDE.md` if the env-var default changes."

### EXECUTOR_SCHEMA_DRIFT

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** executor-schema-drift
- **title_template:** "Executor schema drift: {top_drift_rows} unparsed top-level keys, {dec_drift_rows} unparsed decision keys"
- **sql:**
  ```sql
  -- Top-level drift:
  SELECT key, COUNT(*) AS n
  FROM agent_events,
       LATERAL jsonb_array_elements_text(payload->'unknown_top_level_keys') AS t(key)
  WHERE event_type = 'executor_response'
    AND occurred_at > now() - interval '7 days'
  GROUP BY 1
  HAVING COUNT(*) >= 3;

  -- Decision-level drift:
  SELECT key, COUNT(*) AS n
  FROM agent_events,
       LATERAL jsonb_array_elements_text(payload->'unknown_decision_keys') AS t(key)
  WHERE event_type = 'executor_response'
    AND occurred_at > now() - interval '7 days'
  GROUP BY 1
  HAVING COUNT(*) >= 3;
  ```
- **finding_when:** "either query returns rows"
- **body_template:** "Executor response contains JSON fields not in our canonical key sets. Either the prompt is requesting new fields the parser doesn't handle, or the LLM is emitting drift we should either consume or suppress. Where `top_drift_rows` = row count of query 1 (distinct top-level unknown keys with count >= 3), `dec_drift_rows` = row count of query 2 (distinct decision-level unknown keys with count >= 3). Update EXECUTOR_KNOWN_*_KEYS or the parser in v2/agent.py."
- **suggested_fix:** "In `v2/agent.py`, either add the new keys to `EXECUTOR_KNOWN_TOP_LEVEL_KEYS` / `EXECUTOR_KNOWN_DECISION_KEYS` (if they should be consumed), or tighten the executor prompt to stop emitting them. If the keys carry useful data, plumb them through the parser."

### EXECUTOR_PARSE_FAILURE_RATE

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** executor-parse-failure-rate
- **title_template:** "Executor JSON parse failed on {parse_failed}/{total} non-truncated calls (last 14d)"
- **sql:**
  ```sql
  SELECT COUNT(*) AS total,
         COUNT(*) FILTER (
           WHERE (payload->>'parse_succeeded')::boolean = false
             AND payload->>'error' NOT LIKE 'max_tokens%'
         ) AS parse_failed
  FROM agent_events
  WHERE event_type = 'executor_response'
    AND occurred_at > now() - interval '14 days';
  ```
- **finding_when:** "total >= 5 AND parse_failed > 0 AND parse_failed/total >= 0.05"
- **body_template:** "Executor responses are failing JSON parse for reasons other than max_tokens truncation. Likely causes: prompt regression causing prose responses, fenced-code-block edge cases the stripper misses, or the LLM returning structured output in a different shape. total={total}, parse_failed={parse_failed}, rate={rate}. Escalates to critical when rate >= 0.15 (warn otherwise). Inspect raw_response_text_truncated on recent failures."
- **suggested_fix:** "Inspect recent executor_response rows with `parse_succeeded=false` and `error NOT LIKE 'max_tokens%'`. If prose is leaking, harden the prompt in `v2/executor.py`. If the JSON shape changed, update the parser in `v2/agent.py`."

### CLASSIFIER_ERROR_RATE

- **env:** prod
- **severity:** warn
- **category:** health
- **worktype:** code
- **topic_slug:** classifier-error-rate
- **title_template:** "{n} classifier purpose(s) with error rate >= 10% in last 7d"
- **sql:**
  ```sql
  SELECT payload->>'purpose' AS purpose,
         COUNT(*) AS total,
         COUNT(*) FILTER (WHERE COALESCE((payload->>'success')::boolean, true) = false) AS errors
  FROM agent_events
  WHERE event_type = 'agent_call'
    AND payload->>'purpose' LIKE 'classifier_%'
    AND occurred_at > now() - interval '7 days'
  GROUP BY 1
  HAVING COUNT(*) >= 10;
  ```
- **finding_when:** "any classifier_* purpose with total >= 10 AND errors/total >= 0.10"
- **body_template:** "Classifier calls failing in the pipeline stage. Investigate handler exceptions or API errors. Per-classifier rates: {classifiers}. Escalates to critical (vs warn) if any rate >= 0.25."
- **suggested_fix:** "Inspect the failing classifier purpose in `v2/classifier.py` / `v2/pipeline.py`. Common causes: Anthropic API errors (rate limit, network), input shape mismatch, or prompt regression. Add retries with backoff or fix the underlying bug."

### AGENT_CALL_ERROR_RATE_BY_PURPOSE

- **env:** prod
- **severity:** warn
- **category:** health
- **worktype:** code
- **topic_slug:** agent-call-error-rate-by-purpose
- **title_template:** "{n} agent purpose(s) with error rate >= 10% in last 7d"
- **sql:**
  ```sql
  SELECT payload->>'purpose' AS purpose,
         COUNT(*) AS total,
         COUNT(*) FILTER (WHERE COALESCE((payload->>'success')::boolean, true) = false) AS errors
  FROM agent_events
  WHERE event_type = 'agent_call'
    AND occurred_at > now() - interval '7 days'
    AND COALESCE(payload->>'purpose', '') NOT LIKE 'classifier_%'
  GROUP BY 1
  HAVING COUNT(*) >= 10;
  ```
- **finding_when:** "any non-classifier purpose with total >= 10 AND errors/total >= 0.10"
- **body_template:** "Generic per-purpose error tracking. Investigate the specific purpose's call site for handler exceptions or upstream API errors. Per-purpose rates: {purposes}. Escalates to critical (vs warn) if any rate >= 0.25."
- **suggested_fix:** "Identify the failing purpose from the finding, find its call site (`grep -rn 'purpose=\"<name>\"' v2/`), and inspect for unhandled exceptions or upstream API regressions. Fix the bug or add retry/backoff."

### LOOP_RECOVERY_BURST

- **env:** prod
- **severity:** warn
- **category:** health
- **worktype:** code
- **topic_slug:** loop-recovery-burst
- **title_template:** "{n} loop-recovery reason(s) firing >= 3 times in 7d"
- **sql:**
  ```sql
  SELECT payload->>'reason' AS reason, COUNT(*) AS n
  FROM agent_events
  WHERE event_type = 'loop_recovery'
    AND occurred_at > now() - interval '7 days'
  GROUP BY 1
  HAVING COUNT(*) >= 3;
  ```
- **finding_when:** "any loop_recovery `reason` with count >= 3 in last 7d"
- **body_template:** "`run_agentic_loop` recovery branches (max_tokens retry / context-length aggressive prune) firing often. Investigate prompt size, message history pruning, or model max_tokens. Recoveries: {recoveries}."
- **suggested_fix:** "In `v2/claude_client.py` (`run_agentic_loop`), inspect the named recovery reason. If it's max_tokens-retry, raise the model max_tokens. If it's context-length pruning, audit the message-history pruning policy."

### LOOP_MAX_TURNS_HIT

- **env:** prod
- **severity:** warn
- **category:** health
- **worktype:** code
- **topic_slug:** loop-max-turns-hit
- **title_template:** "Executor loop hit max_turns {total_n} times across stages"
- **sql:**
  ```sql
  SELECT stage_name, COUNT(*) AS n,
         array_agg(DISTINCT session_id ORDER BY session_id DESC) AS session_ids
  FROM agent_events
  WHERE event_type = 'loop_completion'
    AND payload->>'stop_reason' = 'max_turns'
    AND occurred_at > now() - interval '7 days'
  GROUP BY 1;
  ```
- **finding_when:** "total >= 1 loop_completion event with stop_reason='max_turns' in last 7d"
- **body_template:** "`run_agentic_loop` exited because it hit max_turns, not because Claude returned end_turn. Strategist or reflection didn't finish its task - playbook may be partial, rules may not have been proposed/retired. Either the prompt is asking for too much, the tool surface is too noisy, or max_turns needs raising.\n\nPer-stage breakdown:\n{rows}\n\nWhere `total_n` = sum of `n` across all rows, and `{rows}` is the SQL result rows as a list of (stage_name, n, session_ids[:5]) tuples. Sessions to investigate: see session_ids column. Escalates to critical when total_n >= 3 (warn otherwise)."
- **suggested_fix:** "In `v2/claude_client.py` (`run_agentic_loop`), either raise `max_turns` for the affected stage, trim the tool surface, or simplify the prompt so the agent can converge. Cross-check the affected sessions to see what the agent was looping on."

### CACHE_HIT_RATIO_DEGRADATION

- **env:** prod
- **severity:** info
- **category:** cost
- **worktype:** code
- **topic_slug:** cache-hit-ratio-degradation
- **title_template:** "{n} agent purpose(s) with cache_read share dropped >= 30pp vs prior 7d"
- **sql:**
  ```sql
  WITH recent AS (
      SELECT payload->>'purpose' AS purpose,
             SUM((payload->>'cache_read_tokens')::int) AS cache_read,
             SUM((payload->>'cache_creation_tokens')::int) AS cache_creation,
             SUM((payload->>'input_tokens')::int) AS input_tok,
             COUNT(*) AS n
      FROM agent_events
      WHERE event_type = 'agent_call'
        AND occurred_at > now() - interval '7 days'
        AND COALESCE((payload->>'success')::boolean, true) = true
      GROUP BY 1
  ),
  prior AS (
      SELECT payload->>'purpose' AS purpose,
             SUM((payload->>'cache_read_tokens')::int) AS cache_read,
             SUM((payload->>'cache_creation_tokens')::int) AS cache_creation,
             SUM((payload->>'input_tokens')::int) AS input_tok,
             COUNT(*) AS n
      FROM agent_events
      WHERE event_type = 'agent_call'
        AND occurred_at > now() - interval '14 days'
        AND occurred_at <= now() - interval '7 days'
        AND COALESCE((payload->>'success')::boolean, true) = true
      GROUP BY 1
  )
  SELECT r.purpose,
         r.cache_read::float / NULLIF(r.cache_read + r.cache_creation + r.input_tok, 0) AS recent_ratio,
         p.cache_read::float / NULLIF(p.cache_read + p.cache_creation + p.input_tok, 0) AS prior_ratio,
         r.n AS recent_n,
         p.n AS prior_n
  FROM recent r JOIN prior p ON p.purpose = r.purpose
  WHERE r.n >= 10 AND p.n >= 10;
  ```
- **finding_when:** "any purpose with recent_n >= 10 AND prior_n >= 10 AND (prior_ratio - recent_ratio) >= 0.30"
- **body_template:** "Cache-read token share dropped sharply for one or more purposes. Likely cause: a refactor moved/removed an `ephemeral` cache breakpoint in `cached_system` or tool definitions. This silently doubles API cost. Inspect recent changes to system prompts and tool registration. Degradations: {degradations}."
- **suggested_fix:** "Bisect recent commits to the modules emitting the flagged purpose for changes to `cached_system` or tool registration. Restore the missing `cache_control: {type: ephemeral}` breakpoint."

### AGENT_CALL_LATENCY_DRIFT

- **env:** prod
- **severity:** info
- **category:** health
- **worktype:** code
- **topic_slug:** agent-call-latency-drift
- **title_template:** "{n} agent purpose(s) with p95 latency >= 2.0x prior 7-day window"
- **sql:**
  ```sql
  WITH recent AS (
      SELECT payload->>'purpose' AS purpose,
             percentile_cont(0.95) WITHIN GROUP
                 (ORDER BY (payload->>'duration_ms')::int) AS p95,
             COUNT(*) AS n
      FROM agent_events
      WHERE event_type = 'agent_call'
        AND occurred_at > now() - interval '7 days'
        AND payload ? 'duration_ms'
      GROUP BY 1
  ),
  prior AS (
      SELECT payload->>'purpose' AS purpose,
             percentile_cont(0.95) WITHIN GROUP
                 (ORDER BY (payload->>'duration_ms')::int) AS p95,
             COUNT(*) AS n
      FROM agent_events
      WHERE event_type = 'agent_call'
        AND occurred_at > now() - interval '14 days'
        AND occurred_at <= now() - interval '7 days'
        AND payload ? 'duration_ms'
      GROUP BY 1
  )
  SELECT r.purpose, r.p95 AS recent_p95, p.p95 AS prior_p95
  FROM recent r JOIN prior p ON p.purpose = r.purpose
  WHERE r.n >= 10 AND p.n >= 10
    AND p.p95 > 0
    AND r.p95 >= 2.0 * p.p95;
  ```
- **finding_when:** "any purpose with recent_n >= 10 AND prior_n >= 10 AND prior_p95 > 0 AND recent_p95 >= 2.0 * prior_p95"
- **body_template:** "Per-purpose p95 latency in last 7d is significantly elevated vs prior 7d. Drifts: {drifts}. Common causes: upstream API regressions (Anthropic), prompt growth that bloats inference, or tool-call expansion."
- **suggested_fix:** "Inspect the flagged purpose's recent commits for prompt growth or tool-surface changes. If the regression is Anthropic-side, monitor and consider raising stage timeouts. If it's local, trim the prompt or cache more aggressively."

### LLM_CONTEXT_MESSAGE_OR_BLOCK_MALFORMED

- **env:** both
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** llm-context-message-or-block-malformed
- **title_template:** "{n} llm_call_contexts row(s) with malformed message or content block (last 7d)"
- **sql:**
  ```sql
  SELECT lcc.id, lcc.session_id, lcc.purpose, lcc.sequence
  FROM llm_call_contexts lcc,
       LATERAL jsonb_array_elements(lcc.messages) AS msg
  WHERE lcc.created_at > now() - interval '7 days'
    AND (
      NOT (msg ? 'role')
      OR NOT (msg ? 'content')
      OR (
        jsonb_typeof(msg->'content') = 'array'
        AND EXISTS (
          SELECT 1 FROM jsonb_array_elements(msg->'content') AS block
          WHERE NOT (block ? 'type')
        )
      )
    )
  GROUP BY lcc.id, lcc.session_id, lcc.purpose, lcc.sequence;
  ```
- **finding_when:** "rows returned"
- **body_template:** "`llm_call_contexts` rows contain `messages` entries missing `role`/`content` or content blocks missing `type`. Either `_serialize_content_blocks` in `v2/claude_client.py` regressed, or the writer is being fed a non-standard message shape. Affected (id, session, purpose, seq): {rows}."
- **suggested_fix:** "Inspect `v2/claude_client.py::_serialize_content_blocks` for a regression in how content blocks are serialized (e.g., pydantic shape change or a new block type). Also check the call sites passing `messages` into `_call_with_retry` — they must use the anthropic-standard `[{role, content}]` shape, with `content` either a string or an array of `{type, ...}` blocks."

### LLM_CONTEXT_RESPONSE_BLOCK_MALFORMED

- **env:** both
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** llm-context-response-block-malformed
- **title_template:** "{n} llm_call_contexts row(s) with response_content block missing `type` (last 7d)"
- **sql:**
  ```sql
  SELECT DISTINCT lcc.id, lcc.session_id, lcc.purpose
  FROM llm_call_contexts lcc,
       LATERAL jsonb_array_elements(lcc.response_content) AS block
  WHERE lcc.response_content IS NOT NULL
    AND lcc.created_at > now() - interval '7 days'
    AND NOT (block ? 'type');
  ```
- **finding_when:** "rows returned"
- **body_template:** "Response content blocks captured without a `type` field. The dashboard's `render_blocks` macro falls back to an Unknown-block path when this happens, but the root cause is that `_serialize_content_blocks` produced a shape that lost the type. Affected (id, session, purpose): {rows}."
- **suggested_fix:** "Inspect `v2/claude_client.py::_serialize_content_blocks` — confirm that anthropic SDK content blocks expose `.model_dump()` and that the dict it returns includes `type`. If the test-fixture fallback path (`vars(block)`) is leaking into production, the test-fixture and production block shapes have diverged."

### LLM_CONTEXT_MISSING_SYSTEM_PROMPT

- **env:** both
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** llm-context-missing-system-prompt
- **title_template:** "{n} captured llm_call_contexts row(s) with NULL system_prompt (last 7d)"
- **sql:**
  ```sql
  SELECT id, session_id, purpose, sequence
  FROM llm_call_contexts
  WHERE created_at > now() - interval '7 days'
    AND system_prompt IS NULL;
  ```
- **finding_when:** "rows returned"
- **body_template:** "Every captured purpose (executor, strategist_loop, reflection_loop) is expected to have a non-null `system_prompt`. NULL here usually means the writer hit a `system` value shape it didn't know how to extract text from. Past regression: strategist/reflection passed `system` as a cache-control list-of-blocks and the `isinstance(str)` guard stored NULL — fixed by extracting text from the list. Affected (id, session, purpose, seq): {rows}."
- **suggested_fix:** "Inspect `v2/claude_client.py::_record_call_context` — confirm both the `isinstance(str)` and `isinstance(list)` branches still work. If a new shape is being passed (e.g., a future cache-control wrapper or anthropic SDK variant), extend the extraction logic to handle it. Re-run a strategist loop and confirm the resulting rows have non-null `system_prompt`."

### LLM_CONTEXT_EXECUTOR_RESPONSE_NOT_JSON

- **env:** both
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** llm-context-executor-response-not-json
- **title_template:** "{n} executor row(s) whose response text doesn't start with `{{` (last 14d, non-truncated)"
- **sql:**
  ```sql
  WITH executor_text AS (
    SELECT lcc.id, lcc.session_id, lcc.stop_reason,
      string_agg(block->>'text', '' ORDER BY ord) AS response_text
    FROM llm_call_contexts lcc,
         LATERAL jsonb_array_elements(lcc.response_content) WITH ORDINALITY AS t(block, ord)
    WHERE lcc.purpose = 'executor'
      AND lcc.created_at > now() - interval '14 days'
      AND lcc.response_content IS NOT NULL
      AND block->>'type' = 'text'
    GROUP BY lcc.id, lcc.session_id, lcc.stop_reason
  )
  SELECT id, session_id, stop_reason
  FROM executor_text
  WHERE response_text IS NOT NULL
    AND COALESCE(stop_reason, '') != 'max_tokens'
    AND regexp_replace(ltrim(response_text), '^```(json)?[[:space:]]*', '', 'i') NOT LIKE '{%';
  ```
- **finding_when:** "rows returned"
- **body_template:** "The executor's response text (assembled from response_content text blocks) does not start with `{{` after stripping a leading markdown code fence (`` ``` `` or `` ```json ``), indicating the model went off-format — refused, replied in prose, or returned something the parser cannot consume. The parser in `v2/agent.py` already strips these fences before `json.loads`, so they are not counted as drift. `max_tokens`-truncated rows are excluded (those are tracked by `EXECUTOR_TRUNCATION_RATE`). Affected (id, session, stop_reason): {rows}. Look at the `/llm-call/<id>` page on the local dashboard to see what the model actually said."
- **suggested_fix:** "Inspect each affected row's full response via `/llm-call/<id>` on the local dashboard. If the executor is replying in prose, tighten the prompt in `v2/agent.py::TRADING_SYSTEM_PROMPT` to insist on JSON-only output. If it's a refusal pattern (e.g., the executor balking at sector-cap signals), either teach the executor to emit a structured `hold` decision instead, or revisit the upstream input shape so the refusal is no longer prompted. This check complements `EXECUTOR_PARSE_FAILURE_RATE` (which looks at the post-parse telemetry); this one looks at the raw assistant text. If the parser in `v2/agent.py` grows additional wrapper-stripping (e.g., a new prefix the model emits), update the `regexp_replace` here to match — the check should only fire on shapes the parser truly cannot consume."

### LLM_CONTEXT_MISSING_ROWS_FOR_PURPOSE

- **env:** both
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** llm-context-missing-rows-for-purpose
- **title_template:** "{n} purpose(s) with downstream artifacts but zero llm_call_contexts rows (last 7d)"
- **sql:**
  ```sql
  WITH coverage AS (
    SELECT
      'executor' AS purpose,
      (SELECT COUNT(*) FROM decisions
         WHERE date > current_date - interval '7 days') AS expected,
      (SELECT COUNT(*) FROM llm_call_contexts
         WHERE purpose = 'executor'
           AND created_at > now() - interval '7 days') AS captured
    UNION ALL
    SELECT
      'reflection_loop',
      (SELECT COUNT(*) FROM strategy_memos
         WHERE created_at > now() - interval '7 days'),
      (SELECT COUNT(*) FROM llm_call_contexts
         WHERE purpose = 'reflection_loop'
           AND created_at > now() - interval '7 days')
  )
  SELECT purpose, expected, captured
  FROM coverage
  WHERE expected > 0 AND captured = 0;
  ```
- **finding_when:** "rows returned"
- **body_template:** "A pipeline purpose produced downstream artifacts (executor → decisions, reflection_loop → strategy_memos) but recorded zero rows to `llm_call_contexts`. `LLM_CONTEXT_MISSING_SYSTEM_PROMPT` only fires on NULL `system_prompt`, so a purpose whose rows are missing entirely would slip past it silently. The `/llm-call/<id>` dashboard view and the session-detail \"LLM Calls\" section will be empty for the affected purpose. Affected (purpose, expected, captured): {rows}."
- **suggested_fix:** "Confirm `v2/claude_client.py::_call_with_retry` still calls `_record_call_context` in its `finally` block and that the affected purpose is in `_CONTEXT_LOGGED_PURPOSES`. Check that the call site that produces the downstream artifact (`v2/agent.py` for executor, `v2/strategy.py` for reflection) routes through `_call_with_retry` / `run_agentic_loop` with the correct `purpose=` kwarg — a regression that bypasses those helpers (e.g., calling the SDK directly) would silently drop the capture without raising. Wired in commit `aef76ea`; if rows have stopped appearing after a code change since then, bisect from there."

### RULE_CHURN_SHORT_LIVED

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** rule-churn-short-lived
- **title_template:** "{short_lived} of {retired} retired rule(s) in last 30d had lifetime < 7d"
- **sql:**
  ```sql
  SELECT
    COUNT(*) AS retired_30d,
    COUNT(*) FILTER (WHERE retired_at - created_at < interval '7 days') AS short_lived_7d,
    array_agg(id ORDER BY id) FILTER (WHERE retired_at - created_at < interval '7 days') AS short_lived_ids
  FROM strategy_rules
  WHERE retired_at IS NOT NULL
    AND retired_at > now() - interval '30 days';
  ```
- **finding_when:** "retired_30d >= 5 AND short_lived_7d / retired_30d >= 0.25"
- **body_template:** "Reflection is retiring rules within a week of proposing them at an elevated rate. Either the strategist proposes rules too eagerly, reflection retires them too eagerly, or the same rule_text keeps getting re-proposed after retirement (degenerate learning loop). retired_30d={retired_30d}, short_lived_7d={short_lived_7d}, short_lived_ids={short_lived_ids}. Escalates to critical when short_lived_7d / retired_30d >= 0.50 (warn otherwise)."
- **suggested_fix:** "Inspect the offending rule_ids — read `rule_text` and `retirement_reason` for each in `strategy_rules`. If many are `Bulk pruning: rule accumulation...`, the proposing side in `v2/ideation_claude.py` is too liberal; tighten the propose_rule prompt to require attribution evidence. If many are `Built on broken signal-citation pipeline...` or similar root-cause retirements, the reflection prompt in `v2/strategy.py` is correctly retiring — but the system is wasting strategist effort. Consider gating `propose_rule` behind a minimum sample_size_30d threshold."

### RULE_ACTIVE_DORMANT_LONG_LIVED

- **env:** prod
- **severity:** warn
- **category:** quality
- **worktype:** code
- **topic_slug:** rule-active-dormant-long-lived
- **title_template:** "{n} active rule(s) older than 30d with zero executor citations in last 14d"
- **sql:**
  ```sql
  SELECT
    COUNT(*) AS n,
    array_agg(id ORDER BY id) AS dormant_ids
  FROM strategy_rules r
  WHERE r.status = 'active'
    AND r.created_at < now() - interval '30 days'
    AND NOT EXISTS (
      SELECT 1 FROM decisions d
      WHERE d.date > (now() - interval '14 days')::date
        AND d.reasoning ~* ('\mrule\s*#?\s*' || r.id || '\M')
    );
  ```
- **finding_when:** "n >= 5"
- **body_template:** "Active rules that have been alive >30d but went uncited by the executor for the last 14d. Reflection runs daily yet hasn't retired them — the most common shape of an unresolved rule contradiction is a pair where one rule is dormant (effectively dead) but stays active because reflection's retirement cap or tenure floor blocked action. n={n}, dormant_ids={dormant_ids}. Persistent dormants pollute strategist context and prolong contradictions flagged by past RULE_CONTRADICTION findings. Escalates to critical when n >= 15."
- **suggested_fix:** "For each dormant_id, read `rule_text` and `retirement_reason` (the latter will be null on actives) in `strategy_rules`. If the rule is clearly stale (superseded by attribution, contradicted by a newer rule, or about a feature that no longer exists), retire it via `tool_retire_rule` in the next strategy session. If retirement is being blocked by the per-session cap (`MAX_RETIREMENTS_PER_SESSION` in `v2/strategy.py`), consider raising the cap when n is high. If the executor citation regex (`_RULE_CITATION_RE` in `v2/trader.py`) has drifted from how reflection actually writes rule references, fix the regex first — false dormants will fire here."

## Ideation pass

After running every deterministic check, perform one ideation pass. The goal is to surface findings the deterministic checks won't catch — concept-level gaps, cost trends, prompt-engineering hunches, missed audit coverage.

**Sources to read:**

- Last 5 rows of `strategy_memos` (the trading system's session journal):
  ```sql
  SELECT session_id, content, created_at FROM strategy_memos ORDER BY created_at DESC LIMIT 5;
  ```
- Last 14 days of `decisions`:
  ```sql
  SELECT id, symbol, action, qty, reasoning, outcome_summary, created_at
  FROM decisions WHERE created_at > now() - interval '14 days'
  ORDER BY created_at DESC;
  ```
- Recent commits on `main`:
  ```bash
  git log --since="14 days ago" --oneline main
  ```
- The dashboard pages (read via `curl http://localhost:3000/<path>` or by reading the corresponding template + rendering function in `dashboard/`):
  - `/mistakes`
  - `/attribution`
  - `/performance`
  - `/strategy`

**What to look for:**

1. **Audit gaps.** A pattern in the decisions or memos that today's deterministic checks would miss. Example: rule #N has been retired but still appears in playbook actions. Category: `audit_gap`.
2. **App improvements.** Concrete, actionable changes to the trading code or prompts. Example: the executor prompt mentions a tool that has been renamed. Category: `app_improvement`.

**Constraints:**

- Emit at most 3 ideation findings per tick. The cap protects ticket churn.
- Each finding's `topic_slug` must be a kebab-case noun phrase derived from the finding's core idea (e.g., `rule-27-flip-flop`, `executor-token-budget`). The slug + check_code together must dedup the finding across days, so be consistent: if the same idea recurs, use the same slug.
- `check_code` for ideation findings is one of: `ideation_audit_gap` | `ideation_app_improvement`.
- `worktype` for ideation findings defaults to `code`. Override to `db` only when the natural fix is a SQL mutation.
