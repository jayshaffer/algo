# Audit Finding #7 — Refine `DECISIONS_NO_SIGNAL_REFS` and sibling check

**Date:** 2026-05-06
**Status:** Draft
**Type:** Code change in `v2/audit.py` + tests

## Problem

The self-healing audit raised `DECISIONS_NO_SIGNAL_REFS` as **critical**
("13 of 29 recent buy/sell decisions have no signal_refs (13
on-playbook)"). The body text blames "Strategist→playbook→executor
signal_refs wiring is being honored intermittently."

Investigation shows the wiring is **not the issue**. All 13 missing
decisions trace back to playbook actions whose underlying theses have
**zero `thesis_signals` rows**. Two distinct legitimate sub-cases account
for most of the gap:

1. **Adopted theses (`source='adoption'`)** — positions adopted from
   the live portfolio without an ideation-time citation. By design they
   have no signal_refs. Examples in the current finding: thesis #182
   (GOOGL), thesis #224 (SCHW).
2. **Playbook actions with `thesis_id IS NULL`** — hand-crafted /
   adopted-position actions with no thesis to derive signals from.
   Example: playbook_action #164 in finding #7's evidence.

The remaining sub-case is the genuine signal:

3. **`claude_ideation` theses with no signals** — strategist created
   the thesis but failed to call `signal_refs` / `add_signal_refs`.
   Examples: thesis #178 (AMZN), thesis #205 (CRM).

Today the check lumps all three together and over-fires critical. The
sibling check `check_theses_missing_signal_refs` (currently warn at
56/69 = 81%) has the same flaw — it counts adopted theses against the
strategist.

## Goal

Refine both checks so:

- Adopted theses and `thesis_id IS NULL` actions are **excluded** from
  the "missing" count.
- Severity is driven only by the **genuinely missing** sub-bucket
  (`claude_ideation` theses lacking signals, on-playbook decisions
  derived from them).
- Body text and evidence accurately describe the residual problem.
- Open finding #7 will then either auto-supersede or drop in severity
  on the next audit run; any residual `genuinely_missing` count is the
  real strategist-citation bug to chase in a follow-up.

## Approach

### Single finding per check, not split

Keep `DECISIONS_NO_SIGNAL_REFS` as one finding with rich evidence
buckets. Splitting into separate `…_ADOPTED` / `…_NO_THESIS` info
findings adds noise without informational value — the dashboard's
`/audit` page can render the breakdown from the existing JSON evidence.

### Code changes — `v2/audit.py`

**`check_decisions_missing_signal_refs` (line 393):**

Replace the SQL with a query that joins `playbook_actions` and `theses`
so each decision can be classified into exactly one of five buckets
(in priority order — first match wins):

- `has_refs` — at least one `decision_signals` row exists
- `excluded_off_playbook` — `is_off_playbook=true` (existing carve-out;
  off-playbook trades are documented to lack refs)
- `excluded_no_thesis` — on-playbook but `playbook_action.thesis_id IS
  NULL` (hand-crafted action, no thesis to derive signals from)
- `excluded_adoption` — on-playbook, thesis exists,
  `theses.source='adoption'`
- `genuinely_missing` — on-playbook, thesis exists,
  `theses.source != 'adoption'` (typically `claude_ideation`)

Compute `severity = "critical" if genuinely_missing / total > 0.10 else
"warn"`. Continue to skip the finding entirely if `genuinely_missing == 0`
AND `excluded_*` buckets are all the gap (meaning the system is behaving
correctly — every gap is explained).

Update the title and body:

- Title: `"{genuinely_missing} of {total} recent buy/sell decisions
  derived from ideation theses have no signal_refs ({excluded_adoption}
  adopted, {excluded_no_thesis} no-thesis excluded)"`
- Body: `"Strategist created theses without citing signals via
  `signal_refs`/`add_signal_refs`, so executor decisions derived from
  them have nothing to attribute. Fix in v2/ideation_claude.py prompt
  or downstream tool-call validation."`

Evidence JSON gains the new buckets:

```json
{
  "total": 29,
  "has_refs": 16,
  "excluded_off_playbook": 0,
  "excluded_no_thesis": 4,
  "excluded_adoption": 4,
  "genuinely_missing": 5,
  "decision_ids": [/* genuinely_missing only */]
}
```

`decision_ids` becomes the IDs of the **`genuinely_missing`** bucket
only — the bucket worth investigating. The other buckets are
counts only; their IDs aren't needed for follow-up.

**`check_theses_missing_signal_refs` (line 445):**

Add `AND t.source != 'adoption'` to both the `COUNT(*)` and the
`SELECT t.id` queries. Keep the 25% threshold and the existing body
text (it correctly blames the strategist already).

### Tests — `tests/v2/test_audit.py`

Update `TestCheckDecisionsMissingSignalRefs` (line 424):

- Update existing `fetchone.return_value` mocks to include the new keys
  (`excluded_adoption`, `excluded_no_thesis`, `genuinely_missing`,
  `has_refs`).
- Add: `test_all_missing_explained_by_adoption_and_no_thesis_no_finding`
  — when `genuinely_missing == 0` and the gap is fully explained, no
  finding is emitted.
- Add: `test_genuinely_missing_above_10pct_critical` — confirms the
  new severity boundary triggers on the new bucket only.

Update `TestCheckThesesMissingSignalRefs` (line 452):

- Existing tests still apply (they use mock cursor, the SQL change is
  invisible). No new test required, but add one for documentation:
  `test_adoption_theses_excluded_from_total` — verifies the cursor is
  invoked with a query that filters out `source='adoption'` (assert via
  the `cur.execute` call_args).

### Verification

1. Unit tests: `pytest tests/v2/test_audit.py` green.
2. Run audit in prod (`task audit`) and inspect the new evidence JSON.
   Expected: finding #7 is auto-superseded or dropped to `warn`; if
   `genuinely_missing > 0` remains, that's the residual bug to chase
   in a separate spec.
3. Inspect finding #8 (`THESES_NO_SIGNAL_REFS`): should drop materially
   from 56/69 since most of those are likely adoption theses.

## Out of scope

- Fixing the strategist's failure to cite signals on `claude_ideation`
  theses (the residual `genuinely_missing` bucket). That's a follow-up
  spec once the audit accurately surfaces the count.
- Changes to the dashboard `/audit` page rendering. The new evidence
  JSON keys are additive; existing rendering will keep working and the
  next dashboard pass can surface the buckets explicitly.
- Backfilling `thesis_signals` for old adopted theses (they were
  correctly omitted by design).

## Risks

- **Hiding a real regression behind the adoption exclusion.** Mitigated
  by keeping `excluded_adoption` visible in evidence; if it climbs to
  an unexpected fraction of recent decisions, that itself is worth
  investigating (something is over-classifying theses as adopted).
- **Test mocks drift from real SQL shape.** The unit tests use
  `MagicMock` `fetchone`, which doesn't validate the SQL. Mitigated by
  the prod-DB verification step — a malformed query will surface
  immediately on `task audit`.

## Files touched

- `v2/audit.py` — two functions modified
- `tests/v2/test_audit.py` — fixtures + new test cases
