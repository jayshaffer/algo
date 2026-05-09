# Flip-Flop Reflection Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface ticker-level round-trip counts (same ticker, opposing actions within a short window) to the strategist's reflection stage so it can self-diagnose the Rule 27 oscillation pattern that's been driving GOOGL/CRM/NVDA/AMZN flip-flops.

**Architecture:** Single-query addition. A new `analyze_round_trips()` function in `v2/patterns.py` returns ticker-level round-trip counts over a window. The existing `tool_get_session_summary` (in `v2/strategy.py`) — which the reflection LLM already calls every session — appends a "Round-Trips" section to its output when any are found. No new tool, no prompt change, no executor change, no DB schema change. The strategist sees evidence on the next session and proposes/revises rules on its own via the existing learning loop.

**Tech Stack:** Python 3.x, psycopg2 (raw SQL via `get_cursor()` context manager), pytest with mocked cursors via existing `mock_db`/`mock_cursor` fixtures in `tests/v2/conftest.py`.

**Why minimal-change:** The strategist learns by writing rules to `strategy_rules` (DB), not by code edits. The smallest leverage point is feeding it data it currently can't see. Reflection memos already enumerate rule citations and signal attribution but never count opposing-action pairs per ticker. Adding ~50 LOC of query + ~15 LOC of formatting is enough to break the blind spot.

---

## Files

- **Modify:** `v2/patterns.py` — add `RoundTrip` dataclass and `analyze_round_trips()` function (top-level, alongside existing `analyze_*` functions)
- **Modify:** `v2/strategy.py` — extend `tool_get_session_summary()` (around lines 274–338) to append a round-trip section
- **Modify:** `tests/v2/test_patterns.py` — add `TestAnalyzeRoundTrips` class
- **Modify:** `tests/v2/test_strategy.py` — extend `TestToolGetSessionSummary` with two new tests

Total: 4 files, ~120 lines added, 0 deletions. No new modules, no new files.

---

## Task 1: Add `RoundTrip` dataclass and `analyze_round_trips()` to `v2/patterns.py`

**Files:**
- Modify: `v2/patterns.py` (add new dataclass after `ConfidenceCorrelation` around line 53; add new function after `analyze_confidence_correlation()` around line 226)
- Test: `tests/v2/test_patterns.py` (add `TestAnalyzeRoundTrips` class at end of file)

### What this function does

Returns a list of `RoundTrip` records, one per ticker that had at least `min_pairs` opposing-action pairs within `gap_days` of each other in the lookback window. A "pair" is any two decisions on the same ticker where the later one has the opposite `action` of the earlier one and they are within `gap_days` calendar days. Sort descending by `pair_count`.

The query JOINs the `decisions` table to itself, restricted to `action IN ('buy','sell')` (excludes `hold` and `invalid`). The strategist consumes the result via the session summary text.

- [ ] **Step 1.1: Write the failing test for the dataclass + empty-result case**

Add to `tests/v2/test_patterns.py` (append at end of file):

```python
class TestAnalyzeRoundTrips:
    """Tests for analyze_round_trips() — surfaces flip-flop patterns."""

    def test_returns_empty_list_when_no_pairs(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []

        result = analyze_round_trips(days=30, gap_days=7, min_pairs=2)

        assert result == []

    def test_returns_round_trip_objects(self, mock_db):
        from v2.patterns import RoundTrip, analyze_round_trips
        from datetime import date
        mock_db.fetchall.return_value = [
            {
                "ticker": "GOOGL",
                "pair_count": 11,
                "first_date": date(2026, 4, 15),
                "last_date": date(2026, 5, 6),
            },
            {
                "ticker": "CRM",
                "pair_count": 9,
                "first_date": date(2026, 3, 10),
                "last_date": date(2026, 5, 5),
            },
        ]

        result = analyze_round_trips(days=60, gap_days=14, min_pairs=2)

        assert len(result) == 2
        assert result[0] == RoundTrip(
            ticker="GOOGL",
            pair_count=11,
            first_date=date(2026, 4, 15),
            last_date=date(2026, 5, 6),
        )
        assert result[1].ticker == "CRM"

    def test_sql_self_joins_decisions_on_opposite_action(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []

        analyze_round_trips(days=30, gap_days=7, min_pairs=2)

        sql = mock_db.execute.call_args[0][0]
        # Must self-join on opposing actions within a date gap
        assert "decisions" in sql.lower()
        assert "b.action <> a.action" in sql
        # Must restrict to executed buy/sell, not hold/invalid
        assert "action IN ('buy', 'sell')" in sql or "action in ('buy','sell')" in sql.lower()
        # Must group by ticker
        assert "GROUP BY" in sql.upper()
        # Must apply min_pairs threshold via HAVING
        assert "HAVING" in sql.upper()

    def test_passes_window_and_gap_parameters(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []

        analyze_round_trips(days=45, gap_days=10, min_pairs=3)

        params = mock_db.execute.call_args[0][1]
        # Three substitutions in order: lookback days, gap days, min pairs
        assert 45 in params
        assert 10 in params
        assert 3 in params

    def test_default_parameters(self, mock_db):
        """Defaults should match what the reflection stage will use."""
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []

        analyze_round_trips()

        params = mock_db.execute.call_args[0][1]
        # Defaults: 30-day lookback, 7-day gap, min 2 pairs
        assert 30 in params
        assert 7 in params
        assert 2 in params
```

- [ ] **Step 1.2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_patterns.py::TestAnalyzeRoundTrips -v`

Expected: 5 FAILS with `ImportError: cannot import name 'RoundTrip'` / `cannot import name 'analyze_round_trips'`.

- [ ] **Step 1.3: Add the dataclass to `v2/patterns.py`**

In `v2/patterns.py`, after the existing `ConfidenceCorrelation` dataclass (around line 53), add:

```python
@dataclass
class RoundTrip:
    """Same-ticker opposing-action pair count over a window.

    A round-trip is any pair of decisions (a, b) on the same ticker where
    a is earlier than b, b.action != a.action, and they're within
    gap_days of each other. We count all such pairs per ticker.
    """
    ticker: str
    pair_count: int
    first_date: object  # date — kept loose to match other dataclasses' pattern
    last_date: object
```

- [ ] **Step 1.4: Add the function to `v2/patterns.py`**

In `v2/patterns.py`, after `analyze_confidence_correlation()` (around line 226), add:

```python
def analyze_round_trips(
    days: int = 30,
    gap_days: int = 7,
    min_pairs: int = 2,
) -> list[RoundTrip]:
    """Find tickers that flip-flopped (opposing actions within gap_days).

    Self-joins `decisions` to itself on same ticker, opposite action,
    later date within gap_days. Returns one row per ticker that had
    at least min_pairs such pairs in the lookback window, sorted by
    pair_count descending.

    Used by the reflection stage to surface churn that signal-level
    attribution can't see — same ticker, multiple buy/sell cycles
    in a short window indicates strategy oscillation rather than
    signal mis-calibration.
    """
    with get_cursor() as cur:
        cur.execute("""
            WITH bs AS (
                SELECT id, date, ticker, action
                FROM decisions
                WHERE date > CURRENT_DATE - INTERVAL '1 day' * %s
                  AND action IN ('buy', 'sell')
            )
            SELECT a.ticker,
                   COUNT(*) AS pair_count,
                   MIN(a.date) AS first_date,
                   MAX(b.date) AS last_date
            FROM bs a
            JOIN bs b
              ON a.ticker = b.ticker
             AND b.id > a.id
             AND b.action <> a.action
             AND (b.date - a.date) <= %s
            GROUP BY a.ticker
            HAVING COUNT(*) >= %s
            ORDER BY pair_count DESC
        """, (days, gap_days, min_pairs))

        return [
            RoundTrip(
                ticker=row["ticker"],
                pair_count=row["pair_count"],
                first_date=row["first_date"],
                last_date=row["last_date"],
            )
            for row in cur.fetchall()
        ]
```

- [ ] **Step 1.5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_patterns.py::TestAnalyzeRoundTrips -v`

Expected: 5 PASS.

- [ ] **Step 1.6: Commit**

```bash
git add v2/patterns.py tests/v2/test_patterns.py
git commit -m "$(cat <<'EOF'
feat(patterns): add analyze_round_trips for flip-flop detection

Self-join over decisions to count same-ticker opposing-action pairs
within a short gap. Surfaces churn that signal-level attribution
cannot see — strategy oscillation rather than signal miscalibration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Append round-trip section to `tool_get_session_summary` output

**Files:**
- Modify: `v2/strategy.py:274-338` (extend `tool_get_session_summary`)
- Test: `tests/v2/test_strategy.py` (extend `TestToolGetSessionSummary`)

### What this change does

The reflection agent already calls `get_session_summary` once per session. We append a single new section at the end of its output. When `analyze_round_trips()` returns no rows, we say so explicitly (a positive signal — "no flip-flops" is itself useful evidence). When it returns rows, we render up to 5 lines.

### Output format (what the LLM will read)

When round-trips exist:
```
Round-Trips (past 30d, ≥2 opposing actions ≤7d apart):
  GOOGL: 11 pairs (2026-04-15 to 2026-05-06)
  CRM: 9 pairs (2026-03-10 to 2026-05-05)
  NVDA: 9 pairs (2026-03-12 to 2026-03-30)
  AMZN: 7 pairs (2026-03-27 to 2026-05-04)
  SCHW: 2 pairs (2026-04-17 to 2026-04-30)
```

When none:
```
Round-Trips (past 30d, ≥2 opposing actions ≤7d apart): none.
```

- [ ] **Step 2.1: Write the failing test for the populated case**

Append to `TestToolGetSessionSummary` class in `tests/v2/test_strategy.py` (after `test_orphan_thesis_excluded_from_signal_labels`):

```python
    @patch("v2.strategy.analyze_round_trips")
    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_renders_round_trips_when_present(
        self, mock_decisions, mock_attr, mock_round_trips, mock_db, mock_cursor,
    ):
        from datetime import date
        from v2.strategy import tool_get_session_summary
        from v2.patterns import RoundTrip
        mock_decisions.return_value = [make_decision_row()]
        mock_attr.return_value = "Attribution data here"
        mock_cursor.fetchall.return_value = []
        mock_round_trips.return_value = [
            RoundTrip("GOOGL", 11, date(2026, 4, 15), date(2026, 5, 6)),
            RoundTrip("CRM", 9, date(2026, 3, 10), date(2026, 5, 5)),
        ]

        result = tool_get_session_summary()

        assert "Round-Trips" in result
        assert "GOOGL: 11 pairs" in result
        assert "CRM: 9 pairs" in result
        assert "2026-04-15" in result
        assert "2026-05-06" in result

    @patch("v2.strategy.analyze_round_trips")
    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_renders_round_trips_none_marker_when_empty(
        self, mock_decisions, mock_attr, mock_round_trips, mock_db, mock_cursor,
    ):
        from v2.strategy import tool_get_session_summary
        mock_decisions.return_value = [make_decision_row()]
        mock_attr.return_value = "Attribution data here"
        mock_cursor.fetchall.return_value = []
        mock_round_trips.return_value = []

        result = tool_get_session_summary()

        # Explicit "none" so the LLM sees the absence as evidence,
        # not a missing section it might paper over.
        assert "Round-Trips" in result
        assert "none" in result.lower()

    @patch("v2.strategy.analyze_round_trips")
    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_round_trips_uses_30d_window_7d_gap_min_2(
        self, mock_decisions, mock_attr, mock_round_trips, mock_db, mock_cursor,
    ):
        """Defaults must match what the strategist needs: 30d lookback
        is the same window used elsewhere in the summary, 7d gap captures
        same-week churn, min 2 pairs filters one-off entry/exit pairs."""
        from v2.strategy import tool_get_session_summary
        mock_decisions.return_value = []
        mock_attr.return_value = ""
        mock_cursor.fetchall.return_value = []
        mock_round_trips.return_value = []

        tool_get_session_summary()

        mock_round_trips.assert_called_once_with(days=30, gap_days=7, min_pairs=2)

    @patch("v2.strategy.analyze_round_trips")
    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_round_trips_caps_display_at_5(
        self, mock_decisions, mock_attr, mock_round_trips, mock_db, mock_cursor,
    ):
        """If 6+ tickers flip-flopped, render top 5 plus a count footer."""
        from datetime import date
        from v2.strategy import tool_get_session_summary
        from v2.patterns import RoundTrip
        mock_decisions.return_value = []
        mock_attr.return_value = ""
        mock_cursor.fetchall.return_value = []
        mock_round_trips.return_value = [
            RoundTrip(f"T{i}", 10 - i, date(2026, 4, 1), date(2026, 5, 1))
            for i in range(7)
        ]

        result = tool_get_session_summary()

        # Top 5 rendered
        for i in range(5):
            assert f"T{i}: {10-i} pairs" in result
        # 6th and 7th not rendered as line items
        assert "T5:" not in result
        assert "T6:" not in result
        # But total count is acknowledged
        assert "7 total" in result or "(7 tickers)" in result
```

- [ ] **Step 2.2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestToolGetSessionSummary -v`

Expected: 4 NEW tests FAIL (existing tests still pass). Failures will be a mix of `AttributeError: module 'v2.strategy' has no attribute 'analyze_round_trips'` and assertion failures on missing "Round-Trips" string.

- [ ] **Step 2.3: Add the import to `v2/strategy.py`**

In `v2/strategy.py`, find the existing imports near the top of the file. Locate the block of `from .X import ...` statements. After the existing `from .formation import build_formation_context` (around line 23), add:

```python
from .patterns import analyze_round_trips
```

(If `analyze_round_trips` is patched as `v2.strategy.analyze_round_trips` in tests, it must be imported as a name in this module — `from .patterns import` works for that. Importing the module and calling `patterns.analyze_round_trips` would not, because `patch("v2.strategy.analyze_round_trips")` only replaces the bound name.)

- [ ] **Step 2.4: Append the round-trip section in `tool_get_session_summary`**

In `v2/strategy.py`, locate the end of `tool_get_session_summary` (the `return "\n".join(lines)` near line 338). Just before that return, after the existing `lines.append(get_attribution_summary())` block, insert:

```python
    # Round-trip evidence — surfaces same-ticker opposing-action churn
    # that signal-level attribution cannot see. Uses the same 30d window
    # as the rest of this summary; gap_days=7 captures same-week flips.
    round_trips = analyze_round_trips(days=30, gap_days=7, min_pairs=2)
    lines.append("")
    if round_trips:
        lines.append("Round-Trips (past 30d, ≥2 opposing actions ≤7d apart):")
        for rt in round_trips[:5]:
            lines.append(
                f"  {rt.ticker}: {rt.pair_count} pairs "
                f"({rt.first_date} to {rt.last_date})"
            )
        if len(round_trips) > 5:
            lines.append(f"  ... ({len(round_trips)} total)")
    else:
        lines.append("Round-Trips (past 30d, ≥2 opposing actions ≤7d apart): none.")
```

- [ ] **Step 2.5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestToolGetSessionSummary -v`

Expected: All tests PASS (the 4 existing + 4 new = 8 total).

- [ ] **Step 2.6: Run the full v2 test suite to catch regressions**

Run: `python3 -m pytest tests/v2/ -q`

Expected: Same pass count as before this plan + 9 new tests passing (5 in test_patterns + 4 in test_strategy). No new failures elsewhere.

- [ ] **Step 2.7: Commit**

```bash
git add v2/strategy.py tests/v2/test_strategy.py
git commit -m "$(cat <<'EOF'
feat(strategy): surface round-trip evidence in session summary

Reflection LLM already calls get_session_summary every session; this
appends a Round-Trips section so the strategist can see ticker-level
flip-flop counts. Currently invisible to it — signal attribution
counts citations, not opposing-action sequences.

Renders top 5 tickers with ≥2 opposing-action pairs ≤7d apart over
the past 30d, or an explicit "none" marker when clean. The strategist
will use this to revise Rule 27 (oscillating bind/lift conditions)
via the existing learning loop — no executor change required.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Verify against prod data and confirm output

This is a one-shot smoke test against the running prod DB to confirm the query produces the expected output before merging. No code change in this task; it's a sanity check that the implementation matches the dive-in finding (GOOGL ~11, CRM ~9, NVDA ~9, AMZN ~7).

- [ ] **Step 3.1: Confirm the prod db is reachable**

Run: `docker compose ps db`

Expected: `algo-db-1` listed with `running` status. If not, run `docker compose up -d db` first (this starts only the db, no trading agent).

- [ ] **Step 3.2: Run the query interactively against prod**

Run:
```bash
set -a; source .env; set +a; \
docker compose exec -T trading python3 -c "
from v2.patterns import analyze_round_trips
for rt in analyze_round_trips(days=60, gap_days=14, min_pairs=2):
    print(f'{rt.ticker}: {rt.pair_count} pairs ({rt.first_date} to {rt.last_date})')
"
```

Note: this requires the trading container running. If it's not, you can run the same query directly in psql:
```bash
set -a; source .env; set +a; \
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
WITH bs AS (
  SELECT id, date, ticker, action FROM decisions
  WHERE date > CURRENT_DATE - INTERVAL '60 days' AND action IN ('buy','sell')
)
SELECT a.ticker, COUNT(*) AS pairs, MIN(a.date), MAX(b.date)
FROM bs a JOIN bs b ON a.ticker = b.ticker AND b.id > a.id
                   AND b.action <> a.action AND (b.date - a.date) <= 14
GROUP BY a.ticker HAVING COUNT(*) >= 2 ORDER BY pairs DESC;"
```

Expected output (from dive-in conducted 2026-05-08):
- `GOOGL: 11 pairs`
- `CRM: 9 pairs`
- `NVDA: 9 pairs`
- `AMZN: 7 pairs`
- A few smaller ones (SCHW, ACN, MDT, XLE) at 1–2 pairs

If the numbers materially diverge (off by ±2 is acceptable as new decisions land daily; off by ±5 is not), pause and re-check the SQL — most likely culprit is the `gap_days` arithmetic or missing `action IN ('buy','sell')` filter.

- [ ] **Step 3.3: Confirm reflection-stage output by triggering a session in dry-run**

Run:
```bash
docker compose exec trading python -m v2.session --stage strategy --dry-run 2>&1 | grep -A 10 "Round-Trips"
```

Expected: A "Round-Trips" block appears in the strategist's tool-result output.

If no session has trading_result data, this stage may exit early before calling `get_session_summary`. In that case, run a unit-style spot check instead:
```bash
docker compose exec trading python3 -c "
from v2.strategy import tool_get_session_summary
print(tool_get_session_summary())
"
```
Expected: full summary text including the new Round-Trips section.

- [ ] **Step 3.4: No commit needed for this task**

Task 3 is verification only. If everything looks right, the work from Tasks 1 and 2 is ready to merge.

---

## What we explicitly are NOT doing in this plan (and why)

- **Not editing Rule 27 directly.** The strategist owns the rules table; manually editing fights the learning loop. Surfacing the evidence lets it revise/retire Rule 27 itself, which is the desired behavior.
- **Not adding executor-side cooldown.** That's a larger change (touches `v2/executor.py` and `v2/trader.py`) and only justified if the reflection-stage fix doesn't reduce churn within ~5 sessions. Re-evaluate then.
- **Not adding a new strategist tool (`get_round_trips`).** The strategist already calls `get_session_summary` every session; piggy-backing on it avoids a new tool definition (which would also require a system-prompt mention). One less surface to maintain.
- **Not filtering by notional.** A $25 round-trip and a $300 round-trip both count as one pair. The strategist can apply its own significance heuristic when it reads the output. Adding a notional threshold here is premature optimization and would hide small-but-frequent churn that the system was already shown to do (CRM 3/10–3/18 was six $30 buys).
- **Not changing `gap_days` defaults to capture longer windows.** 7 days catches same-week flips, which is the high-frequency pattern. The 14-day window from the dive-in produces a longer list; the strategist can request larger windows by reading the function source if it ever needs to (it has tool-use ability), but the default is tuned for what reflection actually needs.

---

## Self-Review Checklist (completed)

- **Spec coverage:** Plan covers (a) the query, (b) wiring to reflection, (c) prod verification. The spec is "minimal application-side changes to surface flip-flops to reflection." All three subtasks map to spec requirements.
- **Placeholders:** None. All code blocks contain final code; all commands are concrete.
- **Type consistency:** `RoundTrip` dataclass fields (`ticker: str`, `pair_count: int`, `first_date`, `last_date`) used identically in patterns.py, strategy.py, and both test files. Function signature `analyze_round_trips(days=30, gap_days=7, min_pairs=2)` consistent across all 3 call sites.
- **Mock paths:** `mock_db` fixture in `tests/v2/conftest.py` already patches `v2.database.connection.get_cursor`, which is what `v2/patterns.py` imports from. New tests in `test_patterns.py` use the existing `mock_db` fixture without modification. Tests in `test_strategy.py` patch `v2.strategy.analyze_round_trips` directly — this requires the `from .patterns import analyze_round_trips` form (Step 2.3), not module-attribute access.
