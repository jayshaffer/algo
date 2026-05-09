# Flip-Flop Structural Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the structural blind spots that produce same-ticker flip-flop trading by (a) giving the executor visibility into recent decisions on the tickers it's about to trade, (b) requiring the strategist to justify reversals against its own recent playbooks, and (c) surfacing aggregate round-trip evidence to the reflection stage so the system can self-correct any residual oscillation via the existing learning loop.

**Architecture:** Two-phase synthesis. **Phase A** breaks the loop at its source — the executor gains a new `recent_ticker_decisions` input (1-7 day window per ticker), and the strategist prompts gain a "Reversal Justification" rule pointing at a new `get_recent_playbooks` tool. **Phase B** adds the diagnostic layer — `analyze_round_trips()` in `v2/patterns.py` is wired into the reflection stage's `tool_get_session_summary` so the strategist can see ticker-level churn counts in aggregate. Phase A prevents the symptom; Phase B lets the system notice and write rules against any residual occurrence.

**Tech Stack:** Python 3.x, psycopg2 (raw SQL via `get_cursor()`), Anthropic SDK (Claude Sonnet for strategist, Haiku for executor), pytest with mocked cursors via `mock_db`/`mock_cursor` fixtures in `tests/v2/conftest.py`.

**Why this shape:** From the dive-in (2026-05-08), four structural failures compound to produce flip-flops:
1. Executor's `recent_outcomes` is filtered to `outcome_7d IS NOT NULL` (≥7d old) and `todays_decisions` is today-only — leaves a 1-6 day blind spot where yesterday's sell is invisible to today's buy.
2. The playbook is fully replaced every session via `replace_playbook_actions_atomic`, so the strategist has no continuity artifact when it writes a contradicting action.
3. The strategist prompt has no anti-reversal guidance (compare to the explicit anti-oscillation rule for *rule changes* at `v2/strategy.py:140`).
4. Reflection's session summary caps decisions at 10 rows and never aggregates per-ticker churn counts.

This plan addresses 1 + 3 (Phase A) and 2 + 4 (Phase B). The Tier 3 cooldown enforcement is intentionally deferred — Phase A removes the structural blind spot without imposing a hard rule, and we want to see whether the LLM-side fixes are sufficient before adding deterministic gates.

---

## Files

- **Modify:** `v2/agent.py` — extend `ExecutorInput` dataclass + executor system prompt
- **Modify:** `v2/context.py` — build the new `recent_ticker_decisions` field in `build_executor_input`
- **Modify:** `v2/database/trading_db.py` — add `get_recent_playbooks_with_actions` query
- **Modify:** `v2/tools.py` — add `tool_get_recent_playbooks` + register it in `TOOL_DEFINITIONS`/`TOOL_HANDLERS`
- **Modify:** `v2/ideation_claude.py` — add Reversal Justification rule + tool reference to `_STRATEGIST_TEMPLATE`
- **Modify:** `v2/patterns.py` — add `RoundTrip` dataclass + `analyze_round_trips()` function
- **Modify:** `v2/strategy.py` — append round-trip section to `tool_get_session_summary` output
- **Modify:** `tests/v2/test_agent.py`, `tests/v2/test_context.py`, `tests/v2/test_ideation_claude.py`, `tests/v2/test_tools.py`, `tests/v2/test_patterns.py`, `tests/v2/test_strategy.py` — TDD coverage for each change

Total: 7 production files modified, 6 test files extended, ~280 LOC added (≈110 production, ≈170 tests). 0 new files. 0 schema changes. 0 deletions.

---

# PHASE A — Stop the bleeding

The two changes in this phase address the *cause* of flip-flops: the executor and strategist literally cannot see their own recent reversals. After Phase A, both stages have the data they need to make an informed reversal-or-not call.

## Task 1: Add `recent_ticker_decisions` to executor input

**Files:**
- Modify: `v2/agent.py:50-72` (extend `ExecutorInput` dataclass) and `v2/agent.py:116-166` (executor system prompt)
- Modify: `v2/context.py:418-527` (build the field in `build_executor_input`)
- Test: `tests/v2/test_agent.py` (extend existing `TestExecutorInput` class) and `tests/v2/test_context.py` (extend existing context tests)

### What this field contains

For each ticker present in today's `playbook_actions`, the most recent ≤5 buy/sell decisions on that ticker within the past 7 days, formatted as:

```python
[
    {"ticker": "GOOGL", "date": "2026-05-04", "action": "sell",
     "quantity": 0.17, "price": 383.02,
     "reasoning": "GOOGL is trading at ~$383.67, very near analyst consensus PT of ~$387..."},
    {"ticker": "GOOGL", "date": "2026-04-27", "action": "sell",
     "quantity": 0.28, "price": 369.00,
     "reasoning": "..."},
]
```

Reasoning is included **untruncated** — the field exists specifically to let the executor reason about whether new evidence justifies a reversal, and a 60-char truncation (as in `tool_get_decision_history`) loses the actionable nuance.

The 7-day window matches the `gap_days=7` default used by Task 4's round-trip aggregate, keeping the two diagnostic layers consistent.

- [ ] **Step 1.1: Write failing test for `ExecutorInput` field**

Append to `TestExecutorInput` class in `tests/v2/test_agent.py`:

```python
    def test_executor_input_has_recent_ticker_decisions(self):
        from v2.agent import ExecutorInput
        ei = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
            recent_ticker_decisions=[
                {"ticker": "GOOGL", "date": "2026-05-04", "action": "sell",
                 "quantity": 0.17, "price": 383.02, "reasoning": "near PT"},
            ],
        )
        assert len(ei.recent_ticker_decisions) == 1
        assert ei.recent_ticker_decisions[0]["ticker"] == "GOOGL"

    def test_executor_input_defaults_empty_recent_ticker_decisions(self):
        from v2.agent import ExecutorInput
        ei = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
        )
        assert ei.recent_ticker_decisions == []
```

- [ ] **Step 1.2: Run tests to verify failure**

Run: `python3 -m pytest tests/v2/test_agent.py::TestExecutorInput -v`

Expected: 2 new tests FAIL with `TypeError: ExecutorInput.__init__() got an unexpected keyword argument 'recent_ticker_decisions'` (or similar).

- [ ] **Step 1.3: Add field to `ExecutorInput` dataclass**

In `v2/agent.py`, locate the `ExecutorInput` dataclass at line 52. Add the new field after `todays_decisions: list[dict] = None` (line 66):

```python
    recent_ticker_decisions: list[dict] = None
```

Then in `__post_init__` (after line 71, inside the existing method), add:

```python
        if self.recent_ticker_decisions is None:
            self.recent_ticker_decisions = []
```

- [ ] **Step 1.4: Run dataclass tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_agent.py::TestExecutorInput -v`

Expected: All `TestExecutorInput` tests PASS, including the 2 new ones.

- [ ] **Step 1.5: Write failing test for `build_executor_input` populating the field**

Append to `tests/v2/test_context.py` inside whatever class covers `build_executor_input` (look for tests around `recent_outcomes_filters_none_outcome` near line 123). If no class exists, create:

```python
class TestRecentTickerDecisions:
    """Executor input must surface recent decisions for tickers in the playbook."""

    @patch("v2.context.get_latest_price", return_value=Decimal("100"))
    @patch("v2.context.get_signal_attribution", return_value=[])
    @patch("v2.context.get_positions", return_value=[])
    @patch("v2.context.get_recent_decisions")
    @patch("v2.context.get_pending_playbook_actions")
    @patch("v2.context.get_playbook")
    def test_includes_decisions_on_playbook_tickers_within_7_days(
        self, mock_pb, mock_pb_actions, mock_decisions, mock_pos, mock_attr, mock_price,
        mock_db, mock_cursor,
    ):
        from datetime import date, timedelta
        from v2.context import build_executor_input
        today = date.today()
        mock_pb.return_value = {"id": 1, "market_outlook": "", "risk_notes": ""}
        mock_pb_actions.return_value = [
            {"id": 10, "ticker": "GOOGL", "action": "buy", "thesis_id": 5,
             "reasoning": "add", "confidence": "high", "intent_type": "invest_dollar",
             "intent_magnitude": 100, "priority": 1},
        ]
        mock_decisions.return_value = [
            # GOOGL sell yesterday — must be surfaced
            {"id": 200, "date": today - timedelta(days=1), "ticker": "GOOGL",
             "action": "sell", "quantity": 0.17, "price": 383.02,
             "reasoning": "near consensus PT, trim", "outcome_7d": None,
             "outcome_30d": None},
            # GOOGL buy 5 days ago — must be surfaced
            {"id": 195, "date": today - timedelta(days=5), "ticker": "GOOGL",
             "action": "buy", "quantity": 0.22, "price": 380.00,
             "reasoning": "Cloud blowout", "outcome_7d": None, "outcome_30d": None},
            # GOOGL trade 30 days ago — must NOT be surfaced (outside 7d window)
            {"id": 100, "date": today - timedelta(days=30), "ticker": "GOOGL",
             "action": "buy", "quantity": 0.5, "price": 350.00,
             "reasoning": "old", "outcome_7d": 2.0, "outcome_30d": 5.0},
            # AMZN trade yesterday — must NOT be surfaced (not in playbook)
            {"id": 199, "date": today - timedelta(days=1), "ticker": "AMZN",
             "action": "buy", "quantity": 1.0, "price": 270.00,
             "reasoning": "irrelevant", "outcome_7d": None, "outcome_30d": None},
        ]
        mock_cursor.fetchall.return_value = []  # for any leftover queries

        result = build_executor_input(account_info={"equity": 10000})

        rtd = result.recent_ticker_decisions
        tickers = [r["ticker"] for r in rtd]
        ids = [r.get("id") for r in rtd]
        assert "GOOGL" in tickers
        assert "AMZN" not in tickers, "AMZN not in today's playbook — must be excluded"
        assert 100 not in ids, "30d-old decision must be excluded (outside 7d window)"
        # Both recent GOOGL trades must be surfaced
        assert 200 in ids
        assert 195 in ids

    @patch("v2.context.get_latest_price", return_value=Decimal("100"))
    @patch("v2.context.get_signal_attribution", return_value=[])
    @patch("v2.context.get_positions", return_value=[])
    @patch("v2.context.get_recent_decisions")
    @patch("v2.context.get_pending_playbook_actions")
    @patch("v2.context.get_playbook")
    def test_reasoning_not_truncated(
        self, mock_pb, mock_pb_actions, mock_decisions, mock_pos, mock_attr, mock_price,
        mock_db, mock_cursor,
    ):
        """The whole point of this field is full reasoning context;
        truncation defeats the purpose."""
        from datetime import date, timedelta
        from v2.context import build_executor_input
        long_reasoning = "X" * 500
        today = date.today()
        mock_pb.return_value = {"id": 1, "market_outlook": "", "risk_notes": ""}
        mock_pb_actions.return_value = [
            {"id": 10, "ticker": "GOOGL", "action": "buy", "thesis_id": 5,
             "reasoning": "add", "confidence": "high", "intent_type": "invest_dollar",
             "intent_magnitude": 100, "priority": 1},
        ]
        mock_decisions.return_value = [
            {"id": 200, "date": today - timedelta(days=1), "ticker": "GOOGL",
             "action": "sell", "quantity": 0.17, "price": 383.02,
             "reasoning": long_reasoning, "outcome_7d": None, "outcome_30d": None},
        ]
        mock_cursor.fetchall.return_value = []

        result = build_executor_input(account_info={"equity": 10000})

        assert result.recent_ticker_decisions[0]["reasoning"] == long_reasoning, (
            "reasoning was truncated — executor cannot judge reversal without full text"
        )

    @patch("v2.context.get_latest_price", return_value=Decimal("100"))
    @patch("v2.context.get_signal_attribution", return_value=[])
    @patch("v2.context.get_positions", return_value=[])
    @patch("v2.context.get_recent_decisions", return_value=[])
    @patch("v2.context.get_pending_playbook_actions", return_value=[])
    @patch("v2.context.get_playbook")
    def test_empty_when_no_playbook_actions(
        self, mock_pb, mock_pb_actions, mock_decisions, mock_pos, mock_attr, mock_price,
        mock_db, mock_cursor,
    ):
        from v2.context import build_executor_input
        mock_pb.return_value = None
        mock_cursor.fetchall.return_value = []

        result = build_executor_input(account_info={"equity": 10000})

        assert result.recent_ticker_decisions == []
```

(If your repo uses a different existing pattern for the `mock_db`/decorator stack, mirror it — the fixture names above match `tests/v2/conftest.py:75-104`.)

- [ ] **Step 1.6: Run failing tests to verify**

Run: `python3 -m pytest tests/v2/test_context.py::TestRecentTickerDecisions -v`

Expected: 3 FAIL with `AssertionError` (the field exists but is empty because `build_executor_input` doesn't populate it yet).

- [ ] **Step 1.7: Populate field in `build_executor_input`**

In `v2/context.py`, locate `build_executor_input` around line 418. Find the block that computes `todays_decisions` (lines 505-512). After that block but before the `return ExecutorInput(...)` statement, add:

```python
    # T-flipflop: surface recent decisions on the tickers this session is
    # about to trade. Closes the 1-6 day blind spot — `recent_outcomes` is
    # filtered to outcome_7d IS NOT NULL (≥7d old), `todays_decisions` is
    # today-only. Without this field, yesterday's sell on a ticker is
    # invisible when the executor decides whether to buy it back today.
    today = date.today()
    playbook_tickers = {a.ticker for a in actions}
    seven_days_ago = today - timedelta(days=7)
    recent_ticker_decisions = []
    for d in recent:
        if d["ticker"] not in playbook_tickers:
            continue
        if d["date"] < seven_days_ago or d["date"] > today:
            continue
        if d["action"] not in ("buy", "sell"):
            continue
        recent_ticker_decisions.append({
            "id": d["id"],
            "ticker": d["ticker"],
            "date": str(d["date"]),
            "action": d["action"],
            "quantity": float(d["quantity"]) if d.get("quantity") is not None else None,
            "price": float(d["price"]) if d.get("price") is not None else None,
            # Untruncated — the field exists so the executor can reason
            # about reversal justification, which needs the full nuance.
            "reasoning": d.get("reasoning") or "",
        })
    # Cap at 5 per ticker, ordered by date desc (already in `recent` order
    # because get_recent_decisions ORDER BY date DESC).
    per_ticker_count: dict[str, int] = {}
    capped = []
    for r in recent_ticker_decisions:
        n = per_ticker_count.get(r["ticker"], 0)
        if n < 5:
            capped.append(r)
            per_ticker_count[r["ticker"]] = n + 1
    recent_ticker_decisions = capped
```

Make sure `from datetime import timedelta` is imported at the top of `v2/context.py` if not already (`date` is already imported). If `timedelta` is missing, add it:

```python
from datetime import date, timedelta
```

Then add the field to the `ExecutorInput(...)` return at the bottom of the function:

```python
    return ExecutorInput(
        playbook_actions=actions,
        positions=[dict(p) for p in positions],
        account=account_info,
        attribution_summary=attribution_summary,
        recent_outcomes=recent_outcomes,
        market_outlook=playbook.get("market_outlook", "") if playbook else "No playbook available",
        risk_notes=playbook.get("risk_notes", "") if playbook else "",
        current_prices=current_prices,
        strategy_identity=strategy_identity,
        strategy_rules=strategy_rules,
        equity_summary=equity_summary,
        todays_decisions=todays_decisions,
        recent_ticker_decisions=recent_ticker_decisions,
    )
```

Note: the variable `today` is already defined at line 506 in the `todays_decisions` block. If your refactor places the new block above that line, deduplicate by reusing the existing `today = date.today()`.

- [ ] **Step 1.8: Run context tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_context.py::TestRecentTickerDecisions -v`

Expected: 3 PASS.

- [ ] **Step 1.9: Wire the field through to the JSON sent to the executor**

In `v2/agent.py`, locate `get_trading_decisions` around line 169. Find the `input_data` dict construction (around lines 186-199). Add the new key before `input_json = json.dumps(...)`:

```python
        "todays_decisions": executor_input.todays_decisions,
        "recent_ticker_decisions": executor_input.recent_ticker_decisions,
    }
    input_json = json.dumps(input_data, default=str)
```

- [ ] **Step 1.10: Update executor system prompt**

In `v2/agent.py`, locate `TRADING_SYSTEM_PROMPT` at line 116. Find the `INPUTS (as JSON object):` block (lines 120-132). Append a new entry as #13 after `12. todays_decisions`:

```
13. recent_ticker_decisions — for every ticker in today's playbook, the most recent ≤5 buy/sell decisions on that ticker in the past 7 days (with full reasoning). USE THIS to detect when today's playbook reverses a recent decision.
```

Then locate the `RULES:` section (lines 149-154). Add a new rule after `- If uncertain: HOLD`:

```
- REVERSAL JUSTIFICATION: if your decision on a ticker is the opposite action of any entry in `recent_ticker_decisions` for that ticker, your `reasoning` field MUST explicitly identify (a) the prior decision (date + action), and (b) the new evidence — fundamentals shift, catalyst resolution, price level reached — that justifies reversing. "Re-narrating the same fundamentals" is not new evidence. If you can't articulate (b), HOLD instead.
```

- [ ] **Step 1.11: Write failing test for prompt updates**

Append to `tests/v2/test_agent.py` (find the existing prompt test around line 80, add to that class or create one):

```python
class TestExecutorPromptReversalGuidance:
    def test_prompt_describes_recent_ticker_decisions_input(self):
        from v2.agent import TRADING_SYSTEM_PROMPT
        assert "recent_ticker_decisions" in TRADING_SYSTEM_PROMPT
        assert "past 7 days" in TRADING_SYSTEM_PROMPT.lower() or \
               "7 days" in TRADING_SYSTEM_PROMPT

    def test_prompt_has_reversal_justification_rule(self):
        from v2.agent import TRADING_SYSTEM_PROMPT
        text = TRADING_SYSTEM_PROMPT.lower()
        assert "reversal" in text, "executor must be told to justify reversals"
        assert "new evidence" in text, "rule must require new evidence, not re-narration"

    def test_input_json_includes_recent_ticker_decisions(self, mock_db, mock_cursor):
        """The JSON sent to Claude must include the new field."""
        from unittest.mock import patch, MagicMock
        from v2.agent import ExecutorInput, get_trading_decisions
        ei = ExecutorInput(
            playbook_actions=[], positions=[], account={},
            attribution_summary={}, recent_outcomes=[],
            market_outlook="", risk_notes="",
            recent_ticker_decisions=[
                {"ticker": "GOOGL", "date": "2026-05-04", "action": "sell",
                 "quantity": 0.17, "price": 383.02, "reasoning": "trim"},
            ],
        )

        captured = {}
        def fake_call(client, **kwargs):
            captured["messages"] = kwargs["messages"]
            resp = MagicMock()
            resp.content = [MagicMock(text='{"decisions": [], "thesis_invalidations": [], "market_summary": "", "risk_assessment": ""}')]
            resp.stop_reason = "end_turn"
            return resp

        with patch("v2.agent._call_with_retry", side_effect=fake_call), \
             patch("v2.agent.get_claude_client"):
            get_trading_decisions(ei)

        sent_json = captured["messages"][0]["content"]
        assert "recent_ticker_decisions" in sent_json
        assert "GOOGL" in sent_json
```

- [ ] **Step 1.12: Run prompt tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_agent.py::TestExecutorPromptReversalGuidance -v`

Expected: 3 PASS.

- [ ] **Step 1.13: Run the full v2 suite to catch regressions**

Run: `python3 -m pytest tests/v2/ -q`

Expected: pre-existing pass count + 8 new tests passing. No new failures.

If any pre-existing test fails because it constructs `ExecutorInput` without the new field — they should all still pass because the field defaults to `[]` via `__post_init__`. If a test fails because it checks the input JSON sent to the executor and now sees an extra key, update that test to allow the new key.

- [ ] **Step 1.14: Commit**

```bash
git add v2/agent.py v2/context.py tests/v2/test_agent.py tests/v2/test_context.py
git commit -m "$(cat <<'EOF'
feat(executor): surface recent ticker decisions to close 1-6d blind spot

Executor's existing recent_outcomes is filtered to outcome_7d IS NOT NULL
(≥7d old) and todays_decisions is today-only — leaves a 1-6 day window
where yesterday's sell is invisible to today's buy. New field surfaces
the past 7d of buy/sell decisions per playbook ticker with untruncated
reasoning, plus a Reversal Justification rule in the executor prompt
requiring new evidence (not re-narration) for opposite-side trades.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `tool_get_recent_playbooks` and Reversal Justification rule to strategist prompt

**Files:**
- Modify: `v2/database/trading_db.py` (after `get_playbook_actions` around line 469) — add `get_recent_playbooks_with_actions`
- Modify: `v2/tools.py:349-365` (after `tool_get_decision_history`) — add `tool_get_recent_playbooks` and register it
- Modify: `v2/ideation_claude.py:69-114` (`_STRATEGIST_TEMPLATE`) — add tool reference + Reversal Justification rule
- Test: `tests/v2/test_tools.py` (new tool), `tests/v2/test_ideation_claude.py` (prompt assertions), `tests/v2/test_db.py` or `tests/v2/test_db_redesign.py` (DB query)

### What this tool returns

A formatted text block listing the past N playbooks (default N=3, i.e. yesterday, day-before, three-days-back), each with their actions:

```
Recent Playbooks (3 most recent):

2026-05-04 (Playbook #87):
  ANET: SELL exit_partial_pct=50 — ANET earnings binary tomorrow, trim to lock 26% gain
  AMZN: BUY invest_dollar=300 — multi-catalyst high-conviction, AWS reaccel narrative
  GOOGL: SELL exit_partial_pct=30 — near consensus PT $387, EPS beat partly equity-gain inflated
  CRM: HOLD — defer pending geopolitical stabilization

2026-05-01 (Playbook #84):
  CRM: SELL exit_full — Thesis #205 exit trigger reached, TEAM 28% beat validates SaaS
  ...
```

This is what the strategist needs to see to reason "I said trim ANET 50% yesterday — what changed?" The current `tool_get_decision_history` shows actual *trades* but not the playbook *plans* (which differ when the executor adjusts/skips), and truncates reasoning to 60 chars. The new tool fills that gap.

- [ ] **Step 2.1: Write failing test for the DB query**

Append to `tests/v2/test_db_redesign.py` (or create class in appropriate db test file):

```python
class TestGetRecentPlaybooksWithActions:
    """Surface recent playbook history for the strategist's reversal check."""

    def test_returns_empty_when_no_playbooks(self, mock_db, mock_cursor):
        from v2.database.trading_db import get_recent_playbooks_with_actions
        mock_cursor.fetchall.return_value = []

        result = get_recent_playbooks_with_actions(n=3)

        assert result == []

    def test_returns_playbooks_with_nested_actions(self, mock_db, mock_cursor):
        from datetime import date
        from v2.database.trading_db import get_recent_playbooks_with_actions
        # Single query returns flat (playbook, action) join rows
        mock_cursor.fetchall.return_value = [
            {"pb_id": 87, "pb_date": date(2026, 5, 4), "pb_market_outlook": "fragile",
             "action_id": 401, "ticker": "ANET", "action": "sell",
             "intent_type": "exit_partial_pct", "intent_magnitude": 50,
             "reasoning": "trim pre-earnings"},
            {"pb_id": 87, "pb_date": date(2026, 5, 4), "pb_market_outlook": "fragile",
             "action_id": 402, "ticker": "AMZN", "action": "buy",
             "intent_type": "invest_dollar", "intent_magnitude": 300,
             "reasoning": "multi-catalyst"},
            {"pb_id": 84, "pb_date": date(2026, 5, 1), "pb_market_outlook": "rotational",
             "action_id": 390, "ticker": "CRM", "action": "sell",
             "intent_type": "exit_full", "intent_magnitude": None,
             "reasoning": "exit trigger reached"},
        ]

        result = get_recent_playbooks_with_actions(n=3)

        assert len(result) == 2  # two distinct playbooks
        assert result[0]["pb_id"] == 87
        assert result[0]["pb_date"] == date(2026, 5, 4)
        assert len(result[0]["actions"]) == 2
        assert result[0]["actions"][0]["ticker"] == "ANET"
        assert result[1]["pb_id"] == 84
        assert len(result[1]["actions"]) == 1

    def test_sql_orders_descending_and_limits(self, mock_db, mock_cursor):
        from v2.database.trading_db import get_recent_playbooks_with_actions
        mock_cursor.fetchall.return_value = []

        get_recent_playbooks_with_actions(n=5)

        sql = mock_cursor.execute.call_args[0][0]
        params = mock_cursor.execute.call_args[0][1]
        assert "ORDER BY" in sql.upper()
        assert "DESC" in sql.upper()
        assert 5 in params
```

- [ ] **Step 2.2: Run failing tests**

Run: `python3 -m pytest tests/v2/test_db_redesign.py::TestGetRecentPlaybooksWithActions -v`

Expected: 3 FAIL with `ImportError: cannot import name 'get_recent_playbooks_with_actions'`.

- [ ] **Step 2.3: Add the DB function**

In `v2/database/trading_db.py`, after `get_playbook_actions` (around line 473), add:

```python
def get_recent_playbooks_with_actions(n: int = 3) -> list[dict]:
    """Return the N most recent playbooks with their actions nested.

    Used by the strategist's reversal-justification flow: shows what was
    *planned* in recent sessions (not just what got executed), so the
    strategist can detect when today's plan reverses a recent one.

    Returns: list of dicts ordered by date DESC, each:
      {"pb_id", "pb_date", "pb_market_outlook", "actions": [
          {"action_id", "ticker", "action", "intent_type",
           "intent_magnitude", "reasoning"}, ...
      ]}
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                p.id              AS pb_id,
                p.date            AS pb_date,
                p.market_outlook  AS pb_market_outlook,
                pa.id             AS action_id,
                pa.ticker         AS ticker,
                pa.action         AS action,
                pa.intent_type    AS intent_type,
                pa.intent_magnitude AS intent_magnitude,
                pa.reasoning      AS reasoning
            FROM playbooks p
            LEFT JOIN playbook_actions pa ON pa.playbook_id = p.id
            WHERE p.id IN (
                SELECT id FROM playbooks ORDER BY date DESC LIMIT %s
            )
            ORDER BY p.date DESC, pa.priority ASC NULLS LAST, pa.id ASC
        """, (n,))
        rows = cur.fetchall()

    # Group rows back into playbooks. The query yields one row per
    # (playbook, action) pair; a playbook with no actions yields one row
    # with NULL action fields, which we handle by emitting empty actions.
    by_pb: dict[int, dict] = {}
    order: list[int] = []
    for row in rows:
        pb_id = row["pb_id"]
        if pb_id not in by_pb:
            by_pb[pb_id] = {
                "pb_id": pb_id,
                "pb_date": row["pb_date"],
                "pb_market_outlook": row.get("pb_market_outlook"),
                "actions": [],
            }
            order.append(pb_id)
        if row.get("action_id") is not None:
            by_pb[pb_id]["actions"].append({
                "action_id": row["action_id"],
                "ticker": row["ticker"],
                "action": row["action"],
                "intent_type": row.get("intent_type"),
                "intent_magnitude": row.get("intent_magnitude"),
                "reasoning": row.get("reasoning") or "",
            })
    return [by_pb[pb_id] for pb_id in order]
```

- [ ] **Step 2.4: Run DB tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_db_redesign.py::TestGetRecentPlaybooksWithActions -v`

Expected: 3 PASS.

- [ ] **Step 2.5: Write failing test for the new strategist tool**

Append to `tests/v2/test_tools.py` (or wherever `tool_get_decision_history` is tested):

```python
class TestToolGetRecentPlaybooks:
    """Strategist tool that surfaces yesterday's planned actions."""

    @patch("v2.tools.get_recent_playbooks_with_actions")
    def test_renders_playbooks_with_actions(self, mock_get):
        from datetime import date
        from v2.tools import tool_get_recent_playbooks
        mock_get.return_value = [
            {
                "pb_id": 87, "pb_date": date(2026, 5, 4),
                "pb_market_outlook": "fragile",
                "actions": [
                    {"action_id": 401, "ticker": "ANET", "action": "sell",
                     "intent_type": "exit_partial_pct", "intent_magnitude": 50,
                     "reasoning": "trim pre-earnings binary tomorrow"},
                ],
            },
        ]

        result = tool_get_recent_playbooks(n=3)

        assert "2026-05-04" in result
        assert "ANET" in result
        assert "SELL" in result.upper()
        assert "exit_partial_pct=50" in result
        assert "trim pre-earnings" in result

    @patch("v2.tools.get_recent_playbooks_with_actions", return_value=[])
    def test_empty_marker_when_no_playbooks(self, mock_get):
        from v2.tools import tool_get_recent_playbooks
        result = tool_get_recent_playbooks(n=3)
        assert "no recent playbooks" in result.lower() or "none" in result.lower()

    def test_tool_registered_in_definitions_and_handlers(self):
        from v2.tools import TOOL_DEFINITIONS, TOOL_HANDLERS
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "get_recent_playbooks" in names
        assert "get_recent_playbooks" in TOOL_HANDLERS
```

- [ ] **Step 2.6: Run failing tests**

Run: `python3 -m pytest tests/v2/test_tools.py::TestToolGetRecentPlaybooks -v`

Expected: 3 FAIL.

- [ ] **Step 2.7: Add the tool function and register it**

In `v2/tools.py`, find where `get_recent_decisions` is imported. Add `get_recent_playbooks_with_actions` to the imports from `v2.database.trading_db`. Then after `tool_get_decision_history` (around line 365), add:

```python
def tool_get_recent_playbooks(n: int = 3) -> str:
    """Return the past N playbooks (planned actions) so the strategist
    can detect reversals against its own recent plans before writing a
    new playbook."""
    logger.info(f"Getting recent playbooks (n={n})")
    playbooks = get_recent_playbooks_with_actions(n=n)
    if not playbooks:
        return "No recent playbooks."

    lines = [f"Recent Playbooks ({len(playbooks)} most recent):", ""]
    for pb in playbooks:
        lines.append(f"{pb['pb_date']} (Playbook #{pb['pb_id']}):")
        if not pb["actions"]:
            lines.append("  (no actions)")
        for a in pb["actions"]:
            mag = a.get("intent_magnitude")
            intent = a.get("intent_type") or ""
            mag_str = f"={mag}" if mag is not None else ""
            intent_part = f" {intent}{mag_str}" if intent else ""
            reasoning = (a.get("reasoning") or "").strip()
            lines.append(
                f"  {a['ticker']}: {a['action'].upper()}{intent_part} — {reasoning}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()
```

Then locate `TOOL_DEFINITIONS` (around line 700) and add an entry alongside `get_decision_history`:

```python
        {
            "name": "get_recent_playbooks",
            "description": (
                "Past playbooks (planned actions) for recent sessions. "
                "Use BEFORE writing today's playbook to check whether today's "
                "actions would reverse what you wrote in recent sessions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Count (default: 3)"},
                },
                "required": [],
            },
        },
```

And in `TOOL_HANDLERS` (around line 793):

```python
    "get_recent_playbooks": tool_get_recent_playbooks,
```

- [ ] **Step 2.8: Run tool tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_tools.py::TestToolGetRecentPlaybooks -v`

Expected: 3 PASS.

- [ ] **Step 2.9: Write failing test for strategist prompt updates**

Append to `TestSystemPrompts` class in `tests/v2/test_ideation_claude.py` (around line 215):

```python
    def test_strategist_prompt_mentions_get_recent_playbooks(self):
        assert "get_recent_playbooks" in CLAUDE_STRATEGIST_SYSTEM
        assert "get_recent_playbooks" in CLAUDE_SESSION_STRATEGIST_SYSTEM

    def test_strategist_prompt_has_reversal_justification_rule(self):
        for prompt in (CLAUDE_STRATEGIST_SYSTEM, CLAUDE_SESSION_STRATEGIST_SYSTEM):
            text = prompt.lower()
            assert "reversal" in text, (
                "strategist must be told to justify reversals against recent playbooks"
            )
            assert "new evidence" in text, (
                "rule must require new evidence, not re-narration"
            )
```

- [ ] **Step 2.10: Run failing prompt tests**

Run: `python3 -m pytest tests/v2/test_ideation_claude.py::TestSystemPrompts -v`

Expected: 2 NEW tests FAIL. Pre-existing tests still PASS.

- [ ] **Step 2.11: Update strategist prompts**

In `v2/ideation_claude.py`, locate `_STRATEGIST_TEMPLATE` at line 69. In the `## Tool Usage` section (lines 89-99), add after `- Use \`get_decision_history\` to review recent trading performance`:

```
- Use `get_recent_playbooks` to see what you planned in recent sessions. Always check this BEFORE writing today's playbook — if today's plan reverses a recent action on the same ticker, you must justify the reversal in your reasoning.
```

In the `## Critical Rules` section (lines 101-114), add a new rule after rule 7 (the one ending with the buy/sell intent examples):

```
8. **Reversal Justification.** Before adding a buy/sell action to today's playbook, check `get_recent_playbooks` for the same ticker. If today's action is the opposite of what you wrote in the past 7 days, your action's `reasoning` must explicitly cite (a) the prior playbook date and action, and (b) the new evidence — fundamentals shift, catalyst resolution, price level reached — that justifies reversing. Re-narrating the same fundamentals from a different angle is not new evidence; if you cannot articulate (b), do not propose the action.
```

- [ ] **Step 2.12: Run prompt tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_ideation_claude.py::TestSystemPrompts -v`

Expected: All PASS, including the 2 new ones.

- [ ] **Step 2.13: Run full v2 suite**

Run: `python3 -m pytest tests/v2/ -q`

Expected: pre-existing pass count + ~14 new tests passing across this task and Task 1. No new failures.

- [ ] **Step 2.14: Commit**

```bash
git add v2/database/trading_db.py v2/tools.py v2/ideation_claude.py \
        tests/v2/test_db_redesign.py tests/v2/test_tools.py tests/v2/test_ideation_claude.py
git commit -m "$(cat <<'EOF'
feat(strategist): add get_recent_playbooks tool + reversal-justification rule

Strategist had no continuity artifact: each session writes a fresh
playbook with no anchoring on yesterday's plan. New tool surfaces the
past 3 playbooks with their actions (untruncated reasoning), and a new
Critical Rule 8 requires the strategist to cite (a) the prior decision
and (b) what new evidence justifies a reversal — preventing same-day
or next-day re-narrations of the same fundamentals.

Mirrors the executor-side guard added in the previous commit so both
LLMs in the loop see and reason about the same continuity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

# PHASE B — Diagnostic + self-correction layer

Phase A removes the structural blind spots. Phase B gives the system aggregate visibility so reflection can write rules against any residual oscillation. Without Phase A, Phase B alone relies on the LLM to reason its way out of a problem the system could just prevent. Without Phase B, Phase A has no automated learning signal if the LLM ignores the new context.

## Task 3: Add `analyze_round_trips()` to `v2/patterns.py`

**Files:**
- Modify: `v2/patterns.py` (add `RoundTrip` dataclass after `ConfidenceCorrelation` at line 53; add `analyze_round_trips()` after `analyze_confidence_correlation()` at line 226)
- Test: `tests/v2/test_patterns.py` (append `TestAnalyzeRoundTrips` class)

This is identical in shape to Task 1 of the prior `2026-05-08-flip-flop-reflection-evidence.md` plan — the round-trip query and dataclass are unchanged.

- [ ] **Step 3.1: Write failing tests for the dataclass and function**

Append to `tests/v2/test_patterns.py`:

```python
class TestAnalyzeRoundTrips:
    """Tests for analyze_round_trips() — surfaces flip-flop patterns."""

    def test_returns_empty_list_when_no_pairs(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []
        assert analyze_round_trips(days=30, gap_days=7, min_pairs=2) == []

    def test_returns_round_trip_objects(self, mock_db):
        from datetime import date
        from v2.patterns import RoundTrip, analyze_round_trips
        mock_db.fetchall.return_value = [
            {"ticker": "GOOGL", "pair_count": 11,
             "first_date": date(2026, 4, 15), "last_date": date(2026, 5, 6)},
            {"ticker": "CRM", "pair_count": 9,
             "first_date": date(2026, 3, 10), "last_date": date(2026, 5, 5)},
        ]

        result = analyze_round_trips(days=60, gap_days=14, min_pairs=2)

        assert len(result) == 2
        assert result[0] == RoundTrip(
            ticker="GOOGL", pair_count=11,
            first_date=date(2026, 4, 15), last_date=date(2026, 5, 6),
        )
        assert result[1].ticker == "CRM"

    def test_sql_self_joins_decisions_on_opposite_action(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []
        analyze_round_trips(days=30, gap_days=7, min_pairs=2)

        sql = mock_db.execute.call_args[0][0]
        assert "decisions" in sql.lower()
        assert "b.action <> a.action" in sql
        assert "action IN ('buy', 'sell')" in sql or "action in ('buy','sell')" in sql.lower()
        assert "GROUP BY" in sql.upper()
        assert "HAVING" in sql.upper()

    def test_passes_window_and_gap_parameters(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []
        analyze_round_trips(days=45, gap_days=10, min_pairs=3)

        params = mock_db.execute.call_args[0][1]
        assert 45 in params
        assert 10 in params
        assert 3 in params

    def test_default_parameters(self, mock_db):
        from v2.patterns import analyze_round_trips
        mock_db.fetchall.return_value = []
        analyze_round_trips()

        params = mock_db.execute.call_args[0][1]
        assert 30 in params
        assert 7 in params
        assert 2 in params
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `python3 -m pytest tests/v2/test_patterns.py::TestAnalyzeRoundTrips -v`

Expected: 5 FAIL with `ImportError`.

- [ ] **Step 3.3: Add the dataclass to `v2/patterns.py`**

In `v2/patterns.py`, after the `ConfidenceCorrelation` dataclass (around line 53), add:

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

- [ ] **Step 3.4: Add the function**

After `analyze_confidence_correlation()` (around line 226), add:

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

- [ ] **Step 3.5: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_patterns.py::TestAnalyzeRoundTrips -v`

Expected: 5 PASS.

- [ ] **Step 3.6: Commit**

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

## Task 4: Append round-trip section to `tool_get_session_summary`

**Files:**
- Modify: `v2/strategy.py:274-338` (extend `tool_get_session_summary`)
- Test: `tests/v2/test_strategy.py` (extend `TestToolGetSessionSummary`)

When round-trips exist:
```
Round-Trips (past 30d, ≥2 opposing actions ≤7d apart):
  GOOGL: 11 pairs (2026-04-15 to 2026-05-06)
  CRM: 9 pairs (2026-03-10 to 2026-05-05)
```

When none:
```
Round-Trips (past 30d, ≥2 opposing actions ≤7d apart): none.
```

- [ ] **Step 4.1: Write failing tests**

Append to `TestToolGetSessionSummary` class in `tests/v2/test_strategy.py`:

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

        assert "Round-Trips" in result
        assert "none" in result.lower()

    @patch("v2.strategy.analyze_round_trips")
    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_round_trips_uses_30d_window_7d_gap_min_2(
        self, mock_decisions, mock_attr, mock_round_trips, mock_db, mock_cursor,
    ):
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

        for i in range(5):
            assert f"T{i}: {10-i} pairs" in result
        assert "T5:" not in result
        assert "T6:" not in result
        assert "7 total" in result or "(7 tickers)" in result
```

- [ ] **Step 4.2: Run failing tests**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestToolGetSessionSummary -v`

Expected: 4 NEW tests FAIL.

- [ ] **Step 4.3: Add import to `v2/strategy.py`**

After `from .formation import build_formation_context` (around line 23), add:

```python
from .patterns import analyze_round_trips
```

- [ ] **Step 4.4: Append the round-trip section**

In `v2/strategy.py`, locate the end of `tool_get_session_summary` (the `return "\n".join(lines)` near line 338). Just before that return statement, add:

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

- [ ] **Step 4.5: Run tests to verify they pass**

Run: `python3 -m pytest tests/v2/test_strategy.py::TestToolGetSessionSummary -v`

Expected: All PASS (4 existing + 4 new = 8 total).

- [ ] **Step 4.6: Run full v2 suite**

Run: `python3 -m pytest tests/v2/ -q`

Expected: pre-existing pass count + total ~22 new tests passing. No new failures.

- [ ] **Step 4.7: Commit**

```bash
git add v2/strategy.py tests/v2/test_strategy.py
git commit -m "$(cat <<'EOF'
feat(strategy): surface round-trip evidence in session summary

Reflection LLM already calls get_session_summary every session; this
appends a Round-Trips section so the strategist can see ticker-level
flip-flop counts in aggregate. Pairs with the executor-side and
strategist-side guards in the prior commits — those prevent the
behavior, this lets reflection notice and write rules against any
residual occurrence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Verify against prod data

Smoke test against the running prod DB to confirm all four changes produce the expected data flow before the next session runs against this code.

- [ ] **Step 5.1: Confirm prod db is reachable**

Run: `docker compose ps db`

If not running: `docker compose up -d db` (this starts only db, no trading agent — safe).

- [ ] **Step 5.2: Verify `analyze_round_trips()` against prod**

Run:
```bash
set -a; source .env; set +a; \
docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
WITH bs AS (
  SELECT id, date, ticker, action FROM decisions
  WHERE date > CURRENT_DATE - INTERVAL '30 days' AND action IN ('buy','sell')
)
SELECT a.ticker, COUNT(*) AS pairs
FROM bs a JOIN bs b ON a.ticker = b.ticker AND b.id > a.id
                   AND b.action <> a.action AND (b.date - a.date) <= 7
GROUP BY a.ticker HAVING COUNT(*) >= 2 ORDER BY pairs DESC;"
```

Expected (as of 2026-05-08, may drift): GOOGL ~11, CRM/NVDA/AMZN in the high single digits. If the numbers are dramatically lower (e.g. all 0 or 1), check the SQL — most likely the gap_days arithmetic.

- [ ] **Step 5.3: Verify `get_recent_playbooks_with_actions` returns reasonable data**

Run:
```bash
docker compose up -d trading 2>&1 | tail -3
docker compose exec -T trading python3 -c "
from v2.database.trading_db import get_recent_playbooks_with_actions
for pb in get_recent_playbooks_with_actions(n=3):
    print(f\"{pb['pb_date']} (#{pb['pb_id']}): {len(pb['actions'])} actions\")
    for a in pb['actions'][:3]:
        print(f\"  {a['ticker']} {a['action']} {a.get('intent_type')}={a.get('intent_magnitude')}\")
"
```

Expected: 3 dated playbook entries (most recent first), each with 2-5 actions and reasoning text. If the trading container isn't running or doesn't have the DB env wired, fall back to a direct SQL spot check via psql.

- [ ] **Step 5.4: Spot-check `build_executor_input` produces `recent_ticker_decisions`**

Run:
```bash
docker compose exec -T trading python3 -c "
from v2.context import build_executor_input
ei = build_executor_input(account_info={'equity': 10000, 'cash': 1000, 'buying_power': 1000})
print(f'playbook_actions: {len(ei.playbook_actions)}')
print(f'recent_ticker_decisions: {len(ei.recent_ticker_decisions)}')
for r in ei.recent_ticker_decisions[:5]:
    print(f\"  {r['date']} {r['action'].upper()} {r['ticker']} reasoning_len={len(r['reasoning'])}\")
"
```

Expected: if today's playbook has actions on GOOGL/AMZN/CRM (likely given recent activity), some `recent_ticker_decisions` entries appear with reasoning_len > 60 (i.e. untruncated).

- [ ] **Step 5.5: Spot-check the strategist tool**

Run:
```bash
docker compose exec -T trading python3 -c "
from v2.tools import tool_get_recent_playbooks
print(tool_get_recent_playbooks(n=3))
"
```

Expected: human-readable text listing the past 3 playbooks with actions and reasoning. The strategist will see exactly this.

- [ ] **Step 5.6: Spot-check the reflection summary**

Run:
```bash
docker compose exec -T trading python3 -c "
from v2.strategy import tool_get_session_summary
print(tool_get_session_summary())
"
```

Expected: the existing summary plus a `Round-Trips (past 30d...)` section near the end.

- [ ] **Step 5.7: No commit needed for verification**

If all six checks look right, the work is ready to merge. If any diverge unexpectedly, pause and inspect — the most common cause is a stale prod schema or a typo in a SQL parameter ordering.

---

## What we are NOT doing in this plan (and why)

- **Not adding executor-side hard cooldown enforcement.** A deterministic gate ("refuse opposite-side trades within N days unless reasoning contains keyword X") is the obvious next step if Phase A's LLM-side guidance proves insufficient. We defer it to see whether (a) showing the executor `recent_ticker_decisions` and (b) requiring strategist Reversal Justification is enough on its own. Adding the gate too early hides whether the LLM-side fix worked.
- **Not making Rule 27 numeric.** Already in memory as a known issue. Phase B (round-trip aggregates) gives the reflection stage the evidence to revise Rule 27 itself via the existing learning loop. If after 5 sessions the strategist hasn't proposed a numeric replacement, we'll write one directly.
- **Not changing `recent_outcomes` filter.** That field has a specific purpose (outcome calibration with realized P&L) and the `outcome_7d IS NOT NULL` filter is correct for that purpose. We add a *new* field for the recent-trade-context use case rather than overloading an existing one.
- **Not raising `display_limit=10` in `tool_get_session_summary`.** The 10-row cap is intentional — pre-existing comment at `v2/strategy.py:281-285` notes that earlier truncation biased rule proposals toward recency. The Phase B aggregate (round-trip counts) gives reflection the multi-week view without unbounding the per-row display.
- **Not persisting playbook diffs to a new table.** A simpler path is to recompute the diff on demand from `playbooks` + `playbook_actions` (which already store everything). A new table would duplicate state that's already canonical.
- **Not changing the executor model.** Haiku is the right model for executor by cost/latency. The blind spot is in the input data, not the model.

---

## Sequencing recommendation for execution

If you choose subagent-driven execution, dispatch tasks in this order with a review between each:

1. **Task 1** (executor field + prompt) — most critical, biggest single fix
2. **Task 2** (strategist tool + prompt) — pairs with Task 1 to cover both LLMs
3. **Tasks 3 + 4 in parallel** — both are isolated additions to patterns.py and strategy.py, no shared files
4. **Task 5** — verification, must run last after all code is in place

After Task 2 lands, the next session run will produce different behavior — worth observing one or two sessions before merging Tasks 3+4 just to confirm Phase A is working as intended.

---

## Self-Review Checklist

- **Spec coverage:** Plan addresses all four structural failures from the dive-in synthesis (executor blind spot, playbook discontinuity, missing strategist guidance, reflection aggregation gap). All four map to tasks: blind spot → Task 1; discontinuity → Task 2; strategist guidance → Task 2; reflection aggregation → Tasks 3+4.
- **Placeholders:** None. Every code block contains final code; every command is concrete.
- **Type consistency:** `RoundTrip` dataclass shape consistent across patterns.py, strategy.py, both test files. `recent_ticker_decisions` field shape (`list[dict]` with keys `id`, `ticker`, `date`, `action`, `quantity`, `price`, `reasoning`) consistent across `ExecutorInput` definition, `build_executor_input` populator, JSON serialization in `get_trading_decisions`, and tests. `get_recent_playbooks_with_actions` returns `list[dict]` shape (`pb_id`, `pb_date`, `pb_market_outlook`, `actions[]`) consistent across DB function, tool function, and tests.
- **Mock paths:** New tests use existing `mock_db`/`mock_cursor` fixtures from `tests/v2/conftest.py:75-104`. New `analyze_round_trips` patch target is `v2.strategy.analyze_round_trips` — works only with `from .patterns import analyze_round_trips` (Step 4.3), not module-attribute access.
- **Backwards compat:** New `recent_ticker_decisions` field on `ExecutorInput` defaults to `[]` via `__post_init__`. New tool registration is additive. New DB function is additive. No existing call sites need updating beyond the tests.
- **Cross-task dependencies:** Task 4 imports from Task 3 (`from .patterns import analyze_round_trips`). Task 2 imports from itself only (`get_recent_playbooks_with_actions` is added in same task). Task 1 has no cross-task dependencies. Order: 1 → 2 → 3 → 4 → 5 is safe; 3 and 4 could be reordered as 4 → 3 only if Task 4's import is added speculatively.
