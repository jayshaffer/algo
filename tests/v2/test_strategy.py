"""Tests for v2/strategy.py — strategy reflection stage (Stage 4)."""

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from tests.v2.conftest import (
    make_decision_row,
    make_strategy_state_row,
)
from v2.strategy import run_strategy_reflection


class TestUtcDateHelper:
    """T2.7: `_utc_date` normalises tz-aware and naive datetimes to a
    UTC-frame date so throttle/tenure comparisons stay consistent.
    """

    def test_naive_datetime_treated_as_utc(self):
        from datetime import datetime as dt_cls

        from v2.strategy import _utc_date
        # 23:30 — naive datetime; we treat the value itself as UTC.
        naive = dt_cls(2026, 5, 1, 23, 30, 0)
        assert _utc_date(naive) == naive.date()

    def test_aware_datetime_converted_to_utc(self):
        from datetime import datetime as dt_cls
        from datetime import timedelta
        from datetime import timezone as tz

        from v2.strategy import _utc_date
        # 23:30 ET = 03:30 UTC the next day. Old code's `.date()` on the
        # aware ET datetime keeps ET's date — but for consistent age math
        # we want the UTC date (May 2, not May 1).
        et = tz(timedelta(hours=-4))  # EDT
        aware = dt_cls(2026, 5, 1, 23, 30, 0, tzinfo=et)
        assert _utc_date(aware) == date(2026, 5, 2)


class TestStrategyReflectionResult:
    def test_dataclass_fields(self):
        from v2.strategy import StrategyReflectionResult
        result = StrategyReflectionResult(
            rules_proposed=2,
            rules_retired=1,
            identity_updated=True,
            memo_written=True,
            input_tokens=1000,
            output_tokens=500,
            turns_used=3,
        )
        assert result.rules_proposed == 2
        assert result.rules_retired == 1
        assert result.identity_updated is True
        assert result.memo_written is True
        assert result.input_tokens == 1000
        assert result.output_tokens == 500
        assert result.turns_used == 3


class TestToolUpdateStrategyIdentity:
    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_updates_existing_identity(self, mock_get, mock_insert):
        from datetime import timedelta

        from v2.strategy import tool_update_strategy_identity
        mock_get.return_value = make_strategy_state_row(
            version=2, created_at=datetime.now() - timedelta(days=5),
        )
        mock_insert.return_value = 3

        result = tool_update_strategy_identity(
            identity_text="Contrarian trader",
            risk_posture="aggressive",
            sector_biases={"energy": "overweight"},
            preferred_signals=["macro"],
            avoided_signals=["legal"],
        )
        # clear is now folded into insert_strategy_state; only one call needed
        mock_insert.assert_called_once()
        assert "version 3" in result

    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_bootstraps_first_identity(self, mock_get, mock_insert):
        from v2.strategy import tool_update_strategy_identity
        mock_get.return_value = None
        mock_insert.return_value = 1

        result = tool_update_strategy_identity(
            identity_text="New trader",
            risk_posture="conservative",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )
        mock_insert.assert_called_once()
        assert "version 1" in result

    def test_insert_failure_does_not_wipe_prior_state(self, mock_db, mock_cursor):
        """Clear+insert must be atomic. If the INSERT fails, the prior
        is_current=TRUE row must remain intact (transaction rolls back the
        UPDATE that cleared it).
        """
        from v2.database.trading_db import insert_strategy_state

        # Simulate the INSERT raising mid-cursor; the surrounding get_cursor
        # context manager rolls back on exception.
        cur = mock_cursor
        cur.execute.side_effect = [
            None,                               # UPDATE clears prior is_current
            Exception("simulated INSERT crash") # INSERT fails
        ]

        try:
            insert_strategy_state(
                identity_text="new",
                risk_posture="aggressive",
                sector_biases={},
                preferred_signals=[],
                avoided_signals=[],
                version=2,
            )
            raise AssertionError("expected the simulated INSERT to raise")
        except Exception as e:
            assert "simulated INSERT crash" in str(e)

        # Both UPDATE and INSERT executed against the SAME cursor (i.e. the
        # same connection / transaction). When the get_cursor context exits
        # via exception, conn.rollback() runs — the UPDATE is undone.
        # We verify the cursor saw both calls and no separate connection
        # was opened for the clear.
        assert cur.execute.call_count == 2, (
            f"Expected UPDATE+INSERT on one cursor; got {cur.execute.call_count} calls"
        )


class TestToolProposeRule:
    @patch("v2.strategy.insert_strategy_rule")
    def test_creates_rule(self, mock_insert):
        from v2.strategy import reset_session, tool_propose_rule
        mock_insert.return_value = 5
        reset_session()

        result = tool_propose_rule(
            rule_text="Favor earnings signals",
            category="news_signal:earnings",
            direction="preference",
            confidence=0.7,
            supporting_evidence="62% win rate",
        )
        mock_insert.assert_called_once()
        assert "5" in result
        assert "Created rule" in result

    @patch("v2.strategy.insert_strategy_rule")
    def test_rejects_invalid_confidence(self, mock_insert):
        from v2.strategy import tool_propose_rule

        result = tool_propose_rule(
            rule_text="Favor earnings signals",
            category="news_signal:earnings",
            direction="preference",
            confidence=1.5,
            supporting_evidence="62% win rate",
        )

        assert "confidence" in result
        mock_insert.assert_not_called()

    @patch("v2.strategy.insert_strategy_rule")
    def test_rejects_empty_evidence(self, mock_insert):
        from v2.strategy import tool_propose_rule

        result = tool_propose_rule(
            rule_text="Favor earnings signals",
            category="news_signal:earnings",
            direction="preference",
            confidence=0.7,
            supporting_evidence="",
        )

        assert "supporting_evidence" in result
        mock_insert.assert_not_called()

    @patch("v2.strategy.get_active_strategy_rules")
    @patch("v2.strategy.insert_strategy_rule")
    def test_rejects_exact_duplicate_active_rule(self, mock_insert, mock_active):
        from v2.strategy import tool_propose_rule

        mock_active.return_value = [
            {"id": 9, "rule_text": "Favor earnings signals"},
        ]

        result = tool_propose_rule(
            rule_text=" favor   earnings signals ",
            category="news_signal:earnings",
            direction="preference",
            confidence=0.7,
            supporting_evidence="62% win rate",
        )

        assert "already covers" in result
        mock_insert.assert_not_called()

    @patch("v2.strategy.insert_strategy_rule")
    def test_proposal_session_cap(self, mock_insert):
        """T2.6: per-session proposal cap mirrors the retirement cap so the
        rule table can only grow at a measured rate from either side.
        """
        from v2.strategy import (
            MAX_PROPOSALS_PER_SESSION,
            reset_session,
            tool_propose_rule,
        )
        mock_insert.side_effect = list(range(100, 200))
        reset_session()

        for i in range(MAX_PROPOSALS_PER_SESSION):
            r = tool_propose_rule(
                rule_text=f"rule {i}",
                category="news_signal:earnings",
                direction="preference",
                confidence=0.5,
                supporting_evidence="evidence",
            )
            assert "Created rule" in r

        # Next call hits the cap; no further DB writes.
        capped = tool_propose_rule(
            rule_text="extra",
            category="news_signal:earnings",
            direction="preference",
            confidence=0.5,
            supporting_evidence="evidence",
        )
        assert "limit reached" in capped
        assert mock_insert.call_count == MAX_PROPOSALS_PER_SESSION

    @patch("v2.strategy.insert_strategy_rule")
    def test_proposals_isolated_across_threads(self, mock_insert):
        """T2.6: per-session proposal counter must not leak between
        concurrent invocations. Mirrors the retirement-counter test."""
        import threading

        from v2.strategy import (
            MAX_PROPOSALS_PER_SESSION,
            reset_session,
            tool_propose_rule,
        )

        mock_insert.side_effect = list(range(1, 200))

        results: list[str] = []
        barrier = threading.Barrier(2)

        def worker(label: str):
            reset_session()
            barrier.wait()
            local: list[str] = []
            for i in range(MAX_PROPOSALS_PER_SESSION + 1):
                r = tool_propose_rule(
                    rule_text=f"{label}-{i}",
                    category="news_signal:earnings",
                    direction="preference",
                    confidence=0.5,
                    supporting_evidence="ev",
                )
                local.append(r)
            results.extend(local)

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        created = [r for r in results if "Created rule" in r]
        capped = [r for r in results if "limit reached" in r]
        assert len(created) == MAX_PROPOSALS_PER_SESSION * 2
        assert len(capped) == 2


class TestToolRetireRule:
    @patch("v2.strategy.retire_strategy_rule")
    @patch("v2.database.trading_db.get_strategy_rule")
    def test_retires_existing_rule(self, mock_get_rule, mock_retire):
        from datetime import datetime, timedelta

        from v2.strategy import reset_session, tool_retire_rule
        mock_get_rule.return_value = {
            "id": 1, "status": "active",
            "created_at": datetime.now() - timedelta(days=10),
        }
        mock_retire.return_value = True
        reset_session()
        result = tool_retire_rule(rule_id=1, reason="No longer predictive")
        assert "Retired" in result

    @patch("v2.strategy.retire_strategy_rule")
    @patch("v2.database.trading_db.get_strategy_rule")
    def test_handles_not_found(self, mock_get_rule, mock_retire):
        from v2.strategy import tool_retire_rule
        mock_get_rule.return_value = None
        mock_retire.return_value = False
        result = tool_retire_rule(rule_id=999, reason="test")
        assert "not found" in result.lower() or "Error" in result

    @patch("v2.strategy.retire_strategy_rule")
    @patch("v2.database.trading_db.get_strategy_rule")
    def test_passes_reason_to_db(self, mock_get_rule, mock_retire):
        """retire_rule should pass the reason to the database function."""
        from datetime import datetime, timedelta

        from v2.strategy import reset_session, tool_retire_rule
        mock_get_rule.return_value = {
            "id": 5, "status": "active",
            "created_at": datetime.now() - timedelta(days=10),
        }
        mock_retire.return_value = True
        reset_session()
        tool_retire_rule(rule_id=5, reason="Superseded by structural enforcement")
        mock_retire.assert_called_once_with(rule_id=5, reason="Superseded by structural enforcement")

    @patch("v2.strategy.retire_strategy_rule")
    @patch("v2.database.trading_db.get_strategy_rule")
    def test_session_retirements_isolated_across_threads(self, mock_get_rule, mock_retire):
        """P3.36: per-session retirement counter must not leak between
        concurrent invocations. ContextVar gives each thread its own copy
        so paper + prod sharing one Python process can't clobber each
        other's MAX_RETIREMENTS_PER_SESSION cap."""
        import threading
        from datetime import datetime, timedelta

        from v2.strategy import reset_session, tool_retire_rule

        mock_get_rule.return_value = {
            "id": 1, "status": "active",
            "created_at": datetime.now() - timedelta(days=10),
        }
        mock_retire.return_value = True

        results: list[str] = []
        barrier = threading.Barrier(2)

        def worker(thread_label: str):
            reset_session()
            barrier.wait()
            r1 = tool_retire_rule(rule_id=1, reason=f"{thread_label}-r1")
            r2 = tool_retire_rule(rule_id=2, reason=f"{thread_label}-r2")
            # Each worker should successfully retire 2 rules; the 3rd call
            # would hit the per-session cap.
            r3 = tool_retire_rule(rule_id=3, reason=f"{thread_label}-r3")
            results.extend([r1, r2, r3])

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread did 3 calls; first two succeeded, third hit the cap.
        retired = [r for r in results if "Retired" in r]
        capped = [r for r in results if "limit reached" in r]
        assert len(retired) == 4, (
            f"Expected 2 retirements per thread (4 total), got {len(retired)}: {results}"
        )
        assert len(capped) == 2, (
            f"Expected 1 cap-hit per thread (2 total), got {len(capped)}"
        )


class TestToolWriteStrategyMemo:
    @patch("v2.strategy.insert_strategy_memo")
    @patch("v2.strategy.get_current_strategy_state")
    def test_writes_memo(self, mock_state, mock_insert):
        from v2.strategy import tool_write_strategy_memo
        mock_state.return_value = make_strategy_state_row()
        mock_insert.return_value = 1

        result = tool_write_strategy_memo(
            memo_type="reflection",
            content="Session showed strong momentum plays.",
        )
        mock_insert.assert_called_once()
        assert "Memo written" in result

    @patch("v2.strategy.insert_strategy_memo")
    @patch("v2.strategy.get_current_strategy_state")
    def test_handles_no_strategy_state(self, mock_state, mock_insert):
        from v2.strategy import tool_write_strategy_memo
        mock_state.return_value = None
        mock_insert.return_value = 1

        result = tool_write_strategy_memo(
            memo_type="reflection",
            content="First session reflection.",
        )
        # strategy_state_id should be None
        call_kwargs = mock_insert.call_args
        assert call_kwargs.kwargs.get("strategy_state_id") is None or \
               call_kwargs[1].get("strategy_state_id") is None or \
               call_kwargs[0][3] is None  # positional

    def test_write_strategy_memo_tool_passes_session_id(self):
        from v2.strategy import tool_write_strategy_memo
        with patch("v2.strategy.insert_strategy_memo", return_value=1) as mock_ins, \
             patch("v2.strategy.get_current_strategy_state", return_value=None):
            tool_write_strategy_memo(memo_type="notes", content="t", session_id=42)
        assert mock_ins.call_args.kwargs.get("session_id") == 42


class TestToolGetSessionSummary:
    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_returns_summary_with_decisions(self, mock_decisions, mock_attr, mock_db, mock_cursor):
        from v2.strategy import tool_get_session_summary
        mock_decisions.return_value = [make_decision_row()]
        mock_attr.return_value = "Attribution data here"
        # Signal linkage query
        mock_cursor.fetchall.return_value = [
            {"decision_id": 1, "signal_type": "news_signal", "signal_category": "earnings"},
        ]

        result = tool_get_session_summary()
        assert "Decisions" in result
        assert "AAPL" in result
        assert "Attribution" in result
        assert "news_signal:earnings" in result

    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_handles_no_decisions(self, mock_decisions, mock_attr):
        from v2.strategy import tool_get_session_summary
        mock_decisions.return_value = []
        mock_attr.return_value = "No data"

        result = tool_get_session_summary()
        assert "No recent decisions" in result

    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_orphan_fk_guards_present_in_sql(
        self, mock_decisions, mock_attr, mock_db, mock_cursor,
    ):
        """T1.11: SQL must LEFT JOIN theses and filter all three orphan
        signal types so reflection doesn't read broken FK rows as evidence.
        """
        from v2.strategy import tool_get_session_summary
        mock_decisions.return_value = [make_decision_row()]
        mock_attr.return_value = "Attribution data here"
        mock_cursor.fetchall.return_value = []

        tool_get_session_summary()

        sql = mock_cursor.execute.call_args[0][0]
        assert "LEFT JOIN theses" in sql, "Must LEFT JOIN theses for thesis orphan filter"
        assert "ds.signal_type != 'news_signal' OR ns.id IS NOT NULL" in sql
        assert "ds.signal_type != 'macro_signal' OR ms.id IS NOT NULL" in sql
        assert "ds.signal_type != 'thesis' OR t.id IS NOT NULL" in sql

    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_orphan_thesis_excluded_from_signal_labels(
        self, mock_decisions, mock_attr, mock_db, mock_cursor,
    ):
        """If the SQL filtering works, an orphan thesis row simply doesn't
        come back from the join; the rendered signal label list for that
        decision is empty.
        """
        from v2.strategy import tool_get_session_summary
        mock_decisions.return_value = [make_decision_row()]
        mock_attr.return_value = "Attribution data here"
        # Simulate the DB filtering out the orphan: only the valid news_signal
        # comes back. The orphan thesis row is suppressed by the WHERE clause.
        mock_cursor.fetchall.return_value = [
            {"decision_id": 1, "signal_type": "news_signal", "signal_category": "earnings"},
        ]

        result = tool_get_session_summary()
        assert "news_signal:earnings" in result
        assert "thesis:" not in result, (
            "Orphan thesis label leaked through to reflection text"
        )

    @patch("v2.strategy.analyze_round_trips")
    @patch("v2.strategy.get_attribution_summary")
    @patch("v2.strategy.get_recent_decisions")
    def test_renders_round_trips_when_present(
        self, mock_decisions, mock_attr, mock_round_trips, mock_db, mock_cursor,
    ):
        from datetime import date

        from v2.patterns import RoundTrip
        from v2.strategy import tool_get_session_summary
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

        from v2.patterns import RoundTrip
        from v2.strategy import tool_get_session_summary
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


class TestStrategyToolDefinitions:
    def test_all_tools_defined(self):
        from v2.strategy import STRATEGY_TOOL_DEFINITIONS
        names = [d.get("name") for d in STRATEGY_TOOL_DEFINITIONS]
        assert "get_strategy_identity" in names
        assert "get_strategy_rules" in names
        assert "get_strategy_history" in names
        assert "get_session_summary" in names
        assert "update_strategy_identity" in names
        assert "propose_rule" in names
        assert "retire_rule" in names
        assert "write_strategy_memo" in names

    def test_handlers_match_definitions(self):
        from v2.strategy import STRATEGY_TOOL_DEFINITIONS, STRATEGY_TOOL_HANDLERS
        for defn in STRATEGY_TOOL_DEFINITIONS:
            name = defn["name"]
            assert name in STRATEGY_TOOL_HANDLERS, f"Missing handler for {name}"


class TestCountActions:
    def test_counts_proposed_rules(self):
        from v2.strategy import _count_actions
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "Created rule ID 1: test"},
                {"type": "tool_result", "content": "Created rule ID 2: test2"},
            ]},
        ]
        proposed, retired, identity_updated, memo_written = _count_actions(messages)
        assert proposed == 2

    def test_counts_retired_rules(self):
        from v2.strategy import _count_actions
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "Retired rule ID 1. Reason: stale"},
            ]},
        ]
        proposed, retired, identity_updated, memo_written = _count_actions(messages)
        assert retired == 1

    def test_detects_identity_update(self):
        from v2.strategy import _count_actions
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "Strategy identity updated to version 2 (ID: 3)"},
            ]},
        ]
        proposed, retired, identity_updated, memo_written = _count_actions(messages)
        assert identity_updated is True

    def test_detects_memo_written(self):
        from v2.strategy import _count_actions
        messages = [
            {"role": "user", "content": [
                {"type": "tool_result", "content": "Memo written (ID: 1)"},
            ]},
        ]
        proposed, retired, identity_updated, memo_written = _count_actions(messages)
        assert memo_written is True

    def test_empty_messages(self):
        from v2.strategy import _count_actions
        proposed, retired, identity_updated, memo_written = _count_actions([])
        assert proposed == 0
        assert retired == 0
        assert identity_updated is False
        assert memo_written is False

    def test_soft_guard_warning_does_not_count_as_identity_update(self):
        """P3.33: tool_update_strategy_identity's soft-guard returns a string
        like 'Warning: Identity was updated within the last 3 days...'. The
        old loose substring `"identity updated"` was brittle — if the
        warning were ever rephrased to drop "was", reflection would
        report identity_updated=True even when the guard rejected the
        update. Use the unique success-message prefix instead."""
        from v2.strategy import _count_actions
        # Both the actual current warning AND a hypothetical rephrasing
        # without the "was" gap must NOT trigger identity_updated.
        for guard_text in [
            "Warning: Identity was updated within the last 3 days "
            "(v2 on 2026-04-30). Consider writing a memo instead.",
            # Defense-in-depth: hypothetical future rephrasing.
            "Warning: Identity updated 2 days ago — consider waiting.",
        ]:
            messages = [
                {"role": "user", "content": [
                    {"type": "tool_result", "content": guard_text},
                ]},
            ]
            _, _, identity_updated, _ = _count_actions(messages)
            assert identity_updated is False, (
                f"Soft-guard text {guard_text!r} should not count as a real "
                "identity update — only the success message should."
            )


class TestRunStrategyReflection:
    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    @patch("v2.strategy.build_formation_context", return_value="")
    def test_returns_reflection_result(self, mock_formation, mock_loop, mock_client):
        from v2.claude_client import AgenticLoopResult
        from v2.strategy import StrategyReflectionResult, run_strategy_reflection

        mock_loop.return_value = AgenticLoopResult(
            messages=[
                {"role": "user", "content": "Begin reflection"},
                {"role": "assistant", "content": [MagicMock(text="Done reflecting", type="text")]},
            ],
            turns_used=2,
            stop_reason="end_turn",
            input_tokens=1000,
            output_tokens=500,
        )

        result = run_strategy_reflection()
        assert isinstance(result, StrategyReflectionResult)
        assert result.turns_used == 2
        assert result.input_tokens == 1000
        assert result.output_tokens == 500

    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    @patch("v2.strategy.build_formation_context", return_value="")
    def test_passes_system_prompt_and_tools(self, mock_formation, mock_loop, mock_client):
        from v2.claude_client import AgenticLoopResult
        from v2.strategy import STRATEGY_REFLECTION_SYSTEM, STRATEGY_TOOL_DEFINITIONS, STRATEGY_TOOL_HANDLERS, run_strategy_reflection

        mock_loop.return_value = AgenticLoopResult(
            messages=[],
            turns_used=1,
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=50,
        )

        run_strategy_reflection(model="claude-opus-4-6", max_turns=5)

        call_kwargs = mock_loop.call_args
        assert call_kwargs.kwargs["system"] == STRATEGY_REFLECTION_SYSTEM
        assert call_kwargs.kwargs["tools"] == STRATEGY_TOOL_DEFINITIONS
        # `get_session_summary` and `write_strategy_memo` are wrapped by
        # session-aware partials; check the remaining handlers identity-match
        # and the wrapped slots are present.
        passed_handlers = call_kwargs.kwargs["tool_handlers"]
        assert "get_session_summary" in passed_handlers
        assert "write_strategy_memo" in passed_handlers
        for name, fn in STRATEGY_TOOL_HANDLERS.items():
            if name in ("get_session_summary", "write_strategy_memo"):
                continue
            assert passed_handlers[name] is fn
        assert call_kwargs.kwargs["max_turns"] == 5
        assert call_kwargs.kwargs["model"] == "claude-opus-4-6"

    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    @patch("v2.strategy.build_formation_context", return_value="")
    def test_counts_actions_from_messages(self, mock_formation, mock_loop, mock_client):
        from v2.claude_client import AgenticLoopResult
        from v2.strategy import run_strategy_reflection

        mock_loop.return_value = AgenticLoopResult(
            messages=[
                {"role": "user", "content": [
                    {"type": "tool_result", "content": "Created rule ID 1: test"},
                    {"type": "tool_result", "content": "Retired rule ID 2. Reason: stale"},
                    {"type": "tool_result", "content": "Strategy identity updated to version 2 (ID: 3)"},
                    {"type": "tool_result", "content": "Memo written (ID: 1)"},
                ]},
                {"role": "assistant", "content": [MagicMock(text="Done", type="text")]},
            ],
            turns_used=3,
            stop_reason="end_turn",
            input_tokens=2000,
            output_tokens=800,
        )

        result = run_strategy_reflection()
        assert result.rules_proposed == 1
        assert result.rules_retired == 1
        assert result.identity_updated is True
        assert result.memo_written is True

    @patch("v2.strategy.get_claude_client")
    @patch("v2.strategy.run_agentic_loop")
    @patch("v2.strategy.build_formation_context", return_value="")
    def test_passes_session_id_and_stage_name_to_loop(self, mock_formation, mock_loop, mock_client):
        """`run_strategy_reflection` must thread session_id + stage_name='reflection'
        to the agentic loop so tool_invocation events carry session context."""
        from v2.claude_client import AgenticLoopResult
        from v2.strategy import run_strategy_reflection

        mock_loop.return_value = AgenticLoopResult(
            messages=[],
            turns_used=1,
            stop_reason="end_turn",
            input_tokens=0,
            output_tokens=0,
        )

        run_strategy_reflection(session_id=42)

        call_kwargs = mock_loop.call_args.kwargs
        assert call_kwargs["session_id"] == 42
        assert call_kwargs["stage_name"] == "reflection"


class TestEvidenceShownEvent:
    """`tool_get_session_summary_with_telemetry` is a session-aware wrapper
    around the real session-summary tool. It emits an `evidence_shown` event
    keyed on the session so the auditor can check whether reflection actually
    *saw* round-trip evidence — currently invisible because round-trip data
    is computed inline and never persisted."""

    def test_emits_evidence_shown_with_round_trips(self, mock_db):
        from datetime import date

        from v2.patterns import RoundTrip
        from v2.strategy import tool_get_session_summary_with_telemetry

        with patch("v2.strategy.tool_get_session_summary", return_value="summary"), \
             patch("v2.strategy.analyze_round_trips") as mock_rt, \
             patch("v2.strategy.record_event") as mock_rec:
            mock_rt.return_value = [
                RoundTrip(
                    ticker="GOOGL",
                    pair_count=7,
                    first_date=date(2026, 4, 17),
                    last_date=date(2026, 5, 8),
                ),
            ]
            tool_get_session_summary_with_telemetry(days=30, session_id=42)

        evidence_calls = [
            c for c in mock_rec.call_args_list
            if c.kwargs.get("event_type") == "evidence_shown"
        ]
        assert len(evidence_calls) == 1
        ev = evidence_calls[0].kwargs
        assert ev["session_id"] == 42
        assert ev["stage_name"] == "reflection"
        assert ev["payload"]["evidence_kind"] == "round_trips"
        assert ev["payload"]["items"][0]["ticker"] == "GOOGL"
        assert ev["payload"]["items"][0]["pair_count"] == 7
        assert ev["payload"]["summary"]["n_tickers"] == 1

    def test_emits_evidence_shown_when_no_round_trips(self, mock_db):
        """Auditor needs the explicit empty event to distinguish 'reflection
        was shown nothing' from 'wrapper never ran'."""
        from v2.strategy import tool_get_session_summary_with_telemetry

        with patch("v2.strategy.tool_get_session_summary", return_value="summary"), \
             patch("v2.strategy.analyze_round_trips", return_value=[]), \
             patch("v2.strategy.record_event") as mock_rec:
            tool_get_session_summary_with_telemetry(days=30, session_id=42)

        evidence_calls = [
            c for c in mock_rec.call_args_list
            if c.kwargs.get("event_type") == "evidence_shown"
        ]
        assert len(evidence_calls) == 1
        ev = evidence_calls[0].kwargs
        assert ev["payload"]["items"] == []
        assert ev["payload"]["summary"]["n_tickers"] == 0

    def test_no_event_when_session_id_none(self, mock_db):
        from v2.strategy import tool_get_session_summary_with_telemetry

        with patch("v2.strategy.tool_get_session_summary", return_value="summary"), \
             patch("v2.strategy.analyze_round_trips", return_value=[]), \
             patch("v2.strategy.record_event") as mock_rec:
            tool_get_session_summary_with_telemetry(days=30, session_id=None)

        # record_event called with session_id=None is itself a no-op, but the
        # call must still happen so the wrapper's contract stays simple.
        # Auditor relies on session_id being non-null for any persisted row.
        evidence_calls = [
            c for c in mock_rec.call_args_list
            if c.kwargs.get("event_type") == "evidence_shown"
        ]
        assert len(evidence_calls) == 1
        assert evidence_calls[0].kwargs["session_id"] is None


class TestRuleTenureGuard:
    def test_cannot_retire_rule_before_min_tenure(self, mock_db):
        """Rules must be active for at least 5 days before retirement."""
        from datetime import datetime, timedelta
        mock_db.fetchone.return_value = {
            "id": 1, "rule_text": "Test rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=2),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }
        from v2.strategy import tool_retire_rule
        result = tool_retire_rule(rule_id=1, reason="Not working")
        assert "too new" in result.lower() or "minimum tenure" in result.lower()

    def test_can_retire_rule_after_min_tenure(self, mock_db):
        """Rules active for >= 5 days can be retired normally."""
        from datetime import datetime, timedelta
        mock_db.fetchone.return_value = {
            "id": 1, "rule_text": "Old rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=6),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }
        mock_db.rowcount = 1
        from v2.strategy import reset_session, tool_retire_rule
        reset_session()
        result = tool_retire_rule(rule_id=1, reason="Data no longer supports it")
        assert "retired" in result.lower()


class TestRetirementCap:
    def test_max_retirements_per_session(self, mock_db):
        """At most 2 rules can be retired per session to prevent mass purges."""
        from datetime import datetime, timedelta
        old_rule = {
            "id": 1, "rule_text": "Old rule", "status": "active",
            "created_at": datetime.now() - timedelta(days=30),
            "retirement_reason": None, "retired_at": None,
            "category": "test", "direction": "constraint",
            "confidence": Decimal("0.8"), "supporting_evidence": "test",
        }
        mock_db.fetchone.return_value = old_rule
        mock_db.rowcount = 1
        from v2.strategy import reset_session, tool_retire_rule
        reset_session()
        r1 = tool_retire_rule(rule_id=1, reason="Data changed")
        assert "retired" in r1.lower()
        r2 = tool_retire_rule(rule_id=2, reason="No longer valid")
        assert "retired" in r2.lower()
        r3 = tool_retire_rule(rule_id=3, reason="Also bad")
        assert "limit" in r3.lower() or "maximum" in r3.lower()


class TestReflectionFormationInjection:
    def test_formation_context_in_reflection_system_prompt(self):
        """Formation context should appear in reflection system prompt."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100

        with patch("v2.strategy.get_claude_client", return_value=MagicMock()), \
             patch("v2.strategy.reset_session"), \
             patch("v2.strategy.run_agentic_loop") as mock_loop, \
             patch("v2.strategy.build_formation_context",
                   return_value="## FORMATION MODE ACTIVE\nBe exploratory"):
            mock_loop.return_value = mock_result

            run_strategy_reflection(model="claude-haiku-4-5-20251001", max_turns=3)

        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") if call_kwargs[1] else None
        if system_prompt is None:
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "reflection" in arg.lower():
                    system_prompt = arg
                    break
        assert "FORMATION MODE ACTIVE" in system_prompt

    def test_no_formation_context_when_graduated(self):
        """Empty formation context should not alter the reflection prompt."""
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.turns_used = 1
        mock_result.stop_reason = "end_turn"
        mock_result.input_tokens = 500
        mock_result.output_tokens = 100

        with patch("v2.strategy.get_claude_client", return_value=MagicMock()), \
             patch("v2.strategy.reset_session"), \
             patch("v2.strategy.run_agentic_loop") as mock_loop, \
             patch("v2.strategy.build_formation_context", return_value=""):
            mock_loop.return_value = mock_result

            run_strategy_reflection(model="claude-haiku-4-5-20251001", max_turns=3)

        call_kwargs = mock_loop.call_args
        system_prompt = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system") if call_kwargs[1] else None
        if system_prompt is None:
            for arg in call_kwargs.args:
                if isinstance(arg, str) and "reflection" in arg.lower():
                    system_prompt = arg
                    break
        assert "FORMATION MODE" not in system_prompt


class TestIdentityUpdateGuard:
    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_warns_if_recently_updated(self, mock_get, mock_insert):
        """Should return warning if identity was updated within 3 days."""

        from v2.strategy import tool_update_strategy_identity
        mock_get.return_value = make_strategy_state_row(
            version=5,
            created_at=datetime.now(UTC),
        )

        result = tool_update_strategy_identity(
            identity_text="New identity",
            risk_posture="aggressive",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )

        assert "Warning" in result
        # clear is now folded into insert_strategy_state; guard blocks both
        mock_insert.assert_not_called()

    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_warning_does_not_promise_retry_will_succeed(self, mock_get, mock_insert):
        """P3.42: the prior warning text said 'To proceed anyway, call
        update_strategy_identity again.' — but a retry hits the same guard
        (or finds a now-0-days-old freshly-written state), so the LLM is
        misled into wasted turns. The guard is hard, not soft. The warning
        must not promise a retry path."""
        from v2.strategy import tool_update_strategy_identity
        mock_get.return_value = make_strategy_state_row(
            version=5,
            created_at=datetime.now(UTC),
        )

        result = tool_update_strategy_identity(
            identity_text="New identity",
            risk_posture="aggressive",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )

        assert "Warning" in result
        # The misleading instruction must be gone — the strategist should not
        # be told to retry a guard that fires unconditionally.
        assert "call update_strategy_identity again" not in result.lower()
        assert "call again" not in result.lower()

    @patch("v2.strategy.insert_strategy_state")
    @patch("v2.strategy.get_current_strategy_state")
    def test_allows_update_if_not_recent(self, mock_get, mock_insert):
        """Should allow update if identity hasn't been updated in >3 days."""
        from datetime import timedelta

        from v2.strategy import tool_update_strategy_identity
        mock_get.return_value = make_strategy_state_row(
            version=5,
            created_at=datetime.now(UTC) - timedelta(days=5),
        )
        mock_insert.return_value = 6

        result = tool_update_strategy_identity(
            identity_text="New identity",
            risk_posture="moderate",
            sector_biases={},
            preferred_signals=[],
            avoided_signals=[],
        )

        assert "version 6" in result
        # clear is now folded into insert_strategy_state; one call covers both
        mock_insert.assert_called_once()
