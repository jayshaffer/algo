# v2 Live-Trade Social Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single bare daily recap with two new social-post types (per-trade posts after the daily session, plus a separate pre-market take), gated behind a feature flag so the existing recap path keeps running until the new pipeline is validated. Implements Spec #2 of the audience-growth strategy (`docs/superpowers/specs/2026-05-03-live-trade-pipeline-design.md`).

**Architecture:**
- New self-contained modules: `v2/social_trades.py` (per-trade posts) and `v2/premarket.py` (pre-market posts). Both reuse the existing client factories (`get_twitter_client`, `get_bluesky_client`), low-level posters (`post_tweet`, `post_to_bluesky`), and the `insert_tweet` audit log. The old `run_twitter_stage` / `run_bluesky_stage` orchestrators are left intact for now — they continue to run when the new pipeline's feature flag is off, and will be deleted in a follow-up plan after one week of clean prod runs (per spec rollout plan).
- One narrow schema migration (`tweets.decision_id` nullable FK) extends the rerun-guard key from `(session_date, type, platform)` to `(decision_id, platform)` so multiple per-day trade posts can each be deduped independently.
- Weekend / NYSE-holiday checks already exist inside `v2/backfill.py` (`_is_trading_day`, `NYSE_HOLIDAYS`). This plan extracts them into a shared `v2/market_calendar.py` module so `premarket.py` can reuse the same canonical list without a wrong-direction dependency on a learning-loop module.
- Trade-post URLs are constructed deterministically (`{DASHBOARD_URL}/trade/{decision_id}/`, `{DASHBOARD_URL}/thesis/{thesis_id}/`) — the LLM never builds them, eliminating a class of broken-link risks.

**Tech Stack:** Python 3.12, pytest, psycopg2 (raw SQL via `get_cursor()` autocommit context manager), Anthropic SDK (Claude Haiku for both post types), tweepy (Twitter), atproto (Bluesky). Tests mock all external services via `tests/v2/conftest.py` fixtures (`mock_db`, `mock_cursor`, `mock_claude_client`).

**Spec defaults locked in (per spec "Open questions"):**
- Notional threshold for trade posts: `$100` minimum (skip micro-trades).
- Quiet-day fallback (mini-recap when no postable decisions): runs on trading days only; weekends/holidays produce no post at all so the account doesn't spam non-content.
- `ALGO_TRADE_POST_DRY_RUN=1` env var supported; logs the generated post body and skips both platform posts AND the DB log row (true dry-run, leaves no trace).
- Pre-market cron: independent module entrypoint (`python -m v2.premarket`), invoked from a new dedicated `premarket:stage` Taskfile target. Same pattern as `entertainment.py`. Not added to `v2/session.py`.

**Conventions:**
- One commit per task. Commit messages use `feat(v2): ...` for new features, `refactor(v2): ...` for internal moves, `chore(v2): ...` for migration / docs. Match the prefix style of recent history (`git log --oneline -10`).
- Run the targeted test file after each change: `python3 -m pytest tests/v2/test_<module>.py -v`.
- Run the full v2 suite at the end of each task to catch regressions: `python3 -m pytest tests/v2/ -q`.
- Tests requiring `DATABASE_URL` / `ALPACA_*` env vars: the test suite is wired to mock these at module load, but if you run on bare host Python 3.12 without the docker container's env, export `ALPACA_API_KEY=fake ALPACA_SECRET_KEY=fake ALPACA_BASE_URL=https://paper-api.alpaca.markets ALPACA_PAPER=true ANTHROPIC_API_KEY=fake DATABASE_URL=postgresql://fake:fake@nohost:5432/test` first. Or run inside the existing docker image: `docker run --rm --env-file /home/jay/dev/algo/.env -v <worktree>/v2:/app/v2 -v <worktree>/tests:/app/tests -v <worktree>/pytest.ini:/app/pytest.ini -v <worktree>/trading:/app/trading algo-trading python -m pytest tests/v2/ -q`.

---

## Task 1: Extract `is_trading_day` + `NYSE_HOLIDAYS` into `v2/market_calendar.py`

**Why:** `premarket.py` (Task 9-10) needs the weekend/holiday gate. The canonical list and predicate already live in `v2/backfill.py:19-44`. Importing them from `backfill` would tie premarket to a learning-loop module — wrong direction. Extract to a small shared module so both consumers (`backfill.py`, the new `premarket.py`) can import from a neutral location.

**Files:**
- Create: `v2/market_calendar.py`
- Modify: `v2/backfill.py:1-50` (replace local definitions with re-exports for backward compat, or delete and update imports)
- Create: `tests/v2/test_market_calendar.py`

- [ ] **Step 1: Write the failing test**

Create `tests/v2/test_market_calendar.py`:

```python
"""Tests for the shared NYSE trading-day calendar."""

from datetime import date

from v2.market_calendar import NYSE_HOLIDAYS, is_trading_day


class TestIsTradingDay:
    def test_weekday_non_holiday_is_trading_day(self):
        # 2026-05-04 is a Monday, not a holiday
        assert is_trading_day(date(2026, 5, 4)) is True

    def test_saturday_is_not_trading_day(self):
        assert is_trading_day(date(2026, 5, 9)) is False

    def test_sunday_is_not_trading_day(self):
        assert is_trading_day(date(2026, 5, 10)) is False

    def test_known_holiday_is_not_trading_day(self):
        # 2026-07-03 is observed July 4th holiday (the actual 4th is Saturday)
        assert is_trading_day(date(2026, 7, 3)) is False

    def test_christmas_2026_is_not_trading_day(self):
        assert is_trading_day(date(2026, 12, 25)) is False


class TestNyseHolidaysCoverage:
    def test_includes_2026_holidays(self):
        # Spot-check key 2026 holidays
        assert date(2026, 1, 1) in NYSE_HOLIDAYS  # New Year's
        assert date(2026, 7, 3) in NYSE_HOLIDAYS  # July 4th observed
        assert date(2026, 12, 25) in NYSE_HOLIDAYS  # Christmas

    def test_includes_2027_holidays(self):
        assert date(2027, 1, 1) in NYSE_HOLIDAYS
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/v2/test_market_calendar.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'v2.market_calendar'`.

- [ ] **Step 3: Create `v2/market_calendar.py` by extracting from backfill.py**

Create `v2/market_calendar.py` with the existing definitions copied verbatim from `v2/backfill.py:15-44`:

```python
"""NYSE trading-day calendar — shared weekend/holiday predicate.

Maintained manually because adding pandas_market_calendars is overkill
for ~10 dates a year. Sources: https://www.nyse.com/markets/hours-calendars
(verify yearly). Half-days (early closes) are NOT included — they still
produce a daily bar.
"""

from datetime import date


NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20),
    date(2025, 2, 17), date(2025, 4, 18), date(2025, 5, 26),
    date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
})


def is_trading_day(d: date) -> bool:
    """True if `d` is a weekday and not in the NYSE holiday list."""
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS
```

- [ ] **Step 4: Update `v2/backfill.py` to re-export from market_calendar**

In `v2/backfill.py`, replace lines 15-44 (the comment block + `NYSE_HOLIDAYS = frozenset({...})` + `def _is_trading_day(...)`) with:

```python
from .market_calendar import NYSE_HOLIDAYS, is_trading_day as _is_trading_day  # noqa: F401  re-export
```

The leading underscore on `_is_trading_day` is preserved as the module-private alias backfill's other functions use. `NYSE_HOLIDAYS` is also re-exported for any caller (or test) that imports it from `backfill`.

- [ ] **Step 5: Run market_calendar tests + the existing backfill tests**

```bash
python3 -m pytest tests/v2/test_market_calendar.py tests/v2/test_backfill.py -v
```

Expected: all market_calendar tests PASS; existing backfill tests still PASS (the re-export preserves behavior).

- [ ] **Step 6: Run the full v2 suite**

```bash
python3 -m pytest tests/v2/ -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add v2/market_calendar.py v2/backfill.py tests/v2/test_market_calendar.py
git commit -m "$(cat <<'EOF'
refactor(v2): extract NYSE_HOLIDAYS / is_trading_day into market_calendar

Lift the canonical NYSE trading-day predicate out of backfill.py into a
neutral v2/market_calendar.py so the new premarket post stage can import
it without depending on a learning-loop module. backfill.py keeps the
exact same _is_trading_day name via re-export to preserve all existing
imports and behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Schema migration — add `tweets.decision_id` column

**Why:** Today's rerun guard `posted_tweet_exists(session_date, type, platform)` is keyed `(session_date, type, platform)` — fine for one-recap-per-day, broken for many-trade-posts-per-day. Extend the schema with a nullable FK so trade-post deduplication can key on `(decision_id, platform)`. Nullable keeps the migration backwards-compatible: existing recap rows have `decision_id=NULL`.

**Files:**
- Create: `db/init/023_tweets_decision_id.sql`

- [ ] **Step 1: Verify the next sequence number**

```bash
ls db/init/ | sort | tail -3
```

Expected: highest existing is `022_news_macro_published_at_tz.sql`. New file is `023_*`.

- [ ] **Step 2: Create the migration file**

Create `db/init/023_tweets_decision_id.sql`:

```sql
-- 023_tweets_decision_id.sql: link tweets back to source decisions.
--
-- Today's rerun guard posted_tweet_exists(session_date, type, platform)
-- assumes one recap tweet per day. The new live-trade pipeline posts
-- one tweet per significant decision, so we need to dedup per-decision
-- per-platform. Add a nullable FK so existing recap rows aren't broken
-- and the new trade rows carry their source decision id.
--
-- Index supports the new query
--   SELECT 1 FROM tweets WHERE decision_id = $1 AND platform = $2
-- which fires once per candidate decision in run_trade_posts_stage.

ALTER TABLE tweets
    ADD COLUMN IF NOT EXISTS decision_id INTEGER
    REFERENCES decisions(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_tweets_decision_id_platform
    ON tweets(decision_id, platform) WHERE decision_id IS NOT NULL;
```

- [ ] **Step 3: Apply the migration to the dev database**

The dev pipeline runs against `algo-db-1` (port 5432). Apply:

```bash
docker exec -i algo-db-1 psql -U algo -d trading < db/init/023_tweets_decision_id.sql
```

Expected output: `ALTER TABLE` then `CREATE INDEX`. If either is silently `NOTICE: column "decision_id" of relation "tweets" already exists, skipping`, you've already run it — fine.

Apply the same to the paper DB (port 5433):

```bash
docker exec -i algo-db-paper-1 psql -U algo -d trading < db/init/023_tweets_decision_id.sql
```

- [ ] **Step 4: Verify the column landed**

```bash
docker exec -i algo-db-1 psql -U algo -d trading -c "\d tweets" | grep decision_id
```

Expected: a line like `decision_id      | integer                  | | |`.

- [ ] **Step 5: Commit**

```bash
git add db/init/023_tweets_decision_id.sql
git commit -m "$(cat <<'EOF'
chore(v2): migration for tweets.decision_id

Add nullable FK + (decision_id, platform) partial index so the new live-
trade pipeline can dedup posts per decision per platform. Existing recap
rows leave decision_id NULL — backwards-compatible.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extend `insert_tweet` and add `posted_tweet_for_decision_exists`

**Files:**
- Modify: `v2/database/trading_db.py:744` (extend `insert_tweet` signature) and add a new function nearby
- Modify: `tests/v2/test_db.py` or `tests/v2/test_twitter.py::TestInsertTweet` (extend insert tests + add new lookup test)

- [ ] **Step 1: Locate the existing `insert_tweet` function**

```bash
grep -n "^def insert_tweet\|^def posted_tweet_exists" v2/database/trading_db.py
```

Expected: `insert_tweet` near line 744, `posted_tweet_exists` near line 755.

Read the existing implementations to confirm the SQL shape before editing:

```bash
sed -n '740,775p' v2/database/trading_db.py
```

- [ ] **Step 2: Write the failing tests**

Find the existing `TestInsertTweet` class in `tests/v2/test_twitter.py` (line 21) — that's where insert_tweet is exercised today. Add to that class:

```python
def test_insert_tweet_persists_decision_id(self, mock_db, mock_cursor):
    """When a decision_id is supplied, it must land on the row."""
    from datetime import date
    from v2.database.trading_db import insert_tweet

    mock_cursor.fetchone.return_value = {"id": 99}

    inserted_id = insert_tweet(
        session_date=date(2026, 5, 4),
        tweet_type="trade",
        tweet_text="Bought 12 $NVDA",
        tweet_id="tw_abc",
        posted=True,
        platform="twitter",
        decision_id=42,
    )

    # The cursor's execute was called once with an INSERT containing decision_id.
    sql, params = mock_cursor.execute.call_args[0]
    assert "decision_id" in sql.lower(), (
        f"INSERT must reference decision_id column; got SQL: {sql}"
    )
    assert 42 in params, (
        f"decision_id=42 must be passed in execute params; got: {params}"
    )
    assert inserted_id == 99
```

Add a new test class right after `TestInsertTweet` for the new lookup function:

```python
class TestPostedTweetForDecisionExists:
    """The trade-post rerun guard keys on (decision_id, platform), not date."""

    def test_returns_true_when_row_exists(self, mock_db, mock_cursor):
        from v2.database.trading_db import posted_tweet_for_decision_exists

        mock_cursor.fetchone.return_value = {"id": 7}

        assert posted_tweet_for_decision_exists(decision_id=42, platform="twitter") is True
        sql, params = mock_cursor.execute.call_args[0]
        assert "decision_id" in sql.lower()
        assert "platform" in sql.lower()
        assert params == (42, "twitter")

    def test_returns_false_when_no_row(self, mock_db, mock_cursor):
        from v2.database.trading_db import posted_tweet_for_decision_exists

        mock_cursor.fetchone.return_value = None

        assert posted_tweet_for_decision_exists(decision_id=42, platform="bluesky") is False

    def test_returns_false_when_only_unposted_row(self, mock_db, mock_cursor):
        """A row where posted=FALSE shouldn't block re-posting."""
        from v2.database.trading_db import posted_tweet_for_decision_exists

        # SQL filters posted=TRUE, so simulate no match
        mock_cursor.fetchone.return_value = None

        assert posted_tweet_for_decision_exists(decision_id=42, platform="twitter") is False
        sql, _params = mock_cursor.execute.call_args[0]
        # Confirm the SQL filters posted=TRUE
        assert "posted" in sql.lower(), (
            f"Lookup must filter on posted=TRUE; got SQL: {sql}"
        )
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python3 -m pytest tests/v2/test_twitter.py::TestInsertTweet::test_insert_tweet_persists_decision_id tests/v2/test_twitter.py::TestPostedTweetForDecisionExists -v
```

Expected: FAIL — current `insert_tweet` has no `decision_id` param; `posted_tweet_for_decision_exists` doesn't exist.

- [ ] **Step 4: Extend `insert_tweet` and add `posted_tweet_for_decision_exists`**

In `v2/database/trading_db.py`, find the existing `insert_tweet` function (line ~744). Replace it (and add the new function right after) with:

```python
def insert_tweet(
    session_date,
    tweet_type: str,
    tweet_text: str,
    tweet_id: str | None = None,
    posted: bool = False,
    error: str | None = None,
    platform: str = "twitter",
    decision_id: int | None = None,
) -> int:
    """Log a tweet/post to the audit table.

    decision_id (new) ties a per-trade post back to its source decision,
    enabling the (decision_id, platform) rerun guard used by the live-
    trade pipeline. Recap and entertainment posts leave it NULL.
    """
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO tweets (
                session_date, tweet_type, tweet_text, tweet_id,
                posted, error, platform, decision_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (session_date, tweet_type, tweet_text, tweet_id,
              posted, error, platform, decision_id))
        return cur.fetchone()["id"]


def posted_tweet_for_decision_exists(decision_id: int, platform: str) -> bool:
    """True if a successful (posted=TRUE) tweet already exists for this
    decision on this platform. Used by run_trade_posts_stage to skip
    decisions that were posted on a prior session run."""
    with get_cursor() as cur:
        cur.execute("""
            SELECT 1
            FROM tweets
            WHERE decision_id = %s
              AND platform = %s
              AND posted = TRUE
            LIMIT 1
        """, (decision_id, platform))
        return cur.fetchone() is not None
```

- [ ] **Step 5: Run the new tests + existing tweet-related tests**

```bash
python3 -m pytest tests/v2/test_twitter.py::TestInsertTweet tests/v2/test_twitter.py::TestPostedTweetExists tests/v2/test_twitter.py::TestPostedTweetForDecisionExists -v
```

Expected: all pass. The new `decision_id` param defaults to None so existing callers are unaffected.

- [ ] **Step 6: Run the full v2 suite**

```bash
python3 -m pytest tests/v2/ -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_twitter.py
git commit -m "$(cat <<'EOF'
feat(v2): extend insert_tweet with decision_id + add per-decision lookup

Add nullable decision_id param to insert_tweet so trade posts can record
which decision they came from. Add posted_tweet_for_decision_exists
keyed on (decision_id, platform, posted=TRUE) — the new rerun guard for
the live-trade pipeline (existing recap guard on session_date stays
unchanged for the old path).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `select_postable_decisions_for_date` to `trading_db.py`

**Why:** The trade-post stage iterates over today's decisions joined with their underlying thesis. Decisions don't carry `thesis_id` directly — the link goes through `decisions.playbook_action_id → playbook_actions.thesis_id → theses.id`. Off-playbook trades (`is_off_playbook=TRUE`) have no playbook_action_id and therefore no thesis link; the spec allows them but they post without a thesis URL.

Filtering: action != 'hold', notional `ABS(quantity * price) >= min_notional`, ordered by abs notional desc, limited.

**Files:**
- Modify: `v2/database/trading_db.py` (add function near `get_recent_decisions`)
- Modify: `tests/v2/test_db.py` (add test class)

- [ ] **Step 1: Write the failing test**

Add to `tests/v2/test_db.py` (the file's existing test classes follow `TestXxx` per-function naming). Append:

```python
class TestSelectPostableDecisionsForDate:
    """Live-trade pipeline selector: today's significant non-hold decisions."""

    def test_filters_holds_and_micro_trades_orders_by_notional(self, mock_db, mock_cursor):
        """Sanity-check the SQL: WHERE filters action and notional, ORDER BY
        is abs(quantity*price) DESC, LIMIT applied."""
        from datetime import date
        from v2.database.trading_db import select_postable_decisions_for_date

        # We don't actually execute the SQL — we inspect what was sent.
        mock_cursor.fetchall.return_value = [
            {"id": 1, "ticker": "NVDA", "action": "buy", "quantity": 10,
             "price": 500.0, "reasoning": "AI tailwind", "thesis_id": 7,
             "thesis_text": "AI demand", "thesis_direction": "long",
             "is_off_playbook": False},
        ]

        result = select_postable_decisions_for_date(
            session_date=date(2026, 5, 4),
            min_notional=100.0,
            limit=5,
        )

        sql, params = mock_cursor.execute.call_args[0]
        sql_lower = sql.lower()
        # Filters
        assert "where" in sql_lower
        assert "d.action" in sql_lower or "action" in sql_lower
        assert "hold" in sql_lower, "must filter out holds"
        assert "abs(d.quantity * d.price)" in sql_lower or "abs(quantity * price)" in sql_lower, (
            "must filter on absolute notional"
        )
        # Joins to surface thesis info
        assert "playbook_actions" in sql_lower
        assert "theses" in sql_lower
        # Order + limit
        assert "order by" in sql_lower
        assert "desc" in sql_lower
        assert "limit" in sql_lower
        # Params: session_date, min_notional, limit
        assert date(2026, 5, 4) in params
        assert 100.0 in params
        assert 5 in params

        # Returns the rows as-is from fetchall
        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"
        assert result[0]["thesis_id"] == 7

    def test_returns_empty_when_no_decisions(self, mock_db, mock_cursor):
        from datetime import date
        from v2.database.trading_db import select_postable_decisions_for_date

        mock_cursor.fetchall.return_value = []

        result = select_postable_decisions_for_date(
            session_date=date(2026, 5, 4), min_notional=100.0, limit=5,
        )
        assert result == []

    def test_off_playbook_decision_returns_with_null_thesis(self, mock_db, mock_cursor):
        """Off-playbook decisions are postable but carry no thesis link."""
        from datetime import date
        from v2.database.trading_db import select_postable_decisions_for_date

        mock_cursor.fetchall.return_value = [
            {"id": 2, "ticker": "TSLA", "action": "sell", "quantity": 5,
             "price": 250.0, "reasoning": "stop hit", "thesis_id": None,
             "thesis_text": None, "thesis_direction": None,
             "is_off_playbook": True},
        ]

        result = select_postable_decisions_for_date(
            session_date=date(2026, 5, 4), min_notional=100.0, limit=5,
        )
        assert result[0]["thesis_id"] is None
        assert result[0]["is_off_playbook"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/v2/test_db.py::TestSelectPostableDecisionsForDate -v
```

Expected: FAIL — `ImportError: cannot import name 'select_postable_decisions_for_date'`.

- [ ] **Step 3: Add the function**

In `v2/database/trading_db.py`, append after `get_recent_decisions` (around line 200):

```python
def select_postable_decisions_for_date(
    session_date,
    min_notional: float,
    limit: int,
) -> list[dict]:
    """Return today's non-hold decisions worth posting about.

    Joined with their underlying thesis via playbook_actions; off-playbook
    decisions return rows with NULL thesis fields. Ordered by absolute
    notional value descending so the top `limit` are the highest-impact
    trades for the day. Filtered to ABS(quantity*price) >= min_notional
    so micro-trades don't spam the social feed.
    """
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                d.id,
                d.date,
                d.ticker,
                d.action,
                d.quantity,
                d.price,
                d.reasoning,
                d.is_off_playbook,
                pa.thesis_id AS thesis_id,
                t.thesis     AS thesis_text,
                t.direction  AS thesis_direction
            FROM decisions d
            LEFT JOIN playbook_actions pa ON pa.id = d.playbook_action_id
            LEFT JOIN theses t            ON t.id  = pa.thesis_id
            WHERE d.date = %s
              AND d.action != 'hold'
              AND d.price IS NOT NULL
              AND d.quantity IS NOT NULL
              AND ABS(d.quantity * d.price) >= %s
            ORDER BY ABS(d.quantity * d.price) DESC
            LIMIT %s
        """, (session_date, min_notional, limit))
        return cur.fetchall()
```

- [ ] **Step 4: Run the test**

```bash
python3 -m pytest tests/v2/test_db.py::TestSelectPostableDecisionsForDate -v
```

Expected: PASS.

- [ ] **Step 5: Run full v2 suite**

```bash
python3 -m pytest tests/v2/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add v2/database/trading_db.py tests/v2/test_db.py
git commit -m "$(cat <<'EOF'
feat(v2): add select_postable_decisions_for_date

Live-trade pipeline selector. Joins today's decisions with their
underlying thesis (via playbook_actions; off-playbook trades return
NULL thesis fields), filters out holds and micro-trades below the
configured notional floor, orders by absolute notional desc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `v2/social_trades.py` — system prompt + `generate_trade_post`

**Why:** Pure-function piece of the new pipeline: take one decision dict + optional thesis info + a dashboard URL, produce post text. No I/O beyond the Anthropic call. Keeps the orchestrator (Task 6) thin.

**Files:**
- Create: `v2/social_trades.py`
- Create: `tests/v2/test_social_trades.py`

- [ ] **Step 1: Write the failing test**

Create `tests/v2/test_social_trades.py`:

```python
"""Tests for v2/social_trades.py — per-trade social post pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_claude_response(json_data: dict):
    """Helper: shape a MagicMock the way `_call_with_retry` returns one."""
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(json_data))]
    return response


class TestGenerateTradePost:
    """Pure path: decision + optional thesis → post body. URL appended
    deterministically after generation, not by the LLM.

    Patches `v2.social_trades.get_claude_client` and `v2.social_trades._call_with_retry`
    directly (the names imported into social_trades), not the originals in
    v2.claude_client — `from x import y` binds y by reference at module load,
    so patching `v2.claude_client.get_claude_client` doesn't reach social_trades.
    """

    @patch("v2.social_trades._call_with_retry")
    @patch("v2.social_trades.get_claude_client")
    def test_generates_text_and_appends_trade_url(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Bought 12 $NVDA — AI demand still pulling."}
        )

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "AI tailwind",
                    "thesis_id": None, "thesis_text": None,
                    "thesis_direction": None, "is_off_playbook": False}

        result = generate_trade_post(
            decision=decision,
            dashboard_base_url="https://dash.example.com",
        )

        assert result is not None
        assert "Bought 12 $NVDA" in result["text"]
        assert result["text"].endswith("https://dash.example.com/trade/99/")
        assert result["decision_id"] == 99
        assert result["type"] == "trade"

    @patch("v2.social_trades._call_with_retry")
    @patch("v2.social_trades.get_claude_client")
    def test_appends_thesis_url_when_thesis_present(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Bought 12 $NVDA — backing the AI thesis."}
        )

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "AI tailwind",
                    "thesis_id": 42, "thesis_text": "AI demand pulling",
                    "thesis_direction": "long", "is_off_playbook": False}

        result = generate_trade_post(
            decision=decision,
            dashboard_base_url="https://dash.example.com",
        )

        assert "/trade/99/" in result["text"]
        assert "/thesis/42/" in result["text"]

    @patch("v2.social_trades._call_with_retry")
    @patch("v2.social_trades.get_claude_client")
    def test_no_dashboard_base_url_skips_url_append(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response({"text": "Bought 12 $NVDA."})

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "x",
                    "thesis_id": None, "thesis_text": None,
                    "thesis_direction": None, "is_off_playbook": False}

        result = generate_trade_post(decision=decision, dashboard_base_url="")
        assert "http" not in result["text"]

    @patch("v2.social_trades._call_with_retry", side_effect=Exception("API outage"))
    @patch("v2.social_trades.get_claude_client")
    def test_llm_failure_returns_none(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "x",
                    "thesis_id": None, "thesis_text": None,
                    "thesis_direction": None, "is_off_playbook": False}

        result = generate_trade_post(decision=decision, dashboard_base_url="")
        assert result is None

    @patch("v2.social_trades._call_with_retry")
    @patch("v2.social_trades.get_claude_client")
    def test_llm_returns_malformed_json_returns_none(self, mock_get_client, mock_retry):
        from v2.social_trades import generate_trade_post

        mock_get_client.return_value = MagicMock()
        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="not json at all")]
        mock_retry.return_value = bad_response

        decision = {"id": 99, "ticker": "NVDA", "action": "buy",
                    "quantity": 12, "price": 500.0, "reasoning": "x",
                    "thesis_id": None, "thesis_text": None,
                    "thesis_direction": None, "is_off_playbook": False}

        result = generate_trade_post(decision=decision, dashboard_base_url="")
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/v2/test_social_trades.py::TestGenerateTradePost -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'v2.social_trades'`.

- [ ] **Step 3: Create `v2/social_trades.py`**

Create `v2/social_trades.py`:

```python
"""Live-trade social pipeline -- Bikini Bottom Capital (v2).

Per-fill posts after the daily session: one tweet per significant new
decision, each linking to its /trade/<id>/ page on the public dashboard.
Replaces the bare daily recap when ALGO_ENABLE_TRADE_POSTS=1.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date

from .claude_client import _call_with_retry, get_claude_client
from .database.trading_db import (
    insert_tweet,
    posted_tweet_exists,
    posted_tweet_for_decision_exists,
    select_postable_decisions_for_date,
)

logger = logging.getLogger("social_trades")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

TRADE_POST_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital.
The bot just made a trade. You're posting about it on social media.

Your voice:
- Casual, direct. Like sharing a play with a friend who trades.
- Don't oversell. Reference the actual reasoning — not generic excitement.
- Occasional dry humor. Never try-hard.

Generate ONE post about this single trade.

Respond with JSON: {"text": "post text here"}

Rules:
- 180 chars max (URL gets appended after — leave room).
- Lead with the action: "Bought 12 $NVDA at $X" / "Trimmed $TSLA back to half size".
- One concrete reason. The thesis text is provided — pull from it, don't invent.
- $CASHTAG only for the ticker actually traded.
- No "not financial advice", no hashtag spam, no emoji walls.
- If there's a thesis, your post should make a reader want to click through to read it."""


def _build_trade_context(decision: dict) -> str:
    """Plain-text summary of one decision + its thesis, fed to the LLM."""
    parts = [
        f"Trade: {decision['action'].upper()} {decision['quantity']} "
        f"{decision['ticker']} @ ${decision['price']}",
        f"Reasoning: {decision.get('reasoning', '')}",
    ]
    if decision.get("thesis_text"):
        parts.append(
            f"Thesis ({decision.get('thesis_direction', 'long')}): "
            f"{decision['thesis_text']}"
        )
    if decision.get("is_off_playbook"):
        parts.append("Note: this is an off-playbook trade.")
    return "\n".join(parts)


def _build_url_suffix(decision: dict, dashboard_base_url: str) -> str:
    """Deterministic trade + (optional) thesis URL append. Empty if no
    DASHBOARD_URL configured — bare text post."""
    if not dashboard_base_url:
        return ""
    base = dashboard_base_url.rstrip("/")
    parts = [f"{base}/trade/{decision['id']}/"]
    if decision.get("thesis_id"):
        parts.append(f"{base}/thesis/{decision['thesis_id']}/")
    return "\n" + "\n".join(parts)


def generate_trade_post(
    decision: dict,
    dashboard_base_url: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Generate one social-post body for a single decision.

    Returns dict {text, type='trade', decision_id} or None if generation
    fails (LLM error / malformed JSON / no text).
    """
    context = _build_trade_context(decision)
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=TRADE_POST_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        raw = response.content[0].text.strip()
        logger.info("AI response for decision %s:\n%s", decision["id"], raw)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except Exception as e:
        logger.error("Failed to generate trade post for decision %s: %s",
                     decision.get("id"), e)
        return None

    body = result.get("text")
    if not body or not isinstance(body, str):
        logger.warning("LLM returned no text or malformed response: %s", result)
        return None

    text = body + _build_url_suffix(decision, dashboard_base_url)
    return {"text": text, "type": "trade", "decision_id": decision["id"]}
```

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest tests/v2/test_social_trades.py::TestGenerateTradePost -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add v2/social_trades.py tests/v2/test_social_trades.py
git commit -m "$(cat <<'EOF'
feat(v2): social_trades.generate_trade_post + system prompt

Pure helper: take a decision dict (joined with thesis info) + the
dashboard base URL, produce post text with deterministic trade/thesis
URL suffix. LLM never builds URLs — eliminates broken-link risk.
Off-playbook decisions are flagged in the LLM context so the model can
optionally acknowledge they didn't come from the daily strategist run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `v2/social_trades.py` — `run_trade_posts_stage` orchestrator

**Why:** The wrapper around `generate_trade_post` that does the actual posting + DB logging + per-decision rerun guard + per-decision error isolation, for both Twitter and Bluesky. Spec section "Architecture > New module" + "Error handling".

**Files:**
- Modify: `v2/social_trades.py` (append the orchestrator)
- Modify: `tests/v2/test_social_trades.py` (add `TestRunTradePostsStage` class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_social_trades.py`:

```python
class TestRunTradePostsStage:
    """End-to-end stage orchestrator. All external calls mocked."""

    def _decision(self, decision_id: int, ticker: str = "NVDA"):
        return {
            "id": decision_id, "ticker": ticker, "action": "buy",
            "quantity": 10, "price": 500.0, "reasoning": "AI tailwind",
            "thesis_id": None, "thesis_text": None,
            "thesis_direction": None, "is_off_playbook": False,
        }

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client")
    @patch("v2.social_trades.get_twitter_client")
    def test_posts_one_per_decision_to_each_platform(
        self, mock_tw_client, mock_bs_client, mock_select,
        mock_gen, mock_post_tw, mock_post_bs,
        mock_dedup, mock_insert,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        mock_select.return_value = [self._decision(1), self._decision(2, "TSLA")]
        mock_gen.side_effect = [
            {"text": "Bought 10 $NVDA", "type": "trade", "decision_id": 1},
            {"text": "Bought 10 $TSLA", "type": "trade", "decision_id": 2},
        ]
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "trade", "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "trade", "error": None}

        result = run_trade_posts_stage(date(2026, 5, 4))

        # Two decisions × two platforms = 4 inserts
        assert mock_insert.call_count == 4
        assert result.posts_attempted == 2
        assert result.posts_succeeded_twitter == 2
        assert result.posts_succeeded_bluesky == 2
        assert result.skipped is False

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists")
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    @patch("v2.social_trades.get_twitter_client")
    def test_per_decision_dedup_skips_already_posted(
        self, mock_tw_client, mock_bs_client, mock_select,
        mock_gen, mock_post_tw, mock_post_bs,
        mock_dedup, mock_insert,
    ):
        """If decision 2 was already posted to Twitter, skip the post for it
        but still attempt decision 1."""
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        mock_tw_client.return_value = object()
        mock_select.return_value = [self._decision(1), self._decision(2)]
        # decision_id=2 already posted on Twitter
        mock_dedup.side_effect = lambda decision_id, platform: (
            decision_id == 2 and platform == "twitter"
        )
        mock_gen.return_value = {"text": "x", "type": "trade", "decision_id": 1}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "trade", "error": None}

        result = run_trade_posts_stage(date(2026, 5, 4))

        # Only decision 1 was generated + posted on Twitter (decision 2 skipped)
        assert mock_post_tw.call_count == 1
        assert result.posts_skipped_dedup == 1

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_for_decision_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_trade_post")
    @patch("v2.social_trades.select_postable_decisions_for_date")
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    @patch("v2.social_trades.get_twitter_client")
    def test_one_decision_failure_does_not_drop_others(
        self, mock_tw_client, mock_bs_client, mock_select,
        mock_gen, mock_post_tw, mock_post_bs,
        mock_dedup, mock_insert,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        mock_tw_client.return_value = object()
        mock_select.return_value = [self._decision(1), self._decision(2)]
        # First decision generation fails, second succeeds
        mock_gen.side_effect = [None, {"text": "x", "type": "trade", "decision_id": 2}]
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "trade", "error": None}

        result = run_trade_posts_stage(date(2026, 5, 4))

        # The good one still posted
        assert mock_post_tw.call_count == 1
        assert result.posts_succeeded_twitter == 1
        # The bad one was recorded as failed (with an error) but didn't break the loop
        assert result.posts_failed >= 1

    @patch("v2.social_trades.select_postable_decisions_for_date", return_value=[])
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    @patch("v2.social_trades.get_twitter_client")
    def test_no_decisions_falls_through_to_quiet_day_handler(
        self, mock_tw_client, mock_bs_client, mock_select,
    ):
        """When no postable decisions, the stage delegates to the quiet-day
        path (tested separately in TestQuietDayFallback). Here we only
        confirm it doesn't crash and reports the empty case."""
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        mock_tw_client.return_value = object()
        with patch("v2.social_trades._post_quiet_day_recap", return_value=None) as mock_quiet:
            result = run_trade_posts_stage(date(2026, 5, 4))

        assert result.posts_attempted == 0
        mock_quiet.assert_called_once()

    @patch("v2.social_trades.get_twitter_client", return_value=None)
    @patch("v2.social_trades.get_bluesky_client", return_value=None)
    def test_skipped_when_no_credentials_on_either_platform(
        self, mock_bs, mock_tw,
    ):
        from datetime import date
        from v2.social_trades import run_trade_posts_stage

        result = run_trade_posts_stage(date(2026, 5, 4))
        assert result.skipped is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/v2/test_social_trades.py::TestRunTradePostsStage -v
```

Expected: FAIL — `run_trade_posts_stage` doesn't exist.

- [ ] **Step 3: Implement the orchestrator**

Append to `v2/social_trades.py`:

```python
# ---------------------------------------------------------------------------
# Imports for the orchestrator (kept here so the pure helpers above stay
# importable in tests without dragging in the whole post stack).
# ---------------------------------------------------------------------------

from .twitter import get_twitter_client, post_tweet           # noqa: E402
from .bluesky import get_bluesky_client, post_to_bluesky      # noqa: E402


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------

@dataclass
class TradePostsStageResult:
    """Aggregate outcome of run_trade_posts_stage.

    posts_attempted: how many decisions we tried to post about
    posts_skipped_dedup: per-(decision, platform) skips where rerun guard fired
    posts_succeeded_*: count of successful posts per platform
    posts_failed: per-decision generate-or-post failures (excludes dedup skips)
    quiet_day_recap_posted: True if the no-decisions fallback fired
    skipped: stage was a full no-op (no creds for either platform OR dry-run)
    errors: aggregated per-decision/per-platform error strings
    """
    posts_attempted: int = 0
    posts_skipped_dedup: int = 0
    posts_succeeded_twitter: int = 0
    posts_succeeded_bluesky: int = 0
    posts_failed: int = 0
    quiet_day_recap_posted: bool = False
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MIN_NOTIONAL = 100.0
DEFAULT_MAX_POSTS_PER_SESSION = 5


def _is_dry_run() -> bool:
    return os.environ.get("ALGO_TRADE_POST_DRY_RUN") == "1"


def _platform_post_one(
    platform: str,
    post_body: dict,
    decision: dict,
    client,
    poster,
    session_date: date,
    result: TradePostsStageResult,
) -> None:
    """Post one body to one platform; record the outcome.

    Mutates `result` and emits one row to `tweets` regardless of post outcome
    (so a failed post still leaves an audit trail). Dry-run mode short-circuits
    before both the platform call AND the DB write.
    """
    decision_id = decision["id"]

    # Per-(decision, platform) rerun guard
    try:
        if posted_tweet_for_decision_exists(decision_id, platform):
            result.posts_skipped_dedup += 1
            logger.info("Skip %s post for decision %s — already posted on %s",
                        platform, decision_id, platform)
            return
    except Exception as e:
        # Don't block on transient DB error — proceed to post and let the
        # insert below fail visibly if persistence is broken.
        logger.warning("Dedup check failed for decision %s on %s: %s; proceeding",
                       decision_id, platform, e)

    if _is_dry_run():
        logger.info("[DRY-RUN] %s post for decision %s:\n%s",
                    platform, decision_id, post_body["text"])
        return

    try:
        post_result = poster(post_body, client=client)
    except Exception as e:
        result.posts_failed += 1
        result.errors.append(f"{platform} post failed for decision {decision_id}: {e}")
        logger.error("%s post failed for decision %s: %s", platform, decision_id, e)
        return

    db_logged = True
    try:
        insert_tweet(
            session_date=session_date,
            tweet_type="trade",
            tweet_text=post_result["text"],
            tweet_id=post_result.get("tweet_id") or post_result.get("post_id"),
            posted=post_result["posted"],
            error=post_result.get("error"),
            platform=platform,
            decision_id=decision_id,
        )
    except Exception as e:
        db_logged = False
        result.errors.append(f"DB log failed for decision {decision_id} on {platform}: {e}")
        logger.error("DB log failed for decision %s on %s: %s",
                     decision_id, platform, e)

    if post_result["posted"] and db_logged:
        if platform == "twitter":
            result.posts_succeeded_twitter += 1
        else:
            result.posts_succeeded_bluesky += 1
    else:
        result.posts_failed += 1


def _post_quiet_day_recap(
    session_date: date,
    twitter_client,
    bluesky_client,
    result: TradePostsStageResult,
) -> None:
    """Quiet-day fallback: when no postable decisions, post the existing
    Mr. Krabs-style recap so the account doesn't go dark on trading days.

    Implemented in Task 7. Stub here so Task 6's orchestrator can wire it up.
    """
    # Real implementation lands in Task 7. For now: do nothing so Task 6 tests
    # can patch this symbol and verify it's called.
    return None


def run_trade_posts_stage(
    session_date: date | None = None,
    min_notional: float = DEFAULT_MIN_NOTIONAL,
    max_posts: int = DEFAULT_MAX_POSTS_PER_SESSION,
) -> TradePostsStageResult:
    """Post one tweet per significant non-hold decision today.

    See module docstring for full pipeline. Reuses the existing client
    factories + low-level posters from twitter.py / bluesky.py.
    """
    if session_date is None:
        session_date = date.today()

    result = TradePostsStageResult()

    twitter_client = get_twitter_client()
    bluesky_client = get_bluesky_client()
    if twitter_client is None and bluesky_client is None:
        result.skipped = True
        logger.info("Trade-posts stage skipped — no platform credentials")
        return result

    try:
        decisions = select_postable_decisions_for_date(
            session_date=session_date,
            min_notional=min_notional,
            limit=max_posts,
        )
    except Exception as e:
        result.errors.append(f"Failed to select postable decisions: {e}")
        logger.error("select_postable_decisions failed: %s", e)
        return result

    if not decisions:
        # Quiet-day fallback. The real implementation arrives in Task 7;
        # for Task 6 this is a wired-up call that just returns.
        _post_quiet_day_recap(session_date, twitter_client, bluesky_client, result)
        return result

    dashboard_base_url = os.environ.get("DASHBOARD_URL", "")

    for decision in decisions:
        result.posts_attempted += 1

        post_body = generate_trade_post(
            decision=decision,
            dashboard_base_url=dashboard_base_url,
        )
        if post_body is None:
            result.posts_failed += 1
            result.errors.append(
                f"Generation failed for decision {decision['id']}"
            )
            continue

        if twitter_client is not None:
            _platform_post_one(
                "twitter", post_body, decision, twitter_client,
                post_tweet, session_date, result,
            )
        if bluesky_client is not None:
            _platform_post_one(
                "bluesky", post_body, decision, bluesky_client,
                post_to_bluesky, session_date, result,
            )

    logger.info(
        "Trade-posts stage complete: attempted=%d, twitter_ok=%d, bluesky_ok=%d, "
        "failed=%d, dedup_skipped=%d",
        result.posts_attempted, result.posts_succeeded_twitter,
        result.posts_succeeded_bluesky, result.posts_failed,
        result.posts_skipped_dedup,
    )
    return result
```

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest tests/v2/test_social_trades.py -v
```

Expected: all `TestGenerateTradePost` and `TestRunTradePostsStage` tests pass.

- [ ] **Step 5: Run full v2 suite**

```bash
python3 -m pytest tests/v2/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add v2/social_trades.py tests/v2/test_social_trades.py
git commit -m "$(cat <<'EOF'
feat(v2): run_trade_posts_stage orchestrator

Iterate today's postable decisions; per-decision per-platform: dedup-check,
generate, post, log. Per-decision failures isolated — one bad LLM response
doesn't drop the rest of the burst. Empty-decisions case delegates to a
quiet-day recap stub (Task 7 lands the real fallback). ALGO_TRADE_POST_DRY_RUN
short-circuits both the platform post and the audit row so dry-run leaves no
trace.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `social_trades.py` — quiet-day fallback (mini-recap)

**Why:** When `select_postable_decisions_for_date` returns empty (a hold-only day, a no-trade day, a day with all trades below the notional floor), don't go dark — post the existing recap-style summary using `gather_tweet_context` + the Mr. Krabs prompt. Spec section "Architecture > Session orchestrator changes" + "Open questions" #2.

Skip the fallback entirely on weekends/holidays so the account doesn't post nothing-content on non-trading days.

**Files:**
- Modify: `v2/social_trades.py:_post_quiet_day_recap` (replace the stub)
- Modify: `tests/v2/test_social_trades.py` (add `TestQuietDayFallback`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_social_trades.py`:

```python
class TestQuietDayFallback:
    @patch("v2.social_trades.is_trading_day", return_value=False)
    def test_skips_quiet_day_recap_on_non_trading_day(self, mock_is_td):
        """Weekends/holidays produce no post — even the quiet-day recap is muted."""
        from datetime import date
        from v2.social_trades import _post_quiet_day_recap, TradePostsStageResult

        result = TradePostsStageResult()
        _post_quiet_day_recap(date(2026, 5, 9), object(), object(), result)  # Saturday
        assert result.quiet_day_recap_posted is False

    @patch("v2.social_trades.insert_tweet", return_value=1)
    @patch("v2.social_trades.posted_tweet_exists", return_value=False)
    @patch("v2.social_trades.post_to_bluesky")
    @patch("v2.social_trades.post_tweet")
    @patch("v2.social_trades.generate_bluesky_post")
    @patch("v2.social_trades.generate_tweet")
    @patch("v2.social_trades.gather_tweet_context", return_value="ctx")
    @patch("v2.social_trades.is_trading_day", return_value=True)
    def test_posts_recap_on_trading_day_no_decisions(
        self, mock_is_td, mock_ctx, mock_gen_tw, mock_gen_bs,
        mock_post_tw, mock_post_bs, mock_dedup, mock_insert,
    ):
        from datetime import date
        from v2.social_trades import _post_quiet_day_recap, TradePostsStageResult

        mock_gen_tw.return_value = {"text": "Quiet day.", "type": "recap"}
        mock_gen_bs.return_value = {"text": "Quiet day.", "type": "recap"}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "Quiet day.", "type": "recap", "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "Quiet day.", "type": "recap", "error": None}

        result = TradePostsStageResult()
        _post_quiet_day_recap(date(2026, 5, 4), object(), object(), result)

        assert result.quiet_day_recap_posted is True
        mock_post_tw.assert_called_once()
        mock_post_bs.assert_called_once()

    @patch("v2.social_trades.posted_tweet_exists", return_value=True)
    @patch("v2.social_trades.gather_tweet_context", return_value="ctx")
    @patch("v2.social_trades.is_trading_day", return_value=True)
    def test_dedup_blocks_recap_when_already_posted(
        self, mock_is_td, mock_ctx, mock_dedup,
    ):
        """Existing rerun guard: if a recap was already posted today on
        either platform, don't repost on that platform."""
        from datetime import date
        from v2.social_trades import _post_quiet_day_recap, TradePostsStageResult

        result = TradePostsStageResult()
        with patch("v2.social_trades.post_tweet") as mock_post_tw, \
             patch("v2.social_trades.post_to_bluesky") as mock_post_bs:
            _post_quiet_day_recap(date(2026, 5, 4), object(), object(), result)

        # Both platforms guarded → no posts attempted
        mock_post_tw.assert_not_called()
        mock_post_bs.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/v2/test_social_trades.py::TestQuietDayFallback -v
```

Expected: FAIL — the stub returns None and never sets `quiet_day_recap_posted`.

- [ ] **Step 3: Replace the stub with the real implementation**

In `v2/social_trades.py`, locate the `_post_quiet_day_recap` stub from Task 6 and replace it with the full implementation. Add the required imports at the top of the file (next to the existing `from .twitter import` line):

```python
from .market_calendar import is_trading_day                        # noqa: E402
from .twitter import gather_tweet_context, generate_tweet           # noqa: E402
from .bluesky import generate_bluesky_post                          # noqa: E402
```

Replace `_post_quiet_day_recap` with:

```python
def _post_quiet_day_recap(
    session_date: date,
    twitter_client,
    bluesky_client,
    result: TradePostsStageResult,
) -> None:
    """Quiet-day fallback: post the existing recap-style summary on trading
    days when there are no postable decisions. Skipped entirely on weekends
    and NYSE holidays so the account doesn't spam non-content."""
    if not is_trading_day(session_date):
        logger.info("Quiet-day recap skipped — %s is not a trading day", session_date)
        return

    try:
        context = gather_tweet_context(session_date)
    except Exception as e:
        result.errors.append(f"Quiet-day context failed: {e}")
        logger.error("Quiet-day context gather failed: %s", e)
        return

    posted_any = False

    if twitter_client is not None:
        try:
            already = posted_tweet_exists(session_date, "recap", "twitter")
        except Exception as e:
            logger.warning("Quiet-day twitter dedup check failed: %s", e)
            already = False
        if not already:
            tweet = generate_tweet(context)
            if tweet:
                if _is_dry_run():
                    logger.info("[DRY-RUN] quiet-day twitter recap:\n%s", tweet["text"])
                    posted_any = True
                else:
                    try:
                        post_result = post_tweet(tweet, client=twitter_client)
                        insert_tweet(
                            session_date=session_date,
                            tweet_type=post_result.get("type", "recap"),
                            tweet_text=post_result["text"],
                            tweet_id=post_result.get("tweet_id"),
                            posted=post_result["posted"],
                            error=post_result.get("error"),
                            platform="twitter",
                        )
                        if post_result["posted"]:
                            posted_any = True
                    except Exception as e:
                        result.errors.append(f"Quiet-day twitter recap failed: {e}")
                        logger.error("Quiet-day twitter recap failed: %s", e)

    if bluesky_client is not None:
        try:
            already = posted_tweet_exists(session_date, "recap", "bluesky")
        except Exception as e:
            logger.warning("Quiet-day bluesky dedup check failed: %s", e)
            already = False
        if not already:
            post = generate_bluesky_post(context)
            if post:
                if _is_dry_run():
                    logger.info("[DRY-RUN] quiet-day bluesky recap:\n%s", post["text"])
                    posted_any = True
                else:
                    try:
                        post_result = post_to_bluesky(post, client=bluesky_client)
                        insert_tweet(
                            session_date=session_date,
                            tweet_type=post_result.get("type", "recap"),
                            tweet_text=post_result["text"],
                            tweet_id=post_result.get("post_id"),
                            posted=post_result["posted"],
                            error=post_result.get("error"),
                            platform="bluesky",
                        )
                        if post_result["posted"]:
                            posted_any = True
                    except Exception as e:
                        result.errors.append(f"Quiet-day bluesky recap failed: {e}")
                        logger.error("Quiet-day bluesky recap failed: %s", e)

    result.quiet_day_recap_posted = posted_any
```

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest tests/v2/test_social_trades.py -v
```

Expected: all `TestQuietDayFallback` tests pass; existing `TestRunTradePostsStage` tests still pass.

- [ ] **Step 5: Run full v2 suite**

```bash
python3 -m pytest tests/v2/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add v2/social_trades.py tests/v2/test_social_trades.py
git commit -m "$(cat <<'EOF'
feat(v2): quiet-day recap fallback in run_trade_posts_stage

When no postable decisions land, fall through to the existing recap path
on trading days only. Weekends and NYSE holidays produce no post at all
so the account doesn't spam non-content. Reuses gather_tweet_context +
the Mr. Krabs prompt unchanged so existing copy quality carries over.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire `run_trade_posts_stage` into `v2/session.py` behind `ALGO_ENABLE_TRADE_POSTS`

**Why:** Spec rollout: deploy code with the new pipeline gated behind `ALGO_ENABLE_TRADE_POSTS=1`, leaving the existing recap path running when the flag is off. After one week of clean prod runs, a follow-up plan will delete the old path.

**Files:**
- Modify: `v2/session.py` (add new wrapper + branching after Stage 4)
- Modify: `tests/v2/test_session.py` (add a branching test)

- [ ] **Step 1: Read the current twitter+bluesky stage wiring**

```bash
sed -n '478,490p' v2/session.py
```

Confirm the calls look like:

```python
_run_twitter_stage_wrapper(result, session_id, completed_stages, skip_twitter)
_run_bluesky_stage_wrapper(result, session_id, completed_stages, skip_bluesky)
_run_dashboard_stage_wrapper(result, session_id, completed_stages, skip_dashboard)
```

Find the existing `_run_twitter_stage_wrapper` definition — it provides the template for the new wrapper. Note its signature, how it appends errors to `result`, and how it logs `session_stages` completion.

```bash
grep -n "_run_twitter_stage_wrapper" v2/session.py
```

- [ ] **Step 2: Write the failing test**

Add to `tests/v2/test_session.py`. Find the existing `TestRunSession` (or similarly-named) class that exercises the orchestrator. Append a new sub-test:

```python
def test_trade_posts_flag_routes_to_new_stage_and_skips_old(
    self, mock_db, mock_cursor, monkeypatch,
):
    """When ALGO_ENABLE_TRADE_POSTS=1, run_trade_posts_stage runs and
    the legacy run_twitter_stage / run_bluesky_stage are NOT called."""
    from unittest.mock import patch, MagicMock
    from v2.session import run_session
    from v2.social_trades import TradePostsStageResult

    monkeypatch.setenv("ALGO_ENABLE_TRADE_POSTS", "1")

    # Stub everything before stage 5; we only care about the social branch.
    with patch("v2.session.run_pipeline", return_value=MagicMock()), \
         patch("v2.session.run_strategist_loop", return_value=MagicMock()), \
         patch("v2.session.run_trading_session", return_value=MagicMock()), \
         patch("v2.session.run_strategy_stage", return_value=MagicMock()), \
         patch("v2.session.run_dashboard_stage", return_value=MagicMock()), \
         patch("v2.session.get_session_for_date", return_value=None), \
         patch("v2.session.insert_session_record", return_value=42), \
         patch("v2.session.run_trade_posts_stage",
               return_value=TradePostsStageResult()) as mock_new, \
         patch("v2.session.run_twitter_stage") as mock_old_tw, \
         patch("v2.session.run_bluesky_stage") as mock_old_bs:
        run_session(
            dry_run=False, model="x", executor_model="y", max_turns=1,
            skip_pipeline=False, skip_ideation=False, skip_executor=False,
            skip_strategy=False, skip_twitter=False, skip_bluesky=False,
            skip_dashboard=True, pipeline_hours=24, force=False,
        )

    mock_new.assert_called_once()
    mock_old_tw.assert_not_called()
    mock_old_bs.assert_not_called()


def test_trade_posts_flag_off_runs_legacy_stages(
    self, mock_db, mock_cursor, monkeypatch,
):
    """Default behavior: ALGO_ENABLE_TRADE_POSTS unset → legacy path runs."""
    from unittest.mock import patch, MagicMock
    from v2.session import run_session
    from v2.twitter import TwitterStageResult
    from v2.bluesky import BlueskyStageResult

    monkeypatch.delenv("ALGO_ENABLE_TRADE_POSTS", raising=False)

    with patch("v2.session.run_pipeline", return_value=MagicMock()), \
         patch("v2.session.run_strategist_loop", return_value=MagicMock()), \
         patch("v2.session.run_trading_session", return_value=MagicMock()), \
         patch("v2.session.run_strategy_stage", return_value=MagicMock()), \
         patch("v2.session.run_dashboard_stage", return_value=MagicMock()), \
         patch("v2.session.get_session_for_date", return_value=None), \
         patch("v2.session.insert_session_record", return_value=42), \
         patch("v2.session.run_trade_posts_stage") as mock_new, \
         patch("v2.session.run_twitter_stage",
               return_value=TwitterStageResult()) as mock_old_tw, \
         patch("v2.session.run_bluesky_stage",
               return_value=BlueskyStageResult()) as mock_old_bs:
        run_session(
            dry_run=False, model="x", executor_model="y", max_turns=1,
            skip_pipeline=False, skip_ideation=False, skip_executor=False,
            skip_strategy=False, skip_twitter=False, skip_bluesky=False,
            skip_dashboard=True, pipeline_hours=24, force=False,
        )

    mock_new.assert_not_called()
    mock_old_tw.assert_called_once()
    mock_old_bs.assert_called_once()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python3 -m pytest tests/v2/test_session.py -k "trade_posts_flag" -v
```

Expected: FAIL — `run_trade_posts_stage` isn't imported in `v2/session.py` yet.

- [ ] **Step 4: Wire the new stage in `v2/session.py`**

Open `v2/session.py`. Add the import at the top with the other module imports:

```python
from .social_trades import run_trade_posts_stage, TradePostsStageResult
```

Find the `SessionResult` dataclass (around line 58). Add a new field:

```python
@dataclass
class SessionResult:
    # ... existing fields ...
    trade_posts_result: TradePostsStageResult | None = None
```

Add it next to the existing `twitter_result` / `bluesky_result` fields so the layout is contiguous.

Find the existing `_run_twitter_stage_wrapper` function (use grep). Add a sibling wrapper right next to it:

```python
def _run_trade_posts_stage_wrapper(
    result: SessionResult,
    session_id: int | None,
    completed_stages: set,
    skip: bool,
) -> None:
    """New live-trade pipeline stage; runs in place of the legacy twitter+bluesky
    stages when ALGO_ENABLE_TRADE_POSTS=1. Idempotency, error trapping, and
    session_stages completion follow the same shape as the legacy wrappers."""
    if skip:
        return
    if "trade_posts" in completed_stages:
        logger.info("Trade-posts stage already completed for this session; skipping")
        return
    try:
        result.trade_posts_result = run_trade_posts_stage()
    except Exception as e:
        logger.error("Trade-posts stage crashed: %s", e)
        result.errors.append(f"trade_posts: {e}")
        return
    if session_id is not None:
        try:
            _record_stage_complete(session_id, "trade_posts")
        except Exception as e:
            logger.warning("Failed to record trade_posts stage completion: %s", e)
```

(`_record_stage_complete` is the existing helper used by other wrappers; if it's named differently in your tree, match the existing pattern.)

Find the lines that call the legacy wrappers (around line 481-482):

```python
_run_twitter_stage_wrapper(result, session_id, completed_stages, skip_twitter)
_run_bluesky_stage_wrapper(result, session_id, completed_stages, skip_bluesky)
```

Replace those two lines with a feature-flag branch:

```python
if os.environ.get("ALGO_ENABLE_TRADE_POSTS") == "1":
    # New live-trade pipeline. --skip-twitter / --skip-bluesky still apply
    # but propagate inside run_trade_posts_stage (which decides per-platform
    # based on the available client). Treat skip_twitter+skip_bluesky as
    # an OR-skip of the whole stage for simplicity until we add a dedicated
    # --skip-trade-posts flag.
    skip_combined = skip_twitter and skip_bluesky
    _run_trade_posts_stage_wrapper(result, session_id, completed_stages, skip_combined)
else:
    _run_twitter_stage_wrapper(result, session_id, completed_stages, skip_twitter)
    _run_bluesky_stage_wrapper(result, session_id, completed_stages, skip_bluesky)
```

Make sure `import os` is at the top of `v2/session.py` (it almost certainly already is — `grep -n "^import os" v2/session.py` to confirm).

- [ ] **Step 5: Run the new branch tests**

```bash
python3 -m pytest tests/v2/test_session.py -k "trade_posts_flag" -v
```

Expected: both tests PASS.

- [ ] **Step 6: Run full v2 suite**

```bash
python3 -m pytest tests/v2/ -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add v2/session.py tests/v2/test_session.py
git commit -m "$(cat <<'EOF'
feat(v2): gate live-trade pipeline behind ALGO_ENABLE_TRADE_POSTS

When the env var is set to "1", session.py runs run_trade_posts_stage in
place of the legacy run_twitter_stage / run_bluesky_stage. Default (flag
off) preserves the current recap behavior so we can roll out the new
pipeline without forcing the cutover. After one week of clean prod runs
we'll delete the legacy path in a follow-up plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `v2/premarket.py` — context + post generation

**Why:** Pre-market post: forward-looking take on watched names, posted before the next session. Independent module + entrypoint, invoked from a separate cron schedule (Taskfile target lands in Task 11).

**Files:**
- Create: `v2/premarket.py` (context + generation only; orchestrator lands in Task 10)
- Create: `tests/v2/test_premarket.py`
- Modify: `tests/v2/conftest.py` (add `v2.premarket.get_cursor` to the `mock_db` patch list)

- [ ] **Step 1: Write the failing test**

Create `tests/v2/test_premarket.py`:

```python
"""Tests for v2/premarket.py — pre-market social post pipeline."""

import json
from datetime import date
from unittest.mock import MagicMock, patch


class TestGatherPremarketContext:
    def test_assembles_active_theses_and_latest_memo(self, mock_db, mock_cursor):
        from v2.premarket import gather_premarket_context

        # First fetchall: active theses (top 5 by confidence)
        # Second fetchone: latest strategy memo
        mock_cursor.fetchall.side_effect = [
            [
                {"ticker": "NVDA", "direction": "long", "thesis": "AI demand",
                 "confidence": "high"},
                {"ticker": "TSLA", "direction": "long", "thesis": "EV cycle",
                 "confidence": "medium"},
            ],
        ]
        mock_cursor.fetchone.side_effect = [
            {"content": "Yesterday: NVDA broke out, holding overnight."},
        ]

        ctx = gather_premarket_context(today=date(2026, 5, 4))

        assert "ACTIVE THESES" in ctx
        assert "NVDA" in ctx
        assert "TSLA" in ctx
        assert "STRATEGY MEMO" in ctx
        assert "broke out" in ctx

    def test_handles_no_theses_no_memo(self, mock_db, mock_cursor):
        from v2.premarket import gather_premarket_context

        mock_cursor.fetchall.side_effect = [[]]
        mock_cursor.fetchone.side_effect = [None]

        ctx = gather_premarket_context(today=date(2026, 5, 4))
        # Empty-state still returns a non-empty string so the LLM has
        # something to anchor on
        assert isinstance(ctx, str)
        assert len(ctx) > 0


def _make_claude_response(json_data: dict):
    """Helper: shape a MagicMock the way `_call_with_retry` returns one."""
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(json_data))]
    return response


class TestGeneratePremarketPost:
    @patch("v2.premarket._call_with_retry")
    @patch("v2.premarket.get_claude_client")
    def test_generates_text(self, mock_get_client, mock_retry):
        from v2.premarket import generate_premarket_post

        mock_get_client.return_value = MagicMock()
        mock_retry.return_value = _make_claude_response(
            {"text": "Watching $NVDA into open. AI demand still pulling."}
        )

        result = generate_premarket_post("ctx")
        assert result is not None
        assert "$NVDA" in result["text"]
        assert result["type"] == "premarket"

    @patch("v2.premarket._call_with_retry", side_effect=Exception("API down"))
    @patch("v2.premarket.get_claude_client")
    def test_llm_failure_returns_none(self, mock_get_client, mock_retry):
        from v2.premarket import generate_premarket_post

        mock_get_client.return_value = MagicMock()

        assert generate_premarket_post("ctx") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/v2/test_premarket.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'v2.premarket'`.

- [ ] **Step 3: Update `tests/v2/conftest.py` to patch `v2.premarket.get_cursor`**

Open `tests/v2/conftest.py`. Find the `mock_db` fixture (around line 86) — it has a series of `patch("v2.<module>.get_cursor", _get_cursor)` lines. Add a new line for `v2.premarket` so `gather_premarket_context` (which imports `get_cursor` directly) sees the mock:

```python
         patch("v2.bluesky.get_cursor", _get_cursor), \
         patch("v2.dashboard_publish.get_cursor", _get_cursor), \
         patch("v2.premarket.get_cursor", _get_cursor), \
```

(Order doesn't matter — just place it next to the other patches.)

- [ ] **Step 4: Create `v2/premarket.py` (context + generation)**

Create `v2/premarket.py`:

```python
"""Pre-market social post pipeline -- Bikini Bottom Capital (v2).

Forward-looking take posted before the next session. Triggered by cron
(see Taskfile premarket:stage target), separate from the daily session.
Skipped on weekends and NYSE holidays.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date

from .claude_client import _call_with_retry, get_claude_client
from .database.connection import get_cursor
from .database.trading_db import insert_tweet, posted_tweet_exists
from .market_calendar import is_trading_day

logger = logging.getLogger("premarket")


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def gather_premarket_context(today: date | None = None) -> str:
    """Plain-text summary of what the bot is watching pre-market.

    Sections:
    - Top 5 active theses by confidence
    - Latest strategy memo (yesterday's reflection)
    """
    if today is None:
        today = date.today()

    sections: list[str] = []
    with get_cursor() as cur:
        cur.execute(
            "SELECT ticker, direction, thesis, confidence FROM theses "
            "WHERE status = 'active' "
            "ORDER BY CASE confidence "
            "  WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
            "  created_at DESC LIMIT 5"
        )
        theses = cur.fetchall()
        if theses:
            lines = ["ACTIVE THESES:"]
            for t in theses:
                lines.append(
                    f"  {t['ticker']} ({t['direction']}, {t['confidence']}): {t['thesis']}"
                )
            sections.append("\n".join(lines))

        cur.execute("SELECT content FROM strategy_memos ORDER BY created_at DESC LIMIT 1")
        memo = cur.fetchone()
        if memo and memo.get("content"):
            sections.append(f"STRATEGY MEMO:\n  {memo['content']}")

    if not sections:
        return f"Pre-market for {today}. No active theses; no recent memo."
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

PREMARKET_SYSTEM_PROMPT = """You run an algorithmic trading operation called Bikini Bottom Capital.
You're posting before market open. The bot will run its session after close.

Your voice:
- Casual, observational. What you're watching, what's interesting.
- Forward-looking but not predictive. No "this will rip" claims.
- Honest about uncertainty.

Respond with JSON: {"text": "post text here"}

Rules:
- 220 chars max (no URL appended for this type).
- Reference 1–2 names from your current theses or pre-market movers.
- One observation about what you're watching today.
- No P&L claims, no historical performance flexes.
- $CASHTAG only for tickers you mention."""


def generate_premarket_post(
    context: str,
    model: str = "claude-haiku-4-5-20251001",
) -> dict | None:
    """Generate one pre-market post body. Returns dict or None on failure."""
    try:
        client = get_claude_client()
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=512,
            system=PREMARKET_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        raw = response.content[0].text.strip()
        logger.info("AI response:\n%s", raw)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0].strip()
        result = json.loads(raw)
    except Exception as e:
        logger.error("Failed to generate pre-market post: %s", e)
        return None

    text = result.get("text")
    if not text or not isinstance(text, str):
        logger.warning("LLM returned no text or malformed response: %s", result)
        return None

    return {"text": text, "type": "premarket"}
```

- [ ] **Step 5: Run the tests**

```bash
python3 -m pytest tests/v2/test_premarket.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add v2/premarket.py tests/v2/test_premarket.py tests/v2/conftest.py
git commit -m "$(cat <<'EOF'
feat(v2): premarket context gather + post generation

gather_premarket_context: top-5 active theses + latest strategy memo.
generate_premarket_post: Haiku call with the forward-looking voice
prompt. Orchestrator with weekend/holiday skip + posting lands in Task 10.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `v2/premarket.py` — `run_premarket_stage` orchestrator + `__main__`

**Files:**
- Modify: `v2/premarket.py` (append the orchestrator + module-main entry)
- Modify: `tests/v2/test_premarket.py` (add `TestRunPremarketStage`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/v2/test_premarket.py`:

```python
class TestRunPremarketStage:
    @patch("v2.premarket.is_trading_day", return_value=False)
    def test_skipped_on_weekend(self, mock_is_td):
        from v2.premarket import run_premarket_stage

        result = run_premarket_stage(today=date(2026, 5, 9))  # Saturday
        assert result.skipped is True
        assert "weekend" in (result.skip_reason or "").lower() or \
               "trading day" in (result.skip_reason or "").lower()

    @patch("v2.premarket.is_trading_day", return_value=True)
    @patch("v2.premarket.posted_tweet_exists", return_value=True)
    @patch("v2.premarket.get_twitter_client")
    @patch("v2.premarket.get_bluesky_client")
    def test_skipped_when_already_posted_today(
        self, mock_bs, mock_tw, mock_dedup, mock_is_td,
    ):
        """Idempotent: if both platforms already posted today, the whole
        stage short-circuits."""
        from v2.premarket import run_premarket_stage

        mock_tw.return_value = object()
        mock_bs.return_value = object()
        # posted_tweet_exists returns True for both platforms

        result = run_premarket_stage(today=date(2026, 5, 4))
        assert result.skipped is True

    @patch("v2.premarket.is_trading_day", return_value=True)
    @patch("v2.premarket.insert_tweet", return_value=1)
    @patch("v2.premarket.posted_tweet_exists", return_value=False)
    @patch("v2.premarket.post_to_bluesky")
    @patch("v2.premarket.post_tweet")
    @patch("v2.premarket.generate_premarket_post")
    @patch("v2.premarket.gather_premarket_context", return_value="ctx")
    @patch("v2.premarket.get_twitter_client")
    @patch("v2.premarket.get_bluesky_client")
    def test_posts_to_both_platforms_on_trading_day(
        self, mock_bs_client, mock_tw_client, mock_ctx, mock_gen,
        mock_post_tw, mock_post_bs, mock_dedup, mock_insert, mock_is_td,
    ):
        from v2.premarket import run_premarket_stage

        mock_tw_client.return_value = object()
        mock_bs_client.return_value = object()
        mock_gen.return_value = {"text": "Watching $NVDA", "type": "premarket"}
        mock_post_tw.return_value = {"posted": True, "tweet_id": "tw1",
                                     "text": "x", "type": "premarket", "error": None}
        mock_post_bs.return_value = {"posted": True, "post_id": "bs1",
                                     "text": "x", "type": "premarket", "error": None}

        result = run_premarket_stage(today=date(2026, 5, 4))

        assert result.skipped is False
        assert result.twitter_posted is True
        assert result.bluesky_posted is True
        assert mock_insert.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/v2/test_premarket.py::TestRunPremarketStage -v
```

Expected: FAIL — `run_premarket_stage` doesn't exist.

- [ ] **Step 3: Append the orchestrator + `__main__` to `v2/premarket.py`**

Append to `v2/premarket.py`:

```python
# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

from .twitter import get_twitter_client, post_tweet           # noqa: E402
from .bluesky import get_bluesky_client, post_to_bluesky      # noqa: E402


@dataclass
class PremarketStageResult:
    skipped: bool = False
    skip_reason: str | None = None
    twitter_posted: bool = False
    bluesky_posted: bool = False
    errors: list[str] = field(default_factory=list)


def _is_dry_run() -> bool:
    return os.environ.get("ALGO_TRADE_POST_DRY_RUN") == "1"


def run_premarket_stage(today: date | None = None) -> PremarketStageResult:
    """Generate + post one pre-market take to Twitter and Bluesky.

    Skipped on weekends and NYSE holidays. Idempotent via
    posted_tweet_exists(today, "premarket", platform) so cron retries
    don't double-post.
    """
    if today is None:
        today = date.today()

    result = PremarketStageResult()

    if not is_trading_day(today):
        result.skipped = True
        result.skip_reason = f"{today} is not a trading day"
        logger.info("Pre-market stage skipped — %s", result.skip_reason)
        return result

    twitter_client = get_twitter_client()
    bluesky_client = get_bluesky_client()
    if twitter_client is None and bluesky_client is None:
        result.skipped = True
        result.skip_reason = "no platform credentials"
        logger.info("Pre-market stage skipped — no platform credentials")
        return result

    # Both platforms might already be posted (cron retry on the same day).
    tw_already = False
    bs_already = False
    try:
        if twitter_client is not None:
            tw_already = posted_tweet_exists(today, "premarket", "twitter")
        if bluesky_client is not None:
            bs_already = posted_tweet_exists(today, "premarket", "bluesky")
    except Exception as e:
        logger.warning("Pre-market dedup check failed: %s; proceeding", e)

    if (twitter_client is None or tw_already) and (bluesky_client is None or bs_already):
        result.skipped = True
        result.skip_reason = "already posted on all configured platforms"
        logger.info("Pre-market stage skipped — already posted today")
        return result

    try:
        context = gather_premarket_context(today)
    except Exception as e:
        result.errors.append(f"Context gather failed: {e}")
        logger.error("Pre-market context gather failed: %s", e)
        return result

    post = generate_premarket_post(context)
    if post is None:
        result.errors.append("LLM generation returned None")
        return result

    if twitter_client is not None and not tw_already:
        if _is_dry_run():
            logger.info("[DRY-RUN] premarket twitter:\n%s", post["text"])
            result.twitter_posted = True
        else:
            try:
                post_result = post_tweet(post, client=twitter_client)
                insert_tweet(
                    session_date=today,
                    tweet_type="premarket",
                    tweet_text=post_result["text"],
                    tweet_id=post_result.get("tweet_id"),
                    posted=post_result["posted"],
                    error=post_result.get("error"),
                    platform="twitter",
                )
                result.twitter_posted = post_result["posted"]
            except Exception as e:
                result.errors.append(f"Twitter post/log failed: {e}")
                logger.error("Twitter post/log failed: %s", e)

    if bluesky_client is not None and not bs_already:
        if _is_dry_run():
            logger.info("[DRY-RUN] premarket bluesky:\n%s", post["text"])
            result.bluesky_posted = True
        else:
            try:
                post_result = post_to_bluesky(post, client=bluesky_client)
                insert_tweet(
                    session_date=today,
                    tweet_type="premarket",
                    tweet_text=post_result["text"],
                    tweet_id=post_result.get("post_id"),
                    posted=post_result["posted"],
                    error=post_result.get("error"),
                    platform="bluesky",
                )
                result.bluesky_posted = post_result["posted"]
            except Exception as e:
                result.errors.append(f"Bluesky post/log failed: {e}")
                logger.error("Bluesky post/log failed: %s", e)

    logger.info("Pre-market stage complete: twitter=%s, bluesky=%s",
                result.twitter_posted, result.bluesky_posted)
    return result


if __name__ == "__main__":  # pragma: no cover
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
    res = run_premarket_stage()
    if res.errors:
        import sys
        sys.exit(1)
```

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest tests/v2/test_premarket.py -v
```

Expected: all pass.

- [ ] **Step 5: Run full v2 suite**

```bash
python3 -m pytest tests/v2/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add v2/premarket.py tests/v2/test_premarket.py
git commit -m "$(cat <<'EOF'
feat(v2): run_premarket_stage orchestrator + module entrypoint

Skip on weekends/holidays via market_calendar.is_trading_day. Idempotent
via posted_tweet_exists(today, "premarket", platform) so cron retries
don't double-post. ALGO_TRADE_POST_DRY_RUN supported. Module __main__
entry runs the full stage so cron can invoke `python -m v2.premarket`
directly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Taskfile target for the pre-market stage

**Files:**
- Modify: `Taskfile.yml`

- [ ] **Step 1: Read the existing entertainment target**

```bash
grep -n -A 6 "entertainment:" Taskfile.yml
```

Use it as the template — entertainment uses the same shape: a top-level Taskfile target invoking `python -m v2.<module>` inside the trading container.

- [ ] **Step 2: Add the `premarket` target**

In `Taskfile.yml`, add a new entry under the `tasks:` block, modeled on `entertainment`. Place it near the entertainment target for grouping. The block should look like:

```yaml
  premarket:
    desc: Run the pre-market social post stage (skips on weekends/holidays)
    cmds:
      - docker compose exec trading python -m v2.premarket {{.CLI_ARGS}}
```

- [ ] **Step 3: Verify by listing tasks**

```bash
task -l 2>&1 | grep -E "premarket|entertainment"
```

Expected: both `premarket` and `entertainment` appear with their descriptions.

- [ ] **Step 4: Smoke-test the target (dry-run)**

```bash
ALGO_TRADE_POST_DRY_RUN=1 task premarket
```

Expected: container runs `python -m v2.premarket`; logs show either "Pre-market stage skipped — no platform credentials" (if creds aren't on the container env) or "[DRY-RUN] premarket twitter:" output. Either is fine — the goal is no crash.

- [ ] **Step 5: Commit**

```bash
git add Taskfile.yml
git commit -m "$(cat <<'EOF'
chore(v2): Taskfile target for premarket stage

`task premarket` runs `python -m v2.premarket` inside the trading
container. Cron should invoke this target on weekday mornings ~07:30 ET.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Update CLAUDE.md to document new stages and flags

**Why:** The session-stage table in CLAUDE.md lists the existing pipeline. The new flag and pre-market stage need to be discoverable.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read the current stage table**

```bash
grep -n -A 15 "v2 Daily Session" CLAUDE.md
```

Locate the markdown table that lists Stage 0–6.

- [ ] **Step 2: Update the table + add a new section**

In `CLAUDE.md`, find the row for Stage 5 (`twitter.py`, `bluesky.py`). Update its description to:

```
| 5 | `twitter.py` / `bluesky.py` (legacy) or `social_trades.py` (new, gated by `ALGO_ENABLE_TRADE_POSTS=1`) | Social posting |
```

After the stage table, add a new subsection:

```markdown
### Pre-market post stage

Independent of the daily session. Triggered by cron via `task premarket`
(or `python -m v2.premarket` directly). Skipped on weekends and NYSE
holidays. Posts a forward-looking take referencing 1–2 names from
active theses + the latest session memo.

### Live-trade pipeline feature flag

When `ALGO_ENABLE_TRADE_POSTS=1`, Stage 5 runs `run_trade_posts_stage`
instead of the legacy `run_twitter_stage` + `run_bluesky_stage`:

- Iterates today's significant non-hold decisions (notional ≥ `$100`).
- Posts one tweet per decision to Twitter + Bluesky, each linking to
  `/trade/<id>/` and (if present) `/thesis/<id>/` on the public dashboard.
- Caps at 5 posts per session.
- Quiet-day fallback: if no postable decisions, posts a mini-recap on
  trading days only.
- `ALGO_TRADE_POST_DRY_RUN=1` logs generated post bodies and skips both
  platform posts and the DB audit row.

The legacy recap path (twitter.py / bluesky.py orchestrators) stays
intact while the new pipeline is being validated; a follow-up plan will
delete it after one week of clean prod runs.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: live-trade pipeline + premarket stage in CLAUDE.md

Document the new Stage 5 routing (ALGO_ENABLE_TRADE_POSTS), the
ALGO_TRADE_POST_DRY_RUN flag, and the independent premarket cron stage.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Run the full v2 test suite**

```bash
python3 -m pytest tests/v2/ -q
```

Expected: all tests pass. Pre-pass count was 916 (per the baseline run); the new count should be 916 + (number of new tests added across tasks 1-10) = ~940-945.

- [ ] **Run the full repo test suite**

```bash
python3 -m pytest tests/ -q
```

Expected: all pass. The legacy `tests/test_*.py` modules cover v1 dashboard/trading and shouldn't be affected by any of these changes.

- [ ] **End-to-end smoke test (dry-run, no real posts)**

```bash
docker compose exec -e ALGO_ENABLE_TRADE_POSTS=1 -e ALGO_TRADE_POST_DRY_RUN=1 \
    trading python -m v2.session --skip-dashboard
```

Expected: session runs through all stages; Stage 5 logs `[DRY-RUN] twitter post for decision N:` lines (or "skipped — no platform credentials" / "no postable decisions, falling through to quiet-day"). No exceptions.

- [ ] **Smoke test the premarket entrypoint**

```bash
docker compose exec -e ALGO_TRADE_POST_DRY_RUN=1 trading python -m v2.premarket
```

Expected: either "skipped — not a trading day" (if today is weekend/holiday), "skipped — no platform credentials" (if creds missing on container env), or `[DRY-RUN] premarket ...` log lines. No crash.

- [ ] **Run with coverage**

```bash
python3 -m pytest tests/v2/ --cov=v2 --cov-report=term-missing -q
```

Expected: v2 coverage holds at or above the pre-pass level. The new modules (`market_calendar.py`, `social_trades.py`, `premarket.py`) should each have ≥85% line coverage from the new tests.
