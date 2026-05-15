# Pinchy

A Claude-powered trading bot that runs a real Alpaca brokerage account. It journals every decision, scores its own signals against outcomes, reflects on its mistakes after the close, and publishes everything.

**Live:**
- Dashboard — [pinchy.dev](https://pinchy.dev) *(domain swap pending)*
- What's been working — [/attribution/](https://pinchy.dev/attribution/)
- What hasn't — [/mistakes/](https://pinchy.dev/mistakes/)
- Today's reasoning — [/activity/](https://pinchy.dev/activity/)

## What this is

An experiment to see whether an agentic LLM, given Alpaca read/write access and a feedback loop of its own past trades, can find a durable edge in liquid US equities.

This is a side project, not a product. It's also not a success story — at the time of writing, the edge is **unproven**. The interesting work is in the infrastructure around the question: the journaling, the attribution scoring, the reflection loop, and the public dashboard that makes it impossible to selectively report wins.

Built and maintained by [Jay Shaffer](https://github.com/jayshaffer). Claude (Sonnet and Opus, depending on the stage) wrote most of the code, including this README; the strategy and architecture decisions are mine.

## How it works

The system runs once a day after the US market close. Each session is a sequence of five stages that share state via Postgres:

| Stage | Module | What it does |
|-------|--------|--------------|
| 0 | `backfill.py`, `attribution.py` | Look up 7-day and 30-day outcomes for recent decisions; recompute which signal types are predicting P&L |
| 1 | `pipeline.py` | Fetch broad market news, filter by relevance with embeddings, classify with Haiku, store as signals |
| 2 | `ideation_claude.py` | Strategist loop: Opus manages trade theses and writes today's playbook using 11 DB-backed tools |
| 3 | `trader.py` | Executor loop: Haiku reads the playbook plus current portfolio state and produces buy/sell/hold decisions; orders go to Alpaca |
| 4 | `strategy.py` | Reflection: Opus reviews the day's outcomes, proposes or retires trading rules, writes a session memo |
| 5 | `dashboard_publish.py` | Render the static public site, upload to Cloudflare Pages |

The strategist and executor are deliberately split. The strategist is slower, smarter, and writes a daily playbook (a list of structured actions: buy this much of X if Y triggers). The executor is fast, follows the playbook, and is the only stage that can move money. This split is the cheapest way to keep the smart model from over-trading and the cheap model from making strategic mistakes.

Run-to-run continuity comes from four persistent artifacts: a **trading identity** the reflection stage updates, a set of **strategy rules** with explicit propose/retire lifecycles, **theses** with entry/exit triggers carried across sessions, and **signal attribution scores** showing which signal types have been predictive over the trailing 30 days.

For a deeper architecture walk-through, see [CLAUDE.md](./CLAUDE.md).

## One real failure mode

A vignette to give a sense of the kind of bugs that show up when an LLM agent runs an account:

In late April through early May 2026, the bot round-tripped GOOGL **eleven times in twenty-two days**. CRM did it nine times, NVDA nine, AMZN seven. Each round-trip: a small trim citing "Rule #27 — $500/day cap during fragile macro windows" plus some binary event (FOMC, ceasefire talks, earnings), then a rebuy a day or two later at a higher price citing "Rule #27 has lifted now that the event resolved." Net of fees and slippage, every cycle was negative.

The bug wasn't in the executor. The bug was that **Rule #27 had no numeric bind or lift threshold** — both conditions were narrative phrases the strategist re-interpreted every morning. The reflection stage didn't catch it because reflection counted rule *citations* (more citations = "more disciplined") instead of counting *round-trips per ticker*. The system was systematically rewarding the behavior that was bleeding money.

Fix path: surface flip-flop evidence to the reflection stage so the strategist confronts its own round-trips; add numeric thresholds to oscillating rules; add an executor-side cooldown on opposite-side decisions on the same ticker. A full postmortem with the queries and the fix is coming as a separate writeup.

This is what most of the work on this project looks like. The agent does something locally reasonable, but the reasonable-locally behavior compounds into a structural failure that's only visible when you query the right shape of the decisions table.

## Stack

- **LLM** — Anthropic Claude (Opus for the strategist, Sonnet for reflection, Haiku for the executor and news classifier) via the official Python SDK
- **Data** — Alpaca for market data and order execution; Alpaca news feed for signals
- **Storage** — PostgreSQL 16 + pgvector for embedding-based news deduplication
- **Runtime** — Docker Compose stack (trading agent, db, dashboard) with separate prod and paper-trading stacks selected by overlay
- **Public surface** — Static HTML pages assembled and pushed to Cloudflare Pages each session

## Running it yourself

You'll need an Alpaca account (paper trading is fine for trying it), an Anthropic API key, and Docker.

```bash
git clone https://github.com/jayshaffer/algo.git pinchy
cd pinchy
cp .env.example .env       # fill in ALPACA_*, ANTHROPIC_API_KEY, POSTGRES_*
docker compose up -d
docker compose exec trading python -m v2.session --dry-run
```

The `--dry-run` flag runs the full session but blocks order submission and skips the dashboard publish — useful for confirming the stack is wired correctly without touching your account.

To run for real, drop `--dry-run`. To run on a schedule, install the included [crontab](./crontab) which fires the daily session weekday afternoons.

There's also a paper-trading overlay if you want to test changes against the paper account without disturbing the prod stack:

```bash
task paper:session       # uses .env.paper, separate Postgres volume, port 3001 dashboard
```

The full task list is in [Taskfile.yml](./Taskfile.yml).

## What's not in here

- **No backtesting framework.** This is a forward-test only. The point is to learn from real fills, not from a curve-fit replay of historical data.
- **No options, no shorts, no crypto.** Liquid US equities, long-only. Adding instruments before proving an edge on the simple case is yak-shaving.
- **No machine learning model training.** The "learning" is the LLM reflection loop. No PyTorch, no scikit-learn, no fine-tuning — the whole point is to see how far an off-the-shelf agentic loop can go.

## Repo layout

- `v2/` — active codebase
- `trading/`, `dashboard/` — legacy v1 (sunset, partial pieces reused)
- `tests/` — ~1,965 tests across v1 and v2, all external deps mocked
- `db/init/` — schema (one file per table); `db/migrations/` — incremental changes
- `public_dashboard/` — static-site source for the public dashboard
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — design docs and implementation plans, kept as a record of how the project evolved

## License

MIT — see [LICENSE](./LICENSE).
