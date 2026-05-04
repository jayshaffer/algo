---
name: AI-audience methodology pages — umbrella
date: 2026-05-03
status: superseded-by-children
parent: 2026-05-03-audience-growth-overview.md
depends_on: 2026-05-03-dashboard-permalinks-design.md
---

# AI-audience methodology pages

Phase 3 of the audience-growth strategy. Adds three standing pages on the public dashboard so the AI / agentic-systems-builder audience has somewhere to land: a methodology page, a model-and-cost transparency page, and a sample tool-call trace viewer. Posts reference these; visitors find more reasons to follow.

Depends on Spec #1 for the rendering pipeline. No dependency on Specs #2 or #3.

## Decomposed into three sub-specs

Original draft (2026-05-03) bundled all three pages. They have very different shapes — `/about/` is hand-authored Markdown with no schema work; `/internals/` adds a telemetry table and instruments every Claude API call; `/trace/` adds a redaction surface that can leak strategist tool calls if it regresses. Decomposed 2026-05-04 to keep blast radius and review focus tight per page.

| # | Sub-spec | Risk | Ship order |
|---|---|---|---|
| 4a | [/about/ methodology page](2026-05-04-about-page-design.md) | Low | First |
| 4b | [/internals/ model & cost transparency](2026-05-04-internals-page-design.md) | Medium (instruments hot path) | Second |
| 4c | [/trace/ tool-call viewer + redaction](2026-05-04-trace-page-design.md) | High (redaction = leak surface) | Third |

Each sub-spec is independent and can ship in any order — recommended order above is by risk and value-per-task. 4b and 4c both add a new DB migration; numbering assumes 4b ships first (`004_model_usage.sql`, then `005_strategist_traces.sql`). If they swap, renumber accordingly.

## Cross-cutting context

These items appeared in the original draft and apply to all three sub-specs:

- **Initial state.** `model_usage` and `strategist_traces` start empty. The first publish after each sub-spec lands shows sparse data; placeholder copy covers it. Acceptable.
- **File-count ceiling.** Spec #1's per-trade and per-thesis pages already share the 20,000-file Cloudflare Pages limit. 4a adds 1 page; 4b adds 1; 4c adds 1 per eligible session (~250/year). Single shared headroom check before 4c ships — count current emitted pages, project growth rate.
- **Voice.** /about/ and /internals/ stay grounded in what the system actually does. No comparison-to-other-bots framing.

## Out of scope (deferred)

- Prompt source-of-truth viewer. Prompts evolve fast; pinning them invites stale-content issues.
- Real-time / live trace viewer. Static, per-published-session.
- New ingestion or telemetry beyond what's strictly needed for these pages.
