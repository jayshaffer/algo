---
name: Audience growth strategy — overview
date: 2026-05-03
status: draft
---

# Audience growth strategy — overview

Umbrella doc that captures the strategy decisions made during brainstorming and points at the four child specs that implement them.

## Goal

Grow the public following of Bikini Bottom Capital across two audiences simultaneously:

1. **FinTwit / retail traders** — already in $TICKER chatter; want trades, P&L, hot takes.
2. **AI / agentic-systems builders** — interested in *how* the system reasons; want prompts, attribution, honest failures.

Audience growth, not pure transparency or operator usefulness, is the optimization target. Engagement, frequency, and shareability win over polish.

## Constraints accepted

- **Broadcast only** — no replies, no quote-tweets, no reading mentions. Keeps us on Twitter free tier (1,500 writes/month) and removes the bad-reply risk surface.
- **Same content, both platforms** — Twitter is the dominant water; Bluesky mirrors. No platform-specific content strategy in this round.
- **Voice continuity** — keep the "Bikini Bottom Capital" identity (currently Mr. Krabs–adjacent). Voice may evolve per post type but the brand stays.
- **Once-daily session** — trades happen post-close; "live trade" posts are clustered around session-end, not intra-day Alpaca polling. Pre-market post is a separate scheduled trigger.

## Cross-section content rotation (locked)

The High/High cell of the audience matrix is "Why I bought X" — per-trade thesis posts that lead FinTwit-style with the buy/sell and follow with the agent's reasoning.

Daily:
- 1 pre-market take (anchor; persistent theses + open positions)
- N live-trade posts after the session (typically 1–3, capped at 5)
- 1 midday entertainment post (existing pipeline; unchanged)

Weekly:
- 1 "what I got wrong" post (Sunday or Monday)
- 1 attribution roundup post (image-first; chart screenshot)

Cut from the rotation: open call-outs (front-running optics), the current bare post-close recap (folded into live-trade posts), system-internals long-form on Twitter (Bluesky-only if at all), per-event drawdown alerts (let them emerge naturally).

Total: ~10–15 posts/week per platform. Well under free-tier limits.

## Dashboard role

The dashboard stops being "operator status page that's also public." Two new jobs:

1. **Receipt for every post.** Each tweet links to a specific permalink: trade, thesis, mistakes log, attribution chart.
2. **Conversion page.** Visitor lands on the receipt, sees enough other interesting stuff to follow.

Architectural fork chosen: **pre-render per-page HTML at publish time** (Stage 6 emits `/trade/<id>/`, `/thesis/<id>/`, `/about/`, etc., with full OG / Twitter card meta tags and dynamic OG images). Keeps the SPA shell for the homepage; layers static pages around it. No JS framework introduced.

## Phasing & child specs

Four specs, intended to ship in dependency order. Each gets its own implementation plan after this brainstorm cycle.

| # | Spec | Depends on | Unlocks |
|---|---|---|---|
| 1 | [Dashboard permalinks & OG infrastructure](2026-05-03-dashboard-permalinks-design.md) | — | Per-trade and per-thesis pages, OG images, Twitter cards |
| 2 | [Live-trade social pipeline](2026-05-03-live-trade-pipeline-design.md) | #1 | Per-fill posts + pre-market anchor; replaces current bare recap |
| 3 | [Mistakes log & attribution panel](2026-05-03-mistakes-attribution-design.md) | #1 | Weekly mistakes post, weekly attribution roundup post |
| 4 | [AI-audience methodology pages](2026-05-03-ai-audience-pages-design.md) | #1 | `/about`, model/cost transparency, sample tool-call trace |

Spec #1 is load-bearing for everything else; specs #2–4 can be re-ordered freely after it lands.

## Out of scope (deferred)

- Reply / engagement loops (paid Twitter API tier)
- Real-time intra-day posting (would require Alpaca polling or webhooks)
- Bluesky-specific content variants (long-form, image rich, etc.)
- Email / RSS / Discord distribution
- Per-platform OG image variants
- Comments or reactions on the dashboard
- Migration to a JS-based static site generator (Astro, Eleventy)
