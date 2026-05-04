---
name: AI-audience methodology — /about/ page
date: 2026-05-04
status: draft
parent: 2026-05-03-ai-audience-pages-design.md
depends_on: 2026-05-03-dashboard-permalinks-design.md
---

# /about/ methodology page

Sub-spec 4a of the AI-audience methodology phase. Smallest of the three; ships first. Stand-alone "how it works" page that posts and the dashboard nav can link to.

## Goal

A single new page at `/about/index.html` published by Stage 6. 80% hand-authored Markdown rendered to HTML, 20% auto-populated stats injected per publish so the page can never silently drift from the live system.

## Non-goals

- No prompt source-of-truth viewer.
- No comparison framing against other systems.
- No new ingestion or telemetry. Pull only what the codebase already exposes.

## Architecture

### Content source

Hand-authored Markdown lives at `public_dashboard/about.md`. Sections (proposed; final wording in implementation):

- **What this is** — one paragraph.
- **The daily loop** — the 7 session stages, brief per-stage descriptions.
- **Models** — which Claude model handles which stage. Auto-injected from `agent.DEFAULT_EXECUTOR_MODEL` and `strategy.DEFAULT_REFLECTION_MODEL` so the page tracks code.
- **Data** — what gets ingested (Alpaca news, market data, account state) and what doesn't.
- **Honesty** — limitations, known issues, what the dashboard does and doesn't reflect.
- **Code** — link to the GitHub repo.

### Markdown rendering — open question

Two options for the implementation plan to choose between:

1. **`markdown-it-py`** — battle-tested dependency. Adds one wheel. Renders any Markdown the author writes without surprises.
2. **Hand-rolled subset renderer** — ~150 lines covering h1/h2/h3, paragraphs, bullet lists, code spans, links. No dependency. Author has to stay inside the subset.

Recommend Option 1 (`markdown-it-py`) unless the dependency is objected to. The /about/ page wants to read like a real article, not a constrained subset.

### Auto-injected fields

Use the same `string.Template` substitution pattern as Spec #1's other pages. Substitution dict gathered in `gather_dashboard_data` and consumed by the renderer:

```python
{
    "executor_model": agent.DEFAULT_EXECUTOR_MODEL,
    "reflection_model": strategy.DEFAULT_REFLECTION_MODEL,
    "stage_count": 7,
    "publish_date": today.isoformat(),
}
```

If the substitution dict and the Markdown disagree (e.g. Markdown mentions a model that's no longer the default), the auto-injected value wins.

### Changes summary

| File | Change |
|---|---|
| `public_dashboard/about.md` | NEW (hand-authored content) |
| `v2/dashboard_pages.py` | Add `render_about_page(markdown_source, substitutions)` |
| `v2/dashboard_publish.py` | Gather substitutions; emit `<deploy>/about/index.html` |
| `requirements.txt` (or equivalent) | Add `markdown-it-py` if Option 1 chosen |

### Data flow

```
Stage 6 (daily):
  └─ run_dashboard_stage
      ├─ ... existing flow ...
      └─ load about.md → render_about_page → /about/index.html
```

### Error handling

- Missing `about.md` raises at publish time — fail loudly, don't emit a placeholder. The page is hand-authored content; its absence is a bug.
- Substitution lookup failures (key in template, not in dict) raise the same way.
- Markdown library failures fall back to wrapping the raw text in `<pre>` and emitting the page anyway, with a logged warning. Better to publish ugly than to break the publish pipeline.

## Testing

- `tests/test_dashboard_pages.py`: `render_about_page` returns expected sections; auto-injected fields appear in output; missing template key raises.
- `tests/test_dashboard_publish.py`: publish writes `<deploy>/about/index.html`; substitutions sourced correctly.

## Open questions

- Markdown lib vs hand-rolled (above).
- Whether the /about/ page lands in the dashboard nav header or only via direct link from posts. Default: nav header link, since this is conversion surface.
