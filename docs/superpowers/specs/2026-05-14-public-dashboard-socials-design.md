# Public Dashboard — Social Follow Links

**Date:** 2026-05-14
**Status:** Approved, ready for implementation plan.

## Goal

Add Twitter/X and Bluesky follow links to the public dashboard so visitors
can find the bot's social accounts from any page.

## Scope

A single change: two icon links in the top-right of the site nav, on every
page emitted by `v2/dashboard_publish.py`.

**Out of scope** (explicitly declined during brainstorming):

- A `/socials/` page or new nav entry.
- A posts feed surfaced from the `tweets` audit table.
- Per-trade or per-thesis embeds of social posts.
- A "latest posts" strip on the homepage.

## Design

### Markup — `v2/dashboard_pages.py`

In `_render_nav()` (currently around `v2/dashboard_pages.py:48`), append a
new `.socials` cluster after the `.links` row but before the closing
`<div></div></nav>`. Two anchors:

| Platform | URL                                                      | `aria-label`         |
|----------|----------------------------------------------------------|----------------------|
| X        | `https://x.com/bbottomcapital`                           | `Follow on X`        |
| Bluesky  | `https://bsky.app/profile/bbottomcapital.bsky.social`    | `Follow on Bluesky`  |

Each link:

- `target="_blank" rel="noopener noreferrer"`.
- Contains an inline SVG icon (`fill="currentColor"`, sized via CSS, no
  external requests, no asset files to deploy).
- X icon: the stylized X glyph.
- Bluesky icon: the Bluesky butterfly.

Handles are hardcoded in the template. They are not deploy-environment
specific, and threading config in for two strings is over-engineering.

### Styles — `public_dashboard/styles.css`

Add a small block near the existing `.site-nav` rules:

```css
.site-nav .socials {
  display: flex; align-items: center; gap: 0.6rem; margin-left: 1.2rem;
}
.site-nav .socials a {
  color: var(--text-dim); display: inline-flex; align-items: center;
}
.site-nav .socials a:hover { color: var(--text); }
.site-nav .socials svg { width: 18px; height: 18px; display: block; }
```

The existing `@media (max-width: 640px)` block lets `.site-nav .container`
flex-wrap, so the `.socials` cluster will naturally flow onto the same
row as `.links` (or wrap below) on narrow viewports. No mobile-specific
rules are required beyond removing `margin-left` if it ends up awkward —
verify in the browser before declaring done.

### Testing

One unit test in the existing dashboard rendering test module:

- Render any page (e.g. via `render_homepage(...)` with minimal fixtures
  or by calling `_render_nav` directly if it's accessible from the
  test seam).
- Assert the rendered HTML contains both URLs and both `aria-label`
  strings.

No new fixtures or DB rows required.

### Verification

Before claiming done:

- Run the dashboard test suite.
- Run `python -m v2.dashboard_publish` (or the relevant build entry
  point) against a local DB and open the produced static HTML in a
  browser to confirm icons render at expected size, hover-state works,
  and mobile (≤640px) layout isn't broken.

## Risks

- **None material.** Static HTML/CSS change to the public deploy.
  Worst-case rollback is reverting one commit.
- The hardcoded X handle (`@bbottomcapital`) was provided by the user
  during brainstorming; if the account name changes later we update one
  string in `_render_nav`.
