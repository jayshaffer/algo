# Theses Dashboard Section Design

## Overview

Add a dedicated Theses page to the dashboard that displays thesis analytics and detailed thesis cards. This surfaces the ideation system's trade conviction ideas for visibility and tracking.

## Analytics Summary

Top of page displays 6 metric cards:

| Metric | Description | Query |
|--------|-------------|-------|
| Active Theses | Count where status='active' | `COUNT(*) WHERE status='active'` |
| Executed | Count where status='executed' | `COUNT(*) WHERE status='executed'` |
| Invalidated | Count where status='invalidated' | `COUNT(*) WHERE status='invalidated'` |
| Expired | Count where status='expired' | `COUNT(*) WHERE status='expired'` |
| Success Rate | % of executed theses that were profitable | Join theses → decisions → outcomes |
| Avg Confidence | Distribution of active thesis confidence | `COUNT(*) GROUP BY confidence WHERE status='active'` |

If no decisions are linked to theses yet, Success Rate shows "N/A".

## Thesis Cards

Grid layout (2 columns desktop, 1 column mobile). Each card contains:

**Header:**
- Ticker name
- Direction badge: Long (green), Short (red), Avoid (gray)
- Confidence badge: High/Medium/Low with color coding
- Status badge: Active (blue), Executed (green), Invalidated (red), Expired (gray)

**Body:**
- **Thesis** - Core reasoning (truncatable with expand)
- **Entry Trigger** - Conditions that trigger entry
- **Exit Trigger** - When to close position
- **Invalidation** - What proves thesis wrong

**Footer:**
- Created date
- Last updated date
- Source badge: ideation / manual

## Filtering & Sorting

- **Filter dropdown:** All / Active / Executed / Invalidated / Expired
- **Sort options:** Newest / Oldest / Confidence / Ticker
- **Default:** Active only, sorted by newest

## Implementation

### Files to Create/Modify

**New:**
- `dashboard/templates/theses.html` - Page template

**Modify:**
- `dashboard/app.py` - Add `/theses` route
- `dashboard/queries.py` - Add thesis query functions
- `dashboard/templates/base.html` - Add nav link (if separate base exists) or update nav in templates

### Route

```python
@app.route('/theses')
def theses():
    status_filter = request.args.get('status', 'active')
    sort_by = request.args.get('sort', 'newest')

    stats = queries.get_thesis_stats()
    theses = queries.get_theses(status_filter, sort_by)

    return render_template('theses.html', stats=stats, theses=theses,
                          current_filter=status_filter, current_sort=sort_by)
```

### Query Functions

```python
def get_thesis_stats():
    """Return dict with counts by status and success rate."""

def get_theses(status_filter='active', sort_by='newest'):
    """Return filtered/sorted thesis list."""
```

### Navigation

Add "Theses" link to navbar between Signals and Decisions in all templates.

## Dependencies

None - uses existing Flask, Jinja2, Tailwind stack.
