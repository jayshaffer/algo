# Strategist Cache-Thrashing Fix

**Date:** 2026-05-08
**Status:** Design — pending implementation plan
**Owner:** jay

## Problem

The Claude strategist costs ~$5 per paper/prod run on Opus 4.6 (observed:
$4.54 and $5.82 across two captured sessions). ~93% of that cost is
input-side, dominated by `cache_creation_tokens`. The cache-creation /
cache-read ratio is ~35%, which is high for a single in-session agentic
loop — once content has been seen, it should be cache-read, not
cache-created.

Root cause: `_truncate_old_tool_results` in `v2/claude_client.py` modifies
the message history mid-conversation. Each turn it walks the *original*
`messages` and truncates any tool-result older than the recent-3 window.
Because each new turn pushes one more message past that boundary, a
*different* message is freshly truncated each turn — which changes the
prefix Anthropic's prompt cache uses as its key. The cache match drops
back to the position of the newly truncated message, and everything from
that point forward is re-billed as `cache_creation` instead of `cache_read`.

Estimated savings from fixing this: ~$2–3 per strategist run, ~$500–750/year
at one run/trading day.

## Goal

Stabilize the message prefix across turns of `run_agentic_loop` so the
prompt cache stops thrashing. Specifically: drop strategist
`cache_creation_tokens` from ~220K/session (avg of two observed runs) to
~80K/session (one-shot upfront cost), with `cache_read_tokens` rising
modestly to compensate.

## Non-Goals

- Model swap (e.g., Sonnet 4.6).
- Trimming `_build_pre_seeded_context` size.
- Changes to tool definitions or system prompt.
- Cache breakpoint placement in `_messages_with_cache_breakpoint`. The
  moving breakpoint on the latest user message is the correct in-session
  pattern; it was investigated and ruled out as the bug.

## Approach

Two phases land in sequence. Both ship in one branch, but Phase 1 is
self-contained and gives us the data to finalize Phase 2's threshold.

### Phase 1 — Measurement

Add `output_chars` to the existing `tool_invocation` agent_event payload
in `v2/claude_client.py` (~line 498). One-line addition; permanent
telemetry.

```python
record_event(
    session_id=session_id,
    stage_name=stage_name or "unknown",
    event_type="tool_invocation",
    payload={
        "tool_name": tool_name,
        "args": tool_input if isinstance(tool_input, dict) else {"_raw": str(tool_input)},
        "success": not result.is_error,
        "error": (result.content if result.is_error else None),
        "duration_ms": duration_ms,
        "output_chars": len(result.content),  # NEW
    },
)
```

Run one paper strategist session to populate data:

```bash
task paper:session
# or: docker compose -f docker-compose.yml -f docker-compose.paper.yml exec trading-paper python -m v2.session --stage strategist
```

Query the per-tool size distribution:

```sql
SELECT
  payload->>'tool_name' AS tool,
  COUNT(*) AS calls,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY (payload->>'output_chars')::int) AS p50,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY (payload->>'output_chars')::int) AS p90,
  MAX((payload->>'output_chars')::int) AS max
FROM agent_events
WHERE event_type = 'tool_invocation' AND stage_name = 'strategist'
GROUP BY 1 ORDER BY p90 DESC;
```

Record the resulting numbers in this spec (TBD section below) before
Phase 2 lands.

### Phase 2 — Cache fix

Single file changed: `v2/claude_client.py`.

**Delete:**

- `_truncate_old_tool_results` function (lines ~281–326)
- Constants `_TRUNCATION_THRESHOLD = 300` and `_KEEP_RECENT_EXCHANGES = 3`
- The `pruned = _truncate_old_tool_results(messages)` call in
  `run_agentic_loop` (~line 363); replace with `pruned = messages`

**Add:**

- New constant `_TOOL_RESULT_TRUNCATION_THRESHOLD = <TBD from Phase 1>`
- New helper `_truncate_tool_result_blocks(content_blocks, threshold)` —
  pure function, takes a list of content blocks (the user message's
  `content`), returns a copy with any oversized `tool_result` blocks
  clipped to `threshold` chars + `"...[truncated]"`. No
  message-history awareness; does one block list at a time.

**Modify:**

In `run_agentic_loop`, at the point where `tool_results` are appended
back into `messages` (~line 540), wrap the list through
`_truncate_tool_result_blocks` before appending. The truncation is now
*frozen* at the moment of addition — a tool-result's representation in
`messages` never changes once added.

**Unchanged:**

- `_messages_with_cache_breakpoint` — breakpoint placement is correct.
- `_aggressive_prune` — still the safety net for genuine
  context-length-exceeded errors.
- System prompt and tools cache_control markers.

### Threshold value (TBD pending Phase 1)

Heuristic: pick a value at or slightly above the **p90 of the heaviest
tools** observed in Phase 1 — high enough that small tools never get
truncated and the model retains useful detail from older calls, low
enough that a worst-case session of ~15 turns with all-large tool
results stays well under the 200K context window.

Provisional placeholder before measurement: `2000` chars. Final value
recorded here after Phase 1 run:

```
TBD — fill in after Phase 1 paper session
- p90 of get_news_signals: ____
- p90 of web_search: ____
- p90 of get_market_snapshot: ____
- chosen threshold: ____
```

## Architecture / Data Flow

**Before (current):**

```
each turn:
  build messages = [u0, a1, u1_full, ..., aN, uN]
  _truncate_old_tool_results(messages):
    keep last 3 user-tool-result indices intact
    clip everything older to 300 chars + "...[truncated]"
  → returns NEW list with newly-clipped older messages
  → cache prefix matches up to first newly-clipped position only
```

**After (Phase 2):**

```
each turn:
  handlers run, return tool_results: list[ToolResult]
  _truncate_tool_result_blocks(tool_results, _TOOL_RESULT_TRUNCATION_THRESHOLD):
    clip oversized blocks ONCE, here, before they enter messages
  messages.append({"role": "user", "content": <truncated tool_results>})
  → messages list is byte-stable from this point on
  → cache prefix matches all previously-seen turns; only the newest
    user+assistant content is cache_created
```

## Risks

1. **Threshold too tight → playbook quality regression.** The strategist
   may rely on full text from older `get_news_signals` or `web_search`
   calls when writing the playbook several turns later. Symptom: lower
   playbook quality post-fix (fewer/weaker actions, less specific
   reasoning). Mitigation: Phase 1 chooses threshold from observed
   distribution at p90, not p50; cache fix is correct at any threshold,
   so if quality regresses we raise the value without re-architecting.

2. **Context overflow on a long session.** Without proactive
   truncation, a pathological session with many large tool calls could
   approach the 200K context window. Mitigation: `_aggressive_prune`
   already exists for this case (one-time recovery per loop). If it
   starts firing on real runs, that's the signal to lower the threshold.

3. **Single-sample bias in Phase 1.** One paper run may not show the
   worst-case tool sizes. Mitigation: take p90 across the run, build in
   buffer, plan to revisit threshold after a week of prod runs.

4. **Telemetry overhead.** `output_chars` adds ~10 bytes per
   `tool_invocation` event. Negligible.

## Validation

Compare two paper strategist sessions.

**Pre-fix baseline:** already on file in `session_stages`:
- session 3251 (2026-05-06): cache_create=190,843 cache_read=355,192 cost=$4.54
- session 3405 (2026-05-07): cache_create=251,043 cache_read=463,908 cost=$5.82
- average baseline: cache_create ≈ 220K, cost ≈ $5.18

**Post-fix run:** same pipeline, fresh paper strategist session after
Phase 2 lands.

**Success criterion:** post-fix `cost_usd` for the strategist stage is
≥40% lower than pre-fix average — i.e., **≤ $3.10**. If reduction is
under 40%, investigate before declaring success (likely indicates the
cache invalidation has another driver beyond truncation).

**Quality sanity check:** post-fix playbook must contain ≥1 action (the
existing `get_playbook(session_date) is None` check in `session.py:295`
already enforces this). Manual eyeball of one post-fix playbook for
obvious degradation: do thesis updates and rationale read coherently?
Are decisions consistent with the active theses? Compare against
~3 prior playbooks for tone/depth.

## Testing

- **Phase 1:** no new unit tests; the `output_chars` field is observable
  via SQL on `agent_events`.
- **Phase 2:**
  - Unit test for `_truncate_tool_result_blocks`: small block passes
    through unchanged; oversized block is clipped to threshold +
    `"...[truncated]"`; non-tool-result blocks pass through; empty list
    returns empty list.
  - Unit test that `messages` passed to a second turn is byte-identical
    to the first turn's content for the previously-seen prefix.
    (Construct a 5-turn message history; truncate-on-add; assert that
    after appending turn 6's tool result, turns 1–5 in the list are
    untouched.)
  - Existing `_aggressive_prune` and `_messages_with_cache_breakpoint`
    tests stay; this change does not affect their behavior.

## Out of scope (revisit later)

- **Per-tool thresholds.** Currently a single global threshold. If
  Phase 1 shows tools with very different size profiles where the
  global value cuts off small ones too aggressively, switch to a dict
  of per-tool thresholds. Add complexity only if data justifies it.
- **Sonnet swap.** Cheaper Opus is more attractive than risky Sonnet,
  per earlier discussion. Revisit only if the cache fix doesn't bring
  cost low enough.
- **Pre-seeded context shrink.** `_build_pre_seeded_context` is the
  cache_creation floor on turn 1. Trimming it is a separate optimization
  with its own quality risk. Defer until Phase 2 results are in.
