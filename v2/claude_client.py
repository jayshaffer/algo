"""Claude API client with tool handling and agentic loop."""

import contextlib
import contextvars
import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic

from .database.trading_db import insert_llm_call_context
from .pricing import stage_cost_usd
from .telemetry import record_event

logger = logging.getLogger(__name__)

# Runaway-cost guard for agentic loops. Cumulative token cost (priced via
# the model_pricing table) is checked after every turn; crossing the ceiling
# aborts the loop with stop_reason="cost_ceiling". Read at module import —
# container restart required after changing. Set <= 0 to disable.
LOOP_COST_CEILING_USD = float(os.environ.get("ALGO_LOOP_COST_CEILING_USD", "30"))

# Retry configuration for Claude API calls
API_MAX_RETRIES = 3
API_RATE_LIMIT_DELAY = 60  # seconds; rate limits reset per minute
API_RETRY_BASE_DELAY = 2  # seconds; for non-rate-limit errors
API_RETRY_BACKOFF_FACTOR = 2
API_RETRY_JITTER_MAX = 1.0  # seconds

RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


class AgentPurpose:
    """Canonical `purpose` values for `agent_call` telemetry. Use these
    constants at call sites to keep auditor SQL filter values stable."""
    EXECUTOR = "executor"
    CLASSIFIER_NEWS = "classifier_news"
    CLASSIFIER_MACRO = "classifier_macro"
    CLASSIFIER_RELEVANCE = "classifier_relevance"
    STRATEGIST_LOOP = "strategist_loop"
    REFLECTION_LOOP = "reflection_loop"


_CONTEXT_LOGGED_PURPOSES = frozenset({
    AgentPurpose.EXECUTOR,
    AgentPurpose.STRATEGIST_LOOP,
    AgentPurpose.REFLECTION_LOOP,
})


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class AgenticLoopResult:
    """Result from running the agentic loop."""

    messages: list[dict]
    turns_used: int
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class UsageAccumulator:
    """Sums Claude API token usage across calls within a stage.

    Populated by `_record_usage` (called from `_call_with_retry`) when
    a `capture_usage()` block is active; consumed by session.py to
    write per-stage token counts to the database.
    """

    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    mixed_models: bool = False

    def add(self, model: str, usage) -> None:
        if self.model is None:
            self.model = model
        elif self.model != model:
            self.mixed_models = True
        self.input_tokens          += (usage.input_tokens or 0)
        self.output_tokens         += (usage.output_tokens or 0)
        self.cache_creation_tokens += (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self.cache_read_tokens     += (getattr(usage, "cache_read_input_tokens", 0) or 0)


_current_usage: contextvars.ContextVar[UsageAccumulator | None] = contextvars.ContextVar(
    "_current_usage", default=None
)


@contextlib.contextmanager
def capture_usage():
    """Open an accumulator that records all Claude calls until the block exits.

    Stage code is unaffected — instrumentation lives at `_call_with_retry`.
    Sessions wrap each stage in this block to collect per-stage usage.
    """
    acc = UsageAccumulator()
    token = _current_usage.set(acc)
    try:
        yield acc
    finally:
        _current_usage.reset(token)


def _record_usage(model: str, usage) -> None:
    """Record one API call's usage into the active accumulator (if any).

    No-op when called outside a capture_usage() block — the function is
    safe to call from any code path; production callers don't need to
    know whether tracking is on.
    """
    acc = _current_usage.get()
    if acc is not None:
        acc.add(model, usage)


def get_claude_client() -> anthropic.Anthropic:
    """Create Claude client. Requires ANTHROPIC_API_KEY environment variable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Get your key at https://console.anthropic.com/"
        )

    return anthropic.Anthropic(api_key=api_key, max_retries=5)


def _serialize_content_blocks(content) -> list:
    """Convert anthropic content blocks to JSON-serializable dicts.

    Real anthropic SDK blocks are pydantic models exposing `.model_dump()`.
    Test fixtures use SimpleNamespace (handled via `vars()`). User-message
    content already arrives as plain dicts and passes through unchanged.
    """
    result = []
    for block in content or []:
        if isinstance(block, dict):
            result.append(block)
        elif hasattr(block, "model_dump"):
            result.append(block.model_dump())
        else:
            result.append({k: v for k, v in vars(block).items()})
    return result


def _record_call_context(
    *,
    session_id,
    stage_name,
    purpose,
    create_kwargs: dict,
    message,
    duration_ms: int,
) -> None:
    """Persist the LLM round-trip for forensic replay.

    Gated on (a) a real session_id and (b) a captured purpose. Errors
    are logged and swallowed — context logging must never break a
    session.
    """
    if session_id is None:
        return
    if purpose not in _CONTEXT_LOGGED_PURPOSES:
        return
    try:
        usage = getattr(message, "usage", None) if message is not None else None
        response_content = (
            _serialize_content_blocks(message.content)
            if message is not None and getattr(message, "content", None) is not None
            else None
        )
        raw_system = create_kwargs.get("system")
        if isinstance(raw_system, str):
            system_prompt = raw_system
        elif isinstance(raw_system, list):
            # run_agentic_loop wraps system in a cache-control list-of-blocks
            system_prompt = "\n".join(
                b.get("text", "") for b in raw_system if isinstance(b, dict) and b.get("type") == "text"
            ) or None
        else:
            system_prompt = None
        # Normalize the messages list for JSON storage: prior assistant turns
        # carry raw SDK pydantic blocks (we keep them native so the API
        # accepts them on the next turn). psycopg2's Json() adapter can't
        # serialize those — dump them here at the recording boundary.
        recorded_messages = [
            {
                "role": m["role"],
                "content": (
                    _serialize_content_blocks(m["content"])
                    if isinstance(m.get("content"), list)
                    else m.get("content")
                ),
            }
            for m in (create_kwargs.get("messages") or [])
        ]
        insert_llm_call_context(
            session_id=session_id,
            stage_name=stage_name or "unknown",
            purpose=purpose,
            model=create_kwargs.get("model"),
            system_prompt=system_prompt,
            messages=recorded_messages,
            tool_definitions=create_kwargs.get("tools"),
            response_content=response_content,
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) if usage else None,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None) if usage else None,
            stop_reason=getattr(message, "stop_reason", None) if message else None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        logger.warning(
            "Failed to record llm_call_context (session=%s stage=%s purpose=%s): %s",
            session_id, stage_name, purpose, e,
        )


def _call_with_retry(client, max_retries=API_MAX_RETRIES, **create_kwargs):
    """Call client.messages.stream() with retry and exponential backoff.

    Streams to completion and returns the assembled Message — same shape
    as a non-streaming response. Streaming is required because the SDK
    refuses non-streaming requests whose max_tokens × estimated
    time-per-token exceeds 10 minutes; with max_tokens=32000 (Opus 4.x
    model max) the heuristic trips on the first turn. See:
    https://github.com/anthropics/anthropic-sdk-python#long-requests

    Telemetry kwargs (`session_id`, `stage_name`, `purpose`) are popped
    out of `create_kwargs` before forwarding to the SDK. Retries collapse
    into a single `agent_call` event with the final outcome.
    """
    session_id = create_kwargs.pop("session_id", None)
    stage_name = create_kwargs.pop("stage_name", None)
    purpose = create_kwargs.pop("purpose", None)

    started = time.monotonic()
    success = False
    error_msg: str | None = None
    message = None
    try:
        for attempt in range(max_retries + 1):
            try:
                with client.messages.stream(**create_kwargs) as stream:
                    message = stream.get_final_message()
                _record_usage(create_kwargs["model"], message.usage)
                success = True
                return message
            except RETRYABLE_ERRORS as e:
                if attempt == max_retries:
                    logger.error(
                        "API call failed after %d retries: %s", max_retries, e
                    )
                    error_msg = f"{type(e).__name__}: {str(e)[:500]}"
                    raise
                if isinstance(e, anthropic.RateLimitError):
                    delay = API_RATE_LIMIT_DELAY + random.uniform(0, API_RETRY_JITTER_MAX)
                else:
                    delay = (
                        API_RETRY_BASE_DELAY * (API_RETRY_BACKOFF_FACTOR ** attempt)
                        + random.uniform(0, API_RETRY_JITTER_MAX)
                    )
                logger.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    e,
                    delay,
                )
                time.sleep(delay)
    except Exception as e:
        if error_msg is None:
            error_msg = f"{type(e).__name__}: {str(e)[:500]}"
        raise
    finally:
        duration_ms = int((time.monotonic() - started) * 1000)
        payload = {
            "model": create_kwargs.get("model"),
            "purpose": purpose or "unknown",
            "duration_ms": duration_ms,
            "success": success,
            "error": error_msg,
        }
        if message is not None:
            payload["stop_reason"] = message.stop_reason
            payload["input_tokens"] = getattr(message.usage, "input_tokens", 0) or 0
            payload["output_tokens"] = getattr(message.usage, "output_tokens", 0) or 0
            payload["cache_creation_tokens"] = (
                getattr(message.usage, "cache_creation_input_tokens", 0) or 0
            )
            payload["cache_read_tokens"] = (
                getattr(message.usage, "cache_read_input_tokens", 0) or 0
            )
        record_event(
            session_id=session_id,
            stage_name=stage_name or "unknown",
            event_type="agent_call",
            payload=payload,
        )
        _record_call_context(
            session_id=session_id,
            stage_name=stage_name,
            purpose=purpose,
            create_kwargs=create_kwargs,
            message=message,
            duration_ms=duration_ms,
        )


def _messages_with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Return a shallow copy of messages with cache_control on the last user message."""
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if result[i]["role"] == "user":
            content = result[i]["content"]
            if isinstance(content, str):
                result[i] = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                    ],
                }
            elif isinstance(content, list) and content:
                new_content = list(content)
                new_content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
                result[i] = {"role": "user", "content": new_content}
            break
    return result


_TRUNCATION_THRESHOLD = 300  # chars; tool results longer than this get truncated
_KEEP_RECENT_EXCHANGES = 3  # number of recent assistant+user pairs to keep intact

_CONTEXT_LENGTH_MARKERS = ("too long", "context length", "context_length")


def _looks_like_context_length_error(exc: Exception) -> bool:
    """Best-effort detection of Anthropic context-length-exceeded errors.
    The SDK surfaces these as BadRequestError with a message like
    'prompt is too long: 250000 tokens > 200000 maximum'."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _CONTEXT_LENGTH_MARKERS)


def _aggressive_prune(messages: list[dict]) -> list[dict]:
    """Drop everything except the initial user prompt and the most recent
    exchange. Used as a last-ditch recovery when the API rejects a request
    for being over the context window — `_truncate_old_tool_results` only
    shortens individual tool_results, which isn't enough when the chat is
    long enough on its own."""
    if len(messages) <= 4:
        return messages
    # First message is the initial user prompt; keep last 4 messages so we
    # preserve a complete (assistant tool_use → user tool_result) exchange
    # plus its reply pair.
    pruned = [messages[0], *messages[-4:]]
    # Defense in depth: slicing can collapse `user → user` adjacency between
    # the prepended initial prompt and the start of the tail (the tail starts
    # with `user` whenever the loop's invariant gets perturbed by an
    # upstream bug). The Anthropic API rejects same-role adjacency, so drop
    # the offending head-of-tail entry to restore alternation. Walking the
    # whole list keeps us safe against any future shape drift too.
    deduped: list[dict] = [pruned[0]]
    for msg in pruned[1:]:
        if msg["role"] == deduped[-1]["role"]:
            continue
        deduped.append(msg)
    return deduped


def _truncate_old_tool_results(messages: list[dict]) -> list[dict]:
    """Truncate tool results from older exchanges to reduce context growth.

    Keeps the most recent _KEEP_RECENT_EXCHANGES pairs intact.
    For older user messages containing tool_result blocks, truncates
    the content string to _TRUNCATION_THRESHOLD characters.
    """
    # Find cutoff: keep last N user messages with tool results untouched
    tool_result_indices = [
        i for i, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m.get("content"), list)
        and any(
            isinstance(item, dict) and item.get("type") == "tool_result"
            for item in m["content"]
        )
    ]

    if len(tool_result_indices) <= _KEEP_RECENT_EXCHANGES:
        return messages

    # Indices to truncate (all but the last N)
    truncate_set = set(tool_result_indices[:-_KEEP_RECENT_EXCHANGES])

    result = []
    for i, msg in enumerate(messages):
        if i not in truncate_set:
            result.append(msg)
            continue

        new_content = []
        for item in msg["content"]:
            if (
                isinstance(item, dict)
                and item.get("type") == "tool_result"
                and isinstance(item.get("content"), str)
                and len(item["content"]) > _TRUNCATION_THRESHOLD
            ):
                new_content.append({
                    **item,
                    "content": item["content"][:_TRUNCATION_THRESHOLD] + "...[truncated]",
                })
            else:
                new_content.append(item)
        result.append({**msg, "content": new_content})

    return result


def run_agentic_loop(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    initial_message: str,
    tools: list[dict],
    tool_handlers: dict[str, Callable],
    max_turns: int = 20,
    session_id: int | None = None,
    stage_name: str | None = None,
    purpose: str = AgentPurpose.STRATEGIST_LOOP,
    max_cost_usd: float | None = None,
) -> AgenticLoopResult:
    """Run an agentic loop where Claude uses tools until it completes its task.

    `max_cost_usd` caps the loop's cumulative token cost (USD, priced via
    model_pricing); None uses the module default LOOP_COST_CEILING_USD and
    <= 0 disables the check. Crossing the ceiling stops the loop with
    stop_reason="cost_ceiling" — callers' output validation (e.g. the
    strategist's missing-playbook check) then fails the stage loudly.
    """
    messages = [{"role": "user", "content": initial_message}]
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation = 0
    total_cache_read = 0
    stop_reason = "max_turns"

    # Prepare system prompt and tools with cache breakpoints
    cached_system = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    cached_tools = list(tools)
    if cached_tools:
        cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

    max_tokens_recoveries = 0
    context_length_recoveries = 0
    cost_ceiling = LOOP_COST_CEILING_USD if max_cost_usd is None else max_cost_usd
    cost_unpriceable_warned = False

    for turn in range(max_turns):
        logger.info(f"Agentic loop turn {turn + 1}/{max_turns}")

        # Truncate old tool results to reduce context growth
        pruned = _truncate_old_tool_results(messages)

        # Anthropic requires max_tokens; pass the model's documented max so
        # truncation only happens on genuinely pathological generations.
        try:
            response = _call_with_retry(
                client,
                model=model,
                max_tokens=32000,
                system=cached_system,
                tools=cached_tools,
                messages=_messages_with_cache_breakpoint(pruned),
                session_id=session_id,
                stage_name=stage_name,
                purpose=purpose,
            )
        except anthropic.BadRequestError as e:
            # Graceful degradation for context-length-exceeded: the
            # per-turn `_truncate_old_tool_results` only shortens individual
            # tool_results, which can fall short of the limit if the
            # message history itself is long. Aggressively prune to the
            # initial prompt + most recent exchange and retry once.
            # Bounded to 1 recovery per loop to prevent runaway costs.
            if _looks_like_context_length_error(e) and context_length_recoveries < 1:
                context_length_recoveries += 1
                logger.warning(
                    "Context length exceeded on turn %d (%s); aggressively "
                    "pruning message history and retrying once",
                    turn + 1, e,
                )
                record_event(
                    session_id=session_id,
                    stage_name=stage_name or "unknown",
                    event_type="loop_recovery",
                    payload={
                        "reason": "context_length",
                        "turn": turn + 1,
                        "model": model,
                    },
                )
                messages = _aggressive_prune(messages)
                continue
            raise

        # Track token usage
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        total_cache_creation += getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        total_cache_read += getattr(response.usage, "cache_read_input_tokens", 0) or 0

        # Cost ceiling: abort before spending another turn once cumulative
        # cost crosses the cap. Unpriceable models (no model_pricing row,
        # DB unreachable) skip the check — cost telemetry must never crash
        # a session — but get one loud warning.
        if cost_ceiling > 0:
            try:
                loop_cost = stage_cost_usd(
                    model, total_input_tokens, total_output_tokens,
                    total_cache_creation, total_cache_read,
                )
            except Exception as e:
                loop_cost = None
                if not cost_unpriceable_warned:
                    cost_unpriceable_warned = True
                    logger.warning(
                        "Cost ceiling check disabled for this loop — could not "
                        "price model %s: %s", model, e,
                    )
            if loop_cost is not None and loop_cost >= cost_ceiling:
                stop_reason = "cost_ceiling"
                logger.error(
                    "Agentic loop aborted on turn %d: cumulative cost $%.2f "
                    "crossed the $%.2f ceiling (ALGO_LOOP_COST_CEILING_USD)",
                    turn + 1, loop_cost, cost_ceiling,
                )
                record_event(
                    session_id=session_id,
                    stage_name=stage_name or "unknown",
                    event_type="loop_abort",
                    payload={
                        "reason": "cost_ceiling",
                        "turn": turn + 1,
                        "model": model,
                        "cost_usd": round(loop_cost, 4),
                        "ceiling_usd": cost_ceiling,
                    },
                )
                messages.append({"role": "assistant", "content": response.content})
                break

        # Drop the truncated turn (incomplete tool_use blocks would break the next call) and retry once with a concision nudge.
        if response.stop_reason == "max_tokens" and max_tokens_recoveries < 1:
            max_tokens_recoveries += 1
            logger.warning(
                "max_tokens hit on turn %d; discarding truncated response and "
                "retrying with concision instruction",
                turn + 1,
            )
            record_event(
                session_id=session_id,
                stage_name=stage_name or "unknown",
                event_type="loop_recovery",
                payload={
                    "reason": "max_tokens",
                    "turn": turn + 1,
                    "model": model,
                },
            )
            # Bug fix: the prior turn already ended with a `user` message
            # (initial prompt on turn 1, tool_results on later turns).
            # Appending a second `user` here would put two user messages
            # back-to-back — the Anthropic API rejects that. Insert a
            # synthetic assistant stand-in so role alternation holds. We
            # can't reuse `response.content` because it contains the
            # truncated/partial tool_use blocks we're explicitly discarding.
            messages.append({
                "role": "assistant",
                "content": "[response truncated by output limit — discarded]",
            })
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was truncated by the output limit "
                    "and discarded. Please be more concise. If you have not yet "
                    "called `write_playbook`, call it now with succinct fields. "
                    "Otherwise, end your turn with a brief summary."
                ),
            })
            continue

        # Add assistant response to history using the SDK's native content
        # blocks — the API accepts and re-serializes them correctly. We
        # do NOT model_dump() here because newer SDK blocks (e.g.
        # ParsedTextBlock) carry fields like `parsed_output` that the API
        # rejects when echoed back. Serialization for forensic recording
        # happens at the recording boundary instead.
        messages.append({"role": "assistant", "content": response.content})

        # Check stop reason
        if response.stop_reason == "end_turn":
            logger.info("Claude completed task (end_turn)")
            stop_reason = "end_turn"
            break

        if response.stop_reason != "tool_use":
            logger.warning(f"Unexpected stop reason: {response.stop_reason}")
            stop_reason = response.stop_reason
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id

                logger.info(f"Executing tool: {tool_name}")
                logger.debug(f"Tool input: {tool_input}")

                started = time.monotonic()
                handler = tool_handlers.get(tool_name)
                if handler is None:
                    result = ToolResult(
                        tool_use_id=tool_use_id,
                        content=f"Error: Unknown tool '{tool_name}'",
                        is_error=True,
                    )
                else:
                    try:
                        output = handler(**tool_input)
                        result = ToolResult(
                            tool_use_id=tool_use_id,
                            content=str(output),
                        )
                    except Exception as e:
                        logger.exception(f"Tool {tool_name} failed")
                        result = ToolResult(
                            tool_use_id=tool_use_id,
                            content=f"Error: {e}",
                            is_error=True,
                        )
                duration_ms = int((time.monotonic() - started) * 1000)
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
                        "output_chars": len(result.content),
                    },
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_use_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }
                )

        # Add tool results to messages
        messages.append({"role": "user", "content": tool_results})

    else:
        logger.warning(f"Agentic loop hit max turns ({max_turns})")

    turns_used = len([m for m in messages if m["role"] == "assistant"])

    record_event(
        session_id=session_id,
        stage_name=stage_name or "unknown",
        event_type="loop_completion",
        payload={
            "stop_reason": stop_reason,
            "turns_used": turns_used,
            "model": model,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_creation_tokens": total_cache_creation,
            "cache_read_tokens": total_cache_read,
        },
    )

    return AgenticLoopResult(
        messages=messages,
        turns_used=turns_used,
        stop_reason=stop_reason,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cache_creation_input_tokens=total_cache_creation,
        cache_read_input_tokens=total_cache_read,
    )


def extract_final_text(messages: list[dict]) -> str | None:
    """Extract the final text response from conversation history.

    Walks assistant messages in reverse. For the first assistant message that
    has any text blocks, concatenates all of them (responses can split synthesis
    across multiple blocks). If no assistant message has any text at all
    (tool-driven loop with no narrative), falls back to a summary of the most
    recent assistant turn's tool calls so the caller still gets a non-None
    string — otherwise the strategist memo gets stamped with the caller's
    "No summary available" placeholder and reflection loses the signal.
    """
    def _block_field(block, field: str, default=""):
        """Read a block field whether the block is a dict (post-serialization)
        or an SDK pydantic model (defensive fallback for tests / call sites
        that bypass run_agentic_loop's normalization)."""
        if isinstance(block, dict):
            return block.get(field, default)
        return getattr(block, field, default)

    last_assistant_with_content: list | None = None
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            continue
        if last_assistant_with_content is None:
            last_assistant_with_content = content
        text_blocks = [
            _block_field(block, "text", "")
            for block in content
            if _block_field(block, "type", None) == "text"
        ]
        if text_blocks:
            return "".join(text_blocks)

    if last_assistant_with_content is not None:
        tool_names = [
            _block_field(block, "name", "")
            for block in last_assistant_with_content
            if _block_field(block, "type", None) == "tool_use"
        ]
        tool_names = [n for n in tool_names if n]
        if tool_names:
            return f"[no narrative summary; final tool calls: {', '.join(tool_names)}]"
    return None
