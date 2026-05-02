"""Claude API client with tool handling and agentic loop."""

import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

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


def get_claude_client() -> anthropic.Anthropic:
    """Create Claude client. Requires ANTHROPIC_API_KEY environment variable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Get your key at https://console.anthropic.com/"
        )

    return anthropic.Anthropic(api_key=api_key, max_retries=5)


def _call_with_retry(client, max_retries=API_MAX_RETRIES, **create_kwargs):
    """Call client.messages.create() with retry and exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**create_kwargs)
        except RETRYABLE_ERRORS as e:
            if attempt == max_retries:
                logger.error(
                    "API call failed after %d retries: %s", max_retries, e
                )
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
    return [messages[0], *messages[-4:]]


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
) -> AgenticLoopResult:
    """Run an agentic loop where Claude uses tools until it completes its task."""
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
                messages = _aggressive_prune(messages)
                continue
            raise

        # Track token usage
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        total_cache_creation += getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        total_cache_read += getattr(response.usage, "cache_read_input_tokens", 0) or 0

        # Drop the truncated turn (incomplete tool_use blocks would break the next call) and retry once with a concision nudge.
        if response.stop_reason == "max_tokens" and max_tokens_recoveries < 1:
            max_tokens_recoveries += 1
            logger.warning(
                "max_tokens hit on turn %d; discarding truncated response and "
                "retrying with concision instruction",
                turn + 1,
            )
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

        # Add assistant response to history
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
            getattr(block, "text", "")
            for block in content
            if getattr(block, "type", None) == "text"
        ]
        if text_blocks:
            return "".join(text_blocks)

    if last_assistant_with_content is not None:
        tool_names = [
            getattr(block, "name", "")
            for block in last_assistant_with_content
            if getattr(block, "type", None) == "tool_use"
        ]
        tool_names = [n for n in tool_names if n]
        if tool_names:
            return f"[no narrative summary; final tool calls: {', '.join(tool_names)}]"
    return None
