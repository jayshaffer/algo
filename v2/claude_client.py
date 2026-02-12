"""Claude API client with tool handling and agentic loop."""

import os
import time
import random
import logging
from dataclasses import dataclass
from typing import Callable, Optional

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

    for turn in range(max_turns):
        logger.info(f"Agentic loop turn {turn + 1}/{max_turns}")

        response = _call_with_retry(
            client,
            model=model,
            max_tokens=4096,
            system=cached_system,
            tools=cached_tools,
            messages=_messages_with_cache_breakpoint(messages),
        )

        # Track token usage
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        total_cache_creation += getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        total_cache_read += getattr(response.usage, "cache_read_input_tokens", 0) or 0

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


def extract_final_text(messages: list[dict]) -> Optional[str]:
    """Extract the final text response from conversation history."""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        return block.text
            elif isinstance(content, str):
                return content
    return None
