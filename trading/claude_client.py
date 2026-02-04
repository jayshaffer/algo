"""Claude API client with tool handling and agentic loop."""

import os
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import anthropic

logger = logging.getLogger(__name__)


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


def get_claude_client() -> anthropic.Anthropic:
    """
    Create Claude client.

    Requires ANTHROPIC_API_KEY environment variable.

    Returns:
        Configured Anthropic client

    Raises:
        ValueError: If no API key is found
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Get your key at https://console.anthropic.com/"
        )

    return anthropic.Anthropic(api_key=api_key)


def run_agentic_loop(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    initial_message: str,
    tools: list[dict],
    tool_handlers: dict[str, Callable],
    max_turns: int = 20,
) -> AgenticLoopResult:
    """
    Run an agentic loop where Claude uses tools until it completes its task.

    Args:
        client: Anthropic client
        model: Model ID (e.g., 'claude-sonnet-4-20250514')
        system: System prompt
        initial_message: Initial user message to start the loop
        tools: List of tool definitions
        tool_handlers: Dict mapping tool names to handler functions
        max_turns: Maximum conversation turns (safety limit)

    Returns:
        AgenticLoopResult with conversation history and metadata
    """
    messages = [{"role": "user", "content": initial_message}]
    total_input_tokens = 0
    total_output_tokens = 0
    stop_reason = "max_turns"

    for turn in range(max_turns):
        logger.info(f"Agentic loop turn {turn + 1}/{max_turns}")

        # Call Claude
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Track token usage
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

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
    )


def extract_final_text(messages: list[dict]) -> Optional[str]:
    """
    Extract the final text response from conversation history.

    Args:
        messages: Conversation history

    Returns:
        Final text from Claude's last response, or None
    """
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
