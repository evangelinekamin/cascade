"""Shared OpenAI-compatible tool calling logic.

Used by both OpenAIProvider and OpenRouterProvider since they share
the same chat completions API format for tool calling.
"""

import json
from typing import Optional, TYPE_CHECKING

import httpx

from .base import ToolEvent, ToolEventCallback

if TYPE_CHECKING:
    from .base import Message
    from ..tools.schema import ToolDef


def openai_ask_with_tools(
    client: httpx.Client,
    url: str,
    headers: dict,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    messages: list["Message"],
    tools: dict[str, "ToolDef"],
    system: Optional[str] = None,
    max_rounds: int = 5,
    on_tool_event: ToolEventCallback = None,
) -> tuple[str, list[dict]]:
    """OpenAI-compatible tool calling loop.

    Args:
        client: httpx.Client instance.
        url: Chat completions endpoint URL.
        headers: Request headers with auth.
        model: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        messages: Conversation history as Message dicts.
        tools: Mapping of tool_name -> ToolDef.
        system: Optional system prompt.
        max_rounds: Maximum tool-calling round trips.

    Returns:
        Tuple of (final_text_response, tool_calls_log).
    """
    from ..tools.executor import ToolExecutor

    executor = ToolExecutor(tools)

    # Build OpenAI tool definitions
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            },
        }
        for td in tools.values()
    ]

    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(
        {"role": m["role"], "content": m["content"]}
        for m in messages
    )

    tool_log = []
    content = ""

    for round_num in range(max_rounds):
        payload = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "tools": tool_defs,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return f"Error: {e}", tool_log

        choices = data.get("choices", [])
        if not choices:
            return "", tool_log

        message = choices[0].get("message", {})
        finish_reason = choices[0].get("finish_reason", "stop")

        tool_calls = message.get("tool_calls", [])
        content = message.get("content", "") or ""

        if not tool_calls or finish_reason != "tool_calls":
            return content, tool_log

        # Append the assistant message (must include tool_calls)
        api_messages.append(message)

        # Execute each tool call
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            try:
                tool_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                tool_args = {}

            if on_tool_event:
                on_tool_event(ToolEvent(
                    kind="tool_start",
                    tool_name=tool_name,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    tool_input=tool_args,
                ))

            result = executor.execute(tool_name, tool_args)
            tool_log.append({
                "tool": tool_name,
                "input": tool_args,
                "output": result,
            })

            if on_tool_event:
                on_tool_event(ToolEvent(
                    kind="tool_done",
                    tool_name=tool_name,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    tool_input=tool_args,
                    tool_output=result,
                ))

            api_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return content, tool_log
