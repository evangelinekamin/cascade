"""Execute tool calls and return results.

Provides a safe execution wrapper that catches exceptions and returns
structured results for the tool calling loop. Supports hook lifecycle
events for tool_call (pre-execution) and tool_result (post-execution).

The ConcurrentToolExecutor extends this with batch parallelism: tools marked
``concurrency_safe`` run together in a thread pool, while every other call is a
serialisation barrier. Results are always returned in request order.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from .schema import ToolDef
from ..hooks import HookEvent, HookContext, HookRunner


class ToolExecutor:
    """Execute registered tools by name with argument dicts.

    Supports Pi-style tool lifecycle hooks:
    - tool_call: fired before execution, can block or modify arguments
    - tool_result: fired after execution, can modify the result
    """

    def __init__(
        self,
        tools: dict[str, ToolDef],
        hook_runner: Optional[HookRunner] = None,
    ):
        self._tools = dict(tools)
        self._hook_runner = hook_runner

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the result as a JSON string.

        Fires tool_call hook before execution (can block/transform args).
        Fires tool_result hook after execution (can transform result).

        Args:
            tool_name: Name of the tool to call.
            arguments: Keyword arguments for the tool handler.

        Returns:
            JSON-encoded result string. On error, returns a JSON error object.
        """
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        # Fire tool_call hook (can block or transform arguments)
        if self._hook_runner:
            ctx = HookContext(
                event=HookEvent.TOOL_CALL.value,
                tool_name=tool_name,
                tool_input=tuple(arguments.items()),
            )
            hook_result = self._hook_runner.emit(HookEvent.TOOL_CALL, ctx)
            if hook_result is not None:
                if hook_result.block:
                    return json.dumps({
                        "error": f"Tool '{tool_name}' blocked by hook: {hook_result.reason}",
                    })
                if hook_result.transformed_value is not None:
                    if not isinstance(hook_result.transformed_value, dict):
                        return json.dumps({
                            "error": (
                                f"Hook returned invalid arguments type for {tool_name}: "
                                f"{type(hook_result.transformed_value).__name__} (expected dict)"
                            ),
                        })
                    arguments = hook_result.transformed_value

        tool = self._tools[tool_name]
        try:
            result = tool.handler(**arguments)
            result_str = json.dumps({"result": result})

            # Fire tool_result hook (can transform result)
            if self._hook_runner:
                ctx = HookContext(
                    event=HookEvent.TOOL_RESULT.value,
                    tool_name=tool_name,
                    tool_input=tuple(arguments.items()),
                    tool_output=result_str,
                )
                hook_result = self._hook_runner.emit(HookEvent.TOOL_RESULT, ctx)
                if hook_result is not None and hook_result.transformed_value is not None:
                    result_str = json.dumps({"result": hook_result.transformed_value})

            return result_str

        except TypeError as e:
            return json.dumps({"error": f"Invalid arguments for {tool_name}: {e}"})
        except Exception as e:
            return json.dumps({"error": f"Tool {tool_name} failed: {e}"})


Call = tuple[str, dict[str, Any]]


class ConcurrentToolExecutor(ToolExecutor):
    """Run a batch of tool calls, overlapping the concurrency-safe ones.

    Maximal runs of consecutive ``concurrency_safe`` calls execute together in
    a thread pool. Every other call is a serialisation barrier: it runs alone,
    after all preceding calls have finished and before any following call
    starts. Results are returned in request order, so callers can zip them back
    against the calls they submitted.

    Single-call execution and the full hook lifecycle are inherited unchanged
    from ToolExecutor; this subclass only decides what is allowed to overlap.
    """

    def __init__(
        self,
        tools: dict[str, ToolDef],
        hook_runner: Optional[HookRunner] = None,
        max_workers: Optional[int] = None,
    ):
        super().__init__(tools, hook_runner)
        self._max_workers = max_workers

    def execute_batch(self, calls: list[Call]) -> list[str]:
        """Execute *calls* in request order, overlapping safe runs.

        Args:
            calls: Ordered ``(tool_name, arguments)`` pairs to execute.

        Returns:
            JSON-encoded result strings aligned one-to-one with *calls*.
        """
        results: list[str] = []
        segment: list[Call] = []

        for tool_name, arguments in calls:
            if self._is_concurrency_safe(tool_name):
                segment.append((tool_name, arguments))
                continue
            # A non-safe call is a barrier: flush pending safe calls, then run
            # this one exclusively before any later call is considered.
            results.extend(self._run_parallel(segment))
            segment = []
            results.append(self.execute(tool_name, arguments))

        results.extend(self._run_parallel(segment))
        return results

    def _is_concurrency_safe(self, tool_name: str) -> bool:
        tool = self._tools.get(tool_name)
        return tool is not None and tool.concurrency_safe

    def _run_parallel(self, calls: list[Call]) -> list[str]:
        """Run a run of concurrency-safe calls together, preserving order."""
        if not calls:
            return []
        if len(calls) == 1:
            tool_name, arguments = calls[0]
            return [self.execute(tool_name, arguments)]

        workers = min(len(calls), self._max_workers or len(calls))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda call: self.execute(*call), calls))
