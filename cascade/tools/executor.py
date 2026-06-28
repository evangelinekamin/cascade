"""Execute tool calls and return results.

Provides a safe execution wrapper that catches exceptions and returns
structured results for the tool calling loop. Supports hook lifecycle
events for tool_call (pre-execution) and tool_result (post-execution).

The ConcurrentToolExecutor extends this with async support: tools marked
``concurrency_safe`` run in parallel, others get exclusive access.
"""

import asyncio
import json
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


class ConcurrentToolExecutor(ToolExecutor):
    """Tool executor with async concurrency control.

    Concurrent-safe tools run in parallel. Non-concurrent tools acquire
    an exclusive lock and wait for all running tools to finish first.
    """

    def __init__(
        self,
        tools: dict[str, ToolDef],
        hook_runner: Optional[HookRunner] = None,
    ):
        super().__init__(tools, hook_runner)
        self._exclusive_lock = asyncio.Lock()
        self._running_count = 0
        self._all_done = asyncio.Event()
        self._all_done.set()

    async def execute_async(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool with concurrency control.

        Concurrent tools run alongside each other. Non-concurrent tools
        wait for all running tools to finish, then run exclusively.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        if tool.concurrency_safe:
            return await self._run_concurrent(tool_name, arguments)
        return await self._run_exclusive(tool_name, arguments)

    async def execute_batch(
        self,
        calls: list[tuple[str, dict[str, Any]]],
    ) -> list[str]:
        """Execute a batch of tool calls, parallelising where safe.

        Concurrent calls run together; non-concurrent calls are serialised.
        """
        concurrent = []
        sequential = []

        for tool_name, args in calls:
            tool = self._tools.get(tool_name)
            if tool is not None and tool.concurrency_safe:
                concurrent.append((tool_name, args))
            else:
                sequential.append((tool_name, args))

        results: list[str] = []

        # Run concurrent batch in parallel
        if concurrent:
            tasks = [self.execute_async(n, a) for n, a in concurrent]
            results.extend(await asyncio.gather(*tasks))

        # Run sequential tools one at a time
        for tool_name, args in sequential:
            results.append(await self.execute_async(tool_name, args))

        return results

    async def _run_concurrent(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Run a concurrent-safe tool without exclusive lock."""
        self._running_count += 1
        self._all_done.clear()
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.execute, tool_name, arguments)
        finally:
            self._running_count -= 1
            if self._running_count == 0:
                self._all_done.set()

    async def _run_exclusive(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Run a non-concurrent tool with exclusive access."""
        async with self._exclusive_lock:
            # Wait for all concurrent tools to finish
            await self._all_done.wait()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.execute, tool_name, arguments)
