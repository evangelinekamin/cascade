"""Execute tool calls and return results.

Provides a safe execution wrapper that catches exceptions and returns
structured results for the tool calling loop. Supports hook lifecycle
events for tool_call (pre-execution) and tool_result (post-execution).

The ConcurrentToolExecutor extends this with batch parallelism: tools marked
``concurrency_safe`` run together in a thread pool, while every other call is a
serialisation barrier. Results are always returned in request order.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from .schema import ToolDef
from .permissions import PermissionContext
from ..hooks import HookEvent, HookContext, HookRunner


_DEFAULT_MAX_WORKERS = 8


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
        permissions=None,
        permission_context: Optional[PermissionContext] = None,
    ):
        self._tools = dict(tools)
        self._hook_runner = hook_runner
        # PermissionEngine (tools/permissions.py). Evaluated BEFORE hooks:
        # deny/sacred verdicts must not be bypassable by hook transforms.
        self._permissions = permissions
        self._permission_context = permission_context or PermissionContext()

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
        if not isinstance(arguments, dict):
            return self._apply_result_hook(
                tool_name,
                {},
                json.dumps({
                    "error": (
                        f"Invalid arguments for {tool_name}: expected object, "
                        f"got {type(arguments).__name__}"
                    ),
                }),
            )

        if tool_name not in self._tools:
            return self._apply_result_hook(
                tool_name,
                arguments,
                json.dumps({"error": f"Unknown tool: {tool_name}"}),
            )

        if self._permissions is not None:
            verdict = self._permissions.resolve(
                self._tools.get(tool_name),
                tool_name,
                arguments,
                context=self._permission_context,
            )
            if verdict.rule == "escalation":
                # The denial limit is hit: stop the loop rather than let the
                # model keep re-raising the same blocked call.
                from .permissions import PermissionAbort

                raise PermissionAbort(verdict.reason)
            if verdict.decision != "allow":
                return self._apply_result_hook(
                    tool_name,
                    arguments,
                    json.dumps({
                        "error": f"Tool '{tool_name}' not permitted: {verdict.reason}",
                    }),
                )

        # Fire tool_call hook (can block or transform arguments)
        original_arguments = arguments
        if self._hook_runner:
            ctx = HookContext(
                event=HookEvent.TOOL_CALL.value,
                tool_name=tool_name,
                tool_input=tuple(arguments.items()),
            )
            hook_result = self._hook_runner.emit(HookEvent.TOOL_CALL, ctx)
            if hook_result is not None:
                if hook_result.block:
                    return self._apply_result_hook(
                        tool_name,
                        arguments,
                        json.dumps({
                            "error": (
                                f"Tool '{tool_name}' blocked by hook: "
                                f"{hook_result.reason}"
                            ),
                        }),
                    )
                if hook_result.transformed_value is not None:
                    if not isinstance(hook_result.transformed_value, dict):
                        return self._apply_result_hook(
                            tool_name,
                            arguments,
                            json.dumps({
                                "error": (
                                    f"Hook returned invalid arguments type for {tool_name}: "
                                    f"{type(hook_result.transformed_value).__name__} "
                                    "(expected dict)"
                                ),
                            }),
                        )
                    arguments = hook_result.transformed_value

        tool = self._tools[tool_name]
        # The original call is checked first so a denied/sacred request cannot
        # be laundered through a transform. If a hook changes an allowed call,
        # check the final arguments too so it cannot widen permission silently.
        if self._permissions is not None and arguments != original_arguments:
            verdict = self._permissions.resolve(
                tool,
                tool_name,
                arguments,
                context=self._permission_context,
            )
            if verdict.rule == "escalation":
                from .permissions import PermissionAbort

                raise PermissionAbort(verdict.reason)
            if verdict.decision != "allow":
                return self._apply_result_hook(
                    tool_name,
                    arguments,
                    json.dumps({
                        "error": (
                            f"Transformed tool '{tool_name}' not permitted: "
                            f"{verdict.reason}"
                        ),
                    }),
                )

        arguments, validation_error = self._validated_arguments(tool, arguments)
        if validation_error:
            return self._apply_result_hook(
                tool_name,
                arguments,
                json.dumps({"error": validation_error}),
            )

        try:
            result = tool.handler(**arguments)
            result_str = json.dumps({"result": result})
        except TypeError as e:
            result_str = json.dumps({"error": f"Invalid arguments for {tool_name}: {e}"})
        except Exception as e:
            result_str = json.dumps({"error": f"Tool {tool_name} failed: {e}"})
        return self._apply_result_hook(tool_name, arguments, result_str)

    def _apply_result_hook(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_str: str,
    ) -> str:
        """Emit TOOL_RESULT for every outcome, including denials and failures."""
        if not self._hook_runner:
            return result_str
        ctx = HookContext(
            event=HookEvent.TOOL_RESULT.value,
            tool_name=tool_name,
            tool_input=tuple(arguments.items()),
            tool_output=result_str,
        )
        hook_result = self._hook_runner.emit(HookEvent.TOOL_RESULT, ctx)
        if hook_result is not None and hook_result.transformed_value is not None:
            try:
                return json.dumps({"result": hook_result.transformed_value})
            except (TypeError, ValueError):
                # A malformed observational hook must not destroy the real tool
                # result or turn a successful call into a provider-loop crash.
                return result_str
        return result_str

    @staticmethod
    def _validated_arguments(
        tool: ToolDef,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[str]]:
        """Validate and conservatively coerce common small-model arg mistakes."""
        schema = tool.parameters or {}
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return arguments, None

        required = schema.get("required", ())
        missing = [name for name in required if name not in arguments]
        if missing:
            return arguments, (
                f"Invalid arguments for {tool.name}: missing required "
                f"{', '.join(sorted(missing))}"
            )

        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(arguments) - set(properties))
            if unexpected:
                return arguments, (
                    f"Invalid arguments for {tool.name}: unexpected "
                    f"{', '.join(unexpected)}"
                )

        coerced = dict(arguments)
        for name, value in arguments.items():
            prop = properties.get(name)
            if not isinstance(prop, dict):
                continue
            converted, error = _coerce_schema_value(value, prop)
            if error:
                return arguments, f"Invalid argument '{name}' for {tool.name}: {error}"
            coerced[name] = converted
        return coerced, None


def _coerce_schema_value(value: Any, schema: dict) -> tuple[Any, Optional[str]]:
    """Coerce only unambiguous scalar/JSON-string representations."""
    if "anyOf" in schema:
        for variant in schema["anyOf"]:
            converted, error = _coerce_schema_value(value, variant)
            if error is None:
                return converted, None
        return value, "does not match any allowed type"

    expected = schema.get("type")
    converted = value
    if expected == "string":
        if not isinstance(value, str):
            return value, f"expected string, got {type(value).__name__}"
    elif expected == "integer":
        if isinstance(value, bool):
            return value, "expected integer, got bool"
        if isinstance(value, int):
            converted = value
        elif isinstance(value, str):
            try:
                converted = int(value.strip())
            except ValueError:
                return value, f"expected integer, got {value!r}"
        else:
            return value, f"expected integer, got {type(value).__name__}"
    elif expected == "number":
        if isinstance(value, bool):
            return value, "expected number, got bool"
        if isinstance(value, (int, float)):
            converted = value
        elif isinstance(value, str):
            try:
                converted = float(value.strip())
            except ValueError:
                return value, f"expected number, got {value!r}"
        else:
            return value, f"expected number, got {type(value).__name__}"
    elif expected == "boolean":
        if isinstance(value, bool):
            converted = value
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            converted = value.strip().lower() == "true"
        else:
            return value, f"expected boolean, got {type(value).__name__}"
    elif expected in {"array", "object"}:
        target_type = list if expected == "array" else dict
        if isinstance(value, target_type):
            converted = value
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return value, f"expected {expected}, got invalid JSON string"
            if not isinstance(parsed, target_type):
                return value, f"expected {expected}, got {type(parsed).__name__}"
            converted = parsed
        else:
            return value, f"expected {expected}, got {type(value).__name__}"

    if "enum" in schema and converted not in schema["enum"]:
        return value, f"expected one of {schema['enum']!r}, got {converted!r}"
    return converted, None


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
        permissions=None,
        permission_context: Optional[PermissionContext] = None,
    ):
        super().__init__(
            tools,
            hook_runner,
            permissions=permissions,
            permission_context=permission_context,
        )
        self._max_workers = (
            max(1, int(max_workers))
            if max_workers is not None
            else _DEFAULT_MAX_WORKERS
        )

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

        results: list[str] = []
        for wave in self._non_conflicting_waves(calls):
            if len(wave) == 1:
                results.append(self.execute(*wave[0]))
                continue
            workers = min(len(wave), self._max_workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results.extend(pool.map(lambda call: self.execute(*call), wave))
        return results

    def _non_conflicting_waves(self, calls: list[Call]) -> list[list[Call]]:
        """Partition a safe segment so overlapping mutable paths never race."""
        waves: list[list[Call]] = []
        current: list[Call] = []
        for call in calls:
            if current and any(self._calls_conflict(existing, call) for existing in current):
                waves.append(current)
                current = []
            current.append(call)
        if current:
            waves.append(current)
        return waves

    def _calls_conflict(self, left: Call, right: Call) -> bool:
        left_tool = self._tools.get(left[0])
        right_tool = self._tools.get(right[0])
        if left_tool is None or right_tool is None:
            return True
        if left_tool.is_read_only and right_tool.is_read_only:
            return False

        left_paths = _resource_paths(left[1])
        right_paths = _resource_paths(right[1])
        return any(
            _paths_overlap(left_path, right_path)
            for left_path in left_paths
            for right_path in right_paths
        )


_PATH_ARGUMENTS = frozenset({
    "path",
    "file",
    "file_path",
    "directory",
    "root",
    "cwd",
    "working_directory",
})


def _resource_paths(arguments: dict[str, Any]) -> tuple[str, ...]:
    paths = []
    for key, value in arguments.items():
        if key not in _PATH_ARGUMENTS or not isinstance(value, str) or not value:
            continue
        paths.append(os.path.abspath(os.path.normpath(os.path.expanduser(value))))
    return tuple(paths)


def _paths_overlap(left: str, right: str) -> bool:
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return False
    return common == left or common == right
