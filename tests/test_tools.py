"""Tests for the tool system: schema, executor, and reflection."""

import json
import threading
import time


from cascade.tools.schema import callable_to_tool_def, _annotation_to_schema
from cascade.tools.executor import ToolExecutor, ConcurrentToolExecutor
from cascade.tools.reflection import (
    reflect,
    get_reflection_log,
    clear_reflection_log,
    ReflectionPlugin,
)


class TestAnnotationToSchema:
    """Tests for Python type -> JSON Schema conversion."""

    def test_str(self):
        assert _annotation_to_schema(str) == {"type": "string"}

    def test_int(self):
        assert _annotation_to_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _annotation_to_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _annotation_to_schema(bool) == {"type": "boolean"}

    def test_list(self):
        assert _annotation_to_schema(list) == {"type": "array"}

    def test_dict(self):
        assert _annotation_to_schema(dict) == {"type": "object"}

    def test_none_fallback(self):
        assert _annotation_to_schema(None) == {"type": "string"}


class TestCallableToToolDef:
    """Tests for converting callables to ToolDef."""

    def test_basic_function(self):
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        td = callable_to_tool_def("greet", greet)
        assert td.name == "greet"
        assert td.description == "Say hello."
        assert td.parameters["properties"]["name"]["type"] == "string"
        assert "name" in td.parameters["required"]

    def test_multiple_params(self):
        def add(a: int, b: int) -> int:
            return a + b

        td = callable_to_tool_def("add", add, description="Add two numbers")
        assert "a" in td.parameters["properties"]
        assert "b" in td.parameters["properties"]
        assert td.parameters["properties"]["a"]["type"] == "integer"

    def test_optional_param(self):
        def maybe(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        td = callable_to_tool_def("maybe", maybe)
        assert "name" in td.parameters["required"]
        assert "greeting" not in td.parameters.get("required", [])

    def test_no_annotations(self):
        def raw(x):
            return x

        td = callable_to_tool_def("raw", raw, description="fallback")
        assert td.parameters["properties"]["x"]["type"] == "string"

    def test_staticmethod_skips_self(self):
        class Foo:
            @staticmethod
            def bar(x: str) -> str:
                return x

        td = callable_to_tool_def("bar", Foo.bar)
        assert "self" not in td.parameters["properties"]
        assert "x" in td.parameters["properties"]

    def test_docstring_used_as_description(self):
        def helper(x: str) -> str:
            """A helpful function."""
            return x

        td = callable_to_tool_def("helper", helper)
        assert td.description == "A helpful function."

    def test_fallback_description(self):
        def nodoc(x: str) -> str:
            return x

        td = callable_to_tool_def("nodoc", nodoc, description="My fallback")
        assert td.description == "My fallback"

    def test_lambda(self):
        td = callable_to_tool_def("lam", lambda x: x, description="lambda test")
        assert td.name == "lam"


class TestToolExecutor:
    """Tests for ToolExecutor."""

    def _make_executor(self):
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        def fail(x: str) -> str:
            raise ValueError("intentional error")

        tools = {
            "greet": callable_to_tool_def("greet", greet, "Greet"),
            "fail": callable_to_tool_def("fail", fail, "Fail"),
        }
        return ToolExecutor(tools)

    def test_execute_success(self):
        executor = self._make_executor()
        result = json.loads(executor.execute("greet", {"name": "World"}))
        assert result["result"] == "Hello, World!"

    def test_execute_unknown_tool(self):
        executor = self._make_executor()
        result = json.loads(executor.execute("nonexistent", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_handler_error(self):
        executor = self._make_executor()
        result = json.loads(executor.execute("fail", {"x": "test"}))
        assert "error" in result
        assert "intentional error" in result["error"]

    def test_execute_bad_arguments(self):
        executor = self._make_executor()
        result = json.loads(executor.execute("greet", {"wrong_param": "test"}))
        assert "error" in result

    def test_tool_names(self):
        executor = self._make_executor()
        assert sorted(executor.tool_names) == ["fail", "greet"]

    def test_has_tool(self):
        executor = self._make_executor()
        assert executor.has_tool("greet") is True
        assert executor.has_tool("missing") is False


class TestToolExecutorWithHooks:
    """Tests for ToolExecutor with hook lifecycle integration."""

    def _make_executor_with_hooks(self, hook_defs):
        from cascade.hooks import HookRunner

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tools = {
            "greet": callable_to_tool_def("greet", greet, "Greet"),
        }
        runner = HookRunner(hooks=tuple(hook_defs))
        return ToolExecutor(tools, hook_runner=runner)

    def test_no_hooks_backward_compat(self):
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tools = {"greet": callable_to_tool_def("greet", greet, "Greet")}
        executor = ToolExecutor(tools)
        result = json.loads(executor.execute("greet", {"name": "World"}))
        assert result["result"] == "Hello, World!"

    def test_hook_blocks_tool(self):
        from cascade.hooks import HookEvent, HookDefinition, HookResult

        def blocker(ctx):
            return HookResult(block=True, reason="Dangerous tool")

        executor = self._make_executor_with_hooks([
            HookDefinition(name="blocker", event=HookEvent.TOOL_CALL, handler=blocker),
        ])
        result = json.loads(executor.execute("greet", {"name": "World"}))
        assert "error" in result
        assert "blocked" in result["error"].lower()
        assert "Dangerous tool" in result["error"]

    def test_hook_transforms_arguments(self):
        from cascade.hooks import HookEvent, HookDefinition, HookResult

        def transformer(ctx):
            return HookResult(transformed_value={"name": "Transformed"})

        executor = self._make_executor_with_hooks([
            HookDefinition(name="transformer", event=HookEvent.TOOL_CALL, handler=transformer),
        ])
        result = json.loads(executor.execute("greet", {"name": "Original"}))
        assert result["result"] == "Hello, Transformed!"

    def test_hook_transforms_result(self):
        from cascade.hooks import HookEvent, HookDefinition, HookResult

        def result_transformer(ctx):
            return HookResult(transformed_value="Modified result")

        executor = self._make_executor_with_hooks([
            HookDefinition(name="rt", event=HookEvent.TOOL_RESULT, handler=result_transformer),
        ])
        result = json.loads(executor.execute("greet", {"name": "World"}))
        assert result["result"] == "Modified result"

    def test_hook_invalid_transform_type(self):
        from cascade.hooks import HookEvent, HookDefinition, HookResult

        def bad_transformer(ctx):
            return HookResult(transformed_value="not a dict")

        executor = self._make_executor_with_hooks([
            HookDefinition(name="bad", event=HookEvent.TOOL_CALL, handler=bad_transformer),
        ])
        result = json.loads(executor.execute("greet", {"name": "World"}))
        assert "error" in result
        assert "invalid arguments type" in result["error"].lower()

    def test_hook_error_does_not_crash(self):
        from cascade.hooks import HookEvent, HookDefinition

        def crasher(ctx):
            raise RuntimeError("Hook exploded")

        executor = self._make_executor_with_hooks([
            HookDefinition(name="crash", event=HookEvent.TOOL_CALL, handler=crasher),
        ])
        # Hook error is non-fatal (emit returns None on error), tool runs normally
        result = json.loads(executor.execute("greet", {"name": "World"}))
        assert result["result"] == "Hello, World!"

    def test_passthrough_hook(self):
        from cascade.hooks import HookEvent, HookDefinition

        def noop(ctx):
            return None

        executor = self._make_executor_with_hooks([
            HookDefinition(name="noop", event=HookEvent.TOOL_CALL, handler=noop),
        ])
        result = json.loads(executor.execute("greet", {"name": "World"}))
        assert result["result"] == "Hello, World!"


class TestReflection:
    """Tests for the reflection tool."""

    def setup_method(self):
        clear_reflection_log()

    def test_valid_reflection(self):
        result = reflect("difficulty", "this is hard")
        assert "Reflection noted" in result
        assert "difficulty" in result

    def test_invalid_situation(self):
        result = reflect("invalid_situation", "nope")
        assert "Invalid situation" in result

    def test_log_capture(self):
        reflect("uncertainty", "not sure about this")
        log = get_reflection_log()
        assert len(log) == 1
        assert log[0]["situation"] == "uncertainty"
        assert log[0]["thought"] == "not sure about this"
        assert "timestamp" in log[0]

    def test_clear_log(self):
        reflect("recognition", "good work")
        assert len(get_reflection_log()) == 1
        clear_reflection_log()
        assert len(get_reflection_log()) == 0

    def test_multiple_reflections(self):
        reflect("difficulty", "one")
        reflect("endings", "two")
        reflect("conflict", "three")
        assert len(get_reflection_log()) == 3

    def test_all_valid_situations(self):
        for situation in ("difficulty", "conflict", "uncertainty", "recognition", "endings"):
            result = reflect(situation, "test")
            assert "Reflection noted" in result


class TestReflectionPlugin:
    """Tests for the ReflectionPlugin as a BasePlugin."""

    def test_plugin_properties(self):
        plugin = ReflectionPlugin()
        assert plugin.name == "reflection"
        assert "reflection" in plugin.description.lower()

    def test_plugin_get_tools(self):
        plugin = ReflectionPlugin()
        tools = plugin.get_tools()
        assert "reflect" in tools
        assert callable(tools["reflect"])


class TestConcurrencySafe:
    """Tests for the ToolDef concurrency classification."""

    def _tool(self, **flags):
        def fn(x: str) -> str:
            return x

        return callable_to_tool_def("fn", fn, **flags)

    def test_read_only_is_concurrency_safe(self):
        td = self._tool(read_only=True)
        assert td.is_read_only is True
        assert td.concurrency_safe is True

    def test_explicit_concurrent_is_concurrency_safe(self):
        td = self._tool(concurrent=True)
        assert td.is_concurrent is True
        assert td.concurrency_safe is True

    def test_default_tool_is_not_concurrency_safe(self):
        td = self._tool()
        assert td.concurrency_safe is False

    def test_destructive_tool_is_not_concurrency_safe(self):
        td = self._tool(destructive=True)
        assert td.is_destructive is True
        assert td.concurrency_safe is False

    def test_destructive_overrides_read_only(self):
        td = self._tool(read_only=True, destructive=True)
        assert td.concurrency_safe is False

    def test_flags_default_off_for_existing_callers(self):
        td = self._tool()
        assert (td.is_read_only, td.is_concurrent, td.is_destructive) == (False, False, False)


class _ConcurrencyProbe:
    """Records the peak number of overlapping tool invocations."""

    def __init__(self, hold: float = 0.05):
        self._lock = threading.Lock()
        self._hold = hold
        self.active = 0
        self.max_active = 0

    def run(self, tag: str) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self._hold)
            return tag
        finally:
            with self._lock:
                self.active -= 1


class TestConcurrentToolExecutor:
    """Tests for batch execution with concurrency control."""

    @staticmethod
    def _results(raw: list[str]) -> list:
        return [json.loads(r).get("result", json.loads(r)) for r in raw]

    def test_safe_calls_run_in_parallel(self):
        probe = _ConcurrencyProbe(hold=0.05)

        def read(tag: str) -> str:
            return probe.run(tag)

        tools = {"read": callable_to_tool_def("read", read, read_only=True)}
        executor = ConcurrentToolExecutor(tools)
        calls = [("read", {"tag": t}) for t in ("a", "b", "c")]

        results = executor.execute_batch(calls)

        assert self._results(results) == ["a", "b", "c"]
        # At least two read-only calls were inside the tool at the same time.
        assert probe.max_active >= 2

    def test_unsafe_calls_are_serialised(self):
        probe = _ConcurrencyProbe(hold=0.02)

        def write(tag: str) -> str:
            return probe.run(tag)

        tools = {"write": callable_to_tool_def("write", write)}
        executor = ConcurrentToolExecutor(tools)
        calls = [("write", {"tag": t}) for t in ("a", "b", "c")]

        results = executor.execute_batch(calls)

        assert self._results(results) == ["a", "b", "c"]
        # Mutating calls never overlapped.
        assert probe.max_active == 1

    def test_mixed_batch_preserves_request_order(self):
        def read(x: str) -> str:
            return f"r:{x}"

        def write(x: str) -> str:
            return f"w:{x}"

        tools = {
            "read": callable_to_tool_def("read", read, read_only=True),
            "write": callable_to_tool_def("write", write),
        }
        executor = ConcurrentToolExecutor(tools)
        calls = [
            ("read", {"x": "1"}),
            ("read", {"x": "2"}),
            ("write", {"x": "3"}),
            ("read", {"x": "4"}),
        ]

        results = self._results(executor.execute_batch(calls))

        assert results == ["r:1", "r:2", "w:3", "r:4"]

    def test_unsafe_call_is_an_ordering_barrier(self):
        # The write must be ordered after the first read and before the second,
        # so the reads observe state from before and after the mutation.
        state = {"v": "initial"}

        def read() -> str:
            return state["v"]

        def write(value: str) -> bool:
            state["v"] = value
            return True

        tools = {
            "read": callable_to_tool_def("read", read, read_only=True),
            "write": callable_to_tool_def("write", write),
        }
        executor = ConcurrentToolExecutor(tools)
        calls = [
            ("read", {}),
            ("write", {"value": "updated"}),
            ("read", {}),
        ]

        results = self._results(executor.execute_batch(calls))

        assert results == ["initial", True, "updated"]

    def test_unknown_tool_returns_error_in_its_slot(self):
        def read(x: str) -> str:
            return x

        tools = {"read": callable_to_tool_def("read", read, read_only=True)}
        executor = ConcurrentToolExecutor(tools)
        calls = [("read", {"x": "ok"}), ("missing", {}), ("read", {"x": "fine"})]

        results = [json.loads(r) for r in executor.execute_batch(calls)]

        assert results[0]["result"] == "ok"
        assert "Unknown tool" in results[1]["error"]
        assert results[2]["result"] == "fine"

    def test_batch_reuses_hook_lifecycle(self):
        from cascade.hooks import HookEvent, HookDefinition, HookResult, HookRunner

        def read(x: str) -> str:
            return x

        def blocker(ctx):
            return HookResult(block=True, reason="Dangerous")

        tools = {"read": callable_to_tool_def("read", read, read_only=True)}
        runner = HookRunner(
            hooks=(HookDefinition(name="b", event=HookEvent.TOOL_CALL, handler=blocker),),
        )
        executor = ConcurrentToolExecutor(tools, hook_runner=runner)

        results = [json.loads(r) for r in executor.execute_batch([("read", {"x": "a"})])]

        assert "blocked" in results[0]["error"].lower()

    def test_empty_batch_returns_empty_list(self):
        executor = ConcurrentToolExecutor({})
        assert executor.execute_batch([]) == []
