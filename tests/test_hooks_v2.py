"""Tests for the v2 hook system: rich lifecycle events and Python module hooks."""

import pytest

from cascade.hooks.events import HookEvent, EVENT_MAP
from cascade.hooks.context import HookContext, HookResult
from cascade.hooks.runner import HookDefinition, HookRunner
from cascade.hooks.loader import load_hooks_from_config
from cascade.hooks.matchers import compile_matcher, matches_any


class TestHookEvents:
    """Tests for the expanded lifecycle events."""

    def test_all_legacy_events_present(self):
        assert HookEvent.BEFORE_ASK == "before_ask"
        assert HookEvent.AFTER_RESPONSE == "after_response"
        assert HookEvent.ON_EXIT == "on_exit"
        assert HookEvent.ON_ERROR == "on_error"

    def test_new_lifecycle_events(self):
        assert HookEvent.INPUT_RECEIVED == "input_received"
        assert HookEvent.CONTEXT_BUILD == "context_build"
        assert HookEvent.BEFORE_PROVIDER_REQUEST == "before_provider_request"
        assert HookEvent.TOOL_CALL == "tool_call"
        assert HookEvent.TOOL_RESULT == "tool_result"
        assert HookEvent.EPISODE_GENERATED == "episode_generated"
        assert HookEvent.PROVIDER_SWITCH == "provider_switch"
        assert HookEvent.SESSION_START == "session_start"

    def test_event_map_covers_all(self):
        for event in HookEvent:
            assert event.value in EVENT_MAP
            assert EVENT_MAP[event.value] == event


class TestHookContext:
    """Tests for the rich hook context object."""

    def test_basic_context(self):
        ctx = HookContext(
            event="before_ask",
            provider="claude",
            prompt="Hello world",
        )
        assert ctx.event == "before_ask"
        assert ctx.provider == "claude"
        assert ctx.prompt == "Hello world"

    def test_to_env_dict(self):
        ctx = HookContext(
            event="tool_call",
            provider="gemini",
            tool_name="read_file",
            metadata=(("extra_key", "extra_value"),),
        )
        env = ctx.to_env_dict()
        assert env["CASCADE_EVENT"] == "tool_call"
        assert env["CASCADE_PROVIDER"] == "gemini"
        assert env["CASCADE_TOOL_NAME"] == "read_file"
        assert env["CASCADE_EXTRA_KEY"] == "extra_value"

    def test_to_env_dict_skips_empty(self):
        ctx = HookContext(event="before_ask")
        env = ctx.to_env_dict()
        assert "CASCADE_PROVIDER" not in env
        assert "CASCADE_TOOL_NAME" not in env

    def test_tool_context(self):
        ctx = HookContext(
            event="tool_call",
            tool_name="write_file",
            tool_input=(("path", "/tmp/test.py"), ("content", "hello")),
        )
        assert ctx.tool_name == "write_file"
        assert dict(ctx.tool_input)["path"] == "/tmp/test.py"

    def test_frozen(self):
        ctx = HookContext(event="before_ask", provider="claude")
        with pytest.raises(AttributeError):
            ctx.provider = "gemini"

    def test_prompt_not_in_env(self):
        """Prompt content must not be passed to shell env (injection risk)."""
        ctx = HookContext(event="before_ask", prompt="echo pwned; rm -rf /")
        env = ctx.to_env_dict()
        assert "CASCADE_PROMPT" not in env
        assert "CASCADE_PROMPT_LENGTH" in env


class TestHookResult:
    """Tests for hook result signaling."""

    def test_passthrough(self):
        result = HookResult()
        assert not result.block
        assert result.transformed_value is None

    def test_block(self):
        result = HookResult(block=True, reason="Security gate")
        assert result.block is True
        assert result.reason == "Security gate"

    def test_transform(self):
        result = HookResult(transformed_value={"modified": True})
        assert not result.block
        assert result.transformed_value == {"modified": True}

    def test_frozen(self):
        result = HookResult()
        with pytest.raises(AttributeError):
            result.block = True


class TestPythonHooks:
    """Tests for Python module hook execution."""

    def test_python_hook_passthrough(self):
        def my_hook(ctx: HookContext):
            return None  # passthrough

        hook = HookDefinition(
            name="passthrough",
            event=HookEvent.BEFORE_ASK,
            handler=my_hook,
        )
        assert hook.is_python_hook is True

        runner = HookRunner(hooks=(hook,))
        results = runner.run_hooks(HookEvent.BEFORE_ASK)
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["type"] == "python"

    def test_python_hook_blocking(self):
        def blocking_hook(ctx: HookContext):
            return HookResult(block=True, reason="Blocked by test")

        hook = HookDefinition(
            name="blocker",
            event=HookEvent.TOOL_CALL,
            handler=blocking_hook,
        )
        runner = HookRunner(hooks=(hook,))
        results = runner.run_hooks(HookEvent.TOOL_CALL)
        assert results[0]["blocked"] is True
        assert "Blocked by test" in results[0]["output"]

    def test_python_hook_transform(self):
        def transform_hook(ctx: HookContext):
            return HookResult(transformed_value="transformed prompt")

        hook = HookDefinition(
            name="transformer",
            event=HookEvent.INPUT_RECEIVED,
            handler=transform_hook,
        )
        runner = HookRunner(hooks=(hook,))
        results = runner.run_hooks(HookEvent.INPUT_RECEIVED)
        assert results[0]["transformed_value"] == "transformed prompt"

    def test_python_hook_error_handling(self):
        def bad_hook(ctx: HookContext):
            raise ValueError("Hook crashed")

        hook = HookDefinition(
            name="bad",
            event=HookEvent.BEFORE_ASK,
            handler=bad_hook,
        )
        runner = HookRunner(hooks=(hook,))
        results = runner.run_hooks(HookEvent.BEFORE_ASK)
        assert results[0]["success"] is False
        assert "Hook crashed" in results[0]["output"]

    def test_blocking_stops_remaining_hooks(self):
        calls = []

        def hook_a(ctx):
            calls.append("a")
            return HookResult(block=True, reason="Stop")

        def hook_b(ctx):
            calls.append("b")
            return None

        hooks = (
            HookDefinition(name="a", event=HookEvent.TOOL_CALL, handler=hook_a, priority=1),
            HookDefinition(name="b", event=HookEvent.TOOL_CALL, handler=hook_b, priority=2),
        )
        runner = HookRunner(hooks=hooks)
        runner.run_hooks(HookEvent.TOOL_CALL)
        assert calls == ["a"]  # b should not have run


class TestHookEmit:
    """Tests for the emit() API."""

    def test_emit_no_hooks(self):
        runner = HookRunner()
        result = runner.emit(HookEvent.BEFORE_ASK, HookContext(event="before_ask"))
        assert result is None

    def test_emit_blocking(self):
        def blocker(ctx):
            return HookResult(block=True, reason="Nope")

        hook = HookDefinition(name="b", event=HookEvent.TOOL_CALL, handler=blocker)
        runner = HookRunner(hooks=(hook,))
        result = runner.emit(
            HookEvent.TOOL_CALL,
            HookContext(event="tool_call", tool_name="bash"),
        )
        assert result is not None
        assert result.block is True

    def test_emit_transform(self):
        def transformer(ctx):
            return HookResult(transformed_value={"modified": True})

        hook = HookDefinition(name="t", event=HookEvent.TOOL_RESULT, handler=transformer)
        runner = HookRunner(hooks=(hook,))
        result = runner.emit(
            HookEvent.TOOL_RESULT,
            HookContext(event="tool_result", tool_name="read_file"),
        )
        assert result is not None
        assert result.transformed_value == {"modified": True}

    def test_emit_passthrough(self):
        def noop(ctx):
            return None

        hook = HookDefinition(name="n", event=HookEvent.BEFORE_ASK, handler=noop)
        runner = HookRunner(hooks=(hook,))
        result = runner.emit(
            HookEvent.BEFORE_ASK,
            HookContext(event="before_ask"),
        )
        assert result is None


class TestHookPriority:
    """Tests for hook execution ordering."""

    def test_priority_ordering(self):
        calls = []

        def make_hook(name, priority):
            def handler(ctx):
                calls.append(name)
                return None
            return HookDefinition(
                name=name,
                event=HookEvent.BEFORE_ASK,
                handler=handler,
                priority=priority,
            )

        hooks = (
            make_hook("low", 100),
            make_hook("high", 1),
            make_hook("mid", 50),
        )
        runner = HookRunner(hooks=hooks)
        runner.run_hooks(HookEvent.BEFORE_ASK)
        assert calls == ["high", "mid", "low"]


class TestHookRunnerImmutability:
    """Tests for immutable runner operations."""

    def test_add_hook_returns_new_runner(self):
        runner = HookRunner()
        hook = HookDefinition(name="test", event=HookEvent.BEFORE_ASK, command="echo hi")
        new_runner = runner.add_hook(hook)
        assert runner.hook_count == 0
        assert new_runner.hook_count == 1


class TestMixedHooks:
    """Tests for mixing shell and Python hooks."""

    def test_shell_and_python_together(self):
        def py_hook(ctx):
            return None

        hooks = (
            HookDefinition(name="shell", event=HookEvent.BEFORE_ASK, command="echo shell"),
            HookDefinition(name="python", event=HookEvent.BEFORE_ASK, handler=py_hook),
        )
        runner = HookRunner(hooks=hooks)
        results = runner.run_hooks(HookEvent.BEFORE_ASK)
        assert len(results) == 2
        assert results[0]["type"] == "shell"
        assert results[1]["type"] == "python"


class TestLoaderV2:
    """Tests for loading new-format hook configs."""

    def test_new_events_loadable(self):
        data = [
            {"name": "input_hook", "event": "input_received", "command": "echo input"},
            {"name": "tool_hook", "event": "tool_call", "command": "echo tool"},
            {"name": "switch_hook", "event": "provider_switch", "command": "echo switch"},
        ]
        hooks = load_hooks_from_config(data)
        assert len(hooks) == 3
        assert hooks[0].event == HookEvent.INPUT_RECEIVED
        assert hooks[1].event == HookEvent.TOOL_CALL
        assert hooks[2].event == HookEvent.PROVIDER_SWITCH

    def test_priority_in_config(self):
        data = [
            {"name": "high", "event": "before_ask", "command": "echo", "priority": 10},
            {"name": "low", "event": "before_ask", "command": "echo", "priority": 200},
        ]
        hooks = load_hooks_from_config(data)
        assert hooks[0].priority == 10
        assert hooks[1].priority == 200

    def test_legacy_config_still_works(self):
        data = [
            {"name": "legacy", "event": "before_ask", "command": "echo legacy"},
        ]
        hooks = load_hooks_from_config(data)
        assert len(hooks) == 1
        assert hooks[0].event == HookEvent.BEFORE_ASK

    def test_module_with_nonexistent_path_skipped(self):
        data = [
            {"name": "bad", "event": "before_ask", "module": "/nonexistent/hook.py"},
        ]
        hooks = load_hooks_from_config(data)
        assert len(hooks) == 0

    def test_describe_includes_type(self):
        def handler(ctx):
            return None

        hooks = (
            HookDefinition(name="shell", event=HookEvent.BEFORE_ASK, command="echo"),
            HookDefinition(name="python", event=HookEvent.BEFORE_ASK, handler=handler),
        )
        runner = HookRunner(hooks=hooks)
        desc = runner.describe()
        assert desc[0]["type"] == "shell"
        assert desc[1]["type"] == "python"


class TestToolMatchers:
    """Tests for Claude Code-style tool pattern matching (``if:`` filters)."""

    def test_colon_prefix_matches_command_start(self):
        m = compile_matcher("Bash(git:*)")
        assert m.matches("Bash", {"command": "git status"}) is True
        assert m.matches("Bash", {"command": "git log --oneline"}) is True

    def test_colon_prefix_does_not_match_other_command(self):
        m = compile_matcher("Bash(git:*)")
        assert m.matches("Bash", {"command": "npm install"}) is False

    def test_colon_prefix_requires_prefix_at_start(self):
        # "git" appears but not at the start -> no match.
        m = compile_matcher("Bash(git:*)")
        assert m.matches("Bash", {"command": "sudo git push"}) is False

    def test_wildcard_anywhere_still_matches(self):
        m = compile_matcher("Bash(*rm*)")
        assert m.matches("Bash", {"command": "rm -rf x"}) is True
        assert m.matches("Bash", {"command": "ls -la"}) is False

    def test_tool_name_must_match(self):
        m = compile_matcher("Bash(git:*)")
        assert m.matches("Write", {"command": "git status"}) is False

    def test_no_arg_pattern_matches_any_arguments(self):
        m = compile_matcher("Bash")
        assert m.matches("Bash", {"command": "anything at all"}) is True
        assert m.matches("Bash", None) is True

    def test_star_matches_any_tool(self):
        m = compile_matcher("*")
        assert m.matches("Bash", {"command": "git status"}) is True
        assert m.matches("Write", {"path": "/tmp/x"}) is True

    def test_colon_prefix_no_match_when_arguments_missing(self):
        m = compile_matcher("Bash(git:*)")
        assert m.matches("Bash", None) is False

    def test_matches_any_across_matchers(self):
        matchers = (compile_matcher("Bash(git:*)"), compile_matcher("Write"))
        assert matches_any(matchers, "Write", {"path": "/tmp/x"}) is True
        assert matches_any(matchers, "Bash", {"command": "git status"}) is True
        assert matches_any(matchers, "Bash", {"command": "npm install"}) is False


class TestToolFilterIntegration:
    """Tests for ``tool_filter`` wiring through HookDefinition, loader, runner."""

    def test_hook_without_filter_matches_every_tool(self):
        hook = HookDefinition(name="all", event=HookEvent.TOOL_CALL, command="echo hi")
        assert hook.tool_filter is None
        assert hook.matches_tool("Bash", {"command": "git status"}) is True
        assert hook.matches_tool("Write", {"path": "/tmp/x"}) is True
        assert hook.matches_tool("AnyTool") is True

    def test_loader_compiles_if_filter(self):
        data = [
            {
                "name": "git_audit",
                "event": "tool_call",
                "if": "Bash(git:*)",
                "command": "echo git",
            },
        ]
        hooks = load_hooks_from_config(data)
        assert len(hooks) == 1
        hook = hooks[0]
        assert hook.tool_filter is not None
        assert hook.matches_tool("Bash", {"command": "git status"}) is True
        assert hook.matches_tool("Bash", {"command": "npm install"}) is False

    def test_runner_filters_hooks_by_tool(self):
        data = [
            {
                "name": "git_only",
                "event": "tool_call",
                "if": "Bash(git:*)",
                "command": "echo git",
            },
        ]
        runner = HookRunner(hooks=load_hooks_from_config(data))
        # Selected for a git command...
        selected = runner.hooks_for_event(
            HookEvent.TOOL_CALL, tool_name="Bash", tool_args={"command": "git status"}
        )
        assert len(selected) == 1
        # ...but filtered out for a non-git command.
        filtered = runner.hooks_for_event(
            HookEvent.TOOL_CALL, tool_name="Bash", tool_args={"command": "npm install"}
        )
        assert len(filtered) == 0
