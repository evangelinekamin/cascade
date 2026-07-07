"""Tests for the shared worktree-agent primitive."""

import tempfile
from unittest.mock import MagicMock

from cascade.swarm.workspace import WorkspaceTools, run_agent_in_worktree


def test_workspace_tools_read_write_within_root():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        assert "Wrote" in tools.write_file("sub/a.txt", "hello")
        assert tools.read_file("sub/a.txt") == "hello"
        assert any("a.txt" in entry for entry in tools.list_files("sub"))


def test_workspace_tools_rejects_escape():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        # read returns an error string rather than touching anything outside root
        assert "Error" in tools.read_file("../../../../etc/passwd")
        # write outside the root is refused
        assert "Error" in tools.write_file("../escape.txt", "x")


def test_write_file_flags_broken_python_syntax():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        result = tools.write_file("mod.py", "def broken(:\n    pass\n")
        assert "syntax check failed" in result
        assert "line" in result.lower()


def test_write_file_passes_valid_python():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        assert tools.write_file("mod.py", "def ok():\n    return 1\n") == "Wrote mod.py"


def test_write_file_skips_syntax_check_for_non_python():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        # invalid "python" but a .txt -> written, not syntax-checked
        result = tools.write_file("notes.txt", "def broken(:")
        assert "Wrote" in result and "syntax" not in result.lower()


def test_replace_in_file_exact_match():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        tools.write_file("m.py", "x = 1\ny = 2\n")
        result = tools.replace_in_file("m.py", "y = 2", "y = 3")
        assert "replaced" in result
        assert tools.read_file("m.py") == "x = 1\ny = 3\n"


def test_replace_in_file_tolerates_indentation_difference():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        tools.write_file("m.py", "def f():\n        return 1\n")  # 8-space body
        # the model supplies 4-space indent: exact match fails, line-trim succeeds
        result = tools.replace_in_file(
            "m.py", "def f():\n    return 1", "def f():\n    return 2"
        )
        assert "replaced" in result
        assert "return 2" in tools.read_file("m.py")


def test_replace_in_file_reports_not_found():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        tools.write_file("m.py", "a = 1\n")
        assert "not found" in tools.replace_in_file("m.py", "nonexistent", "x")


def test_replace_in_file_refuses_ambiguous_match():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        tools.write_file("m.py", "x = 1\nx = 1\n")
        result = tools.replace_in_file("m.py", "x = 1", "x = 2")
        assert "2 places" in result
        # nothing changed when the target is ambiguous
        assert tools.read_file("m.py") == "x = 1\nx = 1\n"


def test_workspace_tools_build_exposes_the_edit_tool_set():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root).build()
        assert set(tools) == {
            "read_file", "write_file", "append_file", "replace_in_file",
            "list_files", "run_command",
        }


def test_workspace_read_tools_are_concurrency_safe():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root).build()
        # Reads can overlap; mutations and command execution must stay exclusive.
        assert tools["read_file"].concurrency_safe is True
        assert tools["list_files"].concurrency_safe is True
        assert tools["write_file"].concurrency_safe is False
        assert tools["append_file"].concurrency_safe is False
        assert tools["replace_in_file"].concurrency_safe is False
        assert tools["run_command"].concurrency_safe is False


def test_run_command_returns_stdout():
    with tempfile.TemporaryDirectory() as root:
        out = WorkspaceTools(root).run_command("echo hi")
        assert "hi" in out


def test_run_command_runs_in_worktree_cwd():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        assert "Wrote" in tools.write_file("marker.txt", "payload")
        # cwd is the worktree, so a relative path written above is visible here.
        out = tools.run_command("cat marker.txt")
        assert "payload" in out


def test_run_command_surfaces_nonzero_exit_without_raising():
    with tempfile.TemporaryDirectory() as root:
        out = WorkspaceTools(root).run_command(
            "python3 -c \"import sys; sys.stderr.write('boom'); sys.exit(3)\""
        )
        assert "3" in out     # exit code is surfaced, not raised
        assert "boom" in out  # stderr is captured too


def test_run_command_truncates_long_output_keeping_tail():
    with tempfile.TemporaryDirectory() as root:
        out = WorkspaceTools(root).run_command(
            "python3 -c \"print('A' * 6000); print('END_MARKER')\""
        )
        assert "[...truncated...]" in out
        assert "END_MARKER" in out  # the tail (where errors/summaries live) survives
        assert len(out) <= 4100     # capped near 4000 chars, head dropped


def test_run_command_timeout_returns_message_not_raise():
    with tempfile.TemporaryDirectory() as root:
        out = WorkspaceTools(root, command_timeout=0.2).run_command("sleep 5")
        assert "timed out" in out.lower()


def test_run_agent_api_provider_uses_workspace_tools():
    provider = MagicMock()
    provider._use_cli_proxy = False
    provider.ask_with_tools.return_value = ("done", [])

    out = run_agent_in_worktree(provider, "do it", "/tmp/wt", system="sys")

    assert out == "done"
    provider.ask_with_tools.assert_called_once()
    provider.ask_single.assert_not_called()
    # the tool set handed to the provider is the sandboxed workspace tool set
    _args, kwargs = provider.ask_with_tools.call_args
    tools_arg = _args[1] if len(_args) > 1 else kwargs.get("tools")
    assert set(tools_arg) == {
        "read_file", "write_file", "append_file", "replace_in_file",
        "list_files", "run_command",
    }


def test_run_agent_cli_proxy_uses_ask_single_in_workdir():
    provider = MagicMock()
    provider._use_cli_proxy = True
    provider.ask_single.return_value = "cli done"

    out = run_agent_in_worktree(provider, "do it", "/tmp/wt", system="sys")

    assert out == "cli done"
    provider.ask_single.assert_called_once()
    provider.ask_with_tools.assert_not_called()
    provider.working_directory.assert_called_once_with("/tmp/wt")


def test_run_agent_forwards_max_rounds_to_ask_with_tools():
    provider = MagicMock()
    provider._use_cli_proxy = False
    provider.ask_with_tools.return_value = ("done", [])

    run_agent_in_worktree(provider, "do it", "/tmp/wt", system="sys", max_rounds=15)

    _args, kwargs = provider.ask_with_tools.call_args
    assert kwargs.get("max_rounds") == 15


def test_run_agent_defaults_max_rounds_to_15():
    # A local model reads several files before it can write; the build loop must
    # give it enough tool-calling rounds (15), not the ask_with_tools default of 5.
    provider = MagicMock()
    provider._use_cli_proxy = False
    provider.ask_with_tools.return_value = ("done", [])

    run_agent_in_worktree(provider, "do it", "/tmp/wt")

    _args, kwargs = provider.ask_with_tools.call_args
    assert kwargs.get("max_rounds") == 15
