"""Tests for the shared worktree-agent primitive."""

import tempfile
from unittest.mock import MagicMock

from cascade.swarm.workspace import WorkspaceTools, run_agent_in_worktree


def test_workspace_tools_read_write_within_root():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        assert tools.write_file("sub/a.txt", "hello") is True
        assert tools.read_file("sub/a.txt") == "hello"
        assert any("a.txt" in entry for entry in tools.list_files("sub"))


def test_workspace_tools_rejects_escape():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root)
        # read returns an error string rather than touching anything outside root
        assert "Error" in tools.read_file("../../../../etc/passwd")
        # write outside the root is refused
        assert tools.write_file("../escape.txt", "x") is False


def test_workspace_tools_build_exposes_four_tools():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root).build()
        assert set(tools) == {"read_file", "write_file", "append_file", "list_files"}


def test_workspace_read_tools_are_concurrency_safe():
    with tempfile.TemporaryDirectory() as root:
        tools = WorkspaceTools(root).build()
        # Reads can overlap; mutations must stay exclusive.
        assert tools["read_file"].concurrency_safe is True
        assert tools["list_files"].concurrency_safe is True
        assert tools["write_file"].concurrency_safe is False
        assert tools["append_file"].concurrency_safe is False


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
    assert set(tools_arg) == {"read_file", "write_file", "append_file", "list_files"}


def test_run_agent_cli_proxy_uses_ask_single_in_workdir():
    provider = MagicMock()
    provider._use_cli_proxy = True
    provider.ask_single.return_value = "cli done"

    out = run_agent_in_worktree(provider, "do it", "/tmp/wt", system="sys")

    assert out == "cli done"
    provider.ask_single.assert_called_once()
    provider.ask_with_tools.assert_not_called()
    provider.working_directory.assert_called_once_with("/tmp/wt")
