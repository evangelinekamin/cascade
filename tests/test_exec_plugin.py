"""Tests for the opt-in shell execution plugin."""

import tempfile
from pathlib import Path

from cascade.plugins.execution import ExecPlugin
from cascade.plugins.registry import get_plugin_registry
from cascade.config import ConfigManager


def test_run_command_returns_output_and_exit_code():
    out = ExecPlugin.run_command("echo cascade-ok")
    assert "Exit code: 0" in out
    assert "cascade-ok" in out


def test_run_command_reports_nonzero_exit():
    out = ExecPlugin.run_command("exit 3")
    assert "Exit code: 3" in out


def test_run_command_truncates_long_output_to_tail():
    out = ExecPlugin.run_command("for i in $(seq 1 5000); do echo line$i; done")
    assert "truncated" in out
    assert "line5000" in out  # the tail is what's kept


def test_exec_plugin_registered_with_run_command_tool():
    registry = get_plugin_registry()
    assert "exec" in registry
    assert "run_command" in registry["exec"]().get_tools()


def test_tools_config_defaults_exec_off():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(str(Path(tmpdir) / "config.yaml"))
        assert manager.get_tools_config()["exec"] is False
        # and the freshly written template carries the off default too
        assert manager.data["tools"]["exec"] is False
