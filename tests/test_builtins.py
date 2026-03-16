"""Tests for built-in agent commands (verify, review, checkpoint)."""

from unittest.mock import MagicMock, patch

from cascade.agents.builtins import _run_cmd, cmd_verify, cmd_review, cmd_checkpoint


class TestRunCmd:
    def test_success(self):
        output, rc = _run_cmd("echo hello")
        assert rc == 0
        assert "hello" in output

    def test_failure(self):
        output, rc = _run_cmd("false")
        assert rc != 0

    def test_stderr_captured(self):
        output, rc = _run_cmd("echo err >&2")
        assert "err" in output

    def test_timeout(self):
        output, rc = _run_cmd("sleep 10", timeout=1)
        assert rc == -1
        assert "timed out" in output


class TestCmdVerify:
    def test_no_commands_configured(self):
        app = MagicMock()
        result = cmd_verify(app, {})
        assert "No verification commands" in result

    @patch("cascade.agents.builtins._run_cmd")
    def test_runs_configured_commands(self, mock_run):
        mock_run.return_value = ("all good", 0)
        app = MagicMock()
        app.get_provider.return_value.ask_single.return_value = "All PASS"
        app.prompt_pipeline.build.return_value = None

        config = {"lint": "ruff check .", "test": "pytest"}
        result = cmd_verify(app, config)

        assert mock_run.call_count == 2
        assert result == "All PASS"

    @patch("cascade.agents.builtins._run_cmd")
    def test_skips_empty_commands(self, mock_run):
        mock_run.return_value = ("ok", 0)
        app = MagicMock()
        app.get_provider.return_value.ask_single.return_value = "summary"
        app.prompt_pipeline.build.return_value = None

        config = {"lint": "ruff check .", "build": "", "test": ""}
        cmd_verify(app, config)

        mock_run.assert_called_once()


class TestCmdReview:
    @patch("cascade.agents.builtins._run_cmd")
    def test_no_changes(self, mock_run):
        mock_run.return_value = ("", 0)
        app = MagicMock()
        result = cmd_review(app)
        assert "No changes" in result

    @patch("cascade.agents.builtins._run_cmd")
    def test_git_diff_failure(self, mock_run):
        mock_run.return_value = ("fatal: not a git repo", 128)
        app = MagicMock()
        result = cmd_review(app)
        assert "git diff failed" in result

    @patch("cascade.agents.builtins._run_cmd")
    def test_sends_diff_to_provider(self, mock_run):
        mock_run.return_value = ("+ new line\n- old line", 0)
        app = MagicMock()
        app.get_provider.return_value.ask_single.return_value = "LGTM"
        app.prompt_pipeline.build.return_value = None

        result = cmd_review(app)
        assert result == "LGTM"
        call_args = app.get_provider.return_value.ask_single.call_args
        assert "+ new line" in call_args[0][0]

    @patch("cascade.agents.builtins._run_cmd")
    def test_with_base_ref(self, mock_run):
        mock_run.return_value = ("diff output", 0)
        app = MagicMock()
        app.get_provider.return_value.ask_single.return_value = "ok"
        app.prompt_pipeline.build.return_value = None

        cmd_review(app, base_ref="main")
        mock_run.assert_called_with("git diff main")


class TestCmdCheckpoint:
    @patch("cascade.agents.builtins._run_cmd")
    def test_tests_fail_skips_commit(self, mock_run):
        mock_run.return_value = ("FAILED test_foo.py", 1)
        app = MagicMock()
        app.config.data = {}

        result = cmd_checkpoint(app, label="v1", test_cmd="pytest")
        assert "Tests failed" in result
        assert "checkpoint skipped" in result
        mock_run.assert_called_once_with("pytest")

    @patch("cascade.agents.builtins._run_cmd")
    def test_tests_pass_commits(self, mock_run):
        # First call: test passes. Second: git add. Third: git commit.
        mock_run.side_effect = [
            ("3 passed", 0),
            ("", 0),
            ("checkpoint: v1", 0),
        ]
        app = MagicMock()
        app.config.data = {}

        result = cmd_checkpoint(app, label="v1", test_cmd="pytest")
        assert "Checkpoint committed" in result
        assert mock_run.call_count == 3

    @patch("cascade.agents.builtins._run_cmd")
    def test_default_test_cmd_from_config(self, mock_run):
        mock_run.return_value = ("ok", 0)
        app = MagicMock()
        app.config.data = {
            "workflows": {"verify": {"test": "make test"}},
        }

        cmd_checkpoint(app, label="x")
        first_call = mock_run.call_args_list[0]
        assert first_call[0][0] == "make test"
