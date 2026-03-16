"""Tests for /upload and /context command handlers."""

from unittest.mock import MagicMock, patch

from cascade.auth import DetectedCredential
from cascade.context.memory import ContextBuilder
from cascade.commands import CommandHandler, COMMANDS


class TestUploadCommandDef:
    """Verify the command definitions exist."""

    def test_upload_command_registered(self):
        names = [c.name for c in COMMANDS]
        assert "upload" in names

    def test_context_command_registered(self):
        names = [c.name for c in COMMANDS]
        assert "context" in names

    def test_init_command_registered(self):
        names = [c.name for c in COMMANDS]
        assert "init" in names

    def test_login_command_registered(self):
        names = [c.name for c in COMMANDS]
        assert "login" in names


class TestContextCommand:
    """Tests for _cmd_context()."""

    def _make_handler(self):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.context_builder = ContextBuilder()
        app.cli_app = cli_app
        handler = CommandHandler(app)
        # Capture posted messages
        posted = []
        handler._post_system = lambda msg: posted.append(msg)
        return handler, cli_app.context_builder, posted

    def test_context_empty(self):
        handler, ctx, posted = self._make_handler()
        handler._cmd_context([])
        assert len(posted) == 1
        assert "No uploaded context" in posted[0]

    def test_context_with_sources(self):
        handler, ctx, posted = self._make_handler()
        ctx.add_text("hello world", label="test.txt")
        handler._cmd_context([])
        assert len(posted) == 1
        assert "test.txt" in posted[0]
        assert "1" in posted[0]  # source count

    def test_context_clear(self):
        handler, ctx, posted = self._make_handler()
        ctx.add_text("hello world", label="test.txt")
        assert ctx.source_count == 1
        handler._cmd_context(["clear"])
        assert ctx.source_count == 0
        assert "cleared" in posted[0].lower()


class TestUploadCommand:
    """Tests for _cmd_upload()."""

    def _make_handler(self):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.context_builder = ContextBuilder()
        app.cli_app = cli_app
        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda msg: posted.append(msg)
        return handler, cli_app.context_builder, posted

    def test_upload_status_not_running(self):
        handler, ctx, posted = self._make_handler()
        handler._cmd_upload(["status"])
        assert "stopped" in posted[0].lower()

    def test_upload_stop_not_running(self):
        handler, ctx, posted = self._make_handler()
        handler._cmd_upload(["stop"])
        assert "not running" in posted[0].lower()

    def test_upload_stop_running(self):
        handler, ctx, posted = self._make_handler()
        mock_server = MagicMock()
        mock_server.running = True
        handler._upload_server = mock_server
        handler._cmd_upload(["stop"])
        mock_server.stop.assert_called_once()
        assert "stopped" in posted[0].lower()

    def test_upload_already_running(self):
        handler, ctx, posted = self._make_handler()
        mock_server = MagicMock()
        mock_server.running = True
        mock_server.host = "0.0.0.0"
        mock_server.port = 9222
        handler._upload_server = mock_server
        handler._cmd_upload([])
        assert "already running" in posted[0].lower()

    @patch("cascade.commands.CommandHandler._post_system")
    def test_upload_missing_deps(self, mock_post):
        handler, ctx, posted = self._make_handler()
        with patch.dict("sys.modules", {"cascade.web.server": None}):
            # Force ImportError by patching the import
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def fake_import(name, *args, **kwargs):
                if name == "cascade.web.server" or (
                    "web" in name and "server" in name
                ):
                    raise ImportError("no web")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                handler._cmd_upload([])
                # Should report missing deps
                assert any("not installed" in p.lower() for p in posted)

    def test_upload_status_with_sources(self):
        handler, ctx, posted = self._make_handler()
        ctx.add_text("some content", label="doc.txt")
        handler._cmd_upload(["status"])
        assert "1" in posted[0]  # source count


class TestInitCommand:
    """Tests for _cmd_init()."""

    def _make_handler(self):
        app = MagicMock()
        cli_app = MagicMock()
        app.cli_app = cli_app
        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda msg: posted.append(msg)
        return handler, posted

    def test_init_warns_if_exists(self, tmp_path, monkeypatch):
        handler, posted = self._make_handler()
        # Create existing .cascade with agents.yaml
        cascade_dir = tmp_path / ".cascade"
        cascade_dir.mkdir()
        (cascade_dir / "agents.yaml").touch()

        monkeypatch.chdir(tmp_path)
        handler._cmd_init([])
        assert any("already exists" in p for p in posted)

    def test_init_runs_worker_for_new_project(self, tmp_path, monkeypatch):
        handler, posted = self._make_handler()
        monkeypatch.chdir(tmp_path)

        # _run_in_worker is called for new projects; mock it to capture the fn
        captured = []
        handler._run_in_worker = lambda fn, label="": captured.append(fn)

        handler._cmd_init(["general"])
        assert len(captured) == 1

        # Execute the captured function to verify it works
        result = captured[0]()
        assert "general" in result
        assert (tmp_path / ".cascade").is_dir()


class TestConfigReloadCommand:
    """Tests for /config reload behavior."""

    class _DummyConfig:
        def __init__(self):
            self.data = {}

    def _make_handler(self):
        app = MagicMock()
        app.state.provider_tokens = {}
        app.screen = MagicMock()

        cli_app = MagicMock()
        cli_app.providers = {"old": MagicMock()}
        cli_app.config = self._DummyConfig()
        cli_app._apply_detected_credentials = MagicMock()
        cli_app._build_prompt_pipeline = MagicMock(return_value="pipeline")

        def _init():
            cli_app.providers = {"new": MagicMock()}

        cli_app._init_providers = MagicMock(side_effect=_init)

        app.cli_app = cli_app
        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda msg: posted.append(msg)
        return handler, cli_app, posted

    @patch("cascade.auth.detect_all")
    def test_config_reload_refreshes_credentials_and_provider_set(self, mock_detect):
        mock_detect.return_value = ["cred"]
        handler, cli_app, posted = self._make_handler()

        handler._cmd_config(["reload"])

        assert cli_app.credentials == ["cred"]
        cli_app._apply_detected_credentials.assert_called_once()
        cli_app._init_providers.assert_called_once()
        assert set(cli_app.providers.keys()) == {"new"}
        assert posted
        assert "Config reloaded." in posted[-1]
        assert "Added: new" in posted[-1]
        assert "Removed: old" in posted[-1]


class TestLoginCommand:
    """Tests for /login command in TUI mode."""

    def _make_handler(self):
        app = MagicMock()
        app.state.provider_tokens = {}
        app.screen = MagicMock()

        cli_app = MagicMock()
        cli_app.config = MagicMock()
        cli_app.providers = {"gemini": MagicMock()}
        cli_app._init_providers = MagicMock()
        cli_app._build_prompt_pipeline = MagicMock(return_value="pipeline")
        prov = MagicMock()
        prov.ping.return_value = True
        cli_app.get_provider = MagicMock(return_value=prov)

        app.cli_app = cli_app
        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda msg: posted.append(msg)
        handler._run_in_worker = lambda fn, label="": posted.append(fn())
        return handler, cli_app, posted

    @patch("cascade.auth.detect_codex")
    @patch("cascade.auth.detect_claude")
    @patch("cascade.auth.detect_gemini")
    def test_login_status(self, mock_gemini, mock_claude, mock_codex):
        handler, cli_app, posted = self._make_handler()
        mock_gemini.return_value = DetectedCredential(
            provider="gemini",
            source="Gemini CLI",
            token="ya29.test",
            email="user@gmail.com",
            plan="Google One AI Pro",
        )
        mock_claude.return_value = None
        mock_codex.return_value = None

        handler._cmd_login([])
        assert posted
        assert "Auth status:" in posted[-1]
        assert "gemini: detected" in posted[-1]

    def test_login_usage(self):
        handler, cli_app, posted = self._make_handler()
        handler._cmd_login(["bad-provider"])
        assert "Usage: /login <gemini|claude|openai>" in posted[-1]

    @patch("cascade.commands.Path")
    @patch("cascade.auth.detect_gemini")
    def test_login_missing_credential(self, mock_gemini, mock_path):
        handler, cli_app, posted = self._make_handler()
        mock_gemini.return_value = None
        # Prevent filesystem access to ~/.gemini/oauth_creds.json
        mock_path.home.return_value.__truediv__ = lambda *a: MagicMock(is_file=lambda: False)

        handler._cmd_login(["gemini"])
        assert "No gemini CLI credentials found." in posted[-1]

    @patch("cascade.auth.detect_gemini")
    def test_login_syncs_and_verifies(self, mock_gemini):
        handler, cli_app, posted = self._make_handler()
        mock_gemini.return_value = DetectedCredential(
            provider="gemini",
            source="Gemini CLI",
            token="ya29.new",
            email="",
            plan="",
        )

        handler._cmd_login(["gemini"])

        cli_app.config.apply_credential.assert_called_once_with(
            "gemini", "ya29.new", overwrite=True,
        )
        cli_app.config.save.assert_called_once()
        cli_app._init_providers.assert_called_once()
        cli_app._build_prompt_pipeline.assert_called_once()
        assert "synced and verified" in posted[-1]
