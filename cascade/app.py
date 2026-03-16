"""Textual App subclass that wraps the existing CascadeCore from cli.py.

Gets providers, config, hooks, tools for free via the CLI app.
"""


from textual.app import App
from textual.binding import Binding

from .history import HistoryDB
from .state import CascadeState
from .theme import MODES


class CascadeTUI(App):
    """The fullscreen Textual TUI for Cascade."""

    CSS_PATH = "cascade.tcss"

    BINDINGS = [
        Binding("shift+tab", "cycle_mode", "Cycle Mode", show=False),
        Binding("ctrl+c", "exit_app", "Exit", show=False, priority=True),
        Binding("ctrl+d", "exit_app", "Exit", show=False),
    ]

    def __init__(self, cli_app=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cli_app = cli_app
        self.state = CascadeState()
        self.db = HistoryDB()
        self._db_session: dict | None = None

        # Populate state from CLI app
        if cli_app:
            default_provider = cli_app.config.get_default_provider()
            self.state.active_provider = default_provider
            # Find the mode for this provider
            for mode_name, mode_cfg in MODES.items():
                if mode_cfg["provider"] == default_provider:
                    self.state.mode = mode_name
                    break

            # Initialize provider token counters for all known providers
            for name in cli_app.providers:
                if name not in self.state.provider_tokens:
                    self.state.provider_tokens[name] = 0

        # Resolve cwd and branch
        import os
        import subprocess
        self.state.cwd = os.getcwd()
        try:
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            self.state.branch = branch or "main"
        except Exception:
            self.state.branch = ""

    def on_mount(self) -> None:
        self.state.bind(self)

        from .screens.main import MainScreen
        providers = self.cli_app.providers if self.cli_app else {}
        self.push_screen(MainScreen(
            active_provider=self.state.active_provider,
            mode=self.state.mode,
            providers=providers,
        ))

    def action_cycle_mode(self) -> None:
        """Delegate to the current screen."""
        screen = self.screen
        if hasattr(screen, "action_cycle_mode"):
            screen.action_cycle_mode()

    def action_exit_app(self) -> None:
        """Delegate to the current screen or exit directly."""
        screen = self.screen
        if hasattr(screen, "action_exit_app"):
            screen.action_exit_app()
        else:
            self.exit()

    # ------------------------------------------------------------------
    # History persistence
    # ------------------------------------------------------------------

    def ensure_session(self) -> dict:
        """Create a history DB session if one does not exist yet."""
        if self._db_session is not None:
            return self._db_session

        provider = self.state.active_provider
        model = ""
        if self.cli_app:
            prov = self.cli_app.providers.get(provider)
            if prov:
                model = prov.config.model

        self._db_session = self.db.create_session(
            provider=provider, model=model, title="",
        )
        return self._db_session

    def record_message(self, role: str, content: str, token_count: int = 0) -> None:
        """Record a message to the history database."""
        session = self.ensure_session()
        self.db.add_message(
            session["id"], role=role, content=content, token_count=token_count,
        )

        # Auto-title from first user message
        if role == "user" and not session.get("title"):
            title = content[:60]
            self.db.update_session_title(session["id"], title)
            self._db_session["title"] = title
