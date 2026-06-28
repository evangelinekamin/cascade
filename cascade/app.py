"""Textual App subclass that wraps the existing CascadeCore from cli.py.

Gets providers, config, hooks, tools for free via the CLI app.
"""

import asyncio
from collections import OrderedDict

from textual.app import App
from textual.binding import Binding

from .history import BranchingSession, HistoryDB
from .state import CascadeState, ProviderChanged, ThinkingChanged
from .theme import MODES, get_provider_theme

_TITLE_SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


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
        self._branching_session: BranchingSession | None = None
        self._title_timer = None
        self._title_idx = 0
        self._title_activities: OrderedDict[str, tuple[str, str]] = OrderedDict()

        # Populate state from CLI app
        if cli_app:
            available_providers = list(cli_app.providers.keys())
            default_provider = cli_app.config.get_default_provider()
            if available_providers and default_provider not in cli_app.providers:
                default_provider = available_providers[0]
            self.state.active_provider = default_provider
            configured_mode = cli_app.config.get_default_mode_for_provider(default_provider)
            if isinstance(configured_mode, str) and configured_mode in MODES:
                self.state.mode = configured_mode
            else:
                self.state.mode = get_provider_theme(default_provider).default_mode
            prov = cli_app.providers.get(default_provider)
            if prov is not None:
                model = cli_app.config.get_model_for(default_provider, self.state.mode, fast=False)
                if isinstance(model, str) and model:
                    prov.config.model = model

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
        self._sync_window_title()

        # Fire SESSION_START hook
        if self.cli_app:
            from .hooks import HookEvent, HookContext
            self.cli_app.hook_runner.emit(
                HookEvent.SESSION_START,
                HookContext(
                    event=HookEvent.SESSION_START.value,
                    provider=self.state.active_provider,
                    mode=self.state.mode,
                    session_id=self.state.session_id,
                ),
            )

        from .screens.main import MainScreen
        providers = self.cli_app.providers if self.cli_app else {}
        self.push_screen(MainScreen(
            active_provider=self.state.active_provider,
            mode=self.state.mode,
            providers=providers,
        ))

    def on_unmount(self) -> None:
        if self._title_timer is not None:
            self._title_timer.stop()
            self._title_timer = None
        self._sync_window_title()

    def on_provider_changed(self, event: ProviderChanged) -> None:
        del event
        self._sync_window_title()

    def on_thinking_changed(self, event: ThinkingChanged) -> None:
        if event.thinking:
            self.start_title_activity("chat", event.provider, event.thought or "thinking")
        else:
            self.stop_title_activity("chat")

    def start_title_activity(self, source: str, provider: str, label: str) -> None:
        """Register a busy activity that should animate in the terminal title."""
        normalized_label = self._normalize_title_label(label)
        normalized_provider = (provider or self.state.active_provider or "cascade").strip()
        self._title_activities.pop(source, None)
        self._title_activities[source] = (normalized_provider, normalized_label)
        self._ensure_title_timer()
        self._sync_window_title()

    def update_title_activity(self, source: str, label: str, provider: str | None = None) -> None:
        """Refresh the label for an existing title activity."""
        current_provider = provider
        if current_provider is None:
            current_provider = self._title_activities.get(source, (self.state.active_provider, ""))[0]
        self.start_title_activity(source, current_provider, label)

    def stop_title_activity(self, source: str) -> None:
        """Remove a busy activity from the terminal title."""
        self._title_activities.pop(source, None)
        if not self._title_activities and self._title_timer is not None:
            self._title_timer.stop()
            self._title_timer = None
            self._title_idx = 0
        self._sync_window_title()

    def _tick_title(self) -> None:
        if not self._title_activities:
            if self._title_timer is not None:
                self._title_timer.stop()
                self._title_timer = None
            self._title_idx = 0
            self._sync_window_title()
            return
        self._title_idx = (self._title_idx + 1) % len(_TITLE_SPINNER_FRAMES)
        self._sync_window_title()

    def _ensure_title_timer(self) -> None:
        if self._title_timer is None:
            try:
                asyncio.get_running_loop()
                self._title_timer = self.set_interval(0.1, self._tick_title)
            except RuntimeError:
                self._title_timer = None

    @staticmethod
    def _normalize_title_label(label: str) -> str:
        compact = " ".join(str(label or "").split()).strip() or "working"
        return compact if len(compact) <= 72 else f"{compact[:69]}..."

    def _base_window_title(self) -> str:
        parts = ["cascade"]
        provider = (self.state.active_provider or "").strip()
        session_id = (self.state.session_id or "").strip()
        if provider:
            parts.append(provider)
        if session_id:
            parts.append(session_id)
        return " . ".join(parts)

    def _formatted_window_title(self) -> str:
        base = self._base_window_title()
        if not self._title_activities:
            return base
        provider, label = next(reversed(self._title_activities.values()))
        frame = _TITLE_SPINNER_FRAMES[self._title_idx % len(_TITLE_SPINNER_FRAMES)]
        provider = (provider or "").strip()
        if provider and provider != self.state.active_provider and provider not in label:
            return f"{frame} {base} . {provider} . {label}"
        return f"{frame} {base} . {label}"

    def _sync_window_title(self) -> None:
        try:
            self.console.set_window_title(self._formatted_window_title())
        except Exception:
            pass

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

        session = self.db.create_session(
            provider=provider,
            model=model,
            title="",
            session_id=self.state.session_id,
        )
        return self.adopt_session(session)

    def adopt_session(self, session: dict) -> dict:
        """Bind application state to an existing history session."""
        self._db_session = session
        self.state.set_session_id(session["id"])
        self._branching_session = BranchingSession(self.db, session["id"])
        self._sync_window_title()
        return session

    def get_branching_session(self) -> BranchingSession:
        """Return the branching wrapper for the active history session."""
        session = self.ensure_session()
        if self._branching_session is None or self._db_session is not session:
            self._branching_session = BranchingSession(self.db, session["id"])
        return self._branching_session

    def record_message(self, role: str, content: str, token_count: int = 0) -> None:
        """Record a message to the history database."""
        session = self.ensure_session()
        branching = self.get_branching_session()
        provider = ""
        if role not in {"user", "system", "assistant"}:
            provider = role
        elif role == "assistant":
            provider = session.get("provider", "")
        branching.add_message(
            role=role,
            content=content,
            provider=provider,
            token_count=token_count,
        )

        # Auto-title from first user message
        if role == "user" and not session.get("title"):
            title = content[:60]
            self.db.update_session_title(session["id"], title)
            self._db_session["title"] = title
