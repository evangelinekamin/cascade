"""Textual App subclass that wraps the existing CascadeCore from cli.py.

Gets providers, config, hooks, tools for free via the CLI app.
"""

import asyncio
import os
import sys
from collections import OrderedDict

from textual.app import App
from textual.binding import Binding

from .history import BranchingSession, HistoryDB
from .state import CascadeState, ProviderChanged, ThinkingChanged
from .swarm.lifecycle import RunLedger
from .theme import MODES, get_provider_theme

_TITLE_SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
_OSC_TITLE = "\x1b]0;{title}\x07"
_DUMB_TERMS = {"dumb", "unknown"}


def _printable(title: str) -> str:
    """Drop control characters: an OSC string ends at the first BEL or ESC.

    Activity labels carry model-authored text, so an unfiltered title could
    close the escape early and leave the rest running as terminal commands.
    """
    return "".join(char for char in title if char.isprintable())


def _emit_terminal_title(title: str) -> bool:
    """Write an OSC title to the terminal; True when the escape was written.

    Textual builds ``App.console`` on a null file, so Rich's set_window_title is
    a no-op inside a running app and the tab keeps whatever the shell left there.
    The escape has to go to the stream the driver itself owns -- stderr, which
    stays attached to the terminal even when stdout is redirected.
    """
    stream = sys.__stderr__
    if stream is None or os.environ.get("TERM", "").strip().lower() in _DUMB_TERMS:
        return False
    try:
        if not stream.isatty():
            return False
        stream.write(_OSC_TITLE.format(title=_printable(title)))
        stream.flush()
    except Exception:
        # Blanket by design: on_unmount calls this ahead of closing the run
        # ledger and the history DB, so a cosmetic title must never be able to
        # abort session cleanup and strand an uncheckpointed WAL.
        return False
    return True


class CascadeTUI(App):
    """The fullscreen Textual TUI for Cascade."""

    CSS_PATH = "cascade.tcss"

    # Cascade has its own slash-command palette (input_frame autocomplete);
    # Textual's stock ctrl+p palette would leak unbranded commands (theme
    # switching, etc.) into the calm-MAGI design.
    ENABLE_COMMAND_PALETTE = False

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
        self.run_ledger = RunLedger(self.db.path)
        self.recovered_run_count = self.run_ledger.mark_interrupted()
        self._db_session: dict | None = None
        self._branching_session: BranchingSession | None = None
        self._title_timer = None
        self._title_idx = 0
        self._title_activities: OrderedDict[str, tuple[str, str]] = OrderedDict()
        # Last string actually written to the terminal; repeats are skipped so
        # the 10Hz activity tick cannot thrash the tab strip.
        self._last_emitted_title: str | None = None

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
        self._clear_window_title()
        try:
            self.run_ledger.close()
        except Exception:
            pass
        try:
            self.db.close()
        except Exception:
            pass

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

    def _project_name(self) -> str:
        """Basename of the working directory."""
        return os.path.basename((self.state.cwd or "").rstrip(os.sep))

    def _title_tokens(self) -> list[str]:
        """Title tokens, most-distinguishing first.

        Terminal tabs are narrow and truncate on the RIGHT, so the project
        directory leads: two cascade tabs in different repos must stay tellable
        apart at ~20 visible characters, which "cascade . build . <project>"
        does not survive. The rest still rides along for wide tabs and the
        window title.
        """
        parts = [
            token
            for token in (self._project_name(), (self.state.mode or "").strip())
            if token
        ]
        parts.append("cascade")
        parts.extend(
            token
            for token in (
                (self.state.active_provider or "").strip(),
                (self.state.session_id or "").strip(),
            )
            if token
        )
        return parts

    def _with_activity(self, base: str) -> str:
        """Append the busiest activity, when there is one.

        Deliberately carries no spinner frame: the title is re-synced on every
        animation tick, and an animated tab makes the strip relayout ~10x a
        second -- flicker exactly while she is scanning tabs. The label alone
        changes rarely, which (with the emit memo) collapses writes to roughly
        one per real state change.
        """
        if not self._title_activities:
            return base
        provider, label = next(reversed(self._title_activities.values()))
        provider = (provider or "").strip()
        if provider and provider != self.state.active_provider and provider not in label:
            return f"{base} . {provider} . {label}"
        return f"{base} . {label}"

    def _terminal_window_title(self) -> str:
        """The session line as the terminal tab shows it, with workspace context."""
        return self._with_activity(" . ".join(self._title_tokens()))

    def _sync_window_title(self) -> None:
        title = self._terminal_window_title()
        if title == self._last_emitted_title:
            return
        if _emit_terminal_title(title):
            self._last_emitted_title = title

    def _clear_window_title(self) -> None:
        """Hand the tab back on exit.

        The previous title cannot be read back portably, so clear it: the
        terminal falls back to its own default rather than keeping a dead
        session's name in the tab strip. Bypasses the emit memo -- shutdown
        must always write, whatever was last sent.
        """
        self._last_emitted_title = None
        _emit_terminal_title("")

    def action_cycle_mode(self) -> None:
        """Delegate to the current screen."""
        screen = self.screen
        if hasattr(screen, "action_cycle_mode"):
            screen.action_cycle_mode()

    def action_exit_app(self) -> None:
        """Delegate Ctrl+C to the current screen, or exit directly.

        The screen owns the interrupt semantics (interrupt a run, clear a filled
        input, then require a second press to exit); a screen without them exits.
        """
        screen = self.screen
        if hasattr(screen, "action_exit_app"):
            screen.action_exit_app()
        else:
            self.exit()

    def on_text_selected(self, event) -> None:
        """Auto-copy a drag-selection on mouse release, like Claude Code.

        The highlight is left in place so the user can see what was copied; it
        clears on the next click (Textual's default). Confirmation is a quiet
        bottom-right note, not a popup.
        """
        try:
            text = self.screen.get_selected_text()
        except Exception:
            text = None
        if not text:
            return
        self.copy_to_clipboard(text)
        self._flash_status(self._copied_message(len(text)))

    def _flash_status(self, message: str) -> None:
        """Show a brief, unobtrusive note in the screen's bottom-right corner."""
        flash = getattr(self.screen, "flash_status", None)
        if callable(flash):
            try:
                flash(message)
            except Exception:
                pass

    @staticmethod
    def _copied_message(n: int) -> str:
        """Human-readable clipboard confirmation for *n* copied characters."""
        return f"Copied {n:,} character{'' if n == 1 else 's'} to clipboard"

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

    def persist_context(self) -> None:
        """Snapshot episodes + compaction summary for the active session.

        Called after compaction events; a session is only created if one
        already exists (no phantom sessions for empty chats).
        """
        if self._db_session is None:
            return
        try:
            chat_roles = lambda m: m.role != "system"
            compacted_chat = [
                m for m in self.state.messages
                if m.metadata.get("compacted") and chat_roles(m)
            ]
            boundary = compacted_chat[-1].content[:200] if compacted_chat else ""
            self.db.save_context(
                self._db_session["id"],
                list(self.state.episodes),
                self.state.compaction_summary,
                compacted_through=len(compacted_chat),
                compaction_boundary=boundary,
                compaction_count=self.state.compaction_count,
            )
        except Exception:
            pass

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

        # Persist the model that actually produced this provider turn, so
        # /resume lands back on the last-used model (not the mode default).
        # This is the one chokepoint every turn passes through, regardless of
        # how the model was selected (command or Shift+Tab).
        if provider:
            cli_app = getattr(self, "cli_app", None)
            prov = cli_app.providers.get(provider) if cli_app else None
            model = getattr(getattr(prov, "config", None), "model", "") or ""
            if model:
                self.db.update_session_model(session["id"], model)

        # Auto-title from first user message
        if role == "user" and not session.get("title"):
            title = content[:60]
            self.db.update_session_title(session["id"], title)
            self._db_session["title"] = title
