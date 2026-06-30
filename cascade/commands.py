"""Slash command parsing and dispatch for the Cascade TUI.

Commands post state messages rather than manipulating widgets directly.
"""

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .theme import MODES, PALETTE, PROVIDERS


@dataclass(frozen=True)
class CommandDef:
    """Definition of a slash command for autocomplete."""
    name: str
    usage: str
    description: str


@dataclass
class ProgressHandle:
    """Mounted command progress row plus its title-spinner source."""

    indicator: object | None
    title_source: str | None
    provider: str


# Canonical list of available commands (used by autocomplete and /help)
COMMANDS: tuple[CommandDef, ...] = (
    CommandDef("exit", "/exit", "Exit cascade"),
    CommandDef("quit", "/quit", "Exit cascade"),
    CommandDef("fast", "/fast", "Toggle fast model for current provider"),
    CommandDef("model", "/model <provider|reset>", "Switch active provider"),
    CommandDef("mode", "/mode <name>", "Switch mode (design, plan, build, test)"),
    CommandDef("mode-provider", "/mode-provider <mode> <provider>", "Assign a provider to a mode"),
    CommandDef("mode-model", "/mode-model <mode> <model|reset>", "Set or clear a mode-specific model"),
    CommandDef("providers", "/providers", "List available providers"),
    CommandDef("agent", "/agent <name> <prompt>", "Run a named agent"),
    CommandDef("agents", "/agents", "List available agents"),
    CommandDef("workflow", "/workflow <name> <prompt>", "Run a named workflow"),
    CommandDef("verify", "/verify", "Run lint/test/build and summarize"),
    CommandDef("review", "/review [ref]", "Code review of uncommitted changes"),
    CommandDef("checkpoint", "/checkpoint <label>", "Test-gated git commit"),
    CommandDef("shannon", "/shannon <url>", "Launch Shannon pentesting"),
    CommandDef("init", "/init [type]", "Initialize .cascade/ project config"),
    CommandDef("upload", "/upload [stop|status]", "Start drag-and-drop upload server"),
    CommandDef("context", "/context [clear]", "Show or clear uploaded context"),
    CommandDef("login", "/login [provider]", "Sync provider auth from installed CLIs"),
    CommandDef("config", "/config reload", "Reload config from disk"),
    CommandDef("clear", "/clear", "Clear chat history from screen"),
    CommandDef("compact", "/compact", "Compact conversation memory now"),
    CommandDef("history", "/history [limit]", "List recent chat sessions"),
    CommandDef("resume", "/resume <id>", "Resume a previous session"),
    CommandDef("export", "/export [id]", "Export session messages to a file"),
    CommandDef("swarm", "/swarm <task>", "Multi-model swarm dispatch"),
    CommandDef("solve", "/solve <task>", "Code a task in an isolated worktree, verified by tests"),
    CommandDef("pipeline", "/pipeline <objective>", "Decompose a build into ordered steps, each verified by tests"),
    CommandDef(
        "compete",
        "/compete [--providers a,b] [--judge x] <task>",
        "Run the same task across providers and pick a winner",
    ),
    CommandDef(
        "compete-code",
        "/compete-code [--providers a,b] [--judge x] <task>",
        "Run a coding task in isolated worktrees and keep the winning workspace",
    ),
    CommandDef("episodes", "/episodes", "Show episode history"),
    CommandDef("tree", "/tree", "Show session branch tree"),
    CommandDef("branch", "/branch [label]", "Create a branch at current point"),
    CommandDef("mark", "/mark [label]", "Insert a bookmark separator"),
    CommandDef("time", "/time", "Show current time"),
    CommandDef("help", "/help", "Show available commands"),
)

_PROGRESS_DETAIL_RE = re.compile(r"^\[(?P<provider>[^\]]+)\]\s*(?P<message>.*)$")


def get_matching_commands(prefix: str) -> list[CommandDef]:
    """Return commands whose name starts with prefix (without the /)."""
    prefix = prefix.lower()
    return [c for c in COMMANDS if c.name.startswith(prefix) and c.name != "quit"]


class CommandHandler:
    """Parses and dispatches slash commands from user input."""

    def __init__(self, app) -> None:
        self.app = app
        self._shannon = None
        self._upload_server = None

    def is_command(self, text: str) -> bool:
        return text.startswith("/")

    def handle(self, text: str) -> bool:
        """Handle the command. Returns True if it was a command."""
        if not self.is_command(text):
            return False

        parts = text.split()
        cmd = parts[0][1:].lower()
        args = parts[1:]

        handler = {
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
            "fast": self._cmd_fast,
            "model": self._cmd_model,
            "mode": self._cmd_mode,
            "mode-provider": self._cmd_mode_provider,
            "mode-model": self._cmd_mode_model,
            "help": self._cmd_help,
            "providers": self._cmd_providers,
            "agent": self._cmd_agent,
            "agents": self._cmd_agents,
            "workflow": self._cmd_workflow,
            "verify": self._cmd_verify,
            "review": self._cmd_review,
            "checkpoint": self._cmd_checkpoint,
            "shannon": self._cmd_shannon,
            "init": self._cmd_init,
            "upload": self._cmd_upload,
            "context": self._cmd_context,
            "login": self._cmd_login,
            "config": self._cmd_config,
            "clear": self._cmd_clear,
            "compact": self._cmd_compact,
            "history": self._cmd_history,
            "resume": self._cmd_resume,
            "export": self._cmd_export,
            "swarm": self._cmd_swarm,
            "solve": self._cmd_solve,
            "pipeline": self._cmd_pipeline,
            "compete": self._cmd_compete,
            "compete-code": self._cmd_compete_code,
            "episodes": self._cmd_episodes,
            "tree": self._cmd_tree,
            "branch": self._cmd_branch,
            "mark": self._cmd_mark,
            "time": self._cmd_time,
        }.get(cmd)

        if handler:
            handler(args)
        else:
            self.app.notify(f"Unknown command: /{cmd}")

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _post_system(self, text: str, *, force_scroll: bool = True) -> None:
        """Mount a system message in the chat."""
        try:
            from .widgets.message import ChatHistory, MessageWidget
            chat = self.app.screen.query_one(ChatHistory)
            chat.mount(MessageWidget("system", text))
            screen = getattr(self.app, "screen", None)
            if screen is not None and hasattr(screen, "_scroll_chat_end"):
                screen._scroll_chat_end(chat, force=force_scroll)
            else:
                chat.scroll_end(animate=False)
        except Exception:
            self.app.notify(text)

    def _mount_progress_indicator(self, label: str):
        """Attach a live spinner row for long-running slash commands."""
        provider = getattr(getattr(self.app, "state", None), "active_provider", "gemini")
        title_source = f"command:{id(self)}:{datetime.datetime.now().timestamp()}"
        starter = getattr(self.app, "start_title_activity", None)
        if callable(starter):
            try:
                starter(title_source, provider, label)
            except Exception:
                title_source = None
        try:
            from .widgets.message import ChatHistory, ThinkingIndicator

            chat = self.app.screen.query_one(ChatHistory)
            indicator = ThinkingIndicator(provider=provider, label=label)
            chat.mount(indicator)
            screen = getattr(self.app, "screen", None)
            if screen is not None and hasattr(screen, "_scroll_chat_end"):
                screen._scroll_chat_end(chat, force=True)
            else:
                chat.scroll_end(animate=False)
            return ProgressHandle(indicator=indicator, title_source=title_source, provider=provider)
        except Exception:
            return ProgressHandle(indicator=None, title_source=title_source, provider=provider)

    def _set_progress_indicator_label(self, handle, label: str) -> None:
        """Safely update a long-running command spinner label."""
        if handle is None:
            return
        updater = getattr(handle, "set_label", None)
        if not callable(updater):
            updater = getattr(getattr(handle, "indicator", None), "set_label", None)
        if callable(updater):
            try:
                updater(label)
            except Exception:
                pass
        activity_updater = getattr(self.app, "update_title_activity", None)
        if callable(activity_updater) and getattr(handle, "title_source", None):
            try:
                activity_updater(handle.title_source, label, getattr(handle, "provider", None))
            except Exception:
                pass

    def _clear_progress_indicator(self, handle) -> None:
        """Remove a live spinner row if it was mounted."""
        if handle is None:
            return
        stopper = getattr(self.app, "stop_title_activity", None)
        title_source = getattr(handle, "title_source", None)
        if callable(stopper) and title_source:
            try:
                stopper(title_source)
            except Exception:
                pass
        try:
            remover = getattr(handle, "remove", None)
            if not callable(remover):
                indicator = getattr(handle, "indicator", None)
                remover = getattr(indicator, "remove", None)
            if callable(remover):
                remover()
        except Exception:
            pass

    @staticmethod
    def _format_progress_label(
        providers: list[str],
        provider_states: dict[str, str],
        judge_status: str = "",
    ) -> str:
        """Render a compact one-line status summary for multi-provider commands."""
        parts = [
            f"{provider} {provider_states.get(provider, 'queued')}"
            for provider in providers
        ]
        if judge_status:
            parts.append(judge_status)
        return " | ".join(parts)

    @staticmethod
    def _update_progress_states(
        stage: str,
        detail: str,
        provider_states: dict[str, str],
        judge_provider: str | None = None,
    ) -> str:
        """Apply an on_progress event to the provider-state summary."""
        if stage == "judging":
            return f"judge {judge_provider or 'running'}"

        match = _PROGRESS_DETAIL_RE.match(detail or "")
        if not match:
            return ""

        provider = match.group("provider").strip().lower()
        message = match.group("message").strip().lower()

        if stage == "workspace":
            provider_states[provider] = "prepared"
        elif stage == "competing":
            provider_states[provider] = "running"
        elif stage == "result":
            if message.startswith("done"):
                provider_states[provider] = "done"
            elif message.startswith("failed"):
                provider_states[provider] = "failed"
            else:
                provider_states[provider] = message or "done"

        return ""

    def _record_history_message(self, role: str, content: str, token_count: int = 0) -> None:
        """Persist a message to history when the app exposes record_message()."""
        recorder = getattr(self.app, "record_message", None)
        if not callable(recorder):
            return
        try:
            recorder(role, content, token_count=token_count)
        except TypeError:
            recorder(role, content, token_count)
        except Exception:
            pass

    def _seed_session_title(self, title: str) -> None:
        """Set a session title for slash-command-only sessions when needed."""
        session = getattr(self.app, "_db_session", None)
        db = getattr(self.app, "db", None)
        if not session or session.get("title") or db is None:
            return
        short = title[:60]
        try:
            db.update_session_title(session["id"], short)
            session["title"] = short
        except Exception:
            pass

    def _record_command_line(self, command_line: str, title: str | None = None) -> None:
        """Persist the slash command invocation as a user message."""
        self._record_history_message("user", command_line)
        if title:
            self._seed_session_title(title)

    @staticmethod
    def _history_provider_for_message(message: dict, session_provider: str) -> str:
        """Resolve the provider label for a persisted history message."""
        stored_role = str(message.get("role", "assistant"))
        if stored_role == "assistant":
            return str(message.get("metadata", {}).get("provider") or session_provider or "assistant")
        return stored_role

    @classmethod
    def _state_role_for_history_message(cls, message: dict, session_provider: str) -> str:
        """Map a persisted history role onto the in-memory chat role."""
        stored_role = str(message.get("role", "assistant"))
        if stored_role == "user":
            return "you"
        if stored_role == "assistant":
            return cls._history_provider_for_message(message, session_provider)
        return stored_role

    @staticmethod
    def _configured_providers(cli_app) -> list[str]:
        """Return configured providers from the live CLI app."""
        return list(getattr(cli_app, "providers", {}).keys())

    @staticmethod
    def _mode_for_provider(cli_app, provider: str) -> str:
        """Return the default mode associated with a provider."""
        if cli_app is not None and hasattr(cli_app, "config"):
            mode_name = cli_app.config.get_default_mode_for_provider(provider)
            if isinstance(mode_name, str) and mode_name in MODES:
                return mode_name
        for mode_name, mode_cfg in MODES.items():
            if mode_cfg.get("provider") == provider:
                return mode_name
        return "design"

    @staticmethod
    def _mode_provider(cli_app, mode_name: str) -> str:
        """Return the configured provider for a mode."""
        if cli_app is not None and hasattr(cli_app, "config"):
            provider_name = cli_app.config.get_mode_provider(mode_name)
            if isinstance(provider_name, str) and provider_name in PROVIDERS:
                return provider_name
        return MODES.get(mode_name, {}).get("provider", "gemini")

    @classmethod
    def _available_modes(cls, cli_app) -> tuple[str, ...]:
        """Return modes available with the currently configured providers."""
        configured = cls._configured_providers(cli_app)
        if cli_app is not None and hasattr(cli_app, "config"):
            modes = cli_app.config.get_available_modes(configured)
            if isinstance(modes, tuple) and all(isinstance(mode, str) for mode in modes):
                return tuple(mode for mode in modes if mode in MODES)
        from .theme import get_available_modes
        return get_available_modes(configured)

    @staticmethod
    def _apply_provider_model(app, provider_name: str, mode_name: str, *, fast: bool = False) -> str:
        """Sync a provider object's active model from config and return the chosen model."""
        cli_app = getattr(app, "cli_app", None)
        if cli_app is None:
            return ""
        prov = cli_app.providers.get(provider_name)
        if prov is None:
            return ""
        model = cli_app.config.get_model_for(provider_name, mode_name, fast=fast)
        if isinstance(model, str) and model:
            prov.config.model = model
        return model or str(getattr(getattr(prov, "config", None), "model", "") or "")

    def _set_active_mode_and_provider(self, provider_name: str, mode_name: str, *, fast: bool = False) -> None:
        """Apply provider/mode selection to state, widgets, and model config."""
        self._apply_provider_model(self.app, provider_name, mode_name, fast=fast)
        self.app.state.set_provider(provider_name, mode_name)
        self.app.state.fast_mode = fast
        try:
            screen = self.app.screen
            screen._active_provider = provider_name
            screen._mode = mode_name
            inp = screen.query_one("InputFrame")
            inp.active_provider = provider_name
            inp.mode = mode_name
        except Exception:
            pass
        try:
            from .widgets.header import ProviderGhostTable
            self.app.screen.query_one(ProviderGhostTable).set_active(provider_name)
            self.app.screen.query_one(ProviderGhostTable).refresh()
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_compete_args(
        args: list[str],
    ) -> tuple[list[str] | None, str | None, str | None]:
        """Parse optional provider/judge flags from /compete args."""
        providers: list[str] | None = None
        judge: str | None = None
        index = 0

        while index < len(args):
            token = args[index]
            if token == "--providers":
                index += 1
                if index >= len(args):
                    return None, None, None
                raw = args[index]
                providers = [name.strip().lower() for name in raw.split(",") if name.strip()]
                if not providers:
                    return None, None, None
                index += 1
                continue
            if token.startswith("--providers="):
                raw = token.split("=", 1)[1]
                providers = [name.strip().lower() for name in raw.split(",") if name.strip()]
                if not providers:
                    return None, None, None
                index += 1
                continue
            if token == "--judge":
                index += 1
                if index >= len(args):
                    return None, None, None
                judge = args[index].strip().lower() or None
                if judge is None:
                    return None, None, None
                index += 1
                continue
            if token.startswith("--judge="):
                judge = token.split("=", 1)[1].strip().lower() or None
                if judge is None:
                    return None, None, None
                index += 1
                continue
            break

        objective = " ".join(args[index:]).strip()
        if not objective:
            return None, None, None
        return providers, judge, objective

    def _resolve_compete_request(
        self,
        args: list[str],
        command_name: str,
        label: str,
    ):
        providers_arg, judge_arg, objective = self._parse_compete_args(args)
        if objective is None:
            self._post_system(
                f"Usage: /{command_name} [--providers a,b] [--judge x] <task description>"
            )
            return None

        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self._post_system(f"{label} requires CLI app.")
            return None

        available = list(cli_app.providers.keys())
        if len(available) < 2:
            self._post_system(
                f"{label} needs 2+ providers. Have: {available}. "
                f"Use /login to add more."
            )
            return None

        selected = providers_arg or available
        deduped: list[str] = []
        for provider_name in selected:
            if provider_name not in deduped:
                deduped.append(provider_name)
        selected = deduped

        invalid = [provider_name for provider_name in selected if provider_name not in cli_app.providers]
        if invalid:
            self._post_system(
                f"{label} provider(s) not found: {', '.join(invalid)}. "
                f"Available: {', '.join(available)}"
            )
            return None
        if len(selected) < 2:
            self._post_system(
                f"{label} needs 2+ selected providers. Selected: {', '.join(selected)}"
            )
            return None
        if judge_arg and judge_arg not in cli_app.providers:
            self._post_system(
                f"{label} judge '{judge_arg}' not found. Available: {', '.join(available)}"
            )
            return None

        return cli_app, selected, judge_arg, objective

    @staticmethod
    def _diff_summary(diff_stat: str, changed_files: list[str]) -> str:
        """Return a short one-line diff summary for competition output."""
        lines = [line.strip() for line in diff_stat.splitlines() if line.strip()]
        if lines:
            return lines[-1]
        file_count = len(changed_files)
        if file_count == 1:
            return "1 file changed"
        if file_count > 1:
            return f"{file_count} files changed"
        return "no diff"

    def _get_shannon(self):
        """Lazy-init Shannon integration via entry point discovery."""
        if self._shannon is None:
            from .integrations import get_integration
            cls = get_integration("shannon")
            if cls is None:
                return None
            cli_app = getattr(self.app, "cli_app", None)
            cfg = {}
            if cli_app:
                cfg = cli_app.config.get_integrations_config().get("shannon", {})
            self._shannon = cls(
                config_path=cfg.get("path", ""),
                print_fn=self._post_system,
            )
        return self._shannon

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_exit(self, args: list[str]) -> None:
        self.app.action_exit_app()

    def _cmd_fast(self, args: list[str]) -> None:
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        provider_name = self.app.state.active_provider
        prov = cli_app.providers.get(provider_name)
        if prov is None:
            self.app.notify(f"Provider '{provider_name}' not active")
            return

        provider_data = cli_app.config.data.get("providers", {}).get(provider_name, {})
        fast_model = str(provider_data.get("fast_model", "") or "").strip()

        if not fast_model:
            self.app.notify(f"No fast_model configured for {provider_name}")
            return

        # Toggle
        state = self.app.state
        if state.fast_mode:
            label = self._apply_provider_model(
                self.app,
                provider_name,
                state.mode,
                fast=False,
            )
            state.fast_mode = False
        else:
            label = self._apply_provider_model(
                self.app,
                provider_name,
                state.mode,
                fast=True,
            ) or fast_model
            state.fast_mode = True

        # Update ghost table to show new model
        try:
            from .widgets.header import ProviderGhostTable
            self.app.screen.query_one(ProviderGhostTable).refresh()
        except Exception:
            pass

        mode_label = "fast" if state.fast_mode else "default"
        self.app.notify(f"{provider_name}: {label} ({mode_label})")

    def _cmd_model(self, args: list[str]) -> None:
        if not args:
            self.app.notify("Usage: /model <provider|reset>")
            return
        name = args[0].lower()
        cli_app = getattr(self.app, "cli_app", None)
        configured = self._configured_providers(cli_app)
        if name == "reset":
            mode_name = self.app.state.mode
            provider_name = self._mode_provider(cli_app, mode_name)
            if configured and provider_name not in configured:
                available_modes = self._available_modes(cli_app)
                if not available_modes:
                    self.app.notify("No configured providers available")
                    return
                mode_name = available_modes[0]
                provider_name = self._mode_provider(cli_app, mode_name)
            self._set_active_mode_and_provider(provider_name, mode_name, fast=False)
            self.app.notify(f"Provider reset to {provider_name} for {mode_name} mode")
            return
        if configured:
            if name not in configured:
                self.app.notify(
                    f"Provider '{name}' is not configured. Available: {', '.join(configured)}"
                )
                return
        elif name not in PROVIDERS:
            self.app.notify(f"Provider '{name}' not found. Available: {', '.join(PROVIDERS)}")
            return
        mode_name = self._mode_for_provider(cli_app, name)
        self._set_active_mode_and_provider(name, mode_name, fast=False)
        self.app.notify(f"Switched to {name} ({mode_name} mode)")

    def _cmd_mode(self, args: list[str]) -> None:
        if not args:
            self.app.notify("Usage: /mode <name>")
            return
        mode_name = args[0].lower()
        cli_app = getattr(self.app, "cli_app", None)
        configured = self._configured_providers(cli_app)
        available_modes = self._available_modes(cli_app)
        if mode_name not in MODES:
            self.app.notify(f"Mode '{mode_name}' not found. Available: {', '.join(available_modes)}")
            return
        provider = self._mode_provider(cli_app, mode_name)
        if configured and provider not in configured:
            self.app.notify(
                f"Mode '{mode_name}' is unavailable because provider '{provider}' is not configured. "
                f"Available modes: {', '.join(available_modes)}"
            )
            return
        self._set_active_mode_and_provider(provider, mode_name, fast=False)
        self.app.notify(f"Switched to {mode_name} mode")

    def _cmd_mode_provider(self, args: list[str]) -> None:
        if len(args) != 2:
            self.app.notify("Usage: /mode-provider <mode> <provider>")
            return
        mode_name = args[0].lower()
        provider_name = args[1].lower()
        if mode_name not in MODES:
            self.app.notify(f"Mode '{mode_name}' not found. Available: {', '.join(MODES)}")
            return
        if provider_name not in PROVIDERS:
            self.app.notify(f"Provider '{provider_name}' not found. Available: {', '.join(PROVIDERS)}")
            return

        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        modes = cli_app.config.data.setdefault("modes", {})
        entry = modes.setdefault(mode_name, {})
        entry["provider"] = provider_name
        cli_app.config.save()

        if self.app.state.mode == mode_name:
            configured = self._configured_providers(cli_app)
            if configured and provider_name in configured:
                self._set_active_mode_and_provider(provider_name, mode_name, fast=False)
                self.app.notify(f"{mode_name} mode now uses {provider_name}")
                return

        self.app.notify(f"Saved {mode_name} provider: {provider_name}")

    def _cmd_mode_model(self, args: list[str]) -> None:
        if len(args) < 2:
            self.app.notify("Usage: /mode-model <mode> <model|reset>")
            return
        mode_name = args[0].lower()
        if mode_name not in MODES:
            self.app.notify(f"Mode '{mode_name}' not found. Available: {', '.join(MODES)}")
            return

        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        raw_model = " ".join(args[1:]).strip()
        modes = cli_app.config.data.setdefault("modes", {})
        entry = modes.setdefault(mode_name, {})
        if raw_model.lower() == "reset":
            entry["model"] = ""
            cli_app.config.save()
            provider_name = self._mode_provider(cli_app, mode_name)
            if self.app.state.mode == mode_name and self.app.state.active_provider == provider_name:
                self._set_active_mode_and_provider(provider_name, mode_name, fast=False)
            self.app.notify(f"Cleared {mode_name} mode model override")
            return

        entry["model"] = raw_model
        cli_app.config.save()

        provider_name = self._mode_provider(cli_app, mode_name)
        if self.app.state.mode == mode_name and self.app.state.active_provider == provider_name:
            self._set_active_mode_and_provider(provider_name, mode_name, fast=False)
            self.app.notify(f"{mode_name} mode now uses {raw_model}")
            return

        self.app.notify(f"Saved {mode_name} mode model: {raw_model}")

    def _cmd_help(self, args: list[str]) -> None:
        lines = []
        for c in COMMANDS:
            if c.name == "quit":
                continue
            lines.append(f"  {c.usage:<22s} {c.description}")
        self._post_system("Commands:\n" + "\n".join(lines))

    def _cmd_providers(self, args: list[str]) -> None:
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app and cli_app.providers:
            lines = []
            for name, prov in cli_app.providers.items():
                model = prov.config.model if hasattr(prov, "config") else "?"
                active = " (active)" if name == self.app.state.active_provider else ""
                lines.append(f"  {name}: {model}{active}")
            text = "Available providers:\n" + "\n".join(lines)
        else:
            text = "No providers configured."
        self._post_system(text)

    # ------------------------------------------------------------------
    # Agent / Workflow commands
    # ------------------------------------------------------------------

    def _run_in_worker(self, fn, label: str = "agent") -> None:
        """Run *fn* in a worker thread and post the result as a system message.

        Uses the same ThinkingIndicator + StreamMessage pattern as
        _send_to_provider.
        """
        try:
            from .widgets.message import ChatHistory, ThinkingIndicator

            chat = self.app.screen.query_one(ChatHistory)
            provider = getattr(getattr(self.app, "state", None), "active_provider", "gemini")

            thinking = ThinkingIndicator(provider=provider, label=label)
            chat.mount(thinking)
            chat.scroll_end(animate=False)
            title_source = f"worker:{label}:{datetime.datetime.now().timestamp()}"
            starter = getattr(self.app, "start_title_activity", None)
            if callable(starter):
                try:
                    starter(title_source, provider, label)
                except Exception:
                    title_source = None

            def _worker():
                try:
                    result = fn()
                    self.app.call_from_thread(self._finish_worker, thinking, result, title_source)
                except Exception as e:
                    self.app.call_from_thread(self._finish_worker, thinking, f"Error: {e}", title_source)

            self.app.screen.run_worker(_worker, thread=True, exclusive=False)
        except Exception:
            # Fallback: run synchronously
            try:
                result = fn()
                self._post_system(result)
            except Exception as e:
                self._post_system(f"Error: {e}")

    def _finish_worker(self, thinking, result: str, title_source: str | None = None) -> None:
        """Remove thinking indicator and post the result."""
        stopper = getattr(self.app, "stop_title_activity", None)
        if callable(stopper) and title_source:
            try:
                stopper(title_source)
            except Exception:
                pass
        try:
            thinking.remove()
        except Exception:
            pass
        self._post_system(result)

    def _cmd_agent(self, args: list[str]) -> None:
        if len(args) < 2:
            self._post_system("Usage: /agent <name> <prompt>")
            return

        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        name = args[0]
        if name not in cli_app.agents:
            available = ", ".join(sorted(cli_app.agents)) or "(none)"
            self._post_system(f"Unknown agent '{name}'. Available: {available}")
            return

        prompt = " ".join(args[1:])
        agent = cli_app.agents[name]

        def _do():
            return cli_app._agent_runner.run(agent, prompt)

        self._run_in_worker(_do, label=f"agent:{name}")

    def _cmd_agents(self, args: list[str]) -> None:
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None or not cli_app.agents:
            self._post_system("No agents configured. Add them to .cascade/agents.yaml")
            return

        lines = ["Available agents:"]
        for agent in cli_app.agents.values():
            lines.append(f"  {agent.to_summary()}")
        self._post_system("\n".join(lines))

    def _cmd_workflow(self, args: list[str]) -> None:
        if len(args) < 2:
            self._post_system("Usage: /workflow <name> <prompt>")
            return

        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        name = args[0]
        if name not in cli_app.workflows:
            available = ", ".join(sorted(cli_app.workflows)) or "(none)"
            self._post_system(f"Unknown workflow '{name}'. Available: {available}")
            return

        prompt = " ".join(args[1:])
        workflow = cli_app.workflows[name]

        def _do():
            return cli_app._workflow_runner.run(workflow, prompt)

        self._run_in_worker(_do, label=f"workflow:{name}")

    def _cmd_verify(self, args: list[str]) -> None:
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        verify_config = cli_app.config.get_workflows_config().get("verify", {})

        def _do():
            from .agents.builtins import cmd_verify
            return cmd_verify(cli_app, verify_config, print_fn=None)

        self._run_in_worker(_do, label="verify")

    def _cmd_review(self, args: list[str]) -> None:
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        base_ref = args[0] if args else ""

        def _do():
            from .agents.builtins import cmd_review
            return cmd_review(cli_app, base_ref=base_ref, print_fn=None)

        self._run_in_worker(_do, label="review")

    def _cmd_checkpoint(self, args: list[str]) -> None:
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        label = " ".join(args) if args else "checkpoint"

        def _do():
            from .agents.builtins import cmd_checkpoint
            return cmd_checkpoint(cli_app, label=label, print_fn=None)

        self._run_in_worker(_do, label="checkpoint")

    def _cmd_shannon(self, args: list[str]) -> None:
        if not args:
            self._post_system(
                "Usage: /shannon <url> [repo] | /shannon logs [id] "
                "| /shannon workspaces | /shannon stop"
            )
            return

        shannon = self._get_shannon()
        if shannon is None:
            self._post_system(
                "Shannon integration not available. "
                "Install it with: pip install cascade-cli  (it registers via entry points)"
            )
            return
        subcmd = args[0].lower()

        if subcmd == "stop":
            shannon.cmd_stop()
        elif subcmd == "logs":
            wf_id = args[1] if len(args) > 1 else ""
            shannon.cmd_logs(wf_id)
        elif subcmd == "workspaces":
            shannon.cmd_workspaces()
        elif subcmd.startswith("http://") or subcmd.startswith("https://"):
            repo = args[1] if len(args) > 1 else ""
            shannon.cmd_start(subcmd, repo)
        else:
            self._post_system(f"Unknown shannon subcommand: {subcmd}")

    def _cmd_init(self, args: list[str]) -> None:
        from pathlib import Path
        from .agents.templates import detect_project_type
        from .agents.init import run_init

        project_dir = Path(".").resolve()

        # Check if .cascade/ already fully exists
        cascade_dir = project_dir / ".cascade"
        if cascade_dir.is_dir() and (cascade_dir / "agents.yaml").is_file():
            self._post_system(
                ".cascade/ already exists with agents.yaml. "
                "Use /config reload to pick up changes."
            )
            return

        project_type = args[0] if args else detect_project_type(project_dir)

        def _do():
            return run_init(project_dir, project_type)

        self._run_in_worker(_do, label="init")

    def _cmd_upload(self, args: list[str]) -> None:
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        ctx = cli_app.context_builder

        # /upload stop
        if args and args[0].lower() == "stop":
            if self._upload_server and self._upload_server.running:
                self._upload_server.stop()
                self._post_system("Upload server stopped.")
            else:
                self._post_system("Upload server is not running.")
            return

        # /upload status
        if args and args[0].lower() == "status":
            running = self._upload_server and self._upload_server.running
            status = "running" if running else "stopped"
            self._post_system(
                f"Upload server: {status}\n"
                f"Sources: {ctx.source_count}\n"
                f"Tokens: ~{ctx.token_estimate}"
            )
            return

        # /upload [--host H] [--port P]
        if self._upload_server and self._upload_server.running:
            url = f"http://{self._upload_server.host}:{self._upload_server.port}"
            self._post_system(f"Upload server already running at {url}")
            return

        try:
            from .web.server import FileUploaderServer
        except ImportError:
            self._post_system(
                "Web dependencies not installed. Run: pip install cascade-cli[web]"
            )
            return

        host = "0.0.0.0"
        port = 9222
        for i, p in enumerate(args):
            if p == "--host" and i + 1 < len(args):
                host = args[i + 1]
            elif p == "--port" and i + 1 < len(args):
                try:
                    port = int(args[i + 1])
                except ValueError:
                    pass

        self._upload_server = FileUploaderServer(ctx, host=host, port=port)
        url = self._upload_server.start()
        self._post_system(f"Upload server started at {url}")

    def _cmd_context(self, args: list[str]) -> None:
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        ctx = cli_app.context_builder

        # /context clear
        if args and args[0].lower() == "clear":
            ctx.clear()
            self._post_system("Context cleared.")
            return

        # /context -- show sources
        sources = ctx.list_sources()
        if not sources:
            self._post_system("No uploaded context sources.")
            return

        lines = [f"Context sources ({ctx.source_count}, ~{ctx.token_estimate} tokens):"]
        for s in sources:
            lines.append(f"  [{s['type']}] {s['label']} ({s['size']} chars)")
        self._post_system("\n".join(lines))

    def _cmd_login(self, args: list[str]) -> None:
        """Sync provider credentials from installed CLI tools."""
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        from .auth import detect_claude, detect_codex, detect_gemini

        detectors = {
            "gemini": (
                detect_gemini,
                "Run `gemini`, then use `/auth` and choose Login with Google.",
            ),
            "claude": (detect_claude, "claude login"),
            "openai": (detect_codex, "codex login"),
        }

        if not args:
            lines = ["Auth status:"]
            for provider, (detect_fn, _hint) in detectors.items():
                cred = detect_fn()
                if cred:
                    label = cred.source
                    if cred.email:
                        label += f" ({cred.email})"
                    if cred.plan:
                        label += f" [{cred.plan}]"
                    lines.append(f"  {provider}: detected via {label}")
                else:
                    lines.append(f"  {provider}: not detected")
            lines.append("")
            lines.append("Use /login <provider> to sync detected CLI credentials.")
            self._post_system("\n".join(lines))
            return

        provider = args[0].lower()
        if provider not in detectors:
            self._post_system("Usage: /login <gemini|claude|openai>")
            return

        detect_fn, login_hint = detectors[provider]
        cred = detect_fn()
        if cred is None:
            if provider == "gemini":
                oauth_path = Path.home() / ".gemini" / "oauth_creds.json"
                if oauth_path.is_file():
                    try:
                        data = json.loads(oauth_path.read_text(encoding="utf-8"))
                        expiry_ms = data.get("expiry_date")
                        if isinstance(expiry_ms, (int, float)):
                            expiry_dt = datetime.datetime.fromtimestamp(expiry_ms / 1000.0)
                            self._post_system(
                                "No valid gemini CLI credentials found. "
                                f"Detected expired token (expired {expiry_dt}). "
                                f"{login_hint} Then retry /login gemini."
                            )
                            return
                    except Exception:
                        pass
            self._post_system(
                f"No {provider} CLI credentials found. "
                f"{login_hint} Then retry /login {provider}."
            )
            return

        cli_app.config.apply_credential(provider, cred.token, overwrite=True)
        cli_app.config.save()

        # Reinitialize providers and prompt pipeline so changes apply immediately.
        cli_app._init_providers()
        cli_app.prompt_pipeline = cli_app._build_prompt_pipeline()

        try:
            from .widgets.header import ProviderGhostTable
            table = self.app.screen.query_one(ProviderGhostTable)
            table._providers = cli_app.providers
            table.refresh()
        except Exception:
            pass

        def _do() -> str:
            try:
                prov = cli_app.get_provider(provider)
            except Exception as e:
                return f"{provider} credential synced, but provider is unavailable: {e}"
            ok = prov.ping()
            if ok:
                return f"{provider} credential synced and verified."
            return (
                f"{provider} credential synced, but ping failed. "
                f"Re-run `{login_hint}` and try again."
            )

        self._run_in_worker(_do, label=f"login:{provider}")

    def _cmd_config(self, args: list[str]) -> None:
        if not args or args[0].lower() != "reload":
            self._post_system("Usage: /config reload")
            return

        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self.app.notify("No app available")
            return

        # Re-read config from disk
        old_providers = set(cli_app.providers.keys())
        cli_app.config = type(cli_app.config)()
        if hasattr(cli_app, "memory_config") and hasattr(cli_app.config, "get_memory_config"):
            cli_app.memory_config = cli_app.config.get_memory_config()

        # Re-detect credentials and re-init providers.
        # _apply_detected_credentials mutates config and returns None.
        from .auth import detect_all
        cli_app.credentials = detect_all()
        cli_app._apply_detected_credentials()
        cli_app._init_providers()

        # Rebuild prompt pipeline
        cli_app.prompt_pipeline = cli_app._build_prompt_pipeline()

        new_providers = set(cli_app.providers.keys())
        added = new_providers - old_providers
        removed = old_providers - new_providers

        # Update provider token counters in state
        for name in new_providers:
            if name not in self.app.state.provider_tokens:
                self.app.state.provider_tokens[name] = 0

        # Refresh ghost table
        try:
            from .widgets.header import ProviderGhostTable
            table = self.app.screen.query_one(ProviderGhostTable)
            table._providers = cli_app.providers
            table.refresh()
        except Exception:
            pass

        parts = ["Config reloaded."]
        if added:
            parts.append(f"Added: {', '.join(sorted(added))}")
        if removed:
            parts.append(f"Removed: {', '.join(sorted(removed))}")
        parts.append(f"Active providers: {', '.join(sorted(new_providers))}")
        self._post_system("\n".join(parts))

    # ------------------------------------------------------------------
    # Clear / Compact
    # ------------------------------------------------------------------

    def _cmd_clear(self, args: list[str]) -> None:
        """Clear the chat display (state/history preserved)."""
        try:
            from .widgets.message import ChatHistory
            chat = self.app.screen.query_one(ChatHistory)
            chat.remove_children()
        except Exception:
            pass
        self.app.notify("Chat cleared (history preserved)")

    def _cmd_compact(self, args: list[str]) -> None:
        """Force a cross-model memory compaction now."""
        screen = self.app.screen
        if hasattr(screen, "_trigger_summary_compaction"):
            screen._trigger_summary_compaction(reason="manual", force=True)
            self.app.notify("Memory compaction started")
        else:
            self.app.notify("Compaction not available on this screen")

    # ------------------------------------------------------------------
    # History / Resume / Export
    # ------------------------------------------------------------------

    def _cmd_history(self, args: list[str]) -> None:
        """List recent chat sessions."""
        limit = 10
        if args:
            try:
                limit = int(args[0])
            except ValueError:
                pass

        sessions = self.app.db.list_sessions(limit=limit)
        if not sessions:
            self._post_system("No sessions found.")
            return

        lines = ["Recent sessions:"]
        for s in sessions:
            title = s.get("title", "(untitled)")[:40] or "(untitled)"
            created = s.get("created_at", "")[:16]
            sid = s["id"]
            lines.append(f"  {sid}  {created}  {title}")
        lines.append("")
        lines.append("Use /resume <id> to continue a session.")
        self._post_system("\n".join(lines))

    def _cmd_resume(self, args: list[str]) -> None:
        """Resume a previous session by loading its messages."""
        if not args:
            self._post_system("Usage: /resume <session_id>")
            return

        session_id = args[0]
        session = self.app.db.get_session(session_id)
        if session is None:
            self._post_system(f"Session '{session_id}' not found.")
            return

        messages = self.app.db.get_session_messages(session_id)
        if not messages:
            self._post_system(f"Session '{session_id}' has no messages.")
            return

        session_provider = str(session.get("provider", "assistant") or "assistant")

        # Adopt this session as the current one and reset in-memory conversation state.
        self.app.adopt_session(session)
        self.app.state.reset_session(session_id=session["id"])
        cli_app = getattr(self.app, "cli_app", None)
        if session_provider in getattr(cli_app, "providers", {}):
            restored_mode = self._mode_for_provider(cli_app, session_provider)
            self._apply_provider_model(self.app, session_provider, restored_mode, fast=False)
            self.app.state.set_provider(session_provider, restored_mode)
            try:
                self.app.screen._active_provider = session_provider
                self.app.screen._mode = restored_mode
            except Exception:
                pass

        # Load messages into state
        for msg in messages:
            role = self._state_role_for_history_message(msg, session_provider)
            token_count = int(msg.get("token_count", 0))
            self.app.state.add_message(
                role, msg["content"], tokens=token_count,
            )
            if msg["role"] != "user" and token_count > 0 and role != "system":
                self.app.state.provider_tokens[role] = (
                    self.app.state.provider_tokens.get(role, 0) + token_count
                )
                self.app.state.total_tokens += token_count

        if len(self.app.state.messages) > 12:
            try:
                from .conversation import compact_messages_with_episodes

                new_episodes, remaining = compact_messages_with_episodes(
                    self.app.state.messages,
                    keep_recent=6,
                )
                compacted_count = max(len(self.app.state.messages) - len(remaining), 0)
                self.app.state.apply_episode_compaction(compacted_count, new_episodes)
            except Exception:
                pass

        # Mount message widgets in chat history
        try:
            from .screens.main import WelcomeHeader
            from .widgets.header import ProviderGhostTable
            from .widgets.input_frame import InputFrame
            from .widgets.message import ChatHistory, MessageWidget
            from .widgets.status_bar import StatusBar

            chat = self.app.screen.query_one(ChatHistory)
            chat.remove_children()
            for msg in messages:
                role = self._state_role_for_history_message(msg, session_provider)
                if role == "you":
                    content = msg["content"]
                    line_count = content.count("\n") + 1
                    display = (
                        f"[pasted content 1 + {line_count - 1} lines]"
                        if line_count >= 2
                        else content
                    )
                    chat.mount(MessageWidget("you", display))
                else:
                    chat.mount(MessageWidget(role, msg["content"]))
            chat.scroll_end(animate=False)

            if hasattr(self.app.screen, "_header_visible"):
                self.app.screen._header_visible = False
            if hasattr(self.app.screen, "_cross_model_summary"):
                self.app.screen._cross_model_summary = ""
            if hasattr(self.app.screen, "_turns_since_summary"):
                self.app.screen._turns_since_summary = 0
            if hasattr(self.app.screen, "_summary_compaction_running"):
                self.app.screen._summary_compaction_running = False
            self.app.screen.query_one(WelcomeHeader).display = False
            self.app.screen.query_one(StatusBar).update_tokens(self.app.state.provider_tokens)
            frame = self.app.screen.query_one(InputFrame)
            frame.active_provider = self.app.state.active_provider
            frame.mode = self.app.state.mode
            frame.token_count = self.app.state.total_tokens
            try:
                self.app.screen.query_one(ProviderGhostTable).set_active(self.app.state.active_provider)
            except Exception:
                pass
        except Exception:
            pass

        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is not None:
            from .hooks import HookEvent, HookContext
            cli_app.hook_runner.emit(
                HookEvent.SESSION_RESUME,
                HookContext(
                    event=HookEvent.SESSION_RESUME.value,
                    provider=self.app.state.active_provider,
                    mode=self.app.state.mode,
                    session_id=session["id"],
                    metadata=(("message_count", str(len(messages))),),
                ),
            )

        title = session.get("title", "(untitled)")
        self._post_system(
            f"Resumed session: {title} ({len(messages)} messages)"
        )

    def _cmd_export(self, args: list[str]) -> None:
        """Export session messages to a Markdown file."""
        if not args:
            session = self.app._db_session
            if session is None:
                ensure_session = getattr(self.app, "ensure_session", None)
                if callable(ensure_session):
                    try:
                        session = ensure_session()
                    except Exception:
                        session = None
            if session is None:
                self._post_system("No active session. Use /export <session_id>.")
                return
            session_id = session["id"]
        else:
            session_id = args[0]

        session = self.app.db.get_session(session_id)
        if session is None:
            self._post_system(f"Session '{session_id}' not found.")
            return

        messages = self.app.db.get_session_messages(session_id)
        if not messages:
            self._post_system(f"No messages found for session {session_id}.")
            return

        title = session.get("title", "untitled")
        provider = session.get("provider", "assistant")
        model = session.get("model", "")
        model_label = model or provider
        out_path = Path.cwd() / f"cascade-session-{session_id}.md"

        lines = [f"# Cascade Session: {title}", f"Model: {model_label}", ""]
        for msg in messages:
            provider_label = self._history_provider_for_message(msg, provider)
            if provider_label == "user":
                role = "USER"
            elif provider_label == provider:
                role = model_label
            else:
                role = provider_label
            ts = msg.get("timestamp", "")[:19]
            lines.append(f"## {role} ({ts})")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        self._post_system(f"Exported {len(messages)} messages to {out_path}")

    def _cmd_mark(self, args: list[str]) -> None:
        label = " ".join(args) if args else datetime.datetime.now().strftime("%I:%M %p")
        command_line = f"/mark {' '.join(args)}".rstrip()
        self._record_command_line(command_line, title=command_line)
        try:
            from rich.text import Text
            from textual.widgets import Static
            from .widgets.message import ChatHistory

            chat = self.app.screen.query_one(ChatHistory)
            sep = Static(
                Text(f"\u2500\u2500\u2500 {label} \u2500\u2500\u2500", style=f"dim {PALETTE.text_dim}"),
                classes="bookmark",
            )
            chat.mount(sep)
            chat.scroll_end(animate=False)
            self._record_history_message("system", f"--- {label} ---")
        except Exception:
            pass

    def _cmd_time(self, args: list[str]) -> None:
        try:
            from .utils.time import formatted_time, get_timezone
            now = formatted_time(tz=get_timezone())
        except Exception:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.app.notify(f"Time: {now}")

    def _cmd_swarm(self, args: list[str]) -> None:
        """Multi-model swarm dispatch: plan, execute, synthesize."""
        if not args:
            self._post_system("Usage: /swarm <task description>")
            return

        objective = " ".join(args)
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self._post_system("Swarm requires CLI app.")
            return

        available = list(cli_app.providers.keys())
        if len(available) < 2:
            self._post_system(
                f"Swarm needs 2+ providers. Have: {available}. "
                f"Use /login to add more."
            )
            return

        self._post_system(
            f"Swarm dispatching: {objective}\n"
            f"Providers: {', '.join(available)}"
        )
        self._record_command_line(f"/swarm {objective}", title=f"[Swarm] {objective}")

        def _on_progress(stage: str, detail: str) -> None:
            self.app.call_from_thread(self._post_system, f"[{stage}] {detail}")

        def _worker() -> None:
            try:
                from .swarm import SwarmOrchestrator
                swarm = SwarmOrchestrator(cli_app)
                result = swarm.execute(objective, on_progress=_on_progress)

                # Format result
                lines = [f"Swarm complete. Providers used: {', '.join(result.providers_used)}"]
                lines.append(f"Total tokens: {result.total_tokens:,}")
                lines.append("")

                for sr in result.subtask_results:
                    status = "OK" if sr.success else f"FAIL: {sr.error}"
                    lines.append(f"  [{sr.task_id}] {sr.provider}: {status}")

                lines.append("")
                lines.append("--- Synthesis ---")
                lines.append(result.synthesis)

                final = "\n".join(lines)
                self.app.call_from_thread(self._post_system, final)
                self.app.call_from_thread(self._record_history_message, "system", final)

                # Record in state
                self.app.call_from_thread(
                    self.app.state.add_message,
                    "system", f"[Swarm] {objective}",
                )
                # Record synthesis as a message from the orchestrator
                if result.synthesis:
                    from .episodes import generate_episode
                    episode = generate_episode(
                        user_content=f"[Swarm] {objective}",
                        assistant_content=result.synthesis,
                        provider="swarm",
                        tokens=result.total_tokens,
                    )
                    self.app.call_from_thread(self.app.state.add_episode, episode)

            except Exception as e:
                self.app.call_from_thread(self._post_system, f"Swarm error: {e}")
                self.app.call_from_thread(self._record_history_message, "system", f"Swarm error: {e}")

        screen = self.app.screen
        screen.run_worker(_worker, thread=True, exclusive=False)

    def _cmd_solve(self, args: list[str]) -> None:
        """Code a task in an isolated worktree, verified by tests (iterate to green)."""
        if not args:
            self._post_system("Usage: /solve <task description>")
            return

        objective = " ".join(args)
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self._post_system("Solve requires CLI app.")
            return

        provider = getattr(getattr(self.app, "state", None), "active_provider", None)

        self._post_system(f"Solving: {objective}")
        self._record_command_line(f"/solve {objective}", title=f"[Solve] {objective}")
        progress = self._mount_progress_indicator(f"solving: {objective[:60]}")

        def _call_ui(fn, *call_args) -> None:
            caller = getattr(self.app, "call_from_thread", None)
            if callable(caller):
                caller(fn, *call_args)
            else:
                fn(*call_args)

        def _on_progress(stage: str, detail: str) -> None:
            label = f"{stage}: {detail}"
            label = label[:100] if len(label) > 100 else label
            if progress is None:
                _call_ui(self._post_system, f"[{stage}] {detail}")
            else:
                _call_ui(self._set_progress_indicator_label, progress, label)

        def _worker() -> None:
            try:
                from .swarm.solve import run_solve

                result = run_solve(
                    cli_app, objective, provider_name=provider, on_progress=_on_progress
                )

                outcome = "PASSED" if result.passed else "FAILED"
                lines = [
                    f"Solve {outcome} after {result.iterations} "
                    f"iteration(s) on {result.provider}"
                ]
                if result.models_used:
                    seq: list[str] = []
                    for m in result.models_used:
                        if not seq or seq[-1] != m:
                            seq.append(m)
                    lines.append("Models: " + " -> ".join(seq))
                if result.error:
                    lines.append(f"Error: {result.error}")
                if result.changed_files:
                    lines.append("Files: " + ", ".join(result.changed_files[:8]))
                if result.diff_stat:
                    lines.append(result.diff_stat)
                if result.worktree_path:
                    lines.append(f"Worktree: {result.worktree_path}")
                if result.diff_excerpt:
                    lines.append("")
                    lines.append("--- Verified diff ---")
                    lines.append(result.diff_excerpt)

                final = "\n".join(lines)
                _call_ui(self._clear_progress_indicator, progress)
                _call_ui(self._post_system, final)
                _call_ui(self._record_history_message, "system", final)
                _call_ui(self.app.state.add_message, "system", f"[Solve] {objective}")
            except Exception as e:
                _call_ui(self._clear_progress_indicator, progress)
                _call_ui(self._post_system, f"Solve error: {e}")
                _call_ui(self._record_history_message, "system", f"Solve error: {e}")

        screen = self.app.screen
        screen.run_worker(_worker, thread=True, exclusive=False)

    def _cmd_pipeline(self, args: list[str]) -> None:
        """Decompose a build into ordered steps, each run as a verified worker."""
        if not args:
            self._post_system("Usage: /pipeline <objective>")
            return

        objective = " ".join(args)
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            self._post_system("Pipeline requires CLI app.")
            return

        provider = getattr(getattr(self.app, "state", None), "active_provider", None)

        self._post_system(f"Pipeline: {objective}")
        self._record_command_line(f"/pipeline {objective}", title=f"[Pipeline] {objective}")
        progress = self._mount_progress_indicator(f"pipeline: {objective[:60]}")

        def _call_ui(fn, *call_args) -> None:
            caller = getattr(self.app, "call_from_thread", None)
            if callable(caller):
                caller(fn, *call_args)
            else:
                fn(*call_args)

        def _on_progress(stage: str, detail: str) -> None:
            label = f"{stage}: {detail}"
            label = label[:100] if len(label) > 100 else label
            if progress is None:
                _call_ui(self._post_system, f"[{stage}] {detail}")
            else:
                _call_ui(self._set_progress_indicator_label, progress, label)

        def _worker() -> None:
            try:
                from .swarm.pipeline import run_pipeline

                result = run_pipeline(
                    cli_app, objective, provider_name=provider, on_progress=_on_progress
                )

                passed_steps = sum(1 for s in result.steps if s.passed)
                outcome = "PASSED" if result.passed else "FAILED"
                lines = [
                    f"Pipeline {outcome}: {passed_steps}/{len(result.steps)} "
                    f"steps verified on {result.provider}"
                ]
                if result.error:
                    lines.append(f"Error: {result.error}")
                for step in result.steps:
                    mark = "OK" if step.passed else "FAIL"
                    lines.append(
                        f"  [{step.id}] {mark} ({step.iterations} iter): {step.description}"
                    )
                if result.changed_files:
                    lines.append("Files: " + ", ".join(result.changed_files[:8]))
                if result.diff_stat:
                    lines.append(result.diff_stat)
                if result.worktree_path:
                    lines.append(f"Worktree: {result.worktree_path}")
                if result.diff_excerpt:
                    lines.append("")
                    lines.append("--- Verified diff ---")
                    lines.append(result.diff_excerpt)

                final = "\n".join(lines)
                _call_ui(self._clear_progress_indicator, progress)
                _call_ui(self._post_system, final)
                _call_ui(self._record_history_message, "system", final)
                _call_ui(self.app.state.add_message, "system", f"[Pipeline] {objective}")
            except Exception as e:
                _call_ui(self._clear_progress_indicator, progress)
                _call_ui(self._post_system, f"Pipeline error: {e}")
                _call_ui(self._record_history_message, "system", f"Pipeline error: {e}")

        screen = self.app.screen
        screen.run_worker(_worker, thread=True, exclusive=False)

    def _cmd_compete(self, args: list[str]) -> None:
        """Run the same task across providers and let a judge pick a winner."""
        request = self._resolve_compete_request(args, "compete", "Competition")
        if request is None:
            return
        cli_app, selected, judge_arg, objective = request
        provider_states = {provider: "queued" for provider in selected}
        judge_status = ""

        self._post_system(
            f"Competition dispatching: {objective}\n"
            f"Providers: {', '.join(selected)}\n"
            f"Judge: {judge_arg or 'auto'}"
        )
        self._record_command_line(
            f"/compete {' '.join(args)}",
            title=f"[Compete] {objective}",
        )
        progress = self._mount_progress_indicator(
            self._format_progress_label(selected, provider_states)
        )

        def _call_ui(fn, *call_args) -> None:
            caller = getattr(self.app, "call_from_thread", None)
            if callable(caller):
                caller(fn, *call_args)
            else:
                fn(*call_args)

        def _on_progress(stage: str, detail: str) -> None:
            nonlocal judge_status
            judge_status = self._update_progress_states(
                stage,
                detail,
                provider_states,
                judge_arg or "auto",
            ) or judge_status
            label = self._format_progress_label(selected, provider_states, judge_status)
            label = label[:100] if len(label) > 100 else label
            if progress is None:
                _call_ui(self._post_system, f"[{stage}] {detail}")
            else:
                _call_ui(self._set_progress_indicator_label, progress, label)

        def _worker() -> None:
            try:
                from .swarm import CompetitionOrchestrator

                compete = CompetitionOrchestrator(cli_app, judge_provider=judge_arg)
                result = compete.execute(objective, providers=selected, on_progress=_on_progress)

                lines = [f"Competition complete. Judge: {result.judge_provider}"]
                winner_label = result.winner_provider or "none"
                lines.append(f"Winner: {winner_label}")
                lines.append(f"Total tokens: {result.total_tokens:,}")
                lines.append("")

                for entry in result.entries:
                    if entry.success:
                        status = "OK"
                    else:
                        status = f"FAIL: {entry.error}"
                    lines.append(
                        f"  [{entry.provider}] {status} "
                        f"({entry.duration_seconds:.2f}s, {entry.tokens:,} tokens)"
                    )

                if result.judgment and result.judgment.rationale:
                    lines.append("")
                    lines.append(f"Judge rationale: {result.judgment.rationale}")
                if result.judgment and result.judgment.summary:
                    lines.append(f"Judge summary: {result.judgment.summary}")
                if result.winner_response:
                    lines.append("")
                    lines.append("--- Winner ---")
                    lines.append(result.winner_response)

                final = "\n".join(lines)
                _call_ui(self._clear_progress_indicator, progress)
                _call_ui(self._post_system, final)
                _call_ui(self._record_history_message, "system", final)

                _call_ui(
                    self.app.state.add_message,
                    "system", f"[Compete] {objective}",
                )
                if result.winner_response:
                    from .episodes import generate_episode

                    episode = generate_episode(
                        user_content=f"[Compete] {objective}",
                        assistant_content=result.winner_response,
                        provider="compete",
                        tokens=result.total_tokens,
                    )
                    _call_ui(self.app.state.add_episode, episode)

            except Exception as e:
                _call_ui(self._clear_progress_indicator, progress)
                _call_ui(self._post_system, f"Competition error: {e}")
                _call_ui(self._record_history_message, "system", f"Competition error: {e}")

        screen = self.app.screen
        screen.run_worker(_worker, thread=True, exclusive=False)

    def _cmd_compete_code(self, args: list[str]) -> None:
        """Run the same coding task across providers in isolated git worktrees."""
        request = self._resolve_compete_request(args, "compete-code", "Code competition")
        if request is None:
            return
        cli_app, selected, judge_arg, objective = request
        provider_states = {provider: "queued" for provider in selected}
        judge_status = ""

        self._post_system(
            f"Code competition dispatching: {objective}\n"
            f"Providers: {', '.join(selected)}\n"
            f"Judge: {judge_arg or 'auto'}"
        )
        self._record_command_line(
            f"/compete-code {' '.join(args)}",
            title=f"[Compete Code] {objective}",
        )
        progress = self._mount_progress_indicator(
            self._format_progress_label(selected, provider_states)
        )

        def _call_ui(fn, *call_args) -> None:
            caller = getattr(self.app, "call_from_thread", None)
            if callable(caller):
                caller(fn, *call_args)
            else:
                fn(*call_args)

        def _on_progress(stage: str, detail: str) -> None:
            nonlocal judge_status
            judge_status = self._update_progress_states(
                stage,
                detail,
                provider_states,
                judge_arg or "auto",
            ) or judge_status
            label = self._format_progress_label(selected, provider_states, judge_status)
            label = label[:100] if len(label) > 100 else label
            if progress is None:
                _call_ui(self._post_system, f"[{stage}] {detail}")
            else:
                _call_ui(self._set_progress_indicator_label, progress, label)

        def _worker() -> None:
            try:
                from .swarm import CompetitionOrchestrator

                compete = CompetitionOrchestrator(cli_app, judge_provider=judge_arg)
                result = compete.execute_code(objective, providers=selected, on_progress=_on_progress)

                lines = [f"Code competition complete. Judge: {result.judge_provider}"]
                winner_label = result.winner_provider or "none"
                lines.append(f"Winner: {winner_label}")
                lines.append(f"Total tokens: {result.total_tokens:,}")

                winner_entry = next(
                    (
                        entry for entry in result.entries
                        if entry.provider == result.winner_provider and entry.retained and entry.worktree_path
                    ),
                    None,
                )
                if winner_entry is not None:
                    lines.append(f"Winner worktree: {winner_entry.worktree_path}")

                lines.append("")

                for entry in result.entries:
                    status = "OK" if entry.success else f"FAIL: {entry.error}"
                    diff_summary = self._diff_summary(entry.diff_stat, entry.changed_files)
                    lines.append(
                        f"  [{entry.provider}] {status} "
                        f"({entry.duration_seconds:.2f}s, {entry.tokens:,} tokens) | {diff_summary}"
                    )
                    changed = ", ".join(entry.changed_files[:6]) if entry.changed_files else "no file changes"
                    lines.append(f"    Files: {changed}")
                    if entry.retained and entry.worktree_path:
                        lines.append(f"    Worktree: {entry.worktree_path}")
                    elif entry.worktree_path:
                        lines.append("    Worktree: removed")

                if result.judgment and result.judgment.rationale:
                    lines.append("")
                    lines.append(f"Judge rationale: {result.judgment.rationale}")
                if result.judgment and result.judgment.summary:
                    lines.append(f"Judge summary: {result.judgment.summary}")
                if result.winner_response:
                    lines.append("")
                    lines.append("--- Winner ---")
                    lines.append(result.winner_response)

                final = "\n".join(lines)
                _call_ui(self._clear_progress_indicator, progress)
                _call_ui(self._post_system, final)
                _call_ui(self._record_history_message, "system", final)

                _call_ui(
                    self.app.state.add_message,
                    "system", f"[Compete Code] {objective}",
                )
                if result.winner_response:
                    from .episodes import generate_episode

                    episode = generate_episode(
                        user_content=f"[Compete Code] {objective}",
                        assistant_content=result.winner_response,
                        provider="compete-code",
                        tokens=result.total_tokens,
                    )
                    _call_ui(self.app.state.add_episode, episode)

            except Exception as e:
                _call_ui(self._clear_progress_indicator, progress)
                _call_ui(self._post_system, f"Code competition error: {e}")
                _call_ui(self._record_history_message, "system", f"Code competition error: {e}")

        screen = self.app.screen
        screen.run_worker(_worker, thread=True, exclusive=False)

    def _cmd_episodes(self, args: list[str]) -> None:
        """Show episode history."""
        self._record_command_line("/episodes", title="/episodes")
        episodes = self.app.state.episodes
        if not episodes:
            self._post_system("No episodes recorded yet.")
            self._record_history_message("system", "No episodes recorded yet.")
            return

        lines = [f"Episode history ({len(episodes)} episodes):"]
        for ep in episodes[-10:]:  # Show last 10
            actions = ", ".join(ep.actions[:3]) if ep.actions else "none"
            artifacts = ", ".join(ep.artifacts[:3]) if ep.artifacts else "none"
            lines.append(
                f"\n  [{ep.id}] {ep.provider}\n"
                f"    Objective: {ep.objective[:80]}\n"
                f"    Actions: {actions}\n"
                f"    Files: {artifacts}\n"
                f"    Tokens: {ep.tokens_consumed:,}"
            )

        output = "\n".join(lines)
        self._post_system(output)
        self._record_history_message("system", output)

    def _get_branching_session(self):
        """Lazy-init branching session for the current session."""
        return self.app.get_branching_session()

    def _cmd_tree(self, args: list[str]) -> None:
        """Show the session branch tree."""
        self._record_command_line("/tree", title="/tree")
        try:
            bs = self._get_branching_session()
            tree_str = bs.format_tree()
            self._post_system(tree_str)
            self._record_history_message("system", tree_str)
        except Exception as e:
            self._post_system(f"Tree error: {e}")
            self._record_history_message("system", f"Tree error: {e}")

    def _cmd_branch(self, args: list[str]) -> None:
        """Create a branch at the current point."""
        label = " ".join(args) if args else f"branch-{len(self.app.state.messages)}"
        self._record_command_line(f"/branch {' '.join(args)}".rstrip(), title=f"/branch {label}")

        try:
            bs = self._get_branching_session()
            screen = self.app.screen
            provider = getattr(screen, "_active_provider", "")
            branch = bs.create_branch(label=label, provider=provider)
            output = (
                f"Branch '{branch.label}' created (id: {branch.branch_id})\n"
                f"Branching from message: {branch.leaf_id[:12] if branch.leaf_id else 'root'}"
            )
            self._post_system(output)
            self._record_history_message("system", output)
        except Exception as e:
            self._post_system(f"Branch error: {e}")
            self._record_history_message("system", f"Branch error: {e}")
