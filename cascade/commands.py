"""Slash command parsing and dispatch for the Cascade TUI.

Commands post state messages rather than manipulating widgets directly.
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path

from .theme import MODES, PALETTE, get_provider_theme


@dataclass(frozen=True)
class CommandDef:
    """Definition of a slash command for autocomplete."""
    name: str
    usage: str
    description: str


# Canonical list of available commands (used by autocomplete and /help)
COMMANDS: tuple[CommandDef, ...] = (
    CommandDef("exit", "/exit", "Exit cascade"),
    CommandDef("quit", "/quit", "Exit cascade"),
    CommandDef("fast", "/fast", "Toggle fast model for current provider"),
    CommandDef("model", "/model <provider>", "Switch active provider"),
    CommandDef("mode", "/mode <name>", "Switch mode (design, plan, build, test)"),
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
    CommandDef("mark", "/mark [label]", "Insert a bookmark separator"),
    CommandDef("time", "/time", "Show current time"),
    CommandDef("help", "/help", "Show available commands"),
)


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

    def _post_system(self, text: str) -> None:
        """Mount a system message in the chat."""
        try:
            from .widgets.message import ChatHistory, MessageWidget
            chat = self.app.screen.query_one(ChatHistory)
            chat.mount(MessageWidget("system", text))
            chat.scroll_end(animate=False)
        except Exception:
            self.app.notify(text)

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

        # Read fast_model from raw config data
        provider_data = cli_app.config.data.get("providers", {}).get(provider_name, {})
        fast_model = provider_data.get("fast_model", "")
        primary_model = provider_data.get("model", "")

        if not fast_model:
            self.app.notify(f"No fast_model configured for {provider_name}")
            return

        # Toggle
        state = self.app.state
        if state.fast_mode:
            prov.config.model = primary_model
            state.fast_mode = False
            label = primary_model
        else:
            prov.config.model = fast_model
            state.fast_mode = True
            label = fast_model

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
            self.app.notify("Usage: /model <provider>")
            return
        name = args[0].lower()
        from .theme import PROVIDERS
        if name not in PROVIDERS:
            self.app.notify(f"Provider '{name}' not found. Available: {', '.join(PROVIDERS)}")
            return
        pt = get_provider_theme(name)
        self.app.state.set_provider(name, pt.default_mode)
        # Reset fast mode on provider switch
        self.app.state.fast_mode = False
        try:
            screen = self.app.screen
            screen._active_provider = name
            screen._mode = pt.default_mode
            inp = screen.query_one("InputFrame")
            inp.active_provider = name
            inp.mode = pt.default_mode
        except Exception:
            pass
        try:
            from .widgets.header import ProviderGhostTable
            self.app.screen.query_one(ProviderGhostTable).set_active(name)
        except Exception:
            pass
        self.app.notify(f"Switched to {name}")

    def _cmd_mode(self, args: list[str]) -> None:
        if not args:
            self.app.notify("Usage: /mode <name>")
            return
        mode_name = args[0].lower()
        if mode_name not in MODES:
            self.app.notify(f"Mode '{mode_name}' not found. Available: {', '.join(MODES)}")
            return
        provider = MODES[mode_name]["provider"]
        self.app.state.set_provider(provider, mode_name)
        self.app.state.fast_mode = False
        try:
            screen = self.app.screen
            screen._active_provider = provider
            screen._mode = mode_name
            inp = screen.query_one("InputFrame")
            inp.active_provider = provider
            inp.mode = mode_name
        except Exception:
            pass
        try:
            from .widgets.header import ProviderGhostTable
            self.app.screen.query_one(ProviderGhostTable).set_active(provider)
        except Exception:
            pass
        self.app.notify(f"Switched to {mode_name} mode")

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

            thinking = ThinkingIndicator(label)
            chat.mount(thinking)
            chat.scroll_end(animate=False)

            def _worker():
                try:
                    result = fn()
                    self.app.call_from_thread(self._finish_worker, thinking, result)
                except Exception as e:
                    self.app.call_from_thread(self._finish_worker, thinking, f"Error: {e}")

            self.app.screen.run_worker(_worker, thread=True, exclusive=False)
        except Exception:
            # Fallback: run synchronously
            try:
                result = fn()
                self._post_system(result)
            except Exception as e:
                self._post_system(f"Error: {e}")

    def _finish_worker(self, thinking, result: str) -> None:
        """Remove thinking indicator and post the result."""
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

        # Adopt this session as the current one
        self.app._db_session = session

        # Load messages into state
        provider = session.get("provider", "assistant")
        for msg in messages:
            role = "you" if msg["role"] == "user" else provider
            self.app.state.add_message(
                role, msg["content"], tokens=msg.get("token_count", 0),
            )

        # Mount message widgets in chat history
        try:
            from .widgets.message import ChatHistory, MessageWidget

            chat = self.app.screen.query_one(ChatHistory)
            for msg in messages:
                if msg["role"] == "user":
                    content = msg["content"]
                    line_count = content.count("\n") + 1
                    display = (
                        f"[pasted content 1 + {line_count - 1} lines]"
                        if line_count >= 2
                        else content
                    )
                    chat.mount(MessageWidget("you", display))
                else:
                    chat.mount(MessageWidget(provider, msg["content"]))
            chat.scroll_end(animate=False)
        except Exception:
            pass

        title = session.get("title", "(untitled)")
        self._post_system(
            f"Resumed session: {title} ({len(messages)} messages)"
        )

    def _cmd_export(self, args: list[str]) -> None:
        """Export session messages to a Markdown file."""
        if not args:
            session = self.app._db_session
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
            if msg["role"] == "user":
                role = "USER"
            else:
                role = model_label
            ts = msg.get("timestamp", "")[:19]
            lines.append(f"## {role} ({ts})")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        self._post_system(f"Exported {len(messages)} messages to {out_path}")

    def _cmd_mark(self, args: list[str]) -> None:
        label = " ".join(args) if args else datetime.datetime.now().strftime("%I:%M %p")
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
        except Exception:
            pass

    def _cmd_time(self, args: list[str]) -> None:
        try:
            from .utils.time import formatted_time, get_timezone
            now = formatted_time(tz=get_timezone())
        except Exception:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.app.notify(f"Time: {now}")
