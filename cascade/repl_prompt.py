"""prompt_toolkit REPL with gutter rendering, mode cycling, and floating input."""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from .cli import CascadeCore
from .context.memory import ContextBuilder
from .history import HistoryDB
from .hooks import HookEvent
from .ui.banner import render_banner
from .ui.gutter import render_bookmark, render_user_gutter
from .ui.input_container import build_prompt_prefix, print_input_bottom, print_input_top, print_mode_indicator
from .ui.mode import ModeState
from .ui.odometer import render_exit_summary
from .ui.spinner import GutterSpinner
from .ui.status import render_status_table
from .ui.statusbar import build_status_bar
from .ui.stream import StreamRenderer
from .ui.theme import DEFAULT_THEME, console

_COMMANDS = [
    "/providers",
    "/switch",
    "/history",
    "/resume",
    "/sessions",
    "/context",
    "/project",
    "/add",
    "/add-dir",
    "/clear-context",
    "/upload",
    "/upload stop",
    "/tools",
    "/hooks",
    "/prompt",
    "/shannon",
    "/shannon logs",
    "/shannon workspaces",
    "/shannon stop",
    "/login",
    "/time",
    "/mark",
    "/model",
    "/mode",
    "/help",
    "/exit",
    "/quit",
]

_PROMPT_STYLE = Style.from_dict(
    {
        "bottom-toolbar": "#aaaaaa bg:#1a1a2e",
        "bottom-toolbar.text": "#aaaaaa",
    }
)


@dataclass
class SessionStats:
    """Track per-session usage metrics."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: float = field(default_factory=time.monotonic)
    message_count: int = 0
    response_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    provider_tokens: dict = field(default_factory=dict)

    def record_input(self, text: str, actual_tokens: int = 0) -> int:
        self.message_count += 1
        tokens = actual_tokens if actual_tokens > 0 else max(len(text) // 4, 1)
        self.total_input_tokens += tokens
        return tokens

    def record_output(
        self,
        text: str,
        provider: str = "",
        actual_usage: Optional[tuple[int, int]] = None,
    ) -> int:
        self.response_count += 1
        if actual_usage and (actual_usage[0] or actual_usage[1]):
            out = actual_usage[1]
            inp = actual_usage[0]
            self.total_output_tokens += out
            self.total_input_tokens += inp - max(len(text) // 4, 1)  # correct estimate
        else:
            out = max(len(text) // 4, 1)
            self.total_output_tokens += out

        if provider:
            prev = self.provider_tokens.get(provider, 0)
            self.provider_tokens[provider] = prev + out
        return out

    @property
    def wall_time(self) -> str:
        elapsed = time.monotonic() - self.start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        if minutes > 0:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def format_tokens(self, count: int) -> str:
        if count >= 1000:
            return f"~{count / 1000:.1f}k"
        return f"~{count}"


class CascadePromptREPL:
    """Full-featured REPL with gutter rendering, mode cycling, and floating input."""

    def __init__(self, app: CascadeCore, auth_info: str = ""):
        self.app = app
        self.db = HistoryDB()
        self.session = None
        self.context_builder = ContextBuilder()
        self._upload_server = None
        self._auth_info = auth_info
        self.stats = SessionStats()
        self.mode = ModeState()

        # Shannon integration (lazy via entry points)
        self._shannon = None
        self._shannon_cfg = app.config.get_integrations_config().get("shannon", {})

        # Key bindings
        self._kb = KeyBindings()
        self._setup_keybindings()

        hist_path = Path("~/.config/cascade/prompt_history").expanduser()
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompt_session = PromptSession(
            history=FileHistory(str(hist_path)),
            completer=WordCompleter(_COMMANDS, sentence=True),
            style=_PROMPT_STYLE,
            bottom_toolbar=self._bottom_toolbar,
            key_bindings=self._kb,
        )

    @property
    def current_provider(self) -> str:
        return self.mode.active_provider

    def _current_model(self) -> str:
        prov = self.app.providers.get(self.current_provider)
        if prov:
            return prov.config.model
        return "?"

    def _setup_keybindings(self) -> None:
        @self._kb.add("s-tab")
        def _cycle_mode(event):
            old_mode = self.mode.mode_name
            self.mode = self.mode.cycle()
            render_bookmark(f"{old_mode} -> {self.mode.mode_name}")

    def _bottom_toolbar(self) -> HTML:
        return build_status_bar(provider_tokens=self.stats.provider_tokens)

    def _ensure_session(self) -> None:
        if self.session is None:
            provider = self.current_provider
            model = self._current_model()
            self.session = self.db.create_session(
                provider=provider, model=model, title="",
            )
            self.stats = SessionStats(session_id=self.session["id"])

    def welcome(self) -> None:
        from cascade import __version__
        palette = DEFAULT_THEME.palette
        provider_count = len(self.app.providers)
        summary = f"{provider_count} provider{'s' if provider_count != 1 else ''} active"

        # 1. Banner
        console.print(render_banner())
        console.print(f"  v{__version__} · {summary}", style="dim")

        # 2. Ghost table
        render_status_table(self.app.providers, self.current_provider)

        # 3. Help hint
        console.print("  /model to switch \u00b7 /help for commands", style=f"dim {palette.text_dim}")
        console.print()

    def _print_exit_summary(self) -> None:
        """Render exit summary with odometer animation and run ON_EXIT hooks."""
        s = self.stats
        provider = self.current_provider
        model = self._current_model()

        self.app.hook_runner.run_hooks(HookEvent.ON_EXIT, context={
            "session_id": s.session_id,
            "messages": str(s.message_count),
            "provider": provider,
        })

        render_exit_summary(
            session_id=s.session_id,
            messages=s.message_count,
            responses=s.response_count,
            input_tokens=s.total_input_tokens,
            output_tokens=s.total_output_tokens,
            wall_time=s.wall_time,
            provider=provider,
            model=model,
            provider_tokens=s.provider_tokens,
            console=console,
        )

    def show_help(self) -> None:
        palette = DEFAULT_THEME.palette
        console.print()
        console.print("  Commands:", style=f"bold {palette.inline_code}")
        console.print()
        help_entries = [
            ("/providers", "List available providers and their status"),
            ("/switch <name>", "Switch to a different provider"),
            ("/history", "Show recent conversation sessions"),
            ("/resume [id]", "Resume a previous session (latest if no id)"),
            ("/sessions", "Alias for /history"),
            ("/context", "Show project context and loaded files"),
            ("/project", "Alias for /context"),
            ("/add <file>", "Add a file to conversation context"),
            ("/add-dir <path> [glob]", "Add directory contents to context"),
            ("/clear-context", "Clear all added context"),
            ("/upload [--host H --port P]", "Start the file upload web server"),
            ("/upload stop", "Stop the upload server"),
            ("/tools", "List available tools"),
            ("/hooks", "List configured hooks"),
            ("/prompt", "Show the assembled system prompt"),
            ("/shannon <url> [repo]", "Launch Shannon pentesting on target"),
            ("/shannon logs [id]", "Tail Shannon workflow logs"),
            ("/shannon workspaces", "List Shannon workspaces"),
            ("/shannon stop", "Stop active Shannon run"),
            ("/login [provider]", "Authenticate with a provider (or show status)"),
            ("/time", "Show current time"),
            ("/mark [label]", "Insert a session bookmark"),
            ("/model <name>", "Override provider (/model reset to clear)"),
            ("/mode <name>", "Switch to a specific mode"),
            ("/help", "Show this help"),
            ("/exit, /quit", "Exit CASCADE"),
        ]
        for cmd, desc in help_entries:
            console.print(f"    {cmd:<28} {desc}", style="dim")
        console.print()
        console.print("  Shift+Tab cycles modes: design -> plan -> build -> test", style="dim")
        console.print("  Or just type your question to chat.\n", style="dim")

    def list_providers(self) -> None:
        palette = DEFAULT_THEME.palette
        console.print("\nAvailable providers:", style=f"bold {palette.inline_code}")
        for name in self.app.providers.keys():
            theme = DEFAULT_THEME.get_provider(name)
            status = "[active]" if name == self.current_provider else "       "
            console.print(f"  {status} {name}", style=f"{theme.accent}")
        console.print()

    def switch_provider(self, name: str) -> None:
        if name not in self.app.providers:
            console.print(f"Provider '{name}' not found.", style="dim red")
            self.list_providers()
            return
        self.mode = self.mode.with_override(name)
        console.print(f"Provider override: {name}", style=f"dim {DEFAULT_THEME.palette.inline_code}")

    def show_history(self, limit: int = 10) -> None:
        sessions = self.db.list_sessions(limit=limit)
        if not sessions:
            console.print("No sessions found.", style="dim")
            return
        palette = DEFAULT_THEME.palette
        console.print("\nRecent sessions:", style=f"bold {palette.inline_code}")
        for s in sessions:
            title = s["title"] or "(untitled)"
            console.print(
                f"  {s['id']}  {title}  [{s['provider']}]  {s['created_at'][:16]}",
                style="dim",
            )
        console.print()

    def resume_session(self, session_id: str = "") -> None:
        if not session_id:
            sessions = self.db.list_sessions(limit=1)
            if not sessions:
                console.print("No sessions to resume.", style="dim")
                return
            session_id = sessions[0]["id"]

        session_data = self.db.get_session(session_id)
        if session_data is None:
            console.print(f"Session '{session_id}' not found.", style="dim red")
            return

        self.session = session_data
        if session_data["provider"] and session_data["provider"] in self.app.providers:
            self.mode = self.mode.with_override(session_data["provider"])

        messages = self.db.get_session_messages(session_id)
        palette = DEFAULT_THEME.palette
        console.print(
            f"\nResumed session {session_id} ({len(messages)} messages)",
            style=f"dim {palette.inline_code}",
        )
        for msg in messages[-4:]:
            role_label = "You" if msg["role"] == "user" else "AI"
            text = msg["content"][:120]
            if len(msg["content"]) > 120:
                text += "..."
            console.print(f"  {role_label}: {text}", style="dim")
        console.print()

    def handle_command(self, line: str) -> bool:
        """Handle slash commands. Returns True to continue, False to exit."""
        if not line.startswith("/"):
            return True

        parts = line.strip().split(None, 1)
        cmd = parts[0].lstrip("/").lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("exit", "quit"):
            self._print_exit_summary()
            return False
        elif cmd == "providers":
            self.list_providers()
        elif cmd == "switch":
            if arg:
                self.switch_provider(arg)
            else:
                console.print("Usage: /switch <provider>", style="dim")
        elif cmd == "time":
            from .utils.time import formatted_time, get_timezone
            console.print(f"Time: {formatted_time(tz=get_timezone())}", style="dim")
        elif cmd in ("history", "sessions"):
            self.show_history()
        elif cmd == "resume":
            self.resume_session(arg)
        elif cmd in ("context", "project"):
            self._show_context()
        elif cmd == "add":
            if arg:
                self.context_builder.add_file(arg)
                console.print(f"Added: {arg}", style=f"dim {DEFAULT_THEME.palette.inline_code}")
            else:
                console.print("Usage: /add <file>", style="dim")
        elif cmd == "add-dir":
            parts_inner = arg.split(None, 1) if arg else []
            if parts_inner:
                dir_path = parts_inner[0]
                glob_pattern = parts_inner[1] if len(parts_inner) > 1 else "*"
                self.context_builder.add_directory(dir_path, glob_pattern)
                console.print(f"Added directory: {dir_path} ({glob_pattern})", style=f"dim {DEFAULT_THEME.palette.inline_code}")
            else:
                console.print("Usage: /add-dir <path> [glob]", style="dim")
        elif cmd == "clear-context":
            self.context_builder.clear()
            console.print("Context cleared.", style=f"dim {DEFAULT_THEME.palette.inline_code}")
        elif cmd == "upload":
            self._handle_upload(arg)
        elif cmd == "tools":
            self._show_tools()
        elif cmd == "hooks":
            self._show_hooks()
        elif cmd == "prompt":
            self._show_prompt()
        elif cmd == "shannon":
            self._handle_shannon(arg)
        elif cmd == "login":
            self._handle_login(arg)
        elif cmd == "mark":
            render_bookmark(arg if arg else None)
        elif cmd == "model":
            self._handle_model(arg)
        elif cmd == "mode":
            self._handle_mode(arg)
        elif cmd == "help":
            self.show_help()
        else:
            console.print(f"Unknown command: /{cmd}", style="dim red")
        return True

    def _handle_model(self, arg: str) -> None:
        """Handle /model <name> or /model reset."""
        if not arg:
            console.print(f"Current: {self.current_provider} ({self._current_model()})", style="dim")
            console.print("Usage: /model <provider> or /model reset", style="dim")
            return
        if arg.lower() == "reset":
            self.mode = self.mode.with_override(None)
            console.print(f"Provider override cleared. Using {self.current_provider}", style="dim")
        elif arg in self.app.providers:
            self.mode = self.mode.with_override(arg)
            console.print(f"Provider override: {arg}", style=f"dim {DEFAULT_THEME.palette.inline_code}")
        else:
            console.print(f"Provider '{arg}' not found.", style="dim red")

    def _handle_mode(self, arg: str) -> None:
        """Handle /mode <name> to switch directly to a mode."""
        from .ui.mode import MODE_ORDER
        if not arg:
            console.print(f"Current mode: {self.mode.mode_name}", style="dim")
            console.print(f"Available: {', '.join(MODE_ORDER)}", style="dim")
            return
        target = arg.lower()
        if target not in MODE_ORDER:
            console.print(f"Unknown mode: {target}. Available: {', '.join(MODE_ORDER)}", style="dim red")
            return
        old_mode = self.mode.mode_name
        idx = MODE_ORDER.index(target)
        self.mode = ModeState(index=idx, override_provider=self.mode.override_provider)
        render_bookmark(f"{old_mode} -> {target}")

    def _show_context(self) -> None:
        palette = DEFAULT_THEME.palette
        ctx = self.app.project
        if ctx.found:
            console.print(f"\n{ctx.summary()}", style=f"bold {palette.inline_code}")
            if ctx.system_prompt:
                preview = ctx.system_prompt[:200]
                if len(ctx.system_prompt) > 200:
                    preview += "..."
                console.print(f"  System prompt: {preview}", style="dim")
            if ctx.context_files:
                for name in ctx.context_files:
                    console.print(f"  Context file: {name}", style="dim")
        else:
            console.print("\nNo .cascade/ directory found.", style="dim")

        sources = self.context_builder.list_sources()
        if sources:
            console.print(
                f"\nAdded context ({len(sources)} sources, ~{self.context_builder.token_estimate} tokens):",
                style=f"bold {palette.inline_code}",
            )
            for s in sources:
                console.print(f"  [{s['type']}] {s['label']} ({s['size']} chars)", style="dim")
        elif not ctx.found:
            console.print("No context added. Use /add <file> to add context.", style="dim")
        console.print()

    def _show_tools(self) -> None:
        tools = self.app.tool_registry
        if not tools:
            console.print("\nNo tools available.", style="dim")
            return
        palette = DEFAULT_THEME.palette
        console.print(f"\nAvailable tools ({len(tools)}):", style=f"bold {palette.inline_code}")
        for name, tool_def in tools.items():
            desc = tool_def.description.split("\n")[0][:80]
            console.print(f"  {name:<20} {desc}", style="dim")
        console.print()

    def _show_hooks(self) -> None:
        hooks = self.app.hook_runner.describe()
        palette = DEFAULT_THEME.palette
        if not hooks:
            console.print("\nNo hooks configured.", style="dim")
            console.print("  Add hooks in ~/.config/cascade/config.yaml", style="dim")
            return
        console.print(f"\nConfigured hooks ({len(hooks)}):", style=f"bold {palette.inline_code}")
        for h in hooks:
            status = "on" if h["enabled"] else "off"
            console.print(
                f"  [{status}] {h['name']:<16} {h['event']:<16} {h['command']}",
                style="dim",
            )
        console.print()

    def _show_prompt(self) -> None:
        ctx_text = self.context_builder.build()
        pipeline = self.app.prompt_pipeline
        if ctx_text:
            from .prompts.layers import PRIORITY_REPL_CONTEXT
            pipeline = pipeline.add_layer("repl_context", ctx_text, PRIORITY_REPL_CONTEXT)

        palette = DEFAULT_THEME.palette
        layers = pipeline.describe()
        console.print(f"\nSystem prompt layers ({len(layers)}):", style=f"bold {palette.inline_code}")
        for layer in layers:
            console.print(
                f"  [{layer['priority']:>2}] {layer['name']:<24} ({layer['length']} chars)",
                style="dim",
            )

        full_prompt = pipeline.build()
        console.print(f"\nTotal length: {len(full_prompt)} chars", style="dim")
        if full_prompt:
            preview = full_prompt[:500]
            if len(full_prompt) > 500:
                preview += "\n..."
            console.print(f"\n{preview}", style="dim")
        console.print()

    def _handle_upload(self, arg: str) -> None:
        if arg.strip().lower() == "stop":
            if self._upload_server and self._upload_server.running:
                self._upload_server.stop()
                console.print("Upload server stopped.", style=f"dim {DEFAULT_THEME.palette.inline_code}")
            else:
                console.print("No upload server running.", style="dim")
            return

        if self._upload_server and self._upload_server.running:
            console.print(
                f"Upload server already running at http://{self._upload_server.host}:{self._upload_server.port}",
                style=f"dim {DEFAULT_THEME.palette.inline_code}",
            )
            return

        try:
            from .web.server import FileUploaderServer
        except ImportError:
            console.print(
                "Web dependencies not installed. Run: pip install cascade-cli[web]",
                style="dim red",
            )
            return

        host = "0.0.0.0"
        port = 9222
        parts = arg.split()
        for i, p in enumerate(parts):
            if p == "--host" and i + 1 < len(parts):
                host = parts[i + 1]
            elif p == "--port" and i + 1 < len(parts):
                try:
                    port = int(parts[i + 1])
                except ValueError:
                    pass

        self._upload_server = FileUploaderServer(
            self.context_builder, host=host, port=port,
        )
        url = self._upload_server.start()
        console.print(f"Upload server started at {url}", style=f"bold {DEFAULT_THEME.palette.inline_code}")

    def _get_shannon(self):
        """Lazy-init Shannon integration via entry point discovery."""
        if self._shannon is None:
            from .integrations import get_integration
            cls = get_integration("shannon")
            if cls is None:
                return None
            self._shannon = cls(config_path=self._shannon_cfg.get("path", ""))
        return self._shannon

    def _handle_shannon(self, arg: str) -> None:
        parts = arg.strip().split(None, 1)
        if not parts:
            console.print(
                "Usage: /shannon <url> [repo] | /shannon logs [id] "
                "| /shannon workspaces | /shannon stop",
                style="dim",
            )
            return

        shannon = self._get_shannon()
        if shannon is None:
            console.print(
                "Shannon integration not available. "
                "Reinstall cascade-cli or check entry points.",
                style="dim red",
            )
            return

        subcmd = parts[0].lower()
        if subcmd == "stop":
            shannon.cmd_stop()
        elif subcmd == "logs":
            workflow_id = parts[1].strip() if len(parts) > 1 else ""
            shannon.cmd_logs(workflow_id)
        elif subcmd == "workspaces":
            shannon.cmd_workspaces()
        elif subcmd.startswith("http://") or subcmd.startswith("https://"):
            url = subcmd
            repo = parts[1].strip() if len(parts) > 1 else ""
            shannon.cmd_start(url, repo)
        else:
            console.print(f"Unknown shannon subcommand: {subcmd}", style="dim red")

    def _handle_login(self, arg: str) -> None:
        try:
            from .auth_flow import login, show_auth_status
        except ImportError:
            console.print("Auth flow module not available.", style="dim red")
            return

        provider = arg.strip().lower() if arg.strip() else ""
        if not provider:
            show_auth_status()
        else:
            result = login(provider)
            if result:
                console.print(
                    f"Authenticated with {result.provider} via {result.method}.",
                    style=f"dim {DEFAULT_THEME.palette.inline_code}",
                )
                self.app.config.apply_credential(result.provider, result.token)
                self.app.config.save()
            else:
                console.print(f"Login cancelled or failed for {provider}.", style="dim")

    def _ask_provider(self, prompt: str) -> tuple[str, Optional[tuple[int, int]]]:
        """Send a prompt to the current provider with streaming, return (text, usage)."""
        provider_name = self.current_provider
        prov = self.app.get_provider(provider_name)

        # Build system prompt
        ctx_text = self.context_builder.build() or None
        pipeline = self.app.prompt_pipeline
        if ctx_text:
            from .prompts.layers import PRIORITY_REPL_CONTEXT
            pipeline = pipeline.add_layer("repl_context", ctx_text, PRIORITY_REPL_CONTEXT)
        final_system = pipeline.build() or None

        # Run BEFORE_ASK hooks
        self.app.hook_runner.run_hooks(HookEvent.BEFORE_ASK, context={
            "prompt": prompt,
            "provider": prov.name,
        })

        # Get theme for rendering
        theme = DEFAULT_THEME.get_provider(provider_name)

        # Spinner while waiting for first token
        spinner = GutterSpinner(theme.abbreviation, theme.accent)
        spinner.start()

        # Stream the response
        renderer = StreamRenderer(theme, console)
        full_text = ""
        try:
            for chunk in prov.stream_single(prompt, final_system):
                if spinner._running:
                    spinner.stop()
                full_text += chunk
                renderer.feed(chunk)
        except Exception:
            spinner.stop()
            raise
        finally:
            if spinner._running:
                spinner.stop()

        renderer.finish()

        # Run AFTER_RESPONSE hooks
        self.app.hook_runner.run_hooks(HookEvent.AFTER_RESPONSE, context={
            "response_length": str(len(full_text)),
            "provider": prov.name,
            "tool_calls": "0",
        })

        # Capture metadata
        usage = prov.last_usage
        from .providers.response import ProviderResponse
        self.app.last_response_meta = ProviderResponse(
            text=full_text,
            input_tokens=usage[0] if usage else 0,
            output_tokens=usage[1] if usage else 0,
            model=prov.config.model,
            provider=prov.name,
        )

        return full_text, usage

    def run(self) -> None:
        """Start the REPL loop."""
        self.welcome()

        try:
            while True:
                try:
                    # Floating input container
                    theme = self.mode.theme
                    print_input_top(theme, self.stats.total_tokens)

                    prefix = build_prompt_prefix(theme)
                    line = self.prompt_session.prompt(prefix)

                    print_input_bottom(theme)
                    print_mode_indicator(theme, self.mode.mode_name)

                    if not line.strip():
                        continue

                    if not self.handle_command(line):
                        break

                    if not line.startswith("/"):
                        self._ensure_session()

                        # Render user message through gutter
                        render_user_gutter(line, console)

                        input_tokens = self.stats.record_input(line)
                        self.db.add_message(
                            self.session["id"],
                            role="user",
                            content=line,
                            token_count=input_tokens,
                        )
                        try:
                            response, usage = self._ask_provider(line)

                            actual = usage if usage else None
                            output_tokens = self.stats.record_output(
                                response,
                                provider=self.current_provider,
                                actual_usage=actual,
                            )
                            self.db.add_message(
                                self.session["id"],
                                role="assistant",
                                content=response,
                                token_count=output_tokens,
                            )
                            if not self.session.get("title"):
                                title = line[:60]
                                self.db.update_session_title(
                                    self.session["id"], title,
                                )
                                self.session["title"] = title
                        except Exception as e:
                            from .ui.output import render_error
                            render_error(str(e))

                except KeyboardInterrupt:
                    console.print()
                    continue

        except EOFError:
            self._print_exit_summary()
