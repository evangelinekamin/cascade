"""Interactive REPL mode for Cascade - the main entry point."""

import time
import uuid

from .cli import CascadeCore
from .history import HistoryDB
from .ui.theme import console, CYAN, VIOLET
from .utils.time import formatted_time, get_timezone


class CascadeREPL:
    """Interactive REPL interface for Cascade."""

    def __init__(self, app: CascadeCore):
        self.app = app
        self.current_provider = app.config.get_default_provider()
        self.db = HistoryDB()
        self.session = None
        self._session_id = uuid.uuid4().hex[:12]
        self._start_time = time.monotonic()
        self._msg_count = 0
        self._resp_count = 0
        self._input_tokens = 0
        self._output_tokens = 0

        self._shannon = None
        self._shannon_cfg = app.config.get_integrations_config().get("shannon", {})

    def _ensure_session(self) -> None:
        """Create a session if one isn't active."""
        if self.session is None:
            provider = self.current_provider
            model = ""
            prov = self.app.providers.get(provider)
            if prov:
                model = prov.config.model
            self.session = self.db.create_session(
                provider=provider, model=model, title="",
            )

    def welcome(self):
        """Show welcome message."""
        console.print(
            f"\n[{CYAN}]CASCADE[/] Multi-model AI Assistant\n",
            style=f"bold {CYAN}"
        )
        console.print(f"Time: {formatted_time(tz=get_timezone())}", style="dim")
        console.print(f"Default provider: {self.current_provider}", style="dim")
        console.print("Type /help for commands.\n", style="dim")

    def show_help(self):
        """Show a dedicated help screen listing all commands."""
        console.print("\nCommands:", style=f"bold {CYAN}")
        console.print("  /providers       - List available providers")
        console.print("  /switch <name>   - Switch default provider")
        console.print("  /history         - Show recent sessions")
        console.print("  /resume [id]     - Resume a previous session")
        console.print("  /sessions        - List all sessions")
        console.print("  /shannon <url>   - Launch Shannon pentesting")
        console.print("  /shannon stop    - Stop active Shannon run")
        console.print("  /login [provider]- Authenticate with a provider")
        console.print("  /time            - Show current time")
        console.print("  /help            - Show this help")
        console.print("  /exit, /quit     - Exit CASCADE")
        console.print("\nOtherwise, just type your question!\n", style="dim")

    def list_providers(self):
        """Show available providers."""
        console.print("\nAvailable providers:", style=f"bold {CYAN}")
        for name in self.app.providers.keys():
            status = "[active]" if name == self.current_provider else "       "
            console.print(f"  {status} {name}")
        console.print()

    def switch_provider(self, name: str):
        """Switch to a different provider."""
        if name not in self.app.providers:
            console.print(f"Provider '{name}' not found.", style="dim red")
            self.list_providers()
            return
        self.current_provider = name
        console.print(f"Switched to {name}", style=f"dim {CYAN}")

    def show_history(self, limit: int = 10):
        """Show recent sessions."""
        sessions = self.db.list_sessions(limit=limit)
        if not sessions:
            console.print("No sessions found.", style="dim")
            return
        console.print("\nRecent sessions:", style=f"bold {CYAN}")
        for s in sessions:
            title = s["title"] or "(untitled)"
            console.print(
                f"  {s['id']}  {title}  [{s['provider']}]  {s['created_at'][:16]}",
                style="dim",
            )
        console.print()

    def resume_session(self, session_id: str = ""):
        """Resume a previous session."""
        if not session_id:
            sessions = self.db.list_sessions(limit=1)
            if not sessions:
                console.print("No sessions to resume.", style="dim")
                return
            session_id = sessions[0]["id"]

        session = self.db.get_session(session_id)
        if session is None:
            console.print(f"Session '{session_id}' not found.", style="dim red")
            return

        self.session = session
        if session["provider"] and session["provider"] in self.app.providers:
            self.current_provider = session["provider"]

        messages = self.db.get_session_messages(session_id)
        console.print(
            f"\nResumed session {session_id} ({len(messages)} messages)",
            style=f"dim {CYAN}",
        )
        # Replay last few messages for context
        for msg in messages[-4:]:
            role_label = "You" if msg["role"] == "user" else "AI"
            text = msg["content"][:120]
            if len(msg["content"]) > 120:
                text += "..."
            console.print(f"  {role_label}: {text}", style="dim")
        console.print()

    def _print_exit_summary(self) -> None:
        from rich.panel import Panel
        from .hooks import HookEvent

        # Run ON_EXIT hooks
        self.app.hook_runner.run_hooks(HookEvent.ON_EXIT, context={
            "session_id": self._session_id,
            "messages": str(self._msg_count),
            "provider": self.current_provider,
        })

        elapsed = time.monotonic() - self._start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        wall = f"{minutes}m {seconds:02d}s" if minutes > 0 else f"{seconds}s"

        def _fmt(n: int) -> str:
            return f"~{n / 1000:.1f}k" if n >= 1000 else f"~{n}"

        model = ""
        prov = self.app.providers.get(self.current_provider)
        if prov:
            model = prov.config.model

        lines = [
            f"  [bold {CYAN}]Session Summary[/]",
            f"  Session ID:    {self._session_id}",
            f"  Messages:      {self._msg_count} (you) / {self._resp_count} (assistant)",
            f"  Tokens:        {_fmt(self._input_tokens)} input / {_fmt(self._output_tokens)} output",
            f"  Wall Time:     {wall}",
            f"  Provider:      {self.current_provider} ({model})",
        ]
        console.print()
        console.print(Panel("\n".join(lines), border_style=VIOLET, padding=(0, 1), expand=False))
        console.print()

    def handle_command(self, line: str) -> bool:
        """Handle special commands. Returns True to continue, False to exit."""
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
            console.print(f"Time: {formatted_time(tz=get_timezone())}", style="dim")

        elif cmd in ("history", "sessions"):
            self.show_history()

        elif cmd == "resume":
            self.resume_session(arg)

        elif cmd == "shannon":
            self._handle_shannon(arg)

        elif cmd == "login":
            self._handle_login(arg)

        elif cmd == "help":
            self.show_help()

        else:
            console.print(f"Unknown command: /{cmd}", style="dim red")

        return True

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
        """Dispatch /shannon subcommands."""
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
                "Install it with: pip install cascade-cli",
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
        """Handle /login [provider] command."""
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
                    style=f"dim {CYAN}",
                )
                self.app.config.apply_credential(result.provider, result.token, overwrite=True)
                self.app.config.save()

    def run(self):
        """Start the REPL loop."""
        self.welcome()

        try:
            while True:
                try:
                    prompt = f"[{self.current_provider}] > "
                    line = input(prompt)

                    if not line.strip():
                        continue

                    if not self.handle_command(line):
                        break

                    if not line.startswith("/"):
                        self._ensure_session()
                        input_tokens = max(len(line) // 4, 1)
                        self._msg_count += 1
                        self._input_tokens += input_tokens
                        self.db.add_message(
                            self.session["id"], role="user", content=line,
                            token_count=input_tokens,
                        )
                        try:
                            response = self.app.ask(line, provider=self.current_provider)
                            output_tokens = max(len(response) // 4, 1)
                            self._resp_count += 1
                            self._output_tokens += output_tokens
                            self.db.add_message(
                                self.session["id"], role="assistant", content=response,
                                token_count=output_tokens,
                            )
                            if not self.session.get("title"):
                                title = line[:60]
                                self.db.update_session_title(self.session["id"], title)
                                self.session["title"] = title
                        except Exception as e:
                            console.print(f"Error: {e}", style="dim red")

                except KeyboardInterrupt:
                    console.print("\n")
                    continue

        except EOFError:
            self._print_exit_summary()


def main():
    """Entry point for `cascade` command.

    Creates the CLI CascadeCore (providers, config, hooks, tools),
    wraps it in a Textual CascadeTUI, and runs the fullscreen app.
    """
    from .cli import CascadeCore as CLIApp
    from .app import CascadeTUI

    cli_app = CLIApp()
    tui = CascadeTUI(cli_app=cli_app)
    tui.run(mouse=False)


if __name__ == "__main__":
    main()
