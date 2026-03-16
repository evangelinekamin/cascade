"""Cascade CLI - Beautiful multi-model AI assistant."""

import click
import sys
from typing import Optional

from .auth import detect_all
from .config import ConfigManager
from .context import ProjectContext
from .hooks import HookEvent, HookRunner, load_hooks_from_config
from .prompts import build_default_prompt, PromptPipeline
from .prompts.layers import (
    PRIORITY_DEFAULT,
    PRIORITY_PROJECT_SYSTEM,
    PRIORITY_PROJECT_CONTEXT,
    PRIORITY_USER_OVERRIDE,
    PRIORITY_REPL_CONTEXT,
)
from .providers.registry import discover_providers, get_registry
from .providers.response import ProviderResponse
from .tools import build_tool_registry
from .ui import render_header, render_response, render_comparison
from .ui.output import render_error, stream_response
from .ui.theme import console, CYAN, VIOLET
from .plugins import FileOpsPlugin
from .agents.loader import load_agents_from_dict
from .agents.runner import AgentRunner
from .agents.workflow import load_workflows_from_dict, WorkflowRunner
from .context.memory import ContextBuilder


class CascadeCore:
    """Main Cascade application."""

    def __init__(self):
        self.config = ConfigManager()
        self.memory_config = self.config.get_memory_config()
        self.credentials = detect_all()
        self._apply_detected_credentials()
        self.providers = {}
        self._conversation: list[tuple[str, str, str]] = []
        self._conversation_by_provider: dict[str, list[tuple[str, str]]] = {}
        self._cross_model_summary: str = ""
        self._last_provider_for_memory: Optional[str] = None
        self._summary_turns_since_compact: int = 0
        self._init_providers()
        self.file_ops = FileOpsPlugin()
        self.project = ProjectContext()
        self.prompt_pipeline = self._build_prompt_pipeline()
        self.hook_runner = self._build_hook_runner()
        self.tool_registry = self._build_tool_registry()
        self.context_builder = ContextBuilder()
        self.last_response_meta: Optional[ProviderResponse] = None

        # Agent & workflow system
        self.agents = load_agents_from_dict(self.project.agents)
        self.workflows = load_workflows_from_dict(
            self.project.agents.get("workflows", {}),
        )
        self._agent_runner = AgentRunner(self)
        self._workflow_runner = WorkflowRunner(
            self._agent_runner, self.agents,
        )

    def _apply_detected_credentials(self) -> None:
        """Auto-enable providers from detected CLI credentials."""
        for cred in self.credentials:
            self.config.apply_credential(cred.provider, cred.token)

    def _init_providers(self) -> None:
        """Initialize enabled providers from the registry."""
        # Rebuild from scratch so config reload drops disabled providers.
        self.providers = {}
        discover_providers()
        provider_classes = get_registry()

        for provider_name, provider_class in provider_classes.items():
            config = self.config.get_provider_config(provider_name)
            if config:
                try:
                    self.providers[provider_name] = provider_class(config)
                except Exception as e:
                    console.print(f"Failed to initialize {provider_name}: {e}", style="dim red")

    def _build_prompt_pipeline(self) -> PromptPipeline:
        """Assemble the system prompt pipeline from config and project context."""
        prompt_config = self.config.get_prompt_config()
        pipeline = PromptPipeline()

        if prompt_config.get("use_default_system_prompt", True):
            default_prompt = build_default_prompt(
                include_design_language=prompt_config.get("include_design_language", True),
                design_md_path=prompt_config.get("design_md_path") or None,
            )
            pipeline = pipeline.add_layer("default", default_prompt, PRIORITY_DEFAULT)

        if self.project.found:
            if self.project.system_prompt:
                pipeline = pipeline.add_layer(
                    "project_system", self.project.system_prompt, PRIORITY_PROJECT_SYSTEM,
                )
            for name, content in self.project.context_files.items():
                pipeline = pipeline.add_layer(
                    f"project_context:{name}", content, PRIORITY_PROJECT_CONTEXT,
                )

        return pipeline

    def _build_hook_runner(self) -> HookRunner:
        """Load hooks from config."""
        hooks_data = self.config.get_hooks_config()
        hooks = load_hooks_from_config(hooks_data)
        return HookRunner(hooks)

    def _build_tool_registry(self) -> dict:
        """Build tool registry from enabled plugins."""
        tools_config = self.config.get_tools_config()

        # Ensure plugins are loaded
        from .plugins import get_plugin_registry
        registry = build_tool_registry()

        # Filter by tools config
        filtered = {}
        for tool_name, tool_def in registry.items():
            # Check if the plugin that provides this tool is enabled
            enabled = True
            for plugin_name, is_enabled in tools_config.items():
                if not is_enabled:
                    # Check if this tool belongs to the disabled plugin
                    plugin_registry = get_plugin_registry()
                    if plugin_name in plugin_registry:
                        plugin_tools = plugin_registry[plugin_name]().get_tools()
                        if tool_name in plugin_tools:
                            enabled = False
                            break
            if enabled:
                filtered[tool_name] = tool_def

        return filtered

    def get_provider(self, name: Optional[str] = None):
        """Get a provider by name or default."""
        provider_name = name or self.config.get_default_provider()

        if provider_name not in self.providers:
            raise click.ClickException(
                f"Provider '{provider_name}' not found or not enabled. "
                f"Available: {list(self.providers.keys())}"
            )

        return self.providers[provider_name]

    def _memory_policy(self) -> str:
        return str(self.memory_config.get("cross_model_memory", "summary"))

    def _build_turn_blocks(
        self,
        turns: list[tuple[str, str]],
        max_turns: int,
        max_chars: int,
    ) -> str:
        if not turns:
            return ""
        selected = turns[-max_turns:]
        blocks = []
        total = 0
        for user_text, assistant_text in selected:
            u = user_text.strip()
            a = assistant_text.strip()
            if len(u) > 600:
                u = u[:600] + "..."
            if len(a) > 900:
                a = a[:900] + "..."
            block = f"User: {u}\nAssistant: {a}"
            block_len = len(block) + 2
            if total + block_len > max_chars:
                continue
            blocks.append(block)
            total += block_len
        return "\n\n".join(blocks)

    def _compact_summary_heuristic(self) -> None:
        """Refresh cross-model summary without extra model calls."""
        if not self._conversation:
            self._cross_model_summary = ""
            self._summary_turns_since_compact = 0
            return

        recent = self._conversation[-10:]
        providers = []
        goals = []
        latest_files = []
        for provider_name, user_text, assistant_text in recent:
            if provider_name not in providers:
                providers.append(provider_name)
            goals.append(user_text.strip().replace("\n", " ")[:140])
            # Lightweight file hint extraction for handoff continuity.
            for token in (user_text + "\n" + assistant_text).split():
                if "/" in token or token.endswith((".py", ".md", ".yaml", ".yml", ".json", ".ts", ".tsx", ".js")):
                    cleaned = token.strip("`.,:;()[]{}\"'")
                    if cleaned and cleaned not in latest_files:
                        latest_files.append(cleaned)
                if len(latest_files) >= 8:
                    break

        summary_lines = [
            "Cross-model handoff summary:",
            f"- Recent providers: {', '.join(providers[-4:])}",
        ]
        if goals:
            summary_lines.append(f"- Latest objective: {goals[-1]}")
        if len(goals) > 1:
            summary_lines.append(f"- Prior objective: {goals[-2]}")
        if latest_files:
            summary_lines.append(f"- Files/areas touched: {', '.join(latest_files[:8])}")

        summary = "\n".join(summary_lines)
        max_chars = int(self.memory_config.get("summary_max_chars", 1800))
        self._cross_model_summary = summary[:max_chars]
        self._summary_turns_since_compact = 0

    def _build_conversation_context(self, provider_name: str) -> str:
        """Build conversation context according to memory policy."""
        policy = self._memory_policy()
        if policy == "off":
            return ""

        if policy == "full":
            turns = [(u, a) for _p, u, a in self._conversation]
            blocks = self._build_turn_blocks(turns, max_turns=8, max_chars=6000)
            if not blocks:
                return ""
            return "Conversation history (recent turns):\n\n" + blocks

        # summary mode
        parts = []
        if self._cross_model_summary:
            parts.append(self._cross_model_summary)

        local_turns = self._conversation_by_provider.get(provider_name, [])
        blocks = self._build_turn_blocks(local_turns, max_turns=5, max_chars=3200)
        if blocks:
            parts.append("Current-provider recent turns:\n\n" + blocks)

        return "\n\n".join(parts).strip()

    def record_turn(self, provider_name: str, prompt: str, response: str) -> None:
        """Record a completed turn for memory context."""
        self._conversation.append((provider_name, prompt, response))
        if len(self._conversation) > 48:
            self._conversation = self._conversation[-48:]

        provider_turns = self._conversation_by_provider.setdefault(provider_name, [])
        provider_turns.append((prompt, response))
        if len(provider_turns) > 24:
            self._conversation_by_provider[provider_name] = provider_turns[-24:]

        policy = self._memory_policy()
        if policy == "summary":
            switched = (
                self._last_provider_for_memory is not None
                and self._last_provider_for_memory != provider_name
            )
            self._summary_turns_since_compact += 1
            interval = int(self.memory_config.get("summary_turn_interval", 6))
            if switched or self._summary_turns_since_compact >= interval or not self._cross_model_summary:
                self._compact_summary_heuristic()
        self._last_provider_for_memory = provider_name

    def ask(
        self,
        prompt: str,
        provider: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False,
        context_text: Optional[str] = None,
    ) -> str:
        """Ask a single question with full system prompt, tools, and hooks."""
        prov = self.get_provider(provider)

        # Build the system prompt from pipeline
        pipeline = self.prompt_pipeline
        if context_text:
            pipeline = pipeline.add_layer("repl_context", context_text, PRIORITY_REPL_CONTEXT)
        history_context = self._build_conversation_context(prov.name)
        if history_context:
            pipeline = pipeline.add_layer("conversation_history", history_context, PRIORITY_REPL_CONTEXT)
        if system:
            pipeline = pipeline.add_layer("user_override", system, PRIORITY_USER_OVERRIDE)

        final_system = pipeline.build() or None

        # Run BEFORE_ASK hooks
        self.hook_runner.run_hooks(HookEvent.BEFORE_ASK, context={
            "prompt": prompt,
            "provider": prov.name,
        })

        # Build messages list (CLI mode uses single-turn by default;
        # conversation context is already in the system prompt pipeline)
        messages = [{"role": "user", "content": prompt}]

        # Ask with or without tools
        tool_log = []
        if self.tool_registry and not stream:
            response, tool_log = prov.ask_with_tools(
                messages, self.tool_registry, system=final_system,
            )
        elif stream:
            response = stream_response(prov.stream(messages, final_system), prov.name)
        else:
            response = prov.ask(messages, final_system)

        # Capture response metadata from provider
        usage = prov.last_usage or (0, 0)
        self.last_response_meta = ProviderResponse(
            text=response,
            input_tokens=usage[0],
            output_tokens=usage[1],
            model=prov.config.model,
            provider=prov.name,
        )

        if not stream:
            render_response(response, provider=prov.name)

        self.record_turn(prov.name, prompt, response)

        # Run AFTER_RESPONSE hooks
        self.hook_runner.run_hooks(HookEvent.AFTER_RESPONSE, context={
            "response_length": str(len(response)),
            "provider": prov.name,
            "tool_calls": str(len(tool_log)),
        })

        return response

    def run_agent(self, name: str, prompt: str) -> str:
        """Run a named agent by name. Raises KeyError if not found."""
        agent = self.agents[name]
        return self._agent_runner.run(agent, prompt)

    def run_workflow(self, name: str, prompt: str) -> str:
        """Run a named workflow by name. Raises KeyError if not found."""
        workflow = self.workflows[name]
        return self._workflow_runner.run(workflow, prompt)

    def compare(self, prompt: str, providers: Optional[list[str]] = None) -> list[dict]:
        """Compare responses from multiple providers."""
        provider_names = providers or list(self.providers.keys())

        if not provider_names:
            raise click.ClickException("No providers available for comparison")

        results = []
        for provider_name in provider_names:
            if provider_name not in self.providers:
                continue

            prov = self.providers[provider_name]
            response = prov.compare(prompt)
            results.append(response)

        render_comparison(results)
        return results

    def chat(self, provider: Optional[str] = None) -> None:
        """Interactive chat mode."""
        prov = self.get_provider(provider)
        render_header("CASCADE CHAT", f"Model: {prov.config.model}")
        console.print("\nType 'quit' or 'exit' to leave\n", style=f"dim {CYAN}")

        messages = []

        try:
            while True:
                prompt = click.prompt(f"[{prov.name}]", default="", show_default=False)

                if prompt.lower() in ["quit", "exit"]:
                    console.print("Goodbye!", style=f"dim {VIOLET}")
                    break

                if not prompt.strip():
                    continue

                response = prov.ask_single(prompt)
                render_response(response, provider=prov.name)
                messages.append({"prompt": prompt, "response": response})
        except KeyboardInterrupt:
            console.print("\n\nInterrupted.", style="dim red")

    def analyze(self, file_path: str, prompt: Optional[str] = None, provider: Optional[str] = None) -> str:
        """Analyze a file with AI."""
        content = self.file_ops.read_file(file_path)
        if content.startswith("Error"):
            raise click.ClickException(content)

        analysis_prompt = prompt or "Analyze this code and provide insights:"
        full_prompt = f"{analysis_prompt}\n\n```\n{content}\n```"

        prov = self.get_provider(provider)
        response = prov.ask_single(full_prompt)
        render_response(response, provider=prov.name)

        return response


# Global app instance
_app = None

def get_app() -> CascadeCore:
    """Get or create the app instance."""
    global _app
    if _app is None:
        _app = CascadeCore()
    return _app


# CLI Commands
@click.group()
def cli():
    """CASCADE - Multi-model AI assistant CLI.

    Ask questions, compare providers, chat interactively, analyze files.
    """
    pass


@cli.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--provider", "-p", help="Provider to use (gemini, claude)")
@click.option("--system", "-s", help="System prompt")
@click.option("--stream", is_flag=True, help="Stream response in real-time")
def ask(prompt, provider, system, stream):
    """Ask a single question."""
    app = get_app()
    prompt_text = " ".join(prompt)
    try:
        app.ask(prompt_text, provider=provider, system=system, stream=stream)
    except click.ClickException:
        raise
    except Exception as e:
        render_error(str(e))
        sys.exit(1)


@cli.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--providers", "-p", multiple=True, help="Providers to compare")
def compare(prompt, providers):
    """Compare responses from multiple providers."""
    app = get_app()
    prompt_text = " ".join(prompt)
    try:
        app.compare(prompt_text, providers=list(providers) if providers else None)
    except click.ClickException:
        raise
    except Exception as e:
        render_error(str(e))
        sys.exit(1)


@cli.command()
@click.option("--provider", "-p", help="Provider to use")
def chat(provider):
    """Start interactive chat mode (TUI)."""
    from .repl import main as tui_main
    tui_main()


@cli.command()
@click.argument("file_path")
@click.option("--prompt", "-p", help="Custom analysis prompt")
@click.option("--provider", "-pr", help="Provider to use")
def analyze(file_path, prompt, provider):
    """Analyze a file with AI."""
    app = get_app()
    try:
        app.analyze(file_path, prompt=prompt, provider=provider)
    except click.ClickException:
        raise
    except Exception as e:
        render_error(str(e))
        sys.exit(1)


@cli.command()
def config():
    """Show configuration."""
    app = get_app()
    config_path = app.config.config_path

    console.print(f"Config file: {config_path}")
    console.print(f"Enabled providers: {app.config.get_enabled_providers()}")
    console.print(f"Default provider: {app.config.get_default_provider()}")


@cli.command()
def setup():
    """Run the interactive setup wizard."""
    from .setup_flow import SetupWizard

    wizard = SetupWizard()
    wizard.run()


@cli.command(name="init")
@click.argument("project_type", required=False, default=None)
def init_project(project_type):
    """Initialize a .cascade/ project directory."""
    from pathlib import Path
    from .agents.templates import detect_project_type, PROJECT_TYPES
    from .agents.init import run_init

    project_dir = Path(".").resolve()
    detected = detect_project_type(project_dir)

    console.print("\n[bold]CASCADE Init[/bold]", style=CYAN)
    console.print(f"Project directory: {project_dir}\n", style="dim")

    if project_type is None:
        console.print(f"Detected project type: {detected}", style=f"bold {CYAN}")
        console.print()
        for i, pt in enumerate(PROJECT_TYPES, 1):
            marker = " (detected)" if pt == detected else ""
            console.print(f"  {i}. {pt}{marker}")
        console.print()

        try:
            choice = input(f"Project type [{detected}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = ""

        if choice:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(PROJECT_TYPES):
                    project_type = PROJECT_TYPES[idx]
                else:
                    project_type = detected
            except ValueError:
                project_type = choice if choice in PROJECT_TYPES else detected
        else:
            project_type = detected

    console.print(f"\nUsing template: {project_type}\n", style="dim")

    # Feature toggles
    features = {"system_prompt": True, "agents": True, "context": True}
    for feat in features:
        try:
            answer = input(f"  Enable {feat}? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer in ("n", "no"):
            features[feat] = False

    console.print()
    summary = run_init(
        project_dir,
        project_type,
        print_fn=lambda msg: console.print(msg, style="dim"),
        enable_system_prompt=features["system_prompt"],
        enable_agents=features["agents"],
        enable_context=features["context"],
    )
    console.print(f"\n{summary}\n", style=f"bold {CYAN}")


@cli.command()
@click.option("--limit", "-n", default=20, help="Number of sessions to show")
@click.option("--search", "-s", default="", help="Search sessions by keyword")
def history(limit, search):
    """Show conversation history."""
    from .history import HistoryDB

    db = HistoryDB()
    sessions = db.search_sessions(search, limit=limit) if search else db.list_sessions(limit=limit)

    if not sessions:
        console.print("No sessions found.", style="dim")
        return

    console.print(f"\nConversation history ({len(sessions)} sessions):\n", style=f"bold {CYAN}")
    for s in sessions:
        title = s["title"] or "(untitled)"
        console.print(
            f"  {s['id']}  {title}  [{s['provider']}]  {s['created_at'][:16]}",
            style="dim",
        )
    console.print()
    db.close()


if __name__ == "__main__":
    cli()
