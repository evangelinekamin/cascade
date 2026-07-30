"""Cascade CLI - Beautiful multi-model AI assistant."""

import click
import json
import os
import subprocess
import sys
import time
from typing import Optional

from .auth import detect_all
from .config import ConfigManager
from .context import ProjectContext
from .hooks import HookContext, HookEvent, HookRunner, load_hooks_from_config
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
from .providers.usage import Usage
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
        self.permission_engine = self._build_permission_engine()
        self._wire_provider_hooks()
        self.tool_registry = self._build_tool_registry()
        self.context_builder = ContextBuilder()
        self.last_response_meta: Optional[ProviderResponse] = None
        self.last_tool_log: tuple[dict, ...] = ()

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
        self._wire_provider_hooks()

    def _build_permission_engine(self):
        """Build the popup-free broker with conservative project inheritance."""
        from .tools.permissions import PermissionEngine
        from .tools.reviewer import ModelPermissionReviewer

        cfg = self.config.get_permissions_config()
        project = self.project.permissions if isinstance(self.project.permissions, dict) else {}
        posture = PermissionEngine.normalize_posture(cfg.get("posture"))
        # Checked-in policy may tighten the user's posture, never loosen it.
        posture_rank = {"yolo": 0, "auto": 1, "safe": 2, "readonly": 3}
        project_posture = str(project.get("posture") or "").lower()
        if (
            project_posture in posture_rank
            and posture_rank[project_posture] > posture_rank[posture]
        ):
            posture = project_posture

        def _project_tightens(key: str) -> tuple:
            extra = project.get(key)
            extra = [str(v) for v in extra] if isinstance(extra, list) else []
            return tuple(cfg[key] + extra)

        reviewer_cfg = cfg.get("reviewer")
        reviewer_cfg = reviewer_cfg if isinstance(reviewer_cfg, dict) else {}
        self._permission_reviewer_config = dict(reviewer_cfg)
        review_handler = None
        if reviewer_cfg.get("enabled", True):
            review_handler = ModelPermissionReviewer(
                self._new_permission_reviewer_provider,
                timeout=float(reviewer_cfg.get("timeout", 10.0)),
            )

        return PermissionEngine(
            posture=posture,
            # A repository cannot teach the broker to trust itself.
            allow=tuple(cfg["allow"]),
            deny=_project_tightens("deny"),
            ask=_project_tightens("ask"),
            workspace_root=os.getcwd(),
            review_handler=review_handler,
        )

    def _new_permission_reviewer_provider(self):
        """Create a fresh direct provider for one non-agentic safety review."""
        from dataclasses import replace

        reviewer_cfg = getattr(self, "_permission_reviewer_config", {})
        requested_provider = str(reviewer_cfg.get("provider") or "")
        requested_model = str(reviewer_cfg.get("model") or "")
        try:
            orchestration = self.config.get_orchestration_config()
        except Exception:
            orchestration = {}
        if not isinstance(orchestration, dict):
            orchestration = {}
        router_provider = str(orchestration.get("router_provider") or "")
        router_model = str(orchestration.get("router_model") or "")
        try:
            default_provider = str(self.config.get_default_provider() or "")
        except Exception:
            default_provider = ""
        providers = getattr(self, "providers", {})

        candidates = []
        for name in (
            requested_provider,
            router_provider,
            default_provider,
            "openrouter",
            "openai",
            *providers.keys(),
        ):
            if name and name not in candidates:
                candidates.append(name)

        for name in candidates:
            base = providers.get(name)
            if base is None:
                continue
            if (
                getattr(base, "_use_cli_proxy", False)
                or getattr(base, "_use_oauth_cli", False)
            ):
                continue
            model = (
                requested_model
                if requested_model and name == requested_provider
                else router_model
                if router_model and name == router_provider
                else base.config.model
            )
            try:
                config = replace(
                    base.config,
                    model=model,
                    temperature=0.0,
                    max_tokens=200,
                )
                reviewer = type(base)(config)
            except Exception:
                continue
            reviewer.name = name
            reviewer.hook_runner = None
            reviewer.permission_engine = None
            return reviewer
        return None

    def _wire_provider_hooks(self) -> None:
        """Attach the hook runner + permission gate to every provider."""
        runner = getattr(self, "hook_runner", None)
        engine = getattr(self, "permission_engine", None)
        for provider in self.providers.values():
            if runner is not None:
                provider.hook_runner = runner
            if engine is not None:
                provider.permission_engine = engine

    def _build_prompt_pipeline(self) -> PromptPipeline:
        """Assemble the system prompt pipeline from config and project context."""
        prompt_config = self.config.get_prompt_config()
        pipeline = PromptPipeline()

        if prompt_config.get("use_default_system_prompt", True):
            # The base pipeline is mode-agnostic: design.md is mode-specific
            # content and is injected per-request against the ACTIVE mode by the
            # TUI assembler (a static default mode here would ride every request
            # regardless of a later Shift+Tab).
            default_prompt = build_default_prompt(include_design_language=False)
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
        render: bool = True,
    ) -> str:
        """Ask a single question with full system prompt, tools, and hooks."""
        prov = self.get_provider(provider)

        input_hook = self.hook_runner.emit(
            HookEvent.INPUT_RECEIVED,
            HookContext(
                event=HookEvent.INPUT_RECEIVED.value,
                provider=prov.name,
                prompt=prompt,
            ),
        )
        if input_hook is not None:
            if input_hook.block:
                raise RuntimeError(f"Input blocked by hook: {input_hook.reason}")
            if input_hook.transformed_value is not None:
                if not isinstance(input_hook.transformed_value, str):
                    raise RuntimeError("Input hook returned an invalid prompt (expected text)")
                prompt = input_hook.transformed_value

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
        context_hook = self.hook_runner.emit(
            HookEvent.CONTEXT_BUILD,
            HookContext(
                event=HookEvent.CONTEXT_BUILD.value,
                provider=prov.name,
                prompt=prompt,
                system_prompt=final_system or "",
            ),
        )
        if context_hook is not None:
            if context_hook.block:
                raise RuntimeError(f"Context blocked by hook: {context_hook.reason}")
            if context_hook.transformed_value is not None:
                final_system = str(context_hook.transformed_value)

        # Run BEFORE_ASK hooks
        before_results = self.hook_runner.run_hooks(HookEvent.BEFORE_ASK, context={
            "prompt": prompt,
            "provider": prov.name,
        })
        blocked = next((result for result in before_results if result.get("blocked")), None)
        if blocked is not None:
            raise RuntimeError(f"Request blocked by hook: {blocked.get('output', '')}")

        # Build messages list (CLI mode uses single-turn by default;
        # conversation context is already in the system prompt pipeline)
        messages = [{"role": "user", "content": prompt}]
        request_hook = self.hook_runner.emit(
            HookEvent.BEFORE_PROVIDER_REQUEST,
            HookContext(
                event=HookEvent.BEFORE_PROVIDER_REQUEST.value,
                provider=prov.name,
                prompt=prompt,
                messages=tuple(messages),
                system_prompt=final_system or "",
            ),
        )
        if request_hook is not None:
            if request_hook.block:
                raise RuntimeError(f"Request blocked by hook: {request_hook.reason}")
            if request_hook.transformed_value is not None:
                transformed = request_hook.transformed_value
                if not (
                    isinstance(transformed, (list, tuple))
                    and all(isinstance(message, dict) for message in transformed)
                ):
                    raise RuntimeError(
                        "Request hook returned invalid messages (expected a list of objects)"
                    )
                messages = [dict(message) for message in transformed]

        # Ask with or without tools
        tool_log = []
        try:
            if self.tool_registry and not stream:
                response, tool_log = prov.ask_with_tools(
                    messages, self.tool_registry, system=final_system,
                )
            elif stream:
                response = stream_response(prov.stream(messages, final_system), prov.name)
            else:
                response = prov.ask(messages, final_system)
        except Exception as exc:
            self.hook_runner.emit(
                HookEvent.ON_ERROR,
                HookContext(
                    event=HookEvent.ON_ERROR.value,
                    provider=prov.name,
                    prompt=prompt,
                    error=str(exc),
                ),
            )
            raise

        # Capture response metadata from provider
        usage = prov.last_usage or Usage()
        self.last_response_meta = ProviderResponse(
            text=response,
            input_tokens=usage.prompt_total,
            output_tokens=usage.output,
            model=prov.config.model,
            provider=prov.name,
        )
        self.last_tool_log = tuple(dict(entry) for entry in tool_log)

        if not stream and render:
            render_response(response, provider=prov.name)

        self.record_turn(prov.name, prompt, response)

        # Run AFTER_RESPONSE hooks
        self.hook_runner.emit(
            HookEvent.AFTER_RESPONSE,
            HookContext(
                event=HookEvent.AFTER_RESPONSE.value,
                provider=prov.name,
                prompt=prompt,
                response=response,
                tool_log=tuple(dict(entry) for entry in tool_log),
            ),
        )

        return response

    def run_automatic(
        self,
        prompt: str,
        *,
        provider: Optional[str] = None,
        mode: str = "build",
    ) -> dict:
        """Run an ordinary prompt through the same automatic router as the TUI."""
        from .harness import summarize_run
        from .prompts.default import get_mode_directive
        from .swarm.auto import WorkflowKind, execute_auto, select_workflow

        provider_name = provider or self.config.get_default_provider()
        prov = self.get_provider(provider_name)
        started = time.perf_counter()
        decision = select_workflow(self, prompt, mode)
        if decision.workflow == WorkflowKind.CHAT:
            response = self.ask(
                prompt,
                provider=provider_name,
                system=get_mode_directive(mode),
                render=False,
            )
            meta = self.last_response_meta or ProviderResponse(
                text=response,
                provider=prov.name,
                model=prov.config.model,
            )
            usage = prov.last_usage or Usage()
            metrics = summarize_run(
                self.last_tool_log,
                usage,
                time.perf_counter() - started,
            )
            changed_files: list[str] = []
            diff_stat = ""
            try:
                status = subprocess.run(
                    ["git", "status", "--short"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if status.returncode == 0:
                    changed_files = [
                        line[3:].split(" -> ")[-1]
                        for line in status.stdout.splitlines()
                        if len(line) > 3
                    ]
                diff = subprocess.run(
                    ["git", "diff", "--stat"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if diff.returncode == 0:
                    diff_stat = diff.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
            return {
                "schema_version": 1,
                "objective": prompt,
                "outcome": "succeeded" if response.strip() else "failed",
                "workflow": decision.workflow.value,
                "route_reason": decision.reason,
                "route_confidence": decision.confidence,
                "router_provider": decision.router_provider,
                "router_model": decision.router_model,
                "history_hint": decision.history_hint,
                "provider": provider_name,
                "model": meta.model,
                "text": response,
                "worktree_path": os.getcwd(),
                "changed_files": changed_files,
                "diff_stat": diff_stat,
                "verification_kind": "",
                "iterations": 1,
                "duration_seconds": metrics.duration_seconds,
                "tool_metrics_available": True,
                "tool_calls": metrics.tool_calls,
                "tool_errors": metrics.tool_errors,
                "duplicate_reads": metrics.duplicate_reads,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "cost": metrics.cost or 0.0,
            }

        result = execute_auto(self, prompt, provider_name, decision, mode=mode)
        return {
            "schema_version": 1,
            "objective": prompt,
            "outcome": result.outcome.value,
            "workflow": result.decision.workflow.value,
            "route_reason": result.decision.reason,
            "route_confidence": result.decision.confidence,
            "router_provider": result.decision.router_provider,
            "router_model": result.decision.router_model,
            "history_hint": result.decision.history_hint,
            "provider": result.execution_provider,
            "model": " -> ".join(dict.fromkeys(result.models_used)),
            "text": result.text,
            "worktree_path": result.worktree_path,
            "changed_files": list(result.changed_files),
            "diff_stat": result.diff_stat,
            "verification_kind": result.verification_kind,
            "iterations": result.iterations,
            "duration_seconds": time.perf_counter() - started,
            # Native CLI proxies own their internal tool loop; until a provider
            # emits structured tool events, zero would be a false measurement.
            "tool_metrics_available": False,
            "tool_calls": None,
            "tool_errors": None,
            "duplicate_reads": None,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost": result.cost,
            "tokens_by_provider": [list(item) for item in result.tokens_by_provider],
            "cost_by_provider": [list(item) for item in result.cost_by_provider],
        }

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


@cli.command(name="run")
@click.argument("prompt", nargs=-1, required=True)
@click.option("--provider", "-p", help="Provider to use for implementation.")
@click.option(
    "--mode",
    type=click.Choice(["design", "plan", "build", "test"]),
    default="build",
    show_default=True,
)
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable run receipt.")
def run_prompt(prompt, provider, mode, as_json):
    """Run one prompt with automatic chat/recon/solve/pipeline/fanout routing."""
    app = get_app()
    try:
        result = app.run_automatic(" ".join(prompt), provider=provider, mode=mode)
    except Exception as exc:
        if as_json:
            click.echo(json.dumps({"outcome": "failed", "error": str(exc)}))
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        console.print(result["text"])


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable diagnostics.")
@click.option("--refresh", is_flag=True, help="Re-probe CLIs instead of using the cache.")
def doctor(as_json, refresh):
    """Check provider CLIs, auth configuration, Git, and permission capabilities."""
    from .capabilities import format_doctor, run_doctor

    app = get_app()
    report = run_doctor(app.config, refresh=refresh)
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        console.print(format_doctor(report))


@cli.command(name="eval")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--provider", "-p", default="", help="Provider under evaluation.")
@click.option(
    "--mode",
    type=click.Choice(["design", "plan", "build", "test"]),
    default="build",
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=str))
@click.option("--keep", is_flag=True, help="Keep fixture/worktree paths for debugging.")
@click.option(
    "--task",
    "task_ids",
    multiple=True,
    help="Run only this task ID; repeat to select several.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the complete JSON report.")
def evaluate(manifest, provider, mode, output, keep, task_ids, as_json):
    """Run a manifest of real repository tasks and independently verify them."""
    from pathlib import Path

    from .evaluation import load_eval_manifest, run_evaluation, select_eval_tasks

    try:
        tasks = select_eval_tasks(load_eval_manifest(manifest), task_ids)
        report = run_evaluation(
            tasks,
            manifest=str(Path(manifest).resolve()),
            provider=provider,
            mode=mode,
            keep=keep,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    payload = report.to_dict()
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        try:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded + "\n", encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(f"Could not write evaluation report: {exc}") from exc
    if as_json:
        click.echo(encoded)
    else:
        console.print(
            f"Evaluation: {report.passed}/{report.total} passed "
            f"({report.pass_rate:.0%})"
        )
        for result in report.results:
            mark = "PASS" if result.passed else "FAIL"
            detail = result.error or result.workflow or result.outcome
            console.print(f"  {mark} {result.id}: {detail}")
        if output:
            console.print(f"  report: {output}", style="dim")


@cli.command(name="benchmark")
@click.option(
    "--repeats",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Number of timed serial and parallel samples.",
)
@click.option(
    "--calls",
    "calls_per_repeat",
    type=click.IntRange(min=2),
    default=4,
    show_default=True,
    help="Independent tool calls in each sample.",
)
@click.option(
    "--delay",
    "delay_seconds",
    type=click.FloatRange(min=0),
    default=0.02,
    show_default=True,
    help="Synthetic latency per tool call, in seconds.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=str),
    help="Write the JSON report to this path.",
)
@click.option(
    "--baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Compare the result with a previous JSON report.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the full JSON report.")
def benchmark(
    repeats,
    calls_per_repeat,
    delay_seconds,
    output,
    baseline,
    as_json,
):
    """Benchmark local harness mechanics without calling a model."""
    import json
    from pathlib import Path

    from .harness import compare_reports, run_harness_benchmark

    report = run_harness_benchmark(
        repeats=repeats,
        calls_per_repeat=calls_per_repeat,
        delay_seconds=delay_seconds,
    )
    payload = report.to_dict()
    if baseline:
        try:
            baseline_payload = json.loads(Path(baseline).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise click.ClickException(f"Could not read baseline: {exc}") from exc
        if not isinstance(baseline_payload, dict):
            raise click.ClickException("Baseline must contain one JSON object.")
        payload["baseline_delta"] = compare_reports(report, baseline_payload)

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        try:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded + "\n")
        except OSError as exc:
            raise click.ClickException(f"Could not write benchmark report: {exc}") from exc

    if as_json:
        click.echo(encoded)
        return

    console.print("\n[bold]Local harness benchmark[/bold]", style=CYAN)
    console.print(
        f"  tool batch: {report.serial_seconds * 1000:.2f} ms serial → "
        f"{report.parallel_seconds * 1000:.2f} ms parallel "
        f"({report.speedup:.2f}×)"
    )
    console.print(
        f"  hook dispatch: {report.hook_p50_ms:.3f} ms p50 / "
        f"{report.hook_p95_ms:.3f} ms p95"
    )
    console.print(
        f"  ordering: {'ok' if report.results_ordered else 'FAILED'} · "
        f"tool errors: {report.tool_errors} · schema: {report.schema_bytes} bytes"
    )
    if output:
        console.print(f"  report: {output}", style="dim")
    if "baseline_delta" in payload:
        delta = payload["baseline_delta"]
        parallel_delta = delta.get("parallel_seconds_pct")
        if parallel_delta is not None:
            console.print(
                f"  versus baseline: parallel latency {parallel_delta:+.1f}%",
                style="dim",
            )
    console.print()


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
