"""Main chat screen for the Cascade TUI.

Composes WelcomeHeader + ChatHistory + InputFrame + StatusBar.
Bridges to synchronous provider.stream() via run_worker(thread=True).
"""

import datetime
import time
from contextlib import nullcontext
from typing import Iterator

from rich.text import Text
from textual import events
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Input, Static

from ..episodes import generate_episode
from ..providers.base import ToolEvent
from ..providers.usage import Usage
from ..widgets.header import WelcomeHeader, ProviderGhostTable
from ..widgets.message import ChatHistory, MessageWidget, ThinkingIndicator
from ..widgets.input_frame import InputFrame
from ..widgets.status_bar import StatusBar
from ..widgets.stream_message import StreamMessage
from ..widgets.tool_call import ToolCallWidget, render_tool_widget
from ..theme import PALETTE, MODE_CYCLE, MODES, get_provider_theme
from ..commands import CommandHandler
from ..hooks import HookContext, HookEvent
from ..keybindings import ChordManager, ChordState
from ..swarm.lifecycle import RunCancelled, RunContext
from ..swarm.outcome import RunOutcome


def summarize_user_prompt(prompt: str) -> str:
    """Return a compact display string for pasted multi-line content."""
    line_count = prompt.count("\n") + 1
    if line_count >= 2:
        return f"[pasted content 1 + {line_count - 1} lines]"
    return prompt


class MainScreen(Screen):
    """The core chat interface."""

    _STREAM_BATCH_INTERVAL_SECONDS = 0.03
    _STREAM_BATCH_MAX_CHARS = 1024
    # A stream pause longer than this, right after a sentence, becomes a paragraph
    # break -- a model that pauses between thoughts then reads as separate paragraphs.
    _PAUSE_NEWLINE_SECONDS = 0.8
    # Tool-using chat is a light agent: more than the 5-round default so a model
    # can read a few files before answering, but well under /solve's 15 -- chat is
    # not a verified build, and big edit-test-commit tasks belong in /solve.
    _CHAT_TOOL_MAX_ROUNDS = 10
    # Ctrl+C on an empty input arms exit; a second press within this window
    # confirms it, so a stray Ctrl+C never quits mid-session.
    _EXIT_HINT = "press ctrl+c again to exit"
    _EXIT_WINDOW = 3.0

    BINDINGS = [
        ("shift+tab", "cycle_mode", "Cycle Mode"),
        ("ctrl+c", "exit_app", "Exit"),
        ("ctrl+d", "exit_app", "Exit"),
        ("escape", "blur_input", "Focus Chat"),
        ("pageup", "scroll_up", "Scroll Up"),
        ("pagedown", "scroll_down", "Scroll Down"),
        ("home", "scroll_home", "Scroll Top"),
        ("end", "scroll_end", "Scroll Bottom"),
    ]

    def __init__(
        self,
        active_provider: str = "gemini",
        mode: str = "design",
        providers: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._providers = providers or {}
        if self._providers and active_provider not in self._providers:
            active_provider = next(iter(self._providers))
            mode = get_provider_theme(active_provider).default_mode
        self._active_provider = active_provider
        self._mode = mode
        self._memory_policy = "summary"
        self._compaction_summary_enabled = True
        self._summary_failures = 0
        self._header_visible = True
        self._cmd_handler: CommandHandler | None = None
        self._thinking: ThinkingIndicator | None = None
        self._exit_hook_fired = False
        self._activity_timer = None
        self._activity_provider = None
        self._last_seen_activity = None
        self._active_run: RunContext | None = None
        self._exit_armed = False
        self._exit_timer = None
        self._chords = self._build_chord_manager()

    @staticmethod
    def _build_chord_manager() -> ChordManager:
        """Set up default chord bindings."""
        cm = ChordManager(timeout=1.0)
        cm.register("ctrl+x ctrl+k", "kill_workers")
        cm.register("ctrl+x ctrl+e", "export_session")
        cm.register("ctrl+x ctrl+h", "toggle_hooks")
        return cm

    def on_key(self, event: events.Key) -> None:
        """Route keypresses through the chord manager first."""
        result = self._chords.feed(event.key)
        if result.state == ChordState.PENDING:
            event.stop()
            event.prevent_default()
        elif result.state == ChordState.MATCHED:
            event.stop()
            event.prevent_default()
            handler = getattr(self, f"action_{result.action}", None)
            if handler is not None:
                handler()

    # ------------------------------------------------------------------
    # Chord actions (ctrl+x prefix)
    # ------------------------------------------------------------------

    def _post_system_message(self, text: str) -> None:
        """Post a system message via the command handler, with a toast fallback."""
        if self._cmd_handler is not None:
            self._cmd_handler._post_system(text)
        else:
            self.app.notify(text)

    def action_kill_workers(self) -> None:
        """Cancel the active run and reject all of its later callbacks."""
        if self._interrupt_active_run():
            return
        try:
            self.workers.cancel_all()
        except Exception:
            pass
        self._reset_cancelled_ui()
        self._post_system_message("No active run to cancel.")

    def _interrupt_active_run(self) -> bool:
        """Cancel the active run if any; return whether one was interrupted.

        Clearing the run identity is what makes callbacks already queued by the
        old worker stale. A subsequent prompt gets a new run id immediately.
        """
        run = self._active_run
        if run is None:
            return False
        reason = "cancelled by user"
        run.cancel(reason)
        run.finish(RunOutcome.CANCELLED, error=reason)
        self._active_run = None
        try:
            self.workers.cancel_all()
        except Exception:
            pass
        self._reset_cancelled_ui()
        self._post_system_message(
            f"Cancelled run {run.id[:8]}. Provider work will stop at its next checkpoint."
        )
        return True

    def _reset_cancelled_ui(self) -> None:
        """Return the input/UI to idle without recording an assistant response."""
        self._stop_activity_poll()
        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.app.state.set_thinking(self._active_provider, False)
        self._set_input_locked(False)
        stream = getattr(self, "_stream_msg", None)
        if stream is not None:
            try:
                stream.remove()
            except Exception:
                pass
            self._stream_msg = None

    def _call_for_run(
        self,
        run: RunContext,
        callback,
        *args,
        terminal: bool = False,
    ) -> None:
        """Schedule a UI callback that is accepted only for the active run id."""
        self.app.call_from_thread(
            self._dispatch_for_run,
            run.id,
            terminal,
            callback,
            args,
        )

    def _emit_for_run(
        self,
        run: RunContext | None,
        callback,
        *args,
        terminal: bool = False,
    ) -> None:
        """Use identity-gated dispatch, with a direct path for focused unit calls."""
        if run is None:
            self.app.call_from_thread(callback, *args)
        else:
            self._call_for_run(run, callback, *args, terminal=terminal)

    def _dispatch_for_run(self, run_id: str, terminal: bool, callback, args: tuple) -> None:
        """Drop callbacks from cancelled, superseded, or otherwise stale workers."""
        active = self._active_run
        if active is None or active.id != run_id or active.cancelled:
            return
        try:
            callback(*args)
        finally:
            if terminal and self._active_run is active:
                self._active_run = None

    def action_export_session(self) -> None:
        """Export the current session via the existing /export path (ctrl+x ctrl+e)."""
        if self._cmd_handler is not None:
            self._cmd_handler._cmd_export([])
        else:
            self.app.notify("Export unavailable: command handler not ready.")

    def action_toggle_hooks(self) -> None:
        """Toggle the hooks system on or off (ctrl+x ctrl+h)."""
        cli_app = getattr(self.app, "cli_app", None)
        runner = getattr(cli_app, "hook_runner", None)
        if runner is None:
            self._post_system_message("Hooks unavailable: no hook runner on this session.")
            return
        runner.enabled = not runner.enabled
        state = "enabled" if runner.enabled else "disabled"
        self._post_system_message(f"Hooks {state} ({runner.hook_count} registered).")

    def compose(self) -> ComposeResult:
        yield WelcomeHeader(
            active_provider=self._active_provider,
            providers=self._providers,
            id="welcome_header",
        )
        yield ChatHistory()
        yield InputFrame(
            active_provider=self._active_provider,
            mode=self._mode,
        )
        yield StatusBar(
            provider_tokens=dict(self.app.state.provider_tokens),
        )

    def on_mount(self) -> None:
        try:
            self.query_one("#main_input").focus()
        except Exception:
            pass
        self._cmd_handler = CommandHandler(self.app)
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is not None:
            cfg = cli_app.config.get_memory_config()
            self._memory_policy = str(cfg.get("cross_model_memory", "summary"))
            self._compaction_summary_enabled = bool(cfg.get("compaction_summary", True))
            engine = getattr(cli_app, "permission_engine", None)
            if engine is not None:
                engine.ask_handler = self._ask_permission
        recovered = int(getattr(self.app, "recovered_run_count", 0) or 0)
        if recovered:
            noun = "run" if recovered == 1 else "runs"
            self._post_system_message(
                f"Recovered {recovered} interrupted {noun} from the previous process. "
                "Any recorded review worktrees were preserved."
            )
            self.app.recovered_run_count = 0

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Multiline paste: ChatInput stores the full text in _pending_paste
        inp = event.input
        if hasattr(inp, "_pending_paste") and inp._pending_paste is not None:
            prompt = inp._pending_paste.strip()
            inp._pending_paste = None
        else:
            prompt = event.value.strip()
        if not prompt:
            return

        if self.app.state.is_thinking:
            if self._cmd_handler and self._cmd_handler.is_command(prompt):
                cmd = prompt.lstrip("/").split(None, 1)[0].lower()
                if cmd in {"exit", "quit"}:
                    self._cmd_handler.handle(prompt)
                    return
            self.app.notify("Wait for the current response to finish.")
            return

        # Record in input history for up-arrow recall
        if hasattr(inp, "record"):
            inp.record(prompt)

        # Clear input
        event.input.value = ""

        # Slash commands
        if self._cmd_handler and self._cmd_handler.is_command(prompt):
            self._cmd_handler.handle(prompt)
            return

        # Fire INPUT_RECEIVED hook (can transform prompt)
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is not None:
            hook_result = cli_app.hook_runner.emit(
                HookEvent.INPUT_RECEIVED,
                HookContext(
                    event=HookEvent.INPUT_RECEIVED.value,
                    prompt=prompt,
                    provider=self._active_provider,
                    mode=self._mode,
                ),
            )
            if hook_result is not None:
                if hook_result.block:
                    return
                if hook_result.transformed_value is not None:
                    prompt = hook_result.transformed_value

        # Hide welcome header on first real message
        if self._header_visible:
            self._header_visible = False
            try:
                self.query_one(WelcomeHeader).display = False
            except Exception:
                pass

        # Record user message in state + history DB
        self.app.state.add_message("you", prompt)
        self.app.record_message("user", prompt)

        # Mount user message widget and trim overflow
        chat = self.query_one(ChatHistory)
        chat.mount(MessageWidget("you", summarize_user_prompt(prompt)))
        self.call_later(chat.trim_overflow)
        self._scroll_chat_end(chat, force=True)

        # Kick off provider response in a worker thread
        self._send_to_provider(prompt)

    # ------------------------------------------------------------------
    # Provider streaming bridge
    # ------------------------------------------------------------------

    def _send_to_provider(self, prompt: str) -> None:
        """Start a background worker that calls the synchronous provider."""
        chat = self.query_one(ChatHistory)
        self._set_input_locked(True)

        # Show thinking spinner
        self._thinking = ThinkingIndicator(self._active_provider)
        chat.mount(self._thinking)
        self._scroll_chat_end(chat, force=True)
        self.app.state.set_thinking(self._active_provider, True)

        # Mount a StreamMessage that will accumulate chunks
        self._stream_msg = StreamMessage(self._active_provider)
        chat.mount(self._stream_msg)
        self._scroll_chat_end(chat, force=True)

        provider_name = self._active_provider
        session = self.app.ensure_session()
        provider = getattr(self.app, "cli_app", None)
        provider_obj = provider.providers.get(provider_name) if provider is not None else None
        model = getattr(getattr(provider_obj, "config", None), "model", "")
        run = RunContext(
            objective=prompt,
            workflow="routing",
            provider=provider_name,
            model=str(model or ""),
            session_id=session["id"],
            ledger=getattr(self.app, "run_ledger", None),
        )
        self._active_run = run
        self._start_activity_poll(self._providers.get(provider_name))
        def _worker() -> None:
            self._provider_worker(prompt, provider_name, run)

        self.run_worker(
            _worker,
            thread=True,
            exclusive=True,
        )

    @staticmethod
    def _should_use_tools(prov) -> bool:
        """Return True if the provider supports direct tool calling (not CLI proxy)."""
        if getattr(prov, "_use_cli_proxy", False):
            return False
        if getattr(prov, "_use_oauth_cli", False):
            return False
        return True

    def _build_system_prompt(self, cli_app, prompt: str, provider_name: str) -> str | None:
        """Build the final system prompt from the pipeline.

        Injects mode-specific directive and upload context.
        Conversation history is passed directly via messages, not here.
        """
        pipeline = cli_app.prompt_pipeline
        from ..prompts.layers import PRIORITY_MODE, PRIORITY_REPL_CONTEXT
        from ..prompts.default import get_mode_directive

        # Inject mode-specific directive
        directive = get_mode_directive(self._mode)
        if directive:
            pipeline = pipeline.add_layer("mode_directive", directive, PRIORITY_MODE)

        if cli_app.context_builder.source_count > 0:
            upload_ctx = cli_app.context_builder.build()
            pipeline = pipeline.add_layer(
                "upload_context", upload_ctx, PRIORITY_REPL_CONTEXT,
            )
        return pipeline.build() or None

    def _provider_worker(
        self,
        prompt: str,
        provider_name: str,
        run: RunContext | None = None,
    ):
        """Run in a worker thread -- calls synchronous provider.stream() or ask_with_tools()."""
        managed_run = run is not None
        run = run or RunContext(objective=prompt, provider=provider_name)

        def _call(callback, *args, terminal: bool = False) -> None:
            if managed_run:
                self._call_for_run(run, callback, *args, terminal=terminal)
            else:
                self.app.call_from_thread(callback, *args)

        try:
            run.checkpoint()
        except RunCancelled:
            return
        cli_app = self.app.cli_app
        if cli_app is None:
            run.finish(RunOutcome.FAILED, error="No CLI app available")
            _call(self._on_stream_error, "No CLI app available", terminal=True)
            return

        prov = cli_app.providers.get(provider_name)
        if prov is None:
            error = f"Provider '{provider_name}' not available"
            run.finish(RunOutcome.BLOCKED, error=error)
            _call(self._on_stream_error, error, terminal=True)
            return

        run.start(workflow="routing", provider=provider_name, model=prov.config.model)

        # Build system prompt (no longer includes conversation history)
        final_system = self._build_system_prompt(cli_app, prompt, provider_name)

        # Fire CONTEXT_BUILD hook (can inject/modify context)
        ctx_hook = cli_app.hook_runner.emit(
            HookEvent.CONTEXT_BUILD,
            HookContext(
                event=HookEvent.CONTEXT_BUILD.value,
                provider=provider_name,
                prompt=prompt,
                system_prompt=final_system or "",
            ),
        )
        if ctx_hook and ctx_hook.transformed_value is not None:
            final_system = str(ctx_hook.transformed_value)

        # Build conversation history from state, injecting episodes
        from ..conversation import (
            state_messages_to_provider, should_compact,
            compact_messages, compact_messages_with_episodes,
        )
        from ..episodes import prune_live_episodes

        # Episode-based compaction BEFORE building the payload: convert old
        # turns to episodes when real occupancy crosses the threshold or the
        # raw window would silently clip them. Anchored on the last round's
        # actual usage; the tail is a chars/4 estimate.
        chat_messages = list(self.app.state.messages)
        episode_list = list(self.app.state.episodes)
        summary_text = self.app.state.compaction_summary
        if not isinstance(summary_text, str):
            summary_text = ""

        state_anchor = self.app.state.context_anchor
        if not isinstance(state_anchor, Usage):
            state_anchor = None
        if chat_messages and should_compact(
            chat_messages,
            provider_name,
            model=prov.config.model or "",
            configured_window=prov.config.context_window,
            anchor=state_anchor,
        ):
            try:
                active_messages = [
                    msg for msg in chat_messages
                    if not msg.metadata.get("compacted")
                ]
                new_episodes, remaining = compact_messages_with_episodes(
                    chat_messages, keep_recent=6,
                )
                compacted_count = max(len(active_messages) - len(remaining), 0)
                if compacted_count > 0 or new_episodes:
                    kept_exchanges = sum(1 for m in remaining if m.role == "you")
                    remaining_ids = {id(m) for m in remaining}
                    compacted_msgs = [
                        m for m in active_messages if id(m) not in remaining_ids
                    ]
                    _call(
                        self.app.state.apply_episode_compaction,
                        compacted_count,
                        new_episodes,
                    )
                    _call(self.app.state.prune_live_episodes, kept_exchanges)
                    _call(self.app.state.mark_compaction)
                    # Queue the durable bookkeeping BEFORE the (slow) tier-2
                    # summary call so a mid-summary cancel still leaves the
                    # compaction fully applied, visible, and persisted.
                    _call(
                        self._post_compaction_note,
                        f"compacted {compacted_count} turns into "
                        f"{len(new_episodes)} episodes",
                    )
                    _call(self._refresh_context_display)
                    _call(self.app.persist_context)
                    # Tier 2: structured summary of the compacted range on a
                    # fast-tier clone. Episodes already carry the content, so
                    # a failed/skipped summary degrades quality, not safety.
                    new_summary = None
                    if live_run.token is None or not live_run.token.cancelled:
                        new_summary = self._generate_compaction_summary(
                            prov, provider_name, compacted_msgs,
                        )
                    if new_summary:
                        summary_text = new_summary
                        # Plain call_from_thread: idempotent state/DB
                        # snapshots must survive a cancelled run.
                        self.app.call_from_thread(
                            self.app.state.set_compaction_summary, new_summary,
                        )
                        self.app.call_from_thread(self.app.persist_context)
                    # Mirror the state change locally: this worker builds the
                    # payload from its own snapshot, not by re-reading state.
                    episode_list = prune_live_episodes(
                        episode_list, kept_exchanges,
                    ) + new_episodes
                    chat_messages = remaining
            except Exception as ep_err:
                import logging
                logging.getLogger("cascade").warning("Episode compaction failed: %s", ep_err)

        messages = state_messages_to_provider(
            messages=chat_messages,
            target_provider=provider_name,
            policy=self._memory_policy,
            episodes=episode_list if episode_list else None,
            compaction_summary=summary_text,
        )

        # Run BEFORE_ASK hooks (legacy)
        cli_app.hook_runner.run_hooks(HookEvent.BEFORE_ASK, context={
            "prompt": prompt,
            "provider": provider_name,
        })

        # Fire BEFORE_PROVIDER_REQUEST (can inspect/modify messages)
        req_hook = cli_app.hook_runner.emit(
            HookEvent.BEFORE_PROVIDER_REQUEST,
            HookContext(
                event=HookEvent.BEFORE_PROVIDER_REQUEST.value,
                provider=provider_name,
                prompt=prompt,
                messages=tuple(messages),
                system_prompt=final_system or "",
            ),
        )
        if req_hook and req_hook.block:
            error = f"Request blocked by hook: {req_hook.reason}"
            run.finish(RunOutcome.BLOCKED, error=error)
            _call(self._on_stream_error, error, terminal=True)
            return

        # Normal prompts in configured modes can be routed to a
        # capability-constrained workflow. This is intentionally after request
        # hooks/compaction, but before ordinary chat dispatch. Slash commands are
        # retained as explicit overrides and debugging controls.
        from ..swarm.auto import WorkflowKind, select_workflow, should_auto_orchestrate

        if should_auto_orchestrate(cli_app, self._mode):
            _call(self._on_stream_activity, "selecting workflow...")
            run.checkpoint()
            if managed_run:
                decision = select_workflow(
                    cli_app, prompt, self._mode, cancel_token=run.token,
                )
            else:
                decision = select_workflow(cli_app, prompt, self._mode)
            run.checkpoint()
            run.add_tokens(decision.input_tokens, decision.output_tokens)
            run.add_cost(decision.cost)
            if decision.workflow != WorkflowKind.CHAT:
                if managed_run:
                    self._auto_orchestration_worker(
                        cli_app, prompt, provider_name, decision, run,
                    )
                else:
                    self._auto_orchestration_worker(
                        cli_app, prompt, provider_name, decision,
                    )
                return

        run.start(workflow="chat")

        # Decide: tool-calling path or streaming path
        tool_registry = getattr(cli_app, "tool_registry", None)
        use_tools = (
            tool_registry
            and len(tool_registry) > 0
            and self._should_use_tools(prov)
        )

        if use_tools:
            self._tool_worker(
                cli_app, prov, messages, provider_name, final_system, tool_registry,
                run if managed_run else None,
            )
        else:
            self._stream_worker(
                cli_app, prov, messages, provider_name, final_system,
                run if managed_run else None,
            )

    def _auto_orchestration_worker(
        self,
        cli_app,
        prompt: str,
        provider_name: str,
        decision,
        run: RunContext | None = None,
    ) -> None:
        """Execute and record a model-selected non-chat workflow."""
        from ..swarm.auto import execute_auto

        live_run = run or RunContext(
            objective=prompt,
            workflow=decision.workflow.value,
            provider=provider_name,
        )
        live_run.start(workflow=decision.workflow.value)

        def _progress(stage: str, detail: str) -> None:
            live_run.checkpoint()
            self._emit_for_run(
                run,
                self._on_stream_activity,
                f"{stage}: {detail}",
            )

        def _tool_event(event: ToolEvent) -> None:
            live_run.checkpoint()
            self._emit_for_run(run, self._on_tool_event, event)

        try:
            result = execute_auto(
                cli_app,
                prompt,
                provider_name,
                decision,
                on_progress=_progress,
                on_tool_event=_tool_event,
                cancel_token=live_run.token,
                run_context=live_run,
            )
            live_run.checkpoint()
            response_text = result.text
            self._emit_for_run(run, self._on_stream_chunk, response_text)

            if hasattr(cli_app, "record_turn"):
                cli_app.record_turn(provider_name, prompt, response_text)

            # Orchestration ran in lane clones; the base provider's anchor
            # is stale for this turn. Honest display: unknown until the
            # next direct response.
            _call(self.app.state.set_context_anchor, None)
            _call(self._refresh_context_display)
            total_tokens = result.input_tokens + result.output_tokens
            episode = generate_episode(
                user_content=prompt,
                assistant_content=response_text,
                provider=provider_name,
                tokens=total_tokens,
            )
            self._emit_for_run(run, self.app.state.add_episode, episode)
            cli_app.hook_runner.emit(
                HookEvent.EPISODE_GENERATED,
                HookContext(
                    event=HookEvent.EPISODE_GENERATED.value,
                    provider=provider_name,
                    episode_id=episode.id,
                ),
            )
            live_run.finish(
                result.outcome,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost=result.cost,
            )
            self._emit_for_run(
                run,
                self._on_stream_done,
                provider_name,
                response_text,
                result.input_tokens,
                result.output_tokens,
                terminal=True,
            )
            cli_app.hook_runner.run_hooks(
                HookEvent.AFTER_RESPONSE,
                context={
                    "response_length": str(len(response_text)),
                    "provider": provider_name,
                    "tool_calls": "orchestrated",
                    "workflow": decision.workflow.value,
                    "outcome": result.outcome.value,
                },
            )
        except RunCancelled as exc:
            live_run.finish(RunOutcome.CANCELLED, error=str(exc))
        except Exception as exc:
            live_run.finish(RunOutcome.FAILED, error=str(exc))
            self._emit_for_run(
                run, self._on_stream_error, str(exc), terminal=True,
            )

    def _stream_worker(
        self,
        cli_app,
        prov,
        messages,
        provider_name,
        final_system,
        run: RunContext | None = None,
    ):
        """Streaming path -- token-by-token output."""
        full_response = []
        # Extract the user prompt for record_turn (last user message)
        prompt = messages[-1]["content"] if messages else ""
        live_run = run or RunContext(
            objective=prompt, workflow="chat", provider=provider_name,
        )
        live_run.start(workflow="chat", model=prov.config.model)
        try:
            cancellation_scope = getattr(prov, "cancellation_scope", None)
            scope = cancellation_scope(live_run.token) if callable(cancellation_scope) else nullcontext()
            with scope:
                last_time = time.monotonic()
                ended_sentence = False
                for chunk in self._coalesce_stream_chunks(prov.stream(messages, final_system)):
                    live_run.checkpoint()
                    now = time.monotonic()
                    chunk = self._pause_paragraph_break(
                        chunk, ended_sentence, now - last_time, self._PAUSE_NEWLINE_SECONDS
                    )
                    last_time = now
                    stripped = chunk.rstrip()
                    if stripped:
                        ended_sentence = stripped[-1] in ".!?:"
                    full_response.append(chunk)
                    self._emit_for_run(run, self._on_stream_chunk, chunk)

            live_run.checkpoint()
            response_text = "".join(full_response)
            if hasattr(cli_app, "record_turn"):
                cli_app.record_turn(provider_name, prompt, response_text)

            # Generate episode for this interaction
            usage = prov.last_usage or Usage()
            live_run.add_tokens(usage.prompt_total, usage.output)
            live_run.add_cost(usage.cost or 0.0)
            total_tokens = usage.total
            episode = generate_episode(
                user_content=prompt,
                assistant_content=response_text,
                provider=provider_name,
                tokens=total_tokens,
            )
            self._emit_for_run(run, self.app.state.add_episode, episode)
            cli_app.hook_runner.emit(
                HookEvent.EPISODE_GENERATED,
                HookContext(
                    event=HookEvent.EPISODE_GENERATED.value,
                    provider=provider_name,
                    episode_id=episode.id,
                ),
            )

            _call(self.app.state.set_context_anchor, prov.last_round_usage)
            live_run.finish(RunOutcome.SUCCEEDED)
            self._emit_for_run(
                run,
                self._on_stream_done,
                provider_name,
                response_text,
                usage.prompt_total,
                usage.output,
                terminal=True,
            )

            cli_app.hook_runner.run_hooks(HookEvent.AFTER_RESPONSE, context={
                "response_length": str(len(response_text)),
                "provider": provider_name,
                "tool_calls": "0",
            })

        except RunCancelled as exc:
            usage = prov.last_usage or Usage()
            live_run.add_tokens(usage.prompt_total, usage.output)
            live_run.add_cost(usage.cost or 0.0)
            live_run.finish(RunOutcome.CANCELLED, error=str(exc))
        except Exception as e:
            usage = prov.last_usage or Usage()
            live_run.add_tokens(usage.prompt_total, usage.output)
            live_run.add_cost(usage.cost or 0.0)
            live_run.finish(RunOutcome.FAILED, error=str(e))
            self._emit_for_run(run, self._on_stream_error, str(e), terminal=True)

    @staticmethod
    def _pause_paragraph_break(
        chunk: str, ended_sentence: bool, gap: float, threshold: float
    ) -> str:
        """Prefix *chunk* with a paragraph break after a long, sentence-ending pause.

        A model that stops between thoughts then reads as separate paragraphs. Only
        breaks when the previous text ended a sentence (``.!?:``) -- never mid-sentence
        -- and never doubles an existing leading newline, so it can't mangle output.
        """
        if ended_sentence and gap > threshold and not chunk.startswith("\n"):
            return "\n\n" + chunk
        return chunk

    @classmethod
    def _coalesce_stream_chunks(cls, chunks: Iterator[str]) -> Iterator[str]:
        """Batch rapid streaming chunks before they cross into the TUI thread.

        This preserves fast first-token feedback while reducing `call_from_thread`
        traffic and expensive StreamMessage re-layouts for providers that emit
        many tiny fragments.
        """
        pending: list[str] = []
        pending_chars = 0
        last_emit = -cls._STREAM_BATCH_INTERVAL_SECONDS

        for chunk in chunks:
            if not chunk:
                continue

            now = time.monotonic()
            if not pending and (now - last_emit) >= cls._STREAM_BATCH_INTERVAL_SECONDS:
                yield chunk
                last_emit = now
                continue

            pending.append(chunk)
            pending_chars += len(chunk)

            if (
                pending_chars >= cls._STREAM_BATCH_MAX_CHARS
                or (now - last_emit) >= cls._STREAM_BATCH_INTERVAL_SECONDS
            ):
                yield "".join(pending)
                pending.clear()
                pending_chars = 0
                last_emit = now

        if pending:
            yield "".join(pending)

    def _tool_worker(
        self,
        cli_app,
        prov,
        messages,
        provider_name,
        final_system,
        tools,
        run: RunContext | None = None,
    ):
        """Tool-calling path -- non-streaming with tool progress events."""
        # Extract the user prompt for record_turn (last user message)
        prompt = messages[-1]["content"] if messages else ""
        live_run = run or RunContext(
            objective=prompt, workflow="chat", provider=provider_name,
        )
        live_run.start(workflow="chat", model=prov.config.model)

        def on_tool_event(event: ToolEvent) -> None:
            live_run.checkpoint()
            self._emit_for_run(run, self._on_tool_event, event)

        try:
            cancellation_scope = getattr(prov, "cancellation_scope", None)
            scope = cancellation_scope(live_run.token) if callable(cancellation_scope) else nullcontext()
            with scope:
                response_text, tool_log = prov.ask_with_tools(
                    messages,
                    tools,
                    system=final_system,
                    max_rounds=self._CHAT_TOOL_MAX_ROUNDS,
                    on_tool_event=on_tool_event,
                )
            live_run.checkpoint()

            if hasattr(cli_app, "record_turn"):
                cli_app.record_turn(provider_name, prompt, response_text)

            # Generate episode with tool call data
            usage = prov.last_usage or Usage()
            live_run.add_tokens(usage.prompt_total, usage.output)
            live_run.add_cost(usage.cost or 0.0)
            total_tokens = usage.total
            episode = generate_episode(
                user_content=prompt,
                assistant_content=response_text,
                provider=provider_name,
                tokens=total_tokens,
                tool_log=tool_log,
            )
            self._emit_for_run(run, self.app.state.add_episode, episode)
            cli_app.hook_runner.emit(
                HookEvent.EPISODE_GENERATED,
                HookContext(
                    event=HookEvent.EPISODE_GENERATED.value,
                    provider=provider_name,
                    episode_id=episode.id,
                ),
            )

            _call(self.app.state.set_context_anchor, prov.last_round_usage)
            live_run.finish(RunOutcome.SUCCEEDED)
            self._emit_for_run(
                run,
                self._on_tool_done,
                provider_name, response_text, usage.prompt_total, usage.output, tool_log,
                terminal=True,
            )

            cli_app.hook_runner.run_hooks(HookEvent.AFTER_RESPONSE, context={
                "response_length": str(len(response_text)),
                "provider": provider_name,
                "tool_calls": str(len(tool_log)),
            })

        except RunCancelled as exc:
            usage = prov.last_usage or Usage()
            live_run.add_tokens(usage.prompt_total, usage.output)
            live_run.add_cost(usage.cost or 0.0)
            live_run.finish(RunOutcome.CANCELLED, error=str(exc))
        except Exception as e:
            usage = prov.last_usage or Usage()
            live_run.add_tokens(usage.prompt_total, usage.output)
            live_run.add_cost(usage.cost or 0.0)
            live_run.finish(RunOutcome.FAILED, error=str(e))
            self._emit_for_run(run, self._on_stream_error, str(e), terminal=True)

    def _generate_compaction_summary(
        self,
        prov,
        provider_name: str,
        compacted,
        custom: str = "",
    ):
        """Tier-2 structured summary of a compacted range, breaker-guarded.

        Runs on a fast-tier immutable clone of the active provider (never
        mutates shared config). Two consecutive failures disable the
        summarizer for the session; episodes still carry the content.
        Returns the validated summary text or None.
        """
        if not self._compaction_summary_enabled:
            return None
        from dataclasses import replace as dc_replace

        from ..conversation import SUMMARY_MIN_CHARS, summarize_for_compaction

        if sum(len(m.content) for m in compacted) < SUMMARY_MIN_CHARS:
            return None

        cli_app = getattr(self.app, "cli_app", None)
        clone = None
        summary = None
        try:
            cfg = prov.config
            fast_model = cfg.model
            if cli_app is not None:
                candidate = cli_app.config.get_model_for(
                    provider_name, self._mode, fast=True,
                )
                if isinstance(candidate, str) and candidate:
                    fast_model = candidate
            clone = type(prov)(
                dc_replace(cfg, model=fast_model, temperature=0.2, max_tokens=8000)
            )
            summary = summarize_for_compaction(
                clone.ask_single,
                compacted,
                previous_summary=self.app.state.compaction_summary,
                custom_instructions=custom,
            )
        except Exception:
            summary = None
        finally:
            client = getattr(clone, "client", None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

        if summary is None:
            self._summary_failures += 1
            if self._summary_failures >= 2:
                self._compaction_summary_enabled = False
        else:
            self._summary_failures = 0
        return summary

    def _ask_permission(self, tool_name: str, arguments: dict, verdict) -> str:
        """Resolve an "ask" verdict by prompting the user.

        Called from a worker thread inside the tool loop; blocks that
        worker until the modal resolves (timeout denies). Never call on
        the UI thread -- the guard below turns that into a deny rather
        than a deadlock.
        """
        import threading

        from .permission import PermissionScreen

        result: dict = {}
        done = threading.Event()
        screen_ref: dict = {}

        def _show() -> None:
            def _resolved(answer) -> None:
                result["answer"] = answer or "deny"
                done.set()

            try:
                screen_ref["screen"] = PermissionScreen(
                    tool_name, arguments, verdict.reason,
                )
                self.app.push_screen(screen_ref["screen"], _resolved)
            except Exception:
                result["answer"] = "deny"
                done.set()

        try:
            self.app.call_from_thread(_show)
        except Exception:
            return "deny"
        if not done.wait(timeout=120.0):
            # Timed out waiting for the user: dismiss the modal so it does
            # not linger on the screen stack after we return deny.
            def _dismiss() -> None:
                scr = screen_ref.get("screen")
                if scr is not None and scr.is_running:
                    try:
                        scr.dismiss("deny")
                    except Exception:
                        pass

            try:
                self.app.call_from_thread(_dismiss)
            except Exception:
                pass
            return "deny"
        return result.get("answer", "deny")

    def _post_compaction_note(self, text: str) -> None:
        """Mount a dim one-line separator noting a compaction event."""
        try:
            chat = self.query_one(ChatHistory)
            note = Static(
                Text(f"─── {text} ───", style=f"dim {PALETTE.text_dim}"),
                classes="bookmark",
            )
            chat.mount(note)
            chat.scroll_end(animate=False)
        except Exception:
            pass

    def _refresh_context_display(self) -> None:
        """Push current context occupancy into the status bar and input frame.

        One accounting source: budget thresholds + the state anchor (the
        last round's real usage). Anchor None renders as "ctx ?".
        """
        from ..context.budget import compact_threshold, warn_threshold, window_for

        state = self.app.state
        provider_name = state.active_provider
        cli_app = getattr(self.app, "cli_app", None)
        prov = cli_app.providers.get(provider_name) if cli_app else None
        model = (prov.config.model if prov else "") or ""
        configured = prov.config.context_window if prov else None
        window = window_for(provider_name, model, configured)
        threshold = compact_threshold(window)
        warn = warn_threshold(window)
        anchor = state.context_anchor

        try:
            bar = self.query_one(StatusBar)
            frame = self.query_one(InputFrame)
        except Exception:
            return

        if anchor is None:
            bar.update_context(None, threshold, warn, state.compaction_count)
            frame.context_label = "ctx ?" if state.compaction_count else ""
            return

        from ..conversation import unsent_tail_chars
        from ..context.budget import estimate_tokens_from_chars

        active = [m for m in state.messages if not m.metadata.get("compacted")]
        tokens = anchor.total + estimate_tokens_from_chars(unsent_tail_chars(active))
        pct = min(tokens * 100 // threshold, 999) if threshold > 0 else 0
        bar.update_context(tokens, threshold, warn, state.compaction_count)
        if tokens >= 1000:
            tok_str = f"{tokens / 1000:.1f}k"
        else:
            tok_str = str(tokens)
        frame.context_label = f"ctx {tok_str} · {pct}%"

    def _on_stream_chunk(self, chunk: str) -> None:
        """Called from worker thread via app.call_from_thread."""
        if hasattr(self, "_stream_msg"):
            chat = self.query_one(ChatHistory)
            follow = self._should_follow_chat(chat)
            self._stream_msg.feed(chunk)
            if follow:
                self._scroll_chat_end(chat, force=True)

    def _on_stream_activity(self, activity: str) -> None:
        """Show live provider activity while waiting for model output."""
        if self._thinking:
            if len(activity) > 100:
                activity = activity[:97] + "..."
            self._thinking.set_label(activity)

    def _on_tool_event(self, event: ToolEvent) -> None:
        """Handle tool progress events on the main thread."""
        if event.kind == "tool_start":
            if self._thinking:
                self._thinking.set_label(f"calling {event.tool_name}...")
        elif event.kind == "tool_done":
            if self._thinking:
                self._thinking.set_label(f"{event.tool_name} done")
            chat = self.query_one(ChatHistory)
            follow = self._should_follow_chat(chat)
            chat.mount(render_tool_widget(
                event.tool_name, event.tool_input, event.tool_output,
            ))
            if follow:
                self._scroll_chat_end(chat, force=True)

    def _on_tool_done(
        self,
        provider: str,
        full_text: str,
        input_tokens: int,
        output_tokens: int,
        tool_log: list[dict],
    ) -> None:
        """Called when tool-calling loop completes."""
        self._stop_activity_poll()

        # Remove thinking indicator
        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.app.state.set_thinking(provider, False)
        self._set_input_locked(False)

        # Feed full response into the StreamMessage
        if hasattr(self, "_stream_msg"):
            self._stream_msg.feed(full_text)
            self._stream_msg.finish()
            self._stream_msg = None

        try:
            self._scroll_chat_end(self.query_one(ChatHistory))
        except Exception:
            pass

        # Record in state + history DB
        total = input_tokens + output_tokens
        self.app.state.add_message(provider, full_text, tokens=total)
        self.app.state.update_tokens(provider, input_tokens, output_tokens)
        self.app.record_message(provider, full_text, token_count=total)

        # Update status bar
        try:
            self.query_one(StatusBar).update_tokens(self.app.state.provider_tokens)
        except Exception:
            pass

        self._refresh_context_display()


    def _on_stream_done(
        self, provider: str, full_text: str, input_tokens: int, output_tokens: int,
    ) -> None:
        """Called when streaming is complete."""
        self._stop_activity_poll()

        # Remove thinking indicator
        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.app.state.set_thinking(provider, False)
        self._set_input_locked(False)

        # Finalize the stream message
        if hasattr(self, "_stream_msg"):
            self._stream_msg.finish()
            self._stream_msg = None

        try:
            chat = self.query_one(ChatHistory)
            self._scroll_chat_end(chat)
            self.call_later(chat.trim_overflow)
        except Exception:
            pass

        # Record in state + history DB
        total = input_tokens + output_tokens
        self.app.state.add_message(provider, full_text, tokens=total)
        self.app.state.update_tokens(provider, input_tokens, output_tokens)
        self.app.record_message(provider, full_text, token_count=total)

        # Update status bar
        try:
            self.query_one(StatusBar).update_tokens(self.app.state.provider_tokens)
        except Exception:
            pass

        self._refresh_context_display()


    def _on_stream_error(self, error_msg: str) -> None:
        """Called when streaming fails."""
        self._stop_activity_poll()

        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.app.state.set_thinking(self._active_provider, False)
        self._set_input_locked(False)

        if hasattr(self, "_stream_msg") and self._stream_msg is not None:
            self._stream_msg.finish()
            self._stream_msg = None

        chat = self.query_one(ChatHistory)
        follow = self._should_follow_chat(chat)
        chat.mount(MessageWidget("system", f"Error: {error_msg}"))
        if follow:
            self._scroll_chat_end(chat, force=True)

    # ------------------------------------------------------------------
    # Mode cycling
    # ------------------------------------------------------------------

    def action_cycle_mode(self) -> None:
        previous_provider = self._active_provider
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is not None and hasattr(cli_app, "config"):
            candidate_modes = cli_app.config.get_available_modes(self._providers.keys())
            if isinstance(candidate_modes, tuple) and all(isinstance(mode, str) for mode in candidate_modes):
                available_modes = tuple(mode for mode in candidate_modes if mode in MODES)
            else:
                from ..theme import get_available_modes
                available_modes = get_available_modes(self._providers.keys())
        else:
            from ..theme import get_available_modes
            available_modes = get_available_modes(self._providers.keys())
        if not available_modes:
            available_modes = MODE_CYCLE
        if self._mode not in available_modes:
            next_mode = available_modes[0]
        elif len(available_modes) == 1:
            return
        else:
            current_idx = available_modes.index(self._mode)
            next_idx = (current_idx + 1) % len(available_modes)
            next_mode = available_modes[next_idx]
        self._mode = next_mode
        if cli_app is not None and hasattr(cli_app, "config"):
            configured_provider = cli_app.config.get_mode_provider(self._mode)
            if isinstance(configured_provider, str) and configured_provider in self._providers:
                self._active_provider = configured_provider
            else:
                self._active_provider = MODES[self._mode]["provider"]
            prov = cli_app.providers.get(self._active_provider)
            if prov is not None:
                model = cli_app.config.get_model_for(self._active_provider, self._mode, fast=False)
                if isinstance(model, str) and model:
                    prov.config.model = model
        else:
            self._active_provider = MODES[self._mode]["provider"]

        # Update state
        self.app.state.fast_mode = False
        self.app.state.set_provider(self._active_provider, self._mode)

        # Update widgets
        try:
            inp = self.query_one(InputFrame)
            inp.active_provider = self._active_provider
            inp.mode = self._mode
        except Exception:
            pass

        try:
            self.query_one(ProviderGhostTable).set_active(self._active_provider)
        except Exception:
            pass

        # Insert bookmark separator
        chat = self.query_one(ChatHistory)
        now = datetime.datetime.now().strftime("%I:%M %p")
        sep_text = Text(
            f"\u2500\u2500\u2500 {now} . switching to {self._mode} mode \u2500\u2500\u2500",
            style=f"dim {PALETTE.text_dim}",
        )
        sep = Static(sep_text, classes="bookmark")
        chat.mount(sep)
        self._scroll_chat_end(chat, force=True)

        if previous_provider != self._active_provider:
            # Fire PROVIDER_SWITCH hook
            cli_app = getattr(self.app, "cli_app", None)
            if cli_app is not None:
                cli_app.hook_runner.emit(
                    HookEvent.PROVIDER_SWITCH,
                    HookContext(
                        event=HookEvent.PROVIDER_SWITCH.value,
                        provider=self._active_provider,
                        mode=self._mode,
                        metadata=(("previous_provider", previous_provider),),
                    ),
                )

            # Auto-branch on provider switch
            try:
                bs = self.app.get_branching_session()
                label = f"{previous_provider}->{self._active_provider}"
                bs.create_branch(label=label, provider=self._active_provider)
            except Exception:
                pass  # branching failure is non-fatal


    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def action_exit_app(self) -> None:
        """Ctrl+C: interrupt a run, else clear a filled input, else confirm-exit.

        Mirrors Claude Code -- a first press interrupts an in-flight generation
        or clears typed input; only a second press on an empty input exits, so a
        stray Ctrl+C (or a terminal folding Ctrl+Shift+C into it) never quits.
        """
        if self._interrupt_active_run():
            self._disarm_exit()
            return

        inp = self._input_widget()
        if inp is not None and (inp.value or getattr(inp, "_pending_paste", None)):
            inp.value = ""
            inp._pending_paste = None
            self._disarm_exit()
            return

        if not self._exit_armed:
            self._exit_armed = True
            self.flash_status(self._EXIT_HINT, self._EXIT_WINDOW)
            try:
                self._exit_timer = self.set_timer(self._EXIT_WINDOW, self._disarm_exit)
            except Exception:
                # No running loop (or scheduling failed): keep the arm, just
                # without an auto-expiry. The exit handler must never crash.
                self._exit_timer = None
            return

        self._disarm_exit()
        self._perform_exit()

    def _input_widget(self):
        """The prompt Input widget, or None if it is not mounted."""
        try:
            return self.query_one("#main_input", Input)
        except Exception:
            return None

    def _disarm_exit(self) -> None:
        """Cancel a pending second-press-to-exit."""
        self._exit_armed = False
        if self._exit_timer is not None:
            self._exit_timer.stop()
            self._exit_timer = None

    def flash_status(self, message: str, timeout: float = 1.5) -> None:
        """Show a brief, unobtrusive note in the bottom-right status corner."""
        try:
            self.query_one(StatusBar).flash(message, timeout)
        except Exception:
            pass

    def _perform_exit(self) -> None:
        from .exit import ExitScreen

        active_run = self._active_run
        if active_run is not None:
            reason = "cancelled because Cascade is exiting"
            active_run.cancel(reason)
            active_run.finish(RunOutcome.CANCELLED, error=reason)
            self._active_run = None
            try:
                self.workers.cancel_all()
            except Exception:
                pass

        if not self._exit_hook_fired:
            cli_app = getattr(self.app, "cli_app", None)
            if cli_app is not None:
                session_id = (
                    self.app._db_session["id"]
                    if getattr(self.app, "_db_session", None) is not None
                    else self.app.state.session_id
                )
                cli_app.hook_runner.emit(
                    HookEvent.ON_EXIT,
                    HookContext(
                        event=HookEvent.ON_EXIT.value,
                        provider=self._active_provider,
                        mode=self._mode,
                        session_id=session_id,
                        metadata=(("messages", str(self.app.state.message_count)),),
                    ),
                )
            self._exit_hook_fired = True

        elapsed = self.app.state.elapsed
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        uptime = f"{minutes:02d}:{seconds:02d}"

        self.app.push_screen(ExitScreen(
            session_id=self.app.state.session_id,
            uptime=uptime,
            messages_sent=self.app.state.message_count,
            messages_received=self.app.state.response_count,
            tokens=dict(self.app.state.provider_tokens),
        ))

    def action_blur_input(self) -> None:
        try:
            self.query_one("#main_input").blur()
            self.query_one(ChatHistory).focus()
        except Exception:
            pass

    def action_scroll_up(self) -> None:
        try:
            self.query_one(ChatHistory).scroll_page_up(animate=False)
        except Exception:
            pass

    def action_scroll_down(self) -> None:
        try:
            self.query_one(ChatHistory).scroll_page_down(animate=False)
        except Exception:
            pass

    def action_scroll_home(self) -> None:
        try:
            self.query_one(ChatHistory).scroll_home(animate=False)
        except Exception:
            pass

    def action_scroll_end(self) -> None:
        try:
            self.query_one(ChatHistory).scroll_end(animate=False)
        except Exception:
            pass

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        try:
            self.query_one(ChatHistory).scroll_relative(y=-6, animate=False, force=True)
            event.stop()
            event.prevent_default()
        except Exception:
            pass

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        try:
            self.query_one(ChatHistory).scroll_relative(y=6, animate=False, force=True)
            event.stop()
            event.prevent_default()
        except Exception:
            pass

    def _set_input_locked(self, locked: bool) -> None:
        try:
            inp = self.query_one("#main_input", Input)
            inp.disabled = locked
            if not locked:
                inp.focus()
        except Exception:
            pass

    @staticmethod
    def _should_follow_chat(chat: ChatHistory, threshold: float = 2.0) -> bool:
        try:
            return (chat.max_scroll_y - chat.scroll_y) <= threshold
        except Exception:
            return True

    def _scroll_chat_end(self, chat: ChatHistory, force: bool = False) -> None:
        if force or self._should_follow_chat(chat):
            chat.scroll_end(animate=False)

    def _start_activity_poll(self, provider) -> None:
        self._stop_activity_poll()
        self._activity_provider = provider
        self._last_seen_activity = None
        if provider is None:
            return

        def _tick() -> None:
            if self._thinking is None or self._activity_provider is None:
                return
            activity = getattr(self._activity_provider, "last_activity", None)
            if activity and activity != self._last_seen_activity:
                self._last_seen_activity = activity
                self._on_stream_activity(activity)

        self._activity_timer = self.set_interval(0.2, _tick)

    def _stop_activity_poll(self) -> None:
        if self._activity_timer is not None:
            self._activity_timer.stop()
            self._activity_timer = None
        self._activity_provider = None
        self._last_seen_activity = None
