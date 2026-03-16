"""Main chat screen for the Cascade TUI.

Composes WelcomeHeader + ChatHistory + InputFrame + StatusBar.
Bridges to synchronous provider.stream() via run_worker(thread=True).
"""

import datetime

from rich.text import Text
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Input, Static

from ..providers.base import ProviderConfig, ToolEvent
from ..widgets.header import WelcomeHeader, ProviderGhostTable
from ..widgets.message import ChatHistory, MessageWidget, ThinkingIndicator
from ..widgets.input_frame import InputFrame
from ..widgets.status_bar import StatusBar
from ..widgets.stream_message import StreamMessage
from ..widgets.tool_call import ToolCallWidget
from ..theme import PALETTE, MODE_CYCLE, MODES
from ..commands import CommandHandler


def summarize_user_prompt(prompt: str) -> str:
    """Return a compact display string for pasted multi-line content."""
    line_count = prompt.count("\n") + 1
    if line_count >= 2:
        return f"[pasted content 1 + {line_count - 1} lines]"
    return prompt


class MainScreen(Screen):
    """The core chat interface."""

    BINDINGS = [
        ("shift+tab", "cycle_mode", "Cycle Mode"),
        ("ctrl+c", "exit_app", "Exit"),
        ("ctrl+d", "exit_app", "Exit"),
        ("escape", "blur_input", "Focus Chat"),
        ("pageup", "scroll_up", "Scroll Up"),
        ("pagedown", "scroll_down", "Scroll Down"),
    ]

    def __init__(
        self,
        active_provider: str = "gemini",
        mode: str = "design",
        providers: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._active_provider = active_provider
        self._mode = mode
        self._providers = providers or {}
        self._memory_policy = "summary"
        self._summary_turn_interval = 6
        self._summary_provider_pref = "auto"
        self._summary_max_chars = 1800
        self._cross_model_summary = ""
        self._turns_since_summary = 0
        self._summary_compaction_running = False
        self._header_visible = True
        self._cmd_handler: CommandHandler | None = None
        self._thinking: ThinkingIndicator | None = None

    def compose(self) -> ComposeResult:
        yield WelcomeHeader(
            active_provider=self._active_provider,
            providers=self._providers,
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
            self._summary_turn_interval = int(cfg.get("summary_turn_interval", 6))
            self._summary_provider_pref = str(cfg.get("summary_provider", "auto"))
            self._summary_max_chars = int(cfg.get("summary_max_chars", 1800))

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

        # Record in input history for up-arrow recall
        if hasattr(inp, "record"):
            inp.record(prompt)

        # Clear input
        event.input.value = ""

        # Slash commands
        if self._cmd_handler and self._cmd_handler.is_command(prompt):
            self._cmd_handler.handle(prompt)
            return

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

        # Mount user message widget
        chat = self.query_one(ChatHistory)
        chat.mount(MessageWidget("you", summarize_user_prompt(prompt)))
        chat.scroll_end(animate=False)

        # Kick off provider response in a worker thread
        self._send_to_provider(prompt)

    # ------------------------------------------------------------------
    # Provider streaming bridge
    # ------------------------------------------------------------------

    def _send_to_provider(self, prompt: str) -> None:
        """Start a background worker that calls the synchronous provider."""
        chat = self.query_one(ChatHistory)

        # Show thinking spinner
        self._thinking = ThinkingIndicator(self._active_provider)
        chat.mount(self._thinking)
        chat.scroll_end(animate=False)
        self.app.state.set_thinking(self._active_provider, True)

        # Mount a StreamMessage that will accumulate chunks
        self._stream_msg = StreamMessage(self._active_provider)
        chat.mount(self._stream_msg)

        provider_name = self._active_provider
        def _worker() -> None:
            self._provider_worker(prompt, provider_name)

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

    def _provider_worker(self, prompt: str, provider_name: str):
        """Run in a worker thread -- calls synchronous provider.stream() or ask_with_tools()."""
        cli_app = self.app.cli_app
        if cli_app is None:
            self.app.call_from_thread(self._on_stream_error, "No CLI app available")
            return

        prov = cli_app.providers.get(provider_name)
        if prov is None:
            self.app.call_from_thread(
                self._on_stream_error, f"Provider '{provider_name}' not available",
            )
            return

        # Build system prompt (no longer includes conversation history)
        final_system = self._build_system_prompt(cli_app, prompt, provider_name)

        # Build conversation history from state
        from ..conversation import state_messages_to_provider, needs_compaction, compact_messages
        messages = state_messages_to_provider(
            messages=list(self.app.state.messages),
            target_provider=provider_name,
            policy=self._memory_policy,
            cross_model_summary=self._cross_model_summary,
        )

        # Auto-compact if approaching context window limit
        if messages and needs_compaction(messages, provider_name):
            try:
                messages = compact_messages(messages, prov, keep_recent=6)
            except Exception:
                pass  # compaction failure is non-fatal; send uncompacted

        # Run hooks
        from ..hooks import HookEvent
        cli_app.hook_runner.run_hooks(HookEvent.BEFORE_ASK, context={
            "prompt": prompt,
            "provider": provider_name,
        })

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
            )
        else:
            self._stream_worker(cli_app, prov, messages, provider_name, final_system)

    def _stream_worker(self, cli_app, prov, messages, provider_name, final_system):
        """Streaming path -- token-by-token output."""
        from ..hooks import HookEvent

        full_response = []
        last_seen_activity = None
        # Extract the user prompt for record_turn (last user message)
        prompt = messages[-1]["content"] if messages else ""
        try:
            for chunk in prov.stream(messages, final_system):
                full_response.append(chunk)
                self.app.call_from_thread(self._on_stream_chunk, chunk)
                # Activity is now filtered in the provider layer; poll last_activity
                if prov.last_activity and prov.last_activity != last_seen_activity:
                    last_seen_activity = prov.last_activity
                    self.app.call_from_thread(self._on_stream_activity, last_seen_activity)

            response_text = "".join(full_response)
            if hasattr(cli_app, "record_turn"):
                cli_app.record_turn(provider_name, prompt, response_text)

            usage = prov.last_usage or (0, 0)
            self.app.call_from_thread(
                self._on_stream_done, provider_name, response_text, usage[0], usage[1],
            )

            cli_app.hook_runner.run_hooks(HookEvent.AFTER_RESPONSE, context={
                "response_length": str(len(response_text)),
                "provider": provider_name,
                "tool_calls": "0",
            })

        except Exception as e:
            self.app.call_from_thread(self._on_stream_error, str(e))

    def _tool_worker(self, cli_app, prov, messages, provider_name, final_system, tools):
        """Tool-calling path -- non-streaming with tool progress events."""
        from ..hooks import HookEvent

        # Extract the user prompt for record_turn (last user message)
        prompt = messages[-1]["content"] if messages else ""

        def on_tool_event(event: ToolEvent) -> None:
            self.app.call_from_thread(self._on_tool_event, event)

        try:
            response_text, tool_log = prov.ask_with_tools(
                messages,
                tools,
                system=final_system,
                on_tool_event=on_tool_event,
            )

            if hasattr(cli_app, "record_turn"):
                cli_app.record_turn(provider_name, prompt, response_text)

            usage = prov.last_usage or (0, 0)
            self.app.call_from_thread(
                self._on_tool_done,
                provider_name, response_text, usage[0], usage[1], tool_log,
            )

            cli_app.hook_runner.run_hooks(HookEvent.AFTER_RESPONSE, context={
                "response_length": str(len(response_text)),
                "provider": provider_name,
                "tool_calls": str(len(tool_log)),
            })

        except Exception as e:
            self.app.call_from_thread(self._on_stream_error, str(e))

    def _format_history_blocks(
        self,
        messages: list,
        max_messages: int,
        max_chars: int,
    ) -> str:
        selected = messages[-max_messages:]
        blocks = []
        total = 0
        for msg in selected:
            if msg.role == "system":
                continue
            if msg.role == "you":
                role = "User"
            elif msg.role in self._providers:
                role = f"Assistant ({msg.role})"
            else:
                role = "Assistant"
            content = msg.content.strip()
            if not content:
                continue
            if len(content) > 700:
                content = content[:700] + "..."
            block = f"{role}: {content}"
            block_len = len(block) + 2
            if total + block_len > max_chars:
                continue
            blocks.append(block)
            total += block_len
        return "\n\n".join(blocks)

    def _build_history_context(self, current_prompt: str, provider_name: str) -> str:
        """Build prompt context from prior turns according to memory policy."""
        if self._memory_policy == "off":
            return ""

        messages = list(self.app.state.messages)
        if messages and messages[-1].role == "you" and messages[-1].content == current_prompt:
            messages = messages[:-1]
        if not messages:
            return ""

        if self._memory_policy == "full":
            blocks = self._format_history_blocks(messages, max_messages=12, max_chars=6000)
            if not blocks:
                return ""
            return "Conversation history (recent turns):\n\n" + blocks

        # summary mode
        local_messages = [m for m in messages if m.role in ("you", provider_name)]
        local_blocks = self._format_history_blocks(local_messages, max_messages=8, max_chars=3200)

        parts = []
        if self._cross_model_summary:
            parts.append("Cross-model handoff summary:\n\n" + self._cross_model_summary)
        if local_blocks:
            parts.append("Current-provider recent turns:\n\n" + local_blocks)
        merged = "\n\n".join(parts).strip()
        if not merged:
            return ""
        return merged

    def _summary_transcript(self) -> str:
        """Create a compact transcript payload for summary generation."""
        messages = [m for m in self.app.state.messages if m.role != "system"]
        blocks = self._format_history_blocks(messages, max_messages=24, max_chars=7000)
        return blocks

    def _fallback_summary(self) -> str:
        """Heuristic summary when model-based compaction is unavailable."""
        transcript = self._summary_transcript()
        if not transcript:
            return ""
        lines = transcript.splitlines()
        objective = ""
        decisions = []
        files = []
        for ln in lines:
            if ln.startswith("User:") and not objective:
                objective = ln.replace("User:", "", 1).strip()
            if "Assistant" in ln and len(decisions) < 2:
                decisions.append(ln.split(":", 1)[-1].strip())
            for token in ln.split():
                clean = token.strip("`.,:;()[]{}\"'")
                if "/" in clean or clean.endswith((".py", ".md", ".yaml", ".yml", ".json", ".ts", ".tsx", ".js")):
                    if clean not in files:
                        files.append(clean)
                if len(files) >= 6:
                    break
        out = [
            f"- Objective: {objective or 'Continue current task.'}",
            f"- Recent decisions: {' | '.join(decisions) if decisions else 'None yet.'}",
            f"- Files/areas: {', '.join(files) if files else 'N/A'}",
            "- Open TODOs: Continue from latest unresolved request.",
        ]
        return "\n".join(out)[:self._summary_max_chars]

    def _summary_provider_candidates(self) -> list[str]:
        pref = self._summary_provider_pref.lower()
        if pref != "auto":
            return [pref]
        ordered = ["claude", "gemini", "openai", "openrouter", self._active_provider]
        out = []
        for name in ordered:
            if name in self._providers and name not in out:
                out.append(name)
        return out

    def _generate_cross_model_summary(self) -> str:
        """Generate/refresh handoff summary using a fast provider when possible."""
        cli_app = getattr(self.app, "cli_app", None)
        if cli_app is None:
            return self._fallback_summary()

        transcript = self._summary_transcript()
        if not transcript:
            return ""

        system = (
            "You produce compact engineering handoff summaries for model switching.\n"
            "Keep it factual, terse, and implementation-focused."
        )
        prompt = (
            "Update this cross-model summary for the ongoing coding session.\n"
            "Keep under 1800 characters.\n\n"
            "Format:\n"
            "- Objective\n"
            "- Design constraints\n"
            "- Decisions made\n"
            "- Files/areas touched\n"
            "- Open TODOs\n"
            "- Risks/questions\n\n"
            f"Previous summary:\n{self._cross_model_summary or '(none)'}\n\n"
            f"Recent transcript:\n{transcript}"
        )

        for provider_name in self._summary_provider_candidates():
            try:
                base_cfg = cli_app.config.get_provider_config(provider_name)
                provider_obj = cli_app.providers.get(provider_name)
                if base_cfg is None or provider_obj is None:
                    continue
                provider_data = cli_app.config.data.get("providers", {}).get(provider_name, {})
                summary_model = provider_data.get("fast_model") or base_cfg.model
                provider_cls = type(provider_obj)
                summary_provider = provider_cls(
                    ProviderConfig(
                        api_key=base_cfg.api_key,
                        model=summary_model,
                        base_url=base_cfg.base_url,
                        temperature=0.2,
                        max_tokens=900,
                    )
                )
                summary = summary_provider.ask_single(prompt, system=system).strip()
                if hasattr(summary_provider, "client"):
                    try:
                        summary_provider.client.close()
                    except Exception:
                        pass
                if summary:
                    return summary[:self._summary_max_chars]
            except Exception:
                continue
        return self._fallback_summary()

    def _on_summary_ready(self, summary: str) -> None:
        self._summary_compaction_running = False
        if summary:
            self._cross_model_summary = summary[:self._summary_max_chars]
        self._turns_since_summary = 0

    def _on_summary_error(self, _msg: str) -> None:
        self._summary_compaction_running = False

    def _trigger_summary_compaction(self, reason: str, force: bool = False) -> None:
        if self._memory_policy != "summary":
            return
        if self._summary_compaction_running:
            return
        if not force and self._turns_since_summary < self._summary_turn_interval:
            return
        if not self.app.state.messages:
            return

        self._summary_compaction_running = True
        if self._thinking:
            self._thinking.set_label(f"compacting memory ({reason})...")

        def _worker() -> None:
            try:
                summary = self._generate_cross_model_summary()
                self.app.call_from_thread(self._on_summary_ready, summary)
            except Exception as e:
                self.app.call_from_thread(self._on_summary_error, str(e))

        self.run_worker(_worker, thread=True, exclusive=False)

    def _on_stream_chunk(self, chunk: str) -> None:
        """Called from worker thread via app.call_from_thread."""
        if hasattr(self, "_stream_msg"):
            self._stream_msg.feed(chunk)
            chat = self.query_one(ChatHistory)
            chat.scroll_end(animate=False)

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
            chat.mount(ToolCallWidget(
                tool_name=event.tool_name,
                tool_input=event.tool_input,
                tool_output=event.tool_output,
            ))
            chat.scroll_end(animate=False)

    def _on_tool_done(
        self,
        provider: str,
        full_text: str,
        input_tokens: int,
        output_tokens: int,
        tool_log: list[dict],
    ) -> None:
        """Called when tool-calling loop completes."""
        # Remove thinking indicator
        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.app.state.set_thinking(provider, False)

        # Feed full response into the StreamMessage
        if hasattr(self, "_stream_msg"):
            self._stream_msg.feed(full_text)
            self._stream_msg.finish()

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

        # Update input frame token count
        try:
            self.query_one(InputFrame).token_count = self.app.state.total_tokens
        except Exception:
            pass

        self._turns_since_summary += 1
        self._trigger_summary_compaction(reason="periodic", force=False)

    def _on_stream_done(
        self, provider: str, full_text: str, input_tokens: int, output_tokens: int,
    ) -> None:
        """Called when streaming is complete."""
        # Remove thinking indicator
        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.app.state.set_thinking(provider, False)

        # Finalize the stream message
        if hasattr(self, "_stream_msg"):
            self._stream_msg.finish()

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

        # Update input frame token count
        try:
            self.query_one(InputFrame).token_count = self.app.state.total_tokens
        except Exception:
            pass

        self._turns_since_summary += 1
        self._trigger_summary_compaction(reason="periodic", force=False)

    def _on_stream_error(self, error_msg: str) -> None:
        """Called when streaming fails."""
        if self._thinking:
            self._thinking.remove()
            self._thinking = None
        self.app.state.set_thinking(self._active_provider, False)

        chat = self.query_one(ChatHistory)
        chat.mount(MessageWidget("system", f"Error: {error_msg}"))
        chat.scroll_end(animate=False)

    # ------------------------------------------------------------------
    # Mode cycling
    # ------------------------------------------------------------------

    def action_cycle_mode(self) -> None:
        previous_provider = self._active_provider
        current_idx = MODE_CYCLE.index(self._mode)
        next_idx = (current_idx + 1) % len(MODE_CYCLE)
        self._mode = MODE_CYCLE[next_idx]
        self._active_provider = MODES[self._mode]["provider"]

        # Update state
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
        chat.scroll_end(animate=False)

        if previous_provider != self._active_provider:
            self._trigger_summary_compaction(
                reason=f"switch {previous_provider}->{self._active_provider}",
                force=True,
            )

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def action_exit_app(self) -> None:
        from .exit import ExitScreen

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
