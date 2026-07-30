"""Google Gemini provider implementation.

Supports two auth paths:
- Gemini API key (direct HTTP requests)
- Gemini CLI OAuth token (``ya29.*``), proxied through ``gemini -p``
"""

from typing import Optional, Iterator, TYPE_CHECKING
import json
import os
import shutil
import httpx
from .base import BaseProvider, ProviderConfig, Message, ToolEvent, ToolEventCallback
from ._cli_proxy import CLIProxyConfig, GeminiEventHandler, stream_cli_proxy
from ._openai_tools import (
    _CHARS_PER_TOKEN,
    _CONTEXT_BUDGET_FRACTION,
    _KEEP_RECENT_MESSAGES,
    _invalidates_read,
    _read_dedup_key,
)
from .registry import register_provider
from .usage import Usage
from ..context.budget import window_for

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


def _estimate_gemini_tokens(contents: list[dict]) -> int:
    """Rough token count (~1 token / 4 chars) for Gemini contents dicts.

    Counts part text, functionCall args, and functionResponse results -- where
    accumulated file reads pile up. Mirrors the OpenAI-loop estimator.
    """
    chars = 0
    for entry in contents:
        for part in entry.get("parts") or ():
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                chars += len(text)
            call = part.get("functionCall")
            if isinstance(call, dict):
                chars += len(json.dumps(call.get("args") or {}, default=str))
            response = part.get("functionResponse")
            if isinstance(response, dict):
                result = response.get("response", {}).get("result")
                if isinstance(result, str):
                    chars += len(result)
    return chars // _CHARS_PER_TOKEN


def _compact_gemini_tool_results(
    contents: list[dict], budget: int, keep_recent: int,
) -> list[dict]:
    """Elide old, large functionResponse results to a stub within *budget*.

    Only the inner result string is replaced; the functionResponse part and its
    ``name`` stay, so the model still sees which call each response answers.
    Protects the first turn and the most recent *keep_recent* entries. Pure:
    returns new dicts on eviction; the input list is never mutated.
    """
    if _estimate_gemini_tokens(contents) <= budget:
        return contents
    protected: set[int] = set()
    for idx, entry in enumerate(contents):
        if entry.get("role") == "user":
            protected.add(idx)
            break
    if keep_recent > 0:
        protected.update(range(max(0, len(contents) - keep_recent), len(contents)))

    result = [dict(entry) for entry in contents]
    for idx, entry in enumerate(result):
        if _estimate_gemini_tokens(result) <= budget:
            break
        if idx in protected or not isinstance(entry.get("parts"), list):
            continue
        new_parts: list = []
        changed = False
        for part in entry["parts"]:
            response = part.get("functionResponse") if isinstance(part, dict) else None
            inner = (
                response.get("response", {}).get("result")
                if isinstance(response, dict)
                else None
            )
            if isinstance(inner, str):
                stub = f"[elided to fit context: tool result, {len(inner)} chars]"
                if len(stub) < len(inner):
                    merged = {
                        **response,
                        "response": {**response.get("response", {}), "result": stub},
                    }
                    new_parts.append({**part, "functionResponse": merged})
                    changed = True
                    continue
            new_parts.append(part)
        if changed:
            result[idx] = {**entry, "parts": new_parts}
    return result


def _elided_read_indices(before: list[dict], after: list[dict]) -> set[int]:
    """Contents indices whose functionResponse results compaction just elided.

    Compaction only stubs functionResponse result strings, so an entry whose
    ``parts`` changed had a read result evicted. Any read-dedup entry pointing
    at that index must be dropped, else a repeat read is told "[already read
    above]" while the bytes it named are gone.
    """
    return {
        idx
        for idx, (old, new) in enumerate(zip(before, after))
        if old.get("parts") != new.get("parts")
    }


@register_provider("gemini")
class GeminiProvider(BaseProvider):
    """Google Gemini API provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta/models"
        self.client = httpx.Client(timeout=60.0)
        # OAuth tokens (from Gemini CLI) start with "ya29." and use Bearer auth
        # API keys use ?key= query param
        self._use_bearer = config.api_key.startswith("ya29.")
        self._use_oauth_cli = self._use_bearer
        self._gemini_bin = shutil.which("gemini")
        self._use_cli_proxy = self._use_oauth_cli and bool(self._gemini_bin)
        default_activity = "1" if self._use_cli_proxy else "0"
        self._emit_activity = (
            os.getenv("CASCADE_GEMINI_ACTIVITY", default_activity).lower()
            not in ("0", "false", "no", "off")
        )
        self._cli_help_text: Optional[str] = None
        self._approval_mode_supported: Optional[bool] = None
        self._skip_trust_supported: Optional[bool] = None

    def _cli_help(self) -> str:
        """Load Gemini CLI capability text once per provider instance."""
        if self._cli_help_text is not None:
            return self._cli_help_text
        help_text = "--approval-mode --skip-trust"
        if self._gemini_bin:
            try:
                import subprocess

                out = subprocess.run(
                    [self._gemini_bin, "--help"],
                    capture_output=True, text=True, timeout=5,
                )
                help_text = out.stdout + out.stderr
            except Exception:
                pass
        self._cli_help_text = help_text
        return help_text

    def _supports_approval_mode(self) -> bool:
        """Whether this gemini CLI advertises --approval-mode (version-gated).

        Cached per instance. Probes `gemini --help`; if the probe fails we
        assume support (the flag is standard on current builds) rather than
        silently dropping the posture mapping.
        """
        if self._approval_mode_supported is not None:
            return self._approval_mode_supported
        supported = "--approval-mode" in self._cli_help()
        self._approval_mode_supported = supported
        return supported

    def _supports_skip_trust(self) -> bool:
        """Whether this Gemini build can suppress workspace-trust UI."""
        if self._skip_trust_supported is not None:
            return self._skip_trust_supported
        supported = "--skip-trust" in self._cli_help()
        self._skip_trust_supported = supported
        return supported

    def get_fallback_model(self) -> Optional[str]:
        """Fall back from Gemini Pro to Flash on rate limits."""
        if "pro" in self.config.model:
            return self.config.model.replace("pro", "flash")
        return None

    def _auth_params(self) -> tuple[dict, dict]:
        """Return (headers, params) for authentication."""
        headers = {"Content-Type": "application/json"}
        if self._use_bearer:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
            return headers, {}
        return headers, {"key": self.config.api_key}

    def _messages_to_contents(
        self, messages: list[Message], system: Optional[str] = None,
    ) -> list[dict]:
        """Convert provider messages to Gemini contents format."""
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        for msg in messages:
            gemini_role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})
        return contents

    def _stream_via_cli(
        self,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream assistant text by proxying through ``gemini -p``."""
        if not self._gemini_bin:
            yield "Error: gemini CLI not found in PATH for OAuth mode."
            return

        full_prompt = self._condense_for_cli(messages)
        workdir = self.get_working_directory()
        if system:
            condensed = self._condense_system_for_cli(system)
            if condensed:
                full_prompt = f"System instructions:\n{condensed}\n\n{full_prompt}"

        # `-p` must never fall back to an approval dialog. Gemini does not
        # expose a Claude-style model classifier, so auto uses its sandboxed
        # yolo lane; safe auto-approves edits but denies any remaining
        # non-interactive confirmation; readonly is plan mode.
        posture = getattr(
            getattr(self, "permission_engine", None), "posture", "auto",
        )
        approval = {
            "auto": "yolo",
            "yolo": "yolo",
            "safe": "auto_edit",
            "readonly": "plan",
        }.get(posture, "yolo")
        cmd = [
            self._gemini_bin, "-p", full_prompt,
            "--output-format", "stream-json",
        ]
        # --approval-mode is version-gated; only pass it when this gemini
        # build advertises it, so an older CLI is not broken outright.
        if self._supports_approval_mode():
            cmd.extend(["--approval-mode", approval])
        if posture == "auto":
            # Current Gemini also enables a sandbox implicitly for yolo, but
            # keep Cascade's auto boundary explicit and stable across versions.
            cmd.append("--sandbox")
        if self._supports_skip_trust():
            cmd.append("--skip-trust")
        cmd.extend(["--include-directories", workdir])
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        handler = GeminiEventHandler()
        cfg = CLIProxyConfig(
            binary=self._gemini_bin,
            cli_name="gemini",
            cmd_args=cmd,
            cwd=workdir,
        )
        cancel_token = self.cancellation_token()
        if cancel_token is None:
            yield from stream_cli_proxy(cfg, handler, self._emit_activity)
        else:
            yield from stream_cli_proxy(
                cfg, handler, self._emit_activity, cancel_token,
            )
        if handler.last_usage:
            self._last_usage = handler.last_usage
            self._last_round_usage = handler.last_usage

    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        """Get a complete response from Gemini."""
        return "".join(self.stream(messages, system))

    def stream(self, messages: list[Message], system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from Gemini."""
        self._last_usage = None
        self._last_round_usage = None
        self.reset_activity_state()
        if self._use_cli_proxy:
            yield from self._filter_activity(self._stream_via_cli(messages, system))
            return

        try:
            url = f"{self.base_url}/{self.config.model}:streamGenerateContent"
            headers, params = self._auth_params()

            contents = self._messages_to_contents(messages, system)

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens or 2048,
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            }

            with self.client.stream(
                "POST", url, json=payload, params=params, headers=headers
            ) as response, self.cancellation_callback(
                getattr(response, "close", lambda: None)
            ):
                response.raise_for_status()
                for line in response.iter_lines():
                    self.raise_if_cancelled()
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "candidates" in data:
                                for candidate in data["candidates"]:
                                    if "content" in candidate:
                                        for part in candidate["content"].get("parts", []):
                                            if "text" in part:
                                                yield part["text"]
                            usage = data.get("usageMetadata", {})
                            if usage:
                                parsed = Usage.from_gemini(usage)
                                if parsed.total:
                                    self._last_usage = parsed
                                    self._last_round_usage = parsed
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(str(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(str(exc)) from exc

    def ask_with_tools(
        self,
        messages: list[Message],
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
        on_pending_message=None,  # accepted for parity; dispatch-on-completion.
    ) -> tuple[str, list[dict]]:
        """Gemini-native tool calling using function_declarations."""
        if self._use_cli_proxy:
            return self.ask(messages, system), []
        self._last_usage = None
        self._last_round_usage = None

        from ..tools.executor import ConcurrentToolExecutor
        from ..tools.permissions import (
            PermissionAbort,
            permission_context_from_messages,
        )

        executor = ConcurrentToolExecutor(
            tools,
            hook_runner=self.hook_runner,
            permissions=self.permission_engine,
            permission_context=permission_context_from_messages(
                messages,
                provider="gemini",
                model=self.config.model,
            ),
        )

        # Build Gemini function declarations
        function_declarations = []
        for td in tools.values():
            decl = {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            }
            function_declarations.append(decl)

        contents = self._messages_to_contents(messages, system)

        tool_log = []
        headers, params = self._auth_params()
        # Bound accumulated tool results and skip re-reading a known path, as the
        # shared OpenAI loop does -- otherwise every file read is re-sent in full
        # on every round.
        budget = int(
            window_for("gemini", self.config.model, self.config.context_window)
            * _CONTEXT_BUDGET_FRACTION
        )
        # dedup_key -> the contents index of the entry holding that read's
        # functionResponse. When budget compaction later ELIDES that result, the
        # key is dropped so a repeat read re-fetches the file instead of
        # "[already read above]" pointing at content that is gone. Mirrors the
        # shared OpenAI loop (which keys by tool_call_id).
        seen_reads: dict[tuple[str, str], int] = {}

        text_parts = []
        for round_num in range(max_rounds):
            self.raise_if_cancelled()
            _before = contents
            contents = _compact_gemini_tool_results(
                contents, budget, _KEEP_RECENT_MESSAGES,
            )
            # Drop dedup keys whose read result was just elided.
            if seen_reads and _before is not contents:
                elided = _elided_read_indices(_before, contents)
                if elided:
                    seen_reads = {
                        key: idx
                        for key, idx in seen_reads.items()
                        if idx not in elided
                    }
            url = f"{self.base_url}/{self.config.model}:generateContent"
            payload = {
                "contents": contents,
                "tools": [{"function_declarations": function_declarations}],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens or 2048,
                },
            }

            try:
                response = self.client.post(url, json=payload, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(str(exc)) from exc
            except httpx.RequestError as exc:
                raise RuntimeError(str(exc)) from exc
            self.raise_if_cancelled()

            round_meta = data.get("usageMetadata", {})
            if isinstance(round_meta, dict) and round_meta:
                round_usage = Usage.from_gemini(round_meta)
                if round_usage.total:
                    prev = self._last_usage or Usage()
                    self._last_usage = prev.add(round_usage)
                    self._last_round_usage = round_usage

            # Parse response parts
            candidates = data.get("candidates", [])
            if not candidates:
                return "", tool_log

            parts = candidates[0].get("content", {}).get("parts", [])

            text_parts = []
            function_calls = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    function_calls.append(part["functionCall"])

            if not function_calls:
                return "".join(text_parts), tool_log

            # Append the model response
            contents.append({"role": "model", "parts": parts})

            # This round's functionResponses land in one entry appended below;
            # its index is the current length. Reads dedup against it so an
            # elision of that entry can later invalidate them.
            response_index = len(contents)

            batch_calls = [
                (fc["name"], fc.get("args", {}))
                for fc in function_calls
            ]
            batch_keys = [
                _read_dedup_key(tool_name, tool_args)
                for tool_name, tool_args in batch_calls
            ]
            concrete_keys = [key for key in batch_keys if key is not None]
            can_batch = (
                len(batch_calls) > 1
                and all(
                    (tool := executor.get_tool(tool_name)) is not None
                    and tool.concurrency_safe
                    for tool_name, _tool_args in batch_calls
                )
                and not any(key in seen_reads for key in concrete_keys)
                and len(concrete_keys) == len(set(concrete_keys))
            )
            if can_batch:
                for tool_name, tool_args in batch_calls:
                    if on_tool_event:
                        on_tool_event(ToolEvent(
                            kind="tool_start",
                            tool_name=tool_name,
                            round_num=round_num,
                            max_rounds=max_rounds,
                            tool_input=tool_args,
                        ))
                try:
                    batch_results = executor.execute_batch(batch_calls)
                except PermissionAbort as exc:
                    return (
                        "".join(text_parts) + f"\n\n[stopped: {exc}]"
                    ).strip(), tool_log

                response_parts = []
                for fc, tool_args, dedup_key, result in zip(
                    function_calls,
                    (call[1] for call in batch_calls),
                    batch_keys,
                    batch_results,
                ):
                    self.raise_if_cancelled()
                    tool_name = fc["name"]
                    if dedup_key is not None:
                        seen_reads[dedup_key] = response_index
                    else:
                        edited = _invalidates_read(tool_name, tool_args)
                        if edited is not None and seen_reads:
                            seen_reads = {
                                key: idx
                                for key, idx in seen_reads.items()
                                if key[1] != edited
                            }
                    tool_log.append({
                        "tool": tool_name,
                        "input": tool_args,
                        "output": result,
                    })
                    if on_tool_event:
                        on_tool_event(ToolEvent(
                            kind="tool_done",
                            tool_name=tool_name,
                            round_num=round_num,
                            max_rounds=max_rounds,
                            tool_input=tool_args,
                            tool_output=result,
                        ))
                    response_parts.append({
                        "functionResponse": {
                            "name": tool_name,
                            "response": {"result": result},
                        }
                    })
                contents.append({"role": "user", "parts": response_parts})
                continue

            # Execute each function call
            response_parts = []
            for fc in function_calls:
                self.raise_if_cancelled()
                tool_name = fc["name"]
                tool_args = fc.get("args", {})

                if on_tool_event:
                    on_tool_event(ToolEvent(
                        kind="tool_start",
                        tool_name=tool_name,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        tool_input=tool_args,
                    ))

                dedup_key = _read_dedup_key(tool_name, tool_args)
                if dedup_key is not None and dedup_key in seen_reads:
                    result = f"[already read above: {dedup_key[1]}]"
                else:
                    try:
                        result = executor.execute(tool_name, tool_args)
                    except PermissionAbort as exc:
                        return ("".join(text_parts) + f"\n\n[stopped: {exc}]").strip(), tool_log
                    if dedup_key is not None:
                        seen_reads[dedup_key] = response_index
                    else:
                        # A successful edit makes any cached read of that path
                        # stale: drop its dedup keys so the next read re-fetches.
                        edited = _invalidates_read(tool_name, tool_args)
                        if edited is not None and seen_reads:
                            seen_reads = {
                                key: idx
                                for key, idx in seen_reads.items()
                                if key[1] != edited
                            }
                self.raise_if_cancelled()
                tool_log.append({
                    "tool": tool_name,
                    "input": tool_args,
                    "output": result,
                })

                if on_tool_event:
                    on_tool_event(ToolEvent(
                        kind="tool_done",
                        tool_name=tool_name,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        tool_input=tool_args,
                        tool_output=result,
                    ))

                response_parts.append({
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"result": result},
                    }
                })

            contents.append({"role": "user", "parts": response_parts})

        return "".join(text_parts) if text_parts else "", tool_log

    def compare(self, prompt: str, system: Optional[str] = None) -> dict:
        """Generate comparison data."""
        response = self.ask_single(prompt, system)
        return {
            "provider": self.name,
            "model": self.config.model,
            "response": response,
            "length": len(response),
        }

    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass
