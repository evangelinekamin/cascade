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
from .registry import register_provider

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


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
        if system:
            condensed = self._condense_system_for_cli(system)
            if condensed:
                full_prompt = f"System instructions:\n{condensed}\n\n{full_prompt}"

        cmd = [
            self._gemini_bin, "-p", full_prompt,
            "--output-format", "stream-json",
            "--include-directories", os.getcwd(),
        ]
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        handler = GeminiEventHandler()
        cfg = CLIProxyConfig(binary=self._gemini_bin, cli_name="gemini", cmd_args=cmd)
        yield from stream_cli_proxy(cfg, handler, self._emit_activity)
        if handler.last_usage:
            self._last_usage = handler.last_usage

    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        """Get a complete response from Gemini."""
        return "".join(self.stream(messages, system))

    def stream(self, messages: list[Message], system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from Gemini."""
        self._last_usage = None
        self._last_activity = None
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

            with self.client.stream("POST", url, json=payload, params=params, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
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
                            in_t = usage.get("promptTokenCount", 0)
                            out_t = usage.get("candidatesTokenCount", 0)
                            if in_t or out_t:
                                self._last_usage = (in_t, out_t)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"

    def ask_with_tools(
        self,
        messages: list[Message],
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
    ) -> tuple[str, list[dict]]:
        """Gemini-native tool calling using function_declarations."""
        if self._use_cli_proxy:
            return self.ask(messages, system), []

        from ..tools.executor import ToolExecutor

        executor = ToolExecutor(tools)

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

        text_parts = []
        for round_num in range(max_rounds):
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
            except Exception as e:
                return f"Error: {e}", tool_log

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

            # Execute each function call
            response_parts = []
            for fc in function_calls:
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

                result = executor.execute(tool_name, tool_args)
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
