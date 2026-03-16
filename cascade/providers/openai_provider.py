"""OpenAI provider for GPT-4o, o1, o3, and Codex models."""

import json
import os
import shutil
from typing import Optional, Iterator, TYPE_CHECKING
import httpx
from .base import BaseProvider, ProviderConfig, Message, ToolEventCallback
from ._cli_proxy import CLIProxyConfig, CodexEventHandler, stream_cli_proxy
from ._openai_tools import openai_ask_with_tools
from .registry import register_provider

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


@register_provider("openai")
class OpenAIProvider(BaseProvider):
    """OpenAI API provider - supports custom base_url for Azure/proxies."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.client = httpx.Client(timeout=60.0)
        self._codex_bin = shutil.which("codex")
        self._use_oauth_cli = self._looks_like_jwt(config.api_key)
        self._use_cli_proxy = (
            self._use_oauth_cli
            and bool(self._codex_bin)
            and self.base_url.rstrip("/") == "https://api.openai.com/v1"
        )
        default_activity = "1" if self._use_cli_proxy else "0"
        self._emit_activity = (
            os.getenv(
                "CASCADE_OPENAI_ACTIVITY",
                os.getenv("CASCADE_CODEX_ACTIVITY", default_activity),
            ).lower()
            not in ("0", "false", "no", "off")
        )

    @staticmethod
    def _looks_like_jwt(token: str) -> bool:
        token = (token or "").strip()
        return token.startswith("eyJ") and token.count(".") >= 2

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _stream_via_cli(
        self,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream assistant text by proxying through ``codex exec --json``."""
        if not self._codex_bin:
            yield "Error: codex CLI not found in PATH for OAuth mode."
            return

        full_prompt = self._condense_for_cli(messages)
        if system:
            condensed = self._condense_system_for_cli(system)
            if condensed:
                full_prompt = f"System instructions:\n{condensed}\n\n{full_prompt}"

        cmd = [self._codex_bin, "exec", "--json", "--cd", os.getcwd()]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        cmd.append(full_prompt)

        handler = CodexEventHandler()
        cfg = CLIProxyConfig(binary=self._codex_bin, cli_name="codex", cmd_args=cmd)
        yield from stream_cli_proxy(cfg, handler, self._emit_activity)
        if handler.last_usage:
            self._last_usage = handler.last_usage

    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        """Get a complete response from OpenAI."""
        return "".join(self.stream(messages, system))

    def stream(self, messages: list[Message], system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from OpenAI."""
        self._last_usage = None
        self._last_activity = None
        if self._use_cli_proxy:
            yield from self._filter_activity(self._stream_via_cli(messages, system))
            return
        if self._use_oauth_cli and not self._use_cli_proxy:
            if not self._codex_bin:
                yield "Error: Codex OAuth token detected, but codex CLI is not in PATH."
            else:
                yield (
                    "Error: Codex OAuth token requires the default OpenAI base URL "
                    "(https://api.openai.com/v1)."
                )
            return

        try:
            url = f"{self.base_url}/chat/completions"

            api_messages = []
            if system:
                api_messages.append({"role": "system", "content": system})
            api_messages.extend(
                {"role": m["role"], "content": m["content"]}
                for m in messages
            )

            payload = {
                "model": self.config.model,
                "messages": api_messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens:
                payload["max_tokens"] = self.config.max_tokens

            with self.client.stream("POST", url, json=payload, headers=self._headers()) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        usage = data.get("usage")
                        if usage:
                            self._last_usage = (
                                usage.get("prompt_tokens", 0),
                                usage.get("completion_tokens", 0),
                            )
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
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
        """OpenAI-native tool calling."""
        if self._use_cli_proxy:
            return self.ask(messages, system), []

        return openai_ask_with_tools(
            client=self.client,
            url=f"{self.base_url}/chat/completions",
            headers=self._headers(),
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=messages,
            tools=tools,
            system=system,
            max_rounds=max_rounds,
            on_tool_event=on_tool_event,
        )

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
