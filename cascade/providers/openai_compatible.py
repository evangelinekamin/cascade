"""Keyless OpenAI-compatible provider for self-hosted / remote endpoints.

One generic provider for any OpenAI-compatible ``/v1`` endpoint that needs no
API key. It is registered under two names:

- ``local`` -- a model running on this machine (e.g. Qwen via llama-swap on
  ``127.0.0.1:8080``). Fast, no network hop.
- ``glm``   -- a stronger model on a remote box (e.g. GLM over an overlay
  network). The escalation tier.

Endpoint and model come from config; the class defaults target the local case.
Because it reuses the shared OpenAI tool loop, either name works as a Cascade
verified-build agent (``/solve``, ``/pipeline``) out of the box. Escalating
``local -> glm`` across the two endpoints is a routing concern handled a layer
up, not a per-request toggle here.
"""

import json
from typing import Optional, Iterator, TYPE_CHECKING

import httpx

from .base import BaseProvider, ProviderConfig, Message, ToolEventCallback
from .registry import register_provider
from ._openai_tools import openai_ask_with_tools

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


@register_provider("local")
@register_provider("glm")
class OpenAICompatibleProvider(BaseProvider):
    """Keyless OpenAI-compatible provider (self-hosted local / remote endpoints)."""

    _DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
    _DEFAULT_MODEL = "qwen36"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or self._DEFAULT_BASE_URL
        # Self-hosted models may cold-load on first request -- allow headroom.
        self.client = httpx.Client(timeout=120.0)

    @property
    def model(self) -> str:
        """Live model id.

        Read from config on every access (not cached) so a model swap that
        mutates ``config.model`` in place takes effect immediately.
        """
        return self.config.model or self._DEFAULT_MODEL

    def _headers(self) -> dict:
        """Build request headers, sending auth only when a key is configured."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def validate(self) -> bool:
        """Keyless endpoint: a model is all we require (base_url is defaulted)."""
        return bool(self.model)

    @staticmethod
    def _build_api_messages(
        messages: list[Message],
        system: Optional[str] = None,
    ) -> list[dict]:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(
            {"role": m["role"], "content": m["content"]}
            for m in messages
        )
        return api_messages

    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        """Get a complete response from the endpoint."""
        return "".join(self.stream(messages, system))

    def stream(self, messages: list[Message], system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from the OpenAI-compatible endpoint."""
        self._last_usage = None
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": self._build_api_messages(messages, system),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens

        try:
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
                    except json.JSONDecodeError:
                        continue

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
    ) -> tuple[str, list[dict]]:
        """OpenAI-compatible tool calling via the shared loop."""
        self._last_usage = None
        return openai_ask_with_tools(
            client=self.client,
            url=f"{self.base_url}/chat/completions",
            headers=self._headers(),
            model=self.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=messages,
            tools=tools,
            system=system,
            max_rounds=max_rounds,
            on_tool_event=on_tool_event,
            on_usage=lambda usage: setattr(self, "_last_usage", usage),
        )

    def compare(self, prompt: str, system: Optional[str] = None) -> dict:
        """Generate comparison data."""
        response = self.ask_single(prompt, system)
        return {
            "provider": self.name,
            "model": self.model,
            "response": response,
            "length": len(response),
        }

    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass
