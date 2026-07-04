"""OpenRouter provider for multi-model access (Qwen, Llama, etc)."""

import json
from typing import Optional, Iterator, TYPE_CHECKING
import httpx
from .base import BaseProvider, ProviderConfig, Message, ToolEventCallback
from .registry import register_provider
from ._openai_tools import openai_ask_with_tools

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


@register_provider("openrouter")
class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API - OpenAI-compatible endpoint."""

    _APP_URL = "https://github.com/Evangeline-Development-Company/cascade"
    _APP_TITLE = "Cascade"
    _RETRYABLE_FALLBACK_STATUS_CODES = frozenset((502, 503))
    _DEFAULT_FALLBACK_MODEL = "minimax/minimax-m2.5"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://openrouter.ai/api/v1"
        self.client = httpx.Client(timeout=60.0)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._APP_URL,
            "X-OpenRouter-Title": self._APP_TITLE,
        }

    def get_fallback_model(self) -> Optional[str]:
        """Return the configured OpenRouter fallback model."""
        fallback = self.config.fallback_model or self._DEFAULT_FALLBACK_MODEL
        if not fallback or fallback == self.config.model:
            return None
        return fallback

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

    def _stream_with_model(
        self,
        model: str,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": self._build_api_messages(messages, system),
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

    @classmethod
    def _should_try_fallback(cls, exc: httpx.HTTPStatusError) -> bool:
        return exc.response.status_code in cls._RETRYABLE_FALLBACK_STATUS_CODES

    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        """Get a complete response from OpenRouter."""
        return "".join(self.stream(messages, system))

    def stream(self, messages: list[Message], system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from OpenRouter."""
        self._last_usage = None
        try:
            yield from self._stream_with_model(self.config.model, messages, system)
        except httpx.HTTPStatusError as exc:
            fallback_model = self.get_fallback_model()
            if self._should_try_fallback(exc) and fallback_model:
                try:
                    yield from self._stream_with_model(fallback_model, messages, system)
                    return
                except httpx.HTTPStatusError as fallback_exc:
                    raise RuntimeError(str(fallback_exc)) from fallback_exc
                except httpx.RequestError as fallback_exc:
                    raise RuntimeError(str(fallback_exc)) from fallback_exc
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
        """OpenAI-compatible tool calling via OpenRouter."""
        self._last_usage = None
        try:
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
                on_usage=lambda usage: setattr(self, "_last_usage", usage),
                context_window=self.config.context_window,
            )
        except RuntimeError as exc:
            fallback_model = self.get_fallback_model()
            cause = exc.__cause__
            if (
                fallback_model
                and isinstance(cause, httpx.HTTPStatusError)
                and self._should_try_fallback(cause)
            ):
                return openai_ask_with_tools(
                    client=self.client,
                    url=f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    model=fallback_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    messages=messages,
                    tools=tools,
                    system=system,
                    max_rounds=max_rounds,
                    on_tool_event=on_tool_event,
                    on_usage=lambda usage: setattr(self, "_last_usage", usage),
                    context_window=self.config.context_window,
                )
            raise

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
