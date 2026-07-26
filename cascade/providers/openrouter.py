"""OpenRouter provider for multi-model access (Qwen, Llama, etc)."""

import hashlib
import json
from typing import Optional, Iterator, TYPE_CHECKING
import httpx
from .base import BaseProvider, ProviderConfig, Message, ToolEventCallback
from .registry import register_provider
from ._openai_tools import openai_ask_with_tools, _http_error_text
from .usage import Usage
from ..context.budget import window_for

if TYPE_CHECKING:
    from ..tools.schema import ToolDef

# The live endpoint-metadata fetch is disabled wholesale by the test suite (which
# has no network and asserts on unmutated config); production leaves it on.
_META_ENABLED = True


@register_provider("openrouter")
class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API - OpenAI-compatible endpoint."""

    _APP_URL = "https://github.com/Evangeline-Development-Company/cascade"
    _APP_TITLE = "Cascade"
    _RETRYABLE_FALLBACK_STATUS_CODES = frozenset((429, 502, 503))
    _DEFAULT_FALLBACK_MODEL = "minimax/minimax-m2.5"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://openrouter.ai/api/v1"
        self.client = httpx.Client(timeout=60.0)
        self._last_generation_id: Optional[str] = None
        self._meta_applied = False

    def _apply_meta(self) -> None:
        """Once per model, fold live OpenRouter endpoint metadata into the config.

        Best-effort and lazy (first request): sets the real context window (the
        conservative min across routable endpoints) when none is configured, and
        prefers unquantized, fast, cheap endpoints when no order is pinned. Any
        failure leaves the configured defaults untouched.
        """
        if self._meta_applied or not _META_ENABLED:
            return
        self._meta_applied = True
        try:
            from .openrouter_meta import model_meta

            meta = model_meta(self.config.model, base_url=self.base_url)
        except Exception:
            meta = None
        if meta is None:
            return
        if self.config.context_window is None and meta.effective_context:
            self.config.context_window = meta.effective_context
        prefs = dict(self.config.provider_preferences or {})
        prefs.setdefault("require_parameters", True)
        prefs.setdefault("allow_fallbacks", True)
        # Respect an explicitly pinned order; otherwise route by the quality rank
        # (unquantized > throughput > price).
        if not prefs.get("order") and meta.recommended_order:
            prefs["order"] = list(meta.recommended_order[:6])
        self.config.provider_preferences = prefs

    @property
    def last_generation_id(self) -> Optional[str]:
        """OpenRouter generation id for correlation and provider debugging."""
        return self._last_generation_id

    @property
    def last_cost(self) -> Optional[float]:
        """Reported request cost when OpenRouter includes it in usage."""
        return self._last_usage.cost if self._last_usage else None

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
    def _session_key(system: Optional[str], messages: list[Message]) -> Optional[str]:
        """A stable per-conversation id so OpenRouter pins the whole conversation
        to one upstream host and keeps its prompt cache warm (sticky sessions).

        Derived from the conversation's stable prefix -- the system prompt and the
        first message, which do not change across the turns of one conversation --
        so every follow-up request carries the same id and lands on the same
        cache. Hashed, so no prompt content is exposed and it always fits the
        256-char cap. Returns None when there is nothing to key on.
        """
        first = messages[0]["content"] if messages else ""
        if not (system or first):
            return None
        basis = f"{system or ''}\x00{first}".encode("utf-8", "ignore")
        return "cascade-" + hashlib.sha256(basis).hexdigest()[:32]

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
        if self.config.provider_preferences:
            payload["provider"] = self.config.provider_preferences
        session_id = self._session_key(system, messages)
        if session_id:
            payload["session_id"] = session_id

        with self.client.stream(
            "POST", url, json=payload, headers=self._headers()
        ) as response, self.cancellation_callback(
            getattr(response, "close", lambda: None)
        ):
            # On an error status the body carries OpenRouter's real reason; load
            # it before raising so it is not lost with the unread stream.
            if getattr(response, "status_code", 200) >= 400:
                try:
                    response.read()
                except Exception:
                    pass
            response.raise_for_status()
            self._last_generation_id = getattr(response, "headers", {}).get(
                "X-Generation-Id"
            )
            for line in response.iter_lines():
                self.raise_if_cancelled()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                error = data.get("error")
                if error:
                    metadata = error.get("metadata", {}) if isinstance(error, dict) else {}
                    error_type = metadata.get("error_type") if isinstance(metadata, dict) else None
                    message = (
                        error.get("message", "OpenRouter stream failed")
                        if isinstance(error, dict)
                        else str(error)
                    )
                    suffix = f" ({error_type})" if error_type else ""
                    # A mid-stream failure cannot be retried safely: partial output
                    # may already have been shown. Raise so it is never recorded as
                    # a successful, merely-short completion.
                    raise RuntimeError(f"OpenRouter stream error{suffix}: {message}")

                usage = data.get("usage")
                if usage:
                    self._last_usage = Usage.from_openai(usage)
                    self._last_round_usage = self._last_usage
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
        self._apply_meta()
        self._last_usage = None
        self._last_round_usage = None
        self._last_generation_id = None
        try:
            yield from self._stream_with_model(self.config.model, messages, system)
        except httpx.HTTPStatusError as exc:
            fallback_model = self.get_fallback_model()
            if self._should_try_fallback(exc) and fallback_model:
                try:
                    yield from self._stream_with_model(fallback_model, messages, system)
                    return
                except httpx.HTTPStatusError as fallback_exc:
                    raise RuntimeError(_http_error_text(fallback_exc)) from fallback_exc
                except httpx.RequestError as fallback_exc:
                    raise RuntimeError(str(fallback_exc)) from fallback_exc
            raise RuntimeError(_http_error_text(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(str(exc)) from exc

    def ask_structured(
        self,
        prompt: str,
        schema: dict,
        *,
        system: Optional[str] = None,
        schema_name: str = "cascade_response",
    ) -> dict:
        """Return a strict JSON-schema response using OpenRouter's native field.

        This is intentionally non-streaming: it is used for small control-plane
        decisions where a single validated object is more useful than incremental
        text. Provider preferences (including ``require_parameters``) are preserved.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": self._build_api_messages(
                [{"role": "user", "content": prompt}], system
            ),
            "stream": False,
            "temperature": self.config.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self.config.max_tokens:
            payload["max_tokens"] = self.config.max_tokens
        if self.config.provider_preferences:
            payload["provider"] = self.config.provider_preferences

        self._last_usage = None
        self._last_round_usage = None
        self._last_generation_id = None
        try:
            response = self.client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(_http_error_text(exc)) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(str(exc)) from exc

        self._last_generation_id = getattr(response, "headers", {}).get(
            "X-Generation-Id"
        )
        data = response.json()
        if data.get("error"):
            raise RuntimeError(f"OpenRouter structured response failed: {data['error']}")
        usage = data.get("usage") or {}
        if usage:
            self._last_usage = Usage.from_openai(usage)
            self._last_round_usage = self._last_usage

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter structured response contained no choices")
        choice = choices[0]
        if choice.get("finish_reason") == "error" or choice.get("error"):
            raise RuntimeError(f"OpenRouter structured response failed: {choice.get('error')}")
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, dict):
            return content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenRouter structured response contained no JSON content")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenRouter returned invalid structured JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenRouter structured response was not an object")
        return parsed

    def ask_with_tools(
        self,
        messages: list[Message],
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
        on_pending_message=None,
    ) -> tuple[str, list[dict]]:
        """OpenAI-compatible tool calling via OpenRouter."""
        self._apply_meta()
        self._last_usage = None
        self._last_round_usage = None
        session_id = self._session_key(system, messages)
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
                on_pending_message=on_pending_message,
                on_usage=lambda usage: setattr(self, "_last_usage", usage),
                on_round_usage=lambda usage: setattr(self, "_last_round_usage", usage),
                hook_runner=self.hook_runner,
                permissions=self.permission_engine,
                context_window=window_for(
                    "openrouter", self.config.model, self.config.context_window,
                ),
                provider_preferences=self.config.provider_preferences,
                session_id=session_id,
                check_cancelled=self.raise_if_cancelled,
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
                    on_pending_message=on_pending_message,
                    on_usage=lambda usage: setattr(self, "_last_usage", usage),
                on_round_usage=lambda usage: setattr(self, "_last_round_usage", usage),
                hook_runner=self.hook_runner,
                permissions=self.permission_engine,
                    context_window=window_for(
                    "openrouter", self.config.model, self.config.context_window,
                ),
                    provider_preferences=self.config.provider_preferences,
                    session_id=session_id,
                    check_cancelled=self.raise_if_cancelled,
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
