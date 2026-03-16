"""Base provider interface for all AI models."""

from abc import ABC, abstractmethod
from typing import Optional, Iterator, Callable, TypedDict, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


class Message(TypedDict, total=False):
    """A single conversation message passed to providers."""

    role: str       # "user" | "assistant" | "system"
    content: str
    provider: str   # which provider generated this (for cross-model context)


@dataclass(frozen=True)
class ToolEvent:
    """Progress event emitted during tool-calling rounds."""

    kind: str  # "tool_start" | "tool_done"
    tool_name: str
    round_num: int
    max_rounds: int
    tool_input: dict = field(default_factory=dict)
    tool_output: str = ""


ToolEventCallback = Optional[Callable[[ToolEvent], None]]


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class BaseProvider(ABC):
    """Abstract base class for all AI providers."""

    _ACTIVITY_PREFIX = "[[cascade_activity]] "

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = self.__class__.__name__
        self._last_usage: Optional[tuple[int, int]] = None
        self._last_activity: Optional[str] = None
        self._emit_activity: bool = False

    @property
    def last_usage(self) -> Optional[tuple[int, int]]:
        """Token usage from last ask/stream call: (input_tokens, output_tokens)."""
        return self._last_usage

    @property
    def last_activity(self) -> Optional[str]:
        """Most recent activity status from CLI proxy, or None."""
        return self._last_activity

    def _activity(self, message: str) -> Optional[str]:
        """Encode a status line into a chunk the TUI can detect."""
        if not self._emit_activity:
            return None
        return f"{self._ACTIVITY_PREFIX}{message}"

    def _filter_activity(self, chunks: Iterator[str]) -> Iterator[str]:
        """Strip activity prefix messages from stream, storing them for TUI access."""
        for chunk in chunks:
            if isinstance(chunk, str) and chunk.startswith(self._ACTIVITY_PREFIX):
                self._last_activity = chunk[len(self._ACTIVITY_PREFIX):].strip()
                continue
            yield chunk

    @staticmethod
    def _condense_system_for_cli(system: str) -> str:
        """Strip verbose boilerplate from system prompt for CLI proxy mode.

        CLI proxy passes system instructions inside the prompt text, so
        a multi-kilobyte system prompt overwhelms the actual user request.
        Keep identity and mode directive, drop quality gates / workflow /
        conventions that the underlying model already knows.
        """
        if not system:
            return ""
        lines = system.split("\n")
        kept: list[str] = []
        skip_sections = {
            "Quality Gates:", "Workflow:", "Tool Use:",
            "Conventions:", "Current date:",
        }
        skipping = False
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(s) for s in skip_sections):
                skipping = True
                continue
            # A new non-indented non-empty line after a skipped section
            # means we hit the next section — check if it's also skippable
            if skipping and stripped and not line.startswith(" ") and not line.startswith("-"):
                if not any(stripped.startswith(s) for s in skip_sections):
                    skipping = False
            if not skipping:
                kept.append(line)
        # Collapse runs of blank lines
        result: list[str] = []
        for line in kept:
            if line.strip() == "" and result and result[-1].strip() == "":
                continue
            result.append(line)
        return "\n".join(result).strip()

    def _condense_for_cli(self, messages: list[Message]) -> str:
        """Build a single prompt string from a messages list for CLI proxy mode.

        Extracts the last message as the current prompt and prepends
        condensed context from earlier messages.
        """
        if not messages:
            return ""
        context_lines = []
        for msg in messages[:-1]:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:500]
            context_lines.append(f"{role_label}: {content}")
        current_prompt = messages[-1]["content"]
        if context_lines:
            return (
                "Previous conversation context:\n"
                + "\n".join(context_lines[-6:])
                + "\n\nCurrent request:\n"
                + current_prompt
            )
        return current_prompt

    @abstractmethod
    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        """Send messages and get a complete response."""
        pass

    def ask_single(self, prompt: str, system: Optional[str] = None) -> str:
        """Convenience: single-prompt call. Wraps ask()."""
        return self.ask([{"role": "user", "content": prompt}], system)

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream tokens from the provider."""
        pass

    def stream_single(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Convenience: single-prompt call. Wraps stream()."""
        return self.stream([{"role": "user", "content": prompt}], system)

    @abstractmethod
    def compare(self, prompt: str, system: Optional[str] = None) -> dict:
        """Generate and return structured comparison data."""
        pass

    def ask_with_tools(
        self,
        messages: list[Message],
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
    ) -> tuple[str, list[dict]]:
        """Ask with tool calling support.

        Subclasses implement provider-native tool calling. The default
        falls back to a plain ask() with no tool support.

        Returns:
            Tuple of (final_text_response, tool_calls_log).
        """
        return self.ask(messages, system), []

    def get_fallback_model(self) -> Optional[str]:
        """Return a cheaper/faster model to fall back to on rate limits.

        Subclasses override to provide provider-specific fallback logic.
        Returns None when no fallback is available.
        """
        return None

    def validate(self) -> bool:
        """Validate provider configuration and connectivity."""
        return bool(self.config.api_key and self.config.model)

    def ping(self) -> bool:
        """Test connectivity with a minimal API call. Returns True on success."""
        try:
            result = self.ask_single("Reply with the single word OK.")
            return bool(result and len(result.strip()) > 0)
        except Exception:
            return False
