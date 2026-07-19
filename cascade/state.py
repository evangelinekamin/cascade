"""Reactive state for the Cascade TUI.

State mutations post Textual Messages so widgets can watch for changes.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, TYPE_CHECKING
import time

from .history.wordid import generate_word_id
from .episodes import Episode
from .providers.usage import Usage

from textual.message import Message

if TYPE_CHECKING:
    from textual.app import App


# ---------------------------------------------------------------------------
# Chat message (renamed to avoid collision with textual.message.Message)
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single chat message."""

    role: str           # "you" or provider name
    content: str
    tokens: int = 0
    msg_type: str = "text"  # text, code, write, edit, system
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Textual messages (events posted to the app message bus)
# ---------------------------------------------------------------------------

class ProviderChanged(Message):
    """The active provider changed."""

    def __init__(self, provider: str, mode: str) -> None:
        super().__init__()
        self.provider = provider
        self.mode = mode


class TokensUpdated(Message):
    """Token counts changed."""

    def __init__(self, provider: str, input_tokens: int, output_tokens: int) -> None:
        super().__init__()
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class ThinkingChanged(Message):
    """Provider started or stopped thinking."""

    def __init__(self, provider: str, thinking: bool, thought: str = "") -> None:
        super().__init__()
        self.provider = provider
        self.thinking = thinking
        self.thought = thought


class NewMessage(Message):
    """A new chat message was added."""

    def __init__(self, message: ChatMessage) -> None:
        super().__init__()
        self.message = message


class StreamChunk(Message):
    """A chunk of streamed response text arrived."""

    def __init__(self, provider: str, chunk: str, done: bool = False) -> None:
        super().__init__()
        self.provider = provider
        self.chunk = chunk
        self.done = done


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class CascadeState:
    """Global mutable state for the application.

    Call mutator methods (not bare attribute sets) so that Textual messages
    are posted to the app's message bus.
    """

    def __init__(self) -> None:
        self.session_id: str = generate_word_id()
        self.start_time: float = time.monotonic()
        self.active_provider: str = "gemini"
        self.mode: str = "design"
        self.messages: List[ChatMessage] = []
        self.total_tokens: int = 0
        self.provider_tokens: Dict[str, int] = self._default_provider_totals()
        self.cwd: str = "."
        self.branch: str = "main"
        self.episodes: List[Episode] = []
        # Context accounting: the last round's real Usage (the anchor) and
        # how often this session has compacted. anchor None = unknown (fresh
        # session or just compacted) -- the indicator shows "?".
        self.context_anchor: Optional[Usage] = None
        self.compaction_count: int = 0
        self.compaction_summary: str = ""
        self.is_thinking: bool = False
        self.current_thought: str = ""
        self.fast_mode: bool = False
        self._app: Optional["App"] = None

    @staticmethod
    def _default_provider_totals() -> Dict[str, int]:
        return {
            "gemini": 0,
            "claude": 0,
            "openai": 0,
            "openrouter": 0,
        }

    def bind(self, app: "App") -> None:
        """Bind to a Textual App so mutations can post messages."""
        self._app = app

    def _post(self, msg: Message) -> None:
        if self._app is not None:
            self._app.post_message(msg)

    # -- Mutators ----------------------------------------------------------

    def set_provider(self, provider: str, mode: str) -> None:
        if provider != self.active_provider:
            # Occupancy is per-provider: pricing one provider's usage
            # against another's window lies. Unknown until it responds.
            self.context_anchor = None
        self.active_provider = provider
        self.mode = mode
        self._post(ProviderChanged(provider, mode))

    def set_session_id(self, session_id: str) -> None:
        self.session_id = session_id

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Clear conversation-scoped state when starting or resuming a session."""
        if session_id is not None:
            self.session_id = session_id
        self.messages = []
        self.total_tokens = 0
        current_keys = tuple(self.provider_tokens.keys())
        self.provider_tokens = self._default_provider_totals()
        for key in current_keys:
            self.provider_tokens.setdefault(key, 0)
        self.episodes = []
        self.context_anchor = None
        self.compaction_count = 0
        self.compaction_summary = ""
        self.is_thinking = False
        self.current_thought = ""

    def add_message(self, role: str, content: str, tokens: int = 0,
                    msg_type: str = "text", metadata: Optional[Dict] = None) -> ChatMessage:
        msg = ChatMessage(
            role=role,
            content=content,
            tokens=tokens,
            msg_type=msg_type,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        self._post(NewMessage(msg))
        return msg

    def update_tokens(self, provider: str, input_tokens: int, output_tokens: int) -> None:
        total = input_tokens + output_tokens
        self.provider_tokens[provider] = self.provider_tokens.get(provider, 0) + total
        self.total_tokens += total
        self._post(TokensUpdated(provider, input_tokens, output_tokens))

    def set_thinking(self, provider: str, thinking: bool, thought: str = "") -> None:
        self.is_thinking = thinking
        self.current_thought = thought
        self._post(ThinkingChanged(provider, thinking, thought))

    def set_context_anchor(self, anchor: Optional[Usage]) -> None:
        """Record the last round's real usage -- the context-occupancy anchor."""
        self.context_anchor = anchor

    def mark_compaction(self) -> None:
        """Compaction happened: occupancy is unknown until the next response."""
        self.compaction_count += 1
        self.context_anchor = None

    def set_compaction_summary(self, summary: str) -> None:
        self.compaction_summary = summary

    def add_episode(self, episode: Episode) -> None:
        """Record a completed episode for context compaction."""
        self.episodes.append(episode)

    def apply_episode_compaction(self, compacted_count: int, episodes: List[Episode]) -> None:
        """Attach new episodes and mark the oldest active messages as compacted."""
        if compacted_count <= 0 and not episodes:
            return

        remaining = compacted_count
        for msg in self.messages:
            if remaining <= 0:
                break
            if msg.metadata.get("compacted"):
                continue
            msg.metadata["compacted"] = True
            remaining -= 1

        self.episodes.extend(episodes)

    def prune_live_episodes(self, keep_last: int) -> None:
        """Drop live episodes superseded by compaction (see episodes module)."""
        from .episodes import prune_live_episodes

        self.episodes = prune_live_episodes(self.episodes, keep_last)

    def post_stream_chunk(self, provider: str, chunk: str, done: bool = False) -> None:
        self._post(StreamChunk(provider, chunk, done))

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def message_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "you")

    @property
    def response_count(self) -> int:
        return sum(1 for m in self.messages if m.role != "you")
