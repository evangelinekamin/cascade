"""Conversation history conversion and context window management.

Converts CascadeState ChatMessage objects into provider-ready Message dicts,
with support for cross-model context policies, episode-based compaction,
and automatic compaction when conversations approach the model's context limit.

Episode-based compaction (inspired by Slate's Thread Weaving) replaces
lossy LLM summarization with structured episode records that preserve
key decisions, artifacts, and outcomes without burning model tokens.
"""

from typing import TYPE_CHECKING

from .context.budget import compact_threshold, current_tokens, window_for
from .episodes import Episode, compact_to_episodes, episodes_to_context

if TYPE_CHECKING:
    from .providers.usage import Usage
    from .state import ChatMessage
    from .providers.base import BaseProvider, Message


def _injectable_episodes(
    episodes: list[Episode],
    target_provider: str,
    policy: str = "summary",
) -> list[Episode]:
    """Episodes worth injecting for this provider under this policy.

    "live" episodes mirror turns that are still present as raw messages.
    Under "summary" the raw window drops other providers' turns, so live
    episodes inject cross-provider only. Under "full" every provider's
    raw turns are already included — live episodes never inject. Under
    "off" the user opted out of cross-model context entirely.

    Non-live episodes (compaction, orchestration) are the sole carrier of
    their content and inject everywhere — except that "off" restricts
    them to the target provider's own history plus orchestration results.
    """
    if policy == "full":
        return [ep for ep in episodes if ep.source != "live"]
    if policy == "off":
        return [
            ep for ep in episodes
            if ep.source != "live"
            and (ep.provider == target_provider or ep.source == "orchestration")
        ]
    return [
        ep for ep in episodes
        if ep.source != "live" or ep.provider != target_provider
    ]


def state_messages_to_provider(
    messages: list["ChatMessage"],
    target_provider: str,
    policy: str = "summary",
    episodes: list[Episode] | None = None,
    max_messages: int = 40,
    max_chars: int = 80_000,
) -> list["Message"]:
    """Convert CascadeState messages to provider-ready message list.

    Handles cross-model context injection based on policy:
    - "off": Only include messages from target_provider (and user messages)
    - "summary": Recent same-provider turns; other providers via episodes
    - "full": Include all recent messages regardless of provider

    Episodes are injected as structured context before the raw messages,
    filtered by provenance so live same-provider turns are never sent
    twice (see _injectable_episodes).
    """
    result: list[dict] = []
    visible_messages = [
        msg for msg in messages
        if not msg.metadata.get("compacted")
    ]

    injectable = _injectable_episodes(episodes or [], target_provider, policy)
    if injectable:
        episode_context = episodes_to_context(injectable, max_chars=max_chars // 4)
        if episode_context:
            result.append({
                "role": "user",
                "content": f"[Prior session context]\n{episode_context}",
            })
            result.append({
                "role": "assistant",
                "content": "Understood, I have the episode context from prior interactions.",
            })

    if policy == "off":
        for msg in visible_messages[-max_messages:]:
            if msg.role == "you":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == target_provider:
                result.append({"role": "assistant", "content": msg.content})

    elif policy == "summary":
        for msg in visible_messages[-max_messages:]:
            if msg.role == "you":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == target_provider:
                result.append({"role": "assistant", "content": msg.content})

    elif policy == "full":
        for msg in visible_messages[-max_messages:]:
            if msg.role == "you":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == target_provider:
                result.append({"role": "assistant", "content": msg.content})
            else:
                # Message from a different provider -- include as context
                result.append({
                    "role": "user",
                    "content": f"[Response from {msg.role}]\n{msg.content}",
                })
                result.append({
                    "role": "assistant",
                    "content": "Noted.",
                })

    # Enforce character budget by trimming oldest messages
    total_chars = sum(len(m["content"]) for m in result)
    while total_chars > max_chars and len(result) > 2:
        removed = result.pop(0)
        total_chars -= len(removed["content"])

    return result


def estimate_tokens(messages: list["Message"]) -> int:
    """Rough token estimate. ~1 token per 4 chars for English text."""
    return sum(len(m.get("content", "")) for m in messages) // 4


def needs_compaction(
    messages: list["Message"],
    provider: str,
    model: str = "",
    configured_window: int | None = None,
    anchor: "Usage | None" = None,
) -> bool:
    """Return True when context occupancy exceeds the compaction threshold.

    With an ``anchor`` (the last provider response's Usage), occupancy is
    the anchored total plus a chars/4 estimate of ``messages`` — the
    caller passes only the un-sent tail. Without one, the whole message
    list is estimated (legacy behavior).
    """
    window = window_for(provider, model, configured_window)
    chars = sum(len(m.get("content", "")) for m in messages)
    return current_tokens(anchor, chars) > compact_threshold(window)


def compact_messages_with_episodes(
    chat_messages: list["ChatMessage"],
    keep_recent: int = 6,
) -> tuple[list[Episode], list["ChatMessage"]]:
    """Episode-based compaction: convert old messages to episodes.

    Instead of burning model tokens on summarization, this extracts
    structured episodes from older messages. Episodes are compact,
    lossless for key information, and work across model switches.

    Returns:
        Tuple of (episodes, recent_messages).
    """
    active_messages = [
        msg for msg in chat_messages
        if not msg.metadata.get("compacted")
    ]
    return compact_to_episodes(active_messages, keep_recent=keep_recent)


def compact_messages(
    messages: list["Message"],
    provider: "BaseProvider",
    keep_recent: int = 6,
) -> list["Message"]:
    """Legacy compaction: summarize older messages via model call.

    Kept as fallback when episode-based compaction is not available.
    Prefer compact_messages_with_episodes() for new code.
    """
    if len(messages) <= keep_recent:
        return list(messages)

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m.get('content', '')[:1000]}"
        for m in old_messages
    )

    summary = provider.ask_single(
        "Summarize this conversation for continuation. Be concise but preserve "
        "key decisions, file paths, code changes, and open tasks:\n\n" + transcript,
        system="You produce compact engineering handoff summaries. Under 800 words.",
    )

    compacted: list[dict] = [
        {"role": "user", "content": f"[Conversation summary]\n{summary}"},
        {"role": "assistant", "content": "Understood, I have the context. Continuing."},
    ]
    compacted.extend(recent_messages)
    return compacted
