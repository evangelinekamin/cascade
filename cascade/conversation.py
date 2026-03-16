"""Conversation history conversion and context window management.

Converts CascadeState ChatMessage objects into provider-ready Message dicts,
with support for cross-model context policies and automatic compaction
when conversations approach the model's context limit.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import ChatMessage
    from .providers.base import BaseProvider, Message


# Rough context window sizes by model family (tokens)
CONTEXT_WINDOWS: dict[str, int] = {
    "gemini": 1_000_000,
    "claude": 200_000,
    "openai": 200_000,
    "openrouter": 128_000,
}

COMPACTION_THRESHOLD = 0.75


def state_messages_to_provider(
    messages: list["ChatMessage"],
    target_provider: str,
    policy: str = "summary",
    cross_model_summary: str = "",
    max_messages: int = 40,
    max_chars: int = 80_000,
) -> list["Message"]:
    """Convert CascadeState messages to provider-ready message list.

    Handles cross-model context injection based on policy:
    - "off": Only include messages from target_provider (and user messages)
    - "summary": Include cross-model summary + recent same-provider turns
    - "full": Include all recent messages regardless of provider
    """
    result: list[dict] = []

    if policy == "off":
        for msg in messages[-max_messages:]:
            if msg.role == "you":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == target_provider:
                result.append({"role": "assistant", "content": msg.content})

    elif policy == "summary":
        if cross_model_summary:
            result.append({
                "role": "user",
                "content": f"[Context from previous model interactions]\n{cross_model_summary}",
            })
            result.append({
                "role": "assistant",
                "content": "Understood, I have the context from the previous interactions.",
            })
        for msg in messages[-max_messages:]:
            if msg.role == "you":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == target_provider:
                result.append({"role": "assistant", "content": msg.content})

    elif policy == "full":
        for msg in messages[-max_messages:]:
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


def needs_compaction(messages: list["Message"], provider: str) -> bool:
    """Return True when estimated tokens exceed the compaction threshold."""
    window = CONTEXT_WINDOWS.get(provider, 128_000)
    return estimate_tokens(messages) > int(window * COMPACTION_THRESHOLD)


def compact_messages(
    messages: list["Message"],
    provider: "BaseProvider",
    keep_recent: int = 6,
) -> list["Message"]:
    """Compact older messages into a summary, keeping recent ones intact.

    Uses the provider's own ask_single to generate a concise handoff
    summary of the older conversation turns.  Returns a new message
    list with the summary prepended and the *keep_recent* most recent
    messages appended unchanged.
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
