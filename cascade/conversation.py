"""Conversation history conversion and context window management.

Converts CascadeState ChatMessage objects into provider-ready Message dicts,
with support for cross-model context policies, episode-based compaction,
and automatic compaction when conversations approach the model's context limit.

Episode-based compaction (inspired by Slate's Thread Weaving) replaces
lossy LLM summarization with structured episode records that preserve
key decisions, artifacts, and outcomes without burning model tokens.
"""

import time
from typing import TYPE_CHECKING

from .context.budget import compact_threshold, current_tokens, window_for
from .episodes import Episode, compact_to_episodes, episodes_to_context

# Raw messages kept in the provider payload; older turns must be carried
# by compaction episodes, so exceeding this is itself a compaction trigger
# (otherwise the clip in state_messages_to_provider silently drops them).
RAW_MESSAGE_WINDOW = 40

# A gap at or above this (seconds) is worth marking in the timeline so the
# model can tell the user stepped away. Smaller consecutive gaps stay
# unmarked to avoid cluttering rapid back-and-forth.
TIMELINE_GAP_SECONDS = 600


def _humanize_gap(seconds: float) -> str:
    """'2h 15m later' / '3d later' -- coarse elapsed-time phrasing."""
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m later"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m later" if minutes else f"{hours}h later"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h later" if hours else f"{days}d later"


def _timeline_marker(ts: float, prev_ts: float, force: bool) -> str:
    """A bracketed time marker to prepend to a message, or '' for none.

    Emitted on the first stamped message, on calendar-day changes, on gaps
    >= TIMELINE_GAP_SECONDS, and whenever ``force`` (the latest turn, so the
    model always knows the current time). Absolute timestamps make the
    payload a temporal transcript; a relative hint is added for large gaps.
    Markers depend only on immutable message timestamps, so the payload
    prefix stays byte-stable across turns.
    """
    if ts <= 0:
        return ""
    if prev_ts <= 0:
        return f"[{time.strftime('%Y-%m-%d %a %H:%M', time.localtime(ts))}]\n"
    same_day = time.localtime(ts)[:3] == time.localtime(prev_ts)[:3]
    gap = ts - prev_ts
    big_gap = gap >= TIMELINE_GAP_SECONDS
    if not same_day:
        suffix = f", {_humanize_gap(gap)}" if big_gap else ""
        return f"[{time.strftime('%Y-%m-%d %a %H:%M', time.localtime(ts))}{suffix}]\n"
    if big_gap:
        return f"[{time.strftime('%H:%M', time.localtime(ts))}, {_humanize_gap(gap)}]\n"
    if force:
        return f"[{time.strftime('%H:%M', time.localtime(ts))}]\n"
    return ""

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
    compaction_summary: str = "",
    max_messages: int = RAW_MESSAGE_WINDOW,
    max_chars: int = 80_000,
    with_timeline: bool = False,
) -> list["Message"]:
    """Convert CascadeState messages to provider-ready message list.

    Handles cross-model context injection based on policy:
    - "off": Only include messages from target_provider (and user messages)
    - "summary": Recent same-provider turns; other providers via episodes
    - "full": Include all recent messages regardless of provider

    Episodes are injected as structured context before the raw messages,
    filtered by provenance so live same-provider turns are never sent
    twice (see _injectable_episodes).

    With ``with_timeline``, messages carry bracketed time markers (from their
    immutable creation timestamps) at points where time meaningfully jumps,
    so the model has temporal awareness of the session.
    """
    result: list[dict] = []
    visible_messages = [
        msg for msg in messages
        if not msg.metadata.get("compacted")
    ]

    injectable = _injectable_episodes(episodes or [], target_provider, policy)
    context_parts: list[str] = []
    # The carried summary mixes providers by construction; policy "off"
    # opted out of cross-model context entirely, so it never injects there
    # (episodes get the same treatment in _injectable_episodes).
    if compaction_summary and policy == "off":
        compaction_summary = ""
    if compaction_summary:
        context_parts.append(
            "Structured summary of compacted earlier turns:\n"
            + compaction_summary
        )
    if injectable:
        episode_context = episodes_to_context(injectable, max_chars=max_chars // 4)
        if episode_context:
            context_parts.append(episode_context)
    if context_parts:
        body = "\n\n".join(context_parts)
        result.append({
            "role": "user",
            "content": (
                "[Prior session context]\n" + body + "\n\n"
                "Continue seamlessly from this context. Do not acknowledge, "
                "recap, or refer to this summary in your replies."
            ),
        })
        result.append({
            "role": "assistant",
            "content": "Understood, I have the episode context from prior interactions.",
        })

    # off and summary emit identically here (they differ only in which
    # episodes inject, handled above); full additionally carries other
    # providers' turns inline. One loop covers all three, plus the optional
    # timeline markers keyed to the last EMITTED message's timestamp.
    include_cross = policy == "full"
    window = visible_messages[-max_messages:]
    prev_ts = 0.0
    for i, msg in enumerate(window):
        marker = (
            _timeline_marker(msg.timestamp, prev_ts, force=(i == len(window) - 1))
            if with_timeline else ""
        )
        emitted = True
        if msg.role == "you":
            result.append({"role": "user", "content": marker + msg.content})
        elif msg.role == target_provider:
            result.append({"role": "assistant", "content": marker + msg.content})
        elif include_cross:
            result.append({
                "role": "user",
                "content": marker + f"[Response from {msg.role}]\n{msg.content}",
            })
            result.append({"role": "assistant", "content": "Noted."})
        else:
            emitted = False
        # Gap is measured between messages the model actually sees, so a
        # skipped cross-provider turn's elapsed time folds into the next
        # visible gap rather than being double-counted.
        if emitted and msg.timestamp > 0:
            prev_ts = msg.timestamp

    # Enforce character budget by trimming oldest messages
    total_chars = sum(len(m["content"]) for m in result)
    while total_chars > max_chars and len(result) > 2:
        removed = result.pop(0)
        total_chars -= len(removed["content"])

    return result


def estimate_tokens(messages: list["Message"]) -> int:
    """Rough token estimate. ~1 token per 4 chars for English text."""
    return sum(len(m.get("content", "")) for m in messages) // 4


def unsent_tail_chars(messages: list["ChatMessage"]) -> int:
    """Chars of messages appended after the last provider response.

    The occupancy anchor (last round's real usage) already accounts for
    every message the model has seen; only the trailing user/system tail
    since that response needs estimating.
    """
    tail = 0
    for msg in reversed(messages):
        if msg.role not in ("you", "system"):
            break
        tail += len(msg.content)
    return tail


def should_compact(
    chat_messages: list["ChatMessage"],
    provider: str,
    model: str = "",
    configured_window: int | None = None,
    anchor: "Usage | None" = None,
) -> bool:
    """Compaction decision for the live conversation (pre-clip).

    Fires when token occupancy (anchored on the last round's real usage
    plus a chars/4 estimate) exceeds the window's compaction threshold,
    OR when the active message count exceeds RAW_MESSAGE_WINDOW — beyond
    which state_messages_to_provider would silently drop turns that no
    episode carries yet.
    """
    active = [m for m in chat_messages if not m.metadata.get("compacted")]
    if len(active) > RAW_MESSAGE_WINDOW:
        return True
    window = window_for(provider, model, configured_window)
    if anchor is not None:
        # The anchor already covers everything sent; estimating the whole
        # list on top of it double-counts (~2x) and fires compaction early.
        chars = unsent_tail_chars(active)
    else:
        chars = sum(len(m.content) for m in active)
    return current_tokens(anchor, chars) > compact_threshold(window)


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


_SUMMARY_SECTIONS = """1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections (exact paths; why each matters; short verbatim snippets only when load-bearing)
4. Errors and Fixes (include any user corrections verbatim)
5. Problem Solving (approaches tried, what worked, what was ruled out)
6. All User Messages (condensed, chronological)
7. Pending Tasks
8. Current Work and Next Step (quote the most recent instructions verbatim to prevent drift)"""

SUMMARY_SYSTEM_PROMPT = (
    "You produce structured context-compaction summaries for an ongoing "
    "coding session. Fidelity over brevity: preserve exact file paths, "
    "decisions, error messages, and user corrections. Output only the "
    "summary, nothing else."
)

# Compacted ranges smaller than this are carried fine by episodes alone --
# an LLM call would cost more than it preserves.
SUMMARY_MIN_CHARS = 4_000

# Defensive cap on the summarization request itself (~37k tokens). Beyond
# this the oldest turns are dropped with an explicit truncation marker.
SUMMARY_MAX_INPUT_CHARS = 150_000

_TRUNCATION_MARKER = "[earlier turns truncated to fit the compaction request]"


def build_compaction_summary_prompt(
    compacted: list["ChatMessage"],
    previous_summary: str = "",
    custom_instructions: str = "",
    max_input_chars: int = SUMMARY_MAX_INPUT_CHARS,
) -> str:
    """Build the tier-2 summarization prompt from FULL message contents.

    Never pre-truncates individual messages -- that destroys exactly what
    the summary needs. If the whole transcript exceeds ``max_input_chars``
    the oldest turns are dropped with an explicit marker instead.
    """
    blocks: list[str] = []
    for msg in compacted:
        label = "User" if msg.role == "you" else msg.role.capitalize()
        blocks.append(f"{label}: {msg.content}")

    transcript = "\n\n".join(blocks)
    if len(transcript) > max_input_chars:
        kept: list[str] = []
        total = 0
        for block in reversed(blocks):
            if total + len(block) > max_input_chars:
                break
            kept.append(block)
            total += len(block) + 2
        kept.reverse()
        transcript = _TRUNCATION_MARKER + "\n\n" + "\n\n".join(kept)

    parts = [
        "The following older conversation turns are being compacted out of "
        "the context window. Write a structured summary with EXACTLY these "
        "sections:",
        _SUMMARY_SECTIONS,
    ]
    if previous_summary:
        parts.append(
            "A previous summary already covers even earlier turns -- merge "
            "it forward so nothing is lost:\n" + previous_summary
        )
    if custom_instructions:
        parts.append("Additional instructions: " + custom_instructions)
    parts.append("Conversation to compact:\n\n" + transcript)
    return "\n\n".join(parts)


def validate_compaction_summary(summary: str) -> bool:
    """Reject empty, trivially short, or error-shaped summarizer output.

    Originals are never destroyed regardless (messages are only flagged
    compacted), but an invalid summary must not become injected context.
    """
    text = (summary or "").strip()
    if len(text) < 200:
        return False
    lowered = text.lower()
    if lowered.startswith(("error", "i cannot", "i can't", "sorry")):
        return False
    return True


def _looks_like_overflow(exc: Exception) -> bool:
    """Heuristic: does this provider error indicate a too-large request?"""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "context", "token", "too large", "too long", "413",
            "length", "maximum", "exceed",
        )
    )


def summarize_for_compaction(
    ask,
    compacted: list["ChatMessage"],
    previous_summary: str = "",
    custom_instructions: str = "",
) -> str | None:
    """Run the tier-2 summary through ``ask(prompt, system) -> str``.

    Returns the validated summary, or None when the range is too small to
    be worth a model call, the call fails, or the output fails validation.
    One retry with the oldest half dropped covers request-overflow errors.
    """
    total_chars = sum(len(m.content) for m in compacted)
    if total_chars < SUMMARY_MIN_CHARS:
        return None

    prompt = build_compaction_summary_prompt(
        compacted, previous_summary, custom_instructions,
    )
    try:
        summary = ask(prompt, SUMMARY_SYSTEM_PROMPT)
    except Exception as exc:
        if _looks_like_overflow(exc):
            # Size-shaped failure: drop the oldest half with an explicit
            # marker so the gap is visible in the produced summary.
            half = compacted[len(compacted) // 2:]
            retry_prompt = build_compaction_summary_prompt(
                half, previous_summary, custom_instructions,
                max_input_chars=SUMMARY_MAX_INPUT_CHARS // 2,
            )
            retry_prompt = (
                retry_prompt
                + "\n\nNote: earlier turns were dropped to fit this request; "
                "state that explicitly in the Pending Tasks section."
            )
        else:
            # Transient failure: plain retry, full range intact.
            retry_prompt = prompt
        try:
            summary = ask(retry_prompt, SUMMARY_SYSTEM_PROMPT)
        except Exception:
            return None

    return summary.strip() if validate_compaction_summary(summary) else None


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
