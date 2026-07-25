"""Episode-based compaction for conversation context.

Inspired by Slate's Thread Weaving architecture: instead of lossy
summarization when context fills up, generate structured "episodes"
that capture what was attempted, what succeeded, and key conclusions.

Episodes are compact, structured, and lossless for the important bits.
When context fills up, drop raw message history but keep episodes.
Cross-model episodes work naturally since they're structured data.
"""

import hashlib
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import ChatMessage


@dataclass(frozen=True)
class Episode:
    """A compressed record of a completed interaction sequence.

    Each episode captures the essential information from one exchange:
    what the user wanted, what actions were taken, what the result was,
    and what files/artifacts were involved.
    """

    id: str
    timestamp: float
    provider: str
    objective: str
    actions: tuple[str, ...]
    outcome: str
    artifacts: tuple[str, ...]
    tokens_consumed: int
    raw_turn_count: int
    # Provenance: "live" episodes mirror turns still present as raw
    # messages (injected only cross-provider); "compaction" episodes are
    # the sole carrier of compacted-away turns; "orchestration" episodes
    # carry lane results that never enter the raw message stream.
    source: str = "live"


# File extensions that indicate artifact references
_ARTIFACT_EXTENSIONS = frozenset((
    ".py", ".md", ".yaml", ".yml", ".json", ".ts", ".tsx",
    ".js", ".jsx", ".css", ".html", ".toml", ".cfg", ".ini",
    ".sh", ".bash", ".go", ".rs", ".java", ".c", ".cpp", ".h",
))


def _make_episode_id(provider: str, timestamp: float) -> str:
    """Generate a unique episode id with nonce to prevent collisions."""
    raw = f"{provider}:{timestamp}:{uuid.uuid4().hex[:8]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _extract_objective(user_content: str, max_len: int = 200) -> str:
    """Extract a concise objective from the user prompt."""
    # Take first meaningful line (skip code fences and their content)
    in_fence = False
    for line in user_content.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped and not stripped.startswith(("[", "---")):
            if len(stripped) > max_len:
                return stripped[:max_len] + "..."
            return stripped
    # Fallback: truncate the whole thing
    text = user_content.strip().replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _extract_artifacts(text: str, max_artifacts: int = 10) -> tuple[str, ...]:
    """Extract file paths and artifact references from text."""
    artifacts: list[str] = []
    seen: set[str] = set()

    for token in text.split():
        cleaned = token.strip("`.,;:()[]{}\"'<>")
        if not cleaned:
            continue

        # Path-like tokens (skip URLs)
        if cleaned.startswith(("http://", "https://", "ftp://")):
            continue
        is_path = "/" in cleaned
        is_file = any(cleaned.endswith(ext) for ext in _ARTIFACT_EXTENSIONS)

        if is_path or is_file:
            if cleaned not in seen:
                seen.add(cleaned)
                artifacts.append(cleaned)
                if len(artifacts) >= max_artifacts:
                    break

    return tuple(artifacts)


def _extract_actions(assistant_content: str, tool_log: list[dict] | None = None) -> tuple[str, ...]:
    """Extract action descriptions from tool calls or response content."""
    actions: list[str] = []

    # Tool calls are the most explicit actions
    if tool_log:
        for call in tool_log:
            tool_name = call.get("tool_name", call.get("name", "unknown"))
            actions.append(f"tool:{tool_name}")
        return tuple(actions)

    # Heuristic: look for action-like patterns in the response
    action_patterns = [
        (r"(?:created|wrote|generated)\s+[`'\"]?(\S+)[`'\"]?", "created"),
        (r"(?:modified|edited|updated)\s+[`'\"]?(\S+)[`'\"]?", "modified"),
        (r"(?:deleted|removed)\s+[`'\"]?(\S+)[`'\"]?", "deleted"),
        (r"(?:ran|executed|running)\s+[`'\"]?(.+?)[`'\"]?(?:\s|$)", "ran"),
    ]

    for pattern, action_type in action_patterns:
        for match in re.finditer(pattern, assistant_content, re.IGNORECASE):
            target = match.group(1).strip("`.,'\"")
            if target:
                actions.append(f"{action_type}:{target}")
                if len(actions) >= 8:
                    return tuple(actions)

    return tuple(actions)


# Fenced code (examples, snippets) is never the turn's OUTCOME, so it is stripped
# before the failure scan -- an ``except Exception:`` in a code sample must not be
# mistaken for a real failure.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Runner-shaped failure signals only -- deliberately narrow so success prose
# ("0 failed", "added Exception handling", "fixed the TypeError") does not match.
# A count must be non-zero; bare "Exception"/"NamedError:" are excluded because
# they appear just as often when reporting a fix as when reporting a break.
_FAILURE_LINE_RE = re.compile(
    r"^.*(?:"
    r"\bFAILED\b"                               # pytest FAILED line
    r"|\b(?!0\b)\d+\s+(?:failed|failing)\b"     # "3 failed" but not "0 failed"
    r"|Traceback \(most recent call last\)"     # a real traceback header
    r"|\bAssertionError\b"
    r"|[✗❌]"                          # cross marks
    r").*$",
    re.MULTILINE,
)


def _extract_outcome(assistant_content: str, max_len: int = 300) -> str:
    """Extract a concise outcome from the assistant response.

    A turn that reports failures keeps the FAIL lines verbatim -- a generic
    closing paragraph ("let me know if you need anything") would drop exactly
    what a later model needs to act on. Otherwise falls back to the last
    meaningful (non-code) paragraph as the conclusion.
    """
    scannable = _CODE_FENCE_RE.sub("", assistant_content)
    failures: list[str] = []
    seen: set[str] = set()
    for match in _FAILURE_LINE_RE.finditer(scannable):
        line = match.group(0).strip()
        if line and line not in seen:
            seen.add(line)
            failures.append(line)
    if failures:
        joined = "\n".join(failures)
        return joined[:max_len] + "..." if len(joined) > max_len else joined

    # Take the last meaningful paragraph as the conclusion
    paragraphs = [p.strip() for p in assistant_content.split("\n\n") if p.strip()]
    if not paragraphs:
        return assistant_content[:max_len]

    # Prefer the last non-code paragraph
    for para in reversed(paragraphs):
        # Skip code fences and their content
        if para.startswith("```") or para.endswith("```"):
            continue
        if len(para) >= 10:
            if len(para) > max_len:
                return para[:max_len] + "..."
            return para

    # Fallback: first non-code paragraph, or first paragraph
    for para in paragraphs:
        if not para.startswith("```"):
            if len(para) > max_len:
                return para[:max_len] + "..."
            return para

    result = paragraphs[0]
    if len(result) > max_len:
        return result[:max_len] + "..."
    return result


def generate_episode(
    user_content: str,
    assistant_content: str,
    provider: str,
    tokens: int = 0,
    tool_log: list[dict] | None = None,
    turn_count: int = 1,
    source: str = "live",
) -> Episode:
    """Generate an episode from a completed interaction.

    Args:
        user_content: The user's prompt text.
        assistant_content: The assistant's response text.
        provider: Which provider handled the interaction.
        tokens: Total tokens consumed.
        tool_log: Optional list of tool call records.
        turn_count: Number of message turns this episode covers.

    Returns:
        A structured Episode record.
    """
    now = time.time()
    combined_text = user_content + "\n" + assistant_content

    return Episode(
        id=_make_episode_id(provider, now),
        timestamp=now,
        provider=provider,
        objective=_extract_objective(user_content),
        actions=_extract_actions(assistant_content, tool_log),
        outcome=_extract_outcome(assistant_content),
        artifacts=_extract_artifacts(combined_text),
        tokens_consumed=tokens,
        raw_turn_count=turn_count,
        source=source,
    )


def episodes_to_context(episodes: list[Episode], max_chars: int = 4000) -> str:
    """Render episodes as structured context for the model.

    Returns a compact text block summarizing the episode history,
    suitable for injection into the system prompt or message context.
    """
    if not episodes:
        return ""

    blocks: list[str] = []
    total_chars = 0

    # Most recent episodes first (they're most relevant)
    for ep in reversed(episodes):
        lines = [f"[Episode {ep.id} | {ep.provider}]"]
        lines.append(f"  Objective: {ep.objective}")

        if ep.actions:
            actions_str = ", ".join(ep.actions[:6])
            lines.append(f"  Actions: {actions_str}")

        lines.append(f"  Outcome: {ep.outcome}")

        if ep.artifacts:
            artifacts_str = ", ".join(ep.artifacts[:6])
            lines.append(f"  Files: {artifacts_str}")

        block = "\n".join(lines)
        block_len = len(block) + 2

        if total_chars + block_len > max_chars:
            break

        blocks.append(block)
        total_chars += block_len

    # Reverse back to chronological order
    blocks.reverse()

    header = f"Episode history ({len(blocks)} of {len(episodes)} episodes):"
    return header + "\n\n" + "\n\n".join(blocks)


def prune_live_episodes(
    episodes: list[Episode],
    keep_last: int,
) -> list[Episode]:
    """Drop live episodes superseded by compaction, order-preserving.

    After compaction, old turns are carried by their new "compaction"
    episodes; the per-turn live episodes covering them are redundant.
    Keeps every non-live episode plus the last ``keep_last`` live ones
    (those mirroring the raw turns that were kept).
    """
    live_indices = [i for i, ep in enumerate(episodes) if ep.source == "live"]
    drop = set(live_indices[:-keep_last] if keep_last > 0 else live_indices)
    return [ep for i, ep in enumerate(episodes) if i not in drop]


def compact_to_episodes(
    messages: list["ChatMessage"],
    keep_recent: int = 6,
) -> tuple[list[Episode], list["ChatMessage"]]:
    """Split messages into episodes (old) + raw messages (recent).

    Generates episodes from older messages and returns them alongside
    the most recent messages that should be kept as-is.

    Returns:
        Tuple of (episodes, recent_messages).
    """
    if len(messages) <= keep_recent:
        return [], list(messages)

    old_messages = messages[:-keep_recent]
    recent_messages = list(messages[-keep_recent:])

    episodes: list[Episode] = []
    i = 0

    while i < len(old_messages):
        msg = old_messages[i]

        if msg.role == "you":
            # Find the corresponding assistant response
            user_content = msg.content
            assistant_content = ""
            provider = "unknown"
            tokens = 0
            turn_count = 1

            if i + 1 < len(old_messages) and old_messages[i + 1].role != "you":
                assistant_msg = old_messages[i + 1]
                assistant_content = assistant_msg.content
                provider = assistant_msg.role
                tokens = msg.tokens + assistant_msg.tokens
                turn_count = 2
                i += 2
            else:
                i += 1

            episode = generate_episode(
                user_content=user_content,
                assistant_content=assistant_content,
                provider=provider,
                tokens=tokens,
                turn_count=turn_count,
                source="compaction",
            )
            episodes.append(episode)
        else:
            # Orphan assistant message -- wrap it too
            episode = generate_episode(
                user_content="(continued)",
                assistant_content=msg.content,
                provider=msg.role,
                tokens=msg.tokens,
                turn_count=1,
                source="compaction",
            )
            episodes.append(episode)
            i += 1

    return episodes, recent_messages
