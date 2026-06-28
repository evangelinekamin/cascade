"""Chat message widgets with gutter attribution.

ChatHistory -- scrollable container
MessageWidget -- horizontal row: GutterLabel + MessageBody
ThinkingIndicator -- braille spinner during provider processing
"""

import re
import time
from collections import OrderedDict
from hashlib import md5

from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static

from ..theme import PALETTE, get_accent, get_shimmer, get_abbreviation


# ---------------------------------------------------------------------------
# Markdown parse cache (LRU, keyed by content hash)
# ---------------------------------------------------------------------------

_MD_CACHE_MAX = 500
_md_cache: OrderedDict[str, Text] = OrderedDict()

# Fast-path: skip markdown parsing for content with no syntax markers
_MD_SYNTAX_RE = re.compile(r"[#*`|>\[~_]|\n\n|\d+\. ")


def _cache_key(content: str) -> str:
    return md5(content.encode(), usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# Inline markdown helpers (ported from cascade/ui/markdown.py)
# ---------------------------------------------------------------------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADER = re.compile(r"^(#{1,6})\s+(.*)")


def _render_md_line(line: str) -> Text:
    """Convert a single markdown line to styled Rich Text."""
    # Headers
    m = _HEADER.match(line)
    if m:
        level = len(m.group(1))
        style = f"bold {PALETTE.text_bright}" if level <= 2 else f"bold {PALETTE.text_primary}"
        return Text(m.group(2), style=style)

    # Bullet lists
    stripped = line.lstrip()
    if stripped.startswith(("- ", "* ", "+ ")):
        indent = len(line) - len(stripped)
        t = Text()
        t.append(" " * indent)
        t.append(stripped[0], style=PALETTE.text_dim)
        t.append_text(_inline_format(f" {stripped[2:]}"))
        return t

    # Numbered lists
    num_match = re.match(r"^(\s*\d+\.)\s+(.*)", line)
    if num_match:
        t = Text()
        t.append(num_match.group(1), style=PALETTE.text_dim)
        t.append_text(_inline_format(f" {num_match.group(2)}"))
        return t

    return _inline_format(line)


def _inline_format(text: str) -> Text:
    """Apply inline code, bold, italic, link formatting."""
    result = Text()
    spans: list[tuple[int, int, str, str]] = []
    for m in _INLINE_CODE.finditer(text):
        spans.append((m.start(), m.end(), "code", m.group(1)))
    for m in _BOLD.finditer(text):
        spans.append((m.start(), m.end(), "bold", m.group(1)))
    for m in _ITALIC.finditer(text):
        spans.append((m.start(), m.end(), "italic", m.group(1)))
    for m in _LINK.finditer(text):
        spans.append((m.start(), m.end(), "link", m.group(1)))

    spans.sort(key=lambda s: s[0])
    filtered = []
    last_end = 0
    for start, end, kind, content in spans:
        if start >= last_end:
            filtered.append((start, end, kind, content))
            last_end = end

    pos = 0
    for start, end, kind, content in filtered:
        if start > pos:
            result.append(text[pos:start], style=PALETTE.text_primary)
        if kind == "code":
            result.append(f" {content} ", style=f"on {PALETTE.surface} {PALETTE.inline_code}")
        elif kind == "bold":
            result.append(content, style=f"bold {PALETTE.text_bright}")
        elif kind == "italic":
            result.append(content, style=f"italic {PALETTE.text_primary}")
        elif kind == "link":
            result.append(content, style=f"underline {PALETTE.inline_code}")
        pos = end

    if pos < len(text):
        result.append(text[pos:], style=PALETTE.text_primary)

    return result


def render_content(content: str) -> Text:
    """Render multi-line markdown content (prose only, no fenced blocks).

    Uses an LRU cache keyed by content hash and a fast-path that skips
    parsing for plain text with no markdown syntax markers.
    """
    if not content:
        return Text("")

    # Fast-path: no markdown syntax detected
    if not _MD_SYNTAX_RE.search(content):
        return Text(content, style=PALETTE.text_primary)

    # Check cache
    key = _cache_key(content)
    if key in _md_cache:
        _md_cache.move_to_end(key)
        return _md_cache[key].copy()

    # Full parse
    result = Text()
    for i, line in enumerate(content.split("\n")):
        if i > 0:
            result.append("\n")
        result.append_text(_render_md_line(line))

    # Store in cache with LRU eviction
    _md_cache[key] = result.copy()
    if len(_md_cache) > _MD_CACHE_MAX:
        _md_cache.popitem(last=False)

    return result


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class ChatHistory(VerticalScroll):
    """Scrollable container with virtual widget pooling.

    Caps mounted widgets at ``max_widgets``. When the limit is exceeded,
    the oldest widgets are unmounted and their data stored in
    ``_overflow`` for later export or scroll-back re-mount.
    """

    DEFAULT_CSS = """
    ChatHistory {
        height: 1fr;
        width: 100%;
        padding: 1 2;
        background: #0d1117;
    }
    """

    def __init__(self, max_widgets: int = 200, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_widgets = max_widgets
        self._overflow: list[tuple[str, str]] = []  # (role, content) pairs
        self._overflow_indicator: Static | None = None

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.scroll_relative(y=-6, animate=False, force=True)
        event.stop()
        event.prevent_default()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.scroll_relative(y=6, animate=False, force=True)
        event.stop()
        event.prevent_default()

    async def trim_overflow(self) -> None:
        """Remove oldest widgets when exceeding the pool cap."""
        children = list(self.children)
        # Only trim MessageWidget instances (not ThinkingIndicator, StreamMessage)
        msg_widgets = [c for c in children if isinstance(c, MessageWidget)]
        excess = len(msg_widgets) - self._max_widgets
        if excess <= 0:
            return

        for widget in msg_widgets[:excess]:
            role = getattr(widget, "_role", "")
            content = getattr(widget, "_content", "")
            self._overflow.append((role, content))
            await widget.remove()

        # Show/update overflow indicator
        count = len(self._overflow)
        if self._overflow_indicator is None:
            self._overflow_indicator = Static(
                Text(f"  {count} earlier messages  ", style=f"dim {PALETTE.text_muted}"),
                classes="overflow-indicator",
            )
            await self.mount(self._overflow_indicator, before=0)
        else:
            self._overflow_indicator.update(
                Text(f"  {count} earlier messages  ", style=f"dim {PALETTE.text_muted}")
            )

    @property
    def overflow_messages(self) -> list[tuple[str, str]]:
        """Access unmounted messages for export/history."""
        return list(self._overflow)


class GutterSeparator(Static):
    """Thin vertical hairline between gutter and message body."""

    DEFAULT_CSS = """
    GutterSeparator {
        width: 1;
        height: auto;
        padding: 0;
    }
    """

    def render(self) -> Text:
        return Text("\u2502", style=f"dim {PALETTE.border_subtle}")


class MessageWidget(Widget):
    """A single message row: gutter label + separator + body content."""

    DEFAULT_CSS = """
    MessageWidget {
        height: auto;
        width: 100%;
        padding: 0 0 1 0;
        layout: horizontal;
    }
    MessageWidget.user-message {
        background: #111820;
    }
    """

    def __init__(self, role: str, content: str, tokens: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._role = role
        self._content = content
        self._tokens = tokens
        if role == "you":
            self.add_class("user-message")

    def compose(self) -> ComposeResult:
        yield GutterLabel(self._role)
        yield GutterSeparator()
        yield MessageBody(self._content)


class GutterLabel(Static):
    """Fixed-width right-aligned label: provider name in accent / 'you' in dim."""

    DEFAULT_CSS = """
    GutterLabel {
        width: 10;
        min-width: 10;
        max-width: 10;
        height: auto;
        text-align: right;
        padding-right: 1;
    }
    """

    def __init__(self, role: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._role = role

    def render(self) -> Text:
        if self._role == "you":
            return Text(f"{'you':>8}", style=f"dim {PALETTE.text_dim}")
        if self._role == "system":
            return Text(f"{'sys':>8}", style=f"dim {PALETTE.text_dim}")
        accent = get_accent(self._role)
        abbr = get_abbreviation(self._role)
        return Text(f"{abbr:>8}", style=f"bold {accent}")


class MessageBody(Static):
    """The message content with inline markdown rendering."""

    DEFAULT_CSS = """
    MessageBody {
        width: 1fr;
        height: auto;
        padding-left: 1;
    }
    """

    def __init__(self, content: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content

    def render(self) -> Text:
        return render_content(self._content)


class ThinkingIndicator(Static):
    """Braille spinner shown while provider is processing."""

    SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
        width: 100%;
        padding: 0 0 0 12;
    }
    """

    def __init__(self, provider: str = "gemini", label: str = "thinking...", **kwargs) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._label = label
        self._idx = 0
        self._timer = None
        self._started_at = time.monotonic()

    def on_mount(self) -> None:
        self._started_at = time.monotonic()
        self._timer = self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        self._idx = (self._idx + 1) % len(self.SPINNER_FRAMES)
        self.refresh()

    def on_unmount(self) -> None:
        if self._timer:
            self._timer.stop()

    def set_label(self, label: str) -> None:
        """Update activity label while the spinner is active."""
        self._label = label
        self.refresh()

    def render(self) -> Text:
        # Oscillate between accent and shimmer for smooth animation
        color = get_accent(self._provider) if self._idx % 2 == 0 else get_shimmer(self._provider)
        ch = self.SPINNER_FRAMES[self._idx]
        elapsed = max(0, int(time.monotonic() - self._started_at))
        t = Text()
        t.append(ch, style=f"bold {color}")
        label = self._label.strip()
        if elapsed > 0:
            t.append(f" {label}  {elapsed}s", style=f"dim {PALETTE.text_dim}")
        else:
            t.append(f" {label}", style=f"dim {PALETTE.text_dim}")
        return t
