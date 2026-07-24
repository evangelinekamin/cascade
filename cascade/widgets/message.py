"""Chat message widgets with gutter attribution.

ChatHistory -- scrollable container
MessageWidget -- horizontal row: GutterLabel + MessageBody
ThinkingIndicator -- braille spinner during provider processing
"""

import re
import time
from collections import OrderedDict
from hashlib import md5

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.message import Message
from textual.containers import Vertical, VerticalScroll
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
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HEADER = re.compile(r"^(#{1,6})\s+(.*)")
# A markdown table separator row: | --- | :--: | ---: |
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def render_md_line(line: str) -> Text:
    """Public: render a single markdown line (used by the stream renderer)."""
    return _render_md_line(line)


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
    spans: list[tuple[int, int, str, str, str]] = []
    for m in _INLINE_CODE.finditer(text):
        spans.append((m.start(), m.end(), "code", m.group(1), ""))
    for m in _BOLD.finditer(text):
        spans.append((m.start(), m.end(), "bold", m.group(1), ""))
    for m in _ITALIC.finditer(text):
        spans.append((m.start(), m.end(), "italic", m.group(1), ""))
    for m in _LINK.finditer(text):
        spans.append((m.start(), m.end(), "link", m.group(1), m.group(2)))

    spans.sort(key=lambda s: s[0])
    filtered = []
    last_end = 0
    for start, end, kind, content, extra in spans:
        if start >= last_end:
            filtered.append((start, end, kind, content, extra))
            last_end = end

    pos = 0
    for start, end, kind, content, extra in filtered:
        if start > pos:
            result.append(text[pos:start], style=PALETTE.text_primary)
        if kind == "code":
            result.append(f" {content} ", style=f"on {PALETTE.surface} {PALETTE.inline_code}")
        elif kind == "bold":
            result.append(content, style=f"bold {PALETTE.text_bright}")
        elif kind == "italic":
            result.append(content, style=f"italic {PALETTE.text_primary}")
        elif kind == "link":
            # Keep the URL visible -- a link with no target is decorative.
            result.append(content, style=f"underline {PALETTE.inline_code}")
            if extra and extra != content:
                result.append(f" ({extra})", style=f"dim {PALETTE.text_dim}")
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

    # Block-aware parse: tables and blockquotes span multiple lines and are
    # rendered as blocks; everything else is line-by-line.
    lines = content.split("\n")
    result = Text()
    first = True
    i = 0
    while i < len(lines):
        line = lines[i]
        # Table: a `|` row immediately followed by a separator row.
        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            block = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and "|" in lines[j] and lines[j].strip():
                block.append(lines[j])
                j += 1
            if not first:
                result.append("\n")
            result.append_text(_render_table(block))
            first = False
            i = j
            continue
        if not first:
            result.append("\n")
        if line.lstrip().startswith("> "):
            result.append_text(_render_blockquote(line))
        else:
            result.append_text(_render_md_line(line))
        first = False
        i += 1

    # Store in cache with LRU eviction
    _md_cache[key] = result.copy()
    if len(_md_cache) > _MD_CACHE_MAX:
        _md_cache.popitem(last=False)

    return result


def _split_table_row(row: str) -> list[str]:
    """Split a `| a | b |` row into stripped cell strings."""
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split("|")]


def _render_table(rows: list[str]) -> Text:
    """Render a markdown table as aligned columns (header bold + a rule).

    Models emit tables constantly; without this they render as raw pipe
    soup. Column widths are computed from the visible cell text so the
    columns line up in a monospace terminal.
    """
    header = _split_table_row(rows[0])
    body = [_split_table_row(r) for r in rows[2:]]  # rows[1] is the separator
    ncol = max([len(header)] + [len(r) for r in body]) if body else len(header)

    def _cell(cells: list[str], idx: int) -> str:
        return cells[idx] if idx < len(cells) else ""

    # Column widths use terminal DISPLAY width of the rendered cell (after
    # inline formatting strips markdown syntax), so CJK/wide glyphs and
    # markdown-styled cells still line up. Capped so one long cell can't
    # blow out the table.
    widths = []
    for c in range(ncol):
        w = cell_len(_inline_format(_cell(header, c)).plain)
        for r in body:
            w = max(w, cell_len(_inline_format(_cell(r, c)).plain))
        widths.append(min(w, 40))

    out = Text()
    # Header
    for c in range(ncol):
        if c:
            out.append("  ")
        out.append_text(_pad_inline(_cell(header, c), widths[c], bold=True))
    out.append("\n")
    # Rule
    out.append("  ".join("─" * widths[c] for c in range(ncol)), style=f"dim {PALETTE.text_dim}")
    # Body rows
    for r in body:
        out.append("\n")
        for c in range(ncol):
            if c:
                out.append("  ")
            out.append_text(_pad_inline(_cell(r, c), widths[c]))
    return out


def _pad_inline(cell: str, width: int, bold: bool = False) -> Text:
    """Inline-format a cell and right-pad to *width* display columns."""
    t = _inline_format(cell)
    if bold:
        t.stylize(f"bold {PALETTE.text_bright}")
    pad = width - cell_len(t.plain)
    if pad > 0:
        t.append(" " * pad)
    return t


def _render_blockquote(line: str) -> Text:
    """Render a `> quoted` line with a dim vertical bar."""
    content = line.lstrip()[2:]
    t = Text()
    t.append("│ ", style=f"dim {PALETTE.text_dim}")
    t.append_text(_inline_format(content))
    return t


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class OverflowIndicator(Static):
    """Clickable banner announcing trimmed messages held in overflow.

    Clicking (or activating) it asks the enclosing ``ChatHistory`` to
    re-mount a page of the most recently-trimmed messages, so a long
    session's scroll-back history stays reachable in the live UI.
    """

    DEFAULT_CSS = """
    OverflowIndicator {
        width: 100%;
        height: 1;
        text-align: center;
    }
    OverflowIndicator:hover {
        background: #161b22;
    }
    """

    class RestoreRequested(Message):
        """The user asked to reveal earlier (trimmed) messages."""

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.post_message(self.RestoreRequested())


class ChatHistory(VerticalScroll):
    """Scrollable container with virtual widget pooling.

    Caps mounted widgets at ``max_widgets``. When the limit is exceeded,
    the oldest widgets are unmounted and their data stored in
    ``_overflow``. A clickable banner then lets the user re-mount that
    history a page at a time (``restore_page``) -- and it remains available
    to ``/export``.
    """

    # How many trimmed messages a single banner click brings back.
    _RESTORE_PAGE = 50

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
        self._overflow_indicator: OverflowIndicator | None = None

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.scroll_relative(y=-6, animate=False, force=True)
        event.stop()
        event.prevent_default()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.scroll_relative(y=6, animate=False, force=True)
        event.stop()
        event.prevent_default()

    @staticmethod
    def _overflow_label(count: int) -> Text:
        return Text(
            f"  ▲ {count} earlier messages · click to show  ",
            style=f"dim {PALETTE.text_muted}",
        )

    async def trim_overflow(self) -> None:
        """Remove oldest content widgets when exceeding the pool cap.

        Counts ALL content rows -- messages, tool calls, diff/write blocks,
        compaction/bookmark notes -- not just MessageWidget, so an agentic
        session full of tool widgets cannot grow the mounted-widget count
        without bound. The overflow indicator and any live widget (thinking
        indicator, actively-streaming message) are always the newest child,
        so trimming oldest-first never removes them.
        """
        skip = {"ThinkingIndicator", "_ProseBody", "AutocompleteDropdown"}
        content = [
            c for c in self.children
            if c is not self._overflow_indicator
            and type(c).__name__ not in skip
        ]
        excess = len(content) - self._max_widgets
        if excess <= 0:
            return

        for widget in content[:excess]:
            # Only MessageWidget carries recoverable role/content; other rows
            # (tool output, diffs) are transient and dropped.
            if isinstance(widget, MessageWidget):
                self._overflow.append(
                    (getattr(widget, "_role", ""), getattr(widget, "_content", "")),
                )
            await widget.remove()

        # Show/update the overflow indicator only when messages were trimmed
        # (tool-row trimming alone leaves nothing recoverable to announce).
        count = len(self._overflow)
        if count == 0:
            return
        if self._overflow_indicator is None:
            self._overflow_indicator = OverflowIndicator(
                self._overflow_label(count), classes="overflow-indicator",
            )
            await self.mount(self._overflow_indicator, before=0)
        else:
            self._overflow_indicator.update(self._overflow_label(count))

    async def on_overflow_indicator_restore_requested(
        self, event: OverflowIndicator.RestoreRequested,
    ) -> None:
        event.stop()
        await self.restore_page()

    async def restore_page(self) -> None:
        """Re-mount the newest page of trimmed messages below the banner.

        Pages from the tail of ``_overflow`` (the messages just older than
        what is currently shown) so the reconstructed window stays contiguous,
        and mounts them in chronological order directly under the banner. The
        oldest-first trim invariant is preserved: a later ``trim_overflow``
        re-appends these top-most rows to ``_overflow`` in the same order.
        """
        if not self._overflow:
            return
        page = self._overflow[-self._RESTORE_PAGE:]
        del self._overflow[-len(page):]

        widgets = [MessageWidget(role, content) for role, content in page]
        anchor = self._overflow_indicator
        if anchor is not None:
            await self.mount_all(widgets, after=anchor)
        else:
            await self.mount_all(widgets, before=0)

        if self._overflow:
            self._overflow_indicator.update(self._overflow_label(len(self._overflow)))
        elif self._overflow_indicator is not None:
            await self._overflow_indicator.remove()
            self._overflow_indicator = None

        # Keep the freshly revealed history in view rather than jumping.
        if widgets:
            self.call_after_refresh(
                self.scroll_to_widget, widgets[0], top=True, animate=False,
            )

    @property
    def overflow_messages(self) -> list[tuple[str, str]]:
        """Access unmounted messages for export/history."""
        return list(self._overflow)


class GutterSeparator(Static):
    """Provider-accent left stroke marking each message (a colored ``\u258d``).

    Color-codes every message by whose it is -- the single biggest "why it looks
    better" move (kimchi) -- replacing the flat dim hairline.
    """

    DEFAULT_CSS = """
    GutterSeparator {
        width: 1;
        height: auto;
        padding: 0;
    }
    """

    def __init__(self, role: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._role = role

    def render(self) -> Text:
        if self._role in ("you", "system", ""):
            return Text("\u258d", style=f"dim {PALETTE.border_subtle}")
        return Text("\u258d", style=get_accent(self._role))


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
        background: #1b2130;
        border-left: thick #5a9cf0;
        margin-top: 1;
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
        yield GutterSeparator(self._role)
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


class TurnIndicator(ThinkingIndicator):
    """Turn status pinned just above the input frame for the whole turn.

    ThinkingIndicator mounts inline in the transcript, so on a long turn it
    scrolls off the top and stops being useful. This one lives in the screen's
    flow between the transcript and the input frame -- the 1fr transcript keeps
    it docked at the bottom -- so the spinner, activity text and elapsed clock
    stay visible until the turn ends, then it collapses to nothing. Reuses the
    parent's spinner rendering; start()/stop() own the spinner timer instead of
    mount/unmount, because the widget outlives any single turn.
    """

    DEFAULT_CSS = """
    TurnIndicator {
        display: none;
        height: 1;
        width: 100%;
        padding: 0 0 0 12;
    }
    TurnIndicator.active {
        display: block;
    }
    """

    def on_mount(self) -> None:
        # Persistent: the parent auto-starts its spinner on mount, but this one
        # must stay collapsed until a turn begins. start()/stop() drive it.
        pass

    def start(self, provider: str, label: str = "thinking...") -> None:
        """Reveal the indicator for a new turn and run the spinner."""
        self._provider = provider
        self._label = label
        self._idx = 0
        self._started_at = time.monotonic()
        self.add_class("active")
        if self._timer is None:
            self._timer = self.set_interval(0.1, self._tick)
        self.refresh()

    def stop(self) -> None:
        """Collapse to nothing and halt the spinner."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.remove_class("active")

    @property
    def active(self) -> bool:
        return self.has_class("active")


def summarize_queued_prompt(prompt: str, width: int = 56) -> str:
    """One-line, length-capped preview of a queued prompt."""
    body = prompt.strip()
    line = body.splitlines()[0] if body else ""
    extra = body.count("\n")
    if len(line) > width:
        line = line[: width - 1].rstrip() + "…"
    if extra:
        line = f"{line} (+{extra} lines)"
    return line


class QueuePreview(Vertical):
    """Pending type-ahead prompts, shown just above the input frame.

    Collapses to nothing when the queue is empty. Holds no queue state of its
    own: the screen owns the FIFO deque and calls refresh_queue() from every
    site that mutates it (enqueue, drain, interrupt-clear, retract). Each row
    is clickable to retract that prompt before it runs.
    """

    class RetractRequested(Message):
        """A queued prompt should be dropped before it is dispatched."""

        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    DEFAULT_CSS = """
    QueuePreview {
        display: none;
        height: auto;
        width: 100%;
        padding: 0;
    }
    QueuePreview.has-items {
        display: block;
    }
    """

    def refresh_queue(self, prompts: list[str]) -> None:
        """Rebuild the rows from the current queue contents."""
        self.remove_children()
        if prompts:
            self.add_class("has-items")
            self.mount_all([QueuedPromptRow(i, p) for i, p in enumerate(prompts)])
        else:
            self.remove_class("has-items")


class QueuedPromptRow(Static):
    """A single queued prompt; click it to retract it before it runs."""

    DEFAULT_CSS = """
    QueuedPromptRow {
        height: 1;
        width: 100%;
        padding: 0 0 0 12;
    }
    QueuedPromptRow:hover {
        background: #161b22;
    }
    """

    def __init__(self, index: int, prompt: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._index = index
        self._prompt = prompt

    def render(self) -> Text:
        t = Text()
        t.append("× ", style=PALETTE.text_dim)  # retract affordance
        t.append(summarize_queued_prompt(self._prompt), style=f"dim {PALETTE.text_dim}")
        t.append("  queued", style=f"dim {PALETTE.text_muted}")
        return t

    def on_click(self, event: events.Click) -> None:
        event.stop()
        # Index is fresh: the screen rebuilds the whole preview on every queue
        # mutation, so the row's index always matches the live deque.
        self.post_message(QueuePreview.RetractRequested(self._index))
