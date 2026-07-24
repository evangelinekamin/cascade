"""Render sweep: mount and paint every widget and screen at least once.

Three production crashes shared one profile -- a widget or screen the unit
suite CONSTRUCTED but never RENDERED, so a bad Rich style or a wrong Panel
kwarg only detonated in front of a user:

  * ``Panel(background=...)``          -> TypeError (Panel has no such kwarg),
                                          killed the app on the first file write
  * ``border_style "<color> 40%"``     -> MissingStyle ("40%" is not a color)
  * ``PALETTE.text`` (it is ``text_primary``) -> AttributeError on every
                                          permission prompt

Constructing a widget runs none of that code. This sweep DISCOVERS every
``Widget`` in ``cascade/widgets/`` and every ``Screen`` in ``cascade/screens/``
by walking the packages -- a newly added widget is covered automatically or the
registration test fails loudly -- gives each a representative instance via a
name-keyed factory, then forces a real render two ways:

  1. mount into a live Textual app and ``pilot.pause()`` so layout and paint run
     (this alone catches all three crashes above);
  2. for Rich-renderable widgets (those with their own ``render()``), also print
     ``render()`` through a Rich ``Console`` to the null device, because a bad
     style string only explodes when Rich RESOLVES it -- and a bare Console
     resolves a Panel's border even with color stripped.

A canary at the end proves the mount path actually detonates on a bad style, so
the sweep can never rot into theater that renders nothing.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from typing import Callable

import pytest
from rich.console import Console
from rich.errors import MissingStyle
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static

import cascade.screens as screens_pkg
import cascade.widgets as widgets_pkg
from cascade.commands import CommandDef
from cascade.state import CascadeState

# -- widgets (cascade/widgets/) ------------------------------------------------
from cascade.widgets.autocomplete import AutocompleteDropdown, CommandSuggestion
from cascade.widgets.code_block import CodeBlock
from cascade.widgets.diff_block import DiffBlock, WriteBlock
from cascade.widgets.header import ProviderGhostTable, WelcomeHeader
from cascade.widgets.input_frame import (
    ChatTextArea,
    FramedInput,
    InputFrame,
    ModeIndicator,
)
from cascade.widgets.message import (
    ChatHistory,
    GutterLabel,
    GutterSeparator,
    MessageBody,
    MessageWidget,
    OverflowIndicator,
    QueuePreview,
    QueuedPromptRow,
    ThinkingIndicator,
    TurnIndicator,
)
from cascade.widgets.odometer import OdometerCounter
from cascade.widgets.status_bar import StatusBar
from cascade.widgets.stream_message import StreamMessage, _ProseBody
from cascade.widgets.tool_call import (
    ToolActivityLog,
    ToolCallWidget,
    _ToolActivityRow,
    _ToolBody,
    _ToolGutter,
)

# -- screens (cascade/screens/) ------------------------------------------------
from cascade.screens.exit import ExitScreen
from cascade.screens.log_viewer import LogViewerScreen
from cascade.screens.main import MainScreen
from cascade.screens.permission import PermissionScreen
from cascade.screens.session_picker import SessionPickerScreen


# ---------------------------------------------------------------------------
# Discovery -- walk the packages so a new class is covered or fails loudly.
# ---------------------------------------------------------------------------

def _discover(pkg, base: type, exclude: type | None = None) -> dict[str, type]:
    """Every class defined in ``pkg`` that subclasses ``base`` (not ``exclude``).

    Keyed by class name. ``obj.__module__`` gates on classes DEFINED in the
    package, so re-exported imports (e.g. ``Static``) are not double-counted.
    """
    found: dict[str, type] = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{pkg.__name__}.{mod.name}")
        for obj in vars(module).values():
            if not inspect.isclass(obj) or obj.__module__ != module.__name__:
                continue
            if not issubclass(obj, base):
                continue
            if exclude is not None and issubclass(obj, exclude):
                continue
            found[obj.__name__] = obj
    return found


# A Screen is a Widget subclass, so the widget walk must exclude screens; the
# screen walk keeps only Screen subclasses (helper Statics living inside a
# screen module are painted transitively when their screen is pushed).
WIDGET_CLASSES: dict[str, type] = _discover(widgets_pkg, Widget, exclude=Screen)
SCREEN_CLASSES: dict[str, type] = _discover(screens_pkg, Screen)


# ---------------------------------------------------------------------------
# Factories -- representative constructor args, read from each real __init__.
# ---------------------------------------------------------------------------

_CMD = CommandDef("model", "/model <provider|reset>", "Switch active provider")

WIDGET_FACTORY: dict[str, Callable[[], Widget]] = {
    "AutocompleteDropdown": lambda: AutocompleteDropdown(),
    "ChatHistory": lambda: ChatHistory(max_widgets=10),
    "ChatTextArea": lambda: ChatTextArea(),
    "CodeBlock": lambda: CodeBlock("def f():\n    return 1\n", language="python", provider="openai"),
    "CommandSuggestion": lambda: CommandSuggestion(_CMD, selected=True),
    "DiffBlock": lambda: DiffBlock(
        "src/main.py", [(1, " ", "ctx"), (2, "-", "old line"), (3, "+", "new line")], lines_changed=2,
    ),
    "FramedInput": lambda: FramedInput("openai", context_label="ctx 5%"),
    "GutterLabel": lambda: GutterLabel("openai"),
    "GutterSeparator": lambda: GutterSeparator("claude"),
    "InputFrame": lambda: InputFrame(active_provider="gemini", mode="design", context_label="ctx 12.4k . 7%"),
    "MessageBody": lambda: MessageBody("a paragraph with `code` and **bold**"),
    "MessageWidget": lambda: MessageWidget("claude", "hello **world** with `code`", tokens=12),
    "ModeIndicator": lambda: ModeIndicator("build"),
    "OdometerCounter": lambda: OdometerCounter(123456),
    "OverflowIndicator": lambda: OverflowIndicator(Text("3 earlier messages")),
    "ProviderGhostTable": lambda: ProviderGhostTable(providers={}, active_provider="claude"),
    "StatusBar": lambda: StatusBar(cwd="~/proj", branch="main", provider_tokens={"gemini": 1200, "claude": 800}),
    "StreamMessage": lambda: StreamMessage("openai"),
    "ThinkingIndicator": lambda: ThinkingIndicator(provider="gemini", label="thinking..."),
    "TurnIndicator": lambda: TurnIndicator(provider="gemini", label="thinking..."),
    "QueuePreview": lambda: QueuePreview(),
    "QueuedPromptRow": lambda: QueuedPromptRow(0, "a queued prompt"),
    "ToolActivityLog": lambda: ToolActivityLog(),
    "_ToolActivityRow": lambda: _ToolActivityRow("read_file", {"path": "a.py"}, "ok"),
    "ToolCallWidget": lambda: ToolCallWidget("read_file", {"path": "a.py"}, "ok"),
    "WelcomeHeader": lambda: WelcomeHeader(active_provider="gemini", providers={}, version="0.3.0"),
    "WriteBlock": lambda: WriteBlock("package.json", '{"name": "x"}\n' * 20, language="json"),
    "_ProseBody": lambda: _ProseBody("prose line one\nprose line two"),
    "_ToolBody": lambda: _ToolBody("read_file", {"path": "a.py"}, "file contents here"),
    "_ToolGutter": lambda: _ToolGutter(),
}

SCREEN_FACTORY: dict[str, Callable[[], Screen]] = {
    "ExitScreen": lambda: ExitScreen("sess-abc123", "2m 14s", 4, 4, {"gemini": 1200, "claude": 800}),
    "LogViewerScreen": lambda: LogViewerScreen(
        "/solve log",
        ["+ added", "- removed", "@@ hunk @@", "diff --git a b", "[editing] file.py", "plain line"],
    ),
    "MainScreen": lambda: MainScreen(active_provider="gemini", mode="design"),
    "PermissionScreen": lambda: PermissionScreen("write_file", {"path": "src/config.ts"}, "workspace write"),
    "SessionPickerScreen": lambda: SessionPickerScreen([
        {"id": "night-river", "title": "Fix the parser", "provider": "claude",
         "model": "claude-opus-4-8", "message_count": 12,
         "updated_at": "2026-07-24T09:00:00+00:00", "cwd": "/home/eve/proj"},
    ]),
}

# Classes that genuinely cannot render standalone go here with a reason, never a
# silent omission. Empty today: every discovered class renders on a host.
WIDGET_SKIP: frozenset[str] = frozenset()
SCREEN_SKIP: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Hosts and render helpers
# ---------------------------------------------------------------------------

class _WidgetHost(App):
    """Minimal app to mount a single widget under test into."""

    def compose(self) -> ComposeResult:
        yield Static("host")


class _ScreenHost(App):
    """Host that carries ``state`` -- MainScreen reads ``self.app.state`` in
    compose/on_mount; harmless for the screens that do not."""

    def __init__(self) -> None:
        super().__init__()
        self.state = CascadeState()

    def compose(self) -> ComposeResult:
        yield Static("host")


def _is_rich_renderable(widget: Widget) -> bool:
    """True when the widget draws via its own ``render()`` (Static-style),
    False for pure containers that compose children (base ``Widget.render``)."""
    return type(widget).render is not Widget.render


# Widgets whose render() must return a specific Rich type. A widget that wraps
# its own render() in a bare ``except`` (CodeBlock, DiffBlock, WriteBlock all
# do) will otherwise PASS this sweep even when its Rich path is fully broken --
# the exception is swallowed and a degraded Text is returned. Asserting the
# type turns that silent degradation back into a failure.
RENDER_TYPE = {
    "CodeBlock": Panel,
    "DiffBlock": Panel,
    "WriteBlock": Panel,
}


def _force_rich_render(widget: Widget) -> None:
    """Resolve the widget's ``render()`` through Rich to the null device.

    Paint alone catches most style faults, but this is the direct reproduction
    of the ``Panel(background=...)`` / ``border_style "40%"`` crashes: Rich
    validates a Panel at construction and resolves its border on print. Where a
    widget promises a specific renderable, assert it, so a self-swallowed render
    fault (which degrades to Text) cannot slip through.
    """
    rendered = widget.render()
    expected = RENDER_TYPE.get(type(widget).__name__)
    if expected is not None:
        assert isinstance(rendered, expected), (
            f"{type(widget).__name__}.render() returned {type(rendered).__name__}, "
            f"expected {expected.__name__} -- its Rich path likely raised and was "
            "swallowed by its own except (the Panel(background=)/bad-style class)."
        )
    with open(os.devnull, "w") as devnull:
        Console(file=devnull, width=80).print(rendered)


# ---------------------------------------------------------------------------
# Registration completeness -- a new class must be wired up or fail loudly.
# ---------------------------------------------------------------------------

def test_widget_registry_covers_every_discovered_widget() -> None:
    uncovered = sorted(set(WIDGET_CLASSES) - set(WIDGET_FACTORY) - WIDGET_SKIP)
    assert not uncovered, (
        f"widgets in cascade/widgets/ with no render-sweep factory: {uncovered}. "
        "Add a factory (read its __init__) or an explicit WIDGET_SKIP with a reason."
    )


def test_screen_registry_covers_every_discovered_screen() -> None:
    uncovered = sorted(set(SCREEN_CLASSES) - set(SCREEN_FACTORY) - SCREEN_SKIP)
    assert not uncovered, (
        f"screens in cascade/screens/ with no render-sweep factory: {uncovered}. "
        "Add a factory (read its __init__) or an explicit SCREEN_SKIP with a reason."
    )


def test_discovery_found_the_known_classes() -> None:
    # Guards discovery itself: if the walk silently stopped finding classes the
    # per-class sweeps would vacuously pass. These anchors must always be seen.
    assert {"WriteBlock", "DiffBlock", "StatusBar", "StreamMessage"} <= set(WIDGET_CLASSES)
    assert {"PermissionScreen", "MainScreen", "ExitScreen"} <= set(SCREEN_CLASSES)


# ---------------------------------------------------------------------------
# The sweep -- one parametrized case per class so a failure names the culprit.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(WIDGET_CLASSES))
async def test_widget_mounts_and_renders(name: str) -> None:
    if name in WIDGET_SKIP:
        pytest.skip(f"{name} cannot render standalone (documented skip)")
    factory = WIDGET_FACTORY.get(name)
    if factory is None:
        pytest.fail(f"{name} discovered but has no factory")

    app = _WidgetHost()
    async with app.run_test() as pilot:
        widget = factory()
        await pilot.app.mount(widget)
        await pilot.pause()  # runs compose + layout + paint
        if _is_rich_renderable(widget):
            _force_rich_render(widget)
        await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(SCREEN_CLASSES))
async def test_screen_mounts_and_renders(name: str) -> None:
    if name in SCREEN_SKIP:
        pytest.skip(f"{name} cannot render standalone (documented skip)")
    factory = SCREEN_FACTORY.get(name)
    if factory is None:
        pytest.fail(f"{name} discovered but has no factory")

    app = _ScreenHost()
    async with app.run_test() as pilot:
        app.push_screen(factory())
        await pilot.pause()  # compose + on_mount
        await pilot.pause()  # paint the pushed screen
        assert type(app.screen).__name__ == name


# ---------------------------------------------------------------------------
# Canary -- prove mount+pause truly forces a render, so the sweep is not
# theater that mounts everything yet resolves nothing.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sweep_detonates_on_a_bad_style() -> None:
    class _Detonator(Static):
        DEFAULT_CSS = "_Detonator { height: 3; width: 100%; }"

        def render(self) -> Panel:
            # "40%" is not a color -- the exact shape of crash #2.
            return Panel(Text("boom"), border_style="#ffffff 40%")

    app = _WidgetHost()
    with pytest.raises(MissingStyle):
        async with app.run_test() as pilot:
            await pilot.app.mount(_Detonator())
            await pilot.pause()
