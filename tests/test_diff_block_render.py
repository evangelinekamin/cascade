"""DiffBlock/WriteBlock actually render (regression: Panel had no 'background').

These widgets became live in Phase D (mounted on every write/edit tool call).
Their render() built ``Panel(background=...)`` -- a kwarg Rich's Panel does not
accept -- so the first file write crashed the whole Textual app at layout time.
The unit suite never called render(), so it stayed green. These call render()
directly to reproduce the crash and lock in the fix.
"""

from rich.console import Console
from rich.panel import Panel

from cascade.widgets.diff_block import DiffBlock, WriteBlock


def _renders(widget) -> Panel:
    panel = widget.render()
    assert isinstance(panel, Panel)
    # Force Rich to actually realize it (Panel kwargs are validated at
    # construction, but render to be certain the whole path is exercised).
    Console(file=open("/dev/null", "w"), width=80).print(panel)
    return panel


def test_write_block_renders_without_panel_kwarg_error():
    block = WriteBlock("package.json", '{\n  "name": "x"\n}\n' * 20, language="json")
    _renders(block)


def test_diff_block_renders_without_panel_kwarg_error():
    diff = [(1, " ", "unchanged"), (2, "-", "old line"), (3, "+", "new line")]
    _renders(DiffBlock("src/main.py", diff, lines_changed=2))


def test_write_block_folds_long_file_but_still_renders():
    block = WriteBlock("big.py", "\n".join(f"line {i}" for i in range(200)))
    panel = _renders(block)
    assert panel is not None
