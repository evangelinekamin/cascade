"""read_file is bounded + pageable so one read can't flood the context."""

from cascade.plugins.file_ops import (
    FileOpsPlugin,
    _MAX_READ_LINES,
    _MAX_READ_CHARS,
)


def test_small_file_returned_verbatim(tmp_path):
    p = tmp_path / "small.py"
    p.write_text("line1\nline2\nline3")
    assert FileOpsPlugin.read_file(str(p)) == "line1\nline2\nline3"


def test_large_file_truncated_with_spill_notice(tmp_path):
    p = tmp_path / "big.py"
    p.write_text("\n".join(f"line {i}" for i in range(5000)))
    out = FileOpsPlugin.read_file(str(p))
    assert "line 0" in out                 # head kept
    assert "line 4999" not in out          # tail dropped
    assert "of 5000" in out                # total reported
    assert "offset/limit" in out           # paging hint present
    # The full file was spilled and is recoverable.
    assert "file-reads" in out


def test_default_cap_is_line_bounded(tmp_path):
    p = tmp_path / "many.txt"
    p.write_text("\n".join(str(i) for i in range(_MAX_READ_LINES + 500)))
    out = FileOpsPlugin.read_file(str(p))
    body = out.split("\n\n[")[0]
    assert body.count("\n") + 1 <= _MAX_READ_LINES


def test_offset_and_limit_page_a_window(tmp_path):
    p = tmp_path / "paged.txt"
    p.write_text("\n".join(f"L{i}" for i in range(100)))
    out = FileOpsPlugin.read_file(str(p), offset=10, limit=5)
    body = out.split("\n\n[")[0]
    assert body == "L10\nL11\nL12\nL13\nL14"
    assert "lines 11-15 of 100" in out


def test_char_cap_on_pathological_long_lines(tmp_path):
    p = tmp_path / "wide.txt"
    p.write_text("x" * (_MAX_READ_CHARS * 2))  # one enormous line
    out = FileOpsPlugin.read_file(str(p))
    body = out.split("\n\n[")[0]
    assert len(body) <= _MAX_READ_CHARS
    assert "truncated at" in out


def test_missing_file_reports_error(tmp_path):
    out = FileOpsPlugin.read_file(str(tmp_path / "nope.txt"))
    assert out.startswith("Error reading file:")
