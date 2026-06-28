"""Tests for the context builder / memory system."""

from cascade.context.memory import ContextBuilder


def test_empty_build():
    cb = ContextBuilder()
    assert cb.build() == ""
    assert cb.source_count == 0
    assert cb.token_estimate == 0


def test_add_text():
    cb = ContextBuilder()
    result = cb.add_text("Hello world", label="greeting")
    assert result is cb  # fluent API
    assert cb.source_count == 1
    assert cb.token_estimate > 0


def test_add_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("file content here")
    cb = ContextBuilder()
    cb.add_file(str(f))
    assert cb.source_count == 1
    built = cb.build()
    assert "file content here" in built
    assert "test.txt" in built


def test_add_file_not_found():
    cb = ContextBuilder()
    cb.add_file("/nonexistent/path.txt")
    assert cb.source_count == 1
    sources = cb.list_sources()
    assert sources[0]["type"] == "error"


def test_add_directory(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    (tmp_path / ".hidden").write_text("hidden")
    cb = ContextBuilder()
    cb.add_directory(str(tmp_path), "*.txt")
    assert cb.source_count == 2  # hidden file excluded


def test_add_directory_not_found():
    cb = ContextBuilder()
    cb.add_directory("/nonexistent/dir")
    assert cb.source_count == 1
    assert cb.list_sources()[0]["type"] == "error"


def test_add_image(tmp_path):
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG fake image data")
    cb = ContextBuilder()
    cb.add_image(str(img))
    assert cb.source_count == 1
    assert cb.list_sources()[0]["type"] == "image"


def test_build_format():
    cb = ContextBuilder()
    cb.add_text("test data", label="test")
    built = cb.build()
    assert "--- Context Sources ---" in built
    assert "test data" in built
    assert "1 sources" in built


def test_build_reuses_cached_result_until_context_changes():
    cb = ContextBuilder()
    cb.add_text("alpha", label="first")

    first = cb.build()
    second = cb.build()

    assert first is second

    cb.add_text("beta", label="second")
    third = cb.build()

    assert third != first
    assert "beta" in third


def test_clear():
    cb = ContextBuilder()
    cb.add_text("data")
    assert cb.source_count == 1
    cb.clear()
    assert cb.source_count == 0
    assert cb.build() == ""


def test_fluent_chaining():
    cb = ContextBuilder()
    result = cb.add_text("a").add_text("b").add_text("c")
    assert result is cb
    assert cb.source_count == 3


def test_token_limit(tmp_path):
    """Token limit should stop adding files from a directory."""
    cb = ContextBuilder(max_tokens=10)  # very low limit
    for i in range(50):
        (tmp_path / f"file_{i:03d}.txt").write_text("x" * 200)
    cb.add_directory(str(tmp_path))
    # Should have stopped before adding all 50
    assert cb.source_count < 50


def test_list_sources():
    cb = ContextBuilder()
    cb.add_text("hello", label="greeting")
    sources = cb.list_sources()
    assert len(sources) == 1
    assert sources[0]["type"] == "text"
    assert sources[0]["label"] == "greeting"
    assert sources[0]["size"] == 5
