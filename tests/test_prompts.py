"""Tests for the prompt system: default prompt and pipeline layers."""

import os
import tempfile
from pathlib import Path

import pytest

from cascade.prompts.default import (
    build_default_prompt,
    _find_design_md,
)
from cascade.prompts.layers import (
    PromptLayer,
    PromptPipeline,
    PRIORITY_DEFAULT,
    PRIORITY_PROJECT_SYSTEM,
    PRIORITY_REPL_CONTEXT,
)


class TestDefaultPrompt:
    """Tests for the default system prompt builder."""

    def test_identity_present(self):
        prompt = build_default_prompt(include_design_language=False)
        assert "Cascade" in prompt
        assert "elegant solution" in prompt

    def test_quality_gates_present(self):
        prompt = build_default_prompt(include_design_language=False)
        assert "established architecture" in prompt
        assert "requested outcome" in prompt
        assert "hardcoded secrets" in prompt

    def test_workflow_present(self):
        prompt = build_default_prompt(include_design_language=False)
        assert "definition of done" in prompt
        assert "genuinely independent" in prompt
        assert "Verify the actual result" in prompt

    def test_tool_use_section(self):
        prompt = build_default_prompt(include_design_language=False)
        assert "repeat the same failed action" in prompt.lower()
        assert "proactively" in prompt.lower()

    def test_conventions_present(self):
        prompt = build_default_prompt(include_design_language=False)
        assert "do not commit" in prompt.lower()
        assert "emojis" in prompt.lower()

    def test_grounded_handoff_present(self):
        prompt = build_default_prompt(include_design_language=False)
        assert "remaining risk" in prompt
        assert "do not invent busywork" in prompt

    def test_current_date_injected(self):
        prompt = build_default_prompt(
            include_design_language=False,
            current_date="2026-02-27",
        )
        assert "2026-02-27" in prompt

    def test_custom_date_override(self):
        prompt = build_default_prompt(
            include_design_language=False,
            current_date="1999-12-31",
        )
        assert "1999-12-31" in prompt

    def test_design_language_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            design_path = Path(tmpdir) / "design.md"
            design_path.write_text("Swiss Design rules apply here.")
            prompt = build_default_prompt(
                include_design_language=True,
                design_md_path=str(design_path),
            )
            assert "Swiss Design rules" in prompt

    def test_design_language_missing_explicit_falls_back(self, tmp_path, monkeypatch):
        """When explicit path is missing, cwd fallback still runs."""
        monkeypatch.chdir(tmp_path)
        prompt = build_default_prompt(
            include_design_language=True,
            design_md_path="/nonexistent/design.md",
        )
        # No design.md in tmp_path, so no design language section
        assert "Design Language:" not in prompt


class TestFindDesignMd:
    """Tests for design.md discovery."""

    def test_explicit_path(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("test design content")
            f.flush()
            result = _find_design_md(explicit_path=f.name)
            assert result == "test design content"
            os.unlink(f.name)

    def test_explicit_path_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _find_design_md(explicit_path="/nonexistent/path.md")
        assert result is None

    def test_search_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            design_path = Path(tmpdir) / "design.md"
            design_path.write_text("found via search")
            result = _find_design_md(search_dirs=[tmpdir])
            assert result == "found via search"


class TestPromptLayer:
    """Tests for the frozen PromptLayer dataclass."""

    def test_frozen(self):
        layer = PromptLayer(name="test", content="hello", priority=10)
        with pytest.raises(AttributeError):
            layer.name = "changed"

    def test_fields(self):
        layer = PromptLayer(name="test", content="hello", priority=10)
        assert layer.name == "test"
        assert layer.content == "hello"
        assert layer.priority == 10


class TestPromptPipeline:
    """Tests for the immutable PromptPipeline."""

    def test_empty_pipeline(self):
        pipeline = PromptPipeline()
        assert pipeline.build() == ""
        assert pipeline.layer_count == 0

    def test_add_layer_returns_new_instance(self):
        p1 = PromptPipeline()
        p2 = p1.add_layer("test", "content", 10)
        assert p1.layer_count == 0
        assert p2.layer_count == 1

    def test_build_single_layer(self):
        pipeline = PromptPipeline().add_layer("test", "hello world", 10)
        assert pipeline.build() == "hello world"

    def test_build_multiple_layers_sorted(self):
        pipeline = (
            PromptPipeline()
            .add_layer("last", "third", 30)
            .add_layer("first", "first", 10)
            .add_layer("middle", "second", 20)
        )
        result = pipeline.build()
        parts = result.split("\n\n")
        assert parts == ["first", "second", "third"]

    def test_skip_empty_content(self):
        pipeline = (
            PromptPipeline()
            .add_layer("real", "hello", 10)
            .add_layer("empty", "", 20)
            .add_layer("whitespace", "   ", 30)
        )
        assert pipeline.layer_count == 1

    def test_remove_layer(self):
        pipeline = (
            PromptPipeline()
            .add_layer("keep", "stay", 10)
            .add_layer("remove", "go away", 20)
        )
        filtered = pipeline.remove_layer("remove")
        assert filtered.layer_count == 1
        assert "go away" not in filtered.build()

    def test_has_layer(self):
        pipeline = PromptPipeline().add_layer("test", "content", 10)
        assert pipeline.has_layer("test") is True
        assert pipeline.has_layer("other") is False

    def test_describe(self):
        pipeline = (
            PromptPipeline()
            .add_layer("default", "x" * 100, PRIORITY_DEFAULT)
            .add_layer("project", "y" * 50, PRIORITY_PROJECT_SYSTEM)
        )
        desc = pipeline.describe()
        assert len(desc) == 2
        assert desc[0]["name"] == "default"
        assert desc[0]["priority"] == PRIORITY_DEFAULT
        assert desc[0]["length"] == 100
        assert desc[1]["name"] == "project"

    def test_standard_priorities_ordering(self):
        pipeline = (
            PromptPipeline()
            .add_layer("repl", "repl_ctx", PRIORITY_REPL_CONTEXT)
            .add_layer("default", "identity", PRIORITY_DEFAULT)
            .add_layer("project", "proj_sys", PRIORITY_PROJECT_SYSTEM)
        )
        result = pipeline.build()
        idx_default = result.index("identity")
        idx_project = result.index("proj_sys")
        idx_repl = result.index("repl_ctx")
        assert idx_default < idx_project < idx_repl

    def test_content_is_stripped(self):
        pipeline = PromptPipeline().add_layer("test", "  hello  ", 10)
        assert pipeline.build() == "hello"
