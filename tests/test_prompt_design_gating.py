"""Design-language gating: design.md must be scoped to design mode by default,
with an explicit prompts.include_design_language setting overriding the mode.

Regression guard for two bugs:
  - design.md riding on every system prompt regardless of mode (biasing output)
  - an explicit setting ignored at the layer that actually delivers the section
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from cascade.cli import CascadeCore
from cascade.config import ConfigManager
from cascade.prompts.default import build_default_prompt, design_language_section

_DESIGN_MARKER = "Scarlett Red Swiss avant-garde brief"


def _write_design(dirpath: Path) -> Path:
    design = dirpath / "design.md"
    design.write_text(_DESIGN_MARKER, encoding="utf-8")
    return design


# ---------------------------------------------------------------------------
# build_default_prompt: the injecting layer owns the gate
# ---------------------------------------------------------------------------

def test_auto_includes_design_only_in_design_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        design = _write_design(Path(tmpdir))
        prompt = build_default_prompt(
            include_design_language=None, mode="design", design_md_path=str(design),
        )
        assert _DESIGN_MARKER in prompt
        assert "Design Language:" in prompt


def test_auto_excludes_design_in_non_design_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        design = _write_design(Path(tmpdir))
        for mode in ("plan", "build", "test"):
            prompt = build_default_prompt(
                include_design_language=None, mode=mode, design_md_path=str(design),
            )
            assert _DESIGN_MARKER not in prompt, mode


def test_auto_excludes_design_when_no_mode():
    """The original bug: design.md rode every request. Auto + no mode -> absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        design = _write_design(Path(tmpdir))
        prompt = build_default_prompt(
            include_design_language=None, mode=None, design_md_path=str(design),
        )
        assert _DESIGN_MARKER not in prompt


def test_explicit_true_forces_on_in_non_design_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        design = _write_design(Path(tmpdir))
        prompt = build_default_prompt(
            include_design_language=True, mode="build", design_md_path=str(design),
        )
        assert _DESIGN_MARKER in prompt


def test_explicit_false_forces_off_in_design_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        design = _write_design(Path(tmpdir))
        prompt = build_default_prompt(
            include_design_language=False, mode="design", design_md_path=str(design),
        )
        assert _DESIGN_MARKER not in prompt


# ---------------------------------------------------------------------------
# config exposes the setting as a tri-state (absent == auto)
# ---------------------------------------------------------------------------

def test_fresh_config_omits_setting_so_absence_means_auto():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(str(Path(tmpdir) / "config.yaml"))
        assert "include_design_language" not in manager.data["prompts"]
        assert manager.get_prompt_config().get("include_design_language") is None


def test_config_passes_explicit_setting_through():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(str(Path(tmpdir) / "config.yaml"))
        manager.data["prompts"]["include_design_language"] = False
        assert manager.get_prompt_config().get("include_design_language") is False
        manager.data["prompts"]["include_design_language"] = True
        assert manager.get_prompt_config().get("include_design_language") is True


# ---------------------------------------------------------------------------
# design_language_section: the exact gate main._build_system_prompt applies
# per-request against the ACTIVE mode (not the static default mode).
# ---------------------------------------------------------------------------

def test_section_scopes_to_the_active_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        design = str(_write_design(Path(tmpdir)))
        assert design_language_section(None, "design", design)   # active design -> on
        assert not design_language_section(None, "build", design)  # active build -> off
        assert design_language_section(True, "build", design)    # forced on
        assert not design_language_section(False, "design", design)  # forced off


# ---------------------------------------------------------------------------
# real wiring: the base pipeline is mode-agnostic (design.md is injected
# per-request), so it must NEVER carry design.md -- otherwise a Shift+Tab out
# of design mode could not turn it off. Guards the round-1 regression where the
# section was pinned to the static default mode in the base prompt.
# ---------------------------------------------------------------------------

def _pipeline_text(manager: ConfigManager) -> str:
    core = SimpleNamespace(config=manager, project=SimpleNamespace(found=False))
    return CascadeCore._build_prompt_pipeline(core).build()


def test_base_pipeline_is_mode_agnostic_even_for_a_design_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        design = _write_design(Path(tmpdir))
        # gemini -> design mode default, and even an explicit True: the BASE
        # pipeline still must not bake it in (the active-mode assembler does).
        for setting in (None, True):
            manager = ConfigManager(str(Path(tmpdir) / f"{setting}.yaml"))
            manager.data["defaults"]["provider"] = "gemini"
            manager.data["prompts"]["design_md_path"] = str(design)
            if setting is not None:
                manager.data["prompts"]["include_design_language"] = setting
            assert _DESIGN_MARKER not in _pipeline_text(manager), setting


def test_per_request_prompt_follows_the_live_active_mode():
    """End-to-end: _build_system_prompt injects design.md by the ACTIVE mode,
    so a Shift+Tab (which only mutates self._mode) actually toggles it."""
    from cascade.screens.main import MainScreen

    with tempfile.TemporaryDirectory() as tmpdir:
        design = _write_design(Path(tmpdir))
        manager = ConfigManager(str(Path(tmpdir) / "config.yaml"))
        manager.data["prompts"]["design_md_path"] = str(design)  # auto (None)
        cli_app = SimpleNamespace(
            prompt_pipeline=CascadeCore._build_prompt_pipeline(
                SimpleNamespace(config=manager, project=SimpleNamespace(found=False))
            ),
            config=manager,
            context_builder=SimpleNamespace(source_count=0),
        )
        screen = MainScreen(active_provider="gemini", mode="design")

        screen._mode = "design"
        assert _DESIGN_MARKER in screen._build_system_prompt(cli_app, "hi", "gemini")

        screen._mode = "build"  # what Shift+Tab does
        assert _DESIGN_MARKER not in screen._build_system_prompt(cli_app, "hi", "gemini")
