"""Hardening tests for the `misc` group.

Covers:
- design.md forced ON reaches the /compete lanes (competition.py)
- recon_max_rounds malformed-value fallback matches the default (config.py)
"""

from unittest.mock import MagicMock

from cascade.config import ConfigManager
from cascade.swarm.competition import CompetitionOrchestrator


def _make_app(prompt_cfg: dict, base_prompt: str = "BASE PROMPT"):
    app = MagicMock()
    app.providers = {"claude": MagicMock()}
    app.config.get_prompt_config.return_value = prompt_cfg
    pipeline = MagicMock()
    pipeline.build.return_value = base_prompt
    app.prompt_pipeline = pipeline
    return app


class TestCompeteDesignLanguage:
    def test_forced_on_design_reaches_competition_system(self, tmp_path):
        design = tmp_path / "design.md"
        design.write_text("USE CALM MAGI TOKENS")
        app = _make_app(
            {
                "include_design_language": True,
                "design_md_path": str(design),
            }
        )
        compete = CompetitionOrchestrator(app, judge_provider="claude")

        system = compete._build_competition_system("EXTRA")

        assert system is not None
        assert "BASE PROMPT" in system
        assert "USE CALM MAGI TOKENS" in system
        assert "EXTRA" in system

    def test_forced_off_design_omitted(self, tmp_path):
        design = tmp_path / "design.md"
        design.write_text("SHOULD NOT APPEAR")
        app = _make_app(
            {
                "include_design_language": False,
                "design_md_path": str(design),
            }
        )
        compete = CompetitionOrchestrator(app, judge_provider="claude")

        system = compete._build_competition_system()

        assert system == "BASE PROMPT"

    def test_auto_gate_scopes_to_design_mode(self, tmp_path):
        design = tmp_path / "design.md"
        design.write_text("MODE SCOPED BRIEF")
        app = _make_app(
            {
                "include_design_language": None,
                "design_md_path": str(design),
            }
        )
        app.state = MagicMock()
        app.state.mode = "design"
        compete = CompetitionOrchestrator(app, judge_provider="claude")

        system = compete._build_competition_system()

        assert system is not None
        assert "MODE SCOPED BRIEF" in system

    def test_auto_gate_omitted_outside_design_mode(self, tmp_path):
        design = tmp_path / "design.md"
        design.write_text("MODE SCOPED BRIEF")
        app = _make_app(
            {
                "include_design_language": None,
                "design_md_path": str(design),
            }
        )
        app.state = MagicMock()
        app.state.mode = "build"
        compete = CompetitionOrchestrator(app, judge_provider="claude")

        system = compete._build_competition_system()

        assert system == "BASE PROMPT"


class TestReconMaxRoundsFallback:
    def test_malformed_value_falls_back_to_default(self):
        cfg = ConfigManager()
        cfg.data["orchestration"] = {"recon_max_rounds": "not-a-number"}
        merged = cfg.get_orchestration_config()
        assert merged["recon_max_rounds"] == 16

    def test_default_when_absent_is_16(self):
        cfg = ConfigManager()
        cfg.data["orchestration"] = {}
        merged = cfg.get_orchestration_config()
        assert merged["recon_max_rounds"] == 16
