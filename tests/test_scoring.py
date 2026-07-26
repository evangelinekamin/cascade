"""Tests for /score: cache-hit math, disqualification, ranking, and manifests.

No real builds and no network calls: `_project_verify_test`, `_run_tests_in`
(the verify-command subprocess), and the judge's `ask_structured`/`ask_single`
are always mocked or faked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from cascade.commands import CommandHandler
from cascade.swarm.scoring import (
    CompetitionScorer,
    Leaderboard,
    ScoredCompetitor,
    cache_hit_pct,
    discover_latest_manifest,
    load_competitors,
    rank_leaderboard,
    resolve_manifest_path,
    tokens_per_second,
)
from cascade.swarm.scoring_rank import disqualification


def _row(
    label="m",
    model="m",
    gate="pass",
    quality_total=None,
    cost=1.0,
    tok_per_s=10.0,
    disqualified=False,
    pct=50.0,
) -> ScoredCompetitor:
    """A minimal ScoredCompetitor for exercising the pure ranking function."""
    return ScoredCompetitor(
        label=label,
        model=model,
        gate=gate,
        gate_output_tail="",
        spec_completeness=None,
        correctness=None,
        scope_discipline=None,
        quality_total=quality_total,
        quality_summary="",
        cost=cost,
        tokens=100,
        tok_per_s=tok_per_s,
        cache_hit_pct=pct,
        disqualified=disqualified,
    )


def _write_manifest(path: Path, objective: str, competitors: list) -> None:
    path.write_text(json.dumps({"objective": objective, "competitors": competitors}))


# --- Pure metrics -------------------------------------------------------------


class TestCacheHitPct:
    def test_normal_ratio(self):
        assert cache_hit_pct(50, 100) == 50.0

    def test_zero_prompt_total_is_zero_not_a_crash(self):
        assert cache_hit_pct(0, 0) == 0.0

    def test_full_cache_hit(self):
        assert cache_hit_pct(100, 100) == 100.0

    def test_negative_prompt_total_guarded_like_zero(self):
        assert cache_hit_pct(5, -1) == 0.0


class TestTokensPerSecond:
    def test_normal_rate(self):
        assert tokens_per_second(100, 2.0) == 50.0

    def test_zero_duration_guards_div_by_zero(self):
        assert tokens_per_second(500, 0.0) == 0.0

    def test_negative_duration_guards_div_by_zero(self):
        assert tokens_per_second(500, -1.0) == 0.0


# --- Disqualification ----------------------------------------------------------


class TestDisqualification:
    def test_disqualifies_below_one_percent_cache_hit(self):
        dq, notes = disqualification(0.5, True, ("a.py",))
        assert dq is True
        assert any("cache-hit" in n for n in notes)

    def test_not_disqualified_at_or_above_threshold(self):
        dq, notes = disqualification(1.0, True, ("a.py",))
        assert dq is False
        assert notes == ()

    def test_disqualifies_and_notes_no_op_when_no_changes(self):
        dq, notes = disqualification(50.0, False, ())
        assert dq is True
        assert notes == ("no-op: produced no changes",)

    def test_both_reasons_recorded_when_both_apply(self):
        dq, notes = disqualification(0.0, False, ())
        assert dq is True
        assert len(notes) == 2

    def test_success_with_no_changed_files_is_not_a_no_op(self):
        # success=True + empty changed_files does not occur in practice (see
        # CompetitionOrchestrator's "no changes produced" -> success=False), but
        # the predicate is literally `not success`, so confirm it stays False.
        dq, notes = disqualification(50.0, True, ())
        assert dq is False
        assert notes == ()


# --- Ranking ---------------------------------------------------------------------


class TestRankLeaderboard:
    def test_pass_beats_inconclusive_beats_fail(self):
        rows = [
            _row(label="fail-row", gate="fail"),
            _row(label="pass-row", gate="pass"),
            _row(label="inconclusive-row", gate="inconclusive"),
        ]
        ranked = rank_leaderboard(rows)
        assert [r.label for r in ranked] == ["pass-row", "inconclusive-row", "fail-row"]

    def test_quality_total_descending_with_none_sorting_last(self):
        rows = [
            _row(label="no-judge", quality_total=None),
            _row(label="low-quality", quality_total=5),
            _row(label="high-quality", quality_total=13),
        ]
        ranked = rank_leaderboard(rows)
        assert [r.label for r in ranked] == ["high-quality", "low-quality", "no-judge"]

    def test_cost_ascending_then_tok_per_s_descending_tiebreak(self):
        rows = [
            _row(label="expensive", cost=5.0, tok_per_s=100.0),
            _row(label="cheap-slow", cost=1.0, tok_per_s=10.0),
            _row(label="cheap-fast", cost=1.0, tok_per_s=50.0),
        ]
        ranked = rank_leaderboard(rows)
        assert [r.label for r in ranked] == ["cheap-fast", "cheap-slow", "expensive"]

    def test_disqualified_rows_always_sort_after_non_disqualified(self):
        rows = [
            _row(label="dq-but-great", quality_total=15, cost=0.01, disqualified=True),
            _row(label="ok-but-plain", quality_total=1, cost=9.0, disqualified=False),
        ]
        ranked = rank_leaderboard(rows)
        assert [r.label for r in ranked] == ["ok-but-plain", "dq-but-great"]

    def test_disqualified_rows_still_sorted_by_the_same_secondary_keys(self):
        rows = [
            _row(label="dq-fail", gate="fail", disqualified=True),
            _row(label="dq-pass", gate="pass", disqualified=True),
        ]
        ranked = rank_leaderboard(rows)
        assert [r.label for r in ranked] == ["dq-pass", "dq-fail"]


# --- Manifest path resolution + loading -----------------------------------------


class TestResolveManifestPath:
    def test_direct_file_path(self, tmp_path):
        manifest = tmp_path / "competition.json"
        manifest.write_text("{}")
        assert resolve_manifest_path(str(manifest)) == manifest

    def test_run_root_dir_finds_competition_json_inside(self, tmp_path):
        manifest = tmp_path / "competition.json"
        manifest.write_text("{}")
        assert resolve_manifest_path(str(tmp_path)) == manifest

    def test_missing_path_returns_none(self, tmp_path):
        assert resolve_manifest_path(str(tmp_path / "nope")) is None

    def test_dir_without_manifest_returns_none(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert resolve_manifest_path(str(empty_dir)) is None


class TestDiscoverLatestManifest:
    def test_picks_the_most_recently_written_manifest(self, tmp_path, monkeypatch):
        import os

        monkeypatch.setenv("CASCADE_WORKTREE_ROOT", str(tmp_path))
        older = tmp_path / "cascade-compete-aaa"
        older.mkdir()
        (older / "competition.json").write_text("{}")
        newer = tmp_path / "cascade-compete-bbb"
        newer.mkdir()
        (newer / "competition.json").write_text("{}")
        now = 1_700_000_000.0
        os.utime(older / "competition.json", (now - 100, now - 100))
        os.utime(newer / "competition.json", (now, now))

        assert discover_latest_manifest() == str(newer / "competition.json")

    def test_none_when_cache_root_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CASCADE_WORKTREE_ROOT", str(tmp_path / "does-not-exist"))
        assert discover_latest_manifest() is None

    def test_none_when_no_manifests_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CASCADE_WORKTREE_ROOT", str(tmp_path))
        (tmp_path / "unrelated-dir").mkdir()
        assert discover_latest_manifest() is None


class TestLoadCompetitors:
    def test_loads_from_run_root_dir(self, tmp_path):
        _write_manifest(
            tmp_path / "competition.json",
            "do the thing",
            [
                {
                    "label": "claude",
                    "model": "claude-x",
                    "success": True,
                    "changed_files": ["a.py"],
                    "worktree_path": "/tmp/wt",
                    "cost": 0.01,
                    "tokens": 100,
                    "duration_seconds": 2.0,
                    "cache_read": 50,
                    "prompt_total": 100,
                },
            ],
        )
        resolved, competitors = load_competitors([str(tmp_path)])
        assert resolved == (str(tmp_path / "competition.json"),)
        assert len(competitors) == 1
        assert competitors[0].label == "claude"
        assert competitors[0].objective == "do the thing"
        assert competitors[0].cache_read == 50
        assert competitors[0].prompt_total == 100

    def test_loads_from_direct_manifest_file_path(self, tmp_path):
        manifest = tmp_path / "run" / "competition.json"
        manifest.parent.mkdir()
        _write_manifest(manifest, "obj", [{"label": "openai"}])

        resolved, competitors = load_competitors([str(manifest)])

        assert resolved == (str(manifest),)
        assert competitors[0].label == "openai"
        # Fields absent from a minimal manifest entry default safely.
        assert competitors[0].success is False
        assert competitors[0].cache_read == 0
        assert competitors[0].changed_files == ()

    def test_skips_missing_and_malformed_paths_without_raising(self, tmp_path):
        good = tmp_path / "good"
        good.mkdir()
        _write_manifest(good / "competition.json", "obj", [{"label": "a"}])
        bad_json = tmp_path / "bad" / "competition.json"
        bad_json.parent.mkdir()
        bad_json.write_text("{not json")
        missing = tmp_path / "missing"

        resolved, competitors = load_competitors([str(good), str(bad_json), str(missing)])

        assert resolved == (str(good / "competition.json"),)
        assert len(competitors) == 1

    def test_merges_competitors_across_multiple_manifests(self, tmp_path):
        one = tmp_path / "one"
        one.mkdir()
        _write_manifest(one / "competition.json", "obj1", [{"label": "a"}])
        two = tmp_path / "two"
        two.mkdir()
        _write_manifest(two / "competition.json", "obj2", [{"label": "b"}])

        resolved, competitors = load_competitors([str(one), str(two)])

        assert len(resolved) == 2
        assert {c.label for c in competitors} == {"a", "b"}

    def test_skips_competitor_entries_missing_a_label(self, tmp_path):
        _write_manifest(
            tmp_path / "competition.json", "obj", [{"model": "x"}, {"label": "y"}]
        )
        _, competitors = load_competitors([str(tmp_path)])
        assert [c.label for c in competitors] == ["y"]


# --- CompetitionScorer: gate + judge side effects (mocked) -----------------------


class TestCompetitionScorerPipeline:
    @staticmethod
    def _manifest(tmp_path, competitors, objective="add a widget"):
        run_root = tmp_path / "run"
        run_root.mkdir()
        manifest = run_root / "competition.json"
        _write_manifest(manifest, objective, competitors)
        return manifest

    def test_gate_pass_and_structured_judge_produce_a_full_row(self, tmp_path, monkeypatch):
        worktree = tmp_path / "wt"
        worktree.mkdir()
        manifest = self._manifest(
            tmp_path,
            [
                {
                    "label": "claude",
                    "model": "claude-x",
                    "success": True,
                    "changed_files": ["a.py"],
                    "worktree_path": str(worktree),
                    "cost": 0.02,
                    "tokens": 1000,
                    "duration_seconds": 10.0,
                    "cache_read": 900,
                    "prompt_total": 1000,
                },
            ],
        )
        import cascade.swarm.scoring as scoring_mod

        monkeypatch.setattr(scoring_mod, "_project_verify_test", lambda path: "pytest -q")
        monkeypatch.setattr(
            scoring_mod, "_run_tests_in", lambda cmd, cwd, timeout: ("3 passed", 0)
        )
        judge = MagicMock()
        judge.ask_structured.return_value = {
            "spec_completeness": 4,
            "correctness": 5,
            "scope_discipline": 5,
            "summary": "Solid, focused change.",
        }

        scorer = CompetitionScorer([str(manifest)], judge)
        leaderboard = scorer.score()

        assert leaderboard.manifest_paths == (str(manifest),)
        row = leaderboard.rows[0]
        assert row.gate == "pass"
        assert row.quality_total == 14
        assert row.quality_summary == "Solid, focused change."
        assert row.cache_hit_pct == 90.0
        assert row.disqualified is False
        judge.ask_structured.assert_called_once()
        prompt = judge.ask_structured.call_args.args[0]
        assert "add a widget" in prompt

    def test_falls_back_to_ask_single_when_ask_structured_is_absent(self, tmp_path, monkeypatch):
        worktree = tmp_path / "wt"
        worktree.mkdir()
        manifest = self._manifest(
            tmp_path,
            [
                {
                    "label": "gemini",
                    "success": True,
                    "changed_files": ["a.py"],
                    "worktree_path": str(worktree),
                    "cache_read": 10,
                    "prompt_total": 100,
                },
            ],
        )
        import cascade.swarm.scoring as scoring_mod

        monkeypatch.setattr(scoring_mod, "_project_verify_test", lambda path: None)

        class _NoStructuredJudge:
            def ask_single(self, prompt, system=None):
                return json.dumps(
                    {
                        "spec_completeness": 2,
                        "correctness": 2,
                        "scope_discipline": 3,
                        "summary": "Partial.",
                    }
                )

        scorer = CompetitionScorer([str(manifest)], _NoStructuredJudge())
        row = scorer.score().rows[0]

        assert row.gate == "inconclusive"  # no verify command could be resolved
        assert row.quality_total == 7
        assert row.quality_summary == "Partial."

    def test_gate_classifies_infra_failure_as_inconclusive_not_fail(self, tmp_path, monkeypatch):
        worktree = tmp_path / "wt"
        worktree.mkdir()
        manifest = self._manifest(
            tmp_path,
            [
                {
                    "label": "deepseek",
                    "success": True,
                    "changed_files": ["a.py"],
                    "worktree_path": str(worktree),
                    "cache_read": 50,
                    "prompt_total": 100,
                },
            ],
        )
        import cascade.swarm.scoring as scoring_mod

        monkeypatch.setattr(scoring_mod, "_project_verify_test", lambda path: "npm test")
        monkeypatch.setattr(
            scoring_mod,
            "_run_tests_in",
            lambda cmd, cwd, timeout: ("sh: npm: command not found", 127),
        )
        judge = MagicMock()
        judge.ask_structured.return_value = {
            "spec_completeness": 3,
            "correctness": 3,
            "scope_discipline": 3,
            "summary": "ok",
        }

        row = CompetitionScorer([str(manifest)], judge).score().rows[0]

        assert row.gate == "inconclusive"

    def test_gate_times_out_is_inconclusive(self, tmp_path, monkeypatch):
        worktree = tmp_path / "wt"
        worktree.mkdir()
        manifest = self._manifest(
            tmp_path,
            [
                {
                    "label": "slow",
                    "success": True,
                    "changed_files": ["a.py"],
                    "worktree_path": str(worktree),
                    "cache_read": 50,
                    "prompt_total": 100,
                },
            ],
        )
        import cascade.swarm.scoring as scoring_mod

        monkeypatch.setattr(scoring_mod, "_project_verify_test", lambda path: "pytest -q")
        monkeypatch.setattr(
            scoring_mod,
            "_run_tests_in",
            lambda cmd, cwd, timeout: ("[tests timed out after 300s]", -1),
        )
        judge = MagicMock()
        judge.ask_structured.return_value = {
            "spec_completeness": 1,
            "correctness": 1,
            "scope_discipline": 1,
            "summary": "x",
        }

        row = CompetitionScorer([str(manifest)], judge).score().rows[0]

        assert row.gate == "inconclusive"

    def test_verify_override_replaces_project_verify_test_entirely(self, tmp_path, monkeypatch):
        worktree = tmp_path / "wt"
        worktree.mkdir()
        manifest = self._manifest(
            tmp_path,
            [
                {
                    "label": "x",
                    "success": True,
                    "changed_files": ["a.py"],
                    "worktree_path": str(worktree),
                    "cache_read": 50,
                    "prompt_total": 100,
                },
            ],
        )
        import cascade.swarm.scoring as scoring_mod

        def _must_not_be_called(path):
            raise AssertionError("_project_verify_test must not run when override is given")

        monkeypatch.setattr(scoring_mod, "_project_verify_test", _must_not_be_called)
        captured = {}

        def _fake_run_tests_in(cmd, cwd, timeout):
            captured["cmd"] = cmd
            return "ok", 0

        monkeypatch.setattr(scoring_mod, "_run_tests_in", _fake_run_tests_in)
        judge = MagicMock()
        judge.ask_structured.return_value = {
            "spec_completeness": 1,
            "correctness": 1,
            "scope_discipline": 1,
            "summary": "x",
        }

        CompetitionScorer([str(manifest)], judge, verify_command_override="make check").score()

        assert captured["cmd"] == "make check"

    def test_judge_error_records_none_quality_and_a_note_never_crashes(self, tmp_path, monkeypatch):
        worktree = tmp_path / "wt"
        worktree.mkdir()
        manifest = self._manifest(
            tmp_path,
            [
                {
                    "label": "x",
                    "success": True,
                    "changed_files": ["a.py"],
                    "worktree_path": str(worktree),
                    "cache_read": 50,
                    "prompt_total": 100,
                },
            ],
        )
        import cascade.swarm.scoring as scoring_mod

        monkeypatch.setattr(scoring_mod, "_project_verify_test", lambda path: None)
        judge = MagicMock()
        judge.ask_structured.side_effect = RuntimeError("network down")

        leaderboard = CompetitionScorer([str(manifest)], judge).score()  # must not raise

        row = leaderboard.rows[0]
        assert row.quality_total is None
        assert row.spec_completeness is None
        assert any("quality judge unavailable" in n for n in row.notes)

    def test_no_op_entry_is_disqualified_with_no_op_note_end_to_end(self, tmp_path):
        manifest = self._manifest(
            tmp_path,
            [
                {
                    "label": "lazy-model",
                    "success": False,
                    "error": "no changes produced",
                    "changed_files": [],
                    "worktree_path": "",
                    "cache_read": 80,
                    "prompt_total": 100,
                },
            ],
        )
        judge = MagicMock()
        judge.ask_structured.return_value = {
            "spec_completeness": 0,
            "correctness": 0,
            "scope_discipline": 0,
            "summary": "",
        }

        row = CompetitionScorer([str(manifest)], judge).score().rows[0]

        assert row.gate == "inconclusive"  # empty worktree_path -> missing worktree
        assert row.disqualified is True
        assert "no-op: produced no changes" in row.notes


# --- /score command ------------------------------------------------------------


class TestParseScoreArgs:
    def test_parses_paths_and_verify_override(self):
        result = CommandHandler._parse_score_args(
            ["/tmp/run1", "/tmp/run2", "--verify", "npm test"]
        )
        assert result == (["/tmp/run1", "/tmp/run2"], "npm test")

    def test_supports_inline_equals_form(self):
        result = CommandHandler._parse_score_args(["--verify=make check", "/tmp/run"])
        assert result == (["/tmp/run"], "make check")

    def test_no_args_returns_empty_paths_and_no_override(self):
        assert CommandHandler._parse_score_args([]) == ([], None)

    def test_dangling_verify_flag_is_malformed(self):
        assert CommandHandler._parse_score_args(["--verify"]) is None


class TestRenderLeaderboard:
    def test_marks_disqualified_rows_and_formats_columns(self):
        leaderboard = Leaderboard(
            rows=(
                ScoredCompetitor(
                    label="a",
                    model="model-a",
                    gate="pass",
                    gate_output_tail="",
                    spec_completeness=5,
                    correctness=4,
                    scope_discipline=3,
                    quality_total=12,
                    quality_summary="Nice work.",
                    cost=0.5,
                    tokens=200,
                    tok_per_s=20.0,
                    cache_hit_pct=75.0,
                    disqualified=False,
                    changed_files=("a.py", "b.py"),
                    notes=(),
                ),
                ScoredCompetitor(
                    label="b",
                    model="",
                    gate="fail",
                    gate_output_tail="",
                    spec_completeness=None,
                    correctness=None,
                    scope_discipline=None,
                    quality_total=None,
                    quality_summary="",
                    cost=0.1,
                    tokens=50,
                    tok_per_s=5.0,
                    cache_hit_pct=0.0,
                    disqualified=True,
                    changed_files=(),
                    notes=("disqualified: 0.0% cache-hit (below the 1% threshold)",),
                ),
            ),
            manifest_paths=("/tmp/competition.json",),
        )

        text = CommandHandler._render_leaderboard(leaderboard)

        assert "Leaderboard (2 competitor(s)):" in text
        assert "model-a" in text
        assert "12/15" in text
        assert "2 (DQ)" in text
        assert "| b |" in text  # falls back to the label when model is blank
        assert "-" in text  # quality column for the None-quality row


class TestWriteScoresJson:
    def test_writes_json_next_to_first_manifest(self, tmp_path):
        manifest = tmp_path / "run" / "competition.json"
        manifest.parent.mkdir()
        manifest.write_text("{}")
        leaderboard = Leaderboard(
            rows=(
                ScoredCompetitor(
                    label="a",
                    model="a",
                    gate="pass",
                    gate_output_tail="",
                    spec_completeness=1,
                    correctness=1,
                    scope_discipline=1,
                    quality_total=3,
                    quality_summary="",
                    cost=0.0,
                    tokens=0,
                    tok_per_s=0.0,
                    cache_hit_pct=0.0,
                    disqualified=False,
                ),
            ),
            manifest_paths=(str(manifest),),
        )

        out = CommandHandler._write_scores_json(leaderboard)

        assert out == str(manifest.parent / "scores.json")
        data = json.loads(Path(out).read_text())
        assert data["rows"][0]["label"] == "a"

    def test_returns_empty_string_when_no_manifest_paths(self):
        assert CommandHandler._write_scores_json(Leaderboard()) == ""


class TestScoreCommand:
    def test_renders_leaderboard_and_writes_scores_json(self, tmp_path):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {"claude": MagicMock()}
        app.cli_app = cli_app
        app.call_from_thread.side_effect = lambda fn, *a: fn(*a)
        app.screen = MagicMock()
        app.screen.run_worker.side_effect = lambda fn, **kwargs: fn()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)
        progress = MagicMock()
        progress.docked = False
        handler._mount_progress_indicator = MagicMock(return_value=progress)

        manifest = tmp_path / "competition.json"
        manifest.write_text(json.dumps({"objective": "obj", "competitors": []}))

        fake_leaderboard = Leaderboard(
            rows=(
                ScoredCompetitor(
                    label="claude",
                    model="claude-x",
                    gate="pass",
                    gate_output_tail="",
                    spec_completeness=5,
                    correctness=5,
                    scope_discipline=5,
                    quality_total=15,
                    quality_summary="Great.",
                    cost=0.01,
                    tokens=100,
                    tok_per_s=50.0,
                    cache_hit_pct=90.0,
                    disqualified=False,
                    changed_files=("a.py",),
                    notes=(),
                ),
            ),
            manifest_paths=(str(manifest),),
        )

        with patch("cascade.swarm.CompetitionOrchestrator") as mock_orch, patch(
            "cascade.swarm.scoring.CompetitionScorer"
        ) as mock_scorer_cls:
            mock_orch.return_value._judge_provider = "claude"
            mock_scorer_cls.return_value.score.return_value = fake_leaderboard

            handler._cmd_score([str(manifest)])

        mock_scorer_cls.assert_called_once_with(
            [str(manifest)], cli_app.providers["claude"], verify_command_override=None,
        )
        assert "Leaderboard (1 competitor(s)):" in posted[-1]
        assert "claude-x" in posted[-1]
        assert "Great." in posted[-1]
        scores_path = tmp_path / "scores.json"
        assert scores_path.is_file()
        written = json.loads(scores_path.read_text())
        assert written["rows"][0]["label"] == "claude"

    def test_auto_discovers_manifest_when_no_paths_given(self, tmp_path):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {"claude": MagicMock()}
        app.cli_app = cli_app
        app.call_from_thread.side_effect = lambda fn, *a: fn(*a)
        app.screen = MagicMock()
        app.screen.run_worker.side_effect = lambda fn, **kwargs: fn()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)
        handler._mount_progress_indicator = MagicMock(return_value=MagicMock(docked=False))

        manifest = tmp_path / "competition.json"
        manifest.write_text(json.dumps({"objective": "obj", "competitors": []}))
        empty_leaderboard = Leaderboard(rows=(), manifest_paths=(str(manifest),))

        with patch("cascade.swarm.CompetitionOrchestrator") as mock_orch, patch(
            "cascade.swarm.scoring.discover_latest_manifest", return_value=str(manifest)
        ) as mock_discover, patch(
            "cascade.swarm.scoring.CompetitionScorer"
        ) as mock_scorer_cls:
            mock_orch.return_value._judge_provider = "claude"
            mock_scorer_cls.return_value.score.return_value = empty_leaderboard

            handler._cmd_score([])

        mock_discover.assert_called_once()
        assert "Manifest(s) loaded but contained no competitors." in posted[-1]

    def test_reports_when_no_manifest_is_discoverable(self, tmp_path):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {"claude": MagicMock()}
        app.cli_app = cli_app
        app.screen = MagicMock()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)

        with patch("cascade.swarm.scoring.discover_latest_manifest", return_value=None):
            handler._cmd_score([])

        assert "No manifest path given" in posted[-1]

    def test_requires_cli_app(self):
        app = MagicMock()
        app.cli_app = None
        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)

        handler._cmd_score(["/tmp/x"])

        assert posted == ["Score requires CLI app."]

    def test_malformed_verify_flag_shows_usage(self):
        app = MagicMock()
        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)

        handler._cmd_score(["--verify"])

        assert "Usage: /score" in posted[0]
