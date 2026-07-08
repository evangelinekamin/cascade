"""Tests for the parallel verified fan-out orchestrator."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import cascade.swarm.fanout as fanout_mod
from cascade.swarm.fanout import FanoutTask, _parse_subtasks, run_fanout


def test_parse_subtasks_parses_json():
    response = """Here is the plan:
    {"subtasks": [
      {"id": "sub_1", "description": "add A", "prompt": "add A to a.py", "files": ["a.py"]},
      {"id": "sub_2", "description": "add B", "prompt": "add B to b.py", "files": ["b.py"]}
    ]}"""
    tasks = _parse_subtasks("build A and B", response)
    assert [t.id for t in tasks] == ["sub_1", "sub_2"]
    assert tasks[0].files == ("a.py",)


def test_parse_subtasks_falls_back_to_whole_objective():
    tasks = _parse_subtasks("do the thing", "no json here at all")
    assert len(tasks) == 1
    assert tasks[0].prompt == "do the thing"


def _fanout_app():
    app = MagicMock()
    prov = MagicMock()
    prov.config = SimpleNamespace(model="frontier")
    app.providers = {"openrouter": prov}
    app.config.get_default_provider.return_value = "openrouter"
    app.config.get_model_for.return_value = "frontier"
    app.config.get_bulk_model.return_value = "bulk"
    app.config.get_escalation_target.return_value = None
    app.config.data = {"workflows": {"verify": {"test": "pytest"}}}
    return app


def _patch_manager(monkeypatch, apply_results=None):
    """Fake WorktreeManager; apply_results maps subtask path suffix -> apply bool."""
    fm = MagicMock()
    fm.prepare.side_effect = lambda name: SimpleNamespace(path=f"/tmp/wt/{name}")
    fm.diff_patch.side_effect = lambda path: f"patch::{path}"
    fm.apply_patch.side_effect = lambda integ_path, patch: (
        apply_results.get(patch, True) if apply_results else True
    )
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="2 files changed", diff_excerpt="+A\n+B", changed_files=("a.py", "b.py")
    )
    monkeypatch.setattr(fanout_mod, "WorktreeManager", lambda *a, **k: fm)
    return fm


def test_run_fanout_merges_passing_subtasks_and_verifies(monkeypatch):
    app = _fanout_app()
    monkeypatch.setattr(fanout_mod, "plan_subtasks", lambda *a, **k: [
        FanoutTask("sub_1", "add A", "add A to a.py", ("a.py",)),
        FanoutTask("sub_2", "add B", "add B to b.py", ("b.py",)),
    ])
    fm = _patch_manager(monkeypatch)
    monkeypatch.setattr(
        fanout_mod, "run_verified_task",
        lambda *a, **k: (SimpleNamespace(passed=True), ["bulk"], ["openrouter"]),
    )
    monkeypatch.setattr(fanout_mod, "_run_tests_in", lambda c, w, t: ("ok", 0))

    result = run_fanout(app, "build A and B")

    assert result.passed is True
    assert len(result.subs) == 2
    assert all(s.passed and s.integrated for s in result.subs)
    assert result.worktree_path == "/tmp/wt/_integration"
    assert fm.apply_patch.call_count == 2  # one merge per passing subtask


def test_run_fanout_skips_failing_and_conflicting_subtasks(monkeypatch):
    app = _fanout_app()
    monkeypatch.setattr(fanout_mod, "plan_subtasks", lambda *a, **k: [
        FanoutTask("sub_1", "ok", "p1", ("a.py",)),      # passes + merges
        FanoutTask("sub_2", "fails", "p2", ("b.py",)),   # fails -> never merged
        FanoutTask("sub_3", "conflict", "p3", ("a.py",)),  # passes but conflicts
    ])
    # sub_3's patch conflicts on apply
    _patch_manager(monkeypatch, apply_results={"patch::/tmp/wt/sub_3": False})

    passed_map = {"p1": True, "p2": False, "p3": True}
    monkeypatch.setattr(
        fanout_mod, "run_verified_task",
        lambda prov, path, prompt, *a, **k: (
            SimpleNamespace(passed=passed_map[prompt]), ["bulk"], ["openrouter"]
        ),
    )
    monkeypatch.setattr(fanout_mod, "_run_tests_in", lambda c, w, t: ("ok", 0))

    result = run_fanout(app, "mixed")

    by_id = {s.id: s for s in result.subs}
    assert by_id["sub_1"].integrated is True
    assert by_id["sub_2"].passed is False and by_id["sub_2"].integrated is False
    assert by_id["sub_3"].passed is True and by_id["sub_3"].integrated is False  # conflict
    assert result.passed is True  # tests green and at least one subtask merged


def test_run_fanout_missing_provider_errors():
    app = _fanout_app()
    app.providers = {}
    result = run_fanout(app, "x", provider_name="ghost")
    assert result.passed is False
    assert "not available" in result.error


def test_run_fanout_decomposes_on_the_escalation_frontier(monkeypatch):
    """Frontier directs bulk: the director is the escalation model (Opus), not the
    cheap primary."""
    app = _fanout_app()
    claude = MagicMock()
    claude.config = SimpleNamespace(model="claude-opus-4-8")
    app.providers = {
        "openrouter": MagicMock(config=SimpleNamespace(model="deepseek")),
        "claude": claude,
    }
    app.config.get_escalation_target.return_value = "claude"
    app.config.get_model_for.side_effect = lambda name, mode=None, fast=False: (
        "claude-opus-4-8" if name == "claude" else "deepseek"
    )

    captured = {}

    def fake_plan(app_, objective, director, director_model, on_progress=None):
        captured["director"] = director
        captured["model"] = director_model
        return [FanoutTask("sub_1", "x", "p1", ())]

    monkeypatch.setattr(fanout_mod, "plan_subtasks", fake_plan)
    _patch_manager(monkeypatch)
    monkeypatch.setattr(
        fanout_mod, "run_verified_task",
        lambda *a, **k: (SimpleNamespace(passed=True), ["b"], ["p"]),
    )
    monkeypatch.setattr(fanout_mod, "_run_tests_in", lambda c, w, t: ("ok", 0))

    run_fanout(app, "x", provider_name="openrouter")

    assert captured["director"] is claude
    assert captured["model"] == "claude-opus-4-8"


def test_clone_provider_gives_an_independent_config():
    from dataclasses import dataclass
    from cascade.swarm.fanout import _clone_provider

    @dataclass
    class _Cfg:
        model: str

    class _Prov:
        def __init__(self, config):
            self.config = config

    original = _Prov(_Cfg(model="m"))
    clone = _clone_provider(original)
    assert clone is not original and clone.config is not original.config
    clone.config.model = "changed"
    assert original.config.model == "m"  # mutating the clone never touches the original


def test_clone_provider_falls_back_for_non_dataclass_config():
    from cascade.swarm.fanout import _clone_provider

    class _Prov:
        def __init__(self, config):
            self.config = config

    original = _Prov(SimpleNamespace(model="m"))  # not a dataclass -> can't replace()
    assert _clone_provider(original) is original
    assert _clone_provider(None) is None


def test_run_fanout_aborts_on_a_red_base(monkeypatch):
    app = _fanout_app()
    monkeypatch.setattr(
        fanout_mod, "plan_subtasks", lambda *a, **k: [FanoutTask("sub_1", "x", "p1", ())]
    )
    _patch_manager(monkeypatch)
    ran: list[int] = []
    monkeypatch.setattr(
        fanout_mod, "run_verified_task",
        lambda *a, **k: (ran.append(1), (SimpleNamespace(passed=True), ["b"], ["p"]))[1],
    )
    # the base test suite fails -> the fan-out must bail before running any subtask
    monkeypatch.setattr(fanout_mod, "_run_tests_in", lambda c, w, t: ("2 failed", 1))

    result = run_fanout(app, "x")

    assert result.passed is False
    assert "not green" in result.error
    assert ran == []


# --- WorktreeManager merge primitives ------------------------------------------


def test_patched_paths_extracts_diff_targets():
    from cascade.swarm.worktree import WorktreeManager

    patch = (
        "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n a\n+b\n"
        "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n@@ -1 +1,2 @@\n c\n+d\n"
    )
    assert WorktreeManager._patched_paths(patch) == ("x.py", "y.py")


def test_has_conflict_markers_detects_only_marked_files(tmp_path):
    from cascade.swarm.worktree import WorktreeManager

    (tmp_path / "clean.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text("<<<<<<< ours\nx\n=======\ny\n>>>>>>> theirs\n")
    assert WorktreeManager._has_conflict_markers(tmp_path, ("clean.py",)) is False
    assert WorktreeManager._has_conflict_markers(tmp_path, ("bad.py",)) is True


def test_apply_patch_3way_lands_nonoverlapping_shared_edits(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    from cascade.swarm.worktree import WorktreeManager

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True)

    git("init")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "mod.py").write_text("A\nB\nC\n")
    git("add", "-A")
    git("commit", "-m", "base")

    monkeypatch.setenv("CASCADE_WORKTREE_ROOT", str(tmp_path / "wts"))
    mgr = WorktreeManager(cwd=str(repo))
    try:
        # two subtasks edit the same file in non-overlapping regions
        w1 = mgr.prepare("sub_1").path
        (Path(w1) / "mod.py").write_text("A\nA2\nB\nC\n")  # insert near the top
        w2 = mgr.prepare("sub_2").path
        (Path(w2) / "mod.py").write_text("A\nB\nC\nC2\n")  # append at the bottom

        integ = mgr.prepare("_integration").path
        assert mgr.apply_patch(integ, mgr.diff_patch(w1)) is True   # plain apply
        assert mgr.apply_patch(integ, mgr.diff_patch(w2)) is True   # 3-way merges

        merged = (Path(integ) / "mod.py").read_text()
        assert "A2" in merged and "C2" in merged  # both edits landed
    finally:
        mgr.cleanup()
