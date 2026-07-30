"""/apply lands a solve patch on the real working tree, atomically."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from cascade.commands import CommandHandler
from cascade.swarm.worktree import WorktreeManager


def _init_repo(root: Path):
    def git(*a):
        subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)
    git("init")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (root / "mod.py").write_text("value = 1\n")
    git("add", "-A")
    git("commit", "-m", "init")


class TestApplyToTree:
    def test_applies_a_patch(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        # A patch that changes value = 1 -> value = 2
        patch = (
            "diff --git a/mod.py b/mod.py\n"
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        )
        ok, msg = WorktreeManager.apply_to_tree(str(tmp_path), patch)
        assert ok, msg
        assert (tmp_path / "mod.py").read_text() == "value = 2\n"

    def test_conflict_leaves_tree_untouched(self, tmp_path):
        _init_repo(tmp_path)
        # The file has drifted; the patch's context no longer matches.
        (tmp_path / "mod.py").write_text("value = 99\n")
        patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        )
        ok, msg = WorktreeManager.apply_to_tree(str(tmp_path), patch)
        assert not ok
        assert "could not apply" in msg
        # Atomic: the drifted content is untouched.
        assert (tmp_path / "mod.py").read_text() == "value = 99\n"

    def test_empty_patch(self, tmp_path):
        ok, msg = WorktreeManager.apply_to_tree(str(tmp_path), "")
        assert not ok
        assert "nothing" in msg


class TestApplyCommand:
    def _handler(self):
        app = MagicMock()
        posted = []
        h = CommandHandler(app)
        h._post_system = lambda m: posted.append(m)
        return h, posted

    def test_apply_without_solve_says_nothing(self):
        h, posted = self._handler()
        h._cmd_apply([])
        assert "Nothing to apply" in posted[0]

    def test_apply_lands_and_clears(self, tmp_path, monkeypatch):
        _init_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        h, posted = self._handler()
        h._last_solve_patch = (
            "diff --git a/mod.py b/mod.py\n--- a/mod.py\n+++ b/mod.py\n"
            "@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        )
        h._last_solve_changed = ("mod.py",)
        h._cmd_apply([])
        assert "Applied" in posted[0] and "mod.py" in posted[0]
        assert (tmp_path / "mod.py").read_text() == "value = 2\n"
        # One-shot: a second apply is a no-op.
        h._cmd_apply([])
        assert "Nothing to apply" in posted[-1]
