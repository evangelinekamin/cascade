"""/apply must land repo-root-relative patches even from a subdirectory."""

import subprocess
from pathlib import Path

from cascade.swarm.worktree import WorktreeManager


def _init_repo(root: Path):
    def git(*a):
        subprocess.run(["git", *a], cwd=root, check=True, capture_output=True)
    git("init")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (root / "src").mkdir()
    (root / "src" / "foo.py").write_text("value = 1\n")
    (root / "tests").mkdir()
    (root / "tests" / "keep.py").write_text("x = 0\n")
    git("add", "-A")
    git("commit", "-m", "init")


_PATCH = (
    "diff --git a/src/foo.py b/src/foo.py\n"
    "--- a/src/foo.py\n+++ b/src/foo.py\n"
    "@@ -1 +1 @@\n-value = 1\n+value = 2\n"
)


def test_apply_from_subdirectory_lands_repo_root_path(tmp_path):
    """Review-critical: running from tests/, a src/ path used to silently
    skip (rc 0) and report false success, discarding the change."""
    _init_repo(tmp_path)
    subdir = tmp_path / "tests"
    ok, msg = WorktreeManager.apply_to_tree(str(subdir), _PATCH)
    assert ok, msg
    assert (tmp_path / "src" / "foo.py").read_text() == "value = 2\n"


def test_apply_from_repo_root_still_works(tmp_path):
    _init_repo(tmp_path)
    ok, msg = WorktreeManager.apply_to_tree(str(tmp_path), _PATCH)
    assert ok, msg
    assert (tmp_path / "src" / "foo.py").read_text() == "value = 2\n"


def test_apply_outside_git_repo_reports_failure(tmp_path):
    ok, msg = WorktreeManager.apply_to_tree(str(tmp_path), _PATCH)
    assert not ok
    assert "git repository" in msg
