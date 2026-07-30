"""/init also initializes a git repo so /solve can branch worktrees."""

import subprocess
import pytest

from cascade.agents.init import run_init, _ensure_git_repo


def _has_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_git(), reason="git not installed")


def test_ensure_git_repo_creates_repo_with_commit(tmp_path):
    (tmp_path / "index.js").write_text("console.log(1)")
    lines = _ensure_git_repo(tmp_path)
    assert (tmp_path / ".git").is_dir()
    assert any("initial commit" in ln for ln in lines)
    # HEAD exists (a worktree can branch from it).
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True,
    )
    assert head.returncode == 0 and head.stdout.strip()


def test_ensure_git_repo_is_idempotent(tmp_path):
    _ensure_git_repo(tmp_path)
    lines = _ensure_git_repo(tmp_path)
    assert any("already initialized" in ln for ln in lines)


def test_ensure_git_repo_empty_dir_gets_empty_commit(tmp_path):
    _ensure_git_repo(tmp_path)
    assert (tmp_path / ".git").is_dir()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True,
    )
    assert head.returncode == 0  # HEAD exists even with nothing to commit


def test_run_init_scaffolds_and_git_inits(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "x"}')
    summary = run_init(tmp_path, "web")
    assert (tmp_path / ".cascade").is_dir()
    assert (tmp_path / ".git").is_dir()
    assert "git repo" in summary


def test_run_init_can_skip_git(tmp_path):
    run_init(tmp_path, "web", init_git=False)
    assert (tmp_path / ".cascade").is_dir()
    assert not (tmp_path / ".git").exists()
