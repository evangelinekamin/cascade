import subprocess
from pathlib import Path

from cascade.swarm.worktree import WorktreeManager


def _git(repo, *args):
    subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_snapshot_discards_untracked_runtime_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("VALUE = 1\n")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-qm", "baseline")

    manager = WorktreeManager(cwd=str(repo))
    prepared = manager.prepare("worker")
    root = Path(prepared.path)
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-313.pyc").write_bytes(b"generated")
    pytest_cache = root / ".pytest_cache"
    pytest_cache.mkdir()
    (pytest_cache / "README.md").write_text("generated")
    (root / "app.py").write_text("VALUE = 2\n")
    # Agent/workspace diff probes commonly mark untracked files intent-to-add;
    # the artifact filter must handle that state as well as plain `??` files.
    _git(root, "add", "-N", ".")

    snapshot = manager.capture_snapshot(prepared.path)

    assert snapshot.changed_files == ("app.py",)
    assert not cache.exists()
    assert not pytest_cache.exists()
    assert "pycache" not in manager.diff_patch(prepared.path)
    manager.cleanup()
