"""Helpers for worktree-backed coding competitions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tempfile


@dataclass(frozen=True)
class PreparedWorktree:
    """A per-provider git worktree prepared from the current repo state."""

    provider: str
    path: str


@dataclass(frozen=True)
class WorktreeSnapshot:
    """Captured git state from an isolated worktree after execution."""

    status: str = ""
    changed_files: tuple[str, ...] = ()
    diff_stat: str = ""
    diff_excerpt: str = ""


@dataclass
class WorktreeManager:
    """Create detached git worktrees that mirror the current working tree."""

    cwd: str | None = None
    diff_excerpt_chars: int = 6000
    repo_root: str = field(init=False)
    temp_root: str = field(init=False)
    _source_patch: str = field(init=False, repr=False)
    _untracked_files: tuple[str, ...] = field(init=False, repr=False)
    _baseline_refs: dict[str, str] = field(init=False, default_factory=dict, repr=False)
    _prepared: dict[str, PreparedWorktree] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        base_cwd = Path(self.cwd or os.getcwd()).resolve()
        self.repo_root = self._git(
            ["rev-parse", "--show-toplevel"],
            cwd=str(base_cwd),
        ).strip()
        cache_root = self._cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        self.temp_root = tempfile.mkdtemp(prefix="cascade-compete-", dir=str(cache_root))
        self._source_patch = self._git(["diff", "--binary", "HEAD"], cwd=self.repo_root)
        self._untracked_files = self._list_untracked_files()

    def prepare(self, provider: str) -> PreparedWorktree:
        """Create a detached worktree for *provider* and sync dirty repo state into it."""
        if provider in self._prepared:
            return self._prepared[provider]

        worktree_path = Path(self.temp_root, provider)
        self._git(["worktree", "add", "--detach", str(worktree_path), "HEAD"], cwd=self.repo_root)
        try:
            self._apply_source_state(worktree_path)
            self._baseline_refs[str(worktree_path)] = self._capture_baseline(worktree_path)
        except Exception:
            self.remove_path(str(worktree_path))
            raise

        prepared = PreparedWorktree(provider=provider, path=str(worktree_path))
        self._prepared[provider] = prepared
        return prepared

    def capture_snapshot(self, worktree_path: str) -> WorktreeSnapshot:
        """Collect git status and a clipped diff from the worktree."""
        baseline_ref = self._baseline_refs.get(worktree_path, "HEAD")
        self._git(["add", "-N", "."], cwd=worktree_path, check=False)
        status = self._git(["status", "--short"], cwd=worktree_path, check=False)
        diff_stat = self._git(["diff", "--stat", baseline_ref], cwd=worktree_path, check=False)
        diff_text = self._git(["diff", "--binary", baseline_ref], cwd=worktree_path, check=False)
        changed_files = tuple(self._parse_changed_files(status))
        return WorktreeSnapshot(
            status=status.strip(),
            changed_files=changed_files,
            diff_stat=diff_stat.strip(),
            diff_excerpt=self._clip_text(diff_text.strip()),
        )

    def diff_patch(self, worktree_path: str) -> str:
        """Full (unclipped) binary diff of a worktree vs its baseline.

        Unlike ``capture_snapshot``'s clipped excerpt, this is a re-appliable patch
        -- the fan-out integrator collects one per subtask worktree and replays the
        passing ones onto a fresh integration worktree.
        """
        baseline_ref = self._baseline_refs.get(worktree_path, "HEAD")
        self._git(["add", "-N", "."], cwd=worktree_path, check=False)
        return self._git(["diff", "--binary", baseline_ref], cwd=worktree_path, check=False)

    def apply_patch(self, worktree_path: str, patch: str) -> bool:
        """Apply *patch* into a worktree. Returns True on success, False on conflict.

        Used to merge one subtask's verified diff into the integration worktree;
        a clean False (rather than a raise) lets the integrator skip a conflicting
        subtask and report it instead of aborting the whole fan-out.
        """
        if not patch.strip():
            return True
        try:
            self._git(
                ["apply", "--whitespace=nowarn", "-"],
                cwd=worktree_path,
                input_text=patch,
            )
            return True
        except Exception:
            return False

    def cleanup(self, keep_provider: str = "") -> None:
        """Remove temporary worktrees, optionally keeping the winner's workspace."""
        for provider, prepared in list(self._prepared.items()):
            if keep_provider and provider == keep_provider:
                continue
            self.remove_path(prepared.path)
            self._prepared.pop(provider, None)

        if not keep_provider:
            shutil.rmtree(self.temp_root, ignore_errors=True)

    def remove_path(self, path: str) -> None:
        """Force-remove a managed worktree path."""
        self._git(["worktree", "remove", "--force", path], cwd=self.repo_root, check=False)
        shutil.rmtree(path, ignore_errors=True)
        self._baseline_refs.pop(path, None)

    def _apply_source_state(self, worktree_path: Path) -> None:
        if self._source_patch.strip():
            self._git(
                ["apply", "--whitespace=nowarn", "-"],
                cwd=str(worktree_path),
                input_text=self._source_patch,
            )

        repo_root = Path(self.repo_root)
        for rel_path in self._untracked_files:
            source = repo_root / rel_path
            target = worktree_path / rel_path
            if not source.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def _list_untracked_files(self) -> tuple[str, ...]:
        raw = self._git(
            ["ls-files", "--others", "--exclude-standard", "-z"],
            cwd=self.repo_root,
            check=False,
        )
        if not raw:
            return ()
        return tuple(entry for entry in raw.split("\0") if entry)

    @staticmethod
    def _parse_changed_files(status_text: str) -> list[str]:
        files: list[str] = []
        for line in status_text.splitlines():
            line = line.rstrip()
            if len(line) < 4:
                continue
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if path:
                files.append(path)
        return files

    def _clip_text(self, text: str) -> str:
        if len(text) <= self.diff_excerpt_chars:
            return text
        head = self.diff_excerpt_chars // 2
        tail = self.diff_excerpt_chars - head - len("\n...\n")
        return f"{text[:head].rstrip()}\n...\n{text[-tail:].lstrip()}"

    @staticmethod
    def _cache_root() -> Path:
        override = os.environ.get("CASCADE_WORKTREE_ROOT", "").strip()
        if override:
            return Path(override).expanduser()
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        if xdg_cache_home:
            return Path(xdg_cache_home).expanduser() / "cascade" / "worktrees"
        return Path.home() / ".cache" / "cascade" / "worktrees"

    def _capture_baseline(self, worktree_path: Path) -> str:
        """Commit mirrored source-tree dirt into the isolated worktree baseline."""
        path_str = str(worktree_path)
        self._git(["add", "-A"], cwd=path_str, check=False)
        status = self._git(["status", "--short"], cwd=path_str, check=False).strip()
        if not status:
            return self._git(["rev-parse", "HEAD"], cwd=path_str).strip()

        self._git(
            [
                "-c", "user.name=Cascade",
                "-c", "user.email=cascade@local",
                "commit",
                "--no-gpg-sign",
                "-m", "cascade compete baseline",
            ],
            cwd=path_str,
        )
        return self._git(["rev-parse", "HEAD"], cwd=path_str).strip()

    @staticmethod
    def _git(
        args: list[str],
        cwd: str,
        check: bool = True,
        input_text: str | None = None,
    ) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            input=input_text,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
            raise RuntimeError(detail)
        output = result.stdout
        if result.stderr and not output:
            output = result.stderr
        return output
