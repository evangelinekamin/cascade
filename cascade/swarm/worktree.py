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
        self._discard_untracked_generated_artifacts(worktree_path, baseline_ref)
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
        self._discard_untracked_generated_artifacts(worktree_path, baseline_ref)
        self._git(["add", "-N", "."], cwd=worktree_path, check=False)
        return self._git(["diff", "--binary", baseline_ref], cwd=worktree_path, check=False)

    def apply_patch(self, worktree_path: str, patch: str) -> bool:
        """Apply *patch* into a worktree. Returns True on success, False on conflict.

        Tries a plain apply first (clean, no side effects on conflict), then falls
        back to a 3-way merge -- which lands non-overlapping edits to an already-
        touched file (e.g. two subtasks each appending a different export to the
        same __init__.py, which a plain apply rejects because the context shifted).
        On a genuine overlapping conflict the touched files are restored to their
        pre-apply state so no conflict markers leak into the integration, and the
        subtask is reported as a conflict rather than aborting the whole fan-out.
        """
        if not patch.strip():
            return False

        # Plain apply first -- clean and atomic; on any conflict it changes nothing.
        # Stage the result so the index tracks accumulated merges: git apply --3way
        # refuses ("does not match index") if the working tree has drifted from the
        # index, which happens after a prior working-tree-only apply.
        try:
            self._git(
                ["apply", "--whitespace=nowarn", "-"],
                cwd=worktree_path,
                input_text=patch,
            )
            self._git(["add", "-A"], cwd=worktree_path, check=False)
            return True
        except Exception:
            pass

        # 3-way merge -- lands non-overlapping edits to an already-touched file
        # (two subtasks each appending a different export to the same __init__.py).
        # Snapshot the touched files so a genuine overlapping conflict rolls back
        # cleanly instead of leaving markers in the integration.
        root = Path(worktree_path)
        affected = self._patched_paths(patch)
        saved = {rel: self._read_if_exists(root / rel) for rel in affected}
        try:
            self._git(
                ["apply", "--3way", "--whitespace=nowarn", "-"],
                cwd=worktree_path,
                input_text=patch,
            )
            if not self._has_conflict_markers(root, affected):
                self._git(["add", "-A"], cwd=worktree_path, check=False)
                return True
        except Exception:
            pass

        # Overlapping conflict (or hard failure): undo just the files this patch
        # touched and resync the index, leaving prior merges intact.
        for rel, content in saved.items():
            target = root / rel
            if content is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(content)
        self._git(["add", "-A"], cwd=worktree_path, check=False)
        return False

    @staticmethod
    def apply_to_tree(root: str, patch: str) -> "tuple[bool, str]":
        """Apply *patch* onto the repo containing *root* (for /apply).

        The solve patch paths are repo-root-relative, so the apply MUST run
        from the repo top-level -- run from a subdirectory, git apply
        silently SKIPS paths outside it and still returns success, which
        would discard the verified changes. We resolve the top-level, apply
        there, and treat any 'Skipped' hunk as a failure so a partial/no-op
        apply is never reported as success. Plain ``git apply`` is atomic:
        on conflict the tree is left untouched. Returns ``(applied, msg)``.
        """
        if not patch.strip():
            return False, "nothing to apply"
        try:
            top = WorktreeManager._git(
                ["rev-parse", "--show-toplevel"], cwd=root,
            ).strip()
        except Exception:
            return False, "not inside a git repository"
        if not top:
            return False, "could not resolve the repository root"
        try:
            out = WorktreeManager._git(
                ["apply", "--whitespace=nowarn", "-"],
                cwd=top,
                input_text=patch,
            )
        except Exception as exc:
            return False, (
                f"could not apply cleanly ({exc}); the working tree may have "
                "changed since the solve. Review it manually."
            )
        if "Skipped" in out:
            # git apply skipped a path (rc 0) -- a partial/no-op apply. Report
            # failure so the caller does not discard the patch as 'applied'.
            return False, "some changes could not be applied; review manually"
        return True, "applied"

    @staticmethod
    def _patched_paths(patch: str) -> tuple[str, ...]:
        """Repo-relative paths a unified diff writes to (its ``+++ b/`` targets)."""
        paths = [
            line[len("+++ b/"):].strip()
            for line in patch.splitlines()
            if line.startswith("+++ b/")
        ]
        return tuple(dict.fromkeys(paths))

    @staticmethod
    def _read_if_exists(path: Path) -> "str | None":
        try:
            return path.read_text()
        except Exception:
            return None

    @staticmethod
    def _has_conflict_markers(root: Path, paths: tuple[str, ...]) -> bool:
        for rel in paths:
            content = WorktreeManager._read_if_exists(root / rel)
            if content and "<<<<<<<" in content:
                return True
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

    @staticmethod
    def _is_generated_artifact(path: str) -> bool:
        """Whether an untracked path is execution debris, never a deliverable."""
        parsed = Path(path)
        cache_dirs = {
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".nox",
        }
        return (
            any(part in cache_dirs for part in parsed.parts)
            or parsed.suffix in {".pyc", ".pyo"}
            or parsed.name in {".coverage", ".DS_Store"}
        )

    def _discard_untracked_generated_artifacts(
        self,
        worktree_path: str,
        baseline_ref: str,
    ) -> None:
        """Remove generated files absent from baseline, including intent-to-add."""
        baseline_raw = self._git(
            ["ls-tree", "-r", "--name-only", "-z", baseline_ref],
            cwd=worktree_path,
            check=False,
        )
        baseline_files = {entry for entry in baseline_raw.split("\0") if entry}
        status = self._git(
            ["status", "--short", "--untracked-files=all"],
            cwd=worktree_path,
            check=False,
        )
        root = Path(worktree_path)
        parents: set[Path] = set()
        for relative in self._parse_changed_files(status):
            if relative in baseline_files or not self._is_generated_artifact(relative):
                continue
            target = root / relative
            try:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
                parents.update(parent for parent in target.parents if parent != root)
            except OSError:
                continue
            # An earlier `git add -N` may have made this path intent-to-add.
            # Reset just that non-baseline path so it cannot survive in the diff.
            self._git(
                ["reset", "-q", baseline_ref, "--", relative],
                cwd=worktree_path,
                check=False,
            )
        for parent in sorted(parents, key=lambda path: len(path.parts), reverse=True):
            try:
                parent.rmdir()
            except OSError:
                pass

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
