"""End-to-end evaluation against real local repository fixtures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml


@dataclass(frozen=True)
class EvalTask:
    id: str
    prompt: str
    fixture: str
    verify: tuple[str, ...]
    expected_files: tuple[str, ...] = ()
    protected_files: tuple[str, ...] = ()
    timeout: float = 900.0


@dataclass(frozen=True)
class EvalResult:
    id: str
    passed: bool
    outcome: str
    workflow: str
    provider: str
    model: str
    duration_seconds: float
    input_tokens: int
    output_tokens: int
    cost: float
    tool_metrics_available: bool
    tool_calls: int | None
    tool_errors: int | None
    duplicate_reads: int | None
    changed_files: tuple[str, ...]
    worktree_path: str
    verification: tuple[dict, ...]
    missing_files: tuple[str, ...] = ()
    modified_protected_files: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class EvalReport:
    generated_at: str
    manifest: str
    provider: str
    mode: str
    results: tuple[EvalResult, ...]

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "manifest": self.manifest,
            "provider": self.provider,
            "mode": self.mode,
            "passed": self.passed,
            "total": self.total,
            "pass_rate": self.pass_rate,
            "results": [asdict(result) for result in self.results],
        }


def load_eval_manifest(path: str | Path) -> tuple[EvalTask, ...]:
    """Load a strict YAML/JSON task manifest, resolving fixtures relative to it."""
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read evaluation manifest: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("evaluation manifest must contain a tasks list")
    tasks = []
    seen = set()
    for index, raw in enumerate(payload["tasks"], 1):
        if not isinstance(raw, dict):
            raise ValueError(f"task {index} must be an object")
        task_id = str(raw.get("id") or f"task-{index}").strip()
        prompt = str(raw.get("prompt") or "").strip()
        fixture_value = str(raw.get("fixture") or "").strip()
        verify_value = raw.get("verify") or ()
        expected_value = raw.get("expected_files") or ()
        protected_value = raw.get("protected_files")
        if not task_id or task_id in seen:
            raise ValueError(f"task {index} has a duplicate/empty id")
        if not prompt or not fixture_value:
            raise ValueError(f"task {task_id} requires prompt and fixture")
        if isinstance(verify_value, str):
            verify_value = (verify_value,)
        if not isinstance(verify_value, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in verify_value
        ):
            raise ValueError(f"task {task_id} verify must be a command list")
        if isinstance(expected_value, str):
            expected_value = (expected_value,)
        fixture = Path(fixture_value).expanduser()
        if not fixture.is_absolute():
            fixture = manifest_path.parent / fixture
        if not fixture.is_dir():
            raise ValueError(f"task {task_id} fixture is not a directory: {fixture}")
        if protected_value is None:
            protected_value = tuple(
                str(candidate.relative_to(fixture))
                for candidate in sorted(fixture.rglob("*"))
                if candidate.is_file()
                and not any(
                    part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
                    for part in candidate.relative_to(fixture).parts
                )
                and candidate.suffix not in {".pyc", ".pyo"}
                and (
                    "tests" in candidate.relative_to(fixture).parts
                    or candidate.name.startswith("test_")
                    or candidate.name in {"conftest.py", "pytest.ini"}
                )
            )
        elif isinstance(protected_value, str):
            protected_value = (protected_value,)
        if not isinstance(protected_value, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in protected_value
        ):
            raise ValueError(f"task {task_id} protected_files must be a path list")
        normalized_protected = []
        for item in protected_value:
            relative = Path(item)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"task {task_id} protected file must stay inside its fixture: {item}"
                )
            candidate = fixture / relative
            if not candidate.is_file():
                raise ValueError(
                    f"task {task_id} protected file does not exist: {candidate}"
                )
            normalized_protected.append(str(relative))
        seen.add(task_id)
        tasks.append(
            EvalTask(
                id=task_id,
                prompt=prompt,
                fixture=str(fixture.resolve()),
                verify=tuple(item.strip() for item in verify_value),
                expected_files=tuple(str(item) for item in expected_value),
                protected_files=tuple(normalized_protected),
                timeout=max(float(raw.get("timeout", 900.0)), 1.0),
            )
        )
    if not tasks:
        raise ValueError("evaluation manifest has no tasks")
    return tuple(tasks)


def select_eval_tasks(
    tasks: tuple[EvalTask, ...], task_ids: tuple[str, ...]
) -> tuple[EvalTask, ...]:
    """Select requested task IDs while preserving manifest order."""
    if not task_ids:
        return tasks
    requested = set(task_ids)
    known = {task.id for task in tasks}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown evaluation task(s): {', '.join(unknown)}")
    return tuple(task for task in tasks if task.id in requested)


def _prepare_fixture(task: EvalTask, root: Path) -> Path:
    destination = root / task.id
    shutil.copytree(
        task.fixture,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"
        ),
    )
    if not (destination / ".git").exists():
        commands = (
            ("git", "init", "-q"),
            ("git", "config", "user.email", "cascade-eval@example.invalid"),
            ("git", "config", "user.name", "Cascade Eval"),
            ("git", "add", "-A"),
            ("git", "commit", "--allow-empty", "-qm", "evaluation fixture"),
        )
        for command in commands:
            result = subprocess.run(
                command, cwd=destination, capture_output=True, text=True, timeout=30
            )
            if result.returncode:
                raise RuntimeError(
                    f"could not prepare {task.id}: "
                    f"{(result.stderr or result.stdout).strip()}"
                )
    return destination


def _decode_json_output(output: str) -> dict:
    """Decode a JSON receipt even if a provider printed harmless setup chatter."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        start = output.find("{")
        end = output.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Cascade produced no JSON run receipt")
        payload = json.loads(output[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Cascade run receipt was not an object")
    return payload


def default_eval_executor(
    task: EvalTask,
    repo: Path,
    provider: str,
    mode: str,
) -> dict:
    """Invoke Cascade in a subprocess so each task has clean process state."""
    command = [
        sys.executable,
        "-m",
        "cascade.cli",
        "run",
        task.prompt,
        "--mode",
        mode,
        "--json",
    ]
    if provider:
        command.extend(("--provider", provider))
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        package_root
        if not existing_pythonpath
        else package_root + os.pathsep + existing_pythonpath
    )
    result = subprocess.run(
        command,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=task.timeout,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return _decode_json_output(result.stdout)


def _run_verification(root: Path, commands: tuple[str, ...], timeout: float) -> tuple[dict, ...]:
    results = []
    for command in commands:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            results.append(
                {
                    "command": command,
                    "passed": completed.returncode == 0,
                    "returncode": completed.returncode,
                    "duration_seconds": time.perf_counter() - started,
                    "output": ((completed.stdout or "") + (completed.stderr or ""))[-4000:],
                }
            )
        except subprocess.TimeoutExpired as exc:
            output = ((exc.stdout or "") + (exc.stderr or "")) if isinstance(
                exc.stdout, str
            ) else ""
            results.append(
                {
                    "command": command,
                    "passed": False,
                    "returncode": -1,
                    "duration_seconds": time.perf_counter() - started,
                    "output": (output + "\nverification timed out")[-4000:],
                }
            )
            break
    return tuple(results)


def run_evaluation(
    tasks: tuple[EvalTask, ...],
    *,
    manifest: str = "",
    provider: str = "",
    mode: str = "build",
    keep: bool = False,
    executor: Callable[[EvalTask, Path, str, str], dict] = default_eval_executor,
    temp_parent: Optional[str | Path] = None,
) -> EvalReport:
    """Run isolated tasks and independently verify the resulting worktrees."""
    temp_root = Path(
        tempfile.mkdtemp(prefix="cascade-eval-", dir=str(temp_parent) if temp_parent else None)
    )
    results = []
    try:
        for task in tasks:
            repo = _prepare_fixture(task, temp_root)
            protected_snapshot = {
                path: (repo / path).read_bytes() for path in task.protected_files
            }
            started = time.perf_counter()
            try:
                receipt = executor(task, repo, provider, mode)
                reported_root = Path(str(receipt.get("worktree_path") or repo))
                verify_root = reported_root if reported_root.is_dir() else repo
                verification = _run_verification(verify_root, task.verify, task.timeout)
                missing = tuple(
                    path for path in task.expected_files if not (verify_root / path).exists()
                )
                modified_protected = tuple(
                    path
                    for path, original in protected_snapshot.items()
                    if not (verify_root / path).is_file()
                    or (verify_root / path).read_bytes() != original
                )
                passed = (
                    str(receipt.get("outcome")) == "succeeded"
                    and all(item["passed"] for item in verification)
                    and not missing
                    and not modified_protected
                )
                outcome = str(receipt.get("outcome") or "unknown")
                run_error = str(receipt.get("error") or "")
                if outcome != "succeeded" and not run_error:
                    run_error = str(receipt.get("text") or "")[-2000:]
                if modified_protected and not run_error:
                    run_error = (
                        "protected verification files changed: "
                        + ", ".join(modified_protected)
                    )
                results.append(
                    EvalResult(
                        id=task.id,
                        passed=passed,
                        outcome=outcome,
                        workflow=str(receipt.get("workflow") or ""),
                        provider=str(receipt.get("provider") or provider),
                        model=str(receipt.get("model") or ""),
                        duration_seconds=float(
                            receipt.get("duration_seconds") or time.perf_counter() - started
                        ),
                        input_tokens=int(receipt.get("input_tokens") or 0),
                        output_tokens=int(receipt.get("output_tokens") or 0),
                        cost=float(receipt.get("cost") or 0.0),
                        tool_metrics_available=bool(
                            receipt.get("tool_metrics_available", False)
                        ),
                        tool_calls=(
                            int(receipt["tool_calls"])
                            if receipt.get("tool_calls") is not None
                            else None
                        ),
                        tool_errors=(
                            int(receipt["tool_errors"])
                            if receipt.get("tool_errors") is not None
                            else None
                        ),
                        duplicate_reads=(
                            int(receipt["duplicate_reads"])
                            if receipt.get("duplicate_reads") is not None
                            else None
                        ),
                        changed_files=tuple(receipt.get("changed_files") or ()),
                        worktree_path=str(verify_root) if keep else "",
                        verification=verification,
                        missing_files=missing,
                        modified_protected_files=modified_protected,
                        error=run_error,
                    )
                )
            except Exception as exc:
                results.append(
                    EvalResult(
                        id=task.id,
                        passed=False,
                        outcome="failed",
                        workflow="",
                        provider=provider,
                        model="",
                        duration_seconds=time.perf_counter() - started,
                        input_tokens=0,
                        output_tokens=0,
                        cost=0.0,
                        tool_metrics_available=False,
                        tool_calls=None,
                        tool_errors=None,
                        duplicate_reads=None,
                        changed_files=(),
                        worktree_path=str(repo) if keep else "",
                        verification=(),
                        error=str(exc),
                    )
                )
        return EvalReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            manifest=manifest,
            provider=provider,
            mode=mode,
            results=tuple(results),
        )
    finally:
        if not keep:
            shutil.rmtree(temp_root, ignore_errors=True)
