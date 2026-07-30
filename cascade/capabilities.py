"""Cached runtime capability checks for Cascade and provider CLIs."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class CliCapability:
    name: str
    available: bool
    path: str = ""
    version: str = ""
    features: tuple[str, ...] = ()
    missing_features: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class DoctorReport:
    generated_at: str
    python: str
    python_ok: bool
    git: CliCapability
    provider_clis: tuple[CliCapability, ...]
    configured_providers: tuple[str, ...]
    permission_posture: str
    cache_hit: bool = False

    @property
    def ok(self) -> bool:
        return self.python_ok and self.git.available and bool(self.configured_providers)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


_CLI_FEATURES = {
    "claude": {
        "command": ("claude", "--help"),
        "markers": {
            "non-interactive": "--print",
            "structured-output": "stream-json",
            "permission-mode": "--permission-mode",
            "permission-bypass": "--dangerously-skip-permissions",
            "extra-workspace": "--add-dir",
        },
    },
    "gemini": {
        "command": ("gemini", "--help"),
        "markers": {
            "non-interactive": "--prompt",
            "structured-output": "stream-json",
            "approval-mode": "--approval-mode",
            "permission-bypass": "--yolo",
            "sandbox": "--sandbox",
        },
    },
    "codex": {
        "command": ("codex", "exec", "--help"),
        "markers": {
            "non-interactive": "codex exec",
            "structured-output": "--json",
            "sandbox": "--sandbox",
            "permission-bypass": "--dangerously-bypass-approvals-and-sandbox",
            "extra-workspace": "--add-dir",
            "resume": "resume",
        },
    },
}


def _default_runner(command: tuple[str, ...], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def _binary_fingerprint(which: Callable[[str], Optional[str]]) -> dict[str, str | int]:
    fingerprint: dict[str, str | int] = {
        "python": platform.python_version(),
        "cascade_schema": 2,
    }
    for name in ("git", *_CLI_FEATURES):
        path = which(name) or ""
        fingerprint[f"{name}_path"] = path
        try:
            stat = Path(path).stat() if path else None
        except OSError:
            stat = None
        fingerprint[f"{name}_mtime_ns"] = stat.st_mtime_ns if stat else 0
        fingerprint[f"{name}_size"] = stat.st_size if stat else 0
    return fingerprint


def _probe_cli(
    name: str,
    *,
    which: Callable[[str], Optional[str]],
    runner: Callable[[tuple[str, ...], float], subprocess.CompletedProcess],
) -> CliCapability:
    path = which(name)
    if not path:
        return CliCapability(name=name, available=False, error="not found on PATH")
    spec = _CLI_FEATURES[name]
    try:
        help_result = runner(spec["command"], 5.0)
        version_result = runner((name, "--version"), 3.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return CliCapability(name=name, available=False, path=path, error=str(exc))
    help_text = (help_result.stdout or "") + (help_result.stderr or "")
    version = ((version_result.stdout or "") + (version_result.stderr or "")).strip()
    found = tuple(
        feature
        for feature, marker in spec["markers"].items()
        if marker.lower() in help_text.lower()
    )
    missing = tuple(feature for feature in spec["markers"] if feature not in found)
    return CliCapability(
        name=name,
        available=help_result.returncode == 0,
        path=path,
        version=version.splitlines()[0][:160] if version else "",
        features=found,
        missing_features=missing,
        error="" if help_result.returncode == 0 else help_text.strip()[:240],
    )


def _probe_git(
    *,
    which: Callable[[str], Optional[str]],
    runner: Callable[[tuple[str, ...], float], subprocess.CompletedProcess],
) -> CliCapability:
    path = which("git")
    if not path:
        return CliCapability(name="git", available=False, error="not found on PATH")
    try:
        version_result = runner(("git", "--version"), 3.0)
        worktree_result = runner(("git", "worktree", "-h"), 3.0)
    except (OSError, subprocess.SubprocessError) as exc:
        return CliCapability(name="git", available=False, path=path, error=str(exc))
    worktree_help = (worktree_result.stdout or "") + (worktree_result.stderr or "")
    features = ("worktree",) if "worktree" in worktree_help.lower() else ()
    return CliCapability(
        name="git",
        available=version_result.returncode == 0,
        path=path,
        version=(version_result.stdout or version_result.stderr or "").strip(),
        features=features,
        missing_features=() if features else ("worktree",),
    )


def run_doctor(
    config=None,
    *,
    refresh: bool = False,
    cache_path: str | Path | None = None,
    which: Callable[[str], Optional[str]] = shutil.which,
    runner: Callable[[tuple[str, ...], float], subprocess.CompletedProcess] = _default_runner,
) -> DoctorReport:
    """Inspect the actual installed binaries, caching by binary fingerprint."""
    from .runtime import user_cache_path

    destination = Path(cache_path or user_cache_path("capabilities.json")).expanduser()
    fingerprint = _binary_fingerprint(which)
    cached: dict = {}
    if not refresh:
        try:
            cached = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
    if cached.get("fingerprint") == fingerprint and isinstance(cached.get("probes"), dict):
        probe_payload = cached["probes"]

        def _decode_capability(item: dict) -> CliCapability:
            return CliCapability(
                **{
                    **item,
                    "features": tuple(item.get("features") or ()),
                    "missing_features": tuple(item.get("missing_features") or ()),
                }
            )

        git = _decode_capability(probe_payload["git"])
        clis = tuple(
            _decode_capability(item) for item in probe_payload["provider_clis"]
        )
        cache_hit = True
    else:
        git = _probe_git(which=which, runner=runner)
        clis = tuple(
            _probe_cli(name, which=which, runner=runner) for name in _CLI_FEATURES
        )
        cache_hit = False
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "probes": {
                            "git": asdict(git),
                            "provider_clis": [asdict(item) for item in clis],
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    configured: tuple[str, ...] = ()
    posture = "unknown"
    if config is not None:
        try:
            configured = tuple(config.get_enabled_providers())
        except Exception:
            configured = ()
        try:
            posture = str(config.get_permissions_config().get("posture") or "auto")
        except Exception:
            posture = "unknown"
    return DoctorReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        python=sys.version.split()[0],
        python_ok=sys.version_info >= (3, 10),
        git=git,
        provider_clis=clis,
        configured_providers=configured,
        permission_posture=posture,
        cache_hit=cache_hit,
    )


def format_doctor(report: DoctorReport) -> str:
    """Human-readable doctor output shared by CLI and TUI."""
    lines = [
        f"Cascade doctor: {'ready' if report.ok else 'needs attention'}",
        f"Python {report.python}: {'ok' if report.python_ok else 'requires 3.10+'}",
        f"Git: {report.git.version or report.git.error}",
        "Providers configured: " + (", ".join(report.configured_providers) or "none"),
        f"Permission posture: {report.permission_posture} (popup-free)",
    ]
    for item in report.provider_clis:
        if not item.available:
            lines.append(f"{item.name}: unavailable ({item.error})")
            continue
        features = ", ".join(item.features) or "basic CLI"
        lines.append(f"{item.name}: {item.version or 'available'} [{features}]")
        if item.missing_features:
            lines.append("  unavailable flags: " + ", ".join(item.missing_features))
    if report.cache_hit:
        lines.append("Capability probe: cached (use --refresh after upgrading a CLI)")
    return "\n".join(lines)
