import subprocess

from cascade.capabilities import run_doctor


class _Config:
    def get_enabled_providers(self):
        return ["claude", "codex"]

    def get_permissions_config(self):
        return {"posture": "auto"}


def _which(name):
    return f"/fake/{name}" if name != "gemini" else None


def _runner(command, _timeout):
    text = {
        ("git", "--version"): "git version 2.50",
        ("git", "worktree", "-h"): "usage: git worktree add",
        ("claude", "--help"): "--print --output-format stream-json "
        "--permission-mode --dangerously-skip-permissions --add-dir",
        ("claude", "--version"): "claude 2.1",
        ("codex", "exec", "--help"): "codex exec --json --sandbox "
        "--dangerously-bypass-approvals-and-sandbox --add-dir resume",
        ("codex", "--version"): "codex 1.2",
    }.get(tuple(command), "")
    return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")


def test_doctor_probes_and_caches_binary_capabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cascade.capabilities._binary_fingerprint",
        lambda _which: {"stable": True},
    )
    cache = tmp_path / "capabilities.json"
    first = run_doctor(
        _Config(), cache_path=cache, which=_which, runner=_runner
    )
    second = run_doctor(
        _Config(),
        cache_path=cache,
        which=_which,
        runner=lambda *_args: (_ for _ in ()).throw(AssertionError("cache missed")),
    )

    assert first.ok and not first.cache_hit
    assert second.cache_hit
    assert second.permission_posture == "auto"
    assert "permission-bypass" in next(
        item for item in second.provider_clis if item.name == "codex"
    ).features
    gemini = next(item for item in second.provider_clis if item.name == "gemini")
    assert not gemini.available
