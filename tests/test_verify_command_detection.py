"""/solve resolves an ecosystem-appropriate verify command (regression).

A Node project used to inherit the global python `pytest` default and die with
'python: not found'. Detection now picks a command matching the project's
manifests, and honors a global command only when it's the same ecosystem.
"""

from pathlib import Path
from types import SimpleNamespace

from cascade.swarm.solve import (
    DEFAULT_TEST_CMD,
    _detect_test_command,
    _project_verify_test,
    _test_command,
)


def _app(test_cmd=""):
    return SimpleNamespace(
        config=SimpleNamespace(
            data={"workflows": {"verify": {"test": test_cmd}}} if test_cmd else {}
        )
    )


def test_detect_node_with_test_script(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest run"}}')
    assert _detect_test_command(tmp_path) == "npm test"


def test_detect_node_greenfield_ts_uses_typecheck(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "x"}')  # no test script
    (tmp_path / "tsconfig.json").write_text("{}")
    assert _detect_test_command(tmp_path) == "npx tsc --noEmit"


def test_detect_rust_and_go(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]")
    assert _detect_test_command(tmp_path) == "cargo test"
    (tmp_path / "Cargo.toml").unlink()
    (tmp_path / "go.mod").write_text("module x")
    assert _detect_test_command(tmp_path) == "go test ./..."


def test_detect_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]")
    assert _detect_test_command(tmp_path) == "python -m pytest -x -q"


def test_node_project_ignores_global_python_command(tmp_path):
    # The exact bug from the trace: global config is python, project is Node.
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest"}}')
    cmd = _test_command(_app("python -m pytest -x -q"), root=tmp_path)
    assert cmd == "npm test"


def test_python_project_keeps_custom_global_runner(tmp_path):
    # Same ecosystem -> respect the user's custom runner.
    (tmp_path / "pyproject.toml").write_text("[project]")
    cmd = _test_command(_app("uv run pytest"), root=tmp_path)
    assert cmd == "uv run pytest"


def test_project_cascade_yaml_wins(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "x"}}')
    cascade_dir = tmp_path / ".cascade"
    cascade_dir.mkdir()
    (cascade_dir / "agents.yaml").write_text(
        "workflows:\n  verify:\n    test: npm run ci\n"
    )
    # Explicit .cascade/agents.yaml beats detection (the file /init writes).
    assert _project_verify_test(str(tmp_path)) == "npm run ci"
    assert _test_command(_app("python -m pytest"), root=tmp_path) == "npm run ci"


def test_unknown_project_falls_back_to_default(tmp_path):
    assert _detect_test_command(tmp_path) is None
    assert _test_command(_app(), root=tmp_path) == DEFAULT_TEST_CMD
