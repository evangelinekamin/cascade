"""Seeded property checks for boundaries where one missed spelling is costly."""

import random
import string

import pytest

from cascade.swarm.auto import WorkflowKind, _parse_route_payload
from cascade.swarm.workspace import WorkspaceTools
from cascade.tools.permissions import PermissionEngine
from cascade.tools.schema import callable_to_tool_def
from cascade.web.url_safety import normalize_url, url_safety_error


def _command_tool():
    return callable_to_tool_def(
        "run_command",
        lambda command="": command,
        read_only=False,
    )


def test_seeded_root_deletion_variants_always_hit_the_circuit_breaker(tmp_path):
    rng = random.Random(712)
    engine = PermissionEngine(posture="yolo", workspace_root=str(tmp_path))
    tool = _command_tool()
    targets = ("/", "/*", "/[!.]*", "~", "~/*", "$HOME/{*,.*}", "${HOME}/*")
    flags = ("-rf", "-fr", "-r -f", "--recursive --force")
    wrappers = ("", "sudo ", "doas ", "env SAFE=1 ")
    for _ in range(500):
        command = f"{rng.choice(wrappers)}rm {rng.choice(flags)} {rng.choice(targets)}"
        verdict = engine.evaluate(tool, "run_command", {"command": command})
        assert verdict.decision == "deny", command
        assert verdict.rule == "circuit-breaker", command


def test_seeded_non_command_mentions_do_not_become_false_positive_deletions(tmp_path):
    rng = random.Random(913)
    engine = PermissionEngine(posture="yolo", workspace_root=str(tmp_path))
    tool = _command_tool()
    for _ in range(300):
        padding = "".join(rng.choice(string.ascii_lowercase) for _ in range(12))
        command = f"echo {padding} rm -rf /"
        assert engine.evaluate(tool, "run_command", {"command": command}).decision == "allow"


def test_seeded_workspace_prefix_traps_never_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    tools = WorkspaceTools(str(root))
    rng = random.Random(412)
    for _ in range(300):
        sibling = f"repo{rng.choice(string.ascii_letters)}"
        outside = root.parent / sibling / "file.txt"
        with pytest.raises(ValueError, match="escapes workspace"):
            tools._resolve(str(outside))
        with pytest.raises(ValueError, match="escapes workspace"):
            tools._resolve("../" + sibling + "/file.txt")


def test_seeded_malformed_routes_and_urls_fail_predictably():
    rng = random.Random(117)
    alphabet = string.ascii_letters + string.digits + ":/?#[]{}% \x00"
    for _ in range(500):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 80)))
        normalized = normalize_url(value)
        assert isinstance(normalized, str)
        # Literal/private hosts exercise SSRF parsing without DNS/network calls.
        url = f"https://127.{rng.randrange(256)}.{rng.randrange(256)}.{rng.randrange(256)}/"
        assert url_safety_error(url) is not None
        try:
            workflow, _reason, confidence, tier = _parse_route_payload(value)
        except (ValueError, TypeError):
            continue
        assert workflow in WorkflowKind
        assert 0.0 <= confidence <= 1.0
        assert tier in {"fast", "bulk", "frontier"}
