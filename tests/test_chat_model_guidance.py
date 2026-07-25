"""Per-model steering reaches the chat tool loop, not just isolated solves.

worker_guidance_for() counters a model's known failure modes (e.g. DeepSeek
rewriting whole files instead of surgical edits). It was applied only in the
verified-solve worker; _build_system_prompt now also injects it for the chat
loop, since that is where the model calls the same edit tools -- but only for
tool-capable providers (the guidance is about tool behavior).
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from cascade.cli import CascadeCore
from cascade.config import ConfigManager
from cascade.screens.main import MainScreen

_GUIDANCE_MARKER = "replace_in_file"  # a distinctive phrase from _DEEPSEEK_GUIDANCE


_CHAT_TOOLS = ["read_file", "write_file", "list_files", "append_file"]


def _cli_app(providers):
    tmp = tempfile.mkdtemp()
    manager = ConfigManager(str(Path(tmp) / "config.yaml"))
    return SimpleNamespace(
        prompt_pipeline=CascadeCore._build_prompt_pipeline(
            SimpleNamespace(config=manager, project=SimpleNamespace(found=False))
        ),
        config=manager,
        context_builder=SimpleNamespace(source_count=0),
        providers=providers,
        tool_registry=SimpleNamespace(tool_names=lambda: list(_CHAT_TOOLS)),
    )


def _provider(model, **flags):
    return SimpleNamespace(config=SimpleNamespace(model=model), **flags)


def test_guidance_drops_tools_absent_from_the_active_toolset():
    from cascade.swarm.solve import worker_guidance_for

    chat_tools = {"read_file", "write_file", "list_files", "append_file"}
    solve_tools = chat_tools | {"replace_in_file", "run_command"}
    # Chat lacks replace_in_file, so the line steering toward it is dropped...
    assert "replace_in_file" not in worker_guidance_for("deepseek/x", available_tools=chat_tools)
    # ...but the solve worktree has it, and the unfiltered form keeps everything.
    assert "replace_in_file" in worker_guidance_for("deepseek/x", available_tools=solve_tools)
    assert "replace_in_file" in worker_guidance_for("deepseek/x")
    # A universal line (names no tool) always survives the filter.
    assert "narrate" in worker_guidance_for("deepseek/x", available_tools=chat_tools).lower()


def test_deepseek_chat_gets_guidance_but_not_the_missing_tool():
    cli_app = _cli_app({"openrouter": _provider("deepseek/deepseek-v4-flash")})
    screen = MainScreen(active_provider="openrouter", mode="build")
    system = screen._build_system_prompt(cli_app, "add a helper", "openrouter")
    # Guidance is present (the universal steering) but not the chat-absent tool.
    assert "narrate" in system.lower()
    assert _GUIDANCE_MARKER not in system  # replace_in_file filtered out for chat


def test_non_deepseek_chat_gets_no_guidance():
    cli_app = _cli_app({"openai": _provider("gpt-5.6-terra")})
    screen = MainScreen(active_provider="openai", mode="build")
    system = screen._build_system_prompt(cli_app, "add a helper", "openai") or ""
    assert _GUIDANCE_MARKER not in system


def test_cli_proxy_provider_gets_no_guidance():
    # A CLI-proxy provider runs its own tool set, so our steering does not apply.
    cli_app = _cli_app({"claude": _provider("deepseek/deepseek-v4-flash", _use_cli_proxy=True)})
    screen = MainScreen(active_provider="claude", mode="build")
    system = screen._build_system_prompt(cli_app, "add a helper", "claude") or ""
    assert _GUIDANCE_MARKER not in system


def test_missing_providers_attr_does_not_crash():
    # The design-gating fixture builds a cli_app without providers; the guidance
    # step must degrade to a no-op rather than raise.
    tmp = tempfile.mkdtemp()
    manager = ConfigManager(str(Path(tmp) / "config.yaml"))
    cli_app = SimpleNamespace(
        prompt_pipeline=CascadeCore._build_prompt_pipeline(
            SimpleNamespace(config=manager, project=SimpleNamespace(found=False))
        ),
        config=manager,
        context_builder=SimpleNamespace(source_count=0),
    )
    screen = MainScreen(active_provider="openrouter", mode="build")
    # Should not raise despite no `providers` attribute.
    screen._build_system_prompt(cli_app, "hi", "openrouter")
