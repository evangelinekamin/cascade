"""Tests for the OpenAI provider."""

import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from cascade.providers.base import ProviderConfig
from cascade.providers.openai_provider import OpenAIProvider


def test_openai_accepts_provider_config():
    """OpenAIProvider should accept a ProviderConfig."""
    config = ProviderConfig(
        api_key="test-key",
        model="gpt-4o",
    )
    provider = OpenAIProvider(config)
    assert provider.config is config
    assert provider.config.api_key == "test-key"
    assert provider.config.model == "gpt-4o"


def test_openai_abc_compliance():
    """OpenAIProvider should implement all BaseProvider abstract methods."""
    config = ProviderConfig(api_key="test-key", model="test-model")
    provider = OpenAIProvider(config)
    assert hasattr(provider, "ask")
    assert hasattr(provider, "stream")
    assert hasattr(provider, "compare")
    assert callable(provider.ask)
    assert callable(provider.stream)
    assert callable(provider.compare)


def test_openai_default_base_url():
    """Should use OpenAI base URL by default."""
    config = ProviderConfig(api_key="test-key", model="gpt-4o")
    provider = OpenAIProvider(config)
    assert provider.base_url == "https://api.openai.com/v1"


def test_openai_custom_base_url():
    """Should accept custom base URL for Azure/proxies."""
    config = ProviderConfig(
        api_key="test-key",
        model="gpt-4",
        base_url="https://my-azure.openai.azure.com/v1",
    )
    provider = OpenAIProvider(config)
    assert provider.base_url == "https://my-azure.openai.azure.com/v1"


def test_openai_validation():
    """Should validate with valid config."""
    config = ProviderConfig(api_key="sk-test", model="gpt-4o")
    provider = OpenAIProvider(config)
    assert provider.validate() is True


def test_openai_validation_no_key():
    """Should fail validation without API key."""
    config = ProviderConfig(api_key="", model="gpt-4o")
    provider = OpenAIProvider(config)
    assert provider.validate() is False


def test_uses_cli_proxy_for_codex_oauth_when_binary_exists():
    """JWT OAuth token should route through codex CLI when available."""
    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex")
        )
    assert provider._use_cli_proxy is True


def test_does_not_use_cli_proxy_for_standard_api_key():
    """Regular OpenAI API keys should keep direct API path."""
    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="sk-test-key", model="gpt-4o")
        )
    assert provider._use_cli_proxy is False


def test_stream_cli_parses_agent_message_and_usage():
    """Codex JSONL output should yield assistant text and capture usage."""

    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = iter(
                [
                    '{"type":"thread.started","thread_id":"t"}\n',
                    '{"type":"turn.started"}\n',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"OK"}}\n',
                    '{"type":"turn.completed","usage":{"input_tokens":9,"output_tokens":2}}\n',
                ]
            )
            self.returncode = 0

        def wait(self):
            return 0

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        with patch.dict("os.environ", {"CASCADE_OPENAI_ACTIVITY": "0"}, clear=False):
            provider = OpenAIProvider(
                ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex")
            )

    with patch("cascade.providers._cli_proxy.subprocess.Popen", _FakePopen):
        chunks = list(provider.stream_single("Reply with OK"))

    assert chunks == ["OK"]
    assert provider.last_usage == (9, 2)


def test_oauth_token_without_codex_binary_returns_clear_error():
    """OAuth token without codex CLI should not fall back to raw API."""
    with patch("cascade.providers.openai_provider.shutil.which", return_value=None):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex")
        )

    chunks = list(provider.stream_single("hello"))
    assert chunks == ["Error: Codex OAuth token detected, but codex CLI is not in PATH."]


def test_cli_proxy_uses_focused_workspace_for_non_agentic_file_audit(tmp_path):
    """Non-editing Codex requests should run in a focused scratch workspace."""
    source_file = tmp_path / "frontend" / "src" / "lib" / "api.ts"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("export const value = 1;\n", encoding="utf-8")
    workspace_root = tmp_path / ".codex-cache"
    captured = {}

    def _fake_stream(cfg, handler, emit_activity):
        captured["cwd"] = cfg.cwd
        captured["cmd_args"] = list(cfg.cmd_args)
        captured["prompt"] = cfg.cmd_args[-1]
        captured["mirrored_file"] = Path(cfg.cwd, "frontend", "src", "lib", "api.ts").read_text(
            encoding="utf-8",
        )
        captured["workspace_note"] = Path(cfg.cwd, "CASCADE_WORKSPACE.md").read_text(
            encoding="utf-8",
        )
        handler.last_usage = (11, 3)
        yield "OK"

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex"),
        )

    with patch("cascade.providers.openai_provider.stream_cli_proxy", side_effect=_fake_stream):
        with patch.dict("os.environ", {"CASCADE_CODEX_WORKSPACE_ROOT": str(workspace_root)}, clear=False):
            with provider.working_directory(str(tmp_path)):
                chunks = list(provider.stream_single(
                    "Audit frontend/src/lib/api.ts. Do not edit anything yet.",
                ))

    assert chunks == ["OK"]
    assert captured["cwd"] != str(tmp_path)
    assert "--skip-git-repo-check" in captured["cmd_args"]
    assert "--sandbox" in captured["cmd_args"]
    assert "read-only" in captured["cmd_args"]
    assert "temporary focused workspace" in captured["prompt"]
    assert "frontend/src/lib/api.ts" in captured["prompt"]
    assert captured["mirrored_file"] == "export const value = 1;\n"
    assert "Mirrored files:" in captured["workspace_note"]
    assert str(workspace_root) in captured["cwd"]
    assert provider.last_usage == (11, 3)


def test_cli_proxy_keeps_repo_workspace_for_agentic_requests(tmp_path):
    """Implementation prompts should keep the full repo-backed workspace."""
    source_file = tmp_path / "frontend" / "src" / "lib" / "api.ts"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("export const value = 1;\n", encoding="utf-8")
    captured = {}

    def _fake_stream(cfg, handler, emit_activity):
        captured["cwd"] = cfg.cwd
        captured["cmd_args"] = list(cfg.cmd_args)
        captured["prompt"] = cfg.cmd_args[-1]
        yield "OK"

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex"),
        )

    with patch("cascade.providers.openai_provider.stream_cli_proxy", side_effect=_fake_stream):
        with provider.working_directory(str(tmp_path)):
            chunks = list(provider.stream_single(
                "Implement one safe cleanup in frontend/src/lib/api.ts and update the file directly.",
            ))

    assert chunks == ["OK"]
    assert captured["cwd"] == str(tmp_path)
    assert "--skip-git-repo-check" not in captured["cmd_args"]
    assert "--ephemeral" not in captured["cmd_args"]
    assert "--sandbox" in captured["cmd_args"]
    assert "workspace-write" in captured["cmd_args"]
    assert "temporary focused workspace" not in captured["prompt"]


def test_cli_proxy_reuses_codex_session_for_repeated_focused_workspace_requests(tmp_path):
    source_file = tmp_path / "frontend" / "src" / "lib" / "api.ts"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("export const value = 1;\n", encoding="utf-8")
    workspace_root = tmp_path / ".codex-cache"
    calls: list[dict] = []

    def _fake_stream(cfg, handler, emit_activity):
        calls.append({
            "cwd": cfg.cwd,
            "cmd_args": list(cfg.cmd_args),
            "emit_activity": emit_activity,
        })
        if len(calls) == 1:
            handler.thread_id = "thread-123"
            handler.last_usage = (10, 2)
        else:
            handler.last_usage = (12, 3)
        yield "OK"

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex"),
        )

    with patch("cascade.providers.openai_provider.stream_cli_proxy", side_effect=_fake_stream):
        with patch.dict("os.environ", {"CASCADE_CODEX_WORKSPACE_ROOT": str(workspace_root)}, clear=False):
            with provider.working_directory(str(tmp_path)):
                first = list(provider.stream_single(
                    "Audit frontend/src/lib/api.ts. Do not edit anything yet.",
                ))
                second = list(provider.stream_single(
                    "Audit frontend/src/lib/api.ts again and rank the risks.",
                ))

    assert first == ["OK"]
    assert second == ["OK"]
    assert calls[0]["cmd_args"][:4] == ["/usr/bin/codex", "exec", "--json", "--cd"]
    assert "--skip-git-repo-check" in calls[0]["cmd_args"]
    assert calls[1]["cmd_args"][:5] == ["/usr/bin/codex", "exec", "resume", "thread-123", "--json"]
    assert "--skip-git-repo-check" in calls[1]["cmd_args"]
    assert calls[0]["cwd"] == calls[1]["cwd"]
    assert "Previous conversation context:" not in calls[1]["cmd_args"][-1]
    assert "Current request:\nAudit frontend/src/lib/api.ts again and rank the risks." in calls[1]["cmd_args"][-1]
    assert provider.last_usage == (12, 3)


def test_cli_proxy_retries_fresh_exec_when_cached_session_fails(tmp_path):
    source_file = tmp_path / "frontend" / "src" / "lib" / "api.ts"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("export const value = 1;\n", encoding="utf-8")
    workspace_root = tmp_path / ".codex-cache"
    calls: list[list[str]] = []

    def _fake_stream(cfg, handler, emit_activity):
        calls.append(list(cfg.cmd_args))
        if len(calls) == 1:
            handler.thread_id = "thread-123"
            handler.last_usage = (10, 2)
            yield "OK"
            return
        if len(calls) == 2:
            raise RuntimeError("session not found")
        handler.thread_id = "thread-456"
        handler.last_usage = (11, 4)
        yield "OK"

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex"),
        )

    with patch("cascade.providers.openai_provider.stream_cli_proxy", side_effect=_fake_stream):
        with patch.dict("os.environ", {"CASCADE_CODEX_WORKSPACE_ROOT": str(workspace_root)}, clear=False):
            with provider.working_directory(str(tmp_path)):
                first = list(provider.stream_single(
                    "Audit frontend/src/lib/api.ts. Do not edit anything yet.",
                ))
                second = list(provider.stream_single(
                    "Audit frontend/src/lib/api.ts again and rank the risks.",
                ))

    assert first == ["OK"]
    assert second == ["OK"]
    assert calls[1][:5] == ["/usr/bin/codex", "exec", "resume", "thread-123", "--json"]
    assert calls[2][:4] == ["/usr/bin/codex", "exec", "--json", "--cd"]
    assert provider._codex_sessions
    assert next(iter(provider._codex_sessions.values())) == "thread-456"


def test_resume_prompt_keeps_full_replay_when_synthetic_context_present(tmp_path):
    source_file = tmp_path / "frontend" / "src" / "lib" / "api.ts"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("export const value = 1;\n", encoding="utf-8")
    workspace_root = tmp_path / ".codex-cache"
    calls: list[list[str]] = []

    def _fake_stream(cfg, handler, emit_activity):
        calls.append(list(cfg.cmd_args))
        handler.last_usage = (9, 2)
        yield "OK"

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex"),
        )

    provider._codex_sessions["focused:dummy:none"] = "thread-123"

    @contextmanager
    def _fake_workspace(_messages, _system):
        yield (
            "FULL PROMPT WITH REPLAY",
            str(workspace_root / "dummy"),
            "focused",
            "focused:dummy:none",
        )

    with patch.object(provider, "_cli_workspace", _fake_workspace):
        with patch("cascade.providers.openai_provider.stream_cli_proxy", side_effect=_fake_stream):
            messages = [
                {"role": "user", "content": "[Context from previous model interactions]\nsummary"},
                {"role": "assistant", "content": "Understood, I have the context from the previous interactions."},
                {"role": "user", "content": "Audit frontend/src/lib/api.ts again and rank the risks."},
            ]
            list(provider.stream(messages))

    assert calls[0][:5] == ["/usr/bin/codex", "exec", "resume", "thread-123", "--json"]
    assert calls[0][-1] == "FULL PROMPT WITH REPLAY"


def test_find_filename_match_uses_cached_index(tmp_path):
    source_file = tmp_path / "frontend" / "src" / "lib" / "api.ts"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("export const value = 1;\n", encoding="utf-8")

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex"),
        )

    with patch("cascade.providers.openai_provider.os.walk", wraps=os.walk) as mock_walk:
        first = provider._find_filename_match("api.ts", tmp_path)
        second = provider._find_filename_match("api.ts", tmp_path)

    assert first == source_file
    assert second == source_file
    assert mock_walk.call_count == 1


def test_cli_proxy_keeps_repo_workspace_for_repo_wide_audits_without_file_refs(tmp_path):
    """Repo-wide audits without concrete files should not lose broad workspace access."""
    captured = {}

    def _fake_stream(cfg, handler, emit_activity):
        captured["cwd"] = cfg.cwd
        captured["cmd_args"] = list(cfg.cmd_args)
        yield "OK"

    with patch("cascade.providers.openai_provider.shutil.which", return_value="/usr/bin/codex"):
        provider = OpenAIProvider(
            ProviderConfig(api_key="eyJ.a.b", model="gpt-5.3-codex"),
        )

    with patch("cascade.providers.openai_provider.stream_cli_proxy", side_effect=_fake_stream):
        with provider.working_directory(str(tmp_path)):
            chunks = list(provider.stream_single(
                "Give me a frontend audit of this repo and call out the riskiest areas.",
            ))

    assert chunks == ["OK"]
    assert captured["cwd"] == str(tmp_path)
    assert "--skip-git-repo-check" not in captured["cmd_args"]
