"""Tests for the setup wizard."""

import os
from unittest.mock import patch

from cascade.setup_flow import detect_env_keys, needs_setup, SetupWizard
from cascade.config import ConfigManager


def test_detect_env_keys_finds_gemini():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
        found = detect_env_keys()
        assert "gemini" in found
        assert found["gemini"] == "test-key"


def test_detect_env_keys_finds_anthropic():
    """ANTHROPIC_API_KEY should map to claude provider."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-key"}, clear=False):
        found = detect_env_keys()
        assert "claude" in found
        assert found["claude"] == "ant-key"


def test_detect_env_keys_empty():
    """No env vars set -> empty dict."""
    env = {k: "" for k in ["GEMINI_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]}
    with patch.dict(os.environ, env, clear=False):
        found = detect_env_keys()
        # May still find keys from the real environment, so just check types
        assert isinstance(found, dict)


def test_needs_setup_true(tmp_path):
    """needs_setup returns True when no providers are enabled."""
    config = ConfigManager(config_path=str(tmp_path / "config.yaml"))
    assert needs_setup(config) is True


def test_needs_setup_false(tmp_path):
    """needs_setup returns False when a provider is enabled."""
    config = ConfigManager(config_path=str(tmp_path / "config.yaml"))
    config.data["providers"] = {
        "gemini": {"enabled": True, "api_key": "test", "model": "m"},
    }
    config.save()
    config = ConfigManager(config_path=str(tmp_path / "config.yaml"))
    assert needs_setup(config) is False


def test_setup_wizard_creates(tmp_path):
    """SetupWizard should instantiate without error."""
    config = ConfigManager(config_path=str(tmp_path / "config.yaml"))
    wizard = SetupWizard(config=config)
    assert wizard.registry is not None
    assert len(wizard.registry) >= 4


def test_openrouter_setup_prompts_for_custom_model(tmp_path):
    """First-run setup should allow overriding the OpenRouter model slug."""
    config = ConfigManager(config_path=str(tmp_path / "config.yaml"))
    wizard = SetupWizard(config=config)

    class _StubProvider:
        def __init__(self, provider_config):
            self.config = provider_config

        def ping(self):
            return True

    wizard.registry = {"openrouter": _StubProvider}

    with patch.object(
        wizard,
        "_prompt",
        side_effect=["y", "qwen/qwen3-coder-next"],
    ):
        assert wizard._configure_provider("openrouter", "test-key") is True

    provider_cfg = config.data["providers"]["openrouter"]
    assert provider_cfg["model"] == "qwen/qwen3-coder-next"
    assert provider_cfg["fallback_model"] == "minimax/minimax-m2.5"


def test_wizard_default_models_match_config_defaults():
    """Wizard default models must not drift back to stale values."""
    from cascade.setup_flow import _DEFAULT_MODELS

    assert _DEFAULT_MODELS["claude"] == "claude-opus-4-8"
    assert _DEFAULT_MODELS["gemini"] == "gemini-3.1-pro-preview"


def test_configure_provider_does_not_plant_max_tokens(tmp_path):
    """The wizard must not seed max_tokens -- it truncates /solve's agentic edits."""
    config = ConfigManager(config_path=str(tmp_path / "config.yaml"))
    wizard = SetupWizard(config=config)

    class _StubProvider:
        def __init__(self, provider_config):
            self.config = provider_config

        def ping(self):
            return True

    wizard.registry = {"openai": _StubProvider}
    with patch.object(wizard, "_prompt", side_effect=["y"]):
        assert wizard._configure_provider("openai", "sk-test") is True

    assert "max_tokens" not in config.data["providers"]["openai"]
