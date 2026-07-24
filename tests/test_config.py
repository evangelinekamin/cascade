"""Tests for configuration system."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from cascade.auth import DetectedCredential
from cascade.config import ConfigManager


def _fake_cred(provider: str, token: str) -> DetectedCredential:
    return DetectedCredential(
        provider=provider, source="test", token=token, email="", plan="",
    )


def test_config_creation():
    """Test config file creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))
        
        assert config_path.exists()
        assert "providers" in manager.data


def test_get_default_provider():
    """Test getting default provider."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))
        
        default = manager.get_default_provider()
        assert default == "gemini"


def test_mode_config_defaults_follow_builtin_mapping():
    """Mode config should default to the builtin provider mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        assert manager.get_mode_provider("design") == "gemini"
        assert manager.get_mode_provider("plan") == "claude"
        assert manager.get_mode_provider("build") == "openai"
        assert manager.get_mode_provider("test") == "openrouter"


def test_get_model_for_respects_mode_override():
    """Mode-level model override should beat the provider default model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["modes"]["design"]["provider"] = "openrouter"
        manager.data["modes"]["design"]["model"] = "kwaipilot/kat-coder-pro-v2"

        assert manager.get_model_for("openrouter", "design") == "kwaipilot/kat-coder-pro-v2"
        assert manager.get_model_for("openrouter", "test") == "qwen/qwen3.5-9b"


def test_openrouter_default_exposes_ultrafast_fast_model():
    """OpenRouter ships a fast_model so /ultrafast and /fast can reach mercury-2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        assert (
            manager.data["providers"]["openrouter"]["fast_model"]
            == "inception/mercury-2"
        )
        assert (
            manager.get_model_for("openrouter", "test", fast=True)
            == "inception/mercury-2"
        )


def test_apply_credential_openrouter_sets_ultrafast_fast_model():
    """Syncing an OpenRouter credential seeds the ultrafast fast_model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.apply_credential("openrouter", "sk-or-test-token")

        entry = manager.data["providers"]["openrouter"]
        assert entry["fast_model"] == "inception/mercury-2"


def test_get_available_modes_uses_configured_mode_providers():
    """Mode availability should follow configured mode providers, not hardcoded defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["modes"]["design"]["provider"] = "openrouter"
        manager.data["modes"]["plan"]["provider"] = "openrouter"

        available = manager.get_available_modes({"openrouter"})

        assert available == ("design", "plan", "test")


def test_env_var_resolution():
    """Test environment variable resolution."""
    import os
    os.environ["TEST_KEY"] = "test_value"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))
        
        resolved = manager._resolve_env_var("${TEST_KEY}")
        assert resolved == "test_value"


def test_non_env_var_passthrough():
    """Test that non-env-var values pass through."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        value = manager._resolve_env_var("plain_value")
        assert value == "plain_value"


def test_apply_credential_enables_provider():
    """apply_credential enables a provider and resolves its live token read-through."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        # gemini starts disabled in default config
        assert manager.get_provider_config("gemini") is None

        manager.apply_credential("gemini", "fake-gemini-token")
        # The live secret is never written to config; a sentinel is.
        assert manager.data["providers"]["gemini"]["api_key"] == "@gemini-cli"

        with patch("cascade.auth.detect_gemini",
                   return_value=_fake_cred("gemini", "fake-gemini-token")):
            config = manager.get_provider_config("gemini")
        assert config is not None
        assert config.api_key == "fake-gemini-token"
        # Model comes from the default config (already set before apply_credential)
        assert config.model == "gemini-3.1-pro-preview"


def test_apply_credential_does_not_overwrite_existing():
    """Test that apply_credential skips already-configured providers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        # Manually enable with a key
        manager.data["providers"]["gemini"]["enabled"] = True
        manager.data["providers"]["gemini"]["api_key"] = "my-real-key"

    manager.apply_credential("gemini", "ya29.should-be-ignored")
    config = manager.get_provider_config("gemini")
    assert config.api_key == "my-real-key"


def test_apply_credential_overwrite_updates_existing():
    """Overwriting a stale inlined token replaces it with the sentinel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["providers"]["gemini"]["enabled"] = True
        manager.data["providers"]["gemini"]["api_key"] = "old-inlined-token"

        manager.apply_credential("gemini", "fresh-gemini-token", overwrite=True)
        assert manager.data["providers"]["gemini"]["api_key"] == "@gemini-cli"

        with patch("cascade.auth.detect_gemini",
                   return_value=_fake_cred("gemini", "fresh-gemini-token")):
            config = manager.get_provider_config("gemini")
        assert config.api_key == "fresh-gemini-token"


def test_apply_credential_new_provider():
    """apply_credential works for a provider not in default config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.apply_credential("openai", "fake-openai-token")
        assert manager.data["providers"]["openai"]["api_key"] == "@codex-cli"

        with patch("cascade.auth.detect_codex",
                   return_value=_fake_cred("openai", "fake-openai-token")):
            config = manager.get_provider_config("openai")
        assert config is not None
        assert config.api_key == "fake-openai-token"


def test_get_provider_config_allows_keyless_when_requires_key_false():
    """A provider marked requires_key: false works with no API key (self-hosted)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["providers"]["local"] = {
            "enabled": True,
            "requires_key": False,
            "base_url": "http://example.local/v1",
            "model": "some/model",
        }

        config = manager.get_provider_config("local")
        assert config is not None
        assert config.api_key == ""
        assert config.base_url == "http://example.local/v1"
        assert config.model == "some/model"


def test_get_provider_config_still_requires_key_by_default():
    """Keyless is opt-in: without requires_key: false, a missing key still gates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["providers"]["custom"] = {
            "enabled": True,
            "model": "some/model",
        }

        assert manager.get_provider_config("custom") is None


def test_provider_config_context_window_defaults_to_none():
    """No explicit context_window means None: resolved via context.budget."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["providers"]["openai"]["enabled"] = True
        manager.data["providers"]["openai"]["api_key"] = "literal-key"
        config = manager.get_provider_config("openai")
        assert config.context_window is None


def test_provider_config_reads_explicit_context_window():
    """An explicit context_window in provider_data is threaded onto the config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["providers"]["custom"] = {
            "enabled": True,
            "requires_key": False,
            "base_url": "http://example.local/v1",
            "model": "some/model",
            "context_window": 8192,
        }
        config = manager.get_provider_config("custom")
        assert config.context_window == 8192


def test_default_template_gives_local_and_glm_small_windows():
    """Self-hosted local/glm providers ship with a 32K window in the template."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        assert manager.data["providers"]["local"]["context_window"] == 32768
        assert manager.data["providers"]["glm"]["context_window"] == 32768


def test_default_template_leaves_big_api_providers_at_default_window():
    """Large API providers do not pin a small window in the template."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        for name in ("claude", "gemini", "openai", "openrouter"):
            assert "context_window" not in manager.data["providers"][name]


def test_default_template_openrouter_pins_good_quant_hosts():
    """The openrouter template pins good-quant fast upstream hosts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        prefs = manager.data["providers"]["openrouter"]["provider_preferences"]
        assert prefs == {
            "order": ["Baidu", "Fireworks", "Alibaba"],
            "allow_fallbacks": True,
        }


def test_get_provider_config_reads_provider_preferences():
    """provider_preferences from provider_data is threaded onto the config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["providers"]["openrouter"]["enabled"] = True
        manager.data["providers"]["openrouter"]["api_key"] = "sk-test"

        config = manager.get_provider_config("openrouter")
        assert config is not None
        assert config.provider_preferences == {
            "order": ["Baidu", "Fireworks", "Alibaba"],
            "allow_fallbacks": True,
        }


def test_provider_config_defaults_provider_preferences_to_none():
    """A provider without provider_preferences leaves the field as None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["providers"]["openai"]["enabled"] = True
        manager.data["providers"]["openai"]["api_key"] = "literal-key"
        config = manager.get_provider_config("openai")
        assert config.provider_preferences is None


def test_get_bulk_model_defaults_to_frontier_model_not_fast_model():
    """/solve's bulk tier uses the provider model, never the /ultrafast fast_model.

    A fresh openrouter config ships fast_model=inception/mercury-2 for /ultrafast;
    that flaky speed model must not silently become /solve's builder.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        assert manager.data["providers"]["openrouter"]["fast_model"] == "inception/mercury-2"
        assert manager.get_bulk_model("openrouter") == "qwen/qwen3.5-9b"


def test_get_bulk_model_prefers_explicit_bulk_model():
    """An explicit bulk_model wins, giving /solve a cheap first tier below model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        manager.data["providers"]["openrouter"]["bulk_model"] = "deepseek/deepseek-v4-flash"
        assert manager.get_bulk_model("openrouter") == "deepseek/deepseek-v4-flash"


def test_get_escalation_target_reads_provider_escalate_to():
    """/solve's cross-provider escalation target comes from providers.<name>.escalate_to."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))

        # unset by default -> None (escalate within the same provider)
        assert manager.get_escalation_target("openrouter") is None
        # a provider name -> the cross-provider handoff target
        manager.data["providers"]["openrouter"]["escalate_to"] = "claude"
        assert manager.get_escalation_target("openrouter") == "claude"


def test_memory_config_defaults():
    """Memory config should expose safe defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))
        cfg = manager.get_memory_config()
        assert cfg["cross_model_memory"] == "summary"
        assert cfg["summary_turn_interval"] >= 1
        assert cfg["summary_max_chars"] >= 400


def test_memory_config_invalid_values_are_sanitized():
    """Invalid memory config values should fall back to sane values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        manager = ConfigManager(str(config_path))
        manager.data["memory"] = {
            "cross_model_memory": "totally-invalid",
            "summary_turn_interval": 0,
            "summary_max_chars": 1,
        }
        cfg = manager.get_memory_config()
        assert cfg["cross_model_memory"] == "summary"
        assert cfg["summary_turn_interval"] == 1
        assert cfg["summary_max_chars"] == 400
        assert "summary_provider" not in cfg


def test_orchestration_defaults_use_cerebras_through_openrouter():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(str(Path(tmpdir) / "config.yaml"))

        cfg = manager.get_orchestration_config()

        assert cfg["enabled"] is True
        assert cfg["modes"] == ["design", "plan", "build", "test"]
        assert cfg["router_provider"] == "openrouter"
        assert cfg["router_model"] == "openai/gpt-oss-120b"
        assert cfg["recon_provider"] == "openrouter"
        assert cfg["recon_model"] == "openai/gpt-oss-120b"
        assert cfg["fast_provider"] == "openrouter"
        assert cfg["fast_model"] == "inception/mercury-2"
        assert cfg["fast_provider_preferences"] == {
            "allow_fallbacks": True,
            "require_parameters": True,
        }
        assert cfg["provider_preferences"] == {
            "order": ["cerebras"],
            "allow_fallbacks": True,
            "require_parameters": True,
        }


def test_orchestration_config_sanitizes_modes_preferences_and_rounds():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConfigManager(str(Path(tmpdir) / "config.yaml"))
        manager.data["orchestration"] = {
            "modes": ["BUILD", "invalid"],
            "provider_preferences": "bad",
            "recon_max_rounds": 999,
        }

        cfg = manager.get_orchestration_config()

        assert cfg["modes"] == ["build"]
        assert cfg["provider_preferences"]["order"] == ["cerebras"]
        assert cfg["recon_max_rounds"] == 20
