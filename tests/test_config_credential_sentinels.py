"""/login must not inline raw OAuth tokens.

apply_credential stores a read-through sentinel; get_provider_config resolves it
via the source CLI's detector in memory only. Existing inlined tokens and the
${VAR} indirection keep working. Tokens here are obvious placeholders, never
real-looking secrets.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from cascade.auth import DetectedCredential
from cascade.config import ConfigManager


def _cred(provider: str, token: str) -> DetectedCredential:
    return DetectedCredential(
        provider=provider, source="test", token=token, email="", plan="",
    )


def _manager(tmpdir: str) -> ConfigManager:
    return ConfigManager(str(Path(tmpdir) / "config.yaml"))


# provider -> (sentinel, detector attribute on cascade.auth, placeholder token)
_CASES = {
    "claude": ("@claude-cli", "cascade.auth.detect_claude", "fake-claude-token"),
    "openai": ("@codex-cli", "cascade.auth.detect_codex", "fake-openai-token"),
    "gemini": ("@gemini-cli", "cascade.auth.detect_gemini", "fake-gemini-token"),
}


def test_apply_credential_writes_sentinel_not_token():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        for provider, (sentinel, _detector, token) in _CASES.items():
            manager.apply_credential(provider, token, overwrite=True)
            stored = manager.data["providers"][provider]["api_key"]
            assert stored == sentinel, provider
            assert stored != token, provider


def test_get_provider_config_resolves_sentinel_read_through():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        for provider, (_sentinel, detector, token) in _CASES.items():
            manager.apply_credential(provider, token, overwrite=True)
            with patch(detector, return_value=_cred(provider, token)):
                config = manager.get_provider_config(provider)
            assert config is not None, provider
            assert config.api_key == token, provider


def test_resolution_never_mutates_stored_sentinel():
    """Read-through resolves in memory; the persisted value stays the sentinel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        manager.apply_credential("gemini", "fake-gemini-token", overwrite=True)
        with patch("cascade.auth.detect_gemini",
                   return_value=_cred("gemini", "fake-gemini-token")):
            manager.get_provider_config("gemini")
        assert manager.data["providers"]["gemini"]["api_key"] == "@gemini-cli"


def test_sentinel_resolves_empty_and_disables_when_cli_logged_out():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        manager.apply_credential("claude", "fake-claude-token", overwrite=True)
        with patch("cascade.auth.detect_claude", return_value=None):
            assert manager.get_provider_config("claude") is None


def test_token_is_never_persisted_to_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        manager.apply_credential("claude", "fake-claude-secret", overwrite=True)
        manager.save()
        raw = (Path(tmpdir) / "config.yaml").read_text(encoding="utf-8")
        assert "fake-claude-secret" not in raw
        assert "@claude-cli" in raw


def test_migration_legacy_inlined_token_still_resolves():
    """An existing config holding a raw inlined token keeps working."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        manager.data["providers"]["claude"]["enabled"] = True
        manager.data["providers"]["claude"]["api_key"] = "legacy-inlined-token"

        # No detection needed: a literal passes straight through.
        config = manager.get_provider_config("claude")
        assert config is not None
        assert config.api_key == "legacy-inlined-token"


def test_migration_startup_apply_keeps_existing_inlined_token():
    """Startup credential application (no overwrite) must not clobber a legacy token."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        manager.data["providers"]["claude"]["enabled"] = True
        manager.data["providers"]["claude"]["api_key"] = "legacy-inlined-token"

        manager.apply_credential("claude", "fresh-token")  # overwrite=False
        assert manager.data["providers"]["claude"]["api_key"] == "legacy-inlined-token"


def test_env_var_indirection_unchanged(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        monkeypatch.setenv("CASCADE_TEST_KEY", "value-from-env")
        manager.data["providers"]["openai"]["enabled"] = True
        manager.data["providers"]["openai"]["api_key"] = "${CASCADE_TEST_KEY}"
        config = manager.get_provider_config("openai")
        assert config is not None
        assert config.api_key == "value-from-env"


def test_openrouter_credential_stays_inline():
    """Providers without a source CLI (openrouter's pasted key) are not sentinel-ized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = _manager(tmpdir)
        manager.apply_credential("openrouter", "fake-openrouter-key", overwrite=True)
        assert manager.data["providers"]["openrouter"]["api_key"] == "fake-openrouter-key"
