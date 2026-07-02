"""Tests for the keyless OpenAI-compatible provider.

One generic provider serves any keyless OpenAI-compatible ``/v1`` endpoint and
is registered under two names: ``local`` (a model on this machine, e.g. Qwen)
and ``glm`` (a stronger model on a remote box). It reuses the shared OpenAI
tool loop, so either name works as a Cascade verified-build agent.
"""

from cascade.providers.base import ProviderConfig
from cascade.providers.openai_compatible import OpenAICompatibleProvider
from cascade.providers import openai_compatible as oai_compat_mod
from cascade.providers.registry import get_registry, discover_providers


def _config(**overrides) -> ProviderConfig:
    base = {"api_key": "", "model": ""}
    base.update(overrides)
    return ProviderConfig(**base)


def test_registered_under_both_local_and_glm():
    """discover_providers should surface the provider under 'local' and 'glm'.

    Compared by name, not identity: discover_providers() reloads provider
    modules, so the registered class object differs from the import above.
    """
    discover_providers()
    registry = get_registry()
    for name in ("local", "glm"):
        registered = registry.get(name)
        assert registered is not None, name
        assert registered.__name__ == "OpenAICompatibleProvider"


def test_falls_back_to_local_class_defaults():
    """With an empty config, the provider defaults to the local Qwen endpoint."""
    provider = OpenAICompatibleProvider(_config())
    assert provider.base_url == "http://127.0.0.1:8080/v1"
    assert provider.model == "qwen36"


def test_config_overrides_defaults():
    """An explicit base_url/model in config wins over the class defaults.

    This is how the 'glm' registration points at the remote box.
    """
    provider = OpenAICompatibleProvider(
        _config(
            base_url="http://512s2.netbird.cloud:52415/v1",
            model="pipenetwork/GLM-5.2-MLX-8bit",
        )
    )
    assert provider.base_url == "http://512s2.netbird.cloud:52415/v1"
    assert provider.model == "pipenetwork/GLM-5.2-MLX-8bit"


def test_model_reads_config_live():
    """model must read config live so an in-place model swap takes effect."""
    provider = OpenAICompatibleProvider(_config(model="qwen36"))
    assert provider.model == "qwen36"
    provider.config.model = "pipenetwork/GLM-5.2-MLX-8bit"  # simulate a swap
    assert provider.model == "pipenetwork/GLM-5.2-MLX-8bit"


def test_omits_auth_header_when_keyless():
    """A blank key must not produce a bogus 'Bearer ' Authorization header."""
    headers = OpenAICompatibleProvider(_config())._headers()
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_includes_auth_header_when_key_present():
    """A placeholder/real key is forwarded as a bearer token when supplied."""
    headers = OpenAICompatibleProvider(_config(api_key="placeholder"))._headers()
    assert headers["Authorization"] == "Bearer placeholder"


def test_validate_requires_only_a_model():
    """Keyless endpoints validate: the model defaults even from an empty config."""
    assert OpenAICompatibleProvider(_config()).validate() is True


def test_ask_with_tools_delegates_to_shared_openai_loop(monkeypatch):
    """The agentic build path routes through the shared OpenAI tool loop."""
    captured = {}

    def fake_loop(**kwargs):
        captured.update(kwargs)
        return ("done", [{"tool": "write_file"}])

    monkeypatch.setattr(oai_compat_mod, "openai_ask_with_tools", fake_loop)

    provider = OpenAICompatibleProvider(_config(model="qwen36"))
    text, log = provider.ask_with_tools(
        [{"role": "user", "content": "edit the file"}], tools={}
    )

    assert text == "done"
    assert log == [{"tool": "write_file"}]
    assert captured["model"] == "qwen36"
    assert captured["url"] == f"{provider.base_url}/chat/completions"
    assert "Authorization" not in captured["headers"]


def test_local_and_glm_have_distinct_themes():
    """Both names render with their own accent, not the neutral fallback."""
    from cascade.theme import get_provider_theme, _NEUTRAL

    local = get_provider_theme("local")
    glm = get_provider_theme("glm")
    assert local is not _NEUTRAL and glm is not _NEUTRAL
    assert local.abbreviation == "loc"
    assert glm.abbreviation == "glm"
    assert local.accent != glm.accent
