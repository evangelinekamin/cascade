"""Tests for Gemini provider auth/path behavior."""

from unittest.mock import patch

from cascade.providers.base import ProviderConfig
from cascade.providers.gemini import GeminiProvider
from cascade.providers.usage import Usage
from cascade.tools.permissions import PermissionEngine


def test_uses_cli_proxy_for_oauth_token_when_gemini_binary_exists():
    """ya29 OAuth token should route through gemini CLI when available."""
    with patch("cascade.providers.gemini.shutil.which", return_value="/usr/bin/gemini"):
        provider = GeminiProvider(
            ProviderConfig(api_key="ya29.test-token", model="gemini-3.1-pro-preview")
        )
    assert provider._use_cli_proxy is True


def test_does_not_use_cli_proxy_for_api_key():
    """Regular API key should use direct Gemini API calls."""
    with patch("cascade.providers.gemini.shutil.which", return_value="/usr/bin/gemini"):
        provider = GeminiProvider(
            ProviderConfig(api_key="AIzaSy-test-key", model="gemini-2.5-flash")
        )
    assert provider._use_cli_proxy is False


def test_stream_cli_parses_assistant_messages_and_usage():
    """CLI stream-json output should yield assistant deltas and capture usage."""

    class _FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.stdout = iter(
                [
                    '{"type":"init","session_id":"x"}\n',
                    '{"type":"message","role":"assistant","content":"Hel","delta":true}\n',
                    '{"type":"message","role":"assistant","content":"lo","delta":true}\n',
                    '{"type":"result","status":"success","stats":{"input_tokens":10,"output_tokens":4}}\n',
                ]
            )
            self.returncode = 0

        def wait(self):
            return 0

    with patch("cascade.providers.gemini.shutil.which", return_value="/usr/bin/gemini"):
        with patch.dict("os.environ", {"CASCADE_GEMINI_ACTIVITY": "0"}, clear=False):
            provider = GeminiProvider(
                ProviderConfig(api_key="ya29.test-token", model="gemini-3.1-pro-preview")
            )

    with patch("cascade.providers._cli_proxy.subprocess.Popen", _FakePopen):
        chunks = list(provider.stream_single("Say hello"))

    assert chunks == ["Hel", "lo"]
    assert provider.last_usage == Usage(input=10, output=4)


def test_gemini_has_use_oauth_cli_attribute():
    """_use_oauth_cli should be consistent with _use_bearer for ya29 tokens."""
    with patch("cascade.providers.gemini.shutil.which", return_value="/usr/bin/gemini"):
        provider = GeminiProvider(
            ProviderConfig(api_key="ya29.test-token", model="gemini-3.1-pro-preview")
        )
    assert provider._use_oauth_cli is True
    assert provider._use_oauth_cli == provider._use_bearer


def _gemini_proxy_cmd(posture):
    captured = {}
    with patch("cascade.providers.gemini.shutil.which", return_value="/usr/bin/gemini"):
        provider = GeminiProvider(
            ProviderConfig(api_key="ya29.test-token", model="gemini-test")
        )
    provider.permission_engine = PermissionEngine(posture=posture)
    provider._approval_mode_supported = True
    provider._skip_trust_supported = True

    def fake_stream(cfg, handler, emit, *args):
        captured["cmd"] = list(cfg.cmd_args)
        return iter(())

    with patch("cascade.providers.gemini.stream_cli_proxy", side_effect=fake_stream):
        list(provider._stream_via_cli([{"role": "user", "content": "work"}]))
    return captured["cmd"]


def test_cli_permission_modes_are_popup_free():
    auto = _gemini_proxy_cmd("auto")
    assert auto[auto.index("--approval-mode") + 1] == "yolo"
    assert "--sandbox" in auto
    assert "--skip-trust" in auto

    yolo = _gemini_proxy_cmd("yolo")
    assert yolo[yolo.index("--approval-mode") + 1] == "yolo"

    safe = _gemini_proxy_cmd("safe")
    assert safe[safe.index("--approval-mode") + 1] == "auto_edit"

    readonly = _gemini_proxy_cmd("readonly")
    assert readonly[readonly.index("--approval-mode") + 1] == "plan"
