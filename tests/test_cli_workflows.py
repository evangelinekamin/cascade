import json
import importlib
from types import SimpleNamespace

from click.testing import CliRunner

from cascade.capabilities import CliCapability, DoctorReport

cli_module = importlib.import_module("cascade.cli")


def test_headless_run_emits_one_structured_receipt(monkeypatch):
    fake = SimpleNamespace(
        run_automatic=lambda prompt, provider=None, mode="build": {
            "schema_version": 1,
            "objective": prompt,
            "outcome": "succeeded",
            "workflow": "fanout",
            "provider": provider,
            "mode": mode,
            "text": "done",
        }
    )
    monkeypatch.setattr(cli_module, "get_app", lambda: fake)

    result = CliRunner().invoke(
        cli_module.cli,
        ["run", "build both pieces", "--provider", "openai", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["workflow"] == "fanout"
    assert payload["objective"] == "build both pieces"
    assert payload["provider"] == "openai"


def test_doctor_cli_supports_machine_readable_output(monkeypatch):
    capability = CliCapability(name="git", available=True, version="git test")
    report = DoctorReport(
        generated_at="now",
        python="3.13",
        python_ok=True,
        git=capability,
        provider_clis=(),
        configured_providers=("openai",),
        permission_posture="auto",
    )
    monkeypatch.setattr(cli_module, "get_app", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr("cascade.capabilities.run_doctor", lambda *_args, **_kwargs: report)

    result = CliRunner().invoke(cli_module.cli, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["permission_posture"] == "auto"
