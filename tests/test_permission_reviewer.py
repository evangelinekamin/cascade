"""Popup-free model permission reviewer behavior."""

import json
import time

from cascade.tools.permissions import (
    PermissionContext,
    PermissionReview,
)
from cascade.tools.reviewer import ModelPermissionReviewer, _parse_decision


class _Client:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Provider:
    def __init__(self, response, delay=0):
        self.response = response
        self.delay = delay
        self.client = _Client()
        self.prompt = ""
        self.system = ""

    def ask_single(self, prompt, system=None):
        self.prompt = prompt
        self.system = system or ""
        if self.delay:
            time.sleep(self.delay)
        return self.response


def _review(**arguments):
    return PermissionReview(
        tool_name="write_file",
        arguments=arguments,
        reason="safe posture reviews mutations",
        rule="posture",
        workspace_root="/workspace",
        context=PermissionContext(
            objective="update the parser",
            provider="openai",
            model="model-a",
            mode="build",
        ),
    )


def test_reviewer_uses_toolless_prompt_and_omits_source_payloads():
    provider = _Provider('{"decision":"allow","reason":"bounded workspace edit"}')
    reviewer = ModelPermissionReviewer(lambda: provider, timeout=1)

    decision = reviewer(_review(
        path="parser.py",
        content="TOP SECRET SOURCE",
        metadata={"payload": "NESTED SECRET"},
    ))

    assert decision.allow
    payload = json.loads(provider.prompt)
    assert payload["objective"] == "update the parser"
    assert payload["pending_action"]["arguments"]["path"] == "parser.py"
    assert "TOP SECRET SOURCE" not in provider.prompt
    assert "NESTED SECRET" not in provider.prompt
    assert "omitted payload" in provider.prompt
    assert "must return" in provider.system
    assert provider.client.closed


def test_reviewer_accepts_fenced_json():
    decision = _parse_decision(
        '```json\n{"decision":"deny","reason":"external side effect"}\n```'
    )
    assert decision is not None
    assert not decision.allow
    assert decision.reason == "external side effect"


def test_invalid_response_fails_closed():
    provider = _Provider("sure, looks fine")
    decision = ModelPermissionReviewer(lambda: provider, timeout=1)(_review(path="x.py"))
    assert not decision.allow
    assert "invalid JSON" in decision.reason


def test_missing_provider_fails_closed():
    decision = ModelPermissionReviewer(lambda: None, timeout=1)(_review(path="x.py"))
    assert not decision.allow
    assert "no direct reviewer provider" in decision.reason


def test_review_timeout_fails_closed_promptly():
    provider = _Provider('{"decision":"allow","reason":"late"}', delay=0.25)
    started = time.monotonic()
    decision = ModelPermissionReviewer(lambda: provider, timeout=0.1)(
        _review(path="x.py")
    )
    elapsed = time.monotonic() - started

    assert not decision.allow
    assert "timed out" in decision.reason
    assert elapsed < 0.2
