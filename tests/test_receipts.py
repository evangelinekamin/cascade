from cascade.receipts import (
    build_receipt_payload,
    format_receipt_markdown,
    suggested_next_action,
)


def test_receipt_includes_route_verification_tasks_and_grounded_follow_up():
    runs = [{
        "id": "run-1",
        "workflow": "fanout",
        "status": "succeeded",
        "objective": "Build both endpoints",
        "provider": "openai",
        "model": "gpt",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost": 0.1,
        "worktree_path": "/tmp/cascade-wt",
        "error": "",
        "metadata": {
            "route": "fanout",
            "route_reason": "independent endpoints",
            "verification_kind": "test-suite",
            "changed_files": ["health.py", "version.py"],
        },
    }]
    payload = build_receipt_payload(
        {"id": "session-1", "title": "Demo", "provider": "openai", "model": "gpt"},
        [{"role": "USER", "timestamp": "now", "content": "Do it"}],
        runs,
        {"run-1": [{"task_id": "health", "status": "integrated", "description": "Health"}]},
    )
    markdown = format_receipt_markdown(payload)

    assert "independent endpoints" in markdown
    assert "Verification kind: `test-suite`" in markdown
    assert "`health`" in markdown
    assert "git -C /tmp/cascade-wt diff" in markdown
    assert "/apply" in suggested_next_action(runs)


def test_receipt_does_not_invent_a_follow_up_without_run_evidence():
    assert suggested_next_action([]) == ""
