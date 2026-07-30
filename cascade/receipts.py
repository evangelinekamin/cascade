"""Portable session transcripts with reproducible orchestration receipts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def suggested_next_action(runs: list[dict[str, Any]]) -> str:
    """Return one grounded action, or nothing when the record implies none."""
    if not runs:
        return ""
    latest = runs[0]
    status = str(latest.get("status") or "")
    path = str(latest.get("worktree_path") or "")
    if status == "succeeded" and path:
        return f"Review the verified changes with `git -C {path} diff`, then use `/apply`."
    if status in {"failed", "blocked", "partial", "interrupted"}:
        return "Inspect the failed task/error in this receipt before retrying the request."
    return ""


def build_receipt_payload(
    session: dict[str, Any],
    messages: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    tasks_by_run: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build the format-neutral export payload."""
    normalized_runs = []
    for run in runs:
        item = dict(run)
        item["tasks"] = tasks_by_run.get(str(run.get("id") or ""), [])
        normalized_runs.append(item)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session": dict(session),
        "messages": messages,
        "runs": normalized_runs,
        "suggested_next_action": suggested_next_action(runs),
    }


def format_receipt_markdown(payload: dict[str, Any]) -> str:
    """Render a readable transcript plus exact run/task evidence."""
    session = payload["session"]
    title = session.get("title") or "untitled"
    provider = session.get("provider") or "assistant"
    model = session.get("model") or provider
    lines = [
        f"# Cascade Session: {title}",
        "",
        f"- Session: `{session.get('id', '')}`",
        f"- Model: `{model}`",
        f"- Exported: {payload['generated_at']}",
        "",
        "## Transcript",
        "",
    ]
    for message in payload["messages"]:
        role = str(message.get("role") or "assistant")
        timestamp = str(message.get("timestamp") or "")[:19]
        lines.extend(
            [
                f"### {role} ({timestamp})",
                "",
                str(message.get("content") or ""),
                "",
            ]
        )

    lines.extend(["## Run receipts", ""])
    runs = payload["runs"]
    if not runs:
        lines.extend(["No orchestration runs were recorded for this session.", ""])
    for run in runs:
        lines.extend(
            [
                f"### {run.get('workflow', 'run')} · {run.get('status', 'unknown')}",
                "",
                f"- Run ID: `{run.get('id', '')}`",
                f"- Objective: {run.get('objective', '')}",
                f"- Provider/model: `{run.get('provider', '')}` / `{run.get('model', '')}`",
                f"- Tokens: {int(run.get('input_tokens') or 0):,} in / "
                f"{int(run.get('output_tokens') or 0):,} out",
                f"- Cost: {float(run.get('cost') or 0.0):.6f}",
            ]
        )
        if run.get("worktree_path"):
            lines.append(f"- Worktree: `{run['worktree_path']}`")
            lines.append(
                f"- Reproduce review: `git -C {run['worktree_path']} diff`"
            )
        metadata = run.get("metadata") or {}
        if metadata.get("route_reason"):
            lines.append(
                f"- Route: `{metadata.get('route', run.get('workflow', ''))}` — "
                f"{metadata['route_reason']}"
            )
        if metadata.get("verification_kind"):
            lines.append(f"- Verification kind: `{metadata['verification_kind']}`")
        if metadata.get("changed_files"):
            lines.append("- Changed files: " + ", ".join(metadata["changed_files"]))
        if run.get("error"):
            lines.append(f"- Error: {run['error']}")
        tasks = run.get("tasks") or []
        if tasks:
            lines.extend(["", "Tasks:", ""])
            for task in tasks:
                detail = f"- [{task.get('status', 'unknown')}] `{task.get('task_id', '')}`"
                if task.get("description"):
                    detail += f": {task['description']}"
                lines.append(detail)
                if task.get("error"):
                    lines.append(f"  - Error: {task['error']}")
        lines.append("")

    follow_up = payload.get("suggested_next_action") or ""
    if follow_up:
        lines.extend(["## Suggested next action", "", follow_up, ""])
    return "\n".join(lines).rstrip() + "\n"
