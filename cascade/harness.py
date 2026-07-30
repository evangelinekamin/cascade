"""Deterministic measurements for Cascade's local agent harness.

This benchmark deliberately makes no provider calls and spends no tokens. It
measures mechanics that should stay stable across model changes: safe tool
batching, result ordering, hook overhead, and tool-schema size. Reports are
JSON-serializable so a previous run can be checked in and used as a baseline.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from .hooks import HookContext, HookDefinition, HookEvent, HookRunner
from .providers.usage import Usage
from .tools.executor import ConcurrentToolExecutor, ToolExecutor
from .tools.schema import callable_to_tool_def


@dataclass(frozen=True)
class RunMetrics:
    """Normalized metrics for one real or synthetic agent run."""

    duration_seconds: float
    tool_calls: int
    tool_errors: int
    duplicate_reads: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float | None = None

    @property
    def cache_ratio(self) -> float:
        prompt = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / prompt if prompt else 0.0


@dataclass(frozen=True)
class HarnessReport:
    """Offline benchmark report suitable for JSON baselines."""

    generated_at: str
    repeats: int
    calls_per_repeat: int
    delay_seconds: float
    serial_seconds: float
    parallel_seconds: float
    speedup: float
    hook_p50_ms: float
    hook_p95_ms: float
    schema_bytes: int
    results_ordered: bool
    tool_errors: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_run(
    tool_log: Iterable[dict],
    usage: Usage | None,
    duration_seconds: float,
) -> RunMetrics:
    """Summarize provider-independent usage and tool-loop health."""
    entries = list(tool_log)
    errors = 0
    duplicate_reads = 0
    for entry in entries:
        output = str(entry.get("output", ""))
        if "[already read above:" in output:
            duplicate_reads += 1
        try:
            decoded = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if (
            isinstance(decoded, dict) and "error" in decoded
        ) or output.startswith("[tool-call error]"):
            errors += 1
    normalized = usage or Usage()
    return RunMetrics(
        duration_seconds=max(float(duration_seconds), 0.0),
        tool_calls=len(entries),
        tool_errors=errors,
        duplicate_reads=duplicate_reads,
        input_tokens=normalized.input,
        output_tokens=normalized.output,
        cache_read_tokens=normalized.cache_read,
        cache_write_tokens=normalized.cache_write,
        cost=normalized.cost,
    )


def run_harness_benchmark(
    *,
    repeats: int = 5,
    calls_per_repeat: int = 4,
    delay_seconds: float = 0.02,
) -> HarnessReport:
    """Measure serial versus safe-batch execution without external services."""
    repeats = max(int(repeats), 1)
    calls_per_repeat = max(int(calls_per_repeat), 2)
    delay_seconds = max(float(delay_seconds), 0.0)

    def read_item(tag: str) -> str:
        """Read one independent synthetic item."""
        if delay_seconds:
            time.sleep(delay_seconds)
        return tag

    tool = callable_to_tool_def(
        "read_item",
        read_item,
        read_only=True,
    )
    tools = {"read_item": tool}
    calls = [
        ("read_item", {"tag": f"item-{index}"})
        for index in range(calls_per_repeat)
    ]
    expected = [f"item-{index}" for index in range(calls_per_repeat)]
    serial_executor = ToolExecutor(tools)
    parallel_executor = ConcurrentToolExecutor(tools)

    serial_samples = []
    parallel_samples = []
    ordered = True
    errors = 0
    for _ in range(repeats):
        start = time.perf_counter()
        serial_raw = [
            serial_executor.execute(tool_name, arguments)
            for tool_name, arguments in calls
        ]
        serial_samples.append(time.perf_counter() - start)

        start = time.perf_counter()
        parallel_raw = parallel_executor.execute_batch(calls)
        parallel_samples.append(time.perf_counter() - start)

        for raw_group in (serial_raw, parallel_raw):
            decoded = [json.loads(raw) for raw in raw_group]
            errors += sum(1 for item in decoded if "error" in item)
            ordered = ordered and [item.get("result") for item in decoded] == expected

    hook_runner = HookRunner(hooks=(
        HookDefinition(
            name="benchmark-noop",
            event=HookEvent.BEFORE_ASK,
            handler=lambda _ctx: None,
        ),
    ))
    hook_samples = []
    for _ in range(max(20, repeats * calls_per_repeat)):
        start = time.perf_counter()
        hook_runner.emit(
            HookEvent.BEFORE_ASK,
            HookContext(event=HookEvent.BEFORE_ASK.value),
        )
        hook_samples.append((time.perf_counter() - start) * 1000)

    serial_seconds = statistics.median(serial_samples)
    parallel_seconds = statistics.median(parallel_samples)
    speedup = serial_seconds / parallel_seconds if parallel_seconds else 0.0
    sorted_hooks = sorted(hook_samples)
    p95_index = min(int(len(sorted_hooks) * 0.95), len(sorted_hooks) - 1)
    schema_bytes = len(
        json.dumps(tool.parameters, sort_keys=True, separators=(",", ":")).encode()
    )
    return HarnessReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        repeats=repeats,
        calls_per_repeat=calls_per_repeat,
        delay_seconds=delay_seconds,
        serial_seconds=serial_seconds,
        parallel_seconds=parallel_seconds,
        speedup=speedup,
        hook_p50_ms=statistics.median(hook_samples),
        hook_p95_ms=sorted_hooks[p95_index],
        schema_bytes=schema_bytes,
        results_ordered=ordered,
        tool_errors=errors,
    )


def compare_reports(current: HarnessReport, baseline: dict[str, Any]) -> dict[str, float]:
    """Return signed percentage changes; negative latency is an improvement."""
    deltas = {}
    for field in ("serial_seconds", "parallel_seconds", "hook_p50_ms", "hook_p95_ms"):
        old = baseline.get(field)
        new = getattr(current, field)
        if isinstance(old, (int, float)) and old:
            deltas[f"{field}_pct"] = ((new - float(old)) / float(old)) * 100
    old_speedup = baseline.get("speedup")
    if isinstance(old_speedup, (int, float)) and old_speedup:
        deltas["speedup_pct"] = (
            (current.speedup - float(old_speedup)) / float(old_speedup)
        ) * 100
    return deltas
