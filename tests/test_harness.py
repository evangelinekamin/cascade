import json

from cascade.harness import compare_reports, run_harness_benchmark, summarize_run
from cascade.providers.usage import Usage


def test_offline_harness_report_is_ordered_and_error_free():
    report = run_harness_benchmark(
        repeats=1,
        calls_per_repeat=3,
        delay_seconds=0,
    )

    assert report.results_ordered is True
    assert report.tool_errors == 0
    assert report.schema_bytes > 0
    assert report.serial_seconds >= 0
    assert report.parallel_seconds >= 0


def test_run_metrics_normalize_usage_and_tool_health():
    metrics = summarize_run(
        [
            {"output": '{"result":"ok"}'},
            {"output": '{"error":"bad args"}'},
            {"output": "[already read above: a.py]"},
        ],
        Usage(input=10, output=4, cache_read=30, cache_write=10, cost=0.02),
        1.25,
    )

    assert metrics.tool_calls == 3
    assert metrics.tool_errors == 1
    assert metrics.duplicate_reads == 1
    assert metrics.cache_ratio == 0.6
    assert metrics.cost == 0.02


def test_report_comparison_uses_signed_percent_changes():
    report = run_harness_benchmark(
        repeats=1,
        calls_per_repeat=2,
        delay_seconds=0,
    )
    baseline = report.to_dict()
    baseline["parallel_seconds"] = max(report.parallel_seconds * 2, 0.001)
    baseline["speedup"] = max(report.speedup / 2, 0.001)

    deltas = compare_reports(report, json.loads(json.dumps(baseline)))

    assert deltas["parallel_seconds_pct"] < 0
    assert deltas["speedup_pct"] > 0
