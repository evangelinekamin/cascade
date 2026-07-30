import json

from click.testing import CliRunner

from cascade.cli import cli


def test_benchmark_command_outputs_machine_readable_report():
    result = CliRunner().invoke(
        cli,
        ["benchmark", "--repeats", "1", "--calls", "2", "--delay", "0", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["repeats"] == 1
    assert payload["calls_per_repeat"] == 2
    assert payload["results_ordered"] is True
    assert payload["tool_errors"] == 0


def test_benchmark_command_writes_report_and_compares_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    report_path = tmp_path / "reports" / "current.json"
    baseline.write_text(
        json.dumps({
            "serial_seconds": 1,
            "parallel_seconds": 1,
            "hook_p50_ms": 1,
            "hook_p95_ms": 1,
            "speedup": 1,
        })
    )

    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "--repeats",
            "1",
            "--calls",
            "2",
            "--delay",
            "0",
            "--baseline",
            str(baseline),
            "--output",
            str(report_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    written = json.loads(report_path.read_text())
    assert "baseline_delta" in written
    assert json.loads(result.output) == written
