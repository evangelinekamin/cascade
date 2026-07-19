"""Output filters: the two rtk non-negotiables + per-command behavior."""

from cascade.tools.filters import (
    apply_output_filter,
    filter_grep,
    filter_pytest,
    wants_structured_output,
)


class TestNonNegotiables:
    def test_never_worse_keeps_raw_when_filter_would_not_shrink(self):
        raw = "short output\nExit 0"
        # A grep filter on tiny output returns it unchanged -> raw kept
        assert apply_output_filter("grep x .", raw) == raw

    def test_structured_output_bypasses_filtering(self):
        raw = '{"failed": 5}\n' + "noise\n" * 500
        assert apply_output_filter("pytest --json", raw) == raw
        assert apply_output_filter("git status | wc -l", raw) == raw

    def test_recovery_hint_appended_on_real_saving(self):
        raw = "\n".join(f"tests/test_{i}.py::t PASSED" for i in range(200))
        raw += "\n===== 200 passed in 3s ====="
        out = apply_output_filter("pytest -q", raw, recovery_hint="read /tmp/x.txt")
        assert len(out) < len(raw)
        assert "full text: read /tmp/x.txt" in out

    def test_unknown_command_returns_raw(self):
        raw = "some output\n" * 100
        assert apply_output_filter("frobnicate --hard", raw) == raw


class TestStructuredDetection:
    def test_flags_and_pipes(self):
        assert wants_structured_output("gh pr list --json number")
        assert wants_structured_output("cat x | jq .")
        assert wants_structured_output("ls | wc -l")
        assert not wants_structured_output("pytest -q")


class TestPytestFilter:
    def test_all_green_collapses_to_summary(self):
        raw = "\n".join(f"test_{i} PASSED" for i in range(100))
        raw += "\n===== 100 passed in 2.3s ====="
        out = filter_pytest(raw)
        assert "100 passed" in out
        assert "test_5 PASSED" not in out

    def test_failures_retained_with_summary(self):
        raw = (
            "\n".join(f"test_{i} PASSED" for i in range(50))
            + "\n_____ test_broken _____\n"
            + "    assert 1 == 2\nE   AssertionError\n"
            + "===== short test summary info =====\n"
            + "FAILED tests/test_x.py::test_broken\n"
            + "===== 1 failed, 50 passed in 1s ====="
        )
        out = filter_pytest(raw)
        assert "test_broken" in out
        assert "AssertionError" in out
        assert "1 failed, 50 passed" in out
        assert "test_5 PASSED" not in out

    def test_python_m_pytest_routes_to_pytest_filter(self):
        raw = "\n".join(f"t{i} PASSED" for i in range(100)) + "\n===== 100 passed in 1s ====="
        out = apply_output_filter("python -m pytest tests/", raw)
        assert "100 passed" in out
        assert len(out) < len(raw)


class TestGrepFilter:
    def test_groups_by_file_over_threshold(self):
        raw = "\n".join(f"src/a.py:{i}: match here" for i in range(60))
        out = filter_grep(raw)
        assert "src/a.py: 60 matches" in out
        assert "more in this file" in out
