"""Contract tests for AgentMemory -- the governed shared blackboard."""

import pytest

from cascade.swarm.memory import AgentMemory, MemoryEntry


def test_report_returns_entry_and_persists():
    mem = AgentMemory(":memory:")
    entry = mem.report("run1", "worker-a", "result", "added foo.py", refs=("foo.py",))
    assert isinstance(entry, MemoryEntry)
    assert entry.kind == "result"
    assert entry.agent == "worker-a"
    assert entry.refs == ("foo.py",)

    rows = mem.entries("run1")
    assert len(rows) == 1
    assert rows[0].content == "added foo.py"
    assert rows[0].refs == ("foo.py",)


def test_entries_are_scoped_by_run():
    mem = AgentMemory(":memory:")
    mem.report("run1", "a", "fact", "x")
    mem.report("run2", "b", "fact", "y")
    assert [e.content for e in mem.entries("run1")] == ["x"]
    assert [e.content for e in mem.entries("run2")] == ["y"]


def test_entries_filter_by_kind():
    mem = AgentMemory(":memory:")
    mem.report("r", "a", "decision", "use sqlite")
    mem.report("r", "a", "blocker", "stuck on import")
    assert [e.content for e in mem.entries("r", kind="blocker")] == ["stuck on import"]


def test_report_rejects_unknown_kind():
    mem = AgentMemory(":memory:")
    with pytest.raises(ValueError):
        mem.report("r", "a", "bogus", "x")


def test_digest_curates_the_run():
    mem = AgentMemory(":memory:")
    mem.report("r", "director", "contract", "Worker A owns foo.py; Worker B owns bar.py")
    mem.report("r", "a", "decision", "use a frozen dataclass")
    mem.report("r", "a", "result", "first attempt", refs=("foo.py",))
    mem.report("r", "a", "result", "foo.py done, tests green", refs=("foo.py",))
    mem.report("r", "b", "result", "bar.py done", refs=("bar.py",))
    mem.report("r", "b", "blocker", "needs A's interface")

    digest = mem.digest("r")

    assert "Worker A owns foo.py" in digest          # contract
    assert "use a frozen dataclass" in digest        # decision
    assert "foo.py done, tests green" in digest       # latest result for a
    assert "first attempt" not in digest              # superseded result dropped
    assert "bar.py done" in digest                    # latest result for b
    assert "needs A's interface" in digest            # open blocker


def test_digest_empty_run_is_a_string():
    mem = AgentMemory(":memory:")
    assert isinstance(mem.digest("nope"), str)
