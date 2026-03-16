"""Tests for SQLite conversation history."""


import pytest

from cascade.history.database import HistoryDB


@pytest.fixture
def db(tmp_path):
    """Create a temporary HistoryDB."""
    db_path = str(tmp_path / "test_history.db")
    hist = HistoryDB(db_path=db_path)
    yield hist
    hist.close()


def test_create_session(db):
    session = db.create_session(provider="gemini", model="gemini-2.0-flash", title="Test")
    assert session["provider"] == "gemini"
    assert session["model"] == "gemini-2.0-flash"
    assert session["title"] == "Test"
    assert "-" in session["id"]  # word-based ID like "amber-petal"


def test_list_sessions(db):
    db.create_session(title="First")
    db.create_session(title="Second")
    sessions = db.list_sessions()
    assert len(sessions) == 2
    # Most recent first
    assert sessions[0]["title"] == "Second"


def test_get_session(db):
    created = db.create_session(title="Find me")
    found = db.get_session(created["id"])
    assert found is not None
    assert found["title"] == "Find me"


def test_get_session_not_found(db):
    assert db.get_session("nonexistent") is None


def test_delete_session(db):
    session = db.create_session(title="Delete me")
    assert db.delete_session(session["id"]) is True
    assert db.get_session(session["id"]) is None


def test_delete_session_not_found(db):
    assert db.delete_session("nonexistent") is False


def test_add_message(db):
    session = db.create_session()
    msg = db.add_message(session["id"], role="user", content="Hello")
    assert msg["role"] == "user"
    assert msg["content"] == "Hello"
    assert msg["session_id"] == session["id"]


def test_get_session_messages(db):
    session = db.create_session()
    db.add_message(session["id"], role="user", content="Hi")
    db.add_message(session["id"], role="assistant", content="Hello!")
    messages = db.get_session_messages(session["id"])
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_search_sessions_by_title(db):
    db.create_session(title="Python debugging")
    db.create_session(title="Rust performance")
    results = db.search_sessions("Python")
    assert len(results) == 1
    assert results[0]["title"] == "Python debugging"


def test_search_sessions_by_message_content(db):
    s = db.create_session(title="Generic")
    db.add_message(s["id"], role="user", content="How do I fix a segfault?")
    results = db.search_sessions("segfault")
    assert len(results) == 1


def test_update_session_title(db):
    session = db.create_session(title="Old title")
    db.update_session_title(session["id"], "New title")
    updated = db.get_session(session["id"])
    assert updated["title"] == "New title"


def test_message_metadata(db):
    session = db.create_session()
    msg = db.add_message(
        session["id"], role="user", content="test",
        metadata={"tokens_in": 5},
    )
    assert msg["metadata"] == {"tokens_in": 5}


def test_session_metadata(db):
    session = db.create_session(metadata={"context": "project-x"})
    assert session["metadata"] == {"context": "project-x"}


def test_cascade_delete_messages(db):
    """Deleting a session should cascade-delete its messages."""
    session = db.create_session()
    db.add_message(session["id"], role="user", content="msg1")
    db.add_message(session["id"], role="assistant", content="msg2")
    db.delete_session(session["id"])
    messages = db.get_session_messages(session["id"])
    assert len(messages) == 0
