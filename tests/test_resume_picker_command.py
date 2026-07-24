"""/resume dispatch: no-arg opens the picker, <id> still resumes directly."""

from unittest.mock import MagicMock

from cascade.commands import CommandHandler
from cascade.history.database import HistoryDB
from cascade.screens.session_picker import SessionPickerScreen
from cascade.state import CascadeState


class _App:
    def __init__(self, db: HistoryDB, cwd: str) -> None:
        self.db = db
        self.state = CascadeState()
        self.state.cwd = cwd
        self.screen = MagicMock()
        self.push_screen = MagicMock()

    def notify(self, _text: str) -> None:
        return None


def _handler(db: HistoryDB, cwd: str):
    app = _App(db, cwd)
    handler = CommandHandler(app)
    posted: list[str] = []
    handler._post_system = lambda text: posted.append(text)
    return app, handler, posted


def test_resume_no_arg_opens_picker_with_dir_sessions(tmp_path):
    db = HistoryDB(db_path=str(tmp_path / "h.db"))
    here = db.create_session(title="Here", provider="claude", cwd="/proj/alpha")
    db.create_session(title="Elsewhere", cwd="/proj/beta")
    unknown = db.create_session(title="Legacy", cwd="")

    app, handler, _ = _handler(db, "/proj/alpha")
    handler._cmd_resume([])

    app.push_screen.assert_called_once()
    screen = app.push_screen.call_args.args[0]
    assert isinstance(screen, SessionPickerScreen)
    ids = [s["id"] for s in screen._sessions]
    assert here["id"] in ids
    assert unknown["id"] in ids
    assert all(s["title"] != "Elsewhere" for s in screen._sessions)
    db.close()


def test_resume_picker_callback_resumes_selected_and_ignores_cancel(tmp_path):
    db = HistoryDB(db_path=str(tmp_path / "h.db"))
    db.create_session(title="Here", provider="claude", cwd="/proj/alpha")

    app, handler, _ = _handler(db, "/proj/alpha")
    resumed: list[str] = []
    handler._resume_session_id = lambda sid: resumed.append(sid)

    handler._cmd_resume([])
    callback = app.push_screen.call_args.args[1]

    callback(None)              # escape -> cancel, no resume
    assert resumed == []
    callback("amber-petal")     # enter -> resume the chosen id
    assert resumed == ["amber-petal"]
    db.close()


def test_resume_no_arg_no_sessions_reports_and_skips_picker(tmp_path):
    db = HistoryDB(db_path=str(tmp_path / "h.db"))
    db.create_session(title="Elsewhere", cwd="/proj/beta")

    app, handler, posted = _handler(db, "/proj/alpha")
    handler._cmd_resume([])

    app.push_screen.assert_not_called()
    assert posted == ["No sessions to resume in this directory."]
    db.close()


def test_resume_with_id_still_resumes_directly(tmp_path):
    db = HistoryDB(db_path=str(tmp_path / "h.db"))

    app, handler, _ = _handler(db, "/proj/alpha")
    resumed: list[str] = []
    handler._resume_session_id = lambda sid: resumed.append(sid)

    handler._cmd_resume(["night-river"])

    app.push_screen.assert_not_called()
    assert resumed == ["night-river"]
    db.close()


def test_history_is_directory_scoped(tmp_path):
    db = HistoryDB(db_path=str(tmp_path / "h.db"))
    here = db.create_session(title="Here", provider="claude", cwd="/proj/alpha")
    db.add_message(here["id"], role="user", content="hi")
    db.create_session(title="Elsewhere", cwd="/proj/beta")
    db.create_session(title="Legacy", cwd="")

    _, handler, posted = _handler(db, "/proj/alpha")
    handler._cmd_history([])

    body = posted[-1]
    assert "Here" in body
    assert "Legacy" in body          # unknown-dir stays reachable
    assert "Elsewhere" not in body   # other project scoped out
    assert "(1 msgs)" in body
    db.close()
