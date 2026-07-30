"""ChatTextArea multiline composer: submit, newline, history, value compat."""

from cascade.widgets.input_frame import ChatTextArea


class TestValueCompat:
    def test_value_reads_and_writes_text(self):
        ta = ChatTextArea()
        ta.value = "hello world"
        assert ta.value == "hello world"
        assert ta.text == "hello world"

    def test_value_none_clears(self):
        ta = ChatTextArea()
        ta.value = "x"
        ta.value = None
        assert ta.value == ""


class TestHistory:
    def test_record_dedups_consecutive(self):
        ta = ChatTextArea()
        ta.record("a")
        ta.record("a")
        ta.record("b")
        assert ta._history == ["a", "b"]

    def test_record_resets_index_and_draft(self):
        ta = ChatTextArea()
        ta._history_idx = 2
        ta._draft = "d"
        ta.record("new")
        assert ta._history_idx == -1
        assert ta._draft == ""


class TestSubmitMessage:
    def test_submit_posts_message_with_text(self):
        ta = ChatTextArea()
        ta.value = "my prompt"
        posted = []
        ta.post_message = lambda m: posted.append(m)
        ta._submit()
        assert len(posted) == 1
        assert isinstance(posted[0], ChatTextArea.Submitted)
        assert posted[0].value == "my prompt"
        assert posted[0].text_area is ta


class TestNewlineAction:
    def test_action_newline_inserts(self):
        ta = ChatTextArea()
        ta.value = "line1"
        inserted = []
        ta.insert = lambda s: inserted.append(s)
        ta.action_newline()
        assert inserted == ["\n"]
