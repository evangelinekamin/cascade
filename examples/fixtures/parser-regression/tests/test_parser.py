from src.parser import parse_row


def test_plain_fields():
    assert parse_row("one,two") == ["one", "two"]


def test_quoted_field():
    assert parse_row('one,"two,three"') == ["one", "two,three"]
