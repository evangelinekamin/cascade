import pytest

from src.pagination import paginate


def test_exact_multiple_does_not_create_an_extra_page():
    result = paginate(list(range(20)), page=2, page_size=10)
    assert result["items"] == list(range(10, 20))
    assert result["total_pages"] == 2


def test_partial_last_page():
    result = paginate(list(range(21)), page=3, page_size=10)
    assert result["items"] == [20]
    assert result["total_pages"] == 3


def test_empty_collection_has_no_pages():
    assert paginate([], page=1, page_size=10)["total_pages"] == 0


@pytest.mark.parametrize(("page", "page_size"), [(0, 10), (-1, 10), (1, 0), (1, -2)])
def test_invalid_page_arguments(page, page_size):
    with pytest.raises(ValueError):
        paginate([1, 2, 3], page=page, page_size=page_size)
