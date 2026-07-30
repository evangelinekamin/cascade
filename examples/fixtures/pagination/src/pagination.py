def paginate(items: list, page: int, page_size: int) -> dict:
    """Return one 1-indexed page and stable pagination metadata."""
    total = len(items)
    total_pages = (total + page_size) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    }
