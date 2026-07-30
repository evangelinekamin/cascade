def parse_row(value: str) -> list[str]:
    # Deliberately broken evaluation fixture: quoted commas are split.
    return value.split(",")
