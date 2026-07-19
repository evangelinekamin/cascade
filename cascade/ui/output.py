"""Output rendering for the one-shot `cascade-cli` commands.

Self-contained: theme-accented headers plus plain text bodies. The old
gutter/stream renderer machinery lived here for the Rich REPL, which is
gone; one-shot output does not need incremental markdown rendering.
"""

import sys
from typing import Iterator

from rich.text import Text

from .theme import DEFAULT_THEME, console


def _print_provider_header(provider: str) -> None:
    if not provider:
        return
    theme = DEFAULT_THEME.get_provider(provider)
    header = Text()
    header.append(f"{theme.abbreviation} ", style=f"bold {theme.accent}")
    header.append("| ", style=f"dim {DEFAULT_THEME.palette.text_muted}")
    header.append(f"[{provider}]", style=f"dim {DEFAULT_THEME.palette.text_dim}")
    console.print(header)


def render_response(
    text: str,
    provider: str = "",
    thinking: str = "",
    language: str = "text",
) -> None:
    """Render a complete response with a provider-accented header."""
    if thinking:
        palette = DEFAULT_THEME.palette
        think_line = Text()
        think_line.append("[thinking] ", style=f"dim {palette.text_dim}")
        think_line.append(thinking[:200], style=f"dim {palette.text}")
        console.print(think_line)

    _print_provider_header(provider)
    console.print(Text(text))


def stream_response(
    stream_iter: Iterator[str],
    provider: str = "",
) -> str:
    """Stream chunks to stdout as they arrive; return the full text."""
    _print_provider_header(provider)

    full_text = ""
    for chunk in stream_iter:
        full_text += chunk
        sys.stdout.write(chunk)
        sys.stdout.flush()

    if not full_text.endswith("\n"):
        sys.stdout.write("\n")
        sys.stdout.flush()
    return full_text


def render_error(text: str) -> None:
    """Render an error message."""
    palette = DEFAULT_THEME.palette
    err = Text()
    err.append("err ", style=f"bold {palette.error}")
    err.append("| ", style=f"dim {palette.text_muted}")
    err.append(text, style=palette.error)
    console.print(err)


def render_comparison(results: list[dict]) -> None:
    """Render comparison results from multiple providers."""
    for result in results:
        provider = result.get("provider", "unknown")
        response = result.get("response", "")
        _print_provider_header(provider)
        console.print(Text(response))
        console.print()


def render_thinking(text: str) -> None:
    """Render thinking/processing output."""
    palette = DEFAULT_THEME.palette
    t = Text()
    t.append("... ", style=f"dim {palette.spinner}")
    t.append("| ", style=f"dim {palette.text_muted}")
    t.append(text, style=f"dim {palette.text}")
    console.print(t)
