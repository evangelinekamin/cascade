"""Context builder for assembling file and text sources into conversations.

Provides a fluent API to add files, directories, and text snippets.
Tracks approximate token count to avoid exceeding provider limits.
"""

import base64
import mimetypes
from pathlib import Path


# Rough approximation: 1 token ~= 4 characters for English text
_CHARS_PER_TOKEN = 4

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


class ContextBuilder:
    """Accumulate context from multiple sources with a fluent API."""

    def __init__(self, max_tokens: int = 100_000):
        self._sources: list[dict] = []
        self._max_tokens = max_tokens
        self._current_chars = 0
        self._cached_build = ""
        self._dirty = False

    @property
    def token_estimate(self) -> int:
        return self._current_chars // _CHARS_PER_TOKEN

    @property
    def source_count(self) -> int:
        return len(self._sources)

    def add_text(self, text: str, label: str = "text") -> "ContextBuilder":
        """Add a raw text snippet."""
        self._sources.append({"type": "text", "label": label, "content": text})
        self._current_chars += len(text)
        self._dirty = True
        return self

    def add_file(self, path: str) -> "ContextBuilder":
        """Add a single file's contents (text or base64-encoded image)."""
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            self._sources.append({
                "type": "error",
                "label": str(file_path),
                "content": f"File not found: {file_path}",
            })
            self._dirty = True
            return self

        if file_path.suffix.lower() in _IMAGE_EXTENSIONS:
            return self._add_image(file_path)

        try:
            content = file_path.read_text(encoding="utf-8")
            self._sources.append({
                "type": "file",
                "label": file_path.name,
                "path": str(file_path),
                "content": content,
            })
            self._current_chars += len(content)
            self._dirty = True
        except Exception as e:
            self._sources.append({
                "type": "error",
                "label": str(file_path),
                "content": f"Error reading file: {e}",
            })
            self._dirty = True
        return self

    def add_directory(self, path: str, glob: str = "*") -> "ContextBuilder":
        """Add all matching files from a directory."""
        dir_path = Path(path).expanduser().resolve()
        if not dir_path.is_dir():
            self._sources.append({
                "type": "error",
                "label": str(dir_path),
                "content": f"Directory not found: {dir_path}",
            })
            return self

        for file_path in sorted(dir_path.glob(glob)):
            if file_path.is_file() and not file_path.name.startswith("."):
                if self.token_estimate >= self._max_tokens:
                    self._sources.append({
                        "type": "warning",
                        "label": "limit",
                        "content": f"Token limit reached (~{self.token_estimate} tokens). Skipping remaining files.",
                    })
                    break
                self.add_file(str(file_path))
        return self

    def add_image(self, path: str) -> "ContextBuilder":
        """Add an image file (base64-encoded)."""
        file_path = Path(path).expanduser().resolve()
        return self._add_image(file_path)

    def _add_image(self, file_path: Path) -> "ContextBuilder":
        """Internal: base64-encode an image file."""
        if not file_path.is_file():
            self._sources.append({
                "type": "error",
                "label": str(file_path),
                "content": f"Image not found: {file_path}",
            })
            self._dirty = True
            return self

        try:
            data = file_path.read_bytes()
            mime_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
            encoded = base64.b64encode(data).decode("ascii")
            self._sources.append({
                "type": "image",
                "label": file_path.name,
                "path": str(file_path),
                "mime_type": mime_type,
                "content": encoded,
            })
            # Images count roughly by their base64 size
            self._current_chars += len(encoded)
            self._dirty = True
        except Exception as e:
            self._sources.append({
                "type": "error",
                "label": str(file_path),
                "content": f"Error reading image: {e}",
            })
            self._dirty = True
        return self

    def build(self) -> str:
        """Assemble all sources into a structured context string."""
        if not self._sources:
            return ""
        if not self._dirty and self._cached_build:
            return self._cached_build

        parts = ["--- Context Sources ---\n"]
        for source in self._sources:
            stype = source["type"]
            label = source["label"]
            content = source["content"]

            if stype == "image":
                parts.append(f"### [Image] {label}\n(base64-encoded, {len(content)} chars)\n")
            elif stype in ("error", "warning"):
                parts.append(f"### [{stype.upper()}] {label}\n{content}\n")
            else:
                parts.append(f"### {label}\n{content}\n")

        parts.append(f"\n--- End Context ({self.source_count} sources, ~{self.token_estimate} tokens) ---")
        self._cached_build = "\n".join(parts)
        self._dirty = False
        return self._cached_build

    def clear(self) -> "ContextBuilder":
        """Reset all context sources."""
        self._sources.clear()
        self._current_chars = 0
        self._cached_build = ""
        self._dirty = False
        return self

    def list_sources(self) -> list[dict]:
        """Return a summary list of all added sources."""
        return [
            {
                "type": s["type"],
                "label": s["label"],
                "size": len(s["content"]),
            }
            for s in self._sources
        ]
