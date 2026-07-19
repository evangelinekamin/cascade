"""Starlette-based file upload server for Cascade.

Runs in a daemon thread alongside the CLI REPL. Shares the REPL's
ContextBuilder instance so uploaded files appear in context immediately.

Optional dependency: pip install cascade-cli[web]
"""

import threading
from typing import Optional

from cascade.context.memory import ContextBuilder

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    _HAS_STARLETTE = True
except ImportError:
    _HAS_STARLETTE = False


def _check_deps() -> None:
    if not _HAS_STARLETTE:
        raise ImportError(
            "Web dependencies not installed. Run: pip install cascade-cli[web]"
        )


class FileUploaderServer:
    """Lightweight web server for drag-and-drop file uploads."""

    def __init__(
        self,
        context_builder: ContextBuilder,
        # Loopback only: this is an unauthenticated upload endpoint for the
        # local TUI. Pass an explicit host to expose it on a LAN deliberately.
        host: str = "127.0.0.1",
        port: int = 9222,
        max_upload_bytes: int = 5 * 1024 * 1024,
    ):
        _check_deps()
        self.context_builder = context_builder
        self.host = host
        self.port = port
        self.max_upload_bytes = max_upload_bytes
        self._thread: Optional[threading.Thread] = None
        self._server = None

        routes = [
            Route("/", self._index, methods=["GET"]),
            Route("/upload", self._upload, methods=["POST"]),
            Route("/context", self._context, methods=["GET"]),
            Route("/health", self._health, methods=["GET"]),
        ]
        self.app = Starlette(routes=routes)

    async def _index(self, request: Request) -> HTMLResponse:
        from .templates import UPLOAD_HTML
        return HTMLResponse(UPLOAD_HTML)

    async def _upload(self, request: Request) -> JSONResponse:
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            return JSONResponse({"ok": False, "error": "No file provided"}, status_code=400)

        try:
            data = await upload.read()
            filename = upload.filename or "untitled"
            if len(data) > self.max_upload_bytes:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"File exceeds upload limit ({self.max_upload_bytes} bytes)",
                    },
                    status_code=413,
                )

            # Detect if binary (image) or text
            try:
                text = data.decode("utf-8")
                self.context_builder.add_text(text, label=filename)
            except UnicodeDecodeError:
                # Binary file - treat as image/data
                import base64
                encoded = base64.b64encode(data).decode("ascii")
                self.context_builder._sources.append({
                    "type": "upload",
                    "label": filename,
                    "content": encoded,
                })
                self.context_builder._current_chars += len(encoded)

            return JSONResponse({"ok": True, "filename": filename})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    async def _context(self, request: Request) -> JSONResponse:
        return JSONResponse({
            "source_count": self.context_builder.source_count,
            "token_estimate": self.context_builder.token_estimate,
            "sources": self.context_builder.list_sources(),
        })

    async def _health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    def start(self) -> str:
        """Start the server in a daemon thread. Returns the URL."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        """Signal the server to shut down."""
        if self._server is not None:
            self._server.should_exit = True
            self._thread = None
            self._server = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
