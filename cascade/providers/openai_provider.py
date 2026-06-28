"""OpenAI provider for GPT-4o, o1, o3, and Codex models."""

import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Iterator, TYPE_CHECKING
import httpx
from .base import BaseProvider, ProviderConfig, Message, ToolEventCallback
from ._cli_proxy import CLIProxyConfig, CodexEventHandler, stream_cli_proxy
from ._openai_tools import openai_ask_with_tools
from .registry import register_provider

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


_AGENTIC_HINTS = (
    "apply the change",
    "apply the winner",
    "create ",
    "delete ",
    "edit ",
    "fix ",
    "implement ",
    "make your changes",
    "modify ",
    "patch ",
    "refactor ",
    "rename ",
    "run tests",
    "save ",
    "update ",
    "write code",
)

_NON_AGENTIC_HINTS = (
    "analyze ",
    "audit ",
    "compare ",
    "do not edit",
    "don't edit",
    "explain ",
    "rank ",
    "review ",
    "summarize ",
    "what changed",
    "which provider won",
)

_REPO_SCOPE_HINTS = (
    "codebase",
    "project",
    "repo",
    "repository",
    "workspace",
)

_PATH_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".json", ".jsx", ".kt", ".md", ".mjs", ".py",
    ".rs", ".scss", ".sh", ".sql", ".svelte", ".toml", ".ts", ".tsx",
    ".txt", ".yaml", ".yml",
}

_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".svelte-kit",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "venv",
}

_SYNTHETIC_CONTEXT_PREFIXES = (
    "[Prior session context]",
    "[Context from previous model interactions]",
    "[Conversation summary]",
    "[Response from ",
)

_FILENAME_INDEX_TTL_SECONDS = 30.0


@register_provider("openai")
class OpenAIProvider(BaseProvider):
    """OpenAI API provider - supports custom base_url for Azure/proxies."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.client = httpx.Client(timeout=60.0)
        self._codex_bin = shutil.which("codex")
        self._codex_sessions: dict[str, str] = {}
        self._filename_index_cache: dict[str, tuple[float, dict[str, Path]]] = {}
        self._line_count_cache: dict[str, tuple[int, int, int]] = {}
        self._use_oauth_cli = self._looks_like_jwt(config.api_key)
        self._use_cli_proxy = (
            self._use_oauth_cli
            and bool(self._codex_bin)
            and self.base_url.rstrip("/") == "https://api.openai.com/v1"
        )
        default_activity = "1" if self._use_cli_proxy else "0"
        self._emit_activity = (
            os.getenv(
                "CASCADE_OPENAI_ACTIVITY",
                os.getenv("CASCADE_CODEX_ACTIVITY", default_activity),
            ).lower()
            not in ("0", "false", "no", "off")
        )
        self._reuse_cli_sessions = (
            os.getenv("CASCADE_CODEX_REUSE_SESSION", "1").lower()
            not in ("0", "false", "no", "off")
        )

    @staticmethod
    def _looks_like_jwt(token: str) -> bool:
        token = (token or "").strip()
        return token.startswith("eyJ") and token.count(".") >= 2

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_cli_prompt(
        self,
        messages: list[Message],
        system: Optional[str],
    ) -> str:
        full_prompt = self._condense_for_cli(messages)
        if system:
            condensed = self._condense_system_for_cli(system)
            if condensed:
                full_prompt = f"System instructions:\n{condensed}\n\n{full_prompt}"
        return full_prompt

    @staticmethod
    def _latest_user_prompt(messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return messages[-1]["content"] if messages else ""

    def _should_use_repo_workspace(
        self,
        messages: list[Message],
        system: Optional[str],
        workdir: str,
    ) -> bool:
        prompt = self._latest_user_prompt(messages).lower()
        if self._looks_agentic_request(prompt, system):
            return True
        if any(hint in prompt for hint in _NON_AGENTIC_HINTS):
            return False

        referenced_files = self._resolve_referenced_files(messages, workdir)
        if referenced_files:
            return False

        return any(hint in prompt for hint in _REPO_SCOPE_HINTS)

    @staticmethod
    def _looks_agentic_request(prompt: str, system: Optional[str]) -> bool:
        system_text = (system or "").lower()
        combined = f"{system_text}\n{prompt}"

        if "do not edit" in combined or "don't edit" in combined:
            return False
        return any(hint in combined for hint in _AGENTIC_HINTS)

    @staticmethod
    def _clean_path_token(token: str) -> str:
        return token.strip().strip("`'\".,:;()[]{}")

    def _candidate_path_tokens(self, text: str) -> list[str]:
        candidates: list[str] = []
        for match in re.finditer(r"(?:/|\.?/)?(?:[\w.+\-]+/)*[\w.+\-]+\.[A-Za-z0-9]+", text):
            token = self._clean_path_token(match.group(0))
            if not token:
                continue
            suffix = Path(token).suffix.lower()
            if suffix not in _PATH_EXTENSIONS:
                continue
            candidates.append(token)
        return candidates

    def _resolve_candidate_path(self, token: str, root: Path) -> Optional[Path]:
        candidate = Path(token)
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve()
            except FileNotFoundError:
                return None
            return resolved if resolved.is_file() else None

        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return resolved
        return None

    def _find_filename_match(self, filename: str, root: Path) -> Optional[Path]:
        index = self._get_filename_index(root)
        return index.get(filename)

    def _get_filename_index(self, root: Path) -> dict[str, Path]:
        root_key = str(root.resolve())
        now = time.monotonic()
        cached = self._filename_index_cache.get(root_key)
        if cached is not None and (now - cached[0]) < _FILENAME_INDEX_TTL_SECONDS:
            return cached[1]

        index: dict[str, Path] = {}
        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
            for filename in filenames:
                if filename in index:
                    continue
                path = Path(current_root, filename)
                if path.is_file():
                    index[filename] = path

        self._filename_index_cache[root_key] = (now, index)
        return index

    def _resolve_referenced_files(self, messages: list[Message], workdir: str) -> list[Path]:
        root = Path(workdir).resolve()
        resolved: list[Path] = []
        seen: set[Path] = set()

        for message in reversed(messages[-8:]):
            for token in self._candidate_path_tokens(message.get("content", "")):
                path = self._resolve_candidate_path(token, root)
                if path is None or path in seen:
                    continue
                seen.add(path)
                resolved.append(path)
                if len(resolved) >= 8:
                    return resolved

        if resolved:
            return resolved

        for message in reversed(messages[-6:]):
            for token in re.findall(r"\b[\w.+\-]+\.[A-Za-z0-9]+\b", message.get("content", "")):
                cleaned = self._clean_path_token(token)
                suffix = Path(cleaned).suffix.lower()
                if suffix not in _PATH_EXTENSIONS:
                    continue
                path = self._find_filename_match(cleaned, root)
                if path is None or path in seen:
                    continue
                seen.add(path)
                resolved.append(path)
                if len(resolved) >= 8:
                    return resolved

        return resolved

    def _line_count(self, path: Path) -> int:
        try:
            stat = path.stat()
        except Exception:
            return 0

        cache_key = str(path.resolve())
        cached = self._line_count_cache.get(cache_key)
        if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                count = sum(1 for _ in handle)
        except Exception:
            return 0

        self._line_count_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, count)
        return count

    @staticmethod
    def _focused_workspace_root() -> Path:
        raw = os.getenv("CASCADE_CODEX_WORKSPACE_ROOT", "")
        if raw:
            root = Path(raw).expanduser()
        else:
            root = Path.home() / ".cache" / "cascade" / "codex-workspaces"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _reset_workspace_dir(root: Path) -> None:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _relative_workspace_paths(source_root: Path, files: list[Path]) -> list[str]:
        relative_paths: list[str] = []
        for path in files:
            try:
                relative_paths.append(path.relative_to(source_root).as_posix())
            except ValueError:
                relative_paths.append(path.name)
        return relative_paths

    def _focused_workspace_key(self, source_root: Path, relative_paths: list[str]) -> str:
        material = "\0".join(
            ["focused", str(source_root), self.config.model, *relative_paths]
        ).encode("utf-8")
        return hashlib.sha1(material, usedforsecurity=False).hexdigest()

    def _system_fingerprint(self, system: Optional[str]) -> str:
        condensed = self._condense_system_for_cli(system or "")
        if not condensed:
            return "none"
        digest = hashlib.sha1(condensed.encode("utf-8"), usedforsecurity=False).hexdigest()
        return digest[:12]

    def _mirror_focused_files(
        self,
        source_root: Path,
        scratch_root: Path,
        files: list[Path],
    ) -> list[str]:
        mirrored: list[str] = []
        for path in files:
            try:
                relative = path.relative_to(source_root)
            except ValueError:
                relative = Path(path.name)
            target = scratch_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            mirrored.append(relative.as_posix())

        note_lines = [
            "# Focused Codex workspace",
            "",
            "This temporary workspace contains only the files Cascade selected for this request.",
            f"Source root: {source_root}",
        ]
        if mirrored:
            note_lines.extend(["", "Mirrored files:"])
            note_lines.extend(f"- {rel}" for rel in mirrored)
        else:
            note_lines.extend(["", "No repository files were mirrored for this request."])
        (scratch_root / "CASCADE_WORKSPACE.md").write_text(
            "\n".join(note_lines) + "\n",
            encoding="utf-8",
        )
        return mirrored

    def _augment_prompt_for_focused_workspace(
        self,
        prompt: str,
        files: list[Path],
        source_root: Path,
    ) -> str:
        note_lines = [
            "Workspace note:",
            "- You are running in a temporary focused workspace, not the full repository.",
            f"- Original repository root: {source_root}",
        ]
        if files:
            note_lines.append("- Cascade mirrored only these referenced files into the workspace:")
            for path in files:
                try:
                    relative = path.relative_to(source_root).as_posix()
                except ValueError:
                    relative = path.name
                line_count = self._line_count(path)
                suffix = f" ({line_count} lines)" if line_count else ""
                note_lines.append(f"  - {relative}{suffix}")
        else:
            note_lines.append("- No repository files were mirrored for this request.")
        note_lines.append(
            "- If you need broader repository context, say that explicitly instead of assuming it."
        )
        return "\n".join(note_lines) + "\n\n" + prompt

    @staticmethod
    def _has_synthetic_context(messages: list[Message]) -> bool:
        for message in messages[:-1]:
            content = str(message.get("content", "")).lstrip()
            if any(content.startswith(prefix) for prefix in _SYNTHETIC_CONTEXT_PREFIXES):
                return True
        return False

    def _build_resume_prompt(
        self,
        messages: list[Message],
        system: Optional[str],
        workspace_mode: str,
        full_prompt: str,
        session_id: str | None,
    ) -> str:
        if workspace_mode != "focused" or not session_id:
            return full_prompt
        if self._has_synthetic_context(messages):
            return full_prompt

        current_prompt = self._latest_user_prompt(messages).strip()
        if not current_prompt:
            return full_prompt

        parts: list[str] = []
        condensed = self._condense_system_for_cli(system or "")
        if condensed:
            parts.append(f"System instructions:\n{condensed}")
        parts.append(
            "You are continuing an existing Cascade Codex session in the same focused workspace."
        )
        parts.append(f"Current request:\n{current_prompt}")
        return "\n\n".join(parts)

    @contextmanager
    def _cli_workspace(
        self,
        messages: list[Message],
        system: Optional[str],
    ):
        prompt = self._build_cli_prompt(messages, system)
        workdir = self.get_working_directory()
        agentic = self._looks_agentic_request(self._latest_user_prompt(messages).lower(), system)
        system_fingerprint = self._system_fingerprint(system)
        if self._should_use_repo_workspace(messages, system, workdir):
            workspace_mode = "repo-write" if agentic else "repo-read"
            session_key = (
                f"{workspace_mode}:{Path(workdir).resolve()}:{self.config.model}:{system_fingerprint}"
            )
            yield prompt, workdir, workspace_mode, session_key
            return

        referenced_files = self._resolve_referenced_files(messages, workdir)
        prompt_lower = self._latest_user_prompt(messages).lower()
        if not referenced_files and any(hint in prompt_lower for hint in _REPO_SCOPE_HINTS):
            workspace_mode = "repo-read"
            session_key = (
                f"{workspace_mode}:{Path(workdir).resolve()}:{self.config.model}:{system_fingerprint}"
            )
            yield prompt, workdir, workspace_mode, session_key
            return

        source_root = Path(workdir).resolve()
        relative_paths = self._relative_workspace_paths(source_root, referenced_files)
        workspace_key = self._focused_workspace_key(source_root, relative_paths)
        scratch_root = self._focused_workspace_root() / workspace_key
        self._reset_workspace_dir(scratch_root)
        self._mirror_focused_files(source_root, scratch_root, referenced_files)
        focused_prompt = self._augment_prompt_for_focused_workspace(
            prompt,
            referenced_files,
            source_root,
        )
        session_key = f"focused:{workspace_key}:{system_fingerprint}"
        yield focused_prompt, str(scratch_root), "focused", session_key

    def _build_cli_cmd(
        self,
        prompt: str,
        workdir: str,
        workspace_mode: str,
        session_id: str | None = None,
    ) -> list[str]:
        if session_id:
            cmd = [self._codex_bin, "exec", "resume", session_id, "--json"]
            if workspace_mode == "focused":
                cmd.append("--skip-git-repo-check")
        else:
            cmd = [self._codex_bin, "exec", "--json", "--cd", workdir]
            if workspace_mode == "focused":
                cmd.extend(["--skip-git-repo-check", "--sandbox", "read-only"])
            elif workspace_mode == "repo-write":
                cmd.extend(["--sandbox", "workspace-write"])

        if self.config.model:
            cmd.extend(["--model", self.config.model])
        cmd.append(prompt)
        return cmd

    def _stream_via_cli(
        self,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream assistant text by proxying through ``codex exec --json``."""
        if not self._codex_bin:
            yield "Error: codex CLI not found in PATH for OAuth mode."
            return

        with self._cli_workspace(messages, system) as (full_prompt, workdir, workspace_mode, session_key):
            attempts = 2 if self._reuse_cli_sessions else 1
            for attempt in range(attempts):
                session_id = (
                    self._codex_sessions.get(session_key)
                    if self._reuse_cli_sessions
                    else None
                )
                prompt = self._build_resume_prompt(
                    messages,
                    system,
                    workspace_mode,
                    full_prompt,
                    session_id,
                )
                cmd = self._build_cli_cmd(
                    prompt,
                    workdir,
                    workspace_mode,
                    session_id=session_id,
                )
                handler = CodexEventHandler()
                cfg = CLIProxyConfig(
                    binary=self._codex_bin,
                    cli_name="codex",
                    cmd_args=cmd,
                    cwd=workdir,
                )
                try:
                    yield from stream_cli_proxy(cfg, handler, self._emit_activity)
                except RuntimeError:
                    if session_id and attempt == 0 and not handler.saw_text:
                        self._codex_sessions.pop(session_key, None)
                        continue
                    raise

                if handler.thread_id:
                    self._codex_sessions[session_key] = handler.thread_id
                elif session_id:
                    self._codex_sessions.setdefault(session_key, session_id)
                if handler.last_usage:
                    self._last_usage = handler.last_usage
                return

    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        """Get a complete response from OpenAI."""
        return "".join(self.stream(messages, system))

    def stream(self, messages: list[Message], system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from OpenAI."""
        self._last_usage = None
        self.reset_activity_state()
        if self._use_cli_proxy:
            yield from self._filter_activity(self._stream_via_cli(messages, system))
            return
        if self._use_oauth_cli and not self._use_cli_proxy:
            if not self._codex_bin:
                yield "Error: Codex OAuth token detected, but codex CLI is not in PATH."
            else:
                yield (
                    "Error: Codex OAuth token requires the default OpenAI base URL "
                    "(https://api.openai.com/v1)."
                )
            return

        try:
            url = f"{self.base_url}/chat/completions"

            api_messages = []
            if system:
                api_messages.append({"role": "system", "content": system})
            api_messages.extend(
                {"role": m["role"], "content": m["content"]}
                for m in messages
            )

            payload = {
                "model": self.config.model,
                "messages": api_messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens:
                payload["max_tokens"] = self.config.max_tokens

            with self.client.stream("POST", url, json=payload, headers=self._headers()) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        usage = data.get("usage")
                        if usage:
                            self._last_usage = (
                                usage.get("prompt_tokens", 0),
                                usage.get("completion_tokens", 0),
                            )
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"Error: {str(e)}"

    def ask_with_tools(
        self,
        messages: list[Message],
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
    ) -> tuple[str, list[dict]]:
        """OpenAI-native tool calling."""
        if self._use_cli_proxy:
            return self.ask(messages, system), []
        self._last_usage = None

        return openai_ask_with_tools(
            client=self.client,
            url=f"{self.base_url}/chat/completions",
            headers=self._headers(),
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=messages,
            tools=tools,
            system=system,
            max_rounds=max_rounds,
            on_tool_event=on_tool_event,
            on_usage=lambda usage: setattr(self, "_last_usage", usage),
        )

    def compare(self, prompt: str, system: Optional[str] = None) -> dict:
        """Generate comparison data."""
        response = self.ask_single(prompt, system)
        return {
            "provider": self.name,
            "model": self.config.model,
            "response": response,
            "length": len(response),
        }

    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass
