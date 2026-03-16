"""Shannon autonomous pentesting framework integration.

Manages launching Shannon from within the Cascade REPL, including:
- Path resolution (config, env var, well-known defaults)
- Repo symlink setup (cwd -> shannon/repos/{name})
- Credential forwarding (ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN)
- Background subprocess management with prefixed output streaming
"""

import os
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable

_DEFAULT_PATHS = [
    Path("/home/evangeline/Projects/shannon"),
    Path.home() / "shannon",
]


class ShannonIntegration:
    """Wrapper around the Shannon CLI (./shannon script)."""

    def __init__(
        self,
        config_path: str = "",
        print_fn: Optional[Callable[[str], None]] = None,
    ):
        self._config_path = config_path
        self._process: Optional[subprocess.Popen] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._print = print_fn or print

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def find_path(self) -> Optional[Path]:
        """Resolve the Shannon installation directory.

        Priority: config path > $SHANNON_HOME > well-known defaults.
        Returns None if no valid installation found.
        """
        candidates = []

        if self._config_path:
            candidates.append(Path(self._config_path).expanduser())

        env_home = os.environ.get("SHANNON_HOME", "")
        if env_home:
            candidates.append(Path(env_home))

        candidates.extend(_DEFAULT_PATHS)

        for candidate in candidates:
            script = candidate / "shannon"
            if script.is_file():
                return candidate

        return None

    def ensure_repo(self, shannon_path: Path, repo_name: str) -> Path:
        """Create a symlink from cwd into shannon/repos/{repo_name}.

        Returns the repos directory path. Skips if link already exists.
        """
        repos_dir = shannon_path / "repos"
        repos_dir.mkdir(parents=True, exist_ok=True)

        link_path = repos_dir / repo_name
        cwd = Path.cwd()

        if link_path.is_symlink():
            if link_path.resolve() == cwd.resolve():
                return link_path
            # Different target -- remove stale link
            link_path.unlink()
        elif link_path.exists():
            # Real directory with that name already exists
            return link_path

        link_path.symlink_to(cwd)
        self._print(f"[shannon] Linked {cwd} -> {link_path}")
        return link_path

    def build_env(self) -> dict[str, str]:
        """Build environment variables for the Shannon subprocess.

        Forwards Claude credentials and sets max output tokens.
        """
        env = dict(os.environ)

        # Forward Claude credentials
        for key in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
            val = os.environ.get(key, "")
            if val:
                env[key] = val
                break
        else:
            # Try to detect from Claude CLI credentials
            try:
                from ..auth import detect_claude
                cred = detect_claude()
                if cred:
                    token = cred.token
                    if token.startswith("sk-ant-"):
                        env["ANTHROPIC_API_KEY"] = token
                    else:
                        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
            except Exception:
                pass

        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
        return env

    def cmd_start(
        self,
        url: str,
        repo: str = "",
        extra_args: Optional[list[str]] = None,
    ) -> bool:
        """Launch Shannon in the background.

        Returns True if started successfully, False otherwise.
        """
        if self.running:
            self._print("[shannon] Already running. Use /shannon stop first.")
            return False

        shannon_path = self.find_path()
        if shannon_path is None:
            self._print(
                "[shannon] Shannon not found. Set SHANNON_HOME or "
                "integrations.shannon.path in config."
            )
            return False

        if not repo:
            repo = Path.cwd().name

        self.ensure_repo(shannon_path, repo)

        cmd = [
            str(shannon_path / "shannon"),
            "start",
            f"URL={url}",
            f"REPO={repo}",
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = self.build_env()

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(shannon_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            self._print(f"[shannon] Failed to start: {exc}")
            return False

        self._stream_thread = threading.Thread(
            target=self._stream_output,
            daemon=True,
        )
        self._stream_thread.start()

        self._print(f"[shannon] Started: {url} (repo={repo}, pid={self._process.pid})")
        return True

    def cmd_logs(self, workflow_id: str = "") -> None:
        """Show Shannon logs (foreground, short-lived)."""
        shannon_path = self.find_path()
        if shannon_path is None:
            self._print("[shannon] Shannon not found.")
            return

        cmd = [str(shannon_path / "shannon"), "logs"]
        if workflow_id:
            cmd.append(workflow_id)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(shannon_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                self._print(result.stdout.rstrip())
            if result.stderr:
                self._print(result.stderr.rstrip())
        except subprocess.TimeoutExpired:
            self._print("[shannon] Logs command timed out.")
        except OSError as exc:
            self._print(f"[shannon] Error: {exc}")

    def cmd_workspaces(self) -> None:
        """List Shannon workspaces (foreground, short-lived)."""
        shannon_path = self.find_path()
        if shannon_path is None:
            self._print("[shannon] Shannon not found.")
            return

        cmd = [str(shannon_path / "shannon"), "workspaces"]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(shannon_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                self._print(result.stdout.rstrip())
            if result.stderr:
                self._print(result.stderr.rstrip())
        except subprocess.TimeoutExpired:
            self._print("[shannon] Workspaces command timed out.")
        except OSError as exc:
            self._print(f"[shannon] Error: {exc}")

    def cmd_stop(self) -> None:
        """Stop the active Shannon process."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._print("[shannon] Stopped.")
        else:
            # Try running ./shannon stop directly
            shannon_path = self.find_path()
            if shannon_path:
                try:
                    subprocess.run(
                        [str(shannon_path / "shannon"), "stop"],
                        cwd=str(shannon_path),
                        timeout=15,
                    )
                    self._print("[shannon] Stop command sent.")
                except (OSError, subprocess.TimeoutExpired):
                    self._print("[shannon] No active process to stop.")
            else:
                self._print("[shannon] No active process to stop.")

        self._process = None
        self._stream_thread = None

    def _stream_output(self) -> None:
        """Stream subprocess stdout to console with [shannon] prefix."""
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        try:
            for line in proc.stdout:
                self._print(f"[shannon] {line.rstrip()}")
        except (ValueError, OSError):
            pass
