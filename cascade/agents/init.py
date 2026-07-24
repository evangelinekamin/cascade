"""Project scaffolding for .cascade/ directories.

Creates the directory structure and writes template files based on
the detected or chosen project type.
"""

import subprocess
from pathlib import Path
from typing import Callable, Optional

import yaml

from .templates import get_template


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd),
        capture_output=True, text=True, timeout=30,
    )


def _ensure_git_repo(path: Path) -> list[str]:
    """Init a git repo with a first commit so /solve can branch worktrees.

    No-op if already a repo. Sets a repo-local identity only when git has no
    global one (so the initial commit can't fail with 'identity unknown'),
    and respects .gitignore via ``git add -A``. Returns progress lines.
    """
    if (path / ".git").exists():
        return ["  skipped: git repo (already initialized)"]
    try:
        init = _run_git(["init", "-b", "main"], path)
        if init.returncode != 0:  # older git without -b
            _run_git(["init"], path)
            _run_git(["checkout", "-B", "main"], path)

        who = _run_git(["config", "user.email"], path)
        if not who.stdout.strip():
            _run_git(["config", "--local", "user.email", "cascade@localhost"], path)
            _run_git(["config", "--local", "user.name", "Cascade"], path)

        _run_git(["add", "-A"], path)
        commit = _run_git(
            ["commit", "-m", "chore: initial commit (cascade init)"], path,
        )
        if commit.returncode == 0:
            return ["  created: git repo (main) + initial commit"]
        # Nothing to commit (empty dir): still leave an empty root commit so
        # worktree lanes have a HEAD to branch from.
        _run_git(["commit", "--allow-empty", "-m", "chore: cascade init"], path)
        return ["  created: git repo (main), empty initial commit"]
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return [f"  skipped: git init failed ({exc})"]


def run_init(
    path: Path,
    project_type: str,
    print_fn: Optional[Callable[[str], None]] = None,
    *,
    enable_system_prompt: bool = True,
    enable_agents: bool = True,
    enable_context: bool = True,
    init_git: bool = True,
) -> str:
    """Scaffold a .cascade/ project directory.

    Args:
        path: Project root directory.
        project_type: One of the known project types (python, web, etc.).
        print_fn: Optional callback for progress messages.
        enable_system_prompt: Write system_prompt.md from template.
        enable_agents: Write agents.yaml from template.
        enable_context: Create the context/ subdirectory.

    Returns:
        A summary string of what was created/skipped.
    """
    path = path.resolve()
    cascade_dir = path / ".cascade"
    template = get_template(project_type)

    created: list[str] = []
    skipped: list[str] = []

    def _log(msg: str) -> None:
        if print_fn:
            print_fn(msg)

    # Create .cascade/ root
    if not cascade_dir.is_dir():
        cascade_dir.mkdir(parents=True, exist_ok=True)
        created.append(".cascade/")
    else:
        skipped.append(".cascade/ (already exists)")

    # system_prompt.md
    if enable_system_prompt:
        sp_path = cascade_dir / "system_prompt.md"
        if sp_path.exists():
            skipped.append("system_prompt.md (already exists)")
            _log("  skipped: system_prompt.md (already exists)")
        else:
            sp_path.write_text(template["system_prompt"] + "\n", encoding="utf-8")
            created.append("system_prompt.md")
            _log("  created: system_prompt.md")

    # agents.yaml (agents + workflows merged)
    if enable_agents:
        agents_path = cascade_dir / "agents.yaml"
        if agents_path.exists():
            skipped.append("agents.yaml (already exists)")
            _log("  skipped: agents.yaml (already exists)")
        else:
            agents_data = dict(template.get("agents", {}))
            workflows = template.get("workflows")
            if workflows:
                agents_data["workflows"] = dict(workflows)
            verify = template.get("verify")
            if verify:
                agents_data.setdefault("workflows", {})
                agents_data["workflows"]["verify"] = dict(verify)
            agents_path.write_text(
                yaml.dump(agents_data, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            created.append("agents.yaml")
            _log("  created: agents.yaml")

    # context/ directory
    if enable_context:
        ctx_dir = cascade_dir / "context"
        if ctx_dir.is_dir():
            skipped.append("context/ (already exists)")
            _log("  skipped: context/ (already exists)")
        else:
            ctx_dir.mkdir(parents=True, exist_ok=True)
            created.append("context/")
            _log("  created: context/")

    # Git repo: /solve and the other worktree lanes need a HEAD to branch from,
    # so a fresh project is made into a committed repo here (opt-out via flag).
    git_lines: list[str] = []
    if init_git:
        git_lines = _ensure_git_repo(path)
        for line in git_lines:
            _log(line)

    # Build summary
    lines = [f"Initialized .cascade/ for '{project_type}' project"]
    if created:
        lines.append(f"  created: {', '.join(created)}")
    if skipped:
        lines.append(f"  skipped: {', '.join(skipped)}")
    lines.extend(git_lines)

    return "\n".join(lines)
