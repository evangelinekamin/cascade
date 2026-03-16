"""Built-in agent commands that work without agents.yaml.

These use the active provider directly -- no named agent required.
"""

import subprocess
import shlex
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..cli import CascadeCore


def _run_cmd(cmd: str, timeout: int = 120) -> tuple[str, int]:
    """Run a shell command and return (output, returncode).

    Captures both stdout and stderr.  Returns partial output on timeout.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output = output + "\n" + result.stderr if output else result.stderr
        return output.strip(), result.returncode
    except subprocess.TimeoutExpired as e:
        partial = ""
        if e.stdout:
            partial = e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="replace")
        return f"[timed out after {timeout}s]\n{partial}".strip(), -1


_REVIEW_PROMPT = """\
You are a senior code reviewer.  Review the following git diff and produce a
structured report.  Classify each finding as CRITICAL, HIGH, MEDIUM, or LOW.

Focus on:
- Bugs and logic errors
- Security vulnerabilities
- Performance issues
- Readability and maintainability

If the diff is clean, say so briefly.

```diff
{diff}
```"""

_VERIFY_PROMPT = """\
Below are the results of running project verification commands.
Summarize the overall status (PASS / FAIL / PARTIAL) and call out any
failures or warnings that need attention.

{results}"""


def cmd_verify(
    app: "CascadeCore",
    config: dict,
    print_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """Run configured verification commands and summarize results.

    ``config`` is the ``workflows.verify`` dict from config.yaml::

        verify:
          lint: "ruff check ."
          test: "python -m pytest -x -q"
          build: ""
          audit: ""
    """
    _print = print_fn or (lambda _s: None)
    sections = []

    for label in ("lint", "test", "build", "audit"):
        cmd = config.get(label, "")
        if not cmd:
            continue
        _print(f"Running {label}: {cmd}")
        output, rc = _run_cmd(cmd)
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        sections.append(f"## {label} -- {status}\n```\n{output}\n```")

    if not sections:
        return "No verification commands configured."

    combined = "\n\n".join(sections)
    prompt = _VERIFY_PROMPT.format(results=combined)

    prov = app.get_provider()
    system = app.prompt_pipeline.build() or None
    return prov.ask_single(prompt, system)


def cmd_review(
    app: "CascadeCore",
    base_ref: str = "",
    print_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """Run ``git diff`` and ask the provider for a structured code review."""
    _print = print_fn or (lambda _s: None)

    diff_cmd = f"git diff {base_ref}" if base_ref else "git diff"
    if base_ref:
        diff_cmd = f"git diff {shlex.quote(base_ref)}"
    _print(f"Running: {diff_cmd}")
    diff_output, rc = _run_cmd(diff_cmd)

    if rc != 0:
        return f"git diff failed (exit {rc}): {diff_output}"
    if not diff_output.strip():
        return "No changes to review."

    prompt = _REVIEW_PROMPT.format(diff=diff_output)
    prov = app.get_provider()
    system = app.prompt_pipeline.build() or None
    return prov.ask_single(prompt, system)


def cmd_checkpoint(
    app: "CascadeCore",
    label: str = "checkpoint",
    test_cmd: str = "",
    print_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """Run tests, then commit if they pass.

    ``test_cmd`` defaults to the verify.test config entry or
    ``python -m pytest -x -q``.
    """
    _print = print_fn or (lambda _s: None)

    if not test_cmd:
        wf_config = app.config.data.get("workflows", {}).get("verify", {})
        test_cmd = wf_config.get("test", "python -m pytest -x -q")

    _print(f"Running tests: {test_cmd}")
    output, rc = _run_cmd(test_cmd)

    if rc != 0:
        return f"Tests failed (exit {rc}) -- checkpoint skipped.\n{output}"

    # Stage all and commit
    _run_cmd("git add -A")
    commit_msg = f"checkpoint: {label}"
    commit_output, commit_rc = _run_cmd(f"git commit -m {shlex.quote(commit_msg)}")

    if commit_rc != 0:
        return f"Commit failed: {commit_output}"

    return f"Checkpoint committed: {commit_msg}\n{output}"
