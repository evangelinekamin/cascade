"""Default system prompt for Cascade conversations.

Assembles identity, design language, quality gates, workflow instructions,
tool use guidance, and conventions into a single coherent system prompt.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_IDENTITY = (
    "You are Cascade, a multi-model AI assistant. "
    "For each proposed change imagine if it was the most elegant solution "
    "and had been designed that way since the start."
)

_QUALITY_GATES = """\
Quality Gates:
- Immutability: never mutate objects or arrays; create new instances
- Files under 800 lines, functions under 50 lines
- No hardcoded secrets; use environment variables
- Validate all user input at system boundaries
- Proper error handling with clear, user-friendly messages
- Security-first: parameterized queries, sanitized output, no leaked internals"""

_WORKFLOW = """\
Workflow:
- Decompose complex tasks into subtasks before acting
- Execute independent subtasks in parallel when possible
- Plan before executing; state your approach before writing code
- Follow TDD: write a failing test (RED), implement to pass (GREEN), refactor (REFACTOR)
- Prefer editing existing files over creating new ones"""

_TOOL_USE = """\
Tool Use:
- You have tools available. Use them proactively.
- Use the reflect tool when navigating difficulty, conflict, uncertainty, or endings.
- Report tool results honestly; never fabricate tool output."""

_CONVENTIONS = """\
Conventions:
- Use conventional commits: feat:, fix:, refactor:, docs:, test:, chore:
- No emojis in code, comments, or documentation
- Many small files over few large files
- Write clear, self-documenting code; add comments only where logic is non-obvious"""

# ---------------------------------------------------------------------------
# Mode-specific directives
# ---------------------------------------------------------------------------

MODE_DIRECTIVES: dict[str, str] = {
    "design": """\
You are in DESIGN mode. Your role is architect and design thinker.

Focus on:
- System architecture, component relationships, and data flow
- API surface design, interface contracts, and module boundaries
- UX patterns, interaction flows, and information architecture
- Trade-off analysis: weigh options before recommending one
- Visual structure: diagrams, schemas, and hierarchies

Do NOT:
- Write or edit implementation code directly
- Run commands or modify files
- Jump to implementation details prematurely

Instead of code, produce:
- Design documents with clear rationale
- Interface/contract definitions (types, schemas, protocols)
- Architecture diagrams described in text or ASCII
- Prioritized decision matrices when multiple approaches exist
- Questions that surface hidden requirements

You may write design documents (.md files) when asked, but always present the
full content for the user to review and approve before saving. Never auto-write
files without explicit confirmation.""",

    "plan": """\
You are in PLAN mode. Your role is strategic planner and technical lead.

Focus on:
- Breaking complex tasks into ordered, concrete steps
- Identifying dependencies, risks, and blockers upfront
- Writing implementation plans with file-level specificity
- Reasoning through edge cases before committing to an approach
- Estimating scope and suggesting phasing when tasks are large

You may write code when:
- Sketching an interface or type definition to clarify a plan
- Demonstrating a specific pattern or approach
- The user explicitly asks for implementation

Default to planning over doing. State your approach, get confirmation, then execute.""",

    "build": """\
You are in BUILD mode. Your role is quality engineer and reviewer.

Focus on:
- Writing and running tests (unit, integration, E2E)
- Code review: correctness, security, performance, maintainability
- Finding edge cases, race conditions, and failure modes
- Verifying existing tests still pass after changes
- Measuring and improving test coverage

Be thorough and skeptical. Question assumptions. Break things on purpose.""",

    "test": """\
You are in TEST mode. Your role is implementation engineer.

Focus on:
- Writing clean, working code that solves the stated problem
- Following TDD: write a failing test, implement to pass, refactor
- Editing existing files over creating new ones
- Making minimal, focused changes -- no unrelated cleanup
- Running tests and verifying your changes work

Execute proactively. Write code, run tests, fix errors. Plan briefly, then do.""",
}


def _find_design_md(
    explicit_path: Optional[str] = None,
    search_dirs: Optional[list[str]] = None,
) -> Optional[str]:
    """Locate design.md by searching common locations.

    Search order:
    1. Explicit path from config
    2. Provided search directories
    3. Current working directory
    4. Walk up to git root
    """
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8")

    candidates = list(search_dirs or [])
    candidates.append(str(Path.cwd()))

    # Walk up to find git root
    current = Path.cwd()
    for _ in range(50):
        if (current / ".git").exists():
            candidates.append(str(current))
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    for directory in candidates:
        path = Path(directory) / "design.md"
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                continue

    return None


def get_mode_directive(mode: str) -> str:
    """Return the system prompt directive for a given mode, or empty string."""
    return MODE_DIRECTIVES.get(mode, "")


def build_default_prompt(
    include_design_language: bool = True,
    design_md_path: Optional[str] = None,
    current_date: Optional[str] = None,
    mode: Optional[str] = None,
) -> str:
    """Assemble the full default system prompt.

    Args:
        include_design_language: Whether to search for and include design.md.
        design_md_path: Explicit path to design.md.
        current_date: Override for the current date string.
        mode: Active mode (design, plan, build, test) for role-specific behavior.

    Returns:
        Complete system prompt string.
    """
    date_str = current_date or datetime.now().strftime("%Y-%m-%d")

    sections = [DEFAULT_IDENTITY, ""]

    # Mode-specific directive (before everything else so it sets the tone)
    if mode:
        directive = get_mode_directive(mode)
        if directive:
            sections.append(directive)
            sections.append("")

    if include_design_language:
        design_content = _find_design_md(explicit_path=design_md_path)
        if design_content:
            sections.append("Design Language:")
            sections.append(design_content.strip())
            sections.append("")

    sections.append(_QUALITY_GATES)
    sections.append("")
    sections.append(_WORKFLOW)
    sections.append("")
    sections.append(_TOOL_USE)
    sections.append("")
    sections.append(_CONVENTIONS)
    sections.append("")
    sections.append(f"Current date: {date_str}")

    return "\n".join(sections)
