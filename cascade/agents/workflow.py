"""Workflow definitions and runner.

A workflow chains multiple agent steps, piping each agent's output into
the next step's ``{input}`` placeholder.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

from ..hooks import HookContext, HookEvent

if TYPE_CHECKING:
    from .runner import AgentRunner
    from .schema import AgentDef


@dataclass(frozen=True)
class WorkflowStep:
    """A single step in a workflow pipeline."""

    agent: str
    prompt_template: str = "{input}"
    label: str = ""


@dataclass(frozen=True)
class WorkflowDef:
    """Immutable definition of a multi-step workflow."""

    name: str
    description: str = ""
    steps: tuple[WorkflowStep, ...] = ()


def load_workflows_from_dict(data: dict[str, Any]) -> dict[str, WorkflowDef]:
    """Parse workflow definitions from the ``workflows`` section of agents.yaml.

    Expected format::

        feature:
          description: "Plan then review"
          steps:
            - agent: planner
              label: "Planning"
            - agent: reviewer
              prompt_template: "Review this plan:\\n{input}"
              label: "Review"
    """
    workflows: dict[str, WorkflowDef] = {}

    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        raw_steps = entry.get("steps")
        if not isinstance(raw_steps, list):
            continue
        try:
            steps = tuple(
                WorkflowStep(
                    agent=str(s["agent"]),
                    prompt_template=str(s.get("prompt_template", "{input}")),
                    label=str(s.get("label", "")),
                )
                for s in raw_steps
                if isinstance(s, dict) and "agent" in s
            )
            if not steps:
                continue
            workflows[name] = WorkflowDef(
                name=name,
                description=str(entry.get("description", "")),
                steps=steps,
            )
        except Exception:
            continue

    return workflows


class WorkflowRunner:
    """Executes a WorkflowDef by chaining AgentRunner calls."""

    def __init__(
        self,
        agent_runner: "AgentRunner",
        agents: dict[str, "AgentDef"],
        print_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._runner = agent_runner
        self._agents = agents
        self._print = print_fn or (lambda _s: None)

    def run(self, workflow: WorkflowDef, initial_prompt: str) -> str:
        """Run all steps sequentially, returning the final output."""
        current = initial_prompt
        hook_runner = getattr(
            getattr(self._runner, "_app", None),
            "hook_runner",
            None,
        )
        if hook_runner is not None:
            start = hook_runner.emit(
                HookEvent.WORKFLOW_START,
                HookContext(
                    event=HookEvent.WORKFLOW_START.value,
                    workflow=workflow.name,
                    prompt=current,
                ),
            )
            if start is not None:
                if start.block:
                    raise RuntimeError(f"Workflow blocked by hook: {start.reason}")
                if start.transformed_value is not None:
                    if not isinstance(start.transformed_value, str):
                        raise RuntimeError(
                            "Workflow hook returned an invalid prompt (expected text)"
                        )
                    current = start.transformed_value

        try:
            for i, step in enumerate(workflow.steps):
                agent = self._agents.get(step.agent)
                if agent is None:
                    raise RuntimeError(
                        f"Workflow '{workflow.name}' step {i + 1} references "
                        f"unknown agent '{step.agent}'"
                    )

                label = step.label or f"Step {i + 1}: {step.agent}"
                self._print(f"[{label}]")

                prompt = step.prompt_template.replace("{input}", current)
                # ``current`` is already present in the rendered prompt. Passing it
                # again as a system-prompt layer doubled every intermediate output
                # in the next model request.
                current = self._runner.run(agent, prompt)
        except Exception as exc:
            if hook_runner is not None:
                hook_runner.emit(
                    HookEvent.ON_ERROR,
                    HookContext(
                        event=HookEvent.ON_ERROR.value,
                        workflow=workflow.name,
                        prompt=initial_prompt,
                        error=str(exc),
                    ),
                )
            raise

        if hook_runner is not None:
            end = hook_runner.emit(
                HookEvent.WORKFLOW_END,
                HookContext(
                    event=HookEvent.WORKFLOW_END.value,
                    workflow=workflow.name,
                    prompt=initial_prompt,
                    response=current,
                ),
            )
            if end is not None:
                if end.block:
                    raise RuntimeError(f"Workflow result blocked by hook: {end.reason}")
                if end.transformed_value is not None:
                    if not isinstance(end.transformed_value, str):
                        raise RuntimeError(
                            "Workflow end hook returned an invalid response "
                            "(expected text)"
                        )
                    current = end.transformed_value

        return current
