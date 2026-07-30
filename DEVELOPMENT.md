# Developing Cascade

Cascade is a Python 3.10+ terminal coding agent. The `cascade` entry point
starts the Textual application; `cascade-cli` exposes one-shot and diagnostic
commands.

## Architecture

The important package boundaries are:

- `cascade/screens`, `cascade/widgets`, `cascade/app.py`: Textual application
  and UI state.
- `cascade/providers`: provider adapters and native tool-calling loops.
- `cascade/tools`: schema generation, permissions, execution, output
  filtering, and loop guardrails.
- `cascade/hooks`: lifecycle events, matchers, configuration loading, and
  Python/subprocess hook execution.
- `cascade/context`, `cascade/episodes.py`, `cascade/conversation.py`: context
  budgets, episode compaction, and cross-provider memory.
- `cascade/agents`: named agents and sequential workflows.
- `cascade/swarm`: automatic routing, isolated solve/competition/pipeline/
  fan-out workflows, verification, cancellation, and durable run records.
- `cascade/evaluation.py`, `cascade/harness.py`: real repository evaluations
  and deterministic local harness measurements.
- `cascade/capabilities.py`, `cascade/receipts.py`: cached runtime diagnostics
  and portable session/run evidence.
- `cascade/history`: SQLite-backed sessions, branching, and generated IDs.
- `cascade/plugins`: built-in file, execution, reflection, and web tools.

`CascadeCore` in `cascade/cli.py` owns configuration, providers, hooks,
permissions, tools, agents, and workflows. Provider clones used by agent and
swarm lanes must inherit both the shared hook runner and permission engine.

## Lifecycle and tool invariants

Keep these properties intact when extending the agent harness:

1. Permission rules inspect the original tool call before hooks run.
2. If a hook transforms tool arguments, permissions inspect the transformed
   call again.
3. `tool_result` fires for success, denial, validation failure, unknown tool,
   and handler failure.
4. Hook transforms chain in priority order; a later hook sees the previous
   transform.
5. Policy hooks on `tool_call` fail closed unless explicitly configured
   otherwise. Observability hooks fail open.
6. Tool results remain ordered, bounded, and JSON-encoded at provider
   boundaries.
7. Provider loops may batch only tools marked concurrency-safe. Unknown,
   destructive, and overlapping-path mutations remain serial barriers.
   Worker pools are bounded, and background permission reviews stay
   serialized even when the approved handlers can overlap.
8. Named agents and named/automatic workflows emit their lifecycle boundary
   events, including an error event when execution raises.
9. Context compaction uses structured episodes. Do not add a second,
   independent summarization path.
10. Permission resolution never opens a UI prompt. Explicit denies and the
    root/home deletion circuit breaker precede yolo; ambiguous auto/safe
    actions use a fresh tool-less reviewer and fail closed if it is unavailable.

The public subprocess hook protocol is documented in the README. Python hooks
receive `HookContext` and return `HookResult`. Do not run subprocess hooks on
Textual's event thread; timeout handling must terminate the subprocess group,
not only the shell parent.

## Adding a provider

Implement the relevant `BaseProvider` methods in `cascade/providers`, register
the provider through the registry decorator, and preserve the shared
`hook_runner`, `permission_engine`, usage accounting, cancellation, and
provider-specific tool-call IDs when cloning or recursing.

## Adding a tool

Expose a typed Python callable through a plugin `get_tools()` method.
`callable_to_tool_def` derives a closed JSON schema from its signature.
Use `Literal` for constrained strings, mark read-only operations
`concurrency_safe=True` only when overlap is genuinely safe, and keep returned
payloads compact enough for an agent context window. Calls returned in one
provider turn are executed as a safe batch when every tool opts in; result
messages are still appended in the model's original order.

## Harness benchmark

`cascade-cli benchmark` is deterministic, offline, and provider-independent.
It is the quick feedback loop for batching, hook overhead, ordering, error
counts, and schema size:

```bash
cascade-cli benchmark --output .cascade/benchmark.json
cascade-cli benchmark --baseline .cascade/benchmark.json --json
```

Do not interpret zero-delay wall-clock changes as meaningful performance
results. Use the default synthetic delay (or a representative explicit
`--delay`) and compare several repeats on the same machine.

## Real task evaluation

`cascade-cli eval` is the quality loop for model, prompt, router, and
orchestration changes. A manifest task must name a local fixture, an ordinary
user prompt, independent verification commands, and optionally expected files.
Fixtures should be small enough to run repeatedly but representative of real
repository work:

```bash
cascade-cli eval examples/eval.example.yaml --provider openai \
  --output .cascade/eval-openai.json
cascade-cli eval examples/eval.example.yaml --provider openai \
  --task search-cache-lru-ttl
```

Do not use the agent's own claim as the pass condition. Evaluation always runs
the manifest checks in the returned worktree, and automatically rejects changes
to conventional test files, `conftest.py`, and `pytest.ini`. Use
`protected_files` in a manifest for any additional grader assets. Keep task
prompts provider-neutral so route and model comparisons remain meaningful.

## Verification

Run the complete suite:

```bash
python3 -m pytest -q
```

Useful focused checks:

```bash
python3 -m pytest -q tests/test_hooks_v2.py tests/test_tools.py
python3 -m pytest -q tests/test_tool_calling.py tests/test_harness.py
python3 -m compileall -q cascade tests
ruff check cascade tests
```

The repository predates the current Ruff configuration and still has broad
style debt. For focused changes, at minimum keep touched files free of syntax,
undefined-name, and unused-import failures, and avoid mixing unrelated
formatting into functional patches.
