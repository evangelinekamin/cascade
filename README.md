# Cascade

A multi-provider AI coding agent in the terminal. One Textual TUI over Claude,
Gemini, OpenAI, and OpenRouter, with mode-based provider switching, tool use,
hooks, session history, and multi-agent orchestration (swarm and competition).

## Install

```
pip install -e .
```

This puts a `cascade` command on your PATH. Requires Python 3.10+.

## Usage

```
cascade
```

`Shift+Tab` cycles modes (design / plan / build / test), each bound to a
provider. Ordinary prompts are automatically classified in every mode:

- conversation stays on the selected chat model;
- repository questions use a read-only reconnaissance worker;
- focused edits run in an isolated worktree and must pass the configured tests;
- dependent changes use a sequential verified pipeline; and
- genuinely independent, disjoint changes may fan out and merge in parallel.

The default control/recon lane is `openai/gpt-oss-120b` through OpenRouter,
preferring the `cerebras` endpoint and requiring support for every requested
parameter. Tiny, low-blast-radius solves may start on
`stepfun/step-3.7-flash`, then escalate to the selected frontier provider if the
verification gate fails. These defaults only activate when OpenRouter is
configured; otherwise Cascade falls back to conservative local routing.

The everyday slash surface is intentionally small: `/status`, `/model`, `/mode`,
`/context`, `/history`, `/resume`, `/export`, `/doctor`, `/apply`, and `/help`.
Commands such as `/solve`, `/pipeline`, and `/fanout` remain available as
explicit overrides/debug controls, but normal use should not require remembering
them. `/help` shows everyday controls; `/help all` shows the complete registry.

`/status` consolidates the active model/mode, context occupancy, permission
posture, latest route and its reason, historical route tie-breaker, tokens,
cost, outcome, and retained worktree. `/doctor` fingerprints the installed
Claude, Gemini, Codex, and Git binaries and reports which noninteractive,
structured-output, sandbox, permission, resume, and extra-workspace flags are
actually available. Probe results are cached until a binary changes; use
`/doctor refresh` after an in-place CLI upgrade.

Each normal prompt now has a durable run ID and lifecycle record in Cascade's
history database. Pipeline and fan-out plans journal their task states,
dependencies, owned files, models, token totals, and review worktree paths. If
OpenRouter returns usage cost, that is journaled with the run as well. If
Cascade or the machine exits mid-run, open records are marked `interrupted` on
the next startup rather than being mistaken for successful or disappearing.

`Ctrl+X Ctrl+K` cancels the active model-selected run. Cancellation is shared
with provider loops, native CLI subprocesses, workspace commands, parallel
workers, and verification subprocesses; stale callbacks from that run are
discarded, so they cannot appear in a later turn. A cancelled solve/pipeline
keeps its single review worktree when one exists. Fan-out removes per-task
scratch worktrees and keeps only a reviewable integration worktree.

Automatic routing can be tuned or disabled in `~/.config/cascade/config.yaml`:

```yaml
orchestration:
  enabled: true
  modes: [design, plan, build, test]
  router_provider: openrouter
  router_model: openai/gpt-oss-120b
  recon_provider: openrouter
  recon_model: openai/gpt-oss-120b
  fast_provider: openrouter
  fast_model: stepfun/step-3.7-flash
  provider_preferences:
    order: [cerebras]
    allow_fallbacks: true
    require_parameters: true
    # Optional privacy filters; stricter filters can reduce endpoint availability.
    # data_collection: deny
    # zdr: true
  fast_provider_preferences:
    allow_fallbacks: true
    require_parameters: true
```

To measure Cascade's local harness without spending tokens or contacting a
provider, run:

```bash
cascade-cli benchmark --output .cascade/benchmark.json
cascade-cli benchmark --baseline .cascade/benchmark.json --json
```

The report tracks safe-batch speedup, result ordering, tool errors, hook
dispatch latency, and schema size. Baseline comparisons report signed
percentage changes so regressions can be caught between passes.

For a real model/repository benchmark, the included manifest provides ten
deterministic tasks spanning bug fixes, API behavior, parsing, caching,
retry logic, and larger algorithms:

```bash
cascade-cli eval examples/eval.example.yaml --provider openai \
  --output .cascade/eval.json
cascade-cli eval examples/eval.example.yaml --provider openai \
  --task pcb-bom-rollup --task pagination-boundaries
```

Each fixture is copied to an isolated temporary repository. Cascade handles the
ordinary prompt through automatic routing, then the evaluator independently
runs the manifest's verification commands against Cascade's reported worktree.
The JSON report includes pass rate, route, changed files, duration, tokens,
cost, bounded verification output, and tool errors/duplicate reads when the
provider exposes its tool loop. Opaque native CLI lanes report those fields as
`null`, never a misleading zero. Test and grader configuration files are
snapshotted before each run; modifying one makes the task fail even if the
edited checks pass. Add `--keep` when debugging a failure.

`cascade-cli run "implement the requested change" --json` exposes the same
automatic router as a structured one-shot command for scripts and external
harnesses.

`/export` writes a Markdown transcript plus durable run/task receipts: route
reason, task outcomes, model/provider, tokens, cost, changed files, verification
kind, errors, worktree, and a reproducible diff command. `/export --json`
produces the same data as a versioned machine-readable object. A suggested
follow-up is included only when it follows from recorded state (for example, a
passing retained worktree that is ready for review/application).

## Permissions without popups

Cascade never pauses a run for a permission dialog. The default `auto` posture
runs read-only tools, workspace edits, transparent development commands, and
fixed-endpoint web search immediately. Ambiguous operations—protected paths,
opaque shell, unusual network fetches, and writes outside the workspace—go to
a fresh, tool-less model for a background allow/deny decision. If that reviewer
is unavailable, times out, or returns malformed output, the action is denied
and the coding model can re-plan.

```yaml
permissions:
  posture: auto       # auto | yolo | safe | readonly
  allow: []
  deny: []
  ask: []             # legacy name: force background review, never a popup
  reviewer:
    enabled: true
    provider: ""      # blank: orchestration/default direct API provider
    model: ""
    timeout: 10
```

`yolo` runs everything except explicit deny rules and a hard circuit breaker
for recursively deleting `/` or the user's home directory. `safe` reviews
every mutation; `readonly` denies every mutation. Repeated review denials stop
the tool loop after three consecutive or twenty total denials.

The reviewer receives the latest user objective plus action metadata, but no
tool results and no file-content, patch, or request-body payloads. Configure
`reviewer.provider` if that metadata must stay with a particular direct API
provider. A repository's checked-in policy may add denies/reviews or tighten
the posture, but cannot enable yolo or add trusted allow rules.

OAuth CLI proxies own their internal tool calls, so Cascade maps the same
postures onto each CLI's native noninteractive boundary: Claude `auto`, Gemini
sandboxed yolo, and Codex `approval_policy="never"` with a workspace sandbox.
The explicit root/home deletion circuit breaker applies to Cascade-managed
tools; proxy tools remain bounded by their CLI's sandbox and policy engine.

## Hooks

Hooks can observe, transform, or block Cascade lifecycle events. Configure them
under `hooks` in `~/.config/cascade/config.yaml`:

```yaml
hooks:
  - name: protect-secrets
    event: tool_call
    if: "Read(.env*)"
    command: "python3 ~/.config/cascade/hooks/protect_secrets.py"
    timeout: 10
    priority: 50

  - name: add-project-context
    event: context_build
    module: ~/.config/cascade/hooks/add_context.py
```

Shell hooks receive a JSON `HookContext` object on stdin and a small set of
`CASCADE_*` metadata variables in their environment. They may print one JSON
control object:

```json
{"block": true, "reason": "protected path"}
```

or:

```json
{"transformed_value": {"path": "safe/example.env"}}
```

Exit status `2` also blocks an event. Tool-call hooks fail closed by default;
set `fail_closed: false` only when a policy hook is intentionally advisory.
Other events fail open so logging and notification hooks do not break a run.
Transforms run in priority order, and each hook sees the previous hook's
output. Run `/hooks` to inspect the active definitions and the last ten
outcomes. TUI input hooks run outside the render thread, and timed-out shell
hooks terminate their subprocess group so descendants cannot linger.

Python module hooks must live under `.cascade/hooks`,
`~/.cascade/hooks`, or `~/.config/cascade/hooks`, and expose:

```python
from cascade.hooks import HookResult

def hook(ctx):
    if ctx.event == "input_received":
        return HookResult(transformed_value=ctx.prompt.strip())
    return None
```

Lifecycle events are `session_start`, `session_resume`, `input_received`,
`agent_start`, `workflow_start`, `before_ask`, `context_build`,
`before_provider_request`, `tool_call`, `tool_result`, `after_response`,
`agent_end`, `workflow_end`, `episode_generated`, `provider_switch`,
`on_error`, and `on_exit`. Agent/workflow hooks receive `agent_name` and
`workflow` in structured context (and `CASCADE_AGENT` / `CASCADE_WORKFLOW` as
compatibility metadata).
