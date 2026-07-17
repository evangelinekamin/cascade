# Cascade

A multi-provider AI coding agent in the terminal. One Textual TUI over Claude,
Gemini, OpenAI, and OpenRouter, with mode-based provider switching, tool use,
hooks, session history, and multi-agent orchestration (swarm and competition).

## Install

```
pip install -e .
```

This puts a `cascade` command on your PATH. Requires Python 3.9+.

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
`inception/mercury-2`, then escalate to the selected frontier provider if the
verification gate fails. These defaults only activate when OpenRouter is
configured; otherwise Cascade falls back to conservative local routing.

Slash commands such as `/solve`, `/pipeline`, and `/fanout` remain available as
explicit overrides and debugging controls. Type `/help` to list all commands.

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
  fast_model: inception/mercury-2
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
