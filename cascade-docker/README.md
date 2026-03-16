# Cascade Docker CLI Proxies

Optional Docker containers for running Gemini, Claude, and Codex CLI tools
in isolated environments. This is an alternative to the default subprocess
path -- you do not need Docker to use Cascade.

## Quick Start

Build all images:

```bash
docker compose build
```

Run a single provider (arguments are passed through to the CLI):

```bash
docker compose run --rm gemini -p "Hello" --output-format stream-json
docker compose run --rm claude -p "Hello" --output-format stream-json
docker compose run --rm codex exec --json "Hello"
```

## Credential Sharing

Credentials are mounted read-only from the host at
`~/.config/cascade/tokens/`. Override the path with:

```bash
CASCADE_TOKENS=/path/to/tokens docker compose run --rm gemini -p "Hello"
```

## Project Directory

The current working directory is mounted at `/workspace` inside the
container. Override with:

```bash
CASCADE_PROJECT=/path/to/project docker compose run --rm claude -p "Hello"
```

## Design

- Each container runs as a non-root `cascade` user.
- Images are kept minimal (node:20-slim for JS CLIs, ubuntu:24.04 for Claude).
- No credentials are baked into images -- they are always mounted at runtime.
- Containers are not long-running services; they exit after the CLI completes.
