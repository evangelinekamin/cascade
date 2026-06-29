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
provider. Type `/help` to list the available commands.
