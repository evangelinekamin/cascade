# Cascade Refactor Design Document

**Author:** Design review by Claude (Opus), to be implemented via Claude Code
**Date:** March 2026
**Status:** Ready for implementation (diagnostic baseline captured)
**Priority order:** Phase 1 → 2 → 3 → 4 → 5 (each phase should be working before starting the next)
**Diagnostic baseline:** All 4 providers init + ping + stream OK. All 3 subscription providers use CLI proxy mode.

---

## Context & Motivation

Cascade is a multi-model AI assistant CLI/TUI that unifies Gemini, Claude, OpenAI, and OpenRouter behind a single Textual interface with Shift+Tab model switching. The primary auth path uses OAuth tokens from installed CLI tools (gemini-cli, claude, codex) rather than API keys, proxying through those CLIs via subprocess.

The project has solid architectural bones — provider registry with decorators, reactive TUI state, prompt pipeline with priority layers, clean provider abstraction — but several foundational issues are preventing it from being daily-drivable. This document specifies the changes needed, in priority order.

### Key Design Principles

1. **Subscription-first:** OAuth/CLI proxy mode is the primary path, not a fallback. Design for it.
2. **Conversation continuity across models:** Switching providers mid-session must feel seamless.
3. **Fail loud, recover gracefully:** Errors should be visible and actionable, not swallowed silently.
4. **Trim before adding:** Every new feature must justify its complexity cost.

### Diagnostic Baseline (March 5, 2026)

Run from `~/Projects/cascade` with all three CLIs installed and authenticated:

```
1. Credential detection:
   claude: Claude Code CLI (plan=max)     — sk-ant-oat01-*
   gemini: Gemini CLI (plan=Google One AI Pro, email=evangelinekamin07@gmail.com) — ya29.*
   openai: Codex CLI (plan=plus, email=evangelinehelsinki@gmail.com) — eyJ*

2. Provider initialization:
   gemini:     OK (model=gemini-3.1-pro-preview, oauth=False, cli_proxy=True)
   claude:     OK (model=claude-opus-4-6, oauth=True, cli_proxy=True)
   openrouter: OK (model=qwen/qwen3.5-35b-a3b, oauth=False, cli_proxy=False)
   openai:     OK (model=gpt-5.3-codex, oauth=True, cli_proxy=True)

3. Provider connectivity: ALL PASS

4. Streaming: ALL PRODUCE OUTPUT (but see findings below)
```

**Key findings that affect priority ordering:**

1. **All three subscription providers use CLI proxy mode.** This means direct API code paths (Gemini JSON parsing, safety settings, token refresh) are NOT exercised in normal use. Fixes for those are deprioritized.
2. **Activity prefix messages leak into raw output.** CLI proxy providers emit `[[cascade_activity]] ...` status lines that the TUI filters, but any other code path (Click CLI commands, the diagnostic script, agents) sees garbled output. Filtering must move into the provider layer.
3. **Usage/token tracking is broken in proxy mode.** Gemini and OpenAI return `usage=None`. Claude returns `(3, 5)` which is a handshake artifact, not real usage. Status bar token counts are wrong for 3 of 4 providers.
4. **Gemini reports `oauth=False` despite using an OAuth token.** The Gemini provider uses `_use_bearer` internally while Claude/OpenAI use `_use_oauth_cli` — inconsistent naming. Minor but confusing for debugging.
5. **Only OpenRouter hits the direct API path** (it has no CLI tool). It's the only provider where the httpx streaming code, retry logic, and JSON parsing actually matter today.

These findings reshape the priority of Phase 2. See below.

---

## Phase 1: Stateful Conversations (Critical)

### Problem

Every call to `provider.stream()` and `provider.ask()` sends a single user message with no conversation history. The provider has no memory of previous turns. This makes multi-turn conversations incoherent and is the single biggest usability issue.

`CascadeState.messages` already tracks the full conversation as `ChatMessage` objects, but this data is never passed to providers.

### Design

#### 1.1 Change the BaseProvider interface

Update `BaseProvider` to accept a message list instead of a single prompt string. The old single-prompt signature remains as a convenience that wraps the new one.

```python
# cascade/providers/base.py

from typing import TypedDict

class Message(TypedDict, total=False):
    role: str       # "user" | "assistant" | "system"
    content: str
    provider: str   # which provider generated this (for cross-model context)

class BaseProvider(ABC):

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream tokens. `messages` is the full conversation history."""
        pass

    def stream_single(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Convenience: single-prompt call. Wraps stream()."""
        return self.stream([{"role": "user", "content": prompt}], system)

    @abstractmethod
    def ask(self, messages: list[Message], system: Optional[str] = None) -> str:
        pass

    def ask_single(self, prompt: str, system: Optional[str] = None) -> str:
        return self.ask([{"role": "user", "content": prompt}], system)

    def ask_with_tools(
        self,
        messages: list[Message],
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
    ) -> tuple[str, list[dict]]:
        return self.ask(messages, system), []
```

#### 1.2 Convert CascadeState.messages to provider format

Add a utility in `cascade/state.py` or a new `cascade/conversation.py`:

```python
def state_messages_to_provider(
    messages: list[ChatMessage],
    target_provider: str,
    policy: str = "summary",            # "off" | "summary" | "full"
    cross_model_summary: str = "",
    max_messages: int = 40,
    max_chars: int = 80_000,
) -> list[Message]:
    """Convert CascadeState messages to provider-ready message list.

    Handles cross-model context injection based on policy:
    - "off": Only include messages from target_provider (and user messages)
    - "summary": Include cross-model summary + recent same-provider turns
    - "full": Include all recent messages regardless of provider
    """
    result: list[Message] = []

    if policy == "off":
        # Only user messages and responses from this specific provider
        for msg in messages[-max_messages:]:
            if msg.role == "you":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == target_provider:
                result.append({"role": "assistant", "content": msg.content})

    elif policy == "summary":
        # Inject cross-model summary as a system-ish context message,
        # then include recent same-provider turns
        if cross_model_summary:
            result.append({
                "role": "user",
                "content": f"[Context from previous model interactions]\n{cross_model_summary}",
            })
            result.append({
                "role": "assistant",
                "content": "Understood, I have the context from the previous interactions.",
            })
        for msg in messages[-max_messages:]:
            if msg.role == "you":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == target_provider:
                result.append({"role": "assistant", "content": msg.content})

    elif policy == "full":
        # Include everything, annotating cross-provider responses
        for msg in messages[-max_messages:]:
            if msg.role == "you":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == target_provider:
                result.append({"role": "assistant", "content": msg.content})
            else:
                # Message from a different provider — include as context
                result.append({
                    "role": "user",
                    "content": f"[Response from {msg.role}]\n{msg.content}",
                })
                result.append({
                    "role": "assistant",
                    "content": "Noted.",
                })

    # Enforce character budget by trimming oldest messages
    total_chars = sum(len(m["content"]) for m in result)
    while total_chars > max_chars and len(result) > 2:
        removed = result.pop(0)
        total_chars -= len(removed["content"])

    return result
```

#### 1.3 Update each provider implementation

Each provider converts the `list[Message]` to its native format:

**Claude provider** (`cascade/providers/claude.py`):
```python
def stream(self, messages: list[Message], system=None) -> Iterator[str]:
    # ... existing auth checks ...
    payload = {
        "model": self.config.model,
        "max_tokens": self.config.max_tokens or 4096,
        "temperature": self.config.temperature,
        "stream": True,
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ],
    }
    if system:
        payload["system"] = system
    # ... rest of streaming logic unchanged ...
```

**Gemini provider** (`cascade/providers/gemini.py`):
```python
def stream(self, messages: list[Message], system=None) -> Iterator[str]:
    # ... existing auth checks ...
    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    for msg in messages:
        gemini_role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})
    # ... rest of streaming logic, using contents ...
```

**OpenAI provider** (`cascade/providers/openai_provider.py`):
```python
def stream(self, messages: list[Message], system=None) -> Iterator[str]:
    # ... existing auth checks ...
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages.extend(
        {"role": m["role"], "content": m["content"]}
        for m in messages
    )
    # ... rest of streaming logic, using api_messages ...
```

**OpenRouter provider** — same as OpenAI (it's OpenAI-compatible).

#### 1.4 Update the CLI proxy path

For OAuth/CLI proxy mode, conversation history can't be passed directly to the subprocess (each CLI manages its own session). Two options:

**Option A (recommended for now):** Continue sending only the latest prompt via CLI proxy, but prepend a condensed context block. Per the diagnostic baseline, this is the path used by all three subscription providers (gemini, claude, openai), making it the most impactful code path to get right:

```python
def _stream_via_cli(self, messages: list[Message], system=None) -> Iterator[str]:
    # Build a single prompt that includes recent context
    context_lines = []
    for msg in messages[:-1]:  # all but the last (current) message
        role_label = "User" if msg["role"] == "user" else "Assistant"
        # Truncate old messages aggressively
        content = msg["content"][:500]
        context_lines.append(f"{role_label}: {content}")

    current_prompt = messages[-1]["content"] if messages else ""

    if context_lines:
        full_prompt = (
            "Previous conversation context:\n"
            + "\n".join(context_lines[-6:])  # last 3 turns
            + "\n\nCurrent request:\n"
            + current_prompt
        )
    else:
        full_prompt = current_prompt

    # ... existing subprocess spawn logic using full_prompt ...
```

**Option B (future, with Docker):** Run each CLI as a persistent MCP server or long-lived process that maintains its own conversation state. See Phase 4.

#### 1.5 Wire it into MainScreen._provider_worker()

```python
def _provider_worker(self, prompt: str, provider_name: str):
    cli_app = self.app.cli_app
    prov = cli_app.providers.get(provider_name)
    # ... existing checks ...

    final_system = self._build_system_prompt(cli_app, prompt, provider_name)

    # NEW: Build conversation history from state
    from ..conversation import state_messages_to_provider
    messages = state_messages_to_provider(
        messages=list(self.app.state.messages),
        target_provider=provider_name,
        policy=self._memory_policy,
        cross_model_summary=self._cross_model_summary,
    )

    # ... hook execution ...

    tool_registry = getattr(cli_app, "tool_registry", None)
    use_tools = (
        tool_registry
        and len(tool_registry) > 0
        and self._should_use_tools(prov)
    )

    if use_tools:
        self._tool_worker(cli_app, prov, messages, provider_name, final_system, tool_registry)
    else:
        self._stream_worker(cli_app, prov, messages, provider_name, final_system)
```

Update `_stream_worker` and `_tool_worker` to pass `messages` instead of `prompt`.

#### 1.6 Update CascadeApp.ask() in cli.py

The Click CLI path also needs updating for the `ask` and `compare` commands:

```python
def ask(self, prompt, provider=None, system=None, stream=False, context_text=None):
    prov = self.get_provider(provider)
    # Build messages from conversation history
    messages = [{"role": "user", "content": prompt}]
    # For CLI mode, conversation history is simpler (single-turn by default)
    # The REPL adds history; the one-shot CLI commands don't need it

    # ... existing pipeline/hook logic ...

    if self.tool_registry and not stream:
        response, tool_log = prov.ask_with_tools(messages, self.tool_registry, system=final_system)
    elif stream:
        response = stream_response(prov.stream(messages, final_system), prov.name)
    else:
        response = prov.ask(messages, final_system)
    # ...
```

### Tests to add

- `test_provider_receives_conversation_history`: Mock a provider, send 3 turns, verify all 3 arrive in `stream()`
- `test_cross_model_summary_injection`: Verify summary policy injects context correctly
- `test_cli_proxy_context_condensation`: Verify CLI proxy mode includes truncated recent context
- `test_message_format_per_provider`: Verify Gemini gets `model`/`user` roles, Claude/OpenAI get `assistant`/`user`

---

## Phase 2: Reliability & Error Handling

> **Priority reordering note (based on diagnostic baseline):** All three subscription
> providers (gemini, claude, openai) run through CLI proxy mode. The direct API code
> paths are only exercised by OpenRouter today. This phase is reordered to fix the
> issues that affect the *actual* production path first, with direct-API fixes grouped
> at the end as lower priority.

### 2.1 Move activity filtering into the provider layer (quick win)

**Problem:** CLI proxy providers emit `[[cascade_activity]] ...` status lines mixed
into the stream output. Currently only `MainScreen._on_stream_chunk()` filters these.
Any other consumer of `provider.stream()` — the Click CLI `ask` command, agents,
the diagnostic script — sees garbled text with activity messages in the response.

**Fix:** Add filtering to `BaseProvider` so *all* callers get clean output, while
still making activity data available for the TUI.

```python
# cascade/providers/base.py — add to BaseProvider

_ACTIVITY_PREFIX = "[[cascade_activity]] "

@property
def last_activity(self) -> Optional[str]:
    """Most recent activity status from CLI proxy, or None."""
    return getattr(self, "_last_activity", None)

def _filter_activity(self, chunks: Iterator[str]) -> Iterator[str]:
    """Strip activity prefix messages from stream, storing them for TUI access."""
    for chunk in chunks:
        if isinstance(chunk, str) and chunk.startswith(self._ACTIVITY_PREFIX):
            self._last_activity = chunk[len(self._ACTIVITY_PREFIX):].strip()
            continue
        yield chunk
```

Then in each provider's `stream()` method that uses CLI proxy, wrap the return:

```python
# In GeminiProvider.stream(), ClaudeProvider.stream(), OpenAIProvider.stream():
def stream(self, messages, system=None):
    self._last_usage = None
    self._last_activity = None
    if self._use_cli_proxy:
        yield from self._filter_activity(self._stream_via_cli(messages, system))
        return
    # ... rest of direct API path ...
```

The TUI's `_on_stream_chunk` can then check `provider.last_activity` on a timer
or after each chunk batch, instead of parsing the prefix out of content chunks.
Remove the activity filtering from `MainScreen._provider_worker()` / `_stream_worker()`.

**Impact:** Fixes garbled output everywhere. ~30 minutes of work.

### 2.2 Fix CLI proxy usage/token tracking

**Problem:** Token usage is broken for all CLI proxy providers:
- Gemini: `usage=None`
- OpenAI: `usage=None`
- Claude: `usage=(3, 5)` (handshake artifact, not real usage)

The CLI tools emit usage data in their JSON stream output (typically in a `result`
or `stats` event at the end of the stream), but the current `_stream_via_cli()`
implementations either don't parse it or parse the wrong event.

**Fix:** Each provider's `_stream_via_cli()` needs to look for the result/stats
event at the end of the stream and extract real token counts.

```python
# In GeminiProvider._stream_via_cli() — the test already shows the expected format:
# {"type":"result","status":"success","stats":{"input_tokens":10,"output_tokens":4}}

for raw_line in proc.stdout:
    line = raw_line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        non_json_lines.append(line)
        continue

    event_type = event.get("type", "")

    if event_type == "message" and event.get("role") == "assistant":
        delta = event.get("content", "")
        if event.get("delta"):
            yield delta
        # Also handle non-delta full content messages
        elif delta and not saw_text:
            yield delta

    elif event_type == "result":
        stats = event.get("stats", {})
        in_t = stats.get("input_tokens", 0)
        out_t = stats.get("output_tokens", 0)
        if in_t or out_t:
            self._last_usage = (in_t, out_t)
```

Apply similar parsing to Claude and OpenAI proxy paths, matching each CLI's
actual JSON event schema. Check the existing test in `test_gemini_provider.py`
(`test_stream_cli_parses_assistant_messages_and_usage`) for the expected Gemini
format — it already tests for this correctly, which suggests the production code
may have drifted from what the test expects.

**Impact:** Fixes token tracking in the status bar for all subscription providers.

### 2.3 Fix attribute naming inconsistency

**Problem:** Gemini provider uses `_use_bearer` while Claude/OpenAI use
`_use_oauth_cli`. Diagnostic shows Gemini reporting `oauth=False` despite
being an OAuth token. This makes debugging confusing.

**Fix:** Add `_use_oauth_cli` to GeminiProvider for consistency:

```python
# In GeminiProvider.__init__():
self._use_bearer = config.api_key.startswith("ya29.")
self._use_oauth_cli = self._use_bearer  # Consistent with other providers
self._gemini_bin = shutil.which("gemini")
self._use_cli_proxy = self._use_oauth_cli and bool(self._gemini_bin)
```

### 2.4 Retry with exponential backoff

Add a shared retry utility. All three reference CLIs implement this. Since CLI
proxy mode shells out to subprocesses, retry logic here primarily helps:
- OpenRouter (the only direct-API provider currently in use)
- Any future API-key users
- CLI proxy subprocess failures (non-zero exit codes, timeouts)

```python
# cascade/providers/retry.py

import time
import random
from functools import wraps

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds

class RetryableError(Exception):
    """Raised when an API call fails with a retryable status code."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)

def retry_on_transient_error(max_retries=MAX_RETRIES, base_delay=BASE_DELAY):
    """Decorator that retries on transient HTTP errors with jittered backoff."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except RetryableError as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(delay)
                    continue
            raise last_error
        return wrapper
    return decorator
```

For the direct API path, catch `httpx.HTTPStatusError` and raise `RetryableError`
for retryable codes. For CLI proxy mode, add retry on non-zero subprocess exit
codes (the CLI tools return non-zero on transient server errors):

```python
# In _stream_via_cli(), after proc.wait():
if proc.returncode != 0 and not saw_text:
    raise RetryableError(503, f"CLI exited with code {proc.returncode}")
```

### 2.5 Fix the `chat` command bug in cli.py

The `chat` command is currently indented inside the `setup` command's scope and
imports `CascadeApp` from `.app` (the Textual class) instead of using the CLI backend.

```python
# Move chat to top-level, after the @cli.command() for setup

@cli.command()
@click.option("--provider", "-p", help="Provider to use")
def chat(provider):
    """Start interactive chat mode (TUI)."""
    from .repl import main as tui_main
    tui_main()
```

### 2.6 Rename CascadeApp to CascadeCore

Rename the backend class in `cli.py` from `CascadeApp` to `CascadeCore` (or
`CascadeBackend`) to eliminate confusion with the Textual `CascadeTUI` in
`app.py`. Update all imports. This is a mechanical find-and-replace.

### 2.7 Model fallback (nice-to-have)

Inspired by Gemini CLI's auto-fallback when rate-limited:

```python
# cascade/providers/base.py — add to BaseProvider
def get_fallback_model(self) -> Optional[str]:
    """Return a cheaper/faster model to fall back to on rate limits."""
    return None

# In GeminiProvider:
def get_fallback_model(self) -> Optional[str]:
    if "pro" in self.config.model:
        return self.config.model.replace("pro", "flash")
    return None
```

The retry logic can attempt the fallback model after exhausting retries on primary.

### 2.8 Direct API path fixes (low priority — only affects API key users)

These fixes are still correct but deprioritized since all subscription providers
use CLI proxy mode. Implement them when bandwidth allows, or if someone reports
issues using API keys directly.

**Gemini JSON array parsing:** The `streamGenerateContent` endpoint can return
array-wrapped JSON. Replace line-by-line parsing with a buffered `raw_decode` approach:

```python
# Only applies to GeminiProvider.stream() when NOT in cli_proxy mode
buffer = ""
for chunk in response.iter_text():
    buffer += chunk
    while buffer:
        buffer = buffer.lstrip(" ,[\n\r")
        if not buffer or buffer[0] == ']':
            buffer = buffer.lstrip("] \n\r")
            continue
        try:
            data, end_idx = json.JSONDecoder().raw_decode(buffer)
            buffer = buffer[end_idx:]
            # ... extract text from candidates as before ...
        except json.JSONDecodeError:
            break  # incomplete, wait for more data
```

**Gemini safety settings:** Replace invalid `HARM_CATEGORY_UNSPECIFIED`:

```python
"safetySettings": [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
],
```

**Gemini OAuth token refresh:** For direct API mode with `ya29.*` tokens (not
proxied through CLI), add a refresh mechanism. Only needed if someone uses a Gemini
OAuth token without the gemini CLI installed:

```python
# cascade/auth.py — add refresh_gemini_token()
def refresh_gemini_token() -> Optional[str]:
    creds_path = Path.home() / ".gemini" / "oauth_creds.json"
    data = _read_json(creds_path)
    if data is None:
        return None
    refresh_token = data.get("refresh_token")
    client_id = data.get("client_id", "")
    client_secret = data.get("client_secret", "")
    if not all([refresh_token, client_id, client_secret]):
        return None
    try:
        import httpx
        resp = httpx.post("https://oauth2.googleapis.com/token", data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        })
        resp.raise_for_status()
        new_data = resp.json()
        data["access_token"] = new_data["access_token"]
        data["expiry_date"] = int(time.time() * 1000) + (new_data.get("expires_in", 3600) * 1000)
        creds_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return new_data["access_token"]
    except Exception:
        return None
```

### Tests to add

- `test_activity_filtering`: Verify `_filter_activity()` strips prefixes and stores last activity
- `test_stream_returns_clean_output`: Verify `stream()` output has no activity prefixes for all providers
- `test_cli_proxy_usage_parsing`: Verify `_stream_via_cli()` extracts real token counts from result events
- `test_retry_on_429`: Mock a 429 response, verify retry with backoff
- `test_retry_exhaustion`: Verify error surfaces after max retries
- `test_cli_proxy_retry_on_nonzero_exit`: Verify subprocess failure triggers retry
- `test_gemini_oauth_attribute_consistency`: Verify `_use_oauth_cli` is set for ya29 tokens

---

## Phase 3: Scope Trimming & Module Boundaries

### 3.1 Make Shannon an optional package

Move `cascade/integrations/shannon.py` and the `/shannon` command to a separate installable:

```
cascade-shannon/
├── pyproject.toml    # depends on cascade-cli
├── cascade_shannon/
│   ├── __init__.py
│   └── integration.py
```

Register via entry point so Cascade discovers it if installed:

```toml
# cascade-shannon/pyproject.toml
[project.entry-points."cascade.integrations"]
shannon = "cascade_shannon:ShannonIntegration"
```

### 3.2 Make the web upload server optional

Same pattern for `cascade/web/server.py`. It's already behind a `[web]` extra — just move it fully out so it doesn't add import weight to the core.

### 3.3 Simplify agent tool restrictions

Currently agents can specify `allowed_tools` as a tuple of tool names. This is fine, but don't build out more granular permissions (per-tool-per-agent configs, tool deny lists, etc.) until there's a concrete need. Keep the schema as-is but document that it's a simple allowlist.

### 3.4 Audit and trim unused config surface

Review `config.yaml` for settings that don't actually do anything or that duplicate each other. Specifically check:
- `prompts.include_design_language` vs `prompts.design_md_path` — are both needed?
- `memory.summary_provider` — does this actually get used, or does the fallback logic always kick in?
- `hooks` — are any hooks actually configured by default, or is this dead code?

---

## Phase 4: CLI Proxy Hardening & Docker (Future)

### 4.1 The case for Docker

**Problem:** Installing Cascade today requires separately installing gemini-cli (npm), claude (binary), and codex (npm), then authenticating each one individually. This is friction.

**Solution:** Package each CLI in a lightweight Docker container with a JSON-RPC or stdio interface. Cascade's setup wizard handles pulling images and running OAuth flows.

```
cascade-docker/
├── gemini/
│   ├── Dockerfile          # FROM node:20-slim, npm install -g @google/gemini-cli
│   └── entrypoint.sh       # gemini mcp-server (or similar persistent mode)
├── claude/
│   ├── Dockerfile          # FROM ubuntu:24.04, install claude binary
│   └── entrypoint.sh
├── codex/
│   ├── Dockerfile          # FROM node:20-slim, npm install -g @openai/codex
│   └── entrypoint.sh
└── docker-compose.yml
```

**Auth flow:** Mount `~/.config/cascade/tokens/` into each container. Run the CLI's login command inside the container, which writes creds to the mounted volume. On subsequent runs, creds are already there.

**Benefits:**
- `cascade setup` becomes: "pull 3 Docker images, run auth for each"
- Portable across machines — just copy the tokens directory
- Each CLI runs as a persistent process, maintaining its own conversation state
- Version-pinned — you control which CLI version you're running
- Parallel instances for subagent workloads become trivial

**Trade-offs:**
- Docker dependency (but most dev machines have it)
- ~500MB per container image (but they share the node:20-slim base)
- Slightly higher latency for first request (container startup)
- May not work on the school computer if Docker isn't available

**Recommendation:** Build this as an optional backend. The subprocess-based CLI proxy remains the default. If Docker is available and the user runs `cascade setup --docker`, use the containerized approach.

### 4.2 MCP server mode for each CLI

Rather than parsing CLI stdout, use MCP server mode where available:

- **Codex:** Already supports `codex mcp-server` with `codex()` and `codex-reply()` tools
- **Gemini CLI:** Check if `gemini mcp-server` or equivalent exists; if not, use the `stream-json` output format which is more structured than plain text
- **Claude Code:** Supports `claude mcp-server-mode` (verify exact flag)

This gives structured JSON communication instead of fragile stdout parsing.

### 4.3 Persistent CLI sessions

Instead of spawning a new subprocess for each message, keep the CLI running and feed it messages:

```python
class PersistentCLISession:
    """Keeps a CLI tool running as a long-lived subprocess."""

    def __init__(self, binary: str, args: list[str]):
        self.proc = subprocess.Popen(
            [binary] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def send(self, message: str) -> Iterator[str]:
        """Send a message and yield response chunks."""
        self.proc.stdin.write(message + "\n")
        self.proc.stdin.flush()
        # Read structured JSON responses until turn complete
        for line in self.proc.stdout:
            event = json.loads(line)
            if event.get("type") == "message" and event.get("role") == "assistant":
                yield event.get("content", "")
            if event.get("type") == "result":
                break

    def close(self):
        self.proc.terminate()
```

This approach keeps the CLI's own conversation state alive across turns, giving you real multi-turn conversations through the proxy without needing to re-inject history.

---

## Phase 5: Context Window Management (Future)

### 5.1 Compaction strategy

Inspired by all three reference CLIs. When conversation history approaches the model's context limit, compact it.

```python
# cascade/conversation.py

COMPACTION_THRESHOLD = 0.75  # Compact when 75% of context window is estimated full

# Rough context window sizes by model family
CONTEXT_WINDOWS = {
    "gemini": 1_000_000,    # Gemini 2.5 Pro
    "claude": 200_000,      # Claude Sonnet/Opus
    "openai": 200_000,      # GPT-4o/Codex
    "openrouter": 128_000,  # varies, conservative default
}

def estimate_tokens(messages: list[Message]) -> int:
    """Rough token estimate. 1 token ≈ 4 chars for English text."""
    return sum(len(m["content"]) for m in messages) // 4

def needs_compaction(messages: list[Message], provider: str) -> bool:
    window = CONTEXT_WINDOWS.get(provider, 128_000)
    estimated = estimate_tokens(messages)
    return estimated > (window * COMPACTION_THRESHOLD)

def compact_messages(
    messages: list[Message],
    provider: "BaseProvider",
    keep_recent: int = 6,
) -> list[Message]:
    """Compact older messages into a summary, keep recent ones intact."""
    if len(messages) <= keep_recent:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    # Build compaction prompt
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:1000]}"
        for m in old_messages
    )

    summary = provider.ask_single(
        f"Summarize this conversation for continuation. Be concise but preserve "
        f"key decisions, file paths, code changes, and open tasks:\n\n{transcript}",
        system="You produce compact engineering handoff summaries. Under 800 words.",
    )

    compacted = [
        {"role": "user", "content": f"[Conversation summary]\n{summary}"},
        {"role": "assistant", "content": "Understood, I have the context. Continuing."},
    ]
    compacted.extend(recent_messages)
    return compacted
```

### 5.2 Prompt ordering for cache efficiency

Following Codex's pattern — put stable content first, variable content last:

```
[System prompt - stable across turns]           ← cacheable
[Tool definitions - stable within session]       ← cacheable
[Project context files - stable within session]  ← cacheable
[Design language prompt - stable]                ← cacheable
[Compacted history summary - changes on compact] ← partially cacheable
[Recent conversation turns - changes every turn] ← not cached
[Current user message]                           ← never cached
```

The prompt pipeline's priority system already supports this — just verify the numeric priorities produce this ordering. Lower priority number = earlier in prompt = more cacheable.

Current priorities in `cascade/prompts/layers.py`:
- 10: default system prompt ✓ (good, first)
- 20: design language ✓
- 30: project system prompt ✓
- 40: project context files ✓
- 50: user override ✓
- 60: REPL context (uploads, history) ✓ (good, last)

This ordering is already correct for cache efficiency. Just document it.

---

## Appendix A: File-by-file change list

### Phase 1 changes:
- `cascade/providers/base.py` — New `Message` type, update `BaseProvider` signatures
- `cascade/providers/gemini.py` — Update `stream()`, `ask()`, `ask_with_tools()`, `_stream_via_cli()` signatures
- `cascade/providers/claude.py` — Same signature updates
- `cascade/providers/openai_provider.py` — Same signature updates
- `cascade/providers/openrouter.py` — Same signature updates
- `cascade/conversation.py` — **NEW FILE** — `state_messages_to_provider()`, `estimate_tokens()`, `needs_compaction()`, `compact_messages()`
- `cascade/screens/main.py` — Update `_provider_worker()`, `_stream_worker()`, `_tool_worker()` to pass messages
- `cascade/cli.py` — Update `CascadeApp.ask()` to build message list
- `cascade/state.py` — No changes needed (ChatMessage already has what we need)
- `tests/test_providers.py` — Update mock provider tests for new signatures
- `tests/test_conversation.py` — **NEW FILE** — Tests for message conversion

### Phase 2 changes (ordered by priority):
- `cascade/providers/base.py` — Add `_filter_activity()`, `last_activity` property, `_ACTIVITY_PREFIX` constant
- `cascade/providers/gemini.py` — Wrap `_stream_via_cli()` with `_filter_activity()`, fix usage parsing in CLI stream, add `_use_oauth_cli` attribute
- `cascade/providers/claude.py` — Wrap `_stream_via_cli()` with `_filter_activity()`, fix usage parsing in CLI stream
- `cascade/providers/openai_provider.py` — Wrap `_stream_via_cli()` with `_filter_activity()`, fix usage parsing in CLI stream
- `cascade/screens/main.py` — Remove activity prefix filtering from `_stream_worker()` (now handled in provider layer), use `provider.last_activity` for TUI status
- `cascade/providers/retry.py` — **NEW FILE** — Retry decorator and `RetryableError`
- `cascade/providers/openrouter.py` — Add retry (this is the only direct-API provider in active use)
- `cascade/cli.py` — Fix `chat` command indentation/import, rename class to `CascadeCore`
- All files importing `CascadeApp` from `cli.py` — Update import name
- `tests/test_activity_filtering.py` — **NEW FILE** — Tests for activity filtering and usage tracking
- **(Low priority)** `cascade/providers/gemini.py` — Fix JSON array parsing, safety settings, token refresh (direct API path only)
- **(Low priority)** `cascade/auth.py` — Add `refresh_gemini_token()` (direct API path only)

### Phase 3 changes:
- Move Shannon integration to optional package
- Move web server to optional package
- Audit config.yaml surface

---

## Appendix B: Design patterns stolen from reference CLIs

| Pattern | Source | How to apply in Cascade |
|---------|--------|------------------------|
| Activity/status filtering in provider layer | Diagnostic finding | `_filter_activity()` on BaseProvider (Phase 2.1) |
| Turn state machine | Gemini CLI | Model each provider interaction as a Turn with phases |
| Auto model fallback on rate limit | Gemini CLI | `get_fallback_model()` on BaseProvider |
| Conversation branching/save points | Gemini CLI | `/save` and `/resume` commands (future) |
| Static-first prompt ordering for cache hits | Codex CLI | Already correct in prompt pipeline priorities |
| Stateless requests with full history | Codex CLI | Phase 1 implements this |
| Compaction with summary preservation | All three | Phase 5 implements this |
| Subagent context isolation | Claude Code | Agents already isolated; keep it that way |
| MCP server mode for tool integration | Codex CLI | Phase 4 for CLI proxying |
| JSON-RPC App Server pattern | Codex CLI | Consider for future multi-client support |
| Checkpoint/undo for file changes | Claude Code | Future: snapshot before tool execution |

---

## Appendix C: Diagnostic script

Before starting implementation, run this script to establish a baseline of what's working.
Run `python cascade_diag.py` before and after each phase to verify progress.

```python
#!/usr/bin/env python3
"""cascade_diag.py — Diagnose provider health.

Run from the cascade project root with venv activated:
    python cascade_diag.py
"""

import sys
sys.path.insert(0, ".")

from cascade.auth import detect_all
from cascade.config import ConfigManager
from cascade.providers.registry import discover_providers, get_registry
from cascade.providers.base import ProviderConfig

ACTIVITY_PREFIX = "[[cascade_activity]] "

def main():
    print("=== Cascade Diagnostics ===\n")

    # 1. Check credentials
    print("1. Credential detection:")
    creds = detect_all()
    for c in creds:
        print(f"   {c.provider}: {c.source} (plan={c.plan}, email={c.email})")
        token_preview = c.token[:20] + "..." if len(c.token) > 20 else c.token
        print(f"     token: {token_preview}")
    if not creds:
        print("   No credentials detected.")
    print()

    # 2. Check provider initialization
    print("2. Provider initialization:")
    discover_providers()
    registry = get_registry()
    config = ConfigManager()

    for name, cls in registry.items():
        cfg = config.get_provider_config(name)
        if cfg is None:
            print(f"   {name}: NOT CONFIGURED")
            continue
        try:
            provider = cls(cfg)
            proxy = getattr(provider, '_use_cli_proxy', False)
            oauth = getattr(provider, '_use_oauth_cli', False)
            bearer = getattr(provider, '_use_bearer', None)
            line = f"   {name}: OK (model={cfg.model}, oauth={oauth}, cli_proxy={proxy})"
            # Flag inconsistent oauth attributes
            if bearer is not None and bearer != oauth:
                line += f"  ⚠️  _use_bearer={bearer} != _use_oauth_cli={oauth}"
            print(line)
        except Exception as e:
            print(f"   {name}: INIT FAILED — {e}")
    print()

    # 3. Test each provider with a simple ping
    print("3. Provider connectivity (ping test):")
    for name, cls in registry.items():
        cfg = config.get_provider_config(name)
        if cfg is None:
            continue
        try:
            provider = cls(cfg)
            ok = provider.ping()
            print(f"   {name}: {'OK' if ok else 'FAILED (ping returned False)'}")
        except Exception as e:
            print(f"   {name}: FAILED — {e}")
    print()

    # 4. Test streaming with a minimal prompt — filter activity messages
    print("4. Streaming test (5-word response):")
    for name, cls in registry.items():
        cfg = config.get_provider_config(name)
        if cfg is None:
            continue
        try:
            provider = cls(cfg)
            chunks = []
            activity_count = 0
            for chunk in provider.stream("Say exactly: hello world", system=None):
                # Check for activity prefix leaking (Phase 2.1 fix target)
                if isinstance(chunk, str) and chunk.startswith(ACTIVITY_PREFIX):
                    activity_count += 1
                    continue
                chunks.append(chunk)
                if len("".join(chunks)) > 200:
                    break
            response = "".join(chunks)[:100]
            usage = provider.last_usage
            line = f'   {name}: "{response}" (usage={usage})'
            if activity_count > 0:
                line += f"  ⚠️  {activity_count} activity msgs leaked into stream"
            if usage is None:
                line += "  ⚠️  no usage data"
            elif usage == (0, 0):
                line += "  ⚠️  zero usage"
            elif usage[0] + usage[1] < 10:
                line += "  ⚠️  suspiciously low usage (handshake artifact?)"
            print(line)
        except Exception as e:
            print(f"   {name}: FAILED — {e}")

    # 5. Check for known issues
    print("\n5. Known issue checks:")
    checks_passed = 0
    checks_total = 0

    # Check: activity filtering in provider layer
    checks_total += 1
    has_filter = hasattr(
        list(registry.values())[0] if registry else object,
        '_filter_activity',
    )
    if has_filter:
        print("   ✓ Activity filtering in provider layer")
        checks_passed += 1
    else:
        print("   ✗ Activity filtering NOT in provider layer (Phase 2.1)")

    # Check: BaseProvider accepts messages list
    checks_total += 1
    import inspect
    base_cls = list(registry.values())[0] if registry else None
    if base_cls:
        sig = inspect.signature(base_cls.stream)
        params = list(sig.parameters.keys())
        if "messages" in params:
            print("   ✓ Provider.stream() accepts messages list")
            checks_passed += 1
        else:
            print("   ✗ Provider.stream() still uses single prompt (Phase 1)")
    else:
        print("   ? No providers to check")

    # Check: CascadeApp renamed
    checks_total += 1
    try:
        from cascade.cli import CascadeCore
        print("   ✓ CascadeApp renamed to CascadeCore")
        checks_passed += 1
    except ImportError:
        print("   ✗ CascadeApp NOT yet renamed (Phase 2.6)")

    print(f"\n   {checks_passed}/{checks_total} checks passed")

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
```
