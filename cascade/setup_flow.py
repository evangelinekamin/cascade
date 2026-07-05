"""Interactive setup wizard for first-run configuration."""

import os
from typing import Optional

from .auth import detect_all
from .config import ConfigManager
from .providers.base import ProviderConfig
from .providers.registry import discover_providers, get_registry
from .ui.theme import console, CYAN, VIOLET

# Map of provider name -> list of env var names to check
_ENV_VARS = {
    "gemini": ["GEMINI_API_KEY"],
    "claude": ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
}

# Default models per provider
_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "claude": "claude-sonnet-4-6",
    "openrouter": "qwen/qwen3.5-9b",
    "openai": "gpt-5.3-codex",
}

_DEFAULT_FALLBACK_MODELS = {
    "openrouter": "minimax/minimax-m2.5",
}

# Speed-over-quality one-shot models reachable via /ultrafast and /fast.
_DEFAULT_FAST_MODELS = {
    "openrouter": "inception/mercury-2",
}


def detect_env_keys() -> dict[str, str]:
    """Detect which API keys are available from environment variables.

    Returns a dict of provider_name -> resolved_api_key.
    """
    found = {}
    for provider, var_names in _ENV_VARS.items():
        for var in var_names:
            value = os.getenv(var, "")
            if value:
                found[provider] = value
                break
    return found


class SetupWizard:
    """Interactive first-run setup flow."""

    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        discover_providers()
        self.registry = get_registry()

    def run(self) -> None:
        """Run the full setup wizard."""
        console.print("\n[bold]CASCADE Setup[/bold]\n", style=CYAN)

        # 1. Detect CLI credentials first
        console.print("Detecting CLI tool credentials...\n", style="dim")
        cli_creds = detect_all()
        cli_providers = {}
        if cli_creds:
            console.print("Found CLI credentials:", style=f"bold {CYAN}")
            for cred in cli_creds:
                label = f"  {cred.source}"
                if cred.email:
                    label += f" ({cred.email})"
                if cred.plan:
                    label += f" [{cred.plan}]"
                console.print(label, style="dim")
                cli_providers[cred.provider] = cred.token
            console.print()

        # 2. Detect env var API keys
        console.print("Detecting API keys from environment...\n", style="dim")
        env_keys = detect_env_keys()
        if env_keys:
            console.print("Detected env keys:", style=f"bold {CYAN}")
            for name, _key in env_keys.items():
                masked = _key[:4] + "..." + _key[-4:] if len(_key) > 8 else "***"
                console.print(f"  {name}: {masked}", style="dim")
            console.print()

        # Merge: CLI creds take priority, then env keys
        detected = {**env_keys, **cli_providers}

        if not detected:
            console.print("No credentials detected.\n", style="dim")

        enabled_providers = []

        # Walk through each known provider
        for name in sorted(self.registry.keys()):
            enabled = self._configure_provider(name, detected.get(name, ""))
            if enabled:
                enabled_providers.append(name)

        # Choose default
        if enabled_providers:
            default = self._choose_default(enabled_providers)
            self.config.data.setdefault("defaults", {})["provider"] = default
        else:
            console.print(
                "No providers configured. Run 'cascade-cli setup' later to configure.",
                style="dim",
            )

        self.config.save()
        console.print("\nSetup complete. Configuration saved.\n", style=f"bold {CYAN}")

    def _configure_provider(self, name: str, detected_key: str) -> bool:
        """Configure a single provider. Returns True if enabled."""
        if detected_key:
            answer = self._prompt(
                f"Enable {name}? (key detected) [Y/n]: ", default="y"
            )
            if answer.lower() not in ("y", "yes", ""):
                return False
            api_key = detected_key
        else:
            # Try interactive auth flow first
            try:
                from .auth_flow import login
                console.print(f"\n  Authenticate {name}?", style=f"dim {CYAN}")
                answer = self._prompt(
                    f"  Run interactive login for {name}? [Y/n/key]: ", default="y"
                )
                if answer.lower() in ("y", "yes", ""):
                    result = login(name)
                    if result:
                        api_key = result.token
                    else:
                        return False
                elif answer.lower() in ("n", "no", "skip"):
                    return False
                else:
                    # Treat as pasted API key
                    api_key = answer
            except ImportError:
                # Fallback to raw input
                answer = self._prompt(
                    f"Enter API key for {name} (or press Enter to skip): ", default=""
                )
                if not answer:
                    return False
                api_key = answer

        # Resolve provider model before validation.
        model = self._select_model(name, _DEFAULT_MODELS.get(name, ""))
        fallback_model = _DEFAULT_FALLBACK_MODELS.get(name)
        fast_model = _DEFAULT_FAST_MODELS.get(name)

        # Validate with a ping
        provider_cls = self.registry.get(name)
        if provider_cls and api_key:
            console.print(f"  Testing {name}...", style="dim", end=" ")
            try:
                prov = provider_cls(
                    ProviderConfig(
                        api_key=api_key,
                        model=model,
                        fallback_model=fallback_model,
                    )
                )
                if prov.ping():
                    console.print("OK", style="green")
                else:
                    console.print("Warning: no response", style="yellow")
            except Exception as e:
                console.print(f"Error: {e}", style="red")

        # Save to config
        providers = self.config.data.setdefault("providers", {})
        providers.setdefault(name, {})
        providers[name]["enabled"] = True
        providers[name]["api_key"] = api_key
        providers[name]["model"] = model
        if fallback_model:
            providers[name]["fallback_model"] = fallback_model
        if fast_model:
            providers[name]["fast_model"] = fast_model
        providers[name].setdefault("temperature", 0.7)
        providers[name].setdefault("max_tokens", 2048)
        return True

    def _select_model(self, name: str, default_model: str) -> str:
        """Resolve a provider model, prompting when customization matters."""
        if name != "openrouter":
            return default_model

        console.print(
            f"  OpenRouter model [default: {default_model}]",
            style="dim",
        )
        console.print(
            "  Examples: qwen/qwen3.5-9b, qwen/qwen3-coder-next, minimax/minimax-m2.5",
            style="dim",
        )
        chosen = self._prompt(
            f"  Enter model slug for {name} [{default_model}]: ",
            default=default_model,
        ).strip()
        return chosen or default_model

    def _choose_default(self, enabled: list[str]) -> str:
        """Let the user choose a default provider."""
        if len(enabled) == 1:
            console.print(f"Default provider: {enabled[0]}", style="dim")
            return enabled[0]

        console.print("\nChoose default provider:", style=f"bold {VIOLET}")
        for i, name in enumerate(enabled, 1):
            console.print(f"  {i}. {name}")

        while True:
            choice = self._prompt(f"Enter number [1-{len(enabled)}]: ", default="1")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(enabled):
                    return enabled[idx]
            except ValueError:
                pass
            console.print("Invalid choice.", style="dim red")

    @staticmethod
    def _prompt(message: str, default: str = "") -> str:
        """Read user input with a default."""
        try:
            value = input(message)
            return value if value else default
        except (EOFError, KeyboardInterrupt):
            return default


def needs_setup(config: Optional[ConfigManager] = None) -> bool:
    """Check if setup is needed (no enabled providers)."""
    cfg = config or ConfigManager()
    return len(cfg.get_enabled_providers()) == 0
