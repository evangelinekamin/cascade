"""Configuration management for Cascade."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Collection
import yaml

from .providers.base import ProviderConfig
from .theme import MODE_CYCLE, MODES


_DEFAULT_MODE_CONFIG = {
    mode_name: {
        "provider": mode_cfg["provider"],
        "model": "",
    }
    for mode_name, mode_cfg in MODES.items()
}


class ConfigManager:
    """Manage Cascade configuration from YAML."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "~/.config/cascade/config.yaml").expanduser()
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            self._create_default_config()
            return self._read_yaml()
        
        return self._read_yaml()

    def _read_yaml(self) -> Dict[str, Any]:
        """Read and parse YAML file."""
        try:
            with open(self.config_path, "r") as f:
                content = yaml.safe_load(f)
                return content or {}
        except Exception as e:
            print(f"Error reading config: {e}")
            return {}

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            "providers": {
                "gemini": {
                    "enabled": False,
                    "api_key": "${GEMINI_API_KEY}",
                    "model": "gemini-3.1-pro-preview",
                    "fast_model": "gemini-3-flash-preview",
                    "temperature": 0.7,
                },
                "claude": {
                    "enabled": False,
                    "api_key": "${CLAUDE_API_KEY}",
                    "model": "claude-opus-4-8",
                    "fast_model": "claude-sonnet-5",
                    "temperature": 0.7,
                },
                "openrouter": {
                    "enabled": False,
                    "api_key": "${OPENROUTER_API_KEY}",
                    "model": "qwen/qwen3.5-9b",
                    "fallback_model": "minimax/minimax-m2.5",
                    # Speed-over-quality one-shot tier (/ultrafast, /fast): a very
                    # fast, deliberately flaky diffusion LM.
                    "fast_model": "inception/mercury-2",
                    "temperature": 0.7,
                    # Pin good-quant fast upstream hosts. Throughput-sorted
                    # routing silently truncates completions, so prefer known
                    # solid-quant providers and still allow fallbacks.
                    "provider_preferences": {
                        "order": ["Baidu", "Fireworks", "Alibaba"],
                        "allow_fallbacks": True,
                    },
                },
                "openai": {
                    "enabled": False,
                    "api_key": "${OPENAI_API_KEY}",
                    "model": "gpt-5.3-codex",
                    "temperature": 0.7,
                },
                "local": {
                    # Keyless OpenAI-compatible model on THIS machine (fast tier).
                    # Flip enabled to true to use it -- no API key required.
                    "enabled": False,
                    "requires_key": False,
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "qwen36",
                    "temperature": 0.7,
                    # Self-hosted small window -- keeps the agentic loop from
                    # overflowing a 32K-context local model.
                    "context_window": 32768,
                },
                "glm": {
                    # Keyless OpenAI-compatible model on a remote box (strong tier).
                    # local -> glm escalation is handled by routing (see roadmap).
                    "enabled": False,
                    "requires_key": False,
                    "base_url": "http://512s2.netbird.cloud:52415/v1",
                    "model": "pipenetwork/GLM-5.2-MLX-8bit",
                    "temperature": 0.7,
                    "context_window": 32768,
                },
            },
            "defaults": {
                "provider": "gemini",
                "theme": "deep-stream",
            },
            "modes": _DEFAULT_MODE_CONFIG,
            "prompts": {
                "use_default_system_prompt": True,
                "include_design_language": True,
                "design_md_path": "",
            },
            "memory": {
                # How much cross-provider context to carry between model switches.
                # off     -> no prior context
                # summary -> carry compact handoff summary + provider-local turns
                # full    -> carry recent full transcript across providers
                "cross_model_memory": "summary",
                # Re-compact summary every N assistant turns in summary mode.
                "summary_turn_interval": 6,
                # Preferred provider to generate summaries (or "auto").
                # Hard cap for compact summary text included in prompts.
                "summary_max_chars": 1800,
            },
            "hooks": [],
            "tools": {
                "reflection": True,
                "file_ops": True,
                # Opt-in shell tool for direct-API chat models (test/build).
                # Off by default -- it runs real commands in the launch dir.
                "exec": False,
            },
            "workflows": {
                "verify": {
                    "lint": "ruff check .",
                    "test": "python -m pytest -x -q",
                    "build": "",
                    "audit": "",
                },
            },
            "orchestration": {
                # Normal prompts in these modes are classified into chat,
                # read-only reconnaissance, focused solve, sequential pipeline,
                # or safe fanout. Slash commands remain useful as debug controls.
                "enabled": True,
                "modes": ["design", "plan", "build", "test"],
                "router_provider": "openrouter",
                "router_model": "openai/gpt-oss-120b",
                "recon_provider": "openrouter",
                "recon_model": "openai/gpt-oss-120b",
                "fast_provider": "openrouter",
                "fast_model": "inception/mercury-2",
                "fast_provider_preferences": {
                    "allow_fallbacks": True,
                    "require_parameters": True,
                },
                "provider_preferences": {
                    # Cerebras is selected through OpenRouter, not configured as
                    # a separate Cascade provider. Other hosts may take over when
                    # it is unavailable, but only if they support required params.
                    "order": ["cerebras"],
                    "allow_fallbacks": True,
                    "require_parameters": True,
                },
                "recon_max_rounds": 10,
            },
            "integrations": {
                "shannon": {
                    "path": "",
                },
            },
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

    def apply_credential(self, provider_name: str, token: str, overwrite: bool = False) -> None:
        """Auto-enable a provider using a detected CLI credential.

        Only applies if the provider isn't already enabled with a resolved key,
        unless overwrite=True.
        """
        providers = self.data.setdefault("providers", {})
        entry = providers.setdefault(provider_name, {})

        # Don't overwrite if already enabled with a real key
        existing_key = self._resolve_env_var(entry.get("api_key", ""))
        if entry.get("enabled") and existing_key:
            if not overwrite:
                return

        # Default models per provider
        default_models = {
            "gemini": "gemini-3.1-pro-preview",
            "claude": "claude-opus-4-8",
            "openai": "gpt-5.3-codex",
            "openrouter": "qwen/qwen3.5-9b",
        }

        entry["enabled"] = True
        entry["api_key"] = token
        entry.setdefault("model", default_models.get(provider_name, ""))
        if provider_name == "openrouter":
            entry.setdefault("fallback_model", "minimax/minimax-m2.5")
            entry.setdefault("fast_model", "inception/mercury-2")
        entry.setdefault("temperature", 0.7)

    def get_provider_config(self, provider_name: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        provider_data = self.data.get("providers", {}).get(provider_name, {})

        if not provider_data.get("enabled", False):
            return None

        api_key = self._resolve_env_var(provider_data.get("api_key", ""))
        # Self-hosted / local endpoints can opt out of the key requirement.
        if not api_key and provider_data.get("requires_key", True):
            return None

        return ProviderConfig(
            api_key=api_key,
            model=provider_data.get("model", ""),
            base_url=provider_data.get("base_url"),
            temperature=provider_data.get("temperature", 0.7),
            max_tokens=provider_data.get("max_tokens"),
            fallback_model=provider_data.get("fallback_model"),
            context_window=provider_data.get("context_window"),
            provider_preferences=provider_data.get("provider_preferences"),
        )

    def _resolve_env_var(self, value: str) -> str:
        """Resolve environment variable references like ${VAR_NAME}."""
        if not value.startswith("${") or not value.endswith("}"):
            return value
        
        var_name = value[2:-1]
        return os.getenv(var_name, "")

    def get_default_provider(self) -> str:
        """Get the default provider name."""
        return self.data.get("defaults", {}).get("provider", "gemini")

    def get_mode_config(self, mode_name: str) -> Dict[str, str]:
        """Return provider/model config for a mode with defaults applied."""
        base = dict(_DEFAULT_MODE_CONFIG.get(mode_name, {}))
        raw = self.data.get("modes", {}).get(mode_name, {})
        if isinstance(raw, dict):
            provider = raw.get("provider")
            model = raw.get("model")
            if isinstance(provider, str) and provider.strip():
                base["provider"] = provider.strip().lower()
            if isinstance(model, str):
                base["model"] = model.strip()
        return base

    def get_mode_provider(self, mode_name: str) -> str:
        """Return the configured provider for a mode."""
        return self.get_mode_config(mode_name).get(
            "provider",
            MODES.get(mode_name, {}).get("provider", "gemini"),
        )

    def get_mode_model_override(self, mode_name: str) -> str:
        """Return the configured model override for a mode, if any."""
        return self.get_mode_config(mode_name).get("model", "")

    def get_model_for(self, provider_name: str, mode_name: Optional[str] = None, fast: bool = False) -> str:
        """Resolve the model to use for a provider in the given mode."""
        provider_data = self.data.get("providers", {}).get(provider_name, {})
        if fast:
            fast_model = str(provider_data.get("fast_model", "") or "").strip()
            if fast_model:
                return fast_model

        if mode_name:
            mode_cfg = self.get_mode_config(mode_name)
            mode_provider = mode_cfg.get("provider", "")
            mode_model = mode_cfg.get("model", "")
            if mode_provider == provider_name and mode_model:
                return mode_model

        return str(provider_data.get("model", "") or "").strip()

    def get_bulk_model(self, provider_name: str, mode_name: Optional[str] = None) -> str:
        """Resolve the bulk (cheap-tier) model for /solve's early iterations.

        Prefers an explicit ``bulk_model``; otherwise uses the provider's frontier
        ``model``. It deliberately does NOT fall back to ``fast_model`` -- that slot
        is the /ultrafast speed lane (often a fast-but-flaky model), which must never
        become /solve's builder. Set ``bulk_model`` to run a cheaper model first and
        escalate to ``model`` on failure.
        """
        provider_data = self.data.get("providers", {}).get(provider_name, {})
        bulk_model = str(provider_data.get("bulk_model", "") or "").strip()
        if bulk_model:
            return bulk_model
        return self.get_model_for(provider_name, mode_name, fast=False)

    def get_escalation_target(self, provider_name: str) -> Optional[str]:
        """Return the provider to hand off to when *provider_name* stalls in /solve.

        Reads ``providers.<name>.escalate_to`` -- a provider name (e.g. "claude")
        that a stalled cheap-bulk /solve escalates to, so a stronger provider
        finishes the hard part. None means stay within the same provider (bulk
        model -> frontier model), the prior behavior.
        """
        provider_data = self.data.get("providers", {}).get(provider_name, {})
        target = str(provider_data.get("escalate_to", "") or "").strip()
        return target or None

    def get_default_mode_for_provider(self, provider_name: str) -> str:
        """Return the first configured mode that maps to the given provider."""
        for mode_name in MODE_CYCLE:
            if self.get_mode_provider(mode_name) == provider_name:
                return mode_name
        for mode_name, mode_cfg in MODES.items():
            if mode_cfg.get("provider") == provider_name:
                return mode_name
        return "design"

    def get_available_modes(self, configured_providers: Collection[str] | None = None) -> tuple[str, ...]:
        """Return modes whose configured provider exists in this session."""
        if not configured_providers:
            return MODE_CYCLE
        available = set(configured_providers)
        return tuple(
            mode_name
            for mode_name in MODE_CYCLE
            if self.get_mode_provider(mode_name) in available
        )

    def get_enabled_providers(self) -> list[str]:
        """Get list of enabled provider names."""
        providers = self.data.get("providers", {})
        return [name for name, config in providers.items() if config.get("enabled", False)]

    def get_prompt_config(self) -> Dict[str, Any]:
        """Get prompt system configuration."""
        defaults = {
            "use_default_system_prompt": True,
            "include_design_language": True,
            "design_md_path": "",
        }
        config = self.data.get("prompts", {})
        return {**defaults, **config} if config else defaults

    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory behavior configuration."""
        defaults: Dict[str, Any] = {
            "cross_model_memory": "summary",
            "summary_turn_interval": 6,
            "summary_max_chars": 1800,
        }
        config = self.data.get("memory", {})
        merged = {**defaults, **(config or {})}

        policy = str(merged.get("cross_model_memory", "summary")).lower()
        if policy not in ("off", "summary", "full"):
            policy = "summary"
        merged["cross_model_memory"] = policy

        try:
            interval = int(merged.get("summary_turn_interval", 6))
        except Exception:
            interval = 6
        merged["summary_turn_interval"] = max(1, interval)

        try:
            max_chars = int(merged.get("summary_max_chars", 1800))
        except Exception:
            max_chars = 1800
        merged["summary_max_chars"] = max(400, max_chars)


        return merged

    def get_hooks_config(self) -> list:
        """Get hooks configuration (list of hook definitions)."""
        return self.data.get("hooks", [])

    def get_tools_config(self) -> Dict[str, bool]:
        """Get tools enable/disable configuration."""
        defaults = {
            "reflection": True,
            "file_ops": True,
            "exec": False,
        }
        config = self.data.get("tools", {})
        return {**defaults, **config} if config else defaults

    def get_workflows_config(self) -> Dict[str, Any]:
        """Get workflows configuration (verify commands, etc.)."""
        defaults: Dict[str, Any] = {
            "verify": {
                "lint": "ruff check .",
                "test": "python -m pytest -x -q",
                "build": "",
                "audit": "",
            },
        }
        config = self.data.get("workflows", {})
        return {**defaults, **config} if config else defaults

    def get_orchestration_config(self) -> Dict[str, Any]:
        """Return sanitized automatic workflow-routing configuration."""
        defaults: Dict[str, Any] = {
            "enabled": True,
            "modes": ["design", "plan", "build", "test"],
            "router_provider": "openrouter",
            "router_model": "openai/gpt-oss-120b",
            "recon_provider": "openrouter",
            "recon_model": "openai/gpt-oss-120b",
            "fast_provider": "openrouter",
            "fast_model": "inception/mercury-2",
            "fast_provider_preferences": {
                "allow_fallbacks": True,
                "require_parameters": True,
            },
            "provider_preferences": {
                "order": ["cerebras"],
                "allow_fallbacks": True,
                "require_parameters": True,
            },
            "recon_max_rounds": 10,
        }
        raw = self.data.get("orchestration", {})
        merged = {**defaults, **(raw if isinstance(raw, dict) else {})}
        merged["enabled"] = bool(merged.get("enabled", True))

        raw_modes = merged.get("modes", defaults["modes"])
        if not isinstance(raw_modes, (list, tuple)):
            raw_modes = defaults["modes"]
        modes = [
            str(mode).strip().lower()
            for mode in raw_modes
            if str(mode).strip().lower() in MODE_CYCLE
        ]
        merged["modes"] = modes or list(defaults["modes"])

        for key in (
            "router_provider",
            "router_model",
            "recon_provider",
            "recon_model",
            "fast_provider",
            "fast_model",
        ):
            value = str(merged.get(key, defaults[key]) or "").strip()
            merged[key] = value or defaults[key]

        preferences = merged.get("provider_preferences")
        if not isinstance(preferences, dict):
            preferences = defaults["provider_preferences"]
        merged["provider_preferences"] = dict(preferences)
        fast_preferences = merged.get("fast_provider_preferences")
        if not isinstance(fast_preferences, dict):
            fast_preferences = defaults["fast_provider_preferences"]
        merged["fast_provider_preferences"] = dict(fast_preferences)

        try:
            rounds = int(merged.get("recon_max_rounds", 10))
        except (TypeError, ValueError):
            rounds = 10
        merged["recon_max_rounds"] = max(1, min(rounds, 20))
        return merged

    def get_integrations_config(self) -> Dict[str, Any]:
        """Get integrations configuration."""
        defaults = {
            "shannon": {"path": ""},
        }
        config = self.data.get("integrations", {})
        return {**defaults, **config} if config else defaults

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False)
