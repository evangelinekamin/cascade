"""Per-model behavioral settings for the tool-calling loop.

One seam that adapts the loop to a model's known quirks -- the pattern every
mature cheap-model harness (aider's ModelSettings, opencode's per-model
temperature table, Roo's provider settings) relies on. Cheap open models need
different handling from frontier ones: coding-appropriate sampling, and runtime
enforcement that a text-only "I'll now edit..." round is not the end of the turn.

Matched by case-insensitive substring of the model id, most specific first;
models with no entry get the neutral default (nothing overridden).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    """How the tool loop should treat a given model.

    ``temperature`` overrides the provider default when set (cheap coders are far
    more reliable near 0). ``nudge_on_narration`` bounces a text-only round back
    at models prone to narrating a plan instead of calling their tools.
    ``replay_reasoning`` re-attaches ``reasoning_content`` on replayed assistant
    tool-call turns for models that require it (DeepSeek/Kimi lineage).
    """

    temperature: float | None = None
    nudge_on_narration: bool = False
    replay_reasoning: bool = False


_NEUTRAL = ModelSettings()

# Substring -> settings, checked in order (put more specific ids first if needed).
_REGISTRY: tuple[tuple[str, ModelSettings], ...] = (
    ("deepseek", ModelSettings(temperature=0.0, nudge_on_narration=True, replay_reasoning=True)),
    ("kimi", ModelSettings(temperature=1.0, nudge_on_narration=True, replay_reasoning=True)),
    ("glm", ModelSettings(temperature=1.0, nudge_on_narration=True)),
    ("minimax", ModelSettings(temperature=1.0, nudge_on_narration=True)),
    ("qwen", ModelSettings(temperature=0.55, nudge_on_narration=True)),
    ("mimo", ModelSettings(nudge_on_narration=True)),
    ("mercury", ModelSettings(nudge_on_narration=True)),
    ("gpt-oss", ModelSettings(nudge_on_narration=True)),
)


def settings_for(model: str) -> ModelSettings:
    """The settings for *model*, or the neutral default when none match."""
    model_id = (model or "").lower()
    for needle, settings in _REGISTRY:
        if needle in model_id:
            return settings
    return _NEUTRAL
