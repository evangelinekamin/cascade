"""Immutable prompt pipeline that assembles system prompts from ordered layers.

Each layer has a name, content string, and priority (lower = earlier in output).
The pipeline is immutable: add_layer returns a new PromptPipeline instance.
"""

from dataclasses import dataclass


# Standard priorities for well-known layers
PRIORITY_DEFAULT = 10
PRIORITY_MODE = 15
PRIORITY_DESIGN = 20
PRIORITY_PROJECT_SYSTEM = 30
PRIORITY_PROJECT_CONTEXT = 40
PRIORITY_USER_OVERRIDE = 50
PRIORITY_REPL_CONTEXT = 60


@dataclass(frozen=True)
class PromptLayer:
    """A single layer in the prompt pipeline."""

    name: str
    content: str
    priority: int


class PromptPipeline:
    """Immutable pipeline that builds a system prompt from prioritized layers.

    Usage:
        pipeline = (
            PromptPipeline()
            .add_layer("default", default_prompt, priority=10)
            .add_layer("project", project_prompt, priority=30)
        )
        system_prompt = pipeline.build()
    """

    def __init__(self, layers: tuple[PromptLayer, ...] = ()):
        self._layers = layers

    @property
    def layers(self) -> tuple[PromptLayer, ...]:
        return self._layers

    @property
    def layer_count(self) -> int:
        return len(self._layers)

    def add_layer(self, name: str, content: str, priority: int) -> "PromptPipeline":
        """Return a new pipeline with the given layer added.

        If content is empty or whitespace-only, the layer is skipped.
        """
        if not content or not content.strip():
            return self
        layer = PromptLayer(name=name, content=content.strip(), priority=priority)
        return PromptPipeline(self._layers + (layer,))

    def remove_layer(self, name: str) -> "PromptPipeline":
        """Return a new pipeline without layers matching the given name."""
        filtered = tuple(layer for layer in self._layers if layer.name != name)
        return PromptPipeline(filtered)

    def has_layer(self, name: str) -> bool:
        """Check if a layer with the given name exists."""
        return any(layer.name == name for layer in self._layers)

    def build(self) -> str:
        """Assemble all layers sorted by priority into a single string.

        Layers are joined with double newlines. Empty result if no layers.
        """
        if not self._layers:
            return ""
        sorted_layers = sorted(self._layers, key=lambda layer: layer.priority)
        return "\n\n".join(layer.content for layer in sorted_layers)

    def describe(self) -> list[dict]:
        """Return a summary of layers for display (sorted by priority)."""
        sorted_layers = sorted(self._layers, key=lambda layer: layer.priority)
        return [
            {
                "name": layer.name,
                "priority": layer.priority,
                "length": len(layer.content),
            }
            for layer in sorted_layers
        ]
