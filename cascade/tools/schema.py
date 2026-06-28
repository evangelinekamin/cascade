"""Convert Python callables into JSON Schema tool definitions.

Inspects type annotations on function signatures to build the JSON Schema
that providers need for native function calling.
"""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, get_type_hints


_TYPE_MAP = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


@dataclass(frozen=True)
class ToolDef:
    """A tool definition with its JSON Schema and handler.

    Tools with ``is_concurrent=True`` may execute in parallel with other
    concurrent tools. Non-concurrent tools get exclusive access (serialised).
    Read-only tools should set ``is_read_only=True`` (implies concurrent).
    """

    name: str
    description: str
    parameters: dict
    handler: Callable
    is_concurrent: bool = False
    is_read_only: bool = False
    is_destructive: bool = False

    @property
    def concurrency_safe(self) -> bool:
        """Whether this tool can safely run alongside others."""
        return self.is_concurrent or self.is_read_only


def _annotation_to_schema(annotation: Any) -> dict:
    """Convert a Python type annotation to a JSON Schema type."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}

    schema = _TYPE_MAP.get(annotation)
    if schema:
        return dict(schema)

    # Handle Optional[X] (Union[X, None])
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin is type(None):
        return {"type": "string"}

    # typing.Optional[X] is Union[X, None]
    if origin is not None and len(args) == 2 and type(None) in args:
        inner = [a for a in args if a is not type(None)][0]
        return _annotation_to_schema(inner)

    # list[X]
    if origin is list and args:
        return {"type": "array", "items": _annotation_to_schema(args[0])}

    # dict[str, X]
    if origin is dict:
        return {"type": "object"}

    return {"type": "string"}


def callable_to_tool_def(
    name: str,
    fn: Callable,
    description: str = "",
) -> ToolDef:
    """Build a ToolDef from a Python callable using its signature and docstring.

    Args:
        name: Tool name for the registry.
        fn: The callable to introspect.
        description: Fallback description if the function has no docstring.

    Returns:
        A ToolDef with JSON Schema parameters derived from type annotations.
    """
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn) or description

    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, param.annotation)
        prop_schema = _annotation_to_schema(annotation)

        # Extract parameter description from docstring Args section
        param_doc = _extract_param_doc(doc, param_name)
        if param_doc:
            prop_schema["description"] = param_doc

        properties[param_name] = prop_schema

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return ToolDef(
        name=name,
        description=doc,
        parameters=parameters,
        handler=fn,
    )


def _extract_param_doc(docstring: str, param_name: str) -> Optional[str]:
    """Extract a parameter's description from a Google-style docstring."""
    if not docstring:
        return None

    lines = docstring.split("\n")
    in_args = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("args:"):
            in_args = True
            continue

        if in_args:
            if stripped and not stripped.startswith("-") and ":" not in stripped[:30]:
                # Left the Args section
                if not stripped.startswith(" "):
                    in_args = False
                    continue

            # Match "param_name: description" or "param_name (type): description"
            if stripped.startswith(f"{param_name}:") or stripped.startswith(f"{param_name} ("):
                colon_idx = stripped.index(":")
                return stripped[colon_idx + 1:].strip()

    return None
