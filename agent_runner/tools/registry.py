"""Tool registry — maps tool names to schemas and handler callables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass
class ToolDef:
    """A single tool definition: its API schema and its handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    """Stores tool definitions and exposes them as Claude API schemas.

    Usage::

        registry = ToolRegistry()

        @registry.tool(
            name="get_weather",
            description="Get the current weather for a location.",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        )
        def get_weather(params):
            return f"Weather in {params['location']}: sunny, 22°C"
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> Callable:
        """Decorator that registers a handler function as a tool."""

        def decorator(fn: ToolHandler) -> ToolHandler:
            self._tools[name] = ToolDef(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=fn,
            )
            return fn

        return decorator

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        """Imperatively register a tool (alternative to the decorator)."""
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolHandler | None:
        """Return the handler for *name*, or ``None``."""
        defn = self._tools.get(name)
        return defn.handler if defn else None

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # API serialization
    # ------------------------------------------------------------------

    def to_api_schema(self) -> list[dict[str, Any]]:
        """Return the list of tool dicts expected by ``client.messages.create``."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]
