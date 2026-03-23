"""Tool registry — maps tool names to schemas and handler callables."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], Any]


@dataclass
class ToolDef:
    """A single tool definition: its API schema, handler, and metadata."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    category: str = "general"
    is_write: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Stores tool definitions and exposes them as Claude API schemas.

    Usage::

        registry = ToolRegistry()

        # Decorator style
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
            return f"Weather in {params['location']}: sunny, 22C"

        # Shorthand for simple tools
        @registry.simple_tool("add", "Add two numbers", a=float, b=float)
        def add(params):
            return params["a"] + params["b"]
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
        category: str = "general",
        is_write: bool = False,
    ) -> Callable:
        """Decorator that registers a handler function as a tool."""

        def decorator(fn: ToolHandler) -> ToolHandler:
            self._tools[name] = ToolDef(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=fn,
                category=category,
                is_write=is_write,
            )
            return fn

        return decorator

    def simple_tool(
        self,
        name: str,
        description: str,
        category: str = "general",
        is_write: bool = False,
        **params: type,
    ) -> Callable:
        """Decorator for tools with simple parameter types.

        Generates the JSON Schema from Python types::

            @registry.simple_tool("calc", "Evaluate math", expression=str)
            def calc(params):
                return eval(params["expression"])
        """
        type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
        properties = {}
        for pname, ptype in params.items():
            properties[pname] = {"type": type_map.get(ptype, "string")}

        schema = {
            "type": "object",
            "properties": properties,
            "required": list(params.keys()),
        }
        return self.tool(name, description, schema, category=category, is_write=is_write)

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: ToolHandler,
        category: str = "general",
        is_write: bool = False,
    ) -> None:
        """Imperatively register a tool (alternative to the decorator)."""
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            category=category,
            is_write=is_write,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolHandler | None:
        """Return the handler for *name*, or ``None``."""
        defn = self._tools.get(name)
        return defn.handler if defn else None

    def get_def(self, name: str) -> ToolDef | None:
        """Return the full ToolDef for *name*, or ``None``."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> list[str]:
        """Return tool names in a given category."""
        return [t.name for t in self._tools.values() if t.category == category]

    def write_tools(self) -> set[str]:
        """Return names of all tools marked as write operations."""
        return {t.name for t in self._tools.values() if t.is_write}

    def read_tools(self) -> set[str]:
        """Return names of all tools NOT marked as write operations."""
        return {t.name for t in self._tools.values() if not t.is_write}

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
