"""Bridge MCP server tools into the agent runner's ToolRegistry.

This module connects an MCP server to the runner so that Claude can
use MCP-provided tools alongside local tools.  When combined with the
ShadowInterceptor, you get "shadow mode" — the agent runs against real
MCP servers but all write-side effects are captured, not executed.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_runner.mcp.client import MCPClient, MCPToolSchema
from agent_runner.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPToolBridge:
    """Register MCP server tools into a ToolRegistry.

    Parameters
    ----------
    client : MCPClient
        A started MCP client connection.
    registry : ToolRegistry
        The runner's tool registry to populate.
    prefix : str
        Optional namespace prefix for tool names to avoid collisions,
        e.g. ``"fs_"`` turns ``read_file`` into ``fs_read_file``.
    write_tool_names : set[str] | None
        MCP tool names considered write operations (for shadow mode).
        If provided, these are returned so callers can configure the
        ShadowInterceptor accordingly.
    """

    def __init__(
        self,
        client: MCPClient,
        registry: ToolRegistry,
        prefix: str = "",
        write_tool_names: set[str] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.prefix = prefix
        self.write_tool_names = write_tool_names or set()
        self._registered_tools: list[str] = []

    def bridge_tools(self) -> list[str]:
        """Discover MCP tools and register them in the runner's registry.

        Returns the list of registered tool names (with prefix applied).
        """
        mcp_tools = self.client.list_tools()
        for tool_schema in mcp_tools:
            registered_name = self._register_one(tool_schema)
            self._registered_tools.append(registered_name)
        return self._registered_tools

    def get_write_tools(self) -> set[str]:
        """Return prefixed names of tools marked as write operations."""
        return {f"{self.prefix}{name}" for name in self.write_tool_names}

    def get_read_tools(self) -> set[str]:
        """Return prefixed names of tools NOT marked as write operations."""
        return {
            name for name in self._registered_tools
            if name not in self.get_write_tools()
        }

    def _register_one(self, schema: MCPToolSchema) -> str:
        """Register a single MCP tool into the local registry."""
        registered_name = f"{self.prefix}{schema.name}"
        client = self.client  # capture for closure

        def handler(params: dict[str, Any]) -> Any:
            return client.call_tool(schema.name, params)

        self.registry.register(
            name=registered_name,
            description=schema.description,
            input_schema=schema.input_schema,
            handler=handler,
        )
        logger.info("Bridged MCP tool: %s -> %s", schema.name, registered_name)
        return registered_name
