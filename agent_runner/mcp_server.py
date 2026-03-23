"""MCP server that exposes agent-runner tools via the MCP protocol.

This lets Claude Code (or any MCP client) use agent-runner's built-in tools
directly, without writing any Python.  Install as a Claude Code plugin or
run standalone::

    # As an MCP server (stdio transport)
    python -m agent_runner.mcp_server

    # With specific tools only
    python -m agent_runner.mcp_server --tools calculator,read_file,list_files

    # With custom tools from a Python module
    python -m agent_runner.mcp_server --extend my_tools:register

The server speaks JSON-RPC over stdio, compatible with the MCP protocol.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from typing import Any

from agent_runner.tools.registry import ToolRegistry
from agent_runner.tools.builtins import register_builtins

logger = logging.getLogger(__name__)

SERVER_NAME = "agent-runner"
SERVER_VERSION = "0.2.0"
PROTOCOL_VERSION = "2024-11-05"


class MCPServer:
    """Lightweight MCP server that exposes ToolRegistry tools over stdio.

    This server implements the MCP protocol's tool-related methods:
    - ``initialize`` / ``notifications/initialized``
    - ``tools/list``
    - ``tools/call``
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process a single JSON-RPC request and return a response."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return self._respond(req_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })

        elif method == "notifications/initialized":
            # Notification, no response needed
            return self._respond(req_id, {})

        elif method == "tools/list":
            tools = []
            for defn in self.registry._tools.values():
                tool_entry: dict[str, Any] = {
                    "name": defn.name,
                    "description": defn.description,
                    "inputSchema": defn.input_schema,
                }
                tools.append(tool_entry)
            return self._respond(req_id, {"tools": tools})

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            handler = self.registry.get(tool_name)

            if handler is None:
                return self._error(req_id, -32602, f"Unknown tool: {tool_name}")

            try:
                result = handler(arguments)
                return self._respond(req_id, {
                    "content": [{"type": "text", "text": str(result)}],
                    "isError": False,
                })
            except Exception as exc:
                logger.exception("Tool '%s' failed", tool_name)
                return self._respond(req_id, {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                })

        else:
            return self._error(req_id, -32601, f"Method not found: {method}")

    def run_stdio(self) -> None:
        """Main loop: read JSON-RPC from stdin, write responses to stdout."""
        logger.info("MCP server starting (stdio transport)")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                response = {"jsonrpc": "2.0", "id": None, "error": {
                    "code": -32700, "message": f"Parse error: {exc}"
                }}
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

    @staticmethod
    def _respond(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def build_registry(
    tools: list[str] | None = None,
    extend: str | None = None,
) -> ToolRegistry:
    """Build a ToolRegistry with builtins and optional extensions.

    Parameters
    ----------
    tools : list[str] | None
        Restrict to these tool names.  None = all builtins.
    extend : str | None
        Import path ``module:function`` — function receives the registry
        and can register additional tools.  Example: ``my_tools:register``
    """
    registry = ToolRegistry()
    register_builtins(registry)

    if extend:
        module_path, func_name = extend.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        register_fn = getattr(mod, func_name)
        register_fn(registry)

    if tools is not None:
        all_names = registry.list_names()
        for name in all_names:
            if name not in tools:
                registry._tools.pop(name, None)

    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Runner MCP Server")
    parser.add_argument(
        "--tools", type=str, default=None,
        help="Comma-separated tool names to expose (default: all builtins)",
    )
    parser.add_argument(
        "--extend", type=str, default=None,
        help="Python module:function to register additional tools (e.g. my_tools:register)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # MCP uses stdout for JSON-RPC
    )

    tool_list = [t.strip() for t in args.tools.split(",")] if args.tools else None
    registry = build_registry(tools=tool_list, extend=args.extend)

    logger.info("Serving %d tools: %s", len(registry.list_names()), registry.list_names())

    server = MCPServer(registry)
    server.run_stdio()


if __name__ == "__main__":
    main()
