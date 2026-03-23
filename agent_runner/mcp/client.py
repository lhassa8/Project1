"""MCP client — connects to an MCP server via stdio or SSE transport.

This is a lightweight client that speaks the MCP protocol to discover
tools from a remote server and forward tool calls to it.  It does NOT
depend on any MCP SDK — it implements just enough of the JSON-RPC
handshake to list tools and call them.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPToolSchema:
    """A tool definition received from an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClient:
    """Connect to an MCP server over stdio and call its tools.

    Parameters
    ----------
    command : list[str]
        The command to launch the MCP server process, e.g.
        ``["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]``.
    env : dict[str, str] | None
        Extra environment variables for the server process.
    """

    def __init__(self, command: list[str], env: dict[str, str] | None = None) -> None:
        self.command = command
        self.env = env
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._tools: dict[str, MCPToolSchema] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the MCP server subprocess."""
        import os

        merged_env = {**os.environ, **(self.env or {})}
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merged_env,
        )
        logger.info("Started MCP server: %s (pid=%d)", self.command, self._process.pid)
        self._initialize()

    def stop(self) -> None:
        """Shut down the MCP server subprocess."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=5)
            logger.info("Stopped MCP server (pid=%d)", self._process.pid)
            self._process = None

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # JSON-RPC over stdio
    # ------------------------------------------------------------------

    def _send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and read the response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("MCP server not started")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        line = json.dumps(request) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server closed stdout unexpectedly")

        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        return response.get("result", {})

    def _initialize(self) -> None:
        """Perform the MCP initialize handshake."""
        self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agent-runner", "version": "0.1.0"},
        })
        # Send initialized notification (no response expected for notifications,
        # but we send it as a request for simplicity in stdio mode)
        self._send("notifications/initialized")

    # ------------------------------------------------------------------
    # Tool discovery and invocation
    # ------------------------------------------------------------------

    def list_tools(self) -> list[MCPToolSchema]:
        """Discover tools from the MCP server."""
        result = self._send("tools/list")
        tools = []
        for t in result.get("tools", []):
            schema = MCPToolSchema(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            )
            self._tools[schema.name] = schema
            tools.append(schema)
        logger.info("Discovered %d MCP tools: %s", len(tools), [t.name for t in tools])
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool on the MCP server."""
        result = self._send("tools/call", {"name": name, "arguments": arguments})
        # MCP returns content as a list of content blocks
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else json.dumps(result)
