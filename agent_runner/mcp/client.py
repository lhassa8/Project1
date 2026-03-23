"""MCP client — connects to an MCP server via stdio or SSE transport.

This is a lightweight client that speaks the MCP protocol to discover
tools from a remote server and forward tool calls to it.  It does NOT
depend on any MCP SDK — it implements just enough of the JSON-RPC
handshake to list tools and call them.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
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
    timeout : float
        Timeout in seconds for waiting on server responses (default 30.0).
    max_retries : int
        Number of automatic reconnection attempts on server crash (default 2).
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.command = command
        self.env = env
        self.timeout = timeout
        self.max_retries = max_retries
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._tools: dict[str, MCPToolSchema] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the MCP server subprocess."""
        self._launch_process()
        self._initialize()

    def _launch_process(self) -> None:
        """Start the underlying subprocess (without MCP handshake)."""
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

    def stop(self) -> None:
        """Shut down the MCP server subprocess."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
            logger.info("Stopped MCP server (pid=%d)", self._process.pid)
            self._process = None

    def _restart(self) -> None:
        """Terminate the old process and start a new one."""
        logger.warning("Restarting MCP server process")
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                    self._process.wait(timeout=2)
                except Exception:
                    pass
            self._process = None
        self._request_id = 0
        self._launch_process()
        self._initialize()

    def health_check(self) -> bool:
        """Verify the server is alive by checking the process status.

        Returns True if the server process is running, False otherwise.
        """
        if not self._process:
            return False
        return self._process.poll() is None

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # JSON-RPC over stdio
    # ------------------------------------------------------------------

    def _read_with_timeout(self, timeout: float) -> str:
        """Read a line from stdout with timeout."""
        q: queue.Queue[str | Exception] = queue.Queue()

        def reader() -> None:
            try:
                line = self._process.stdout.readline()  # type: ignore[union-attr]
                q.put(line)
            except Exception as e:
                q.put(e)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            result = q.get(timeout=timeout)
            if isinstance(result, Exception):
                raise result
            return result  # type: ignore[return-value]
        except queue.Empty:
            raise TimeoutError(
                f"MCP server did not respond within {timeout}s"
            )

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

        response_line = self._read_with_timeout(self.timeout)
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
        """Invoke a tool on the MCP server.

        If the server crashes mid-call, automatically reconnects and retries
        up to ``max_retries`` times.
        """
        last_error: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                result = self._send("tools/call", {"name": name, "arguments": arguments})
                # MCP returns content as a list of content blocks
                content = result.get("content", [])
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts) if texts else json.dumps(result)
            except (RuntimeError, TimeoutError, OSError, BrokenPipeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "MCP call_tool failed (attempt %d/%d): %s — reconnecting",
                        attempt + 1,
                        1 + self.max_retries,
                        exc,
                    )
                    try:
                        self._restart()
                    except Exception as restart_exc:
                        logger.error("Failed to restart MCP server: %s", restart_exc)
                        raise last_error from restart_exc
                else:
                    raise
        # Should not reach here, but just in case
        raise last_error  # type: ignore[misc]
