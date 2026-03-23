"""Integration tests for MCP client with real subprocesses."""

from __future__ import annotations

import sys
import pytest

from agent_runner.mcp.client import MCPClient


# A simple test MCP server script that handles basic MCP protocol
TEST_SERVER_SCRIPT = '''
import json, sys
for line in sys.stdin:
    req = json.loads(line.strip())
    method = req.get("method", "")
    rid = req.get("id")
    if method == "initialize":
        resp = {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "test", "version": "0.1"}}}
    elif method == "notifications/initialized":
        resp = {"jsonrpc": "2.0", "id": rid, "result": {}}
    elif method == "tools/list":
        resp = {"jsonrpc": "2.0", "id": rid, "result": {"tools": [{"name": "echo", "description": "Echo back", "inputSchema": {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}}]}}
    elif method == "tools/call":
        args = req.get("params", {}).get("arguments", {})
        resp = {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": f"echo: {args.get('msg', '')}"}]}}
    else:
        resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "not found"}}
    sys.stdout.write(json.dumps(resp) + "\\n")
    sys.stdout.flush()
'''

# A server that sleeps forever on any request (for timeout testing)
SLOW_SERVER_SCRIPT = '''
import json, sys, time
for line in sys.stdin:
    req = json.loads(line.strip())
    rid = req.get("id")
    method = req.get("method", "")
    if method == "initialize":
        resp = {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "slow", "version": "0.1"}}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif method == "notifications/initialized":
        resp = {"jsonrpc": "2.0", "id": rid, "result": {}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    else:
        # Sleep forever — never respond
        time.sleep(999)
'''


class TestMCPIntegrationListTools:
    def test_real_mcp_list_tools(self, tmp_path):
        """Start a real MCP server subprocess and list tools."""
        script = tmp_path / "server.py"
        script.write_text(TEST_SERVER_SCRIPT)

        client = MCPClient(command=[sys.executable, str(script)], timeout=10.0)
        try:
            client.start()
            tools = client.list_tools()
            assert len(tools) == 1
            assert tools[0].name == "echo"
            assert tools[0].description == "Echo back"
            assert "msg" in tools[0].input_schema.get("properties", {})
        finally:
            client.stop()


class TestMCPIntegrationCallTool:
    def test_real_mcp_call_tool(self, tmp_path):
        """Start a real MCP server subprocess and call a tool."""
        script = tmp_path / "server.py"
        script.write_text(TEST_SERVER_SCRIPT)

        client = MCPClient(command=[sys.executable, str(script)], timeout=10.0)
        try:
            client.start()
            result = client.call_tool("echo", {"msg": "hello world"})
            assert result == "echo: hello world"
        finally:
            client.stop()


class TestMCPIntegrationTimeout:
    def test_timeout_on_slow_server(self, tmp_path):
        """A server that never responds should trigger TimeoutError."""
        script = tmp_path / "slow_server.py"
        script.write_text(SLOW_SERVER_SCRIPT)

        client = MCPClient(
            command=[sys.executable, str(script)],
            timeout=1.0,
            max_retries=0,  # Don't retry so the test is fast
        )
        try:
            client.start()
            with pytest.raises(TimeoutError, match="did not respond within"):
                client.call_tool("echo", {"msg": "hi"})
        finally:
            client.stop()


class TestMCPIntegrationHealthCheck:
    def test_health_check(self, tmp_path):
        """Health check returns True for a running server."""
        script = tmp_path / "server.py"
        script.write_text(TEST_SERVER_SCRIPT)

        client = MCPClient(command=[sys.executable, str(script)], timeout=10.0)
        try:
            client.start()
            assert client.health_check() is True
        finally:
            client.stop()

    def test_health_check_not_started(self):
        """Health check returns False when no process is running."""
        client = MCPClient(command=["false"])
        assert client.health_check() is False

    def test_health_check_after_stop(self, tmp_path):
        """Health check returns False after the server is stopped."""
        script = tmp_path / "server.py"
        script.write_text(TEST_SERVER_SCRIPT)

        client = MCPClient(command=[sys.executable, str(script)], timeout=10.0)
        client.start()
        client.stop()
        assert client.health_check() is False
