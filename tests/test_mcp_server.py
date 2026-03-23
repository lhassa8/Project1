"""Tests for the MCP server that exposes agent-runner tools."""

from __future__ import annotations

import json

from agent_runner.mcp_server import MCPServer, build_registry
from agent_runner.tools.registry import ToolRegistry


class TestMCPServer:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.registry.register(
            "echo", "Echo input back",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            lambda p: f"Echo: {p['text']}",
        )
        self.server = MCPServer(self.registry)

    def _request(self, method: str, params: dict | None = None, req_id: int = 1) -> dict:
        req = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params:
            req["params"] = params
        return self.server.handle_request(req)

    def test_initialize(self):
        resp = self._request("initialize")
        assert resp["result"]["serverInfo"]["name"] == "agent-runner"
        assert "tools" in resp["result"]["capabilities"]

    def test_tools_list(self):
        resp = self._request("tools/list")
        tools = resp["result"]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "echo"
        assert tools[0]["inputSchema"]["type"] == "object"

    def test_tools_call_success(self):
        resp = self._request("tools/call", {"name": "echo", "arguments": {"text": "hello"}})
        content = resp["result"]["content"]
        assert content[0]["text"] == "Echo: hello"
        assert resp["result"]["isError"] is False

    def test_tools_call_unknown_tool(self):
        resp = self._request("tools/call", {"name": "nope", "arguments": {}})
        assert "error" in resp
        assert "Unknown tool" in resp["error"]["message"]

    def test_tools_call_handler_error(self):
        self.registry.register(
            "fail", "Always fails",
            {"type": "object", "properties": {}},
            lambda p: 1 / 0,
        )
        resp = self._request("tools/call", {"name": "fail", "arguments": {}})
        assert resp["result"]["isError"] is True
        assert "Error" in resp["result"]["content"][0]["text"]

    def test_unknown_method(self):
        resp = self._request("foo/bar")
        assert "error" in resp
        assert resp["error"]["code"] == -32601


class TestBuildRegistry:
    def test_default_builds_all_builtins(self):
        registry = build_registry()
        names = registry.list_names()
        assert "calculator" in names
        assert "shell" in names
        assert "read_file" in names

    def test_filter_tools(self):
        registry = build_registry(tools=["calculator", "read_file"])
        names = registry.list_names()
        assert "calculator" in names
        assert "read_file" in names
        assert "shell" not in names
        assert "write_file" not in names

    def test_builtins_with_metadata(self):
        registry = build_registry()
        assert "shell" in registry.write_tools()
        assert "calculator" in registry.read_tools()
