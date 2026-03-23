"""Tests for MCP client and bridge (mocked — no real MCP server needed)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import json

from agent_runner.mcp.client import MCPClient, MCPToolSchema
from agent_runner.mcp.bridge import MCPToolBridge
from agent_runner.tools.registry import ToolRegistry


class TestMCPClient:
    def test_tool_schema_dataclass(self):
        t = MCPToolSchema(name="test", description="A test tool", input_schema={"type": "object"})
        assert t.name == "test"

    @patch("agent_runner.mcp.client.subprocess.Popen")
    def test_list_tools(self, mock_popen):
        proc = MagicMock()
        mock_popen.return_value = proc
        proc.pid = 1234

        # initialize response, initialized response, tools/list response
        proc.stdout.readline.side_effect = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n",
            json.dumps({"jsonrpc": "2.0", "id": 2, "result": {}}) + "\n",
            json.dumps({
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Read a file",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                            },
                        }
                    ]
                },
            }) + "\n",
        ]

        client = MCPClient(command=["fake-server"])
        client.start()
        tools = client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "read_file"
        client.stop()

    @patch("agent_runner.mcp.client.subprocess.Popen")
    def test_call_tool(self, mock_popen):
        proc = MagicMock()
        mock_popen.return_value = proc
        proc.pid = 1234

        proc.stdout.readline.side_effect = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n",
            json.dumps({"jsonrpc": "2.0", "id": 2, "result": {}}) + "\n",
            json.dumps({
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "content": [{"type": "text", "text": "file contents here"}]
                },
            }) + "\n",
        ]

        client = MCPClient(command=["fake-server"])
        client.start()
        result = client.call_tool("read_file", {"path": "/tmp/test.txt"})
        assert "file contents here" in result
        client.stop()


class TestMCPToolBridge:
    def test_bridge_registers_tools(self):
        registry = ToolRegistry()
        client = MagicMock()
        client.list_tools.return_value = [
            MCPToolSchema("read", "Read", {"type": "object"}),
            MCPToolSchema("write", "Write", {"type": "object"}),
        ]
        client.call_tool.return_value = "result"

        bridge = MCPToolBridge(
            client=client,
            registry=registry,
            prefix="mcp_",
            write_tool_names={"write"},
        )
        names = bridge.bridge_tools()
        assert names == ["mcp_read", "mcp_write"]
        assert registry.get("mcp_read") is not None
        assert registry.get("mcp_write") is not None

    def test_get_write_and_read_tools(self):
        registry = ToolRegistry()
        client = MagicMock()
        client.list_tools.return_value = [
            MCPToolSchema("read", "Read", {"type": "object"}),
            MCPToolSchema("write", "Write", {"type": "object"}),
        ]

        bridge = MCPToolBridge(
            client=client,
            registry=registry,
            prefix="mcp_",
            write_tool_names={"write"},
        )
        bridge.bridge_tools()
        assert bridge.get_write_tools() == {"mcp_write"}
        assert bridge.get_read_tools() == {"mcp_read"}

    def test_bridged_handler_calls_mcp(self):
        registry = ToolRegistry()
        client = MagicMock()
        client.list_tools.return_value = [
            MCPToolSchema("echo", "Echo", {"type": "object"}),
        ]
        client.call_tool.return_value = "echoed!"

        bridge = MCPToolBridge(client=client, registry=registry)
        bridge.bridge_tools()

        handler = registry.get("echo")
        result = handler({"msg": "hi"})
        assert result == "echoed!"
        client.call_tool.assert_called_once_with("echo", {"msg": "hi"})
