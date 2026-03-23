---
name: agent-runner
description: Use the agent-runner library to build and run Claude-powered agents with tool calling, shadow mode, approval gates, and MCP bridging. Activate when the user wants to create an agent, run tools in a loop, or set up shadow mode for safe experimentation.
---

# Agent Runner

Build and run Claude-powered agents with native tool calling.

## Quick Start (one line)

```python
from agent_runner import quick_run
result = quick_run("What is sqrt(144)?")
```

## Standard Setup

```python
from agent_runner import AgentRunner, ToolRegistry
from agent_runner.tools.builtins import register_builtins

registry = ToolRegistry()
register_builtins(registry)

runner = AgentRunner(
    system_prompt="You are a helpful assistant.",
    tools=registry,
)
result = runner.run("What is 2^10?")
```

## Register Custom Tools

### Shorthand (simple types)

```python
@registry.simple_tool("weather", "Get weather for a city", city=str)
def weather(params):
    return f"Sunny in {params['city']}"
```

### Full schema (complex types)

```python
@registry.tool(
    name="search",
    description="Search a database",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["query"],
    },
    category="database",
    is_write=False,
)
def search(params):
    return db.search(params["query"], limit=params.get("limit", 10))
```

## Multi-Turn Conversations

```python
conv = runner.conversation()
r1 = conv.ask("Hello!")
r2 = conv.ask("Follow up")  # history auto-carried
print(conv.total_usage.total_tokens)
```

## Shadow Mode (capture writes without executing)

```python
from agent_runner.interceptors import ShadowInterceptor

shadow = ShadowInterceptor.from_registry(registry)
runner = AgentRunner(
    system_prompt="...",
    tools=registry,
    interceptors=[shadow],
)
result = runner.run("Create a config file")
# Review what would have happened:
for cap in shadow.captured_writes:
    print(f"{cap['tool']}: {cap['input']}")
# Then replay if approved:
shadow.replay(registry)
```

## Approval Gates

```python
from agent_runner.interceptors import ApprovalInterceptor

# Interactive (stdin)
approver = ApprovalInterceptor(require_approval_for={"shell", "write_file"})

# Programmatic (for CI, bots, web UIs)
approver = ApprovalInterceptor(
    require_approval_for={"shell"},
    on_approval=lambda name, inp: my_approval_logic(name, inp),
)
```

## Bridge MCP Server Tools

```python
from agent_runner.mcp.client import MCPClient
from agent_runner.mcp.bridge import MCPToolBridge

client = MCPClient(command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
client.start()
bridge = MCPToolBridge(client=client, registry=registry, prefix="fs_")
bridge.bridge_tools()
# Now Claude can use fs_read_file, fs_write_file, etc.
```

## Configuration

Set via code, CLI flags, config file, or `AGENT_CONFIG` env var:

```json
{
  "model": "claude-sonnet-4-20250514",
  "max_turns": 25,
  "max_tokens": 8192,
  "shadow": true,
  "approve": ["shell", "write_file"],
  "stream": true
}
```

## Key APIs

- `quick_run(prompt)` — zero-setup one-liner
- `AgentRunner(system_prompt, tools, ...)` — full control
- `runner.conversation()` — multi-turn context
- `registry.simple_tool(name, desc, **params)` — shorthand tool registration
- `registry.write_tools()` / `registry.read_tools()` — metadata queries
- `result.failed_tools()` — inspect errors
- `ShadowInterceptor.from_registry(r)` — auto-configured shadow mode
- `HookManager(strict=True)` — lifecycle hooks with typo detection
