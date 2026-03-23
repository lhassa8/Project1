# Agent Runner

A native Claude tool-call loop with a composable interception layer. No frameworks — you own the loop.

## Architecture

```
User Prompt
    │
    ▼
┌──────────────────┐
│   AgentRunner     │  ← owns the request → response → tool-call loop
│                    │
│  ┌──────────────┐ │
│  │ Claude API    │ │  ← direct anthropic SDK calls
│  └──────┬───────┘ │
│         │         │
│  ┌──────▼───────┐ │
│  │ Interceptor   │ │  ← logging → approval → shadow → ...
│  │ Pipeline      │ │
│  └──────┬───────┘ │
│         │         │
│  ┌──────▼───────┐ │
│  │ ToolRegistry  │ │  ← calculator, shell, read_file, write_file, ...
│  └──────────────┘ │
└──────────────────┘
```

## Quick Start

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-...

# Simple one-shot
python main.py "What is sqrt(144) * 3?"

# Shadow mode — captures writes, doesn't execute them
python main.py --shadow "Create a file called plan.md with a project outline"

# Approval gate on dangerous tools
python main.py --approve shell,write_file "List all Python files"

# Interactive mode
python main.py
```

## Interceptors

Interceptors form a pipeline that every tool call passes through before execution:

| Interceptor | Purpose |
|---|---|
| `LoggingInterceptor` | Records all tool calls for audit/replay |
| `ApprovalInterceptor` | Human-in-the-loop gate for sensitive tools |
| `ShadowInterceptor` | Captures writes without executing — "dry run" mode |

Custom interceptors implement `Interceptor.intercept()` and return `(InterceptAction, data)`.

## Adding Custom Tools

```python
from agent_runner.tools.registry import ToolRegistry

registry = ToolRegistry()

@registry.tool(
    name="lookup_user",
    description="Look up a user by email",
    input_schema={
        "type": "object",
        "properties": {"email": {"type": "string"}},
        "required": ["email"],
    },
)
def lookup_user(params):
    return db.users.find_one({"email": params["email"]})
```

## Phase 2 Roadmap

- **MCP server passthrough** with shadow mode — run against real systems, capture all writes for rollback
- **Shareable run links** — send an approval request to a stakeholder for async review
- **Streaming support** — stream partial responses while tools execute
