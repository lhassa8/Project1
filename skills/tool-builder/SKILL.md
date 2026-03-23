---
name: tool-builder
description: Create custom tools for the agent-runner framework. Activate when the user wants to build a new tool, define tool schemas, or extend the agent with custom capabilities.
---

# Tool Builder

Create custom tools for the agent-runner framework.

## Simple Tools (recommended for most cases)

Use `simple_tool` when your tool has basic typed parameters:

```python
from agent_runner import ToolRegistry

registry = ToolRegistry()

@registry.simple_tool("translate", "Translate text to a language", text=str, target_lang=str)
def translate(params):
    # Your implementation here
    return translated_text
```

Supported types: `str`, `int`, `float`, `bool`.

## Full Schema Tools

Use the full decorator when you need optional params, defaults, enums, or nested objects:

```python
@registry.tool(
    name="query_db",
    description="Query a database with SQL",
    input_schema={
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SQL query to execute"},
            "database": {
                "type": "string",
                "enum": ["production", "staging", "dev"],
                "description": "Which database to query"
            },
            "limit": {"type": "integer", "default": 100}
        },
        "required": ["sql"],
    },
    category="database",
    is_write=False,
)
def query_db(params):
    return execute_query(params["sql"], db=params.get("database", "dev"))
```

## Tool Metadata

Mark tools with metadata for automatic shadow mode and approval configuration:

```python
@registry.simple_tool("deploy", "Deploy to environment",
                       category="infrastructure", is_write=True,
                       env=str, version=str)
def deploy(params):
    ...
```

- `category` — groups tools (`"file"`, `"network"`, `"database"`, `"system"`, etc.)
- `is_write=True` — marks tool as a write operation (auto-captured in shadow mode)

## Imperative Registration

For dynamic tool creation (e.g., from config files or plugin discovery):

```python
def make_api_tool(endpoint, method="GET"):
    def handler(params):
        return requests.request(method, endpoint, json=params).json()

    registry.register(
        name=f"api_{endpoint.split('/')[-1]}",
        description=f"{method} {endpoint}",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
        category="api",
        is_write=(method != "GET"),
    )
```

## Best Practices

1. **Return strings** — tool output goes to Claude as text; return human-readable strings
2. **Handle errors gracefully** — return error messages as strings rather than raising
3. **Set `is_write`** — so shadow mode captures mutations automatically
4. **Use categories** — for organization and filtering with `registry.list_by_category()`
5. **Keep descriptions clear** — Claude uses the description to decide when to call the tool
6. **Validate at registration** — schema is checked automatically; broken schemas fail fast
