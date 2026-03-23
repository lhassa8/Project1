---
description: Run a prompt through the agent-runner tool-call loop with built-in tools
disable-model-invocation: false
---

Run the user's prompt through the agent-runner Python library. Use the `quick_run` function for simple cases or set up a full `AgentRunner` with custom tools and interceptors for complex cases.

Steps:
1. If the user provided a simple prompt, use `quick_run()`
2. If they need custom tools, shadow mode, or approval gates, set up a full runner
3. Show the result including tool calls made and token usage

```python
from agent_runner import quick_run
result = quick_run("<user's prompt here>")
print(result.final_text)
print(f"Tools used: {len(result.tool_call_log)}, Tokens: {result.usage.total_tokens}")
```
