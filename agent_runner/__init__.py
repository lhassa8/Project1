"""Agent Runner — a native Claude tool-call loop with an interception layer.

Quick start::

    from agent_runner import AgentRunner, ToolRegistry, HookManager
    from agent_runner.tools.builtins import register_builtins
    from agent_runner.interceptors import ShadowInterceptor

    registry = ToolRegistry()
    register_builtins(registry)

    hooks = HookManager()

    @hooks.on("tool_call")
    def on_tool(name, inp, action, output):
        print(f"[{action}] {name}")

    runner = AgentRunner(
        system_prompt="You are a helpful assistant.",
        tools=registry,
        hooks=hooks,
    )
    result = runner.run("What is 2^10?")
    print(result.final_text)
"""

from agent_runner.config import AgentConfig
from agent_runner.hooks import HookManager
from agent_runner.interceptors.base import InterceptAction, Interceptor
from agent_runner.runner import AgentRunner, RunResult, TokenUsage
from agent_runner.tools.registry import ToolRegistry

__all__ = [
    "AgentConfig",
    "AgentRunner",
    "HookManager",
    "InterceptAction",
    "Interceptor",
    "RunResult",
    "TokenUsage",
    "ToolRegistry",
]
