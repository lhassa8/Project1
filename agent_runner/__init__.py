"""Agent Runner — a native Claude tool-call loop with full control.

Quick start (one line)::

    from agent_runner import quick_run
    result = quick_run("What is sqrt(144)?")
    print(result.final_text)   # "12.0"

Standard usage::

    from agent_runner import AgentRunner, ToolRegistry
    from agent_runner.tools.builtins import register_builtins

    registry = ToolRegistry()
    register_builtins(registry)

    runner = AgentRunner(
        system_prompt="You are a helpful assistant.",
        tools=registry,
    )
    result = runner.run("What is 2^10?")
    print(result.final_text)

Multi-turn conversation::

    conv = runner.conversation()
    conv.ask("Hello!")
    conv.ask("What did I just say?")  # carries full history

Custom tools (shorthand)::

    @registry.simple_tool("weather", "Get weather", city=str)
    def weather(params):
        return f"Sunny in {params['city']}"

Streaming::

    runner = AgentRunner(
        system_prompt="...",
        tools=registry,
        on_text=lambda chunk: print(chunk, end="", flush=True),
    )

Shadow mode with approval::

    from agent_runner.interceptors import ShadowInterceptor

    shadow = ShadowInterceptor(
        write_tools=registry.write_tools(),
        read_tools=registry.read_tools(),
    )
    runner = AgentRunner(
        system_prompt="...",
        tools=registry,
        interceptors=[shadow],
    )
"""

from agent_runner.config import AgentConfig
from agent_runner.hooks import HookManager
from agent_runner.interceptors.base import InterceptAction, Interceptor
from agent_runner.runner import AgentRunner, Conversation, RunResult, TokenUsage
from agent_runner.tools.registry import ToolRegistry

__all__ = [
    "AgentConfig",
    "AgentRunner",
    "Conversation",
    "HookManager",
    "InterceptAction",
    "Interceptor",
    "RunResult",
    "TokenUsage",
    "ToolRegistry",
    "quick_run",
]


def quick_run(
    prompt: str,
    *,
    tools: list[str] | None = None,
    model: str | None = None,
    system_prompt: str = "You are a helpful assistant with access to tools. Use them as needed.",
    shadow: bool = False,
    stream: bool = False,
    max_turns: int = 25,
    max_tokens: int = 4096,
    api_key: str | None = None,
) -> RunResult:
    """Run a single prompt with sensible defaults — zero boilerplate.

    Parameters
    ----------
    prompt : str
        The user prompt.
    tools : list[str] | None
        Which built-in tools to enable.  ``None`` = all builtins.
        Example: ``["calculator", "read_file"]``
    model : str | None
        Claude model.  Defaults to Sonnet.
    system_prompt : str
        System prompt.
    shadow : bool
        If True, capture writes instead of executing them.
    stream : bool
        If True, print tokens to stdout as they arrive.
    max_turns : int
        Max tool-call round trips.
    max_tokens : int
        Max response tokens per API call.
    api_key : str | None
        Anthropic API key (falls back to env var).

    Returns
    -------
    RunResult
        The complete result including text, tool log, usage, and timing.

    Examples
    --------
    >>> result = quick_run("What is 2 + 2?")
    >>> result = quick_run("List files in /tmp", tools=["list_files"])
    >>> result = quick_run("Write hello.txt", shadow=True)
    """
    from agent_runner.tools.builtins import register_builtins
    from agent_runner.interceptors.logging import LoggingInterceptor
    from agent_runner.interceptors.shadow import ShadowInterceptor

    registry = ToolRegistry()
    register_builtins(registry)

    # Filter to requested tools only
    if tools is not None:
        all_names = registry.list_names()
        for name in all_names:
            if name not in tools:
                registry._tools.pop(name, None)

    interceptors: list[Interceptor] = [LoggingInterceptor()]

    if shadow:
        interceptors.append(ShadowInterceptor(
            write_tools=registry.write_tools(),
            read_tools=registry.read_tools(),
        ))

    on_text = None
    if stream:
        on_text = lambda chunk: print(chunk, end="", flush=True)

    kwargs: dict[str, any] = {
        "system_prompt": system_prompt,
        "tools": registry,
        "interceptors": interceptors,
        "max_turns": max_turns,
        "max_tokens": max_tokens,
        "on_text": on_text,
    }
    if model:
        kwargs["model"] = model
    if api_key:
        kwargs["api_key"] = api_key

    runner = AgentRunner(**kwargs)
    result = runner.run(prompt)

    if stream:
        print()  # newline after streamed output

    return result
