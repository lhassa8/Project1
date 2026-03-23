"""
Native tool-call loop runner for the Claude API.

This module owns the full request → tool-call → tool-result loop without
delegating to any framework.  Every tool call passes through an interception
pipeline before execution, giving callers full control over approval, logging,
mutation, and rollback.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from agent_runner.hooks import HookManager
from agent_runner.interceptors.base import InterceptAction, Interceptor
from agent_runner.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_TURNS = 25
DEFAULT_MODEL = "claude-sonnet-4-20250514"


@dataclass
class TokenUsage:
    """Tracks token consumption and estimated cost across an entire run."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def estimated_cost(self, model: str) -> float:
        """Rough cost estimate in USD.  Rates per million tokens."""
        rates = {
            "claude-sonnet-4-20250514": (3.0, 15.0),
            "claude-opus-4-20250514": (15.0, 75.0),
            "claude-haiku-3-5-20241022": (0.80, 4.0),
        }
        in_rate, out_rate = rates.get(model, (3.0, 15.0))
        return (self.input_tokens * in_rate + self.output_tokens * out_rate) / 1_000_000


@dataclass
class RunResult:
    """Holds everything produced by a single agent run."""

    final_text: str = ""
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns_used: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0


class AgentRunner:
    """Execute a multi-turn Claude conversation with native tool-call handling.

    Parameters
    ----------
    system_prompt : str
        The system prompt that frames the agent's behaviour.
    tools : ToolRegistry
        Registry of tool implementations the agent can invoke.
    interceptors : list[Interceptor] | None
        Ordered pipeline of interceptors applied to every tool call.
    model : str
        The Claude model to use.
    max_turns : int
        Hard ceiling on tool-call round-trips to prevent runaway loops.
    api_key : str | None
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    on_text : Callable[[str], None] | None
        Callback invoked for each text chunk during streaming.
        If provided, the runner uses the streaming API.
    hooks : HookManager | None
        Lifecycle event hooks for observing runner events.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: ToolRegistry,
        interceptors: list[Interceptor] | None = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        api_key: str | None = None,
        on_text: Callable[[str], None] | None = None,
        hooks: HookManager | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools
        self.interceptors = interceptors or []
        self.model = model
        self.max_turns = max_turns
        self.on_text = on_text
        self.hooks = hooks or HookManager()
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str, conversation: list[dict[str, Any]] | None = None) -> RunResult:
        """Run the agent loop to completion for a single user query.

        Parameters
        ----------
        user_message : str
            The new user message to process.
        conversation : list[dict] | None
            Optional prior conversation history.  If provided, the new message
            is appended and the full history is sent to the API.
        """
        if conversation is not None:
            messages = conversation
            messages.append({"role": "user", "content": user_message})
        else:
            messages = [{"role": "user", "content": user_message}]
        result = RunResult()
        result.messages = messages

        self.hooks.emit("run_start", user_message)

        t0 = time.monotonic()
        for turn in range(self.max_turns):
            self.hooks.emit("turn_start", turn + 1, messages)

            if self.on_text:
                response = self._call_api_streaming(messages)
            else:
                response = self._call_api(messages)

            assistant_content = response.content
            stop_reason = response.stop_reason

            # Track token usage
            if hasattr(response, "usage") and response.usage:
                result.usage.add(
                    input_tokens=getattr(response.usage, "input_tokens", 0),
                    output_tokens=getattr(response.usage, "output_tokens", 0),
                )

            # Append the full assistant turn
            messages.append({"role": "assistant", "content": assistant_content})

            self.hooks.emit("turn_end", turn + 1, stop_reason)

            # If the model stopped naturally, we're done
            if stop_reason == "end_turn":
                result.final_text = self._extract_text(assistant_content)
                result.turns_used = turn + 1
                break

            # Process tool calls
            if stop_reason == "tool_use":
                tool_results = self._process_tool_calls(assistant_content, result)
                messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("Agent hit max turns (%d)", self.max_turns)
            result.final_text = self._extract_text(assistant_content)
            result.turns_used = self.max_turns

        result.elapsed_seconds = time.monotonic() - t0
        self.hooks.emit("run_complete", result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict[str, Any]]) -> Any:
        """Single Claude API call (non-streaming)."""
        return self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.system_prompt,
            tools=self.tools.to_api_schema(),
            messages=messages,
        )

    def _call_api_streaming(self, messages: list[dict[str, Any]]) -> Any:
        """Streaming Claude API call — invokes on_text for each text delta."""
        with self._client.messages.stream(
            model=self.model,
            max_tokens=4096,
            system=self.system_prompt,
            tools=self.tools.to_api_schema(),
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                self.on_text(text)
            return stream.get_final_message()

    def _process_tool_calls(
        self, assistant_content: list, result: RunResult
    ) -> list[dict[str, Any]]:
        """Run each tool_use block through the interception pipeline, then execute."""
        tool_results: list[dict[str, Any]] = []

        for block in assistant_content:
            if block.type != "tool_use":
                continue

            call_record = {
                "tool": block.name,
                "input": block.input,
                "id": block.id,
                "action": None,
                "output": None,
            }

            # --- Interception pipeline ---
            action, modified_input = self._intercept(block.name, block.input)
            call_record["action"] = action.value

            if action == InterceptAction.ALLOW:
                output = self._execute_tool(block.name, modified_input)
                call_record["output"] = output
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(output)}
                )

            elif action == InterceptAction.DENY:
                deny_msg = "Tool call denied by interceptor."
                call_record["output"] = deny_msg
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": deny_msg, "is_error": True}
                )

            elif action == InterceptAction.MOCK:
                mock_output = modified_input  # interceptor replaces input with mock value
                call_record["output"] = mock_output
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(mock_output)}
                )

            result.tool_call_log.append(call_record)
            self.hooks.emit("tool_call", block.name, block.input, call_record["action"], call_record["output"])

        return tool_results

    def _intercept(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[InterceptAction, Any]:
        """Run the interceptor chain.  First non-ALLOW action wins."""
        current_input = tool_input
        for interceptor in self.interceptors:
            action, current_input = interceptor.intercept(tool_name, current_input)
            if action != InterceptAction.ALLOW:
                return action, current_input
        return InterceptAction.ALLOW, current_input

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> Any:
        """Look up and execute a registered tool."""
        handler = self.tools.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return handler(tool_input)
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", name)
            return f"Error executing tool '{name}': {exc}"

    @staticmethod
    def _extract_text(content: list) -> str:
        """Pull plain text out of an assistant response."""
        parts = [block.text for block in content if hasattr(block, "text")]
        return "\n".join(parts)
