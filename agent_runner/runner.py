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
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from agent_runner.events import EventStream, make_event as _make_event
from agent_runner.hooks import HookManager
from agent_runner.interceptors.base import InterceptAction, Interceptor
from agent_runner.sandbox import ResourceLimits, SandboxViolation
from agent_runner.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_TURNS = 25
DEFAULT_MAX_TOKENS = 4096
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


class ContextManager:
    """Manage conversation history within token budgets."""

    def __init__(
        self,
        max_context_tokens: int = 100_000,
        strategy: str = "sliding_window",  # "sliding_window" or "keep_ends"
        reserve_tokens: int = 4096,        # reserve for response
    ):
        self.max_context_tokens = max_context_tokens
        self.strategy = strategy
        self.reserve_tokens = reserve_tokens

    def _estimate_tokens(self, message: dict) -> int:
        """Estimate tokens for a single message (~4 chars per token)."""
        content = message.get("content", "")
        if isinstance(content, str):
            char_count = len(content)
        elif isinstance(content, list):
            char_count = len(json.dumps(content))
        else:
            char_count = len(str(content))
        return max(char_count // 4, 1)

    def _total_tokens(self, messages: list[dict]) -> int:
        """Estimate total tokens for a list of messages."""
        return sum(self._estimate_tokens(m) for m in messages)

    def trim(self, messages: list[dict]) -> list[dict]:
        """Trim messages to fit within the token budget.

        Strategies:
        - sliding_window: keep last N messages that fit
        - keep_ends: keep first message + last N messages that fit

        Token estimation: ~4 chars per token (rough but fast).
        """
        budget = self.max_context_tokens - self.reserve_tokens
        if budget <= 0:
            return messages[-1:] if messages else []

        if self._total_tokens(messages) <= budget:
            return messages

        if self.strategy == "keep_ends":
            return self._trim_keep_ends(messages, budget)
        else:
            return self._trim_sliding_window(messages, budget)

    def _trim_sliding_window(self, messages: list[dict], budget: int) -> list[dict]:
        """Keep last N messages that fit within budget."""
        result: list[dict] = []
        running = 0
        for msg in reversed(messages):
            cost = self._estimate_tokens(msg)
            if running + cost > budget:
                break
            result.append(msg)
            running += cost
        result.reverse()
        return result

    def _trim_keep_ends(self, messages: list[dict], budget: int) -> list[dict]:
        """Keep first message + last N messages that fit."""
        if not messages:
            return []
        if len(messages) <= 2:
            return list(messages)

        first = messages[0]
        first_cost = self._estimate_tokens(first)
        remaining_budget = budget - first_cost
        if remaining_budget <= 0:
            return [first]

        # Fill from the end
        tail: list[dict] = []
        running = 0
        for msg in reversed(messages[1:]):
            cost = self._estimate_tokens(msg)
            if running + cost > remaining_budget:
                break
            tail.append(msg)
            running += cost
        tail.reverse()
        return [first] + tail


@dataclass
class RunResult:
    """Holds everything produced by a single agent run."""

    final_text: str = ""
    tool_call_log: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns_used: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    elapsed_seconds: float = 0.0

    def failed_tools(self) -> list[dict[str, Any]]:
        """Return tool calls that resulted in errors."""
        return [c for c in self.tool_call_log if c.get("error")]


class Conversation:
    """Manages multi-turn conversation state.

    Usage::

        conv = runner.conversation()
        r1 = conv.ask("Hello!")
        r2 = conv.ask("Follow up question")   # carries full history
        r3 = conv.ask("One more")
        print(conv.total_usage.total_tokens)
    """

    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner
        self._messages: list[dict[str, Any]] = []
        self.total_usage = TokenUsage()
        self.results: list[RunResult] = []

    def ask(self, message: str) -> RunResult:
        """Send a message and get the response, carrying forward full history."""
        result = self._runner.run(message, conversation=self._messages)
        self._messages = result.messages
        self.total_usage.add(result.usage.input_tokens, result.usage.output_tokens)
        self.results.append(result)
        return result

    def reset(self) -> None:
        """Clear conversation history and start fresh."""
        self._messages.clear()
        self.results.clear()
        self.total_usage = TokenUsage()

    def save(self, path: str) -> None:
        """Save conversation state to a JSON file."""
        data = {
            "messages": self._messages,
            "total_usage": {
                "input_tokens": self.total_usage.input_tokens,
                "output_tokens": self.total_usage.output_tokens,
            },
            "metadata": {
                "model": self._runner.model,
                "timestamp": time.time(),
            },
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: str) -> None:
        """Load conversation state from a JSON file."""
        data = json.loads(Path(path).read_text())
        self._messages = data["messages"]
        usage = data.get("total_usage", {})
        self.total_usage = TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


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
    max_tokens : int
        Maximum response tokens per API call.
    api_key : str | None
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    on_text : Callable[[str], None] | None
        Callback invoked for each text chunk during streaming.
        If provided, the runner uses the streaming API.
    hooks : HookManager | None
        Lifecycle event hooks for observing runner events.
    retries : int
        Number of retries on transient API errors (rate limits, timeouts).
    retry_delay : float
        Initial delay in seconds between retries (doubles each attempt).
    resource_limits : ResourceLimits | None
        Resource limits (max tool calls, output size, cost budget, timeouts).
    context_manager : ContextManager | None
        Optional context window manager to trim messages before API calls.
    event_stream : EventStream | None
        Optional event stream for structured observability.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: ToolRegistry,
        interceptors: list[Interceptor] | None = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
        on_text: Callable[[str], None] | None = None,
        hooks: HookManager | None = None,
        retries: int = 2,
        retry_delay: float = 1.0,
        resource_limits: ResourceLimits | None = None,
        context_manager: ContextManager | None = None,
        event_stream: EventStream | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools
        self.interceptors = interceptors or []
        self.model = model
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.on_text = on_text
        self.hooks = hooks or HookManager()
        self.retries = retries
        self.retry_delay = retry_delay
        self.resource_limits = resource_limits or ResourceLimits()
        self.context_manager = context_manager
        self.event_stream = event_stream
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

        run_id = uuid.uuid4().hex[:12]

        self.hooks.emit("run_start", user_message)
        if self.event_stream:
            self.event_stream.record(_make_event(
                "run_start", run_id, 0, {"prompt": user_message}
            ))

        t0 = time.monotonic()
        for turn in range(self.max_turns):
            self.hooks.emit("turn_start", turn + 1, messages)
            if self.event_stream:
                self.event_stream.record(_make_event(
                    "turn_start", run_id, turn + 1, {}
                ))

            # Apply context window trimming before API call
            api_messages = messages
            if self.context_manager:
                api_messages = self.context_manager.trim(messages)

            if self.on_text:
                response = self._call_api_streaming(api_messages)
            else:
                response = self._call_api(api_messages)

            assistant_content = response.content
            stop_reason = response.stop_reason

            # Track token usage
            input_toks = 0
            output_toks = 0
            if hasattr(response, "usage") and response.usage:
                input_toks = getattr(response.usage, "input_tokens", 0)
                output_toks = getattr(response.usage, "output_tokens", 0)
                result.usage.add(
                    input_tokens=input_toks,
                    output_tokens=output_toks,
                )

            if self.event_stream:
                self.event_stream.record(_make_event(
                    "api_request", run_id, turn + 1,
                    {"input_tokens": input_toks, "output_tokens": output_toks},
                ))

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
                tool_results = self._process_tool_calls(
                    assistant_content, result, run_id=run_id, turn=turn + 1,
                )
                messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("Agent hit max turns (%d)", self.max_turns)
            result.final_text = self._extract_text(assistant_content)
            result.turns_used = self.max_turns

        result.elapsed_seconds = time.monotonic() - t0
        self.hooks.emit("run_complete", result)
        if self.event_stream:
            self.event_stream.record(_make_event(
                "run_complete", run_id, result.turns_used,
                {
                    "total_input_tokens": result.usage.input_tokens,
                    "total_output_tokens": result.usage.output_tokens,
                    "elapsed_seconds": result.elapsed_seconds,
                    "cost_usd": result.usage.estimated_cost(self.model),
                },
            ))
        return result

    def estimate_cost(self, prompt: str, expected_turns: int = 3) -> dict:
        """Estimate cost before running.

        Returns {"min_usd": float, "max_usd": float, "model": str}
        """
        # Estimate prompt tokens (~4 chars per token)
        prompt_tokens = max(len(prompt) // 4, 1)
        # System prompt tokens
        system_tokens = max(len(self.system_prompt) // 4, 1)
        base_input = prompt_tokens + system_tokens

        rates = {
            "claude-sonnet-4-20250514": (3.0, 15.0),
            "claude-opus-4-20250514": (15.0, 75.0),
            "claude-haiku-3-5-20241022": (0.80, 4.0),
        }
        in_rate, out_rate = rates.get(self.model, (3.0, 15.0))

        # Min: single turn, short response
        min_input = base_input
        min_output = 200  # minimal response
        min_cost = (min_input * in_rate + min_output * out_rate) / 1_000_000

        # Max: expected_turns turns, each with max_tokens output
        # Each turn adds previous context to input
        total_input = 0
        for t in range(expected_turns):
            total_input += base_input + t * self.max_tokens
        total_output = expected_turns * self.max_tokens
        max_cost = (total_input * in_rate + total_output * out_rate) / 1_000_000

        return {"min_usd": min_cost, "max_usd": max_cost, "model": self.model}

    def conversation(self) -> Conversation:
        """Create a new multi-turn conversation context.

        Usage::

            conv = runner.conversation()
            r1 = conv.ask("Hello")
            r2 = conv.ask("Follow up")  # carries history automatically
        """
        return Conversation(self)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_with_retry(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call *fn* with exponential backoff on transient errors."""
        delay = self.retry_delay
        last_exc = None
        for attempt in range(1 + self.retries):
            try:
                return fn(*args, **kwargs)
            except (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ) as exc:
                last_exc = exc
                if attempt < self.retries:
                    logger.warning(
                        "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, 1 + self.retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
        raise last_exc  # type: ignore[misc]

    def _call_api(self, messages: list[dict[str, Any]]) -> Any:
        """Single Claude API call (non-streaming) with retry."""
        return self._call_with_retry(
            self._client.messages.create,
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            tools=self.tools.to_api_schema(),
            messages=messages,
        )

    def _call_api_streaming(self, messages: list[dict[str, Any]]) -> Any:
        """Streaming Claude API call — invokes on_text for each text delta."""
        def _do_stream() -> Any:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=self.tools.to_api_schema(),
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    self.on_text(text)
                return stream.get_final_message()
        return self._call_with_retry(_do_stream)

    def _process_tool_calls(
        self, assistant_content: list, result: RunResult,
        run_id: str = "", turn: int = 0,
    ) -> list[dict[str, Any]]:
        """Run each tool_use block through the interception pipeline, then execute."""
        tool_results: list[dict[str, Any]] = []

        for block in assistant_content:
            if block.type != "tool_use":
                continue

            # --- Resource limit: tool call count ---
            try:
                self.resource_limits.check_tool_count(len(result.tool_call_log) + 1)
            except SandboxViolation as exc:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Error: {exc}", "is_error": True,
                })
                continue

            # --- Resource limit: cost budget ---
            try:
                self.resource_limits.check_cost(result.usage.estimated_cost(self.model))
            except SandboxViolation as exc:
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": f"Error: {exc}", "is_error": True,
                })
                continue

            call_record: dict[str, Any] = {
                "tool": block.name,
                "input": block.input,
                "id": block.id,
                "action": None,
                "output": None,
                "error": None,
            }

            # --- Interception pipeline ---
            action, modified_input = self._intercept(block.name, block.input)
            call_record["action"] = action.value

            tool_t0 = time.monotonic()

            if action == InterceptAction.ALLOW:
                output = self._execute_tool(block.name, modified_input, call_record)
                output_str = self.resource_limits.truncate_output(str(output))
                call_record["output"] = output_str
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output_str}
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

            tool_duration_ms = (time.monotonic() - tool_t0) * 1000

            result.tool_call_log.append(call_record)
            self.hooks.emit("tool_call", block.name, block.input, call_record["action"], call_record["output"])

            if self.event_stream:
                event_data: dict[str, Any] = {
                    "tool": block.name,
                    "action": call_record["action"],
                    "duration_ms": tool_duration_ms,
                }
                if call_record["error"]:
                    self.event_stream.record(_make_event(
                        "error", run_id, turn, {"tool": block.name, "error": call_record["error"]}
                    ))
                    event_data["error"] = call_record["error"]
                self.event_stream.record(_make_event(
                    "tool_call", run_id, turn, event_data,
                ))

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

    def _execute_tool(self, name: str, tool_input: dict[str, Any], call_record: dict[str, Any] | None = None) -> Any:
        """Look up and execute a registered tool."""
        handler = self.tools.get(name)
        if handler is None:
            error_msg = f"Error: unknown tool '{name}'"
            if call_record is not None:
                call_record["error"] = error_msg
            return error_msg
        try:
            return handler(tool_input)
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", name)
            error_msg = f"Error executing tool '{name}': {exc}"
            if call_record is not None:
                call_record["error"] = error_msg
            return error_msg

    @staticmethod
    def _extract_text(content: list) -> str:
        """Pull plain text out of an assistant response."""
        parts = [block.text for block in content if hasattr(block, "text")]
        return "\n".join(parts)
