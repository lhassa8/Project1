"""Base interceptor interface."""

from __future__ import annotations

import enum
from typing import Any


class InterceptAction(enum.Enum):
    """What the runner should do with a tool call after interception."""

    ALLOW = "allow"
    DENY = "deny"
    MOCK = "mock"  # return a synthetic result without executing


class Interceptor:
    """Abstract base for tool-call interceptors.

    Subclass and override ``intercept`` to inspect / mutate / block tool calls
    before the runner executes them.

    The interceptor chain is evaluated in order.  The first interceptor that
    returns anything other than ``ALLOW`` short-circuits the chain.

    Example — rate-limiting interceptor::

        import time

        class RateLimitInterceptor(Interceptor):
            def __init__(self, max_calls: int, window: float = 60.0):
                self.max_calls = max_calls
                self.window = window
                self._timestamps: list[float] = []

            def intercept(self, tool_name, tool_input):
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < self.window]
                if len(self._timestamps) >= self.max_calls:
                    return InterceptAction.DENY, None
                self._timestamps.append(now)
                return InterceptAction.ALLOW, tool_input

    Example — PII redaction interceptor::

        class RedactPIIInterceptor(Interceptor):
            def intercept(self, tool_name, tool_input):
                sanitized = {k: redact(v) for k, v in tool_input.items()}
                return InterceptAction.ALLOW, sanitized
    """

    def intercept(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[InterceptAction, Any]:
        """Inspect a pending tool call.

        Returns
        -------
        action : InterceptAction
            Whether to allow, deny, or mock the call.
        data : Any
            - For ALLOW: the (possibly modified) tool_input.
            - For DENY:  ignored (may be ``None``).
            - For MOCK:  the synthetic result to return instead of executing.
        """
        return InterceptAction.ALLOW, tool_input
