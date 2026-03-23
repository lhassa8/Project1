"""Interceptor that logs every tool call for audit / replay.

Supports PII-safe logging by redacting sensitive fields and truncating
large outputs to prevent credential leaks and log bloat.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from agent_runner.interceptors.base import InterceptAction, Interceptor

logger = logging.getLogger(__name__)

# Default fields to redact from logs
DEFAULT_REDACT_FIELDS = frozenset({
    "api_key", "apikey", "api_secret", "secret", "password", "passwd",
    "token", "access_token", "refresh_token", "private_key", "credentials",
    "authorization",
})


class LoggingInterceptor(Interceptor):
    """Logs tool name and input, then allows execution to continue.

    Parameters
    ----------
    redact_fields : set[str] | None
        Field names to redact from log output.  Matching is case-insensitive.
        Defaults to common secret field names (api_key, password, token, etc.).
        Pass an empty set to disable redaction.
    max_input_length : int
        Maximum characters of tool input to log.  0 = unlimited.
    """

    def __init__(
        self,
        redact_fields: set[str] | None = None,
        max_input_length: int = 2000,
    ) -> None:
        self.log: list[dict[str, Any]] = []
        self.redact_fields = (
            redact_fields if redact_fields is not None else DEFAULT_REDACT_FIELDS
        )
        self.max_input_length = max_input_length

    def intercept(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[InterceptAction, Any]:
        entry = {"tool": tool_name, "input": tool_input}
        self.log.append(entry)

        # Log a sanitized version
        safe_input = self._sanitize(tool_input)
        logger.info("Tool call: %s(%s)", tool_name, safe_input)
        return InterceptAction.ALLOW, tool_input

    def _sanitize(self, tool_input: dict[str, Any]) -> str:
        """Redact sensitive fields and truncate for logging."""
        if self.redact_fields:
            sanitized = self._redact(tool_input)
        else:
            sanitized = tool_input

        text = json.dumps(sanitized, default=str)
        if self.max_input_length and len(text) > self.max_input_length:
            text = text[:self.max_input_length] + "...(truncated)"
        return text

    def _redact(self, data: Any) -> Any:
        """Recursively redact sensitive fields."""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key.lower() in self.redact_fields:
                    result[key] = "[REDACTED]"
                else:
                    result[key] = self._redact(value)
            return result
        elif isinstance(data, list):
            return [self._redact(item) for item in data]
        return data
