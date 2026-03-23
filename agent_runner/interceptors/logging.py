"""Interceptor that logs every tool call for audit / replay."""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_runner.interceptors.base import InterceptAction, Interceptor

logger = logging.getLogger(__name__)


class LoggingInterceptor(Interceptor):
    """Logs tool name and input, then allows execution to continue.

    Collected entries are available in ``self.log`` for later inspection.
    """

    def __init__(self) -> None:
        self.log: list[dict[str, Any]] = []

    def intercept(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> tuple[InterceptAction, Any]:
        entry = {"tool": tool_name, "input": tool_input}
        self.log.append(entry)
        logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_input, default=str))
        return InterceptAction.ALLOW, tool_input
