"""JSON Schema validation for agent configuration files.

Provides a complete JSON Schema definition (``CONFIG_SCHEMA``) for
``agent.json`` and a lightweight validator that uses only the stdlib —
no ``jsonschema`` dependency required.

Usage::

    from agent_runner.config_schema import validate_config, ConfigError

    errors = validate_config(data)
    if errors:
        raise ConfigError("\\n".join(errors))
"""

from __future__ import annotations

from typing import Any


class ConfigError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        msg = f"Configuration validation failed with {len(errors)} error(s):\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


# ------------------------------------------------------------------
# JSON Schema — covers every field from AgentConfig + MCPConfig
# ------------------------------------------------------------------

CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "description": "Claude model identifier.",
        },
        "max_turns": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum tool-call round trips.",
        },
        "max_tokens": {
            "type": "integer",
            "minimum": 1,
            "description": "Max response tokens per API call.",
        },
        "system_prompt": {
            "type": "string",
            "description": "System prompt sent to the model.",
        },
        "shadow": {
            "type": "boolean",
            "description": "Enable shadow mode (capture writes instead of executing).",
        },
        "share": {
            "type": "boolean",
            "description": "Enable run sharing.",
        },
        "stream": {
            "type": "boolean",
            "description": "Enable streaming output.",
        },
        "approve": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tool names requiring explicit approval.",
        },
        "mcp": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to launch the MCP server.",
                },
                "write_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "MCP tools considered write operations.",
                },
            },
            "additionalProperties": False,
        },
        "tools": {
            "type": "object",
            "additionalProperties": {"type": "boolean"},
            "description": "Map of tool name -> enabled flag.",
        },
        "sandbox_roots": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Allowed filesystem roots for sandboxing.",
        },
        "sandbox_deny": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Glob patterns for files denied by sandbox.",
        },
        "shell_allow": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Allowed shell commands.",
        },
        "shell_deny": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Denied shell commands.",
        },
        "max_tool_calls": {
            "type": "integer",
            "minimum": 0,
            "description": "Max total tool calls per run (0 = unlimited).",
        },
        "max_cost_usd": {
            "type": "number",
            "minimum": 0,
            "description": "Max API cost in USD per run (0 = unlimited).",
        },
        "audit": {
            "type": "boolean",
            "description": "Enable audit logging.",
        },
        "audit_path": {
            "type": "string",
            "description": "Path for audit log file.",
        },
        "approval_policy": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["allow", "deny", "ask"],
                    },
                    "condition": {"type": "object"},
                },
                "required": ["tool", "action"],
            },
            "description": "Fine-grained approval rules.",
        },
    },
    "additionalProperties": False,
}

# Valid actions for approval_policy rules
_VALID_APPROVAL_ACTIONS = {"allow", "deny", "ask"}

# Python type name mapping for pretty messages
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}

_TYPE_NAMES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


# ------------------------------------------------------------------
# Minimal validator (stdlib only)
# ------------------------------------------------------------------

def _pretty_type(value: Any) -> str:
    """Human-readable type name for a value."""
    return _TYPE_NAMES.get(type(value), type(value).__name__)


def _validate_type(value: Any, expected_type: str, path: str, errors: list[str]) -> bool:
    """Check that *value* matches the JSON Schema type. Returns True if valid."""
    py_type = _JSON_TYPE_MAP.get(expected_type)
    if py_type is None:
        return True  # Unknown type — skip

    # Special case: JSON Schema "integer" should reject bools (which are int subclass in Python)
    if expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"config.{path}: expected integer, got {_pretty_type(value)}")
            return False
        return True

    # Special case: "number" accepts int and float but not bool
    if expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"config.{path}: expected number, got {_pretty_type(value)}")
            return False
        return True

    # Special case: "boolean" must be actual bool
    if expected_type == "boolean":
        if not isinstance(value, bool):
            errors.append(f"config.{path}: expected boolean, got {_pretty_type(value)}")
            return False
        return True

    if not isinstance(value, py_type):
        errors.append(f"config.{path}: expected {expected_type}, got {_pretty_type(value)}")
        return False
    return True


def _validate_value(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    """Recursively validate *value* against a JSON Schema node."""
    expected_type = schema.get("type")

    if expected_type:
        if not _validate_type(value, expected_type, path, errors):
            return  # Type mismatch — no point checking further

    # Minimum / maximum (numeric)
    if "minimum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema["minimum"]:
            errors.append(
                f"config.{path}: value {value} is below minimum {schema['minimum']}"
            )

    if "maximum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > schema["maximum"]:
            errors.append(
                f"config.{path}: value {value} is above maximum {schema['maximum']}"
            )

    # Enum
    if "enum" in schema:
        if value not in schema["enum"]:
            allowed = ", ".join(repr(v) for v in schema["enum"])
            errors.append(
                f"config.{path}: invalid value {value!r}, must be one of: {allowed}"
            )

    # Array items
    if expected_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                _validate_value(item, item_schema, f"{path}[{i}]", errors)

    # Object properties
    if expected_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties")

        # Required fields
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"config.{path}.{req}: required field is missing")

        for key, val in value.items():
            child_path = f"{path}.{key}" if path else key

            if key in properties:
                _validate_value(val, properties[key], child_path, errors)
            elif additional is not None:
                if additional is False:
                    errors.append(f"config.{child_path}: unknown field")
                elif isinstance(additional, dict):
                    _validate_value(val, additional, child_path, errors)


def _validate_approval_policy(policy: list[Any], errors: list[str]) -> None:
    """Extra semantic validation for approval_policy entries."""
    if not isinstance(policy, list):
        return

    for i, rule in enumerate(policy):
        if not isinstance(rule, dict):
            continue
        path = f"approval_policy[{i}]"

        # Validate action enum
        action = rule.get("action")
        if action is not None and action not in _VALID_APPROVAL_ACTIONS:
            allowed = ", ".join(repr(a) for a in sorted(_VALID_APPROVAL_ACTIONS))
            errors.append(
                f"config.{path}.action: invalid value {action!r}, must be one of: {allowed}"
            )

        # Validate tool is a string
        tool = rule.get("tool")
        if tool is not None and not isinstance(tool, str):
            errors.append(
                f"config.{path}.tool: expected string, got {_pretty_type(tool)}"
            )


def _validate_approve_list(approve: Any, errors: list[str]) -> None:
    """Validate that approve list items are all strings."""
    if not isinstance(approve, list):
        return
    for i, item in enumerate(approve):
        if not isinstance(item, str):
            errors.append(
                f"config.approve[{i}]: expected string, got {_pretty_type(item)}"
            )


def validate_config(data: dict[str, Any]) -> list[str]:
    """Validate a configuration dictionary against ``CONFIG_SCHEMA``.

    Returns a list of human-readable error messages.  An empty list means
    the configuration is valid.

    Parameters
    ----------
    data : dict
        The parsed configuration data (e.g. from ``json.loads``).

    Returns
    -------
    list[str]
        Validation errors.  Empty means valid.
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append("config: expected object at top level, got " + _pretty_type(data))
        return errors

    _validate_value(data, CONFIG_SCHEMA, "", errors)

    # Extra semantic checks
    if "approval_policy" in data:
        _validate_approval_policy(data["approval_policy"], errors)

    if "approve" in data:
        _validate_approve_list(data["approve"], errors)

    return errors
