"""Tests for agent_runner.config_schema."""

from __future__ import annotations

import pytest

from agent_runner.config_schema import CONFIG_SCHEMA, ConfigError, validate_config


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _valid_config() -> dict:
    """Return a minimal valid configuration."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_turns": 25,
        "max_tokens": 4096,
        "system_prompt": "You are a helpful assistant.",
        "shadow": False,
        "share": False,
        "stream": False,
        "approve": ["shell", "write_file"],
        "tools": {"shell": True, "read_file": True},
        "sandbox_roots": ["/tmp"],
        "sandbox_deny": ["*.key"],
        "shell_allow": ["ls"],
        "shell_deny": ["sudo"],
        "max_tool_calls": 100,
        "max_cost_usd": 5.0,
        "audit": True,
        "audit_path": ".agent_audit.jsonl",
        "approval_policy": [
            {"tool": "shell", "action": "ask"},
            {"tool": "write_file", "action": "allow"},
        ],
        "mcp": {
            "command": "npx -y @modelcontextprotocol/server-filesystem /tmp",
            "write_tools": ["write_file"],
        },
    }


# ------------------------------------------------------------------
# Valid config
# ------------------------------------------------------------------

class TestValidConfig:
    def test_full_valid_config(self) -> None:
        errors = validate_config(_valid_config())
        assert errors == []

    def test_minimal_config(self) -> None:
        errors = validate_config({})
        assert errors == []

    def test_only_model(self) -> None:
        errors = validate_config({"model": "claude-sonnet-4-20250514"})
        assert errors == []

    def test_max_cost_usd_accepts_int(self) -> None:
        """Number fields should accept both int and float."""
        errors = validate_config({"max_cost_usd": 10})
        assert errors == []


# ------------------------------------------------------------------
# Invalid types
# ------------------------------------------------------------------

class TestInvalidTypes:
    def test_model_not_string(self) -> None:
        errors = validate_config({"model": 123})
        assert len(errors) == 1
        assert "config.model" in errors[0]
        assert "expected string" in errors[0]
        assert "got integer" in errors[0]

    def test_max_turns_not_integer(self) -> None:
        errors = validate_config({"max_turns": "ten"})
        assert len(errors) == 1
        assert "config.max_turns" in errors[0]
        assert "expected integer" in errors[0]
        assert "got string" in errors[0]

    def test_shadow_not_boolean(self) -> None:
        errors = validate_config({"shadow": "yes"})
        assert len(errors) == 1
        assert "config.shadow" in errors[0]
        assert "expected boolean" in errors[0]

    def test_approve_not_array(self) -> None:
        errors = validate_config({"approve": "shell"})
        assert len(errors) == 1
        assert "config.approve" in errors[0]
        assert "expected array" in errors[0]

    def test_tools_not_object(self) -> None:
        errors = validate_config({"tools": ["shell"]})
        assert len(errors) == 1
        assert "config.tools" in errors[0]
        assert "expected object" in errors[0]

    def test_boolean_not_accepted_as_integer(self) -> None:
        """Python bool is a subclass of int, but schema should reject it."""
        errors = validate_config({"max_turns": True})
        assert len(errors) >= 1
        assert any("expected integer" in e for e in errors)

    def test_boolean_not_accepted_as_number(self) -> None:
        errors = validate_config({"max_cost_usd": True})
        assert len(errors) >= 1
        assert any("expected number" in e for e in errors)

    def test_tools_values_must_be_boolean(self) -> None:
        errors = validate_config({"tools": {"shell": "yes"}})
        assert len(errors) >= 1
        assert any("expected boolean" in e for e in errors)

    def test_non_dict_top_level(self) -> None:
        errors = validate_config("not a dict")  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "expected object at top level" in errors[0]


# ------------------------------------------------------------------
# Missing required fields
# ------------------------------------------------------------------

class TestMissingRequiredFields:
    def test_approval_policy_missing_tool(self) -> None:
        errors = validate_config({
            "approval_policy": [{"action": "allow"}],
        })
        assert any("tool" in e and "required" in e for e in errors)

    def test_approval_policy_missing_action(self) -> None:
        errors = validate_config({
            "approval_policy": [{"tool": "shell"}],
        })
        assert any("action" in e and "required" in e for e in errors)


# ------------------------------------------------------------------
# Invalid enum values
# ------------------------------------------------------------------

class TestInvalidEnumValues:
    def test_invalid_approval_action(self) -> None:
        errors = validate_config({
            "approval_policy": [{"tool": "shell", "action": "maybe"}],
        })
        assert len(errors) >= 1
        assert any("maybe" in e for e in errors)
        assert any("allow" in e or "deny" in e or "ask" in e for e in errors)

    def test_valid_approval_actions(self) -> None:
        for action in ("allow", "deny", "ask"):
            errors = validate_config({
                "approval_policy": [{"tool": "shell", "action": action}],
            })
            assert errors == [], f"Expected no errors for action={action!r}, got {errors}"


# ------------------------------------------------------------------
# Approval policy validation
# ------------------------------------------------------------------

class TestApprovalPolicyValidation:
    def test_tool_must_be_string(self) -> None:
        errors = validate_config({
            "approval_policy": [{"tool": 123, "action": "allow"}],
        })
        assert any("tool" in e and "string" in e for e in errors)

    def test_multiple_rules_validated(self) -> None:
        errors = validate_config({
            "approval_policy": [
                {"tool": "shell", "action": "allow"},
                {"tool": "write_file", "action": "invalid_action"},
            ],
        })
        assert any("invalid_action" in e for e in errors)

    def test_approve_list_items_must_be_strings(self) -> None:
        errors = validate_config({"approve": ["shell", 42, True]})
        assert any("approve[1]" in e for e in errors)
        assert any("approve[2]" in e for e in errors)


# ------------------------------------------------------------------
# Range validation
# ------------------------------------------------------------------

class TestRangeValidation:
    def test_max_turns_minimum(self) -> None:
        errors = validate_config({"max_turns": 0})
        assert any("below minimum" in e for e in errors)

    def test_max_tokens_minimum(self) -> None:
        errors = validate_config({"max_tokens": -1})
        assert any("below minimum" in e for e in errors)

    def test_max_cost_usd_minimum(self) -> None:
        errors = validate_config({"max_cost_usd": -0.5})
        assert any("below minimum" in e for e in errors)


# ------------------------------------------------------------------
# Unknown fields (additionalProperties)
# ------------------------------------------------------------------

class TestUnknownFields:
    def test_unknown_top_level_field(self) -> None:
        errors = validate_config({"unknown_field": "value"})
        assert any("unknown field" in e for e in errors)

    def test_unknown_mcp_field(self) -> None:
        errors = validate_config({"mcp": {"command": "foo", "unknown": "bar"}})
        assert any("unknown field" in e for e in errors)


# ------------------------------------------------------------------
# Pretty error messages
# ------------------------------------------------------------------

class TestPrettyErrorMessages:
    def test_error_includes_path(self) -> None:
        errors = validate_config({"max_turns": "not_a_number"})
        assert errors[0].startswith("config.max_turns:")

    def test_error_includes_expected_and_got(self) -> None:
        errors = validate_config({"model": 42})
        assert "expected string" in errors[0]
        assert "got integer" in errors[0]

    def test_nested_path(self) -> None:
        errors = validate_config({"mcp": {"command": 123}})
        assert "config.mcp.command" in errors[0]

    def test_array_index_in_path(self) -> None:
        errors = validate_config({"approve": [123]})
        assert "approve[0]" in errors[0]


# ------------------------------------------------------------------
# ConfigError exception
# ------------------------------------------------------------------

class TestConfigError:
    def test_stores_errors_list(self) -> None:
        err = ConfigError(["error 1", "error 2"])
        assert err.errors == ["error 1", "error 2"]
        assert "2 error(s)" in str(err)
        assert "error 1" in str(err)
        assert "error 2" in str(err)

    def test_is_exception(self) -> None:
        with pytest.raises(ConfigError):
            raise ConfigError(["bad config"])


# ------------------------------------------------------------------
# Schema completeness
# ------------------------------------------------------------------

class TestSchemaCompleteness:
    """Ensure CONFIG_SCHEMA covers all AgentConfig fields."""

    def test_schema_has_all_config_fields(self) -> None:
        from agent_runner.config import AgentConfig
        import dataclasses

        schema_props = set(CONFIG_SCHEMA["properties"].keys())
        config_fields = {f.name for f in dataclasses.fields(AgentConfig)}

        # Every config field should appear in the schema
        missing = config_fields - schema_props
        assert missing == set(), f"Schema missing fields: {missing}"
