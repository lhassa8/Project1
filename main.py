#!/usr/bin/env python3
"""Entry point — run the agent with built-in tools and optional interceptors.

Usage::

    # Basic one-shot
    python main.py "What is 2^10?"

    # Streaming mode (tokens printed as they arrive)
    python main.py --stream "Explain quicksort"

    # Shadow mode (captures writes, doesn't execute them)
    python main.py --shadow "Create a file called hello.txt with a greeting"

    # Shadow mode + shareable approval link
    python main.py --shadow --share "Deploy config changes to staging"

    # With approval gate on dangerous tools
    python main.py --approve shell,write_file "List files in /tmp"

    # With sandbox (restrict to project directory)
    python main.py --sandbox-roots ./src,/tmp "Read the config file"

    # With cost budget
    python main.py --max-cost 2.00 "Refactor all files"

    # With audit trail and log rotation
    python main.py --audit "Do something important"

    # Bridge an MCP server's tools (with circuit breaker protection)
    python main.py --mcp "npx -y @modelcontextprotocol/server-filesystem /tmp" "List files"

    # Start the approval review server (with auth + health checks)
    python main.py --serve-approvals --auth-token mytoken123

    # Replay an approved run's captured writes
    python main.py --replay abc123def456

    # Use a config file instead of CLI flags
    python main.py --config agent.json "Do something"

    # Interactive multi-turn conversation
    python main.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from agent_runner.runner import AgentRunner
from agent_runner.tools.registry import ToolRegistry
from agent_runner.tools.builtins import register_builtins
from agent_runner.interceptors.logging import LoggingInterceptor
from agent_runner.interceptors.approval import ApprovalInterceptor
from agent_runner.interceptors.shadow import ShadowInterceptor
from agent_runner.sandbox import PathSandbox, ShellPolicy, ResourceLimits
from agent_runner.events import EventStream

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Native Claude tool-call runner")
    p.add_argument("prompt", nargs="?", help="User prompt (omit for interactive mode)")
    p.add_argument("--model", default=None, help="Claude model to use")
    p.add_argument("--max-turns", type=int, default=25, help="Max tool-call round trips")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max response tokens per API call")
    p.add_argument(
        "--shadow",
        action="store_true",
        help="Shadow mode: capture writes without executing them",
    )
    p.add_argument(
        "--share",
        action="store_true",
        help="Save shadow run for async stakeholder approval (requires --shadow)",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens to stdout as they arrive",
    )
    p.add_argument(
        "--approve",
        type=str,
        default=None,
        help="Comma-separated tool names requiring human approval",
    )
    p.add_argument(
        "--sandbox-roots",
        type=str,
        default=None,
        help="Comma-separated allowed directories for file operations (default: cwd)",
    )
    p.add_argument(
        "--shell-allow",
        type=str,
        default=None,
        help="Comma-separated allowed shell commands (allowlist mode)",
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=0.0,
        help="Max cost budget in USD (0 = unlimited)",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Enable audit trail logging to .agent_audit.jsonl",
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite database path for persistent storage (default: .agent_runs.db)",
    )
    p.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="Webhook URL for async approval notifications",
    )
    p.add_argument(
        "--webhook-secret",
        type=str,
        default=None,
        help="HMAC secret for signing webhook payloads",
    )
    p.add_argument(
        "--mcp",
        type=str,
        default=None,
        help="MCP server command to bridge tools from (e.g. 'npx -y @mcp/server /tmp')",
    )
    p.add_argument(
        "--mcp-writes",
        type=str,
        default=None,
        help="Comma-separated MCP tool names to treat as writes in shadow mode",
    )
    p.add_argument(
        "--serve-approvals",
        action="store_true",
        help="Start the approval review web server",
    )
    p.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="Bearer token for API authentication on approval server",
    )
    p.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Replay captured writes from an approved run ID",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (JSON or YAML). Auto-discovers agent.json/agent.yaml if present.",
    )
    p.add_argument("--port", type=int, default=8811, help="Port for approval server")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- Load config (file + CLI merge) ---
    from agent_runner.config import AgentConfig
    from agent_runner.config_schema import validate_config

    if args.config:
        cfg = AgentConfig.from_file(args.config)
    else:
        cfg = AgentConfig.discover() or AgentConfig()
    cfg.merge_cli(args)

    # Merge new CLI args into config
    if getattr(args, "sandbox_roots", None):
        cfg.sandbox_roots = [r.strip() for r in args.sandbox_roots.split(",")]
    if getattr(args, "shell_allow", None):
        cfg.shell_allow = [c.strip() for c in args.shell_allow.split(",")]
    if getattr(args, "max_cost", 0.0) > 0:
        cfg.max_cost_usd = args.max_cost
    if getattr(args, "audit", False):
        cfg.audit = True

    # --- Graceful shutdown ---
    from agent_runner.graceful import ShutdownManager

    shutdown = ShutdownManager()
    shutdown.install_signal_handlers()

    # --- Metrics ---
    from agent_runner.metrics import MetricsRegistry, MetricsCollector

    metrics = MetricsRegistry()
    metrics_collector = MetricsCollector(metrics)

    # --- Replay approved runs ---
    if args.replay:
        _replay_run(args)
        return

    # --- Serve approval UI ---
    if args.serve_approvals:
        _serve_approvals(args, metrics, shutdown)
        return

    # --- Sandbox ---
    path_sandbox = None
    if cfg.sandbox_roots:
        path_sandbox = PathSandbox(
            allowed_roots=cfg.sandbox_roots,
            denied_patterns=cfg.sandbox_deny,
        )
        logger.info("Path sandbox: roots=%s, deny=%s", cfg.sandbox_roots, cfg.sandbox_deny)

    shell_policy = None
    if cfg.shell_allow or cfg.shell_deny:
        shell_policy = ShellPolicy(
            allow=cfg.shell_allow or None,
            deny=cfg.shell_deny or None,
        )
        logger.info("Shell policy: allow=%s, deny=%s", cfg.shell_allow, cfg.shell_deny)

    # --- Tools ---
    registry = ToolRegistry()
    register_builtins(registry, path_sandbox=path_sandbox, shell_policy=shell_policy)

    # --- MCP bridge (with circuit breaker) ---
    mcp_client = None
    mcp_bridge = None
    if cfg.mcp and cfg.mcp.command:
        from agent_runner.mcp.resilient_client import ResilientMCPClient
        from agent_runner.mcp.bridge import MCPToolBridge

        mcp_command = cfg.mcp.command.split()
        mcp_client = ResilientMCPClient(command=mcp_command)
        mcp_client.start()

        mcp_write_names = set(cfg.mcp.write_tools)

        mcp_bridge = MCPToolBridge(
            client=mcp_client,
            registry=registry,
            prefix="mcp_",
            write_tool_names=mcp_write_names,
        )
        bridged = mcp_bridge.bridge_tools()
        logger.info("Bridged %d MCP tools: %s", len(bridged), bridged)

        # Validate write tool names exist
        for name in mcp_write_names:
            prefixed = f"mcp_{name}"
            if not registry.has(prefixed):
                logger.warning(
                    "MCP write tool '%s' was not found in bridged tools. "
                    "Check mcp.write_tools config.", name,
                )

    # --- Interceptors ---
    interceptors = []

    log_interceptor = LoggingInterceptor()
    interceptors.append(log_interceptor)

    shadow_interceptor = None
    if cfg.shadow:
        write_tools = registry.write_tools()
        read_tools = registry.read_tools()
        if mcp_bridge:
            write_tools |= mcp_bridge.get_write_tools()
            read_tools |= mcp_bridge.get_read_tools()
        shadow_interceptor = ShadowInterceptor(
            write_tools=write_tools, read_tools=read_tools
        )
        interceptors.append(shadow_interceptor)

    # Approval: use policy if configured, otherwise simple tool list
    if cfg.approval_policy:
        from agent_runner.approval_policy import ApprovalPolicy

        policy = ApprovalPolicy.from_config(cfg.approval_policy)

        def _policy_approval(tool_name: str, tool_input: dict) -> bool:
            defn = registry.get_def(tool_name)
            is_write = defn.is_write if defn else False
            decision = policy.evaluate(tool_name, tool_input, is_write=is_write)
            if decision == "auto_approve":
                return True
            if decision == "auto_deny":
                return False
            print(f"\n--- Approval Required (policy: require_review) ---")
            print(f"Tool:  {tool_name}")
            print(f"Input: {json.dumps(tool_input, indent=2, default=str)[:500]}")
            answer = input("Allow? [y/N] ").strip().lower()
            return answer in ("y", "yes")

        interceptors.append(ApprovalInterceptor(on_approval=_policy_approval))
        logger.info("Using approval policy with %d rules", len(policy.rules))
    elif cfg.approve:
        interceptors.append(ApprovalInterceptor(require_approval_for=set(cfg.approve)))

    # --- Resource limits ---
    resource_limits = ResourceLimits(
        max_tool_calls=cfg.max_tool_calls,
        max_cost_usd=cfg.max_cost_usd,
    )

    # --- Event stream + metrics bridge ---
    event_stream = EventStream()
    metrics_collector.connect(event_stream)

    # --- Audit trail (with rotation) ---
    audit_log = None
    if cfg.audit:
        from agent_runner.sharing.audit import AuditEntry
        from agent_runner.sharing.audit_rotation import RotatingAuditLog

        audit_log = RotatingAuditLog(
            path=cfg.audit_path,
            max_bytes=10_000_000,
            backup_count=5,
            compress=True,
        )
        logger.info("Audit trail enabled: %s (rotating at 10MB)", cfg.audit_path)

        def _audit_handler(event):
            audit_log.record(AuditEntry(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                event=event.event_type,
                run_id=event.run_id,
                tool=event.data.get("tool"),
                reviewer=None,
                details=event.data,
            ))

        event_stream.on_event(_audit_handler)

    # --- Webhook notifications ---
    webhook_notifier = None
    if getattr(args, "webhook_url", None):
        from agent_runner.sharing.webhooks import WebhookConfig, WebhookNotifier

        wh_config = WebhookConfig(
            url=args.webhook_url,
            secret=args.webhook_secret or "",
        )
        webhook_notifier = WebhookNotifier(configs=[wh_config])
        logger.info("Webhook notifications enabled: %s", args.webhook_url)

    # --- Database (SQLite persistence) ---
    db_store = None
    db_path = getattr(args, "db", None)
    if db_path or cfg.share:
        from agent_runner.sharing.database import Database, SQLiteRunStore

        db = Database(db_path or ".agent_runs.db")
        db_store = SQLiteRunStore(db)
        shutdown.register(db.close)
        logger.info("SQLite persistence: %s", db_path or ".agent_runs.db")

    # --- Runner ---
    on_text = None
    if cfg.stream:
        on_text = lambda chunk: print(chunk, end="", flush=True)

    runner_kwargs: dict[str, Any] = {
        "system_prompt": cfg.system_prompt,
        "tools": registry,
        "interceptors": interceptors,
        "on_text": on_text,
        "max_turns": cfg.max_turns,
        "max_tokens": cfg.max_tokens,
        "resource_limits": resource_limits,
        "event_stream": event_stream,
    }
    if cfg.model:
        runner_kwargs["model"] = cfg.model

    runner = AgentRunner(**runner_kwargs)

    # --- Execute ---
    def _handle_result(prompt: str, result: Any) -> None:
        """Display a run result and handle shadow captures."""
        if cfg.stream:
            print()  # newline after streamed output

        print(f"\n--- Response ({result.turns_used} turns, "
              f"{result.usage.total_tokens} tokens, "
              f"{result.elapsed_seconds:.1f}s, "
              f"~${result.usage.estimated_cost(runner.model):.4f}) ---")
        if not cfg.stream:
            print(result.final_text)

        if result.tool_call_log:
            print(f"\n--- Tool Calls ({len(result.tool_call_log)}) ---")
            for call in result.tool_call_log:
                status = f"  {call['tool']}  [{call['action']}]"
                if call.get("error"):
                    status += f"  ERROR: {call['error'][:80]}"
                print(status)

        if shadow_interceptor and shadow_interceptor.captured_writes:
            from agent_runner.shadow.diff import ShadowDiff

            diff = ShadowDiff(shadow_interceptor.state)
            print(f"\n{diff.summary()}")

            if cfg.share:
                store = db_store
                if store is None:
                    from agent_runner.sharing.run_store import RunStore
                    store = RunStore()

                record = store.save(
                    prompt=prompt,
                    final_text=result.final_text,
                    tool_call_log=result.tool_call_log,
                    captured_writes=shadow_interceptor.captured_writes,
                )

                # Send webhook notification if configured
                if webhook_notifier:
                    try:
                        webhook_notifier.notify_approval_needed(record)
                        print("  Webhook notification sent.")
                    except Exception as exc:
                        logger.warning("Webhook notification failed: %s", exc)

                print(f"\n  Share this link for approval:")
                print(f"  http://localhost:{args.port}/review/{record.id}")
                print(f"  (Start server with: python main.py --serve-approvals)")
            else:
                answer = input("\nReplay captured writes? [y/N/diff] ").strip().lower()
                if answer == "diff":
                    print(diff.full_diff())
                    answer = input("Replay? [y/N] ").strip().lower()
                if answer in ("y", "yes"):
                    from agent_runner.shadow.replay import TransactionReplay

                    replay_result = TransactionReplay(shadow_interceptor.state).execute()
                    if replay_result.success:
                        print(f"  Applied {len(replay_result.completed)} changes.")
                    else:
                        print(f"  FAILED: {replay_result.failed}")
                        if replay_result.rolled_back:
                            print("  All changes rolled back.")

    try:
        if args.prompt:
            # One-shot mode
            with shutdown.operation("agent_run"):
                print(f"\n{'='*60}")
                print(f"Prompt: {args.prompt}")
                print("=" * 60)
                result = runner.run(args.prompt)
                _handle_result(args.prompt, result)
        else:
            # Multi-turn interactive mode
            print("Interactive mode (multi-turn). Type your prompt (Ctrl-D to exit).")
            conversation: list[dict[str, Any]] = []
            try:
                while not shutdown.is_shutting_down:
                    line = input("\n> ").strip()
                    if not line:
                        continue
                    print(f"\n{'='*60}")
                    with shutdown.operation("agent_run"):
                        result = runner.run(line, conversation=conversation)
                        conversation = result.messages
                        _handle_result(line, result)
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
    finally:
        if mcp_client:
            mcp_client.stop()

        # Print metrics summary
        summary = event_stream.summary()
        if summary.get("tool_calls", 0) > 0:
            stats = event_stream.tool_stats()
            print(f"\n--- Session Metrics ---")
            for name, s in stats.items():
                print(f"  {name}: {s['calls']} calls, {s['total_ms']:.0f}ms"
                      + (f", {s['errors']} errors" if s['errors'] else ""))
            if summary.get("cost_usd", 0) > 0:
                print(f"  Total cost: ${summary['cost_usd']:.4f}")

        shutdown.shutdown()


def _replay_run(args: argparse.Namespace) -> None:
    """Replay captured writes from an approved run."""
    from agent_runner.sharing.run_store import RunStore, RunStatus

    store = RunStore()
    record = store.get(args.replay)
    if record is None:
        print(f"Error: run '{args.replay}' not found.")
        sys.exit(1)
    if record.status not in (RunStatus.APPROVED, RunStatus.REPLAYED):
        print(f"Error: run '{args.replay}' has status '{record.status.value}' (must be approved).")
        sys.exit(1)

    registry = ToolRegistry()
    register_builtins(registry)

    print(f"Replaying run {record.id} ({len(record.captured_writes)} captured writes)...")
    for i, cap in enumerate(record.captured_writes):
        handler = registry.get(cap["tool"])
        if handler is None:
            print(f"  [{i+1}] {cap['tool']}: SKIPPED (unknown tool)")
            continue
        try:
            output = handler(cap["input"])
            print(f"  [{i+1}] {cap['tool']}: {str(output)[:200]}")
        except Exception as exc:
            print(f"  [{i+1}] {cap['tool']}: ERROR — {exc}")

    store.mark_replayed(record.id)
    print(f"\nDone. Run {record.id} marked as replayed.")


def _serve_approvals(
    args: argparse.Namespace,
    metrics: Any,
    shutdown: Any,
) -> None:
    """Start the approval review server with health checks and auth."""
    from agent_runner.sharing.server import create_approval_app
    from agent_runner.sharing.run_store import RunStore
    from agent_runner.health import HealthCheck, HealthStatus

    store = RunStore()

    # Health check
    health = HealthCheck()
    health.register("run_store", lambda: (HealthStatus.HEALTHY, "OK"), critical=True)

    server = create_approval_app(
        store,
        port=args.port,
        auth_token=getattr(args, "auth_token", None),
    )

    print(f"Approval server running at http://localhost:{args.port}")
    print(f"  Review runs: http://localhost:{args.port}/runs")
    print(f"  Health:      http://localhost:{args.port}/health")
    if getattr(args, "auth_token", None):
        print(f"  Auth:        Bearer token required for API endpoints")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        shutdown.shutdown()


if __name__ == "__main__":
    main()
