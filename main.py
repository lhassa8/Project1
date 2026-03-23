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

    # Bridge an MCP server's tools into the runner
    python main.py --mcp "npx -y @modelcontextprotocol/server-filesystem /tmp" "List files"

    # Start the approval review server
    python main.py --serve-approvals

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
from typing import Any

from agent_runner.runner import AgentRunner
from agent_runner.tools.registry import ToolRegistry
from agent_runner.tools.builtins import register_builtins
from agent_runner.interceptors.logging import LoggingInterceptor
from agent_runner.interceptors.approval import ApprovalInterceptor
from agent_runner.interceptors.shadow import ShadowInterceptor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Native Claude tool-call runner")
    p.add_argument("prompt", nargs="?", help="User prompt (omit for interactive mode)")
    p.add_argument("--model", default=None, help="Claude model to use")
    p.add_argument("--max-turns", type=int, default=25, help="Max tool-call round trips")
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

    if args.config:
        cfg = AgentConfig.from_file(args.config)
    else:
        cfg = AgentConfig.discover() or AgentConfig()
    cfg.merge_cli(args)

    # --- Replay approved runs ---
    if args.replay:
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
        return

    # --- Serve approval UI ---
    if args.serve_approvals:
        from agent_runner.sharing.server import create_approval_app
        from agent_runner.sharing.run_store import RunStore

        store = RunStore()
        server = create_approval_app(store, port=args.port)
        print(f"Approval server running at http://localhost:{args.port}")
        print("Review pending runs at http://localhost:{}/runs".format(args.port))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
        return

    # --- Tools ---
    registry = ToolRegistry()
    register_builtins(registry)

    # --- MCP bridge ---
    mcp_client = None
    mcp_bridge = None
    if cfg.mcp and cfg.mcp.command:
        from agent_runner.mcp.client import MCPClient
        from agent_runner.mcp.bridge import MCPToolBridge

        mcp_command = cfg.mcp.command.split()
        mcp_client = MCPClient(command=mcp_command)
        mcp_client.start()

        mcp_write_names = set(cfg.mcp.write_tools)

        mcp_bridge = MCPToolBridge(
            client=mcp_client,
            registry=registry,
            prefix="mcp_",
            write_tool_names=mcp_write_names,
        )
        bridged = mcp_bridge.bridge_tools()
        print(f"Bridged {len(bridged)} MCP tools: {bridged}")

    # --- Interceptors ---
    interceptors = []

    log_interceptor = LoggingInterceptor()
    interceptors.append(log_interceptor)

    shadow_interceptor = None
    if cfg.shadow:
        write_tools = {"write_file", "shell"}
        read_tools = {"read_file", "calculator"}
        if mcp_bridge:
            write_tools |= mcp_bridge.get_write_tools()
            read_tools |= mcp_bridge.get_read_tools()
        shadow_interceptor = ShadowInterceptor(
            write_tools=write_tools, read_tools=read_tools
        )
        interceptors.append(shadow_interceptor)

    if cfg.approve:
        interceptors.append(ApprovalInterceptor(require_approval_for=set(cfg.approve)))

    # --- Runner ---
    on_text = None
    if cfg.stream:
        on_text = lambda chunk: print(chunk, end="", flush=True)

    runner_kwargs: dict[str, Any] = {
        "system_prompt": cfg.system_prompt,
        "tools": registry,
        "interceptors": interceptors,
        "on_text": on_text,
    }
    if cfg.model:
        runner_kwargs["model"] = cfg.model
    runner_kwargs["max_turns"] = cfg.max_turns

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
                print(f"  {call['tool']}  [{call['action']}]")

        if shadow_interceptor and shadow_interceptor.captured_writes:
            print(f"\n--- Shadow Captures ({len(shadow_interceptor.captured_writes)}) ---")
            for cap in shadow_interceptor.captured_writes:
                print(f"  {cap['tool']}: {json.dumps(cap['input'], default=str)[:120]}")

            if cfg.share:
                from agent_runner.sharing.run_store import RunStore

                store = RunStore()
                record = store.save(
                    prompt=prompt,
                    final_text=result.final_text,
                    tool_call_log=result.tool_call_log,
                    captured_writes=shadow_interceptor.captured_writes,
                )
                print(f"\n  Share this link for approval:")
                print(f"  http://localhost:{args.port}/review/{record.id}")
                print(f"  (Start server with: python main.py --serve-approvals)")
            else:
                answer = input("\nReplay captured writes? [y/N] ").strip().lower()
                if answer in ("y", "yes"):
                    results = shadow_interceptor.replay(registry)
                    for r in results:
                        print(f"  Replayed {r['tool']}: {r['output']}")

    try:
        if args.prompt:
            # One-shot mode
            print(f"\n{'='*60}")
            print(f"Prompt: {args.prompt}")
            print("=" * 60)
            result = runner.run(args.prompt)
            _handle_result(args.prompt, result)
        else:
            # Multi-turn interactive mode — conversation history persists
            print("Interactive mode (multi-turn). Type your prompt (Ctrl-D to exit).")
            conversation: list[dict[str, Any]] = []
            try:
                while True:
                    line = input("\n> ").strip()
                    if not line:
                        continue
                    print(f"\n{'='*60}")
                    result = runner.run(line, conversation=conversation)
                    conversation = result.messages  # carry forward full history
                    _handle_result(line, result)
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
    finally:
        if mcp_client:
            mcp_client.stop()


if __name__ == "__main__":
    main()
