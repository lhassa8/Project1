#!/usr/bin/env python3
"""Entry point — run the agent with built-in tools and optional interceptors.

Usage::

    # Basic one-shot
    python main.py "What is 2^10?"

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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

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
    p.add_argument("--port", type=int, default=8811, help="Port for approval server")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

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
    if args.mcp:
        from agent_runner.mcp.client import MCPClient
        from agent_runner.mcp.bridge import MCPToolBridge

        mcp_command = args.mcp.split()
        mcp_client = MCPClient(command=mcp_command)
        mcp_client.start()

        mcp_write_names = set()
        if args.mcp_writes:
            mcp_write_names = {t.strip() for t in args.mcp_writes.split(",")}

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

    # Always log
    log_interceptor = LoggingInterceptor()
    interceptors.append(log_interceptor)

    shadow_interceptor = None
    if args.shadow:
        write_tools = {"write_file", "shell"}
        read_tools = {"read_file", "calculator"}
        if mcp_bridge:
            write_tools |= mcp_bridge.get_write_tools()
            read_tools |= mcp_bridge.get_read_tools()
        shadow_interceptor = ShadowInterceptor(
            write_tools=write_tools, read_tools=read_tools
        )
        interceptors.append(shadow_interceptor)

    if args.approve:
        tool_names = {t.strip() for t in args.approve.split(",")}
        interceptors.append(ApprovalInterceptor(require_approval_for=tool_names))

    # --- System prompt ---
    system_prompt = (
        "You are a helpful assistant with access to tools. "
        "Use the available tools to accomplish the user's request. "
        "Think step-by-step and use tools as needed."
    )

    # --- Runner ---
    runner_kwargs = {"system_prompt": system_prompt, "tools": registry, "interceptors": interceptors}
    if args.model:
        runner_kwargs["model"] = args.model
    if args.max_turns:
        runner_kwargs["max_turns"] = args.max_turns

    runner = AgentRunner(**runner_kwargs)

    # --- Execute ---
    if args.prompt:
        prompts = [args.prompt]
    else:
        print("Interactive mode. Type your prompt (Ctrl-D to exit).")
        prompts = []
        try:
            while True:
                line = input("\n> ")
                if line.strip():
                    prompts.append(line.strip())
        except (EOFError, KeyboardInterrupt):
            pass

    try:
        for prompt in prompts:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt}")
            print("=" * 60)

            result = runner.run(prompt)

            print(f"\n--- Response ({result.turns_used} turns) ---")
            print(result.final_text)

            if result.tool_call_log:
                print(f"\n--- Tool Calls ({len(result.tool_call_log)}) ---")
                for call in result.tool_call_log:
                    print(f"  {call['tool']}  [{call['action']}]")

            if shadow_interceptor and shadow_interceptor.captured_writes:
                print(f"\n--- Shadow Captures ({len(shadow_interceptor.captured_writes)}) ---")
                for cap in shadow_interceptor.captured_writes:
                    print(f"  {cap['tool']}: {json.dumps(cap['input'], default=str)[:120]}")

                # --- Shareable run link ---
                if args.share:
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
    finally:
        if mcp_client:
            mcp_client.stop()


if __name__ == "__main__":
    main()
