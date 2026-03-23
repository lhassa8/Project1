#!/usr/bin/env python3
"""Entry point — run the agent with built-in tools and optional interceptors.

Usage::

    # Basic interactive mode
    python main.py "What is 2^10?"

    # Shadow mode (captures writes, doesn't execute them)
    python main.py --shadow "Create a file called hello.txt with a greeting"

    # With approval gate on dangerous tools
    python main.py --approve shell,write_file "List files in /tmp"
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
        "--approve",
        type=str,
        default=None,
        help="Comma-separated tool names requiring human approval",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- Tools ---
    registry = ToolRegistry()
    register_builtins(registry)

    # --- Interceptors ---
    interceptors = []

    # Always log
    log_interceptor = LoggingInterceptor()
    interceptors.append(log_interceptor)

    shadow_interceptor = None
    if args.shadow:
        shadow_interceptor = ShadowInterceptor()
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
            answer = input("\nReplay captured writes? [y/N] ").strip().lower()
            if answer in ("y", "yes"):
                results = shadow_interceptor.replay(registry)
                for r in results:
                    print(f"  Replayed {r['tool']}: {r['output']}")


if __name__ == "__main__":
    main()
