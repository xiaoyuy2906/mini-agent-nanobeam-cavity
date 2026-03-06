#!/usr/bin/env python
"""main.py — Single-process Python TUI for the nanobeam cavity agent.

Replaces the Node.js chat.js + agent_server.py JSON bridge.
One process. No IPC. No serialization overhead.

Usage: uv run python main.py
"""

import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()

from core.agent import (
    CavityAgent,
    ThoughtEvent,
    ToolStartEvent,
    ToolEndEvent,
    TextEvent,
    DoneEvent,
    ErrorEvent,
)


# --- Minimal TUI (no heavy dependencies) ---

# ANSI color codes — works in any modern terminal
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"


def print_banner():
    print(f"{CYAN}{BOLD}")
    print("  ╔══════════════════════════════════════╗")
    print("  ║   Nanobeam Cavity Design Agent       ║")
    print("  ║   ReAct + SWE-agent architecture     ║")
    print("  ╚══════════════════════════════════════╝")
    print(f"{RESET}{DIM}  Type your request. 'quit' to exit.{RESET}")
    print()


def format_result_summary(name: str, result: dict) -> str:
    """One-line summary for tool_end display."""
    if not result.get("ok", True):
        return f"{RED}ERROR: {result.get('error', 'unknown')}{RESET}"

    r = result.get("result", {})
    parts = []
    if isinstance(r, dict):
        if r.get("Q"):
            parts.append(f"Q={r['Q']:,.0f}")
        if r.get("V"):
            parts.append(f"V={r['V']:.3f}")
        if r.get("qv_ratio"):
            parts.append(f"Q/V={r['qv_ratio']:,.0f}")
        if r.get("resonance_nm"):
            parts.append(f"res={r['resonance_nm']:.1f}nm")

    if result.get("message"):
        parts.append(result["message"])

    return "  ".join(parts) if parts else "ok"


async def main():
    print_banner()

    try:
        agent = CavityAgent()
    except ValueError as e:
        print(f"{RED}Error: {e}{RESET}")
        sys.exit(1)

    print(f"{DIM}Agent ready (model: {agent.model}){RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        try:
            async for event in agent.run(user_input):
                if isinstance(event, ThoughtEvent):
                    # Show thought in dim — this is the ReAct THOUGHT
                    print(f"\n{DIM}[THOUGHT] {event.text}{RESET}")

                elif isinstance(event, ToolStartEvent):
                    params_str = ", ".join(
                        f"{k}={v}" for k, v in event.input.items()
                        if k != "hypothesis"
                    )
                    print(f"{YELLOW}  > {event.name}({params_str}){RESET}")

                elif isinstance(event, ToolEndEvent):
                    ok = event.result.get("ok", True)
                    symbol = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
                    summary = format_result_summary(event.name, event.result)
                    print(f"  {symbol} {BOLD}{event.name}{RESET}  {DIM}{summary}{RESET}")

                elif isinstance(event, TextEvent):
                    # Final agent response text — already shown as ThoughtEvent
                    pass

                elif isinstance(event, ErrorEvent):
                    print(f"\n{RED}[ERROR] {event.message}{RESET}")

                elif isinstance(event, DoneEvent):
                    print()

        except Exception as e:
            print(f"\n{RED}[ERROR] {e}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
