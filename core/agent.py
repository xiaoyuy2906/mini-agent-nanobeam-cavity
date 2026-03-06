"""CavityAgent — ReAct agent with SWE-agent-style history management.

Architecture:
- Tool registry: tools self-register via decorators (no if-else dispatch)
- History: sliding window compression (old observations → one-line summaries)
- Forced thought: hypothesis is required on design_cavity, plus reflection injection
- Single process: no IPC, no JSON bridge, just async generator
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from core.history import compress_history, truncate_observation
from core.state import CavityDesignState
from core.tool_registry import dispatch, get_all_schemas

# Import tools to trigger registration
import core.tools  # noqa: F401

_log = lambda *a, **kw: print(*a, file=sys.stderr, **kw)

# --- Event types yielded to the UI layer ---

class AgentEvent:
    """Base event yielded by the agent loop."""
    pass

class ThoughtEvent(AgentEvent):
    def __init__(self, text: str):
        self.text = text

class ToolStartEvent(AgentEvent):
    def __init__(self, name: str, input: dict):
        self.name = name
        self.input = input

class ToolEndEvent(AgentEvent):
    def __init__(self, name: str, result: dict):
        self.name = name
        self.result = result

class TextEvent(AgentEvent):
    def __init__(self, text: str):
        self.text = text

class DoneEvent(AgentEvent):
    pass

class ErrorEvent(AgentEvent):
    def __init__(self, message: str):
        self.message = message


# --- Reflection (injected periodically, like ReAct's forced reasoning) ---

REFLECTION_INTERVAL = 5

REFLECTION_PROMPT = (
    "You have completed {n} tool calls. Before continuing, you MUST reflect:\n"
    "1. Current best Q/V and how it's trending\n"
    "2. Which parameter had the most impact\n"
    "3. Whether to change strategy or continue current sweep\n"
    "State your updated plan, then call your next tool."
)


class CavityAgent:
    """ReAct agent for nanobeam cavity design."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        self.client = (
            AsyncAnthropic(api_key=api_key, base_url=base_url)
            if base_url
            else AsyncAnthropic(api_key=api_key)
        )
        self.model = os.getenv("MODEL_NAME", "claude-sonnet-4-6")
        self.state = CavityDesignState()
        self.messages: list[dict] = []
        self.tool_call_count = 0
        self.system_prompt = self._build_system_prompt()
        self.tools = get_all_schemas()

    def _build_system_prompt(self) -> str:
        skills_path = Path(__file__).parent.parent / "skills.md"
        try:
            skills_text = skills_path.read_text(encoding="utf-8").strip()
        except OSError:
            skills_text = ""

        # ReAct enforcement: explicit Thought-Action-Observation structure
        react_preamble = (
            "You are a ReAct agent for nanobeam photonic crystal cavity design.\n\n"
            "## STRICT ReAct Protocol\n"
            "Every turn you MUST follow this exact structure:\n\n"
            "THOUGHT: [Analyze the current situation. What did you learn from the last "
            "observation? What should you try next and why?]\n\n"
            "Then call exactly ONE tool.\n\n"
            "You will receive an OBSERVATION (tool result). Then repeat.\n\n"
            "NEVER call a tool without first writing a THOUGHT section.\n"
            "NEVER skip the THOUGHT — it is mandatory.\n\n"
            "## CRITICAL: USER INPUT OVERRIDES EVERYTHING\n"
            "If the user gives explicit instructions, follow them exactly.\n"
            "Never invent missing unit-cell geometry. If a required value is missing, ask.\n"
            "Before the FIRST FDTD run, show all unit-cell inputs and ask user confirmation.\n"
            "Only proceed after user says 'confirm fdtd'.\n\n"
        )

        if skills_text:
            return react_preamble + skills_text
        return react_preamble + self._fallback_prompt()

    @staticmethod
    def _fallback_prompt() -> str:
        return (
            "## Goal\n"
            "Maximize Q/V. Q > 1,000,000 and V < 0.5 (lambda/n)^3 is excellent.\n\n"
            "## Tools\n"
            "- set_unit_cell: configure geometry (call first)\n"
            "- design_cavity: build GDS + run FDTD\n"
            "- view_history: inspect previous designs\n"
            "- compare_designs: compare specific iterations\n"
            "- get_best_design: retrieve current best\n"
            "- analyze_sensitivity: compute parameter sensitivities\n"
            "- suggest_next_experiment: data-driven next step recommendation\n"
        )

    async def run(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """Run one user turn through the ReAct loop. Yields events for the UI."""
        self.messages.append({"role": "user", "content": user_input})

        while True:
            # SWE-agent pattern: compress history before each LLM call
            compressed = compress_history(self.messages)

            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    tools=self.tools,
                    messages=compressed,
                )
            except Exception as e:
                yield ErrorEvent(str(e))
                return

            # Store the full (uncompressed) assistant response
            self.messages.append({"role": "assistant", "content": response.content})

            # Yield text blocks (the THOUGHT part of ReAct)
            for block in response.content:
                if block.type == "text" and block.text:
                    yield ThoughtEvent(block.text)

            # If no tool use, the agent is done for this turn
            if response.stop_reason != "tool_use":
                # Yield any final text as a response
                for block in response.content:
                    if block.type == "text" and block.text:
                        yield TextEvent(block.text)
                break

            # Execute tools
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = []

            for block in tool_blocks:
                yield ToolStartEvent(block.name, block.input)

                result = await dispatch(block.name, block.input, self)
                self.tool_call_count += 1

                yield ToolEndEvent(block.name, result)

                # SWE-agent pattern: format observation for LLM readability
                formatted = self._format_tool_result(block.name, result)
                formatted = truncate_observation(formatted)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": formatted,
                })

            # Inject reflection prompt periodically (ReAct forced reasoning)
            if (
                self.tool_call_count > 0
                and self.tool_call_count % REFLECTION_INTERVAL == 0
            ):
                tool_results.append({
                    "type": "text",
                    "text": REFLECTION_PROMPT.format(n=self.tool_call_count),
                })

            self.messages.append({"role": "user", "content": tool_results})

        yield DoneEvent()

    @staticmethod
    def _format_tool_result(tool_name: str, result: dict) -> str:
        """Format tool result as human-readable text, not raw JSON.

        This is the SWE-agent ACI principle: tool outputs should be
        formatted for LLM comprehension, not dumped as JSON.
        """
        if not result.get("ok", True):
            error = result.get("error", "Unknown error")
            return f"ERROR: {error}"

        if tool_name == "design_cavity":
            r = result.get("result", {})
            iteration = result.get("iteration", "?")
            q = r.get("Q")
            v = r.get("V")
            qv = r.get("qv_ratio")
            res_nm = r.get("resonance_nm")
            best = result.get("best_qv_ratio", 0)

            lines = [f"=== Iteration #{iteration} Result ==="]
            if q is not None:
                lines.append(f"  Q factor:    {q:,.0f}")
            if v is not None:
                lines.append(f"  Mode volume: {v:.4f} (lambda/n)^3")
            if qv is not None:
                lines.append(f"  Q/V ratio:   {qv:,.0f}")
            if res_nm is not None:
                lines.append(f"  Resonance:   {res_nm:.2f} nm")
            lines.append(f"  Best Q/V so far: {best:,.0f}")
            return "\n".join(lines)

        if tool_name == "set_unit_cell":
            msg = result.get("message", "Unit cell configured")
            return f"OK: {msg}"

        if tool_name == "view_history":
            history = result.get("history", [])
            total = result.get("total", 0)
            if not history:
                return "No designs yet."
            lines = [f"Design history ({total} total):"]
            for entry in history:
                i = entry.get("iteration", "?")
                r = entry.get("result", {})
                q = r.get("Q", "N/A")
                v = r.get("V", "N/A")
                qv = r.get("qv_ratio", "N/A")
                res = r.get("resonance_nm", "N/A")
                q_str = f"{q:,.0f}" if isinstance(q, (int, float)) else str(q)
                v_str = f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
                qv_str = f"{qv:,.0f}" if isinstance(qv, (int, float)) else str(qv)
                res_str = f"{res:.1f}nm" if isinstance(res, (int, float)) else str(res)
                lines.append(f"  #{i}: Q={q_str}  V={v_str}  Q/V={qv_str}  res={res_str}")
            return "\n".join(lines)

        if tool_name == "compare_designs":
            designs = result.get("designs", [])
            if not designs:
                return "No designs to compare."
            lines = ["Design comparison:"]
            for d in designs:
                if "error" in d:
                    lines.append(f"  #{d.get('iteration','?')}: {d['error']}")
                    continue
                i = d.get("iteration", "?")
                r = d.get("result", {})
                p = d.get("params", {})
                lines.append(f"  #{i}: Q={r.get('Q', 'N/A'):,.0f}  V={r.get('V', 'N/A'):.3f}  Q/V={r.get('qv_ratio', 'N/A'):,.0f}")
                lines.append(f"      params: min_a={p.get('min_a_percent','?')}%  taper={p.get('num_taper_holes','?')}  mirror={p.get('num_mirror_holes','?')}")
            return "\n".join(lines)

        if tool_name == "get_best_design":
            best = result.get("best_design", {})
            if not best:
                return result.get("message", "No design yet.")
            r = best.get("result", {})
            p = best.get("params", {})
            return (
                f"Best design (iteration #{best.get('iteration','?')}):\n"
                f"  Q={r.get('Q', 'N/A'):,.0f}  V={r.get('V', 'N/A'):.3f}  "
                f"Q/V={r.get('qv_ratio', 'N/A'):,.0f}  res={r.get('resonance_nm', 'N/A'):.1f}nm\n"
                f"  min_a={p.get('min_a_percent','?')}%  taper_holes={p.get('num_taper_holes','?')}  "
                f"mirror_holes={p.get('num_mirror_holes','?')}"
            )

        if tool_name in ("analyze_sensitivity", "suggest_next_experiment"):
            # These are already structured — format nicely
            return json.dumps(result, indent=2, default=str)

        # Fallback: compact JSON
        return json.dumps(result, default=str)
