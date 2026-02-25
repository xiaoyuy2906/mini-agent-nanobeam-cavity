"""
CLI entry point for nanobeam cavity design agent.

Usage:
    cavity chat                          # interactive chat (default)
    cavity auto [--iterations N]         # auto optimization
    cavity sweep PARAM START END [STEP]  # parameter sweep
    cavity history [--last N]            # view design history
    cavity best                          # show best design
"""

import os
import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from core.agent import CavityAgent
from core.state import CavityDesignState
from tools.toolset import Toolset

load_dotenv()

app = typer.Typer(
    name="cavity",
    help="Nanobeam cavity design & optimization agent",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

LOG_FILE = "cavity_design_log.json"


def _build_agent() -> CavityAgent:
    """Create agent from environment variables."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set. Check your .env file.[/red]")
        raise typer.Exit(1)

    model = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
    base_url = os.getenv("ANTHROPIC_BASE_URL")

    toolset = Toolset(api_key=api_key, model_name=model, base_url=base_url)
    state = CavityDesignState()
    return CavityAgent(toolset=toolset, state=state)


def _show_best(agent: CavityAgent) -> None:
    """Pretty-print the current best design."""
    best = agent.state.best_design
    if not best:
        console.print("[dim]No designs yet.[/dim]")
        return

    result = best["result"]
    params = best["params"]

    table = Table(title="Best Design", show_header=False, border_style="bright_cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Iteration", str(best["iteration"]))
    q = result.get("Q")
    table.add_row("Q", f"{q:,}" if q else "N/A")
    v = result.get("V")
    table.add_row("V", f"{v:.3f} (λ/n)³" if v else "N/A")
    qv = result.get("qv_ratio")
    table.add_row("Q/V", f"{qv:,.0f}" if qv else "N/A")
    res = result.get("resonance_nm")
    table.add_row("Resonance", f"{res:.2f} nm" if res else "N/A")
    table.add_row("Taper holes", str(params.get("num_taper_holes", "—")))
    table.add_row("Mirror holes", str(params.get("num_mirror_holes", "—")))
    table.add_row("min_a %", str(params.get("min_a_percent", "—")))
    table.add_row("GDS file", result.get("gds_file", "N/A"))

    console.print(table)


def _show_history(agent: CavityAgent, last_n: int | None = None) -> None:
    """Print design history as a rich table."""
    history = agent.state.design_history
    if not history:
        console.print("[dim]No history.[/dim]")
        return

    if last_n:
        history = history[-last_n:]

    table = Table(title="Design History", border_style="blue")
    table.add_column("#", style="dim")
    table.add_column("Q", justify="right")
    table.add_column("V", justify="right")
    table.add_column("Q/V", justify="right")
    table.add_column("Res. (nm)", justify="right")
    table.add_column("Period", justify="right")
    table.add_column("min_a %", justify="right")

    for entry in history:
        r = entry["result"]
        p = entry["params"]
        q = r.get("Q")
        v = r.get("V")
        qv = r.get("qv_ratio")
        table.add_row(
            str(entry["iteration"]),
            f"{q:,}" if q else "—",
            f"{v:.3f}" if v else "—",
            f"{qv:,.0f}" if qv else "—",
            f"{r.get('resonance_nm', 0):.1f}" if r.get("resonance_nm") else "—",
            str(p.get("period_nm", p.get("period", "—"))),
            str(p.get("min_a_percent", "—")),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def chat():
    """Interactive chat with the cavity design agent."""
    agent = _build_agent()

    provider = os.getenv("MODEL_PROVIDER", "claude")
    model = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")

    console.print(
        Panel(
            "[bold]Nanobeam Cavity Designer[/bold]\n\n"
            f"Model: {model} ({provider})\n"
            f"Log:   {LOG_FILE}\n\n"
            "Commands inside chat:\n"
            "  [cyan]confirm[/cyan]  — confirm FDTD inputs\n"
            "  [cyan]history[/cyan]  — show design history\n"
            "  [cyan]best[/cyan]     — show best design\n"
            "  [cyan]quit[/cyan]     — exit",
            title="cavity chat",
            border_style="bright_cyan",
        )
    )

    try:
        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            cmd = user_input.lower()

            if cmd in ("quit", "exit", "q"):
                break

            if cmd in ("history", "view_history"):
                _show_history(agent)
                continue

            if cmd in ("best", "get_best"):
                _show_best(agent)
                continue

            # TODO: wire up to agent.chat() once the LLM conversation loop
            #       is implemented in core/agent.py
            console.print(f"[dim](received: {user_input})[/dim]")
            console.print(
                "[yellow]Chat loop not yet wired — implement core/agent.chat()[/yellow]"
            )
    finally:
        if agent.state.unit_cell and agent.state.iteration > 0:
            agent.state.save_log(LOG_FILE)
            console.print(
                f"[dim]Saved {agent.state.iteration} iterations to {LOG_FILE}[/dim]"
            )


@app.command()
def auto(
    iterations: int = typer.Option(10, "--iterations", "-n", help="Max iterations per batch"),
    constraint: Optional[str] = typer.Option(None, "--constraint", "-c", help="Extra instruction for the optimizer"),
):
    """Run automated optimization loop."""
    agent = _build_agent()

    console.print(
        f"[bold]Auto optimization[/bold]  iterations={iterations}"
        + (f"  constraint={constraint!r}" if constraint else "")
    )

    # TODO: wire up to agent.run_optimization_loop() once implemented
    console.print("[yellow]Auto mode not yet wired — implement core/agent.run_optimization_loop()[/yellow]")
    _show_best(agent)


@app.command()
def sweep(
    param: str = typer.Argument(..., help="Parameter name (e.g. min_a_percent, hole_rx_nm)"),
    start: float = typer.Argument(..., help="Start value"),
    end: float = typer.Argument(..., help="End value"),
    step: Optional[float] = typer.Argument(None, help="Step size (auto-determined if omitted)"),
):
    """Run a deterministic parameter sweep."""
    agent = _build_agent()

    step_str = f" step {step}" if step else ""
    console.print(f"[bold]Sweep[/bold]  {param}  {start} → {end}{step_str}")

    # TODO: wire up to agent.run_manual_sweep() once implemented
    console.print("[yellow]Sweep not yet wired — implement core/agent.run_manual_sweep()[/yellow]")
    _show_best(agent)


@app.command()
def history(
    last: Optional[int] = typer.Option(None, "--last", "-n", help="Show only last N entries"),
):
    """View design history from the log file."""
    state = CavityDesignState()

    # Try to load from log
    # For now show a hint — load_log needs a unit_cell key
    import json
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
        for config_key, log_data in data.items():
            console.print(f"\n[bold cyan]Config:[/bold cyan] {config_key}")
            entries = log_data.get("design_history", [])
            if last:
                entries = entries[-last:]

            table = Table(border_style="blue")
            table.add_column("#", style="dim")
            table.add_column("Q", justify="right")
            table.add_column("V", justify="right")
            table.add_column("Q/V", justify="right")
            table.add_column("Res. (nm)", justify="right")

            for entry in entries:
                r = entry.get("result", {})
                q = r.get("Q")
                v = r.get("V")
                qv = r.get("qv_ratio")
                table.add_row(
                    str(entry.get("iteration", "?")),
                    f"{q:,}" if q else "—",
                    f"{v:.3f}" if v else "—",
                    f"{qv:,.0f}" if qv else "—",
                    f"{r.get('resonance_nm', 0):.1f}" if r.get("resonance_nm") else "—",
                )
            console.print(table)

            best = log_data.get("best_design")
            if best:
                br = best.get("result", {})
                console.print(
                    f"  Best: Q={br.get('Q', 'N/A'):,}  V={br.get('V', 'N/A')}  "
                    f"Q/V={br.get('qv_ratio', 'N/A')}"
                )
    else:
        console.print(f"[dim]No log file found at {LOG_FILE}[/dim]")


@app.command()
def best():
    """Show the best design from the log file."""
    import json
    if not os.path.exists(LOG_FILE):
        console.print(f"[dim]No log file found at {LOG_FILE}[/dim]")
        return

    with open(LOG_FILE, "r") as f:
        data = json.load(f)

    for config_key, log_data in data.items():
        best_entry = log_data.get("best_design")
        if not best_entry:
            continue

        r = best_entry.get("result", {})
        p = best_entry.get("params", {})

        table = Table(
            title=f"Best Design — {config_key[:40]}",
            show_header=False,
            border_style="bright_cyan",
        )
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("Iteration", str(best_entry.get("iteration", "?")))
        q = r.get("Q")
        table.add_row("Q", f"{q:,}" if q else "N/A")
        v = r.get("V")
        table.add_row("V", f"{v:.3f} (λ/n)³" if v else "N/A")
        qv = r.get("qv_ratio")
        table.add_row("Q/V", f"{qv:,.0f}" if qv else "N/A")
        res = r.get("resonance_nm")
        table.add_row("Resonance", f"{res:.2f} nm" if res else "N/A")
        table.add_row("GDS file", r.get("gds_file", "N/A"))
        console.print(table)


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    """Nanobeam cavity design & optimization agent."""
    if ctx.invoked_subcommand is None:
        chat()


if __name__ == "__main__":
    app()
