#!/usr/bin/env node
/**
 * chat.js — Node.js terminal chat for nanobeam cavity design agent.
 * Spawns agent_server.py as a subprocess, communicates via JSON lines.
 *
 * Usage: node chat.js
 */
const { spawn } = require("child_process");
const readline = require("readline");
const chalk = require("chalk");
const figlet = require("figlet");

async function main() {
  const py = spawn("uv", ["run", "python", "agent_server.py"], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  py.stderr.pipe(process.stderr); // Lumerical/debug output stays visible

  // User input interface
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: chalk.green.bold("You: "),
    terminal: true,
  });

  rl.on("line", (input) => {
    input = input.trim();
    if (!input) {
      rl.prompt();
      return;
    }
    if (["quit", "exit", "q"].includes(input.toLowerCase())) {
      console.log(chalk.dim("Goodbye."));
      py.stdin.end();
      process.exit(0);
    }
    py.stdin.write(
      JSON.stringify({ type: "user_message", content: input }) + "\n"
    );
    // Prompt is suppressed until Python emits "done"
  });

  rl.on("close", () => {
    py.stdin.end();
    process.exit(0);
  });

  py.on("exit", (code) => {
    if (code !== 0 && code !== null)
      console.error(chalk.red(`\n[Python exited: ${code}]`));
    process.exit(code ?? 0);
  });

  // Read Python stdout line-by-line via async iterator — no manual buffer needed
  const pyLines = readline.createInterface({ input: py.stdout });
  for await (const line of pyLines) {
    if (!line.trim()) continue;
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      process.stderr.write(`[parse error] ${line}\n`);
      continue;
    }
    handleEvent(event, rl);
  }
}

let agentTurnStarted = false;

function handleEvent(event, rl) {
  switch (event.type) {
    case "ready":
      printBanner();
      rl.prompt();
      break;

    case "text":
      if (!agentTurnStarted) {
        process.stdout.write(chalk.green.bold("\nAgent: "));
        agentTurnStarted = true;
      }
      process.stdout.write(chalk.cyan(event.delta || ""));
      break;

    case "tool_start":
      agentTurnStarted = false;
      process.stdout.write("\n");
      console.log(
        chalk.yellow("  [tool] ") +
        chalk.bold(event.name) +
        chalk.dim(" " + JSON.stringify(event.input))
      );
      break;

    case "tool_end": {
      const r = event.result?.result || event.result || {};
      const ok = event.result?.ok !== false;
      const symbol = ok ? chalk.green("  ✓") : chalk.red("  ✗");
      const parts = [
        r.Q ? `Q=${Number(r.Q).toLocaleString()}` : null,
        r.V ? `V=${Number(r.V).toFixed(3)}` : null,
        r.qv_ratio ? `Q/V=${Number(r.qv_ratio).toLocaleString()}` : null,
        !ok && (event.result?.error || event.result?.message)
          ? event.result.error || event.result.message
          : null,
      ].filter(Boolean);
      console.log(
        `${symbol} ${chalk.bold(event.name)}` +
        (parts.length ? `  ${chalk.dim(parts.join("  "))}` : "")
      );
      break;
    }

    case "done":
      agentTurnStarted = false;
      process.stdout.write("\n");
      rl.prompt();
      break;

    case "error":
      process.stdout.write("\n");
      console.error(chalk.red(`[error] ${event.message}`));
      rl.prompt();
      break;

    default:
      process.stderr.write(`[unknown event] ${line}\n`);
  }
}

function printBanner() {
  console.log(chalk.cyan.bold(figlet.textSync("Nanobeam Cavity Agent", { font: "Small" })));
  console.log(chalk.dim("Nanobeam Cavity Designer — type your request. 'quit' to exit.\n"));
}

main().catch((err) => {
  console.error(chalk.red(`[fatal] ${err.message}`));
  process.exit(1);
});
