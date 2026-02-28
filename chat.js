#!/usr/bin/env node
/**
 * chat.js — Node.js terminal chat for nanobeam cavity design agent.
 * Spawns agent_server.py as a subprocess, communicates via JSON lines.
 *
 * Usage: node chat.js
 */
const { spawn } = require("child_process");
const readline = require("readline");

// ANSI colors (zero-dep)
const c = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
};
const fmt = (color, text) => `${color}${text}${c.reset}`;

// Spawn Python agent_server.py
const py = spawn("uv", ["run", "python", "agent_server.py"], {
  stdio: ["pipe", "pipe", "pipe"],
  env: { ...process.env, PYTHONUNBUFFERED: "1" },
});
py.stderr.pipe(process.stderr); // Lumerical/debug output stays visible

// Newline-delimited JSON buffer
let jsonBuffer = "";
py.stdout.on("data", (chunk) => {
  jsonBuffer += chunk.toString();
  const lines = jsonBuffer.split("\n");
  jsonBuffer = lines.pop(); // keep partial line
  for (const line of lines) handleLine(line.trim());
});

function handleLine(line) {
  if (!line) return;
  let event;
  try {
    event = JSON.parse(line);
  } catch (e) {
    process.stderr.write(`[parse error] ${line}\n`);
    return;
  }

  switch (event.type) {
    case "ready":
      printBanner();
      rl.prompt();
      break;

    case "text":
      process.stdout.write(fmt(c.cyan, event.delta || ""));
      break;

    case "tool_start":
      process.stdout.write("\n");
      console.log(
        fmt(c.yellow, "  [tool] ") +
          fmt(c.bold, event.name) +
          fmt(c.dim, " " + JSON.stringify(event.input))
      );
      break;

    case "tool_end": {
      const r =
        event.result?.result || event.result || {};
      const ok = event.result?.ok !== false;
      const symbol = ok ? fmt(c.green, "  ✓") : fmt(c.red, "  ✗");
      const parts = [
        r.Q ? `Q=${Number(r.Q).toLocaleString()}` : null,
        r.V ? `V=${Number(r.V).toFixed(3)}` : null,
        r.qv_ratio
          ? `Q/V=${Number(r.qv_ratio).toLocaleString()}`
          : null,
        !ok && (event.result?.error || event.result?.message)
          ? event.result.error || event.result.message
          : null,
      ].filter(Boolean);
      console.log(
        `${symbol} ${fmt(c.bold, event.name)}` +
          (parts.length ? `  ${fmt(c.dim, parts.join("  "))}` : "")
      );
      break;
    }

    case "done":
      process.stdout.write("\n");
      rl.prompt();
      break;

    case "error":
      process.stdout.write("\n");
      console.error(fmt(c.red, `[error] ${event.message}`));
      rl.prompt();
      break;

    default:
      process.stderr.write(`[unknown event] ${line}\n`);
  }
}

// Readline interface — prompt is suppressed until Python emits "ready" or "done"
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  prompt: fmt(c.green + c.bold, "You: ") + c.reset,
  terminal: true,
});

rl.on("line", (input) => {
  input = input.trim();
  if (!input) {
    rl.prompt();
    return;
  }
  if (["quit", "exit", "q"].includes(input.toLowerCase())) {
    console.log(fmt(c.dim, "Goodbye."));
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
    console.error(fmt(c.red, `\n[Python exited: ${code}]`));
  process.exit(code ?? 0);
});

function printBanner() {
  console.log(fmt(c.cyan + c.bold, "\n  Nanobeam Cavity Designer"));
  console.log(fmt(c.dim, "  Type your request. 'quit' to exit.\n"));
}
