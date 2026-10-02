# Bedrock — Personal Computer Task Agent

A lightweight, multi-agent assistant that performs real tasks on your computer.
You describe a task in plain language. Bedrock plans it, picks tools, runs them
locally, checks its own work and reports back. Everything is shown in a Rich CLI.

```
User task → Planner → Reasoner ⇄ Executor → Tool → Observation → … → Verifier → Summarizer
```

Bedrock runs as a single Python process. It talks to an LLM API (Groq by default),
runs sandboxed Python tools, and uses no frameworks, databases or local models.

## Quick start

```bash
uv sync
cp .env.example .env          # then set GROQ_API_KEY
uv run pc-agent "How much disk space is available?"
uv run pc-agent               # interactive mode
```

Example tasks:

```text
Find all PDFs in ~/Downloads created this month.
Find the largest files in my home directory.
Show me which processes are consuming the most memory.
Find duplicate files in ~/Documents.
Find all markdown notes about Docker and create a consolidated summary in ~/Documents/docker.md.
Organize my Downloads folder by file type.
```

## CLI options

| Option | Effect |
| --- | --- |
| `--dry-run` | Read-only tools run as normal. Anything with side effects is only described: "would execute …". |
| `--verbose`, `-v` | Show raw tool output and the reasoner's working notes. |
| `--no-color` | Plain output (`NO_COLOR` is also respected). |
| `--model <name>` | Override `LLM_MODEL`. |
| `--allow-path <dir>` | Restrict filesystem tools to this directory (repeatable). |
| `--allow-destructive` | Enable `filesystem.delete`. |
| `--allow-python` | Enable `python.execute`. |
| `--max-steps <n>` | Reasoning/execution steps per attempt (default 20). |

## The agents

Each agent has its own module, prompt and permissions:

| Agent | Can | Cannot |
| --- | --- | --- |
| **Planner** (`agents/planner.py`) | Turn the request into a goal, steps and a list of risks | Run tools |
| **Reasoner** (`agents/reasoner.py`) | Read a compact view of the state and choose one action at a time, keeping notes as working memory | Touch the system |
| **Executor** (`agents/executor.py`) | Validate, permission-check and run tools. It is plain Python with no LLM. | Choose actions |
| **Verifier** (`agents/verifier.py`) | Run deterministic checks (files exist, are non-empty, moved files are in place) and get an LLM verdict; it has read-only tools | Modify anything |
| **Summarizer** (`agents/summarizer.py`) | Write the final report (falls back to a deterministic summary if the LLM is unavailable) | Use tools |

`core/orchestrator.py` is a small state machine:
plan → reason/execute/observe loop → verify → retry the loop if verification fails
(`max_verification_retries`, default 2) → summarize. Agents share a `TaskState`
(`core/state.py`); raw conversation history is never passed between them. Agent
outputs are validated with Pydantic models (`core/decisions.py`). Invalid model
output is sent back to the model for up to two retries.

## Tools

| Tool | Risk | Notes |
| --- | --- | --- |
| `filesystem.list` / `search` / `read` / `duplicates` | read | Search by glob, size or date, and sort by size. Read handles text and PDF (via `pypdf`). |
| `filesystem.write` | write | Overwriting an existing file needs `overwrite=true` **and** confirmation. |
| `filesystem.copy` | write | Never overwrites. |
| `filesystem.move` | destructive | Always asks for confirmation; you can approve all moves for the task. |
| `filesystem.delete` | destructive | Off unless `--allow-destructive` is set; always asks for confirmation. |
| `system.disk_usage` / `processes` / `environment` | read | `environment` returns variable names only, never secret values. |
| `shell.run` | privileged | See the safety section below. |
| `python.execute` | privileged | Off unless `--allow-python` is set. Runs in an isolated subprocess; always asks for confirmation. |

To add a tool, write a function `(args, ctx) -> ToolResult` and a `ToolArgs`
model, then return a `Tool(...)` from the module's `tools()` function. The model
can only call tools in the registry (`tools/registry.py`).

## Safety

- **Sandbox:** filesystem tools only work under the allowed roots: your home directory and the
  current directory by default, or whatever `PC_AGENT_ALLOWED_PATHS` / `--allow-path` set.
  Paths are resolved first, so symlinks and `..` can't escape the sandbox.
- **Secrets:** tools refuse `~/.ssh`, `~/.aws`, `~/.gnupg`, `.env`, `*.pem`, keys and
  credential files. Tool output and logs are scrubbed of any secret environment values.
  Subprocesses run with secret variables removed from their environment. API keys never reach the model.
- **Shell:** commands run **without a shell**. Pipes, redirects, `;`, `&&`, `$()` and
  backticks are rejected. Read-only commands (`ls`, `du`, `df`, `find`, `ps`, …) run directly.
  Anything else (`rm`, `mv`, `chmod`, `dd`, `systemctl`, unknown programs, `find -delete`, …)
  needs confirmation. `sudo`, `env`, nested shells and similar commands are always blocked.
- **Confirmation:** the agent stops and waits for `[y]`/`[n]`. Declining cancels the task.
  Without an interactive terminal, every risky action is declined.
- **Validation:** tool arguments are validated with Pydantic, and unknown arguments are rejected.
- **Logging:** every task writes `logs/task-<timestamp>.jsonl`. It records the plan, decisions,
  permission assessments, confirmations, tool calls and results, and verifications.

## Configuration

See `.env.example`. The main variables are `LLM_PROVIDER` (`groq`), `LLM_MODEL` and `GROQ_API_KEY`.
To add another provider, implement `LLMClient.generate()` in `llm/client.py`
and register it in `create_client()`. Nothing else depends on the provider.

## Development

```bash
uv run pytest
```

The tests cover the filesystem tools, including the sandbox and PDF reading. They also cover
permission classification, the registry and executor (dry-run, redaction), and the full
orchestration loop using a scripted LLM: recovery, verification retries, confirmation and
loop guards.

### Benchmarks

The benchmark suite lives in `tests/benchmark/`. It runs real tasks through the
real orchestrator in fresh temporary workspaces and validates the results by
inspecting the filesystem and the tools' actual output.

```bash
uv run pytest tests/benchmark tests/security tests/integration   # deterministic, offline
uv run python -m tests.benchmark.runner                          # all suites, scripted, JSON + terminal report
uv run python -m tests.benchmark.runner --runs 3 --category filesystem
uv run python -m tests.benchmark.runner --suite security
uv run python -m tests.benchmark.runner --mode live --runs 3      # needs GROQ_API_KEY
```

| Suite | What it measures |
| --- | --- |
| Tasks (`tasks.json`, 11 tasks) | Searching files by extension, size, date and content; duplicates; largest files; PDF reading; writing a summary; organizing files; disk and process inspection |
| Reliability (`reliability.py`) | Injected faults: invalid JSON, native tool calls, tool errors, verifier rejection, LLM outage, step/retry limits, declined confirmation |
| Security (`security_scenarios.py`) | 46 unauthorized attempts and 19 valid operations, tried at the direct-tool, executor and orchestrator layers, using a fake `HOME` and synthetic secrets |

**Scripted mode** (the default) replaces the LLM with predetermined replies. It needs no network
or API key. It measures the tools, the orchestration, recovery and the safety controls; it does
**not** measure how well a model reasons. **Live mode** runs only the task suite against the
configured model. Its results are non-deterministic and are reported separately. During
unattended runs, filesystem confirmations are approved, because those tools can only reach the
temporary workspace. `shell.run` and `python.execute` confirmations are always declined.

Reports are written to `benchmark_results/benchmark-<timestamp>.json`. Each one contains every
execution record, the configuration and environment metadata. Before writing, the report is
checked for secret values and is refused if one is found. A metric with no data is reported as
`n/a` or `unavailable`, never as 0.

```
src/pc_agent/
  main.py, config.py
  agents/   base, planner, reasoner, executor, verifier, summarizer
  core/     state, orchestrator, decisions, permissions, events (JSONL log)
  tools/    registry, filesystem, shell, system, python
  llm/      base (interface), client (Groq + factory)
  cli/      interface (argparse/REPL), renderer (Rich), confirmation
```
