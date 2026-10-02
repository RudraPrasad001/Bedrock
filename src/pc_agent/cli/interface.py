"""Command-line entry point: argument parsing, one-shot and interactive modes."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from pc_agent import __version__
from pc_agent.cli.renderer import RichRenderer
from pc_agent.config import Config

EXIT_WORDS = {"exit", "quit", ":q"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pc-agent",
        description="Bedrock: a multi-agent assistant that performs tasks on your computer.",
    )
    parser.add_argument("task", nargs="*", help="Task to perform. Omit for interactive mode.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run; change nothing.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool output and agent notes.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output.")
    parser.add_argument("--model", help="LLM model to use (overrides LLM_MODEL).")
    parser.add_argument("--allow-python", action="store_true", help="Enable the python.execute tool.")
    parser.add_argument("--allow-destructive", action="store_true", help="Enable filesystem.delete.")
    parser.add_argument(
        "--allow-path", action="append", default=[], metavar="PATH",
        help="Restrict filesystem tools to this directory (repeatable). Default: home and cwd.",
    )
    parser.add_argument("--max-steps", type=int, help="Max reasoning steps per attempt.")
    parser.add_argument("--version", action="version", version=f"pc-agent {__version__}")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    config = Config.from_env()
    config.dry_run = args.dry_run
    config.verbose = args.verbose
    config.color = not args.no_color
    config.allow_python = config.allow_python or args.allow_python
    config.allow_destructive = config.allow_destructive or args.allow_destructive
    if args.model:
        config.model = args.model
    if args.allow_path:
        config.allowed_paths = [Path(p).expanduser().resolve() for p in args.allow_path]
    if args.max_steps:
        config.max_steps = args.max_steps
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    console = Console(no_color=not config.color, highlight=False)

    from pc_agent.core.orchestrator import Orchestrator
    from pc_agent.llm.base import LLMError
    from pc_agent.llm.client import create_client

    try:
        llm = create_client(config)
    except LLMError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2

    renderer = RichRenderer(console, verbose=config.verbose, log_dir=config.log_dir)
    orchestrator = Orchestrator(config, llm, renderer)
    renderer.banner(config.model, config.dry_run, config.allowed_paths)

    def run(task: str) -> int:
        renderer.show_task(task)
        state = orchestrator.run(task)
        return 0 if state.status.value == "COMPLETED" else 1

    if args.task:
        return run(" ".join(args.task))

    console.print("[dim]Type a task, or 'exit' to quit. Ctrl+C cancels a running task.[/dim]")
    code = 0
    while True:
        try:
            task = console.input("\n[bold]>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return code
        if not task:
            continue
        if task.lower() in EXIT_WORDS:
            return code
        code = run(task)
