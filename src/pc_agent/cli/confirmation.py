"""Blocking confirmation dialog for risky actions."""

from __future__ import annotations

import sys

from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from pc_agent.core.decisions import AgentDecision, PermissionAssessment


def describe_action(decision: AgentDecision) -> str:
    args = decision.arguments
    if decision.action == "shell.run":
        return f"Command: {args.get('command', '')}"
    if decision.action == "python.execute":
        code = str(args.get("code", ""))
        return "Python code:\n" + (code if len(code) < 1200 else code[:1200] + "\n…")
    if decision.action in ("filesystem.move", "filesystem.copy"):
        return f"{decision.action}: {args.get('source')} → {args.get('destination')}"
    if decision.action == "filesystem.write":
        return f"Write file: {args.get('path')} ({len(str(args.get('content', ''))):,} chars)"
    return f"{decision.action}: {', '.join(f'{k}={v}' for k, v in args.items())}"


def ask_confirmation(
    console: Console, decision: AgentDecision, assessment: PermissionAssessment, allow_all: bool
) -> str:
    body = [
        Text(describe_action(decision), style="bold"),
        Text(""),
        Text(assessment.explanation or f"This is a {assessment.risk.value} operation."),
    ]
    if decision.reason:
        body.append(Text(f"Reason: {decision.reason}", style="dim"))
    options = Text("\n[y] Execute    [n] Cancel task")
    if allow_all:
        options.append(f"    [a] Allow all {decision.action} for this task")
    body.append(options)
    console.print(
        Panel(Group(*body), title="ACTION REQUIRES CONFIRMATION", title_align="left",
              border_style="bold red", expand=False)
    )
    if not sys.stdin.isatty():
        console.print("[yellow]No interactive terminal; action cancelled.[/yellow]")
        return "no"
    choices = ["y", "n", "a"] if allow_all else ["y", "n"]
    try:
        answer = Prompt.ask("Proceed?", console=console, choices=choices, default="n")
    except (EOFError, KeyboardInterrupt):
        return "no"
    return {"y": "yes", "a": "all"}.get(answer, "no")
