"""Privileged shell tool.

Commands run without a shell (no pipes, redirects or substitution), with a
scrubbed environment and a timeout. Anything outside the read-only allowlist
requires explicit user confirmation.
"""

from __future__ import annotations

import os
import subprocess

from pydantic import Field

from pc_agent.config import Config
from pc_agent.core.decisions import PermissionAssessment, RiskLevel, ToolResult
from pc_agent.core.permissions import (
    CommandClass,
    PathPolicyError,
    classify_command,
    scrubbed_environment,
)
from pc_agent.tools.registry import Tool, ToolArgs, ToolContext

MAX_OUTPUT_CHARS = 10_000


class ShellArgs(ToolArgs):
    command: str = Field(description="A single command, no pipes/redirects/;/&&. e.g. 'du -sh ~/Downloads'")
    cwd: str | None = Field(default=None, description="Working directory (must be inside allowed paths)")
    timeout: int = Field(default=30, ge=1, le=300)


def assess_shell(args: ShellArgs, ctx: ToolContext) -> PermissionAssessment:
    verdict = classify_command(args.command, ctx.path_policy)
    if verdict.classification == CommandClass.BLOCKED:
        return PermissionAssessment(allowed=False, risk=RiskLevel.PRIVILEGED, explanation=verdict.explanation)
    if verdict.classification == CommandClass.DANGEROUS:
        return PermissionAssessment(
            allowed=True, risk=RiskLevel.PRIVILEGED, requires_confirmation=True, explanation=verdict.explanation
        )
    return PermissionAssessment(allowed=True, risk=RiskLevel.READ, explanation=verdict.explanation)


def run_shell(args: ShellArgs, ctx: ToolContext) -> ToolResult:
    verdict = classify_command(args.command, ctx.path_policy)
    if verdict.classification == CommandClass.BLOCKED:
        return ToolResult.fail(f"Command blocked: {verdict.explanation}", command=args.command)
    try:
        cwd = ctx.path_policy.resolve(args.cwd) if args.cwd else None
    except PathPolicyError as exc:
        return ToolResult.fail(str(exc))
    # No shell means no tilde expansion, so do it here.
    argv = [os.path.expanduser(a) if a.startswith("~") else a for a in verdict.argv]
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=scrubbed_environment(),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return ToolResult.fail(f"Command not found: {argv[0]}", command=args.command)
    except subprocess.TimeoutExpired:
        return ToolResult.fail(f"Command timed out after {args.timeout}s", command=args.command)
    except OSError as exc:
        return ToolResult.fail(f"Could not run command: {exc}", command=args.command)

    stdout, stderr = proc.stdout, proc.stderr
    truncated = len(stdout) > MAX_OUTPUT_CHARS
    output = stdout[:MAX_OUTPUT_CHARS] + ("\n[output truncated]" if truncated else "")
    if stderr.strip():
        output += f"\n[stderr]\n{stderr[:2000]}"
    lines = len(stdout.splitlines())
    if proc.returncode != 0:
        return ToolResult(
            success=False,
            output=output,
            error=f"Exit code {proc.returncode}: {stderr.strip()[:300] or 'no stderr'}",
            summary=f"`{args.command}` failed with exit code {proc.returncode}",
            metadata={"command": args.command, "exit_code": proc.returncode},
        )
    return ToolResult.ok(
        output or "(no output)",
        f"`{args.command}` -> {lines} line{'s' if lines != 1 else ''} of output",
        command=args.command,
        exit_code=0,
        truncated=truncated,
    )


def tools(config: Config) -> list[Tool]:
    return [
        Tool(
            "shell.run",
            "Run one shell command without a shell. Prefer dedicated tools. "
            "Read-only commands (ls, du, df, find, ps...) run directly; others need user confirmation.",
            ShellArgs, run_shell, RiskLevel.PRIVILEGED, assess=assess_shell,
        )
    ]
