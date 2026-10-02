"""Opt-in Python execution tool. Disabled unless --allow-python is given."""

from __future__ import annotations

import subprocess
import sys
import tempfile

from pydantic import Field

from pc_agent.config import Config
from pc_agent.core.decisions import RiskLevel, ToolResult
from pc_agent.core.permissions import scrubbed_environment
from pc_agent.tools.registry import Tool, ToolArgs, ToolContext

MAX_OUTPUT_CHARS = 10_000


class PythonArgs(ToolArgs):
    code: str = Field(description="Python source to run; print() results to stdout")
    timeout: int = Field(default=30, ge=1, le=120)


def execute_python(args: PythonArgs, ctx: ToolContext) -> ToolResult:
    with tempfile.TemporaryDirectory(prefix="pc-agent-") as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", args.code],
                cwd=workdir,
                env=scrubbed_environment(),
                capture_output=True,
                text=True,
                timeout=args.timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"Python code timed out after {args.timeout}s")
    output = proc.stdout[:MAX_OUTPUT_CHARS]
    if proc.returncode != 0:
        return ToolResult(
            success=False,
            output=output,
            error=proc.stderr.strip()[-1500:] or f"exit code {proc.returncode}",
            summary=f"Python exited with code {proc.returncode}",
        )
    return ToolResult.ok(output or "(no output)", f"Python ran OK ({len(output.splitlines())} lines of output)")


def tools(config: Config) -> list[Tool]:
    return [
        Tool(
            "python.execute",
            "Run a short Python script in an isolated subprocess (for data analysis, e.g. CSV statistics). "
            "Requires user confirmation.",
            PythonArgs, execute_python, RiskLevel.PRIVILEGED,
            enabled=config.allow_python,
            disabled_reason="arbitrary Python execution is opt-in; start with --allow-python",
        )
    ]
