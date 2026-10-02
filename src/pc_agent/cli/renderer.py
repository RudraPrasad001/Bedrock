"""Rich rendering of the multi-agent workflow. Implements the AgentUI protocol."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from pc_agent.cli.confirmation import ask_confirmation
from pc_agent.core.decisions import (
    AgentDecision,
    PermissionAssessment,
    Plan,
    ReasonerOutput,
    RiskLevel,
    ToolResult,
    Verification,
)
from pc_agent.core.state import TaskState, TaskStatus

AGENT_STYLES = {
    "planner": "bold magenta",
    "reasoner": "bold cyan",
    "executor": "bold yellow",
    "verifier": "bold blue",
    "summarizer": "bold green",
    "orchestrator": "bold red",
}
RISK_STYLES = {
    RiskLevel.READ: "green",
    RiskLevel.WRITE: "yellow",
    RiskLevel.DESTRUCTIVE: "red",
    RiskLevel.PRIVILEGED: "bold red",
}
INDENT = "  "
HOME = str(Path.home())


def short(text: str) -> str:
    """Display paths under the home directory as ~/..."""
    return text.replace(HOME + "/", "~/") if HOME != "/" else text


def format_call(action: str, arguments: dict, max_value: int = 120) -> str:
    """Render a tool call as `tool(\n  key="value",\n)`."""
    if not arguments:
        return f"{action}()"
    lines = [f"{action}("]
    for key, value in arguments.items():
        if value is None:
            continue  # unset optional arguments are noise
        if isinstance(value, str) and len(value) > max_value:
            value = value[:max_value].replace("\n", "\\n") + f"... ({len(value):,} chars)"
            rendered = f'"{value}"'
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        rendered = short(rendered)
        lines.append(f"    {key}={rendered},")
    lines.append(")")
    return "\n".join(lines)


class RichRenderer:
    def __init__(self, console: Console, verbose: bool = False, log_dir: Path | None = None):
        self.console = console
        self.verbose = verbose
        self.log_dir = log_dir
        self._step_count = 0

    # -- layout helpers -------------------------------------------------

    def _header(self, agent: str, subtitle: str = "") -> None:
        self.console.print(Rule(style="grey35"))
        title = Text("◉ ", style=AGENT_STYLES.get(agent, "bold"))
        title.append(agent.upper(), style=AGENT_STYLES.get(agent, "bold"))
        if subtitle:
            title.append(f"  {subtitle}", style="dim")
        self.console.print(title)

    def _line(self, text: str | Text, style: str = "") -> None:
        if isinstance(text, str):
            text = Text(short(text), style=style)
        self.console.print(Text(INDENT) + text)

    def banner(self, model: str, dry_run: bool, allowed_paths: list[Path]) -> None:
        info = Text()
        info.append("Bedrock", style="bold")
        info.append(" · Personal Computer Agent\n", style="bold")
        info.append(f"model {model}", style="dim")
        info.append(f"  ·  sandbox {', '.join(str(p) for p in allowed_paths)}", style="dim")
        if dry_run:
            info.append("\nDRY RUN — no changes will be made", style="bold yellow")
        self.console.print(Panel(info, border_style="grey50", expand=False))

    def show_task(self, request: str) -> None:
        self.console.print()
        self.console.print(Text("Task", style="bold"))
        for line in request.splitlines() or [""]:
            self.console.print(Text("> ", style="dim") + Text(line))

    # -- AgentUI ----------------------------------------------------------

    @contextmanager
    def thinking(self, agent: str, message: str):
        style = AGENT_STYLES.get(agent, "bold")
        with self.console.status(Text(f"{agent.upper()}  {message}", style=style), spinner="dots"):
            yield

    def show_plan(self, plan: Plan) -> None:
        self._header("planner", "task understood")
        self._line(Text("Goal: ", style="bold") + Text(plan.goal))
        self._line("Plan:", "bold")
        for step in plan.steps:
            row = Text(f"  {step.id}. {step.description}")
            if step.expected_tool:
                row.append(f"  [{step.expected_tool}]", style="dim")
            self._line(row)
        if plan.risks:
            self._line("Risks:", "bold yellow")
            for risk in plan.risks:
                self._line(f"  ! {risk}", "yellow")

    def show_reasoning(self, output: ReasonerOutput) -> None:
        self._step_count += 1
        subtitle = f"step {output.step_id}" if output.step_id else ""
        self._header("reasoner", subtitle)
        self._line(output.reason, "italic")
        if output.status == "continue" and output.action:
            call = Text("→ ", style="cyan") + Text(format_call(output.action, output.arguments), style="cyan")
            for line in call.split("\n"):
                self._line(line)
        elif output.status == "complete":
            self._line("✓ Work complete — handing over to verifier", "green")
        elif output.status == "failed":
            self._line("✗ Task cannot be completed", "red")
        if output.note and self.verbose:
            self._line(f"note: {output.note}", "dim")

    def show_tool_call(self, decision: AgentDecision, assessment: PermissionAssessment) -> None:
        self._header("executor")
        risk = Text(f"[{assessment.risk.value}]", style=RISK_STYLES[assessment.risk])
        self._line(Text(f"Running {decision.action} ") + risk)
        if not assessment.allowed:
            self._line(f"✗ Blocked: {assessment.explanation}", "red")

    def show_tool_result(self, decision: AgentDecision, result: ToolResult) -> None:
        if result.metadata.get("blocked"):
            return  # already shown by show_tool_call
        if result.success:
            style = "yellow" if result.metadata.get("dry_run") else "green"
            self._line(f"✓ {result.summary or 'done'}", style)
        else:
            self._line(f"✗ {result.error}", "red")
        if self.verbose and result.output:
            body = result.output if len(result.output) <= 3000 else result.output[:3000] + "\n…"
            self.console.print(Panel(Text(body), border_style="grey35", title="output", title_align="left"))

    def confirm(self, decision: AgentDecision, assessment: PermissionAssessment, allow_all: bool):
        return ask_confirmation(self.console, decision, assessment, allow_all)

    def show_verification(self, verification: Verification, attempt: int) -> None:
        self._header("verifier", f"attempt {attempt}" if attempt > 1 else "")
        for check in verification.checks:
            self._line(f"{'✓' if check.passed else '✗'} {check.check}", "green" if check.passed else "red")
        if verification.success:
            self._line("✓ Result verified", "bold green")
        else:
            for issue in verification.issues:
                self._line(f"! {issue}", "yellow")
            if verification.recommended_action:
                self._line(f"→ {verification.recommended_action}", "cyan")

    def show_summary(self, summary: str, state: TaskState) -> None:
        self._header("summarizer")
        border = {
            TaskStatus.COMPLETED: "green",
            TaskStatus.CANCELLED: "yellow",
        }.get(state.status, "red")
        footer = Text(
            f"{state.status.value} · {len(state.observations)} tool calls"
            + (" · dry run" if state.dry_run else ""),
            style="dim",
        )
        if self.log_dir:
            footer.append(f" · log {self.log_dir / f'task-{state.task_id}.jsonl'}", style="dim")
        self.console.print(Panel(Group(Markdown(summary), Text(), footer), border_style=border, padding=(1, 2)))

    def show_error(self, agent: str, message: str) -> None:
        self._header(agent, "error")
        self._line(f"✗ {message}", "bold red")
