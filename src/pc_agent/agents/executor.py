"""Execution Agent: the only component that invokes tools.

It has no LLM. It validates the requested action against the registry,
assesses its risk, honours dry-run, runs the tool, redacts secrets and
records file changes in the task state.
"""

from __future__ import annotations

import time

from pc_agent.core.decisions import AgentDecision, PermissionAssessment, RiskLevel, ToolResult
from pc_agent.core.events import EventLog, NullLog
from pc_agent.core.permissions import PathPolicyError, redact
from pc_agent.core.state import TaskState
from pc_agent.tools.registry import ToolContext, ToolRegistry, ToolValidationError


class ExecutorAgent:
    name = "executor"

    def __init__(self, registry: ToolRegistry, context: ToolContext, log: EventLog | None = None):
        self.registry = registry
        self.context = context
        self.log = log or NullLog()

    @property
    def dry_run(self) -> bool:
        return self.context.config.dry_run

    def assess(self, decision: AgentDecision, log: bool = True) -> PermissionAssessment:
        """Validate the action and decide whether it may run / needs confirmation."""
        try:
            tool, args = self.registry.validate(decision.action, decision.arguments)
            assessment = tool.assess_call(args, self.context)
        except (ToolValidationError, PathPolicyError) as exc:
            assessment = PermissionAssessment(allowed=False, explanation=str(exc))
        except (OSError, ValueError) as exc:
            assessment = PermissionAssessment(allowed=False, explanation=f"{type(exc).__name__}: {exc}")
        if log:
            self.log.log(
                self.name, "assessment", tool=decision.action, arguments=decision.arguments,
                assessment=assessment.model_dump(mode="json"),
            )
        return assessment

    def run(self, decision: AgentDecision, state: TaskState) -> ToolResult:
        """Execute an already-approved action and return a structured observation."""
        try:
            tool, args = self.registry.validate(decision.action, decision.arguments)
        except ToolValidationError as exc:
            return ToolResult.fail(str(exc))

        # Re-checked here so run() is safe even if a caller skipped assess().
        assessment = self.assess(decision, log=False)
        if not assessment.allowed:
            return ToolResult.fail(f"Not allowed: {assessment.explanation}", blocked=True)

        if self.dry_run and assessment.risk != RiskLevel.READ:
            self.log.log(self.name, "dry_run", tool=decision.action, arguments=decision.arguments)
            return ToolResult.ok(
                f"DRY RUN: would execute {decision.action} with {decision.arguments}. No action was performed.",
                f"DRY RUN - would execute {decision.action}",
                dry_run=True,
            )

        self.log.log(self.name, "tool_call", tool=decision.action, arguments=decision.arguments)
        started = time.perf_counter()
        try:
            result = tool.func(args, self.context)
        except Exception as exc:  # a tool bug must never crash the agent
            result = ToolResult.fail(f"Tool crashed: {type(exc).__name__}: {exc}")
        elapsed = round(time.perf_counter() - started, 3)

        result.output = redact(result.output)
        result.error = redact(result.error)
        result.metadata["duration_s"] = elapsed
        if result.success:
            self._track_files(decision.action, result, state)
        self.log.log(
            self.name, "tool_result", tool=decision.action, success=result.success,
            summary=result.summary, error=result.error, metadata=result.metadata,
        )
        return result

    @staticmethod
    def _track_files(action: str, result: ToolResult, state: TaskState) -> None:
        meta = result.metadata
        if action == "filesystem.write":
            state.record_file(meta["path"], meta.get("change", "created"))
        elif action == "filesystem.copy":
            state.record_file(meta["destination"], "created")
        elif action == "filesystem.move":
            state.record_file(meta["source"], "deleted")
            state.record_file(meta["destination"], "created")
        elif action == "filesystem.delete":
            state.record_file(meta["path"], "deleted")
