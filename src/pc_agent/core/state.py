"""Shared task state. Agents communicate exclusively through this object."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pc_agent.core.decisions import AgentDecision, Plan, ToolResult, Verification


class TaskStatus(str, Enum):
    PLANNING = "PLANNING"
    REASONING = "REASONING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    VERIFYING = "VERIFYING"
    SUMMARIZING = "SUMMARIZING"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass
class Observation:
    """The result of executing one action."""

    index: int
    decision: AgentDecision
    result: ToolResult


@dataclass
class TaskState:
    user_request: str
    task_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d-%H%M%S-")
        + uuid.uuid4().hex[:6]
    )
    dry_run: bool = False

    plan: Plan | None = None
    current_step: int = 1

    actions: list[AgentDecision] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # Reasoner working memory

    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    verifications: list[Verification] = field(default_factory=list)

    answer: str | None = None  # Reasoner's findings at completion
    summary: str | None = None  # Summarizer's final message
    status: TaskStatus = TaskStatus.PLANNING

    @property
    def completed(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def add_observation(self, decision: AgentDecision, result: ToolResult) -> Observation:
        obs = Observation(index=len(self.observations) + 1, decision=decision, result=result)
        self.observations.append(obs)
        if not result.success and result.error:
            self.errors.append(f"{decision.action}: {result.error}")
        return obs

    def record_file(self, path: str, kind: str) -> None:
        bucket = {
            "created": self.files_created,
            "modified": self.files_modified,
            "deleted": self.files_deleted,
        }[kind]
        if path not in bucket:
            bucket.append(path)

    def consecutive_failures(self) -> int:
        count = 0
        for obs in reversed(self.observations):
            if obs.result.success:
                break
            count += 1
        return count

    def last_verification(self) -> Verification | None:
        return self.verifications[-1] if self.verifications else None

    def snapshot(self) -> dict[str, Any]:
        """A compact dict of the state, used for logging."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "actions": len(self.actions),
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "errors": self.errors,
        }
