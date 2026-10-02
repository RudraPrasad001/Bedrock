"""Structured messages exchanged between agents.

Every agent output is validated against one of these Pydantic models, so the
orchestrator never has to parse free-form natural language.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RiskLevel(str, Enum):
    """How much damage a tool call can do."""

    READ = "read"  # no side effects
    WRITE = "write"  # creates or changes files
    DESTRUCTIVE = "destructive"  # removes/moves data, can't easily be undone
    PRIVILEGED = "privileged"  # arbitrary command / code execution


class PlanStep(BaseModel):
    id: int
    description: str
    expected_tool: str | None = None


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)


class ReasonerOutput(BaseModel):
    """Raw output of the Reasoning Agent, as produced by the LLM."""

    status: Literal["continue", "complete", "failed"] = "continue"
    reason: str = Field(description="Short, user-visible reasoning summary.")
    step_id: int | None = None
    action: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    note: str | None = Field(
        default=None,
        description="Facts from the latest observation worth remembering.",
    )
    answer: str | None = Field(
        default=None, description="Findings/answer when status is complete."
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def _null_arguments(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _action_required(self) -> "ReasonerOutput":
        if self.status == "continue" and not self.action:
            raise ValueError("'action' is required when status is 'continue'")
        return self


class AgentDecision(BaseModel):
    """An action request handed from the Reasoner to the Executor."""

    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    step_id: int | None = None
    requires_confirmation: bool = False


class ToolResult(BaseModel):
    success: bool
    output: str | None = None
    error: str | None = None
    summary: str | None = None  # one-line, human-readable
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, output: str, summary: str, **metadata: Any) -> "ToolResult":
        return cls(success=True, output=output, summary=summary, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> "ToolResult":
        return cls(success=False, error=error, summary=error, metadata=metadata)


class PermissionAssessment(BaseModel):
    """The Executor's verdict on whether an action may run."""

    allowed: bool
    risk: RiskLevel = RiskLevel.READ
    requires_confirmation: bool = False
    explanation: str = ""


class VerificationCheck(BaseModel):
    check: str
    passed: bool


class Verification(BaseModel):
    success: bool
    checks: list[VerificationCheck] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    recommended_action: str | None = None
