"""The agent loop: a small, explicit state machine.

    PLANNING -> (REASONING -> EXECUTING -> OBSERVING)* -> VERIFYING
             -> retry if verification failed -> SUMMARIZING -> done
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Literal, Protocol

from pc_agent.agents.base import AgentError
from pc_agent.agents.executor import ExecutorAgent
from pc_agent.agents.planner import PlannerAgent
from pc_agent.agents.reasoner import ReasonerAgent
from pc_agent.agents.summarizer import SummarizerAgent
from pc_agent.agents.verifier import VerifierAgent
from pc_agent.config import Config
from pc_agent.core.decisions import (
    AgentDecision,
    PermissionAssessment,
    Plan,
    ReasonerOutput,
    RiskLevel,
    ToolResult,
    Verification,
)
from pc_agent.core.events import EventLog, NullLog
from pc_agent.core.permissions import PathPolicy
from pc_agent.core.state import TaskState, TaskStatus
from pc_agent.llm.base import LLMClient, LLMError
from pc_agent.tools.registry import ToolContext, ToolRegistry, build_registry

ConfirmAnswer = Literal["yes", "no", "all"]


class AgentUI(Protocol):
    """Everything the orchestrator needs from a user interface."""

    def thinking(self, agent: str, message: str) -> AbstractContextManager: ...
    def show_plan(self, plan: Plan) -> None: ...
    def show_reasoning(self, output: ReasonerOutput) -> None: ...
    def show_tool_call(self, decision: AgentDecision, assessment: PermissionAssessment) -> None: ...
    def show_tool_result(self, decision: AgentDecision, result: ToolResult) -> None: ...
    def confirm(self, decision: AgentDecision, assessment: PermissionAssessment, allow_all: bool) -> ConfirmAnswer: ...
    def show_verification(self, verification: Verification, attempt: int) -> None: ...
    def show_summary(self, summary: str, state: TaskState) -> None: ...
    def show_error(self, agent: str, message: str) -> None: ...


class SilentUI:
    """A UI that shows nothing; confirmations get a fixed answer. Used in tests."""

    def __init__(self, confirm_answer: ConfirmAnswer = "no"):
        self.confirm_answer = confirm_answer
        self.confirmations: list[AgentDecision] = []

    def thinking(self, agent, message):
        return nullcontext()

    def confirm(self, decision, assessment, allow_all):
        self.confirmations.append(decision)
        return self.confirm_answer

    def __getattr__(self, name):  # show_* methods are no-ops
        if name.startswith("show_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


# Tools whose confirmations may be granted for the rest of a task with one answer.
BATCH_APPROVABLE = {"filesystem.move", "filesystem.write"}


class Orchestrator:
    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        ui: AgentUI,
        registry: ToolRegistry | None = None,
    ):
        self.config = config
        self.llm = llm
        self.ui = ui
        self.registry = registry or build_registry(config)
        self.context = ToolContext(config=config, path_policy=PathPolicy(config.allowed_paths))
        self.log: EventLog = NullLog()

    def _build_agents(self) -> None:
        self.planner = PlannerAgent(self.llm, self.registry.catalog(detailed=False), self.log)
        self.reasoner = ReasonerAgent(self.llm, self.registry.catalog(), self.config.max_consecutive_failures, self.log)
        self.executor = ExecutorAgent(self.registry, self.context, self.log)
        self.verifier = VerifierAgent(self.llm, self.registry.read_only(), self.context, self.log)
        self.summarizer = SummarizerAgent(self.llm, self.log)

    # ------------------------------------------------------------------

    def run(self, request: str, log: EventLog | None = None) -> TaskState:
        state = TaskState(user_request=request, dry_run=self.config.dry_run)
        self.log = log if log is not None else EventLog.for_task(self.config.log_dir, state.task_id)
        self._build_agents()
        self.log.log("orchestrator", "task_start", request=request, dry_run=state.dry_run,
                     model=self.config.model, tools=self.registry.enabled_names())
        self._approved_tools: set[str] = set()

        try:
            self._plan(state)
            verified = False
            for attempt in range(1, self.config.max_verification_retries + 2):
                self._act(state)
                if state.status in (TaskStatus.CANCELLED, TaskStatus.FAILED):
                    break
                verified = self._verify(state, attempt)
                if verified:
                    break
            if state.status not in (TaskStatus.CANCELLED, TaskStatus.FAILED):
                state.status = TaskStatus.COMPLETED if verified else TaskStatus.FAILED
        except (LLMError, AgentError) as exc:
            state.status = TaskStatus.FAILED
            state.errors.append(str(exc))
            self.ui.show_error("orchestrator", str(exc))
            self.log.log("orchestrator", "error", error=str(exc))
        except KeyboardInterrupt:
            state.status = TaskStatus.CANCELLED
            state.errors.append("Interrupted by user")

        self._summarize(state)
        self.log.log("orchestrator", "task_end", state=state.snapshot())
        return state

    # ------------------------------------------------------------------

    def _plan(self, state: TaskState) -> None:
        state.status = TaskStatus.PLANNING
        with self.ui.thinking("planner", "Understanding task..."):
            state.plan = self.planner.run(state)
        self.ui.show_plan(state.plan)

    def _act(self, state: TaskState) -> None:
        """Reason -> execute -> observe until the reasoner says the work is done."""
        for _ in range(self.config.max_steps):
            state.status = TaskStatus.REASONING
            with self.ui.thinking("reasoner", "Determining next action..."):
                output = self.reasoner.run(state)
            self.ui.show_reasoning(output)

            if output.status == "complete":
                state.answer = output.answer or output.reason
                return
            if output.status == "failed":
                state.errors.append(f"Reasoner gave up: {output.reason}")
                state.status = TaskStatus.FAILED
                return

            decision = self.reasoner.to_decision(output)
            state.actions.append(decision)
            result = self._execute(state, decision)
            if result is None:  # user declined
                return

            state.status = TaskStatus.OBSERVING
            state.add_observation(decision, result)
            self.ui.show_tool_result(decision, result)
            if not result.success and not self.reasoner.handle_failure(state):
                state.errors.append("Too many consecutive failures; stopping.")
                state.status = TaskStatus.FAILED
                return

        state.errors.append(f"Step limit ({self.config.max_steps}) reached before the task was complete")
        self.log.log("orchestrator", "step_limit", steps=self.config.max_steps)

    def _execute(self, state: TaskState, decision: AgentDecision) -> ToolResult | None:
        state.status = TaskStatus.EXECUTING
        if self.reasoner.is_repeat(state, decision):
            assessment = PermissionAssessment(allowed=False, explanation="repeated action")
            self.ui.show_tool_call(decision, assessment)
            return ToolResult.fail(
                "This exact action already succeeded twice; its result is in the observations. "
                "Use it, choose a different action, or complete the task.",
                blocked=True,
            )

        assessment = self.executor.assess(decision)
        decision.requires_confirmation = assessment.requires_confirmation
        self.ui.show_tool_call(decision, assessment)
        if not assessment.allowed:
            return ToolResult.fail(f"Not allowed: {assessment.explanation}", blocked=True)

        needs_confirmation = (
            assessment.requires_confirmation
            and not state.dry_run  # nothing with side effects runs in dry-run
            and decision.action not in self._approved_tools
        )
        if needs_confirmation:
            state.status = TaskStatus.WAITING_FOR_CONFIRMATION
            allow_all = decision.action in BATCH_APPROVABLE and assessment.risk != RiskLevel.PRIVILEGED
            answer = self.ui.confirm(decision, assessment, allow_all)
            self.log.log("orchestrator", "confirmation", tool=decision.action,
                         arguments=decision.arguments, answer=answer)
            if answer == "no":
                state.add_observation(decision, ToolResult.fail("User declined the action"))
                state.status = TaskStatus.CANCELLED
                return None
            if answer == "all" and allow_all:
                self._approved_tools.add(decision.action)
            state.status = TaskStatus.EXECUTING

        with self.ui.thinking("executor", f"Running {decision.action}..."):
            return self.executor.run(decision, state)

    def _verify(self, state: TaskState, attempt: int) -> bool:
        state.status = TaskStatus.VERIFYING
        with self.ui.thinking("verifier", "Checking result..."):
            verification = self.verifier.run(state)
        state.verifications.append(verification)
        self.ui.show_verification(verification, attempt)
        return verification.success

    def _summarize(self, state: TaskState) -> None:
        # The status already holds the final outcome (COMPLETED/FAILED/CANCELLED),
        # which is exactly what the summarizer needs to report.
        with self.ui.thinking("summarizer", "Writing summary..."):
            state.summary = self.summarizer.run(state)
        self.log.log("summarizer", "summary", summary=state.summary)
        self.ui.show_summary(state.summary, state)
