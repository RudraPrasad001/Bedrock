"""Reliability scenarios: inject failures into the real orchestrator.

Each scenario reuses a benchmark task, perturbs the scripted LLM (or wraps it
with a fault injector) and records whether the system recovered. A run only
counts as recovered when the task's own deterministic checks pass, i.e. the
intended outcome was achieved, not merely because the program terminated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from pc_agent.core.state import TaskState, TaskStatus
from pc_agent.llm.base import LLMResponse

from tests.benchmark.harness import (
    BenchmarkUI,
    FaultInjectingLLM,
    RunResult,
    run_task,
    task_by_id,
)
from tests.benchmark.validation import CheckResult
from tests.benchmark.workspace import Workspace, sha256


@dataclass
class ReliabilityRecord:
    scenario: str
    injected_failure: str
    task_id: str
    run_index: int
    recoverable: bool  # True: the intended outcome should still be achieved
    recovered: bool | None  # None for non-recoverable scenarios
    expected_behavior_met: bool  # recoverable: recovered; otherwise: correct safe termination
    orchestrator_status: str
    terminated_within_limits: bool
    retry_count: int
    structured_output_retries: int
    verification_retries: int
    duration_s: float
    baseline_duration_s: float | None
    llm_calls: int
    tool_calls: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scenario:
    id: str
    group: str  # structured_output | tool_failure | verification | limits | llm_unavailable | confirmation
    description: str
    task_id: str
    recoverable: bool
    run: Callable[[dict, int], RunResult]
    expect: Callable[[RunResult], tuple[bool, str]]


# -- helpers -----------------------------------------------------------------


def _act(action: str, **arguments) -> dict:
    return {"status": "continue", "reason": f"Use {action}", "step_id": 1, "action": action, "arguments": arguments}


def _complete(answer: str = "done") -> dict:
    return {"status": "complete", "reason": "Done.", "answer": answer}


def _outcome_achieved(result: RunResult) -> tuple[bool, str]:
    record = result.record
    ok = record.status == "success" and record.orchestrator_status == TaskStatus.COMPLETED.value
    failed = [v["detail"] for v in record.validation if not v["passed"]]
    return ok, f"status={record.orchestrator_status}, checks={'pass' if not failed else failed}"


# -- A. invalid structured output --------------------------------------------


def run_invalid_output(task: dict, i: int) -> RunResult:
    def perturb(script, ws):
        script["planner"] = ["Sure! Here's my plan: search for PDFs.", *script["planner"]]
        script["reasoner"] = [{"status": "continue", "reason": "missing the action field"}, *script["reasoner"]]
        script["verifier"] = ['{"success": tru', *script["verifier"]]
        return script

    return run_task(task, run_index=i, script_override=perturb, scenario="A-invalid-structured-output")


def run_native_tool_call(task: dict, i: int) -> RunResult:
    """The planner emits a native tool call (seen with gpt-oss on Groq); it must retry."""

    def perturb(script, ws):
        call = LLMResponse(content="", tool_call={"name": "filesystem.search",
                                                  "arguments": {"path": f"{ws}/Downloads", "pattern": "*.pdf"}})
        script["planner"] = [call, *script["planner"]]
        return script

    return run_task(task, run_index=i, script_override=perturb, scenario="A2-native-tool-call")


# -- B. tool execution failure -----------------------------------------------


def run_tool_failure(task: dict, i: int) -> RunResult:
    def perturb(script, ws):
        script["reasoner"] = [
            _act("filesystem.read", path=f"{ws}/Downloads/Report.PDF"),  # wrong case -> Not a file
            _act("filesystem.search", path=f"{ws}/Downloads", pattern="*.pdf"),
            *script["reasoner"],
        ]
        return script

    return run_task(task, run_index=i, script_override=perturb, scenario="B-tool-failure")


def expect_tool_failure(result: RunResult) -> tuple[bool, str]:
    ok, detail = _outcome_achieved(result)
    first_failed = bool(result.state and result.state.observations and not result.state.observations[0].result.success)
    return ok and first_failed, f"{detail}, first tool call failed={first_failed}"


# -- C. verification failure then retry ----------------------------------------


def run_verification_retry(task: dict, i: int) -> RunResult:
    """Same steps as the clean task, but the first report omits Kubernetes, so the
    work is the baseline plus one rejected verification and one corrective action."""

    def perturb(script, ws):
        reasoner = [dict(step) for step in script["reasoner"]]
        write = next(s for s in reasoner if s.get("action") == "filesystem.write")
        write["arguments"] = {**write["arguments"], "content": "# Notes summary\n\n## Docker\nContainers.\n"}
        script["reasoner"] = reasoner + [
            _act("filesystem.write", path=write["arguments"]["path"], append=True,
                 content="\n## Kubernetes\nOrchestrates containers (pods, deployments, services).\n"),
            _complete("added the missing Kubernetes section"),
        ]
        script["verifier"] = [
            {"success": False, "issues": ["summary.md does not cover kubernetes.md"],
             "recommended_action": "Add a Kubernetes section"},
            {"success": True, "checks": [], "issues": []},
        ]
        return script

    return run_task(task, run_index=i, script_override=perturb, scenario="C-verification-retry")


def expect_verification_retry(result: RunResult) -> tuple[bool, str]:
    ok, detail = _outcome_achieved(result)
    retried = result.record.verification_retries == 1
    return ok and retried, f"{detail}, verification_retries={result.record.verification_retries}"


# -- D. persistent verification failure ---------------------------------------


def run_persistent_verification_failure(task: dict, i: int) -> RunResult:
    def perturb(script, ws):
        reasoner = []
        for n in range(1, 4):  # distinct arguments so the repeat guard is not what stops it
            reasoner += [_act("filesystem.search", path=f"{ws}/Downloads", pattern="*.pdf", max_results=10 + n),
                         _complete()]
        script["reasoner"] = reasoner
        script["verifier"] = [{"success": False, "issues": ["always wrong"]}] * 3
        return script

    return run_task(task, run_index=i, script_override=perturb, scenario="D-persistent-verification-failure")


def expect_persistent_failure(result: RunResult) -> tuple[bool, str]:
    state = result.state
    attempts = len(state.verifications) if state else 0
    limit = 2 + 1  # Config.max_verification_retries default + the first attempt
    ok = bool(state and state.status == TaskStatus.FAILED and attempts == limit
              and result.record.terminated_within_limits)
    return ok, f"status={result.record.orchestrator_status}, verification attempts={attempts} (limit {limit})"


# -- E. maximum execution steps ----------------------------------------------

MAX_STEPS = 5


def run_max_steps(task: dict, i: int) -> RunResult:
    def perturb(script, ws):
        # An agent that never completes: endless, always-successful, never-identical actions.
        script["reasoner"] = [_act("filesystem.list", path=f"{ws}/Downloads", max_entries=n) for n in range(1, 200)]
        script["verifier"] = [{"success": False, "issues": ["not complete"]}] * 5
        return script

    return run_task(task, run_index=i, script_override=perturb,
                    config_overrides={"max_steps": MAX_STEPS}, scenario="E-max-steps")


def expect_max_steps(result: RunResult) -> tuple[bool, str]:
    state = result.state
    budget = MAX_STEPS * 3
    observations = len(state.observations) if state else -1
    ok = bool(state and state.status == TaskStatus.FAILED and observations == budget
              and any("Step limit" in e for e in state.errors))
    return ok, f"status={result.record.orchestrator_status}, tool calls={observations} (budget {budget})"


# -- F. LLM unavailable during summarization ----------------------------------


def run_summarizer_outage(task: dict, i: int) -> RunResult:
    return run_task(task, run_index=i, wrap_llm=lambda llm: FaultInjectingLLM(llm, "summarizer"),
                    scenario="F-llm-unavailable-summarizer")


def expect_summarizer_fallback(result: RunResult) -> tuple[bool, str]:
    ok, detail = _outcome_achieved(result)
    summary = result.state.summary if result.state else ""
    fallback = bool(summary) and summary.startswith("Task completed.") and result.record.llm_failures >= 1
    return ok and fallback, f"{detail}, deterministic fallback summary={fallback}"


# -- G. confirmation rejection -----------------------------------------------


def _sources_untouched(ws: Workspace, state: TaskState) -> CheckResult:
    rel = "Downloads/report.pdf"
    intact = ws.path(rel).is_file() and sha256(ws.path(rel)) == ws.manifest[rel]
    no_dirs = not any(p.is_dir() and p.name != "nested" for p in ws.path("Downloads").iterdir())
    return CheckResult("declined_action_not_executed", intact and no_dirs,
                       "report.pdf intact, no folders created" if intact and no_dirs else "side effects found")


def run_confirmation_rejected(task: dict, i: int) -> RunResult:
    return run_task(task, run_index=i, ui=BenchmarkUI(decline_all=True),
                    extra_checks=[_sources_untouched], scenario="G-confirmation-rejected")


def expect_cancelled(result: RunResult) -> tuple[bool, str]:
    state = result.state
    untouched = any(v["check"] == "declined_action_not_executed" and v["passed"] for v in result.record.validation)
    ok = bool(state and state.status == TaskStatus.CANCELLED and untouched)
    return ok, f"status={result.record.orchestrator_status}, untouched={untouched}"


SCENARIOS = [
    Scenario("A-invalid-structured-output", "structured_output",
             "Planner prose, reasoner schema violation and truncated verifier JSON",
             "fs-find-by-extension", True, run_invalid_output, _outcome_achieved),
    Scenario("A2-native-tool-call", "structured_output",
             "Planner answers with a native tool call instead of JSON",
             "fs-find-by-extension", True, run_native_tool_call, _outcome_achieved),
    Scenario("B-tool-failure", "tool_failure",
             "First tool call fails (wrong path); reasoner must adapt",
             "content-read-pdf", True, run_tool_failure, expect_tool_failure),
    Scenario("C-verification-retry", "verification",
             "Verifier rejects an incomplete report once; agent must fix it",
             "gen-markdown-summary", True, run_verification_retry, expect_verification_retry),
    Scenario("F-llm-unavailable-summarizer", "llm_unavailable",
             "Summarizer LLM call raises; deterministic summary must be used",
             "fs-find-by-extension", True, run_summarizer_outage, expect_summarizer_fallback),
    Scenario("D-persistent-verification-failure", "limits",
             "Verifier always rejects; must stop after max_verification_retries",
             "fs-find-by-extension", False, run_persistent_verification_failure, expect_persistent_failure),
    Scenario("E-max-steps", "limits",
             f"Agent never completes; must stop at max_steps={MAX_STEPS} per attempt",
             "fs-find-by-extension", False, run_max_steps, expect_max_steps),
    Scenario("G-confirmation-rejected", "confirmation",
             "User declines a destructive move; nothing may be moved",
             "org-by-extension", False, run_confirmation_rejected, expect_cancelled),
]


def run_scenario(scenario: Scenario, run_index: int = 0, baseline_s: float | None = None) -> ReliabilityRecord:
    task = task_by_id(scenario.task_id)
    result = scenario.run(task, run_index)
    met, detail = scenario.expect(result)
    record = result.record
    return ReliabilityRecord(
        scenario=scenario.id,
        injected_failure=scenario.description,
        task_id=scenario.task_id,
        run_index=run_index,
        recoverable=scenario.recoverable,
        recovered=met if scenario.recoverable else None,
        expected_behavior_met=met,
        orchestrator_status=record.orchestrator_status,
        terminated_within_limits=record.terminated_within_limits,
        retry_count=record.retry_count,
        structured_output_retries=record.structured_output_retries,
        verification_retries=record.verification_retries,
        duration_s=record.duration_s,
        baseline_duration_s=baseline_s,
        llm_calls=record.llm_calls,
        tool_calls=record.tool_calls,
        detail=detail,
    )


def scenario_by_id(scenario_id: str) -> Scenario:
    return next(s for s in SCENARIOS if s.id == scenario_id)


def run_reliability(runs: int = 1) -> list[ReliabilityRecord]:
    """Run every scenario `runs` times, each paired with clean baselines of its task."""
    records = []
    for scenario in SCENARIOS:
        task = task_by_id(scenario.task_id)
        baselines = [run_task(task, run_index=i).record.duration_s for i in range(runs)]
        baseline = sum(baselines) / len(baselines)
        for i in range(runs):
            records.append(run_scenario(scenario, i, baseline_s=baseline))
    return records

