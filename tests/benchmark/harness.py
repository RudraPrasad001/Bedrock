"""Benchmark execution harness.

Runs a task through the real Orchestrator in a fresh, isolated workspace and
turns the outcome into an ExecutionRecord. Measurements come from three
sources, none of which require changes to production code:

- MeteredLLM wraps the LLM client and counts every request (including failed ones).
- The orchestrator's JSONL EventLog supplies tool calls, reasoning steps,
  structured-output retries and provider-reported token usage.
- TaskState supplies verification attempts, status and file changes.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from pc_agent.config import Config
from pc_agent.core.events import EventLog
from pc_agent.core.orchestrator import Orchestrator, SilentUI
from pc_agent.core.permissions import redact
from pc_agent.core.state import TaskState
from pc_agent.llm.base import LLMClient, LLMError, LLMResponse

from tests.benchmark.validation import CheckResult, validate
from tests.benchmark.workspace import Workspace, build_workspace
from tests.conftest import AGENT_MARKERS, ScriptedLLM

TASKS_FILE = Path(__file__).parent / "tasks.json"

DEFAULT_PLAN = {"goal": "{instruction}", "steps": [{"id": 1, "description": "{instruction}"}], "risks": []}
DEFAULT_VERDICT = {"success": True, "checks": [], "issues": []}
DEFAULT_SUMMARY = "Task completed."


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def load_tasks(categories: list[str] | None = None) -> list[dict]:
    tasks = json.loads(TASKS_FILE.read_text())["tasks"]
    if categories:
        tasks = [t for t in tasks if t["category"] in categories]
    return tasks


def task_by_id(task_id: str) -> dict:
    return next(t for t in load_tasks() if t["id"] == task_id)


def substitute(node: Any, ws: Path) -> Any:
    """Replace '{ws}' in every string of a JSON-like structure."""
    if isinstance(node, str):
        return node.replace("{ws}", str(ws))
    if isinstance(node, list):
        return [substitute(n, ws) for n in node]
    if isinstance(node, dict):
        return {k: substitute(v, ws) for k, v in node.items()}
    return node


def scripted_replies(task: dict, ws: Path) -> dict[str, list]:
    """Full per-agent script for a task, with defaults for non-reasoner agents."""
    script = substitute(task.get("script", {}), ws)
    instruction = substitute(task["instruction"], ws)
    plan = json.loads(json.dumps(DEFAULT_PLAN).replace("{instruction}", instruction.replace('"', "'")))
    return {
        "planner": script.get("planner", [plan]),
        "reasoner": script["reasoner"],
        "verifier": script.get("verifier", [DEFAULT_VERDICT] * 4),
        "summarizer": script.get("summarizer", [DEFAULT_SUMMARY] * 2),
    }


# ---------------------------------------------------------------------------
# LLM wrappers (benchmark-only)
# ---------------------------------------------------------------------------


def agent_of(system_prompt: str) -> str:
    return next((a for marker, a in AGENT_MARKERS.items() if f"You are the {marker}" in system_prompt), "unknown")


class MeteredLLM(LLMClient):
    """Counts every request made to the wrapped client, including failures."""

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.model = inner.model
        self.calls = 0
        self.failures = 0
        self.calls_by_agent: dict[str, int] = {}

    def generate(self, system_prompt, messages, tools=None, json_mode=False) -> LLMResponse:
        self.calls += 1
        agent = agent_of(system_prompt)
        self.calls_by_agent[agent] = self.calls_by_agent.get(agent, 0) + 1
        try:
            return self.inner.generate(system_prompt, messages, tools=tools, json_mode=json_mode)
        except Exception:
            self.failures += 1
            raise


class FaultInjectingLLM(LLMClient):
    """Raises LLMError for the first `times` calls made by `agent` (simulates an outage)."""

    def __init__(self, inner: LLMClient, agent: str, times: int = 1_000_000):
        self.inner = inner
        self.model = inner.model
        self.agent = agent
        self.remaining = times
        self.injected = 0

    def generate(self, system_prompt, messages, tools=None, json_mode=False) -> LLMResponse:
        if agent_of(system_prompt) == self.agent and self.remaining > 0:
            self.remaining -= 1
            self.injected += 1
            raise LLMError(f"Injected fault: {self.agent} LLM unavailable")
        return self.inner.generate(system_prompt, messages, tools=tools, json_mode=json_mode)


class BenchmarkUI(SilentUI):
    """Confirmation policy for unattended runs.

    Filesystem tools are path-sandboxed to the temporary workspace, so their
    confirmations are approved. shell.run / python.execute are NOT sandboxed by
    path, so they are always declined - even in live mode a model can never run
    an unconfirmed privileged command against the real machine.
    """

    def __init__(self, decline_all: bool = False):
        super().__init__()
        self.decline_all = decline_all

    def confirm(self, decision, assessment, allow_all):
        self.confirmations.append(decision)
        if self.decline_all or not decision.action.startswith("filesystem."):
            return "no"
        return "yes"


# ---------------------------------------------------------------------------
# Execution records
# ---------------------------------------------------------------------------


@dataclass
class ExecutionRecord:
    task_id: str
    category: str
    mode: str
    run_index: int
    status: str  # success | partial | failed
    orchestrator_status: str
    expected: str
    actual: dict
    validation: list[dict]
    duration_s: float
    llm_calls: int
    llm_failures: int
    tool_calls: int
    reasoning_steps: int
    verification_attempts: int
    verification_retries: int
    structured_output_retries: int
    retry_count: int
    tokens: dict | None  # None = provider reported no usage (NOT zero)
    terminated_within_limits: bool
    errors: list[str] = field(default_factory=list)
    scenario: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_events(log_path: Path) -> dict:
    events = []
    if log_path.exists():
        events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    usage = [e["usage"] for e in events if e["event"] == "llm_response" and e.get("usage")]
    llm_responses = sum(1 for e in events if e["event"] == "llm_response")
    tokens = None
    if usage:
        prompt = sum(u.get("prompt_tokens", 0) for u in usage)
        completion = sum(u.get("completion_tokens", 0) for u in usage)
        tokens = {
            "prompt": prompt,
            "completion": completion,
            "total": prompt + completion,
            # Some responses (e.g. recovered tool calls) carry no usage; the total is then a lower bound.
            "responses_with_usage": len(usage),
            "responses_total": llm_responses,
        }
    return {
        "tool_calls": sum(1 for e in events if e["event"] == "tool_call"),
        "reasoning_steps": sum(1 for e in events if e["agent"] == "reasoner" and e["event"] == "decision"),
        "invalid_outputs": sum(1 for e in events if e["event"] == "invalid_output"),
        "tokens": tokens,
    }


def classify(results: list[CheckResult]) -> str:
    if results and all(r.passed for r in results):
        return "success"
    if any(r.passed for r in results):
        return "partial"
    return "failed"


@dataclass
class RunResult:
    record: ExecutionRecord
    state: TaskState | None
    extra: dict = field(default_factory=dict)


ExtraCheck = Callable[[Workspace, TaskState], CheckResult]


def run_task(
    task: dict,
    mode: str = "scripted",
    run_index: int = 0,
    *,
    llm: LLMClient | None = None,
    script_override: Callable[[dict[str, list], Path], dict[str, list]] | None = None,
    wrap_llm: Callable[[LLMClient], LLMClient] | None = None,
    ui: SilentUI | None = None,
    config_overrides: dict | None = None,
    extra_checks: list[ExtraCheck] | None = None,
    scenario: str | None = None,
) -> RunResult:
    """Execute one task in a fresh temporary workspace and validate the outcome."""
    with tempfile.TemporaryDirectory(prefix="bedrock-bench-") as tmp:
        tmp_path = Path(tmp)
        ws = build_workspace(tmp_path / "workspace")
        log_path = tmp_path / "logs" / "events.jsonl"

        if mode == "scripted":
            script = scripted_replies(task, ws.root)
            if script_override:
                script = script_override(script, ws.root)
            base_llm: LLMClient = ScriptedLLM(script)
        else:
            if llm is None:
                raise ValueError("live mode requires an LLM client")
            base_llm = llm
        if wrap_llm:
            base_llm = wrap_llm(base_llm)
        metered = MeteredLLM(base_llm)

        config = Config(
            model=getattr(base_llm, "model", "") or "scripted",
            allowed_paths=[ws.root],  # the agent can only touch the temporary workspace
            log_dir=tmp_path / "logs",
        )
        for key, value in (config_overrides or {}).items():
            setattr(config, key, value)

        instruction = substitute(task["instruction"], ws.root)
        orchestrator = Orchestrator(config, metered, ui or BenchmarkUI())
        errors: list[str] = []
        state: TaskState | None = None
        started = time.perf_counter()
        try:
            state = orchestrator.run(instruction, log=EventLog(log_path))
        except Exception as exc:  # harness-level failure, e.g. exhausted script
            errors.append(f"{type(exc).__name__}: {exc}")
        duration = time.perf_counter() - started

        results: list[CheckResult] = []
        if state is not None:
            results = validate(task.get("checks", []), mode, ws, state)
            for check in extra_checks or []:
                try:
                    results.append(check(ws, state))
                except Exception as exc:
                    results.append(CheckResult(getattr(check, "__name__", "extra"), False, f"check error: {exc}"))
            errors.extend(state.errors)

        events = parse_events(log_path)
        verification_attempts = len(state.verifications) if state else 0
        step_budget = config.max_steps * (config.max_verification_retries + 1)
        observations = len(state.observations) if state else 0
        record = ExecutionRecord(
            task_id=task["id"],
            category=task["category"],
            mode=mode,
            run_index=run_index,
            status=classify(results),
            orchestrator_status=state.status.value if state else "HARNESS_ERROR",
            expected=task.get("expected", ""),
            actual={
                "answer": redact((state.answer or "")[:500]) if state else None,
                "files_created": [_rel(p, ws) for p in state.files_created] if state else [],
                "files_modified": [_rel(p, ws) for p in state.files_modified] if state else [],
                "files_deleted": [_rel(p, ws) for p in state.files_deleted] if state else [],
                "actions": [
                    {"tool": o.decision.action, "success": o.result.success, "summary": _rel_text(o.result.summary, ws)}
                    for o in (state.observations if state else [])
                ],
            },
            validation=[{"check": r.check, "passed": r.passed, "detail": _rel_text(r.detail, ws)} for r in results],
            duration_s=round(duration, 6),
            llm_calls=metered.calls,
            llm_failures=metered.failures,
            tool_calls=events["tool_calls"],
            reasoning_steps=events["reasoning_steps"],
            verification_attempts=verification_attempts,
            verification_retries=max(verification_attempts - 1, 0),
            structured_output_retries=events["invalid_outputs"],
            retry_count=events["invalid_outputs"] + max(verification_attempts - 1, 0),
            tokens=events["tokens"],
            terminated_within_limits=bool(state and state.completed and observations <= step_budget),
            errors=[_rel_text(redact(e), ws) for e in errors],
            scenario=scenario,
        )
        return RunResult(record=record, state=state, extra={"workspace": str(ws.root)})


def _rel(path: str, ws: Workspace) -> str:
    return path[len(str(ws.root)) + 1 :] if path.startswith(str(ws.root) + "/") else path


def _rel_text(text: str | None, ws: Workspace) -> str:
    """Strip the temporary workspace prefix so reports are stable across runs."""
    return (text or "").replace(str(ws.root) + "/", "").replace(str(ws.root), "<ws>")
