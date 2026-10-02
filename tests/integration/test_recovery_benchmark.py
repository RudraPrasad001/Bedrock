"""Reliability: injected failures must be recovered from, measured by task outcome."""

from __future__ import annotations

import pytest

from tests.benchmark.harness import run_task, task_by_id
from tests.benchmark.reliability import SCENARIOS, run_scenario, scenario_by_id

RECOVERABLE = [s for s in SCENARIOS if s.recoverable]


@pytest.mark.parametrize("scenario", RECOVERABLE, ids=[s.id for s in RECOVERABLE])
def test_recovers_from_injected_failure(scenario):
    record = run_scenario(scenario)
    assert record.recovered, record.detail
    assert record.orchestrator_status == "COMPLETED"
    assert record.terminated_within_limits


def test_structured_output_retries_are_counted():
    record = run_scenario(scenario_by_id("A-invalid-structured-output"))
    assert record.structured_output_retries == 3  # planner, reasoner and verifier each retried once


def test_verification_retry_is_counted():
    assert run_scenario(scenario_by_id("C-verification-retry")).verification_retries == 1


def test_summarizer_outage_uses_deterministic_fallback():
    scenario = scenario_by_id("F-llm-unavailable-summarizer")
    result = scenario.run(task_by_id(scenario.task_id), 0)
    assert result.record.llm_failures >= 1
    assert result.state.summary.startswith("Task completed.")
    assert "2 PDFs" in result.state.summary  # built from the reasoner's findings, not an LLM


def test_confirmation_rejection_prevents_the_action():
    record = run_scenario(scenario_by_id("G-confirmation-rejected"))
    assert record.expected_behavior_met, record.detail
    assert record.orchestrator_status == "CANCELLED"


def test_termination_alone_is_not_counted_as_recovery():
    """Negative control: after a tool failure the agent claims success without
    doing the work. It terminates cleanly, but must not be scored as recovered."""
    task = task_by_id("content-read-pdf")

    def give_up(script, ws):
        script["reasoner"] = [
            {"status": "continue", "reason": "x", "action": "filesystem.read",
             "arguments": {"path": f"{ws}/Downloads/missing.pdf"}},
            {"status": "complete", "reason": "x", "answer": "Revenue grew 12 percent."},
        ]
        return script

    record = run_task(task, script_override=give_up).record
    assert record.terminated_within_limits and record.orchestrator_status == "COMPLETED"
    assert record.status == "failed"
