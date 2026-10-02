"""Loop limits: every run must terminate within the configured budget."""

from __future__ import annotations

import pytest

from tests.benchmark.harness import run_task, task_by_id
from tests.benchmark.reliability import run_scenario, scenario_by_id


def test_persistent_verification_failure_stops_at_retry_limit():
    record = run_scenario(scenario_by_id("D-persistent-verification-failure"))
    assert record.expected_behavior_met, record.detail
    assert record.orchestrator_status == "FAILED"
    assert record.verification_retries == 2


def test_max_steps_scenario_stops_at_budget():
    record = run_scenario(scenario_by_id("E-max-steps"))
    assert record.expected_behavior_met, record.detail
    assert record.terminated_within_limits


def _never_completes(script, ws):
    script["reasoner"] = [
        {"status": "continue", "reason": "x", "action": "filesystem.list",
         "arguments": {"path": f"{ws}/Downloads", "max_entries": n}}
        for n in range(1, 200)
    ]
    script["verifier"] = [{"success": False, "issues": ["incomplete"]}] * 10
    return script


@pytest.mark.parametrize("max_steps,retries", [(1, 0), (3, 1), (4, 2)])
def test_step_budget_is_exact(max_steps, retries):
    record = run_task(
        task_by_id("fs-find-by-extension"),
        script_override=_never_completes,
        config_overrides={"max_steps": max_steps, "max_verification_retries": retries},
    ).record
    assert record.tool_calls == max_steps * (retries + 1)
    assert record.verification_attempts == retries + 1
    assert record.orchestrator_status == "FAILED"
    assert record.terminated_within_limits


def test_consecutive_failures_stop_the_loop():
    def always_fails(script, ws):
        script["reasoner"] = [
            {"status": "continue", "reason": "x", "action": "filesystem.read",
             "arguments": {"path": f"{ws}/missing-{n}.txt"}}
            for n in range(50)
        ]
        return script

    record = run_task(task_by_id("content-read-pdf"), script_override=always_fails,
                      config_overrides={"max_consecutive_failures": 3}).record
    assert record.orchestrator_status == "FAILED"
    assert record.tool_calls == 3
    assert any("consecutive failures" in e for e in record.errors)


def test_repeat_guard_blocks_identical_loops():
    def repeats(script, ws):
        step = {"status": "continue", "reason": "x", "action": "filesystem.search",
                "arguments": {"path": f"{ws}/Downloads", "pattern": "*.pdf"}}
        script["reasoner"] = [step] * 3 + [{"status": "complete", "reason": "x", "answer": "done"}]
        return script

    result = run_task(task_by_id("fs-find-by-extension"), script_override=repeats)
    assert result.record.tool_calls == 2  # third identical call is short-circuited, not executed
    assert "already succeeded twice" in result.state.observations[2].result.error
