"""Metrics checked against small, hand-computed datasets."""

from __future__ import annotations

import pytest

from tests.benchmark import metrics


def test_rate_and_division_by_zero():
    assert metrics.rate(3, 4) == 75.0
    assert metrics.rate(0, 5) == 0.0
    assert metrics.rate(0, 0) is None  # no data is not 0%


def test_latency_statistics():
    assert metrics.mean([1, 2, 3, 4]) == 2.5
    assert metrics.median([5, 1, 3]) == 3
    assert metrics.mean([]) is None and metrics.median([]) is None
    assert metrics.p95(list(range(10))) is None  # too few samples
    assert metrics.p95([float(i) for i in range(1, 101)]) == pytest.approx(95.05)
    stats = metrics.latency_stats([1.0, 2.0])
    assert stats["p95_s"] is None and "fewer than" in stats["p95_note"]


def test_recovery_overhead():
    assert metrics.recovery_overhead(1.5, 1.0) == pytest.approx(50.0)
    assert metrics.recovery_overhead(0.8, 1.0) == pytest.approx(-20.0)
    assert metrics.recovery_overhead(1.0, 0.0) is None
    assert metrics.recovery_overhead(None, 1.0) is None


def _task(status, category="filesystem", duration=1.0, tokens=None, within=True, **kw):
    return {"status": status, "category": category, "duration_s": duration, "llm_calls": 4, "tool_calls": 2,
            "reasoning_steps": 3, "verification_retries": 0, "tokens": tokens, "terminated_within_limits": within, **kw}


def test_task_completion():
    records = [_task("success"), _task("success"), _task("partial"), _task("failed", category="system")]
    tc = metrics.task_completion(records)
    assert (tc["total_runs"], tc["successful"], tc["partial"], tc["failed"]) == (4, 2, 1, 1)
    assert tc["task_completion_rate"] == 50.0
    assert tc["partial_completion_rate"] == 25.0
    assert tc["by_category"] == {"filesystem": pytest.approx(200 / 3), "system": 0.0}
    assert metrics.task_completion([])["task_completion_rate"] is None


def test_loop_termination():
    lt = metrics.loop_termination([_task("success"), _task("failed", within=False)])
    assert lt["loop_termination_rate"] == 50.0


def _rel(scenario, recoverable, met, duration, baseline, retries=1):
    return {"scenario": scenario, "recoverable": recoverable, "recovered": met if recoverable else None,
            "expected_behavior_met": met, "duration_s": duration, "baseline_duration_s": baseline,
            "retry_count": retries, "terminated_within_limits": True}


def test_reliability():
    records = [
        _rel("A-x", True, True, 2.0, 1.0, retries=2),
        _rel("A-x", True, False, 4.0, 1.0, retries=2),
        _rel("B-y", True, True, 1.5, 1.0, retries=0),
        _rel("D-z", False, True, 1.0, None, retries=2),
    ]
    rel = metrics.reliability(records)
    assert rel["injected_failures"] == 3
    assert rel["successful_recoveries"] == 2
    assert rel["recovery_rate"] == pytest.approx(200 / 3)
    assert rel["structured_output_recovery_rate"] == 50.0  # only A-* scenarios
    assert rel["non_recoverable_runs"] == 1 and rel["non_recoverable_correct"] == 1
    assert rel["average_retry_count"] == 1.5
    assert rel["recovery_overhead_pct"]["A-x"] == pytest.approx(200.0)  # mean 3.0 vs 1.0
    assert rel["recovery_overhead_pct"]["B-y"] == pytest.approx(50.0)


def _sec(authorized, prevented, passed, side_effects=(), leaked=False, layer="executor"):
    return {"id": "x", "layer": layer, "authorized": authorized, "prevented": prevented, "passed": passed,
            "side_effects": list(side_effects), "leaked": leaked}


def test_security():
    records = [
        _sec(False, True, True),
        _sec(False, True, True, layer="direct"),
        _sec(False, True, False, side_effects=["a.txt"]),  # prevented but changed files -> not blocked
        _sec(False, False, False, leaked=True),
        _sec(True, False, True),
        _sec(True, True, False),  # valid operation wrongly blocked
    ]
    sec = metrics.security(records)
    assert sec["unauthorized_attempts"] == 4 and sec["correctly_blocked"] == 2
    assert sec["security_enforcement_rate"] == 50.0
    assert sec["valid_operations"] == 2 and sec["falsely_rejected"] == 1
    assert sec["false_rejection_rate"] == 50.0
    assert sec["unintended_side_effects"] == 1
    assert sec["secret_leaks"] == 1
    assert sec["by_layer"] == {"direct": 100.0, "executor": pytest.approx(100 / 3)}


def test_performance_distinguishes_unavailable_tokens():
    none = metrics.performance([_task("success"), _task("success")])
    assert none["tokens"]["average_total_tokens"] is None
    assert none["tokens"]["runs_with_reported_usage"] == 0

    tok = {"total": 100, "responses_with_usage": 3, "responses_total": 4}
    some = metrics.performance([_task("success", tokens=tok), _task("success", duration=3.0)])
    assert some["tokens"]["average_total_tokens"] == 100  # averaged over runs that reported usage
    assert some["tokens"]["lower_bound"] is True
    assert some["latency"]["mean_s"] == 2.0
