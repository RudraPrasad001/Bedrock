"""Benchmark metrics, computed only from execution records.

Every rate returns None (not 0) when its denominator is zero, so "no data"
is never confused with "0%". Percentiles are only reported when the sample
is large enough to mean something.
"""

from __future__ import annotations

import statistics
from typing import Iterable

MIN_P95_SAMPLES = 20


def rate(numerator: int, denominator: int) -> float | None:
    """Percentage, or None if there is nothing to divide by."""
    return None if denominator == 0 else numerator / denominator * 100


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def p95(values: Iterable[float]) -> float | None:
    """95th percentile (inclusive method); None below MIN_P95_SAMPLES samples."""
    values = sorted(values)
    if len(values) < MIN_P95_SAMPLES:
        return None
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def recovery_overhead(failure_latency: float | None, normal_latency: float | None) -> float | None:
    """(failure - normal) / normal * 100; None if either is missing or normal is not positive."""
    if failure_latency is None or normal_latency is None or normal_latency <= 0:
        return None
    return (failure_latency - normal_latency) / normal_latency * 100


def latency_stats(durations: list[float]) -> dict:
    return {
        "samples": len(durations),
        "mean_s": mean(durations),
        "median_s": median(durations),
        "p95_s": p95(durations),
        "p95_note": None if len(durations) >= MIN_P95_SAMPLES
        else f"not reported: fewer than {MIN_P95_SAMPLES} samples",
    }


# ---------------------------------------------------------------------------
# Section metrics (records are plain dicts, as stored in the JSON report)
# ---------------------------------------------------------------------------


def task_completion(records: list[dict]) -> dict:
    total = len(records)
    successful = sum(r["status"] == "success" for r in records)
    partial = sum(r["status"] == "partial" for r in records)
    return {
        "total_runs": total,
        "successful": successful,
        "partial": partial,
        "failed": total - successful - partial,
        "task_completion_rate": rate(successful, total),
        "partial_completion_rate": rate(partial, total),
        "by_category": {
            category: rate(
                sum(r["status"] == "success" for r in records if r["category"] == category),
                sum(r["category"] == category for r in records),
            )
            for category in sorted({r["category"] for r in records})
        },
    }


def loop_termination(records: list[dict]) -> dict:
    total = len(records)
    within = sum(bool(r["terminated_within_limits"]) for r in records)
    return {"runs": total, "terminated_within_limits": within, "loop_termination_rate": rate(within, total)}


def reliability(records: list[dict]) -> dict:
    recoverable = [r for r in records if r["recoverable"]]
    structured = [r for r in recoverable if r["scenario"].startswith("A")]
    recovered = sum(bool(r["recovered"]) for r in recoverable)

    overhead = {}
    for scenario in sorted({r["scenario"] for r in recoverable}):
        runs = [r for r in recoverable if r["scenario"] == scenario]
        overhead[scenario] = recovery_overhead(
            mean(r["duration_s"] for r in runs),
            mean(r["baseline_duration_s"] for r in runs if r["baseline_duration_s"] is not None),
        )
    return {
        "injected_failures": len(recoverable),
        "successful_recoveries": recovered,
        "recovery_rate": rate(recovered, len(recoverable)),
        "structured_output_recovery_rate": rate(sum(bool(r["recovered"]) for r in structured), len(structured)),
        "non_recoverable_runs": len(records) - len(recoverable),
        "non_recoverable_correct": sum(r["expected_behavior_met"] for r in records if not r["recoverable"]),
        "average_retry_count": mean(r["retry_count"] for r in records),
        "loop_termination": loop_termination(records),
        "recovery_overhead_pct": overhead,
    }


def security(records: list[dict]) -> dict:
    unauthorized = [r for r in records if not r["authorized"]]
    authorized = [r for r in records if r["authorized"]]
    blocked = sum(r["passed"] for r in unauthorized)
    falsely_rejected = sum(r["prevented"] for r in authorized)
    return {
        "unauthorized_attempts": len(unauthorized),
        "correctly_blocked": blocked,
        "security_enforcement_rate": rate(blocked, len(unauthorized)),
        "valid_operations": len(authorized),
        "falsely_rejected": falsely_rejected,
        "false_rejection_rate": rate(falsely_rejected, len(authorized)),
        "unintended_side_effects": sum(len(r["side_effects"]) for r in unauthorized),
        "secret_leaks": sum(r["leaked"] for r in records),
        "failures": [f"{r['id']}[{r['layer']}]" for r in records if not r["passed"]],
        "by_layer": {
            layer: rate(sum(r["passed"] for r in unauthorized if r["layer"] == layer),
                        sum(r["layer"] == layer for r in unauthorized))
            for layer in sorted({r["layer"] for r in unauthorized})
        },
    }


def performance(records: list[dict]) -> dict:
    with_tokens = [r["tokens"] for r in records if r.get("tokens")]
    return {
        "latency": latency_stats([r["duration_s"] for r in records]),
        "average_llm_calls": mean(r["llm_calls"] for r in records),
        "average_tool_calls": mean(r["tool_calls"] for r in records),
        "average_reasoning_steps": mean(r["reasoning_steps"] for r in records),
        "average_verification_retries": mean(r["verification_retries"] for r in records),
        "tokens": {
            # Provider-reported usage only. None means unavailable, never estimated.
            "runs_with_reported_usage": len(with_tokens),
            "runs_total": len(records),
            "average_total_tokens": mean(t["total"] for t in with_tokens),
            "lower_bound": any(t["responses_with_usage"] < t["responses_total"] for t in with_tokens),
        },
    }
