"""Benchmark reports: a JSON file for machines and a plain-text summary for people."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pc_agent import __version__
from pc_agent.core.permissions import redact, secret_values

from tests.benchmark import metrics

WIDTH = 60


class ReportLeakError(RuntimeError):
    pass


def environment_metadata() -> dict:
    """Non-sensitive description of where the benchmark ran. No environment variables."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=5, cwd=Path(__file__).parent).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "bedrock_version": __version__,
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def build_report(config: dict, tasks: list[dict], reliability: list[dict], security: list[dict]) -> dict:
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "environment": environment_metadata(),
        "metrics": {},
        "records": {"tasks": tasks, "reliability": reliability, "security": security},
    }
    if tasks:
        report["metrics"]["task_completion"] = metrics.task_completion(tasks)
        report["metrics"]["loop_termination"] = metrics.loop_termination(tasks)
        report["metrics"]["performance"] = metrics.performance(tasks)
    if reliability:
        report["metrics"]["reliability"] = metrics.reliability(reliability)
    if security:
        report["metrics"]["security"] = metrics.security(security)
    return report


def to_json(report: dict, extra_secrets: list[str] | None = None) -> str:
    """Serialise and refuse to emit anything containing a known secret value."""
    text = redact(json.dumps(report, indent=2, default=str)) or ""
    for secret in [*secret_values(), *(extra_secrets or [])]:
        if secret and secret in text:
            raise ReportLeakError("a secret value would have been written to the benchmark report")
    return text


def write_json(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.json"
    path.write_text(to_json(report) + "\n")
    return path


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _num(value: float | None, unit: str = "", digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def _seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 1000:.1f}ms" if value < 1 else f"{value:.2f}s"


def _row(label: str, value) -> str:
    return f"{label + ':':<44}{value}"


def _section(title: str) -> list[str]:
    return ["", title, "-" * len(title)]


def format_terminal(report: dict) -> str:
    cfg, m = report["config"], report["metrics"]
    lines = ["=" * WIDTH, "BEDROCK BENCHMARK REPORT".center(WIDTH), "=" * WIDTH]
    if cfg["mode"] == "live":
        lines.append(f"LLM: LIVE ({cfg['model']}) - results depend on the model and vary between runs")
    else:
        lines.append("LLM: SCRIPTED - agent decisions are predetermined; this measures tools,")
        lines.append("orchestration and safety, not model reasoning quality")
    lines.append(f"Runs per task: {cfg['runs']}   Categories: {', '.join(cfg['categories']) or 'all'}")

    if "task_completion" in m:
        tc = m["task_completion"]
        lines += _section("TASK COMPLETION")
        lines += [
            _row("Total task runs", tc["total_runs"]),
            _row("Successful", tc["successful"]),
            _row("Partial", tc["partial"]),
            _row("Failed", tc["failed"]),
            _row("Task completion rate", _pct(tc["task_completion_rate"])),
            _row("Partial completion rate", _pct(tc["partial_completion_rate"])),
        ]
        lines += [_row(f"  {cat}", _pct(value)) for cat, value in tc["by_category"].items()]
        failed = [r for r in report["records"]["tasks"] if r["status"] != "success"]
        for r in failed:
            reasons = "; ".join(v["detail"] for v in r["validation"] if not v["passed"]) or "; ".join(r["errors"])
            lines.append(f"  ! {r['task_id']} (run {r['run_index']}): {r['status']} - {reasons[:120]}")

    if "reliability" in m:
        rel = m["reliability"]
        lines += _section("RELIABILITY")
        lines += [
            _row("Injected recoverable failures", rel["injected_failures"]),
            _row("Successful recoveries", rel["successful_recoveries"]),
            _row("Recovery rate", _pct(rel["recovery_rate"])),
            _row("Structured output recovery rate", _pct(rel["structured_output_recovery_rate"])),
            _row("Limit/cancel scenarios correct",
                 f"{rel['non_recoverable_correct']}/{rel['non_recoverable_runs']}"),
            _row("Loop termination rate", _pct(rel["loop_termination"]["loop_termination_rate"])),
            _row("Average retry count", _num(rel["average_retry_count"])),
        ]
        lines.append("  Recovery overhead vs. clean baseline of the same task:")
        for scenario, value in rel["recovery_overhead_pct"].items():
            lines.append(_row(f"    {scenario}", _pct(value)))
        if cfg["mode"] == "scripted":
            lines.append("  (scripted runs take milliseconds, so overhead reflects local processing only)")
        for r in report["records"]["reliability"]:
            if not r["expected_behavior_met"]:
                lines.append(f"  ! {r['scenario']} (run {r['run_index']}): {r['detail'][:120]}")

    if "security" in m:
        sec = m["security"]
        lines += _section("SECURITY")
        lines += [
            _row("Unauthorized attempts", sec["unauthorized_attempts"]),
            _row("Correctly blocked", sec["correctly_blocked"]),
            _row("Security enforcement rate", _pct(sec["security_enforcement_rate"])),
            _row("Unintended side effects", sec["unintended_side_effects"]),
            _row("Secret leaks", sec["secret_leaks"]),
            _row("Valid operations", sec["valid_operations"]),
            _row("Falsely rejected", sec["falsely_rejected"]),
            _row("False rejection rate", _pct(sec["false_rejection_rate"])),
        ]
        lines += [_row(f"  {layer} layer enforcement", _pct(v)) for layer, v in sec["by_layer"].items()]
        lines += [f"  ! failed: {f}" for f in sec["failures"]]
        lines.append("  (Finite adversarial suite: passing it is evidence, not proof, of security.)")

    if "performance" in m:
        perf, lat = m["performance"], m["performance"]["latency"]
        tokens = perf["tokens"]
        lines += _section("PERFORMANCE (task runs)")
        lines += [
            _row("Average latency", _seconds(lat["mean_s"])),
            _row("Median latency", _seconds(lat["median_s"])),
            _row("P95 latency", _seconds(lat["p95_s"]) if lat["p95_s"] is not None else f"n/a ({lat['samples']} samples)"),
            _row("Average LLM calls", _num(perf["average_llm_calls"])),
            _row("Average tool calls", _num(perf["average_tool_calls"])),
            _row("Average reasoning steps", _num(perf["average_reasoning_steps"])),
            _row("Average verification retries", _num(perf["average_verification_retries"])),
            _row("Average tokens (provider-reported)",
                 "unavailable" if tokens["average_total_tokens"] is None
                 else f"{tokens['average_total_tokens']:.0f}" + (" (lower bound)" if tokens["lower_bound"] else "")),
        ]
        if "loop_termination" in m:
            lines.append(_row("Loop termination rate", _pct(m["loop_termination"]["loop_termination_rate"])))

    lines += ["", "=" * WIDTH]
    return "\n".join(lines)
