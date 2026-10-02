"""Standalone benchmark runner.

    uv run python -m tests.benchmark.runner                      # scripted, all suites
    uv run python -m tests.benchmark.runner --runs 3 --category filesystem
    uv run python -m tests.benchmark.runner --mode live --suite tasks

Scripted mode (default) needs no API key and no network: the LLM is replaced
by predetermined replies, so it measures Bedrock's tools, orchestration,
recovery paths and safety controls - not model quality.

Live mode sends the task suite to the configured LLM (Groq) and is
non-deterministic. Its results are reported separately and must not be used
as unit-test expectations. The reliability and security suites are always
deterministic and are only run in scripted mode.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tests.benchmark import harness, report
from tests.benchmark.reliability import run_reliability
from tests.benchmark.security_scenarios import run_security

SUITES = ("tasks", "reliability", "security")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    categories = sorted({t["category"] for t in harness.load_tasks()})
    parser = argparse.ArgumentParser(prog="python -m tests.benchmark.runner", description="Bedrock benchmark suite")
    parser.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per task/scenario (default 1)")
    parser.add_argument("--category", action="append", choices=categories, default=[],
                        help="Only run task in this category (repeatable)")
    parser.add_argument("--suite", action="append", choices=SUITES, default=[],
                        help="Suites to run (repeatable). Default: all in scripted mode, tasks in live mode")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"), help="Directory for JSON reports")
    parser.add_argument("--model", help="Live mode: override LLM_MODEL")
    parser.add_argument("--quiet", action="store_true", help="No per-run progress lines")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if not args.suite:
        args.suite = ["tasks"] if args.mode == "live" else list(SUITES)
    if args.mode == "live" and set(args.suite) - {"tasks"}:
        parser.error("reliability and security suites are deterministic; run them in scripted mode")
    return args


def live_client(model: str | None):
    from pc_agent.config import Config
    from pc_agent.llm.client import create_client

    config = Config.from_env()  # loads .env
    if config.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Live mode requires GROQ_API_KEY (set it in .env). Use --mode scripted for offline runs.")
    if model:
        config.model = model
    return create_client(config)


def progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    show = not args.quiet
    llm = live_client(args.model) if args.mode == "live" else None
    model = llm.model if llm else "scripted"
    if args.mode == "live":
        progress(True, f"LIVE LLM BENCHMARK - model {model}. Results depend on the model and are non-deterministic.")

    task_records: list[dict] = []
    if "tasks" in args.suite:
        for task in harness.load_tasks(args.category):
            for i in range(args.runs):
                record = harness.run_task(task, args.mode, i, llm=llm).record
                task_records.append(record.to_dict())
                progress(show, f"[task] {record.task_id:28} run {i}  {record.status:8} {record.duration_s:.3f}s")

    reliability_records: list[dict] = []
    if "reliability" in args.suite:
        for record in run_reliability(args.runs):
            reliability_records.append(record.to_dict())
            progress(show, f"[reliability] {record.scenario:36} run {record.run_index}  "
                           f"{'ok' if record.expected_behavior_met else 'FAILED'}")

    security_records: list[dict] = []
    if "security" in args.suite:
        for record in run_security():
            security_records.append(record.to_dict())
            if not record.passed:
                progress(show, f"[security] FAILED {record.id}[{record.layer}]: {record.note}")
        progress(show, f"[security] {len(security_records)} cases run")

    config = {
        "mode": args.mode,
        "model": model,
        "runs": args.runs,
        "categories": args.category,
        "suites": args.suite,
    }
    result = report.build_report(config, task_records, reliability_records, security_records)
    path = report.write_json(result, args.output)
    print(report.format_terminal(result))
    print(f"JSON report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
