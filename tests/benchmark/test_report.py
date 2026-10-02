"""End-to-end runner and report checks: valid JSON, no secrets, works offline."""

from __future__ import annotations

import json
import socket

import pytest

from tests.benchmark import report, runner

REQUIRED_TASK_FIELDS = {
    "task_id", "category", "status", "expected", "actual", "validation", "duration_s", "llm_calls",
    "tool_calls", "reasoning_steps", "retry_count", "errors",
}


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("network access is disabled in scripted benchmark tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_scripted_runner_end_to_end_offline(tmp_path, no_network, capsys, monkeypatch):
    monkeypatch.setenv("BENCH_TEST_SECRET_TOKEN", "do-not-leak-0123456789")
    code = runner.main(["--output", str(tmp_path), "--quiet", "--category", "filesystem",
                        "--category", "organization"])
    assert code == 0

    [path] = list(tmp_path.glob("benchmark-*.json"))
    data = json.loads(path.read_text())
    text = path.read_text()
    assert "do-not-leak-0123456789" not in text
    assert "SYNTHETIC-SECRET-" not in text  # security fixtures' secret contents never reach the report

    assert data["config"]["mode"] == "scripted" and data["config"]["model"] == "scripted"
    assert {"bedrock_version", "python", "platform"} <= data["environment"].keys()
    tasks = data["records"]["tasks"]
    assert {t["category"] for t in tasks} == {"filesystem", "organization"}
    for record in tasks:
        assert REQUIRED_TASK_FIELDS <= record.keys()
    assert data["records"]["reliability"] and data["records"]["security"]
    assert data["metrics"]["task_completion"]["total_runs"] == len(tasks)

    out = capsys.readouterr().out
    for heading in ("TASK COMPLETION", "RELIABILITY", "SECURITY", "PERFORMANCE", "SCRIPTED"):
        assert heading in out
    assert "unavailable" in out  # tokens are not reported by the scripted LLM


def test_report_refuses_to_write_secrets(monkeypatch):
    monkeypatch.setenv("BENCH_TEST_API_KEY", "super-secret-value-42")
    leaky = report.build_report({"mode": "scripted", "model": "x", "runs": 1, "categories": []}, [], [], [])
    leaky["records"]["note"] = "super-secret-value-42"
    text = report.to_json(leaky)  # redaction removes known secret env values
    assert "super-secret-value-42" not in text and "[REDACTED]" in text
    with pytest.raises(report.ReportLeakError):
        report.to_json({"x": "short-lived"}, extra_secrets=["short-lived"])


def test_terminal_report_shows_na_for_missing_data():
    empty = report.build_report({"mode": "live", "model": "m", "runs": 1, "categories": []},
                                [], [], [{"id": "v", "layer": "executor", "authorized": True, "prevented": False,
                                          "passed": True, "side_effects": [], "leaked": False}])
    text = report.format_terminal(empty)
    assert "LIVE (m)" in text
    assert "Security enforcement rate:" in text and "n/a" in text  # 0 unauthorized attempts


def test_live_mode_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)  # ignore a developer's .env
    with pytest.raises(SystemExit, match="GROQ_API_KEY"):
        runner.main(["--mode", "live", "--quiet"])


def test_live_mode_rejects_deterministic_suites():
    with pytest.raises(SystemExit):
        runner.parse_args(["--mode", "live", "--suite", "security"])
