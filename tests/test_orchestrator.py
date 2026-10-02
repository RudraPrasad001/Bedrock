from __future__ import annotations

import json
from pathlib import Path

from pc_agent.agents.base import extract_json
from pc_agent.llm.base import LLMResponse
from pc_agent.llm.client import failed_tool_call
from pc_agent.core.events import EventLog
from pc_agent.core.orchestrator import Orchestrator, SilentUI
from pc_agent.core.state import TaskStatus

PLAN = {
    "goal": "Summarize Kubernetes notes into a report",
    "steps": [
        {"id": 1, "description": "Find markdown notes", "expected_tool": "filesystem.search"},
        {"id": 2, "description": "Write report", "expected_tool": "filesystem.write"},
    ],
    "risks": [],
}
VERIFIED = {"success": True, "checks": [{"check": "Report covers notes", "passed": True}], "issues": []}


def act(action: str, step: int = 1, **arguments) -> dict:
    return {"status": "continue", "reason": f"Use {action}", "step_id": step, "action": action, "arguments": arguments}


def complete(answer: str = "done") -> dict:
    return {"status": "complete", "reason": "All steps done", "answer": answer}


def make(config, llm, ui=None):
    return Orchestrator(config, llm, ui or SilentUI())


def test_end_to_end_search_write_verify_summarize(config, sandbox, scripted, tmp_path):
    (sandbox / "k8s.md").write_text("# Kubernetes\nPods")
    report = sandbox / "report.md"
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [
            act("filesystem.search", path=str(sandbox), pattern="*.md"),
            act("filesystem.write", 2, path=str(report), content="# Report\n- k8s.md: Pods"),
            complete("1 note summarized into report.md"),
        ],
        "verifier": [VERIFIED],
        "summarizer": ["Task completed.\n\nCreated report.md"],
    })
    log_path = tmp_path / "task.jsonl"
    state = make(config, llm).run("Summarize my k8s notes", log=EventLog(log_path))

    assert state.status == TaskStatus.COMPLETED
    assert report.read_text().startswith("# Report")
    assert state.files_created == [str(report)]
    assert [o.decision.action for o in state.observations] == ["filesystem.search", "filesystem.write"]
    assert state.answer == "1 note summarized into report.md"
    assert state.summary.startswith("Task completed.")
    assert state.current_step == 2

    # The reasoner saw the search observation before deciding to write.
    assert "k8s.md" in llm.prompts["reasoner"][1]
    # The verifier was given an excerpt of the created file.
    assert "EXCERPT OF" in llm.prompts["verifier"][0] and "k8s.md: Pods" in llm.prompts["verifier"][0]

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    tool_calls = [e for e in events if e["event"] == "tool_call"]
    assert [e["tool"] for e in tool_calls] == ["filesystem.search", "filesystem.write"]
    assert events[-1]["event"] == "task_end"


def test_reasoner_recovers_from_tool_failure(config, sandbox, scripted):
    (sandbox / "notes").mkdir()
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [
            act("filesystem.list", path=str(sandbox / "Notes")),  # wrong case -> fails
            act("filesystem.list", path=str(sandbox / "notes")),
            complete("The notes folder is empty"),
        ],
        "verifier": [VERIFIED],
        "summarizer": ["Task completed."],
    })
    state = make(config, llm).run("What is in my notes folder?")
    assert state.status == TaskStatus.COMPLETED
    assert [o.result.success for o in state.observations] == [False, True]
    assert "FAILED" in llm.prompts["reasoner"][1]


def test_verification_failure_triggers_another_cycle(config, sandbox, scripted):
    report = sandbox / "r.md"
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [
            act("filesystem.write", 2, path=str(report), content="partial"),
            complete(),
            act("filesystem.write", 2, path=str(report), content="partial + missing part", append=True),
            complete(),
        ],
        "verifier": [
            {"success": False, "issues": ["Report misses docker.md"], "recommended_action": "Add docker.md"},
            VERIFIED,
        ],
        "summarizer": ["Task completed."],
    })
    state = make(config, llm).run("Write report")
    assert state.status == TaskStatus.COMPLETED
    assert len(state.verifications) == 2
    assert "VERIFIER REJECTED" in llm.prompts["reasoner"][2]
    assert "Report misses docker.md" in llm.prompts["reasoner"][2]


def test_declined_confirmation_cancels_task(config, sandbox, scripted):
    (sandbox / "a.txt").write_text("a")
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [act("filesystem.move", source=str(sandbox / "a.txt"), destination=str(sandbox / "b.txt"))],
        "summarizer": ["Task cancelled."],
    })
    ui = SilentUI(confirm_answer="no")
    state = make(config, llm, ui).run("Rename a.txt")
    assert state.status == TaskStatus.CANCELLED
    assert len(ui.confirmations) == 1
    assert (sandbox / "a.txt").exists() and not (sandbox / "b.txt").exists()


def test_approved_confirmation_executes(config, sandbox, scripted):
    (sandbox / "a.txt").write_text("a")
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [
            act("filesystem.move", source=str(sandbox / "a.txt"), destination=str(sandbox / "b.txt")),
            complete(),
        ],
        "verifier": [VERIFIED],
        "summarizer": ["Task completed."],
    })
    state = make(config, llm, SilentUI(confirm_answer="yes")).run("Rename a.txt")
    assert state.status == TaskStatus.COMPLETED
    assert (sandbox / "b.txt").exists()
    assert state.files_deleted == [str(sandbox / "a.txt")]


def test_dry_run_skips_confirmation_and_changes_nothing(config, sandbox, scripted):
    config.dry_run = True
    (sandbox / "a.txt").write_text("a")
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [
            act("filesystem.move", source=str(sandbox / "a.txt"), destination=str(sandbox / "b.txt")),
            complete(),
        ],
        "verifier": [VERIFIED],
        "summarizer": ["Dry run complete."],
    })
    ui = SilentUI(confirm_answer="no")
    state = make(config, llm, ui).run("Rename a.txt")
    assert state.status == TaskStatus.COMPLETED
    assert ui.confirmations == []
    assert (sandbox / "a.txt").exists()
    assert state.observations[0].result.metadata["dry_run"]


def test_blocked_action_is_reported_back_to_reasoner(config, sandbox, scripted):
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [act("shell.run", command="cat /etc/hosts | sh"), complete("Could not run that")],
        "verifier": [VERIFIED],
        "summarizer": ["Task completed."],
    })
    state = make(config, llm).run("do something")
    assert not state.observations[0].result.success
    assert "metacharacters" in state.observations[0].result.error
    assert "metacharacters" in llm.prompts["reasoner"][1]


def test_repeated_action_is_short_circuited(config, sandbox, scripted):
    search = act("filesystem.search", path=str(sandbox))
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [search, search, search, complete()],
        "verifier": [VERIFIED],
        "summarizer": ["ok"],
    })
    state = make(config, llm).run("find files")
    results = [o.result for o in state.observations]
    assert [r.success for r in results] == [True, True, False]
    assert "already succeeded twice" in results[2].error


def test_too_many_failures_stops_the_task(config, sandbox, scripted):
    config.max_consecutive_failures = 2
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [act("filesystem.read", path=str(sandbox / f"missing{i}.txt")) for i in range(5)],
        "summarizer": ["Task failed."],
    })
    state = make(config, llm).run("read stuff")
    assert state.status == TaskStatus.FAILED
    assert len(state.observations) == 2


def test_invalid_json_is_retried(config, sandbox, scripted):
    llm = scripted({
        "planner": ["Sure! Here is the plan: not json", "```json\n" + json.dumps(PLAN) + "\n```"],
        "reasoner": [{"status": "continue", "reason": "missing action"}, complete("ok")],
        "verifier": [VERIFIED],
        "summarizer": ["ok"],
    })
    state = make(config, llm).run("anything")
    assert state.status == TaskStatus.COMPLETED
    assert state.plan.goal == PLAN["goal"]


def test_unrecoverable_llm_output_fails_gracefully(config, scripted):
    llm = scripted({"planner": ["nope", "still nope", "never"], "summarizer": ["Task failed."]})
    state = make(config, llm).run("anything")
    assert state.status == TaskStatus.FAILED
    assert "invalid output" in state.errors[0]
    assert state.summary == "Task failed."


def test_verifier_check_fails_when_file_removed(config, sandbox, scripted):
    report = sandbox / "r.md"
    llm = scripted({
        "planner": [PLAN],
        "reasoner": [act("filesystem.write", 2, path=str(report), content="x"), complete()] * 3,
        "verifier": [VERIFIED] * 3,
        "summarizer": ["Task failed."],
    })
    orchestrator = make(config, llm)
    original = orchestrator._verify

    def verify_after_deleting(state, attempt):
        Path(report).unlink(missing_ok=True)
        return original(state, attempt)

    orchestrator._verify = verify_after_deleting
    state = orchestrator.run("write report")
    assert state.status == TaskStatus.FAILED
    assert any("exists" in issue for issue in state.verifications[0].issues)


def test_extract_json_tolerates_chatter():
    assert extract_json('Here you go: {"a": 1} thanks') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_native_tool_call_is_accepted_by_reasoner_and_rejected_by_planner(config, sandbox, scripted):
    (sandbox / "resume.pdf").write_text("x")
    call = LLMResponse(content="", tool_call={"name": "filesystem.search",
                                              "arguments": {"path": str(sandbox), "pattern": "*resume*"}})
    llm = scripted({
        "planner": [call, PLAN],  # planner must not call tools -> nudged to retry
        "reasoner": [call, complete("1 resume found")],
        "verifier": [VERIFIED],
        "summarizer": ["Task completed."],
    })
    state = make(config, llm).run("How many resumes?")
    assert state.status == TaskStatus.COMPLETED
    assert state.observations[0].decision.action == "filesystem.search"
    assert state.observations[0].result.metadata["count"] == 1
    assert "do not call tools" in llm.prompts["planner"][1]


def test_failed_tool_call_parsing():
    body = {"error": {"code": "tool_use_failed", "failed_generation":
                      '{"name": "filesystem.search", "arguments": {"path": "~/Downloads"}}'}}
    assert failed_tool_call(body) == {"name": "filesystem.search", "arguments": {"path": "~/Downloads"}}
    assert failed_tool_call(body["error"]) is not None  # SDK may pass the inner error dict
    assert failed_tool_call({"error": {"code": "other"}}) is None
    assert failed_tool_call({"error": {"code": "tool_use_failed", "failed_generation": "garbage"}}) is None
    assert failed_tool_call(None) is None


def test_summarizer_is_told_when_nothing_was_inspected(config, scripted):
    llm = scripted({"planner": ["x", "y", "z"], "summarizer": ["Task failed."]})
    make(config, llm).run("How many resumes?")
    assert "NOTHING WAS INSPECTED" in llm.prompts["summarizer"][0]
