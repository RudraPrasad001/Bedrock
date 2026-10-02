"""Task-completion benchmark, scripted mode, plus negative controls for the validators."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.benchmark.harness import load_tasks, run_task, task_by_id
from tests.benchmark.validation import validate
from tests.benchmark.workspace import CSV_SIZE, OLD_MTIME, RECENT_MTIME, build_workspace

TASKS = load_tasks()


def test_at_least_ten_tasks_with_required_fields():
    assert len(TASKS) >= 10
    assert len({t["id"] for t in TASKS}) == len(TASKS)
    for task in TASKS:
        assert {"id", "category", "instruction", "fixture", "expected", "checks"} <= task.keys()
        assert task["checks"], task["id"]


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_task_scripted(task):
    record = run_task(task, "scripted").record
    failures = [v for v in record.validation if not v["passed"]]
    assert record.status == "success", failures or record.errors
    assert record.orchestrator_status == "COMPLETED"
    assert record.terminated_within_limits
    assert record.tool_calls >= 1 and record.llm_calls >= 4
    assert record.tokens is None  # scripted LLM reports no usage: unavailable, not zero


# -- workspace ----------------------------------------------------------------


def test_workspace_is_deterministic(tmp_path):
    a = build_workspace(tmp_path / "a")
    b = build_workspace(tmp_path / "b")
    assert a.manifest == b.manifest
    assert a.path("Downloads/image.png").stat().st_size == 1_200_000
    assert a.path("Downloads/archive.zip").stat().st_size == 3_000_000
    assert a.path("Misc/data.csv").stat().st_size == CSV_SIZE
    assert a.path("Downloads/notes.txt").stat().st_mtime == OLD_MTIME
    assert a.path("Downloads/report.pdf").stat().st_mtime == RECENT_MTIME


def test_duplicates_fixture_is_content_based(workspace):
    dup, copy, near = (workspace.path(f"Documents/{n}") for n in ("duplicate.txt", "duplicate_copy.txt",
                                                                   "near_duplicate.txt"))
    assert dup.read_bytes() == copy.read_bytes()
    assert near.stat().st_size == dup.stat().st_size and near.read_bytes() != dup.read_bytes()


def test_each_run_gets_a_fresh_workspace():
    task = task_by_id("org-by-extension")
    first = run_task(task, run_index=0)
    second = run_task(task, run_index=1)  # would fail if files had already been moved
    assert first.record.status == second.record.status == "success"
    assert first.extra["workspace"] != second.extra["workspace"]
    assert not Path(first.extra["workspace"]).exists()  # cleaned up


def test_runs_never_touch_files_outside_the_workspace(isolated_home):
    for task in TASKS:
        result = run_task(task)
        ws = result.extra["workspace"]
        for path in result.state.files_created + result.state.files_modified + result.state.files_deleted:
            assert path.startswith(ws + "/"), path
    assert os.listdir(isolated_home) == []  # nothing appeared in (fake) HOME


# -- negative controls: the validators must be able to fail ---------------------


def _swap_reasoner(new_steps):
    def override(script, ws):
        script["reasoner"] = [{**s, "arguments": {k: (v.replace("{ws}", str(ws)) if isinstance(v, str) else v)
                                                   for k, v in s.get("arguments", {}).items()}}
                              for s in new_steps]
        return script
    return override


def test_wrong_search_fails_validation():
    task = task_by_id("fs-find-by-extension")
    wrong = [{"status": "continue", "reason": "x", "action": "filesystem.search",
              "arguments": {"path": "{ws}/Downloads", "pattern": "*.pdf", "recursive": False}},
             {"status": "complete", "reason": "x", "answer": "2 PDFs"}]  # claims success anyway
    record = run_task(task, script_override=_swap_reasoner(wrong)).record
    assert record.orchestrator_status == "COMPLETED"  # the agent believes it succeeded...
    assert record.status == "failed"  # ...but the deterministic check disagrees


def test_size_only_duplicate_detection_would_fail(workspace):
    from pc_agent.core.state import TaskState
    from pc_agent.core.decisions import AgentDecision, ToolResult
    import json

    state = TaskState("t")
    fake = {"duplicate_groups": [{"files": [str(workspace.path(f"Documents/{n}")) for n in
                                            ("duplicate.txt", "duplicate_copy.txt", "near_duplicate.txt")]}]}
    state.add_observation(AgentDecision(action="filesystem.duplicates"), ToolResult.ok(json.dumps(fake), "x"))
    [result] = validate(task_by_id("fs-find-duplicates")["checks"], "scripted", workspace, state)
    assert not result.passed


def test_incomplete_organization_is_partial():
    task = task_by_id("org-by-extension")

    def only_two_moves(script, ws):
        script["reasoner"] = script["reasoner"][:2] + script["reasoner"][-1:]
        return script

    record = run_task(task, script_override=only_two_moves).record
    assert record.status == "partial"  # nested/ untouched passes, organization check fails
    assert any("not moved" in v["detail"] for v in record.validation)


def test_summary_missing_topic_fails():
    task = task_by_id("gen-markdown-summary")

    def docker_only(script, ws):
        for step in script["reasoner"]:
            if step.get("action") == "filesystem.write":
                step["arguments"]["content"] = "# Summary\n\nDocker only.\n"
        return script

    record = run_task(task, script_override=docker_only).record
    assert record.status == "partial"
    assert any(v["check"] == "file_contains" and not v["passed"] for v in record.validation)


def test_live_only_checks_are_skipped_in_scripted_mode(workspace):
    from pc_agent.core.state import TaskState

    checks = [{"type": "answer_mentions", "tokens": ["nope"], "modes": ["live"]}]
    assert validate(checks, "scripted", workspace, TaskState("t")) == []
    assert not validate(checks, "live", workspace, TaskState("t"))[0].passed
