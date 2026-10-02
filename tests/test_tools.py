from __future__ import annotations

import json

import pytest

from pc_agent.agents.executor import ExecutorAgent
from pc_agent.config import Config
from pc_agent.core.decisions import AgentDecision, RiskLevel
from pc_agent.core.state import TaskState
from pc_agent.tools.registry import ToolValidationError, build_registry


def test_registry_contains_core_tools(registry):
    for name in ("filesystem.list", "filesystem.search", "filesystem.read", "filesystem.write",
                 "system.disk_usage", "system.processes", "shell.run"):
        assert name in registry.enabled_names()


def test_destructive_and_python_tools_are_opt_in(config):
    registry = build_registry(config)
    assert "filesystem.delete" not in registry.enabled_names()
    assert "python.execute" not in registry.enabled_names()
    with pytest.raises(ToolValidationError, match="opt-in"):
        registry.get("filesystem.delete")

    enabled = build_registry(Config(allowed_paths=config.allowed_paths, allow_destructive=True, allow_python=True))
    assert {"filesystem.delete", "python.execute"} <= set(enabled.enabled_names())


def test_unknown_tool_is_rejected(registry):
    with pytest.raises(ToolValidationError, match="Unknown tool"):
        registry.validate("os.system", {"cmd": "ls"})


def test_arguments_are_validated(registry):
    with pytest.raises(ToolValidationError, match="path"):
        registry.validate("filesystem.list", {})
    with pytest.raises(ToolValidationError, match="Extra inputs"):
        registry.validate("filesystem.list", {"path": "~", "evil": True})
    with pytest.raises(ToolValidationError, match="max_results"):
        registry.validate("filesystem.search", {"path": "~", "max_results": 10_000})


def test_read_only_view(registry):
    view = registry.read_only()
    assert "filesystem.read" in view.enabled_names()
    assert "filesystem.write" not in view.enabled_names()
    assert "shell.run" not in view.enabled_names()


def test_catalog_lists_signatures(registry):
    catalog = registry.catalog()
    assert 'sort_by: "name"|"size"|"modified"' in catalog
    assert "filesystem.delete" not in catalog


# -- shell ------------------------------------------------------------------


def test_shell_safe_command_runs(registry, ctx, sandbox):
    tool, args = registry.validate("shell.run", {"command": "ls", "cwd": str(sandbox)})
    (sandbox / "hello.txt").write_text("hi")
    assert tool.assess_call(args, ctx).requires_confirmation is False
    result = tool.func(args, ctx)
    assert result.success and "hello.txt" in result.output


def test_shell_dangerous_command_requires_confirmation(registry, ctx):
    tool, args = registry.validate("shell.run", {"command": "rm -rf /tmp/x"})
    assessment = tool.assess_call(args, ctx)
    assert assessment.allowed and assessment.requires_confirmation
    assert assessment.risk == RiskLevel.PRIVILEGED


def test_shell_blocked_command_never_runs(registry, ctx):
    tool, args = registry.validate("shell.run", {"command": "ls; rm -rf ~"})
    assert not tool.assess_call(args, ctx).allowed
    assert not tool.func(args, ctx).success


def test_shell_nonzero_exit_is_a_failure(registry, ctx, sandbox):
    tool, args = registry.validate("shell.run", {"command": f"ls {sandbox}/missing"})
    result = tool.func(args, ctx)
    assert not result.success and "Exit code" in result.error


# -- system -----------------------------------------------------------------


def test_disk_usage(registry, ctx):
    tool, args = registry.validate("system.disk_usage", {"path": "/"})
    result = tool.func(args, ctx)
    assert result.success
    assert json.loads(result.output)["filesystems"][0]["free_bytes"] > 0


def test_disk_usage_is_allowed_outside_sandbox(ctx, registry):
    executor = ExecutorAgent(registry, ctx)
    decision = AgentDecision(action="system.disk_usage", arguments={"path": "/"})
    assert executor.assess(decision).allowed
    assert executor.run(decision, TaskState("t")).success


def test_processes(registry, ctx):
    tool, args = registry.validate("system.processes", {"limit": 5})
    result = tool.func(args, ctx)
    assert result.success
    assert len(json.loads(result.output)["processes"]) <= 5


def test_environment_hides_secrets(registry, ctx, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_supersecretvalue123")
    tool, args = registry.validate("system.environment", {})
    result = tool.func(args, ctx)
    assert "GROQ_API_KEY" not in result.output
    assert "gsk_supersecretvalue123" not in result.output


# -- executor ---------------------------------------------------------------


def test_executor_dry_run_does_not_write(config, ctx, registry, sandbox):
    config.dry_run = True
    executor = ExecutorAgent(registry, ctx)
    state = TaskState(user_request="t", dry_run=True)
    target = sandbox / "report.md"
    result = executor.run(AgentDecision(action="filesystem.write", arguments={"path": str(target), "content": "x"}),
                          state)
    assert result.success and result.metadata["dry_run"]
    assert "DRY RUN" in result.output
    assert not target.exists()
    assert state.files_created == []


def test_executor_dry_run_still_reads(config, ctx, registry, sandbox):
    config.dry_run = True
    (sandbox / "a.txt").write_text("content")
    result = ExecutorAgent(registry, ctx).run(
        AgentDecision(action="filesystem.read", arguments={"path": str(sandbox / "a.txt")}), TaskState("t")
    )
    assert result.success and "content" in result.output


def test_executor_tracks_files_and_redacts(ctx, registry, sandbox, monkeypatch):
    monkeypatch.setenv("SOME_SECRET", "hunter2hunter2")
    (sandbox / "leak.txt").write_text("password=hunter2hunter2")
    executor = ExecutorAgent(registry, ctx)
    state = TaskState("t")
    read = executor.run(AgentDecision(action="filesystem.read", arguments={"path": str(sandbox / "leak.txt")}), state)
    assert "hunter2hunter2" not in read.output and "[REDACTED]" in read.output

    executor.run(AgentDecision(action="filesystem.write",
                               arguments={"path": str(sandbox / "new.md"), "content": "x"}), state)
    assert state.files_created == [str(sandbox / "new.md")]


def test_executor_turns_bad_requests_into_failed_results(ctx, registry):
    executor = ExecutorAgent(registry, ctx)
    result = executor.run(AgentDecision(action="filesystem.nuke", arguments={}), TaskState("t"))
    assert not result.success and "Unknown tool" in result.error
    assessment = executor.assess(AgentDecision(action="filesystem.read", arguments={"path": "/etc/passwd"}))
    assert not assessment.allowed
