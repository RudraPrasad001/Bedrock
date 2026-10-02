"""Deterministic post-execution checks.

Checks inspect the resulting filesystem and the real tool output recorded in
TaskState. They never trust the model's own claim of success.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pc_agent.core.state import TaskState

from tests.benchmark.workspace import Workspace, sha256


@dataclass
class CheckResult:
    check: str
    passed: bool
    detail: str = ""


def _successful_outputs(state: TaskState, tool: str | None) -> list[str]:
    return [
        o.result.output or ""
        for o in state.observations
        if o.result.success and (tool is None or o.decision.action == tool)
    ]


def _paths_in_output(output: str, ws: Workspace) -> set[str]:
    """Workspace-relative file paths mentioned in a tool output.

    JSON outputs are walked structurally (ignoring the 'root' key, which is a
    directory); plain-text outputs are scanned for absolute workspace paths.
    """
    root = str(ws.root)
    found: set[str] = set()

    def add(value: str) -> None:
        if value.startswith(root + "/"):
            found.add(value[len(root) + 1 :])

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k != "root":
                    walk(v, k)
        elif isinstance(node, list):
            for item in node:
                walk(item, key)
        elif isinstance(node, str):
            add(node)

    try:
        walk(json.loads(output))
    except (json.JSONDecodeError, TypeError):
        for match in re.findall(re.escape(root) + r"/[^\s\"']+", output):
            add(match)
    return found


def check_observation_paths(spec: dict, ws: Workspace, state: TaskState) -> CheckResult:
    """Some successful call of `tool` returned exactly the expected file set."""
    expected = set(spec["expected"])
    outputs = _successful_outputs(state, spec["tool"])
    if not outputs:
        return CheckResult("observation_paths", False, f"no successful {spec['tool']} call")
    for output in outputs:
        if _paths_in_output(output, ws) == expected:
            return CheckResult("observation_paths", True, f"{spec['tool']} returned exactly {sorted(expected)}")
    last = sorted(_paths_in_output(outputs[-1], ws))
    return CheckResult("observation_paths", False, f"expected {sorted(expected)}, last output had {last}")


def check_observation_contains(spec: dict, ws: Workspace, state: TaskState) -> CheckResult:
    hit = any(spec["text"] in out for out in _successful_outputs(state, spec["tool"]))
    return CheckResult("observation_contains", hit, f"'{spec['text']}' {'found' if hit else 'not found'}")


def check_answer_mentions(spec: dict, ws: Workspace, state: TaskState) -> CheckResult:
    answer = f"{state.answer or ''}\n{state.summary or ''}".lower()
    missing = [t for t in spec["tokens"] if t.lower() not in answer]
    return CheckResult("answer_mentions", not missing, f"missing {missing}" if missing else "all tokens present")


def check_file_contains(spec: dict, ws: Workspace, state: TaskState) -> CheckResult:
    path = ws.path(spec["path"])
    if not path.is_file():
        return CheckResult("file_contains", False, f"{spec['path']} does not exist")
    text = path.read_text(errors="replace").lower()
    missing = [t for t in spec["all"] if t.lower() not in text]
    return CheckResult("file_contains", not missing, f"missing {missing}" if missing else f"{spec['path']} ok")


def check_unchanged(spec: dict, ws: Workspace, state: TaskState) -> CheckResult:
    changed = [
        rel for rel in spec["paths"]
        if not ws.path(rel).is_file() or sha256(ws.path(rel)) != ws.manifest[rel]
    ]
    return CheckResult("unchanged", not changed, f"changed {changed}" if changed else "unchanged")


def check_organized_by_extension(spec: dict, ws: Workspace, state: TaskState) -> CheckResult:
    """Each file left its original spot, exists exactly once below `dir` with
    identical content, and files share a subfolder iff they share an extension."""
    base = ws.path(spec["dir"])
    problems: list[str] = []
    folder_by_ext: dict[str, set[Path]] = {}
    for rel in spec["files"]:
        original = ws.path(rel)
        if original.exists():
            problems.append(f"{rel} not moved")
            continue
        digest = ws.manifest[rel]
        matches = [p for p in base.rglob(original.name) if p.is_file() and sha256(p) == digest]
        if len(matches) != 1:
            problems.append(f"{original.name}: {len(matches)} intact copies found")
            continue
        if matches[0].parent == base:
            problems.append(f"{original.name} still at top level")
            continue
        folder_by_ext.setdefault(original.suffix.lower(), set()).add(matches[0].parent)
    for ext, folders in folder_by_ext.items():
        if len(folders) != 1:
            problems.append(f"{ext} files split across {len(folders)} folders")
    all_folders = [next(iter(f)) for f in folder_by_ext.values() if len(f) == 1]
    if len(all_folders) != len(set(all_folders)):
        problems.append("different extensions share a folder")
    return CheckResult("organized_by_extension", not problems, "; ".join(problems) or "organized")


def check_tool_succeeded(spec: dict, ws: Workspace, state: TaskState) -> CheckResult:
    used = [o.decision.action for o in state.observations if o.result.success]
    ok = any(tool in used for tool in spec["tools"])
    return CheckResult("tool_succeeded", ok, f"successful tools: {sorted(set(used))}")


CHECKS = {
    "observation_paths": check_observation_paths,
    "observation_contains": check_observation_contains,
    "answer_mentions": check_answer_mentions,
    "file_contains": check_file_contains,
    "unchanged": check_unchanged,
    "organized_by_extension": check_organized_by_extension,
    "tool_succeeded": check_tool_succeeded,
}


def validate(checks: list[dict], mode: str, ws: Workspace, state: TaskState) -> list[CheckResult]:
    """Run every check that applies to `mode`. A crashing check counts as failed."""
    results = []
    for spec in checks:
        if mode not in spec.get("modes", ["scripted", "live"]):
            continue
        try:
            results.append(CHECKS[spec["type"]](spec, ws, state))
        except Exception as exc:  # a broken check must not abort the benchmark
            results.append(CheckResult(spec["type"], False, f"check error: {type(exc).__name__}: {exc}"))
    return results
