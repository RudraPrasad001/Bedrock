"""Verification Agent: checks whether the task actually succeeded.

It combines deterministic checks on the resulting state (do the files exist,
are they non-empty) with an LLM judgement. It only has read-only tool access.
"""

from __future__ import annotations

from pathlib import Path

from pc_agent.agents.base import AgentError, LLMAgent, truncate
from pc_agent.core.decisions import ToolResult, Verification, VerificationCheck
from pc_agent.core.state import TaskState
from pc_agent.llm.base import LLMError
from pc_agent.tools.registry import ToolContext, ToolRegistry, ToolValidationError

SYSTEM_PROMPT = """You are the VERIFIER of a local computer task agent.
You judge whether the user's task was actually accomplished, based on evidence:
the actions taken, their results, automated file checks and excerpts of created files.
You cannot run or change anything.

Be strict but fair:
- Fail if a requested file is missing, empty, or obviously incomplete (e.g. a report
  that omits items the notes say should be included).
- Fail if the final answer is unsupported by the observations.
- Pass if the evidence shows the request was satisfied. Do not invent extra requirements.
- In DRY RUN mode, simulated actions count as done.

Do not use function/tool calling. Respond with ONLY a JSON object, as plain text:
{"success": true, "checks": [{"check": "Report exists", "passed": true}], "issues": [],
 "recommended_action": null}"""

EXCERPT_CHARS = 3000
MAX_EXCERPTS = 3


class VerifierAgent(LLMAgent):
    name = "verifier"

    def __init__(self, llm, read_only_tools: ToolRegistry, context: ToolContext, log=None):
        super().__init__(llm, log)
        self.system_prompt = SYSTEM_PROMPT
        self.tools = read_only_tools
        self.context = context

    def _read(self, action: str, **arguments) -> ToolResult:
        try:
            tool, args = self.tools.validate(action, arguments)
        except ToolValidationError as exc:
            return ToolResult.fail(str(exc))
        return tool.func(args, self.context)

    def file_checks(self, state: TaskState) -> tuple[list[VerificationCheck], dict[str, str]]:
        written = {
            o.result.metadata.get("path")
            for o in state.observations
            if o.decision.action == "filesystem.write" and o.result.success
        }
        checks: list[VerificationCheck] = []
        excerpts: dict[str, str] = {}

        # Files the agent authored get a detailed check and an excerpt for the LLM.
        for path in [p for p in state.files_created + state.files_modified if p in written]:
            exists = Path(path).is_file()
            checks.append(VerificationCheck(check=f"{path} exists", passed=exists))
            if not exists:
                continue
            checks.append(VerificationCheck(check=f"{path} is non-empty", passed=Path(path).stat().st_size > 0))
            if len(excerpts) < MAX_EXCERPTS:
                result = self._read("filesystem.read", path=path, max_chars=EXCERPT_CHARS)
                if result.success and result.output:
                    excerpts[path] = result.output

        # Bulk copies/moves/deletes are checked in aggregate to keep the output readable.
        others = [p for p in state.files_created if p not in written]
        if others:
            present = sum(Path(p).exists() for p in others)
            checks.append(VerificationCheck(
                check=f"{present}/{len(others)} copied/moved files present at destination",
                passed=present == len(others),
            ))
        if state.files_deleted:
            gone = sum(not Path(p).exists() for p in state.files_deleted)
            checks.append(VerificationCheck(
                check=f"{gone}/{len(state.files_deleted)} removed/moved source paths are gone",
                passed=gone == len(state.files_deleted),
            ))
        return checks, excerpts

    def build_prompt(self, state: TaskState, checks: list[VerificationCheck], excerpts: dict[str, str]) -> str:
        lines = [f"TASK: {state.user_request}"]
        if state.dry_run:
            lines.append("MODE: DRY RUN")
        if state.plan:
            lines.append(f"GOAL: {state.plan.goal}")
        lines.append("ACTIONS:")
        for obs in state.observations:
            status = "OK" if obs.result.success else "FAILED"
            detail = obs.result.summary if obs.result.success else obs.result.error
            lines.append(f" #{obs.index} {obs.decision.action} -> {status}: {detail}")
        if not state.observations:
            lines.append(" (none)")
        if state.notes:
            lines.append("AGENT NOTES:")
            lines += [f" - {truncate(n, 600)}" for n in state.notes]
        lines.append(f"FINAL ANSWER FROM AGENT: {state.answer or '(none)'}")
        if checks:
            lines.append("AUTOMATED CHECKS:")
            lines += [f" [{'PASS' if c.passed else 'FAIL'}] {c.check}" for c in checks]
        for path, text in excerpts.items():
            lines.append(f"EXCERPT OF {path}:\n{truncate(text, EXCERPT_CHARS)}")
        return "\n".join(lines)

    def run(self, state: TaskState) -> Verification:
        checks, excerpts = self.file_checks(state)
        hard_failures = [c.check for c in checks if not c.passed]
        try:
            verdict = self.ask_json(self.build_prompt(state, checks, excerpts), Verification)
        except (AgentError, LLMError) as exc:
            # Fall back to the deterministic checks alone.
            self.log.log(self.name, "llm_unavailable", error=str(exc))
            verdict = Verification(success=state.answer is not None or bool(checks), issues=[])
        # Automated checks are authoritative and always shown first.
        verdict.checks = checks + [c for c in verdict.checks if c.check not in {x.check for x in checks}]
        if hard_failures:
            verdict.success = False
            verdict.issues = [f"Check failed: {c}" for c in hard_failures] + verdict.issues
        if not verdict.success and not verdict.issues:
            verdict.issues = ["The verifier judged the task incomplete"]
        self.log.log(self.name, "verification", verification=verdict.model_dump())
        return verdict
