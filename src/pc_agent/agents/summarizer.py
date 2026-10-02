"""Summary Agent: writes the final human-readable report. No tool access."""

from __future__ import annotations

from pc_agent.agents.base import LLMAgent, truncate
from pc_agent.core.state import TaskState, TaskStatus
from pc_agent.llm.base import LLMError

SYSTEM_PROMPT = """You are the SUMMARIZER of a local computer task agent.
Write the final message to the user from the facts given. You cannot run anything.

- Start with one line: "Task completed.", "Task partially completed.", "Task failed." or "Task cancelled."
- Then give the concrete results (counts, sizes, findings) as short bullet points.
- List files created/modified/deleted with full paths, if any.
- Mention failures, skipped items and verifier issues honestly.
- In dry-run mode, say clearly that no changes were made.
- Be concise: at most ~15 lines. Use plain markdown.
- Only report results that appear in FINDINGS or NOTES. If nothing was inspected,
  say the question could not be answered; never claim that something was not found."""


class SummarizerAgent(LLMAgent):
    name = "summarizer"
    system_prompt = SYSTEM_PROMPT

    def facts(self, state: TaskState) -> str:
        lines = [f"TASK: {state.user_request}", f"STATUS: {state.status.value}"]
        if state.dry_run:
            lines.append("MODE: DRY RUN (no changes were made)")
        if state.plan:
            lines.append(f"GOAL: {state.plan.goal}")
        lines.append(f"FINDINGS: {state.answer or '(none)'}")
        if state.notes:
            lines.append("NOTES:")
            lines += [f" - {truncate(n, 500)}" for n in state.notes[-8:]]
        lines.append(f"ACTIONS EXECUTED: {len(state.observations)}")
        if not state.observations:
            lines.append("NOTHING WAS INSPECTED: there are no results to report.")
        for label, files in (
            ("CREATED", state.files_created),
            ("MODIFIED", state.files_modified),
            ("DELETED", state.files_deleted),
        ):
            if files:
                lines.append(f"FILES {label}: {', '.join(files)}")
        if state.errors:
            lines.append("ERRORS:")
            lines += [f" - {truncate(e, 300)}" for e in state.errors[-6:]]
        verification = state.last_verification()
        if verification:
            lines.append(f"VERIFIED: {verification.success}")
            lines += [f" issue: {i}" for i in verification.issues]
        return "\n".join(lines)

    def run(self, state: TaskState) -> str:
        try:
            summary = self.ask_text(self.facts(state))
            if summary:
                return summary
        except LLMError as exc:
            self.log.log(self.name, "llm_unavailable", error=str(exc))
        return self.fallback(state)

    @staticmethod
    def fallback(state: TaskState) -> str:
        """Deterministic summary used when the LLM is unavailable."""
        headline = {
            TaskStatus.COMPLETED: "Task completed.",
            TaskStatus.CANCELLED: "Task cancelled.",
        }.get(state.status, "Task failed.")
        lines = [headline]
        if state.dry_run:
            lines.append("Dry run: no changes were made.")
        if state.answer:
            lines += ["", state.answer]
        for label, files in (("Created", state.files_created), ("Modified", state.files_modified),
                             ("Deleted", state.files_deleted)):
            if files:
                lines += ["", f"{label}:"] + [f"- {f}" for f in files]
        if state.errors:
            lines += ["", "Errors:"] + [f"- {e}" for e in state.errors[-5:]]
        return "\n".join(lines)
