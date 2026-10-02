"""Reasoning Agent: inspects the task state and chooses the next single action.

It never touches the filesystem; it only emits structured action requests.
"""

from __future__ import annotations

import json

from pc_agent.agents.base import LLMAgent, environment_context, truncate
from pc_agent.core.decisions import AgentDecision, ReasonerOutput
from pc_agent.core.state import TaskState

SYSTEM_PROMPT = """You are the REASONER of a local computer task agent.
You look at the task state and decide the single next action. An executor runs it
and you will see the observation on your next turn. You never run anything yourself.

Tools you may request (exact names and argument names):
{tools}

Rules:
- Request exactly ONE tool call per reply, using only the tools above.
- Paths may use "~". Prefer dedicated tools over shell.run.
- Read the observations. Do not repeat an action that already succeeded.
- If an action failed, adapt: fix the arguments, try another approach, or skip what is inaccessible.
- Older observation outputs are truncated in later turns. Put facts you will need later
  (file lists, key points from documents you read, numbers) in "note".
- When creating reports, write the complete final content (markdown) in filesystem.write.
- Only read files that are relevant to the task.
- When the task is fully done, reply with status "complete" and put the concrete findings
  (numbers, paths, names) in "answer". Answer-only tasks (e.g. disk space) are complete once you have the data.
- Use status "failed" only if the task is impossible.
- In DRY RUN mode, actions with side effects are simulated; treat "DRY RUN" results as done.

Do not use function/tool calling. Respond with ONLY a JSON object, as plain text:
{{"status": "continue", "reason": "<one or two sentences, user-visible>", "step_id": 1,
  "action": "<tool name>", "arguments": {{...}}, "note": null}}
or
{{"status": "complete", "reason": "...", "answer": "<findings for the user>", "note": null}}"""

LATEST_OUTPUT_CHARS = 8000
RECENT_OUTPUT_CHARS = 1500
RECENT_WINDOW = 3


def _compact_args(arguments: dict) -> str:
    shown = {
        k: (f"<{len(v)} chars>" if k == "content" and isinstance(v, str) else v)
        for k, v in arguments.items()
    }
    return truncate(json.dumps(shown, ensure_ascii=False), 300)


class ReasonerAgent(LLMAgent):
    name = "reasoner"

    def __init__(self, llm, tool_catalog: str, max_consecutive_failures: int = 4, log=None):
        super().__init__(llm, log)
        self.system_prompt = SYSTEM_PROMPT.format(tools=tool_catalog)
        self.max_consecutive_failures = max_consecutive_failures

    # -- prompt -----------------------------------------------------------

    def build_view(self, state: TaskState) -> str:
        """A compact view of the state. Raw conversation history is never sent."""
        lines = [environment_context()]
        if state.dry_run:
            lines.append("MODE: DRY RUN (side effects are simulated)")
        lines += ["", f"TASK: {state.user_request}"]
        if state.plan:
            lines.append(f"GOAL: {state.plan.goal}")
            lines.append("PLAN:")
            for step in state.plan.steps:
                marker = "->" if step.id == state.current_step else "  "
                tool = f" [{step.expected_tool}]" if step.expected_tool else ""
                lines.append(f" {marker} {step.id}. {step.description}{tool}")
        if state.notes:
            lines.append("NOTES (your working memory):")
            lines += [f" - {n}" for n in state.notes]
        for label, files in (
            ("FILES CREATED", state.files_created),
            ("FILES MODIFIED", state.files_modified),
            ("FILES DELETED", state.files_deleted),
        ):
            if files:
                lines.append(f"{label}: {', '.join(files)}")
        verification = state.last_verification()
        if verification and not verification.success:
            lines.append("VERIFIER REJECTED THE RESULT. Fix these issues before completing:")
            lines += [f" - {issue}" for issue in verification.issues]
            if verification.recommended_action:
                lines.append(f" Recommended: {verification.recommended_action}")

        if not state.observations:
            lines.append("\nOBSERVATIONS: none yet. Choose the first action.")
            return "\n".join(lines)

        lines.append("\nOBSERVATIONS (oldest first):")
        total = len(state.observations)
        for obs in state.observations:
            result = obs.result
            status = "OK" if result.success else "FAILED"
            detail = result.summary if result.success else result.error
            lines.append(
                f"#{obs.index} {obs.decision.action} {_compact_args(obs.decision.arguments)} -> {status}: {detail}"
            )
            age = total - obs.index
            if age == 0:
                body = result.output if result.success else (result.output or "")
                if body:
                    lines.append(f"   output:\n{truncate(body, LATEST_OUTPUT_CHARS)}")
            elif age < RECENT_WINDOW and result.output:
                lines.append(f"   output: {truncate(result.output, RECENT_OUTPUT_CHARS)}")
        lines.append("\nDecide the next action.")
        return "\n".join(lines)

    # -- decisions --------------------------------------------------------

    def from_tool_call(self, call: dict) -> dict:
        """Some models answer with a native tool call; it is still a valid action request."""
        return {
            "status": "continue",
            "reason": f"Next action: {call['name']}",
            "action": call["name"],
            "arguments": call["arguments"],
        }

    def run(self, state: TaskState) -> ReasonerOutput:
        output = self.ask_json(self.build_view(state), ReasonerOutput)
        if output.step_id and state.plan and 1 <= output.step_id <= len(state.plan.steps):
            state.current_step = output.step_id
        if output.note:
            state.notes.append(output.note.strip())
        self.log.log(self.name, "decision", decision=output.model_dump())
        return output

    @staticmethod
    def to_decision(output: ReasonerOutput) -> AgentDecision:
        return AgentDecision(
            action=output.action or "",
            arguments=output.arguments,
            reason=output.reason,
            step_id=output.step_id,
        )

    def handle_failure(self, state: TaskState) -> bool:
        """Decide whether to keep going after a failed action.

        The failure itself stays in the observations, so the next reasoning
        turn sees it and can adapt. We give up after too many in a row.
        """
        failures = state.consecutive_failures()
        self.log.log(self.name, "failure", consecutive=failures)
        return failures < self.max_consecutive_failures

    @staticmethod
    def is_repeat(state: TaskState, decision: AgentDecision) -> bool:
        """True if this exact action already succeeded twice (a likely loop)."""
        same = [
            o for o in state.observations
            if o.decision.action == decision.action
            and o.decision.arguments == decision.arguments
            and o.result.success
        ]
        return len(same) >= 2
