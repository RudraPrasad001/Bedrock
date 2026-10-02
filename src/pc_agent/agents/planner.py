"""Planner Agent: turns a request into a structured plan. Never executes tools."""

from __future__ import annotations

from pc_agent.agents.base import LLMAgent, environment_context
from pc_agent.core.decisions import Plan
from pc_agent.core.state import TaskState

SYSTEM_PROMPT = """You are the PLANNER of a local computer task agent.
Your only job is to understand the user's task and produce a short, concrete plan.
You cannot run tools; other agents will execute the plan using the tools listed below.

Rules:
- Identify the desired outcome as a one-sentence goal.
- Break the task into 2-6 high-level steps, in order. Fewer is better.
- For each step, name the tool most likely to be used (from the list) or null.
- List risky or destructive operations (deleting, moving, overwriting files, privileged commands) in "risks".
- Resolve vague locations to concrete paths (e.g. "Downloads" -> "~/Downloads").
- If a report/file is requested without a location, choose a sensible path and say so in the step.

Available tools:
{tools}

Do not use function/tool calling. Respond with ONLY a JSON object, as plain text:
{{"goal": "...", "steps": [{{"id": 1, "description": "...", "expected_tool": "filesystem.search"}}], "risks": []}}"""


class PlannerAgent(LLMAgent):
    name = "planner"

    def __init__(self, llm, tool_catalog: str, log=None):
        super().__init__(llm, log)
        self.system_prompt = SYSTEM_PROMPT.format(tools=tool_catalog)

    def run(self, state: TaskState) -> Plan:
        prompt = f"{environment_context()}\n\nUser task:\n{state.user_request}"
        plan = self.ask_json(prompt, Plan)
        for index, step in enumerate(plan.steps, start=1):
            step.id = index  # normalise ids regardless of model output
        self.log.log(self.name, "plan", plan=plan.model_dump())
        return plan
