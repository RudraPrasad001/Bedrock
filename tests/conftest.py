from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import pytest

from pc_agent.config import Config
from pc_agent.core.permissions import PathPolicy
from pc_agent.llm.base import LLMClient, LLMResponse
from pc_agent.tools.registry import ToolContext, build_registry


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


@pytest.fixture
def config(sandbox: Path, tmp_path: Path) -> Config:
    return Config(allowed_paths=[sandbox.resolve()], log_dir=tmp_path / "logs")


@pytest.fixture
def ctx(config: Config) -> ToolContext:
    return ToolContext(config=config, path_policy=PathPolicy(config.allowed_paths))


@pytest.fixture
def registry(config: Config):
    return build_registry(config)


AGENT_MARKERS = {
    "PLANNER": "planner",
    "REASONER": "reasoner",
    "VERIFIER": "verifier",
    "SUMMARIZER": "summarizer",
}


class ScriptedLLM(LLMClient):
    """Returns pre-scripted replies per agent, recording every prompt it sees."""

    model = "scripted"

    def __init__(self, script: dict[str, list]):
        self.queues = {agent: deque(replies) for agent, replies in script.items()}
        self.prompts: dict[str, list[str]] = defaultdict(list)

    def generate(self, system_prompt, messages, tools=None, json_mode=False):
        agent = next(a for marker, a in AGENT_MARKERS.items() if f"You are the {marker}" in system_prompt)
        self.prompts[agent].append(messages[-1]["content"])
        queue = self.queues.get(agent)
        if not queue:
            raise AssertionError(f"No scripted reply left for {agent}")
        reply = queue.popleft()
        if isinstance(reply, LLMResponse):
            return reply
        return LLMResponse(content=reply if isinstance(reply, str) else json.dumps(reply))


@pytest.fixture
def scripted():
    return ScriptedLLM
