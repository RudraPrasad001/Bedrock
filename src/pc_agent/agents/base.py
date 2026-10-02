"""Shared plumbing for LLM-backed agents."""

from __future__ import annotations

import json
import os
import platform
import re
from datetime import date
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from pc_agent.core.events import EventLog, NullLog
from pc_agent.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)

JSON_RETRIES = 2


class AgentError(RuntimeError):
    pass


def environment_context() -> str:
    return (
        f"Today: {date.today().isoformat()}. OS: {platform.system()} {platform.release()}. "
        f"Home: {Path.home()}. Working directory: {os.getcwd()}."
    )


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences and chatter."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


class LLMAgent:
    """An agent whose behaviour is defined by its system prompt and output schema."""

    name: str = "agent"
    system_prompt: str = ""

    def __init__(self, llm: LLMClient, log: EventLog | None = None):
        self.llm = llm
        self.log = log or NullLog()

    def ask_json(self, prompt: str, schema: type[T]) -> T:
        """Ask the LLM for a JSON object and validate it, retrying on bad output."""
        messages = [{"role": "user", "content": prompt}]
        last_error = ""
        for attempt in range(JSON_RETRIES + 1):
            response = self.llm.generate(self.system_prompt, messages, json_mode=True)
            self.log.log(self.name, "llm_response", attempt=attempt, usage=response.usage,
                         tool_call=response.tool_call)
            try:
                if response.tool_call is not None:
                    data = self.from_tool_call(response.tool_call)
                    if data is None:
                        raise ValueError("you called a tool; do not call tools, reply with JSON text")
                else:
                    data = extract_json(response.content)
                return schema.model_validate(data)
            except (ValueError, ValidationError) as exc:
                # ValidationError subclasses ValueError, so check it first.
                last_error = _brief(exc) if isinstance(exc, ValidationError) else str(exc)
                self.log.log(self.name, "invalid_output", error=last_error, content=response.content[:2000])
                messages += [
                    {"role": "assistant", "content": response.content or json.dumps(response.tool_call)},
                    {
                        "role": "user",
                        "content": f"That reply was invalid ({last_error}). "
                        "Respond again with ONLY one valid JSON object in the required format.",
                    },
                ]
        raise AgentError(f"{self.name} produced invalid output: {last_error}")

    def from_tool_call(self, call: dict) -> dict | None:
        """Map a native tool call to this agent's output schema. None = not allowed."""
        return None

    def ask_text(self, prompt: str) -> str:
        response = self.llm.generate(self.system_prompt, [{"role": "user", "content": prompt}])
        self.log.log(self.name, "llm_response", usage=response.usage)
        return response.content.strip()


def _brief(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'object'}: {e['msg']}" for e in exc.errors()
    )
