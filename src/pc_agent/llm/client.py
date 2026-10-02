"""Concrete LLM providers and the factory used by the app."""

from __future__ import annotations

import json
import os
import time

from pc_agent.config import Config
from pc_agent.llm.base import LLMClient, LLMError, LLMResponse


class GroqClient(LLMClient):
    def __init__(self, model: str, api_key: str, temperature: float = 0.1, max_retries: int = 2):
        try:
            from groq import Groq  # lazy import keeps CLI startup fast
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'groq' package is not installed") from exc
        self.model = model
        self.temperature = temperature
        self._client = Groq(api_key=api_key, max_retries=max_retries, timeout=60)

    def generate(self, system_prompt, messages, tools=None, json_mode=False) -> LLMResponse:
        import groq

        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
        for attempt in range(3):
            try:
                completion = self._client.chat.completions.create(**kwargs)
                break
            except groq.RateLimitError as exc:
                if attempt == 2:
                    raise LLMError(f"Rate limited by Groq: {exc}") from exc
                time.sleep(2 * (attempt + 1))
            except groq.BadRequestError as exc:
                # Some models (e.g. gpt-oss) emit a native tool call when the prompt
                # describes tools. Groq rejects it but returns the generation.
                tool_call = failed_tool_call(exc.body)
                if tool_call is not None:
                    return LLMResponse(content="", model=self.model, tool_call=tool_call)
                # JSON mode occasionally rejects a malformed generation; retry once without it.
                if json_mode and "json" in str(exc).lower() and attempt == 0:
                    kwargs.pop("response_format", None)
                    continue
                raise LLMError(f"Groq rejected the request: {exc}") from exc
            except groq.APIError as exc:
                raise LLMError(f"Groq API error: {exc}") from exc
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
            },
        )


def failed_tool_call(body: object) -> dict | None:
    """Extract the tool call from a Groq `tool_use_failed` error body, if any."""
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict) or error.get("code") != "tool_use_failed":
        return None
    try:
        call = json.loads(error.get("failed_generation") or "")
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    arguments = call.get("arguments") or call.get("parameters") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return {"name": call["name"], "arguments": arguments if isinstance(arguments, dict) else {}}


def create_client(config: Config) -> LLMClient:
    """Build the LLM client selected by LLM_PROVIDER. API keys never leave this function."""
    if config.provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        if not config.model:
            raise LLMError("LLM_MODEL is not set")
        return GroqClient(config.model, api_key, temperature=config.temperature)
    raise LLMError(f"Unsupported LLM_PROVIDER '{config.provider}'. Supported: groq")
