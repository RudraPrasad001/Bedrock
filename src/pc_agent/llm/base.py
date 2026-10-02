"""Provider-agnostic LLM interface. The rest of the app depends only on this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    # Set when the model emitted a native tool call instead of text, as
    # {"name": str, "arguments": dict}. Agents decide how to interpret it.
    tool_call: dict | None = None


class LLMClient(ABC):
    model: str = ""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return the assistant reply for a chat-style conversation.

        `messages` use the {"role": "user"|"assistant", "content": str} shape.
        When `json_mode` is set the provider should constrain output to a
        single JSON object.
        """
