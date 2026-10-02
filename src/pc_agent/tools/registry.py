"""Central tool registry.

The LLM can only request tools that are registered here. Arguments are
validated with each tool's Pydantic model before anything executes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationError

from pc_agent.config import Config
from pc_agent.core.decisions import PermissionAssessment, RiskLevel, ToolResult
from pc_agent.core.permissions import PathPolicy


# Argument names that hold filesystem paths and must stay inside the sandbox.
PATH_FIELDS = ("path", "source", "destination", "cwd")


class ToolArgs(BaseModel):
    """Base class for tool argument models. Unknown arguments are rejected."""

    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolContext:
    config: Config
    path_policy: PathPolicy


ToolFunc = Callable[[Any, ToolContext], ToolResult]
AssessFunc = Callable[[Any, ToolContext], PermissionAssessment]


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[ToolArgs]
    func: ToolFunc
    risk: RiskLevel
    assess: AssessFunc | None = None  # per-call risk; defaults to static `risk`
    enabled: bool = True
    disabled_reason: str = ""
    sandboxed: bool = True  # False only for tools that read filesystem metadata, never contents

    def assess_call(self, args: ToolArgs, ctx: ToolContext) -> PermissionAssessment:
        # Path arguments are sandbox-checked up front, so a blocked path is
        # reported as "not allowed" rather than failing mid-execution.
        for field_name in PATH_FIELDS if self.sandboxed else ():
            value = getattr(args, field_name, None)
            if isinstance(value, str):
                ctx.path_policy.resolve(value)  # raises PathPolicyError
        if self.assess is not None:
            return self.assess(args, ctx)
        needs_confirmation = self.risk in (RiskLevel.DESTRUCTIVE, RiskLevel.PRIVILEGED)
        return PermissionAssessment(
            allowed=True, risk=self.risk, requires_confirmation=needs_confirmation
        )

    def signature(self) -> str:
        """Compact argument description for LLM prompts."""
        parts = []
        for name, info in self.args_model.model_fields.items():
            annotation = _type_name(info.annotation)
            if info.is_required():
                parts.append(f"{name}: {annotation}")
            else:
                parts.append(f"{name}: {annotation} = {json.dumps(info.default)}")
        return f"{self.name}({', '.join(parts)})"

    def argument_notes(self) -> list[str]:
        return [
            f"{name}: {info.description}"
            for name, info in self.args_model.model_fields.items()
            if info.description
        ]


def _type_name(annotation: Any) -> str:
    if get_origin(annotation) is Literal:
        return "|".join(json.dumps(v) for v in get_args(annotation))
    name = getattr(annotation, "__name__", None)
    if name and not get_args(annotation):
        return name
    return str(annotation).replace("typing.", "")


class ToolValidationError(ValueError):
    pass


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolValidationError(
                f"Unknown tool '{name}'. Available tools: {', '.join(self.enabled_names())}"
            )
        if not tool.enabled:
            raise ToolValidationError(f"Tool '{name}' is disabled: {tool.disabled_reason}")
        return tool

    def enabled_names(self) -> list[str]:
        return [name for name, tool in self.tools.items() if tool.enabled]

    def validate(self, name: str, arguments: dict[str, Any]) -> tuple[Tool, ToolArgs]:
        tool = self.get(name)
        if not isinstance(arguments, dict):
            raise ToolValidationError("Tool arguments must be an object")
        try:
            return tool, tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolValidationError(f"Invalid arguments for {name}: {problems}") from None

    def catalog(self, detailed: bool = True) -> str:
        """Human/LLM readable list of enabled tools."""
        lines = []
        for tool in self.tools.values():
            if not tool.enabled:
                continue
            lines.append(f"- {tool.signature()} [{tool.risk.value}]\n    {tool.description}")
            if detailed:
                lines += [f"    · {note}" for note in tool.argument_notes()]
        return "\n".join(lines)

    def read_only(self) -> "ToolRegistry":
        """A view of this registry exposing only side-effect-free tools."""
        return ToolRegistry(
            {n: t for n, t in self.tools.items() if t.risk == RiskLevel.READ and t.enabled}
        )


def build_registry(config: Config) -> ToolRegistry:
    """Create the registry with every built-in tool."""
    from pc_agent.tools import filesystem, python, shell, system

    registry = ToolRegistry()
    for module in (filesystem, system, shell, python):
        for tool in module.tools(config):
            registry.register(tool)
    return registry
