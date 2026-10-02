"""Runtime configuration, loaded from environment variables and CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Config:
    provider: str = "groq"
    model: str = DEFAULT_MODELS["groq"]
    temperature: float = 0.1

    dry_run: bool = False
    verbose: bool = False
    color: bool = True

    allow_python: bool = False  # python.execute is opt-in
    allow_destructive: bool = False  # filesystem.delete is opt-in

    # Filesystem tools may only touch paths below these roots.
    allowed_paths: list[Path] = field(default_factory=list)
    log_dir: Path = Path("logs")

    max_steps: int = 20  # reasoner/executor iterations per cycle
    max_verification_retries: int = 2
    max_consecutive_failures: int = 4

    @classmethod
    def from_env(cls) -> "Config":
        from dotenv import load_dotenv

        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        allowed = os.getenv("PC_AGENT_ALLOWED_PATHS")
        if allowed:
            allowed_paths = [
                Path(p).expanduser().resolve() for p in allowed.split(os.pathsep) if p
            ]
        else:
            allowed_paths = [Path.home().resolve(), Path.cwd().resolve()]
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            allow_python=_env_bool("PC_AGENT_ALLOW_PYTHON"),
            allow_destructive=_env_bool("PC_AGENT_ALLOW_DESTRUCTIVE"),
            allowed_paths=allowed_paths,
            log_dir=Path(os.getenv("PC_AGENT_LOG_DIR", "logs")).expanduser(),
            max_steps=_env_int("PC_AGENT_MAX_STEPS", 20),
        )
