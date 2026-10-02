"""JSONL execution log. One file per task, one line per event."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pc_agent.core.permissions import redact


class EventLog:
    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_task(cls, log_dir: Path, task_id: str) -> "EventLog":
        return cls(log_dir / f"task-{task_id}.jsonl")

    def log(self, agent: str, event: str, **data: Any) -> None:
        if self.path is None:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "agent": agent,
            "event": event,
            **data,
        }
        line = redact(json.dumps(record, default=str, ensure_ascii=False))
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass  # logging must never break the agent


class NullLog(EventLog):
    def __init__(self) -> None:
        super().__init__(None)
