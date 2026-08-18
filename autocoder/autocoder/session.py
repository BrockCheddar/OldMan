"""
Persistence: RunState is written to disk after every completed step and every
escalation, so a crash or Ctrl-C loses at most one in-progress step.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .config import Config
from .planner import RunState


class SessionStore:
    def __init__(self, config: Config):
        self.config = config
        self.config.session_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: RunState) -> None:
        self.config.session_file.write_text(
            json.dumps(state.to_dict(), indent=2), encoding="utf-8"
        )

    def load_state(self) -> RunState | None:
        if not self.config.session_file.exists():
            return None
        return RunState.from_dict(
            json.loads(self.config.session_file.read_text(encoding="utf-8"))
        )

    def log_event(self, kind: str, data: dict[str, Any]) -> None:
        record = {"ts": time.time(), "kind": kind, **data}
        with open(self.config.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
