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

    def _ensure_session_dir(self) -> None:
        """Recreates .autocoder/ if it's gone missing mid-run -- e.g. a
        `git clean -x` (which removes gitignored files too, defeating the
        .gitignore entry Workspace already adds for this exact directory),
        a human's manual cleanup, or antivirus/disk-cleanup tooling. Without
        this, every write for the rest of the process raises
        FileNotFoundError and crashes the whole run on the very next save
        -- a much worse outcome than silently starting a fresh, empty
        .autocoder/ and continuing (the in-memory RunState the caller is
        about to write IS the source of truth at that point; recreating
        the directory loses nothing that wasn't already gone)."""
        self.config.session_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: RunState) -> None:
        self._ensure_session_dir()
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
        self._ensure_session_dir()
        record = {"ts": time.time(), "kind": kind, **data}
        with open(self.config.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
