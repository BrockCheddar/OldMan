"""
Per-workspace durable decisions.

Unlike lessons.py, a decision is not captured automatically by the harness --
the model has to choose to call record_decision when it makes a real
architecture/technology/convention choice a LATER step needs to stay
consistent with. What this module guarantees is PROPAGATION, not capture:
once record_decision is called, the decision reliably reaches every future
outer (planner) and inner (coder) system prompt, mechanically, with no
model relay required. See decisions_store_gameplan.txt LIMITATION section
for why capture itself is a separate, unsolved problem.

Scope is per-workspace (.autocoder/decisions.json), not global -- same
reasoning as lessons.py: a decision made building one project has no
business leaking into an unrelated project's context.

Lifecycle is a single `active: bool`, not a state machine. Recording a new
decision with `supersedes` set to an earlier decision's id flips that
entry's active to False. Superseded entries stay in the JSON file (full
history preserved) but drop out of summary_text() (so they never
masquerade as current).

Capped and summarized on injection, same discipline as lessons.py -- this
can only ever cost a small, bounded amount of context. The cap eviction
differs from lessons.py's plain FIFO, though: lessons have no notion of
"still relevant", but a decision does. Eviction prefers dropping the
oldest INACTIVE (already-superseded) entries first, and only touches an
active entry if the store is still over cap after every inactive entry is
gone -- so a live, currently-relevant decision doesn't silently fall off
the end the way a plain FIFO cap would let it.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

MAX_DECISIONS_STORED = 20
DEFAULT_SUMMARY_COUNT = 8
MAX_FIELD_CHARS = 500
MAX_CONTEXT_CHARS = 200


@dataclass
class Decision:
    id: str
    decision: str              # e.g. "Use SQLAlchemy ORM, not raw sqlite3"
    originating_step: str      # context only, not enforced/validated
    recorded_at: float = 0.0
    active: bool = True
    supersedes: str | None = None  # id of the decision this replaces, if any


@dataclass
class AddResult:
    id: str
    # True: supersedes was given and matched a real entry.
    # False: supersedes was given but didn't match anything (kept as a new,
    #        non-superseding decision -- the caller should be told so it
    #        can retry with a correct id rather than silently losing the link).
    # None: no supersedes was given.
    superseded_found: bool | None


class DecisionsStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[Decision]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        raw = data.get("decisions", [])
        return [Decision(**{k: v for k, v in d.items() if k in Decision.__dataclass_fields__}) for d in raw]

    def add(self, decision: str, originating_step: str = "", supersedes: str | None = None) -> AddResult | None:
        if not decision.strip():
            return None  # nothing worth recording
        decisions = self.load()

        superseded_found: bool | None = None
        if supersedes:
            superseded_found = False
            for d in decisions:
                if d.id == supersedes and d.active:
                    d.active = False
                    superseded_found = True
                    break

        new = Decision(
            id=uuid.uuid4().hex[:8],
            decision=decision.strip()[:MAX_FIELD_CHARS],
            originating_step=originating_step[:MAX_CONTEXT_CHARS],
            recorded_at=time.time(),
            active=True,
            supersedes=supersedes if superseded_found else None,
        )
        decisions.append(new)
        decisions = self._evict_to_cap(decisions)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"decisions": [asdict(d) for d in decisions]}, indent=2),
            encoding="utf-8",
        )
        return AddResult(id=new.id, superseded_found=superseded_found)

    @staticmethod
    def _evict_to_cap(decisions: list[Decision]) -> list[Decision]:
        """Drop oldest-first, but only ever touch inactive (already
        superseded) entries unless the store is still over cap once every
        inactive entry is gone -- see module docstring."""
        result = list(decisions)
        while len(result) > MAX_DECISIONS_STORED:
            inactive_positions = [i for i, d in enumerate(result) if not d.active]
            del result[inactive_positions[0] if inactive_positions else 0]
        return result

    def summary_text(self, max_decisions: int = DEFAULT_SUMMARY_COUNT) -> str:
        active = [d for d in self.load() if d.active]
        active = active[-max_decisions:]
        if not active:
            return "(none yet)"
        lines = [f"- [{d.id}] {d.decision}" for d in active]
        return "\n".join(lines)
