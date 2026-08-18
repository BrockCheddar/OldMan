"""
Per-workspace verified lessons.

Deliberately narrow: a lesson is only recorded when a failure was followed,
in the SAME run, by a confirmed successful outcome (an acceptance check
passing, or the goal being accepted as done). We never record raw failures
on their own -- an unverified failure is just noise, and worse, the model's
own account of "what went wrong" can be a hallucination (seen in practice:
fabricated exception types that don't exist in this codebase). Recording
those as "lessons" would teach future runs to trust fabricated patterns.

Scope is per-workspace (.autocoder/lessons.json), not global -- a fact
learned building one project ("this package name collides with an installed
library") has no business leaking into an unrelated project's context.

Capped and summarized on injection so this can only ever cost a small,
bounded amount of context, never grow into a second transcript competing
with the actual task for the model's limited attention.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

MAX_LESSONS_STORED = 20
DEFAULT_SUMMARY_COUNT = 8
MAX_FIELD_CHARS = 1000


@dataclass
class Lesson:
    context: str        # e.g. "step: Create the contact book package structure"
    symptom: str         # what failed / got stuck (truncated)
    fix: str              # what the eventual successful outcome was (truncated)
    recorded_at: float = 0.0


class LessonsStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[Lesson]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        raw = data.get("lessons", [])
        return [Lesson(**{k: v for k, v in l.items() if k in Lesson.__dataclass_fields__}) for l in raw]

    def add(self, context: str, symptom: str, fix: str) -> None:
        if not symptom.strip() or not fix.strip():
            return  # nothing worth recording
        lessons = self.load()
        lessons.append(Lesson(
            context=context[:200],
            symptom=symptom[:MAX_FIELD_CHARS],
            fix=fix[:MAX_FIELD_CHARS],
            recorded_at=time.time(),
        ))
        lessons = lessons[-MAX_LESSONS_STORED:]  # cap: drop oldest first
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"lessons": [asdict(l) for l in lessons]}, indent=2),
            encoding="utf-8",
        )

    def summary_text(self, max_lessons: int = DEFAULT_SUMMARY_COUNT) -> str:
        lessons = self.load()[-max_lessons:]
        if not lessons:
            return "(none yet)"
        lines = []
        for l in lessons:
            lines.append(f"- [{l.context}] got stuck on: {l.symptom[:250]}\n  what eventually worked: {l.fix[:250]}")
        return "\n".join(lines)
