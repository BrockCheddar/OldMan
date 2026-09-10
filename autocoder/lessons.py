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

Lifecycle matches decisions.py: a single `active: bool`, not a state
machine. Recording a new lesson with `supersedes` set to an earlier
lesson's id flips that entry's active to False. Superseded entries stay in
the JSON file (full history preserved) but drop out of summary_text() (so
a lesson verified once doesn't masquerade as still-current after a better
fix is found). Eviction at cap prefers dropping the oldest INACTIVE entry
first, and only touches an active entry if the store is still over cap
once every inactive entry is gone -- see decisions.py for the identical
reasoning.

FLAW 11 (intent): symptom+fix alone can look like a match for a NEW
failure that is actually unrelated -- two failures can share surface
symptoms ("tests failing") while the fix that was actually correct only
applied because of what the run was trying to do at the time. `intent`
records that original goal/context so a reader (human or model) can judge
relevance, not just pattern-match on symptom text.

FLAW 13 (auto-matching): lessons used to only ever be dumped as the most
recent N, for the model to read and judge relevance itself every time.
find_relevant() does simple, deterministic keyword-overlap scoring against
a caller-supplied query (typically the current step's title+objective, or
an escalation's failure reason) so the lessons actually shown are the ones
plausibly about the SAME problem, not just the most recent ones. This is
intentionally not fuzzy/ML matching -- deterministic and free, since this
harness has to work against a small local model with no spare capacity
for an extra classification call.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

MAX_LESSONS_STORED = 20
DEFAULT_SUMMARY_COUNT = 8
MAX_FIELD_CHARS = 1000

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "not", "no", "as", "by", "from", "into", "over",
    "step", "run", "build", "add", "test", "tests",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


@dataclass
class Lesson:
    context: str        # e.g. "step: Create the contact book package structure"
    symptom: str         # what failed / got stuck (truncated)
    fix: str              # what the eventual successful outcome was (truncated)
    recorded_at: float = 0.0
    id: str = ""
    active: bool = True
    supersedes: str | None = None  # id of the lesson this replaces, if any
    intent: str = ""      # FLAW 11: the goal/context that made `fix` correct


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

    def add(self, context: str, symptom: str, fix: str, supersedes: str | None = None,
             intent: str = "") -> None:
        if not symptom.strip() or not fix.strip():
            return  # nothing worth recording
        lessons = self.load()

        superseded_found = False
        if supersedes:
            for l in lessons:
                if l.id == supersedes and l.active:
                    l.active = False
                    superseded_found = True
                    break

        lessons.append(Lesson(
            context=context[:200],
            symptom=symptom[:MAX_FIELD_CHARS],
            fix=fix[:MAX_FIELD_CHARS],
            recorded_at=time.time(),
            id=uuid.uuid4().hex[:8],
            active=True,
            supersedes=supersedes if superseded_found else None,
            intent=intent[:300],
        ))
        lessons = self._evict_to_cap(lessons)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"lessons": [asdict(l) for l in lessons]}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _evict_to_cap(lessons: list[Lesson]) -> list[Lesson]:
        """Drop oldest-first, but only ever touch inactive (already
        superseded) entries unless the store is still over cap once every
        inactive entry is gone -- see module docstring."""
        result = list(lessons)
        while len(result) > MAX_LESSONS_STORED:
            inactive_positions = [i for i, l in enumerate(result) if not l.active]
            del result[inactive_positions[0] if inactive_positions else 0]
        return result

    def summary_text(self, max_lessons: int = DEFAULT_SUMMARY_COUNT) -> str:
        active = [l for l in self.load() if l.active]
        active = active[-max_lessons:]
        return self.render(active)

    def find_relevant(self, query: str, max_results: int = 5) -> list[Lesson]:
        """FLAW 13: score every active lesson's (context+intent+symptom)
        keyword overlap against `query`, return the top matches (score > 0
        only) ranked by score then recency. An empty result means "nothing
        plausibly related" -- callers should fall back to summary_text()'s
        recent-N in that case, not show nothing."""
        query_words = _keywords(query)
        if not query_words:
            return []
        scored: list[tuple[int, float, Lesson]] = []
        for l in self.load():
            if not l.active:
                continue
            lesson_words = _keywords(f"{l.context} {l.intent} {l.symptom}")
            score = len(query_words & lesson_words)
            if score > 0:
                scored.append((score, l.recorded_at, l))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [l for _, _, l in scored[:max_results]]

    @staticmethod
    def render(lessons: list[Lesson]) -> str:
        if not lessons:
            return "(none yet)"
        lines = []
        for l in lessons:
            intent_line = f"\n  intent at the time: {l.intent[:200]}" if l.intent else ""
            lines.append(
                f"- [{l.context}] got stuck on: {l.symptom[:250]}{intent_line}\n"
                f"  what eventually worked: {l.fix[:250]}"
            )
        return "\n".join(lines)
