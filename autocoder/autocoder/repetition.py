"""
Repetition detection (the "cycle/no-progress detection" the original
gameplan flagged as worth keeping and which was never actually built).

Deliberately narrow: this doesn't try to detect every possible loop, only
the specific, cheap-to-detect pattern seen in practice -- the SAME tool
hitting the SAME target (file path, or command) a run of times in a row
with no acceptance-check attempt in between. That's a strong, low-false-
-positive signal that the model is thrashing rather than making progress,
without fighting legitimate repeated reads/greps while exploring.
"""
from __future__ import annotations

import json
from typing import Any

DEFAULT_THRESHOLD = 4
DEFAULT_FORCE_ESCALATE_AFTER_FIRES = 3


def _target_key(name: str, tool_input: dict[str, Any]) -> str:
    """What counts as 'the same thing' for repetition purposes: the path for
    file tools, the command string for run_command, or a canonical dump of
    the whole input as a fallback. Content/old_str differences don't matter
    here -- hammering the same FILE repeatedly without checking is the
    signal, even if each edit attempt differs."""
    target = tool_input.get("path") or tool_input.get("command")
    if target is None:
        target = json.dumps(tool_input, sort_keys=True, default=str)
    return f"{name}:{target}"


class RepetitionGuard:
    def __init__(self, threshold: int = DEFAULT_THRESHOLD,
                 force_escalate_after_fires: int = DEFAULT_FORCE_ESCALATE_AFTER_FIRES):
        self.threshold = threshold
        self.force_escalate_after_fires = force_escalate_after_fires
        self._last_key: str | None = None
        self._consecutive_count: int = 0
        self._fire_count: int = 0

    def record(self, name: str, tool_input: dict[str, Any]) -> str | None:
        """Call after a tool is dispatched. Returns a warning string every
        `threshold`th consecutive call to the same target (4, 8, 12, ...) --
        not just the first time. A single lifetime warning per target proved
        insufficient in practice: the model can verbally acknowledge the
        notice ("I've written this file N times without reading it") and
        then continue the exact same pattern for many more calls with no
        further nudge, since nothing was left to catch it. Re-firing
        periodically keeps correcting a thrash that doesn't self-resolve."""
        key = _target_key(name, tool_input)
        if key == self._last_key:
            self._consecutive_count += 1
        else:
            self._last_key = key
            self._consecutive_count = 1

        if self._consecutive_count % self.threshold == 0:
            self._fire_count += 1
            target = key.split(":", 1)[1] if ":" in key else key
            return (
                f"[REPETITION NOTICE] You have called {name} on '{target}' "
                f"{self._consecutive_count} times in a row without an acceptance check "
                "in between. Stop and verify your actual current state first -- read "
                "the file back, or run a real check -- before making further "
                "changes to it blind."
            )
        return None

    def should_force_escalate(self) -> bool:
        """True once the notice has fired repeatedly for the same thrash
        without the model self-correcting. Confirmed in practice: the
        notice fired 3 separate times (at 4, 8, and 12 repeats) and the
        model kept repeating anyway, well past a point where a text nudge
        was clearly not going to work -- the run went on to hit a
        context-overflow crash shortly after. Past this many ignored
        warnings, escalate to a human instead of sending yet another
        nudge that has already been shown not to work."""
        return self._fire_count >= self.force_escalate_after_fires

    def checkpoint(self) -> None:
        """Call after any acceptance-check attempt (pass or fail) -- that's
        a genuine verification step, so the repetition count resets rather
        than penalizing a model that's correctly iterating: try, check,
        adjust, check again."""
        self._last_key = None
        self._consecutive_count = 0
        self._fire_count = 0
