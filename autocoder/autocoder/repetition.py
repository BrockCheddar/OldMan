"""
Repetition detection (the "cycle/no-progress detection" the original
gameplan flagged as worth keeping and which was never actually built).

Two independent detectors:
  1. Consecutive: the SAME tool hitting the SAME target back-to-back N
     times with no acceptance-check attempt in between.
  2. Cyclic: a small rotating SET of targets (e.g. the same ~10 files,
     reread in a different order each lap) filling an entire rolling
     window with no acceptance-check attempt in between. Confirmed needed
     in practice: a model rotated through the same ~12 files in varying
     order, never repeating the exact same target 4x in a row, so detector
     1 alone never fired again after its first hit -- the model was
     effectively invisible to consecutive-only detection despite making
     zero real progress for dozens of turns.
Both are deliberately narrow and cheap rather than trying to catch every
possible loop shape, to keep the false-positive rate low against
legitimate repeated reads/greps while exploring.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Any

DEFAULT_THRESHOLD = 4
DEFAULT_FORCE_ESCALATE_AFTER_FIRES = 3

# Cyclic detector: if a full window of this many calls touches at most
# this many distinct targets with zero checkpoints in between, that's a
# rotating thrash rather than varied exploration (which would touch many
# more distinct targets over the same span). Tuned against a real captured
# thrash (8 files, reshuffled each lap) and a real broad-exploration case
# (25 distinct files, no repeats) -- window=30/max=15 (half the window)
# catches the former at the first full window and never fires on the
# latter.
DEFAULT_CYCLE_WINDOW = 30
DEFAULT_CYCLE_DISTINCT_MAX = 15
DEFAULT_CYCLE_FORCE_ESCALATE_AFTER_FIRES = 2


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
                 force_escalate_after_fires: int = DEFAULT_FORCE_ESCALATE_AFTER_FIRES,
                 cycle_window: int = DEFAULT_CYCLE_WINDOW,
                 cycle_distinct_max: int = DEFAULT_CYCLE_DISTINCT_MAX,
                 cycle_force_escalate_after_fires: int = DEFAULT_CYCLE_FORCE_ESCALATE_AFTER_FIRES):
        self.threshold = threshold
        self.force_escalate_after_fires = force_escalate_after_fires
        self._last_key: str | None = None
        self._consecutive_count: int = 0
        self._fire_count: int = 0

        self.cycle_distinct_max = cycle_distinct_max
        self.cycle_force_escalate_after_fires = cycle_force_escalate_after_fires
        self._window: deque[str] = deque(maxlen=cycle_window)
        self._cycle_fire_count: int = 0

    def record(self, name: str, tool_input: dict[str, Any]) -> str | None:
        """Call after a tool is dispatched. Returns a warning string every
        `threshold`th consecutive call to the same target (4, 8, 12, ...) --
        not just the first time. A single lifetime warning per target proved
        insufficient in practice: the model can verbally acknowledge the
        notice ("I've written this file N times without reading it") and
        then continue the exact same pattern for many more calls with no
        further nudge, since nothing was left to catch it. Re-firing
        periodically keeps correcting a thrash that doesn't self-resolve.

        Also feeds the cyclic detector's rolling window regardless of
        whether the consecutive check fires -- if BOTH fire on the same
        call, both notices are concatenated so nothing is silently
        dropped."""
        key = _target_key(name, tool_input)
        if key == self._last_key:
            self._consecutive_count += 1
        else:
            self._last_key = key
            self._consecutive_count = 1

        consecutive_warning = None
        if self._consecutive_count % self.threshold == 0:
            self._fire_count += 1
            target = key.split(":", 1)[1] if ":" in key else key
            consecutive_warning = (
                f"[REPETITION NOTICE] You have called {name} on '{target}' "
                f"{self._consecutive_count} times in a row without an acceptance check "
                "in between. Stop and verify your actual current state first -- read "
                "the file back, or run a real check -- before making further "
                "changes to it blind. If you're not sure your current approach even "
                "works, consider discard_and_restart instead of guessing again."
            )

        cyclic_warning = None
        self._window.append(key)
        if len(self._window) == self._window.maxlen:
            distinct = len(set(self._window))
            if distinct <= self.cycle_distinct_max:
                self._cycle_fire_count += 1
                cyclic_warning = (
                    f"[CYCLIC REPETITION NOTICE] Your last {len(self._window)} tool calls "
                    f"touched only {distinct} distinct target(s), in rotation, with no "
                    "acceptance-check attempt in between. Reordering which file you read "
                    "next doesn't count as progress -- you're re-reading the same small "
                    "set of things. Stop and either call mark_step_done / propose_step "
                    "with what you already know, or discard_and_restart if the approach "
                    "isn't working."
                )
                self._window.clear()  # avoid firing again on every subsequent call

        if consecutive_warning and cyclic_warning:
            return f"{consecutive_warning}\n\n{cyclic_warning}"
        return consecutive_warning or cyclic_warning

    def should_force_escalate(self) -> bool:
        """True once either detector has fired repeatedly without the model
        self-correcting. Consecutive: confirmed in practice, notice fired 3
        separate times and the model kept repeating anyway. Cyclic: lower
        threshold (2) since each fire already represents a full window's
        worth (16 calls) of confirmed non-progress, a much stronger signal
        per-fire than the consecutive case."""
        return (self._fire_count >= self.force_escalate_after_fires
                or self._cycle_fire_count >= self.cycle_force_escalate_after_fires)

    def checkpoint(self) -> None:
        """Call after any acceptance-check attempt (pass or fail) -- that's
        a genuine verification step, so both detectors reset rather than
        penalizing a model that's correctly iterating: try, check, adjust,
        check again."""
        self._last_key = None
        self._consecutive_count = 0
        self._fire_count = 0
        self._window.clear()
        self._cycle_fire_count = 0
