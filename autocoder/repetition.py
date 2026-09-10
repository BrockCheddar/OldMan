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

# Stall detector: catches what the other two structurally cannot. Confirmed
# needed in practice -- a real run touched close to 20 distinct files across
# a single step (list_dir, several read_files, several ast/regex probe
# scripts via run_command), never hit the same exact target 4x running, and
# never sat in a narrow enough rotation to trip the cyclic window either --
# genuinely broad, always-technically-new-looking exploration, for dozens of
# calls, with zero acceptance-check attempts the entire time. Consecutive
# and cyclic both key off REPEATING something; this keys off the total
# count since the last real checkpoint, so it doesn't matter whether the
# model keeps finding new things to look at -- if none of it has been
# checked against reality in a long time, that's the signal, independent of
# whether any single target repeats.
DEFAULT_STALL_THRESHOLD = 25
DEFAULT_STALL_FORCE_ESCALATE_AFTER_FIRES = 2

_STALL_WARNING_TEMPLATE = (
    "[STALL NOTICE] {n} tool calls with no acceptance-check attempt (mark_step_done "
    "or declare_done, pass or fail) succeeding or failing in between. Constantly "
    "varying what you look at avoids the other repetition detectors, but it still "
    "isn't progress if none of it has been checked against reality. Propose a step "
    "and call mark_step_done with what you already know, or declare_done if the "
    "goal-level check already passes -- don't keep exploring indefinitely."
)


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
                 cycle_force_escalate_after_fires: int = DEFAULT_CYCLE_FORCE_ESCALATE_AFTER_FIRES,
                 stall_threshold: int = DEFAULT_STALL_THRESHOLD,
                 stall_force_escalate_after_fires: int = DEFAULT_STALL_FORCE_ESCALATE_AFTER_FIRES):
        self.threshold = threshold
        self.force_escalate_after_fires = force_escalate_after_fires
        self._last_key: str | None = None
        self._consecutive_count: int = 0
        self._fire_count: int = 0

        self.cycle_distinct_max = cycle_distinct_max
        self.cycle_force_escalate_after_fires = cycle_force_escalate_after_fires
        self._window: deque[str] = deque(maxlen=cycle_window)
        self._cycle_fire_count: int = 0

        self.stall_threshold = stall_threshold
        self.stall_force_escalate_after_fires = stall_force_escalate_after_fires
        self._calls_since_checkpoint: int = 0
        self._stall_fire_count: int = 0

        # Tracks the offset of the last read_file call on the current
        # _last_key, so forward pagination through one large file isn't
        # mistaken for repetition. See _is_forward_pagination below.
        self._last_read_offset: int | None = None

    @staticmethod
    def _is_forward_pagination(name: str, tool_input: dict[str, Any],
                                same_key: bool, last_offset: int | None) -> bool:
        """True when this is a read_file call that picks up exactly where
        the previous call on the SAME file left off (or later) -- genuine
        progress through a large file, not the same region re-read. Confirmed
        necessary: shrinking read_file's default page size means a large
        file now legitimately takes several consecutive calls to get
        through, and the consecutive-repetition key is just the path (by
        design -- see _target_key), so without this check that pagination
        alone trips the same-target-N-times-in-a-row detector on its own,
        false-flagging the exact paginate-don't-dump-the-whole-file behavior
        the harness wants to encourage. Only forward, non-overlapping
        progress counts -- re-reading the same offset, or going backward,
        is still exactly the "not making progress" pattern the detector is
        supposed to catch, so those still count as repetition."""
        if not same_key or name != "read_file" or last_offset is None:
            return False
        offset = int(tool_input.get("offset", 1))
        return offset > last_offset

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
        same_key = (key == self._last_key)
        forward_pagination = self._is_forward_pagination(
            name, tool_input, same_key, self._last_read_offset)

        if name == "read_file":
            self._last_read_offset = int(tool_input.get("offset", 1))
        elif not same_key:
            self._last_read_offset = None

        if forward_pagination:
            # Real progress through the same file -- neither increment the
            # consecutive count nor feed the cyclic window as a repeat of
            # this key, but don't reset _consecutive_count to 0 either:
            # a model that paginates forward for a while and then genuinely
            # starts thrashing on the same file (offset stops advancing)
            # should still accumulate from where it left off, not get a
            # fresh grace period just because some of the reads happened to
            # be legitimate.
            self._last_key = key
            self._calls_since_checkpoint += 1
            stall_warning = None
            if self._calls_since_checkpoint % self.stall_threshold == 0:
                self._stall_fire_count += 1
                stall_warning = _STALL_WARNING_TEMPLATE.format(n=self._calls_since_checkpoint)
            return stall_warning
        if same_key:
            self._consecutive_count += 1
        else:
            self._last_key = key
            self._consecutive_count = 1
        self._calls_since_checkpoint += 1

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

        stall_warning = None
        if self._calls_since_checkpoint % self.stall_threshold == 0:
            self._stall_fire_count += 1
            stall_warning = _STALL_WARNING_TEMPLATE.format(n=self._calls_since_checkpoint)

        warnings = [w for w in (consecutive_warning, cyclic_warning, stall_warning) if w]
        return "\n\n".join(warnings) if warnings else None

    def should_force_escalate(self) -> bool:
        """True once any detector has fired repeatedly without the model
        self-correcting. Consecutive: confirmed in practice, notice fired 3
        separate times and the model kept repeating anyway. Cyclic: lower
        threshold (2) since each fire already represents a full window's
        worth (30 calls) of confirmed non-progress, a much stronger signal
        per-fire than the consecutive case. Stall: same reasoning as cyclic
        -- two fires means 50 calls with zero verification, regardless of
        how varied they looked."""
        return (self._fire_count >= self.force_escalate_after_fires
                or self._cycle_fire_count >= self.cycle_force_escalate_after_fires
                or self._stall_fire_count >= self.stall_force_escalate_after_fires)

    def checkpoint(self) -> None:
        """Call after any acceptance-check attempt (pass or fail) -- that's
        a genuine verification step, so all detectors reset rather than
        penalizing a model that's correctly iterating: try, check, adjust,
        check again."""
        self._last_key = None
        self._consecutive_count = 0
        self._fire_count = 0
        self._window.clear()
        self._cycle_fire_count = 0
        self._calls_since_checkpoint = 0
        self._stall_fire_count = 0
        self._last_read_offset = None
