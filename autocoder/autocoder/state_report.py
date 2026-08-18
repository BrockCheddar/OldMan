"""
State-report compaction.

Old design (still used for light nudges, see llm.trim_history_to_fit):
cuts old tool_result text down to a head piece and a tail piece, with a
placeholder in between. This keeps SOME text, but the kept text is often
not useful on its own, and the model must guess what the missing middle
held.

New design (used here, for the coder's inner loop): once history grows
past a small "recent" window, replace everything older than that window
with ONE fresh block -- a state report built entirely from live checks
against the real workspace (current file listing, current content of the
files in scope, current git status, the most recent acceptance-check
result). Never from old chat text. This can never be stale, because it is
rebuilt fresh, from the real world, every time it is used.

This trades the STORY of how we got here for a clean, true picture of
where we are now. Past attempts that failed are still remembered
separately and more durably, via the `lessons` system (verified,
carried across whole runs) -- this module is not a replacement for that.
"""
from __future__ import annotations

from .llm import Turn


def compact_history_with_state_report(
    history: list[Turn],
    state_report: str,
    keep_recent_turns: int = 4,
) -> list[Turn]:
    """
    Returns a NEW history list: the first turn (the step's seed instruction)
    is kept, then one fresh Turn holding `state_report`, then the most
    recent `keep_recent_turns` turns are kept as-is for immediate
    continuity (so the model still sees exactly what it just did).
    Everything else -- the "ancient middle" -- is dropped, not truncated.

    If history is already short enough that there's nothing old to drop,
    returns history unchanged.
    """
    if len(history) <= keep_recent_turns + 1:
        return history

    seed = history[0]
    recent = history[-keep_recent_turns:]
    report_turn = Turn(role="user", text=state_report)
    return [seed, report_turn, *recent]
