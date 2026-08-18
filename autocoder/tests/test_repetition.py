import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.repetition import RepetitionGuard


def test_no_warning_below_threshold():
    guard = RepetitionGuard(threshold=4)
    for _ in range(3):
        result = guard.record("edit_file", {"path": "a.py", "old_str": "x", "new_str": "y"})
    assert result is None


def test_warning_fires_exactly_at_threshold():
    guard = RepetitionGuard(threshold=4)
    results = [guard.record("edit_file", {"path": "a.py", "old_str": "x", "new_str": str(i)}) for i in range(4)]
    assert results[:3] == [None, None, None]
    assert results[3] is not None
    assert "a.py" in results[3]
    assert "edit_file" in results[3]


def test_warning_refires_every_threshold_repeats_not_just_once():
    """
    Regression: seen in practice (weather.py test run) -- the guard fired
    once at call 4, the model verbally acknowledged the notice in a text-only
    response, then continued write_file-ing the SAME file 15+ more times
    with total silence, because the old design only ever warned once per
    target for its whole lifetime. A thrash that doesn't self-correct after
    one nudge needs to keep getting nudged: every `threshold`th consecutive
    call (4, 8, 12, ...), not just the 4th.
    """
    guard = RepetitionGuard(threshold=3)
    results = [guard.record("write_file", {"path": "a.py", "content": str(i)}) for i in range(9)]
    # fires at the 3rd, 6th, and 9th calls (indices 2, 5, 8), silent otherwise
    fired_indices = [i for i, r in enumerate(results) if r is not None]
    assert fired_indices == [2, 5, 8]


def test_warning_count_in_message_reflects_actual_consecutive_calls():
    guard = RepetitionGuard(threshold=3)
    for _ in range(5):
        guard.record("write_file", {"path": "a.py", "content": "x"})
    result = guard.record("write_file", {"path": "a.py", "content": "x"})  # 6th call
    assert "6 times in a row" in result


def test_different_content_same_path_still_counts_as_repetition():
    """The whole point: different edits to the SAME file, back to back, is
    the thrash pattern -- content differing each time should not defeat
    detection."""
    guard = RepetitionGuard(threshold=3)
    guard.record("edit_file", {"path": "a.py", "old_str": "1", "new_str": "2"})
    guard.record("edit_file", {"path": "a.py", "old_str": "2", "new_str": "3"})
    result = guard.record("edit_file", {"path": "a.py", "old_str": "3", "new_str": "4"})
    assert result is not None


def test_different_targets_do_not_trigger_warning():
    guard = RepetitionGuard(threshold=3)
    guard.record("read_file", {"path": "a.py"})
    guard.record("read_file", {"path": "b.py"})
    result = guard.record("read_file", {"path": "c.py"})
    assert result is None


def test_checkpoint_resets_the_count():
    guard = RepetitionGuard(threshold=3)
    guard.record("edit_file", {"path": "a.py", "old_str": "1", "new_str": "2"})
    guard.record("edit_file", {"path": "a.py", "old_str": "2", "new_str": "3"})
    guard.checkpoint()  # e.g. an acceptance-check attempt happened
    result = guard.record("edit_file", {"path": "a.py", "old_str": "3", "new_str": "4"})
    assert result is None  # count restarted, only 1 since checkpoint


def test_run_command_uses_command_string_as_target():
    guard = RepetitionGuard(threshold=3)
    guard.record("run_command", {"command": "pytest -q"})
    guard.record("run_command", {"command": "pytest -q"})
    result = guard.record("run_command", {"command": "pytest -q"})
    assert result is not None
    assert "pytest -q" in result
