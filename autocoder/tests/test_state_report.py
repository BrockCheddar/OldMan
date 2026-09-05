import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.llm import Turn, ToolCall, ToolResult
from autocoder.state_report import compact_history_with_state_report


def _turn(i: int) -> Turn:
    return Turn(role="tool_results", tool_results=[ToolResult(tool_call_id=f"c{i}", content=f"output {i}")])


def test_short_history_is_left_unchanged():
    history = [Turn(role="user", text="seed"), _turn(1), _turn(2)]
    result = compact_history_with_state_report(history, "REPORT", keep_recent_turns=4)
    assert result == history


def test_long_history_gets_compacted_to_seed_report_and_recent():
    history = [Turn(role="user", text="seed")] + [_turn(i) for i in range(1, 11)]
    result = compact_history_with_state_report(history, "REPORT TEXT", keep_recent_turns=3)

    assert result[0].text == "seed"
    assert result[1].text == "REPORT TEXT"
    assert len(result) == 2 + 3  # seed + report + 3 recent
    assert result[-3:] == history[-3:]


def test_dropped_middle_content_is_not_present_anywhere_in_result():
    history = [Turn(role="user", text="seed")] + [_turn(i) for i in range(1, 11)]
    result = compact_history_with_state_report(history, "REPORT", keep_recent_turns=2)

    all_text = []
    for t in result:
        if t.text:
            all_text.append(t.text)
        for tr in t.tool_results:
            all_text.append(tr.content)
    joined = "\n".join(all_text)

    # outputs 3 through 8 were in the dropped middle (kept: seed, report, output 9, output 10)
    for i in range(3, 9):
        assert f"output {i}" not in joined
    assert "output 9" in joined
    assert "output 10" in joined


def test_state_report_replaces_gap_with_full_real_content_not_a_placeholder():
    """The old trim design left a head+gap+tail fragment with no real
    meaning in the gap. The new design's replacement text is the full,
    real state report -- not a truncated fragment of anything."""
    history = [Turn(role="user", text="seed")] + [_turn(i) for i in range(1, 11)]
    report = "STATE REPORT -- attempt 2/3\nCurrent files: a.py, b.py\nGit status: clean"
    result = compact_history_with_state_report(history, report, keep_recent_turns=2)
    assert result[1].text == report  # inserted whole, not cut


def _outer_loop_iteration(i: int) -> list[Turn]:
    """One realistic outer-loop iteration: user context -> assistant with
    tool_calls -> tool_results answering those calls. Mirrors agent.py's
    _outer_loop, which appends exactly these three turns per pass."""
    return [
        Turn(role="user", text=f"context {i}"),
        Turn(role="assistant", text=None, tool_calls=[ToolCall(id=f"c{i}", name="run_command", input={})]),
        Turn(role="tool_results", tool_results=[ToolResult(tool_call_id=f"c{i}", content=f"output {i}")]),
    ]


def _assistant_ids(turn: Turn) -> set[str]:
    return {c.id for c in turn.tool_calls}


def _tool_result_ids(turn: Turn) -> set[str]:
    return {tr.tool_call_id for tr in turn.tool_results}


def _assert_no_orphaned_tool_results(history: list[Turn]) -> None:
    """The invariant the review recommends testing directly: every
    retained tool_results turn's ids must be answered by tool_calls in the
    immediately preceding assistant turn in the SAME list."""
    for i, turn in enumerate(history):
        if turn.role != "tool_results" or not turn.tool_results:
            continue
        assert i > 0, "tool_results turn with no preceding turn at all"
        parent = history[i - 1]
        assert parent.role == "assistant", (
            f"tool_results turn at index {i} has no answering assistant turn "
            f"immediately before it (found role={parent.role!r})"
        )
        assert _tool_result_ids(turn) <= _assistant_ids(parent), (
            f"tool_results turn at index {i} has ids not covered by its "
            f"preceding assistant turn's tool_calls"
        )


def test_no_orphaned_tool_results_across_outer_loop_lengths():
    """H2 regression: reproduces the review's finding directly -- for every
    outer-loop length from 2 to 7 iterations, compaction used to retain a
    tool_results turn whose parent assistant turn had just been dropped."""
    for num_iterations in range(2, 8):
        history: list[Turn] = [Turn(role="user", text="seed")]
        for i in range(num_iterations):
            history.extend(_outer_loop_iteration(i))
        result = compact_history_with_state_report(history, "REPORT", keep_recent_turns=4)
        _assert_no_orphaned_tool_results(result)


def test_widened_window_still_contains_the_expected_recent_content():
    """Widening the cut by one turn to avoid an orphan shouldn't lose or
    duplicate anything -- the pulled-in assistant turn is simply included
    alongside the report, not instead of it."""
    history: list[Turn] = [Turn(role="user", text="seed")]
    for i in range(5):
        history.extend(_outer_loop_iteration(i))
    result = compact_history_with_state_report(history, "REPORT", keep_recent_turns=4)
    _assert_no_orphaned_tool_results(result)
    # the most recent iteration's output must still be present
    all_content = [tr.content for t in result for tr in t.tool_results]
    assert "output 4" in all_content
