import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.llm import Turn, ToolResult
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
