import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.planner import RunState, CompletedStep


def test_initial_state():
    s = RunState(goal="build something")
    assert s.next_index() == 1
    assert s.completed_summary_text() == "(nothing completed yet)"
    assert s.status == "running"


def test_next_index_increments_with_steps():
    s = RunState(goal="g")
    s.completed_steps.append(CompletedStep(1, "step one", "did it", "true", "", "", 0))
    assert s.next_index() == 2


def test_completed_summary_text():
    s = RunState(goal="g")
    s.completed_steps.append(CompletedStep(1, "Create file", "wrote hello.py", "python -m py_compile hello.py", "", "", 0))
    text = s.completed_summary_text()
    assert "Create file" in text
    assert "wrote hello.py" in text


def test_roundtrip_serialisation():
    s = RunState(goal="build x", scratchpad="notes here")
    s.completed_steps.append(CompletedStep(1, "t", "s", "cmd", "out", "err", 0))
    s2 = RunState.from_dict(s.to_dict())
    assert s2.goal == "build x"
    assert s2.scratchpad == "notes here"
    assert len(s2.completed_steps) == 1
    assert s2.completed_steps[0].title == "t"
    assert s2.completed_steps[0].exit_code == 0


def test_status_roundtrip():
    s = RunState(goal="g", status="done")
    assert RunState.from_dict(s.to_dict()).status == "done"
