import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.planner import RunState, CompletedStep, StepContext


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


def test_step_context_prompt_text_omits_baseline_when_unset():
    ctx = StepContext(title="t", objective="o", acceptance_command="true")
    assert "BASELINE" not in ctx.prompt_text()


def test_step_context_prompt_text_includes_baseline_when_set():
    ctx = StepContext(
        title="t", objective="o", acceptance_command="true",
        baseline_check_result="$ true\nexit 1\nstdout:\n\nstderr:\nAttributeError: boom",
    )
    text = ctx.prompt_text()
    assert "BASELINE" in text
    assert "AttributeError: boom" in text


def test_completed_summary_folds_old_steps_beyond_cap():
    s = RunState(goal="g")
    for i in range(1, 51):
        s.completed_steps.append(CompletedStep(i, f"step {i}", f"did {i}", "true", "", "", 0))
    text = s.completed_summary_text()
    # oldest steps folded into one line, not rendered individually
    assert "step 1:" not in text
    assert "step 20:" not in text
    assert "20 earlier steps completed" in text
    # most recent MAX_DETAILED_COMPLETED_STEPS still rendered in full
    assert "step 21:" in text
    assert "step 50:" in text
    # on-disk record is untouched regardless of the rendered cap
    assert len(s.completed_steps) == 50


def test_completed_summary_no_folding_under_cap():
    s = RunState(goal="g")
    for i in range(1, 5):
        s.completed_steps.append(CompletedStep(i, f"step {i}", f"did {i}", "true", "", "", 0))
    text = s.completed_summary_text()
    assert "earlier steps completed" not in text
    assert "step 1:" in text


def test_goal_decomposition_replace_assigns_ids_and_survives_round_trip():
    from autocoder.planner import GoalDecomposition
    d = GoalDecomposition()
    d.replace(["part a", "part b"])
    assert len(d.subtasks) == 2
    assert d.subtasks[0].id != d.subtasks[1].id
    assert all(s.status == "pending" for s in d.subtasks)

    d.subtasks[0].status = "done"
    restored = GoalDecomposition.from_dict(d.to_dict())
    assert restored.get(d.subtasks[0].id).status == "done"
    assert restored.get(d.subtasks[1].id).status == "pending"


def test_goal_decomposition_replace_is_capped():
    from autocoder.planner import GoalDecomposition, MAX_SUBTASKS
    d = GoalDecomposition()
    d.replace([f"t{i}" for i in range(MAX_SUBTASKS + 10)])
    assert len(d.subtasks) == MAX_SUBTASKS


def test_run_state_round_trip_preserves_decomposition(tmp_path):
    from autocoder.planner import RunState
    s = RunState(goal="g")
    s.decomposition.replace(["x", "y"])
    s.decomposition.subtasks[0].status = "done"
    restored = RunState.from_dict(s.to_dict())
    assert len(restored.decomposition.subtasks) == 2
    assert restored.decomposition.subtasks[0].status == "done"


def test_replace_preserves_status_for_exact_title_matches():
    """The exact scenario from a real conversation: 10 items, 5 done,
    then set_goal_decomposition called again to add an 11th item. Progress
    on 1-10 must survive if their titles are repeated verbatim."""
    from autocoder.planner import GoalDecomposition
    d = GoalDecomposition()
    d.replace([f"item {i}" for i in range(1, 11)])
    old_ids = [s.id for s in d.subtasks]
    for i in range(5):
        d.subtasks[i].status = "done"

    preserved, added, dropped = d.replace([f"item {i}" for i in range(1, 12)])

    assert len(preserved) == 10
    assert added == ["item 11"]
    assert dropped == []
    statuses = {s.title: s.status for s in d.subtasks}
    for i in range(1, 6):
        assert statuses[f"item {i}"] == "done"
    for i in range(6, 12):
        assert statuses[f"item {i}"] == "pending"
    # ids for preserved items are stable across the re-declare
    new_ids = {s.title: s.id for s in d.subtasks}
    for i in range(1, 11):
        assert new_ids[f"item {i}"] == old_ids[i - 1]


def test_replace_is_case_and_whitespace_insensitive_for_matching():
    from autocoder.planner import GoalDecomposition
    d = GoalDecomposition()
    d.replace(["Do the thing"])
    d.subtasks[0].status = "done"

    preserved, added, dropped = d.replace(["  do the thing  "])
    assert len(preserved) == 1
    assert d.subtasks[0].status == "done"


def test_replace_treats_reworded_title_as_new_not_preserved():
    from autocoder.planner import GoalDecomposition
    d = GoalDecomposition()
    d.replace(["Implement the CLI"])
    d.subtasks[0].status = "done"

    preserved, added, dropped = d.replace(["Implement the CLI wrapper"])  # reworded
    assert preserved == []
    assert added == ["Implement the CLI wrapper"]
    assert dropped[0].status == "done"  # old "done" item wasn't carried over
    assert d.subtasks[0].status == "pending"  # new entry starts fresh


def test_replace_reports_dropped_done_items():
    """The one genuinely bad case: an old DONE item simply isn't in the
    new list at all -- must be visible in the return value, not silent."""
    from autocoder.planner import GoalDecomposition
    d = GoalDecomposition()
    d.replace(["keep me", "drop me"])
    d.subtasks[0].status = "done"
    d.subtasks[1].status = "done"

    preserved, added, dropped = d.replace(["keep me"])  # "drop me" omitted
    assert len(preserved) == 1
    assert len(dropped) == 1
    assert dropped[0].title == "drop me"
    assert dropped[0].status == "done"
