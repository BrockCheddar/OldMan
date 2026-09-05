import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fakes import FakeLLMClient, tool_response, text_response
from autocoder.agent import Agent
from autocoder.config import Config, LLMBackendConfig, Budget, ApprovalPolicy


def make_config(tmp_path, **budget_overrides):
    defaults = {"default_command_timeout": 30, "max_steps_per_attempt": 20}
    defaults.update(budget_overrides)
    return Config(
        workspace_root=tmp_path / "ws",
        llm=LLMBackendConfig(context_window_tokens=32768),
        budget=Budget(**defaults),
        approval=ApprovalPolicy(mode="auto"),
    )


def test_happy_path_propose_then_done(tmp_path):
    """
    Outer: propose_step -> Inner: write_file + mark_step_done (passes) ->
    Outer: declare_done (auto-accepted via final_acceptance_command)
    """
    llm = FakeLLMClient([
        # outer turn 1: propose a step
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        # inner turn 1: write the file
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        # inner turn 2: mark it done
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        # outer turn 2: declare overall done
        tool_response("declare_done", {"summary": "all done"}, call_id="o2"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    # final acceptance: hello.py compiles cleanly
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert len(state.completed_steps) == 1
    assert state.completed_steps[0].title == "Create hello.py"
    assert (agent.workspace.root / "hello.py").exists()
    log = agent.workspace.git_log()
    assert "step 1" in log.stdout


def test_acceptance_failure_retries_then_succeeds(tmp_path):
    """
    Inner loop: mark_step_done fails acceptance twice (no file created),
    then writes the file and succeeds on the third call.
    """
    llm = FakeLLMClient([
        # outer: propose
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        # inner attempt 1: immediately declare done (file doesn't exist yet -> fails)
        tool_response("mark_step_done", {"summary": "done?"}, call_id="i1"),
        # inner attempt 2: declare done again (still fails)
        tool_response("mark_step_done", {"summary": "done?"}, call_id="i2"),
        # inner attempt 3: write the file THEN declare done
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i3"),
        tool_response("mark_step_done", {"summary": "now it exists"}, call_id="i4"),
        # outer: declare overall done
        tool_response("declare_done", {"summary": "finished"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path, max_subtask_attempts=3), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert (agent.workspace.root / "hello.py").exists()


def test_resume_picks_up_completed_steps(tmp_path):
    """A saved state with one completed step is resumed and the agent
    proceeds to declare done without redoing the completed work."""
    from autocoder.planner import RunState, CompletedStep

    cfg = make_config(tmp_path)
    # pre-create workspace so Agent.__init__ finds an existing .git
    ws_root = cfg.workspace_root
    from autocoder.workspace import Workspace
    ws = Workspace.create(ws_root, source_repo=None)
    (ws.root / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    ws.git_commit_all("pre-existing step")

    state = RunState(goal="make hello.py")
    state.completed_steps.append(
        CompletedStep(1, "Create hello.py", "wrote it", "python -m py_compile hello.py", "", "", 0)
    )
    from autocoder.session import SessionStore
    SessionStore(cfg).save_state(state)

    llm = FakeLLMClient([
        tool_response("declare_done", {"summary": "all good"}, call_id="o1"),
    ])
    agent = Agent(cfg, llm=llm)
    result = agent.run(goal=None, resume=True,
                       final_acceptance_command="python -m py_compile hello.py")

    assert result.status == "done"
    # completed_steps from the saved session must still be there
    assert result.completed_steps[0].title == "Create hello.py"


def test_escalation_abort(tmp_path, monkeypatch):
    """Step fails all attempts -> escalation -> human types 'abort'."""
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Impossible",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",
        }),
        # inner: immediately mark done (always fails the acceptance check)
        tool_response("mark_step_done", {"summary": "sure"}, call_id="i1"),
    ])

    monkeypatch.setattr("builtins.input", lambda prompt="": "abort")
    agent = Agent(make_config(tmp_path, max_subtask_attempts=1, max_replan_cycles=0), llm=llm)

    from autocoder.agent import AgentAborted
    import pytest
    with pytest.raises(AgentAborted):
        agent.run(goal="impossible task", resume=False)


def test_llm_error_retry_succeeds(tmp_path, monkeypatch):
    """
    Regression: a backend failure (timeout, connection drop, etc -- raised
    as LLMError) must NOT crash the whole process. The human is asked to
    retry or abort; on retry the exact same call is attempted again.
    """
    from autocoder.llm import LLMError

    llm = FakeLLMClient([
        LLMError("simulated timeout talking to local server"),
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": "retry")

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert (agent.workspace.root / "hello.py").exists()


def test_llm_error_abort_saves_state(tmp_path, monkeypatch):
    """
    Regression: choosing 'abort' after a backend failure must exit cleanly
    (AgentAborted, not a raw crash) and mark the session aborted on disk
    rather than leaving it in an ambiguous state.
    """
    from autocoder.llm import LLMError
    from autocoder.agent import AgentAborted
    import pytest

    llm = FakeLLMClient([LLMError("simulated timeout talking to local server")])
    monkeypatch.setattr("builtins.input", lambda prompt="": "abort")

    cfg = make_config(tmp_path)
    agent = Agent(cfg, llm=llm)
    with pytest.raises(AgentAborted):
        agent.run(goal="make hello.py", resume=False)

    saved = agent.session.load_state()
    assert saved.status == "aborted"


def test_outer_budget_exceeded_guidance_actually_resumes(tmp_path, monkeypatch):
    """
    Regression: outer loop hits its step budget while just exploring (no
    propose_step yet). Human gives guidance (not 'abort'). The run must
    actually continue and finish -- not immediately re-raise BudgetExceeded
    on the very next iteration because the counter was never reset.
    """
    llm = FakeLLMClient([
        # burn the tiny outer budget with harmless exploration calls
        tool_response("list_dir", {"path": "."}, call_id="e1"),
        tool_response("list_dir", {"path": "."}, call_id="e2"),
        tool_response("list_dir", {"path": "."}, call_id="e3"),
        # after guidance, model finally proposes a real step
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": "run it and make the gif")

    cfg = make_config(tmp_path, max_steps_per_attempt=2)  # tiny cap -> hits budget after 2 exploration calls
    agent = Agent(cfg, llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert (agent.workspace.root / "hello.py").exists()
    assert "run it and make the gif" in state.scratchpad


def test_failed_step_escalation_guidance_resumes_instead_of_ending_run(tmp_path, monkeypatch):
    """
    Regression: a step that exhausts all acceptance-check attempts escalates.
    If the human gives guidance (not 'abort'), the run must keep going, not
    unconditionally end (it previously called `return` regardless of what
    the human typed).
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Impossible on first try",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",
        }),
        # inner: immediately mark done, always fails the check, exhausts the single allowed attempt
        tool_response("mark_step_done", {"summary": "sure"}, call_id="i1"),
        # after guidance, model proposes a real, achievable step
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }, call_id="p2"),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i2"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i3"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": "try something achievable instead")

    agent = Agent(make_config(tmp_path, max_subtask_attempts=1, max_replan_cycles=0), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert (agent.workspace.root / "hello.py").exists()


def test_mark_step_done_without_propose_step_is_rejected(tmp_path):
    """
    Regression: model calls mark_step_done directly from the outer loop
    (skipping propose_step). Must get a clear error back and be redirected,
    not fall through to toolbox.dispatch which has no handler for it.
    The model then calls propose_step properly and completes.
    """
    llm = FakeLLMClient([
        # outer turn 1: model incorrectly calls mark_step_done without proposing first
        tool_response("mark_step_done", {"summary": "done already"}, call_id="bad1"),
        # outer turn 2: model now correctly proposes a step
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        # inner: write + mark done
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote it"}, call_id="i2"),
        # outer: declare done
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert (agent.workspace.root / "hello.py").exists()


def test_scratchpad_persists_across_steps(tmp_path):
    """Scratchpad written in mark_step_done is present in the next outer turn."""
    llm = FakeLLMClient([
        # outer turn 1: propose step 1
        tool_response("propose_step", {
            "title": "Step one",
            "acceptance_command": "python -m py_compile a.py",
        }),
        # inner: write file + mark done with scratchpad update
        tool_response("write_file", {"path": "a.py", "content": "x=1\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote a.py", "scratchpad": "remember: need b.py next"}, call_id="i2"),
        # outer turn 2: declare done (check scratchpad is in system prompt)
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make a.py", resume=False,
                      final_acceptance_command="python -m py_compile a.py")

    assert state.status == "done"
    assert "remember: need b.py next" in state.scratchpad


def test_decision_recorded_in_step_one_reaches_step_two_coder_prompt_verbatim(tmp_path):
    """
    The propagation guarantee the decisions store exists for: a decision
    recorded via record_decision during step one's inner loop must show up,
    verbatim, in step two's inner-loop system prompt -- with the planner
    never restating it anywhere in step two's objective/findings text.
    """
    llm = FakeLLMClient([
        # outer turn 1: propose step one
        tool_response("propose_step", {
            "title": "Step one",
            "acceptance_command": "python -c \"pass\"",
        }),
        # inner step one, attempt 1: record a decision, then mark done.
        # Deliberately no write_file -- the acceptance command doesn't need one.
        tool_response("record_decision", {"decision": "Use SQLAlchemy ORM, not raw sqlite3"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "recorded the ORM decision"}, call_id="i2"),
        # outer turn 2: propose step two -- objective says nothing about SQLAlchemy
        tool_response("propose_step", {
            "title": "Step two",
            "acceptance_command": "python -c \"pass\"",
        }),
        # inner step two, attempt 1: this is the system prompt under test
        tool_response("mark_step_done", {"summary": "step two done"}, call_id="i3"),
        # outer turn 3: declare done
        tool_response("declare_done", {"summary": "all done"}, call_id="o3"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="build the thing", resume=False,
                      final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    assert len(state.completed_steps) == 2

    # find the inner-loop system prompt for step two's first attempt: the
    # call whose history opens with step two's StepContext, not step one's.
    step_two_calls = [
        c for c in llm.calls
        if c["history"] and "Step two" in (c["history"][0].text or "")
    ]
    assert step_two_calls, "expected at least one LLM call scoped to step two"
    step_two_system = step_two_calls[0]["system"]

    assert "Use SQLAlchemy ORM, not raw sqlite3" in step_two_system
    # and the planner's own step-two framing never had to restate it
    assert "SQLAlchemy" not in step_two_calls[0]["history"][0].text


def test_declare_done_human_rejection_feedback_reaches_model(tmp_path, monkeypatch):
    """
    Regression: previously, saying 'N' to 'Accept as done?' sent the model a
    fixed generic message with no reason. The human's typed explanation must
    actually reach the model (as the tool_result) and be saved to scratchpad.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        # first declare_done -- will be rejected by the human below
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
        # model tries again after seeing the feedback
        tool_response("declare_done", {"summary": "added the missing docstring"}, call_id="o2"),
    ])

    answers = iter(["n", "needs a module docstring explaining what it does", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False)  # no acceptance command -> interactive path

    assert state.status == "done"
    assert "needs a module docstring explaining what it does" in state.scratchpad


def test_declare_done_acceptance_command_failure_feedback_reaches_model(tmp_path):
    """
    Regression: when a final_acceptance_command exists and fails, the model
    previously got no information at all -- just 'Goal not yet accepted.
    Keep working.' The real command output must now reach the model.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        # declares done, but the final acceptance command will fail (file doesn't exist)
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
        # after seeing the failure, proposes a real step to create the missing file
        # (must go through propose_step -- a bare outer-loop write_file is rejected)
        tool_response("propose_step", {
            "title": "Create goal_marker.txt",
            "acceptance_command": "python -c \"open('goal_marker.txt')\"",
        }, call_id="p2"),
        tool_response("write_file", {"path": "goal_marker.txt", "content": "ok\n"}, call_id="i3"),
        tool_response("mark_step_done", {"summary": "created goal_marker.txt"}, call_id="i4"),
        tool_response("declare_done", {"summary": "done for real"}, call_id="o2"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(
        goal="make hello.py", resume=False,
        final_acceptance_command="python -c \"open('goal_marker.txt')\"",
    )

    assert state.status == "done"
    assert "Final acceptance check FAILED" in state.scratchpad
    assert (agent.workspace.root / "goal_marker.txt").exists()


def test_propose_step_acceptance_command_warns_about_shell_differences():
    """
    Regression: the model writes acceptance_command as part of propose_step,
    before the inner loop's system prompt (which also has an OS hint) is ever
    shown. The warning has to live in the tool schema itself or a Windows
    user's model will keep proposing acceptance commands with tail/head/grep
    that don't exist on cmd.exe.
    """
    from autocoder.agent import PROPOSE_STEP_TOOL
    desc = PROPOSE_STEP_TOOL["input_schema"]["properties"]["acceptance_command"]["description"]
    assert "cmd.exe" in desc
    assert "tail" in desc and "head" in desc
    assert "test" in desc  # POSIX 'test'/'[' builtin, the specific bug that motivated this


def test_lesson_recorded_when_step_succeeds_after_failed_attempt(tmp_path):
    """
    Regression / feature: a step that fails its acceptance check at least
    once, then succeeds, should record a verified lesson -- not silently
    discard the fact that something didn't work the first time.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        # attempt 1: declares done with nothing written -> acceptance fails
        tool_response("mark_step_done", {"summary": "should be done"}, call_id="i1"),
        # attempt 2: actually writes the file, then succeeds
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i2"),
        tool_response("mark_step_done", {"summary": "now really created it"}, call_id="i3"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path, max_subtask_attempts=3), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    lessons = agent.lessons.load()
    assert len(lessons) == 1
    assert "py_compile" in lessons[0].symptom
    assert "now really created it" in lessons[0].fix


def test_lesson_recorded_when_escalation_guidance_leads_to_success(tmp_path, monkeypatch):
    """
    Feature: when a step exhausts all attempts and escalates, and the human's
    guidance leads to a subsequent success, that pairing (what was stuck,
    what the human said, what worked afterward) should be recorded as a
    lesson for future runs in this same workspace.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Use a POSIX-only check",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",  # always fails
        }),
        tool_response("mark_step_done", {"summary": "sure"}, call_id="i1"),
        # after guidance, propose a real, achievable step
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }, call_id="p2"),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i2"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i3"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": "use a python check instead")

    agent = Agent(make_config(tmp_path, max_subtask_attempts=1, max_replan_cycles=0), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    lessons = agent.lessons.load()
    assert len(lessons) == 1
    assert "use a python check instead" in lessons[0].symptom
    assert "hello.py" in lessons[0].fix


def test_no_lesson_recorded_when_step_succeeds_on_first_try(tmp_path):
    """A step that just works on the first attempt has nothing to teach --
    no lesson should be recorded (recording every success would defeat the
    point of keeping this small and worth reading)."""
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert agent.lessons.load() == []


def test_repetition_guard_warning_reaches_model_in_inner_loop(tmp_path):
    """
    Regression: the model can thrash on the same file repeatedly (many
    edit_file calls on the same target) with nothing telling it to stop and
    check its actual state. The 4th identical-target call should get a
    [REPETITION NOTICE] appended to its own tool_result content.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "x=1\n"}, call_id="i1"),
        tool_response("write_file", {"path": "hello.py", "content": "x=2\n"}, call_id="i2"),
        tool_response("write_file", {"path": "hello.py", "content": "x=3\n"}, call_id="i3"),
        # 4th identical-target call in a row -- should trigger the warning
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i4"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i5"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"

    # Calls: [0]=outer propose_step, [1..4]=inner write_file x4, [5]=inner
    # call after the 4th write_file's tool_result (containing the warning)
    # has been appended to history.
    sixth_call_history = llm.calls[5]["history"]
    all_tool_result_text = "\n".join(
        tr.content
        for turn in sixth_call_history
        for tr in turn.tool_results
    )
    assert "REPETITION NOTICE" in all_tool_result_text
    assert "hello.py" in all_tool_result_text


def test_inner_loop_failure_reason_distinguishes_derailment_from_real_failure():
    """
    Regression: a step that derails (model runs out of steps without ever
    attempting a real acceptance check) used to be reported identically to
    a step with genuine, repeated failed checks -- 'failed its acceptance
    check repeatedly' either way. That's misleading: those are different
    situations and the human should be told which one actually happened.
    """
    from autocoder.agent import _inner_loop_failure_reason

    only_derailed = _inner_loop_failure_reason("x", check_failures=0, derailed_attempts=2)
    assert "ran out of its step budget" in only_derailed

    only_real_failures = _inner_loop_failure_reason("x", check_failures=3, derailed_attempts=0)
    assert "ran out of its step budget" not in only_real_failures
    assert "failed" in only_real_failures.lower()

    mixed = _inner_loop_failure_reason("x", check_failures=1, derailed_attempts=1)
    assert "1 failed acceptance check" in mixed
    assert "1 attempt(s) that ran out of steps" in mixed


def test_derailed_attempt_does_not_corrupt_next_attempts_history(tmp_path):
    """
    Regression: a step attempt that derails (empty tool_calls, burning the
    whole step budget) used to leave its corrupted/hallucinated conversation
    in `history`, which the NEXT attempt then inherited and built on instead
    of starting clean -- compounding the exact problem that caused the
    derailment. Each attempt must start from a fresh copy of the outer
    context, with only an explicit, honest note about what happened before.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        # attempt 1: two empty/text-only responses -> burns the tiny step
        # budget (max_steps_per_attempt=2) without ever calling a tool
        text_response("I need to think about this some more..."),
        text_response("Let me reconsider the approach..."),
        # attempt 2 (fresh history): a real, working sequence
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    cfg = make_config(tmp_path, max_steps_per_attempt=2, max_subtask_attempts=2)
    agent = Agent(cfg, llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"

    # Find the call that starts attempt 2 (right after the derailment) and
    # confirm its history does NOT contain the hallucinated text from
    # attempt 1, but DOES contain the honest note about running out of steps.
    attempt_2_start_history = llm.calls[3]["history"]  # [0]=outer, [1,2]=attempt1's 2 empty turns, [3]=attempt2 start
    texts = [t.text for t in attempt_2_start_history if t.text]
    assert not any("think about this some more" in t for t in texts)
    assert not any("reconsider the approach" in t for t in texts)
    assert any("ran out of its step budget" in t for t in texts)


def test_passing_checks_print_confirmation(tmp_path, capsys):
    """
    Regression: a FAILED acceptance check has always printed a clear
    '[check] FAILED (attempt N/M)' line, but a PASSING check printed nothing
    at all -- the log just silently moved on, making it genuinely hard to
    tell from the terminal whether a check passed or was still pending.
    Both the per-step check and the final goal-level check need a symmetric
    PASSED confirmation.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    out = capsys.readouterr().out
    assert "[check] PASSED (attempt 1/" in out
    assert "[done-check] PASSED" in out


def test_approval_prompt_pauses_and_resumes_the_budget_clock(tmp_path, monkeypatch):
    """
    Regression: budget.pause()/resume() exist and are unit-tested in
    isolation (test_budget.py), but the actual wiring through Agent ->
    ToolBox -> the approval prompt is what matters in practice. Confirms
    the real end-to-end path invokes both.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Run something that needs approval",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        # pip install always needs approval regardless of policy mode
        tool_response("run_command", {"command": "pip install something"}, call_id="i1"),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i2"),
        tool_response("mark_step_done", {"summary": "done"}, call_id="i3"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")  # deny the install, doesn't matter for this test

    cfg = make_config(tmp_path)
    cfg.approval.mode = "smart"
    agent = Agent(cfg, llm=llm)

    pause_calls = []
    resume_calls = []
    agent.budget.pause = lambda: pause_calls.append(True)
    agent.budget.resume = lambda: resume_calls.append(True)
    # re-wire the toolbox to use the spied methods (it captured the original
    # bound methods at construction time)
    agent.toolbox._on_pause = agent.budget.pause
    agent.toolbox._on_resume = agent.budget.resume

    agent.run(goal="make hello.py", resume=False, final_acceptance_command="python -m py_compile hello.py")

    assert len(pause_calls) >= 1
    assert len(resume_calls) >= 1


def test_validate_python_dash_c_syntax_catches_real_broken_command():
    """
    Regression: this exact command (a stray '-' in an import statement)
    appeared as a model-written acceptance_command in practice and failed
    its real check 3 times in a row -- a guaranteed-unwinnable check burning
    the entire attempt budget before ever being caught.
    """
    from autocoder.agent import _validate_python_dash_c_syntax

    problem = _validate_python_dash_c_syntax(
        '''python -c "import texttransform.-uppercase; print('import works')"'''
    )
    assert problem is not None
    assert "syntax error" in problem.lower()


def test_validate_python_dash_c_syntax_accepts_valid_code():
    from autocoder.agent import _validate_python_dash_c_syntax

    assert _validate_python_dash_c_syntax(
        '''python -c "import ast; ast.parse(open('hello.py').read())"'''
    ) is None


def test_validate_python_dash_c_syntax_ignores_non_python_c_commands():
    from autocoder.agent import _validate_python_dash_c_syntax

    assert _validate_python_dash_c_syntax("pytest -q") is None
    assert _validate_python_dash_c_syntax("python -m py_compile hello.py") is None


def test_validate_python_dash_c_syntax_handles_cd_prefix():
    from autocoder.agent import _validate_python_dash_c_syntax

    assert _validate_python_dash_c_syntax('cd demo && python -c "print(1)"') is None
    broken = _validate_python_dash_c_syntax('cd demo && python -c "import x.-y"')
    assert broken is not None


def test_validate_python_dash_c_syntax_catches_sys_exit_without_import_sys():
    """
    Regression: seen twice in practice (game_of_life.py, then again with
    potentialupgrades.txt) -- `sys.exit(...)` used with only `import os`
    present. This is valid Python SYNTAX (ast.parse alone won't catch it)
    but a guaranteed NameError at actual run time, making the acceptance
    check fail every time regardless of what the model implements.
    """
    from autocoder.agent import _validate_python_dash_c_syntax

    real_cmd = (
        '''python -c "import os; sys.exit(0 if os.path.exists('f.txt') '''
        '''and os.path.getsize('f.txt') > 100 else 1)"'''
    )
    problem = _validate_python_dash_c_syntax(real_cmd)
    assert problem is not None
    assert "sys" in problem
    assert "NameError" in problem


def test_validate_python_dash_c_syntax_accepts_command_with_correct_imports():
    from autocoder.agent import _validate_python_dash_c_syntax

    ok_cmd = (
        '''python -c "import os, sys; sys.exit(0 if os.path.exists('f.txt') else 1)"'''
    )
    assert _validate_python_dash_c_syntax(ok_cmd) is None


def test_validate_python_dash_c_syntax_does_not_false_positive_on_local_variables():
    """
    A local variable named the same as a common stdlib module, used with a
    dotted method call, must NOT be flagged -- this is completely ordinary
    code (e.g. `results.append(...)`), not a missing import.
    """
    from autocoder.agent import _validate_python_dash_c_syntax

    ordinary = '''python -c "results = []; results.append(1); print(results)"'''
    assert _validate_python_dash_c_syntax(ordinary) is None

    # even a variable literally named after a stdlib module is fine if it's
    # locally assigned -- that's intentional shadowing, not a missing import
    shadowed = '''python -c "os = {'key': 'value'}; print(os['key'])"'''
    assert _validate_python_dash_c_syntax(shadowed) is None


def test_missing_import_check_finds_multiple_missing_modules():
    from autocoder.agent import _find_missing_stdlib_imports

    code = "sys.exit(0 if os.path.exists('f') else 1)"
    missing = _find_missing_stdlib_imports(code)
    assert set(missing) == {"sys", "os"}


def test_propose_step_with_broken_acceptance_command_rejected_before_inner_loop(tmp_path):
    """
    Regression: a propose_step call with a syntactically broken python -c
    acceptance_command must be rejected immediately -- no inner-loop tool
    calls (write_file, mark_step_done) should ever be consumed on an attempt
    that could never possibly pass.
    """
    llm = FakeLLMClient([
        # first attempt: broken acceptance_command, must be rejected outright
        tool_response("propose_step", {
            "title": "Create plugin files",
            "acceptance_command": '''python -c "import texttransform.-uppercase"''',
        }, call_id="p1"),
        # model corrects itself and proposes a valid one
        tool_response("propose_step", {
            "title": "Create plugin files",
            "acceptance_command": "python -m py_compile hello.py",
        }, call_id="p2"),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "done"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert len(state.completed_steps) == 1
    # the rejected first attempt must not have been recorded as any kind of
    # completed or failed step -- it never got the chance to run at all
    assert state.completed_steps[0].title == "Create plugin files"


def test_outer_loop_rejects_write_file_even_if_model_calls_it_anyway(tmp_path):
    """
    Regression: the tool LIST offered to the model excludes write_file at
    the outer level, but that alone doesn't stop a local model from calling
    it anyway if the backend doesn't strictly enforce the offered schema --
    seen repeatedly in this harness's history with other hallucinated/
    off-schema tool calls. dispatch() must never actually execute a mutating
    call the outer loop wasn't supposed to be able to make, regardless of
    whether the model was "supposed" to know better.
    """
    llm = FakeLLMClient([
        # model hallucinates a write_file call directly in the outer loop,
        # bypassing propose_step entirely
        tool_response("write_file", {"path": "sneaky.py", "content": "malicious=True\n"}, call_id="bad1"),
        # then does it properly
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }, call_id="p1"),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    # the hallucinated outer-loop write_file must NOT have actually run
    assert not (agent.workspace.root / "sneaky.py").exists()
    assert (agent.workspace.root / "hello.py").exists()


def test_declare_done_schema_discourages_wrapping_in_a_step():
    """
    Regression: seen in practice -- after the actual goal was fully done,
    the model invented a step just to "declare completion" with its own
    self-written acceptance check, which referenced a file from an earlier,
    since-corrected plan and could never pass again. declare_done should be
    called directly; it's independently verified already.
    """
    from autocoder.agent import DECLARE_DONE_TOOL
    desc = DECLARE_DONE_TOOL["description"]
    assert "directly" in desc.lower()
    assert "propose_step" in desc


def test_outer_system_prompt_discourages_wrapping_declare_done_in_a_step(tmp_path):
    from autocoder.agent import Agent
    from autocoder.planner import RunState

    agent = Agent(make_config(tmp_path), llm=FakeLLMClient([]))
    prompt = agent._outer_system_prompt(RunState(goal="test goal"))
    assert "declare_done directly" in prompt or "declare done directly" in prompt.lower()











def test_coder_receives_step_context_not_planner_history(tmp_path):
    """The planner may inspect a lot, but its tool-result history must not
    cross the planner -> coder context boundary."""
    llm = FakeLLMClient([
        # Outer planner: deliberately create a large-ish exploration result.
        tool_response("read_file", {"path": "missing-but-harmless.txt", "offset": 0, "limit": 10}, call_id="o1"),
        tool_response("propose_step", {
            "title": "Create hello.py",
            "objective": "Create the small hello program.",
            "files": ["hello.py"],
            "relevant_regions": [],
            "findings": "No existing hello.py was found.",
            "acceptance_command": "python -m py_compile hello.py",
        }, call_id="o2"),
        # Inner coder.
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "created hello.py"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o3"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(
        goal="make hello.py",
        resume=False,
        final_acceptance_command="python -m py_compile hello.py",
    )

    assert state.status == "done"
    # calls: outer read, outer propose, inner write, inner mark, dirty-file
    # condensation flush (hello.py was written then the step passed), outer done
    assert len(llm.calls) == 6
    coder_call = llm.calls[2]
    assert coder_call["history_len"] == 1
    assert "Create hello.py" in coder_call["history"][0].text
    assert "read_file" not in coder_call["history"][0].text
    assert "No existing hello.py was found." in coder_call["history"][0].text


def test_state_report_compaction_fires_and_carries_real_files_list(tmp_path):
    """
    Integration test: with a small context window, the coder's history
    should get compacted via a live state report once it grows large
    enough -- and that report must be built from a proper file LIST, even
    when the model returned `files` as a bare string (regression: a bare
    string iterated character-by-character would corrupt the report).
    """
    big_output = "x" * 600  # each tool_result adds real bulk

    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "objective": "make hello.py",
            "acceptance_command": "python -m py_compile hello.py",
            "files": "hello.py",  # bare string, not a list -- must not become ['h','e','l',...]
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("run_command", {"command": f"python -c \"print('{big_output}')\""}, call_id="i2"),
        tool_response("run_command", {"command": f"python -c \"print('{big_output}')\""}, call_id="i3"),
        tool_response("run_command", {"command": f"python -c \"print('{big_output}')\""}, call_id="i4"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i5"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])

    cfg = Config(
        workspace_root=tmp_path / "ws",
        llm=LLMBackendConfig(context_window_tokens=700),  # small -> forces compaction quickly
        budget=Budget(default_command_timeout=30, max_steps_per_attempt=20, max_output_tokens=50),
        approval=ApprovalPolicy(mode="auto"),
    )
    agent = Agent(cfg, llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"

    # find a call whose history contains a compacted state report
    found_report = None
    for call in llm.calls:
        for turn in call["history"]:
            if turn.text and turn.text.startswith("STATE REPORT"):
                found_report = turn.text
    assert found_report is not None, "state-report compaction never fired"
    # the file list must appear as the real filename, not scattered characters
    assert "hello.py" in found_report
    assert "--- hello.py ---" in found_report


def test_outer_loop_repetition_force_escalates_after_repeated_ignored_warnings(tmp_path, monkeypatch):
    """
    Regression: in a real run, the repetition notice re-fired 3 times (at
    4, 8, and 12 repeats of the same run_command call) and the model kept
    repeating anyway, well past the point a text nudge was clearly not
    working -- the run went on to hit a context-overflow crash shortly
    after. Past 3 ignored fires, the harness must escalate to a human
    directly instead of sending yet another warning that has already
    been shown not to help.
    """
    from autocoder.agent import AgentAborted
    import pytest

    responses = [
        tool_response("run_command", {"command": "python -c \"print(1)\""}, call_id=f"c{i}")
        for i in range(12)  # 12 identical calls = fires at 4, 8, 12 -> 3rd fire triggers escalation
    ]
    llm = FakeLLMClient(responses)
    monkeypatch.setattr("builtins.input", lambda prompt="": "abort")

    agent = Agent(make_config(tmp_path), llm=llm)
    with pytest.raises(AgentAborted):
        agent.run(goal="do something repetitive", resume=False)

    saved = agent.session.load_state()
    assert saved.status == "aborted"


def test_inner_loop_repetition_force_escalates_after_repeated_ignored_warnings(tmp_path, monkeypatch):
    """Same as the outer-loop version, but for repetition happening inside
    an active step's inner loop."""
    from autocoder.agent import AgentAborted
    import pytest

    responses = [
        tool_response("propose_step", {
            "title": "Create hello.py",
            "objective": "make hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
    ] + [
        tool_response("write_file", {"path": "hello.py", "content": f"x = {i}\n"}, call_id=f"i{i}")
        for i in range(12)
    ]
    llm = FakeLLMClient(responses)
    monkeypatch.setattr("builtins.input", lambda prompt="": "abort")

    agent = Agent(make_config(tmp_path), llm=llm)
    with pytest.raises(AgentAborted):
        agent.run(goal="make hello.py", resume=False)

    saved = agent.session.load_state()
    assert saved.status == "aborted"


def test_outer_loop_state_report_compaction_fires_with_live_facts(tmp_path):
    """
    Regression: the outer loop's history was still using the old head/gap/
    tail trim, never the new state-report compaction -- confirmed by a real
    crash that happened entirely in outer-loop activity (no [step] header
    active), which the inner-loop-only fix from the prior session could not
    have prevented. The outer loop must now compact via a live state report
    too, once its own history grows too large.
    """
    big_output = "y" * 600

    responses = [
        tool_response("run_command", {"command": f"python -c \"print('{big_output}A')\""}, call_id="c1"),
        tool_response("run_command", {"command": f"python -c \"print('{big_output}B')\""}, call_id="c2"),
        tool_response("run_command", {"command": f"python -c \"print('{big_output}C')\""}, call_id="c3"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ]
    llm = FakeLLMClient(responses)

    cfg = Config(
        workspace_root=tmp_path / "ws",
        llm=LLMBackendConfig(context_window_tokens=700),
        budget=Budget(default_command_timeout=30, max_steps_per_attempt=20, max_output_tokens=50),
        approval=ApprovalPolicy(mode="auto"),
    )
    agent = Agent(cfg, llm=llm)
    state = agent.run(goal="run some diagnostics", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    found_report = None
    for call in llm.calls:
        for turn in call["history"]:
            if turn.text and turn.text.startswith("STATE REPORT (outer loop)"):
                found_report = turn.text
    assert found_report is not None, "outer-loop state-report compaction never fired"
    assert "Goal: run some diagnostics" in found_report
    assert "Git status" in found_report




def test_declare_done_rejected_when_a_later_step_regresses_an_earlier_one(tmp_path):
    """M2 regression: step 2's own acceptance command (compiling other.py)
    doesn't notice that step 2 also broke hello.py, which step 1 already
    verified. Without re-running prior acceptance commands at declare_done
    time, this would be silently accepted. The model must be told about
    the regression and get a chance to fix it before the run can end."""
    llm = FakeLLMClient([
        # outer 1: propose step 1 -- create hello.py
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote hello.py"}, call_id="i2"),

        # outer 2: propose step 2 -- creates other.py, but ALSO corrupts
        # hello.py as a side effect. Step 2's own acceptance command only
        # checks other.py, so it passes without noticing.
        tool_response("propose_step", {
            "title": "Create other.py",
            "acceptance_command": "python -m py_compile other.py",
        }, call_id="o2"),
        tool_response("write_file", {"path": "hello.py", "content": "def broken(:\n"}, call_id="i3"),
        tool_response("write_file", {"path": "other.py", "content": "print('ok')\n"}, call_id="i4"),
        tool_response("mark_step_done", {"summary": "wrote other.py"}, call_id="i5"),

        # outer 3: declare done -- must be REJECTED, hello.py regressed
        tool_response("declare_done", {"summary": "all done"}, call_id="o3"),

        # outer 4: model fixes hello.py via a new step
        tool_response("propose_step", {
            "title": "Fix hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }, call_id="o4"),
        tool_response("write_file", {"path": "hello.py", "content": "print('fixed')\n"}, call_id="i6"),
        tool_response("mark_step_done", {"summary": "fixed hello.py"}, call_id="i7"),

        # outer 5: declare done again -- now accepted
        tool_response("declare_done", {"summary": "all done, for real"}, call_id="o5"),
    ])

    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="build stuff", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert len(state.completed_steps) == 3
    assert (agent.workspace.root / "hello.py").read_text() == "print('fixed')\n"

    # the rejection feedback must have actually reached the model, not just
    # been printed and discarded -- check every Turn sent in every call for
    # a tool_results entry mentioning the regression
    feedback_seen = any(
        "Regression detected" in tr.content
        for call in llm.calls
        for turn in call["history"]
        for tr in turn.tool_results
    )
    assert feedback_seen, "regression feedback never appeared in any prompt sent to the model"


def test_declare_done_with_no_completed_steps_skips_regression_check(tmp_path):
    """Zero completed steps means nothing to regress -- declare_done should
    behave exactly as before (governed only by final_acceptance_command)."""
    llm = FakeLLMClient([
        tool_response("declare_done", {"summary": "trivially done"}),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="do nothing", resume=False, final_acceptance_command="python -c \"pass\"")
    assert state.status == "done"
    assert state.completed_steps == []


def test_failed_attempt_junk_is_reverted_before_next_retry(tmp_path):
    """M3 regression: a failed attempt writes a stray file it never meant
    to keep (e.g. scratch/debug output) alongside a broken hello.py. The
    NEXT attempt fixes hello.py and passes -- without the revert, the
    stray file would get swept into that passing commit by `git add -A`.
    With the revert, the stray file must not exist after the step lands."""
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        # attempt 1: writes a broken hello.py AND an unrelated stray file, then fails the check
        tool_response("write_file", {"path": "hello.py", "content": "def broken(:\n"}, call_id="i1"),
        tool_response("write_file", {"path": "stray_debug.txt", "content": "oops\n"}, call_id="i2"),
        tool_response("mark_step_done", {"summary": "attempt 1"}, call_id="i3"),
        # attempt 2: fixes hello.py correctly and passes
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i4"),
        tool_response("mark_step_done", {"summary": "attempt 2"}, call_id="i5"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])

    agent = Agent(make_config(tmp_path, max_subtask_attempts=3), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")

    assert state.status == "done"
    assert (agent.workspace.root / "hello.py").read_text() == "print('hi')\n"
    # attempt 1's stray file must NOT have survived into the passing commit
    assert not (agent.workspace.root / "stray_debug.txt").exists()


def test_first_attempt_revert_is_a_harmless_noop(tmp_path):
    """Reverting on the very first attempt (nothing to revert yet) must
    not error or discard anything from before the step started."""
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create hello.py",
            "acceptance_command": "python -m py_compile hello.py",
        }),
        tool_response("write_file", {"path": "hello.py", "content": "print('hi')\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote it"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="make hello.py", resume=False,
                      final_acceptance_command="python -m py_compile hello.py")
    assert state.status == "done"
    assert (agent.workspace.root / "hello.py").read_text() == "print('hi')\n"


def test_call_llm_trims_oversized_history_before_sending(tmp_path):
    """A wiring regression: llm.trim_history_to_fit existed but was never
    actually called anywhere -- imported into agent.py, never invoked.
    The only overflow protection was compact_history_with_state_report,
    which runs AFTER a call returns, not before the next one goes out.
    This asserts every prompt _call_llm actually sends stays within
    budget, using a tiny context window to force the gap to matter, and
    checks against llm.count_tokens -- the exact function _call_llm
    actually calls -- not a separate re-derived estimate."""
    llm = FakeLLMClient([
        tool_response("read_file", {"path": "big.txt"}, call_id="r1"),
        tool_response("propose_step", {
            "title": "noop", "acceptance_command": "python -c \"pass\"",
        }, call_id="o2"),
        tool_response("write_file", {"path": "x.txt", "content": "ok\n"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "done"}, call_id="i2"),
        tool_response("declare_done", {"summary": "all done"}, call_id="o3"),
    ])

    config = make_config(tmp_path)
    config.llm.context_window_tokens = 4000
    config.budget.max_output_tokens = 1000
    agent = Agent(config, llm=llm)

    (agent.workspace.root / "big.txt").write_text("y" * 20_000, encoding="utf-8")

    state = agent.run(goal="do something small", resume=False,
                      final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"

    budget_tokens = max(config.llm.context_window_tokens - config.budget.max_output_tokens, 1)
    for call in llm.calls:
        size = llm.count_tokens(call["system"], call["history"], None)
        assert size <= budget_tokens, (
            f"a prompt of {size} tokens went out against a {budget_tokens}-token budget -- "
            "trim_history_to_fit isn't actually being applied before the call"
        )


def test_scratchpad_writes_are_capped(tmp_path):
    """The scratchpad lives in plain Turn.text (the outer loop's
    context_msg), not a tool_results entry -- trim_history_to_fit can
    never shrink it. It has no other size limit and is fully
    model-writable via update_scratchpad, so an unbounded write must be
    capped at the write site itself."""
    llm = FakeLLMClient([
        tool_response("update_scratchpad", {"scratchpad": "z" * 50_000}, call_id="s1"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="test scratchpad cap", resume=False,
                      final_acceptance_command="python -c \"pass\"")
    assert state.status == "done"
    assert len(state.scratchpad) <= Agent.MAX_SCRATCHPAD_CHARS + 100  # + truncation marker


def _first_inner_system_prompt_for(llm, title):
    calls = [c for c in llm.calls if c["history"] and f"STEP: {title}" in (c["history"][0].text or "")]
    assert calls, f"expected at least one LLM call scoped to step {title!r}"
    return calls[0]["system"]


def test_baseline_check_surfaces_real_failure_before_any_work(tmp_path):
    """
    The harness runs acceptance_command once, mechanically, right when the
    step is proposed -- before the coder has touched anything -- and the
    raw result (not an interpretation of it) must be visible in the first
    inner-loop system prompt.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create marker file",
            "acceptance_command": (
                "python -c \"import pathlib,sys; "
                "sys.exit(0 if pathlib.Path('marker.txt').exists() else 1)\""
            ),
        }),
        tool_response("write_file", {"path": "marker.txt", "content": "x"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "created marker.txt"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    system = _first_inner_system_prompt_for(llm, "Create marker file")
    assert "BASELINE" in system
    assert "exit 1" in system  # marker.txt didn't exist yet when the baseline ran


def test_baseline_check_flags_a_check_that_already_passes(tmp_path):
    """A check that exits 0 before any change is made is as broken as one
    that can never pass -- this is a plain exit-code fact, not a judgment
    call, so it's safe to surface mechanically."""
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Vacuous step",
            "acceptance_command": "python -c \"pass\"",
        }),
        tool_response("mark_step_done", {"summary": "nothing to do"}, call_id="i1"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    system = _first_inner_system_prompt_for(llm, "Vacuous step")
    assert "already exits 0" in system


def test_discard_and_restart_does_not_count_against_attempt_budget(tmp_path):
    """
    max_subtask_attempts=1 means exactly one mark_step_done attempt is
    allowed. If discard_and_restart silently counted as an attempt, the
    mark_step_done below would be attempt #2 and the step would fail.
    It should succeed instead -- proving discard_and_restart is free.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Write good.txt, not bad.txt",
            "acceptance_command": (
                "python -c \"import pathlib,sys; "
                "sys.exit(1 if pathlib.Path('bad.txt').exists() else "
                "(0 if pathlib.Path('good.txt').exists() else 1))\""
            ),
        }),
        # wrong approach first
        tool_response("write_file", {"path": "bad.txt", "content": "oops"}, call_id="i1"),
        tool_response("discard_and_restart", {"reason": "wrong file, starting over"}, call_id="i2"),
        # fresh attempt -- history resets, this is a new LLM call scoped to the same step
        tool_response("write_file", {"path": "good.txt", "content": "ok"}, call_id="i3"),
        tool_response("mark_step_done", {"summary": "wrote good.txt"}, call_id="i4"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    agent = Agent(make_config(tmp_path, max_subtask_attempts=1), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    assert len(state.completed_steps) == 1
    ws = agent.config.workspace_root
    assert not (ws / "bad.txt").exists()  # reverted
    assert (ws / "good.txt").exists()


def test_discard_and_restart_logs_an_event_with_the_reason(tmp_path):
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Retry step",
            "acceptance_command": "python -c \"pass\"",
        }),
        tool_response("discard_and_restart", {"reason": "changed my mind"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "done"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    config = make_config(tmp_path)
    agent = Agent(config, llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    events = [json.loads(line) for line in config.log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    discarded = [e for e in events if e.get("kind") == "attempt_discarded"]
    assert len(discarded) == 1
    assert discarded[0]["reason"] == "changed my mind"
    assert discarded[0]["title"] == "Retry step"


def test_revise_acceptance_command_swaps_the_check_and_reruns_baseline(tmp_path):
    """
    The whole point: a step whose original acceptance_command can never
    pass (references a file that will never exist) succeeds once the
    model revises it to something correct -- proving the swap actually
    takes effect for the mark_step_done check that follows.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Broken check",
            "acceptance_command": (
                "python -c \"import pathlib,sys; "
                "sys.exit(0 if pathlib.Path('this_will_never_exist.xyz').exists() else 1)\""
            ),
        }),
        tool_response("revise_acceptance_command", {
            "new_acceptance_command": "python -c \"pass\"",
            "reason": "original check referenced a file nothing in this step creates",
        }, call_id="i1"),
        tool_response("mark_step_done", {"summary": "corrected the check"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    assert len(state.completed_steps) == 1
    completed = state.completed_steps[0]
    assert completed.acceptance_command == "python -c \"pass\""
    assert completed.original_acceptance_command == (
        "python -c \"import pathlib,sys; "
        "sys.exit(0 if pathlib.Path('this_will_never_exist.xyz').exists() else 1)\""
    )


def test_revise_acceptance_command_is_capped_at_one_per_step(tmp_path):
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Two revisions attempted",
            "acceptance_command": "python -c \"pass\"",
        }),
        tool_response("revise_acceptance_command", {
            "new_acceptance_command": "python -c \"pass\"",
            "reason": "first revision",
        }, call_id="i1"),
        tool_response("revise_acceptance_command", {
            "new_acceptance_command": "python -c \"import sys; sys.exit(1)\"",
            "reason": "second revision, should be rejected",
        }, call_id="i2"),
        tool_response("mark_step_done", {"summary": "done"}, call_id="i3"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    # the second (rejected) revision must NOT have taken effect -- the
    # step must still pass with the command from the first revision
    assert state.completed_steps[0].acceptance_command == "python -c \"pass\""

    calls = [c for c in llm.calls if c["history"] and "Two revisions attempted" in (c["history"][0].text or "")]
    tool_result_texts = []
    for c in calls:
        for t in c["history"]:
            if t.role == "tool_results":
                tool_result_texts.extend(r.content for r in t.tool_results)
    assert any("already been revised once" in t for t in tool_result_texts)


def test_revise_acceptance_command_does_not_reset_repetition_guard_or_history(tmp_path):
    """Unlike discard_and_restart, revising the check should NOT wipe the
    in-progress attempt -- the model's prior work in this attempt should
    still be visible in the next turn's history."""
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Same attempt continues",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",
        }),
        tool_response("write_file", {"path": "marker.txt", "content": "x"}, call_id="i1"),
        tool_response("revise_acceptance_command", {
            "new_acceptance_command": (
                "python -c \"import pathlib,sys; "
                "sys.exit(0 if pathlib.Path('marker.txt').exists() else 1)\""
            ),
            "reason": "original check was unconditionally failing, unrelated to the work",
        }, call_id="i2"),
        tool_response("mark_step_done", {"summary": "done"}, call_id="i3"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    assert (agent.config.workspace_root / "marker.txt").exists()


def test_acceptance_command_revised_event_logs_old_new_and_reason(tmp_path):
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Loggable revision",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",
        }),
        tool_response("revise_acceptance_command", {
            "new_acceptance_command": "python -c \"pass\"",
            "reason": "old command always failed regardless of implementation",
        }, call_id="i1"),
        tool_response("mark_step_done", {"summary": "done"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    config = make_config(tmp_path)
    agent = Agent(config, llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    events = [json.loads(line) for line in config.log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    revised = [e for e in events if e.get("kind") == "acceptance_command_revised"]
    assert len(revised) == 1
    assert revised[0]["old_command"] == "python -c \"import sys; sys.exit(1)\""
    assert revised[0]["new_command"] == "python -c \"pass\""
    assert revised[0]["reason"] == "old command always failed regardless of implementation"
    assert revised[0]["title"] == "Loggable revision"


def test_failed_mark_step_done_tells_the_model_its_edits_were_reverted(tmp_path):
    """
    Regression: a failed mark_step_done silently reverts the workspace
    (agent.py's git_revert_to_last_commit call), but the model was never
    actually told that happened in the common case of continuing the same
    attempt -- last_attempt_note only ever reached history on a fresh
    attempt, which this path doesn't take. Confirmed against a real run:
    the model cited a stale "36 passed" result from before an unannounced
    revert and burned its remaining budget confused about missing files.
    The tool_result shown for every failed check must say so directly.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Create good.txt",
            "acceptance_command": (
                "python -c \"import pathlib,sys; "
                "sys.exit(0 if pathlib.Path('good.txt').exists() else 1)\""
            ),
        }),
        # wrong file first -- this attempt will fail
        tool_response("write_file", {"path": "wrong.txt", "content": "oops"}, call_id="i1"),
        tool_response("mark_step_done", {"summary": "wrote wrong.txt"}, call_id="i2"),
        # same attempt continues (no break) -- this is exactly the path
        # that used to leave the model uninformed
        tool_response("write_file", {"path": "good.txt", "content": "ok"}, call_id="i3"),
        tool_response("mark_step_done", {"summary": "wrote good.txt"}, call_id="i4"),
        tool_response("declare_done", {"summary": "done"}, call_id="o2"),
    ])
    agent = Agent(make_config(tmp_path), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    ws = agent.config.workspace_root
    assert not (ws / "wrong.txt").exists()  # actually reverted
    assert (ws / "good.txt").exists()

    calls = [c for c in llm.calls if c["history"] and "Create good.txt" in (c["history"][0].text or "")]
    tool_result_texts = []
    for c in calls:
        for t in c["history"]:
            if t.role == "tool_results":
                tool_result_texts.extend(r.content for r in t.tool_results)
    failure_texts = [t for t in tool_result_texts if "Acceptance check FAILED" in t]
    assert failure_texts, "expected at least one failed-check tool_result"
    assert any("reverted" in t for t in failure_texts)
    assert any("re-verify" in t or "no longer reflects" in t for t in failure_texts)


# ── autonomous replan tier ──────────────────────────────────────────────────

def test_replan_triggers_before_escalation_and_succeeds(tmp_path):
    """
    Core behavior: a step that exhausts max_subtask_attempts gets an
    autonomous replan (mechanical -- the harness decides, not a tool call)
    BEFORE any human escalation. If the replan produces a workable
    revision, the run finishes with no human input at all.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Impossible on first try",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",  # always fails
        }),
        tool_response("mark_step_done", {"summary": "sure"}, call_id="i1"),
        # replan pass: planner LLM revises the step
        text_response("OBJECTIVE: create a trivial file\nACCEPTANCE_COMMAND: python -c \"pass\""),
        tool_response("mark_step_done", {"summary": "fixed after replan"}, call_id="i2"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    agent = Agent(make_config(tmp_path, max_subtask_attempts=1, max_replan_cycles=1), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    assert len(state.completed_steps) == 1
    assert state.completed_steps[0].acceptance_command == "python -c \"pass\""

    events = [json.loads(line) for line in agent.config.log_file.read_text().splitlines()]
    assert any(e["kind"] == "replan_triggered" for e in events)


def test_replan_cap_forces_escalation(tmp_path, monkeypatch):
    """
    Once replan_count reaches max_replan_cycles, the harness stops trying
    on its own and escalates to the human -- the replan tier limits human
    involvement, it doesn't eliminate escalation for a genuinely stuck step.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Never works",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",
        }),
        tool_response("mark_step_done", {"summary": "try 1"}, call_id="i1"),
        # replan pass returns the same (still-broken) command -- doesn't help
        text_response("OBJECTIVE: same\nACCEPTANCE_COMMAND: python -c \"import sys; sys.exit(1)\""),
        tool_response("mark_step_done", {"summary": "try 2"}, call_id="i2"),
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": "abort")
    agent = Agent(make_config(tmp_path, max_subtask_attempts=1, max_replan_cycles=1), llm=llm)

    from autocoder.agent import AgentAborted
    import pytest
    with pytest.raises(AgentAborted):
        agent.run(goal="g", resume=False)

    events = [json.loads(line) for line in agent.config.log_file.read_text().splitlines()]
    replan_events = [e for e in events if e["kind"] == "replan_triggered"]
    assert len(replan_events) == 1  # replanned once, then escalated -- cap respected


def test_revise_acceptance_command_cap_resets_each_replan_cycle(tmp_path):
    """
    revise_acceptance_command's one-use cap is per REPLAN CYCLE, not per
    step lifetime: a revision used before a replan must not block a second
    revision used after that replan.
    """
    llm = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Broken check, revised twice across a replan",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",
        }),
        tool_response("revise_acceptance_command", {
            "new_acceptance_command": "python -c \"raise SystemExit(1)\"",
            "reason": "first revision, still deliberately fails",
        }, call_id="i1"),
        tool_response("mark_step_done", {"summary": "try"}, call_id="i2"),
        # exhausted at max_subtask_attempts=1 -> replan, keeps the same command
        text_response("OBJECTIVE: same\nACCEPTANCE_COMMAND: python -c \"raise SystemExit(1)\""),
        # this revision must NOT be rejected -- revisions_this_cycle reset to 0
        tool_response("revise_acceptance_command", {
            "new_acceptance_command": "python -c \"pass\"",
            "reason": "second revision, allowed in the new cycle",
        }, call_id="i3"),
        tool_response("mark_step_done", {"summary": "fixed"}, call_id="i4"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    agent = Agent(make_config(tmp_path, max_subtask_attempts=1, max_replan_cycles=1), llm=llm)
    state = agent.run(goal="g", resume=False, final_acceptance_command="python -c \"pass\"")

    assert state.status == "done"
    assert state.completed_steps[0].acceptance_command == "python -c \"pass\""

    tool_result_texts = []
    for c in llm.calls:
        for t in c["history"]:
            if t.role == "tool_results":
                tool_result_texts.extend(r.content for r in t.tool_results)
    assert not any("already been revised" in t for t in tool_result_texts)


def test_resume_gives_a_fresh_replan_budget_for_the_next_step(tmp_path):
    """
    replan_count/revisions_this_cycle live on the ephemeral StepContext,
    never on RunState -- consistent with session.py's existing invariant
    that a crash loses at most one in-progress step. A step proposed after
    resume gets the FULL max_replan_cycles again; nothing about a prior
    (even crashed) step's replan usage carries over, since it was never
    persisted in the first place.
    """
    cfg = make_config(tmp_path, max_subtask_attempts=1, max_replan_cycles=1)

    llm1 = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Step 1", "acceptance_command": "python -c \"pass\"",
        }),
        tool_response("mark_step_done", {"summary": "done"}, call_id="i1"),
        KeyboardInterrupt(),  # simulate the run being interrupted right after step 1
    ])
    agent1 = Agent(cfg, llm=llm1)
    state = agent1.run(goal="g", resume=False)
    assert state.status == "running"  # interrupted, not done -- exactly what "resume" is for
    assert len(state.completed_steps) == 1

    # Fresh Agent + fresh LLM client against the SAME session dir --
    # standin for a resumed process. Step 2 exhausts its attempt, gets a
    # full replan cycle, then succeeds.
    llm2 = FakeLLMClient([
        tool_response("propose_step", {
            "title": "Step 2, fails once",
            "acceptance_command": "python -c \"import sys; sys.exit(1)\"",
        }),
        tool_response("mark_step_done", {"summary": "try"}, call_id="i2"),
        text_response("OBJECTIVE: fixed\nACCEPTANCE_COMMAND: python -c \"pass\""),
        tool_response("mark_step_done", {"summary": "fixed"}, call_id="i3"),
        tool_response("declare_done", {"summary": "done"}, call_id="o1"),
    ])
    agent2 = Agent(cfg, llm=llm2)
    state2 = agent2.run(goal=None, resume=True, final_acceptance_command="python -c \"pass\"")

    assert state2.status == "done"
    assert len(state2.completed_steps) == 2
