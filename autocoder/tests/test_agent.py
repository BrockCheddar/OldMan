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
    agent = Agent(make_config(tmp_path, max_subtask_attempts=1), llm=llm)

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

    agent = Agent(make_config(tmp_path, max_subtask_attempts=1), llm=llm)
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

    agent = Agent(make_config(tmp_path, max_subtask_attempts=1), llm=llm)
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
    # calls: outer read, outer propose, inner write, inner mark, outer done
    assert len(llm.calls) == 5
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


