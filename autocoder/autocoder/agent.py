"""
Incremental agent loop.

Sequence per iteration:
  1. Build a context message from RunState (completed steps + scratchpad).
  2. Call the model. It either:
     a. Uses a workspace tool (read_file, write_file, edit_file, run_command,
        list_dir, search_code, git_diff, git_log, ask_human) -- executed
        immediately, result fed back, loop continues.
     b. Calls propose_step: declares ONE concrete next action and the
        acceptance command that will verify it. The harness then runs that
        plan through a tight inner loop (the model executes workspace tools
        until it calls mark_step_done), verifies with the real acceptance
        command, commits, appends to the log, and returns to the outer loop.
     c. Calls declare_done: claims the overall goal is complete. The harness
        verifies with the goal-level acceptance command (if supplied) or
        asks the human. If accepted, the run ends cleanly.
  3. Model never sees a pre-written list of future steps. Every decision is
     made after observing real evidence from the previous one.

Every LLM call is synchronous. Exactly one request is in flight at a time.
"""
from __future__ import annotations

import ast
import re
import sys

from .approval import ask_human_question
from .budget import BudgetTracker, BudgetExceeded
from .config import Config
from .lessons import LessonsStore
from .llm import build_llm_client, LLMError, Turn, ToolResult, trim_history_to_fit
from .osinfo import shell_description
from .planner import RunState, CompletedStep, StepContext
from .repetition import RepetitionGuard
from .session import SessionStore
from .state_report import compact_history_with_state_report
from .tools import ToolBox, ToolError, TOOL_SCHEMAS
from .workspace import Workspace


class AgentAborted(RuntimeError):
    pass


# ── additional tools for the incremental loop ─────────────────────────────

PROPOSE_STEP_TOOL = {
    "name": "propose_step",
    "description": (
        "Declare ONE concrete next step toward the goal. You must specify "
        "what you are about to do and a real shell command that will prove "
        "it is done (exits 0 on success). After calling this, use workspace "
        "tools to carry out the work, then call mark_step_done."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short label for this step, e.g. 'Create src/__init__.py'"},
            "objective": {
                "type": "string",
                "description": "What this step must accomplish. This is handed to a fresh coder context."
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files the coder should focus on. Use an empty array if no specific file is known yet."
            },
            "relevant_regions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Relevant functions, symbols, line ranges, selectors, or other code regions already identified."
            },
            "findings": {
                "type": "string",
                "description": "Compact evidence discovered during exploration: errors, root-cause clues, constraints, and important observations. Do not paste whole files."
            },
            "acceptance_command": {
                "type": "string",
                "description": (
                    "Real shell command run by the harness after mark_step_done. "
                    "Must exit 0 to accept the step. Use a compile check, test run, "
                    "or existence check -- never an empty string. "
                    "IMPORTANT: this runs through the OS shell described in your system "
                    "prompt -- on Windows/cmd.exe there are no pipe tools like tail, head, "
                    "grep, or find, and no POSIX builtins like 'test' or '['; use "
                    "'python -m py_compile', 'python script.py', 'dir filename', or a short "
                    "'python -c \"...\"' one-liner instead."
                ),
            },
        },
        "required": ["title", "objective", "files", "acceptance_command"],
    },
}

MARK_STEP_DONE_TOOL = {
    "name": "mark_step_done",
    "description": (
        "Declare the current step complete. The harness will run the "
        "acceptance_command you gave in propose_step. If it exits 0 the "
        "step is committed and you return to choosing the next step. "
        "If it fails you will see the real output and must keep working."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "What you did and why it satisfies the step."},
            "scratchpad": {"type": "string", "description": "Optional: update your working notes (replaces previous scratchpad)."},
        },
        "required": ["summary"],
    },
}

DISCARD_AND_RESTART_TOOL = {
    "name": "discard_and_restart",
    "description": (
        "Give up on the current approach and start this step over from a clean "
        "slate. Your uncommitted edits are reverted back to the last passing "
        "commit -- exactly like a failed mark_step_done -- but this does NOT "
        "count against your attempt budget. Use this instead of continuing to "
        "edit blind when you notice you're going in circles: repeated edits to "
        "the same file without real progress, or you've realized your current "
        "approach fundamentally can't work. Prefer this over patching more "
        "guesses on top of an already-tangled state."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "What you were trying, and why it isn't working -- so you don't repeat it.",
            },
        },
        "required": ["reason"],
    },
}

DECLARE_DONE_TOOL = {
    "name": "declare_done",
    "description": (
        "Declare that the overall goal is fully complete. Call this DIRECTLY -- "
        "do NOT create a propose_step just to 'declare completion' or 'finalize the work'. "
        "declare_done is independently verified for you (a real acceptance command, or a "
        "human), so it doesn't need its own step or its own acceptance check first. "
        "Only call this when ALL necessary work is done and verified."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief summary of everything accomplished."},
        },
        "required": ["summary"],
    },
}

UPDATE_SCRATCHPAD_TOOL = {
    "name": "update_scratchpad",
    "description": "Update your working notes without marking any step done.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scratchpad": {"type": "string"},
        },
        "required": ["scratchpad"],
    },
}

# Outer loop gets read/explore tools only -- NOT write_file/edit_file. Those
# mutate files with zero acceptance-check gate, and the outer loop is meant
# for observing and deciding, not acting. Seen in practice: after a step
# passed its check and was committed, the model used write_file again from
# the outer loop to silently overwrite that already-verified file, with
# nothing re-checking it and the run's own bookkeeping never reflecting
# that a regression could have just happened. Any real file change must go
# through propose_step -> the inner loop, where mark_step_done actually
# re-verifies the result.
_OUTER_SAFE_TOOL_NAMES = {
    "read_file", "list_dir", "search_code", "run_command", "git_diff", "git_log",
    "ask_human", "record_decision",
}
OUTER_WORKSPACE_TOOLS = [t for t in TOOL_SCHEMAS if t["name"] in _OUTER_SAFE_TOOL_NAMES]
OUTER_TOOLS = OUTER_WORKSPACE_TOOLS + [PROPOSE_STEP_TOOL, DECLARE_DONE_TOOL, UPDATE_SCRATCHPAD_TOOL]

# Tools available inside an active step (executing it) -- full access,
# because mark_step_done re-verifies with the real acceptance command.
INNER_TOOLS = TOOL_SCHEMAS + [MARK_STEP_DONE_TOOL, DISCARD_AND_RESTART_TOOL]


# ── Agent ──────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, config: Config, llm=None):
        self.config = config
        self.workspace = (
            Workspace.create(config.workspace_root, config.source_repo)
            if not (config.workspace_root / ".git").exists()
            else Workspace(config.workspace_root)
        )
        self.session = SessionStore(config)
        self.llm = llm if llm is not None else build_llm_client(config.llm)
        self.budget = BudgetTracker(config.budget)
        self.toolbox = ToolBox(self.workspace, config, on_pause=self.budget.pause, on_resume=self.budget.resume)
        self.lessons = LessonsStore(config.lessons_file)
        # Set by _escalate() when the human gives guidance (not abort); only
        # turned into a recorded lesson once something afterward actually
        # succeeds -- an escalation that never resolves teaches nothing.
        self._pending_escalation_lesson: dict | None = None

    # ── public entry point ─────────────────────────────────────────────────

    def run(self, goal: str | None, resume: bool,
            final_acceptance_command: str | None = None) -> RunState:
        state = self.session.load_state() if resume else None
        if state is None:
            if not goal:
                raise ValueError("no existing session and no goal given")
            state = RunState(goal=goal)
            self.session.save_state(state)
            print(f"[run] goal: {goal!r}")

        try:
            self._outer_loop(state, final_acceptance_command)
        except KeyboardInterrupt:
            self.session.save_state(state)
            print("\n[interrupted] session saved -- resume with `autocoder resume --workspace ...`")
            return state

        self._print_summary(state)
        return state

    # ── LLM call wrapper: backend failures don't crash the run ─────────────

    MAX_SCRATCHPAD_CHARS = 4000

    def _set_scratchpad(self, state: "RunState", text: str) -> None:
        """Every scratchpad write goes through here. The scratchpad is
        embedded directly into the outer loop's context_msg as plain text,
        not a tool_results entry -- trim_history_to_fit only ever shrinks
        tool_results, so an unbounded scratchpad would sail right past
        that safety net. This is the actual cap for this specific field,
        keeping the most recent content (usually what's still relevant)
        rather than the oldest."""
        if len(text) > self.MAX_SCRATCHPAD_CHARS:
            text = "...[older scratchpad content trimmed]...\n" + text[-self.MAX_SCRATCHPAD_CHARS:]
        state.scratchpad = text

    def _call_llm(self, state: RunState, system: str, history: list[Turn], tools: list[dict]):
        """
        Wraps self.llm.complete(). A backend failure/timeout (LLMError) no
        longer propagates as an unhandled crash -- it's reported, the human
        is asked to retry or abort, and completed work stays exactly as
        already saved on disk either way.
        """
        while True:
            try:
                # Last line of defense: shrink oversized tool_results in
                # place if the REAL prompt token count (via the backend's
                # own tokenizer, not a character guess) would exceed the
                # real context window. compact_history_with_state_report
                # (see call sites above/below) handles the common case at
                # coarser points in the loop, but it only runs AFTER a call
                # returns and results are appended -- a turn appended just
                # before THIS call (e.g. a growing scratchpad/completed-
                # steps summary) could still overflow with nothing catching
                # it. Inside the try so a tokenize-endpoint failure gets the
                # same retry/abort handling as any other backend failure.
                _, trimmed = trim_history_to_fit(
                    history, system, self.llm.count_tokens,
                    self.config.llm.context_window_tokens,
                    reserve_output_tokens=self.config.budget.max_output_tokens,
                    tools=tools,
                )
                if trimmed:
                    print("[context] trimmed oversized tool output to fit context window")
                    self.session.log_event("context_trim", {})

                response = self.llm.complete(
                    system=system, history=history, tools=tools,
                    max_tokens=self.config.budget.max_output_tokens,
                )
                self.budget.record_usage(response.input_tokens, response.output_tokens)
                return response
            except LLMError as e:
                print(f"\n[llm-error] {e}")
                self.budget.pause()
                choice = ask_human_question(
                    "The model backend failed or timed out (see above). "
                    "Type 'retry' to try that request again, or 'abort' to stop "
                    "this run (everything completed so far is already saved):"
                )
                self.budget.resume()
                if choice.strip().lower() == "abort":
                    self.session.log_event("llm_error_abort", {"error": str(e)})
                    state.status = "aborted"
                    self.session.save_state(state)
                    raise AgentAborted()
                self.session.log_event("llm_error_retry", {"error": str(e)})

    # ── outer loop: choose next step ───────────────────────────────────────

    def _build_outer_state_report(self, state: RunState) -> str:
        """
        Same idea as _build_state_report, applied at the run level instead
        of the step level. Built entirely from live checks against the
        real workspace and the RunState's own current fields -- never from
        old chat text -- so it can never be stale, unlike a truncated
        fragment of old tool output.
        """
        lines = [
            "STATE REPORT (outer loop)",
            f"Goal: {state.goal}",
            "",
            f"Completed steps so far:\n{state.completed_summary_text()}",
            "",
            f"Scratchpad:\n{state.scratchpad or '(empty)'}",
            "",
            "Current workspace files (live listing, not memory):",
        ]
        try:
            listing = self.toolbox.dispatch("list_dir", {"path": "."})
        except ToolError as e:
            listing = f"(could not list workspace: {e})"
        lines.append(listing[:1500])

        git_status = self.workspace.git_status()
        lines.append(f"\nGit status (uncommitted changes, live check):\n{git_status.stdout[:800] or '(clean)'}")

        return "\n".join(lines)

    def _outer_loop(self, state: RunState, final_acceptance_command: str | None) -> None:
        history: list[Turn] = []
        self.budget.new_outer_span()
        repetition_guard = RepetitionGuard()

        while state.status == "running":
            # Reset for record_decision's originating_step -- _inner_loop
            # overwrites this to the active step's title for its duration.
            self.toolbox.current_context = "outer loop (exploration)"
            warning = self.budget.wall_clock_warning()
            if warning:
                print(f"[budget] {warning}")

            try:
                self.budget.outer_step()
            except BudgetExceeded as e:
                print(f"[budget] {e}")
                self._escalate(state, reason=str(e))
                if state.status != "running":
                    return
                continue

            system = self._outer_system_prompt(state)
            context_msg = (
                f"COMPLETED STEPS SO FAR:\n{state.completed_summary_text()}\n\n"
                f"SCRATCHPAD:\n{state.scratchpad or '(empty)'}\n\n"
                "What is your next action?"
            )
            history.append(Turn(role="user", text=context_msg))

            response = self._call_llm(state, system, history, OUTER_TOOLS)
            history.append(Turn(role="assistant", text=response.text, tool_calls=response.tool_calls))

            if not response.tool_calls:
                if response.text:
                    _safe_print(f"  [model] {response.text.strip()[:300]}")
                history.append(Turn(role="user", text=(
                    "Use propose_step to declare your next concrete action, "
                    "declare_done if the goal is fully complete, "
                    "or a workspace tool if you need to explore first."
                )))
                if self._prompt_over_budget(system, history, OUTER_TOOLS):
                    report = self._build_outer_state_report(state)
                    history = compact_history_with_state_report(history, report)
                    self.session.log_event("state_report_compaction", {"scope": "outer"})
                continue

            # process tool calls one at a time in the order returned
            tool_results: list[ToolResult] = []
            acted = False
            force_escalate_reason: str | None = None
            history_reset = False

            for call in response.tool_calls:

                if call.name == "mark_step_done":
                    # Model called the inner-loop tool from the outer loop --
                    # it skipped propose_step. Reject explicitly and redirect.
                    tool_results.append(ToolResult(
                        tool_call_id=call.id, is_error=True,
                        content=(
                            "mark_step_done is only valid inside an active step. "
                            "You must call propose_step first to declare what you "
                            "are about to do and its acceptance command, then use "
                            "workspace tools to do the work, then call mark_step_done."
                        ),
                    ))
                    acted = True
                    continue

                if call.name == "declare_done":
                    summary = call.input.get("summary", "")
                    accepted, feedback = self._verify_done(state, summary, final_acceptance_command)
                    repetition_guard.checkpoint()
                    if accepted:
                        state.status = "done"
                        self._finalize_pending_lesson(fix_summary=f"goal accepted as done: {summary}")
                        self.session.save_state(state)
                        return
                    # not accepted -- tell the model exactly why, not just "keep going"
                    self._set_scratchpad(state, state.scratchpad + f"\n[Feedback on declare_done attempt]: {feedback}")
                    self.session.save_state(state)
                    tool_results.append(ToolResult(
                        tool_call_id=call.id, is_error=True,
                        content=f"Goal not yet accepted.\n{feedback}\nKeep working, then try declare_done again.",
                    ))
                    acted = True
                    continue

                if call.name == "propose_step":
                    title = call.input.get("title", "step")
                    objective = call.input.get("objective", title)
                    acceptance_cmd = call.input.get("acceptance_command", "")
                    files = call.input.get("files", []) or []
                    relevant_regions = call.input.get("relevant_regions", []) or []
                    findings = call.input.get("findings", "") or ""
                    # A model can return a bare string ("main.py") instead of
                    # a list (["main.py"]). Iterating a string yields its
                    # characters one by one -- silently, with no error --
                    # which would corrupt every state report built from it.
                    if isinstance(files, str):
                        files = [files]
                    if isinstance(relevant_regions, str):
                        relevant_regions = [relevant_regions]
                    if not acceptance_cmd:
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content="acceptance_command is required and must be non-empty.",
                        ))
                        acted = True
                        continue
                    syntax_problem = _validate_python_dash_c_syntax(acceptance_cmd)
                    if syntax_problem:
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content=(
                                f"acceptance_command is invalid and would fail every time, "
                                f"regardless of what you implement: {syntax_problem}\n"
                                f"Command was: {acceptance_cmd}\n"
                                "Call propose_step again with a corrected, valid acceptance_command."
                            ),
                        ))
                        acted = True
                        continue
                    step_context = StepContext(
                        title=title,
                        objective=objective,
                        acceptance_command=acceptance_cmd,
                        files=[str(x) for x in files],
                        relevant_regions=[str(x) for x in relevant_regions],
                        findings=findings,
                    )
                    print(f"\n[step] {state.next_index()}: {title}")
                    print(f"  [check-will-run] {acceptance_cmd}")
                    step_context.baseline_check_result = self._run_baseline_check(step_context)
                    # IMPORTANT: do not seed the coder with planner history.
                    # The structured StepContext is the explicit context boundary.
                    completed, failure_reason = self._inner_loop(state, step_context)
                    if completed is None:
                        # step failed after max attempts -- escalate with an
                        # accurate reason (real check failures vs the model
                        # derailing and running out of steps are different
                        # situations and shouldn't be reported the same way)
                        self._escalate(state, reason=failure_reason)
                        if state.status != "running":
                            return
                        # human gave guidance (not abort) -- start fresh with
                        # it in the scratchpad rather than ending the run
                        history = []
                        history_reset = True
                        acted = True
                        break
                    state.completed_steps.append(completed)
                    state.status = "running"
                    self._finalize_pending_lesson(fix_summary=f"step '{title}' completed: {completed.summary}")
                    self.session.save_state(state)
                    self.session.log_event("step_completed", {"index": completed.index, "title": completed.title})
                    # reset outer history after each completed step so context
                    # stays bounded and old tool output doesn't pile up
                    history = []
                    history_reset = True
                    self.budget.new_outer_span()  # real progress -- grant a fresh outer budget
                    acted = True
                    break  # one propose_step per outer turn

                if call.name == "update_scratchpad":
                    self._set_scratchpad(state, call.input.get("scratchpad", ""))
                    self.session.save_state(state)
                    tool_results.append(ToolResult(tool_call_id=call.id, content="Scratchpad updated."))
                    acted = True
                    continue

                # workspace tool
                if call.name not in _OUTER_SAFE_TOOL_NAMES:
                    # Not offered at this level -- reject explicitly rather
                    # than silently dispatching it if the model emits it
                    # anyway (local models don't always strictly respect the
                    # offered tool list). write_file/edit_file specifically
                    # must never mutate a file outside the verified
                    # propose_step -> inner-loop path.
                    tool_results.append(ToolResult(
                        tool_call_id=call.id, is_error=True,
                        content=(
                            f"'{call.name}' is not available here -- this is the outer loop, "
                            "read-only exploration only. Call propose_step to make an actual "
                            "change; it hands off to a step where your work gets verified."
                        ),
                    ))
                    acted = True
                    continue
                print(f"  [tool] {call.name}({_short(call.input)})")
                try:
                    output = self.toolbox.dispatch(call.name, call.input)
                    warning = repetition_guard.record(call.name, call.input)
                    if warning:
                        output = f"{output}\n\n{warning}"
                        print(f"  {warning}")
                        if repetition_guard.should_force_escalate():
                            force_escalate_reason = (
                                f"stuck repeating {call.name} on the same target even after "
                                f"{repetition_guard.force_escalate_after_fires} repetition warnings "
                                "were already given and ignored"
                            )
                    tool_results.append(ToolResult(tool_call_id=call.id, content=output))
                except ToolError as e:
                    print(f"    -> ERROR: {e}")
                    tool_results.append(ToolResult(tool_call_id=call.id, content=f"ERROR: {e}", is_error=True))
                self.session.log_event("tool_call", {"name": call.name})
                acted = True
                if force_escalate_reason:
                    break

            if tool_results and not history_reset:
                history.append(Turn(role="tool_results", tool_results=tool_results))
                if self._prompt_over_budget(system, history, OUTER_TOOLS):
                    report = self._build_outer_state_report(state)
                    history = compact_history_with_state_report(history, report)
                    self.session.log_event("state_report_compaction", {"scope": "outer"})

            if force_escalate_reason:
                self._escalate(state, reason=force_escalate_reason)
                if state.status != "running":
                    return
                history = []
                repetition_guard.checkpoint()
                continue

    def _prompt_over_budget(self, system: str, history: list[Turn], tools: list[dict]) -> bool:
        budget_tokens = max(self.config.llm.context_window_tokens - self.config.budget.max_output_tokens, 1)
        return self.llm.count_tokens(system, history, tools) > budget_tokens

    def _build_state_report(self, step_context: StepContext, attempt: int, max_attempts: int,
                             last_check_summary: str = "") -> str:
        """
        Builds a compact 'you are HERE NOW' block from the real, current
        workspace state -- a live file listing, live content of the files
        in scope, live git status -- never from old chat text. Used to
        replace old history once it grows too large, instead of truncating
        individual tool_result strings into a head piece, a gap, and a
        tail piece that the model must guess the meaning of.
        """
        lines = [
            f"STATE REPORT -- attempt {attempt}/{max_attempts}",
            f"Step: {step_context.title}",
            f"Objective: {step_context.objective}",
            f"Acceptance command: {step_context.acceptance_command}",
            "",
            "Current workspace files (live listing, not memory):",
        ]
        try:
            listing = self.toolbox.dispatch("list_dir", {"path": "."})
        except ToolError as e:
            listing = f"(could not list workspace: {e})"
        lines.append(listing[:1500])

        if step_context.files:
            lines.append("\nCurrent content of files in scope (live read, not memory):")
            for path in step_context.files[:5]:
                try:
                    content = self.toolbox.dispatch("read_file", {"path": path, "limit": 200})
                except ToolError as e:
                    content = f"(could not read {path}: {e})"
                lines.append(f"\n--- {path} ---\n{content[:3000]}")

        git_status = self.workspace.git_status()
        lines.append(f"\nGit status (uncommitted changes, live check):\n{git_status.stdout[:800] or '(clean)'}")

        if last_check_summary:
            lines.append(f"\nMost recent acceptance-check result:\n{last_check_summary}")

        if step_context.findings:
            lines.append(f"\nPlanner findings (from exploration before this step):\n{step_context.findings}")

        return "\n".join(lines)

    # ── inner loop: execute one declared step ──────────────────────────────

    def _run_baseline_check(self, step_context: StepContext) -> str:
        """
        Mechanical, model-agnostic: run acceptance_command once against the
        untouched workspace, right when the step is proposed, before any
        code changes exist. No interpretation of the result -- just the raw
        exit code/stdout/stderr, formatted the same way a real failed
        mark_step_done attempt's failure_detail already is, for consistency.

        This exists so the coder starts from the real, actual result of its
        own acceptance command instead of discovering it fresh after several
        failed guesses -- and, as a free side effect of the same one call,
        catches acceptance commands that already pass with no changes made
        (a check that can't fail is as broken as one that can't pass).
        """
        result = self.toolbox.run_gated_command(
            step_context.acceptance_command,
            timeout=self.config.budget.default_command_timeout,
        )
        header = (
            f"$ {step_context.acceptance_command}\n"
            f"exit {result.exit_code}{' (TIMED OUT)' if result.timed_out else ''}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        if result.exit_code == 0 and not result.timed_out:
            return (
                f"{header}\n\n"
                "NOTE: this already exits 0 against the CURRENT workspace, before "
                "any change has been made. Either this step isn't necessary, or "
                "the acceptance command doesn't actually verify the new capability "
                "-- double check it before proceeding."
            )
        return header

    def _inner_loop(
        self,
        state: RunState,
        step_context: StepContext,
    ) -> tuple[CompletedStep | None, str]:
        """
        Runs the model in a tight loop with only workspace tools + mark_step_done
        until the acceptance command passes or attempts are exhausted.
        Returns (CompletedStep, "") on success, or (None, failure_reason) on
        failure -- failure_reason accurately distinguishes a real failed
        acceptance check from the model derailing and running out of its
        step budget without ever attempting a check, since those are
        different situations and reporting them identically to the human
        (as "failed its acceptance check repeatedly") is misleading.
        """
        attempts = 0
        check_failures = 0
        derailed_attempts = 0
        last_failure_detail = ""    # only set by a REAL failed acceptance check
        last_attempt_note = ""      # carried into the next attempt's fresh history
        last_check_summary = ""     # most recent real acceptance-check result, for state reports
        repetition_guard = RepetitionGuard()
        # record_decision's originating_step label for the duration of this
        # step; _outer_loop resets it back to the default on its next turn.
        self.toolbox.current_context = f"step: {step_context.title}"

        while attempts < self.config.budget.max_subtask_attempts:
            self.budget.new_attempt()
            # Clean slate: discard whatever the previous attempt (if any)
            # left behind. Without this, a failed attempt's half-finished
            # or broken edits stay in the working tree and get inherited by
            # the next attempt -- and if THAT attempt passes, git_commit_all's
            # `git add -A` sweeps the leftover junk in as part of a
            # "passing" commit. A no-op on the first attempt (nothing to
            # revert yet).
            self.workspace.git_revert_to_last_commit()
            system = self._inner_system_prompt(state, step_context)
            # Fresh coder context every attempt. The planner conversation is
            # deliberately NOT carried across this boundary. Only the compact
            # StepContext plus actionable information from a failed attempt
            # is allowed into the new context.
            history: list[Turn] = []
            history.append(Turn(role="user", text=step_context.prompt_text()))
            if last_attempt_note:
                history.append(Turn(role="user", text=last_attempt_note))
            repetition_guard.checkpoint()

            while True:
                try:
                    self.budget.step()
                except BudgetExceeded as e:
                    self.session.log_event("step_budget_exceeded", {"title": step_context.title, "detail": str(e)})
                    derailed_attempts += 1
                    attempts += 1
                    last_attempt_note = (
                        "A previous attempt on this step ran out of its step budget "
                        "without ever calling mark_step_done. Its edits have been reverted -- "
                        "the workspace is back to the last passing commit, so start clean. "
                        "Work directly and efficiently toward mark_step_done this time."
                    )
                    break  # try another attempt, with fresh history

                response = self._call_llm(state, system, history, INNER_TOOLS)
                history.append(Turn(role="assistant", text=response.text, tool_calls=response.tool_calls))

                if not response.tool_calls:
                    if response.text:
                        _safe_print(f"    [model] {response.text.strip()[:300]}")
                    history.append(Turn(role="user", text=(
                        "Use workspace tools to implement the step, "
                        "then call mark_step_done when it is ready to verify."
                    )))
                    if self._prompt_over_budget(system, history, INNER_TOOLS):
                        report = self._build_state_report(
                            step_context, attempts + 1,
                            self.config.budget.max_subtask_attempts, last_check_summary,
                        )
                        history = compact_history_with_state_report(history, report)
                        self.session.log_event("state_report_compaction", {"title": step_context.title})
                    continue

                mark_call = next((c for c in response.tool_calls if c.name == "mark_step_done"), None)
                discard_call = next((c for c in response.tool_calls if c.name == "discard_and_restart"), None)
                tool_results: list[ToolResult] = []

                for call in response.tool_calls:
                    if call.name in ("mark_step_done", "discard_and_restart"):
                        continue
                    print(f"    [tool] {call.name}({_short(call.input)})")
                    try:
                        output = self.toolbox.dispatch(call.name, call.input)
                        warning = repetition_guard.record(call.name, call.input)
                        if warning:
                            output = f"{output}\n\n{warning}"
                            print(f"    {warning}")
                            if repetition_guard.should_force_escalate():
                                return None, (
                                    f"step '{step_context.title}': stuck repeating {call.name} on the "
                                    f"same target even after {repetition_guard.force_escalate_after_fires} "
                                    "repetition warnings were already given and ignored"
                                )
                        tool_results.append(ToolResult(tool_call_id=call.id, content=output))
                    except ToolError as e:
                        print(f"      -> ERROR: {e}")
                        tool_results.append(ToolResult(tool_call_id=call.id, content=f"ERROR: {e}", is_error=True))
                    self.session.log_event("tool_call", {"name": call.name})

                if discard_call is not None:
                    # Voluntary clean-slate reset -- deliberately does NOT
                    # increment `attempts`. The whole point is to give a
                    # cheap alternative to blindly patching on top of a
                    # tangled state, so there's no cost that would make a
                    # model avoid reaching for it (see mark_step_done's
                    # revert-on-failure, which this is meant to complement,
                    # not replace). The actual revert happens at the top of
                    # the next attempt-loop iteration, same as the
                    # BudgetExceeded path above -- not duplicated here.
                    reason = discard_call.input.get("reason", "")
                    print(f"    [discard] restarting step '{step_context.title}': {reason[:200]}")
                    self.session.log_event("attempt_discarded", {"title": step_context.title, "reason": reason})
                    repetition_guard.checkpoint()
                    last_attempt_note = (
                        f"A previous attempt called discard_and_restart: {reason}\n"
                        "Its edits have been reverted -- you're starting from the last "
                        "passing commit again, not from what it left behind. Take a "
                        "genuinely different approach this time."
                    )
                    break  # fresh attempt, same as BudgetExceeded above -- attempts NOT incremented

                if mark_call is not None:
                    attempts += 1
                    repetition_guard.checkpoint()
                    summary = mark_call.input.get("summary", "")
                    if mark_call.input.get("scratchpad"):
                        self._set_scratchpad(state, mark_call.input["scratchpad"])

                    print(f"    [check] {step_context.acceptance_command}")
                    result = self.toolbox.run_gated_command(
                        step_context.acceptance_command,
                        timeout=self.config.budget.default_command_timeout,
                    )
                    passed = result.exit_code == 0 and not result.timed_out
                    self.session.log_event("acceptance_check", {
                        "title": step_context.title, "attempt": attempts,
                        "exit_code": result.exit_code, "timed_out": result.timed_out,
                    })

                    if passed:
                        print(f"    [check] PASSED (attempt {attempts}/{self.config.budget.max_subtask_attempts})")
                        self.workspace.git_commit_all(
                            f"[autocoder] step {state.next_index()}: {step_context.title}\n\n{summary}"
                        )
                        if last_failure_detail:
                            self.lessons.add(
                                context=f"step: {step_context.title}",
                                symptom=last_failure_detail,
                                fix=f"eventually succeeded with: {summary}",
                            )
                        return CompletedStep(
                            index=state.next_index(),
                            title=step_context.title,
                            summary=summary,
                            acceptance_command=step_context.acceptance_command,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            exit_code=result.exit_code,
                        ), ""

                    failure_detail = (
                        f"$ {step_context.acceptance_command}\n"
                        f"exit {result.exit_code}{' (TIMED OUT)' if result.timed_out else ''}\n"
                        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                    )
                    last_failure_detail = failure_detail
                    last_check_summary = failure_detail[-800:]
                    last_attempt_note = (
                        f"A previous attempt's mark_step_done failed this exact acceptance check:\n"
                        f"{failure_detail}\nThat attempt's edits have been reverted -- you're starting "
                        "from the last passing commit again, not from what it left behind. "
                        "Fix the actual underlying problem this time, don't just retry the same thing."
                    )
                    check_failures += 1
                    print(f"    [check] FAILED (attempt {attempts}/{self.config.budget.max_subtask_attempts})")
                    # discard this attempt's edits now, not just at the top
                    # of the next outer-loop pass -- a failed mark_step_done
                    # keeps the SAME inner history/context going (see
                    # last_attempt_note above), it doesn't restart the loop,
                    # so this is the actual retry point that needs the revert.
                    self.workspace.git_revert_to_last_commit()
                    if attempts >= self.config.budget.max_subtask_attempts:
                        return None, _inner_loop_failure_reason(step_context.title, check_failures, derailed_attempts)

                    tool_results.append(ToolResult(
                        tool_call_id=mark_call.id, is_error=True,
                        content=(
                            f"Acceptance check FAILED (attempt {attempts}/"
                            f"{self.config.budget.max_subtask_attempts}).\n"
                            f"{failure_detail}\n"
                            "Keep working, then call mark_step_done again."
                        ),
                    ))

                if tool_results:
                    history.append(Turn(role="tool_results", tool_results=tool_results))
                    if self._prompt_over_budget(system, history, INNER_TOOLS):
                        report = self._build_state_report(
                            step_context, attempts,
                            self.config.budget.max_subtask_attempts, last_check_summary,
                        )
                        history = compact_history_with_state_report(history, report)
                        self.session.log_event("state_report_compaction", {"title": step_context.title})

        return None, _inner_loop_failure_reason(step_context.title, check_failures, derailed_attempts)

    # ── done verification ──────────────────────────────────────────────────

    def _check_no_regressions(self, state: RunState) -> str | None:
        """Re-run every previously-completed step's acceptance command.
        Without this, step 7 can silently break what step 3 verified --
        each step's check only runs once, at the moment it's marked done,
        and git_commit_all's `git add -A` sweeps up whatever the working
        tree looks like at that point regardless. Returns None if every
        prior step still passes, or feedback text (naming which step
        regressed) to send back to the model if not."""
        for step in state.completed_steps:
            result = self.toolbox.run_gated_command(
                step.acceptance_command,
                timeout=self.config.budget.default_command_timeout,
            )
            if result.exit_code != 0 or result.timed_out:
                print(f"[regression-check] step {step.index} ('{step.title}') now FAILS")
                return (
                    f"Regression detected: step {step.index} ('{step.title}') used to pass "
                    f"its acceptance command but does not anymore.\n"
                    f"$ {step.acceptance_command}\n"
                    f"exit {result.exit_code}{' (TIMED OUT)' if result.timed_out else ''}\n"
                    f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}\n"
                    "Fix this regression before declaring the goal done."
                )
        if state.completed_steps:
            print(f"[regression-check] {len(state.completed_steps)}/{len(state.completed_steps)} prior steps still pass")
        return None

    def _verify_done(self, state: RunState, summary: str,
                      final_acceptance_command: str | None) -> tuple[bool, str]:
        """Returns (accepted, feedback). feedback is empty when accepted;
        otherwise it's real command output (acceptance-command path) or
        whatever the human typed (interactive path) -- either way it goes
        back to the model, not just a generic 'try again'."""
        regression_feedback = self._check_no_regressions(state)
        if regression_feedback:
            return False, regression_feedback

        if final_acceptance_command:
            print(f"[done-check] {final_acceptance_command}")
            result = self.toolbox.run_gated_command(
                final_acceptance_command,
                timeout=self.config.budget.default_command_timeout,
            )
            if result.exit_code == 0 and not result.timed_out:
                print(f"[done-check] PASSED")
                return True, ""
            print(f"[done-check] FAILED (exit {result.exit_code})")
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            feedback = (
                f"Final acceptance check FAILED.\n$ {final_acceptance_command}\n"
                f"exit {result.exit_code}{' (TIMED OUT)' if result.timed_out else ''}\n"
                f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
            )
            return False, feedback

        # No final acceptance command -- ask the human.
        print(f"\n[done] model declares goal complete:\n  {summary}")
        self.budget.pause()
        answer = input("Accept as done? [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            self.budget.resume()
            return True, ""
        reason = input("What's missing or not right? (sent back to the agent): ").strip()
        self.budget.resume()
        if not reason:
            reason = "Human rejected without giving a specific reason. Re-examine your work critically."
        return False, reason

    def _finalize_pending_lesson(self, fix_summary: str) -> None:
        pending = self._pending_escalation_lesson
        if pending is None:
            return
        self.lessons.add(
            context=pending["context"],
            symptom=f"{pending['symptom']}\nHuman guidance given: {pending['guidance']}",
            fix=f"run continued and succeeded afterward: {fix_summary}",
        )
        self._pending_escalation_lesson = None

    # ── escalation ────────────────────────────────────────────────────────

    def _escalate(self, state: RunState, reason: str) -> None:
        self.budget.pause()
        guidance = ask_human_question(
            f"The agent is stuck: {reason}\n\n"
            f"Completed so far:\n{state.completed_summary_text()}\n\n"
            "Type guidance to continue with, 'abort' to stop:"
        )
        self.budget.resume()
        if guidance.strip().lower() == "abort":
            state.status = "aborted"
            self.session.save_state(state)
            raise AgentAborted()
        self._set_scratchpad(state, state.scratchpad + f"\n[Human guidance]: {guidance}")
        self.session.save_state(state)
        # Recorded as a real lesson only if something afterward actually
        # succeeds (see the two _finalize_pending_lesson() call sites below).
        # An escalation that never resolves teaches nothing verified.
        self._pending_escalation_lesson = {
            "context": f"escalation: {reason}",
            "symptom": reason,
            "guidance": guidance,
        }
        # Fresh budget span -- without this, the next outer_step() call would
        # immediately re-raise on an already-over-cap counter and guidance
        # could never actually resume the run.
        self.budget.new_outer_span()

    # ── system prompts ─────────────────────────────────────────────────────

    def _outer_system_prompt(self, state: RunState) -> str:
        return f"""You are an autonomous coding agent working in a real git repository.

OS: {shell_description()}

GOAL: {state.goal}

LESSONS FROM PAST RUNS IN THIS WORKSPACE (verified -- each was an actual problem
followed by a confirmed successful outcome, not a guess):
{self.lessons.summary_text()}

ACTIVE DECISIONS FOR THIS PROJECT (durable choices already made -- do not
contradict these; if you make a NEW architecture/technology/convention choice
here, call record_decision right away for that one choice -- one decision
per call, in the moment you decide it, not bundled with others or saved up
for later):
{self.toolbox.decisions.summary_text()}

Your job: make incremental, verifiable progress toward the goal, in small
steps -- not one step that quietly builds the whole thing.
Each iteration: either explore the workspace with READ-ONLY tools to gather
evidence (read_file, list_dir, search_code, run_command, git_diff, git_log),
then call propose_step to declare ONE concrete next action with a real
acceptance command -- or call declare_done when the goal is fully complete.

WHAT COUNTS AS "ONE STEP": a single focused unit of work you could describe
in one sentence and verify with one specific command -- e.g. "add the
Contact dataclass and its field validators", not "build the contact book".
If your plan for a step touches several unrelated pieces (the data model AND
storage AND every CLI subcommand AND tests, say), that is not one step --
propose the smallest useful piece of it now, and propose the rest as
separate later steps once this one is verified. Most steps should touch
1-3 files. Needing 20+ tool calls to finish a step is a sign it should have
been split before you started, not evidence it was one step all along.

Rules:
- write_file and edit_file are NOT available here on purpose. Any actual
  file change must go through propose_step: it hands off to a step where
  mark_step_done re-verifies your work with a real acceptance command.
  A change made without that verification could silently regress something
  that already passed and was committed, with nothing catching it.
- propose_step ONE step at a time. Do not plan ahead; decide after seeing
  real evidence from the previous step.
- Every acceptance_command must be a real shell command that exits 0 on
  success (compile check, test run, file existence check, etc), and it
  must actually verify what the step claims to do. If the only check you
  can write is shallow (e.g. a file merely exists) while the work you're
  planning is much bigger than that, the step itself is too big -- shrink
  the step until a single specific command can meaningfully verify it.
- Use the scratchpad (update_scratchpad or the scratchpad field of
  mark_step_done) to keep working notes across steps.
- Only call declare_done when ALL work is complete and verifiable.
- Do NOT propose_step just to "declare completion" or "finalize" -- call
  declare_done directly instead. It's independently verified already; a
  step wrapping it adds a redundant, self-invented check that can go stale
  (e.g. checking for a file from an earlier, since-corrected plan) and fail
  for reasons unrelated to whether the actual goal was met.
"""

    def _inner_system_prompt(self, state: RunState, step_context: StepContext) -> str:
        return f"""You are an autonomous coding agent working in a real git repository.

OS: {shell_description()}

GOAL: {state.goal}

COMPLETED SO FAR:
{state.completed_summary_text()}

LESSONS FROM PAST RUNS IN THIS WORKSPACE (verified -- each was an actual problem
followed by a confirmed successful outcome, not a guess):
{self.lessons.summary_text()}

ACTIVE DECISIONS FOR THIS PROJECT (durable choices already made -- do not
contradict these; if you make a NEW architecture/technology/convention choice
here, call record_decision right away for that one choice -- one decision
per call, in the moment you decide it, not bundled with others or saved up
for later):
{self.toolbox.decisions.summary_text()}

IMPORTANT CONTEXT BOUNDARY:
The planner/exploration conversation is intentionally not included in this context.
You have been given a compact handoff below. Trust its findings as useful clues, but
inspect the actual current filesystem state before editing. Do NOT page through an
entire large file unless the task genuinely requires it. Prefer search_code and
small, targeted read_file regions around the relevant symbols or error locations.

{step_context.prompt_text()}

Use workspace tools to implement this step. When ready, call mark_step_done.
The harness verifies with the real acceptance command above -- your word alone is not enough.
Read files before editing. edit_file requires an exact unique match for old_str.
If you notice you're going in circles -- repeated edits to the same file with no
real progress, or you've realized your current approach fundamentally can't work --
call discard_and_restart instead of patching another guess on top of a tangled
state. It reverts to the last passing commit and gives you a clean slate, at no
cost to your attempt budget.
"""

    # ── summary ────────────────────────────────────────────────────────────

    def _print_summary(self, state: RunState) -> None:
        print("\n" + "=" * 60)
        status_label = {"done": "COMPLETE", "aborted": "ABORTED", "running": "INCOMPLETE"}.get(state.status, state.status)
        print(f"RUN {status_label} -- {self.budget.summary()}")
        for s in state.completed_steps:
            print(f"  [{s.index}] {s.title}")
        log = self.workspace.git_log(n=len(state.completed_steps) + 2)
        print("\nCommits made:")
        print(log.stdout)
        print("=" * 60)


# ── helpers ────────────────────────────────────────────────────────────────

_PYTHON_DASH_C_RE = re.compile(
    r'python3?(?:\.exe)?\s+-c\s+(["\'])(.*)\1\s*$',
    re.IGNORECASE | re.DOTALL,
)

# Common stdlib module names, checked when used as `modname.attr` (e.g.
# `sys.exit`, `os.path`) with no corresponding import anywhere in the code.
# Deliberately a fixed, narrow list -- not "every possible name" -- so this
# can't false-positive on an ordinary local variable used with a dotted
# method call (e.g. `results.append(...)`), which would trigger constantly
# if any unimported name were flagged.
_COMMON_STDLIB_MODULES = {
    "sys", "os", "json", "re", "math", "subprocess", "pathlib", "shutil",
    "glob", "ast", "io", "time", "datetime", "itertools", "functools",
    "collections", "random", "string", "csv", "argparse", "logging",
    "traceback", "typing", "unittest", "socket", "threading",
}


def _find_missing_stdlib_imports(code: str) -> list[str]:
    """
    Best-effort: finds common stdlib module names used as `modname.attr`
    that are never imported AND never locally bound (assigned, a function
    parameter, or a for-loop variable) anywhere in the code. A name that's
    locally bound is treated as intentional shadowing, not a missing import,
    so this stays narrow rather than flagging legitimate code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # syntax errors are reported separately; nothing more to add here

    imported: set[str] = set()
    locally_bound: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        locally_bound.add(sub.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.target is not None:
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    locally_bound.add(sub.id)
        elif isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                locally_bound.add(arg.arg)
        elif isinstance(node, ast.For):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    locally_bound.add(sub.id)

    used_as_module_base: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            used_as_module_base.add(node.value.id)

    return sorted(
        name for name in used_as_module_base
        if name in _COMMON_STDLIB_MODULES and name not in imported and name not in locally_bound
    )


def _validate_python_dash_c_syntax(acceptance_cmd: str) -> str | None:
    """
    Best-effort: if acceptance_cmd is (or ends with) a `python -c "<code>"`
    invocation, checks the inline code for two cheap-to-catch, guaranteed-
    to-fail-forever problems before the step is ever accepted:

    1. An outright SyntaxError -- e.g. a stray '-' in an import statement
       (`import texttransform.-uppercase`) made an acceptance command fail
       every single time regardless of what the actual plugin files did.

    2. A common stdlib module used but never imported -- e.g.
       `sys.exit(...)` with only `import os` present. This is NOT a syntax
       error (ast.parse() alone can't catch it -- it's a NameError only
       visible at actual run time), but it's the exact same failure shape:
       a check that can never pass, seen twice in practice with this
       precise sys.exit/import-sys pattern, burning a full attempt cycle
       discovering that the CHECK itself, not the implementation, was broken.

    Deliberately narrow: only validates this one common, cheap-to-check
    shape. Doesn't attempt to validate arbitrary shell syntax, multi-command
    chains, or non-Python commands -- returns None (no problem detected) for
    anything it doesn't recognize, rather than false-rejecting valid work.
    """
    match = _PYTHON_DASH_C_RE.search(acceptance_cmd.strip())
    if not match:
        return None
    code = match.group(2)
    # Best-effort unescape of shell-escaped quotes inside the -c argument.
    # Not a full shell parser -- just enough to catch the common case.
    code = code.replace('\\"', '"').replace("\\'", "'")
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"the inline Python code has a syntax error and would fail every time: {e}"

    missing = _find_missing_stdlib_imports(code)
    if missing:
        names = ", ".join(missing)
        example = f"import {missing[0]}"
        return (
            f"the inline Python code uses {names} but never imports it -- this will raise "
            f"NameError and fail every time, regardless of what you implement. Add '{example}'"
            f"{' (and the others)' if len(missing) > 1 else ''} at the start of the -c code."
        )
    return None


def _inner_loop_failure_reason(title: str, check_failures: int, derailed_attempts: int) -> str:
    """An honest account of why a step failed -- a real failed acceptance
    check and the model derailing/running out of steps without ever
    attempting one are different situations and shouldn't be reported
    identically to the human."""
    if check_failures and derailed_attempts:
        return (f"step '{title}': {check_failures} failed acceptance check(s) and "
                f"{derailed_attempts} attempt(s) that ran out of steps without ever "
                f"completing a check")
    if derailed_attempts and not check_failures:
        return (f"step '{title}': ran out of its step budget {derailed_attempts} time(s) "
                f"without ever completing a real acceptance check (the model may have "
                f"gotten stuck generating text instead of tool calls)")
    return f"step '{title}' failed its acceptance check {check_failures} time(s)"


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode("utf-8", errors="replace").decode("utf-8")
        sys.stdout.buffer.write((safe + "\n").encode("utf-8", errors="replace"))


def _short(d: dict) -> str:
    parts = []
    for k, v in d.items():
        s = str(v)
        parts.append(f"{k}={s[:60]!r}{'...' if len(s) > 60 else ''}")
    return ", ".join(parts)
