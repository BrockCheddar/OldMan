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
import json
import os
import re
import sys
import time
from pathlib import Path

from .approval import ask_human_question
from .budget import BudgetTracker, BudgetExceeded
from .config import Config
from .lessons import LessonsStore, _keywords
from .llm import build_llm_client, LLMError, Turn, ToolResult, trim_history_to_fit
from .osinfo import shell_description
from .planner import RunState, CompletedStep, StepContext, MAX_SUBTASKS
from .repetition import RepetitionGuard
from .session import SessionStore
from .state_report import compact_history_with_state_report
from .tools import ToolBox, ToolError, TOOL_SCHEMAS, validate_command_for_os, looks_like_mutating_command
from .workspace import Workspace


class AgentAborted(RuntimeError):
    pass


# ── outer-loop exploration carry-over (bounded, so it can't itself blow the budget) ──

_EXPLORATION_LOG_MAX_ENTRIES = 8
_EXPLORATION_ENTRY_CHAR_CAP = 1500
_EXPLORATION_TOTAL_CHAR_CAP = 9000

# Total chars the "files in scope" block of a state report may spend on
# live file content, and the floor given to any single file even when
# there are many files in scope. Replaces a hard `files[:5]` cap that
# dropped files 6+ silently.
_FILES_IN_SCOPE_CHAR_BUDGET = 15000
_FILES_IN_SCOPE_MIN_PER_FILE = 800

# condensed_files storage: cap per-file, not on the whole structure --
# with a dict, total size naturally scales with file count instead of one
# shared budget that any single pass could blow through or empty out.
_CONDENSED_FILE_ENTRY_CHAR_CAP = 1500
# condensed_files DISPLAY: separate cap for how much of the dict gets
# rendered into any one prompt (condensation call, state report, or
# per-turn context). Storage can hold the whole codebase's worth of
# entries; a single render must still fit in context regardless of repo
# size -- this is the same risk flagged earlier for the old single-string
# field, now guarded explicitly rather than left implicit.
_CONDENSED_FILES_RENDER_CHAR_CAP = 12000


def _parse_condensed_sections(text: str) -> dict[str, str]:
    """Parses the condensation call's `### <path>` per-file format back into
    a dict. Tolerant of minor formatting drift (extra blank lines, trailing
    whitespace) since this is a small local model's output, not a strict
    machine format -- but the delimiter itself (### at line start) is
    exact, so a malformed response yields an empty or partial dict rather
    than a mis-split mess."""
    sections: dict[str, str] = {}
    current_path: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("### "):
            if current_path is not None:
                sections[current_path] = "\n".join(current_lines).strip()
            current_path = line[4:].strip()
            current_lines = []
        elif current_path is not None:
            current_lines.append(line)
    if current_path is not None:
        sections[current_path] = "\n".join(current_lines).strip()
    return {path: note for path, note in sections.items() if note}


def _render_exploration_log(exploration_log: list[tuple[str, str]] | None) -> str:
    """Most-recent-first, each entry capped, total output capped. Returns ''
    if there's nothing to show."""
    if not exploration_log:
        return ""
    recent = exploration_log[-_EXPLORATION_LOG_MAX_ENTRIES:]
    blocks = []
    total = 0
    for label, content in reversed(recent):
        snippet = content[:_EXPLORATION_ENTRY_CHAR_CAP]
        block = f"--- {label} ---\n{snippet}"
        if total + len(block) > _EXPLORATION_TOTAL_CHAR_CAP:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _render_condensed_files(condensed_files: dict[str, str] | None,
                             char_cap: int = _CONDENSED_FILES_RENDER_CHAR_CAP) -> str:
    """Renders the per-file condensed-understanding dict into one block for
    display/prompting. Bounded regardless of how many files the dict holds
    -- a 300-file repo's full dict must never be dumped unbounded into a
    single prompt (that's exactly the context-overflow risk flagged
    earlier in this project, now guarded explicitly). Stops adding entries
    once the cap is hit and says so, rather than silently truncating mid-
    entry or dropping entries with no signal."""
    if not condensed_files:
        return ""
    blocks = []
    total = 0
    omitted = 0
    for path, note in condensed_files.items():
        block = f"### {path}\n{note}"
        if total + len(block) > char_cap:
            omitted += 1
            continue
        blocks.append(block)
        total += len(block)
    rendered = "\n\n".join(blocks)
    if omitted:
        rendered += f"\n\n[{omitted} more file(s) condensed but omitted here -- over the display budget]"
    return rendered


def _merge_exploration_into_findings(findings: str, exploration_log: list[tuple[str, str]] | None,
                                      condensed_files: dict[str, str] | None = None) -> str:
    """
    propose_step's `findings` field is whatever the model chose to type in
    -- if it forgot to copy something over, the coder's fresh context never
    sees it. This appends two safety nets underneath the model's own
    (usually better curated) summary: the harness-condensed per-file
    understanding first (denser, covers everything read so far, not just
    the current span), then raw exploration output as a last resort for
    whatever hasn't been condensed yet.
    """
    parts = [findings] if findings else []
    condensed_block = _render_condensed_files(condensed_files)
    if condensed_block:
        parts.append("[Auto-attached: harness-condensed per-file understanding built while reading]\n" + condensed_block)
    exploration_block = _render_exploration_log(exploration_log)
    if exploration_block:
        parts.append("[Auto-attached: raw exploration output not yet condensed]\n" + exploration_block)
    return "\n\n".join(parts)


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

REVISE_ACCEPTANCE_COMMAND_TOOL = {
    "name": "revise_acceptance_command",
    "description": (
        "Replace acceptance_command for THIS step with a corrected one, when you "
        "have concrete evidence the CHECK ITSELF is wrong -- not that your "
        "implementation is failing it. Evidence means something specific: the "
        "check references something that can never exist (a wrong assumption "
        "about a library's structure), depends on a program not available on "
        "this platform, or checks a path/location that can't be right. 'It keeps "
        "failing' is NOT evidence the check is wrong -- that's normal debugging; "
        "keep working and call mark_step_done again. This does NOT let you "
        "verify something new the step didn't originally cover -- that belongs "
        "in a future step, not a rewritten check. Usable ONCE per step. You will "
        "get back a fresh baseline result for the new command, same as the one "
        "you saw when this step started."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "new_acceptance_command": {
                "type": "string",
                "description": "The corrected command. Must still verify the same thing the step was meant to verify.",
            },
            "reason": {
                "type": "string",
                "description": "The concrete evidence the old command was wrong (not just failing).",
            },
        },
        "required": ["new_acceptance_command", "reason"],
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

SET_GOAL_DECOMPOSITION_TOOL = {
    "name": "set_goal_decomposition",
    "description": (
        "Replace the durable breakdown of the goal into distinct parts, if the "
        "goal has more than one genuinely separate piece worth tracking (e.g. "
        "'add feature X, then update the docs, then add tests' -- three parts). "
        "Skip this entirely for a goal that's already one focused piece of work. "
        "This is a FULL REPLACE, not an append -- pass every part you still want "
        "tracked, including ones already done (use check_off_subtask for those "
        "instead of dropping them here). If a title here matches an existing "
        "entry's title EXACTLY, it keeps its done/pending status; a reworded "
        "title is treated as a new item starting at pending, and any old item "
        "you don't include at all is dropped from tracking entirely -- so if "
        "you're only adding one new part, copy the existing titles verbatim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": f"Short titles, one per distinct part of the goal. Max {MAX_SUBTASKS}.",
            },
        },
        "required": ["subtasks"],
    },
}

STRENGTHEN_STEP_CHECK_TOOL = {
    "name": "strengthen_step_check",
    "description": (
        "Add an ADDITIONAL acceptance command to an already-completed step, "
        "when you've discovered that step's original check only proved a "
        "narrower property than what later work now depends on (e.g. a "
        "function's original check only covered typical input, and a later "
        "step now relies on it handling an edge case too). Does not replace "
        "the original command -- both are re-run by the regression check "
        "from now on. Use this to make a real gap in verification durable, "
        "not to duplicate a check that already covers the case."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "step_index": {"type": "integer", "description": "The [index] of the completed step, as shown in COMPLETED STEPS SO FAR."},
            "additional_command": {
                "type": "string",
                "description": "Real shell command, must exit 0 on success -- same rules as any acceptance_command.",
            },
            "reason": {"type": "string", "description": "What gap this closes, concretely."},
        },
        "required": ["step_index", "additional_command", "reason"],
    },
}

CHECK_OFF_SUBTASK_TOOL = {
    "name": "check_off_subtask",
    "description": (
        "Mark one entry from the goal decomposition as done, by its id (shown "
        "in parentheses next to each entry). Does not affect anything else in "
        "the breakdown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The subtask's id, e.g. 'a1b2c3d4'."},
        },
        "required": ["id"],
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
    "read_file", "code_skeleton", "list_dir", "search_code", "run_command", "git_diff", "git_log",
    "ask_human", "record_decision",
}
OUTER_WORKSPACE_TOOLS = [t for t in TOOL_SCHEMAS if t["name"] in _OUTER_SAFE_TOOL_NAMES]
OUTER_TOOLS = OUTER_WORKSPACE_TOOLS + [
    PROPOSE_STEP_TOOL, DECLARE_DONE_TOOL, UPDATE_SCRATCHPAD_TOOL,
    SET_GOAL_DECOMPOSITION_TOOL, CHECK_OFF_SUBTASK_TOOL, STRENGTHEN_STEP_CHECK_TOOL,
]

# Tools available inside an active step (executing it) -- full access,
# because mark_step_done re-verifies with the real acceptance command.
INNER_TOOLS = TOOL_SCHEMAS + [MARK_STEP_DONE_TOOL, DISCARD_AND_RESTART_TOOL, REVISE_ACCEPTANCE_COMMAND_TOOL]


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
        # Only actually builds a second client if planner_llm is configured
        # separately; the normal case (one local model) resolves to the
        # same backend/config as self.llm, called sequentially like every
        # other request -- see LLMBackendConfig docstring on why nothing
        # here is allowed to run concurrently against a single-slot server.
        self.planner_llm = (
            self.llm if config.planner_llm is None
            else build_llm_client(config.planner_llm)
        )
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

        # FLAW 7: wire the durable-pending-question hooks to THIS run's
        # state now that it exists (ToolBox itself is constructed once in
        # __init__, before any RunState does).
        self.toolbox._on_question_asked = lambda q: self._persist_pending_question(state, q)
        self.toolbox._on_question_answered = lambda: self._clear_pending_question(state)

        # A question left in state.pending_question means a previous run
        # was killed (crash, kill -9 -- NOT the Ctrl-C path below, which
        # already saves cleanly) while blocked waiting on the human, after
        # the question was persisted but before any answer came back. The
        # in-memory chat history that asked it is gone either way, so
        # silently discarding the question would silently discard whatever
        # made it worth asking. Re-ask it now, fold the answer into the
        # scratchpad so the model sees it on its very next turn, and clear
        # the flag -- conservatively re-asking is the safe direction here,
        # not assuming an answer that may never have actually arrived.
        if state.pending_question:
            print("\n[resume] a question was left pending from an earlier, interrupted run:")
            answer = ask_human_question(state.pending_question)
            self._set_scratchpad(
                state,
                state.scratchpad
                + f"\n[Resumed question]: {state.pending_question}\n[Answer]: {answer}"
            )
            state.pending_question = None
            self.session.log_event("pending_question_resolved_on_resume", {"answer": answer})
            self.session.save_state(state)

        try:
            self._outer_loop(state, final_acceptance_command)
        except KeyboardInterrupt:
            self.session.save_state(state)
            print("\n[interrupted] session saved -- resume with `autocoder resume --workspace ...`")
            return state

        self._print_summary(state)
        return state

    def _persist_pending_question(self, state: RunState, question: str) -> None:
        state.pending_question = question
        self.session.log_event("question_pending", {"question": question})
        self.session.save_state(state)

    def _clear_pending_question(self, state: RunState) -> None:
        state.pending_question = None
        self.session.save_state(state)

    def reopen(self, additional_instructions: str) -> RunState:
        """
        FLAW 1: declare_done (and the run-ending escalation path) used to
        be a true dead end -- state.status flips to "done"/"aborted" and
        _outer_loop's `while state.status == "running"` guard means a
        plain `resume` on that session is a silent no-op forever after.
        There was nothing durable to re-scope back onto, either -- goal
        was a bare string.

        This is the human-facing correction path: "actually, X still needs
        fixing" without starting a fresh workspace/session and losing the
        completed-steps history, lessons, and decisions already recorded.
        It is NOT a model-callable tool -- by the time a run has reached
        "done"/"aborted" the process has already exited; there is no live
        loop for the model to call anything from. Call this, THEN
        agent.run(resume=True) to actually continue.
        """
        state = self.session.load_state()
        if state is None:
            raise ValueError("no existing session to reopen -- use 'start' for a new one")
        if state.status == "running":
            raise ValueError(
                "session is still running -- reopen is only for a session "
                "that already ended (done/aborted); resume it normally instead"
            )
        state.goal = f"{state.goal}\n\n[REOPENED] {additional_instructions}"
        state.decomposition.add(additional_instructions)
        state.status = "running"
        self.session.log_event("goal_reopened", {"additional_instructions": additional_instructions})
        self.session.save_state(state)
        return state

    def export_session(self, output_path: Path) -> dict:
        """
        FLAW 10 (part 1): all state was previously workspace-local with no
        way out -- no export, no continuing a run on a different machine.
        Bundles the session state plus this workspace's lessons/decisions
        (the durable knowledge, not just the in-progress run) into one
        portable JSON file, tagged with the git commit it was verified
        against so an eventual import can warn if the code has moved on.
        """
        state = self.session.load_state()
        if state is None:
            raise ValueError("no session in this workspace to export")
        bundle = {
            "schema_version": 1,
            "exported_at": time.time(),
            "git_head": self.workspace.git_head(),
            "session": state.to_dict(),
            "lessons": self._read_json_if_exists(self.config.lessons_file),
            "decisions": self._read_json_if_exists(self.config.decisions_file),
        }
        output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        return bundle

    def import_session(self, bundle_path: Path, force: bool = False) -> RunState:
        """
        FLAW 10 (part 2): the other half of export_session -- restores a
        bundle into THIS workspace so a run can continue on a different
        machine/clone. Refuses to clobber an existing session unless
        force=True (this is a destructive overwrite of local session
        state, not a merge). Warns rather than blocks on a git_head
        mismatch: the code may have moved on since export, which means
        completed_steps' acceptance commands aren't guaranteed to still
        apply -- but the human asked for this import, so inform, don't
        override that decision.
        """
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("schema_version") != 1:
            raise ValueError(f"unrecognized export schema_version: {bundle.get('schema_version')!r}")
        if self.session.load_state() is not None and not force:
            raise ValueError(
                "a session already exists in this workspace -- pass force=True to overwrite it"
            )

        current_head = self.workspace.git_head()
        bundle_head = bundle.get("git_head")
        if bundle_head and current_head and bundle_head != current_head:
            print(
                f"[import] WARNING: bundle was exported at commit {bundle_head[:12]}, "
                f"this workspace is at {current_head[:12]}. Completed steps' acceptance "
                "commands may no longer apply -- consider a regression check once resumed."
            )

        state = RunState.from_dict(bundle["session"])
        self.session.save_state(state)
        if bundle.get("lessons") is not None:
            self.config.lessons_file.write_text(json.dumps(bundle["lessons"], indent=2), encoding="utf-8")
        if bundle.get("decisions") is not None:
            self.config.decisions_file.write_text(json.dumps(bundle["decisions"], indent=2), encoding="utf-8")
        self.session.log_event("session_imported", {"bundle_git_head": bundle_head, "current_git_head": current_head})
        return state

    def import_lessons(self, other_lessons_path: Path) -> int:
        """
        FLAW 10 (part 3) / cross-workspace knowledge sharing: lessons are
        genuinely useful beyond the run (or even the project) that
        produced them -- a lesson about this model's quirks, or a recurring
        environment gotcha, is worth carrying into a brand new workspace.
        This is a MERGE, not an overwrite: only active lessons not already
        present (by symptom+fix, ignoring which project they came from)
        are added, each through the normal add() path so eviction/cap
        rules still apply. Returns how many were actually added.
        """
        other_raw = self._read_json_if_exists(other_lessons_path)
        if not other_raw:
            return 0
        existing = {(l.symptom, l.fix) for l in self.lessons.load()}
        added = 0
        for entry in other_raw.get("lessons", []):
            if not entry.get("active", True):
                continue
            key = (entry.get("symptom", ""), entry.get("fix", ""))
            if key in existing:
                continue
            self.lessons.add(
                context=entry.get("context", "(imported from another workspace)"),
                symptom=entry.get("symptom", ""),
                fix=entry.get("fix", ""),
                intent=entry.get("intent", ""),
            )
            existing.add(key)
            added += 1
        if added:
            self.session.log_event("lessons_imported", {"count": added, "source": str(other_lessons_path)})
        return added

    @staticmethod
    def _read_json_if_exists(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

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

    MAX_AUTO_READ_LOG_CHARS = 4000

    def _append_auto_read_note(self, state: "RunState", tool_name: str, target: str, output: str) -> None:
        """Harness-side, not model-side: fires on every real read_file/
        search_code result in the outer loop regardless of whether the
        model ever calls update_scratchpad. Purely observational -- one
        short line per read, never fed back into the model's own prompt,
        so it costs zero extra prompt tokens. Saved to disk immediately so
        `session.json` reflects reading progress in near-real time instead
        of only updating at step/compaction boundaries."""
        first_line = output.splitlines()[0] if output.strip() else "(empty)"
        note = f"[{tool_name}] {target} -> {first_line}"[:300]
        combined = f"{state.auto_read_log}\n{note}" if state.auto_read_log else note
        if len(combined) > self.MAX_AUTO_READ_LOG_CHARS:
            combined = "...[earlier reads trimmed]...\n" + combined[-self.MAX_AUTO_READ_LOG_CHARS:]
        state.auto_read_log = combined
        self.session.save_state(state)

    def _condense_batch(self, state: "RunState", touched_files: set[str],
                         exploration_log: list[tuple[str, str]]) -> None:
        """
        Harness-triggered, not model-triggered: fires automatically once
        `condense_batch_size` distinct files have been read, regardless of
        whether the model ever calls update_scratchpad. This is the fix
        for "read many files, understanding evaporates."

        Per-file dict, not a single growing string -- confirmed bug in the
        string version: `state.condensed_notes = text` let ANY single pass
        silently overwrite the entire field, discarding every prior file's
        coverage the moment a later pass's response didn't happen to
        re-mention it (observed directly: agent.py was condensed 7 times
        across a real run, then vanished from the final saved notes because
        the last pass's batch didn't include it). Here, a pass can only
        write keys for the files actually in `touched_files` -- structurally
        incapable of touching any other file's entry, no matter what the
        model returns.

        The full existing dict is still shown as READ-ONLY context (so the
        model can correctly describe cross-file relationships -- e.g. file A
        calling into a class defined in file B -- even though this pass only
        writes A's entry), bounded by _render_condensed_files so a large
        repo's full dict can't itself overflow this call's own prompt.
        """
        batch_content = _render_exploration_log(
            [(label, content) for label, content in exploration_log
             if any(path in label for path in touched_files)]
        )
        if not batch_content:
            return
        file_list = ", ".join(sorted(touched_files))
        system = (
            "You are building per-file notes on a codebase being explored file by "
            "file. You will be given (a) existing notes on OTHER files already "
            "covered, as read-only reference context, and (b) raw content just "
            "read from a NEW batch of files. Describe ONLY the new batch's files -- "
            "what each one's purpose is, key classes/functions, and how it connects "
            "to anything in the reference context. Be concrete and specific, not "
            "generic. Output ONE section per new-batch file, in exactly this "
            "format, nothing else:\n"
            "### <exact file path>\n<description>\n\n"
            f"The new batch contains exactly these files -- output a section for "
            f"each one and nothing else: {file_list}"
        )
        existing_context = _render_condensed_files(state.condensed_files)
        user_text = (
            f"Existing notes on other files (read-only reference):\n"
            f"{existing_context or '(nothing condensed yet -- this is the first batch)'}\n\n"
            f"New batch to describe:\n{batch_content}\n\n"
            "Write the per-file sections now."
        )
        try:
            # Call the backend directly, not self._call_llm -- that wrapper
            # escalates backend failures to an interactive human prompt,
            # which is correct for the main loop but wrong here: this is a
            # best-effort auxiliary pass, and a timeout on it should never
            # stop the run or demand a human answer. Worst case, this batch
            # doesn't get condensed and exploration_log/auto_read_log still
            # hold the raw record.
            response = self.llm.complete(
                system=system, history=[Turn(role="user", text=user_text)], tools=[],
                max_tokens=self.config.budget.max_output_tokens,
            )
            self.budget.record_usage(response.input_tokens, response.output_tokens)
        except LLMError as e:
            self.session.log_event("condensation_pass_failed", {"error": str(e)})
            return
        if not response.text:
            return
        parsed = _parse_condensed_sections(response.text)
        # Write-scoping enforced here, not just requested in the prompt:
        # only apply entries whose path was actually in this batch. A
        # model ignoring the instruction and describing (or re-describing)
        # some other file cannot touch that file's real entry through this
        # path -- the dict key space outside `touched_files` is simply
        # never written to, regardless of what comes back.
        applied = 0
        for path, note in parsed.items():
            if path not in touched_files:
                continue
            if len(note) > _CONDENSED_FILE_ENTRY_CHAR_CAP:
                note = note[:_CONDENSED_FILE_ENTRY_CHAR_CAP]
            state.condensed_files[path] = note
            applied += 1
        if applied == 0:
            # Model didn't follow the format closely enough to parse --
            # log it, but don't guess; existing entries (if any) for these
            # files are left exactly as they were rather than risking a
            # bad partial write.
            self.session.log_event("condensation_parse_failed", {"files": sorted(touched_files)})
            return
        self.session.log_event("condensation_pass", {"files": sorted(touched_files), "applied": applied})
        self.session.save_state(state)

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

    def _build_outer_state_report(self, state: RunState,
                                   exploration_log: list[tuple[str, str]] | None = None,
                                   final_acceptance_command: str | None = None) -> str:
        """
        Same idea as _build_state_report, applied at the run level instead
        of the step level. Built entirely from live checks against the
        real workspace and the RunState's own current fields -- never from
        old chat text -- so it can never be stale, unlike a truncated
        fragment of old tool output.

        exploration_log carries what read_file/search_code actually
        returned this span, independent of the chat history being
        compacted away. Without it, compaction used to wipe every file the
        model had read down to a bare filename in the directory listing --
        the model would "forget" file contents it had just seen the moment
        history grew too large. Bounded per-entry and in total so this
        can't itself become the next context blowout.
        """
        lines = [
            "STATE REPORT (outer loop)",
            f"Goal: {state.goal}",
            "",
        ]
        if final_acceptance_command:
            lines.append(
                f"Goal-level completion check (run it yourself via run_command before "
                f"deciding whether to keep going): $ {final_acceptance_command}"
            )
            lines.append("")
        lines.extend([
            f"Completed steps so far:\n{state.completed_summary_text()}",
            "",
            f"Scratchpad:\n{state.scratchpad or '(empty)'}",
            "",
        ])
        if state.condensed_files:
            lines.append(f"Harness-condensed per-file understanding built while reading so far "
                         f"(check this before reading more -- if it already covers what you "
                         f"need, propose_step or declare_done instead):\n{_render_condensed_files(state.condensed_files)}")
            lines.append("")
        lines.append("Current workspace files (live listing, not memory):")
        try:
            listing = self.toolbox.dispatch("list_dir", {"path": "."})
        except ToolError as e:
            listing = f"(could not list workspace: {e})"
        lines.append(listing[:1500])

        git_status = self.workspace.git_status()
        lines.append(f"\nGit status (uncommitted changes, live check):\n{git_status.stdout[:800] or '(clean)'}")

        exploration_block = _render_exploration_log(exploration_log)
        if exploration_block:
            lines.append(f"\nWhat you've already read this session (from real tool output, "
                          f"most recent first):\n{exploration_block}")

        return "\n".join(lines)

    def _flush_dirty_condensed(self, state: "RunState", dirty_files: set[str]) -> None:
        """Refreshes condensed_files entries for paths edited (write_file/
        edit_file) since the last flush, right after mark_step_done commits.

        Fixes a real gap in the read-triggered condensation above: that
        mechanism only ever re-condenses a file when the model re-reads it,
        but write_file/edit_file don't touch files_since_condense at all --
        so an edited file's entry can keep describing pre-edit content
        indefinitely, with nothing to correct it, since there's no
        model-facing tool to hand-edit one entry (deliberately -- that
        would reopen the single-point-of-overwrite bug the dict design
        exists to prevent). Runs post-commit rather than per-edit so a step
        with several edit_file calls in a row against the same file only
        pays for one condensation pass, and never against edits later
        reverted by a failed acceptance check.

        Reads each file fresh off disk rather than relying on the read/
        exploration log -- an edited-but-never-re-read file has no log
        entry to reuse, and a fresh read is also correct for a file that
        WAS re-read earlier in the step but edited again afterward.
        """
        synthetic_log: list[tuple[str, str]] = []
        touched: set[str] = set()
        for path in dirty_files:
            try:
                content = self.toolbox.dispatch("read_file", {"path": path})
            except ToolError:
                # Deleted (or moved) since being edited -- nothing to
                # condense; drop any existing entry rather than let it
                # describe a file that no longer exists.
                state.condensed_files.pop(path, None)
                continue
            synthetic_log.append((f"read_file({path})", content))
            touched.add(path)
        if touched:
            self._condense_batch(state, touched, synthetic_log)

    # Real per-repo file-reading workload isn't knowable in advance, so
    # thresholds tuned for a 25-file run either nudge/condense too
    # aggressively on a huge repo or barely fire at all -- this counts
    # actual files in the workspace once, at run start, so downstream
    # thresholds scale with the real target instead of a hand-picked
    # constant. Deliberately cheap (os.walk, not a model turn) and
    # deliberately excludes dirs no one is going to read.
    _SCALE_EXCLUDED_DIRS = {".git", ".autocoder", "node_modules", "__pycache__",
                            ".venv", "venv", ".pytest_cache", "dist", "build"}

    def _count_workspace_files(self) -> int:
        count = 0
        for root, dirs, files in os.walk(self.workspace.root):
            dirs[:] = [d for d in dirs if d not in self._SCALE_EXCLUDED_DIRS and not d.startswith(".")]
            count += len(files)
        return count

    # Distinct files per condensation batch = file_count // this, floored
    # at the minimum. ~1/6th of the repo per batch keeps a 25-file repo at
    # a batch of ~5 and a 300-file repo at a batch of 50 -- proportional
    # cadence rather than a flat count that's wrong at either extreme.
    _CONDENSE_DIVISOR = 6
    _CONDENSE_BATCH_MIN = 4

    def _condensation_batch_size(self) -> int:
        file_count = self._count_workspace_files()
        return max(self._CONDENSE_BATCH_MIN, file_count // self._CONDENSE_DIVISOR)

    def _outer_loop(self, state: RunState, final_acceptance_command: str | None) -> None:
        history: list[Turn] = []
        self.budget.new_outer_span()
        repetition_guard = RepetitionGuard()
        # Running record of what read_file/search_code actually returned
        # this outer-loop span, independent of chat history. Chat history
        # gets trimmed/compacted for token budget; this does not, so a
        # propose_step handoff or a state-report compaction can still draw
        # on everything read so far instead of only what's left in `history`.
        exploration_log: list[tuple[str, str]] = []
        # Distinct files (not raw read calls, so paginating one big file
        # via offset/limit never counts as "many files") touched since the
        # last condensation pass.
        files_since_condense: set[str] = set()
        condense_batch_size = self._condensation_batch_size()
        # Consecutive count of propose_step calls made while the goal-level
        # acceptance command was already passing. Reset whenever a proposal
        # happens while it's genuinely not passing yet (a legitimately
        # different situation). See the propose_step handling below for why
        # this exists and what happens at each count.
        redundant_step_attempts = 0
        # Files targeted by the most recently COMPLETED step, so the
        # redundancy gate below can tell "a new step re-targeting the same
        # deliverable while the goal check already passes" (the real,
        # demonstrated failure) apart from "a new, unrelated step happens to
        # be proposed after the goal's proxy check already passes" (which a
        # goal with a narrow/coincidentally-satisfied acceptance command
        # would trigger constantly and wrongly). Only ever compares against
        # the LAST step, not every step ever completed -- narrower, but
        # matches the actual evidence (two consecutive steps, same file)
        # without guessing how far back to look.
        last_completed_step_files: set[str] = set()
        # FLAW 8/14: counts completed steps since the last self-audit pass.
        # Reset to 0 whenever an audit actually runs; a value of 0 for
        # self_audit_every_n_steps disables the mechanism entirely.
        steps_since_audit = 0
        # Length of the current run of consecutive completed steps whose
        # titles all look like the same fix repeated (see
        # _steps_look_like_repeats). A fresh step that ISN'T similar to the
        # one before it restarts the streak at 1 (itself), not 0 -- it's
        # still one step. Escalates once the streak hits
        # _CONSECUTIVE_SIMILAR_STEP_THRESHOLD rather than letting the model
        # keep proposing another near-identical "fix".
        consecutive_similar_steps = 0

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

            system = self._outer_system_prompt(state, final_acceptance_command)
            # condensed_notes was previously only shown at the propose_step
            # handoff and inside the inner loop -- meaning the model could
            # be sitting on a harness-built, near-complete understanding of
            # the codebase while still in the outer loop with zero
            # visibility into it, having no evidence it had already covered
            # enough to act. Included here, every turn, same treatment as
            # scratchpad, so it can actually inform the next decision
            # instead of only appearing after the model has already
            # committed to propose_step.
            context_msg = (
                f"GOAL BREAKDOWN:\n{_decomposition_with_reconciliation_hints(state)}\n\n"
                f"COMPLETED STEPS SO FAR:\n{state.completed_summary_text()}\n\n"
                f"SCRATCHPAD:\n{state.scratchpad or '(empty)'}\n\n"
                f"HARNESS-CONDENSED UNDERSTANDING, per file (built automatically as you read "
                f"files -- check this before deciding to read more; if it already covers what "
                f"you need, propose_step or declare_done instead of re-reading):\n"
                f"{_render_condensed_files(state.condensed_files) or '(nothing condensed yet)'}\n\n"
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
                    report = self._build_outer_state_report(state, exploration_log, final_acceptance_command)
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
                        self.session.log_event("propose_step_rejected", {
                            "title": title, "reason": "missing acceptance_command",
                        })
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content="acceptance_command is required and must be non-empty.",
                        ))
                        acted = True
                        continue
                    validation_problem = _validate_acceptance_command(acceptance_cmd)
                    if validation_problem:
                        self.session.log_event("propose_step_rejected", {
                            "title": title, "reason": validation_problem,
                            "acceptance_command": acceptance_cmd,
                        })
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content=(
                                f"acceptance_command is invalid and would fail every time, "
                                f"regardless of what you implement: {validation_problem}\n"
                                f"Command was: {acceptance_cmd}\n"
                                "Call propose_step again with a corrected, valid acceptance_command."
                            ),
                        ))
                        acted = True
                        continue

                    # A new step proposed while the goal-level check already
                    # passes is a real, demonstrated failure mode -- not
                    # hypothetical: a run kept the goal-level check passing
                    # after step 1, then spent another ~60 minutes producing
                    # a byte-identical rewrite via step 2 anyway. An
                    # advisory-only response to that (like the repetition
                    # notices below, which the same run also ignored twice)
                    # isn't enough -- this gets one real rejection with a
                    # concrete ask, then auto-completes rather than trusting
                    # a third attempt behaves differently.
                    if final_acceptance_command and state.completed_steps and last_completed_step_files \
                            and (set(str(x) for x in files) & last_completed_step_files):
                        already_passing, _ = self._run_final_acceptance_command(final_acceptance_command)
                        if already_passing:
                            redundant_step_attempts += 1
                            if redundant_step_attempts >= 2:
                                accepted, feedback = self._verify_done(
                                    state,
                                    "goal-level acceptance command was already passing when a "
                                    "redundant step was proposed for the second time in a row -- "
                                    "auto-completed rather than allowing another unnecessary step",
                                    final_acceptance_command,
                                )
                                if accepted:
                                    state.status = "done"
                                    self._finalize_pending_lesson(
                                        fix_summary="goal accepted as done: auto-completed after a "
                                                    "redundant step proposal")
                                    self.session.save_state(state)
                                    return
                                # regression check (part of _verify_done) failed even though the
                                # acceptance command alone passed moments ago -- genuinely let it
                                # continue rather than force-closing on a real problem.
                                redundant_step_attempts = 0
                                self._set_scratchpad(
                                    state, state.scratchpad +
                                    f"\n[Feedback on auto-completion attempt]: {feedback}")
                                self.session.save_state(state)
                                tool_results.append(ToolResult(
                                    tool_call_id=call.id, is_error=True,
                                    content=f"Auto-completion check failed.\n{feedback}\nContinuing "
                                            "with this step proposal.",
                                ))
                                acted = True
                                continue
                            self.session.log_event("propose_step_rejected", {
                                "title": title, "reason": "goal-level acceptance already passing",
                            })
                            tool_results.append(ToolResult(
                                tool_call_id=call.id, is_error=True,
                                content=(
                                    "The goal-level acceptance command already passes right now -- "
                                    "the goal appears to already be satisfied by what's been done so "
                                    "far. If you believe more work is genuinely needed, your next "
                                    "propose_step must name the SPECIFIC problem you're fixing (not a "
                                    "general redo of the same deliverable). If the goal is actually "
                                    "done, call declare_done instead."
                                ),
                            ))
                            acted = True
                            continue
                        else:
                            redundant_step_attempts = 0
                    else:
                        redundant_step_attempts = 0

                    self.session.log_event("propose_step_accepted", {"title": title})
                    findings = _merge_exploration_into_findings(findings, exploration_log, state.condensed_files)
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
                    step_context.original_acceptance_command = step_context.acceptance_command
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
                        exploration_log = []
                        files_since_condense = set()
                        history_reset = True
                        acted = True
                        break
                    state.completed_steps.append(completed)
                    last_completed_step_files = set(step_context.files)
                    state.status = "running"
                    repetition_guard.checkpoint()
                    self._finalize_pending_lesson(fix_summary=f"step '{title}' completed: {completed.summary}")
                    self.session.save_state(state)
                    self.session.log_event("step_completed", {"index": completed.index, "title": completed.title})

                    if len(state.completed_steps) == 1:
                        consecutive_similar_steps = 1
                    elif _steps_look_like_repeats(state.completed_steps[-2], state.completed_steps[-1]):
                        consecutive_similar_steps += 1
                    else:
                        consecutive_similar_steps = 1
                    if consecutive_similar_steps >= _CONSECUTIVE_SIMILAR_STEP_THRESHOLD:
                        recent_titles = [s.title for s in state.completed_steps[-_CONSECUTIVE_SIMILAR_STEP_THRESHOLD:]]
                        self.session.log_event("consecutive_similar_steps_escalated", {"titles": recent_titles})
                        self._escalate(
                            state,
                            f"The last {_CONSECUTIVE_SIMILAR_STEP_THRESHOLD} completed steps all look like the "
                            f"same fix repeated under different titles: {recent_titles}. Each one passed its "
                            "own check when it ran, but something is likely making the same problem recur "
                            "(e.g. a test or verification step mutating a file that an earlier step's check "
                            "pins to an exact state) rather than each being a genuinely new problem. Continuing "
                            "to propose another near-identical fix is unlikely to break the cycle."
                        )
                        consecutive_similar_steps = 1

                    steps_since_audit += 1
                    if self.config.budget.self_audit_every_n_steps > 0 \
                            and steps_since_audit >= self.config.budget.self_audit_every_n_steps:
                        self._run_self_audit(state)
                        steps_since_audit = 0
                    # reset outer history after each completed step so context
                    # stays bounded and old tool output doesn't pile up
                    history = []
                    exploration_log = []
                    files_since_condense = set()
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

                if call.name == "set_goal_decomposition":
                    titles = call.input.get("subtasks", []) or []
                    if isinstance(titles, str):
                        titles = [titles]
                    titles = [str(t) for t in titles if str(t).strip()]
                    if not titles:
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content="subtasks must be a non-empty list of titles.",
                        ))
                        acted = True
                        continue
                    preserved, added_titles, dropped = state.decomposition.replace(titles)
                    self.session.save_state(state)
                    dropped_done = [s.title for s in dropped if s.status == "done"]
                    self.session.log_event("goal_decomposition_set", {
                        "count": len(state.decomposition.subtasks),
                        "preserved": len(preserved),
                        "added": len(added_titles),
                        "dropped": [s.title for s in dropped],
                        "dropped_done": dropped_done,
                    })
                    result_lines = ["Breakdown set:", state.decomposition.summary_text()]
                    if preserved:
                        result_lines.append(
                            f"({len(preserved)} item(s) matched an existing title exactly and kept "
                            "their done/pending status; reword a title if you want it treated as new.)"
                        )
                    if dropped_done:
                        result_lines.append(
                            "WARNING: the following were marked done before and are NOT in this new "
                            f"list, so they're no longer tracked at all: {dropped_done}"
                        )
                    tool_results.append(ToolResult(
                        tool_call_id=call.id,
                        content="\n".join(result_lines),
                    ))
                    acted = True
                    continue

                if call.name == "check_off_subtask":
                    subtask_id = str(call.input.get("id", "")).strip()
                    subtask = state.decomposition.get(subtask_id)
                    if subtask is None:
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content=(
                                f"No subtask with id '{subtask_id}'. Current breakdown:\n"
                                f"{state.decomposition.summary_text()}"
                            ),
                        ))
                        acted = True
                        continue
                    subtask.status = "done"
                    self.session.save_state(state)
                    self.session.log_event("subtask_checked_off", {"id": subtask_id, "title": subtask.title})
                    tool_results.append(ToolResult(
                        tool_call_id=call.id,
                        content=f"Marked done: {subtask.title}",
                    ))
                    acted = True
                    continue

                if call.name == "strengthen_step_check":
                    try:
                        step_index = int(call.input.get("step_index"))
                    except (TypeError, ValueError):
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content="step_index must be an integer matching a completed step's [index].",
                        ))
                        acted = True
                        continue
                    target_step = next((s for s in state.completed_steps if s.index == step_index), None)
                    if target_step is None:
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content=f"No completed step with index {step_index}.",
                        ))
                        acted = True
                        continue
                    additional_command = str(call.input.get("additional_command", "")).strip()
                    validation_problem = _validate_acceptance_command(additional_command)
                    if validation_problem:
                        tool_results.append(ToolResult(
                            tool_call_id=call.id, is_error=True,
                            content=f"additional_command is invalid: {validation_problem}",
                        ))
                        acted = True
                        continue
                    target_step.additional_checks.append(additional_command)
                    self.session.save_state(state)
                    self.session.log_event("step_check_strengthened", {
                        "step_index": step_index, "additional_command": additional_command,
                        "reason": call.input.get("reason", ""),
                    })
                    tool_results.append(ToolResult(
                        tool_call_id=call.id,
                        content=(
                            f"Added. Step {step_index} now also re-runs: {additional_command}\n"
                            "This will be checked on every future regression check, including declare_done."
                        ),
                    ))
                    acted = True
                    continue

                # workspace tool
                mutation_reason = None
                if call.name == "run_command":
                    mutation_reason = looks_like_mutating_command(call.input.get("command", "") or "")
                if call.name not in _OUTER_SAFE_TOOL_NAMES or mutation_reason:
                    # Not offered at this level -- reject explicitly rather
                    # than silently dispatching it if the model emits it
                    # anyway (local models don't always strictly respect the
                    # offered tool list). write_file/edit_file specifically
                    # must never mutate a file outside the verified
                    # propose_step -> inner-loop path (see _OUTER_SAFE_TOOL_NAMES
                    # comment for why -- an unverified outer-loop write once
                    # silently clobbered an already-checked file). run_command
                    # IS offered here (it's needed for read-only investigation:
                    # listing, compiling, running tests) but gets the same
                    # treatment when it looks like it would mutate something --
                    # otherwise it's a second, ungated path to the exact thing
                    # write_file is blocked from doing (confirmed in practice:
                    # a run stuck here used run_command to create and repeatedly
                    # overwrite a file rather than ever calling propose_step).
                    # What we CAN safely do is stop a small local model from
                    # thrashing on the same rejection: track it through the
                    # same repetition guard real tool calls use, and on repeat
                    # give it the exact propose_step call to make instead of
                    # just repeating the same "no" a fourth time.
                    if mutation_reason:
                        blocked_reason = (
                            f"run_command rejected: it {mutation_reason}. The outer loop is "
                            "read-only exploration only -- run_command here may inspect the "
                            "workspace (list, read, compile-check, run tests) but never "
                            "create, modify, delete, or install anything. Call propose_step "
                            "to make an actual change, including creating any deliverable "
                            "file (a summary, a report, anything with real content); it "
                            "hands off to a step where your work gets verified."
                        )
                    else:
                        blocked_reason = (
                            f"'{call.name}' is not available here -- this is the outer loop, "
                            "read-only exploration only. Call propose_step to make an actual "
                            "change; it hands off to a step where your work gets verified."
                        )
                    warning = repetition_guard.record(call.name, call.input)
                    if warning and call.name in ("write_file", "edit_file"):
                        target = call.input.get("path", "the file")
                        blocked_reason += (
                            f"\n\nYou've tried this more than once -- call propose_step now with "
                            f"files=[\"{target}\"] and an objective describing the change you were "
                            "about to make; you'll get write_file back inside that step."
                        )
                        if repetition_guard.should_force_escalate():
                            force_escalate_reason = (
                                f"stuck repeating blocked {call.name} on the same target from the "
                                "outer loop instead of calling propose_step"
                            )
                    elif warning and mutation_reason:
                        blocked_reason += (
                            "\n\nYou've tried this more than once -- call propose_step now "
                            "describing what you're trying to create or change; you'll get "
                            "an unrestricted run_command (and write_file) back inside that step."
                        )
                        if repetition_guard.should_force_escalate():
                            force_escalate_reason = (
                                "stuck repeating a blocked mutating run_command from the outer "
                                "loop instead of calling propose_step"
                            )
                    tool_results.append(ToolResult(
                        tool_call_id=call.id, is_error=True, content=blocked_reason,
                    ))
                    acted = True
                    if force_escalate_reason:
                        break
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
                    if call.name in ("read_file", "code_skeleton", "search_code"):
                        target = call.input.get("path") or call.input.get("query") or ""
                        exploration_log.append((f"{call.name}({target})", output))
                        self._append_auto_read_note(state, call.name, target, output)
                        if call.name in ("read_file", "code_skeleton") and target:
                            files_since_condense.add(target)
                            if len(files_since_condense) >= condense_batch_size:
                                self._condense_batch(state, files_since_condense, exploration_log)
                                files_since_condense = set()
                except ToolError as e:
                    print(f"    -> ERROR: {e}")
                    error_msg = str(e)
                    # validate_command_for_os's rejection for multi-line/heredoc
                    # commands tells the model to use write_file to stage the
                    # content instead -- correct advice inside a step, but
                    # write_file isn't offered here (see the _OUTER_SAFE_TOOL_NAMES
                    # rejection above). Left uncorrected, this is a dead end:
                    # the model is pointed at a tool it doesn't have, with no
                    # indication that propose_step is the way to get it.
                    if call.name == "run_command" and "write_file" in error_msg:
                        error_msg += (
                            "\n\nNote: write_file isn't available in this outer loop. "
                            "Call propose_step first (with a valid acceptance_command) -- "
                            "you'll get write_file back inside that step to create the "
                            "content properly, in one call, instead of building it "
                            "through single-line shell commands."
                        )
                    tool_results.append(ToolResult(tool_call_id=call.id, content=f"ERROR: {error_msg}", is_error=True))
                self.session.log_event("tool_call", {"name": call.name})
                acted = True
                if force_escalate_reason:
                    break

            if tool_results and not history_reset:
                history.append(Turn(role="tool_results", tool_results=tool_results))
                if self._prompt_over_budget(system, history, OUTER_TOOLS):
                    report = self._build_outer_state_report(state, exploration_log, final_acceptance_command)
                    history = compact_history_with_state_report(history, report)
                    self.session.log_event("state_report_compaction", {"scope": "outer"})

            if force_escalate_reason:
                self._escalate(state, reason=force_escalate_reason)
                if state.status != "running":
                    return
                history = []
                exploration_log = []
                repetition_guard.checkpoint()
                continue

    def _prompt_over_budget(self, system: str, history: list[Turn], tools: list[dict]) -> bool:
        budget_tokens = max(self.config.llm.context_window_tokens - self.config.budget.max_output_tokens, 1)
        return self.llm.count_tokens(system, history, tools) > budget_tokens

    def _build_state_report(self, step_context: StepContext, attempt: int, max_attempts: int,
                             last_check_summary: str = "", condensed_files: dict[str, str] | None = None) -> str:
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
        condensed_block = _render_condensed_files(condensed_files)
        if condensed_block:
            lines.append(f"\nHarness-condensed per-file understanding built while reading so far:\n{condensed_block}")

        if step_context.files:
            # Budget by total characters, not a fixed file count -- a fixed
            # `files[:5]` slice silently dropped files 6+ from a larger
            # scope with no signal to the model that anything was cut.
            # Each file still gets a floor so a big early file can't starve
            # every later one down to nothing.
            budget = _FILES_IN_SCOPE_CHAR_BUDGET
            per_file_cap = max(budget // max(len(step_context.files), 1), _FILES_IN_SCOPE_MIN_PER_FILE)
            lines.append("\nCurrent content of files in scope (live read, not memory):")
            shown = 0
            for path in step_context.files:
                if shown >= budget:
                    remaining = len(step_context.files) - step_context.files.index(path)
                    lines.append(f"\n[{remaining} more file(s) in scope omitted -- over the state-report "
                                 f"character budget; read them directly with read_file if needed]")
                    break
                try:
                    content = self.toolbox.dispatch("read_file", {"path": path, "limit": 200})
                except ToolError as e:
                    content = f"(could not read {path}: {e})"
                block = f"\n--- {path} ---\n{content[:per_file_cap]}"
                lines.append(block)
                shown += len(block)

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

    def _try_replan(self, state: RunState, step_context: StepContext,
                     failure_reason: str, last_failure_detail: str) -> bool:
        """
        Mechanical switch, not a tool call the model chooses: fires only
        when a step has exhausted max_subtask_attempts. Exists to limit
        human involvement in a FAILED RESPONSE -- a step that's stuck --
        as distinct from a genuine blocker, which still escalates.

        Returns True if a replan happened (step_context has been revised
        in place and the caller should reset its attempt counters and
        keep going), or False once max_replan_cycles is exhausted or the
        planner call itself failed, meaning the caller must escalate.
        """
        if step_context.replan_count >= self.config.budget.max_replan_cycles:
            return False

        prompt = (
            f"STEP: {step_context.title}\n"
            f"CURRENT OBJECTIVE:\n{step_context.objective}\n\n"
            f"CURRENT ACCEPTANCE COMMAND:\n  {step_context.acceptance_command}\n\n"
            f"This step exhausted {self.config.budget.max_subtask_attempts} attempts "
            f"without passing.\n{failure_reason}\n\n"
            f"LAST FAILURE DETAIL:\n{last_failure_detail or '(none recorded)'}\n\n"
            f"SCRATCHPAD:\n{state.scratchpad or '(empty)'}\n\n"
            "Decide whether the OBJECTIVE or ACCEPTANCE COMMAND itself was wrong given "
            "this evidence, and produce a corrected version a coder can actually complete. "
            "Keep the same scope -- narrow or fix it, don't expand it into new work.\n\n"
            "Respond with EXACTLY two lines, nothing else:\n"
            "OBJECTIVE: <revised objective, or the same text if it was fine>\n"
            "ACCEPTANCE_COMMAND: <revised command, or the same command if it was fine>"
        )
        try:
            response = self.planner_llm.complete(
                system=(
                    "You are the replanning pass for a stuck autonomous coding step. "
                    "You do not write code -- you only decide whether the step's "
                    "definition was wrong and correct it."
                ),
                history=[Turn(role="user", text=prompt)],
                tools=[],
                max_tokens=self.config.budget.max_output_tokens,
            )
        except LLMError as e:
            self.session.log_event("replan_llm_error", {"title": step_context.title, "error": str(e)})
            return False  # can't reach the planner -- don't fake a cycle, just escalate

        step_context.replan_count += 1
        step_context.revisions_this_cycle = 0
        new_objective, new_cmd = _parse_replan_response(response.text or "")
        if new_objective:
            step_context.objective = new_objective
        if new_cmd:
            problem = _validate_acceptance_command(new_cmd)
            if problem:
                self.session.log_event("replan_acceptance_command_rejected", {
                    "title": step_context.title, "reason": problem, "command": new_cmd,
                })
            elif new_cmd != step_context.acceptance_command:
                step_context.acceptance_command = new_cmd
                step_context.baseline_check_result = self._run_baseline_check(step_context)

        print(f"    [replan] cycle {step_context.replan_count}/{self.config.budget.max_replan_cycles} "
              f"for '{step_context.title}'")
        self.session.log_event("replan_triggered", {
            "title": step_context.title,
            "replan_count": step_context.replan_count,
            "reason": failure_reason,
            "new_objective": new_objective,
            "new_acceptance_command": new_cmd,
        })
        return True

    def _run_self_audit(self, state: RunState) -> None:
        """
        FLAW 8 + FLAW 14, wired together as one mechanism rather than two:
        every `self_audit_every_n_steps` completed steps, a dedicated
        planner_llm pass (not offered to the model as a tool -- this is a
        harness-scheduled checkpoint, not a model choice) reviews the goal,
        the decomposition, and the completed-step log, then:
          - flags if the run looks like it's drifted off the actual goal
            (FLAW 8: nothing previously paused the loop to ask this on its
            own schedule -- only the repetition detectors, which react to
            explicit stalling/cycling, not silent drift while steps keep
            passing);
          - distills the step log into a fresh scratchpad note (FLAW 14:
            replaces the previous scratchpad rather than letting it grow
            or go stale, a deliberate "take stock" pass on a schedule
            rather than only reactively at file-batch/condensation points).
        Best-effort: an LLMError here is logged and skipped, exactly like
        the existing condensation passes -- a failed audit must never be
        the thing that halts or derails an otherwise-healthy run.
        """
        prompt = (
            f"GOAL:\n{state.goal}\n\n"
            f"GOAL BREAKDOWN:\n{_decomposition_with_reconciliation_hints(state)}\n\n"
            f"COMPLETED STEPS SO FAR:\n{state.completed_summary_text()}\n\n"
            f"CURRENT SCRATCHPAD:\n{state.scratchpad or '(empty)'}\n\n"
            "Take stock. Based ONLY on the evidence above:\n"
            "1. Is this run still working toward the stated GOAL, or has it "
            "drifted onto something else?\n"
            "2. Is there a genuinely new, distinct piece of the goal that "
            "isn't yet tracked in the breakdown and should be added?\n"
            "3. Do any [ ] (not yet checked off) breakdown items above look "
            "genuinely satisfied by the completed steps -- not just a '>>' "
            "hint, actually satisfied on reading the evidence? List their ids.\n"
            "4. Distill the completed-steps log into a short note capturing "
            "what actually matters going forward -- this REPLACES the "
            "current scratchpad, so keep anything from it that's still "
            "relevant.\n\n"
            "Respond with EXACTLY four lines, nothing else:\n"
            "ON_TRACK: yes or no\n"
            "NEW_SUBTASK: <short title, or NONE>\n"
            "SATISFIED_SUBTASKS: <comma-separated ids, or NONE>\n"
            "NOTE: <the distilled note>"
        )
        try:
            response = self.planner_llm.complete(
                system=(
                    "You are the periodic self-audit pass for an autonomous coding "
                    "run. You do not write code and you do not decide the next step "
                    "-- you only check the run is still on track, reconcile the "
                    "tracked breakdown against what's actually been completed, and "
                    "distill progress so far."
                ),
                history=[Turn(role="user", text=prompt)],
                tools=[],
                max_tokens=self.config.budget.max_output_tokens,
            )
        except LLMError as e:
            self.session.log_event("self_audit_llm_error", {"error": str(e)})
            return

        on_track, new_subtask, note, satisfied_ids = _parse_self_audit_response(response.text or "")
        if new_subtask:
            state.decomposition.add(new_subtask)

        # Real bug, seen in an actual run: the model set a decomposition,
        # did all the matching work across 8 steps, and never once called
        # check_off_subtask -- every entry sat at "pending" the entire run.
        # The context-level hint (_decomposition_with_reconciliation_hints)
        # is the first line of defense; this is the second -- a dedicated
        # review pass explicitly asked to reconcile, with the authority to
        # actually check things off rather than only hoping the outer-loop
        # model notices a hint. Unlike the outer loop's check_off_subtask
        # tool, self-audit isn't a model choice about what to do next, it's
        # a harness-scheduled correctness pass -- same trust level as the
        # scratchpad rewrite it already does unprompted.
        newly_satisfied = []
        for subtask_id in satisfied_ids:
            subtask = state.decomposition.get(subtask_id)
            if subtask is not None and subtask.status != "done":
                subtask.status = "done"
                newly_satisfied.append(subtask_id)
        if newly_satisfied:
            self.session.log_event("self_audit_reconciled_subtasks", {"ids": newly_satisfied})

        # FLAW 2: piggyback the regression check on this same checkpoint,
        # rather than only discovering a regression once, all at once, at
        # declare_done. Catching it here means it surfaces on the model's
        # very next turn instead of after however many more steps happen
        # to follow before the goal is declared done.
        regression_feedback = self._check_no_regressions(state)
        if regression_feedback:
            on_track = False
            note = (regression_feedback + "\n\n" + note).strip() if note else regression_feedback
            self.session.log_event("self_audit_found_regression", {})

        if note:
            drift_prefix = (
                "[SELF-AUDIT: this run may have drifted from the goal -- re-read "
                "GOAL below before continuing]\n" if not on_track else ""
            )
            self._set_scratchpad(state, drift_prefix + note)
        self.session.log_event("self_audit", {
            "on_track": on_track, "new_subtask": new_subtask, "has_note": bool(note),
            "satisfied_subtasks": newly_satisfied,
        })
        self.session.save_state(state)

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
        # Same condensation mechanism as the outer loop (Agent._condense_batch),
        # extended here because a model can do all its "understanding
        # building" reading inside a step instead of during outer-loop
        # exploration -- confirmed happening in practice: propose_step was
        # called after only 2 list_dir calls, then dozens of read_file
        # calls happened here with none of the outer loop's memory safety
        # nets in effect. Persists across attempts within this step
        # (failed attempts don't erase understanding already built).
        inner_exploration_log: list[tuple[str, str]] = []
        inner_files_since_condense: set[str] = set()
        condense_batch_size = self._condensation_batch_size()
        # Paths written/edited since the last successful mark_step_done.
        # Deliberately NOT condensed on every write (see _condense_batch's
        # dict design -- that guards against overwrite, this is a separate
        # problem: a condensed_files entry describing pre-edit content once
        # the model has since changed that file). Reset on a genuine fresh
        # attempt (below) since a revert wipes whatever it would describe;
        # left alone across a failed-but-continuing mark_step_done because
        # the eventual flush always re-reads the file fresh off disk, so an
        # over-inclusive set only costs an extra read, never a wrong note.
        dirty_files: set[str] = set()

        while True:
            if attempts >= self.config.budget.max_subtask_attempts:
                # MECHANICAL switch, not a tool the model calls: the harness
                # itself decides whether to retry autonomously (replan) or
                # give up and page a human (escalate). This exists to limit
                # human involvement in a FAILED RESPONSE -- a step that's
                # stuck -- as distinct from a genuine blocker; see
                # _try_replan for what "stuck" gets one more shot at.
                failure_reason = _inner_loop_failure_reason(step_context.title, check_failures, derailed_attempts)
                if self._try_replan(state, step_context, failure_reason, last_failure_detail):
                    attempts = 0
                    check_failures = 0
                    derailed_attempts = 0
                    last_attempt_note = (
                        f"This step was autonomously replanned (cycle {step_context.replan_count}/"
                        f"{self.config.budget.max_replan_cycles}) after exhausting its attempts. "
                        "Work toward the objective and acceptance_command below -- they may have "
                        "been revised based on why the previous attempts failed."
                    )
                    continue
                return None, failure_reason
            self.budget.new_attempt()
            # Clean slate: discard whatever the previous attempt (if any)
            # left behind. Without this, a failed attempt's half-finished
            # or broken edits stay in the working tree and get inherited by
            # the next attempt -- and if THAT attempt passes, git_commit_all's
            # `git add -A` sweeps the leftover junk in as part of a
            # "passing" commit. A no-op on the first attempt (nothing to
            # revert yet).
            self.workspace.git_revert_to_last_commit()
            dirty_files = set()
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

                try:
                    response = self._call_llm(state, system, history, INNER_TOOLS)
                except BudgetExceeded as e:
                    # Same funnel as the budget.step() path above: this can
                    # fire mid-attempt (record_usage's max_total_tokens
                    # ceiling, raised deep inside _call_llm) just as easily
                    # as the step-count budget can. Without this catch it
                    # propagated uncaught past the attempt loop entirely --
                    # no revert, no replan chance -- abandoning whatever the
                    # attempt had in progress. Routing it through the same
                    # attempts/derailed_attempts counters means it now gets
                    # the same replan-then-escalate treatment.
                    self.session.log_event("total_token_budget_exceeded", {"title": step_context.title, "detail": str(e)})
                    derailed_attempts += 1
                    attempts += 1
                    last_attempt_note = (
                        "A previous attempt on this step ran out of the total token budget "
                        "without ever calling mark_step_done. Its edits have been reverted -- "
                        "the workspace is back to the last passing commit, so start clean. "
                        "Work directly and efficiently toward mark_step_done this time."
                    )
                    break  # try another attempt, with fresh history
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
                            state.condensed_files,
                        )
                        history = compact_history_with_state_report(history, report)
                        self.session.log_event("state_report_compaction", {"title": step_context.title})
                    continue

                mark_call = next((c for c in response.tool_calls if c.name == "mark_step_done"), None)
                discard_call = next((c for c in response.tool_calls if c.name == "discard_and_restart"), None)
                revise_call = next((c for c in response.tool_calls if c.name == "revise_acceptance_command"), None)
                tool_results: list[ToolResult] = []

                for call in response.tool_calls:
                    if call.name in ("mark_step_done", "discard_and_restart", "revise_acceptance_command"):
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
                    if call.name in ("write_file", "edit_file") and not tool_results[-1].is_error:
                        target = call.input.get("path") or ""
                        if target:
                            dirty_files.add(target)
                    if call.name in ("read_file", "code_skeleton", "search_code") and not tool_results[-1].is_error:
                        target = call.input.get("path") or call.input.get("query") or ""
                        inner_exploration_log.append((f"{call.name}({target})", tool_results[-1].content))
                        if call.name in ("read_file", "code_skeleton") and target:
                            inner_files_since_condense.add(target)
                            if len(inner_files_since_condense) >= condense_batch_size:
                                self._condense_batch(state, inner_files_since_condense, inner_exploration_log)
                                inner_files_since_condense = set()

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

                if revise_call is not None:
                    # Fixes the CHECK, not the code -- deliberately lighter
                    # weight than discard_and_restart: no revert, no fresh
                    # history, no attempt cost. Stays in the same attempt;
                    # the model just keeps going against the corrected
                    # command. Capped at 1/step, mechanically -- not a
                    # judgment call about whether a given revision is
                    # legitimate (see REVISE_ACCEPTANCE_COMMAND_TOOL's
                    # description for why that judgment isn't the harness's
                    # to make), just a bound on how far this can go unseen.
                    if step_context.revisions_this_cycle >= 1:
                        tool_results.append(ToolResult(
                            tool_call_id=revise_call.id, is_error=True,
                            content=(
                                "acceptance_command has already been revised once this "
                                "cycle -- that's the limit. Keep working against the current "
                                "acceptance_command, or call discard_and_restart if the "
                                "APPROACH needs to change, not the check."
                            ),
                        ))
                    else:
                        new_cmd = str(revise_call.input.get("new_acceptance_command", "")).strip()
                        reason = str(revise_call.input.get("reason", ""))
                        revise_problem = _validate_acceptance_command(new_cmd) if new_cmd else None
                        if not new_cmd:
                            tool_results.append(ToolResult(
                                tool_call_id=revise_call.id, is_error=True,
                                content="new_acceptance_command must be non-empty",
                            ))
                        elif revise_problem:
                            tool_results.append(ToolResult(
                                tool_call_id=revise_call.id, is_error=True,
                                content=(
                                    f"new_acceptance_command is invalid and would fail every "
                                    f"time: {revise_problem}\nCommand was: {new_cmd}\n"
                                    "This attempt at revise_acceptance_command was rejected and "
                                    "does not count against your one revision -- call it again "
                                    "with a corrected command."
                                ),
                            ))
                        else:
                            old_cmd = step_context.acceptance_command
                            step_context.acceptance_command = new_cmd
                            step_context.acceptance_command_revisions += 1
                            step_context.revisions_this_cycle += 1
                            print(f"    [revise] acceptance_command for '{step_context.title}': {reason[:200]}")
                            self.session.log_event("acceptance_command_revised", {
                                "title": step_context.title, "old_command": old_cmd,
                                "new_command": new_cmd, "reason": reason,
                            })
                            baseline = self._run_baseline_check(step_context)
                            step_context.baseline_check_result = baseline
                            # Rebuild for the rest of THIS attempt so every
                            # subsequent turn sees the corrected command and
                            # its fresh baseline -- not just the tool result
                            # below, which only the model's next turn reads.
                            system = self._inner_system_prompt(state, step_context)
                            tool_results.append(ToolResult(
                                tool_call_id=revise_call.id,
                                content=(
                                    f"acceptance_command updated to:\n  {new_cmd}\n\n"
                                    f"Baseline against the CURRENT workspace with this new "
                                    f"command (same as what you saw at the start of this step, "
                                    f"just re-run against the corrected command):\n{baseline}"
                                ),
                            ))

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
                        if dirty_files:
                            self._flush_dirty_condensed(state, dirty_files)
                        if last_failure_detail:
                            self.lessons.add(
                                context=f"step: {step_context.title}",
                                symptom=last_failure_detail,
                                fix=f"eventually succeeded with: {summary}",
                                intent=step_context.objective or state.goal,
                            )
                        return CompletedStep(
                            index=state.next_index(),
                            title=step_context.title,
                            summary=summary,
                            acceptance_command=step_context.acceptance_command,
                            stdout=result.stdout,
                            stderr=result.stderr,
                            exit_code=result.exit_code,
                            original_acceptance_command=(
                                step_context.original_acceptance_command
                                if step_context.acceptance_command_revisions > 0 else None
                            ),
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
                        break  # top of the outer loop decides: replan or escalate

                    tool_results.append(ToolResult(
                        tool_call_id=mark_call.id, is_error=True,
                        content=(
                            f"Acceptance check FAILED (attempt {attempts}/"
                            f"{self.config.budget.max_subtask_attempts}).\n"
                            f"{failure_detail}\n\n"
                            "Your edits for this attempt have been reverted -- the workspace "
                            "is back to the last passing commit, not what you just wrote. "
                            "Anything you confirmed working a moment ago (via read_file, "
                            "run_command, a test run, etc.) no longer reflects the current "
                            "files -- re-verify before relying on it, don't assume it's still "
                            "there. Fix the actual underlying problem, then call "
                            "mark_step_done again."
                        ),
                    ))

                if tool_results:
                    history.append(Turn(role="tool_results", tool_results=tool_results))
                    if self._prompt_over_budget(system, history, INNER_TOOLS):
                        report = self._build_state_report(
                            step_context, attempts,
                            self.config.budget.max_subtask_attempts, last_check_summary,
                            state.condensed_files,
                        )
                        history = compact_history_with_state_report(history, report)
                        self.session.log_event("state_report_compaction", {"title": step_context.title})

    # ── done verification ──────────────────────────────────────────────────

    def _check_no_regressions(self, state: RunState) -> str | None:
        """Re-run every previously-completed step's acceptance command --
        AND (FLAW 2) every additional_checks entry added later via
        strengthen_step_check, since a step's original command was only
        ever proven against the input/case that existed when it was
        written. Without this, step 7 can silently break what step 3
        verified -- each step's check only runs once, at the moment it's
        marked done, and git_commit_all's `git add -A` sweeps up whatever
        the working tree looks like at that point regardless. Returns None
        if every check on every prior step still passes, or feedback text
        (naming which step and which specific command regressed) to send
        back to the model if not."""
        for step in state.completed_steps:
            checks = [step.acceptance_command] + step.additional_checks
            for cmd in checks:
                result = self.toolbox.run_gated_command(
                    cmd, timeout=self.config.budget.default_command_timeout,
                )
                if result.exit_code != 0 or result.timed_out:
                    print(f"[regression-check] step {step.index} ('{step.title}') now FAILS: {cmd}")
                    return (
                        f"Regression detected: step {step.index} ('{step.title}') used to pass "
                        f"this check but does not anymore.\n"
                        f"$ {cmd}\n"
                        f"exit {result.exit_code}{' (TIMED OUT)' if result.timed_out else ''}\n"
                        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}\n"
                        "Fix this regression before declaring the goal done."
                    )
        if state.completed_steps:
            total_checks = sum(1 + len(s.additional_checks) for s in state.completed_steps)
            print(f"[regression-check] {total_checks}/{total_checks} checks across "
                  f"{len(state.completed_steps)} prior step(s) still pass")
        return None

    def _verify_done(self, state: RunState, summary: str,
                      final_acceptance_command: str | None) -> tuple[bool, str]:
        """Returns (accepted, feedback). feedback is empty when accepted;
        otherwise it's real command output (acceptance-command path) or
        whatever the human typed (interactive path) -- either way it goes
        back to the model, not just a generic 'try again'."""
        # Real bug, seen in an actual run: the model declared the goal
        # done citing "three" completed items while the tracked breakdown
        # actually had four, two of which were still "pending" despite
        # matching completed work almost exactly. The reconciliation hint
        # (_decomposition_with_reconciliation_hints) only ever SUGGESTS --
        # it can't help if the model doesn't act on it before declaring
        # done. This sweep is the backstop: applied unconditionally, right
        # here, regardless of accept/reject outcome, so the persisted
        # decomposition record reflects reality by the time anyone (the
        # human reviewing this, or a future reopen()) looks at it. Unlike
        # the set_goal_decomposition preservation fix, getting this match
        # wrong is low-stakes -- worst case a bookkeeping entry reads
        # "done" a little generously; it can never flip something the
        # other direction or affect what's actually been verified.
        reconciled_ids = []
        for s in state.decomposition.subtasks:
            if s.status != "done" and _find_satisfying_completed_step(s, state.completed_steps):
                s.status = "done"
                reconciled_ids.append(s.id)
        if reconciled_ids:
            self.session.log_event("declare_done_reconciled_subtasks", {"ids": reconciled_ids})
            self.session.save_state(state)

        regression_feedback = self._check_no_regressions(state)
        if regression_feedback:
            return False, regression_feedback

        if final_acceptance_command:
            return self._run_final_acceptance_command(final_acceptance_command)

        # No final acceptance command -- ask the human.
        print(f"\n[done] model declares goal complete:\n  {summary}")
        self.budget.pause()
        try:
            answer = input("Accept as done? [y/N]: ").strip().lower()
        except EOFError:
            # No one's there to answer -- an unattended run with no TTY.
            # Left uncaught this crashes the whole run right as it was
            # trying to finish cleanly. Default to the same outcome a
            # human declining silently would produce.
            print("[no input available -- treating as not accepted]")
            self.budget.resume()
            return False, ("No human was available to confirm completion (unattended run, "
                            "no TTY). Re-examine your work critically, or supply a final "
                            "acceptance command next run so this can be checked automatically.")
        if answer in ("y", "yes"):
            self.budget.resume()
            return True, ""
        try:
            reason = input("What's missing or not right? (sent back to the agent): ").strip()
        except EOFError:
            reason = ""
        self.budget.resume()
        if not reason:
            reason = "Human rejected without giving a specific reason. Re-examine your work critically."
        return False, reason

    def _run_final_acceptance_command(self, final_acceptance_command: str) -> tuple[bool, str]:
        """Runs the goal-level acceptance command and returns (accepted,
        feedback), same contract as _verify_done. Split out so the outer
        loop can also run this speculatively -- e.g. before letting a new
        propose_step through -- without duplicating the command-execution
        and feedback-formatting logic."""
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

    def _finalize_pending_lesson(self, fix_summary: str) -> None:
        pending = self._pending_escalation_lesson
        if pending is None:
            return
        self.lessons.add(
            context=pending["context"],
            symptom=f"{pending['symptom']}\nHuman guidance given: {pending['guidance']}",
            fix=f"run continued and succeeded afterward: {fix_summary}",
            intent=pending.get("intent", ""),
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
            "intent": state.goal,
        }
        # Fresh budget span -- without this, the next outer_step() call would
        # immediately re-raise on an already-over-cap counter and guidance
        # could never actually resume the run.
        self.budget.new_outer_span()

    # ── system prompts ─────────────────────────────────────────────────────

    def _outer_system_prompt(self, state: RunState, final_acceptance_command: str | None = None) -> str:
        if final_acceptance_command:
            goal_check_section = f"""GOAL-LEVEL COMPLETION CHECK: the overall goal is considered complete once
this exact command exits 0:
    $ {final_acceptance_command}
You can run this yourself via run_command at any point to check where you
actually stand -- it's read-only investigation, not a mutation, so it's
allowed here. Checking it before deciding whether to propose another step
or call declare_done costs you one tool call and removes the guesswork.
Note this is still just a mechanical proxy for the goal, not a substitute
for actually thinking the goal is met -- passing it doesn't excuse
incomplete or careless work, and it failing doesn't necessarily mean your
last step was wrong."""
        else:
            goal_check_section = (
                "GOAL-LEVEL COMPLETION CHECK: none was supplied for this run -- "
                "declare_done will ask a human to confirm completion instead."
            )
        return f"""You are an autonomous coding agent working in a real git repository.

OS: {shell_description()}

GOAL: {state.goal}

{goal_check_section}

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
- run_command IS available here, but for read-only investigation only --
  listing, reading, compiling, running tests. It will be rejected if it
  looks like it would create, modify, delete, or install anything (that's
  still a mutation without verification, same reasoning as write_file
  above, and use of the workspace's real shell doesn't exempt it).
- This includes non-code deliverables (a summary, a report, any file with
  real content) -- there is no other way to create or write one from here.
  On Windows specifically, run_command can't build multi-line file content
  at all (no heredocs, no multi-line `python -c`), so a document-writing
  goal MUST go through propose_step to get write_file, even though nothing
  about the task looks like "coding". Trying to assemble a file line by
  line through repeated single-line shell commands is fragile and NOT a
  substitute -- propose_step for it instead.
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
- If the goal has multiple genuinely separate parts, call
  set_goal_decomposition once, early, to track them; check items off with
  check_off_subtask as they're finished. Skip this for a goal that's
  already one focused piece of work -- it exists for tracking, not busywork.
  Do this checking-off AS PART OF finishing a step that satisfies one of
  the tracked parts, not as a separate thing to remember later -- the
  breakdown is only useful if it stays current. A '>>' hint next to a
  breakdown item means a completed step looks like it satisfies it; verify
  and check it off if so.
- If you discover an already-completed step's acceptance check only
  proved a narrower property than what later work now depends on, use
  strengthen_step_check to add a further command to it. Every completed
  step's checks (original plus any added) are re-run before the goal can
  be declared done, and periodically during the run.
- Only call declare_done when ALL work is complete and verifiable.
- Do NOT propose_step just to "declare completion" or "finalize" -- call
  declare_done directly instead. It's independently verified already; a
  step wrapping it adds a redundant, self-invented check that can go stale
  (e.g. checking for a file from an earlier, since-corrected plan) and fail
  for reasons unrelated to whether the actual goal was met.
"""

    def _inner_system_prompt(self, state: RunState, step_context: StepContext) -> str:
        # FLAW 13: match lessons against THIS step's actual failure surface
        # (title+objective) rather than just showing the most recent N --
        # a lesson from an unrelated part of the project shown here purely
        # because it's recent is a distraction, not a lesson learned.
        # Falls back to the recent-N view when nothing scores a match, so
        # early in a project (few lessons, little keyword overlap yet)
        # lessons are never silently hidden.
        matches = self.lessons.find_relevant(f"{step_context.title} {step_context.objective}")
        lessons_text = LessonsStore.render(matches) if matches else self.lessons.summary_text()
        return f"""You are an autonomous coding agent working in a real git repository.

OS: {shell_description()}

GOAL: {state.goal}

COMPLETED SO FAR:
{state.completed_summary_text()}

LESSONS FROM PAST RUNS IN THIS WORKSPACE (verified -- each was an actual problem
followed by a confirmed successful outcome, not a guess):
{lessons_text}

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
If mark_step_done keeps failing and you have concrete evidence the ACCEPTANCE
COMMAND itself is wrong (not your code) -- it references something that can't
exist, depends on a program not available on this platform, checks the wrong
path -- call revise_acceptance_command instead of continuing to grind against a
check that can never pass. Usable once per step.
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


# Command-starting tokens that don't exist on cmd.exe -- catches the same
# failure shape as the python -c checks below: a command that would fail
# every single time on this machine regardless of what gets implemented,
# discovered only after burning a full attempt (and, before the stash fix
# above, after the edit that would have passed it was destroyed).
def _validate_acceptance_command(acceptance_cmd: str) -> str | None:
    """Runs all cheap, deterministic acceptance_command checks and returns
    the first problem found, or None. OS-compatibility check now lives in
    tools.validate_command_for_os, shared with run_command's own gate in
    run_gated_command, so the same set of known-bad patterns is caught in
    both places instead of only at acceptance-check time."""
    return validate_command_for_os(acceptance_cmd) or _validate_python_dash_c_syntax(acceptance_cmd)


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


def _parse_replan_response(text: str) -> tuple[str, str]:
    """Parses the replan pass's `OBJECTIVE: ...` / `ACCEPTANCE_COMMAND: ...`
    two-line format. Tolerant of a small local model wrapping either value
    across multiple lines (everything up to the next recognized label is
    folded into the current one) -- but the labels themselves must appear
    at line start, same tolerance level as _parse_condensed_sections.
    Returns ('', '') for either field the response didn't include, so the
    caller can tell 'no change' apart from 'this replan produced nothing
    usable' without guessing."""
    objective_lines: list[str] = []
    command_lines: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.upper().startswith("OBJECTIVE:"):
            current = objective_lines
            current.append(line.split(":", 1)[1].strip())
        elif line.upper().startswith("ACCEPTANCE_COMMAND:"):
            current = command_lines
            current.append(line.split(":", 1)[1].strip())
        elif current is not None:
            current.append(line)
    return "\n".join(objective_lines).strip(), "\n".join(command_lines).strip()


def _parse_self_audit_response(text: str) -> tuple[bool, str, str, list[str]]:
    """Parses the self-audit pass's four-line format:
        ON_TRACK: yes|no
        NEW_SUBTASK: <title, or NONE>
        SATISFIED_SUBTASKS: <comma-separated ids, or NONE>
        NOTE: <distilled note to carry forward>
    Same tolerance rules as _parse_replan_response. Defaults to on_track=True
    on a malformed/unparseable response -- a best-effort audit pass that
    came back garbled should never itself be the thing that halts or
    derails a run; it just means this cycle's audit didn't add anything."""
    on_track = True
    new_subtask = ""
    satisfied_ids: list[str] = []
    note_lines: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        upper = line.upper()
        if upper.startswith("ON_TRACK:"):
            on_track = "no" not in line.split(":", 1)[1].strip().lower()
            current = None
        elif upper.startswith("NEW_SUBTASK:"):
            new_subtask = line.split(":", 1)[1].strip()
            current = None
        elif upper.startswith("SATISFIED_SUBTASKS:"):
            raw = line.split(":", 1)[1].strip()
            if raw.upper() != "NONE" and raw:
                satisfied_ids = [s.strip() for s in raw.split(",") if s.strip()]
            current = None
        elif upper.startswith("NOTE:"):
            current = note_lines
            current.append(line.split(":", 1)[1].strip())
        elif current is not None:
            current.append(line)
    note = "\n".join(note_lines).strip()
    if new_subtask.upper() == "NONE":
        new_subtask = ""
    return on_track, new_subtask, note, satisfied_ids


def _find_satisfying_completed_step(subtask: "Subtask", completed_steps: list[CompletedStep]) -> CompletedStep | None:
    """Shared by the reconciliation hint (surfaces a suggestion, never
    writes) and the declare_done reconciliation sweep (actually applies
    it, see _verify_done). Same deterministic keyword-overlap approach
    used throughout this file (lessons matching, the thrash detector) --
    not fuzzy/ML matching. Returns the strongest-overlap completed step,
    or None if nothing scores at least 2 shared significant keywords."""
    subtask_words = _keywords(subtask.title)
    best: tuple[int, CompletedStep] | None = None
    for step in completed_steps:
        overlap = len(subtask_words & _keywords(f"{step.title} {step.summary}"))
        if overlap >= 2 and (best is None or overlap > best[0]):
            best = (overlap, step)
    return best[1] if best else None


def _decomposition_with_reconciliation_hints(state: RunState) -> str:
    """
    Real bug, seen in an actual run: the model called set_goal_decomposition
    once, then did all four matching pieces of work across 8 completed
    steps, and never once called check_off_subtask -- every subtask sat at
    "pending" for the entire run despite being done, because nothing ever
    prompted reconciliation between the two. Self-audit alone wasn't
    enough: it only fires every N steps and only ever ADDS subtasks, never
    reconciles existing ones.

    This is a deterministic (same keyword-overlap approach as
    lessons.find_relevant), always-on hint shown on EVERY outer turn, not
    just at self-audit cadence -- it never auto-checks anything off itself
    (that stays the model's call, same as every other tool use in this
    harness), it just makes the mismatch impossible to miss for more than
    one turn.
    """
    if not state.decomposition.subtasks:
        return state.decomposition.summary_text()
    lines = []
    for s in state.decomposition.subtasks:
        note = f" -- {s.notes}" if s.notes else ""
        if s.status == "done":
            lines.append(f"  [x] ({s.id}) {s.title}{note}")
            continue
        match = _find_satisfying_completed_step(s, state.completed_steps)
        if match:
            lines.append(
                f"  [ ] ({s.id}) {s.title}{note}\n"
                f"      >> looks satisfied by completed step {match.index} "
                f"('{match.title}') -- call check_off_subtask if it actually is"
            )
        else:
            lines.append(f"  [ ] ({s.id}) {s.title}{note}")
    return "\n".join(lines)


_CONSECUTIVE_SIMILAR_STEP_THRESHOLD = 3


def _steps_look_like_repeats(a: CompletedStep, b: CompletedStep) -> bool:
    """
    Real bug, seen in an actual run: 27 of 35 completed steps were all
    "fix data/expenses.csv duplicate row" under cosmetically different
    titles ('Fix...', 'Rewrite...', 'Trim...', 'Remove duplicate...'),
    because a regression kept recurring (something else in the run was
    mutating a tracked data file as a side effect) and the model treated
    each recurrence as a fresh, unrelated problem instead of recognizing
    the pattern. Nothing caught this: each individual step DID pass its
    own acceptance check when it ran, so the inner-loop stall/repetition
    detectors (which react to a step's own tool-call pattern) never fired,
    and self-audit's regression detection only ever left a note -- it
    never stopped the model from proposing yet another near-identical fix.

    Deliberately narrow and deterministic (same keyword-overlap approach
    as lessons.find_relevant, not fuzzy/ML matching): two completed steps
    "look like repeats" only if their titles share enough significant
    keywords that they're very likely re-attempting the same underlying
    fix, not just coincidentally similar wording for genuinely different
    work.
    """
    a_words = _keywords(a.title)
    b_words = _keywords(b.title)
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    smaller = min(len(a_words), len(b_words))
    return overlap >= 2 and overlap / smaller >= 0.5


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
