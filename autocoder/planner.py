"""
Incremental step-by-step orchestration state.

Replaces the upfront multi-step plan with a running log: after each
completed step the model sees the full evidence of what happened before
deciding what to do next. There are no pre-declared dependencies, no
upfront subtask graph, and no possibility of a dependency-cycle deadlock
caused by the model speculating about a future it hasn't seen yet.

The only persistent state is RunState: the goal, an ordered log of
completed steps (each with its acceptance evidence), and a scratchpad
the model can freely update with working notes.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Literal

MAX_SUBTASKS = 40


@dataclass
class Subtask:
    id: str
    title: str
    status: Literal["pending", "done"] = "pending"
    notes: str = ""


@dataclass
class GoalDecomposition:
    """Durable structure for 'what's left', so the outer loop isn't
    re-interpreting a bare goal string on every read. Deliberately just a
    flat list with a status flag -- no dependency graph, no ordering
    constraints. The outer loop already provides sequencing (one
    propose_step at a time); this only tracks which pieces of a
    multi-part goal are still open, so a step doesn't need to squint at
    the original goal text to figure out what remains.
    """
    subtasks: list[Subtask] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"subtasks": [asdict(s) for s in self.subtasks]}

    @classmethod
    def from_dict(cls, d: dict | None) -> "GoalDecomposition":
        if not d:
            return cls()
        return cls(subtasks=[Subtask(**s) for s in d.get("subtasks", [])])

    def get(self, subtask_id: str) -> Subtask | None:
        for s in self.subtasks:
            if s.id == subtask_id:
                return s
        return None

    def add(self, title: str) -> Subtask:
        """Append one new entry without disturbing existing ones -- distinct
        from replace(), which re-plans the whole breakdown. Used by
        Agent.reopen(): re-scoping a finished run must not silently wipe
        out the record of what was already tracked as done."""
        subtask = Subtask(id=uuid.uuid4().hex[:8], title=title[:200])
        if len(self.subtasks) < MAX_SUBTASKS:
            self.subtasks.append(subtask)
        return subtask

    def replace(self, titles: list[str]) -> tuple[list[Subtask], list[str], list[Subtask]]:
        """
        Full replace, used by set_goal_decomposition -- but NOT a blind
        wipe. Real bug, seen in an actual run: re-declaring the breakdown
        mid-run (e.g. to add a genuinely new item) used to reassign fresh
        ids and reset EVERY entry to pending, including ones already
        checked off -- silent, total data loss, with no warning and
        nothing in the log beyond a bare count.

        Now: any new title that exactly matches (case/whitespace
        insensitive) an existing entry's title carries over that entry's
        id, status, and notes. A title that's reworded, even slightly, is
        NOT matched -- deliberately exact, not fuzzy. This operation
        WRITES state (a status flip); unlike the reconciliation hint
        (which only ever suggests, never writes) or the declare_done
        sweep (where a wrong match is harmless bookkeeping), a wrong
        match here could silently un-do or fabricate a "done" status.
        Precision over recall.

        Returns (preserved, added_titles, dropped):
          preserved     -- Subtasks that kept their id/status/notes.
          added_titles  -- titles with no old match (fresh pending entries).
          dropped       -- old Subtasks that didn't reappear at all (including
                            any that were "done" -- worth flagging to the caller,
                            since this is the one case that's a genuine loss of
                            tracked completion, not a reset).
        """
        old_by_title = {s.title.strip().lower(): s for s in self.subtasks}
        seen_old_keys: set[str] = set()
        preserved: list[Subtask] = []
        added_titles: list[str] = []
        new_subtasks: list[Subtask] = []
        for t in titles[:MAX_SUBTASKS]:
            key = t.strip().lower()
            old = old_by_title.get(key)
            if old is not None:
                carried = Subtask(id=old.id, title=t, status=old.status, notes=old.notes)
                new_subtasks.append(carried)
                preserved.append(carried)
                seen_old_keys.add(key)
            else:
                new_subtasks.append(Subtask(id=uuid.uuid4().hex[:8], title=t))
                added_titles.append(t)
        dropped = [s for s in self.subtasks if s.title.strip().lower() not in seen_old_keys]
        self.subtasks = new_subtasks
        return preserved, added_titles, dropped

    def summary_text(self) -> str:
        if not self.subtasks:
            return "(not decomposed -- use set_goal_decomposition if the goal has multiple distinct parts worth tracking separately)"
        lines = []
        for s in self.subtasks:
            mark = "x" if s.status == "done" else " "
            note = f" -- {s.notes}" if s.notes else ""
            lines.append(f"  [{mark}] ({s.id}) {s.title}{note}")
        return "\n".join(lines)


@dataclass
class CompletedStep:
    index: int
    title: str
    summary: str          # model's description of what it did
    acceptance_command: str
    stdout: str
    stderr: str
    exit_code: int
    # Set only if revise_acceptance_command was used during this step --
    # acceptance_command above always holds whatever command actually
    # verified the step; this preserves what it started as, so a human
    # reviewing session.json later can see a check moved, not just its
    # final state.
    original_acceptance_command: str | None = None
    # FLAW 2: a step's original acceptance_command is proven exactly once,
    # against exactly the input/case that existed when it was written --
    # nothing re-verifies it against a DIFFERENT case later (e.g. a
    # function that only got tested against positive numbers, then later
    # work in the run starts relying on it handling negatives too). This
    # lets strengthen_step_check append more commands after the fact,
    # without disturbing the original (which stays the historical record
    # of what actually passed when the step was committed). All of them
    # -- original plus every entry here -- are re-run by the regression
    # check, not just the original.
    additional_checks: list[str] = field(default_factory=list)


@dataclass
class StepContext:
    """Compact handoff from the exploratory/planner loop to the coder loop.

    This is intentionally ephemeral: it contains the actionable findings
    needed to execute one step, not the planner conversation that produced
    them. The coder should be able to start from this packet with a fresh
    context window and inspect only what it needs from the workspace.
    """
    title: str
    objective: str
    acceptance_command: str
    files: list[str] = field(default_factory=list)
    relevant_regions: list[str] = field(default_factory=list)
    findings: str = ""
    # Filled in once, mechanically, before the coder ever sees this step:
    # the harness runs acceptance_command against the untouched workspace
    # and records exactly what happened. Not interpreted or classified --
    # just the raw result, so the coder starts from the real baseline
    # instead of discovering it after several blind guesses.
    baseline_check_result: str = ""
    # Set once, in the outer loop, right after baseline_check_result --
    # before any revision can happen. See revise_acceptance_command.
    original_acceptance_command: str = ""
    acceptance_command_revisions: int = 0
    # Autonomous replan cycles used by this step so far (see Agent._replan_step).
    # A fail (max_subtask_attempts exhausted) checks this against
    # budget.max_replan_cycles before escalating to a human.
    replan_count: int = 0
    # revise_acceptance_command's cap is per-REPLAN-CYCLE, not per-step
    # lifetime: reset to 0 every time a replan happens, so an autonomous
    # replan that also needs to fix a provably-wrong check doesn't collide
    # with (or get blocked by) the step's original one-time allowance.
    revisions_this_cycle: int = 0

    def prompt_text(self) -> str:
        files = "\n".join(f"  - {p}" for p in self.files) or "  (none specified -- locate as needed)"
        regions = "\n".join(f"  - {r}" for r in self.relevant_regions) or "  (none specified -- locate as needed)"
        baseline = (
            f"\n\nBASELINE (what ACCEPTANCE COMMAND produced when run against the "
            f"CURRENT workspace, before you change anything -- this is your starting "
            f"point, not a hint about what's wrong):\n{self.baseline_check_result}"
            if self.baseline_check_result else ""
        )
        return (
            f"STEP: {self.title}\n\n"
            f"OBJECTIVE:\n{self.objective}\n\n"
            f"FILES TO FOCUS ON:\n{files}\n\n"
            f"RELEVANT REGIONS / SYMBOLS:\n{regions}\n\n"
            f"PLANNER FINDINGS:\n{self.findings or '(none -- inspect the workspace yourself)'}\n\n"
            f"ACCEPTANCE COMMAND:\n  {self.acceptance_command}"
            f"{baseline}"
        )


@dataclass
class RunState:
    goal: str
    completed_steps: list[CompletedStep] = field(default_factory=list)
    scratchpad: str = ""  # model-writable working notes; not verified by the harness
    # Harness-written, not model-written: one line appended automatically
    # every time read_file/search_code actually returns something in the
    # outer loop. Exists so file-reading progress is visible on disk
    # (session.json) regardless of whether the model ever calls
    # update_scratchpad -- it's a log, not a substitute for the model's own
    # notes, so it's kept in its own field rather than mixed into scratchpad.
    auto_read_log: str = ""
    # Harness-condensed understanding, built automatically as files are
    # read (see Agent._condense_batch). Keyed by filepath, NOT a single
    # growing string -- confirmed bug in the string version: any single
    # condensation pass could (and did) silently overwrite the ENTIRE
    # field, discarding every prior file's coverage the moment a later
    # pass didn't happen to re-mention it. A dict makes that structurally
    # impossible: a pass covering files [a, b] can only ever write keys
    # 'a' and 'b', regardless of what it returns -- every other file's
    # entry is untouched no matter what.
    condensed_files: dict[str, str] = field(default_factory=dict)
    status: Literal["running", "done", "aborted"] = "running"
    decomposition: GoalDecomposition = field(default_factory=GoalDecomposition)
    # FLAW 7: set the moment ask_human is called, cleared the moment an
    # answer comes back -- see ToolBox's on_question_asked/on_question_answered
    # hooks. Durable so a crash while blocked waiting on the human doesn't
    # lose the fact a question was ever pending.
    pending_question: str | None = None
    # Only bounds the rendered PROMPT text (see completed_summary_text) --
    # session.json's completed_steps list itself is never truncated, so the
    # full on-disk audit trail is untouched. Without this, a run of a few
    # hundred steps grows this section's token cost linearly forever, even
    # though each individual step's rendered line is already small.
    MAX_DETAILED_COMPLETED_STEPS = 30

    # ---- serialisation (for resumable sessions) ----

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "completed_steps": [asdict(s) for s in self.completed_steps],
            "scratchpad": self.scratchpad,
            "auto_read_log": self.auto_read_log,
            "condensed_files": self.condensed_files,
            "status": self.status,
            "decomposition": self.decomposition.to_dict(),
            "pending_question": self.pending_question,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunState":
        return cls(
            goal=d["goal"],
            completed_steps=[CompletedStep(**s) for s in d.get("completed_steps", [])],
            scratchpad=d.get("scratchpad", ""),
            auto_read_log=d.get("auto_read_log", ""),
            condensed_files=d.get("condensed_files", {}),
            status=d.get("status", "running"),
            decomposition=GoalDecomposition.from_dict(d.get("decomposition")),
            pending_question=d.get("pending_question"),
        )

    # ---- summary helpers for system prompts ----

    def completed_summary_text(self) -> str:
        if not self.completed_steps:
            return "(nothing completed yet)"
        cap = self.MAX_DETAILED_COMPLETED_STEPS
        detailed = self.completed_steps[-cap:] if len(self.completed_steps) > cap else self.completed_steps
        lines = []
        folded_count = len(self.completed_steps) - len(detailed)
        if folded_count > 0:
            first, last = self.completed_steps[0].index, detailed[0].index - 1
            lines.append(f"  [{first}-{last}] {folded_count} earlier steps completed (see session.json for full log)")
        for s in detailed:
            lines.append(f"  [{s.index}] {s.title}: {s.summary}")
        return "\n".join(lines)

    def next_index(self) -> int:
        return len(self.completed_steps) + 1
