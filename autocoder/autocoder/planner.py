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
from dataclasses import dataclass, field, asdict
from typing import Literal


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

    # ---- serialisation (for resumable sessions) ----

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "completed_steps": [asdict(s) for s in self.completed_steps],
            "scratchpad": self.scratchpad,
            "auto_read_log": self.auto_read_log,
            "condensed_files": self.condensed_files,
            "status": self.status,
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
        )

    # ---- summary helpers for system prompts ----

    def completed_summary_text(self) -> str:
        if not self.completed_steps:
            return "(nothing completed yet)"
        lines = []
        for s in self.completed_steps:
            lines.append(f"  [{s.index}] {s.title}: {s.summary}")
        return "\n".join(lines)

    def next_index(self) -> int:
        return len(self.completed_steps) + 1
