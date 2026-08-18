from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Budget


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetTracker:
    budget: Budget
    started_at: float = field(default_factory=time.time)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    steps_this_attempt: int = 0        # inner loop: steps within one subtask attempt
    outer_steps_this_span: int = 0     # outer loop: steps since the last completed step or fresh human guidance
    paused_seconds: float = 0.0        # cumulative time spent blocked on human input -- excluded from elapsed_seconds()
    _pause_started_at: float | None = field(default=None, repr=False)

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if self.budget.max_total_tokens is not None:
            if self.total_input_tokens + self.total_output_tokens > self.budget.max_total_tokens:
                raise BudgetExceeded(
                    f"total token budget exceeded: "
                    f"{self.total_input_tokens + self.total_output_tokens} > {self.budget.max_total_tokens}"
                )

    def new_attempt(self) -> None:
        """Inner loop only: reset step count for a new subtask attempt."""
        self.steps_this_attempt = 0

    def step(self) -> None:
        """Inner loop only: count one tool-calling turn within the current subtask attempt."""
        self.steps_this_attempt += 1
        if self.steps_this_attempt > self.budget.max_steps_per_attempt:
            raise BudgetExceeded(
                f"step budget for this attempt exceeded: "
                f"{self.steps_this_attempt} > {self.budget.max_steps_per_attempt}"
            )

    def new_outer_span(self) -> None:
        """Outer loop only: reset step count. Called after a step completes
        or the human provides fresh guidance during escalation, so escalating
        actually grants a new budget instead of the very next outer_step()
        call immediately re-raising on an already-over-cap counter."""
        self.outer_steps_this_span = 0

    def outer_step(self) -> None:
        """Outer loop only: count one iteration of choosing/exploring toward
        the next step. Deliberately independent of steps_this_attempt --
        they used to share one counter that only the inner loop ever reset,
        which meant the outer loop could never recover once it hit budget."""
        self.outer_steps_this_span += 1
        if self.outer_steps_this_span > self.budget.max_steps_per_attempt:
            raise BudgetExceeded(
                f"outer-loop step budget exceeded: "
                f"{self.outer_steps_this_span} > {self.budget.max_steps_per_attempt}"
            )

    def pause(self) -> None:
        """Call right before any blocking input() (approval prompt,
        escalation question, done-confirmation) so the human's response time
        doesn't count as the agent 'running'. Idempotent -- a second pause()
        before a matching resume() is a no-op, not a leak."""
        if self._pause_started_at is None:
            self._pause_started_at = time.time()

    def resume(self) -> None:
        """Call right after the blocking input() returns."""
        if self._pause_started_at is not None:
            self.paused_seconds += time.time() - self._pause_started_at
            self._pause_started_at = None

    def elapsed_seconds(self) -> float:
        """Active time only -- total wall clock since the run started, minus
        any time spent paused waiting on a human response. A run that sat
        idle at an approval prompt for hours must not report that idle time
        as if the agent had been working the whole time."""
        raw = time.time() - self.started_at
        # If currently paused, don't count the in-progress pause either.
        in_progress_pause = (time.time() - self._pause_started_at) if self._pause_started_at is not None else 0.0
        return raw - self.paused_seconds - in_progress_pause

    def wall_clock_warning(self) -> str | None:
        elapsed = self.elapsed_seconds()
        if elapsed > self.budget.max_wall_clock_seconds:
            return (f"Run has been active for {elapsed/60:.1f} minutes "
                    f"(excluding time waiting on you), over the "
                    f"{self.budget.max_wall_clock_seconds/60:.0f}-minute soft cap.")
        return None

    def summary(self) -> str:
        return (f"tokens: {self.total_input_tokens} in / {self.total_output_tokens} out "
                f"| active: {self.elapsed_seconds():.0f}s "
                f"| waiting on you: {self.paused_seconds:.0f}s "
                f"| steps this attempt: {self.steps_this_attempt}")
