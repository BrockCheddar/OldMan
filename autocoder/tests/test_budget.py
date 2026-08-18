import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.budget import BudgetTracker
from autocoder.config import Budget


class FakeClock:
    """Deterministic, monkeypatchable stand-in for time.time()."""
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_elapsed_seconds_excludes_paused_time(monkeypatch):
    """
    Regression: a run that sat idle at an approval/escalation prompt for a
    long stretch was reporting that idle time as if the agent had been
    actively running the whole time (628 minutes reported for a run that
    was actually mostly sitting untouched).
    """
    clock = FakeClock()
    monkeypatch.setattr("autocoder.budget.time.time", clock)

    tracker = BudgetTracker(budget=Budget(), started_at=clock.now)
    clock.advance(10)          # 10s of real agent activity
    tracker.pause()
    clock.advance(3600)        # 1 hour sitting idle at a prompt
    tracker.resume()
    clock.advance(5)           # 5s more activity

    assert tracker.elapsed_seconds() == 15  # only the active time, not the hour paused


def test_elapsed_seconds_excludes_in_progress_pause(monkeypatch):
    """If still paused (never resumed), the in-progress pause must also be
    excluded, not just completed pause/resume pairs."""
    clock = FakeClock()
    monkeypatch.setattr("autocoder.budget.time.time", clock)

    tracker = BudgetTracker(budget=Budget(), started_at=clock.now)
    clock.advance(10)
    tracker.pause()
    clock.advance(9999)  # still paused, hasn't called resume() yet

    assert tracker.elapsed_seconds() == 10


def test_pause_is_idempotent():
    """Calling pause() twice without a resume() in between must not double-count."""
    tracker = BudgetTracker(budget=Budget())
    tracker.pause()
    tracker.pause()  # should be a no-op, not overwrite the original pause start
    assert tracker._pause_started_at is not None


def test_resume_without_pause_is_a_noop():
    tracker = BudgetTracker(budget=Budget())
    tracker.resume()  # must not raise
    assert tracker.paused_seconds == 0.0


def test_wall_clock_warning_not_triggered_by_idle_time(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("autocoder.budget.time.time", clock)

    tracker = BudgetTracker(budget=Budget(max_wall_clock_seconds=60), started_at=clock.now)
    clock.advance(5)            # 5s active
    tracker.pause()
    clock.advance(10000)        # long idle stretch, well over the 60s cap
    tracker.resume()

    assert tracker.wall_clock_warning() is None  # only 5s of real active time


def test_wall_clock_warning_still_fires_for_genuine_active_overrun(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("autocoder.budget.time.time", clock)

    tracker = BudgetTracker(budget=Budget(max_wall_clock_seconds=60), started_at=clock.now)
    clock.advance(120)  # genuinely 120s of active time, no pauses at all

    warning = tracker.wall_clock_warning()
    assert warning is not None
    assert "active for" in warning
    assert "waiting on you" in warning
