import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.osinfo import shell_description


def test_windows_guidance_warns_about_dev_null(monkeypatch):
    """Real-world regression: the model wrote an acceptance_command using
    `2>/dev/null` (a POSIX-only path) on Windows, which errors on cmd.exe
    instead of silently discarding output -- causing an otherwise-correct
    check to fail for a reason unrelated to the actual code, wasting an
    attempt (and, with the retry-reverts-to-last-commit behavior, actually
    discarding the attempt's real work). The guidance must call this out
    explicitly, the same way it already does for `test`/`[` and `-p`."""
    monkeypatch.setattr("platform.system", lambda: "Windows")
    desc = shell_description()
    assert "/dev/null" in desc
    assert "2>nul" in desc


def test_windows_guidance_still_covers_existing_gotchas(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    desc = shell_description()
    assert "-p" in desc
    assert "test" in desc and "cmd.exe" in desc


def test_posix_guidance_unaffected(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    desc = shell_description()
    assert "/dev/null" not in desc
    assert "POSIX" in desc
