import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talk import _workspace_has_existing_content


def test_nonexistent_workspace_has_no_content(tmp_path):
    assert _workspace_has_existing_content(tmp_path / "does_not_exist") is False


def test_empty_workspace_has_no_content(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    assert _workspace_has_existing_content(ws) is False


def test_baseline_scaffolding_alone_is_not_existing_content(tmp_path):
    """.git, .autocoder, .gitignore, .gitkeep are created automatically by
    every workspace -- their presence alone must not trigger the warning."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".git").mkdir()
    (ws / ".autocoder").mkdir()
    (ws / ".gitignore").write_text(".autocoder/\n")
    (ws / ".gitkeep").write_text("")
    assert _workspace_has_existing_content(ws) is False


def test_real_leftover_file_counts_as_existing_content(tmp_path):
    """
    Regression: reusing a workspace from a previous, unrelated task used to
    silently build the new goal alongside the old task's files with no
    warning -- the model would explore, find leftover files, and sometimes
    build inside them instead of starting fresh.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".git").mkdir()
    (ws / "demo_project").mkdir()
    (ws / "demo_project" / "pyproject.toml").write_text("[build-system]\n")
    assert _workspace_has_existing_content(ws) is True
