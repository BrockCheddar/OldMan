import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from autocoder.workspace import Workspace
from autocoder.tools import ToolBox, ToolError
from autocoder.config import Config, ApprovalPolicy


def make_toolbox(tmp_path, approval_mode="smart", confirm_hook=None):
    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    cfg = Config(workspace_root=ws.root, approval=ApprovalPolicy(mode=approval_mode))
    return ToolBox(ws, cfg, on_command_needs_approval=confirm_hook), ws


def test_write_then_read_file(tmp_path):
    tb, ws = make_toolbox(tmp_path)
    tb.dispatch("write_file", {"path": "a.py", "content": "line1\nline2\nline3\n"})
    out = tb.dispatch("read_file", {"path": "a.py"})
    assert "line1" in out and "line3" in out
    assert "showing lines 1-3 of 3" in out


def test_read_file_paging(tmp_path):
    tb, ws = make_toolbox(tmp_path)
    content = "\n".join(f"line{i}" for i in range(1, 101))
    tb.dispatch("write_file", {"path": "big.txt", "content": content})
    out = tb.dispatch("read_file", {"path": "big.txt", "offset": 50, "limit": 5})
    assert "line50" in out and "line54" in out
    assert "line55" not in out
    assert "showing lines 50-54 of 100" in out


def test_edit_file_requires_unique_match(tmp_path):
    tb, ws = make_toolbox(tmp_path)
    tb.dispatch("write_file", {"path": "a.py", "content": "x = 1\nx = 1\n"})
    with pytest.raises(ToolError, match="matches 2 locations"):
        tb.dispatch("edit_file", {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"})


def test_edit_file_rejects_no_match(tmp_path):
    tb, ws = make_toolbox(tmp_path)
    tb.dispatch("write_file", {"path": "a.py", "content": "hello\n"})
    with pytest.raises(ToolError, match="not found"):
        tb.dispatch("edit_file", {"path": "a.py", "old_str": "goodbye", "new_str": "hi"})


def test_edit_file_applies_unique_match(tmp_path):
    tb, ws = make_toolbox(tmp_path)
    tb.dispatch("write_file", {"path": "a.py", "content": "def foo():\n    return 1\n"})
    tb.dispatch("edit_file", {"path": "a.py", "old_str": "return 1", "new_str": "return 2"})
    assert (ws.root / "a.py").read_text() == "def foo():\n    return 2\n"


def test_list_dir_ignores_git_internals(tmp_path):
    tb, ws = make_toolbox(tmp_path)
    tb.dispatch("write_file", {"path": "src/main.py", "content": "pass"})
    out = tb.dispatch("list_dir", {"path": "."})
    assert "src/main.py" in out.replace("\\", "/")
    assert ".git/" not in out.replace("\\", "/")


def test_search_code_fallback_finds_match(tmp_path):
    tb, ws = make_toolbox(tmp_path)
    tb.dispatch("write_file", {"path": "src/main.py", "content": "def target_function():\n    pass\n"})
    out = tb.dispatch("search_code", {"pattern": "target_function"})
    assert "main.py" in out.replace("\\", "/")


def test_write_and_read_file_with_non_ascii_content(tmp_path):
    """Regression test: write_file/read_file must always use UTF-8 explicitly.
    Without an explicit encoding, Python falls back to the OS locale encoding
    (cp1252 on most Windows installs), which crashes on ordinary characters
    like em dashes, curly quotes, or checkmarks that a model commonly writes
    into README/docstring content."""
    tb, ws = make_toolbox(tmp_path)
    content = "Title\n\nThis uses an em dash \u2014 curly quotes \u201cyes\u201d and a check \u2713.\n"
    tb.dispatch("write_file", {"path": "README.md", "content": content})
    out = tb.dispatch("read_file", {"path": "README.md"})
    assert "\u2014" in out and "\u2713" in out
    assert (ws.root / "README.md").read_text(encoding="utf-8") == content


def test_run_command_safe_prefix_skips_approval(tmp_path):
    calls = []
    tb, ws = make_toolbox(tmp_path, confirm_hook=lambda cmd, decision: calls.append(cmd) or True)
    tb.dispatch("run_command", {"command": "git status"})
    assert calls == []  # hook never invoked -- safe prefix, auto-approved


def test_run_command_unsafe_asks_for_approval_and_can_be_denied(tmp_path):
    tb, ws = make_toolbox(tmp_path, confirm_hook=lambda cmd, decision: False)
    out = tb.dispatch("run_command", {"command": "pip install requests"})
    assert "DENIED BY HUMAN" in out


def test_run_command_unsafe_approved_runs(tmp_path):
    import platform
    cmd = "echo hi-there" if platform.system() != "Windows" else "echo hi-there"
    tb, ws = make_toolbox(tmp_path, confirm_hook=lambda cmd, decision: True)
    out = tb.dispatch("run_command", {"command": cmd})
    assert "hi-there" in out


def test_path_traversal_blocked_via_tool(tmp_path):
    tb, ws = make_toolbox(tmp_path)
    with pytest.raises(ToolError):
        tb.dispatch("read_file", {"path": "../../../etc/passwd"})


def test_absolute_path_outside_workspace_is_tool_error_not_crash(tmp_path):
    """
    Regression: the model can hallucinate an absolute path (seen in practice:
    a Windows path from what looks like the model's own training data,
    completely unrelated to the real workspace). Workspace.resolve() rejects
    it correctly, but that used to be a WorkspaceError that no handler or
    dispatch() caught -- propagating as an unhandled crash instead of a
    normal, recoverable tool_result the model could see and retry from.

    Uses a POSIX-style absolute path so this is meaningful on every OS this
    test suite runs on -- pathlib only treats "C:\\..." as absolute on
    Windows itself, so that exact real-world string can't be used portably
    here, but the mechanism (resolve() + relative_to() rejecting anything
    outside the workspace root) is identical regardless of which OS's
    absolute-path syntax triggers it.
    """
    tb, ws = make_toolbox(tmp_path)
    outside_path = "/definitely/outside/the/workspace/file.py"
    with pytest.raises(ToolError, match="use a path relative to the workspace root"):
        tb.dispatch("read_file", {"path": outside_path})
    # every handler that resolves a path must be protected the same way
    with pytest.raises(ToolError):
        tb.dispatch("write_file", {"path": outside_path, "content": "x"})
    with pytest.raises(ToolError):
        tb.dispatch("list_dir", {"path": outside_path})


def test_dispatch_catches_stray_workspace_error_as_fallback(tmp_path):
    """The dispatch()-level fallback: even if some handler ever forgets to
    route through _resolve(), a WorkspaceError must still surface as a
    ToolError, never propagate raw."""
    tb, ws = make_toolbox(tmp_path)
    with pytest.raises(ToolError):
        tb.dispatch("edit_file", {
            "path": "/definitely/outside/workspace.py",
            "old_str": "x", "new_str": "y",
        })


def test_tool_schemas_contains_no_dead_done_tool():
    """
    Regression: TOOL_SCHEMAS once still contained a leftover 'mark_subtask_done'
    tool from before the incremental-loop rewrite, alongside the real
    'mark_step_done'. Both were offered to the model with nearly identical
    descriptions, no handler existed for the old one, and the model kept
    calling it and getting silent 'unknown tool' errors in a loop. Every
    tool name in TOOL_SCHEMAS must have either a real ToolBox handler or be
    one of the specially-intercepted names -- nothing offered to the model
    should be a dead end.
    """
    from autocoder.tools import TOOL_SCHEMAS, ToolBox
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert "mark_subtask_done" not in names
    for name in names:
        assert hasattr(ToolBox, f"_tool_{name}"), (
            f"TOOL_SCHEMAS offers '{name}' but ToolBox has no _tool_{name} handler for it"
        )


def test_outer_and_inner_tool_lists_have_exactly_one_done_tool_each():
    """
    Regression: the outer loop must expose exactly one way to finish a step
    (declare_done) and the inner loop exactly one way to finish a step
    (mark_step_done) -- never both a real and a dead/duplicate 'done' tool
    with a name similar enough to confuse a small local model.
    """
    from autocoder.agent import OUTER_TOOLS, INNER_TOOLS
    outer_names = {t["name"] for t in OUTER_TOOLS}
    inner_names = {t["name"] for t in INNER_TOOLS}

    assert "mark_subtask_done" not in outer_names
    assert "mark_subtask_done" not in inner_names
    assert "declare_done" in outer_names
    assert "mark_step_done" not in outer_names  # only valid inside an active step
    assert "mark_step_done" in inner_names
    assert "declare_done" not in inner_names  # only valid at the outer level


def test_outer_tools_exclude_write_file_and_edit_file():
    """
    Regression: the outer loop used to have full write_file/edit_file access
    with zero acceptance-check gate. Seen in practice: after a step passed
    its check and was committed, the model used write_file again from the
    outer loop to silently overwrite that already-verified file, with
    nothing re-checking it. Any real file mutation must go through
    propose_step into the verified inner loop.
    """
    from autocoder.agent import OUTER_TOOLS, INNER_TOOLS
    outer_names = {t["name"] for t in OUTER_TOOLS}
    inner_names = {t["name"] for t in INNER_TOOLS}

    assert "write_file" not in outer_names
    assert "edit_file" not in outer_names
    # inner loop must still have full access -- mark_step_done re-verifies
    assert "write_file" in inner_names
    assert "edit_file" in inner_names
    # read-only exploration must still be available at the outer level
    assert "read_file" in outer_names
    assert "list_dir" in outer_names
    assert "search_code" in outer_names


