import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from autocoder.workspace import Workspace, WorkspaceError


def test_create_fresh_workspace_has_initial_commit(tmp_path):
    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    log = ws.git_log()
    assert "initial empty workspace" in log.stdout


def test_clone_from_existing_git_repo(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    Workspace.create(source, source_repo=None)  # gives it a .git + initial commit
    (source / "hello.txt").write_text("hi")

    ws = Workspace(source)
    ws.git_commit_all("add hello.txt")

    cloned = Workspace.create(tmp_path / "clone", source_repo=source)
    assert (cloned.root / "hello.txt").exists()


def test_copies_non_git_folder_and_inits_git(tmp_path):
    source = tmp_path / "plain_folder"
    source.mkdir()
    (source / "a.txt").write_text("data")

    ws = Workspace.create(tmp_path / "ws", source_repo=source)
    assert (ws.root / "a.txt").read_text() == "data"
    assert "initial import" in ws.git_log().stdout


def test_resolve_blocks_path_traversal(tmp_path):
    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    with pytest.raises(WorkspaceError):
        ws.resolve("../../etc/passwd")


def test_resolve_allows_nested_path(tmp_path):
    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    p = ws.resolve("sub/dir/file.txt")
    assert p.parent.name == "dir"


def test_run_command_executes_in_workspace_cwd(tmp_path):
    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    (ws.root / "marker.txt").write_text("present")
    import platform
    cmd = "dir" if platform.system() == "Windows" else "ls"
    result = ws.run_command(cmd, timeout=10)
    assert "marker.txt" in result.stdout


def test_git_commit_and_revert(tmp_path):
    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    (ws.root / "f.txt").write_text("v1")
    ws.git_commit_all("v1")

    (ws.root / "f.txt").write_text("v2 - uncommitted")
    assert ws.git_status().stdout.strip() != ""

    ws.git_revert_to_last_commit()
    assert (ws.root / "f.txt").read_text() == "v1"


def test_run_command_timeout(tmp_path):
    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    import platform
    if platform.system() == "Windows":
        cmd = "ping -n 5 127.0.0.1 > nul"
    else:
        cmd = "sleep 5"
    result = ws.run_command(cmd, timeout=1)
    assert result.timed_out


def test_run_command_os_error_does_not_crash(tmp_path, monkeypatch):
    """
    Regression: a command line too long for Windows to launch raised a bare
    FileNotFoundError (WinError 206) that nothing caught, crashing the
    whole process with an unhandled traceback -- same shape as two earlier
    bugs (an exception type the harness didn't expect escaping its
    containment). Must come back as a normal failed CommandResult instead.
    """
    from autocoder import workspace as workspace_module

    ws = Workspace.create(tmp_path / "ws", source_repo=None)  # real git init, before the patch

    def _raise_oserror(*args, **kwargs):
        raise FileNotFoundError("[WinError 206] The filename or extension is too long")

    monkeypatch.setattr(workspace_module.subprocess, "Popen", _raise_oserror)
    result = ws.run_command("some command", timeout=10)

    assert result.exit_code != 0
    assert not result.timed_out
    assert "FileNotFoundError" in result.stderr
    assert "too long" in result.stderr


def test_run_command_does_not_inherit_stdin(tmp_path, monkeypatch):
    """
    Regression: a command that reads from stdin without us explicitly
    redirecting it inherits our own stdin and can block forever waiting
    for input that will never come -- confirmed in practice, and
    indistinguishable from a slow command until you check whether it's
    actually using any CPU (it won't be).
    """
    from autocoder import workspace as workspace_module

    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    captured = {}
    real_popen = workspace_module.subprocess.Popen

    class _Recording(real_popen):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(workspace_module.subprocess, "Popen", _Recording)
    ws.run_command("echo hi", timeout=10)
    assert captured.get("stdin") is workspace_module.subprocess.DEVNULL


def test_kill_process_tree_uses_taskkill_on_windows(monkeypatch):
    from autocoder import workspace as workspace_module

    monkeypatch.setattr(workspace_module.os, "name", "nt")
    calls = []
    monkeypatch.setattr(
        workspace_module.subprocess, "run",
        lambda *a, **kw: calls.append(a[0]),
    )

    class _FakeProc:
        pid = 4242
        def wait(self, timeout=None):
            return 0

    workspace_module._kill_process_tree(_FakeProc())
    assert calls == [["taskkill", "/T", "/F", "/PID", "4242"]]


def test_kill_process_tree_uses_process_group_on_posix(monkeypatch):
    from autocoder import workspace as workspace_module

    monkeypatch.setattr(workspace_module.os, "name", "posix")
    killed = {}
    monkeypatch.setattr(workspace_module.os, "getpgid", lambda pid: 999)
    monkeypatch.setattr(workspace_module.os, "killpg", lambda pgid, sig: killed.update(pgid=pgid, sig=sig))

    class _FakeProc:
        pid = 4242
        def wait(self, timeout=None):
            return 0

    workspace_module._kill_process_tree(_FakeProc())
    assert killed == {"pgid": 999, "sig": workspace_module.signal.SIGKILL}


def test_run_command_kills_full_process_tree_on_timeout(tmp_path):
    """
    Real regression test (not mocked): the old code's proc.kill() only
    reached the immediate shell child. A detached grandchild that outlives
    it and keeps holding the inherited stdout pipe open used to hang
    communicate() forever, past whatever timeout was configured -- because
    the pipe can't hit EOF while anything still holds it open. This must
    return promptly regardless.
    """
    import platform
    if platform.system() == "Windows":
        import pytest as _pytest
        _pytest.skip("POSIX-specific detached-grandchild reproduction")

    ws = Workspace.create(tmp_path / "ws", source_repo=None)
    # backgrounds a grandchild that outlives the shell that spawned it,
    # inheriting the same stdout pipe -- the exact shape of the bug.
    cmd = "(sleep 30 &) ; sleep 30"
    start = time.time()
    result = ws.run_command(cmd, timeout=1)
    elapsed = time.time() - start

    assert result.timed_out
    assert elapsed < 10  # old code would hang for the full 30s+
