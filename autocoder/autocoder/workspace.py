"""
Workspace = the real, on-disk, git-backed directory a task executes in.

No Docker/microVM here (deferred by design decision -- see README "Isolation
model"). This is still a genuine improvement over an in-memory VFS: files
persist across restarts, a real shell runs real toolchains, and git gives
unlimited, inspectable undo instead of a 10-entry patch history.

Isolation that DOES exist:
  - The agent only ever runs commands with cwd fixed to the workspace dir.
  - Every accepted change is a real commit, so any state is recoverable.
Isolation that does NOT exist (be aware):
  - A malicious or badly-instructed command can still touch anything your
    Windows user account can touch (network, other files, etc). There is no
    OS-level sandbox. The approval gate (approval.py) is the main defense
    until a container backend is added.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class WorkspaceError(RuntimeError):
    pass


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """
    Kill the whole process tree, not just the immediate child.

    proc.kill() alone only reaches the process we spawned directly. With
    shell=True on Windows that's cmd.exe -- the real command (e.g. `python
    script.py`) runs as cmd.exe's own child, a grandchild of ours, and is
    never touched by proc.kill(). Confirmed in practice: that grandchild
    was left running, orphaned, inheriting the same stdout/stderr pipes we
    were reading from -- so even after cmd.exe died, communicate() kept
    blocking forever waiting for a pipe EOF that couldn't happen until the
    orphan also exited. The timeout= we passed in couldn't rescue that; it
    isn't a logic bug on our side, it's what shell=True does on Windows.

    On POSIX the child was launched with start_new_session=True (see
    _run), which puts it and everything it spawns in its own process
    group -- os.killpg reaches all of them, including anything it
    backgrounded or detached.
    """
    if os.name == "nt":
        # taskkill /T recurses into the whole tree; plain proc.kill() does not.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True, timeout=10,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run(cmd: list[str] | str, cwd: Path, timeout: int, shell: bool) -> CommandResult:
    start = time.time()
    command_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    popen_kwargs: dict = dict(
        cwd=str(cwd),
        shell=shell,
        # Never let a spawned command block on stdin. Nothing running
        # headless under this harness has a legitimate reason to want
        # interactive input, and a command that reads from stdin without
        # us setting this inherits our own stdin and can hang forever
        # waiting for input that will never come -- indistinguishable
        # from a genuinely slow command until you check whether it's
        # actually using any CPU (it won't be).
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # So _kill_process_tree can reach every descendant, not just the
    # process we spawn directly -- see its docstring.
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except OSError as e:
        return CommandResult(
            command=command_str,
            exit_code=-1,
            stdout="",
            stderr=(
                f"{type(e).__name__}: {e}\n"
                "This usually means the command line itself could not be launched by the "
                "OS -- often because it's too long (Windows has a hard command-line length "
                "limit) or references something that doesn't exist. Try a shorter command, "
                "or write a script to a file and run that file instead of one long inline command."
            ),
            duration_s=time.time() - start,
        )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return CommandResult(
            command=command_str,
            exit_code=proc.returncode,
            stdout=(stdout or "")[-20000:],  # cap so one runaway command can't blow the context
            stderr=(stderr or "")[-20000:],
            duration_s=time.time() - start,
        )
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        # Now that every process holding the pipes is actually dead, this
        # drains whatever partial output exists and returns promptly --
        # it does not re-block the way the first call did.
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return CommandResult(
            command=command_str,
            exit_code=-1,
            stdout=(stdout or "")[-20000:],
            stderr=(stderr or "")[-20000:],
            duration_s=time.time() - start,
            timed_out=True,
        )
    except OSError as e:
        # Same shape as two earlier bugs: an exception type this code
        # didn't expect, escaping past its own try/except and crashing the
        # whole process. Confirmed in practice: a very long `python -c
        # "..."` command exceeded Windows' command-line length limit and
        # raised a bare FileNotFoundError (WinError 206) that nothing here
        # caught. OSError is the base class for that and other real launch
        # failures (permission errors, missing interpreter, etc.) -- report
        # them as a normal failed command instead of an unhandled crash.
        _kill_process_tree(proc)
        return CommandResult(
            command=command_str,
            exit_code=-1,
            stdout="",
            stderr=(
                f"{type(e).__name__}: {e}\n"
                "This usually means the command line itself could not be launched by the "
                "OS -- often because it's too long (Windows has a hard command-line length "
                "limit) or references something that doesn't exist. Try a shorter command, "
                "or write a script to a file and run that file instead of one long inline command."
            ),
            duration_s=time.time() - start,
        )


class Workspace:
    # Directories the harness manages itself and the model must never write
    # into directly: .git/ because a hook planted here (e.g.
    # .git/hooks/post-commit) is auto-executing code the harness triggers
    # itself on every `git commit` -- a clean bypass of the approval gate,
    # since nothing re-checks a hook before it runs. .autocoder/ because
    # it's the agent's own resumable session state (goal, completed-step
    # log); letting the model rewrite it defeats its purpose as a record
    # of what actually happened.
    _PROTECTED_WRITE_DIRS = (".git", ".autocoder")

    def __init__(self, root: Path):
        self.root = root

    # ---------- setup ----------

    @classmethod
    def create(cls, workspace_root: Path, source_repo: Path | None) -> "Workspace":
        workspace_root.mkdir(parents=True, exist_ok=True)
        ws = cls(workspace_root)
        if source_repo is not None:
            ws._clone_from(source_repo)
        else:
            ws._init_fresh()
        (workspace_root / ".autocoder").mkdir(exist_ok=True)
        return ws

    def _clone_from(self, source_repo: Path) -> None:
        source_repo = source_repo.resolve()
        if not source_repo.exists():
            raise WorkspaceError(f"source repo does not exist: {source_repo}")
        is_git = (source_repo / ".git").exists()
        if is_git:
            # Real, local clone -- gives us an independent working tree +
            # full history, and pushing back to the original is just a
            # normal `git push` from inside the workspace later.
            result = _run(["git", "clone", "--local", str(source_repo), str(self.root)],
                           cwd=source_repo.parent, timeout=120, shell=False)
            if result.exit_code != 0:
                raise WorkspaceError(f"git clone failed: {result.stderr}")
            self._git_config_identity()
            if self._ensure_gitignore():
                self._git(["add", ".gitignore"])
                self._git(["commit", "-m", "autocoder: ignore .autocoder/ session dir"])
        else:
            # Not a git repo yet -- copy files, then initialize one so the
            # rest of the harness (commits, diffs, undo) works uniformly.
            shutil.copytree(source_repo, self.root, dirs_exist_ok=True)
            self._git_init()
            self._ensure_gitignore()
            self._git(["add", "-A"])
            self._git(["commit", "-m", "autocoder: initial import of existing project"])

    def _init_fresh(self) -> None:
        self._git_init()
        (self.root / ".gitkeep").write_text("", encoding="utf-8")
        self._ensure_gitignore()
        self._git(["add", "-A"])
        self._git(["commit", "-m", "autocoder: initial empty workspace"])

    def _ensure_gitignore(self) -> bool:
        """Makes sure .autocoder/ (session state, logs) is gitignored, so
        `git clean -fd` (the undo/discard primitive) can never delete the
        agent's own session state along with a failed attempt's junk files.
        Returns True if the file was created/modified."""
        gitignore = self.root / ".gitignore"
        entry = ".autocoder/"
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        if entry in existing.splitlines():
            return False
        with open(gitignore, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(entry + "\n")
        return True

    def _git_init(self) -> None:
        result = _run(["git", "init"], cwd=self.root, timeout=30, shell=False)
        if result.exit_code != 0:
            raise WorkspaceError(f"git init failed: {result.stderr}")
        self._git_config_identity()

    def _git_config_identity(self) -> None:
        # Local-only identity so commits work even with no global git config
        # (a fresh clone doesn't inherit the source repo's identity either).
        _run(["git", "config", "user.email", "autocoder@local"], cwd=self.root, timeout=10, shell=False)
        _run(["git", "config", "user.name", "autocoder"], cwd=self.root, timeout=10, shell=False)

    # ---------- git helpers (Phase 5: real VCS instead of a custom undo stack) ----------

    def _git(self, args: list[str], timeout: int = 30) -> CommandResult:
        return _run(["git"] + args, cwd=self.root, timeout=timeout, shell=False)

    def git_status(self) -> CommandResult:
        return self._git(["status", "--porcelain=v1"])

    def git_diff(self, staged: bool = False) -> CommandResult:
        args = ["diff"] + (["--cached"] if staged else [])
        return self._git(args)

    def git_log(self, n: int = 10) -> CommandResult:
        return self._git(["log", f"-{n}", "--oneline"])

    def git_commit_all(self, message: str) -> CommandResult:
        self._git(["add", "-A"])
        return self._git(["commit", "-m", message, "--allow-empty-message", "--allow-empty"])

    def git_revert_to_last_commit(self) -> CommandResult:
        """Discard all uncommitted changes -- the 'undo' primitive."""
        self._git(["reset", "--hard", "HEAD"])
        return self._git(["clean", "-fd"])

    def git_checkpoint_tag(self, name: str) -> CommandResult:
        return self._git(["tag", "-f", name])

    # ---------- filesystem ----------

    def resolve(self, relative_path: str) -> Path:
        """Resolve a path the model gave us, refusing to escape the workspace."""
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            raise WorkspaceError(
                f"path '{relative_path}' resolves outside the workspace ({self.root}); refused"
            )
        return candidate

    def resolve_for_write(self, relative_path: str) -> Path:
        """Like resolve(), but additionally refuses to write inside a
        protected directory (see _PROTECTED_WRITE_DIRS). Read access
        through resolve() is unaffected -- this is specifically for the
        tools that create or modify files."""
        candidate = self.resolve(relative_path)
        rel_parts = candidate.relative_to(self.root.resolve()).parts
        if rel_parts and rel_parts[0] in self._PROTECTED_WRITE_DIRS:
            raise WorkspaceError(
                f"path '{relative_path}' is inside the protected '{rel_parts[0]}/' "
                "directory and cannot be written to"
            )
        return candidate

    # ---------- command execution ----------

    def run_command(self, command: str, timeout: int = 120) -> CommandResult:
        """
        Runs on the real OS shell, cwd pinned to the workspace root.
        NOTE: on Windows this runs via the default shell (cmd.exe/PowerShell
        depending on how Python was installed); on the pytest/CI runs used
        while developing this project it's a POSIX shell. Either way,
        `shell=True` is required for pipes/redirects the model may write.
        """
        return _run(command, cwd=self.root, timeout=timeout, shell=True)
