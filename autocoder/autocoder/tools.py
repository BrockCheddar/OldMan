"""
Tool surface for the agent, using Anthropic's native tool-calling format
(input_schema per tool) instead of asking the model to emit a bespoke JSON
blob that then gets brace-matched out of raw text. Every tool result is a
structured dict; the calling loop turns malformed calls into a normal
tool_result error the model can see and retry, rather than falling through
to regex guessing.
"""
from __future__ import annotations

import fnmatch
import os
import platform
import re
from pathlib import Path
from typing import Any, Callable

from .approval import (
    ApprovalDecision, classify_command, confirm_with_human, ask_human_question,
    detect_workspace_escape, split_command_segments,
)
from .config import Config
from .decisions import DecisionsStore
from .workspace import CommandResult, Workspace, WorkspaceError

# Real Windows/POSIX incompatibilities confirmed in practice, not just
# theorized -- each of these has been observed actually breaking a run:
#   - leading POSIX-only tool names: acceptance_command using `test -f` on
#     cmd.exe (silently destroyed generated work via git revert before the
#     git-stash fix existed).
#   - embedded literal newlines: a multi-line `python -c "\n...` command
#     passed through `cmd /c "..."` -- cmd.exe's parsing of multi-line /c
#     arguments is fragile and commonly mis-splits or mis-quotes, causing
#     the command to silently do nothing or something unintended.
#   - heredoc (`<<`): has no meaning in cmd.exe at all; `cat > x << 'EOF'`
#     fails outright (confirmed: the target file was never even created).
_WINDOWS_INCOMPATIBLE_LEADING_TOKENS = {
    "test", "[", "rm", "ls", "cat", "grep", "find", "touch", "mkdir",
    "which", "head", "tail", "chmod", "sh", "bash",
}


def validate_command_for_os(command: str) -> str | None:
    """Best-effort: flag a shell command (whether it's an acceptance check
    or an ordinary run_command call) that will fail every time on this
    machine's actual shell, before it ever runs. Narrow on purpose --
    false negatives just mean the command runs and fails normally like any
    other failed command; false positives would block valid work. Applies
    equally to run_command tool calls and acceptance_command execution,
    since both go through the same shell and the same failure modes were
    observed in both places."""
    if platform.system() != "Windows":
        return None
    if "\n" in command:
        return (
            "command contains an embedded newline -- multi-line commands run through "
            "Windows cmd.exe (`cmd /c \"...\"`) are unreliable and commonly get mis-parsed "
            "or silently do nothing, especially combined with quotes. Keep it to a single "
            "line, or write a script file first (e.g. via write_file) and run it with a "
            "short single-line command like `python script.py`."
        )
    if "<<" in command:
        return (
            "command uses a heredoc (<<), which has no equivalent in Windows cmd.exe and "
            "will fail outright. Write the content with write_file instead, or use a "
            "single-line `python -c \"...\"` command."
        )
    segments = re.split(r"&&|\|\||;", command)
    for segment in segments:
        tokens = segment.strip().split()
        if not tokens:
            continue
        head = tokens[0]
        if head in _WINDOWS_INCOMPATIBLE_LEADING_TOKENS:
            return (
                f"command uses `{head}`, which doesn't exist on this machine's shell "
                "(Windows cmd.exe) and would fail every time. Use a `python -c \"...\"` "
                "one-liner instead, or write_file for creating/editing files."
            )
        if "/dev/null" in segment:
            return (
                "command redirects to /dev/null, which doesn't exist on Windows and will "
                "error instead of silencing output. Use `2>nul` or drop the redirect."
            )
    return None


# Leading verbs (per segment, after splitting on &&/||/;/|) that mutate the
# filesystem in some way -- creating, deleting, moving, copying, or
# installing something. Covers both POSIX and Windows spellings since a
# model may reach for either regardless of which shell actually runs it.
_MUTATING_LEADING_TOKENS = {
    "rm", "del", "erase", "rd", "rmdir", "md", "mkdir",
    "mv", "move", "cp", "copy", "ren", "rename", "xcopy", "robocopy",
    "touch", "tee",
}
# Package managers are only mutating for specific subcommands -- "pip
# show"/"npm list"/"go vet" are legitimate read-only investigation and
# must stay allowed; "pip install"/"npm add"/"go get" actually change the
# workspace (installed packages, lockfiles, go.sum) and shouldn't.
_MUTATING_PACKAGE_MANAGER_SUBCOMMANDS = {
    "pip": {"install", "uninstall"},
    "pip3": {"install", "uninstall"},
    "npm": {"install", "i", "uninstall", "remove", "rm", "add", "ci", "update"},
    "yarn": {"add", "remove", "install", "upgrade"},
    "cargo": {"add", "remove", "install", "update"},
    "go": {"get", "install", "mod"},
}
# Same idea for git subcommands specifically -- "git status"/"git diff"/
# "git log" are read-only and fine; these change tracked or working-tree
# state.
_MUTATING_GIT_SUBCOMMANDS = {
    "add", "commit", "checkout", "restore", "apply", "merge", "rebase",
    "reset", "stash", "clean", "rm", "mv",
}
# In-place edit flags, checked as a substring since they can appear
# anywhere in a segment's arguments, not just as the leading token.
_IN_PLACE_EDIT_MARKERS = ("sed -i", "perl -i")
# Best-effort textual signals that an embedded script (most commonly a
# `python -c "..."` one-liner, but this deliberately isn't limited to that
# shape) writes to the filesystem. Checked against the whole command, not
# per segment, since these appear inside a quoted argument.
_SCRIPTED_WRITE_MARKERS = (
    "write_text(", "write_bytes(", ".write(", "os.remove(", "os.unlink(",
    "os.rename(", "os.replace(", "os.makedirs(", "os.mkdir(",
    "shutil.copy", "shutil.move", "shutil.rmtree",
)
_OPEN_WRITE_MODE_RE = re.compile(r"""open\([^)]*['"][wax]\+?b?['"]""")


def looks_like_mutating_command(command: str) -> str | None:
    """Best-effort: flag a run_command invocation that appears to create,
    modify, move, or delete something in the workspace (or install a
    dependency into it), for the outer loop specifically to reject the
    same way it already rejects write_file/edit_file -- see
    _OUTER_SAFE_TOOL_NAMES in agent.py. The inner loop does NOT use this:
    a step's mutations are already scoped by the revert-on-fail/commit-on-
    accept cycle around mark_step_done, so run_command there doesn't need
    a separate mutation ban the way the outer loop (which has no such
    cycle at all) does.

    Same philosophy as validate_command_for_os: a text pattern match on an
    arbitrary shell/script string, not a real parser. False negatives just
    mean an unverified mutation slips through uncaught, same as before this
    existed; false positives would block legitimate read-only exploration,
    which is the worse failure to bias against here.
    """
    if _OPEN_WRITE_MODE_RE.search(command):
        return "opens a file in a write/append mode"
    lowered = command.lower()
    for marker in _SCRIPTED_WRITE_MARKERS:
        if marker in lowered:
            return f"contains '{marker.rstrip('(')}', which writes to the filesystem"
    for marker in _IN_PLACE_EDIT_MARKERS:
        if marker in lowered:
            return f"uses '{marker}' (in-place file edit)"
    segments = split_command_segments(command)
    if segments is None:
        return None  # unbalanced quotes: validate_command_for_os already flags this separately
    for segment in segments:
        tokens = segment.strip().split()
        if not tokens:
            continue
        head = tokens[0].lower()
        if head in _MUTATING_LEADING_TOKENS:
            return f"uses `{head}`, which mutates the filesystem"
        pkg_subcommands = _MUTATING_PACKAGE_MANAGER_SUBCOMMANDS.get(head)
        if pkg_subcommands and len(tokens) > 1 and tokens[1].lower() in pkg_subcommands:
            return f"uses `{head} {tokens[1].lower()}`, which installs/modifies dependencies"
        if head == "git" and len(tokens) > 1 and tokens[1].lower() in _MUTATING_GIT_SUBCOMMANDS:
            return f"uses `git {tokens[1].lower()}`, which changes repo state"
        # A redirect into a real file (not a discard target) writes to the
        # workspace regardless of which command precedes it.
        redirect = re.search(r"(?<![0-9&])(>{1,2})(?!&)\s*(\S+)", segment)
        if redirect and redirect.group(2).lower() not in ("nul", "/dev/null"):
            return f"redirects output to '{redirect.group(2)}'"
    return None

IGNORE_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv",
                    ".autocoder", "dist", "build", ".mypy_cache", ".pytest_cache"}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a text file from the workspace, with real line numbers. "
            "For large files, use offset/limit to page through it across "
            "multiple calls -- there is no penalty for reading a large file "
            "in several chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "offset": {"type": "integer", "description": "1-indexed line number to start at.", "default": 1},
                "limit": {"type": "integer", "description": "Max lines to return.", "default": 2000},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a NEW file, or COMPLETELY REPLACE an existing one's entire contents. "
            "If the file already exists, read it first: write_file does not merge or append -- "
            "anything not included in your new content is gone, including code you were told to "
            "keep as-is. Use edit_file instead for any change to an existing file where you're "
            "adding to or modifying part of it rather than intentionally replacing the whole thing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Find-and-replace inside an existing file. old_str must match the file's "
            "current content EXACTLY and appear exactly once -- if it matches zero or "
            "multiple times, this fails on purpose (rather than guessing) and tells you "
            "why, so include enough surrounding context to make old_str unique."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories under a path (recursive, ignores .git/node_modules/etc).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "max_entries": {"type": "integer", "default": 300},
            },
            "required": [],
        },
    },
    {
        "name": "search_code",
        "description": "Search file contents for a regex pattern across the workspace (like grep -rn).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "max_results": {"type": "integer", "default": 100},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a real shell command with cwd pinned to the workspace root (e.g. "
            "installing a package, running a build, running tests). Commands outside "
            "a small safe list (git status/diff/log, test runners, linters, read-only "
            "listing) will pause and ask a human to approve before running."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    },
    {
        "name": "git_diff",
        "description": "Show the current uncommitted diff (working tree vs last commit).",
        "input_schema": {"type": "object", "properties": {"staged": {"type": "boolean", "default": False}}},
    },
    {
        "name": "git_log",
        "description": "Show recent commit history.",
        "input_schema": {"type": "object", "properties": {"n": {"type": "integer", "default": 10}}},
    },
    {
        "name": "record_decision",
        "description": (
            "Record a durable project decision -- an architecture, technology, or "
            "convention choice a LATER step needs to stay consistent with (e.g. "
            "picking a library, a data format, a naming convention, or deliberately "
            "deferring a feature). Once recorded, it appears automatically in every "
            "future prompt in both loops -- you never need to restate it yourself. "
            "To change an earlier decision, pass its id (shown in brackets in the "
            "'ACTIVE DECISIONS' section of this prompt) as supersedes; it stops "
            "appearing as active, replaced by this new one. Do not call this for "
            "routine implementation detail -- only for choices a future step could "
            "otherwise contradict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "description": "The decision itself, stated plainly, e.g. 'Use SQLAlchemy ORM, not raw sqlite3'.",
                },
                "supersedes": {
                    "type": "string",
                    "description": (
                        "Optional. The [id] of an earlier active decision this one "
                        "replaces. Leave unset if this is a new decision, not a "
                        "change to an existing one."
                    ),
                },
            },
            "required": ["decision"],
        },
    },
    {
        "name": "ask_human",
        "description": (
            "Ask the human a clarifying question when the goal or a subtask is "
            "genuinely ambiguous, rather than guessing and building on top of a "
            "guess. Use sparingly -- only for decisions that would be expensive to reverse."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
]


class ToolError(RuntimeError):
    """Raised for malformed tool input -- caught by the loop and sent back
    to the model as a tool_result error so it can repair its own call."""


class ToolBox:
    def __init__(self, workspace: Workspace, config: Config,
                 on_command_needs_approval: Callable[[str, ApprovalDecision], bool] | None = None,
                 on_pause: Callable[[], None] | None = None,
                 on_resume: Callable[[], None] | None = None):
        self.ws = workspace
        self.config = config
        # Overridable hook so tests/CLIs can supply their own approval UI.
        self._confirm_hook = on_command_needs_approval
        # Called around any blocking input() so time spent waiting on the
        # human isn't counted as the agent "running" (see budget.py).
        self._on_pause = on_pause
        self._on_resume = on_resume
        # record_decision is dispatched identically from the outer and inner
        # loops (see agent.py) via this single shared store -- there is
        # deliberately no per-loop handler to keep in sync.
        self.decisions = DecisionsStore(config.decisions_file)
        # Label attached to decisions.add() as `originating_step` -- context
        # only, never validated. agent.py sets this to the active step's
        # title before entering the inner loop and resets it back to the
        # default at the top of each outer-loop turn.
        self.current_context = "outer loop (exploration)"

    def _blocking_input(self, fn: Callable[[], Any]) -> Any:
        if self._on_pause:
            self._on_pause()
        try:
            return fn()
        finally:
            if self._on_resume:
                self._on_resume()

    # ---------- dispatch ----------

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> str:
        """Returns a string to put back in the tool_result content. Raises
        ToolError for bad input (caller turns that into an error tool_result)."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            raise ToolError(f"unknown tool '{name}'")
        try:
            return handler(tool_input)
        except WorkspaceError as e:
            # Defense in depth: individual handlers should route path
            # resolution through self._resolve() (below) so this is
            # normally never reached -- but a path-safety violation must
            # never propagate as a raw crash regardless of which handler
            # triggered it, so it's converted here too as a fallback.
            raise ToolError(str(e))

    def _resolve(self, path: str) -> Path:
        """Resolve a model-given path, turning an out-of-workspace path into
        a normal ToolError (fed back to the model so it can retry with a
        real relative path) instead of an unhandled WorkspaceError crash."""
        try:
            return self.ws.resolve(path)
        except WorkspaceError as e:
            raise ToolError(
                f"{e} -- use a path relative to the workspace root, not an absolute path."
            )

    def _resolve_for_write(self, path: str) -> Path:
        """Like _resolve, but for the mutation tools (write_file,
        edit_file) -- also refuses paths inside .git/ or .autocoder/."""
        try:
            return self.ws.resolve_for_write(path)
        except WorkspaceError as e:
            raise ToolError(str(e))

    # ---------- handlers ----------

    # A single absurdly long line (minified JS, a huge JSON blob written on
    # one line) stays under the line-count limit but can still blow the
    # model's context budget by itself -- read_file only ever capped
    # *lines* returned, never characters. MAX_READ_FILE_CHARS is a backstop
    # for the same reason on the other axis: many long-but-under-the-cap
    # lines can still add up to something huge.
    MAX_LINE_CHARS = 2000
    MAX_READ_FILE_CHARS = 100_000

    def _tool_read_file(self, inp: dict[str, Any]) -> str:
        path = self._require_str(inp, "path")
        offset = int(inp.get("offset", 1))
        limit = int(inp.get("limit", 2000))
        target = self._resolve(path)
        if not target.exists():
            raise ToolError(f"file does not exist: {path}")
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start = max(offset - 1, 0)
        chunk = lines[start:start + limit]
        rendered = []
        for i, line in enumerate(chunk):
            if len(line) > self.MAX_LINE_CHARS:
                line = line[:self.MAX_LINE_CHARS] + f"...[line truncated, {len(line)} chars total]"
            rendered.append(f"{start + i + 1:>6}\t{line}")
        numbered = "\n".join(rendered)
        if len(numbered) > self.MAX_READ_FILE_CHARS:
            numbered = numbered[:self.MAX_READ_FILE_CHARS] + "\n...[output truncated, use offset/limit to page further]"
        footer = f"\n[showing lines {start+1}-{start+len(chunk)} of {total}]"
        return (numbered or "[empty file]") + footer

    def _tool_write_file(self, inp: dict[str, Any]) -> str:
        path = self._require_str(inp, "path")
        content = inp.get("content", "")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        target = self._resolve_for_write(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        return f"{'overwrote' if existed else 'created'} {path} ({len(content)} bytes)"

    def _tool_edit_file(self, inp: dict[str, Any]) -> str:
        path = self._require_str(inp, "path")
        old_str = inp.get("old_str", "")
        new_str = inp.get("new_str", "")
        if not old_str:
            raise ToolError("old_str must not be empty")
        target = self._resolve_for_write(path)
        if not target.exists():
            raise ToolError(f"file does not exist: {path} (use write_file to create it)")
        content = target.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_str)
        if count == 0:
            raise ToolError(
                "old_str not found in file -- it must match the file's CURRENT content "
                "exactly (whitespace included). Re-read the file and copy the exact text."
            )
        if count > 1:
            raise ToolError(
                f"old_str matches {count} locations, not 1 -- add more surrounding "
                "context so it uniquely identifies a single location."
            )
        target.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
        return f"edited {path} (1 replacement)"

    def _tool_list_dir(self, inp: dict[str, Any]) -> str:
        rel = inp.get("path", ".") or "."
        max_entries = int(inp.get("max_entries", 300))
        root = self._resolve(rel)
        if not root.exists():
            raise ToolError(f"path does not exist: {rel}")
        entries: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES]
            rel_dir = os.path.relpath(dirpath, self.ws.root)
            for f in sorted(filenames):
                entries.append(os.path.normpath(os.path.join(rel_dir, f)))
                if len(entries) >= max_entries:
                    entries.append(f"... truncated at {max_entries} entries")
                    return "\n".join(entries)
        return "\n".join(entries) if entries else "[empty]"

    def _tool_search_code(self, inp: dict[str, Any]) -> str:
        pattern = self._require_str(inp, "pattern")
        rel = inp.get("path", ".") or "."
        max_results = int(inp.get("max_results", 100))
        root = self._resolve(rel)

        rg = self._try_ripgrep(pattern, root, max_results)
        if rg is not None:
            return rg

        # Pure-Python fallback (no ripgrep on PATH).
        import re
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}")
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        rel_path = os.path.relpath(fpath, self.ws.root)
                        results.append(f"{rel_path}:{i}:{line.strip()[:200]}")
                        if len(results) >= max_results:
                            return "\n".join(results) + f"\n... truncated at {max_results} results"
        return "\n".join(results) if results else "[no matches]"

    def _try_ripgrep(self, pattern: str, root: Path, max_results: int) -> str | None:
        import shutil as _shutil
        import subprocess
        if _shutil.which("rg") is None:
            return None
        try:
            proc = subprocess.run(
                ["rg", "-n", "--no-heading", "-e", pattern, str(root)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None  # fall through to the pure-Python search below
        if proc.returncode not in (0, 1):  # rg returns 1 for "no matches", that's fine
            return None
        lines = proc.stdout.strip().splitlines()
        if not lines:
            return "[no matches]"
        if len(lines) > max_results:
            return "\n".join(lines[:max_results]) + f"\n... truncated at {max_results} results"
        return "\n".join(lines)

    def run_gated_command(self, command: str, timeout: int) -> CommandResult:
        """Single chokepoint for executing a shell command against the
        workspace: classify it against the approval policy and, if it needs
        a human's OK, block for one before running anything.

        This is the ONLY place a command is allowed to reach
        Workspace.run_command from agent code. Both the run_command tool
        and the agent's acceptance-command execution sites (propose_step's
        acceptance_command, and the final declare_done acceptance check)
        go through here, so the approval gate can't be bypassed just by
        calling the workspace directly instead of going through a tool.
        A denied command returns exit_code=1 with an explanatory stderr
        rather than raising, matching how a real failing check looks.
        """
        os_problem = validate_command_for_os(command)
        if os_problem:
            return CommandResult(
                command=command, exit_code=1, stdout="",
                stderr=f"[REJECTED -- would fail on this OS] {os_problem}", duration_s=0.0,
            )
        decision = classify_command(command, self.config.approval)
        escape_reason = detect_workspace_escape(command)
        if escape_reason:
            # Overrides classify_command's own decision, including under
            # "auto" mode -- an escape-looking command is exactly the case
            # "auto" shouldn't get to skip past silently. Still just a
            # confirmation prompt, not a hard block: this is a best-effort
            # pattern match (see detect_workspace_escape), not a real
            # sandbox boundary, so a human gets the final call rather than
            # the harness assuming the worst and refusing outright.
            decision = ApprovalDecision(
                True, f"command {escape_reason} -- may reach outside the workspace"
            )
        approved = True
        if decision.needs_confirmation:
            if self._confirm_hook is not None:
                approved = self._blocking_input(lambda: self._confirm_hook(command, decision))
            else:
                approved = self._blocking_input(lambda: confirm_with_human(
                    f"Agent wants to run:\n  {command}\nReason for asking: {decision.reason}"
                ))
        if not approved:
            return CommandResult(
                command=command, exit_code=1, stdout="",
                stderr="[DENIED BY HUMAN] command was not run", duration_s=0.0,
            )
        return self.ws.run_command(command, timeout=timeout)

    def _tool_run_command(self, inp: dict[str, Any]) -> str:
        command = self._require_str(inp, "command")
        timeout = int(inp.get("timeout", self.config.budget.default_command_timeout))
        result = self.run_gated_command(command, timeout)
        status = "TIMED OUT" if result.timed_out else f"exit code {result.exit_code}"
        return (f"$ {command}\n[{status}, {result.duration_s:.1f}s]\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")

    def _tool_git_diff(self, inp: dict[str, Any]) -> str:
        staged = bool(inp.get("staged", False))
        result = self.ws.git_diff(staged=staged)
        return result.stdout or "[no differences]"

    def _tool_git_log(self, inp: dict[str, Any]) -> str:
        n = int(inp.get("n", 10))
        result = self.ws.git_log(n=n)
        return result.stdout or "[no commits yet]"

    def _tool_ask_human(self, inp: dict[str, Any]) -> str:
        question = self._require_str(inp, "question")
        return self._blocking_input(lambda: ask_human_question(question))

    def _tool_record_decision(self, inp: dict[str, Any]) -> str:
        decision = self._require_str(inp, "decision")
        supersedes = inp.get("supersedes") or None
        if supersedes is not None and not isinstance(supersedes, str):
            raise ToolError("'supersedes' must be a string id if provided")
        # _require_str already guarantees `decision` is non-empty, so
        # DecisionsStore.add can only return None on blank input -- which
        # can't happen here; the assert documents that invariant.
        result = self.decisions.add(
            decision=decision,
            originating_step=self.current_context,
            supersedes=supersedes,
        )
        assert result is not None
        if supersedes and result.superseded_found is False:
            return (
                f"Recorded as new decision [{result.id}], but '{supersedes}' did not "
                "match any active decision id, so nothing was superseded -- check the "
                "id shown in the ACTIVE DECISIONS section and call record_decision "
                "again with supersedes set correctly if that was intended."
            )
        if supersedes:
            return f"Recorded [{result.id}], superseding [{supersedes}]."
        return f"Recorded [{result.id}]."

    # mark_step_done/propose_step/declare_done are intentionally NOT handled
    # here -- agent.py intercepts those tool_use blocks before dispatch,
    # because acting on them requires the current step's acceptance-check
    # context (and, for propose_step, transitioning between the outer and
    # inner loop) rather than a stateless tool handler.

    # ---------- helpers ----------

    @staticmethod
    def _require_str(inp: dict[str, Any], key: str) -> str:
        val = inp.get(key)
        if not isinstance(val, str) or not val:
            raise ToolError(f"'{key}' is required and must be a non-empty string")
        return val
