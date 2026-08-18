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
from pathlib import Path
from typing import Any, Callable

from .approval import ApprovalDecision, classify_command, confirm_with_human, ask_human_question
from .config import Config
from .workspace import Workspace, WorkspaceError

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

    # ---------- handlers ----------

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
        numbered = "\n".join(f"{start + i + 1:>6}\t{line}" for i, line in enumerate(chunk))
        footer = f"\n[showing lines {start+1}-{start+len(chunk)} of {total}]"
        return (numbered or "[empty file]") + footer

    def _tool_write_file(self, inp: dict[str, Any]) -> str:
        path = self._require_str(inp, "path")
        content = inp.get("content", "")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        target = self._resolve(path)
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
        target = self._resolve(path)
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
        if _shutil.which("rg") is None:
            return None
        result = self.ws.run_command(
            f'rg -n --no-heading -e {_shell_quote(pattern)} {_shell_quote(str(root))} | head -n {max_results}',
            timeout=30,
        )
        if result.exit_code not in (0, 1):  # rg returns 1 for "no matches", that's fine
            return None
        out = result.stdout.strip()
        return out if out else "[no matches]"

    def _tool_run_command(self, inp: dict[str, Any]) -> str:
        command = self._require_str(inp, "command")
        timeout = int(inp.get("timeout", self.config.budget.default_command_timeout))
        decision = classify_command(command, self.config.approval)
        approved = True
        if decision.needs_confirmation:
            if self._confirm_hook is not None:
                approved = self._blocking_input(lambda: self._confirm_hook(command, decision))
            else:
                approved = self._blocking_input(lambda: confirm_with_human(
                    f"Agent wants to run:\n  {command}\nReason for asking: {decision.reason}"
                ))
        if not approved:
            return "[DENIED BY HUMAN] command was not run: " + command
        result = self.ws.run_command(command, timeout=timeout)
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


def _shell_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)
