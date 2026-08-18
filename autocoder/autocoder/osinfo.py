"""
run_command executes via the real OS shell (see workspace.py: shell=True).
That means cmd.exe on Windows, /bin/sh elsewhere -- and Unix flags like
`mkdir -p` are NOT understood by cmd.exe. Rather than let the model discover
that by trial and error (burning step budget, sometimes hallucinating after
repeated confusion), we tell it up front what shell it's actually talking to.
"""
from __future__ import annotations

import platform


def shell_description() -> str:
    system = platform.system()
    if system == "Windows":
        return (
            "Windows. run_command executes via cmd.exe, NOT bash/PowerShell/WSL. "
            "Concretely: `mkdir foo\\bar\\baz` already creates intermediate directories "
            "on its own -- do not pass `-p` (cmd.exe's mkdir does not understand it and "
            "will try to create a literally-named '-p' entry instead of doing what you want). "
            "Use `dir` not `ls`, `del`/`type` not `rm`/`cat`, `\\` as the path separator in "
            "commands (forward slashes work fine inside file tool paths, just not always in "
            "shell commands). `test` and `[` are POSIX shell builtins and DO NOT EXIST on "
            "cmd.exe -- `test -f somefile` will fail every time, not because the file is "
            "missing but because `test` isn't a recognized command at all. For file-existence "
            "or any other check, use a `python -c \"...\"` one-liner instead (e.g. "
            "`python -c \"import os,sys; sys.exit(0 if os.path.exists('f') else 1)\"`) -- "
            "it behaves identically regardless of OS and is what you should default to for "
            "acceptance_command whenever you're unsure a shell builtin exists on cmd.exe."
        )
    return (
        f"{system}, a POSIX system. run_command executes via /bin/sh, so standard Unix "
        "tools (mkdir -p, ls, rm, find, grep, etc.) all work as expected."
    )
