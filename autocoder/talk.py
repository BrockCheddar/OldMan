#!/usr/bin/env python3
"""
talk.py -- a plain-English front end for autocoder.

Run: python talk.py
Asks what you want built, where, and (optionally) how to verify it's done,
then runs the same Agent used by `python -m autocoder`. No CLI flags, no
shell quoting to get right -- just answer the questions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autocoder.agent import Agent, AgentAborted
from autocoder.config import load_config, DEFAULT_CONFIG_FILENAME
from autocoder.llm import LLMError


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


_BASELINE_ENTRIES = {".git", ".autocoder", ".gitignore", ".gitkeep"}


def _workspace_has_existing_content(workspace: Path) -> bool:
    """True if the folder has real files in it already -- not just the git/
    session scaffolding every workspace gets automatically. Used to warn
    before a fresh goal gets built alongside a previous, unrelated task's
    files in the same directory instead of a clean folder."""
    if not workspace.exists():
        return False
    return any(item.name not in _BASELINE_ENTRIES for item in workspace.iterdir())


def run_one_task() -> str:
    """Returns 'done', 'aborted', 'interrupted', or 'incomplete'."""
    workspace = Path(ask("Workspace folder to work in", "./workspace")).resolve()

    config_path = Path(DEFAULT_CONFIG_FILENAME).resolve()
    if not config_path.exists():
        print(f"[note] no {DEFAULT_CONFIG_FILENAME} in this directory -- using defaults "
              f"(local llama-server at http://127.0.0.1:8080/v1). Copy "
              f"autocoder.config.example.json to {DEFAULT_CONFIG_FILENAME} to customize.\n")

    session_file = workspace / ".autocoder" / "session.json"
    resume = False
    goal = None
    source_repo = None

    if session_file.exists():
        choice = ask("Found an in-progress session here. Resume it, or start something new? (resume/new)", "resume")
        resume = choice.lower().startswith("r")

    if not resume and _workspace_has_existing_content(workspace):
        print(f"\n[warning] '{workspace}' already has files in it from a previous task "
              "(not just session bookkeeping) -- a new goal here would be built ALONGSIDE "
              "that existing content in the same folder, not a clean start.")
        choice2 = ask("Use a different (clean) workspace folder instead, or continue in this one?",
                      "different")
        if not choice2.lower().startswith("c"):
            new_ws = input("New workspace folder: ").strip()
            while not new_ws:
                new_ws = input("(please enter a folder path) ").strip()
            workspace = Path(new_ws).resolve()
            session_file = workspace / ".autocoder" / "session.json"  # a fresh folder has no session
            resume = False

    if not resume:
        goal = input("\nWhat do you want built or changed? ").strip()
        while not goal:
            goal = input("(please describe what you want) ").strip()
        src = ask("Existing project folder to build on (blank to start fresh)", "")
        source_repo = Path(src).resolve() if src else None

    acceptance = ask(
        "\nOptional: a command that proves the WHOLE goal is done, e.g. 'pytest' "
        "(blank = I'll ask you to confirm when the agent thinks it's finished)",
        "",
    ) or None

    config = load_config(workspace_root=workspace, source_repo=source_repo, config_path=config_path)
    print(f"\n[config] provider={config.llm.provider} model={config.llm.model}")
    print(f"[config] workspace={workspace}\n")

    try:
        agent = Agent(config)
        state = agent.run(goal=goal, resume=resume, final_acceptance_command=acceptance)
    except AgentAborted:
        print("\nRun aborted.")
        return "aborted"
    except LLMError as e:
        print(f"\n[fatal] {e}", file=sys.stderr)
        return "aborted"
    except KeyboardInterrupt:
        print("\n[interrupted] progress saved -- run this script again to resume.")
        return "interrupted"

    return state.status if state.status in ("done", "aborted") else "incomplete"


def main() -> int:
    print("autocoder -- tell me what you want built, in plain English.")
    print("(Ctrl-C at any point saves your progress; run this again to resume.)\n")

    last_status = "done"
    while True:
        last_status = run_one_task()
        if last_status == "interrupted":
            return 130
        if last_status != "done":
            return 1
        again = ask("\nAnything else? (y/n)", "n")
        if not again.lower().startswith("y"):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
