from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import Agent, AgentAborted
from .config import load_config, DEFAULT_CONFIG_FILENAME
from .llm import LLMError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autocoder", description="Incremental autonomous coding agent.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start", help="Start a new task.")
    start.add_argument("goal", help="What to build or change, in plain English.")
    start.add_argument("--workspace", required=True)
    start.add_argument("--source-repo", default=None)
    start.add_argument("--acceptance", default=None,
                       help="Optional shell command that must exit 0 for the whole goal to be accepted as done.")
    start.add_argument("--config", default=None)

    resume = sub.add_parser("resume", help="Resume an in-progress session.")
    resume.add_argument("--workspace", required=True)
    resume.add_argument("--acceptance", default=None)
    resume.add_argument("--config", default=None)

    reopen = sub.add_parser("reopen", help="Reopen a done/aborted session with additional instructions.")
    reopen.add_argument("instructions", help="What still needs doing, in plain English.")
    reopen.add_argument("--workspace", required=True)
    reopen.add_argument("--acceptance", default=None)
    reopen.add_argument("--config", default=None)

    export = sub.add_parser("export", help="Export this workspace's session + lessons + decisions to a portable file.")
    export.add_argument("--workspace", required=True)
    export.add_argument("--out", required=True, help="Path to write the export bundle to.")
    export.add_argument("--config", default=None)

    import_cmd = sub.add_parser("import", help="Import a session bundle into this workspace, to continue elsewhere.")
    import_cmd.add_argument("bundle", help="Path to a bundle produced by `export`.")
    import_cmd.add_argument("--workspace", required=True)
    import_cmd.add_argument("--force", action="store_true", help="Overwrite an existing session in this workspace.")
    import_cmd.add_argument("--config", default=None)

    import_lessons = sub.add_parser("import-lessons", help="Merge another workspace's lessons.json into this one.")
    import_lessons.add_argument("lessons_path", help="Path to another workspace's .autocoder/lessons.json.")
    import_lessons.add_argument("--workspace", required=True)
    import_lessons.add_argument("--config", default=None)

    args = parser.parse_args(argv)

    workspace_root = Path(args.workspace).resolve()
    config_path = Path(args.config).resolve() if args.config else Path(DEFAULT_CONFIG_FILENAME).resolve()
    source_repo = Path(args.source_repo).resolve() if getattr(args, "source_repo", None) else None

    config = load_config(workspace_root=workspace_root, source_repo=source_repo, config_path=config_path)

    print(f"[config] provider={config.llm.provider} model={config.llm.model} "
          f"base_url={getattr(config.llm, 'base_url', '(n/a)')}")
    print(f"[config] workspace={workspace_root}")

    try:
        agent = Agent(config)
        if args.cmd == "start":
            agent.run(goal=args.goal, resume=False, final_acceptance_command=args.acceptance)
        elif args.cmd == "reopen":
            try:
                agent.reopen(args.instructions)
            except ValueError as e:
                print(f"[error] {e}", file=sys.stderr)
                return 1
            agent.run(goal=None, resume=True, final_acceptance_command=args.acceptance)
        elif args.cmd == "export":
            try:
                agent.export_session(Path(args.out).resolve())
            except ValueError as e:
                print(f"[error] {e}", file=sys.stderr)
                return 1
            print(f"[export] wrote {args.out}")
        elif args.cmd == "import":
            try:
                agent.import_session(Path(args.bundle).resolve(), force=args.force)
            except (ValueError, FileNotFoundError) as e:
                print(f"[error] {e}", file=sys.stderr)
                return 1
            print(f"[import] session restored -- resume with `autocoder resume --workspace {workspace_root}`")
        elif args.cmd == "import-lessons":
            count = agent.import_lessons(Path(args.lessons_path).resolve())
            print(f"[import-lessons] added {count} new lesson(s)")
        elif args.cmd == "resume":
            agent.run(goal=None, resume=True, final_acceptance_command=args.acceptance)
    except AgentAborted:
        print("Run aborted.")
        return 1
    except LLMError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[interrupted] resume with `autocoder resume --workspace ...`")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
