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
        else:
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
