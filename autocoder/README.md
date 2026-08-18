# autocoder

Gameplan phases 1, 2, 3, 5, and lite versions of 4/6/7, implemented. Not phase 8 (multi-agent) yet.

## What this is

A planner/executor coding agent that runs against a real git working
directory on disk, using native tool-calling against a model backend picked
entirely by config. Default backend is your local llama-server (Ornith9B).
Anthropic is available as an alternate config, not a dependency.

No Docker/container isolation (your call) — tasks run in a real folder,
isolated by git, not by the OS. See "Isolation model" below before pointing
this at anything you care about.

## Install

```
pip install -r requirements.txt
```

Nothing to install for the local-model path — it's stdlib `urllib` only.
`pip install anthropic` only if you set `"provider": "anthropic"`.

## Run

1. Start llama-server with Ornith9B, tool-calling enabled, e.g.:
   ```
   llama-server -m ornith9b.gguf --port 8080 --ctx-size 32768 --jinja
   ```
   `--jinja` (or your build's equivalent flag) matters — it's what makes
   the server parse `tools`/`tool_calls` instead of ignoring them. If the
   agent's planner ever fails with "did not call submit_plan", check this
   first.

2. Copy `autocoder.config.example.json` to `autocoder.config.json` and set
   `context_window_tokens` to match your `--ctx-size` exactly. This isn't
   cosmetic — it's what the context-guard uses to decide when to start
   trimming old tool output so a long task doesn't overflow your server's
   real window.

3. Run:
   ```
   python -m autocoder start "add input validation to the signup form" --workspace ./myproject --source-repo /path/to/existing/repo
   ```
   Omit `--source-repo` to start an empty project in `--workspace`.

4. If it stops (crash, Ctrl-C, budget hit):
   ```
   python -m autocoder resume --workspace ./myproject
   ```

To use Anthropic instead: copy `autocoder.config.anthropic.example.json` to
`autocoder.config.json`, `export ANTHROPIC_API_KEY=...`. Nothing else
changes — same code path either way.

## Config file (`autocoder.config.json`)

- `llm.provider`: `"openai_compat"` (llama-server/LM Studio/vLLM/Ollama) or `"anthropic"`.
- `llm.request_timeout_s`: defaults to 1800s. Local inference is not instant — this is a
  "the server actually hung" timeout, not a "hurry up" timeout. Raise it further if needed;
  nothing in this codebase assumes a fast response.
- `llm.context_window_tokens`: must match your actual server context size.
- `planner_llm`: optional, only set this if you're running a second model/endpoint for
  planning. Left unset (the normal case with one local model), planning and execution
  share the same backend — one model, called strictly sequentially, never two requests
  in flight at once.
- `budget`: step/attempt/wall-clock/token caps — see `config.py` for each field's meaning.
- `approval.mode`: `"smart"` (default — asks only for risky commands: installs, pushes,
  deletes, network calls, `sudo`), `"auto"` (never asks), `"ask"` (always asks).

## Isolation model

- Every task runs in a real folder, in a real git repo, with real subprocess execution.
- The workspace's own `.autocoder/` session dir is gitignored automatically so the
  undo mechanism (`git reset --hard && git clean -fd`) can never delete session state
  along with a discarded attempt.
- There is **no OS-level sandbox**. A command the model runs can touch anything your
  Windows account can touch. The approval gate is the real defense right now — read
  `config.py`'s `SAFE_COMMAND_PREFIXES`/`ALWAYS_CONFIRM_SUBSTRINGS` and adjust if the
  defaults are wrong for how you work.
- `workspace.py`'s `Workspace` class is the seam where a Docker/Firecracker executor
  would plug in later without touching the planner, tools, or agent loop.

## What actually verifies a subtask is done

Not the model's word. The planner is required to give every subtask a real
`acceptance_command`. When the model calls `mark_subtask_done`, the harness runs that
command for real and only commits + marks it done if it exits 0. Failing that a set
number of times triggers re-planning (retry with new guidance / revise the remaining
plan / escalate to you), not a silent "STUCK".

## Tests

```
pip install -r requirements-dev.txt
pytest tests/ -q            # 36 tests, no network/model required (fake LLM client)
python e2e_smoke_test.py    # real CLI subprocess against a fake local HTTP server
```

## Not built yet (see gameplan.md for the rest)

- Container/microVM execution (Phase 1's stronger isolation option)
- Independent reviewer agent, separate from the coder (Phase 4's adversarial review)
- Multi-agent specialization (Phase 8)
