"""
Central configuration for the autocoder agent.

Design point: the model backend is picked entirely by config, not code.
Default is your local llama-server (Ornith9B) OpenAI-compatible endpoint.
Swapping to Anthropic (or any other OpenAI-compatible server -- LM Studio,
vLLM, Ollama's /v1 endpoint, etc.) means editing the config file, not this
module.

All calls to the model are synchronous and strictly sequential -- the
planner runs to completion, then the executor makes one call, waits for the
full response, runs tools, makes the next call. Nothing here spins up a
second concurrent request against the same backend, so a single-slot local
server (the normal llama-server setup) is never asked to serve two requests
at once and made to queue/starve one behind the other.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


DEFAULT_CONFIG_FILENAME = "autocoder.config.json"

# Commands that never need human approval to run (read-only / local-only,
# safe-by-default). Matched against the start of the command, case-insensitive,
# AFTER the command has been split on shell operators (&&, ||, ;, |) and
# EVERY segment is required to match one of these -- see classify_command.
#
# IMPORTANT: bare interpreters ("python ", "node ", etc.) must never appear
# here. They're general-purpose code execution, not read-only commands --
# "python -c \"import shutil; shutil.rmtree(...)\"" would match a bare
# "python " prefix just as well as a real test run. List the exact safe
# invocation forms instead (module flags, not the interpreter itself).
SAFE_COMMAND_PREFIXES = {
    "git status", "git diff", "git log", "git show", "git branch",
    "ls", "dir", "cat", "type", "find", "tree",
    "mkdir",
    "pytest", "python -m pytest", "python3 -m pytest", "python.exe -m pytest",
    "npm test", "npm run test", "yarn test",
    "go test", "cargo test",
    "mypy", "pyright", "eslint", "tsc",
    "python -m py_compile", "python3 -m py_compile", "python.exe -m py_compile",
    "node --check",
}

# Substrings that ALWAYS force an approval prompt, even over a safe prefix match.
ALWAYS_CONFIRM_SUBSTRINGS = {
    "rm -rf", "rmdir /s", "del /f", "del /s",
    "git push", "git reset --hard", "git clean -f",
    "curl ", "wget ", "invoke-webrequest", "invoke-restmethod",
    "pip install", "pip3 install", "npm install", "npm i ",
    "yarn add", "cargo add", "go get", "go install",
    "sudo ", "chmod 777", "shutdown", "format ",
    " > /dev/", "mkfs",
}


@dataclass
class LLMBackendConfig:
    provider: str = "openai_compat"  # "openai_compat" (llama-server/LM Studio/vLLM/Ollama) or "anthropic"

    # --- openai_compat (default: your local llama-server) ---
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "ornith9b"
    api_key: str = "not-needed"          # llama-server ignores this; some servers require any non-empty string
    request_timeout_s: int = 1800        # 30 min. Local CPU/partial-GPU inference on long contexts is slow -- do not shrink this casually.
    context_window_tokens: int = 32768   # MUST match your llama-server --ctx-size (or lower). Wrong value here either
                                          # wastes context headroom or lets the harness overflow the server's real window.

    # --- anthropic (only used if provider == "anthropic") ---
    api_key_env: str = "ANTHROPIC_API_KEY"


@dataclass
class Budget:
    max_subtask_attempts: int = 3        # acceptance-check attempts per subtask before checking for an autonomous replan
    max_replan_cycles: int = 2           # autonomous replans of a stuck step before escalating to the human
    max_steps_per_attempt: int = 40      # tool-call turns before an attempt is abandoned
    max_output_tokens: int = 8000        # per single model turn
    max_wall_clock_seconds: int = 21600  # 6h soft cap for a whole run -- warns, does not kill. Local inference is slower; size accordingly.
    max_total_tokens: int | None = None  # None = unlimited
    default_command_timeout: int = 300   # seconds, used when the model doesn't specify one for run_command
    # FLAW 8/14: every N completed steps, pause and run a dedicated
    # "take stock" pass (a planner_llm call, not offered to the model as a
    # tool) that checks the run is still solving the right problem and
    # distills the step log into a fresh scratchpad note, instead of
    # scratchpad drift being purely up to whether the model happens to
    # keep it current. 0 disables it entirely.
    self_audit_every_n_steps: int = 6


@dataclass
class ApprovalPolicy:
    mode: str = "smart"  # "auto" | "ask" | "smart" (smart = ask unless on the safe list)


@dataclass
class FinalReviewPolicy:
    """Governs who decides whether a declare_done with no final_acceptance_command
    gets accepted, when there's no automated command to check it. Separate from
    ApprovalPolicy on purpose -- that governs whether a SHELL COMMAND is safe to
    run; this governs whether declared WORK is correct. Different axis, different
    failure mode, so a different config surface rather than overloading "approval"
    to mean two things.
    """
    mode: str = "human"  # "human" | "llm_auto"
    # "human": today's behavior, unchanged -- interactive [y/N] prompt.
    # "llm_auto": a dedicated planner_llm review call decides instead, no human
    #   prompt at all. Built at the person's explicit request, with the caution
    #   already on record: across two real runs in testing, the interactive
    #   human gate did NOT catch either of two real, confirmed bugs (a check
    #   satisfied by corrupting the SVG width attribute rather than fixing the
    #   check; a maze solver that mutated the maze to make its own "solved"
    #   path look valid) -- both were accepted with a bare "y"/"ok" and only
    #   found by separate, deliberate review afterward. This mode is not a
    #   downgrade from a reliable safeguard; it's automating a checkpoint that,
    #   as actually used, wasn't catching this class of bug either way. It is a
    #   genuinely different question being asked, not a rubber stamp -- see
    #   the system prompt in Agent._llm_final_review.


@dataclass
class Config:
    workspace_root: Path
    source_repo: Path | None = None
    llm: LLMBackendConfig = field(default_factory=LLMBackendConfig)
    planner_llm: LLMBackendConfig | None = None   # None => reuse `llm` for planning too (normal for one local model)
    budget: Budget = field(default_factory=Budget)
    approval: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    final_review: FinalReviewPolicy = field(default_factory=FinalReviewPolicy)

    def effective_planner_llm(self) -> LLMBackendConfig:
        return self.planner_llm if self.planner_llm is not None else self.llm

    @property
    def session_dir(self) -> Path:
        return self.workspace_root / ".autocoder"

    @property
    def session_file(self) -> Path:
        return self.session_dir / "session.json"

    @property
    def log_file(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def lessons_file(self) -> Path:
        return self.session_dir / "lessons.json"

    @property
    def decisions_file(self) -> Path:
        return self.session_dir / "decisions.json"


def load_llm_backend(d: dict) -> LLMBackendConfig:
    return LLMBackendConfig(**{**asdict(LLMBackendConfig()), **d})


def load_config(workspace_root: Path, source_repo: Path | None, config_path: Path | None) -> Config:
    """
    Reads a JSON config file (see autocoder.config.example.json). Any field
    left out uses the dataclass default -- editing the config file to change
    providers never requires touching this code.
    """
    data: dict = {}
    if config_path is not None and config_path.exists():
        data = json.loads(config_path.read_text())

    llm = load_llm_backend(data.get("llm", {}))
    planner_raw = data.get("planner_llm")
    planner_llm = load_llm_backend(planner_raw) if planner_raw else None

    budget = Budget(**{**asdict(Budget()), **data.get("budget", {})})
    approval = ApprovalPolicy(**{**asdict(ApprovalPolicy()), **data.get("approval", {})})
    final_review = FinalReviewPolicy(**{**asdict(FinalReviewPolicy()), **data.get("final_review", {})})

    return Config(
        workspace_root=workspace_root,
        source_repo=source_repo,
        llm=llm,
        planner_llm=planner_llm,
        budget=budget,
        approval=approval,
        final_review=final_review,
    )
