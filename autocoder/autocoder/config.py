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
    max_subtask_attempts: int = 3        # acceptance-check attempts per subtask before triggering a re-plan
    max_replan_cycles: int = 2           # re-plan cycles per subtask before forcing escalation to the human
    max_steps_per_attempt: int = 40      # tool-call turns before an attempt is abandoned
    max_output_tokens: int = 8000        # per single model turn
    max_wall_clock_seconds: int = 21600  # 6h soft cap for a whole run -- warns, does not kill. Local inference is slower; size accordingly.
    max_total_tokens: int | None = None  # None = unlimited
    default_command_timeout: int = 300   # seconds, used when the model doesn't specify one for run_command


@dataclass
class ApprovalPolicy:
    mode: str = "smart"  # "auto" | "ask" | "smart" (smart = ask unless on the safe list)


@dataclass
class Config:
    workspace_root: Path
    source_repo: Path | None = None
    llm: LLMBackendConfig = field(default_factory=LLMBackendConfig)
    planner_llm: LLMBackendConfig | None = None   # None => reuse `llm` for planning too (normal for one local model)
    budget: Budget = field(default_factory=Budget)
    approval: ApprovalPolicy = field(default_factory=ApprovalPolicy)

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

    return Config(
        workspace_root=workspace_root,
        source_repo=source_repo,
        llm=llm,
        planner_llm=planner_llm,
        budget=budget,
        approval=approval,
    )
