"""
Approval gates (Phase 7): anything irreversible or externally visible should
get a human's eyes on it before it runs, unless the policy is set to "auto".

This module only decides WHETHER to ask and prints the prompt; it never
silently escalates privilege and never runs anything itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import ApprovalPolicy, ALWAYS_CONFIRM_SUBSTRINGS, SAFE_COMMAND_PREFIXES

# A leading "cd <dir> && rest" (or "cd /d <dir> &&" on Windows for a
# cross-drive change) doesn't itself need approval: each run_command call is
# a fresh subprocess rooted at the workspace anyway (see workspace.py), so
# `cd` only affects that one invocation, never the harness's own state or a
# later call. Stripped ONLY for the safe-prefix match below -- the
# ALWAYS_CONFIRM_SUBSTRINGS check still runs against the full original
# command first, so this can't be used to sneak a dangerous command past it.
_CD_PREFIX_RE = re.compile(r"^cd(?:\s+/d)?\s+\S+\s*(?:&&|;)\s*", re.IGNORECASE)


@dataclass
class ApprovalDecision:
    needs_confirmation: bool
    reason: str


def classify_command(command: str, policy: ApprovalPolicy) -> ApprovalDecision:
    lowered = command.strip().lower()

    if policy.mode == "auto":
        return ApprovalDecision(False, "approval policy set to auto")
    if policy.mode == "ask":
        return ApprovalDecision(True, "approval policy set to ask-always")

    # mode == "smart"
    for bad in ALWAYS_CONFIRM_SUBSTRINGS:
        if bad in lowered:
            return ApprovalDecision(True, f"command contains '{bad.strip()}' (always confirmed)")

    effective = _CD_PREFIX_RE.sub("", lowered, count=1)
    for safe in SAFE_COMMAND_PREFIXES:
        if effective.startswith(safe):
            reason = f"matches safe read-only prefix '{safe}'"
            if effective != lowered:
                reason += " (after a leading 'cd ... &&')"
            return ApprovalDecision(False, reason)

    return ApprovalDecision(True, "not on the safe-prefix list")


def confirm_with_human(prompt: str) -> bool:
    """Blocking CLI confirmation. Swap this out for a GUI/webhook prompt later
    without touching any caller -- it's the only place a 'yes' is granted."""
    print("\n" + "=" * 60)
    print("APPROVAL REQUIRED")
    print(prompt)
    print("=" * 60)
    answer = input("Proceed? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def ask_human_question(question: str) -> str:
    """Used by the ask_human tool when the model itself is escalating a
    genuine ambiguity (Phase 4's 'ask a clarifying question' path)."""
    print("\n" + "-" * 60)
    print("THE AGENT IS ASKING FOR CLARIFICATION:")
    print(question)
    print("-" * 60)
    return input("Your answer: ").strip()
