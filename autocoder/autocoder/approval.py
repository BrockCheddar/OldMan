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

# A bare "cd <dir>" (or "cd /d <dir>" on Windows for a cross-drive change)
# segment doesn't itself need approval: each run_command call is a fresh
# subprocess rooted at the workspace anyway (see workspace.py), so `cd` only
# affects that one invocation, never the harness's own state or a later
# call. This is checked per-segment below, after splitting -- the
# ALWAYS_CONFIRM_SUBSTRINGS check still runs against the full original
# command first, so this can't be used to sneak a dangerous command past it.
_CD_SEGMENT_RE = re.compile(r"^cd(?:\s+/d)?\s+\S+$", re.IGNORECASE)

# Two-character operators must be checked before their single-character
# component (e.g. "&&" before "&", "||" before "|") so a compound operator
# isn't split in the middle.
_TWO_CHAR_OPERATORS = ("&&", "||")
_ONE_CHAR_OPERATORS = (";", "|")


def _split_command_segments(command: str) -> list[str] | None:
    """Split a shell command into its individual pipeline/chain segments on
    &&, ||, ;, and | -- but not when those characters appear inside a
    quoted string, so e.g. a semicolon embedded in a `python -c "..."`
    argument isn't mistaken for a second command.

    Returns None if quoting is unbalanced at the end of the string, since
    that means the split can't be trusted -- callers should treat that as
    "needs approval" rather than silently classifying on a broken parse.

    This is a classifier, not a full shell parser: it's deliberately
    conservative (single/double quote tracking only, no backslash-escape
    handling), which is the right direction to err in for a safety check.
    """
    segments: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
        elif not in_single and not in_double and command[i:i + 2] in _TWO_CHAR_OPERATORS:
            segments.append("".join(current))
            current = []
            i += 1  # extra advance for the operator's second character
        elif not in_single and not in_double and ch in _ONE_CHAR_OPERATORS:
            segments.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if in_single or in_double:
        return None
    segments.append("".join(current))
    return [s.strip() for s in segments if s.strip()]


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

    segments = _split_command_segments(command)
    if segments is None:
        return ApprovalDecision(True, "command has unbalanced quotes and could not be safely classified")
    if not segments:
        return ApprovalDecision(True, "empty command")

    matched_prefixes: list[str] = []
    for seg in segments:
        seg_lowered = seg.lower()
        if _CD_SEGMENT_RE.match(seg_lowered):
            matched_prefixes.append("cd")
            continue  # bare cd: safe on its own, see comment above
        hit = next((safe for safe in SAFE_COMMAND_PREFIXES if seg_lowered.startswith(safe)), None)
        if hit is None:
            reason = "not on the safe-prefix list"
            if len(segments) > 1:
                reason = f"segment '{seg}' is {reason}"
            return ApprovalDecision(True, reason)
        matched_prefixes.append(hit)

    if len(segments) > 1:
        reason = f"every segment matches a safe prefix ({', '.join(matched_prefixes)})"
    else:
        reason = f"matches safe read-only prefix '{matched_prefixes[0]}'"
    return ApprovalDecision(False, reason)


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
