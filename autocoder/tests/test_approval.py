import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocoder.approval import classify_command
from autocoder.config import ApprovalPolicy


def test_plain_safe_command_needs_no_approval():
    d = classify_command("pytest -q", ApprovalPolicy(mode="smart"))
    assert not d.needs_confirmation


def test_plain_unsafe_command_needs_approval():
    d = classify_command("pip install requests", ApprovalPolicy(mode="smart"))
    assert d.needs_confirmation


def test_cd_prefixed_safe_command_needs_no_approval():
    """
    Regression: 'cd demo_project && pytest -q' used to require approval even
    though 'pytest -q' alone is safe, because the safe-prefix check only
    looked at the start of the string and 'cd' isn't a safe prefix itself.
    Each run_command call is a fresh subprocess rooted at the workspace
    regardless of an internal cd, so this should be treated the same as the
    bare command.
    """
    d = classify_command("cd demo_project && pytest -q", ApprovalPolicy(mode="smart"))
    assert not d.needs_confirmation


def test_cd_prefixed_python_c_needs_approval():
    """H4 regression: bare 'python -c ...' used to be auto-approved because
    'python ' was on the safe-prefix list. python -c runs arbitrary code
    (e.g. shutil.rmtree), so it must require approval like any other
    non-safe segment -- only specific narrow forms (python -m pytest,
    python -m py_compile) are pre-approved."""
    d = classify_command(
        'cd demo_project && python -c "from src.server import start_server"',
        ApprovalPolicy(mode="smart"),
    )
    assert d.needs_confirmation


def test_cd_prefixed_pytest_module_form_needs_no_approval():
    d = classify_command(
        "cd demo_project && python -m pytest -q",
        ApprovalPolicy(mode="smart"),
    )
    assert not d.needs_confirmation


def test_chained_safe_and_unsafe_command_needs_approval():
    """H4 regression: 'git status && python evil.py' used to auto-run
    because only the head of the string was checked."""
    d = classify_command("git status && python evil.py", ApprovalPolicy(mode="smart"))
    assert d.needs_confirmation


def test_piped_unsafe_command_needs_approval():
    """H4 regression: 'ls; python -c "..."' used to auto-run for the same
    reason -- only the first segment was ever inspected."""
    d = classify_command('ls; python -c "import os; os.system(1)"', ApprovalPolicy(mode="smart"))
    assert d.needs_confirmation


def test_semicolon_inside_quoted_arg_is_not_treated_as_a_new_segment():
    """A ';' inside a python -c string argument is part of that one
    argument, not a second shell command -- splitting must respect quotes."""
    d = classify_command('pytest -k "a; b"', ApprovalPolicy(mode="smart"))
    assert not d.needs_confirmation


def test_cd_with_windows_slash_d_flag_also_stripped():
    d = classify_command("cd /d D:\\other\\project && pytest -q", ApprovalPolicy(mode="smart"))
    assert not d.needs_confirmation


def test_cd_prefix_does_not_bypass_always_confirm():
    """The cd-stripping must never let something genuinely dangerous slip
    through -- always-confirm substrings are checked against the FULL
    original command, before any stripping happens."""
    d = classify_command("cd demo_project && pip install something-sketchy", ApprovalPolicy(mode="smart"))
    assert d.needs_confirmation


def test_cd_prefix_with_unsafe_rest_still_needs_approval():
    d = classify_command("cd demo_project && rm -rf build", ApprovalPolicy(mode="smart"))
    assert d.needs_confirmation


def test_auto_mode_never_confirms_even_unsafe_commands():
    d = classify_command("pip install anything", ApprovalPolicy(mode="auto"))
    assert not d.needs_confirmation


def test_ask_mode_always_confirms_even_safe_commands():
    d = classify_command("pytest -q", ApprovalPolicy(mode="ask"))
    assert d.needs_confirmation
