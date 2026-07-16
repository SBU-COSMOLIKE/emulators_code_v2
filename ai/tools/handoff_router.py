#!/usr/bin/env python3
"""Carry approved handoff blocks between manual web conversations.

A source note is the Markdown file under ``ai/notes/`` that contains the
Architect's checked plan. The user gives every request and correction only to
the Architect. A human courier may paste a generated handoff block unchanged
into the Implementer or Red Team conversation.

Copies of returned blocks and command output are saved under
``ai/notes/relay/``. They support the Architect's review, but they do not
replace the source note. The Architect reruns every required check and alone
decides ``GO`` or ``NO-GO``.

For one ticket, this program:

1. checks the Architect's source note;
2. puts the approved Implementer block on the clipboard;
3. waits for the Implementer's returned block;
4. runs the local check commands and saves their exact output;
5. includes Red Team only when the source note requires it; and
6. puts every saved result on the clipboard for the Architect.

Usage:

    python ai/tools/handoff_router.py --note ai/notes/<spec>.md --section \\
        "Implementation directive"
    python ai/tools/handoff_router.py --note ai/notes/<spec>.md

The source note may choose three roles, two roles, or Sol as the Implementer.
``--mode``, ``--skip-redteam``, and ``--severity`` can confirm that saved
choice. They cannot change it.

Use ``--gate-cmd`` to name a ticket's local check command:

    --gate-cmd "PYTHONPATH=. <cocoa-python> ai/gates/checks/<child>.py"

Use ``--status`` to read the current Git branches and saved records without
changing the clipboard or waiting for another conversation:

    python ai/tools/handoff_router.py --status
"""

import argparse
import datetime
import fcntl
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(AI_ROOT)
NOTES_DIR = os.path.join(AI_ROOT, "notes")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from handoff_contract import DirectiveError
from handoff_contract import nonnegative_character_limit
from handoff_contract import resolve_character_limit
from handoff_contract import validate_directive_file
RELAY_DIR = os.path.join(NOTES_DIR, "relay")
DISCOVERY_SEVERITIES = ("high", "medium", "low")
DISCOVERY_SEVERITY_ENVIRONMENT = "MAILBOX_DISCOVERY_SEVERITY"
RUN_RESERVATIONS_DIR = os.path.join(RELAY_DIR, ".router-runs")
ROUTER_LOCK_PATH = os.path.join(
    tempfile.gettempdir(),
    "cocoa-handoff-router-" + str(os.getuid()) + ".lock",
)

COCOA_PYTHON = ("/Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa"
                "/.local/bin/python")

DEFAULT_GATE_COMMANDS = [
    COCOA_PYTHON + " -m compileall -q ai/gates emulator",
    "PYTHONPATH=. " + COCOA_PYTHON + " ai/gates/run_board.py --list",
    "PYTHONPATH=. " + COCOA_PYTHON + " ai/gates/checks/board_selftest.py",
]

SECOND_IMPLEMENTER_MODE_SENTENCE = (
    "OpenAI Sol — this is a role as second Implementer for this unit.")


def copy_to_clipboard(text):
    """Put text on the system clipboard (pbcopy on macOS, pyperclip else).

    Arguments:
      text = the full prompt string to copy.
    """
    if sys.platform == "darwin":
        proc = subprocess.run(["pbcopy"],
                              input=text.encode("utf-8"))
        if proc.returncode != 0:
            raise RuntimeError("pbcopy failed")
        return
    import pyperclip
    pyperclip.copy(text)


def read_clipboard():
    """Return the current clipboard text (pbpaste on macOS, pyperclip else)."""
    if sys.platform == "darwin":
        proc = subprocess.run(["pbpaste"],
                              capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError("pbpaste failed")
        return proc.stdout.decode("utf-8", errors="replace")
    import pyperclip
    return pyperclip.paste()


def has_handoff_header(text, header):
    """Return whether text contains the expected handoff heading.

    A bare token in prose is not a handoff. The heading must begin its own
    line, exactly as the role-file templates require.

    Arguments:
      text   = the clipboard text to inspect.
      header = the complete Markdown heading prefix, including ``###``.

    Returns:
      True when the heading begins a line, otherwise False.
    """
    for line in text.splitlines():
        if line.startswith(header):
            return True
    return False


def wait_for_block(header, last_copied):
    """Block until the clipboard holds a new handoff with the right heading.

    The comparison baseline is the text THIS script last copied, so the
    instruction itself cannot be mistaken for the response. Requiring a
    Markdown heading also prevents ordinary prose that mentions the handoff
    token from being mistaken for a return block.

    Arguments:
      header      = the complete heading prefix that identifies the block,
                    e.g. "### IMPLEMENTER_HANDOFF:".
      last_copied = the exact string this script most recently placed on
                    the clipboard.

    Returns:
      the returned clipboard text.
    """
    print("... waiting for a copied block headed '" + header + "'")
    while True:
        current = read_clipboard()
        if current != last_copied and has_handoff_header(current, header):
            return current
        time.sleep(0.5)


def reserve_run_sequence(stamp=None):
    """Atomically reserve a unique transport-copy sequence.

    The former second-resolution timestamp let two router runs choose the
    same filenames and silently overwrite one another. Each successful
    ``mkdir`` below is a persistent reservation, so a later run cannot reuse
    a sequence even after the first process exits.

    Arguments:
      stamp = optional timestamp text for a deterministic scratch probe. When
              omitted, use the current local time to the second.

    Returns:
      the reserved sequence text used as the transport-copy filename prefix.
    """
    os.makedirs(RUN_RESERVATIONS_DIR, exist_ok=True)
    if stamp is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = 0
    while True:
        seq = stamp
        if suffix > 0:
            seq += "-" + str(suffix).zfill(2)
        reservation = os.path.join(RUN_RESERVATIONS_DIR, seq)
        try:
            os.mkdir(reservation)
            return seq
        except FileExistsError:
            suffix += 1


def acquire_router_lock():
    """Take the machine-wide lock that protects the shared clipboard.

    ``flock`` belongs to the open file descriptor, so a killed process
    releases it automatically. The persistent lock file therefore has no
    stale-file failure mode. The path is machine-wide rather than worktree-
    local because every worktree uses the same system clipboard.

    Returns:
      an open file object that must remain alive for the complete relay run.

    Raises:
      RuntimeError: another router process already owns the clipboard flow.
    """
    lock_file = open(ROUTER_LOCK_PATH, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        holder = lock_file.read().strip()
        lock_file.close()
        detail = ""
        if holder:
            detail = " (holder " + holder + ")"
        raise RuntimeError("another handoff router is already running" + detail)
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write("pid=" + str(os.getpid()) + "\n")
    lock_file.flush()
    return lock_file


def release_router_lock(lock_file):
    """Release a lock returned by :func:`acquire_router_lock`.

    Arguments:
      lock_file = the open lock file returned by ``acquire_router_lock``.

    Returns:
      None.
    """
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def resolve_note_path(note):
    """Resolve one direct, non-symlink Markdown note under ``ai/notes``.

    Arguments:
      note = an absolute path or repository-relative path naming a direct
             child of this router checkout's ``ai/notes`` directory.

    Returns:
      ``(path, display_path)`` with an absolute path for I/O and the canonical
      repository-relative path for prompts.

    Raises:
      DirectiveError when the path escapes the source-note directory, names a
      transport subdirectory, uses a non-Markdown suffix, or is a symlink.
    """
    if os.path.isabs(note):
        path = os.path.abspath(note)
    else:
        path = os.path.abspath(os.path.join(REPO_ROOT, note))
    notes_path = os.path.abspath(NOTES_DIR)
    notes_real = os.path.realpath(notes_path)
    expected_notes_real = os.path.join(
        os.path.realpath(REPO_ROOT), "ai", "notes")
    if (os.path.islink(notes_path) or not os.path.isdir(notes_path)
            or notes_real != expected_notes_real):
        raise DirectiveError(
            "Architect source-note directory must be this checkout's real "
            "ai/notes directory, not a symlink or redirected path")
    path_real = os.path.realpath(path)
    if (os.path.dirname(path) != notes_path
            or os.path.dirname(path_real) != notes_real):
        raise DirectiveError(
            "Architect source note must be a direct file inside this "
            "router checkout's ai/notes directory")
    if os.path.splitext(path)[1].casefold() != ".md":
        raise DirectiveError("Architect source note must end in .md")
    basename = os.path.basename(path)
    if (len(basename.encode("utf-8")) > 255
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*\.md", basename)
            is None):
        raise DirectiveError(
            "Architect source note must use a safe ASCII Markdown filename")
    if os.path.islink(path) or path_real != os.path.join(
            notes_real, os.path.basename(path)):
        raise DirectiveError("Architect source note must not be a symlink")
    return (path, os.path.relpath(path, REPO_ROOT))


def archive(seq, name, text):
    """Save one supporting copy under ai/notes/relay/ and return its path.

    Arguments:
      seq  = the run sequence stamp (shared by all files of this run).
      name = short role tag for the filename ("implementer", "sol", ...).
      text = the returned block or command-output text.
    """
    os.makedirs(RELAY_DIR, exist_ok=True)
    path = os.path.join(RELAY_DIR, seq + "-" + name + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!-- SUPPORTING COPY ONLY. The agent-written source note\n"
                "     that this block cites remains authoritative.\n"
                "     Saved by ai/tools/handoff_router.py. -->\n\n")
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    return os.path.relpath(path, REPO_ROOT)


def run_gates(commands, seq):
    """Run local checks and save their complete output.

    The console shows one result line per command. Complete output goes to a
    log file under ``ai/notes/relay/``.

    Arguments:
      commands = list of shell command strings to run from the repo root.
      seq      = the run sequence stamp for the log filename.

    Returns:
      (log_path, all_green) -- the saved log path and whether every command
      returned exit code zero.
    """
    lines = []
    all_green = True
    for cmd in commands:
        proc = subprocess.run(cmd,
                              shell=True,
                              capture_output=True,
                              text=True,
                              cwd=REPO_ROOT)
        verdict = "PASS" if proc.returncode == 0 else "FAIL"
        if proc.returncode != 0:
            all_green = False
        print("  [" + verdict + "] " + cmd + "  (rc=" + str(proc.returncode)
              + ")")
        lines.append("$ " + cmd)
        lines.append("rc=" + str(proc.returncode))
        lines.append("--- stdout ---")
        lines.append(proc.stdout)
        lines.append("--- stderr ---")
        lines.append(proc.stderr)
        lines.append("")
    log_path = archive(seq, "gates-log", "\n".join(lines))
    return (log_path, all_green)


def verify_execution_checkout(checkout):
    """Bind this router process and its gate log to the declared checkout.

    The manual router runs commands from its own repository root. A directive
    naming another checkout must therefore use that checkout's copy of this
    script, never test an implementation accidentally against main.
    """
    worktree = checkout["Worktree"]
    if os.path.realpath(worktree) != os.path.realpath(REPO_ROOT):
        raise DirectiveError(
            "Execution checkout Worktree does not match this router; run "
            "the router from " + worktree)

    def git_value(arguments, label):
        proc = subprocess.run(
            ["git"] + arguments,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True)
        if proc.returncode != 0:
            raise DirectiveError(
                "cannot verify Execution checkout " + label + ": "
                + proc.stderr.strip())
        return proc.stdout.strip()

    top = git_value(["rev-parse", "--show-toplevel"], "worktree")
    if os.path.realpath(top) != os.path.realpath(REPO_ROOT):
        raise DirectiveError("Execution checkout is not this Git worktree")
    git_dir = git_value(["rev-parse", "--git-dir"], "Git directory")
    common_dir = git_value(
        ["rev-parse", "--git-common-dir"], "common Git directory")

    def absolute_git_path(path):
        if not os.path.isabs(path):
            path = os.path.join(REPO_ROOT, path)
        return os.path.realpath(path)

    if absolute_git_path(git_dir) == absolute_git_path(common_dir):
        raise DirectiveError(
            "Execution checkout must be a registered linked worktree, not "
            "the repository's primary checkout")
    records = git_value(
        ["worktree", "list", "--porcelain"], "worktree registry")
    registered = [line[len("worktree "):]
                  for line in records.splitlines()
                  if line.startswith("worktree ")]
    if len([path for path in registered
            if os.path.realpath(path) == os.path.realpath(REPO_ROOT)]) != 1:
        raise DirectiveError(
            "Execution checkout is not uniquely registered as a Git "
            "worktree")
    actual_ref = git_value(
        ["symbolic-ref", "--quiet", "HEAD"], "branch")
    expected_ref = checkout["Branch"]
    if not expected_ref.startswith("refs/heads/"):
        expected_ref = "refs/heads/" + expected_ref
    if actual_ref != expected_ref:
        raise DirectiveError(
            "Execution checkout Branch mismatch: expected "
            + checkout["Branch"] + ", found " + actual_ref)
    actual_head = git_value(["rev-parse", "HEAD"], "base").lower()
    if actual_head != checkout["Base"].lower():
        raise DirectiveError(
            "Execution checkout Base mismatch: expected "
            + checkout["Base"] + ", found " + actual_head)


def _git(args_list):
    """Run one git command and return its stdout text (empty on failure).

    Arguments:
      args_list = the git arguments, e.g. ["log", "--oneline", "-1", "main"].
    """
    proc = subprocess.run(["git"] + args_list,
                          capture_output=True,
                          text=True,
                          cwd=REPO_ROOT)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def status_report():
    """Print current work saved by Git and the note files.

    No clipboard, no waiting. If the user needs help interpreting the output,
    they give it to the Architect. The user does not send it to the
    Implementer or Red Team.
    """
    print("== AI WORK STATUS ==\n")

    # Compare the Architect's work branch with main, the user's branch.
    main_tip = _git(["log", "--oneline", "-1", "main"])
    print("latest saved version on main: " + main_tip)
    branches = _git(["branch", "--list", "claude/*", "codex/*",
                     "--format=%(refname:short) %(committerdate:unix)"])
    working = ""
    newest = 0
    for line in branches.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[0].startswith("claude/"):
            continue
        if int(parts[1]) > newest:
            newest = int(parts[1])
            working = parts[0]
    if working:
        tip = _git(["log", "--oneline", "-1", working])
        ahead = _git(["rev-list", "--count", "main.." + working])
        print("Architect work branch:        " + tip)
        if ahead != "0":
            print("  -> " + ahead + " saved change(s) are not on main.")
            print("     After a GO verdict, only the Architect runs:")
            print("     git merge --no-edit " + working
                  + " && git push origin main")
        else:
            print("  -> main already includes this branch's saved changes.")

    # Show Red Team or second-Implementer branches and whether main or the
    # Architect work branch already includes them.
    print("\ncodex/* branches:")
    any_open = False
    for line in branches.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[0].startswith("codex/"):
            continue
        name = parts[0]
        merge_targets = ["main"]
        if working:
            merge_targets.append(working)
        is_integrated = False
        for target in merge_targets:
            merged = subprocess.run(
                ["git", "merge-base", "--is-ancestor", name, target],
                capture_output=True,
                cwd=REPO_ROOT)
            if merged.returncode == 0:
                is_integrated = True
                break
        if is_integrated:
            state = "integrated"
        else:
            state = "OPEN -- awaiting Architect audit/merge (or still in work)"
            any_open = True
        tip = _git(["log", "--oneline", "-1", name])
        print("  [" + state + "] " + tip)
    if not any_open:
        print("  (none open)")

    # Show the latest saved Architect decisions.
    print("\nlatest Architect records in ai/notes/gates-and-board.md:")
    gb = os.path.join(NOTES_DIR, "gates-and-board.md")
    if os.path.isfile(gb):
        heads = []
        with open(gb, encoding="utf-8") as f:
            for line in f:
                if line.startswith("## "):
                    heads.append(line.rstrip())
        for head in heads[-6:]:
            print("  " + head)

    # Show recent copies of handoff blocks saved by this tool.
    if os.path.isdir(RELAY_DIR):
        names = []
        for name in os.listdir(RELAY_DIR):
            path = os.path.join(RELAY_DIR, name)
            if name.endswith(".md") and os.path.isfile(path):
                names.append(name)
        names.sort()
        if names:
            print("\nrecent copied handoff records (supporting records only):")
            for name in names[-3:]:
                print("  " + os.path.relpath(RELAY_DIR, REPO_ROOT)
                      + "/" + name)

    print("\nNext action:")
    print("  1. If a codex/* branch says OPEN, give this status to the")
    print("     Architect for review.")
    print("  2. If saved changes are not on main, the Architect audits them")
    print("     and runs the printed Git commands only after a GO verdict.")
    print("  3. Otherwise, the work is idle. Give the next request to the")
    print("     Architect, who may start a validated --note run.")


def main():
    parser = argparse.ArgumentParser(
        description="copy approved Architect instructions between manual "
                    "web conversations")
    parser.add_argument("--status",
                        action="store_true",
                        help="show saved AI work, changes not yet on main, "
                             "and recent Architect records, then exit")
    parser.add_argument("--note",
                        required=False,
                        help="source note under ai/notes/ containing the "
                             "Architect's checked Implementation directive")
    parser.add_argument("--section",
                        default="",
                        help="optional exact section name; only "
                             "'Implementation directive' is valid")
    parser.add_argument("--mode",
                        choices=["redteam", "second-implementer"],
                        default=None,
                        help="confirm the role plan saved by the Architect; "
                             "this option cannot change that plan")
    parser.add_argument("--skip-redteam", "--no-red-team",
                        dest="skip_redteam",
                        action="store_true",
                        help="confirm that the Architect note chose only "
                             "Architect and Implementer; this option cannot "
                             "remove Red Team from another plan")
    parser.add_argument("--gate-cmd",
                        action="append",
                        default=[],
                        help="local check command (repeatable); replaces the "
                             "default check commands")
    parser.add_argument(
        "--max", metavar="characters",
        type=nonnegative_character_limit, default=None,
        help="character-change limit that the directive must match; when "
             "omitted, use MAILBOX_MAX_CHARACTERS if present, otherwise 0")
    parser.add_argument(
        "--severity", choices=DISCOVERY_SEVERITIES, default=None,
        help="confirm the discovery severity saved in the Architect note; "
             "this option cannot change that value")
    args = parser.parse_args()

    if args.max is not None and (not args.note or args.status):
        print("--max is valid only with a --note run")
        return 1
    role_confirmation_used = (
        args.mode is not None
        or args.skip_redteam
        or args.severity is not None)
    if role_confirmation_used and (not args.note or args.status):
        print("--mode, --skip-redteam, and --severity only confirm the "
              "Role plan in a --note run")
        return 1
    if args.status:
        status_report()
        return 0
    if not args.note:
        print("either --status or --note is required (see --help)")
        return 1
    if (args.section
            and args.section.strip().casefold() != "implementation directive"):
        print("--section may name only the validated 'Implementation "
              "directive' section")
        return 1
    try:
        expected_max = resolve_character_limit(cli_value=args.max)
    except DirectiveError as exc:
        print("refused character-change limit: " + str(exc))
        return 1
    try:
        note_path, note_display = resolve_note_path(args.note)
        directive = validate_directive_file(
            role="architect", path=note_path, expected_max=expected_max)
        verify_execution_checkout(
            checkout=directive["execution_checkout"])
    except DirectiveError as exc:
        print("refused incomplete Architect directive: " + str(exc))
        return 1
    role_plan = directive["role_plan"]
    if role_plan["uses_sol_as_implementer"]:
        expected_mode = "second-implementer"
    elif role_plan["uses_red_team"]:
        expected_mode = "redteam"
    else:
        expected_mode = None
    if args.mode is not None and args.mode != expected_mode:
        print("refused role confirmation: --mode " + args.mode
              + " does not match the Architect Role plan `"
              + role_plan["roles"] + "`")
        return 1
    if args.skip_redteam and role_plan["route"] != "two-role":
        print("refused role confirmation: --skip-redteam does not match the "
              "Architect Role plan `" + role_plan["roles"] + "`")
        return 1
    if args.severity is not None:
        if not role_plan["uses_red_team"]:
            print("refused severity confirmation: the Architect Role plan "
                  "does not include Red Team")
            return 1
        if args.severity != role_plan["discovery_severity"]:
            print("refused severity confirmation: --severity "
                  + args.severity + " does not match the Architect Role "
                  "plan " + role_plan["discovery_severity"])
            return 1
    inherited_severity = os.environ.get(DISCOVERY_SEVERITY_ENVIRONMENT)
    if (inherited_severity is not None
            and inherited_severity not in DISCOVERY_SEVERITIES):
        print("refused discovery severity: "
              + DISCOVERY_SEVERITY_ENVIRONMENT
              + " must be exactly high, medium, or low")
        return 1
    if (role_plan["uses_red_team"]
            and inherited_severity is not None
            and inherited_severity != role_plan["discovery_severity"]):
        print("refused discovery severity: Architect Role plan "
              + role_plan["discovery_severity"] + " does not match "
              + DISCOVERY_SEVERITY_ENVIRONMENT + " "
              + inherited_severity)
        return 1
    discovery_severity = role_plan["discovery_severity"]
    try:
        router_lock = acquire_router_lock()
    except RuntimeError as exc:
        print(str(exc))
        return 1
    seq = reserve_run_sequence()
    where = note_display + ', section "Implementation directive"'
    budget = directive["character_change_budget"]
    budget_prompt = (
        "Binding character-change budget: limit "
        + str(budget["limit"]) + " characters; planned maximum "
        + str(budget["planned_maximum"]) + " characters. Zero removes the "
        "size cap only; readable complete tested work remains mandatory.\n\n")
    role_prompt = (
        "Architect's validated role plan: " + role_plan["roles"] + ". "
        "Discovery severity: " + discovery_severity + ". The runner and "
        "human courier may not change this plan.\n\n")
    severity_prompt = (
        "User severity setting for any new Red Team ticket: "
        + discovery_severity + ". High means severe core harm, data loss, "
        "halted operation, or wrong science. Medium also permits a probable "
        "normal-operation bug but excludes improbable edge cases. Low "
        "permits any concrete discovered bug. Record Red Team severity, "
        "probable or improbable likelihood, likelihood evidence, and whether "
        "the finding meets this setting. "
        "The Architect accepts, upgrades, or downgrades the rating and "
        "makes the final GO or NO-GO decision.\n\n")
    total_steps = 4 if role_plan["uses_red_team"] else 3

    # [1] One execution owner receives this unit. Second-Implementer mode
    # assigns it to Sol INSTEAD OF Opus; it never duplicates one directive.
    if role_plan["uses_sol_as_implementer"]:
        implementer_prompt = (
            "### ARCHITECT_HANDOFF (relay)\n\n"
            + SECOND_IMPLEMENTER_MODE_SENTENCE + "\n\n"
            + role_prompt
            + budget_prompt
            + "The decision-complete Implementation directive is in "
            + where + ". Read .codex/REDTEAM_ROLE.md for this explicit mode "
            "switch, then .claude/OPUS_ROLE.md. Validate the directive and "
            "verify its Execution checkout before editing. Execute this unit "
            "as its only owner; do not perform a Red Team review. Append "
            "evidence under the note's sibling evidence heading and reply "
            "with IMPLEMENTER_HANDOFF.\n\n### ENDS\n")
        implementer_name = "Sol second Implementer"
        archive_name = "second-implementer"
    else:
        implementer_prompt = (
            "### ARCHITECT_HANDOFF (relay)\n\n"
            + role_prompt
            + budget_prompt
            + "The decision-complete Implementation directive for your next "
            "unit is in " + where + " of this repository. Read "
            ".claude/OPUS_ROLE.md, read that entry and its [[links]], run the "
            "directive check, verify its Execution checkout, then follow its "
            "ordered plan and reply with your IMPLEMENTER_HANDOFF block (a "
            "short return; append the full result under the sibling evidence "
            "heading first).\n\n### ENDS\n")
        implementer_name = "Opus"
        archive_name = "implementer"

    copy_to_clipboard(implementer_prompt)
    print("[1/" + str(total_steps) + "] " + implementer_name
          + " instruction copied -- paste it unchanged into that session.")
    implementer_block = wait_for_block(
        header="### IMPLEMENTER_HANDOFF:",
        last_copied=implementer_prompt)
    path = archive(seq, archive_name, implementer_block)
    print("      returned block saved -> " + path)

    # [2] Run local checks and save their exact output. The Architect still
    # reruns every check required by the directive before deciding.
    commands = list(args.gate_cmd if args.gate_cmd else DEFAULT_GATE_COMMANDS)
    if budget["limit"] > 0:
        guard = directive["ticket_change_guard"]
        guard_tool = guard["tool"]
        if not os.path.isabs(guard_tool):
            guard_tool = os.path.join(REPO_ROOT, guard_tool.lstrip("./"))
        commands.append(shlex.join([
            sys.executable,
            guard_tool,
            "--repo", guard["repo"],
            "--base", guard["base"],
            "--max", str(guard["max"]),
        ]))
    print("[2/" + str(total_steps) + "] running the local checks:")
    log_path, all_green = run_gates(commands=commands, seq=seq)
    print("      checks " + ("ALL PASS" if all_green else "NOT all green")
          + " -> " + log_path)

    redteam_block = ""
    if role_plan["uses_red_team"]:
        sol_prompt = (
            "### ARCHITECT_REDTEAM_HANDOFF (relay)\n\n"
            + role_prompt
            + budget_prompt
            + severity_prompt
            + "The named delta is specified in " + where + ". Read\n"
            ".codex/REDTEAM_ROLE.md and stay within that delta. The\n"
            "saved Implementer return is at " + path + " and\n"
            "the local check log is at " + log_path + ". A confirmed\n"
            "finding needs a validated, implementation-ready candidate\n"
            "Repair directive in ai/notes/; return it to the Architect in\n"
            "ARCHITECT_REDTEAM_HANDOFF, never directly to the Implementer.\n\n"
            "### ENDS\n")
        copy_to_clipboard(sol_prompt)
        print("[3/4] Red Team instruction copied -- paste it unchanged into "
              "the Sol session.")
        redteam_block = wait_for_block(
            header="### ARCHITECT_REDTEAM_HANDOFF:",
            last_copied=sol_prompt)
        sol_path = archive(seq, "sol", redteam_block)
        print("      returned block saved -> " + sol_path)

    # [3 or 4] Return every record to the Architect. The Architect reruns the
    # required checks; the router's log is supporting evidence, not a verdict.
    architect_prompt = (
        "### RELAY FOR AUDIT\n\n"
        + role_prompt
        + budget_prompt
        + (severity_prompt if redteam_block else "")
        + "Unit spec: " + where + "\n"
        "Implementer return (saved copy): " + path + "\n"
        + ("Red Team return (saved copy): " + sol_path + "\n"
           if redteam_block else "")
        + "Local check log: " + log_path + "\n\n"
        + "Local check summary: "
        + ("ALL PASS.\n" if all_green else
           "NOT all green. The Architect must inspect the failed command "
           "and issue NO-GO or new instructions; this tool does not "
           "close the ticket.\n")
        + "Review per your role file, including your own reruns of every\n"
        "required check. The saved blocks and check log support the review;\n"
        "they do not replace it.\n\n"
        "### ENDS\n")
    copy_to_clipboard(architect_prompt)
    print("[" + str(total_steps) + "/" + str(total_steps)
          + "] Architect return prompt copied -- paste it unchanged into "
          "the Architect session for the verdict.")
    release_router_lock(router_lock)
    return 0


if __name__ == "__main__":
    sys.exit(main())
