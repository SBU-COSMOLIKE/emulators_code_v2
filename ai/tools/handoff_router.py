#!/usr/bin/env python3
"""Clipboard relay for the three-agent loop (Fable / Opus / Sol web sessions).

The user runs each agent in its own web session (subscription plans, no API).
This router removes the copy/paste bookkeeping while KEEPING the program's
communication rules intact:

  - NOTES-FIRST: the prompts this script copies are ROUTING SUMMARIES that
    point at `ai/notes/` entries. The substance always lives in the note the
    agents themselves write. Captured chat blocks are archived under
    `ai/notes/relay/` as TRANSPORT COPIES ONLY -- they are never the source of
    record; the agent-written note is.
  - GATE INTEGRITY: the router runs the validation gates LOCALLY, on this
    machine, and archives the raw log. The agents never get to invent that
    output. The Architect still re-runs evidence per its role file; the
    router's log is corroborating input, not the audit.
  - ROLE FILES GOVERN: the prompts never restate role rules. Each session
    resolves its role from the handoff block it receives (see CLAUDE.md).
    The one required mode sentence -- the second-Implementer declaration --
    is inserted verbatim when --mode second-implementer is passed.

Flow (one unit per run):

    blueprint note ready
          |
          v
    [1] copy the Opus routing prompt      -> paste into the Opus session
    [2] capture IMPLEMENTER_HANDOFF       <- copy the block from Opus
    [3] run the local gates, archive log
    [4] copy the Sol routing prompt       -> paste into the Sol session
        (red-team mode, or second-Implementer mode with
         --mode second-implementer;
         skipped entirely with --skip-redteam)
    [5] capture the Sol handoff           <- copy the block from Sol
    [6] copy the Fable routing prompt     -> paste into the Fable session
          |
          v
    Fable audits per its role file (its own re-runs), emits the verdict.

Usage:

    python ai/tools/handoff_router.py --note ai/notes/gates-and-board.md \\
        --section "SECOND-IMPLEMENTER ASSIGNMENT 1" \\
        --mode second-implementer
    python ai/tools/handoff_router.py --note ai/notes/<spec>.md            # full loop
    python ai/tools/handoff_router.py --note ai/notes/<spec>.md --skip-redteam

The gate commands default to the board's cheap surfaces and can be replaced
with the unit's own validation gate:

    --gate-cmd "PYTHONPATH=. <cocoa-python> ai/gates/checks/<child>.py"

Lost between manual handoffs? Run the status sweep -- no clipboard, no
waiting, just the current program state read mechanically from git and the
notes:

    python ai/tools/handoff_router.py --status
"""

import argparse
import datetime
import fcntl
import os
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(AI_ROOT)
NOTES_DIR = os.path.join(AI_ROOT, "notes")
RELAY_DIR = os.path.join(NOTES_DIR, "relay")
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
    routing prompt itself cannot be captured as the response. Requiring a
    Markdown heading also prevents ordinary prose that mentions the handoff
    token from being mistaken for a return block.

    Arguments:
      header      = the complete heading prefix that identifies the block,
                    e.g. "### IMPLEMENTER_HANDOFF:".
      last_copied = the exact string this script most recently placed on
                    the clipboard.

    Returns:
      the captured clipboard text.
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
    """Resolve a note argument from the router's repository, not the shell.

    Arguments:
      note = an absolute path or a repository-relative note path.

    Returns:
      ``(path, display_path)`` with an absolute path for I/O and a compact
      repository-relative path for prompts when the note is inside the repo.
    """
    if os.path.isabs(note):
        path = os.path.abspath(note)
    else:
        path = os.path.abspath(os.path.join(REPO_ROOT, note))
    try:
        inside_repo = os.path.commonpath([REPO_ROOT, path]) == REPO_ROOT
    except ValueError:
        inside_repo = False
    if inside_repo:
        display_path = os.path.relpath(path, REPO_ROOT)
    else:
        display_path = path
    return (path, display_path)


def archive(seq, name, text):
    """Write one transport copy under ai/notes/relay/ and return its path.

    Arguments:
      seq  = the run sequence stamp (shared by all files of this run).
      name = short role tag for the filename ("implementer", "sol", ...).
      text = the captured block or log text.
    """
    os.makedirs(RELAY_DIR, exist_ok=True)
    path = os.path.join(RELAY_DIR, seq + "-" + name + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!-- TRANSPORT COPY (non-authoritative). The source of\n"
                "     record is the agent-written note this block cites.\n"
                "     Archived by ai/tools/handoff_router.py. -->\n\n")
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    return os.path.relpath(path, REPO_ROOT)


def run_gates(commands, seq):
    """Run the validation gates locally; archive the full log.

    The console shows one verdict line per command (essential-only house
    rule); the complete streams go to the relay log file.

    Arguments:
      commands = list of shell command strings to run from the repo root.
      seq      = the run sequence stamp for the log filename.

    Returns:
      (log_path, all_green) -- the archive path and whether every command
      exited zero.
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
    """Print where the program stands, read mechanically from git + notes.

    No clipboard, no waiting. The output is itself a valid re-orientation
    text: paste it into any of the three sessions and the agent can pick up
    from the notes it names.
    """
    print("== HANDOFF STATUS (mechanical sweep) ==\n")

    # 1. the working branch vs main: is a landing block pending?
    main_tip = _git(["log", "--oneline", "-1", "main"])
    print("main tip:            " + main_tip)
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
        print("working branch:      " + tip)
        if ahead != "0":
            print("  -> " + ahead + " commit(s) not on main. Landing block:")
            print("     git merge --no-edit " + working
                  + " && git push origin main")
        else:
            print("  -> main is current; no landing block pending.")

    # 2. red-team / second-Implementer branches: integrated or awaiting Fable?
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
            state = "OPEN -- awaiting Fable audit/merge (or still in work)"
            any_open = True
        tip = _git(["log", "--oneline", "-1", name])
        print("  [" + state + "] " + tip)
    if not any_open:
        print("  (none open)")

    # 3. the newest adjudication records (their titles carry the verdicts).
    print("\nlatest records in ai/notes/gates-and-board.md:")
    gb = os.path.join(NOTES_DIR, "gates-and-board.md")
    if os.path.isfile(gb):
        heads = []
        with open(gb, encoding="utf-8") as f:
            for line in f:
                if line.startswith("## "):
                    heads.append(line.rstrip())
        for head in heads[-6:]:
            print("  " + head)

    # 4. the newest relay transport copies, if the router has run.
    if os.path.isdir(RELAY_DIR):
        names = []
        for name in os.listdir(RELAY_DIR):
            path = os.path.join(RELAY_DIR, name)
            if name.endswith(".md") and os.path.isfile(path):
                names.append(name)
        names.sort()
        if names:
            print("\nnewest relay transport copies (non-authoritative):")
            for name in names[-3:]:
                print("  " + os.path.relpath(RELAY_DIR, REPO_ROOT)
                      + "/" + name)

    print("\nNext action, in order of precedence:")
    print("  1. any OPEN codex branch above -> relay its handoff (or this")
    print("     status text) to the Fable session for audit + merge.")
    print("  2. a pending landing block -> run it (it is printed above).")
    print("  3. otherwise -> the loop is idle; start the next unit with")
    print("     --note, or paste this status into any session and ask.")


def main():
    parser = argparse.ArgumentParser(
        description="Clipboard relay for the Fable/Opus/Sol loop")
    parser.add_argument("--status",
                        action="store_true",
                        help="print the mechanical program status (branches, "
                             "pending landing block, latest records) and exit")
    parser.add_argument("--note",
                        required=False,
                        help="ai/notes/ file carrying the ARCHITECT handoff "
                             "(the substance; the prompt only points here)")
    parser.add_argument("--section",
                        default="",
                        help="section title inside the note (optional)")
    parser.add_argument("--mode",
                        choices=["redteam", "second-implementer"],
                        default="redteam",
                        help="Sol's mode: adversarial (default) or second "
                             "Implementer (inserts the explicit declaration)")
    parser.add_argument("--skip-redteam",
                        action="store_true",
                        help="route Opus -> gates -> Fable, no Sol step")
    parser.add_argument("--gate-cmd",
                        action="append",
                        default=[],
                        help="validation command (repeatable); replaces the "
                             "default board surfaces")
    args = parser.parse_args()

    if args.status:
        status_report()
        return 0
    if not args.note:
        print("either --status or --note is required (see --help)")
        return 1
    note_path, note_display = resolve_note_path(args.note)
    if not os.path.isfile(note_path):
        print("no such note: " + note_path)
        return 1
    try:
        router_lock = acquire_router_lock()
    except RuntimeError as exc:
        print(str(exc))
        return 1
    seq = reserve_run_sequence()
    where = note_display
    if args.section:
        where += ' , section "' + args.section + '"'

    # [1] Opus routing prompt: a pointer, not a payload (notes-first).
    opus_prompt = (
        "### ARCHITECT_HANDOFF (relay)\n\n"
        "The blueprint for your next unit is in " + where + " of this\n"
        "repository. Read that entry (and the [[links]] it cites), execute\n"
        "per your role file, and reply with your IMPLEMENTER_HANDOFF block\n"
        "(a routing summary; your substance goes into the same note first).\n\n"
        "### ENDS\n")
    copy_to_clipboard(opus_prompt)
    print("[1/4] Opus routing prompt copied -- paste it into the Opus "
          "session.")

    # [2] capture the Implementer's return.
    opus_block = wait_for_block(header="### IMPLEMENTER_HANDOFF:",
                                last_copied=opus_prompt)
    path = archive(seq, "implementer", opus_block)
    print("      captured -> " + path)

    # [3] objective local gates (the anti-hallucination anchor).
    commands = args.gate_cmd if args.gate_cmd else DEFAULT_GATE_COMMANDS
    print("[2/4] running the local validation gates:")
    log_path, all_green = run_gates(commands=commands, seq=seq)
    print("      gates " + ("ALL PASS" if all_green else "NOT all green")
          + " -> " + log_path)

    sol_block = ""
    if not args.skip_redteam:
        # [4] Sol routing prompt; the mode sentence is the ONLY inline rule.
        if args.mode == "second-implementer":
            mode_line = SECOND_IMPLEMENTER_MODE_SENTENCE + "\n\n"
        else:
            mode_line = ""
        sol_prompt = (
            "### ARCHITECT_REDTEAM_HANDOFF (relay)\n\n"
            + mode_line +
            "The unit under review is specified in " + where + ". The\n"
            "Implementer's return block is archived at " + path + " (a\n"
            "transport copy; the agent-written note is the record). The\n"
            "router's local gate log is at " + log_path + ".\n"
            "Work per your role file and reply with your\n"
            "ARCHITECT_REDTEAM_HANDOFF block; write your substance under\n"
            "ai/notes/ first.\n\n"
            "### ENDS\n")
        copy_to_clipboard(sol_prompt)
        print("[3/4] Sol routing prompt copied (" + args.mode + " mode) -- "
              "paste it into the Sol session.")
        sol_block = wait_for_block(
            header="### ARCHITECT_REDTEAM_HANDOFF:",
            last_copied=sol_prompt)
        sol_path = archive(seq, "sol", sol_block)
        print("      captured -> " + sol_path)

    # [5] Fable routing prompt: point at everything; the audit is Fable's
    #     own (its role file requires its own re-runs -- the router's log is
    #     corroborating input, never a substitute).
    fable_prompt = (
        "### RELAY FOR AUDIT\n\n"
        "Unit spec: " + where + "\n"
        "Implementer return (transport copy): " + path + "\n"
        + ("Sol return (transport copy): " + sol_path + "\n"
           if sol_block else "")
        + "Router's local gate log: " + log_path + "\n\n"
        "Audit per your role file -- including your own re-runs of the\n"
        "evidence. The archived blocks and the gate log are inputs, not\n"
        "the audit.\n\n"
        "### ENDS\n")
    copy_to_clipboard(fable_prompt)
    print("[4/4] Fable routing prompt copied -- paste it into the Fable "
          "session for the verdict.")
    release_router_lock(router_lock)
    return 0


if __name__ == "__main__":
    sys.exit(main())
