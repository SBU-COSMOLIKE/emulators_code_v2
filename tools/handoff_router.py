#!/usr/bin/env python3
"""Clipboard relay for the three-agent loop (Fable / Opus / Sol web sessions).

The user runs each agent in its own web session (subscription plans, no API).
This router removes the copy/paste bookkeeping while KEEPING the program's
communication rules intact:

  - NOTES-FIRST: the prompts this script copies are ROUTING SUMMARIES that
    point at `notes/` entries. The substance always lives in the note the
    agents themselves write. Captured chat blocks are archived under
    `notes/relay/` as TRANSPORT COPIES ONLY -- they are never the source of
    record; the agent-written note is.
  - GATE INTEGRITY: the router runs the validation gates LOCALLY, on this
    machine, and archives the raw log. The agents never get to invent that
    output. The Architect still re-runs evidence per its role file; the
    router's log is corroborating input, not the audit.
  - ROLE FILES GOVERN: the prompts never restate role rules. Each session
    resolves its role from the handoff block it receives (see CLAUDE.md).
    The one required mode sentence -- the backup-Implementer declaration --
    is inserted verbatim when --mode backup is passed.

Flow (one unit per run):

    blueprint note ready
          |
          v
    [1] copy the Opus routing prompt      -> paste into the Opus session
    [2] capture IMPLEMENTER_HANDOFF       <- copy the block from Opus
    [3] run the local gates, archive log
    [4] copy the Sol routing prompt       -> paste into the Sol session
        (red-team mode, or backup-Implementer mode with --mode backup;
         skipped entirely with --skip-redteam)
    [5] capture the Sol handoff           <- copy the block from Sol
    [6] copy the Fable routing prompt     -> paste into the Fable session
          |
          v
    Fable audits per its role file (its own re-runs), emits the verdict.

Usage:

    python tools/handoff_router.py --note notes/gates-and-board.md \\
        --section "BACKUP-IMPLEMENTER ASSIGNMENT 1" --mode backup
    python tools/handoff_router.py --note notes/<spec>.md            # full loop
    python tools/handoff_router.py --note notes/<spec>.md --skip-redteam

The gate commands default to the board's cheap surfaces and can be replaced
with the unit's own validation gate:

    --gate-cmd "PYTHONPATH=. <cocoa-python> gates/checks/<child>.py"

Lost between manual handoffs? Run the status sweep -- no clipboard, no
waiting, just the current program state read mechanically from git and the
notes:

    python tools/handoff_router.py --status
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

NOTES_DIR = "notes"
RELAY_DIR = os.path.join(NOTES_DIR, "relay")

COCOA_PYTHON = ("/Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa"
                "/.local/bin/python")

DEFAULT_GATE_COMMANDS = [
    COCOA_PYTHON + " -m compileall -q gates emulator",
    "PYTHONPATH=. " + COCOA_PYTHON + " gates/run_board.py --list",
    "PYTHONPATH=. " + COCOA_PYTHON + " gates/checks/board_selftest.py",
]

BACKUP_MODE_SENTENCE = ("OpenAI Sol -- this is a role as backup Implementer "
                        "for this unit.")


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
        return proc.stdout.decode("utf-8", errors="replace")
    import pyperclip
    return pyperclip.paste()


def wait_for_block(marker, last_copied):
    """Block until the clipboard holds a NEW text containing the marker.

    The comparison baseline is the text THIS script last copied, so the
    routing prompt itself (which may mention the marker) can never be
    captured as the response.

    Arguments:
      marker      = the substring that identifies the expected block,
                    e.g. "IMPLEMENTER_HANDOFF".
      last_copied = the exact string this script most recently placed on
                    the clipboard.

    Returns:
      the captured clipboard text.
    """
    print("... waiting for a copied block containing '" + marker + "'")
    while True:
        current = read_clipboard()
        if current != last_copied and marker in current:
            return current
        time.sleep(0.5)


def archive(seq, name, text):
    """Write one transport copy under notes/relay/ and return its path.

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
                "     Archived by tools/handoff_router.py. -->\n\n")
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    return path


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
                              text=True)
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
                          text=True)
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

    # 2. red-team / backup branches: integrated or awaiting Fable?
    print("\ncodex/* branches:")
    any_open = False
    for line in branches.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[0].startswith("codex/"):
            continue
        name = parts[0]
        merged = subprocess.run(
            ["git", "merge-base", "--is-ancestor", name, working],
            capture_output=True)
        if merged.returncode == 0:
            state = "integrated"
        else:
            state = "OPEN -- awaiting Fable audit/merge (or still in work)"
            any_open = True
        tip = _git(["log", "--oneline", "-1", name])
        print("  [" + state + "] " + tip)
    if not any_open:
        print("  (none open)")

    # 3. the newest adjudication records (their titles carry the verdicts).
    print("\nlatest records in notes/gates-and-board.md:")
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
        names = sorted(os.listdir(RELAY_DIR))
        if names:
            print("\nnewest relay transport copies (non-authoritative):")
            for name in names[-3:]:
                print("  notes/relay/" + name)

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
                        help="notes/ file carrying the ARCHITECT handoff "
                             "(the substance; the prompt only points here)")
    parser.add_argument("--section",
                        default="",
                        help="section title inside the note (optional)")
    parser.add_argument("--mode",
                        choices=["redteam", "backup"],
                        default="redteam",
                        help="Sol's mode: adversarial (default) or backup "
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
    if not os.path.isfile(args.note):
        print("no such note: " + args.note)
        return 1
    seq = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    where = args.note
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
    opus_block = wait_for_block(marker="IMPLEMENTER_HANDOFF",
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
        if args.mode == "backup":
            mode_line = BACKUP_MODE_SENTENCE + "\n\n"
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
            "notes/ first.\n\n"
            "### ENDS\n")
        copy_to_clipboard(sol_prompt)
        print("[3/4] Sol routing prompt copied (" + args.mode + " mode) -- "
              "paste it into the Sol session.")
        sol_block = wait_for_block(marker="REDTEAM_HANDOFF",
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
