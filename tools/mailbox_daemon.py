#!/usr/bin/env python3
"""File mailbox + headless dispatch: the loop runs with NO copy/paste.

The medium is a directory of message files; the wake-up is this daemon
invoking each agent's CLI headlessly when a message addressed to it appears.

    notes/mailbox/NNN-to-fable.md      -> dispatched to the Fable CLI
    notes/mailbox/NNN-to-opus.md       -> dispatched to the Opus CLI
    notes/mailbox/NNN-to-sol.md        -> dispatched to the Sol (Codex) CLI
    notes/mailbox/done/                -> processed messages move here

A message file is a ROUTING SUMMARY (the notes-first rule holds: the
substance lives in the `notes/` entry the message cites). Each dispatched
agent is asked to end its turn by (1) writing its substance to `notes/` and
(2) dropping its outbound handoff as the NEXT numbered message file — so the
loop continues without a human relay.

What stays manual, on purpose:
  - merges/pushes to main are ALWAYS the user's (the daemon never runs git);
  - the daemon only dispatches messages; it never edits code or notes itself;
  - every dispatch's full CLI output is archived under notes/relay/.

The CLI commands below are configurable. `claude -p` runs one headless turn
against the subscription; the session needs enough tool permission to work
unattended (set via the harness settings or the flags here). Adjust SOL_CMD
to the Codex CLI's headless form installed on this machine.

Usage:
    python tools/mailbox_daemon.py --dry-run        # show what would run
    python tools/mailbox_daemon.py --once           # process backlog, exit
    python tools/mailbox_daemon.py --watch          # poll every 20 s
    python tools/mailbox_daemon.py --send opus --unit "notes/<spec>.md ..."
                                                    # drop a first message
"""

import argparse
import datetime
import glob
import os
import re
import subprocess
import sys
import time

# All work and all mailbox traffic live in the SHARED WORKTREE (the branch
# the agents actually develop on), never the bare main-repo checkout.
WORKTREE = ("/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2"
            "/.claude/worktrees/amazing-keller-e798b6")

MAILBOX = os.path.join(WORKTREE, "notes", "mailbox")
DONE = os.path.join(MAILBOX, "done")
RELAY_DIR = os.path.join(WORKTREE, "notes", "relay")

# One headless command per lane. Each receives the message text as its
# prompt argument (appended by dispatch()). --permission-mode acceptEdits
# lets a headless turn edit files without a human at the prompt; shell
# commands still obey the project permission settings (git push stays
# deniable there -- the user owns that policy file).
AGENT_COMMANDS = {
    "fable": ["claude", "-p", "--model", "claude-fable-5",
              "--permission-mode", "acceptEdits"],
    "opus": ["claude", "-p", "--model", "claude-opus-4-8",
             "--permission-mode", "acceptEdits"],
    # Verified by the red team's read-only probe (codex-cli 0.144.2; the
    # conventions note records the probe): workspace-write sandbox rooted at
    # the repo, which contains every worktree Sol works in.
    "sol": ["/Applications/ChatGPT.app/Contents/Resources/codex",
            "exec",
            "--model", "gpt-5.6-sol",
            "--sandbox", "workspace-write",
            "--cd",
            "/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2"],
}

# The working directory each dispatched agent starts in. Sol's command
# carries its own --cd, so its subprocess cwd just matches that root.
AGENT_CWD = {
    "fable": WORKTREE,
    "opus": WORKTREE,
    "sol": "/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2",
}

# A message still carrying template placeholders has no job in it; refuse
# it instead of burning a live headless turn (learned from dispatch 0001).
PLACEHOLDER_MARKERS = ["<spec>", "<X>", "<section>", "<unit>"]

PREAMBLE = (
    "You are invoked headlessly by tools/mailbox_daemon.py (no human is\n"
    "watching this turn). Resolve your role per CLAUDE.md from the block\n"
    "below. The substance is in the notes/ entries the message cites --\n"
    "read them first. Do the work per your role file. End your turn by\n"
    "(1) writing your substance to the appropriate notes/ entry and\n"
    "(2) writing your outbound handoff block to a NEW file\n"
    "notes/mailbox/<seq>-to-<fable|opus|sol>.md using the next sequence\n"
    "number (see the mailbox directory). Merges and pushes to main remain\n"
    "the user's alone -- print a landing block in the note instead of\n"
    "running one.\n\n"
    "--- MESSAGE ---\n")


def next_seq():
    """Return the next zero-padded mailbox sequence number as a string."""
    highest = 0
    pattern = os.path.join(MAILBOX, "*.md")
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        match = re.match(r"(\d+)-to-", name)
        if match:
            value = int(match.group(1))
            if value > highest:
                highest = value
    for path in glob.glob(os.path.join(DONE, "*.md")):
        name = os.path.basename(path)
        match = re.match(r"(\d+)-to-", name)
        if match:
            value = int(match.group(1))
            if value > highest:
                highest = value
    return "%04d" % (highest + 1)


def pending_messages():
    """Return the sorted list of unprocessed message paths."""
    found = []
    for path in sorted(glob.glob(os.path.join(MAILBOX, "*.md"))):
        name = os.path.basename(path)
        if re.match(r"\d+-to-(fable|opus|sol)\.md$", name):
            found.append(path)
    return found


def dispatch(path, dry_run):
    """Send one message file to its addressee's headless CLI.

    Arguments:
      path    = the mailbox message file.
      dry_run = True to print the would-be command without running it.

    Returns:
      True when the dispatch ran (or would run) cleanly.
    """
    name = os.path.basename(path)
    agent = re.match(r"\d+-to-(fable|opus|sol)\.md$", name).group(1)
    with open(path, encoding="utf-8") as f:
        message = f.read()
    for marker in PLACEHOLDER_MARKERS:
        if marker in message:
            print("REFUSED " + name + ": the body still contains the "
                  "template placeholder '" + marker + "' -- fill in the "
                  "real note path/section and re-send.")
            return False
    command = AGENT_COMMANDS[agent] + [PREAMBLE + message]

    if dry_run:
        print("[dry-run] would dispatch " + name + " -> "
              + " ".join(AGENT_COMMANDS[agent])
              + "  (cwd " + AGENT_CWD[agent] + ")")
        return True

    print("dispatching " + name + " -> " + agent + " ...")
    proc = subprocess.run(command,
                          capture_output=True,
                          text=True,
                          cwd=AGENT_CWD[agent])
    os.makedirs(RELAY_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(RELAY_DIR, stamp + "-dispatch-" + agent + ".log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(AGENT_COMMANDS[agent]) + " <message>\n")
        f.write("rc=" + str(proc.returncode) + "\n--- stdout ---\n")
        f.write(proc.stdout)
        f.write("\n--- stderr ---\n")
        f.write(proc.stderr)
    print("  rc=" + str(proc.returncode) + "  log -> " + log_path)
    # show the reply's tail on the terminal so activity is visible live.
    reply_lines = proc.stdout.strip().splitlines()
    for line in reply_lines[-8:]:
        print("  | " + line)

    os.makedirs(DONE, exist_ok=True)
    os.rename(path, os.path.join(DONE, name))
    return proc.returncode == 0


def send(agent, text):
    """Drop a new message into the mailbox (the loop's entry point).

    Arguments:
      agent = "fable", "opus", or "sol".
      text  = the routing summary (point at notes/; do not inline specs).
    """
    os.makedirs(MAILBOX, exist_ok=True)
    path = os.path.join(MAILBOX, next_seq() + "-to-" + agent + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    print("queued " + path)


def main():
    parser = argparse.ArgumentParser(
        description="file mailbox + headless dispatch for the agent loop")
    parser.add_argument("--dry-run", action="store_true",
                        help="show pending dispatches without running them")
    parser.add_argument("--once", action="store_true",
                        help="process the current backlog and exit")
    parser.add_argument("--watch", action="store_true",
                        help="poll the mailbox every 20 seconds")
    parser.add_argument("--send", metavar="AGENT",
                        choices=["fable", "opus", "sol"],
                        help="queue a message to this agent and exit")
    parser.add_argument("--unit", default="",
                        help="the message text for --send (a routing "
                             "summary pointing at notes/)")
    args = parser.parse_args()

    if args.send:
        if not args.unit:
            print("--send needs --unit with the routing-summary text")
            return 1
        send(agent=args.send, text=args.unit)
        return 0

    if args.dry_run or args.once:
        backlog = pending_messages()
        if not backlog:
            print("mailbox empty")
            return 0
        for path in backlog:
            dispatch(path=path, dry_run=args.dry_run)
        return 0

    if args.watch:
        print("watching " + MAILBOX + " (Ctrl-C to stop)")
        while True:
            for path in pending_messages():
                dispatch(path=path, dry_run=False)
            time.sleep(20)

    print("choose one of --dry-run / --once / --watch / --send (see --help)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
