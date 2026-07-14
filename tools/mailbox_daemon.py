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
import threading
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
    # Absolute path: the user's conda shells resolve an OLDER claude binary
    # with a separate (logged-out) credential store; this one is the
    # logged-in v2.1.208 install (diagnosed 2026-07-14).
    "fable": ["/Users/vivianmiranda/.local/bin/claude", "-p",
              "--model", "claude-fable-5",
              "--permission-mode", "acceptEdits"],
    "opus": ["/Users/vivianmiranda/.local/bin/claude", "-p",
             "--model", "claude-opus-4-8",
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
PLACEHOLDER_MARKERS = ["<spec>", "<X>", "<section>", "<unit>",
                       "your message here"]

PREAMBLE = (
    "You are invoked headlessly by tools/mailbox_daemon.py (no human is\n"
    "watching this turn). Resolve your role per CLAUDE.md from the block\n"
    "below. The substance is in the notes/ entries the message cites --\n"
    "read them first. Do the work per your role file. End your turn by\n"
    "(1) writing your substance to the appropriate notes/ entry and\n"
    "(2) writing your outbound handoff block to a NEW file\n"
    "<seq>-to-<fable|opus|sol>.md using the next sequence number, INSIDE\n"
    "THIS EXACT DIRECTORY (your cwd may differ -- a relative notes/mailbox\n"
    "path is wrong unless it resolves here):\n"
    "    " + MAILBOX + "\n"
    "Merges and pushes to main remain\n"
    "the user's alone -- print a landing block in the note instead of\n"
    "running one.\n\n"
    "--- MESSAGE ---\n")


def next_seq():
    """Return the next zero-padded mailbox sequence number as a string.

    Scans EVERY directory under the mailbox (root, done/, failed/, any
    hand-made quarantine like hold/): a number parked anywhere is still
    claimed, and handing it out twice makes two messages look like one.
    """
    highest = 0
    pattern = os.path.join(MAILBOX, "**", "*.md")
    for path in glob.glob(pattern, recursive=True):
        name = os.path.basename(path)
        match = re.match(r"(\d+)[a-z]?-to-", name)
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
            # park it like a failed dispatch: a refusal's cause (the
            # unfilled body) persists until a human edits the file, so
            # leaving it in the mailbox would re-refuse it every poll.
            failed_dir = os.path.join(MAILBOX, "failed")
            os.makedirs(failed_dir, exist_ok=True)
            os.rename(path, os.path.join(failed_dir, name))
            print("REFUSED " + name + ": the body still contains the "
                  "template placeholder '" + marker + "' -- parked in "
                  "failed/; fill in the real text and requeue.")
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

    if proc.returncode != 0:
        # a failed dispatch is NOT done: park it in failed/ so it is never
        # silently consumed, and never hot-retried while the cause persists.
        # Requeue after fixing the cause:  mv notes/mailbox/failed/<f> notes/mailbox/
        failed_dir = os.path.join(MAILBOX, "failed")
        os.makedirs(failed_dir, exist_ok=True)
        os.rename(path, os.path.join(failed_dir, name))
        if "Not logged in" in proc.stdout:
            print("  !! the headless CLI is LOGGED OUT -- run `claude` in a "
                  "terminal, type /login, then requeue from failed/.")
        else:
            print("  !! dispatch FAILED -- message parked in failed/, see "
                  "the log above.")
        return False

    os.makedirs(DONE, exist_ok=True)
    try:
        os.rename(path, os.path.join(DONE, name))
    except FileNotFoundError:
        # someone quarantined the file by hand while its turn was in
        # flight (the 2026-07-14 hold/ intervention): the turn already
        # ran, so the message counts as handled wherever it now lives.
        print("  note: " + name + " was moved by hand mid-dispatch; "
              "leaving it where it is.")
    return True


def drain_lane(paths, dry_run):
    """Dispatch ONE agent's pending messages, in order (a worker body).

    Arguments:
      paths   = this agent's message files, already sorted by sequence.
      dry_run = True to print the would-be commands without running them.
    """
    for path in paths:
        dispatch(path=path, dry_run=dry_run)


def process_backlog(dry_run):
    """Dispatch the whole backlog: lanes in PARALLEL, each lane in order.

    The three agents are independent sessions, so Opus can execute a unit
    while Sol attacks another -- but two messages to the SAME agent must
    stay sequential (a lane is one conversation partner, not a pool), and
    two agents sharing a WORKING DIRECTORY must too: concurrent turns in
    one git tree race each other's index (the 2026-07-14 incident where a
    live edit was swept into another agent's commit). So the parallel unit
    is the cwd: Fable+Opus (same worktree) serialize; Sol runs alongside.

    Arguments:
      dry_run = True to print the would-be commands without running them.

    Returns:
      True when there was a backlog to process.
    """
    backlog = pending_messages()
    if not backlog:
        return False
    lanes = {}
    for path in backlog:
        name = os.path.basename(path)
        agent = re.match(r"\d+-to-(fable|opus|sol)\.md$", name).group(1)
        cwd = AGENT_CWD[agent]
        if cwd not in lanes:
            lanes[cwd] = []
        lanes[cwd].append(path)
    workers = []
    for cwd in sorted(lanes):
        worker = threading.Thread(target=drain_lane,
                                  kwargs={"paths": lanes[cwd],
                                          "dry_run": dry_run})
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
    return True


def send(agent, text):
    """Drop a new message into the mailbox (the loop's entry point).

    Arguments:
      agent = "fable", "opus", or "sol".
      text  = the routing summary (point at notes/; do not inline specs).
    """
    os.makedirs(MAILBOX, exist_ok=True)
    # O_EXCL claims the number atomically: a concurrent sender that
    # computed the same sequence loses the race and retries on the next.
    for _ in range(20):
        path = os.path.join(MAILBOX, next_seq() + "-to-" + agent + ".md")
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        print("queued " + path)
        return
    print("could not claim a sequence number after 20 tries -- "
          "is something flooding the mailbox?")


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
    parser.add_argument("--ping", metavar="AGENT",
                        choices=["fable", "opus", "sol"],
                        help="queue a transport-confirmation ping to this "
                             "agent (its reply lands as a -to-user.md file "
                             "the daemon never dispatches)")
    parser.add_argument("--unit", default="",
                        help="the message text for --send (a routing "
                             "summary pointing at notes/)")
    args = parser.parse_args()

    if args.ping:
        ping_text = (
            "RELAY CONFIRMATION PING for " + args.ping + ". This is a "
            "transport test only; no unit is assigned and no repository "
            "file may change. Reply by creating ONE new file,\n"
            "notes/mailbox/<next-sequence>-to-user.md, whose entire body "
            "is one line:\n\n"
            "    PONG " + args.ping + " from <your model name>\n\n"
            "Then stop. (Files addressed -to-user are read by the human; "
            "the daemon never dispatches them.)\n")
        send(agent=args.ping, text=ping_text)
        return 0

    if args.send:
        if not args.unit:
            print("--send needs --unit with the routing-summary text")
            return 1
        send(agent=args.send, text=args.unit)
        return 0

    if args.dry_run or args.once:
        if not process_backlog(dry_run=args.dry_run):
            print("mailbox empty")
        return 0

    if args.watch:
        # single-instance lock: a second concurrent watcher double-dispatches
        # the same messages (the 2026-07-14 zombie-race incident). The lock
        # file holds the owner's pid; a dead owner's lock is reclaimed.
        lock_path = os.path.join(MAILBOX, ".watch.lock")
        os.makedirs(MAILBOX, exist_ok=True)
        if os.path.isfile(lock_path):
            with open(lock_path, encoding="utf-8") as f:
                old_pid = f.read().strip()
            alive = subprocess.run(["kill", "-0", old_pid],
                                   capture_output=True)
            if alive.returncode == 0:
                print("another watch is already running (pid " + old_pid
                      + ") -- refusing to start a second one.")
                return 1
            print("reclaiming a dead watcher's lock (pid " + old_pid + ")")
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        print("watching " + MAILBOX + " (Ctrl-C to stop; safe only "
              "BETWEEN dispatches -- killing a dispatch mid-flight dooms "
              "the agent's turn)")
        # a daemon fix is a no-op for the loop already running (the
        # 2026-07-14 placeholder incident): watch our own source and
        # exit when it changes, so stale code can never keep dispatching.
        # Exiting (not self-reloading) is deliberate -- a restart is one
        # keystroke and never picks up a half-saved edit.
        source_stamp = os.path.getmtime(os.path.abspath(__file__))
        try:
            while True:
                process_backlog(dry_run=False)
                if os.path.getmtime(os.path.abspath(__file__)) \
                        != source_stamp:
                    print("daemon source changed on disk -- exiting so "
                          "the next start runs it (relaunch --watch).")
                    return 0
                time.sleep(20)
        finally:
            if os.path.isfile(lock_path):
                os.remove(lock_path)

    print("choose one of --dry-run / --once / --watch / --send (see --help)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
