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
agent with a relayable result is asked to end its turn by (1) writing its
substance to `notes/` and (2) dropping its outbound handoff as the NEXT
numbered message file, so the loop continues without a human relay. An
inbound whose binding instruction explicitly says TERMINAL and no reply is
owed ends without an outbound; ambiguity follows the ordinary outbound rule.

What stays manual, on purpose:
  - merges/pushes to main are ALWAYS the user's (the daemon never runs git);
  - the daemon only dispatches messages; it never edits code or notes itself;
  - every dispatch's full CLI output is archived under notes/relay/.

Every path the daemon uses -- the mailbox, the relay logs, the working
directory each agent starts in -- is DERIVED from this file's own location
(it lives at <worktree>/tools/), so a clone on another computer runs
unedited and the worktree you launch it from is the one it coordinates.
AGENT_COMMANDS, the CLI binary paths, is the one machine-specific block.
`claude -p` runs one headless turn against the subscription; the session
needs enough tool permission to work unattended (set via the harness
settings or the flags there).

Usage:
    python tools/mailbox_daemon.py --help           # all options + defaults
    python tools/mailbox_daemon.py --dry-run        # show what would run
    python tools/mailbox_daemon.py --once           # process backlog, exit
    python tools/mailbox_daemon.py --watch          # poll every 20 s
    python tools/mailbox_daemon.py --send opus --unit "notes/<spec>.md ..."
                                                    # drop a first message
    python tools/mailbox_daemon.py --watch --opus-effort high
                                                    # dial one agent's effort
        --fable-effort / --opus-effort take low|medium|high|xhigh|max
        (claude CLI; defaults xhigh and max); --sol-effort takes
        minimal|low|medium|high|xhigh (codex CLI; default xhigh)
    python tools/mailbox_daemon.py --watch --dispatch-timeout 90
                                                    # allow longer turns
    python tools/mailbox_daemon.py --watch --claude-context 400000 \
                                           --sol-context 300000
                                                    # context budgets: a turn
        compacts (summarizes its own history and continues) whenever its
        live context reaches the budget; --claude-context covers Fable
        and Opus, --sol-context covers Sol; both default to 500000
"""

import argparse
import datetime
import fcntl
import glob
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

# All work and all mailbox traffic live in the SHARED WORKTREE (the branch
# the agents actually develop on), never the bare main-repo checkout. That
# worktree is DERIVED, never configured: this file lives at
# <worktree>/tools/mailbox_daemon.py, so the directory above tools/ is the
# worktree root. A clone on a new machine therefore runs unedited, and the
# worktree you launch the watch FROM is the one whose mailbox it watches.
WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_root_of(worktree):
    """Return the repository root that owns a given worktree directory.

    A Claude Code worktree sits at <repo>/.claude/worktrees/<name>, so the
    repository is three directories up. When the daemon is instead run from
    an ordinary checkout (no .claude/worktrees/ segment above it), that
    checkout IS the repository and is returned unchanged.

    Arguments:
      worktree = the worktree root, i.e. the directory holding tools/.

    Returns:
      The absolute path of the repository root.
    """
    worktrees_dir = os.path.dirname(worktree)          # <repo>/.claude/worktrees
    claude_dir = os.path.dirname(worktrees_dir)        # <repo>/.claude
    if (os.path.basename(worktrees_dir) == "worktrees"
            and os.path.basename(claude_dir) == ".claude"):
        return os.path.dirname(claude_dir)
    return worktree


REPO_ROOT = repo_root_of(worktree=WORKTREE)

MAILBOX = os.path.join(WORKTREE, "notes", "mailbox")
DONE = os.path.join(MAILBOX, "done")
RELAY_DIR = os.path.join(WORKTREE, "notes", "relay")

# THE ONE MACHINE-SPECIFIC BLOCK IN THIS FILE. Everything else derives from
# the daemon's own location, so a fresh clone runs unedited; these CLI binary
# paths cannot be derived, because they depend on where each vendor's CLI is
# installed on this computer. On a new machine, edit the binary paths here and
# nothing else (`which claude` and `which codex` find them).
#
# One headless command per lane. Each receives the message text as its
# prompt argument (appended by dispatch()). --permission-mode acceptEdits
# lets a headless turn edit files without a human at the prompt; shell
# commands still obey the project permission settings (git push stays
# deniable there -- the user owns that policy file).
# The reasoning-effort levels each CLI accepts, and the defaults the
# loop runs at when --watch is launched with no effort flags
# (USER 2026-07-14): Fable audits at "xhigh"; Opus builds at "max" (the
# claude CLI's top tier); Sol runs at "xhigh" (the codex CLI's top
# tier). Override per launch with --fable-effort / --opus-effort /
# --sol-effort.
CLAUDE_EFFORT_CHOICES = ["low", "medium", "high", "xhigh", "max"]
# Sol's model rejects "minimal" (API 400, verified live 2026-07-14);
# its legal set is the one below.
CODEX_EFFORT_CHOICES = ["none", "low", "medium", "high", "xhigh"]
DEFAULT_FABLE_EFFORT = "xhigh"
DEFAULT_OPUS_EFFORT = "max"
DEFAULT_SOL_EFFORT = "xhigh"

# Context budgets per dispatched turn (USER 2026-07-14: no bot runs
# with a context window above X tokens, where X is a command-line key
# and Sol's key is separate). Neither CLI takes a hard cap, so both are
# told to COMPACT (summarize their own history and continue) whenever
# the live context reaches the budget, instead of growing toward their
# native 1M windows: the claude CLI (Fable, Opus) reads
# CLAUDE_CODE_AUTO_COMPACT_WINDOW from the environment; the codex CLI
# (Sol) takes -c model_auto_compact_token_limit (accepted live,
# 2026-07-14). Override per launch with --claude-context / --sol-context.
DEFAULT_CLAUDE_CONTEXT_BUDGET = 500000
DEFAULT_SOL_CONTEXT_BUDGET = 500000

# dispatch() reads this for the claude environment; main() rebinds it
# from --claude-context. Sol's budget rides inside AGENT_COMMANDS.
CLAUDE_CONTEXT_BUDGET = DEFAULT_CLAUDE_CONTEXT_BUDGET

# A dispatched turn that runs past this many minutes is killed and its
# message parked in failed/ for inspection. The guard exists because a
# claude turn once printed "Execution error" and then hung, holding its
# lane for 21 minutes until a human Ctrl-C'd the watch (2026-07-14).
# Long legitimate turns exist (a big review can run 20+ minutes), so
# the default is generous; raise it per launch with --dispatch-timeout.
DISPATCH_TIMEOUT_MINUTES = 60


def build_agent_commands(fable_effort, opus_effort, sol_effort,
                         sol_context_budget):
    """Assemble the per-agent headless CLI commands at the given settings.

    Arguments:
      fable_effort       = claude CLI effort level for Fable dispatches
                           (one of CLAUDE_EFFORT_CHOICES).
      opus_effort        = claude CLI effort level for Opus dispatches
                           (one of CLAUDE_EFFORT_CHOICES).
      sol_effort         = codex CLI reasoning-effort level for Sol
                           dispatches (one of CODEX_EFFORT_CHOICES).
      sol_context_budget = tokens of live context at which a Sol turn
                           compacts (the claude sessions' budget rides
                           the environment instead -- see dispatch()).

    Returns:
      dict mapping "fable"/"opus"/"sol" to the argv list dispatch()
      appends the message to.
    """
    commands = {
        # Absolute path: the user's conda shells resolve an OLDER claude
        # binary with a separate (logged-out) credential store; this one
        # is the logged-in v2.1.208 install (diagnosed 2026-07-14).
        "fable": ["/Users/vivianmiranda/.local/bin/claude", "-p",
                  "--model", "claude-fable-5",
                  "--effort", fable_effort,
                  "--permission-mode", "acceptEdits"],
        "opus": ["/Users/vivianmiranda/.local/bin/claude", "-p",
                 "--model", "claude-opus-4-8",
                 "--effort", opus_effort,
                 "--permission-mode", "acceptEdits"],
        # Verified by the red team's read-only probe (codex-cli 0.144.2;
        # the conventions note records the probe): workspace-write sandbox
        # rooted at the repo, which contains every worktree Sol works in.
        # service_tier=standard keeps codex Fast Mode OFF for dispatched
        # turns (USER 2026-07-14): the standard tier is slower in
        # wall-clock time but far cheaper against the token quota, and an
        # unattended mailbox turn never needs the speed. Pinned here
        # because the user's global ~/.codex/config.toml says "priority"
        # -- a dispatch must not inherit that default.
        "sol": ["/Applications/ChatGPT.app/Contents/Resources/codex",
                "exec",
                "--model", "gpt-5.6-sol",
                "-c", "model_reasoning_effort=" + sol_effort,
                "-c", "service_tier=standard",
                "-c", ("model_auto_compact_token_limit="
                       + str(sol_context_budget)),
                "--sandbox", "workspace-write",
                "--cd", REPO_ROOT],
    }
    return commands


# main() rebuilds this from the command-line flags; the module-level
# value keeps imports and direct function calls working at the defaults.
AGENT_COMMANDS = build_agent_commands(
    fable_effort=DEFAULT_FABLE_EFFORT,
    opus_effort=DEFAULT_OPUS_EFFORT,
    sol_effort=DEFAULT_SOL_EFFORT,
    sol_context_budget=DEFAULT_SOL_CONTEXT_BUDGET)

# The working directory each dispatched agent starts in. Fable and Opus
# develop in this worktree; Sol works from the repository root (its command
# carries the same root in its own --cd), which is what puts it in a
# different lane -- see process_backlog().
AGENT_CWD = {
    "fable": WORKTREE,
    "opus": WORKTREE,
    "sol": REPO_ROOT,
}

# A message still carrying template placeholders has no job in it; refuse
# it instead of burning a live headless turn (learned from dispatch 0001).
PLACEHOLDER_MARKERS = ["<spec>", "<X>", "<section>", "<unit>",
                       "your message here"]

# When the TOTAL open execution backlog reaches this many units, the
# queue counts as SATURATED: the red team becomes the SECOND IMPLEMENTER,
# and the Architect hands it build units so the backlog drains on two
# execution tracks (.claude/FABLE_ROLE.md, "Second-Implementer
# assignments"; the mode switch is always an explicit sentence in the
# handoff, never implied by this number alone). The total is queued
# mailbox messages PLUS the "- OPEN" lines of notes/backlog.md -- the
# program's ledger of every unit still owed execution and audit (user
# rule, 2026-07-14: demand is what saturates the queue, not the
# dispatch rate).
SECOND_IMPLEMENTER_THRESHOLD = 10
BACKLOG_LEDGER = os.path.join(WORKTREE, "notes", "backlog.md")

# One landed milestone = ONE FULL AUDIT TRAIL: the feature, its
# witness/gate leg, and the notes audit record — a few hundred changed
# lines. Unlanded content past this many lines means an audited unit is
# overdue for its own squash landing to main (user rule, 2026-07-14,
# after seven hours of work landed as one 12,000-line main commit).
# Measured as the CONTENT diff against main, never as a commit count:
# a squash landing leaves the old branch commits outside main's
# ancestry forever, so commit counts overstate the debt permanently.
# report_landing_debt() prints the meter with every demand report.
LANDING_DEBT_LINE_LIMIT = 400


def backlog_ledger_count():
    """Count the open units recorded in the backlog ledger.

    Returns:
      The number of lines in notes/backlog.md starting "- OPEN" (zero
      when the ledger does not exist).
    """
    if not os.path.isfile(BACKLOG_LEDGER):
        return 0
    count = 0
    with open(BACKLOG_LEDGER, encoding="utf-8") as f:
        for line in f:
            if line.startswith("- OPEN"):
                count = count + 1
    return count

PREAMBLE = (
    "You are invoked headlessly by tools/mailbox_daemon.py (no human is\n"
    "watching this turn). Resolve your role per CLAUDE.md from the block\n"
    "below. The substance is in the notes/ entries the message cites --\n"
    "read them first. Do the work per your role file. Ordinary rule: end\n"
    "your turn by\n"
    "(1) writing your substance to the appropriate notes/ entry and\n"
    "(2) writing your outbound handoff block to a NEW file\n"
    "<seq>-to-<fable|opus|sol>.md using the next sequence number, INSIDE\n"
    "THIS EXACT DIRECTORY (your cwd may differ -- a relative notes/mailbox\n"
    "path is wrong unless it resolves here):\n"
    "    " + MAILBOX + "\n"
    "Narrow exception: if and only if the inbound's binding instruction\n"
    "explicitly says the thread is TERMINAL and no reply is owed, write no\n"
    "outbound merely to satisfy this wrapper. Ambiguity follows the ordinary\n"
    "rule: record the substance and write the outbound.\n"
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
    for path in glob.glob(os.path.join(MAILBOX, "*.md")):
        name = os.path.basename(path)
        if re.match(r"\d+-to-(fable|opus|sol)\.md$", name):
            found.append(path)
    found.sort(key=message_sequence)
    return found


def message_sequence(path):
    """Return the numeric sequence at the start of a message filename.

    Arguments:
      path = a mailbox message path accepted by pending_messages().

    Returns:
      The integer before ``-to-`` in the filename.
    """
    name = os.path.basename(path)
    return int(name.split("-to-", maxsplit=1)[0])


def placeholder_in(message):
    """Return a marker only when the whole body is an unfilled template.

    A real audit may need to discuss a literal such as ``<unit>``. Treating
    every substring occurrence as an unfilled template rejects that audit.

    Arguments:
      message = the decoded mailbox body.

    Returns:
      The matching marker, or None when the body carries real text.
    """
    body = message.strip()
    for marker in PLACEHOLDER_MARKERS:
        if body == marker:
            return marker
    return None


def move_without_overwrite(path, directory):
    """Move a message into a state directory without replacing history.

    Arguments:
      path      = the current message path.
      directory = the destination directory.

    Returns:
      The destination path, or None when that name is already present or the
      source was claimed first.
    """
    os.makedirs(directory, exist_ok=True)
    destination = os.path.join(directory, os.path.basename(path))
    try:
        os.link(path, destination)
    except FileExistsError:
        print("  !! refusing to overwrite existing message state: "
              + destination)
        return None
    except FileNotFoundError:
        return None
    os.unlink(path)
    return destination


def claim_message(path):
    """Atomically remove a message from the pending queue before dispatch.

    A claimed message remains in ``inflight/`` if the daemon is interrupted.
    That ambiguous state requires a human decision and is never dispatched a
    second time automatically.

    Arguments:
      path = the pending mailbox path.

    Returns:
      The inflight path, or None when another process claimed it first.
    """
    claimed = move_without_overwrite(
        path=path,
        directory=os.path.join(MAILBOX, "inflight"))
    if claimed is None:
        print("  note: " + os.path.basename(path)
              + " was already claimed; skipping duplicate dispatch.")
    return claimed


def acquire_dispatch_lock():
    """Acquire the process-wide dispatch-loop lock without a PID race.

    Returns:
      An open locked file, or None when another loop owns the lock.
    """
    os.makedirs(MAILBOX, exist_ok=True)
    lock_path = os.path.join(MAILBOX, ".dispatch.lock")
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        owner = lock_file.read().strip()
        lock_file.close()
        print("another dispatch loop is already running (pid "
              + (owner or "unknown") + "); refusing to overlap it.")
        return None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def release_dispatch_lock(lock_file):
    """Release a lock returned by acquire_dispatch_lock().

    Arguments:
      lock_file = the open locked file.

    Returns:
      None.
    """
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


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
    dispatch_path = path
    if not dry_run:
        dispatch_path = claim_message(path=path)
        if dispatch_path is None:
            return False
    try:
        with open(dispatch_path, encoding="utf-8") as f:
            message = f.read()
    except (OSError, UnicodeError) as exc:
        if dry_run:
            print("[dry-run] would refuse " + name + ": cannot read UTF-8: "
                  + str(exc))
            return False
        move_without_overwrite(
            path=dispatch_path,
            directory=os.path.join(MAILBOX, "failed"))
        print("refused " + name + ": cannot read the body as UTF-8: "
              + str(exc) + "; parked in failed/.")
        return False

    marker = placeholder_in(message=message)
    if marker is not None:
        if dry_run:
            print("[dry-run] would refuse " + name
                  + ": the whole body is template placeholder '" + marker
                  + "'; no file changed.")
            return False
        move_without_overwrite(
            path=dispatch_path,
            directory=os.path.join(MAILBOX, "failed"))
        print("refused " + name + ": the whole body is the template "
              "placeholder '" + marker + "'; parked in failed/; fill "
              "in the real text and requeue.")
        return False

    if "\x00" in message:
        if dry_run:
            print("[dry-run] would refuse " + name
                  + ": the body contains a NUL byte; no file changed.")
            return False
        move_without_overwrite(
            path=dispatch_path,
            directory=os.path.join(MAILBOX, "failed"))
        print("refused " + name + ": the body contains a NUL byte, which "
              "cannot be a command argument; parked in failed/.")
        return False

    command = AGENT_COMMANDS[agent] + [PREAMBLE + message]

    if dry_run:
        print("[dry-run] would dispatch " + name + " -> "
              + " ".join(AGENT_COMMANDS[agent])
              + "  (cwd " + AGENT_CWD[agent] + ")")
        return True

    print("dispatching " + name + " -> " + agent + " ...")
    # Stream the agent's output straight into the relay log AS IT RUNS
    # (stderr folded in -- the codex CLI narrates its progress there), and
    # heartbeat once a minute so a long turn is distinguishable from a hang:
    # elapsed time always moves, and the log size moves whenever the agent
    # emits anything. A buffered subprocess.run() here once left the
    # terminal silent for an entire multi-minute turn.
    os.makedirs(RELAY_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(RELAY_DIR, stamp + "-dispatch-" + agent + ".log")
    started = time.time()
    proc = None
    launch_error = None
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(AGENT_COMMANDS[agent]) + " <message>\n")
        f.write("--- live output (stdout+stderr interleaved) ---\n")
        f.flush()
        # the claude CLI takes its context budget from the environment
        # (Sol's rides its own -c flag in the command instead): compact
        # whenever the live context reaches the budget, rather than
        # growing to the native 1M-token window.
        env = os.environ.copy()
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(CLAUDE_CONTEXT_BUDGET)
        try:
            proc = subprocess.Popen(command,
                                    stdout=f,
                                    stderr=subprocess.STDOUT,
                                    cwd=AGENT_CWD[agent],
                                    env=env)
        except (OSError, ValueError) as exc:
            launch_error = exc
            f.write("\n--- dispatch could not start: " + str(exc) + " ---\n")
        if proc is not None:
            next_beat = started + 60.0
            deadline = started + DISPATCH_TIMEOUT_MINUTES * 60.0
            while proc.poll() is None:
                time.sleep(5)
                if time.time() >= deadline:
                    # a hung CLI would hold this lane forever (seen live:
                    # a turn printed "Execution error" then produced
                    # nothing for 21 minutes). Kill it; the non-zero exit
                    # code below parks the claimed message in failed/.
                    proc.kill()
                    proc.wait()
                    print("  timed out " + name + " after "
                          + str(DISPATCH_TIMEOUT_MINUTES) + " min; the "
                          "turn was killed and the message parks in "
                          "failed/; requeue it by moving it back to the "
                          "mailbox (or relaunch with a larger "
                          "--dispatch-timeout).")
                    break
                if time.time() >= next_beat:
                    elapsed_min = (time.time() - started) / 60.0
                    log_kb = os.path.getsize(log_path) / 1024.0
                    print("  ... " + name + " still running "
                          + "(%.0f min elapsed, log %.1f kB; tail -f %s)"
                          % (elapsed_min, log_kb, log_path))
                    next_beat += 60.0
            f.write("\n--- rc=" + str(proc.returncode) + " ---\n")

    if launch_error is not None:
        move_without_overwrite(
            path=dispatch_path,
            directory=os.path.join(MAILBOX, "failed"))
        print("  !! dispatch could not start: " + str(launch_error)
              + "; message parked in failed/; log -> " + log_path)
        return False

    print("  rc=" + str(proc.returncode) + "  log -> " + log_path)
    # show the reply's tail on the terminal so activity is visible live.
    with open(log_path, encoding="utf-8") as f:
        reply_lines = f.read().strip().splitlines()
    for line in reply_lines[-8:]:
        print("  | " + line)

    if proc.returncode != 0:
        # a failed dispatch is NOT done: park it in failed/ so it is never
        # silently consumed, and never hot-retried while the cause persists.
        # Requeue after fixing the cause:  mv notes/mailbox/failed/<f> notes/mailbox/
        move_without_overwrite(
            path=dispatch_path,
            directory=os.path.join(MAILBOX, "failed"))
        # the turn's output lives in the log file (it streams there;
        # proc.stdout is None under Popen with a file handle).
        if "Not logged in" in "\n".join(reply_lines):
            print("  !! the headless CLI is logged out; run `claude` in a "
                  "terminal, type /login, then requeue from failed/.")
        else:
            print("  !! dispatch failed; message parked in failed/, see "
                  "the log above.")
        return False

    done_path = move_without_overwrite(path=dispatch_path, directory=DONE)
    if done_path is None:
        # Someone quarantined the inflight file by hand, or a historical
        # archive already owns the name. Never overwrite either state.
        print("  note: " + name + " could not move to done/; leaving the "
              "existing state untouched.")
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
    report_demand(backlog=backlog)
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


def report_demand(backlog):
    """Print the queue-depth line + the second-Implementer tripwire.

    The demand total is the queued mailbox messages PLUS the "- OPEN"
    lines of notes/backlog.md (user rule, 2026-07-14: demand is what
    saturates the queue, not the dispatch rate). Printed by every watch
    pass that holds work AND by every --send, so the person queueing a
    message always sees the load they are adding to.

    Arguments:
      backlog = the current pending message paths (pending_messages()).
    """
    depth = {"fable": 0, "opus": 0, "sol": 0}
    for path in backlog:
        name = os.path.basename(path)
        agent = re.match(r"\d+-to-(fable|opus|sol)\.md$", name).group(1)
        depth[agent] = depth[agent] + 1
    ledger = backlog_ledger_count()
    total = len(backlog) + ledger
    print("queue depth: opus=" + str(depth["opus"])
          + " sol=" + str(depth["sol"])
          + " fable=" + str(depth["fable"])
          + " | open backlog (notes/backlog.md): " + str(ledger)
          + " | total demand: " + str(total))
    if total >= SECOND_IMPLEMENTER_THRESHOLD:
        print("  hint: total open demand is at or past "
              + str(SECOND_IMPLEMENTER_THRESHOLD) + " units; the red "
              "team is now the second implementer: build units flow to "
              "it as well as to Opus "
              "(.claude/FABLE_ROLE.md, Second-Implementer assignments).")
    report_landing_debt()


def report_landing_debt():
    """Print how much branch content has not yet landed on main.

    The milestone that must land is ONE FULL AUDIT TRAIL: the feature,
    its witness or gate leg, and the notes audit record. Debt past
    LANDING_DEBT_LINE_LIMIT changed lines means an audited unit is
    sitting unlanded, which is how the 12,000-line batch landing of
    2026-07-14 happened (user rule: land at every audit-GO boundary,
    one unit per squash commit). Content is measured with git diff
    against main -- a commit count would never drop after a squash
    landing, because squashing leaves the original branch commits
    outside main's ancestry.
    """
    proc = subprocess.run(["git", "diff", "--shortstat", "main", "HEAD"],
                          capture_output=True,
                          text=True,
                          cwd=WORKTREE)
    if proc.returncode != 0:
        # no main ref (fresh clone mid-setup): the debt line is a
        # courtesy meter, not a gate -- stay silent rather than crash.
        return
    stat = proc.stdout.strip()
    if stat == "":
        print("landing debt: none; the branch and main hold the "
              "same content")
        return
    # --shortstat prints e.g. " 3 files changed, 120 insertions(+), 4
    # deletions(-)"; the debt is the total lines touched either way.
    changed_lines = 0
    for count, keyword in re.findall(r"(\d+) (insertion|deletion)", stat):
        changed_lines = changed_lines + int(count)
    print("landing debt: " + stat + " vs main")
    if changed_lines > LANDING_DEBT_LINE_LIMIT:
        print("  hint: more than " + str(LANDING_DEBT_LINE_LIMIT)
              + " unlanded lines means at least one full audit trail "
              "is overdue; squash-land the audited unit(s) to main "
              "now, one unit per commit "
              "(.claude/FABLE_ROLE.md, Landing granularity).")


def send(agent, text, dry_run):
    """Drop a new message into the mailbox (the loop's entry point).

    Arguments:
      agent   = "fable", "opus", or "sol".
      text    = the routing summary (point at notes/; do not inline specs).
      dry_run = True to print the file that WOULD be queued and write
                nothing. Rehearsing --send used to queue a real message
                (main() returned before the dry-run branch ever ran), so a
                junk body became a live dispatched turn as soon as a watch
                picked it up -- the 0022 audit's unrunnable gate leg.

    Returns:
      True when the message was queued, or would be queued in a dry run.
    """
    if dry_run:
        print("[dry-run] would queue "
              + os.path.join(MAILBOX, next_seq() + "-to-" + agent + ".md"))
        return True
    os.makedirs(MAILBOX, exist_ok=True)
    lock_path = os.path.join(MAILBOX, ".sequence.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            for _ in range(20):
                path = os.path.join(
                    MAILBOX,
                    next_seq() + "-to-" + agent + ".md")
                handle, temporary = tempfile.mkstemp(
                    prefix=".message-",
                    dir=MAILBOX)
                try:
                    with os.fdopen(handle, "w", encoding="utf-8") as f:
                        f.write(text)
                        if not text.endswith("\n"):
                            f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
                    try:
                        # The same-directory link publishes a complete inode
                        # atomically and refuses to replace a manually created
                        # destination.
                        os.link(temporary, path)
                    except FileExistsError:
                        continue
                    print("queued " + path)
                    report_demand(backlog=pending_messages())
                    return True
                finally:
                    if os.path.isfile(temporary):
                        os.remove(temporary)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    print("could not claim a sequence number after 20 tries; "
          "is something flooding the mailbox?")
    return False


def main():
    # both are rebound below from the parsed command line; Python wants
    # the global declaration before the first mention of either name.
    global AGENT_COMMANDS
    global DISPATCH_TIMEOUT_MINUTES
    global CLAUDE_CONTEXT_BUDGET

    parser = argparse.ArgumentParser(
        description="file mailbox + headless dispatch for the agent loop")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would happen and change nothing: "
                             "pending dispatches are printed, not run, and "
                             "--send/--ping print the message file they "
                             "would queue without writing it")
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
    parser.add_argument("--fable-effort", default=DEFAULT_FABLE_EFFORT,
                        choices=CLAUDE_EFFORT_CHOICES,
                        help="claude CLI reasoning effort for Fable "
                             "dispatches (default: "
                             + DEFAULT_FABLE_EFFORT + ")")
    parser.add_argument("--opus-effort", default=DEFAULT_OPUS_EFFORT,
                        choices=CLAUDE_EFFORT_CHOICES,
                        help="claude CLI reasoning effort for Opus "
                             "dispatches (default: "
                             + DEFAULT_OPUS_EFFORT + ")")
    parser.add_argument("--sol-effort", default=DEFAULT_SOL_EFFORT,
                        choices=CODEX_EFFORT_CHOICES,
                        help="codex CLI reasoning effort for Sol "
                             "dispatches (default: "
                             + DEFAULT_SOL_EFFORT + ")")
    parser.add_argument("--dispatch-timeout", metavar="MINUTES",
                        type=int, default=DISPATCH_TIMEOUT_MINUTES,
                        help="kill a dispatched turn that runs past "
                             "this many minutes and park its message "
                             "in failed/ (default: "
                             + str(DISPATCH_TIMEOUT_MINUTES) + ")")
    parser.add_argument("--claude-context", metavar="TOKENS",
                        type=int, default=DEFAULT_CLAUDE_CONTEXT_BUDGET,
                        help="Fable and Opus turns compact their "
                             "context whenever it reaches this many "
                             "tokens (default: "
                             + str(DEFAULT_CLAUDE_CONTEXT_BUDGET) + ")")
    parser.add_argument("--sol-context", metavar="TOKENS",
                        type=int, default=DEFAULT_SOL_CONTEXT_BUDGET,
                        help="Sol turns compact their context whenever "
                             "it reaches this many tokens (default: "
                             + str(DEFAULT_SOL_CONTEXT_BUDGET) + ")")
    args = parser.parse_args()

    DISPATCH_TIMEOUT_MINUTES = args.dispatch_timeout
    CLAUDE_CONTEXT_BUDGET = args.claude_context

    # Rebuild the dispatch commands at the requested efforts. The watch
    # start line echoes the levels so a terminal scroll-back always
    # shows what this loop instance was launched with.
    AGENT_COMMANDS = build_agent_commands(
        fable_effort=args.fable_effort,
        opus_effort=args.opus_effort,
        sol_effort=args.sol_effort,
        sol_context_budget=args.sol_context)
    if args.watch:
        print("effort levels: fable=" + args.fable_effort
              + " opus=" + args.opus_effort
              + " sol=" + args.sol_effort)
        print("context budgets: fable/opus=" + str(args.claude_context)
              + " sol=" + str(args.sol_context)
              + " tokens (a turn compacts at its budget)")

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
        queued = send(agent=args.ping, text=ping_text, dry_run=args.dry_run)
        return 0 if queued else 1

    if args.send:
        if not args.unit:
            print("--send needs --unit with the routing-summary text")
            return 1
        queued = send(agent=args.send, text=args.unit, dry_run=args.dry_run)
        return 0 if queued else 1

    if args.dry_run:
        if not process_backlog(dry_run=args.dry_run):
            print("mailbox empty")
        return 0

    if args.once:
        dispatch_lock = acquire_dispatch_lock()
        if dispatch_lock is None:
            return 1
        try:
            if not process_backlog(dry_run=False):
                print("mailbox empty")
        finally:
            release_dispatch_lock(lock_file=dispatch_lock)
        return 0

    if args.watch:
        # --once and --watch share one kernel-released lock. This closes both
        # the check-then-write race between watchers and the older gap where
        # --once could overlap a live watcher in the same working directory.
        dispatch_lock = acquire_dispatch_lock()
        if dispatch_lock is None:
            return 1
        print("watching " + MAILBOX + " (Ctrl-C to stop; safe only "
              "between dispatches; killing a dispatch mid-flight dooms "
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
                    print("daemon source changed on disk; exiting so "
                          "the next start runs it (relaunch --watch).")
                    return 0
                time.sleep(20)
        finally:
            release_dispatch_lock(lock_file=dispatch_lock)

    print("choose one of --dry-run / --once / --watch / --send (see --help)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
