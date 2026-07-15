#!/usr/bin/env python3
"""File mailbox + headless dispatch: the loop runs with NO copy/paste.

The medium is a directory of message files; the wake-up is this daemon
invoking each agent's CLI headlessly when a message addressed to it appears.

    ai/notes/mailbox/NNN-to-fable.md      -> Architect route (legacy address)
    ai/notes/mailbox/NNN-to-opus.md       -> Implementer route (legacy address)
    ai/notes/mailbox/NNN-to-sol.md        -> dispatched to the Sol (Codex) CLI
    ai/notes/mailbox/done/                -> processed messages move here

A message file is a ROUTING SUMMARY (the notes-first rule holds: the
substance lives in the `ai/notes/` entry the message cites). Each dispatched
agent with a relayable result is asked to end its turn by (1) writing its
substance to `ai/notes/` and (2) dropping its outbound handoff as the NEXT
numbered message file, so the loop continues without a human relay. An
inbound whose binding instruction explicitly says TERMINAL and no reply is
owed ends without an outbound; ambiguity follows the ordinary outbound rule.

What stays manual, on purpose:
  - merges/pushes to main are ALWAYS the user's (the daemon never runs git);
  - the daemon only dispatches messages; it never edits code or notes itself;
  - every dispatch's full CLI output is archived under ai/notes/relay/.

Every path the daemon uses -- the mailbox, the relay logs, the working
directory each agent starts in -- is DERIVED from this file's own location
(it lives at <worktree>/ai/tools/), so a clone on another computer runs
unedited and the worktree you launch it from is the one it coordinates.
AGENT_COMMANDS, the CLI binary paths, is the one machine-specific block.
`claude -p` runs one headless turn against the subscription; the session
needs enough tool permission to work unattended (set via the harness
settings or the flags there).

Usage:
    python ai/tools/mailbox_daemon.py --help           # all options + defaults
    python ai/tools/mailbox_daemon.py --dry-run        # show what would run
    python ai/tools/mailbox_daemon.py --once           # process backlog, exit
    python ai/tools/mailbox_daemon.py --watch          # poll every 20 s
    python ai/tools/mailbox_daemon.py --send opus --unit "ai/notes/<spec>.md ..."
                                                    # drop a first message
    python ai/tools/mailbox_daemon.py --send sol --ticket-kind closure \
        --unit "Close the existing ledger item in ai/notes/<spec>.md."
                                                    # classify every Sol send
    python ai/tools/mailbox_daemon.py --watch --fix-only Yes
                                                    # close existing work only
    python ai/tools/mailbox_daemon.py --watch --opus-effort high
                                                    # dial one agent's effort
        --fable-effort / --opus-effort take low|medium|high|xhigh|max
        (claude CLI; defaults xhigh and max); --sol-effort takes
        none|low|medium|high|xhigh (codex CLI; default xhigh)
    python ai/tools/mailbox_daemon.py --watch --architect-model opus \
                                           --implementer-model sonnet
                                                    # choose Claude models by role
    python ai/tools/mailbox_daemon.py --watch --dispatch-timeout 90
                                                    # allow longer turns
    python ai/tools/mailbox_daemon.py --watch --claude-context 400000 \
                                           --sol-context 300000
                                                    # context budgets: a turn
        compacts (summarizes its own history and continues) whenever its
        live context reaches the budget; --claude-context covers the
        Architect and Implementer, --sol-context covers Sol; both default
        to 500000
"""

import argparse
import datetime
import fcntl
import glob
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time

# All work and all mailbox traffic live in the SHARED WORKTREE (the branch
# the agents actually develop on), never the bare main-repo checkout. That
# worktree is DERIVED, never configured: this file lives at
# <worktree>/ai/tools/mailbox_daemon.py, so the directory containing ai/ is the
# worktree root. A clone on a new machine therefore runs unedited, and the
# worktree you launch the watch FROM is the one whose mailbox it watches.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ROOT = os.path.dirname(SCRIPT_DIR)
WORKTREE = os.path.dirname(AI_ROOT)


def repo_root_of(worktree):
    """Return the repository root that owns a given worktree directory.

    A Claude Code worktree sits at <repo>/.claude/worktrees/<name>, so the
    repository is three directories up. When the daemon is instead run from
    an ordinary checkout (no .claude/worktrees/ segment above it), that
    checkout IS the repository and is returned unchanged.

    Arguments:
      worktree = the worktree root, i.e. the directory holding ai/tools/.

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

MAILBOX = os.path.join(AI_ROOT, "notes", "mailbox")
DONE = os.path.join(MAILBOX, "done")
RELAY_DIR = os.path.join(AI_ROOT, "notes", "relay")

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
# (USER 2026-07-14): the Architect route audits at "xhigh"; the Implementer
# route builds at "max" (the claude CLI's top tier); Sol runs at "xhigh"
# (the codex CLI's top tier). The historical --fable-effort and
# --opus-effort names remain stable route controls.
CLAUDE_EFFORT_CHOICES = ["low", "medium", "high", "xhigh", "max"]
# Sol's model rejects "minimal" (API 400, verified live 2026-07-14);
# its legal set is the one below.
CODEX_EFFORT_CHOICES = ["none", "low", "medium", "high", "xhigh"]
DEFAULT_FABLE_EFFORT = "xhigh"
DEFAULT_OPUS_EFFORT = "max"
DEFAULT_SOL_EFFORT = "xhigh"

# Model choice is independent of role. The fable/opus mailbox addresses are
# stable legacy route keys, while these defaults preserve existing launches.
# Any non-whitespace Claude alias or full model ID accepted by
# `claude --model` can override them per invocation.
DEFAULT_ARCHITECT_MODEL = "claude-fable-5"
DEFAULT_IMPLEMENTER_MODEL = "claude-opus-4-8"

# Context budgets per dispatched turn (USER 2026-07-14: no bot runs
# with a context window above X tokens, where X is a command-line key
# and Sol's key is separate). Neither CLI takes a hard cap, so both are
# told to COMPACT (summarize their own history and continue) whenever
# the live context reaches the budget, instead of growing toward their
# native 1M windows: the Claude Architect/Implementer routes read
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
MAX_DISPATCH_TIMEOUT_MINUTES = 1000000
MAX_TIMEOUT_HISTORY_BYTES = 262144
MAX_TIMEOUT_HISTORY_EVENTS = 1000

# A watch periodically manufactures one GLOBAL safe-stop opportunity.  Five
# completed child turns is frequent compared with the multi-minute turns this
# daemon runs, while the time bound prevents a sparse or slow queue from going
# indefinitely without an all-idle window.  These are watch-only: --once and
# --dry-run retain their finite, delay-free behavior.
RENDEZVOUS_DISPATCH_INTERVAL = 5
RENDEZVOUS_MINUTE_INTERVAL = 15
SAFE_KILL_COUNTDOWN_SECONDS = 20
WATCH_POLL_SECONDS = 20


def report_in_flight_status(count):
    """Print the truthful unsafe status for one or more live children."""
    noun = "turn" if count == 1 else "turns"
    print(str(count) + " " + noun
          + " in flight; not safe to stop.", flush=True)


def report_admitted_status():
    """Expire any earlier safe line before an attempt can claim its file."""
    print("dispatch preparation admitted; not safe to stop.", flush=True)


def report_safe_interval_closed():
    """Invalidate a completed safe interval before admissions can reopen."""
    print("safe interval ended; not safe to stop.", flush=True)


class _RendezvousPermit:
    """One watch-global release from before claim through state publication."""

    def __init__(self):
        self.launched = False
        self.reaped = False
        self.released = False


class SafeKillRendezvous:
    """Close watch admissions periodically and prove every lane is idle.

    ``active_attempts`` deliberately covers more than live children.  A turn
    that passed the admission gate but has not reached Popen can already have
    claimed its mailbox file, so an advertised safe window must wait for that
    whole attempt as well as for every launched child.
    """

    def __init__(self, source_path=None, source_stamp=None):
        self._lock = threading.Condition()
        self._active_attempts = 0
        self._in_flight = 0
        self._completed = 0
        self._draining = False
        self._deadline = self._next_deadline()
        self._source_path = source_path
        self._source_stamp = source_stamp
        self._source_changed = False

    @staticmethod
    def _next_deadline():
        return (time.monotonic()
                + float(RENDEZVOUS_MINUTE_INTERVAL) * 60.0)

    def _arm_if_due_locked(self):
        if (self._completed >= RENDEZVOUS_DISPATCH_INTERVAL
                or time.monotonic() >= self._deadline):
            self._draining = True

    def _stop_for_source_change_locked(self):
        if self._source_path is None:
            return
        try:
            changed = (os.path.getmtime(self._source_path)
                       != self._source_stamp)
        except OSError:
            changed = True
        if changed:
            self._source_changed = True
            self._draining = True

    def begin_attempt(self):
        """Return a release permit, or None once the global drain is armed."""
        while True:
            with self._lock:
                self._stop_for_source_change_locked()
                self._arm_if_due_locked()
                if self._draining:
                    return None
                # Reserve cadence capacity across all cwd lanes.  A refusal
                # or Popen failure later frees the reservation; a reaped child
                # converts it into one completed turn.  This prevents a fast
                # lane from starting turn K+1 while turn K is still live.
                if (self._completed + self._active_attempts
                        < RENDEZVOUS_DISPATCH_INTERVAL):
                    permit = _RendezvousPermit()
                    self._active_attempts = self._active_attempts + 1
                else:
                    self._lock.wait()
                    continue
            # This flushed transition happens before begin_attempt returns,
            # so dispatch cannot claim the root message while an expired
            # ordinary-poll or countdown line is still the visible status.
            try:
                report_admitted_status()
            except BaseException:
                # A broken output stream must not strand an unreturned permit
                # and make the global gate appear permanently busy.
                with self._lock:
                    self._active_attempts = self._active_attempts - 1
                    self._lock.notify_all()
                raise
            return permit

    def source_changed(self):
        """Return whether an admission observed a stale daemon source."""
        with self._lock:
            return self._source_changed

    def turn_started(self, permit):
        """Record a successful Popen and print the exact unsafe status."""
        with self._lock:
            if permit.launched:
                raise RuntimeError("rendezvous permit launched twice")
            permit.launched = True
            self._in_flight = self._in_flight + 1
            count = self._in_flight
            report_in_flight_status(count=count)

    def turn_finished(self, permit):
        """Count one reaped child regardless of its exit or archive result."""
        with self._lock:
            if not permit.launched or permit.reaped:
                raise RuntimeError("invalid rendezvous child completion")
            permit.reaped = True
            self._in_flight = self._in_flight - 1
            self._completed = self._completed + 1
            self._arm_if_due_locked()
            count = self._in_flight
            if count:
                report_in_flight_status(count=count)
            self._lock.notify_all()

    def finish_attempt(self, permit):
        """Release post-child state work and freeze on an unreaped child."""
        with self._lock:
            if permit.released:
                raise RuntimeError("rendezvous permit released twice")
            permit.released = True
            self._active_attempts = self._active_attempts - 1
            if permit.launched and not permit.reaped:
                # Never advertise safety, or release later work, after losing
                # truthful custody of a child process.
                self._draining = True
            self._arm_if_due_locked()
            self._lock.notify_all()

    def window_ready(self):
        """Return True only for a due drain with no child or preparation."""
        with self._lock:
            self._arm_if_due_locked()
            return (self._draining and self._active_attempts == 0
                    and self._in_flight == 0)

    def all_idle(self):
        """Return whether no admitted attempt or launched child remains."""
        with self._lock:
            return self._active_attempts == 0 and self._in_flight == 0

    def reset_after_safe_opportunity(self):
        """Start a fresh cadence epoch after a proven all-idle interval."""
        with self._lock:
            if self._active_attempts != 0 or self._in_flight != 0:
                raise RuntimeError("cannot reset a non-idle rendezvous")
            self._completed = 0
            self._draining = False
            self._deadline = self._next_deadline()
            self._lock.notify_all()


# main() owns this only while a locked --watch is live.  Keeping the public
# process_backlog()/drain_lane()/dispatch() call shapes unchanged preserves
# finite callers and the existing focused reproduction suites.
_ACTIVE_WATCH_RENDEZVOUS = None
_RENDEZVOUS_LOCAL = threading.local()


def _rendezvous_turn_started():
    """Bind a successful Popen to this worker's active watch permit."""
    controller = _ACTIVE_WATCH_RENDEZVOUS
    permit = getattr(_RENDEZVOUS_LOCAL, "permit", None)
    if controller is not None and permit is not None:
        controller.turn_started(permit=permit)


def _rendezvous_turn_finished():
    """Bind a reaped child to this worker's active watch permit."""
    controller = _ACTIVE_WATCH_RENDEZVOUS
    permit = getattr(_RENDEZVOUS_LOCAL, "permit", None)
    if controller is not None and permit is not None:
        controller.turn_finished(permit=permit)


def waiting_messages_text(count):
    """Return a grammatically exact root-queue count for safe statuses."""
    if count == 0:
        return "no messages waiting"
    noun = "message" if count == 1 else "messages"
    return str(count) + " " + noun + " waiting"


def run_safe_kill_countdown(controller):
    """Print the main-thread 20-second all-idle window, then reopen work."""
    if not controller.window_ready():
        raise RuntimeError("safe-kill countdown requested before all-idle")
    for seconds_more in range(SAFE_KILL_COUNTDOWN_SECONDS - 1, -1, -1):
        waiting = len(pending_messages())
        print("all lanes idle; safe to Ctrl-C for " + str(seconds_more)
              + "s more; " + waiting_messages_text(count=waiting) + ".",
              flush=True)
        time.sleep(1)
    report_safe_interval_closed()
    controller.reset_after_safe_opportunity()


def report_ordinary_safe_poll(controller):
    """Mark the existing idle poll delay as an ordinary safe opportunity."""
    if not controller.all_idle():
        return False
    waiting = len(pending_messages())
    print("all lanes idle; safe to Ctrl-C for this "
          + str(WATCH_POLL_SECONDS) + "s poll; "
          + waiting_messages_text(count=waiting) + ".", flush=True)
    controller.reset_after_safe_opportunity()
    return True


def positive_int(value):
    """Parse an argparse integer that must be strictly positive."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer") from exc
    if parsed <= 0 or parsed > MAX_DISPATCH_TIMEOUT_MINUTES:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer no larger than "
            + str(MAX_DISPATCH_TIMEOUT_MINUTES))
    return parsed


def truthy_fix_only(value):
    """Parse the deliberately forgiving truthy value for ``--fix-only``.

    The user explicitly allowed capitalization mistakes and surrounding
    whitespace.  Other supplied values are errors rather than silently
    disabling a safety mode because of a typo.
    """
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    raise argparse.ArgumentTypeError(
        "value must be 1, true, or yes (capitalization is ignored)")


def validate_model_name(value):
    """Accept one Claude model alias or full ID without shell ambiguity."""
    if (not isinstance(value, str) or not value or "\x00" in value
            or any(character.isspace() for character in value)):
        raise argparse.ArgumentTypeError(
            "Claude model must be one non-whitespace alias or full name")
    return value


def build_agent_commands(fable_effort, opus_effort, sol_effort,
                         sol_context_budget,
                         architect_model=DEFAULT_ARCHITECT_MODEL,
                         implementer_model=DEFAULT_IMPLEMENTER_MODEL):
    """Assemble the per-agent headless CLI commands at the given settings.

    Arguments:
      fable_effort       = claude CLI effort level for the Architect route
                           (legacy fable address; CLAUDE_EFFORT_CHOICES).
      opus_effort        = claude CLI effort level for the Implementer route
                           (legacy opus address; CLAUDE_EFFORT_CHOICES).
      sol_effort         = codex CLI reasoning-effort level for Sol
                           dispatches (one of CODEX_EFFORT_CHOICES).
      sol_context_budget = tokens of live context at which a Sol turn
                           compacts (the claude sessions' budget rides
                           the environment instead -- see dispatch()).
      architect_model    = Claude alias or full ID launched on the legacy
                           fable route.
      implementer_model  = Claude alias or full ID launched on the legacy
                           opus route.

    Returns:
      dict mapping "fable"/"opus"/"sol" to the argv list dispatch()
      appends the message to.
    """
    architect_model = validate_model_name(value=architect_model)
    implementer_model = validate_model_name(value=implementer_model)
    commands = {
        # Absolute path: the user's conda shells resolve an OLDER claude
        # binary with a separate (logged-out) credential store; this one
        # is the logged-in v2.1.208 install (diagnosed 2026-07-14).
        "fable": ["/Users/vivianmiranda/.local/bin/claude", "-p",
                  "--model", architect_model,
                  "--effort", fable_effort,
                  "--permission-mode", "acceptEdits"],
        "opus": ["/Users/vivianmiranda/.local/bin/claude", "-p",
                 "--model", implementer_model,
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

# The working directory each dispatched agent starts in. The Architect and
# Implementer routes (legacy fable/opus keys) develop in this worktree; Sol
# works from the repository root (its command carries the same root in its own
# --cd), which is what puts it in a different lane -- see process_backlog().
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
# mailbox messages PLUS the "- OPEN" lines of ai/notes/backlog.md -- the
# program's ledger of every unit still owed execution and audit (user
# rule, 2026-07-14: demand is what saturates the queue, not the
# dispatch rate).
SECOND_IMPLEMENTER_THRESHOLD = 10
BACKLOG_LEDGER = os.path.join(AI_ROOT, "notes", "backlog.md")
SOL_TICKET_KINDS = ("closure", "discovery")
SOL_DISPATCH_TICKET_KINDS = SOL_TICKET_KINDS + ("transport",)
SOL_TICKET_HEADER = "MAILBOX-TICKET: "
FIX_ONLY_ENVIRONMENT = "MAILBOX_FIX_ONLY"
FIX_ONLY_LOCK_NAME = ".fix-only.lock"

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

# One sequence grammar owns both allocation and dispatch-time currency. The
# optional letter is historical (messages such as 0107a); the recipient is
# deliberately unrestricted here because archived -to-user messages and
# hand-made hold directories still claim their sequence numbers.
MESSAGE_SEQUENCE_RE = re.compile(r"(\d+)[a-z]?-to-")
PENDING_MESSAGE_RE = re.compile(r"\d+-to-(fable|opus|sol)\.md$")
WATCH_LOCK_OWNER_RE = re.compile(r"watch pid [1-9]\d*$")
STATE_GUARD_SUFFIX = ".state-guard"


def backlog_ledger_count():
    """Count the open units recorded in the backlog ledger.

    Returns:
      The number of lines in ai/notes/backlog.md starting "- OPEN" (zero
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
    "You are invoked headlessly by ai/tools/mailbox_daemon.py (no human is\n"
    "watching this turn). Resolve your role per CLAUDE.md from the block\n"
    "below. The substance is in the ai/notes/ entries the message cites --\n"
    "read them first. Do the work per your role file. Ordinary rule: end\n"
    "your turn by\n"
    "(1) writing your substance to the appropriate ai/notes/ entry and\n"
    "(2) writing your outbound handoff block to a NEW file\n"
    "<seq>-to-<fable|opus|sol>.md using the next sequence number, INSIDE\n"
    "THIS EXACT DIRECTORY (your cwd may differ -- a relative ai/notes/mailbox\n"
    "path is wrong unless it resolves here):\n"
    "    " + MAILBOX + "\n"
    "Every work outbound addressed to Sol must start with exactly one of\n"
    "these classification lines:\n"
    "    MAILBOX-TICKET: closure\n"
    "    MAILBOX-TICKET: discovery\n"
    "Use closure only for work that retires an existing - OPEN ledger line;\n"
    "use discovery when the product is new findings. The daemon refuses to\n"
    "guess a class from prose. The daemon's exact no-work transport ping is\n"
    "the sole reserved MAILBOX-TICKET: transport exception.\n"
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
        value = sequence_in_name(name=os.path.basename(path))
        if value is not None:
            if value > highest:
                highest = value
    return "%04d" % (highest + 1)


def pending_messages():
    """Return the sorted list of unprocessed message paths."""
    found = []
    for path in glob.glob(os.path.join(MAILBOX, "*.md")):
        name = os.path.basename(path)
        if PENDING_MESSAGE_RE.match(name):
            found.append(path)
    found.sort(key=message_sequence)
    return found


def total_open_demand(backlog=None):
    """Return queued messages plus literal open lines in the ledger."""
    if backlog is None:
        backlog = pending_messages()
    return len(backlog) + backlog_ledger_count()


def sol_ticket_kind(message):
    """Return a Sol message's exact first-line class, or ``None``.

    Free-form prose is deliberately never classified.  LF and CRLF are both
    accepted as physical line endings, but whitespace, aliases, and a header
    appearing later in the body do not count.
    """
    match = re.match(
        r"\A" + re.escape(SOL_TICKET_HEADER)
        + r"(" + "|".join(map(re.escape, SOL_DISPATCH_TICKET_KINDS))
        + r")(?:\r?\n|\Z)",
        message)
    if match is None:
        return None
    return match.group(1)


def sol_ticket_body(message):
    """Return the body after a valid Sol classification line."""
    match = re.match(
        r"\A" + re.escape(SOL_TICKET_HEADER)
        + r"(?:" + "|".join(map(re.escape, SOL_DISPATCH_TICKET_KINDS))
        + r")(?:\r?\n|\Z)",
        message)
    if match is None:
        return message
    return message[match.end():]


def transport_ping_text(agent):
    """Return the one no-work transport payload reserved for ``--ping``."""
    return (
        "RELAY CONFIRMATION PING for " + agent + ". This is a "
        "transport test only; no unit is assigned and no repository "
        "file may change. Reply by creating ONE new file,\n"
        "ai/notes/mailbox/<next-sequence>-to-user.md, whose entire body "
        "is one line:\n\n"
        "    PONG " + agent + " from <your model name>\n\n"
        "Then stop. (Files addressed -to-user are read by the human; "
        "the daemon never dispatches them.)\n")


def sol_ticket_payload(ticket_kind, text):
    """Build the byte-stable persisted envelope for a Sol message."""
    payload = SOL_TICKET_HEADER + ticket_kind + "\n\n" + text
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def valid_sol_transport(message):
    """Return whether ``message`` is exactly the daemon's Sol ping."""
    return message == sol_ticket_payload(
        ticket_kind="transport", text=transport_ping_text(agent="sol"))


def fix_only_environment_active():
    """Return whether this send inherited a fix-only watch contract."""
    value = os.environ.get(FIX_ONLY_ENVIRONMENT)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def sol_ticket_refusal(ticket_kind, total, fix_only,
                       transport_valid=False):
    """Return the binding refusal reason for a Sol ticket, or ``None``."""
    if ticket_kind == "transport":
        if transport_valid:
            return None
        return ("MAILBOX-TICKET: transport is reserved for the daemon's "
                "exact --ping sol payload")
    if ticket_kind not in SOL_TICKET_KINDS:
        return ("missing or invalid first line; every Sol ticket must start "
                "with exactly 'MAILBOX-TICKET: closure' or "
                "'MAILBOX-TICKET: discovery'")
    if fix_only and ticket_kind != "closure":
        return ("fix-only watch is closing-only; discovery tickets and new "
                "backlog lines are forbidden until the watch is restarted "
                "without --fix-only")
    if (ticket_kind == "discovery"
            and total >= SECOND_IMPLEMENTER_THRESHOLD):
        return ("total open demand is " + str(total) + ", at or past "
                + str(SECOND_IMPLEMENTER_THRESHOLD)
                + "; append this discovery ticket to the END of "
                "ai/notes/backlog.md instead and wait until total demand "
                "falls below the threshold")
    return None


def inflight_lane_blockers():
    """Return unresolved inflight agent messages grouped by cwd lane.

    Only exact dispatchable message names participate. A hand-made file or an
    archived ``-to-user`` note under inflight cannot block an agent lane, but
    an unresolved Fable message blocks Opus too because those recipients share
    one working directory.
    """
    blockers = {}
    seen = {}
    patterns = [
        os.path.join(MAILBOX, "inflight", "*.md"),
        os.path.join(MAILBOX, "inflight",
                     "*.md" + STATE_GUARD_SUFFIX),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    for path in paths:
        name = blocker_message_name(path=path)
        match = PENDING_MESSAGE_RE.match(name)
        if match is None:
            continue
        cwd = AGENT_CWD[match.group(1)]
        if cwd not in blockers:
            blockers[cwd] = []
            seen[cwd] = set()
        if name in seen[cwd]:
            continue
        seen[cwd].add(name)
        blockers[cwd].append(path)
    for paths in blockers.values():
        paths.sort(key=message_sequence)
    return blockers


def blocker_message_name(path):
    """Return the exact agent basename encoded by an inflight blocker."""
    name = os.path.basename(path)
    if name.endswith(STATE_GUARD_SUFFIX):
        return name[:-len(STATE_GUARD_SUFFIX)]
    return name


def report_inflight_lane_block(blocker_paths, pending_count):
    """Print one clear cross-pass lane-block diagnostic."""
    blocker_names = [blocker_message_name(path=path)
                     for path in blocker_paths]
    if pending_count:
        waiting = (str(pending_count)
                   + " pending message(s) sharing that working directory "
                   "will wait.")
    else:
        waiting = ("no pending root messages share that working directory "
                   "yet.")
    print("  lane blocked by unresolved inflight message(s) "
          + ", ".join(blocker_names) + "; " + waiting)


def message_sequence(path):
    """Return the numeric sequence at the start of a message filename.

    Arguments:
      path = a mailbox message path accepted by pending_messages().

    Returns:
      The integer before ``-to-`` in the filename.
    """
    value = sequence_in_name(name=os.path.basename(path))
    if value is None:
        raise ValueError("not a numbered mailbox message: " + path)
    return value


def sequence_in_name(name):
    """Return a mailbox filename's numeric sequence, if it has one.

    This is the single parser used by both ``next_seq()`` and the dispatch
    currency snapshot, so a message cannot count for allocation while being
    invisible to the dispatch-time maximum.

    Arguments:
      name = a basename from anywhere in the mailbox store.

    Returns:
      The leading integer, or None when the name is not a numbered message.
    """
    match = MESSAGE_SEQUENCE_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def dispatch_currency(dispatch_path, agent):
    """Take one post-claim snapshot and derive its mechanical currency.

    The maximum spans every ``*.md`` below the mailbox, including done,
    failed, hold, and -to-user messages. The newer-message count is narrower:
    only root-pending agent messages whose recipient shares this dispatch's
    working-directory lane count. This is evidence for the receiving human or
    agent, never a semantic decision that the message is obsolete.

    Arguments:
      dispatch_path = the already-claimed inflight message.
      agent         = its recipient.

    Returns:
      ``(store_max_sequence, newer_root_pending_in_lane)``.
    """
    snapshot = glob.glob(os.path.join(MAILBOX, "**", "*.md"),
                         recursive=True)
    dispatched_sequence = message_sequence(path=dispatch_path)
    store_max = 0
    newer_in_lane = 0
    mailbox_root = os.path.abspath(MAILBOX)
    for path in snapshot:
        value = sequence_in_name(name=os.path.basename(path))
        if value is None:
            continue
        if value > store_max:
            store_max = value
        if os.path.dirname(os.path.abspath(path)) != mailbox_root:
            continue
        pending_match = PENDING_MESSAGE_RE.match(os.path.basename(path))
        if pending_match is None or value <= dispatched_sequence:
            continue
        queued_agent = pending_match.group(1)
        if AGENT_CWD[queued_agent] == AGENT_CWD[agent]:
            newer_in_lane = newer_in_lane + 1
    return store_max, newer_in_lane


def timeout_history_path(name):
    """Return the daemon-owned timeout history sidecar for one message."""
    return os.path.join(MAILBOX, ".dispatch-history", name + ".json")


def timeout_events(name):
    """Read the timeout-only event list for one message basename.

    A missing sidecar means the message has never timed out. A malformed
    daemon-owned sidecar is not treated as an empty history: dispatch must not
    erase the only evidence that an earlier turn was killed.
    """
    path = timeout_history_path(name=name)
    try:
        with open(path, encoding="utf-8") as f:
            if os.fstat(f.fileno()).st_size > MAX_TIMEOUT_HISTORY_BYTES:
                raise ValueError("timeout history is too large in " + path)
            try:
                payload = json.load(f)
            except (RecursionError, OverflowError) as exc:
                raise ValueError(
                    "timeout history is too deeply nested in " + path) \
                    from exc
    except FileNotFoundError:
        return []
    if not isinstance(payload, dict):
        raise ValueError("timeout history is not a mapping in " + path)
    if payload.get("schema") != 1 or payload.get("message") != name:
        raise ValueError("invalid timeout-history identity in " + path)
    events = payload.get("timeouts")
    if not isinstance(events, list):
        raise ValueError("invalid timeout-history event list in " + path)
    if len(events) > MAX_TIMEOUT_HISTORY_EVENTS:
        raise ValueError("too many timeout-history events in " + path)
    normalized = []
    for event in events:
        duration = event.get("killed_after_minutes") \
            if isinstance(event, dict) else None
        if not valid_duration(value=duration, strictly_positive=True):
            raise ValueError("invalid timeout duration in " + path)
        observed = event.get("observed_elapsed_minutes")
        if (observed is not None
                and not valid_duration(value=observed,
                                       strictly_positive=False)):
            raise ValueError("invalid observed timeout duration in " + path)
        clean_event = {"killed_after_minutes": duration}
        if observed is not None:
            clean_event["observed_elapsed_minutes"] = observed
        normalized.append(clean_event)
    return normalized


def valid_duration(value, strictly_positive):
    """Return whether a JSON duration is numeric, finite, and in range.

    Integers are finite by definition; avoiding ``math.isfinite`` for them
    also keeps an attacker-controlled enormous JSON integer from raising an
    OverflowError during validation.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if value > MAX_DISPATCH_TIMEOUT_MINUTES:
        return False
    if strictly_positive:
        return value > 0
    return value >= 0


def write_timeout_history(name, killed_after_minutes,
                          observed_elapsed_minutes=None):
    """Append one timeout event through an fsynced atomic replacement.

    This function is called only after the timeout guard kills a child.
    Ordinary nonzero exits never create or append a sidecar.
    """
    if not valid_duration(value=killed_after_minutes,
                          strictly_positive=True):
        raise ValueError("killed-after timeout must be positive")
    if (observed_elapsed_minutes is not None
            and not valid_duration(value=observed_elapsed_minutes,
                                   strictly_positive=False)):
        raise ValueError("observed timeout duration must be nonnegative")
    events = timeout_events(name=name)
    if len(events) >= MAX_TIMEOUT_HISTORY_EVENTS:
        raise ValueError("timeout history reached its event limit")
    event = {"killed_after_minutes": killed_after_minutes}
    if observed_elapsed_minutes is not None:
        event["observed_elapsed_minutes"] = observed_elapsed_minutes
    events.append(event)
    payload = {"schema": 1, "message": name, "timeouts": events}
    directory = os.path.dirname(timeout_history_path(name=name))
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".timeout-", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, timeout_history_path(name=name))
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def exact_duration(value):
    """Format a stored float without changing its represented value."""
    return format(value, ".17g")


def dispatch_banner(store_max, newer_in_lane, previous_timeout_minutes,
                    fix_only=False):
    """Build the mechanical pre-preamble hint for a live dispatch."""
    lines = [
        "--- DISPATCH CURRENCY (mechanical hint only) ---",
        "store-wide mailbox max sequence at claim: %04d" % store_max,
        ("newer messages queued in this working-directory lane: "
         + str(newer_in_lane)),
        ("This marker is not a semantic supersession oracle; read the "
         "mailbox and cited notes first."),
    ]
    if previous_timeout_minutes is not None:
        lines.append(
            "this dispatch previously ran for "
            + exact_duration(value=previous_timeout_minutes)
            + " minutes and was killed")
    if fix_only:
        lines.append(
            "fix-only watch: active; close existing ledger lines only; "
            "create no discovery tickets or new backlog lines.")
    lines.append("--- END DISPATCH CURRENCY ---")
    return "\n".join(lines) + "\n\n"


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


def mailbox_path_is_unredirected(mailbox):
    """Return whether ``mailbox`` stays inside its lexical repository path.

    ``O_NOFOLLOW`` protects the final lock file, but would still follow a
    symlink used as an earlier ``notes`` or ``mailbox`` component.  Compare
    real paths relative to the repository's own real path so symlinks *above*
    the checkout remain harmless while redirects *inside* it are rejected.
    """
    repository = os.path.abspath(REPO_ROOT)
    candidate = os.path.abspath(mailbox)
    try:
        if os.path.commonpath([repository, candidate]) != repository:
            return False
        relative = os.path.relpath(candidate, repository)
    except (OSError, ValueError):
        return False
    expected = os.path.normpath(os.path.join(
        os.path.realpath(repository), relative))
    return os.path.realpath(candidate) == expected


def held_lock_probe(mailbox, lock_name):
    """Probe a regular exact-path lock and its bounded owner metadata.

    The probe is deliberately read-only.  Opening a missing lock must never
    create it because both ``--send --dry-run`` and a refused discovery promise
    zero filesystem mutation.  A shared nonblocking probe coexists with other
    diagnostics but is refused by the exclusive lock held by the real owner.

    Returns:
      ``(held, owner)``. ``held`` is true only when the exact regular inode is
      actively locked. ``owner`` is its bounded ASCII text, or ``None`` when
      held metadata is malformed. Symlinks, redirected parents, stale files,
      replacements, and devices never count as held.
    """
    lock_path = os.path.join(mailbox, lock_name)
    descriptor = None
    probe_acquired = False
    try:
        if not mailbox_path_is_unredirected(mailbox=mailbox):
            return False, None
        before = os.lstat(lock_path)
        if not stat.S_ISREG(before.st_mode):
            return False, None
        flags = os.O_RDONLY | os.O_NONBLOCK
        flags = flags | getattr(os, "O_CLOEXEC", 0)
        flags = flags | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags)
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)):
            return False, None
        try:
            # A watch/once loop owns an exclusive flock.  SH is intentional:
            # simultaneous send diagnostics can all acquire it, so they can
            # never mistake one another for a live watcher.
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            probe_acquired = True
            return False, None
        except BlockingIOError:
            pass
        # The path may have been replaced after open().  A lock on an
        # unlinked/orphaned inode does not protect the filename a future watch
        # would use, so it cannot suppress the warning.
        current = os.lstat(lock_path)
        if (not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino)
                != (opened.st_dev, opened.st_ino)):
            return False, None
        # Bound the read so a corrupt/sparse lock cannot consume unbounded
        # memory.  os.pread leaves the descriptor offset untouched.
        owner_bytes = os.pread(descriptor, 129, 0)
        if len(owner_bytes) > 128:
            return True, None
        try:
            owner = owner_bytes.decode("ascii")
        except UnicodeError:
            return True, None
        return True, owner
    except OSError:
        return False, None
    finally:
        if descriptor is not None:
            if probe_acquired:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                os.close(descriptor)
            except OSError:
                pass


def held_lock_owner(mailbox, lock_name):
    """Return valid owner text for an actively held exact-path lock."""
    held, owner = held_lock_probe(mailbox=mailbox, lock_name=lock_name)
    if not held:
        return None
    return owner


def dispatch_lock_is_live_watch(mailbox):
    """Return whether ``mailbox`` has an exact held ``watch pid N`` lock."""
    owner = held_lock_owner(mailbox=mailbox, lock_name=".dispatch.lock")
    if owner is None:
        return False
    return WATCH_LOCK_OWNER_RE.fullmatch(owner) is not None


def fix_only_watch_is_active(mailbox=None):
    """Return whether this mailbox's reserved mode lock is actively held.

    Owner text is diagnostic, not authority: once the exact-path regular lock
    is held, malformed or concurrently damaged metadata must fail closed as
    fix-only.  Unlocked stale files still read inactive.
    """
    if mailbox is None:
        mailbox = MAILBOX
    held, _ = held_lock_probe(
        mailbox=mailbox, lock_name=FIX_ONLY_LOCK_NAME)
    return held


def mailbox_candidates():
    """Return every mailbox whose watcher could serve this repository.

    The current mailbox and the main checkout are always included.  Worktree
    discovery uses scandir instead of ``glob('*')`` so a legal hidden
    worktree name is not silently missed.  Paths are absolute, de-duplicated,
    and sorted to keep warning output deterministic.
    """
    candidates = {
        os.path.abspath(MAILBOX),
        os.path.abspath(os.path.join(REPO_ROOT, "ai", "notes", "mailbox")),
    }
    worktrees = os.path.join(REPO_ROOT, ".claude", "worktrees")
    try:
        if not mailbox_path_is_unredirected(mailbox=worktrees):
            return sorted(candidates)
        worktrees_state = os.lstat(worktrees)
        if not stat.S_ISDIR(worktrees_state.st_mode):
            return sorted(candidates)
        with os.scandir(worktrees) as entries:
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                candidates.add(os.path.abspath(os.path.join(
                    entry.path, "ai", "notes", "mailbox")))
    except OSError:
        pass
    return sorted(candidates)


def warn_if_mailbox_unwatched():
    """Warn when a send targets a mailbox with no live watch loop.

    The warning is advisory: callers continue to publish (or rehearse) the
    message.  Other watched mailboxes are reported as recovery clues, not as
    alternative destinations; the daemon never silently reroutes a send.
    """
    own_mailbox = os.path.abspath(MAILBOX)
    if dispatch_lock_is_live_watch(mailbox=own_mailbox):
        return
    print("  !! warning: no active watch is polling this mailbox: "
          + own_mailbox)
    for candidate in mailbox_candidates():
        if candidate == own_mailbox:
            continue
        if dispatch_lock_is_live_watch(mailbox=candidate):
            print("  !! warning: another mailbox under this repository has "
                  "a live watch: " + candidate)


def acquire_dispatch_lock(mode="unknown"):
    """Acquire the process-wide dispatch-loop lock without a PID race.

    Arguments:
      mode = ``watch`` or ``once`` for command-line loops.  The default keeps
             older direct callers compatible but is deliberately not treated
             as proof of an active watcher by send diagnostics.

    Returns:
      An open locked file, or None when another loop owns the lock.
    """
    if mode not in ("watch", "once"):
        mode = "unknown"
    os.makedirs(MAILBOX, exist_ok=True)
    lock_path = os.path.join(MAILBOX, ".dispatch.lock")
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        owner = lock_file.read().strip()
        lock_file.close()
        print("another dispatch loop is already running ("
              + (owner or "owner unknown") + "); refusing to overlap it.")
        return None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(mode + " pid " + str(os.getpid()))
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


def acquire_fix_only_lock_while_sequence_locked():
    """Create the mode marker after the caller serializes publishers."""
    if not mailbox_path_is_unredirected(mailbox=MAILBOX):
        print("cannot activate fix-only mode on a redirected mailbox path")
        return None
    lock_path = os.path.join(MAILBOX, FIX_ONLY_LOCK_NAME)
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
    flags = flags | getattr(os, "O_CLOEXEC", 0)
    flags = flags | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        print("cannot activate fix-only mode: " + str(exc))
        return None
    lock_file = os.fdopen(descriptor, "r+", encoding="utf-8")

    def path_still_names_opened_inode(opened):
        """Return whether the public mode path still names this descriptor."""
        try:
            current = os.lstat(lock_path)
        except OSError:
            return False
        return (stat.S_ISREG(current.st_mode)
                and (opened.st_dev, opened.st_ino)
                == (current.st_dev, current.st_ino))

    try:
        opened = os.fstat(lock_file.fileno())
        if (not stat.S_ISREG(opened.st_mode)
                or not path_still_names_opened_inode(opened=opened)):
            print("cannot activate fix-only mode: mode lock is not an "
                  "unchanged regular file")
            lock_file.close()
            return None
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        print("cannot activate fix-only mode: its mode lock is already held "
              "or unreadable (" + str(exc) + ")")
        lock_file.close()
        return None
    if not path_still_names_opened_inode(opened=opened):
        print("cannot activate fix-only mode: mode lock path changed while "
              "its lock was acquired")
        release_fix_only_lock(lock_file=lock_file)
        return None
    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write("fix-only watch pid " + str(os.getpid()))
        lock_file.flush()
        os.fsync(lock_file.fileno())
    except OSError as exc:
        print("cannot activate fix-only mode: could not publish its owner ("
              + str(exc) + ")")
        release_fix_only_lock(lock_file=lock_file)
        return None
    if not path_still_names_opened_inode(opened=opened):
        print("cannot activate fix-only mode: mode lock path changed while "
              "its owner was published")
        release_fix_only_lock(lock_file=lock_file)
        return None
    return lock_file


def acquire_fix_only_lock():
    """Atomically activate fix-only mode relative to message publication.

    Sol senders perform their final policy check while holding the same
    sequence lock.  Therefore a concurrent sender either publishes wholly
    before activation or observes the held mode marker and refuses; it cannot
    publish after the watch has become fix-only.
    """
    os.makedirs(MAILBOX, exist_ok=True)
    sequence_path = os.path.join(MAILBOX, ".sequence.lock")
    try:
        with open(sequence_path, "a+", encoding="utf-8") as sequence_file:
            fcntl.flock(sequence_file.fileno(), fcntl.LOCK_EX)
            try:
                return acquire_fix_only_lock_while_sequence_locked()
            finally:
                fcntl.flock(sequence_file.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        print("cannot activate fix-only mode: sequence lock failed ("
              + str(exc) + ")")
        return None


def release_fix_only_lock(lock_file):
    """Release a lock returned by ``acquire_fix_only_lock``."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def dispatch(path, dry_run, fix_only=False):
    """Send one message file to its addressee's headless CLI.

    Arguments:
      path    = the mailbox message file.
      dry_run  = True to print the would-be command without running it.
      fix_only = True when the owning watch may launch declared closures only.

    Returns:
      True when the dispatch ran (or would run) cleanly.
    """
    name = os.path.basename(path)
    agent_match = PENDING_MESSAGE_RE.match(name)
    if agent_match is None:
        raise ValueError("not a pending agent message: " + path)
    agent = agent_match.group(1)
    # Take one policy snapshot before claim_message() removes this candidate.
    # Dispatch evaluates all OTHER current demand: the already-published
    # candidate must not turn an authorized 9 -> send into a self-refusal at
    # 10.  New concurrent work still counts, so a ticket can be deferred when
    # other demand independently reaches the threshold before launch.
    demand_before_claim = None
    if agent == "sol":
        pending_before_claim = pending_messages()
        demand_before_claim = total_open_demand(
            backlog=pending_before_claim)
        candidate = os.path.abspath(path)
        if any(os.path.abspath(item) == candidate
               for item in pending_before_claim):
            demand_before_claim = max(0, demand_before_claim - 1)
    dispatch_path = path
    currency = None
    prior_timeout = None
    if not dry_run:
        dispatch_path = claim_message(path=path)
        if dispatch_path is None:
            return False
        if not valid_duration(value=DISPATCH_TIMEOUT_MINUTES,
                              strictly_positive=True):
            print("refused " + name + ": dispatch timeout must be between "
                  "1 and " + str(MAX_DISPATCH_TIMEOUT_MINUTES)
                  + " minutes; leaving the claimed message in inflight/.")
            return False
        # One recursive view, taken only after the atomic claim, owns both
        # currency numbers. Re-globbing each number would let a concurrent
        # sender make the banner internally inconsistent.
        currency = dispatch_currency(dispatch_path=dispatch_path, agent=agent)
        try:
            history = timeout_events(name=name)
        except (OSError, ValueError, json.JSONDecodeError,
                OverflowError, RecursionError) as exc:
            print("refused " + name + ": cannot verify its timeout history: "
                  + str(exc) + "; leaving the claimed message in inflight/.")
            return False
        if history:
            prior_timeout = history[-1]["killed_after_minutes"]
    try:
        # Preserve the mailbox body's exact newline bytes. The prompt contract
        # makes the decoded body its exact suffix; default text-mode universal
        # newline translation would silently rewrite a valid CRLF message.
        with open(dispatch_path, encoding="utf-8", newline="") as f:
            message = f.read()
    except (OSError, UnicodeError) as exc:
        if dry_run:
            print("[dry-run] would refuse " + name + ": cannot read UTF-8: "
                  + str(exc))
            return False
        if park_failed_message(dispatch_path=dispatch_path):
            print("refused " + name + ": cannot read the body as UTF-8: "
                  + str(exc) + "; parked in failed/.")
        else:
            print("refused " + name + ": cannot read the body as UTF-8: "
                  + str(exc) + "; failed-state move was not verified; "
                  "inspect inflight/ and failed/.")
        return False

    ticket_kind = None
    if agent == "sol":
        ticket_kind = sol_ticket_kind(message=message)
        reason = sol_ticket_refusal(
            ticket_kind=ticket_kind,
            total=demand_before_claim,
            fix_only=fix_only,
            transport_valid=valid_sol_transport(message=message))
        if reason is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": " + reason
                      + "; no file changed.")
                return False
            if park_failed_message(dispatch_path=dispatch_path):
                print("refused " + name + ": " + reason
                      + "; parked in failed/.")
            else:
                print("refused " + name + ": " + reason
                      + "; failed-state move was not verified; inspect "
                      "inflight/ and failed/.")
            return False

    placeholder_body = (sol_ticket_body(message=message)
                        if agent == "sol" else message)
    marker = placeholder_in(message=placeholder_body)
    if marker is not None:
        if dry_run:
            print("[dry-run] would refuse " + name
                  + ": the whole body is template placeholder '" + marker
                  + "'; no file changed.")
            return False
        if park_failed_message(dispatch_path=dispatch_path):
            print("refused " + name + ": the whole body is the template "
                  "placeholder '" + marker + "'; parked in failed/; fill "
                  "in the real text and requeue.")
        else:
            print("refused " + name + ": the whole body is the template "
                  "placeholder '" + marker + "'; failed-state move was "
                  "not verified; inspect inflight/ and failed/.")
        return False

    if "\x00" in message:
        if dry_run:
            print("[dry-run] would refuse " + name
                  + ": the body contains a NUL byte; no file changed.")
            return False
        if park_failed_message(dispatch_path=dispatch_path):
            print("refused " + name + ": the body contains a NUL byte, "
                  "which cannot be a command argument; parked in failed/.")
        else:
            print("refused " + name + ": the body contains a NUL byte, "
                  "which cannot be a command argument; failed-state move "
                  "was not verified; inspect inflight/ and failed/.")
        return False

    if dry_run:
        print("[dry-run] would dispatch " + name + " -> "
              + " ".join(AGENT_COMMANDS[agent])
              + "  (cwd " + AGENT_CWD[agent] + ")")
        return True

    banner = dispatch_banner(
        store_max=currency[0],
        newer_in_lane=currency[1],
        previous_timeout_minutes=prior_timeout,
        fix_only=fix_only)
    # The dynamic banner precedes the byte-unchanged PREAMBLE. Consequently
    # PREAMBLE's --- MESSAGE --- delimiter remains immediately before the
    # exact raw mailbox body, and the body remains the prompt's exact suffix.
    command = AGENT_COMMANDS[agent] + [banner + PREAMBLE + message]

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
    timed_out = False
    timeout_history_error = None
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
        if fix_only:
            env[FIX_ONLY_ENVIRONMENT] = "1"
        else:
            env.pop(FIX_ONLY_ENVIRONMENT, None)
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
            _rendezvous_turn_started()
            try:
                next_beat = started + 60.0
                deadline = started + DISPATCH_TIMEOUT_MINUTES * 60.0
                while proc.poll() is None:
                    time.sleep(5)
                    now = time.time()
                    if now >= deadline:
                        # The child can finish naturally during sleep. Poll
                        # once more at the deadline and kill only a process
                        # that is still live now; otherwise a successful turn
                        # would be mislabeled as timed out and poisoned with
                        # kill history.
                        if proc.poll() is not None:
                            break
                        # a hung CLI would hold this lane forever (seen live:
                        # a turn printed "Execution error" then produced
                        # nothing for 21 minutes). Kill it; the non-zero exit
                        # code below parks the claimed message in failed/.
                        proc.kill()
                        proc.wait()
                        timed_out = True
                        # The timeout setting is the stable killed-after
                        # threshold promised to a later retry. The poll loop
                        # can observe the child a fraction late; retain that
                        # elapsed value for diagnostics without letting
                        # scheduler jitter leak into the human-facing retry
                        # sentence.
                        killed_after_minutes = DISPATCH_TIMEOUT_MINUTES
                        observed_elapsed_minutes = (now - started) / 60.0
                        try:
                            write_timeout_history(
                                name=name,
                                killed_after_minutes=killed_after_minutes,
                                observed_elapsed_minutes=(
                                    observed_elapsed_minutes))
                        except (OSError, ValueError, json.JSONDecodeError,
                                OverflowError, RecursionError) as exc:
                            timeout_history_error = exc
                        print("  timed out " + name + " after "
                              + exact_duration(value=killed_after_minutes)
                              + " min; the turn was killed; its recovery "
                              "state will be verified after the log closes.")
                        break
                    if now >= next_beat:
                        elapsed_min = (now - started) / 60.0
                        log_kb = os.path.getsize(log_path) / 1024.0
                        print("  ... " + name + " still running "
                              + "(%.0f min elapsed, log %.1f kB; tail -f %s)"
                              % (elapsed_min, log_kb, log_path))
                        next_beat += 60.0
            finally:
                # If an unexpected monitor/log exception occurs, do not leave
                # an untracked child behind a future all-clear.  Reap it when
                # possible; otherwise the rendezvous permit remains visibly
                # in flight and permanently closes admissions.
                try:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                finally:
                    if proc.poll() is not None:
                        _rendezvous_turn_finished()
            f.write("\n--- rc=" + str(proc.returncode) + " ---\n")

    if launch_error is not None:
        parked = park_failed_message(dispatch_path=dispatch_path)
        state = "message parked in failed/" if parked \
            else "failed-state move was not verified"
        print("  !! dispatch could not start: " + str(launch_error)
              + "; " + state + "; log -> " + log_path)
        return False

    print("  rc=" + str(proc.returncode) + "  log -> " + log_path)
    # show the reply's tail on the terminal so activity is visible live.
    with open(log_path, encoding="utf-8") as f:
        reply_lines = f.read().strip().splitlines()
    for line in reply_lines[-8:]:
        print("  | " + line)

    if timed_out:
        if timeout_history_error is not None:
            # Without its durable marker, a requeue would present the killed
            # turn as fresh. Keep the claimed file out of the pending root
            # until a human can repair the sidecar failure.
            print("  !! could not persist timeout history: "
                  + str(timeout_history_error)
                  + "; leaving the claimed message in inflight/; log -> "
                  + log_path)
            return False
        if park_failed_message(dispatch_path=dispatch_path):
            print("  timeout recovery verified: message parked in failed/; "
                  "requeue it by moving it back to the mailbox (or relaunch "
                  "with a larger --dispatch-timeout).")
        else:
            print("  !! timeout recovery failed: the failed/ state was not "
                  "verified; inspect inflight/ before requeueing.")
        return False

    if proc.returncode != 0:
        # a failed dispatch is NOT done: park it in failed/ so it is never
        # silently consumed, and never hot-retried while the cause persists.
        # Requeue after fixing the cause:  mv ai/notes/mailbox/failed/<f> ai/notes/mailbox/
        parked = park_failed_message(dispatch_path=dispatch_path)
        # the turn's output lives in the log file (it streams there;
        # proc.stdout is None under Popen with a file handle).
        if not parked:
            print("  !! dispatch failed and its failed/ state was not "
                  "verified; inspect inflight/ and failed/; log -> "
                  + log_path)
        elif "Not logged in" in "\n".join(reply_lines):
            print("  !! the headless CLI is logged out; run `claude` in a "
                  "terminal, type /login, then requeue from failed/.")
        else:
            print("  !! dispatch failed; message parked in failed/, see "
                  "the log above.")
        return False

    return archive_consumed_message(dispatch_path=dispatch_path)


def park_failed_message(dispatch_path):
    """Move a claimed message to failed and verify its exact inode."""
    _, verified = verified_state_move(
        dispatch_path=dispatch_path,
        directory=os.path.join(MAILBOX, "failed"))
    return verified


def regular_inode(path):
    """Return ``(device, inode)`` only for an exact regular-file path."""
    try:
        details = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(details.st_mode):
        return None
    return details.st_dev, details.st_ino


def restore_state_source(guard_path, dispatch_path, source_inode):
    """Restore the exact claimed inode from its safety guard if necessary."""
    if not os.path.lexists(dispatch_path):
        try:
            os.link(guard_path, dispatch_path)
        except OSError:
            pass
    return regular_inode(path=dispatch_path) == source_inode


def remove_state_guard(guard_path, source_inode):
    """Remove only the unchanged safety hardlink owned by this move."""
    if regular_inode(path=guard_path) != source_inode:
        return False
    try:
        os.unlink(guard_path)
    except OSError:
        return False
    return not os.path.lexists(guard_path)


def verified_state_move(dispatch_path, directory):
    """Move one regular inode and prove the destination owns that inode.

    Returns:
      ``(destination, verified)``. The destination is None when publication
      itself failed; verification also requires the source path to be absent.
    """
    source_inode = regular_inode(path=dispatch_path)
    if source_inode is None:
        return None, False
    # move_without_overwrite() publishes by hardlink and then unlinks the
    # inflight source. Keep one same-inode guard beside that source until the
    # final destination identity is proven. A verification race can therefore
    # restore the exact inflight blocker, and a guard that itself cannot be
    # cleaned is recognized by inflight_lane_blockers() across later passes.
    guard_path = dispatch_path + STATE_GUARD_SUFFIX
    try:
        os.link(dispatch_path, guard_path)
    except OSError:
        return None, False
    if regular_inode(path=guard_path) != source_inode:
        return None, False
    destination = move_without_overwrite(
        path=dispatch_path,
        directory=directory)
    if destination is None:
        restored = restore_state_source(
            guard_path=guard_path,
            dispatch_path=dispatch_path,
            source_inode=source_inode)
        if restored:
            remove_state_guard(
                guard_path=guard_path,
                source_inode=source_inode)
        return None, False
    destination_inode = regular_inode(path=destination)
    verified = (destination_inode == source_inode
                and not os.path.lexists(dispatch_path))
    if not verified:
        restored = restore_state_source(
            guard_path=guard_path,
            dispatch_path=dispatch_path,
            source_inode=source_inode)
        if restored:
            remove_state_guard(
                guard_path=guard_path,
                source_inode=source_inode)
        return destination, False
    if not remove_state_guard(
            guard_path=guard_path,
            source_inode=source_inode):
        # A leftover exact-name guard is itself a durable lane blocker. Restore
        # the ordinary inflight name too when the guard still owns our inode.
        restore_state_source(
            guard_path=guard_path,
            dispatch_path=dispatch_path,
            source_inode=source_inode)
        return destination, False
    return destination, True


def archive_consumed_message(dispatch_path):
    """Move a clean dispatch to done and verify the archive before success.

    Returns:
      True only when the exact destination is a regular file after the move.
    """
    name = os.path.basename(dispatch_path)
    done_path, verified = verified_state_move(
        dispatch_path=dispatch_path,
        directory=DONE)
    if done_path is None:
        # Someone quarantined the inflight file by hand, or a historical
        # archive already owns the name. Never overwrite either state.
        print("  note: " + name + " could not move to done/; leaving the "
              "existing state untouched; dispatch is not consumed.")
        return False
    if not verified:
        print("  !! done archive verification failed for " + name
              + "; dispatch is not consumed.")
        return False
    print("  archived " + name + " in done/; dispatch consumed.")
    return True


def drain_lane(paths, dry_run, fix_only=False):
    """Dispatch ONE agent's pending messages, in order (a worker body).

    Arguments:
      paths   = this agent's message files, already sorted by sequence.
      dry_run  = True to print the would-be commands without running them.
      fix_only = True to launch only declared Sol closures.
    """
    all_consumed = True
    for path in paths:
        controller = (_ACTIVE_WATCH_RENDEZVOUS
                      if not dry_run else None)
        permit = None
        if controller is not None:
            permit = controller.begin_attempt()
            if permit is None:
                # A watch-global rendezvous is due.  Leave this exact root
                # message untouched; main performs the safe window only after
                # every lane worker has returned.
                break
            _RENDEZVOUS_LOCAL.permit = permit
        try:
            consumed = dispatch(path=path, dry_run=dry_run, fix_only=fix_only)
        finally:
            if controller is not None:
                try:
                    del _RENDEZVOUS_LOCAL.permit
                finally:
                    controller.finish_attempt(permit=permit)
        if not consumed:
            all_consumed = False
            # A false result can mean the head is still inflight because its
            # archive or failed-state move was ambiguous. Do not release later
            # work in the same lane past an unresolved head.
            break
    return all_consumed


def process_backlog(dry_run, fix_only=False):
    """Dispatch the whole backlog: lanes in PARALLEL, each lane in order.

    The three agents are independent sessions, so Opus can execute a unit
    while Sol attacks another -- but two messages to the SAME agent must
    stay sequential (a lane is one conversation partner, not a pool), and
    two agents sharing a WORKING DIRECTORY must too: concurrent turns in
    one git tree race each other's index (the 2026-07-14 incident where a
    live edit was swept into another agent's commit). So the parallel unit
    is the cwd: Fable+Opus (same worktree) serialize; Sol runs alongside.

    Arguments:
      dry_run  = True to print the would-be commands without running them.
      fix_only = True when a watch is closing existing ledger work only.

    Returns:
      None when there was no backlog, True when every message was consumed
      (or would dispatch in a dry run), and False when any dispatch or done
      archive failed.
    """
    backlog = pending_messages()
    blockers = inflight_lane_blockers()
    if not backlog:
        if not blockers:
            return None
        for cwd in sorted(blockers):
            report_inflight_lane_block(
                blocker_paths=blockers[cwd],
                pending_count=0)
        return False
    report_demand(backlog=backlog)
    lanes = {}
    for path in backlog:
        name = os.path.basename(path)
        agent = PENDING_MESSAGE_RE.match(name).group(1)
        cwd = AGENT_CWD[agent]
        if cwd not in lanes:
            lanes[cwd] = []
        lanes[cwd].append(path)
    # An inflight message predating this pass represents an unresolved turn:
    # it may have edited the shared tree even though its archive failed. Do
    # not release later work in that working-directory lane on a subsequent
    # watch pass. Other cwd lanes remain independent and may still drain.
    workers = []
    lane_outcomes = {}
    outcome_lock = threading.Lock()

    def drain_and_record(cwd, paths, dry_run, fix_only):
        """Run one cwd lane and retain failure even if its worker raises."""
        try:
            consumed = drain_lane(
                paths=paths, dry_run=dry_run, fix_only=fix_only)
        except Exception as exc:
            print("  !! dispatch lane failed: " + str(exc)
                  + "; lane is not consumed.")
            consumed = False
        with outcome_lock:
            lane_outcomes[cwd] = consumed

    for cwd in sorted(blockers):
        report_inflight_lane_block(
            blocker_paths=blockers[cwd],
            pending_count=len(lanes.get(cwd, [])))

    for cwd in sorted(lanes):
        if cwd in blockers:
            lane_outcomes[cwd] = False
            continue
        worker = threading.Thread(target=drain_and_record,
                                  kwargs={"cwd": cwd,
                                          "paths": lanes[cwd],
                                          "dry_run": dry_run,
                                          "fix_only": fix_only})
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
    return (not blockers
            and len(lane_outcomes) == len(lanes)
            and all(lane_outcomes.values()))


def report_demand(backlog):
    """Print the queue-depth line + the second-Implementer tripwire.

    The demand total is the queued mailbox messages PLUS the "- OPEN"
    lines of ai/notes/backlog.md (user rule, 2026-07-14: demand is what
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
    total = total_open_demand(backlog=backlog)
    print("queue depth: opus=" + str(depth["opus"])
          + " sol=" + str(depth["sol"])
          + " fable=" + str(depth["fable"])
          + " | open backlog (ai/notes/backlog.md): " + str(ledger)
          + " | total demand: " + str(total))
    if total >= SECOND_IMPLEMENTER_THRESHOLD:
        print("  hint: total open demand is at or past "
              + str(SECOND_IMPLEMENTER_THRESHOLD) + " units; the red "
              "team is now the second implementer: build units flow to "
              "it as well as to the primary Implementer route "
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


def send(agent, text, dry_run, ticket_kind=None):
    """Drop a new message into the mailbox (the loop's entry point).

    Arguments:
      agent   = "fable", "opus", or "sol".
      text    = the routing summary (point at ai/notes/; do not inline specs).
      dry_run    = True to print the file that would be queued and write
                   nothing. Rehearsing --send used to queue a real message
                   (main() returned before the dry-run branch ever ran), so a
                   junk body became a live dispatched turn as soon as a watch
                   picked it up -- the 0022 audit's unrunnable gate leg.
      ticket_kind = ``closure`` or ``discovery`` for public Sol work.  The
                    exact internal Sol ping alone uses ``transport``.

    Returns:
      True when the message was queued, or would be queued in a dry run.
    """
    def refusal_now():
        """Return a current Sol-send refusal without changing disk."""
        if agent != "sol":
            return None
        transport_valid = (
            ticket_kind == "transport"
            and text == transport_ping_text(agent="sol"))
        return sol_ticket_refusal(
            ticket_kind=ticket_kind,
            total=total_open_demand(),
            fix_only=(fix_only_environment_active()
                      or fix_only_watch_is_active()),
            transport_valid=transport_valid)

    reason = refusal_now()
    if reason is not None:
        print("refused --send sol: " + reason + ".")
        return False

    payload = text
    if agent == "sol":
        if ticket_kind in SOL_DISPATCH_TICKET_KINDS:
            payload = sol_ticket_payload(
                ticket_kind=ticket_kind, text=text)
        else:
            # refusal_now() already handles this path. Keep the invariant
            # explicit in case its policy is refactored later.
            print("refused --send sol: invalid ticket classification.")
            return False

    if dry_run:
        print("[dry-run] would queue "
              + os.path.join(MAILBOX, next_seq() + "-to-" + agent + ".md"))
        warn_if_mailbox_unwatched()
        return True
    os.makedirs(MAILBOX, exist_ok=True)
    lock_path = os.path.join(MAILBOX, ".sequence.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Concurrent senders can both observe threshold-minus-one before
            # either takes the sequence lock. Recheck while serialized so at
            # most one publishes across the boundary.
            reason = refusal_now()
            if reason is not None:
                print("refused --send sol: " + reason + ".")
                return False
            for _ in range(20):
                path = os.path.join(
                    MAILBOX,
                    next_seq() + "-to-" + agent + ".md")
                handle, temporary = tempfile.mkstemp(
                    prefix=".message-",
                    dir=MAILBOX)
                try:
                    with os.fdopen(handle, "w", encoding="utf-8") as f:
                        f.write(payload)
                        if not payload.endswith("\n"):
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
                    warn_if_mailbox_unwatched()
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
    global _ACTIVE_WATCH_RENDEZVOUS

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
    parser.add_argument("--fix-only", metavar="value", type=truthy_fix_only,
                        default=None,
                        help="with --watch, close existing ledger work only; "
                             "the value accepts 1, true, or yes in any "
                             "capitalization")
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
                             "summary pointing at ai/notes/)")
    parser.add_argument("--ticket-kind", choices=SOL_TICKET_KINDS,
                        help="required with --send sol: declare whether the "
                             "unit closes existing work or seeks new "
                             "findings")
    parser.add_argument("--architect-model", metavar="MODEL",
                        type=validate_model_name,
                        default=DEFAULT_ARCHITECT_MODEL,
                        help="Claude model alias or full name for the "
                             "Architect route (legacy fable address; "
                             "default: " + DEFAULT_ARCHITECT_MODEL + ")")
    parser.add_argument("--implementer-model", metavar="MODEL",
                        type=validate_model_name,
                        default=DEFAULT_IMPLEMENTER_MODEL,
                        help="Claude model alias or full name for the "
                             "Implementer route (legacy opus address; "
                             "default: " + DEFAULT_IMPLEMENTER_MODEL + ")")
    parser.add_argument("--fable-effort", default=DEFAULT_FABLE_EFFORT,
                        choices=CLAUDE_EFFORT_CHOICES,
                        help="claude CLI reasoning effort for the Architect "
                             "route (legacy fable address; default: "
                             + DEFAULT_FABLE_EFFORT + ")")
    parser.add_argument("--opus-effort", default=DEFAULT_OPUS_EFFORT,
                        choices=CLAUDE_EFFORT_CHOICES,
                        help="claude CLI reasoning effort for the Implementer "
                             "route (legacy opus address; default: "
                             + DEFAULT_OPUS_EFFORT + ")")
    parser.add_argument("--sol-effort", default=DEFAULT_SOL_EFFORT,
                        choices=CODEX_EFFORT_CHOICES,
                        help="codex CLI reasoning effort for Sol "
                             "dispatches (default: "
                             + DEFAULT_SOL_EFFORT + ")")
    parser.add_argument("--dispatch-timeout", metavar="MINUTES",
                        type=positive_int, default=DISPATCH_TIMEOUT_MINUTES,
                        help="kill a dispatched turn that runs past "
                             "this many minutes and park its message "
                             "in failed/ (default: "
                             + str(DISPATCH_TIMEOUT_MINUTES) + ")")
    parser.add_argument("--claude-context", metavar="TOKENS",
                        type=int, default=DEFAULT_CLAUDE_CONTEXT_BUDGET,
                        help="Architect and Implementer Claude turns compact "
                             "their context whenever it reaches this many "
                             "tokens (default: "
                             + str(DEFAULT_CLAUDE_CONTEXT_BUDGET) + ")")
    parser.add_argument("--sol-context", metavar="TOKENS",
                        type=int, default=DEFAULT_SOL_CONTEXT_BUDGET,
                        help="Sol turns compact their context whenever "
                             "it reaches this many tokens (default: "
                             + str(DEFAULT_SOL_CONTEXT_BUDGET) + ")")
    args = parser.parse_args()

    if args.fix_only is not None:
        conflicting_action = (
            not args.watch or args.once or args.send is not None
            or args.ping is not None or args.dry_run)
        if conflicting_action:
            print("--fix-only is valid only with --watch by itself")
            return 1
    if args.ticket_kind is not None and args.send != "sol":
        print("--ticket-kind is valid only with --send sol")
        return 1
    if args.send == "sol" and args.ticket_kind is None:
        print("--send sol needs --ticket-kind closure or discovery; "
              "the daemon will not guess from prose")
        return 1
    primary_actions = sum((
        bool(args.once),
        bool(args.watch),
        args.send is not None,
        args.ping is not None,
    ))
    if primary_actions > 1:
        print("choose only one primary action: --once, --watch, --send, "
              "or --ping")
        return 1
    if args.watch and args.dry_run:
        print("--dry-run is finite and cannot be combined with --watch")
        return 1

    fix_only = args.fix_only is True

    DISPATCH_TIMEOUT_MINUTES = args.dispatch_timeout
    CLAUDE_CONTEXT_BUDGET = args.claude_context

    # Rebuild the dispatch commands at the requested models and efforts. The
    # watch start lines echo both so terminal scroll-back identifies the exact
    # role assignment independently of the legacy route filenames.
    AGENT_COMMANDS = build_agent_commands(
        fable_effort=args.fable_effort,
        opus_effort=args.opus_effort,
        sol_effort=args.sol_effort,
        sol_context_budget=args.sol_context,
        architect_model=args.architect_model,
        implementer_model=args.implementer_model)
    if args.watch:
        print("role models: architect=" + args.architect_model
              + " implementer=" + args.implementer_model
              + " (legacy routes fable/opus)")
        print("effort levels: architect/fable=" + args.fable_effort
              + " implementer/opus=" + args.opus_effort
              + " sol=" + args.sol_effort)
        print("context budgets: architect/implementer="
              + str(args.claude_context)
              + " sol=" + str(args.sol_context)
              + " tokens (a turn compacts at its budget)")

    if args.ping:
        ping_text = transport_ping_text(agent=args.ping)
        queued = send(
            agent=args.ping,
            text=ping_text,
            dry_run=args.dry_run,
            ticket_kind="transport" if args.ping == "sol" else None)
        return 0 if queued else 1

    if args.send:
        if not args.unit:
            print("--send needs --unit with the routing-summary text")
            return 1
        queued = send(
            agent=args.send,
            text=args.unit,
            dry_run=args.dry_run,
            ticket_kind=args.ticket_kind)
        return 0 if queued else 1

    if args.dry_run:
        outcome = process_backlog(dry_run=args.dry_run)
        if outcome is None:
            print("mailbox empty")
            return 0
        if not outcome:
            print("one or more mailbox messages would not be consumed.")
            return 1
        return 0

    if args.once:
        dispatch_lock = acquire_dispatch_lock(mode="once")
        if dispatch_lock is None:
            return 1
        try:
            outcome = process_backlog(dry_run=False)
            if outcome is None:
                print("mailbox empty")
            elif not outcome:
                print("one or more mailbox messages were not consumed.")
                return 1
        finally:
            release_dispatch_lock(lock_file=dispatch_lock)
        return 0

    if args.watch:
        # --once and --watch share one kernel-released lock. This closes both
        # the check-then-write race between watchers and the older gap where
        # --once could overlap a live watcher in the same working directory.
        dispatch_lock = acquire_dispatch_lock(mode="watch")
        if dispatch_lock is None:
            return 1
        fix_only_lock = None
        if fix_only:
            fix_only_lock = acquire_fix_only_lock()
            if fix_only_lock is None:
                release_dispatch_lock(lock_file=dispatch_lock)
                return 1
        print("watching " + MAILBOX + " (stop only when an all-lanes-idle "
              "line says Ctrl-C is safe; never stop while a turn is in "
              "flight)")
        if fix_only:
            print("fix-only watch active: closing existing ledger work "
                  "only; no discovery tickets or new backlog lines")
        # a daemon fix is a no-op for the loop already running (the
        # 2026-07-14 placeholder incident): watch our own source and
        # exit when it changes, so stale code can never keep dispatching.
        # Exiting (not self-reloading) is deliberate -- a restart is one
        # keystroke and never picks up a half-saved edit.
        source_path = os.path.abspath(__file__)
        source_stamp = os.path.getmtime(source_path)
        rendezvous = SafeKillRendezvous(
            source_path=source_path, source_stamp=source_stamp)
        _ACTIVE_WATCH_RENDEZVOUS = rendezvous
        first_pass = True
        try:
            while True:
                # Preserve the existing first-pass call shape for finite
                # witnesses, then check before every later release as well as
                # after every joined pass.  A source edit during an idle safe
                # interval therefore cannot receive one stale dispatch.
                if (not first_pass
                        and os.path.getmtime(source_path) != source_stamp):
                    print("daemon source changed on disk; exiting so "
                          "the next start runs it (relaunch --watch).")
                    return 0
                first_pass = False
                if fix_only:
                    process_backlog(dry_run=False, fix_only=True)
                else:
                    process_backlog(dry_run=False)
                if (rendezvous.source_changed()
                        or os.path.getmtime(source_path) != source_stamp):
                    print("daemon source changed on disk; exiting so "
                          "the next start runs it (relaunch --watch).")
                    return 0
                if rendezvous.window_ready():
                    run_safe_kill_countdown(controller=rendezvous)
                    # Queued work resumes immediately after the manufactured
                    # window rather than paying an extra ordinary poll delay.
                    continue
                ordinary_safe = report_ordinary_safe_poll(
                    controller=rendezvous)
                time.sleep(WATCH_POLL_SECONDS)
                if ordinary_safe:
                    # The next loop may spawn lane workers.  Expire the
                    # visible safe status in the main thread before any such
                    # worker can receive an admission permit.
                    report_safe_interval_closed()
        finally:
            _ACTIVE_WATCH_RENDEZVOUS = None
            if fix_only_lock is not None:
                release_fix_only_lock(lock_file=fix_only_lock)
            release_dispatch_lock(lock_file=dispatch_lock)

    print("choose one of --dry-run / --once / --watch / --send (see --help)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
