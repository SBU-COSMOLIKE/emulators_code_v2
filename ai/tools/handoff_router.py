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
3. waits for the Implementer's returned block and proves that every planned
   subagent returned structured evidence;
4. runs the local check commands and saves their exact output;
5. puts the Implementer result and check output on the clipboard for the
   Architect.

The Architect audits that result next. A ``GO`` authorizes the mailbox daemon
to create the ticket's landing commit; it does not tell the Architect to merge
or push. When the saved role plan includes Red Team, the daemon starts the
separate post-landing review after it records that commit. This router never
inserts Red Team between implementation and the Architect's audit.

Usage:

    python ai/tools/handoff_router.py --note ai/notes/<spec>.md --section \\
        "Implementation directive"
    python ai/tools/handoff_router.py --note ai/notes/<spec>.md

The source note may choose three roles or two roles. ``--mode``,
``--skip-redteam``, and ``--severity`` can confirm that saved choice. They
cannot change it. A plan that assigns Sol to implementation is unsupported
and is refused before the router copies or saves anything.

Use ``--gate-cmd`` to name a ticket's local check command:

    --gate-cmd "PYTHONPATH=. <cocoa-python> ai/gates/checks/<child>.py"

Use ``--status`` to read the current Git branches and saved records without
changing the clipboard or waiting for another conversation:

    python ai/tools/handoff_router.py --status
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
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
from handoff_contract import extract_blocked_implementer_capability_evidence
from handoff_contract import extract_implementer_subagent_evidence
from handoff_contract import nonnegative_character_limit
from handoff_contract import resolve_character_limit
from handoff_contract import validate_directive_file
from handoff_contract import validate_implementer_subagent_evidence
from handoff_contract import validate_implementer_handoff_subagent_evidence
RELAY_DIR = os.path.join(NOTES_DIR, "relay")
DISCOVERY_SEVERITIES = ("high", "medium", "low")
DISCOVERY_SEVERITY_ENVIRONMENT = "MAILBOX_DISCOVERY_SEVERITY"
RUN_RESERVATIONS_DIR = os.path.join(RELAY_DIR, ".router-runs")
ROUTER_LOCK_PATH = os.path.join(
    tempfile.gettempdir(),
    "cocoa-handoff-router-" + str(os.getuid()) + ".lock",
)
IMPLEMENTER_SUBAGENT_EVIDENCE_MARKER = "- **Subagent work:**"
ROUTE_RECORD_NAME = ".active-route"
SUPPORTING_COPY_PREFIX = (
    "<!-- SUPPORTING COPY ONLY. The agent-written source note\n"
    "     that this block cites remains authoritative.\n"
    "     Saved by ai/tools/handoff_router.py. -->\n\n")
GATE_RECEIPT_RE = re.compile(
    r"^<!-- ROUTER GATES v1 commands=([0-9a-f]{64}) "
    r"result=(pass|fail) body=([0-9a-f]{64}) -->$")
CANDIDATE_COMMIT_RE = re.compile(
    r"^- \*\*Candidate commit:\*\* `([0-9a-f]{40})`$")

COCOA_PYTHON = ("/Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa"
                "/.local/bin/python")

DEFAULT_GATE_COMMANDS = [
    COCOA_PYTHON + " -m compileall -q ai/gates emulator",
    "PYTHONPATH=. " + COCOA_PYTHON + " ai/gates/run_board.py --list",
    "PYTHONPATH=. " + COCOA_PYTHON + " ai/gates/checks/board_selftest.py",
]

PRIMARY_STATE_RELATIVE = os.path.join(
    ".claude", "worktrees", ".mailbox-primary-worktree.json")
ARCHITECT_BRANCH_PREFIX = "refs/heads/claude/"
RESERVED_ROLE_NAMES = frozenset({"mailbox-implementer", "mailbox-sol"})
RESERVED_ROLE_BRANCHES = frozenset({
    "refs/heads/claude/mailbox-implementer",
    "refs/heads/codex/mailbox-sol",
})
PRIMARY_STATE_SCHEMA = 3
PRIMARY_TOPOLOGY = "separate-role-worktrees-v1"
MAX_PRIMARY_STATE_BYTES = 16 * 1024
MAX_BACKLOG_BYTES = 16 * 1024 * 1024
OPEN_BACKLOG_TICKET_RE = re.compile(
    r"^- OPEN \*\*(CRITICAL|HIGH|MEDIUM|LOW)\*\* "
    r"\*\*(BUG FIX|NEW FUNCTIONALITY)\*\* — "
    r"\[([^\]\n]+)\]\(#([a-z0-9](?:[a-z0-9-]*[a-z0-9])?)\)$")
NEAR_OPEN_BACKLOG_TICKET_RE = re.compile(
    r"^[ \t]*-[ \t]*open\b", re.IGNORECASE)
DETAIL_ANCHOR_RE = re.compile(
    r'^<a id="([a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"></a>$')
REOPEN_COUNT_RE = re.compile(
    r"^\*\*Red Team reopen count: (0|[1-9][0-9]*)\.\*\*$")
NEAR_REOPEN_COUNT_RE = re.compile(
    r"^[ \t]*(?:[-+*][ \t]+)?(?:\*\*)?[ \t]*"
    r"red(?:[ -]+)team[ \t]+reopen[ \t]+count\b",
    re.IGNORECASE)
class BacklogLedgerError(RuntimeError):
    """The saved primary backlog cannot safely authorize a role decision."""


class StatusError(RuntimeError):
    """Git could not provide a trustworthy work-status answer."""


def _read_regular_file(path, label, maximum_bytes):
    """Read one bounded regular file without following redirected paths."""
    path = os.path.abspath(path)
    if os.path.realpath(path) != path:
        raise BacklogLedgerError(label + " uses a redirected path: " + path)
    try:
        initial = os.lstat(path)
    except OSError as exc:
        raise BacklogLedgerError(
            "cannot inspect " + label + " " + path + ": " + str(exc))
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise BacklogLedgerError(label + " is not a regular file: " + path)
    if initial.st_size > maximum_bytes:
        raise BacklogLedgerError(label + " is too large: " + path)
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BacklogLedgerError(
            "cannot open " + label + " " + path + ": " + str(exc))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BacklogLedgerError(
                label + " is not a regular file: " + path)
        payload = os.read(descriptor, maximum_bytes + 1)
        after = os.fstat(descriptor)
        current = os.lstat(path)

        def snapshot(info):
            return (info.st_dev, info.st_ino, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)

        if (snapshot(initial) != snapshot(before)
                or snapshot(before) != snapshot(after)
                or snapshot(after) != snapshot(current)
                or after.st_size != len(payload)):
            raise BacklogLedgerError(label + " changed while read: " + path)
    except OSError as exc:
        raise BacklogLedgerError(
            "cannot read " + label + " " + path + ": " + str(exc))
    finally:
        os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise BacklogLedgerError(label + " is too large: " + path)
    return payload


def _json_object_without_duplicate_keys(pairs):
    """Build one JSON object while refusing repeated security fields."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise BacklogLedgerError(
                "primary-worktree state repeats key " + repr(key))
        result[key] = value
    return result


def _checked_git(cwd, arguments, label):
    """Return exact Git output or fail the authoritative-path proof."""
    try:
        proc = subprocess.run(
            ["git"] + list(arguments), cwd=cwd, capture_output=True)
    except OSError as exc:
        raise BacklogLedgerError(
            "cannot inspect " + label + ": " + str(exc))
    try:
        stdout = proc.stdout.decode("utf-8", errors="strict")
        stderr = proc.stderr.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BacklogLedgerError(
            "cannot inspect " + label + ": Git output is not UTF-8: "
            + str(exc))
    if proc.returncode != 0:
        detail = stderr.strip()
        if detail:
            detail = ": " + detail
        raise BacklogLedgerError("cannot inspect " + label + detail)
    return stdout.strip()


def _git_common_directory(checkout):
    """Return the real common Git directory for one checkout."""
    value = _checked_git(
        checkout, ["rev-parse", "--path-format=absolute", "--git-common-dir"],
        "the repository common Git directory")
    return os.path.realpath(value)


def authoritative_backlog_path():
    """Resolve ``backlog.md`` only from the saved Claude-primary record.

    Tests may replace this zero-argument resolver with a scratch resolver.
    Production callers never derive the ledger from the directive's execution
    checkout, because that ignored file may differ between worktrees.
    """
    common = _git_common_directory(REPO_ROOT)
    if os.path.basename(common) != ".git" or not os.path.isdir(common):
        raise BacklogLedgerError(
            "repository common Git directory is not a normal .git directory")
    repository = os.path.dirname(common)
    state_path = os.path.join(repository, PRIMARY_STATE_RELATIVE)
    payload = _read_regular_file(
        state_path, "primary-worktree state", MAX_PRIMARY_STATE_BYTES)
    try:
        state = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_json_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BacklogLedgerError(
            "primary-worktree state is not exact UTF-8 JSON: " + str(exc))
    if not isinstance(state, dict):
        raise BacklogLedgerError("primary-worktree state must be an object")
    schema = state.get("schema")
    expected_keys = {
        "schema", "repository", "name", "path", "branch", "topology"}
    if type(schema) is int and schema in {1, 2}:
        raise BacklogLedgerError(
            "retired primary-worktree state schema; stop old mailbox "
            "processes; preserve and update the saved primary worktree; "
            "move the retired state file aside for recovery; then run the "
            "current `python3 ai/tools/mailbox_daemon.py --once` there")
    if type(schema) is not int or schema != PRIMARY_STATE_SCHEMA:
        raise BacklogLedgerError("unsupported primary-worktree state schema")
    if set(state) != expected_keys:
        raise BacklogLedgerError(
            "primary-worktree state has unexpected or missing keys")
    for key in ("repository", "name", "path", "branch"):
        value = state[key]
        if (not isinstance(value, str) or not value
                or "\x00" in value or "\n" in value or "\r" in value):
            raise BacklogLedgerError(
                "invalid primary-worktree state field " + key)
    if state["topology"] != PRIMARY_TOPOLOGY:
        raise BacklogLedgerError("unsupported primary-worktree topology")
    if (not os.path.isabs(state["repository"])
            or state["repository"] != common):
        raise BacklogLedgerError(
            "primary-worktree state names a different repository")
    primary = os.path.abspath(state["path"])
    managed = os.path.join(repository, ".claude", "worktrees")
    if (not os.path.isabs(state["path"])
            or os.path.dirname(primary) != os.path.abspath(managed)
            or state["name"] != os.path.basename(primary)
            or state["name"] in {".", ".."}
            or "/" in state["name"]
            or state["name"] in RESERVED_ROLE_NAMES
            or not state["branch"].startswith(ARCHITECT_BRANCH_PREFIX)
            or state["branch"] in RESERVED_ROLE_BRANCHES):
        raise BacklogLedgerError(
            "primary-worktree state does not name the managed Claude primary")
    try:
        primary_info = os.lstat(primary)
    except OSError as exc:
        raise BacklogLedgerError(
            "cannot inspect saved primary worktree: " + str(exc))
    if (stat.S_ISLNK(primary_info.st_mode)
            or not stat.S_ISDIR(primary_info.st_mode)
            or os.path.realpath(primary) != primary):
        raise BacklogLedgerError(
            "saved primary worktree is not a real managed directory")

    registry = _checked_git(
        repository, ["worktree", "list", "--porcelain"],
        "the registered worktrees")
    matches = []
    for record_text in registry.split("\n\n"):
        fields = record_text.splitlines()
        if not fields or not fields[0].startswith("worktree "):
            continue
        record = {"path": fields[0][len("worktree "):], "branch": None,
                  "detached": False, "prunable": False}
        for field in fields[1:]:
            if field.startswith("branch "):
                record["branch"] = field[len("branch "):]
            elif field == "detached":
                record["detached"] = True
            elif field.startswith("prunable"):
                record["prunable"] = True
        if os.path.realpath(record["path"]) == primary:
            matches.append(record)
    if (len(matches) != 1 or matches[0]["branch"] != state["branch"]
            or matches[0]["detached"] or matches[0]["prunable"]):
        raise BacklogLedgerError(
            "saved primary worktree is not uniquely registered on "
            + state["branch"])
    if _git_common_directory(primary) != common:
        raise BacklogLedgerError(
            "saved primary worktree belongs to a different repository")
    top = _checked_git(
        primary, ["rev-parse", "--show-toplevel"],
        "the saved primary top level")
    branch = _checked_git(
        primary, ["symbolic-ref", "--quiet", "HEAD"],
        "the saved primary branch")
    if os.path.realpath(top) != primary or branch != state["branch"]:
        raise BacklogLedgerError(
            "saved primary checkout does not match its registered path and "
            "branch")
    backlog = os.path.join(primary, "ai", "notes", "backlog.md")
    if os.path.realpath(backlog) != os.path.abspath(backlog):
        raise BacklogLedgerError(
            "authoritative backlog uses a redirected path: " + backlog)
    return backlog


def backlog_severity_counts(backlog_path=None):
    """Return fully validated open-ticket counts from the primary backlog."""
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0,
              "high_bug_fix": 0, "high_new_functionality": 0,
              "unclassified": 0}
    if backlog_path is None:
        backlog_path = authoritative_backlog_path()
    try:
        text = _read_regular_file(
            backlog_path, "authoritative backlog", MAX_BACKLOG_BYTES).decode(
                "utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BacklogLedgerError(
            "authoritative backlog is not UTF-8: " + str(exc))
    index_anchors = set()
    index_severities = {}
    detail_anchor_counts = {}
    detail_anchor_positions = {}
    invalid_detail_anchors = set()
    lines = text.splitlines()
    for line_number, line in enumerate(lines):
        detail_match = DETAIL_ANCHOR_RE.fullmatch(line)
        if detail_match is not None:
            anchor = detail_match.group(1)
            detail_anchor_counts[anchor] = (
                detail_anchor_counts.get(anchor, 0) + 1)
            detail_anchor_positions.setdefault(anchor, []).append(line_number)
            if (line_number + 1 >= len(lines)
                    or not lines[line_number + 1].startswith("## ")):
                invalid_detail_anchors.add(anchor)
        if NEAR_OPEN_BACKLOG_TICKET_RE.match(line) is None:
            continue
        match = OPEN_BACKLOG_TICKET_RE.fullmatch(line)
        if (match is None
                or (match.group(1) == "CRITICAL"
                    and match.group(2) != "BUG FIX")
                or match.group(3).strip() != match.group(3)
                or not any(character.isalnum()
                           for character in match.group(3))):
            counts["unclassified"] += 1
            continue
        anchor = match.group(4)
        if anchor in index_anchors:
            counts["unclassified"] += 1
            continue
        index_anchors.add(anchor)
        severity = match.group(1).lower()
        index_severities[anchor] = severity
        ticket_type = match.group(2).lower().replace(" ", "_")
        counts[severity] += 1
        if severity == "high":
            counts["high_" + ticket_type] += 1
    for anchor in index_anchors:
        if (detail_anchor_counts.get(anchor, 0) != 1
                or anchor in invalid_detail_anchors):
            counts["unclassified"] += 1
            continue
        position = detail_anchor_positions[anchor][0]
        end = next(
            (line_number for line_number in range(position + 2, len(lines))
             if (lines[line_number].startswith("# ")
                 or lines[line_number].startswith("## "))),
            len(lines))
        reopen_rows = [
            line for line in lines[position + 2:end]
            if NEAR_REOPEN_COUNT_RE.match(line) is not None]
        canonical_rows = [
            REOPEN_COUNT_RE.fullmatch(line) for line in reopen_rows]
        canonical_rows = [match for match in canonical_rows
                          if match is not None]
        if len(reopen_rows) != 1 or len(canonical_rows) != 1:
            counts["unclassified"] += 1
            continue
        reopen_count = int(canonical_rows[0].group(1))
        if reopen_count > 5 and index_severities[anchor] != "low":
            counts["unclassified"] += 1
    return counts


def widespread_review_refusal(counts=None):
    """Return a refusal message unless widespread discovery may begin.

    The structured Architect field is the only switch for widespread review.
    A malformed open backlog line cannot prove that the Critical, High, and
    Medium groups are empty, so it fails closed before router side effects.
    """
    if counts is None:
        counts = backlog_severity_counts()
    if counts["unclassified"]:
        return (
            "refused widespread Red Team review: the authoritative backlog "
            "has " + str(counts["unclassified"])
            + " malformed open ticket(s), so the router cannot prove that "
            "its Critical, High, and Medium groups are empty")
    non_low = counts["critical"] + counts["high"] + counts["medium"]
    if non_low:
        return (
            "refused widespread Red Team review: the authoritative backlog "
            "has " + str(non_low) + " open Critical, High, or Medium "
            "ticket(s); widespread discovery may begin only when that count "
            "is zero")
    return None


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


def _read_recovery_file(path, label, maximum_bytes):
    """Read a recovery file without accepting a redirected final entry."""
    if os.path.islink(path):
        raise BacklogLedgerError(label + " must not be a symlink")
    return _read_regular_file(os.path.realpath(path), label, maximum_bytes)


def active_route_record():
    """Return the validated active manual route, or None when idle."""
    record_path = os.path.join(
        RUN_RESERVATIONS_DIR, ROUTE_RECORD_NAME)
    if not os.path.lexists(record_path):
        return None
    data = _read_recovery_file(
        record_path, "router recovery record", 16 * 1024)
    try:
        record = tuple(data.decode("utf-8", errors="strict").splitlines())
    except UnicodeError as exc:
        raise BacklogLedgerError(
            "router recovery record is not UTF-8") from exc
    valid_shape = (
        (len(record) == 6 and record[0] == "route-v2")
        or (len(record) == 8 and record[0] == "route-v3"))
    if (not valid_shape
            or re.fullmatch(
                r"[0-9]{8}-[0-9]{6}(?:-[0-9]+)?", record[1]) is None
            or re.fullmatch(r"[0-9a-f]{40}", record[3]) is None
            or re.fullmatch(r"[0-9a-f]{64}", record[4]) is None
            or re.fullmatch(r"[0-9a-f]{64}", record[5]) is None
            or (len(record) == 8 and (
                re.fullmatch(r"[0-9a-f]{40}", record[6]) is None
                or re.fullmatch(r"[0-9a-f]{64}", record[7]) is None))
            or data != ("\n".join(record) + "\n").encode("utf-8")):
        raise BacklogLedgerError("router recovery record is malformed")
    try:
        reservation = os.lstat(os.path.join(
            RUN_RESERVATIONS_DIR, record[1]))
    except OSError as exc:
        raise BacklogLedgerError(
            "router sequence reservation is missing") from exc
    if not stat.S_ISDIR(reservation.st_mode):
        raise BacklogLedgerError(
            "router sequence reservation must be a plain directory")
    return record


def route_sequence(note_path, note_display, base, commands, create=True):
    """Resume one exact route, or optionally save a new one before work."""
    note_digest = hashlib.sha256(_read_recovery_file(
        note_path, "Architect source note", MAX_BACKLOG_BYTES)).hexdigest()
    commands_digest = gate_commands_digest(commands)
    os.makedirs(RUN_RESERVATIONS_DIR, exist_ok=True)
    record = active_route_record()
    if record is not None:
        if record[2:6] != (
                note_display, base, note_digest, commands_digest):
            raise BacklogLedgerError(
                "an unfinished route names a different note, version, or "
                "check command list")
        return record[1]
    if not create:
        return None
    seq = reserve_run_sequence()
    if os.path.lexists(os.path.join(RELAY_DIR, seq + "-implementer.md")):
        raise BacklogLedgerError(
            "new route collides with existing Implementer evidence")
    record_path = os.path.join(RUN_RESERVATIONS_DIR, ROUTE_RECORD_NAME)
    fields = (
        "route-v2", seq, note_display, base, note_digest, commands_digest)
    publish_complete_text(
        path=record_path, text="\n".join(fields) + "\n")
    return seq


def abandon_active_route(expected_sequence):
    """Remove only the exact active-route pointer named by the user."""
    record = active_route_record()
    if record is None:
        raise BacklogLedgerError("there is no active manual route")
    if record[1] != expected_sequence:
        raise BacklogLedgerError(
            "requested sequence does not match the active manual route")
    os.remove(os.path.join(RUN_RESERVATIONS_DIR, ROUTE_RECORD_NAME))
    return record


def recovered_implementer_return(seq):
    """Return one complete saved handoff, or None before it is published."""
    path = os.path.join(RELAY_DIR, seq + "-implementer.md")
    if not os.path.lexists(path):
        return None
    data = _read_recovery_file(
        path, "saved Implementer return", MAX_BACKLOG_BYTES)
    text = data.decode("utf-8", errors="strict")
    if not text.startswith(SUPPORTING_COPY_PREFIX):
        raise BacklogLedgerError(
            "saved Implementer return has no complete supporting-copy header")
    return text[len(SUPPORTING_COPY_PREFIX):]


def implementer_candidate_commit(handoff):
    """Read the one canonical candidate row from a saved handoff."""
    candidate_lines = [line for line in handoff.splitlines()
                       if "Candidate commit:" in line]
    if not candidate_lines:
        return None
    matches = [CANDIDATE_COMMIT_RE.fullmatch(line)
               for line in candidate_lines]
    if len(matches) != 1 or matches[0] is None:
        raise BacklogLedgerError(
            "saved Implementer return has an invalid candidate commit row")
    return matches[0].group(1)


def recovered_candidate_commit(note_path, note_display, base, commands):
    """Return the candidate named by this route's complete saved return."""
    seq = route_sequence(
        note_path, note_display, base, commands, create=False)
    if seq is None:
        return None
    record = active_route_record()
    handoff = recovered_implementer_return(seq)
    if handoff is None:
        return record[6] if len(record) == 8 else None
    candidate = implementer_candidate_commit(handoff)
    if (len(record) == 8 and (
            candidate != record[6]
            or hashlib.sha256(handoff.encode("utf-8")).hexdigest()
            != record[7])):
        raise BacklogLedgerError(
            "saved Implementer return changed after candidate binding")
    return candidate


def remember_candidate_return(seq, candidate, handoff):
    """Bind an accepted candidate before publishing its full return."""
    record = active_route_record()
    if record is None or record[1] != seq:
        raise BacklogLedgerError("active route changed before candidate save")
    if len(record) == 8 and record[6] != candidate:
        raise BacklogLedgerError(
            "replacement Implementer return names a different candidate")
    saved_handoff = handoff if handoff.endswith("\n") else handoff + "\n"
    fields = ("route-v3",) + record[1:6] + (
        candidate,
        hashlib.sha256(saved_handoff.encode("utf-8")).hexdigest())
    publish_complete_text(
        path=os.path.join(RUN_RESERVATIONS_DIR, ROUTE_RECORD_NAME),
        text="\n".join(fields) + "\n")


def finish_route():
    """Record that this route no longer needs crash recovery."""
    os.remove(os.path.join(RUN_RESERVATIONS_DIR, ROUTE_RECORD_NAME))


def gate_commands_digest(commands):
    """Bind the ordered check commands without executing them."""
    payload = json.dumps(
        list(commands), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def recovered_gate_result(seq, commands):
    """Return a verified saved check result, or None before publication."""
    path = os.path.join(RELAY_DIR, seq + "-gates-log.md")
    if not os.path.lexists(path):
        return None
    data = _read_recovery_file(path, "saved check log", MAX_BACKLOG_BYTES)
    text = data.decode("utf-8", errors="strict")
    if not text.startswith(SUPPORTING_COPY_PREFIX):
        raise BacklogLedgerError("saved check log has no complete header")
    receipt, separator, body = text[len(SUPPORTING_COPY_PREFIX):].partition(
        "\n")
    if not receipt.startswith("<!-- ROUTER GATES v1 "):
        return None
    match = GATE_RECEIPT_RE.fullmatch(receipt)
    if (not separator or match is None
            or match.group(1) != gate_commands_digest(commands)
            or match.group(3) != hashlib.sha256(
                body.encode("utf-8")).hexdigest()):
        raise BacklogLedgerError("saved check log does not match this route")
    return (os.path.relpath(path, REPO_ROOT), match.group(2) == "pass")


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


def publish_complete_text(path, text):
    """Make a complete synced text file visible in one atomic step."""
    directory = os.path.dirname(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(path) + ".tmp-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass


def archive(seq, name, text):
    """Save one supporting copy under ai/notes/relay/ and return its path.

    Arguments:
      seq  = the run sequence stamp (shared by all files of this run).
      name = short role tag for the filename ("implementer", "sol", ...).
      text = the returned block or command-output text.
    """
    os.makedirs(RELAY_DIR, exist_ok=True)
    path = os.path.join(RELAY_DIR, seq + "-" + name + ".md")
    payload = (SUPPORTING_COPY_PREFIX
               + text + ("" if text.endswith("\n") else "\n"))
    if len(payload.encode("utf-8")) > MAX_BACKLOG_BYTES:
        raise BacklogLedgerError(
            "supporting copy is too large for safe recovery")
    publish_complete_text(path=path, text=payload)
    return os.path.relpath(path, REPO_ROOT)


def manual_capability_cycle(directive, source_note):
    """Bind a manual checkpoint to one canonical source note and base."""
    if (not isinstance(source_note, str)
            or not source_note.startswith("ai/notes/")
            or source_note.count("/") != 2):
        raise DirectiveError(
            "manual capability checkpoint needs one canonical source note")
    note_digest = hashlib.sha256(
        source_note.encode("utf-8")).hexdigest()
    return ("manual-router-" + note_digest + "@"
            + directive["execution_checkout"]["Base"])


def save_manual_capability_checkpoint(seq, cycle, source_note, archive_path,
                                      handoff_text, capability_failure):
    """Bind one blocked return to bytes that the router actually received."""
    saved_handoff = (handoff_text if handoff_text.endswith("\n")
                     else handoff_text + "\n")
    digest = hashlib.sha256(saved_handoff.encode("utf-8")).hexdigest()
    payload = {
        "schema": 3,
        "cycle": cycle,
        "source_note": source_note,
        "handoff_sha256": digest,
        "archive": archive_path,
        "capability_checked": capability_failure["capability_checked"],
        "attempted_operation": capability_failure["attempted_operation"],
        "raw_failure": capability_failure["raw_failure"],
    }
    path = os.path.join(RELAY_DIR, seq + "-capability-checkpoint.json")
    publish_complete_text(
        path=path,
        text=json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return digest


def verify_manual_capability_checkpoint(directive, source_note):
    """Prove a capability exception against a saved blocked handoff."""
    if directive["parallel_work_plan"]["mode"] != "capability-unavailable":
        return
    expected = directive["capability_checkpoint"]
    current_cycle = manual_capability_cycle(
        directive=directive, source_note=source_note)
    if expected["cycle"] != current_cycle:
        raise DirectiveError(
            "capability checkpoint Source cycle does not match this manual "
            "router checkout")
    matches = 0
    for name in os.listdir(RELAY_DIR):
        if not name.endswith("-capability-checkpoint.json"):
            continue
        path = os.path.join(RELAY_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if (set(payload) != {
                    "schema", "cycle", "source_note", "handoff_sha256",
                    "archive",
                    "capability_checked", "attempted_operation",
                    "raw_failure"}
                    or payload["schema"] != 3
                    or payload["cycle"] != expected["cycle"]
                    or payload["source_note"] != source_note
                    or payload["handoff_sha256"]
                    != expected["handoff_sha256"]):
                continue
            archive_path = os.path.join(REPO_ROOT, payload["archive"])
            with open(archive_path, "r", encoding="utf-8") as stream:
                saved = stream.read()
            marker = "     Saved by ai/tools/handoff_router.py. -->\n\n"
            if marker not in saved:
                continue
            handoff = saved.split(marker, 1)[1]
            capability = extract_blocked_implementer_capability_evidence(
                handoff_text=handoff)
            expected_failure = directive["parallel_work_plan"]
            exact_failure = all(
                payload[field] == capability[field]
                and payload[field] == expected_failure[field]
                for field in ("capability_checked", "attempted_operation",
                              "raw_failure"))
            if (hashlib.sha256(handoff.encode("utf-8")).hexdigest()
                    == expected["handoff_sha256"] and exact_failure):
                matches += 1
        except (OSError, UnicodeError, ValueError, TypeError,
                DirectiveError):
            continue
    if matches == 0:
        raise DirectiveError(
            "capability exception is not bound to a saved blocked "
            "IMPLEMENTER_HANDOFF for the current cycle")


def run_gates(commands, seq, router_lock):
    """Run local checks and save their complete output.

    The console shows one result line per command. Complete output goes to a
    log file under ``ai/notes/relay/``.

    Arguments:
      commands = list of shell command strings to run from the repo root.
      seq      = the run sequence stamp for the log filename.
      router_lock = open router lock inherited by each gate process.

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
                              cwd=REPO_ROOT,
                              pass_fds=(router_lock.fileno(),))
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
    body = "\n".join(lines)
    receipt = (
        "<!-- ROUTER GATES v1 commands=" + gate_commands_digest(commands)
        + " result=" + ("pass" if all_green else "fail")
        + " body=" + hashlib.sha256(body.encode("utf-8")).hexdigest()
        + " -->\n")
    log_path = archive(seq, "gates-log", receipt + body)
    return (log_path, all_green)


def verify_execution_checkout(checkout, recovered_candidate=None,
                              resume_active_route=False):
    """Bind this router process and its gate log to the declared checkout.

    The manual router runs commands from its own repository root. A directive
    naming another checkout must therefore use that checkout's copy of this
    script, never test an implementation accidentally against main.

    ``resume_active_route`` accepts one clean descendant commit after an
    existing route stopped before copying its first return. The return value
    is that candidate commit, or ``None`` while the checkout remains at base.
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
    base = checkout["Base"].lower()
    actual_head = git_value(["rev-parse", "HEAD"], "base").lower()
    if recovered_candidate is None:
        if actual_head == base:
            return None
        if not resume_active_route:
            raise DirectiveError(
                "Execution checkout Base mismatch: expected "
                + checkout["Base"] + ", found " + actual_head)
        recovered_candidate = actual_head
    if actual_head != recovered_candidate.lower():
        raise DirectiveError(
            "Execution checkout Base mismatch: expected "
            + recovered_candidate + ", found " + actual_head)
    if git_value(
            ["status", "--porcelain=v1", "--untracked-files=all"],
            "status"):
        raise DirectiveError(
            "saved Implementer candidate checkout is not clean")
    merge_base = git_value(
        ["merge-base", base, recovered_candidate], "candidate ancestry")
    if merge_base.lower() != base:
        raise DirectiveError(
            "saved Implementer candidate does not descend from the "
            "Execution checkout Base")
    return recovered_candidate.lower()


def _git(args_list):
    """Run one status Git command and return its stdout text.

    Arguments:
      args_list = the git arguments, e.g. ["log", "--oneline", "-1", "main"].
    """
    try:
        proc = subprocess.run(["git"] + args_list,
                              capture_output=True,
                              text=True,
                              cwd=REPO_ROOT)
    except OSError as exc:
        raise StatusError("Git status command could not start: "
                          + str(exc)) from exc
    if proc.returncode != 0:
        raise StatusError(
            "Git status command failed: " + proc.stderr.strip())
    return proc.stdout.strip()


def _branch_is_ancestor(branch, target):
    """Return Git ancestry while distinguishing false from query failure."""
    try:
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", branch, target],
            capture_output=True, text=True, cwd=REPO_ROOT)
    except OSError as exc:
        raise StatusError("Git ancestry check could not start: "
                          + str(exc)) from exc
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    raise StatusError("Git ancestry check failed: " + proc.stderr.strip())


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
    claude_branches = []
    for line in branches.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[0].startswith("claude/"):
            continue
        claude_branches.append((int(parts[1]), parts[0]))
    claude_branches.sort(reverse=True)
    open_claude_branches = []
    print("\nArchitect work branches:")
    for _date, name in claude_branches:
        ahead = _git(["rev-list", "--count", "main.." + name])
        if ahead != "0":
            open_claude_branches.append(name)
            tip = _git(["log", "--oneline", "-1", name])
            print("  [OPEN] " + name + ": " + tip)
            print("  -> " + ahead + " saved change(s) are not on main.")
            print("     Do not merge or push this branch by hand.")
            print("     After the Architect saves its exact GO request, run")
            print("     python3 ai/tools/mailbox_daemon.py --once")
            print("     so the daemon can verify and land the audited commit.")
    if not open_claude_branches:
        print("  (none open)")

    # Show advisory Red Team branches and whether main or the Architect work
    # branch already includes them.
    print("\ncodex/* branches:")
    any_open = False
    for line in branches.splitlines():
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not parts[0].startswith("codex/"):
            continue
        name = parts[0]
        merge_targets = ["main"] + open_claude_branches
        is_integrated = False
        for target in merge_targets:
            if _branch_is_ancestor(branch=name, target=target):
                is_integrated = True
                break
        if is_integrated:
            state = "integrated"
        else:
            state = (
                "OPEN: awaiting Architect audit and daemon landing, or still "
                "in work")
            any_open = True
        tip = _git(["log", "--oneline", "-1", name])
        print("  [" + state + "] " + tip)
    if not any_open:
        print("  (none open)")

    try:
        active = active_route_record()
    except BacklogLedgerError as exc:
        raise StatusError("cannot read active manual route: " + str(exc))
    print("\nactive manual route:")
    if active is None:
        print("  (none)")
    else:
        print("  sequence: " + active[1])
        print("  source note: " + active[2])
        print("  base commit: " + active[3])
        print("  abandon only if obsolete: --abandon-route " + active[1])

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
    """Run one router action: status, admin queue, route release, or relay.

    ``--status`` only reads and reports. ``--architect-notes-admin`` queues
    one permanent-note update turn from an already bound Architect process.
    ``--abandon-route`` releases one obsolete manual route. A ``--note``
    relay validates the Architect's source note, then drives the clipboard
    workflow: it copies each generated block, waits for the returned block,
    runs the named check commands, and saves the records under
    ``ai/notes/relay/``. Exactly one action is accepted per invocation.
    """
    parser = argparse.ArgumentParser(
        description="copy approved Architect instructions between manual "
                    "web conversations")
    parser.add_argument("--status",
                        action="store_true",
                        help="show saved AI work, changes not yet on main, "
                             "and recent Architect records, then exit")
    parser.add_argument(
        "--abandon-route", metavar="sequence", default=None,
        help="release one obsolete manual route by the exact sequence shown "
             "by --status; saved evidence remains untouched")
    parser.add_argument(
        "--architect-notes-admin", metavar="summary", default=None,
        help="Architect-only: queue one dedicated permanent-note update "
             "turn with this plain-language summary, then exit")
    parser.add_argument("--note",
                        required=False,
                        help="source note under ai/notes/ containing the "
                             "Architect's checked Implementation directive")
    parser.add_argument("--section",
                        default="",
                        help="optional exact section name; only "
                             "'Implementation directive' is valid")
    parser.add_argument("--mode",
                        choices=["redteam"],
                        default=None,
                        help="confirm that the Architect note assigns Sol "
                             "to Red Team review; this option cannot change "
                             "another plan")
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

    if args.architect_notes_admin is not None:
        conflicting = (args.status or args.abandon_route is not None
                       or args.note or args.section
                       or args.mode is not None or args.skip_redteam
                       or bool(args.gate_cmd) or args.max is not None
                       or args.severity is not None)
        if conflicting:
            print("--architect-notes-admin is a separate Architect-only "
                  "operation and cannot be combined with other routes")
            return 1
        try:
            import mailbox_daemon
        except (ImportError, OSError, SyntaxError) as exc:
            print("cannot load the authoritative mailbox publisher: "
                  + str(exc))
            return 1
        return (0 if mailbox_daemon.send_architect_notes_admin(
            text=args.architect_notes_admin, dry_run=False) else 1)

    if args.abandon_route is not None:
        conflicting = (args.status or args.note or args.section
                       or args.mode is not None or args.skip_redteam
                       or bool(args.gate_cmd) or args.max is not None
                       or args.severity is not None)
        if conflicting:
            print("--abandon-route is a separate recovery action and cannot "
                  "be combined with another route")
            return 1
        try:
            router_lock = acquire_router_lock()
        except RuntimeError as exc:
            print(str(exc))
            return 1
        try:
            record = abandon_active_route(args.abandon_route)
        except (BacklogLedgerError, UnicodeError) as exc:
            print("refused route abandonment: " + str(exc))
            return 1
        finally:
            release_router_lock(router_lock)
        print("Released obsolete manual route " + record[1] + ".")
        print("Its sequence reservation and saved evidence were preserved.")
        return 0

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
        try:
            status_report()
        except StatusError as exc:
            print("cannot read AI work status: " + str(exc))
            return 1
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
        budget = directive["character_change_budget"]
        commands = list(
            args.gate_cmd if args.gate_cmd else DEFAULT_GATE_COMMANDS)
        if budget["limit"] > 0:
            guard = directive["ticket_change_guard"]
            guard_tool = guard["tool"]
            if not os.path.isabs(guard_tool):
                guard_tool = os.path.join(
                    REPO_ROOT, guard_tool.lstrip("./"))
            commands.append(shlex.join([
                sys.executable,
                guard_tool,
                "--repo", guard["repo"],
                "--base", guard["base"],
                "--max", str(guard["max"]),
            ]))
    except (BacklogLedgerError, DirectiveError) as exc:
        print("refused incomplete Architect directive: " + str(exc))
        return 1
    role_plan = directive["role_plan"]
    if role_plan["uses_red_team"]:
        expected_mode = "redteam"
    else:
        expected_mode = None
    if (role_plan["uses_red_team"]
            and role_plan["review_scope"] == "widespread"):
        try:
            refusal = widespread_review_refusal()
        except BacklogLedgerError as exc:
            print("refused widespread Red Team review: " + str(exc))
            return 1
        if refusal is not None:
            print(refusal)
            return 1
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
    try:
        recovered_candidate = recovered_candidate_commit(
            note_path=note_path, note_display=note_display,
            base=directive["execution_checkout"]["Base"],
            commands=commands)
        active_record = active_route_record()
        resume_active_route = (
            recovered_candidate is None and active_record is not None
            and len(active_record) == 6
            and recovered_implementer_return(active_record[1]) is None)
        recovered_candidate = verify_execution_checkout(
            checkout=directive["execution_checkout"],
            recovered_candidate=recovered_candidate,
            resume_active_route=resume_active_route)
        verify_manual_capability_checkpoint(
            directive=directive, source_note=note_display)
    except (BacklogLedgerError, DirectiveError) as exc:
        release_router_lock(router_lock)
        print("refused incomplete Architect directive: " + str(exc))
        return 1
    where = note_display + ', section "Implementation directive"'
    budget_prompt = (
        "Binding character-change budget: limit "
        + str(budget["limit"]) + " characters; planned maximum "
        + str(budget["planned_maximum"]) + " characters. Zero removes the "
        "size cap only; readable complete tested work remains mandatory.\n\n")
    role_prompt = (
        "Architect's validated role plan: " + role_plan["roles"] + ". "
        "Discovery severity: " + discovery_severity + ". Review scope: "
        + role_plan["review_scope"] + ". The runner and human courier may "
        "not change this plan.\n\n")
    if role_plan["review_scope"] == "widespread":
        later_redteam_prompt = (
            "Post-acceptance Red Team plan: enabled with widespread scope. "
            "First audit the Implementer result. If it earns GO, return the "
            "exact decision-only architect-go block and do not merge, commit, "
            "or push. After the daemon records landing L, create a separate "
            "Architect-authored Red Team handoff for the widespread search "
            "saved in " + where + ". Any ticket discovered by that search is "
            "Low. This later advice does not approve or block L.\n\n")
    elif role_plan["review_scope"] == "bounded":
        later_redteam_prompt = (
            "Post-acceptance Red Team plan: enabled with bounded scope. "
            "First audit the Implementer result. If it earns GO, return the "
            "exact decision-only architect-go block and do not merge, commit, "
            "or push. After the daemon records landing L, create a separate "
            "Architect-authored Red Team handoff that names L and reviews "
            "only the behavior it directly affects. This later advice does "
            "not approve or block L.\n\n")
    else:
        later_redteam_prompt = (
            "Post-acceptance Red Team plan: disabled. Complete the "
            "Architect audit without creating a Red Team handoff.\n\n")
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
    total_steps = 3

    # [1] The Implementer receives this unit.
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
        "heading first). In that block, use the exact marker row "
        + IMPLEMENTER_SUBAGENT_EVIDENCE_MARKER + ", put every structured "
        "Subagent return below it in planned order or repeat the exact "
        "Subagents not required heading and Reason, and make "
        "- **Blockers/findings:** the next bold handoff field.\n\n"
        "### ENDS\n")
    implementer_name = "Opus"
    archive_name = "implementer"
    try:
        seq = route_sequence(
            note_path=note_path, note_display=note_display,
            base=directive["execution_checkout"]["Base"],
            commands=commands)
        implementer_block = recovered_implementer_return(seq=seq)
    except (BacklogLedgerError, UnicodeError) as exc:
        release_router_lock(router_lock)
        print("refused router recovery: " + str(exc))
        return 1
    path = os.path.relpath(
        os.path.join(RELAY_DIR, seq + "-implementer.md"), REPO_ROOT)
    recovered_return = implementer_block is not None
    if implementer_block is None:
        copy_to_clipboard(implementer_prompt)
        print("[1/" + str(total_steps) + "] " + implementer_name
              + " instruction copied -- paste it unchanged into that "
              "session.")
        implementer_block = wait_for_block(
            header="### IMPLEMENTER_HANDOFF:",
            last_copied=implementer_prompt)
    else:
        print("[1/" + str(total_steps) + "] recovered the complete saved "
              "Implementer return; no new Implementer work was requested.")
    if recovered_candidate is not None:
        try:
            returned_candidate = implementer_candidate_commit(
                implementer_block)
        except BacklogLedgerError as exc:
            release_router_lock(router_lock)
            print("refused router recovery: " + str(exc))
            return 1
        if returned_candidate != recovered_candidate:
            release_router_lock(router_lock)
            print("refused router recovery: return does not name the saved "
                  "candidate")
            return 1
    try:
        evidence_result = validate_implementer_handoff_subagent_evidence(
            parallel_work_plan=directive["parallel_work_plan"],
            handoff_text=implementer_block)
    except DirectiveError as exc:
        release_router_lock(router_lock)
        print("refused Implementer subagent evidence: " + str(exc))
        return 1
    if evidence_result["completion_ready"]:
        try:
            candidate = implementer_candidate_commit(implementer_block)
            if (recovered_candidate is not None
                    and candidate != recovered_candidate):
                raise BacklogLedgerError(
                    "replacement Implementer return names a different "
                    "candidate")
            if candidate is not None:
                verify_execution_checkout(
                    checkout=directive["execution_checkout"],
                    recovered_candidate=candidate)
                remember_candidate_return(seq, candidate, implementer_block)
        except (BacklogLedgerError, DirectiveError) as exc:
            release_router_lock(router_lock)
            print("refused Implementer candidate: " + str(exc))
            return 1
    if not recovered_return:
        try:
            path = archive(seq, archive_name, implementer_block)
        except BacklogLedgerError as exc:
            release_router_lock(router_lock)
            print("refused Implementer return: " + str(exc))
            return 1
        print("      returned block saved -> " + path)
    else:
        print("      using saved return -> " + path)

    if not evidence_result["completion_ready"]:
        cycle = manual_capability_cycle(
            directive=directive, source_note=note_display)
        digest = save_manual_capability_checkpoint(
            seq=seq, cycle=cycle, source_note=note_display,
            archive_path=path,
            handoff_text=implementer_block,
            capability_failure=evidence_result["capability_failure"])
        architect_prompt = (
            "### IMPLEMENTER CHECKPOINT FOR ARCHITECT\n\n"
            "The Implementer returned a blocked subagent checkpoint. This "
            "is not a candidate and cannot receive GO. Read the saved return "
            "at " + path + ", decide the next directive, and preserve these "
            "mechanical binding rows if a capability-unavailable retry is "
            "justified:\n\n"
            "- Source cycle: `" + cycle + "`\n"
            "- Source handoff SHA-256: `" + digest + "`\n\n"
            "### ENDS\n")
        copy_to_clipboard(architect_prompt)
        finish_route()
        print("[checkpoint] blocked Implementer return copied to the "
              "Architect; no checks or final-GO route were started.")
        release_router_lock(router_lock)
        return 0

    # [2] Run local checks and save their exact output. The Architect still
    # reruns every check required by the directive before deciding.
    try:
        saved_gates = recovered_gate_result(seq=seq, commands=commands)
    except (BacklogLedgerError, UnicodeError) as exc:
        release_router_lock(router_lock)
        print("refused router recovery: " + str(exc))
        return 1
    if saved_gates is None:
        print("[2/" + str(total_steps) + "] running the local checks:")
        try:
            log_path, all_green = run_gates(
                commands=commands, seq=seq, router_lock=router_lock)
        except BacklogLedgerError as exc:
            release_router_lock(router_lock)
            print("refused local check log: " + str(exc))
            return 1
    else:
        log_path, all_green = saved_gates
        print("[2/" + str(total_steps) + "] recovered the complete saved "
              "check log; checks were not rerun.")
    print("      checks " + ("ALL PASS" if all_green else "NOT all green")
          + " -> " + log_path)

    # [3] Return the implementation records to the Architect before any Red
    # Team work. The Architect reruns the required checks and returns an exact
    # GO decision immediately. The daemon, not the Architect, creates landing
    # L. A saved Red Team plan is a later advisory action against L.
    architect_prompt = (
        "### RELAY FOR AUDIT\n\n"
        + role_prompt
        + budget_prompt
        + (severity_prompt if role_plan["uses_red_team"] else "")
        + later_redteam_prompt
        + "Unit spec: " + where + "\n"
        "Implementer return (saved copy): " + path + "\n"
        + "Local check log: " + log_path + "\n\n"
        + "Local check summary: "
        + ("ALL PASS.\n" if all_green else
           "NOT all green. The Architect must inspect the failed command "
           "and issue NO-GO or new instructions; this tool does not "
           "close the ticket.\n")
        + "Review per your role file, including your own reruns of every\n"
        "required check. The saved blocks and check log support the review;\n"
        "they do not replace it. Do not wait for Red Team before this audit\n"
        "or your exact architect-go decision. Before GO, close and seal this\n"
        "ticket in the backlog. Do not merge, commit, update\n"
        "main, or push; save the decision as a to-daemon message and let\n"
        "mailbox_daemon.py create and record landing L.\n\n"
        "### ENDS\n")
    copy_to_clipboard(architect_prompt)
    finish_route()
    print("[" + str(total_steps) + "/" + str(total_steps)
          + "] Architect return prompt copied -- paste it unchanged into "
          "the Architect session for the verdict.")
    release_router_lock(router_lock)
    return 0


if __name__ == "__main__":
    sys.exit(main())
