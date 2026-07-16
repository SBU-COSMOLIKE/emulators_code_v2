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
  - merges/pushes to main require an explicit user grant. The one standing
    grant carried by every Architect dispatch is narrow: an Architect audit
    that records GO lands that audited unit in the same turn, after the
    mandatory foreign-commit STOP walk. Implementer and Red Team turns never
    inherit that grant;
  - the daemon only dispatches messages; it never edits code or notes itself;
  - every dispatch's full CLI output is archived under ai/notes/relay/.

Every live action converges on one persisted primary coordination worktree.
The first valid action creates or safely adopts it; later actions validate
the saved Git identity and re-exec this file from that worktree. Architect and
Implementer share its uncommitted code, notes, and index. A second persisted
worktree belongs to Sol. Ordinary agent turns never start in the user's main
checkout.
AGENT_COMMANDS, the CLI binary paths, is the one machine-specific block.
`claude -p` runs one headless turn against the subscription; the session
needs enough tool permission to work unattended (set via the harness
settings or the flags there).

Usage:
    python ai/tools/mailbox_daemon.py --help           # all options + defaults
    python ai/tools/mailbox_daemon.py --dry-run        # show what would run
    python ai/tools/mailbox_daemon.py --once           # process backlog, exit
    python ai/tools/mailbox_daemon.py --watch          # poll every 20 s
    python ai/tools/mailbox_daemon.py --watch --cycle 2
                                                    # stop safely after 2 cycles
    python ai/tools/mailbox_daemon.py --watch --skip-redteam
                                                    # Architect + Implementer only
    python ai/tools/mailbox_daemon.py --send architect \
        --unit "Coordinate the ticket in ai/notes/<spec>.md."
                                                    # user's only work target
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

# Once main() passes CLI validation, every live action proves that this file
# lives in the saved primary coordination worktree (or re-execs the copy that
# does). Paths below can therefore remain simple derivations while launcher
# checkouts converge on one shared mailbox, notes tree, and index.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ROOT = os.path.dirname(SCRIPT_DIR)
WORKTREE = os.path.dirname(AI_ROOT)


def repo_root_of(worktree):
    """Return the shared repository root that owns a worktree directory.

    A linked checkout's ``.git`` file points to its private administrative
    directory, whose ``commondir`` identifies the main repository. Reading
    those tiny Git-owned files avoids spawning Git during module import (and
    keeps import-only tests pure). A real live action later re-proves the same
    identity with Git itself.

    Arguments:
      worktree = the worktree root, i.e. the directory holding ai/tools/.

    Returns:
      The absolute path of the repository root.
    """
    worktree = os.path.abspath(worktree)
    dot_git = os.path.join(worktree, ".git")
    try:
        dot_git_info = os.lstat(dot_git)
    except OSError:
        dot_git_info = None
    if dot_git_info is not None and stat.S_ISDIR(dot_git_info.st_mode):
        return worktree
    if (dot_git_info is not None and stat.S_ISREG(dot_git_info.st_mode)
            and dot_git_info.st_size <= 4096):
        try:
            with open(dot_git, "r", encoding="utf-8") as stream:
                git_line = stream.read(4097).strip()
            if git_line.startswith("gitdir: "):
                git_directory = git_line[len("gitdir: "):]
                if not os.path.isabs(git_directory):
                    git_directory = os.path.join(worktree, git_directory)
                git_directory = os.path.realpath(git_directory)
                common_file = os.path.join(git_directory, "commondir")
                common_info = os.lstat(common_file)
                if (stat.S_ISREG(common_info.st_mode)
                        and common_info.st_size <= 4096):
                    with open(common_file, "r", encoding="utf-8") as stream:
                        common = stream.read(4097).strip()
                    if not os.path.isabs(common):
                        common = os.path.join(git_directory, common)
                    common = os.path.realpath(common)
                    if os.path.basename(common) == ".git":
                        return os.path.dirname(common)
        except (OSError, UnicodeError):
            pass

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

# A run may limit the text changed by one ticket. Zero is deliberately
# unlimited: the maintenance policy is opt-in, while every readability and
# completeness rule remains active. The command-line choice is copied into
# each child environment and prompt so every role sees one value.
DEFAULT_MAX_CHARACTERS = 0
MAX_CHARACTERS = DEFAULT_MAX_CHARACTERS

# Discovery severity is a per-ticket statement of the user's minimum harm
# level for opening new work. A watch also supplies the default that its
# Architect must save on any discovery ticket it creates. The saved ticket
# value wins when that message is later dispatched by another watch.
DISCOVERY_SEVERITIES = ("high", "medium", "low")
DEFAULT_DISCOVERY_SEVERITY = "medium"
DISCOVERY_SEVERITY = DEFAULT_DISCOVERY_SEVERITY

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
MAX_BACKLOG_LEDGER_BYTES = 16777216

# A watch periodically manufactures one GLOBAL safe-stop opportunity.  Five
# completed child turns is frequent compared with the multi-minute turns this
# daemon runs, while the time bound prevents a sparse or slow queue from going
# indefinitely without an all-idle window.  These are watch-only: --once and
# --dry-run retain their finite, delay-free behavior.
RENDEZVOUS_DISPATCH_INTERVAL = 5
RENDEZVOUS_MINUTE_INTERVAL = 15
SAFE_KILL_COUNTDOWN_SECONDS = 20
WATCH_POLL_SECONDS = 20
MAX_CYCLE_COUNT = 1000000

# One durable coordination checkout belongs to the Claude ROLE pair, not to
# either Claude model. A second durable checkout belongs to Sol. A first live
# CLI action creates (or deliberately adopts) the Claude checkout, creates the
# exact Sol checkout, and saves both Git identities. Later invocations prove
# both records before any dispatch. REPO_ROOT remains the user's checkout.
PRIMARY_WORKTREE_NAME = "mailbox-primary"
PRIMARY_BRANCH = "refs/heads/claude/mailbox-primary"
PRIMARY_STATE_NAME = ".mailbox-primary-worktree.json"
PRIMARY_LOCK_NAME = ".mailbox-primary-worktree.lock"
LEGACY_PRIMARY_STATE_SCHEMA = 1
PRIMARY_STATE_SCHEMA = 2
PRIMARY_TOPOLOGY_MARKER = "dedicated-sol-worktree-v1"
SOL_WORKTREE_NAME = "mailbox-sol"
SOL_BRANCH = "refs/heads/codex/mailbox-sol"
SOL_STATE_NAME = ".mailbox-sol-worktree.json"
SOL_STATE_SCHEMA = 1
MAILBOX_TOPOLOGY_VERSION = 2
MAILBOX_PROTOCOL_VERSION = 2
MAIN_CHECKOUT_TURN_LOCK_NAME = ".main-checkout-turn.lock"
MAX_PRIMARY_STATE_BYTES = 16384
MAX_PRIMARY_DAEMON_BYTES = 2 * 1024 * 1024
MAX_PRIMARY_ARCHIVE_FILE_BYTES = 16 * 1024 * 1024
MAX_PRIMARY_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_PRIMARY_ARCHIVE_ENTRIES = 10000
PRIMARY_ARCHIVE_RUNTIME_LOCKS = frozenset({
    ".dispatch.lock",
    ".sequence.lock",
    ".fix-only.lock",
    ".skip-redteam.lock",
})
CURRENT_ADOPTION_SAFE_REASONS = frozenset({
    "numbered mailbox history exists",
    "relay evidence exists",
    "live watcher or once lock is held",
})


class PrimaryWorktreeError(RuntimeError):
    """A persisted coordination checkout is absent, unsafe, or ambiguous."""


def _raise_walk_error(error):
    """Make ``os.walk`` traversal failures explicit instead of suppressing."""
    raise error


def primary_state_paths(repository_root):
    """Return every deterministic path used by primary-worktree bootstrap."""
    repository = os.path.abspath(repository_root)
    managed_root = os.path.join(repository, ".claude", "worktrees")
    return {
        "managed_root": managed_root,
        "state": os.path.join(managed_root, PRIMARY_STATE_NAME),
        "lock": os.path.join(managed_root, PRIMARY_LOCK_NAME),
        "default_path": os.path.join(managed_root, PRIMARY_WORKTREE_NAME),
        "default_branch": PRIMARY_BRANCH,
    }


def sol_state_paths(repository_root):
    """Return every deterministic path used by Sol-worktree bootstrap."""
    repository = os.path.abspath(repository_root)
    managed_root = os.path.join(repository, ".claude", "worktrees")
    return {
        "managed_root": managed_root,
        "state": os.path.join(managed_root, SOL_STATE_NAME),
        "default_path": os.path.join(managed_root, SOL_WORKTREE_NAME),
        "default_branch": SOL_BRANCH,
    }


def _plain_directory(path, label, create=False):
    """Prove that ``path`` is one ordinary directory, optionally creating it."""
    if not os.path.lexists(path):
        if not create:
            raise PrimaryWorktreeError(label + " does not exist: " + path)
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            # Two clean-clone first runs can both observe the absent managed
            # root before either opens the shared bootstrap lock. The winner's
            # ordinary directory is accepted by the lstat proof below.
            pass
        except OSError as exc:
            raise PrimaryWorktreeError(
                "cannot create " + label + " " + path + ": " + str(exc))
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot inspect " + label + " " + path + ": " + str(exc))
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PrimaryWorktreeError(
            label + " must be a real directory, not a redirect: " + path)
    return (info.st_dev, info.st_ino)


def _require_directory_identity(path, identity, label):
    """Prove a locked directory pathname still names its original inode."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot revalidate " + label + " " + path + ": " + str(exc))
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino) != identity):
        raise PrimaryWorktreeError(
            label + " changed while primary state was being prepared: "
            + path)


def _managed_primary_root(repository_root, create=False):
    """Return the non-symlinked repo-local worktree container."""
    repository = os.path.abspath(repository_root)
    if os.path.realpath(repository) != repository:
        raise PrimaryWorktreeError(
            "repository root must not be reached through a symlink: "
            + repository)
    _plain_directory(path=repository, label="repository root")
    claude_root = os.path.join(repository, ".claude")
    _plain_directory(path=claude_root, label=".claude directory")
    managed_root = os.path.join(claude_root, "worktrees")
    _plain_directory(path=managed_root, label="managed worktree directory",
                     create=create)
    return managed_root


def _run_git(repository_root, arguments, check=True):
    """Run one argv-only Git command and return its completed process."""
    command = ["git", "-C", os.path.abspath(repository_root)] + list(arguments)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise PrimaryWorktreeError("cannot run git: " + str(exc))
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if len(detail) > 1000:
            detail = detail[:1000] + "..."
        if detail:
            detail = ": " + detail
        raise PrimaryWorktreeError(
            "git " + " ".join(arguments) + " failed" + detail)
    return result


def git_common_directory(checkout):
    """Return the canonical Git common directory owning ``checkout``."""
    result = _run_git(repository_root=checkout,
                      arguments=["rev-parse", "--git-common-dir"])
    try:
        value = result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError(
            "git common-directory output is not UTF-8: " + str(exc))
    if not value:
        raise PrimaryWorktreeError("git returned an empty common directory")
    if not os.path.isabs(value):
        value = os.path.join(os.path.abspath(checkout), value)
    return os.path.realpath(value)


def registered_worktrees(repository_root):
    """Parse ``git worktree list --porcelain -z`` without path ambiguity."""
    result = _run_git(
        repository_root=repository_root,
        arguments=["worktree", "list", "--porcelain", "-z"])
    records = []
    record = None
    try:
        fields = result.stdout.split(b"\x00")
        for raw in fields:
            if raw == b"":
                if record is not None:
                    records.append(record)
                    record = None
                continue
            field = raw.decode("utf-8", errors="strict")
            key, separator, value = field.partition(" ")
            if key == "worktree":
                if not separator or not value or record is not None:
                    raise PrimaryWorktreeError(
                        "malformed git worktree registry")
                record = {"path": os.path.abspath(value), "flags": set()}
                continue
            if record is None:
                raise PrimaryWorktreeError(
                    "git worktree registry field precedes worktree path")
            if key in {"HEAD", "branch"}:
                if not separator or key in record:
                    raise PrimaryWorktreeError(
                        "duplicate or malformed worktree " + key + " field")
                record[key] = value
            else:
                record["flags"].add(key)
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError(
            "git worktree registry is not UTF-8: " + str(exc))
    if record is not None:
        records.append(record)
    if not records:
        raise PrimaryWorktreeError("git reports no registered worktrees")
    return records


def _duplicate_key_refusal(pairs):
    """JSON object hook which rejects duplicate state keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise PrimaryWorktreeError(
                "primary-worktree state repeats key " + repr(key))
        result[key] = value
    return result


def load_primary_state(path):
    """Read one bounded, regular, exact-schema primary-worktree record."""
    try:
        initial = os.lstat(path)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot inspect primary-worktree state " + path + ": " + str(exc))
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise PrimaryWorktreeError(
            "primary-worktree state is not a regular file: " + path)
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot open primary-worktree state " + path + ": " + str(exc))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PrimaryWorktreeError(
                "primary-worktree state is not a regular file: " + path)
        if before.st_size > MAX_PRIMARY_STATE_BYTES:
            raise PrimaryWorktreeError(
                "primary-worktree state exceeds "
                + str(MAX_PRIMARY_STATE_BYTES) + " bytes: " + path)
        payload = os.read(descriptor, MAX_PRIMARY_STATE_BYTES + 1)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        if ((initial.st_dev, initial.st_ino) != (before.st_dev, before.st_ino)
                or (before.st_dev, before.st_ino)
                != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino)
                != (current.st_dev, current.st_ino)
                or after.st_size != len(payload)):
            raise PrimaryWorktreeError(
                "primary-worktree state changed while being read: " + path)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot read primary-worktree state " + path + ": " + str(exc))
    finally:
        os.close(descriptor)
    if len(payload) > MAX_PRIMARY_STATE_BYTES:
        raise PrimaryWorktreeError(
            "primary-worktree state exceeds "
            + str(MAX_PRIMARY_STATE_BYTES) + " bytes: " + path)
    try:
        text = payload.decode("utf-8", errors="strict")
        state = json.loads(text, object_pairs_hook=_duplicate_key_refusal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrimaryWorktreeError(
            "primary-worktree state is not exact UTF-8 JSON: " + str(exc))
    if not isinstance(state, dict):
        raise PrimaryWorktreeError("primary-worktree state must be an object")
    base_keys = {"schema", "repository", "name", "path", "branch"}
    if type(state.get("schema")) is not int:
        raise PrimaryWorktreeError(
            "unsupported primary-worktree state schema")
    schema = state["schema"]
    if schema == LEGACY_PRIMARY_STATE_SCHEMA:
        expected = base_keys
    elif schema == PRIMARY_STATE_SCHEMA:
        expected = base_keys | {"topology"}
    else:
        raise PrimaryWorktreeError(
            "unsupported primary-worktree state schema")
    if set(state) != expected:
        raise PrimaryWorktreeError(
            "primary-worktree state keys must be exactly "
            + ", ".join(sorted(expected)))
    for key in ("repository", "name", "path", "branch"):
        value = state[key]
        if (not isinstance(value, str) or not value or "\x00" in value
                or "\n" in value or "\r" in value):
            raise PrimaryWorktreeError(
                "invalid primary-worktree state field " + key)
    if not os.path.isabs(state["repository"]):
        raise PrimaryWorktreeError("state repository must be absolute")
    if not os.path.isabs(state["path"]):
        raise PrimaryWorktreeError("state path must be absolute")
    if (state["name"] != os.path.basename(state["path"])
            or state["name"] in {".", ".."}
            or "/" in state["name"]):
        raise PrimaryWorktreeError("state name must equal the path basename")
    if not state["branch"].startswith("refs/heads/"):
        raise PrimaryWorktreeError("state branch must be an attached head ref")
    if (schema == PRIMARY_STATE_SCHEMA
            and state["topology"] != PRIMARY_TOPOLOGY_MARKER):
        raise PrimaryWorktreeError(
            "primary-worktree topology marker is unsupported")
    return state


def _path_key(path):
    """Return a stable lexical comparison key for a registered worktree."""
    return os.path.normcase(os.path.abspath(path))


def _record_at_path(records, path):
    """Return the unique registry record at ``path``, or ``None``."""
    matches = [record for record in records
               if _path_key(record["path"]) == _path_key(path)]
    if len(matches) > 1:
        raise PrimaryWorktreeError(
            "git reports the worktree path more than once: " + path)
    return matches[0] if matches else None


def _managed_child_path(path, managed_root):
    """Prove ``path`` is one direct, non-symlinked managed child."""
    candidate = os.path.abspath(path)
    if os.path.dirname(candidate) != os.path.abspath(managed_root):
        raise PrimaryWorktreeError(
            "primary worktree must be a direct child of " + managed_root
            + ": " + candidate)
    try:
        info = os.lstat(candidate)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot inspect primary worktree " + candidate + ": " + str(exc))
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PrimaryWorktreeError(
            "primary worktree must be a real directory: " + candidate)
    if os.path.dirname(os.path.realpath(candidate)) != os.path.realpath(
            managed_root):
        raise PrimaryWorktreeError(
            "primary worktree escapes the managed directory: " + candidate)
    return candidate


def _validate_primary_record(record, branch, repository_root):
    """Prove a registry record, checkout, branch, and daemon all agree."""
    managed_root = _managed_primary_root(repository_root=repository_root)
    if "prunable" in record["flags"]:
        raise PrimaryWorktreeError(
            "primary worktree is prunable: " + record["path"])
    if "detached" in record["flags"] or "branch" not in record:
        raise PrimaryWorktreeError(
            "primary worktree must have an attached branch: " + record["path"])
    if record["branch"] != branch:
        raise PrimaryWorktreeError(
            "primary branch mismatch at " + record["path"] + ": expected "
            + branch + ", found " + record["branch"])
    path = _managed_child_path(path=record["path"],
                               managed_root=managed_root)
    top = _run_git(repository_root=path,
                   arguments=["rev-parse", "--show-toplevel"])
    try:
        top_path = top.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError(
            "worktree top-level output is not UTF-8: " + str(exc))
    if os.path.realpath(top_path) != os.path.realpath(path):
        raise PrimaryWorktreeError(
            "registered primary top level does not match its path: " + path)
    repository = git_common_directory(checkout=repository_root)
    if git_common_directory(checkout=path) != repository:
        raise PrimaryWorktreeError(
            "primary worktree belongs to a different repository: " + path)
    symbolic = _run_git(repository_root=path,
                        arguments=["symbolic-ref", "-q", "HEAD"])
    try:
        symbolic_branch = symbolic.stdout.decode(
            "utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError(
            "primary branch output is not UTF-8: " + str(exc))
    if symbolic_branch != branch:
        raise PrimaryWorktreeError(
            "checked-out primary branch does not match state: " + path)
    daemon = os.path.join(path, "ai", "tools", "mailbox_daemon.py")
    try:
        daemon_info = os.lstat(daemon)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "primary daemon is missing at " + daemon + ": " + str(exc))
    if (stat.S_ISLNK(daemon_info.st_mode)
            or not stat.S_ISREG(daemon_info.st_mode)):
        raise PrimaryWorktreeError(
            "primary daemon must be a regular non-symlink file: " + daemon)
    return path


def _atomic_write_primary_state(state, path):
    """Publish primary authority by fsync + same-directory atomic replace."""
    directory = os.path.dirname(path)
    _plain_directory(path=directory, label="managed worktree directory")
    payload = (json.dumps(state, sort_keys=True, indent=2) + "\n").encode(
        "utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".tmp-", dir=directory)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(directory, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_primary_state(state, repository_root, allow_move=False,
                           state_path=None):
    """Validate persisted authority; accept only a Git-authorized move."""
    repository = git_common_directory(checkout=repository_root)
    if state["repository"] != repository:
        raise PrimaryWorktreeError(
            "primary-worktree state names a different repository")
    managed_root = _managed_primary_root(repository_root=repository_root)
    stored_path = os.path.abspath(state["path"])
    if os.path.dirname(stored_path) != managed_root:
        raise PrimaryWorktreeError(
            "saved primary path is outside the managed directory: "
            + stored_path)
    records = registered_worktrees(repository_root=repository_root)
    record = _record_at_path(records=records, path=stored_path)
    resolved = dict(state)
    if record is None:
        branch_matches = [item for item in records
                          if item.get("branch") == state["branch"]]
        if (len(branch_matches) != 1 or os.path.lexists(stored_path)):
            raise PrimaryWorktreeError(
                "saved primary path is no longer registered; state was "
                "preserved for manual recovery: " + stored_path)
        moved = branch_matches[0]
        moved_path = _validate_primary_record(
            record=moved, branch=state["branch"],
            repository_root=repository_root)
        resolved["path"] = moved_path
        resolved["name"] = os.path.basename(moved_path)
        if allow_move:
            if state_path is None:
                state_path = primary_state_paths(repository_root)["state"]
            _atomic_write_primary_state(
                state=resolved, path=state_path)
            print("primary coordination worktree moved by git; saved "
                  + moved_path, flush=True)
        return resolved
    _validate_primary_record(record=record, branch=state["branch"],
                             repository_root=repository_root)
    return resolved


def _transport_evidence_at_notes(notes, reason_prefix=""):
    """Inspect one current or pre-migration notes root without writing it."""
    reasons = []
    mailbox = os.path.join(notes, "mailbox")
    relay = os.path.join(notes, "relay")
    message_name = re.compile(r"\d+[a-z]?-to-[^.]+\.md$")

    for label, root in (("mailbox", mailbox), ("relay", relay)):
        if not os.path.lexists(root):
            continue
        try:
            info = os.lstat(root)
        except OSError as exc:
            reasons.append(reason_prefix + label
                           + " cannot be inspected: " + str(exc))
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            reasons.append(reason_prefix + label
                           + " is redirected or irregular")
            continue
        visited = 0
        found = None
        try:
            for directory, names, files in os.walk(
                    root, followlinks=False, onerror=_raise_walk_error):
                names.sort()
                files.sort()
                for name in list(names):
                    visited += 1
                    if visited > MAX_PRIMARY_ARCHIVE_ENTRIES:
                        found = (reason_prefix + label
                                 + " evidence scan exceeds "
                                 + str(MAX_PRIMARY_ARCHIVE_ENTRIES)
                                 + " entries")
                        break
                    entry = os.path.join(directory, name)
                    entry_info = os.lstat(entry)
                    if stat.S_ISLNK(entry_info.st_mode):
                        found = (reason_prefix + label
                                 + " contains a redirected directory")
                        break
                if found is not None:
                    break
                for name in files:
                    visited += 1
                    if visited > MAX_PRIMARY_ARCHIVE_ENTRIES:
                        found = (reason_prefix + label
                                 + " evidence scan exceeds "
                                 + str(MAX_PRIMARY_ARCHIVE_ENTRIES)
                                 + " entries")
                        break
                    entry = os.path.join(directory, name)
                    entry_info = os.lstat(entry)
                    if (stat.S_ISLNK(entry_info.st_mode)
                            or not stat.S_ISREG(entry_info.st_mode)):
                        found = (reason_prefix + label
                                 + " contains an irregular entry")
                        break
                    if label == "relay":
                        found = reason_prefix + "relay evidence exists"
                        break
                    relative = os.path.relpath(entry, root)
                    if (os.path.dirname(relative) in {"", "."}
                            and name in PRIMARY_ARCHIVE_RUNTIME_LOCKS):
                        continue
                    if message_name.fullmatch(name):
                        found = (reason_prefix
                                 + "numbered mailbox history exists")
                    else:
                        found = (reason_prefix
                                 + "unrecognized mailbox entry exists")
                    break
                if found is not None:
                    break
        except OSError as exc:
            found = (reason_prefix + label
                     + " cannot be scanned: " + str(exc))
        if found is not None:
            reasons.append(found)

    lock_probes = (
        (".dispatch.lock", "live watcher or once lock is held"),
        (".sequence.lock", "live sender or sequence lock is held"),
    )
    for lock_name, held_reason in lock_probes:
        lock_path = os.path.join(mailbox, lock_name)
        if not os.path.lexists(lock_path):
            continue
        flags = os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = None
        try:
            descriptor = os.open(lock_path, flags)
            opened = os.fstat(descriptor)
            current = os.lstat(lock_path)
            if (not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (current.st_dev, current.st_ino)):
                reasons.append(reason_prefix
                               + lock_name
                               + " is redirected or irregular")
            else:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    reasons.append(reason_prefix + held_reason)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as exc:
            reasons.append(reason_prefix
                           + lock_name + " cannot be inspected: " + str(exc))
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return reasons


def coordination_transport_evidence(worktree):
    """Return current and pre-``ai/`` coordination evidence in a worktree."""
    reasons = _transport_evidence_at_notes(
        notes=os.path.join(worktree, "ai", "notes"))
    reasons.extend(_transport_evidence_at_notes(
        notes=os.path.join(worktree, "notes"),
        reason_prefix="legacy pre-ai "))
    return reasons


def _primary_state_for_record(record, repository_root):
    """Build the exact persisted record for one already-validated checkout."""
    return {
        "schema": PRIMARY_STATE_SCHEMA,
        "repository": git_common_directory(checkout=repository_root),
        "name": os.path.basename(record["path"]),
        "path": os.path.abspath(record["path"]),
        "branch": record["branch"],
        "topology": PRIMARY_TOPOLOGY_MARKER,
    }


def _archived_transport_manifest(worktree):
    """Return copyable archived-only transport, or ``None`` if unsafe.

    A pre-primary installation may have completed messages under ``done/``
    plus relay logs in main. Those immutable archives can be bridged into a
    new primary without guessing queue state. Pending, inflight, failed,
    redirected, irregular, or unrecognized mailbox content is never bridged.
    """
    notes = os.path.join(worktree, "ai", "notes")
    mailbox = os.path.join(notes, "mailbox")
    relay = os.path.join(notes, "relay")
    message_name = re.compile(r"(\d+)[a-z]?-to-[^.]+\.md$")
    manifest = []
    visited = 0
    total_bytes = 0
    mailbox_sequences = set()

    for label, root in (("mailbox", mailbox), ("relay", relay)):
        if not os.path.lexists(root):
            continue
        try:
            root_info = os.lstat(root)
        except OSError:
            return None
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(
                root_info.st_mode):
            return None
        try:
            for directory, names, files in os.walk(
                    root, followlinks=False, onerror=_raise_walk_error):
                names.sort()
                files.sort()
                for name in names:
                    visited += 1
                    if visited > MAX_PRIMARY_ARCHIVE_ENTRIES:
                        return None
                    entry_info = os.lstat(os.path.join(directory, name))
                    if stat.S_ISLNK(entry_info.st_mode):
                        return None
                for name in files:
                    visited += 1
                    if visited > MAX_PRIMARY_ARCHIVE_ENTRIES:
                        return None
                    source = os.path.join(directory, name)
                    entry_info = os.lstat(source)
                    if (stat.S_ISLNK(entry_info.st_mode)
                            or not stat.S_ISREG(entry_info.st_mode)):
                        return None
                    relative = os.path.relpath(source, root)
                    if label == "mailbox":
                        parts = relative.split(os.sep)
                        if (len(parts) == 1
                                and name in PRIMARY_ARCHIVE_RUNTIME_LOCKS):
                            continue
                        match = message_name.fullmatch(name)
                        if (len(parts) < 2 or parts[0] != "done"
                                or match is None):
                            return None
                        sequence = int(match.group(1))
                        if sequence in mailbox_sequences:
                            return None
                        mailbox_sequences.add(sequence)
                    if entry_info.st_size > MAX_PRIMARY_ARCHIVE_FILE_BYTES:
                        return None
                    total_bytes += entry_info.st_size
                    if total_bytes > MAX_PRIMARY_ARCHIVE_TOTAL_BYTES:
                        return None
                    manifest.append((
                        source, os.path.join(label, relative),
                        entry_info.st_size, entry_info.st_dev,
                        entry_info.st_ino, entry_info.st_mtime_ns))
        except OSError:
            return None
    return sorted(manifest, key=lambda item: item[1])


def _safe_main_archive_bridge(evidence, repository_root, default_path):
    """Return whether all evidence is one resumable archived-main bridge."""
    allowed_paths = {_path_key(repository_root), _path_key(default_path)}
    allowed_reasons = {
        "numbered mailbox history exists",
        "relay evidence exists",
    }
    main_seen = False
    for path, reasons in evidence:
        if _path_key(path) not in allowed_paths:
            return False
        if not reasons or not set(reasons).issubset(allowed_reasons):
            return False
        if _path_key(path) == _path_key(repository_root):
            main_seen = True
    main_manifest = _archived_transport_manifest(worktree=repository_root)
    if not main_seen or main_manifest is None or not main_manifest:
        return False
    if any(_path_key(path) == _path_key(default_path)
           for path, _reasons in evidence):
        default_manifest = _archived_transport_manifest(
            worktree=default_path)
        if default_manifest is None:
            return False
        main_by_relative = {
            relative: (source, size)
            for source, relative, size, _dev, _ino, _mtime in main_manifest
        }
        for (copied_source, relative, copied_size, _dev, _ino,
             _mtime) in default_manifest:
            legacy = main_by_relative.get(relative)
            if (legacy is None or legacy[1] != copied_size
                    or not _regular_files_equal(
                        first=legacy[0], second=copied_source)):
                return False
    return True


def _open_legacy_transport_lock(path, nonblocking):
    """Open one regular legacy mailbox lock and take exclusive ownership."""
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot open legacy transport lock " + path + ": " + str(exc))
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)):
            raise PrimaryWorktreeError(
                "legacy transport lock is redirected or irregular: " + path)
        operation = fcntl.LOCK_EX
        if nonblocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError:
            raise PrimaryWorktreeError(
                "legacy transport is live; stop its watcher before primary "
                "bootstrap: " + path)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        if ((opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino)
                != (current.st_dev, current.st_ino)):
            raise PrimaryWorktreeError(
                "legacy transport lock changed while waiting: " + path)
        return os.fdopen(descriptor, "r+", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _release_legacy_transport_lock(lock_file):
    """Release one legacy bridge lock."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def _ensure_plain_relative_directory(root, relative):
    """Create a relative directory tree without accepting any redirect."""
    _plain_directory(path=root, label="archive bridge root")
    current = root
    if not relative or relative == ".":
        return current
    for component in relative.split(os.sep):
        if component in {"", ".", ".."}:
            raise PrimaryWorktreeError(
                "invalid archive bridge directory component")
        current = os.path.join(current, component)
        _plain_directory(path=current, label="archive bridge directory",
                         create=True)
    return current


def _regular_files_equal(first, second):
    """Compare two regular non-symlink files without following replacements."""
    descriptors = []
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        for path in (first, second):
            initial = os.lstat(path)
            if (stat.S_ISLNK(initial.st_mode)
                    or not stat.S_ISREG(initial.st_mode)):
                return False
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if ((initial.st_dev, initial.st_ino)
                    != (opened.st_dev, opened.st_ino)
                    or not stat.S_ISREG(opened.st_mode)):
                os.close(descriptor)
                return False
            descriptors.append((descriptor, opened, path))
        if descriptors[0][1].st_size != descriptors[1][1].st_size:
            return False
        while True:
            left = os.read(descriptors[0][0], 1048576)
            right = os.read(descriptors[1][0], 1048576)
            if left != right:
                return False
            if not left:
                break
        for descriptor, opened, path in descriptors:
            after = os.fstat(descriptor)
            current = os.lstat(path)
            if ((opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                    or (after.st_dev, after.st_ino)
                    != (current.st_dev, current.st_ino)):
                return False
        return True
    except OSError:
        return False
    finally:
        for descriptor, _opened, _path in descriptors:
            os.close(descriptor)


def _copy_regular_archive_file(source, destination, expected_size):
    """Idempotently publish one exact archive copy without overwriting."""
    parent = os.path.dirname(destination)
    if (expected_size < 0
            or expected_size > MAX_PRIMARY_ARCHIVE_FILE_BYTES):
        raise PrimaryWorktreeError(
            "legacy archive exceeds the bounded copy size: " + source)
    if os.path.lexists(destination):
        if not _regular_files_equal(first=source, second=destination):
            raise PrimaryWorktreeError(
                "archive bridge destination conflicts with legacy bytes: "
                + destination)
        return
    source_flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    try:
        source_initial = os.lstat(source)
        if (stat.S_ISLNK(source_initial.st_mode)
                or not stat.S_ISREG(source_initial.st_mode)):
            raise PrimaryWorktreeError(
                "legacy archive source is not a regular file: " + source)
        source_descriptor = os.open(source, source_flags)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot open legacy archive " + source + ": " + str(exc))
    temporary_descriptor = -1
    temporary = None
    try:
        source_opened = os.fstat(source_descriptor)
        if ((source_initial.st_dev, source_initial.st_ino)
                != (source_opened.st_dev, source_opened.st_ino)
                or source_opened.st_size != expected_size):
            raise PrimaryWorktreeError(
                "legacy archive changed before copy: " + source)
        temporary_descriptor, temporary = tempfile.mkstemp(
            prefix=".primary-archive-", dir=parent)
        os.fchmod(temporary_descriptor, source_opened.st_mode & 0o777)
        copied = 0
        while copied < expected_size:
            chunk = os.read(
                source_descriptor, min(1048576, expected_size - copied))
            if not chunk:
                raise PrimaryWorktreeError(
                    "legacy archive shortened during copy: " + source)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(temporary_descriptor, view)
                view = view[written:]
        if os.read(source_descriptor, 1):
            raise PrimaryWorktreeError(
                "legacy archive grew during copy: " + source)
        os.fsync(temporary_descriptor)
        source_after = os.fstat(source_descriptor)
        source_current = os.lstat(source)
        if ((source_opened.st_dev, source_opened.st_ino)
                != (source_after.st_dev, source_after.st_ino)
                or (source_after.st_dev, source_after.st_ino)
                != (source_current.st_dev, source_current.st_ino)
                or source_after.st_size != expected_size
                or source_after.st_size != os.fstat(
                    temporary_descriptor).st_size):
            raise PrimaryWorktreeError(
                "legacy archive changed during copy: " + source)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if not _regular_files_equal(first=source, second=destination):
                raise PrimaryWorktreeError(
                    "archive bridge destination raced with different bytes: "
                    + destination)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        os.close(source_descriptor)
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary is not None:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass


def _publish_primary_record(record, repository_root, bridge_main=False,
                            fence_empty_main=False):
    """Publish one selected record behind the applicable legacy locks."""
    state = _primary_state_for_record(
        record=record, repository_root=repository_root)
    state_file = primary_state_paths(repository_root)["state"]
    if not bridge_main and not fence_empty_main:
        _atomic_write_primary_state(state=state, path=state_file)
        return state

    mailbox = os.path.join(repository_root, "ai", "notes", "mailbox")
    parent = os.path.dirname(mailbox)
    _plain_directory(path=parent, label="legacy notes directory")
    mailbox_identity = _plain_directory(
        path=mailbox, label="legacy mailbox", create=True)
    dispatch_lock = _open_legacy_transport_lock(
        path=os.path.join(mailbox, ".dispatch.lock"), nonblocking=True)
    sequence_lock = None
    try:
        sequence_lock = _open_legacy_transport_lock(
            path=os.path.join(mailbox, ".sequence.lock"), nonblocking=True)
        _require_directory_identity(
            path=mailbox, identity=mailbox_identity,
            label="legacy mailbox")
        manifest = _archived_transport_manifest(worktree=repository_root)
        if bridge_main and (manifest is None or not manifest):
            raise PrimaryWorktreeError(
                "legacy main transport is no longer archived-only; state "
                "was not published")
        if fence_empty_main and (manifest is None or manifest):
            raise PrimaryWorktreeError(
                "legacy main transport appeared before primary publication; "
                "state was not published")
        pre_ai_reasons = _transport_evidence_at_notes(
            notes=os.path.join(repository_root, "notes"),
            reason_prefix="legacy pre-ai ")
        if pre_ai_reasons:
            raise PrimaryWorktreeError(
                "pre-migration main transport appeared before primary "
                "publication; state was not published: "
                + ", ".join(pre_ai_reasons))
        copied_manifest = _archived_transport_manifest(
            worktree=record["path"])
        if copied_manifest is None:
            raise PrimaryWorktreeError(
                "primary contains active or irregular transport during "
                "archive bridge; state was not published")
        if bridge_main:
            main_by_relative = {
                relative: (source, size)
                for source, relative, size, _dev, _ino, _mtime in manifest
            }
            for (copied_source, relative, copied_size, _dev, _ino,
                 _mtime) in copied_manifest:
                legacy = main_by_relative.get(relative)
                if (legacy is None or legacy[1] != copied_size
                        or not _regular_files_equal(
                            first=legacy[0], second=copied_source)):
                    raise PrimaryWorktreeError(
                        "primary contains transport that is not an exact "
                        "subset of the main archive; state was not published")
        target_notes = os.path.join(record["path"], "ai", "notes")
        if bridge_main:
            for (source, relative, expected_size, _dev, _ino,
                 _mtime) in manifest:
                relative_parent = os.path.dirname(relative)
                destination_parent = _ensure_plain_relative_directory(
                    root=target_notes, relative=relative_parent)
                destination = os.path.join(destination_parent,
                                           os.path.basename(relative))
                _copy_regular_archive_file(
                    source=source, destination=destination,
                    expected_size=expected_size)
        final_manifest = _archived_transport_manifest(
            worktree=repository_root)
        if final_manifest != manifest:
            raise PrimaryWorktreeError(
                "legacy main archive changed during bridge; state was not "
                "published")
        if bridge_main:
            for (source, relative, expected_size, _dev, _ino,
                 _mtime) in final_manifest:
                destination = os.path.join(target_notes, relative)
                if (os.lstat(destination).st_size != expected_size
                        or not _regular_files_equal(
                            first=source, second=destination)):
                    raise PrimaryWorktreeError(
                        "primary archive copy failed final byte validation; "
                        "state was not published: " + destination)
        final_pre_ai_reasons = _transport_evidence_at_notes(
            notes=os.path.join(repository_root, "notes"),
            reason_prefix="legacy pre-ai ")
        if final_pre_ai_reasons:
            raise PrimaryWorktreeError(
                "pre-migration main transport changed during primary "
                "publication; state was not published: "
                + ", ".join(final_pre_ai_reasons))
        _require_directory_identity(
            path=mailbox, identity=mailbox_identity,
            label="legacy mailbox")
        _validate_primary_record(
            record=record, branch=record["branch"],
            repository_root=repository_root)
        _atomic_write_primary_state(state=state, path=state_file)
    finally:
        if sequence_lock is not None:
            _release_legacy_transport_lock(lock_file=sequence_lock)
        _release_legacy_transport_lock(lock_file=dispatch_lock)
    if bridge_main:
        print("bridged archived main-checkout mailbox and relay history into "
              "the primary without deleting the originals", flush=True)
    return state


def _format_evidence(candidates):
    """Format legacy coordination stores for one actionable refusal."""
    return "; ".join(sorted(path + " (" + ", ".join(reasons) + ")"
                            for path, reasons in candidates))


def _branch_exists(repository_root, branch):
    """Return whether an exact local branch ref already exists."""
    result = _run_git(repository_root=repository_root,
                      arguments=["show-ref", "--verify", "--quiet", branch],
                      check=False)
    if result.returncode not in {0, 1}:
        raise PrimaryWorktreeError("cannot inspect primary branch collision")
    return result.returncode == 0


def provision_or_adopt_primary(repository_root, current_worktree):
    """Select one primary checkout under the already-held bootstrap lock."""
    paths = primary_state_paths(repository_root=repository_root)
    _managed_primary_root(repository_root=repository_root, create=True)
    records = registered_worktrees(repository_root=repository_root)
    evidence = []
    for record in records:
        reasons = coordination_transport_evidence(worktree=record["path"])
        if reasons:
            evidence.append((os.path.abspath(record["path"]), reasons))
    bridge_main = _safe_main_archive_bridge(
        evidence=evidence, repository_root=repository_root,
        default_path=paths["default_path"])

    default_record = _record_at_path(
        records=records, path=paths["default_path"])
    if default_record is not None:
        if default_record.get("branch") != PRIMARY_BRANCH:
            raise PrimaryWorktreeError(
                "default primary path is registered on another branch: "
                + paths["default_path"])
        foreign_evidence = [item for item in evidence
                            if _path_key(item[0])
                            != _path_key(paths["default_path"])]
        if foreign_evidence and not bridge_main:
            raise PrimaryWorktreeError(
                "refusing interrupted-bootstrap recovery because other "
                "coordination stores exist: "
                + _format_evidence(foreign_evidence))
        _validate_primary_record(record=default_record,
                                 branch=PRIMARY_BRANCH,
                                 repository_root=repository_root)
        return _publish_primary_record(
            record=default_record, repository_root=repository_root,
            bridge_main=bridge_main, fence_empty_main=not bridge_main)

    branch_records = [record for record in records
                      if record.get("branch") == PRIMARY_BRANCH]
    if branch_records:
        raise PrimaryWorktreeError(
            "primary branch is already checked out at an unexpected path: "
            + ", ".join(sorted(record["path"] for record in branch_records)))
    if os.path.lexists(paths["default_path"]):
        raise PrimaryWorktreeError(
            "default primary path exists but is not a registered worktree: "
            + paths["default_path"])

    current_record = _record_at_path(records=records, path=current_worktree)
    if evidence:
        if (not bridge_main and len(evidence) == 1
                and current_record is not None
                and _path_key(evidence[0][0]) == _path_key(current_worktree)
                and set(evidence[0][1]).issubset(
                    CURRENT_ADOPTION_SAFE_REASONS)
                and current_record.get("branch") not in {None,
                                                         "refs/heads/main"}
                and os.path.dirname(os.path.abspath(current_worktree))
                == paths["managed_root"]):
            _validate_primary_record(
                record=current_record, branch=current_record["branch"],
                repository_root=repository_root)
            print("adopting current coordination worktree and preserving its "
                  "mailbox: " + os.path.abspath(current_worktree), flush=True)
            return _publish_primary_record(
                record=current_record, repository_root=repository_root)
        if (not bridge_main and len(evidence) == 1
                and current_record is not None
                and _path_key(evidence[0][0])
                == _path_key(current_worktree)):
            raise PrimaryWorktreeError(
                "current coordination worktree cannot be adopted because "
                "its transport is pre-migration or unsafe; preserve and "
                "deliberately migrate or repair it before retrying: "
                + _format_evidence(evidence))
        if not bridge_main:
            raise PrimaryWorktreeError(
                "existing coordination transport must be selected "
                "explicitly; run the command once from the one intended "
                "linked worktree. Candidates: " + _format_evidence(evidence))

    if _branch_exists(repository_root=repository_root, branch=PRIMARY_BRANCH):
        raise PrimaryWorktreeError(
            "primary branch already exists without its registered default "
            "worktree; refusing to reset or reuse it: " + PRIMARY_BRANCH)

    short_branch = PRIMARY_BRANCH[len("refs/heads/"):]
    _run_git(repository_root=repository_root,
             arguments=["worktree", "add", "-b", short_branch,
                        paths["default_path"], "main"])
    refreshed = registered_worktrees(repository_root=repository_root)
    created = _record_at_path(records=refreshed,
                              path=paths["default_path"])
    if created is None:
        raise PrimaryWorktreeError(
            "git created no registered primary worktree; no state was saved")
    _validate_primary_record(record=created, branch=PRIMARY_BRANCH,
                             repository_root=repository_root)
    if not bridge_main:
        appeared = []
        for candidate in refreshed:
            reasons = coordination_transport_evidence(
                worktree=candidate["path"])
            if reasons:
                appeared.append((os.path.abspath(candidate["path"]), reasons))
        if appeared:
            raise PrimaryWorktreeError(
                "coordination transport appeared during primary bootstrap; "
                "the new worktree was preserved but state was not published: "
                + _format_evidence(appeared))
    print("created primary coordination worktree " + paths["default_path"]
          + " on " + PRIMARY_BRANCH, flush=True)
    return _publish_primary_record(
        record=created, repository_root=repository_root,
        bridge_main=bridge_main, fence_empty_main=not bridge_main)


def _upgrade_primary_topology_state(state, repository_root):
    """Accept topology-aware state; never guess that every old process stopped."""
    if state["schema"] == PRIMARY_STATE_SCHEMA:
        return state
    if state["schema"] != LEGACY_PRIMARY_STATE_SCHEMA:
        raise PrimaryWorktreeError(
            "cannot upgrade unsupported primary-worktree state")
    # An old process can validate schema 1, pause before taking the dispatch
    # lock, and resume after an apparent in-place migration. No filesystem
    # lock introduced by this newer code can make that already-admitted old
    # process re-read state. Automatic migration would therefore make a false
    # safety claim. Preserve every byte and require an explicit stopped-old-
    # runtime recovery instead.
    raise PrimaryWorktreeError(
        "legacy schema-1 mailbox state cannot be migrated safely while an "
        "older daemon may already be admitted; stop every old mailbox "
        "process, preserve the saved primary worktree and mailbox, update "
        "that worktree to this daemon version, move the old local state "
        "file aside for recovery, then run the current daemon from the "
        "saved primary path to initialize the new topology")


def _require_primary_daemon_topology_support(primary_path):
    """Refuse re-exec into a stale primary daemon that would ignore Sol."""
    daemon = os.path.join(primary_path, "ai", "tools", "mailbox_daemon.py")
    try:
        initial = os.lstat(daemon)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot inspect saved primary daemon: " + str(exc))
    if (stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode)
            or initial.st_size > MAX_PRIMARY_DAEMON_BYTES):
        raise PrimaryWorktreeError(
            "saved primary daemon is redirected, irregular, or too large: "
            + daemon)
    flags = os.O_RDONLY | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(daemon, flags)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot open saved primary daemon safely: " + str(exc))
    try:
        opened = os.fstat(descriptor)
        chunks = []
        remaining = MAX_PRIMARY_DAEMON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        source = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.lstat(daemon)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot read saved primary daemon safely: " + str(exc))
    finally:
        os.close(descriptor)
    identities = ((initial.st_dev, initial.st_ino),
                  (opened.st_dev, opened.st_ino),
                  (after.st_dev, after.st_ino),
                  (current.st_dev, current.st_ino))
    if (len(set(identities)) != 1 or not stat.S_ISREG(opened.st_mode)
            or after.st_size != len(source)
            or len(source) > MAX_PRIMARY_DAEMON_BYTES):
        raise PrimaryWorktreeError(
            "saved primary daemon changed while compatibility was checked: "
            + daemon)
    declarations = re.findall(
        br"(?m)^MAILBOX_TOPOLOGY_VERSION = ([0-9]+)$", source)
    if declarations != [str(MAILBOX_TOPOLOGY_VERSION).encode("ascii")]:
        raise PrimaryWorktreeError(
            "saved primary daemon predates dedicated Sol worktrees; update "
            "that non-main worktree from main without discarding its local "
            "work, then retry: " + primary_path)
    protocol_declarations = re.findall(
        br"(?m)^MAILBOX_PROTOCOL_VERSION = ([0-9]+)$", source)
    if protocol_declarations != [
            str(MAILBOX_PROTOCOL_VERSION).encode("ascii")]:
        raise PrimaryWorktreeError(
            "saved primary daemon does not enforce the current "
            "Architect-only user entry point; "
            "update that non-main worktree from main without discarding "
            "its local work, then retry: " + primary_path)


def validated_primary_notes(primary_path):
    """Return the canonical non-redirected shared notes directory."""
    primary = os.path.realpath(primary_path)
    _plain_directory(path=primary_path, label="saved primary worktree")
    ai_root = os.path.join(primary_path, "ai")
    notes = os.path.join(ai_root, "notes")
    _plain_directory(path=ai_root, label="saved primary ai directory")
    _plain_directory(path=notes, label="saved primary notes directory")
    expected_ai = os.path.join(primary, "ai")
    expected_notes = os.path.join(expected_ai, "notes")
    if (os.path.realpath(ai_root) != expected_ai
            or os.path.realpath(notes) != expected_notes):
        raise PrimaryWorktreeError(
            "saved primary notes directory is redirected")
    return expected_notes


def validate_authoritative_role_files(primary_path):
    """Return stable proofs for Sol's primary roles and ticket tools."""
    primary = os.path.abspath(primary_path)
    primary_real = os.path.realpath(primary)
    directory_paths = (
        ("saved primary worktree", primary, primary_real),
        ("saved primary .codex directory",
         os.path.join(primary, ".codex"),
         os.path.join(primary_real, ".codex")),
        ("saved primary .claude directory",
         os.path.join(primary, ".claude"),
         os.path.join(primary_real, ".claude")),
        ("saved primary ai directory",
         os.path.join(primary, "ai"),
         os.path.join(primary_real, "ai")),
        ("saved primary tools directory",
         os.path.join(primary, "ai", "tools"),
         os.path.join(primary_real, "ai", "tools")),
    )
    directory_proof = []
    for label, path, expected_real in directory_paths:
        identity = _plain_directory(path=path, label=label)
        if os.path.realpath(path) != expected_real:
            raise PrimaryWorktreeError(label + " is redirected: " + path)
        directory_proof.append((label, path, identity))

    authoritative_files = (
        ("role", os.path.join(
            primary, ".codex", "REDTEAM_ROLE.md")),
        ("role", os.path.join(
            primary, ".claude", "OPUS_ROLE.md")),
        ("ticket tool", os.path.join(
            primary, "ai", "tools", "handoff_contract.py")),
        ("ticket tool", os.path.join(
            primary, "ai", "tools", "ticket_change_guard.py")),
    )
    file_proof = []
    for kind, path in authoritative_files:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise PrimaryWorktreeError(
                "authoritative Sol " + kind + " is missing: " + str(exc))
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise PrimaryWorktreeError(
                "authoritative Sol " + kind + " must be a regular file: "
                + path)
        expected_real = os.path.join(
            primary_real, os.path.relpath(path, primary))
        if os.path.realpath(path) != expected_real:
            raise PrimaryWorktreeError(
                "authoritative Sol " + kind + " is redirected: " + path)
        identity = (info.st_dev, info.st_ino, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        file_proof.append((kind, path, identity))

    proof = {
        "directories": tuple(directory_proof),
        "files": tuple(file_proof),
    }
    recheck_authoritative_role_files(proof=proof)
    return proof


def recheck_authoritative_role_files(proof):
    """Require every authoritative parent and file to keep its identity."""
    if (not isinstance(proof, dict)
            or set(proof) != {"directories", "files"}):
        raise PrimaryWorktreeError(
            "authoritative Sol role-file proof is missing or malformed")
    for label, path, identity in proof["directories"]:
        _require_directory_identity(
            path=path, identity=identity, label=label)
    for kind, path, identity in proof["files"]:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise PrimaryWorktreeError(
                "cannot revalidate authoritative Sol " + kind + ": "
                + str(exc))
        current = (info.st_dev, info.st_ino, info.st_size,
                   info.st_mtime_ns, info.st_ctime_ns)
        if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
                or current != identity):
            raise PrimaryWorktreeError(
                "authoritative Sol " + kind
                + " changed after topology validation: " + path)


def _sol_state_for_record(record, repository_root):
    """Build the exact persisted Sol record for one validated checkout."""
    return {
        "schema": SOL_STATE_SCHEMA,
        "repository": git_common_directory(checkout=repository_root),
        "name": os.path.basename(record["path"]),
        "path": os.path.abspath(record["path"]),
        "branch": record["branch"],
    }


def validate_sol_state(state, repository_root, primary_state,
                       allow_move=False):
    """Validate the saved Sol identity and prove role checkouts are distinct."""
    if state["schema"] != SOL_STATE_SCHEMA:
        raise PrimaryWorktreeError("unsupported Sol-worktree state schema")
    if state["branch"] != SOL_BRANCH:
        raise PrimaryWorktreeError(
            "saved Sol worktree must use " + SOL_BRANCH)
    if primary_state["branch"] == state["branch"]:
        raise PrimaryWorktreeError(
            "Sol and Claude must use different branches")
    resolved = validate_primary_state(
        state=state, repository_root=repository_root, allow_move=False,
        state_path=sol_state_paths(repository_root)["state"])
    sol_path = os.path.realpath(resolved["path"])
    if sol_path == os.path.realpath(repository_root):
        raise PrimaryWorktreeError(
            "Sol worktree must not be the user's repository checkout")
    if sol_path == os.path.realpath(primary_state["path"]):
        raise PrimaryWorktreeError(
            "Sol and Claude must use different worktrees")
    if allow_move and resolved != state:
        _atomic_write_primary_state(
            state=resolved, path=sol_state_paths(repository_root)["state"])
        print("Sol worktree moved by git; saved " + resolved["path"],
              flush=True)
    return resolved


def provision_or_reuse_sol(repository_root, primary_state):
    """Create or validate the one persisted Sol worktree under bootstrap lock."""
    paths = sol_state_paths(repository_root=repository_root)
    _managed_primary_root(repository_root=repository_root, create=True)
    if os.path.lexists(paths["state"]):
        state = load_primary_state(path=paths["state"])
        return validate_sol_state(
            state=state, repository_root=repository_root,
            primary_state=primary_state, allow_move=True)

    records = registered_worktrees(repository_root=repository_root)
    default_record = _record_at_path(
        records=records, path=paths["default_path"])
    if default_record is not None:
        if default_record.get("branch") != SOL_BRANCH:
            raise PrimaryWorktreeError(
                "default Sol path is registered on another branch: "
                + paths["default_path"])
        _validate_primary_record(
            record=default_record, branch=SOL_BRANCH,
            repository_root=repository_root)
        state = _sol_state_for_record(
            record=default_record, repository_root=repository_root)
        state = validate_sol_state(
            state=state, repository_root=repository_root,
            primary_state=primary_state)
        _atomic_write_primary_state(state=state, path=paths["state"])
        print("recovered exact interrupted Sol-worktree bootstrap "
              + state["path"], flush=True)
        return state

    branch_records = [record for record in records
                      if record.get("branch") == SOL_BRANCH]
    if branch_records:
        raise PrimaryWorktreeError(
            "Sol branch is already checked out at an unexpected path: "
            + ", ".join(sorted(record["path"]
                               for record in branch_records)))
    if os.path.lexists(paths["default_path"]):
        raise PrimaryWorktreeError(
            "default Sol path exists but is not a registered worktree: "
            + paths["default_path"])
    if _branch_exists(repository_root=repository_root, branch=SOL_BRANCH):
        raise PrimaryWorktreeError(
            "Sol branch already exists without its registered default "
            "worktree; refusing to reset or reuse it: " + SOL_BRANCH)

    base = _run_git(
        repository_root=repository_root,
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"])
    try:
        base_commit = base.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError(
            "main commit identity is not ASCII: " + str(exc))
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", base_commit):
        raise PrimaryWorktreeError("git returned an invalid main commit")
    short_branch = SOL_BRANCH[len("refs/heads/"):]
    _run_git(
        repository_root=repository_root,
        arguments=["worktree", "add", "-b", short_branch,
                   paths["default_path"], base_commit])
    refreshed = registered_worktrees(repository_root=repository_root)
    created = _record_at_path(
        records=refreshed, path=paths["default_path"])
    if created is None:
        raise PrimaryWorktreeError(
            "git created no registered Sol worktree; no Sol state was saved")
    _validate_primary_record(
        record=created, branch=SOL_BRANCH, repository_root=repository_root)
    state = _sol_state_for_record(
        record=created, repository_root=repository_root)
    state = validate_sol_state(
        state=state, repository_root=repository_root,
        primary_state=primary_state)
    _atomic_write_primary_state(state=state, path=paths["state"])
    print("created Sol worktree " + state["path"] + " on " + SOL_BRANCH,
          flush=True)
    return state


def _open_primary_lock(repository_root):
    """Open and exclusively lock the repo-shared bootstrap inode."""
    paths = primary_state_paths(repository_root=repository_root)
    _managed_primary_root(repository_root=repository_root, create=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(paths["lock"], flags, 0o600)
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot open primary-worktree lock: " + str(exc))
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(paths["lock"])
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)):
            raise PrimaryWorktreeError(
                "primary-worktree lock is redirected or irregular")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        after = os.fstat(descriptor)
        current = os.lstat(paths["lock"])
        if ((opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino)
                != (current.st_dev, current.st_ino)):
            raise PrimaryWorktreeError(
                "primary-worktree lock changed while waiting")
        return os.fdopen(descriptor, "r+", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _release_primary_lock(lock_file):
    """Release the kernel-owned primary bootstrap lock."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def configure_agent_worktrees(primary_path, sol_path):
    """Bind every role to its already-validated dispatch checkout."""
    AGENT_CWD["fable"] = os.path.abspath(primary_path)
    AGENT_CWD["opus"] = os.path.abspath(primary_path)
    AGENT_CWD["sol"] = os.path.abspath(sol_path)


def ensure_primary_execution(live_action, dry_run):
    """Validate both agent worktrees and re-exec from the Claude primary.

    This is deliberately a CLI-boundary operation.  Importing this module for
    focused function tests remains pure; the real ``__main__`` call invokes it
    after every CLI semantic check and before any mailbox path is touched.
    """
    global ACTIVE_TOPOLOGY

    paths = primary_state_paths(repository_root=REPO_ROOT)
    sol_paths = sol_state_paths(repository_root=REPO_ROOT)
    state_exists = os.path.lexists(paths["state"])
    if dry_run and not state_exists:
        print("[dry-run] agent worktrees are not initialized; a live action "
              "would create Claude at " + paths["default_path"] + " on "
              + PRIMARY_BRANCH + " and Sol at " + sol_paths["default_path"]
              + " on " + SOL_BRANCH + ". Previewing this launcher mailbox "
              "read-only.")
        configure_agent_worktrees(
            primary_path=WORKTREE, sol_path=sol_paths["default_path"])
        return None
    if dry_run:
        state = load_primary_state(path=paths["state"])
        state = validate_primary_state(
            state=state, repository_root=REPO_ROOT, allow_move=False)
        try:
            _require_primary_daemon_topology_support(
                primary_path=state["path"])
        except PrimaryWorktreeError as exc:
            print("[dry-run] " + str(exc) + ". No action would be "
                  "dispatched until that checkout is updated.")
            configure_agent_worktrees(
                primary_path=state["path"],
                sol_path=sol_paths["default_path"])
            return state
        if state["schema"] == LEGACY_PRIMARY_STATE_SCHEMA:
            print("[dry-run] this is legacy schema-1 state; a live action "
                  "would refuse until every old daemon is stopped and the "
                  "two-worktree topology is deliberately initialized.")
            configure_agent_worktrees(
                primary_path=state["path"],
                sol_path=sol_paths["default_path"])
            return state
        if os.path.lexists(sol_paths["state"]):
            sol_state = load_primary_state(path=sol_paths["state"])
            sol_state = validate_sol_state(
                state=sol_state, repository_root=REPO_ROOT,
                primary_state=state, allow_move=False)
        else:
            print("[dry-run] a live action would create and save Sol at "
                  + sol_paths["default_path"] + " on " + SOL_BRANCH + ".")
            sol_state = {"path": sol_paths["default_path"]}
        shared_notes = validated_primary_notes(primary_path=state["path"])
        validate_authoritative_role_files(primary_path=state["path"])
    elif live_action:
        lock_file = _open_primary_lock(repository_root=REPO_ROOT)
        try:
            # A waiter must re-read after flock: another first run may have
            # published the single authority while this process was blocked.
            if os.path.lexists(paths["state"]):
                state = load_primary_state(path=paths["state"])
                state = validate_primary_state(
                    state=state, repository_root=REPO_ROOT, allow_move=True)
            else:
                state = provision_or_adopt_primary(
                    repository_root=REPO_ROOT, current_worktree=WORKTREE)
            state = _upgrade_primary_topology_state(
                state=state, repository_root=REPO_ROOT)
            _require_primary_daemon_topology_support(
                primary_path=state["path"])
            sol_state = provision_or_reuse_sol(
                repository_root=REPO_ROOT, primary_state=state)
            shared_notes = validated_primary_notes(
                primary_path=state["path"])
            validate_authoritative_role_files(primary_path=state["path"])
        finally:
            _release_primary_lock(lock_file=lock_file)
    else:
        return None

    configure_agent_worktrees(
        primary_path=state["path"], sol_path=sol_state["path"])
    if live_action:
        ACTIVE_TOPOLOGY = {
            "primary_state": paths["state"],
            "sol_state": sol_paths["state"],
            "primary_path": os.path.abspath(state["path"]),
            "sol_path": os.path.abspath(sol_state["path"]),
            "shared_notes": shared_notes,
        }
    if os.path.realpath(WORKTREE) == os.path.realpath(state["path"]):
        return state
    daemon = os.path.join(state["path"], "ai", "tools",
                          "mailbox_daemon.py")
    print("routing this action to saved primary coordination worktree "
          + state["path"] + " (" + state["branch"] + ")", flush=True)
    try:
        os.chdir(state["path"])
        os.execv(sys.executable,
                 [sys.executable, daemon] + list(sys.argv[1:]))
    except OSError as exc:
        raise PrimaryWorktreeError(
            "cannot re-exec saved primary daemon " + daemon + ": " + str(exc))
    raise PrimaryWorktreeError("saved primary daemon unexpectedly returned")


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
    """Return a grammatically exact waiting-message count."""
    if count == 0:
        return "no messages waiting"
    noun = "message" if count == 1 else "messages"
    return str(count) + " " + noun + " waiting"


def run_safe_kill_countdown(controller):
    """Print 20 safe seconds when no role is starting or running."""
    if not controller.window_ready():
        raise RuntimeError(
            "safe Ctrl-C countdown requested while a role is still active")
    for seconds_more in range(SAFE_KILL_COUNTDOWN_SECONDS - 1, -1, -1):
        waiting = len(pending_messages())
        print("every enabled role is idle; safe to Ctrl-C for "
              + str(seconds_more)
              + "s more; " + waiting_messages_text(count=waiting) + ".",
              flush=True)
        time.sleep(1)
    report_safe_interval_closed()
    controller.reset_after_safe_opportunity()


def report_ordinary_safe_poll(controller, reset_cadence=True):
    """Report a safe Ctrl-C wait when every role job is idle.

    An unlimited watch starts a new work period after this wait. A watch with
    ``--cycle`` counts only a completed five-request or 15-minute work period,
    so this ordinary mailbox check does not complete a cycle.
    """
    if not controller.all_idle():
        return False
    waiting = len(pending_messages())
    print("every enabled role is idle; safe to Ctrl-C for this "
          + str(WATCH_POLL_SECONDS) + "s poll; "
          + waiting_messages_text(count=waiting) + ".", flush=True)
    if reset_cadence:
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


def nonnegative_cycle_count(value):
    """Parse an argparse cycle count, including zero's drain-all meaning."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "cycle count must be a nonnegative integer") from exc
    if parsed < 0 or parsed > MAX_CYCLE_COUNT:
        raise argparse.ArgumentTypeError(
            "cycle count must be a nonnegative integer no larger than "
            + str(MAX_CYCLE_COUNT))
    return parsed


def nonnegative_max_characters(value):
    """Parse a ticket character limit, where zero means unlimited."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise argparse.ArgumentTypeError(
            "max characters must use only decimal digits 0 through 9")
    return int(value)


def strict_cycle_ledger_count():
    """Read the cycle-zero ledger fail-closed from one verified regular file."""
    try:
        before = os.lstat(BACKLOG_LEDGER)
    except OSError as exc:
        return None, "cannot stat backlog ledger: " + str(exc)
    if not stat.S_ISREG(before.st_mode):
        return None, "backlog ledger is not a regular file"
    if before.st_size > MAX_BACKLOG_LEDGER_BYTES:
        return None, "backlog ledger is too large to verify"
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags = flags | os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags = flags | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags = flags | os.O_NOFOLLOW
    try:
        descriptor = os.open(BACKLOG_LEDGER, flags)
    except OSError as exc:
        return None, "cannot open backlog ledger: " + str(exc)
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino):
            return None, "backlog ledger changed identity while opening"
        chunks = []
        size = 0
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            size = size + len(chunk)
            if size > MAX_BACKLOG_LEDGER_BYTES:
                return None, "backlog ledger grew too large while reading"
            chunks.append(chunk)
        try:
            text = b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, "backlog ledger is not valid UTF-8: " + str(exc)
        metadata_before = (
            opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        try:
            current = os.lstat(BACKLOG_LEDGER)
        except OSError as exc:
            return None, "cannot restat backlog ledger: " + str(exc)
        if (not stat.S_ISREG(current.st_mode)
                or current.st_dev != opened.st_dev
                or current.st_ino != opened.st_ino):
            return None, "backlog ledger changed identity while reading"
        after_identity = os.fstat(descriptor)
        metadata_after_identity = (
            after_identity.st_size, after_identity.st_mtime_ns,
            after_identity.st_ctime_ns)
        if metadata_after_identity != metadata_before:
            return None, "backlog ledger changed while verifying identity"
    except OSError as exc:
        return None, "cannot verify backlog ledger: " + str(exc)
    finally:
        os.close(descriptor)
    count = sum(1 for line in text.splitlines()
                if line.startswith("- OPEN"))
    return count, None


def acquire_cycle_completion_barrier(backlog_outcome,
                                     skip_redteam=False):
    """Return a held send barrier only when cycle-zero work is verified done.

    Daemon sends serialize publication through ``.sequence.lock``. Holding
    that lock from the final queue/ledger scan until the watch lock is released
    gives zero mode a real cutoff: a racing send either lands before the scan
    and prevents exit, or lands after the watcher is no longer advertised.
    """
    if backlog_outcome is False:
        return None, None
    lock_path = os.path.join(MAILBOX, ".sequence.lock")
    lock_file = None
    try:
        lock_file = open(lock_path, "a+", encoding="utf-8")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        if lock_file is not None:
            try:
                lock_file.close()
            except OSError:
                pass
        return None, "cannot lock mailbox publication: " + str(exc)
    try:
        ledger, error = strict_cycle_ledger_count()
        waiting_before = enabled_pending_messages(
            skip_redteam=skip_redteam)
        waiting_after = enabled_pending_messages(
            skip_redteam=skip_redteam)
    except OSError as exc:
        ledger = None
        error = "cannot verify pending mailbox messages: " + str(exc)
        waiting_before = []
        waiting_after = []
    if error is None and ledger == 0 and not waiting_before and not waiting_after:
        return lock_file, None
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()
    return None, error


def release_cycle_completion_barrier(lock_file):
    """Release the final cycle-zero send barrier after watch-lock release."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def report_cycle_limit_exit(completed_cycles, cycle_limit,
                            skip_redteam=False):
    """Report a positive cycle limit after every active role job ends."""
    waiting = len(pending_messages())
    ledger = backlog_ledger_count()
    cycle_noun = "cycle" if completed_cycles == 1 else "cycles"
    ledger_noun = "item" if ledger == 1 else "items"
    print("cycle limit reached (" + str(completed_cycles) + "/"
          + str(cycle_limit) + " " + cycle_noun
          + "); every enabled role is idle; watcher stopped; "
          + waiting_messages_text(count=waiting) + "; " + str(ledger)
          + " backlog " + ledger_noun
          + " still begin with '- OPEN'.", flush=True)
    if skip_redteam:
        report_deferred_sol_messages()


def report_cycle_work_complete(completed_cycles, skip_redteam=False):
    """Report cycle zero after its waiting-work checks all pass."""
    noun = "cycle" if completed_cycles == 1 else "cycles"
    if skip_redteam:
        deferred = len(deferred_sol_messages())
        deferred_noun = "message" if deferred == 1 else "messages"
        print("cycle work complete after " + str(completed_cycles) + " "
              + noun + "; no Architect or Implementer message is waiting "
              "or running; ai/notes/backlog.md has no '- OPEN' item; "
              + str(deferred) + " Red Team " + deferred_noun
              + " remain waiting; watcher stopped.", flush=True)
        return
    print("cycle work complete after " + str(completed_cycles) + " " + noun
          + "; no role message is waiting or running; "
          "ai/notes/backlog.md has no '- OPEN' item; watcher stopped.",
          flush=True)


def report_cycle_completion_unverified(error):
    """Explain why zero mode stayed live instead of claiming completion."""
    print("cycle zero cannot verify completion: " + error
          + "; watcher remains active.", flush=True)


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
                         implementer_model=DEFAULT_IMPLEMENTER_MODEL,
                         sol_worktree=None, shared_notes=None):
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
      sol_worktree       = validated worktree used as Sol's cwd and Codex
                           workspace root (default: deterministic first-run
                           path; live dispatch always passes saved state).
      shared_notes       = exact Claude-primary notes directory granted as
                           Sol's only additional writable directory.

    Returns:
      dict mapping "fable"/"opus"/"sol" to the argv list dispatch()
      appends the message to.
    """
    architect_model = validate_model_name(value=architect_model)
    implementer_model = validate_model_name(value=implementer_model)
    if sol_worktree is None:
        sol_worktree = sol_state_paths(REPO_ROOT)["default_path"]
    if shared_notes is None:
        shared_notes = os.path.join(WORKTREE, "ai", "notes")
    sol_worktree = os.path.abspath(sol_worktree)
    shared_notes = os.path.abspath(shared_notes)
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
        # Workspace-write is rooted at Sol's validated worktree. The only
        # additional writable directory is the Claude primary's authoritative
        # notes transport. REPO_ROOT and the rest of the primary are never
        # granted to an ordinary Sol turn.
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
                "--cd", sol_worktree,
                "--add-dir", shared_notes],
    }
    return commands


# main() rebuilds this from the command-line flags; the module-level
# value keeps imports and direct function calls working at the defaults.
AGENT_COMMANDS = build_agent_commands(
    fable_effort=DEFAULT_FABLE_EFFORT,
    opus_effort=DEFAULT_OPUS_EFFORT,
    sol_effort=DEFAULT_SOL_EFFORT,
    sol_context_budget=DEFAULT_SOL_CONTEXT_BUDGET)

# The working directory each dispatched agent starts in. CLI bootstrap proves
# WORKTREE is the persisted primary before live dispatch begins. The Architect
# and Implementer routes (legacy fable/opus keys) share that worktree. Sol's
# deterministic default is replaced by its validated saved path at the CLI
# boundary; the user checkout is never a live-agent fallback.
AGENT_CWD = {
    "fable": WORKTREE,
    "opus": WORKTREE,
    "sol": sol_state_paths(REPO_ROOT)["default_path"],
}

# Set only after a live CLI action validates both persisted records. Imported
# focused tests remain side-effect-free and may supply their own synthetic
# working directories without pretending to have passed live bootstrap.
ACTIVE_TOPOLOGY = None

# A message still carrying template placeholders has no job in it; refuse
# it instead of burning a live headless turn (learned from dispatch 0001).
PLACEHOLDER_MARKERS = ["<spec>", "<X>", "<section>", "<unit>",
                       "your message here"]

# At this total, the watcher reminds the Architect that Sol can receive a
# separate implementation job. Sol remains the Red Team unless that specific
# message contains the exact second-Implementer declaration. The total is the
# waiting mailbox messages plus the "- OPEN" lines in ai/notes/backlog.md.
SECOND_IMPLEMENTER_THRESHOLD = 10
BACKLOG_LEDGER = os.path.join(AI_ROOT, "notes", "backlog.md")
SOL_TICKET_KINDS = ("closure", "discovery")
SOL_DISPATCH_TICKET_KINDS = SOL_TICKET_KINDS + ("transport",)
SOL_TICKET_HEADER = "MAILBOX-TICKET: "
SOL_SEVERITY_HEADER = "MAILBOX-SEVERITY: "
SECOND_IMPLEMENTER_MODE_SENTENCE = (
    "OpenAI Sol — this is a role as second Implementer for this unit.")
SECOND_IMPLEMENTER_RELAY_HEADING_RE = re.compile(
    r"^###[ \t]+ARCHITECT_HANDOFF(?:$|[ \t(:].*)")
FIX_ONLY_ENVIRONMENT = "MAILBOX_FIX_ONLY"
FIX_ONLY_LOCK_NAME = ".fix-only.lock"
SKIP_REDTEAM_ENVIRONMENT = "MAILBOX_SKIP_REDTEAM"
SKIP_REDTEAM_LOCK_NAME = ".skip-redteam.lock"
MAX_CHARACTERS_ENVIRONMENT = "MAILBOX_MAX_CHARACTERS"
DISCOVERY_SEVERITY_ENVIRONMENT = "MAILBOX_DISCOVERY_SEVERITY"

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
LANDING_DEBT_STATE_NAME = ".landing-debt-state.json"
LANDING_DEBT_STATE_SCHEMA = 1
MAX_LANDING_DEBT_STATE_BYTES = 16384
MAX_AUTOMATIC_MESSAGE_SCAN_BYTES = 65536
AUTOMATIC_LANDING_DEBT_MARKER = "MAILBOX-AUTO: landing-debt-v1"

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
    "below. The substance is in entries under this exact notes directory:\n"
    "    " + os.path.join(AI_ROOT, "notes") + "\n"
    "Read the cited entries there first. Do the work per your role file.\n"
    "USER CONTACT RULE: the user gives every ticket request, clarification,\n"
    "policy choice, and scope change to the Architect. Only the Architect\n"
    "turn may interpret or answer that substance. Implementer and Red Team\n"
    "turns act only on an Architect-authored handoff. A human may copy an\n"
    "unchanged handoff between sessions as a courier, but that does not make\n"
    "the human its author and the human must not add instructions.\n"
    "Ordinary rule: end\n"
    "your turn by\n"
    "(1) writing your substance to the appropriate entry INSIDE the exact\n"
    "notes directory above and\n"
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
    "guess a class from prose. A discovery must add this exact second line,\n"
    "using the binding value in MAILBOX_DISCOVERY_SEVERITY:\n"
    "    MAILBOX-SEVERITY: LEVEL\n"
    "Replace LEVEL with exactly high, medium, or low.\n"
    "A public request to the Architect uses that same severity line as its\n"
    "first line. The daemon validates it and exports its value to the\n"
    "Architect turn; it is not a Sol ticket classification.\n"
    "That saved value records the user's minimum severity for a new ticket.\n"
    "The daemon's exact no-work transport ping is\n"
    "the sole reserved MAILBOX-TICKET: transport exception.\n"
    "Narrow exception: if and only if the inbound's binding instruction\n"
    "explicitly says the thread is TERMINAL and no reply is owed, write no\n"
    "outbound merely to satisfy this wrapper. Ambiguity follows the ordinary\n"
    "rule: record the substance and write the outbound.\n"
    "Git landing authority is role-specific. Obey an Architect-only standing\n"
    "grant when one immediately precedes this common wrapper; without such a\n"
    "grant, do not merge or push.\n\n"
    "--- MESSAGE ---\n")

ARCHITECT_LANDING_PREAMBLE = (
    "ARCHITECT STANDING LANDING GRANT (user ruling 2026-07-14):\n"
    "When this audit turn records GO, perform that audited unit's squash\n"
    "landing to main and push it in THIS SAME TURN; do not merely print a\n"
    "landing block. Before EVERY squash, run `git log main..<branch> "
    "--oneline` and walk every foreign commit. Any commit not covered by an\n"
    "Architect GO is a STOP: abort the whole-branch squash, then land only a\n"
    "fully audited subset or wait. Land exactly one audited unit per squash\n"
    "commit and merge main back into the working branch after landing. This\n"
    "grant belongs only to the Architect lane.\n\n")

ARCHITECT_ROLE_PREAMBLE = (
    "ROUTE ROLE: You are the Architect / Auditor. Read and obey\n"
    ".claude/FABLE_ROLE.md before acting. You own design reasoning. Before\n"
    "sending work to an Implementer, write the complete Implementation\n"
    "directive in the cited note and run ai/tools/handoff_contract.py; a\n"
    "goal summary or unresolved design choice is not dispatchable.\n\n")

IMPLEMENTER_ROLE_PREAMBLE = (
    "ROUTE ROLE: You are the Implementer. Read and obey\n"
    ".claude/OPUS_ROLE.md before acting. Run the cited Architect directive\n"
    "check before editing. Follow the ordered plan; if design is missing or\n"
    "contradictory, return a blocker instead of making that decision.\n\n")

REDTEAM_ROLE_PREAMBLE = (
    "ROUTE ROLE: You are the bounded Red Team. Read and obey the exact\n"
    "authoritative role file named below before acting. This inbound did not place the\n"
    "exact second-Implementer declaration in the required first body line;\n"
    "quoting it elsewhere never changes this role. A confirmed finding must include a validated,\n"
    "implementation-ready Repair directive, but it returns to the Architect\n"
    "as candidate input and never executes itself.\n\n")

SECOND_IMPLEMENTER_ROLE_PREAMBLE = (
    "ROUTE ROLE: You are the second Implementer for this unit, not the Red\n"
    "Team. Read the exact authoritative Red Team and Implementer role files\n"
    "named below. Validate the Architect directive and\n"
    "verify its exact linked worktree, non-main branch, and base before any\n"
    "edit. Return a blocker rather than choosing a checkout or design.\n\n")


def agent_preamble(agent, message=None):
    """Return role-specific standing text that precedes the common wrapper."""
    if agent == "fable":
        return ARCHITECT_ROLE_PREAMBLE + ARCHITECT_LANDING_PREAMBLE
    if agent == "opus":
        return IMPLEMENTER_ROLE_PREAMBLE
    if agent == "sol":
        primary = AGENT_CWD["fable"]
        authoritative = (
            "AUTHORITATIVE ROLE FILES (read these absolute paths, not stale "
            "copies in the Sol checkout):\n"
            "    " + os.path.join(primary, ".codex",
                                  "REDTEAM_ROLE.md") + "\n"
            "    " + os.path.join(primary, ".claude",
                                  "OPUS_ROLE.md") + "\n"
            "AUTHORITATIVE TICKET TOOLS (run these absolute primary paths, "
            "not relative copies in the Sol checkout):\n"
            "    " + os.path.join(
                primary, "ai", "tools", "handoff_contract.py") + "\n"
            "    " + os.path.join(
                primary, "ai", "tools", "ticket_change_guard.py") + "\n"
            "For a character check, pass `--repo` followed by the exact "
            "candidate worktree from the directive. Never measure the Sol "
            "checkout merely because it is this turn's current folder.\n")
        if message is not None and sol_second_implementer_assignment(
                message=message):
            return (SECOND_IMPLEMENTER_ROLE_PREAMBLE + authoritative
                    + "Execution checkout must name this saved Sol "
                    "worktree exactly:\n    " + AGENT_CWD["sol"] + "\n\n")
        return REDTEAM_ROLE_PREAMBLE + authoritative + "\n"
    raise ValueError("unknown mailbox agent: " + repr(agent))


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


def enabled_pending_messages(skip_redteam=False):
    """Return root messages eligible for this watch topology.

    The ordinary three-role topology returns every dispatchable message.
    A two-role watch excludes only exact ``to-sol`` roots; those files stay
    in place for a later Sol-enabled watch.
    """
    backlog = pending_messages()
    if not skip_redteam:
        return backlog
    return [path for path in backlog
            if PENDING_MESSAGE_RE.match(os.path.basename(path)).group(1)
            != "sol"]


def deferred_sol_messages():
    """Return exact pending Sol roots held by a two-role watch."""
    return [path for path in pending_messages()
            if PENDING_MESSAGE_RE.match(os.path.basename(path)).group(1)
            == "sol"]


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


def sol_ticket_body_after_kind(message):
    """Return the bytes after a valid Sol classification line."""
    match = re.match(
        r"\A" + re.escape(SOL_TICKET_HEADER)
        + r"(?:" + "|".join(map(re.escape, SOL_DISPATCH_TICKET_KINDS))
        + r")(?:\r?\n|\Z)",
        message)
    if match is None:
        return message
    return message[match.end():]


def sol_discovery_severity_problem(message):
    """Return a saved discovery-severity envelope error, or ``None``.

    Old discovery messages had only the ticket line. They keep the documented
    medium default. Once a second line uses the reserved severity prefix, it
    must contain one exact supported value and may appear only once.
    """
    ticket_kind = sol_ticket_kind(message=message)
    remainder = sol_ticket_body_after_kind(message=message)
    severity_like_line = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*severity[ \t]*:")
    if ticket_kind != "discovery":
        if re.search(severity_like_line, remainder) is not None:
            return ("MAILBOX-SEVERITY is reserved for discovery tickets "
                    "and must not appear on another ticket kind")
        return None
    if not remainder.startswith(SOL_SEVERITY_HEADER):
        if re.search(severity_like_line, remainder) is not None:
            return ("MAILBOX-SEVERITY must use its exact spelling and be "
                    "the second physical line of a discovery ticket")
        return None
    match = re.match(
        r"\A" + re.escape(SOL_SEVERITY_HEADER)
        + r"(" + "|".join(map(re.escape, DISCOVERY_SEVERITIES))
        + r")(?:\r?\n|\Z)",
        remainder)
    if match is None:
        return ("invalid discovery severity line; use exactly "
                "'MAILBOX-SEVERITY: high', 'MAILBOX-SEVERITY: medium', "
                "or 'MAILBOX-SEVERITY: low'")
    if re.search(severity_like_line, remainder[match.end():]) is not None:
        return "duplicate MAILBOX-SEVERITY line"
    return None


def sol_discovery_severity(message):
    """Return a discovery ticket's saved severity, including legacy default."""
    if sol_ticket_kind(message=message) != "discovery":
        return None
    if sol_discovery_severity_problem(message=message) is not None:
        return None
    remainder = sol_ticket_body_after_kind(message=message)
    match = re.match(
        r"\A" + re.escape(SOL_SEVERITY_HEADER)
        + r"(" + "|".join(map(re.escape, DISCOVERY_SEVERITIES))
        + r")(?:\r?\n|\Z)",
        remainder)
    if match is None:
        return DEFAULT_DISCOVERY_SEVERITY
    return match.group(1)


def sol_ticket_body(message):
    """Return the human body after valid Sol envelope lines."""
    remainder = sol_ticket_body_after_kind(message=message)
    if sol_ticket_kind(message=message) != "discovery":
        return remainder
    if sol_discovery_severity_problem(message=message) is not None:
        return remainder
    match = re.match(
        r"\A" + re.escape(SOL_SEVERITY_HEADER)
        + r"(?:" + "|".join(map(re.escape, DISCOVERY_SEVERITIES))
        + r")(?:\r?\n|\Z)",
        remainder)
    if match is None:
        return remainder
    return remainder[match.end():]


def sol_second_implementer_assignment(message):
    """Return whether Sol's first assignment line switches its role.

    The mandatory ``MAILBOX-TICKET`` line and one optional Architect relay
    heading are transport wrappers. The next nonblank line must be the exact
    declaration. A later quotation is ordinary Red Team prose.
    """
    if sol_ticket_kind(message=message) not in SOL_TICKET_KINDS:
        return False
    lines = [line for line in sol_ticket_body(message=message).splitlines()
             if line.strip()]
    if (lines and SECOND_IMPLEMENTER_RELAY_HEADING_RE.fullmatch(lines[0])
            is not None):
        lines = lines[1:]
    return bool(lines and lines[0] == SECOND_IMPLEMENTER_MODE_SENTENCE)


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


def sol_ticket_payload(ticket_kind, text, discovery_severity=None):
    """Build the byte-stable persisted envelope for a Sol message."""
    if ticket_kind == "discovery":
        if discovery_severity is None:
            discovery_severity = DEFAULT_DISCOVERY_SEVERITY
        if discovery_severity not in DISCOVERY_SEVERITIES:
            raise ValueError("invalid discovery severity: "
                             + repr(discovery_severity))
        payload = (SOL_TICKET_HEADER + ticket_kind + "\n"
                   + SOL_SEVERITY_HEADER + discovery_severity + "\n\n"
                   + text)
    else:
        if discovery_severity is not None:
            raise ValueError(
                "discovery severity is valid only for discovery tickets")
        payload = SOL_TICKET_HEADER + ticket_kind + "\n\n" + text
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def architect_user_request_payload(text, discovery_severity=None):
    """Build the persisted public envelope addressed only to Architect."""
    if discovery_severity is None:
        discovery_severity = DEFAULT_DISCOVERY_SEVERITY
    if discovery_severity not in DISCOVERY_SEVERITIES:
        raise ValueError("invalid discovery severity: "
                         + repr(discovery_severity))
    payload = (SOL_SEVERITY_HEADER + discovery_severity + "\n\n" + text)
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def architect_user_request_severity(message):
    """Return a valid public Architect envelope severity, or ``None``."""
    if not message.startswith(SOL_SEVERITY_HEADER):
        return None
    first_line = message.splitlines()[0]
    severity = first_line[len(SOL_SEVERITY_HEADER):]
    if severity not in DISCOVERY_SEVERITIES:
        return None
    if not message.startswith(first_line + "\n\n"):
        return None
    return severity


def architect_user_request_body(message):
    """Return the exact user text after a valid Architect envelope."""
    severity = architect_user_request_severity(message=message)
    if severity is None:
        return message
    return message[len(SOL_SEVERITY_HEADER + severity + "\n\n"):]


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


def skip_redteam_environment_active():
    """Return whether this send inherited a two-role watch contract."""
    value = os.environ.get(SKIP_REDTEAM_ENVIRONMENT)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def resolve_discovery_severity(cli_value=None):
    """Bind an explicit severity to the inherited run default."""
    inherited = os.environ.get(DISCOVERY_SEVERITY_ENVIRONMENT)
    if inherited is not None and inherited not in DISCOVERY_SEVERITIES:
        raise ValueError(
            DISCOVERY_SEVERITY_ENVIRONMENT
            + " must be exactly high, medium, or low")
    if cli_value is None:
        return (DEFAULT_DISCOVERY_SEVERITY
                if inherited is None else inherited)
    if cli_value not in DISCOVERY_SEVERITIES:
        raise ValueError("discovery severity must be high, medium, or low")
    if inherited is not None and cli_value != inherited:
        raise ValueError(
            "--severity " + cli_value + " does not match inherited "
            + DISCOVERY_SEVERITY_ENVIRONMENT + " " + inherited)
    return cli_value


def sol_ticket_refusal(ticket_kind, total, fix_only,
                       transport_valid=False, discovery_severity=None):
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
    if ticket_kind == "discovery":
        if discovery_severity is None:
            discovery_severity = DEFAULT_DISCOVERY_SEVERITY
        if discovery_severity not in DISCOVERY_SEVERITIES:
            return ("a discovery ticket needs one severity: high, medium, "
                    "or low")
    elif discovery_severity is not None:
        return "--severity is valid only for discovery tickets"
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


def inflight_lane_blockers(skip_redteam=False):
    """Return unresolved inflight agent messages grouped by cwd lane.

    Only exact dispatchable message names participate. A hand-made file or an
    archived ``-to-user`` note under inflight cannot block an agent lane, but
    an unresolved Fable message blocks Opus too because those recipients share
    one working directory. Two-role mode may ignore a historical Sol blocker
    only when Sol's cwd is separate from both enabled Claude cwd lanes. In an
    ordinary checkout all routes can share the repository root, so that same
    Sol blocker must continue to hold the shared tree closed.
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
        agent = match.group(1)
        cwd = AGENT_CWD[agent]
        enabled_claude_cwds = {
            AGENT_CWD["fable"], AGENT_CWD["opus"]}
        if (skip_redteam and agent == "sol"
                and cwd not in enabled_claude_cwds):
            continue
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
                    fix_only=False, skip_redteam=False,
                    discovery_severity=None, saved_discovery=False,
                    saved_architect_request=False):
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
    if skip_redteam:
        lines.append(
            "two-role watch: the Red Team and entire Sol route are disabled; "
            "create no to-sol messages; route Implementer evidence to the "
            "Architect and Architect repair handoffs to the Implementer.")
    lines.append("--- END DISPATCH CURRENCY ---")
    lines.append("")
    lines.append("--- DISCOVERY SEVERITY (binding) ---")
    if discovery_severity is None:
        discovery_severity = DISCOVERY_SEVERITY
    if saved_discovery:
        lines.append(
            "user's saved minimum severity for this discovery: "
            + discovery_severity)
    elif saved_architect_request:
        lines.append(
            "user's saved minimum severity for any discovery requested "
            "by this ticket: " + discovery_severity)
    else:
        lines.append(
            "minimum severity to save on any new discovery ticket: "
            + discovery_severity)
    lines.append(
        "high: only a bug that severely impacts core functionality, causes "
        "data loss, halts system operations, or makes the science wrong.")
    lines.append(
        "medium: high bugs plus a less severe bug that can affect normal "
        "operation and has a probable path; a merely theoretical or "
        "improbable edge case does not qualify.")
    lines.append(
        "low: any concrete discovered bug may qualify, including an "
        "improbable edge case; an unsupported guess is not a discovery.")
    if fix_only:
        lines.append(
            "fix-only is stronger than this setting: create no discovery "
            "ticket or new backlog line.")
    if skip_redteam:
        lines.append(
            "the Sol route is disabled: create no discovery ticket while "
            "this two-role watch is active.")
    lines.append(
        "The Red Team records User severity setting, Red Team severity, "
        "Likelihood (probable or improbable), Likelihood evidence, and "
        "Meets user setting (yes or no).")
    lines.append(
        "The Architect accepts, upgrades, or downgrades that rating with an "
        "evidence-based reason, then makes the final GO or NO-GO ticket "
        "decision. The Red Team never opens the ticket itself.")
    lines.append("--- END DISCOVERY SEVERITY ---")
    lines.append("")
    lines.append("--- TICKET CHARACTER BUDGET (binding) ---")
    primary = AGENT_CWD["fable"]
    contract_tool = os.path.join(
        primary, "ai", "tools", "handoff_contract.py")
    change_tool = os.path.join(
        primary, "ai", "tools", "ticket_change_guard.py")
    if MAX_CHARACTERS == 0:
        lines.append(
            "ticket limit: none (--max 0); readability, complete behavior, "
            "tests, explanations, and failure handling remain required; "
            "obfuscated work is NO-GO.")
        lines.append(
            "The Architect must record the unlimited budget and validate the "
            "structured directive by running `python3 " + contract_tool
            + " architect NOTE_ABSOLUTE_PATH --max 0` before GO.")
    else:
        value = str(MAX_CHARACTERS)
        lines.append(
            "ticket limit: at most " + value + " characters added plus "
            "deleted from the directive Base.")
        lines.append(
            "Before final GO or ticket closure, the Architect must run "
            "`python3 " + change_tool
            + " --repo EXECUTION_WORKTREE --base BASE --max " + value
            + "`. The program path belongs to the primary AI worktree; "
            "--repo selects the exact proposed commit.")
        lines.append(
            "The Architect must record the structured budget evidence and "
            "validate the structured directive by running `python3 "
            + contract_tool + " architect NOTE_ABSOLUTE_PATH --max " + value
            + "` before GO.")
        lines.append(
            "Over-limit, unmeasurable, or obfuscated work is NO-GO; never "
            "compress readable code, omit tests or explanations, or leave "
            "requested behavior incomplete to fit.")
    lines.append("--- END TICKET CHARACTER BUDGET ---")
    return "\n".join(lines) + "\n\n"


def report_ticket_character_limit():
    """Print the effective per-ticket text limit at live startup."""
    if MAX_CHARACTERS == 0:
        print("ticket character limit: none (--max 0)")
        return
    print("ticket character limit: " + str(MAX_CHARACTERS)
          + " added plus deleted characters per ticket")


def report_discovery_severity(fix_only=False, skip_redteam=False):
    """Print the default saved on new discovery tickets for this run."""
    line = "discovery severity default: " + DISCOVERY_SEVERITY
    if fix_only:
        line = line + " (inactive while fix-only forbids discovery)"
    elif skip_redteam:
        line = line + " (inactive while the Sol route is disabled)"
    else:
        line = line + " (saved on each new discovery ticket)"
    print(line)


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


def skip_redteam_watch_is_active(mailbox=None):
    """Return whether this mailbox has a live two-role watch marker."""
    if mailbox is None:
        mailbox = MAILBOX
    held, _ = held_lock_probe(
        mailbox=mailbox, lock_name=SKIP_REDTEAM_LOCK_NAME)
    return held


def skip_redteam_policy_active():
    """Return whether this process or its mailbox is in two-role mode."""
    return (skip_redteam_environment_active()
            or skip_redteam_watch_is_active())


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


def acquire_skip_redteam_lock_while_sequence_locked():
    """Create the two-role mode marker after publishers are serialized."""
    if not mailbox_path_is_unredirected(mailbox=MAILBOX):
        print("cannot disable the red-team route on a redirected mailbox "
              "path")
        return None
    lock_path = os.path.join(MAILBOX, SKIP_REDTEAM_LOCK_NAME)
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
    flags = flags | getattr(os, "O_CLOEXEC", 0)
    flags = flags | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        print("cannot disable the red-team route: " + str(exc))
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
            print("cannot disable the red-team route: mode lock is not an "
                  "unchanged regular file")
            lock_file.close()
            return None
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        print("cannot disable the red-team route: its mode lock is already "
              "held or unreadable (" + str(exc) + ")")
        lock_file.close()
        return None
    if not path_still_names_opened_inode(opened=opened):
        print("cannot disable the red-team route: mode lock path changed "
              "while its lock was acquired")
        release_skip_redteam_lock(lock_file=lock_file)
        return None
    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write("two-role watch pid " + str(os.getpid()))
        lock_file.flush()
        os.fsync(lock_file.fileno())
    except OSError as exc:
        print("cannot disable the red-team route: could not publish its "
              "owner (" + str(exc) + ")")
        release_skip_redteam_lock(lock_file=lock_file)
        return None
    if not path_still_names_opened_inode(opened=opened):
        print("cannot disable the red-team route: mode lock path changed "
              "while its owner was published")
        release_skip_redteam_lock(lock_file=lock_file)
        return None
    return lock_file


def acquire_skip_redteam_lock():
    """Atomically disable Sol dispatch relative to daemon message sends."""
    # Refuse a redirected mailbox before creating even its sequence-lock
    # file. The inner check stays binding because the path can still change
    # between this preflight and publication of the mode marker.
    if not mailbox_path_is_unredirected(mailbox=MAILBOX):
        print("cannot disable the red-team route on a redirected mailbox "
              "path")
        return None
    os.makedirs(MAILBOX, exist_ok=True)
    sequence_path = os.path.join(MAILBOX, ".sequence.lock")
    try:
        with open(sequence_path, "a+", encoding="utf-8") as sequence_file:
            fcntl.flock(sequence_file.fileno(), fcntl.LOCK_EX)
            try:
                return acquire_skip_redteam_lock_while_sequence_locked()
            finally:
                fcntl.flock(sequence_file.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        print("cannot disable the red-team route: sequence lock failed ("
              + str(exc) + ")")
        return None


def release_skip_redteam_lock(lock_file):
    """Release a lock returned by ``acquire_skip_redteam_lock``."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def main_checkout_turn_lock_path():
    """Return the one ignored lock shared by every repository worktree."""
    repository = os.path.abspath(REPO_ROOT)
    _plain_directory(path=repository, label="repository root")
    claude_root = os.path.join(repository, ".claude")
    _plain_directory(path=claude_root, label=".claude directory", create=True)
    managed_root = os.path.join(claude_root, "worktrees")
    _plain_directory(
        path=managed_root, label="managed worktree directory", create=True)
    return os.path.join(managed_root, MAIN_CHECKOUT_TURN_LOCK_NAME)


def acquire_main_checkout_turn_lock():
    """Serialize turns carrying Architect-only main-landing authority."""
    try:
        lock_path = main_checkout_turn_lock_path()
    except (OSError, PrimaryWorktreeError) as exc:
        print("cannot serialize the main checkout: " + str(exc))
        return None
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        print("cannot serialize the main checkout: " + str(exc))
        return None
    lock_file = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        opened = os.fstat(lock_file.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("main-checkout turn lock is not a regular file")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        current = os.lstat(lock_path)
        if (not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)):
            raise OSError("main-checkout turn lock path changed")
    except OSError as exc:
        print("cannot serialize the main checkout: " + str(exc))
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()
        return None
    return lock_file


def release_main_checkout_turn_lock(lock_file):
    """Release an Architect main-checkout turn lock."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def validate_live_sol_dispatch_topology():
    """Re-prove Sol's saved checkout and narrow notes grant before claim."""
    if ACTIVE_TOPOLOGY is None:
        raise PrimaryWorktreeError(
            "live Sol dispatch has no validated agent-worktree topology")
    lock_file = _open_primary_lock(repository_root=REPO_ROOT)
    try:
        primary = load_primary_state(path=ACTIVE_TOPOLOGY["primary_state"])
        if primary["schema"] != PRIMARY_STATE_SCHEMA:
            raise PrimaryWorktreeError(
                "live dispatch requires topology-aware primary state")
        primary = validate_primary_state(
            state=primary, repository_root=REPO_ROOT, allow_move=False)
        sol = load_primary_state(path=ACTIVE_TOPOLOGY["sol_state"])
        sol = validate_sol_state(
            state=sol, repository_root=REPO_ROOT, primary_state=primary,
            allow_move=False)
        notes = validated_primary_notes(primary_path=primary["path"])
        authoritative_files = validate_authoritative_role_files(
            primary_path=primary["path"])
        expected = ACTIVE_TOPOLOGY
        if (os.path.abspath(primary["path"]) != expected["primary_path"]
                or os.path.abspath(sol["path"]) != expected["sol_path"]
                or notes != expected["shared_notes"]
                or AGENT_CWD["fable"] != expected["primary_path"]
                or AGENT_CWD["opus"] != expected["primary_path"]
                or AGENT_CWD["sol"] != expected["sol_path"]
                or os.path.realpath(os.path.join(AI_ROOT, "notes"))
                != expected["shared_notes"]):
            raise PrimaryWorktreeError(
                "saved agent topology changed after this process started")
        command = AGENT_COMMANDS["sol"]
        if command.count("--cd") != 1 or command.count("--add-dir") != 1:
            raise PrimaryWorktreeError(
                "Sol command must carry one exact cwd and notes grant")
        cd_index = command.index("--cd")
        add_index = command.index("--add-dir")
        if cd_index + 1 >= len(command) or add_index + 1 >= len(command):
            raise PrimaryWorktreeError(
                "Sol command is missing its cwd or notes value")
        if (os.path.abspath(command[cd_index + 1])
                != expected["sol_path"]
                or os.path.realpath(command[add_index + 1])
                != expected["shared_notes"]):
            raise PrimaryWorktreeError(
                "Sol command no longer matches saved worktree state")
        sol_identity = _plain_directory(
            path=expected["sol_path"], label="saved Sol worktree")
        notes_identity = _plain_directory(
            path=expected["shared_notes"], label="shared notes directory")
        return {
            "sol_path": expected["sol_path"],
            "sol_identity": sol_identity,
            "notes_path": expected["shared_notes"],
            "notes_identity": notes_identity,
            "authoritative_files": authoritative_files,
        }
    finally:
        _release_primary_lock(lock_file=lock_file)


def recheck_sol_dispatch_directories(proof):
    """Prove launch pathnames still name the pre-claim directories."""
    if proof is None:
        raise PrimaryWorktreeError(
            "live Sol dispatch is missing its topology proof")
    _require_directory_identity(
        path=proof["sol_path"], identity=proof["sol_identity"],
        label="saved Sol worktree")
    _require_directory_identity(
        path=proof["notes_path"], identity=proof["notes_identity"],
        label="shared notes directory")
    recheck_authoritative_role_files(proof=proof["authoritative_files"])


def revalidate_sol_dispatch_topology(proof):
    """Re-prove all Git and command bindings without accepting a new inode."""
    recheck_sol_dispatch_directories(proof=proof)
    current = validate_live_sol_dispatch_topology()
    if current != proof:
        raise PrimaryWorktreeError(
            "saved Sol worktree topology changed after message claim")
    recheck_sol_dispatch_directories(proof=current)
    return current


def dispatch(path, dry_run, fix_only=False, skip_redteam=False):
    """Serialize Architect landing authority, then run one dispatch."""
    match = PENDING_MESSAGE_RE.match(os.path.basename(path))
    if match is None:
        raise ValueError("not a pending agent message: " + path)
    agent = match.group(1)
    if dry_run or agent != "fable":
        return dispatch_under_main_checkout_lock(
            path=path, dry_run=dry_run, fix_only=fix_only,
            skip_redteam=skip_redteam)
    lock_file = acquire_main_checkout_turn_lock()
    if lock_file is None:
        print("refused " + os.path.basename(path) + ": the Architect "
              "main-landing turn lock could not be proved; root message "
              "left untouched.")
        return False
    try:
        return dispatch_under_main_checkout_lock(
            path=path, dry_run=dry_run, fix_only=fix_only,
            skip_redteam=skip_redteam)
    finally:
        release_main_checkout_turn_lock(lock_file=lock_file)


def dispatch_under_main_checkout_lock(
        path, dry_run, fix_only=False, skip_redteam=False):
    """Send one message file to its addressee's headless CLI.

    Arguments:
      path    = the mailbox message file.
      dry_run  = True to print the would-be command without running it.
      fix_only = True when the owning watch may launch declared closures only.
      skip_redteam = True when the owning watch excludes every Sol turn.

    Returns:
      True when the dispatch ran (or would run) cleanly.
    """
    name = os.path.basename(path)
    agent_match = PENDING_MESSAGE_RE.match(name)
    if agent_match is None:
        raise ValueError("not a pending agent message: " + path)
    agent = agent_match.group(1)
    if skip_redteam and agent == "sol":
        print("deferred " + name + ": this two-role watch has the Sol route "
              "disabled; the root message remains untouched.")
        return False
    sol_topology_proof = None
    if agent == "sol" and not dry_run:
        try:
            sol_topology_proof = validate_live_sol_dispatch_topology()
        except (OSError, PrimaryWorktreeError) as exc:
            print("refused " + name + ": saved Sol worktree validation "
                  "failed (" + str(exc) + "); message left untouched.")
            return False
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
    effective_discovery_severity = DISCOVERY_SEVERITY
    saved_architect_severity = None
    if agent == "fable" and message.startswith(SOL_SEVERITY_HEADER):
        saved_architect_severity = architect_user_request_severity(
            message=message)
        if saved_architect_severity is None:
            reason = ("invalid public Architect request header; use exactly "
                      "'MAILBOX-SEVERITY: high', 'MAILBOX-SEVERITY: "
                      "medium', or 'MAILBOX-SEVERITY: low', followed by "
                      "one blank line")
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
        effective_discovery_severity = saved_architect_severity
    if agent == "sol":
        ticket_kind = sol_ticket_kind(message=message)
        severity_problem = sol_discovery_severity_problem(message=message)
        saved_severity = sol_discovery_severity(message=message)
        if saved_severity is not None:
            effective_discovery_severity = saved_severity
        reason = severity_problem
        if reason is None:
            reason = sol_ticket_refusal(
                ticket_kind=ticket_kind,
                total=demand_before_claim,
                fix_only=fix_only,
                transport_valid=valid_sol_transport(message=message),
                discovery_severity=saved_severity)
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

    if agent == "sol":
        placeholder_body = sol_ticket_body(message=message)
    elif agent == "fable":
        placeholder_body = architect_user_request_body(message=message)
    else:
        placeholder_body = message
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
        fix_only=fix_only,
        skip_redteam=skip_redteam,
        discovery_severity=effective_discovery_severity,
        saved_discovery=(ticket_kind == "discovery"),
        saved_architect_request=(saved_architect_severity is not None))
    # The dynamic banner precedes the byte-unchanged PREAMBLE. The
    # role-specific banner sits between them. Consequently PREAMBLE's
    # --- MESSAGE --- delimiter remains immediately before the exact raw
    # mailbox body, and the body remains the prompt's exact suffix. Only the
    # Architect route receives landing authority.
    command = AGENT_COMMANDS[agent] + [
        banner + agent_preamble(agent=agent, message=message) + PREAMBLE
        + message]

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
        env[MAX_CHARACTERS_ENVIRONMENT] = str(MAX_CHARACTERS)
        env[DISCOVERY_SEVERITY_ENVIRONMENT] = effective_discovery_severity
        env["MAILBOX_PRIMARY_WORKTREE"] = AGENT_CWD["fable"]
        env["MAILBOX_SHARED_NOTES"] = os.path.join(
            AGENT_CWD["fable"], "ai", "notes")
        env["MAILBOX_HANDOFF_CONTRACT"] = os.path.join(
            AGENT_CWD["fable"], "ai", "tools", "handoff_contract.py")
        env["MAILBOX_TICKET_CHANGE_GUARD"] = os.path.join(
            AGENT_CWD["fable"], "ai", "tools", "ticket_change_guard.py")
        if fix_only:
            env[FIX_ONLY_ENVIRONMENT] = "1"
        else:
            env.pop(FIX_ONLY_ENVIRONMENT, None)
        if skip_redteam:
            env[SKIP_REDTEAM_ENVIRONMENT] = "1"
        else:
            env.pop(SKIP_REDTEAM_ENVIRONMENT, None)
        try:
            if agent == "sol":
                revalidate_sol_dispatch_topology(
                    proof=sol_topology_proof)
            proc = subprocess.Popen(command,
                                    stdout=f,
                                    stderr=subprocess.STDOUT,
                                    cwd=AGENT_CWD[agent],
                                    env=env)
            try:
                if agent == "sol":
                    revalidate_sol_dispatch_topology(
                        proof=sol_topology_proof)
            except (OSError, PrimaryWorktreeError):
                proc.kill()
                proc.wait()
                raise
        except (OSError, ValueError) as exc:
            launch_error = exc
            f.write("\n--- dispatch could not start: " + str(exc) + " ---\n")
        except PrimaryWorktreeError as exc:
            launch_error = exc
            proc = None
            f.write("\n--- dispatch topology changed before launch: "
                    + str(exc) + " ---\n")
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


def drain_lane(paths, dry_run, fix_only=False, skip_redteam=False):
    """Dispatch ONE agent's pending messages, in order (a worker body).

    Arguments:
      paths   = this agent's message files, already sorted by sequence.
      dry_run  = True to print the would-be commands without running them.
      fix_only = True to launch only declared Sol closures.
      skip_redteam = True to exclude the Sol route from this watch.
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
            if skip_redteam:
                consumed = dispatch(
                    path=path, dry_run=dry_run, fix_only=fix_only,
                    skip_redteam=True)
            else:
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


def process_backlog(dry_run, fix_only=False, skip_redteam=False):
    """Dispatch the whole backlog: lanes in PARALLEL, each lane in order.

    In the default topology the three agents are independent sessions, so
    Opus can execute a unit while Sol attacks another -- but two messages to
    the SAME agent must
    stay sequential (a lane is one conversation partner, not a pool), and
    two agents sharing a WORKING DIRECTORY must too: concurrent turns in
    one git tree race each other's index (the 2026-07-14 incident where a
    live edit was swept into another agent's commit). So the parallel unit
    is the cwd: Fable+Opus (same worktree) serialize. Sol has its own saved
    worktree and can run alongside either Claude role. Only Architect turns
    take the root landing-authority lock.

    Arguments:
      dry_run  = True to print the would-be commands without running them.
      fix_only = True when a watch is closing existing ledger work only.
      skip_redteam = True for a watch that dispatches only Claude routes.

    Returns:
      None when there was no backlog, True when every message was consumed
      (or would dispatch in a dry run), and False when any dispatch or done
      archive failed.
    """
    all_backlog = pending_messages()
    if skip_redteam:
        backlog = [path for path in all_backlog
                   if PENDING_MESSAGE_RE.match(
                       os.path.basename(path)).group(1) != "sol"]
    else:
        backlog = all_backlog
    if skip_redteam:
        blockers = inflight_lane_blockers(skip_redteam=True)
    else:
        blockers = inflight_lane_blockers()
    if all_backlog:
        if skip_redteam:
            report_demand(backlog=all_backlog, skip_redteam=True)
        else:
            report_demand(backlog=all_backlog)
    if skip_redteam:
        report_deferred_sol_messages()
    if not backlog:
        if not blockers:
            return None
        for cwd in sorted(blockers):
            report_inflight_lane_block(
                blocker_paths=blockers[cwd],
                pending_count=0)
        return False
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

    def drain_and_record(cwd, paths, dry_run, fix_only, skip_redteam):
        """Run one cwd lane and retain failure even if its worker raises."""
        try:
            if skip_redteam:
                consumed = drain_lane(
                    paths=paths, dry_run=dry_run, fix_only=fix_only,
                    skip_redteam=True)
            else:
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
                                          "fix_only": fix_only,
                                          "skip_redteam": skip_redteam})
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
    return (not blockers
            and len(lane_outcomes) == len(lanes)
            and all(lane_outcomes.values()))


def report_deferred_sol_messages():
    """Print the exact number of root Sol messages held by this watch."""
    deferred = len(deferred_sol_messages())
    if deferred == 0:
        return
    noun = "message" if deferred == 1 else "messages"
    print("red-team route disabled; leaving " + str(deferred) + " to-sol "
          + noun + " queued and untouched.")


def report_demand(backlog, skip_redteam=False):
    """Print the waiting-work counts and second-Implementer reminder.

    The total is the waiting mailbox messages plus the "- OPEN" lines in
    ai/notes/backlog.md. Every watch pass that finds work and every --send
    prints it, so the person adding a message can see the resulting count.

    Arguments:
      backlog = Current waiting message paths from pending_messages().
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
    if total >= SECOND_IMPLEMENTER_THRESHOLD and not skip_redteam:
        print("  hint: " + str(SECOND_IMPLEMENTER_THRESHOLD)
              + " or more items are waiting. Ask the Architect to give Sol "
              "separate implementation jobs as a second Implementer, but "
              "only an Architect message with the required declaration "
              "changes Sol's role; "
              "otherwise Sol remains the Red Team.")
    report_landing_debt()


def landing_debt_snapshot():
    """Measure the current branch's content delta from main.

    Returns a mapping with ``available``, ``stat``, and ``changed_lines``.
    Git failures remain data instead of disappearing: every demand report
    must still print one truthful debt line when the main ref is unavailable.
    """
    proc = subprocess.run(
        ["git", "diff", "--shortstat", "main..HEAD"],
        capture_output=True,
        text=True,
        cwd=WORKTREE)
    if proc.returncode != 0:
        return {
            "available": False,
            "stat": "",
            "changed_lines": 0,
            "returncode": proc.returncode,
        }
    shortstat = proc.stdout.strip()
    changed_lines = 0
    # --shortstat prints e.g. "3 files changed, 120 insertions(+), 4
    # deletions(-)"; debt is the total content lines touched either way.
    for count, _keyword in re.findall(
            r"(\d+) (insertion|deletion)", shortstat):
        changed_lines = changed_lines + int(count)
    return {
        "available": True,
        "stat": shortstat,
        "changed_lines": changed_lines,
        "returncode": 0,
    }


def report_landing_debt(snapshot=None):
    """Print how much branch content has not yet landed on main.

    The milestone that must land is ONE FULL AUDIT TRAIL: the feature,
    its witness or gate leg, and the audit decision. Debt past
    LANDING_DEBT_LINE_LIMIT changed lines means an audited unit is
    sitting unlanded, which is how the 12,000-line batch landing of
    2026-07-14 happened (user rule: land at every audit-GO boundary,
    one unit per squash commit). Content is measured with git diff
    against main -- a commit count would never drop after a squash
    landing, because squashing leaves the original branch commits
    outside main's ancestry.

    Returns the structured snapshot so callers may reuse the measurement.
    """
    if snapshot is None:
        snapshot = landing_debt_snapshot()
    if not snapshot["available"]:
        print("landing debt: unavailable; git diff --shortstat main..branch "
              "exited " + str(snapshot["returncode"]))
        return snapshot
    if snapshot["stat"] == "":
        print("landing debt: none; the branch and main hold the "
              "same content")
        return snapshot
    print("landing debt: " + snapshot["stat"] + " vs main")
    if snapshot["changed_lines"] > LANDING_DEBT_LINE_LIMIT:
        print("  hint: more than " + str(LANDING_DEBT_LINE_LIMIT)
              + " unlanded lines means at least one full audit trail "
              "is overdue; squash-land the audited unit(s) to main "
              "now, one unit per commit "
              "(.claude/FABLE_ROLE.md, Landing granularity).")
    return snapshot


def landing_debt_state_path():
    """Return the ignored per-mailbox deduplication state path."""
    return os.path.join(MAILBOX, LANDING_DEBT_STATE_NAME)


def fsync_directory(directory):
    """Make a completed same-directory namespace transition durable."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def acquire_landing_debt_sequence_lock():
    """Acquire the publication lock without following or blocking on devices."""
    lock_path = os.path.join(MAILBOX, ".sequence.lock")
    try:
        parent = os.lstat(MAILBOX)
        if not stat.S_ISDIR(parent.st_mode):
            raise OSError("mailbox is not a regular directory")
        flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        print("landing-debt auto-handoff blocked: sequence lock failed ("
              + str(exc) + ").")
        return None
    lock_file = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        opened = os.fstat(lock_file.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("sequence lock is not a regular file")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        parent_after = os.lstat(MAILBOX)
        current = os.lstat(lock_path)
        if ((parent.st_dev, parent.st_ino)
                != (parent_after.st_dev, parent_after.st_ino)
                or not stat.S_ISDIR(parent_after.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)):
            raise OSError("sequence lock path changed")
    except OSError as exc:
        print("landing-debt auto-handoff blocked: sequence lock failed ("
              + str(exc) + ").")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()
        return None
    return lock_file


def release_landing_debt_sequence_lock(lock_file):
    """Release a landing-debt sequence lock."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def stable_regular_bytes(path, maximum_bytes, label, missing_ok=False,
                         complete=True):
    """Read one bounded, nonblocking, unchanged file or leading prefix."""
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ValueError(label + " disappeared before it could be read")
    except OSError as exc:
        raise ValueError("cannot inspect " + label + ": " + str(exc)) \
            from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(label + " is not a regular file")
    if complete and before.st_size > maximum_bytes:
        raise ValueError(label + " is too large")
    flags = os.O_RDONLY | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("cannot open " + label + ": " + str(exc)) from exc
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or opened.st_size != before.st_size
                or opened.st_mtime_ns != before.st_mtime_ns):
            raise ValueError(label + " changed while it was opened")
        chunks = []
        remaining = maximum_bytes + 1 if complete else maximum_bytes
        while remaining > 0:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining = remaining - len(chunk)
        raw = b"".join(chunks)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as exc:
        raise ValueError(label + " changed after it was read") from exc
    if ((complete and len(raw) > maximum_bytes)
            or (after_open.st_dev, after_open.st_ino) != identity
            or (after_path.st_dev, after_path.st_ino) != identity
            or not stat.S_ISREG(after_path.st_mode)
            or after_open.st_size != before.st_size
            or after_path.st_size != before.st_size
            or after_open.st_mtime_ns != before.st_mtime_ns
            or after_path.st_mtime_ns != before.st_mtime_ns
            or (complete and len(raw) != before.st_size)):
        raise ValueError(label + " changed while it was read")
    return raw


def unique_json_object(pairs):
    """Build one JSON object while refusing every duplicate key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + str(key))
        result[key] = value
    return result


def read_landing_debt_state():
    """Read and validate the active landing-debt episode state.

    Missing state starts generation one. Malformed, redirected, or oversized
    state is a blocker, never permission to flood another Architect message.
    """
    raw = stable_regular_bytes(
        path=landing_debt_state_path(),
        maximum_bytes=MAX_LANDING_DEBT_STATE_BYTES,
        label="landing-debt state",
        missing_ok=True)
    if raw is None:
        return {"schema": LANDING_DEBT_STATE_SCHEMA,
                "generation": 1, "active": False}
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
            OverflowError, ValueError) as exc:
        raise ValueError("landing-debt state is invalid JSON") from exc
    generation = payload.get("generation") \
        if isinstance(payload, dict) else None
    active = payload.get("active") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict)
            or set(payload) != {"schema", "generation", "active"}
            or payload.get("schema") != LANDING_DEBT_STATE_SCHEMA
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or generation > MAX_CYCLE_COUNT
            or not isinstance(active, bool)):
        raise ValueError("landing-debt state has an invalid schema")
    return {"schema": LANDING_DEBT_STATE_SCHEMA,
            "generation": generation, "active": active}


def write_landing_debt_state(state):
    """Publish validated debt state through an fsynced atomic replacement."""
    os.makedirs(MAILBOX, exist_ok=True)
    payload = (json.dumps(state, sort_keys=True, separators=(",", ":"))
               + "\n").encode("utf-8")
    handle, temporary = tempfile.mkstemp(
        prefix=".landing-debt-", dir=MAILBOX)
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, landing_debt_state_path())
        fsync_directory(directory=MAILBOX)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def automatic_landing_debt_marker(generation):
    """Return the stable identity of one high-debt episode's message."""
    return (AUTOMATIC_LANDING_DEBT_MARKER
            + " generation=" + str(generation))


def automatic_landing_debt_message(snapshot, generation):
    """Build the bounded landing-only instruction for the Architect lane."""
    return (
        automatic_landing_debt_marker(generation=generation) + "\n"
        "LANDING-ONLY ARCHITECT TURN. The watcher measured "
        + str(snapshot["changed_lines"]) + " changed content lines ("
        + snapshot["stat"] + ") against main, past the "
        + str(LANDING_DEBT_LINE_LIMIT) + "-line limit. Do no implementation "
        "and no broad review. Audit each unadjudicated unit in this bounded "
        "debt, then land each GO unit as its own squash commit. Before EVERY "
        "squash run `git log main..<branch> --oneline`; any foreign commit "
        "without an Architect GO is a STOP, so abort that whole-branch "
        "squash and land only a fully audited subset or wait. Push each "
        "accepted landing and merge main back into the working branch. "
        "Binding: after all audited units land, this thread is TERMINAL and "
        "no reply is owed; if a STOP remains, write only the bounded audit "
        "handoff needed to clear it.\n")


def automatic_landing_debt_message_exists(generation):
    """Return whether this episode's marker exists in any mailbox state."""
    marker = automatic_landing_debt_marker(
        generation=generation).encode("ascii")
    if not os.path.isdir(MAILBOX):
        return False
    for directory, child_directories, names in os.walk(
            MAILBOX, followlinks=False, onerror=_raise_walk_error):
        child_directories[:] = [
            name for name in child_directories
            if not os.path.islink(os.path.join(directory, name))]
        for name in names:
            if re.fullmatch(r"\d+[a-z]?-to-fable\.md", name) is None:
                continue
            path = os.path.join(directory, name)
            prefix = stable_regular_bytes(
                path=path,
                maximum_bytes=len(marker) + 1,
                label="Fable message " + name,
                complete=False)
            if prefix not in (marker, marker + b"\n"):
                continue
            raw = stable_regular_bytes(
                path=path,
                maximum_bytes=MAX_AUTOMATIC_MESSAGE_SCAN_BYTES,
                label="Fable message " + name)
            lines = raw.splitlines()
            if lines and lines[0] == marker:
                if (len(lines) < 2
                        or not lines[1].startswith(
                            b"LANDING-ONLY ARCHITECT TURN.")):
                    raise ValueError(
                        "automatic landing-debt marker has an invalid body "
                        "in " + path)
                return True
    return False


def publish_message_locked(agent, payload, attempts=20):
    """Atomically publish one message while the caller holds sequence lock."""
    for _ in range(attempts):
        path = os.path.join(
            MAILBOX, next_seq() + "-to-" + agent + ".md")
        handle, temporary = tempfile.mkstemp(prefix=".message-", dir=MAILBOX)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                if not payload.endswith("\n"):
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Same-directory hard-link publication never replaces a
                # manually created destination or exposes partial bytes.
                os.link(temporary, path)
            except FileExistsError:
                continue
            # The state may suppress replay only after the directory entry
            # itself survives a crash, not merely the payload inode.
            fsync_directory(directory=MAILBOX)
            return path
        finally:
            if os.path.isfile(temporary):
                os.remove(temporary)
    return None


def reconcile_landing_debt_handoff(snapshot=None):
    """Measure one watch pass and queue at most one handoff per debt episode.

    The active state remains set after the message moves to inflight, done, or
    failed, so repeated high-debt passes cannot flood Fable. Once content debt
    returns to or below the limit, the next generation is armed for a future
    independent high-debt episode.

    Returns the structured measurement made by this watch pass.
    """
    if snapshot is None:
        snapshot = landing_debt_snapshot()
    if not snapshot["available"]:
        return snapshot
    os.makedirs(MAILBOX, exist_ok=True)
    lock_file = acquire_landing_debt_sequence_lock()
    if lock_file is None:
        return snapshot
    try:
        try:
            state = read_landing_debt_state()
        except ValueError as exc:
            print("landing-debt auto-handoff blocked: " + str(exc)
                  + ".")
            return snapshot
        if snapshot["changed_lines"] <= LANDING_DEBT_LINE_LIMIT:
            episode_seen = state["active"]
            if not episode_seen:
                try:
                    episode_seen = automatic_landing_debt_message_exists(
                        generation=state["generation"])
                except (OSError, ValueError) as exc:
                    print("landing-debt auto-handoff blocked: "
                          + str(exc) + ".")
                    return snapshot
            if episode_seen:
                if state["generation"] >= MAX_CYCLE_COUNT:
                    print("landing-debt auto-handoff blocked: episode "
                          "generation limit reached.")
                    return snapshot
                state = {
                    "schema": LANDING_DEBT_STATE_SCHEMA,
                    "generation": state["generation"] + 1,
                    "active": False,
                }
                write_landing_debt_state(state=state)
            return snapshot
        if state["active"]:
            return snapshot
        try:
            marker_exists = automatic_landing_debt_message_exists(
                generation=state["generation"])
        except (OSError, ValueError) as exc:
            print("landing-debt auto-handoff blocked: " + str(exc)
                  + ".")
            return snapshot
        if marker_exists:
            # Crash recovery can observe a linked marker from a publisher
            # that died before its directory fsync. Make that marker
            # durable before state is allowed to suppress a replay.
            fsync_directory(directory=MAILBOX)
            state["active"] = True
            write_landing_debt_state(state=state)
            return snapshot
        payload = automatic_landing_debt_message(
            snapshot=snapshot, generation=state["generation"])
        path = publish_message_locked(agent="fable", payload=payload)
        if path is None:
            print("landing-debt auto-handoff could not claim a sequence "
                  "number after 20 tries.")
            return snapshot
        print("queued automatic landing-debt handoff " + path)
        state["active"] = True
        # Do not dispatch the durable marker until active state is also
        # durable. An exception stops this watch pass with the message
        # still pending; a later pass adopts its exact marker first.
        write_landing_debt_state(state=state)
        return snapshot
    finally:
        release_landing_debt_sequence_lock(lock_file=lock_file)


def send(agent, text, dry_run, ticket_kind=None, severity=None):
    """Save one internal mailbox message or one user request for Architect.

    Arguments:
      agent   = recipient name "fable", "opus", or "sol" used inside this
                program. The public command line maps its sole ``architect``
                target to ``fable``. Role-to-role callers use this function
                or save the next numbered mailbox file.
      text    = exact message text; internal role messages point to the source
                note under ``ai/notes/``.
      dry_run = True to print the file path without writing the message.
      ticket_kind = ``closure`` or ``discovery`` for internal Sol work. The
                    exact internal Sol ping alone uses ``transport``.
      severity = the Architect-approved minimum ``high``, ``medium``, or
                 ``low`` value for an internal Sol discovery. Omission uses
                 the inherited run value or medium. Other ticket kinds and
                 internal recipients accept no severity here.

    Returns:
      True when the message was queued, or would be queued in a dry run.
    """
    try:
        effective_severity = (
            resolve_discovery_severity(cli_value=severity)
            if ticket_kind == "discovery" else severity)
    except ValueError as exc:
        print("refused --send " + agent + ": " + str(exc) + ".")
        return False

    def refusal_now():
        """Return a current Sol-send refusal without changing disk."""
        if agent != "sol":
            if severity is not None:
                return "--severity is valid only with --send sol discovery"
            return None
        if skip_redteam_policy_active():
            return ("an active two-role watch has the Sol route disabled; "
                    "wait for it to end or restart without --skip-redteam")
        transport_valid = (
            ticket_kind == "transport"
            and text == transport_ping_text(agent="sol"))
        return sol_ticket_refusal(
            ticket_kind=ticket_kind,
            total=total_open_demand(),
            fix_only=(fix_only_environment_active()
                      or fix_only_watch_is_active()),
            transport_valid=transport_valid,
            discovery_severity=effective_severity)

    reason = refusal_now()
    if reason is not None:
        print("refused --send " + agent + ": " + reason + ".")
        return False

    payload = text
    if agent == "sol":
        if ticket_kind in SOL_DISPATCH_TICKET_KINDS:
            payload = sol_ticket_payload(
                ticket_kind=ticket_kind, text=text,
                discovery_severity=effective_severity)
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
                print("refused --send " + agent + ": " + reason + ".")
                return False
            for _ in range(20):
                path = publish_message_locked(
                    agent=agent, payload=payload, attempts=1)
                if path is not None:
                    print("queued " + path)
                    warn_if_mailbox_unwatched()
                    if skip_redteam_policy_active():
                        report_demand(
                            backlog=pending_messages(), skip_redteam=True)
                    else:
                        report_demand(backlog=pending_messages())
                    return True
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
    global MAX_CHARACTERS
    global DISCOVERY_SEVERITY
    global _ACTIVE_WATCH_RENDEZVOUS

    parser = argparse.ArgumentParser(
        description="save mailbox requests and start the assigned role for "
                    "each request")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the message files and work this command "
                             "would handle, but do not start a role or write "
                             "a message file")
    parser.add_argument("--once", action="store_true",
                        help="start every request that is waiting now, "
                             "then exit")
    parser.add_argument("--watch", action="store_true",
                        help="check the mailbox every 20 seconds and start "
                             "waiting requests")
    parser.add_argument("--cycle", metavar="count",
                        type=nonnegative_cycle_count, default=None,
                        help="with --watch, stop after this many work "
                             "periods; one period ends after five requests "
                             "finish or 15 minutes pass from its start, but "
                             "waits for every job already starting or "
                             "running to finish; 0 "
                             "instead waits until no enabled role has a "
                             "waiting message and ai/notes/backlog.md has no "
                             "open item; omit this option to keep watching")
    parser.add_argument(
        "--max", dest="max_characters", metavar="characters",
        type=nonnegative_max_characters, default=None,
        help="with --watch or --once, limit each ticket to this many added "
             "and removed characters, counted from the starting saved Git "
             "version named in the Architect's instructions; use only "
             "digits 0 through 9; 0 means no limit (default: 0)")
    parser.add_argument("--skip-redteam", "--no-red-team",
                        dest="skip_redteam", action="store_true",
                        help="with --watch, start Architect and Implementer "
                             "jobs but no Red Team job; Red Team messages "
                             "remain waiting for a later watch without this "
                             "option")
    parser.add_argument("--fix-only", metavar="value", type=truthy_fix_only,
                        default=None,
                        help="with --watch, tell roles to finish work already "
                             "recorded in ai/notes/backlog.md and refuse new "
                             "Red Team discovery messages; this option does "
                             "not create mailbox requests from backlog text; "
                             "the value accepts 1, true, or yes in any "
                             "capitalization")
    parser.add_argument("--send", metavar="{architect}",
                        choices=["architect"],
                        help="save the user's ticket request for the "
                             "Architect and exit")
    parser.add_argument("--ping", metavar="{architect}",
                        choices=["architect"],
                        help="save a connection-check message for the "
                             "Architect; its reply is saved in a -to-user.md "
                             "file and is not sent to another role")
    parser.add_argument("--unit", default="",
                        help="the user's request text for --send architect; "
                             "include the path to its source note in "
                             "ai/notes/")
    parser.add_argument(
        "--severity", choices=DISCOVERY_SEVERITIES, default=None,
        help="minimum severity for new discovery tickets: high keeps only "
             "bugs that severely impact core functionality, cause data "
             "loss, halt system operations, or make the science wrong; "
             "medium also keeps probable normal-operation bugs "
             "but not improbable edge cases; low keeps every concrete "
             "discovered bug; with --send architect, save the choice for "
             "that request (default: medium)")
    parser.add_argument("--architect-model", metavar="MODEL",
                        type=validate_model_name,
                        default=DEFAULT_ARCHITECT_MODEL,
                        help="Claude model alias or full name used for the "
                             "Architect; mailbox filenames for this role "
                             "still contain fable (default: "
                             + DEFAULT_ARCHITECT_MODEL + ")")
    parser.add_argument("--implementer-model", metavar="MODEL",
                        type=validate_model_name,
                        default=DEFAULT_IMPLEMENTER_MODEL,
                        help="Claude model alias or full name used for the "
                             "Implementer; mailbox filenames for this role "
                             "still contain opus (default: "
                             + DEFAULT_IMPLEMENTER_MODEL + ")")
    parser.add_argument("--fable-effort", default=DEFAULT_FABLE_EFFORT,
                        choices=CLAUDE_EFFORT_CHOICES,
                        help="claude CLI reasoning effort for the Architect "
                             "(default: " + DEFAULT_FABLE_EFFORT + ")")
    parser.add_argument("--opus-effort", default=DEFAULT_OPUS_EFFORT,
                        choices=CLAUDE_EFFORT_CHOICES,
                        help="claude CLI reasoning effort for the Implementer "
                             "(default: " + DEFAULT_OPUS_EFFORT + ")")
    parser.add_argument("--sol-effort", default=DEFAULT_SOL_EFFORT,
                        choices=CODEX_EFFORT_CHOICES,
                        help="codex CLI reasoning effort for the Red Team "
                             "(default: "
                             + DEFAULT_SOL_EFFORT + ")")
    parser.add_argument("--dispatch-timeout", metavar="MINUTES",
                        type=positive_int, default=DISPATCH_TIMEOUT_MINUTES,
                        help="stop a running role after this many minutes and "
                             "try to move its request file to failed/; if the "
                             "result or move cannot be verified, the file may "
                             "remain in inflight/ for inspection (default: "
                             + str(DISPATCH_TIMEOUT_MINUTES) + ")")
    parser.add_argument("--claude-context", metavar="TOKENS",
                        type=positive_int,
                        default=DEFAULT_CLAUDE_CONTEXT_BUDGET,
                        help="ask Claude to replace older Architect and "
                             "Implementer conversation text with a shorter "
                             "summary when it reaches this many tokens "
                             "(default: "
                             + str(DEFAULT_CLAUDE_CONTEXT_BUDGET) + ")")
    parser.add_argument("--sol-context", metavar="TOKENS",
                        type=positive_int, default=DEFAULT_SOL_CONTEXT_BUDGET,
                        help="ask Codex to replace older Red Team "
                             "conversation text with a shorter summary when "
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
    if args.cycle is not None and not args.watch:
        print("--cycle is valid only with --watch")
        return 1
    if args.max_characters is not None:
        conflicting_action = (
            not (args.watch or args.once)
            or args.send is not None or args.ping is not None)
        if conflicting_action:
            print("--max is valid only with --watch or --once")
            return 1
    if args.skip_redteam:
        conflicting_action = (
            not args.watch or args.once or args.send is not None
            or args.ping is not None or args.dry_run)
        if conflicting_action:
            print("--skip-redteam is valid only with --watch")
            return 1
    if args.severity is not None:
        severity_run = args.watch or args.once
        severity_send = args.send == "architect"
        if not (severity_run or severity_send):
            print("--severity is valid only with --watch, --once, or "
                  "--send architect")
            return 1
    if args.send is not None and not args.unit:
        print("--send architect needs --unit with the user's request text")
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

    selected_discovery_severity = DEFAULT_DISCOVERY_SEVERITY
    severity_action = args.watch or args.once or args.send == "architect"
    if severity_action:
        try:
            selected_discovery_severity = resolve_discovery_severity(
                cli_value=args.severity)
        except ValueError as exc:
            print("refused discovery severity: " + str(exc))
            return 1

    # Select the durable coordination checkout after EVERY semantic refusal
    # and before command rebuilding, mailbox mkdir, lock, claim, or send.
    # Import-based focused tests remain pure; the real CLI/subprocess path
    # proves primary provisioning and re-exec end to end.
    if __name__ == "__main__" and (primary_actions or args.dry_run):
        try:
            ensure_primary_execution(
                live_action=bool(primary_actions), dry_run=args.dry_run)
        except PrimaryWorktreeError as exc:
            print("primary worktree error: " + str(exc))
            return 1

    fix_only = args.fix_only is True
    skip_redteam = args.skip_redteam
    MAX_CHARACTERS = (DEFAULT_MAX_CHARACTERS
                      if args.max_characters is None
                      else args.max_characters)
    DISCOVERY_SEVERITY = selected_discovery_severity

    DISPATCH_TIMEOUT_MINUTES = args.dispatch_timeout
    CLAUDE_CONTEXT_BUDGET = args.claude_context

    if args.watch or args.once:
        report_ticket_character_limit()
        report_discovery_severity(
            fix_only=fix_only, skip_redteam=skip_redteam)

    # Rebuild the dispatch commands at the requested models and efforts. The
    # watch start lines echo both so terminal scroll-back identifies the exact
    # role assignment independently of the legacy route filenames.
    AGENT_COMMANDS = build_agent_commands(
        fable_effort=args.fable_effort,
        opus_effort=args.opus_effort,
        sol_effort=args.sol_effort,
        sol_context_budget=args.sol_context,
        architect_model=args.architect_model,
        implementer_model=args.implementer_model,
        sol_worktree=AGENT_CWD["sol"],
        shared_notes=(ACTIVE_TOPOLOGY["shared_notes"]
                      if ACTIVE_TOPOLOGY is not None
                      else os.path.join(AGENT_CWD["fable"], "ai", "notes")))
    if args.watch:
        print("role models: architect=" + args.architect_model
              + " implementer=" + args.implementer_model
              + " (internal mailbox names: fable/opus)")
        if skip_redteam:
            print("effort levels: architect/fable=" + args.fable_effort
                  + " implementer/opus=" + args.opus_effort
                  + " sol=disabled")
            print("context budgets: architect/implementer="
                  + str(args.claude_context)
                  + " sol=disabled (a Claude turn compacts at its budget)")
            print("two-role watch: Red Team and the entire Sol route are "
                  "disabled; existing to-sol messages stay queued and "
                  "untouched")
        else:
            print("effort levels: architect/fable=" + args.fable_effort
                  + " implementer/opus=" + args.opus_effort
                  + " sol=" + args.sol_effort)
            print("context budgets: architect/implementer="
                  + str(args.claude_context)
                  + " sol=" + str(args.sol_context)
                  + " tokens (a turn compacts at its budget)")
        if args.cycle == 0:
            if skip_redteam:
                print("cycle 0: wait until no Architect or Implementer "
                      "message is waiting or running and "
                      "ai/notes/backlog.md has no '- OPEN' item, then exit; "
                      "this two-role watch ignores Red Team messages")
            else:
                print("cycle 0: wait until no role message is waiting or "
                      "running and ai/notes/backlog.md has no '- OPEN' item, "
                      "then exit")
        elif args.cycle is not None:
            print("cycle limit: stop after " + str(args.cycle)
                  + " completed work periods; finish every role job already "
                  "starting or running before exit")

    if args.ping:
        ping_text = transport_ping_text(agent="architect")
        queued = send(
            agent="fable",
            text=ping_text,
            dry_run=args.dry_run)
        return 0 if queued else 1

    if args.send:
        request = architect_user_request_payload(
            text=args.unit,
            discovery_severity=selected_discovery_severity)
        queued = send(
            agent="fable",
            text=request,
            dry_run=args.dry_run)
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
        skip_redteam_lock = None
        if skip_redteam:
            skip_redteam_lock = acquire_skip_redteam_lock()
            if skip_redteam_lock is None:
                if fix_only_lock is not None:
                    release_fix_only_lock(lock_file=fix_only_lock)
                release_dispatch_lock(lock_file=dispatch_lock)
                return 1
        print("watching " + MAILBOX + " (press Ctrl-C only when the program "
              "says every enabled role is idle and shows the 20-second "
              "countdown; do not stop while a role is starting or running)")
        if fix_only:
            print("fix-only watch active: finish only work already listed "
                  "in ai/notes/backlog.md; do not add tickets for newly "
                  "found problems or new backlog lines")
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
        completed_cycles = 0
        cycle_completion_barrier = None
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
                # This watch-only hook runs on idle and busy passes alike.
                # Above-limit content debt publishes one deduplicated
                # Architect landing-only turn before the pass snapshots the
                # pending queue; --once, --dry-run, and --send never invoke it.
                reconcile_landing_debt_handoff()
                if fix_only:
                    if skip_redteam:
                        backlog_outcome = process_backlog(
                            dry_run=False, fix_only=True,
                            skip_redteam=True)
                    else:
                        backlog_outcome = process_backlog(dry_run=False, fix_only=True)
                else:
                    if skip_redteam:
                        backlog_outcome = process_backlog(
                            dry_run=False, skip_redteam=True)
                    else:
                        backlog_outcome = process_backlog(dry_run=False)
                if (rendezvous.source_changed()
                        or os.path.getmtime(source_path) != source_stamp):
                    print("daemon source changed on disk; exiting so "
                          "the next start runs it (relaunch --watch).")
                    return 0
                if rendezvous.window_ready():
                    completed_cycles = completed_cycles + 1
                    if args.cycle == 0:
                        barrier, completion_error = (
                            acquire_cycle_completion_barrier(
                                backlog_outcome=backlog_outcome,
                                skip_redteam=skip_redteam))
                        if barrier is not None:
                            cycle_completion_barrier = barrier
                            if skip_redteam:
                                report_cycle_work_complete(
                                    completed_cycles=completed_cycles,
                                    skip_redteam=True)
                            else:
                                report_cycle_work_complete(
                                    completed_cycles=completed_cycles)
                            return 0
                        if completion_error is not None:
                            report_cycle_completion_unverified(
                                error=completion_error)
                    if (args.cycle is not None and args.cycle > 0
                            and completed_cycles >= args.cycle):
                        if skip_redteam:
                            report_cycle_limit_exit(
                                completed_cycles=completed_cycles,
                                cycle_limit=args.cycle,
                                skip_redteam=True)
                        else:
                            report_cycle_limit_exit(
                                completed_cycles=completed_cycles,
                                cycle_limit=args.cycle)
                        return 0
                    run_safe_kill_countdown(controller=rendezvous)
                    # Queued work resumes immediately after the manufactured
                    # window rather than paying an extra ordinary poll delay.
                    continue
                if args.cycle == 0 and rendezvous.all_idle():
                    barrier, completion_error = (
                        acquire_cycle_completion_barrier(
                            backlog_outcome=backlog_outcome,
                            skip_redteam=skip_redteam))
                    if barrier is not None:
                        cycle_completion_barrier = barrier
                        if skip_redteam:
                            report_cycle_work_complete(
                                completed_cycles=completed_cycles,
                                skip_redteam=True)
                        else:
                            report_cycle_work_complete(
                                completed_cycles=completed_cycles)
                        return 0
                    if completion_error is not None:
                        report_cycle_completion_unverified(
                            error=completion_error)
                ordinary_safe = report_ordinary_safe_poll(
                    controller=rendezvous,
                    reset_cadence=args.cycle is None)
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
            if skip_redteam_lock is None:
                release_dispatch_lock(lock_file=dispatch_lock)
                if cycle_completion_barrier is not None:
                    release_cycle_completion_barrier(
                        lock_file=cycle_completion_barrier)
            else:
                release_dispatch_lock(lock_file=dispatch_lock)
                release_skip_redteam_lock(lock_file=skip_redteam_lock)
                if cycle_completion_barrier is not None:
                    release_cycle_completion_barrier(
                        lock_file=cycle_completion_barrier)

    print("choose one of --dry-run / --once / --watch / --send (see --help)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
