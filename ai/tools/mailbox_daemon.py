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
  - the Architect makes the GO/NO-GO decision but never changes main. A GO
    names one immutable Implementer candidate C in a decision-only request;
  - after the Architect process exits, the parent daemon verifies C, creates
    and records exact local squash landing L, queues the optional Sol review,
    and then attempts a bounded push. Implementer and Red Team turns never
    receive that landing authority;
  - the daemon changes only mailbox state and the exact accepted Git landing;
    it never authors source code or permanent notes;
  - every dispatch's full CLI output is archived under ai/notes/relay/.

Every live mailbox action converges on three persisted role worktrees. The first valid
action creates or safely adopts the Architect primary, the Implementer, and
Sol checkouts; later actions validate every saved Git identity and re-exec
this file from the Architect primary. The roles share the Architect's notes
directory, but never a mutable code checkout or index. Ordinary agent turns
never start in the user's main checkout.
`--ping` instead uses an empty temporary directory and creates no mailbox work.
AGENT_COMMANDS, the CLI binary paths, is the one machine-specific block.
`claude -p` runs one headless turn against the subscription; the session
needs enough tool permission to work unattended (set via the harness
settings or the flags there).

Usage:
    python ai/tools/mailbox_daemon.py --help           # all options + defaults
    python ai/tools/mailbox_daemon.py --dry-run        # show what would run
    python ai/tools/mailbox_daemon.py --once           # process backlog, exit
    python ai/tools/mailbox_daemon.py --ping           # check Claude + Sol
    python ai/tools/mailbox_daemon.py --ping --skip-redteam
                                                    # check Claude only
    python ai/tools/mailbox_daemon.py --clean-all      # discard all local AI
                                                    # worktrees and branches
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
    python ai/tools/mailbox_daemon.py --watch --dispatch-timeout 180
                                                    # extend emergency limit
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
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time

# Once main() passes CLI validation, every live action proves that this file
# lives in the saved primary coordination worktree (or re-execs the copy that
# does). Paths below can therefore remain simple derivations while launcher
# checkouts converge on one shared mailbox and notes tree. Their code trees,
# branches, and indexes stay separate.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_ROOT = os.path.dirname(SCRIPT_DIR)
WORKTREE = os.path.dirname(AI_ROOT)


def _load_local_role_contract_tool():
    """Load the protected reader beside this exact daemon file."""
    path = os.path.join(SCRIPT_DIR, "role_contract.py")
    try:
        from ai.tools import role_contract as packaged_tool
    except ImportError:
        packaged_tool = None
    if (packaged_tool is not None
            and os.path.realpath(packaged_tool.__file__)
            == os.path.realpath(path)):
        return packaged_tool
    spec = importlib.util.spec_from_file_location(
        "_mailbox_local_role_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the protected role contract reader")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


_ROLE_CONTRACT_TOOL = _load_local_role_contract_tool()
ROLE_CONTRACT = _ROLE_CONTRACT_TOOL.ROLE_CONTRACT


def _load_local_tool(filename, module_name, error, register=False):
    """Load one protected helper beside this exact daemon file."""
    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(
        module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(error)
    tool = importlib.util.module_from_spec(spec)
    if register:
        # dataclasses resolves the reopening record through sys.modules.
        sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    return tool


_REOPEN_TRANSITION = _load_local_tool(
    "reopen_transition.py", "_mailbox_local_reopen_transition",
    "cannot load the reopening transition checker", register=True)
_PROVIDER_HEALTH = _load_local_tool(
    "provider_health.py", "_mailbox_local_provider_health",
    "cannot load the provider health checker")
_CANDIDATE_ADMISSION = _load_local_tool(
    "candidate_admission.py", "_mailbox_local_candidate_admission",
    "cannot load the candidate admission checker")
_REVIEW_DISPATCH = _load_local_tool(
    "review_dispatch.py", "_mailbox_local_review_dispatch",
    "cannot load the routine review dispatcher")


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
DEFAULT_REVIEW_EFFORT = ROLE_CONTRACT["runtime"]["routine_review_effort"]
REVIEW_EFFORT = DEFAULT_REVIEW_EFFORT

# Model choice is independent of role. The fable/opus mailbox addresses are
# stable legacy route keys, while these defaults preserve existing launches.
# Any non-whitespace Claude alias or full model ID accepted by
# `claude --model` can override them per invocation.
DEFAULT_ARCHITECT_MODEL = "claude-fable-5"
DEFAULT_IMPLEMENTER_MODEL = "claude-opus-4-8"
IMPLEMENTER_PROVIDERS = ("claude", "ollama")
DEFAULT_IMPLEMENTER_PROVIDER = "claude"
SOL_MODEL = "gpt-5.6-sol"

CLAUDE_EXECUTABLE = "/Users/vivianmiranda/.local/bin/claude"
OLLAMA_EXECUTABLE = "ollama"
CODEX_EXECUTABLE = "/Applications/ChatGPT.app/Contents/Resources/codex"
PROVIDER_PING_TIMEOUT_SECONDS = 120

# Context budgets per dispatched turn (USER 2026-07-14: no bot runs
# with a context window above X tokens, where X is a command-line key
# and Sol's key is separate). Neither CLI takes a hard cap, so both are
# told to COMPACT (summarize their own history and continue) whenever
# the live context reaches the budget, instead of growing toward their
# native 1M windows: the Architect and Implementer coding runtimes read
# CLAUDE_CODE_AUTO_COMPACT_WINDOW from the environment; the Codex CLI
# (Sol) takes -c model_auto_compact_token_limit (accepted live,
# 2026-07-14). Override per launch with --claude-context / --sol-context.
# Each dispatch is also explicitly non-persistent, so a later turn cannot
# inherit an earlier ticket's provider conversation.
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

# Discovery breadth is saved beside severity in every discovery envelope.
# Dispatch must never reconstruct this decision from prose: a note pointer can
# replace the prose while the exact scope header remains durable.
DISCOVERY_SCOPES = ("bounded", "widespread")
DEFAULT_DISCOVERY_SCOPE = "bounded"

# dispatch() reads this for the claude environment; main() rebinds it
# from --claude-context. Sol's budget rides inside AGENT_COMMANDS.
CLAUDE_CONTEXT_BUDGET = DEFAULT_CLAUDE_CONTEXT_BUDGET

# A dispatched turn that runs past this many minutes is killed and its
# message parked in failed/ for inspection. The guard exists because a
# claude turn once printed "Execution error" and then hung, holding its
# lane for 21 minutes until a human Ctrl-C'd the watch (2026-07-14).
# A long Implementer turn first pauses for an Architect complexity review.
# The later hard timeout remains a fallback for a CLI that stops responding.
IMPLEMENTER_REVIEW_MINUTES = ROLE_CONTRACT["runtime"][
    "implementer_review_minutes"]
DEFAULT_DISPATCH_TIMEOUT_MINUTES = ROLE_CONTRACT["runtime"][
    "dispatch_timeout_default_minutes"]
DISPATCH_TIMEOUT_MINUTES = DEFAULT_DISPATCH_TIMEOUT_MINUTES
MAX_DISPATCH_TIMEOUT_MINUTES = 1000000
MAX_TIMEOUT_HISTORY_BYTES = 262144
MAX_TIMEOUT_HISTORY_EVENTS = 1000
MAX_BACKLOG_LEDGER_BYTES = 16777216

# A watch periodically manufactures one GLOBAL manual safe-stop opportunity.
# This cadence is deliberately unrelated to a ticket cycle. A ticket cycle
# always belongs to one ticket. In the default three-role mode it ends after
# the daemon-recorded local landing L receives its correlated Red Team return.
# In a mode without Red Team it ends when the daemon records local landing L.
# Five child turns or fifteen minutes merely creates the occasional 20-second
# Ctrl-C window.
# These are watch-only: --once and --dry-run retain their finite, delay-free
# behavior.
RENDEZVOUS_DISPATCH_INTERVAL = 5
RENDEZVOUS_MINUTE_INTERVAL = 15
SAFE_KILL_COUNTDOWN_SECONDS = 20
WATCH_POLL_SECONDS = 20
MAX_CYCLE_COUNT = 1000000
TICKET_CYCLE_STATE_NAME = ".ticket-cycle-state.json"
TICKET_CYCLE_LOCK_NAME = ".ticket-cycle-state.lock"
TICKET_CYCLE_STATE_SCHEMA = 6
MAX_TICKET_CYCLE_STATE_BYTES = 1024 * 1024
MAX_TICKET_CYCLE_RECORDS = 10000
CANDIDATE_STATE_NAME = ".ticket-candidate-state.json"
CANDIDATE_STATE_SCHEMA = 1
MAX_CANDIDATE_STATE_BYTES = 1024 * 1024
IMPLEMENTER_DELIVERY_PREFIX = ".validated-implementer-return-for-"
ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA = 1
MAX_ARCHITECT_NOTES_ADMIN_JOURNAL_BYTES = 16 * 1024
CANDIDATE_REF_ROOT = "refs/mailbox/cycles"
AUDIT_WORKTREE_PREFIX = "mailbox-audit-"
_PROTECTED_PATHS = ROLE_CONTRACT["protected_paths"]
ARCHITECT_PERMANENT_NOTE_PATHS = tuple(_PROTECTED_PATHS["permanent_notes"])
ARCHITECT_PROTECTED_REFERENCE_PATHS = tuple(
    _PROTECTED_PATHS["protected_reference_files"])
ROLE_CONTRACT_RELATIVE_PATH = _PROTECTED_PATHS["contract"]
BACKLOG_RELATIVE_PATH = ROLE_CONTRACT["backlog"]["path"]
ARCHITECT_ROLE_PATHS = tuple(_PROTECTED_PATHS["role_files"])
ARCHITECT_GUARD_PATHS_BY_NAME = dict(_PROTECTED_PATHS["guard_files"])
PERMANENT_NOTE_GUARD_RELATIVE_PATH = (
    ARCHITECT_GUARD_PATHS_BY_NAME["permanent_note_guard"])
ROLE_CONTRACT_TOOL_RELATIVE_PATH = (
    ARCHITECT_GUARD_PATHS_BY_NAME["role_contract_reader"])
ARCHITECT_GUARD_PATHS = tuple(ARCHITECT_GUARD_PATHS_BY_NAME.values())
ARCHITECT_TRUSTED_TOOL_PATHS_BY_NAME = dict(
    _PROTECTED_PATHS["trusted_tools"])
ARCHITECT_TRUSTED_TOOL_PATHS = (
    ARCHITECT_GUARD_PATHS
    + tuple(ARCHITECT_TRUSTED_TOOL_PATHS_BY_NAME.values()))
ARCHITECT_PROTECTED_POLICY_PATHS = (
    ARCHITECT_PERMANENT_NOTE_PATHS
    + ARCHITECT_PROTECTED_REFERENCE_PATHS
    + ARCHITECT_ROLE_PATHS + (ROLE_CONTRACT_RELATIVE_PATH,))
ARCHITECT_PROTECTED_TRACKED_PATHS = (
    ARCHITECT_PROTECTED_POLICY_PATHS + ARCHITECT_GUARD_PATHS)


def candidate_forbidden_files_from_contract(contract):
    """Return paths that no Implementer candidate may change."""
    protected = contract["protected_paths"]
    return frozenset(
        protected["candidate_forbidden_files"]
        + protected["permanent_notes"]
        + protected["protected_reference_files"]
        + protected["role_files"]
        + [protected["contract"]])


def control_plane_files_from_contract(contract):
    """Return the historical tool set needed to refuse old saved work."""
    protected = contract["protected_paths"]
    return frozenset(
        list(protected["guard_files"].values())
        + list(protected["trusted_tools"].values())
    )


ARCHITECT_CANDIDATE_FORBIDDEN_PREFIXES = tuple(
    _PROTECTED_PATHS["candidate_forbidden_prefixes"])


def _local_role_contract_tool():
    """Load the contract reader beside this daemon, never from another tree."""
    return _ROLE_CONTRACT_TOOL


def validate_role_contract_bindings(contract=None):
    """Validate one policy snapshot and its non-configurable safety floor."""
    tool = _local_role_contract_tool()
    if contract is None:
        contract = tool.load_role_contract(
            os.path.join(WORKTREE, ROLE_CONTRACT_RELATIVE_PATH))
        if contract != ROLE_CONTRACT:
            raise tool.RoleContractError(
                "role contract changed after daemon startup; restart before "
                "admitting more work")
    else:
        tool.validate_role_contract(contract)

    protected = contract["protected_paths"]
    worktrees = contract["worktrees"]
    if worktrees != ROLE_CONTRACT["worktrees"]:
        raise tool.RoleContractError(
            "worktree policy changes require an explicit saved-state "
            "migration")
    expected_branch_refs = tuple(
        "refs/heads/" + prefix for prefix in (
            worktrees["claude_branch_prefix"],
            worktrees["sol_branch_prefix"],
            worktrees["legacy_cleanup_prefix"]))
    if tuple(AI_BRANCH_PREFIXES) != expected_branch_refs:
        raise tool.RoleContractError(
            "cleanup branch prefixes disagree with the role contract")
    runtime_transport = {
        os.path.relpath(path, WORKTREE).replace(os.sep, "/") + "/"
        for path in (MAILBOX, RELAY_DIR)}
    if not runtime_transport.issubset(
            set(protected["candidate_forbidden_prefixes"])):
        raise tool.RoleContractError(
            "mailbox paths are not protected from candidates")
    return contract


def role_contract_snapshot_problem():
    """Describe a contract edit made after this process loaded its policy."""
    try:
        current = _local_role_contract_tool().load_role_contract(
            os.path.join(WORKTREE, ROLE_CONTRACT_RELATIVE_PATH))
    except (OSError, RuntimeError, ValueError) as exc:
        return "role contract on disk is invalid: " + str(exc)
    if current != ROLE_CONTRACT:
        return ("role contract changed after daemon startup; restart the "
                "watcher before admitting more work")
    return None


def report_role_contract_restart():
    """Print the current policy stop and return its process exit status."""
    problem = role_contract_snapshot_problem()
    if problem is None:
        problem = ("role contract changed during this mailbox pass; restart "
                   "before admitting more work")
    print(problem + ".")
    return 1 if problem.startswith("role contract on disk is invalid") else 0


def role_contract_exit_status():
    """Return a watch exit code unless an exact policy landing may finish."""
    problem = role_contract_snapshot_problem()
    if problem is None:
        return None
    invalid = problem.startswith("role contract on disk is invalid")
    if not invalid and architect_notes_transition_pending():
        return None
    return report_role_contract_restart()


BACKLOG_GUARD_STATE_NAME = ".backlog-guard.json"
BACKLOG_SYNC_RECOVERY_NAME = ".backlog-sync-recovery"
MAX_PROTECTED_NOTE_BYTES = ROLE_CONTRACT["limits"][
    "protected_policy_file_bytes"]
MAX_BACKLOG_GUARD_STATE_BYTES = 16 * 1024
PROTECTED_STATE_RECHECK_ATTEMPTS = 20
PROTECTED_STATE_RECHECK_SECONDS = 0.05

# Three durable agent checkouts keep the Architect's coordination files, the
# Implementer's source edits, and the Red Team's review files out of each
# other's Git indexes. A first live action creates or adopts the Architect
# primary, then creates the exact Implementer and Sol checkouts. Later actions
# prove all three saved identities before dispatch. REPO_ROOT remains the
# user's checkout.
_WORKTREE_POLICY = ROLE_CONTRACT["worktrees"]
PRIMARY_WORKTREE_NAME = _WORKTREE_POLICY["architect_name"]
PRIMARY_BRANCH = _WORKTREE_POLICY["architect_branch"]
PRIMARY_STATE_NAME = ".mailbox-primary-worktree.json"
PRIMARY_LOCK_NAME = ".mailbox-primary-worktree.lock"
LEGACY_PRIMARY_STATE_SCHEMA = 1
PREVIOUS_PRIMARY_STATE_SCHEMA = 2
PREVIOUS_PRIMARY_TOPOLOGY_MARKER = "dedicated-sol-worktree-v1"
PRIMARY_STATE_SCHEMA = 3
PRIMARY_TOPOLOGY_MARKER = _WORKTREE_POLICY["topology"]
IMPLEMENTER_WORKTREE_NAME = _WORKTREE_POLICY["implementer_name"]
IMPLEMENTER_BRANCH = _WORKTREE_POLICY["implementer_branch"]
IMPLEMENTER_STATE_NAME = ".mailbox-implementer-worktree.json"
IMPLEMENTER_STATE_SCHEMA = 1
SOL_WORKTREE_NAME = _WORKTREE_POLICY["sol_name"]
SOL_BRANCH = _WORKTREE_POLICY["sol_branch"]
SOL_STATE_NAME = ".mailbox-sol-worktree.json"
SOL_STATE_SCHEMA = 1
CLAUDE_BRANCH_PREFIX = _WORKTREE_POLICY["claude_branch_prefix"]
SOL_BRANCH_PREFIX = _WORKTREE_POLICY["sol_branch_prefix"]
LEGACY_CLEANUP_PREFIX = _WORKTREE_POLICY["legacy_cleanup_prefix"]
CLEANUP_ACTION = _WORKTREE_POLICY["cleanup_action"]
AI_BRANCH_PREFIXES = (
    "refs/heads/" + CLAUDE_BRANCH_PREFIX,
    "refs/heads/" + SOL_BRANCH_PREFIX,
    "refs/heads/" + LEGACY_CLEANUP_PREFIX,
)
MAILBOX_TOPOLOGY_VERSION = 3
MAILBOX_PROTOCOL_VERSION = 5
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
    TICKET_CYCLE_LOCK_NAME,
})
CURRENT_ADOPTION_SAFE_REASONS = frozenset({
    "numbered mailbox history exists",
    "relay evidence exists",
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


def implementer_state_paths(repository_root):
    """Return deterministic paths used by Implementer bootstrap."""
    repository = os.path.abspath(repository_root)
    managed_root = os.path.join(repository, ".claude", "worktrees")
    return {
        "managed_root": managed_root,
        "state": os.path.join(managed_root, IMPLEMENTER_STATE_NAME),
        "default_path": os.path.join(
            managed_root, IMPLEMENTER_WORKTREE_NAME),
        "default_branch": IMPLEMENTER_BRANCH,
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


def _run_git(repository_root, arguments, check=True, input_bytes=None):
    """Run one argv-only Git command and return its completed process."""
    command = ["git", "-C", os.path.abspath(repository_root)] + list(arguments)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False,
                                input=input_bytes)
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
    elif schema == PREVIOUS_PRIMARY_STATE_SCHEMA:
        expected = base_keys | {"topology"}
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
    expected_topology = {
        PREVIOUS_PRIMARY_STATE_SCHEMA: PREVIOUS_PRIMARY_TOPOLOGY_MARKER,
        PRIMARY_STATE_SCHEMA: PRIMARY_TOPOLOGY_MARKER,
    }.get(schema)
    if (expected_topology is not None
            and state["topology"] != expected_topology):
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


def _remove_archive_copy_temporaries(worktree):
    """Remove regular copy residues left by an interrupted archive bridge."""
    roots = (
        os.path.join(worktree, "ai", "notes", "mailbox", "done"),
        os.path.join(worktree, "ai", "notes", "relay"),
    )
    for root in roots:
        if not os.path.lexists(root):
            continue
        info = os.lstat(root)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            continue
        for directory, _names, files in os.walk(
                root, followlinks=False, onerror=_raise_walk_error):
            removed = False
            for name in files:
                if not name.startswith(".primary-archive-"):
                    continue
                path = os.path.join(directory, name)
                entry = os.lstat(path)
                if stat.S_ISREG(entry.st_mode) and not stat.S_ISLNK(
                        entry.st_mode):
                    os.remove(path)
                    removed = True
            if removed:
                fsync_directory(directory=directory)


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


def _publish_adopted_primary_record(record, repository_root):
    """Publish an existing coordinator only while its mailbox is idle."""
    mailbox = os.path.join(
        record["path"], "ai", "notes", "mailbox")
    _plain_directory(
        path=os.path.dirname(mailbox), label="adopted notes directory")
    identity = _plain_directory(
        path=mailbox, label="adopted mailbox", create=True)
    dispatch_lock = _open_legacy_transport_lock(
        path=os.path.join(mailbox, ".dispatch.lock"), nonblocking=True)
    sequence_lock = None
    try:
        sequence_lock = _open_legacy_transport_lock(
            path=os.path.join(mailbox, ".sequence.lock"), nonblocking=True)
        _require_directory_identity(
            path=mailbox, identity=identity, label="adopted mailbox")
        reasons = coordination_transport_evidence(worktree=record["path"])
        allowed = CURRENT_ADOPTION_SAFE_REASONS | {
            "live watcher or once lock is held",
            "live sender or sequence lock is held",
        }
        if not reasons or not set(reasons).issubset(allowed):
            raise PrimaryWorktreeError(
                "adopted coordination transport changed before publication")
        _validate_primary_record(
            record=record, branch=record["branch"],
            repository_root=repository_root)
        state = _primary_state_for_record(
            record=record, repository_root=repository_root)
        _atomic_write_primary_state(
            state=state, path=primary_state_paths(repository_root)["state"])
        return state
    finally:
        if sequence_lock is not None:
            _release_legacy_transport_lock(lock_file=sequence_lock)
        _release_legacy_transport_lock(lock_file=dispatch_lock)


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
    default_record = _record_at_path(
        records=records, path=paths["default_path"])
    if (default_record is not None
            and default_record.get("branch") == PRIMARY_BRANCH):
        _remove_archive_copy_temporaries(worktree=default_record["path"])
    evidence = []
    for record in records:
        reasons = coordination_transport_evidence(worktree=record["path"])
        if reasons:
            evidence.append((os.path.abspath(record["path"]), reasons))
    bridge_main = _safe_main_archive_bridge(
        evidence=evidence, repository_root=repository_root,
        default_path=paths["default_path"])

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
            return _publish_adopted_primary_record(
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
    if state["schema"] not in {
            LEGACY_PRIMARY_STATE_SCHEMA, PREVIOUS_PRIMARY_STATE_SCHEMA}:
        raise PrimaryWorktreeError(
            "cannot upgrade unsupported primary-worktree state")
    # An old process can validate schema 1, pause before taking the dispatch
    # lock, and resume after an apparent in-place migration. No filesystem
    # lock introduced by this newer code can make that already-admitted old
    # process re-read state. Automatic migration would therefore make a false
    # safety claim. Preserve every byte and require an explicit stopped-old-
    # runtime recovery instead.
    raise PrimaryWorktreeError(
        "the saved mailbox topology predates the separate Implementer "
        "worktree and cannot be migrated safely while an "
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


def _bootstrap_ticket_state(primary_path):
    """Read the primary mailbox's current ticket state with strict schema."""
    path = os.path.join(primary_path, "ai", "notes", "mailbox",
                        TICKET_CYCLE_STATE_NAME)
    if not os.path.isfile(path):
        return empty_ticket_cycle_state()
    try:
        raw = stable_regular_bytes(
            path=path, maximum_bytes=MAX_TICKET_CYCLE_STATE_BYTES,
            label="bootstrap ticket-cycle state")
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_key_refusal)
    except (OSError, ValueError, UnicodeDecodeError,
            json.JSONDecodeError) as exc:
        raise PrimaryWorktreeError(
            "cannot verify bootstrap ticket-cycle state: " + str(exc)) \
            from exc
    try:
        return validate_ticket_cycle_state(payload=payload)
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(str(exc)) from exc


def _bootstrap_candidate_state(primary_path):
    """Read the primary mailbox's candidate ownership without globals."""
    path = os.path.join(primary_path, "ai", "notes", "mailbox",
                        CANDIDATE_STATE_NAME)
    try:
        raw = stable_regular_bytes(
            path=path, maximum_bytes=MAX_CANDIDATE_STATE_BYTES,
            label="bootstrap ticket-candidate state", missing_ok=True)
    except (OSError, ValueError) as exc:
        raise PrimaryWorktreeError(str(exc)) from exc
    if raw is None:
        return empty_candidate_state()
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_key_refusal)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError,
            OverflowError, RecursionError) as exc:
        raise PrimaryWorktreeError(
            "bootstrap ticket-candidate state is not exact JSON") from exc
    if (not isinstance(payload, dict)
            or set(payload) != {"schema", "cycles"}
            or payload.get("schema") != CANDIDATE_STATE_SCHEMA
            or not isinstance(payload.get("cycles"), dict)
            or len(payload["cycles"]) > MAX_TICKET_CYCLE_RECORDS):
        raise PrimaryWorktreeError(
            "bootstrap ticket-candidate state has invalid keys")
    normalized = {}
    try:
        for cycle_id, record in payload["cycles"].items():
            expected_ref = cycle_candidate_ref(cycle_id=cycle_id)
            if (not isinstance(record, dict)
                    or set(record) != {"ref", "commit"}
                    or record.get("ref") != expected_ref
                    or not isinstance(record.get("commit"), str)
                    or FULL_COMMIT_RE.fullmatch(record["commit"]) is None):
                raise PrimaryWorktreeError(
                    "bootstrap ticket-candidate state has an invalid "
                    "cycle record")
            normalized[cycle_id] = {
                "ref": expected_ref, "commit": record["commit"]}
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(str(exc)) from exc
    return {"schema": CANDIDATE_STATE_SCHEMA, "cycles": normalized}


def _bootstrap_root_ticket_authority(message, target, ticket_state,
                                     candidate_state):
    """Prove a still-root ordinary GO is the exact journaled C-to-L work."""
    cycle_id, candidate_commit, mode, problem = _architect_go_request(
        message=message)
    if problem is not None:
        return False, problem
    record = candidate_state["cycles"].get(cycle_id)
    candidate_ref = cycle_candidate_ref(cycle_id=cycle_id)
    if (record != {"ref": candidate_ref, "commit": candidate_commit}
            or git_ref_commit(reference=candidate_ref) != candidate_commit):
        return False, "root Architect GO has no exact saved candidate C"
    landing_ref = cycle_landing_ref(cycle_id=cycle_id)
    saved_landing = git_ref_commit(reference=landing_ref)
    if saved_landing != target:
        if saved_landing is not None:
            try:
                saved_parent = _verify_prepared_landing(
                    cycle_id=cycle_id,
                    candidate_commit=candidate_commit,
                    landing_commit=saved_landing)
                main_problem = _prepared_landing_main_problem(
                    candidate_commit=candidate_commit,
                    landing_commit=saved_landing,
                    parent_commit=saved_parent,
                    current_main=target)
            except TicketCycleStateError as exc:
                return False, str(exc)
            if main_problem is not None:
                return False, main_problem
        return False, "root Architect GO has no exact target landing ref L"
    try:
        parent = _verify_prepared_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=target)
        _require_ancestor_or_same(
            ancestor=cycle_starting_commit(cycle_id), descendant=parent,
            label="root Architect GO landing does not preserve its base")
    except TicketCycleStateError as exc:
        return False, str(exc)
    active = ticket_state["active"].get(cycle_id)
    completed = ticket_state["completed"].get(cycle_id)
    if completed is not None:
        if completed != target:
            return False, "root Architect GO completed state names another L"
    elif (active is None or active.get("mode") != mode
          or (active.get("phase") != "implementation"
              and active.get("commit") != target)):
        return False, "root Architect GO does not match its saved cycle"
    return True, None


def _bootstrap_primary_ahead_notes_authority(primary_path, base_commit,
                                             notes_commit):
    """Prove clean primary P may wait ahead of main B for exact GO replay."""
    mailbox = os.path.join(primary_path, "ai", "notes", "mailbox")
    receipt_matches = []
    for directory in (mailbox, os.path.join(mailbox, "inflight")):
        for path in glob.glob(os.path.join(directory, "*-to-daemon.md")):
            try:
                raw = stable_regular_bytes(
                    path=path,
                    maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="ahead-primary permanent-note GO")
                message = raw.decode("utf-8", errors="strict")
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            returned_base, returned_notes, problem = (
                _architect_notes_go_request(message=message))
            if (problem is None and returned_base == base_commit
                    and returned_notes == notes_commit):
                receipt_matches.append((path, raw))
    if len(receipt_matches) != 1:
        return False
    try:
        require_architect_notes_commit_object(
            base_commit=base_commit, notes_commit=notes_commit)
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(str(exc)) from exc

    inflight = os.path.join(mailbox, "inflight")
    done = os.path.join(mailbox, "done")
    relay = os.path.join(primary_path, "ai", "notes", "relay")
    journal_prefix = ".pending-notes-admin-"
    journal_suffix = ".json"
    journal_paths = sorted(glob.glob(os.path.join(
        relay, journal_prefix + "*" + journal_suffix)))
    if len(journal_paths) != 1:
        raise PrimaryWorktreeError(
            "ahead Architect primary needs exactly one retained admin "
            "recovery journal; found " + str(len(journal_paths)))
    journal_name = os.path.basename(journal_paths[0])
    request_name = journal_name[len(journal_prefix):-len(journal_suffix)]
    request_match = PENDING_MESSAGE_RE.fullmatch(request_name)
    if request_match is None or request_match.group(1) != "fable":
        raise PrimaryWorktreeError(
            "ahead Architect primary has a malformed admin recovery "
            "journal name")
    admin_paths = [
        path for path in (os.path.join(inflight, request_name),
                          os.path.join(done, request_name))
        if regular_inode(path=path) is not None]
    if len(admin_paths) != 1:
        raise PrimaryWorktreeError(
            "ahead Architect primary recovery journal needs exactly one "
            "saved inflight or archived admin request; found "
            + str(len(admin_paths)))
    admin_path = admin_paths[0]
    try:
        admin_message = stable_regular_bytes(
            path=admin_path,
            maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="ahead-primary note admin").decode(
                "utf-8", errors="strict")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise PrimaryWorktreeError(
            "cannot verify saved permanent-note admin: " + str(exc)) \
            from exc
    if not is_architect_notes_admin_message(message=admin_message):
        raise PrimaryWorktreeError(
            "saved permanent-note admin is malformed")
    try:
        journal = read_architect_notes_admin_journal(
            request_name=request_name,
            request_message=admin_message, relay_dir=relay)
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(
            "ahead Architect primary has no valid admin recovery journal: "
            + str(exc)) from exc
    receipt_hash = hashlib.sha256(receipt_matches[0][1]).hexdigest()
    if (journal["base"] != base_commit
            or journal["phase"] != "validated-commit"
            or journal["notes_commit"] != notes_commit
            or journal["receipt_sha256"] != receipt_hash):
        raise PrimaryWorktreeError(
            "saved admin journal does not bind exact B/P receipt")
    try:
        _validate_current_protected_primary_state(
            primary_worktree=primary_path)
    except PrimaryWorktreeError:
        raise
    return True


def clean_user_main_matches(target):
    """
    Return whether the user checkout proves one ordinary main update.

    The repository's top folder belongs to the user. A clean checkout attached
    to `main` may therefore authorize its own exact commit without an internal
    ticket landing receipt.

    Arguments:
      target = the full commit currently stored in `refs/heads/main`.

    Returns:
      True only when invoked from the user checkout while it is clean,
      attached to `main`, and checked out at `target`.
    """
    branch = _run_git(
        repository_root=REPO_ROOT,
        arguments=["symbolic-ref", "-q", "HEAD"], check=False)
    try:
        branch_name = branch.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return False
    return (os.path.realpath(WORKTREE) == os.path.realpath(REPO_ROOT)
            and branch.returncode == 0
            and branch_name == "refs/heads/main"
            and worktree_head(worktree=REPO_ROOT) == target
            and not _tracked_worktree_changes(worktree=REPO_ROOT))


def _prepare_primary_backlog_overlay(primary_path, primary_head, target):
    """Preserve one sealed Architect backlog while its branch advances."""
    if not _clean_worktree_status(worktree=primary_path):
        return None
    try:
        sealed = _architect_only_sealed_backlog(worktree=primary_path)
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(str(exc)) from exc
    if sealed is None:
        raise PrimaryWorktreeError(
            "stale Architect primary has work beyond its sealed backlog; "
            "landing authority cannot advance it automatically")
    old = _run_git(
        repository_root=primary_path,
        arguments=["show", primary_head + ":" + BACKLOG_RELATIVE_PATH],
        check=False)
    new = _run_git(
        repository_root=primary_path,
        arguments=["show", target + ":" + BACKLOG_RELATIVE_PATH],
        check=False)
    if old.returncode != 0 or new.returncode != 0 or old.stdout != new.stdout:
        raise PrimaryWorktreeError(
            "main and the sealed Architect backlog both changed; preserve "
            "both versions and reconcile them before startup")
    backlog = os.path.join(primary_path, BACKLOG_RELATIVE_PATH)
    recovery = os.path.join(
        primary_path, "ai", "notes", BACKLOG_SYNC_RECOVERY_NAME)
    if os.path.lexists(recovery):
        raise PrimaryWorktreeError(
            "backlog sync recovery already exists; restart once to recover "
            "it before advancing the Architect primary")
    os.replace(backlog, recovery)
    restored = _run_git(
        repository_root=primary_path,
        arguments=["restore", "--source=HEAD", "--staged", "--worktree",
                   "--", BACKLOG_RELATIVE_PATH],
        check=False)
    if restored.returncode != 0 or _clean_worktree_status(primary_path):
        os.replace(recovery, backlog)
        raise PrimaryWorktreeError(
            "sealed Architect backlog could not be prepared for primary "
            "synchronization")
    return sealed, recovery


def bootstrap_sync_primary_from_main_authority(primary_path, primary_branch):
    """
    Advance a stale clean Architect worktree to an accepted main commit.

    An internal landing receipt proves a watcher-created commit. When no
    ticket is active, the clean user-owned main checkout may instead prove an
    ordinary user commit or pull.

    Arguments:
      primary_path = the saved Architect worktree.
      primary_branch = that worktree's saved full branch name.

    Returns:
      True when the Architect worktree advances; otherwise False.

    Raises:
      PrimaryWorktreeError if the worktree is dirty, divergent, or lacks
      either form of authority.
    """
    current_main = _run_git(
        repository_root=REPO_ROOT,
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"])
    try:
        target = current_main.stdout.decode(
            "ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError("current main is not ASCII") from exc
    primary_head = worktree_head(worktree=primary_path)
    if primary_head == target:
        return False
    _symbolic_worktree_branch(
        worktree=primary_path, expected_branch=primary_branch,
        label="Architect")
    old_backlog = _run_git(
        repository_root=primary_path,
        arguments=["cat-file", "-e",
                   primary_head + ":" + BACKLOG_RELATIVE_PATH],
        check=False)
    new_backlog = _run_git(
        repository_root=primary_path,
        arguments=["show", target + ":" + BACKLOG_RELATIVE_PATH],
        check=False)
    working_backlog = os.path.join(primary_path, BACKLOG_RELATIVE_PATH)
    if (old_backlog.returncode != 0 and new_backlog.returncode == 0
            and os.path.lexists(working_backlog)):
        sealed = _validate_sealed_backlog(primary_worktree=primary_path)
        if sealed != new_backlog.stdout:
            raise PrimaryWorktreeError(
                "tracked backlog migration conflicts with the sealed local "
                "backlog; both versions were preserved")
        os.unlink(working_backlog)
    ahead = _run_git(
        repository_root=REPO_ROOT,
        arguments=["merge-base", "--is-ancestor", target, primary_head],
        check=False)
    if ahead.returncode == 0:
        if _bootstrap_primary_ahead_notes_authority(
                primary_path=primary_path, base_commit=target,
                notes_commit=primary_head):
            print("kept saved Architect primary at authorized permanent-note "
                  "commit " + primary_head + " while main remains at its "
                  "exact base " + target, flush=True)
            return False
        raise PrimaryWorktreeError(
            "Architect primary is ahead of main without one exact pending "
            "B/P permanent-note GO")
    if ahead.returncode != 1:
        raise PrimaryWorktreeError(
            "cannot compare saved Architect primary with current main")
    try:
        _require_ancestor_or_same(
            ancestor=primary_head, descendant=target,
            label="stale Architect primary is not an ancestor of main")
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(str(exc)) from exc

    mailbox = os.path.join(primary_path, "ai", "notes", "mailbox")
    ticket_state = _bootstrap_ticket_state(primary_path=primary_path)
    candidate_state = _bootstrap_candidate_state(primary_path=primary_path)
    authorities = []
    root_problems = []
    for directory in (mailbox, os.path.join(mailbox, "done"),
                      os.path.join(mailbox, "inflight")):
        for path in glob.glob(os.path.join(directory, "*-to-daemon.md")):
            try:
                raw = stable_regular_bytes(
                    path=path,
                    maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="bootstrap landing request")
                message = raw.decode("utf-8", errors="strict")
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            if message.startswith(
                    MAILBOX_RETURN_HEADER + "architect-notes-go"):
                base, notes_commit, problem = (
                    _architect_notes_go_request(message=message))
                if directory == mailbox:
                    if problem is not None:
                        root_problems.append(problem)
                        continue
                    if notes_commit != target or base != primary_head:
                        root_problems.append(
                            "root Architect notes GO does not name exact "
                            "primary B and current main P")
                        continue
                if problem is None and notes_commit == target:
                    try:
                        require_architect_notes_commit_object(
                            base_commit=base, notes_commit=notes_commit)
                    except TicketCycleStateError as exc:
                        raise PrimaryWorktreeError(str(exc)) from exc
                    authorities.append(("notes", notes_commit))
                continue
            if directory == mailbox:
                valid, problem = _bootstrap_root_ticket_authority(
                    message=message, target=target,
                    ticket_state=ticket_state,
                    candidate_state=candidate_state)
                if valid:
                    authorities.append(("ticket", target))
                else:
                    root_problems.append(problem)
                continue
            cycle_id, _candidate, mode, problem = _architect_go_request(
                message=message)
            if problem is not None:
                continue
            recorded = ticket_state["completed"].get(cycle_id)
            if recorded is None:
                active = ticket_state["active"].get(cycle_id)
                if active is not None:
                    recorded = active.get("commit")
            if recorded is None:
                landing_ref = cycle_landing_ref(cycle_id=cycle_id)
                result = _run_git(
                    repository_root=REPO_ROOT,
                    arguments=["rev-parse", "--verify",
                               landing_ref + "^{commit}"], check=False)
                if result.returncode == 0:
                    try:
                        recorded = result.stdout.decode(
                            "ascii", errors="strict").strip()
                    except UnicodeDecodeError:
                        recorded = None
            if recorded == target and mode in ARCHITECT_COMMIT_MODES:
                authorities.append(("ticket", target))
    if root_problems:
        raise PrimaryWorktreeError(
            "stale Architect primary has an invalid root landing request: "
            + "; ".join(root_problems))
    if not authorities:
        if not clean_user_main_matches(target=target):
            raise PrimaryWorktreeError(
                "main is ahead of the Architect primary without an exact "
                "landing request or one clean user-owned main update")
        authority_label = "clean user main update"
    else:
        authority_label = "daemon-recorded landing"
    backlog_overlay = _prepare_primary_backlog_overlay(
        primary_path=primary_path, primary_head=primary_head, target=target)
    result = _run_git(
        repository_root=primary_path,
        arguments=["merge", "--ff-only", target], check=False)
    if result.returncode != 0:
        if backlog_overlay is not None:
            os.replace(backlog_overlay[1], working_backlog)
        raise PrimaryWorktreeError(
            "accepted main authority could not fast-forward the clean "
            "Architect primary")
    if backlog_overlay is not None:
        os.replace(backlog_overlay[1], working_backlog)
        try:
            restored_overlay = _architect_only_sealed_backlog(
                worktree=primary_path)
        except TicketCycleStateError as exc:
            raise PrimaryWorktreeError(str(exc)) from exc
        if restored_overlay != backlog_overlay[0]:
            raise PrimaryWorktreeError(
                "sealed Architect backlog was not restored after primary "
                "synchronization")
    elif _clean_worktree_status(worktree=primary_path):
        raise PrimaryWorktreeError(
            "accepted main authority did not leave a clean Architect primary")
    if worktree_head(worktree=primary_path) != target:
        raise PrimaryWorktreeError(
            "accepted main authority did not advance the Architect primary")
    print("advanced saved Architect primary to " + authority_label + " "
          + target, flush=True)
    return True


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
    """Return stable proofs for every primary role and ticket tool."""
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

    authoritative_files = tuple(
        ("role", os.path.join(primary, *path.split("/")))
        for path in ARCHITECT_ROLE_PATHS)
    authoritative_files += tuple(
        ("trusted tool", os.path.join(primary, *path.split("/")))
        for path in ARCHITECT_TRUSTED_TOOL_PATHS)
    authoritative_files += ((
        "role contract",
        os.path.join(primary, *ROLE_CONTRACT_RELATIVE_PATH.split("/"))),)
    file_proof = []
    for kind, path in authoritative_files:
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise PrimaryWorktreeError(
                "authoritative " + kind + " is missing: " + str(exc))
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise PrimaryWorktreeError(
                "authoritative " + kind + " must be a regular file: "
                + path)
        expected_real = os.path.join(
            primary_real, os.path.relpath(path, primary))
        if os.path.realpath(path) != expected_real:
            raise PrimaryWorktreeError(
                "authoritative " + kind + " is redirected: " + path)
        identity = (info.st_dev, info.st_ino, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        file_proof.append((kind, path, identity))

    proof = {
        "directories": tuple(directory_proof),
        "files": tuple(file_proof),
    }
    recheck_authoritative_role_files(proof=proof)
    return proof


def recheck_authoritative_role_files(proof, mutable_paths=()):
    """Require authoritative files to stay fixed outside an admin turn."""
    if (not isinstance(proof, dict)
            or set(proof) != {"directories", "files"}):
        raise PrimaryWorktreeError(
            "authoritative role-file proof is missing or malformed")
    for label, path, identity in proof["directories"]:
        _require_directory_identity(
            path=path, identity=identity, label=label)
    primary = proof["directories"][0][1]
    for kind, path, identity in proof["files"]:
        if os.path.relpath(path, primary) in mutable_paths:
            continue
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise PrimaryWorktreeError(
                "cannot revalidate authoritative " + kind + ": "
                + str(exc))
        current = (info.st_dev, info.st_ino, info.st_size,
                   info.st_mtime_ns, info.st_ctime_ns)
        if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
                or current != identity):
            raise PrimaryWorktreeError(
                "authoritative " + kind
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


def _implementer_state_for_record(record, repository_root):
    """Build the exact persisted Implementer record."""
    return {
        "schema": IMPLEMENTER_STATE_SCHEMA,
        "repository": git_common_directory(checkout=repository_root),
        "name": os.path.basename(record["path"]),
        "path": os.path.abspath(record["path"]),
        "branch": record["branch"],
    }


def validate_implementer_state(state, repository_root, primary_state,
                               allow_move=False):
    """Validate the fixed Implementer branch and its distinct checkout."""
    if state["schema"] != IMPLEMENTER_STATE_SCHEMA:
        raise PrimaryWorktreeError(
            "unsupported Implementer-worktree state schema")
    if state["branch"] != IMPLEMENTER_BRANCH:
        raise PrimaryWorktreeError(
            "saved Implementer worktree must use " + IMPLEMENTER_BRANCH)
    if primary_state["branch"] == state["branch"]:
        raise PrimaryWorktreeError(
            "Architect and Implementer must use different branches")
    resolved = validate_primary_state(
        state=state, repository_root=repository_root, allow_move=False,
        state_path=implementer_state_paths(repository_root)["state"])
    implementer_path = os.path.realpath(resolved["path"])
    if implementer_path == os.path.realpath(repository_root):
        raise PrimaryWorktreeError(
            "Implementer worktree must not be the user's checkout")
    if implementer_path == os.path.realpath(primary_state["path"]):
        raise PrimaryWorktreeError(
            "Architect and Implementer must use different worktrees")
    if allow_move and resolved != state:
        _atomic_write_primary_state(
            state=resolved,
            path=implementer_state_paths(repository_root)["state"])
        print("Implementer worktree moved by git; saved "
              + resolved["path"], flush=True)
    return resolved


def _tracked_worktree_changes(worktree):
    """Return staged, unstaged, or nonignored untracked worktree changes."""
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "-C", worktree, "status", "--porcelain=v1", "-z",
         "--untracked-files=all", "--ignore-submodules=none"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env=environment)
    if result.returncode != 0:
        raise PrimaryWorktreeError(
            "cannot inspect tracked or untracked changes in " + worktree)
    return result.stdout


def _optional_ref_commit(repository_root, reference):
    """Return one full ref commit, or ``None`` when the ref is absent."""
    result = _run_git(
        repository_root=repository_root,
        arguments=["rev-parse", "--verify", "--quiet",
                   reference + "^{commit}"], check=False)
    if result.returncode != 0:
        return None
    try:
        commit = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError(reference + " is not ASCII") from exc
    if FULL_COMMIT_RE.fullmatch(commit) is None:
        raise PrimaryWorktreeError(reference + " is not a full commit")
    return commit


def implementer_authority_snapshot(repository_root=None):
    """Snapshot Git state that an Implementer turn has no authority to move."""
    repository_root = REPO_ROOT if repository_root is None else repository_root
    symbolic = _run_git(
        repository_root=repository_root,
        arguments=["symbolic-ref", "-q", "HEAD"], check=False)
    if symbolic.returncode not in (0, 1):
        raise PrimaryWorktreeError(
            "cannot inspect the user's checked-out branch")
    return {
        "local main": _optional_ref_commit(
            repository_root, "refs/heads/main"),
        "origin/main": _optional_ref_commit(
            repository_root, "refs/remotes/origin/main"),
        "user checkout branch": (
            symbolic.stdout if symbolic.returncode == 0 else None),
        "user checkout HEAD": worktree_head(worktree=repository_root),
        "user checkout status": _tracked_worktree_changes(repository_root),
    }


def implementer_authority_changes(before, repository_root=None):
    """Name protected Git state that moved during one Implementer turn."""
    after = implementer_authority_snapshot(repository_root=repository_root)
    return [name for name in before if before[name] != after[name]]


def provision_or_reuse_implementer(repository_root, primary_state):
    """Create or validate the one fixed Implementer checkout."""
    paths = implementer_state_paths(repository_root=repository_root)
    _managed_primary_root(repository_root=repository_root, create=True)
    if os.path.lexists(paths["state"]):
        state = load_primary_state(path=paths["state"])
        return validate_implementer_state(
            state=state, repository_root=repository_root,
            primary_state=primary_state, allow_move=True)

    records = registered_worktrees(repository_root=repository_root)
    default_record = _record_at_path(
        records=records, path=paths["default_path"])
    if default_record is not None:
        if default_record.get("branch") != IMPLEMENTER_BRANCH:
            raise PrimaryWorktreeError(
                "default Implementer path is registered on another branch: "
                + paths["default_path"])
        _validate_primary_record(
            record=default_record, branch=IMPLEMENTER_BRANCH,
            repository_root=repository_root)
        state = _implementer_state_for_record(
            record=default_record, repository_root=repository_root)
        state = validate_implementer_state(
            state=state, repository_root=repository_root,
            primary_state=primary_state)
        _atomic_write_primary_state(state=state, path=paths["state"])
        print("recovered exact interrupted Implementer-worktree bootstrap "
              + state["path"], flush=True)
        return state

    branch_records = [record for record in records
                      if record.get("branch") == IMPLEMENTER_BRANCH]
    if branch_records:
        raise PrimaryWorktreeError(
            "Implementer branch is already checked out at an unexpected "
            "path: " + ", ".join(sorted(
                record["path"] for record in branch_records)))
    if os.path.lexists(paths["default_path"]):
        raise PrimaryWorktreeError(
            "default Implementer path exists but is not a registered "
            "worktree: " + paths["default_path"])
    if _branch_exists(
            repository_root=repository_root, branch=IMPLEMENTER_BRANCH):
        raise PrimaryWorktreeError(
            "Implementer branch already exists without its registered "
            "default worktree; refusing to reset or reuse it: "
            + IMPLEMENTER_BRANCH)
    if _tracked_worktree_changes(worktree=primary_state["path"]):
        raise PrimaryWorktreeError(
            "cannot split an Implementer worktree from a primary checkout "
            "with uncommitted tracked changes; commit or preserve that work "
            "before retrying")

    base = _run_git(
        repository_root=primary_state["path"],
        arguments=["rev-parse", "--verify", "HEAD^{commit}"])
    try:
        base_commit = base.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError(
            "primary commit identity is not ASCII") from exc
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", base_commit):
        raise PrimaryWorktreeError(
            "git returned an invalid primary commit")
    short_branch = IMPLEMENTER_BRANCH[len("refs/heads/"):]
    _run_git(
        repository_root=repository_root,
        arguments=["worktree", "add", "-b", short_branch,
                   paths["default_path"], base_commit])
    refreshed = registered_worktrees(repository_root=repository_root)
    created = _record_at_path(
        records=refreshed, path=paths["default_path"])
    if created is None:
        raise PrimaryWorktreeError(
            "git created no registered Implementer worktree; no state was "
            "saved")
    _validate_primary_record(
        record=created, branch=IMPLEMENTER_BRANCH,
        repository_root=repository_root)
    state = _implementer_state_for_record(
        record=created, repository_root=repository_root)
    state = validate_implementer_state(
        state=state, repository_root=repository_root,
        primary_state=primary_state)
    _atomic_write_primary_state(state=state, path=paths["state"])
    print("created Implementer worktree " + state["path"] + " on "
          + IMPLEMENTER_BRANCH, flush=True)
    return state


def validate_distinct_agent_states(primary_state, implementer_state,
                                   sol_state):
    """Prove the three role checkouts and branches are pairwise distinct."""
    paths = {
        os.path.realpath(primary_state["path"]),
        os.path.realpath(implementer_state["path"]),
        os.path.realpath(sol_state["path"]),
    }
    branches = {
        primary_state["branch"], implementer_state["branch"],
        sol_state["branch"],
    }
    if len(paths) != 3 or len(branches) != 3:
        raise PrimaryWorktreeError(
            "Architect, Implementer, and Sol require distinct worktrees "
            "and branches")


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


def configure_agent_worktrees(primary_path, implementer_path, sol_path,
                              primary_branch=PRIMARY_BRANCH):
    """Bind every role to its already-validated dispatch checkout."""
    AGENT_CWD["fable"] = os.path.abspath(primary_path)
    AGENT_CWD["opus"] = os.path.abspath(implementer_path)
    AGENT_CWD["sol"] = os.path.abspath(sol_path)
    AGENT_BRANCH["fable"] = primary_branch


def ensure_primary_execution(live_action, dry_run):
    """Validate all agent worktrees and re-exec from the primary.

    This is deliberately a CLI-boundary operation.  Importing this module for
    focused function tests remains pure; the real ``__main__`` call invokes it
    after every CLI semantic check and before any mailbox path is touched.
    """
    global ACTIVE_TOPOLOGY

    paths = primary_state_paths(repository_root=REPO_ROOT)
    implementer_paths = implementer_state_paths(repository_root=REPO_ROOT)
    sol_paths = sol_state_paths(repository_root=REPO_ROOT)
    state_exists = os.path.lexists(paths["state"])
    if dry_run and not state_exists:
        print("[dry-run] agent worktrees are not initialized; a live action "
              "would create Architect at " + paths["default_path"] + " on "
              + PRIMARY_BRANCH + ", Implementer at "
              + implementer_paths["default_path"] + " on "
              + IMPLEMENTER_BRANCH + ", and Sol at "
              + sol_paths["default_path"] + " on " + SOL_BRANCH
              + ". Previewing this launcher mailbox "
              "read-only.")
        configure_agent_worktrees(
            primary_path=WORKTREE,
            implementer_path=implementer_paths["default_path"],
            sol_path=sol_paths["default_path"])
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
                implementer_path=implementer_paths["default_path"],
                sol_path=sol_paths["default_path"],
                primary_branch=state["branch"])
            return state
        if state["schema"] != PRIMARY_STATE_SCHEMA:
            print("[dry-run] this saved state predates separate Architect "
                  "and Implementer worktrees; a live action would refuse "
                  "until every old daemon is stopped and the "
                  "separate-role topology is deliberately initialized.")
            configure_agent_worktrees(
                primary_path=state["path"],
                implementer_path=implementer_paths["default_path"],
                sol_path=sol_paths["default_path"],
                primary_branch=state["branch"])
            return state
        implementer_exists = os.path.lexists(implementer_paths["state"])
        if implementer_exists:
            implementer_state = load_primary_state(
                path=implementer_paths["state"])
            implementer_state = validate_implementer_state(
                state=implementer_state, repository_root=REPO_ROOT,
                primary_state=state, allow_move=False)
        else:
            print("[dry-run] a live action would create and save the "
                  "Implementer at " + implementer_paths["default_path"]
                  + " on " + IMPLEMENTER_BRANCH + ".")
            implementer_state = {"path": implementer_paths["default_path"],
                                 "branch": IMPLEMENTER_BRANCH}
        sol_exists = os.path.lexists(sol_paths["state"])
        if sol_exists:
            sol_state = load_primary_state(path=sol_paths["state"])
            sol_state = validate_sol_state(
                state=sol_state, repository_root=REPO_ROOT,
                primary_state=state, allow_move=False)
        else:
            print("[dry-run] a live action would create and save Sol at "
                  + sol_paths["default_path"] + " on " + SOL_BRANCH + ".")
            sol_state = {"path": sol_paths["default_path"]}
        if implementer_exists and sol_exists:
            validate_distinct_agent_states(
                primary_state=state, implementer_state=implementer_state,
                sol_state=sol_state)
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
            bootstrap_sync_primary_from_main_authority(
                primary_path=state["path"], primary_branch=state["branch"])
            _require_primary_daemon_topology_support(
                primary_path=state["path"])
            implementer_state = provision_or_reuse_implementer(
                repository_root=REPO_ROOT, primary_state=state)
            sol_state = provision_or_reuse_sol(
                repository_root=REPO_ROOT, primary_state=state)
            validate_distinct_agent_states(
                primary_state=state, implementer_state=implementer_state,
                sol_state=sol_state)
            shared_notes = validated_primary_notes(
                primary_path=state["path"])
            _bridge_local_sealed_backlog(primary_worktree=state["path"])
            validate_authoritative_role_files(primary_path=state["path"])
        finally:
            _release_primary_lock(lock_file=lock_file)
    else:
        return None

    configure_agent_worktrees(
        primary_path=state["path"],
        implementer_path=implementer_state["path"],
        sol_path=sol_state["path"], primary_branch=state["branch"])
    running_in_primary = (
        os.path.realpath(WORKTREE) == os.path.realpath(state["path"]))
    if live_action:
        ACTIVE_TOPOLOGY = {
            "primary_state": paths["state"],
            "implementer_state": implementer_paths["state"],
            "sol_state": sol_paths["state"],
            "primary_path": os.path.abspath(state["path"]),
            "primary_branch": state["branch"],
            "implementer_path": os.path.abspath(
                implementer_state["path"]),
            "sol_path": os.path.abspath(sol_state["path"]),
            "shared_notes": shared_notes,
        }
    if live_action and running_in_primary:
        main_result = _run_git(
            repository_root=REPO_ROOT,
            arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"])
        try:
            main_commit = main_result.stdout.decode(
                "ascii", errors="strict").strip()
            role_heads = {
                worktree_head(worktree=path)
                for path in (state["path"], implementer_state["path"],
                             sol_state["path"])
            }
            if (worktree_head(worktree=state["path"]) == main_commit
                    and role_heads != {main_commit}):
                sync_all_clean_role_baselines(target=main_commit)
        except (UnicodeDecodeError, OSError, TicketCycleStateError) as exc:
            raise PrimaryWorktreeError(
                "cannot align idle AI worktrees with current main: "
                + str(exc)) from exc
    if running_in_primary:
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

    def __init__(self, source_path=None, source_stamp=None,
                 ticket_cycle_limit=None, ticket_cycle_topology=None):
        if (ticket_cycle_limit is not None
                and ticket_cycle_topology not in ARCHITECT_COMMIT_MODES):
            raise ValueError(
                "a finite ticket-cycle controller needs a valid topology")
        self._lock = threading.Condition()
        self._active_attempts = 0
        self._in_flight = 0
        self._completed = 0
        self._ticket_cycles_completed = 0
        self._ticket_cycle_limit = ticket_cycle_limit
        self._ticket_cycle_topology = ticket_cycle_topology
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

    def begin_attempt(self, ignore_ticket_limit=False):
        """Return a permit, optionally for cycle-free administration."""
        while True:
            with self._lock:
                self._stop_for_source_change_locked()
                if (not ignore_ticket_limit
                        and self._ticket_cycle_limit is not None
                        and self._ticket_cycles_completed
                        >= self._ticket_cycle_limit):
                    return None
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

    def ticket_cycle_returned(self):
        """Record one completed ticket.

        A normal ticket reaches this method after its Red Team return. A
        ticket in a no-Red-Team mode reaches it after the daemon records local
        landing L. Child-turn cadence and manual safe-stop windows never call
        it.
        """
        with self._lock:
            self._ticket_cycles_completed = (
                self._ticket_cycles_completed + 1)
            self._lock.notify_all()

    def completed_ticket_cycles(self):
        """Return the completed ticket-cycle count for this watch."""
        with self._lock:
            return self._ticket_cycles_completed

    def ticket_cycle_limit_reached(self):
        """Return whether a positive cycle limit has already been met."""
        with self._lock:
            return (self._ticket_cycle_limit is not None
                    and self._ticket_cycles_completed
                    >= self._ticket_cycle_limit)

    def ticket_cycle_limit_value(self):
        """Return the positive ticket limit, or ``None`` when unbounded."""
        with self._lock:
            return self._ticket_cycle_limit

    def ticket_cycle_topology_value(self):
        """Return the topology bound to this finite watch, if any."""
        with self._lock:
            return self._ticket_cycle_topology

    def restore_completed_ticket_cycles(self, count):
        """Restore durable progress for an interrupted finite watch."""
        if (isinstance(count, bool) or not isinstance(count, int)
                or count < 0):
            raise ValueError("restored ticket-cycle count must be nonnegative")
        with self._lock:
            if self._ticket_cycles_completed != 0:
                raise ValueError("ticket-cycle progress was already restored")
            if (self._ticket_cycle_limit is not None
                    and count > self._ticket_cycle_limit):
                raise ValueError("restored progress exceeds the cycle limit")
            self._ticket_cycles_completed = count

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
_TOKEN_EXHAUSTION_STOP = threading.Event()


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


def _ticket_cycle_completed():
    """Count one verified ticket completion for the active watch."""
    controller = _ACTIVE_WATCH_RENDEZVOUS
    if controller is not None:
        controller.ticket_cycle_returned()


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

    This ordinary mailbox check never completes a ticket cycle. Only a
    correlated Red Team return for one daemon-recorded local landing L does
    that in the default three-role mode.
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
    failed_debt = architect_notes_failed_debt_error()
    if failed_debt is not None:
        return None, failed_debt
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
        try:
            active_cycles = active_ticket_cycle_count(
                skip_redteam=skip_redteam)
        except TicketCycleStateError as exc:
            active_cycles = None
            error = "cannot verify ticket-cycle state: " + str(exc)
        waiting_before = enabled_pending_messages(
            skip_redteam=skip_redteam)
        waiting_after = enabled_pending_messages(
            skip_redteam=skip_redteam)
    except OSError as exc:
        ledger = None
        error = "cannot verify pending mailbox messages: " + str(exc)
        waiting_before = []
        waiting_after = []
    notes_pending = architect_notes_transition_pending()
    if (error is None and ledger == 0 and active_cycles == 0
            and not notes_pending
            and not waiting_before and not waiting_after):
        return lock_file, None
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()
    return None, error


def release_cycle_completion_barrier(lock_file):
    """Release the final cycle-zero send barrier after watch-lock release."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def acquire_positive_cycle_exit_barrier(backlog_outcome,
                                        skip_redteam=False):
    """Fence sends and refuse finite exit while note administration waits."""
    failed_debt = architect_notes_failed_debt_error()
    if failed_debt is not None:
        return None, failed_debt
    if backlog_outcome is False:
        return None, None
    lock_path = os.path.join(MAILBOX, ".sequence.lock")
    lock_file = None
    try:
        lock_file = open(lock_path, "a+", encoding="utf-8")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        active = active_ticket_cycle_count(skip_redteam=skip_redteam)
        notes_pending = architect_notes_transition_pending()
    except (OSError, TicketCycleStateError) as exc:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except OSError:
                pass
        return None, "cannot verify finite-cycle exit: " + str(exc)
    if active == 0 and not notes_pending:
        return lock_file, None
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()
    return None, None


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
        print("implementation drain complete after "
              + str(completed_cycles) + " " + noun
              + "; no enabled Architect or Implementer message is waiting "
              "or running; ai/notes/backlog.md has no '- OPEN' item; "
              "disabled Red Team work remains untouched; watcher stopped.",
              flush=True)
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
    """Accept one provider model name without shell ambiguity."""
    if (not isinstance(value, str) or not value or "\x00" in value
            or any(character.isspace() for character in value)):
        raise argparse.ArgumentTypeError(
            "model must be one non-whitespace alias or full name")
    return value


def build_agent_commands(fable_effort, opus_effort, sol_effort,
                         sol_context_budget,
                         architect_model=DEFAULT_ARCHITECT_MODEL,
                         implementer_model=DEFAULT_IMPLEMENTER_MODEL,
                         sol_worktree=None, shared_notes=None,
                         implementer_provider=DEFAULT_IMPLEMENTER_PROVIDER):
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
      implementer_model  = Model name launched on the legacy opus route.
      sol_worktree       = validated worktree used as Sol's cwd and Codex
                           workspace root (default: deterministic first-run
                           path; live dispatch always passes saved state).
      shared_notes       = exact Claude-primary notes directory granted as
                           Sol's only additional writable directory.
      implementer_provider = ``claude`` for Anthropic Claude or ``ollama``
                           for an Ollama-served open-weight model.

    Returns:
      dict mapping "fable"/"opus"/"sol" to the argv list dispatch()
      appends the message to.
    """
    architect_model = validate_model_name(value=architect_model)
    implementer_model = validate_model_name(value=implementer_model)
    if implementer_provider not in IMPLEMENTER_PROVIDERS:
        raise ValueError(
            "Implementer provider must be claude or ollama")
    if sol_worktree is None:
        sol_worktree = sol_state_paths(REPO_ROOT)["default_path"]
    if shared_notes is None:
        shared_notes = os.path.join(WORKTREE, "ai", "notes")
    sol_worktree = os.path.abspath(sol_worktree)
    shared_notes = os.path.abspath(shared_notes)
    if implementer_provider == "claude":
        implementer_command = [
            CLAUDE_EXECUTABLE, "-p", "--no-session-persistence",
            "--model", implementer_model, "--effort", opus_effort,
            "--permission-mode", "acceptEdits"]
    else:
        # Ollama's supported headless coding route. Arguments after the
        # second ``--`` belong to Claude Code, which keeps the existing
        # worktree, tool, hook, and evidence boundary around the local model.
        implementer_command = [
            OLLAMA_EXECUTABLE, "launch", "claude",
            "--model", implementer_model, "--yes", "--",
            "-p", "--no-session-persistence",
            "--permission-mode", "acceptEdits"]
    commands = {
        # Absolute path: the user's conda shells resolve an OLDER claude
        # binary with a separate (logged-out) credential store; this one
        # is the logged-in v2.1.208 install (diagnosed 2026-07-14).
        "fable": [CLAUDE_EXECUTABLE, "-p", "--no-session-persistence",
                  "--model", architect_model,
                  "--effort", fable_effort,
                  "--permission-mode", "acceptEdits"],
        "opus": implementer_command,
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
        "sol": [CODEX_EXECUTABLE,
                "exec", "--ephemeral",
                "--model", SOL_MODEL,
                "-c", "model_reasoning_effort=" + sol_effort,
                "-c", "service_tier=standard",
                "-c", ("model_auto_compact_token_limit="
                       + str(sol_context_budget)),
                "--sandbox", "workspace-write",
                "--cd", sol_worktree,
                "--add-dir", shared_notes],
    }
    return commands


def check_provider_connectivity(
        architect_model, include_sol, dry_run=False,
        implementer_provider=DEFAULT_IMPLEMENTER_PROVIDER,
        implementer_model=DEFAULT_IMPLEMENTER_MODEL):
    """Check every distinct provider selected for this watch."""
    return _PROVIDER_HEALTH.check_connectivity(
        architect_model=architect_model,
        implementer_provider=implementer_provider,
        implementer_model=implementer_model,
        include_sol=include_sol,
        dry_run=dry_run,
        nonce=secrets.token_hex(16),
        claude_executable=CLAUDE_EXECUTABLE,
        ollama_executable=OLLAMA_EXECUTABLE,
        codex_executable=CODEX_EXECUTABLE,
        sol_model=SOL_MODEL,
        timeout=PROVIDER_PING_TIMEOUT_SECONDS,
        run=subprocess.run)


def routine_review_command(
        command, *, agent, ticket_kind=None, candidate_audit=False,
        reopening=False, checkpoint=False, integration=False, effort=None):
    """Return the exact command and label for one lower-cost review turn."""
    kind = _REVIEW_DISPATCH.review_kind(
        agent=agent, ticket_kind=ticket_kind,
        candidate_audit=candidate_audit, reopening=reopening,
        checkpoint=checkpoint, integration=integration)
    if kind is None:
        return list(command), None
    selected = REVIEW_EFFORT if effort is None else effort
    return (_REVIEW_DISPATCH.command_with_effort(
                command, agent=agent, effort=selected), kind)


def implementer_checkpoint_settings(python, hook_path):
    """Return the Implementer's time and context checkpoint hooks."""
    hook = {
        "type": "command",
        "command": python,
        "args": [hook_path],
        "timeout": 5,
    }
    return {"hooks": {
        "PostToolBatch": [{"hooks": [hook]}],
        "Stop": [{"hooks": [hook]}],
        "PreCompact": [{"matcher": "auto", "hooks": [hook]}],
    }}


# main() rebuilds this from the command-line flags; the module-level
# value keeps imports and direct function calls working at the defaults.
AGENT_COMMANDS = build_agent_commands(
    fable_effort=DEFAULT_FABLE_EFFORT,
    opus_effort=DEFAULT_OPUS_EFFORT,
    sol_effort=DEFAULT_SOL_EFFORT,
    sol_context_budget=DEFAULT_SOL_CONTEXT_BUDGET)

# The working directory each dispatched agent starts in. CLI bootstrap proves
# WORKTREE is replaced with the three validated saved paths at the CLI
# boundary. The defaults keep imports pure without making the user checkout a
# live Implementer or Sol fallback.
AGENT_CWD = {
    "fable": WORKTREE,
    "opus": implementer_state_paths(REPO_ROOT)["default_path"],
    "sol": sol_state_paths(REPO_ROOT)["default_path"],
}
AGENT_BRANCH = {
    "fable": PRIMARY_BRANCH,
    "opus": IMPLEMENTER_BRANCH,
    "sol": SOL_BRANCH,
}


def mailbox_lane_cwd(agent):
    """Return a serialization identity for an AI route or local daemon lane."""
    if agent == "daemon":
        return MAILBOX
    return AGENT_CWD[agent]

# Set only after a live CLI action validates both persisted records. Imported
# focused tests remain side-effect-free and may supply their own synthetic
# working directories without pretending to have passed live bootstrap.
ACTIVE_TOPOLOGY = None

# A message still carrying template placeholders has no job in it; refuse
# it instead of burning a live headless turn (learned from dispatch 0001).
PLACEHOLDER_MARKERS = ["<spec>", "<X>", "<section>", "<unit>",
                       "your message here"]

# Discovery admission uses the severity written in the open-ticket index.
# Ten open non-Low tickets stop new discovery. Backlog counts never select an
# AI role. The historical declaration is parsed only so it can be rejected;
# Sol is always advisory Red Team in a three-role watch.
DISCOVERY_ADMISSION_THRESHOLD = 10
BACKLOG_LEDGER = os.path.join(
    WORKTREE, *ROLE_CONTRACT["backlog"]["path"].split("/"))
OPEN_BACKLOG_TICKET_RE = re.compile(
    r"^- OPEN \*\*(CRITICAL|HIGH|MEDIUM|LOW)\*\* "
    r"\*\*(BUG FIX|NEW FUNCTIONALITY)\*\* — "
    r"\[([^]\r\n]+)\]\(#([a-z0-9]+(?:-[a-z0-9]+)*)\)$")
OPEN_BACKLOG_CANDIDATE_RE = re.compile(
    r"^\s*-\s+OPEN(?:\s|$)", re.IGNORECASE)
BACKLOG_DETAIL_ANCHOR_RE = re.compile(
    r'^<a id="([a-z0-9]+(?:-[a-z0-9]+)*)"></a>$')
BACKLOG_REOPEN_COUNT_RE = re.compile(
    r"^\*\*Red Team reopen count: (0|[1-9][0-9]*)\.\*\*$")
BACKLOG_REOPEN_COUNT_CANDIDATE_RE = re.compile(
    r"Red[ \t]+Team[ \t]+reopen[ \t]+count\b", re.IGNORECASE)
BACKLOG_REOPENING_RE = re.compile(
    r"^\*\*Red Team reopening: "
    r"(allowed|barred by Architect NO-GO)\.\*\*$")
BACKLOG_REOPENING_CANDIDATE_RE = re.compile(
    r"Red[ \t]+Team[ \t]+reopening\b", re.IGNORECASE)
SOL_TICKET_KINDS = (
    "closure", "discovery", "policy", "control-plane")
SOL_DISPATCH_TICKET_KINDS = SOL_TICKET_KINDS + ("transport",)
SOL_TICKET_HEADER = "MAILBOX-TICKET: "
SOL_SEVERITY_HEADER = "MAILBOX-SEVERITY: "
SOL_SCOPE_HEADER = "MAILBOX-SCOPE: "
MAILBOX_FLOW_HEADER = "MAILBOX-FLOW: "
MAILBOX_ADMISSION_HEADER = "MAILBOX-ADMISSION: "
MAILBOX_ADMIN_HEADER = "MAILBOX-ADMIN: "
MAILBOX_CYCLE_HEADER = "MAILBOX-CYCLE: "
MAILBOX_COMMIT_HEADER = "MAILBOX-COMMIT: "
MAILBOX_CANDIDATE_HEADER = "MAILBOX-CANDIDATE: "
MAILBOX_BASE_HEADER = "MAILBOX-BASE: "
MAILBOX_NOTES_COMMIT_HEADER = "MAILBOX-NOTES-COMMIT: "
MAILBOX_RETURN_HEADER = "MAILBOX-RETURN: "
MAILBOX_RESULT_HEADER = "MAILBOX-RESULT: "
MAILBOX_MODE_HEADER = "MAILBOX-MODE: "
MAILBOX_DECISION_HEADER = "MAILBOX-DECISION: "
BACKLOG_CLOSE_REQUIRED_HEADER = "BACKLOG-CLOSE-REQUIRED: "
ARCHITECT_FIX_ONLY_REQUEST = (
    "MAILBOX-MAINTENANCE: existing-bug-fixes\n\n"
    "Select the next Open BUG FIX allowed by watch severity. Exclude "
    "features, discovery, and Low edge cases. Send one plan or no-ticket.\n")
PUBLIC_ARCHITECT_NO_TICKET_RETURN = "architect-no-ticket"
PUBLIC_ARCHITECT_NO_TICKET_DECISION = "NO TICKET"
REDTEAM_REVIEW_RESULTS = ("NO CHANGE", "REOPEN")
CONTROL_PLANE_REVIEW_RESULTS = (
    "ACCEPT-CONTROL-PLANE", "REJECT-CONTROL-PLANE")
TICKET_CLASSES = ("ordinary", "protected-control-plane")
BLOCKED_REDTEAM_DIRECTORY = "blocked-red-team-required"
ARCHITECT_COMMIT_MODES = ("normal", "two-role")
TICKET_ANCHOR_PATTERN = r"[a-z0-9]+(?:-[a-z0-9]+)*"
REDTEAM_REVIEW_TICKET_RE = re.compile(TICKET_ANCHOR_PATTERN)
FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
CYCLE_ID_RE = re.compile(TICKET_ANCHOR_PATTERN + r"@[0-9a-f]{40}")
ARCHITECT_DIRECTIVE_LINE_RE = re.compile(
    r"^- \*\*Directive:\*\* \[(ai/notes/"
    r"[A-Za-z0-9][A-Za-z0-9._-]*\.md),[^\]\n]*"
    r"Implementation directive[^\]\n]*\]$",
    re.MULTILINE)
IMPLEMENTER_CANDIDATE_LINE_RE = re.compile(
    r"^- \*\*Candidate commit:\*\* `?([0-9a-f]{40})`?$",
    re.MULTILINE)
IMPLEMENTER_CHECKPOINT_HEADING = (
    "### IMPLEMENTER_HANDOFF: CHECKPOINT")
CONTEXT_HANDOFF_HEADING = "### IMPLEMENTER_HANDOFF: CONTEXT HANDOFF"
CONTEXT_HANDOFF_FIELDS = (
    "Ticket and cycle",
    "Base commit",
    "Current worktree HEAD",
    "Candidate created",
)
CONTEXT_HANDOFF_SECTIONS = (
    "Completed",
    "Known failures",
    "Rejected approaches",
    "Uncommitted changes",
    "Next exact action",
    "Do not revisit",
)
IMPLEMENTER_CHECKPOINT_CURRENT_STATE = (
    f"- **Current state:** {IMPLEMENTER_REVIEW_MINUTES} minutes reached; "
    "work is paused and may be "
    "stuck.")
IMPLEMENTER_CHECKPOINT_DECISION_PREFIX = "- **Checkpoint decision:**"
FIX_ONLY_ENVIRONMENT = "MAILBOX_FIX_ONLY"
FIX_ONLY_LOCK_NAME = ".fix-only.lock"
SKIP_REDTEAM_ENVIRONMENT = "MAILBOX_SKIP_REDTEAM"
SKIP_REDTEAM_LOCK_NAME = ".skip-redteam.lock"
MAX_CHARACTERS_ENVIRONMENT = "MAILBOX_MAX_CHARACTERS"
DISCOVERY_SEVERITY_ENVIRONMENT = "MAILBOX_DISCOVERY_SEVERITY"
DISCOVERY_SCOPE_ENVIRONMENT = "MAILBOX_DISCOVERY_SCOPE"
MAILBOX_ROLE_ENVIRONMENT = "MAILBOX_ROLE"
IMPLEMENTER_CHECKPOINT_DEADLINE_ENVIRONMENT = (
    "MAILBOX_IMPLEMENTER_CHECKPOINT_DEADLINE")
IMPLEMENTER_CHECKPOINT_STATE_ENVIRONMENT = (
    "MAILBOX_IMPLEMENTER_CHECKPOINT_STATE")

# The demand report shows the size of saved, active Implementer candidates.
# It is read-only: every candidate already has its same-cycle Architect route,
# so a second automatically manufactured audit would be redundant and stale.
LANDING_DEBT_LINE_LIMIT = 400

# One sequence grammar owns both allocation and dispatch-time currency. The
# optional letter is historical (messages such as 0107a); the recipient is
# deliberately unrestricted here because archived -to-user messages and
# hand-made hold directories still claim their sequence numbers.
MESSAGE_SEQUENCE_RE = re.compile(r"(\d+)[a-z]?-to-")
PENDING_MESSAGE_RE = re.compile(r"\d+-to-(fable|opus|sol|daemon)\.md$")
WATCH_LOCK_OWNER_RE = re.compile(r"watch pid [1-9]\d*$")
STATE_GUARD_SUFFIX = ".state-guard"


def backlog_ledger_count():
    """Count every open ticket recorded in the backlog ledger.

    Returns:
      The number of classified and unclassified lines beginning ``- OPEN``.
      Zero is returned when the ledger does not exist.
    """
    counts = backlog_severity_counts()
    return (counts["critical"] + counts["high"] + counts["medium"]
            + counts["low"] + counts["unclassified"])


def backlog_severity_counts():
    """Count open backlog tickets by the Architect's final severity.

    ``Critical`` is a final backlog classification, not a public discovery
    setting. An open line that lacks one exact classification is reported as
    ``unclassified`` so malformed bookkeeping cannot disappear silently.
    Each indexed open ticket must also have one exact Red Team reopen-count
    row in its detailed section. After more than five reopens, only Low is a
    valid severity. The Red Team remains advisory: this bookkeeping rule does
    not wait for a review or block the Architect from closing a ticket.

    Returns:
      A mapping with severity totals, High-bug and feature subtotals, and an
      ``unclassified`` count.
    """
    counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "high_bug_fix": 0,
        "high_new_functionality": 0,
        "unclassified": 0,
        "problem": None,
    }
    lines, problem = verified_backlog_lines()
    if problem is not None:
        counts["problem"] = problem
        return counts
    anchor_counts = {}
    anchor_positions = []
    closed_position = next(
        (index for index, line in enumerate(lines)
         if line == "# Closed tickets"), len(lines))
    for line_number, line in enumerate(lines):
        anchor_match = BACKLOG_DETAIL_ANCHOR_RE.fullmatch(line)
        if anchor_match is not None:
            anchor = anchor_match.group(1)
            anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
            anchor_positions.append((line_number, anchor))
    detail_sections = {}
    for position, (start, anchor) in enumerate(anchor_positions):
        if position + 1 < len(anchor_positions):
            end = anchor_positions[position + 1][0]
        else:
            end = closed_position
        end = min(end, closed_position)
        detail_sections.setdefault(anchor, []).append(lines[start + 1:end])
    seen_index_anchors = set()
    for line in lines:
        if OPEN_BACKLOG_CANDIDATE_RE.match(line) is None:
            continue
        match = OPEN_BACKLOG_TICKET_RE.fullmatch(line)
        if match is None:
            counts["unclassified"] += 1
            continue
        anchor = match.group(4)
        if (match.group(1) == "CRITICAL"
                and match.group(2) != "BUG FIX"):
            counts["unclassified"] += 1
            continue
        if anchor in seen_index_anchors or anchor_counts.get(anchor) != 1:
            counts["unclassified"] += 1
            seen_index_anchors.add(anchor)
            continue
        seen_index_anchors.add(anchor)
        severity = match.group(1).lower()
        sections = detail_sections.get(anchor, [])
        if len(sections) != 1:
            counts["unclassified"] += 1
            continue
        reopen_candidates = [
            detail_line for detail_line in sections[0]
            if BACKLOG_REOPEN_COUNT_CANDIDATE_RE.search(detail_line)
            is not None]
        if len(reopen_candidates) != 1:
            counts["unclassified"] += 1
            continue
        reopen_match = BACKLOG_REOPEN_COUNT_RE.fullmatch(
            reopen_candidates[0])
        if reopen_match is None:
            counts["unclassified"] += 1
            continue
        reopening_candidates = [
            detail_line for detail_line in sections[0]
            if BACKLOG_REOPENING_CANDIDATE_RE.search(detail_line)
            is not None]
        if (len(reopening_candidates) != 1
                or BACKLOG_REOPENING_RE.fullmatch(
                    reopening_candidates[0]) is None):
            counts["unclassified"] += 1
            continue
        if "barred by Architect NO-GO" in reopening_candidates[0]:
            # A barred ticket is final and therefore may not be indexed Open.
            counts["unclassified"] += 1
            continue
        reopen_count = reopen_match.group(1)
        reopened_more_than_five = (
            len(reopen_count) > 1 or reopen_count > "5")
        if reopened_more_than_five and severity != "low":
            counts["unclassified"] += 1
            continue
        ticket_type = match.group(2).lower().replace(" ", "_")
        counts[severity] += 1
        if severity == "high":
            counts["high_" + ticket_type] += 1
    return counts


def verified_backlog_lines():
    """Read one stable, regular UTF-8 backlog or return a plain problem."""
    try:
        initial = os.lstat(BACKLOG_LEDGER)
    except FileNotFoundError:
        return None, (
            "ai/notes/backlog.md is missing; restore the tracked file from "
            "the current main branch before ticket dispatch")
    except OSError as exc:
        return None, "cannot inspect ai/notes/backlog.md: " + str(exc)
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        return None, (
            "ai/notes/backlog.md must be one ordinary file, not a redirect "
            "or special file")
    if initial.st_size > MAX_BACKLOG_LEDGER_BYTES:
        return None, "ai/notes/backlog.md exceeds the safe read limit"
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(BACKLOG_LEDGER, flags)
    except OSError as exc:
        return None, "cannot open ai/notes/backlog.md: " + str(exc)
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode)
                or (initial.st_dev, initial.st_ino)
                != (opened.st_dev, opened.st_ino)):
            return None, "ai/notes/backlog.md changed while being opened"
        chunks = []
        size = 0
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BACKLOG_LEDGER_BYTES:
                return None, "ai/notes/backlog.md exceeds the safe read limit"
            chunks.append(chunk)
        after = os.fstat(descriptor)
        current = os.lstat(BACKLOG_LEDGER)
        if ((opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino)
                != (current.st_dev, current.st_ino)
                or opened.st_size != after.st_size
                or opened.st_mtime_ns != after.st_mtime_ns
                or opened.st_ctime_ns != after.st_ctime_ns
                or after.st_size != size):
            return None, "ai/notes/backlog.md changed while being read"
    except OSError as exc:
        return None, "cannot verify ai/notes/backlog.md: " + str(exc)
    finally:
        os.close(descriptor)
    try:
        text = b"".join(chunks).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return None, "ai/notes/backlog.md is not valid UTF-8: " + str(exc)
    return text.splitlines(), None


def backlog_reopening_status(ticket_anchor):
    """Return ``allowed`` or the Architect's permanent ``barred`` decision.

    ``ticket_anchor`` is the part of a cycle identifier before ``@``. A
    missing, duplicate, or malformed record returns ``None`` so callers do not
    invent permission from incomplete backlog prose.
    """
    if (not isinstance(ticket_anchor, str)
            or REDTEAM_REVIEW_TICKET_RE.fullmatch(ticket_anchor) is None):
        return None
    lines, problem = verified_backlog_lines()
    if problem is not None:
        return None
    starts = [index for index, line in enumerate(lines)
              if line == '<a id="' + ticket_anchor + '"></a>']
    if len(starts) != 1:
        return None
    start = starts[0] + 1
    end = next((index for index in range(start, len(lines))
                if BACKLOG_DETAIL_ANCHOR_RE.fullmatch(lines[index])
                is not None or lines[index] == "# Closed tickets"),
               len(lines))
    candidates = [line for line in lines[start:end]
                  if BACKLOG_REOPENING_CANDIDATE_RE.search(line) is not None]
    if len(candidates) != 1:
        return None
    match = BACKLOG_REOPENING_RE.fullmatch(candidates[0])
    return match.group(1) if match is not None else None


def discovery_admission_count():
    """Return open Critical, High, and Medium tickets; Low does not count."""
    counts = backlog_severity_counts()
    return counts["critical"] + counts["high"] + counts["medium"]


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
    "<seq>-to-<fable|opus|sol|daemon>.md using the next sequence number, INSIDE\n"
    "THIS EXACT DIRECTORY (your cwd may differ -- a relative ai/notes/mailbox\n"
    "path is wrong unless it resolves here):\n"
    "    " + MAILBOX + "\n"
    "Every Architect/Implementer exchange for one ticket starts with:\n"
    "    MAILBOX-FLOW: ticket\n"
    "    MAILBOX-CYCLE: TICKET-ANCHOR@FULL-STARTING-COMMIT\n"
    "    MAILBOX-MODE: normal|two-role\n"
    "Keep that exact cycle and mode through every return and re-handoff.\n"
    "After Architect GO, the Architect writes one decision-only terminal\n"
    "<seq>-to-daemon.md request containing only:\n"
    "    MAILBOX-RETURN: architect-go\n"
    "    MAILBOX-CYCLE: THE-SAME-CYCLE\n"
    "    MAILBOX-CANDIDATE: MAILBOX_CANDIDATE_COMMIT\n"
    "    MAILBOX-MODE: normal|two-role\n"
    "    MAILBOX-DECISION: GO\n"
    "The daemon, after this Architect process exits, prepares and verifies\n"
    "the exact squash landing. The Architect must not merge, commit, update\n"
    "main, target the user's checkout, or push.\n"
    "Normal mode then sends one Red Team closure. Two-role mode completes\n"
    "one cycle when the daemon records local landing L because that watch\n"
    "has no Red Team pass for the ticket. One cycle always belongs to one\n"
    "ticket.\n"
    "Every work outbound addressed to Sol must start with exactly one of\n"
    "these classification lines:\n"
    "    MAILBOX-TICKET: closure\n"
    "    MAILBOX-TICKET: discovery\n"
    "    MAILBOX-TICKET: policy\n"
    "    MAILBOX-TICKET: control-plane\n"
    "Use closure only for work that retires an existing - OPEN ledger line;\n"
    "use discovery when the product is new findings. The daemon refuses to\n"
    "guess a class from prose. A discovery must add these exact second and\n"
    "third lines, in this order, using the binding values exported to the\n"
    "turn:\n"
    "    MAILBOX-SEVERITY: LEVEL\n"
    "Replace LEVEL with exactly high, medium, or low.\n"
    "    MAILBOX-SCOPE: SCOPE\n"
    "Replace SCOPE with exactly bounded or widespread.\n"
    "A normal review of daemon-recorded landing L instead adds these exact\n"
    "second and third lines, then one blank line and the handoff:\n"
    "    MAILBOX-CYCLE: THE-SAME-CYCLE\n"
    "    MAILBOX-COMMIT: FULL-DAEMON-LANDING-COMMIT\n"
    "The Red Team return to Fable must begin with exact return, cycle, commit,\n"
    "and result headers. Result is NO CHANGE or REOPEN, never GO/NO-GO. A\n"
    "matching return completes the cycle. Another ticket may start only when\n"
    "the current watch still has an unused cycle slot.\n"
    "Control-plane is the mandatory pre-landing review of protected candidate\n"
    "C. It names MAILBOX-CYCLE and MAILBOX-CANDIDATE. Return to daemon with\n"
    "MAILBOX-RETURN: redteam-control-plane, those same two identities, and\n"
    "MAILBOX-RESULT: ACCEPT-CONTROL-PLANE or REJECT-CONTROL-PLANE. This is\n"
    "the sole exception to ordinary advisory post-landing review: both exact\n"
    "keys are required, but D0 alone validates, creates L, and advances main.\n"
    "Policy is the cycle-free, one-pass adversarial review of an Architect's\n"
    "exact draft change to a protected role or permanent note. Return one\n"
    "advisory answer. The Architect then makes the final decision; do not\n"
    "begin a second review round.\n"
    "A public request to the Architect uses those same severity and scope\n"
    "lines as its first two lines. The daemon validates and exports both\n"
    "values to the Architect turn; they are not a Sol ticket classification.\n"
    "That saved value records the user's minimum severity for a new ticket.\n"
    "The daemon's exact no-work transport ping is\n"
    "the sole reserved MAILBOX-TICKET: transport exception.\n"
    "Narrow exception: if and only if the inbound's binding instruction\n"
    "explicitly says the thread is TERMINAL and no reply is owed, write no\n"
    "outbound merely to satisfy this wrapper. Ambiguity follows the ordinary\n"
    "rule: record the substance and write the outbound.\n"
    "Git landing authority belongs to the parent daemon after the Architect\n"
    "process exits. No AI role may merge, commit, update a ref, change the\n"
    "user's checkout, or push as part of landing a ticket.\n\n"
    "--- MESSAGE ---\n")

CHECKPOINT_LANDING_START = "After Architect GO,"
CHECKPOINT_LANDING_END = "Every work outbound addressed to Sol"
CHECKPOINT_PREAMBLE = (
    PREAMBLE[:PREAMBLE.index(CHECKPOINT_LANDING_START)]
    + PREAMBLE[PREAMBLE.index(CHECKPOINT_LANDING_END):])


def common_preamble_for_dispatch(checkpoint_audit):
    """Omit ordinary landing instructions during a checkpoint review."""
    return CHECKPOINT_PREAMBLE if checkpoint_audit else PREAMBLE

ARCHITECT_LANDING_PREAMBLE = (
    "ARCHITECT DECISION GRANT:\n"
    "When this audit records GO, write the exact architect-go request named\n"
    "by the common wrapper. Bind it to MAILBOX_CANDIDATE_COMMIT. The parent\n"
    "daemon performs the squash landing only after this process exits. Never\n"
    "run merge, commit, update-ref, reset, checkout, or push for the landing,\n"
    "and never target the user's repository checkout. This decision grant\n"
    "belongs only to the Architect lane.\n\n")

ARCHITECT_ROLE_PREAMBLE = (
    "ROUTE ROLE: You are the Architect / Auditor. Read and obey\n"
    ".claude/FABLE_ROLE.md before acting. You own design reasoning. Before\n"
    "sending work to an Implementer, write the complete Implementation\n"
    "directive in the cited note and run ai/tools/handoff_contract.py; a\n"
    "goal summary or unresolved design choice is not dispatchable. Use the\n"
    "exact handoff row: - **Directive:** [ai/notes/<name>.md, exact "
    "Implementation directive section]. Before ending, re-read the outgoing\n"
    "file and use the daemon parsers to require its exact envelope, admission,\n"
    "one Directive row, and no placeholders. Fix it in the same turn; note\n"
    "validation alone does not validate the mailbox file.\n\n")

IMPLEMENTER_ROLE_PREAMBLE = (
    "ROUTE ROLE: You are the Implementer. Read and obey\n"
    ".claude/OPUS_ROLE.md before acting. Run the cited Architect directive\n"
    "check before editing. Follow the ordered plan; if design is missing or\n"
    "contradictory, return a blocker instead of making that decision.\n\n")

REDTEAM_ROLE_PREAMBLE = (
    "ROUTE ROLE: You are the bounded Red Team. Read and obey the exact\n"
    "authoritative role file named below before acting. Sol is advisory and\n"
    "never implements a ticket. A confirmed finding must include a validated,\n"
    "implementation-ready Repair directive, but it returns to the Architect\n"
    "as candidate input and never executes itself.\n\n")

def agent_preamble(agent, message=None):
    """Return role-specific standing text that precedes the common wrapper."""
    if agent == "fable":
        checkpoint_notice = ""
        checkpoint_audit = False
        reopen_turn = False
        if message is not None and message.startswith(MAILBOX_RETURN_HEADER):
            _cycle_id, _, result, _, problem = _redteam_review_receipt(
                message=message)
            reopen_turn = problem is None and result == "REOPEN"
        if message is not None and message.startswith(MAILBOX_FLOW_HEADER):
            cycle_id, _mode, body, problem = _ticket_flow_envelope(
                message=message)
            context_handoff = (
                problem is None and is_implementer_context_handoff(body))
            checkpoint_audit = (
                problem is None and is_implementer_checkpoint_request(body))
            if context_handoff:
                checkpoint_notice = (
                    "IMPLEMENTER CONTEXT HANDOFF: this is the prior "
                    "Implementer's exact record, not a candidate or a "
                    "completed ticket. Check the stated repository state. "
                    "Send one same-cycle replacement handoff with exactly "
                    "one **Checkpoint decision:** `GO` or `NO-GO` row. GO "
                    "continues from this record; NO-GO parks or revises the "
                    "work. Do not rewrite the record as a summary.\n\n")
            elif checkpoint_audit:
                checkpoint_notice = (
                    "90-MINUTE IMPLEMENTER CHECKPOINT: inspect the saved "
                    "candidate, then send one revised same-cycle handoff "
                    "to the Implementer with exactly one **Checkpoint "
                    "decision:** `GO` or `NO-GO` row. This turn cannot land "
                    "the checkpoint candidate.\n\n")
            elif (problem is None
                    and "### IMPLEMENTER_HANDOFF:" in body
                    and "- Acceptance: `blocked`" in body):
                checkpoint_notice = (
                    "BLOCKED IMPLEMENTER CHECKPOINT: this return is not a "
                    "candidate and cannot receive GO. If the actual runtime "
                    "cannot launch subagents, a revised capability exception "
                    "must bind the source note to these exact rows:\n"
                    "- Source cycle: `" + cycle_id + "`\n"
                    "- Source handoff SHA-256: `"
                    + hashlib.sha256(body.encode("utf-8")).hexdigest()
                    + "`\n\n")
        landing = ("" if checkpoint_audit or reopen_turn
                   else ARCHITECT_LANDING_PREAMBLE)
        return checkpoint_notice + ARCHITECT_ROLE_PREAMBLE + landing
    if agent == "opus":
        return (IMPLEMENTER_ROLE_PREAMBLE
                + "AUTHORITATIVE IMPLEMENTER ROLE FILE:\n    "
                + os.path.join(AGENT_CWD["fable"], ".claude",
                               "OPUS_ROLE.md")
                + "\nRead this primary copy instead of a possibly older "
                "copy in the candidate checkout.\n\n")
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


def ticket_cycle_mode_is_enabled(mode, skip_redteam=False):
    """Return whether this watch topology owns ``mode`` work."""
    if skip_redteam:
        return mode == "two-role"
    return mode == "normal"


def canonical_ticket_cycle_topology(skip_redteam=False):
    """Return the single durable identity of this watch's role layout."""
    if skip_redteam:
        return "two-role"
    return "normal"


def message_is_enabled_for_topology(path, skip_redteam=False):
    """Return whether this watch may consume one root mailbox message.

    Messages for a disabled role remain byte-for-byte in the mailbox root.
    Malformed messages stay enabled so the ordinary dispatcher can explain
    and quarantine them instead of silently treating corruption as a role.
    """
    match = PENDING_MESSAGE_RE.match(os.path.basename(path))
    if match is None:
        return False
    agent = match.group(1)
    try:
        message = read_cycle_message(path=path)
    except (OSError, ValueError, TicketCycleStateError):
        return True
    if agent == "daemon":
        if message.startswith(
                MAILBOX_RETURN_HEADER + "redteam-control-plane"):
            return not skip_redteam
        _, _, mode, problem = _architect_go_request(message=message)
        return (True if problem is not None else
                ticket_cycle_mode_is_enabled(
                    mode=mode, skip_redteam=skip_redteam))
    if agent == "sol":
        return not skip_redteam
    if agent == "fable" and message.startswith(MAILBOX_RETURN_HEADER):
        return not skip_redteam
    if message.startswith(MAILBOX_FLOW_HEADER):
        _, mode, _, problem = _ticket_flow_envelope(message=message)
        return (True if problem is not None else
                ticket_cycle_mode_is_enabled(
                    mode=mode, skip_redteam=skip_redteam))
    return True


def enabled_pending_messages(skip_redteam=False):
    """Return root messages eligible for this watch topology.

    The ordinary three-role topology returns every dispatchable message.
    A two-role watch excludes only exact ``to-sol`` roots; those files stay
    in place for a later Sol-enabled watch.
    """
    return [
        path for path in pending_messages()
        if message_is_enabled_for_topology(
            path=path, skip_redteam=skip_redteam)]


def deferred_sol_messages():
    """Return exact pending Sol roots held by a two-role watch."""
    return [path for path in pending_messages()
            if PENDING_MESSAGE_RE.match(os.path.basename(path)).group(1)
            == "sol"]


def fable_message_inode_snapshot():
    """Return regular inodes for every existing Architect-addressed message."""
    snapshot = set()
    if not os.path.isdir(MAILBOX):
        return snapshot
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is not None:
            snapshot.add(inode)
    return snapshot


def daemon_message_inode_snapshot():
    """Return regular inodes for every existing daemon-addressed message."""
    snapshot = set()
    if not os.path.isdir(MAILBOX):
        return snapshot
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-daemon.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is not None:
            snapshot.add(inode)
    return snapshot


def opus_message_inode_snapshot():
    """Return regular inodes for every existing Implementer message."""
    snapshot = set()
    if not os.path.isdir(MAILBOX):
        return snapshot
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-opus.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is not None:
            snapshot.add(inode)
    return snapshot


def sol_message_inode_snapshot():
    """Return regular inodes for every existing Red Team message."""
    snapshot = set()
    if not os.path.isdir(MAILBOX):
        return snapshot
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-sol.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is not None:
            snapshot.add(inode)
    return snapshot


def user_message_inode_snapshot():
    """Return regular inodes for every existing human-addressed message."""
    snapshot = set()
    if not os.path.isdir(MAILBOX):
        return snapshot
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-user.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is not None:
            snapshot.add(inode)
    return snapshot


def matching_new_architect_go(cycle_id, candidate_commit, mode,
                               before_inodes):
    """Prove any GO created by this Architect turn names its exact C."""
    fresh = []
    problems = []
    problem_paths = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-daemon.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is None or inode in before_inodes:
            continue
        try:
            raw = stable_regular_bytes(
                path=path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="Architect GO " + os.path.basename(path))
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            problems.append(os.path.basename(path) + ": " + str(exc))
            problem_paths.append(path)
            continue
        returned_cycle, returned_candidate, returned_mode, problem = (
            _architect_go_request(message=message))
        if problem is not None:
            problems.append(os.path.basename(path) + ": " + problem)
            problem_paths.append(path)
            continue
        if (returned_cycle != cycle_id
                or returned_candidate != candidate_commit
                or returned_mode != mode):
            problems.append(
                os.path.basename(path)
                + ": GO does not name this turn's exact cycle, candidate, "
                "and mode")
            problem_paths.append(path)
            continue
        fresh.append(path)
    if problems:
        return (None, list(dict.fromkeys(problem_paths + fresh)),
                "; ".join(problems))
    if len(fresh) > 1:
        return (None, fresh,
                "expected at most one new exact Architect GO; found "
                + str(len(fresh)))
    return (fresh[0] if fresh else None), [], None


def architect_handoff_problem(message, cycle_id, mode, checkpoint=False):
    """Return why one same-cycle Architect repair is not authorized."""
    returned_cycle, returned_mode, body, problem = (
        _ticket_flow_envelope(message=message))
    if problem is not None:
        return problem
    if returned_cycle != cycle_id:
        return "Architect handoff changed MAILBOX-CYCLE"
    if returned_mode != mode:
        return "Architect handoff changed MAILBOX-MODE"
    if not checkpoint:
        if len(ARCHITECT_DIRECTIVE_LINE_RE.findall(message)) != 1:
            return "repair handoff requires exactly one Directive row"
        return None
    decision_rows = [
        line for line in body.splitlines()
        if line.startswith(IMPLEMENTER_CHECKPOINT_DECISION_PREFIX)]
    accepted_rows = {
        IMPLEMENTER_CHECKPOINT_DECISION_PREFIX + " `GO`",
        IMPLEMENTER_CHECKPOINT_DECISION_PREFIX + " `NO-GO`",
    }
    if len(decision_rows) != 1 or decision_rows[0] not in accepted_rows:
        return "checkpoint handoff requires exactly one GO or NO-GO row"
    return None


def checkpoint_architect_handoff_problem(message, cycle_id, mode):
    """Compatibility name for the stricter checkpoint form."""
    return architect_handoff_problem(
        message=message, cycle_id=cycle_id, mode=mode, checkpoint=True)


def matching_new_architect_handoff(cycle_id, mode, before_inodes,
                                   checkpoint=False, required=True):
    """Find one fresh same-cycle repair handoff from the Architect."""
    fresh = []
    invalid = []
    problems = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-opus.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is None or inode in before_inodes:
            continue
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError) as exc:
            invalid.append(path)
            problems.append(os.path.basename(path) + ": " + str(exc))
            continue
        problem = architect_handoff_problem(
            message=message, cycle_id=cycle_id, mode=mode,
            checkpoint=checkpoint)
        if os.path.dirname(path) != MAILBOX:
            problem = "handoff was not published in the mailbox root"
        if problem is not None:
            invalid.append(path)
            problems.append(os.path.basename(path) + ": " + problem)
        else:
            fresh.append(path)
    if problems:
        return None, list(dict.fromkeys(invalid + fresh)), "; ".join(problems)
    if len(fresh) > 1 or (required and len(fresh) != 1):
        return (None, fresh,
                "expected exactly one new Architect handoff to the "
                "Implementer; found " + str(len(fresh)))
    return (fresh[0] if fresh else None), [], None


def matching_new_checkpoint_handoff(cycle_id, mode, before_inodes):
    """Compatibility name for a required checkpoint decision."""
    return matching_new_architect_handoff(
        cycle_id=cycle_id, mode=mode, before_inodes=before_inodes,
        checkpoint=True, required=True)


def matching_new_architect_notes_go(base_commit, notes_commit,
                                     before_inodes):
    """Prove exactly one fresh note-only GO binds this Fable turn's B and P."""
    fresh = []
    invalid = []
    problems = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-daemon.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is None or inode in before_inodes:
            continue
        try:
            raw = stable_regular_bytes(
                path=path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="Architect notes GO " + os.path.basename(path))
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            invalid.append(path)
            problems.append(os.path.basename(path) + ": " + str(exc))
            continue
        returned_base, returned_notes, problem = (
            _architect_notes_go_request(message=message))
        if problem is not None:
            invalid.append(path)
            problems.append(os.path.basename(path) + ": " + problem)
            continue
        if (returned_base != base_commit
                or returned_notes != notes_commit):
            invalid.append(path)
            problems.append(
                os.path.basename(path)
                + ": notes GO does not bind this turn's exact B and P")
            continue
        fresh.append(path)
    if problems:
        return None, invalid, "; ".join(problems)
    if len(fresh) != 1:
        return (None, fresh,
                "a permanent-note commit requires exactly one fresh "
                "architect-notes-go request; found " + str(len(fresh)))
    return fresh[0], [], None


def _authoritative_handoff_contract_module():
    """Load the already-proved primary contract without importing a copy."""
    path = os.path.join(
        AGENT_CWD["fable"], "ai", "tools", "handoff_contract.py")
    try:
        spec = importlib.util.spec_from_file_location(
            "_mailbox_authoritative_handoff_contract", path)
        if spec is None or spec.loader is None:
            raise ImportError("no Python loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError, SyntaxError) as exc:
        raise TicketCycleStateError(
            "cannot load authoritative handoff contract: " + str(exc)) \
            from exc
    required = (
        "DirectiveError", "validate_directive_file",
        "extract_implementer_subagent_evidence",
        "extract_blocked_implementer_capability_evidence",
        "validate_implementer_handoff_subagent_evidence")
    if any(not hasattr(module, name) for name in required):
        raise TicketCycleStateError(
            "authoritative handoff contract lacks Implementer evidence API")
    return module


def prove_blocked_implementer_checkpoint(cycle_id, handoff_sha256,
                                         contract):
    """Bind a capability retry to one actual prior blocked return."""
    matches = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        try:
            raw = stable_regular_bytes(
                path=path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="blocked Implementer checkpoint")
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError):
            continue
        if not message.startswith(MAILBOX_FLOW_HEADER):
            continue
        returned_cycle, _mode, body, problem = _ticket_flow_envelope(
            message=message)
        if problem is not None or returned_cycle != cycle_id:
            continue
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != handoff_sha256:
            continue
        try:
            evidence = (
                contract.extract_blocked_implementer_capability_evidence(
                handoff_text=body)
            )
        except contract.DirectiveError:
            continue
        matches.append(evidence)
    if len(matches) != 1:
        raise TicketCycleStateError(
            "capability exception is not bound to exactly one actual "
            "blocked IMPLEMENTER_HANDOFF in this MAILBOX-CYCLE")
    return matches[0]


def prepare_implementer_evidence_contract(message, use_saved_limit=False):
    """Freeze the Architect's parsed subagent plan before Opus launches."""
    matches = ARCHITECT_DIRECTIVE_LINE_RE.findall(message)
    if len(matches) != 1:
        raise TicketCycleStateError(
            "Architect handoff must cite exactly one ai/notes source note "
            "and its Implementation directive")
    relative = matches[0]
    note_path = os.path.abspath(os.path.join(AGENT_CWD["fable"], relative))
    notes_root = os.path.abspath(os.path.join(AGENT_CWD["fable"], "ai", "notes"))
    if (os.path.commonpath((note_path, notes_root)) != notes_root
            or os.path.realpath(note_path) != note_path):
        raise TicketCycleStateError(
            "Architect handoff cites a redirected or external source note")
    contract = _authoritative_handoff_contract_module()
    try:
        directive = contract.validate_directive_file(
            role="architect", path=note_path,
            expected_max=(None if use_saved_limit else MAX_CHARACTERS))
    except contract.DirectiveError as exc:
        raise TicketCycleStateError(
            "Architect source directive is invalid: " + str(exc)) from exc
    plan = directive.get("parallel_work_plan")
    if not isinstance(plan, dict):
        raise TicketCycleStateError(
            "Architect source directive has no parsed Parallel work plan")
    if plan.get("mode") == "capability-unavailable":
        cycle_id, _mode, _body, problem = _ticket_flow_envelope(
            message=message)
        checkpoint = directive.get("capability_checkpoint")
        if (problem is not None or not isinstance(checkpoint, dict)
                or checkpoint.get("cycle") != cycle_id):
            raise TicketCycleStateError(
                "capability exception checkpoint does not name the current "
                "MAILBOX-CYCLE")
        prior_failure = prove_blocked_implementer_checkpoint(
            cycle_id=cycle_id,
            handoff_sha256=checkpoint.get("handoff_sha256", ""),
            contract=contract)
        for field in (
                "capability_checked", "attempted_operation", "raw_failure"):
            if prior_failure.get(field) != plan.get(field):
                raise TicketCycleStateError(
                    "capability exception field '" + field
                    + "' does not exactly match the digest-bound blocked "
                    "IMPLEMENTER_HANDOFF")
    role_plan = directive.get("role_plan")
    if (not isinstance(role_plan, dict)
            or role_plan.get("ticket_class") not in TICKET_CLASSES):
        raise TicketCycleStateError(
            "Architect source directive has no validated Ticket class")
    return {"contract": contract, "parallel_work_plan": plan,
            "note_path": note_path,
            "allowed_paths": frozenset(directive["allowed_paths"]),
            "ticket_class": role_plan["ticket_class"]}


def matching_new_implementer_handoff(cycle_id, mode, candidate_commit,
                                     before_inodes, evidence_contract):
    """Prove one same-cycle Opus return and its exact subagent evidence."""
    matches = []
    malformed = []
    malformed_paths = []
    evidence_results = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is None or inode in before_inodes:
            continue
        try:
            raw = stable_regular_bytes(
                path=path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="Implementer return " + os.path.basename(path))
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            malformed.append(os.path.basename(path) + ": " + str(exc))
            malformed_paths.append(path)
            continue
        if not message.startswith(MAILBOX_FLOW_HEADER):
            continue
        returned_cycle, returned_mode, body, problem = (
            _ticket_flow_envelope(message=message))
        if problem is not None:
            malformed.append(os.path.basename(path) + ": " + problem)
            malformed_paths.append(path)
            continue
        if returned_cycle != cycle_id:
            continue
        if returned_mode != mode:
            malformed.append(
                os.path.basename(path) + ": returned mode changed")
            malformed_paths.append(path)
            continue
        candidate_lines = IMPLEMENTER_CANDIDATE_LINE_RE.findall(body)
        if candidate_lines != [candidate_commit]:
            malformed.append(
                os.path.basename(path)
                + ": Candidate commit does not name the exact Opus HEAD")
            malformed_paths.append(path)
            continue
        try:
            evidence_result = evidence_contract["contract"].\
                validate_implementer_handoff_subagent_evidence(
                    parallel_work_plan=(
                        evidence_contract["parallel_work_plan"]),
                    handoff_text=body)
        except evidence_contract["contract"].DirectiveError as exc:
            malformed.append(os.path.basename(path) + ": " + str(exc))
            malformed_paths.append(path)
            continue
        matches.append(path)
        evidence_results.append(evidence_result)
    if malformed:
        return None, malformed_paths, "; ".join(malformed), None
    if len(matches) != 1:
        return (None, [],
                "expected exactly one new same-cycle IMPLEMENTER_HANDOFF; "
                "found " + str(len(matches)), None)
    return (matches[0], [], None,
            bool(evidence_results[0].get("completion_ready")))


def matching_new_redteam_receipt(cycle_id, accepted_commit, before_inodes):
    """Return one new correlated Red Team receipt path and result.

    The scan spans mailbox states so a future refactor that consumes the
    Architect lane concurrently cannot turn a real return into a false
    missing-receipt failure.
    """
    matches = []
    malformed = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is None or inode in before_inodes:
            continue
        try:
            raw = stable_regular_bytes(
                path=path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="Red Team return " + os.path.basename(path))
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            malformed.append(os.path.basename(path) + ": " + str(exc))
            continue
        if not message.startswith(MAILBOX_RETURN_HEADER):
            continue
        returned_cycle, returned_commit, result, _, problem = (
            _redteam_review_receipt(message=message))
        if problem is not None:
            malformed.append(os.path.basename(path) + ": " + problem)
            continue
        if returned_cycle == cycle_id and returned_commit == accepted_commit:
            matches.append((path, result))
    if malformed:
        return None, None, "; ".join(malformed)
    if len(matches) != 1:
        return (None, None,
                "expected exactly one new matching Red Team return; found "
                + str(len(matches)))
    return matches[0][0], matches[0][1], None


def matching_new_control_plane_receipt(cycle_id, candidate,
                                       before_inodes):
    """Prove one new exact Red Team key addressed to D0."""
    matches = []
    malformed = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-daemon.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is None or inode in before_inodes:
            continue
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError) as exc:
            malformed.append(os.path.basename(path) + ": " + str(exc))
            continue
        if not message.startswith(
                MAILBOX_RETURN_HEADER + "redteam-control-plane"):
            continue
        found_cycle, found_candidate, result, _body, problem = (
            _control_plane_review_receipt(message=message))
        if problem is not None:
            malformed.append(os.path.basename(path) + ": " + problem)
            continue
        if found_cycle == cycle_id and found_candidate == candidate:
            matches.append((path, result))
    if malformed:
        return None, None, "; ".join(malformed)
    if len(matches) != 1:
        return (None, None,
                "expected exactly one new exact control-plane return; found "
                + str(len(matches)))
    return matches[0][0], matches[0][1], None


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


def _sol_discovery_envelope(message):
    """Parse the exact persisted discovery envelope without reading prose.

    Returns:
      ``(severity, scope, body, problem)``. A valid discovery has all three
      exact physical header lines in order. Other ticket kinds may not use a
      reserved severity or scope line.
    """
    ticket_kind = sol_ticket_kind(message=message)
    remainder = sol_ticket_body_after_kind(message=message)
    severity_like_line = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*severity[ \t]*:")
    scope_like_line = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*scope[ \t]*:")
    if ticket_kind != "discovery":
        if re.search(severity_like_line, remainder) is not None:
            return (None, None, remainder,
                    "MAILBOX-SEVERITY is reserved for discovery tickets "
                    "and must not appear on another ticket kind")
        if re.search(scope_like_line, remainder) is not None:
            return (None, None, remainder,
                    "MAILBOX-SCOPE is reserved for discovery tickets and "
                    "must not appear on another ticket kind")
        return None, None, remainder, None

    severity_match = re.match(
        r"\A" + re.escape(SOL_SEVERITY_HEADER)
        + r"(" + "|".join(map(re.escape, DISCOVERY_SEVERITIES))
        + r")\r?\n",
        remainder)
    if severity_match is None:
        return (None, None, remainder,
                "a discovery ticket requires exactly "
                "'MAILBOX-SEVERITY: high', 'MAILBOX-SEVERITY: medium', or "
                "'MAILBOX-SEVERITY: low' as its second physical line")

    after_severity = remainder[severity_match.end():]
    scope_match = re.match(
        r"\A" + re.escape(SOL_SCOPE_HEADER)
        + r"(" + "|".join(map(re.escape, DISCOVERY_SCOPES))
        + r")(?:\r?\n|\Z)",
        after_severity)
    if scope_match is None:
        return (None, None, remainder,
                "a discovery ticket requires exactly "
                "'MAILBOX-SCOPE: bounded' or 'MAILBOX-SCOPE: widespread' "
                "as its third physical line")

    body = after_severity[scope_match.end():]
    if re.search(severity_like_line, body) is not None:
        return None, None, remainder, "duplicate MAILBOX-SEVERITY line"
    if re.search(scope_like_line, body) is not None:
        return None, None, remainder, "duplicate MAILBOX-SCOPE line"
    return severity_match.group(1), scope_match.group(1), body, None


def sol_discovery_severity_problem(message):
    """Return an exact discovery-envelope error, or ``None``."""
    return _sol_discovery_envelope(message=message)[3]


def sol_discovery_severity(message):
    """Return a valid discovery ticket's saved severity, or ``None``."""
    severity, _, _, problem = _sol_discovery_envelope(message=message)
    return severity if problem is None else None


def sol_discovery_scope(message):
    """Return a valid discovery ticket's saved scope, or ``None``."""
    _, scope, _, problem = _sol_discovery_envelope(message=message)
    return scope if problem is None else None


def public_architect_sol_downstream_problem(message):
    """Validate one non-cycle Architect request addressed to Red Team."""
    if sol_ticket_kind(message=message) != "discovery":
        return ("public Architect control output to Sol must be an exact "
                "discovery request")
    _severity, _scope, body, problem = _sol_discovery_envelope(
        message=message)
    if problem is not None:
        return problem
    marker = placeholder_in(message=body)
    if marker is not None:
        return "Sol discovery body is only template placeholder '" + marker + "'"
    if "\x00" in message:
        return "Sol discovery contains a NUL byte"
    return None


def _body_architect_admission(body):
    """Return the exact admission token on the first body line.

    A public Architect outcome must be mechanically tied to the request that
    occupied the finite-watch slot.  Looking for a token later in prose would
    allow an unrelated output to consume that slot, so the binding line is
    required first and duplicates are refused.
    """
    match = re.match(
        r"\A" + re.escape(MAILBOX_ADMISSION_HEADER)
        + r"(\d+-to-fable\.md@[0-9a-f]{64})\r?\n", body)
    admission_like = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*admission[ \t]*:")
    if match is None:
        if re.search(admission_like, body) is not None:
            return None, "malformed MAILBOX-ADMISSION line"
        return None, "missing first-body-line MAILBOX-ADMISSION"
    if re.search(admission_like, body[match.end():]) is not None:
        return None, "duplicate MAILBOX-ADMISSION line"
    try:
        split_architect_admission_token(token=match.group(1))
    except TicketCycleStateError as exc:
        return None, str(exc)
    return match.group(1), None


def public_architect_sol_outcome_problem(message, expected_token):
    """Validate one exact, digest-bound public Architect-to-Sol outcome."""
    problem = public_architect_sol_downstream_problem(message=message)
    if problem is not None:
        return problem
    _severity, _scope, body, _problem = _sol_discovery_envelope(
        message=message)
    if body.startswith("\r\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]
    else:
        return "public Architect Sol outcome requires one header/body gap"
    returned_token, admission_problem = _body_architect_admission(body=body)
    if admission_problem is not None:
        return admission_problem
    if returned_token != expected_token:
        return "MAILBOX-ADMISSION does not bind this public request"
    return None


def _public_architect_no_ticket_receipt(message):
    """Parse one explicit no-ticket result from a public Architect turn."""
    match = re.match(
        r"\A" + re.escape(MAILBOX_RETURN_HEADER)
        + re.escape(PUBLIC_ARCHITECT_NO_TICKET_RETURN) + r"\r?\n"
        + re.escape(MAILBOX_ADMISSION_HEADER)
        + r"(\d+-to-fable\.md@[0-9a-f]{64})\r?\n"
        + re.escape(MAILBOX_DECISION_HEADER)
        + re.escape(PUBLIC_ARCHITECT_NO_TICKET_DECISION)
        + r"(?:\r?\n\r?\n(?P<body>[\s\S]*))?\r?\n?\Z",
        message)
    if match is None:
        return None, (
            "no-ticket receipt needs exact MAILBOX-RETURN, "
            "MAILBOX-ADMISSION, and MAILBOX-DECISION headers")
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*"
        r"(?:return|admission|decision)[ \t]*:")
    body = match.group("body") or ""
    if re.search(reserved, body) is not None:
        return None, "duplicate no-ticket receipt header"
    if "\x00" in message:
        return None, "no-ticket receipt contains a NUL byte"
    marker = placeholder_in(message=body)
    if marker is not None:
        return None, (
            "no-ticket receipt body is only template placeholder '"
            + marker + "'")
    try:
        split_architect_admission_token(token=match.group(1))
    except TicketCycleStateError as exc:
        return None, str(exc)
    return match.group(1), None


def public_architect_no_ticket_problem(message, expected_token):
    """Return why a public no-ticket receipt is invalid, or ``None``."""
    returned_token, problem = _public_architect_no_ticket_receipt(
        message=message)
    if problem is not None:
        return problem
    if returned_token != expected_token:
        return "MAILBOX-ADMISSION does not bind this public request"
    return None


def _ticket_flow_envelope(message):
    """Parse one Architect/Implementer exchange inside a ticket cycle."""
    match = re.match(
        r"\A" + re.escape(MAILBOX_FLOW_HEADER) + r"ticket\r?\n"
        + re.escape(MAILBOX_CYCLE_HEADER)
        + r"(" + CYCLE_ID_RE.pattern + r")\r?\n"
        + re.escape(MAILBOX_MODE_HEADER)
        + r"(" + "|".join(map(re.escape, ARCHITECT_COMMIT_MODES))
        + r")\r?\n\r?\n",
        message)
    if match is None:
        return (None, None, message,
                "a ticket exchange needs exact MAILBOX-FLOW, MAILBOX-CYCLE, "
                "and MAILBOX-MODE headers")
    body = message[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:flow|cycle|mode)[ \t]*:")
    if re.search(reserved, body) is not None:
        return None, None, message, "duplicate ticket-cycle flow header"
    return match.group(1), match.group(2), body, None


def is_implementer_checkpoint_request(body):
    """Return whether a handoff asks the Architect for a pause decision."""
    if not isinstance(body, str):
        return False
    lines = body.splitlines()
    return bool(lines) and lines[0] in {
        IMPLEMENTER_CHECKPOINT_HEADING, CONTEXT_HANDOFF_HEADING}


def is_implementer_time_checkpoint(body):
    """Return whether a timed checkpoint begins with its fixed state."""
    if not is_implementer_checkpoint_request(body):
        return False
    first_field = next(
        (line for line in body.splitlines()[1:] if line), "")
    return first_field == IMPLEMENTER_CHECKPOINT_CURRENT_STATE


def is_implementer_context_handoff(body):
    """Return whether the body begins with the context-handoff heading."""
    return (isinstance(body, str)
            and body.splitlines()[:1] == [CONTEXT_HANDOFF_HEADING])


def _context_handoff_field(lines, name):
    """Read one exact field from a context handoff."""
    prefix = "- **" + name + ":** "
    values = [line[len(prefix):].strip("`") for line in lines
              if line.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise TicketCycleStateError(
            "CONTEXT HANDOFF needs exactly one " + name + " field")
    return values[0]


def parse_context_handoff(body):
    """Read the required facts and ordered lists from one small handoff."""
    if not isinstance(body, str) or len(body.encode("utf-8")) > 32 * 1024:
        raise TicketCycleStateError("CONTEXT HANDOFF is missing or too large")
    lines = body.splitlines()
    if not lines or lines[0] != CONTEXT_HANDOFF_HEADING:
        raise TicketCycleStateError("CONTEXT HANDOFF heading is missing")
    record = {name: _context_handoff_field(lines, name)
              for name in CONTEXT_HANDOFF_FIELDS}
    for name in ("Base commit", "Current worktree HEAD"):
        if FULL_COMMIT_RE.fullmatch(record[name]) is None:
            raise TicketCycleStateError(name + " must be one full Git commit")
    if record["Candidate created"] not in {"yes", "no"}:
        raise TicketCycleStateError("Candidate created must be yes or no")
    headings = ["#### " + name for name in CONTEXT_HANDOFF_SECTIONS]
    if any(lines.count(heading) != 1 for heading in headings):
        raise TicketCycleStateError(
            "CONTEXT HANDOFF needs every required list section once")
    positions = [lines.index(heading) for heading in headings]
    if positions != sorted(positions):
        raise TicketCycleStateError("CONTEXT HANDOFF sections are out of order")
    sections = {}
    for index, name in enumerate(CONTEXT_HANDOFF_SECTIONS):
        start = positions[index] + 1
        end = positions[index + 1] if index + 1 < len(positions) else len(lines)
        rows = [line for line in lines[start:end] if line.strip()]
        values = [line[2:].strip() for line in rows
                  if line.startswith("- ")]
        if (len(values) != len(rows) or not values
                or any(value in {"", "...", "[...]"} for value in values)):
            raise TicketCycleStateError(
                name + " must contain concrete bullets or '- none'")
        sections[name] = values
    record["sections"] = sections
    return record


def context_handoff_problem(message, expected_cycle=None,
                            expected_mode=None):
    """Validate one context record against the current Implementer tree."""
    cycle_id, mode, body, problem = _ticket_flow_envelope(message=message)
    if problem is not None:
        return problem
    if expected_cycle is not None and cycle_id != expected_cycle:
        return "CONTEXT HANDOFF changed MAILBOX-CYCLE"
    if expected_mode is not None and mode != expected_mode:
        return "CONTEXT HANDOFF changed MAILBOX-MODE"
    try:
        record = parse_context_handoff(body=body)
    except TicketCycleStateError as exc:
        return str(exc)
    base = cycle_id.rsplit("@", 1)[1]
    if record["Ticket and cycle"] != cycle_id:
        return "CONTEXT HANDOFF does not name its exact ticket and cycle"
    if record["Base commit"] != base:
        return "CONTEXT HANDOFF does not name the cycle base commit"
    try:
        head = worktree_head(worktree=AGENT_CWD["opus"])
        dirty = bool(_clean_worktree_status(worktree=AGENT_CWD["opus"]))
    except (OSError, PrimaryWorktreeError, TicketCycleStateError) as exc:
        return "cannot verify CONTEXT HANDOFF worktree: " + str(exc)
    if record["Current worktree HEAD"] != head:
        return "CONTEXT HANDOFF does not name current Implementer HEAD"
    uncommitted = record["sections"]["Uncommitted changes"]
    if dirty == (uncommitted == ["none"]):
        return "CONTEXT HANDOFF disagrees with current uncommitted changes"
    if (record["Candidate created"] == "yes"
            and (dirty or head == base)):
        return "CONTEXT HANDOFF candidate must be a clean changed commit"
    return None


def matching_new_context_handoff(cycle_id, mode, before_inodes):
    """Find one fresh exact context record written by the Implementer."""
    matches = []
    invalid = []
    problems = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        inode = regular_inode(path=path)
        if inode is None or inode in before_inodes:
            continue
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError) as exc:
            invalid.append(path)
            problems.append(os.path.basename(path) + ": " + str(exc))
            continue
        _found_cycle, _found_mode, body, envelope_problem = (
            _ticket_flow_envelope(message=message))
        if (envelope_problem is not None
                or not is_implementer_context_handoff(body)):
            continue
        problem = context_handoff_problem(
            message=message, expected_cycle=cycle_id, expected_mode=mode)
        if os.path.dirname(path) != MAILBOX:
            problem = "CONTEXT HANDOFF was not published in mailbox root"
        if problem is None:
            matches.append(path)
        else:
            invalid.append(path)
            problems.append(os.path.basename(path) + ": " + problem)
    if problems:
        return None, list(dict.fromkeys(invalid + matches)), "; ".join(problems)
    if len(matches) > 1:
        return None, matches, "expected at most one CONTEXT HANDOFF"
    return (matches[0] if matches else None), [], None


def latest_context_handoff_path(cycle_id, mode):
    """Return the newest valid same-cycle record for a replacement turn."""
    matches = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        if regular_inode(path=path) is None:
            continue
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            continue
        found_cycle, found_mode, body, problem = _ticket_flow_envelope(
            message=message)
        if (problem is None and found_cycle == cycle_id
                and found_mode == mode
                and is_implementer_context_handoff(body)):
            matches.append(path)
    if not matches:
        return None
    path = max(matches, key=lambda item: (
        sequence_in_name(os.path.basename(item)) or -1))
    message = read_cycle_message(path=path)
    problem = context_handoff_problem(
        message=message, expected_cycle=cycle_id, expected_mode=mode)
    if problem is not None:
        raise TicketCycleStateError(
            "saved replacement CONTEXT HANDOFF is stale: " + problem)
    return path


def replacement_context_notice(path):
    """Tell a fresh Implementer where to read the prior exact record."""
    return (
        "REPLACEMENT IMPLEMENTER CONTEXT\n"
        "Read the exact prior Implementer record at:\n" + path + "\n"
        "Verify it against the repository before editing. It is not a "
        "daemon-written summary. Do not repeat an approach listed under "
        "Do not revisit unless the Architect explicitly reopened it.\n\n")


def checkpoint_handoff_problem(message):
    """Validate the timed or context checkpoint sent to the Architect."""
    _, _, body, problem = _ticket_flow_envelope(message=message)
    if problem is not None or not is_implementer_checkpoint_request(body):
        return ("the 90-minute hook or context hook fired without its "
                "checkpoint handoff")
    if is_implementer_context_handoff(body):
        return context_handoff_problem(message=message)
    current_state_rows = [
        line for line in body.splitlines()
        if line.startswith("- **Current state:**")]
    if (not is_implementer_time_checkpoint(body)
            or len(current_state_rows) != 1
            or current_state_rows[0]
            != IMPLEMENTER_CHECKPOINT_CURRENT_STATE):
        return "the checkpoint needs its exact 90-minute Current state"
    return None


def _ticket_architect_admission(message):
    """Return an exact public-request admission carried by an Opus flow.

    Ordinary role-to-role ticket messages carry no admission line.  The
    first Implementer handoff created by a public Architect turn carries the
    exact request basename and SHA-256 so a crash or reordering cannot pair
    the handoff with another public request.
    """
    _cycle_id, _mode, body, problem = _ticket_flow_envelope(message=message)
    if problem is not None:
        return None, None, problem
    match = re.match(
        r"\A" + re.escape(MAILBOX_ADMISSION_HEADER)
        + r"(\d+-to-fable\.md)@([0-9a-f]{64})\r?\n",
        body)
    admission_like = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*admission[ \t]*:")
    if match is None:
        if re.search(admission_like, body) is not None:
            return None, None, "malformed MAILBOX-ADMISSION line"
        return None, None, None
    if re.search(admission_like, body[match.end():]) is not None:
        return None, None, "duplicate MAILBOX-ADMISSION line"
    return match.group(1), match.group(2), None


def _redteam_closure_envelope(message):
    """Parse one post-commit Red Team request.

    Returns ``(cycle_id, commit, body, problem)``.
    """
    remainder = sol_ticket_body_after_kind(message=message)
    if sol_ticket_kind(message=message) != "closure":
        return None, None, remainder, None
    match = re.match(
        r"\A" + re.escape(MAILBOX_CYCLE_HEADER)
        + r"(" + CYCLE_ID_RE.pattern + r")\r?\n"
        + re.escape(MAILBOX_COMMIT_HEADER)
        + r"([0-9a-f]{40})\r?\n\r?\n",
        remainder)
    if match is None:
        return (None, None, remainder,
                "a Red Team closure must name exactly one ticket cycle and "
                "one daemon-recorded local landing L on its second and third "
                "physical lines")
    body = remainder[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:cycle|commit|return|result)"
        r"[ \t]*:")
    if re.search(reserved, body) is not None:
        return None, None, remainder, "duplicate Red Team review header"
    return match.group(1), match.group(2), body, None


def _redteam_control_plane_envelope(message):
    """Parse one mandatory pre-landing review of exact candidate C."""
    remainder = sol_ticket_body_after_kind(message=message)
    if sol_ticket_kind(message=message) != "control-plane":
        return None, None, remainder, None
    match = re.match(
        r"\A" + re.escape(MAILBOX_CYCLE_HEADER)
        + r"(" + CYCLE_ID_RE.pattern + r")\r?\n"
        + re.escape(MAILBOX_CANDIDATE_HEADER)
        + r"([0-9a-f]{40})\r?\n\r?\n",
        remainder)
    if match is None:
        return (None, None, remainder,
                "a control-plane review must name one exact ticket cycle "
                "and full candidate C")
    body = remainder[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:cycle|candidate|return|result)"
        r"[ \t]*:")
    if re.search(reserved, body) is not None:
        return None, None, remainder, "duplicate control-plane review header"
    return match.group(1), match.group(2), body, None


def _control_plane_review_receipt(message):
    """Parse one exact pre-landing Red Team decision addressed to D0."""
    match = re.match(
        r"\A" + re.escape(MAILBOX_RETURN_HEADER)
        + r"redteam-control-plane\r?\n"
        + re.escape(MAILBOX_CYCLE_HEADER)
        + r"(" + CYCLE_ID_RE.pattern + r")\r?\n"
        + re.escape(MAILBOX_CANDIDATE_HEADER)
        + r"([0-9a-f]{40})\r?\n"
        + re.escape(MAILBOX_RESULT_HEADER)
        + r"(" + "|".join(map(re.escape, CONTROL_PLANE_REVIEW_RESULTS))
        + r")\r?\n\r?\n",
        message)
    if match is None:
        return (None, None, None, message,
                "a control-plane return needs exact cycle, full candidate, "
                "and ACCEPT-CONTROL-PLANE or REJECT-CONTROL-PLANE")
    body = message[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:cycle|candidate|return|result)"
        r"[ \t]*:")
    if re.search(reserved, body) is not None:
        return None, None, None, message, "duplicate control-plane receipt"
    return match.group(1), match.group(2), match.group(3), body, None


def redteam_closure_problem(message):
    """Return a closure-envelope problem, or ``None``."""
    return _redteam_closure_envelope(message=message)[3]


def redteam_closure_ticket(message):
    """Return the one reviewed ticket-cycle identifier, when valid."""
    ticket, _, _, problem = _redteam_closure_envelope(message=message)
    return ticket if problem is None else None


def redteam_closure_commit(message):
    """Return the one full daemon-recorded landing L, when valid."""
    _, commit, _, problem = _redteam_closure_envelope(message=message)
    return commit if problem is None else None


def _redteam_review_receipt(message):
    """Parse the Red Team's correlated return to the Architect.

    The result vocabulary belongs to the advisory role: ``NO CHANGE`` means
    the accepted fix still stands, and ``REOPEN`` supplies evidence for later
    Architect assessment. Architect ``GO``/``NO-GO`` are deliberately absent.
    """
    match = re.match(
        r"\A" + re.escape(MAILBOX_RETURN_HEADER)
        + r"redteam-closure\r?\n"
        + re.escape(MAILBOX_CYCLE_HEADER)
        + r"(" + CYCLE_ID_RE.pattern + r")\r?\n"
        + re.escape(MAILBOX_COMMIT_HEADER)
        + r"([0-9a-f]{40})\r?\n"
        + re.escape(MAILBOX_RESULT_HEADER)
        + r"(" + "|".join(map(re.escape, REDTEAM_REVIEW_RESULTS))
        + r")\r?\n\r?\n",
        message)
    if match is None:
        return (None, None, None, message,
                "a Red Team return needs exact review ticket, commit, and "
                "NO CHANGE or REOPEN headers")
    body = message[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:cycle|commit|return|result)"
        r"[ \t]*:")
    if re.search(reserved, body) is not None:
        return (None, None, None, message,
                "duplicate Red Team review receipt header")
    return match.group(1), match.group(2), match.group(3), body, None


def _architect_go_request(message):
    """Parse one decision-only GO request bound to the audited candidate."""
    match = re.match(
        r"\A" + re.escape(MAILBOX_RETURN_HEADER)
        + r"architect-go\r?\n"
        + re.escape(MAILBOX_CYCLE_HEADER)
        + r"(" + CYCLE_ID_RE.pattern + r")\r?\n"
        + re.escape(MAILBOX_CANDIDATE_HEADER)
        + r"([0-9a-f]{40})\r?\n"
        + re.escape(MAILBOX_MODE_HEADER)
        + r"(" + "|".join(map(re.escape, ARCHITECT_COMMIT_MODES))
        + r")\r?\n"
        + re.escape(MAILBOX_DECISION_HEADER)
        + r"GO(?:\r?\n|\Z)",
        message)
    if match is None:
        return (None, None, None,
                "an Architect GO request needs exact return, cycle, "
                "candidate, mode, and GO decision headers")
    remainder = message[match.end():]
    if remainder.strip():
        return (None, None, None,
                "an Architect GO request may not carry free-form work")
    return match.group(1), match.group(2), match.group(3), None


def architect_go_request_payload(cycle_id, candidate_commit, mode):
    """Build the decision-only daemon request written after Architect GO."""
    if not isinstance(cycle_id, str) or CYCLE_ID_RE.fullmatch(cycle_id) is None:
        raise ValueError("invalid ticket cycle: " + repr(cycle_id))
    if (not isinstance(candidate_commit, str)
            or FULL_COMMIT_RE.fullmatch(candidate_commit) is None):
        raise ValueError(
            "invalid Implementer candidate: " + repr(candidate_commit))
    if mode not in ARCHITECT_COMMIT_MODES:
        raise ValueError("invalid Architect GO mode: " + repr(mode))
    return (MAILBOX_RETURN_HEADER + "architect-go\n"
            + MAILBOX_CYCLE_HEADER + cycle_id + "\n"
            + MAILBOX_CANDIDATE_HEADER + candidate_commit + "\n"
            + MAILBOX_MODE_HEADER + mode + "\n"
            + MAILBOX_DECISION_HEADER + "GO\n")


def backlog_close_request_payload(cycle_id, candidate_commit, mode):
    """Ask the Architect to close the backlog and repeat its exact GO."""
    architect_go_request_payload(cycle_id, candidate_commit, mode)
    return (
        MAILBOX_FLOW_HEADER + "ticket\n"
        + MAILBOX_CYCLE_HEADER + cycle_id + "\n"
        + MAILBOX_MODE_HEADER + mode + "\n\n"
        + BACKLOG_CLOSE_REQUIRED_HEADER + candidate_commit + "\n\n"
        + "- **Candidate commit:** " + candidate_commit + "\n\n"
        + "Your completed audit already accepted this exact candidate. Do "
          "not repeat the audit or rerun the Implementer. Close and seal "
          "this ticket in backlog.md, then send one fresh exact GO for the "
          "same C. This is bookkeeping recovery only.\n")


def _architect_notes_go_request(message):
    """Parse one body-free permanent-note commit request bound to B and P."""
    match = re.fullmatch(
        re.escape(MAILBOX_RETURN_HEADER) + r"architect-notes-go\r?\n"
        + re.escape(MAILBOX_BASE_HEADER) + r"([0-9a-f]{40})\r?\n"
        + re.escape(MAILBOX_NOTES_COMMIT_HEADER)
        + r"([0-9a-f]{40})\r?\n"
        + re.escape(MAILBOX_DECISION_HEADER) + r"GO(?:\r?\n)?",
        message)
    if match is None:
        return (None, None,
                "an Architect notes GO needs exact return, base, notes "
                "commit, and GO headers with no body")
    if match.group(1) == match.group(2):
        return (None, None,
                "an Architect notes GO must name a new commit")
    return match.group(1), match.group(2), None


def architect_notes_go_request_payload(base_commit, notes_commit):
    """Build the exact parent-daemon request for one permanent-note commit."""
    if (not isinstance(base_commit, str)
            or FULL_COMMIT_RE.fullmatch(base_commit) is None
            or not isinstance(notes_commit, str)
            or FULL_COMMIT_RE.fullmatch(notes_commit) is None
            or base_commit == notes_commit):
        raise ValueError("invalid permanent-note commit request")
    return (MAILBOX_RETURN_HEADER + "architect-notes-go\n"
            + MAILBOX_BASE_HEADER + base_commit + "\n"
            + MAILBOX_NOTES_COMMIT_HEADER + notes_commit + "\n"
            + MAILBOX_DECISION_HEADER + "GO\n")


def _architect_notes_admin_envelope(message):
    """Return the plain-language body of one dedicated note-update turn."""
    match = re.match(
        r"\A" + re.escape(MAILBOX_ADMIN_HEADER)
        + r"permanent-notes\r?\n\r?\n", message)
    if match is None:
        return None, "not a permanent-notes admin request"
    body = message[match.end():]
    if not body.strip():
        return None, "permanent-notes admin request needs an update summary"
    if re.search(
            r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*admin[ \t]*:", body):
        return None, "duplicate MAILBOX-ADMIN header"
    return body, None


def architect_notes_admin_payload(text):
    """Build the exact Architect self-route for a durable note update."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("permanent-notes admin summary must be nonempty")
    if re.search(
            r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*admin[ \t]*:", text):
        raise ValueError("permanent-notes admin summary repeats its header")
    payload = MAILBOX_ADMIN_HEADER + "permanent-notes\n\n" + text
    if not payload.endswith("\n"):
        payload += "\n"
    return payload


def is_architect_notes_admin_message(message):
    """Return whether ``message`` is one valid dedicated note-update turn."""
    _body, problem = _architect_notes_admin_envelope(message=message)
    return problem is None


def architect_notes_admin_journal_path(request_name, relay_dir=None):
    """Return the durable post-child journal for one exact admin request."""
    if PENDING_MESSAGE_RE.fullmatch(request_name) is None:
        raise TicketCycleStateError("invalid admin request journal name")
    directory = RELAY_DIR if relay_dir is None else relay_dir
    return os.path.join(
        directory, ".pending-notes-admin-" + request_name + ".json")


def write_architect_notes_admin_journal(request_name, request_message,
                                        base_commit, phase,
                                        notes_commit=None,
                                        receipt_sha256=None):
    """Atomically bind one admin request to its validated recovery phase."""
    if phase not in {"started", "validated-noop", "validated-commit"}:
        raise TicketCycleStateError("invalid note-admin journal phase")
    if (FULL_COMMIT_RE.fullmatch(str(base_commit)) is None
            or not is_architect_notes_admin_message(
                message=request_message)):
        raise TicketCycleStateError("invalid note-admin journal authority")
    if phase == "validated-commit":
        if (FULL_COMMIT_RE.fullmatch(str(notes_commit)) is None
                or notes_commit == base_commit
                or re.fullmatch(r"[0-9a-f]{64}",
                                str(receipt_sha256)) is None):
            raise TicketCycleStateError(
                "validated note-admin commit journal needs exact P/receipt")
    elif notes_commit is not None or receipt_sha256 is not None:
        raise TicketCycleStateError(
            "non-commit note-admin journal cannot name P/receipt")
    payload = {
        "schema": ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA,
        "request": request_name,
        "request_sha256": hashlib.sha256(
            request_message.encode("utf-8")).hexdigest(),
        "base": base_commit,
        "phase": phase,
        "notes_commit": notes_commit,
        "receipt_sha256": receipt_sha256,
    }
    os.makedirs(RELAY_DIR, exist_ok=True)
    path = architect_notes_admin_journal_path(request_name=request_name)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".pending-notes-admin-", dir=RELAY_DIR)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) \
                as stream:
            descriptor = -1
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(directory=RELAY_DIR)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
    return path


def read_architect_notes_admin_journal(request_name, request_message,
                                       relay_dir=None):
    """Read one exact journal and rebind it to the inflight request bytes."""
    path = architect_notes_admin_journal_path(
        request_name=request_name, relay_dir=relay_dir)
    try:
        raw = stable_regular_bytes(
            path=path,
            maximum_bytes=MAX_ARCHITECT_NOTES_ADMIN_JOURNAL_BYTES,
            label="permanent-note admin journal")
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_key_refusal)
    except (OSError, ValueError, UnicodeDecodeError,
            json.JSONDecodeError) as exc:
        raise TicketCycleStateError(
            "cannot verify permanent-note admin journal: " + str(exc)) \
            from exc
    if (not isinstance(payload, dict)
            or set(payload) != {
                "schema", "request", "request_sha256", "base", "phase",
                "notes_commit", "receipt_sha256"}
            or payload["schema"] != ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA
            or payload["request"] != request_name
            or payload["request_sha256"] != hashlib.sha256(
                request_message.encode("utf-8")).hexdigest()
            or FULL_COMMIT_RE.fullmatch(str(payload["base"])) is None
            or payload["phase"] not in {
                "started", "validated-noop", "validated-commit"}):
        raise TicketCycleStateError(
            "permanent-note admin journal has invalid fields")
    if payload["phase"] == "validated-commit":
        if (FULL_COMMIT_RE.fullmatch(
                str(payload["notes_commit"])) is None
                or payload["notes_commit"] == payload["base"]
                or re.fullmatch(r"[0-9a-f]{64}",
                                str(payload["receipt_sha256"])) is None):
            raise TicketCycleStateError(
                "permanent-note admin commit journal is incomplete")
    elif (payload["notes_commit"] is not None
          or payload["receipt_sha256"] is not None):
        raise TicketCycleStateError(
            "permanent-note admin non-commit journal names a receipt")
    return payload


def remove_architect_notes_admin_journal(request_name):
    """Remove only the journal for a successfully archived admin request."""
    path = architect_notes_admin_journal_path(request_name=request_name)
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    fsync_directory(directory=RELAY_DIR)


def _architect_notes_admin_request_path(request_name):
    """Find the one durable admin request bound to a recovery journal."""
    matches = []
    for directory in (MAILBOX, os.path.join(MAILBOX, "inflight"),
                      os.path.join(MAILBOX, "failed"), DONE):
        path = os.path.join(directory, request_name)
        if regular_inode(path=path) is not None:
            matches.append(path)
    if len(matches) != 1:
        raise TicketCycleStateError(
            "permanent-note admin journal needs exactly one saved request; "
            "found " + str(len(matches)) + " for " + request_name)
    return matches[0]


def _validated_commit_admin_journals(base_commit, notes_commit,
                                     receipt_sha256):
    """Return exact validated-commit journals bound to one B/P receipt."""
    prefix = ".pending-notes-admin-"
    suffix = ".json"
    matches = []
    pattern = os.path.join(RELAY_DIR, prefix + "*" + suffix)
    for journal_path in sorted(glob.glob(pattern)):
        filename = os.path.basename(journal_path)
        request_name = filename[len(prefix):-len(suffix)]
        request_match = PENDING_MESSAGE_RE.fullmatch(request_name)
        if request_match is None or request_match.group(1) != "fable":
            raise TicketCycleStateError(
                "malformed permanent-note admin journal name: "
                + journal_path)
        request_path = _architect_notes_admin_request_path(
            request_name=request_name)
        try:
            request_message = stable_regular_bytes(
                path=request_path,
                maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="journaled permanent-note admin").decode(
                    "utf-8", errors="strict")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise TicketCycleStateError(
                "cannot verify saved permanent-note admin request: "
                + str(exc)) from exc
        journal = read_architect_notes_admin_journal(
            request_name=request_name, request_message=request_message)
        if (journal["phase"] == "validated-commit"
                and journal["base"] == base_commit
                and journal["notes_commit"] == notes_commit
                and journal["receipt_sha256"] == receipt_sha256):
            matches.append((request_name, request_path, journal_path))
    return matches


def retire_validated_commit_admin_journal(base_commit, notes_commit,
                                          receipt_sha256):
    """Remove one journal only after its exact P receipt is consumed."""
    matches = _validated_commit_admin_journals(
        base_commit=base_commit, notes_commit=notes_commit,
        receipt_sha256=receipt_sha256)
    if len(matches) > 1:
        raise TicketCycleStateError(
            "more than one validated admin journal names the same B/P "
            "receipt")
    if not matches:
        return False
    request_name, request_path, _journal_path = matches[0]
    if os.path.dirname(request_path) != DONE:
        raise TicketCycleStateError(
            "validated admin journal cannot retire before its request is "
            "archived")
    remove_architect_notes_admin_journal(request_name=request_name)
    return True


def sol_ticket_body(message):
    """Return the human body after valid Sol envelope lines."""
    remainder = sol_ticket_body_after_kind(message=message)
    if sol_ticket_kind(message=message) == "closure":
        _, _, body, problem = _redteam_closure_envelope(message=message)
        return remainder if problem is not None else body
    if sol_ticket_kind(message=message) == "control-plane":
        _, _, body, problem = _redteam_control_plane_envelope(
            message=message)
        return remainder if problem is not None else body
    if sol_ticket_kind(message=message) != "discovery":
        return remainder
    _, _, body, problem = _sol_discovery_envelope(message=message)
    return remainder if problem is not None else body


def architect_request_scope(text):
    """Classify only an explicit positive command at the start of user text.

    A quotation, a negation, leading prose, or a later mention remains
    bounded. This recognizer is used only while constructing the public
    Architect envelope; dispatch trusts the saved ``MAILBOX-SCOPE`` value.
    """
    positive = (
        r"\A(?:please(?:,)?[ \t]+)?"
        r"(?:instruct[ \t]+the[ \t]+red[ \t]+team[ \t]+to[ \t]+)?"
        r"do[ \t]+a[ \t]+widespread[ \t]+search\b")
    return ("widespread" if re.search(positive, text, re.IGNORECASE)
            is not None else "bounded")


def mailbox_role_for_dispatch(agent, message=None):
    """Return the checksum-guard role for one exact dispatch route."""
    if agent == "fable":
        return "architect"
    if agent == "opus":
        return "implementer"
    if agent == "sol":
        return "red-team"
    raise ValueError("unknown mailbox agent: " + repr(agent))


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


def sol_ticket_payload(ticket_kind, text, discovery_severity=None,
                       discovery_scope=None, review_cycle=None,
                       review_commit=None):
    """Build the byte-stable persisted envelope for a Sol message."""
    if ticket_kind == "discovery":
        if review_cycle is not None or review_commit is not None:
            raise ValueError(
                "review identity is valid only for Red Team reviews")
        if discovery_severity is None:
            discovery_severity = DEFAULT_DISCOVERY_SEVERITY
        if discovery_severity not in DISCOVERY_SEVERITIES:
            raise ValueError("invalid discovery severity: "
                             + repr(discovery_severity))
        if discovery_scope is None:
            discovery_scope = DEFAULT_DISCOVERY_SCOPE
        if discovery_scope not in DISCOVERY_SCOPES:
            raise ValueError("invalid discovery scope: "
                             + repr(discovery_scope))
        payload = (SOL_TICKET_HEADER + ticket_kind + "\n"
                   + SOL_SEVERITY_HEADER + discovery_severity + "\n"
                   + SOL_SCOPE_HEADER + discovery_scope + "\n\n"
                   + text)
    else:
        if discovery_severity is not None:
            raise ValueError(
                "discovery severity is valid only for discovery tickets")
        if discovery_scope is not None:
            raise ValueError(
                "discovery scope is valid only for discovery tickets")
        if ticket_kind in {"closure", "control-plane"} and (
                review_cycle is not None or review_commit is not None):
            if (not isinstance(review_cycle, str)
                    or CYCLE_ID_RE.fullmatch(review_cycle)
                    is None):
                raise ValueError("invalid Red Team review cycle: "
                                 + repr(review_cycle))
            if (not isinstance(review_commit, str)
                    or FULL_COMMIT_RE.fullmatch(review_commit) is None):
                raise ValueError("invalid Red Team review commit: "
                                 + repr(review_commit))
            identity_header = (MAILBOX_COMMIT_HEADER
                               if ticket_kind == "closure"
                               else MAILBOX_CANDIDATE_HEADER)
            payload = (SOL_TICKET_HEADER + ticket_kind + "\n"
                       + MAILBOX_CYCLE_HEADER + review_cycle + "\n"
                       + identity_header + review_commit + "\n\n" + text)
        else:
            if review_cycle is not None or review_commit is not None:
                raise ValueError(
                    "review identity is valid only for Red Team reviews")
            payload = SOL_TICKET_HEADER + ticket_kind + "\n\n" + text
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def redteam_review_receipt_payload(review_cycle, review_commit, result,
                                   text):
    """Build the exact Red Team return that completes one ticket cycle."""
    if (not isinstance(review_cycle, str)
            or CYCLE_ID_RE.fullmatch(review_cycle) is None):
        raise ValueError("invalid Red Team review cycle: "
                         + repr(review_cycle))
    if (not isinstance(review_commit, str)
            or FULL_COMMIT_RE.fullmatch(review_commit) is None):
        raise ValueError("invalid Red Team review commit: "
                         + repr(review_commit))
    if result not in REDTEAM_REVIEW_RESULTS:
        raise ValueError("Red Team review result must be NO CHANGE or REOPEN")
    payload = (MAILBOX_RETURN_HEADER + "redteam-closure\n"
               + MAILBOX_CYCLE_HEADER + review_cycle + "\n"
               + MAILBOX_COMMIT_HEADER + review_commit + "\n"
               + MAILBOX_RESULT_HEADER + result + "\n\n" + text)
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def control_plane_review_receipt_payload(review_cycle, candidate, result,
                                         text):
    """Build one exact Red Team decision for protected candidate C."""
    if (not isinstance(review_cycle, str)
            or CYCLE_ID_RE.fullmatch(review_cycle) is None):
        raise ValueError("invalid control-plane review cycle")
    if (not isinstance(candidate, str)
            or FULL_COMMIT_RE.fullmatch(candidate) is None):
        raise ValueError("invalid control-plane candidate")
    if result not in CONTROL_PLANE_REVIEW_RESULTS:
        raise ValueError("invalid control-plane review result")
    payload = (MAILBOX_RETURN_HEADER + "redteam-control-plane\n"
               + MAILBOX_CYCLE_HEADER + review_cycle + "\n"
               + MAILBOX_CANDIDATE_HEADER + candidate + "\n"
               + MAILBOX_RESULT_HEADER + result + "\n\n" + text)
    return payload if payload.endswith("\n") else payload + "\n"


def architect_user_request_payload(text, discovery_severity=None):
    """Build the persisted public envelope addressed only to Architect."""
    if text == ARCHITECT_FIX_ONLY_REQUEST:
        return text
    if discovery_severity is None:
        discovery_severity = DEFAULT_DISCOVERY_SEVERITY
    discovery_scope = architect_request_scope(text=text)
    if discovery_scope == "widespread":
        discovery_severity = "low"
    if discovery_severity not in DISCOVERY_SEVERITIES:
        raise ValueError("invalid discovery severity: "
                         + repr(discovery_severity))
    payload = (SOL_SEVERITY_HEADER + discovery_severity + "\n"
               + SOL_SCOPE_HEADER + discovery_scope + "\n\n" + text)
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def _architect_user_request_envelope(message):
    """Return ``(severity, scope, body, problem)`` for a public envelope."""
    match = re.match(
        r"\A" + re.escape(SOL_SEVERITY_HEADER)
        + r"(" + "|".join(map(re.escape, DISCOVERY_SEVERITIES)) + r")\r?\n"
        + re.escape(SOL_SCOPE_HEADER)
        + r"(" + "|".join(map(re.escape, DISCOVERY_SCOPES))
        + r")\r?\n\r?\n",
        message)
    if match is None:
        return (None, None, message,
                "a public Architect request needs exact MAILBOX-SEVERITY "
                "and MAILBOX-SCOPE headers, in that order, followed by one "
                "blank line")
    body = message[match.end():]
    reserved_like = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:severity|scope)[ \t]*:")
    if re.search(reserved_like, body) is not None:
        return None, None, message, "duplicate public request header"
    return match.group(1), match.group(2), body, None


def architect_user_request_problem(message):
    """Return a malformed public-envelope reason, or ``None``."""
    return _architect_user_request_envelope(message=message)[3]


def architect_user_request_severity(message):
    """Return a valid public Architect envelope severity, or ``None``."""
    severity, _, _, problem = _architect_user_request_envelope(
        message=message)
    return severity if problem is None else None


def architect_user_request_scope(message):
    """Return a valid public Architect envelope scope, or ``None``."""
    _, scope, _, problem = _architect_user_request_envelope(message=message)
    return scope if problem is None else None


def architect_user_request_body(message):
    """Return the exact user text after a valid Architect envelope."""
    _, _, body, problem = _architect_user_request_envelope(message=message)
    return message if problem is not None else body


def architect_admission_token(request_name, digest):
    """Return the exact token binding one public request to its handoff."""
    match = (PENDING_MESSAGE_RE.fullmatch(request_name)
             if isinstance(request_name, str) else None)
    if (match is None or match.group(1) != "fable"
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise TicketCycleStateError(
            "invalid public Architect admission identity")
    return request_name + "@" + digest


def split_architect_admission_token(token):
    """Return ``(request_name, digest)`` for one exact admission token."""
    if not isinstance(token, str) or "@" not in token:
        raise TicketCycleStateError(
            "invalid public Architect admission token")
    request_name, digest = token.rsplit("@", 1)
    architect_admission_token(request_name=request_name, digest=digest)
    return request_name, digest


def message_claims_architect_admission(path, token):
    """Return whether one mailbox file names this exact public request."""
    try:
        message = read_cycle_message(path=path)
    except (OSError, ValueError, TicketCycleStateError):
        return False
    return MAILBOX_ADMISSION_HEADER + token in message.splitlines()


def architect_admission_prompt(token):
    """Tell one public Architect turn how to bind its single outcome."""
    if token is None:
        return ""
    request_name, digest = split_architect_admission_token(token=token)
    exact = architect_admission_token(
        request_name=request_name, digest=digest)
    return (
        "PUBLIC REQUEST ADMISSION:\n"
        "This public request provisionally occupies one finite ticket slot. "
        "Produce exactly ONE fresh outcome and never remain silent:\n"
        "1. For one Implementer ticket, put this exact line first in the "
        "body immediately after the ticket flow headers and blank line:\n"
        "    " + MAILBOX_ADMISSION_HEADER + exact + "\n"
        "2. For one bounded or widespread Sol discovery request, put the "
        "same exact line first in its body immediately after the Sol "
        "severity and scope headers.\n"
        "3. If this request creates no ticket, write one fresh "
        "<next-sequence>-to-user.md receipt beginning exactly:\n"
        "    " + MAILBOX_RETURN_HEADER
        + PUBLIC_ARCHITECT_NO_TICKET_RETURN + "\n"
        "    " + MAILBOX_ADMISSION_HEADER + exact + "\n"
        "    " + MAILBOX_DECISION_HEADER
        + PUBLIC_ARCHITECT_NO_TICKET_DECISION + "\n"
        "You may put a plain-language answer after one blank line in option "
        "3. Do not produce two outcomes, copy the admission to later work, "
        "or treat silence as success.\n\n")


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


def sol_ticket_refusal(ticket_kind, admission_count, fix_only,
                       transport_valid=False, discovery_severity=None,
                       discovery_scope=None, unclassified_count=0,
                       ledger_problem=None):
    """Return the binding refusal reason for a Sol ticket, or ``None``."""
    if ticket_kind == "transport":
        if transport_valid:
            return None
        return ("MAILBOX-TICKET: transport is reserved for the daemon's "
                "exact --ping sol payload")
    if ticket_kind not in SOL_TICKET_KINDS:
        return ("missing or invalid first line; every Sol ticket must start "
                "with exactly 'MAILBOX-TICKET: closure', "
                "'MAILBOX-TICKET: discovery', 'MAILBOX-TICKET: policy', "
                "or 'MAILBOX-TICKET: control-plane'")
    if ledger_problem is not None:
        return ledger_problem
    if ticket_kind == "discovery":
        if discovery_severity is None:
            discovery_severity = DEFAULT_DISCOVERY_SEVERITY
        if discovery_severity not in DISCOVERY_SEVERITIES:
            return ("a discovery ticket needs one severity: high, medium, "
                    "or low")
        if discovery_scope not in DISCOVERY_SCOPES:
            return ("a discovery ticket needs one saved scope: bounded or "
                    "widespread")
    elif discovery_severity is not None:
        return "--severity is valid only for discovery tickets"
    elif discovery_scope is not None:
        return "discovery scope is valid only for discovery tickets"
    if fix_only and ticket_kind not in {
            "closure", "policy", "control-plane"}:
        return ("fix-only watch is closing-only; discovery tickets and new "
                "backlog lines are forbidden until the watch is restarted "
                "without --fix-only")
    if ticket_kind == "discovery" and unclassified_count:
        return ("the backlog has " + str(unclassified_count)
                + " unclassified open ticket(s); the Architect must assign "
                "each one a valid priority and either BUG FIX or NEW "
                "FUNCTIONALITY before new discovery can enter")
    if (ticket_kind == "discovery"
            and discovery_scope == "widespread"):
        if discovery_severity != "low":
            return ("a widespread search is automatically Low; save exactly "
                    "MAILBOX-SEVERITY: low")
        if admission_count:
            return ("a widespread search waits until no open Critical, High, "
                    "or Medium ticket remains; the current non-Low count is "
                    + str(admission_count) + ". Open Low tickets do not block "
                    "this search")
    if (ticket_kind == "discovery"
            and admission_count >= DISCOVERY_ADMISSION_THRESHOLD):
        return ("the open Critical, High, and Medium ticket count is "
                + str(admission_count) + ", at or past "
                + str(DISCOVERY_ADMISSION_THRESHOLD)
                + "; do not admit this discovery yet. Record it as a local "
                "deferred candidate without a countable '- OPEN' marker. "
                "Low tickets do not count toward this limit. When the count "
                "falls below the threshold, assess the result and insert an "
                "accepted ticket in the matching Critical, High, Medium, or "
                "Low backlog group; only the Architect may designate "
                "Critical")
    return None


def inflight_lane_blockers(skip_redteam=False):
    """Return unresolved inflight agent messages grouped by cwd lane.

    Only exact dispatchable message names participate. A hand-made file or an
    archived ``-to-user`` note under inflight cannot block an agent lane.
    Live topology gives Architect, Implementer, and Sol distinct saved
    directories, so one unresolved role blocks only that role. Imported tests
    may still deliberately assign a shared cwd and retain shared-tree safety.
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
        cwd = mailbox_lane_cwd(agent=agent)
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
        if mailbox_lane_cwd(agent=queued_agent) == mailbox_lane_cwd(agent=agent):
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
                    discovery_severity=None, discovery_scope=None,
                    saved_discovery=False,
                    saved_architect_request=False,
                    candidate_scope=None, routine_review=None):
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
    if candidate_scope is not None:
        result = candidate_scope["result"]
        lines.extend(("--- CANDIDATE TICKET SCOPE (binding) ---",
                      "result: " + result))
        if candidate_scope["paths"]:
            lines.append("paths: " + ", ".join(
                repr(path) for path in candidate_scope["paths"]))
        if result == "SCOPE_EXCEEDED":
            lines.append(
                "Candidate C is preserved, but the Implementer expanded the "
                "ticket. Architect GO explicitly accepts this expansion; a "
                "repair handoff rejects it. Audit the listed paths.")
        lines.extend(("--- END CANDIDATE TICKET SCOPE ---", ""))
    if routine_review is not None:
        lines.extend((
            "--- ROUTINE REVIEW (binding) ---",
            "kind: " + routine_review,
            "Review the named ticket and commit only. This is not a new "
            "discovery search.",
            "ticket character limit: "
            + ("none (--max 0)" if MAX_CHARACTERS == 0 else
               str(MAX_CHARACTERS) + " added plus deleted characters"),
            "--- END ROUTINE REVIEW ---"))
        return "\n".join(lines) + "\n\n"
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
    lines.append("--- DISCOVERY SCOPE (binding) ---")
    if discovery_scope is None:
        discovery_scope = DEFAULT_DISCOVERY_SCOPE
    if saved_discovery:
        lines.append("saved scope for this discovery: " + discovery_scope)
    elif saved_architect_request:
        lines.append(
            "saved scope for discovery requested by this ticket: "
            + discovery_scope)
    else:
        lines.append(
            "scope to save on an ordinary new discovery: "
            + discovery_scope)
    lines.append(
        "bounded: review only the named commit or change and the behavior "
        "it directly affects.")
    lines.append(
        "widespread: search beyond one named change; it must remain Low and "
        "wait until no Critical, High, or Medium ticket is open.")
    lines.append(
        "Trust MAILBOX-SCOPE and MAILBOX_DISCOVERY_SCOPE, not a phrase found "
        "in the body or cited note.")
    lines.append("--- END DISCOVERY SCOPE ---")
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
        line = "minimum bug-fix severity: " + DISCOVERY_SEVERITY
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
    fsync_directory(directory=directory)
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


def live_action_topology_is_current(agent, action):
    """Refuse an action whose saved worktrees were removed by cleanup."""
    if ACTIVE_TOPOLOGY is None:
        return True
    try:
        validate_live_agent_dispatch_topology(agent=agent)
    except (OSError, PrimaryWorktreeError) as exc:
        print(action + " refused: saved worktree topology changed ("
              + str(exc) + ").")
        return False
    return True


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
    if not live_action_topology_is_current("fable", "dispatch"):
        return None
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
    if not live_action_topology_is_current("fable", "dispatch"):
        release_dispatch_lock(lock_file=lock_file)
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
    """Serialize Architect decisions that the parent daemon may land."""
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


def validate_live_agent_dispatch_topology(agent):
    """Re-prove one mutable agent's saved checkout before launch."""
    if agent not in {"fable", "opus", "sol"}:
        raise ValueError(
            "topology proof is defined only for Fable, Opus, and Sol")
    if ACTIVE_TOPOLOGY is None:
        raise PrimaryWorktreeError(
            "live " + agent + " dispatch has no validated topology")
    lock_file = _open_primary_lock(repository_root=REPO_ROOT)
    try:
        primary = load_primary_state(path=ACTIVE_TOPOLOGY["primary_state"])
        if primary["schema"] != PRIMARY_STATE_SCHEMA:
            raise PrimaryWorktreeError(
                "live dispatch requires topology-aware primary state")
        primary = validate_primary_state(
            state=primary, repository_root=REPO_ROOT, allow_move=False)
        implementer = load_primary_state(
            path=ACTIVE_TOPOLOGY["implementer_state"])
        implementer = validate_implementer_state(
            state=implementer, repository_root=REPO_ROOT,
            primary_state=primary, allow_move=False)
        sol = load_primary_state(path=ACTIVE_TOPOLOGY["sol_state"])
        sol = validate_sol_state(
            state=sol, repository_root=REPO_ROOT, primary_state=primary,
            allow_move=False)
        notes = validated_primary_notes(primary_path=primary["path"])
        authoritative_files = validate_authoritative_role_files(
            primary_path=primary["path"])
        validate_distinct_agent_states(
            primary_state=primary, implementer_state=implementer,
            sol_state=sol)
        expected = ACTIVE_TOPOLOGY
        if (os.path.abspath(primary["path"]) != expected["primary_path"]
                or primary["branch"] != expected["primary_branch"]
                or os.path.abspath(implementer["path"])
                != expected["implementer_path"]
                or os.path.abspath(sol["path"]) != expected["sol_path"]
                or notes != expected["shared_notes"]
                or AGENT_CWD["fable"] != expected["primary_path"]
                or AGENT_CWD["opus"] != expected["implementer_path"]
                or AGENT_CWD["sol"] != expected["sol_path"]
                or os.path.realpath(os.path.join(AI_ROOT, "notes"))
                != expected["shared_notes"]):
            raise PrimaryWorktreeError(
                "saved agent topology changed after this process started")
        if agent == "sol":
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
        if agent == "fable":
            role_path = expected["primary_path"]
            role_label = "saved Architect primary worktree"
        elif agent == "opus":
            role_path = expected["implementer_path"]
            role_label = "saved Implementer worktree"
        else:
            role_path = expected["sol_path"]
            role_label = "saved Sol worktree"
        role_identity = _plain_directory(
            path=role_path, label=role_label)
        notes_identity = _plain_directory(
            path=expected["shared_notes"], label="shared notes directory")
        proof = {
            "agent": agent,
            "role_path": role_path,
            "role_identity": role_identity,
            "notes_path": expected["shared_notes"],
            "notes_identity": notes_identity,
            "authoritative_files": authoritative_files,
        }
        if agent == "sol":
            proof["sol_path"] = role_path
            proof["sol_identity"] = role_identity
        return proof
    finally:
        _release_primary_lock(lock_file=lock_file)


def recheck_agent_dispatch_directories(proof, mutable_paths=()):
    """Prove launch pathnames still name the pre-claim directories."""
    if proof is None:
        raise PrimaryWorktreeError(
            "live dispatch is missing its topology proof")
    _require_directory_identity(
        path=proof["role_path"], identity=proof["role_identity"],
        label="saved " + proof["agent"] + " worktree")
    _require_directory_identity(
        path=proof["notes_path"], identity=proof["notes_identity"],
        label="shared notes directory")
    recheck_authoritative_role_files(
        proof=proof["authoritative_files"], mutable_paths=mutable_paths)


def revalidate_agent_dispatch_topology(proof):
    """Re-prove all Git and command bindings without accepting a new inode."""
    recheck_agent_dispatch_directories(proof=proof)
    current = validate_live_agent_dispatch_topology(agent=proof["agent"])
    if current != proof:
        raise PrimaryWorktreeError(
            "saved agent worktree topology changed after message claim")
    recheck_agent_dispatch_directories(proof=current)
    return current


def revalidate_protected_policy_admin_topology(proof):
    """Allow only Architect-owned policy files to change in an admin turn."""
    if not isinstance(proof, dict):
        return revalidate_agent_dispatch_topology(proof=proof)
    mutable = ARCHITECT_ROLE_PATHS + (ROLE_CONTRACT_RELATIVE_PATH,)
    recheck_agent_dispatch_directories(proof=proof, mutable_paths=mutable)
    current = validate_live_agent_dispatch_topology(agent=proof["agent"])
    recheck_agent_dispatch_directories(
        proof=current, mutable_paths=mutable)
    return current


def _architect_ordinary_tracked_state(worktree, base_commit, cached):
    """Return exact non-note tracked state relative to one frozen base."""
    arguments = [
        "diff", "--no-ext-diff", "--no-renames", "--binary",
        "--full-index", "--ignore-submodules=none"]
    if cached:
        arguments.append("--cached")
    arguments.extend([base_commit, "--", "."])
    arguments.extend(
        ":(top,exclude)" + path
        for path in ARCHITECT_PERMANENT_NOTE_PATHS
        + (BACKLOG_RELATIVE_PATH,))
    try:
        result = _run_git(
            repository_root=worktree, arguments=arguments, check=False)
    except PrimaryWorktreeError:
        raise
    if result.returncode != 0:
        raise PrimaryWorktreeError(
            "cannot capture Architect ordinary tracked state")
    return result.stdout


def _ordinary_untracked_worktree_state(worktree):
    """Return every nonignored untracked path in one persistent worktree.

    Mailbox transport, relay logs, and temporary note evidence are ignored by
    Git and therefore do not appear in this proof. The tracked backlog has its
    own Architect seal. A newly created source, test, README, or tool appears.
    """
    try:
        result = _run_git(
            repository_root=worktree,
            arguments=["ls-files", "--others", "--exclude-standard", "-z",
                       "--", "."],
            check=False)
    except PrimaryWorktreeError:
        raise
    if result.returncode != 0:
        raise PrimaryWorktreeError(
            "cannot capture nonignored untracked worktree paths")
    return result.stdout


def _git_path_bytes(worktree, object_name, relative_path, maximum_bytes):
    """Read one bounded tracked blob from an exact commit or the index."""
    result = _run_git(
        repository_root=worktree,
        arguments=["show", object_name + ":" + relative_path],
        check=False)
    if result.returncode != 0:
        raise PrimaryWorktreeError(
            relative_path + " is missing from " + object_name)
    if len(result.stdout) > maximum_bytes:
        raise PrimaryWorktreeError(
            relative_path + " exceeds its protected size limit")
    return result.stdout


def _top_level_tracked_markdown(raw_paths, label):
    """Decode one NUL path list and select top-level ai/notes Markdown."""
    selected = set()
    try:
        values = [raw.decode("utf-8", errors="strict")
                  for raw in raw_paths.split(b"\0") if raw]
    except UnicodeDecodeError as exc:
        raise PrimaryWorktreeError(
            label + " contains a non-UTF-8 tracked path") from exc
    for relative_path in values:
        parent, separator, name = relative_path.rpartition("/")
        if (separator and parent == "ai/notes"
                and name.casefold().endswith(".md")):
            selected.add(relative_path)
    return selected


def _require_exact_permanent_note_set(primary_worktree, head):
    """Require exactly eleven tracked top-level notes in HEAD and the index."""
    expected = set(ARCHITECT_PERMANENT_NOTE_PATHS)
    head_result = _run_git(
        repository_root=primary_worktree,
        arguments=["ls-tree", "-r", "--name-only", "-z", head,
                   "--", "ai/notes"])
    index_result = _run_git(
        repository_root=primary_worktree,
        arguments=["ls-files", "-z", "--", "ai/notes"])
    head_notes = _top_level_tracked_markdown(
        raw_paths=head_result.stdout, label="Architect HEAD")
    index_notes = _top_level_tracked_markdown(
        raw_paths=index_result.stdout, label="Architect index")
    head_notes.discard(BACKLOG_RELATIVE_PATH)
    index_notes.discard(BACKLOG_RELATIVE_PATH)
    if head_notes != expected or index_notes != expected:
        raise PrimaryWorktreeError(
            "Architect HEAD and index must contain exactly the eleven "
            "permanent top-level Markdown notes")


def _validate_protected_tracked_state(primary_worktree):
    """Require protected policy files and their guard to match primary HEAD."""
    try:
        head = worktree_head(worktree=primary_worktree)
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(
            "cannot identify the Architect commit protecting permanent "
            "notes: " + str(exc)) from exc
    _require_exact_permanent_note_set(
        primary_worktree=primary_worktree, head=head)
    for relative_path in ARCHITECT_PROTECTED_TRACKED_PATHS:
        expected = _git_path_bytes(
            worktree=primary_worktree, object_name=head,
            relative_path=relative_path,
            maximum_bytes=MAX_PROTECTED_NOTE_BYTES)
        staged = _git_path_bytes(
            worktree=primary_worktree, object_name="",
            relative_path=relative_path,
            maximum_bytes=MAX_PROTECTED_NOTE_BYTES)
        try:
            working = stable_regular_bytes(
                path=os.path.join(primary_worktree, relative_path),
                maximum_bytes=MAX_PROTECTED_NOTE_BYTES,
                label="protected " + relative_path)
        except (OSError, ValueError) as exc:
            raise PrimaryWorktreeError(str(exc)) from exc
        if staged != expected or working != expected:
            raise PrimaryWorktreeError(
                relative_path + " does not match the current Architect "
                "commit and index")
    try:
        ending_head = worktree_head(worktree=primary_worktree)
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(
            "cannot recheck the Architect commit protecting permanent "
            "notes: " + str(exc)) from exc
    if ending_head != head:
        raise PrimaryWorktreeError(
            "Architect HEAD changed while permanent notes were checked")


def _validate_sealed_backlog(primary_worktree):
    """Return the backlog bytes after matching the Architect-sealed SHA."""
    notes = os.path.join(primary_worktree, "ai", "notes")
    backlog_path = os.path.join(notes, "backlog.md")
    state_path = os.path.join(notes, BACKLOG_GUARD_STATE_NAME)
    backlog_exists = os.path.lexists(backlog_path)
    state_exists = os.path.lexists(state_path)
    if not backlog_exists and not state_exists:
        if (os.path.lexists(backlog_path)
                or os.path.lexists(state_path)):
            raise PrimaryWorktreeError(
                "backlog or its guard appeared while absence was checked")
        return b""
    if backlog_exists != state_exists:
        raise PrimaryWorktreeError(
            "backlog and its Architect-sealed guard must either both exist "
            "or both be absent")
    try:
        state_before = stable_regular_bytes(
            path=state_path, maximum_bytes=MAX_BACKLOG_GUARD_STATE_BYTES,
            label="backlog guard state")
        backlog = stable_regular_bytes(
            path=backlog_path, maximum_bytes=MAX_BACKLOG_LEDGER_BYTES,
            label="Architect backlog")
        state_after = stable_regular_bytes(
            path=state_path, maximum_bytes=MAX_BACKLOG_GUARD_STATE_BYTES,
            label="backlog guard state")
    except (OSError, ValueError) as exc:
        raise PrimaryWorktreeError(str(exc)) from exc
    if state_after != state_before:
        raise PrimaryWorktreeError(
            "backlog guard state changed while the backlog was checked")
    try:
        state = json.loads(
            state_before.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_key_refusal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrimaryWorktreeError(
            "backlog guard state is not exact UTF-8 JSON: " + str(exc)) \
            from exc
    version = state.get("version") if isinstance(state, dict) else None
    expected_fields = {"backlog", "sha256", "version"}
    if version == 2:
        expected_fields.add("previous_sha256")
    if (not isinstance(state, dict)
            or type(version) is not int or version not in {1, 2}
            or set(state) != expected_fields
            or state.get("backlog") != "ai/notes/backlog.md"
            or not isinstance(state.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", state["sha256"]) is None
            or (version == 2 and (
                not isinstance(state.get("previous_sha256"), str)
                or re.fullmatch(
                    r"[0-9a-f]{64}", state["previous_sha256"]) is None))):
        raise PrimaryWorktreeError(
            "backlog guard state has missing, extra, or invalid fields")
    observed = hashlib.sha256(backlog).hexdigest()
    if observed != state["sha256"]:
        raise PrimaryWorktreeError(
            "backlog differs from the SHA-256 last sealed by the Architect")
    return backlog


def require_closed_backlog_ticket(ticket_anchor, sealed_backlog):
    """Prove one ticket is Closed before landing."""
    try:
        lines = sealed_backlog.decode("utf-8", errors="strict").splitlines()
    except (AttributeError, UnicodeDecodeError) as exc:
        raise TicketCycleStateError("backlog is not UTF-8") from exc
    marker = '<a id="' + ticket_anchor + '"></a>'
    if (lines.count("# Closed tickets") != 1 or lines.count(marker) != 1
            or any(OPEN_BACKLOG_CANDIDATE_RE.match(line)
                   and "(#" + ticket_anchor + ")" in line for line in lines)):
        raise BacklogTicketOpenError("ticket is Open: " + ticket_anchor)
    start = lines.index(marker) + 1
    if start <= lines.index("# Closed tickets"):
        raise BacklogTicketOpenError("ticket is Open: " + ticket_anchor)
    end = next((index for index in range(start + 1, len(lines))
                if lines[index].startswith(("## ", '<a id="'))), len(lines))
    section = lines[start:end]
    headings = ["### High-level summary", "### Current status",
                "### What is already fixed", "### What is missing"]
    if (not section or not section[0].startswith("## ")
            or any(section.count(heading) != 1 for heading in headings)):
        raise TicketCycleStateError("invalid ticket: " + ticket_anchor)
    positions = [section.index(heading) for heading in headings]
    if positions != sorted(positions):
        raise TicketCycleStateError("invalid ticket: " + ticket_anchor)
    status = [line for line in section[positions[1] + 1:positions[2]]
              if line.startswith("**CLOSED.**")]
    missing = section[positions[3] + 1:]
    missing = missing[:next((index for index, line in enumerate(missing)
                            if line.startswith(("### ", "<details>"))),
                           len(missing))]
    if len(status) != 1 or [line for line in missing if line] != [
            "Nothing for this ticket."]:
        raise TicketCycleStateError("invalid ticket: " + ticket_anchor)


def _bridge_local_sealed_backlog(primary_worktree):
    """Adopt legacy local state or initialize the tracked backlog seal."""
    names = ("backlog.md", BACKLOG_GUARD_STATE_NAME)
    source_notes = os.path.join(REPO_ROOT, "ai", "notes")
    target_notes = os.path.join(primary_worktree, "ai", "notes")
    targets = [os.path.join(target_notes, name) for name in names]
    recovery = os.path.join(target_notes, BACKLOG_SYNC_RECOVERY_NAME)
    if os.path.lexists(recovery):
        saved = stable_regular_bytes(
            path=recovery, maximum_bytes=MAX_BACKLOG_LEDGER_BYTES,
            label="backlog sync recovery")
        if not os.path.lexists(targets[0]):
            os.replace(recovery, targets[0])
        else:
            working = stable_regular_bytes(
                path=targets[0], maximum_bytes=MAX_BACKLOG_LEDGER_BYTES,
                label="Architect backlog")
            if working == saved:
                os.unlink(recovery)
            else:
                head = _run_git(
                    repository_root=primary_worktree,
                    arguments=["show", "HEAD:" + BACKLOG_RELATIVE_PATH],
                    check=False)
                if head.returncode != 0 or head.stdout != working:
                    raise PrimaryWorktreeError(
                        "backlog sync recovery conflicts with visible work")
                os.replace(recovery, targets[0])
    if os.path.lexists(targets[0]) and not os.path.lexists(targets[1]):
        committed = _run_git(
            repository_root=primary_worktree,
            arguments=["show", "HEAD:" + BACKLOG_RELATIVE_PATH],
            check=False)
        if (committed.returncode == 0
                and len(committed.stdout) <= MAX_BACKLOG_LEDGER_BYTES):
            working = stable_regular_bytes(
                path=targets[0], maximum_bytes=MAX_BACKLOG_LEDGER_BYTES,
                label="tracked Architect backlog")
        else:
            working = None
        if working is not None and committed.stdout == working:
            _atomic_write_primary_state(
                state={"backlog": BACKLOG_RELATIVE_PATH,
                       "sha256": hashlib.sha256(working).hexdigest(),
                       "version": 1},
                path=targets[1])
    if all(os.path.lexists(path) for path in targets):
        try:
            _validate_sealed_backlog(primary_worktree=primary_worktree)
        except PrimaryWorktreeError:
            if _clean_worktree_status(worktree=primary_worktree):
                raise
            committed = _run_git(
                repository_root=primary_worktree,
                arguments=["show", "HEAD:" + BACKLOG_RELATIVE_PATH],
                check=False)
            working = stable_regular_bytes(
                path=targets[0], maximum_bytes=MAX_BACKLOG_LEDGER_BYTES,
                label="tracked Architect backlog")
            if committed.returncode != 0 or committed.stdout != working:
                raise
            _atomic_write_primary_state(
                state={"backlog": BACKLOG_RELATIVE_PATH,
                       "sha256": hashlib.sha256(working).hexdigest(),
                       "version": 1},
                path=targets[1])
            _validate_sealed_backlog(primary_worktree=primary_worktree)
        return

    sources = [os.path.join(source_notes, name) for name in names]
    if not any(os.path.lexists(path) for path in sources):
        _validate_sealed_backlog(primary_worktree=primary_worktree)
        return
    _validate_sealed_backlog(primary_worktree=REPO_ROOT)

    for source, target in zip(sources, targets):
        if (os.path.lexists(target)
                and not _regular_files_equal(source, target)):
            raise PrimaryWorktreeError(
                "primary backlog conflicts: " + target)
    for source, target in zip(sources, targets):
        if not os.path.lexists(target):
            _copy_regular_archive_file(
                source=source, destination=target,
                expected_size=os.lstat(source).st_size)
    _validate_sealed_backlog(primary_worktree=primary_worktree)


def _validate_current_protected_primary_state(primary_worktree):
    """Accept current Architect authority, including a concurrent seal/commit."""
    last_error = None
    for attempt in range(PROTECTED_STATE_RECHECK_ATTEMPTS):
        try:
            _validate_protected_tracked_state(
                primary_worktree=primary_worktree)
            _validate_sealed_backlog(primary_worktree=primary_worktree)
            return
        except PrimaryWorktreeError as exc:
            last_error = exc
            if attempt + 1 < PROTECTED_STATE_RECHECK_ATTEMPTS:
                time.sleep(PROTECTED_STATE_RECHECK_SECONDS)
    raise PrimaryWorktreeError(
        "shared Architect-owned notes are not in an accepted state: "
        + str(last_error))


def _capture_shared_protected_state():
    """Return a proof that can revalidate shared protected notes by authority."""
    primary = AGENT_CWD["fable"]
    identity = _plain_directory(
        path=primary, label="saved Architect primary worktree")
    _validate_current_protected_primary_state(
        primary_worktree=primary)
    return {"primary": primary, "identity": identity}


def _recheck_shared_protected_state(proof):
    """Allow only an Architect-sealed backlog or committed permanent notes."""
    if (not isinstance(proof, dict)
            or set(proof) != {"identity", "primary"}):
        raise PrimaryWorktreeError(
            "shared protected-note proof is missing or malformed")
    _require_directory_identity(
        path=proof["primary"], identity=proof["identity"],
        label="saved Architect primary worktree")
    _validate_current_protected_primary_state(
        primary_worktree=proof["primary"])


def capture_persistent_role_state(agent):
    """Freeze tracked-state authority for persistent non-Implementer roles."""
    if agent not in {"fable", "opus", "sol"}:
        return None
    worktree = AGENT_CWD[agent]
    shared_proof = (_capture_shared_protected_state()
                    if agent in {"opus", "sol"} else None)
    try:
        head = worktree_head(worktree=worktree)
    except TicketCycleStateError as exc:
        raise PrimaryWorktreeError(
            "cannot capture persistent " + agent + " HEAD: " + str(exc)) \
            from exc
    if agent == "sol":
        if _tracked_worktree_changes(worktree=worktree):
            raise PrimaryWorktreeError(
                "saved Sol worktree has tracked or nonignored untracked "
                "changes; preserve them manually before Red Team dispatch")
        return {"agent": agent, "worktree": worktree, "head": head,
                "shared_proof": shared_proof}
    if agent == "opus":
        return {"agent": agent, "worktree": worktree,
                "shared_proof": shared_proof}
    return {
        "agent": agent,
        "worktree": worktree,
        "base": head,
        "worktree_state": _architect_ordinary_tracked_state(
            worktree=worktree, base_commit=head, cached=False),
        "index_state": _architect_ordinary_tracked_state(
            worktree=worktree, base_commit=head, cached=True),
        "untracked_state": _ordinary_untracked_worktree_state(
            worktree=worktree),
    }


def implementer_checkpoint_delivered(state_path):
    """Return true only after the hook records its complete instruction."""
    if not state_path:
        return False
    marker = stable_regular_bytes(
        path=state_path, maximum_bytes=32,
        label="Implementer checkpoint marker", missing_ok=True)
    return marker == b"triggered\n"


def recheck_persistent_role_state(proof):
    """Refuse tracked edits outside the authority of Fable or Sol."""
    if proof is None:
        return
    agent = proof["agent"]
    worktree = proof["worktree"]
    if agent in {"opus", "sol"}:
        _recheck_shared_protected_state(proof=proof["shared_proof"])
    if agent == "opus":
        return
    if agent == "sol":
        try:
            current_head = worktree_head(worktree=worktree)
        except TicketCycleStateError as exc:
            raise PrimaryWorktreeError(
                "cannot recheck persistent Sol HEAD: " + str(exc)) from exc
        if (current_head != proof["head"]
                or _tracked_worktree_changes(worktree=worktree)):
            raise PrimaryWorktreeError(
                "Red Team changed tracked or nonignored untracked files in "
                "its persistent worktree; the changes were preserved for "
                "inspection")
        return
    if agent != "fable":
        raise PrimaryWorktreeError("unknown persistent role-state proof")
    current_worktree = _architect_ordinary_tracked_state(
        worktree=worktree, base_commit=proof["base"], cached=False)
    current_index = _architect_ordinary_tracked_state(
        worktree=worktree, base_commit=proof["base"], cached=True)
    current_untracked = _ordinary_untracked_worktree_state(
        worktree=worktree)
    if (current_worktree != proof["worktree_state"]
            or current_index != proof["index_state"]
            or current_untracked != proof["untracked_state"]):
        raise PrimaryWorktreeError(
            "Architect changed ordinary tracked or nonignored untracked "
            "source, tests, README, or tools in the coordination worktree; "
            "the changes were preserved for inspection")


def validate_live_sol_dispatch_topology():
    """Compatibility wrapper for focused Sol topology witnesses."""
    return validate_live_agent_dispatch_topology(agent="sol")


def recheck_sol_dispatch_directories(proof):
    """Compatibility wrapper for focused Sol directory witnesses."""
    return recheck_agent_dispatch_directories(proof=proof)


def revalidate_sol_dispatch_topology(proof):
    """Compatibility wrapper for focused Sol topology witnesses."""
    return revalidate_agent_dispatch_topology(proof=proof)


CLAUDE_TOKEN_EXHAUSTION_MARKERS = (
    "credit balance is too low", "your org is out of usage",
    "you're out of usage credits", "you've hit your monthly spend limit",
)
SOL_TOKEN_EXHAUSTION_MARKERS = (
    "you've hit your usage limit", "your workspace is out of credits",
    "you're out of credits", "you've reached your workspace credit limit",
    "you hit your spend cap", '"usage_limit_exceeded"',
    '_credits_depleted"', '_usage_limit_reached"',
)


def provider_is_out_of_tokens(agent, reply_lines):
    """Recognize terminal account exhaustion, not transient API failures.

    Arguments:
      agent = Mailbox role name.
      reply_lines = Completed provider-log lines.

    Returns:
      True for a known account limit.
    """
    if agent not in {"fable", "opus", "sol"}:
        return False
    markers = (CLAUDE_TOKEN_EXHAUSTION_MARKERS
               if agent in {"fable", "opus"}
               else SOL_TOKEN_EXHAUSTION_MARKERS)
    tail = "\n".join(reply_lines[-24:]).replace("’", "'").casefold()
    if (agent != "sol" and re.search(
            r"you've hit your (5-hour|five-hour|7-day|seven-day) "
            r"limit\b.*\bresets\b",
            tail)):
        return True
    return any(marker in tail for marker in markers)


def dispatch(path, dry_run, fix_only=False, skip_redteam=False,
             new_reservation_cycle=None, architect_admission=None):
    """Serialize Architect GO decisions, then run one dispatch."""
    match = PENDING_MESSAGE_RE.match(os.path.basename(path))
    if match is None:
        raise ValueError("not a pending agent message: " + path)
    agent = match.group(1)
    if dry_run or agent != "fable":
        return dispatch_under_main_checkout_lock(
            path=path, dry_run=dry_run, fix_only=fix_only,
            skip_redteam=skip_redteam,
            new_reservation_cycle=new_reservation_cycle,
            architect_admission=architect_admission)
    notes_admin_reserved = False
    try:
        message = read_cycle_message(path=path)
        notes_admin_reserved = is_architect_notes_admin_message(
            message=message)
    except (OSError, ValueError, TicketCycleStateError):
        # The normal dispatch validator owns the precise refusal and archive.
        notes_admin_reserved = False
    lock_file = acquire_main_checkout_turn_lock()
    if lock_file is None:
        print("refused " + os.path.basename(path) + ": the Architect "
              "GO-decision lock could not be proved; root message "
              "left untouched.")
        return False
    notes_lock = None
    try:
        if notes_admin_reserved:
            notes_lock = acquire_ticket_cycle_lock()
            _require_no_ordinary_landing_transition_locked(
                current_dispatch_path=path)
        return dispatch_under_main_checkout_lock(
            path=path, dry_run=dry_run, fix_only=fix_only,
            skip_redteam=skip_redteam,
            new_reservation_cycle=new_reservation_cycle,
            architect_admission=architect_admission,
            notes_admin_reserved=notes_admin_reserved)
    except TicketCycleStateError as exc:
        print("deferred " + os.path.basename(path)
              + ": permanent-note admin turn requires an idle ticket "
              "boundary (" + str(exc) + "); root message remains queued.")
        return False
    finally:
        if notes_lock is not None:
            release_ticket_cycle_lock(lock_file=notes_lock)
        release_main_checkout_turn_lock(lock_file=lock_file)


def dispatch_under_main_checkout_lock(
        path, dry_run, fix_only=False, skip_redteam=False,
        new_reservation_cycle=None, architect_admission=None,
        notes_admin_reserved=False):
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
    if agent == "daemon":
        raise ValueError(
            "local daemon receipts must use consume_daemon_message, not an "
            "AI dispatch route")
    if not message_is_enabled_for_topology(
            path=path, skip_redteam=skip_redteam):
        hint = "run the matching watch role later"
        print("deferred " + name + ": its saved role is disabled by this "
              "watch; " + hint + "; the root message remains untouched.")
        return False
    agent_topology_proof = None
    persistent_role_state = None
    architect_turn_base = None
    if agent in {"fable", "opus", "sol"} and not dry_run:
        try:
            agent_topology_proof = validate_live_agent_dispatch_topology(
                agent=agent)
            persistent_role_state = capture_persistent_role_state(
                agent=agent)
            if (agent == "fable"
                    and isinstance(persistent_role_state, dict)):
                # Keep the turn's exact starting commit separately from the
                # richer persistent-state proof.  Focused embeddings may use
                # an opaque proof token, while the production proof remains
                # a dictionary; neither case may make the post-turn binding
                # depend on the shape of that token.
                architect_turn_base = persistent_role_state.get("base")
        except (OSError, PrimaryWorktreeError) as exc:
            print("refused " + name + ": saved " + agent
                  + " worktree validation "
                  "failed (" + str(exc) + "); message left untouched.")
            return False
    # Take one severity snapshot before claim_message() moves the mailbox
    # file. Mailbox files are queue information, not accepted backlog tickets,
    # so they do not participate in either severity threshold.
    severity_counts_before_claim = None
    admission_count_before_claim = None
    if agent == "sol":
        severity_counts_before_claim = backlog_severity_counts()
        admission_count_before_claim = (
            severity_counts_before_claim["critical"]
            + severity_counts_before_claim["high"]
            + severity_counts_before_claim["medium"])
    dispatch_path = path
    currency = None
    prior_timeout = None
    if not dry_run:
        if not valid_duration(value=DISPATCH_TIMEOUT_MINUTES,
                              strictly_positive=True):
            print("refused " + name + ": dispatch timeout must be between "
                  "1 and " + str(MAX_DISPATCH_TIMEOUT_MINUTES)
                  + " minutes; message left queued.")
            return False
        try:
            history = timeout_events(name=name)
        except (OSError, ValueError, json.JSONDecodeError,
                OverflowError, RecursionError) as exc:
            print("refused " + name + ": cannot verify its timeout history: "
                  + str(exc) + "; message left queued.")
            return False
        dispatch_path = claim_message(path=path)
        if dispatch_path is None:
            if new_reservation_cycle is not None:
                release_unstarted_ticket_reservation(
                    cycle_id=new_reservation_cycle)
            return False
        # One recursive view, taken only after the atomic claim, owns both
        # currency numbers. Re-globbing each number would let a concurrent
        # sender make the banner internally inconsistent.
        currency = dispatch_currency(dispatch_path=dispatch_path, agent=agent)
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

    notes_admin_body, notes_admin_problem = (
        _architect_notes_admin_envelope(message=message))
    notes_admin_turn = notes_admin_problem is None
    if (message.startswith(MAILBOX_ADMIN_HEADER)
            and not notes_admin_turn):
        if dry_run:
            print("[dry-run] would refuse " + name + ": "
                  + notes_admin_problem)
            return False
        parked = park_failed_message(dispatch_path=dispatch_path)
        print("refused " + name + ": " + notes_admin_problem + "; "
              + ("parked in failed/." if parked else
                 "failed-state move was not verified."))
        return False
    if not dry_run and notes_admin_turn != notes_admin_reserved:
        parked = park_failed_message(dispatch_path=dispatch_path)
        print("refused " + name + ": permanent-note admin identity changed "
              "across its exclusive reservation; "
              + ("parked in failed/." if parked else
                 "failed-state move was not verified."))
        return False

    ticket_kind = None
    review_cycle_id = None
    review_accepted_commit = None
    review_receipt_before = None
    reopen_decision_cycle = None
    reopen_decision_commit = None
    reopen_before = None
    reopen_brief = ""
    effective_discovery_severity = DISCOVERY_SEVERITY
    effective_discovery_scope = DEFAULT_DISCOVERY_SCOPE
    saved_architect_severity = None
    saved_architect_scope = None
    flow_mode = None
    architect_checkpoint_audit = False
    integration_revalidation = None
    if message.startswith(MAILBOX_FLOW_HEADER):
        _, flow_mode, flow_body, flow_problem = _ticket_flow_envelope(
            message=message)
        checkpoint_request = (
            agent == "fable" and flow_problem is None
            and is_implementer_checkpoint_request(flow_body))
        checkpoint_problem = (checkpoint_handoff_problem(message=message)
                              if checkpoint_request else None)
        architect_checkpoint_audit = (
            checkpoint_request and checkpoint_problem is None)
        if flow_problem is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": "
                      + flow_problem)
                return False
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": " + flow_problem + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False
        if (agent == "fable"
                and flow_body.startswith(
                    "CONTROL-PLANE-INTEGRATION: REVALIDATE\n")):
            try:
                integration_revalidation = (
                    control_plane_integration_request(message=message))
            except TicketCycleStateError as exc:
                if dry_run:
                    print("[dry-run] would refuse " + name + ": " + str(exc))
                    return False
                parked = park_failed_message(dispatch_path=dispatch_path)
                print("refused " + name + ": " + str(exc) + "; "
                      + ("parked in failed/." if parked else
                         "failed-state move was not verified."))
                return False
        if not ticket_cycle_mode_is_enabled(
                mode=flow_mode, skip_redteam=skip_redteam):
            reason = ("MAILBOX-MODE: " + flow_mode
                      + " belongs to another watch role")
        else:
            reason = None
        if reason is None and checkpoint_problem is not None:
            reason = checkpoint_problem
        if reason is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": " + reason)
                return False
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": " + reason + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False
    if agent == "opus" and not message.startswith(MAILBOX_FLOW_HEADER):
        reason = ("Implementer work must carry one ticket-cycle flow "
                  "header; ask the Architect to reissue the handoff")
        if dry_run:
            print("[dry-run] would refuse " + name + ": " + reason)
            return False
        parked = park_failed_message(dispatch_path=dispatch_path)
        print("refused " + name + ": " + reason + "; "
              + ("parked in failed/." if parked else
                 "failed-state move was not verified."))
        return False
    if agent == "fable" and message.startswith(MAILBOX_RETURN_HEADER):
        returned_cycle, returned_commit, returned_result, _, receipt_problem = (
            _redteam_review_receipt(
            message=message)
        )
        if receipt_problem is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": "
                      + receipt_problem)
                return False
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": " + receipt_problem + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False
        if returned_result == "REOPEN":
            reopen_decision_cycle = returned_cycle
            reopen_decision_commit = returned_commit
            try:
                reopen_before = current_reopen_ticket(
                    cycle_id=returned_cycle)
                reopen_brief = _REOPEN_TRANSITION.architect_brief(
                    ticket=reopen_before, cycle=returned_cycle,
                    landing=returned_commit)
            except TicketCycleStateError as exc:
                parked = park_prelaunch_message(dispatch_path=dispatch_path)
                print("refused " + name + ": reopening state could not be "
                      "proved (" + str(exc) + "); "
                      + ("retained in prelaunch/." if parked else
                         "failed-state move was not verified."))
                return False
    if agent == "fable" and message.startswith(SOL_SEVERITY_HEADER):
        architect_request_problem = architect_user_request_problem(
            message=message)
        saved_architect_severity = architect_user_request_severity(
            message=message)
        saved_architect_scope = architect_user_request_scope(message=message)
        if architect_request_problem is not None:
            reason = ("invalid public Architect request header; "
                      + architect_request_problem)
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
        effective_discovery_scope = saved_architect_scope
    maintenance_request = message == ARCHITECT_FIX_ONLY_REQUEST
    if (architect_admission is not None and (agent != "fable" or not (
            saved_architect_severity is not None or maintenance_request))):
        parked = park_failed_message(dispatch_path=dispatch_path)
        print("refused " + name + ": saved Architect admission does not "
              "name this exact public request; "
              + ("parked in failed/." if parked else
                 "failed-state move was not verified."))
        return False
    if agent == "sol":
        ticket_kind = sol_ticket_kind(message=message)
        severity_problem = sol_discovery_severity_problem(message=message)
        saved_severity = sol_discovery_severity(message=message)
        saved_scope = sol_discovery_scope(message=message)
        if saved_severity is not None:
            effective_discovery_severity = saved_severity
        if saved_scope is not None:
            effective_discovery_scope = saved_scope
        reason = severity_problem
        if reason is None:
            reason = sol_ticket_refusal(
                ticket_kind=ticket_kind,
                admission_count=admission_count_before_claim,
                fix_only=fix_only,
                transport_valid=valid_sol_transport(message=message),
                discovery_severity=saved_severity,
                discovery_scope=saved_scope,
                unclassified_count=(
                    severity_counts_before_claim["unclassified"]),
                ledger_problem=severity_counts_before_claim["problem"])
        if reason is None and ticket_kind == "closure":
            reason = redteam_closure_problem(message=message)
        if reason is None and ticket_kind == "control-plane":
            _control_cycle, _control_candidate, _body, reason = (
                _redteam_control_plane_envelope(message=message))
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

    implementer_evidence_contract = None
    implementer_return_before = None
    control_review_cycle = None
    control_review_candidate = None
    control_review_before = None
    if agent == "opus" and ACTIVE_TOPOLOGY is not None:
        try:
            implementer_evidence_contract = (
                prepare_implementer_evidence_contract(message=message))
            implementer_return_before = fable_message_inode_snapshot()
        except FatalArchitectLandingError:
            raise
        except (OSError, TicketCycleStateError) as exc:
            reason = "Implementer evidence contract refused: " + str(exc)
            retry_after_note_fix = (
                len(ARCHITECT_DIRECTIVE_LINE_RE.findall(message)) == 1)
            parked = (park_prelaunch_message(dispatch_path=dispatch_path)
                      if retry_after_note_fix else
                      park_failed_message(dispatch_path=dispatch_path))
            state = ("retained in prelaunch/." if retry_after_note_fix
                     else "parked in failed/.")
            print("refused " + name + ": " + reason + "; "
                  + (state if parked else
                     "failed-state move was not verified."))
            return False

    registered_cycle_id = None
    if not dry_run:
        try:
            registered_cycle_id, _ = register_ticket_cycle_message(
                agent=agent, message=message,
                skip_redteam=skip_redteam,
                path_scope=(implementer_evidence_contract.get("allowed_paths")
                            if implementer_evidence_contract is not None
                            else None),
                ticket_class=(implementer_evidence_contract.get(
                    "ticket_class", "ordinary")
                    if implementer_evidence_contract is not None
                    else "ordinary"))
        except TicketCycleStateError as exc:
            reason = "ticket-cycle state refused this message: " + str(exc)
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": " + reason + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False
    if agent == "sol" and ticket_kind == "closure":
        review_cycle_id = redteam_closure_ticket(message=message)
        review_accepted_commit = redteam_closure_commit(message=message)
        try:
            review_ticket = current_reopen_ticket(cycle_id=review_cycle_id)
            reopen_brief = _REOPEN_TRANSITION.redteam_brief(
                ticket=review_ticket, cycle=review_cycle_id,
                landing=review_accepted_commit)
        except (TicketCycleStateError,
                _REOPEN_TRANSITION.ReopenTransitionError) as exc:
            if dry_run:
                print("[dry-run] would refuse " + name + ": closure state "
                      "could not be proved (" + str(exc) + ")")
                return False
            parked = park_prelaunch_message(dispatch_path=dispatch_path)
            print("refused " + name + ": closure state could not be proved ("
                  + str(exc) + "); "
                  + ("retained in prelaunch/." if parked else
                     "failed-state move was not verified."))
            return False
        if not dry_run:
            review_receipt_before = fable_message_inode_snapshot()
    if agent == "sol" and ticket_kind == "control-plane":
        control_review_cycle, control_review_candidate, _body, _problem = (
            _redteam_control_plane_envelope(message=message))
        if not dry_run:
            control_review_before = daemon_message_inode_snapshot()
    if agent == "sol":
        placeholder_body = sol_ticket_body(message=message)
    elif message.startswith(MAILBOX_FLOW_HEADER):
        _, _, placeholder_body, _ = _ticket_flow_envelope(message=message)
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

    implementer_starting_head = None
    implementer_authority_before = None
    audit_cycle_id = None
    audit_commit = None
    audit_worktree = None
    candidate_scope = None
    replacement_context_path = None
    architect_go_before = None
    admin_opus_before = None
    architect_opus_before = None
    architect_fable_before = None
    architect_sol_before = None
    architect_user_before = None
    try:
        if agent == "fable":
            architect_go_before = daemon_message_inode_snapshot()
            if architect_checkpoint_audit or registered_cycle_id is not None:
                architect_opus_before = opus_message_inode_snapshot()
            elif notes_admin_turn:
                admin_opus_before = opus_message_inode_snapshot()
            elif architect_admission is not None:
                architect_opus_before = opus_message_inode_snapshot()
                architect_fable_before = fable_message_inode_snapshot()
                architect_sol_before = sol_message_inode_snapshot()
                architect_user_before = user_message_inode_snapshot()
        if agent == "opus" and registered_cycle_id is not None:
            replacement_context_path = latest_context_handoff_path(
                cycle_id=registered_cycle_id, mode=flow_mode)
            implementer_starting_head = prepare_implementer_cycle_checkout(
                cycle_id=registered_cycle_id,
                preserve_current=replacement_context_path is not None)
            implementer_authority_before = implementer_authority_snapshot()
        elif (agent == "fable" and registered_cycle_id is not None):
            audit_commit = candidate_commit_for_cycle(
                cycle_id=registered_cycle_id)
            if audit_commit is not None:
                audit_cycle_id = registered_cycle_id
                candidate_scope = candidate_scope_for_cycle(
                    cycle_id=audit_cycle_id,
                    candidate_commit=audit_commit)
                audit_worktree = create_audit_snapshot(
                    cycle_id=audit_cycle_id, commit=audit_commit,
                    agent="fable")
        elif agent == "sol" and ticket_kind == "closure":
            audit_cycle_id = review_cycle_id
            audit_commit = review_accepted_commit
            audit_worktree = create_audit_snapshot(
                cycle_id=audit_cycle_id, commit=audit_commit, agent="sol")
        elif agent == "sol" and ticket_kind == "control-plane":
            audit_cycle_id = control_review_cycle
            audit_commit = control_review_candidate
            control = control_plane_ticket_state(
                cycle_id=audit_cycle_id, candidate_commit=audit_commit)
            if (control is None
                    or control["architect_candidate"] != audit_commit):
                raise TicketCycleStateError(
                    "control-plane review lacks D0-recorded Architect GO(C)")
            audit_worktree = create_audit_snapshot(
                cycle_id=audit_cycle_id, commit=audit_commit, agent="sol")
    except (OSError, PrimaryWorktreeError, TicketCycleStateError) as exc:
        parked = park_prelaunch_message(dispatch_path=dispatch_path)
        print("refused " + name + ": exact cycle checkout failed ("
              + str(exc) + "); "
              + ("retained in prelaunch/." if parked else
                 "failed-state move was not verified."))
        return False

    command_prefix = list(AGENT_COMMANDS[agent])
    command_prefix, routine_review = routine_review_command(
        command_prefix,
        agent=agent,
        ticket_kind=ticket_kind,
        candidate_audit=(audit_commit is not None),
        reopening=(reopen_decision_cycle is not None),
        checkpoint=architect_checkpoint_audit,
        integration=(integration_revalidation is not None))
    banner = dispatch_banner(
        store_max=currency[0],
        newer_in_lane=currency[1],
        previous_timeout_minutes=prior_timeout,
        fix_only=fix_only,
        skip_redteam=skip_redteam,
        discovery_severity=effective_discovery_severity,
        discovery_scope=effective_discovery_scope,
        saved_discovery=(ticket_kind == "discovery"),
        saved_architect_request=(saved_architect_severity is not None),
        candidate_scope=candidate_scope,
        routine_review=routine_review)
    if replacement_context_path is not None:
        banner += replacement_context_notice(path=replacement_context_path)
    # The dynamic banner precedes the byte-unchanged PREAMBLE. The
    # role-specific banner sits between them. Consequently PREAMBLE's
    # --- MESSAGE --- delimiter remains immediately before the exact raw
    # mailbox body, and the body remains the prompt's exact suffix. The
    # Architect route receives decision authority. The parent daemon owns the
    # exact local landing after that process exits.
    os.makedirs(RELAY_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(RELAY_DIR, stamp + "-dispatch-" + agent + ".log")
    checkpoint_state_path = None
    if agent == "opus":
        checkpoint_state_path = log_path + "." + name + ".checkpoint"
        settings = implementer_checkpoint_settings(
            python=sys.executable,
            hook_path=os.path.join(
                AGENT_CWD["fable"], "ai", "tools",
                "implementer_checkpoint_hook.py"))
        command_prefix += [
            "--settings", json.dumps(settings, separators=(",", ":"))]
    common_preamble = common_preamble_for_dispatch(
        checkpoint_audit=architect_checkpoint_audit)
    command = command_prefix + ["--",
        banner + reopen_brief + agent_preamble(agent=agent, message=message)
        + architect_admission_prompt(token=architect_admission)
        + common_preamble + message]

    if notes_admin_turn:
        try:
            write_architect_notes_admin_journal(
                request_name=name, request_message=message,
                base_commit=architect_turn_base, phase="started")
        except (OSError, TicketCycleStateError) as exc:
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("  !! permanent-note admin journal could not be started: "
                  + str(exc) + "; "
                  + ("message parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False

    if routine_review is not None:
        print("routine review: " + routine_review + " at "
              + REVIEW_EFFORT + " effort.")
    print("dispatching " + name + " -> " + agent + " ...")
    # Stream the agent's output straight into the relay log AS IT RUNS
    # (stderr folded in -- the codex CLI narrates its progress there), and
    # heartbeat once a minute so a long turn is distinguishable from a hang:
    # elapsed time always moves, and the log size moves whenever the agent
    # emits anything. A buffered subprocess.run() here once left the
    # terminal silent for an entire multi-minute turn.
    started = time.time()
    proc = None
    child_started = False
    launch_error = None
    timed_out = False
    timeout_history_error = None
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(command_prefix) + " <message>\n")
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
        env[DISCOVERY_SCOPE_ENVIRONMENT] = effective_discovery_scope
        env[MAILBOX_ROLE_ENVIRONMENT] = mailbox_role_for_dispatch(
            agent=agent, message=message)
        if agent == "opus":
            if os.path.lexists(checkpoint_state_path):
                launch_error = OSError(
                    "Implementer checkpoint marker already exists")
            env[IMPLEMENTER_CHECKPOINT_DEADLINE_ENVIRONMENT] = repr(
                time.monotonic() + IMPLEMENTER_REVIEW_MINUTES * 60.0)
            env[IMPLEMENTER_CHECKPOINT_STATE_ENVIRONMENT] = (
                checkpoint_state_path)
        else:
            env.pop(IMPLEMENTER_CHECKPOINT_DEADLINE_ENVIRONMENT, None)
            env.pop(IMPLEMENTER_CHECKPOINT_STATE_ENVIRONMENT, None)
        if architect_admission is not None:
            env["MAILBOX_ARCHITECT_ADMISSION"] = architect_admission
        else:
            env.pop("MAILBOX_ARCHITECT_ADMISSION", None)
        if notes_admin_turn:
            env["MAILBOX_NOTES_BASE"] = architect_turn_base
        else:
            env.pop("MAILBOX_NOTES_BASE", None)
        env["MAILBOX_PRIMARY_WORKTREE"] = AGENT_CWD["fable"]
        env["MAILBOX_IMPLEMENTER_WORKTREE"] = AGENT_CWD["opus"]
        env["MAILBOX_EXECUTION_WORKTREE"] = AGENT_CWD["opus"]
        env["MAILBOX_SHARED_NOTES"] = os.path.join(
            AGENT_CWD["fable"], "ai", "notes")
        env["MAILBOX_HANDOFF_CONTRACT"] = os.path.join(
            AGENT_CWD["fable"], "ai", "tools", "handoff_contract.py")
        env["MAILBOX_TICKET_CHANGE_GUARD"] = os.path.join(
            AGENT_CWD["fable"], "ai", "tools", "ticket_change_guard.py")
        if audit_worktree is not None:
            env["MAILBOX_CANDIDATE_COMMIT"] = audit_commit
            env["MAILBOX_AUDIT_WORKTREE"] = audit_worktree
        else:
            env.pop("MAILBOX_CANDIDATE_COMMIT", None)
            env.pop("MAILBOX_AUDIT_WORKTREE", None)
        if fix_only:
            env[FIX_ONLY_ENVIRONMENT] = "1"
        else:
            env.pop(FIX_ONLY_ENVIRONMENT, None)
        if skip_redteam:
            env[SKIP_REDTEAM_ENVIRONMENT] = "1"
        else:
            env.pop(SKIP_REDTEAM_ENVIRONMENT, None)
        try:
            if launch_error is not None:
                raise launch_error
            if agent in {"fable", "opus", "sol"}:
                revalidate_agent_dispatch_topology(
                    proof=agent_topology_proof)
            recheck_persistent_role_state(proof=persistent_role_state)
            proc = subprocess.Popen(command,
                                    stdout=f,
                                    stderr=subprocess.STDOUT,
                                    cwd=AGENT_CWD[agent],
                                    env=env)
            child_started = True
            try:
                if agent in {"fable", "opus", "sol"}:
                    if notes_admin_turn:
                        revalidate_protected_policy_admin_topology(
                            proof=agent_topology_proof)
                    else:
                        revalidate_agent_dispatch_topology(
                            proof=agent_topology_proof)
                if not notes_admin_turn:
                    recheck_persistent_role_state(
                        proof=persistent_role_state)
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
                        try:
                            log_kb = os.fstat(f.fileno()).st_size / 1024.0
                        except OSError:
                            print("  ... " + name + " still running "
                                  + "(%.0f min elapsed, log size unavailable; "
                                  "tail -f %s)" % (elapsed_min, log_path))
                        else:
                            print("  ... " + name + " still running "
                                  + "(%.0f min elapsed, log %.1f kB; "
                                  "tail -f %s)"
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

    authority_changes = []
    if proc is not None and agent == "opus" \
            and implementer_authority_before is not None:
        try:
            authority_changes = implementer_authority_changes(
                before=implementer_authority_before)
        except (OSError, PrimaryWorktreeError,
                TicketCycleStateError) as exc:
            authority_changes = ["snapshot could not be verified: " + str(exc)]
    if authority_changes:
        for return_path in glob.glob(os.path.join(MAILBOX, "*-to-fable.md")):
            inode = regular_inode(path=return_path)
            if (implementer_return_before is not None
                    and inode is not None
                    and inode not in implementer_return_before):
                park_failed_message(dispatch_path=return_path)
        parked = park_failed_message(dispatch_path=dispatch_path)
        print("IMPLEMENTER AUTHORITY VIOLATION:")
        for changed in authority_changes:
            print("- " + changed + " changed during the Implementer turn.")
        print("Candidate or partial work preserved in " + AGENT_CWD["opus"]
              + "; nothing landed. "
              + ("Request parked in failed/." if parked else
                 "Request state needs manual inspection."))
        raise ImplementerAuthorityViolationError(authority_changes)

    persistent_role_error = None
    if proc is not None and agent in {"fable", "opus", "sol"}:
        try:
            if notes_admin_turn:
                revalidate_protected_policy_admin_topology(
                    proof=agent_topology_proof)
            else:
                revalidate_agent_dispatch_topology(
                    proof=agent_topology_proof)
            if not notes_admin_turn:
                recheck_persistent_role_state(
                    proof=persistent_role_state)
        except (OSError, PrimaryWorktreeError) as exc:
            persistent_role_error = exc

    if audit_worktree is not None:
        try:
            remove_audit_snapshot(
                cycle_id=audit_cycle_id, commit=audit_commit, agent=agent)
        except (OSError, PrimaryWorktreeError,
                TicketCycleStateError) as exc:
            if launch_error is None:
                launch_error = PrimaryWorktreeError(
                    "audit snapshot cleanup failed: " + str(exc))

    if launch_error is not None:
        if child_started:
            parked = park_failed_message(dispatch_path=dispatch_path)
            state = "message parked in failed/" if parked \
                else "failed-state move was not verified"
        else:
            parked = park_prelaunch_message(dispatch_path=dispatch_path)
            state = "message retained in prelaunch/" if parked \
                else "pre-launch state move was not verified"
        print("  !! dispatch could not start: " + str(launch_error)
              + "; " + state + "; log -> " + log_path)
        return False

    if persistent_role_error is not None:
        parked = park_failed_message(dispatch_path=dispatch_path)
        state = "message parked in failed/" if parked \
            else "failed-state move was not verified"
        print("  !! dispatch violated its persistent role boundary: "
              + str(persistent_role_error) + "; " + state
              + "; changes preserved; log -> " + log_path)
        return False

    print("  rc=" + str(proc.returncode) + "  log -> " + log_path)
    # show the reply's tail on the terminal so activity is visible live.
    try:
        with open(log_path, encoding="utf-8") as f:
            reply_lines = f.read().strip().splitlines()
    except (OSError, UnicodeError) as exc:
        reply_lines = []
        print("  warning: relay log tail is unavailable: " + str(exc))
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
        if provider_is_out_of_tokens(agent=agent, reply_lines=reply_lines):
            parked = park_failed_message(dispatch_path=dispatch_path)
            _TOKEN_EXHAUSTION_STOP.set()
            raise RoleTokenExhaustionError(
                agent=agent,
                request_path=(os.path.join(MAILBOX, "failed", name)
                              if parked else None))
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

    implementer_delivery_receipt = None
    architect_delivery_receipt = None
    if agent == "opus" and registered_cycle_id is not None:
        implementer_completion_ready = True
        implementer_context_handoff = False
        if implementer_evidence_contract is not None:
            try:
                returned_candidate = worktree_head(
                    worktree=AGENT_CWD["opus"])
                implementer_return, invalid_returns, evidence_problem = (
                    matching_new_context_handoff(
                        cycle_id=registered_cycle_id, mode=flow_mode,
                        before_inodes=implementer_return_before))
                implementer_context_handoff = implementer_return is not None
                if (evidence_problem is None
                        and not implementer_context_handoff):
                    implementer_return, invalid_returns, evidence_problem, \
                        implementer_completion_ready = (
                        matching_new_implementer_handoff(
                            cycle_id=registered_cycle_id, mode=flow_mode,
                            candidate_commit=returned_candidate,
                            before_inodes=implementer_return_before,
                            evidence_contract=implementer_evidence_contract))
                elif implementer_context_handoff:
                    implementer_completion_ready = False
            except (OSError, PrimaryWorktreeError,
                    TicketCycleStateError) as exc:
                implementer_return = None
                invalid_returns = []
                evidence_problem = str(exc)
                implementer_completion_ready = None
            if (evidence_problem is None
                    and implementer_checkpoint_delivered(
                        checkpoint_state_path)):
                try:
                    returned_message = read_cycle_message(
                        path=implementer_return)
                except (OSError, ValueError,
                        TicketCycleStateError) as exc:
                    evidence_problem = str(exc)
                else:
                    evidence_problem = checkpoint_handoff_problem(
                        message=returned_message)
                if (evidence_problem is None
                        and returned_candidate == implementer_starting_head):
                    evidence_problem = (
                        "the 90-minute checkpoint needs a new clean "
                        "checkpoint commit")
                if evidence_problem is not None:
                    invalid_returns = ([implementer_return]
                                       if implementer_return else [])
                    implementer_completion_ready = None
            if (evidence_problem is None
                    and implementer_completion_ready is False
                    and not implementer_context_handoff
                    and (returned_candidate != implementer_starting_head
                         or _clean_worktree_status(
                             worktree=AGENT_CWD["opus"]))):
                invalid_returns = [implementer_return]
                evidence_problem = (
                    "blocked subagent evidence is valid only when Opus HEAD "
                    "still equals its cycle starting commit and the "
                    "Implementer worktree has no tracked or untracked edit")
                implementer_completion_ready = None
            if evidence_problem is not None:
                for return_path in invalid_returns:
                    park_failed_message(dispatch_path=return_path)
                parked = park_failed_message(dispatch_path=dispatch_path)
                print("  !! Implementer returned rc=0 but its same-cycle "
                      "subagent evidence was refused before candidate "
                      "freeze: " + evidence_problem + "; "
                      + ("message parked in failed/." if parked else
                         "failed-state move was not verified."))
                return False
        if implementer_completion_ready:
            try:
                if implementer_evidence_contract is not None:
                    implementer_delivery_receipt = (
                        write_implementer_delivery_receipt(
                            request_path=dispatch_path,
                            return_path=implementer_return))
                candidate = record_implementer_candidate(
                    cycle_id=registered_cycle_id,
                    starting_head=implementer_starting_head)
            except (OSError, ValueError, PrimaryWorktreeError,
                    TicketCycleStateError) as exc:
                if implementer_delivery_receipt is not None:
                    try:
                        preserved = (candidate_commit_for_cycle(
                            cycle_id=registered_cycle_id)
                            == returned_candidate)
                    except (OSError, TicketCycleStateError):
                        preserved = True
                    if not preserved:
                        os.remove(implementer_delivery_receipt)
                        fsync_directory(directory=MAILBOX)
                        implementer_delivery_receipt = None
                parked = (False if implementer_delivery_receipt is not None
                          else park_failed_message(
                              dispatch_path=dispatch_path))
                print("  !! Implementer returned rc=0 but its exact "
                      "candidate could not be preserved: " + str(exc) + "; "
                      + ("message parked in failed/." if parked else
                         ("delivery receipt retained with the inflight "
                          "request for restart recovery."
                          if implementer_delivery_receipt is not None else
                          "failed-state move was not verified.")))
                return False
            if candidate is not None:
                print("  preserved Implementer candidate " + candidate
                      + " for " + registered_cycle_id + ".")
        else:
            if implementer_context_handoff:
                print("  Implementer saved an exact CONTEXT HANDOFF; the "
                      "Architect may start a replacement on the same ticket, "
                      "but no candidate was frozen and no cycle completed.")
            else:
                print("  Implementer returned a blocked subagent checkpoint; "
                      "the Architect may revise the same ticket, but no "
                      "candidate was frozen and no GO boundary advanced.")

    if control_review_cycle is not None:
        receipt_path, control_result, receipt_problem = (
            matching_new_control_plane_receipt(
                cycle_id=control_review_cycle,
                candidate=control_review_candidate,
                before_inodes=control_review_before))
        if receipt_problem is not None:
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("  !! Red Team process returned rc=0 but its exact "
                  "control-plane decision was not proved: "
                  + receipt_problem + "; "
                  + ("message parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False
        try:
            # Persist the second key here, where D0 has just proved that the
            # exact receipt was newly produced by this successful Sol turn.
            # A structured file that merely appears in the mailbox has no
            # authority to create this decision.
            record_control_plane_redteam_decision(
                cycle_id=control_review_cycle,
                candidate_commit=control_review_candidate,
                decision=control_result)
        except TicketCycleStateError as exc:
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("  !! Red Team decision could not be saved: " + str(exc)
                  + "; " + ("message parked in failed/." if parked else
                            "failed-state move was not verified."))
            return False
        if not archive_consumed_message(dispatch_path=dispatch_path):
            return False
        print("authenticated mandatory Red Team decision " + control_result
              + " for exact protected C " + control_review_candidate
              + "; D0 will consume " + os.path.basename(receipt_path)
              + ".")
        return True

    if review_cycle_id is not None:
        receipt_path, review_result, receipt_problem = (
            matching_new_redteam_receipt(
                cycle_id=review_cycle_id,
                accepted_commit=review_accepted_commit,
                before_inodes=review_receipt_before))
        if receipt_problem is not None:
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("  !! Red Team process returned rc=0 but its correlated "
                  "receipt was not proved: " + receipt_problem + "; "
                  + ("message parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False
        if not archive_consumed_message(dispatch_path=dispatch_path):
            return False
        if not redteam_review_completes_cycle(review_result):
            print("Red Team returned REOPEN for " + review_cycle_id
                  + " at " + review_accepted_commit
                  + "; the same cycle remains active until the Architect "
                    "records GO or NO-GO.")
            return True
        try:
            completed_now = complete_ticket_cycle(
                cycle_id=review_cycle_id,
                accepted_commit=review_accepted_commit)
        except TicketCycleStateError as exc:
            print("  !! Red Team request was archived and receipt "
                  + os.path.basename(receipt_path)
                  + " exists, but cycle state was not completed: "
                  + str(exc))
            return False
        deliver_pending_ticket_cycle_returns()
        print("ticket cycle complete: Red Team returned " + review_result
              + " for " + review_cycle_id + " at "
              + review_accepted_commit + ".")
        return True

    if reopen_decision_cycle is not None:
        try:
            decision = architect_reopen_decision(
                cycle_id=reopen_decision_cycle, before=reopen_before)
            completed_now = complete_ticket_cycle(
                cycle_id=reopen_decision_cycle,
                accepted_commit=reopen_decision_commit)
        except (PrimaryWorktreeError, TicketCycleStateError) as exc:
            requeued = requeue_retryable_daemon_message(
                dispatch_path=dispatch_path)
            print("  !! Architect REOPEN decision was not accepted: "
                  + str(exc) + "; "
                  + ("the exact request was requeued."
                     if requeued else
                     "the inflight request remains preserved."))
            return False
        if not archive_consumed_message(dispatch_path=dispatch_path):
            print("  !! Architect REOPEN decision is durable, but its input "
                  "could not be archived.")
            return False
        deliver_pending_ticket_cycle_returns()
        print("ticket cycle complete: Architect returned " + decision
              + " to Red Team REOPEN for " + reopen_decision_cycle
              + " at " + reopen_decision_commit + ".")
        return bool(completed_now or decision)

    if (agent == "fable" and audit_cycle_id is not None
            and audit_commit is not None
            and architect_turn_base is not None):
        if worktree_head(worktree=AGENT_CWD["fable"]) != architect_turn_base:
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("  !! Architect candidate audit changed the persistent "
                  "primary HEAD; note commits require a separate no-ticket "
                  "turn; " + ("message parked in failed/." if parked else
                               "failed-state move was not verified."))
            return False
        try:
            _validate_current_protected_primary_state(
                primary_worktree=AGENT_CWD["fable"])
        except PrimaryWorktreeError as exc:
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("  !! Architect candidate audit changed a protected "
                  "permanent note, its guard, or the sealed backlog: "
                  + str(exc) + "; "
                  + ("message parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False
        go_path, invalid_go_paths, go_problem = matching_new_architect_go(
            cycle_id=audit_cycle_id, candidate_commit=audit_commit,
            mode=flow_mode, before_inodes=architect_go_before)
        handoff_path = None
        if architect_checkpoint_audit:
            handoff_path, invalid_handoffs, handoff_problem = (
                matching_new_checkpoint_handoff(
                    cycle_id=audit_cycle_id, mode=flow_mode,
                    before_inodes=architect_opus_before))
            checkpoint_outputs = invalid_handoffs
            if handoff_path is not None:
                checkpoint_outputs.append(handoff_path)
            if go_path is not None:
                invalid_go_paths.append(go_path)
                go_problem = "a progress checkpoint cannot receive landing GO"
            if go_problem is None and handoff_problem is not None:
                go_problem = handoff_problem
            if go_problem is not None:
                invalid_go_paths = list(dict.fromkeys(
                    invalid_go_paths + checkpoint_outputs))
        else:
            handoff_path, invalid_handoffs, handoff_problem = (
                matching_new_architect_handoff(
                    cycle_id=audit_cycle_id, mode=flow_mode,
                    before_inodes=architect_opus_before,
                    required=False))
            if handoff_problem is not None:
                go_problem = (handoff_problem if go_problem is None else
                              go_problem + "; " + handoff_problem)
            elif go_problem is None and ((go_path is None)
                                         == (handoff_path is None)):
                go_problem = (
                    "candidate audit requires exactly one outcome: "
                    "landing GO or same-cycle Implementer repair")
            if go_problem is not None:
                invalid_go_paths = list(dict.fromkeys(
                    invalid_go_paths + invalid_handoffs
                    + [path for path in (go_path, handoff_path)
                       if path is not None]))
        if go_problem is not None:
            for invalid_path in invalid_go_paths:
                park_failed_message(dispatch_path=invalid_path)
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("  !! Architect returned rc=0 but its daemon GO boundary "
                  "was refused: " + go_problem + "; "
                  + ("message parked in failed/." if parked else
                     "failed-state move was not verified."))
            return False
        if go_path is not None:
            print("  authenticated Architect GO for exact candidate "
                  + audit_commit + "; the daemon will prepare its landing "
                  "after this Architect turn releases the main lock.")
        elif handoff_path is not None:
            print("  authenticated Architect repair handoff for "
                  + audit_cycle_id + ".")
        try:
            architect_delivery_receipt = (
                write_implementer_delivery_receipt(
                    request_path=dispatch_path,
                    return_path=go_path or handoff_path))
            if (go_path is not None
                    and control_plane_ticket_state(
                        cycle_id=audit_cycle_id,
                        candidate_commit=audit_commit) is not None):
                # The delivery hard link is deliberately short-lived. Save
                # the protected Architect key while D0 can still prove that
                # this exact Architect turn created this exact GO(C).
                record_control_plane_architect_go(
                    cycle_id=audit_cycle_id,
                    candidate_commit=audit_commit)
                if integration_revalidation is not None:
                    if (integration_revalidation["cycle_id"] != audit_cycle_id
                            or integration_revalidation["candidate"]
                            != audit_commit):
                        raise TicketCycleStateError(
                            "integration audit changed its exact cycle or C")
                    record_control_plane_integration_go(
                        cycle_id=audit_cycle_id,
                        candidate_commit=audit_commit,
                        new_main=integration_revalidation["new_main"],
                        evidence=os.path.basename(
                            architect_delivery_receipt))
        except (OSError, ValueError, TicketCycleStateError) as exc:
            print("  !! validated Architect outcome could not be journaled: "
                  + str(exc) + "; request kept in inflight/.")
            return False

    if (agent == "fable" and audit_cycle_id is None
            and architect_turn_base is not None):
        base_commit = architect_turn_base
        notes_commit = worktree_head(worktree=AGENT_CWD["fable"])
        fresh_daemon = [
            candidate for candidate in glob.glob(
                os.path.join(MAILBOX, "**", "*-to-daemon.md"),
                recursive=True)
            if (regular_inode(path=candidate) is not None
                and regular_inode(path=candidate) not in
                architect_go_before)]
        fresh_opus = []
        opus_before = (admin_opus_before if notes_admin_turn
                       else architect_opus_before)
        if opus_before is not None:
            fresh_opus = [
                candidate for candidate in glob.glob(
                    os.path.join(MAILBOX, "**", "*-to-opus.md"),
                    recursive=True)
                if (regular_inode(path=candidate) is not None
                    and regular_inode(path=candidate) not in
                    opus_before)]
        fresh_fable = []
        fresh_sol = []
        fresh_user = []
        if architect_fable_before is not None:
            fresh_fable = [
                candidate for candidate in glob.glob(
                    os.path.join(MAILBOX, "**", "*-to-fable.md"),
                    recursive=True)
                if (regular_inode(path=candidate) is not None
                    and regular_inode(path=candidate) not in
                    architect_fable_before)]
        if architect_sol_before is not None:
            fresh_sol = [
                candidate for candidate in glob.glob(
                    os.path.join(MAILBOX, "**", "*-to-sol.md"),
                    recursive=True)
                if (regular_inode(path=candidate) is not None
                    and regular_inode(path=candidate) not in
                    architect_sol_before)]
        if architect_user_before is not None:
            fresh_user = [
                candidate for candidate in glob.glob(
                    os.path.join(MAILBOX, "**", "*-to-user.md"),
                    recursive=True)
                if (regular_inode(path=candidate) is not None
                    and regular_inode(path=candidate) not in
                    architect_user_before)]
        if notes_admin_turn:
            if fresh_opus:
                for invalid_path in fresh_opus:
                    park_failed_message(dispatch_path=invalid_path)
                parked = park_failed_message(dispatch_path=dispatch_path)
                print("  !! permanent-note admin turn created an "
                      "Implementer handoff; note administration is "
                         "cycle-free; "
                      + ("message parked in failed/." if parked else
                         "failed-state move was not verified."))
                return False
        if notes_commit == base_commit:
            try:
                if (notes_admin_turn
                        and _clean_worktree_status(
                            worktree=AGENT_CWD["fable"])):
                    raise PrimaryWorktreeError(
                        "Architect admin turn left uncommitted changes")
                _validate_current_protected_primary_state(
                    primary_worktree=AGENT_CWD["fable"])
            except PrimaryWorktreeError as exc:
                parked = park_failed_message(dispatch_path=dispatch_path)
                print("  !! Architect left a protected permanent note, its "
                      "guard, or the sealed backlog different from commit "
                      "B: " + str(exc) + "; "
                      + ("message parked in failed/." if parked else
                         "failed-state move was not verified."))
                return False
            if architect_admission is not None:
                fresh_fable = [
                    path for path in fresh_fable
                    if message_claims_architect_admission(
                        path=path, token=architect_admission)]
                fresh_outputs = (
                    fresh_opus + fresh_sol + fresh_user
                    + fresh_fable + fresh_daemon)
                outcome_problem = None
                outcome_kind = None
                outcome_path = None
                if len(fresh_outputs) != 1:
                    outcome_problem = (
                        "public Architect admission requires exactly one "
                        "fresh digest-bound outcome; found "
                        + str(len(fresh_outputs)))
                elif fresh_fable or fresh_daemon:
                    outcome_path = fresh_outputs[0]
                    outcome_problem = (
                        "public Architect admission cannot return through "
                        + os.path.basename(outcome_path).split("-to-", 1)[-1])
                else:
                    outcome_path = fresh_outputs[0]
                    if os.path.dirname(outcome_path) != MAILBOX:
                        outcome_problem = (
                            "public Architect outcome was not published in "
                            "the mailbox root")
                    else:
                        try:
                            outcome_message = read_cycle_message(
                                path=outcome_path)
                        except (OSError, ValueError,
                                TicketCycleStateError) as exc:
                            outcome_problem = str(exc)
                        else:
                            if fresh_opus:
                                if not outcome_message.startswith(
                                        MAILBOX_FLOW_HEADER):
                                    outcome_problem = (
                                        "Implementer outcome lacks its exact "
                                        "ticket flow envelope")
                                else:
                                    try:
                                        converted_cycle, _ = (
                                            register_ticket_cycle_message(
                                                agent="opus",
                                                message=outcome_message,
                                                skip_redteam=skip_redteam,
                                                architect_admission=(
                                                    architect_admission),
                                                implementer_request_name=(
                                                    os.path.basename(
                                                        outcome_path))))
                                    except TicketCycleStateError as exc:
                                        outcome_problem = str(exc)
                                    else:
                                        outcome_kind = (
                                            "Implementer ticket "
                                            + str(converted_cycle))
                            elif fresh_sol:
                                outcome_problem = (
                                    public_architect_sol_outcome_problem(
                                        message=outcome_message,
                                        expected_token=(
                                            architect_admission)))
                                if outcome_problem is None:
                                    try:
                                        release_architect_ticket_admission(
                                            token=architect_admission)
                                    except TicketCycleStateError as exc:
                                        outcome_problem = str(exc)
                                    else:
                                        outcome_kind = "Sol advisory request"
                            else:
                                outcome_problem = (
                                    public_architect_no_ticket_problem(
                                        message=outcome_message,
                                        expected_token=(
                                            architect_admission)))
                                if outcome_problem is None:
                                    try:
                                        release_architect_ticket_admission(
                                            token=architect_admission)
                                    except TicketCycleStateError as exc:
                                        outcome_problem = str(exc)
                                    else:
                                        outcome_kind = "no-ticket receipt"
                if outcome_problem is not None:
                    for invalid_path in fresh_outputs:
                        park_failed_message(dispatch_path=invalid_path)
                    parked = park_failed_message(
                        dispatch_path=dispatch_path)
                    print("  !! Architect returned rc=0 but its public "
                          "request outcome was refused: "
                          + outcome_problem + "; the provisional admission "
                          "was retained; "
                          + ("message parked in failed/." if parked else
                             "failed-state move was not verified."))
                    return False
                print("  authenticated public Architect outcome: "
                      + outcome_kind + "; exact output "
                      + os.path.basename(outcome_path)
                      + " remains queued for its recipient.")
            if maintenance_request and fresh_opus:
                send(agent="fable", text=ARCHITECT_FIX_ONLY_REQUEST,
                     dry_run=False)
            if fresh_daemon and architect_admission is None:
                for invalid_path in fresh_daemon:
                    park_failed_message(dispatch_path=invalid_path)
                parked = park_failed_message(dispatch_path=dispatch_path)
                print("  !! Architect created a daemon request without one "
                      "new permanent-note commit; "
                      + ("message parked in failed/." if parked else
                         "failed-state move was not verified."))
                return False
            if notes_admin_turn:
                try:
                    write_architect_notes_admin_journal(
                        request_name=name, request_message=message,
                        base_commit=base_commit, phase="validated-noop")
                except (OSError, TicketCycleStateError) as exc:
                    parked = park_failed_message(
                        dispatch_path=dispatch_path)
                    print("  !! validated no-op admin result could not be "
                          "journaled: " + str(exc) + "; "
                          + ("message parked in failed/." if parked else
                             "failed-state move was not verified."))
                    return False
        else:
            if not notes_admin_turn:
                for invalid_path in fresh_daemon:
                    park_failed_message(dispatch_path=invalid_path)
                parked = park_failed_message(dispatch_path=dispatch_path)
                print("  !! Architect changed permanent notes outside the "
                      "dedicated MAILBOX-ADMIN: permanent-notes route; "
                      + ("message parked in failed/." if parked else
                         "failed-state move was not verified."))
                return False
            go_path, invalid_paths, note_problem = (
                matching_new_architect_notes_go(
                    base_commit=base_commit, notes_commit=notes_commit,
                    before_inodes=architect_go_before))
            if note_problem is None:
                try:
                    require_architect_notes_commit(
                        base_commit=base_commit, notes_commit=notes_commit)
                    if notes_admin_reserved:
                        _require_no_ordinary_landing_transition_locked(
                            current_dispatch_path=go_path)
                    else:
                        require_no_ordinary_landing_transition(
                            current_dispatch_path=go_path)
                except (OSError, TicketCycleStateError) as exc:
                    note_problem = str(exc)
                    invalid_paths = [go_path]
            if note_problem is not None:
                for invalid_path in invalid_paths:
                    if invalid_path is not None:
                        park_failed_message(dispatch_path=invalid_path)
                parked = park_failed_message(dispatch_path=dispatch_path)
                print("  !! Architect permanent-note commit was refused: "
                      + note_problem + "; "
                      + ("message parked in failed/." if parked else
                         "failed-state move was not verified."))
                return False
            try:
                receipt_raw = stable_regular_bytes(
                    path=go_path,
                    maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="permanent-note GO receipt")
                write_architect_notes_admin_journal(
                    request_name=name, request_message=message,
                    base_commit=base_commit, phase="validated-commit",
                    notes_commit=notes_commit,
                    receipt_sha256=hashlib.sha256(receipt_raw).hexdigest())
            except (OSError, ValueError, TicketCycleStateError) as exc:
                parked = park_failed_message(dispatch_path=dispatch_path)
                print("  !! validated permanent-note commit could not be "
                      "journaled: " + str(exc) + "; "
                      + ("message parked in failed/." if parked else
                         "failed-state move was not verified."))
                return False
            print("  authenticated permanent-note commit " + notes_commit
                  + " on exact main baseline " + base_commit
                  + "; parent daemon will fast-forward it after this turn.")

    archived = archive_consumed_message(dispatch_path=dispatch_path)
    if archived and implementer_delivery_receipt is not None:
        os.remove(implementer_delivery_receipt)
        fsync_directory(directory=MAILBOX)
    if archived and architect_delivery_receipt is not None:
        os.remove(architect_delivery_receipt)
        fsync_directory(directory=MAILBOX)
    if archived and notes_admin_turn:
        try:
            journal = read_architect_notes_admin_journal(
                request_name=name, request_message=message)
            if journal["phase"] == "validated-noop":
                remove_architect_notes_admin_journal(request_name=name)
            elif journal["phase"] == "validated-commit":
                print("  retained validated permanent-note admin journal "
                      "until its exact P receipt is consumed.")
            else:
                raise TicketCycleStateError(
                    "archived permanent-note admin still has only its "
                    "pre-child journal")
        except (OSError, TicketCycleStateError) as exc:
            print("  warning: admin request is archived, but its recovery "
                  "journal needs user attention: " + str(exc))
    return archived


def park_failed_message(dispatch_path):
    """Move a claimed message to failed and verify its exact inode."""
    _, verified = verified_state_move(
        dispatch_path=dispatch_path,
        directory=os.path.join(MAILBOX, "failed"))
    return verified


def park_prelaunch_message(dispatch_path):
    """Retain a request that was refused before its agent process started."""
    _, verified = verified_state_move(
        dispatch_path=dispatch_path,
        directory=os.path.join(MAILBOX, "prelaunch"))
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


def regular_file_has_prefix(path, prefix):
    """Read only one ASCII prefix while proving a stable regular inode."""
    if not isinstance(prefix, bytes) or not prefix:
        raise ValueError("file prefix must be nonempty bytes")
    initial = os.lstat(path)
    if not stat.S_ISREG(initial.st_mode):
        return False
    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        raw = os.read(descriptor, len(prefix))
        after = os.fstat(descriptor)
        current = os.lstat(path)
    finally:
        os.close(descriptor)
    identities = ((initial.st_dev, initial.st_ino),
                  (before.st_dev, before.st_ino),
                  (after.st_dev, after.st_ino),
                  (current.st_dev, current.st_ino))
    metadata = ((initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns),
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns),
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                (current.st_size, current.st_mtime_ns,
                 current.st_ctime_ns))
    if len(set(identities)) != 1 or len(set(metadata)) != 1:
        raise ValueError("file changed while its prefix was checked")
    return raw == prefix


def restore_state_source(guard_path, dispatch_path, source_inode):
    """Restore the exact claimed inode from its safety guard if necessary."""
    if not os.path.lexists(dispatch_path):
        try:
            os.link(guard_path, dispatch_path)
            fsync_directory(directory=os.path.dirname(dispatch_path))
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
        fsync_directory(directory=os.path.dirname(guard_path))
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


def git_commit_exists(commit):
    """Return whether the primary coordination repository owns this commit."""
    if not isinstance(commit, str) or FULL_COMMIT_RE.fullmatch(commit) is None:
        return False
    process = subprocess.run(
        ["git", "cat-file", "-e", commit + "^{commit}"],
        cwd=AGENT_CWD["fable"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False)
    return process.returncode == 0


def git_commit_descends_from(starting_commit, accepted_commit):
    """Return whether daemon-recorded landing L descends from the base."""
    if (not git_commit_exists(commit=starting_commit)
            or not git_commit_exists(commit=accepted_commit)
            or starting_commit == accepted_commit):
        return False
    process = subprocess.run(
        ["git", "merge-base", "--is-ancestor", starting_commit,
         accepted_commit],
        cwd=AGENT_CWD["fable"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=False)
    return process.returncode == 0


def cycle_candidate_ref(cycle_id):
    """Return one path-safe, deterministic private ref for a ticket."""
    if not isinstance(cycle_id, str) or CYCLE_ID_RE.fullmatch(cycle_id) is None:
        raise TicketCycleStateError("invalid cycle id for candidate ref")
    digest = hashlib.sha256(cycle_id.encode("utf-8")).hexdigest()
    return CANDIDATE_REF_ROOT + "/" + digest + "/candidate"


def candidate_state_path():
    """Return the ignored primary record binding cycles to immutable refs."""
    return os.path.join(MAILBOX, CANDIDATE_STATE_NAME)


def empty_candidate_state():
    """Return a fresh candidate-state payload."""
    return {"schema": CANDIDATE_STATE_SCHEMA, "cycles": {}}


def read_candidate_state():
    """Read the bounded exact-schema candidate record."""
    try:
        raw = stable_regular_bytes(
            path=candidate_state_path(),
            maximum_bytes=MAX_CANDIDATE_STATE_BYTES,
            label="ticket-candidate state", missing_ok=True)
    except (OSError, ValueError) as exc:
        raise TicketCycleStateError(str(exc)) from exc
    if raw is None:
        return empty_candidate_state()
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError,
            OverflowError, RecursionError) as exc:
        raise TicketCycleStateError(
            "ticket-candidate state is not exact JSON") from exc
    if (not isinstance(payload, dict)
            or set(payload) != {"schema", "cycles"}
            or payload.get("schema") != CANDIDATE_STATE_SCHEMA
            or not isinstance(payload.get("cycles"), dict)
            or len(payload["cycles"]) > MAX_TICKET_CYCLE_RECORDS):
        raise TicketCycleStateError(
            "ticket-candidate state has invalid keys")
    normalized = {}
    for cycle_id, record in payload["cycles"].items():
        expected_ref = cycle_candidate_ref(cycle_id=cycle_id)
        if (not isinstance(record, dict)
                or set(record) != {"ref", "commit"}
                or record.get("ref") != expected_ref
                or not isinstance(record.get("commit"), str)
                or FULL_COMMIT_RE.fullmatch(record["commit"]) is None):
            raise TicketCycleStateError(
                "ticket-candidate state has an invalid cycle record")
        normalized[cycle_id] = {
            "ref": expected_ref, "commit": record["commit"]}
    return {"schema": CANDIDATE_STATE_SCHEMA, "cycles": normalized}


def write_candidate_state(state):
    """Publish candidate state by same-directory atomic replacement."""
    if (not isinstance(state, dict)
            or set(state) != {"schema", "cycles"}
            or state.get("schema") != CANDIDATE_STATE_SCHEMA):
        raise TicketCycleStateError(
            "refusing malformed ticket-candidate state")
    # Round-trip through the strict reader's structural rules before write.
    for cycle_id, record in state["cycles"].items():
        if (not isinstance(record, dict)
                or record.get("ref") != cycle_candidate_ref(cycle_id)
                or FULL_COMMIT_RE.fullmatch(
                    str(record.get("commit", ""))) is None):
            raise TicketCycleStateError(
                "refusing malformed ticket-candidate cycle")
    os.makedirs(MAILBOX, exist_ok=True)
    payload = (json.dumps(state, sort_keys=True, indent=2) + "\n").encode(
        "utf-8")
    if len(payload) > MAX_CANDIDATE_STATE_BYTES:
        raise TicketCycleStateError("ticket-candidate state is too large")
    descriptor, temporary = tempfile.mkstemp(
        prefix=CANDIDATE_STATE_NAME + ".tmp-", dir=MAILBOX)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, candidate_state_path())
        fsync_directory(directory=MAILBOX)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


def git_ref_commit(reference):
    """Return one private ref's full commit, or None when it is absent."""
    result = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["rev-parse", "--verify", "--quiet",
                   reference + "^{commit}"],
        check=False)
    if result.returncode == 1:
        return None
    if result.returncode != 0:
        raise TicketCycleStateError(
            "cannot inspect candidate ref " + reference)
    try:
        commit = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "candidate ref is not ASCII") from exc
    if FULL_COMMIT_RE.fullmatch(commit) is None:
        raise TicketCycleStateError("candidate ref has invalid commit")
    return commit


def candidate_record_locked(cycle_id, ticket_state, candidate_state,
                            recover=True):
    """Return and verify one candidate, adopting an interrupted ref write."""
    active = ticket_state["active"].get(cycle_id)
    record = candidate_state["cycles"].get(cycle_id)
    reference = cycle_candidate_ref(cycle_id=cycle_id)
    ref_commit = git_ref_commit(reference=reference)
    if (ref_commit is not None
            and (record is None or record["commit"] != ref_commit)):
        previous = (cycle_starting_commit(cycle_id)
                    if record is None else record["commit"])
        if (not recover or active is None
                or active["phase"] != "implementation"
                or not git_commit_descends_from(
                    starting_commit=previous,
                    accepted_commit=ref_commit)):
            raise TicketCycleStateError(
                "unowned candidate ref exists for " + cycle_id)
        record = {"ref": reference, "commit": ref_commit}
        candidate_state["cycles"][cycle_id] = record
        write_candidate_state(state=candidate_state)
    if record is None:
        return None
    if ref_commit != record["commit"]:
        raise TicketCycleStateError(
            "candidate state and Git ref disagree for " + cycle_id)
    if (active is None or active["phase"] != "implementation"
            or not git_commit_descends_from(
                starting_commit=cycle_starting_commit(cycle_id),
                accepted_commit=record["commit"])):
        raise TicketCycleStateError(
            "candidate ref does not belong to an active implementation")
    return record


def _clean_worktree_status(worktree):
    """Return exact porcelain bytes without permitting index refresh."""
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "-C", worktree, "status", "--porcelain=v1", "-z",
         "--untracked-files=normal", "--ignore-submodules=none"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env=environment)
    if result.returncode != 0:
        raise TicketCycleStateError(
            "cannot inspect Implementer worktree status")
    return result.stdout


def worktree_head(worktree):
    """Return the exact full commit checked out in one worktree."""
    result = _run_git(
        repository_root=worktree,
        arguments=["rev-parse", "--verify", "HEAD^{commit}"])
    try:
        commit = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError("worktree HEAD is not ASCII") from exc
    if FULL_COMMIT_RE.fullmatch(commit) is None:
        raise TicketCycleStateError("worktree HEAD is not a full commit")
    return commit


def prepare_implementer_cycle_checkout(cycle_id, preserve_current=False):
    """Select the cycle tip, or preserve a validated context checkpoint."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        ticket_state = read_ticket_cycle_state()
        active = ticket_state["active"].get(cycle_id)
        if active is None or active["phase"] != "implementation":
            raise TicketCycleStateError(
                "Implementer checkout has no active implementation cycle")
        candidate_state = read_candidate_state()
        record = candidate_record_locked(
            cycle_id=cycle_id, ticket_state=ticket_state,
            candidate_state=candidate_state)
        target = (record["commit"] if record is not None
                  else cycle_starting_commit(cycle_id))
        worktree = AGENT_CWD["opus"]
        if preserve_current:
            return worktree_head(worktree=worktree)
        if _clean_worktree_status(worktree=worktree):
            raise TicketCycleStateError(
                "Implementer worktree is not clean; refusing to reset it")
        current = worktree_head(worktree=worktree)
        preserved = {item["commit"]
                     for item in candidate_state["cycles"].values()}
        if current != target and current not in preserved:
            try:
                _require_ancestor_or_same(
                    ancestor=current, descendant=target,
                    label="Implementer HEAD is not an ancestor of the "
                          "ticket base")
            except TicketCycleStateError as exc:
                main_commit = _exact_git_object(
                    arguments=["rev-parse", "--verify",
                               "refs/heads/main^{commit}"],
                    label="current main commit")
                if record is not None or current != main_commit:
                    raise TicketCycleStateError(
                        "Implementer HEAD is not a saved candidate, an older "
                        "ticket-base ancestor, or the trusted main baseline; "
                        "refusing to discard " + current) from exc
                _require_ancestor_or_same(
                    ancestor=target, descendant=current,
                    label="ticket base is not an ancestor of the trusted "
                          "main baseline")
        _run_git(
            repository_root=worktree,
            arguments=["reset", "--hard", target])
        if worktree_head(worktree=worktree) != target:
            raise TicketCycleStateError(
                "Implementer reset did not select the requested candidate")
        return target
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def candidate_forbidden_paths(changed_paths, ticket_class="ordinary",
                              contract=ROLE_CONTRACT):
    """Return paths forbidden for this validated ticket class."""
    if ticket_class not in TICKET_CLASSES:
        raise TicketCycleStateError("invalid ticket class")
    forbidden_files = candidate_forbidden_files_from_contract(contract)
    control_plane_files = control_plane_files_from_contract(contract)
    forbidden_prefixes = tuple(
        contract["protected_paths"]["candidate_forbidden_prefixes"])
    return _CANDIDATE_ADMISSION.forbidden_paths(
        changed_paths,
        forbidden_files=forbidden_files,
        control_plane_files=control_plane_files,
        forbidden_prefixes=forbidden_prefixes,
        protected_control_plane=(ticket_class == "protected-control-plane"))


def ticket_class_configuration_problem(ticket_class, skip_redteam=False):
    """Explain why this trusted watcher cannot run one ticket class."""
    if ticket_class not in TICKET_CLASSES:
        return "invalid ticket class"
    if ticket_class == "protected-control-plane":
        return ("protected-control-plane is reserved for Architect-owned "
                "ai/notes administration and cannot dispatch an Implementer; "
                "keep an ai/tools ticket Open for external maintenance")
    return None


def candidate_changed_paths(base_commit, candidate_commit, repository=None):
    """Return every repository path changed from ticket base B to C."""
    if repository is None:
        repository = AGENT_CWD["opus"]
    changed = _run_git(
        repository_root=repository,
        arguments=["diff", "--name-only", "-z", "--no-renames",
                   base_commit, candidate_commit, "--", "."])
    try:
        return {
            item.decode("utf-8", errors="strict")
            for item in changed.stdout.split(b"\0") if item}
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "Implementer candidate contains a non-UTF-8 path") from exc


def classify_candidate_scope(changed_paths, path_scope,
                             ticket_class="ordinary"):
    """Classify C against global protection and its ticket file list."""
    protected = candidate_forbidden_paths(
        changed_paths, ticket_class=ticket_class)
    return _CANDIDATE_ADMISSION.classify(
        changed_paths, path_scope, protected)


def candidate_scope_for_cycle(cycle_id, candidate_commit):
    """Recompute the exact ticket-scope result shown to the Architect."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        record = read_ticket_cycle_state()["active"].get(cycle_id)
        path_scope = None if record is None else record.get("path_scope")
        ticket_class = ("ordinary" if record is None else
                        record.get("ticket_class", "ordinary"))
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)
    # A ticket already running when this field was introduced has no frozen
    # scope. Preserve that one ticket under the earlier Architect-only audit;
    # every newly launched Implementer handoff records the scope below.
    if path_scope is None:
        return None
    changed = candidate_changed_paths(
        base_commit=cycle_starting_commit(cycle_id),
        candidate_commit=candidate_commit)
    result, paths = classify_candidate_scope(
        changed, path_scope, ticket_class=ticket_class)
    return {"result": result, "paths": sorted(paths)}


def record_implementer_candidate(cycle_id, starting_head):
    """Atomically preserve a successful clean Opus commit for its cycle."""
    worktree = AGENT_CWD["opus"]
    if _clean_worktree_status(worktree=worktree):
        raise TicketCycleStateError(
            "successful Implementer turn left an uncommitted worktree")
    candidate = worktree_head(worktree=worktree)
    if candidate == starting_head:
        return None
    if not git_commit_descends_from(
            starting_commit=starting_head, accepted_commit=candidate):
        raise TicketCycleStateError(
            "Implementer result is not a new descendant of its saved base")
    changed_paths = candidate_changed_paths(
        base_commit=cycle_starting_commit(cycle_id),
        candidate_commit=candidate)
    lock_file = acquire_ticket_cycle_lock()
    try:
        ticket_state = read_ticket_cycle_state()
        active = ticket_state["active"].get(cycle_id)
        if active is None or active["phase"] != "implementation":
            raise TicketCycleStateError(
                "candidate commit has no active implementation cycle")
        path_scope = active.get("path_scope")
        ticket_class = active.get("ticket_class", "ordinary")
        scope_result, scope_paths = classify_candidate_scope(
            changed_paths, path_scope or changed_paths,
            ticket_class=ticket_class)
        if scope_result == "PROTECTED_PATH_VIOLATION":
            raise TicketCycleStateError(
                scope_result + ": "
                + ", ".join(repr(path) for path in sorted(scope_paths)))
        candidate_state = read_candidate_state()
        prior = candidate_record_locked(
            cycle_id=cycle_id, ticket_state=ticket_state,
            candidate_state=candidate_state)
        expected = (prior["commit"] if prior is not None else "0" * 40)
        if starting_head != (prior["commit"] if prior is not None
                             else cycle_starting_commit(cycle_id)):
            raise TicketCycleStateError(
                "Implementer result began from another cycle tip")
        if ticket_class == "protected-control-plane":
            # A stale prepared L was built from the prior C. Retire only that
            # private journal before publishing a revised candidate.
            landing_reference = cycle_landing_ref(cycle_id=cycle_id)
            prior_landing = git_ref_commit(reference=landing_reference)
            if prior_landing is not None:
                _run_git(
                    repository_root=AGENT_CWD["fable"],
                    arguments=["update-ref", "-d", landing_reference,
                               prior_landing])
                if git_ref_commit(reference=landing_reference) is not None:
                    raise TicketCycleStateError(
                        "superseded protected landing was not retired")
        reference = cycle_candidate_ref(cycle_id=cycle_id)
        _run_git(
            repository_root=AGENT_CWD["fable"],
            arguments=["update-ref", reference, candidate, expected])
        candidate_state["cycles"][cycle_id] = {
            "ref": reference, "commit": candidate}
        write_candidate_state(state=candidate_state)
        if ticket_class == "protected-control-plane":
            # Every revision is a new immutable C. Neither earlier key nor
            # earlier integration or shadow evidence can authorize it.
            ticket_state["active"][cycle_id] = dict(
                active, control_plane=empty_control_plane_state())
            write_ticket_cycle_state(state=ticket_state)
        if scope_result == "SCOPE_EXCEEDED":
            print("  SCOPE_EXCEEDED; candidate preserved for Architect: "
                  + ", ".join(repr(path) for path in sorted(scope_paths)))
        return candidate
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def candidate_commit_for_cycle(cycle_id):
    """Return the verified immutable candidate for one active cycle."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        ticket_state = read_ticket_cycle_state()
        candidate_state = read_candidate_state()
        record = candidate_record_locked(
            cycle_id=cycle_id, ticket_state=ticket_state,
            candidate_state=candidate_state)
        return None if record is None else record["commit"]
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def write_implementer_delivery_receipt(request_path, return_path):
    """Hard-link a validated role return before its request is archived."""
    request = stable_regular_bytes(
        path=request_path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
        label="Implementer request")
    request_name = os.path.basename(request_path)
    match = PENDING_MESSAGE_RE.fullmatch(request_name)
    request_agent = match.group(1) if match is not None else None
    if request_agent not in {"opus", "fable"}:
        raise TicketCycleStateError(
            "invalid request name for delivery recovery")
    return_raw = stable_regular_bytes(
        path=return_path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
        label="Implementer return")
    return_name = os.path.basename(return_path)
    match = PENDING_MESSAGE_RE.fullmatch(return_name)
    return_agent = match.group(1) if match is not None else None
    if (request_agent, return_agent) not in {
            ("opus", "fable"), ("fable", "daemon"), ("fable", "opus")}:
        raise TicketCycleStateError(
            "invalid delivery-receipt route")
    path = os.path.join(
        MAILBOX, IMPLEMENTER_DELIVERY_PREFIX
        + "@".join((request_name, hashlib.sha256(request).hexdigest(),
                    return_name, hashlib.sha256(return_raw).hexdigest())))
    created = False
    try:
        os.link(return_path, path, follow_symlinks=False)
        created = True
    except FileExistsError:
        pass
    try:
        linked = stable_regular_bytes(
            path=path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="Implementer delivery receipt")
        if linked != return_raw:
            raise TicketCycleStateError(
                "Implementer return changed while its receipt was saved")
        fsync_directory(directory=MAILBOX)
    except BaseException:
        if created:
            os.remove(path)
        raise
    return path


def recover_implementer_deliveries():
    """Finish exact candidate deliveries interrupted after a valid return."""
    pattern = os.path.join(MAILBOX, IMPLEMENTER_DELIVERY_PREFIX + "*")
    recovered = 0
    for receipt_path in sorted(glob.glob(pattern)):
        receipt_name = os.path.basename(receipt_path)
        encoded = receipt_name[len(IMPLEMENTER_DELIVERY_PREFIX):]
        fields = encoded.split("@")
        if len(fields) != 4:
            raise TicketCycleStateError(
                "Implementer delivery receipt has the wrong filename")
        request_name, request_sha256, return_name, return_sha256 = fields
        request_match = PENDING_MESSAGE_RE.fullmatch(request_name)
        return_match = PENDING_MESSAGE_RE.fullmatch(return_name)
        request_agent = (request_match.group(1)
                         if request_match is not None else None)
        return_agent = (return_match.group(1)
                        if return_match is not None else None)
        if ((request_agent, return_agent) not in {
                ("opus", "fable"), ("fable", "daemon"), ("fable", "opus")}
                or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None
                or re.fullmatch(r"[0-9a-f]{64}", return_sha256) is None):
            raise TicketCycleStateError(
                "Implementer delivery receipt has the wrong filename")
        return_paths = [os.path.join(directory, return_name)
                        for directory in (MAILBOX,
                                          os.path.join(MAILBOX, "inflight"),
                                          DONE)
                        if os.path.lexists(os.path.join(
                            directory, return_name))]
        if len(return_paths) != 1:
            raise TicketCycleStateError(
                "validated Implementer return has "
                + str(len(return_paths)) + " mailbox locations")
        return_raw = stable_regular_bytes(
            path=return_paths[0],
            maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="validated Implementer return")
        receipt_raw = stable_regular_bytes(
            path=receipt_path,
            maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="Implementer delivery receipt")
        if (hashlib.sha256(return_raw).hexdigest() != return_sha256
                or hashlib.sha256(receipt_raw).hexdigest() != return_sha256):
            raise TicketCycleStateError(
                "Implementer return changed before delivery recovery")
        inflight = os.path.join(MAILBOX, "inflight", request_name)
        done = os.path.join(DONE, request_name)
        guard = inflight + STATE_GUARD_SUFFIX
        done_inode = regular_inode(path=done)
        inflight_inode = regular_inode(path=inflight)
        if done_inode is not None:
            for leftover in (inflight, guard):
                if (os.path.lexists(leftover)
                        and regular_inode(path=leftover) != done_inode):
                    raise TicketCycleStateError(
                        "interrupted request archive changed identity")
            request_path = done
        elif inflight_inode is not None:
            if (os.path.lexists(guard)
                    and regular_inode(path=guard) != inflight_inode):
                raise TicketCycleStateError(
                    "interrupted request guard changed identity")
            request_path = inflight
        else:
            raise TicketCycleStateError(
                "interrupted Implementer request is missing")
        request_raw = stable_regular_bytes(
            path=request_path,
            maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="interrupted Implementer request")
        if hashlib.sha256(request_raw).hexdigest() != request_sha256:
            raise TicketCycleStateError(
                "Implementer request changed before delivery recovery")
        if done_inode is not None:
            for leftover in (inflight, guard):
                if os.path.lexists(leftover):
                    os.remove(leftover)
            fsync_directory(directory=os.path.dirname(inflight))
        elif os.path.lexists(guard):
            os.remove(guard)
            fsync_directory(directory=os.path.dirname(inflight))
        request_message = read_cycle_message(path=request_path)
        cycle_id, mode, request_body, problem = _ticket_flow_envelope(
            message=request_message)
        if problem is not None:
            raise TicketCycleStateError(problem)
        returned_message = read_cycle_message(path=receipt_path)
        returned_cycle, returned_mode, returned_body, problem = (
            _ticket_flow_envelope(message=returned_message))
        if request_agent == "fable":
            candidate = candidate_commit_for_cycle(cycle_id=cycle_id)
            if (candidate is None
                    or IMPLEMENTER_CANDIDATE_LINE_RE.findall(
                        request_message) != [candidate]):
                raise TicketCycleStateError(
                    "saved Architect audit does not name its exact candidate")
            if return_agent == "daemon":
                returned_cycle, returned_candidate, returned_mode, problem = (
                    _architect_go_request(message=returned_message))
                if (problem is not None or returned_cycle != cycle_id
                        or returned_candidate != candidate
                        or returned_mode != mode):
                    raise TicketCycleStateError(
                        "saved Architect GO does not match its audit")
            else:
                problem = architect_handoff_problem(
                    message=returned_message, cycle_id=cycle_id, mode=mode,
                    checkpoint=is_implementer_checkpoint_request(
                        body=request_body))
                if problem is not None:
                    raise TicketCycleStateError(
                        "saved Architect repair is invalid: "
                        + problem)
            if os.path.dirname(request_path) != os.path.abspath(DONE):
                if not archive_consumed_message(dispatch_path=request_path):
                    raise TicketCycleStateError(
                        "interrupted Architect request could not be archived")
            os.remove(receipt_path)
            fsync_directory(directory=MAILBOX)
            recovered += 1
            print("recovered validated delivery for " + request_name)
            continue
        candidates = (IMPLEMENTER_CANDIDATE_LINE_RE.findall(returned_body)
                      if problem is None else [])
        if (returned_cycle != cycle_id or returned_mode != mode
                or len(candidates) != 1):
            raise TicketCycleStateError(
                "saved Implementer return is not a completed handoff")
        candidate = candidates[0]
        starting_head = candidate_commit_for_cycle(cycle_id=cycle_id)
        if starting_head != candidate:
            if starting_head is None:
                starting_head = cycle_starting_commit(cycle_id=cycle_id)
            if worktree_head(worktree=AGENT_CWD["opus"]) != candidate:
                raise TicketCycleStateError(
                    "Implementer worktree no longer holds the delivered "
                    "candidate")
            if record_implementer_candidate(
                    cycle_id=cycle_id,
                    starting_head=starting_head) != candidate:
                raise TicketCycleStateError(
                    "delivered candidate was not preserved")
        if os.path.dirname(request_path) != os.path.abspath(DONE):
            if not archive_consumed_message(dispatch_path=request_path):
                raise TicketCycleStateError(
                    "interrupted Implementer request could not be archived")
        os.remove(receipt_path)
        fsync_directory(directory=MAILBOX)
        recovered += 1
        print("recovered validated delivery for " + request_name)
    return recovered


def audit_snapshot_path(cycle_id, agent):
    """Return a deterministic managed path for one exact audit checkout."""
    if agent not in {"fable", "sol"}:
        raise ValueError("audit snapshot agent must be fable or sol")
    digest = hashlib.sha256(cycle_id.encode("utf-8")).hexdigest()[:24]
    return os.path.join(
        _managed_primary_root(repository_root=REPO_ROOT),
        AUDIT_WORKTREE_PREFIX + digest + "-" + agent)


def _validate_audit_record(record, path, commit):
    """Prove one registered detached audit worktree names one commit."""
    expected = _managed_child_path(
        path=path,
        managed_root=_managed_primary_root(repository_root=REPO_ROOT))
    if (record is None or "detached" not in record["flags"]
            or "branch" in record or "prunable" in record["flags"]):
        raise PrimaryWorktreeError(
            "audit worktree must be registered and detached: " + expected)
    if git_common_directory(checkout=expected) != git_common_directory(
            checkout=REPO_ROOT):
        raise PrimaryWorktreeError(
            "audit worktree belongs to another repository")
    if worktree_head(worktree=expected) != commit:
        raise PrimaryWorktreeError(
            "audit worktree does not name the exact candidate commit")
    return expected


def create_audit_snapshot(cycle_id, commit, agent):
    """Create or recover a detached exact-commit checkout for one audit."""
    if (not isinstance(commit, str)
            or FULL_COMMIT_RE.fullmatch(commit) is None
            or not git_commit_exists(commit=commit)):
        raise TicketCycleStateError("audit commit is not an exact commit")
    path = audit_snapshot_path(cycle_id=cycle_id, agent=agent)
    lock_file = _open_primary_lock(repository_root=REPO_ROOT)
    try:
        records = registered_worktrees(repository_root=REPO_ROOT)
        record = _record_at_path(records=records, path=path)
        if record is not None:
            return _validate_audit_record(
                record=record, path=path, commit=commit)
        if os.path.lexists(path):
            raise PrimaryWorktreeError(
                "audit path exists without a registered worktree: " + path)
        _run_git(
            repository_root=REPO_ROOT,
            arguments=["worktree", "add", "--detach", path, commit])
        refreshed = registered_worktrees(repository_root=REPO_ROOT)
        created = _record_at_path(records=refreshed, path=path)
        return _validate_audit_record(
            record=created, path=path, commit=commit)
    finally:
        _release_primary_lock(lock_file=lock_file)


def remove_audit_snapshot(cycle_id, commit, agent):
    """Remove only the unchanged disposable snapshot created for this turn."""
    path = audit_snapshot_path(cycle_id=cycle_id, agent=agent)
    lock_file = _open_primary_lock(repository_root=REPO_ROOT)
    try:
        records = registered_worktrees(repository_root=REPO_ROOT)
        record = _record_at_path(records=records, path=path)
        if record is None:
            if os.path.lexists(path):
                raise PrimaryWorktreeError(
                    "unregistered audit path remains: " + path)
            return
        _validate_audit_record(record=record, path=path, commit=commit)
        if _tracked_worktree_changes(worktree=path):
            raise PrimaryWorktreeError(
                "audit changed tracked files; preserving snapshot " + path)
        # Ignored bytecode or test caches are disposable inside this exact,
        # detached, commit-bound checkout. No user or candidate work lives
        # here, so force removes only audit artifacts after the tracked proof.
        _run_git(
            repository_root=REPO_ROOT,
            arguments=["worktree", "remove", "--force", path])
        _run_git(
            repository_root=REPO_ROOT,
            arguments=["worktree", "prune"])
        if (os.path.lexists(path)
                or _record_at_path(
                    records=registered_worktrees(repository_root=REPO_ROOT),
                    path=path) is not None):
            raise PrimaryWorktreeError(
                "audit snapshot removal could not be verified")
    finally:
        _release_primary_lock(lock_file=lock_file)


def discard_interrupted_audit_snapshot(cycle_id, commit, agent):
    """Remove one exact interrupted audit checkout, including its edits."""
    path = audit_snapshot_path(cycle_id=cycle_id, agent=agent)
    lock_file = _open_primary_lock(repository_root=REPO_ROOT)
    try:
        records = registered_worktrees(repository_root=REPO_ROOT)
        record = _record_at_path(records=records, path=path)
        if record is None:
            if os.path.lexists(path):
                raise PrimaryWorktreeError(
                    "unregistered audit path remains: " + path)
            return
        _validate_audit_record(record=record, path=path, commit=commit)
        _run_git(REPO_ROOT, ["worktree", "remove", "--force", path])
        _run_git(REPO_ROOT, ["worktree", "prune"])
        if os.path.lexists(path):
            raise PrimaryWorktreeError(
                "interrupted audit checkout was not removed")
    finally:
        _release_primary_lock(lock_file=lock_file)


def _exact_git_object(arguments, label):
    """Return one full Git object name from a bounded read-only command."""
    try:
        result = _run_git(
            repository_root=AGENT_CWD["fable"],
            arguments=arguments, check=False)
    except PrimaryWorktreeError as exc:
        raise TicketCycleStateError(
            "cannot inspect " + label + ": " + str(exc)) from exc
    if result.returncode != 0:
        detail = result.stderr.decode(
            "utf-8", errors="replace").strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        raise TicketCycleStateError(
            "cannot inspect " + label
            + ((": " + detail) if detail else ""))
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(label + " is not ASCII") from exc
    if FULL_COMMIT_RE.fullmatch(value) is None:
        raise TicketCycleStateError(label + " is not one exact Git object")
    return value


def _single_commit_parent(commit):
    """Return the sole parent of a squash landing commit."""
    try:
        result = _run_git(
            repository_root=AGENT_CWD["fable"],
            arguments=["rev-list", "--parents", "-n", "1", commit],
            check=False)
    except PrimaryWorktreeError as exc:
        raise TicketCycleStateError(
            "cannot inspect landing parents: " + str(exc)) from exc
    if result.returncode != 0:
        raise TicketCycleStateError("cannot inspect landing parents")
    try:
        fields = result.stdout.decode(
            "ascii", errors="strict").strip().split()
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "landing parent record is not ASCII") from exc
    if (len(fields) != 2 or fields[0] != commit
            or FULL_COMMIT_RE.fullmatch(fields[1]) is None):
        raise TicketCycleStateError(
            "Architect landing must be one ordinary commit with one parent")
    return fields[1]


def _require_ancestor_or_same(ancestor, descendant, label):
    """Require ``descendant`` to preserve ``ancestor`` in its lineage."""
    if _commit_is_ancestor(ancestor=ancestor, descendant=descendant,
                           label=label):
        return
    raise TicketCycleStateError(label)


def _commit_is_ancestor(ancestor, descendant, label):
    """Return whether one exact commit remains in another's history."""
    if ancestor == descendant:
        return True
    try:
        result = _run_git(
            repository_root=AGENT_CWD["fable"],
            arguments=["merge-base", "--is-ancestor", ancestor,
                       descendant],
            check=False)
    except PrimaryWorktreeError as exc:
        raise TicketCycleStateError(
            "cannot inspect " + label + ": " + str(exc)) from exc
    if result.returncode not in {0, 1}:
        raise TicketCycleStateError("cannot inspect " + label)
    return result.returncode == 0


STALE_INTEGRATION_REVALIDATION = (
    "STALE — REQUIRES INTEGRATION REVALIDATION")
STALE_INTEGRATION_RE = re.compile(
    re.escape(STALE_INTEGRATION_REVALIDATION)
    + r": C=([0-9a-f]{40}) L=([0-9a-f]{40})"
      r" M0=([0-9a-f]{40}) M1=([0-9a-f]{40})")


def stale_integration_details(problem):
    """Return exact C, L, M0, and M1 from D0's own stale diagnosis."""
    match = STALE_INTEGRATION_RE.search(str(problem))
    if match is None:
        return None
    return {
        "candidate": match.group(1), "stale_landing": match.group(2),
        "old_main": match.group(3), "new_main": match.group(4),
    }


def _prepared_landing_main_problem(candidate_commit, landing_commit,
                                   parent_commit, current_main):
    """Explain why prepared L cannot replace the current main commit."""
    if current_main in {parent_commit, landing_commit}:
        return None
    if _commit_is_ancestor(
            ancestor=landing_commit, descendant=current_main,
            label="whether main already contains prepared landing L"):
        return ("main already contains prepared landing L=" + landing_commit
                + " followed by newer commits; durable-state recovery is "
                  "required, not candidate revalidation")
    if _commit_is_ancestor(
            ancestor=parent_commit, descendant=current_main,
            label="whether main preserves prepared landing parent M0"):
        candidate = (candidate_commit if candidate_commit is not None
                     else "not-applicable")
        return (
            STALE_INTEGRATION_REVALIDATION + ": C=" + candidate
            + " L=" + landing_commit + " M0=" + parent_commit
            + " M1=" + current_main + "; inspect M0-to-M1, its interaction "
              "with C, and the provisional combined result on M1. Repeat "
              "the complete candidate audit only if the intervening change "
              "affects C's assumptions, APIs, tests, numerical behavior, "
              "or dependencies")
    return ("main no longer descends from prepared landing parent M0="
            + parent_commit + "; history requires user reconciliation")


def _exact_squash_tree(parent_commit, candidate_commit):
    """Return the tree made by cleanly squashing candidate onto parent."""
    try:
        result = _run_git(
            repository_root=AGENT_CWD["fable"],
            arguments=["merge-tree", "--write-tree", parent_commit,
                       candidate_commit],
            check=False)
    except PrimaryWorktreeError as exc:
        raise TicketCycleStateError(
            "cannot calculate the candidate squash: " + str(exc)) from exc
    if result.returncode != 0:
        raise TicketCycleStateError(
            "the audited candidate does not squash cleanly onto the "
            "landing parent")
    try:
        tree = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "calculated squash tree is not ASCII") from exc
    if FULL_COMMIT_RE.fullmatch(tree) is None:
        raise TicketCycleStateError(
            "git did not return one exact calculated squash tree")
    return tree


def _tree_with_backlog(tree, backlog):
    """Return ``tree`` with the Architect-sealed backlog bytes."""
    with tempfile.TemporaryDirectory(prefix="mailbox-backlog-index-") as tmp:
        environment = os.environ.copy()
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            environment.pop(name, None)
        environment["GIT_INDEX_FILE"] = os.path.join(tmp, "index")

        def git(arguments, input_bytes=None):
            result = subprocess.run(
                ["git", "-C", AGENT_CWD["fable"]] + arguments,
                env=environment, input=input_bytes, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False)
            if result.returncode != 0:
                raise TicketCycleStateError(
                    "cannot add the sealed backlog to the landing tree")
            return result.stdout

        git(["read-tree", tree])
        blob = git(["hash-object", "-w", "--stdin"], backlog).decode(
            "ascii", errors="strict").strip()
        git(["update-index", "--add", "--cacheinfo", "100644", blob,
             BACKLOG_RELATIVE_PATH])
        result = git(["write-tree"]).decode("ascii", errors="strict").strip()
    if FULL_COMMIT_RE.fullmatch(result) is None:
        raise TicketCycleStateError("Git did not return one backlog tree")
    return result


def _landing_backlog(landing_commit):
    """Read the exact backlog that one prepared landing preserves."""
    result = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["show", landing_commit + ":" + BACKLOG_RELATIVE_PATH],
        check=False)
    if result.returncode != 0 or len(result.stdout) > MAX_BACKLOG_LEDGER_BYTES:
        raise TicketCycleStateError(
            "prepared landing has no valid tracked backlog")
    return result.stdout


def cycle_landing_ref(cycle_id):
    """Return the private crash-journal ref for one prepared landing."""
    return cycle_candidate_ref(cycle_id=cycle_id).rsplit("/", 1)[0] \
        + "/landing"


def _candidate_commit_message(candidate_commit):
    """Return the exact human message that the Architect approved in C.

    Architect GO names the full candidate commit, so the commit message is
    already part of the immutable object under review. The daemon reads that
    message instead of inventing an internal-only subject for the squash
    landing.
    """
    if FULL_COMMIT_RE.fullmatch(candidate_commit) is None:
        raise TicketCycleStateError(
            "candidate message requires one full commit hash")
    result = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["cat-file", "commit", candidate_commit],
        check=False)
    if result.returncode != 0:
        raise TicketCycleStateError(
            "cannot read the approved candidate commit message")
    _headers, separator, message_bytes = result.stdout.partition(b"\n\n")
    if not separator:
        raise TicketCycleStateError(
            "approved candidate commit has no message separator")
    try:
        message = message_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "approved candidate commit message is not UTF-8") from exc
    if not message.strip() or "\0" in message:
        raise TicketCycleStateError(
            "approved candidate commit has no usable human message")
    reserved = re.compile(
        r"^mailbox-(?:cycle|candidate)[ \t]*:", re.IGNORECASE)
    if any(reserved.match(line) for line in message.splitlines()):
        raise TicketCycleStateError(
            "approved candidate commit message uses a reserved mailbox "
            "recovery label")
    return message


def _landing_commit_message(cycle_id, candidate_commit):
    """Copy C's approved message and append exact recovery facts for L."""
    candidate_message = _candidate_commit_message(candidate_commit)
    if candidate_message.endswith("\n\n"):
        separator = ""
    elif candidate_message.endswith("\n"):
        separator = "\n"
    else:
        separator = "\n\n"
    return (
        candidate_message + separator
        + "Mailbox-Cycle: " + cycle_id + "\n"
        + "Mailbox-Candidate: " + candidate_commit + "\n")


def _verify_prepared_landing(cycle_id, candidate_commit, landing_commit,
                             expected_backlog=None):
    """Return L's parent after proving the exact journaled C -> L squash."""
    parent_commit = _single_commit_parent(commit=landing_commit)
    backlog = _landing_backlog(landing_commit=landing_commit)
    if expected_backlog is not None and backlog != expected_backlog:
        raise TicketCycleStateError(
            "prepared landing backlog differs from the Architect seal")
    expected_tree = _tree_with_backlog(
        tree=_exact_squash_tree(
            parent_commit=parent_commit, candidate_commit=candidate_commit),
        backlog=backlog)
    landing_tree = _exact_git_object(
        arguments=["rev-parse", "--verify", landing_commit + "^{tree}"],
        label="prepared landing tree")
    if landing_tree != expected_tree:
        raise TicketCycleStateError(
            "prepared landing tree is not the exact candidate squash")
    result = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["cat-file", "commit", landing_commit],
        check=False)
    if result.returncode != 0:
        raise TicketCycleStateError("cannot inspect prepared landing message")
    _headers, separator, message_bytes = result.stdout.partition(b"\n\n")
    if not separator:
        raise TicketCycleStateError(
            "prepared landing commit has no message separator")
    try:
        message = message_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "prepared landing message is not UTF-8") from exc
    if message != _landing_commit_message(
            cycle_id=cycle_id, candidate_commit=candidate_commit):
        raise TicketCycleStateError(
            "prepared landing message does not bind its cycle and candidate")
    return parent_commit


def prepare_exact_squash_landing(cycle_id, candidate_commit, mode,
                                 sealed_backlog=None):
    """Create or reuse exact L without touching any checkout or branch."""
    tool_changes = sorted(
        path for path in candidate_changed_paths(
            base_commit=cycle_starting_commit(cycle_id),
            candidate_commit=candidate_commit,
            repository=AGENT_CWD["fable"])
        if path.startswith("ai/tools/"))
    if tool_changes:
        raise TicketCycleStateError(
            "external-maintainer-only ai/tools change cannot land: "
            + ", ".join(repr(path) for path in tool_changes))
    lock_file = acquire_ticket_cycle_lock()
    try:
        ticket_state = read_ticket_cycle_state()
        active = ticket_state["active"].get(cycle_id)
        if (active is None or active["phase"] != "implementation"
                or active["mode"] != mode):
            raise TicketCycleStateError(
                "Architect GO has no matching implementation cycle")
        candidate_state = read_candidate_state()
        record = candidate_record_locked(
            cycle_id=cycle_id, ticket_state=ticket_state,
            candidate_state=candidate_state)
        if record is None or record["commit"] != candidate_commit:
            raise TicketCycleStateError(
                "Architect GO does not name the exact saved candidate")
        if sealed_backlog is None:
            sealed_backlog = _validate_sealed_backlog(
                primary_worktree=AGENT_CWD["fable"])
        reference = cycle_landing_ref(cycle_id=cycle_id)
        prepared = git_ref_commit(reference=reference)
        current_main = _exact_git_object(
            arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
            label="current main commit")
        if prepared is not None:
            parent = _verify_prepared_landing(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                landing_commit=prepared, expected_backlog=sealed_backlog)
            main_problem = _prepared_landing_main_problem(
                candidate_commit=candidate_commit,
                landing_commit=prepared, parent_commit=parent,
                current_main=current_main)
            if main_problem is not None:
                raise RetryableArchitectLandingError(main_problem)
            return prepared, parent, reference
        _require_ancestor_or_same(
            ancestor=cycle_starting_commit(cycle_id),
            descendant=current_main,
            label="landing parent does not preserve the cycle base")
        tree = _tree_with_backlog(
            tree=_exact_squash_tree(
                parent_commit=current_main,
                candidate_commit=candidate_commit),
            backlog=sealed_backlog)
        parent_tree = _exact_git_object(
            arguments=["rev-parse", "--verify", current_main + "^{tree}"],
            label="landing parent tree")
        if tree == parent_tree:
            raise TicketCycleStateError(
                "the audited candidate produces an empty squash landing")
        result = _run_git(
            repository_root=AGENT_CWD["fable"],
            arguments=["commit-tree", tree, "-p", current_main,
                       "-F", "-"],
            check=False,
            input_bytes=_landing_commit_message(
                cycle_id=cycle_id,
                candidate_commit=candidate_commit).encode("utf-8"))
        if result.returncode != 0:
            detail = result.stderr.decode(
                "utf-8", errors="replace").strip()[:500]
            raise TicketCycleStateError(
                "cannot create exact squash landing"
                + (": " + detail if detail else ""))
        try:
            landing = result.stdout.decode(
                "ascii", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise TicketCycleStateError(
                "created landing commit is not ASCII") from exc
        if FULL_COMMIT_RE.fullmatch(landing) is None:
            raise TicketCycleStateError(
                "commit-tree did not return one exact landing commit")
        update = _run_git(
            repository_root=AGENT_CWD["fable"],
            arguments=["update-ref", reference, landing, "0" * 40],
            check=False)
        if update.returncode != 0:
            raise TicketCycleStateError(
                "cannot publish the exact landing crash journal")
        if git_ref_commit(reference=reference) != landing:
            raise TicketCycleStateError(
                "landing crash journal did not preserve the created commit")
        parent = _verify_prepared_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=landing, expected_backlog=sealed_backlog)
        return landing, parent, reference
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def _user_checkout_status():
    """Return exact tracked/untracked status without refreshing the index."""
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "-C", REPO_ROOT, "status", "--porcelain=v1", "-z",
         "--untracked-files=normal", "--ignore-submodules=none"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env=environment)
    if result.returncode != 0:
        raise TicketCycleStateError("cannot inspect the user's main checkout")
    return result.stdout


def land_prepared_commit_in_clean_user_checkout(
        landing, parent, candidate_commit=None):
    """Fast-forward a clean attached main checkout; never reset or force."""
    symbolic = _run_git(
        repository_root=REPO_ROOT,
        arguments=["symbolic-ref", "-q", "HEAD"], check=False)
    if (symbolic.returncode != 0
            or symbolic.stdout.decode("utf-8", errors="replace").strip()
            != "refs/heads/main"):
        raise TicketCycleStateError(
            "the user's checkout is not attached to local main")
    current = _exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    if current == landing:
        if _user_checkout_status():
            raise RetryableArchitectLandingError(
                "local main reached the prepared landing but the user's "
                "checkout is not clean")
        return
    main_problem = _prepared_landing_main_problem(
        candidate_commit=candidate_commit, landing_commit=landing,
        parent_commit=parent, current_main=current)
    if main_problem is not None:
        raise RetryableArchitectLandingError(main_problem)
    if _user_checkout_status():
        raise RetryableArchitectLandingError(
            "the user's main checkout has staged, unstaged, or untracked "
            "work; exact landing was preserved without touching it")
    result = _run_git(
        repository_root=REPO_ROOT,
        arguments=["merge", "--ff-only", landing], check=False)
    if result.returncode != 0:
        raise TicketCycleStateError(
            "clean user main could not fast-forward to the prepared landing")
    after = _exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="landed main commit")
    if after != landing or _user_checkout_status():
        raise TicketCycleStateError(
            "local main did not verify as one clean exact landing")


def _push_debt_path(landing):
    return os.path.join(RELAY_DIR, "pending-main-push-" + landing + ".txt")


def write_push_debt(landing, detail):
    """Durably record that exact local L still needs remote verification."""
    os.makedirs(RELAY_DIR, exist_ok=True)
    debt = _push_debt_path(landing=landing)
    payload = (
        "Local main contains verified landing " + landing + ".\n"
        "Push is still required: git push origin " + landing
        + ":refs/heads/main\n"
        "Last push result: " + detail + "\n")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".pending-main-push-", dir=RELAY_DIR)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) \
                as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, debt)
        fsync_directory(directory=RELAY_DIR)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass
    return debt


def push_exact_landing_or_record_debt(landing):
    """Attempt one non-force push; preserve a durable user action on failure."""
    command = ["git", "-C", AGENT_CWD["fable"], "push", "--porcelain",
               "origin", landing + ":refs/heads/main"]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=120)
        pushed = result.returncode == 0
        detail = (result.stderr + result.stdout).decode(
            "utf-8", errors="replace").strip()[:2000]
    except (OSError, subprocess.TimeoutExpired) as exc:
        pushed = False
        detail = str(exc)
    if pushed:
        try:
            verify = subprocess.run(
                ["git", "-C", AGENT_CWD["fable"], "ls-remote", "--refs",
                 "origin", "refs/heads/main"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=120)
            try:
                fields = verify.stdout.decode(
                    "ascii", errors="strict").strip().split()
            except UnicodeDecodeError:
                fields = []
            verified = (verify.returncode == 0
                        and fields == [landing, "refs/heads/main"])
            remote_detail = (verify.stderr + verify.stdout).decode(
                "utf-8", errors="replace").strip()[:2000]
        except (OSError, subprocess.TimeoutExpired) as exc:
            verified = False
            remote_detail = str(exc)
        if not verified:
            pushed = False
            detail = (detail + "\nremote verification: " + remote_detail) \
                .strip()
    debt = _push_debt_path(landing=landing)
    if pushed:
        try:
            os.remove(debt)
        except FileNotFoundError:
            pass
        return True, ""
    write_push_debt(landing=landing, detail=detail)
    return False, detail


def retire_cycle_landing_ref(cycle_id, landing):
    """Retire only the exact crash-journal ref after receipt archival."""
    reference = cycle_landing_ref(cycle_id=cycle_id)
    current = git_ref_commit(reference=reference)
    if current is None:
        return
    if current != landing:
        raise TicketCycleStateError(
            "landing crash journal changed before retirement")
    _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["update-ref", "-d", reference, landing])


def recorded_landing_for_architect_go(cycle_id, mode):
    """Return durable L after a prior partial consume, or ``None``."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        completed = state["completed"].get(cycle_id)
        if completed is not None:
            return completed
        current = state["active"].get(cycle_id)
        if current is None:
            raise TicketCycleStateError(
                "Architect GO has no active or completed ticket cycle")
        if current["mode"] != mode:
            raise TicketCycleStateError(
                "Architect GO changed the ticket's saved mode")
        if current["phase"] == "implementation":
            return None
        if (current["phase"] in {
                "committed-awaiting-closure", "awaiting-redteam"}
                and current["commit"] is not None):
            return current["commit"]
        raise TicketCycleStateError(
            "Architect GO found an unsupported ticket-cycle phase")
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def redteam_closure_request_payload(cycle_id, landing):
    """Build the daemon-owned advisory review request for exact L."""
    return sol_ticket_payload(
        ticket_kind="closure", review_cycle=cycle_id,
        review_commit=landing,
        text=(
            "Review the exact daemon-created landing commit " + landing
            + " for ticket " + cycle_id + ". Focus on this ticket and the "
            "behavior directly affected by its landing. Return the exact "
            "correlated NO CHANGE or REOPEN receipt; this review is "
            "advisory and does not undo the Architect's local landing."))


def control_plane_review_request_payload(cycle_id, candidate):
    """Build D0's mandatory pre-landing Red Team request for exact C."""
    return sol_ticket_payload(
        ticket_kind="control-plane", review_cycle=cycle_id,
        review_commit=candidate,
        text=(
            "Review exact protected control-plane candidate " + candidate
            + " for ticket " + cycle_id + ". D0 has recorded Architect "
              "GO for this immutable candidate, but no landing exists. "
              "Inspect the bounded control-plane change adversarially. "
              "Return exactly one redteam-control-plane receipt addressed "
              "to daemon with ACCEPT-CONTROL-PLANE or "
              "REJECT-CONTROL-PLANE. You cannot land the change."))


def matching_control_plane_review_request(cycle_id, candidate):
    """Return the sole saved mandatory review request, when present."""
    matches = []
    conflicts = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-sol.md"),
                          recursive=True):
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            continue
        if sol_ticket_kind(message=message) != "control-plane":
            continue
        found_cycle, found_candidate, _body, problem = (
            _redteam_control_plane_envelope(message=message))
        if found_cycle != cycle_id:
            continue
        if problem is None and found_candidate == candidate:
            matches.append(path)
        else:
            conflicts.append(path)
    if conflicts or len(matches) > 1:
        raise TicketCycleStateError(
            "control-plane review identity conflicts with saved work")
    return matches[0] if matches else None


def publish_control_plane_review_request(cycle_id, candidate):
    """Publish once after D0 has durably recorded Architect GO(C)."""
    existing = matching_control_plane_review_request(
        cycle_id=cycle_id, candidate=candidate)
    if existing is not None:
        return existing
    lock_file = acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise RetryableArchitectLandingError(
            "cannot lock mailbox for protected Red Team review")
    try:
        existing = matching_control_plane_review_request(
            cycle_id=cycle_id, candidate=candidate)
        if existing is not None:
            return existing
        path = publish_message_locked(
            agent="sol", payload=control_plane_review_request_payload(
                cycle_id=cycle_id, candidate=candidate))
        if path is None:
            raise RetryableArchitectLandingError(
                "could not publish protected Red Team review")
        return path
    finally:
        release_mailbox_sequence_lock(lock_file=lock_file)


def publish_control_plane_repair_request(cycle_id, candidate, mode):
    """Return a rejected protected C to the Architect exactly once."""
    marker = "CONTROL-PLANE-REPAIR: " + candidate
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            continue
        found_cycle, found_mode, body, problem = _ticket_flow_envelope(
            message=message)
        if (problem is None and found_cycle == cycle_id
                and found_mode == mode and marker in body):
            return path
    lock_file = acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise RetryableArchitectLandingError(
            "cannot lock mailbox for protected repair return")
    try:
        payload = (MAILBOX_FLOW_HEADER + "ticket\n"
                   + MAILBOX_CYCLE_HEADER + cycle_id + "\n"
                   + MAILBOX_MODE_HEADER + mode + "\n\n"
                   + marker + "\n\n"
                   + "The mandatory pre-landing Red Team review rejected "
                     "this exact candidate. Read its saved evidence, reopen "
                     "the ticket, and send one same-cycle Implementer repair "
                     "handoff. Do not send another GO for this candidate.\n")
        path = publish_message_locked(agent="fable", payload=payload)
        if path is None:
            raise RetryableArchitectLandingError(
                "could not publish protected repair return")
        return path
    finally:
        release_mailbox_sequence_lock(lock_file=lock_file)


def control_plane_integration_request_payload(
        cycle_id, candidate, stale_landing, old_main, new_main, mode):
    """Build the same-cycle Architect check for one moved main branch."""
    for label, value in (("candidate", candidate),
                         ("stale landing", stale_landing),
                         ("old main", old_main), ("new main", new_main)):
        if FULL_COMMIT_RE.fullmatch(value) is None:
            raise ValueError("invalid " + label + " commit")
    if CYCLE_ID_RE.fullmatch(cycle_id) is None:
        raise ValueError("invalid integration-revalidation cycle")
    if mode not in ARCHITECT_COMMIT_MODES:
        raise ValueError("invalid integration-revalidation mode")
    return (
        MAILBOX_FLOW_HEADER + "ticket\n"
        + MAILBOX_CYCLE_HEADER + cycle_id + "\n"
        + MAILBOX_MODE_HEADER + mode + "\n\n"
        + "CONTROL-PLANE-INTEGRATION: REVALIDATE\n"
        + "INTEGRATION-CANDIDATE: " + candidate + "\n"
        + "STALE-LANDING: " + stale_landing + "\n"
        + "OLD-MAIN: " + old_main + "\n"
        + "NEW-MAIN: " + new_main + "\n\n"
        + "- **Candidate commit:** `" + candidate + "`\n\n"
        + "Main advanced after the protected landing was prepared. Audit "
          "only the interaction of OLD-MAIN to NEW-MAIN with exact C. "
          "Inspect the provisional combined result and rerun every newly "
          "relevant acceptance check. The earlier Architect and Red Team "
          "approvals remain bound to C. If the integration is still safe, "
          "return the ordinary exact architect-go receipt for C. Otherwise "
          "return one same-cycle Implementer repair handoff.\n")


def control_plane_integration_request(message):
    """Parse the daemon-owned M0-to-M1 revalidation request."""
    cycle_id, mode, body, problem = _ticket_flow_envelope(message=message)
    if problem is not None or not body.startswith(
            "CONTROL-PLANE-INTEGRATION: REVALIDATE\n"):
        return None
    match = re.match(
        r"\ACONTROL-PLANE-INTEGRATION: REVALIDATE\r?\n"
        r"INTEGRATION-CANDIDATE: ([0-9a-f]{40})\r?\n"
        r"STALE-LANDING: ([0-9a-f]{40})\r?\n"
        r"OLD-MAIN: ([0-9a-f]{40})\r?\n"
        r"NEW-MAIN: ([0-9a-f]{40})\r?\n\r?\n",
        body)
    if match is None:
        raise TicketCycleStateError(
            "control-plane integration request has malformed identities")
    candidate, landing, old_main, new_main = match.groups()
    if IMPLEMENTER_CANDIDATE_LINE_RE.findall(body) != [candidate]:
        raise TicketCycleStateError(
            "control-plane integration request does not bind exact C")
    return {
        "cycle_id": cycle_id, "mode": mode, "candidate": candidate,
        "stale_landing": landing, "old_main": old_main,
        "new_main": new_main,
    }


def matching_control_plane_integration_request(
        cycle_id, candidate, stale_landing, old_main, new_main):
    """Return one already-published request for the same stale event."""
    expected = (cycle_id, candidate, stale_landing, old_main, new_main)
    matches = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        try:
            parsed = control_plane_integration_request(
                read_cycle_message(path=path))
        except (OSError, ValueError, TicketCycleStateError):
            continue
        if parsed is None:
            continue
        found = (parsed["cycle_id"], parsed["candidate"],
                 parsed["stale_landing"], parsed["old_main"],
                 parsed["new_main"])
        if found == expected:
            matches.append(path)
    if len(matches) > 1:
        raise TicketCycleStateError(
            "more than one integration request names the same stale event")
    return matches[0] if matches else None


def publish_control_plane_integration_request(
        cycle_id, candidate, stale_landing, old_main, new_main, mode):
    """Publish exactly one autonomous Architect integration audit."""
    arguments = dict(
        cycle_id=cycle_id, candidate=candidate,
        stale_landing=stale_landing, old_main=old_main, new_main=new_main)
    existing = matching_control_plane_integration_request(**arguments)
    if existing is not None:
        return existing
    lock_file = acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise RetryableArchitectLandingError(
            "cannot lock mailbox for integration revalidation")
    try:
        existing = matching_control_plane_integration_request(**arguments)
        if existing is not None:
            return existing
        path = publish_message_locked(
            agent="fable", payload=control_plane_integration_request_payload(
                mode=mode, **arguments))
        if path is None:
            raise RetryableArchitectLandingError(
                "could not publish integration revalidation")
        return path
    finally:
        release_mailbox_sequence_lock(lock_file=lock_file)


def matching_redteam_closure_request(cycle_id, landing):
    """Return the sole saved Sol closure request, if one already exists."""
    matches = []
    conflicts = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-sol.md"),
                          recursive=True):
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            continue
        if sol_ticket_kind(message=message) != "closure":
            continue
        returned_cycle, returned_landing, _body, problem = (
            _redteam_closure_envelope(message=message))
        if returned_cycle != cycle_id:
            continue
        if problem is not None or returned_landing != landing:
            conflicts.append(path)
        else:
            matches.append(path)
    if conflicts:
        raise TicketCycleStateError(
            "another Sol closure request uses this cycle with a different "
            "or malformed landing")
    if len(matches) > 1:
        raise TicketCycleStateError(
            "more than one Sol closure request names this cycle and landing")
    return matches[0] if matches else None


def publish_redteam_closure_request(cycle_id, landing):
    """Publish or recover the one normal-mode Sol review of exact L."""
    existing = matching_redteam_closure_request(
        cycle_id=cycle_id, landing=landing)
    if existing is not None:
        return existing
    lock_file = acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise RetryableArchitectLandingError(
            "cannot lock the mailbox sequence for the Red Team request")
    try:
        existing = matching_redteam_closure_request(
            cycle_id=cycle_id, landing=landing)
        if existing is not None:
            return existing
        path = publish_message_locked(
            agent="sol",
            payload=redteam_closure_request_payload(
                cycle_id=cycle_id, landing=landing))
        if path is None:
            raise RetryableArchitectLandingError(
                "could not publish the Red Team request after 20 attempts")
        return path
    finally:
        release_mailbox_sequence_lock(lock_file=lock_file)


def control_plane_ticket_state(cycle_id, candidate_commit=None):
    """Return a copy of one protected ticket's durable state."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        if active is None or active.get("ticket_class", "ordinary") != (
                "protected-control-plane"):
            return None
        if candidate_commit is not None:
            saved = read_candidate_state()["cycles"].get(cycle_id)
            if saved is None or saved["commit"] != candidate_commit:
                raise TicketCycleStateError(
                    "protected decision does not name saved candidate C")
        return dict(active["control_plane"])
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def require_validated_architect_go_receipt(cycle_id, candidate_commit):
    """Require D0's saved proof that one Architect turn produced GO(C)."""
    matches = []
    pattern = os.path.join(MAILBOX, IMPLEMENTER_DELIVERY_PREFIX + "*")
    for path in glob.glob(pattern):
        fields = os.path.basename(path)[len(IMPLEMENTER_DELIVERY_PREFIX):] \
            .split("@")
        if len(fields) != 4:
            continue
        request_name, request_digest, return_name, return_digest = fields
        request_match = PENDING_MESSAGE_RE.fullmatch(request_name)
        return_match = PENDING_MESSAGE_RE.fullmatch(return_name)
        if (request_match is None or request_match.group(1) != "fable"
                or return_match is None
                or return_match.group(1) != "daemon"):
            continue
        try:
            raw = stable_regular_bytes(
                path=path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="validated Architect GO receipt")
            message = raw.decode("utf-8", errors="strict")
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        found_cycle, found_candidate, _mode, problem = (
            _architect_go_request(message=message))
        if (problem is None and found_cycle == cycle_id
                and found_candidate == candidate_commit
                and hashlib.sha256(raw).hexdigest() == return_digest
                and re.fullmatch(r"[0-9a-f]{64}", request_digest)):
            matches.append(path)
    if len(matches) != 1:
        raise TicketCycleStateError(
            "protected Architect GO lacks exactly one D0-validated "
            "Architect-turn delivery receipt")


def record_control_plane_architect_go(cycle_id, candidate_commit):
    """Persist the first key before publishing mandatory Red Team work."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        saved = read_candidate_state()["cycles"].get(cycle_id)
        if (active is None or active["phase"] != "implementation"
                or active.get("ticket_class") != "protected-control-plane"
                or saved is None or saved["commit"] != candidate_commit):
            raise TicketCycleStateError(
                "Architect protected GO does not name active candidate C")
        control = dict(active["control_plane"])
        prior = control["architect_candidate"]
        if prior == candidate_commit:
            return
        if prior is not None and prior != candidate_commit:
            raise TicketCycleStateError(
                "protected Architect decision changed candidate C")
        # The receipt is created by D0 only after it validates the fresh
        # Architect outcome. Check it while holding the state lock, then save
        # the decision so later recovery no longer depends on the short-lived
        # delivery hard link.
        require_validated_architect_go_receipt(
            cycle_id=cycle_id, candidate_commit=candidate_commit)
        control["architect_candidate"] = candidate_commit
        state["active"][cycle_id] = dict(active, control_plane=control)
        write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def record_control_plane_redteam_decision(cycle_id, candidate_commit,
                                          decision):
    """Persist the second exact key; it grants no landing by itself."""
    if decision not in CONTROL_PLANE_REVIEW_RESULTS:
        raise TicketCycleStateError("invalid protected Red Team decision")
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        saved = read_candidate_state()["cycles"].get(cycle_id)
        if (active is None or active["phase"] != "implementation"
                or active.get("ticket_class") != "protected-control-plane"
                or saved is None or saved["commit"] != candidate_commit):
            raise TicketCycleStateError(
                "Red Team protected decision does not name active C")
        control = dict(active["control_plane"])
        prior = control["redteam_result"]
        if (prior is not None
                and (prior != decision
                     or control["redteam_candidate"] != candidate_commit)):
            raise TicketCycleStateError(
                "protected Red Team decision changed identity")
        control["redteam_result"] = decision
        control["redteam_candidate"] = candidate_commit
        state["active"][cycle_id] = dict(active, control_plane=control)
        write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def record_control_plane_integration_stale(
        cycle_id, candidate_commit, stale_landing, old_main, new_main):
    """Preserve both C approvals while recording that prepared L is stale."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        if (active is None
                or active.get("ticket_class") != "protected-control-plane"
                or active["phase"] != "implementation"):
            raise TicketCycleStateError(
                "stale integration has no active protected ticket")
        control = dict(active["control_plane"])
        if not control_plane_keys_ready(
                control=control, candidate_commit=candidate_commit):
            raise TicketCycleStateError(
                "stale integration did not preserve both exact-C approvals")
        control.update({
            "integration_status": "STALE",
            "integration_main": new_main,
            "stale_landing": stale_landing,
            "stale_parent": old_main,
            "integration_evidence": None,
            # C passed the first shadow, but the replacement landing must be
            # checked as the exact combined tree on M1.
            "shadow_status": None,
            "shadow_evidence": None,
        })
        state["active"][cycle_id] = dict(active, control_plane=control)
        write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def record_control_plane_integration_go(
        cycle_id, candidate_commit, new_main, evidence):
    """Record a fresh Architect GO for the exact C-on-M1 interaction."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        if (active is None
                or active.get("ticket_class") != "protected-control-plane"
                or active["phase"] != "implementation"):
            raise TicketCycleStateError(
                "integration GO has no active protected ticket")
        control = dict(active["control_plane"])
        if (not control_plane_keys_ready(
                control=control, candidate_commit=candidate_commit)
                or control["integration_status"] != "STALE"
                or control["integration_main"] != new_main):
            raise TicketCycleStateError(
                "integration GO changed C, M1, or either approval")
        current_main = _exact_git_object(
            arguments=["rev-parse", "--verify",
                       "refs/heads/main^{commit}"],
            label="main at integration revalidation")
        if current_main != new_main:
            raise TicketCycleStateError(
                "main advanced again before integration GO was recorded")
        control["integration_status"] = "REVALIDATED"
        control["integration_evidence"] = evidence
        state["active"][cycle_id] = dict(active, control_plane=control)
        write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def prepare_revalidated_control_plane_landing(cycle_id, candidate_commit):
    """Retire only stale L after proving main still equals approved M1."""
    control = control_plane_ticket_state(
        cycle_id=cycle_id, candidate_commit=candidate_commit)
    if control is None or control["integration_status"] != "REVALIDATED":
        return
    current_main = _exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="main before revalidated protected landing")
    approved_main = control["integration_main"]
    stale_landing = control["stale_landing"]
    old_main = control["stale_parent"]
    reference = cycle_landing_ref(cycle_id=cycle_id)
    journaled = git_ref_commit(reference=reference)
    if current_main != approved_main:
        # A crash may have occurred after D0 replaced old L with a new
        # provisional landing on the last revalidated main. Bind the next
        # stale event to the landing actually in the private journal, not to
        # an older L that is no longer retryable.
        if journaled is not None and journaled != stale_landing:
            stale_landing = journaled
            old_main = _verify_prepared_landing(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                landing_commit=journaled)
        problem = _prepared_landing_main_problem(
            candidate_commit=candidate_commit,
            landing_commit=stale_landing, parent_commit=old_main,
            current_main=current_main)
        raise RetryableArchitectLandingError(
            problem or "main changed after integration revalidation")
    if journaled is None:
        return
    if journaled != stale_landing:
        parent = _verify_prepared_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=journaled)
        if parent != approved_main:
            raise TicketCycleStateError(
                "replacement landing journal has an unapproved parent")
        return
    _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["update-ref", "-d", reference, stale_landing])
    if git_ref_commit(reference=reference) is not None:
        raise TicketCycleStateError("stale landing journal was not retired")


def protected_landing_ready(cycle_id, candidate_commit):
    """Require both independently persisted decisions for exact C."""
    control = control_plane_ticket_state(
        cycle_id=cycle_id, candidate_commit=candidate_commit)
    if control is None:
        return True
    return control_plane_keys_ready(
        control=control, candidate_commit=candidate_commit)


def control_plane_keys_ready(control, candidate_commit):
    """Pure exact-C two-key decision used by D0 and focused tests."""
    return (isinstance(control, dict)
            and FULL_COMMIT_RE.fullmatch(candidate_commit) is not None
            and control.get("architect_candidate") == candidate_commit
            and control.get("redteam_candidate") == candidate_commit
            and control.get("redteam_result") == "ACCEPT-CONTROL-PLANE")


def control_plane_redteam_key_matches(control, candidate_commit, decision):
    """Return whether D0 already saved this exact Sol decision."""
    return (isinstance(control, dict)
            and decision in CONTROL_PLANE_REVIEW_RESULTS
            and control.get("redteam_candidate") == candidate_commit
            and control.get("redteam_result") == decision)


def _live_control_plane_fingerprint():
    """Hash D0's live state and trusted refs around a shadow run."""
    digest = hashlib.sha256()
    for name in (TICKET_CYCLE_STATE_NAME, CANDIDATE_STATE_NAME):
        path = os.path.join(MAILBOX, name)
        digest.update(name.encode("utf-8") + b"\0")
        try:
            raw = stable_regular_bytes(
                path=path, maximum_bytes=MAX_TICKET_CYCLE_STATE_BYTES,
                label="live control-plane state", missing_ok=True)
        except (OSError, ValueError) as exc:
            raise TicketCycleStateError(str(exc)) from exc
        digest.update(b"<missing>" if raw is None else raw)
    if ACTIVE_TOPOLOGY is not None:
        for name in ("primary_state", "implementer_state", "sol_state"):
            path = ACTIVE_TOPOLOGY[name]
            digest.update(path.encode("utf-8") + b"\0")
            try:
                raw = stable_regular_bytes(
                    path=path, maximum_bytes=MAX_PRIMARY_STATE_BYTES,
                    label="live control-plane topology")
            except (OSError, ValueError) as exc:
                raise TicketCycleStateError(str(exc)) from exc
            digest.update(raw)
    refs = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["for-each-ref", "--format=%(refname) %(objectname)",
                   CANDIDATE_REF_ROOT, "refs/heads/main"])
    digest.update(refs.stdout)
    return digest.hexdigest()


def trusted_control_plane_check(commit, label):
    """Run D1 in a standalone temporary repository under D0's driver.

    This is protocol isolation, not a hostile-process sandbox.  D1 receives
    no path to the live mailbox or Git common directory. D0 verifies that its
    own state and refs are byte-identical after the bounded checks.
    """
    if FULL_COMMIT_RE.fullmatch(commit) is None:
        raise TicketCycleStateError("control-plane check needs a full commit")
    before = _live_control_plane_fingerprint()
    os.makedirs(RELAY_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = os.path.join(
        RELAY_DIR, stamp + "-control-plane-" + label + ".log")
    commands = []
    with tempfile.TemporaryDirectory(prefix="mailbox-control-plane-") as root:
        repository = os.path.join(root, "repo")
        commands.append(["git", "init", "--quiet", repository])
        commands.append([
            "git", "-C", repository, "fetch", "--quiet", "--no-tags",
            AGENT_CWD["fable"], commit])
        commands.append([
            "git", "-C", repository, "checkout", "--quiet", "--detach",
            "FETCH_HEAD"])
        # This program is D0's harness. It is generated outside the candidate
        # checkout, while every imported function below comes from D1 at C.
        # Candidate tests are intentionally not imported or trusted here.
        probe = """
import os
import subprocess
import sys
shadow_repository = os.path.realpath(os.getcwd())
from ai.tools import handoff_contract as h
from ai.tools import mailbox_daemon as d
from ai.tools.role_contract import ROLE_CONTRACT

assert os.path.realpath(d.REPO_ROOT) == shadow_repository

base = (
    '- Roles: `Architect + Implementer + Red Team`\\n'
    '- Discovery severity: `medium`\\n'
    '- Review scope: `bounded`\\n')
assert h._require_architect_role_plan(
    base + '- Ticket class: `ordinary`')['ticket_class'] == 'ordinary'
try:
    h._require_architect_role_plan(
        base + '- Ticket class: `protected-control-plane`')
except h.DirectiveError as exc:
    assert 'reserved for Architect-owned ai/notes administration' in str(exc)
else:
    raise AssertionError('protected-control-plane plan was accepted')
assert d.ticket_class_configuration_problem('ordinary', True) is None
for skip_redteam in (False, True):
    problem = d.ticket_class_configuration_problem(
        'protected-control-plane', skip_redteam)
    assert 'Architect-owned ai/notes administration' in problem
    assert 'ticket Open' in problem

tool = ROLE_CONTRACT['protected_paths']['trusted_tools']['mailbox_daemon']
result, paths = d.classify_candidate_scope(
    {tool}, {tool}, ticket_class='ordinary')
assert result == 'PROTECTED_PATH_VIOLATION' and paths == {tool}
result, paths = d.classify_candidate_scope(
    {tool}, {tool}, ticket_class='protected-control-plane')
assert result == 'PROTECTED_PATH_VIOLATION' and paths == {tool}
other = 'emulator/unplanned.py'
result, paths = d.classify_candidate_scope(
    {other}, {tool}, ticket_class='protected-control-plane')
assert result == 'SCOPE_EXCEEDED' and paths == {other}

cycle = 'protected-shadow@' + '1' * 40
c1, c2 = '2' * 40, '3' * 40
request = d.control_plane_review_request_payload(cycle, c1)
found_cycle, found_candidate, _body, problem = (
    d._redteam_control_plane_envelope(request))
assert problem is None and (found_cycle, found_candidate) == (cycle, c1)
receipt = d.control_plane_review_receipt_payload(
    cycle, c1, 'ACCEPT-CONTROL-PLANE', 'accepted')
found_cycle, found_candidate, result, _body, problem = (
    d._control_plane_review_receipt(receipt))
assert problem is None
assert (found_cycle, found_candidate, result) == (
    cycle, c1, 'ACCEPT-CONTROL-PLANE')

control = d.empty_control_plane_state()
assert not d.control_plane_keys_ready(control, c1)
control['architect_candidate'] = c1
assert not d.control_plane_keys_ready(control, c1)
control['redteam_candidate'] = c2
control['redteam_result'] = 'ACCEPT-CONTROL-PLANE'
assert not d.control_plane_keys_ready(control, c1)
control['redteam_candidate'] = c1
control['redteam_result'] = 'REJECT-CONTROL-PLANE'
assert not d.control_plane_keys_ready(control, c1)
control['redteam_result'] = 'ACCEPT-CONTROL-PLANE'
assert d.control_plane_keys_ready(control, c1)
assert not d.control_plane_keys_ready(control, c2)

state = d.empty_ticket_cycle_state()
state['active'][cycle] = {
    'phase': 'implementation', 'commit': None, 'mode': 'normal',
    'route': 'primary', 'ticket_class': 'protected-control-plane',
    'path_scope': [tool], 'control_plane': control}
normalized = d.validate_ticket_cycle_state(state)
assert normalized['active'][cycle]['control_plane'] == control

os.makedirs(d.MAILBOX, exist_ok=True)
d._bridge_local_sealed_backlog(shadow_repository)
owner = d.acquire_dispatch_lock(mode='once')
assert owner is not None
try:
    assert d.acquire_dispatch_lock(mode='once') is None
finally:
    d.release_dispatch_lock(owner)

# Drive the real D1 state and landing functions from this D0-owned program.
# Every Git object, state file, and journal below belongs to the disposable
# candidate checkout. The outer D0 process separately fingerprints the live
# state and refs before and after this child exits.
os.environ['GIT_AUTHOR_NAME'] = 'D0 shadow harness'
os.environ['GIT_AUTHOR_EMAIL'] = 'shadow@example.invalid'
os.environ['GIT_COMMITTER_NAME'] = 'D0 shadow harness'
os.environ['GIT_COMMITTER_EMAIL'] = 'shadow@example.invalid'

def git_result(arguments, stdin=None):
    return subprocess.run(
        ['git', '-C', shadow_repository] + list(arguments),
        input=stdin, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)

def git(arguments, stdin=None):
    result = git_result(arguments, stdin)
    assert result.returncode == 0, (arguments, result.stderr)
    return result.stdout.strip()

def ref_or_none(reference):
    result = git_result(['rev-parse', '--verify', '--quiet',
                         reference + '^{commit}'])
    assert result.returncode in (0, 1), result.stderr
    return result.stdout.strip() if result.returncode == 0 else None

def child_commit(parent, path, content, message):
    git(['read-tree', parent])
    blob = git(['hash-object', '-w', '--stdin'], content)
    git(['update-index', '--add', '--cacheinfo', '100644', blob, path])
    tree = git(['write-tree'])
    commit = git(['commit-tree', tree, '-p', parent], message + '\\n')
    git(['read-tree', 'HEAD'])
    return commit

base_commit = git(['rev-parse', 'HEAD'])
git(['update-ref', 'refs/heads/main', base_commit])
candidate = child_commit(
    base_commit, 'shadow-candidate.txt', 'candidate\\n',
    'shadow candidate')
new_main = child_commit(
    base_commit, 'shadow-main.txt', 'new main\\n',
    'concurrent main')
other_candidate = new_main
cycle = 'protected-shadow-landing@' + base_commit
other_cycle = 'protected-shadow-other@' + base_commit
candidate_ref = d.cycle_candidate_ref(cycle)
landing_ref = d.cycle_landing_ref(cycle)
git(['update-ref', candidate_ref, candidate, '0' * 40])

candidate_state = d.empty_candidate_state()
candidate_state['cycles'][cycle] = {
    'ref': candidate_ref, 'commit': candidate}
d.write_candidate_state(candidate_state)

def save_control(control):
    state = d.empty_ticket_cycle_state()
    state['active'][cycle] = {
        'phase': 'implementation', 'commit': None, 'mode': 'normal',
        'route': 'primary', 'ticket_class': 'protected-control-plane',
        'path_scope': [tool], 'control_plane': control}
    d.write_ticket_cycle_state(state)

def landing_must_be_blocked(label):
    assert ref_or_none(landing_ref) is None, label
    try:
        d.execute_architect_go_locked(cycle, candidate, 'normal')
    except d.TicketCycleStateError:
        pass
    else:
        raise AssertionError(label + ' unexpectedly created a landing')
    assert ref_or_none(landing_ref) is None, label

# No Architect decision is a NO-GO. Neither no keys nor Architect alone can
# reach the landing primitive. Red Team acceptance alone is also insufficient.
control = d.empty_control_plane_state()
save_control(control)
landing_must_be_blocked('missing Architect and Red Team decisions')
control['architect_candidate'] = candidate
save_control(control)
landing_must_be_blocked('missing Red Team decision')
redteam_only = d.empty_control_plane_state()
redteam_only['redteam_candidate'] = candidate
redteam_only['redteam_result'] = 'ACCEPT-CONTROL-PLANE'
assert not d.control_plane_keys_ready(redteam_only, candidate)
redteam_only_state = d.empty_ticket_cycle_state()
redteam_only_state['active'][cycle] = {
    'phase': 'implementation', 'commit': None, 'mode': 'normal',
    'route': 'primary', 'ticket_class': 'protected-control-plane',
    'path_scope': [tool], 'control_plane': redteam_only}
try:
    d.validate_ticket_cycle_state(redteam_only_state)
except d.TicketCycleStateError:
    pass
else:
    raise AssertionError('Red Team acceptance survived without Architect GO')
assert ref_or_none(landing_ref) is None
no_go = d.architect_go_request_payload(cycle, candidate, 'normal').replace(
    d.MAILBOX_DECISION_HEADER + 'GO',
    d.MAILBOX_DECISION_HEADER + 'NO-GO')
assert d._architect_go_request(no_go)[3] is not None

# Exact cycle and candidate identity are checked by D1's real state writers.
for wrong_cycle, wrong_candidate in (
        (other_cycle, candidate), (cycle, other_candidate)):
    try:
        d.record_control_plane_redteam_decision(
            wrong_cycle, wrong_candidate, 'ACCEPT-CONTROL-PLANE')
    except d.TicketCycleStateError:
        pass
    else:
        raise AssertionError('wrong Red Team identity was accepted')
try:
    d.prepare_exact_squash_landing(cycle, other_candidate, 'normal')
except d.TicketCycleStateError:
    pass
else:
    raise AssertionError('wrong candidate reached landing preparation')

# A rejection names exact C but still cannot create L.
control['redteam_candidate'] = candidate
control['redteam_result'] = 'REJECT-CONTROL-PLANE'
save_control(control)
landing_must_be_blocked('Red Team rejection')

# Both exact-C decisions survive a fresh import, which represents a daemon
# restart reading only the serialized files and private candidate ref.
control['redteam_result'] = 'ACCEPT-CONTROL-PLANE'
save_control(control)
restart_probe = '''
from ai.tools import mailbox_daemon as restarted
cycle, candidate, candidate_ref = __import__('sys').argv[1:]
control = restarted.control_plane_ticket_state(cycle, candidate)
assert restarted.control_plane_keys_ready(control, candidate)
saved = restarted.read_candidate_state()['cycles'][cycle]
assert saved == {'ref': candidate_ref, 'commit': candidate}
assert restarted.git_ref_commit(candidate_ref) == candidate
'''
restart = subprocess.run(
    [sys.executable, '-c', restart_probe, cycle, candidate, candidate_ref],
    cwd=shadow_repository, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, check=False)
assert restart.returncode == 0, restart.stdout
assert d.protected_landing_ready(cycle, candidate)

# D1 may calculate inside this disposable checkout, but only the D0-owned
# harness decides whether its real landing primitive behaved correctly.
landing, parent, returned_ref = d.prepare_exact_squash_landing(
    cycle, candidate, 'normal')
assert returned_ref == landing_ref
assert parent == base_commit
parent_row = git(['rev-list', '--parents', '-n', '1', landing]).split()
assert parent_row == [landing, base_commit]
assert ref_or_none(landing_ref) == landing
landing_tree = git(['rev-parse', landing + '^{tree}'])
expected_tree = git(['merge-tree', '--write-tree', base_commit, candidate])
assert landing_tree == expected_tree
assert git(['rev-parse', 'refs/heads/main']) == base_commit

# If main changes after L was prepared, the existing L and new main remain
# untouched and the real D1 path must request integration revalidation.
git(['update-ref', 'refs/heads/main', new_main, base_commit])
try:
    d.prepare_exact_squash_landing(cycle, candidate, 'normal')
except d.RetryableArchitectLandingError as exc:
    assert d.STALE_INTEGRATION_REVALIDATION in str(exc)
else:
    raise AssertionError('changed main reused a stale protected landing')
assert git(['rev-parse', 'refs/heads/main']) == new_main
assert ref_or_none(landing_ref) == landing
print('D0_SHADOW_SCENARIOS_PASSED')
print('CONTROL_PLANE_HEALTHY', ROLE_CONTRACT['schema_version'])
"""
        commands.append([
            sys.executable, "-m", "py_compile",
            os.path.join(repository, "ai", "tools", "role_contract.py"),
            os.path.join(repository, "ai", "tools", "handoff_contract.py"),
            os.path.join(repository, "ai", "tools", "mailbox_daemon.py")])
        commands.append([sys.executable, "-c", probe])
        environment = os.environ.copy()
        for name in tuple(environment):
            if name.startswith("MAILBOX_"):
                del environment[name]
        environment["PYTHONPATH"] = repository
        ok = True
        with open(log_path, "w", encoding="utf-8") as stream:
            for command in commands:
                stream.write("$ " + " ".join(command) + "\n")
                try:
                    result = subprocess.run(
                        command, cwd=(repository
                                      if os.path.isdir(repository) else root),
                        env=environment, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, check=False,
                        timeout=120)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    stream.write(type(exc).__name__ + ": " + str(exc)
                                 + "\n")
                    stream.write("rc=not-completed\n")
                    ok = False
                    break
                stream.write(result.stdout)
                stream.write("rc=" + str(result.returncode) + "\n")
                if result.returncode != 0:
                    ok = False
                    break
            stream.flush()
            os.fsync(stream.fileno())
    after = _live_control_plane_fingerprint()
    if after != before:
        raise TicketCycleStateError(
            "shadow D1 changed D0 live state or private refs")
    return ok, log_path


def record_control_plane_check(cycle_id, candidate_commit, kind, ok,
                               evidence):
    """Persist one D0 shadow or post-landing health result."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        saved = read_candidate_state()["cycles"].get(cycle_id)
        if (active is None
                or active.get("ticket_class") != "protected-control-plane"
                or saved is None
                or saved["commit"] != candidate_commit):
            raise TicketCycleStateError(
                "control-plane check does not name exact saved candidate C")
        control = dict(active["control_plane"])
        if kind == "shadow":
            control["shadow_status"] = "PASSED" if ok else "FAILED"
            control["shadow_evidence"] = evidence
        elif kind == "health":
            control["health_status"] = (
                "HEALTHY" if ok else "CONTROL_PLANE_HEALTH_FAILED")
            control["health_evidence"] = evidence
        else:
            raise TicketCycleStateError("unknown control-plane check kind")
        state["active"][cycle_id] = dict(active, control_plane=control)
        write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def execute_architect_go_locked(cycle_id, candidate_commit, mode,
                                sealed_backlog=None):
    """Land C as exact L and durably advance state before any push."""
    protected = control_plane_ticket_state(
        cycle_id=cycle_id, candidate_commit=candidate_commit) is not None
    if protected and not protected_landing_ready(
            cycle_id=cycle_id, candidate_commit=candidate_commit):
        raise TicketCycleStateError(
            "protected landing lacks exact Architect and Red Team keys")
    if protected:
        prepare_revalidated_control_plane_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit)
    landing = recorded_landing_for_architect_go(
        cycle_id=cycle_id, mode=mode)
    if landing is None:
        landing, parent, _reference = prepare_exact_squash_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            mode=mode, sealed_backlog=sealed_backlog)
    else:
        parent = _verify_prepared_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=landing, expected_backlog=sealed_backlog)
        journaled = git_ref_commit(
            reference=cycle_landing_ref(cycle_id=cycle_id))
        if journaled is not None and journaled != landing:
            raise TicketCycleStateError(
                "durable cycle state and landing crash journal disagree")
    if protected:
        control = control_plane_ticket_state(
            cycle_id=cycle_id, candidate_commit=candidate_commit)
        if (control["integration_status"] == "REVALIDATED"
                and control["shadow_status"] != "PASSED"):
            shadow_ok, shadow_log = trusted_control_plane_check(
                commit=landing, label="integration-shadow")
            record_control_plane_check(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                kind="shadow", ok=shadow_ok, evidence=shadow_log)
            if not shadow_ok:
                raise RetryableArchitectLandingError(
                    "SHADOW_VALIDATION_FAILED for exact revalidated "
                    "integration L=" + landing + "; evidence -> "
                    + shadow_log)
    preflight_role_baseline_sync(
        target=landing, retiring_candidate=candidate_commit)
    land_prepared_commit_in_clean_user_checkout(
        landing=landing, parent=parent,
        candidate_commit=candidate_commit)
    completed_now = record_architect_commit(
        cycle_id=cycle_id, accepted_commit=landing, mode=mode)
    if mode == "normal" and not protected:
        publish_redteam_closure_request(
            cycle_id=cycle_id, landing=landing)
    return landing, completed_now


def require_architect_landing_locked(cycle_id, landing_commit,
                                     ticket_state):
    """Bind candidate C to its exact, distinct squash landing L on main."""
    if ACTIVE_TOPOLOGY is None:
        # Pure function tests do not represent a live dispatch topology.
        return None
    candidate_state = read_candidate_state()
    record = candidate_record_locked(
        cycle_id=cycle_id, ticket_state=ticket_state,
        candidate_state=candidate_state)
    if record is None:
        raise TicketCycleStateError(
            "Architect landing has no saved candidate for this cycle")
    candidate_commit = record["commit"]
    if landing_commit == candidate_commit:
        raise TicketCycleStateError(
            "daemon landing record names the Implementer candidate, not its "
            "distinct squash landing")
    if not git_commit_exists(commit=landing_commit):
        raise TicketCycleStateError(
            "daemon landing record names a missing landing commit")
    current_main = _exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    if current_main != landing_commit:
        raise TicketCycleStateError(
            "daemon landing record does not name the current main landing")
    parent_commit = _single_commit_parent(commit=landing_commit)
    _require_ancestor_or_same(
        ancestor=cycle_starting_commit(cycle_id),
        descendant=parent_commit,
        label="landing parent does not preserve the cycle base")
    expected_tree = _tree_with_backlog(
        tree=_exact_squash_tree(
            parent_commit=parent_commit, candidate_commit=candidate_commit),
        backlog=_landing_backlog(landing_commit=landing_commit))
    landing_tree = _exact_git_object(
        arguments=["rev-parse", "--verify", landing_commit + "^{tree}"],
        label="Architect landing tree")
    if landing_tree != expected_tree:
        raise TicketCycleStateError(
            "Architect landing tree is not the exact candidate plus its "
            "sealed backlog on the landing parent")
    return candidate_commit


def _require_retirement_landing_locked(cycle_id, landing_commit,
                                       ticket_state):
    """Prove one durable L authorizes retirement of this cycle's C."""
    active = ticket_state["active"].get(cycle_id)
    completed = ticket_state["completed"].get(cycle_id)
    recorded = completed
    if recorded is None and active is not None:
        if active["phase"] == "implementation":
            raise TicketCycleStateError(
                "candidate cannot retire before its daemon landing")
        recorded = active["commit"]
    if recorded != landing_commit:
        raise TicketCycleStateError(
            "candidate retirement does not name its durable landing")
    if not git_commit_exists(commit=landing_commit):
        raise TicketCycleStateError(
            "candidate retirement landing no longer exists")
    current_main = _exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    _require_ancestor_or_same(
        ancestor=landing_commit, descendant=current_main,
        label="current main does not preserve the retired cycle landing")


def retire_cycle_candidate_locked(cycle_id, candidate_commit,
                                  landing_commit):
    """Hand off exact clean C to L, then delete only C's ownership."""
    if ACTIVE_TOPOLOGY is None:
        return True
    ticket_state = read_ticket_cycle_state()
    _require_retirement_landing_locked(
        cycle_id=cycle_id, landing_commit=landing_commit,
        ticket_state=ticket_state)
    state = read_candidate_state()
    record = state["cycles"].get(cycle_id)
    reference = cycle_candidate_ref(cycle_id=cycle_id)
    current = git_ref_commit(reference=reference)
    if record is None:
        if current is not None:
            raise TicketCycleStateError(
                "accepted cycle has an unowned candidate ref")
        return True
    if record["commit"] != candidate_commit:
        raise TicketCycleStateError(
            "refusing to retire another candidate commit")
    worktree = AGENT_CWD["opus"]
    head = worktree_head(worktree=worktree)
    if head == record["commit"]:
        if _clean_worktree_status(worktree=worktree):
            # Another ticket may already be editing from C. Keep both the
            # state row and ref until that work is saved or moved; deleting
            # them here would erase the only durable authority for C.
            return False
        _run_git(
            repository_root=worktree,
            arguments=["reset", "--hard", landing_commit])
        if (worktree_head(worktree=worktree) != landing_commit
                or _clean_worktree_status(worktree=worktree)):
            raise TicketCycleStateError(
                "Implementer checkout did not complete exact C-to-L handoff")
        head = landing_commit
    preserved_heads = {
        item["commit"] for other_cycle, item in state["cycles"].items()
        if other_cycle != cycle_id
    }
    preserved_heads.update(
        cycle_starting_commit(other_cycle)
        for other_cycle, item in ticket_state["active"].items()
        if other_cycle != cycle_id and item["phase"] == "implementation")
    if head != landing_commit and head not in preserved_heads:
        # A concurrent Implementer turn is allowed, but only durable cycle
        # state may prove that its HEAD is not abandoned work.
        return False
    if current is not None:
        if current != record["commit"]:
            raise TicketCycleStateError(
                "candidate ref changed before retirement")
        _run_git(
            repository_root=AGENT_CWD["fable"],
            arguments=["update-ref", "-d", reference, record["commit"]])
    del state["cycles"][cycle_id]
    write_candidate_state(state=state)
    return True


def retire_superseded_failed_architect_go(cycle_id, candidate_commit, mode):
    """Archive rejected GO after landing."""
    paths = glob.glob(os.path.join(MAILBOX, "failed", "*-to-daemon.md"))
    for path in sorted(paths, key=message_sequence):
        try:
            returned = _architect_go_request(read_cycle_message(path=path))
        except (OSError, ValueError, TicketCycleStateError):
            continue
        if returned != (cycle_id, candidate_commit, mode, None):
            continue
        _destination, verified = verified_state_move(path, DONE)
        if not verified:
            print("  warning: rejected GO remains: " + os.path.basename(path))


def retire_cycle_candidate(cycle_id, candidate_commit, landing_commit, mode):
    """Retire exact C after durable GO state, preserving concurrent work."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        retired = retire_cycle_candidate_locked(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=landing_commit)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)
    retire_superseded_failed_architect_go(
        cycle_id=cycle_id, candidate_commit=candidate_commit, mode=mode)
    return retired


def _symbolic_worktree_branch(worktree, expected_branch, label):
    """Require one persistent role checkout to stay on its saved branch."""
    result = _run_git(
        repository_root=worktree,
        arguments=["symbolic-ref", "-q", "HEAD"], check=False)
    try:
        branch = result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(label + " branch is not UTF-8") from exc
    if result.returncode != 0 or branch != expected_branch:
        raise TicketCycleStateError(
            label + " checkout left its saved branch")


def _architect_only_sealed_backlog(worktree):
    """Return a sealed backlog when it is the Architect's only change."""
    changed = _run_git(
        repository_root=worktree,
        arguments=["diff", "--name-only", "-z", "HEAD", "--", "."])
    try:
        paths = {item.decode("utf-8", errors="strict")
                 for item in changed.stdout.split(b"\0") if item}
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "Architect changed a non-UTF-8 path") from exc
    untracked = _run_git(
        repository_root=worktree,
        arguments=["ls-files", "--others", "--exclude-standard", "-z",
                   "--", "."])
    if paths != {BACKLOG_RELATIVE_PATH} or untracked.stdout:
        return None
    try:
        return _validate_sealed_backlog(primary_worktree=worktree)
    except PrimaryWorktreeError as exc:
        raise TicketCycleStateError(str(exc)) from exc


def _architect_backlog_matches_target(worktree, target):
    """Return whether the sealed backlog is the only change and is in L."""
    working = _architect_only_sealed_backlog(worktree=worktree)
    if working is None:
        return False
    return working == _landing_backlog(landing_commit=target)


def _clear_landed_architect_backlog(worktree, target):
    """Restore old bytes before a fast-forward that contains the same edit."""
    if not _architect_backlog_matches_target(
            worktree=worktree, target=target):
        raise TicketCycleStateError(
            "Architect checkout has work beyond the backlog in this landing")
    backlog = os.path.join(worktree, BACKLOG_RELATIVE_PATH)
    recovery = os.path.join(
        worktree, "ai", "notes", BACKLOG_SYNC_RECOVERY_NAME)
    if os.path.lexists(recovery):
        raise TicketCycleStateError("backlog sync recovery already exists")
    os.replace(backlog, recovery)
    result = _run_git(
        repository_root=worktree,
        arguments=["restore", "--source=HEAD", "--staged", "--worktree",
                   "--", BACKLOG_RELATIVE_PATH],
        check=False)
    if result.returncode != 0 or _clean_worktree_status(worktree=worktree):
        os.replace(recovery, backlog)
        raise TicketCycleStateError(
            "Architect backlog could not be prepared for baseline sync")
    return recovery


def sync_clean_role_baseline(worktree, expected_branch, target, label):
    """Fast-forward one clean role baseline to an exact landed commit."""
    _symbolic_worktree_branch(
        worktree=worktree, expected_branch=expected_branch, label=label)
    recovery = None
    if _clean_worktree_status(worktree=worktree):
        if label != "Architect":
            raise TicketCycleStateError(
                label + " checkout has staged, unstaged, or untracked work")
        recovery = _clear_landed_architect_backlog(
            worktree=worktree, target=target)
    current = worktree_head(worktree=worktree)
    if current == target:
        return False
    _require_ancestor_or_same(
        ancestor=current, descendant=target,
        label=label + " baseline is not an ancestor of the landing")
    result = _run_git(
        repository_root=worktree,
        arguments=["merge", "--ff-only", target], check=False)
    if result.returncode != 0:
        if recovery is not None:
            os.replace(recovery, os.path.join(
                worktree, BACKLOG_RELATIVE_PATH))
        raise TicketCycleStateError(
            label + " baseline could not fast-forward to the landing")
    if (worktree_head(worktree=worktree) != target
            or _clean_worktree_status(worktree=worktree)):
        raise TicketCycleStateError(
            label + " baseline did not advance cleanly to the landing")
    if recovery is not None:
        os.unlink(recovery)
    return True


def _role_baseline_plan_locked(target, retiring_candidate=None):
    """Preflight all role baselines without changing a checkout."""
    candidate_state = read_candidate_state()
    ticket_state = read_ticket_cycle_state()
    plan = []
    for worktree, branch, label in (
            (AGENT_CWD["fable"], AGENT_BRANCH["fable"], "Architect"),
            (AGENT_CWD["sol"], SOL_BRANCH, "Red Team")):
        _symbolic_worktree_branch(
            worktree=worktree, expected_branch=branch, label=label)
        current = worktree_head(worktree=worktree)
        sealed_overlay = (
            _architect_only_sealed_backlog(worktree=worktree)
            if label == "Architect" and current == target else None)
        if (_clean_worktree_status(worktree=worktree)
                and sealed_overlay is None
                and not (label == "Architect"
                         and _architect_backlog_matches_target(
                             worktree=worktree, target=target))):
            raise TicketCycleStateError(
                label + " checkout has work that baseline sync would touch")
        _require_ancestor_or_same(
            ancestor=current, descendant=target,
            label=label + " baseline is not an ancestor of the landing")
        plan.append((worktree, branch, label, current != target))
    opus_head = worktree_head(worktree=AGENT_CWD["opus"])
    preserved = {record["commit"]
                 for record in candidate_state["cycles"].values()}
    active_bases = {
        cycle_starting_commit(cycle_id)
        for cycle_id, record in ticket_state["active"].items()
        if record["phase"] == "implementation"}
    preserved.update(active_bases)
    _symbolic_worktree_branch(
        worktree=AGENT_CWD["opus"], expected_branch=IMPLEMENTER_BRANCH,
        label="Implementer")
    if opus_head == retiring_candidate:
        if _clean_worktree_status(worktree=AGENT_CWD["opus"]):
            raise TicketCycleStateError(
                "Implementer candidate C has unsaved work")
        plan.append((AGENT_CWD["opus"], IMPLEMENTER_BRANCH,
                     "Implementer candidate", False))
    elif opus_head in preserved:
        plan.append((AGENT_CWD["opus"], IMPLEMENTER_BRANCH,
                     "Implementer preserved work", False))
    elif any(git_commit_descends_from(
            starting_commit=opus_head, accepted_commit=base)
            for base in active_bases):
        plan.append((AGENT_CWD["opus"], IMPLEMENTER_BRANCH,
                     "Implementer older active base", False))
    else:
        if _clean_worktree_status(worktree=AGENT_CWD["opus"]):
            raise TicketCycleStateError(
                "Implementer checkout has work that baseline sync would "
                "touch")
        _require_ancestor_or_same(
            ancestor=opus_head, descendant=target,
            label="Implementer baseline is not an ancestor of the landing")
        plan.append((AGENT_CWD["opus"], IMPLEMENTER_BRANCH,
                     "Implementer", opus_head != target))
    return tuple(plan)


def preflight_role_baseline_sync(target, retiring_candidate=None):
    """Prove every role can preserve or fast-forward before main changes."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        return _role_baseline_plan_locked(
            target=target, retiring_candidate=retiring_candidate)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def sync_all_clean_role_baselines(target):
    """Advance clean idle role baselines under exact ticket-state authority."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        plan = _role_baseline_plan_locked(target=target)
        changed = False
        for worktree, branch, label, should_sync in plan:
            if should_sync:
                changed = (sync_clean_role_baseline(
                    worktree=worktree, expected_branch=branch,
                    target=target, label=label) or changed)
        return changed
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def _permanent_note_commit_paths(base_commit, notes_commit):
    """Return exact modified paths while refusing structural Git changes."""
    summary = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["diff", "--summary", base_commit, notes_commit,
                   "--", "."])
    if summary.stdout:
        raise TicketCycleStateError(
            "permanent-note commit changes a path mode, type, name, or "
            "existence")
    result = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["diff", "--name-only", "-z", "--diff-filter=M",
                   base_commit, notes_commit, "--", "."])
    try:
        paths = [item.decode("utf-8", errors="strict")
                 for item in result.stdout.split(b"\0") if item]
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "permanent-note commit path is not UTF-8") from exc
    if (not paths
            or len(paths) != len(set(paths))
            or not set(paths).issubset(set(
                ARCHITECT_PROTECTED_POLICY_PATHS))):
        raise TicketCycleStateError(
            "protected-policy commit must modify only a role file, the "
            "protected YAML contract, or one of the exact eleven permanent "
            "notes")
    return tuple(paths)


def require_architect_notes_commit_object(base_commit, notes_commit):
    """Prove immutable B-to-P history and its exact note-only path set."""
    if (not git_commit_exists(commit=base_commit)
            or not git_commit_exists(commit=notes_commit)):
        raise TicketCycleStateError(
            "Architect notes request names a missing B or P commit")
    if _single_commit_parent(commit=notes_commit) != base_commit:
        raise TicketCycleStateError(
            "Architect notes P must be exactly one commit directly on B")
    _permanent_note_commit_paths(
        base_commit=base_commit, notes_commit=notes_commit)


def require_architect_notes_commit(base_commit, notes_commit,
                                   allow_landed_replay=False):
    """Prove clean one-parent B-to-P authority for a note-only landing."""
    primary = AGENT_CWD["fable"]
    _symbolic_worktree_branch(
        worktree=primary, expected_branch=AGENT_BRANCH["fable"],
        label="Architect")
    if _clean_worktree_status(worktree=primary):
        raise TicketCycleStateError(
            "Architect note checkout is not clean at commit P")
    if worktree_head(worktree=primary) != notes_commit:
        raise TicketCycleStateError(
            "Architect notes GO does not name primary HEAD P")
    require_architect_notes_commit_object(
        base_commit=base_commit, notes_commit=notes_commit)
    try:
        _validate_protected_tracked_state(primary_worktree=primary)
    except PrimaryWorktreeError as exc:
        raise TicketCycleStateError(str(exc)) from exc
    try:
        proposed_contract = _local_role_contract_tool().load_role_contract(
            os.path.join(primary, ROLE_CONTRACT_RELATIVE_PATH))
        validate_role_contract_bindings(contract=proposed_contract)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TicketCycleStateError(
            "proposed role contract is invalid: " + str(exc)) from exc
    current_main = _exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    allowed = {base_commit, notes_commit} if allow_landed_replay \
        else {base_commit}
    if current_main not in allowed:
        raise TicketCycleStateError(
            "Architect notes B is not the exact current main baseline")
    return current_main


def _require_no_ordinary_landing_transition_locked(current_dispatch_path):
    """Refuse P while ordinary durable work exists; caller holds state lock."""
    ticket_state = read_ticket_cycle_state()
    candidate_state = read_candidate_state()
    if ticket_state["active"] or candidate_state["cycles"]:
        raise TicketCycleStateError(
            "permanent notes wait until every active ticket and candidate "
            "is retired")
    refs = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["for-each-ref", "--format=%(refname)",
                   CANDIDATE_REF_ROOT])
    if refs.stdout.strip():
        raise TicketCycleStateError(
            "permanent notes wait until every candidate/landing ref is "
            "retired")
    current_key = _path_key(current_dispatch_path)
    for directory in (MAILBOX, os.path.join(MAILBOX, "inflight"),
                      os.path.join(MAILBOX, "failed")):
        for path in glob.glob(os.path.join(directory, "*-to-daemon.md")):
            if _path_key(path) == current_key:
                continue
            try:
                message = read_cycle_message(path=path)
            except (OSError, ValueError, TicketCycleStateError) as exc:
                raise TicketCycleStateError(
                    "cannot verify another daemon request: " + str(exc)) \
                    from exc
            if message.startswith(
                    MAILBOX_RETURN_HEADER + "architect-go"):
                raise TicketCycleStateError(
                    "permanent notes wait for the ordinary Architect GO")


def require_no_ordinary_landing_transition(current_dispatch_path):
    """Refuse P while any ordinary ticket, C/ref, landing ref, or GO remains."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        _require_no_ordinary_landing_transition_locked(
            current_dispatch_path=current_dispatch_path)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def architect_notes_transition_pending():
    """Return whether a durable note admin turn or P landing is unresolved."""
    for directory in (MAILBOX, os.path.join(MAILBOX, "inflight"),
                      os.path.join(MAILBOX, "failed")):
        for suffix, header in (
                ("*-to-fable.md", MAILBOX_ADMIN_HEADER),
                ("*-to-daemon.md",
                 MAILBOX_RETURN_HEADER + "architect-notes-go")):
            for path in glob.glob(os.path.join(directory, suffix)):
                try:
                    matches = regular_file_has_prefix(
                        path=path, prefix=header.encode("ascii"))
                except (OSError, ValueError):
                    continue
                if matches:
                    return True
    return False


ARCHITECT_NOTES_DEBT_PREFIX = "permanent-note user action required: "


def failed_architect_notes_transition_paths():
    """Return exact failed admin/P files that no watcher may retry itself."""
    failed = os.path.join(MAILBOX, "failed")
    found = []
    for suffix, header in (
            ("*-to-fable.md", MAILBOX_ADMIN_HEADER),
            ("*-to-daemon.md",
             MAILBOX_RETURN_HEADER + "architect-notes-go")):
        for path in glob.glob(os.path.join(failed, suffix)):
            try:
                matches = regular_file_has_prefix(
                    path=path, prefix=header.encode("ascii"))
            except (OSError, ValueError):
                continue
            if matches:
                found.append(path)
    return sorted(found, key=message_sequence)


def architect_notes_failed_debt_error():
    """Explain failed-only note debt as a finite user-action stop."""
    paths = failed_architect_notes_transition_paths()
    if not paths:
        return None
    relative = [os.path.relpath(path, MAILBOX) for path in paths]
    return (ARCHITECT_NOTES_DEBT_PREFIX
            + ", ".join(relative)
            + "; inspect the saved failure, correct its cause, then move "
              "only the verified exact request back to the mailbox root "
              "before restarting the watcher")


def message_belongs_to_active_cycle(path, active_cycles):
    """Return whether one root agent message advances an admitted ticket."""
    match = PENDING_MESSAGE_RE.match(os.path.basename(path))
    if match is None:
        return False
    try:
        message = read_cycle_message(path=path)
    except (OSError, ValueError, TicketCycleStateError):
        return False
    agent = match.group(1)
    if agent in {"fable", "opus"} and message.startswith(
            MAILBOX_FLOW_HEADER):
        cycle_id, _mode, _body, problem = _ticket_flow_envelope(
            message=message)
        return problem is None and cycle_id in active_cycles
    if agent == "fable" and message.startswith(MAILBOX_RETURN_HEADER):
        cycle_id, _commit, result, _body, problem = (
            _redteam_review_receipt(message=message))
        return (problem is None and result == "REOPEN"
                and cycle_id in active_cycles)
    if agent == "sol" and sol_ticket_kind(message=message) == "closure":
        cycle_id = redteam_closure_ticket(message=message)
        return cycle_id in active_cycles
    if agent == "sol" and sol_ticket_kind(message=message) == "control-plane":
        cycle_id, _candidate, _body, problem = (
            _redteam_control_plane_envelope(message=message))
        return problem is None and cycle_id in active_cycles
    return False


def requeue_retryable_daemon_message(dispatch_path):
    """Return one valid inflight GO to root without calling it malformed."""
    _path, verified = verified_state_move(
        dispatch_path=dispatch_path, directory=MAILBOX)
    return verified


def publish_backlog_close_request(cycle_id, candidate_commit, mode):
    """Queue one exact Architect correction while preserving accepted C."""
    payload = backlog_close_request_payload(
        cycle_id=cycle_id, candidate_commit=candidate_commit, mode=mode)
    for directory in (MAILBOX, os.path.join(MAILBOX, "inflight"),
                      os.path.join(MAILBOX, "prelaunch")):
        for path in glob.glob(os.path.join(directory, "*-to-fable.md")):
            try:
                if read_cycle_message(path=path) == payload:
                    return path
            except (OSError, ValueError, TicketCycleStateError):
                continue
    if not send(agent="fable", text=payload, dry_run=False):
        raise RetryableArchitectLandingError(
            "could not publish backlog-close recovery")
    matches = [path for path in glob.glob(
        os.path.join(MAILBOX, "*-to-fable.md"))
        if read_cycle_message(path=path) == payload]
    if len(matches) != 1:
        raise RetryableArchitectLandingError(
            "backlog-close recovery was not published exactly once")
    return matches[0]


def defer_protected_stale_integration(
        dispatch_path, cycle_id, candidate_commit, mode, problem):
    """Save a moved-main event and queue its same-cycle Architect audit."""
    details = stale_integration_details(problem=problem)
    if details is None or details["candidate"] != candidate_commit:
        raise TicketCycleStateError(
            "protected stale diagnosis changed exact candidate C")
    record_control_plane_integration_stale(
        cycle_id=cycle_id, candidate_commit=candidate_commit,
        stale_landing=details["stale_landing"],
        old_main=details["old_main"], new_main=details["new_main"])
    request = publish_control_plane_integration_request(
        cycle_id=cycle_id, candidate=candidate_commit,
        stale_landing=details["stale_landing"],
        old_main=details["old_main"], new_main=details["new_main"],
        mode=mode)
    deferred = move_without_overwrite(
        path=dispatch_path,
        directory=os.path.join(MAILBOX, "integration-stale"))
    if deferred is None and os.path.lexists(dispatch_path):
        raise TicketCycleStateError(
            "stale Architect GO could not enter its durable waiting state")
    return request


def prepared_landing_reached_main(cycle_id):
    """Return whether main contains this cycle's journaled landing."""
    landing = git_ref_commit(reference=cycle_landing_ref(cycle_id=cycle_id))
    if landing is None:
        return False
    current = _exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit after landing error")
    if current == landing:
        return True
    result = _run_git(
        repository_root=AGENT_CWD["fable"],
        arguments=["merge-base", "--is-ancestor", landing, current],
        check=False)
    if result.returncode not in {0, 1}:
        raise TicketCycleStateError(
            "cannot determine whether main contains the prepared landing")
    return result.returncode == 0


def finish_claimed_architect_go(dispatch_path, cycle_id,
                                candidate_commit, mode):
    """Finish or replay one already-claimed, well-formed Architect GO."""
    name = os.path.basename(dispatch_path)
    try:
        active = read_ticket_cycle_state()["active"].get(cycle_id)
        if (active is None or active["mode"] != mode
                or candidate_commit_for_cycle(cycle_id) != candidate_commit):
            raise TicketCycleStateError(
                "Architect GO changed the active cycle, mode, or candidate")
        sealed_backlog = _validate_sealed_backlog(
            primary_worktree=AGENT_CWD["fable"])
        require_closed_backlog_ticket(
            ticket_anchor=cycle_ticket_anchor(cycle_id),
            sealed_backlog=sealed_backlog)
    except BacklogTicketOpenError:
        request = publish_backlog_close_request(
            cycle_id=cycle_id, candidate_commit=candidate_commit, mode=mode)
        if not archive_consumed_message(dispatch_path=dispatch_path):
            raise RetryableArchitectLandingError(
                "accepted GO could not enter backlog-close recovery")
        print("backlog closure required before landing " + candidate_commit
              + "; preserved C and the prior audit; queued "
              + os.path.basename(request) + " for bookkeeping and one "
                "fresh exact GO.")
        return True, 0, None
    except (PrimaryWorktreeError, TicketCycleStateError) as exc:
        parked = park_failed_message(dispatch_path=dispatch_path)
        state = "parked." if parked else "move failed."
        print("refused " + name + ": " + str(exc)
              + "; C and its cycle remain. Close and seal the ticket, then "
              "send a fresh GO; " + state)
        return False, 0, None
    protected = control_plane_ticket_state(
        cycle_id=cycle_id, candidate_commit=candidate_commit)
    if protected is not None:
        try:
            record_control_plane_architect_go(
                cycle_id=cycle_id, candidate_commit=candidate_commit)
            protected = control_plane_ticket_state(
                cycle_id=cycle_id, candidate_commit=candidate_commit)
            if protected["redteam_result"] is None:
                request = publish_control_plane_review_request(
                    cycle_id=cycle_id, candidate=candidate_commit)
                print("protected Architect GO(C) recorded; waiting for the "
                      "mandatory Red Team decision on exact C; request "
                      + os.path.basename(request) + ".")
                return None, 0, None
            if protected["redteam_result"] == "REJECT-CONTROL-PLANE":
                publish_control_plane_repair_request(
                    cycle_id=cycle_id, candidate=candidate_commit,
                    mode=mode)
                rejected_path = move_without_overwrite(
                    path=dispatch_path,
                    directory=os.path.join(MAILBOX, "redteam-rejected"))
                if rejected_path is None:
                    return False, 0, None
                print("protected candidate C was rejected by Red Team; C "
                      "was preserved and an Architect repair turn was "
                      "queued.")
                return True, 0, None
            if (protected["health_status"]
                    == "CONTROL_PLANE_HEALTH_FAILED"):
                raise FatalArchitectLandingError(
                    "CONTROL_PLANE_HEALTH_FAILED: D0 is recovery-only; "
                    "inspect " + protected["health_evidence"]
                    + " and preserve the recorded landing")
            if protected["shadow_status"] == "FAILED":
                print("SHADOW_VALIDATION_FAILED: D0 preserved C, both "
                      "decisions, and the evidence at "
                      + protected["shadow_evidence"] + ".")
                return None, 0, None
            if (protected["shadow_status"] != "PASSED"
                    and protected["integration_status"]
                    != "REVALIDATED"):
                shadow_ok, shadow_log = trusted_control_plane_check(
                    commit=candidate_commit, label="shadow")
                record_control_plane_check(
                    cycle_id=cycle_id, candidate_commit=candidate_commit,
                    kind="shadow", ok=shadow_ok, evidence=shadow_log)
                if not shadow_ok:
                    print("SHADOW_VALIDATION_FAILED: D0 did not create L; "
                          "evidence -> " + shadow_log)
                    return None, 0, None
        except FatalArchitectLandingError:
            raise
        except (OSError, TicketCycleStateError) as exc:
            print("protected control-plane gate stopped before landing: "
                  + str(exc) + "; C and GO remain preserved.")
            return None, 0, None
    main_lock = acquire_main_checkout_turn_lock()
    if main_lock is None:
        requeued = requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        raise FatalArchitectLandingError(
            "daemon landing lock was unavailable; "
            + ("the exact GO was returned to the mailbox root"
               if requeued else
               "the inflight GO remains preserved for recovery")
            + ". Stop the other landing process and restart.")
    try:
        landing, completed = execute_architect_go_locked(
            cycle_id=cycle_id, candidate_commit=candidate_commit, mode=mode,
            sealed_backlog=sealed_backlog)
    except RetryableArchitectLandingError as exc:
        release_main_checkout_turn_lock(lock_file=main_lock)
        if (protected is not None
                and stale_integration_details(problem=exc) is not None):
            try:
                request = defer_protected_stale_integration(
                    dispatch_path=dispatch_path, cycle_id=cycle_id,
                    candidate_commit=candidate_commit, mode=mode,
                    problem=exc)
            except (OSError, TicketCycleStateError) as recovery_exc:
                raise FatalArchitectLandingError(
                    str(exc) + "; C and both approvals remain preserved, "
                    "but D0 could not queue integration revalidation: "
                    + str(recovery_exc)) from exc
            print(STALE_INTEGRATION_REVALIDATION + ": C and both approvals "
                  "were preserved; Architect integration audit queued as "
                  + os.path.basename(request) + ".")
            return None, 0, None
        requeued = requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        preserved = ("the exact GO was returned to the mailbox root"
                     if requeued else
                     "the inflight GO remains preserved for recovery")
        if STALE_INTEGRATION_REVALIDATION in str(exc):
            remedy = (
                "Automated integration revalidation is not supported yet; "
                "keep "
                "C, L, GO, and the user's work preserved, and do not restart "
                "this cycle until that recovery is handled explicitly")
        elif "SHADOW_VALIDATION_FAILED" in str(exc):
            remedy = (
                "Keep C, both approvals, the revalidated landing, and its "
                "named evidence preserved; do not install that landing")
        elif ("durable-state recovery" in str(exc)
              or "history requires user reconciliation" in str(exc)):
            remedy = (
                "Keep C, L, GO, and the user's work preserved and inspect "
                "the named Git history before retrying")
        else:
            remedy = (
                "Make the user's unchanged-parent main checkout clean, then "
                "restart this watcher")
        raise FatalArchitectLandingError(
            str(exc) + "; " + preserved + ". " + remedy + ".") from exc
    except (OSError, PrimaryWorktreeError, TicketCycleStateError) as exc:
        try:
            landing_reached_main = prepared_landing_reached_main(
                cycle_id=cycle_id)
        except (OSError, PrimaryWorktreeError,
                TicketCycleStateError) as proof_exc:
            release_main_checkout_turn_lock(lock_file=main_lock)
            requeued = requeue_retryable_daemon_message(
                dispatch_path=dispatch_path)
            raise FatalArchitectLandingError(
                "landing recovery could not determine whether main already "
                "advanced: " + str(proof_exc) + "; "
                + ("the exact GO was returned to the mailbox root"
                   if requeued else
                   "the inflight GO remains preserved for recovery")
                + ". Repair Git access, then restart.") from exc
        if landing_reached_main:
            release_main_checkout_turn_lock(lock_file=main_lock)
            requeued = requeue_retryable_daemon_message(
                dispatch_path=dispatch_path)
            raise FatalArchitectLandingError(
                "main already contains the prepared landing, but its "
                "final checks did not finish: " + str(exc) + "; "
                + ("the exact GO was returned to the mailbox root"
                   if requeued else
                   "the inflight GO remains preserved for recovery")
                + ". Restart to finish the same landing.") from exc
        release_main_checkout_turn_lock(lock_file=main_lock)
        parked = park_failed_message(dispatch_path=dispatch_path)
        print("refused " + name + ": exact local landing was not accepted: "
              + str(exc) + "; "
              + ("parked in failed/." if parked else
                 "failed-state move was not verified."))
        return False, 0, None
    if protected is not None:
        try:
            health_ok, health_log = trusted_control_plane_check(
                commit=landing, label="health")
            record_control_plane_check(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                kind="health", ok=health_ok, evidence=health_log)
            if not health_ok:
                release_main_checkout_turn_lock(lock_file=main_lock)
                raise FatalArchitectLandingError(
                    "CONTROL_PLANE_HEALTH_FAILED: L is preserved at "
                    + landing + " and D0 is stopping before new work; "
                      "inspect " + health_log
                      + ". Repair with the preserved trusted controller; "
                        "do not rewrite history")
            completed = int(complete_protected_ticket_cycle(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                landing=landing))
        except FatalArchitectLandingError:
            raise
        except (OSError, TicketCycleStateError) as exc:
            release_main_checkout_turn_lock(lock_file=main_lock)
            raise FatalArchitectLandingError(
                "CONTROL_PLANE_HEALTH_FAILED: L is preserved at "
                + landing + "; D0 health state could not finish: "
                + str(exc)) from exc
    # State first, archive second. A crash in between leaves one inflight GO
    # whose exact landing, state, and closure publication replay idempotently.
    try:
        write_push_debt(
            landing=landing,
            detail="local landing recorded; remote push not yet attempted")
    except OSError as exc:
        release_main_checkout_turn_lock(lock_file=main_lock)
        requeued = requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        raise FatalArchitectLandingError(
            "local landing state is durable, but its required push-debt "
            "note could not be written: " + str(exc) + "; "
            + ("the exact GO was returned to the mailbox root"
               if requeued else
               "the inflight GO remains preserved for recovery")
            + ". Repair relay-directory writes, then restart.") from exc
    if not archive_consumed_message(dispatch_path=dispatch_path):
        release_main_checkout_turn_lock(lock_file=main_lock)
        return False, 0, landing
    try:
        retire_cycle_landing_ref(cycle_id=cycle_id, landing=landing)
    except (OSError, TicketCycleStateError) as exc:
        print("  warning: durable state and GO archive are complete, but "
              "the private landing journal remains for recovery: "
              + str(exc))
    try:
        retired = retire_cycle_candidate(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=landing, mode=mode)
    except (OSError, TicketCycleStateError) as exc:
        print("  warning: durable state and GO archive are complete, but "
              "the private candidate journal remains for recovery: "
              + str(exc))
        retired = False
    try:
        sync_all_clean_role_baselines(target=landing)
    except (OSError, TicketCycleStateError) as exc:
        release_main_checkout_turn_lock(lock_file=main_lock)
        raise FatalArchitectLandingError(
            "local landing is durable, but clean role baselines did not "
            "finish advancing to it: " + str(exc)
            + "; restart to replay the archived GO recovery") from exc
    release_main_checkout_turn_lock(lock_file=main_lock)
    deliver_pending_ticket_cycle_returns()
    if protected is not None:
        print("protected ticket cycle complete after exact Architect GO(C), "
              "Red Team ACCEPT(C), D0 shadow validation, and healthy L: "
              + cycle_id + ".")
    elif mode == "two-role":
        if completed:
            print("ticket cycle complete at the exact local landing: "
                  + cycle_id + ".")
        else:
            print("ticket cycle was already complete at the exact local "
                  "landing: " + cycle_id + ".")
    else:
        print("recorded exact local landing " + landing
              + " for ticket cycle " + cycle_id
              + "; its advisory Red Team review is queued.")
    try:
        pushed, detail = push_exact_landing_or_record_debt(landing=landing)
    except (OSError, ValueError) as exc:
        pushed = False
        detail = str(exc)
    if pushed:
        print("verified remote main at exact landing " + landing + ".")
    else:
        print("local landing is complete; remote push remains follow-up "
              "debt for " + landing + (": " + detail if detail else "."))
    return True, completed, landing


DAEMON_MESSAGE_CONSUMED = "consumed"
DAEMON_NOTE_DEFERRED = "retryable-note-deferred"
DAEMON_CONTROL_PLANE_WAITING = "control-plane-waiting"
DAEMON_MESSAGE_HARD_STOP = "hard-stop"
ROLE_CONTRACT_RESTART_REQUIRED = "role-contract-restart-required"


def finish_claimed_architect_notes_go(dispatch_path, base_commit,
                                      notes_commit, return_outcome=False):
    """Fast-forward exact note-only P, sync clean roles, archive, and push."""
    def result(consumed, outcome):
        ordinary = (consumed, notes_commit)
        return ordinary + (outcome,) if return_outcome else ordinary

    name = os.path.basename(dispatch_path)
    main_lock = acquire_main_checkout_turn_lock()
    if main_lock is None:
        requeue_retryable_daemon_message(dispatch_path=dispatch_path)
        raise FatalArchitectLandingError(
            "Architect note landing lock was unavailable; exact request "
            "was preserved for restart")
    main_advanced = False
    try:
        try:
            current_main = require_architect_notes_commit(
                base_commit=base_commit, notes_commit=notes_commit,
                allow_landed_replay=True)
        except TicketCycleStateError as exc:
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": note-only landing was invalid: "
                  + str(exc) + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return result(False, DAEMON_MESSAGE_HARD_STOP)
        landed_replay = current_main == notes_commit
        if not landed_replay:
            try:
                require_no_ordinary_landing_transition(
                    current_dispatch_path=dispatch_path)
            except TicketCycleStateError as exc:
                requeued = requeue_retryable_daemon_message(
                    dispatch_path=dispatch_path)
                print("deferred " + name + ": " + str(exc) + "; "
                      + ("request returned to mailbox root"
                         if requeued else
                         "request remains preserved in inflight")
                      + ". Older admitted ticket work may continue.")
                return result(False, DAEMON_NOTE_DEFERRED)
            # Re-prove the no-ticket barrier and exact B/P immediately before
            # changing the user checkout. The main-turn lock prevents a
            # landing between these checks and the ff-only operation.
            require_no_ordinary_landing_transition(
                current_dispatch_path=dispatch_path)
            current_main = require_architect_notes_commit(
                base_commit=base_commit, notes_commit=notes_commit,
                allow_landed_replay=True)
            if current_main != base_commit:
                raise TicketCycleStateError(
                    "permanent-note B changed before its exact landing")
        preflight_role_baseline_sync(target=notes_commit)
        land_prepared_commit_in_clean_user_checkout(
            landing=notes_commit, parent=base_commit)
        main_advanced = True
        try:
            write_push_debt(
                landing=notes_commit,
                detail="local permanent-note landing recorded; remote push "
                       "not yet attempted")
        except OSError as exc:
            requeued = requeue_retryable_daemon_message(
                dispatch_path=dispatch_path)
            raise FatalArchitectLandingError(
                "permanent-note P reached main, but push debt could not be "
                "saved: " + str(exc) + "; "
                + ("request returned to mailbox root" if requeued else
                   "request remains preserved in inflight")) from exc
        sync_all_clean_role_baselines(target=notes_commit)
    except RetryableArchitectLandingError as exc:
        requeued = requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        raise FatalArchitectLandingError(
            str(exc) + "; permanent-note request "
            + ("returned to mailbox root" if requeued
               else "remains preserved in inflight")) from exc
    except FatalArchitectLandingError:
        raise
    except (OSError, TicketCycleStateError) as exc:
        requeued = requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        phase = ("after P reached main" if main_advanced else
                 "before P changed main")
        raise FatalArchitectLandingError(
            "permanent-note landing stopped " + phase + ": " + str(exc)
            + "; " + ("request returned to mailbox root" if requeued else
                       "request remains preserved in inflight")
            + "; restart after correcting the named role baseline") from exc
    finally:
        release_main_checkout_turn_lock(lock_file=main_lock)
    try:
        receipt_raw = stable_regular_bytes(
            path=dispatch_path,
            maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="consumed permanent-note GO receipt")
    except (OSError, ValueError) as exc:
        raise FatalArchitectLandingError(
            "permanent-note P and role baselines are ready, but the exact "
            "GO receipt could not be reread before archive: " + str(exc)) \
            from exc
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if not archive_consumed_message(dispatch_path=dispatch_path):
        return result(False, DAEMON_MESSAGE_HARD_STOP)
    try:
        retired_journal = retire_validated_commit_admin_journal(
            base_commit=base_commit, notes_commit=notes_commit,
            receipt_sha256=receipt_sha256)
    except (OSError, TicketCycleStateError) as exc:
        raise FatalArchitectLandingError(
            "permanent-note GO is archived and P baselines are verified, "
            "but its validated admin journal remains: " + str(exc)) \
            from exc
    if retired_journal:
        print("retired validated permanent-note admin journal after exact "
              "P receipt consumption.")
    try:
        pushed, detail = push_exact_landing_or_record_debt(
            landing=notes_commit)
    except (OSError, ValueError) as exc:
        pushed, detail = False, str(exc)
    if pushed:
        print("verified remote main at permanent-note commit "
              + notes_commit + ".")
    else:
        print("permanent-note landing is complete; remote push remains "
              "follow-up debt for " + notes_commit
              + (": " + detail if detail else "."))
    return result(True, DAEMON_MESSAGE_CONSUMED)


def consume_daemon_message(path, dry_run=False, return_outcome=False):
    """Validate and consume one candidate-bound Architect GO locally.

    The daemon recipient is not an AI lane. It exists so cycle completion is
    based on a saved Architect decision instead of inference from prose or a
    changing backlog count.
    """
    def result(outcome):
        return outcome if return_outcome \
            else outcome == DAEMON_MESSAGE_CONSUMED

    name = os.path.basename(path)
    dispatch_path = path
    if not dry_run:
        dispatch_path = claim_message(path=path)
        if dispatch_path is None:
            return result(DAEMON_MESSAGE_HARD_STOP)
    try:
        with open(dispatch_path, encoding="utf-8", newline="") as stream:
            message = stream.read()
    except (OSError, UnicodeError) as exc:
        if dry_run:
            print("[dry-run] would refuse " + name + ": " + str(exc))
            return result(DAEMON_MESSAGE_HARD_STOP)
        parked = park_failed_message(dispatch_path=dispatch_path)
        print("refused " + name + ": cannot read Architect GO request; "
              + ("parked in failed/." if parked else
                 "failed-state move was not verified."))
        return result(DAEMON_MESSAGE_HARD_STOP)
    if message.startswith(
            MAILBOX_RETURN_HEADER + "redteam-control-plane"):
        cycle_id, candidate_commit, decision, _body, problem = (
            _control_plane_review_receipt(message=message))
        if problem is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": " + problem)
                return result(DAEMON_MESSAGE_HARD_STOP)
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": " + problem + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return result(DAEMON_MESSAGE_HARD_STOP)
        if dry_run:
            print("[dry-run] would record " + decision
                  + " for exact protected candidate " + candidate_commit)
            return result(DAEMON_MESSAGE_CONSUMED)
        try:
            control = control_plane_ticket_state(
                cycle_id=cycle_id, candidate_commit=candidate_commit)
            authenticated = control_plane_redteam_key_matches(
                control=control, candidate_commit=candidate_commit,
                decision=decision)
            if not authenticated:
                raise TicketCycleStateError(
                    "control-plane receipt lacks a D0-recorded successful "
                    "Sol dispatch")
        except TicketCycleStateError as exc:
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": " + str(exc) + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return result(DAEMON_MESSAGE_HARD_STOP)
        if not archive_consumed_message(dispatch_path=dispatch_path):
            return result(DAEMON_MESSAGE_HARD_STOP)
        go_paths = []
        for go_path in glob.glob(os.path.join(
                MAILBOX, "inflight", "*-to-daemon.md")):
            try:
                go_message = read_cycle_message(path=go_path)
            except (OSError, ValueError, TicketCycleStateError):
                continue
            found_cycle, found_candidate, found_mode, go_problem = (
                _architect_go_request(message=go_message))
            if (go_problem is None and found_cycle == cycle_id
                    and found_candidate == candidate_commit):
                go_paths.append((go_path, found_mode))
        if len(go_paths) != 1:
            print("protected Red Team decision is durable, but its exact "
                  "inflight Architect GO was not uniquely found; restart "
                  "will preserve the decision and recover the same C.")
            return result(DAEMON_CONTROL_PLANE_WAITING)
        consumed, _completed, _landing = finish_claimed_architect_go(
            dispatch_path=go_paths[0][0], cycle_id=cycle_id,
            candidate_commit=candidate_commit, mode=go_paths[0][1])
        if consumed is None:
            return result(DAEMON_CONTROL_PLANE_WAITING)
        return result(DAEMON_MESSAGE_CONSUMED if consumed
                      else DAEMON_MESSAGE_HARD_STOP)
    if message.startswith(MAILBOX_RETURN_HEADER + "architect-notes-go"):
        base_commit, notes_commit, problem = (
            _architect_notes_go_request(message=message))
        if problem is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": " + problem)
                return result(DAEMON_MESSAGE_HARD_STOP)
            parked = park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": " + problem + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return result(DAEMON_MESSAGE_HARD_STOP)
        if dry_run:
            print("[dry-run] would land exact permanent-note commit "
                  + notes_commit + " on " + base_commit)
            return result(DAEMON_MESSAGE_CONSUMED)
        _consumed, _notes_commit, outcome = (
            finish_claimed_architect_notes_go(
            dispatch_path=dispatch_path, base_commit=base_commit,
            notes_commit=notes_commit, return_outcome=True))
        return result(outcome)
    cycle_id, candidate_commit, mode, problem = _architect_go_request(
        message=message)
    if problem is not None:
        if dry_run:
            print("[dry-run] would refuse " + name + ": " + problem)
            return result(DAEMON_MESSAGE_HARD_STOP)
        parked = park_failed_message(dispatch_path=dispatch_path)
        print("refused " + name + ": " + problem + "; "
              + ("parked in failed/." if parked else
                 "failed-state move was not verified."))
        return result(DAEMON_MESSAGE_HARD_STOP)
    if dry_run:
        print("[dry-run] would prepare and locally land exact candidate "
              + candidate_commit + " from Architect GO " + name)
        return result(DAEMON_MESSAGE_CONSUMED)
    consumed, _completed, _landing = finish_claimed_architect_go(
        dispatch_path=dispatch_path, cycle_id=cycle_id,
        candidate_commit=candidate_commit, mode=mode)
    if consumed is None:
        return result(DAEMON_CONTROL_PLANE_WAITING)
    return result(DAEMON_MESSAGE_CONSUMED if consumed
                  else DAEMON_MESSAGE_HARD_STOP)


def release_unstarted_ticket_reservation(cycle_id, expected_mode=None):
    """Remove only a new implementation reservation that was never claimed."""
    lock_file = acquire_ticket_cycle_lock()
    released = False
    try:
        state = read_ticket_cycle_state()
        current = state["active"].get(cycle_id)
        if (current is not None and current["phase"] == "implementation"
                and current["commit"] is None
                and current["route"] == "primary"
                and (expected_mode is None
                     or current["mode"] == expected_mode)):
            del state["active"][cycle_id]
            write_ticket_cycle_state(state=state)
            released = True
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)
    return released


def ticket_cycle_has_live_message(cycle_id):
    """Return whether a root or inflight message still owns this cycle."""
    header = MAILBOX_CYCLE_HEADER + cycle_id
    for directory in (MAILBOX, os.path.join(MAILBOX, "inflight")):
        for path in glob.glob(os.path.join(directory, "*-to-*.md")):
            if PENDING_MESSAGE_RE.match(os.path.basename(path)) is None:
                continue
            try:
                message = read_cycle_message(path=path)
            except (OSError, ValueError, TicketCycleStateError):
                return True
            if header in message.splitlines():
                return True
    return False


def recover_failed_implementer_preflight():
    """Release a proved pre-launch reservation left by an older daemon."""
    recovered = 0
    pattern = os.path.join(MAILBOX, "failed", "*-to-opus.md")
    for path in sorted(glob.glob(pattern), key=message_sequence):
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            continue
        if (not message.startswith(MAILBOX_FLOW_HEADER)
                or len(ARCHITECT_DIRECTIVE_LINE_RE.findall(message)) == 1):
            continue
        cycle_id, mode, _body, problem = _ticket_flow_envelope(
            message=message)
        if (problem is not None
                or ticket_cycle_has_live_message(cycle_id=cycle_id)
                or candidate_commit_for_cycle(cycle_id) is not None
                or worktree_head(AGENT_CWD["opus"])
                != cycle_starting_commit(cycle_id)
                or _clean_worktree_status(AGENT_CWD["opus"])):
            continue
        if release_unstarted_ticket_reservation(
                cycle_id=cycle_id, expected_mode=mode):
            recovered += 1
            print("released pre-launch reservation for failed "
                  + os.path.basename(path))
    return recovered


def recover_failed_implementer_returns():
    """Revalidate a completed return without rerunning the Implementer."""
    recovered = 0
    requests = [
        path for directory in (os.path.join(MAILBOX, "failed"),
                               os.path.join(MAILBOX, "inflight"))
        for path in glob.glob(os.path.join(directory, "*-to-opus.md"))]
    for request_path in sorted(requests, key=message_sequence):
        try:
            request = read_cycle_message(path=request_path)
            cycle_id, mode, _body, problem = _ticket_flow_envelope(
                message=request)
            active = read_ticket_cycle_state()["active"].get(cycle_id)
            if (problem is not None or active is None
                    or active["phase"] != "implementation"
                    or active["commit"] is not None
                    or active["mode"] != mode
                    or architect_handoff_problem(
                        message=request, cycle_id=cycle_id,
                        mode=mode) is not None
                    or candidate_commit_for_cycle(cycle_id) is not None):
                continue
            candidate = worktree_head(worktree=AGENT_CWD["opus"])
            if (candidate == cycle_starting_commit(cycle_id)
                    or _clean_worktree_status(AGENT_CWD["opus"])):
                continue
            contract = prepare_implementer_evidence_contract(
                message=request, use_saved_limit=True)
            return_path, _invalid, evidence_problem, ready = (
                matching_new_implementer_handoff(
                    cycle_id=cycle_id, mode=mode,
                    candidate_commit=candidate,
                    before_inodes=frozenset(), evidence_contract=contract))
            if evidence_problem is not None or not ready:
                continue
            if os.path.dirname(request_path) != os.path.join(
                    MAILBOX, "inflight"):
                request_path, moved = verified_state_move(
                    dispatch_path=request_path,
                    directory=os.path.join(MAILBOX, "inflight"))
                if not moved:
                    raise TicketCycleStateError(
                        "validated Implementer request could not be restored "
                        "for delivery")
            if os.path.dirname(return_path) != MAILBOX:
                return_path, moved = verified_state_move(
                    dispatch_path=return_path, directory=MAILBOX)
                if not moved:
                    raise TicketCycleStateError(
                        "validated Implementer return could not be restored "
                        "for Architect review")
            write_implementer_delivery_receipt(
                request_path=request_path, return_path=return_path)
            recovered += 1
            print("revalidated completed Implementer return "
                  + os.path.basename(return_path)
                  + "; candidate will be preserved without rerunning the "
                    "Implementer")
        except (OSError, ValueError, PrimaryWorktreeError,
                TicketCycleStateError):
            continue
    return recovered


def live_implementer_owns_architect_admission(token):
    """Return whether a valid queued Implementer handoff owns ``token``."""
    request_name, digest = split_architect_admission_token(token=token)
    for directory in (MAILBOX, os.path.join(MAILBOX, "inflight"),
                      os.path.join(MAILBOX, "prelaunch"), DONE):
        for path in glob.glob(os.path.join(directory, "*-to-opus.md")):
            try:
                message = read_cycle_message(path=path)
            except (OSError, ValueError, TicketCycleStateError):
                return True
            flow_name, flow_digest, problem = (
                _ticket_architect_admission(message=message))
            if (problem is None and flow_name == request_name
                    and flow_digest == digest):
                return True
    return False


def retire_failed_public_architect_admission(path):
    """Release one exact failed public request without retrying its turn."""
    name = os.path.basename(path)
    match = PENDING_MESSAGE_RE.fullmatch(name)
    if (match is None or match.group(1) != "fable"
            or os.path.dirname(path) != os.path.join(MAILBOX, "failed")):
        return False
    sequence_lock = acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise TicketCycleStateError("cannot lock failed admission recovery")
    state_lock = None
    try:
        state_lock = acquire_ticket_cycle_lock()
        state = read_ticket_cycle_state()
        record = state["architect_admissions"].get(name)
        if record is None:
            return False
        other_states = [
            os.path.join(MAILBOX, name),
            os.path.join(MAILBOX, "prelaunch", name),
            os.path.join(DONE, name),
            os.path.join(MAILBOX, "inflight", name),
            os.path.join(MAILBOX, "inflight", name + STATE_GUARD_SUFFIX),
        ]
        if any(os.path.lexists(candidate) for candidate in other_states):
            return False
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            return False
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if (message == ARCHITECT_FIX_ONLY_REQUEST
                or architect_user_request_problem(message=message)
                is not None):
            return False
        if (record["sequence"] != message_sequence(path)
                or record["sha256"] != digest):
            raise TicketCycleStateError(
                "failed public Architect request changed identity")
        token = architect_admission_token(
            request_name=name, digest=digest)
        if live_implementer_owns_architect_admission(token=token):
            return False
        del state["architect_admissions"][name]
        write_ticket_cycle_state(state=state)
        print("released finite-cycle slot for failed " + name
              + "; the failed Architect turn was not retried")
        return True
    finally:
        if state_lock is not None:
            release_ticket_cycle_lock(lock_file=state_lock)
        release_mailbox_sequence_lock(lock_file=sequence_lock)


def recover_failed_public_architect_admissions():
    """Retire exact failed public requests left charged by an older run."""
    recovered = 0
    pattern = os.path.join(MAILBOX, "failed", "*-to-fable.md")
    for path in sorted(glob.glob(pattern), key=message_sequence):
        if retire_failed_public_architect_admission(path=path):
            recovered += 1
    return recovered


def recover_failed_architect_outcome():
    """Restore paid work parked with newer user mail."""
    lock_file = acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise TicketCycleStateError("cannot lock outcome recovery")
    try:
        admissions = read_ticket_cycle_state()["architect_admissions"]
        recovered = 0
        failed = os.path.join(MAILBOX, "failed")
        for request_name, record in admissions.items():
            request_path = os.path.join(failed, request_name)
            if not os.path.lexists(request_path):
                continue
            request = read_cycle_message(path=request_path)
            if request != ARCHITECT_FIX_ONLY_REQUEST:
                continue
            if (hashlib.sha256(request.encode("utf-8")).hexdigest()
                    != record["sha256"]):
                raise TicketCycleStateError("failed request changed identity")
            token = architect_admission_token(
                request_name=request_name, digest=record["sha256"])
            outcomes = [
                path for directory in (failed,
                    os.path.join(MAILBOX, "prelaunch"))
                for path in glob.glob(os.path.join(directory, "*-to-opus.md"))
                if message_claims_architect_admission(path, token)]
            if not outcomes:
                continue
            if len(outcomes) != 1:
                raise TicketCycleStateError(
                    "multiple bound outcomes")
            outcome_path = outcomes[0]
            outcome = read_cycle_message(path=outcome_path)
            if os.path.dirname(outcome_path) == failed:
                outcome_path, moved = verified_state_move(
                    dispatch_path=outcome_path,
                    directory=os.path.join(MAILBOX, "prelaunch"))
                if not moved:
                    raise TicketCycleStateError(
                        "could not preserve recovered plan")
            for path in glob.glob(os.path.join(failed, "*-to-fable.md")):
                name = os.path.basename(path)
                if (name in admissions or message_sequence(path)
                        < message_sequence(outcome_path)):
                    continue
                message = read_cycle_message(path=path)
                if architect_user_request_problem(message) is not None:
                    continue
                _restored, moved = verified_state_move(
                    dispatch_path=path, directory=MAILBOX)
                if not moved:
                    raise TicketCycleStateError(
                        "could not restore user request " + name)
            register_ticket_cycle_message(
                agent="opus", message=outcome,
                skip_redteam=(record["mode"] == "two-role"),
                architect_admission=token,
                implementer_request_name=os.path.basename(outcome_path))
            if not archive_consumed_message(dispatch_path=request_path):
                raise TicketCycleStateError(
                    "could not archive completed request")
            recovered += 1
            print("recovered " + os.path.basename(outcome_path)
                  + " without rerunning " + request_name)
        return recovered
    finally:
        release_mailbox_sequence_lock(lock_file=lock_file)


def recover_failed_open_ticket_go():
    """Retry an exact GO parked by the retired Open-ticket behavior."""
    active = read_ticket_cycle_state()["active"]
    recovered = 0
    paths = glob.glob(os.path.join(MAILBOX, "failed", "*-to-daemon.md"))
    for path in sorted(paths, key=message_sequence):
        try:
            cycle_id, candidate, mode, problem = _architect_go_request(
                message=read_cycle_message(path=path))
            record = active.get(cycle_id)
            if (problem is not None or record is None
                    or record["phase"] != "implementation"
                    or record["mode"] != mode
                    or candidate_commit_for_cycle(cycle_id) != candidate):
                continue
        except (OSError, ValueError, TicketCycleStateError):
            continue
        path, moved = verified_state_move(
            dispatch_path=path, directory=os.path.join(MAILBOX, "inflight"))
        if not moved:
            raise TicketCycleStateError(
                "accepted GO could not be restored for recovery")
        recovered += 1
        print("recovered accepted GO " + os.path.basename(path)
              + " without repeating the candidate audit")
    return recovered


def recover_prelaunch_messages():
    """Requeue requests durably retained before any agent process started."""
    sequence_lock = acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise TicketCycleStateError("cannot lock pre-launch recovery")
    recovered = 0
    try:
        pattern = os.path.join(MAILBOX, "prelaunch", "*-to-*.md")
        for path in sorted(glob.glob(pattern), key=message_sequence):
            read_cycle_message(path=path)
            recovered_path, moved = verified_state_move(
                dispatch_path=path, directory=MAILBOX)
            if not moved:
                raise TicketCycleStateError(
                    "could not requeue pre-launch message "
                    + os.path.basename(path))
            recovered += 1
            print("requeued pre-launch message " + recovered_path)
        return recovered
    finally:
        release_mailbox_sequence_lock(lock_file=sequence_lock)


def restart_implementer_from_architect_handoff():
    """Discard interrupted implementation work and requeue its exact plan."""
    recover_interrupted_mailbox_moves()
    sequence_lock = acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise TicketCycleStateError("cannot lock Implementer restart")
    try:
        ticket_state = read_ticket_cycle_state()
        matches = []
        for directory in (
                MAILBOX, os.path.join(MAILBOX, "inflight"),
                os.path.join(MAILBOX, "failed"),
                os.path.join(MAILBOX, "prelaunch")):
            for path in glob.glob(os.path.join(directory, "*-to-opus.md")):
                message = read_cycle_message(path=path)
                cycle_id, mode, _body, problem = _ticket_flow_envelope(
                    message=message)
                active = ticket_state["active"].get(cycle_id)
                if (problem is not None or active is None
                        or active["phase"] != "implementation"
                        or active["commit"] is not None
                        or active["mode"] != mode):
                    continue
                if architect_handoff_problem(
                        message=message, cycle_id=cycle_id, mode=mode) is None:
                    matches.append((path, cycle_id))
        if len(matches) != 1:
            raise TicketCycleStateError(
                "Implementer restart needs exactly one active Architect "
                "handoff; found " + str(len(matches)))
        handoff, cycle_id = matches[0]
        if candidate_commit_for_cycle(cycle_id=cycle_id) is not None:
            raise TicketCycleStateError(
                "the Implementer already produced candidate C; return it "
                "to the Architect instead of restarting")
        for path in glob.glob(
                os.path.join(MAILBOX, "**", "*-to-fable.md"),
                recursive=True):
            message = read_cycle_message(path=path)
            returned_cycle, _mode, body, problem = _ticket_flow_envelope(
                message=message)
            if (problem is None and returned_cycle == cycle_id
                    and "### IMPLEMENTER_HANDOFF:" in body):
                raise TicketCycleStateError(
                    "the Implementer already returned work for this cycle; "
                    "send it to the Architect instead of restarting")

        worktree = AGENT_CWD["opus"]
        _symbolic_worktree_branch(
            worktree=worktree, expected_branch=IMPLEMENTER_BRANCH,
            label="Implementer")
        base = cycle_starting_commit(cycle_id=cycle_id)
        if not git_commit_exists(commit=base):
            raise TicketCycleStateError(
                "the Architect handoff names a missing base commit")
        _run_git(worktree, ["reset", "--hard", base])
        _run_git(worktree, ["clean", "-fd", "--", "."])
        if (worktree_head(worktree=worktree) != base
                or _clean_worktree_status(worktree=worktree)):
            raise TicketCycleStateError(
                "Implementer work could not be discarded cleanly")

        if os.path.dirname(handoff) != MAILBOX:
            recovered, moved = verified_state_move(
                dispatch_path=handoff, directory=MAILBOX)
            if not moved:
                raise TicketCycleStateError(
                    "the exact Architect handoff could not be requeued")
            handoff = recovered
        print("Architect handoff preserved: " + handoff)
        print("Interrupted Implementer work discarded; ticket base: " + base)
        print("Restart ready: launch --watch with the desired Implementer.")
        return handoff
    finally:
        release_mailbox_sequence_lock(lock_file=sequence_lock)


def restart_redteam_from_architect_handoff():
    """Discard interrupted Red Team work and requeue its exact request."""
    recover_interrupted_mailbox_moves()
    sequence_lock = acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise TicketCycleStateError("cannot lock Red Team restart")
    try:
        matches = []
        for directory in (
                MAILBOX, os.path.join(MAILBOX, "inflight"),
                os.path.join(MAILBOX, "failed"),
                os.path.join(MAILBOX, "prelaunch")):
            for path in glob.glob(os.path.join(directory, "*-to-sol.md")):
                message = read_cycle_message(path=path)
                kind = sol_ticket_kind(message=message)
                if kind in {"closure", "control-plane"}:
                    matches.append((path, message, kind))
        if len(matches) != 1:
            raise TicketCycleStateError(
                "Red Team restart needs exactly one active handoff; found "
                + str(len(matches)))
        handoff, message, kind = matches[0]
        audit_cycle = None
        audit_commit = None
        if kind == "closure":
            problem = redteam_closure_problem(message=message)
            if problem is not None:
                raise TicketCycleStateError(problem)
            audit_cycle = redteam_closure_ticket(message=message)
            audit_commit = redteam_closure_commit(message=message)
            if any_matching_redteam_receipt(
                    cycle_id=audit_cycle, accepted_commit=audit_commit):
                raise TicketCycleStateError(
                    "the Red Team already returned its review; send that "
                    "result to the Architect instead of restarting")
        elif kind == "control-plane":
            audit_cycle, audit_commit, _body, problem = (
                _redteam_control_plane_envelope(message=message))
            if problem is not None:
                raise TicketCycleStateError(problem)
            control = control_plane_ticket_state(
                cycle_id=audit_cycle, candidate_commit=audit_commit)
            if control is None or control["architect_candidate"] != (
                    audit_commit):
                raise TicketCycleStateError(
                    "the protected Red Team handoff lacks Architect GO(C)")
            if control["redteam_result"] is not None:
                raise TicketCycleStateError(
                    "the protected Red Team decision is already recorded")

        if audit_cycle is not None:
            discard_interrupted_audit_snapshot(
                cycle_id=audit_cycle, commit=audit_commit, agent="sol")
        worktree = AGENT_CWD["sol"]
        _symbolic_worktree_branch(
            worktree=worktree, expected_branch=SOL_BRANCH, label="Red Team")
        target = _exact_git_object(
            arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
            label="current main commit")
        _run_git(worktree, ["reset", "--hard", target])
        _run_git(worktree, ["clean", "-fd", "--", "."])
        if (worktree_head(worktree=worktree) != target
                or _clean_worktree_status(worktree=worktree)):
            raise TicketCycleStateError(
                "Red Team work could not be discarded cleanly")

        if os.path.dirname(handoff) != MAILBOX:
            recovered, moved = verified_state_move(
                dispatch_path=handoff, directory=MAILBOX)
            if not moved:
                raise TicketCycleStateError(
                    "the exact Red Team handoff could not be requeued")
            handoff = recovered
        print("Architect-to-Red-Team handoff preserved: " + handoff)
        print("Interrupted Red Team work discarded; baseline: " + target)
        print("Restart ready: launch --watch with Red Team enabled.")
        return handoff
    finally:
        release_mailbox_sequence_lock(lock_file=sequence_lock)


def recover_interrupted_mailbox_moves():
    """Collapse exact hardlink debris left by an interrupted state move."""
    sequence_lock = acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise TicketCycleStateError("cannot lock mailbox-move recovery")
    recovered = 0
    inflight_directory = os.path.join(MAILBOX, "inflight")
    try:
        names = set()
        for pattern in ("*.md", "*.md" + STATE_GUARD_SUFFIX):
            for path in glob.glob(os.path.join(inflight_directory, pattern)):
                names.add(blocker_message_name(path=path))
        for name in sorted(names, key=message_sequence):
            inflight = os.path.join(inflight_directory, name)
            guard = inflight + STATE_GUARD_SUFFIX
            root = os.path.join(MAILBOX, name)
            destinations = [
                os.path.join(DONE, name),
                os.path.join(MAILBOX, "failed", name),
                os.path.join(MAILBOX, "prelaunch", name),
            ]

            def inode(path):
                value = regular_inode(path=path)
                if value is None and os.path.lexists(path):
                    raise TicketCycleStateError(
                        "mailbox move recovery found a non-regular state: "
                        + path)
                return value

            inflight_inode = inode(inflight)
            guard_inode = inode(guard)
            root_inode = inode(root)
            terminal = [(path, inode(path)) for path in destinations]
            terminal = [(path, value) for path, value in terminal
                        if value is not None]
            known = [value for value in (inflight_inode, guard_inode)
                     if value is not None]
            if len(set(known)) > 1:
                raise TicketCycleStateError(
                    "interrupted mailbox move changed its guard identity")
            source_inode = known[0] if known else None

            if root_inode is not None:
                if (source_inode is None or root_inode != source_inode
                        or guard_inode is not None or terminal):
                    raise TicketCycleStateError(
                        "interrupted mailbox claim has conflicting states")
                os.unlink(inflight)
                fsync_directory(directory=inflight_directory)
                recovered += 1
                print("recovered interrupted claim " + name)
                continue

            if terminal:
                if (len(terminal) != 1 or source_inode is None
                        or terminal[0][1] != source_inode):
                    raise TicketCycleStateError(
                        "interrupted mailbox move has conflicting destinations")
                for leftover in (inflight, guard):
                    if os.path.lexists(leftover):
                        os.unlink(leftover)
                fsync_directory(directory=inflight_directory)
                recovered += 1
                print("finished interrupted mailbox move " + name)
                continue

            if inflight_inode is not None and guard_inode == inflight_inode:
                os.unlink(guard)
                fsync_directory(directory=inflight_directory)
                recovered += 1
                print("removed interrupted state guard for " + name)
            elif guard_inode is not None:
                raise TicketCycleStateError(
                    "interrupted mailbox guard has no recoverable source")
        return recovered
    finally:
        release_mailbox_sequence_lock(lock_file=sequence_lock)


def blocked_redteam_directory():
    """Return the durable queue for protected work that needs Red Team."""
    return os.path.join(MAILBOX, BLOCKED_REDTEAM_DIRECTORY)


def recover_blocked_redteam_messages(skip_redteam=False):
    """Keep old tool-edit requests parked for external maintenance."""
    directory = blocked_redteam_directory()
    parked = glob.glob(os.path.join(directory, "*-to-opus.md"))
    if parked and not skip_redteam:
        print("old protected tool requests remain parked for external "
              "ai/tools maintenance; their backlog tickets stay Open")
    return 0


def block_protected_ticket_without_redteam(path):
    """Durably block one validated protected handoff before reservation."""
    match = PENDING_MESSAGE_RE.fullmatch(os.path.basename(path))
    if match is None or match.group(1) != "opus":
        return False
    try:
        message = read_cycle_message(path=path)
        evidence = prepare_implementer_evidence_contract(message=message)
    except (OSError, ValueError, TicketCycleStateError):
        return False
    if evidence["ticket_class"] != "protected-control-plane":
        return False
    blocked = move_without_overwrite(
        path=path, directory=blocked_redteam_directory())
    if blocked is None:
        return False
    print("BLOCKED_RED_TEAM_REQUIRED: Protected control-plane tickets "
          "require Red Team review. This daemon was started with "
          "--skip-redteam, so " + os.path.basename(path)
          + " was not run. Restart without --skip-redteam; the exact "
            "request was preserved at " + blocked + ".")
    return True


def recover_before_dispatch(fix_only=False, skip_redteam=False):
    """Recover restart-safe mailbox state before a live dispatch pass."""
    failed_health = control_plane_health_failure()
    if failed_health is not None:
        cycle_id, evidence = failed_health
        raise TicketCycleStateError(
            "CONTROL_PLANE_HEALTH_FAILED: protected cycle " + cycle_id
            + " is recovery-only; inspect " + evidence
            + ", preserve its recorded landing, and repair it with the "
              "trusted controller before dispatching new work")
    recover_interrupted_mailbox_moves()
    recover_failed_architect_outcome()
    recover_failed_open_ticket_go()
    if fix_only:
        recover_failed_maintenance_admission()
    recover_failed_implementer_returns()
    recover_implementer_deliveries()
    recover_failed_implementer_preflight()
    recover_prelaunch_messages()
    recover_failed_public_architect_admissions()
    recover_blocked_redteam_messages(skip_redteam=skip_redteam)
    return reconcile_ticket_cycle_state()


def implementer_reservation_preflight_problem(path, message):
    """Return a permanent pre-launch problem before a slot is reserved."""
    if "\x00" in message:
        return "the message contains a NUL byte"
    if not valid_duration(value=DISPATCH_TIMEOUT_MINUTES,
                          strictly_positive=True):
        return "the dispatch timeout is invalid"
    try:
        timeout_events(name=os.path.basename(path))
    except (OSError, ValueError, json.JSONDecodeError,
            OverflowError, RecursionError) as exc:
        return "timeout history cannot be verified: " + str(exc)
    _, _, body, problem = _ticket_flow_envelope(message=message)
    if problem is None and placeholder_in(message=body) is not None:
        return "the Implementer body is only a template placeholder"
    return problem


def reserve_architect_ticket_before_claim(path, skip_redteam=False):
    """Durably charge one ticket-selecting request before Architect launch.

    The exact request basename and SHA-256 stay charged until the same turn's
    first Implementer handoff carries their admission token.  This closes the
    interval in which a finite watch used to launch two public requests before
    either one had reached the Implementer lane.
    """
    controller = _ACTIVE_WATCH_RENDEZVOUS
    if controller is None:
        return None, None
    name = os.path.basename(path)
    match = PENDING_MESSAGE_RE.fullmatch(name)
    if match is None or match.group(1) != "fable":
        return None, None
    try:
        message = read_cycle_message(path=path)
    except (OSError, ValueError, TicketCycleStateError):
        return None, None
    maintenance = message == ARCHITECT_FIX_ONLY_REQUEST
    finite = controller.ticket_cycle_limit_value() is not None
    if not finite and not maintenance:
        return None, None
    if (not maintenance
            and (not message.startswith(SOL_SEVERITY_HEADER)
            or architect_user_request_problem(message=message) is not None
            or placeholder_in(
                message=architect_user_request_body(message=message))
            is not None
            or "\x00" in message)):
        return None, None
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    topology = canonical_ticket_cycle_topology(
        skip_redteam=skip_redteam)
    record = {"mode": topology, "sequence": message_sequence(path),
              "sha256": digest}
    token = architect_admission_token(
        request_name=name, digest=digest)
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        existing = state["architect_admissions"].get(name)
        if existing is not None:
            if existing != record:
                raise TicketCycleStateError(
                    "saved public Architect admission changed identity")
            return None, token
        for earlier_path in pending_messages():
            if message_sequence(earlier_path) >= record["sequence"]:
                break
            earlier_match = PENDING_MESSAGE_RE.fullmatch(
                os.path.basename(earlier_path))
            if earlier_match is None or earlier_match.group(1) != "opus":
                continue
            try:
                earlier_message = read_cycle_message(path=earlier_path)
            except (OSError, ValueError, TicketCycleStateError):
                continue
            earlier_cycle, earlier_mode, _body, earlier_problem = (
                _ticket_flow_envelope(message=earlier_message))
            if (earlier_problem is None
                    and ticket_cycle_mode_is_enabled(
                        mode=earlier_mode, skip_redteam=skip_redteam)
                    and earlier_cycle not in state["active"]):
                return ("an earlier Implementer handoff must reserve its "
                        "ticket before this public request", None)
        if architect_notes_transition_pending():
            return ("a permanent-note admin turn or P landing is still "
                    "pending; no newer ticket may be admitted", None)
        used = finite_cycle_capacity_used(
            state=state, skip_redteam=skip_redteam)
        if (finite and used >= controller.ticket_cycle_limit_value()):
            return ("the finite watch has already reserved all "
                    + str(controller.ticket_cycle_limit_value())
                    + " ticket cycle(s)", None)
        state["architect_admissions"][name] = record
        write_ticket_cycle_state(state=state)
        return None, token
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def release_architect_ticket_admission(token):
    """Atomically retire one exact public request that created no ticket."""
    request_name, digest = split_architect_admission_token(token=token)
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        record = state["architect_admissions"].get(request_name)
        if record is None or record["sha256"] != digest:
            raise TicketCycleStateError(
                "public Architect admission changed before release")
        del state["architect_admissions"][request_name]
        write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def reserve_implementer_ticket_before_claim(path, skip_redteam=False):
    """Reserve one finite-watch ticket slot while its root file is untouched.

    Malformed or otherwise invalid messages are left for the ordinary
    dispatch validator, which can park them with a concrete explanation. Only
    the positive-cycle capacity refusal is returned here because it is a
    valid ticket for a later watch, not a failed message.
    """
    match = PENDING_MESSAGE_RE.match(os.path.basename(path))
    if match is None or match.group(1) != "opus":
        return None, None
    try:
        message = read_cycle_message(path=path)
    except (OSError, ValueError, TicketCycleStateError):
        return None, None
    if not message.startswith(MAILBOX_FLOW_HEADER):
        return None, None
    preflight_problem = implementer_reservation_preflight_problem(
        path=path, message=message)
    if preflight_problem is not None:
        return None, None
    try:
        evidence = prepare_implementer_evidence_contract(message=message)
    except (OSError, TicketCycleStateError):
        # Reserve capacity before claim even when the later dispatch
        # validator will refuse a malformed ordinary handoff. If no child
        # starts, drain_lane releases this provisional reservation. A valid
        # protected handoff always reaches the successful branch below and
        # therefore freezes its real class and path scope.
        evidence = {
            "allowed_paths": None,
            "ticket_class": "ordinary",
        }
    try:
        _, _, created = register_ticket_cycle_message(
            agent="opus", message=message,
            skip_redteam=skip_redteam,
            return_reservation=True,
            implementer_request_name=os.path.basename(path),
            path_scope=evidence["allowed_paths"],
            ticket_class=evidence["ticket_class"])
    except TicketCycleLimitDeferred as exc:
        return str(exc), None
    except (OSError, TicketCycleStateError):
        return None, None
    if not created:
        return None, None
    parsed_cycle, _, _, problem = _ticket_flow_envelope(message=message)
    admission_name, _admission_digest, admission_problem = (
        _ticket_architect_admission(message=message))
    converted_public_admission = (
        admission_problem is None and admission_name is not None)
    return None, (parsed_cycle if (problem is None
                                   and not converted_public_admission)
                  else None)


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
        if _TOKEN_EXHAUSTION_STOP.is_set():
            all_consumed = False
            break
        try:
            maintenance = (read_cycle_message(path=path)
                           == ARCHITECT_FIX_ONLY_REQUEST)
        except (OSError, ValueError, TicketCycleStateError):
            maintenance = False
        if skip_redteam and block_protected_ticket_without_redteam(path=path):
            # This configuration refusal is a consumed queue action, not a
            # ticket cycle and not an Implementer failure. Continue to any
            # ordinary two-role work behind it.
            continue
        if (maintenance
                and (not fix_only
                     or active_ticket_cycle_count(
                         skip_redteam=skip_redteam,
                         exclude_admission=os.path.basename(path))
                     or any(candidate.endswith("-to-opus.md")
                            for candidate in pending_messages()))):
            if not fix_only:
                print("deferred " + os.path.basename(path)
                      + ": needs a --fix-only watcher; left queued.")
            all_consumed = False
            continue
        notes_admin = False
        try:
            notes_admin = regular_file_has_prefix(
                path=path,
                prefix=MAILBOX_ADMIN_HEADER.encode("ascii"))
        except (OSError, ValueError):
            pass
        controller = (_ACTIVE_WATCH_RENDEZVOUS
                      if not dry_run else None)
        if (controller is not None
                and controller.ticket_cycle_limit_reached()
                and not notes_admin):
            # The message remains in the mailbox root for a later watch. A
            # child already launched in another lane may still finish, but no
            # additional turn is admitted after the requested count returns.
            break
        permit = None
        if controller is not None:
            permit = controller.begin_attempt(
                ignore_ticket_limit=notes_admin)
            if permit is None:
                # A watch-global rendezvous is due.  Leave this exact root
                # message untouched; main performs the safe window only after
                # every lane worker has returned.
                break
            _RENDEZVOUS_LOCAL.permit = permit
        new_reservation_cycle = None
        architect_admission = None
        consumed = False
        try:
            if not dry_run:
                deferred, architect_admission = (
                    reserve_architect_ticket_before_claim(
                        path=path, skip_redteam=skip_redteam))
                if deferred is not None:
                    print("deferred " + os.path.basename(path) + ": "
                          + deferred + "; root message remains untouched.")
                    continue
                deferred, new_reservation_cycle = (
                    reserve_implementer_ticket_before_claim(
                        path=path,
                        skip_redteam=skip_redteam))
                if deferred is not None:
                    print("deferred " + os.path.basename(path) + ": "
                          + deferred + "; root message remains untouched.")
                    # A later file may continue an already reserved ticket.
                    continue
            if skip_redteam:
                consumed = dispatch(
                    path=path, dry_run=dry_run, fix_only=fix_only,
                    skip_redteam=True,
                    new_reservation_cycle=new_reservation_cycle,
                    architect_admission=architect_admission)
            else:
                consumed = dispatch(
                    path=path, dry_run=dry_run, fix_only=fix_only,
                    new_reservation_cycle=new_reservation_cycle,
                    architect_admission=architect_admission)
        finally:
            if controller is not None:
                try:
                    if (new_reservation_cycle is not None
                            and not consumed
                            and not permit.launched):
                        release_unstarted_ticket_reservation(
                            cycle_id=new_reservation_cycle)
                    del _RENDEZVOUS_LOCAL.permit
                finally:
                    controller.finish_attempt(permit=permit)
        if not consumed and architect_admission is not None:
            retire_failed_public_architect_admission(
                path=os.path.join(
                    MAILBOX, "failed", os.path.basename(path)))
        if not consumed:
            all_consumed = False
            # A false result can mean the head is still inflight because its
            # archive or failed-state move was ambiguous. Do not release later
            # work in the same lane past an unresolved head.
            break
    return all_consumed


def process_backlog(dry_run, fix_only=False, skip_redteam=False):
    """Dispatch the whole backlog: lanes in PARALLEL, each lane in order.

    Live topology gives each role a separate saved working directory. The
    Architect may audit one frozen candidate while the Implementer advances
    another cycle and Sol reviews an exact daemon-recorded landing L. Two
    messages to the same role remain sequential, and imported tests that
    deliberately share a cwd remain serialized. The parallel unit is still
    the cwd. Architect decisions and parent-daemon landing transitions share
    one root lock; the Implementer and Red Team lanes do not take it.

    Arguments:
      dry_run  = True to print the would-be commands without running them.
      fix_only = True when a watch is closing existing ledger work only.
      skip_redteam = True for a watch that dispatches only Architect and
                     Implementer routes.

    Returns:
      None when there was no backlog, True when every message was consumed
      (or would dispatch in a dry run), and False when any dispatch or done
      archive failed. ROLE_CONTRACT_RESTART_REQUIRED stops a pass after a
      protected contract update, before another message can start.
    """
    _TOKEN_EXHAUSTION_STOP.clear()
    if not dry_run:
        try:
            read_ticket_cycle_state()
        except (OSError, ValueError, TicketCycleStateError) as exc:
            print("refused mailbox pass: cannot verify ticket-cycle state ("
                  + str(exc) + "); no new role work was started. Repair the "
                  "saved ticket-cycle state, then run the watcher again.")
            return False
    all_backlog = pending_messages()
    all_daemon_paths = [
        path for path in all_backlog
        if PENDING_MESSAGE_RE.match(os.path.basename(path)).group(1)
        == "daemon"]
    daemon_paths = [
        path for path in all_daemon_paths
        if message_is_enabled_for_topology(
            path=path, skip_redteam=skip_redteam)]
    policy_problem = role_contract_snapshot_problem()
    policy_recovery_only = (
        policy_problem is not None and architect_notes_transition_pending())
    if policy_recovery_only:
        # The Architect primary already contains a proposed new contract.
        # The old process may finish only that exact P landing. It must not
        # land an ordinary candidate or start a role under stale policy.
        daemon_paths = [
            path for path in daemon_paths
            if regular_file_has_prefix(
                path=path,
                prefix=(MAILBOX_RETURN_HEADER
                        + "architect-notes-go").encode("ascii"))]
    daemon_outcome = True
    for daemon_path in daemon_paths:
        # This GO belongs to a ticket already admitted against the finite
        # limit. Always finish its durable landing/archive recovery. The
        # positive limit gates new role work in drain_lane(), never this
        # already-admitted daemon transition.
        outcome = consume_daemon_message(
            path=daemon_path, dry_run=dry_run, return_outcome=True)
        if outcome == DAEMON_NOTE_DEFERRED:
            # An unlanded P can wait behind a later, already-admitted
            # ordinary GO. Continue the daemon lane so that exact ticket can
            # reach L and clear P's idle-boundary requirement.
            if policy_recovery_only:
                return ROLE_CONTRACT_RESTART_REQUIRED
            daemon_outcome = False
            continue
        if outcome == DAEMON_CONTROL_PLANE_WAITING:
            # The exact GO remains in inflight/ while Sol supplies the
            # second key. Compatible role work, including that review, may
            # continue in this watch.
            continue
        if outcome != DAEMON_MESSAGE_CONSUMED:
            if policy_recovery_only:
                return ROLE_CONTRACT_RESTART_REQUIRED
            daemon_outcome = False
            break
        # Check after each daemon message, not after the complete lane. A P
        # landing therefore cannot release a second daemon request or a role
        # while this process still holds the old policy snapshot.
        if (policy_recovery_only
                or (policy_problem is None
                    and role_contract_snapshot_problem() is not None)):
            return ROLE_CONTRACT_RESTART_REQUIRED
    if policy_recovery_only:
        return ROLE_CONTRACT_RESTART_REQUIRED
    agent_backlog = [path for path in all_backlog
                     if path not in all_daemon_paths]
    backlog = [
        path for path in agent_backlog
        if message_is_enabled_for_topology(
            path=path, skip_redteam=skip_redteam)]
    if skip_redteam:
        blockers = inflight_lane_blockers(skip_redteam=True)
    else:
        blockers = inflight_lane_blockers()
    admin_paths = []
    for candidate in backlog:
        match = PENDING_MESSAGE_RE.match(os.path.basename(candidate))
        if match is None or match.group(1) != "fable":
            continue
        try:
            admin_prefix = regular_file_has_prefix(
                path=candidate,
                prefix=MAILBOX_ADMIN_HEADER.encode("ascii"))
        except (OSError, ValueError):
            admin_prefix = False
        if admin_prefix:
            admin_paths.append(candidate)
    if admin_paths:
        admin_paths.sort(key=message_sequence)
        admin_path = admin_paths[0]
        boundary = message_sequence(admin_path)
        state = read_ticket_cycle_state()
        limit_reached = (
            _ACTIVE_WATCH_RENDEZVOUS is not None
            and _ACTIVE_WATCH_RENDEZVOUS.ticket_cycle_limit_reached())
        earlier = ([] if limit_reached else [
            candidate for candidate in backlog
            if (candidate != admin_path
                and message_sequence(candidate) < boundary)])
        admitted = [candidate for candidate in backlog
                    if (candidate != admin_path
                        and message_belongs_to_active_cycle(
                            path=candidate,
                            active_cycles=state["active"]))]
        older_work = list(dict.fromkeys(earlier + admitted))
        admin_problem = None
        try:
            require_no_ordinary_landing_transition(
                current_dispatch_path=admin_path)
        except (OSError, TicketCycleStateError) as exc:
            admin_problem = str(exc)
        if not older_work and not blockers and admin_problem is None:
            # An eligible note turn is the sole launch in this pass.  The
            # dispatch itself holds main->ticket locks across the child, so a
            # second watcher cannot reserve Opus during B-to-P creation.
            backlog = [admin_path]
        else:
            # Keep the admin root and every later request untouched.  Only
            # work that was already waiting or belongs to an admitted older
            # cycle may advance toward the idle boundary.
            backlog = older_work
            explanation = (admin_problem if admin_problem is not None else
                           "older mailbox work or an inflight lane remains")
            print("deferred " + os.path.basename(admin_path)
                  + ": permanent-note administration waits for an idle "
                  "boundary (" + explanation + ").")
    elif architect_notes_transition_pending():
        # A validated P request may wait behind an older admitted cycle.
        # Continue that cycle, but never admit unrelated/newer work before P
        # reaches main and the clean role baselines.
        active = read_ticket_cycle_state()["active"]
        backlog = [candidate for candidate in backlog
                   if message_belongs_to_active_cycle(
                       path=candidate, active_cycles=active)]
    # Finish an admitted ticket before an older, unrelated user request in
    # the same role lane. Otherwise recovery mail can wait forever behind a
    # request that the finite cycle limit cannot yet admit.
    active = read_ticket_cycle_state()["active"]
    backlog.sort(key=lambda path: (
        not message_belongs_to_active_cycle(path=path, active_cycles=active),
        message_sequence(path)))
    if all_backlog or daemon_paths:
        if skip_redteam:
            report_demand(backlog=all_backlog, skip_redteam=True)
        else:
            report_demand(backlog=all_backlog)
    if skip_redteam:
        report_deferred_sol_messages()
    if not backlog:
        if not blockers:
            return daemon_outcome if daemon_paths else None
        for cwd in sorted(blockers):
            report_inflight_lane_block(
                blocker_paths=blockers[cwd],
                pending_count=0)
        return False
    lanes = {}
    for path in backlog:
        name = os.path.basename(path)
        agent = PENDING_MESSAGE_RE.match(name).group(1)
        cwd = mailbox_lane_cwd(agent=agent)
        if cwd not in lanes:
            lanes[cwd] = []
        lanes[cwd].append(path)
    # An inflight message predating this pass represents an unresolved turn:
    # it may have edited the shared tree even though its archive failed. Do
    # not release later work in that working-directory lane on a subsequent
    # watch pass. Other cwd lanes remain independent and may still drain.
    workers = []
    lane_outcomes = {}
    token_errors = []
    authority_errors = []
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
        except RoleTokenExhaustionError as exc:
            with outcome_lock:
                token_errors.append(exc)
            consumed = False
        except ImplementerAuthorityViolationError as exc:
            with outcome_lock:
                authority_errors.append(exc)
            consumed = False
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
    if not dry_run:
        recover_failed_public_architect_admissions()
    if token_errors:
        order = {"fable": 0, "opus": 1, "sol": 2}
        ordered = sorted(token_errors, key=lambda error: order[error.agent])
        ordered[0].other_errors = ordered[1:]
        raise ordered[0]
    if authority_errors:
        raise authority_errors[0]
    return (daemon_outcome and not blockers
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
    """Print queue depth and the classified backlog counts.

    Queue depth is informational. New-discovery admission counts open
    Critical, High, and Medium tickets. Low tickets do not stop discovery.
    Backlog counts never select Sol's role.

    Arguments:
      backlog = Current waiting message paths from pending_messages().
    """
    depth = {"fable": 0, "opus": 0, "sol": 0, "daemon": 0}
    for path in backlog:
        name = os.path.basename(path)
        agent = PENDING_MESSAGE_RE.match(name).group(1)
        depth[agent] = depth[agent] + 1
    counts = backlog_severity_counts()
    ledger = (counts["critical"] + counts["high"] + counts["medium"]
              + counts["low"] + counts["unclassified"])
    admission = counts["critical"] + counts["high"] + counts["medium"]
    print("queue depth: opus=" + str(depth["opus"])
          + " sol=" + str(depth["sol"])
          + " fable=" + str(depth["fable"])
          + " daemon=" + str(depth["daemon"])
          + " | open backlog: critical=" + str(counts["critical"])
          + " high=" + str(counts["high"])
          + " medium=" + str(counts["medium"])
          + " low=" + str(counts["low"])
          + " unclassified=" + str(counts["unclassified"])
          + " | all open: " + str(ledger)
          + " | discovery admission count: " + str(admission))
    if counts["problem"] is not None:
        print("  warning: " + counts["problem"])
    if counts["unclassified"]:
        print("  warning: classify every open backlog ticket before new "
              "discovery; an unclassified ticket fails closed.")
    report_landing_debt()


def landing_debt_snapshot():
    """Measure only saved, unlanded Implementer candidates.

    The Architect primary branch is a planning lane, so comparing that branch
    with main mistakes completed landings and protected-note edits for code
    awaiting review. Candidate refs plus active ticket state are the daemon's
    durable authority for work that can still need an Architect decision.
    """
    lock_file = acquire_ticket_cycle_lock()
    try:
        ticket_state = read_ticket_cycle_state()
        candidate_state = read_candidate_state()
        changed_lines = 0
        diff_ranges = []
        for cycle_id, saved in candidate_state["cycles"].items():
            active = ticket_state["active"].get(cycle_id)
            if active is None and cycle_id not in ticket_state["completed"]:
                raise TicketCycleStateError(
                    "candidate debt has no active or completed ticket")
            if active is None or active["phase"] != "implementation":
                # A candidate retained across GO archival or checkout handoff
                # is recovery authority, not new work needing another audit.
                continue
            record = candidate_record_locked(
                cycle_id=cycle_id, ticket_state=ticket_state,
                candidate_state=candidate_state, recover=False)
            if record is None or record != saved:
                raise TicketCycleStateError(
                    "active candidate debt lost its durable identity")
            base = cycle_starting_commit(cycle_id)
            diff_ranges.append((base, saved["commit"]))
        for base, candidate in diff_ranges:
            process = subprocess.run(
                ["git", "diff", "--shortstat", base + ".." + candidate],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=AGENT_CWD["fable"], check=False)
            if process.returncode != 0:
                return {
                    "available": False, "stat": "", "changed_lines": 0,
                    "returncode": process.returncode}
            for count, _keyword in re.findall(
                    r"(\d+) (insertion|deletion)", process.stdout):
                changed_lines = changed_lines + int(count)
        active_candidates = len(diff_ranges)
    except (OSError, ValueError, TicketCycleStateError):
        return {
            "available": False, "stat": "", "changed_lines": 0,
            "returncode": 1}
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)
    stat_line = ""
    if active_candidates:
        noun = "candidate" if active_candidates == 1 else "candidates"
        stat_line = (str(active_candidates) + " active " + noun + ", "
                     + str(changed_lines) + " changed lines")
    return {
        "available": True, "stat": stat_line,
        "changed_lines": changed_lines, "returncode": 0}


def report_landing_debt(snapshot=None):
    """Print saved candidate size without treating role branches as debt."""
    if snapshot is None:
        snapshot = landing_debt_snapshot()
    if not snapshot["available"]:
        print("landing debt: unavailable; active candidate state could not "
              "be measured (check exited "
              + str(snapshot["returncode"]) + ")")
        return snapshot
    if snapshot["stat"] == "":
        print("landing debt: none; no saved active candidate is waiting")
        return snapshot
    print("landing debt: " + snapshot["stat"])
    if snapshot["changed_lines"] > LANDING_DEBT_LINE_LIMIT:
        print("  hint: more than " + str(LANDING_DEBT_LINE_LIMIT)
              + " unlanded lines means at least one full audit trail "
              "is overdue; squash-land the audited unit(s) to main "
              "now, one unit per commit "
              "(.claude/FABLE_ROLE.md, Landing granularity).")
    return snapshot


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


def acquire_mailbox_sequence_lock():
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
        print("mailbox publication blocked: sequence lock failed ("
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
        print("mailbox publication blocked: sequence lock failed ("
              + str(exc) + ").")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()
        return None
    return lock_file


def release_mailbox_sequence_lock(lock_file):
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


class TicketCycleStateError(RuntimeError):
    """The daemon-owned ticket-cycle record is unsafe or inconsistent."""


class BacklogTicketOpenError(TicketCycleStateError):
    """An accepted candidate still needs Architect backlog bookkeeping."""


class RetryableArchitectLandingError(TicketCycleStateError):
    """A valid GO is preserved until the user's main checkout is safe."""


class FatalArchitectLandingError(TicketCycleStateError):
    """Stop this process after preserving a valid GO for a later retry."""


class ImplementerAuthorityViolationError(RuntimeError):
    """The Implementer turn coincided with forbidden Git-state movement."""

    def __init__(self, changes):
        self.changes = tuple(changes)
        super().__init__("Implementer authority boundary changed")


class RoleTokenExhaustionError(RuntimeError):
    """One role exhausted its account."""

    ROLE_NAMES = {"fable": "Architect", "opus": "Implementer", "sol": "Sol"}

    def __init__(self, agent, request_path):
        self.agent = agent
        self.role = self.ROLE_NAMES[agent]
        self.request_path = request_path
        self.worktree = AGENT_CWD[agent]
        self.other_errors = []
        super().__init__("Error: " + self.role + " is out of tokens")


def report_role_token_exhaustion(error):
    """Report exhausted roles and preserved work.

    Arguments:
      error = Role exhaustion from a joined dispatch pass.

    Returns:
      None; prints four lines per role.
    """
    for stopped in [error] + error.other_errors:
        print(str(stopped))
        print("Work was preserved in " + stopped.worktree + ".")
        if stopped.request_path is None:
            print("Request preservation is uncertain; inspect inflight/ and "
                  "failed/ before retrying.")
        else:
            print("Request saved at " + stopped.request_path + ".")
        print("Add credits before retrying.")


class TicketCycleLimitDeferred(TicketCycleStateError):
    """A valid new ticket belongs to a later finite watch."""


def ticket_cycle_state_path():
    """Return the ignored daemon-owned ticket-cycle state path."""
    return os.path.join(MAILBOX, TICKET_CYCLE_STATE_NAME)


def empty_ticket_cycle_state():
    """Return a new strict ticket-cycle state value."""
    return {
        "schema": TICKET_CYCLE_STATE_SCHEMA,
        "generation": 0,
        "pending_cycle_returns": 0,
        "finite_watch": None,
        "architect_admissions": {},
        "active": {},
        "completed": {},
        "control_plane_history": {},
    }


def empty_control_plane_state():
    """Return the durable two-key state for one protected candidate."""
    return {
        "architect_candidate": None,
        "redteam_result": None,
        "redteam_candidate": None,
        "shadow_status": None,
        "shadow_evidence": None,
        "integration_status": None,
        "integration_main": None,
        "stale_landing": None,
        "stale_parent": None,
        "integration_evidence": None,
        "health_status": None,
        "health_evidence": None,
    }


def validate_control_plane_relationships(
        control, phase=None, completed_candidate=None):
    """Refuse protected state whose decisions and evidence disagree."""
    architect = control["architect_candidate"]
    redteam = control["redteam_candidate"]
    result = control["redteam_result"]
    shadow = control["shadow_status"]
    shadow_evidence = control["shadow_evidence"]
    integration = control["integration_status"]
    integration_evidence = control["integration_evidence"]
    health = control["health_status"]
    health_evidence = control["health_evidence"]

    if (redteam is None) != (result is None):
        raise TicketCycleStateError(
            "protected Red Team candidate and decision must appear together")
    if result is not None and (architect is None or architect != redteam):
        raise TicketCycleStateError(
            "protected decisions do not accept the same exact C")
    accepted = (architect is not None and redteam == architect
                and result == "ACCEPT-CONTROL-PLANE")

    if (shadow is None) != (shadow_evidence is None):
        raise TicketCycleStateError(
            "protected shadow result and evidence must appear together")
    if shadow is not None and not accepted:
        raise TicketCycleStateError(
            "protected shadow lacks exact-C acceptance")

    integration_values = (
        control["integration_main"], control["stale_landing"],
        control["stale_parent"])
    if integration is None:
        if any(value is not None for value in
               integration_values + (integration_evidence,)):
            raise TicketCycleStateError(
                "protected ticket has incomplete integration state")
    elif any(value is None for value in integration_values):
        raise TicketCycleStateError(
            "protected ticket lacks stale integration identity")
    elif not accepted:
        raise TicketCycleStateError(
            "protected integration lacks exact-C acceptance")
    elif integration == "STALE":
        if integration_evidence is not None or shadow is not None:
            raise TicketCycleStateError(
                "stale protected integration carries later evidence")
    elif integration_evidence is None:
        raise TicketCycleStateError(
            "protected integration revalidation lacks evidence")

    if (health is None) != (health_evidence is None):
        raise TicketCycleStateError(
            "protected health result and evidence must appear together")
    if health is not None and (not accepted or shadow != "PASSED"):
        raise TicketCycleStateError(
            "protected health check lacks accepted PASSED shadow evidence")
    if phase == "implementation" and health is not None:
        raise TicketCycleStateError(
            "unlanded protected ticket carries a health result")
    if phase == "awaiting-redteam":
        raise TicketCycleStateError(
            "protected ticket cannot enter ordinary Red Team closure")
    if phase == "committed-awaiting-closure" and (
            not accepted or shadow != "PASSED"):
        raise TicketCycleStateError(
            "landed protected ticket lacks exact-C acceptance and shadow")

    if completed_candidate is not None:
        if (architect != completed_candidate
                or redteam != completed_candidate
                or result != "ACCEPT-CONTROL-PLANE"):
            raise TicketCycleStateError(
                "completed control-plane history lacks exact accepted C")
        if shadow != "PASSED" or shadow_evidence is None:
            raise TicketCycleStateError(
                "completed control-plane history lacks PASSED shadow evidence")
        if health != "HEALTHY" or health_evidence is None:
            raise TicketCycleStateError(
                "completed control-plane history lacks HEALTHY evidence")
        if integration == "STALE":
            raise TicketCycleStateError(
                "completed control-plane history retains a stale integration")


def control_plane_health_failure(state=None):
    """Return the first durable failed promotion, if one exists."""
    current = read_ticket_cycle_state() if state is None else state
    for cycle_id in sorted(current["active"]):
        record = current["active"][cycle_id]
        control = record.get("control_plane")
        if (record.get("ticket_class") == "protected-control-plane"
                and isinstance(control, dict)
                and control.get("health_status")
                == "CONTROL_PLANE_HEALTH_FAILED"):
            return cycle_id, control.get("health_evidence") or "saved state"
    return None


def validate_ticket_cycle_state(payload):
    """Return current ticket-cycle state; refuse every retired schema."""
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if schema != TICKET_CYCLE_STATE_SCHEMA:
        raise TicketCycleStateError(
            "saved ticket-cycle state uses an unsupported old schema; "
            "stop every older watcher, preserve the state for inspection, "
            "then remove or reinitialize it deliberately")
    required = {"schema", "generation", "active", "completed",
                "pending_cycle_returns", "finite_watch",
                "architect_admissions"}
    optional = {"control_plane_history"}
    if (not isinstance(payload, dict) or not required.issubset(payload)
            or not set(payload).issubset(required | optional)):
        raise TicketCycleStateError("ticket-cycle state has wrong keys")
    generation = payload.get("generation")
    pending_cycle_returns = payload.get("pending_cycle_returns")
    if (payload.get("schema") != TICKET_CYCLE_STATE_SCHEMA
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0 or generation > MAX_CYCLE_COUNT):
        raise TicketCycleStateError("ticket-cycle state has invalid identity")
    if (isinstance(pending_cycle_returns, bool)
            or not isinstance(pending_cycle_returns, int)
            or pending_cycle_returns < 0
            or pending_cycle_returns > generation):
        raise TicketCycleStateError(
            "ticket-cycle state has invalid pending cycle returns")
    active = payload.get("active")
    completed = payload.get("completed")
    architect_admissions = payload.get("architect_admissions")
    finite_watch = payload.get("finite_watch")
    control_plane_history = payload.get("control_plane_history", {})
    if (not isinstance(active, dict) or not isinstance(completed, dict)
            or not isinstance(architect_admissions, dict)):
        raise TicketCycleStateError("ticket-cycle collections are invalid")
    if (len(active) + len(completed) + len(architect_admissions)
            > MAX_TICKET_CYCLE_RECORDS):
        raise TicketCycleStateError("ticket-cycle state has too many records")
    normalized_admissions = {}
    for request_name, record in architect_admissions.items():
        match = (PENDING_MESSAGE_RE.fullmatch(request_name)
                 if isinstance(request_name, str) else None)
        if (match is None or match.group(1) != "fable"
                or not isinstance(record, dict)
                or set(record) != {"mode", "sequence", "sha256"}):
            raise TicketCycleStateError(
                "invalid Architect ticket admission")
        sequence = record.get("sequence")
        digest = record.get("sha256")
        mode = record.get("mode")
        if (isinstance(sequence, bool) or not isinstance(sequence, int)
                or sequence != sequence_in_name(request_name)
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or mode not in ARCHITECT_COMMIT_MODES):
            raise TicketCycleStateError(
                "Architect ticket admission has invalid fields")
        normalized_admissions[request_name] = {
            "mode": mode, "sequence": sequence, "sha256": digest}
    normalized_active = {}
    for cycle_id, record in active.items():
        required_record_keys = {"phase", "commit", "mode", "route"}
        optional_record_keys = {
            "path_scope", "ticket_class", "control_plane"}
        if (not isinstance(cycle_id, str)
                or CYCLE_ID_RE.fullmatch(cycle_id) is None
                or not isinstance(record, dict)
                or not required_record_keys.issubset(record)
                or not set(record).issubset(
                    required_record_keys | optional_record_keys)):
            raise TicketCycleStateError("invalid active ticket-cycle record")
        phase = record.get("phase")
        commit = record.get("commit")
        mode = record.get("mode")
        route = record.get("route")
        path_scope = record.get("path_scope")
        ticket_class = record.get("ticket_class", "ordinary")
        control_plane = record.get("control_plane")
        if ticket_class not in TICKET_CLASSES:
            raise TicketCycleStateError("ticket class is invalid")
        if ticket_class == "protected-control-plane":
            if mode != "normal" or not isinstance(control_plane, dict) \
                    or set(control_plane) != set(empty_control_plane_state()):
                raise TicketCycleStateError(
                    "protected ticket lacks its exact two-key state")
            for field in ("architect_candidate", "redteam_candidate"):
                value = control_plane[field]
                if (value is not None
                        and (not isinstance(value, str)
                             or FULL_COMMIT_RE.fullmatch(value) is None)):
                    raise TicketCycleStateError(
                        "protected ticket has an invalid candidate decision")
            if control_plane["redteam_result"] not in (
                    None,) + CONTROL_PLANE_REVIEW_RESULTS:
                raise TicketCycleStateError(
                    "protected ticket has an invalid Red Team decision")
            if control_plane["shadow_status"] not in (
                    None, "PASSED", "FAILED"):
                raise TicketCycleStateError(
                    "protected ticket has invalid shadow state")
            if control_plane["integration_status"] not in (
                    None, "STALE", "REVALIDATED"):
                raise TicketCycleStateError(
                    "protected ticket has invalid integration state")
            if control_plane["health_status"] not in (
                    None, "HEALTHY", "CONTROL_PLANE_HEALTH_FAILED"):
                raise TicketCycleStateError(
                    "protected ticket has invalid health state")
            for field in ("integration_main", "stale_landing",
                          "stale_parent"):
                value = control_plane[field]
                if (value is not None
                        and (not isinstance(value, str)
                             or FULL_COMMIT_RE.fullmatch(value) is None)):
                    raise TicketCycleStateError(
                        "protected ticket has invalid integration identity")
            for field in ("shadow_evidence", "integration_evidence",
                          "health_evidence"):
                value = control_plane[field]
                if value is not None and (not isinstance(value, str)
                                          or not value
                                          or len(value) > 4096):
                    raise TicketCycleStateError(
                        "protected ticket has invalid evidence location")
        elif control_plane is not None:
            raise TicketCycleStateError(
                "ordinary ticket unexpectedly has protected state")
        if phase not in {"implementation", "committed-awaiting-closure",
                         "awaiting-redteam"}:
            raise TicketCycleStateError("invalid active ticket-cycle phase")
        if phase == "implementation" and commit is not None:
            raise TicketCycleStateError(
                "implementation cycle unexpectedly names landing L")
        if mode not in ARCHITECT_COMMIT_MODES:
            raise TicketCycleStateError("ticket-cycle mode is invalid")
        if route != "primary":
            raise TicketCycleStateError(
                "ticket-cycle mode conflicts with its Implementer route")
        if (phase != "implementation"
                and (not isinstance(commit, str)
                     or FULL_COMMIT_RE.fullmatch(commit) is None)):
            raise TicketCycleStateError(
                "committed ticket cycle lacks a full daemon-recorded "
                "landing L")
        expected_modes = {
            "committed-awaiting-closure": {"normal"},
            "awaiting-redteam": {"normal"},
        }
        if phase != "implementation" and mode not in expected_modes[phase]:
            raise TicketCycleStateError("ticket-cycle mode conflicts with phase")
        if ticket_class == "protected-control-plane":
            validate_control_plane_relationships(
                control=control_plane, phase=phase)
        if path_scope is not None:
            if (not isinstance(path_scope, list) or not path_scope
                    or len(path_scope) > 256
                    or any(not isinstance(path, str) for path in path_scope)
                    or path_scope != sorted(set(path_scope))):
                raise TicketCycleStateError("ticket path scope is invalid")
            for path in path_scope:
                parts = path.split("/")
                if (not parts or any(part in {"", ".", ".."} for part in parts)
                        or path.startswith("/") or "\\" in path
                        or any(mark in path for mark in "*?[]{}")
                        or not path.isprintable()):
                    raise TicketCycleStateError("ticket path scope is invalid")
        normalized = {
            "phase": phase, "commit": commit, "mode": mode,
            "route": route, "ticket_class": ticket_class}
        if "path_scope" in record:
            normalized["path_scope"] = path_scope
        if ticket_class == "protected-control-plane":
            normalized["control_plane"] = dict(control_plane)
        normalized_active[cycle_id] = normalized
    normalized_completed = {}
    for cycle_id, commit in completed.items():
        if (not isinstance(cycle_id, str)
                or CYCLE_ID_RE.fullmatch(cycle_id) is None
                or not isinstance(commit, str)
                or FULL_COMMIT_RE.fullmatch(commit) is None):
            raise TicketCycleStateError("invalid completed ticket-cycle record")
        if cycle_id in normalized_active:
            raise TicketCycleStateError(
                "ticket cycle is both active and completed")
        normalized_completed[cycle_id] = commit
    if (not isinstance(control_plane_history, dict)
            or len(control_plane_history) > MAX_TICKET_CYCLE_RECORDS):
        raise TicketCycleStateError("control-plane history is invalid")
    normalized_control_history = {}
    for cycle_id, record in control_plane_history.items():
        if (cycle_id not in normalized_completed
                or not isinstance(record, dict)
                or set(record) != {"candidate", "landing", "control_plane"}
                or not isinstance(record.get("candidate"), str)
                or FULL_COMMIT_RE.fullmatch(record["candidate"]) is None
                or record.get("landing") != normalized_completed[cycle_id]
                or not isinstance(record.get("control_plane"), dict)
                or set(record["control_plane"])
                != set(empty_control_plane_state())):
            raise TicketCycleStateError(
                "completed control-plane record is invalid")
        control = record["control_plane"]
        for field in ("architect_candidate", "redteam_candidate"):
            value = control[field]
            if (not isinstance(value, str)
                    or FULL_COMMIT_RE.fullmatch(value) is None):
                raise TicketCycleStateError(
                    "completed control-plane decision is invalid")
        if control["redteam_result"] not in CONTROL_PLANE_REVIEW_RESULTS:
            raise TicketCycleStateError(
                "completed control-plane Red Team result is invalid")
        if control["shadow_status"] not in (None, "PASSED", "FAILED"):
            raise TicketCycleStateError(
                "completed control-plane shadow result is invalid")
        if control["integration_status"] not in (
                None, "STALE", "REVALIDATED"):
            raise TicketCycleStateError(
                "completed control-plane integration result is invalid")
        if control["health_status"] not in (
                None, "HEALTHY", "CONTROL_PLANE_HEALTH_FAILED"):
            raise TicketCycleStateError(
                "completed control-plane health result is invalid")
        for field in ("integration_main", "stale_landing",
                      "stale_parent"):
            value = control[field]
            if (value is not None
                    and (not isinstance(value, str)
                         or FULL_COMMIT_RE.fullmatch(value) is None)):
                raise TicketCycleStateError(
                    "completed control-plane integration identity is invalid")
        for field in ("shadow_evidence", "integration_evidence",
                      "health_evidence"):
            value = control[field]
            if value is not None and (not isinstance(value, str)
                                      or not value or len(value) > 4096):
                raise TicketCycleStateError(
                    "completed control-plane evidence is invalid")
        validate_control_plane_relationships(
            control=control, completed_candidate=record["candidate"])
        normalized_control_history[cycle_id] = {
            "candidate": record["candidate"],
            "landing": record["landing"],
            "control_plane": dict(control),
        }
    normalized_finite = None
    if finite_watch is not None:
        if (not isinstance(finite_watch, dict)
                or set(finite_watch)
                != {"limit", "completed", "status", "topology"}):
            raise TicketCycleStateError(
                "finite-watch progress has invalid keys")
        limit = finite_watch.get("limit")
        finite_completed = finite_watch.get("completed")
        status = finite_watch.get("status")
        topology = finite_watch.get("topology")
        if (isinstance(limit, bool) or not isinstance(limit, int)
                or limit <= 0 or limit > MAX_CYCLE_COUNT
                or isinstance(finite_completed, bool)
                or not isinstance(finite_completed, int)
                or finite_completed < 0 or finite_completed > limit
                or status not in {"active", "complete"}
                or topology not in ARCHITECT_COMMIT_MODES
                or (status == "complete" and finite_completed != limit)
                or (status == "complete" and pending_cycle_returns != 0)):
            raise TicketCycleStateError(
                "finite-watch progress is invalid")
        normalized_finite = {
            "limit": limit, "completed": finite_completed,
            "status": status, "topology": topology}
    return {
        "schema": TICKET_CYCLE_STATE_SCHEMA,
        "generation": generation,
        "pending_cycle_returns": pending_cycle_returns,
        "finite_watch": normalized_finite,
        "architect_admissions": normalized_admissions,
        "active": normalized_active,
        "completed": normalized_completed,
        "control_plane_history": normalized_control_history,
    }


def read_ticket_cycle_state():
    """Read the bounded daemon state; a clean missing file starts empty."""
    try:
        raw = stable_regular_bytes(
            path=ticket_cycle_state_path(),
            maximum_bytes=MAX_TICKET_CYCLE_STATE_BYTES,
            label="ticket-cycle state", missing_ok=True)
    except (OSError, ValueError) as exc:
        raise TicketCycleStateError(str(exc)) from exc
    if raw is None:
        return empty_ticket_cycle_state()
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError,
            OverflowError, ValueError) as exc:
        raise TicketCycleStateError(
            "ticket-cycle state is invalid JSON") from exc
    return validate_ticket_cycle_state(payload=payload)


def write_ticket_cycle_state(state):
    """Publish strict cycle state with an atomic replacement and fsync."""
    normalized = validate_ticket_cycle_state(payload=state)
    os.makedirs(MAILBOX, exist_ok=True)
    payload = (json.dumps(normalized, sort_keys=True, separators=(",", ":"))
               + "\n").encode("utf-8")
    if len(payload) > MAX_TICKET_CYCLE_STATE_BYTES:
        raise TicketCycleStateError("ticket-cycle state exceeds its limit")
    handle, temporary = tempfile.mkstemp(prefix=".ticket-cycle-", dir=MAILBOX)
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, ticket_cycle_state_path())
        fsync_directory(directory=MAILBOX)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def acquire_ticket_cycle_lock():
    """Serialize state changes made by independent working-directory lanes."""
    os.makedirs(MAILBOX, exist_ok=True)
    path = os.path.join(MAILBOX, TICKET_CYCLE_LOCK_NAME)
    try:
        lock_file = open(path, "a+", encoding="utf-8")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return lock_file
    except OSError as exc:
        raise TicketCycleStateError(
            "cannot lock ticket-cycle state: " + str(exc)) from exc


def release_ticket_cycle_lock(lock_file):
    """Release one ticket-cycle state lock."""
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def record_pending_ticket_cycle_return(state):
    """Persist one return that a live watch has not counted yet.

    Finite ``--once`` calls have no cycle controller, so their completions
    must not become credit for a later watch. A live watch records the return
    in the same atomic state replacement as completion. If the process dies
    before its in-memory controller is updated, the next watch can replay
    exactly this durable count.
    """
    controller = _ACTIVE_WATCH_RENDEZVOUS
    if controller is None:
        return
    finite_limit = controller.ticket_cycle_limit_value()
    if finite_limit is not None:
        saved = state["finite_watch"]
        if (saved is None or saved["status"] != "active"
                or saved["limit"] != finite_limit
                or saved["topology"]
                != controller.ticket_cycle_topology_value()):
            raise TicketCycleStateError(
                "ticket completion does not match the active finite-watch "
                "topology")
    if state["pending_cycle_returns"] >= MAX_CYCLE_COUNT:
        raise TicketCycleStateError("pending ticket-cycle return count is full")
    state["pending_cycle_returns"] = state["pending_cycle_returns"] + 1


def prepare_finite_watch_progress(limit, topology):
    """Start or resume the durable progress record for ``--cycle N``."""
    if (isinstance(limit, bool) or not isinstance(limit, int)
            or limit <= 0 or limit > MAX_CYCLE_COUNT):
        raise TicketCycleStateError("finite watch limit is invalid")
    if topology not in ARCHITECT_COMMIT_MODES:
        raise TicketCycleStateError("finite watch topology is invalid")
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        saved = state["finite_watch"]
        if saved is None or saved["status"] == "complete":
            saved = {"limit": limit, "completed": 0, "status": "active",
                     "topology": topology}
        else:
            if saved["topology"] != topology:
                raise TicketCycleStateError(
                    "interrupted finite watch belongs to topology "
                    + saved["topology"] + ", not " + topology)
            if saved["completed"] + state["pending_cycle_returns"] > limit:
                raise TicketCycleStateError(
                    "interrupted finite watch already completed more than "
                    "the requested --cycle limit")
            saved = dict(saved, limit=limit)
        state["finite_watch"] = saved
        write_ticket_cycle_state(state=state)
        return saved["completed"]
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def clear_finite_watch_progress(topology):
    """Abandon an interrupted finite limit when this run is not finite."""
    if topology not in ARCHITECT_COMMIT_MODES:
        raise TicketCycleStateError("watch topology is invalid")
    if not os.path.exists(ticket_cycle_state_path()):
        return
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        saved = state["finite_watch"]
        if (saved is not None and saved["status"] == "active"
                and saved["topology"] != topology):
            raise TicketCycleStateError(
                "interrupted finite watch belongs to topology "
                + saved["topology"] + ", not " + topology)
        if saved is not None:
            state["finite_watch"] = None
            write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def finish_finite_watch_progress(limit, completed, topology):
    """Mark a proved finite run complete before its success is reported."""
    if topology not in ARCHITECT_COMMIT_MODES:
        raise TicketCycleStateError("finite watch topology is invalid")
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        saved = state["finite_watch"]
        if (saved is None or saved["status"] != "active"
                or saved["limit"] != limit
                or saved["completed"] != completed
                or saved["topology"] != topology
                or completed != limit
                or state["pending_cycle_returns"] != 0
                or any(record["mode"] == topology for record in
                       state["architect_admissions"].values())):
            raise TicketCycleStateError(
                "finite-watch progress does not prove a clean exit")
        state["finite_watch"] = dict(saved, status="complete")
        write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def deliver_pending_ticket_cycle_returns():
    """Count and acknowledge every durable return for the active watch.

    The state lock serializes concurrent ticket completions.
    The controller is updated before the acknowledgement is written. A crash
    in that narrow gap replays the return into the replacement process; it
    can never lose the return from both durable and in-memory state.
    """
    controller = _ACTIVE_WATCH_RENDEZVOUS
    if controller is None:
        return 0
    # A clean watch with no prior cycle state has nothing to deliver.  Avoid
    # creating a lock file merely to prove that absence; the dispatch lock
    # prevents another watcher from completing a cycle during startup.
    if not os.path.exists(ticket_cycle_state_path()):
        return 0
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        pending = state["pending_cycle_returns"]
        finite_limit = controller.ticket_cycle_limit_value()
        if finite_limit is not None:
            saved = state["finite_watch"]
            if (saved is None or saved["status"] != "active"
                    or saved["limit"] != finite_limit
                    or saved["topology"]
                    != controller.ticket_cycle_topology_value()
                    or saved["completed"]
                    != controller.completed_ticket_cycles()
                    or saved["completed"] + pending > finite_limit):
                raise TicketCycleStateError(
                    "durable finite-watch progress does not match the live "
                    "cycle controller")
            if pending:
                saved = dict(saved, completed=saved["completed"] + pending)
                state["finite_watch"] = saved
                state["pending_cycle_returns"] = 0
                # Durable progress is published before RAM is advanced. A
                # crash between these operations resumes from this value.
                write_ticket_cycle_state(state=state)
                for _ in range(pending):
                    _ticket_cycle_completed()
        else:
            if pending:
                for _ in range(pending):
                    _ticket_cycle_completed()
                state["pending_cycle_returns"] = 0
                write_ticket_cycle_state(state=state)
        return pending
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def cycle_ticket_anchor(cycle_id):
    """Return the backlog anchor carried by one validated cycle id."""
    return cycle_id.split("@", 1)[0]


def cycle_starting_commit(cycle_id):
    """Return the full starting commit carried by one validated cycle id."""
    return cycle_id.split("@", 1)[1]


def require_open_backlog_ticket(ticket_anchor):
    """Prove one cycle begins from exactly one indexed Open ticket."""
    lines, problem = verified_backlog_lines()
    if problem is not None:
        raise TicketCycleStateError(problem)
    indexed = []
    for line in lines:
        match = OPEN_BACKLOG_TICKET_RE.fullmatch(line)
        if match is not None and match.group(4) == ticket_anchor:
            indexed.append(line)
    details = [line for line in lines
               if line == '<a id="' + ticket_anchor + '"></a>']
    if len(indexed) != 1 or len(details) != 1:
        raise TicketCycleStateError(
            "ticket cycle must begin from exactly one indexed Open backlog "
            "ticket: " + ticket_anchor)


def active_cycle_records_for_topology(state, skip_redteam=False):
    """Return active records this watch can advance."""
    return [
        record for record in state["active"].values()
        if ticket_cycle_mode_is_enabled(
            mode=record["mode"], skip_redteam=skip_redteam)]


def architect_admissions_for_topology(state, skip_redteam=False):
    """Return public Architect requests already charged to this watch."""
    return [
        record for record in state["architect_admissions"].values()
        if ticket_cycle_mode_is_enabled(
            mode=record["mode"], skip_redteam=skip_redteam)]


def finite_cycle_capacity_used(state, skip_redteam=False):
    """Return every completed, admitted, or active charged ticket."""
    controller = _ACTIVE_WATCH_RENDEZVOUS
    if controller is None or controller.ticket_cycle_limit_value() is None:
        return None
    topology = canonical_ticket_cycle_topology(skip_redteam=skip_redteam)
    if controller.ticket_cycle_topology_value() != topology:
        raise TicketCycleStateError(
            "finite cycle capacity was requested for another topology")
    saved = state["finite_watch"]
    if (saved is None or saved["status"] != "active"
            or saved["topology"] != topology):
        raise TicketCycleStateError(
            "finite cycle capacity lacks matching durable progress")
    return (controller.completed_ticket_cycles()
            + state["pending_cycle_returns"]
            + len(active_cycle_records_for_topology(
                state=state, skip_redteam=skip_redteam))
            + len(architect_admissions_for_topology(
                state=state, skip_redteam=skip_redteam)))


def register_ticket_cycle_message(
        agent, message, skip_redteam=False, return_reservation=False,
        architect_admission=None, implementer_request_name=None,
        path_scope=None, ticket_class="ordinary"):
    """Register a ticket exchange or post-commit review before dispatch.

    Returns ``(cycle_id, accepted_commit)`` for a normal Red Team closure,
    ``(cycle_id, None)`` for an Architect/Implementer exchange, and
    ``(None, None)`` for cycle-free policy review or unrelated work. A new
    ticket reserves one positive ``--cycle`` slot before its mailbox file is
    claimed.
    """
    cycle_id = None
    accepted_commit = None
    requested_mode = None
    phase = None
    created = False
    class_problem = ticket_class_configuration_problem(
        ticket_class=ticket_class, skip_redteam=skip_redteam)
    if class_problem is not None:
        raise TicketCycleStateError(class_problem)
    if agent in {"fable", "opus"} and message.startswith(MAILBOX_FLOW_HEADER):
        cycle_id, requested_mode, _, problem = _ticket_flow_envelope(
            message=message)
        if problem is not None:
            raise TicketCycleStateError(problem)
        phase = "implementation"
    elif (agent == "sol" and sol_ticket_kind(message=message) == "closure"):
        if skip_redteam:
            raise TicketCycleStateError(
                "this watch does not dispatch Red Team closures")
        cycle_id, accepted_commit, _, problem = (
            _redteam_closure_envelope(message=message))
        if problem is not None:
            raise TicketCycleStateError(problem)
        phase = "awaiting-redteam"
    else:
        if architect_admission is not None:
            raise TicketCycleStateError(
                "Architect admission does not name an Implementer flow")
        return ((None, None, False) if return_reservation
                else (None, None))

    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        completed_commit = state["completed"].get(cycle_id)
        if completed_commit is not None:
            raise TicketCycleStateError(
                "ticket cycle was already completed at " + completed_commit)
        current = state["active"].get(cycle_id)
        if phase == "implementation":
            requested_route = "primary"
            expected_primary_mode = "two-role" if skip_redteam else "normal"
            if not ticket_cycle_mode_is_enabled(
                    mode=requested_mode, skip_redteam=skip_redteam):
                raise TicketCycleStateError(
                    "ticket exchange belongs to another watch role")
            if agent == "opus" and requested_mode != expected_primary_mode:
                raise TicketCycleStateError(
                    "the primary Implementer must use MAILBOX-MODE: "
                    + expected_primary_mode + " for this watch")
            if current is None:
                if agent == "fable":
                    raise TicketCycleStateError(
                        "the Architect route cannot invent a cycle before an "
                        "Implementer handoff")
                if (architect_admission is None
                        and implementer_request_name is not None):
                    request_match = PENDING_MESSAGE_RE.fullmatch(
                        implementer_request_name)
                    if (request_match is None
                            or request_match.group(1) != "opus"):
                        raise TicketCycleStateError(
                            "invalid Implementer request identity")
                    flow_name, flow_digest, admission_problem = (
                        _ticket_architect_admission(message=message))
                    if admission_problem is not None:
                        raise TicketCycleStateError(admission_problem)
                    if flow_name is not None:
                        admission_record = state[
                            "architect_admissions"].get(flow_name)
                        if (admission_record is None
                                or admission_record["sha256"]
                                != flow_digest
                                or admission_record["sequence"]
                                >= message_sequence(
                                    implementer_request_name)):
                            raise TicketCycleStateError(
                                "Implementer flow names no exact earlier "
                                "public Architect admission")
                        architect_admission = architect_admission_token(
                            request_name=flow_name, digest=flow_digest)
                admission = None
                if architect_admission is not None:
                    admission_name, admission_digest = (
                        split_architect_admission_token(
                            token=architect_admission))
                    flow_name, flow_digest, admission_problem = (
                        _ticket_architect_admission(message=message))
                    if admission_problem is not None:
                        raise TicketCycleStateError(admission_problem)
                    if (flow_name != admission_name
                            or flow_digest != admission_digest):
                        raise TicketCycleStateError(
                            "Implementer flow does not carry its exact "
                            "public Architect admission")
                    admission = state["architect_admissions"].get(
                        admission_name)
                    if admission is None:
                        raise TicketCycleStateError(
                            "Implementer flow lacks its exact public "
                            "Architect admission")
                    if admission["sha256"] != admission_digest:
                        raise TicketCycleStateError(
                            "Implementer flow admission digest changed")
                    if admission["mode"] != requested_mode:
                        raise TicketCycleStateError(
                            "Implementer flow changed its admitted watch "
                            "topology")
                if architect_notes_transition_pending():
                    raise TicketCycleLimitDeferred(
                        "a permanent-note admin turn or P landing is still "
                        "pending; no newer ticket may be admitted")
                require_open_backlog_ticket(
                    ticket_anchor=cycle_ticket_anchor(cycle_id))
                starting_commit = cycle_starting_commit(cycle_id)
                if not git_commit_exists(commit=starting_commit):
                    raise TicketCycleStateError(
                        "ticket cycle starting commit does not exist: "
                        + starting_commit)
                current_main = _exact_git_object(
                    arguments=["rev-parse", "--verify",
                               "refs/heads/main^{commit}"],
                    label="current main commit")
                if starting_commit != current_main:
                    raise TicketCycleLimitDeferred(
                        "ticket cycle base is not the exact current main "
                        "commit; wait for any earlier P/L landing, then "
                        "reissue the Architect handoff from that commit")
                if admission is None:
                    used = finite_cycle_capacity_used(
                        state=state, skip_redteam=skip_redteam)
                    controller = _ACTIVE_WATCH_RENDEZVOUS
                    if (used is not None
                            and used >= controller.ticket_cycle_limit_value()):
                        raise TicketCycleLimitDeferred(
                            "the finite watch has already reserved all "
                            + str(controller.ticket_cycle_limit_value())
                            + " ticket cycle(s)")
                state["active"][cycle_id] = {
                    "phase": "implementation", "commit": None,
                    "mode": requested_mode, "route": requested_route,
                    "ticket_class": ticket_class,
                    "path_scope": (sorted(path_scope)
                                   if path_scope is not None else None),
                    "control_plane": (
                        empty_control_plane_state()
                        if ticket_class == "protected-control-plane"
                        else None)}
                if admission is not None:
                    del state["architect_admissions"][admission_name]
                created = True
            elif current["phase"] != "implementation":
                raise TicketCycleStateError(
                    "ticket exchange arrived after the daemon recorded "
                    "landing L")
            elif architect_admission is not None:
                raise TicketCycleStateError(
                    "public Architect admission was already converted")
            elif (current["mode"] != requested_mode
                  or current["route"] != requested_route):
                raise TicketCycleStateError(
                    "ticket exchange changed its saved mode or Implementer "
                    "route")
            elif (agent == "opus"
                  and current.get("ticket_class", "ordinary")
                  != ticket_class):
                raise TicketCycleStateError(
                    "ticket exchange changed its frozen Ticket class")
            elif agent == "opus" and path_scope is not None:
                frozen = current.get("path_scope")
                proposed = sorted(path_scope)
                if frozen is not None and frozen != proposed:
                    raise TicketCycleStateError(
                        "Implementer handoff changed the frozen ticket path "
                        "scope")
                if frozen is None:
                    state["active"][cycle_id] = dict(
                        current, path_scope=proposed)
        else:
            if current is None:
                raise TicketCycleStateError(
                    "Red Team closure has no recorded daemon landing")
            if current["phase"] == "implementation":
                raise TicketCycleStateError(
                    "Red Team closure arrived before the daemon landing was "
                    "recorded")
            if (current["phase"] == "awaiting-redteam"
                    and current["commit"] != accepted_commit):
                raise TicketCycleStateError(
                    "ticket cycle names two different daemon landings")
            if (current is not None
                    and current["phase"] == "committed-awaiting-closure"
                    and current["commit"] != accepted_commit):
                raise TicketCycleStateError(
                    "Red Team closure does not name the recorded daemon "
                    "landing")
            if current["mode"] != "normal" or current["route"] != "primary":
                raise TicketCycleStateError(
                    "only a normal primary ticket receives Red Team closure")
            state["active"][cycle_id] = dict(
                current, phase="awaiting-redteam",
                commit=accepted_commit)
        write_ticket_cycle_state(state=state)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)
    result = (cycle_id, accepted_commit)
    return result + (created,) if return_reservation else result


def complete_ticket_cycle(cycle_id, accepted_commit):
    """Move one correlated Red Team return from active to completed state."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        prior = state["completed"].get(cycle_id)
        if prior is not None:
            if prior == accepted_commit:
                return False
            raise TicketCycleStateError(
                "ticket cycle was completed at another commit")
        current = state["active"].get(cycle_id)
        if (current is None or current["phase"] != "awaiting-redteam"
                or current["commit"] != accepted_commit):
            raise TicketCycleStateError(
                "Red Team return does not match an awaiting ticket cycle")
        del state["active"][cycle_id]
        state["completed"][cycle_id] = accepted_commit
        state["generation"] = state["generation"] + 1
        record_pending_ticket_cycle_return(state=state)
        write_ticket_cycle_state(state=state)
        return True
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def complete_protected_ticket_cycle(cycle_id, candidate_commit, landing):
    """Complete one two-key ticket after D0 records healthy L."""
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        prior = state["completed"].get(cycle_id)
        if prior is not None:
            if prior == landing:
                return False
            raise TicketCycleStateError(
                "protected ticket completed at another landing")
        active = state["active"].get(cycle_id)
        if (active is None
                or active.get("ticket_class") != "protected-control-plane"
                or active["phase"] != "committed-awaiting-closure"
                or active["commit"] != landing):
            raise TicketCycleStateError(
                "protected completion lacks its daemon landing")
        control = active["control_plane"]
        if (control["architect_candidate"] != candidate_commit
                or control["redteam_candidate"] != candidate_commit
                or control["redteam_result"] != "ACCEPT-CONTROL-PLANE"
                or control["shadow_status"] != "PASSED"
                or control["health_status"] != "HEALTHY"):
            raise TicketCycleStateError(
                "protected completion lacks both keys and healthy evidence")
        del state["active"][cycle_id]
        state["completed"][cycle_id] = landing
        state["control_plane_history"][cycle_id] = {
            "candidate": candidate_commit,
            "landing": landing,
            "control_plane": dict(control),
        }
        state["generation"] += 1
        record_pending_ticket_cycle_return(state=state)
        write_ticket_cycle_state(state=state)
        return True
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def record_architect_commit(cycle_id, accepted_commit, mode):
    """Record the daemon squash landing accepted by one Architect GO.

    Returns ``1`` when a two-role ticket completes at this landing record. A
    normal ticket returns ``0`` and waits for its correlated Red Team pass.
    """
    if (not isinstance(cycle_id, str)
            or CYCLE_ID_RE.fullmatch(cycle_id) is None
            or not isinstance(accepted_commit, str)
            or FULL_COMMIT_RE.fullmatch(accepted_commit) is None
            or mode not in ARCHITECT_COMMIT_MODES):
        raise TicketCycleStateError("invalid daemon landing record")
    lock_file = acquire_ticket_cycle_lock()
    completed_now = 0
    try:
        state = read_ticket_cycle_state()
        if cycle_id in state["completed"]:
            if state["completed"][cycle_id] == accepted_commit:
                return 0
            raise TicketCycleStateError(
                "Architect GO cycle was completed at another landing")
        current = state["active"].get(cycle_id)
        if (current is not None and current["phase"] != "implementation"
                and current["commit"] == accepted_commit
                and current["mode"] == mode):
            return 0
        if current is None or current["phase"] != "implementation":
            raise TicketCycleStateError(
                "daemon landing record has no active implementation cycle")
        if current["mode"] != mode:
            raise TicketCycleStateError(
                "Architect GO changed the ticket's saved mode")
        candidate_commit = require_architect_landing_locked(
            cycle_id=cycle_id, landing_commit=accepted_commit,
            ticket_state=state)
        # Git ancestry proves a new landing. An exact landing record already
        # represented by durable completed/active state is idempotent above
        # and must not depend forever on historical Git objects remaining
        # reachable.
        if not git_commit_descends_from(
                starting_commit=cycle_starting_commit(cycle_id),
                accepted_commit=accepted_commit):
            raise TicketCycleStateError(
                "daemon-recorded landing L is not a new descendant of the "
                "cycle base")
        if mode == "two-role":
            del state["active"][cycle_id]
            state["completed"][cycle_id] = accepted_commit
            state["generation"] = state["generation"] + 1
            record_pending_ticket_cycle_return(state=state)
            completed_now = 1
        elif mode == "normal":
            state["active"][cycle_id] = dict(
                current, phase="committed-awaiting-closure",
                commit=accepted_commit)
        write_ticket_cycle_state(state=state)
        # C and its private ref remain reachable until the GO itself reaches
        # done/. Startup recovery still needs C to re-prove an interrupted
        # exact landing or closure publication.
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)
    return completed_now


def active_ticket_cycle_count(skip_redteam=False, exclude_admission=None):
    """Count enabled work, optionally excluding one request's admission.

    Each topology counts only tickets it can advance. Valid work saved for a
    different topology remains active and untouched for a later watch.
    """
    lock_file = acquire_ticket_cycle_lock()
    try:
        state = read_ticket_cycle_state()
        active = active_cycle_records_for_topology(
            state=state, skip_redteam=skip_redteam)
        admissions = architect_admissions_for_topology(
            state=state, skip_redteam=skip_redteam)
        excluded = state["architect_admissions"].get(exclude_admission)
        if (excluded is not None and ticket_cycle_mode_is_enabled(
                mode=excluded["mode"], skip_redteam=skip_redteam)):
            admissions.remove(excluded)
        return len(active) + len(admissions)
    finally:
        release_ticket_cycle_lock(lock_file=lock_file)


def read_cycle_message(path):
    """Read one bounded mailbox message for cycle-state reconciliation."""
    raw = stable_regular_bytes(
        path=path, maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
        label="cycle message " + os.path.basename(path))
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TicketCycleStateError(
            "cycle message is not UTF-8: " + path) from exc


def any_matching_redteam_receipt(cycle_id, accepted_commit):
    """Return whether exactly one persisted receipt matches the review."""
    matches = []
    for path in glob.glob(os.path.join(MAILBOX, "**", "*-to-fable.md"),
                          recursive=True):
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            continue
        if not message.startswith(MAILBOX_RETURN_HEADER):
            continue
        returned_cycle, returned_commit, _, _, problem = (
            _redteam_review_receipt(message=message))
        if (problem is None and returned_cycle == cycle_id
                and returned_commit == accepted_commit):
            matches.append(path)
    if len(matches) > 1:
        raise TicketCycleStateError(
            "more than one Red Team receipt names " + cycle_id + " at "
            + accepted_commit)
    return bool(matches)


def redteam_review_completes_cycle(result):
    """Only NO CHANGE ends a cycle without an Architect decision."""
    return result == "NO CHANGE"


def current_reopen_ticket(cycle_id):
    """Read one mechanically checked ticket before Architect reasoning."""
    try:
        sealed = _validate_sealed_backlog(
            primary_worktree=AGENT_CWD["fable"])
        lines = sealed.decode("utf-8", errors="strict").splitlines()
        return _REOPEN_TRANSITION.inspect_backlog(
            lines=lines, anchor=cycle_ticket_anchor(cycle_id))
    except (UnicodeDecodeError, PrimaryWorktreeError,
            _REOPEN_TRANSITION.ReopenTransitionError) as exc:
        raise TicketCycleStateError(str(exc)) from exc


def architect_reopen_decision(cycle_id, before):
    """Verify the exact backlog transition and return GO or NO-GO."""
    sealed = _validate_sealed_backlog(
        primary_worktree=AGENT_CWD["fable"])
    try:
        lines = sealed.decode("utf-8", errors="strict").splitlines()
        after = _REOPEN_TRANSITION.inspect_backlog(
            lines=lines, anchor=cycle_ticket_anchor(cycle_id))
        return _REOPEN_TRANSITION.validate_after(
            before=before, after=after)
    except (UnicodeDecodeError,
            _REOPEN_TRANSITION.ReopenTransitionError) as exc:
        raise TicketCycleStateError(str(exc)) from exc


def _matching_journaled_notes_go(base_commit, notes_commit,
                                  receipt_sha256):
    """Return one exact B/P receipt whose bytes match an admin journal."""
    matches = []
    for directory in (MAILBOX, os.path.join(MAILBOX, "inflight"), DONE):
        for path in glob.glob(os.path.join(directory, "*-to-daemon.md")):
            try:
                raw = stable_regular_bytes(
                    path=path,
                    maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="journaled permanent-note GO")
                message = raw.decode("utf-8", errors="strict")
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            returned_base, returned_notes, problem = (
                _architect_notes_go_request(message=message))
            if (problem is None and returned_base == base_commit
                    and returned_notes == notes_commit
                    and hashlib.sha256(raw).hexdigest() == receipt_sha256):
                matches.append(path)
    if len(matches) != 1:
        raise TicketCycleStateError(
            "validated permanent-note admin journal needs exactly one "
            "unchanged B/P receipt; found " + str(len(matches)))
    return matches[0]


def _require_safe_noop_admin_recovery(base_commit):
    """Allow a proved no-change admin result after clean later landings."""
    primary = AGENT_CWD["fable"]
    primary_head = worktree_head(worktree=primary)
    current_main = _exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    if primary_head != current_main:
        raise TicketCycleStateError(
            "validated no-op admin needs Architect primary at current main")
    try:
        if _tracked_worktree_changes(worktree=primary):
            raise TicketCycleStateError(
                "validated no-op admin needs a clean Architect primary")
        _validate_current_protected_primary_state(primary_worktree=primary)
    except PrimaryWorktreeError as exc:
        raise TicketCycleStateError(str(exc)) from exc
    _require_ancestor_or_same(
        ancestor=base_commit, descendant=current_main,
        label="validated no-op admin base is not in current main history")


def reconcile_architect_notes_admin_journals():
    """Validate every admin journal and retire only proved done no-ops."""
    prefix = ".pending-notes-admin-"
    suffix = ".json"
    pattern = os.path.join(RELAY_DIR, prefix + "*" + suffix)
    retired = 0
    for journal_path in sorted(glob.glob(pattern)):
        filename = os.path.basename(journal_path)
        request_name = filename[len(prefix):-len(suffix)]
        request_match = PENDING_MESSAGE_RE.fullmatch(request_name)
        if request_match is None or request_match.group(1) != "fable":
            raise TicketCycleStateError(
                "malformed permanent-note admin journal name: "
                + journal_path)
        request_path = _architect_notes_admin_request_path(
            request_name=request_name)
        try:
            request_message = stable_regular_bytes(
                path=request_path,
                maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="saved permanent-note admin").decode(
                    "utf-8", errors="strict")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise TicketCycleStateError(
                "cannot verify saved permanent-note admin " + request_path
                + ": " + str(exc)) from exc
        if not is_architect_notes_admin_message(message=request_message):
            raise TicketCycleStateError(
                "saved permanent-note admin is malformed: " + request_path)
        journal = read_architect_notes_admin_journal(
            request_name=request_name, request_message=request_message)
        directory = os.path.dirname(request_path)
        if directory not in {os.path.join(MAILBOX, "inflight"), DONE}:
            raise TicketCycleStateError(
                "admin recovery journal is bound to an invalid request "
                "state: " + request_path)
        if journal["phase"] == "started":
            # The inflight reconciler prints the stronger warning that the
            # child may still be alive. Never infer a result from P/GO files.
            if directory != os.path.join(MAILBOX, "inflight"):
                raise TicketCycleStateError(
                    "archived permanent-note admin has only a pre-child "
                    "journal: " + request_path)
            continue
        if journal["phase"] == "validated-noop":
            if directory == os.path.join(MAILBOX, "inflight"):
                continue
            _require_safe_noop_admin_recovery(
                base_commit=journal["base"])
            remove_architect_notes_admin_journal(
                request_name=request_name)
            retired += 1
            print("retired archived validated no-op admin journal "
                  + request_name + ".")
            continue
        base_commit = journal["base"]
        notes_commit = journal["notes_commit"]
        _matching_journaled_notes_go(
            base_commit=base_commit, notes_commit=notes_commit,
            receipt_sha256=journal["receipt_sha256"])
        require_architect_notes_commit(
            base_commit=base_commit, notes_commit=notes_commit,
            allow_landed_replay=True)
        try:
            _validate_current_protected_primary_state(
                primary_worktree=AGENT_CWD["fable"])
        except PrimaryWorktreeError as exc:
            raise TicketCycleStateError(str(exc)) from exc
    return retired


def reconcile_inflight_architect_notes_admin():
    """Archive only post-child admin results proved by their durable journal."""
    recovered = 0
    paths = sorted(glob.glob(os.path.join(
        MAILBOX, "inflight", "*-to-fable.md")), key=message_sequence)
    for path in paths:
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError) as exc:
            try:
                is_raw_admin = regular_file_has_prefix(
                    path=path,
                    prefix=MAILBOX_ADMIN_HEADER.encode("ascii"))
            except (OSError, ValueError):
                is_raw_admin = False
            if is_raw_admin:
                raise TicketCycleStateError(
                    "cannot verify inflight permanent-note admin " + path
                    + ": " + str(exc)) from exc
            continue
        if not is_architect_notes_admin_message(message=message):
            if message.startswith(MAILBOX_ADMIN_HEADER):
                raise TicketCycleStateError(
                    "inflight permanent-note admin is malformed: " + path)
            continue
        name = os.path.basename(path)
        journal_path = architect_notes_admin_journal_path(
            request_name=name)
        if not os.path.isfile(journal_path):
            raise TicketCycleStateError(
                "inflight permanent-note admin has no recovery journal: "
                + path + "; inspect its dispatch log and requeue only after "
                  "proving that no child is still running")
        journal = read_architect_notes_admin_journal(
            request_name=name, request_message=message)
        phase = journal["phase"]
        base_commit = journal["base"]
        if phase == "started":
            raise TicketCycleStateError(
                "inflight permanent-note admin has only a pre-child "
                "journal: " + path + "; a child may still be alive or its "
                "result may be unvalidated. Inspect the dispatch log and "
                "process before any manual requeue")
        if phase == "validated-noop":
            _require_safe_noop_admin_recovery(base_commit=base_commit)
        else:
            notes_commit = journal["notes_commit"]
            _matching_journaled_notes_go(
                base_commit=base_commit, notes_commit=notes_commit,
                receipt_sha256=journal["receipt_sha256"])
            require_architect_notes_commit(
                base_commit=base_commit, notes_commit=notes_commit,
                allow_landed_replay=True)
        if not archive_consumed_message(dispatch_path=path):
            raise TicketCycleStateError(
                "validated inflight permanent-note admin could not archive")
        if phase == "validated-noop":
            remove_architect_notes_admin_journal(request_name=name)
        else:
            print("retained validated permanent-note admin journal until "
                  "its exact P receipt is consumed.")
        recovered += 1
        print("recovered validated permanent-note admin result " + name
              + " without rerunning the Architect.")
    return recovered


def reconcile_ticket_cycle_state():
    """Recover cycle state from durable pending and completed messages.

    Returns the number of cycles newly completed during recovery. Historical
    messages already represented in state are idempotent and return zero.
    """
    # Validate even when the mailbox has no messages. Corrupt daemon state is
    # never permission to claim a drain or positive cycle complete.
    read_ticket_cycle_state()
    reconcile_architect_notes_admin_journals()
    reconcile_inflight_architect_notes_admin()
    active_directories = [MAILBOX,
                          os.path.join(MAILBOX, "inflight"),
                          os.path.join(MAILBOX, "failed")]
    active_paths = []
    for directory in active_directories:
        active_paths.extend(glob.glob(os.path.join(directory, "*-to-*.md")))

    # First revalidate implementation identities already admitted into
    # durable state. Merely queued root/failed work must not be registered by
    # startup recovery because finite-cycle capacity is reserved only when a
    # watcher actually admits the ticket.
    registered_cycles = read_ticket_cycle_state()["active"]
    for path in sorted(active_paths):
        name = os.path.basename(path)
        match = PENDING_MESSAGE_RE.match(name)
        if match is None or match.group(1) == "daemon":
            continue
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            # Root corruption belongs to the ordinary dispatcher, which can
            # claim and park the exact inode with a useful reason. Inflight
            # corruption remains a lane blocker. Neither state can register
            # or consume a ticket during recovery.
            continue
        agent = match.group(1)
        is_flow = (agent in {"fable", "opus"}
                   and message.startswith(MAILBOX_FLOW_HEADER))
        cycle_id = None
        if is_flow:
            cycle_id, _, _, problem = _ticket_flow_envelope(message=message)
            if problem is not None:
                continue
        if cycle_id in registered_cycles:
            record = registered_cycles[cycle_id]
            register_ticket_cycle_message(
                agent=agent, message=message,
                skip_redteam=(record["mode"] == "two-role"))

    completed_now = 0
    inflight_daemon = glob.glob(
        os.path.join(MAILBOX, "inflight", "*-to-daemon.md"))
    for path in sorted(inflight_daemon, key=message_sequence):
        message = read_cycle_message(path=path)
        if message.startswith(
                MAILBOX_RETURN_HEADER + "architect-notes-go"):
            base_commit, notes_commit, problem = (
                _architect_notes_go_request(message=message))
            if problem is not None:
                if not park_failed_message(dispatch_path=path):
                    raise TicketCycleStateError(
                        "malformed inflight Architect notes GO could not be "
                        "parked: " + os.path.basename(path) + ": " + problem)
                continue
            consumed, _notes = finish_claimed_architect_notes_go(
                dispatch_path=path, base_commit=base_commit,
                notes_commit=notes_commit)
            if not consumed:
                continue
            # Permanent-note administration is cycle-free.
            continue
        cycle_id, candidate_commit, mode, problem = _architect_go_request(
            message=message)
        if problem is not None:
            if not park_failed_message(dispatch_path=path):
                raise TicketCycleStateError(
                    "malformed inflight Architect GO could not be parked: "
                    + os.path.basename(path) + ": " + problem)
            print("parked malformed Architect GO request "
                  + os.path.basename(path) + " in failed/: " + problem)
            continue
        try:
            consumed, completed, _landing = finish_claimed_architect_go(
                dispatch_path=path, cycle_id=cycle_id,
                candidate_commit=candidate_commit, mode=mode)
        except FatalArchitectLandingError:
            raise
        if not consumed:
            continue
        completed_now = completed_now + completed

    done_daemon = glob.glob(os.path.join(DONE, "*-to-daemon.md"))
    for path in sorted(done_daemon, key=message_sequence):
        message = read_cycle_message(path=path)
        if message.startswith(
                MAILBOX_RETURN_HEADER + "architect-notes-go"):
            base_commit, notes_commit, problem = (
                _architect_notes_go_request(message=message))
            if problem is not None:
                if not park_failed_message(dispatch_path=path):
                    raise TicketCycleStateError(
                        "malformed archived Architect notes GO could not be "
                        "parked: " + os.path.basename(path) + ": " + problem)
                continue
            try:
                receipt_raw = stable_regular_bytes(
                    path=path,
                    maximum_bytes=MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="archived permanent-note GO receipt")
                require_architect_notes_commit_object(
                    base_commit=base_commit, notes_commit=notes_commit)
                current_main = _exact_git_object(
                    arguments=["rev-parse", "--verify",
                               "refs/heads/main^{commit}"],
                    label="current main commit")
                _require_ancestor_or_same(
                    ancestor=notes_commit, descendant=current_main,
                    label="archived permanent-note P is not on main")
                main_lock = acquire_main_checkout_turn_lock()
                if main_lock is None:
                    raise TicketCycleStateError(
                        "cannot lock archived permanent-note recovery")
                try:
                    preflight_role_baseline_sync(target=current_main)
                    sync_all_clean_role_baselines(target=current_main)
                finally:
                    release_main_checkout_turn_lock(lock_file=main_lock)
                retire_validated_commit_admin_journal(
                    base_commit=base_commit, notes_commit=notes_commit,
                    receipt_sha256=hashlib.sha256(receipt_raw).hexdigest())
                if current_main == notes_commit:
                    debt_path = _push_debt_path(landing=notes_commit)
                    if os.path.isfile(debt_path):
                        push_exact_landing_or_record_debt(
                            landing=notes_commit)
            except TicketCycleStateError as exc:
                if not park_failed_message(dispatch_path=path):
                    raise TicketCycleStateError(
                        "rejected archived Architect notes GO could not be "
                        "parked: " + os.path.basename(path) + ": "
                        + str(exc)) from exc
            continue
        cycle_id, candidate_commit, mode, problem = _architect_go_request(
            message=message)
        if problem is not None:
            if not park_failed_message(dispatch_path=path):
                raise TicketCycleStateError(
                    "malformed archived Architect GO could not be parked: "
                    + os.path.basename(path) + ": " + problem)
            print("moved malformed historical Architect GO request "
                  + os.path.basename(path) + " from done/ to failed/: "
                  + problem)
            continue
        try:
            landing = recorded_landing_for_architect_go(
                cycle_id=cycle_id, mode=mode)
            if landing is None:
                raise TicketCycleStateError(
                    "archived Architect GO has no durable local landing")
            if mode == "normal":
                state = read_ticket_cycle_state()
                active = state["active"].get(cycle_id)
                if (active is not None
                        and active["phase"] in {
                            "committed-awaiting-closure",
                            "awaiting-redteam"}):
                    publish_redteam_closure_request(
                        cycle_id=cycle_id, landing=landing)
            retire_cycle_landing_ref(
                cycle_id=cycle_id, landing=landing)
            retire_cycle_candidate(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                landing_commit=landing, mode=mode)
            current_main = _exact_git_object(
                arguments=["rev-parse", "--verify",
                           "refs/heads/main^{commit}"],
                label="current main commit")
            _require_ancestor_or_same(
                ancestor=landing, descendant=current_main,
                label="archived ordinary landing is not on main")
            if current_main == landing:
                main_lock = acquire_main_checkout_turn_lock()
                if main_lock is None:
                    raise TicketCycleStateError(
                        "cannot lock archived role-baseline recovery")
                try:
                    sync_all_clean_role_baselines(target=landing)
                finally:
                    release_main_checkout_turn_lock(lock_file=main_lock)
                debt_path = _push_debt_path(landing=landing)
                if os.path.isfile(debt_path):
                    push_exact_landing_or_record_debt(landing=landing)
        except TicketCycleStateError as exc:
            if not park_failed_message(dispatch_path=path):
                raise TicketCycleStateError(
                    "rejected archived Architect GO could not be parked: "
                    + os.path.basename(path) + ": " + str(exc)) from exc
            print("moved rejected historical Architect GO request "
                  + os.path.basename(path) + " from done/ to failed/: "
                  + str(exc))

    # Register still-waiting review requests after Architect GO recovery has
    # restored their recorded landing phase.
    for path in sorted(active_paths):
        name = os.path.basename(path)
        match = PENDING_MESSAGE_RE.match(name)
        if match is None or match.group(1) != "sol":
            continue
        try:
            message = read_cycle_message(path=path)
        except (OSError, ValueError, TicketCycleStateError):
            continue
        if sol_ticket_kind(message=message) == "closure":
            register_ticket_cycle_message(agent="sol", message=message)

    # A crash can occur after the request reached done/ and before its state
    # replacement. The Red Team return plus archived request is enough to
    # finish that exact transition once, never to infer a missing review from
    # rc alone.
    review_paths = glob.glob(os.path.join(DONE, "*-to-sol.md"))
    review_paths.extend(glob.glob(
        os.path.join(MAILBOX, "inflight", "*-to-sol.md")))
    for path in sorted(review_paths, key=message_sequence):
        message = read_cycle_message(path=path)
        if (sol_ticket_kind(message=message) != "closure"
                or redteam_closure_problem(message=message) is not None):
            continue
        cycle_id = redteam_closure_ticket(message=message)
        commit = redteam_closure_commit(message=message)
        state = read_ticket_cycle_state()
        if state["completed"].get(cycle_id) == commit:
            if os.path.dirname(path) != DONE:
                if not archive_consumed_message(dispatch_path=path):
                    raise TicketCycleStateError(
                        "completed Red Team request could not be archived: "
                        + os.path.basename(path))
            continue
        _receipt_path, review_result, problem = matching_new_redteam_receipt(
            cycle_id=cycle_id, accepted_commit=commit, before_inodes=set())
        if problem is not None:
            raise TicketCycleStateError(problem)
        register_ticket_cycle_message(agent="sol", message=message)
        if redteam_review_completes_cycle(review_result):
            if complete_ticket_cycle(cycle_id=cycle_id,
                                     accepted_commit=commit):
                completed_now = completed_now + 1
        if os.path.dirname(path) != DONE:
            if not archive_consumed_message(dispatch_path=path):
                raise TicketCycleStateError(
                    "recovered Red Team request could not be archived: "
                    + os.path.basename(path))
    return completed_now


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


def send(agent, text, dry_run, ticket_kind=None, severity=None, scope=None):
    """Save one internal mailbox message or one user request for Architect.

    Arguments:
      agent   = recipient name "fable", "opus", or "sol" used inside this
                program. The public command line maps its sole ``architect``
                target to ``fable``. Role-to-role callers use this function
                or save the next numbered mailbox file.
      text    = exact message text; internal role messages point to the source
                note under ``ai/notes/``.
      dry_run = True to print the file path without writing the message.
      ticket_kind = ``closure``, ``discovery``, or ``policy`` for internal
                    Sol work. Policy is the cycle-free review of a protected
                    rule. The exact internal Sol ping alone uses ``transport``.
      severity = the Architect-approved minimum ``high``, ``medium``, or
                 ``low`` value for an internal Sol discovery. Omission uses
                 the inherited run value or medium. Other ticket kinds and
                 internal recipients accept no severity here.
      scope = the exact ``bounded`` or ``widespread`` scope for an internal
              Sol discovery. Omission is bounded. Other ticket kinds and
              recipients accept no scope here.

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
    effective_scope = (
        (DEFAULT_DISCOVERY_SCOPE if scope is None else scope)
        if ticket_kind == "discovery" else scope)
    if (ticket_kind == "discovery"
            and effective_scope not in DISCOVERY_SCOPES):
        print("refused --send " + agent + ": discovery scope must be "
              "bounded or widespread.")
        return False

    def refusal_now():
        """Return a current Sol-send refusal without changing disk."""
        if agent != "sol":
            if severity is not None:
                return "--severity is valid only with --send sol discovery"
            if scope is not None:
                return "scope is valid only with --send sol discovery"
            return None
        if skip_redteam_policy_active():
            return ("an active two-role watch has the Sol route disabled; "
                    "wait for it to end or restart without --skip-redteam")
        transport_valid = (
            ticket_kind == "transport"
            and text == transport_ping_text(agent="sol"))
        counts = backlog_severity_counts()
        reason = sol_ticket_refusal(
            ticket_kind=ticket_kind,
            admission_count=(counts["critical"] + counts["high"]
                             + counts["medium"]),
            fix_only=(fix_only_environment_active()
                      or fix_only_watch_is_active()),
            transport_valid=transport_valid,
            discovery_severity=effective_severity,
            discovery_scope=effective_scope,
            unclassified_count=counts["unclassified"],
            ledger_problem=counts["problem"])
        if reason is not None:
            return reason
        return None

    reason = refusal_now()
    if reason is not None:
        print("refused --send " + agent + ": " + reason + ".")
        return False

    payload = text
    if agent == "sol":
        if ticket_kind in SOL_DISPATCH_TICKET_KINDS:
            payload = sol_ticket_payload(
                ticket_kind=ticket_kind, text=text,
                discovery_severity=effective_severity,
                discovery_scope=effective_scope)
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
    if not live_action_topology_is_current(agent, "--send " + agent):
        return False
    os.makedirs(MAILBOX, exist_ok=True)
    lock_file = acquire_mailbox_sequence_lock()
    if lock_file is None:
        return False
    try:
        if not live_action_topology_is_current(agent, "--send " + agent):
            return False
        # Recheck persisted modes and the current classified backlog while
        # publication is serialized. Queue publication does not itself change
        # either severity count.
        reason = refusal_now()
        if reason is not None:
            print("refused --send " + agent + ": " + reason + ".")
            return False
        for _ in range(20):
            path = publish_message_locked(
                agent=agent, payload=payload, attempts=1)
            if path is not None:
                print("queued " + path)
                try:
                    warn_if_mailbox_unwatched()
                    if skip_redteam_policy_active():
                        report_demand(
                            backlog=pending_messages(), skip_redteam=True)
                    else:
                        report_demand(backlog=pending_messages())
                except Exception as exc:
                    print("  warning: message is queued, but its status "
                          "report failed: " + str(exc))
                return True
    finally:
        release_mailbox_sequence_lock(lock_file=lock_file)
    print("could not claim a sequence number after 20 tries; "
          "is something flooding the mailbox?")
    return False


def recover_failed_maintenance_admission():
    """Requeue one failed fix-only request on restart."""
    sequence_lock = acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise TicketCycleStateError("cannot lock recovery")
    state_lock = None
    try:
        state_lock = acquire_ticket_cycle_lock()
        state = read_ticket_cycle_state()
        match = None
        failed = os.path.join(MAILBOX, "failed")
        for name, record in state["architect_admissions"].items():
            path = os.path.join(failed, name)
            if not os.path.lexists(path):
                continue
            message = read_cycle_message(path=path)
            if hashlib.sha256(message.encode("utf-8")).hexdigest() \
                    != record["sha256"]:
                raise TicketCycleStateError("failed request changed")
            if message == ARCHITECT_FIX_ONLY_REQUEST:
                if match is not None:
                    raise TicketCycleStateError(
                        "multiple maintenance requests failed")
                match = path
        if match is None:
            return
        for duplicate in pending_messages():
            try:
                duplicate_message = read_cycle_message(path=duplicate)
            except (OSError, ValueError, TicketCycleStateError):
                continue
            if duplicate_message == ARCHITECT_FIX_ONLY_REQUEST:
                _parked, moved = verified_state_move(
                    dispatch_path=duplicate, directory=failed)
                if not moved:
                    raise TicketCycleStateError(
                        "could not preserve duplicate")
                print("parked duplicate " + os.path.basename(duplicate)
                      + " in failed/")
        recovered, moved = verified_state_move(
            dispatch_path=match, directory=MAILBOX)
        if not moved:
            raise TicketCycleStateError("could not requeue failed request")
        print("requeued " + recovered)
        return recovered
    finally:
        if state_lock is not None:
            release_ticket_cycle_lock(lock_file=state_lock)
        release_mailbox_sequence_lock(lock_file=sequence_lock)


def send_architect_notes_admin(text, dry_run=False):
    """Publish one narrow Architect-only permanent-note self-route."""
    try:
        contract = validate_role_contract_bindings()
    except (OSError, RuntimeError, ValueError) as exc:
        print("refused permanent-note admin request: role contract error: "
              + str(exc) + ".")
        return False
    if not contract["roles"]["architect"]["may_edit_protected_policy"]:
        print("refused permanent-note admin request: protected role contract "
              "does not grant Architect policy administration.")
        return False
    if os.environ.get(MAILBOX_ROLE_ENVIRONMENT) != "architect":
        print("refused permanent-note admin request: MAILBOX_ROLE must be "
              "architect.")
        return False
    primary = os.environ.get("MAILBOX_PRIMARY_WORKTREE")
    shared_notes = os.environ.get("MAILBOX_SHARED_NOTES")
    if (primary is None or shared_notes is None
            or os.path.realpath(primary) != os.path.realpath(WORKTREE)
            or os.path.realpath(shared_notes)
            != os.path.realpath(os.path.join(AI_ROOT, "notes"))):
        print("refused permanent-note admin request: this process is not "
              "bound to the saved Architect primary and shared notes.")
        return False
    try:
        payload = architect_notes_admin_payload(text=text)
    except ValueError as exc:
        print("refused permanent-note admin request: " + str(exc) + ".")
        return False
    if dry_run:
        print("[dry-run] would queue " + os.path.join(
            MAILBOX, next_seq() + "-to-fable.md"))
        return True
    os.makedirs(MAILBOX, exist_ok=True)
    lock_file = acquire_mailbox_sequence_lock()
    if lock_file is None:
        return False
    try:
        if architect_notes_transition_pending():
            print("refused permanent-note admin request: another note admin "
                  "turn or P landing is already pending.")
            return False
        path = publish_message_locked(agent="fable", payload=payload)
        if path is None:
            print("refused permanent-note admin request: no unique mailbox "
                  "sequence could be published.")
            return False
        print("queued " + path)
        return True
    finally:
        release_mailbox_sequence_lock(lock_file=lock_file)


def _is_ai_branch(branch):
    """Return whether a local branch belongs to an AI-only namespace."""
    return (type(branch) is str
            and branch.startswith(AI_BRANCH_PREFIXES))


def _lock_cleanup_transport(records):
    """Hold every existing mailbox lock until destructive cleanup ends."""
    locks = []
    try:
        for record in sorted(records, key=lambda item: item["path"]):
            for notes in (os.path.join("ai", "notes"), "notes"):
                mailbox = os.path.join(record["path"], notes, "mailbox")
                if not os.path.lexists(mailbox):
                    continue
                identity = _plain_directory(
                    path=mailbox, label="cleanup mailbox")
                for name in (".dispatch.lock", ".sequence.lock"):
                    locks.append(_open_legacy_transport_lock(
                        path=os.path.join(mailbox, name), nonblocking=True))
                _require_directory_identity(
                    path=mailbox, identity=identity,
                    label="cleanup mailbox")
        return locks
    except BaseException:
        for lock_file in reversed(locks):
            _release_legacy_transport_lock(lock_file=lock_file)
        raise


def clean_all_ai_worktrees(repository_root, current_worktree):
    """Discard local AI worktrees and branches after an explicit request."""
    repository = os.path.abspath(repository_root)
    if os.path.realpath(current_worktree) != os.path.realpath(repository):
        raise PrimaryWorktreeError(
            "run --clean-all from the user's main repository folder")
    lock_file = _open_primary_lock(repository)
    transport_locks = []
    try:
        managed_root = _managed_primary_root(repository, create=True)
        records = registered_worktrees(repository)
        root_record = _record_at_path(records, repository)
        if root_record is None or _is_ai_branch(root_record.get("branch")):
            raise PrimaryWorktreeError(
                "the user's repository folder must use a non-AI branch")
        for record in records:
            reasons = coordination_transport_evidence(record["path"])
            if any("live " in reason for reason in reasons):
                raise PrimaryWorktreeError(
                    "stop the live mailbox watcher or sender before "
                    "--clean-all: " + record["path"])
        transport_locks = _lock_cleanup_transport(records=records)
        print("WARNING: --clean-all permanently discards dirty files and "
              "unmerged commits in local AI worktrees.", flush=True)
        _run_git(repository, ["worktree", "prune"])
        records = registered_worktrees(repository)
        preserved = set()
        for record in sorted(records, key=lambda item: item["path"]):
            path = os.path.abspath(record["path"])
            if _path_key(path) == _path_key(repository):
                continue
            managed_child = os.path.dirname(path) == managed_root
            ai_branch = _is_ai_branch(record.get("branch"))
            if not ai_branch and "branch" in record:
                if managed_child:
                    preserved.add(_path_key(path))
                continue
            if not ai_branch and not managed_child:
                continue
            branch = record.get("branch", "detached audit")
            print("discarding AI worktree " + path + " (" + branch + ")",
                  flush=True)
            _run_git(repository, ["worktree", "remove", "--force",
                                  "--force", path])
        _run_git(repository, ["worktree", "prune"])
        with os.scandir(managed_root) as entries:
            stale_paths = sorted(entry.path for entry in entries)
        for path in stale_paths:
            if (os.path.basename(path) == PRIMARY_LOCK_NAME
                    or _path_key(path) in preserved):
                continue
            print("discarding stale AI path " + path, flush=True)
            info = os.lstat(path)
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        output = _run_git(repository, [
            "for-each-ref", "--format=%(refname)", "refs/heads/"])
        branches = [ref for ref in output.stdout.decode("utf-8").splitlines()
                    if _is_ai_branch(ref)]
        for ref in sorted(branches):
            print("deleting local AI branch "
                  + ref[len("refs/heads/"):], flush=True)
            _run_git(repository, ["update-ref", "-d", ref])
            if _branch_exists(repository, ref):
                raise PrimaryWorktreeError(
                    "could not delete local AI branch " + ref)
        print("clean-all finished; main and non-AI branches were not "
              "changed.", flush=True)
    finally:
        for transport_lock in reversed(transport_locks):
            _release_legacy_transport_lock(lock_file=transport_lock)
        _release_primary_lock(lock_file=lock_file)


def main():
    # both are rebound below from the parsed command line; Python wants
    # the global declaration before the first mention of either name.
    global AGENT_COMMANDS
    global DISPATCH_TIMEOUT_MINUTES
    global CLAUDE_CONTEXT_BUDGET
    global MAX_CHARACTERS
    global REVIEW_EFFORT
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
    parser.add_argument(
        CLEANUP_ACTION, action="store_true",
        help="permanently discard every local AI worktree and local "
             "claude/*, codex/*, or legacy worktree-agent-* branch; "
             "dirty files and unmerged commits in those worktrees are lost")
    parser.add_argument(
        "--restart-implementer", action="store_true",
        help="after an interrupted Implementer turn, discard its partial "
             "work and requeue the exact Architect handoff")
    parser.add_argument(
        "--restart-redteam", action="store_true",
        help="after an interrupted Red Team turn, discard its partial work "
             "and requeue the exact Architect-to-Red-Team handoff")
    parser.add_argument("--watch", action="store_true",
                        help="check the mailbox every 20 seconds and start "
                             "waiting requests")
    parser.add_argument("--cycle", metavar="count",
                        type=nonnegative_cycle_count, default=None,
                        help="with --watch, stop after this many completed "
                             "ticket cycles; one cycle is always one ticket; "
                             "with Red Team it ends when the matching review "
                             "returns for daemon-recorded local landing L, "
                             "and without Red Team it ends when the daemon "
                             "records local landing L; 0 "
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
                             "option; with --ping, check the Architect and "
                             "Implementer providers but not Sol")
    parser.add_argument("--fix-only", metavar="value", type=truthy_fix_only,
                        default=None,
                        help="with --send architect, save a backlog-repair "
                             "request; with --watch, run existing bug fixes "
                             "at the watcher's severity; "
                             "the value accepts 1, true, or yes in any "
                             "capitalization")
    parser.add_argument("--send", metavar="{architect}",
                        choices=["architect"],
                        help="save the user's ticket request for the "
                             "Architect and exit")
    parser.add_argument(
        "--ping", action="store_true",
        help="make one small live request to every provider selected for "
             "this run and exit; add --skip-redteam to omit Sol")
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
                        help="model name used for the Implementer; select "
                             "its service with --implementer-provider; "
                             "mailbox filenames still contain opus "
                             "(default: "
                             + DEFAULT_IMPLEMENTER_MODEL + ")")
    parser.add_argument(
        "--implementer-provider", choices=IMPLEMENTER_PROVIDERS,
        default=DEFAULT_IMPLEMENTER_PROVIDER,
        help="service used for the Implementer: claude or ollama; the "
             "Architect remains on Claude (default: claude)")
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
                        help="codex CLI reasoning effort for Sol as Red Team "
                             "(default: "
                             + DEFAULT_SOL_EFFORT + ")")
    parser.add_argument(
        "--review-effort", default=DEFAULT_REVIEW_EFFORT,
        choices=_REVIEW_DISPATCH.REVIEW_EFFORTS,
        help="reasoning effort for routine Architect rechecks and bounded "
             "Red Team ticket reviews; the first Architect plan, every "
             "Implementer turn, and Red Team discovery keep their role "
             "effort (default: " + DEFAULT_REVIEW_EFFORT + ")")
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
                        help="inside one Architect or Implementer turn, ask "
                             "the coding runtime to replace older context "
                             "with a shorter summary at this many tokens "
                             "(default: "
                             + str(DEFAULT_CLAUDE_CONTEXT_BUDGET) + ")")
    parser.add_argument("--sol-context", metavar="TOKENS",
                        type=positive_int, default=DEFAULT_SOL_CONTEXT_BUDGET,
                        help="inside one Red Team turn, ask Codex to replace "
                             "older conversation text with a shorter summary "
                             "at this many tokens (default: "
                             + str(DEFAULT_SOL_CONTEXT_BUDGET) + ")")
    args = parser.parse_args()
    maintenance_send = (
        args.send == "architect" and args.fix_only is True)

    if args.fix_only is not None:
        conflicting_action = (
            not (args.watch or maintenance_send) or args.once or args.ping
            or (args.watch and args.dry_run))
        if conflicting_action:
            print("--fix-only needs --watch or --send architect")
            return 1
    if args.cycle is not None and not args.watch:
        print("--cycle is valid only with --watch")
        return 1
    if args.max_characters is not None:
        conflicting_action = (
            not (args.watch or args.once)
            or args.send is not None or args.ping)
        if conflicting_action:
            print("--max is valid only with --watch or --once")
            return 1
    if args.skip_redteam:
        conflicting_action = (
            not (args.watch or args.ping) or args.once
            or args.send is not None)
        if conflicting_action:
            print("--skip-redteam is valid only with --watch or --ping")
            return 1
    if args.severity is not None:
        severity_run = args.watch or args.once
        severity_send = args.send == "architect" and not maintenance_send
        if not (severity_run or severity_send):
            print("--severity is valid only with --watch, --once, or "
                  "--send architect")
            return 1
    if args.send is not None and bool(args.unit) == maintenance_send:
        print("ordinary --send needs --unit; fix-only --send forbids it")
        return 1
    primary_actions = sum((
        bool(args.once),
        bool(args.watch),
        bool(args.clean_all),
        bool(args.restart_implementer),
        bool(args.restart_redteam),
        args.send is not None,
        bool(args.ping),
    ))
    if primary_actions > 1:
        print("choose only one primary action: --once, --watch, --clean-all, "
              "--restart-implementer, --restart-redteam, --send, or --ping")
        return 1
    if args.watch and args.dry_run:
        print("--dry-run is finite and cannot be combined with --watch")
        return 1
    if args.clean_all and args.dry_run:
        print("--clean-all is already an explicit destructive action and "
              "cannot be combined with --dry-run")
        return 1
    if ((args.restart_implementer or args.restart_redteam)
            and args.dry_run):
        print("restart commands explicitly discard one role's partial work "
              "and cannot be combined with --dry-run")
        return 1

    # Cleanup must run before primary selection: ambiguous old mailbox stores
    # are one reason a user needs this explicit destructive reset.
    if args.clean_all:
        try:
            clean_all_ai_worktrees(
                repository_root=REPO_ROOT, current_worktree=WORKTREE)
        except (OSError, PrimaryWorktreeError) as exc:
            print("clean-all error: " + str(exc))
            return 1
        return 0

    if args.ping:
        return 0 if check_provider_connectivity(
            architect_model=args.architect_model,
            implementer_provider=args.implementer_provider,
            implementer_model=args.implementer_model,
            include_sol=not args.skip_redteam,
            dry_run=args.dry_run) else 1

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
        try:
            validate_role_contract_bindings()
        except (OSError, RuntimeError, ValueError) as exc:
            print("role contract error: " + str(exc))
            return 1

    if args.restart_implementer or args.restart_redteam:
        dispatch_lock = acquire_dispatch_lock(mode="once")
        if dispatch_lock is None:
            return 1
        try:
            try:
                if args.restart_implementer:
                    restart_implementer_from_architect_handoff()
                else:
                    restart_redteam_from_architect_handoff()
            except (OSError, ValueError, PrimaryWorktreeError,
                    TicketCycleStateError) as exc:
                print("role restart refused: " + str(exc))
                return 1
        finally:
            release_dispatch_lock(lock_file=dispatch_lock)
        return 0

    fix_only = args.fix_only is True
    skip_redteam = args.skip_redteam
    if fix_only:
        _backlog_lines, backlog_problem = verified_backlog_lines()
        if backlog_problem is not None:
            print("fix-only cannot start: " + backlog_problem)
            return 1
    watch_topology = canonical_ticket_cycle_topology(
        skip_redteam=skip_redteam)
    MAX_CHARACTERS = (DEFAULT_MAX_CHARACTERS
                      if args.max_characters is None
                      else args.max_characters)
    DISCOVERY_SEVERITY = selected_discovery_severity

    DISPATCH_TIMEOUT_MINUTES = args.dispatch_timeout
    CLAUDE_CONTEXT_BUDGET = args.claude_context
    REVIEW_EFFORT = args.review_effort

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
        implementer_provider=args.implementer_provider,
        sol_worktree=AGENT_CWD["sol"],
        shared_notes=(ACTIVE_TOPOLOGY["shared_notes"]
                      if ACTIVE_TOPOLOGY is not None
                      else os.path.join(AGENT_CWD["fable"], "ai", "notes")))
    if args.watch:
        print("role providers: architect=claude implementer="
              + args.implementer_provider + " red-team=codex")
        print("role models: architect=" + args.architect_model
              + " implementer=" + args.implementer_model
              + " (internal mailbox names: fable/opus)")
        if skip_redteam:
            implementer_effort = (args.opus_effort
                                  if args.implementer_provider == "claude"
                                  else "provider default (ollama)")
            print("effort levels: architect/fable=" + args.fable_effort
                  + " implementer/opus=" + implementer_effort
                  + " sol=disabled routine-review=" + args.review_effort)
            print("context budgets: architect/implementer="
                  + str(args.claude_context)
                  + " sol=disabled (a Claude turn compacts at its budget)")
            print("two-role watch: Red Team and the entire Sol route are "
                  "disabled; existing to-sol messages stay queued and "
                  "untouched")
        else:
            implementer_effort = (args.opus_effort
                                  if args.implementer_provider == "claude"
                                  else "provider default (ollama)")
            print("effort levels: architect/fable=" + args.fable_effort
                  + " implementer/opus=" + implementer_effort
                  + " sol=" + args.sol_effort
                  + " routine-review=" + args.review_effort)
            print("context budgets: architect/implementer="
                  + str(args.claude_context)
                  + " sol=" + str(args.sol_context)
                  + " tokens (a turn compacts at its budget)")
        if args.cycle == 0:
            if skip_redteam:
                print("cycle 0: wait until no Architect or Implementer "
                      "message is waiting or running and "
                      "ai/notes/backlog.md has no '- OPEN' item, then exit; "
                      "this watch ignores Red Team messages")
            else:
                print("cycle 0: wait until no role message is waiting or "
                      "running and ai/notes/backlog.md has no '- OPEN' item, "
                      "then exit")
        elif args.cycle is not None:
            print("cycle limit: stop after " + str(args.cycle)
                  + " completed tickets; one ticket is one cycle; a normal "
                  "cycle ends when its correlated Red Team return names the "
                  "daemon-recorded local landing L, while a ticket without "
                  "Red Team ends when the daemon records local landing L; "
                  "finish every role job already starting or running before "
                  "exit")

    if args.send:
        request = architect_user_request_payload(
            text=(ARCHITECT_FIX_ONLY_REQUEST if maintenance_send
                  else args.unit),
            discovery_severity=selected_discovery_severity)
        queued = send(
            agent="fable",
            text=request,
            dry_run=args.dry_run)
        return 0 if queued else 1

    if args.dry_run:
        failed_debt = architect_notes_failed_debt_error()
        if failed_debt is not None:
            print(failed_debt)
            return 1
        outcome = process_backlog(dry_run=args.dry_run)
        if outcome == ROLE_CONTRACT_RESTART_REQUIRED:
            return report_role_contract_restart()
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
            try:
                recover_before_dispatch(
                    fix_only=fix_only, skip_redteam=skip_redteam)
                failed_debt = architect_notes_failed_debt_error()
                if failed_debt is not None:
                    print(failed_debt)
                    return 1
                outcome = process_backlog(dry_run=False)
            except RoleTokenExhaustionError as exc:
                report_role_token_exhaustion(error=exc)
                return 1
            except ImplementerAuthorityViolationError:
                print("watcher stopped before candidate admission or landing.")
                return 1
            except FatalArchitectLandingError as exc:
                print("Architect landing needs user action: " + str(exc))
                return 1
            except (OSError, ValueError, TicketCycleStateError) as exc:
                print("ticket-cycle recovery failed: " + str(exc)
                      + "; no mailbox work was dispatched.")
                return 1
            if outcome == ROLE_CONTRACT_RESTART_REQUIRED:
                return report_role_contract_restart()
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
            source_path=source_path, source_stamp=source_stamp,
            ticket_cycle_limit=(args.cycle
                                if args.cycle is not None and args.cycle > 0
                                else None),
            ticket_cycle_topology=(
                watch_topology
                if args.cycle is not None and args.cycle > 0 else None))
        _ACTIVE_WATCH_RENDEZVOUS = rendezvous
        first_pass = True
        completed_cycles = 0
        cycle_completion_barrier = None
        try:
            try:
                if args.cycle is not None and args.cycle > 0:
                    restored_progress = prepare_finite_watch_progress(
                        limit=args.cycle, topology=watch_topology)
                    rendezvous.restore_completed_ticket_cycles(
                        count=restored_progress)
                else:
                    clear_finite_watch_progress(topology=watch_topology)
            except (OSError, ValueError, TicketCycleStateError) as exc:
                print("finite cycle recovery failed: " + str(exc)
                      + "; watcher did not start dispatching work.")
                return 1
            try:
                recover_before_dispatch(
                    fix_only=fix_only, skip_redteam=skip_redteam)
                failed_debt = architect_notes_failed_debt_error()
                if failed_debt is not None:
                    print(failed_debt)
                    return 1
            except (OSError, ValueError, TicketCycleStateError) as exc:
                print("ticket-cycle recovery failed: " + str(exc)
                      + "; watcher did not start dispatching work.")
                return 1
            recovered_cycles = deliver_pending_ticket_cycle_returns()
            if recovered_cycles:
                print("recovered " + str(recovered_cycles)
                      + " completed ticket cycle(s) from durable mailbox "
                      "receipts.")
            if args.cycle is not None and args.cycle > 0:
                active_at_start = active_ticket_cycle_count(
                    skip_redteam=skip_redteam)
                used_at_start = (rendezvous.completed_ticket_cycles()
                                 + active_at_start)
                if used_at_start > args.cycle:
                    print("ticket-cycle recovery found "
                          + str(used_at_start) + " completed or active "
                          "tickets, beyond --cycle " + str(args.cycle)
                          + "; watcher did not start new work. Resume with a "
                          "limit of at least " + str(used_at_start) + ".")
                    return 1
            while True:
                contract_exit = role_contract_exit_status()
                if contract_exit is not None:
                    return contract_exit
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
                try:
                    backlog_outcome = process_backlog(
                        dry_run=False, fix_only=fix_only,
                        skip_redteam=skip_redteam)
                except RoleTokenExhaustionError as exc:
                    report_role_token_exhaustion(error=exc)
                    return 1
                except ImplementerAuthorityViolationError:
                    print("watcher stopped before candidate admission or "
                          "landing.")
                    return 1
                except FatalArchitectLandingError as exc:
                    print("Architect landing needs user action: " + str(exc))
                    return 1
                if backlog_outcome == ROLE_CONTRACT_RESTART_REQUIRED:
                    return report_role_contract_restart()
                contract_exit = role_contract_exit_status()
                if contract_exit is not None:
                    return contract_exit
                if (rendezvous.source_changed()
                        or os.path.getmtime(source_path) != source_stamp):
                    print("daemon source changed on disk; exiting so "
                          "the next start runs it (relaunch --watch).")
                    return 0
                completed_cycles = rendezvous.completed_ticket_cycles()
                active_cycles = active_ticket_cycle_count(
                    skip_redteam=skip_redteam)
                if (args.cycle is not None and args.cycle > 0
                        and completed_cycles >= args.cycle
                        and rendezvous.all_idle() and active_cycles):
                    print("cycle limit reached, but " + str(active_cycles)
                          + " extra active ticket cycle(s) remain in saved "
                          "state; refusing to claim a clean exit. No new "
                          "ticket will be started; inspect the ticket-cycle "
                          "state and mailbox receipts.")
                    return 1
                if (args.cycle is not None and args.cycle > 0
                        and completed_cycles >= args.cycle
                        and rendezvous.all_idle()
                        and active_cycles == 0):
                    barrier, completion_error = (
                        acquire_positive_cycle_exit_barrier(
                            backlog_outcome=backlog_outcome,
                            skip_redteam=skip_redteam))
                    if barrier is None:
                        if completion_error is not None:
                            report_cycle_completion_unverified(
                                error=completion_error)
                            if completion_error.startswith(
                                    ARCHITECT_NOTES_DEBT_PREFIX):
                                return 1
                        # Admin turns and P landings are cycle-free, but a
                        # positive limit cannot abandon them.  The next pass
                        # bypasses the reached ticket limit only for that
                        # exact administrative route.
                        continue
                    cycle_completion_barrier = barrier
                    try:
                        finish_finite_watch_progress(
                            limit=args.cycle,
                            completed=completed_cycles,
                            topology=watch_topology)
                    except TicketCycleStateError as exc:
                        print("cycle limit was reached, but durable progress "
                              "could not prove a clean exit: " + str(exc))
                        return 1
                    report_cycle_limit_exit(
                        completed_cycles=completed_cycles,
                        cycle_limit=args.cycle,
                        skip_redteam=skip_redteam)
                    return 0
                if rendezvous.window_ready():
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
                                    skip_redteam=skip_redteam)
                            else:
                                report_cycle_work_complete(
                                    completed_cycles=completed_cycles)
                            return 0
                        if completion_error is not None:
                            report_cycle_completion_unverified(
                                error=completion_error)
                            if completion_error.startswith(
                                    ARCHITECT_NOTES_DEBT_PREFIX):
                                return 1
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
                                skip_redteam=skip_redteam)
                        else:
                            report_cycle_work_complete(
                                completed_cycles=completed_cycles)
                        return 0
                    if completion_error is not None:
                        report_cycle_completion_unverified(
                            error=completion_error)
                        if completion_error.startswith(
                                ARCHITECT_NOTES_DEBT_PREFIX):
                            return 1
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

    print("choose an action such as --watch, --send, or --restart-implementer "
          "(see --help)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
