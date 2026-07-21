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
    python ai/tools/mailbox_daemon.py --watch --architect-context 400000 \
                                           --implementer-context 64000 \
                                           --sol-context 300000
                                                    # context budgets: a turn
        compacts (summarizes its own history and continues) whenever its
        live context reaches its own budget; the three roles have separate
        options; --implementer-context controls the coding shell and does
        not set an Ollama model's maximum context
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
_CONTROL_PLANE_HANDOFF = _load_local_tool(
    "control_plane_handoff.py", "_mailbox_local_control_plane_handoff",
    "cannot load the control-plane handoff checker")


# The daemon's source is split across this file and the mailbox_*.py part
# files beside it. Parts hold definitions only; every constant and every
# runtime-rebindable setting lives HERE. Each part names its collaborators
# through the module-level name ``daemon`` -- this very module, injected
# below -- so the daemon keeps ONE namespace: tests and sibling tools
# address every function as ``mailbox_daemon.<name>``, and rebinding a name
# takes effect everywhere at once.
MAILBOX_PART_FILES = (
    "mailbox_worktrees.py",
    "mailbox_watch.py",
    "mailbox_providers.py",
    "mailbox_envelopes.py",
    "mailbox_tickets.py",
    "mailbox_store.py",
    "mailbox_dispatch.py",
    "mailbox_landing.py",
    "mailbox_control_plane.py",
    "mailbox_recovery.py",
    "mailbox_cycles.py",
)


class _DaemonNamespace:
    """Live view of this module's namespace, handed to every part file.

    Part code reads each collaborator as ``daemon.<name>`` at call time.
    Reads and writes go straight to this module's own variables, so when a
    test rebinds ``mailbox_daemon.<name>``, every part sees the new binding
    immediately. A view is used instead of the module object because tests
    also load isolated daemon copies that are never entered into Python's
    loaded-module registry, where no module object is reachable by name.
    """

    def __init__(self, module_globals):
        object.__setattr__(self, "_module_globals", module_globals)

    def __getattr__(self, name):
        try:
            return self._module_globals[name]
        except KeyError:
            raise AttributeError(
                "mailbox daemon namespace has no name " + repr(name))

    def __setattr__(self, name, value):
        self._module_globals[name] = value


_DAEMON_NAMESPACE = _DaemonNamespace(module_globals=globals())


def _adopt_daemon_part(filename):
    """Load one part file beside this daemon and adopt its definitions.

    Arguments:
      filename = the part's file name inside this daemon's own directory.

    Returns:
      The loaded part module.

    A part's module-level code is only definitions, so it runs without the
    daemon. Right after it runs, the daemon namespace view is stored as the
    part's ``daemon`` attribute, and every name in the part's PART_EXPORTS
    tuple is bound into this module. A name that already exists here is
    refused: two owners for one function would make behavior depend on
    load order.
    """
    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(
        "_mailbox_part_" + filename[:-3], path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load daemon part " + path)
    part = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(part)
    part.daemon = _DAEMON_NAMESPACE
    for name in part.PART_EXPORTS:
        if name in globals():
            raise RuntimeError(
                "daemon part " + filename + " redefines " + name)
        globals()[name] = getattr(part, name)
    return part


for _part_file in MAILBOX_PART_FILES:
    _adopt_daemon_part(filename=_part_file)
del _part_file


def mailbox_sources_changed(source_path, source_stamp, companion_sources):
    """Report whether any daemon source file changed since a watch began.

    Arguments:
      source_path  = this daemon file's absolute path.
      source_stamp = its modification time when the watch started.
      companion_sources = (path, stamp) pairs for every loaded part file.

    Returns:
      True when any watched file's current modification time differs from
      its saved stamp or the file can no longer be read; False otherwise.
    """
    watched = [(source_path, source_stamp)] + list(companion_sources)
    for path, stamp in watched:
        try:
            current = os.path.getmtime(path)
        except OSError:
            return True
        if current != stamp:
            return True
    return False


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
MINIMUM_OLLAMA_CONTEXT = 32768

# Each role has an independent point at which it summarizes an unusually
# long turn. Architect and Implementer shells read
# CLAUDE_CODE_AUTO_COMPACT_WINDOW. Sol receives
# model_auto_compact_token_limit in its command. Override the defaults with
# the three role-specific context options.
# Each dispatch is also explicitly non-persistent, so a later turn cannot
# inherit an earlier ticket's provider conversation.
DEFAULT_ARCHITECT_CONTEXT_BUDGET = 500000
DEFAULT_IMPLEMENTER_CONTEXT_BUDGET = 500000
DEFAULT_OLLAMA_IMPLEMENTER_CONTEXT_BUDGET = 64000
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

# dispatch() reads this for the Architect's Claude environment; main()
# rebinds it from --architect-context. The Implementer limit is saved in
# IMPLEMENTER_RUNTIME, and Sol's limit rides inside AGENT_COMMANDS.
ARCHITECT_CONTEXT_BUDGET = DEFAULT_ARCHITECT_CONTEXT_BUDGET
IMPLEMENTER_RUNTIME = None

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
CONTROL_PLANE_MIGRATION_PATH = (
    "ai/tools/control-plane-state-migration.yaml")
CONTROL_PLANE_PRESERVED_INVARIANTS = (
    "active_ticket_identity",
    "candidate_identity",
    "completed_landing_identity",
    "recovery_state",
)
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


ARCHITECT_CANDIDATE_FORBIDDEN_PREFIXES = tuple(
    _PROTECTED_PATHS["candidate_forbidden_prefixes"])


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


# main() owns this only while a locked --watch is live.  Keeping the public
# process_backlog()/drain_lane()/dispatch() call shapes unchanged preserves
# finite callers and the existing focused reproduction suites.
_ACTIVE_WATCH_RENDEZVOUS = None
_RENDEZVOUS_LOCAL = threading.local()
_TOKEN_EXHAUSTION_STOP = threading.Event()
_NO_ELIGIBLE_MAINTENANCE_WORK = threading.Event()


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
        implementer_model=DEFAULT_IMPLEMENTER_MODEL,
        implementer_compaction_limit=DEFAULT_IMPLEMENTER_CONTEXT_BUDGET):
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
        run=subprocess.run,
        implementer_compaction_limit=implementer_compaction_limit,
        ollama_minimum_context=MINIMUM_OLLAMA_CONTEXT,
        implementer_preamble=agent_preamble(agent="opus"))


# main() rebuilds this from the command-line flags; the module-level
# value keeps imports and direct function calls working at the defaults.
AGENT_COMMANDS = build_agent_commands(
    fable_effort=DEFAULT_FABLE_EFFORT,
    opus_effort=DEFAULT_OPUS_EFFORT,
    sol_effort=DEFAULT_SOL_EFFORT,
    sol_context_budget=DEFAULT_SOL_CONTEXT_BUDGET)
IMPLEMENTER_RUNTIME = implementer_runtime_record(
    provider=DEFAULT_IMPLEMENTER_PROVIDER,
    model=DEFAULT_IMPLEMENTER_MODEL,
    context_limit=DEFAULT_IMPLEMENTER_CONTEXT_BUDGET,
    compaction_limit=DEFAULT_IMPLEMENTER_CONTEXT_BUDGET)

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
IMPLEMENTER_BUDGET_CHECKPOINT_HEADING = (
    "### IMPLEMENTER_HANDOFF: BUDGET BLOCKED")
ARCHITECT_BUDGET_REPAIR_HEADING = (
    "### ARCHITECT_HANDOFF: BUDGET CHECKPOINT — REVISED DIRECTIVE")
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
    "contradictory, return a blocker instead of making that decision. If a\n"
    "clean candidate exceeds a positive ticket character limit, preserve it\n"
    "and return exactly `### IMPLEMENTER_HANDOFF: BUDGET BLOCKED`, one exact\n"
    "Candidate commit row, and one Character-change result row beginning\n"
    "`over limit`. The Architect, not the Implementer, revises the plan.\n\n")

REDTEAM_ROLE_PREAMBLE = (
    "ROUTE ROLE: You are the bounded Red Team. Read and obey the exact\n"
    "authoritative role file named below before acting. Sol is advisory and\n"
    "never implements a ticket. A confirmed finding must include a validated,\n"
    "implementation-ready Repair directive, but it returns to the Architect\n"
    "as candidate input and never executes itself.\n\n")


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
OLLAMA_TOKEN_EXHAUSTION_MARKERS = (
    "usage limit", "rate limit exceeded", "insufficient credits",
    "credit balance is too low", "quota exceeded",
)


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


STALE_INTEGRATION_REVALIDATION = (
    "STALE — REQUIRES INTEGRATION REVALIDATION")
STALE_INTEGRATION_RE = re.compile(
    re.escape(STALE_INTEGRATION_REVALIDATION)
    + r": C=([0-9a-f]{40}) L=([0-9a-f]{40})"
      r" M0=([0-9a-f]{40}) M1=([0-9a-f]{40})")


ARCHITECT_NOTES_DEBT_PREFIX = "permanent-note user action required: "


DAEMON_MESSAGE_CONSUMED = "consumed"
DAEMON_NOTE_DEFERRED = "retryable-note-deferred"
DAEMON_CONTROL_PLANE_WAITING = "control-plane-waiting"
DAEMON_MESSAGE_HARD_STOP = "hard-stop"
ROLE_CONTRACT_RESTART_REQUIRED = "role-contract-restart-required"


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


class TicketCycleLimitDeferred(TicketCycleStateError):
    """A valid new ticket belongs to a later finite watch."""


def attempt_cycle_zero_completion(backlog_outcome, skip_redteam,
                                  completed_cycles):
    """Try the cycle-0 clean exit: take the completion barrier once.

    Arguments:
      backlog_outcome  = the just-finished mailbox pass result.
      skip_redteam     = True when this watch excludes the Sol route.
      completed_cycles = ticket cycles completed by this watch so far.

    Returns:
      (barrier, exit_code). ``barrier`` is the held completion-barrier lock
      file when every enabled role is provably done; the caller then owns
      its release and exits 0. ``exit_code`` is 1 when completion is
      blocked by failed permanent-note debt the user must repair. Both are
      None when the watch must simply continue polling.
    """
    barrier, completion_error = acquire_cycle_completion_barrier(
        backlog_outcome=backlog_outcome, skip_redteam=skip_redteam)
    if barrier is not None:
        report_cycle_work_complete(
            completed_cycles=completed_cycles, skip_redteam=skip_redteam)
        return barrier, None
    if completion_error is not None:
        report_cycle_completion_unverified(error=completion_error)
        if completion_error.startswith(ARCHITECT_NOTES_DEBT_PREFIX):
            return None, 1
    return None, None


def main():
    # both are rebound below from the parsed command line; Python wants
    # the global declaration before the first mention of either name.
    global AGENT_COMMANDS
    global DISPATCH_TIMEOUT_MINUTES
    global ARCHITECT_CONTEXT_BUDGET
    global IMPLEMENTER_RUNTIME
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
                        default=None,
                        help="model name used for the Implementer; select "
                             "its service with --implementer-provider; "
                             "mailbox filenames still contain opus "
                             "(Claude default: "
                             + DEFAULT_IMPLEMENTER_MODEL
                             + "; required explicitly for Ollama)")
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
    parser.add_argument(
        "--architect-context", "--claude-context",
        dest="architect_context", metavar="TOKENS", type=positive_int,
        default=DEFAULT_ARCHITECT_CONTEXT_BUDGET,
        help="inside one Architect turn, replace older context with a "
             "shorter summary at this many tokens; --claude-context is a "
             "compatibility name for this Architect-only option (default: "
             + str(DEFAULT_ARCHITECT_CONTEXT_BUDGET) + ")")
    parser.add_argument(
        "--implementer-context", metavar="TOKENS", type=positive_int,
        default=None,
        help="inside one Implementer turn, ask its Claude Code shell to "
             "replace older context with a shorter summary at this many "
             "tokens; for Ollama this must not exceed the model context "
             "reported by Ollama (default: 500000 with Claude, 64000 with "
             "Ollama)")
    parser.add_argument("--sol-context", metavar="TOKENS",
                        type=positive_int, default=DEFAULT_SOL_CONTEXT_BUDGET,
                        help="inside one Red Team turn, ask Codex to replace "
                             "older conversation text with a shorter summary "
                             "at this many tokens (default: "
                             + str(DEFAULT_SOL_CONTEXT_BUDGET) + ")")
    args = parser.parse_args()
    if args.implementer_model is None:
        if args.implementer_provider == "ollama" and (
                args.watch or args.once or args.ping or args.dry_run):
            print("--implementer-provider ollama requires an explicit "
                  "--implementer-model")
            return 1
        args.implementer_model = DEFAULT_IMPLEMENTER_MODEL
    implementer_context = (
        default_implementer_context(provider=args.implementer_provider)
        if args.implementer_context is None else args.implementer_context)
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
            dry_run=args.dry_run,
            implementer_compaction_limit=implementer_context) else 1

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
    ARCHITECT_CONTEXT_BUDGET = args.architect_context
    REVIEW_EFFORT = args.review_effort

    if args.watch or args.once:
        try:
            selection_problem = active_implementer_runtime_problem(
                provider=args.implementer_provider,
                model=args.implementer_model,
                compaction_limit=implementer_context)
        except (OSError, TicketCycleStateError) as exc:
            print("Implementer runtime state error: " + str(exc))
            return 1
        if selection_problem is not None:
            print("Implementer runtime state error: " + selection_problem)
            return 1
        try:
            IMPLEMENTER_RUNTIME = verified_implementer_runtime(
                provider=args.implementer_provider,
                model=args.implementer_model,
                compaction_limit=implementer_context,
                dry_run=args.dry_run)
        except (OSError, ValueError) as exc:
            print("Implementer provider error: " + str(exc))
            return 1
        try:
            runtime_problem = active_implementer_runtime_problem(
                provider=IMPLEMENTER_RUNTIME["provider"],
                model=IMPLEMENTER_RUNTIME["model"],
                context_limit=IMPLEMENTER_RUNTIME["context_limit"],
                compaction_limit=IMPLEMENTER_RUNTIME["compaction_limit"])
        except (OSError, TicketCycleStateError) as exc:
            print("Implementer runtime state error: " + str(exc))
            return 1
        if runtime_problem is not None:
            print("Implementer runtime state error: " + runtime_problem)
            return 1

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
        implementer_effort = (args.opus_effort
                              if args.implementer_provider == "claude"
                              else "provider default (ollama)")
        sol_effort_text = "disabled" if skip_redteam else args.sol_effort
        print("effort levels: architect/fable=" + args.fable_effort
              + " implementer/opus=" + implementer_effort
              + " sol=" + sol_effort_text
              + " routine-review=" + args.review_effort)
        if args.implementer_provider == "ollama":
            if skip_redteam:
                sol_context_text = "Sol disabled"
            else:
                sol_context_text = ("Sol compacts at "
                                    + str(args.sol_context))
            print("context: Architect compacts at "
                  + str(args.architect_context)
                  + "; Ollama model context="
                  + str(IMPLEMENTER_RUNTIME["context_limit"])
                  + "; Implementer shell compacts at "
                  + str(implementer_context) + "; " + sol_context_text)
        else:
            if skip_redteam:
                sol_budget_text = ("sol=disabled (a Claude turn compacts "
                                   "at its budget)")
            else:
                sol_budget_text = ("sol=" + str(args.sol_context)
                                   + " tokens (a turn compacts at its "
                                   "budget)")
            print("context budgets: architect="
                  + str(args.architect_context) + " implementer="
                  + str(implementer_context) + " " + sol_budget_text)
        if skip_redteam:
            print("two-role watch: Red Team and the entire Sol route are "
                  "disabled; existing to-sol messages stay queued and "
                  "untouched")
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
        # a daemon fix is a no-op for the loop already running: watch our
        # own source and exit when it changes, so stale code can never keep
        # dispatching. Exiting (not self-reloading) is deliberate -- a
        # restart is one keystroke and never picks up a half-saved edit.
        # The daemon's source is this file PLUS every loaded part file, so
        # all of them are watched.
        source_path = os.path.abspath(__file__)
        source_stamp = os.path.getmtime(source_path)
        companion_sources = []
        for part_file in MAILBOX_PART_FILES:
            part_path = os.path.join(SCRIPT_DIR, part_file)
            companion_sources.append(
                (part_path, os.path.getmtime(part_path)))
        companion_sources = tuple(companion_sources)
        rendezvous = SafeKillRendezvous(
            source_path=source_path, source_stamp=source_stamp,
            ticket_cycle_limit=(args.cycle
                                if args.cycle is not None and args.cycle > 0
                                else None),
            ticket_cycle_topology=(
                watch_topology
                if args.cycle is not None and args.cycle > 0 else None),
            companion_sources=companion_sources)
        _ACTIVE_WATCH_RENDEZVOUS = rendezvous
        _NO_ELIGIBLE_MAINTENANCE_WORK.clear()
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
                        and mailbox_sources_changed(
                            source_path=source_path,
                            source_stamp=source_stamp,
                            companion_sources=companion_sources)):
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
                        or mailbox_sources_changed(
                            source_path=source_path,
                            source_stamp=source_stamp,
                            companion_sources=companion_sources)):
                    print("daemon source changed on disk; exiting so "
                          "the next start runs it (relaunch --watch).")
                    return 0
                completed_cycles = rendezvous.completed_ticket_cycles()
                active_cycles = active_ticket_cycle_count(
                    skip_redteam=skip_redteam)
                if (_NO_ELIGIBLE_MAINTENANCE_WORK.is_set()
                        and rendezvous.all_idle() and active_cycles == 0
                        and not enabled_pending_messages(
                            skip_redteam=skip_redteam)):
                    if args.cycle is not None and args.cycle > 0:
                        clear_finite_watch_progress(
                            topology=watch_topology)
                    print("no eligible Open BUG FIX remains at or above "
                          + DISCOVERY_SEVERITY
                          + "; watcher stopped while every role is idle.",
                          flush=True)
                    return 0
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
                        barrier, exit_code = attempt_cycle_zero_completion(
                            backlog_outcome=backlog_outcome,
                            skip_redteam=skip_redteam,
                            completed_cycles=completed_cycles)
                        if barrier is not None:
                            cycle_completion_barrier = barrier
                            return 0
                        if exit_code is not None:
                            return exit_code
                    run_safe_kill_countdown(controller=rendezvous)
                    # Queued work resumes immediately after the manufactured
                    # window rather than paying an extra ordinary poll delay.
                    continue
                if args.cycle == 0 and rendezvous.all_idle():
                    barrier, exit_code = attempt_cycle_zero_completion(
                        backlog_outcome=backlog_outcome,
                        skip_redteam=skip_redteam,
                        completed_cycles=completed_cycles)
                    if barrier is not None:
                        cycle_completion_barrier = barrier
                        return 0
                    if exit_code is not None:
                        return exit_code
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
            release_dispatch_lock(lock_file=dispatch_lock)
            if skip_redteam_lock is not None:
                release_skip_redteam_lock(lock_file=skip_redteam_lock)
            if cycle_completion_barrier is not None:
                release_cycle_completion_barrier(
                    lock_file=cycle_completion_barrier)

    print("choose an action such as --watch, --send, or --restart-implementer "
          "(see --help)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
