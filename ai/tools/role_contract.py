#!/usr/bin/env python3
"""Read the small protected contract shared by the mailbox tools."""

import json
from pathlib import Path, PurePosixPath
import stat


_BOOTSTRAP_CONTRACT_PATH = "ai/notes/role-contract.yaml"
_BOOTSTRAP_CONTRACT_BYTES = 64 * 1024
_BOOTSTRAP_ROLE_FILES = (
    ".claude/FABLE_ROLE.md",
    ".claude/OPUS_ROLE.md",
    ".codex/REDTEAM_ROLE.md",
)
_BOOTSTRAP_GUARD_FILES = {
    "permanent_note_guard": "ai/tools/permanent_note_guard.py",
    "role_contract_reader": "ai/tools/role_contract.py",
}
_BOOTSTRAP_TRUSTED_TOOLS = {
    "backlog_bundle": "ai/tools/backlog_bundle.py",
    "backlog_guard": "ai/tools/backlog_guard.py",
    "candidate_admission": "ai/tools/candidate_admission.py",
    "control_plane_handoff": "ai/tools/control_plane_handoff.py",
    "handoff_contract": "ai/tools/handoff_contract.py",
    "handoff_router": "ai/tools/handoff_router.py",
    "implementer_checkpoint": "ai/tools/implementer_checkpoint_hook.py",
    "mailbox_daemon": "ai/tools/mailbox_daemon.py",
    "mailbox_control_plane": "ai/tools/mailbox_control_plane.py",
    "mailbox_cycles": "ai/tools/mailbox_cycles.py",
    "mailbox_dispatch": "ai/tools/mailbox_dispatch.py",
    "mailbox_envelopes": "ai/tools/mailbox_envelopes.py",
    "mailbox_landing": "ai/tools/mailbox_landing.py",
    "mailbox_providers": "ai/tools/mailbox_providers.py",
    "mailbox_recovery": "ai/tools/mailbox_recovery.py",
    "mailbox_store": "ai/tools/mailbox_store.py",
    "mailbox_tickets": "ai/tools/mailbox_tickets.py",
    "mailbox_watch": "ai/tools/mailbox_watch.py",
    "mailbox_worktrees": "ai/tools/mailbox_worktrees.py",
    "provider_health": "ai/tools/provider_health.py",
    "review_dispatch": "ai/tools/review_dispatch.py",
    "reopen_transition": "ai/tools/reopen_transition.py",
    "ticket_change_guard": "ai/tools/ticket_change_guard.py",
}
_MINIMUM_PERMANENT_NOTES = {
    "ai/notes/MEMORY.md",
    "ai/notes/project-and-history.md",
    "ai/notes/conventions-and-workflow.md",
    "ai/notes/python-changes-go-no-go.md",
    "ai/notes/models-and-designs.md",
    "ai/notes/training-stack.md",
    "ai/notes/artifacts-inference-warmstart.md",
    "ai/notes/data-generation-and-cuts.md",
    "ai/notes/families-background-mps.md",
    "ai/notes/families-scalar-cmb.md",
    "ai/notes/readme-go-no-go.md",
}
_MINIMUM_PROTECTED_REFERENCE_FILES = {
    "ai/notes/implementer-failure-modes.yaml",
}
_BOOTSTRAP_BACKLOG_PATH = "ai/notes/backlog.md"
_BOOTSTRAP_WORKTREES = {
    "architect_branch": "refs/heads/claude/mailbox-primary",
    "architect_name": "mailbox-primary",
    "claude_branch_prefix": "claude/",
    "cleanup_action": "--clean-all",
    "implementer_branch": "refs/heads/claude/mailbox-implementer",
    "implementer_name": "mailbox-implementer",
    "legacy_cleanup_prefix": "worktree-agent-",
    "sol_branch": "refs/heads/codex/mailbox-sol",
    "sol_branch_prefix": "codex/",
    "sol_name": "mailbox-sol",
    "topology": "separate-role-worktrees-v1",
}
_MINIMUM_FORBIDDEN_FILES = {
    "CLAUDE.md", ".gitattributes", ".gitignore", ".gitmodules",
    "ai/notes/backlog.md", "ai/notes/.backlog-guard.json",
    "ai/notes/.backlog-guard.lock",
}
_MINIMUM_FORBIDDEN_PREFIXES = {
    ".claude/", ".codex/", "ai/tools/", "ai/notes/mailbox/",
    "ai/notes/relay/",
}


class RoleContractError(ValueError):
    """The protected role contract is missing, ambiguous, or malformed."""


def _object(pairs):
    """Build one YAML mapping while refusing any repeated key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise RoleContractError("duplicate key: " + repr(key))
        result[key] = value
    return result


def _keys(value, expected, where):
    """Require ``value`` to be a mapping with exactly the expected keys."""
    if type(value) is not dict:
        raise RoleContractError(where + " must be a mapping")
    if set(value) != set(expected):
        raise RoleContractError(where + " must contain exactly: "
                                + ", ".join(expected))


def _type(value, expected, where):
    """Require ``value`` to have exactly the expected Python type."""
    if type(value) is not expected:
        raise RoleContractError(where + " has the wrong value type")


def _path(value, where):
    """Require one repository-relative POSIX path with no escapes."""
    _type(value, str, where)
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or str(path) != value):
        raise RoleContractError(where + " must be a repository-relative path")


def _path_list(value, where):
    """Require one nonempty list of unique repository-relative paths."""
    _type(value, list, where)
    if not value or len(value) != len(set(value)):
        raise RoleContractError(where + " must be a nonempty unique list")
    for index, item in enumerate(value):
        _path(item, where + "[" + str(index) + "]")


def _path_map(value, names, where):
    """Require one mapping from the named keys to valid paths."""
    _keys(value, names, where)
    for name in names:
        _path(value[name], where + "." + name)


def validate_role_contract(value):
    """Require the complete protected contract and return it unchanged."""
    top = ("schema_version", "roles", "candidate", "landing", "backlog",
           "evidence", "limits", "runtime", "protected_paths", "worktrees")
    _keys(value, top, "role contract")
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise RoleContractError("schema_version must be 2")

    roles = value["roles"]
    _keys(roles, ("architect", "implementer", "red_team"), "roles")
    permissions = ("may_edit_source", "may_decide", "may_land",
                   "may_edit_backlog", "may_edit_protected_policy")
    for name in ("architect", "implementer", "red_team"):
        _keys(roles[name], permissions, "roles." + name)
        for permission in permissions:
            _type(roles[name][permission], bool,
                  "roles." + name + "." + permission)

    sections = {
        "candidate": ("creator", "immutable", "full_hash_required"),
        "landing": ("creator", "parent_count", "force_push_allowed",
                    "audited_delta_required"),
        "backlog": ("editor", "path"),
        "evidence": ("red_team_advisory", "protected_policy_review_rounds"),
        "limits": ("protected_policy_file_bytes", "role_contract_bytes"),
        "runtime": ("implementer_review_minutes",
                    "dispatch_timeout_default_minutes",
                    "routine_review_effort"),
        "protected_paths": ("candidate_forbidden_files",
                            "candidate_forbidden_prefixes", "contract",
                            "guard_files", "permanent_notes",
                            "protected_reference_files", "role_files",
                            "trusted_tools"),
        "worktrees": ("architect_branch", "architect_name",
                      "claude_branch_prefix", "cleanup_action",
                      "implementer_branch", "implementer_name",
                      "legacy_cleanup_prefix", "sol_branch",
                      "sol_branch_prefix", "sol_name", "topology"),
    }
    for name, keys in sections.items():
        _keys(value[name], keys, name)

    for name in ("creator",):
        _type(value["candidate"][name], str, "candidate." + name)
    for name in ("immutable", "full_hash_required"):
        _type(value["candidate"][name], bool, "candidate." + name)
    _type(value["landing"]["creator"], str, "landing.creator")
    _type(value["landing"]["parent_count"], int, "landing.parent_count")
    for name in ("force_push_allowed", "audited_delta_required"):
        _type(value["landing"][name], bool, "landing." + name)
    _type(value["backlog"]["editor"], str, "backlog.editor")
    _path(value["backlog"]["path"], "backlog.path")
    _type(value["evidence"]["red_team_advisory"], bool,
          "evidence.red_team_advisory")
    _type(value["evidence"]["protected_policy_review_rounds"], int,
          "evidence.protected_policy_review_rounds")
    for name in ("protected_policy_file_bytes", "role_contract_bytes"):
        _type(value["limits"][name], int, "limits." + name)
        if value["limits"][name] <= 0:
            raise RoleContractError("limits." + name + " must be positive")
    if value["limits"]["role_contract_bytes"] > _BOOTSTRAP_CONTRACT_BYTES:
        raise RoleContractError(
            "limits.role_contract_bytes cannot exceed the protected reader "
            "cap")
    for name in ("implementer_review_minutes",
                 "dispatch_timeout_default_minutes"):
        _type(value["runtime"][name], int, "runtime." + name)
        if value["runtime"][name] <= 0:
            raise RoleContractError("runtime." + name + " must be positive")
    _type(value["runtime"]["routine_review_effort"], str,
          "runtime.routine_review_effort")
    if value["runtime"]["routine_review_effort"] not in {
            "low", "medium", "high", "xhigh"}:
        raise RoleContractError(
            "runtime.routine_review_effort has an unsupported value")

    for name in ("architect_branch", "architect_name", "claude_branch_prefix",
                 "cleanup_action", "implementer_branch", "implementer_name",
                 "legacy_cleanup_prefix", "sol_branch", "sol_branch_prefix",
                 "sol_name", "topology"):
        _type(value["worktrees"][name], str, "worktrees." + name)
        if not value["worktrees"][name]:
            raise RoleContractError("worktrees." + name + " must not be empty")

    protected = value["protected_paths"]
    _path(protected["contract"], "protected_paths.contract")
    if protected["contract"] != _BOOTSTRAP_CONTRACT_PATH:
        raise RoleContractError(
            "protected_paths.contract must match the protected reader path")
    for group in ("candidate_forbidden_files", "permanent_notes",
                  "protected_reference_files", "role_files"):
        _path_list(protected[group], "protected_paths." + group)
    _type(protected["candidate_forbidden_prefixes"], list,
          "protected_paths.candidate_forbidden_prefixes")
    prefixes = protected["candidate_forbidden_prefixes"]
    if not prefixes or len(prefixes) != len(set(prefixes)):
        raise RoleContractError(
            "protected_paths.candidate_forbidden_prefixes must be unique")
    for index, prefix in enumerate(prefixes):
        where = "protected_paths.candidate_forbidden_prefixes[" + str(index) + "]"
        _type(prefix, str, where)
        if not prefix.endswith("/"):
            raise RoleContractError(where + " must end with /")
        _path(prefix[:-1], where)
    _path_map(protected["guard_files"],
              ("permanent_note_guard", "role_contract_reader"),
              "protected_paths.guard_files")
    _path_map(protected["trusted_tools"],
              ("backlog_bundle", "backlog_guard", "candidate_admission",
               "control_plane_handoff", "handoff_contract", "handoff_router",
               "implementer_checkpoint", "mailbox_daemon",
               "mailbox_control_plane", "mailbox_cycles",
               "mailbox_dispatch", "mailbox_envelopes", "mailbox_landing",
               "mailbox_providers", "mailbox_recovery", "mailbox_store",
               "mailbox_tickets", "mailbox_watch", "mailbox_worktrees",
               "provider_health", "review_dispatch", "reopen_transition",
               "ticket_change_guard"),
              "protected_paths.trusted_tools")
    tool_paths = (list(protected["guard_files"].values())
                  + list(protected["trusted_tools"].values()))
    if len(tool_paths) != len(set(tool_paths)):
        raise RoleContractError("protected tool paths must be unique")
    notes_root = PurePosixPath(protected["contract"]).parent
    for note in protected["permanent_notes"]:
        path = PurePosixPath(note)
        if path.parent != notes_root or path.suffix.casefold() != ".md":
            raise RoleContractError(
                "permanent notes must be Markdown files beside the contract")
    for reference in protected["protected_reference_files"]:
        path = PurePosixPath(reference)
        if path.parent != notes_root or path.suffix.casefold() not in {
                ".json", ".yaml", ".yml"}:
            raise RoleContractError(
                "protected reference files must be structured files beside "
                "the contract")

    # An editable contract may tighten configuration, but it cannot grant
    # itself authority that the reader and daemon deliberately never expose.
    safety_floor = {
        "candidate": {"creator": "implementer", "immutable": True,
                      "full_hash_required": True},
        "landing": {"creator": "daemon", "parent_count": 1,
                    "force_push_allowed": False,
                    "audited_delta_required": True},
        "evidence": {"red_team_advisory": True,
                     "protected_policy_review_rounds": 1},
    }
    for section, required in safety_floor.items():
        if value[section] != required:
            raise RoleContractError(
                section + " does not match the compiled safety floor")
    required_permissions = {
        "architect": {
            "may_edit_source": False, "may_decide": True, "may_land": False,
            "may_edit_backlog": True, "may_edit_protected_policy": True,
        },
        "implementer": {
            "may_edit_source": True, "may_decide": False, "may_land": False,
            "may_edit_backlog": False, "may_edit_protected_policy": False,
        },
        "red_team": {
            "may_edit_source": False, "may_decide": False,
            "may_land": False, "may_edit_backlog": False,
            "may_edit_protected_policy": False,
        },
    }
    for role, required in required_permissions.items():
        if value["roles"][role] != required:
            raise RoleContractError(
                "roles." + role + " does not match the compiled safety "
                "floor")
    if value["backlog"]["editor"] != "architect":
        raise RoleContractError(
            "backlog.editor does not match the compiled safety floor")
    if value["backlog"]["path"] != _BOOTSTRAP_BACKLOG_PATH:
        raise RoleContractError(
            "backlog.path requires an explicit protocol migration")
    if value["worktrees"] != _BOOTSTRAP_WORKTREES:
        raise RoleContractError(
            "worktrees requires an explicit saved-state migration")
    if tuple(protected["role_files"]) != _BOOTSTRAP_ROLE_FILES:
        raise RoleContractError(
            "protected role files do not match the bootstrap identities")
    if protected["guard_files"] != _BOOTSTRAP_GUARD_FILES:
        raise RoleContractError(
            "protected guard files do not match the bootstrap identities")
    if protected["trusted_tools"] != _BOOTSTRAP_TRUSTED_TOOLS:
        raise RoleContractError(
            "trusted tools do not match the bootstrap identities")
    if not _MINIMUM_PERMANENT_NOTES.issubset(
            protected["permanent_notes"]):
        raise RoleContractError(
            "protected permanent notes dropped a required note")
    if not _MINIMUM_PROTECTED_REFERENCE_FILES.issubset(
            protected["protected_reference_files"]):
        raise RoleContractError(
            "protected reference files dropped a required file")
    if not _MINIMUM_FORBIDDEN_FILES.issubset(
            protected["candidate_forbidden_files"]):
        raise RoleContractError(
            "candidate forbidden files dropped a safety-floor path")
    if not _MINIMUM_FORBIDDEN_PREFIXES.issubset(
            protected["candidate_forbidden_prefixes"]):
        raise RoleContractError(
            "candidate forbidden prefixes dropped a safety-floor path")
    return value


def load_role_contract(path=None):
    """Read one canonical JSON-compatible YAML contract from a regular file."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / _BOOTSTRAP_CONTRACT_PATH
    else:
        path = Path(path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RoleContractError("role contract must be a regular file")
    if info.st_size > _BOOTSTRAP_CONTRACT_BYTES:
        raise RoleContractError("role contract is too large")
    data = path.read_bytes()
    if len(data) > _BOOTSTRAP_CONTRACT_BYTES:
        raise RoleContractError("role contract is too large")
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RoleContractError("role contract is not canonical JSON-compatible YAML") from error
    validate_role_contract(value)
    if len(data) > value["limits"]["role_contract_bytes"]:
        raise RoleContractError("role contract exceeds its configured limit")
    canonical = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if text != canonical:
        raise RoleContractError("role contract must use canonical formatting")
    return value


ROLE_CONTRACT = load_role_contract()
ROLE_CONTRACT_PATH = ROLE_CONTRACT["protected_paths"]["contract"]
MAX_CONTRACT_BYTES = ROLE_CONTRACT["limits"]["role_contract_bytes"]
