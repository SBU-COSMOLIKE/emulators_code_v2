#!/usr/bin/env python3
"""Read the small protected contract shared by the mailbox tools."""

import json
from pathlib import Path, PurePosixPath
import stat


_BOOTSTRAP_CONTRACT_PATH = "ai/notes/role-contract.yaml"
_BOOTSTRAP_CONTRACT_BYTES = 64 * 1024


class RoleContractError(ValueError):
    """The protected role contract is missing, ambiguous, or malformed."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RoleContractError("duplicate key: " + repr(key))
        result[key] = value
    return result


def _keys(value, expected, where):
    if type(value) is not dict:
        raise RoleContractError(where + " must be a mapping")
    if set(value) != set(expected):
        raise RoleContractError(where + " must contain exactly: "
                                + ", ".join(expected))


def _type(value, expected, where):
    if type(value) is not expected:
        raise RoleContractError(where + " has the wrong value type")


def _path(value, where):
    _type(value, str, where)
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or str(path) != value):
        raise RoleContractError(where + " must be a repository-relative path")


def _path_list(value, where):
    _type(value, list, where)
    if not value or len(value) != len(set(value)):
        raise RoleContractError(where + " must be a nonempty unique list")
    for index, item in enumerate(value):
        _path(item, where + "[" + str(index) + "]")


def _path_map(value, names, where):
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
                    "dispatch_timeout_default_minutes"),
        "protected_paths": ("candidate_forbidden_files",
                            "candidate_forbidden_prefixes", "contract",
                            "guard_files", "permanent_notes", "role_files",
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
    if value["limits"]["role_contract_bytes"] != _BOOTSTRAP_CONTRACT_BYTES:
        raise RoleContractError(
            "limits.role_contract_bytes must match the protected reader cap")
    for name in ("implementer_review_minutes",
                 "dispatch_timeout_default_minutes"):
        _type(value["runtime"][name], int, "runtime." + name)
        if value["runtime"][name] <= 0:
            raise RoleContractError("runtime." + name + " must be positive")

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
    for group in ("candidate_forbidden_files", "permanent_notes", "role_files"):
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
              ("backlog_bundle", "backlog_guard", "handoff_contract",
               "handoff_router", "implementer_checkpoint", "mailbox_daemon",
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
    canonical = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if text != canonical:
        raise RoleContractError("role contract must use canonical formatting")
    return value


ROLE_CONTRACT = load_role_contract()
ROLE_CONTRACT_PATH = ROLE_CONTRACT["protected_paths"]["contract"]
MAX_CONTRACT_BYTES = ROLE_CONTRACT["limits"]["role_contract_bytes"]
