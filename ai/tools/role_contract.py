#!/usr/bin/env python3
"""Read the small protected contract shared by the mailbox tools."""

import json
from pathlib import Path, PurePosixPath
import stat


ROLE_CONTRACT_PATH = "ai/notes/role-contract.yaml"
MAX_CONTRACT_BYTES = 64 * 1024


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


def validate_role_contract(value):
    """Require the complete version-one contract and return it unchanged."""
    top = ("schema_version", "roles", "candidate", "landing", "backlog",
           "evidence", "runtime", "protected_paths")
    _keys(value, top, "role contract")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise RoleContractError("schema_version must be 1")

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
        "backlog": ("editor",),
        "evidence": ("red_team_advisory", "protected_policy_review_rounds"),
        "runtime": ("implementer_review_minutes",
                    "dispatch_timeout_default_minutes"),
        "protected_paths": ("contract", "permanent_notes", "role_files",
                            "guard_files"),
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
    _type(value["evidence"]["red_team_advisory"], bool,
          "evidence.red_team_advisory")
    _type(value["evidence"]["protected_policy_review_rounds"], int,
          "evidence.protected_policy_review_rounds")
    for name in ("implementer_review_minutes",
                 "dispatch_timeout_default_minutes"):
        _type(value["runtime"][name], int, "runtime." + name)
        if value["runtime"][name] <= 0:
            raise RoleContractError("runtime." + name + " must be positive")

    protected = value["protected_paths"]
    _path(protected["contract"], "protected_paths.contract")
    for group in ("permanent_notes", "role_files", "guard_files"):
        _type(protected[group], list, "protected_paths." + group)
        if not protected[group] or len(protected[group]) != len(set(protected[group])):
            raise RoleContractError("protected_paths." + group
                                    + " must be a nonempty unique list")
        for index, item in enumerate(protected[group]):
            _path(item, "protected_paths." + group + "[" + str(index) + "]")
    return value


def load_role_contract(path=None):
    """Read one canonical JSON-compatible YAML contract from a regular file."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "notes" / "role-contract.yaml"
    else:
        path = Path(path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RoleContractError("role contract must be a regular file")
    data = path.read_bytes()
    if len(data) > MAX_CONTRACT_BYTES:
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
