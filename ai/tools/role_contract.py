#!/usr/bin/env python3
"""Read the protected role contract that every mailbox tool obeys.

The role contract is ``ai/notes/role-contract.yaml``: a JSON-compatible
YAML mapping that records who may decide, edit source, edit the backlog,
edit protected policy, and land commits, together with the protected
paths, worktree layout, size limits, and runtime settings the tools
share. This module reads that file, validates it against a safety floor
compiled into this reader, and publishes the result as ``ROLE_CONTRACT``.

The split between file and reader is deliberate. The YAML is editable
through Architect-only protected-policy administration, so it can
tighten configurable settings, such as adding a forbidden path prefix or
changing a timeout. It can never grant authority: the permission matrix,
the creator identities, the no-force-push rule, and the trusted-tool
census must equal the compiled values below, and changing one of those
requires editing this Python file, which no mailbox ticket may touch.

``ROLE_CONTRACT = load_role_contract()`` runs at import time, so every
importer either receives a fully validated contract or fails to import.
There is no partially configured state for a later caller to trust by
accident.
"""

import json
from pathlib import Path, PurePosixPath
import stat


# Bootstrap identities compiled into this reader. The safety floor in
# validate_role_contract requires the YAML to match them exactly, or,
# for the census sets, to still contain every listed entry.
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
    """The protected role contract is missing, ambiguous, or malformed.

    Any raise means the contract cannot be trusted as policy, so the
    importing tool stops before doing mailbox work. There is no partial
    or default fallback contract.
    """


def _object(pairs):
    """Build one mapping from parsed key-value pairs, refusing repeats.

    ``json.loads`` normally builds each JSON object itself as a Python
    ``dict``, and when a key appears twice in the file it silently
    keeps only the last spelling. A file could then show one value to
    a human reading it top to bottom and a different value to the
    program. Passing this function as the parser's
    ``object_pairs_hook`` replaces that default: the parser hands over
    the raw ``(key, value)`` pairs in file order, and the mapping is
    built here, one pair at a time, refusing any key already seen.

    Arguments:
      pairs = the ``(key, value)`` tuples of one JSON object, in file
              order, supplied by ``json.loads``.

    Returns:
      A plain ``dict`` in which each key appeared exactly once.

    Raises:
      RoleContractError naming the repeated key.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise RoleContractError("duplicate key: " + repr(key))
        result[key] = value
    return result


def _keys(value, expected, where):
    """Require ``value`` to be a mapping with exactly the expected keys.

    ``set(value)`` collects a dictionary's keys (iterating a ``dict``
    yields its keys), and comparing two sets ignores order, so the
    rule is: the same keys, in any order, nothing missing and nothing
    extra. Both directions refuse deliberately. A missing key would
    make some tool read an absent entry; an extra key would let the
    file carry an entry that no tool reads, which is how a stale or
    misspelled setting hides. The mapping must also be exactly the
    ``dict`` type; a subclass is rejected for the reason documented on
    ``_type``.

    Arguments:
      value    = the contract entry being checked.
      expected = every key the mapping must contain, and no others.
      where    = the entry's dotted address inside the contract, such
                 as ``roles``, used to build the error message.

    Returns:
      None; a passing check does nothing.

    Raises:
      RoleContractError naming the address and the full required key
      list.
    """
    if type(value) is not dict:
        raise RoleContractError(where + " must be a mapping")
    if set(value) != set(expected):
        raise RoleContractError(where + " must contain exactly: "
                                + ", ".join(expected))


def _type(value, expected, where):
    """Require ``value`` to be exactly the expected type, not a subclass.

    Every Python value carries its type, and ``type(value)`` retrieves
    it: ``type(3)`` is ``int``, ``type("a")`` is ``str``. The check
    compares with ``is``, which asks whether both sides are the same
    object, so it passes only when the value's type is exactly
    ``expected``. The more common test,
    ``isinstance(value, expected)``, is looser: it also accepts every
    subclass, meaning a type derived from ``expected`` that inherits
    its behavior. That looseness matters here because Python defines
    ``bool`` as a subclass of ``int``: ``isinstance(True, int)`` is
    true, and ``True`` behaves as the number 1. Under an ``isinstance``
    check the contract entry ``"parent_count": true`` would silently
    pass as the integer 1; under this exact-type check it is an error
    the author must fix in the YAML.

    Arguments:
      value    = the contract entry being checked, as parsed from
                 ``role-contract.yaml``.
      expected = the one Python type the entry must have, for example
                 ``int`` or ``str``.
      where    = the entry's dotted address inside the contract, such
                 as ``landing.parent_count``; the error repeats it so
                 the reader can find the entry in the YAML.

    Returns:
      None; a passing check does nothing.

    Raises:
      RoleContractError naming the address.
    """
    if type(value) is not expected:
        raise RoleContractError(where + " has the wrong value type")


def _path(value, where):
    """Require one repository-relative POSIX path in canonical spelling.

    ``PurePosixPath`` is a path parser that never touches the disk: it
    splits a string such as ``ai/notes/backlog.md`` into components
    (``parts``) using ``/`` separators and can answer questions about
    the spelling. Three rules keep the path inside the repository and
    in one spelling only. First, the string must be nonempty. Second,
    it must not be absolute (no leading ``/``) and no component may be
    ``..``, the parent-directory step, so it cannot escape the
    repository root. Third, printing the parsed path back
    (``str(path)``) must reproduce the original string exactly, which
    rejects noncanonical spellings such as ``a//b``, ``./a``, or a
    trailing slash. Without the third rule, one file could appear
    under two spellings that later string comparisons would treat as
    different paths.

    Arguments:
      value = the contract entry being checked.
      where = the entry's dotted address inside the contract, used to
              build the error message.

    Returns:
      None; a passing check does nothing.

    Raises:
      RoleContractError, from ``_type`` when the entry is not a
      string, otherwise naming the address.
    """
    _type(value, str, where)
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or str(path) != value):
        raise RoleContractError(where + " must be a repository-relative path")


def _path_list(value, where):
    """Require one nonempty list of unique repository-relative paths.

    Uniqueness is checked on the raw strings first: ``set(value)``
    drops duplicates, so a length change proves a repeat. Each entry
    is then validated by ``_path``, and the address passed down gains
    the entry's position (``enumerate`` counts the entries from 0), so
    a refusal names the exact offender, such as
    ``protected_paths.role_files[2]``.

    Arguments:
      value = the contract entry being checked.
      where = the list's dotted address inside the contract.

    Returns:
      None; a passing check does nothing.

    Raises:
      RoleContractError for a non-list, an empty list, a repeated
      entry, or any entry that fails ``_path``.
    """
    _type(value, list, where)
    if not value or len(value) != len(set(value)):
        raise RoleContractError(where + " must be a nonempty unique list")
    for index, item in enumerate(value):
        _path(item, where + "[" + str(index) + "]")


def _path_map(value, names, where):
    """Require a mapping from exactly the named keys to valid paths.

    This combines the two checks above: ``_keys`` proves the mapping
    holds exactly ``names``, then every value must pass ``_path``. The
    address passed down gains the key, so a refusal names the exact
    entry, such as ``protected_paths.guard_files.role_contract_reader``.

    Arguments:
      value = the contract entry being checked.
      names = every key the mapping must contain, and no others.
      where = the mapping's dotted address inside the contract.

    Returns:
      None; a passing check does nothing.

    Raises:
      RoleContractError from ``_keys`` or ``_path``.
    """
    _keys(value, names, where)
    for name in names:
        _path(value[name], where + "." + name)


def validate_role_contract(value):
    """Require the complete protected contract and return it unchanged.

    Validation runs in three layers, and the error of the first failing
    rule names the exact dotted location:

    1. Shape: exactly the ten top-level keys, ``schema_version`` equal
       to 2, and per-section key sets with exact value types, so an
       entry no tool reads cannot hide in the file.
    2. Value rules: positive limits and timeouts, a compiled cap on
       ``limits.role_contract_bytes``, canonical relative paths, prefix
       entries ending in ``/``, unique tool paths, and permanent notes
       and reference files that live beside the contract with the
       expected suffixes.
    3. The compiled safety floor: the role permission matrix, the
       candidate and landing identities, the advisory Red Team rule,
       the backlog owner and path, the worktree topology, and the
       guard-file, role-file, and trusted-tool censuses must equal the
       bootstrap values compiled into this module, and the minimum
       forbidden files and prefixes must still be present. An edited
       YAML can therefore tighten configuration but never grant
       authority; widening one of these boundaries requires a reviewed
       change to this reader itself.

    Arguments:
      value = the parsed contract mapping: a plain ``dict`` whose
              entries are the sections named above.

    Returns:
      ``value`` unchanged on success, so a caller can write
      ``contract = validate_role_contract(parsed)``.

    Raises:
      RoleContractError on the first violated rule, naming the dotted
      address of the failing entry.
    """
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
    """Read, parse, and validate one role contract from a regular file.

    The read itself is defensive, in this order:

    1. ``lstat`` reports what the path itself is without following a
       symbolic link (a small file that redirects to another path).
       Anything but an ordinary file is refused, so the contract
       cannot be swapped for a redirect to a file elsewhere.
    2. The compiled byte cap is enforced twice, on the reported size
       before reading and on the bytes actually read, because a file
       can grow between the two looks.
    3. The bytes must decode as strict UTF-8, and parsing uses the
       duplicate-key hook ``_object``, so no malformed or two-faced
       text survives to validation.
    4. ``validate_role_contract`` applies the shape, value, and
       safety-floor rules.
    5. Two final checks bind the file to one spelling: its size must
       also respect the contract's own configured
       ``limits.role_contract_bytes``, and re-serializing the parsed
       value with ``json.dumps(value, sort_keys=True, indent=2)`` plus
       one newline must reproduce the file byte for byte. A
       meaning-preserving edit, such as reordered keys or changed
       whitespace, therefore still fails, which keeps exactly one
       accepted spelling for every contract state.

    Arguments:
      path = the contract file to read. None selects
             ``ai/notes/role-contract.yaml`` inside the repository
             that contains this reader, resolved from this file's own
             location, so a tool always reads the contract beside its
             own code.

    Returns:
      The validated contract mapping.

    Raises:
      RoleContractError when any guarantee above fails. A missing file
      surfaces as the ``OSError`` from ``lstat`` instead, because
      there is no contract state to report about.
    """
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
