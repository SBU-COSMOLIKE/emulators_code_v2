"""Role worktree state, provisioning, bootstrap, and cleanup.

Each role works in its own Git worktree: an extra project folder on
its own branch, so an agent edits without touching the user's
checkout. This file creates or safely adopts those folders, records
each folder's saved identity (path, branch, commit) in small state
files, validates every identity before a dispatch, and carries the
Architect's tracked backlog forward when the primary folder advances.
It owns ``--clean-all``, the explicit command that discards every AI
worktree and branch. It also loads the machine role contract and
refuses to run when the shipped trusted tools do not match it.
Docstrings here call a note-only permanent-note landing P and its base
main commit B.

This file is one part of the mailbox daemon and holds definitions only.
``mailbox_daemon.py`` loads it from its own directory, binds the name
``daemon`` below to a live view of its own namespace, and adopts every name
in PART_EXPORTS into that namespace. Code here reaches every constant,
standard-library module, and collaborator through ``daemon.<name>``, so the
daemon keeps one shared namespace no matter how many files store its source.
"""


# Bound by mailbox_daemon.py to a live view of the daemon namespace
# before any of these definitions can run.
daemon = None

PART_EXPORTS = (
    "repo_root_of",
    "candidate_forbidden_files_from_contract",
    "control_plane_files_from_contract",
    "_local_role_contract_tool",
    "validate_role_contract_bindings",
    "role_contract_snapshot_problem",
    "report_role_contract_restart",
    "role_contract_exit_status",
    "_raise_walk_error",
    "primary_state_paths",
    "sol_state_paths",
    "implementer_state_paths",
    "_plain_directory",
    "_require_directory_identity",
    "_managed_primary_root",
    "_run_git",
    "git_common_directory",
    "registered_worktrees",
    "_duplicate_key_refusal",
    "load_primary_state",
    "_path_key",
    "_record_at_path",
    "_managed_child_path",
    "_validate_primary_record",
    "_atomic_write_primary_state",
    "validate_primary_state",
    "_transport_evidence_at_notes",
    "coordination_transport_evidence",
    "_primary_state_for_record",
    "_archived_transport_manifest",
    "_safe_main_archive_bridge",
    "_open_legacy_transport_lock",
    "_release_legacy_transport_lock",
    "_ensure_plain_relative_directory",
    "_regular_files_equal",
    "_copy_regular_archive_file",
    "_remove_archive_copy_temporaries",
    "_publish_primary_record",
    "_publish_adopted_primary_record",
    "_format_evidence",
    "_branch_exists",
    "provision_or_adopt_primary",
    "_upgrade_primary_topology_state",
    "_require_primary_daemon_topology_support",
    "_bootstrap_ticket_state",
    "_bootstrap_candidate_state",
    "_bootstrap_root_ticket_authority",
    "_bootstrap_primary_ahead_notes_authority",
    "clean_user_main_matches",
    "_merge_primary_backlog",
    "_saved_backlog_digest",
    "_reseal_recovered_backlog",
    "_prepare_primary_backlog_overlay",
    "bootstrap_sync_primary_from_main_authority",
    "validated_primary_notes",
    "validate_authoritative_role_files",
    "recheck_authoritative_role_files",
    "_sol_state_for_record",
    "validate_sol_state",
    "provision_or_reuse_sol",
    "_implementer_state_for_record",
    "validate_implementer_state",
    "_tracked_worktree_changes",
    "_optional_ref_commit",
    "implementer_authority_snapshot",
    "implementer_authority_changes",
    "provision_or_reuse_implementer",
    "validate_distinct_agent_states",
    "_open_primary_lock",
    "_release_primary_lock",
    "_is_ai_branch",
    "_lock_cleanup_transport",
    "clean_all_ai_worktrees",
)


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
    worktree = daemon.os.path.abspath(worktree)
    dot_git = daemon.os.path.join(worktree, ".git")
    try:
        dot_git_info = daemon.os.lstat(dot_git)
    except OSError:
        dot_git_info = None
    if dot_git_info is not None and daemon.stat.S_ISDIR(dot_git_info.st_mode):
        return worktree
    if (dot_git_info is not None and daemon.stat.S_ISREG(dot_git_info.st_mode)
            and dot_git_info.st_size <= 4096):
        try:
            with open(dot_git, "r", encoding="utf-8") as stream:
                git_line = stream.read(4097).strip()
            if git_line.startswith("gitdir: "):
                git_directory = git_line[len("gitdir: "):]
                if not daemon.os.path.isabs(git_directory):
                    git_directory = daemon.os.path.join(worktree, git_directory)
                git_directory = daemon.os.path.realpath(git_directory)
                common_file = daemon.os.path.join(git_directory, "commondir")
                common_info = daemon.os.lstat(common_file)
                if (daemon.stat.S_ISREG(common_info.st_mode)
                        and common_info.st_size <= 4096):
                    with open(common_file, "r", encoding="utf-8") as stream:
                        common = stream.read(4097).strip()
                    if not daemon.os.path.isabs(common):
                        common = daemon.os.path.join(git_directory, common)
                    common = daemon.os.path.realpath(common)
                    if daemon.os.path.basename(common) == ".git":
                        return daemon.os.path.dirname(common)
        except (OSError, UnicodeError):
            pass

    worktrees_dir = daemon.os.path.dirname(worktree)          # <repo>/.claude/worktrees
    claude_dir = daemon.os.path.dirname(worktrees_dir)        # <repo>/.claude
    if (daemon.os.path.basename(worktrees_dir) == "worktrees"
            and daemon.os.path.basename(claude_dir) == ".claude"):
        return daemon.os.path.dirname(claude_dir)
    return worktree


def candidate_forbidden_files_from_contract(contract):
    """Return paths that no Implementer candidate may change.

    Arguments:
      contract = the validated role contract.

    Returns:
      Frozen set combining the contract's candidate-forbidden files,
      permanent notes, protected reference files, role files, and the
      contract itself.
    """
    protected = contract["protected_paths"]
    return frozenset(
        protected["candidate_forbidden_files"]
        + protected["permanent_notes"]
        + protected["protected_reference_files"]
        + protected["role_files"]
        + [protected["contract"]])


def control_plane_files_from_contract(contract):
    """Return the historical tool set needed to refuse old saved work.

    Arguments:
      contract = the validated role contract.

    Returns:
      Frozen set of the guard files and trusted tools; saved work
      touching them is refused unless its class unlocked them.
    """
    protected = contract["protected_paths"]
    return frozenset(
        list(protected["guard_files"].values())
        + list(protected["trusted_tools"].values())
    )


def _local_role_contract_tool():
    """Load the contract reader beside this daemon, never from another tree."""
    return daemon._ROLE_CONTRACT_TOOL


def validate_role_contract_bindings(contract=None):
    """Validate one policy snapshot and its non-configurable safety floor.

    With no argument, the contract on disk is reloaded and must equal
    the one this process started with; a supplied contract is
    validated on its own. Either way the worktree policy, cleanup
    branch prefixes, and mailbox-protection prefixes must match the
    running configuration exactly.

    Arguments:
      contract = a contract mapping to validate, or ``None`` to
                 recheck the on-disk file against startup policy.

    Returns:
      The validated contract.

    Raises:
      RoleContractError: for a changed, invalid, or mismatched
        contract.
    """
    tool = daemon._local_role_contract_tool()
    if contract is None:
        contract = tool.load_role_contract(
            daemon.os.path.join(daemon.WORKTREE, daemon.ROLE_CONTRACT_RELATIVE_PATH))
        if contract != daemon.ROLE_CONTRACT:
            raise tool.RoleContractError(
                "role contract changed after daemon startup; restart before "
                "admitting more work")
    else:
        tool.validate_role_contract(contract)

    protected = contract["protected_paths"]
    worktrees = contract["worktrees"]
    if worktrees != daemon.ROLE_CONTRACT["worktrees"]:
        raise tool.RoleContractError(
            "worktree policy changes require an explicit saved-state "
            "migration")
    expected_branch_refs = tuple(
        "refs/heads/" + prefix for prefix in (
            worktrees["claude_branch_prefix"],
            worktrees["sol_branch_prefix"],
            worktrees["legacy_cleanup_prefix"]))
    if tuple(daemon.AI_BRANCH_PREFIXES) != expected_branch_refs:
        raise tool.RoleContractError(
            "cleanup branch prefixes disagree with the role contract")
    runtime_transport = {
        daemon.os.path.relpath(path, daemon.WORKTREE).replace(daemon.os.sep, "/") + "/"
        for path in (daemon.MAILBOX, daemon.RELAY_DIR)}
    if not runtime_transport.issubset(
            set(protected["candidate_forbidden_prefixes"])):
        raise tool.RoleContractError(
            "mailbox paths are not protected from candidates")
    return contract


def role_contract_snapshot_problem():
    """Describe a contract edit made after this process loaded its policy."""
    try:
        current = daemon._local_role_contract_tool().load_role_contract(
            daemon.os.path.join(daemon.WORKTREE, daemon.ROLE_CONTRACT_RELATIVE_PATH))
    except (OSError, RuntimeError, ValueError) as exc:
        return "role contract on disk is invalid: " + str(exc)
    if current != daemon.ROLE_CONTRACT:
        return ("role contract changed after daemon startup; restart the "
                "watcher before admitting more work")
    return None


def report_role_contract_restart():
    """Print the current policy stop and return its process exit status."""
    problem = daemon.role_contract_snapshot_problem()
    if problem is None:
        problem = ("role contract changed during this mailbox pass; restart "
                   "before admitting more work")
    print(problem + ".")
    return 1 if problem.startswith("role contract on disk is invalid") else 0


def role_contract_exit_status():
    """Return a watch exit code unless an exact policy landing may finish."""
    problem = daemon.role_contract_snapshot_problem()
    if problem is None:
        return None
    invalid = problem.startswith("role contract on disk is invalid")
    if not invalid and daemon.architect_notes_transition_pending():
        return None
    return daemon.report_role_contract_restart()


def _raise_walk_error(error):
    """Make ``os.walk`` traversal failures explicit instead of suppressing.

    Arguments:
      error = the traversal error to re-raise.
    """
    raise error


def primary_state_paths(repository_root):
    """Return every deterministic path used by primary-worktree bootstrap.

    Arguments:
      repository_root = the repository's main checkout.

    Returns:
      Mapping with the managed root, the saved state and lock files,
      and the default worktree path and branch for the Architect.
    """
    repository = daemon.os.path.abspath(repository_root)
    managed_root = daemon.os.path.join(repository, ".claude", "worktrees")
    return {
        "managed_root": managed_root,
        "state": daemon.os.path.join(managed_root, daemon.PRIMARY_STATE_NAME),
        "lock": daemon.os.path.join(managed_root, daemon.PRIMARY_LOCK_NAME),
        "default_path": daemon.os.path.join(managed_root, daemon.PRIMARY_WORKTREE_NAME),
        "default_branch": daemon.PRIMARY_BRANCH,
    }


def sol_state_paths(repository_root):
    """Return every deterministic path used by Sol-worktree bootstrap.

    Arguments:
      repository_root = the repository's main checkout.

    Returns:
      Mapping with the managed root, the saved state file, and the
      default worktree path and branch for the Red Team.
    """
    repository = daemon.os.path.abspath(repository_root)
    managed_root = daemon.os.path.join(repository, ".claude", "worktrees")
    return {
        "managed_root": managed_root,
        "state": daemon.os.path.join(managed_root, daemon.SOL_STATE_NAME),
        "default_path": daemon.os.path.join(managed_root, daemon.SOL_WORKTREE_NAME),
        "default_branch": daemon.SOL_BRANCH,
    }


def implementer_state_paths(repository_root):
    """Return deterministic paths used by Implementer bootstrap.

    Arguments:
      repository_root = the repository's main checkout.

    Returns:
      Mapping with the managed root, the saved state file, and the
      default worktree path and branch for the Implementer.
    """
    repository = daemon.os.path.abspath(repository_root)
    managed_root = daemon.os.path.join(repository, ".claude", "worktrees")
    return {
        "managed_root": managed_root,
        "state": daemon.os.path.join(managed_root, daemon.IMPLEMENTER_STATE_NAME),
        "default_path": daemon.os.path.join(
            managed_root, daemon.IMPLEMENTER_WORKTREE_NAME),
        "default_branch": daemon.IMPLEMENTER_BRANCH,
    }


def _plain_directory(path, label, create=False):
    """Prove that ``path`` is one ordinary directory, optionally creating it.

    Arguments:
      path   = the directory to prove.
      label  = name used in error messages.
      create = True to create a missing directory with owner-only
               permissions.

    Returns:
      The directory's ``(device, inode)`` identity.

    Raises:
      daemon.PrimaryWorktreeError: for a missing, uncreatable, or
        non-directory path, or a symbolic link.
    """
    if not daemon.os.path.lexists(path):
        if not create:
            raise daemon.PrimaryWorktreeError(label + " does not exist: " + path)
        try:
            daemon.os.mkdir(path, 0o700)
        except FileExistsError:
            # Two clean-clone first runs can both observe the absent managed
            # root before either opens the shared bootstrap lock. The winner's
            # ordinary directory is accepted by the lstat proof below.
            pass
        except OSError as exc:
            raise daemon.PrimaryWorktreeError(
                "cannot create " + label + " " + path + ": " + str(exc))
    try:
        info = daemon.os.lstat(path)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot inspect " + label + " " + path + ": " + str(exc))
    if daemon.stat.S_ISLNK(info.st_mode) or not daemon.stat.S_ISDIR(info.st_mode):
        raise daemon.PrimaryWorktreeError(
            label + " must be a real directory, not a redirect: " + path)
    return (info.st_dev, info.st_ino)


def _require_directory_identity(path, identity, label):
    """Prove a locked directory pathname still names its original inode.

    Arguments:
      path     = the directory to revalidate.
      identity = the ``(device, inode)`` it must still have.
      label    = name used in error messages.

    Raises:
      daemon.PrimaryWorktreeError: when the path changed type or
        identity.
    """
    try:
        info = daemon.os.lstat(path)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot revalidate " + label + " " + path + ": " + str(exc))
    if (daemon.stat.S_ISLNK(info.st_mode) or not daemon.stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino) != identity):
        raise daemon.PrimaryWorktreeError(
            label + " changed while primary state was being prepared: "
            + path)


def _managed_primary_root(repository_root, create=False):
    """Return the non-symlinked repo-local worktree container.

    Arguments:
      repository_root = the repository's main checkout; it must not
                        be reached through a symbolic link.
      create          = True to create the managed folder if missing.

    Returns:
      The ``.claude/worktrees`` path after every level proves to be a
      plain directory.

    Raises:
      daemon.PrimaryWorktreeError: for a symlinked root or an unsafe
        component.
    """
    repository = daemon.os.path.abspath(repository_root)
    if daemon.os.path.realpath(repository) != repository:
        raise daemon.PrimaryWorktreeError(
            "repository root must not be reached through a symlink: "
            + repository)
    daemon._plain_directory(path=repository, label="repository root")
    claude_root = daemon.os.path.join(repository, ".claude")
    daemon._plain_directory(path=claude_root, label=".claude directory")
    managed_root = daemon.os.path.join(claude_root, "worktrees")
    daemon._plain_directory(path=managed_root, label="managed worktree directory",
                     create=create)
    return managed_root


def _run_git(repository_root, arguments, check=True, input_bytes=None):
    """Run one argv-only Git command and return its completed process.

    Arguments:
      repository_root = folder passed to ``git -C``.
      arguments       = Git subcommand and options.
      check           = True to raise on a nonzero exit.
      input_bytes     = bytes fed to standard input, or ``None``.

    Returns:
      The completed-process object with raw output bytes.

    Raises:
      daemon.PrimaryWorktreeError: when Git cannot start, or fails
        while ``check`` is set.
    """
    command = ["git", "-C", daemon.os.path.abspath(repository_root)] + list(arguments)
    try:
        result = daemon.subprocess.run(command, stdout=daemon.subprocess.PIPE,
                                stderr=daemon.subprocess.PIPE, check=False,
                                input=input_bytes)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError("cannot run git: " + str(exc))
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if len(detail) > 1000:
            detail = detail[:1000] + "..."
        if detail:
            detail = ": " + detail
        raise daemon.PrimaryWorktreeError(
            "git " + " ".join(arguments) + " failed" + detail)
    return result


def git_common_directory(checkout):
    """Return the canonical Git common directory owning ``checkout``.

    Linked worktrees share one common Git directory; comparing it
    identifies which repository a checkout belongs to.

    Arguments:
      checkout = the checkout folder.

    Returns:
      The resolved common directory path.

    Raises:
      daemon.PrimaryWorktreeError: for an empty or non-UTF-8 answer.
    """
    result = daemon._run_git(repository_root=checkout,
                      arguments=["rev-parse", "--git-common-dir"])
    try:
        value = result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError(
            "git common-directory output is not UTF-8: " + str(exc))
    if not value:
        raise daemon.PrimaryWorktreeError("git returned an empty common directory")
    if not daemon.os.path.isabs(value):
        value = daemon.os.path.join(daemon.os.path.abspath(checkout), value)
    return daemon.os.path.realpath(value)


def registered_worktrees(repository_root):
    """Parse ``git worktree list --porcelain -z`` without path ambiguity.

    The NUL-separated form is parsed strictly: every record starts
    with its path, HEAD and branch may appear once each, and any
    other field becomes a flag.

    Arguments:
      repository_root = the repository to list.

    Returns:
      List of records with ``path``, ``flags``, and optional ``HEAD``
      and ``branch``.

    Raises:
      daemon.PrimaryWorktreeError: for a malformed or empty registry.
    """
    result = daemon._run_git(
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
                    raise daemon.PrimaryWorktreeError(
                        "malformed git worktree registry")
                record = {"path": daemon.os.path.abspath(value), "flags": set()}
                continue
            if record is None:
                raise daemon.PrimaryWorktreeError(
                    "git worktree registry field precedes worktree path")
            if key in {"HEAD", "branch"}:
                if not separator or key in record:
                    raise daemon.PrimaryWorktreeError(
                        "duplicate or malformed worktree " + key + " field")
                record[key] = value
            else:
                record["flags"].add(key)
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError(
            "git worktree registry is not UTF-8: " + str(exc))
    if record is not None:
        records.append(record)
    if not records:
        raise daemon.PrimaryWorktreeError("git reports no registered worktrees")
    return records


def _duplicate_key_refusal(pairs):
    """JSON object hook which rejects duplicate state keys.

    Arguments:
      pairs = decoded key-value pairs in document order.

    Returns:
      The mapping.

    Raises:
      daemon.PrimaryWorktreeError: for a duplicate key, which
        ordinary JSON parsing would silently collapse.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise daemon.PrimaryWorktreeError(
                "primary-worktree state repeats key " + repr(key))
        result[key] = value
    return result


def load_primary_state(path):
    """Read one bounded, regular, exact-schema primary-worktree record.

    Arguments:
      path = the saved state file.

    Returns:
      The decoded state mapping after every schema rule passes.

    Raises:
      daemon.PrimaryWorktreeError: for an unsafe read, invalid JSON,
        or a record that fails the schema.
    """
    try:
        initial = daemon.os.lstat(path)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot inspect primary-worktree state " + path + ": " + str(exc))
    if daemon.stat.S_ISLNK(initial.st_mode) or not daemon.stat.S_ISREG(initial.st_mode):
        raise daemon.PrimaryWorktreeError(
            "primary-worktree state is not a regular file: " + path)
    flags = daemon.os.O_RDONLY | daemon.os.O_NONBLOCK
    if hasattr(daemon.os, "O_NOFOLLOW"):
        flags |= daemon.os.O_NOFOLLOW
    try:
        descriptor = daemon.os.open(path, flags)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot open primary-worktree state " + path + ": " + str(exc))
    try:
        before = daemon.os.fstat(descriptor)
        if not daemon.stat.S_ISREG(before.st_mode):
            raise daemon.PrimaryWorktreeError(
                "primary-worktree state is not a regular file: " + path)
        if before.st_size > daemon.MAX_PRIMARY_STATE_BYTES:
            raise daemon.PrimaryWorktreeError(
                "primary-worktree state exceeds "
                + str(daemon.MAX_PRIMARY_STATE_BYTES) + " bytes: " + path)
        payload = daemon.os.read(descriptor, daemon.MAX_PRIMARY_STATE_BYTES + 1)
        after = daemon.os.fstat(descriptor)
        current = daemon.os.lstat(path)
        if ((initial.st_dev, initial.st_ino) != (before.st_dev, before.st_ino)
                or (before.st_dev, before.st_ino)
                != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino)
                != (current.st_dev, current.st_ino)
                or after.st_size != len(payload)):
            raise daemon.PrimaryWorktreeError(
                "primary-worktree state changed while being read: " + path)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot read primary-worktree state " + path + ": " + str(exc))
    finally:
        daemon.os.close(descriptor)
    if len(payload) > daemon.MAX_PRIMARY_STATE_BYTES:
        raise daemon.PrimaryWorktreeError(
            "primary-worktree state exceeds "
            + str(daemon.MAX_PRIMARY_STATE_BYTES) + " bytes: " + path)
    try:
        text = payload.decode("utf-8", errors="strict")
        state = daemon.json.loads(text, object_pairs_hook=daemon._duplicate_key_refusal)
    except (UnicodeDecodeError, daemon.json.JSONDecodeError) as exc:
        raise daemon.PrimaryWorktreeError(
            "primary-worktree state is not exact UTF-8 JSON: " + str(exc))
    if not isinstance(state, dict):
        raise daemon.PrimaryWorktreeError("primary-worktree state must be an object")
    base_keys = {"schema", "repository", "name", "path", "branch"}
    if type(state.get("schema")) is not int:
        raise daemon.PrimaryWorktreeError(
            "unsupported primary-worktree state schema")
    schema = state["schema"]
    if schema == daemon.LEGACY_PRIMARY_STATE_SCHEMA:
        expected = base_keys
    elif schema == daemon.PREVIOUS_PRIMARY_STATE_SCHEMA:
        expected = base_keys | {"topology"}
    elif schema == daemon.PRIMARY_STATE_SCHEMA:
        expected = base_keys | {"topology"}
    else:
        raise daemon.PrimaryWorktreeError(
            "unsupported primary-worktree state schema")
    if set(state) != expected:
        raise daemon.PrimaryWorktreeError(
            "primary-worktree state keys must be exactly "
            + ", ".join(sorted(expected)))
    for key in ("repository", "name", "path", "branch"):
        value = state[key]
        if (not isinstance(value, str) or not value or "\x00" in value
                or "\n" in value or "\r" in value):
            raise daemon.PrimaryWorktreeError(
                "invalid primary-worktree state field " + key)
    if not daemon.os.path.isabs(state["repository"]):
        raise daemon.PrimaryWorktreeError("state repository must be absolute")
    if not daemon.os.path.isabs(state["path"]):
        raise daemon.PrimaryWorktreeError("state path must be absolute")
    if (state["name"] != daemon.os.path.basename(state["path"])
            or state["name"] in {".", ".."}
            or "/" in state["name"]):
        raise daemon.PrimaryWorktreeError("state name must equal the path basename")
    if not state["branch"].startswith("refs/heads/"):
        raise daemon.PrimaryWorktreeError("state branch must be an attached head ref")
    expected_topology = {
        daemon.PREVIOUS_PRIMARY_STATE_SCHEMA: daemon.PREVIOUS_PRIMARY_TOPOLOGY_MARKER,
        daemon.PRIMARY_STATE_SCHEMA: daemon.PRIMARY_TOPOLOGY_MARKER,
    }.get(schema)
    if (expected_topology is not None
            and state["topology"] != expected_topology):
        raise daemon.PrimaryWorktreeError(
            "primary-worktree topology marker is unsupported")
    return state


def _path_key(path):
    """Return a stable lexical comparison key for a registered worktree.

    Arguments:
      path = the path to normalize.

    Returns:
      The absolute, case-normalized path used for comparisons.
    """
    return daemon.os.path.normcase(daemon.os.path.abspath(path))


def _record_at_path(records, path):
    """Return the unique registry record at ``path``, or ``None``.

    Arguments:
      records = the parsed worktree registry.
      path    = the path to look up.

    Returns:
      The single matching record, or ``None``.

    Raises:
      daemon.PrimaryWorktreeError: when the registry reports the
        path more than once.
    """
    matches = [record for record in records
               if daemon._path_key(record["path"]) == daemon._path_key(path)]
    if len(matches) > 1:
        raise daemon.PrimaryWorktreeError(
            "git reports the worktree path more than once: " + path)
    return matches[0] if matches else None


def _managed_child_path(path, managed_root):
    """Prove ``path`` is one direct, non-symlinked managed child.

    Arguments:
      path         = the candidate worktree path.
      managed_root = the managed container it must sit directly in.

    Returns:
      The validated absolute path.

    Raises:
      daemon.PrimaryWorktreeError: for a nested, redirected, or
        non-directory path.
    """
    candidate = daemon.os.path.abspath(path)
    if daemon.os.path.dirname(candidate) != daemon.os.path.abspath(managed_root):
        raise daemon.PrimaryWorktreeError(
            "primary worktree must be a direct child of " + managed_root
            + ": " + candidate)
    try:
        info = daemon.os.lstat(candidate)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot inspect primary worktree " + candidate + ": " + str(exc))
    if daemon.stat.S_ISLNK(info.st_mode) or not daemon.stat.S_ISDIR(info.st_mode):
        raise daemon.PrimaryWorktreeError(
            "primary worktree must be a real directory: " + candidate)
    if (daemon.os.path.dirname(daemon.os.path.realpath(candidate))
            != daemon.os.path.realpath(
            managed_root)):
        raise daemon.PrimaryWorktreeError(
            "primary worktree escapes the managed directory: " + candidate)
    return candidate


def _validate_primary_record(record, branch, repository_root):
    """Prove a registry record, checkout, branch, and daemon all agree.

    Arguments:
      record          = the worktree registry record.
      branch          = the branch reference the checkout must be on.
      repository_root = the repository's main checkout.

    Raises:
      daemon.PrimaryWorktreeError: for a prunable, detached,
        wrong-branch, or foreign record.
    """
    managed_root = daemon._managed_primary_root(repository_root=repository_root)
    if "prunable" in record["flags"]:
        raise daemon.PrimaryWorktreeError(
            "primary worktree is prunable: " + record["path"])
    if "detached" in record["flags"] or "branch" not in record:
        raise daemon.PrimaryWorktreeError(
            "primary worktree must have an attached branch: " + record["path"])
    if record["branch"] != branch:
        raise daemon.PrimaryWorktreeError(
            "primary branch mismatch at " + record["path"] + ": expected "
            + branch + ", found " + record["branch"])
    path = daemon._managed_child_path(path=record["path"],
                               managed_root=managed_root)
    top = daemon._run_git(repository_root=path,
                   arguments=["rev-parse", "--show-toplevel"])
    try:
        top_path = top.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError(
            "worktree top-level output is not UTF-8: " + str(exc))
    if daemon.os.path.realpath(top_path) != daemon.os.path.realpath(path):
        raise daemon.PrimaryWorktreeError(
            "registered primary top level does not match its path: " + path)
    repository = daemon.git_common_directory(checkout=repository_root)
    if daemon.git_common_directory(checkout=path) != repository:
        raise daemon.PrimaryWorktreeError(
            "primary worktree belongs to a different repository: " + path)
    symbolic = daemon._run_git(repository_root=path,
                        arguments=["symbolic-ref", "-q", "HEAD"])
    try:
        symbolic_branch = symbolic.stdout.decode(
            "utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError(
            "primary branch output is not UTF-8: " + str(exc))
    if symbolic_branch != branch:
        raise daemon.PrimaryWorktreeError(
            "checked-out primary branch does not match state: " + path)
    daemon_path = daemon.os.path.join(path, "ai", "tools", "mailbox_daemon.py")
    try:
        daemon_info = daemon.os.lstat(daemon_path)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "primary daemon is missing at " + daemon_path + ": " + str(exc))
    if (daemon.stat.S_ISLNK(daemon_info.st_mode)
            or not daemon.stat.S_ISREG(daemon_info.st_mode)):
        raise daemon.PrimaryWorktreeError(
            "primary daemon must be a regular non-symlink file: "
            + daemon_path)
    return path


def _atomic_write_primary_state(state, path):
    """Publish primary authority by fsync + same-directory atomic replace.

    Arguments:
      state = the JSON-compatible state mapping.
      path  = the destination state file.

    Raises:
      daemon.PrimaryWorktreeError: when the managed directory is
        unsafe.
    """
    directory = daemon.os.path.dirname(path)
    daemon._plain_directory(path=directory, label="managed worktree directory")
    payload = (daemon.json.dumps(state, sort_keys=True, indent=2) + "\n").encode(
        "utf-8")
    descriptor, temporary = daemon.tempfile.mkstemp(
        prefix=daemon.os.path.basename(path) + ".tmp-", dir=directory)
    try:
        daemon.os.fchmod(descriptor, 0o600)
        with daemon.os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            daemon.os.fsync(stream.fileno())
        daemon.os.replace(temporary, path)
        directory_flags = daemon.os.O_RDONLY
        if hasattr(daemon.os, "O_DIRECTORY"):
            directory_flags |= daemon.os.O_DIRECTORY
        directory_descriptor = daemon.os.open(directory, directory_flags)
        try:
            daemon.os.fsync(directory_descriptor)
        finally:
            daemon.os.close(directory_descriptor)
    except BaseException:
        if descriptor >= 0:
            daemon.os.close(descriptor)
        try:
            daemon.os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_primary_state(state, repository_root, allow_move=False,
                           state_path=None):
    """Validate persisted authority; accept only a Git-authorized move.

    The saved record must name this repository and a registered
    managed child on the saved branch. When Git's registry shows the
    worktree at a new managed path, the move is adopted only with
    ``allow_move``; anything else is refused.

    Arguments:
      state           = the loaded primary state.
      repository_root = the repository's main checkout.
      allow_move      = True to adopt a Git-recorded relocation and
                        rewrite the saved state.
      state_path      = the state file to rewrite on adoption, or
                        ``None`` for the default.

    Returns:
      The validated (possibly updated) state mapping.

    Raises:
      daemon.PrimaryWorktreeError: for any disagreement between the
        saved record, the registry, and the checkout.
    """
    repository = daemon.git_common_directory(checkout=repository_root)
    if state["repository"] != repository:
        raise daemon.PrimaryWorktreeError(
            "primary-worktree state names a different repository")
    managed_root = daemon._managed_primary_root(repository_root=repository_root)
    stored_path = daemon.os.path.abspath(state["path"])
    if daemon.os.path.dirname(stored_path) != managed_root:
        raise daemon.PrimaryWorktreeError(
            "saved primary path is outside the managed directory: "
            + stored_path)
    records = daemon.registered_worktrees(repository_root=repository_root)
    record = daemon._record_at_path(records=records, path=stored_path)
    resolved = dict(state)
    if record is None:
        branch_matches = [item for item in records
                          if item.get("branch") == state["branch"]]
        if (len(branch_matches) != 1 or daemon.os.path.lexists(stored_path)):
            raise daemon.PrimaryWorktreeError(
                "saved primary path is no longer registered; state was "
                "preserved for manual recovery: " + stored_path)
        moved = branch_matches[0]
        moved_path = daemon._validate_primary_record(
            record=moved, branch=state["branch"],
            repository_root=repository_root)
        resolved["path"] = moved_path
        resolved["name"] = daemon.os.path.basename(moved_path)
        if allow_move:
            if state_path is None:
                state_path = daemon.primary_state_paths(repository_root)["state"]
            daemon._atomic_write_primary_state(
                state=resolved, path=state_path)
            print("primary coordination worktree moved by git; saved "
                  + moved_path, flush=True)
        return resolved
    daemon._validate_primary_record(record=record, branch=state["branch"],
                             repository_root=repository_root)
    return resolved


def _transport_evidence_at_notes(notes, reason_prefix=""):
    """Inspect one current or pre-migration notes root without writing it.

    Arguments:
      notes         = the notes folder to inspect.
      reason_prefix = text prepended to each reported reason.

    Returns:
      List of printable reasons naming the coordination evidence
      found (numbered messages, relay files); empty when none.
    """
    reasons = []
    mailbox = daemon.os.path.join(notes, "mailbox")
    relay = daemon.os.path.join(notes, "relay")
    message_name = daemon.re.compile(r"\d+[a-z]?-to-[^.]+\.md$")

    for label, root in (("mailbox", mailbox), ("relay", relay)):
        if not daemon.os.path.lexists(root):
            continue
        try:
            info = daemon.os.lstat(root)
        except OSError as exc:
            reasons.append(reason_prefix + label
                           + " cannot be inspected: " + str(exc))
            continue
        if daemon.stat.S_ISLNK(info.st_mode) or not daemon.stat.S_ISDIR(info.st_mode):
            reasons.append(reason_prefix + label
                           + " is redirected or irregular")
            continue
        visited = 0
        found = None
        try:
            for directory, names, files in daemon.os.walk(
                    root, followlinks=False, onerror=daemon._raise_walk_error):
                names.sort()
                files.sort()
                for name in list(names):
                    visited += 1
                    if visited > daemon.MAX_PRIMARY_ARCHIVE_ENTRIES:
                        found = (reason_prefix + label
                                 + " evidence scan exceeds "
                                 + str(daemon.MAX_PRIMARY_ARCHIVE_ENTRIES)
                                 + " entries")
                        break
                    entry = daemon.os.path.join(directory, name)
                    entry_info = daemon.os.lstat(entry)
                    if daemon.stat.S_ISLNK(entry_info.st_mode):
                        found = (reason_prefix + label
                                 + " contains a redirected directory")
                        break
                if found is not None:
                    break
                for name in files:
                    visited += 1
                    if visited > daemon.MAX_PRIMARY_ARCHIVE_ENTRIES:
                        found = (reason_prefix + label
                                 + " evidence scan exceeds "
                                 + str(daemon.MAX_PRIMARY_ARCHIVE_ENTRIES)
                                 + " entries")
                        break
                    entry = daemon.os.path.join(directory, name)
                    entry_info = daemon.os.lstat(entry)
                    if (daemon.stat.S_ISLNK(entry_info.st_mode)
                            or not daemon.stat.S_ISREG(entry_info.st_mode)):
                        found = (reason_prefix + label
                                 + " contains an irregular entry")
                        break
                    if label == "relay":
                        found = reason_prefix + "relay evidence exists"
                        break
                    relative = daemon.os.path.relpath(entry, root)
                    if (daemon.os.path.dirname(relative) in {"", "."}
                            and name in daemon.PRIMARY_ARCHIVE_RUNTIME_LOCKS):
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
        lock_path = daemon.os.path.join(mailbox, lock_name)
        if not daemon.os.path.lexists(lock_path):
            continue
        flags = daemon.os.O_RDWR
        if hasattr(daemon.os, "O_NOFOLLOW"):
            flags |= daemon.os.O_NOFOLLOW
        descriptor = None
        try:
            descriptor = daemon.os.open(lock_path, flags)
            opened = daemon.os.fstat(descriptor)
            current = daemon.os.lstat(lock_path)
            if (not daemon.stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (current.st_dev, current.st_ino)):
                reasons.append(reason_prefix
                               + lock_name
                               + " is redirected or irregular")
            else:
                try:
                    daemon.fcntl.flock(
                        descriptor,
                        daemon.fcntl.LOCK_EX | daemon.fcntl.LOCK_NB)
                except BlockingIOError:
                    reasons.append(reason_prefix + held_reason)
                else:
                    daemon.fcntl.flock(descriptor, daemon.fcntl.LOCK_UN)
        except OSError as exc:
            reasons.append(reason_prefix
                           + lock_name + " cannot be inspected: " + str(exc))
        finally:
            if descriptor is not None:
                daemon.os.close(descriptor)
    return reasons


def coordination_transport_evidence(worktree):
    """Return current and pre-``ai/`` coordination evidence in a worktree.

    Arguments:
      worktree = the checkout to inspect.

    Returns:
      Combined reasons from the current ``ai/notes`` root and the
      legacy pre-``ai`` ``notes`` root.
    """
    reasons = daemon._transport_evidence_at_notes(
        notes=daemon.os.path.join(worktree, "ai", "notes"))
    reasons.extend(daemon._transport_evidence_at_notes(
        notes=daemon.os.path.join(worktree, "notes"),
        reason_prefix="legacy pre-ai "))
    return reasons


def _primary_state_for_record(record, repository_root):
    """Build the exact persisted record for one already-validated checkout.

    Arguments:
      record          = the validated worktree registry record.
      repository_root = the repository's main checkout.

    Returns:
      The state mapping with schema, repository identity, name, path,
      branch, and topology marker.
    """
    return {
        "schema": daemon.PRIMARY_STATE_SCHEMA,
        "repository": daemon.git_common_directory(checkout=repository_root),
        "name": daemon.os.path.basename(record["path"]),
        "path": daemon.os.path.abspath(record["path"]),
        "branch": record["branch"],
        "topology": daemon.PRIMARY_TOPOLOGY_MARKER,
    }


def _archived_transport_manifest(worktree):
    """Return copyable archived-only transport, or ``None`` if unsafe.

    A pre-primary installation may have completed messages under ``done/``
    plus relay logs in main. Those immutable archives can be bridged into a
    new primary without guessing queue state. Pending, inflight, failed,
    redirected, irregular, or unrecognized mailbox content is never bridged.
    """
    notes = daemon.os.path.join(worktree, "ai", "notes")
    mailbox = daemon.os.path.join(notes, "mailbox")
    relay = daemon.os.path.join(notes, "relay")
    message_name = daemon.re.compile(r"(\d+)[a-z]?-to-[^.]+\.md$")
    manifest = []
    visited = 0
    total_bytes = 0
    mailbox_sequences = set()

    for label, root in (("mailbox", mailbox), ("relay", relay)):
        if not daemon.os.path.lexists(root):
            continue
        try:
            root_info = daemon.os.lstat(root)
        except OSError:
            return None
        if daemon.stat.S_ISLNK(root_info.st_mode) or not daemon.stat.S_ISDIR(
                root_info.st_mode):
            return None
        try:
            for directory, names, files in daemon.os.walk(
                    root, followlinks=False, onerror=daemon._raise_walk_error):
                names.sort()
                files.sort()
                for name in names:
                    visited += 1
                    if visited > daemon.MAX_PRIMARY_ARCHIVE_ENTRIES:
                        return None
                    entry_info = daemon.os.lstat(daemon.os.path.join(directory, name))
                    if daemon.stat.S_ISLNK(entry_info.st_mode):
                        return None
                for name in files:
                    visited += 1
                    if visited > daemon.MAX_PRIMARY_ARCHIVE_ENTRIES:
                        return None
                    source = daemon.os.path.join(directory, name)
                    entry_info = daemon.os.lstat(source)
                    if (daemon.stat.S_ISLNK(entry_info.st_mode)
                            or not daemon.stat.S_ISREG(entry_info.st_mode)):
                        return None
                    relative = daemon.os.path.relpath(source, root)
                    if label == "mailbox":
                        parts = relative.split(daemon.os.sep)
                        if (len(parts) == 1
                                and name in daemon.PRIMARY_ARCHIVE_RUNTIME_LOCKS):
                            continue
                        match = message_name.fullmatch(name)
                        if (len(parts) < 2 or parts[0] != "done"
                                or match is None):
                            return None
                        sequence = int(match.group(1))
                        if sequence in mailbox_sequences:
                            return None
                        mailbox_sequences.add(sequence)
                    if entry_info.st_size > daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES:
                        return None
                    total_bytes += entry_info.st_size
                    if total_bytes > daemon.MAX_PRIMARY_ARCHIVE_TOTAL_BYTES:
                        return None
                    manifest.append((
                        source, daemon.os.path.join(label, relative),
                        entry_info.st_size, entry_info.st_dev,
                        entry_info.st_ino, entry_info.st_mtime_ns))
        except OSError:
            return None
    return sorted(manifest, key=lambda item: item[1])


def _safe_main_archive_bridge(evidence, repository_root, default_path):
    """Return whether all evidence is one resumable archived-main bridge.

    Arguments:
      evidence        = ``(path, reasons)`` pairs found in candidate
                        checkouts.
      repository_root = the repository's main checkout.
      default_path    = the default primary worktree path.

    Returns:
      True only when every reason is the archived mailbox or relay
      history of the main checkout or the default primary — the one
      state a bootstrap may adopt automatically.
    """
    allowed_paths = {daemon._path_key(repository_root), daemon._path_key(default_path)}
    allowed_reasons = {
        "numbered mailbox history exists",
        "relay evidence exists",
    }
    main_seen = False
    for path, reasons in evidence:
        if daemon._path_key(path) not in allowed_paths:
            return False
        if not reasons or not set(reasons).issubset(allowed_reasons):
            return False
        if daemon._path_key(path) == daemon._path_key(repository_root):
            main_seen = True
    main_manifest = daemon._archived_transport_manifest(worktree=repository_root)
    if not main_seen or main_manifest is None or not main_manifest:
        return False
    if any(daemon._path_key(path) == daemon._path_key(default_path)
           for path, _reasons in evidence):
        default_manifest = daemon._archived_transport_manifest(
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
                    or not daemon._regular_files_equal(
                        first=legacy[0], second=copied_source)):
                return False
    return True


def _open_legacy_transport_lock(path, nonblocking):
    """Open one regular legacy mailbox lock and take exclusive ownership.

    Arguments:
      path        = the lock file.
      nonblocking = True to fail instead of waiting for the holder.

    Returns:
      The open locked descriptor; the caller must release it.

    Raises:
      daemon.PrimaryWorktreeError: for an unopenable or held lock.
    """
    flags = daemon.os.O_RDWR | daemon.os.O_CREAT
    if hasattr(daemon.os, "O_NOFOLLOW"):
        flags |= daemon.os.O_NOFOLLOW
    try:
        descriptor = daemon.os.open(path, flags, 0o600)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot open legacy transport lock " + path + ": " + str(exc))
    try:
        opened = daemon.os.fstat(descriptor)
        current = daemon.os.lstat(path)
        if (not daemon.stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)):
            raise daemon.PrimaryWorktreeError(
                "legacy transport lock is redirected or irregular: " + path)
        operation = daemon.fcntl.LOCK_EX
        if nonblocking:
            operation |= daemon.fcntl.LOCK_NB
        try:
            daemon.fcntl.flock(descriptor, operation)
        except BlockingIOError:
            raise daemon.PrimaryWorktreeError(
                "legacy transport is live; stop its watcher before primary "
                "bootstrap: " + path)
        after = daemon.os.fstat(descriptor)
        current = daemon.os.lstat(path)
        if ((opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino)
                != (current.st_dev, current.st_ino)):
            raise daemon.PrimaryWorktreeError(
                "legacy transport lock changed while waiting: " + path)
        return daemon.os.fdopen(descriptor, "r+", encoding="utf-8")
    except BaseException:
        daemon.os.close(descriptor)
        raise


def _release_legacy_transport_lock(lock_file):
    """Release one legacy bridge lock.

    Arguments:
      lock_file = the open locked file from the open call.
    """
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def _ensure_plain_relative_directory(root, relative):
    """Create a relative directory tree without accepting any redirect.

    Arguments:
      root     = the proven bridge root.
      relative = the relative folder chain to create component by
                 component, each proven a plain directory.

    Returns:
      The final directory path.

    Raises:
      daemon.PrimaryWorktreeError: for a redirected or non-directory
        component.
    """
    daemon._plain_directory(path=root, label="archive bridge root")
    current = root
    if not relative or relative == ".":
        return current
    for component in relative.split(daemon.os.sep):
        if component in {"", ".", ".."}:
            raise daemon.PrimaryWorktreeError(
                "invalid archive bridge directory component")
        current = daemon.os.path.join(current, component)
        daemon._plain_directory(path=current, label="archive bridge directory",
                         create=True)
    return current


def _regular_files_equal(first, second):
    """Compare two regular non-symlink files without following replacements.

    Arguments:
      first  = one file path.
      second = the other file path.

    Returns:
      True when both are regular files with identical bytes.

    Raises:
      daemon.PrimaryWorktreeError: for an unreadable or non-regular
        file.
    """
    descriptors = []
    flags = daemon.os.O_RDONLY | daemon.os.O_NONBLOCK
    if hasattr(daemon.os, "O_NOFOLLOW"):
        flags |= daemon.os.O_NOFOLLOW
    try:
        for path in (first, second):
            initial = daemon.os.lstat(path)
            if (daemon.stat.S_ISLNK(initial.st_mode)
                    or not daemon.stat.S_ISREG(initial.st_mode)):
                return False
            descriptor = daemon.os.open(path, flags)
            opened = daemon.os.fstat(descriptor)
            if ((initial.st_dev, initial.st_ino)
                    != (opened.st_dev, opened.st_ino)
                    or not daemon.stat.S_ISREG(opened.st_mode)):
                daemon.os.close(descriptor)
                return False
            descriptors.append((descriptor, opened, path))
        if descriptors[0][1].st_size != descriptors[1][1].st_size:
            return False
        while True:
            left = daemon.os.read(descriptors[0][0], 1048576)
            right = daemon.os.read(descriptors[1][0], 1048576)
            if left != right:
                return False
            if not left:
                break
        for descriptor, opened, path in descriptors:
            after = daemon.os.fstat(descriptor)
            current = daemon.os.lstat(path)
            if ((opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                    or (after.st_dev, after.st_ino)
                    != (current.st_dev, current.st_ino)):
                return False
        return True
    except OSError:
        return False
    finally:
        for descriptor, _opened, _path in descriptors:
            daemon.os.close(descriptor)


def _copy_regular_archive_file(source, destination, expected_size):
    """Idempotently publish one exact archive copy without overwriting.

    Rerunning the copy is harmless: an existing byte-identical
    destination is accepted, while any other existing file refuses.

    Arguments:
      source        = the archive file to copy.
      destination   = the destination path; never overwritten.
      expected_size = the source's expected size, bounded.

    Raises:
      daemon.PrimaryWorktreeError: for an oversized source, a
        conflicting destination, or a failed copy.
    """
    parent = daemon.os.path.dirname(destination)
    if (expected_size < 0
            or expected_size > daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES):
        raise daemon.PrimaryWorktreeError(
            "legacy archive exceeds the bounded copy size: " + source)
    if daemon.os.path.lexists(destination):
        if not daemon._regular_files_equal(first=source, second=destination):
            raise daemon.PrimaryWorktreeError(
                "archive bridge destination conflicts with legacy bytes: "
                + destination)
        return
    source_flags = daemon.os.O_RDONLY | daemon.os.O_NONBLOCK
    if hasattr(daemon.os, "O_NOFOLLOW"):
        source_flags |= daemon.os.O_NOFOLLOW
    try:
        source_initial = daemon.os.lstat(source)
        if (daemon.stat.S_ISLNK(source_initial.st_mode)
                or not daemon.stat.S_ISREG(source_initial.st_mode)):
            raise daemon.PrimaryWorktreeError(
                "legacy archive source is not a regular file: " + source)
        source_descriptor = daemon.os.open(source, source_flags)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot open legacy archive " + source + ": " + str(exc))
    temporary_descriptor = -1
    temporary = None
    try:
        source_opened = daemon.os.fstat(source_descriptor)
        if ((source_initial.st_dev, source_initial.st_ino)
                != (source_opened.st_dev, source_opened.st_ino)
                or source_opened.st_size != expected_size):
            raise daemon.PrimaryWorktreeError(
                "legacy archive changed before copy: " + source)
        temporary_descriptor, temporary = daemon.tempfile.mkstemp(
            prefix=".primary-archive-", dir=parent)
        daemon.os.fchmod(temporary_descriptor, source_opened.st_mode & 0o777)
        copied = 0
        while copied < expected_size:
            chunk = daemon.os.read(
                source_descriptor, min(1048576, expected_size - copied))
            if not chunk:
                raise daemon.PrimaryWorktreeError(
                    "legacy archive shortened during copy: " + source)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = daemon.os.write(temporary_descriptor, view)
                view = view[written:]
        if daemon.os.read(source_descriptor, 1):
            raise daemon.PrimaryWorktreeError(
                "legacy archive grew during copy: " + source)
        daemon.os.fsync(temporary_descriptor)
        source_after = daemon.os.fstat(source_descriptor)
        source_current = daemon.os.lstat(source)
        if ((source_opened.st_dev, source_opened.st_ino)
                != (source_after.st_dev, source_after.st_ino)
                or (source_after.st_dev, source_after.st_ino)
                != (source_current.st_dev, source_current.st_ino)
                or source_after.st_size != expected_size
                or source_after.st_size != daemon.os.fstat(
                    temporary_descriptor).st_size):
            raise daemon.PrimaryWorktreeError(
                "legacy archive changed during copy: " + source)
        daemon.os.close(temporary_descriptor)
        temporary_descriptor = -1
        try:
            daemon.os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if not daemon._regular_files_equal(first=source, second=destination):
                raise daemon.PrimaryWorktreeError(
                    "archive bridge destination raced with different bytes: "
                    + destination)
        directory_flags = daemon.os.O_RDONLY
        if hasattr(daemon.os, "O_DIRECTORY"):
            directory_flags |= daemon.os.O_DIRECTORY
        directory_descriptor = daemon.os.open(parent, directory_flags)
        try:
            daemon.os.fsync(directory_descriptor)
        finally:
            daemon.os.close(directory_descriptor)
    finally:
        daemon.os.close(source_descriptor)
        if temporary_descriptor >= 0:
            daemon.os.close(temporary_descriptor)
        if temporary is not None:
            try:
                daemon.os.remove(temporary)
            except FileNotFoundError:
                pass


def _remove_archive_copy_temporaries(worktree):
    """Remove regular copy residues left by an interrupted archive bridge.

    Arguments:
      worktree = the checkout whose done and relay folders are swept
                 for leftover temporary copy files.
    """
    roots = (
        daemon.os.path.join(worktree, "ai", "notes", "mailbox", "done"),
        daemon.os.path.join(worktree, "ai", "notes", "relay"),
    )
    for root in roots:
        if not daemon.os.path.lexists(root):
            continue
        info = daemon.os.lstat(root)
        if daemon.stat.S_ISLNK(info.st_mode) or not daemon.stat.S_ISDIR(info.st_mode):
            continue
        for directory, _names, files in daemon.os.walk(
                root, followlinks=False, onerror=daemon._raise_walk_error):
            removed = False
            for name in files:
                if not name.startswith(".primary-archive-"):
                    continue
                path = daemon.os.path.join(directory, name)
                entry = daemon.os.lstat(path)
                if daemon.stat.S_ISREG(entry.st_mode) and not daemon.stat.S_ISLNK(
                        entry.st_mode):
                    daemon.os.remove(path)
                    removed = True
            if removed:
                daemon.fsync_directory(directory=directory)


def _publish_primary_record(record, repository_root, bridge_main=False,
                            fence_empty_main=False):
    """Publish one selected record behind the applicable legacy locks.

    Arguments:
      record           = the validated registry record to publish.
      repository_root  = the repository's main checkout.
      bridge_main      = True when the main checkout's archived
                        transport must be bridged in first.
      fence_empty_main = True when the main checkout's empty mailbox
                        must be fenced against concurrent legacy use.

    Returns:
      The published state mapping.

    Raises:
      daemon.PrimaryWorktreeError: for a failed bridge, fence, or
        publication.
    """
    state = daemon._primary_state_for_record(
        record=record, repository_root=repository_root)
    state_file = daemon.primary_state_paths(repository_root)["state"]
    if not bridge_main and not fence_empty_main:
        daemon._atomic_write_primary_state(state=state, path=state_file)
        return state

    mailbox = daemon.os.path.join(repository_root, "ai", "notes", "mailbox")
    parent = daemon.os.path.dirname(mailbox)
    daemon._plain_directory(path=parent, label="legacy notes directory")
    mailbox_identity = daemon._plain_directory(
        path=mailbox, label="legacy mailbox", create=True)
    dispatch_lock = daemon._open_legacy_transport_lock(
        path=daemon.os.path.join(mailbox, ".dispatch.lock"), nonblocking=True)
    sequence_lock = None
    try:
        sequence_lock = daemon._open_legacy_transport_lock(
            path=daemon.os.path.join(mailbox, ".sequence.lock"), nonblocking=True)
        daemon._require_directory_identity(
            path=mailbox, identity=mailbox_identity,
            label="legacy mailbox")
        manifest = daemon._archived_transport_manifest(worktree=repository_root)
        if bridge_main and (manifest is None or not manifest):
            raise daemon.PrimaryWorktreeError(
                "legacy main transport is no longer archived-only; state "
                "was not published")
        if fence_empty_main and (manifest is None or manifest):
            raise daemon.PrimaryWorktreeError(
                "legacy main transport appeared before primary publication; "
                "state was not published")
        pre_ai_reasons = daemon._transport_evidence_at_notes(
            notes=daemon.os.path.join(repository_root, "notes"),
            reason_prefix="legacy pre-ai ")
        if pre_ai_reasons:
            raise daemon.PrimaryWorktreeError(
                "pre-migration main transport appeared before primary "
                "publication; state was not published: "
                + ", ".join(pre_ai_reasons))
        copied_manifest = daemon._archived_transport_manifest(
            worktree=record["path"])
        if copied_manifest is None:
            raise daemon.PrimaryWorktreeError(
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
                        or not daemon._regular_files_equal(
                            first=legacy[0], second=copied_source)):
                    raise daemon.PrimaryWorktreeError(
                        "primary contains transport that is not an exact "
                        "subset of the main archive; state was not published")
        target_notes = daemon.os.path.join(record["path"], "ai", "notes")
        if bridge_main:
            for (source, relative, expected_size, _dev, _ino,
                 _mtime) in manifest:
                relative_parent = daemon.os.path.dirname(relative)
                destination_parent = daemon._ensure_plain_relative_directory(
                    root=target_notes, relative=relative_parent)
                destination = daemon.os.path.join(destination_parent,
                                           daemon.os.path.basename(relative))
                daemon._copy_regular_archive_file(
                    source=source, destination=destination,
                    expected_size=expected_size)
        final_manifest = daemon._archived_transport_manifest(
            worktree=repository_root)
        if final_manifest != manifest:
            raise daemon.PrimaryWorktreeError(
                "legacy main archive changed during bridge; state was not "
                "published")
        if bridge_main:
            for (source, relative, expected_size, _dev, _ino,
                 _mtime) in final_manifest:
                destination = daemon.os.path.join(target_notes, relative)
                if (daemon.os.lstat(destination).st_size != expected_size
                        or not daemon._regular_files_equal(
                            first=source, second=destination)):
                    raise daemon.PrimaryWorktreeError(
                        "primary archive copy failed final byte validation; "
                        "state was not published: " + destination)
        final_pre_ai_reasons = daemon._transport_evidence_at_notes(
            notes=daemon.os.path.join(repository_root, "notes"),
            reason_prefix="legacy pre-ai ")
        if final_pre_ai_reasons:
            raise daemon.PrimaryWorktreeError(
                "pre-migration main transport changed during primary "
                "publication; state was not published: "
                + ", ".join(final_pre_ai_reasons))
        daemon._require_directory_identity(
            path=mailbox, identity=mailbox_identity,
            label="legacy mailbox")
        daemon._validate_primary_record(
            record=record, branch=record["branch"],
            repository_root=repository_root)
        daemon._atomic_write_primary_state(state=state, path=state_file)
    finally:
        if sequence_lock is not None:
            daemon._release_legacy_transport_lock(lock_file=sequence_lock)
        daemon._release_legacy_transport_lock(lock_file=dispatch_lock)
    if bridge_main:
        print("bridged archived main-checkout mailbox and relay history into "
              "the primary without deleting the originals", flush=True)
    return state


def _publish_adopted_primary_record(record, repository_root):
    """Publish an existing coordinator only while its mailbox is idle.

    Arguments:
      record          = the registry record of the checkout being
                        adopted.
      repository_root = the repository's main checkout.

    Returns:
      The published state mapping.

    Raises:
      daemon.PrimaryWorktreeError: when the adopted mailbox is live
        or the directories are unsafe.
    """
    mailbox = daemon.os.path.join(
        record["path"], "ai", "notes", "mailbox")
    daemon._plain_directory(
        path=daemon.os.path.dirname(mailbox), label="adopted notes directory")
    identity = daemon._plain_directory(
        path=mailbox, label="adopted mailbox", create=True)
    dispatch_lock = daemon._open_legacy_transport_lock(
        path=daemon.os.path.join(mailbox, ".dispatch.lock"), nonblocking=True)
    sequence_lock = None
    try:
        sequence_lock = daemon._open_legacy_transport_lock(
            path=daemon.os.path.join(mailbox, ".sequence.lock"), nonblocking=True)
        daemon._require_directory_identity(
            path=mailbox, identity=identity, label="adopted mailbox")
        reasons = daemon.coordination_transport_evidence(worktree=record["path"])
        allowed = daemon.CURRENT_ADOPTION_SAFE_REASONS | {
            "live watcher or once lock is held",
            "live sender or sequence lock is held",
        }
        if not reasons or not set(reasons).issubset(allowed):
            raise daemon.PrimaryWorktreeError(
                "adopted coordination transport changed before publication")
        daemon._validate_primary_record(
            record=record, branch=record["branch"],
            repository_root=repository_root)
        state = daemon._primary_state_for_record(
            record=record, repository_root=repository_root)
        daemon._atomic_write_primary_state(
            state=state, path=daemon.primary_state_paths(repository_root)["state"])
        return state
    finally:
        if sequence_lock is not None:
            daemon._release_legacy_transport_lock(lock_file=sequence_lock)
        daemon._release_legacy_transport_lock(lock_file=dispatch_lock)


def _format_evidence(candidates):
    """Format legacy coordination stores for one actionable refusal.

    Arguments:
      candidates = ``(path, reasons)`` pairs.

    Returns:
      One sorted, printable summary line.
    """
    return "; ".join(sorted(path + " (" + ", ".join(reasons) + ")"
                            for path, reasons in candidates))


def _branch_exists(repository_root, branch):
    """Return whether an exact local branch ref already exists.

    Arguments:
      repository_root = the repository to inspect.
      branch          = the full branch reference.

    Returns:
      True when the reference exists.

    Raises:
      daemon.PrimaryWorktreeError: when the inspection itself fails.
    """
    result = daemon._run_git(repository_root=repository_root,
                      arguments=["show-ref", "--verify", "--quiet", branch],
                      check=False)
    if result.returncode not in {0, 1}:
        raise daemon.PrimaryWorktreeError("cannot inspect primary branch collision")
    return result.returncode == 0


def provision_or_adopt_primary(repository_root, current_worktree):
    """Select one primary checkout under the already-held bootstrap lock.

    First run only: the default managed worktree is created or an
    existing coordinator is adopted, with legacy transport evidence
    either bridged (the archived-main case) or refused with an
    actionable listing.

    Arguments:
      repository_root  = the repository's main checkout.
      current_worktree = the checkout this process runs from.

    Returns:
      The published primary state mapping.

    Raises:
      daemon.PrimaryWorktreeError: for ambiguous legacy evidence or a
        failed provision.
    """
    paths = daemon.primary_state_paths(repository_root=repository_root)
    daemon._managed_primary_root(repository_root=repository_root, create=True)
    records = daemon.registered_worktrees(repository_root=repository_root)
    default_record = daemon._record_at_path(
        records=records, path=paths["default_path"])
    if (default_record is not None
            and default_record.get("branch") == daemon.PRIMARY_BRANCH):
        daemon._remove_archive_copy_temporaries(worktree=default_record["path"])
    evidence = []
    for record in records:
        reasons = daemon.coordination_transport_evidence(worktree=record["path"])
        if reasons:
            evidence.append((daemon.os.path.abspath(record["path"]), reasons))
    bridge_main = daemon._safe_main_archive_bridge(
        evidence=evidence, repository_root=repository_root,
        default_path=paths["default_path"])

    if default_record is not None:
        if default_record.get("branch") != daemon.PRIMARY_BRANCH:
            raise daemon.PrimaryWorktreeError(
                "default primary path is registered on another branch: "
                + paths["default_path"])
        foreign_evidence = [item for item in evidence
                            if daemon._path_key(item[0])
                            != daemon._path_key(paths["default_path"])]
        if foreign_evidence and not bridge_main:
            raise daemon.PrimaryWorktreeError(
                "refusing interrupted-bootstrap recovery because other "
                "coordination stores exist: "
                + daemon._format_evidence(foreign_evidence))
        daemon._validate_primary_record(record=default_record,
                                 branch=daemon.PRIMARY_BRANCH,
                                 repository_root=repository_root)
        return daemon._publish_primary_record(
            record=default_record, repository_root=repository_root,
            bridge_main=bridge_main, fence_empty_main=not bridge_main)

    branch_records = [record for record in records
                      if record.get("branch") == daemon.PRIMARY_BRANCH]
    if branch_records:
        raise daemon.PrimaryWorktreeError(
            "primary branch is already checked out at an unexpected path: "
            + ", ".join(sorted(record["path"] for record in branch_records)))
    if daemon.os.path.lexists(paths["default_path"]):
        raise daemon.PrimaryWorktreeError(
            "default primary path exists but is not a registered worktree: "
            + paths["default_path"])

    current_record = daemon._record_at_path(records=records, path=current_worktree)
    if evidence:
        if (not bridge_main and len(evidence) == 1
                and current_record is not None
                and daemon._path_key(evidence[0][0]) == daemon._path_key(current_worktree)
                and set(evidence[0][1]).issubset(
                    daemon.CURRENT_ADOPTION_SAFE_REASONS)
                and current_record.get("branch") not in {None,
                                                         "refs/heads/main"}
                and daemon.os.path.dirname(daemon.os.path.abspath(current_worktree))
                == paths["managed_root"]):
            daemon._validate_primary_record(
                record=current_record, branch=current_record["branch"],
                repository_root=repository_root)
            print("adopting current coordination worktree and preserving its "
                  "mailbox: " + daemon.os.path.abspath(current_worktree), flush=True)
            return daemon._publish_adopted_primary_record(
                record=current_record, repository_root=repository_root)
        if (not bridge_main and len(evidence) == 1
                and current_record is not None
                and daemon._path_key(evidence[0][0])
                == daemon._path_key(current_worktree)):
            raise daemon.PrimaryWorktreeError(
                "current coordination worktree cannot be adopted because "
                "its transport is pre-migration or unsafe; preserve and "
                "deliberately migrate or repair it before retrying: "
                + daemon._format_evidence(evidence))
        if not bridge_main:
            raise daemon.PrimaryWorktreeError(
                "existing coordination transport must be selected "
                "explicitly; run the command once from the one intended "
                "linked worktree. Candidates: " + daemon._format_evidence(evidence))

    if daemon._branch_exists(repository_root=repository_root,
                             branch=daemon.PRIMARY_BRANCH):
        raise daemon.PrimaryWorktreeError(
            "primary branch already exists without its registered default "
            "worktree; refusing to reset or reuse it: " + daemon.PRIMARY_BRANCH)

    short_branch = daemon.PRIMARY_BRANCH[len("refs/heads/"):]
    daemon._run_git(repository_root=repository_root,
             arguments=["worktree", "add", "-b", short_branch,
                        paths["default_path"], "main"])
    refreshed = daemon.registered_worktrees(repository_root=repository_root)
    created = daemon._record_at_path(records=refreshed,
                              path=paths["default_path"])
    if created is None:
        raise daemon.PrimaryWorktreeError(
            "git created no registered primary worktree; no state was saved")
    daemon._validate_primary_record(record=created, branch=daemon.PRIMARY_BRANCH,
                             repository_root=repository_root)
    if not bridge_main:
        appeared = []
        for candidate in refreshed:
            reasons = daemon.coordination_transport_evidence(
                worktree=candidate["path"])
            if reasons:
                appeared.append((daemon.os.path.abspath(candidate["path"]), reasons))
        if appeared:
            raise daemon.PrimaryWorktreeError(
                "coordination transport appeared during primary bootstrap; "
                "the new worktree was preserved but state was not published: "
                + daemon._format_evidence(appeared))
    print("created primary coordination worktree " + paths["default_path"]
          + " on " + daemon.PRIMARY_BRANCH, flush=True)
    return daemon._publish_primary_record(
        record=created, repository_root=repository_root,
        bridge_main=bridge_main, fence_empty_main=not bridge_main)


def _upgrade_primary_topology_state(state, repository_root):
    """Accept topology-aware state; never guess that every old process stopped.

    Arguments:
      state           = the loaded primary state.
      repository_root = the repository's main checkout.

    Returns:
      The state unchanged when it already carries the current schema.

    Raises:
      daemon.PrimaryWorktreeError: for an unknown schema, or a known
        older schema whose upgrade needs a deliberate operator step.
    """
    if state["schema"] == daemon.PRIMARY_STATE_SCHEMA:
        return state
    if state["schema"] not in {
            daemon.LEGACY_PRIMARY_STATE_SCHEMA, daemon.PREVIOUS_PRIMARY_STATE_SCHEMA}:
        raise daemon.PrimaryWorktreeError(
            "cannot upgrade unsupported primary-worktree state")
    # An old process can validate schema 1, pause before taking the dispatch
    # lock, and resume after an apparent in-place migration. No filesystem
    # lock introduced by this newer code can make that already-admitted old
    # process re-read state. Automatic migration would therefore make a false
    # safety claim. Preserve every byte and require an explicit stopped-old-
    # runtime recovery instead.
    raise daemon.PrimaryWorktreeError(
        "the saved mailbox topology predates the separate Implementer "
        "worktree and cannot be migrated safely while an "
        "older daemon may already be admitted; stop every old mailbox "
        "process, preserve the saved primary worktree and mailbox, update "
        "that worktree to this daemon version, move the old local state "
        "file aside for recovery, then run the current daemon from the "
        "saved primary path to initialize the new topology")


def _require_primary_daemon_topology_support(primary_path):
    """Refuse re-exec into a stale primary daemon that would ignore Sol.

    Arguments:
      primary_path = the primary worktree whose daemon copy is
                     inspected for the current topology markers.

    Raises:
      daemon.PrimaryWorktreeError: when the copy predates the
        three-role topology.
    """
    daemon_path = daemon.os.path.join(primary_path, "ai", "tools", "mailbox_daemon.py")
    try:
        initial = daemon.os.lstat(daemon_path)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot inspect saved primary daemon: " + str(exc))
    if (daemon.stat.S_ISLNK(initial.st_mode) or not daemon.stat.S_ISREG(initial.st_mode)
            or initial.st_size > daemon.MAX_PRIMARY_DAEMON_BYTES):
        raise daemon.PrimaryWorktreeError(
            "saved primary daemon is redirected, irregular, or too large: "
            + daemon_path)
    flags = daemon.os.O_RDONLY | daemon.os.O_NONBLOCK
    flags |= getattr(daemon.os, "O_CLOEXEC", 0)
    flags |= getattr(daemon.os, "O_NOFOLLOW", 0)
    try:
        descriptor = daemon.os.open(daemon_path, flags)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot open saved primary daemon safely: " + str(exc))
    try:
        opened = daemon.os.fstat(descriptor)
        chunks = []
        remaining = daemon.MAX_PRIMARY_DAEMON_BYTES + 1
        while remaining:
            chunk = daemon.os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        source = b"".join(chunks)
        after = daemon.os.fstat(descriptor)
        current = daemon.os.lstat(daemon_path)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot read saved primary daemon safely: " + str(exc))
    finally:
        daemon.os.close(descriptor)
    identities = ((initial.st_dev, initial.st_ino),
                  (opened.st_dev, opened.st_ino),
                  (after.st_dev, after.st_ino),
                  (current.st_dev, current.st_ino))
    if (len(set(identities)) != 1 or not daemon.stat.S_ISREG(opened.st_mode)
            or after.st_size != len(source)
            or len(source) > daemon.MAX_PRIMARY_DAEMON_BYTES):
        raise daemon.PrimaryWorktreeError(
            "saved primary daemon changed while compatibility was checked: "
            + daemon_path)
    declarations = daemon.re.findall(
        br"(?m)^MAILBOX_TOPOLOGY_VERSION = ([0-9]+)$", source)
    if declarations != [str(daemon.MAILBOX_TOPOLOGY_VERSION).encode("ascii")]:
        raise daemon.PrimaryWorktreeError(
            "saved primary daemon predates dedicated Sol worktrees; update "
            "that non-main worktree from main without discarding its local "
            "work, then retry: " + primary_path)
    protocol_declarations = daemon.re.findall(
        br"(?m)^MAILBOX_PROTOCOL_VERSION = ([0-9]+)$", source)
    if protocol_declarations != [
            str(daemon.MAILBOX_PROTOCOL_VERSION).encode("ascii")]:
        raise daemon.PrimaryWorktreeError(
            "saved primary daemon does not enforce the current "
            "Architect-only user entry point; "
            "update that non-main worktree from main without discarding "
            "its local work, then retry: " + primary_path)


def _bootstrap_ticket_state(primary_path):
    """Read the primary mailbox's current ticket state with strict schema.

    Arguments:
      primary_path = the primary worktree.

    Returns:
      The validated state mapping, or an empty state when no file
      exists yet.

    Raises:
      daemon.PrimaryWorktreeError: for an unreadable or invalid
        state file.
    """
    path = daemon.os.path.join(primary_path, "ai", "notes", "mailbox",
                        daemon.TICKET_CYCLE_STATE_NAME)
    if not daemon.os.path.isfile(path):
        return daemon.empty_ticket_cycle_state()
    try:
        raw = daemon.stable_regular_bytes(
            path=path, maximum_bytes=daemon.MAX_TICKET_CYCLE_STATE_BYTES,
            label="bootstrap ticket-cycle state")
        payload = daemon.json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=daemon._duplicate_key_refusal)
    except (OSError, ValueError, UnicodeDecodeError,
            daemon.json.JSONDecodeError) as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot verify bootstrap ticket-cycle state: " + str(exc)) \
            from exc
    try:
        return daemon.validate_ticket_cycle_state(payload=payload)
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(str(exc)) from exc


def _bootstrap_candidate_state(primary_path):
    """Read the primary mailbox's candidate ownership without globals.

    Arguments:
      primary_path = the primary worktree.

    Returns:
      The validated candidate-state mapping, or an empty one when no
      file exists yet.

    Raises:
      daemon.PrimaryWorktreeError: for an unreadable or invalid
        state file.
    """
    path = daemon.os.path.join(primary_path, "ai", "notes", "mailbox",
                        daemon.CANDIDATE_STATE_NAME)
    try:
        raw = daemon.stable_regular_bytes(
            path=path, maximum_bytes=daemon.MAX_CANDIDATE_STATE_BYTES,
            label="bootstrap ticket-candidate state", missing_ok=True)
    except (OSError, ValueError) as exc:
        raise daemon.PrimaryWorktreeError(str(exc)) from exc
    if raw is None:
        return daemon.empty_candidate_state()
    try:
        payload = daemon.json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=daemon._duplicate_key_refusal)
    except (UnicodeDecodeError, daemon.json.JSONDecodeError, ValueError,
            OverflowError, RecursionError) as exc:
        raise daemon.PrimaryWorktreeError(
            "bootstrap ticket-candidate state is not exact JSON") from exc
    if (not isinstance(payload, dict)
            or set(payload) != {"schema", "cycles"}
            or payload.get("schema") != daemon.CANDIDATE_STATE_SCHEMA
            or not isinstance(payload.get("cycles"), dict)
            or len(payload["cycles"]) > daemon.MAX_TICKET_CYCLE_RECORDS):
        raise daemon.PrimaryWorktreeError(
            "bootstrap ticket-candidate state has invalid keys")
    normalized = {}
    try:
        for cycle_id, record in payload["cycles"].items():
            expected_ref = daemon.cycle_candidate_ref(cycle_id=cycle_id)
            if (not isinstance(record, dict)
                    or set(record) != {"ref", "commit"}
                    or record.get("ref") != expected_ref
                    or not isinstance(record.get("commit"), str)
                    or daemon.FULL_COMMIT_RE.fullmatch(record["commit"]) is None):
                raise daemon.PrimaryWorktreeError(
                    "bootstrap ticket-candidate state has an invalid "
                    "cycle record")
            normalized[cycle_id] = {
                "ref": expected_ref, "commit": record["commit"]}
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(str(exc)) from exc
    return {"schema": daemon.CANDIDATE_STATE_SCHEMA, "cycles": normalized}


def _bootstrap_root_ticket_authority(message, target, ticket_state,
                                     candidate_state):
    """Prove a still-root ordinary GO is the exact journaled C-to-L work.

    Arguments:
      message         = the pending GO message.
      target          = the baseline commit sync would move to.
      ticket_state    = the primary's bootstrap ticket state.
      candidate_state = the primary's bootstrap candidate state.

    Returns:
      True only when the GO names an active cycle whose saved
      candidate and journaled landing explain the target exactly.
    """
    cycle_id, candidate_commit, mode, problem = daemon._architect_go_request(
        message=message)
    if problem is not None:
        return False, problem
    record = candidate_state["cycles"].get(cycle_id)
    candidate_ref = daemon.cycle_candidate_ref(cycle_id=cycle_id)
    if (record != {"ref": candidate_ref, "commit": candidate_commit}
            or daemon.git_ref_commit(reference=candidate_ref) != candidate_commit):
        return False, "root Architect GO has no exact saved candidate C"
    landing_ref = daemon.cycle_landing_ref(cycle_id=cycle_id)
    saved_landing = daemon.git_ref_commit(reference=landing_ref)
    if saved_landing != target:
        if saved_landing is not None:
            try:
                saved_parent = daemon._verify_prepared_landing(
                    cycle_id=cycle_id,
                    candidate_commit=candidate_commit,
                    landing_commit=saved_landing)
                main_problem = daemon._prepared_landing_main_problem(
                    candidate_commit=candidate_commit,
                    landing_commit=saved_landing,
                    parent_commit=saved_parent,
                    current_main=target)
            except daemon.TicketCycleStateError as exc:
                return False, str(exc)
            if main_problem is not None:
                return False, main_problem
        return False, "root Architect GO has no exact target landing ref L"
    try:
        parent = daemon._verify_prepared_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=target)
        daemon._require_ancestor_or_same(
            ancestor=daemon.cycle_starting_commit(cycle_id), descendant=parent,
            label="root Architect GO landing does not preserve its base")
    except daemon.TicketCycleStateError as exc:
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
    """Prove clean primary P may wait ahead of main B for exact GO replay.

    Arguments:
      primary_path = the primary worktree sitting at P.
      base_commit  = B, the main baseline.
      notes_commit = P, the note-only commit ahead of it.

    Returns:
      True only when exactly one saved notes-GO receipt binds this
      B and P, so a startup sync must not discard the waiting P.
    """
    mailbox = daemon.os.path.join(primary_path, "ai", "notes", "mailbox")
    receipt_matches = []
    for directory in (mailbox, daemon.os.path.join(mailbox, "inflight")):
        for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-daemon.md")):
            try:
                raw = daemon.stable_regular_bytes(
                    path=path,
                    maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="ahead-primary permanent-note GO")
                message = raw.decode("utf-8", errors="strict")
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            returned_base, returned_notes, problem = (
                daemon._architect_notes_go_request(message=message))
            if (problem is None and returned_base == base_commit
                    and returned_notes == notes_commit):
                receipt_matches.append((path, raw))
    if len(receipt_matches) != 1:
        return False
    try:
        daemon.require_architect_notes_commit_object(
            base_commit=base_commit, notes_commit=notes_commit)
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(str(exc)) from exc

    inflight = daemon.os.path.join(mailbox, "inflight")
    done = daemon.os.path.join(mailbox, "done")
    relay = daemon.os.path.join(primary_path, "ai", "notes", "relay")
    journal_prefix = ".pending-notes-admin-"
    journal_suffix = ".json"
    journal_paths = sorted(daemon.glob.glob(daemon.os.path.join(
        relay, journal_prefix + "*" + journal_suffix)))
    if len(journal_paths) != 1:
        raise daemon.PrimaryWorktreeError(
            "ahead Architect primary needs exactly one retained admin "
            "recovery journal; found " + str(len(journal_paths)))
    journal_name = daemon.os.path.basename(journal_paths[0])
    request_name = journal_name[len(journal_prefix):-len(journal_suffix)]
    request_match = daemon.PENDING_MESSAGE_RE.fullmatch(request_name)
    if request_match is None or request_match.group(1) != "fable":
        raise daemon.PrimaryWorktreeError(
            "ahead Architect primary has a malformed admin recovery "
            "journal name")
    admin_paths = [
        path for path in (daemon.os.path.join(inflight, request_name),
                          daemon.os.path.join(done, request_name))
        if daemon.regular_inode(path=path) is not None]
    if len(admin_paths) != 1:
        raise daemon.PrimaryWorktreeError(
            "ahead Architect primary recovery journal needs exactly one "
            "saved inflight or archived admin request; found "
            + str(len(admin_paths)))
    admin_path = admin_paths[0]
    try:
        admin_message = daemon.stable_regular_bytes(
            path=admin_path,
            maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="ahead-primary note admin").decode(
                "utf-8", errors="strict")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot verify saved permanent-note admin: " + str(exc)) \
            from exc
    if not daemon.is_architect_notes_admin_message(message=admin_message):
        raise daemon.PrimaryWorktreeError(
            "saved permanent-note admin is malformed")
    try:
        journal = daemon.read_architect_notes_admin_journal(
            request_name=request_name,
            request_message=admin_message, relay_dir=relay)
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(
            "ahead Architect primary has no valid admin recovery journal: "
            + str(exc)) from exc
    receipt_hash = daemon.hashlib.sha256(receipt_matches[0][1]).hexdigest()
    if (journal["base"] != base_commit
            or journal["phase"] != "validated-commit"
            or journal["notes_commit"] != notes_commit
            or journal["receipt_sha256"] != receipt_hash):
        raise daemon.PrimaryWorktreeError(
            "saved admin journal does not bind exact B/P receipt")
    try:
        daemon._validate_current_protected_primary_state(
            primary_worktree=primary_path)
    except daemon.PrimaryWorktreeError:
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
    branch = daemon._run_git(
        repository_root=daemon.REPO_ROOT,
        arguments=["symbolic-ref", "-q", "HEAD"], check=False)
    try:
        branch_name = branch.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return False
    return (daemon.os.path.realpath(daemon.WORKTREE)
            == daemon.os.path.realpath(daemon.REPO_ROOT)
            and branch.returncode == 0
            and branch_name == "refs/heads/main"
            and daemon.worktree_head(worktree=daemon.REPO_ROOT) == target
            and not daemon._tracked_worktree_changes(worktree=daemon.REPO_ROOT))


def _merge_primary_backlog(base, main, architect):
    """Combine independent main and Architect edits to the tracked backlog.

    A scratch three-way file merge runs in a temporary folder: the
    common base against main's copy and the Architect's copy.

    Arguments:
      base      = the common ancestor's backlog bytes.
      main      = main's backlog bytes.
      architect = the Architect's backlog bytes.

    Returns:
      The merged bytes, or ``None`` when the merge conflicts.
    """
    with daemon.tempfile.TemporaryDirectory(prefix="mailbox-backlog-merge-") as tmp:
        paths = []
        for name, content in (("main", main), ("base", base),
                              ("architect", architect)):
            path = daemon.os.path.join(tmp, name)
            with open(path, "wb") as stream:
                stream.write(content)
            paths.append(path)
        result = daemon.subprocess.run(
            ["git", "merge-file", "--stdout"] + paths,
            stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.PIPE, check=False)
    if result.returncode == 0:
        if len(result.stdout) > daemon.MAX_BACKLOG_LEDGER_BYTES:
            raise daemon.PrimaryWorktreeError(
                "combined Architect backlog exceeds its size limit")
        return result.stdout
    if result.returncode == 1:
        raise daemon.PrimaryWorktreeError(
            "main and the sealed Architect backlog edit the same text; "
            "both versions remain preserved for manual reconciliation")
    detail = result.stderr.decode("utf-8", errors="replace").strip()[:500]
    raise daemon.PrimaryWorktreeError(
        "Git could not compare the main and Architect backlog edits"
        + (": " + detail if detail else ""))


def _saved_backlog_digest(primary_path):
    """Read the accepted digest from the small local guard record.

    Arguments:
      primary_path = the primary worktree.

    Returns:
      The guard's accepted SHA-256, or ``None`` when no valid guard
      exists.
    """
    raw = daemon.stable_regular_bytes(
        path=daemon.os.path.join(primary_path, "ai", "notes",
                          daemon.BACKLOG_GUARD_STATE_NAME),
        maximum_bytes=daemon.MAX_BACKLOG_GUARD_STATE_BYTES,
        label="backlog guard state")
    try:
        state = daemon.json.loads(raw, object_pairs_hook=daemon._duplicate_key_refusal)
        digest = state["sha256"]
    except (KeyError, TypeError, daemon.json.JSONDecodeError) as exc:
        raise daemon.PrimaryWorktreeError("invalid backlog guard state") from exc
    if (not isinstance(digest, str)
            or daemon.re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise daemon.PrimaryWorktreeError("invalid backlog guard digest")
    return digest


def _reseal_recovered_backlog(primary_path, previous_digest, backlog):
    """Bind the guard to backlog bytes combined by the trusted daemon.

    Arguments:
      primary_path    = the primary worktree.
      previous_digest = the last accepted digest, recorded as the
                        seal's predecessor.
      backlog         = the recovered backlog bytes to seal.
    """
    daemon._atomic_write_primary_state(
        state={"backlog": daemon.BACKLOG_RELATIVE_PATH,
               "previous_sha256": previous_digest,
               "sha256": daemon.hashlib.sha256(backlog).hexdigest(), "version": 2},
        path=daemon.os.path.join(primary_path, "ai", "notes",
                          daemon.BACKLOG_GUARD_STATE_NAME))


def _prepare_primary_backlog_overlay(primary_path, primary_head, target):
    """Preserve and, when needed, merge one sealed Architect backlog.

    Arguments:
      primary_path = the primary worktree.
      primary_head = its current HEAD.
      target       = the baseline commit the sync will move to.

    Returns:
      The overlay bytes to restore after the sync — the sealed
      backlog, merged with the target's copy when both sides edited
      it — or ``None`` when nothing needs preserving.

    Raises:
      daemon.PrimaryWorktreeError: when the merge conflicts and the
        user must reconcile.
    """
    if not daemon._clean_worktree_status(worktree=primary_path):
        return None
    try:
        sealed = daemon._architect_only_sealed_backlog(worktree=primary_path)
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(str(exc)) from exc
    if sealed is None:
        raise daemon.PrimaryWorktreeError(
            "stale Architect primary has work beyond its sealed backlog; "
            "landing authority cannot advance it automatically")
    old = daemon._run_git(
        repository_root=primary_path,
        arguments=["show", primary_head + ":" + daemon.BACKLOG_RELATIVE_PATH],
        check=False)
    new = daemon._run_git(
        repository_root=primary_path,
        arguments=["show", target + ":" + daemon.BACKLOG_RELATIVE_PATH],
        check=False)
    if old.returncode != 0 or new.returncode != 0:
        raise daemon.PrimaryWorktreeError(
            "cannot read both tracked backlog versions before primary sync")
    merged = (sealed if old.stdout == new.stdout else
              daemon._merge_primary_backlog(
                  base=old.stdout, main=new.stdout, architect=sealed))
    backlog = daemon.os.path.join(primary_path, daemon.BACKLOG_RELATIVE_PATH)
    recovery = daemon.os.path.join(
        primary_path, "ai", "notes", daemon.BACKLOG_SYNC_RECOVERY_NAME)
    if daemon.os.path.lexists(recovery):
        raise daemon.PrimaryWorktreeError(
            "backlog sync recovery already exists; restart once to recover "
            "it before advancing the Architect primary")
    merged_temporary = None
    if merged != sealed:
        descriptor, merged_temporary = daemon.tempfile.mkstemp(
            prefix=".backlog-sync-merged-",
            dir=daemon.os.path.dirname(recovery))
        try:
            with daemon.os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(merged)
                stream.flush()
                daemon.os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                daemon.os.close(descriptor)
    daemon.os.replace(backlog, recovery)
    if merged_temporary is not None:
        daemon.os.replace(merged_temporary, recovery)
    restored = daemon._run_git(
        repository_root=primary_path,
        arguments=["restore", "--source=HEAD", "--staged", "--worktree",
                   "--", daemon.BACKLOG_RELATIVE_PATH],
        check=False)
    if restored.returncode != 0 or daemon._clean_worktree_status(primary_path):
        daemon.os.replace(recovery, backlog)
        raise daemon.PrimaryWorktreeError(
            "sealed Architect backlog could not be prepared for primary "
            "synchronization")
    return merged, recovery, daemon.hashlib.sha256(sealed).hexdigest()


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
    current_main = daemon._run_git(
        repository_root=daemon.REPO_ROOT,
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"])
    try:
        target = current_main.stdout.decode(
            "ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError("current main is not ASCII") from exc
    primary_head = daemon.worktree_head(worktree=primary_path)
    if primary_head == target:
        return False
    daemon._symbolic_worktree_branch(
        worktree=primary_path, expected_branch=primary_branch,
        label="Architect")
    old_backlog = daemon._run_git(
        repository_root=primary_path,
        arguments=["cat-file", "-e",
                   primary_head + ":" + daemon.BACKLOG_RELATIVE_PATH],
        check=False)
    new_backlog = daemon._run_git(
        repository_root=primary_path,
        arguments=["show", target + ":" + daemon.BACKLOG_RELATIVE_PATH],
        check=False)
    working_backlog = daemon.os.path.join(primary_path, daemon.BACKLOG_RELATIVE_PATH)
    if (old_backlog.returncode != 0 and new_backlog.returncode == 0
            and daemon.os.path.lexists(working_backlog)):
        sealed = daemon._validate_sealed_backlog(primary_worktree=primary_path)
        if sealed != new_backlog.stdout:
            raise daemon.PrimaryWorktreeError(
                "tracked backlog migration conflicts with the sealed local "
                "backlog; both versions were preserved")
        daemon.os.unlink(working_backlog)
    ahead = daemon._run_git(
        repository_root=daemon.REPO_ROOT,
        arguments=["merge-base", "--is-ancestor", target, primary_head],
        check=False)
    if ahead.returncode == 0:
        if daemon._bootstrap_primary_ahead_notes_authority(
                primary_path=primary_path, base_commit=target,
                notes_commit=primary_head):
            print("kept saved Architect primary at authorized permanent-note "
                  "commit " + primary_head + " while main remains at its "
                  "exact base " + target, flush=True)
            return False
        raise daemon.PrimaryWorktreeError(
            "Architect primary is ahead of main without one exact pending "
            "B/P permanent-note GO")
    if ahead.returncode != 1:
        raise daemon.PrimaryWorktreeError(
            "cannot compare saved Architect primary with current main")
    try:
        daemon._require_ancestor_or_same(
            ancestor=primary_head, descendant=target,
            label="stale Architect primary is not an ancestor of main")
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(str(exc)) from exc

    mailbox = daemon.os.path.join(primary_path, "ai", "notes", "mailbox")
    ticket_state = daemon._bootstrap_ticket_state(primary_path=primary_path)
    candidate_state = daemon._bootstrap_candidate_state(primary_path=primary_path)
    authorities = []
    root_problems = []
    for directory in (mailbox, daemon.os.path.join(mailbox, "done"),
                      daemon.os.path.join(mailbox, "inflight")):
        for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-daemon.md")):
            try:
                raw = daemon.stable_regular_bytes(
                    path=path,
                    maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="bootstrap landing request")
                message = raw.decode("utf-8", errors="strict")
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            if message.startswith(
                    daemon.MAILBOX_RETURN_HEADER + "architect-notes-go"):
                base, notes_commit, problem = (
                    daemon._architect_notes_go_request(message=message))
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
                        daemon.require_architect_notes_commit_object(
                            base_commit=base, notes_commit=notes_commit)
                    except daemon.TicketCycleStateError as exc:
                        raise daemon.PrimaryWorktreeError(str(exc)) from exc
                    authorities.append(("notes", notes_commit))
                continue
            if directory == mailbox:
                valid, problem = daemon._bootstrap_root_ticket_authority(
                    message=message, target=target,
                    ticket_state=ticket_state,
                    candidate_state=candidate_state)
                if valid:
                    authorities.append(("ticket", target))
                else:
                    root_problems.append(problem)
                continue
            cycle_id, _candidate, mode, problem = daemon._architect_go_request(
                message=message)
            if problem is not None:
                continue
            recorded = ticket_state["completed"].get(cycle_id)
            if recorded is None:
                active = ticket_state["active"].get(cycle_id)
                if active is not None:
                    recorded = active.get("commit")
            if recorded is None:
                landing_ref = daemon.cycle_landing_ref(cycle_id=cycle_id)
                result = daemon._run_git(
                    repository_root=daemon.REPO_ROOT,
                    arguments=["rev-parse", "--verify",
                               landing_ref + "^{commit}"], check=False)
                if result.returncode == 0:
                    try:
                        recorded = result.stdout.decode(
                            "ascii", errors="strict").strip()
                    except UnicodeDecodeError:
                        recorded = None
            if recorded == target and mode in daemon.ARCHITECT_COMMIT_MODES:
                authorities.append(("ticket", target))
    if root_problems:
        raise daemon.PrimaryWorktreeError(
            "stale Architect primary has an invalid root landing request: "
            + "; ".join(root_problems))
    if not authorities:
        if not daemon.clean_user_main_matches(target=target):
            raise daemon.PrimaryWorktreeError(
                "main is ahead of the Architect primary without an exact "
                "landing request or one clean user-owned main update")
        authority_label = "clean user main update"
    else:
        authority_label = "daemon-recorded landing"
    backlog_overlay = daemon._prepare_primary_backlog_overlay(
        primary_path=primary_path, primary_head=primary_head, target=target)
    result = daemon._run_git(
        repository_root=primary_path,
        arguments=["merge", "--ff-only", target], check=False)
    if result.returncode != 0:
        if backlog_overlay is not None:
            daemon.os.replace(backlog_overlay[1], working_backlog)
        raise daemon.PrimaryWorktreeError(
            "accepted main authority could not fast-forward the clean "
            "Architect primary")
    if backlog_overlay is not None:
        daemon.os.replace(backlog_overlay[1], working_backlog)
        daemon._reseal_recovered_backlog(
            primary_path=primary_path,
            previous_digest=backlog_overlay[2], backlog=backlog_overlay[0])
        try:
            restored_overlay = daemon._architect_only_sealed_backlog(
                worktree=primary_path)
        except daemon.TicketCycleStateError as exc:
            raise daemon.PrimaryWorktreeError(str(exc)) from exc
        if restored_overlay != backlog_overlay[0]:
            raise daemon.PrimaryWorktreeError(
                "sealed Architect backlog was not restored after primary "
                "synchronization")
    elif daemon._clean_worktree_status(worktree=primary_path):
        raise daemon.PrimaryWorktreeError(
            "accepted main authority did not leave a clean Architect primary")
    if daemon.worktree_head(worktree=primary_path) != target:
        raise daemon.PrimaryWorktreeError(
            "accepted main authority did not advance the Architect primary")
    print("advanced saved Architect primary to " + authority_label + " "
          + target, flush=True)
    return True


def validated_primary_notes(primary_path):
    """Return the canonical non-redirected shared notes directory.

    Arguments:
      primary_path = the primary worktree.

    Returns:
      The resolved ``ai/notes`` path after every level proves to be a
      plain directory.

    Raises:
      daemon.PrimaryWorktreeError: for a redirected or missing
        component.
    """
    primary = daemon.os.path.realpath(primary_path)
    daemon._plain_directory(path=primary_path, label="saved primary worktree")
    ai_root = daemon.os.path.join(primary_path, "ai")
    notes = daemon.os.path.join(ai_root, "notes")
    daemon._plain_directory(path=ai_root, label="saved primary ai directory")
    daemon._plain_directory(path=notes, label="saved primary notes directory")
    expected_ai = daemon.os.path.join(primary, "ai")
    expected_notes = daemon.os.path.join(expected_ai, "notes")
    if (daemon.os.path.realpath(ai_root) != expected_ai
            or daemon.os.path.realpath(notes) != expected_notes):
        raise daemon.PrimaryWorktreeError(
            "saved primary notes directory is redirected")
    return expected_notes


def validate_authoritative_role_files(primary_path):
    """Return stable proofs for every primary role and ticket tool.

    Arguments:
      primary_path = the primary worktree.

    Returns:
      Mapping with directory identities and per-file identity proofs
      for the role files and trusted tools a dispatch relies on.

    Raises:
      daemon.PrimaryWorktreeError: for a redirected directory or an
        unreadable authoritative file.
    """
    primary = daemon.os.path.abspath(primary_path)
    primary_real = daemon.os.path.realpath(primary)
    directory_paths = (
        ("saved primary worktree", primary, primary_real),
        ("saved primary .codex directory",
         daemon.os.path.join(primary, ".codex"),
         daemon.os.path.join(primary_real, ".codex")),
        ("saved primary .claude directory",
         daemon.os.path.join(primary, ".claude"),
         daemon.os.path.join(primary_real, ".claude")),
        ("saved primary ai directory",
         daemon.os.path.join(primary, "ai"),
         daemon.os.path.join(primary_real, "ai")),
        ("saved primary tools directory",
         daemon.os.path.join(primary, "ai", "tools"),
         daemon.os.path.join(primary_real, "ai", "tools")),
    )
    directory_proof = []
    for label, path, expected_real in directory_paths:
        identity = daemon._plain_directory(path=path, label=label)
        if daemon.os.path.realpath(path) != expected_real:
            raise daemon.PrimaryWorktreeError(label + " is redirected: " + path)
        directory_proof.append((label, path, identity))

    authoritative_files = tuple(
        ("role", daemon.os.path.join(primary, *path.split("/")))
        for path in daemon.ARCHITECT_ROLE_PATHS)
    authoritative_files += tuple(
        ("trusted tool", daemon.os.path.join(primary, *path.split("/")))
        for path in daemon.ARCHITECT_TRUSTED_TOOL_PATHS)
    authoritative_files += ((
        "role contract",
        daemon.os.path.join(primary, *daemon.ROLE_CONTRACT_RELATIVE_PATH.split("/"))),)
    file_proof = []
    for kind, path in authoritative_files:
        try:
            info = daemon.os.lstat(path)
        except OSError as exc:
            raise daemon.PrimaryWorktreeError(
                "authoritative " + kind + " is missing: " + str(exc))
        if daemon.stat.S_ISLNK(info.st_mode) or not daemon.stat.S_ISREG(info.st_mode):
            raise daemon.PrimaryWorktreeError(
                "authoritative " + kind + " must be a regular file: "
                + path)
        expected_real = daemon.os.path.join(
            primary_real, daemon.os.path.relpath(path, primary))
        if daemon.os.path.realpath(path) != expected_real:
            raise daemon.PrimaryWorktreeError(
                "authoritative " + kind + " is redirected: " + path)
        identity = (info.st_dev, info.st_ino, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        file_proof.append((kind, path, identity))

    proof = {
        "directories": tuple(directory_proof),
        "files": tuple(file_proof),
    }
    daemon.recheck_authoritative_role_files(proof=proof)
    return proof


def recheck_authoritative_role_files(proof, mutable_paths=()):
    """Require authoritative files to stay fixed outside an admin turn.

    Arguments:
      proof         = the mapping from
                      validate_authoritative_role_files.
      mutable_paths = the only files allowed to differ, such as the
                      Architect role files during policy
                      administration.

    Raises:
      daemon.PrimaryWorktreeError: for a malformed proof or a changed
        directory or file outside the mutable set.
    """
    if (not isinstance(proof, dict)
            or set(proof) != {"directories", "files"}):
        raise daemon.PrimaryWorktreeError(
            "authoritative role-file proof is missing or malformed")
    for label, path, identity in proof["directories"]:
        daemon._require_directory_identity(
            path=path, identity=identity, label=label)
    primary = proof["directories"][0][1]
    for kind, path, identity in proof["files"]:
        if daemon.os.path.relpath(path, primary) in mutable_paths:
            continue
        try:
            info = daemon.os.lstat(path)
        except OSError as exc:
            raise daemon.PrimaryWorktreeError(
                "cannot revalidate authoritative " + kind + ": "
                + str(exc))
        current = (info.st_dev, info.st_ino, info.st_size,
                   info.st_mtime_ns, info.st_ctime_ns)
        if (daemon.stat.S_ISLNK(info.st_mode) or not daemon.stat.S_ISREG(info.st_mode)
                or current != identity):
            raise daemon.PrimaryWorktreeError(
                "authoritative " + kind
                + " changed after topology validation: " + path)


def _sol_state_for_record(record, repository_root):
    """Build the exact persisted Sol record for one validated checkout.

    Arguments:
      record          = the validated registry record.
      repository_root = the repository's main checkout.

    Returns:
      The Sol state mapping with schema, repository identity, name,
      path, and branch.
    """
    return {
        "schema": daemon.SOL_STATE_SCHEMA,
        "repository": daemon.git_common_directory(checkout=repository_root),
        "name": daemon.os.path.basename(record["path"]),
        "path": daemon.os.path.abspath(record["path"]),
        "branch": record["branch"],
    }


def validate_sol_state(state, repository_root, primary_state,
                       allow_move=False):
    """Validate the saved Sol identity and prove role checkouts are distinct.

    Arguments:
      state           = the loaded Sol state.
      repository_root = the repository's main checkout.
      primary_state   = the validated primary state; the two
                        checkouts must differ.
      allow_move      = True to adopt a Git-recorded relocation.

    Returns:
      The validated (possibly updated) Sol state.

    Raises:
      daemon.PrimaryWorktreeError: for a schema, branch, registry, or
        distinctness violation.
    """
    if state["schema"] != daemon.SOL_STATE_SCHEMA:
        raise daemon.PrimaryWorktreeError("unsupported Sol-worktree state schema")
    if state["branch"] != daemon.SOL_BRANCH:
        raise daemon.PrimaryWorktreeError(
            "saved Sol worktree must use " + daemon.SOL_BRANCH)
    if primary_state["branch"] == state["branch"]:
        raise daemon.PrimaryWorktreeError(
            "Sol and Claude must use different branches")
    resolved = daemon.validate_primary_state(
        state=state, repository_root=repository_root, allow_move=False,
        state_path=daemon.sol_state_paths(repository_root)["state"])
    sol_path = daemon.os.path.realpath(resolved["path"])
    if sol_path == daemon.os.path.realpath(repository_root):
        raise daemon.PrimaryWorktreeError(
            "Sol worktree must not be the user's repository checkout")
    if sol_path == daemon.os.path.realpath(primary_state["path"]):
        raise daemon.PrimaryWorktreeError(
            "Sol and Claude must use different worktrees")
    if allow_move and resolved != state:
        daemon._atomic_write_primary_state(
            state=resolved, path=daemon.sol_state_paths(repository_root)["state"])
        print("Sol worktree moved by git; saved " + resolved["path"],
              flush=True)
    return resolved


def provision_or_reuse_sol(repository_root, primary_state):
    """Create or validate the one persisted Sol worktree under bootstrap lock.

    Arguments:
      repository_root = the repository's main checkout.
      primary_state   = the validated primary state.

    Returns:
      The validated Sol state, creating the default worktree and
      branch on first run.

    Raises:
      daemon.PrimaryWorktreeError: for a failed provision or
        validation.
    """
    paths = daemon.sol_state_paths(repository_root=repository_root)
    daemon._managed_primary_root(repository_root=repository_root, create=True)
    if daemon.os.path.lexists(paths["state"]):
        state = daemon.load_primary_state(path=paths["state"])
        return daemon.validate_sol_state(
            state=state, repository_root=repository_root,
            primary_state=primary_state, allow_move=True)

    records = daemon.registered_worktrees(repository_root=repository_root)
    default_record = daemon._record_at_path(
        records=records, path=paths["default_path"])
    if default_record is not None:
        if default_record.get("branch") != daemon.SOL_BRANCH:
            raise daemon.PrimaryWorktreeError(
                "default Sol path is registered on another branch: "
                + paths["default_path"])
        daemon._validate_primary_record(
            record=default_record, branch=daemon.SOL_BRANCH,
            repository_root=repository_root)
        state = daemon._sol_state_for_record(
            record=default_record, repository_root=repository_root)
        state = daemon.validate_sol_state(
            state=state, repository_root=repository_root,
            primary_state=primary_state)
        daemon._atomic_write_primary_state(state=state, path=paths["state"])
        print("recovered exact interrupted Sol-worktree bootstrap "
              + state["path"], flush=True)
        return state

    branch_records = [record for record in records
                      if record.get("branch") == daemon.SOL_BRANCH]
    if branch_records:
        raise daemon.PrimaryWorktreeError(
            "Sol branch is already checked out at an unexpected path: "
            + ", ".join(sorted(record["path"]
                               for record in branch_records)))
    if daemon.os.path.lexists(paths["default_path"]):
        raise daemon.PrimaryWorktreeError(
            "default Sol path exists but is not a registered worktree: "
            + paths["default_path"])
    if daemon._branch_exists(repository_root=repository_root, branch=daemon.SOL_BRANCH):
        raise daemon.PrimaryWorktreeError(
            "Sol branch already exists without its registered default "
            "worktree; refusing to reset or reuse it: " + daemon.SOL_BRANCH)

    base = daemon._run_git(
        repository_root=repository_root,
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"])
    try:
        base_commit = base.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError(
            "main commit identity is not ASCII: " + str(exc))
    if not daemon.re.fullmatch(r"[0-9a-fA-F]{40,64}", base_commit):
        raise daemon.PrimaryWorktreeError("git returned an invalid main commit")
    short_branch = daemon.SOL_BRANCH[len("refs/heads/"):]
    daemon._run_git(
        repository_root=repository_root,
        arguments=["worktree", "add", "-b", short_branch,
                   paths["default_path"], base_commit])
    refreshed = daemon.registered_worktrees(repository_root=repository_root)
    created = daemon._record_at_path(
        records=refreshed, path=paths["default_path"])
    if created is None:
        raise daemon.PrimaryWorktreeError(
            "git created no registered Sol worktree; no Sol state was saved")
    daemon._validate_primary_record(
        record=created, branch=daemon.SOL_BRANCH, repository_root=repository_root)
    state = daemon._sol_state_for_record(
        record=created, repository_root=repository_root)
    state = daemon.validate_sol_state(
        state=state, repository_root=repository_root,
        primary_state=primary_state)
    daemon._atomic_write_primary_state(state=state, path=paths["state"])
    print("created Sol worktree " + state["path"] + " on " + daemon.SOL_BRANCH,
          flush=True)
    return state


def _implementer_state_for_record(record, repository_root):
    """Build the exact persisted Implementer record.

    Arguments:
      record          = the validated registry record.
      repository_root = the repository's main checkout.

    Returns:
      The Implementer state mapping with schema, repository
      identity, name, path, and branch.
    """
    return {
        "schema": daemon.IMPLEMENTER_STATE_SCHEMA,
        "repository": daemon.git_common_directory(checkout=repository_root),
        "name": daemon.os.path.basename(record["path"]),
        "path": daemon.os.path.abspath(record["path"]),
        "branch": record["branch"],
    }


def validate_implementer_state(state, repository_root, primary_state,
                               allow_move=False):
    """Validate the fixed Implementer branch and its distinct checkout.

    Arguments:
      state           = the loaded Implementer state.
      repository_root = the repository's main checkout.
      primary_state   = the validated primary state; the two
                        checkouts must differ.
      allow_move      = True to adopt a Git-recorded relocation.

    Returns:
      The validated (possibly updated) Implementer state.

    Raises:
      daemon.PrimaryWorktreeError: for a schema, branch, registry, or
        distinctness violation.
    """
    if state["schema"] != daemon.IMPLEMENTER_STATE_SCHEMA:
        raise daemon.PrimaryWorktreeError(
            "unsupported Implementer-worktree state schema")
    if state["branch"] != daemon.IMPLEMENTER_BRANCH:
        raise daemon.PrimaryWorktreeError(
            "saved Implementer worktree must use " + daemon.IMPLEMENTER_BRANCH)
    if primary_state["branch"] == state["branch"]:
        raise daemon.PrimaryWorktreeError(
            "Architect and Implementer must use different branches")
    resolved = daemon.validate_primary_state(
        state=state, repository_root=repository_root, allow_move=False,
        state_path=daemon.implementer_state_paths(repository_root)["state"])
    implementer_path = daemon.os.path.realpath(resolved["path"])
    if implementer_path == daemon.os.path.realpath(repository_root):
        raise daemon.PrimaryWorktreeError(
            "Implementer worktree must not be the user's checkout")
    if implementer_path == daemon.os.path.realpath(primary_state["path"]):
        raise daemon.PrimaryWorktreeError(
            "Architect and Implementer must use different worktrees")
    if allow_move and resolved != state:
        daemon._atomic_write_primary_state(
            state=resolved,
            path=daemon.implementer_state_paths(repository_root)["state"])
        print("Implementer worktree moved by git; saved "
              + resolved["path"], flush=True)
    return resolved


def _tracked_worktree_changes(worktree):
    """Return staged, unstaged, or nonignored untracked worktree changes.

    Arguments:
      worktree = the checkout to inspect.

    Returns:
      Raw ``git status --porcelain`` bytes; empty means clean.

    Raises:
      daemon.PrimaryWorktreeError: when the status command fails.
    """
    environment = daemon.os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = daemon.subprocess.run(
        ["git", "-C", worktree, "status", "--porcelain=v1", "-z",
         "--untracked-files=all", "--ignore-submodules=none"],
        stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.PIPE, check=False,
        env=environment)
    if result.returncode != 0:
        raise daemon.PrimaryWorktreeError(
            "cannot inspect tracked or untracked changes in " + worktree)
    return result.stdout


def _optional_ref_commit(repository_root, reference):
    """Return one full ref commit, or ``None`` when the ref is absent.

    Arguments:
      repository_root = the repository to inspect.
      reference       = the reference name.

    Returns:
      The full commit, or ``None`` for a missing reference.

    Raises:
      daemon.PrimaryWorktreeError: for an unusable answer.
    """
    result = daemon._run_git(
        repository_root=repository_root,
        arguments=["rev-parse", "--verify", "--quiet",
                   reference + "^{commit}"], check=False)
    if result.returncode != 0:
        return None
    try:
        commit = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError(reference + " is not ASCII") from exc
    if daemon.FULL_COMMIT_RE.fullmatch(commit) is None:
        raise daemon.PrimaryWorktreeError(reference + " is not a full commit")
    return commit


def implementer_authority_snapshot(repository_root=None):
    """Snapshot Git state that an Implementer turn has no authority to move.

    Arguments:
      repository_root = the repository to snapshot, or ``None`` for
                        the main checkout.

    Returns:
      Mapping from protected item (the checked-out branch, main, and
      the role branches) to its current value, compared after the
      turn.
    """
    repository_root = daemon.REPO_ROOT if repository_root is None else repository_root
    symbolic = daemon._run_git(
        repository_root=repository_root,
        arguments=["symbolic-ref", "-q", "HEAD"], check=False)
    if symbolic.returncode not in (0, 1):
        raise daemon.PrimaryWorktreeError(
            "cannot inspect the user's checked-out branch")
    return {
        "local main": daemon._optional_ref_commit(
            repository_root, "refs/heads/main"),
        "origin/main": daemon._optional_ref_commit(
            repository_root, "refs/remotes/origin/main"),
        "user checkout branch": (
            symbolic.stdout if symbolic.returncode == 0 else None),
        "user checkout HEAD": daemon.worktree_head(worktree=repository_root),
        "user checkout status": daemon._tracked_worktree_changes(repository_root),
    }


def implementer_authority_changes(before, repository_root=None):
    """Name protected Git state that moved during one Implementer turn.

    Arguments:
      before          = the pre-turn snapshot.
      repository_root = the repository, or ``None`` for the main
                        checkout.

    Returns:
      The names whose values changed; empty means the turn stayed
      inside its authority.
    """
    after = daemon.implementer_authority_snapshot(repository_root=repository_root)
    return [name for name in before if before[name] != after[name]]


def provision_or_reuse_implementer(repository_root, primary_state):
    """Create or validate the one fixed Implementer checkout.

    Arguments:
      repository_root = the repository's main checkout.
      primary_state   = the validated primary state.

    Returns:
      The validated Implementer state, creating the default worktree
      and branch on first run.

    Raises:
      daemon.PrimaryWorktreeError: for a failed provision or
        validation.
    """
    paths = daemon.implementer_state_paths(repository_root=repository_root)
    daemon._managed_primary_root(repository_root=repository_root, create=True)
    if daemon.os.path.lexists(paths["state"]):
        state = daemon.load_primary_state(path=paths["state"])
        return daemon.validate_implementer_state(
            state=state, repository_root=repository_root,
            primary_state=primary_state, allow_move=True)

    records = daemon.registered_worktrees(repository_root=repository_root)
    default_record = daemon._record_at_path(
        records=records, path=paths["default_path"])
    if default_record is not None:
        if default_record.get("branch") != daemon.IMPLEMENTER_BRANCH:
            raise daemon.PrimaryWorktreeError(
                "default Implementer path is registered on another branch: "
                + paths["default_path"])
        daemon._validate_primary_record(
            record=default_record, branch=daemon.IMPLEMENTER_BRANCH,
            repository_root=repository_root)
        state = daemon._implementer_state_for_record(
            record=default_record, repository_root=repository_root)
        state = daemon.validate_implementer_state(
            state=state, repository_root=repository_root,
            primary_state=primary_state)
        daemon._atomic_write_primary_state(state=state, path=paths["state"])
        print("recovered exact interrupted Implementer-worktree bootstrap "
              + state["path"], flush=True)
        return state

    branch_records = [record for record in records
                      if record.get("branch") == daemon.IMPLEMENTER_BRANCH]
    if branch_records:
        raise daemon.PrimaryWorktreeError(
            "Implementer branch is already checked out at an unexpected "
            "path: " + ", ".join(sorted(
                record["path"] for record in branch_records)))
    if daemon.os.path.lexists(paths["default_path"]):
        raise daemon.PrimaryWorktreeError(
            "default Implementer path exists but is not a registered "
            "worktree: " + paths["default_path"])
    if daemon._branch_exists(
            repository_root=repository_root, branch=daemon.IMPLEMENTER_BRANCH):
        raise daemon.PrimaryWorktreeError(
            "Implementer branch already exists without its registered "
            "default worktree; refusing to reset or reuse it: "
            + daemon.IMPLEMENTER_BRANCH)
    if daemon._tracked_worktree_changes(worktree=primary_state["path"]):
        raise daemon.PrimaryWorktreeError(
            "cannot split an Implementer worktree from a primary checkout "
            "with uncommitted tracked changes; commit or preserve that work "
            "before retrying")

    base = daemon._run_git(
        repository_root=primary_state["path"],
        arguments=["rev-parse", "--verify", "HEAD^{commit}"])
    try:
        base_commit = base.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError(
            "primary commit identity is not ASCII") from exc
    if not daemon.re.fullmatch(r"[0-9a-fA-F]{40,64}", base_commit):
        raise daemon.PrimaryWorktreeError(
            "git returned an invalid primary commit")
    short_branch = daemon.IMPLEMENTER_BRANCH[len("refs/heads/"):]
    daemon._run_git(
        repository_root=repository_root,
        arguments=["worktree", "add", "-b", short_branch,
                   paths["default_path"], base_commit])
    refreshed = daemon.registered_worktrees(repository_root=repository_root)
    created = daemon._record_at_path(
        records=refreshed, path=paths["default_path"])
    if created is None:
        raise daemon.PrimaryWorktreeError(
            "git created no registered Implementer worktree; no state was "
            "saved")
    daemon._validate_primary_record(
        record=created, branch=daemon.IMPLEMENTER_BRANCH,
        repository_root=repository_root)
    state = daemon._implementer_state_for_record(
        record=created, repository_root=repository_root)
    state = daemon.validate_implementer_state(
        state=state, repository_root=repository_root,
        primary_state=primary_state)
    daemon._atomic_write_primary_state(state=state, path=paths["state"])
    print("created Implementer worktree " + state["path"] + " on "
          + daemon.IMPLEMENTER_BRANCH, flush=True)
    return state


def validate_distinct_agent_states(primary_state, implementer_state,
                                   sol_state):
    """Prove the three role checkouts and branches are pairwise distinct.

    Arguments:
      primary_state     = the validated Architect state.
      implementer_state = the validated Implementer state.
      sol_state         = the validated Sol state.

    Raises:
      daemon.PrimaryWorktreeError: when any two roles share a path
        or branch.
    """
    paths = {
        daemon.os.path.realpath(primary_state["path"]),
        daemon.os.path.realpath(implementer_state["path"]),
        daemon.os.path.realpath(sol_state["path"]),
    }
    branches = {
        primary_state["branch"], implementer_state["branch"],
        sol_state["branch"],
    }
    if len(paths) != 3 or len(branches) != 3:
        raise daemon.PrimaryWorktreeError(
            "Architect, Implementer, and Sol require distinct worktrees "
            "and branches")


def _open_primary_lock(repository_root):
    """Open and exclusively lock the repo-shared bootstrap inode.

    Arguments:
      repository_root = the repository's main checkout.

    Returns:
      The open locked file; the caller must release it.

    Raises:
      daemon.PrimaryWorktreeError: when the lock cannot be opened or
        held safely.
    """
    paths = daemon.primary_state_paths(repository_root=repository_root)
    daemon._managed_primary_root(repository_root=repository_root, create=True)
    flags = daemon.os.O_RDWR | daemon.os.O_CREAT
    if hasattr(daemon.os, "O_NOFOLLOW"):
        flags |= daemon.os.O_NOFOLLOW
    try:
        descriptor = daemon.os.open(paths["lock"], flags, 0o600)
    except OSError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot open primary-worktree lock: " + str(exc))
    try:
        opened = daemon.os.fstat(descriptor)
        current = daemon.os.lstat(paths["lock"])
        if (not daemon.stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)):
            raise daemon.PrimaryWorktreeError(
                "primary-worktree lock is redirected or irregular")
        daemon.fcntl.flock(descriptor, daemon.fcntl.LOCK_EX)
        after = daemon.os.fstat(descriptor)
        current = daemon.os.lstat(paths["lock"])
        if ((opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino)
                != (current.st_dev, current.st_ino)):
            raise daemon.PrimaryWorktreeError(
                "primary-worktree lock changed while waiting")
        return daemon.os.fdopen(descriptor, "r+", encoding="utf-8")
    except BaseException:
        daemon.os.close(descriptor)
        raise


def _release_primary_lock(lock_file):
    """Release the kernel-owned primary bootstrap lock.

    Arguments:
      lock_file = the open locked file from the open call.
    """
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def _is_ai_branch(branch):
    """Return whether a local branch belongs to an AI-only namespace.

    Arguments:
      branch = the full branch reference.

    Returns:
      True for a branch under one of the AI cleanup prefixes.
    """
    return (type(branch) is str
            and branch.startswith(daemon.AI_BRANCH_PREFIXES))


def _lock_cleanup_transport(records):
    """Hold every existing mailbox lock until destructive cleanup ends.

    Arguments:
      records = the worktree registry records whose mailbox locks
                must be held so no watcher can act mid-cleanup.

    Returns:
      The held locks; the caller releases them after cleanup.

    Raises:
      daemon.PrimaryWorktreeError: when a live lock cannot be taken,
        which means a watcher is still running.
    """
    locks = []
    try:
        for record in sorted(records, key=lambda item: item["path"]):
            for notes in (daemon.os.path.join("ai", "notes"), "notes"):
                mailbox = daemon.os.path.join(record["path"], notes, "mailbox")
                if not daemon.os.path.lexists(mailbox):
                    continue
                identity = daemon._plain_directory(
                    path=mailbox, label="cleanup mailbox")
                for name in (".dispatch.lock", ".sequence.lock"):
                    locks.append(daemon._open_legacy_transport_lock(
                        path=daemon.os.path.join(mailbox, name), nonblocking=True))
                daemon._require_directory_identity(
                    path=mailbox, identity=identity,
                    label="cleanup mailbox")
        return locks
    except BaseException:
        for lock_file in reversed(locks):
            daemon._release_legacy_transport_lock(lock_file=lock_file)
        raise


def clean_all_ai_worktrees(repository_root, current_worktree):
    """Discard local AI worktrees and branches after an explicit request.

    This is the destructive ``--clean-all`` action: every managed AI
    worktree and every branch in the AI namespaces is removed, dirty
    files and unmerged commits included. It must run from the user's
    main repository folder and takes every mailbox lock first so no
    watcher can act mid-cleanup.

    Arguments:
      repository_root  = the repository's main checkout.
      current_worktree = the folder this command runs from; it must
                         be the main checkout itself.

    Raises:
      daemon.PrimaryWorktreeError: when run elsewhere or when a live
        watcher still holds a mailbox lock.
    """
    repository = daemon.os.path.abspath(repository_root)
    if daemon.os.path.realpath(current_worktree) != daemon.os.path.realpath(repository):
        raise daemon.PrimaryWorktreeError(
            "run --clean-all from the user's main repository folder")
    lock_file = daemon._open_primary_lock(repository)
    transport_locks = []
    try:
        managed_root = daemon._managed_primary_root(repository, create=True)
        records = daemon.registered_worktrees(repository)
        root_record = daemon._record_at_path(records, repository)
        if root_record is None or daemon._is_ai_branch(root_record.get("branch")):
            raise daemon.PrimaryWorktreeError(
                "the user's repository folder must use a non-AI branch")
        for record in records:
            reasons = daemon.coordination_transport_evidence(record["path"])
            if any("live " in reason for reason in reasons):
                raise daemon.PrimaryWorktreeError(
                    "stop the live mailbox watcher or sender before "
                    "--clean-all: " + record["path"])
        transport_locks = daemon._lock_cleanup_transport(records=records)
        print("WARNING: --clean-all permanently discards dirty files and "
              "unmerged commits in local AI worktrees.", flush=True)
        daemon._run_git(repository, ["worktree", "prune"])
        records = daemon.registered_worktrees(repository)
        preserved = set()
        for record in sorted(records, key=lambda item: item["path"]):
            path = daemon.os.path.abspath(record["path"])
            if daemon._path_key(path) == daemon._path_key(repository):
                continue
            managed_child = daemon.os.path.dirname(path) == managed_root
            ai_branch = daemon._is_ai_branch(record.get("branch"))
            if not ai_branch and "branch" in record:
                if managed_child:
                    preserved.add(daemon._path_key(path))
                continue
            if not ai_branch and not managed_child:
                continue
            branch = record.get("branch", "detached audit")
            print("discarding AI worktree " + path + " (" + branch + ")",
                  flush=True)
            daemon._run_git(repository, ["worktree", "remove", "--force",
                                  "--force", path])
        daemon._run_git(repository, ["worktree", "prune"])
        with daemon.os.scandir(managed_root) as entries:
            stale_paths = sorted(entry.path for entry in entries)
        for path in stale_paths:
            if (daemon.os.path.basename(path) == daemon.PRIMARY_LOCK_NAME
                    or daemon._path_key(path) in preserved):
                continue
            print("discarding stale AI path " + path, flush=True)
            info = daemon.os.lstat(path)
            if (daemon.stat.S_ISDIR(info.st_mode)
                    and not daemon.stat.S_ISLNK(info.st_mode)):
                daemon.shutil.rmtree(path)
            else:
                daemon.os.unlink(path)
        output = daemon._run_git(repository, [
            "for-each-ref", "--format=%(refname)", "refs/heads/"])
        branches = [ref for ref in output.stdout.decode("utf-8").splitlines()
                    if daemon._is_ai_branch(ref)]
        for ref in sorted(branches):
            print("deleting local AI branch "
                  + ref[len("refs/heads/"):], flush=True)
            daemon._run_git(repository, ["update-ref", "-d", ref])
            if daemon._branch_exists(repository, ref):
                raise daemon.PrimaryWorktreeError(
                    "could not delete local AI branch " + ref)
        print("clean-all finished; main and non-AI branches were not "
              "changed.", flush=True)
    finally:
        for transport_lock in reversed(transport_locks):
            daemon._release_legacy_transport_lock(lock_file=transport_lock)
        daemon._release_primary_lock(lock_file=lock_file)
