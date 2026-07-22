"""Candidate records, audit snapshots, and the exact squash landing.

The Implementer's finished proposal is one saved commit, called
candidate C. The daemon accepts it by creating a different commit,
landing L, whose content is exactly C squashed onto the current
``main`` together with the sealed backlog. This file records
candidates and the private Git references that keep them reachable,
snapshots the repository state an Architect audit saw, prepares and
re-verifies the exact squash landing, and installs that landing in
the user's clean checkout without rewriting saved history.

Docstrings here also name the ticket's base commit B, the main commit
observed before and after an integration M0 and M1, and the running
watcher D0.

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
    "git_commit_exists",
    "git_commit_descends_from",
    "cycle_candidate_ref",
    "candidate_state_path",
    "empty_candidate_state",
    "read_candidate_state",
    "write_candidate_state",
    "git_ref_commit",
    "candidate_record_locked",
    "_clean_worktree_status",
    "worktree_head",
    "prepare_implementer_cycle_checkout",
    "ticket_class_configuration_problem",
    "candidate_changed_paths",
    "classify_candidate_scope",
    "candidate_scope_for_cycle",
    "record_implementer_candidate",
    "candidate_commit_for_cycle",
    "record_architect_repair_scope",
    "write_implementer_delivery_receipt",
    "recover_implementer_deliveries",
    "audit_snapshot_path",
    "_validate_audit_record",
    "create_audit_snapshot",
    "remove_audit_snapshot",
    "discard_interrupted_audit_snapshot",
    "_exact_git_object",
    "_single_commit_parent",
    "_require_ancestor_or_same",
    "_commit_is_ancestor",
    "stale_integration_details",
    "_prepared_landing_main_problem",
    "_exact_squash_tree",
    "_tree_with_backlog",
    "_landing_backlog",
    "cycle_landing_ref",
    "reopen_decision_ref",
    "_reopen_decision_message",
    "prepare_reopen_decision_landing",
    "_candidate_commit_message",
    "_landing_commit_message",
    "_verify_prepared_landing",
    "prepare_exact_squash_landing",
    "_user_checkout_status",
    "land_prepared_commit_in_clean_user_checkout",
    "_push_debt_path",
    "write_push_debt",
    "push_exact_landing_or_record_debt",
    "retire_cycle_landing_ref",
    "recorded_landing_for_architect_go",
    "redteam_closure_request_payload",
    "control_plane_review_request_payload",
    "matching_control_plane_review_request",
    "publish_control_plane_review_request",
    "publish_control_plane_repair_request",
    "control_plane_integration_request_payload",
    "control_plane_integration_request",
    "matching_control_plane_integration_request",
    "publish_control_plane_integration_request",
    "matching_redteam_closure_request",
    "publish_redteam_closure_request",
)


def git_commit_exists(commit):
    """Return whether the primary coordination repository owns this commit."""
    if not isinstance(commit, str) or daemon.FULL_COMMIT_RE.fullmatch(commit) is None:
        return False
    process = daemon.subprocess.run(
        ["git", "cat-file", "-e", commit + "^{commit}"],
        cwd=daemon.AGENT_CWD["fable"], stdout=daemon.subprocess.DEVNULL,
        stderr=daemon.subprocess.DEVNULL, check=False)
    return process.returncode == 0


def git_commit_descends_from(starting_commit, accepted_commit):
    """Return whether daemon-recorded landing L descends from the base."""
    if (not daemon.git_commit_exists(commit=starting_commit)
            or not daemon.git_commit_exists(commit=accepted_commit)
            or starting_commit == accepted_commit):
        return False
    process = daemon.subprocess.run(
        ["git", "merge-base", "--is-ancestor", starting_commit,
         accepted_commit],
        cwd=daemon.AGENT_CWD["fable"], stdout=daemon.subprocess.DEVNULL,
        stderr=daemon.subprocess.DEVNULL, check=False)
    return process.returncode == 0


def cycle_candidate_ref(cycle_id):
    """Return one path-safe, deterministic private ref for a ticket."""
    if not isinstance(cycle_id, str) or daemon.CYCLE_ID_RE.fullmatch(cycle_id) is None:
        raise daemon.TicketCycleStateError("invalid cycle id for candidate ref")
    digest = daemon.hashlib.sha256(cycle_id.encode("utf-8")).hexdigest()
    return daemon.CANDIDATE_REF_ROOT + "/" + digest + "/candidate"


def candidate_state_path():
    """Return the ignored primary record binding cycles to immutable refs."""
    return daemon.os.path.join(daemon.MAILBOX, daemon.CANDIDATE_STATE_NAME)


def empty_candidate_state():
    """Return a fresh candidate-state payload."""
    return {"schema": daemon.CANDIDATE_STATE_SCHEMA, "cycles": {}}


def read_candidate_state():
    """Read the bounded exact-schema candidate record."""
    try:
        raw = daemon.stable_regular_bytes(
            path=daemon.candidate_state_path(),
            maximum_bytes=daemon.MAX_CANDIDATE_STATE_BYTES,
            label="ticket-candidate state", missing_ok=True)
    except (OSError, ValueError) as exc:
        raise daemon.TicketCycleStateError(str(exc)) from exc
    if raw is None:
        return daemon.empty_candidate_state()
    try:
        payload = daemon.json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=daemon.unique_json_object)
    except (UnicodeDecodeError, daemon.json.JSONDecodeError, ValueError,
            OverflowError, RecursionError) as exc:
        raise daemon.TicketCycleStateError(
            "ticket-candidate state is not exact JSON") from exc
    if (not isinstance(payload, dict)
            or set(payload) != {"schema", "cycles"}
            or payload.get("schema") != daemon.CANDIDATE_STATE_SCHEMA
            or not isinstance(payload.get("cycles"), dict)
            or len(payload["cycles"]) > daemon.MAX_TICKET_CYCLE_RECORDS):
        raise daemon.TicketCycleStateError(
            "ticket-candidate state has invalid keys")
    normalized = {}
    for cycle_id, record in payload["cycles"].items():
        expected_ref = daemon.cycle_candidate_ref(cycle_id=cycle_id)
        if (not isinstance(record, dict)
                or set(record) != {"ref", "commit"}
                or record.get("ref") != expected_ref
                or not isinstance(record.get("commit"), str)
                or daemon.FULL_COMMIT_RE.fullmatch(record["commit"]) is None):
            raise daemon.TicketCycleStateError(
                "ticket-candidate state has an invalid cycle record")
        normalized[cycle_id] = {
            "ref": expected_ref, "commit": record["commit"]}
    return {"schema": daemon.CANDIDATE_STATE_SCHEMA, "cycles": normalized}


def write_candidate_state(state):
    """Publish candidate state by same-directory atomic replacement."""
    if (not isinstance(state, dict)
            or set(state) != {"schema", "cycles"}
            or state.get("schema") != daemon.CANDIDATE_STATE_SCHEMA):
        raise daemon.TicketCycleStateError(
            "refusing malformed ticket-candidate state")
    # Round-trip through the strict reader's structural rules before write.
    for cycle_id, record in state["cycles"].items():
        if (not isinstance(record, dict)
                or record.get("ref") != daemon.cycle_candidate_ref(cycle_id)
                or daemon.FULL_COMMIT_RE.fullmatch(
                    str(record.get("commit", ""))) is None):
            raise daemon.TicketCycleStateError(
                "refusing malformed ticket-candidate cycle")
    daemon.os.makedirs(daemon.MAILBOX, exist_ok=True)
    payload = (daemon.json.dumps(state, sort_keys=True, indent=2) + "\n").encode(
        "utf-8")
    if len(payload) > daemon.MAX_CANDIDATE_STATE_BYTES:
        raise daemon.TicketCycleStateError("ticket-candidate state is too large")
    descriptor, temporary = daemon.tempfile.mkstemp(
        prefix=daemon.CANDIDATE_STATE_NAME + ".tmp-", dir=daemon.MAILBOX)
    try:
        daemon.os.fchmod(descriptor, 0o600)
        with daemon.os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            daemon.os.fsync(stream.fileno())
        daemon.os.replace(temporary, daemon.candidate_state_path())
        daemon.fsync_directory(directory=daemon.MAILBOX)
    except BaseException:
        if descriptor >= 0:
            daemon.os.close(descriptor)
        try:
            daemon.os.remove(temporary)
        except FileNotFoundError:
            pass
        raise


def git_ref_commit(reference):
    """Return one private ref's full commit, or None when it is absent."""
    result = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["rev-parse", "--verify", "--quiet",
                   reference + "^{commit}"],
        check=False)
    if result.returncode == 1:
        return None
    if result.returncode != 0:
        raise daemon.TicketCycleStateError(
            "cannot inspect candidate ref " + reference)
    try:
        commit = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "candidate ref is not ASCII") from exc
    if daemon.FULL_COMMIT_RE.fullmatch(commit) is None:
        raise daemon.TicketCycleStateError("candidate ref has invalid commit")
    return commit


def candidate_record_locked(cycle_id, ticket_state, candidate_state,
                            recover=True):
    """Return and verify one candidate, adopting an interrupted ref write."""
    active = ticket_state["active"].get(cycle_id)
    record = candidate_state["cycles"].get(cycle_id)
    reference = daemon.cycle_candidate_ref(cycle_id=cycle_id)
    ref_commit = daemon.git_ref_commit(reference=reference)
    if (ref_commit is not None
            and (record is None or record["commit"] != ref_commit)):
        previous = (daemon.cycle_starting_commit(cycle_id)
                    if record is None else record["commit"])
        if (not recover or active is None
                or active["phase"] != "implementation"
                or not daemon.git_commit_descends_from(
                    starting_commit=previous,
                    accepted_commit=ref_commit)):
            raise daemon.TicketCycleStateError(
                "unowned candidate ref exists for " + cycle_id)
        record = {"ref": reference, "commit": ref_commit}
        candidate_state["cycles"][cycle_id] = record
        daemon.write_candidate_state(state=candidate_state)
    if record is None:
        return None
    if ref_commit != record["commit"]:
        raise daemon.TicketCycleStateError(
            "candidate state and Git ref disagree for " + cycle_id)
    if (active is None or active["phase"] != "implementation"
            or not daemon.git_commit_descends_from(
                starting_commit=daemon.cycle_starting_commit(cycle_id),
                accepted_commit=record["commit"])):
        raise daemon.TicketCycleStateError(
            "candidate ref does not belong to an active implementation")
    return record


def _clean_worktree_status(worktree):
    """Return exact porcelain bytes without permitting index refresh."""
    environment = daemon.os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = daemon.subprocess.run(
        ["git", "-C", worktree, "status", "--porcelain=v1", "-z",
         "--untracked-files=normal", "--ignore-submodules=none"],
        stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.PIPE, check=False,
        env=environment)
    if result.returncode != 0:
        raise daemon.TicketCycleStateError(
            "cannot inspect Implementer worktree status")
    return result.stdout


def worktree_head(worktree):
    """Return the exact full commit checked out in one worktree."""
    result = daemon._run_git(
        repository_root=worktree,
        arguments=["rev-parse", "--verify", "HEAD^{commit}"])
    try:
        commit = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError("worktree HEAD is not ASCII") from exc
    if daemon.FULL_COMMIT_RE.fullmatch(commit) is None:
        raise daemon.TicketCycleStateError("worktree HEAD is not a full commit")
    return commit


def prepare_implementer_cycle_checkout(
        cycle_id, preserve_current=False, restart_from_base=False):
    """Select the cycle tip, or preserve a validated context checkpoint."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        ticket_state = daemon.read_ticket_cycle_state()
        active = ticket_state["active"].get(cycle_id)
        if active is None or active["phase"] != "implementation":
            raise daemon.TicketCycleStateError(
                "Implementer checkout has no active implementation cycle")
        candidate_state = daemon.read_candidate_state()
        record = daemon.candidate_record_locked(
            cycle_id=cycle_id, ticket_state=ticket_state,
            candidate_state=candidate_state)
        target = (record["commit"] if record is not None
                  and not restart_from_base
                  else daemon.cycle_starting_commit(cycle_id))
        worktree = daemon.AGENT_CWD["opus"]
        if preserve_current:
            return daemon.worktree_head(worktree=worktree)
        if daemon._clean_worktree_status(worktree=worktree):
            raise daemon.TicketCycleStateError(
                "Implementer worktree is not clean; refusing to reset it")
        current = daemon.worktree_head(worktree=worktree)
        preserved = {item["commit"]
                     for item in candidate_state["cycles"].values()}
        if current != target and current not in preserved:
            try:
                daemon._require_ancestor_or_same(
                    ancestor=current, descendant=target,
                    label="Implementer HEAD is not an ancestor of the "
                          "ticket base")
            except daemon.TicketCycleStateError as exc:
                main_commit = daemon._exact_git_object(
                    arguments=["rev-parse", "--verify",
                               "refs/heads/main^{commit}"],
                    label="current main commit")
                if record is not None or current != main_commit:
                    raise daemon.TicketCycleStateError(
                        "Implementer HEAD is not a saved candidate, an older "
                        "ticket-base ancestor, or the trusted main baseline; "
                        "refusing to discard " + current) from exc
                daemon._require_ancestor_or_same(
                    ancestor=target, descendant=current,
                    label="ticket base is not an ancestor of the trusted "
                          "main baseline")
        daemon._run_git(
            repository_root=worktree,
            arguments=["reset", "--hard", target])
        if daemon.worktree_head(worktree=worktree) != target:
            raise daemon.TicketCycleStateError(
                "Implementer reset did not select the requested candidate")
        return target
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def ticket_class_configuration_problem(ticket_class, skip_redteam=False):
    """Explain why this trusted watcher cannot run one ticket class."""
    if ticket_class not in daemon.TICKET_CLASSES:
        return "invalid ticket class"
    if ticket_class == "protected-control-plane":
        return ("protected-control-plane is reserved for Architect-owned "
                "ai/notes administration and cannot dispatch an Implementer; "
                "keep an ai/tools ticket Open for external maintenance")
    return None


def candidate_changed_paths(base_commit, candidate_commit, repository=None):
    """Return every repository path changed from ticket base B to candidate C."""
    if repository is None:
        repository = daemon.AGENT_CWD["opus"]
    changed = daemon._run_git(
        repository_root=repository,
        arguments=["diff", "--name-only", "-z", "--no-renames",
                   base_commit, candidate_commit, "--", "."])
    try:
        return {
            item.decode("utf-8", errors="strict")
            for item in changed.stdout.split(b"\0") if item}
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "Implementer candidate contains a non-UTF-8 path") from exc


def classify_candidate_scope(changed_paths, path_scope,
                             ticket_class="ordinary"):
    """Classify candidate C against global protection and its ticket file list."""
    protected = daemon.candidate_forbidden_paths(
        changed_paths, ticket_class=ticket_class)
    return daemon._CANDIDATE_ADMISSION.classify(
        changed_paths, path_scope, protected)


def candidate_scope_for_cycle(cycle_id, candidate_commit):
    """Recompute the exact ticket-scope result shown to the Architect."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        record = daemon.read_ticket_cycle_state()["active"].get(cycle_id)
        path_scope = None if record is None else record.get("path_scope")
        ticket_class = ("ordinary" if record is None else
                        record.get("ticket_class", "ordinary"))
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)
    # A ticket already running when this field was introduced has no frozen
    # scope. Preserve that one ticket under the earlier Architect-only audit;
    # every newly launched Implementer handoff records the scope below.
    if path_scope is None:
        return None
    changed = daemon.candidate_changed_paths(
        base_commit=daemon.cycle_starting_commit(cycle_id),
        candidate_commit=candidate_commit)
    result, paths = daemon.classify_candidate_scope(
        changed, path_scope, ticket_class=ticket_class)
    return {"result": result, "paths": sorted(paths)}


def record_implementer_candidate(
        cycle_id, starting_head, replace_prior=False):
    """Atomically preserve a successful clean Opus commit for its cycle."""
    worktree = daemon.AGENT_CWD["opus"]
    if daemon._clean_worktree_status(worktree=worktree):
        raise daemon.TicketCycleStateError(
            "successful Implementer turn left an uncommitted worktree")
    candidate = daemon.worktree_head(worktree=worktree)
    if candidate == starting_head:
        return None
    if not daemon.git_commit_descends_from(
            starting_commit=starting_head, accepted_commit=candidate):
        raise daemon.TicketCycleStateError(
            "Implementer result is not a new descendant of its saved base")
    changed_paths = daemon.candidate_changed_paths(
        base_commit=daemon.cycle_starting_commit(cycle_id),
        candidate_commit=candidate)
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        ticket_state = daemon.read_ticket_cycle_state()
        active = ticket_state["active"].get(cycle_id)
        if active is None or active["phase"] != "implementation":
            raise daemon.TicketCycleStateError(
                "candidate commit has no active implementation cycle")
        path_scope = active.get("path_scope")
        ticket_class = active.get("ticket_class", "ordinary")
        scope_result, scope_paths = daemon.classify_candidate_scope(
            changed_paths, path_scope or changed_paths,
            ticket_class=ticket_class)
        if scope_result == "PROTECTED_PATH_VIOLATION":
            raise daemon.TicketCycleStateError(
                scope_result + ": "
                + ", ".join(repr(path) for path in sorted(scope_paths)))
        candidate_state = daemon.read_candidate_state()
        prior = daemon.candidate_record_locked(
            cycle_id=cycle_id, ticket_state=ticket_state,
            candidate_state=candidate_state)
        expected = (prior["commit"] if prior is not None else "0" * 40)
        expected_start = (daemon.cycle_starting_commit(cycle_id)
                          if replace_prior else
                          (prior["commit"] if prior is not None
                           else daemon.cycle_starting_commit(cycle_id)))
        if starting_head != expected_start:
            raise daemon.TicketCycleStateError(
                "Implementer result began from another cycle tip")
        if ticket_class == "protected-control-plane":
            # A stale prepared L was built from the prior C. Retire only that
            # private journal before publishing a revised candidate.
            landing_reference = daemon.cycle_landing_ref(cycle_id=cycle_id)
            prior_landing = daemon.git_ref_commit(reference=landing_reference)
            if prior_landing is not None:
                daemon._run_git(
                    repository_root=daemon.AGENT_CWD["fable"],
                    arguments=["update-ref", "-d", landing_reference,
                               prior_landing])
                if daemon.git_ref_commit(reference=landing_reference) is not None:
                    raise daemon.TicketCycleStateError(
                        "superseded protected landing was not retired")
        reference = daemon.cycle_candidate_ref(cycle_id=cycle_id)
        daemon._run_git(
            repository_root=daemon.AGENT_CWD["fable"],
            arguments=["update-ref", reference, candidate, expected])
        candidate_state["cycles"][cycle_id] = {
            "ref": reference, "commit": candidate}
        daemon.write_candidate_state(state=candidate_state)
        if ticket_class == "protected-control-plane":
            # Every revision is a new immutable C. Neither earlier key nor
            # earlier integration or shadow evidence can authorize it.
            ticket_state["active"][cycle_id] = dict(
                active, control_plane=daemon.empty_control_plane_state())
            daemon.write_ticket_cycle_state(state=ticket_state)
        if scope_result == "SCOPE_EXCEEDED":
            print("  SCOPE_EXCEEDED; candidate preserved for Architect: "
                  + ", ".join(repr(path) for path in sorted(scope_paths)))
        return candidate
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def candidate_commit_for_cycle(cycle_id):
    """Return the verified immutable candidate for one active cycle."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        ticket_state = daemon.read_ticket_cycle_state()
        candidate_state = daemon.read_candidate_state()
        record = daemon.candidate_record_locked(
            cycle_id=cycle_id, ticket_state=ticket_state,
            candidate_state=candidate_state)
        return None if record is None else record["commit"]
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def record_architect_repair_scope(cycle_id, handoff_message):
    """Replace the file list after one authenticated Architect repair."""
    evidence = daemon.prepare_implementer_evidence_contract(
        message=handoff_message)
    proposed = sorted(evidence["allowed_paths"])
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        candidate = daemon.candidate_record_locked(
            cycle_id=cycle_id, ticket_state=state,
            candidate_state=daemon.read_candidate_state())
        if (active is None or active["phase"] != "implementation"
                or candidate is None):
            raise daemon.TicketCycleStateError(
                "Architect repair scope has no preserved active candidate")
        if active.get("ticket_class", "ordinary") != evidence["ticket_class"]:
            raise daemon.TicketCycleStateError(
                "Architect repair changed the frozen Ticket class")
        state["active"][cycle_id] = dict(active, path_scope=proposed)
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def write_implementer_delivery_receipt(request_path, return_path):
    """Hard-link a validated role return before its request is archived."""
    request = daemon.stable_regular_bytes(
        path=request_path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
        label="Implementer request")
    request_name = daemon.os.path.basename(request_path)
    match = daemon.PENDING_MESSAGE_RE.fullmatch(request_name)
    request_agent = match.group(1) if match is not None else None
    if request_agent not in {"opus", "fable"}:
        raise daemon.TicketCycleStateError(
            "invalid request name for delivery recovery")
    return_raw = daemon.stable_regular_bytes(
        path=return_path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
        label="Implementer return")
    return_name = daemon.os.path.basename(return_path)
    match = daemon.PENDING_MESSAGE_RE.fullmatch(return_name)
    return_agent = match.group(1) if match is not None else None
    if (request_agent, return_agent) not in {
            ("opus", "fable"), ("fable", "daemon"), ("fable", "opus")}:
        raise daemon.TicketCycleStateError(
            "invalid delivery-receipt route")
    path = daemon.os.path.join(
        daemon.MAILBOX, daemon.IMPLEMENTER_DELIVERY_PREFIX
        + "@".join((request_name, daemon.hashlib.sha256(request).hexdigest(),
                    return_name, daemon.hashlib.sha256(return_raw).hexdigest())))
    created = False
    try:
        daemon.os.link(return_path, path, follow_symlinks=False)
        created = True
    except FileExistsError:
        pass
    try:
        linked = daemon.stable_regular_bytes(
            path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="Implementer delivery receipt")
        if linked != return_raw:
            raise daemon.TicketCycleStateError(
                "Implementer return changed while its receipt was saved")
        daemon.fsync_directory(directory=daemon.MAILBOX)
    except BaseException:
        if created:
            daemon.os.remove(path)
        raise
    return path


def recover_implementer_deliveries():
    """Finish exact candidate deliveries interrupted after a valid return."""
    pattern = daemon.os.path.join(
        daemon.MAILBOX, daemon.IMPLEMENTER_DELIVERY_PREFIX + "*")
    recovered = 0
    for receipt_path in sorted(daemon.glob.glob(pattern)):
        receipt_name = daemon.os.path.basename(receipt_path)
        encoded = receipt_name[len(daemon.IMPLEMENTER_DELIVERY_PREFIX):]
        fields = encoded.split("@")
        if len(fields) != 4:
            raise daemon.TicketCycleStateError(
                "Implementer delivery receipt has the wrong filename")
        request_name, request_sha256, return_name, return_sha256 = fields
        request_match = daemon.PENDING_MESSAGE_RE.fullmatch(request_name)
        return_match = daemon.PENDING_MESSAGE_RE.fullmatch(return_name)
        request_agent = (request_match.group(1)
                         if request_match is not None else None)
        return_agent = (return_match.group(1)
                        if return_match is not None else None)
        if ((request_agent, return_agent) not in {
                ("opus", "fable"), ("fable", "daemon"), ("fable", "opus")}
                or daemon.re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None
                or daemon.re.fullmatch(r"[0-9a-f]{64}", return_sha256) is None):
            raise daemon.TicketCycleStateError(
                "Implementer delivery receipt has the wrong filename")
        return_paths = [daemon.os.path.join(directory, return_name)
                        for directory in (daemon.MAILBOX,
                                          daemon.os.path.join(daemon.MAILBOX, "inflight"),
                                          daemon.DONE)
                        if daemon.os.path.lexists(daemon.os.path.join(
                            directory, return_name))]
        if len(return_paths) != 1:
            raise daemon.TicketCycleStateError(
                "validated Implementer return has "
                + str(len(return_paths)) + " mailbox locations")
        return_raw = daemon.stable_regular_bytes(
            path=return_paths[0],
            maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="validated Implementer return")
        receipt_raw = daemon.stable_regular_bytes(
            path=receipt_path,
            maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="Implementer delivery receipt")
        if (daemon.hashlib.sha256(return_raw).hexdigest() != return_sha256
                or daemon.hashlib.sha256(receipt_raw).hexdigest() != return_sha256):
            raise daemon.TicketCycleStateError(
                "Implementer return changed before delivery recovery")
        inflight = daemon.os.path.join(daemon.MAILBOX, "inflight", request_name)
        done = daemon.os.path.join(daemon.DONE, request_name)
        guard = inflight + daemon.STATE_GUARD_SUFFIX
        done_inode = daemon.regular_inode(path=done)
        inflight_inode = daemon.regular_inode(path=inflight)
        if done_inode is not None:
            for leftover in (inflight, guard):
                if (daemon.os.path.lexists(leftover)
                        and daemon.regular_inode(path=leftover) != done_inode):
                    raise daemon.TicketCycleStateError(
                        "interrupted request archive changed identity")
            request_path = done
        elif inflight_inode is not None:
            if (daemon.os.path.lexists(guard)
                    and daemon.regular_inode(path=guard) != inflight_inode):
                raise daemon.TicketCycleStateError(
                    "interrupted request guard changed identity")
            request_path = inflight
        else:
            raise daemon.TicketCycleStateError(
                "interrupted Implementer request is missing")
        request_raw = daemon.stable_regular_bytes(
            path=request_path,
            maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="interrupted Implementer request")
        if daemon.hashlib.sha256(request_raw).hexdigest() != request_sha256:
            raise daemon.TicketCycleStateError(
                "Implementer request changed before delivery recovery")
        if done_inode is not None:
            for leftover in (inflight, guard):
                if daemon.os.path.lexists(leftover):
                    daemon.os.remove(leftover)
            daemon.fsync_directory(directory=daemon.os.path.dirname(inflight))
        elif daemon.os.path.lexists(guard):
            daemon.os.remove(guard)
            daemon.fsync_directory(directory=daemon.os.path.dirname(inflight))
        request_message = daemon.read_cycle_message(path=request_path)
        cycle_id, mode, request_body, problem = daemon._ticket_flow_envelope(
            message=request_message)
        if problem is not None:
            raise daemon.TicketCycleStateError(problem)
        returned_message = daemon.read_cycle_message(path=receipt_path)
        returned_cycle, returned_mode, returned_body, problem = (
            daemon._ticket_flow_envelope(message=returned_message))
        if request_agent == "fable":
            candidate = daemon.candidate_commit_for_cycle(cycle_id=cycle_id)
            if (candidate is None
                    or daemon.IMPLEMENTER_CANDIDATE_LINE_RE.findall(
                        request_message) != [candidate]):
                raise daemon.TicketCycleStateError(
                    "saved Architect audit does not name its exact candidate")
            if return_agent == "daemon":
                returned_cycle, returned_candidate, returned_mode, problem = (
                    daemon._architect_go_request(message=returned_message))
                if (problem is not None or returned_cycle != cycle_id
                        or returned_candidate != candidate
                        or returned_mode != mode):
                    raise daemon.TicketCycleStateError(
                        "saved Architect GO does not match its audit")
            else:
                problem = daemon.architect_handoff_problem(
                    message=returned_message, cycle_id=cycle_id, mode=mode,
                    checkpoint=daemon.is_implementer_checkpoint_request(
                        body=request_body),
                    budget=daemon.is_implementer_budget_checkpoint(request_body))
                if problem is not None:
                    raise daemon.TicketCycleStateError(
                        "saved Architect repair is invalid: "
                        + problem)
                daemon.record_architect_repair_scope(
                    cycle_id=cycle_id, handoff_message=returned_message)
            if (daemon.os.path.dirname(request_path)
                    != daemon.os.path.abspath(daemon.DONE)):
                if not daemon.archive_consumed_message(dispatch_path=request_path):
                    raise daemon.TicketCycleStateError(
                        "interrupted Architect request could not be archived")
            daemon.os.remove(receipt_path)
            daemon.fsync_directory(directory=daemon.MAILBOX)
            recovered += 1
            print("recovered validated delivery for " + request_name)
            continue
        candidates = (daemon.IMPLEMENTER_CANDIDATE_LINE_RE.findall(returned_body)
                      if problem is None else [])
        if (returned_cycle != cycle_id or returned_mode != mode
                or len(candidates) != 1):
            raise daemon.TicketCycleStateError(
                "saved Implementer return is not a completed handoff")
        candidate = candidates[0]
        budget_repair = daemon.is_architect_budget_repair(request_body)
        prior_candidate = daemon.candidate_commit_for_cycle(cycle_id=cycle_id)
        if prior_candidate != candidate:
            starting_head = (daemon.cycle_starting_commit(cycle_id=cycle_id)
                             if budget_repair else prior_candidate)
            if starting_head is None:
                starting_head = daemon.cycle_starting_commit(cycle_id=cycle_id)
            if daemon.worktree_head(worktree=daemon.AGENT_CWD["opus"]) != candidate:
                raise daemon.TicketCycleStateError(
                    "Implementer worktree no longer holds the delivered "
                    "candidate")
            if daemon.record_implementer_candidate(
                    cycle_id=cycle_id,
                    starting_head=starting_head,
                    replace_prior=budget_repair) != candidate:
                raise daemon.TicketCycleStateError(
                    "delivered candidate was not preserved")
        if daemon.os.path.dirname(request_path) != daemon.os.path.abspath(daemon.DONE):
            if not daemon.archive_consumed_message(dispatch_path=request_path):
                raise daemon.TicketCycleStateError(
                    "interrupted Implementer request could not be archived")
        daemon.os.remove(receipt_path)
        daemon.fsync_directory(directory=daemon.MAILBOX)
        recovered += 1
        print("recovered validated delivery for " + request_name)
    return recovered


def audit_snapshot_path(cycle_id, agent):
    """Return a deterministic managed path for one exact audit checkout."""
    if agent not in {"fable", "sol"}:
        raise ValueError("audit snapshot agent must be fable or sol")
    digest = daemon.hashlib.sha256(cycle_id.encode("utf-8")).hexdigest()[:24]
    return daemon.os.path.join(
        daemon._managed_primary_root(repository_root=daemon.REPO_ROOT),
        daemon.AUDIT_WORKTREE_PREFIX + digest + "-" + agent)


def _validate_audit_record(record, path, commit):
    """Prove one registered detached audit worktree names one commit."""
    expected = daemon._managed_child_path(
        path=path,
        managed_root=daemon._managed_primary_root(repository_root=daemon.REPO_ROOT))
    if (record is None or "detached" not in record["flags"]
            or "branch" in record or "prunable" in record["flags"]):
        raise daemon.PrimaryWorktreeError(
            "audit worktree must be registered and detached: " + expected)
    if daemon.git_common_directory(checkout=expected) != daemon.git_common_directory(
            checkout=daemon.REPO_ROOT):
        raise daemon.PrimaryWorktreeError(
            "audit worktree belongs to another repository")
    if daemon.worktree_head(worktree=expected) != commit:
        raise daemon.PrimaryWorktreeError(
            "audit worktree does not name the exact candidate commit")
    return expected


def create_audit_snapshot(cycle_id, commit, agent):
    """Create or recover a detached exact-commit checkout for one audit."""
    if (not isinstance(commit, str)
            or daemon.FULL_COMMIT_RE.fullmatch(commit) is None
            or not daemon.git_commit_exists(commit=commit)):
        raise daemon.TicketCycleStateError("audit commit is not an exact commit")
    path = daemon.audit_snapshot_path(cycle_id=cycle_id, agent=agent)
    lock_file = daemon._open_primary_lock(repository_root=daemon.REPO_ROOT)
    try:
        records = daemon.registered_worktrees(repository_root=daemon.REPO_ROOT)
        record = daemon._record_at_path(records=records, path=path)
        if record is not None:
            return daemon._validate_audit_record(
                record=record, path=path, commit=commit)
        if daemon.os.path.lexists(path):
            raise daemon.PrimaryWorktreeError(
                "audit path exists without a registered worktree: " + path)
        daemon._run_git(
            repository_root=daemon.REPO_ROOT,
            arguments=["worktree", "add", "--detach", path, commit])
        refreshed = daemon.registered_worktrees(repository_root=daemon.REPO_ROOT)
        created = daemon._record_at_path(records=refreshed, path=path)
        return daemon._validate_audit_record(
            record=created, path=path, commit=commit)
    finally:
        daemon._release_primary_lock(lock_file=lock_file)


def remove_audit_snapshot(cycle_id, commit, agent):
    """Remove only the unchanged disposable snapshot created for this turn."""
    path = daemon.audit_snapshot_path(cycle_id=cycle_id, agent=agent)
    lock_file = daemon._open_primary_lock(repository_root=daemon.REPO_ROOT)
    try:
        records = daemon.registered_worktrees(repository_root=daemon.REPO_ROOT)
        record = daemon._record_at_path(records=records, path=path)
        if record is None:
            if daemon.os.path.lexists(path):
                raise daemon.PrimaryWorktreeError(
                    "unregistered audit path remains: " + path)
            return
        daemon._validate_audit_record(record=record, path=path, commit=commit)
        if daemon._tracked_worktree_changes(worktree=path):
            raise daemon.PrimaryWorktreeError(
                "audit changed tracked files; preserving snapshot " + path)
        # Ignored bytecode or test caches are disposable inside this exact,
        # detached, commit-bound checkout. No user or candidate work lives
        # here, so force removes only audit artifacts after the tracked proof.
        daemon._run_git(
            repository_root=daemon.REPO_ROOT,
            arguments=["worktree", "remove", "--force", path])
        daemon._run_git(
            repository_root=daemon.REPO_ROOT,
            arguments=["worktree", "prune"])
        if (daemon.os.path.lexists(path)
                or daemon._record_at_path(
                    records=daemon.registered_worktrees(repository_root=daemon.REPO_ROOT),
                    path=path) is not None):
            raise daemon.PrimaryWorktreeError(
                "audit snapshot removal could not be verified")
    finally:
        daemon._release_primary_lock(lock_file=lock_file)


def discard_interrupted_audit_snapshot(cycle_id, commit, agent):
    """Remove one exact interrupted audit checkout, including its edits."""
    path = daemon.audit_snapshot_path(cycle_id=cycle_id, agent=agent)
    lock_file = daemon._open_primary_lock(repository_root=daemon.REPO_ROOT)
    try:
        records = daemon.registered_worktrees(repository_root=daemon.REPO_ROOT)
        record = daemon._record_at_path(records=records, path=path)
        if record is None:
            if daemon.os.path.lexists(path):
                raise daemon.PrimaryWorktreeError(
                    "unregistered audit path remains: " + path)
            return
        daemon._validate_audit_record(record=record, path=path, commit=commit)
        daemon._run_git(daemon.REPO_ROOT, ["worktree", "remove", "--force", path])
        daemon._run_git(daemon.REPO_ROOT, ["worktree", "prune"])
        if daemon.os.path.lexists(path):
            raise daemon.PrimaryWorktreeError(
                "interrupted audit checkout was not removed")
    finally:
        daemon._release_primary_lock(lock_file=lock_file)


def _exact_git_object(arguments, label):
    """Return one full Git object name from a bounded read-only command."""
    try:
        result = daemon._run_git(
            repository_root=daemon.AGENT_CWD["fable"],
            arguments=arguments, check=False)
    except daemon.PrimaryWorktreeError as exc:
        raise daemon.TicketCycleStateError(
            "cannot inspect " + label + ": " + str(exc)) from exc
    if result.returncode != 0:
        detail = result.stderr.decode(
            "utf-8", errors="replace").strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        raise daemon.TicketCycleStateError(
            "cannot inspect " + label
            + ((": " + detail) if detail else ""))
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(label + " is not ASCII") from exc
    if daemon.FULL_COMMIT_RE.fullmatch(value) is None:
        raise daemon.TicketCycleStateError(label + " is not one exact Git object")
    return value


def _single_commit_parent(commit):
    """Return the sole parent of a squash landing commit."""
    try:
        result = daemon._run_git(
            repository_root=daemon.AGENT_CWD["fable"],
            arguments=["rev-list", "--parents", "-n", "1", commit],
            check=False)
    except daemon.PrimaryWorktreeError as exc:
        raise daemon.TicketCycleStateError(
            "cannot inspect landing parents: " + str(exc)) from exc
    if result.returncode != 0:
        raise daemon.TicketCycleStateError("cannot inspect landing parents")
    try:
        fields = result.stdout.decode(
            "ascii", errors="strict").strip().split()
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "landing parent record is not ASCII") from exc
    if (len(fields) != 2 or fields[0] != commit
            or daemon.FULL_COMMIT_RE.fullmatch(fields[1]) is None):
        raise daemon.TicketCycleStateError(
            "Architect landing must be one ordinary commit with one parent")
    return fields[1]


def _require_ancestor_or_same(ancestor, descendant, label):
    """Require ``descendant`` to preserve ``ancestor`` in its lineage."""
    if daemon._commit_is_ancestor(ancestor=ancestor, descendant=descendant,
                           label=label):
        return
    raise daemon.TicketCycleStateError(label)


def _commit_is_ancestor(ancestor, descendant, label):
    """Return whether one exact commit remains in another's history."""
    if ancestor == descendant:
        return True
    try:
        result = daemon._run_git(
            repository_root=daemon.AGENT_CWD["fable"],
            arguments=["merge-base", "--is-ancestor", ancestor,
                       descendant],
            check=False)
    except daemon.PrimaryWorktreeError as exc:
        raise daemon.TicketCycleStateError(
            "cannot inspect " + label + ": " + str(exc)) from exc
    if result.returncode not in {0, 1}:
        raise daemon.TicketCycleStateError("cannot inspect " + label)
    return result.returncode == 0


def stale_integration_details(problem):
    """Return exact C, L, M0, and M1 from D0's own stale diagnosis."""
    match = daemon.STALE_INTEGRATION_RE.search(str(problem))
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
    if daemon._commit_is_ancestor(
            ancestor=landing_commit, descendant=current_main,
            label="whether main already contains prepared landing L"):
        return ("main already contains prepared landing L=" + landing_commit
                + " followed by newer commits; durable-state recovery is "
                  "required, not candidate revalidation")
    if daemon._commit_is_ancestor(
            ancestor=parent_commit, descendant=current_main,
            label="whether main preserves prepared landing parent M0"):
        candidate = (candidate_commit if candidate_commit is not None
                     else "not-applicable")
        return (
            daemon.STALE_INTEGRATION_REVALIDATION + ": C=" + candidate
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
        result = daemon._run_git(
            repository_root=daemon.AGENT_CWD["fable"],
            arguments=["merge-tree", "--write-tree", parent_commit,
                       candidate_commit],
            check=False)
    except daemon.PrimaryWorktreeError as exc:
        raise daemon.TicketCycleStateError(
            "cannot calculate the candidate squash: " + str(exc)) from exc
    if result.returncode != 0:
        raise daemon.TicketCycleStateError(
            "the audited candidate does not squash cleanly onto the "
            "landing parent")
    try:
        tree = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "calculated squash tree is not ASCII") from exc
    if daemon.FULL_COMMIT_RE.fullmatch(tree) is None:
        raise daemon.TicketCycleStateError(
            "git did not return one exact calculated squash tree")
    return tree


def _tree_with_backlog(tree, backlog):
    """Return ``tree`` with the Architect-sealed backlog bytes."""
    with daemon.tempfile.TemporaryDirectory(prefix="mailbox-backlog-index-") as tmp:
        environment = daemon.os.environ.copy()
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            environment.pop(name, None)
        environment["GIT_INDEX_FILE"] = daemon.os.path.join(tmp, "index")

        def git(arguments, input_bytes=None):
            result = daemon.subprocess.run(
                ["git", "-C", daemon.AGENT_CWD["fable"]] + arguments,
                env=environment, input=input_bytes, stdout=daemon.subprocess.PIPE,
                stderr=daemon.subprocess.PIPE, check=False)
            if result.returncode != 0:
                raise daemon.TicketCycleStateError(
                    "cannot add the sealed backlog to the landing tree")
            return result.stdout

        git(["read-tree", tree])
        blob = git(["hash-object", "-w", "--stdin"], backlog).decode(
            "ascii", errors="strict").strip()
        git(["update-index", "--add", "--cacheinfo", "100644", blob,
             daemon.BACKLOG_RELATIVE_PATH])
        result = git(["write-tree"]).decode("ascii", errors="strict").strip()
    if daemon.FULL_COMMIT_RE.fullmatch(result) is None:
        raise daemon.TicketCycleStateError("Git did not return one backlog tree")
    return result


def _landing_backlog(landing_commit):
    """Read the exact backlog that one prepared landing preserves."""
    result = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["show", landing_commit + ":" + daemon.BACKLOG_RELATIVE_PATH],
        check=False)
    if result.returncode != 0 or len(result.stdout) > daemon.MAX_BACKLOG_LEDGER_BYTES:
        raise daemon.TicketCycleStateError(
            "prepared landing has no valid tracked backlog")
    return result.stdout


def cycle_landing_ref(cycle_id):
    """Return the private crash-journal ref for one prepared landing."""
    return daemon.cycle_candidate_ref(cycle_id=cycle_id).rsplit("/", 1)[0] \
        + "/landing"


def reopen_decision_ref(cycle_id):
    """Return the crash-journal ref for one backlog-only reopening decision."""
    return daemon.cycle_candidate_ref(cycle_id=cycle_id).rsplit("/", 1)[0] \
        + "/reopen-decision"


def _reopen_decision_message(cycle_id, reviewed_landing, decision):
    """Name the reviewed landing and the Architect's final advice decision."""
    return (
        "Record Architect " + decision + " on Red Team reopening\n\n"
        + "Mailbox-Cycle: " + cycle_id + "\n"
        + "Mailbox-Reviewed-Landing: " + reviewed_landing + "\n")


def prepare_reopen_decision_landing(cycle_id, reviewed_landing, decision,
                                    backlog):
    """Create or reuse a backlog-only landing for one Red Team decision."""
    reference = daemon.reopen_decision_ref(cycle_id=cycle_id)
    prepared = daemon.git_ref_commit(reference=reference)
    current_main = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    if prepared is not None:
        parent = daemon._single_commit_parent(commit=prepared)
        if daemon._landing_backlog(landing_commit=prepared) != backlog:
            raise daemon.TicketCycleStateError(
                "saved reopening decision has different backlog bytes")
        problem = daemon._prepared_landing_main_problem(
            candidate_commit=None, landing_commit=prepared,
            parent_commit=parent, current_main=current_main)
        if problem is not None:
            raise daemon.RetryableArchitectLandingError(problem)
        return prepared, parent
    daemon._require_ancestor_or_same(
        ancestor=reviewed_landing, descendant=current_main,
        label="current main does not preserve the Red Team-reviewed landing")
    if daemon.worktree_head(worktree=daemon.AGENT_CWD["fable"]) != current_main:
        raise daemon.RetryableArchitectLandingError(
            "Architect primary must match current main before its reopening "
            "decision can land")
    parent_tree = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", current_main + "^{tree}"],
        label="reopening decision parent tree")
    tree = daemon._tree_with_backlog(tree=parent_tree, backlog=backlog)
    if tree == parent_tree:
        raise daemon.TicketCycleStateError(
            "Architect reopening decision did not change the backlog")
    result = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["commit-tree", tree, "-p", current_main, "-F", "-"],
        input_bytes=daemon._reopen_decision_message(
            cycle_id=cycle_id, reviewed_landing=reviewed_landing,
            decision=decision).encode("utf-8"), check=False)
    try:
        landing = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "reopening decision commit is not ASCII") from exc
    if result.returncode != 0 or daemon.FULL_COMMIT_RE.fullmatch(landing) is None:
        raise daemon.TicketCycleStateError(
            "cannot create the backlog-only reopening decision landing")
    update = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["update-ref", reference, landing, "0" * 40],
        check=False)
    if update.returncode != 0:
        raise daemon.TicketCycleStateError(
            "cannot save the reopening decision landing for recovery")
    return landing, current_main


def _candidate_commit_message(candidate_commit):
    """Return the exact human message that the Architect approved in C.

    Architect GO names the full candidate commit, so the commit message is
    already part of the immutable object under review. The daemon reads that
    message instead of inventing an internal-only subject for the squash
    landing.
    """
    if daemon.FULL_COMMIT_RE.fullmatch(candidate_commit) is None:
        raise daemon.TicketCycleStateError(
            "candidate message requires one full commit hash")
    result = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["cat-file", "commit", candidate_commit],
        check=False)
    if result.returncode != 0:
        raise daemon.TicketCycleStateError(
            "cannot read the approved candidate commit message")
    _headers, separator, message_bytes = result.stdout.partition(b"\n\n")
    if not separator:
        raise daemon.TicketCycleStateError(
            "approved candidate commit has no message separator")
    try:
        message = message_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "approved candidate commit message is not UTF-8") from exc
    if not message.strip() or "\0" in message:
        raise daemon.TicketCycleStateError(
            "approved candidate commit has no usable human message")
    reserved = daemon.re.compile(
        r"^mailbox-(?:cycle|candidate)[ \t]*:", daemon.re.IGNORECASE)
    if any(reserved.match(line) for line in message.splitlines()):
        raise daemon.TicketCycleStateError(
            "approved candidate commit message uses a reserved mailbox "
            "recovery label")
    return message


def _landing_commit_message(cycle_id, candidate_commit):
    """Copy candidate C's approved message and add exact recovery facts for L."""
    candidate_message = daemon._candidate_commit_message(candidate_commit)
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
    """Return landing L's parent after proving the journaled C -> L squash."""
    parent_commit = daemon._single_commit_parent(commit=landing_commit)
    backlog = daemon._landing_backlog(landing_commit=landing_commit)
    if expected_backlog is not None and backlog != expected_backlog:
        raise daemon.TicketCycleStateError(
            "prepared landing backlog differs from the Architect seal")
    expected_tree = daemon._tree_with_backlog(
        tree=daemon._exact_squash_tree(
            parent_commit=parent_commit, candidate_commit=candidate_commit),
        backlog=backlog)
    landing_tree = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", landing_commit + "^{tree}"],
        label="prepared landing tree")
    if landing_tree != expected_tree:
        raise daemon.TicketCycleStateError(
            "prepared landing tree is not the exact candidate squash")
    result = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["cat-file", "commit", landing_commit],
        check=False)
    if result.returncode != 0:
        raise daemon.TicketCycleStateError("cannot inspect prepared landing message")
    _headers, separator, message_bytes = result.stdout.partition(b"\n\n")
    if not separator:
        raise daemon.TicketCycleStateError(
            "prepared landing commit has no message separator")
    try:
        message = message_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "prepared landing message is not UTF-8") from exc
    if message != daemon._landing_commit_message(
            cycle_id=cycle_id, candidate_commit=candidate_commit):
        raise daemon.TicketCycleStateError(
            "prepared landing message does not bind its cycle and candidate")
    return parent_commit


def prepare_exact_squash_landing(cycle_id, candidate_commit, mode,
                                 sealed_backlog=None):
    """Create or reuse exact landing L without touching any checkout or branch."""
    tool_changes = sorted(
        path for path in daemon.candidate_changed_paths(
            base_commit=daemon.cycle_starting_commit(cycle_id),
            candidate_commit=candidate_commit,
            repository=daemon.AGENT_CWD["fable"])
        if path.startswith("ai/tools/"))
    if tool_changes:
        raise daemon.TicketCycleStateError(
            "external-maintainer-only ai/tools change cannot land: "
            + ", ".join(repr(path) for path in tool_changes))
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        ticket_state = daemon.read_ticket_cycle_state()
        active = ticket_state["active"].get(cycle_id)
        if (active is None or active["phase"] != "implementation"
                or active["mode"] != mode):
            raise daemon.TicketCycleStateError(
                "Architect GO has no matching implementation cycle")
        candidate_state = daemon.read_candidate_state()
        record = daemon.candidate_record_locked(
            cycle_id=cycle_id, ticket_state=ticket_state,
            candidate_state=candidate_state)
        if record is None or record["commit"] != candidate_commit:
            raise daemon.TicketCycleStateError(
                "Architect GO does not name the exact saved candidate")
        if sealed_backlog is None:
            sealed_backlog = daemon._validate_sealed_backlog(
                primary_worktree=daemon.AGENT_CWD["fable"])
        reference = daemon.cycle_landing_ref(cycle_id=cycle_id)
        prepared = daemon.git_ref_commit(reference=reference)
        current_main = daemon._exact_git_object(
            arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
            label="current main commit")
        if prepared is not None:
            parent = daemon._verify_prepared_landing(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                landing_commit=prepared, expected_backlog=sealed_backlog)
            main_problem = daemon._prepared_landing_main_problem(
                candidate_commit=candidate_commit,
                landing_commit=prepared, parent_commit=parent,
                current_main=current_main)
            if main_problem is not None:
                raise daemon.RetryableArchitectLandingError(main_problem)
            return prepared, parent, reference
        daemon._require_ancestor_or_same(
            ancestor=daemon.cycle_starting_commit(cycle_id),
            descendant=current_main,
            label="landing parent does not preserve the cycle base")
        tree = daemon._tree_with_backlog(
            tree=daemon._exact_squash_tree(
                parent_commit=current_main,
                candidate_commit=candidate_commit),
            backlog=sealed_backlog)
        parent_tree = daemon._exact_git_object(
            arguments=["rev-parse", "--verify", current_main + "^{tree}"],
            label="landing parent tree")
        if tree == parent_tree:
            raise daemon.TicketCycleStateError(
                "the audited candidate produces an empty squash landing")
        result = daemon._run_git(
            repository_root=daemon.AGENT_CWD["fable"],
            arguments=["commit-tree", tree, "-p", current_main,
                       "-F", "-"],
            check=False,
            input_bytes=daemon._landing_commit_message(
                cycle_id=cycle_id,
                candidate_commit=candidate_commit).encode("utf-8"))
        if result.returncode != 0:
            detail = result.stderr.decode(
                "utf-8", errors="replace").strip()[:500]
            raise daemon.TicketCycleStateError(
                "cannot create exact squash landing"
                + (": " + detail if detail else ""))
        try:
            landing = result.stdout.decode(
                "ascii", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise daemon.TicketCycleStateError(
                "created landing commit is not ASCII") from exc
        if daemon.FULL_COMMIT_RE.fullmatch(landing) is None:
            raise daemon.TicketCycleStateError(
                "commit-tree did not return one exact landing commit")
        update = daemon._run_git(
            repository_root=daemon.AGENT_CWD["fable"],
            arguments=["update-ref", reference, landing, "0" * 40],
            check=False)
        if update.returncode != 0:
            raise daemon.TicketCycleStateError(
                "cannot publish the exact landing crash journal")
        if daemon.git_ref_commit(reference=reference) != landing:
            raise daemon.TicketCycleStateError(
                "landing crash journal did not preserve the created commit")
        parent = daemon._verify_prepared_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=landing, expected_backlog=sealed_backlog)
        return landing, parent, reference
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def _user_checkout_status():
    """Return exact tracked/untracked status without refreshing the index."""
    environment = daemon.os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = daemon.subprocess.run(
        ["git", "-C", daemon.REPO_ROOT, "status", "--porcelain=v1", "-z",
         "--untracked-files=normal", "--ignore-submodules=none"],
        stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.PIPE, check=False,
        env=environment)
    if result.returncode != 0:
        raise daemon.TicketCycleStateError("cannot inspect the user's main checkout")
    return result.stdout


def land_prepared_commit_in_clean_user_checkout(
        landing, parent, candidate_commit=None):
    """Fast-forward a clean attached main checkout; never reset or force."""
    symbolic = daemon._run_git(
        repository_root=daemon.REPO_ROOT,
        arguments=["symbolic-ref", "-q", "HEAD"], check=False)
    if (symbolic.returncode != 0
            or symbolic.stdout.decode("utf-8", errors="replace").strip()
            != "refs/heads/main"):
        raise daemon.TicketCycleStateError(
            "the user's checkout is not attached to local main")
    current = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    if current == landing:
        if daemon._user_checkout_status():
            raise daemon.RetryableArchitectLandingError(
                "local main reached the prepared landing but the user's "
                "checkout is not clean")
        return
    main_problem = daemon._prepared_landing_main_problem(
        candidate_commit=candidate_commit, landing_commit=landing,
        parent_commit=parent, current_main=current)
    if main_problem is not None:
        raise daemon.RetryableArchitectLandingError(main_problem)
    if daemon._user_checkout_status():
        raise daemon.RetryableArchitectLandingError(
            "the user's main checkout has staged, unstaged, or untracked "
            "work; exact landing was preserved without touching it")
    result = daemon._run_git(
        repository_root=daemon.REPO_ROOT,
        arguments=["merge", "--ff-only", landing], check=False)
    if result.returncode != 0:
        raise daemon.TicketCycleStateError(
            "clean user main could not fast-forward to the prepared landing")
    after = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="landed main commit")
    if after != landing or daemon._user_checkout_status():
        raise daemon.TicketCycleStateError(
            "local main did not verify as one clean exact landing")


def _push_debt_path(landing):
    """Return the push-debt record path for one landing commit."""
    return daemon.os.path.join(daemon.RELAY_DIR, "pending-main-push-" + landing + ".txt")


def write_push_debt(landing, detail):
    """Durably record that exact local L still needs remote verification."""
    daemon.os.makedirs(daemon.RELAY_DIR, exist_ok=True)
    debt = daemon._push_debt_path(landing=landing)
    payload = (
        "Local main contains verified landing " + landing + ".\n"
        "Push is still required: git push origin " + landing
        + ":refs/heads/main\n"
        "Last push result: " + detail + "\n")
    descriptor, temporary = daemon.tempfile.mkstemp(
        prefix=".pending-main-push-", dir=daemon.RELAY_DIR)
    try:
        daemon.os.fchmod(descriptor, 0o600)
        with daemon.os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) \
                as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            daemon.os.fsync(stream.fileno())
        daemon.os.replace(temporary, debt)
        daemon.fsync_directory(directory=daemon.RELAY_DIR)
    finally:
        if descriptor >= 0:
            daemon.os.close(descriptor)
        try:
            daemon.os.remove(temporary)
        except FileNotFoundError:
            pass
    return debt


def push_exact_landing_or_record_debt(landing):
    """Attempt one non-force push; preserve a durable user action on failure.

    Arguments:
      landing = full commit that verified local ``main`` already contains.

    Returns:
      ``(True, "")`` when the exact push was verified on the remote,
      ``(False, detail)`` when a durable push-debt record was written, and
      ``(None, "")`` when the user chose ``--github no``: nothing contacts
      the remote, no debt is recorded for that choice, and debt records
      from earlier runs stay on disk untouched.
    """
    if not daemon.GITHUB_PUSH_ENABLED:
        print("local landing " + landing + " is verified; the GitHub push "
              "was skipped by user choice (--github no).")
        return None, ""
    command = ["git", "-C", daemon.AGENT_CWD["fable"], "push", "--porcelain",
               "origin", landing + ":refs/heads/main"]
    try:
        result = daemon.subprocess.run(
            command, stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.PIPE,
            check=False, timeout=120)
        pushed = result.returncode == 0
        detail = (result.stderr + result.stdout).decode(
            "utf-8", errors="replace").strip()[:2000]
    except (OSError, daemon.subprocess.TimeoutExpired) as exc:
        pushed = False
        detail = str(exc)
    if pushed:
        try:
            verify = daemon.subprocess.run(
                ["git", "-C", daemon.AGENT_CWD["fable"], "ls-remote", "--refs",
                 "origin", "refs/heads/main"],
                stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.PIPE,
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
        except (OSError, daemon.subprocess.TimeoutExpired) as exc:
            verified = False
            remote_detail = str(exc)
        if not verified:
            pushed = False
            detail = (detail + "\nremote verification: " + remote_detail) \
                .strip()
    debt = daemon._push_debt_path(landing=landing)
    if pushed:
        try:
            daemon.os.remove(debt)
        except FileNotFoundError:
            pass
        return True, ""
    daemon.write_push_debt(landing=landing, detail=detail)
    return False, detail


def retire_cycle_landing_ref(cycle_id, landing):
    """Retire only the exact crash-journal ref after receipt archival."""
    reference = daemon.cycle_landing_ref(cycle_id=cycle_id)
    current = daemon.git_ref_commit(reference=reference)
    if current is None:
        return
    if current != landing:
        raise daemon.TicketCycleStateError(
            "landing crash journal changed before retirement")
    daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["update-ref", "-d", reference, landing])


def recorded_landing_for_architect_go(cycle_id, mode):
    """Return durable landing L after a prior partial consume, or ``None``."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        completed = state["completed"].get(cycle_id)
        if completed is not None:
            return completed
        current = state["active"].get(cycle_id)
        if current is None:
            raise daemon.TicketCycleStateError(
                "Architect GO has no active or completed ticket cycle")
        if current["mode"] != mode:
            raise daemon.TicketCycleStateError(
                "Architect GO changed the ticket's saved mode")
        if current["phase"] == "implementation":
            return None
        if (current["phase"] in {
                "committed-awaiting-closure", "awaiting-redteam"}
                and current["commit"] is not None):
            return current["commit"]
        raise daemon.TicketCycleStateError(
            "Architect GO found an unsupported ticket-cycle phase")
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def redteam_closure_request_payload(cycle_id, landing):
    """Build the daemon-owned advisory review request for exact L."""
    return daemon.sol_ticket_payload(
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
    return daemon.sol_ticket_payload(
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
    for path in daemon.glob.glob(daemon.os.path.join(daemon.MAILBOX, "**", "*-to-sol.md"),
                          recursive=True):
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        if daemon.sol_ticket_kind(message=message) != "control-plane":
            continue
        found_cycle, found_candidate, _body, problem = (
            daemon._redteam_control_plane_envelope(message=message))
        if found_cycle != cycle_id:
            continue
        if problem is None and found_candidate == candidate:
            matches.append(path)
        else:
            conflicts.append(path)
    if conflicts or len(matches) > 1:
        raise daemon.TicketCycleStateError(
            "control-plane review identity conflicts with saved work")
    return matches[0] if matches else None


def publish_control_plane_review_request(cycle_id, candidate):
    """Publish once after D0 has durably recorded Architect GO(C)."""
    existing = daemon.matching_control_plane_review_request(
        cycle_id=cycle_id, candidate=candidate)
    if existing is not None:
        return existing
    lock_file = daemon.acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise daemon.RetryableArchitectLandingError(
            "cannot lock mailbox for protected Red Team review")
    try:
        existing = daemon.matching_control_plane_review_request(
            cycle_id=cycle_id, candidate=candidate)
        if existing is not None:
            return existing
        path = daemon.publish_message_locked(
            agent="sol", payload=daemon.control_plane_review_request_payload(
                cycle_id=cycle_id, candidate=candidate))
        if path is None:
            raise daemon.RetryableArchitectLandingError(
                "could not publish protected Red Team review")
        return path
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=lock_file)


def publish_control_plane_repair_request(cycle_id, candidate, mode):
    """Return a rejected protected candidate C to the Architect exactly once."""
    marker = "CONTROL-PLANE-REPAIR: " + candidate
    for path in daemon.glob.glob(
            daemon.os.path.join(daemon.MAILBOX, "**", "*-to-fable.md"),
            recursive=True):
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        found_cycle, found_mode, body, problem = daemon._ticket_flow_envelope(
            message=message)
        if (problem is None and found_cycle == cycle_id
                and found_mode == mode and marker in body):
            return path
    lock_file = daemon.acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise daemon.RetryableArchitectLandingError(
            "cannot lock mailbox for protected repair return")
    try:
        payload = (daemon.MAILBOX_FLOW_HEADER + "ticket\n"
                   + daemon.MAILBOX_CYCLE_HEADER + cycle_id + "\n"
                   + daemon.MAILBOX_MODE_HEADER + mode + "\n\n"
                   + marker + "\n\n"
                   + "The mandatory pre-landing Red Team review rejected "
                     "this exact candidate. Read its saved evidence, reopen "
                     "the ticket, and send one same-cycle Implementer repair "
                     "handoff. Do not send another GO for this candidate.\n")
        path = daemon.publish_message_locked(agent="fable", payload=payload)
        if path is None:
            raise daemon.RetryableArchitectLandingError(
                "could not publish protected repair return")
        return path
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=lock_file)


def control_plane_integration_request_payload(
        cycle_id, candidate, stale_landing, old_main, new_main, mode):
    """Build the same-cycle Architect check for one moved main branch."""
    for label, value in (("candidate", candidate),
                         ("stale landing", stale_landing),
                         ("old main", old_main), ("new main", new_main)):
        if daemon.FULL_COMMIT_RE.fullmatch(value) is None:
            raise ValueError("invalid " + label + " commit")
    if daemon.CYCLE_ID_RE.fullmatch(cycle_id) is None:
        raise ValueError("invalid integration-revalidation cycle")
    if mode not in daemon.ARCHITECT_COMMIT_MODES:
        raise ValueError("invalid integration-revalidation mode")
    return (
        daemon.MAILBOX_FLOW_HEADER + "ticket\n"
        + daemon.MAILBOX_CYCLE_HEADER + cycle_id + "\n"
        + daemon.MAILBOX_MODE_HEADER + mode + "\n\n"
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
    cycle_id, mode, body, problem = daemon._ticket_flow_envelope(message=message)
    if problem is not None or not body.startswith(
            "CONTROL-PLANE-INTEGRATION: REVALIDATE\n"):
        return None
    match = daemon.re.match(
        r"\ACONTROL-PLANE-INTEGRATION: REVALIDATE\r?\n"
        r"INTEGRATION-CANDIDATE: ([0-9a-f]{40})\r?\n"
        r"STALE-LANDING: ([0-9a-f]{40})\r?\n"
        r"OLD-MAIN: ([0-9a-f]{40})\r?\n"
        r"NEW-MAIN: ([0-9a-f]{40})\r?\n\r?\n",
        body)
    if match is None:
        raise daemon.TicketCycleStateError(
            "control-plane integration request has malformed identities")
    candidate, landing, old_main, new_main = match.groups()
    if daemon.IMPLEMENTER_CANDIDATE_LINE_RE.findall(body) != [candidate]:
        raise daemon.TicketCycleStateError(
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
    for path in daemon.glob.glob(
            daemon.os.path.join(daemon.MAILBOX, "**", "*-to-fable.md"),
            recursive=True):
        try:
            parsed = daemon.control_plane_integration_request(
                daemon.read_cycle_message(path=path))
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        if parsed is None:
            continue
        found = (parsed["cycle_id"], parsed["candidate"],
                 parsed["stale_landing"], parsed["old_main"],
                 parsed["new_main"])
        if found == expected:
            matches.append(path)
    if len(matches) > 1:
        raise daemon.TicketCycleStateError(
            "more than one integration request names the same stale event")
    return matches[0] if matches else None


def publish_control_plane_integration_request(
        cycle_id, candidate, stale_landing, old_main, new_main, mode):
    """Publish exactly one autonomous Architect integration audit."""
    arguments = dict(
        cycle_id=cycle_id, candidate=candidate,
        stale_landing=stale_landing, old_main=old_main, new_main=new_main)
    existing = daemon.matching_control_plane_integration_request(**arguments)
    if existing is not None:
        return existing
    lock_file = daemon.acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise daemon.RetryableArchitectLandingError(
            "cannot lock mailbox for integration revalidation")
    try:
        existing = daemon.matching_control_plane_integration_request(**arguments)
        if existing is not None:
            return existing
        path = daemon.publish_message_locked(
            agent="fable", payload=daemon.control_plane_integration_request_payload(
                mode=mode, **arguments))
        if path is None:
            raise daemon.RetryableArchitectLandingError(
                "could not publish integration revalidation")
        return path
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=lock_file)


def matching_redteam_closure_request(cycle_id, landing):
    """Return the sole saved Sol closure request, if one already exists."""
    matches = []
    conflicts = []
    for path in daemon.glob.glob(daemon.os.path.join(daemon.MAILBOX, "**", "*-to-sol.md"),
                          recursive=True):
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        if daemon.sol_ticket_kind(message=message) != "closure":
            continue
        returned_cycle, returned_landing, _body, problem = (
            daemon._redteam_closure_envelope(message=message))
        if returned_cycle != cycle_id:
            continue
        if problem is not None or returned_landing != landing:
            conflicts.append(path)
        else:
            matches.append(path)
    if conflicts:
        raise daemon.TicketCycleStateError(
            "another Sol closure request uses this cycle with a different "
            "or malformed landing")
    if len(matches) > 1:
        raise daemon.TicketCycleStateError(
            "more than one Sol closure request names this cycle and landing")
    return matches[0] if matches else None


def publish_redteam_closure_request(cycle_id, landing):
    """Publish or recover the one normal-mode Sol review of exact L."""
    existing = daemon.matching_redteam_closure_request(
        cycle_id=cycle_id, landing=landing)
    if existing is not None:
        return existing
    lock_file = daemon.acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise daemon.RetryableArchitectLandingError(
            "cannot lock the mailbox sequence for the Red Team request")
    try:
        existing = daemon.matching_redteam_closure_request(
            cycle_id=cycle_id, landing=landing)
        if existing is not None:
            return existing
        path = daemon.publish_message_locked(
            agent="sol",
            payload=daemon.redteam_closure_request_payload(
                cycle_id=cycle_id, landing=landing))
        if path is None:
            raise daemon.RetryableArchitectLandingError(
                "could not publish the Red Team request after 20 attempts")
        return path
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=lock_file)
