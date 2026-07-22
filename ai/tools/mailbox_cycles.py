"""Ticket-cycle state, registration, completion, and message publication.

A ticket cycle is one ticket's trip from its first Implementer
handoff to its recorded landing. Its name joins the ticket's backlog
anchor and full starting commit, as in ``TICKET-ANCHOR@COMMIT``. This
file owns the durable cycle record: the small JSON state files that
remember each active cycle's phase, mode, and landing commit, the
file locks that serialize every state read and write, admission
against a finite ``--cycle`` budget, and the numbered publication of
new mailbox messages. How a landing commit is created lives in the
landing part; this file records that the cycle reached it. Docstrings
here call the accepted squash landing L and the running watcher D0.

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
    "fsync_directory",
    "acquire_mailbox_sequence_lock",
    "release_mailbox_sequence_lock",
    "stable_regular_bytes",
    "unique_json_object",
    "report_role_token_exhaustion",
    "ticket_cycle_state_path",
    "empty_ticket_cycle_state",
    "empty_control_plane_state",
    "validate_control_plane_relationships",
    "control_plane_health_failure",
    "validate_ticket_cycle_state",
    "read_ticket_cycle_state",
    "active_implementer_runtime_problem",
    "write_ticket_cycle_state",
    "acquire_ticket_cycle_lock",
    "release_ticket_cycle_lock",
    "record_pending_ticket_cycle_return",
    "prepare_finite_watch_progress",
    "clear_finite_watch_progress",
    "finish_finite_watch_progress",
    "deliver_pending_ticket_cycle_returns",
    "cycle_ticket_anchor",
    "cycle_starting_commit",
    "require_open_backlog_ticket",
    "active_cycle_records_for_topology",
    "architect_admissions_for_topology",
    "finite_cycle_capacity_used",
    "register_ticket_cycle_message",
    "complete_ticket_cycle",
    "complete_reopen_ticket_cycle",
    "complete_protected_ticket_cycle",
    "record_architect_commit",
    "active_ticket_cycle_count",
    "read_cycle_message",
    "any_matching_redteam_receipt",
    "redteam_review_completes_cycle",
    "current_reopen_ticket",
    "architect_reopen_decision",
    "land_architect_reopen_decision",
    "_matching_journaled_notes_go",
    "_require_safe_noop_admin_recovery",
    "reconcile_architect_notes_admin_journals",
    "reconcile_inflight_architect_notes_admin",
    "reconcile_ticket_cycle_state",
    "publish_message_locked",
    "send",
    "recover_failed_maintenance_admission",
    "send_architect_notes_admin",
)


def fsync_directory(directory):
    """Make a completed same-directory namespace transition durable."""
    flags = daemon.os.O_RDONLY
    if hasattr(daemon.os, "O_DIRECTORY"):
        flags |= daemon.os.O_DIRECTORY
    descriptor = daemon.os.open(directory, flags)
    try:
        daemon.os.fsync(descriptor)
    finally:
        daemon.os.close(descriptor)


def acquire_mailbox_sequence_lock():
    """Acquire the publication lock without following or blocking on devices."""
    lock_path = daemon.os.path.join(daemon.MAILBOX, ".sequence.lock")
    try:
        parent = daemon.os.lstat(daemon.MAILBOX)
        if not daemon.stat.S_ISDIR(parent.st_mode):
            raise OSError("mailbox is not a regular directory")
        flags = daemon.os.O_RDWR | daemon.os.O_CREAT | daemon.os.O_NONBLOCK
        flags |= getattr(daemon.os, "O_CLOEXEC", 0)
        flags |= getattr(daemon.os, "O_NOFOLLOW", 0)
        descriptor = daemon.os.open(lock_path, flags, 0o600)
    except OSError as exc:
        print("mailbox publication blocked: sequence lock failed ("
              + str(exc) + ").")
        return None
    lock_file = daemon.os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        opened = daemon.os.fstat(lock_file.fileno())
        if not daemon.stat.S_ISREG(opened.st_mode):
            raise OSError("sequence lock is not a regular file")
        daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_EX)
        parent_after = daemon.os.lstat(daemon.MAILBOX)
        current = daemon.os.lstat(lock_path)
        if ((parent.st_dev, parent.st_ino)
                != (parent_after.st_dev, parent_after.st_ino)
                or not daemon.stat.S_ISDIR(parent_after.st_mode)
                or not daemon.stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)):
            raise OSError("sequence lock path changed")
    except OSError as exc:
        print("mailbox publication blocked: sequence lock failed ("
              + str(exc) + ").")
        try:
            daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()
        return None
    return lock_file


def release_mailbox_sequence_lock(lock_file):
    """Release a landing-debt sequence lock."""
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def stable_regular_bytes(path, maximum_bytes, label, missing_ok=False,
                         complete=True):
    """Read one bounded, nonblocking, unchanged file or leading prefix."""
    try:
        before = daemon.os.lstat(path)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ValueError(label + " disappeared before it could be read")
    except OSError as exc:
        raise ValueError("cannot inspect " + label + ": " + str(exc)) \
            from exc
    if not daemon.stat.S_ISREG(before.st_mode):
        raise ValueError(label + " is not a regular file")
    if complete and before.st_size > maximum_bytes:
        raise ValueError(label + " is too large")
    flags = daemon.os.O_RDONLY | daemon.os.O_NONBLOCK
    flags |= getattr(daemon.os, "O_CLOEXEC", 0)
    flags |= getattr(daemon.os, "O_NOFOLLOW", 0)
    try:
        descriptor = daemon.os.open(path, flags)
    except OSError as exc:
        raise ValueError("cannot open " + label + ": " + str(exc)) from exc
    try:
        opened = daemon.os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino)
        if (not daemon.stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or opened.st_size != before.st_size
                or opened.st_mtime_ns != before.st_mtime_ns):
            raise ValueError(label + " changed while it was opened")
        chunks = []
        remaining = maximum_bytes + 1 if complete else maximum_bytes
        while remaining > 0:
            chunk = daemon.os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining = remaining - len(chunk)
        raw = b"".join(chunks)
        after_open = daemon.os.fstat(descriptor)
    finally:
        daemon.os.close(descriptor)
    try:
        after_path = daemon.os.lstat(path)
    except OSError as exc:
        raise ValueError(label + " changed after it was read") from exc
    if ((complete and len(raw) > maximum_bytes)
            or (after_open.st_dev, after_open.st_ino) != identity
            or (after_path.st_dev, after_path.st_ino) != identity
            or not daemon.stat.S_ISREG(after_path.st_mode)
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


def ticket_cycle_state_path():
    """Return the ignored daemon-owned ticket-cycle state path."""
    return daemon.os.path.join(daemon.MAILBOX, daemon.TICKET_CYCLE_STATE_NAME)


def empty_ticket_cycle_state():
    """Return a new strict ticket-cycle state value."""
    return {
        "schema": daemon.TICKET_CYCLE_STATE_SCHEMA,
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
        raise daemon.TicketCycleStateError(
            "protected Red Team candidate and decision must appear together")
    if result is not None and (architect is None or architect != redteam):
        raise daemon.TicketCycleStateError(
            "protected decisions do not accept the same exact C")
    accepted = (architect is not None and redteam == architect
                and result == "ACCEPT-CONTROL-PLANE")

    if (shadow is None) != (shadow_evidence is None):
        raise daemon.TicketCycleStateError(
            "protected shadow result and evidence must appear together")
    if shadow is not None and not accepted:
        raise daemon.TicketCycleStateError(
            "protected shadow lacks exact-C acceptance")

    integration_values = (
        control["integration_main"], control["stale_landing"],
        control["stale_parent"])
    if integration is None:
        if any(value is not None for value in
               integration_values + (integration_evidence,)):
            raise daemon.TicketCycleStateError(
                "protected ticket has incomplete integration state")
    elif any(value is None for value in integration_values):
        raise daemon.TicketCycleStateError(
            "protected ticket lacks stale integration identity")
    elif not accepted:
        raise daemon.TicketCycleStateError(
            "protected integration lacks exact-C acceptance")
    elif integration == "STALE":
        if integration_evidence is not None or shadow is not None:
            raise daemon.TicketCycleStateError(
                "stale protected integration carries later evidence")
    elif integration_evidence is None:
        raise daemon.TicketCycleStateError(
            "protected integration revalidation lacks evidence")

    if (health is None) != (health_evidence is None):
        raise daemon.TicketCycleStateError(
            "protected health result and evidence must appear together")
    if health is not None and (not accepted or shadow != "PASSED"):
        raise daemon.TicketCycleStateError(
            "protected health check lacks accepted PASSED shadow evidence")
    if phase == "implementation" and health is not None:
        raise daemon.TicketCycleStateError(
            "unlanded protected ticket carries a health result")
    if phase == "awaiting-redteam":
        raise daemon.TicketCycleStateError(
            "protected ticket cannot enter ordinary Red Team closure")
    if phase == "committed-awaiting-closure" and (
            not accepted or shadow != "PASSED"):
        raise daemon.TicketCycleStateError(
            "landed protected ticket lacks exact-C acceptance and shadow")

    if completed_candidate is not None:
        if (architect != completed_candidate
                or redteam != completed_candidate
                or result != "ACCEPT-CONTROL-PLANE"):
            raise daemon.TicketCycleStateError(
                "completed control-plane history lacks exact accepted C")
        if shadow != "PASSED" or shadow_evidence is None:
            raise daemon.TicketCycleStateError(
                "completed control-plane history lacks PASSED shadow evidence")
        if health != "HEALTHY" or health_evidence is None:
            raise daemon.TicketCycleStateError(
                "completed control-plane history lacks HEALTHY evidence")
        if integration == "STALE":
            raise daemon.TicketCycleStateError(
                "completed control-plane history retains a stale integration")


def control_plane_health_failure(state=None):
    """Return the first durable failed promotion, if one exists."""
    current = daemon.read_ticket_cycle_state() if state is None else state
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
    if schema != daemon.TICKET_CYCLE_STATE_SCHEMA:
        raise daemon.TicketCycleStateError(
            "saved ticket-cycle state uses an unsupported old schema; "
            "stop every older watcher, preserve the state for inspection, "
            "then remove or reinitialize it deliberately")
    required = {"schema", "generation", "active", "completed",
                "pending_cycle_returns", "finite_watch",
                "architect_admissions"}
    optional = {"control_plane_history"}
    if (not isinstance(payload, dict) or not required.issubset(payload)
            or not set(payload).issubset(required | optional)):
        raise daemon.TicketCycleStateError("ticket-cycle state has wrong keys")
    generation = payload.get("generation")
    pending_cycle_returns = payload.get("pending_cycle_returns")
    if (isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0 or generation > daemon.MAX_CYCLE_COUNT):
        raise daemon.TicketCycleStateError("ticket-cycle state has invalid identity")
    if (isinstance(pending_cycle_returns, bool)
            or not isinstance(pending_cycle_returns, int)
            or pending_cycle_returns < 0
            or pending_cycle_returns > generation):
        raise daemon.TicketCycleStateError(
            "ticket-cycle state has invalid pending cycle returns")
    active = payload.get("active")
    completed = payload.get("completed")
    architect_admissions = payload.get("architect_admissions")
    finite_watch = payload.get("finite_watch")
    control_plane_history = payload.get("control_plane_history", {})
    if (not isinstance(active, dict) or not isinstance(completed, dict)
            or not isinstance(architect_admissions, dict)):
        raise daemon.TicketCycleStateError("ticket-cycle collections are invalid")
    if (len(active) + len(completed) + len(architect_admissions)
            > daemon.MAX_TICKET_CYCLE_RECORDS):
        raise daemon.TicketCycleStateError("ticket-cycle state has too many records")
    normalized_admissions = {}
    for request_name, record in architect_admissions.items():
        match = (daemon.PENDING_MESSAGE_RE.fullmatch(request_name)
                 if isinstance(request_name, str) else None)
        if (match is None or match.group(1) != "fable"
                or not isinstance(record, dict)
                or set(record) != {"mode", "sequence", "sha256"}):
            raise daemon.TicketCycleStateError(
                "invalid Architect ticket admission")
        sequence = record.get("sequence")
        digest = record.get("sha256")
        mode = record.get("mode")
        if (isinstance(sequence, bool) or not isinstance(sequence, int)
                or sequence != daemon.sequence_in_name(request_name)
                or not isinstance(digest, str)
                or daemon.re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or mode not in daemon.ARCHITECT_COMMIT_MODES):
            raise daemon.TicketCycleStateError(
                "Architect ticket admission has invalid fields")
        normalized_admissions[request_name] = {
            "mode": mode, "sequence": sequence, "sha256": digest}
    normalized_active = {}
    for cycle_id, record in active.items():
        required_record_keys = {"phase", "commit", "mode", "route"}
        optional_record_keys = {
            "path_scope", "ticket_class", "control_plane",
            "implementer_runtime"}
        if (not isinstance(cycle_id, str)
                or daemon.CYCLE_ID_RE.fullmatch(cycle_id) is None
                or not isinstance(record, dict)
                or not required_record_keys.issubset(record)
                or not set(record).issubset(
                    required_record_keys | optional_record_keys)):
            raise daemon.TicketCycleStateError("invalid active ticket-cycle record")
        phase = record.get("phase")
        commit = record.get("commit")
        mode = record.get("mode")
        route = record.get("route")
        path_scope = record.get("path_scope")
        ticket_class = record.get("ticket_class", "ordinary")
        control_plane = record.get("control_plane")
        implementer_runtime = record.get("implementer_runtime")
        if implementer_runtime is not None:
            if (not isinstance(implementer_runtime, dict)
                    or set(implementer_runtime) != {
                        "role_address", "provider", "model",
                        "context_limit", "compaction_limit"}):
                raise daemon.TicketCycleStateError(
                    "ticket Implementer runtime has wrong keys")
            try:
                implementer_runtime = daemon.implementer_runtime_record(
                    provider=implementer_runtime["provider"],
                    model=implementer_runtime["model"],
                    context_limit=implementer_runtime["context_limit"],
                    compaction_limit=implementer_runtime[
                        "compaction_limit"])
            except (ValueError, daemon.argparse.ArgumentTypeError) as exc:
                raise daemon.TicketCycleStateError(
                    "ticket Implementer runtime is invalid: "
                    + str(exc)) from exc
        if ticket_class not in daemon.TICKET_CLASSES:
            raise daemon.TicketCycleStateError("ticket class is invalid")
        if ticket_class == "protected-control-plane":
            if mode != "normal" or not isinstance(control_plane, dict) \
                    or set(control_plane) != set(daemon.empty_control_plane_state()):
                raise daemon.TicketCycleStateError(
                    "protected ticket lacks its exact two-key state")
            for field in ("architect_candidate", "redteam_candidate"):
                value = control_plane[field]
                if (value is not None
                        and (not isinstance(value, str)
                             or daemon.FULL_COMMIT_RE.fullmatch(value) is None)):
                    raise daemon.TicketCycleStateError(
                        "protected ticket has an invalid candidate decision")
            if control_plane["redteam_result"] not in (
                    None,) + daemon.CONTROL_PLANE_REVIEW_RESULTS:
                raise daemon.TicketCycleStateError(
                    "protected ticket has an invalid Red Team decision")
            if control_plane["shadow_status"] not in (
                    None, "PASSED", "FAILED"):
                raise daemon.TicketCycleStateError(
                    "protected ticket has invalid shadow state")
            if control_plane["integration_status"] not in (
                    None, "STALE", "REVALIDATED"):
                raise daemon.TicketCycleStateError(
                    "protected ticket has invalid integration state")
            if control_plane["health_status"] not in (
                    None, "HEALTHY", "CONTROL_PLANE_HEALTH_FAILED"):
                raise daemon.TicketCycleStateError(
                    "protected ticket has invalid health state")
            for field in ("integration_main", "stale_landing",
                          "stale_parent"):
                value = control_plane[field]
                if (value is not None
                        and (not isinstance(value, str)
                             or daemon.FULL_COMMIT_RE.fullmatch(value) is None)):
                    raise daemon.TicketCycleStateError(
                        "protected ticket has invalid integration identity")
            for field in ("shadow_evidence", "integration_evidence",
                          "health_evidence"):
                value = control_plane[field]
                if value is not None and (not isinstance(value, str)
                                          or not value
                                          or len(value) > 4096):
                    raise daemon.TicketCycleStateError(
                        "protected ticket has invalid evidence location")
        elif control_plane is not None:
            raise daemon.TicketCycleStateError(
                "ordinary ticket unexpectedly has protected state")
        if phase not in {"implementation", "committed-awaiting-closure",
                         "awaiting-redteam"}:
            raise daemon.TicketCycleStateError("invalid active ticket-cycle phase")
        if phase == "implementation" and commit is not None:
            raise daemon.TicketCycleStateError(
                "implementation cycle unexpectedly names landing L")
        if mode not in daemon.ARCHITECT_COMMIT_MODES:
            raise daemon.TicketCycleStateError("ticket-cycle mode is invalid")
        if route != "primary":
            raise daemon.TicketCycleStateError(
                "ticket-cycle mode conflicts with its Implementer route")
        if (phase != "implementation"
                and (not isinstance(commit, str)
                     or daemon.FULL_COMMIT_RE.fullmatch(commit) is None)):
            raise daemon.TicketCycleStateError(
                "committed ticket cycle lacks a full daemon-recorded "
                "landing L")
        expected_modes = {
            "committed-awaiting-closure": {"normal"},
            "awaiting-redteam": {"normal"},
        }
        if phase != "implementation" and mode not in expected_modes[phase]:
            raise daemon.TicketCycleStateError("ticket-cycle mode conflicts with phase")
        if ticket_class == "protected-control-plane":
            daemon.validate_control_plane_relationships(
                control=control_plane, phase=phase)
        if path_scope is not None:
            if (not isinstance(path_scope, list) or not path_scope
                    or len(path_scope) > 256
                    or any(not isinstance(path, str) for path in path_scope)
                    or path_scope != sorted(set(path_scope))):
                raise daemon.TicketCycleStateError("ticket path scope is invalid")
            for path in path_scope:
                parts = path.split("/")
                if (not parts or any(part in {"", ".", ".."} for part in parts)
                        or path.startswith("/") or "\\" in path
                        or any(mark in path for mark in "*?[]{}")
                        or not path.isprintable()):
                    raise daemon.TicketCycleStateError("ticket path scope is invalid")
        normalized = {
            "phase": phase, "commit": commit, "mode": mode,
            "route": route, "ticket_class": ticket_class}
        if "path_scope" in record:
            normalized["path_scope"] = path_scope
        if implementer_runtime is not None:
            normalized["implementer_runtime"] = dict(implementer_runtime)
        if ticket_class == "protected-control-plane":
            normalized["control_plane"] = dict(control_plane)
        normalized_active[cycle_id] = normalized
    normalized_completed = {}
    for cycle_id, commit in completed.items():
        if (not isinstance(cycle_id, str)
                or daemon.CYCLE_ID_RE.fullmatch(cycle_id) is None
                or not isinstance(commit, str)
                or daemon.FULL_COMMIT_RE.fullmatch(commit) is None):
            raise daemon.TicketCycleStateError("invalid completed ticket-cycle record")
        if cycle_id in normalized_active:
            raise daemon.TicketCycleStateError(
                "ticket cycle is both active and completed")
        normalized_completed[cycle_id] = commit
    if (not isinstance(control_plane_history, dict)
            or len(control_plane_history) > daemon.MAX_TICKET_CYCLE_RECORDS):
        raise daemon.TicketCycleStateError("control-plane history is invalid")
    normalized_control_history = {}
    for cycle_id, record in control_plane_history.items():
        if (cycle_id not in normalized_completed
                or not isinstance(record, dict)
                or set(record) != {"candidate", "landing", "control_plane"}
                or not isinstance(record.get("candidate"), str)
                or daemon.FULL_COMMIT_RE.fullmatch(record["candidate"]) is None
                or record.get("landing") != normalized_completed[cycle_id]
                or not isinstance(record.get("control_plane"), dict)
                or set(record["control_plane"])
                != set(daemon.empty_control_plane_state())):
            raise daemon.TicketCycleStateError(
                "completed control-plane record is invalid")
        control = record["control_plane"]
        for field in ("architect_candidate", "redteam_candidate"):
            value = control[field]
            if (not isinstance(value, str)
                    or daemon.FULL_COMMIT_RE.fullmatch(value) is None):
                raise daemon.TicketCycleStateError(
                    "completed control-plane decision is invalid")
        if control["redteam_result"] not in daemon.CONTROL_PLANE_REVIEW_RESULTS:
            raise daemon.TicketCycleStateError(
                "completed control-plane Red Team result is invalid")
        if control["shadow_status"] not in (None, "PASSED", "FAILED"):
            raise daemon.TicketCycleStateError(
                "completed control-plane shadow result is invalid")
        if control["integration_status"] not in (
                None, "STALE", "REVALIDATED"):
            raise daemon.TicketCycleStateError(
                "completed control-plane integration result is invalid")
        if control["health_status"] not in (
                None, "HEALTHY", "CONTROL_PLANE_HEALTH_FAILED"):
            raise daemon.TicketCycleStateError(
                "completed control-plane health result is invalid")
        for field in ("integration_main", "stale_landing",
                      "stale_parent"):
            value = control[field]
            if (value is not None
                    and (not isinstance(value, str)
                         or daemon.FULL_COMMIT_RE.fullmatch(value) is None)):
                raise daemon.TicketCycleStateError(
                    "completed control-plane integration identity is invalid")
        for field in ("shadow_evidence", "integration_evidence",
                      "health_evidence"):
            value = control[field]
            if value is not None and (not isinstance(value, str)
                                      or not value or len(value) > 4096):
                raise daemon.TicketCycleStateError(
                    "completed control-plane evidence is invalid")
        daemon.validate_control_plane_relationships(
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
            raise daemon.TicketCycleStateError(
                "finite-watch progress has invalid keys")
        limit = finite_watch.get("limit")
        finite_completed = finite_watch.get("completed")
        status = finite_watch.get("status")
        topology = finite_watch.get("topology")
        if (isinstance(limit, bool) or not isinstance(limit, int)
                or limit <= 0 or limit > daemon.MAX_CYCLE_COUNT
                or isinstance(finite_completed, bool)
                or not isinstance(finite_completed, int)
                or finite_completed < 0 or finite_completed > limit
                or status not in {"active", "complete"}
                or topology not in daemon.ARCHITECT_COMMIT_MODES
                or (status == "complete" and finite_completed != limit)
                or (status == "complete" and pending_cycle_returns != 0)):
            raise daemon.TicketCycleStateError(
                "finite-watch progress is invalid")
        normalized_finite = {
            "limit": limit, "completed": finite_completed,
            "status": status, "topology": topology}
    return {
        "schema": daemon.TICKET_CYCLE_STATE_SCHEMA,
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
        raw = daemon.stable_regular_bytes(
            path=daemon.ticket_cycle_state_path(),
            maximum_bytes=daemon.MAX_TICKET_CYCLE_STATE_BYTES,
            label="ticket-cycle state", missing_ok=True)
    except (OSError, ValueError) as exc:
        raise daemon.TicketCycleStateError(str(exc)) from exc
    if raw is None:
        return daemon.empty_ticket_cycle_state()
    try:
        payload = daemon.json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=daemon.unique_json_object)
    except (UnicodeDecodeError, daemon.json.JSONDecodeError, RecursionError,
            OverflowError, ValueError) as exc:
        raise daemon.TicketCycleStateError(
            "ticket-cycle state is invalid JSON") from exc
    return daemon.validate_ticket_cycle_state(payload=payload)


def active_implementer_runtime_problem(
        *, provider, model, compaction_limit, context_limit=None):
    """Refuse a restart that would silently change an active ticket."""
    for cycle_id, record in daemon.read_ticket_cycle_state()["active"].items():
        saved = record.get("implementer_runtime")
        if record["phase"] != "implementation" or saved is None:
            continue
        if (saved["provider"] != provider or saved["model"] != model
                or saved["compaction_limit"] != compaction_limit
                or (context_limit is not None
                    and saved["context_limit"] != context_limit)):
            return ("active ticket " + cycle_id + " is bound to "
                    + saved["provider"] + "/" + saved["model"]
                    + " with context " + str(saved["context_limit"])
                    + " and compaction "
                    + str(saved["compaction_limit"])
                    + "; restart with that exact Implementer runtime or "
                    "begin a new Architect-visible cycle")
    return None


def write_ticket_cycle_state(state):
    """Publish strict cycle state with an atomic replacement and fsync."""
    normalized = daemon.validate_ticket_cycle_state(payload=state)
    daemon.os.makedirs(daemon.MAILBOX, exist_ok=True)
    payload = (daemon.json.dumps(normalized, sort_keys=True, separators=(",", ":"))
               + "\n").encode("utf-8")
    if len(payload) > daemon.MAX_TICKET_CYCLE_STATE_BYTES:
        raise daemon.TicketCycleStateError("ticket-cycle state exceeds its limit")
    handle, temporary = daemon.tempfile.mkstemp(
        prefix=".ticket-cycle-", dir=daemon.MAILBOX)
    try:
        daemon.os.fchmod(handle, 0o600)
        with daemon.os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            daemon.os.fsync(stream.fileno())
        daemon.os.replace(temporary, daemon.ticket_cycle_state_path())
        daemon.fsync_directory(directory=daemon.MAILBOX)
    finally:
        if daemon.os.path.exists(temporary):
            daemon.os.remove(temporary)


def acquire_ticket_cycle_lock():
    """Serialize state changes made by independent working-directory lanes."""
    daemon.os.makedirs(daemon.MAILBOX, exist_ok=True)
    path = daemon.os.path.join(daemon.MAILBOX, daemon.TICKET_CYCLE_LOCK_NAME)
    try:
        lock_file = open(path, "a+", encoding="utf-8")
        daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_EX)
        return lock_file
    except OSError as exc:
        raise daemon.TicketCycleStateError(
            "cannot lock ticket-cycle state: " + str(exc)) from exc


def release_ticket_cycle_lock(lock_file):
    """Release one ticket-cycle state lock."""
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def record_pending_ticket_cycle_return(state):
    """Persist one return that a live watch has not counted yet.

    Finite ``--once`` calls have no cycle controller, so their completions
    must not become credit for a later watch. A live watch records the return
    in the same atomic state replacement as completion. If the process dies
    before its in-memory controller is updated, the next watch can replay
    exactly this durable count.
    """
    controller = daemon._ACTIVE_WATCH_RENDEZVOUS
    if controller is None:
        return
    finite_limit = controller.ticket_cycle_limit_value()
    if finite_limit is not None:
        saved = state["finite_watch"]
        if (saved is None or saved["status"] != "active"
                or saved["limit"] != finite_limit
                or saved["topology"]
                != controller.ticket_cycle_topology_value()):
            raise daemon.TicketCycleStateError(
                "ticket completion does not match the active finite-watch "
                "topology")
    if state["pending_cycle_returns"] >= daemon.MAX_CYCLE_COUNT:
        raise daemon.TicketCycleStateError("pending ticket-cycle return count is full")
    state["pending_cycle_returns"] = state["pending_cycle_returns"] + 1


def prepare_finite_watch_progress(limit, topology):
    """Start or resume the durable progress record for ``--cycle N``."""
    if (isinstance(limit, bool) or not isinstance(limit, int)
            or limit <= 0 or limit > daemon.MAX_CYCLE_COUNT):
        raise daemon.TicketCycleStateError("finite watch limit is invalid")
    if topology not in daemon.ARCHITECT_COMMIT_MODES:
        raise daemon.TicketCycleStateError("finite watch topology is invalid")
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        saved = state["finite_watch"]
        if saved is None or saved["status"] == "complete":
            saved = {"limit": limit, "completed": 0, "status": "active",
                     "topology": topology}
        else:
            if saved["topology"] != topology:
                raise daemon.TicketCycleStateError(
                    "interrupted finite watch belongs to topology "
                    + saved["topology"] + ", not " + topology)
            if saved["completed"] + state["pending_cycle_returns"] > limit:
                raise daemon.TicketCycleStateError(
                    "interrupted finite watch already completed more than "
                    "the requested --cycle limit")
            saved = dict(saved, limit=limit)
        state["finite_watch"] = saved
        daemon.write_ticket_cycle_state(state=state)
        return saved["completed"]
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def clear_finite_watch_progress(topology):
    """Abandon an interrupted finite limit when this run is not finite."""
    if topology not in daemon.ARCHITECT_COMMIT_MODES:
        raise daemon.TicketCycleStateError("watch topology is invalid")
    if not daemon.os.path.exists(daemon.ticket_cycle_state_path()):
        return
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        saved = state["finite_watch"]
        if (saved is not None and saved["status"] == "active"
                and saved["topology"] != topology):
            raise daemon.TicketCycleStateError(
                "interrupted finite watch belongs to topology "
                + saved["topology"] + ", not " + topology)
        if saved is not None:
            state["finite_watch"] = None
            daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def finish_finite_watch_progress(limit, completed, topology):
    """Mark a proved finite run complete before its success is reported."""
    if topology not in daemon.ARCHITECT_COMMIT_MODES:
        raise daemon.TicketCycleStateError("finite watch topology is invalid")
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        saved = state["finite_watch"]
        if (saved is None or saved["status"] != "active"
                or saved["limit"] != limit
                or saved["completed"] != completed
                or saved["topology"] != topology
                or completed != limit
                or state["pending_cycle_returns"] != 0
                or any(record["mode"] == topology for record in
                       state["architect_admissions"].values())):
            raise daemon.TicketCycleStateError(
                "finite-watch progress does not prove a clean exit")
        state["finite_watch"] = dict(saved, status="complete")
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def deliver_pending_ticket_cycle_returns():
    """Count and acknowledge every durable return for the active watch.

    The state lock serializes concurrent ticket completions.
    The controller is updated before the acknowledgement is written. A crash
    in that narrow gap replays the return into the replacement process; it
    can never lose the return from both durable and in-memory state.
    """
    controller = daemon._ACTIVE_WATCH_RENDEZVOUS
    if controller is None:
        return 0
    # A clean watch with no prior cycle state has nothing to deliver.  Avoid
    # creating a lock file merely to prove that absence; the dispatch lock
    # prevents another watcher from completing a cycle during startup.
    if not daemon.os.path.exists(daemon.ticket_cycle_state_path()):
        return 0
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
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
                raise daemon.TicketCycleStateError(
                    "durable finite-watch progress does not match the live "
                    "cycle controller")
            if pending:
                saved = dict(saved, completed=saved["completed"] + pending)
                state["finite_watch"] = saved
                state["pending_cycle_returns"] = 0
                # Durable progress is published before RAM is advanced. A
                # crash between these operations resumes from this value.
                daemon.write_ticket_cycle_state(state=state)
                for _ in range(pending):
                    daemon._ticket_cycle_completed()
        else:
            if pending:
                for _ in range(pending):
                    daemon._ticket_cycle_completed()
                state["pending_cycle_returns"] = 0
                daemon.write_ticket_cycle_state(state=state)
        return pending
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def cycle_ticket_anchor(cycle_id):
    """Return the backlog anchor carried by one validated cycle id."""
    return cycle_id.split("@", 1)[0]


def cycle_starting_commit(cycle_id):
    """Return the full starting commit carried by one validated cycle id."""
    return cycle_id.split("@", 1)[1]


def require_open_backlog_ticket(ticket_anchor):
    """Prove one cycle begins from exactly one indexed Open ticket."""
    lines, problem = daemon.verified_backlog_lines()
    if problem is not None:
        raise daemon.TicketCycleStateError(problem)
    indexed = []
    for line in lines:
        match = daemon.OPEN_BACKLOG_TICKET_RE.fullmatch(line)
        if match is not None and match.group(4) == ticket_anchor:
            indexed.append(line)
    details = [line for line in lines
               if line == '<a id="' + ticket_anchor + '"></a>']
    if len(indexed) != 1 or len(details) != 1:
        raise daemon.TicketCycleStateError(
            "ticket cycle must begin from exactly one indexed Open backlog "
            "ticket: " + ticket_anchor)


def active_cycle_records_for_topology(state, skip_redteam=False):
    """Return active records this watch can advance."""
    return [
        record for record in state["active"].values()
        if daemon.ticket_cycle_mode_is_enabled(
            mode=record["mode"], skip_redteam=skip_redteam)]


def architect_admissions_for_topology(state, skip_redteam=False):
    """Return public Architect requests already charged to this watch."""
    return [
        record for record in state["architect_admissions"].values()
        if daemon.ticket_cycle_mode_is_enabled(
            mode=record["mode"], skip_redteam=skip_redteam)]


def finite_cycle_capacity_used(state, skip_redteam=False):
    """Return every completed, admitted, or active charged ticket."""
    controller = daemon._ACTIVE_WATCH_RENDEZVOUS
    if controller is None or controller.ticket_cycle_limit_value() is None:
        return None
    topology = daemon.canonical_ticket_cycle_topology(skip_redteam=skip_redteam)
    if controller.ticket_cycle_topology_value() != topology:
        raise daemon.TicketCycleStateError(
            "finite cycle capacity was requested for another topology")
    saved = state["finite_watch"]
    if (saved is None or saved["status"] != "active"
            or saved["topology"] != topology):
        raise daemon.TicketCycleStateError(
            "finite cycle capacity lacks matching durable progress")
    return (controller.completed_ticket_cycles()
            + state["pending_cycle_returns"]
            + len(daemon.active_cycle_records_for_topology(
                state=state, skip_redteam=skip_redteam))
            + len(daemon.architect_admissions_for_topology(
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
    class_problem = daemon.ticket_class_configuration_problem(
        ticket_class=ticket_class, skip_redteam=skip_redteam)
    if class_problem is not None:
        raise daemon.TicketCycleStateError(class_problem)
    if agent in {"fable", "opus"} and message.startswith(daemon.MAILBOX_FLOW_HEADER):
        cycle_id, requested_mode, _, problem = daemon._ticket_flow_envelope(
            message=message)
        if problem is not None:
            raise daemon.TicketCycleStateError(problem)
        phase = "implementation"
    elif (agent == "sol" and daemon.sol_ticket_kind(message=message) == "closure"):
        if skip_redteam:
            raise daemon.TicketCycleStateError(
                "this watch does not dispatch Red Team closures")
        cycle_id, accepted_commit, _, problem = (
            daemon._redteam_closure_envelope(message=message))
        if problem is not None:
            raise daemon.TicketCycleStateError(problem)
        phase = "awaiting-redteam"
    else:
        if architect_admission is not None:
            raise daemon.TicketCycleStateError(
                "Architect admission does not name an Implementer flow")
        return ((None, None, False) if return_reservation
                else (None, None))

    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        completed_commit = state["completed"].get(cycle_id)
        if completed_commit is not None:
            raise daemon.TicketCycleStateError(
                "ticket cycle was already completed at " + completed_commit)
        current = state["active"].get(cycle_id)
        if phase == "implementation":
            requested_route = "primary"
            expected_primary_mode = "two-role" if skip_redteam else "normal"
            if not daemon.ticket_cycle_mode_is_enabled(
                    mode=requested_mode, skip_redteam=skip_redteam):
                raise daemon.TicketCycleStateError(
                    "ticket exchange belongs to another watch role")
            if agent == "opus" and requested_mode != expected_primary_mode:
                raise daemon.TicketCycleStateError(
                    "the primary Implementer must use MAILBOX-MODE: "
                    + expected_primary_mode + " for this watch")
            if current is None:
                if agent == "fable":
                    raise daemon.TicketCycleStateError(
                        "the Architect route cannot invent a cycle before an "
                        "Implementer handoff")
                if (architect_admission is None
                        and implementer_request_name is not None):
                    request_match = daemon.PENDING_MESSAGE_RE.fullmatch(
                        implementer_request_name)
                    if (request_match is None
                            or request_match.group(1) != "opus"):
                        raise daemon.TicketCycleStateError(
                            "invalid Implementer request identity")
                    flow_name, flow_digest, admission_problem = (
                        daemon._ticket_architect_admission(message=message))
                    if admission_problem is not None:
                        raise daemon.TicketCycleStateError(admission_problem)
                    if flow_name is not None:
                        admission_record = state[
                            "architect_admissions"].get(flow_name)
                        if (admission_record is None
                                or admission_record["sha256"]
                                != flow_digest
                                or admission_record["sequence"]
                                >= daemon.message_sequence(
                                    implementer_request_name)):
                            raise daemon.TicketCycleStateError(
                                "Implementer flow names no exact earlier "
                                "public Architect admission")
                        architect_admission = daemon.architect_admission_token(
                            request_name=flow_name, digest=flow_digest)
                admission = None
                if architect_admission is not None:
                    admission_name, admission_digest = (
                        daemon.split_architect_admission_token(
                            token=architect_admission))
                    flow_name, flow_digest, admission_problem = (
                        daemon._ticket_architect_admission(message=message))
                    if admission_problem is not None:
                        raise daemon.TicketCycleStateError(admission_problem)
                    if (flow_name != admission_name
                            or flow_digest != admission_digest):
                        raise daemon.TicketCycleStateError(
                            "Implementer flow does not carry its exact "
                            "public Architect admission")
                    admission = state["architect_admissions"].get(
                        admission_name)
                    if admission is None:
                        raise daemon.TicketCycleStateError(
                            "Implementer flow lacks its exact public "
                            "Architect admission")
                    if admission["sha256"] != admission_digest:
                        raise daemon.TicketCycleStateError(
                            "Implementer flow admission digest changed")
                    if admission["mode"] != requested_mode:
                        raise daemon.TicketCycleStateError(
                            "Implementer flow changed its admitted watch "
                            "topology")
                if daemon.architect_notes_transition_pending():
                    raise daemon.TicketCycleLimitDeferred(
                        "a permanent-note admin turn or P landing is still "
                        "pending; no newer ticket may be admitted")
                daemon.require_open_backlog_ticket(
                    ticket_anchor=daemon.cycle_ticket_anchor(cycle_id))
                starting_commit = daemon.cycle_starting_commit(cycle_id)
                if not daemon.git_commit_exists(commit=starting_commit):
                    raise daemon.TicketCycleStateError(
                        "ticket cycle starting commit does not exist: "
                        + starting_commit)
                current_main = daemon._exact_git_object(
                    arguments=["rev-parse", "--verify",
                               "refs/heads/main^{commit}"],
                    label="current main commit")
                if starting_commit != current_main:
                    raise daemon.TicketCycleLimitDeferred(
                        "ticket cycle base is not the exact current main "
                        "commit; wait for any earlier P/L landing, then "
                        "reissue the Architect handoff from that commit")
                if admission is None:
                    used = daemon.finite_cycle_capacity_used(
                        state=state, skip_redteam=skip_redteam)
                    controller = daemon._ACTIVE_WATCH_RENDEZVOUS
                    if (used is not None
                            and used >= controller.ticket_cycle_limit_value()):
                        raise daemon.TicketCycleLimitDeferred(
                            "the finite watch has already reserved all "
                            + str(controller.ticket_cycle_limit_value())
                            + " ticket cycle(s)")
                state["active"][cycle_id] = {
                    "phase": "implementation", "commit": None,
                    "mode": requested_mode, "route": requested_route,
                    "ticket_class": ticket_class,
                    "implementer_runtime": dict(daemon.IMPLEMENTER_RUNTIME),
                    "path_scope": (sorted(path_scope)
                                   if path_scope is not None else None),
                    "control_plane": (
                        daemon.empty_control_plane_state()
                        if ticket_class == "protected-control-plane"
                        else None)}
                if admission is not None:
                    del state["architect_admissions"][admission_name]
                created = True
            elif current["phase"] != "implementation":
                raise daemon.TicketCycleStateError(
                    "ticket exchange arrived after the daemon recorded "
                    "landing L")
            elif architect_admission is not None:
                raise daemon.TicketCycleStateError(
                    "public Architect admission was already converted")
            else:
                saved_runtime = current.get("implementer_runtime")
                if (saved_runtime is not None
                        and saved_runtime != daemon.IMPLEMENTER_RUNTIME):
                    raise daemon.TicketCycleStateError(
                        "active ticket is bound to Implementer "
                        + saved_runtime["provider"] + "/"
                        + saved_runtime["model"] + " with model context "
                        + str(saved_runtime["context_limit"])
                        + " and compaction "
                        + str(saved_runtime["compaction_limit"])
                        + "; start this watch with the same runtime or create "
                        "a new Architect-visible cycle")
                if saved_runtime is None and agent == "opus":
                    state["active"][cycle_id] = dict(
                        current,
                        implementer_runtime=dict(daemon.IMPLEMENTER_RUNTIME))
                    current = state["active"][cycle_id]
                if (current["mode"] != requested_mode
                        or current["route"] != requested_route):
                    raise daemon.TicketCycleStateError(
                        "ticket exchange changed its saved mode or "
                        "Implementer route")
                if (agent == "opus"
                        and current.get("ticket_class", "ordinary")
                        != ticket_class):
                    raise daemon.TicketCycleStateError(
                        "ticket exchange changed its frozen Ticket class")
                if agent == "opus" and path_scope is not None:
                    frozen = current.get("path_scope")
                    proposed = sorted(path_scope)
                    if frozen is not None and frozen != proposed:
                        raise daemon.TicketCycleStateError(
                            "Implementer handoff changed the frozen ticket "
                            "path scope")
                    if frozen is None:
                        state["active"][cycle_id] = dict(
                            current, path_scope=proposed)
        else:
            if current is None:
                raise daemon.TicketCycleStateError(
                    "Red Team closure has no recorded daemon landing")
            if current["phase"] == "implementation":
                raise daemon.TicketCycleStateError(
                    "Red Team closure arrived before the daemon landing was "
                    "recorded")
            if (current["phase"] == "awaiting-redteam"
                    and current["commit"] != accepted_commit):
                raise daemon.TicketCycleStateError(
                    "ticket cycle names two different daemon landings")
            if (current is not None
                    and current["phase"] == "committed-awaiting-closure"
                    and current["commit"] != accepted_commit):
                raise daemon.TicketCycleStateError(
                    "Red Team closure does not name the recorded daemon "
                    "landing")
            if current["mode"] != "normal" or current["route"] != "primary":
                raise daemon.TicketCycleStateError(
                    "only a normal primary ticket receives Red Team closure")
            state["active"][cycle_id] = dict(
                current, phase="awaiting-redteam",
                commit=accepted_commit)
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)
    result = (cycle_id, accepted_commit)
    return result + (created,) if return_reservation else result


def complete_ticket_cycle(cycle_id, accepted_commit):
    """Move one correlated Red Team return from active to completed state."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        prior = state["completed"].get(cycle_id)
        if prior is not None:
            if prior == accepted_commit:
                return False
            raise daemon.TicketCycleStateError(
                "ticket cycle was completed at another commit")
        current = state["active"].get(cycle_id)
        if (current is None or current["phase"] != "awaiting-redteam"
                or current["commit"] != accepted_commit):
            raise daemon.TicketCycleStateError(
                "Red Team return does not match an awaiting ticket cycle")
        del state["active"][cycle_id]
        state["completed"][cycle_id] = accepted_commit
        state["generation"] = state["generation"] + 1
        daemon.record_pending_ticket_cycle_return(state=state)
        daemon.write_ticket_cycle_state(state=state)
        return True
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def complete_reopen_ticket_cycle(cycle_id, reviewed_landing,
                                 decision_landing):
    """Complete a reopened review only after its backlog decision lands."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        prior = state["completed"].get(cycle_id)
        if prior is not None:
            if prior == decision_landing:
                return False
            raise daemon.TicketCycleStateError(
                "reopening decision completed at another landing")
        current = state["active"].get(cycle_id)
        if (current is None or current["phase"] != "awaiting-redteam"
                or current["commit"] != reviewed_landing):
            raise daemon.TicketCycleStateError(
                "reopening decision does not match its reviewed landing")
        del state["active"][cycle_id]
        state["completed"][cycle_id] = decision_landing
        state["generation"] += 1
        daemon.record_pending_ticket_cycle_return(state=state)
        daemon.write_ticket_cycle_state(state=state)
        return True
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def complete_protected_ticket_cycle(cycle_id, candidate_commit, landing):
    """Complete one two-key ticket after D0 records healthy L."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        prior = state["completed"].get(cycle_id)
        if prior is not None:
            if prior == landing:
                return False
            raise daemon.TicketCycleStateError(
                "protected ticket completed at another landing")
        active = state["active"].get(cycle_id)
        if (active is None
                or active.get("ticket_class") != "protected-control-plane"
                or active["phase"] != "committed-awaiting-closure"
                or active["commit"] != landing):
            raise daemon.TicketCycleStateError(
                "protected completion lacks its daemon landing")
        control = active["control_plane"]
        if (control["architect_candidate"] != candidate_commit
                or control["redteam_candidate"] != candidate_commit
                or control["redteam_result"] != "ACCEPT-CONTROL-PLANE"
                or control["shadow_status"] != "PASSED"
                or control["health_status"] != "HEALTHY"):
            raise daemon.TicketCycleStateError(
                "protected completion lacks both keys and healthy evidence")
        del state["active"][cycle_id]
        state["completed"][cycle_id] = landing
        state["control_plane_history"][cycle_id] = {
            "candidate": candidate_commit,
            "landing": landing,
            "control_plane": dict(control),
        }
        state["generation"] += 1
        daemon.record_pending_ticket_cycle_return(state=state)
        daemon.write_ticket_cycle_state(state=state)
        return True
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def record_architect_commit(cycle_id, accepted_commit, mode):
    """Record the daemon squash landing accepted by one Architect GO.

    Returns ``1`` when a two-role ticket completes at this landing record. A
    normal ticket returns ``0`` and waits for its correlated Red Team pass.
    """
    if (not isinstance(cycle_id, str)
            or daemon.CYCLE_ID_RE.fullmatch(cycle_id) is None
            or not isinstance(accepted_commit, str)
            or daemon.FULL_COMMIT_RE.fullmatch(accepted_commit) is None
            or mode not in daemon.ARCHITECT_COMMIT_MODES):
        raise daemon.TicketCycleStateError("invalid daemon landing record")
    lock_file = daemon.acquire_ticket_cycle_lock()
    completed_now = 0
    try:
        state = daemon.read_ticket_cycle_state()
        if cycle_id in state["completed"]:
            if state["completed"][cycle_id] == accepted_commit:
                return 0
            raise daemon.TicketCycleStateError(
                "Architect GO cycle was completed at another landing")
        current = state["active"].get(cycle_id)
        if (current is not None and current["phase"] != "implementation"
                and current["commit"] == accepted_commit
                and current["mode"] == mode):
            return 0
        if current is None or current["phase"] != "implementation":
            raise daemon.TicketCycleStateError(
                "daemon landing record has no active implementation cycle")
        if current["mode"] != mode:
            raise daemon.TicketCycleStateError(
                "Architect GO changed the ticket's saved mode")
        candidate_commit = daemon.require_architect_landing_locked(
            cycle_id=cycle_id, landing_commit=accepted_commit,
            ticket_state=state)
        # Git ancestry proves a new landing. An exact landing record already
        # represented by durable completed/active state is idempotent above
        # and must not depend forever on historical Git objects remaining
        # reachable.
        if not daemon.git_commit_descends_from(
                starting_commit=daemon.cycle_starting_commit(cycle_id),
                accepted_commit=accepted_commit):
            raise daemon.TicketCycleStateError(
                "daemon-recorded landing L is not a new descendant of the "
                "cycle base")
        if mode == "two-role":
            del state["active"][cycle_id]
            state["completed"][cycle_id] = accepted_commit
            state["generation"] = state["generation"] + 1
            daemon.record_pending_ticket_cycle_return(state=state)
            completed_now = 1
        elif mode == "normal":
            state["active"][cycle_id] = dict(
                current, phase="committed-awaiting-closure",
                commit=accepted_commit)
        daemon.write_ticket_cycle_state(state=state)
        # C and its private ref remain reachable until the GO itself reaches
        # done/. Startup recovery still needs C to re-prove an interrupted
        # exact landing or closure publication.
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)
    return completed_now


def active_ticket_cycle_count(skip_redteam=False, exclude_admission=None):
    """Count enabled work, optionally excluding one request's admission.

    Each topology counts only tickets it can advance. Valid work saved for a
    different topology remains active and untouched for a later watch.
    """
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        active = daemon.active_cycle_records_for_topology(
            state=state, skip_redteam=skip_redteam)
        admissions = daemon.architect_admissions_for_topology(
            state=state, skip_redteam=skip_redteam)
        excluded = state["architect_admissions"].get(exclude_admission)
        if (excluded is not None and daemon.ticket_cycle_mode_is_enabled(
                mode=excluded["mode"], skip_redteam=skip_redteam)):
            admissions.remove(excluded)
        return len(active) + len(admissions)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def read_cycle_message(path):
    """Read one bounded mailbox message for cycle-state reconciliation."""
    raw = daemon.stable_regular_bytes(
        path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
        label="cycle message " + daemon.os.path.basename(path))
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "cycle message is not UTF-8: " + path) from exc


def any_matching_redteam_receipt(cycle_id, accepted_commit):
    """Return whether exactly one persisted receipt matches the review."""
    matches = []
    for path in daemon.glob.glob(
            daemon.os.path.join(daemon.MAILBOX, "**", "*-to-fable.md"),
            recursive=True):
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        if not message.startswith(daemon.MAILBOX_RETURN_HEADER):
            continue
        returned_cycle, returned_commit, _, _, problem = (
            daemon._redteam_review_receipt(message=message))
        if (problem is None and returned_cycle == cycle_id
                and returned_commit == accepted_commit):
            matches.append(path)
    if len(matches) > 1:
        raise daemon.TicketCycleStateError(
            "more than one Red Team receipt names " + cycle_id + " at "
            + accepted_commit)
    return bool(matches)


def redteam_review_completes_cycle(result):
    """Only NO CHANGE ends a cycle without an Architect decision."""
    return result == "NO CHANGE"


def current_reopen_ticket(cycle_id):
    """Read one mechanically checked ticket before Architect reasoning."""
    try:
        sealed = daemon._validate_sealed_backlog(
            primary_worktree=daemon.AGENT_CWD["fable"])
        lines = sealed.decode("utf-8", errors="strict").splitlines()
        return daemon._REOPEN_TRANSITION.inspect_backlog(
            lines=lines, anchor=daemon.cycle_ticket_anchor(cycle_id))
    except (UnicodeDecodeError, daemon.PrimaryWorktreeError,
            daemon._REOPEN_TRANSITION.ReopenTransitionError) as exc:
        raise daemon.TicketCycleStateError(str(exc)) from exc


def architect_reopen_decision(cycle_id, before):
    """Verify the exact backlog transition and return GO or NO-GO."""
    sealed = daemon._validate_sealed_backlog(
        primary_worktree=daemon.AGENT_CWD["fable"])
    try:
        lines = sealed.decode("utf-8", errors="strict").splitlines()
        after = daemon._REOPEN_TRANSITION.inspect_backlog(
            lines=lines, anchor=daemon.cycle_ticket_anchor(cycle_id))
        return daemon._REOPEN_TRANSITION.validate_after(
            before=before, after=after)
    except (UnicodeDecodeError,
            daemon._REOPEN_TRANSITION.ReopenTransitionError) as exc:
        raise daemon.TicketCycleStateError(str(exc)) from exc


def land_architect_reopen_decision(dispatch_path, cycle_id,
                                   reviewed_landing, decision):
    """Land, record, and push one sealed GO/NO-GO backlog decision."""
    backlog = daemon._validate_sealed_backlog(
        primary_worktree=daemon.AGENT_CWD["fable"])
    main_lock = daemon.acquire_main_checkout_turn_lock()
    if main_lock is None:
        raise daemon.RetryableArchitectLandingError(
            "reopening decision landing lock is unavailable")
    landing = None
    try:
        landing, parent = daemon.prepare_reopen_decision_landing(
            cycle_id=cycle_id, reviewed_landing=reviewed_landing,
            decision=decision, backlog=backlog)
        daemon.preflight_role_baseline_sync(target=landing)
        daemon.land_prepared_commit_in_clean_user_checkout(
            landing=landing, parent=parent)
        completed = daemon.complete_reopen_ticket_cycle(
            cycle_id=cycle_id, reviewed_landing=reviewed_landing,
            decision_landing=landing)
        daemon.write_push_debt(
            landing=landing,
            detail="reopening decision landed; remote push not yet attempted")
        if not daemon.archive_consumed_message(dispatch_path=dispatch_path):
            raise daemon.RetryableArchitectLandingError(
                "reopening decision landed but its input was not archived")
        daemon.sync_all_clean_role_baselines(target=landing)
        reference = daemon.reopen_decision_ref(cycle_id=cycle_id)
        if daemon.git_ref_commit(reference=reference) == landing:
            daemon._run_git(
                repository_root=daemon.AGENT_CWD["fable"],
                arguments=["update-ref", "-d", reference, landing])
    finally:
        daemon.release_main_checkout_turn_lock(lock_file=main_lock)
    try:
        pushed, detail = daemon.push_exact_landing_or_record_debt(landing=landing)
    except (OSError, ValueError) as exc:
        pushed, detail = False, str(exc)
    if pushed:
        print("verified remote main at reopening decision " + landing + ".")
    elif pushed is not None:
        print("reopening decision is local; remote push remains follow-up "
              "debt for " + landing + (": " + detail if detail else "."))
    return landing, completed


def _matching_journaled_notes_go(base_commit, notes_commit,
                                  receipt_sha256):
    """Return one exact B/P receipt whose bytes match an admin journal."""
    matches = []
    for directory in (daemon.MAILBOX,
                      daemon.os.path.join(daemon.MAILBOX, "inflight"),
                      daemon.DONE):
        for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-daemon.md")):
            try:
                raw = daemon.stable_regular_bytes(
                    path=path,
                    maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="journaled permanent-note GO")
                message = raw.decode("utf-8", errors="strict")
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            returned_base, returned_notes, problem = (
                daemon._architect_notes_go_request(message=message))
            if (problem is None and returned_base == base_commit
                    and returned_notes == notes_commit
                    and daemon.hashlib.sha256(raw).hexdigest() == receipt_sha256):
                matches.append(path)
    if len(matches) != 1:
        raise daemon.TicketCycleStateError(
            "validated permanent-note admin journal needs exactly one "
            "unchanged B/P receipt; found " + str(len(matches)))
    return matches[0]


def _require_safe_noop_admin_recovery(base_commit):
    """Allow a proved no-change admin result after clean later landings."""
    primary = daemon.AGENT_CWD["fable"]
    primary_head = daemon.worktree_head(worktree=primary)
    current_main = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    if primary_head != current_main:
        raise daemon.TicketCycleStateError(
            "validated no-op admin needs Architect primary at current main")
    try:
        if daemon._tracked_worktree_changes(worktree=primary):
            raise daemon.TicketCycleStateError(
                "validated no-op admin needs a clean Architect primary")
        daemon._validate_current_protected_primary_state(primary_worktree=primary)
    except daemon.PrimaryWorktreeError as exc:
        raise daemon.TicketCycleStateError(str(exc)) from exc
    daemon._require_ancestor_or_same(
        ancestor=base_commit, descendant=current_main,
        label="validated no-op admin base is not in current main history")


def reconcile_architect_notes_admin_journals():
    """Validate every admin journal and retire only proved done no-ops."""
    prefix = ".pending-notes-admin-"
    suffix = ".json"
    pattern = daemon.os.path.join(daemon.RELAY_DIR, prefix + "*" + suffix)
    retired = 0
    for journal_path in sorted(daemon.glob.glob(pattern)):
        filename = daemon.os.path.basename(journal_path)
        request_name = filename[len(prefix):-len(suffix)]
        request_match = daemon.PENDING_MESSAGE_RE.fullmatch(request_name)
        if request_match is None or request_match.group(1) != "fable":
            raise daemon.TicketCycleStateError(
                "malformed permanent-note admin journal name: "
                + journal_path)
        request_path = daemon._architect_notes_admin_request_path(
            request_name=request_name)
        try:
            request_message = daemon.stable_regular_bytes(
                path=request_path,
                maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="saved permanent-note admin").decode(
                    "utf-8", errors="strict")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise daemon.TicketCycleStateError(
                "cannot verify saved permanent-note admin " + request_path
                + ": " + str(exc)) from exc
        if not daemon.is_architect_notes_admin_message(message=request_message):
            raise daemon.TicketCycleStateError(
                "saved permanent-note admin is malformed: " + request_path)
        journal = daemon.read_architect_notes_admin_journal(
            request_name=request_name, request_message=request_message)
        directory = daemon.os.path.dirname(request_path)
        if directory not in {
                daemon.os.path.join(daemon.MAILBOX, "inflight"),
                daemon.DONE}:
            raise daemon.TicketCycleStateError(
                "admin recovery journal is bound to an invalid request "
                "state: " + request_path)
        if journal["phase"] == "started":
            # The inflight reconciler prints the stronger warning that the
            # child may still be alive. Never infer a result from P/GO files.
            if directory != daemon.os.path.join(daemon.MAILBOX, "inflight"):
                raise daemon.TicketCycleStateError(
                    "archived permanent-note admin has only a pre-child "
                    "journal: " + request_path)
            continue
        if journal["phase"] == "validated-noop":
            if directory == daemon.os.path.join(daemon.MAILBOX, "inflight"):
                continue
            daemon._require_safe_noop_admin_recovery(
                base_commit=journal["base"])
            daemon.remove_architect_notes_admin_journal(
                request_name=request_name)
            retired += 1
            print("retired archived validated no-op admin journal "
                  + request_name + ".")
            continue
        base_commit = journal["base"]
        notes_commit = journal["notes_commit"]
        daemon._matching_journaled_notes_go(
            base_commit=base_commit, notes_commit=notes_commit,
            receipt_sha256=journal["receipt_sha256"])
        daemon.require_architect_notes_commit(
            base_commit=base_commit, notes_commit=notes_commit,
            allow_landed_replay=True)
        try:
            daemon._validate_current_protected_primary_state(
                primary_worktree=daemon.AGENT_CWD["fable"])
        except daemon.PrimaryWorktreeError as exc:
            raise daemon.TicketCycleStateError(str(exc)) from exc
    return retired


def reconcile_inflight_architect_notes_admin():
    """Archive only post-child admin results proved by their durable journal."""
    recovered = 0
    paths = sorted(daemon.glob.glob(daemon.os.path.join(
        daemon.MAILBOX, "inflight", "*-to-fable.md")), key=daemon.message_sequence)
    for path in paths:
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError) as exc:
            try:
                is_raw_admin = daemon.regular_file_has_prefix(
                    path=path,
                    prefix=daemon.MAILBOX_ADMIN_HEADER.encode("ascii"))
            except (OSError, ValueError):
                is_raw_admin = False
            if is_raw_admin:
                raise daemon.TicketCycleStateError(
                    "cannot verify inflight permanent-note admin " + path
                    + ": " + str(exc)) from exc
            continue
        if not daemon.is_architect_notes_admin_message(message=message):
            if message.startswith(daemon.MAILBOX_ADMIN_HEADER):
                raise daemon.TicketCycleStateError(
                    "inflight permanent-note admin is malformed: " + path)
            continue
        name = daemon.os.path.basename(path)
        journal_path = daemon.architect_notes_admin_journal_path(
            request_name=name)
        if not daemon.os.path.isfile(journal_path):
            raise daemon.TicketCycleStateError(
                "inflight permanent-note admin has no recovery journal: "
                + path + "; inspect its dispatch log and requeue only after "
                  "proving that no child is still running")
        journal = daemon.read_architect_notes_admin_journal(
            request_name=name, request_message=message)
        phase = journal["phase"]
        base_commit = journal["base"]
        if phase == "started":
            raise daemon.TicketCycleStateError(
                "inflight permanent-note admin has only a pre-child "
                "journal: " + path + "; a child may still be alive or its "
                "result may be unvalidated. Inspect the dispatch log and "
                "process before any manual requeue")
        if phase == "validated-noop":
            daemon._require_safe_noop_admin_recovery(base_commit=base_commit)
        else:
            notes_commit = journal["notes_commit"]
            daemon._matching_journaled_notes_go(
                base_commit=base_commit, notes_commit=notes_commit,
                receipt_sha256=journal["receipt_sha256"])
            daemon.require_architect_notes_commit(
                base_commit=base_commit, notes_commit=notes_commit,
                allow_landed_replay=True)
        if not daemon.archive_consumed_message(dispatch_path=path):
            raise daemon.TicketCycleStateError(
                "validated inflight permanent-note admin could not archive")
        if phase == "validated-noop":
            daemon.remove_architect_notes_admin_journal(request_name=name)
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
    daemon.read_ticket_cycle_state()
    daemon.reconcile_architect_notes_admin_journals()
    daemon.reconcile_inflight_architect_notes_admin()
    active_directories = [daemon.MAILBOX,
                          daemon.os.path.join(daemon.MAILBOX, "inflight"),
                          daemon.os.path.join(daemon.MAILBOX, "failed")]
    active_paths = []
    for directory in active_directories:
        active_paths.extend(daemon.glob.glob(daemon.os.path.join(directory, "*-to-*.md")))

    # First revalidate implementation identities already admitted into
    # durable state. Merely queued root/failed work must not be registered by
    # startup recovery because finite-cycle capacity is reserved only when a
    # watcher actually admits the ticket.
    registered_cycles = daemon.read_ticket_cycle_state()["active"]
    for path in sorted(active_paths):
        name = daemon.os.path.basename(path)
        match = daemon.PENDING_MESSAGE_RE.match(name)
        if match is None or match.group(1) == "daemon":
            continue
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            # Root corruption belongs to the ordinary dispatcher, which can
            # claim and park the exact inode with a useful reason. Inflight
            # corruption remains a lane blocker. Neither state can register
            # or consume a ticket during recovery.
            continue
        agent = match.group(1)
        is_flow = (agent in {"fable", "opus"}
                   and message.startswith(daemon.MAILBOX_FLOW_HEADER))
        cycle_id = None
        if is_flow:
            cycle_id, _, _, problem = daemon._ticket_flow_envelope(message=message)
            if problem is not None:
                continue
        if cycle_id in registered_cycles:
            record = registered_cycles[cycle_id]
            daemon.register_ticket_cycle_message(
                agent=agent, message=message,
                skip_redteam=(record["mode"] == "two-role"))

    completed_now = 0
    inflight_daemon = daemon.glob.glob(
        daemon.os.path.join(daemon.MAILBOX, "inflight", "*-to-daemon.md"))
    for path in sorted(inflight_daemon, key=daemon.message_sequence):
        message = daemon.read_cycle_message(path=path)
        if message.startswith(
                daemon.MAILBOX_RETURN_HEADER + "architect-notes-go"):
            base_commit, notes_commit, problem = (
                daemon._architect_notes_go_request(message=message))
            if problem is not None:
                if not daemon.park_failed_message(dispatch_path=path):
                    raise daemon.TicketCycleStateError(
                        "malformed inflight Architect notes GO could not be "
                        "parked: " + daemon.os.path.basename(path) + ": " + problem)
                continue
            consumed, _notes = daemon.finish_claimed_architect_notes_go(
                dispatch_path=path, base_commit=base_commit,
                notes_commit=notes_commit)
            if not consumed:
                continue
            # Permanent-note administration is cycle-free.
            continue
        cycle_id, candidate_commit, mode, problem = daemon._architect_go_request(
            message=message)
        if problem is not None:
            if not daemon.park_failed_message(dispatch_path=path):
                raise daemon.TicketCycleStateError(
                    "malformed inflight Architect GO could not be parked: "
                    + daemon.os.path.basename(path) + ": " + problem)
            print("parked malformed Architect GO request "
                  + daemon.os.path.basename(path) + " in failed/: " + problem)
            continue
        try:
            consumed, completed, _landing = daemon.finish_claimed_architect_go(
                dispatch_path=path, cycle_id=cycle_id,
                candidate_commit=candidate_commit, mode=mode)
        except daemon.FatalArchitectLandingError:
            raise
        if not consumed:
            continue
        completed_now = completed_now + completed

    done_daemon = daemon.glob.glob(daemon.os.path.join(daemon.DONE, "*-to-daemon.md"))
    for path in sorted(done_daemon, key=daemon.message_sequence):
        message = daemon.read_cycle_message(path=path)
        if message.startswith(
                daemon.MAILBOX_RETURN_HEADER + "architect-notes-go"):
            base_commit, notes_commit, problem = (
                daemon._architect_notes_go_request(message=message))
            if problem is not None:
                if not daemon.park_failed_message(dispatch_path=path):
                    raise daemon.TicketCycleStateError(
                        "malformed archived Architect notes GO could not be "
                        "parked: " + daemon.os.path.basename(path) + ": " + problem)
                continue
            try:
                receipt_raw = daemon.stable_regular_bytes(
                    path=path,
                    maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="archived permanent-note GO receipt")
                daemon.require_architect_notes_commit_object(
                    base_commit=base_commit, notes_commit=notes_commit)
                current_main = daemon._exact_git_object(
                    arguments=["rev-parse", "--verify",
                               "refs/heads/main^{commit}"],
                    label="current main commit")
                daemon._require_ancestor_or_same(
                    ancestor=notes_commit, descendant=current_main,
                    label="archived permanent-note P is not on main")
                main_lock = daemon.acquire_main_checkout_turn_lock()
                if main_lock is None:
                    raise daemon.TicketCycleStateError(
                        "cannot lock archived permanent-note recovery")
                try:
                    daemon.preflight_role_baseline_sync(target=current_main)
                    daemon.sync_all_clean_role_baselines(target=current_main)
                finally:
                    daemon.release_main_checkout_turn_lock(lock_file=main_lock)
                daemon.retire_validated_commit_admin_journal(
                    base_commit=base_commit, notes_commit=notes_commit,
                    receipt_sha256=daemon.hashlib.sha256(receipt_raw).hexdigest())
                if current_main == notes_commit:
                    debt_path = daemon._push_debt_path(landing=notes_commit)
                    if daemon.os.path.isfile(debt_path):
                        daemon.push_exact_landing_or_record_debt(
                            landing=notes_commit)
            except daemon.TicketCycleStateError as exc:
                if not daemon.park_failed_message(dispatch_path=path):
                    raise daemon.TicketCycleStateError(
                        "rejected archived Architect notes GO could not be "
                        "parked: " + daemon.os.path.basename(path) + ": "
                        + str(exc)) from exc
            continue
        cycle_id, candidate_commit, mode, problem = daemon._architect_go_request(
            message=message)
        if problem is not None:
            if not daemon.park_failed_message(dispatch_path=path):
                raise daemon.TicketCycleStateError(
                    "malformed archived Architect GO could not be parked: "
                    + daemon.os.path.basename(path) + ": " + problem)
            print("moved malformed historical Architect GO request "
                  + daemon.os.path.basename(path) + " from done/ to failed/: "
                  + problem)
            continue
        try:
            landing = daemon.recorded_landing_for_architect_go(
                cycle_id=cycle_id, mode=mode)
            if landing is None:
                raise daemon.TicketCycleStateError(
                    "archived Architect GO has no durable local landing")
            if mode == "normal":
                state = daemon.read_ticket_cycle_state()
                active = state["active"].get(cycle_id)
                if (active is not None
                        and active["phase"] in {
                            "committed-awaiting-closure",
                            "awaiting-redteam"}):
                    daemon.publish_redteam_closure_request(
                        cycle_id=cycle_id, landing=landing)
            daemon.retire_cycle_landing_ref(
                cycle_id=cycle_id, landing=landing)
            daemon.retire_cycle_candidate(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                landing_commit=landing, mode=mode)
            current_main = daemon._exact_git_object(
                arguments=["rev-parse", "--verify",
                           "refs/heads/main^{commit}"],
                label="current main commit")
            daemon._require_ancestor_or_same(
                ancestor=landing, descendant=current_main,
                label="archived ordinary landing is not on main")
            if current_main == landing:
                main_lock = daemon.acquire_main_checkout_turn_lock()
                if main_lock is None:
                    raise daemon.TicketCycleStateError(
                        "cannot lock archived role-baseline recovery")
                try:
                    daemon.sync_all_clean_role_baselines(target=landing)
                finally:
                    daemon.release_main_checkout_turn_lock(lock_file=main_lock)
                debt_path = daemon._push_debt_path(landing=landing)
                if daemon.os.path.isfile(debt_path):
                    daemon.push_exact_landing_or_record_debt(landing=landing)
        except daemon.TicketCycleStateError as exc:
            if not daemon.park_failed_message(dispatch_path=path):
                raise daemon.TicketCycleStateError(
                    "rejected archived Architect GO could not be parked: "
                    + daemon.os.path.basename(path) + ": " + str(exc)) from exc
            print("moved rejected historical Architect GO request "
                  + daemon.os.path.basename(path) + " from done/ to failed/: "
                  + str(exc))

    # Register still-waiting review requests after Architect GO recovery has
    # restored their recorded landing phase.
    for path in sorted(active_paths):
        name = daemon.os.path.basename(path)
        match = daemon.PENDING_MESSAGE_RE.match(name)
        if match is None or match.group(1) != "sol":
            continue
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        if daemon.sol_ticket_kind(message=message) == "closure":
            daemon.register_ticket_cycle_message(agent="sol", message=message)

    # A crash can occur after the request reached done/ and before its state
    # replacement. The Red Team return plus archived request is enough to
    # finish that exact transition once, never to infer a missing review from
    # rc alone.
    review_paths = daemon.glob.glob(daemon.os.path.join(daemon.DONE, "*-to-sol.md"))
    review_paths.extend(daemon.glob.glob(
        daemon.os.path.join(daemon.MAILBOX, "inflight", "*-to-sol.md")))
    for path in sorted(review_paths, key=daemon.message_sequence):
        message = daemon.read_cycle_message(path=path)
        if (daemon.sol_ticket_kind(message=message) != "closure"
                or daemon.redteam_closure_problem(message=message) is not None):
            continue
        cycle_id = daemon.redteam_closure_ticket(message=message)
        commit = daemon.redteam_closure_commit(message=message)
        state = daemon.read_ticket_cycle_state()
        if state["completed"].get(cycle_id) == commit:
            if daemon.os.path.dirname(path) != daemon.DONE:
                if not daemon.archive_consumed_message(dispatch_path=path):
                    raise daemon.TicketCycleStateError(
                        "completed Red Team request could not be archived: "
                        + daemon.os.path.basename(path))
            continue
        _receipt_path, review_result, problem = daemon.matching_new_redteam_receipt(
            cycle_id=cycle_id, accepted_commit=commit, before_inodes=set())
        if problem is not None:
            raise daemon.TicketCycleStateError(problem)
        daemon.register_ticket_cycle_message(agent="sol", message=message)
        if daemon.redteam_review_completes_cycle(review_result):
            if daemon.complete_ticket_cycle(cycle_id=cycle_id,
                                     accepted_commit=commit):
                completed_now = completed_now + 1
        if daemon.os.path.dirname(path) != daemon.DONE:
            if not daemon.archive_consumed_message(dispatch_path=path):
                raise daemon.TicketCycleStateError(
                    "recovered Red Team request could not be archived: "
                    + daemon.os.path.basename(path))
    return completed_now


def publish_message_locked(agent, payload, attempts=20):
    """Atomically publish one message while the caller holds sequence lock."""
    for _ in range(attempts):
        path = daemon.os.path.join(
            daemon.MAILBOX, daemon.next_seq() + "-to-" + agent + ".md")
        handle, temporary = daemon.tempfile.mkstemp(
            prefix=".message-", dir=daemon.MAILBOX)
        try:
            with daemon.os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
                if not payload.endswith("\n"):
                    stream.write("\n")
                stream.flush()
                daemon.os.fsync(stream.fileno())
            try:
                # Same-directory hard-link publication never replaces a
                # manually created destination or exposes partial bytes.
                daemon.os.link(temporary, path)
            except FileExistsError:
                continue
            # The state may suppress replay only after the directory entry
            # itself survives a crash, not merely the payload inode.
            daemon.fsync_directory(directory=daemon.MAILBOX)
            return path
        finally:
            if daemon.os.path.isfile(temporary):
                daemon.os.remove(temporary)
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
            daemon.resolve_discovery_severity(cli_value=severity)
            if ticket_kind == "discovery" else severity)
    except ValueError as exc:
        print("refused --send " + agent + ": " + str(exc) + ".")
        return False
    effective_scope = (
        (daemon.DEFAULT_DISCOVERY_SCOPE if scope is None else scope)
        if ticket_kind == "discovery" else scope)
    if (ticket_kind == "discovery"
            and effective_scope not in daemon.DISCOVERY_SCOPES):
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
        if daemon.skip_redteam_policy_active():
            return ("an active two-role watch has the Sol route disabled; "
                    "wait for it to end or restart without --skip-redteam")
        transport_valid = (
            ticket_kind == "transport"
            and text == daemon.transport_ping_text(agent="sol"))
        counts = daemon.backlog_severity_counts()
        reason = daemon.sol_ticket_refusal(
            ticket_kind=ticket_kind,
            admission_count=(counts["critical"] + counts["high"]
                             + counts["medium"]),
            fix_only=(daemon.fix_only_environment_active()
                      or daemon.fix_only_watch_is_active()),
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
        if ticket_kind in daemon.SOL_DISPATCH_TICKET_KINDS:
            payload = daemon.sol_ticket_payload(
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
              + daemon.os.path.join(
                  daemon.MAILBOX,
                  daemon.next_seq() + "-to-" + agent + ".md"))
        daemon.warn_if_mailbox_unwatched()
        return True
    if not daemon.live_action_topology_is_current(agent, "--send " + agent):
        return False
    daemon.os.makedirs(daemon.MAILBOX, exist_ok=True)
    lock_file = daemon.acquire_mailbox_sequence_lock()
    if lock_file is None:
        return False
    try:
        if not daemon.live_action_topology_is_current(agent, "--send " + agent):
            return False
        # Recheck persisted modes and the current classified backlog while
        # publication is serialized. Queue publication does not itself change
        # either severity count.
        reason = refusal_now()
        if reason is not None:
            print("refused --send " + agent + ": " + reason + ".")
            return False
        for _ in range(20):
            path = daemon.publish_message_locked(
                agent=agent, payload=payload, attempts=1)
            if path is not None:
                print("queued " + path)
                try:
                    daemon.warn_if_mailbox_unwatched()
                    if daemon.skip_redteam_policy_active():
                        daemon.report_demand(
                            backlog=daemon.pending_messages(), skip_redteam=True)
                    else:
                        daemon.report_demand(backlog=daemon.pending_messages())
                except Exception as exc:
                    print("  warning: message is queued, but its status "
                          "report failed: " + str(exc))
                return True
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=lock_file)
    print("could not claim a sequence number after 20 tries; "
          "is something flooding the mailbox?")
    return False


def recover_failed_maintenance_admission():
    """Requeue one failed fix-only request on restart."""
    sequence_lock = daemon.acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise daemon.TicketCycleStateError("cannot lock recovery")
    state_lock = None
    try:
        state_lock = daemon.acquire_ticket_cycle_lock()
        state = daemon.read_ticket_cycle_state()
        match = None
        failed = daemon.os.path.join(daemon.MAILBOX, "failed")
        for name, record in state["architect_admissions"].items():
            path = daemon.os.path.join(failed, name)
            if not daemon.os.path.lexists(path):
                continue
            message = daemon.read_cycle_message(path=path)
            if daemon.hashlib.sha256(message.encode("utf-8")).hexdigest() \
                    != record["sha256"]:
                raise daemon.TicketCycleStateError("failed request changed")
            if message == daemon.ARCHITECT_FIX_ONLY_REQUEST:
                if match is not None:
                    raise daemon.TicketCycleStateError(
                        "multiple maintenance requests failed")
                match = path
        if match is None:
            return
        for duplicate in daemon.pending_messages():
            try:
                duplicate_message = daemon.read_cycle_message(path=duplicate)
            except (OSError, ValueError, daemon.TicketCycleStateError):
                continue
            if duplicate_message == daemon.ARCHITECT_FIX_ONLY_REQUEST:
                _parked, moved = daemon.verified_state_move(
                    dispatch_path=duplicate, directory=failed)
                if not moved:
                    raise daemon.TicketCycleStateError(
                        "could not preserve duplicate")
                print("parked duplicate " + daemon.os.path.basename(duplicate)
                      + " in failed/")
        recovered, moved = daemon.verified_state_move(
            dispatch_path=match, directory=daemon.MAILBOX)
        if not moved:
            raise daemon.TicketCycleStateError("could not requeue failed request")
        print("requeued " + recovered)
        return recovered
    finally:
        if state_lock is not None:
            daemon.release_ticket_cycle_lock(lock_file=state_lock)
        daemon.release_mailbox_sequence_lock(lock_file=sequence_lock)


def send_architect_notes_admin(text, dry_run=False):
    """Publish one narrow Architect-only permanent-note self-route."""
    try:
        contract = daemon.validate_role_contract_bindings()
    except (OSError, RuntimeError, ValueError) as exc:
        print("refused permanent-note admin request: role contract error: "
              + str(exc) + ".")
        return False
    if not contract["roles"]["architect"]["may_edit_protected_policy"]:
        print("refused permanent-note admin request: protected role contract "
              "does not grant Architect policy administration.")
        return False
    if daemon.os.environ.get(daemon.MAILBOX_ROLE_ENVIRONMENT) != "architect":
        print("refused permanent-note admin request: MAILBOX_ROLE must be "
              "architect.")
        return False
    primary = daemon.os.environ.get("MAILBOX_PRIMARY_WORKTREE")
    shared_notes = daemon.os.environ.get("MAILBOX_SHARED_NOTES")
    if (primary is None or shared_notes is None
            or daemon.os.path.realpath(primary)
            != daemon.os.path.realpath(daemon.WORKTREE)
            or daemon.os.path.realpath(shared_notes)
            != daemon.os.path.realpath(daemon.os.path.join(daemon.AI_ROOT, "notes"))):
        print("refused permanent-note admin request: this process is not "
              "bound to the saved Architect primary and shared notes.")
        return False
    try:
        payload = daemon.architect_notes_admin_payload(text=text)
    except ValueError as exc:
        print("refused permanent-note admin request: " + str(exc) + ".")
        return False
    if dry_run:
        print("[dry-run] would queue " + daemon.os.path.join(
            daemon.MAILBOX, daemon.next_seq() + "-to-fable.md"))
        return True
    daemon.os.makedirs(daemon.MAILBOX, exist_ok=True)
    lock_file = daemon.acquire_mailbox_sequence_lock()
    if lock_file is None:
        return False
    try:
        if daemon.architect_notes_transition_pending():
            print("refused permanent-note admin request: another note admin "
                  "turn or P landing is already pending.")
            return False
        path = daemon.publish_message_locked(agent="fable", payload=payload)
        if path is None:
            print("refused permanent-note admin request: no unique mailbox "
                  "sequence could be published.")
            return False
        print("queued " + path)
        return True
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=lock_file)
