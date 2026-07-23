"""Daemon receipts, interruption recovery, and the backlog pass.

Any run can stop between two steps: a crash, a timeout kill, or
Ctrl-C. This file owns the recovery that makes the next start resume
instead of losing work: replaying a recorded Architect GO whose
archive step never ran, requeueing parked requests whose refusal was
transient, and restarting a killed role turn from its exact saved
handoff. It also owns the backlog pass itself: ``process_backlog``
claims each waiting message in order, runs the matching lane, and
defers Ctrl-C so a durable transition always finishes whole.

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
    "DeferredInterrupts",
    "consume_daemon_message",
    "release_unstarted_ticket_reservation",
    "ticket_cycle_has_live_message",
    "recover_failed_implementer_preflight",
    "revalidate_unmeasurable_budget_handoff",
    "recover_failed_implementer_returns",
    "live_implementer_owns_architect_admission",
    "retire_failed_public_architect_admission",
    "recover_failed_public_architect_admissions",
    "recover_failed_architect_outcome",
    "recover_failed_open_ticket_go",
    "recover_prelaunch_messages",
    "restart_implementer_from_architect_handoff",
    "restart_redteam_from_architect_handoff",
    "recover_interrupted_mailbox_moves",
    "blocked_redteam_directory",
    "recover_blocked_redteam_messages",
    "block_protected_ticket_without_redteam",
    "recover_before_dispatch",
    "implementer_reservation_preflight_problem",
    "reserve_architect_ticket_before_claim",
    "release_architect_ticket_admission",
    "reserve_implementer_ticket_before_claim",
    "drain_lane",
    "process_backlog",
    "report_deferred_sol_messages",
    "report_demand",
    "landing_debt_snapshot",
    "report_landing_debt",
)


class DeferredInterrupts:
    """Delay Ctrl-C while one daemon transition must finish whole.

    On entry, the interrupt signal is redirected to a recorder; on exit
    the previous handler returns and one recorded interrupt is raised, so
    the user's Ctrl-C takes effect at the transition boundary instead of
    half-way through a Git landing sequence. Outside the main thread, or
    when no handler can be installed, the manager does nothing: Python
    delivers Ctrl-C only to the main thread, so worker threads need no
    protection.
    """

    def __init__(self):
        """Start with no recorded Ctrl-C and no installed handler."""
        self._pending = False
        self._previous = None
        self._installed = False

    def _record(self, signum, frame):
        """Remember one Ctrl-C and tell the user it is deferred, not lost.

        Arguments:
          signum = the delivered signal number (unused; the signal
                   module requires this handler signature).
          frame  = the interrupted stack frame (unused, same reason).
        """
        del signum, frame
        self._pending = True
        print("interrupt received: finishing the current mailbox "
              "transition first.", flush=True)

    def __enter__(self):
        """Redirect Ctrl-C to the recorder while the block runs.

        The handler is installed only from the main thread, because
        Python delivers Ctrl-C only there. When no handler can be
        installed the manager degrades to doing nothing.

        Returns:
          This manager, as the ``with`` statement expects.
        """
        in_main_thread = (daemon.threading.current_thread()
                          is daemon.threading.main_thread())
        if in_main_thread:
            try:
                self._previous = daemon.signal.signal(
                    daemon.signal.SIGINT, self._record)
                self._installed = True
            except (ValueError, OSError):
                self._installed = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the previous handler and deliver one deferred Ctrl-C.

        Arguments:
          exc_type  = the in-flight exception class, or ``None``.
          exc_value = its instance, or ``None``.
          traceback = its traceback object, or ``None``.

        Returns:
          False, so an exception raised inside the block propagates.
          A deferred Ctrl-C is raised here only when the block ended
          without an exception of its own.
        """
        if self._installed:
            daemon.signal.signal(daemon.signal.SIGINT, self._previous)
            if self._pending and exc_type is None:
                raise KeyboardInterrupt
        return False


def consume_daemon_message(path, dry_run=False, return_outcome=False):
    """Validate and consume one message addressed to the daemon itself.

    The daemon recipient is not an AI lane. It exists so cycle completion is
    based on a saved Architect decision instead of inference from prose or a
    changing backlog count. Three message kinds arrive here: an ordinary
    Architect GO naming one exact candidate commit, an Architect GO for a
    permanent-note commit, and a protected Red Team receipt supplying the
    second key for a control-plane candidate.

    Arguments:
      path           = the pending ``*-to-daemon.md`` file in the mailbox
                       root.
      dry_run        = True to print what the message would do without
                       claiming, landing, or archiving anything.
      return_outcome = True to return the daemon outcome constant instead
                       of a boolean.

    Returns:
      With ``return_outcome`` False, True only when the message was fully
      consumed. With it True, one of the daemon outcome constants:
      ``DAEMON_MESSAGE_CONSUMED`` (done and archived),
      ``DAEMON_MESSAGE_HARD_STOP`` (the message could not be claimed, or
      was refused and parked in ``failed/``),
      ``DAEMON_CONTROL_PLANE_WAITING`` (the GO stays in ``inflight/``
      until the Red Team's second key arrives), or
      ``DAEMON_NOTE_DEFERRED`` (a permanent-note landing must wait for
      an idle boundary).
    """
    def result(outcome):
        """Convert one outcome constant to the caller's requested form.

        Arguments:
          outcome = one of the daemon outcome constants.

        Returns:
          The constant itself when ``return_outcome`` is set, else the
          boolean "was this message fully consumed".
        """
        return outcome if return_outcome \
            else outcome == daemon.DAEMON_MESSAGE_CONSUMED

    name = daemon.os.path.basename(path)
    dispatch_path = path
    if not dry_run:
        dispatch_path = daemon.claim_message(path=path)
        if dispatch_path is None:
            return result(daemon.DAEMON_MESSAGE_HARD_STOP)
    try:
        with open(dispatch_path, encoding="utf-8", newline="") as stream:
            message = stream.read()
    except (OSError, UnicodeError) as exc:
        if dry_run:
            print("[dry-run] would refuse " + name + ": " + str(exc))
            return result(daemon.DAEMON_MESSAGE_HARD_STOP)
        print("refused " + name + ": cannot read Architect GO request; "
              + daemon.park_failed_outcome(dispatch_path=dispatch_path))
        return result(daemon.DAEMON_MESSAGE_HARD_STOP)
    if message.startswith(
            daemon.MAILBOX_RETURN_HEADER + "redteam-control-plane"):
        cycle_id, candidate_commit, decision, _body, problem = (
            daemon._control_plane_review_receipt(message=message))
        if problem is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": " + problem)
                return result(daemon.DAEMON_MESSAGE_HARD_STOP)
            print("refused " + name + ": " + problem + "; "
                  + daemon.park_failed_outcome(dispatch_path=dispatch_path))
            return result(daemon.DAEMON_MESSAGE_HARD_STOP)
        if dry_run:
            print("[dry-run] would record " + decision
                  + " for exact protected candidate " + candidate_commit)
            return result(daemon.DAEMON_MESSAGE_CONSUMED)
        try:
            control = daemon.control_plane_ticket_state(
                cycle_id=cycle_id, candidate_commit=candidate_commit)
            authenticated = daemon.control_plane_redteam_key_matches(
                control=control, candidate_commit=candidate_commit,
                decision=decision)
            if not authenticated:
                raise daemon.TicketCycleStateError(
                    "control-plane receipt lacks a D0-recorded successful "
                    "Sol dispatch")
        except daemon.TicketCycleStateError as exc:
            print("refused " + name + ": " + str(exc) + "; "
                  + daemon.park_failed_outcome(dispatch_path=dispatch_path))
            return result(daemon.DAEMON_MESSAGE_HARD_STOP)
        if not daemon.archive_consumed_message(dispatch_path=dispatch_path):
            return result(daemon.DAEMON_MESSAGE_HARD_STOP)
        go_paths = []
        for go_path in daemon.glob.glob(daemon.os.path.join(
                daemon.MAILBOX, "inflight", "*-to-daemon.md")):
            try:
                go_message = daemon.read_cycle_message(path=go_path)
            except (OSError, ValueError, daemon.TicketCycleStateError):
                continue
            found_cycle, found_candidate, found_mode, go_problem = (
                daemon._architect_go_request(message=go_message))
            if (go_problem is None and found_cycle == cycle_id
                    and found_candidate == candidate_commit):
                go_paths.append((go_path, found_mode))
        if len(go_paths) != 1:
            print("protected Red Team decision is durable, but its exact "
                  "inflight Architect GO was not uniquely found; restart "
                  "will preserve the decision and recover the same C.")
            return result(daemon.DAEMON_CONTROL_PLANE_WAITING)
        consumed, _completed, _landing = daemon.finish_claimed_architect_go(
            dispatch_path=go_paths[0][0], cycle_id=cycle_id,
            candidate_commit=candidate_commit, mode=go_paths[0][1])
        if consumed is None:
            return result(daemon.DAEMON_CONTROL_PLANE_WAITING)
        return result(daemon.DAEMON_MESSAGE_CONSUMED if consumed
                      else daemon.DAEMON_MESSAGE_HARD_STOP)
    if message.startswith(daemon.MAILBOX_RETURN_HEADER + "architect-notes-go"):
        base_commit, notes_commit, problem = (
            daemon._architect_notes_go_request(message=message))
        if problem is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": " + problem)
                return result(daemon.DAEMON_MESSAGE_HARD_STOP)
            print("refused " + name + ": " + problem + "; "
                  + daemon.park_failed_outcome(dispatch_path=dispatch_path))
            return result(daemon.DAEMON_MESSAGE_HARD_STOP)
        if dry_run:
            print("[dry-run] would land exact permanent-note commit "
                  + notes_commit + " on " + base_commit)
            return result(daemon.DAEMON_MESSAGE_CONSUMED)
        _consumed, _notes_commit, outcome = (
            daemon.finish_claimed_architect_notes_go(
            dispatch_path=dispatch_path, base_commit=base_commit,
            notes_commit=notes_commit, return_outcome=True))
        return result(outcome)
    cycle_id, candidate_commit, mode, problem = daemon._architect_go_request(
        message=message)
    if problem is not None:
        if dry_run:
            print("[dry-run] would refuse " + name + ": " + problem)
            return result(daemon.DAEMON_MESSAGE_HARD_STOP)
        print("refused " + name + ": " + problem + "; "
              + daemon.park_failed_outcome(dispatch_path=dispatch_path))
        return result(daemon.DAEMON_MESSAGE_HARD_STOP)
    if dry_run:
        print("[dry-run] would prepare and locally land exact candidate "
              + candidate_commit + " from Architect GO " + name)
        return result(daemon.DAEMON_MESSAGE_CONSUMED)
    consumed, _completed, _landing = daemon.finish_claimed_architect_go(
        dispatch_path=dispatch_path, cycle_id=cycle_id,
        candidate_commit=candidate_commit, mode=mode)
    if consumed is None:
        return result(daemon.DAEMON_CONTROL_PLANE_WAITING)
    return result(daemon.DAEMON_MESSAGE_CONSUMED if consumed
                  else daemon.DAEMON_MESSAGE_HARD_STOP)


def release_unstarted_ticket_reservation(cycle_id, expected_mode=None):
    """Remove only a new implementation reservation that was never claimed.

    Arguments:
      cycle_id      = the ticket cycle whose reservation may be
                      released.
      expected_mode = required saved mode, or ``None`` to accept any.

    Returns:
      True only when an implementation-phase reservation with no
      candidate commit on the primary route was found and removed;
      anything already claimed is left alone.
    """
    lock_file = daemon.acquire_ticket_cycle_lock()
    released = False
    try:
        state = daemon.read_ticket_cycle_state()
        current = state["active"].get(cycle_id)
        if (current is not None and current["phase"] == "implementation"
                and current["commit"] is None
                and current["route"] == "primary"
                and (expected_mode is None
                     or current["mode"] == expected_mode)):
            del state["active"][cycle_id]
            daemon.write_ticket_cycle_state(state=state)
            released = True
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)
    return released


def ticket_cycle_has_live_message(cycle_id):
    """Return whether a root or inflight message still owns this cycle.

    Arguments:
      cycle_id = the ticket cycle identifier.

    Returns:
      True when any pending or inflight agent message carries the
      cycle header — or cannot be read, which counts as live so an
      unreadable owner is never released.
    """
    header = daemon.MAILBOX_CYCLE_HEADER + cycle_id
    for directory in (daemon.MAILBOX, daemon.os.path.join(daemon.MAILBOX, "inflight")):
        for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-*.md")):
            if daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(path)) is None:
                continue
            try:
                message = daemon.read_cycle_message(path=path)
            except (OSError, ValueError, daemon.TicketCycleStateError):
                return True
            if header in message.splitlines():
                return True
    return False


def recover_failed_implementer_preflight():
    """Release ticket reservations whose Implementer never launched.

    A daemon can die between reserving a finite-watch ticket slot and
    starting the Implementer process. The parked request then sits in
    ``failed/`` while its reservation still holds a slot. For each such
    request this pass proves the launch never happened — the cycle has no
    live message, no saved candidate, and the Implementer checkout still
    sits clean on the ticket's starting commit — and only then releases
    the unclaimed reservation.

    Returns:
      The number of reservations released. Requests failing any proof
      are left exactly as found.
    """
    recovered = 0
    pattern = daemon.os.path.join(daemon.MAILBOX, "failed", "*-to-opus.md")
    for path in sorted(daemon.glob.glob(pattern), key=daemon.message_sequence):
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        if (not message.startswith(daemon.MAILBOX_FLOW_HEADER)
                or len(daemon.ARCHITECT_DIRECTIVE_LINE_RE.findall(message)) == 1):
            continue
        cycle_id, mode, _body, problem = daemon._ticket_flow_envelope(
            message=message)
        if (problem is not None
                or daemon.ticket_cycle_has_live_message(cycle_id=cycle_id)
                or daemon.candidate_commit_for_cycle(cycle_id) is not None
                or daemon.worktree_head(daemon.AGENT_CWD["opus"])
                != daemon.cycle_starting_commit(cycle_id)
                or daemon._clean_worktree_status(daemon.AGENT_CWD["opus"])):
            continue
        if daemon.release_unstarted_ticket_reservation(
                cycle_id=cycle_id, expected_mode=mode):
            recovered += 1
            print("released pre-launch reservation for failed "
                  + daemon.os.path.basename(path))
    return recovered


def revalidate_unmeasurable_budget_handoff(
        path, cycle_id, candidate, maximum):
    """Promote a saved return when the trusted size guard can now count it.

    An Implementer checkpoint that reported an unmeasurable
    character-change result is rerun through the size guard in audit
    mode. When the guard now measures the candidate within the limit,
    the saved file is rewritten in place as a review-request return
    carrying the authoritative count, so finished work is not redone.

    Arguments:
      path      = the saved Implementer return.
      cycle_id  = its ticket cycle.
      candidate = the candidate commit to measure.
      maximum   = the ticket's character limit.

    Returns:
      True when the guard measured within the limit and the return
      was promoted; False leaves the file untouched.
    """
    message = daemon.read_cycle_message(path=path)
    _cycle, _mode, body, problem = daemon._ticket_flow_envelope(message=message)
    result_line = "- **Character-change result:**"
    if (problem is not None or not daemon.is_implementer_budget_checkpoint(body)
            or result_line not in body
            or not daemon.re.search(r"(?i)cannot measure|unmeasurable", body)):
        return False

    guard = daemon.os.path.join(
        daemon.AGENT_CWD["fable"], "ai", "tools", "ticket_change_guard.py")
    command = [
        daemon.sys.executable, guard, "--repo", daemon.AGENT_CWD["opus"],
        "--base", daemon.cycle_starting_commit(cycle_id),
        "--architect-audit", "--candidate", candidate,
        "--max", str(maximum)]
    environment = daemon.os.environ.copy()
    environment[daemon.MAX_CHARACTERS_ENVIRONMENT] = str(maximum)
    environment["MAILBOX_TICKET_CHANGE_GUARD"] = guard
    try:
        result = daemon.subprocess.run(
            command, stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.STDOUT,
            text=True, check=False, env=environment, timeout=30)
    except (OSError, daemon.subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    count_rows = [line for line in result.stdout.splitlines()
                  if line.startswith("changed characters: ")]
    if len(count_rows) != 1:
        return False

    message = message.replace(
        daemon.IMPLEMENTER_BUDGET_CHECKPOINT_HEADING,
        "### IMPLEMENTER_HANDOFF: REQUESTING REVIEW", 1)
    message = message.replace(
        result_line,
        result_line + " within limit; authoritative recovery check: "
        + count_rows[0], 1)
    directory = daemon.os.path.dirname(path)
    descriptor, temporary = daemon.tempfile.mkstemp(
        prefix=".budget-recheck-", dir=directory)
    try:
        with daemon.os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) \
                as stream:
            descriptor = -1
            stream.write(message)
            stream.flush()
            daemon.os.fsync(stream.fileno())
        daemon.os.replace(temporary, path)
        daemon.fsync_directory(directory=directory)
    finally:
        if descriptor >= 0:
            daemon.os.close(descriptor)
        try:
            daemon.os.remove(temporary)
        except FileNotFoundError:
            pass
    print("rechecked formerly unmeasurable Implementer return "
          + daemon.os.path.basename(path) + "; the trusted guard now reports "
          "within limit")
    return True


def recover_failed_implementer_returns():
    """Accept finished Implementer work whose delivery was interrupted.

    A killed run can leave a completed unit split across folders: the
    Architect's request in ``failed/`` or ``inflight/``, the Implementer's
    finished return in ``failed/``, and a real candidate commit sitting in
    the Implementer checkout. Rerunning the role would pay for the work
    twice. This pass proves the pieces still belong together — the ticket
    is still in its implementation phase, the request validates, exactly
    one saved return names the checkout's exact commit, and the evidence
    contract accepts it — then restores the request to ``inflight/``, the
    return to the mailbox root, and writes the delivery receipt.

    A return parked as unmeasurable is first rerun through the trusted
    size guard, so a now-measurable count can promote it in place.

    Returns:
      The number of returns restored for ordinary Architect review.
    """
    recovered = 0
    requests = [
        path for directory in (daemon.os.path.join(daemon.MAILBOX, "failed"),
                               daemon.os.path.join(daemon.MAILBOX, "inflight"))
        for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-opus.md"))]
    for request_path in sorted(requests, key=daemon.message_sequence):
        try:
            request = daemon.read_cycle_message(path=request_path)
            cycle_id, mode, _body, problem = daemon._ticket_flow_envelope(
                message=request)
            active = daemon.read_ticket_cycle_state()["active"].get(cycle_id)
            if (problem is not None or active is None
                    or active["phase"] != "implementation"
                    or active["commit"] is not None
                    or active["mode"] != mode
                    or daemon.architect_handoff_problem(
                        message=request, cycle_id=cycle_id,
                        mode=mode) is not None
                    or daemon.candidate_commit_for_cycle(cycle_id) is not None):
                continue
            candidate = daemon.worktree_head(worktree=daemon.AGENT_CWD["opus"])
            if (candidate == daemon.cycle_starting_commit(cycle_id)
                    or daemon._clean_worktree_status(daemon.AGENT_CWD["opus"])):
                continue
            contract = daemon.prepare_implementer_evidence_contract(
                message=request, use_saved_limit=True)
            saved_returns = []
            for return_path in daemon.glob.glob(daemon.os.path.join(
                    daemon.MAILBOX, "failed", "*-to-fable.md")):
                try:
                    returned = daemon.read_cycle_message(path=return_path)
                    returned_cycle, returned_mode, returned_body, error = (
                        daemon._ticket_flow_envelope(message=returned))
                except (OSError, ValueError, daemon.TicketCycleStateError):
                    continue
                if (error is None and returned_cycle == cycle_id
                        and returned_mode == mode
                        and daemon.IMPLEMENTER_CANDIDATE_LINE_RE.findall(
                            returned_body) == [candidate]):
                    saved_returns.append(return_path)
            if len(saved_returns) == 1:
                daemon.revalidate_unmeasurable_budget_handoff(
                    path=saved_returns[0], cycle_id=cycle_id,
                    candidate=candidate,
                    maximum=contract.get("character_limit", daemon.MAX_CHARACTERS))
            return_path, _invalid, evidence_problem, ready = (
                daemon.matching_new_implementer_handoff(
                    cycle_id=cycle_id, mode=mode,
                    candidate_commit=candidate,
                    before_inodes=frozenset(), evidence_contract=contract))
            if evidence_problem is not None or not ready:
                continue
            if daemon.os.path.dirname(request_path) != daemon.os.path.join(
                    daemon.MAILBOX, "inflight"):
                request_path, moved = daemon.verified_state_move(
                    dispatch_path=request_path,
                    directory=daemon.os.path.join(daemon.MAILBOX, "inflight"))
                if not moved:
                    raise daemon.TicketCycleStateError(
                        "validated Implementer request could not be restored "
                        "for delivery")
            if daemon.os.path.dirname(return_path) != daemon.MAILBOX:
                return_path, moved = daemon.verified_state_move(
                    dispatch_path=return_path, directory=daemon.MAILBOX)
                if not moved:
                    raise daemon.TicketCycleStateError(
                        "validated Implementer return could not be restored "
                        "for Architect review")
            daemon.write_implementer_delivery_receipt(
                request_path=request_path, return_path=return_path)
            recovered += 1
            print("revalidated completed Implementer return "
                  + daemon.os.path.basename(return_path)
                  + "; candidate will be preserved without rerunning the "
                    "Implementer")
        except (OSError, ValueError, daemon.PrimaryWorktreeError,
                daemon.TicketCycleStateError):
            continue
    return recovered


def live_implementer_owns_architect_admission(token):
    """Return whether a valid queued Implementer handoff owns ``token``.

    Arguments:
      token = the public request's admission token.

    Returns:
      True when any Implementer message in the root, inflight,
      prelaunch, or done states carries the token — or cannot be
      read, which counts as owned so a slot is never freed under an
      unreadable owner.
    """
    request_name, digest = daemon.split_architect_admission_token(token=token)
    for directory in (daemon.MAILBOX, daemon.os.path.join(daemon.MAILBOX, "inflight"),
                      daemon.os.path.join(daemon.MAILBOX, "prelaunch"), daemon.DONE):
        for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-opus.md")):
            try:
                message = daemon.read_cycle_message(path=path)
            except (OSError, ValueError, daemon.TicketCycleStateError):
                return True
            flow_name, flow_digest, problem = (
                daemon._ticket_architect_admission(message=message))
            if (problem is None and flow_name == request_name
                    and flow_digest == digest):
                return True
    return False


def retire_failed_public_architect_admission(path):
    """Release one exact failed public request without retrying its turn.

    The release is conservative: the file must live in ``failed/``
    with no sibling in any other state, its saved sequence and digest
    must match the charged record exactly, and no live Implementer
    handoff may own its admission token. Only then is the finite
    ticket slot freed; the failed turn itself is never retried.

    Arguments:
      path = the failed public Architect request.

    Returns:
      True when the slot was freed.

    Raises:
      daemon.TicketCycleStateError: when locking fails or the failed
        request changed identity under its record.
    """
    name = daemon.os.path.basename(path)
    match = daemon.PENDING_MESSAGE_RE.fullmatch(name)
    if (match is None or match.group(1) != "fable"
            or daemon.os.path.dirname(path)
            != daemon.os.path.join(daemon.MAILBOX, "failed")):
        return False
    sequence_lock = daemon.acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise daemon.TicketCycleStateError("cannot lock failed admission recovery")
    state_lock = None
    try:
        state_lock = daemon.acquire_ticket_cycle_lock()
        state = daemon.read_ticket_cycle_state()
        record = state["architect_admissions"].get(name)
        if record is None:
            return False
        other_states = [
            daemon.os.path.join(daemon.MAILBOX, name),
            daemon.os.path.join(daemon.MAILBOX, "prelaunch", name),
            daemon.os.path.join(daemon.DONE, name),
            daemon.os.path.join(daemon.MAILBOX, "inflight", name),
            daemon.os.path.join(
                daemon.MAILBOX, "inflight",
                name + daemon.STATE_GUARD_SUFFIX),
        ]
        if any(daemon.os.path.lexists(candidate) for candidate in other_states):
            return False
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            return False
        digest = daemon.hashlib.sha256(message.encode("utf-8")).hexdigest()
        if (message == daemon.ARCHITECT_FIX_ONLY_REQUEST
                or daemon.architect_user_request_problem(message=message)
                is not None):
            return False
        if (record["sequence"] != daemon.message_sequence(path)
                or record["sha256"] != digest):
            raise daemon.TicketCycleStateError(
                "failed public Architect request changed identity")
        token = daemon.architect_admission_token(
            request_name=name, digest=digest)
        if daemon.live_implementer_owns_architect_admission(token=token):
            return False
        del state["architect_admissions"][name]
        daemon.write_ticket_cycle_state(state=state)
        print("released finite-cycle slot for failed " + name
              + "; the failed Architect turn was not retried")
        return True
    finally:
        if state_lock is not None:
            daemon.release_ticket_cycle_lock(lock_file=state_lock)
        daemon.release_mailbox_sequence_lock(lock_file=sequence_lock)


def recover_failed_public_architect_admissions():
    """Release finite-cycle slots still charged to failed public requests.

    A public ticket-selecting request is charged against the finite-watch
    cycle limit the moment it is admitted. When its Architect turn then
    fails, the request is parked in ``failed/`` but the charge can outlive
    the run. Each parked request whose saved identity still matches, and
    whose admission token no live Implementer handoff claims, has its
    charge dropped so a later watch regains the slot.

    Returns:
      The number of charges released.
    """
    recovered = 0
    pattern = daemon.os.path.join(daemon.MAILBOX, "failed", "*-to-fable.md")
    for path in sorted(daemon.glob.glob(pattern), key=daemon.message_sequence):
        if daemon.retire_failed_public_architect_admission(path=path):
            recovered += 1
    return recovered


def recover_failed_architect_outcome():
    """Recover a finished fix-only Architect plan without a rerun.

    A fix-only maintenance request can complete its paid Architect turn —
    the resulting Implementer plan exists — and still be parked in
    ``failed/`` because a newer user request arrived mid-turn and made
    the turn look unresolved. This pass finds the one plan bound to the
    parked request's admission token, preserves it in ``prelaunch/``,
    requeues the newer user requests parked beside it, registers the
    plan's ticket cycle, and archives the completed request.

    Returns:
      The number of plans recovered.

    Raises:
      daemon.TicketCycleStateError: when the recovery lock cannot be
        taken, a parked request changed identity, several outcomes claim
        one token, or a required move failed — nothing is guessed.
    """
    lock_file = daemon.acquire_mailbox_sequence_lock()
    if lock_file is None:
        raise daemon.TicketCycleStateError("cannot lock outcome recovery")
    try:
        admissions = daemon.read_ticket_cycle_state()["architect_admissions"]
        recovered = 0
        failed = daemon.os.path.join(daemon.MAILBOX, "failed")
        for request_name, record in admissions.items():
            request_path = daemon.os.path.join(failed, request_name)
            if not daemon.os.path.lexists(request_path):
                continue
            request = daemon.read_cycle_message(path=request_path)
            if request != daemon.ARCHITECT_FIX_ONLY_REQUEST:
                continue
            if (daemon.hashlib.sha256(request.encode("utf-8")).hexdigest()
                    != record["sha256"]):
                raise daemon.TicketCycleStateError("failed request changed identity")
            token = daemon.architect_admission_token(
                request_name=request_name, digest=record["sha256"])
            outcomes = [
                path for directory in (failed,
                    daemon.os.path.join(daemon.MAILBOX, "prelaunch"))
                for path in daemon.glob.glob(
                    daemon.os.path.join(directory, "*-to-opus.md"))
                if daemon.message_claims_architect_admission(path, token)]
            if not outcomes:
                continue
            if len(outcomes) != 1:
                raise daemon.TicketCycleStateError(
                    "multiple bound outcomes")
            outcome_path = outcomes[0]
            outcome = daemon.read_cycle_message(path=outcome_path)
            if daemon.os.path.dirname(outcome_path) == failed:
                outcome_path, moved = daemon.verified_state_move(
                    dispatch_path=outcome_path,
                    directory=daemon.os.path.join(daemon.MAILBOX, "prelaunch"))
                if not moved:
                    raise daemon.TicketCycleStateError(
                        "could not preserve recovered plan")
            for path in daemon.glob.glob(daemon.os.path.join(failed, "*-to-fable.md")):
                name = daemon.os.path.basename(path)
                if (name in admissions or daemon.message_sequence(path)
                        < daemon.message_sequence(outcome_path)):
                    continue
                message = daemon.read_cycle_message(path=path)
                if daemon.architect_user_request_problem(message) is not None:
                    continue
                _restored, moved = daemon.verified_state_move(
                    dispatch_path=path, directory=daemon.MAILBOX)
                if not moved:
                    raise daemon.TicketCycleStateError(
                        "could not restore user request " + name)
            daemon.register_ticket_cycle_message(
                agent="opus", message=outcome,
                skip_redteam=(record["mode"] == "two-role"),
                architect_admission=token,
                implementer_request_name=daemon.os.path.basename(outcome_path))
            if not daemon.archive_consumed_message(dispatch_path=request_path):
                raise daemon.TicketCycleStateError(
                    "could not archive completed request")
            recovered += 1
            print("recovered " + daemon.os.path.basename(outcome_path)
                  + " without rerunning " + request_name)
        return recovered
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=lock_file)


def recover_failed_open_ticket_go():
    """Requeue a parked Architect GO whose audited candidate still stands.

    A GO can sit in ``failed/`` even though the decision it records is
    still valid: the named ticket is active in its implementation phase,
    the mode matches, and the saved candidate ref still names the audited
    commit. Such a GO is moved back to ``inflight/`` so the landing can
    finish without repeating the Architect's candidate audit. A GO whose
    ticket state no longer matches stays parked.

    Returns:
      The number of GO messages requeued.

    Raises:
      daemon.TicketCycleStateError: when a matching GO could not be
        moved back to ``inflight/``.
    """
    active = daemon.read_ticket_cycle_state()["active"]
    recovered = 0
    paths = daemon.glob.glob(
        daemon.os.path.join(daemon.MAILBOX, "failed", "*-to-daemon.md"))
    for path in sorted(paths, key=daemon.message_sequence):
        try:
            cycle_id, candidate, mode, problem = daemon._architect_go_request(
                message=daemon.read_cycle_message(path=path))
            record = active.get(cycle_id)
            if (problem is not None or record is None
                    or record["phase"] != "implementation"
                    or record["mode"] != mode
                    or daemon.candidate_commit_for_cycle(cycle_id) != candidate):
                continue
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        path, moved = daemon.verified_state_move(
            dispatch_path=path, directory=daemon.os.path.join(daemon.MAILBOX, "inflight"))
        if not moved:
            raise daemon.TicketCycleStateError(
                "accepted GO could not be restored for recovery")
        recovered += 1
        print("recovered accepted GO " + daemon.os.path.basename(path)
              + " without repeating the candidate audit")
    return recovered


def recover_prelaunch_messages():
    """Return every ``prelaunch/`` message to the mailbox root.

    ``prelaunch/`` retains a message durably after validation but before
    its agent process starts, so a crash in that window loses nothing. At
    daemon startup no agent can be running yet, so everything found there
    is simply requeued in sequence order.

    Returns:
      The number of messages requeued.

    Raises:
      daemon.TicketCycleStateError: when the recovery lock cannot be
        taken or a move fails; the remaining files stay in
        ``prelaunch/``.
    """
    sequence_lock = daemon.acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise daemon.TicketCycleStateError("cannot lock pre-launch recovery")
    recovered = 0
    try:
        pattern = daemon.os.path.join(daemon.MAILBOX, "prelaunch", "*-to-*.md")
        for path in sorted(daemon.glob.glob(pattern), key=daemon.message_sequence):
            daemon.read_cycle_message(path=path)
            recovered_path, moved = daemon.verified_state_move(
                dispatch_path=path, directory=daemon.MAILBOX)
            if not moved:
                raise daemon.TicketCycleStateError(
                    "could not requeue pre-launch message "
                    + daemon.os.path.basename(path))
            recovered += 1
            print("requeued pre-launch message " + recovered_path)
        return recovered
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=sequence_lock)


def restart_implementer_from_architect_handoff():
    """Reset the Implementer checkout and requeue its exact Architect plan.

    Used when an Implementer turn was killed mid-work and nothing worth
    keeping exists yet. The pass demands exactly one active Architect
    handoff across the mailbox folders, refuses when a candidate commit
    or a returned handoff already exists (finished work must go to the
    Architect instead), hard-resets the Implementer checkout to the
    ticket's starting commit, and returns the handoff to the mailbox
    root for a fresh launch.

    Returns:
      The mailbox-root path of the preserved Architect handoff.

    Raises:
      daemon.TicketCycleStateError: when zero or several handoffs match,
        finished work already exists, the base commit is missing, the
        checkout cannot be cleaned, or the handoff cannot be requeued.
    """
    daemon.recover_interrupted_mailbox_moves()
    sequence_lock = daemon.acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise daemon.TicketCycleStateError("cannot lock Implementer restart")
    try:
        ticket_state = daemon.read_ticket_cycle_state()
        matches = []
        for directory in (
                daemon.MAILBOX, daemon.os.path.join(daemon.MAILBOX, "inflight"),
                daemon.os.path.join(daemon.MAILBOX, "failed"),
                daemon.os.path.join(daemon.MAILBOX, "prelaunch")):
            for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-opus.md")):
                message = daemon.read_cycle_message(path=path)
                cycle_id, mode, _body, problem = daemon._ticket_flow_envelope(
                    message=message)
                active = ticket_state["active"].get(cycle_id)
                if (problem is not None or active is None
                        or active["phase"] != "implementation"
                        or active["commit"] is not None
                        or active["mode"] != mode):
                    continue
                if daemon.architect_handoff_problem(
                        message=message, cycle_id=cycle_id, mode=mode) is None:
                    matches.append((path, cycle_id))
        if len(matches) != 1:
            raise daemon.TicketCycleStateError(
                "Implementer restart needs exactly one active Architect "
                "handoff; found " + str(len(matches)))
        handoff, cycle_id = matches[0]
        if daemon.candidate_commit_for_cycle(cycle_id=cycle_id) is not None:
            raise daemon.TicketCycleStateError(
                "the Implementer already produced candidate C; return it "
                "to the Architect instead of restarting")
        for path in daemon.glob.glob(
                daemon.os.path.join(daemon.MAILBOX, "**", "*-to-fable.md"),
                recursive=True):
            message = daemon.read_cycle_message(path=path)
            returned_cycle, _mode, body, problem = daemon._ticket_flow_envelope(
                message=message)
            if (problem is None and returned_cycle == cycle_id
                    and "### IMPLEMENTER_HANDOFF:" in body):
                raise daemon.TicketCycleStateError(
                    "the Implementer already returned work for this cycle; "
                    "send it to the Architect instead of restarting")

        worktree = daemon.AGENT_CWD["opus"]
        daemon._symbolic_worktree_branch(
            worktree=worktree, expected_branch=daemon.IMPLEMENTER_BRANCH,
            label="Implementer")
        base = daemon.cycle_starting_commit(cycle_id=cycle_id)
        if not daemon.git_commit_exists(commit=base):
            raise daemon.TicketCycleStateError(
                "the Architect handoff names a missing base commit")
        daemon._run_git(worktree, ["reset", "--hard", base])
        daemon._run_git(worktree, ["clean", "-fd", "--", "."])
        if (daemon.worktree_head(worktree=worktree) != base
                or daemon._clean_worktree_status(worktree=worktree)):
            raise daemon.TicketCycleStateError(
                "Implementer work could not be discarded cleanly")

        if daemon.os.path.dirname(handoff) != daemon.MAILBOX:
            recovered, moved = daemon.verified_state_move(
                dispatch_path=handoff, directory=daemon.MAILBOX)
            if not moved:
                raise daemon.TicketCycleStateError(
                    "the exact Architect handoff could not be requeued")
            handoff = recovered
        print("Architect handoff preserved: " + handoff)
        print("Interrupted Implementer work discarded; ticket base: " + base)
        print("Restart ready: launch --watch with the desired Implementer.")
        return handoff
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=sequence_lock)


def restart_redteam_from_architect_handoff():
    """Reset the Red Team checkout and requeue its exact review request.

    The Red Team counterpart of the Implementer restart. The pass demands
    exactly one active closure or control-plane handoff, refuses when the
    review result already exists (a saved receipt or a recorded protected
    decision must reach the Architect instead), discards any interrupted
    audit snapshot, hard-resets the Red Team checkout to the current
    ``main`` commit, and returns the handoff to the mailbox root.

    Returns:
      The mailbox-root path of the preserved handoff.

    Raises:
      daemon.TicketCycleStateError: when zero or several handoffs match,
        the review already returned, the handoff fails validation, the
        checkout cannot be cleaned, or the requeue move fails.
    """
    daemon.recover_interrupted_mailbox_moves()
    sequence_lock = daemon.acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise daemon.TicketCycleStateError("cannot lock Red Team restart")
    try:
        matches = []
        for directory in (
                daemon.MAILBOX, daemon.os.path.join(daemon.MAILBOX, "inflight"),
                daemon.os.path.join(daemon.MAILBOX, "failed"),
                daemon.os.path.join(daemon.MAILBOX, "prelaunch")):
            for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-sol.md")):
                message = daemon.read_cycle_message(path=path)
                kind = daemon.sol_ticket_kind(message=message)
                if kind in {"closure", "control-plane"}:
                    matches.append((path, message, kind))
        if len(matches) != 1:
            raise daemon.TicketCycleStateError(
                "Red Team restart needs exactly one active handoff; found "
                + str(len(matches)))
        handoff, message, kind = matches[0]
        audit_cycle = None
        audit_commit = None
        if kind == "closure":
            problem = daemon.redteam_closure_problem(message=message)
            if problem is not None:
                raise daemon.TicketCycleStateError(problem)
            audit_cycle = daemon.redteam_closure_ticket(message=message)
            audit_commit = daemon.redteam_closure_commit(message=message)
            if daemon.any_matching_redteam_receipt(
                    cycle_id=audit_cycle, accepted_commit=audit_commit):
                raise daemon.TicketCycleStateError(
                    "the Red Team already returned its review; send that "
                    "result to the Architect instead of restarting")
        elif kind == "control-plane":
            audit_cycle, audit_commit, _body, problem = (
                daemon._redteam_control_plane_envelope(message=message))
            if problem is not None:
                raise daemon.TicketCycleStateError(problem)
            control = daemon.control_plane_ticket_state(
                cycle_id=audit_cycle, candidate_commit=audit_commit)
            if control is None or control["architect_candidate"] != (
                    audit_commit):
                raise daemon.TicketCycleStateError(
                    "the protected Red Team handoff lacks Architect GO(C)")
            if control["redteam_result"] is not None:
                raise daemon.TicketCycleStateError(
                    "the protected Red Team decision is already recorded")

        if audit_cycle is not None:
            daemon.discard_interrupted_audit_snapshot(
                cycle_id=audit_cycle, commit=audit_commit, agent="sol")
        worktree = daemon.AGENT_CWD["sol"]
        daemon._symbolic_worktree_branch(
            worktree=worktree, expected_branch=daemon.SOL_BRANCH, label="Red Team")
        target = daemon._exact_git_object(
            arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
            label="current main commit")
        daemon._run_git(worktree, ["reset", "--hard", target])
        daemon._run_git(worktree, ["clean", "-fd", "--", "."])
        if (daemon.worktree_head(worktree=worktree) != target
                or daemon._clean_worktree_status(worktree=worktree)):
            raise daemon.TicketCycleStateError(
                "Red Team work could not be discarded cleanly")

        if daemon.os.path.dirname(handoff) != daemon.MAILBOX:
            recovered, moved = daemon.verified_state_move(
                dispatch_path=handoff, directory=daemon.MAILBOX)
            if not moved:
                raise daemon.TicketCycleStateError(
                    "the exact Red Team handoff could not be requeued")
            handoff = recovered
        print("Architect-to-Red-Team handoff preserved: " + handoff)
        print("Interrupted Red Team work discarded; baseline: " + target)
        print("Restart ready: launch --watch with Red Team enabled.")
        return handoff
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=sequence_lock)


def recover_interrupted_mailbox_moves():
    """Finish or roll back state moves interrupted between hardlink steps.

    A durable state move creates a hardlink at the destination before
    unlinking the source, so a crash can leave the same file bytes
    reachable under two names plus a ``.state-guard`` link. For each
    message name with any leftover in ``inflight/``, this pass compares
    the inodes — the filesystem's identity numbers for the underlying
    files — across every possible location and applies the one safe
    repair: finish a move whose destination already exists, undo a claim
    whose root copy survived, or drop a guard link matching its source.
    Any inode disagreement stops the pass instead of guessing.

    Returns:
      The number of interrupted moves repaired.

    Raises:
      daemon.TicketCycleStateError: when the lock cannot be taken, a
        non-regular file sits where a state file should be, or the
        surviving copies disagree about which move was in flight.
    """
    sequence_lock = daemon.acquire_mailbox_sequence_lock()
    if sequence_lock is None:
        raise daemon.TicketCycleStateError("cannot lock mailbox-move recovery")
    recovered = 0
    inflight_directory = daemon.os.path.join(daemon.MAILBOX, "inflight")
    try:
        names = set()
        for pattern in ("*.md", "*.md" + daemon.STATE_GUARD_SUFFIX):
            for path in daemon.glob.glob(
                    daemon.os.path.join(inflight_directory, pattern)):
                names.add(daemon.blocker_message_name(path=path))
        for name in sorted(names, key=daemon.message_sequence):
            inflight = daemon.os.path.join(inflight_directory, name)
            guard = inflight + daemon.STATE_GUARD_SUFFIX
            root = daemon.os.path.join(daemon.MAILBOX, name)
            destinations = [
                daemon.os.path.join(daemon.DONE, name),
                daemon.os.path.join(daemon.MAILBOX, "failed", name),
                daemon.os.path.join(daemon.MAILBOX, "prelaunch", name),
            ]

            def inode(path):
                """Read one location's inode for the identity comparison.

                Arguments:
                  path = one possible location of the moved message.

                Returns:
                  The inode number, or ``None`` when nothing exists at
                  ``path``.

                Raises:
                  daemon.TicketCycleStateError: when something exists
                    there but is not a regular file, such as a symlink.
                """
                value = daemon.regular_inode(path=path)
                if value is None and daemon.os.path.lexists(path):
                    raise daemon.TicketCycleStateError(
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
                raise daemon.TicketCycleStateError(
                    "interrupted mailbox move changed its guard identity")
            source_inode = known[0] if known else None

            if root_inode is not None:
                if (source_inode is None or root_inode != source_inode
                        or guard_inode is not None or terminal):
                    raise daemon.TicketCycleStateError(
                        "interrupted mailbox claim has conflicting states")
                daemon.os.unlink(inflight)
                daemon.fsync_directory(directory=inflight_directory)
                recovered += 1
                print("recovered interrupted claim " + name)
                continue

            if terminal:
                if (len(terminal) != 1 or source_inode is None
                        or terminal[0][1] != source_inode):
                    raise daemon.TicketCycleStateError(
                        "interrupted mailbox move has conflicting destinations")
                for leftover in (inflight, guard):
                    if daemon.os.path.lexists(leftover):
                        daemon.os.unlink(leftover)
                daemon.fsync_directory(directory=inflight_directory)
                recovered += 1
                print("finished interrupted mailbox move " + name)
                continue

            if inflight_inode is not None and guard_inode == inflight_inode:
                daemon.os.unlink(guard)
                daemon.fsync_directory(directory=inflight_directory)
                recovered += 1
                print("removed interrupted state guard for " + name)
            elif guard_inode is not None:
                raise daemon.TicketCycleStateError(
                    "interrupted mailbox guard has no recoverable source")
        return recovered
    finally:
        daemon.release_mailbox_sequence_lock(lock_file=sequence_lock)


def blocked_redteam_directory():
    """Name the durable queue for protected work that needs the Red Team.

    Returns:
      The absolute path of the mailbox subdirectory where a two-role
      watch parks protected tool-edit requests it must not run.
    """
    return daemon.os.path.join(daemon.MAILBOX, daemon.BLOCKED_REDTEAM_DIRECTORY)


def recover_blocked_redteam_messages(skip_redteam=False):
    """Keep old tool-edit requests parked for external maintenance.

    The reminder prints on a full-role watch; the two-role watch that
    parks a request prints its own louder message at that moment.

    Arguments:
      skip_redteam = True when the Sol route is disabled.

    Returns:
      0; the parked requests are reported, never moved.
    """
    directory = daemon.blocked_redteam_directory()
    parked = daemon.glob.glob(daemon.os.path.join(directory, "*-to-opus.md"))
    if parked and not skip_redteam:
        print("old protected tool requests remain parked for external "
              "ai/tools maintenance; their backlog tickets stay Open")
    return 0


def block_protected_ticket_without_redteam(path):
    """Durably block one validated protected handoff before reservation.

    Arguments:
      path = the pending Implementer message.

    Returns:
      True when the message declared the protected-control-plane
      ticket class and was moved into the durable blocked queue with
      the loud restart-without-skip-redteam message; False for every
      other message or a failed move.
    """
    match = daemon.PENDING_MESSAGE_RE.fullmatch(daemon.os.path.basename(path))
    if match is None or match.group(1) != "opus":
        return False
    try:
        message = daemon.read_cycle_message(path=path)
        evidence = daemon.prepare_implementer_evidence_contract(message=message)
    except (OSError, ValueError, daemon.TicketCycleStateError):
        return False
    if evidence["ticket_class"] != "protected-control-plane":
        return False
    blocked = daemon.move_without_overwrite(
        path=path, directory=daemon.blocked_redteam_directory())
    if blocked is None:
        return False
    print("BLOCKED_RED_TEAM_REQUIRED: Protected control-plane tickets "
          "require Red Team review. This daemon was started with "
          "--skip-redteam, so " + daemon.os.path.basename(path)
          + " was not run. Restart without --skip-redteam; the exact "
            "request was preserved at " + blocked + ".")
    return True


def recover_before_dispatch(fix_only=False, skip_redteam=False):
    """Recover restart-safe mailbox state before a live dispatch pass.

    The recoveries run in a fixed order: interrupted moves, failed
    Architect outcomes, open-ticket GOs, maintenance admissions (in
    fix-only watches), Implementer returns, deliveries, and
    pre-launch reservations, prelaunch messages, public admissions,
    and the blocked Red Team queue, ending with ticket-cycle
    reconciliation.

    Arguments:
      fix_only     = True in a fix-only watch.
      skip_redteam = True in a two-role watch.

    Returns:
      The reconciled ticket-cycle report from the final step.

    Raises:
      daemon.TicketCycleStateError: when a protected cycle recorded a
        failed health check; that cycle is recovery-only.
    """
    failed_health = daemon.control_plane_health_failure()
    if failed_health is not None:
        cycle_id, evidence = failed_health
        raise daemon.TicketCycleStateError(
            "CONTROL_PLANE_HEALTH_FAILED: protected cycle " + cycle_id
            + " is recovery-only; inspect " + evidence
            + ", preserve its recorded landing, and repair it with the "
              "trusted controller before dispatching new work")
    daemon.recover_interrupted_mailbox_moves()
    daemon.recover_failed_architect_outcome()
    daemon.recover_failed_open_ticket_go()
    if fix_only:
        daemon.recover_failed_maintenance_admission()
    daemon.recover_failed_implementer_returns()
    daemon.recover_implementer_deliveries()
    daemon.recover_failed_implementer_preflight()
    daemon.recover_prelaunch_messages()
    daemon.recover_failed_public_architect_admissions()
    daemon.recover_blocked_redteam_messages(skip_redteam=skip_redteam)
    return daemon.reconcile_ticket_cycle_state()


def implementer_reservation_preflight_problem(path, message):
    """Return a permanent pre-launch problem before a slot is reserved.

    Only defects a retry cannot fix belong here: a NUL byte, an
    invalid timeout, an unverifiable timeout history, a
    placeholder-only body, or a malformed flow envelope.

    Arguments:
      path    = the pending Implementer message.
      message = its decoded text.

    Returns:
      A printable problem sentence, or ``None`` when the message may
      reserve its slot.
    """
    if "\x00" in message:
        return "the message contains a NUL byte"
    if not daemon.valid_duration(value=daemon.DISPATCH_TIMEOUT_MINUTES,
                          strictly_positive=True):
        return "the dispatch timeout is invalid"
    try:
        daemon.timeout_events(name=daemon.os.path.basename(path))
    except (OSError, ValueError, daemon.json.JSONDecodeError,
            OverflowError, RecursionError) as exc:
        return "timeout history cannot be verified: " + str(exc)
    _, _, body, problem = daemon._ticket_flow_envelope(message=message)
    if problem is None and daemon.placeholder_in(message=body) is not None:
        return "the Implementer body is only a template placeholder"
    return problem


def reserve_architect_ticket_before_claim(path, skip_redteam=False):
    """Durably charge one ticket-selecting request before Architect launch.

    The exact request basename and SHA-256 stay charged until the same
    turn's first Implementer handoff carries their admission token. This
    closes the interval in which a finite watch could otherwise launch
    two public requests before either had reached the Implementer lane.

    Arguments:
      path         = the pending Architect request in the mailbox root.
      skip_redteam = True when this watch runs without the Red Team
                     route; the charge records the matching topology.

    Returns:
      A ``(deferred_reason, token)`` pair. ``(None, None)`` means the
      message is not a chargeable public request and ordinary dispatch
      should continue. ``(None, token)`` means the slot is charged (or
      was already charged with the same identity) and the token must
      appear in the turn's Implementer handoff. ``(reason, None)`` means
      admission must wait — an earlier handoff must reserve first, a
      permanent-note transition is pending, or the finite watch has no
      slot left — and the message stays untouched in the root.

    Raises:
      daemon.TicketCycleStateError: when a saved charge for this name
        no longer matches the file's sequence and digest.
    """
    controller = daemon._ACTIVE_WATCH_RENDEZVOUS
    if controller is None:
        return None, None
    name = daemon.os.path.basename(path)
    match = daemon.PENDING_MESSAGE_RE.fullmatch(name)
    if match is None or match.group(1) != "fable":
        return None, None
    try:
        message = daemon.read_cycle_message(path=path)
    except (OSError, ValueError, daemon.TicketCycleStateError):
        return None, None
    maintenance = message == daemon.ARCHITECT_FIX_ONLY_REQUEST
    finite = controller.ticket_cycle_limit_value() is not None
    if not finite and not maintenance:
        return None, None
    if (not maintenance
            and (not message.startswith(daemon.SOL_SEVERITY_HEADER)
            or daemon.architect_user_request_problem(message=message) is not None
            or daemon.placeholder_in(
                message=daemon.architect_user_request_body(message=message))
            is not None
            or "\x00" in message)):
        return None, None
    digest = daemon.hashlib.sha256(message.encode("utf-8")).hexdigest()
    topology = daemon.canonical_ticket_cycle_topology(
        skip_redteam=skip_redteam)
    record = {"mode": topology, "sequence": daemon.message_sequence(path),
              "sha256": digest}
    token = daemon.architect_admission_token(
        request_name=name, digest=digest)
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        existing = state["architect_admissions"].get(name)
        if existing is not None:
            if existing != record:
                raise daemon.TicketCycleStateError(
                    "saved public Architect admission changed identity")
            return None, token
        for earlier_path in daemon.pending_messages():
            if daemon.message_sequence(earlier_path) >= record["sequence"]:
                break
            earlier_match = daemon.PENDING_MESSAGE_RE.fullmatch(
                daemon.os.path.basename(earlier_path))
            if earlier_match is None or earlier_match.group(1) != "opus":
                continue
            try:
                earlier_message = daemon.read_cycle_message(path=earlier_path)
            except (OSError, ValueError, daemon.TicketCycleStateError):
                continue
            earlier_cycle, earlier_mode, _body, earlier_problem = (
                daemon._ticket_flow_envelope(message=earlier_message))
            if (earlier_problem is None
                    and daemon.ticket_cycle_mode_is_enabled(
                        mode=earlier_mode, skip_redteam=skip_redteam)
                    and earlier_cycle not in state["active"]):
                return ("an earlier Implementer handoff must reserve its "
                        "ticket before this public request", None)
        if daemon.architect_notes_transition_pending():
            return ("a permanent-note admin turn or P landing is still "
                    "pending; no newer ticket may be admitted", None)
        used = daemon.finite_cycle_capacity_used(
            state=state, skip_redteam=skip_redteam)
        if (finite and used >= controller.ticket_cycle_limit_value()):
            return ("the finite watch has already reserved all "
                    + str(controller.ticket_cycle_limit_value())
                    + " ticket cycle(s)", None)
        state["architect_admissions"][name] = record
        daemon.write_ticket_cycle_state(state=state)
        return None, token
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def release_architect_ticket_admission(token):
    """Atomically retire one exact public request that created no ticket.

    Arguments:
      token = the admission token quoted by the no-ticket receipt.

    Raises:
      daemon.TicketCycleStateError: when the charged record is absent
        or its digest disagrees — the slot is then left charged
        rather than freed under a mismatched identity.
    """
    request_name, digest = daemon.split_architect_admission_token(token=token)
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        record = state["architect_admissions"].get(request_name)
        if record is None or record["sha256"] != digest:
            raise daemon.TicketCycleStateError(
                "public Architect admission changed before release")
        del state["architect_admissions"][request_name]
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def reserve_implementer_ticket_before_claim(path, skip_redteam=False):
    """Reserve one finite-watch ticket slot while its root file is untouched.

    Malformed or otherwise invalid messages are left for the ordinary
    dispatch validator, which can park them with a concrete explanation.
    Only the capacity refusal is returned here, because a ticket over the
    finite-cycle limit is valid work for a later watch, not a failed
    message.

    Arguments:
      path         = the pending Implementer handoff in the mailbox root.
      skip_redteam = True when this watch runs without the Red Team
                     route.

    Returns:
      A ``(deferred_reason, new_cycle_id)`` pair. ``(None, None)`` means
      no new reservation was created here — the message is not a valid
      new-ticket handoff, or its ticket was already reserved — and
      ordinary dispatch decides what happens next. ``(reason, None)``
      defers a valid ticket that exceeds the finite-cycle capacity.
      ``(None, cycle_id)`` reports a reservation this call created for a
      plain handoff, so the caller can release it if no agent process
      ever launches; a handoff converting a charged public admission
      returns ``(None, None)`` because the admission record already owns
      its slot.
    """
    match = daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(path))
    if match is None or match.group(1) != "opus":
        return None, None
    try:
        message = daemon.read_cycle_message(path=path)
    except (OSError, ValueError, daemon.TicketCycleStateError):
        return None, None
    if not message.startswith(daemon.MAILBOX_FLOW_HEADER):
        return None, None
    preflight_problem = daemon.implementer_reservation_preflight_problem(
        path=path, message=message)
    if preflight_problem is not None:
        return None, None
    try:
        evidence = daemon.prepare_implementer_evidence_contract(message=message)
    except (OSError, daemon.TicketCycleStateError):
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
        _, _, created = daemon.register_ticket_cycle_message(
            agent="opus", message=message,
            skip_redteam=skip_redteam,
            return_reservation=True,
            implementer_request_name=daemon.os.path.basename(path),
            path_scope=evidence["allowed_paths"],
            ticket_class=evidence["ticket_class"])
    except daemon.TicketCycleLimitDeferred as exc:
        return str(exc), None
    except (OSError, daemon.TicketCycleStateError):
        return None, None
    if not created:
        return None, None
    parsed_cycle, _, _, problem = daemon._ticket_flow_envelope(message=message)
    admission_name, _admission_digest, admission_problem = (
        daemon._ticket_architect_admission(message=message))
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

    Returns:
      True when the lane ended without a failure: every message was
      consumed, deferred by a reservation, or intentionally left queued
      by the finite-cycle limit or a due rendezvous. False when a
      dispatch failed, a maintenance request needed a ``--fix-only``
      watcher, or a token or interrupt stop ended the lane early; an
      unresolved head message then blocks the rest of its lane.
    """
    all_consumed = True
    for path in paths:
        if daemon._TOKEN_EXHAUSTION_STOP.is_set() \
                or daemon._WATCH_INTERRUPT_STOP.is_set():
            all_consumed = False
            break
        try:
            maintenance = (daemon.read_cycle_message(path=path)
                           == daemon.ARCHITECT_FIX_ONLY_REQUEST)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            maintenance = False
        if skip_redteam and daemon.block_protected_ticket_without_redteam(path=path):
            # This configuration refusal is a consumed queue action, not a
            # ticket cycle and not an Implementer failure. Continue to any
            # ordinary two-role work behind it.
            continue
        if (maintenance
                and (not fix_only
                     or daemon.active_ticket_cycle_count(
                         skip_redteam=skip_redteam,
                         exclude_admission=daemon.os.path.basename(path))
                     or any(candidate.endswith("-to-opus.md")
                            for candidate in daemon.pending_messages()))):
            if not fix_only:
                print("deferred " + daemon.os.path.basename(path)
                      + ": needs a --fix-only watcher; left queued.")
            all_consumed = False
            continue
        notes_admin = False
        try:
            notes_admin = daemon.regular_file_has_prefix(
                path=path,
                prefix=daemon.MAILBOX_ADMIN_HEADER.encode("ascii"))
        except (OSError, ValueError):
            pass
        controller = (daemon._ACTIVE_WATCH_RENDEZVOUS
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
            daemon._RENDEZVOUS_LOCAL.permit = permit
        new_reservation_cycle = None
        architect_admission = None
        consumed = False
        try:
            if not dry_run:
                deferred, architect_admission = (
                    daemon.reserve_architect_ticket_before_claim(
                        path=path, skip_redteam=skip_redteam))
                if deferred is not None:
                    print("deferred " + daemon.os.path.basename(path) + ": "
                          + deferred + "; root message remains untouched.")
                    continue
                deferred, new_reservation_cycle = (
                    daemon.reserve_implementer_ticket_before_claim(
                        path=path,
                        skip_redteam=skip_redteam))
                if deferred is not None:
                    print("deferred " + daemon.os.path.basename(path) + ": "
                          + deferred + "; root message remains untouched.")
                    # A later file may continue an already reserved ticket.
                    continue
            consumed = daemon.dispatch(
                path=path, dry_run=dry_run, fix_only=fix_only,
                skip_redteam=skip_redteam,
                new_reservation_cycle=new_reservation_cycle,
                architect_admission=architect_admission)
        finally:
            if controller is not None:
                try:
                    if (new_reservation_cycle is not None
                            and not consumed
                            and not permit.launched):
                        daemon.release_unstarted_ticket_reservation(
                            cycle_id=new_reservation_cycle)
                    del daemon._RENDEZVOUS_LOCAL.permit
                finally:
                    controller.finish_attempt(permit=permit)
        if not consumed and architect_admission is not None:
            daemon.retire_failed_public_architect_admission(
                path=daemon.os.path.join(
                    daemon.MAILBOX, "failed", daemon.os.path.basename(path)))
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
    daemon._TOKEN_EXHAUSTION_STOP.clear()
    daemon._WATCH_INTERRUPT_STOP.clear()
    if not dry_run:
        try:
            daemon.read_ticket_cycle_state()
        except (OSError, ValueError, daemon.TicketCycleStateError) as exc:
            print("refused mailbox pass: cannot verify ticket-cycle state ("
                  + str(exc) + "); no new role work was started. Repair the "
                  "saved ticket-cycle state, then run the watcher again.")
            return False
    all_backlog = daemon.pending_messages()
    all_daemon_paths = [
        path for path in all_backlog
        if daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(path)).group(1)
        == "daemon"]
    daemon_paths = [
        path for path in all_daemon_paths
        if daemon.message_is_enabled_for_topology(
            path=path, skip_redteam=skip_redteam)]
    policy_problem = daemon.role_contract_snapshot_problem()
    policy_recovery_only = (
        policy_problem is not None and daemon.architect_notes_transition_pending())
    if policy_recovery_only:
        # The Architect primary already contains a proposed new contract.
        # The old process may finish only that exact P landing. It must not
        # land an ordinary candidate or start a role under stale policy.
        daemon_paths = [
            path for path in daemon_paths
            if daemon.regular_file_has_prefix(
                path=path,
                prefix=(daemon.MAILBOX_RETURN_HEADER
                        + "architect-notes-go").encode("ascii"))]
    daemon_outcome = True
    for daemon_path in daemon_paths:
        # This GO belongs to a ticket already admitted against the finite
        # limit. Always finish its durable landing/archive recovery. The
        # positive limit gates new role work in drain_lane(), never this
        # already-admitted daemon transition. A Ctrl-C arriving while the
        # transition runs takes effect right after it, never inside it.
        with daemon.DeferredInterrupts():
            outcome = daemon.consume_daemon_message(
                path=daemon_path, dry_run=dry_run, return_outcome=True)
        if outcome == daemon.DAEMON_NOTE_DEFERRED:
            # An unlanded P can wait behind a later, already-admitted
            # ordinary GO. Continue the daemon lane so that exact ticket can
            # reach L and clear P's idle-boundary requirement.
            if policy_recovery_only:
                return daemon.ROLE_CONTRACT_RESTART_REQUIRED
            daemon_outcome = False
            continue
        if outcome == daemon.DAEMON_CONTROL_PLANE_WAITING:
            # The exact GO remains in inflight/ while Sol supplies the
            # second key. Compatible role work, including that review, may
            # continue in this watch.
            continue
        if outcome != daemon.DAEMON_MESSAGE_CONSUMED:
            if policy_recovery_only:
                return daemon.ROLE_CONTRACT_RESTART_REQUIRED
            daemon_outcome = False
            break
        # Check after each daemon message, not after the complete lane. A P
        # landing therefore cannot release a second daemon request or a role
        # while this process still holds the old policy snapshot.
        if (policy_recovery_only
                or (policy_problem is None
                    and daemon.role_contract_snapshot_problem() is not None)):
            return daemon.ROLE_CONTRACT_RESTART_REQUIRED
    if policy_recovery_only:
        return daemon.ROLE_CONTRACT_RESTART_REQUIRED
    agent_backlog = [path for path in all_backlog
                     if path not in all_daemon_paths]
    backlog = [
        path for path in agent_backlog
        if daemon.message_is_enabled_for_topology(
            path=path, skip_redteam=skip_redteam)]
    blockers = daemon.inflight_lane_blockers(skip_redteam=skip_redteam)
    admin_paths = []
    for candidate in backlog:
        match = daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(candidate))
        if match is None or match.group(1) != "fable":
            continue
        try:
            admin_prefix = daemon.regular_file_has_prefix(
                path=candidate,
                prefix=daemon.MAILBOX_ADMIN_HEADER.encode("ascii"))
        except (OSError, ValueError):
            admin_prefix = False
        if admin_prefix:
            admin_paths.append(candidate)
    if admin_paths:
        admin_paths.sort(key=daemon.message_sequence)
        admin_path = admin_paths[0]
        boundary = daemon.message_sequence(admin_path)
        state = daemon.read_ticket_cycle_state()
        limit_reached = (
            daemon._ACTIVE_WATCH_RENDEZVOUS is not None
            and daemon._ACTIVE_WATCH_RENDEZVOUS.ticket_cycle_limit_reached())
        earlier = ([] if limit_reached else [
            candidate for candidate in backlog
            if (candidate != admin_path
                and daemon.message_sequence(candidate) < boundary)])
        admitted = [candidate for candidate in backlog
                    if (candidate != admin_path
                        and daemon.message_belongs_to_active_cycle(
                            path=candidate,
                            active_cycles=state["active"]))]
        older_work = list(dict.fromkeys(earlier + admitted))
        admin_problem = None
        try:
            daemon.require_no_ordinary_landing_transition(
                current_dispatch_path=admin_path)
        except (OSError, daemon.TicketCycleStateError) as exc:
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
            print("deferred " + daemon.os.path.basename(admin_path)
                  + ": permanent-note administration waits for an idle "
                  "boundary (" + explanation + ").")
    elif daemon.architect_notes_transition_pending():
        # A validated P request may wait behind an older admitted cycle.
        # Continue that cycle, but never admit unrelated/newer work before P
        # reaches main and the clean role baselines.
        active = daemon.read_ticket_cycle_state()["active"]
        backlog = [candidate for candidate in backlog
                   if daemon.message_belongs_to_active_cycle(
                       path=candidate, active_cycles=active)]
    # Finish an admitted ticket before an older, unrelated user request in
    # the same role lane. Otherwise recovery mail can wait forever behind a
    # request that the finite cycle limit cannot yet admit.
    active = daemon.read_ticket_cycle_state()["active"]
    backlog.sort(key=lambda path: (
        not daemon.message_belongs_to_active_cycle(path=path, active_cycles=active),
        daemon.message_sequence(path)))
    if all_backlog or daemon_paths:
        daemon.report_demand(backlog=all_backlog, skip_redteam=skip_redteam)
    if skip_redteam:
        daemon.report_deferred_sol_messages()
    if not backlog:
        if not blockers:
            return daemon_outcome if daemon_paths else None
        for cwd in sorted(blockers):
            daemon.report_inflight_lane_block(
                blocker_paths=blockers[cwd],
                pending_count=0)
        return False
    lanes = {}
    for path in backlog:
        name = daemon.os.path.basename(path)
        agent = daemon.PENDING_MESSAGE_RE.match(name).group(1)
        cwd = daemon.mailbox_lane_cwd(agent=agent)
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
    outcome_lock = daemon.threading.Lock()

    def drain_and_record(cwd, paths, dry_run, fix_only, skip_redteam):
        """Run one cwd lane and retain failure even if its worker raises.

        Token-exhaustion and authority-violation errors are collected
        for the caller under the outcome lock; any other exception
        marks the lane not consumed. The lane's outcome is always
        recorded.

        Arguments:
          cwd          = the lane's working directory, the key under
                         which the outcome is recorded.
          paths        = the lane's message files, in sequence order.
          dry_run      = passed through to drain_lane.
          fix_only     = passed through to drain_lane.
          skip_redteam = passed through to drain_lane.
        """
        try:
            consumed = daemon.drain_lane(
                paths=paths, dry_run=dry_run, fix_only=fix_only,
                skip_redteam=skip_redteam)
        except daemon.RoleTokenExhaustionError as exc:
            with outcome_lock:
                token_errors.append(exc)
            consumed = False
        except daemon.ImplementerAuthorityViolationError as exc:
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
        daemon.report_inflight_lane_block(
            blocker_paths=blockers[cwd],
            pending_count=len(lanes.get(cwd, [])))

    for cwd in sorted(lanes):
        if cwd in blockers:
            lane_outcomes[cwd] = False
            continue
        worker = daemon.threading.Thread(target=drain_and_record,
                                  kwargs={"cwd": cwd,
                                          "paths": lanes[cwd],
                                          "dry_run": dry_run,
                                          "fix_only": fix_only,
                                          "skip_redteam": skip_redteam})
        worker.start()
        workers.append(worker)
    # A Ctrl-C here must not abandon running lane turns behind released
    # watch locks. The first interrupt stops new claims and keeps waiting;
    # a second interrupt kills the running agent turns so their lanes can
    # park the requests in failed/ and finish quickly. Either way the
    # interrupt is honored only after every lane worker has returned.
    interrupts = 0
    remaining = list(workers)
    while remaining:
        try:
            remaining[0].join()
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts == 1:
                daemon._WATCH_INTERRUPT_STOP.set()
                print("interrupt received: no new role turn will start; "
                      "waiting for running turns to finish (press Ctrl-C "
                      "again to kill them).", flush=True)
            else:
                print("second interrupt: killing running role turns; "
                      "their requests will be parked in failed/ for "
                      "requeue.", flush=True)
                daemon.kill_live_agent_processes()
            continue
        remaining.pop(0)
    if not dry_run:
        daemon.recover_failed_public_architect_admissions()
    if interrupts:
        raise KeyboardInterrupt
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
    """Print how many root Red Team messages this watch leaves queued.

    A watch launched without the Red Team route must not touch
    ``*-to-sol.md`` files. This report gives the user the exact count,
    so the silence is visibly a choice and not a loss; nothing prints
    when no such message is waiting.
    """
    deferred = len(daemon.deferred_sol_messages())
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
        name = daemon.os.path.basename(path)
        agent = daemon.PENDING_MESSAGE_RE.match(name).group(1)
        depth[agent] = depth[agent] + 1
    counts = daemon.backlog_severity_counts()
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
    daemon.report_landing_debt()


def landing_debt_snapshot():
    """Measure only saved, unlanded Implementer candidates.

    The Architect primary branch is a planning lane, so comparing that branch
    with main mistakes completed landings and protected-note edits for code
    awaiting review. Candidate refs plus active ticket state are the daemon's
    durable authority for work that can still need an Architect decision.

    Returns:
      A dictionary with four keys. ``available`` = False when any state
      read or Git command failed; nothing is guessed. ``stat`` = a
      one-line summary such as "1 active candidate, 42 changed lines",
      empty when no active candidate is waiting. ``changed_lines`` =
      summed insertions and deletions across every active candidate's
      diff against its ticket base. ``returncode`` = 0 on success, else
      the failing command's exit status.
    """
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        ticket_state = daemon.read_ticket_cycle_state()
        candidate_state = daemon.read_candidate_state()
        changed_lines = 0
        diff_ranges = []
        for cycle_id, saved in candidate_state["cycles"].items():
            active = ticket_state["active"].get(cycle_id)
            if active is None and cycle_id not in ticket_state["completed"]:
                raise daemon.TicketCycleStateError(
                    "candidate debt has no active or completed ticket")
            if active is None or active["phase"] != "implementation":
                # A candidate retained across GO archival or checkout handoff
                # is recovery authority, not new work needing another audit.
                continue
            record = daemon.candidate_record_locked(
                cycle_id=cycle_id, ticket_state=ticket_state,
                candidate_state=candidate_state, recover=False)
            if record is None or record != saved:
                raise daemon.TicketCycleStateError(
                    "active candidate debt lost its durable identity")
            base = daemon.cycle_starting_commit(cycle_id)
            diff_ranges.append((base, saved["commit"]))
        for base, candidate in diff_ranges:
            process = daemon.subprocess.run(
                ["git", "diff", "--shortstat", base + ".." + candidate],
                stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.PIPE,
                text=True, cwd=daemon.AGENT_CWD["fable"], check=False)
            if process.returncode != 0:
                return {
                    "available": False, "stat": "", "changed_lines": 0,
                    "returncode": process.returncode}
            for count, _keyword in daemon.re.findall(
                    r"(\d+) (insertion|deletion)", process.stdout):
                changed_lines = changed_lines + int(count)
        active_candidates = len(diff_ranges)
    except (OSError, ValueError, daemon.TicketCycleStateError):
        return {
            "available": False, "stat": "", "changed_lines": 0,
            "returncode": 1}
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)
    stat_line = ""
    if active_candidates:
        noun = "candidate" if active_candidates == 1 else "candidates"
        stat_line = (str(active_candidates) + " active " + noun + ", "
                     + str(changed_lines) + " changed lines")
    return {
        "available": True, "stat": stat_line,
        "changed_lines": changed_lines, "returncode": 0}


def report_landing_debt(snapshot=None):
    """Print saved candidate size without treating role branches as debt.

    Arguments:
      snapshot = a landing-debt snapshot, or ``None`` to take one.

    Returns:
      The snapshot that was reported. Above the line limit, a hint
      reminds the Architect to squash-land the audited units.
    """
    if snapshot is None:
        snapshot = daemon.landing_debt_snapshot()
    if not snapshot["available"]:
        print("landing debt: unavailable; active candidate state could not "
              "be measured (check exited "
              + str(snapshot["returncode"]) + ")")
        return snapshot
    if snapshot["stat"] == "":
        print("landing debt: none; no saved active candidate is waiting")
        return snapshot
    print("landing debt: " + snapshot["stat"])
    if snapshot["changed_lines"] > daemon.LANDING_DEBT_LINE_LIMIT:
        print("  hint: more than " + str(daemon.LANDING_DEBT_LINE_LIMIT)
              + " unlanded lines means at least one full audit trail "
              "is overdue; squash-land the audited unit(s) to main "
              "now, one unit per commit "
              "(.claude/FABLE_ROLE.md, Landing granularity).")
    return snapshot
