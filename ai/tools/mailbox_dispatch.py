"""One message dispatch: launch, monitor, and verify a role turn.

A dispatch takes one claimed mailbox message, launches the assigned
role's command-line program inside that role's own work folder, and
watches the process until the turn ends. This file owns that
lifecycle: the launch, the periodic progress line, the timeout kill,
the process-group kill that stops a turn together with every helper
program it started, and the verified move of the finished request
into ``done/`` or ``failed/``. It does not interpret the returned
text; the envelope part matches returns to their tickets.

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
    "kill_agent_process",
    "kill_live_agent_processes",
    "park_failed_outcome",
    "park_failed_turn_outcome",
    "park_prelaunch_outcome",
    "provider_is_out_of_tokens",
    "provider_failure_guidance",
    "reserve_dispatch_log_path",
    "dispatch",
    "dispatch_under_main_checkout_lock",
    "park_failed_message",
    "park_prelaunch_message",
    "regular_inode",
    "regular_file_has_prefix",
    "restore_state_source",
    "remove_state_guard",
    "verified_state_move",
    "archive_consumed_message",
)


def provider_is_out_of_tokens(agent, reply_lines, implementer_provider=None):
    """Recognize terminal account exhaustion, not transient API failures.

    Arguments:
      agent = Mailbox role name.
      reply_lines = Completed provider-log lines.

    Returns:
      True for a known account limit.
    """
    if agent not in {"fable", "opus", "sol"}:
        return False
    provider = (daemon.IMPLEMENTER_RUNTIME["provider"]
                if agent == "opus" and implementer_provider is None
                else implementer_provider)
    if agent == "sol":
        markers = daemon.SOL_TOKEN_EXHAUSTION_MARKERS
    elif agent == "opus" and provider == "ollama":
        markers = daemon.OLLAMA_TOKEN_EXHAUSTION_MARKERS
    else:
        markers = daemon.CLAUDE_TOKEN_EXHAUSTION_MARKERS
    tail = "\n".join(reply_lines[-24:]).replace("’", "'").casefold()
    if (agent != "sol" and daemon.re.search(
            r"you've hit your (5-hour|five-hour|7-day|seven-day) "
            r"limit\b.*\bresets\b",
            tail)):
        return True
    return any(marker in tail for marker in markers)


def provider_failure_guidance(agent, reply_lines):
    """Return provider-specific recovery text for a failed role process.

    The last lines of the role's reply are searched for known failure
    marks; the returned sentence names the failing service and the
    exact command to run before requeueing from ``failed/``.

    Arguments:
      agent       = the dispatched role.
      reply_lines = the role process's captured output lines.

    Returns:
      One recovery sentence; a generic inspect-the-log sentence when
      no known mark matches.
    """
    text = "\n".join(reply_lines[-40:]).replace("’", "'").casefold()
    if agent != "opus" or daemon.IMPLEMENTER_RUNTIME["provider"] != "ollama":
        if "not logged in" in text:
            return ("the Claude CLI is logged out; run `claude` in a "
                    "terminal, type /login, then requeue from failed/.")
        return "dispatch failed; inspect the relay log before requeueing."
    model = daemon.IMPLEMENTER_RUNTIME["model"]
    if any(mark in text for mark in (
            "connection refused", "could not connect", "service unavailable",
            "dial tcp", "ollama is not running")):
        return ("the Ollama service is unavailable; start it, then requeue "
                "from failed/.")
    if any(mark in text for mark in (
            "pull model manifest", "model not found", "does not exist")):
        return ("the Ollama model is unavailable; run `ollama pull "
                + model + "`, then requeue from failed/.")
    if any(mark in text for mark in (
            "out of memory", "cannot allocate memory", "resource exhausted")):
        return ("the Ollama runtime ran out of local resources; preserve the "
                "candidate, free resources, then requeue from failed/.")
    if "not logged in" in text or "authentication" in text:
        return ("the Ollama/Claude Code integration reported an authentication "
                "failure; verify `ollama launch claude --model " + model
                + "` directly, then requeue from failed/.")
    return ("the Ollama/Claude Code integration failed; inspect the relay "
            "log, verify `ollama launch claude --model " + model
            + "`, then requeue from failed/.")


def kill_agent_process(proc):
    """Stop one dispatched agent CLI together with its process group.

    Arguments:
      proc = the launched agent process.

    The CLI is launched as its own session leader, so its group id equals
    its pid and killing the group also stops every tool subprocess the CLI
    itself started; nothing keeps writing to a role worktree after the
    daemon declares the turn dead. A process that cannot be group-killed
    (already reaped, or a test double without a real pid) falls back to
    the direct kill. The process is always reaped before returning.
    """
    try:
        daemon.os.killpg(proc.pid, daemon.signal.SIGKILL)
    except (AttributeError, TypeError, ProcessLookupError,
            PermissionError, OSError):
        proc.kill()
    proc.wait()


def kill_live_agent_processes():
    """Kill every currently launched agent CLI (second-Ctrl-C path).

    The live-process table is copied under its lock, then each still
    running child is stopped with its whole process group, so a
    second Ctrl-C ends real work instead of waiting politely for it.
    """
    with daemon._LIVE_AGENT_PROCESSES_LOCK:
        live = list(daemon._LIVE_AGENT_PROCESSES.values())
    for proc in live:
        if proc.poll() is None:
            daemon.kill_agent_process(proc=proc)


def park_failed_outcome(dispatch_path):
    """Park one claimed message in failed/ and word the verified outcome.

    Arguments:
      dispatch_path = the claimed mailbox file to park.

    Returns:
      The refusal line's tail sentence: the parked wording when the
      inode-verified move succeeded, otherwise the unverified-move wording
      telling the reader to inspect the mailbox states by hand.
    """
    parked = daemon.park_failed_message(dispatch_path=dispatch_path)
    if parked:
        return "parked in failed/."
    return "failed-state move was not verified."


def park_failed_turn_outcome(dispatch_path):
    """Word the failed/ outcome for a message whose turn already ran.

    Arguments:
      dispatch_path = the claimed mailbox file to park.

    Returns:
      The sentence tail reporting the verified move, or the
      unverified-move wording.
    """
    parked = daemon.park_failed_message(dispatch_path=dispatch_path)
    if parked:
        return "message parked in failed/."
    return "failed-state move was not verified."


def park_prelaunch_outcome(dispatch_path):
    """Park one claimed message in prelaunch/ and word the verified outcome.

    Arguments:
      dispatch_path = the claimed mailbox file to retain.

    Returns:
      The sentence tail reporting the verified retention, or the
      unverified-move wording.
    """
    parked = daemon.park_prelaunch_message(dispatch_path=dispatch_path)
    if parked:
        return "retained in prelaunch/."
    return "failed-state move was not verified."


def reserve_dispatch_log_path(stamp, agent, relay_directory):
    """Reserve one dispatch relay-log path that no other run can claim.

    Two dispatches of the same role inside one clock second would choose the
    same second-precision filename, and the later ``open(..., "w")`` would
    truncate the earlier turn's archived output. Exclusive creation makes
    the reservation atomic: the operating system refuses a name that already
    exists, and the loop appends a two-digit suffix until a fresh name is
    accepted. The reserved file is created empty here; the caller fills it.

    Arguments:
      stamp = readable local-time text for the filename, one-second
              precision; equal stamps exercise the suffix loop.
      agent = mailbox role name embedded in the filename.
      relay_directory = directory that stores the relay logs, created when
                        missing.

    Returns:
      the reserved log path inside ``relay_directory``.
    """
    daemon.os.makedirs(relay_directory, exist_ok=True)
    suffix = 0
    while True:
        name = stamp + "-dispatch-" + agent
        if suffix > 0:
            name += "-" + str(suffix).zfill(2)
        log_path = daemon.os.path.join(relay_directory, name + ".log")
        create_exclusively = (daemon.os.O_CREAT | daemon.os.O_EXCL
                              | daemon.os.O_WRONLY)
        try:
            descriptor = daemon.os.open(log_path, create_exclusively)
        except FileExistsError:
            suffix += 1
            continue
        daemon.os.close(descriptor)
        return log_path


def dispatch(path, dry_run, fix_only=False, skip_redteam=False,
             new_reservation_cycle=None, architect_admission=None):
    """Serialize Architect GO decisions, then run one dispatch.

    Only a live Architect turn takes the main-checkout turn lock: its
    GO decision may land commits on the user's main, so two such
    turns must never overlap. A permanent-note administration turn
    additionally requires an idle ticket boundary and is deferred,
    not failed, when a ticket transition is underway. Every other
    turn goes straight to the locked dispatch body.

    Arguments:
      path                  = the pending mailbox message.
      dry_run               = preview without starting a role.
      fix_only              = True in a fix-only watch.
      skip_redteam          = True in a two-role watch.
      new_reservation_cycle = finite-watch reservation for a new
                              ticket, or ``None``.
      architect_admission   = admission token for a public Architect
                              request, or ``None``.

    Returns:
      True when the dispatch consumed the message; False when it was
      refused or deferred with the root message left in place.

    Raises:
      ValueError: when the path is not a pending agent message.
    """
    match = daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(path))
    if match is None:
        raise ValueError("not a pending agent message: " + path)
    agent = match.group(1)
    if dry_run or agent != "fable":
        return daemon.dispatch_under_main_checkout_lock(
            path=path, dry_run=dry_run, fix_only=fix_only,
            skip_redteam=skip_redteam,
            new_reservation_cycle=new_reservation_cycle,
            architect_admission=architect_admission)
    notes_admin_reserved = False
    try:
        message = daemon.read_cycle_message(path=path)
        notes_admin_reserved = daemon.is_architect_notes_admin_message(
            message=message)
    except (OSError, ValueError, daemon.TicketCycleStateError):
        # The normal dispatch validator owns the precise refusal and archive.
        notes_admin_reserved = False
    lock_file = daemon.acquire_main_checkout_turn_lock()
    if lock_file is None:
        print("refused " + daemon.os.path.basename(path) + ": the Architect "
              "GO-decision lock could not be proved; root message "
              "left untouched.")
        return False
    notes_lock = None
    try:
        if notes_admin_reserved:
            notes_lock = daemon.acquire_ticket_cycle_lock()
            daemon._require_no_ordinary_landing_transition_locked(
                current_dispatch_path=path)
        return daemon.dispatch_under_main_checkout_lock(
            path=path, dry_run=dry_run, fix_only=fix_only,
            skip_redteam=skip_redteam,
            new_reservation_cycle=new_reservation_cycle,
            architect_admission=architect_admission,
            notes_admin_reserved=notes_admin_reserved)
    except daemon.TicketCycleStateError as exc:
        print("deferred " + daemon.os.path.basename(path)
              + ": permanent-note admin turn requires an idle ticket "
              "boundary (" + str(exc) + "); root message remains queued.")
        return False
    finally:
        if notes_lock is not None:
            daemon.release_ticket_cycle_lock(lock_file=notes_lock)
        daemon.release_main_checkout_turn_lock(lock_file=lock_file)


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
    name = daemon.os.path.basename(path)
    agent_match = daemon.PENDING_MESSAGE_RE.match(name)
    if agent_match is None:
        raise ValueError("not a pending agent message: " + path)
    agent = agent_match.group(1)
    if agent == "daemon":
        raise ValueError(
            "local daemon receipts must use consume_daemon_message, not an "
            "AI dispatch route")
    if not daemon.message_is_enabled_for_topology(
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
            agent_topology_proof = daemon.validate_live_agent_dispatch_topology(
                agent=agent)
            persistent_role_state = daemon.capture_persistent_role_state(
                agent=agent)
            if (agent == "fable"
                    and isinstance(persistent_role_state, dict)):
                # Keep the turn's exact starting commit separately from the
                # richer persistent-state proof.  Focused embeddings may use
                # an opaque proof token, while the production proof remains
                # a dictionary; neither case may make the post-turn binding
                # depend on the shape of that token.
                architect_turn_base = persistent_role_state.get("base")
        except (OSError, daemon.PrimaryWorktreeError) as exc:
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
        severity_counts_before_claim = daemon.backlog_severity_counts()
        admission_count_before_claim = (
            severity_counts_before_claim["critical"]
            + severity_counts_before_claim["high"]
            + severity_counts_before_claim["medium"])
    dispatch_path = path
    currency = None
    prior_timeout = None
    if not dry_run:
        if not daemon.valid_duration(value=daemon.DISPATCH_TIMEOUT_MINUTES,
                              strictly_positive=True):
            print("refused " + name + ": dispatch timeout must be between "
                  "1 and " + str(daemon.MAX_DISPATCH_TIMEOUT_MINUTES)
                  + " minutes; message left queued.")
            return False
        try:
            history = daemon.timeout_events(name=name)
        except (OSError, ValueError, daemon.json.JSONDecodeError,
                OverflowError, RecursionError) as exc:
            print("refused " + name + ": cannot verify its timeout history: "
                  + str(exc) + "; message left queued.")
            return False
        dispatch_path = daemon.claim_message(path=path)
        if dispatch_path is None:
            if new_reservation_cycle is not None:
                daemon.release_unstarted_ticket_reservation(
                    cycle_id=new_reservation_cycle)
            return False
        # One recursive view, taken only after the atomic claim, owns both
        # currency numbers. Re-globbing each number would let a concurrent
        # sender make the banner internally inconsistent.
        currency = daemon.dispatch_currency(dispatch_path=dispatch_path, agent=agent)
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
        if daemon.park_failed_message(dispatch_path=dispatch_path):
            print("refused " + name + ": cannot read the body as UTF-8: "
                  + str(exc) + "; parked in failed/.")
        else:
            print("refused " + name + ": cannot read the body as UTF-8: "
                  + str(exc) + "; failed-state move was not verified; "
                  "inspect inflight/ and failed/.")
        return False

    notes_admin_body, notes_admin_problem = (
        daemon._architect_notes_admin_envelope(message=message))
    notes_admin_turn = notes_admin_problem is None
    if (message.startswith(daemon.MAILBOX_ADMIN_HEADER)
            and not notes_admin_turn):
        if dry_run:
            print("[dry-run] would refuse " + name + ": "
                  + notes_admin_problem)
            return False
        print("refused " + name + ": " + notes_admin_problem + "; "
              + daemon.park_failed_outcome(dispatch_path=dispatch_path))
        return False
    if not dry_run and notes_admin_turn != notes_admin_reserved:
        print("refused " + name + ": permanent-note admin identity changed "
              "across its exclusive reservation; "
              + daemon.park_failed_outcome(dispatch_path=dispatch_path))
        return False

    ticket_kind = None
    review_cycle_id = None
    review_accepted_commit = None
    review_receipt_before = None
    reopen_decision_cycle = None
    reopen_decision_commit = None
    reopen_before = None
    reopen_brief = ""
    effective_discovery_severity = daemon.DISCOVERY_SEVERITY
    effective_discovery_scope = daemon.DEFAULT_DISCOVERY_SCOPE
    saved_architect_severity = None
    saved_architect_scope = None
    flow_mode = None
    architect_checkpoint_audit = False
    architect_budget_audit = False
    implementer_budget_repair = False
    integration_revalidation = None
    if message.startswith(daemon.MAILBOX_FLOW_HEADER):
        _, flow_mode, flow_body, flow_problem = daemon._ticket_flow_envelope(
            message=message)
        checkpoint_request = (
            agent == "fable" and flow_problem is None
            and daemon.is_implementer_checkpoint_request(flow_body))
        checkpoint_problem = (daemon.checkpoint_handoff_problem(message=message)
                              if checkpoint_request else None)
        architect_checkpoint_audit = (
            checkpoint_request and checkpoint_problem is None)
        architect_budget_audit = (
            architect_checkpoint_audit
            and daemon.is_implementer_budget_checkpoint(flow_body))
        implementer_budget_repair = (
            agent == "opus" and daemon.is_architect_budget_repair(flow_body))
        if flow_problem is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": "
                      + flow_problem)
                return False
            print("refused " + name + ": " + flow_problem + "; "
                  + daemon.park_failed_outcome(dispatch_path=dispatch_path))
            return False
        if (agent == "fable"
                and flow_body.startswith(
                    "CONTROL-PLANE-INTEGRATION: REVALIDATE\n")):
            try:
                integration_revalidation = (
                    daemon.control_plane_integration_request(message=message))
            except daemon.TicketCycleStateError as exc:
                if dry_run:
                    print("[dry-run] would refuse " + name + ": " + str(exc))
                    return False
                print("refused " + name + ": " + str(exc) + "; "
                      + daemon.park_failed_outcome(dispatch_path=dispatch_path))
                return False
        if not daemon.ticket_cycle_mode_is_enabled(
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
            print("refused " + name + ": " + reason + "; "
                  + daemon.park_failed_outcome(dispatch_path=dispatch_path))
            return False
    if agent == "opus" and not message.startswith(daemon.MAILBOX_FLOW_HEADER):
        reason = ("Implementer work must carry one ticket-cycle flow "
                  "header; ask the Architect to reissue the handoff")
        if dry_run:
            print("[dry-run] would refuse " + name + ": " + reason)
            return False
        print("refused " + name + ": " + reason + "; "
              + daemon.park_failed_outcome(dispatch_path=dispatch_path))
        return False
    if agent == "fable" and message.startswith(daemon.MAILBOX_RETURN_HEADER):
        returned_cycle, returned_commit, returned_result, _, receipt_problem = (
            daemon._redteam_review_receipt(
            message=message)
        )
        if receipt_problem is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": "
                      + receipt_problem)
                return False
            print("refused " + name + ": " + receipt_problem + "; "
                  + daemon.park_failed_outcome(dispatch_path=dispatch_path))
            return False
        if returned_result == "REOPEN":
            reopen_decision_cycle = returned_cycle
            reopen_decision_commit = returned_commit
            try:
                reopen_before = daemon.current_reopen_ticket(
                    cycle_id=returned_cycle)
                reopen_brief = daemon._REOPEN_TRANSITION.architect_brief(
                    ticket=reopen_before, cycle=returned_cycle,
                    landing=returned_commit)
            except daemon.TicketCycleStateError as exc:
                print("refused " + name + ": reopening state could not be "
                      "proved (" + str(exc) + "); "
                      + daemon.park_prelaunch_outcome(dispatch_path=dispatch_path))
                return False
    if agent == "fable" and message.startswith(daemon.SOL_SEVERITY_HEADER):
        architect_request_problem = daemon.architect_user_request_problem(
            message=message)
        saved_architect_severity = daemon.architect_user_request_severity(
            message=message)
        saved_architect_scope = daemon.architect_user_request_scope(message=message)
        if architect_request_problem is not None:
            reason = ("invalid public Architect request header; "
                      + architect_request_problem)
            if dry_run:
                print("[dry-run] would refuse " + name + ": " + reason
                      + "; no file changed.")
                return False
            if daemon.park_failed_message(dispatch_path=dispatch_path):
                print("refused " + name + ": " + reason
                      + "; parked in failed/.")
            else:
                print("refused " + name + ": " + reason
                      + "; failed-state move was not verified; inspect "
                      "inflight/ and failed/.")
            return False
        effective_discovery_severity = saved_architect_severity
        effective_discovery_scope = saved_architect_scope
    maintenance_request = message == daemon.ARCHITECT_FIX_ONLY_REQUEST
    if (architect_admission is not None and (agent != "fable" or not (
            saved_architect_severity is not None or maintenance_request))):
        print("refused " + name + ": saved Architect admission does not "
              "name this exact public request; "
              + daemon.park_failed_outcome(dispatch_path=dispatch_path))
        return False
    if agent == "sol":
        ticket_kind = daemon.sol_ticket_kind(message=message)
        severity_problem = daemon.sol_discovery_severity_problem(message=message)
        saved_severity = daemon.sol_discovery_severity(message=message)
        saved_scope = daemon.sol_discovery_scope(message=message)
        if saved_severity is not None:
            effective_discovery_severity = saved_severity
        if saved_scope is not None:
            effective_discovery_scope = saved_scope
        reason = severity_problem
        if reason is None:
            reason = daemon.sol_ticket_refusal(
                ticket_kind=ticket_kind,
                admission_count=admission_count_before_claim,
                fix_only=fix_only,
                transport_valid=daemon.valid_sol_transport(message=message),
                discovery_severity=saved_severity,
                discovery_scope=saved_scope,
                unclassified_count=(
                    severity_counts_before_claim["unclassified"]),
                ledger_problem=severity_counts_before_claim["problem"])
        if reason is None and ticket_kind == "closure":
            reason = daemon.redteam_closure_problem(message=message)
        if reason is None and ticket_kind == "control-plane":
            _control_cycle, _control_candidate, _body, reason = (
                daemon._redteam_control_plane_envelope(message=message))
        if reason is not None:
            if dry_run:
                print("[dry-run] would refuse " + name + ": " + reason
                      + "; no file changed.")
                return False
            if daemon.park_failed_message(dispatch_path=dispatch_path):
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
    if agent == "opus" and daemon.ACTIVE_TOPOLOGY is not None:
        try:
            implementer_evidence_contract = (
                daemon.prepare_implementer_evidence_contract(message=message))
            implementer_return_before = daemon.fable_message_inode_snapshot()
        except daemon.FatalArchitectLandingError:
            raise
        except (OSError, daemon.TicketCycleStateError) as exc:
            reason = "Implementer evidence contract refused: " + str(exc)
            retry_after_note_fix = (
                len(daemon.ARCHITECT_DIRECTIVE_LINE_RE.findall(message)) == 1)
            parked = (daemon.park_prelaunch_message(dispatch_path=dispatch_path)
                      if retry_after_note_fix else
                      daemon.park_failed_message(dispatch_path=dispatch_path))
            state = ("retained in prelaunch/." if retry_after_note_fix
                     else "parked in failed/.")
            print("refused " + name + ": " + reason + "; "
                  + (state if parked else
                     "failed-state move was not verified."))
            return False

    registered_cycle_id = None
    if not dry_run:
        try:
            registered_cycle_id, _ = daemon.register_ticket_cycle_message(
                agent=agent, message=message,
                skip_redteam=skip_redteam,
                path_scope=(implementer_evidence_contract.get("allowed_paths")
                            if implementer_evidence_contract is not None
                            else None),
                ticket_class=(implementer_evidence_contract.get(
                    "ticket_class", "ordinary")
                    if implementer_evidence_contract is not None
                    else "ordinary"))
        except daemon.TicketCycleStateError as exc:
            reason = "ticket-cycle state refused this message: " + str(exc)
            print("refused " + name + ": " + reason + "; "
                  + daemon.park_failed_outcome(dispatch_path=dispatch_path))
            return False
    if agent == "sol" and ticket_kind == "closure":
        review_cycle_id = daemon.redteam_closure_ticket(message=message)
        review_accepted_commit = daemon.redteam_closure_commit(message=message)
        try:
            review_ticket = daemon.current_reopen_ticket(cycle_id=review_cycle_id)
            reopen_brief = daemon._REOPEN_TRANSITION.redteam_brief(
                ticket=review_ticket, cycle=review_cycle_id,
                landing=review_accepted_commit)
        except (daemon.TicketCycleStateError,
                daemon._REOPEN_TRANSITION.ReopenTransitionError) as exc:
            if dry_run:
                print("[dry-run] would refuse " + name + ": closure state "
                      "could not be proved (" + str(exc) + ")")
                return False
            print("refused " + name + ": closure state could not be proved ("
                  + str(exc) + "); "
                  + daemon.park_prelaunch_outcome(dispatch_path=dispatch_path))
            return False
        if not dry_run:
            review_receipt_before = daemon.fable_message_inode_snapshot()
    if agent == "sol" and ticket_kind == "control-plane":
        control_review_cycle, control_review_candidate, _body, _problem = (
            daemon._redteam_control_plane_envelope(message=message))
        if not dry_run:
            control_review_before = daemon.daemon_message_inode_snapshot()
    if agent == "sol":
        placeholder_body = daemon.sol_ticket_body(message=message)
    elif message.startswith(daemon.MAILBOX_FLOW_HEADER):
        _, _, placeholder_body, _ = daemon._ticket_flow_envelope(message=message)
    elif agent == "fable":
        placeholder_body = daemon.architect_user_request_body(message=message)
    else:
        placeholder_body = message
    marker = daemon.placeholder_in(message=placeholder_body)
    if marker is not None:
        if dry_run:
            print("[dry-run] would refuse " + name
                  + ": the whole body is template placeholder '" + marker
                  + "'; no file changed.")
            return False
        if daemon.park_failed_message(dispatch_path=dispatch_path):
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
        if daemon.park_failed_message(dispatch_path=dispatch_path):
            print("refused " + name + ": the body contains a NUL byte, "
                  "which cannot be a command argument; parked in failed/.")
        else:
            print("refused " + name + ": the body contains a NUL byte, "
                  "which cannot be a command argument; failed-state move "
                  "was not verified; inspect inflight/ and failed/.")
        return False

    if dry_run:
        print("[dry-run] would dispatch " + name + " -> "
              + " ".join(daemon.AGENT_COMMANDS[agent])
              + "  (cwd " + daemon.AGENT_CWD[agent] + ")")
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
            architect_go_before = daemon.daemon_message_inode_snapshot()
            if architect_checkpoint_audit or registered_cycle_id is not None:
                architect_opus_before = daemon.opus_message_inode_snapshot()
            elif notes_admin_turn:
                admin_opus_before = daemon.opus_message_inode_snapshot()
            elif architect_admission is not None:
                architect_opus_before = daemon.opus_message_inode_snapshot()
                architect_fable_before = daemon.fable_message_inode_snapshot()
                architect_sol_before = daemon.sol_message_inode_snapshot()
                architect_user_before = daemon.user_message_inode_snapshot()
        if agent == "opus" and registered_cycle_id is not None:
            replacement_context_path = daemon.latest_context_handoff_path(
                cycle_id=registered_cycle_id, mode=flow_mode)
            implementer_starting_head = daemon.prepare_implementer_cycle_checkout(
                cycle_id=registered_cycle_id,
                preserve_current=replacement_context_path is not None,
                restart_from_base=implementer_budget_repair)
            implementer_authority_before = daemon.implementer_authority_snapshot()
        elif (agent == "fable" and registered_cycle_id is not None):
            audit_commit = daemon.candidate_commit_for_cycle(
                cycle_id=registered_cycle_id)
            if audit_commit is not None:
                audit_cycle_id = registered_cycle_id
                candidate_scope = daemon.candidate_scope_for_cycle(
                    cycle_id=audit_cycle_id,
                    candidate_commit=audit_commit)
                audit_worktree = daemon.create_audit_snapshot(
                    cycle_id=audit_cycle_id, commit=audit_commit,
                    agent="fable")
        elif agent == "sol" and ticket_kind == "closure":
            audit_cycle_id = review_cycle_id
            audit_commit = review_accepted_commit
            audit_worktree = daemon.create_audit_snapshot(
                cycle_id=audit_cycle_id, commit=audit_commit, agent="sol")
        elif agent == "sol" and ticket_kind == "control-plane":
            audit_cycle_id = control_review_cycle
            audit_commit = control_review_candidate
            control = daemon.control_plane_ticket_state(
                cycle_id=audit_cycle_id, candidate_commit=audit_commit)
            if (control is None
                    or control["architect_candidate"] != audit_commit):
                raise daemon.TicketCycleStateError(
                    "control-plane review lacks D0-recorded Architect GO(C)")
            audit_worktree = daemon.create_audit_snapshot(
                cycle_id=audit_cycle_id, commit=audit_commit, agent="sol")
    except (OSError, daemon.PrimaryWorktreeError, daemon.TicketCycleStateError) as exc:
        print("refused " + name + ": exact cycle checkout failed ("
              + str(exc) + "); "
              + daemon.park_prelaunch_outcome(dispatch_path=dispatch_path))
        return False

    command_prefix = list(daemon.AGENT_COMMANDS[agent])
    command_prefix, routine_review = daemon.routine_review_command(
        command_prefix,
        agent=agent,
        ticket_kind=ticket_kind,
        candidate_audit=(audit_commit is not None),
        reopening=(reopen_decision_cycle is not None),
        checkpoint=architect_checkpoint_audit,
        integration=(integration_revalidation is not None))
    banner = daemon.dispatch_banner(
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
        banner += daemon.replacement_context_notice(path=replacement_context_path)
    # The dynamic banner precedes the byte-unchanged PREAMBLE. The
    # role-specific banner sits between them. Consequently PREAMBLE's
    # --- MESSAGE --- delimiter remains immediately before the exact raw
    # mailbox body, and the body remains the prompt's exact suffix. The
    # Architect route receives decision authority. The parent daemon owns the
    # exact local landing after that process exits.
    stamp = daemon.datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = daemon.reserve_dispatch_log_path(
        stamp=stamp, agent=agent, relay_directory=daemon.RELAY_DIR)
    checkpoint_state_path = None
    if agent == "opus":
        checkpoint_state_path = log_path + "." + name + ".checkpoint"
        settings = daemon.implementer_checkpoint_settings(
            python=daemon.sys.executable,
            hook_path=daemon.os.path.join(
                daemon.AGENT_CWD["fable"], "ai", "tools",
                "implementer_checkpoint_hook.py"))
        command_prefix += [
            "--settings", daemon.json.dumps(settings, separators=(",", ":"))]
    common_preamble = daemon.common_preamble_for_dispatch(
        checkpoint_audit=architect_checkpoint_audit)
    command = command_prefix + ["--",
        banner + reopen_brief + daemon.agent_preamble(agent=agent, message=message)
        + daemon.architect_admission_prompt(token=architect_admission)
        + common_preamble + message]

    if notes_admin_turn:
        try:
            daemon.write_architect_notes_admin_journal(
                request_name=name, request_message=message,
                base_commit=architect_turn_base, phase="started")
        except (OSError, daemon.TicketCycleStateError) as exc:
            print("  !! permanent-note admin journal could not be started: "
                  + str(exc) + "; "
                  + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
            return False

    if routine_review is not None:
        print("routine review: " + routine_review + " at "
              + daemon.REVIEW_EFFORT + " effort.")
    print("dispatching " + name + " -> " + agent + " ...")
    # Stream the agent's output straight into the relay log AS IT RUNS
    # (stderr folded in -- the codex CLI narrates its progress there), and
    # heartbeat once a minute so a long turn is distinguishable from a hang:
    # elapsed time always moves, and the log size moves whenever the agent
    # emits anything. A buffered subprocess.run() here once left the
    # terminal silent for an entire multi-minute turn.
    started = daemon.time.time()
    proc = None
    child_started = False
    launch_error = None
    timed_out = False
    timeout_history_error = None
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(command_prefix) + " <message>\n")
        f.write("--- live output (stdout+stderr interleaved) ---\n")
        f.flush()
        # Claude Code reads the active role's compaction point from this
        # environment variable. The Architect and Implementer have separate
        # limits. Sol receives its independent value in the Codex command.
        env = daemon.os.environ.copy()
        if agent in {"fable", "opus"}:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(
                daemon.claude_compaction_limit(agent=agent))
        else:
            env.pop("CLAUDE_CODE_AUTO_COMPACT_WINDOW", None)
        env[daemon.MAX_CHARACTERS_ENVIRONMENT] = str(daemon.MAX_CHARACTERS)
        env[daemon.DISCOVERY_SEVERITY_ENVIRONMENT] = effective_discovery_severity
        env[daemon.DISCOVERY_SCOPE_ENVIRONMENT] = effective_discovery_scope
        env[daemon.MAILBOX_ROLE_ENVIRONMENT] = daemon.mailbox_role_for_dispatch(
            agent=agent, message=message)
        if agent == "opus":
            if daemon.os.path.lexists(checkpoint_state_path):
                launch_error = OSError(
                    "Implementer checkpoint marker already exists")
            env[daemon.IMPLEMENTER_CHECKPOINT_DEADLINE_ENVIRONMENT] = repr(
                daemon.time.monotonic() + daemon.IMPLEMENTER_REVIEW_MINUTES * 60.0)
            env[daemon.IMPLEMENTER_CHECKPOINT_STATE_ENVIRONMENT] = (
                checkpoint_state_path)
        else:
            env.pop(daemon.IMPLEMENTER_CHECKPOINT_DEADLINE_ENVIRONMENT, None)
            env.pop(daemon.IMPLEMENTER_CHECKPOINT_STATE_ENVIRONMENT, None)
        if architect_admission is not None:
            env["MAILBOX_ARCHITECT_ADMISSION"] = architect_admission
        else:
            env.pop("MAILBOX_ARCHITECT_ADMISSION", None)
        if notes_admin_turn:
            env["MAILBOX_NOTES_BASE"] = architect_turn_base
        else:
            env.pop("MAILBOX_NOTES_BASE", None)
        env["MAILBOX_PRIMARY_WORKTREE"] = daemon.AGENT_CWD["fable"]
        env["MAILBOX_IMPLEMENTER_WORKTREE"] = daemon.AGENT_CWD["opus"]
        env["MAILBOX_EXECUTION_WORKTREE"] = daemon.AGENT_CWD["opus"]
        env["MAILBOX_SHARED_NOTES"] = daemon.os.path.join(
            daemon.AGENT_CWD["fable"], "ai", "notes")
        env["MAILBOX_HANDOFF_CONTRACT"] = daemon.os.path.join(
            daemon.AGENT_CWD["fable"], "ai", "tools", "handoff_contract.py")
        env["MAILBOX_TICKET_CHANGE_GUARD"] = daemon.os.path.join(
            daemon.AGENT_CWD["fable"], "ai", "tools", "ticket_change_guard.py")
        if audit_worktree is not None:
            env["MAILBOX_CANDIDATE_COMMIT"] = audit_commit
            env["MAILBOX_AUDIT_WORKTREE"] = audit_worktree
        else:
            env.pop("MAILBOX_CANDIDATE_COMMIT", None)
            env.pop("MAILBOX_AUDIT_WORKTREE", None)
        if fix_only:
            env[daemon.FIX_ONLY_ENVIRONMENT] = "1"
        else:
            env.pop(daemon.FIX_ONLY_ENVIRONMENT, None)
        if skip_redteam:
            env[daemon.SKIP_REDTEAM_ENVIRONMENT] = "1"
        else:
            env.pop(daemon.SKIP_REDTEAM_ENVIRONMENT, None)
        try:
            if launch_error is not None:
                raise launch_error
            if agent in {"fable", "opus", "sol"}:
                daemon.revalidate_agent_dispatch_topology(
                    proof=agent_topology_proof)
            daemon.recheck_persistent_role_state(proof=persistent_role_state)
            # The child leads its own session: the terminal's Ctrl-C then
            # reaches only this daemon, which finishes or kills the turn
            # deliberately, and a timeout kill can stop the whole group.
            proc = daemon.subprocess.Popen(command,
                                    stdout=f,
                                    stderr=daemon.subprocess.STDOUT,
                                    cwd=daemon.AGENT_CWD[agent],
                                    env=env,
                                    start_new_session=True)
            child_started = True
            try:
                if agent in {"fable", "opus", "sol"}:
                    if notes_admin_turn:
                        daemon.revalidate_protected_policy_admin_topology(
                            proof=agent_topology_proof)
                    else:
                        daemon.revalidate_agent_dispatch_topology(
                            proof=agent_topology_proof)
                if not notes_admin_turn:
                    daemon.recheck_persistent_role_state(
                        proof=persistent_role_state)
            except (OSError, daemon.PrimaryWorktreeError):
                daemon.kill_agent_process(proc=proc)
                raise
        except (OSError, ValueError) as exc:
            launch_error = exc
            f.write("\n--- dispatch could not start: " + str(exc) + " ---\n")
        except daemon.PrimaryWorktreeError as exc:
            launch_error = exc
            proc = None
            f.write("\n--- dispatch topology changed before launch: "
                    + str(exc) + " ---\n")
        if proc is not None:
            with daemon._LIVE_AGENT_PROCESSES_LOCK:
                daemon._LIVE_AGENT_PROCESSES[id(proc)] = proc
            daemon._rendezvous_turn_started()
            try:
                next_beat = started + 60.0
                deadline = started + daemon.DISPATCH_TIMEOUT_MINUTES * 60.0
                while proc.poll() is None:
                    daemon.time.sleep(5)
                    now = daemon.time.time()
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
                        daemon.kill_agent_process(proc=proc)
                        timed_out = True
                        # The timeout setting is the stable killed-after
                        # threshold promised to a later retry. The poll loop
                        # can observe the child a fraction late; retain that
                        # elapsed value for diagnostics without letting
                        # scheduler jitter leak into the human-facing retry
                        # sentence.
                        killed_after_minutes = daemon.DISPATCH_TIMEOUT_MINUTES
                        observed_elapsed_minutes = (now - started) / 60.0
                        try:
                            daemon.write_timeout_history(
                                name=name,
                                killed_after_minutes=killed_after_minutes,
                                observed_elapsed_minutes=(
                                    observed_elapsed_minutes))
                        except (OSError, ValueError, daemon.json.JSONDecodeError,
                                OverflowError, RecursionError) as exc:
                            timeout_history_error = exc
                        print("  timed out " + name + " after "
                              + daemon.exact_duration(value=killed_after_minutes)
                              + " min; the turn was killed; its recovery "
                              "state will be verified after the log closes.")
                        break
                    if now >= next_beat:
                        elapsed_min = (now - started) / 60.0
                        try:
                            log_kb = daemon.os.fstat(f.fileno()).st_size / 1024.0
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
                        daemon.kill_agent_process(proc=proc)
                finally:
                    with daemon._LIVE_AGENT_PROCESSES_LOCK:
                        daemon._LIVE_AGENT_PROCESSES.pop(id(proc), None)
                    if proc.poll() is not None:
                        daemon._rendezvous_turn_finished()
            f.write("\n--- rc=" + str(proc.returncode) + " ---\n")

    authority_changes = []
    if proc is not None and agent == "opus" \
            and implementer_authority_before is not None:
        try:
            authority_changes = daemon.implementer_authority_changes(
                before=implementer_authority_before)
        except (OSError, daemon.PrimaryWorktreeError,
                daemon.TicketCycleStateError) as exc:
            authority_changes = ["snapshot could not be verified: " + str(exc)]
    if authority_changes:
        for return_path in daemon.glob.glob(
                daemon.os.path.join(daemon.MAILBOX, "*-to-fable.md")):
            inode = daemon.regular_inode(path=return_path)
            if (implementer_return_before is not None
                    and inode is not None
                    and inode not in implementer_return_before):
                daemon.park_failed_message(dispatch_path=return_path)
        parked = daemon.park_failed_message(dispatch_path=dispatch_path)
        print("IMPLEMENTER AUTHORITY VIOLATION:")
        for changed in authority_changes:
            print("- " + changed + " changed during the Implementer turn.")
        print("Candidate or partial work preserved in " + daemon.AGENT_CWD["opus"]
              + "; nothing landed. "
              + ("Request parked in failed/." if parked else
                 "Request state needs manual inspection."))
        raise daemon.ImplementerAuthorityViolationError(authority_changes)

    persistent_role_error = None
    if proc is not None and agent in {"fable", "opus", "sol"}:
        try:
            if notes_admin_turn:
                daemon.revalidate_protected_policy_admin_topology(
                    proof=agent_topology_proof)
            else:
                daemon.revalidate_agent_dispatch_topology(
                    proof=agent_topology_proof)
            if not notes_admin_turn:
                daemon.recheck_persistent_role_state(
                    proof=persistent_role_state)
        except (OSError, daemon.PrimaryWorktreeError) as exc:
            persistent_role_error = exc

    if audit_worktree is not None:
        try:
            daemon.remove_audit_snapshot(
                cycle_id=audit_cycle_id, commit=audit_commit, agent=agent)
        except (OSError, daemon.PrimaryWorktreeError,
                daemon.TicketCycleStateError) as exc:
            if launch_error is None:
                launch_error = daemon.PrimaryWorktreeError(
                    "audit snapshot cleanup failed: " + str(exc))

    if launch_error is not None:
        if child_started:
            parked = daemon.park_failed_message(dispatch_path=dispatch_path)
            state = "message parked in failed/" if parked \
                else "failed-state move was not verified"
        else:
            parked = daemon.park_prelaunch_message(dispatch_path=dispatch_path)
            state = "message retained in prelaunch/" if parked \
                else "pre-launch state move was not verified"
        print("  !! dispatch could not start: " + str(launch_error)
              + "; " + state + "; log -> " + log_path)
        return False

    if persistent_role_error is not None:
        parked = daemon.park_failed_message(dispatch_path=dispatch_path)
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
        if daemon.park_failed_message(dispatch_path=dispatch_path):
            print("  timeout recovery verified: message parked in failed/; "
                  "requeue it by moving it back to the mailbox (or relaunch "
                  "with a larger --dispatch-timeout).")
        else:
            print("  !! timeout recovery failed: the failed/ state was not "
                  "verified; inspect inflight/ before requeueing.")
        return False

    if proc.returncode != 0:
        if daemon.provider_is_out_of_tokens(agent=agent, reply_lines=reply_lines):
            parked = daemon.park_failed_message(dispatch_path=dispatch_path)
            daemon._TOKEN_EXHAUSTION_STOP.set()
            raise daemon.RoleTokenExhaustionError(
                agent=agent,
                request_path=(daemon.os.path.join(daemon.MAILBOX, "failed", name)
                              if parked else None))
        # a failed dispatch is NOT done: park it in failed/ so it is never
        # silently consumed, and never hot-retried while the cause persists.
        # Requeue after fixing the cause:
        #   mv ai/notes/mailbox/failed/<f> ai/notes/mailbox/
        parked = daemon.park_failed_message(dispatch_path=dispatch_path)
        # the turn's output lives in the log file (it streams there;
        # proc.stdout is None under Popen with a file handle).
        if not parked:
            print("  !! dispatch failed and its failed/ state was not "
                  "verified; inspect inflight/ and failed/; log -> "
                  + log_path)
        else:
            print("  !! " + daemon.provider_failure_guidance(
                agent=agent, reply_lines=reply_lines))
            print("  !! message parked in failed/; see the log above.")
        return False

    implementer_delivery_receipt = None
    architect_delivery_receipt = None
    if agent == "opus" and registered_cycle_id is not None:
        implementer_completion_ready = True
        implementer_context_handoff = False
        if implementer_evidence_contract is not None:
            try:
                returned_candidate = daemon.worktree_head(
                    worktree=daemon.AGENT_CWD["opus"])
                implementer_return, invalid_returns, evidence_problem = (
                    daemon.matching_new_context_handoff(
                        cycle_id=registered_cycle_id, mode=flow_mode,
                        before_inodes=implementer_return_before))
                implementer_context_handoff = implementer_return is not None
                if (evidence_problem is None
                        and not implementer_context_handoff):
                    implementer_return, invalid_returns, evidence_problem, \
                        implementer_completion_ready = (
                        daemon.matching_new_implementer_handoff(
                            cycle_id=registered_cycle_id, mode=flow_mode,
                            candidate_commit=returned_candidate,
                            before_inodes=implementer_return_before,
                            evidence_contract=implementer_evidence_contract))
                elif implementer_context_handoff:
                    implementer_completion_ready = False
            except (OSError, daemon.PrimaryWorktreeError,
                    daemon.TicketCycleStateError) as exc:
                implementer_return = None
                invalid_returns = []
                evidence_problem = str(exc)
                implementer_completion_ready = None
            if (evidence_problem is None
                    and daemon.implementer_checkpoint_delivered(
                        checkpoint_state_path)):
                try:
                    returned_message = daemon.read_cycle_message(
                        path=implementer_return)
                except (OSError, ValueError,
                        daemon.TicketCycleStateError) as exc:
                    evidence_problem = str(exc)
                else:
                    evidence_problem = daemon.checkpoint_handoff_problem(
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
                         or daemon._clean_worktree_status(
                             worktree=daemon.AGENT_CWD["opus"]))):
                invalid_returns = [implementer_return]
                evidence_problem = (
                    "blocked subagent evidence is valid only when Opus HEAD "
                    "still equals its cycle starting commit and the "
                    "Implementer worktree has no tracked or untracked edit")
                implementer_completion_ready = None
            if evidence_problem is not None:
                for return_path in invalid_returns:
                    daemon.park_failed_message(dispatch_path=return_path)
                print("  !! Implementer returned rc=0 but its same-cycle "
                      "subagent evidence was refused before candidate "
                      "freeze: " + evidence_problem + "; "
                      + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
                return False
        if implementer_completion_ready:
            try:
                if implementer_evidence_contract is not None:
                    implementer_delivery_receipt = (
                        daemon.write_implementer_delivery_receipt(
                            request_path=dispatch_path,
                            return_path=implementer_return))
                candidate = daemon.record_implementer_candidate(
                    cycle_id=registered_cycle_id,
                    starting_head=implementer_starting_head,
                    replace_prior=implementer_budget_repair)
            except (OSError, ValueError, daemon.PrimaryWorktreeError,
                    daemon.TicketCycleStateError) as exc:
                if implementer_delivery_receipt is not None:
                    try:
                        preserved = (daemon.candidate_commit_for_cycle(
                            cycle_id=registered_cycle_id)
                            == returned_candidate)
                    except (OSError, daemon.TicketCycleStateError):
                        preserved = True
                    if not preserved:
                        daemon.os.remove(implementer_delivery_receipt)
                        daemon.fsync_directory(directory=daemon.MAILBOX)
                        implementer_delivery_receipt = None
                parked = (False if implementer_delivery_receipt is not None
                          else daemon.park_failed_message(
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
            daemon.matching_new_control_plane_receipt(
                cycle_id=control_review_cycle,
                candidate=control_review_candidate,
                before_inodes=control_review_before))
        if receipt_problem is not None:
            print("  !! Red Team process returned rc=0 but its exact "
                  "control-plane decision was not proved: "
                  + receipt_problem + "; "
                  + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
            return False
        try:
            # Persist the second key here, where D0 has just proved that the
            # exact receipt was newly produced by this successful Sol turn.
            # A structured file that merely appears in the mailbox has no
            # authority to create this decision.
            daemon.record_control_plane_redteam_decision(
                cycle_id=control_review_cycle,
                candidate_commit=control_review_candidate,
                decision=control_result)
        except daemon.TicketCycleStateError as exc:
            print("  !! Red Team decision could not be saved: " + str(exc)
                  + "; " + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
            return False
        if not daemon.archive_consumed_message(dispatch_path=dispatch_path):
            return False
        print("authenticated mandatory Red Team decision " + control_result
              + " for exact protected C " + control_review_candidate
              + "; D0 will consume " + daemon.os.path.basename(receipt_path)
              + ".")
        return True

    if review_cycle_id is not None:
        receipt_path, review_result, receipt_problem = (
            daemon.matching_new_redteam_receipt(
                cycle_id=review_cycle_id,
                accepted_commit=review_accepted_commit,
                before_inodes=review_receipt_before))
        if receipt_problem is not None:
            print("  !! Red Team process returned rc=0 but its correlated "
                  "receipt was not proved: " + receipt_problem + "; "
                  + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
            return False
        if not daemon.archive_consumed_message(dispatch_path=dispatch_path):
            return False
        if not daemon.redteam_review_completes_cycle(review_result):
            print("Red Team returned REOPEN for " + review_cycle_id
                  + " at " + review_accepted_commit
                  + "; the same cycle remains active until the Architect "
                    "records GO or NO-GO.")
            return True
        try:
            completed_now = daemon.complete_ticket_cycle(
                cycle_id=review_cycle_id,
                accepted_commit=review_accepted_commit)
        except daemon.TicketCycleStateError as exc:
            print("  !! Red Team request was archived and receipt "
                  + daemon.os.path.basename(receipt_path)
                  + " exists, but cycle state was not completed: "
                  + str(exc))
            return False
        daemon.deliver_pending_ticket_cycle_returns()
        print("ticket cycle complete: Red Team returned " + review_result
              + " for " + review_cycle_id + " at "
              + review_accepted_commit + ".")
        return True

    if reopen_decision_cycle is not None:
        try:
            decision = daemon.architect_reopen_decision(
                cycle_id=reopen_decision_cycle, before=reopen_before)
            decision_landing, completed_now = daemon.land_architect_reopen_decision(
                dispatch_path=dispatch_path,
                cycle_id=reopen_decision_cycle,
                reviewed_landing=reopen_decision_commit,
                decision=decision)
        except (OSError, daemon.PrimaryWorktreeError,
                daemon.TicketCycleStateError) as exc:
            requeued = daemon.requeue_retryable_daemon_message(
                dispatch_path=dispatch_path)
            print("  !! Architect REOPEN decision was not accepted: "
                  + str(exc) + "; "
                  + ("the exact request was requeued."
                     if requeued else
                     "the inflight request remains preserved."))
            return False
        daemon.deliver_pending_ticket_cycle_returns()
        print("ticket cycle complete: Architect returned " + decision
              + " to Red Team REOPEN for " + reopen_decision_cycle
              + " at " + reopen_decision_commit + "; backlog decision "
                "landed as " + decision_landing + ".")
        # GO and NO-GO both land a decision commit, so the request is
        # consumed either way.
        return True

    if (agent == "fable" and audit_cycle_id is not None
            and audit_commit is not None
            and architect_turn_base is not None):
        if daemon.worktree_head(
                worktree=daemon.AGENT_CWD["fable"]) != architect_turn_base:
            print("  !! Architect candidate audit changed the persistent "
                  "primary HEAD; note commits require a separate no-ticket "
                  "turn; " + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
            return False
        try:
            daemon._validate_current_protected_primary_state(
                primary_worktree=daemon.AGENT_CWD["fable"])
        except daemon.PrimaryWorktreeError as exc:
            print("  !! Architect candidate audit changed a protected "
                  "permanent note, its guard, or the sealed backlog: "
                  + str(exc) + "; "
                  + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
            return False
        go_path, invalid_go_paths, go_problem = daemon.matching_new_architect_go(
            cycle_id=audit_cycle_id, candidate_commit=audit_commit,
            mode=flow_mode, before_inodes=architect_go_before)
        handoff_path = None
        if architect_checkpoint_audit:
            handoff_path, invalid_handoffs, handoff_problem = (
                daemon.matching_new_checkpoint_handoff(
                    cycle_id=audit_cycle_id, mode=flow_mode,
                    before_inodes=architect_opus_before,
                    budget=architect_budget_audit))
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
                daemon.matching_new_architect_handoff(
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
                daemon.park_failed_message(dispatch_path=invalid_path)
            print("  !! Architect returned rc=0 but its daemon GO boundary "
                  "was refused: " + go_problem + "; "
                  + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
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
                daemon.write_implementer_delivery_receipt(
                    request_path=dispatch_path,
                    return_path=go_path or handoff_path))
            if handoff_path is not None:
                daemon.record_architect_repair_scope(
                    cycle_id=audit_cycle_id,
                    handoff_message=daemon.read_cycle_message(path=handoff_path))
            if (go_path is not None
                    and daemon.control_plane_ticket_state(
                        cycle_id=audit_cycle_id,
                        candidate_commit=audit_commit) is not None):
                # The delivery hard link is deliberately short-lived. Save
                # the protected Architect key while D0 can still prove that
                # this exact Architect turn created this exact GO(C).
                daemon.record_control_plane_architect_go(
                    cycle_id=audit_cycle_id,
                    candidate_commit=audit_commit)
                if integration_revalidation is not None:
                    if (integration_revalidation["cycle_id"] != audit_cycle_id
                            or integration_revalidation["candidate"]
                            != audit_commit):
                        raise daemon.TicketCycleStateError(
                            "integration audit changed its exact cycle or C")
                    daemon.record_control_plane_integration_go(
                        cycle_id=audit_cycle_id,
                        candidate_commit=audit_commit,
                        new_main=integration_revalidation["new_main"],
                        evidence=daemon.os.path.basename(
                            architect_delivery_receipt))
        except (OSError, ValueError, daemon.TicketCycleStateError) as exc:
            print("  !! validated Architect outcome could not be journaled: "
                  + str(exc) + "; request kept in inflight/.")
            return False

    if (agent == "fable" and audit_cycle_id is None
            and architect_turn_base is not None):
        base_commit = architect_turn_base
        notes_commit = daemon.worktree_head(worktree=daemon.AGENT_CWD["fable"])
        fresh_daemon = daemon.new_route_paths(
            pattern="*-to-daemon.md",
            before_inodes=architect_go_before)
        fresh_opus = []
        opus_before = (admin_opus_before if notes_admin_turn
                       else architect_opus_before)
        if opus_before is not None:
            fresh_opus = daemon.new_route_paths(
                pattern="*-to-opus.md",
                before_inodes=opus_before)
        fresh_fable = []
        fresh_sol = []
        fresh_user = []
        if architect_fable_before is not None:
            fresh_fable = daemon.new_route_paths(
                pattern="*-to-fable.md",
                before_inodes=architect_fable_before)
        if architect_sol_before is not None:
            fresh_sol = daemon.new_route_paths(
                pattern="*-to-sol.md",
                before_inodes=architect_sol_before)
        if architect_user_before is not None:
            fresh_user = daemon.new_route_paths(
                pattern="*-to-user.md",
                before_inodes=architect_user_before)
        if notes_admin_turn:
            if fresh_opus:
                for invalid_path in fresh_opus:
                    daemon.park_failed_message(dispatch_path=invalid_path)
                print("  !! permanent-note admin turn created an "
                      "Implementer handoff; note administration is "
                         "cycle-free; "
                      + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
                return False
        if notes_commit == base_commit:
            try:
                if (notes_admin_turn
                        and daemon._clean_worktree_status(
                            worktree=daemon.AGENT_CWD["fable"])):
                    raise daemon.PrimaryWorktreeError(
                        "Architect admin turn left uncommitted changes")
                daemon._validate_current_protected_primary_state(
                    primary_worktree=daemon.AGENT_CWD["fable"])
            except daemon.PrimaryWorktreeError as exc:
                print("  !! Architect left a protected permanent note, its "
                      "guard, or the sealed backlog different from commit "
                      "B: " + str(exc) + "; "
                      + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
                return False
            if architect_admission is not None:
                fresh_fable = [
                    path for path in fresh_fable
                    if daemon.message_claims_architect_admission(
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
                        + daemon.os.path.basename(outcome_path).split("-to-", 1)[-1])
                else:
                    outcome_path = fresh_outputs[0]
                    if daemon.os.path.dirname(outcome_path) != daemon.MAILBOX:
                        outcome_problem = (
                            "public Architect outcome was not published in "
                            "the mailbox root")
                    else:
                        try:
                            outcome_message = daemon.read_cycle_message(
                                path=outcome_path)
                        except (OSError, ValueError,
                                daemon.TicketCycleStateError) as exc:
                            outcome_problem = str(exc)
                        else:
                            if fresh_opus:
                                if not outcome_message.startswith(
                                        daemon.MAILBOX_FLOW_HEADER):
                                    outcome_problem = (
                                        "Implementer outcome lacks its exact "
                                        "ticket flow envelope")
                                else:
                                    try:
                                        converted_cycle, _ = (
                                            daemon.register_ticket_cycle_message(
                                                agent="opus",
                                                message=outcome_message,
                                                skip_redteam=skip_redteam,
                                                architect_admission=(
                                                    architect_admission),
                                                implementer_request_name=(
                                                    daemon.os.path.basename(
                                                        outcome_path))))
                                    except daemon.TicketCycleStateError as exc:
                                        outcome_problem = str(exc)
                                    else:
                                        daemon._NO_ELIGIBLE_MAINTENANCE_WORK.clear()
                                        outcome_kind = (
                                            "Implementer ticket "
                                            + str(converted_cycle))
                            elif fresh_sol:
                                outcome_problem = (
                                    daemon.public_architect_sol_outcome_problem(
                                        message=outcome_message,
                                        expected_token=(
                                            architect_admission)))
                                if outcome_problem is None:
                                    try:
                                        daemon.release_architect_ticket_admission(
                                            token=architect_admission)
                                    except daemon.TicketCycleStateError as exc:
                                        outcome_problem = str(exc)
                                    else:
                                        outcome_kind = "Sol advisory request"
                            else:
                                outcome_problem = (
                                    daemon.public_architect_no_ticket_problem(
                                        message=outcome_message,
                                        expected_token=(
                                            architect_admission)))
                                if (outcome_problem is None
                                        and maintenance_request):
                                    try:
                                        eligible = (
                                            daemon.eligible_fix_only_bug_anchors())
                                    except daemon.TicketCycleStateError as exc:
                                        outcome_problem = str(exc)
                                    else:
                                        if eligible:
                                            outcome_problem = (
                                                "maintenance no-ticket refused: "
                                                "eligible Open BUG FIX remains "
                                                + eligible[0])
                                if outcome_problem is None:
                                    try:
                                        daemon.release_architect_ticket_admission(
                                            token=architect_admission)
                                    except daemon.TicketCycleStateError as exc:
                                        outcome_problem = str(exc)
                                    else:
                                        if maintenance_request:
                                            daemon._NO_ELIGIBLE_MAINTENANCE_WORK.set()
                                        outcome_kind = "no-ticket receipt"
                if outcome_problem is not None:
                    for invalid_path in fresh_outputs:
                        daemon.park_failed_message(dispatch_path=invalid_path)
                    retry_maintenance = (
                        maintenance_request
                        and outcome_problem.startswith(
                            "maintenance no-ticket refused:"))
                    if retry_maintenance:
                        parked = daemon.requeue_retryable_daemon_message(
                            dispatch_path=dispatch_path)
                    else:
                        parked = daemon.park_failed_message(
                            dispatch_path=dispatch_path)
                    print("  !! Architect returned rc=0 but its public "
                          "request outcome was refused: "
                          + outcome_problem + "; the provisional admission "
                          "was retained; "
                          + (("request requeued for the same admitted slot."
                              if retry_maintenance else
                              "message parked in failed/.") if parked else
                             "recovery-state move was not verified."))
                    return False
                print("  authenticated public Architect outcome: "
                      + outcome_kind + "; exact output "
                      + daemon.os.path.basename(outcome_path)
                      + " remains queued for its recipient.")
            if maintenance_request and fresh_opus:
                daemon.send(agent="fable", text=daemon.ARCHITECT_FIX_ONLY_REQUEST,
                     dry_run=False)
            if fresh_daemon and architect_admission is None:
                for invalid_path in fresh_daemon:
                    daemon.park_failed_message(dispatch_path=invalid_path)
                print("  !! Architect created a daemon request without one "
                      "new permanent-note commit; "
                      + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
                return False
            if notes_admin_turn:
                try:
                    daemon.write_architect_notes_admin_journal(
                        request_name=name, request_message=message,
                        base_commit=base_commit, phase="validated-noop")
                except (OSError, daemon.TicketCycleStateError) as exc:
                    parked = daemon.park_failed_message(
                        dispatch_path=dispatch_path)
                    print("  !! validated no-op admin result could not be "
                          "journaled: " + str(exc) + "; "
                          + ("message parked in failed/." if parked else
                             "failed-state move was not verified."))
                    return False
        else:
            if not notes_admin_turn:
                for invalid_path in fresh_daemon:
                    daemon.park_failed_message(dispatch_path=invalid_path)
                print("  !! Architect changed permanent notes outside the "
                      "dedicated MAILBOX-ADMIN: permanent-notes route; "
                      + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
                return False
            go_path, invalid_paths, note_problem = (
                daemon.matching_new_architect_notes_go(
                    base_commit=base_commit, notes_commit=notes_commit,
                    before_inodes=architect_go_before))
            if note_problem is None:
                try:
                    daemon.require_architect_notes_commit(
                        base_commit=base_commit, notes_commit=notes_commit)
                    if notes_admin_reserved:
                        daemon._require_no_ordinary_landing_transition_locked(
                            current_dispatch_path=go_path)
                    else:
                        daemon.require_no_ordinary_landing_transition(
                            current_dispatch_path=go_path)
                except (OSError, daemon.TicketCycleStateError) as exc:
                    note_problem = str(exc)
                    invalid_paths = [go_path]
            if note_problem is not None:
                for invalid_path in invalid_paths:
                    if invalid_path is not None:
                        daemon.park_failed_message(dispatch_path=invalid_path)
                print("  !! Architect permanent-note commit was refused: "
                      + note_problem + "; "
                      + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
                return False
            try:
                receipt_raw = daemon.stable_regular_bytes(
                    path=go_path,
                    maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="permanent-note GO receipt")
                daemon.write_architect_notes_admin_journal(
                    request_name=name, request_message=message,
                    base_commit=base_commit, phase="validated-commit",
                    notes_commit=notes_commit,
                    receipt_sha256=daemon.hashlib.sha256(receipt_raw).hexdigest())
            except (OSError, ValueError, daemon.TicketCycleStateError) as exc:
                print("  !! validated permanent-note commit could not be "
                      "journaled: " + str(exc) + "; "
                      + daemon.park_failed_turn_outcome(dispatch_path=dispatch_path))
                return False
            print("  authenticated permanent-note commit " + notes_commit
                  + " on exact main baseline " + base_commit
                  + "; parent daemon will fast-forward it after this turn.")

    archived = daemon.archive_consumed_message(dispatch_path=dispatch_path)
    if archived and implementer_delivery_receipt is not None:
        daemon.os.remove(implementer_delivery_receipt)
        daemon.fsync_directory(directory=daemon.MAILBOX)
    if archived and architect_delivery_receipt is not None:
        daemon.os.remove(architect_delivery_receipt)
        daemon.fsync_directory(directory=daemon.MAILBOX)
    if archived and notes_admin_turn:
        try:
            journal = daemon.read_architect_notes_admin_journal(
                request_name=name, request_message=message)
            if journal["phase"] == "validated-noop":
                daemon.remove_architect_notes_admin_journal(request_name=name)
            elif journal["phase"] == "validated-commit":
                print("  retained validated permanent-note admin journal "
                      "until its exact P receipt is consumed.")
            else:
                raise daemon.TicketCycleStateError(
                    "archived permanent-note admin still has only its "
                    "pre-child journal")
        except (OSError, daemon.TicketCycleStateError) as exc:
            print("  warning: admin request is archived, but its recovery "
                  "journal needs user attention: " + str(exc))
    return archived


def park_failed_message(dispatch_path):
    """Move a claimed message to failed and verify its exact inode.

    Arguments:
      dispatch_path = the claimed mailbox file.

    Returns:
      True only when the destination provably owns the same inode
      and the source name is gone.
    """
    _, verified = daemon.verified_state_move(
        dispatch_path=dispatch_path,
        directory=daemon.os.path.join(daemon.MAILBOX, "failed"))
    return verified


def park_prelaunch_message(dispatch_path):
    """Retain a request that was refused before its agent process started.

    Arguments:
      dispatch_path = the claimed mailbox file.

    Returns:
      True only when the verified move into ``prelaunch/`` succeeded.
    """
    _, verified = daemon.verified_state_move(
        dispatch_path=dispatch_path,
        directory=daemon.os.path.join(daemon.MAILBOX, "prelaunch"))
    return verified


def regular_inode(path):
    """Return ``(device, inode)`` only for an exact regular-file path.

    An inode is the filesystem's identity for a file's contents,
    independent of its name; together with the device number it names
    one file uniquely.

    Arguments:
      path = the path to inspect, without following a final symlink.

    Returns:
      The ``(device, inode)`` pair, or ``None`` for a missing or
      non-regular path.
    """
    try:
        details = daemon.os.lstat(path)
    except OSError:
        return None
    if not daemon.stat.S_ISREG(details.st_mode):
        return None
    return details.st_dev, details.st_ino


def regular_file_has_prefix(path, prefix):
    """Read only one ASCII prefix while proving a stable regular inode.

    The file's identity and metadata are captured four times around
    the read; any change refuses the check rather than trusting a
    file that moved underneath it.

    Arguments:
      path   = the file to check.
      prefix = the nonempty expected leading bytes.

    Returns:
      True when the file starts with exactly those bytes.

    Raises:
      ValueError: for an empty prefix or a file that changed during
        the check.
    """
    if not isinstance(prefix, bytes) or not prefix:
        raise ValueError("file prefix must be nonempty bytes")
    initial = daemon.os.lstat(path)
    if not daemon.stat.S_ISREG(initial.st_mode):
        return False
    flags = daemon.os.O_RDONLY | daemon.os.O_NONBLOCK
    if hasattr(daemon.os, "O_NOFOLLOW"):
        flags |= daemon.os.O_NOFOLLOW
    descriptor = daemon.os.open(path, flags)
    try:
        before = daemon.os.fstat(descriptor)
        raw = daemon.os.read(descriptor, len(prefix))
        after = daemon.os.fstat(descriptor)
        current = daemon.os.lstat(path)
    finally:
        daemon.os.close(descriptor)
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
    """Restore the exact claimed inode from its safety guard if necessary.

    Arguments:
      guard_path    = the same-inode safety hardlink.
      dispatch_path = the original claimed name to restore.
      source_inode  = the identity the restored file must have.

    Returns:
      True when the claimed name again holds the source inode.
    """
    if not daemon.os.path.lexists(dispatch_path):
        try:
            daemon.os.link(guard_path, dispatch_path)
            daemon.fsync_directory(directory=daemon.os.path.dirname(dispatch_path))
        except OSError:
            pass
    return daemon.regular_inode(path=dispatch_path) == source_inode


def remove_state_guard(guard_path, source_inode):
    """Remove only the unchanged safety hardlink owned by this move.

    Arguments:
      guard_path   = the safety hardlink.
      source_inode = the identity the guard must still hold.

    Returns:
      True when the guard held the inode and its name is now gone.
    """
    if daemon.regular_inode(path=guard_path) != source_inode:
        return False
    try:
        daemon.os.unlink(guard_path)
    except OSError:
        return False
    return not daemon.os.path.lexists(guard_path)


def verified_state_move(dispatch_path, directory):
    """Move one regular inode and prove the destination owns that inode.

    A hardlink is a second name for the same inode. The move
    publishes the destination by hardlink, keeps one same-inode
    guard beside the source until the destination's identity is
    proven, and restores the source from the guard when anything
    disagrees.

    Arguments:
      dispatch_path = the claimed source file.
      directory     = the destination state directory.

    Returns:
      ``(destination, verified)``. The destination is None when
      publication itself failed; verification also requires the
      source path to be absent and the guard to be cleanly removed.
    """
    source_inode = daemon.regular_inode(path=dispatch_path)
    if source_inode is None:
        return None, False
    # move_without_overwrite() publishes by hardlink and then unlinks the
    # inflight source. Keep one same-inode guard beside that source until the
    # final destination identity is proven. A verification race can therefore
    # restore the exact inflight blocker, and a guard that itself cannot be
    # cleaned is recognized by inflight_lane_blockers() across later passes.
    guard_path = dispatch_path + daemon.STATE_GUARD_SUFFIX
    try:
        daemon.os.link(dispatch_path, guard_path)
        daemon.fsync_directory(directory=daemon.os.path.dirname(guard_path))
    except OSError:
        return None, False
    if daemon.regular_inode(path=guard_path) != source_inode:
        return None, False
    destination = daemon.move_without_overwrite(
        path=dispatch_path,
        directory=directory)
    if destination is None:
        restored = daemon.restore_state_source(
            guard_path=guard_path,
            dispatch_path=dispatch_path,
            source_inode=source_inode)
        if restored:
            daemon.remove_state_guard(
                guard_path=guard_path,
                source_inode=source_inode)
        return None, False
    destination_inode = daemon.regular_inode(path=destination)
    verified = (destination_inode == source_inode
                and not daemon.os.path.lexists(dispatch_path))
    if not verified:
        restored = daemon.restore_state_source(
            guard_path=guard_path,
            dispatch_path=dispatch_path,
            source_inode=source_inode)
        if restored:
            daemon.remove_state_guard(
                guard_path=guard_path,
                source_inode=source_inode)
        return destination, False
    if not daemon.remove_state_guard(
            guard_path=guard_path,
            source_inode=source_inode):
        # A leftover exact-name guard is itself a durable lane blocker. Restore
        # the ordinary inflight name too when the guard still owns our inode.
        daemon.restore_state_source(
            guard_path=guard_path,
            dispatch_path=dispatch_path,
            source_inode=source_inode)
        return destination, False
    return destination, True


def archive_consumed_message(dispatch_path):
    """Move a clean dispatch to done and verify the archive before success.

    Arguments:
      dispatch_path = the consumed inflight message.

    Returns:
      True only when the exact destination owns the source inode
      after the move; any other state leaves the message where it is
      and reports that dispatch is not consumed.
    """
    name = daemon.os.path.basename(dispatch_path)
    done_path, verified = daemon.verified_state_move(
        dispatch_path=dispatch_path,
        directory=daemon.DONE)
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
