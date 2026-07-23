"""Message preambles, ticket-flow envelopes, and return matching.

An envelope is the machine-read header block at the top of a mailbox
message: lines such as ``MAILBOX-CYCLE`` and ``MAILBOX-MODE`` that
bind the message to one exact ticket. This file builds each role's
preamble (the fixed instruction text placed before the routed
request), lists the pending messages a watch may claim, and matches
every returned block — an Architect GO, an Implementer handoff, a Red
Team receipt — to the exact cycle, commit, and mode it claims to
answer, refusing a mismatch instead of guessing.

Docstrings here shorten repeated names: the Implementer's candidate
commit is C, its accepted squash landing is L, a note-only
permanent-note landing is P, that landing's base main commit is B, and
the running watcher is D0.

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
    "route_inode_snapshot",
    "new_route_paths",
    "common_preamble_for_dispatch",
    "agent_preamble",
    "next_seq",
    "pending_messages",
    "ticket_cycle_mode_is_enabled",
    "canonical_ticket_cycle_topology",
    "message_is_enabled_for_topology",
    "enabled_pending_messages",
    "deferred_sol_messages",
    "fable_message_inode_snapshot",
    "daemon_message_inode_snapshot",
    "opus_message_inode_snapshot",
    "sol_message_inode_snapshot",
    "user_message_inode_snapshot",
    "matching_new_architect_go",
    "architect_handoff_problem",
    "checkpoint_architect_handoff_problem",
    "matching_new_architect_handoff",
    "matching_new_checkpoint_handoff",
    "matching_new_architect_notes_go",
    "_authoritative_handoff_contract_module",
    "directive_before_later_history",
    "prove_blocked_implementer_checkpoint",
    "prepare_implementer_evidence_contract",
    "matching_new_implementer_handoff",
    "matching_new_redteam_receipt",
    "matching_new_control_plane_receipt",
    "sol_ticket_kind",
    "sol_ticket_body_after_kind",
    "_sol_discovery_envelope",
    "sol_discovery_severity_problem",
    "sol_discovery_severity",
    "sol_discovery_scope",
    "public_architect_sol_downstream_problem",
    "_body_architect_admission",
    "public_architect_sol_outcome_problem",
    "_public_architect_no_ticket_receipt",
    "public_architect_no_ticket_problem",
    "_ticket_flow_envelope",
    "is_implementer_checkpoint_request",
    "is_implementer_budget_checkpoint",
    "is_architect_budget_repair",
    "is_implementer_time_checkpoint",
    "is_implementer_context_handoff",
    "_context_handoff_field",
    "parse_context_handoff",
    "context_handoff_problem",
    "matching_new_context_handoff",
    "latest_context_handoff_path",
    "replacement_context_notice",
    "checkpoint_handoff_problem",
    "_ticket_architect_admission",
    "_redteam_closure_envelope",
    "_redteam_control_plane_envelope",
    "_control_plane_review_receipt",
    "redteam_closure_problem",
    "redteam_closure_ticket",
    "redteam_closure_commit",
    "_redteam_review_receipt",
    "_architect_go_request",
    "architect_go_request_payload",
    "backlog_close_request_payload",
    "_architect_notes_go_request",
    "architect_notes_go_request_payload",
    "_architect_notes_admin_envelope",
    "architect_notes_admin_payload",
    "is_architect_notes_admin_message",
    "architect_notes_admin_journal_path",
    "write_architect_notes_admin_journal",
    "read_architect_notes_admin_journal",
    "remove_architect_notes_admin_journal",
    "_architect_notes_admin_request_path",
    "_validated_commit_admin_journals",
    "retire_validated_commit_admin_journal",
)


def common_preamble_for_dispatch(checkpoint_audit):
    """Omit ordinary landing instructions during a checkpoint review.

    Arguments:
      checkpoint_audit = True when this turn reviews a checkpoint.

    Returns:
      The checkpoint preamble or the ordinary one.
    """
    return daemon.CHECKPOINT_PREAMBLE if checkpoint_audit else daemon.PREAMBLE


def agent_preamble(agent, message=None):
    """Return role-specific standing text that precedes the common wrapper.

    Arguments:
      agent   = the dispatched role.
      message = the message being dispatched, or ``None``; Architect
                turns inspect it to pick the reopening, checkpoint, or
                context-handoff variant of the preamble.

    Returns:
      The role's standing preamble text.
    """
    if agent == "fable":
        checkpoint_notice = ""
        checkpoint_audit = False
        reopen_turn = False
        if message is not None and message.startswith(daemon.MAILBOX_RETURN_HEADER):
            _cycle_id, _, result, _, problem = daemon._redteam_review_receipt(
                message=message)
            reopen_turn = problem is None and result == "REOPEN"
        if message is not None and message.startswith(daemon.MAILBOX_FLOW_HEADER):
            cycle_id, _mode, body, problem = daemon._ticket_flow_envelope(
                message=message)
            context_handoff = (
                problem is None and daemon.is_implementer_context_handoff(body))
            checkpoint_audit = (
                problem is None and daemon.is_implementer_checkpoint_request(body))
            if context_handoff:
                checkpoint_notice = (
                    "IMPLEMENTER CONTEXT HANDOFF: this is the prior "
                    "Implementer's exact record, not a candidate or a "
                    "completed ticket. Check the stated repository state. "
                    "Send one same-cycle replacement handoff with exactly "
                    "one **Checkpoint decision:** `GO` or `NO-GO` row. GO "
                    "continues from this record; NO-GO parks or revises the "
                    "work. Do not rewrite the record as a summary.\n\n")
            elif (checkpoint_audit
                  and daemon.is_implementer_budget_checkpoint(body)):
                checkpoint_notice = (
                    "IMPLEMENTER BUDGET CHECKPOINT: the exact candidate is "
                    "preserved but cannot receive GO. Inspect why it exceeds "
                    "the binding character limit. Send one same-cycle "
                    "Implementer handoff beginning exactly `"
                    + daemon.ARCHITECT_BUDGET_REPAIR_HEADING
                    + "`, with exactly one Directive row and "
                    "exactly `- **Checkpoint decision:** `NO-GO``. Revise "
                    "the plan to reduce the change; do not raise the saved "
                    "limit or discard required behavior silently.\n\n")
            elif checkpoint_audit:
                checkpoint_notice = (
                    "90-MINUTE IMPLEMENTER CHECKPOINT: inspect the saved "
                    "candidate, then send one revised same-cycle handoff "
                    "to the Implementer with exactly one **Checkpoint "
                    "decision:** `GO` or `NO-GO` row. This turn cannot land "
                    "the checkpoint candidate.\n\n")
            elif (problem is None
                    and "### IMPLEMENTER_HANDOFF:" in body
                    and "- Acceptance: `blocked`" in body):
                checkpoint_notice = (
                    "BLOCKED IMPLEMENTER CHECKPOINT: this return is not a "
                    "candidate and cannot receive GO. If the actual runtime "
                    "cannot launch subagents, a revised capability exception "
                    "must bind the source note to these exact rows:\n"
                    "- Source cycle: `" + cycle_id + "`\n"
                    "- Source handoff SHA-256: `"
                    + daemon.hashlib.sha256(body.encode("utf-8")).hexdigest()
                    + "`\n\n")
        landing = ("" if checkpoint_audit or reopen_turn
                   else daemon.ARCHITECT_LANDING_PREAMBLE)
        return checkpoint_notice + daemon.ARCHITECT_ROLE_PREAMBLE + landing
    if agent == "opus":
        return (daemon.IMPLEMENTER_ROLE_PREAMBLE
                + "AUTHORITATIVE IMPLEMENTER ROLE FILE:\n    "
                + daemon.os.path.join(daemon.AGENT_CWD["fable"], ".claude",
                               "OPUS_ROLE.md")
                + "\nRead this primary copy instead of a possibly older "
                "copy in the candidate checkout.\n\n")
    if agent == "sol":
        primary = daemon.AGENT_CWD["fable"]
        authoritative = (
            "AUTHORITATIVE ROLE FILES (read these absolute paths, not stale "
            "copies in the Sol checkout):\n"
            "    " + daemon.os.path.join(primary, ".codex",
                                  "REDTEAM_ROLE.md") + "\n"
            "    " + daemon.os.path.join(primary, ".claude",
                                  "OPUS_ROLE.md") + "\n"
            "AUTHORITATIVE TICKET TOOLS (run these absolute primary paths, "
            "not relative copies in the Sol checkout):\n"
            "    " + daemon.os.path.join(
                primary, "ai", "tools", "handoff_contract.py") + "\n"
            "    " + daemon.os.path.join(
                primary, "ai", "tools", "ticket_change_guard.py") + "\n"
            "For a character check, pass `--repo` followed by the exact "
            "candidate worktree from the directive. Never measure the Sol "
            "checkout merely because it is this turn's current folder.\n")
        return daemon.REDTEAM_ROLE_PREAMBLE + authoritative + "\n"
    raise ValueError("unknown mailbox agent: " + repr(agent))


def next_seq():
    """Return the next zero-padded mailbox sequence number as a string.

    Scans EVERY directory under the mailbox (root, done/, failed/, any
    hand-made quarantine like hold/): a number parked anywhere is still
    claimed, and handing it out twice makes two messages look like one.
    """
    highest = 0
    pattern = daemon.os.path.join(daemon.MAILBOX, "**", "*.md")
    for path in daemon.glob.glob(pattern, recursive=True):
        value = daemon.sequence_in_name(name=daemon.os.path.basename(path))
        if value is not None:
            if value > highest:
                highest = value
    return "%04d" % (highest + 1)


def pending_messages():
    """Return the sorted list of unprocessed message paths."""
    found = []
    for path in daemon.glob.glob(daemon.os.path.join(daemon.MAILBOX, "*.md")):
        name = daemon.os.path.basename(path)
        if daemon.PENDING_MESSAGE_RE.match(name):
            found.append(path)
    found.sort(key=daemon.message_sequence)
    return found


def ticket_cycle_mode_is_enabled(mode, skip_redteam=False):
    """Return whether this watch topology owns ``mode`` work.

    Arguments:
      mode         = the ticket's saved mode.
      skip_redteam = True in a two-role watch.

    Returns:
      True when the mode matches this watch's topology: a three-role
      watch owns normal tickets, a two-role watch owns two-role ones.
    """
    if skip_redteam:
        return mode == "two-role"
    return mode == "normal"


def canonical_ticket_cycle_topology(skip_redteam=False):
    """Return the single durable identity of this watch's role layout.

    Arguments:
      skip_redteam = True in a two-role watch.

    Returns:
      ``"two-role"`` or ``"normal"``.
    """
    if skip_redteam:
        return "two-role"
    return "normal"


def message_is_enabled_for_topology(path, skip_redteam=False):
    """Return whether this watch may consume one root mailbox message.

    Messages for a disabled role remain byte-for-byte in the mailbox root.
    Malformed messages stay enabled so the ordinary dispatcher can explain
    and quarantine them instead of silently treating corruption as a role.
    """
    match = daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(path))
    if match is None:
        return False
    agent = match.group(1)
    try:
        message = daemon.read_cycle_message(path=path)
    except (OSError, ValueError, daemon.TicketCycleStateError):
        return True
    if agent == "daemon":
        if message.startswith(
                daemon.MAILBOX_RETURN_HEADER + "redteam-control-plane"):
            return not skip_redteam
        _, _, mode, problem = daemon._architect_go_request(message=message)
        return (True if problem is not None else
                daemon.ticket_cycle_mode_is_enabled(
                    mode=mode, skip_redteam=skip_redteam))
    if agent == "sol":
        return not skip_redteam
    if agent == "fable" and message.startswith(daemon.MAILBOX_RETURN_HEADER):
        return not skip_redteam
    if message.startswith(daemon.MAILBOX_FLOW_HEADER):
        _, mode, _, problem = daemon._ticket_flow_envelope(message=message)
        return (True if problem is not None else
                daemon.ticket_cycle_mode_is_enabled(
                    mode=mode, skip_redteam=skip_redteam))
    return True


def enabled_pending_messages(skip_redteam=False):
    """Return root messages eligible for this watch topology.

    The ordinary three-role topology returns every dispatchable message.
    A two-role watch excludes only exact ``to-sol`` roots; those files stay
    in place for a later Sol-enabled watch.
    """
    return [
        path for path in daemon.pending_messages()
        if daemon.message_is_enabled_for_topology(
            path=path, skip_redteam=skip_redteam)]


def deferred_sol_messages():
    """Return exact pending Sol roots held by a two-role watch."""
    return [path for path in daemon.pending_messages()
            if daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(path)).group(1)
            == "sol"]


def route_inode_snapshot(pattern):
    """Return the inode set of every regular file on one mailbox route.

    Arguments:
      pattern = the route file pattern, for example "*-to-fable.md".

    Returns:
      A set of inode identities covering the mailbox root and its
      subdirectories; empty when the mailbox directory does not exist.
    """
    snapshot = set()
    if not daemon.os.path.isdir(daemon.MAILBOX):
        return snapshot
    for path in daemon.glob.glob(
            daemon.os.path.join(daemon.MAILBOX, "**", pattern),
            recursive=True):
        inode = daemon.regular_inode(path=path)
        if inode is not None:
            snapshot.add(inode)
    return snapshot


def new_route_paths(pattern, before_inodes):
    """Return every mailbox file on one route that a turn newly produced.

    Arguments:
      pattern = the route file pattern, for example "*-to-fable.md".
      before_inodes = the route's inode snapshot taken before the turn ran.

    Returns:
      Paths under the mailbox root and its subdirectories whose regular
      inode is absent from the snapshot.
    """
    fresh = []
    for path in daemon.glob.glob(
            daemon.os.path.join(daemon.MAILBOX, "**", pattern),
            recursive=True):
        inode = daemon.regular_inode(path=path)
        if inode is None or inode in before_inodes:
            continue
        fresh.append(path)
    return fresh


def fable_message_inode_snapshot():
    """Return regular inodes for every existing Architect-addressed message."""
    return daemon.route_inode_snapshot(pattern="*-to-fable.md")


def daemon_message_inode_snapshot():
    """Return regular inodes for every existing daemon-addressed message."""
    return daemon.route_inode_snapshot(pattern="*-to-daemon.md")


def opus_message_inode_snapshot():
    """Return regular inodes for every existing Implementer message."""
    return daemon.route_inode_snapshot(pattern="*-to-opus.md")


def sol_message_inode_snapshot():
    """Return regular inodes for every existing Red Team message."""
    return daemon.route_inode_snapshot(pattern="*-to-sol.md")


def user_message_inode_snapshot():
    """Return regular inodes for every existing human-addressed message."""
    return daemon.route_inode_snapshot(pattern="*-to-user.md")


def matching_new_architect_go(cycle_id, candidate_commit, mode,
                               before_inodes):
    """Prove any GO created by this Architect turn names its exact candidate C.

    Only messages created since the pre-turn snapshot count. Every
    fresh GO must read cleanly and name this exact cycle, candidate,
    and mode; at most one may exist.

    Arguments:
      cycle_id         = the ticket cycle.
      candidate_commit = the candidate C the GO must name.
      mode             = the ticket mode.
      before_inodes    = the pre-turn message-inode snapshot.

    Returns:
      ``(path, offending, problem)``: the single fresh GO (or
      ``None``), the paths a failure must handle, and a printable
      problem (``None`` on success).
    """
    fresh = []
    problems = []
    problem_paths = []
    for path in daemon.new_route_paths(
            pattern="*-to-daemon.md",
            before_inodes=before_inodes):
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="Architect GO " + daemon.os.path.basename(path))
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            problems.append(daemon.os.path.basename(path) + ": " + str(exc))
            problem_paths.append(path)
            continue
        returned_cycle, returned_candidate, returned_mode, problem = (
            daemon._architect_go_request(message=message))
        if problem is not None:
            problems.append(daemon.os.path.basename(path) + ": " + problem)
            problem_paths.append(path)
            continue
        if (returned_cycle != cycle_id
                or returned_candidate != candidate_commit
                or returned_mode != mode):
            problems.append(
                daemon.os.path.basename(path)
                + ": GO does not name this turn's exact cycle, candidate, "
                "and mode")
            problem_paths.append(path)
            continue
        fresh.append(path)
    if problems:
        return (None, list(dict.fromkeys(problem_paths + fresh)),
                "; ".join(problems))
    if len(fresh) > 1:
        return (None, fresh,
                "expected at most one new exact Architect GO; found "
                + str(len(fresh)))
    return (fresh[0] if fresh else None), [], None


def architect_handoff_problem(message, cycle_id, mode, checkpoint=False,
                              budget=False):
    """Return why one same-cycle Architect repair is not authorized.

    An ordinary repair must keep the cycle and mode and carry exactly
    one Directive row. A checkpoint decision instead carries exactly
    one GO or NO-GO row; a budget checkpoint additionally needs the
    revised-plan heading, a NO-GO decision, and one revised Directive
    row.

    Arguments:
      message    = the handoff text.
      cycle_id   = the required cycle.
      mode       = the required mode.
      checkpoint = True for a checkpoint decision.
      budget     = True for a budget checkpoint.

    Returns:
      A printable problem, or ``None`` when the handoff is
      authorized.
    """
    returned_cycle, returned_mode, body, problem = (
        daemon._ticket_flow_envelope(message=message))
    if problem is not None:
        return problem
    if returned_cycle != cycle_id:
        return "Architect handoff changed MAILBOX-CYCLE"
    if returned_mode != mode:
        return "Architect handoff changed MAILBOX-MODE"
    if not checkpoint:
        if len(daemon.ARCHITECT_DIRECTIVE_LINE_RE.findall(message)) != 1:
            return "repair handoff requires exactly one Directive row"
        return None
    decision_rows = [
        line for line in body.splitlines()
        if line.startswith(daemon.IMPLEMENTER_CHECKPOINT_DECISION_PREFIX)]
    accepted_rows = {
        daemon.IMPLEMENTER_CHECKPOINT_DECISION_PREFIX + " `GO`",
        daemon.IMPLEMENTER_CHECKPOINT_DECISION_PREFIX + " `NO-GO`",
    }
    if len(decision_rows) != 1 or decision_rows[0] not in accepted_rows:
        return "checkpoint handoff requires exactly one GO or NO-GO row"
    if budget:
        if not daemon.is_architect_budget_repair(body):
            return "budget checkpoint requires its exact revised-plan heading"
        if decision_rows[0] != (daemon.IMPLEMENTER_CHECKPOINT_DECISION_PREFIX
                                + " `NO-GO`"):
            return "budget checkpoint requires a NO-GO repair decision"
        if len(daemon.ARCHITECT_DIRECTIVE_LINE_RE.findall(message)) != 1:
            return "budget checkpoint requires exactly one revised Directive row"
    return None


def checkpoint_architect_handoff_problem(message, cycle_id, mode):
    """Compatibility name for the stricter checkpoint form.

    Arguments:
      message  = the handoff text.
      cycle_id = the required cycle.
      mode     = the required mode.

    Returns:
      The checkpoint-form problem, or ``None``.
    """
    return daemon.architect_handoff_problem(
        message=message, cycle_id=cycle_id, mode=mode, checkpoint=True)


def matching_new_architect_handoff(cycle_id, mode, before_inodes,
                                   checkpoint=False, required=True,
                                   budget=False):
    """Find one fresh same-cycle repair handoff from the Architect.

    Arguments:
      cycle_id      = the ticket cycle.
      mode          = the ticket mode.
      before_inodes = the pre-turn message-inode snapshot.
      checkpoint    = True for a checkpoint decision.
      required      = True when exactly one handoff must exist.
      budget        = True for a budget checkpoint.

    Returns:
      ``(path, offending, problem)`` as in the other matchers; a
      handoff outside the mailbox root is invalid.
    """
    fresh = []
    invalid = []
    problems = []
    for path in daemon.new_route_paths(
            pattern="*-to-opus.md",
            before_inodes=before_inodes):
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError) as exc:
            invalid.append(path)
            problems.append(daemon.os.path.basename(path) + ": " + str(exc))
            continue
        problem = daemon.architect_handoff_problem(
            message=message, cycle_id=cycle_id, mode=mode,
            checkpoint=checkpoint, budget=budget)
        if daemon.os.path.dirname(path) != daemon.MAILBOX:
            problem = "handoff was not published in the mailbox root"
        if problem is not None:
            invalid.append(path)
            problems.append(daemon.os.path.basename(path) + ": " + problem)
        else:
            fresh.append(path)
    if problems:
        return None, list(dict.fromkeys(invalid + fresh)), "; ".join(problems)
    if len(fresh) > 1 or (required and len(fresh) != 1):
        return (None, fresh,
                "expected exactly one new Architect handoff to the "
                "Implementer; found " + str(len(fresh)))
    return (fresh[0] if fresh else None), [], None


def matching_new_checkpoint_handoff(cycle_id, mode, before_inodes,
                                    budget=False):
    """Compatibility name for a required checkpoint decision.

    Arguments:
      cycle_id      = the ticket cycle.
      mode          = the ticket mode.
      before_inodes = the pre-turn snapshot.
      budget        = True for a budget checkpoint.

    Returns:
      The required-checkpoint matcher's triple.
    """
    return daemon.matching_new_architect_handoff(
        cycle_id=cycle_id, mode=mode, before_inodes=before_inodes,
        checkpoint=True, required=True, budget=budget)


def matching_new_architect_notes_go(base_commit, notes_commit,
                                     before_inodes):
    """Prove exactly one fresh note-only GO binds this Architect turn's B and P.

    Arguments:
      base_commit   = B the GO must name.
      notes_commit  = P the GO must name.
      before_inodes = the pre-turn message-inode snapshot.

    Returns:
      ``(path, offending, problem)`` as in the other matchers.
    """
    fresh = []
    invalid = []
    problems = []
    for path in daemon.new_route_paths(
            pattern="*-to-daemon.md",
            before_inodes=before_inodes):
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="Architect notes GO " + daemon.os.path.basename(path))
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            invalid.append(path)
            problems.append(daemon.os.path.basename(path) + ": " + str(exc))
            continue
        returned_base, returned_notes, problem = (
            daemon._architect_notes_go_request(message=message))
        if problem is not None:
            invalid.append(path)
            problems.append(daemon.os.path.basename(path) + ": " + problem)
            continue
        if (returned_base != base_commit
                or returned_notes != notes_commit):
            invalid.append(path)
            problems.append(
                daemon.os.path.basename(path)
                + ": notes GO does not bind this turn's exact B and P")
            continue
        fresh.append(path)
    if problems:
        return None, invalid, "; ".join(problems)
    if len(fresh) != 1:
        return (None, fresh,
                "a permanent-note commit requires exactly one fresh "
                "architect-notes-go request; found " + str(len(fresh)))
    return fresh[0], [], None


def _authoritative_handoff_contract_module():
    """Load the already-proved primary contract without importing a copy."""
    path = daemon.os.path.join(
        daemon.AGENT_CWD["fable"], "ai", "tools", "handoff_contract.py")
    try:
        spec = daemon.importlib.util.spec_from_file_location(
            "_mailbox_authoritative_handoff_contract", path)
        if spec is None or spec.loader is None:
            raise ImportError("no Python loader")
        module = daemon.importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError, SyntaxError) as exc:
        raise daemon.TicketCycleStateError(
            "cannot load authoritative handoff contract: " + str(exc)) \
            from exc
    required = (
        "DirectiveError", "validate_directive_file", "validate_directive_text",
        "extract_implementer_subagent_evidence",
        "extract_blocked_implementer_capability_evidence",
        "validate_implementer_handoff_subagent_evidence")
    if any(not hasattr(module, name) for name in required):
        raise daemon.TicketCycleStateError(
            "authoritative handoff contract lacks Implementer evidence API")
    return module


def directive_before_later_history(text):
    """Keep one directive and its first evidence block for restart recovery.

    Arguments:
      text = the source note's text.

    Returns:
      The text cut at the first heading after the evidence block, so
      a note that grew later history can still be validated as the
      directive it was when the ticket started.
    """
    evidence = "## Implementation evidence / resume state"
    lines = text.splitlines()
    found_evidence = False
    for index, line in enumerate(lines):
        if line.casefold() == evidence.casefold():
            found_evidence = True
            continue
        if found_evidence and daemon.re.match(r"^#{1,2}(?:\s|$)", line):
            return "\n".join(lines[:index]) + "\n"
    return text


def prove_blocked_implementer_checkpoint(cycle_id, handoff_sha256,
                                         contract):
    """Bind a capability retry to one actual prior blocked return.

    Arguments:
      cycle_id       = the current ticket cycle.
      handoff_sha256 = digest the prior blocked handoff body must
                       hash to.
      contract       = the authoritative handoff-contract module.

    Returns:
      The prior blocked return's capability evidence mapping.

    Raises:
      daemon.TicketCycleStateError: when exactly one digest-bound
        blocked handoff does not exist in this cycle.
    """
    matches = []
    for path in daemon.glob.glob(
            daemon.os.path.join(daemon.MAILBOX, "**", "*-to-fable.md"),
            recursive=True):
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="blocked Implementer checkpoint")
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError):
            continue
        if not message.startswith(daemon.MAILBOX_FLOW_HEADER):
            continue
        returned_cycle, _mode, body, problem = daemon._ticket_flow_envelope(
            message=message)
        if problem is not None or returned_cycle != cycle_id:
            continue
        if daemon.hashlib.sha256(body.encode("utf-8")).hexdigest() != handoff_sha256:
            continue
        try:
            evidence = (
                contract.extract_blocked_implementer_capability_evidence(
                handoff_text=body)
            )
        except contract.DirectiveError:
            continue
        matches.append(evidence)
    if len(matches) != 1:
        raise daemon.TicketCycleStateError(
            "capability exception is not bound to exactly one actual "
            "blocked IMPLEMENTER_HANDOFF in this MAILBOX-CYCLE")
    return matches[0]


def prepare_implementer_evidence_contract(message, use_saved_limit=False):
    """Freeze the Architect's parsed subagent plan before Opus launches.

    The handoff's one cited source note is revalidated through the
    authoritative contract, so the evidence the Implementer must
    later return is fixed before its turn starts. A
    capability-unavailable plan must additionally bind to one actual
    digest-matched blocked handoff from this same cycle, field for
    field.

    Arguments:
      message         = the Architect handoff text.
      use_saved_limit = True during recovery, when the note's own
                        saved budget is trusted and later note
                        history may be cut before validation.

    Returns:
      Mapping with the contract module, the parsed plan, the note
      path, the allowed paths, the character limit, and the ticket
      class.

    Raises:
      daemon.TicketCycleStateError: for a missing or redirected note,
        an invalid directive, or an unbound capability exception.
    """
    matches = daemon.ARCHITECT_DIRECTIVE_LINE_RE.findall(message)
    if len(matches) != 1:
        raise daemon.TicketCycleStateError(
            "Architect handoff must cite exactly one ai/notes source note "
            "and its Implementation directive")
    relative = matches[0]
    note_path = daemon.os.path.abspath(
        daemon.os.path.join(daemon.AGENT_CWD["fable"], relative))
    notes_root = daemon.os.path.abspath(
        daemon.os.path.join(daemon.AGENT_CWD["fable"], "ai", "notes"))
    if (daemon.os.path.commonpath((note_path, notes_root)) != notes_root
            or daemon.os.path.realpath(note_path) != note_path):
        raise daemon.TicketCycleStateError(
            "Architect handoff cites a redirected or external source note")
    contract = daemon._authoritative_handoff_contract_module()
    try:
        directive = contract.validate_directive_file(
            role="architect", path=note_path,
            expected_max=(None if use_saved_limit else daemon.MAX_CHARACTERS))
    except contract.DirectiveError as exc:
        if (use_saved_limit
                and "may repeat only consecutively" in str(exc)):
            try:
                source = daemon.stable_regular_bytes(
                    path=note_path,
                    maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                    label="saved Architect source directive").decode(
                        "utf-8", errors="strict")
                directive = contract.validate_directive_text(
                    role="architect",
                    text=daemon.directive_before_later_history(source),
                    expected_max=None)
            except (UnicodeDecodeError, OSError, ValueError,
                    contract.DirectiveError) as recovery_exc:
                raise daemon.TicketCycleStateError(
                    "Architect source directive is invalid: "
                    + str(recovery_exc)) from recovery_exc
        else:
            raise daemon.TicketCycleStateError(
                "Architect source directive is invalid: " + str(exc)) \
                from exc
    plan = directive.get("parallel_work_plan")
    if not isinstance(plan, dict):
        raise daemon.TicketCycleStateError(
            "Architect source directive has no parsed Parallel work plan")
    if plan.get("mode") == "capability-unavailable":
        cycle_id, _mode, _body, problem = daemon._ticket_flow_envelope(
            message=message)
        checkpoint = directive.get("capability_checkpoint")
        if (problem is not None or not isinstance(checkpoint, dict)
                or checkpoint.get("cycle") != cycle_id):
            raise daemon.TicketCycleStateError(
                "capability exception checkpoint does not name the current "
                "MAILBOX-CYCLE")
        prior_failure = daemon.prove_blocked_implementer_checkpoint(
            cycle_id=cycle_id,
            handoff_sha256=checkpoint.get("handoff_sha256", ""),
            contract=contract)
        for field in (
                "capability_checked", "attempted_operation", "raw_failure"):
            if prior_failure.get(field) != plan.get(field):
                raise daemon.TicketCycleStateError(
                    "capability exception field '" + field
                    + "' does not exactly match the digest-bound blocked "
                    "IMPLEMENTER_HANDOFF")
    role_plan = directive.get("role_plan")
    if (not isinstance(role_plan, dict)
            or role_plan.get("ticket_class") not in daemon.TICKET_CLASSES):
        raise daemon.TicketCycleStateError(
            "Architect source directive has no validated Ticket class")
    return {"contract": contract, "parallel_work_plan": plan,
            "note_path": note_path,
            "allowed_paths": frozenset(directive["allowed_paths"]),
            "character_limit": directive["character_change_budget"]["limit"],
            "ticket_class": role_plan["ticket_class"]}


def matching_new_implementer_handoff(cycle_id, mode, candidate_commit,
                                     before_inodes, evidence_contract):
    """Prove one same-cycle Opus return and its exact subagent evidence.

    Arguments:
      cycle_id          = the ticket cycle.
      mode              = the ticket mode.
      candidate_commit  = the candidate C the return must name.
      before_inodes     = the pre-turn message-inode snapshot.
      evidence_contract = the frozen plan from
                          prepare_implementer_evidence_contract.

    Returns:
      ``(path, offending, problem)`` as in the other matchers, with
      the validated evidence attached when the single return passed.
    """
    matches = []
    malformed = []
    malformed_paths = []
    evidence_results = []
    for path in daemon.new_route_paths(
            pattern="*-to-fable.md",
            before_inodes=before_inodes):
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="Implementer return " + daemon.os.path.basename(path))
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            malformed.append(daemon.os.path.basename(path) + ": " + str(exc))
            malformed_paths.append(path)
            continue
        if not message.startswith(daemon.MAILBOX_FLOW_HEADER):
            continue
        returned_cycle, returned_mode, body, problem = (
            daemon._ticket_flow_envelope(message=message))
        if problem is not None:
            malformed.append(daemon.os.path.basename(path) + ": " + problem)
            malformed_paths.append(path)
            continue
        if returned_cycle != cycle_id:
            continue
        if returned_mode != mode:
            malformed.append(
                daemon.os.path.basename(path) + ": returned mode changed")
            malformed_paths.append(path)
            continue
        candidate_lines = daemon.IMPLEMENTER_CANDIDATE_LINE_RE.findall(body)
        if candidate_lines != [candidate_commit]:
            malformed.append(
                daemon.os.path.basename(path)
                + ": Candidate commit does not name the exact Opus HEAD")
            malformed_paths.append(path)
            continue
        if daemon.is_implementer_budget_checkpoint(body):
            problem = daemon.checkpoint_handoff_problem(message=message)
            if problem is not None:
                malformed.append(daemon.os.path.basename(path) + ": " + problem)
                malformed_paths.append(path)
                continue
            matches.append(path)
            evidence_results.append({"completion_ready": True})
            continue
        try:
            evidence_result = evidence_contract["contract"].\
                validate_implementer_handoff_subagent_evidence(
                    parallel_work_plan=(
                        evidence_contract["parallel_work_plan"]),
                    handoff_text=body)
        except evidence_contract["contract"].DirectiveError as exc:
            malformed.append(daemon.os.path.basename(path) + ": " + str(exc))
            malformed_paths.append(path)
            continue
        matches.append(path)
        evidence_results.append(evidence_result)
    if malformed:
        return None, malformed_paths, "; ".join(malformed), None
    if len(matches) != 1:
        return (None, [],
                "expected exactly one new same-cycle IMPLEMENTER_HANDOFF; "
                "found " + str(len(matches)), None)
    return (matches[0], [], None,
            bool(evidence_results[0].get("completion_ready")))


def matching_new_redteam_receipt(cycle_id, accepted_commit, before_inodes):
    """Return one new correlated Red Team receipt path and result.

    The scan spans mailbox states so a future refactor that consumes the
    Architect lane concurrently cannot turn a real return into a false
    missing-receipt failure.
    """
    matches = []
    malformed = []
    for path in daemon.new_route_paths(
            pattern="*-to-fable.md",
            before_inodes=before_inodes):
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="Red Team return " + daemon.os.path.basename(path))
            message = raw.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, OSError, ValueError) as exc:
            malformed.append(daemon.os.path.basename(path) + ": " + str(exc))
            continue
        if not message.startswith(daemon.MAILBOX_RETURN_HEADER):
            continue
        returned_cycle, returned_commit, result, _, problem = (
            daemon._redteam_review_receipt(message=message))
        if problem is not None:
            malformed.append(daemon.os.path.basename(path) + ": " + problem)
            continue
        if returned_cycle == cycle_id and returned_commit == accepted_commit:
            matches.append((path, result))
    if malformed:
        return None, None, "; ".join(malformed)
    if len(matches) != 1:
        return (None, None,
                "expected exactly one new matching Red Team return; found "
                + str(len(matches)))
    return matches[0][0], matches[0][1], None


def matching_new_control_plane_receipt(cycle_id, candidate,
                                       before_inodes):
    """Prove one new exact Red Team key addressed to D0.

    Arguments:
      cycle_id      = the protected ticket cycle.
      candidate     = the reviewed candidate C.
      before_inodes = the pre-turn message-inode snapshot.

    Returns:
      ``(path, offending, problem)`` as in the other matchers, for
      the single fresh control-plane decision naming this exact
      cycle and candidate.
    """
    matches = []
    malformed = []
    for path in daemon.new_route_paths(
            pattern="*-to-daemon.md",
            before_inodes=before_inodes):
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError) as exc:
            malformed.append(daemon.os.path.basename(path) + ": " + str(exc))
            continue
        if not message.startswith(
                daemon.MAILBOX_RETURN_HEADER + "redteam-control-plane"):
            continue
        found_cycle, found_candidate, result, _body, problem = (
            daemon._control_plane_review_receipt(message=message))
        if problem is not None:
            malformed.append(daemon.os.path.basename(path) + ": " + problem)
            continue
        if found_cycle == cycle_id and found_candidate == candidate:
            matches.append((path, result))
    if malformed:
        return None, None, "; ".join(malformed)
    if len(matches) != 1:
        return (None, None,
                "expected exactly one new exact control-plane return; found "
                + str(len(matches)))
    return matches[0][0], matches[0][1], None


def sol_ticket_kind(message):
    """Return a Sol message's exact first-line class, or ``None``.

    Free-form prose is deliberately never classified.  LF and CRLF are both
    accepted as physical line endings, but whitespace, aliases, and a header
    appearing later in the body do not count.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.SOL_TICKET_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.SOL_DISPATCH_TICKET_KINDS))
        + r")(?:\r?\n|\Z)",
        message)
    if match is None:
        return None
    return match.group(1)


def sol_ticket_body_after_kind(message):
    """Return the bytes after a valid Sol classification line.

    Arguments:
      message = the decoded Sol message.

    Returns:
      The remainder after the ticket-kind line, or the whole message
      when no valid classification line starts it.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.SOL_TICKET_HEADER)
        + r"(?:" + "|".join(map(daemon.re.escape, daemon.SOL_DISPATCH_TICKET_KINDS))
        + r")(?:\r?\n|\Z)",
        message)
    if match is None:
        return message
    return message[match.end():]


def _sol_discovery_envelope(message):
    """Parse the exact persisted discovery envelope without reading prose.

    Returns:
      ``(severity, scope, body, problem)``. A valid discovery has all three
      exact physical header lines in order. Other ticket kinds may not use a
      reserved severity or scope line.
    """
    ticket_kind = daemon.sol_ticket_kind(message=message)
    remainder = daemon.sol_ticket_body_after_kind(message=message)
    severity_like_line = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*severity[ \t]*:")
    scope_like_line = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*scope[ \t]*:")
    if ticket_kind != "discovery":
        if daemon.re.search(severity_like_line, remainder) is not None:
            return (None, None, remainder,
                    "MAILBOX-SEVERITY is reserved for discovery tickets "
                    "and must not appear on another ticket kind")
        if daemon.re.search(scope_like_line, remainder) is not None:
            return (None, None, remainder,
                    "MAILBOX-SCOPE is reserved for discovery tickets and "
                    "must not appear on another ticket kind")
        return None, None, remainder, None

    severity_match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.SOL_SEVERITY_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.DISCOVERY_SEVERITIES))
        + r")\r?\n",
        remainder)
    if severity_match is None:
        return (None, None, remainder,
                "a discovery ticket requires exactly "
                "'MAILBOX-SEVERITY: high', 'MAILBOX-SEVERITY: medium', or "
                "'MAILBOX-SEVERITY: low' as its second physical line")

    after_severity = remainder[severity_match.end():]
    scope_match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.SOL_SCOPE_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.DISCOVERY_SCOPES))
        + r")(?:\r?\n|\Z)",
        after_severity)
    if scope_match is None:
        return (None, None, remainder,
                "a discovery ticket requires exactly "
                "'MAILBOX-SCOPE: bounded' or 'MAILBOX-SCOPE: widespread' "
                "as its third physical line")

    body = after_severity[scope_match.end():]
    if daemon.re.search(severity_like_line, body) is not None:
        return None, None, remainder, "duplicate MAILBOX-SEVERITY line"
    if daemon.re.search(scope_like_line, body) is not None:
        return None, None, remainder, "duplicate MAILBOX-SCOPE line"
    return severity_match.group(1), scope_match.group(1), body, None


def sol_discovery_severity_problem(message):
    """Return an exact discovery-envelope error, or ``None``.

    Arguments:
      message = the decoded Sol discovery message.

    Returns:
      The printable problem, or ``None`` for a valid envelope.
    """
    return daemon._sol_discovery_envelope(message=message)[3]


def sol_discovery_severity(message):
    """Return a valid discovery ticket's saved severity, or ``None``.

    Arguments:
      message = the decoded Sol discovery message.

    Returns:
      The saved severity, or ``None`` for a malformed envelope.
    """
    severity, _, _, problem = daemon._sol_discovery_envelope(message=message)
    return severity if problem is None else None


def sol_discovery_scope(message):
    """Return a valid discovery ticket's saved scope, or ``None``.

    Arguments:
      message = the decoded Sol discovery message.

    Returns:
      The saved scope, or ``None`` for a malformed envelope.
    """
    _, scope, _, problem = daemon._sol_discovery_envelope(message=message)
    return scope if problem is None else None


def public_architect_sol_downstream_problem(message):
    """Validate one non-cycle Architect request addressed to Red Team.

    Arguments:
      message = the decoded outbound message.

    Returns:
      A printable problem — the request must be an exact discovery
      with a real body and no NUL byte — or ``None`` when valid.
    """
    if daemon.sol_ticket_kind(message=message) != "discovery":
        return ("public Architect control output to Sol must be an exact "
                "discovery request")
    _severity, _scope, body, problem = daemon._sol_discovery_envelope(
        message=message)
    if problem is not None:
        return problem
    marker = daemon.placeholder_in(message=body)
    if marker is not None:
        return "Sol discovery body is only template placeholder '" + marker + "'"
    if "\x00" in message:
        return "Sol discovery contains a NUL byte"
    return None


def _body_architect_admission(body):
    """Return the exact admission token on the first body line.

    A public Architect outcome must be mechanically tied to the request that
    occupied the finite-watch slot.  Looking for a token later in prose would
    allow an unrelated output to consume that slot, so the binding line is
    required first and duplicates are refused.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_ADMISSION_HEADER)
        + r"(\d+-to-fable\.md@[0-9a-f]{64})\r?\n", body)
    admission_like = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*admission[ \t]*:")
    if match is None:
        if daemon.re.search(admission_like, body) is not None:
            return None, "malformed MAILBOX-ADMISSION line"
        return None, "missing first-body-line MAILBOX-ADMISSION"
    if daemon.re.search(admission_like, body[match.end():]) is not None:
        return None, "duplicate MAILBOX-ADMISSION line"
    try:
        daemon.split_architect_admission_token(token=match.group(1))
    except daemon.TicketCycleStateError as exc:
        return None, str(exc)
    return match.group(1), None


def public_architect_sol_outcome_problem(message, expected_token):
    """Validate one exact, digest-bound public Architect-to-Sol outcome.

    Arguments:
      message        = the decoded outbound message.
      expected_token = the admission token the body's first line must
                       carry.

    Returns:
      A printable problem, or ``None`` when the outcome is a valid
      discovery bound to the expected admission.
    """
    problem = daemon.public_architect_sol_downstream_problem(message=message)
    if problem is not None:
        return problem
    _severity, _scope, body, _problem = daemon._sol_discovery_envelope(
        message=message)
    if body.startswith("\r\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]
    else:
        return "public Architect Sol outcome requires one header/body gap"
    returned_token, admission_problem = daemon._body_architect_admission(body=body)
    if admission_problem is not None:
        return admission_problem
    if returned_token != expected_token:
        return "MAILBOX-ADMISSION does not bind this public request"
    return None


def _public_architect_no_ticket_receipt(message):
    """Parse one explicit no-ticket result from a public Architect turn.

    Arguments:
      message = the decoded receipt.

    Returns:
      ``(token, None)`` with the receipt's admission token, or
      ``(None, problem)`` for missing or duplicated headers, a NUL
      byte, a placeholder-only body, or an invalid token.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_RETURN_HEADER)
        + daemon.re.escape(daemon.PUBLIC_ARCHITECT_NO_TICKET_RETURN) + r"\r?\n"
        + daemon.re.escape(daemon.MAILBOX_ADMISSION_HEADER)
        + r"(\d+-to-fable\.md@[0-9a-f]{64})\r?\n"
        + daemon.re.escape(daemon.MAILBOX_DECISION_HEADER)
        + daemon.re.escape(daemon.PUBLIC_ARCHITECT_NO_TICKET_DECISION)
        + r"(?:\r?\n\r?\n(?P<body>[\s\S]*))?\r?\n?\Z",
        message)
    if match is None:
        return None, (
            "no-ticket receipt needs exact MAILBOX-RETURN, "
            "MAILBOX-ADMISSION, and MAILBOX-DECISION headers")
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*"
        r"(?:return|admission|decision)[ \t]*:")
    body = match.group("body") or ""
    if daemon.re.search(reserved, body) is not None:
        return None, "duplicate no-ticket receipt header"
    if "\x00" in message:
        return None, "no-ticket receipt contains a NUL byte"
    marker = daemon.placeholder_in(message=body)
    if marker is not None:
        return None, (
            "no-ticket receipt body is only template placeholder '"
            + marker + "'")
    try:
        daemon.split_architect_admission_token(token=match.group(1))
    except daemon.TicketCycleStateError as exc:
        return None, str(exc)
    return match.group(1), None


def public_architect_no_ticket_problem(message, expected_token):
    """Return why a public no-ticket receipt is invalid, or ``None``.

    Arguments:
      message        = the decoded receipt.
      expected_token = the admission token it must carry.

    Returns:
      A printable problem, or ``None`` for a valid receipt bound to
      the expected admission.
    """
    returned_token, problem = daemon._public_architect_no_ticket_receipt(
        message=message)
    if problem is not None:
        return problem
    if returned_token != expected_token:
        return "MAILBOX-ADMISSION does not bind this public request"
    return None


def _ticket_flow_envelope(message):
    """Parse one Architect/Implementer exchange inside a ticket cycle.

    Arguments:
      message = the decoded mailbox message.

    Returns:
      ``(cycle_id, mode, body, problem)``. On success the body is the
      text after the three flow headers; on failure the original
      message is returned as the body with a printable problem.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_FLOW_HEADER) + r"ticket\r?\n"
        + daemon.re.escape(daemon.MAILBOX_CYCLE_HEADER)
        + r"(" + daemon.CYCLE_ID_RE.pattern + r")\r?\n"
        + daemon.re.escape(daemon.MAILBOX_MODE_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.ARCHITECT_COMMIT_MODES))
        + r")\r?\n\r?\n",
        message)
    if match is None:
        return (None, None, message,
                "a ticket exchange needs exact MAILBOX-FLOW, MAILBOX-CYCLE, "
                "and MAILBOX-MODE headers")
    body = message[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:flow|cycle|mode)[ \t]*:")
    if daemon.re.search(reserved, body) is not None:
        return None, None, message, "duplicate ticket-cycle flow header"
    return match.group(1), match.group(2), body, None


def is_implementer_checkpoint_request(body):
    """Return whether a handoff asks the Architect for a pause decision.

    Arguments:
      body = the ticket-flow body.

    Returns:
      True when the first line is the timed, budget, or context
      checkpoint heading.
    """
    if not isinstance(body, str):
        return False
    lines = body.splitlines()
    return bool(lines) and lines[0] in {
        daemon.IMPLEMENTER_CHECKPOINT_HEADING,
        daemon.IMPLEMENTER_BUDGET_CHECKPOINT_HEADING,
        daemon.CONTEXT_HANDOFF_HEADING}


def is_implementer_budget_checkpoint(body):
    """Return whether a clean candidate exceeded the binding size limit.

    Arguments:
      body = the ticket-flow body.

    Returns:
      True when the first line is the budget-checkpoint heading.
    """
    return (isinstance(body, str) and body.splitlines()[:1]
            == [daemon.IMPLEMENTER_BUDGET_CHECKPOINT_HEADING])


def is_architect_budget_repair(body):
    """Return whether the Architect replaced an over-limit plan.

    Arguments:
      body = the ticket-flow body.

    Returns:
      True when the first line is the budget-repair heading.
    """
    return (isinstance(body, str) and body.splitlines()[:1]
            == [daemon.ARCHITECT_BUDGET_REPAIR_HEADING])


def is_implementer_time_checkpoint(body):
    """Return whether a timed checkpoint begins with its fixed state.

    Arguments:
      body = the ticket-flow body.

    Returns:
      True when the checkpoint's first nonempty field is the exact
      minutes-reached state line the hook dictates.
    """
    if not daemon.is_implementer_checkpoint_request(body):
        return False
    first_field = next(
        (line for line in body.splitlines()[1:] if line), "")
    return first_field == daemon.IMPLEMENTER_CHECKPOINT_CURRENT_STATE


def is_implementer_context_handoff(body):
    """Return whether the body begins with the context-handoff heading.

    Arguments:
      body = the ticket-flow body.

    Returns:
      True when the first line is the context-handoff heading.
    """
    return (isinstance(body, str)
            and body.splitlines()[:1] == [daemon.CONTEXT_HANDOFF_HEADING])


def _context_handoff_field(lines, name):
    """Read one exact field from a context handoff.

    Arguments:
      lines = the handoff's lines.
      name  = the bold field name to read.

    Returns:
      The field's value with backtick quoting removed.

    Raises:
      daemon.TicketCycleStateError: when the field is absent, empty,
        or repeated.
    """
    prefix = "- **" + name + ":** "
    values = [line[len(prefix):].strip("`") for line in lines
              if line.startswith(prefix)]
    if len(values) != 1 or not values[0]:
        raise daemon.TicketCycleStateError(
            "CONTEXT HANDOFF needs exactly one " + name + " field")
    return values[0]


def parse_context_handoff(body):
    """Read the required facts and ordered lists from one small handoff.

    Arguments:
      body = the ticket-flow body carrying the context handoff.

    Returns:
      Mapping with the exact bold fields and the ordered bullet
      sections a replacement Implementer needs.

    Raises:
      daemon.TicketCycleStateError: for a missing heading, a bad or
        repeated field, or an oversized record.
    """
    if not isinstance(body, str) or len(body.encode("utf-8")) > 32 * 1024:
        raise daemon.TicketCycleStateError("CONTEXT HANDOFF is missing or too large")
    lines = body.splitlines()
    if not lines or lines[0] != daemon.CONTEXT_HANDOFF_HEADING:
        raise daemon.TicketCycleStateError("CONTEXT HANDOFF heading is missing")
    record = {name: daemon._context_handoff_field(lines, name)
              for name in daemon.CONTEXT_HANDOFF_FIELDS}
    for name in ("Base commit", "Current worktree HEAD"):
        if daemon.FULL_COMMIT_RE.fullmatch(record[name]) is None:
            raise daemon.TicketCycleStateError(name + " must be one full Git commit")
    if record["Candidate created"] not in {"yes", "no"}:
        raise daemon.TicketCycleStateError("Candidate created must be yes or no")
    headings = ["#### " + name for name in daemon.CONTEXT_HANDOFF_SECTIONS]
    if any(lines.count(heading) != 1 for heading in headings):
        raise daemon.TicketCycleStateError(
            "CONTEXT HANDOFF needs every required list section once")
    positions = [lines.index(heading) for heading in headings]
    if positions != sorted(positions):
        raise daemon.TicketCycleStateError("CONTEXT HANDOFF sections are out of order")
    sections = {}
    for index, name in enumerate(daemon.CONTEXT_HANDOFF_SECTIONS):
        start = positions[index] + 1
        end = positions[index + 1] if index + 1 < len(positions) else len(lines)
        rows = [line for line in lines[start:end] if line.strip()]
        values = [line[2:].strip() for line in rows
                  if line.startswith("- ")]
        if (len(values) != len(rows) or not values
                or any(value in {"", "...", "[...]"} for value in values)):
            raise daemon.TicketCycleStateError(
                name + " must contain concrete bullets or '- none'")
        sections[name] = values
    record["sections"] = sections
    return record


def context_handoff_problem(message, expected_cycle=None,
                            expected_mode=None):
    """Validate one context record against the current Implementer tree.

    The record must name its exact cycle, the cycle's base commit,
    and the current Implementer HEAD; its uncommitted-changes section
    must agree with the real tree; and a claimed candidate must be a
    clean commit that moved off the base.

    Arguments:
      message        = the decoded handoff message.
      expected_cycle = the required cycle, or ``None``.
      expected_mode  = the required mode, or ``None``.

    Returns:
      A printable problem, or ``None`` when the record matches
      reality.
    """
    cycle_id, mode, body, problem = daemon._ticket_flow_envelope(message=message)
    if problem is not None:
        return problem
    if expected_cycle is not None and cycle_id != expected_cycle:
        return "CONTEXT HANDOFF changed MAILBOX-CYCLE"
    if expected_mode is not None and mode != expected_mode:
        return "CONTEXT HANDOFF changed MAILBOX-MODE"
    try:
        record = daemon.parse_context_handoff(body=body)
    except daemon.TicketCycleStateError as exc:
        return str(exc)
    base = cycle_id.rsplit("@", 1)[1]
    if record["Ticket and cycle"] != cycle_id:
        return "CONTEXT HANDOFF does not name its exact ticket and cycle"
    if record["Base commit"] != base:
        return "CONTEXT HANDOFF does not name the cycle base commit"
    try:
        head = daemon.worktree_head(worktree=daemon.AGENT_CWD["opus"])
        dirty = bool(daemon._clean_worktree_status(worktree=daemon.AGENT_CWD["opus"]))
    except (OSError, daemon.PrimaryWorktreeError, daemon.TicketCycleStateError) as exc:
        return "cannot verify CONTEXT HANDOFF worktree: " + str(exc)
    if record["Current worktree HEAD"] != head:
        return "CONTEXT HANDOFF does not name current Implementer HEAD"
    uncommitted = record["sections"]["Uncommitted changes"]
    if dirty == (uncommitted == ["none"]):
        return "CONTEXT HANDOFF disagrees with current uncommitted changes"
    if (record["Candidate created"] == "yes"
            and (dirty or head == base)):
        return "CONTEXT HANDOFF candidate must be a clean changed commit"
    return None


def matching_new_context_handoff(cycle_id, mode, before_inodes):
    """Find one fresh exact context record written by the Implementer.

    Arguments:
      cycle_id      = the ticket cycle.
      mode          = the ticket mode.
      before_inodes = the pre-turn message-inode snapshot.

    Returns:
      ``(path, offending, problem)`` as in the other matchers.
    """
    matches = []
    invalid = []
    problems = []
    for path in daemon.new_route_paths(
            pattern="*-to-fable.md",
            before_inodes=before_inodes):
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError) as exc:
            invalid.append(path)
            problems.append(daemon.os.path.basename(path) + ": " + str(exc))
            continue
        _found_cycle, _found_mode, body, envelope_problem = (
            daemon._ticket_flow_envelope(message=message))
        if (envelope_problem is not None
                or not daemon.is_implementer_context_handoff(body)):
            continue
        problem = daemon.context_handoff_problem(
            message=message, expected_cycle=cycle_id, expected_mode=mode)
        if daemon.os.path.dirname(path) != daemon.MAILBOX:
            problem = "CONTEXT HANDOFF was not published in mailbox root"
        if problem is None:
            matches.append(path)
        else:
            invalid.append(path)
            problems.append(daemon.os.path.basename(path) + ": " + problem)
    if problems:
        return None, list(dict.fromkeys(invalid + matches)), "; ".join(problems)
    if len(matches) > 1:
        return None, matches, "expected at most one CONTEXT HANDOFF"
    return (matches[0] if matches else None), [], None


def latest_context_handoff_path(cycle_id, mode):
    """Return the newest valid same-cycle record for a replacement turn.

    Arguments:
      cycle_id = the ticket cycle.
      mode     = the ticket mode.

    Returns:
      The newest matching record's path, or ``None`` when the cycle
      has none.

    Raises:
      daemon.TicketCycleStateError: when the newest record no longer
        matches the live worktree.
    """
    matches = []
    for path in daemon.glob.glob(
            daemon.os.path.join(daemon.MAILBOX, "**", "*-to-fable.md"),
            recursive=True):
        if daemon.regular_inode(path=path) is None:
            continue
        try:
            message = daemon.read_cycle_message(path=path)
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        found_cycle, found_mode, body, problem = daemon._ticket_flow_envelope(
            message=message)
        if (problem is None and found_cycle == cycle_id
                and found_mode == mode
                and daemon.is_implementer_context_handoff(body)):
            matches.append(path)
    if not matches:
        return None
    path = max(matches, key=lambda item: (
        daemon.sequence_in_name(daemon.os.path.basename(item)) or -1))
    message = daemon.read_cycle_message(path=path)
    problem = daemon.context_handoff_problem(
        message=message, expected_cycle=cycle_id, expected_mode=mode)
    if problem is not None:
        raise daemon.TicketCycleStateError(
            "saved replacement CONTEXT HANDOFF is stale: " + problem)
    return path


def replacement_context_notice(path):
    """Tell a fresh Implementer where to read the prior exact record.

    Arguments:
      path = the saved context record's path.

    Returns:
      The notice text prepended to the replacement's dispatch.
    """
    return (
        "REPLACEMENT IMPLEMENTER CONTEXT\n"
        "Read the exact prior Implementer record at:\n" + path + "\n"
        "Verify it against the repository before editing. It is not a "
        "daemon-written summary. Do not repeat an approach listed under "
        "Do not revisit unless the Architect explicitly reopened it.\n\n")


def checkpoint_handoff_problem(message):
    """Validate the timed or context checkpoint sent to the Architect.

    Arguments:
      message = the decoded checkpoint message.

    Returns:
      A printable problem, or ``None`` when the checkpoint carries
      its required heading, fields, and sections.
    """
    _, _, body, problem = daemon._ticket_flow_envelope(message=message)
    if problem is not None or not daemon.is_implementer_checkpoint_request(body):
        return ("the 90-minute hook or context hook fired without its "
                "checkpoint handoff")
    if daemon.is_implementer_context_handoff(body):
        return daemon.context_handoff_problem(message=message)
    if daemon.is_implementer_budget_checkpoint(body):
        candidate_rows = daemon.IMPLEMENTER_CANDIDATE_LINE_RE.findall(body)
        result_rows = [
            line for line in body.splitlines()
            if line.startswith("- **Character-change result:**")]
        if daemon.MAX_CHARACTERS <= 0:
            return "budget checkpoint requires a positive ticket limit"
        if len(candidate_rows) != 1:
            return "budget checkpoint requires one exact Candidate commit row"
        if (len(result_rows) != 1
                or not result_rows[0].startswith(
                    "- **Character-change result:** over limit")):
            return "budget checkpoint requires one over limit result row"
        return None
    current_state_rows = [
        line for line in body.splitlines()
        if line.startswith("- **Current state:**")]
    if (not daemon.is_implementer_time_checkpoint(body)
            or len(current_state_rows) != 1
            or current_state_rows[0]
            != daemon.IMPLEMENTER_CHECKPOINT_CURRENT_STATE):
        return "the checkpoint needs its exact 90-minute Current state"
    return None


def _ticket_architect_admission(message):
    """Return an exact public-request admission carried by an Opus flow.

    Ordinary role-to-role ticket messages carry no admission line.  The
    first Implementer handoff created by a public Architect turn carries the
    exact request basename and SHA-256 so a crash or reordering cannot pair
    the handoff with another public request.
    """
    _cycle_id, _mode, body, problem = daemon._ticket_flow_envelope(message=message)
    if problem is not None:
        return None, None, problem
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_ADMISSION_HEADER)
        + r"(\d+-to-fable\.md)@([0-9a-f]{64})\r?\n",
        body)
    admission_like = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*admission[ \t]*:")
    if match is None:
        if daemon.re.search(admission_like, body) is not None:
            return None, None, "malformed MAILBOX-ADMISSION line"
        return None, None, None
    if daemon.re.search(admission_like, body[match.end():]) is not None:
        return None, None, "duplicate MAILBOX-ADMISSION line"
    return match.group(1), match.group(2), None


def _redteam_closure_envelope(message):
    """Parse one post-commit Red Team request.

    Arguments:
      message = the decoded Sol message.

    Returns:
      ``(cycle_id, commit, body, problem)``. A message of another
      kind returns its remainder with no problem; a malformed closure
      returns the remainder with a printable problem.
    """
    remainder = daemon.sol_ticket_body_after_kind(message=message)
    if daemon.sol_ticket_kind(message=message) != "closure":
        return None, None, remainder, None
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_CYCLE_HEADER)
        + r"(" + daemon.CYCLE_ID_RE.pattern + r")\r?\n"
        + daemon.re.escape(daemon.MAILBOX_COMMIT_HEADER)
        + r"([0-9a-f]{40})\r?\n\r?\n",
        remainder)
    if match is None:
        return (None, None, remainder,
                "a Red Team closure must name exactly one ticket cycle and "
                "one daemon-recorded local landing L on its second and third "
                "physical lines")
    body = remainder[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:cycle|commit|return|result)"
        r"[ \t]*:")
    if daemon.re.search(reserved, body) is not None:
        return None, None, remainder, "duplicate Red Team review header"
    return match.group(1), match.group(2), body, None


def _redteam_control_plane_envelope(message):
    """Parse one mandatory pre-landing review of exact candidate C.

    Arguments:
      message = the decoded Sol message.

    Returns:
      ``(cycle_id, candidate, body, problem)``. A message of another
      kind returns its remainder with no problem; a malformed
      control-plane review returns the remainder with a printable
      problem.
    """
    remainder = daemon.sol_ticket_body_after_kind(message=message)
    if daemon.sol_ticket_kind(message=message) != "control-plane":
        return None, None, remainder, None
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_CYCLE_HEADER)
        + r"(" + daemon.CYCLE_ID_RE.pattern + r")\r?\n"
        + daemon.re.escape(daemon.MAILBOX_CANDIDATE_HEADER)
        + r"([0-9a-f]{40})\r?\n\r?\n",
        remainder)
    if match is None:
        return (None, None, remainder,
                "a control-plane review must name one exact ticket cycle "
                "and full candidate C")
    body = remainder[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:cycle|candidate|return|result)"
        r"[ \t]*:")
    if daemon.re.search(reserved, body) is not None:
        return None, None, remainder, "duplicate control-plane review header"
    return match.group(1), match.group(2), body, None


def _control_plane_review_receipt(message):
    """Parse one exact pre-landing Red Team decision addressed to D0.

    Arguments:
      message = the decoded return.

    Returns:
      ``(cycle_id, candidate, result, body, problem)``; a malformed
      or duplicate-header return carries a printable problem.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_RETURN_HEADER)
        + r"redteam-control-plane\r?\n"
        + daemon.re.escape(daemon.MAILBOX_CYCLE_HEADER)
        + r"(" + daemon.CYCLE_ID_RE.pattern + r")\r?\n"
        + daemon.re.escape(daemon.MAILBOX_CANDIDATE_HEADER)
        + r"([0-9a-f]{40})\r?\n"
        + daemon.re.escape(daemon.MAILBOX_RESULT_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.CONTROL_PLANE_REVIEW_RESULTS))
        + r")\r?\n\r?\n",
        message)
    if match is None:
        return (None, None, None, message,
                "a control-plane return needs exact cycle, full candidate, "
                "and ACCEPT-CONTROL-PLANE or REJECT-CONTROL-PLANE")
    body = message[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:cycle|candidate|return|result)"
        r"[ \t]*:")
    if daemon.re.search(reserved, body) is not None:
        return None, None, None, message, "duplicate control-plane receipt"
    return match.group(1), match.group(2), match.group(3), body, None


def redteam_closure_problem(message):
    """Return a closure-envelope problem, or ``None``.

    Arguments:
      message = the decoded Sol closure message.

    Returns:
      The printable problem, or ``None`` for a valid envelope.
    """
    return daemon._redteam_closure_envelope(message=message)[3]


def redteam_closure_ticket(message):
    """Return the one reviewed ticket-cycle identifier, when valid.

    Arguments:
      message = the decoded Sol closure message.

    Returns:
      The cycle identifier, or ``None`` for a malformed envelope.
    """
    ticket, _, _, problem = daemon._redteam_closure_envelope(message=message)
    return ticket if problem is None else None


def redteam_closure_commit(message):
    """Return the one full daemon-recorded landing L, when valid.

    Arguments:
      message = the decoded Sol closure message.

    Returns:
      The landing commit, or ``None`` for a malformed envelope.
    """
    _, commit, _, problem = daemon._redteam_closure_envelope(message=message)
    return commit if problem is None else None


def _redteam_review_receipt(message):
    """Parse the Red Team's correlated return to the Architect.

    The result vocabulary belongs to the advisory role: ``NO CHANGE`` means
    the accepted fix still stands, and ``REOPEN`` supplies evidence for later
    Architect assessment. Architect ``GO``/``NO-GO`` are deliberately absent.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_RETURN_HEADER)
        + r"redteam-closure\r?\n"
        + daemon.re.escape(daemon.MAILBOX_CYCLE_HEADER)
        + r"(" + daemon.CYCLE_ID_RE.pattern + r")\r?\n"
        + daemon.re.escape(daemon.MAILBOX_COMMIT_HEADER)
        + r"([0-9a-f]{40})\r?\n"
        + daemon.re.escape(daemon.MAILBOX_RESULT_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.REDTEAM_REVIEW_RESULTS))
        + r")\r?\n\r?\n",
        message)
    if match is None:
        return (None, None, None, message,
                "a Red Team return needs exact review ticket, commit, and "
                "NO CHANGE or REOPEN headers")
    body = message[match.end():]
    reserved = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:cycle|commit|return|result)"
        r"[ \t]*:")
    if daemon.re.search(reserved, body) is not None:
        return (None, None, None, message,
                "duplicate Red Team review receipt header")
    return match.group(1), match.group(2), match.group(3), body, None


def _architect_go_request(message):
    """Parse one decision-only GO request bound to the audited candidate.

    Arguments:
      message = the decoded daemon request.

    Returns:
      ``(cycle_id, candidate, mode, problem)``. The request is header
      lines only; any free-form remainder is a problem, so a GO can
      never smuggle extra work.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_RETURN_HEADER)
        + r"architect-go\r?\n"
        + daemon.re.escape(daemon.MAILBOX_CYCLE_HEADER)
        + r"(" + daemon.CYCLE_ID_RE.pattern + r")\r?\n"
        + daemon.re.escape(daemon.MAILBOX_CANDIDATE_HEADER)
        + r"([0-9a-f]{40})\r?\n"
        + daemon.re.escape(daemon.MAILBOX_MODE_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.ARCHITECT_COMMIT_MODES))
        + r")\r?\n"
        + daemon.re.escape(daemon.MAILBOX_DECISION_HEADER)
        + r"GO(?:\r?\n|\Z)",
        message)
    if match is None:
        return (None, None, None,
                "an Architect GO request needs exact return, cycle, "
                "candidate, mode, and GO decision headers")
    remainder = message[match.end():]
    if remainder.strip():
        return (None, None, None,
                "an Architect GO request may not carry free-form work")
    return match.group(1), match.group(2), match.group(3), None


def architect_go_request_payload(cycle_id, candidate_commit, mode):
    """Build the decision-only daemon request written after Architect GO.

    Arguments:
      cycle_id         = the ticket cycle.
      candidate_commit = the accepted candidate C.
      mode             = the ticket mode.

    Returns:
      The exact header-only payload text.

    Raises:
      ValueError: for an invalid cycle, candidate, or mode.
    """
    if not isinstance(cycle_id, str) or daemon.CYCLE_ID_RE.fullmatch(cycle_id) is None:
        raise ValueError("invalid ticket cycle: " + repr(cycle_id))
    if (not isinstance(candidate_commit, str)
            or daemon.FULL_COMMIT_RE.fullmatch(candidate_commit) is None):
        raise ValueError(
            "invalid Implementer candidate: " + repr(candidate_commit))
    if mode not in daemon.ARCHITECT_COMMIT_MODES:
        raise ValueError("invalid Architect GO mode: " + repr(mode))
    return (daemon.MAILBOX_RETURN_HEADER + "architect-go\n"
            + daemon.MAILBOX_CYCLE_HEADER + cycle_id + "\n"
            + daemon.MAILBOX_CANDIDATE_HEADER + candidate_commit + "\n"
            + daemon.MAILBOX_MODE_HEADER + mode + "\n"
            + daemon.MAILBOX_DECISION_HEADER + "GO\n")


def backlog_close_request_payload(cycle_id, candidate_commit, mode):
    """Ask the Architect to close the backlog and repeat its exact GO.

    Arguments:
      cycle_id         = the ticket cycle awaiting bookkeeping.
      candidate_commit = the already-accepted candidate C.
      mode             = the ticket mode.

    Returns:
      The recovery request text; its body says explicitly that the
      audit is done and only backlog closing plus one fresh GO
      remain.

    Raises:
      ValueError: for invalid identifiers, checked through the GO
        payload builder.
    """
    daemon.architect_go_request_payload(cycle_id, candidate_commit, mode)
    return (
        daemon.MAILBOX_FLOW_HEADER + "ticket\n"
        + daemon.MAILBOX_CYCLE_HEADER + cycle_id + "\n"
        + daemon.MAILBOX_MODE_HEADER + mode + "\n\n"
        + daemon.BACKLOG_CLOSE_REQUIRED_HEADER + candidate_commit + "\n\n"
        + "- **Candidate commit:** " + candidate_commit + "\n\n"
        + "Your completed audit already accepted this exact candidate. Do "
          "not repeat the audit or rerun the Implementer. Close and seal "
          "this ticket in backlog.md, then send one fresh exact GO for the "
          "same C. This is bookkeeping recovery only.\n")


def _architect_notes_go_request(message):
    """Parse one body-free permanent-note commit request bound to B and P.

    Arguments:
      message = the decoded daemon request.

    Returns:
      ``(base_commit, notes_commit, problem)``; B and P must differ
      and no body may follow the headers.
    """
    match = daemon.re.fullmatch(
        daemon.re.escape(daemon.MAILBOX_RETURN_HEADER) + r"architect-notes-go\r?\n"
        + daemon.re.escape(daemon.MAILBOX_BASE_HEADER) + r"([0-9a-f]{40})\r?\n"
        + daemon.re.escape(daemon.MAILBOX_NOTES_COMMIT_HEADER)
        + r"([0-9a-f]{40})\r?\n"
        + daemon.re.escape(daemon.MAILBOX_DECISION_HEADER) + r"GO(?:\r?\n)?",
        message)
    if match is None:
        return (None, None,
                "an Architect notes GO needs exact return, base, notes "
                "commit, and GO headers with no body")
    if match.group(1) == match.group(2):
        return (None, None,
                "an Architect notes GO must name a new commit")
    return match.group(1), match.group(2), None


def architect_notes_go_request_payload(base_commit, notes_commit):
    """Build the exact parent-daemon request for one permanent-note commit.

    Arguments:
      base_commit  = B, the main baseline.
      notes_commit = P, the note-only commit to land.

    Returns:
      The exact header-only payload text.

    Raises:
      ValueError: for invalid or equal commits.
    """
    if (not isinstance(base_commit, str)
            or daemon.FULL_COMMIT_RE.fullmatch(base_commit) is None
            or not isinstance(notes_commit, str)
            or daemon.FULL_COMMIT_RE.fullmatch(notes_commit) is None
            or base_commit == notes_commit):
        raise ValueError("invalid permanent-note commit request")
    return (daemon.MAILBOX_RETURN_HEADER + "architect-notes-go\n"
            + daemon.MAILBOX_BASE_HEADER + base_commit + "\n"
            + daemon.MAILBOX_NOTES_COMMIT_HEADER + notes_commit + "\n"
            + daemon.MAILBOX_DECISION_HEADER + "GO\n")


def _architect_notes_admin_envelope(message):
    """Return the plain-language body of one dedicated note-update turn.

    Arguments:
      message = the decoded admin request.

    Returns:
      ``(body, problem)``: the update summary after the admin header,
      or a printable problem for a wrong header, empty body, or
      duplicate header line.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.MAILBOX_ADMIN_HEADER)
        + r"permanent-notes\r?\n\r?\n", message)
    if match is None:
        return None, "not a permanent-notes admin request"
    body = message[match.end():]
    if not body.strip():
        return None, "permanent-notes admin request needs an update summary"
    if daemon.re.search(
            r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*admin[ \t]*:", body):
        return None, "duplicate MAILBOX-ADMIN header"
    return body, None


def architect_notes_admin_payload(text):
    """Build the exact Architect self-route for a durable note update.

    Arguments:
      text = the plain-language update summary.

    Returns:
      The newline-terminated admin payload.

    Raises:
      ValueError: for an empty summary or one that repeats the admin
        header.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("permanent-notes admin summary must be nonempty")
    if daemon.re.search(
            r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*admin[ \t]*:", text):
        raise ValueError("permanent-notes admin summary repeats its header")
    payload = daemon.MAILBOX_ADMIN_HEADER + "permanent-notes\n\n" + text
    if not payload.endswith("\n"):
        payload += "\n"
    return payload


def is_architect_notes_admin_message(message):
    """Return whether ``message`` is one valid dedicated note-update turn.

    Arguments:
      message = the decoded message.

    Returns:
      True for a well-formed permanent-notes admin request.
    """
    _body, problem = daemon._architect_notes_admin_envelope(message=message)
    return problem is None


def architect_notes_admin_journal_path(request_name, relay_dir=None):
    """Return the durable post-child journal for one exact admin request.

    Arguments:
      request_name = the admin request's mailbox filename.
      relay_dir    = the relay folder, or ``None`` for the default.

    Returns:
      The JSON journal path.

    Raises:
      daemon.TicketCycleStateError: for a name that is not a pending
        message.
    """
    if daemon.PENDING_MESSAGE_RE.fullmatch(request_name) is None:
        raise daemon.TicketCycleStateError("invalid admin request journal name")
    directory = daemon.RELAY_DIR if relay_dir is None else relay_dir
    return daemon.os.path.join(
        directory, ".pending-notes-admin-" + request_name + ".json")


def write_architect_notes_admin_journal(request_name, request_message,
                                        base_commit, phase,
                                        notes_commit=None,
                                        receipt_sha256=None):
    """Atomically bind one admin request to its validated recovery phase.

    Arguments:
      request_name    = the admin request's mailbox filename.
      request_message = its exact bytes, bound by digest.
      base_commit     = B, the main baseline the turn started from.
      phase           = ``"started"``, ``"validated-noop"``, or
                        ``"validated-commit"``.
      notes_commit    = P, required in the validated-commit phase.
      receipt_sha256  = the GO receipt's digest, required in the
                        validated-commit phase.

    Raises:
      daemon.TicketCycleStateError: for an invalid phase, authority,
        or commit binding.
    """
    if phase not in {"started", "validated-noop", "validated-commit"}:
        raise daemon.TicketCycleStateError("invalid note-admin journal phase")
    if (daemon.FULL_COMMIT_RE.fullmatch(str(base_commit)) is None
            or not daemon.is_architect_notes_admin_message(
                message=request_message)):
        raise daemon.TicketCycleStateError("invalid note-admin journal authority")
    if phase == "validated-commit":
        if (daemon.FULL_COMMIT_RE.fullmatch(str(notes_commit)) is None
                or notes_commit == base_commit
                or daemon.re.fullmatch(r"[0-9a-f]{64}",
                                str(receipt_sha256)) is None):
            raise daemon.TicketCycleStateError(
                "validated note-admin commit journal needs exact P/receipt")
    elif notes_commit is not None or receipt_sha256 is not None:
        raise daemon.TicketCycleStateError(
            "non-commit note-admin journal cannot name P/receipt")
    payload = {
        "schema": daemon.ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA,
        "request": request_name,
        "request_sha256": daemon.hashlib.sha256(
            request_message.encode("utf-8")).hexdigest(),
        "base": base_commit,
        "phase": phase,
        "notes_commit": notes_commit,
        "receipt_sha256": receipt_sha256,
    }
    daemon.os.makedirs(daemon.RELAY_DIR, exist_ok=True)
    path = daemon.architect_notes_admin_journal_path(request_name=request_name)
    descriptor, temporary = daemon.tempfile.mkstemp(
        prefix=".pending-notes-admin-", dir=daemon.RELAY_DIR)
    try:
        daemon.os.fchmod(descriptor, 0o600)
        with daemon.os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) \
                as stream:
            descriptor = -1
            daemon.json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            daemon.os.fsync(stream.fileno())
        daemon.os.replace(temporary, path)
        daemon.fsync_directory(directory=daemon.RELAY_DIR)
    finally:
        if descriptor >= 0:
            daemon.os.close(descriptor)
        try:
            daemon.os.remove(temporary)
        except FileNotFoundError:
            pass
    return path


def read_architect_notes_admin_journal(request_name, request_message,
                                       relay_dir=None):
    """Read one exact journal and rebind it to the inflight request bytes.

    Arguments:
      request_name    = the admin request's mailbox filename.
      request_message = the inflight request bytes the journal must
                        still bind by digest.
      relay_dir       = the relay folder, or ``None`` for the
                        default.

    Returns:
      The journal mapping, or ``None`` when no journal exists.

    Raises:
      daemon.TicketCycleStateError: for a malformed journal or one
        bound to different request bytes.
    """
    path = daemon.architect_notes_admin_journal_path(
        request_name=request_name, relay_dir=relay_dir)
    try:
        raw = daemon.stable_regular_bytes(
            path=path,
            maximum_bytes=daemon.MAX_ARCHITECT_NOTES_ADMIN_JOURNAL_BYTES,
            label="permanent-note admin journal")
        payload = daemon.json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=daemon._duplicate_key_refusal)
    except (OSError, ValueError, UnicodeDecodeError,
            daemon.json.JSONDecodeError) as exc:
        raise daemon.TicketCycleStateError(
            "cannot verify permanent-note admin journal: " + str(exc)) \
            from exc
    if (not isinstance(payload, dict)
            or set(payload) != {
                "schema", "request", "request_sha256", "base", "phase",
                "notes_commit", "receipt_sha256"}
            or payload["schema"] != daemon.ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA
            or payload["request"] != request_name
            or payload["request_sha256"] != daemon.hashlib.sha256(
                request_message.encode("utf-8")).hexdigest()
            or daemon.FULL_COMMIT_RE.fullmatch(str(payload["base"])) is None
            or payload["phase"] not in {
                "started", "validated-noop", "validated-commit"}):
        raise daemon.TicketCycleStateError(
            "permanent-note admin journal has invalid fields")
    if payload["phase"] == "validated-commit":
        if (daemon.FULL_COMMIT_RE.fullmatch(
                str(payload["notes_commit"])) is None
                or payload["notes_commit"] == payload["base"]
                or daemon.re.fullmatch(r"[0-9a-f]{64}",
                                str(payload["receipt_sha256"])) is None):
            raise daemon.TicketCycleStateError(
                "permanent-note admin commit journal is incomplete")
    elif (payload["notes_commit"] is not None
          or payload["receipt_sha256"] is not None):
        raise daemon.TicketCycleStateError(
            "permanent-note admin non-commit journal names a receipt")
    return payload


def remove_architect_notes_admin_journal(request_name):
    """Remove only the journal for a successfully archived admin request.

    Arguments:
      request_name = the admin request's mailbox filename; a missing
                     journal is not an error.
    """
    path = daemon.architect_notes_admin_journal_path(request_name=request_name)
    try:
        daemon.os.remove(path)
    except FileNotFoundError:
        return
    daemon.fsync_directory(directory=daemon.RELAY_DIR)


def _architect_notes_admin_request_path(request_name):
    """Find the one durable admin request bound to a recovery journal.

    Arguments:
      request_name = the admin request's mailbox filename.

    Returns:
      The single path holding the request across the root, inflight,
      failed, and done states.

    Raises:
      daemon.TicketCycleStateError: when zero or several states hold
        the name, which would make recovery ambiguous.
    """
    matches = []
    for directory in (daemon.MAILBOX, daemon.os.path.join(daemon.MAILBOX, "inflight"),
                      daemon.os.path.join(daemon.MAILBOX, "failed"), daemon.DONE):
        path = daemon.os.path.join(directory, request_name)
        if daemon.regular_inode(path=path) is not None:
            matches.append(path)
    if len(matches) != 1:
        raise daemon.TicketCycleStateError(
            "permanent-note admin journal needs exactly one saved request; "
            "found " + str(len(matches)) + " for " + request_name)
    return matches[0]


def _validated_commit_admin_journals(base_commit, notes_commit,
                                     receipt_sha256):
    """Return exact validated-commit journals bound to one B/P receipt.

    Arguments:
      base_commit    = B the journal must record.
      notes_commit   = P the journal must record.
      receipt_sha256 = the GO receipt digest it must record.

    Returns:
      List of ``(request_name, request_path, journal_path)`` for
      every matching validated-commit journal.

    Raises:
      daemon.TicketCycleStateError: for a malformed journal name, an
        ambiguous saved request, or an unreadable request.
    """
    prefix = ".pending-notes-admin-"
    suffix = ".json"
    matches = []
    pattern = daemon.os.path.join(daemon.RELAY_DIR, prefix + "*" + suffix)
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
                label="journaled permanent-note admin").decode(
                    "utf-8", errors="strict")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise daemon.TicketCycleStateError(
                "cannot verify saved permanent-note admin request: "
                + str(exc)) from exc
        journal = daemon.read_architect_notes_admin_journal(
            request_name=request_name, request_message=request_message)
        if (journal["phase"] == "validated-commit"
                and journal["base"] == base_commit
                and journal["notes_commit"] == notes_commit
                and journal["receipt_sha256"] == receipt_sha256):
            matches.append((request_name, request_path, journal_path))
    return matches


def retire_validated_commit_admin_journal(base_commit, notes_commit,
                                          receipt_sha256):
    """Remove one journal only after its exact P receipt is consumed.

    Arguments:
      base_commit    = B the journal must record.
      notes_commit   = P the journal must record.
      receipt_sha256 = the consumed GO receipt's digest.

    Returns:
      True when one journal was retired; False when none matched.

    Raises:
      daemon.TicketCycleStateError: for several matching journals or
        a request not yet archived in done.
    """
    matches = daemon._validated_commit_admin_journals(
        base_commit=base_commit, notes_commit=notes_commit,
        receipt_sha256=receipt_sha256)
    if len(matches) > 1:
        raise daemon.TicketCycleStateError(
            "more than one validated admin journal names the same B/P "
            "receipt")
    if not matches:
        return False
    request_name, request_path, _journal_path = matches[0]
    if daemon.os.path.dirname(request_path) != daemon.DONE:
        raise daemon.TicketCycleStateError(
            "validated admin journal cannot retire before its request is "
            "archived")
    daemon.remove_architect_notes_admin_journal(request_name=request_name)
    return True
