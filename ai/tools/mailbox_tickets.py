"""Sol ticket payloads, admission tokens, timeout history, and the dispatch banner.

This file writes the daemon-owned request texts: the Sol Red Team
ticket bodies for discovery, closure, and control-plane review, the
saved form of a user's request to the Architect, and the banner a
starting watch prints. An admission token is the short saved proof
that one user request already holds a slot of a finite ``--cycle``
budget, so a restart cannot admit the same ticket twice. The timeout
history is a per-message record of every timeout kill, so a retried
message keeps the killed-after threshold it was promised. Docstrings
here call the Implementer's candidate commit C.

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
    "sol_ticket_body",
    "architect_request_scope",
    "mailbox_role_for_dispatch",
    "transport_ping_text",
    "sol_ticket_payload",
    "redteam_review_receipt_payload",
    "control_plane_review_receipt_payload",
    "architect_user_request_payload",
    "_architect_user_request_envelope",
    "architect_user_request_problem",
    "architect_user_request_severity",
    "architect_user_request_scope",
    "architect_user_request_body",
    "architect_admission_token",
    "split_architect_admission_token",
    "message_claims_architect_admission",
    "architect_admission_prompt",
    "valid_sol_transport",
    "fix_only_environment_active",
    "skip_redteam_environment_active",
    "resolve_discovery_severity",
    "sol_ticket_refusal",
    "inflight_lane_blockers",
    "blocker_message_name",
    "report_inflight_lane_block",
    "message_sequence",
    "sequence_in_name",
    "dispatch_currency",
    "timeout_history_path",
    "timeout_events",
    "valid_duration",
    "write_timeout_history",
    "exact_duration",
    "dispatch_banner",
    "report_ticket_character_limit",
    "report_discovery_severity",
    "placeholder_in",
)


def sol_ticket_body(message):
    """Return the human body after valid Sol envelope lines.

    Each Sol ticket kind saves its own machine-read header lines
    before the human text. When the envelope parses, the body after
    the headers is returned; when it does not, the remainder after
    the kind line is returned unchanged so a malformed message still
    shows its text.

    Arguments:
      message = the decoded mailbox message.

    Returns:
      The human-readable ticket body.
    """
    remainder = daemon.sol_ticket_body_after_kind(message=message)
    if daemon.sol_ticket_kind(message=message) == "closure":
        _, _, body, problem = daemon._redteam_closure_envelope(message=message)
        return remainder if problem is not None else body
    if daemon.sol_ticket_kind(message=message) == "control-plane":
        _, _, body, problem = daemon._redteam_control_plane_envelope(
            message=message)
        return remainder if problem is not None else body
    if daemon.sol_ticket_kind(message=message) != "discovery":
        return remainder
    _, _, body, problem = daemon._sol_discovery_envelope(message=message)
    return remainder if problem is not None else body


def architect_request_scope(text):
    """Classify only an explicit positive command at the start of user text.

    A quotation, a negation, leading prose, or a later mention remains
    bounded. This recognizer is used only while constructing the public
    Architect envelope; dispatch trusts the saved ``MAILBOX-SCOPE`` value.

    Arguments:
      text = the user's request text.

    Returns:
      ``"widespread"`` only for an explicit positive command at the
      start of the text, otherwise ``"bounded"``.
    """
    positive = (
        r"\A(?:please(?:,)?[ \t]+)?"
        r"(?:instruct[ \t]+the[ \t]+red[ \t]+team[ \t]+to[ \t]+)?"
        r"do[ \t]+a[ \t]+widespread[ \t]+search\b")
    return ("widespread" if daemon.re.search(positive, text, daemon.re.IGNORECASE)
            is not None else "bounded")


def mailbox_role_for_dispatch(agent, message=None):
    """Return the checksum-guard role for one exact dispatch route.

    Arguments:
      agent   = ``"fable"``, ``"opus"``, or ``"sol"``.
      message = unused; accepted so every dispatch call site can pass
                the message uniformly.

    Returns:
      ``"architect"``, ``"implementer"``, or ``"red-team"``.

    Raises:
      ValueError: for an unknown agent.
    """
    if agent == "fable":
        return "architect"
    if agent == "opus":
        return "implementer"
    if agent == "sol":
        return "red-team"
    raise ValueError("unknown mailbox agent: " + repr(agent))


def transport_ping_text(agent):
    """Return the one no-work transport payload reserved for ``--ping``.

    Arguments:
      agent = the pinged route name, embedded in the text.

    Returns:
      The exact ping text; the receiving model must reply with one
      PONG line and stop.
    """
    return (
        "RELAY CONFIRMATION PING for " + agent + ". This is a "
        "transport test only; no unit is assigned and no repository "
        "file may change. Reply by creating ONE new file,\n"
        "ai/notes/mailbox/<next-sequence>-to-user.md, whose entire body "
        "is one line:\n\n"
        "    PONG " + agent + " from <your model name>\n\n"
        "Then stop. (Files addressed -to-user are read by the human; "
        "the daemon never dispatches them.)\n")


def sol_ticket_payload(ticket_kind, text, discovery_severity=None,
                       discovery_scope=None, review_cycle=None,
                       review_commit=None):
    """Build the byte-stable persisted envelope for a Sol message.

    Byte-stable means the same inputs always produce identical bytes,
    so a saved message can be compared exactly later. A discovery
    ticket saves its severity and scope headers; a closure or
    control-plane review may save its cycle and commit identity;
    every other combination is refused.

    Arguments:
      ticket_kind        = one of the Sol ticket kinds, such as
                           ``"discovery"`` or ``"closure"``.
      text               = the human ticket text.
      discovery_severity = severity header, discovery tickets only;
                           ``None`` selects the default.
      discovery_scope    = scope header, discovery tickets only;
                           ``None`` selects the default.
      review_cycle       = cycle identity for a closure or
                           control-plane review.
      review_commit      = reviewed commit for the same reviews.

    Returns:
      The newline-terminated payload text.

    Raises:
      ValueError: for an option on the wrong ticket kind or a
        malformed cycle or commit.
    """
    if ticket_kind == "discovery":
        if review_cycle is not None or review_commit is not None:
            raise ValueError(
                "review identity is valid only for Red Team reviews")
        if discovery_severity is None:
            discovery_severity = daemon.DEFAULT_DISCOVERY_SEVERITY
        if discovery_severity not in daemon.DISCOVERY_SEVERITIES:
            raise ValueError("invalid discovery severity: "
                             + repr(discovery_severity))
        if discovery_scope is None:
            discovery_scope = daemon.DEFAULT_DISCOVERY_SCOPE
        if discovery_scope not in daemon.DISCOVERY_SCOPES:
            raise ValueError("invalid discovery scope: "
                             + repr(discovery_scope))
        payload = (daemon.SOL_TICKET_HEADER + ticket_kind + "\n"
                   + daemon.SOL_SEVERITY_HEADER + discovery_severity + "\n"
                   + daemon.SOL_SCOPE_HEADER + discovery_scope + "\n\n"
                   + text)
    else:
        if discovery_severity is not None:
            raise ValueError(
                "discovery severity is valid only for discovery tickets")
        if discovery_scope is not None:
            raise ValueError(
                "discovery scope is valid only for discovery tickets")
        if ticket_kind in {"closure", "control-plane"} and (
                review_cycle is not None or review_commit is not None):
            if (not isinstance(review_cycle, str)
                    or daemon.CYCLE_ID_RE.fullmatch(review_cycle)
                    is None):
                raise ValueError("invalid Red Team review cycle: "
                                 + repr(review_cycle))
            if (not isinstance(review_commit, str)
                    or daemon.FULL_COMMIT_RE.fullmatch(review_commit) is None):
                raise ValueError("invalid Red Team review commit: "
                                 + repr(review_commit))
            identity_header = (daemon.MAILBOX_COMMIT_HEADER
                               if ticket_kind == "closure"
                               else daemon.MAILBOX_CANDIDATE_HEADER)
            payload = (daemon.SOL_TICKET_HEADER + ticket_kind + "\n"
                       + daemon.MAILBOX_CYCLE_HEADER + review_cycle + "\n"
                       + identity_header + review_commit + "\n\n" + text)
        else:
            if review_cycle is not None or review_commit is not None:
                raise ValueError(
                    "review identity is valid only for Red Team reviews")
            payload = daemon.SOL_TICKET_HEADER + ticket_kind + "\n\n" + text
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def redteam_review_receipt_payload(review_cycle, review_commit, result,
                                   text):
    """Build the exact Red Team return that completes one ticket cycle.

    Arguments:
      review_cycle  = the cycle identifier being closed.
      review_commit = the landing commit the Red Team reviewed.
      result        = ``"NO CHANGE"`` or ``"REOPEN"``.
      text          = the review's human text.

    Returns:
      The newline-terminated return payload.

    Raises:
      ValueError: for a malformed cycle, commit, or result.
    """
    if (not isinstance(review_cycle, str)
            or daemon.CYCLE_ID_RE.fullmatch(review_cycle) is None):
        raise ValueError("invalid Red Team review cycle: "
                         + repr(review_cycle))
    if (not isinstance(review_commit, str)
            or daemon.FULL_COMMIT_RE.fullmatch(review_commit) is None):
        raise ValueError("invalid Red Team review commit: "
                         + repr(review_commit))
    if result not in daemon.REDTEAM_REVIEW_RESULTS:
        raise ValueError("Red Team review result must be NO CHANGE or REOPEN")
    payload = (daemon.MAILBOX_RETURN_HEADER + "redteam-closure\n"
               + daemon.MAILBOX_CYCLE_HEADER + review_cycle + "\n"
               + daemon.MAILBOX_COMMIT_HEADER + review_commit + "\n"
               + daemon.MAILBOX_RESULT_HEADER + result + "\n\n" + text)
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def control_plane_review_receipt_payload(review_cycle, candidate, result,
                                         text):
    """Build one exact Red Team decision for protected candidate C.

    C is the Implementer's candidate commit.

    Arguments:
      review_cycle = the control-plane cycle identifier.
      candidate    = the reviewed candidate commit C.
      result       = one of the allowed control-plane results.
      text         = the review's human text.

    Returns:
      The newline-terminated decision payload.

    Raises:
      ValueError: for a malformed cycle, candidate, or result.
    """
    if (not isinstance(review_cycle, str)
            or daemon.CYCLE_ID_RE.fullmatch(review_cycle) is None):
        raise ValueError("invalid control-plane review cycle")
    if (not isinstance(candidate, str)
            or daemon.FULL_COMMIT_RE.fullmatch(candidate) is None):
        raise ValueError("invalid control-plane candidate")
    if result not in daemon.CONTROL_PLANE_REVIEW_RESULTS:
        raise ValueError("invalid control-plane review result")
    payload = (daemon.MAILBOX_RETURN_HEADER + "redteam-control-plane\n"
               + daemon.MAILBOX_CYCLE_HEADER + review_cycle + "\n"
               + daemon.MAILBOX_CANDIDATE_HEADER + candidate + "\n"
               + daemon.MAILBOX_RESULT_HEADER + result + "\n\n" + text)
    return payload if payload.endswith("\n") else payload + "\n"


def architect_user_request_payload(text, discovery_severity=None):
    """Build the persisted public envelope addressed only to Architect.

    The exact fix-only request line is saved bare. Every other
    request gains severity and scope headers; a request that begins
    with an explicit widespread-search command is forced to Low
    severity, the only severity a widespread search may carry.

    Arguments:
      text               = the user's request text.
      discovery_severity = severity to save, or ``None`` for the
                           default.

    Returns:
      The newline-terminated payload.

    Raises:
      ValueError: for an unknown severity.
    """
    if text == daemon.ARCHITECT_FIX_ONLY_REQUEST:
        return text
    if discovery_severity is None:
        discovery_severity = daemon.DEFAULT_DISCOVERY_SEVERITY
    discovery_scope = daemon.architect_request_scope(text=text)
    if discovery_scope == "widespread":
        discovery_severity = "low"
    if discovery_severity not in daemon.DISCOVERY_SEVERITIES:
        raise ValueError("invalid discovery severity: "
                         + repr(discovery_severity))
    payload = (daemon.SOL_SEVERITY_HEADER + discovery_severity + "\n"
               + daemon.SOL_SCOPE_HEADER + discovery_scope + "\n\n" + text)
    if not payload.endswith("\n"):
        payload = payload + "\n"
    return payload


def _architect_user_request_envelope(message):
    """Return ``(severity, scope, body, problem)`` for a public envelope.

    Arguments:
      message = the decoded mailbox message.

    Returns:
      The parsed fields with ``problem`` ``None`` on success. A
      missing or misordered header, or a second header-like line in
      the body, returns the original message as the body with a
      printable problem.
    """
    match = daemon.re.match(
        r"\A" + daemon.re.escape(daemon.SOL_SEVERITY_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.DISCOVERY_SEVERITIES)) + r")\r?\n"
        + daemon.re.escape(daemon.SOL_SCOPE_HEADER)
        + r"(" + "|".join(map(daemon.re.escape, daemon.DISCOVERY_SCOPES))
        + r")\r?\n\r?\n",
        message)
    if match is None:
        return (None, None, message,
                "a public Architect request needs exact MAILBOX-SEVERITY "
                "and MAILBOX-SCOPE headers, in that order, followed by one "
                "blank line")
    body = message[match.end():]
    reserved_like = (
        r"(?im)^[ \t]*mailbox[ \t]*-[ \t]*(?:severity|scope)[ \t]*:")
    if daemon.re.search(reserved_like, body) is not None:
        return None, None, message, "duplicate public request header"
    return match.group(1), match.group(2), body, None


def architect_user_request_problem(message):
    """Return a malformed public-envelope reason, or ``None``."""
    return daemon._architect_user_request_envelope(message=message)[3]


def architect_user_request_severity(message):
    """Return a valid public Architect envelope severity, or ``None``."""
    severity, _, _, problem = daemon._architect_user_request_envelope(
        message=message)
    return severity if problem is None else None


def architect_user_request_scope(message):
    """Return a valid public Architect envelope scope, or ``None``."""
    _, scope, _, problem = daemon._architect_user_request_envelope(message=message)
    return scope if problem is None else None


def architect_user_request_body(message):
    """Return the exact user text after a valid Architect envelope."""
    _, _, body, problem = daemon._architect_user_request_envelope(message=message)
    return message if problem is not None else body


def architect_admission_token(request_name, digest):
    """Return the exact token binding one public request to its handoff.

    Arguments:
      request_name = the public request's mailbox filename; it must
                     be addressed to the Architect route.
      digest       = SHA-256 hexadecimal digest of the saved request.

    Returns:
      ``request_name@digest`` — the token a later ticket must quote.

    Raises:
      daemon.TicketCycleStateError: for a name that is not a pending
        Architect message or a malformed digest.
    """
    match = (daemon.PENDING_MESSAGE_RE.fullmatch(request_name)
             if isinstance(request_name, str) else None)
    if (match is None or match.group(1) != "fable"
            or not isinstance(digest, str)
            or daemon.re.fullmatch(r"[0-9a-f]{64}", digest) is None):
        raise daemon.TicketCycleStateError(
            "invalid public Architect admission identity")
    return request_name + "@" + digest


def split_architect_admission_token(token):
    """Return ``(request_name, digest)`` for one exact admission token.

    Arguments:
      token = the saved token.

    Returns:
      The two halves, revalidated through architect_admission_token.

    Raises:
      daemon.TicketCycleStateError: for a token without ``@`` or with
        invalid halves.
    """
    if not isinstance(token, str) or "@" not in token:
        raise daemon.TicketCycleStateError(
            "invalid public Architect admission token")
    request_name, digest = token.rsplit("@", 1)
    daemon.architect_admission_token(request_name=request_name, digest=digest)
    return request_name, digest


def message_claims_architect_admission(path, token):
    """Return whether one mailbox file names this exact public request.

    Arguments:
      path  = the mailbox file to inspect.
      token = the admission token to look for.

    Returns:
      True only when the file is readable and carries the exact
      admission header line; any read problem counts as no.
    """
    try:
        message = daemon.read_cycle_message(path=path)
    except (OSError, ValueError, daemon.TicketCycleStateError):
        return False
    return daemon.MAILBOX_ADMISSION_HEADER + token in message.splitlines()


def architect_admission_prompt(token):
    """Tell one public Architect turn how to bind its single outcome.

    Arguments:
      token = the request's admission token, or ``None`` for no
              admission (returns the empty string).

    Returns:
      The instruction block requiring exactly one outcome — an
      Implementer ticket, a Sol discovery request, or a no-ticket
      receipt — each quoting the exact admission line.
    """
    if token is None:
        return ""
    request_name, digest = daemon.split_architect_admission_token(token=token)
    exact = daemon.architect_admission_token(
        request_name=request_name, digest=digest)
    return (
        "PUBLIC REQUEST ADMISSION:\n"
        "This public request provisionally occupies one finite ticket slot. "
        "The daemon proved this slot is free now; its decision is "
        "authoritative. Past tickets do not consume it. "
        "Maintenance must choose an eligible Open bug when one remains. "
        "Produce exactly ONE fresh outcome and never remain silent:\n"
        "1. For one Implementer ticket, put this exact line first in the "
        "body immediately after the ticket flow headers and blank line:\n"
        "    " + daemon.MAILBOX_ADMISSION_HEADER + exact + "\n"
        "2. For one bounded or widespread Sol discovery request, put the "
        "same exact line first in its body immediately after the Sol "
        "severity and scope headers.\n"
        "3. If this request creates no ticket, write one fresh "
        "<next-sequence>-to-user.md receipt beginning exactly:\n"
        "    " + daemon.MAILBOX_RETURN_HEADER
        + daemon.PUBLIC_ARCHITECT_NO_TICKET_RETURN + "\n"
        "    " + daemon.MAILBOX_ADMISSION_HEADER + exact + "\n"
        "    " + daemon.MAILBOX_DECISION_HEADER
        + daemon.PUBLIC_ARCHITECT_NO_TICKET_DECISION + "\n"
        "You may put a plain-language answer after one blank line in option "
        "3. Do not produce two outcomes, copy the admission to later work, "
        "or treat silence as success.\n\n")


def valid_sol_transport(message):
    """Return whether ``message`` is exactly the daemon's Sol ping."""
    return message == daemon.sol_ticket_payload(
        ticket_kind="transport", text=daemon.transport_ping_text(agent="sol"))


def fix_only_environment_active():
    """Return whether this send inherited a fix-only watch contract.

    The watch exports the setting to its children, so a helper send
    started inside a fix-only watch obeys the same restriction.

    Returns:
      True when the environment variable holds 1, true, or yes.
    """
    value = daemon.os.environ.get(daemon.FIX_ONLY_ENVIRONMENT)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def skip_redteam_environment_active():
    """Return whether this send inherited a two-role watch contract.

    The watch exports the setting to its children, so a helper send
    started inside a two-role watch also avoids the Sol route.

    Returns:
      True when the environment variable holds 1, true, or yes.
    """
    value = daemon.os.environ.get(daemon.SKIP_REDTEAM_ENVIRONMENT)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def resolve_discovery_severity(cli_value=None):
    """Bind an explicit severity to the inherited run default.

    Arguments:
      cli_value = the ``--severity`` option, or ``None`` when
                  omitted.

    Returns:
      The severity to use: the explicit value when it matches any
      inherited environment value, otherwise the inherited or default
      one.

    Raises:
      ValueError: for an invalid value or a disagreement with the
        inherited environment.
    """
    inherited = daemon.os.environ.get(daemon.DISCOVERY_SEVERITY_ENVIRONMENT)
    if inherited is not None and inherited not in daemon.DISCOVERY_SEVERITIES:
        raise ValueError(
            daemon.DISCOVERY_SEVERITY_ENVIRONMENT
            + " must be exactly high, medium, or low")
    if cli_value is None:
        return (daemon.DEFAULT_DISCOVERY_SEVERITY
                if inherited is None else inherited)
    if cli_value not in daemon.DISCOVERY_SEVERITIES:
        raise ValueError("discovery severity must be high, medium, or low")
    if inherited is not None and cli_value != inherited:
        raise ValueError(
            "--severity " + cli_value + " does not match inherited "
            + daemon.DISCOVERY_SEVERITY_ENVIRONMENT + " " + inherited)
    return cli_value


def sol_ticket_refusal(ticket_kind, admission_count, fix_only,
                       transport_valid=False, discovery_severity=None,
                       discovery_scope=None, unclassified_count=0,
                       ledger_problem=None):
    """Return the binding refusal reason for a Sol ticket, or ``None``.

    The checks run in a fixed order: the transport ping must be the
    daemon's exact payload; the ticket kind must be known; the
    backlog must be readable and classified; discovery needs a valid
    severity and scope; a fix-only watch admits closing work only; a
    widespread search must be Low and wait for an empty non-Low
    backlog; and ordinary discovery stops at the admission threshold.

    Arguments:
      ticket_kind        = the ticket kind from the message.
      admission_count    = open Critical, High, and Medium tickets.
      fix_only           = True in a fix-only watch.
      transport_valid    = True when the transport payload matched.
      discovery_severity = discovery severity, or ``None`` for the
                           default.
      discovery_scope    = saved discovery scope.
      unclassified_count = open backlog lines without one exact
                           classification.
      ledger_problem     = a backlog read problem, or ``None``.

    Returns:
      A printable refusal sentence, or ``None`` when the ticket may
      proceed.
    """
    if ticket_kind == "transport":
        if transport_valid:
            return None
        return ("MAILBOX-TICKET: transport is reserved for the daemon's "
                "exact --ping sol payload")
    if ticket_kind not in daemon.SOL_TICKET_KINDS:
        return ("missing or invalid first line; every Sol ticket must start "
                "with exactly 'MAILBOX-TICKET: closure', "
                "'MAILBOX-TICKET: discovery', 'MAILBOX-TICKET: policy', "
                "or 'MAILBOX-TICKET: control-plane'")
    if ledger_problem is not None:
        return ledger_problem
    if ticket_kind == "discovery":
        if discovery_severity is None:
            discovery_severity = daemon.DEFAULT_DISCOVERY_SEVERITY
        if discovery_severity not in daemon.DISCOVERY_SEVERITIES:
            return ("a discovery ticket needs one severity: high, medium, "
                    "or low")
        if discovery_scope not in daemon.DISCOVERY_SCOPES:
            return ("a discovery ticket needs one saved scope: bounded or "
                    "widespread")
    elif discovery_severity is not None:
        return "--severity is valid only for discovery tickets"
    elif discovery_scope is not None:
        return "discovery scope is valid only for discovery tickets"
    if fix_only and ticket_kind not in {
            "closure", "policy", "control-plane"}:
        return ("fix-only watch is closing-only; discovery tickets and new "
                "backlog lines are forbidden until the watch is restarted "
                "without --fix-only")
    if ticket_kind == "discovery" and unclassified_count:
        return ("the backlog has " + str(unclassified_count)
                + " unclassified open ticket(s); the Architect must assign "
                "each one a valid priority and either BUG FIX or NEW "
                "FUNCTIONALITY before new discovery can enter")
    if (ticket_kind == "discovery"
            and discovery_scope == "widespread"):
        if discovery_severity != "low":
            return ("a widespread search is automatically Low; save exactly "
                    "MAILBOX-SEVERITY: low")
        if admission_count:
            return ("a widespread search waits until no open Critical, High, "
                    "or Medium ticket remains; the current non-Low count is "
                    + str(admission_count) + ". Open Low tickets do not block "
                    "this search")
    if (ticket_kind == "discovery"
            and admission_count >= daemon.DISCOVERY_ADMISSION_THRESHOLD):
        return ("the open Critical, High, and Medium ticket count is "
                + str(admission_count) + ", at or past "
                + str(daemon.DISCOVERY_ADMISSION_THRESHOLD)
                + "; do not admit this discovery yet. Record it as a local "
                "deferred candidate without a countable '- OPEN' marker. "
                "Low tickets do not count toward this limit. When the count "
                "falls below the threshold, assess the result and insert an "
                "accepted ticket in the matching Critical, High, Medium, or "
                "Low backlog group; only the Architect may designate "
                "Critical")
    return None


def inflight_lane_blockers(skip_redteam=False):
    """Return unresolved inflight agent messages grouped by cwd lane.

    Only exact dispatchable message names participate. A hand-made file or an
    archived ``-to-user`` note under inflight cannot block an agent lane.
    Live topology gives Architect, Implementer, and Sol distinct saved
    directories, so one unresolved role blocks only that role. Imported tests
    may still deliberately assign a shared cwd and retain shared-tree safety.

    Arguments:
      skip_redteam = True to ignore Sol messages whose lane is not
                     shared with an enabled Claude role.

    Returns:
      Mapping from lane identity to its blocker paths, sorted by
      message sequence.
    """
    blockers = {}
    seen = {}
    patterns = [
        daemon.os.path.join(daemon.MAILBOX, "inflight", "*.md"),
        daemon.os.path.join(daemon.MAILBOX, "inflight",
                     "*.md" + daemon.STATE_GUARD_SUFFIX),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(daemon.glob.glob(pattern))
    for path in paths:
        name = daemon.blocker_message_name(path=path)
        match = daemon.PENDING_MESSAGE_RE.match(name)
        if match is None:
            continue
        agent = match.group(1)
        cwd = daemon.mailbox_lane_cwd(agent=agent)
        enabled_claude_cwds = {
            daemon.AGENT_CWD["fable"], daemon.AGENT_CWD["opus"]}
        if (skip_redteam and agent == "sol"
                and cwd not in enabled_claude_cwds):
            continue
        if cwd not in blockers:
            blockers[cwd] = []
            seen[cwd] = set()
        if name in seen[cwd]:
            continue
        seen[cwd].add(name)
        blockers[cwd].append(path)
    for paths in blockers.values():
        paths.sort(key=daemon.message_sequence)
    return blockers


def blocker_message_name(path):
    """Return the exact agent basename encoded by an inflight blocker.

    Arguments:
      path = the inflight file, possibly a state-guard sidecar.

    Returns:
      The message basename with any state-guard suffix removed.
    """
    name = daemon.os.path.basename(path)
    if name.endswith(daemon.STATE_GUARD_SUFFIX):
        return name[:-len(daemon.STATE_GUARD_SUFFIX)]
    return name


def report_inflight_lane_block(blocker_paths, pending_count):
    """Print one clear cross-pass lane-block diagnostic.

    Arguments:
      blocker_paths = the unresolved inflight files for one lane.
      pending_count = pending root messages waiting on that lane.
    """
    blocker_names = [daemon.blocker_message_name(path=path)
                     for path in blocker_paths]
    if pending_count:
        waiting = (str(pending_count)
                   + " pending message(s) sharing that working directory "
                   "will wait.")
    else:
        waiting = ("no pending root messages share that working directory "
                   "yet.")
    print("  lane blocked by unresolved inflight message(s) "
          + ", ".join(blocker_names) + "; " + waiting)


def message_sequence(path):
    """Return the numeric sequence at the start of a message filename.

    Arguments:
      path = a mailbox message path accepted by pending_messages().

    Returns:
      The integer before ``-to-`` in the filename.
    """
    value = daemon.sequence_in_name(name=daemon.os.path.basename(path))
    if value is None:
        raise ValueError("not a numbered mailbox message: " + path)
    return value


def sequence_in_name(name):
    """Return a mailbox filename's numeric sequence, if it has one.

    This is the single parser used by both ``next_seq()`` and the dispatch
    currency snapshot, so a message cannot count for allocation while being
    invisible to the dispatch-time maximum.

    Arguments:
      name = a basename from anywhere in the mailbox store.

    Returns:
      The leading integer, or None when the name is not a numbered message.
    """
    match = daemon.MESSAGE_SEQUENCE_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def dispatch_currency(dispatch_path, agent):
    """Take one post-claim snapshot and derive its mechanical currency.

    The maximum spans every ``*.md`` below the mailbox, including done,
    failed, hold, and -to-user messages. The newer-message count is narrower:
    only root-pending agent messages whose recipient shares this dispatch's
    working-directory lane count. This is evidence for the receiving human or
    agent, never a semantic decision that the message is obsolete.

    Arguments:
      dispatch_path = the already-claimed inflight message.
      agent         = its recipient.

    Returns:
      ``(store_max_sequence, newer_root_pending_in_lane)``.
    """
    snapshot = daemon.glob.glob(daemon.os.path.join(daemon.MAILBOX, "**", "*.md"),
                         recursive=True)
    dispatched_sequence = daemon.message_sequence(path=dispatch_path)
    store_max = 0
    newer_in_lane = 0
    mailbox_root = daemon.os.path.abspath(daemon.MAILBOX)
    for path in snapshot:
        value = daemon.sequence_in_name(name=daemon.os.path.basename(path))
        if value is None:
            continue
        if value > store_max:
            store_max = value
        if daemon.os.path.dirname(daemon.os.path.abspath(path)) != mailbox_root:
            continue
        pending_match = daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(path))
        if pending_match is None or value <= dispatched_sequence:
            continue
        queued_agent = pending_match.group(1)
        if (daemon.mailbox_lane_cwd(agent=queued_agent)
                == daemon.mailbox_lane_cwd(agent=agent)):
            newer_in_lane = newer_in_lane + 1
    return store_max, newer_in_lane


def timeout_history_path(name):
    """Return the daemon-owned timeout history sidecar for one message.

    A sidecar is a small companion file that stores facts about
    another file without changing it.

    Arguments:
      name = the message basename.

    Returns:
      The JSON sidecar path under the mailbox's dispatch-history
      folder.
    """
    return daemon.os.path.join(daemon.MAILBOX, ".dispatch-history", name + ".json")


def timeout_events(name):
    """Read the timeout-only event list for one message basename.

    A missing sidecar means the message has never timed out. A malformed
    daemon-owned sidecar is not treated as an empty history: dispatch must not
    erase the only evidence that an earlier turn was killed.

    Arguments:
      name = the message basename.

    Returns:
      The list of normalized timeout events, oldest first; empty when
      the message never timed out.

    Raises:
      ValueError: for an oversized, malformed, or misidentified
        sidecar.
    """
    path = daemon.timeout_history_path(name=name)
    try:
        with open(path, encoding="utf-8") as f:
            if daemon.os.fstat(f.fileno()).st_size > daemon.MAX_TIMEOUT_HISTORY_BYTES:
                raise ValueError("timeout history is too large in " + path)
            try:
                payload = daemon.json.load(f)
            except (RecursionError, OverflowError) as exc:
                raise ValueError(
                    "timeout history is too deeply nested in " + path) \
                    from exc
    except FileNotFoundError:
        return []
    if not isinstance(payload, dict):
        raise ValueError("timeout history is not a mapping in " + path)
    if payload.get("schema") != 1 or payload.get("message") != name:
        raise ValueError("invalid timeout-history identity in " + path)
    events = payload.get("timeouts")
    if not isinstance(events, list):
        raise ValueError("invalid timeout-history event list in " + path)
    if len(events) > daemon.MAX_TIMEOUT_HISTORY_EVENTS:
        raise ValueError("too many timeout-history events in " + path)
    normalized = []
    for event in events:
        duration = event.get("killed_after_minutes") \
            if isinstance(event, dict) else None
        if not daemon.valid_duration(value=duration, strictly_positive=True):
            raise ValueError("invalid timeout duration in " + path)
        observed = event.get("observed_elapsed_minutes")
        if (observed is not None
                and not daemon.valid_duration(value=observed,
                                       strictly_positive=False)):
            raise ValueError("invalid observed timeout duration in " + path)
        clean_event = {"killed_after_minutes": duration}
        if observed is not None:
            clean_event["observed_elapsed_minutes"] = observed
        normalized.append(clean_event)
    return normalized


def valid_duration(value, strictly_positive):
    """Return whether a JSON duration is numeric, finite, and in range.

    Integers are finite by definition; avoiding ``math.isfinite`` for them
    also keeps an attacker-controlled enormous JSON integer from raising an
    OverflowError during validation.

    Arguments:
      value             = the JSON value to check.
      strictly_positive = True to require a value above zero, False
                          to also allow zero.

    Returns:
      True for a numeric, finite, in-range duration. Booleans are
      refused even though Python counts them as integers.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not daemon.math.isfinite(value):
        return False
    if value > daemon.MAX_DISPATCH_TIMEOUT_MINUTES:
        return False
    if strictly_positive:
        return value > 0
    return value >= 0


def write_timeout_history(name, killed_after_minutes,
                          observed_elapsed_minutes=None):
    """Append one timeout event through an fsynced atomic replacement.

    This function is called only after the timeout guard kills a child.
    Ordinary nonzero exits never create or append a sidecar.

    Arguments:
      name                     = the message basename.
      killed_after_minutes     = the promised kill threshold that
                                 fired.
      observed_elapsed_minutes = the measured runtime, when known.

    Raises:
      ValueError: for invalid durations or a full event history.
    """
    if not daemon.valid_duration(value=killed_after_minutes,
                          strictly_positive=True):
        raise ValueError("killed-after timeout must be positive")
    if (observed_elapsed_minutes is not None
            and not daemon.valid_duration(value=observed_elapsed_minutes,
                                   strictly_positive=False)):
        raise ValueError("observed timeout duration must be nonnegative")
    events = daemon.timeout_events(name=name)
    if len(events) >= daemon.MAX_TIMEOUT_HISTORY_EVENTS:
        raise ValueError("timeout history reached its event limit")
    event = {"killed_after_minutes": killed_after_minutes}
    if observed_elapsed_minutes is not None:
        event["observed_elapsed_minutes"] = observed_elapsed_minutes
    events.append(event)
    payload = {"schema": 1, "message": name, "timeouts": events}
    directory = daemon.os.path.dirname(daemon.timeout_history_path(name=name))
    daemon.os.makedirs(directory, exist_ok=True)
    handle, temporary = daemon.tempfile.mkstemp(prefix=".timeout-", dir=directory)
    try:
        with daemon.os.fdopen(handle, "w", encoding="utf-8") as f:
            daemon.json.dump(payload, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")
            f.flush()
            daemon.os.fsync(f.fileno())
        daemon.os.replace(temporary, daemon.timeout_history_path(name=name))
    finally:
        if daemon.os.path.exists(temporary):
            daemon.os.remove(temporary)


def exact_duration(value):
    """Format a stored float without changing its represented value.

    The ``.17g`` format prints enough digits that reading the text
    back yields the identical floating-point number.

    Arguments:
      value = the stored duration.

    Returns:
      The exact decimal text.
    """
    return format(value, ".17g")


def dispatch_banner(store_max, newer_in_lane, previous_timeout_minutes,
                    fix_only=False, skip_redteam=False,
                    discovery_severity=None, discovery_scope=None,
                    saved_discovery=False,
                    saved_architect_request=False,
                    candidate_scope=None, routine_review=None):
    """Build the mechanical pre-preamble hint for a live dispatch.

    The banner is prepended to a dispatched message. It carries only
    mechanical facts and binding reminders: the store's newest
    sequence, newer messages queued on this lane, a previous timeout,
    the fix-only and two-role restrictions, a candidate's path-scope
    verdict, the routine-review kind, the discovery severity and
    scope ladders, and the ticket character budget with the exact
    guard commands to run.

    Arguments:
      store_max                = newest sequence anywhere in the
                                 store.
      newer_in_lane            = newer pending messages on this lane.
      previous_timeout_minutes = last kill threshold, or ``None``.
      fix_only                 = True in a fix-only watch.
      skip_redteam             = True in a two-role watch.
      discovery_severity       = severity to display, or ``None`` for
                                 the run default.
      discovery_scope          = scope to display, or ``None`` for
                                 the default.
      saved_discovery          = True when severity and scope came
                                 saved on this discovery message.
      saved_architect_request  = True when they came saved on a
                                 public Architect request.
      candidate_scope          = the candidate's path verdict
                                 mapping, or ``None``.
      routine_review           = routine-review display name, or
                                 ``None``; a routine review shortens
                                 the banner to its own block.

    Returns:
      The banner text ending in one blank line.
    """
    lines = [
        "--- DISPATCH CURRENCY (mechanical hint only) ---",
        "store-wide mailbox max sequence at claim: %04d" % store_max,
        ("newer messages queued in this working-directory lane: "
         + str(newer_in_lane)),
        ("This marker is not a semantic supersession oracle; read the "
         "mailbox and cited notes first."),
    ]
    if previous_timeout_minutes is not None:
        lines.append(
            "this dispatch previously ran for "
            + daemon.exact_duration(value=previous_timeout_minutes)
            + " minutes and was killed")
    if fix_only:
        lines.append(
            "fix-only watch: active; close existing ledger lines only; "
            "create no discovery tickets or new backlog lines.")
    if skip_redteam:
        lines.append(
            "two-role watch: the Red Team and entire Sol route are disabled; "
            "create no to-sol messages; route Implementer evidence to the "
            "Architect and Architect repair handoffs to the Implementer.")
    lines.append("--- END DISPATCH CURRENCY ---")
    lines.append("")
    if candidate_scope is not None:
        result = candidate_scope["result"]
        lines.extend(("--- CANDIDATE TICKET SCOPE (binding) ---",
                      "result: " + result))
        if candidate_scope["paths"]:
            lines.append("paths: " + ", ".join(
                repr(path) for path in candidate_scope["paths"]))
        if result == "SCOPE_EXCEEDED":
            lines.append(
                "Candidate C is preserved, but the Implementer expanded the "
                "ticket. Architect GO explicitly accepts this expansion; a "
                "repair handoff rejects it. Audit the listed paths.")
        lines.extend(("--- END CANDIDATE TICKET SCOPE ---", ""))
    if routine_review is not None:
        lines.extend((
            "--- ROUTINE REVIEW (binding) ---",
            "kind: " + routine_review,
            "Review the named ticket and commit only. This is not a new "
            "discovery search.",
            "ticket character limit: "
            + ("none (--max 0)" if daemon.MAX_CHARACTERS == 0 else
               str(daemon.MAX_CHARACTERS) + " added plus deleted characters"),
            "--- END ROUTINE REVIEW ---"))
        return "\n".join(lines) + "\n\n"
    lines.append("--- DISCOVERY SEVERITY (binding) ---")
    if discovery_severity is None:
        discovery_severity = daemon.DISCOVERY_SEVERITY
    if saved_discovery:
        lines.append(
            "user's saved minimum severity for this discovery: "
            + discovery_severity)
    elif saved_architect_request:
        lines.append(
            "user's saved minimum severity for any discovery requested "
            "by this ticket: " + discovery_severity)
    else:
        lines.append(
            "minimum severity to save on any new discovery ticket: "
            + discovery_severity)
    lines.append(
        "high: only a bug that severely impacts core functionality, causes "
        "data loss, halts system operations, or makes the science wrong.")
    lines.append(
        "medium: high bugs plus a less severe bug that can affect normal "
        "operation and has a probable path; a merely theoretical or "
        "improbable edge case does not qualify.")
    lines.append(
        "low: any concrete discovered bug may qualify, including an "
        "improbable edge case; an unsupported guess is not a discovery.")
    if fix_only:
        lines.append(
            "fix-only is stronger than this setting: create no discovery "
            "ticket or new backlog line.")
    if skip_redteam:
        lines.append(
            "the Sol route is disabled: create no discovery ticket while "
            "this two-role watch is active.")
    lines.append(
        "The Red Team records User severity setting, Red Team severity, "
        "Likelihood (probable or improbable), Likelihood evidence, and "
        "Meets user setting (yes or no).")
    lines.append(
        "The Architect accepts, upgrades, or downgrades that rating with an "
        "evidence-based reason, then makes the final GO or NO-GO ticket "
        "decision. The Red Team never opens the ticket itself.")
    lines.append("--- END DISCOVERY SEVERITY ---")
    lines.append("")
    lines.append("--- DISCOVERY SCOPE (binding) ---")
    if discovery_scope is None:
        discovery_scope = daemon.DEFAULT_DISCOVERY_SCOPE
    if saved_discovery:
        lines.append("saved scope for this discovery: " + discovery_scope)
    elif saved_architect_request:
        lines.append(
            "saved scope for discovery requested by this ticket: "
            + discovery_scope)
    else:
        lines.append(
            "scope to save on an ordinary new discovery: "
            + discovery_scope)
    lines.append(
        "bounded: review only the named commit or change and the behavior "
        "it directly affects.")
    lines.append(
        "widespread: search beyond one named change; it must remain Low and "
        "wait until no Critical, High, or Medium ticket is open.")
    lines.append(
        "Trust MAILBOX-SCOPE and MAILBOX_DISCOVERY_SCOPE, not a phrase found "
        "in the body or cited note.")
    lines.append("--- END DISCOVERY SCOPE ---")
    lines.append("")
    lines.append("--- TICKET CHARACTER BUDGET (binding) ---")
    primary = daemon.AGENT_CWD["fable"]
    contract_tool = daemon.os.path.join(
        primary, "ai", "tools", "handoff_contract.py")
    change_tool = daemon.os.path.join(
        primary, "ai", "tools", "ticket_change_guard.py")
    if daemon.MAX_CHARACTERS == 0:
        lines.append(
            "ticket limit: none (--max 0); readability, complete behavior, "
            "tests, explanations, and failure handling remain required; "
            "obfuscated work is NO-GO.")
        lines.append(
            "The Architect must record the unlimited budget and validate the "
            "structured directive by running `python3 " + contract_tool
            + " architect NOTE_ABSOLUTE_PATH --max 0` before GO.")
    else:
        value = str(daemon.MAX_CHARACTERS)
        lines.append(
            "ticket limit: at most " + value + " characters added plus "
            "deleted from the directive Base.")
        lines.append(
            "Before final GO or ticket closure, the Architect must run "
            "`python3 " + change_tool
            + " --repo EXECUTION_WORKTREE --base BASE --max " + value
            + "`. The program path belongs to the primary AI worktree; "
            "--repo selects the exact proposed commit.")
        lines.append(
            "The Architect must record the structured budget evidence and "
            "validate the structured directive by running `python3 "
            + contract_tool + " architect NOTE_ABSOLUTE_PATH --max " + value
            + "` before GO.")
        lines.append(
            "Over-limit, unmeasurable, or obfuscated work is NO-GO; never "
            "compress readable code, omit tests or explanations, or leave "
            "requested behavior incomplete to fit.")
    lines.append("--- END TICKET CHARACTER BUDGET ---")
    return "\n".join(lines) + "\n\n"


def report_ticket_character_limit():
    """Print the effective per-ticket text limit at live startup."""
    if daemon.MAX_CHARACTERS == 0:
        print("ticket character limit: none (--max 0)")
        return
    print("ticket character limit: " + str(daemon.MAX_CHARACTERS)
          + " added plus deleted characters per ticket")


def report_discovery_severity(fix_only=False, skip_redteam=False):
    """Print the default saved on new discovery tickets for this run."""
    line = "discovery severity default: " + daemon.DISCOVERY_SEVERITY
    if fix_only:
        line = "minimum bug-fix severity: " + daemon.DISCOVERY_SEVERITY
    elif skip_redteam:
        line = line + " (inactive while the Sol route is disabled)"
    else:
        line = line + " (saved on each new discovery ticket)"
    print(line)


def placeholder_in(message):
    """Return a marker only when the whole body is an unfilled template.

    A real audit may need to discuss a literal such as ``<unit>``. Treating
    every substring occurrence as an unfilled template rejects that audit.

    Arguments:
      message = the decoded mailbox body.

    Returns:
      The matching marker, or None when the body carries real text.
    """
    body = message.strip()
    for marker in daemon.PLACEHOLDER_MARKERS:
        if body == marker:
            return marker
    return None
