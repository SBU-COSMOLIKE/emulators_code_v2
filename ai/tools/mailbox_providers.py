"""Implementer runtime records, provider checks, and backlog ledger queries.

The Implementer may be served by Claude or by an Ollama-served model,
and a restart must not silently change that choice in the middle of a
ticket. This file records the verified provider, model, and context
values for the active cycle and compares a later start against them.
It also selects the cheaper reasoning effort for routine review turns
and answers small questions about the tracked backlog: how many
tickets are open, at which severities, and which bug entries fix-only
maintenance may choose.

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
    "default_implementer_context",
    "implementer_runtime_record",
    "verified_implementer_runtime",
    "routine_review_command",
    "implementer_checkpoint_settings",
    "claude_compaction_limit",
    "mailbox_lane_cwd",
    "backlog_ledger_count",
    "backlog_severity_counts",
    "eligible_fix_only_bug_anchors",
    "verified_backlog_lines",
    "backlog_reopening_status",
    "discovery_admission_count",
)


def default_implementer_context(provider):
    """Return the provider-safe default for one Implementer shell.

    Arguments:
      provider = ``"claude"`` or ``"ollama"``.

    Returns:
      That provider's default Implementer context budget in tokens —
      the conversation size the runtime plans around.

    Raises:
      ValueError: for any other provider name.
    """
    if provider == "claude":
        return daemon.DEFAULT_IMPLEMENTER_CONTEXT_BUDGET
    if provider == "ollama":
        return daemon.DEFAULT_OLLAMA_IMPLEMENTER_CONTEXT_BUDGET
    raise ValueError("Implementer provider must be claude or ollama")


def implementer_runtime_record(
        *, provider, model, context_limit, compaction_limit):
    """Return one validated runtime identity for the stable ``opus`` role.

    The mailbox route name ``opus`` identifies the Implementer no
    matter which provider or model serves it. This record is what a
    later start compares against the one saved for the active cycle,
    so a restart cannot silently change the provider or model in the
    middle of a ticket.

    Arguments:
      provider         = ``"claude"`` or ``"ollama"``.
      model            = model name, validated before it is recorded.
      context_limit    = verified model context in tokens; must be a
                         positive integer, never a Boolean.
      compaction_limit = Claude Code compaction threshold in tokens;
                         must be positive and fit inside the context.

    Returns:
      A mapping with the role address, provider, model, context limit,
      and compaction limit — the exact fields a later start compares.

    Raises:
      ValueError: for an unknown provider, an invalid model name, a
        non-positive limit, or a compaction limit above the context.
    """
    if provider not in daemon.IMPLEMENTER_PROVIDERS:
        raise ValueError("Implementer provider must be claude or ollama")
    model = daemon.validate_model_name(value=model)
    for label, value in (("context limit", context_limit),
                         ("compaction limit", compaction_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Implementer " + label + " must be positive")
    if compaction_limit > context_limit:
        raise ValueError(
            "Implementer compaction limit exceeds its model context")
    return {
        "role_address": "opus",
        "provider": provider,
        "model": model,
        "context_limit": context_limit,
        "compaction_limit": compaction_limit,
    }


def verified_implementer_runtime(
        *, provider, model, compaction_limit, dry_run=False):
    """Build the canonical runtime, verifying Ollama before a live watch.

    A Claude Implementer needs no preflight — Claude Code manages its
    own context — so its record simply equates the context and
    compaction limits. An Ollama Implementer is really launched once,
    through the provider health check, so the recorded context limit
    is a verified value rather than a hope; a dry run skips that
    launch and records the compaction limit as the context.

    Arguments:
      provider         = ``"claude"`` or ``"ollama"``.
      model            = model name for the Implementer.
      compaction_limit = Claude Code compaction threshold in tokens.
      dry_run          = True to skip the live Ollama preflight.

    Returns:
      The validated runtime record from implementer_runtime_record.

    Raises:
      ValueError: for an unknown provider, or when the Ollama
        preflight fails — in that case no ticket is started.
    """
    if provider == "claude":
        return daemon.implementer_runtime_record(
            provider=provider, model=model, context_limit=compaction_limit,
            compaction_limit=compaction_limit)
    if provider != "ollama":
        raise ValueError("Implementer provider must be claude or ollama")
    if dry_run:
        return daemon.implementer_runtime_record(
            provider=provider, model=model, context_limit=compaction_limit,
            compaction_limit=compaction_limit)
    context_limit, _problem = daemon._PROVIDER_HEALTH.check_ollama_implementer(
        model=model, compaction_limit=compaction_limit,
        minimum_context=daemon.MINIMUM_OLLAMA_CONTEXT,
        preamble=daemon.agent_preamble(agent="opus"),
        nonce=daemon.secrets.token_hex(16), ollama_executable=daemon.OLLAMA_EXECUTABLE,
        timeout=daemon.PROVIDER_PING_TIMEOUT_SECONDS, run=daemon.subprocess.run)
    if context_limit is None:
        raise ValueError(
            "Ollama Implementer preflight failed; no ticket was started")
    return daemon.implementer_runtime_record(
        provider=provider, model=model, context_limit=context_limit,
        compaction_limit=compaction_limit)


def routine_review_command(
        command, *, agent, ticket_kind=None, candidate_audit=False,
        reopening=False, checkpoint=False, integration=False, effort=None):
    """Return the exact command and label for one lower-cost review turn.

    Arguments:
      command         = the role's full launch command.
      agent           = ``"fable"`` (the Architect) or ``"sol"`` (the
                        Red Team); other roles never get a routine
                        review.
      ticket_kind     = declared ticket kind, read for a Red Team turn.
      candidate_audit = True for an Architect candidate audit.
      reopening       = True for an Architect reopening decision.
      checkpoint      = True for an Architect checkpoint review.
      integration     = True for an Architect integration
                        revalidation.
      effort          = effort level to select, or ``None`` for the
                        configured routine-review effort.

    Returns:
      ``(command_list, kind)``: the command with its effort replaced
      and the routine-review display name, or a copy of the unchanged
      command and ``None`` when the turn keeps its full effort.
    """
    kind = daemon._REVIEW_DISPATCH.review_kind(
        agent=agent, ticket_kind=ticket_kind,
        candidate_audit=candidate_audit, reopening=reopening,
        checkpoint=checkpoint, integration=integration)
    if kind is None:
        return list(command), None
    selected = daemon.REVIEW_EFFORT if effort is None else effort
    return (daemon._REVIEW_DISPATCH.command_with_effort(
                command, agent=agent, effort=selected), kind)


def implementer_checkpoint_settings(python, hook_path):
    """Return the Implementer's time and context checkpoint hooks.

    The returned mapping is Claude Code settings JSON: it registers
    the checkpoint hook program for the three session moments it
    understands — after a batch of tool calls, when the session tries
    to finish, and before automatic compaction.

    Arguments:
      python    = Python interpreter that runs the hook.
      hook_path = path of ai/tools/implementer_checkpoint_hook.py.

    Returns:
      The settings mapping to merge into the Implementer's launch
      settings; each registration gives the hook five seconds to
      answer.
    """
    hook = {
        "type": "command",
        "command": python,
        "args": [hook_path],
        "timeout": 5,
    }
    return {"hooks": {
        "PostToolBatch": [{"hooks": [hook]}],
        "Stop": [{"hooks": [hook]}],
        "PreCompact": [{"matcher": "auto", "hooks": [hook]}],
    }}


def claude_compaction_limit(agent):
    """Return the independent Claude Code limit for one Claude-backed role.

    Arguments:
      agent = ``"fable"`` for the Architect or ``"opus"`` for the
              Implementer; the two budgets are configured separately
              so one role's long session cannot shrink the other's.

    Returns:
      The compaction threshold in tokens for that role.

    Raises:
      ValueError: for any other role name.
    """
    if agent == "fable":
        return daemon.ARCHITECT_CONTEXT_BUDGET
    if agent == "opus":
        return daemon.IMPLEMENTER_RUNTIME["compaction_limit"]
    raise ValueError("Claude compaction limit has no role " + repr(agent))


def mailbox_lane_cwd(agent):
    """Return a serialization identity for an AI route or local daemon lane.

    A lane is the daemon's unit of one-at-a-time work: two jobs whose
    lanes share this identity must not run at the same time. Each AI
    route serializes on its role's working folder; the daemon's own
    local work serializes on the mailbox folder.

    Arguments:
      agent = a role route name, or ``"daemon"`` for the local lane.

    Returns:
      The path that names the lane.
    """
    if agent == "daemon":
        return daemon.MAILBOX
    return daemon.AGENT_CWD[agent]


def backlog_ledger_count():
    """Count every open ticket recorded in the backlog ledger.

    Returns:
      The number of classified and unclassified lines beginning ``- OPEN``.
      Zero is returned when the ledger does not exist.
    """
    counts = daemon.backlog_severity_counts()
    return (counts["critical"] + counts["high"] + counts["medium"]
            + counts["low"] + counts["unclassified"])


def backlog_severity_counts():
    """Count open backlog tickets by the Architect's final severity.

    ``Critical`` is a final backlog classification, not a public discovery
    setting. An open line that lacks one exact classification is reported as
    ``unclassified`` so malformed bookkeeping cannot disappear silently.
    Each indexed open ticket must also have one exact Red Team reopen-count
    row in its detailed section. After more than five reopens, only Low is a
    valid severity. The Red Team remains advisory: this bookkeeping rule does
    not wait for a review or block the Architect from closing a ticket.

    Returns:
      A mapping with severity totals, High-bug and feature subtotals, and an
      ``unclassified`` count.
    """
    counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "high_bug_fix": 0,
        "high_new_functionality": 0,
        "unclassified": 0,
        "problem": None,
    }
    lines, problem = daemon.verified_backlog_lines()
    if problem is not None:
        counts["problem"] = problem
        return counts
    anchor_counts = {}
    anchor_positions = []
    closed_position = next(
        (index for index, line in enumerate(lines)
         if line == "# Closed tickets"), len(lines))
    for line_number, line in enumerate(lines):
        anchor_match = daemon.BACKLOG_DETAIL_ANCHOR_RE.fullmatch(line)
        if anchor_match is not None:
            anchor = anchor_match.group(1)
            anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
            anchor_positions.append((line_number, anchor))
    detail_sections = {}
    for position, (start, anchor) in enumerate(anchor_positions):
        if position + 1 < len(anchor_positions):
            end = anchor_positions[position + 1][0]
        else:
            end = closed_position
        end = min(end, closed_position)
        detail_sections.setdefault(anchor, []).append(lines[start + 1:end])
    seen_index_anchors = set()
    for line in lines:
        if daemon.OPEN_BACKLOG_CANDIDATE_RE.match(line) is None:
            continue
        match = daemon.OPEN_BACKLOG_TICKET_RE.fullmatch(line)
        if match is None:
            counts["unclassified"] += 1
            continue
        anchor = match.group(4)
        if (match.group(1) == "CRITICAL"
                and match.group(2) != "BUG FIX"):
            counts["unclassified"] += 1
            continue
        if anchor in seen_index_anchors or anchor_counts.get(anchor) != 1:
            counts["unclassified"] += 1
            seen_index_anchors.add(anchor)
            continue
        seen_index_anchors.add(anchor)
        severity = match.group(1).lower()
        sections = detail_sections.get(anchor, [])
        if len(sections) != 1:
            counts["unclassified"] += 1
            continue
        reopen_candidates = [
            detail_line for detail_line in sections[0]
            if daemon.BACKLOG_REOPEN_COUNT_CANDIDATE_RE.search(detail_line)
            is not None]
        if len(reopen_candidates) != 1:
            counts["unclassified"] += 1
            continue
        reopen_match = daemon.BACKLOG_REOPEN_COUNT_RE.fullmatch(
            reopen_candidates[0])
        if reopen_match is None:
            counts["unclassified"] += 1
            continue
        reopening_candidates = [
            detail_line for detail_line in sections[0]
            if daemon.BACKLOG_REOPENING_CANDIDATE_RE.search(detail_line)
            is not None]
        if (len(reopening_candidates) != 1
                or daemon.BACKLOG_REOPENING_RE.fullmatch(
                    reopening_candidates[0]) is None):
            counts["unclassified"] += 1
            continue
        if "barred by Architect NO-GO" in reopening_candidates[0]:
            # A barred ticket is final and therefore may not be indexed Open.
            counts["unclassified"] += 1
            continue
        reopen_count = reopen_match.group(1)
        reopened_more_than_five = (
            len(reopen_count) > 1 or reopen_count > "5")
        if reopened_more_than_five and severity != "low":
            counts["unclassified"] += 1
            continue
        ticket_type = match.group(2).lower().replace(" ", "_")
        counts[severity] += 1
        if severity == "high":
            counts["high_" + ticket_type] += 1
    return counts


def eligible_fix_only_bug_anchors(minimum_severity=None):
    """Return the Open bug anchors that fix-only maintenance may choose.

    Fix-only maintenance repairs an existing open bug instead of
    starting new work. A ticket qualifies when its index line is
    classified BUG FIX at the requested severity or more severe. The
    whole backlog must be clean first: any read problem or
    unclassified open line refuses the query instead of returning a
    partial list a caller might trust.

    Arguments:
      minimum_severity = least severe class to include (``"high"``,
                         ``"medium"``, or ``"low"``), or ``None`` for
                         the configured discovery severity.

    Returns:
      The anchors of every qualifying open bug ticket, in backlog
      order.

    Raises:
      daemon.TicketCycleStateError: for an invalid severity or an
        unreadable or unclassified backlog.
    """
    minimum = (daemon.DISCOVERY_SEVERITY if minimum_severity is None
               else minimum_severity)
    if minimum not in daemon.DISCOVERY_SEVERITIES:
        raise daemon.TicketCycleStateError("invalid severity")
    counts = daemon.backlog_severity_counts()
    problem = counts["problem"]
    if problem is None and counts["unclassified"]:
        problem = "unclassified Open backlog work"
    if problem is not None:
        raise daemon.TicketCycleStateError(problem)
    lines, problem = daemon.verified_backlog_lines()
    if problem is not None:
        raise daemon.TicketCycleStateError(problem)
    order = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    allowed = order[:order.index(minimum.upper()) + 1]
    return [
        match.group(4) for line in lines
        for match in [daemon.OPEN_BACKLOG_TICKET_RE.fullmatch(line)]
        if (match is not None and match.group(1) in allowed
            and match.group(2) == "BUG FIX")
    ]


def verified_backlog_lines():
    """Read one stable, regular UTF-8 backlog or return a plain problem.

    The read defends against a file swapped mid-read: the backlog must
    be an ordinary file (not a symbolic link or other redirect), stay
    within the size limit, and keep the same identity, size, and
    timestamps before, during, and after the read. Otherwise the
    reader reports a problem instead of returning half-updated lines.

    Returns:
      ``(lines, None)`` with the backlog split into lines, or
      ``(None, problem)`` where problem is a printable sentence naming
      what failed and, when the user can fix it, what to do.
    """
    try:
        initial = daemon.os.lstat(daemon.BACKLOG_LEDGER)
    except FileNotFoundError:
        return None, (
            "ai/notes/backlog.md is missing; restore the tracked file from "
            "the current main branch before ticket dispatch")
    except OSError as exc:
        return None, "cannot inspect ai/notes/backlog.md: " + str(exc)
    if daemon.stat.S_ISLNK(initial.st_mode) or not daemon.stat.S_ISREG(initial.st_mode):
        return None, (
            "ai/notes/backlog.md must be one ordinary file, not a redirect "
            "or special file")
    if initial.st_size > daemon.MAX_BACKLOG_LEDGER_BYTES:
        return None, "ai/notes/backlog.md exceeds the safe read limit"
    flags = daemon.os.O_RDONLY | daemon.os.O_NONBLOCK
    if hasattr(daemon.os, "O_NOFOLLOW"):
        flags |= daemon.os.O_NOFOLLOW
    try:
        descriptor = daemon.os.open(daemon.BACKLOG_LEDGER, flags)
    except OSError as exc:
        return None, "cannot open ai/notes/backlog.md: " + str(exc)
    try:
        opened = daemon.os.fstat(descriptor)
        if (not daemon.stat.S_ISREG(opened.st_mode)
                or (initial.st_dev, initial.st_ino)
                != (opened.st_dev, opened.st_ino)):
            return None, "ai/notes/backlog.md changed while being opened"
        chunks = []
        size = 0
        while True:
            chunk = daemon.os.read(descriptor, 65536)
            if not chunk:
                break
            size += len(chunk)
            if size > daemon.MAX_BACKLOG_LEDGER_BYTES:
                return None, "ai/notes/backlog.md exceeds the safe read limit"
            chunks.append(chunk)
        after = daemon.os.fstat(descriptor)
        current = daemon.os.lstat(daemon.BACKLOG_LEDGER)
        if ((opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
                or (after.st_dev, after.st_ino)
                != (current.st_dev, current.st_ino)
                or opened.st_size != after.st_size
                or opened.st_mtime_ns != after.st_mtime_ns
                or opened.st_ctime_ns != after.st_ctime_ns
                or after.st_size != size):
            return None, "ai/notes/backlog.md changed while being read"
    except OSError as exc:
        return None, "cannot verify ai/notes/backlog.md: " + str(exc)
    finally:
        daemon.os.close(descriptor)
    try:
        text = b"".join(chunks).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return None, "ai/notes/backlog.md is not valid UTF-8: " + str(exc)
    return text.splitlines(), None


def backlog_reopening_status(ticket_anchor):
    """Return ``allowed`` or the Architect's permanent ``barred`` decision.

    The ticket's section is bounded at the next detail anchor or at
    the Closed-tickets heading — not at ordinary ``##`` headings — so
    prose after an anchor-less heading still belongs to the preceding
    ticket's section and can make its record ambiguous.

    Arguments:
      ticket_anchor = the ticket's anchor: the part of a cycle
                      identifier before ``@``.

    Returns:
      ``"allowed"``, ``"barred by Architect NO-GO"``, or ``None`` for
      a missing, duplicate, or malformed record — callers must not
      invent permission from incomplete backlog prose.
    """
    if (not isinstance(ticket_anchor, str)
            or daemon.REDTEAM_REVIEW_TICKET_RE.fullmatch(ticket_anchor) is None):
        return None
    lines, problem = daemon.verified_backlog_lines()
    if problem is not None:
        return None
    starts = [index for index, line in enumerate(lines)
              if line == '<a id="' + ticket_anchor + '"></a>']
    if len(starts) != 1:
        return None
    start = starts[0] + 1
    end = next((index for index in range(start, len(lines))
                if daemon.BACKLOG_DETAIL_ANCHOR_RE.fullmatch(lines[index])
                is not None or lines[index] == "# Closed tickets"),
               len(lines))
    candidates = [line for line in lines[start:end]
                  if daemon.BACKLOG_REOPENING_CANDIDATE_RE.search(line) is not None]
    if len(candidates) != 1:
        return None
    match = daemon.BACKLOG_REOPENING_RE.fullmatch(candidates[0])
    return match.group(1) if match is not None else None


def discovery_admission_count():
    """Return open Critical, High, and Medium tickets; Low does not count.

    Low tickets are excluded so a backlog of deferred small work does
    not by itself count as open demand.

    Returns:
      The summed count of open Critical, High, and Medium tickets.
    """
    counts = daemon.backlog_severity_counts()
    return counts["critical"] + counts["high"] + counts["medium"]
