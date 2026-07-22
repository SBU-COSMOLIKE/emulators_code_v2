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
    """Return the provider-safe default for one Implementer shell."""
    if provider == "claude":
        return daemon.DEFAULT_IMPLEMENTER_CONTEXT_BUDGET
    if provider == "ollama":
        return daemon.DEFAULT_OLLAMA_IMPLEMENTER_CONTEXT_BUDGET
    raise ValueError("Implementer provider must be claude or ollama")


def implementer_runtime_record(
        *, provider, model, context_limit, compaction_limit):
    """Return one validated runtime identity for the stable ``opus`` role."""
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
    """Build the canonical runtime, verifying Ollama before a live watch."""
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
    """Return the exact command and label for one lower-cost review turn."""
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
    """Return the Implementer's time and context checkpoint hooks."""
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
    """Return the independent Claude Code limit for one Claude-backed role."""
    if agent == "fable":
        return daemon.ARCHITECT_CONTEXT_BUDGET
    if agent == "opus":
        return daemon.IMPLEMENTER_RUNTIME["compaction_limit"]
    raise ValueError("Claude compaction limit has no role " + repr(agent))


def mailbox_lane_cwd(agent):
    """Return a serialization identity for an AI route or local daemon lane."""
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
    """Return eligible Open bug anchors for this severity."""
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
    """Read one stable, regular UTF-8 backlog or return a plain problem."""
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

    ``ticket_anchor`` is the part of a cycle identifier before ``@``. A
    missing, duplicate, or malformed record returns ``None`` so callers do not
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
    """Return open Critical, High, and Medium tickets; Low does not count."""
    counts = daemon.backlog_severity_counts()
    return counts["critical"] + counts["high"] + counts["medium"]
