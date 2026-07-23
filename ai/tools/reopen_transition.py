#!/usr/bin/env python3
"""Check one Red Team reopening without making the decision.

A reopening is the Red Team's formal request to reopen a closed ticket
because it believes the accepted fix still leaves the ticket's bug. The
backlog is ``ai/notes/backlog.md``, the ticket dashboard. The mailbox
daemon reads and seals files; this module only interprets the already
verified backlog lines. It gives the Architect a short statement of the
current ticket state and checks the exact state left after GO (the
ticket reopens) or NO-GO (it stays closed and further requests are
barred). It never edits the backlog and never decides whether the Red
Team is right.
"""

from dataclasses import dataclass
import re


OPEN_LINE_RE = re.compile(
    r"^- OPEN \*\*(CRITICAL|HIGH|MEDIUM|LOW)\*\* "
    r"\*\*(BUG FIX|NEW FUNCTIONALITY)\*\* — "
    r"\[([^]\r\n]+)\]\(#([a-z0-9]+(?:-[a-z0-9]+)*)\)$")
COUNT_RE = re.compile(
    r"^\*\*Red Team reopen count: (0|[1-9][0-9]*)\.\*\*$")
REOPENING_RE = re.compile(
    r"^\*\*Red Team reopening: "
    r"(allowed|barred by Architect NO-GO)\.\*\*$")
SEVERITY_RE = re.compile(
    r"^\*\*Severity: (CRITICAL|HIGH|MEDIUM|LOW)\.\*\*(?: .*)?$")


class ReopenTransitionError(ValueError):
    """A backlog cannot prove one unambiguous reopening state."""


@dataclass(frozen=True)
class ReopenTicket:
    """The small set of ticket facts needed for a reopening decision.

    Arguments:
      anchor    = the ticket's stable link name: the backlog marks each
                  ticket with an HTML ``<a id="...">`` marker so links
                  and cycles can name it without depending on wording.
      title     = human ticket title from the ``## `` heading after the
                  anchor.
      severity  = current CRITICAL, HIGH, MEDIUM, or LOW classification.
      count     = number of formal Red Team reopening requests decided
                  so far.
      state     = ``"OPEN"`` or ``"CLOSED"``.
      reopening = ``"allowed"``, or ``"barred by Architect NO-GO"`` when
                  an earlier NO-GO permanently barred another request.
    """

    anchor: str
    title: str
    severity: str
    count: int
    state: str
    reopening: str


def _one_match(lines, pattern, label):
    """Return one regular-expression match or reject ambiguous prose.

    Arguments:
      lines   = the ticket's section lines.
      pattern = compiled regular expression; a line must match it in
                full to count.
      label   = short human name of the required line, used in the
                error message.

    Returns:
      The single match object.

    Raises:
      ReopenTransitionError: when zero or several lines match, because
        a decision must never rest on an ambiguous ticket record.
    """
    matches = [match for line in lines
               if (match := pattern.fullmatch(line)) is not None]
    if len(matches) != 1:
        raise ReopenTransitionError(
            "ticket must contain exactly one " + label)
    return matches[0]


def inspect_backlog(lines, anchor):
    """Return the exact reopening facts for one backlog ticket.

    The ticket's section runs from its ``<a id="...">`` anchor marker
    to the next ``## `` heading or anchor, so another ticket cannot
    lend this ticket a severity, counter, or reopening state. The
    section must contain exactly one severity line, one reopen-count
    line, one reopening-permission line, and one status line. Whether
    the ticket is open is decided by position — a section above the
    ``# Closed tickets`` heading is open — and cross-checked against
    the ``- OPEN`` index lines at the top of the backlog: an open
    ticket must appear there exactly once with the same severity, a
    closed one not at all.

    Arguments:
      lines  = backlog lines from the daemon's stable backlog reader.
      anchor = the ticket's anchor name (lowercase words joined by
               hyphens).

    Returns:
      A ReopenTicket carrying the checked facts.

    Raises:
      ReopenTransitionError: when the anchor is malformed, missing, or
        duplicated, or any required line is absent, doubled, or
        inconsistent with the index.
    """
    if not isinstance(anchor, str) or re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", anchor) is None:
        raise ReopenTransitionError("ticket anchor is invalid")
    lines = tuple(lines)
    marker = '<a id="' + anchor + '"></a>'
    if lines.count(marker) != 1 or lines.count("# Closed tickets") != 1:
        raise ReopenTransitionError(
            "backlog must contain one ticket anchor and one Closed heading")
    start = lines.index(marker) + 1
    if start >= len(lines) or not lines[start].startswith("## "):
        raise ReopenTransitionError("ticket anchor is not followed by a title")
    end = next(
        (index for index in range(start + 1, len(lines))
         if lines[index].startswith(("## ", '<a id="'))),
        len(lines))
    section = lines[start:end]
    title = section[0][3:].strip()
    if not title:
        raise ReopenTransitionError("ticket title is empty")

    count = int(_one_match(
        section, COUNT_RE, "Red Team reopen count").group(1))
    reopening = _one_match(
        section, REOPENING_RE, "Red Team reopening state").group(1)
    severity = _one_match(section, SEVERITY_RE, "severity").group(1)

    open_lines = [match for line in lines
                  if (match := OPEN_LINE_RE.fullmatch(line)) is not None
                  and match.group(4) == anchor]
    closed_heading = lines.index("# Closed tickets")
    if start - 1 < closed_heading:
        if len(open_lines) != 1 or open_lines[0].group(1) != severity:
            raise ReopenTransitionError(
                "Open ticket index and detail severity disagree")
        state = "OPEN"
        status_prefix = "**OPEN.**"
    else:
        if open_lines:
            raise ReopenTransitionError(
                "Closed ticket still appears in the Open index")
        state = "CLOSED"
        status_prefix = "**CLOSED.**"
    if sum(line.startswith(status_prefix) for line in section) != 1:
        raise ReopenTransitionError(
            "ticket must contain exactly one " + state + " status")
    return ReopenTicket(
        anchor=anchor, title=title, severity=severity, count=count,
        state=state, reopening=reopening)


def expected_go_severity(ticket):
    """Return the severity the ticket carries after this decision.

    A ticket that has already consumed five formal reopening decisions
    drops to LOW on every later one, so an endlessly re-litigated
    ticket cannot keep claiming high priority. Before that point the
    ticket keeps its current severity. Both outcomes of the decision —
    reopen or stay closed — leave this same severity.

    Arguments:
      ticket = the checked ReopenTicket before the decision.

    Returns:
      ``"LOW"`` when this decision is the sixth or later, otherwise
      the ticket's current severity.
    """
    return "LOW" if ticket.count + 1 > 5 else ticket.severity


def architect_brief(ticket, cycle, landing):
    """Render the compact mechanical facts shown before Architect reasoning.

    The brief opens with the checked ticket facts, then states what the
    Architect may legally do. A ticket that is closed with reopening
    allowed is awaiting the decision, so the brief lists both outcomes
    with the exact count and severity each must leave. The two other
    legal states are recovery states: the backlog already records the
    decision (open again after GO, or closed and barred after NO-GO),
    so the brief says only not to repeat the edit.

    Arguments:
      ticket  = the checked ReopenTicket.
      cycle   = the mailbox cycle name shown in the brief.
      landing = identifier of the accepted landing the Red Team
                reviewed; a landing is a commit the daemon recorded as
                an accepted delivery.

    Returns:
      The brief as one printable string.

    Raises:
      ReopenTransitionError: when the ticket state matches no legal
        reopening situation, such as open but barred.
    """
    heading = (
        "ARCHITECT REOPENING CHECK\n\n"
        "Ticket: " + ticket.title + "\n"
        "Cycle: " + cycle + "\n"
        "Reviewed landing: " + landing + "\n"
        "Current state: " + ticket.state + "\n"
        "Current severity: " + ticket.severity + "\n"
        "Current reopen count: " + str(ticket.count) + "\n"
        "Reopening permission: " + ticket.reopening + "\n\n")
    if ticket.state == "CLOSED" and ticket.reopening == "allowed":
        return heading + (
            "Allowed outcomes:\n"
            "- GO: count " + str(ticket.count + 1) + ", Open at "
            + expected_go_severity(ticket) + ", reopening allowed.\n"
            "- NO-GO: count " + str(ticket.count + 1)
            + ", remain Closed at " + expected_go_severity(ticket)
            + ", reopening barred.\n\n"
            "Assess only the Red Team evidence. After editing and sealing "
            "the backlog, the daemon will check these facts. Do not dispatch "
            "an Implementer in this turn.\n\n")
    if ticket.state == "OPEN" and ticket.reopening == "allowed":
        return heading + (
            "Recovery state: the backlog already records GO. Do not repeat "
            "the edit or increment the count.\n\n")
    if ticket.state == "CLOSED" and ticket.reopening == (
            "barred by Architect NO-GO"):
        return heading + (
            "Recovery state: the backlog already records NO-GO. Do not "
            "repeat the edit or increment the count.\n\n")
    raise ReopenTransitionError("ticket has no legal reopening action")


def redteam_brief(ticket, cycle, landing):
    """Render the checked backlog facts needed for one closure review.

    A closure review is the Red Team's look at a closed ticket's
    accepted landing: it may report NO CHANGE or ask to reopen. The
    brief states which of those outcomes are available — only NO
    CHANGE when an earlier Architect NO-GO permanently barred
    reopening — and reminds the reviewer that the backlog is not
    theirs to edit.

    Arguments:
      ticket  = the checked ReopenTicket; must be CLOSED.
      cycle   = the mailbox cycle name shown in the brief.
      landing = identifier of the accepted landing under review.

    Returns:
      The brief as one printable string.

    Raises:
      ReopenTransitionError: when the ticket is not closed or its
        reopening field holds an unknown value.
    """
    if ticket.state != "CLOSED":
        raise ReopenTransitionError(
            "Red Team closure review requires a Closed ticket")
    heading = (
        "RED TEAM CLOSURE CHECK\n\n"
        "Ticket: " + ticket.title + "\n"
        "Cycle: " + cycle + "\n"
        "Reviewed landing: " + landing + "\n"
        "Severity: " + ticket.severity + "\n"
        "Reopen count: " + str(ticket.count) + "\n"
        "Reopening permission: " + ticket.reopening + "\n\n")
    if ticket.reopening == "allowed":
        return heading + (
            "Allowed outcomes:\n"
            "- NO CHANGE when no concrete bug remains.\n"
            "- REOPEN only with concrete, persuasive evidence that the "
            "named landing still leaves this ticket's bug.\n\n"
            "The daemon supplied the ticket bookkeeping. Review the exact "
            "landing and do not edit the backlog.\n\n")
    if ticket.reopening == "barred by Architect NO-GO":
        return heading + (
            "Allowed outcome: NO CHANGE. An earlier Architect NO-GO "
            "permanently barred another reopening.\n\n")
    raise ReopenTransitionError("ticket has no legal closure-review action")


def validate_after(before, after):
    """Return GO or NO-GO only when the Architect left an exact legal state.

    The comparison is deliberately strict. The ticket identity (anchor
    and title) must not change. If the before state already records a
    decision — a recovery state — the after state must equal it
    exactly. A ticket that was awaiting a decision must gain exactly
    one reopen count and land in exactly one of the two legal end
    states: open with reopening allowed (GO) or closed with reopening
    barred (NO-GO), both at the post-decision severity. Anything else
    is refused rather than guessed at.

    Arguments:
      before = ReopenTicket checked before the Architect's edit.
      after  = ReopenTicket checked after the Architect's edit.

    Returns:
      ``"GO"`` when the ticket reopened, ``"NO-GO"`` when it stayed
      closed and barred.

    Raises:
      ReopenTransitionError: for a changed identity, a re-edited
        recovery state, a wrong count, or an end state that matches
        neither decision.
    """
    if not isinstance(before, ReopenTicket) or not isinstance(
            after, ReopenTicket):
        raise ReopenTransitionError(
            "reopening comparison requires two checked ticket states")
    if before.anchor != after.anchor or before.title != after.title:
        raise ReopenTransitionError(
            "Architect reopening edit changed the ticket identity")
    if before.state == "OPEN" and before.reopening == "allowed":
        if after == before:
            return "GO"
        raise ReopenTransitionError("recovered GO state changed again")
    if (before.state == "CLOSED"
            and before.reopening == "barred by Architect NO-GO"):
        if after == before:
            return "NO-GO"
        raise ReopenTransitionError("recovered NO-GO state changed again")
    if before.state != "CLOSED" or before.reopening != "allowed":
        raise ReopenTransitionError("ticket was not awaiting a decision")
    if after.count != before.count + 1:
        raise ReopenTransitionError(
            "Architect decision must increment the reopen count exactly once")
    if (after.state == "OPEN" and after.reopening == "allowed"
            and after.severity == expected_go_severity(before)):
        return "GO"
    if (after.state == "CLOSED"
            and after.reopening == "barred by Architect NO-GO"
            and after.severity == expected_go_severity(before)):
        return "NO-GO"
    raise ReopenTransitionError(
        "Architect decision did not leave an exact GO or NO-GO state")
