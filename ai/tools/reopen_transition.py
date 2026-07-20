#!/usr/bin/env python3
"""Check one Red Team reopening without making the decision.

The mailbox daemon reads and seals files.  This module only interprets the
already verified backlog lines.  It gives the Architect a short statement of
the current ticket state and checks the exact state left after GO or NO-GO.
It never edits the backlog and never decides whether the Red Team is right.
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
      anchor: Stable backlog link carried by the ticket cycle.
      title: Human ticket title following the anchor.
      severity: Current Critical, High, Medium, or Low classification.
      count: Number of earlier formal Red Team reopening requests.
      state: ``OPEN`` or ``CLOSED``.
      reopening: ``allowed`` or the permanent Architect NO-GO bar.
    """

    anchor: str
    title: str
    severity: str
    count: int
    state: str
    reopening: str


def _one_match(lines, pattern, label):
    """Return one regular-expression match or reject ambiguous prose."""
    matches = [match for line in lines
               if (match := pattern.fullmatch(line)) is not None]
    if len(matches) != 1:
        raise ReopenTransitionError(
            "ticket must contain exactly one " + label)
    return matches[0]


def inspect_backlog(lines, anchor):
    """Return the exact reopening facts for one backlog ticket.

    ``lines`` must come from the daemon's stable backlog reader.  The parser
    stops at the next ticket heading or anchor, so another ticket cannot lend
    this ticket a severity, counter, or reopening state.
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
    """Apply the existing sixth-reopening Low rule to an accepted report."""
    return "LOW" if ticket.count + 1 > 5 else ticket.severity


def architect_brief(ticket, cycle, landing):
    """Render the compact mechanical facts shown before Architect reasoning."""
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
    """Render the checked backlog facts needed for one closure review."""
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
    """Return GO or NO-GO only when the Architect left an exact legal state."""
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
