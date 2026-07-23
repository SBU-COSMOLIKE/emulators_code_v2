#!/usr/bin/env python3
"""Select cheaper commands for routine reviews without changing authority.

Reasoning effort is the launch setting that tells a Claude or Codex
command how much internal deliberation to spend on a turn; higher effort
costs more money and time. A routine review is a bounded judgment call —
approving a checkpoint, auditing a candidate, deciding whether a closed
ticket reopens — that does not need the full effort reserved for design
work. The mailbox daemon (the watcher program that runs the development
roles in turn) decides what kind of turn is being dispatched; this
module only identifies the routine review kinds and replaces the
command's effort option. It cannot read a mailbox, inspect evidence, or
decide a ticket.
"""


REVIEW_EFFORTS = ("low", "medium", "high", "xhigh")


def review_kind(
        *, agent, ticket_kind=None, candidate_audit=False,
        reopening=False, checkpoint=False, integration=False):
    """Name a routine review, or return ``None`` for full-effort work.

    Arguments:
      agent           = the daemon's identifier for the command being
                        launched: ``"fable"`` runs the Architect and
                        ``"sol"`` runs the Red Team; any other value is
                        never a routine review.
      ticket_kind     = the ticket's declared kind, read only for a Red
                        Team turn: ``"closure"`` and ``"control-plane"``
                        reviews are routine.
      candidate_audit = True when the Architect is auditing an
                        Implementer candidate.
      reopening       = True when the Architect is deciding whether a
                        closed ticket may reopen.
      checkpoint      = True when the Architect is reviewing a mid-ticket
                        checkpoint (a partial delivery).
      integration     = True when the Architect is revalidating a
                        candidate after the main branch moved under it.

    Returns:
      A short display name for the routine review, or ``None`` when the
      turn keeps its full effort. When several Architect flags are set,
      precedence is reopening, then integration, then checkpoint, then
      candidate audit.
    """
    if agent == "sol" and ticket_kind == "closure":
        return "Red Team closure"
    if agent == "sol" and ticket_kind == "control-plane":
        return "Red Team control-plane review"
    if agent != "fable":
        return None
    if reopening:
        return "Architect reopening decision"
    if integration:
        return "Architect integration revalidation"
    if checkpoint:
        return "Architect checkpoint review"
    if candidate_audit:
        return "Architect candidate audit"
    return None


def command_with_effort(command, *, agent, effort):
    """Return one Claude or Codex review command at the chosen effort.

    The command is a sequence of argument strings of the kind handed to
    the operating system. The Architect command carries its effort as
    the value after an ``--effort`` option; the Red Team command carries
    it inside one ``model_reasoning_effort=<value>`` argument. The
    function refuses a command whose effort option is absent or appears
    more than once rather than guessing which argument to replace.

    Arguments:
      command = the launch command as a sequence of argument strings;
                it is copied, never edited in place.
      agent   = ``"fable"`` for the Architect command or ``"sol"`` for
                the Red Team command.
      effort  = one of ``REVIEW_EFFORTS``: ``"low"``, ``"medium"``,
                ``"high"``, or ``"xhigh"``.

    Returns:
      A new list equal to ``command`` with only the effort replaced.

    Raises:
      ValueError: when the effort is not a known level, when the command
        lacks exactly one effort option to replace, or when the agent is
        neither the Architect's nor the Red Team's.
    """
    if effort not in REVIEW_EFFORTS:
        raise ValueError("review effort must be low, medium, high, or xhigh")
    updated = list(command)
    if agent == "fable":
        positions = [index for index, item in enumerate(updated)
                     if item == "--effort"]
        if len(positions) != 1 or positions[0] + 1 >= len(updated):
            raise ValueError("Architect command has no exact effort option")
        updated[positions[0] + 1] = effort
        return updated
    if agent == "sol":
        prefix = "model_reasoning_effort="
        positions = [index for index, item in enumerate(updated)
                     if item.startswith(prefix)]
        if len(positions) != 1:
            raise ValueError("Red Team command has no exact effort option")
        updated[positions[0]] = prefix + effort
        return updated
    raise ValueError("routine review effort applies only to Architect or Sol")
