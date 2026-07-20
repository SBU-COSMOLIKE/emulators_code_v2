#!/usr/bin/env python3
"""Select cheaper commands for routine reviews without changing authority.

The mailbox daemon decides what kind of turn is being dispatched. This module
only identifies the routine review kinds and replaces the command's reasoning
effort. It cannot read a mailbox, inspect evidence, or decide a ticket.
"""


REVIEW_EFFORTS = ("low", "medium", "high", "xhigh")


def review_kind(
        *, agent, ticket_kind=None, candidate_audit=False,
        reopening=False, checkpoint=False, integration=False):
    """Name a routine review, or return ``None`` for full-effort work."""
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
    """Return one Claude or Codex review command at the chosen effort."""
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
