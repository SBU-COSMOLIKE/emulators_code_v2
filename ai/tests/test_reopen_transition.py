"""Focused checks for the compact Architect reopening contract."""

import unittest

from ai.tools import reopen_transition


ANCHOR = "open-example-ticket"
CYCLE = ANCHOR + "@" + "1" * 40
LANDING = "2" * 40


def ticket_lines(*, state="CLOSED", severity="HIGH", count=0,
                 reopening="allowed", following_heading=False):
    """Build one human-readable backlog ticket in Open or Closed position."""
    index = []
    if state == "OPEN":
        index = [
            "- OPEN **" + severity + "** **BUG FIX** — "
            "[Example ticket](#" + ANCHOR + ")",
        ]
    detail = [
        '<a id="' + ANCHOR + '"></a>',
        "## Example ticket",
        "",
        "### High-level summary",
        "",
        "A concrete problem.",
        "",
        "### Current status",
        "",
        "**Ticket type: BUG FIX.**",
        "",
        "**Red Team reopen count: " + str(count) + ".**",
        "",
        "**Red Team reopening: " + reopening + ".**",
        "",
        "**" + state + ".** Current result.",
        "",
        "**Severity: " + severity + ".** Concrete consequence.",
        "",
        "### What is already fixed",
        "",
        "The first repair.",
        "",
        "### What is missing",
        "",
        "Nothing for this ticket." if state == "CLOSED" else "One repair.",
    ]
    lines = (index + detail + ["# Closed tickets"] if state == "OPEN"
             else ["# Closed tickets"] + detail)
    if following_heading:
        lines += ["", "## Another group", "", "**Severity: LOW.**"]
    return lines


class ReopenTransitionTests(unittest.TestCase):
    """Keep mechanical checks separate from the Architect's judgment."""

    def test_brief_names_the_two_exact_outcomes(self):
        ticket = reopen_transition.inspect_backlog(
            ticket_lines(following_heading=True), ANCHOR)
        brief = reopen_transition.architect_brief(
            ticket=ticket, cycle=CYCLE, landing=LANDING)
        self.assertEqual(ticket.severity, "HIGH")
        self.assertEqual(ticket.count, 0)
        self.assertIn("GO: count 1, Open at HIGH", brief)
        self.assertIn("NO-GO: count 1, remain Closed", brief)
        self.assertNotIn("Another group", brief)

    def test_redteam_brief_supplies_bookkeeping_without_judgment(self):
        ticket = reopen_transition.inspect_backlog(ticket_lines(), ANCHOR)
        brief = reopen_transition.redteam_brief(
            ticket=ticket, cycle=CYCLE, landing=LANDING)
        self.assertIn("Ticket: Example ticket", brief)
        self.assertIn("Severity: HIGH", brief)
        self.assertIn("Reopen count: 0", brief)
        self.assertIn("NO CHANGE", brief)
        self.assertIn("REOPEN only with concrete, persuasive evidence", brief)
        self.assertNotIn("GO", brief)

        open_ticket = reopen_transition.inspect_backlog(
            ticket_lines(state="OPEN"), ANCHOR)
        with self.assertRaisesRegex(
                reopen_transition.ReopenTransitionError,
                "requires a Closed ticket"):
            reopen_transition.redteam_brief(
                ticket=open_ticket, cycle=CYCLE, landing=LANDING)

    def test_go_preserves_severity_and_increments_once(self):
        before = reopen_transition.inspect_backlog(ticket_lines(), ANCHOR)
        after = reopen_transition.inspect_backlog(
            ticket_lines(state="OPEN", count=1), ANCHOR)
        self.assertEqual(
            reopen_transition.validate_after(before, after), "GO")

        wrong_severity = reopen_transition.inspect_backlog(
            ticket_lines(state="OPEN", severity="MEDIUM", count=1), ANCHOR)
        with self.assertRaisesRegex(
                reopen_transition.ReopenTransitionError,
                "exact GO or NO-GO"):
            reopen_transition.validate_after(before, wrong_severity)

    def test_no_go_closes_bars_and_increments_once(self):
        before = reopen_transition.inspect_backlog(ticket_lines(), ANCHOR)
        after = reopen_transition.inspect_backlog(ticket_lines(
            count=1, reopening="barred by Architect NO-GO"), ANCHOR)
        self.assertEqual(
            reopen_transition.validate_after(before, after), "NO-GO")

        no_increment = reopen_transition.inspect_backlog(ticket_lines(
            count=0, reopening="barred by Architect NO-GO"), ANCHOR)
        with self.assertRaisesRegex(
                reopen_transition.ReopenTransitionError,
                "increment.*exactly once"):
            reopen_transition.validate_after(before, no_increment)

    def test_sixth_accepted_reopening_uses_existing_low_rule(self):
        before = reopen_transition.inspect_backlog(
            ticket_lines(count=5), ANCHOR)
        after = reopen_transition.inspect_backlog(
            ticket_lines(state="OPEN", severity="LOW", count=6), ANCHOR)
        self.assertEqual(
            reopen_transition.validate_after(before, after), "GO")

        closed = reopen_transition.inspect_backlog(ticket_lines(
            severity="LOW", count=6,
            reopening="barred by Architect NO-GO"), ANCHOR)
        self.assertEqual(
            reopen_transition.validate_after(before, closed), "NO-GO")

    def test_restart_accepts_an_already_recorded_decision_without_repeating_it(self):
        go = reopen_transition.inspect_backlog(
            ticket_lines(state="OPEN", count=1), ANCHOR)
        no_go = reopen_transition.inspect_backlog(ticket_lines(
            count=1, reopening="barred by Architect NO-GO"), ANCHOR)
        self.assertEqual(reopen_transition.validate_after(go, go), "GO")
        self.assertEqual(
            reopen_transition.validate_after(no_go, no_go), "NO-GO")


if __name__ == "__main__":
    unittest.main()
