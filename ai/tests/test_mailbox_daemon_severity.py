"""Focused discovery-severity protocol and propagation tests."""

import contextlib
import io
import os
import pathlib
import unittest
from unittest import mock

from ai.tools import backlog_guard
from ai.tests.tools_mailbox_daemon_fix_only_repro import captured_dispatch
from ai.tests.tools_mailbox_daemon_fix_only_repro import captured_send
from ai.tests.tools_mailbox_daemon_fix_only_repro import DAEMON_PATH
from ai.tests.tools_mailbox_daemon_fix_only_repro import read_text_exact
from ai.tests.tools_mailbox_daemon_fix_only_repro import run_main
from ai.tests.tools_mailbox_daemon_fix_only_repro import scratch_daemon
from ai.tests.tools_mailbox_daemon_fix_only_repro import tree_snapshot
from ai.tests.tools_mailbox_daemon_fix_only_repro import write_pending


TICKET = "MAILBOX-TICKET: "
SEVERITY = "MAILBOX-SEVERITY: "
SCOPE = "MAILBOX-SCOPE: "
BASE_COMMIT = "1" * 40


def ticket_flow(text, mode="normal",
                anchor="scratch-high-bug-fix-1"):
    """Build one syntactically current ticket exchange for policy tests."""
    return (
        "MAILBOX-FLOW: ticket\n"
        "MAILBOX-CYCLE: " + anchor + "@" + BASE_COMMIT + "\n"
        "MAILBOX-MODE: " + mode + "\n\n" + text + "\n")


def second_implementer_payload():
    """Build a current Sol second-Implementer assignment envelope."""
    return (
        TICKET + "closure\n"
        + ticket_flow(
            "### ARCHITECT_HANDOFF\n\n"
            "OpenAI Sol — this is a role as second Implementer for this "
            "unit.\nImplement the named repair.",
            mode="emergency-second"))


class MailboxDiscoverySeverityTests(unittest.TestCase):
    """Pin the saved user setting and the two thinking-role decisions."""

    def test_send_saves_default_and_each_explicit_value(self):
        for supplied, expected in (
                (None, "medium"),
                ("high", "high"),
                ("medium", "medium"),
                ("low", "low")):
            with self.subTest(supplied=supplied), \
                    mock.patch.dict(os.environ, {}, clear=True), \
                    scratch_daemon() as (daemon, _, _, _):
                outcome, _ = captured_send(
                    daemon, agent="sol", text="review the named change",
                    dry_run=False, ticket_kind="discovery",
                    severity=supplied)
                pending = [pathlib.Path(path)
                           for path in daemon.pending_messages()]
                self.assertTrue(outcome)
                self.assertEqual(len(pending), 1)
                self.assertEqual(
                    read_text_exact(pending[0]),
                    TICKET + "discovery\n" + SEVERITY + expected
                    + "\n" + SCOPE
                    + "bounded\n\nreview the named change\n")

    def test_discovery_requires_saved_severity_and_scope_and_keeps_body(self):
        with scratch_daemon() as (daemon, _, _, _):
            legacy = TICKET + "discovery\r\n\r\nlegacy body\r\n"
            self.assertIsNone(daemon.sol_discovery_severity(message=legacy))
            self.assertIsNone(daemon.sol_discovery_scope(message=legacy))
            self.assertIsNotNone(
                daemon.sol_discovery_severity_problem(message=legacy))
            self.assertEqual(
                daemon.sol_ticket_body(message=legacy),
                "\r\nlegacy body\r\n")

            current = (TICKET + "discovery\r\n" + SEVERITY
                       + "low\r\n" + SCOPE
                       + "bounded\r\n\r\ncurrent body\r\n")
            self.assertEqual(
                daemon.sol_discovery_severity(message=current), "low")
            self.assertEqual(
                daemon.sol_discovery_scope(message=current), "bounded")
            self.assertEqual(
                daemon.sol_ticket_body(message=current),
                "\r\ncurrent body\r\n")
            assignment = second_implementer_payload()
            self.assertTrue(
                daemon.sol_second_implementer_assignment(message=assignment))

    def test_malformed_or_misplaced_headers_never_launch(self):
        bodies = (
            TICKET + "discovery\n" + SEVERITY + "critical\n"
            + SCOPE + "bounded\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "High\n"
            + SCOPE + "bounded\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low \n"
            + SCOPE + "bounded\n\nbody\n",
            TICKET + "discovery\nmailbox-severity: high\n"
            + SCOPE + "bounded\n\nbody\n",
            TICKET + "discovery\n MAILBOX-SEVERITY: high\n"
            + SCOPE + "bounded\n\nbody\n",
            TICKET + "discovery\nMAILBOX-SEVERITY : high\n"
            + SCOPE + "bounded\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low\n"
            + SCOPE + "bounded\n" + SEVERITY + "high\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low\n"
            + SCOPE + "bounded\n\nbody\n"
            + SEVERITY + "high\n",
            TICKET + "discovery\n\nbody\n" + SEVERITY + "low\n",
            TICKET + "discovery\n" + SEVERITY + "low\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low\n"
            + SCOPE + "wide\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low\n"
            + "mailbox-scope: bounded\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low\n"
            + SCOPE + "bounded\n" + SCOPE + "widespread\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low\n"
            + SCOPE + "bounded\n\nbody\n" + SCOPE + "bounded\n",
            TICKET + "closure\n" + SEVERITY + "medium\n\nbody\n",
            TICKET + "closure\n" + SCOPE + "bounded\n\nbody\n",
            TICKET + "transport\n" + SEVERITY + "medium\n\nbody\n",
        )
        for index, body in enumerate(bodies, start=1):
            with self.subTest(index=index), scratch_daemon() as (
                    daemon, _, mailbox, _):
                path = write_pending(
                    daemon, "%04d-to-sol.md" % index, body)
                launches = []
                outcome, output = captured_dispatch(
                    daemon, path, False, launches)
                failed = mailbox / "failed" / path.name
                self.assertFalse(outcome)
                self.assertEqual(launches, [])
                self.assertTrue(failed.is_file())
                self.assertEqual(read_text_exact(failed), body)
                self.assertTrue(
                    "severity" in output.lower() or "scope" in output.lower())

    def test_saved_value_wins_and_every_child_gets_the_effective_value(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            daemon.DISCOVERY_SEVERITY = "high"
            body = (TICKET + "discovery\n" + SEVERITY
                    + "low\n" + SCOPE
                    + "widespread\n\nRead ai/notes/review.md first.\n")
            path = write_pending(daemon, "0001-to-sol.md", body)
            launches = []
            outcome, _ = captured_dispatch(daemon, path, False, launches)
            self.assertTrue(outcome)
            self.assertEqual(len(launches), 1)
            prompt = launches[0]["command"][-1]
            environment = launches[0]["env"]
            self.assertIn(
                "user's saved minimum severity for this discovery: low",
                prompt)
            self.assertIn(
                "saved scope for this discovery: widespread", prompt)
            self.assertEqual(
                environment["MAILBOX_DISCOVERY_SEVERITY"], "low")
            self.assertEqual(
                environment["MAILBOX_DISCOVERY_SCOPE"], "widespread")
            self.assertTrue(prompt.endswith(body))
            self.assertTrue((mailbox / "done" / path.name).is_file())

        with scratch_daemon() as (daemon, _, _, _):
            daemon.DISCOVERY_SEVERITY = "high"
            legacy = (TICKET + "discovery\n" + SEVERITY + "medium\n"
                      + SCOPE + "bounded\n\nlegacy review\n")
            path = write_pending(daemon, "0002-to-sol.md", legacy)
            launches = []
            outcome, _ = captured_dispatch(daemon, path, False, launches)
            self.assertTrue(outcome)
            self.assertEqual(
                launches[0]["env"]["MAILBOX_DISCOVERY_SEVERITY"],
                "medium")
            self.assertEqual(
                launches[0]["env"]["MAILBOX_DISCOVERY_SCOPE"], "bounded")

        for name, body in (
                ("0003-to-fable.md", "plan a bounded unit\n"),
                ("0004-to-opus.md",
                 ticket_flow("implement a bounded unit")),
                ("0005-to-sol.md",
                 TICKET + "discovery\n" + SEVERITY + "high\n"
                 + SCOPE + "bounded\n\nreview a bounded unit\n")):
            with self.subTest(child=name), scratch_daemon() as (
                    daemon, _, _, _):
                daemon.DISCOVERY_SEVERITY = "high"
                path = write_pending(daemon, name, body)
                launches = []
                with mock.patch.object(
                        daemon, "register_ticket_cycle_message",
                        return_value=None):
                    outcome, _ = captured_dispatch(
                        daemon, path, False, launches)
                self.assertTrue(outcome)
                if name.endswith("-to-sol.md"):
                    self.assertIn(
                        "user's saved minimum severity for this discovery: "
                        "high", launches[0]["command"][-1])
                else:
                    self.assertIn(
                        "minimum severity to save on any new discovery "
                        "ticket: high", launches[0]["command"][-1])
                self.assertEqual(
                    launches[0]["env"]["MAILBOX_DISCOVERY_SEVERITY"],
                    "high")

        with scratch_daemon() as (daemon, _, _, _):
            request = daemon.architect_user_request_payload(
                text=("Please instruct the Red Team to do a widespread "
                      "search. Read ai/notes/search-plan.md first."),
                discovery_severity="high")
            path = write_pending(
                daemon, "0006-to-fable.md", request)
            launches = []
            outcome, _ = captured_dispatch(daemon, path, False, launches)
            self.assertTrue(outcome)
            self.assertEqual(
                launches[0]["env"]["MAILBOX_DISCOVERY_SEVERITY"], "low")
            self.assertEqual(
                launches[0]["env"]["MAILBOX_DISCOVERY_SCOPE"],
                "widespread")
            self.assertIn(
                "saved scope for discovery requested by this ticket: "
                "widespread", launches[0]["command"][-1])
            self.assertTrue(launches[0]["command"][-1].endswith(request))

    def test_mailbox_role_helper_and_child_environment_follow_route(self):
        assignment = second_implementer_payload()
        cases = (
            ("fable", "plan one repair\n", "architect", {}),
            ("opus", ticket_flow("implement one repair"),
             "implementer", {}),
            ("sol", TICKET + "discovery\n" + SEVERITY + "medium\n"
             + SCOPE + "bounded\n\nReview one repair.\n",
             "red-team", {}),
            ("sol", assignment, "implementer", {"open_count": 11}),
        )
        for index, (agent, message, expected, settings) in enumerate(
                cases, start=1):
            with self.subTest(agent=agent, expected=expected), \
                    mock.patch.dict(
                        os.environ, {"MAILBOX_ROLE": "spoofed"}), \
                    scratch_daemon(**settings) as (daemon, _, _, _):
                self.assertEqual(
                    daemon.mailbox_role_for_dispatch(
                        agent=agent, message=message),
                    expected)
                path = write_pending(
                    daemon, "%04d-to-%s.md" % (index, agent), message)
                launches = []
                with mock.patch.object(
                        daemon, "register_ticket_cycle_message",
                        return_value=None):
                    outcome, output = captured_dispatch(
                        daemon, path, False, launches)
                self.assertTrue(outcome, output)
                self.assertEqual(len(launches), 1)
                self.assertEqual(
                    launches[0]["env"]["MAILBOX_ROLE"], expected)

        with scratch_daemon() as (daemon, _, _, _):
            with self.assertRaisesRegex(ValueError, "unknown mailbox agent"):
                daemon.mailbox_role_for_dispatch("user", "message")

    def test_nonarchitect_dispatched_roles_cannot_write_backlog_guard(self):
        with scratch_daemon() as (daemon, _, _, _):
            nonarchitect_roles = (
                daemon.mailbox_role_for_dispatch("opus", "implement"),
                daemon.mailbox_role_for_dispatch(
                    "sol", TICKET + "closure\n\nReview.\n"),
                daemon.mailbox_role_for_dispatch(
                    "sol", second_implementer_payload()),
            )
        self.assertEqual(
            nonarchitect_roles,
            ("implementer", "red-team", "implementer"))
        for role in nonarchitect_roles:
            with self.subTest(role=role), mock.patch.dict(
                    os.environ, {"MAILBOX_ROLE": role}, clear=True):
                with self.assertRaisesRegex(
                        backlog_guard.GuardError, "only the Architect"):
                    backlog_guard._require_architect(acknowledged=True)

    def test_malformed_public_scope_never_reaches_architect(self):
        malformed = (
            SEVERITY + "medium\n\nRead ai/notes/plan.md\n",
            SEVERITY + "medium\n" + SCOPE + "wide\n\nplan\n",
            SEVERITY + "medium\nmailbox-scope: bounded\n\nplan\n",
            SEVERITY + "medium\n" + SCOPE + "bounded\n\n"
            + SCOPE + "widespread\nplan\n",
        )
        for index, body in enumerate(malformed, start=1):
            with self.subTest(index=index), scratch_daemon() as (
                    daemon, _, mailbox, _):
                path = write_pending(
                    daemon, "%04d-to-fable.md" % index, body)
                launches = []
                outcome, output = captured_dispatch(
                    daemon, path, False, launches)
                self.assertFalse(outcome)
                self.assertEqual(launches, [])
                self.assertIn("public Architect request", output)
                self.assertTrue((mailbox / "failed" / path.name).is_file())

    def test_cli_scope_environment_and_run_defaults_fail_closed(self):
        invalid_commands = (
            ["--severity", "high"],
            ["--ping", "architect", "--severity", "high"],
            ["--send", "fable", "--unit", "work", "--severity", "high"],
            ["--send", "sol", "--unit", "close", "--ticket-kind",
             "closure", "--severity", "high"],
        )
        for arguments in invalid_commands:
            with self.subTest(arguments=arguments), scratch_daemon(
                    create_mailbox=False) as (daemon, root, mailbox, _):
                before = tree_snapshot(root)
                rc, output, error = run_main(daemon, arguments)
                self.assertNotEqual(rc, 0)
                self.assertEqual(before, tree_snapshot(root))
                self.assertFalse(mailbox.exists())
                self.assertIn("--severity", output + error)

        with mock.patch.dict(
                os.environ,
                {"MAILBOX_DISCOVERY_SEVERITY": " HIGH "}, clear=True), \
                scratch_daemon(create_mailbox=False) as (
                    daemon, root, mailbox, _):
            before = tree_snapshot(root)
            rc, output, _ = run_main(
                daemon,
                ["--send", "architect", "--unit", "review"])
            self.assertNotEqual(rc, 0)
            self.assertIn("must be exactly", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

        with mock.patch.dict(
                os.environ,
                {"MAILBOX_DISCOVERY_SEVERITY": "low"}, clear=True), \
                scratch_daemon(create_mailbox=False) as (
                    daemon, root, mailbox, _):
            before = tree_snapshot(root)
            rc, output, _ = run_main(
                daemon,
                ["--send", "architect", "--unit", "review",
                 "--severity", "high"])
            self.assertNotEqual(rc, 0)
            self.assertIn("does not match inherited", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

        for value in ("high", "medium", "low"):
            with self.subTest(run_value=value), \
                    mock.patch.dict(os.environ, {}, clear=True), \
                    scratch_daemon() as (daemon, _, _, _):
                rc, output, error = run_main(
                    daemon, ["--once", "--severity", value])
                self.assertEqual(rc, 0)
                self.assertEqual(error, "")
                self.assertIn(
                    "discovery severity default: " + value, output)

    def test_fix_only_and_disabled_redteam_are_stronger_than_low(self):
        for value in ("high", "medium", "low"):
            with self.subTest(fix_only=value), scratch_daemon() as (
                    daemon, _, mailbox, _):
                body = (TICKET + "discovery\n" + SEVERITY
                        + value + "\n" + SCOPE
                        + "bounded\n\nreview one edge\n")
                path = write_pending(daemon, "0001-to-sol.md", body)
                launches = []
                outcome, output = captured_dispatch(
                    daemon, path, True, launches)
                self.assertFalse(outcome)
                self.assertEqual(launches, [])
                self.assertIn("fix-only", output)
                self.assertTrue((mailbox / "failed" / path.name).is_file())

            with self.subTest(disabled_redteam=value), mock.patch.dict(
                    os.environ,
                    {"MAILBOX_SKIP_REDTEAM": "1",
                     "MAILBOX_DISCOVERY_SEVERITY": value}, clear=True), \
                    scratch_daemon(create_mailbox=False) as (
                        daemon, root, mailbox, _):
                before = tree_snapshot(root)
                outcome, output = captured_send(
                    daemon, agent="sol", text="review one edge",
                    dry_run=False, ticket_kind="discovery", severity=value)
                self.assertFalse(outcome)
                self.assertIn("Sol route disabled", output)
                self.assertEqual(before, tree_snapshot(root))
                self.assertFalse(mailbox.exists())

            with self.subTest(demand=value), mock.patch.dict(
                    os.environ,
                    {"MAILBOX_DISCOVERY_SEVERITY": value}, clear=True), \
                    scratch_daemon(
                        open_count=10, create_mailbox=False) as (
                            daemon, root, mailbox, _):
                before = tree_snapshot(root)
                outcome, output = captured_send(
                    daemon, agent="sol", text="review one edge",
                    dry_run=False, ticket_kind="discovery", severity=value)
                self.assertFalse(outcome)
                self.assertIn(
                    "open Critical, High, and Medium ticket count is 10",
                    output)
                self.assertIn("without a countable '- OPEN' marker", output)
                self.assertIn("Low tickets do not count", output)
                self.assertIn("only the Architect may designate Critical",
                              output)
                self.assertEqual(before, tree_snapshot(root))
                self.assertFalse(mailbox.exists())

    def test_backlog_priority_counts_keep_type_and_policy_separate(self):
        with scratch_daemon(
                critical_count=1, open_count=2, high_feature_count=3,
                medium_count=4, low_count=5) as (daemon, _, _, backlog):
            counts = daemon.backlog_severity_counts()
            self.assertEqual(
                counts,
                {"critical": 1, "high": 5, "medium": 4, "low": 5,
                 "high_bug_fix": 2, "high_new_functionality": 3,
                 "unclassified": 0, "problem": None})
            self.assertEqual(daemon.backlog_ledger_count(), 15)
            self.assertEqual(daemon.discovery_admission_count(), 10)
            with backlog.open("a", encoding="utf-8") as stream:
                stream.write("- OPEN malformed ticket\n")
                stream.write(
                    "- OPEN **CRITICAL** **NEW FUNCTIONALITY** — invalid\n")
            self.assertEqual(
                daemon.backlog_severity_counts()["unclassified"], 2)

    def test_reopen_count_five_keeps_severity_but_six_forces_low(self):
        classifications = (
            ("critical", {"critical_count": 1}),
            ("high", {"open_count": 1}),
            ("medium", {"medium_count": 1}),
            ("low", {"low_count": 1}),
        )
        for severity, settings in classifications:
            with self.subTest(reopen_count=5, severity=severity), \
                    scratch_daemon(reopen_count=5, **settings) as (
                        daemon, _, _, _):
                counts = daemon.backlog_severity_counts()
                self.assertEqual(counts[severity], 1)
                self.assertEqual(counts["unclassified"], 0)

        for severity, settings in classifications:
            with self.subTest(reopen_count=6, severity=severity), \
                    scratch_daemon(reopen_count=6, **settings) as (
                        daemon, _, _, _):
                counts = daemon.backlog_severity_counts()
                if severity == "low":
                    self.assertEqual(counts["low"], 1)
                    self.assertEqual(counts["unclassified"], 0)
                else:
                    self.assertEqual(counts[severity], 0)
                    self.assertEqual(counts["unclassified"], 1)

    def test_reopen_count_must_be_one_exact_canonical_row(self):
        transformations = {
            "missing": lambda text: text.replace(
                "**Red Team reopen count: 0.**\n", "", 1),
            "leading_zero": lambda text: text.replace(
                "reopen count: 0", "reopen count: 01", 1),
            "lowercase_label": lambda text: text.replace(
                "Red Team reopen count", "Red team reopen count", 1),
            "missing_period": lambda text: text.replace(
                "count: 0.**", "count: 0**", 1),
            "extra_bullet": lambda text: text.replace(
                "**Red Team reopen count: 0.**",
                "- **Red Team reopen count: 0.**", 1),
            "duplicate": lambda text: text.replace(
                "**Red Team reopen count: 0.**\n",
                "**Red Team reopen count: 0.**\n"
                "**Red Team reopen count: 0.**\n", 1),
        }
        for name, transform in transformations.items():
            with self.subTest(name=name), scratch_daemon(open_count=1) as (
                    daemon, _, _, backlog):
                original = backlog.read_text(encoding="utf-8")
                changed = transform(original)
                self.assertNotEqual(changed, original)
                backlog.write_text(changed, encoding="utf-8")
                counts = daemon.backlog_severity_counts()
                self.assertEqual(counts["high"], 0)
                self.assertEqual(counts["unclassified"], 1)

    def test_admission_counts_non_low_tickets_but_not_queue_or_low(self):
        with scratch_daemon(
                critical_count=1, open_count=3, medium_count=5,
                low_count=40) as (daemon, _, _, _):
            for index in range(20):
                write_pending(daemon, "%04d-to-opus.md" % (index + 1))
            outcome, _ = captured_send(
                daemon, agent="sol", text="bounded review", dry_run=False,
                ticket_kind="discovery", severity="medium")
            self.assertTrue(outcome)

        with scratch_daemon(
                critical_count=1, open_count=3, medium_count=6,
                low_count=40, create_mailbox=False) as (
                    daemon, root, mailbox, _):
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text="bounded review", dry_run=False,
                ticket_kind="discovery", severity="low")
            self.assertFalse(outcome)
            self.assertIn("ticket count is 10", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

    def test_widespread_search_is_low_and_waits_for_empty_non_low_queue(self):
        request = ("Please instruct the Red Team to do a widespread search "
                   "for stale formulas.")
        positives = (
            "Do a widespread search for stale formulas.",
            "PLEASE do a widespread search for stale formulas.",
            "Instruct the Red Team to do a widespread search for stale formulas.",
            request,
        )
        bounded_mentions = (
            "Do not do a widespread search; review this commit.",
            "Please do not do a widespread search.",
            "'Do a widespread search' is a phrase in the note.",
            'The note says "do a widespread search" but do not follow it.',
            "Review this commit, then do a widespread search.",
        )
        with scratch_daemon() as (daemon, _, _, _):
            for text in positives:
                with self.subTest(positive=text):
                    payload = daemon.architect_user_request_payload(
                        text=text, discovery_severity="high")
                    self.assertTrue(payload.startswith(
                        SEVERITY + "low\n" + SCOPE
                        + "widespread\n\n"))
                    self.assertEqual(
                        daemon.architect_user_request_scope(payload),
                        "widespread")
            for text in bounded_mentions:
                with self.subTest(bounded=text):
                    payload = daemon.architect_user_request_payload(
                        text=text, discovery_severity="high")
                    self.assertTrue(payload.startswith(
                        SEVERITY + "high\n" + SCOPE + "bounded\n\n"))
                    self.assertEqual(
                        daemon.architect_user_request_scope(payload),
                        "bounded")

        with scratch_daemon(low_count=20) as (daemon, _, _, _):
            outcome, _ = captured_send(
                daemon, agent="sol",
                text="Read ai/notes/widespread-review.md first.",
                dry_run=False, ticket_kind="discovery", severity="low",
                scope="widespread")
            self.assertTrue(outcome)
            pending = pathlib.Path(daemon.pending_messages()[0])
            self.assertIn(
                SCOPE + "widespread\n\nRead ai/notes/widespread-review.md",
                read_text_exact(pending))

        with scratch_daemon(medium_count=1, create_mailbox=False) as (
                daemon, root, mailbox, _):
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text="Read ai/notes/review.md",
                dry_run=False, ticket_kind="discovery", severity="low",
                scope="widespread")
            self.assertFalse(outcome)
            self.assertIn("widespread search waits", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

    def test_scope_refusals_write_nothing(self):
        cases = (
            ({"medium_count": 1}, "low", "widespread"),
            ({}, "high", "widespread"),
            ({}, "low", "wide"),
        )
        for dry_run in (False, True):
            for settings, severity, scope in cases:
                with self.subTest(
                        dry_run=dry_run, severity=severity, scope=scope), \
                        scratch_daemon(
                            create_mailbox=False, **settings) as (
                                daemon, root, mailbox, _):
                    before = tree_snapshot(root)
                    outcome, output = captured_send(
                        daemon, agent="sol",
                        text="Read ai/notes/scope-review.md first.",
                        dry_run=dry_run, ticket_kind="discovery",
                        severity=severity, scope=scope)
                    self.assertFalse(outcome, output)
                    self.assertEqual(before, tree_snapshot(root))
                    self.assertFalse(mailbox.exists())
                    self.assertNotIn("would queue", output)

        # Dispatch trusts the saved header, not a phrase in the body.
        with scratch_daemon(medium_count=1) as (daemon, _, _, _):
            bounded = (TICKET + "discovery\n" + SEVERITY + "low\n"
                       + SCOPE + "bounded\n\nDo a widespread search.\n")
            path = write_pending(daemon, "0001-to-sol.md", bounded)
            launches = []
            outcome, _ = captured_dispatch(daemon, path, False, launches)
            self.assertTrue(outcome)
            self.assertEqual(len(launches), 1)

        with scratch_daemon(create_mailbox=False) as (
                daemon, root, mailbox, _):
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text="Read ai/notes/review.md",
                dry_run=False, ticket_kind="discovery", severity="high",
                scope="widespread")
            self.assertFalse(outcome)
            self.assertIn("automatically Low", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

    def test_second_implementer_requires_exact_bug_emergency(self):
        assignment = second_implementer_payload().split("\n", 1)[1]

        for settings in (
                {"critical_count": 1, "open_count": 10,
                 "high_feature_count": 20, "medium_count": 20,
                 "low_count": 20},
                {"critical_count": 0, "open_count": 0,
                 "high_feature_count": 11}):
            with self.subTest(refused=settings), scratch_daemon(
                    create_mailbox=False, **settings) as (
                        daemon, root, mailbox, _):
                before = tree_snapshot(root)
                outcome, output = captured_send(
                    daemon, agent="sol", text=assignment, dry_run=False,
                    ticket_kind="closure")
                self.assertFalse(outcome)
                self.assertIn("emergency-only", output)
                self.assertEqual(before, tree_snapshot(root))
                self.assertFalse(mailbox.exists())

        for settings in (
                {"critical_count": 2},
                {"open_count": 11}):
            with self.subTest(allowed=settings), scratch_daemon(
                    **settings) as (daemon, _, _, _):
                outcome, _ = captured_send(
                    daemon, agent="sol", text=assignment, dry_run=False,
                    ticket_kind="closure")
                self.assertTrue(outcome)

        with scratch_daemon() as (daemon, _, _, _):
            outcome, _ = captured_send(
                daemon, agent="sol", text="review the named repair",
                dry_run=False, ticket_kind="closure")
            self.assertTrue(outcome)

    def test_stored_second_implementer_message_is_checked_again_at_dispatch(self):
        assignment = second_implementer_payload()

        for settings in (
                {"critical_count": 1, "open_count": 10},
                {"high_feature_count": 40}):
            with self.subTest(refused=settings), scratch_daemon(
                    **settings) as (daemon, _, mailbox, _):
                path = write_pending(
                    daemon, "0001-to-sol.md", assignment)
                launches = []
                outcome, output = captured_dispatch(
                    daemon, path, False, launches)
                self.assertFalse(outcome)
                self.assertEqual(launches, [])
                self.assertIn("emergency-only", output)
                self.assertTrue(
                    (mailbox / "failed" / path.name).is_file())

        for settings in ({"critical_count": 2}, {"open_count": 11}):
            with self.subTest(allowed=settings), scratch_daemon(
                    **settings) as (daemon, _, mailbox, _):
                path = write_pending(
                    daemon, "0002-to-sol.md", assignment)
                launches = []
                with mock.patch.object(
                        daemon, "register_ticket_cycle_message",
                        return_value=None):
                    outcome, _ = captured_dispatch(
                        daemon, path, False, launches)
                self.assertTrue(outcome)
                self.assertEqual(len(launches), 1)
                self.assertTrue(
                    (mailbox / "done" / path.name).is_file())

        with scratch_daemon() as (daemon, _, mailbox, _):
            ordinary = (TICKET + "discovery\n" + SEVERITY + "medium\n"
                        + SCOPE + "bounded\n\n"
                        "Review the named repair.\n")
            path = write_pending(daemon, "0003-to-sol.md", ordinary)
            launches = []
            outcome, _ = captured_dispatch(
                daemon, path, False, launches)
            self.assertTrue(outcome)
            self.assertEqual(len(launches), 1)
            self.assertTrue((mailbox / "done" / path.name).is_file())

    def test_unclassified_backlog_blocks_discovery_but_not_closure(self):
        with scratch_daemon(create_mailbox=False) as (
                daemon, root, mailbox, backlog):
            with backlog.open("a", encoding="utf-8") as stream:
                stream.write("- OPEN missing priority and type\n")
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text="look for a new problem",
                dry_run=False, ticket_kind="discovery", severity="medium")
            self.assertFalse(outcome)
            self.assertIn("unclassified open ticket", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

    def test_missing_or_incomplete_backlog_never_authorizes_ticket_work(self):
        for ticket_kind in ("closure", "discovery"):
            with self.subTest(missing=ticket_kind), scratch_daemon(
                    create_mailbox=False) as (
                        daemon, root, mailbox, backlog):
                backlog.unlink()
                before = tree_snapshot(root)
                outcome, output = captured_send(
                    daemon, agent="sol", text="work on one ticket",
                    dry_run=False, ticket_kind=ticket_kind,
                    severity=("medium" if ticket_kind == "discovery"
                              else None))
                self.assertFalse(outcome)
                self.assertIn("backlog.md is missing", output)
                self.assertEqual(before, tree_snapshot(root))
                self.assertFalse(mailbox.exists())

        with scratch_daemon(create_mailbox=False) as (
                daemon, root, mailbox, backlog):
            target = root / "redirected-backlog.md"
            target.write_text("", encoding="utf-8")
            backlog.unlink()
            backlog.symlink_to(target)
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text="work on one ticket",
                dry_run=False, ticket_kind="closure")
            self.assertFalse(outcome)
            self.assertIn("ordinary file", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

        malformed_lines = (
            "- OPEN **HIGH** **BUG FIX**\n",
            "  - OPEN **HIGH** **BUG FIX** — [Indented](#indented)\n",
            "- open **HIGH** **BUG FIX** — [Lowercase](#lowercase)\n",
            "- OPEN **HIGH** **BUG FIX** — [Missing detail](#missing)\n",
        )
        with scratch_daemon(create_mailbox=False) as (
                daemon, root, mailbox, backlog):
            backlog.write_text("".join(malformed_lines), encoding="utf-8")
            counts = daemon.backlog_severity_counts()
            self.assertEqual(counts["high"], 0)
            self.assertEqual(counts["unclassified"], len(malformed_lines))
            self.assertFalse(daemon.second_implementer_emergency(counts))
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text="find another problem",
                dry_run=False, ticket_kind="discovery", severity="low")
            self.assertFalse(outcome)
            self.assertIn("unclassified open ticket", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

        with scratch_daemon(open_count=11, create_mailbox=False) as (
                daemon, root, mailbox, backlog):
            duplicate = (
                "- OPEN **HIGH** **BUG FIX** — [Duplicate]"
                "(#scratch-high-bug-fix-1)\n")
            with backlog.open("a", encoding="utf-8") as stream:
                stream.write(duplicate)
            counts = daemon.backlog_severity_counts()
            self.assertEqual(counts["unclassified"], 1)
            self.assertFalse(daemon.second_implementer_emergency(counts))
            assignment = (
                "### ARCHITECT_HANDOFF\n\nOpenAI Sol — this is a role as "
                "second Implementer for this unit.\n")
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text=assignment, dry_run=False,
                ticket_kind="closure")
            self.assertFalse(outcome)
            self.assertIn("malformed open ticket", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

        with scratch_daemon() as (daemon, _, _, backlog):
            with backlog.open("a", encoding="utf-8") as stream:
                stream.write("- OPEN missing priority and type\n")
            outcome, _ = captured_send(
                daemon, agent="sol", text="close the recorded problem",
                dry_run=False, ticket_kind="closure")
            self.assertTrue(outcome)

        assignment = (
            "### ARCHITECT_HANDOFF\n\n"
            "OpenAI Sol — this is a role as second Implementer for this "
            "unit.\n")
        with scratch_daemon(open_count=11, create_mailbox=False) as (
                daemon, root, mailbox, backlog):
            with backlog.open("a", encoding="utf-8") as stream:
                stream.write("- OPEN missing priority and type\n")
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text=assignment, dry_run=False,
                ticket_kind="closure")
            self.assertFalse(outcome)
            self.assertIn("malformed open ticket", output)
            self.assertEqual(before, tree_snapshot(root))
            self.assertFalse(mailbox.exists())

    def test_source_mutations_break_saved_severity_contract(self):
        source = DAEMON_PATH.read_text(encoding="utf-8")

        def replace_exact(old, new):
            self.assertEqual(source.count(old), 1)
            return source.replace(old, new, 1)

        def default_is_medium(candidate):
            with mock.patch.dict(os.environ, {}, clear=True), \
                    scratch_daemon(source=candidate) as (daemon, _, _, _):
                outcome, _ = captured_send(
                    daemon, agent="sol", text="review",
                    dry_run=False, ticket_kind="discovery")
                payload = pathlib.Path(daemon.pending_messages()[0])
                return (outcome and SEVERITY + "medium\n"
                        in read_text_exact(payload))

        def inherited_is_saved(candidate):
            with mock.patch.dict(
                    os.environ,
                    {"MAILBOX_DISCOVERY_SEVERITY": "high"}, clear=True), \
                    scratch_daemon(source=candidate) as (daemon, _, _, _):
                outcome, _ = captured_send(
                    daemon, agent="sol", text="review",
                    dry_run=False, ticket_kind="discovery")
                payload = pathlib.Path(daemon.pending_messages()[0])
                return (outcome and SEVERITY + "high\n"
                        in read_text_exact(payload))

        def explicit_is_saved(candidate):
            with mock.patch.dict(os.environ, {}, clear=True), \
                    scratch_daemon(source=candidate) as (daemon, _, _, _):
                outcome, _ = captured_send(
                    daemon, agent="sol", text="review", dry_run=False,
                    ticket_kind="discovery", severity="high")
                payload = pathlib.Path(daemon.pending_messages()[0])
                return (outcome and SEVERITY + "high\n"
                        in read_text_exact(payload))

        def malformed_refuses(candidate):
            with scratch_daemon(source=candidate) as (
                    daemon, _, mailbox, _):
                path = write_pending(
                    daemon, "0001-to-sol.md",
                    TICKET + "discovery\n" + SEVERITY
                    + "critical\n" + SCOPE + "bounded\n\nbody\n")
                launches = []
                outcome, _ = captured_dispatch(
                    daemon, path, False, launches)
                return (not outcome and launches == []
                        and (mailbox / "failed" / path.name).is_file())

        def saved_value_reaches_child(candidate):
            with scratch_daemon(source=candidate) as (daemon, _, _, _):
                daemon.DISCOVERY_SEVERITY = "high"
                path = write_pending(
                    daemon, "0001-to-sol.md",
                    TICKET + "discovery\n" + SEVERITY
                    + "low\n" + SCOPE + "widespread\n\nbody\n")
                launches = []
                outcome, _ = captured_dispatch(
                    daemon, path, False, launches)
                if not outcome or len(launches) != 1:
                    return False
                return (
                    launches[0]["env"]["MAILBOX_DISCOVERY_SEVERITY"] == "low"
                    and launches[0]["env"]["MAILBOX_DISCOVERY_SCOPE"]
                    == "widespread"
                    and "user's saved minimum severity for this discovery: "
                    "low" in launches[0]["command"][-1]
                    and "saved scope for this discovery: widespread"
                    in launches[0]["command"][-1])

        def high_still_refuses(candidate):
            with scratch_daemon(source=candidate) as (daemon, _, _, _):
                fix_refusal = daemon.sol_ticket_refusal(
                    ticket_kind="discovery", admission_count=0,
                    fix_only=True,
                    discovery_severity="high",
                    discovery_scope="bounded")
                demand_refusal = daemon.sol_ticket_refusal(
                    ticket_kind="discovery", admission_count=10,
                    fix_only=False,
                    discovery_severity="high",
                    discovery_scope="bounded")
                return fix_refusal is not None and demand_refusal is not None

        mutations = (
            ("default changed",
             'DEFAULT_DISCOVERY_SEVERITY = "medium"',
             'DEFAULT_DISCOVERY_SEVERITY = "low"', default_is_medium),
            ("inherited value ignored",
             "    if cli_value is None:\n"
             "        return (DEFAULT_DISCOVERY_SEVERITY\n"
             "                if inherited is None else inherited)\n",
             "    if cli_value is None:\n"
             "        return DEFAULT_DISCOVERY_SEVERITY\n",
             inherited_is_saved),
            ("explicit value not persisted",
             '                   + SOL_SEVERITY_HEADER + discovery_severity '
             '+ "\\n"\n',
             '                   + SOL_SEVERITY_HEADER '
             '+ DEFAULT_DISCOVERY_SEVERITY + "\\n"\n',
             explicit_is_saved),
            ("saved value ignored",
             "        if saved_severity is not None:\n"
             "            effective_discovery_severity = saved_severity\n",
             "        if False:\n"
             "            effective_discovery_severity = saved_severity\n",
             saved_value_reaches_child),
            ("child receives run default",
             "        env[DISCOVERY_SEVERITY_ENVIRONMENT] = "
             "effective_discovery_severity\n",
             "        env[DISCOVERY_SEVERITY_ENVIRONMENT] = "
             "DISCOVERY_SEVERITY\n", saved_value_reaches_child),
            ("saved banner removed",
             '        saved_discovery=(ticket_kind == "discovery"),\n',
             "        saved_discovery=False,\n",
             saved_value_reaches_child),
            ("high bypasses fix-only",
             '    if fix_only and ticket_kind != "closure":\n',
             '    if (fix_only and ticket_kind != "closure"\n'
             '            and discovery_severity != "high"):\n',
             high_still_refuses),
            ("high bypasses demand",
             '    if (ticket_kind == "discovery"\n'
             '            and admission_count >= '
             'DISCOVERY_ADMISSION_THRESHOLD):\n',
             '    if (ticket_kind == "discovery"\n'
             '            and admission_count >= '
             'DISCOVERY_ADMISSION_THRESHOLD\n'
             '            and discovery_severity != "high"):\n',
             high_still_refuses),
        )
        for label, old, new, probe in mutations:
            with self.subTest(mutation=label):
                self.assertTrue(probe(source), "baseline failed: " + label)
                self.assertFalse(
                    probe(replace_exact(old, new)),
                    "mutation survived: " + label)


if __name__ == "__main__":
    unittest.main()
