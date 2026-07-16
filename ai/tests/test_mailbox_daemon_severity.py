"""Focused discovery-severity protocol and propagation tests."""

import contextlib
import io
import os
import pathlib
import unittest
from unittest import mock

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
                    + "\n\nreview the named change\n")

    def test_legacy_discovery_defaults_to_medium_and_keeps_body(self):
        with scratch_daemon() as (daemon, _, _, _):
            legacy = TICKET + "discovery\r\n\r\nlegacy body\r\n"
            self.assertEqual(
                daemon.sol_discovery_severity(message=legacy), "medium")
            self.assertEqual(
                daemon.sol_ticket_body(message=legacy),
                "\r\nlegacy body\r\n")

            current = (TICKET + "discovery\r\n" + SEVERITY
                       + "low\r\n\r\ncurrent body\r\n")
            self.assertEqual(
                daemon.sol_discovery_severity(message=current), "low")
            self.assertEqual(
                daemon.sol_ticket_body(message=current),
                "\r\ncurrent body\r\n")
            assignment = (TICKET + "discovery\n" + SEVERITY + "low\n\n"
                          + daemon.SECOND_IMPLEMENTER_MODE_SENTENCE + "\n")
            self.assertTrue(
                daemon.sol_second_implementer_assignment(message=assignment))

    def test_malformed_or_misplaced_headers_never_launch(self):
        bodies = (
            TICKET + "discovery\n" + SEVERITY + "critical\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "High\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low \n\nbody\n",
            TICKET + "discovery\nmailbox-severity: high\n\nbody\n",
            TICKET + "discovery\n MAILBOX-SEVERITY: high\n\nbody\n",
            TICKET + "discovery\nMAILBOX-SEVERITY : high\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low\n"
            + SEVERITY + "high\n\nbody\n",
            TICKET + "discovery\n" + SEVERITY + "low\n\nbody\n"
            + SEVERITY + "high\n",
            TICKET + "discovery\n\nbody\n" + SEVERITY + "low\n",
            TICKET + "closure\n" + SEVERITY + "medium\n\nbody\n",
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
                self.assertIn("severity", output.lower())

    def test_saved_value_wins_and_every_child_gets_the_effective_value(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            daemon.DISCOVERY_SEVERITY = "high"
            body = (TICKET + "discovery\n" + SEVERITY
                    + "low\n\nreview one change\n")
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
            self.assertEqual(
                environment["MAILBOX_DISCOVERY_SEVERITY"], "low")
            self.assertTrue(prompt.endswith(body))
            self.assertTrue((mailbox / "done" / path.name).is_file())

        with scratch_daemon() as (daemon, _, _, _):
            daemon.DISCOVERY_SEVERITY = "high"
            legacy = TICKET + "discovery\n\nlegacy review\n"
            path = write_pending(daemon, "0002-to-sol.md", legacy)
            launches = []
            outcome, _ = captured_dispatch(daemon, path, False, launches)
            self.assertTrue(outcome)
            self.assertEqual(
                launches[0]["env"]["MAILBOX_DISCOVERY_SEVERITY"],
                "medium")

        for name, body in (
                ("0003-to-fable.md", "plan a bounded unit\n"),
                ("0004-to-opus.md", "implement a bounded unit\n"),
                ("0005-to-sol.md",
                 TICKET + "closure\n\nclose a bounded unit\n")):
            with self.subTest(child=name), scratch_daemon() as (
                    daemon, _, _, _):
                daemon.DISCOVERY_SEVERITY = "high"
                path = write_pending(daemon, name, body)
                launches = []
                outcome, _ = captured_dispatch(
                    daemon, path, False, launches)
                self.assertTrue(outcome)
                self.assertIn(
                    "minimum severity to save on any new discovery ticket: "
                    "high", launches[0]["command"][-1])
                self.assertEqual(
                    launches[0]["env"]["MAILBOX_DISCOVERY_SEVERITY"],
                    "high")

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
                        + value + "\n\nreview one edge\n")
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
                self.assertIn("total open demand is 10", output)
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
                    + "critical\n\nbody\n")
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
                    + "low\n\nbody\n")
                launches = []
                outcome, _ = captured_dispatch(
                    daemon, path, False, launches)
                if not outcome or len(launches) != 1:
                    return False
                return (
                    launches[0]["env"]["MAILBOX_DISCOVERY_SEVERITY"] == "low"
                    and "user's saved minimum severity for this discovery: "
                    "low" in launches[0]["command"][-1])

        def legacy_is_medium(candidate):
            with scratch_daemon(source=candidate) as (daemon, _, _, _):
                daemon.DISCOVERY_SEVERITY = "high"
                return daemon.sol_discovery_severity(
                    message=TICKET + "discovery\n\nlegacy\n") == "medium"

        def high_still_refuses(candidate):
            with scratch_daemon(source=candidate) as (daemon, _, _, _):
                fix_refusal = daemon.sol_ticket_refusal(
                    ticket_kind="discovery", total=0, fix_only=True,
                    discovery_severity="high")
                demand_refusal = daemon.sol_ticket_refusal(
                    ticket_kind="discovery", total=10, fix_only=False,
                    discovery_severity="high")
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
             '+ "\\n\\n"\n',
             '                   + SOL_SEVERITY_HEADER '
             '+ DEFAULT_DISCOVERY_SEVERITY + "\\n\\n"\n',
             explicit_is_saved),
            ("malformed header admitted",
             "        reason = severity_problem\n",
             "        reason = None\n", malformed_refuses),
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
            ("legacy follows mutable run default",
             "        return DEFAULT_DISCOVERY_SEVERITY\n",
             "        return DISCOVERY_SEVERITY\n", legacy_is_medium),
            ("high bypasses fix-only",
             '    if fix_only and ticket_kind != "closure":\n',
             '    if (fix_only and ticket_kind != "closure"\n'
             '            and discovery_severity != "high"):\n',
             high_still_refuses),
            ("high bypasses demand",
             '    if (ticket_kind == "discovery"\n'
             '            and total >= SECOND_IMPLEMENTER_THRESHOLD):\n',
             '    if (ticket_kind == "discovery"\n'
             '            and total >= SECOND_IMPLEMENTER_THRESHOLD\n'
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
