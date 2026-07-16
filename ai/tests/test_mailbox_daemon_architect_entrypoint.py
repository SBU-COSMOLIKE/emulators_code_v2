"""Focused tests for the Architect-only public mailbox entry point."""

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


SEVERITY_HEADER = "MAILBOX-SEVERITY: "
SCOPE_HEADER = "MAILBOX-SCOPE: "


class MailboxArchitectEntrypointTests(unittest.TestCase):
    """Pin the public role boundary without narrowing internal routing."""

    def test_send_architect_queues_one_fable_file_and_preserves_request(self):
        request = (
            "Please coordinate this exact request.\n"
            "Ask for a widespread search only because I wrote those words.")
        with mock.patch.dict(os.environ, {}, clear=True), \
                scratch_daemon(create_mailbox=False) as (
                    daemon, _, mailbox, _):
            rc, output, error = run_main(
                daemon,
                ["--send", "architect", "--unit", request])

            pending = [pathlib.Path(path)
                       for path in daemon.pending_messages()]
            self.assertEqual(rc, 0, output + error)
            self.assertEqual(error, "")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].parent, mailbox)
            self.assertEqual(pending[0].name, "0001-to-fable.md")
            self.assertEqual(
                read_text_exact(pending[0]),
                SEVERITY_HEADER + "medium\n"
                + SCOPE_HEADER + "bounded\n\n" + request + "\n")
            self.assertFalse(list(mailbox.glob("*-to-opus.md")))
            self.assertFalse(list(mailbox.glob("*-to-sol.md")))

    def test_ping_architect_queues_one_fable_architect_ping(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                scratch_daemon(create_mailbox=False) as (
                    daemon, _, mailbox, _):
            rc, output, error = run_main(
                daemon, ["--ping", "architect"])

            pending = [pathlib.Path(path)
                       for path in daemon.pending_messages()]
            self.assertEqual(rc, 0, output + error)
            self.assertEqual(error, "")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].parent, mailbox)
            self.assertEqual(pending[0].name, "0001-to-fable.md")
            self.assertEqual(
                read_text_exact(pending[0]),
                daemon.transport_ping_text(agent="architect"))
            self.assertIn("PING for architect", read_text_exact(pending[0]))
            self.assertNotIn("PING for fable", read_text_exact(pending[0]))

    def test_removed_public_targets_and_ticket_kind_are_zero_write_errors(self):
        commands = []
        for action in ("--send", "--ping"):
            for old_target in ("fable", "opus", "sol"):
                command = [action, old_target]
                if action == "--send":
                    command.extend(["--unit", "must not queue"])
                commands.append(command)
        commands.extend((
            ["--send", "architect", "--unit", "must not queue",
             "--ticket-kind", "closure"],
            ["--send", "architect", "--unit", "must not queue",
             "--ticket-kind", "discovery"],
        ))

        for arguments in commands:
            with self.subTest(arguments=arguments), \
                    mock.patch.dict(os.environ, {}, clear=True), \
                    scratch_daemon(create_mailbox=False) as (
                        daemon, root, mailbox, _):
                before = tree_snapshot(root)
                rc, output, error = run_main(daemon, arguments)
                self.assertNotEqual(rc, 0, output + error)
                self.assertEqual(tree_snapshot(root), before)
                self.assertFalse(mailbox.exists())

    def test_architect_send_saves_each_requested_severity(self):
        for supplied, expected in (
                (None, "medium"),
                ("high", "high"),
                ("medium", "medium"),
                ("low", "low")):
            arguments = [
                "--send", "architect", "--unit", "coordinate one ticket"]
            if supplied is not None:
                arguments.extend(["--severity", supplied])
            with self.subTest(supplied=supplied), \
                    mock.patch.dict(os.environ, {}, clear=True), \
                    scratch_daemon(create_mailbox=False) as (
                        daemon, _, _, _):
                rc, output, error = run_main(daemon, arguments)
                pending = [pathlib.Path(path)
                           for path in daemon.pending_messages()]
                self.assertEqual(rc, 0, output + error)
                self.assertEqual(error, "")
                self.assertEqual(len(pending), 1)
                self.assertEqual(
                    read_text_exact(pending[0]),
                    SEVERITY_HEADER + expected
                    + "\n" + SCOPE_HEADER
                    + "bounded\n\ncoordinate one ticket\n")

    def test_saved_architect_severity_binds_that_dispatch(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                scratch_daemon(create_mailbox=False) as (
                    daemon, _, mailbox, _):
            rc, output, error = run_main(
                daemon,
                ["--send", "architect", "--unit", "coordinate one ticket",
                 "--severity", "low"])
            self.assertEqual(rc, 0, output + error)
            path = pathlib.Path(daemon.pending_messages()[0])

            # A later watch may use a different default.  The setting saved
            # with this user's request must remain the Architect's binding.
            daemon.DISCOVERY_SEVERITY = "high"
            launches = []
            outcome, dispatch_output = captured_dispatch(
                daemon, path, False, launches)
            self.assertTrue(outcome, dispatch_output)
            self.assertEqual(len(launches), 1)
            self.assertEqual(
                launches[0]["env"]["MAILBOX_DISCOVERY_SEVERITY"], "low")
            self.assertEqual(
                launches[0]["env"]["MAILBOX_DISCOVERY_SCOPE"], "bounded")
            self.assertIn("coordinate one ticket", launches[0]["command"][-1])
            self.assertTrue((mailbox / "done" / path.name).is_file())

    def test_help_names_architect_as_the_only_public_target(self):
        with scratch_daemon(create_mailbox=False) as (daemon, _, _, _):
            rc, output, error = run_main(daemon, ["--help"])
        help_text = output + error
        normalized_help = " ".join(help_text.split())
        self.assertEqual(rc, 0)
        self.assertIn("--send {architect}", help_text)
        self.assertIn("--ping {architect}", help_text)
        self.assertIn(
            "save the user's ticket request for the Architect", help_text)
        self.assertIn(
            "save a connection-check message for the Architect", help_text)
        self.assertIn(
            "not sent to another role", normalized_help)
        self.assertNotIn("--ticket-kind", help_text)
        for old_form in (
                "--send fable", "--send opus", "--send sol",
                "--ping fable", "--ping opus", "--ping sol"):
            with self.subTest(old_form=old_form):
                self.assertNotIn(old_form, help_text)

    def test_internal_opus_and_sol_send_paths_remain_functional(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                scratch_daemon() as (daemon, _, _, _):
            opus_outcome, opus_output = captured_send(
                daemon, agent="opus", text="implement the approved unit",
                dry_run=False)
            sol_outcome, sol_output = captured_send(
                daemon, agent="sol", text="review the named commit",
                dry_run=False, ticket_kind="closure")

            pending = [pathlib.Path(path)
                       for path in daemon.pending_messages()]
            self.assertTrue(opus_outcome, opus_output)
            self.assertTrue(sol_outcome, sol_output)
            self.assertEqual(
                [path.name for path in pending],
                ["0001-to-opus.md", "0002-to-sol.md"])
            self.assertEqual(
                read_text_exact(pending[0]),
                "implement the approved unit\n")
            self.assertEqual(
                read_text_exact(pending[1]),
                "MAILBOX-TICKET: closure\n\nreview the named commit\n")

    def test_source_mutations_break_public_choices_and_backend_mapping(self):
        source = DAEMON_PATH.read_text(encoding="utf-8")

        def replace_exact(old, new):
            self.assertEqual(source.count(old), 1)
            return source.replace(old, new, 1)

        def old_sends_refuse(candidate):
            for target in ("fable", "opus", "sol"):
                with mock.patch.dict(os.environ, {}, clear=True), \
                        scratch_daemon(
                            create_mailbox=False, source=candidate) as (
                                daemon, root, mailbox, _):
                    before = tree_snapshot(root)
                    rc, _, _ = run_main(
                        daemon,
                        ["--send", target, "--unit", "must not queue"])
                    if (rc == 0 or tree_snapshot(root) != before
                            or mailbox.exists()):
                        return False
            return True

        def old_pings_refuse(candidate):
            for target in ("fable", "opus", "sol"):
                with mock.patch.dict(os.environ, {}, clear=True), \
                        scratch_daemon(
                            create_mailbox=False, source=candidate) as (
                                daemon, root, mailbox, _):
                    before = tree_snapshot(root)
                    rc, _, _ = run_main(daemon, ["--ping", target])
                    if (rc == 0 or tree_snapshot(root) != before
                            or mailbox.exists()):
                        return False
            return True

        def architect_send_uses_fable(candidate):
            with mock.patch.dict(os.environ, {}, clear=True), \
                    scratch_daemon(
                        create_mailbox=False, source=candidate) as (
                            daemon, _, _, _):
                rc, _, _ = run_main(
                    daemon,
                    ["--send", "architect", "--unit", "one request"])
                pending = [pathlib.Path(path)
                           for path in daemon.pending_messages()]
                return (rc == 0 and len(pending) == 1
                        and pending[0].name == "0001-to-fable.md")

        def architect_ping_uses_fable(candidate):
            with mock.patch.dict(os.environ, {}, clear=True), \
                    scratch_daemon(
                        create_mailbox=False, source=candidate) as (
                            daemon, _, _, _):
                rc, _, _ = run_main(daemon, ["--ping", "architect"])
                pending = [pathlib.Path(path)
                           for path in daemon.pending_messages()]
                return (rc == 0 and len(pending) == 1
                        and pending[0].name == "0001-to-fable.md")

        mutations = (
            (
                "public send choices widened",
                '    parser.add_argument("--send", metavar="{architect}",\n'
                '                        choices=["architect"],\n',
                '    parser.add_argument("--send", metavar="{architect}",\n'
                '                        choices=["architect", "fable", '
                '"opus", "sol"],\n',
                old_sends_refuse,
            ),
            (
                "public ping choices widened",
                '    parser.add_argument("--ping", metavar="{architect}",\n'
                '                        choices=["architect"],\n',
                '    parser.add_argument("--ping", metavar="{architect}",\n'
                '                        choices=["architect", "fable", '
                '"opus", "sol"],\n',
                old_pings_refuse,
            ),
            (
                "architect send mapped to Implementer",
                '    if args.send:\n'
                '        request = architect_user_request_payload(\n'
                '            text=args.unit,\n'
                '            discovery_severity=selected_discovery_severity)\n'
                '        queued = send(\n'
                '            agent="fable",\n'
                '            text=request,\n',
                '    if args.send:\n'
                '        request = architect_user_request_payload(\n'
                '            text=args.unit,\n'
                '            discovery_severity=selected_discovery_severity)\n'
                '        queued = send(\n'
                '            agent="opus",\n'
                '            text=request,\n',
                architect_send_uses_fable,
            ),
            (
                "architect ping mapped to Implementer",
                '    if args.ping:\n'
                '        ping_text = transport_ping_text(agent="architect")\n'
                '        queued = send(\n'
                '            agent="fable",\n',
                '    if args.ping:\n'
                '        ping_text = transport_ping_text(agent="architect")\n'
                '        queued = send(\n'
                '            agent="opus",\n',
                architect_ping_uses_fable,
            ),
        )

        for label, old, new, probe in mutations:
            with self.subTest(mutation=label):
                self.assertTrue(probe(source), "baseline failed: " + label)
                mutant = replace_exact(old, new)
                self.assertFalse(
                    probe(mutant), "mutation survived: " + label)


if __name__ == "__main__":
    unittest.main()
