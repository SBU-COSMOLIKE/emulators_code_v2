"""Focused tests for the Architect-only public mailbox entry point."""

import contextlib
import io
import json
import os
import pathlib
import sys
import unittest
from unittest import mock

from ai.tests.tools_mailbox_daemon_fix_only_repro import captured_dispatch
from ai.tests.tools_mailbox_daemon_fix_only_repro import captured_send
from ai.tests.tools_mailbox_daemon_fix_only_repro import BASE_COMMIT
from ai.tests.tools_mailbox_daemon_fix_only_repro import DAEMON_PATH
from ai.tests.tools_mailbox_daemon_fix_only_repro import read_text_exact
from ai.tests.tools_mailbox_daemon_fix_only_repro import run_main
from ai.tests.tools_mailbox_daemon_fix_only_repro import scratch_daemon
from ai.tests.tools_mailbox_daemon_fix_only_repro import tree_snapshot
from ai.tests.tools_mailbox_daemon_primary_worktree_repro import \
    arm_failed_ancestor_handoff_requeues_and_advances


SEVERITY_HEADER = "MAILBOX-SEVERITY: "
SCOPE_HEADER = "MAILBOX-SCOPE: "


def install_maintenance_architect_child(daemon, mailbox, plan_count):
    """Use a real tiny child process that publishes Implementer plans."""
    daemon.capture_persistent_role_state = (
        lambda agent: {"base": BASE_COMMIT, "agent": agent})
    daemon.recheck_persistent_role_state = lambda proof: proof
    daemon.worktree_head = lambda worktree: BASE_COMMIT
    daemon._validate_current_protected_primary_state = (
        lambda primary_worktree: None)
    cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
    child = (
        "import os, pathlib, sys\n"
        "mailbox = pathlib.Path(sys.argv[1])\n"
        "count = int(sys.argv[2])\n"
        "token = os.environ['MAILBOX_ARCHITECT_ADMISSION']\n"
        "for index in range(count):\n"
        "    body = ('MAILBOX-FLOW: ticket\\nMAILBOX-CYCLE: "
        + cycle
        + "\\nMAILBOX-MODE: normal\\n\\nMAILBOX-ADMISSION: ' "
        "+ token + '\\nImplement the selected bug.\\n')\n"
        "    path = mailbox / ('%04d-to-opus.md' % (index + 2))\n"
        "    path.write_text(body, encoding='utf-8', newline='')\n")
    daemon.AGENT_COMMANDS["fable"] = [
        sys.executable, "-c", child, str(mailbox), str(plan_count)]


def install_maintenance_no_ticket_child(daemon, mailbox):
    """Use a tiny Architect child that claims no eligible ticket exists."""
    daemon.capture_persistent_role_state = (
        lambda agent: {"base": BASE_COMMIT, "agent": agent})
    daemon.recheck_persistent_role_state = lambda proof: proof
    daemon.worktree_head = lambda worktree: BASE_COMMIT
    daemon._validate_current_protected_primary_state = (
        lambda primary_worktree: None)
    child = (
        "import os, pathlib, sys\n"
        "mailbox = pathlib.Path(sys.argv[1])\n"
        "token = os.environ['MAILBOX_ARCHITECT_ADMISSION']\n"
        "body = ('MAILBOX-RETURN: architect-no-ticket\\n' "
        "+ 'MAILBOX-ADMISSION: ' + token + '\\n' "
        "+ 'MAILBOX-DECISION: NO TICKET\\n')\n"
        "(mailbox / '0002-to-user.md').write_text("
        "body, encoding='utf-8', newline='')\n")
    daemon.AGENT_COMMANDS["fable"] = [
        sys.executable, "-c", child, str(mailbox)]


def install_architect_child_with_concurrent_user_requests(
        daemon, mailbox, user_requests):
    """Publish one bound plan while independent user requests arrive."""
    daemon.capture_persistent_role_state = (
        lambda agent: {"base": BASE_COMMIT, "agent": agent})
    daemon.recheck_persistent_role_state = lambda proof: proof
    daemon.worktree_head = lambda worktree: BASE_COMMIT
    daemon._validate_current_protected_primary_state = (
        lambda primary_worktree: None)
    cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
    child = (
        "import os, pathlib, sys\n"
        "mailbox = pathlib.Path(sys.argv[1])\n"
        "token = os.environ['MAILBOX_ARCHITECT_ADMISSION']\n"
        "plan = ('MAILBOX-FLOW: ticket\\nMAILBOX-CYCLE: "
        + cycle
        + "\\nMAILBOX-MODE: normal\\n\\nMAILBOX-ADMISSION: ' "
        "+ token + '\\nImplement the selected bug.\\n')\n"
        "(mailbox / '0014-to-opus.md').write_text("
        "plan, encoding='utf-8', newline='')\n"
        "for index, payload in enumerate(sys.argv[2:], start=14):\n"
        "    (mailbox / ('%04d-to-fable.md' % index)).write_text(\n"
        "        payload, encoding='utf-8', newline='')\n")
    daemon.AGENT_COMMANDS["fable"] = [
        sys.executable, "-c", child, str(mailbox), *user_requests]


class MailboxArchitectEntrypointTests(unittest.TestCase):
    """Pin the public role boundary without narrowing internal routing."""

    def test_architect_must_self_check_the_published_handoff(self):
        with scratch_daemon() as (daemon, _, _, _):
            preamble = daemon.ARCHITECT_ROLE_PREAMBLE
            self.assertIn("Before ending, re-read the outgoing\nfile", preamble)
            self.assertIn("one Directive row", preamble)
            self.assertIn("validation alone does not validate", preamble)

    def test_current_handoff_adds_scope_to_a_legacy_active_cycle(self):
        with scratch_daemon(open_count=1) as (daemon, _, _, _):
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            legacy = daemon.empty_ticket_cycle_state()
            legacy["active"][cycle] = {
                "phase": "implementation",
                "commit": None,
                "mode": "normal",
                "route": "primary",
            }
            pathlib.Path(daemon.ticket_cycle_state_path()).write_text(
                json.dumps(legacy), encoding="utf-8", newline="")

            self.assertIsNone(
                daemon.read_ticket_cycle_state()["active"][cycle].get(
                    "path_scope"))

            message = (
                "MAILBOX-FLOW: ticket\n"
                "MAILBOX-CYCLE: " + cycle + "\n"
                "MAILBOX-MODE: normal\n\n"
                "Implement the validated directive.\n")
            daemon.register_ticket_cycle_message(
                agent="opus", message=message,
                path_scope=[
                    "emulator/training.py",
                    "ai/tests/test_training.py",
                ])

            self.assertEqual(
                daemon.read_ticket_cycle_state()["active"][cycle][
                    "path_scope"],
                ["ai/tests/test_training.py", "emulator/training.py"])

    def test_active_ticket_scope_cannot_change_mid_cycle(self):
        with scratch_daemon(open_count=1) as (daemon, _, _, _):
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            message = (
                "MAILBOX-FLOW: ticket\n"
                "MAILBOX-CYCLE: " + cycle + "\n"
                "MAILBOX-MODE: normal\n\n"
                "Implement the validated directive.\n")
            original_scope = ["emulator/training.py"]
            daemon.register_ticket_cycle_message(
                agent="opus", message=message,
                path_scope=original_scope)

            with self.assertRaisesRegex(
                    daemon.TicketCycleStateError, "path scope"):
                daemon.register_ticket_cycle_message(
                    agent="opus", message=message,
                    path_scope=["emulator/model.py"])

            self.assertEqual(
                daemon.read_ticket_cycle_state()["active"][cycle][
                    "path_scope"],
                original_scope)

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

    def test_fix_only_send_saves_one_policy_free_maintenance_request(self):
        with scratch_daemon(create_mailbox=False) as (daemon, _, _, _):
            rc, output, error = run_main(
                daemon, ["--send", "architect", "--fix-only", "true"])
            pending = [pathlib.Path(path)
                       for path in daemon.pending_messages()]

            self.assertEqual(rc, 0, output + error)
            self.assertEqual(len(pending), 1)
            self.assertEqual(read_text_exact(pending[0]),
                             daemon.ARCHITECT_FIX_ONLY_REQUEST)
            self.assertNotIn(SEVERITY_HEADER, read_text_exact(pending[0]))

    def test_fix_only_send_without_a_backlog_writes_no_request(self):
        with scratch_daemon(create_mailbox=False) as (
                daemon, root, mailbox, backlog):
            backlog.unlink()
            before = tree_snapshot(root)

            rc, output, error = run_main(
                daemon, ["--send", "architect", "--fix-only", "true"])

            self.assertNotEqual(rc, 0, output + error)
            self.assertEqual(tree_snapshot(root), before)
            self.assertFalse(mailbox.exists())
            self.assertIn("backlog.md is missing", output)

    def test_fix_only_send_rejects_policy_that_belongs_on_watcher(self):
        for extra in (["--severity", "high"], ["--unit", "one ticket"]):
            with self.subTest(extra=extra), \
                    scratch_daemon(create_mailbox=False) as (
                        daemon, root, mailbox, _):
                before = tree_snapshot(root)
                rc, output, error = run_main(
                    daemon,
                    ["--send", "architect", "--fix-only", "true"] + extra)
                self.assertNotEqual(rc, 0, output + error)
                self.assertEqual(tree_snapshot(root), before)
                self.assertFalse(mailbox.exists())

    def test_fix_only_request_waits_behind_its_implementer_ticket(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request = mailbox / "0001-to-fable.md"
            request.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            (mailbox / "0002-to-opus.md").write_text(
                "ticket already assigned\n", encoding="utf-8", newline="")

            outcome = daemon.drain_lane(
                paths=[str(request)], dry_run=False, fix_only=True)

            self.assertFalse(outcome)
            self.assertEqual(read_text_exact(request),
                             daemon.ARCHITECT_FIX_ONLY_REQUEST)
            self.assertFalse((mailbox / "inflight" / request.name).exists())

    def test_watch_restart_recovers_the_exact_failed_maintenance_request(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            failed = mailbox / "failed"
            failed.mkdir()
            original = failed / "0013-to-fable.md"
            duplicate = mailbox / "0014-to-fable.md"
            malformed = mailbox / "0015-to-fable.md"
            original.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            duplicate.write_bytes(original.read_bytes())
            malformed.write_bytes(b"\xff")
            state = daemon.read_ticket_cycle_state()
            state["architect_admissions"][original.name] = {
                "mode": "normal",
                "sequence": 13,
                "sha256": daemon.hashlib.sha256(
                    original.read_bytes()).hexdigest(),
            }
            daemon.write_ticket_cycle_state(state=state)

            recovered = daemon.recover_failed_maintenance_admission()

            self.assertEqual(recovered, str(mailbox / original.name))
            self.assertTrue((mailbox / original.name).is_file())
            self.assertFalse(original.exists())
            self.assertTrue((failed / duplicate.name).is_file())
            self.assertFalse(duplicate.exists())
            self.assertTrue(malformed.is_file())
            self.assertIn(
                original.name,
                daemon.read_ticket_cycle_state()["architect_admissions"])

            launches = []
            daemon.dispatch = lambda **kwargs: launches.append(kwargs) or True
            daemon._ACTIVE_WATCH_RENDEZVOUS = daemon.SafeKillRendezvous(
                ticket_cycle_limit=5, ticket_cycle_topology="normal")
            try:
                daemon.prepare_finite_watch_progress(
                    limit=5, topology="normal")
                outcome = daemon.drain_lane(
                    paths=[recovered], dry_run=False, fix_only=True)
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            expected_token = daemon.architect_admission_token(
                request_name=original.name,
                digest=daemon.hashlib.sha256(
                    pathlib.Path(recovered).read_bytes()).hexdigest())
            self.assertTrue(outcome)
            self.assertEqual(len(launches), 1)
            self.assertEqual(
                launches[0]["architect_admission"], expected_token)

    def test_restart_releases_failed_public_request_without_retry(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            failed = mailbox / "failed"
            failed.mkdir()
            request = failed / "0013-to-fable.md"
            request.write_text(
                daemon.architect_user_request_payload("One failed request."),
                encoding="utf-8", newline="")
            state = daemon.read_ticket_cycle_state()
            state["architect_admissions"][request.name] = {
                "mode": "normal", "sequence": 13,
                "sha256": daemon.hashlib.sha256(
                    request.read_bytes()).hexdigest(),
            }
            daemon.write_ticket_cycle_state(state=state)

            daemon.recover_before_dispatch()

            self.assertTrue(request.is_file())
            self.assertEqual(
                daemon.read_ticket_cycle_state()["architect_admissions"], {})
            self.assertEqual(
                daemon.recover_failed_public_architect_admissions(), 0)

    def test_failed_request_keeps_slot_for_live_implementer_handoff(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            failed = mailbox / "failed"
            failed.mkdir()
            request = failed / "0013-to-fable.md"
            request.write_text(
                daemon.architect_user_request_payload("One failed request."),
                encoding="utf-8", newline="")
            digest = daemon.hashlib.sha256(request.read_bytes()).hexdigest()
            token = daemon.architect_admission_token(
                request_name=request.name, digest=digest)
            (mailbox / "0014-to-opus.md").write_text(
                "MAILBOX-FLOW: ticket\n"
                "MAILBOX-CYCLE: one-ticket@" + BASE_COMMIT + "\n"
                "MAILBOX-MODE: normal\n\n"
                "MAILBOX-ADMISSION: " + token + "\n"
                "Implement this exact ticket.\n",
                encoding="utf-8", newline="")
            state = daemon.read_ticket_cycle_state()
            state["architect_admissions"][request.name] = {
                "mode": "normal", "sequence": 13, "sha256": digest}
            daemon.write_ticket_cycle_state(state=state)

            recovered = daemon.recover_failed_public_architect_admissions()

            self.assertEqual(recovered, 0)
            self.assertIn(
                request.name,
                daemon.read_ticket_cycle_state()["architect_admissions"])

    def test_failed_request_keeps_slot_until_inflight_move_is_resolved(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            failed = mailbox / "failed"
            inflight = mailbox / "inflight"
            failed.mkdir()
            inflight.mkdir()
            request = failed / "0013-to-fable.md"
            request.write_text(
                daemon.architect_user_request_payload("One failed request."),
                encoding="utf-8", newline="")
            inflight_request = inflight / request.name
            guard = pathlib.Path(
                str(inflight_request) + daemon.STATE_GUARD_SUFFIX)
            os.link(request, inflight_request)
            os.link(request, guard)
            state = daemon.read_ticket_cycle_state()
            state["architect_admissions"][request.name] = {
                "mode": "normal", "sequence": 13,
                "sha256": daemon.hashlib.sha256(
                    request.read_bytes()).hexdigest(),
            }
            daemon.write_ticket_cycle_state(state=state)

            self.assertEqual(
                daemon.recover_failed_public_architect_admissions(), 0)
            self.assertIn(
                request.name,
                daemon.read_ticket_cycle_state()["architect_admissions"])

            inflight_request.unlink()
            guard.unlink()
            self.assertEqual(
                daemon.recover_failed_public_architect_admissions(), 1)
            self.assertEqual(
                daemon.read_ticket_cycle_state()["architect_admissions"], {})

    def test_live_pass_releases_slot_after_bad_handoff_is_parked(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            failed = mailbox / "failed"
            failed.mkdir()
            request = failed / "0013-to-fable.md"
            request.write_text(
                daemon.architect_user_request_payload("One failed request."),
                encoding="utf-8", newline="")
            bad_handoff = mailbox / "0014-to-opus.md"
            bad_handoff.write_bytes(b"\xff")
            state = daemon.read_ticket_cycle_state()
            state["architect_admissions"][request.name] = {
                "mode": "normal", "sequence": 13,
                "sha256": daemon.hashlib.sha256(
                    request.read_bytes()).hexdigest(),
            }
            daemon.write_ticket_cycle_state(state=state)
            daemon.message_is_enabled_for_topology = lambda **_kwargs: True

            def park_bad_lane(paths, **_kwargs):
                return all(daemon.park_failed_message(path) for path in paths)

            daemon.drain_lane = park_bad_lane
            daemon.process_backlog(dry_run=False)

            self.assertTrue((failed / bad_handoff.name).is_file())
            self.assertEqual(
                daemon.read_ticket_cycle_state()["architect_admissions"], {})

    def test_other_admission_still_defers_fix_only_request(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request = mailbox / "0013-to-fable.md"
            request.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            state = daemon.read_ticket_cycle_state()
            state["architect_admissions"]["0012-to-fable.md"] = {
                "mode": "normal", "sequence": 12, "sha256": "0" * 64}
            daemon.write_ticket_cycle_state(state=state)
            dispatch = mock.Mock(return_value=True)
            daemon.dispatch = dispatch

            outcome = daemon.drain_lane(
                paths=[str(request)], dry_run=False, fix_only=True)

            self.assertFalse(outcome)
            dispatch.assert_not_called()

    def test_prelaunch_refusal_releases_its_new_ticket_reservation(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            request = mailbox / "0015-to-opus.md"
            request.write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                + "\nMAILBOX-MODE: normal\n\nImplement this ticket.\n",
                encoding="utf-8", newline="")
            daemon.dispatch = mock.Mock(return_value=False)
            daemon._ACTIVE_WATCH_RENDEZVOUS = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")
            try:
                daemon.prepare_finite_watch_progress(
                    limit=1, topology="normal")
                outcome = daemon.drain_lane(
                    paths=[str(request)], dry_run=False, fix_only=True)
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            self.assertFalse(outcome)
            self.assertEqual(daemon.active_ticket_cycle_count(), 0)

    def test_public_cycle_requeues_after_directive_preflight_refusal(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            request = mailbox / "0015-to-opus.md"
            request.write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                + "\nMAILBOX-MODE: normal\n\n"
                "- **Directive:** [ai/notes/missing.md, section "
                "Implementation directive]\n",
                encoding="utf-8", newline="")
            state = daemon.read_ticket_cycle_state()
            state["active"][cycle] = {
                "phase": "implementation", "commit": None,
                "mode": "normal", "route": "primary"}
            daemon.write_ticket_cycle_state(state=state)
            daemon.ACTIVE_TOPOLOGY = {}
            daemon.prepare_implementer_evidence_contract = mock.Mock(
                side_effect=daemon.TicketCycleStateError(
                    "source note is missing"))

            outcome = daemon.dispatch(path=str(request), dry_run=False)

            self.assertFalse(outcome)
            self.assertEqual(daemon.active_ticket_cycle_count(), 1)
            held = mailbox / "prelaunch" / request.name
            self.assertTrue(held.is_file())
            self.assertEqual(daemon.recover_prelaunch_messages(), 1)
            self.assertTrue(request.is_file())
            self.assertEqual(daemon.active_ticket_cycle_count(), 1)

    def test_started_dispatch_keeps_its_ticket_reservation(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            request = mailbox / "0015-to-opus.md"
            request.write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                + "\nMAILBOX-MODE: normal\n\nImplement this ticket.\n",
                encoding="utf-8", newline="")
            controller = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")

            def launched_then_refused(**_kwargs):
                permit = daemon._RENDEZVOUS_LOCAL.permit
                controller.turn_started(permit=permit)
                controller.turn_finished(permit=permit)
                return False

            daemon.dispatch = launched_then_refused
            daemon._ACTIVE_WATCH_RENDEZVOUS = controller
            try:
                daemon.prepare_finite_watch_progress(
                    limit=1, topology="normal")
                outcome = daemon.drain_lane(
                    paths=[str(request)], dry_run=False, fix_only=True)
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            self.assertFalse(outcome)
            self.assertEqual(daemon.active_ticket_cycle_count(), 1)

    def test_restart_releases_the_proved_old_prelaunch_reservation(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            failed = mailbox / "failed"
            failed.mkdir()
            request = failed / "0015-to-opus.md"
            request.write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                + "\nMAILBOX-MODE: normal\n\n"
                "- **Directive:** `ai/notes/ticket.md`, section "
                "`Implementation directive`\n",
                encoding="utf-8", newline="")
            state = daemon.read_ticket_cycle_state()
            state["active"][cycle] = {
                "phase": "implementation", "commit": None,
                "mode": "normal", "route": "primary"}
            daemon.write_ticket_cycle_state(state=state)
            daemon.candidate_commit_for_cycle = lambda cycle_id: None
            daemon.worktree_head = lambda worktree: BASE_COMMIT
            daemon._clean_worktree_status = lambda worktree: b""

            recovered = daemon.recover_failed_implementer_preflight()

            self.assertEqual(recovered, 1)
            self.assertEqual(daemon.active_ticket_cycle_count(), 0)
            self.assertTrue(request.is_file())

    def test_quarantined_non_utf8_messages_do_not_break_restart(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            failed = mailbox / "failed"
            failed.mkdir()
            failed_opus = failed / "0015-to-opus.md"
            failed_sol = failed / "0016-to-sol.md"
            failed_opus.write_bytes(b"\xff")
            failed_sol.write_bytes(b"\xff")

            self.assertEqual(daemon.recover_failed_implementer_preflight(), 0)
            self.assertEqual(daemon.reconcile_ticket_cycle_state(), 0)
            self.assertTrue(failed_opus.is_file())
            self.assertTrue(failed_sol.is_file())

    def test_watch_stops_immediately_for_failed_permanent_note_debt(self):
        with scratch_daemon(open_count=1) as (daemon, _, _, _):
            debt = daemon.ARCHITECT_NOTES_DEBT_PREFIX + "inspect failed note"
            daemon.architect_notes_failed_debt_error = lambda: debt
            daemon.process_backlog = mock.Mock(
                side_effect=AssertionError("watch entered its poll loop"))

            rc, output, error = run_main(
                daemon, ["--watch", "--cycle", "1"])

            self.assertNotEqual(rc, 0, output + error)
            self.assertIn(debt, output)
            daemon.process_backlog.assert_not_called()

    def test_restart_keeps_reservation_owned_by_a_corrected_handoff(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            failed = mailbox / "failed"
            failed.mkdir()
            (failed / "0015-to-opus.md").write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                + "\nMAILBOX-MODE: normal\n\n"
                "- **Directive:** `ai/notes/ticket.md`, section "
                "`Implementation directive`\n",
                encoding="utf-8", newline="")
            (mailbox / "0016-to-opus.md").write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                + "\nMAILBOX-MODE: normal\n\n"
                "- **Directive:** [ai/notes/ticket.md, exact "
                "Implementation directive section]\n",
                encoding="utf-8", newline="")
            state = daemon.read_ticket_cycle_state()
            state["active"][cycle] = {
                "phase": "implementation", "commit": None,
                "mode": "normal", "route": "primary"}
            daemon.write_ticket_cycle_state(state=state)
            daemon.candidate_commit_for_cycle = lambda cycle_id: None
            daemon.worktree_head = lambda worktree: BASE_COMMIT
            daemon._clean_worktree_status = lambda worktree: b""

            recovered = daemon.recover_failed_implementer_preflight()

            self.assertEqual(recovered, 0)
            self.assertEqual(daemon.active_ticket_cycle_count(), 1)

    def test_clean_older_implementer_head_advances_to_ticket_base(self):
        with scratch_daemon(open_count=1) as (daemon, _, _, _):
            old_head = "0" * 40
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            state = daemon.read_ticket_cycle_state()
            state["active"][cycle] = {
                "phase": "implementation", "commit": None,
                "mode": "normal", "route": "primary"}
            daemon.write_ticket_cycle_state(state=state)
            current = [old_head]
            daemon._clean_worktree_status = lambda worktree: b""
            daemon.worktree_head = lambda worktree: current[0]
            daemon.candidate_record_locked = lambda **kwargs: None
            ancestry = mock.Mock()
            daemon._require_ancestor_or_same = ancestry

            def reset(repository_root, arguments, check=True):
                self.assertEqual(arguments, ["reset", "--hard", BASE_COMMIT])
                current[0] = BASE_COMMIT
                return mock.Mock(returncode=0, stdout=b"", stderr=b"")

            daemon._run_git = reset
            prepared = daemon.prepare_implementer_cycle_checkout(
                cycle_id=cycle)

            self.assertEqual(prepared, BASE_COMMIT)
            ancestry.assert_called_once_with(
                ancestor=old_head, descendant=BASE_COMMIT,
                label="Implementer HEAD is not an ancestor of the "
                      "ticket base")

    def test_budget_repair_restarts_from_base_not_rejected_candidate(self):
        with scratch_daemon(open_count=1) as (daemon, _, _, _):
            prior = "2" * 40
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            state = daemon.read_ticket_cycle_state()
            state["active"][cycle] = {
                "phase": "implementation", "commit": None,
                "mode": "normal", "route": "primary"}
            daemon.write_ticket_cycle_state(state=state)
            current = [prior]
            daemon._clean_worktree_status = lambda worktree: b""
            daemon.worktree_head = lambda worktree: current[0]
            daemon.candidate_record_locked = lambda **kwargs: {
                "commit": prior}
            daemon.read_candidate_state = lambda: {
                "cycles": {cycle: {"commit": prior}}}

            def reset(repository_root, arguments, check=True):
                self.assertEqual(arguments, ["reset", "--hard", BASE_COMMIT])
                current[0] = BASE_COMMIT
                return mock.Mock(returncode=0, stdout=b"", stderr=b"")

            daemon._run_git = reset
            prepared = daemon.prepare_implementer_cycle_checkout(
                cycle_id=cycle, restart_from_base=True)

            self.assertEqual(prepared, BASE_COMMIT)
            self.assertEqual(current[0], BASE_COMMIT)

    def test_divergent_implementer_head_is_still_preserved(self):
        with scratch_daemon(open_count=1) as (daemon, _, _, _):
            old_head = "0" * 40
            cycle = "scratch-high-bug-fix-1@" + BASE_COMMIT
            state = daemon.read_ticket_cycle_state()
            state["active"][cycle] = {
                "phase": "implementation", "commit": None,
                "mode": "normal", "route": "primary"}
            daemon.write_ticket_cycle_state(state=state)
            daemon._clean_worktree_status = lambda worktree: b""
            daemon.worktree_head = lambda worktree: old_head
            daemon.candidate_record_locked = lambda **kwargs: None
            daemon._require_ancestor_or_same = mock.Mock(
                side_effect=daemon.TicketCycleStateError("divergent"))
            daemon._run_git = mock.Mock()

            with self.assertRaisesRegex(
                    daemon.TicketCycleStateError,
                    "refusing to discard " + old_head):
                daemon.prepare_implementer_cycle_checkout(cycle_id=cycle)
            daemon._run_git.assert_not_called()

    def test_failed_ancestor_handoff_requeues_in_a_real_repository(self):
        self.assertTrue(arm_failed_ancestor_handoff_requeues_and_advances())

    def test_failed_implementer_turn_is_never_automatic_retry(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            failed = mailbox / "failed"
            failed.mkdir()
            request = failed / "0017-to-opus.md"
            request.write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: ticket@"
                + BASE_COMMIT + "\nMAILBOX-MODE: normal\n\n"
                "- **Directive:** [ai/notes/ticket.md, exact "
                "Implementation directive section]\n",
                encoding="utf-8", newline="")
            daemon.write_timeout_history(
                name=request.name, killed_after_minutes=120)

            recovered = daemon.recover_prelaunch_messages()

            self.assertEqual(recovered, 0)
            self.assertTrue(request.is_file())
            self.assertEqual(len(daemon.timeout_events(request.name)), 1)

    def test_fix_only_request_reserves_the_finite_ticket_slot(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            maintenance = mailbox / "0001-to-fable.md"
            maintenance.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            ordinary = mailbox / "0002-to-fable.md"
            ordinary.write_text(
                daemon.architect_user_request_payload("another ticket"),
                encoding="utf-8", newline="")
            daemon._ACTIVE_WATCH_RENDEZVOUS = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")
            try:
                daemon.prepare_finite_watch_progress(
                    limit=1, topology="normal")
                deferred, token = daemon.reserve_architect_ticket_before_claim(
                    path=str(maintenance), skip_redteam=False)
                later_deferred, later_token = (
                    daemon.reserve_architect_ticket_before_claim(
                        path=str(ordinary), skip_redteam=False))
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            self.assertIsNone(deferred)
            self.assertIsNotNone(token)
            self.assertIsNotNone(later_deferred)
            self.assertIsNone(later_token)

    def test_maintenance_admission_is_current_watch_authority(self):
        token = "0001-to-fable.md@" + "a" * 64
        with scratch_daemon() as (daemon, _, _, _):
            prompt = daemon.architect_admission_prompt(token)

        self.assertIn("slot is free now", prompt)
        self.assertIn("decision is authoritative", prompt)
        self.assertIn("Past tickets", prompt)
        self.assertIn("do not consume it", prompt)

    def test_maintenance_no_ticket_requeues_when_high_bug_remains(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            request = mailbox / "0001-to-fable.md"
            request.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            install_maintenance_no_ticket_child(
                daemon=daemon, mailbox=mailbox)
            daemon.DISCOVERY_SEVERITY = "high"
            daemon._ACTIVE_WATCH_RENDEZVOUS = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")
            output = io.StringIO()
            try:
                daemon.prepare_finite_watch_progress(
                    limit=1, topology="normal")
                with contextlib.redirect_stdout(output):
                    outcome = daemon.drain_lane(
                        paths=[str(request)], dry_run=False, fix_only=True)
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            self.assertFalse(outcome)
            self.assertTrue(request.is_file())
            self.assertTrue(
                (mailbox / "failed" / "0002-to-user.md").is_file())
            self.assertIn(
                request.name,
                daemon.read_ticket_cycle_state()["architect_admissions"])
            self.assertIn("eligible Open BUG FIX remains", output.getvalue())
            self.assertIn("same admitted slot", output.getvalue())

    def test_maintenance_no_ticket_stops_only_when_no_bug_is_eligible(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request = mailbox / "0001-to-fable.md"
            request.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            install_maintenance_no_ticket_child(
                daemon=daemon, mailbox=mailbox)
            daemon.DISCOVERY_SEVERITY = "high"
            daemon._ACTIVE_WATCH_RENDEZVOUS = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")
            try:
                daemon.prepare_finite_watch_progress(
                    limit=1, topology="normal")
                outcome = daemon.drain_lane(
                    paths=[str(request)], dry_run=False, fix_only=True)
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            self.assertTrue(outcome)
            self.assertTrue((mailbox / "done" / request.name).is_file())
            self.assertEqual(
                daemon.read_ticket_cycle_state()["architect_admissions"], {})
            self.assertTrue(daemon._NO_ELIGIBLE_MAINTENANCE_WORK.is_set())

    def test_finite_watch_exits_after_truthful_maintenance_no_ticket(self):
        with scratch_daemon() as (daemon, _, _, _):
            daemon.recover_before_dispatch = lambda **_kwargs: None
            daemon.deliver_pending_ticket_cycle_returns = lambda: 0
            daemon.role_contract_exit_status = lambda: None
            daemon.architect_notes_failed_debt_error = lambda: None
            daemon.active_ticket_cycle_count = lambda **_kwargs: 0
            daemon.enabled_pending_messages = lambda **_kwargs: []

            def report_no_work(**_kwargs):
                daemon._NO_ELIGIBLE_MAINTENANCE_WORK.set()
                return True

            daemon.process_backlog = report_no_work
            rc, output, error = run_main(
                daemon,
                ["--watch", "--cycle", "1", "--fix-only", "true",
                 "--severity", "high"])

            self.assertEqual(rc, 0, output + error)
            self.assertIn("no eligible Open BUG FIX remains", output)
            self.assertIsNone(
                daemon.read_ticket_cycle_state()["finite_watch"])

    def test_fix_only_plan_creates_one_waiting_continuation(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            request = mailbox / "0001-to-fable.md"
            request.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            install_maintenance_architect_child(
                daemon=daemon, mailbox=mailbox, plan_count=1)
            controller = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")
            daemon._ACTIVE_WATCH_RENDEZVOUS = controller
            try:
                daemon.prepare_finite_watch_progress(
                    limit=1, topology="normal")
                outcome = daemon.drain_lane(
                    paths=[str(request)], dry_run=False, fix_only=True)
                pending = [pathlib.Path(path)
                           for path in daemon.pending_messages()]
                continuation = next(
                    path for path in pending
                    if path.name.endswith("-to-fable.md"))

                audit = mailbox / "0004-to-fable.md"
                audit.write_text(
                    "MAILBOX-FLOW: ticket\n"
                    "MAILBOX-CYCLE: scratch-high-bug-fix-1@"
                    + BASE_COMMIT
                    + "\nMAILBOX-MODE: normal\n\nAudit this candidate.\n",
                    encoding="utf-8", newline="")
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    daemon.drain_lane(
                        paths=[str(continuation), str(audit)],
                        dry_run=True, fix_only=True)

                state = daemon.read_ticket_cycle_state()
                state["active"] = {}
                daemon.write_ticket_cycle_state(state=state)
                controller.ticket_cycle_returned()
                daemon.drain_lane(
                    paths=[str(continuation)],
                    dry_run=False, fix_only=True)
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            self.assertTrue(outcome)
            self.assertEqual(
                sum(path.name.endswith("-to-opus.md") for path in pending),
                1)
            self.assertEqual(
                sum(path.name.endswith("-to-fable.md") for path in pending),
                1)
            self.assertIn(
                "[dry-run] would dispatch 0004-to-fable.md ->",
                stream.getvalue())
            self.assertTrue(continuation.is_file())

    def test_concurrent_user_requests_are_not_architect_outputs(self):
        requests = [
            "Explain the README sentence.",
            "Open the ordinary-send idempotency bug.",
            "Correct that bug's severity to High.",
        ]
        payloads = [
            "MAILBOX-SEVERITY: high\nMAILBOX-SCOPE: bounded\n\n"
            + request + "\n"
            for request in requests
        ]
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            admitted = mailbox / "0013-to-fable.md"
            admitted.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            install_architect_child_with_concurrent_user_requests(
                daemon=daemon, mailbox=mailbox, user_requests=payloads)
            daemon._ACTIVE_WATCH_RENDEZVOUS = daemon.SafeKillRendezvous(
                ticket_cycle_limit=5, ticket_cycle_topology="normal")
            try:
                daemon.prepare_finite_watch_progress(
                    limit=5, topology="normal")
                outcome = daemon.drain_lane(
                    paths=[str(admitted)], dry_run=False, fix_only=True)
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            self.assertTrue(outcome)
            self.assertTrue((mailbox / "done" / admitted.name).is_file())
            self.assertTrue((mailbox / "0014-to-opus.md").is_file())
            for index, payload in enumerate(payloads, start=14):
                path = mailbox / ("%04d-to-fable.md" % index)
                self.assertEqual(read_text_exact(path), payload)
                self.assertFalse(
                    (mailbox / "failed" / path.name).exists())
            self.assertEqual(
                daemon.read_ticket_cycle_state()["architect_admissions"], {})
            self.assertIn(
                "scratch-high-bug-fix-1@" + BASE_COMMIT,
                daemon.read_ticket_cycle_state()["active"])

    def test_restart_recovers_bound_plan_and_concurrent_user_requests(self):
        with scratch_daemon(open_count=1) as (daemon, root, mailbox, _):
            failed = mailbox / "failed"
            failed.mkdir()
            admitted = failed / "0013-to-fable.md"
            admitted.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            digest = daemon.hashlib.sha256(admitted.read_bytes()).hexdigest()
            token = daemon.architect_admission_token(
                request_name=admitted.name, digest=digest)
            plan = failed / "0014-to-opus.md"
            plan.write_text(
                "MAILBOX-FLOW: ticket\n"
                "MAILBOX-CYCLE: scratch-high-bug-fix-1@" + BASE_COMMIT
                + "\nMAILBOX-MODE: normal\n\n"
                "MAILBOX-ADMISSION: " + token + "\n"
                "- **Directive:** [ai/notes/ticket.md, section "
                "Implementation directive]\n",
                encoding="utf-8", newline="")
            payloads = [
                daemon.architect_user_request_payload(text)
                for text in (
                    "Explain the README sentence.",
                    "Open the ordinary-send idempotency bug.",
                    "Correct that bug's severity to High.",
                )
            ]
            collateral = []
            for index, payload in enumerate(payloads, start=14):
                path = failed / ("%04d-to-fable.md" % index)
                path.write_text(payload, encoding="utf-8", newline="")
                collateral.append(path)
            state = daemon.read_ticket_cycle_state()
            state["architect_admissions"][admitted.name] = {
                "mode": "normal", "sequence": 13, "sha256": digest}
            daemon.write_ticket_cycle_state(state=state)

            call_order = []
            recover_outcome = daemon.recover_failed_architect_outcome
            recover_maintenance = daemon.recover_failed_maintenance_admission

            def recorded_outcome():
                call_order.append("outcome")
                return recover_outcome()

            def recorded_maintenance():
                call_order.append("maintenance")
                return recover_maintenance()

            daemon.recover_failed_architect_outcome = recorded_outcome
            daemon.recover_failed_maintenance_admission = recorded_maintenance
            daemon.recover_before_dispatch(fix_only=True)

            self.assertEqual(call_order[:2], ["outcome", "maintenance"])
            self.assertTrue((mailbox / "done" / admitted.name).is_file())
            self.assertTrue((mailbox / plan.name).is_file())
            for failed_path, payload in zip(collateral, payloads):
                restored = mailbox / failed_path.name
                self.assertEqual(read_text_exact(restored), payload)
                self.assertFalse(failed_path.exists())
            recovered_state = daemon.read_ticket_cycle_state()
            self.assertEqual(recovered_state["architect_admissions"], {})
            self.assertIn(
                "scratch-high-bug-fix-1@" + BASE_COMMIT,
                recovered_state["active"])
            self.assertFalse(admitted.exists())
            self.assertFalse(plan.exists())

            before_repeat = tree_snapshot(root)
            call_order.clear()
            daemon.recover_before_dispatch(fix_only=True)
            self.assertEqual(call_order[:2], ["outcome", "maintenance"])
            self.assertEqual(tree_snapshot(root), before_repeat)

    def test_fix_only_request_refuses_two_implementer_plans(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            request = mailbox / "0001-to-fable.md"
            request.write_text(
                daemon.ARCHITECT_FIX_ONLY_REQUEST,
                encoding="utf-8", newline="")
            install_maintenance_architect_child(
                daemon=daemon, mailbox=mailbox, plan_count=2)
            daemon._ACTIVE_WATCH_RENDEZVOUS = daemon.SafeKillRendezvous()
            try:
                outcome = daemon.drain_lane(
                    paths=[str(request)], dry_run=False, fix_only=True)
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None

            self.assertFalse(outcome)
            self.assertFalse(list(mailbox.glob("*-to-fable.md")))
            self.assertFalse(list(mailbox.glob("*-to-opus.md")))
            self.assertEqual(
                len(list((mailbox / "failed").glob("*-to-opus.md"))), 2)

    def test_bare_ping_checks_providers_without_queuing_role_work(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                scratch_daemon(create_mailbox=False) as (
                    daemon, root, mailbox, _):
            check = mock.Mock(return_value=True)
            daemon.check_provider_connectivity = check
            before = tree_snapshot(root)

            rc, output, error = run_main(daemon, ["--ping"])

            self.assertEqual(rc, 0, output + error)
            self.assertEqual(error, "")
            self.assertEqual(tree_snapshot(root), before)
            self.assertFalse(mailbox.exists())
            check.assert_called_once_with(
                architect_model=daemon.DEFAULT_ARCHITECT_MODEL,
                implementer_provider=daemon.DEFAULT_IMPLEMENTER_PROVIDER,
                implementer_model=daemon.DEFAULT_IMPLEMENTER_MODEL,
                include_sol=True, dry_run=False,
                implementer_compaction_limit=(
                    daemon.DEFAULT_IMPLEMENTER_CONTEXT_BUDGET))

    def test_removed_public_targets_and_ticket_kind_are_zero_write_errors(self):
        commands = []
        for old_target in ("fable", "opus", "sol"):
            commands.append(
                ["--send", old_target, "--unit", "must not queue"])
        for old_target in ("architect", "fable", "opus", "sol"):
            commands.append(["--ping", old_target])
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
                provider_check = mock.Mock(return_value=True)
                daemon.check_provider_connectivity = provider_check
                before = tree_snapshot(root)
                rc, output, error = run_main(daemon, arguments)
                self.assertNotEqual(rc, 0, output + error)
                self.assertEqual(tree_snapshot(root), before)
                self.assertFalse(mailbox.exists())
                provider_check.assert_not_called()

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
            self.assertEqual(launches[0]["command"][-2], "--")
            self.assertIn("coordinate one ticket", launches[0]["command"][-1])
            self.assertTrue((mailbox / "done" / path.name).is_file())

    def test_help_separates_architect_send_from_provider_ping(self):
        with scratch_daemon(create_mailbox=False) as (daemon, _, _, _):
            rc, output, error = run_main(daemon, ["--help"])
        help_text = output + error
        normalized_help = " ".join(help_text.split())
        self.assertEqual(rc, 0)
        self.assertIn("--send {architect}", help_text)
        self.assertIn("--ping", help_text)
        self.assertNotIn("--ping {architect}", help_text)
        self.assertIn(
            "save the user's ticket request for the Architect", help_text)
        self.assertIn(
            "make one small live request to every provider", help_text)
        self.assertIn(
            "with --ping, check the Architect and Implementer providers but "
            "not Sol", normalized_help)
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
                "architect send mapped to Implementer",
                '        queued = send(\n'
                '            agent="fable",\n'
                '            text=request,\n',
                '        queued = send(\n'
                '            agent="opus",\n'
                '            text=request,\n',
                architect_send_uses_fable,
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
