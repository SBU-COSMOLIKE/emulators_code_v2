"""Recovery tests for an Implementer result interrupted before archiving."""

import pathlib
import unittest
from unittest import mock

from ai.tests.tools_mailbox_daemon_fix_only_repro import ACCEPTED_COMMIT
from ai.tests.tools_mailbox_daemon_fix_only_repro import BASE_COMMIT
from ai.tests.tools_mailbox_daemon_fix_only_repro import captured_dispatch
from ai.tests.tools_mailbox_daemon_fix_only_repro import SCRATCH_HIGH_ANCHOR
from ai.tests.tools_mailbox_daemon_fix_only_repro import scratch_daemon


def prepare_delivery(daemon, mailbox):
    """Write one inflight request, its validated return, and the receipt."""
    inflight = mailbox / "inflight"
    inflight.mkdir(parents=True, exist_ok=True)
    request = inflight / "0001-to-opus.md"
    returned = mailbox / "0002-to-fable.md"
    cycle = "scratch-candidate-recovery@" + BASE_COMMIT
    flow = ("MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
            + "\nMAILBOX-MODE: normal\n\n")
    request.write_text(flow + "exact Implementer request\n",
                       encoding="utf-8")
    returned.write_text(
        flow + "- **Candidate commit:** `" + ACCEPTED_COMMIT + "`\n",
        encoding="utf-8")
    receipt = pathlib.Path(daemon.write_implementer_delivery_receipt(
        request_path=str(request), return_path=str(returned)))
    return request, returned, receipt, cycle


def prepare_architect_delivery(daemon, mailbox, outcome_agent):
    """Write one audited candidate request and its validated outcome."""
    inflight = mailbox / "inflight"
    inflight.mkdir(parents=True, exist_ok=True)
    cycle = "scratch-architect-recovery@" + BASE_COMMIT
    flow = ("MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
            + "\nMAILBOX-MODE: normal\n\n")
    request = inflight / "0001-to-fable.md"
    request.write_text(
        flow + "- **Candidate commit:** `" + ACCEPTED_COMMIT + "`\n",
        encoding="utf-8")
    outcome = mailbox / ("0002-to-" + outcome_agent + ".md")
    if outcome_agent == "daemon":
        text = daemon.architect_go_request_payload(
            cycle_id=cycle, candidate_commit=ACCEPTED_COMMIT,
            mode="normal")
    else:
        text = (flow + "- **Directive:** [ai/notes/ticket.md, exact "
                "Implementation directive]\n")
    outcome.write_text(text, encoding="utf-8")
    receipt = pathlib.Path(daemon.write_implementer_delivery_receipt(
        request_path=str(request), return_path=str(outcome)))
    return request, outcome, receipt


class MailboxCandidateDeliveryRecoveryTests(unittest.TestCase):
    """Prove restart finishes only the exact validated delivery."""

    def test_restart_preserves_candidate_and_archives_request_once(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request, returned, receipt, cycle = prepare_delivery(
                daemon=daemon, mailbox=mailbox)
            expected_request = request.read_text(encoding="utf-8")
            preserve = mock.Mock(return_value=ACCEPTED_COMMIT)
            daemon.record_implementer_candidate = preserve
            daemon.candidate_commit_for_cycle = mock.Mock(return_value=None)
            daemon.worktree_head = mock.Mock(return_value=ACCEPTED_COMMIT)

            self.assertEqual(daemon.recover_implementer_deliveries(), 1)
            self.assertFalse(request.exists())
            self.assertEqual(
                (mailbox / "done" / request.name).read_text(
                    encoding="utf-8"),
                expected_request)
            self.assertTrue(returned.is_file())
            self.assertFalse(receipt.exists())
            preserve.assert_called_once_with(
                cycle_id=cycle, starting_head=BASE_COMMIT)

            self.assertEqual(daemon.recover_implementer_deliveries(), 0)
            self.assertEqual(preserve.call_count, 1)

    def test_restart_accepts_archive_completed_before_receipt_removal(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request, _, receipt, _ = prepare_delivery(
                daemon=daemon, mailbox=mailbox)
            done = mailbox / "done"
            done.mkdir()
            request.replace(done / request.name)
            preserve = mock.Mock(return_value=ACCEPTED_COMMIT)
            daemon.record_implementer_candidate = preserve
            daemon.candidate_commit_for_cycle = mock.Mock(
                return_value=ACCEPTED_COMMIT)

            self.assertEqual(daemon.recover_implementer_deliveries(), 1)
            self.assertTrue((done / request.name).is_file())
            self.assertFalse(receipt.exists())
            preserve.assert_not_called()

    def test_changed_request_or_return_is_not_recovered(self):
        for changed in ("request", "return"):
            with self.subTest(changed=changed), scratch_daemon() as (
                    daemon, _, mailbox, _):
                request, returned, receipt, _ = prepare_delivery(
                    daemon=daemon, mailbox=mailbox)
                target = request if changed == "request" else returned
                if changed == "request":
                    target.write_text("changed after validation\n",
                                      encoding="utf-8")
                else:
                    with target.open("a", encoding="utf-8") as stream:
                        stream.write("valid-looking extra explanation\n")
                preserve = mock.Mock(return_value=ACCEPTED_COMMIT)
                daemon.record_implementer_candidate = preserve
                daemon.candidate_commit_for_cycle = mock.Mock(
                    return_value=None)
                daemon.worktree_head = mock.Mock(
                    return_value=ACCEPTED_COMMIT)

                with self.assertRaisesRegex(
                        daemon.TicketCycleStateError,
                        "request changed|return changed"):
                    daemon.recover_implementer_deliveries()

                self.assertTrue(request.is_file())
                self.assertFalse((mailbox / "done" / request.name).exists())
                self.assertTrue(receipt.is_file())
                preserve.assert_not_called()

    def test_deleted_real_return_is_not_recovered(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request, returned, receipt, _ = prepare_delivery(
                daemon=daemon, mailbox=mailbox)
            returned.unlink()
            daemon.record_implementer_candidate = mock.Mock(
                return_value=ACCEPTED_COMMIT)
            daemon.candidate_commit_for_cycle = mock.Mock(return_value=None)

            with self.assertRaisesRegex(
                    daemon.TicketCycleStateError,
                    "validated Implementer return has 0"):
                daemon.recover_implementer_deliveries()

            self.assertTrue(request.is_file())
            self.assertTrue(receipt.is_file())

    def test_restart_finishes_both_interrupted_archive_states(self):
        for source_still_present in (True, False):
            with self.subTest(source_still_present=source_still_present), \
                    scratch_daemon() as (daemon, _, mailbox, _):
                request, _, receipt, _ = prepare_delivery(
                    daemon=daemon, mailbox=mailbox)
                done = mailbox / "done"
                done.mkdir()
                guard = pathlib.Path(str(request) + daemon.STATE_GUARD_SUFFIX)
                request_inode = request.stat().st_ino
                pathlib.Path(guard).hardlink_to(request)
                (done / request.name).hardlink_to(request)
                if not source_still_present:
                    request.unlink()
                daemon.candidate_commit_for_cycle = mock.Mock(
                    return_value=ACCEPTED_COMMIT)
                preserve = mock.Mock(return_value=ACCEPTED_COMMIT)
                daemon.record_implementer_candidate = preserve

                self.assertEqual(daemon.recover_implementer_deliveries(), 1)

                archived = done / request.name
                self.assertEqual(archived.stat().st_ino, request_inode)
                self.assertFalse(request.exists())
                self.assertFalse(guard.exists())
                self.assertFalse(receipt.exists())
                self.assertEqual(
                    daemon.inflight_lane_blockers().get(
                        daemon.mailbox_lane_cwd(agent="opus"), []), [])
                preserve.assert_not_called()

    def test_candidate_refusal_keeps_the_recovery_evidence(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request, _, receipt, _ = prepare_delivery(
                daemon=daemon, mailbox=mailbox)
            daemon.record_implementer_candidate = mock.Mock(
                side_effect=daemon.TicketCycleStateError(
                    "candidate state disagrees"))
            daemon.candidate_commit_for_cycle = mock.Mock(return_value=None)
            daemon.worktree_head = mock.Mock(return_value=ACCEPTED_COMMIT)

            with self.assertRaisesRegex(
                    daemon.TicketCycleStateError,
                    "candidate state disagrees"):
                daemon.recover_implementer_deliveries()

            self.assertTrue(request.is_file())
            self.assertTrue(receipt.is_file())
            self.assertFalse((mailbox / "done" / request.name).exists())

    def test_ordinary_candidate_refusal_parks_instead_of_blocking_restart(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            cycle = SCRATCH_HIGH_ANCHOR + "@" + BASE_COMMIT
            flow = ("MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                    + "\nMAILBOX-MODE: normal\n\n")
            request = mailbox / "0001-to-opus.md"
            request.write_text(flow + "Implement the exact ticket.\n",
                               encoding="utf-8")
            returned = mailbox / "0999-to-fable.md"
            return_body = (
                flow + "- **Candidate commit:** `" + ACCEPTED_COMMIT
                + "`\n")
            daemon.ACTIVE_TOPOLOGY = {"scratch": True}
            daemon.prepare_implementer_evidence_contract = mock.Mock(
                return_value={})
            daemon.prepare_implementer_cycle_checkout = mock.Mock(
                return_value=BASE_COMMIT)
            daemon.matching_new_implementer_handoff = mock.Mock(
                return_value=(str(returned), [], None, True))
            daemon.worktree_head = mock.Mock(return_value=ACCEPTED_COMMIT)
            daemon.record_implementer_candidate = mock.Mock(
                side_effect=daemon.TicketCycleStateError(
                    "forbidden candidate path"))
            daemon.candidate_commit_for_cycle = mock.Mock(return_value=None)

            outcome, _ = captured_dispatch(
                daemon=daemon, path=request, fix_only=False, launches=[],
                review_receipt=return_body)

            self.assertFalse(outcome)
            self.assertTrue((mailbox / "failed" / request.name).is_file())
            self.assertFalse(list(mailbox.glob(
                daemon.IMPLEMENTER_DELIVERY_PREFIX + "*")))
            self.assertTrue(returned.is_file())

    def test_receipt_write_value_error_also_parks_the_request(self):
        with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
            cycle = SCRATCH_HIGH_ANCHOR + "@" + BASE_COMMIT
            flow = ("MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                    + "\nMAILBOX-MODE: normal\n\n")
            request = mailbox / "0001-to-opus.md"
            request.write_text(flow + "Implement the exact ticket.\n",
                               encoding="utf-8")
            returned = mailbox / "0999-to-fable.md"
            daemon.ACTIVE_TOPOLOGY = {"scratch": True}
            daemon.prepare_implementer_evidence_contract = mock.Mock(
                return_value={})
            daemon.prepare_implementer_cycle_checkout = mock.Mock(
                return_value=BASE_COMMIT)
            daemon.matching_new_implementer_handoff = mock.Mock(
                return_value=(str(returned), [], None, True))
            daemon.worktree_head = mock.Mock(return_value=ACCEPTED_COMMIT)
            daemon.write_implementer_delivery_receipt = mock.Mock(
                side_effect=ValueError("receipt file changed"))
            preserve = mock.Mock(return_value=ACCEPTED_COMMIT)
            daemon.record_implementer_candidate = preserve

            outcome, _ = captured_dispatch(
                daemon=daemon, path=request, fix_only=False, launches=[],
                review_receipt=(
                    flow + "- **Candidate commit:** `" + ACCEPTED_COMMIT
                    + "`\n"))

            self.assertFalse(outcome)
            self.assertTrue((mailbox / "failed" / request.name).is_file())
            preserve.assert_not_called()


class MailboxStateMoveRecoveryTests(unittest.TestCase):
    """Recover only exact hardlinks left by an interrupted mailbox move."""

    def test_restart_repairs_claim_after_source_unlink_failure(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request = mailbox / "0001-to-fable.md"
            request.write_text("one request\n", encoding="utf-8")
            with mock.patch.object(
                    daemon.os, "unlink",
                    side_effect=OSError("injected unlink failure")):
                with self.assertRaisesRegex(OSError, "injected"):
                    daemon.claim_message(str(request))
            inflight = mailbox / "inflight" / request.name
            self.assertEqual(request.stat().st_ino, inflight.stat().st_ino)

            self.assertEqual(daemon.recover_interrupted_mailbox_moves(), 1)

            self.assertTrue(request.is_file())
            self.assertFalse(inflight.exists())
            self.assertEqual(daemon.inflight_lane_blockers(), {})

    def test_restart_finishes_exact_terminal_hardlinks(self):
        for state in ("done", "failed", "prelaunch"):
            with self.subTest(state=state), scratch_daemon() as (
                    daemon, _, mailbox, _):
                inflight_dir = mailbox / "inflight"
                inflight_dir.mkdir()
                request = inflight_dir / "0001-to-opus.md"
                request.write_text("one request\n", encoding="utf-8")
                guard = pathlib.Path(
                    str(request) + daemon.STATE_GUARD_SUFFIX)
                guard.hardlink_to(request)
                destination_dir = mailbox / state
                destination_dir.mkdir()
                destination = destination_dir / request.name
                destination.hardlink_to(request)

                self.assertEqual(
                    daemon.recover_interrupted_mailbox_moves(), 1)

                self.assertTrue(destination.is_file())
                self.assertFalse(request.exists())
                self.assertFalse(guard.exists())
                self.assertEqual(daemon.inflight_lane_blockers(), {})

    def test_different_inodes_remain_a_hard_stop(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request = mailbox / "0001-to-fable.md"
            request.write_text("pending\n", encoding="utf-8")
            inflight = mailbox / "inflight"
            inflight.mkdir()
            claimed = inflight / request.name
            claimed.write_text("different\n", encoding="utf-8")

            with self.assertRaisesRegex(
                    daemon.TicketCycleStateError, "conflicting states"):
                daemon.recover_interrupted_mailbox_moves()

            self.assertTrue(request.is_file())
            self.assertTrue(claimed.is_file())


class ArchitectDeliveryRecoveryTests(unittest.TestCase):
    """Do not rerun an audit whose exact GO or repair was already saved."""

    def test_restart_finishes_validated_go_and_repair_deliveries(self):
        for outcome_agent in ("daemon", "opus"):
            with self.subTest(outcome_agent=outcome_agent), \
                    scratch_daemon() as (daemon, _, mailbox, _):
                request, outcome, receipt = prepare_architect_delivery(
                    daemon=daemon, mailbox=mailbox,
                    outcome_agent=outcome_agent)
                original_outcome = outcome.read_bytes()
                daemon.candidate_commit_for_cycle = mock.Mock(
                    return_value=ACCEPTED_COMMIT)

                self.assertEqual(daemon.recover_implementer_deliveries(), 1)

                self.assertFalse(request.exists())
                self.assertTrue((mailbox / "done" / request.name).is_file())
                self.assertEqual(outcome.read_bytes(), original_outcome)
                self.assertFalse(receipt.exists())
                self.assertEqual(daemon.recover_implementer_deliveries(), 0)

    def test_wrong_saved_candidate_keeps_recovery_evidence(self):
        with scratch_daemon() as (daemon, _, mailbox, _):
            request, outcome, receipt = prepare_architect_delivery(
                daemon=daemon, mailbox=mailbox, outcome_agent="daemon")
            daemon.candidate_commit_for_cycle = mock.Mock(
                return_value="3" * 40)

            with self.assertRaisesRegex(
                    daemon.TicketCycleStateError, "exact candidate"):
                daemon.recover_implementer_deliveries()

            self.assertTrue(request.is_file())
            self.assertTrue(outcome.is_file())
            self.assertTrue(receipt.is_file())

if __name__ == "__main__":
    unittest.main()
