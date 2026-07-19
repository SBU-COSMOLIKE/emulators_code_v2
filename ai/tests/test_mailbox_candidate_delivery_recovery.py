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

if __name__ == "__main__":
    unittest.main()
