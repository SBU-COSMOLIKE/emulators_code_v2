"""Recovery test for a candidate ref saved before its state record."""

import unittest
from unittest import mock

from ai.tests.tools_mailbox_daemon_fix_only_repro import ACCEPTED_COMMIT
from ai.tests.tools_mailbox_daemon_fix_only_repro import BASE_COMMIT
from ai.tests.tools_mailbox_daemon_fix_only_repro import scratch_daemon


class MailboxCandidateStateRecoveryTests(unittest.TestCase):
    """Keep a repair candidate reachable after an interrupted state write."""

    def test_recovery_adopts_a_repair_ref_written_before_state(self):
        with scratch_daemon() as (daemon, _, _, _):
            cycle = "scratch-repair-candidate@" + BASE_COMMIT
            prior = "b" * 40
            state = {
                "schema": daemon.CANDIDATE_STATE_SCHEMA,
                "cycles": {cycle: {
                    "ref": daemon.cycle_candidate_ref(cycle_id=cycle),
                    "commit": prior,
                }},
            }
            ticket_state = {"active": {
                cycle: {"phase": "implementation"}}}
            daemon.git_ref_commit = mock.Mock(return_value=ACCEPTED_COMMIT)
            daemon.git_commit_descends_from = mock.Mock(
                side_effect=lambda starting_commit, accepted_commit:
                starting_commit in {BASE_COMMIT, prior}
                and accepted_commit == ACCEPTED_COMMIT)
            write = mock.Mock()
            daemon.write_candidate_state = write

            record = daemon.candidate_record_locked(
                cycle_id=cycle, ticket_state=ticket_state,
                candidate_state=state)

            self.assertEqual(record["commit"], ACCEPTED_COMMIT)
            self.assertEqual(state["cycles"][cycle]["commit"],
                             ACCEPTED_COMMIT)
            write.assert_called_once_with(state=state)


if __name__ == "__main__":
    unittest.main()
