"""Check the exact record passed to a replacement Implementer."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ai.tools import mailbox_daemon as daemon


BASE = "a" * 40
HEAD = "b" * 40
CYCLE = "open-context-handoff@" + BASE


def handoff(*, cycle=CYCLE, base=BASE, head=HEAD, candidate="no",
            uncommitted=("none",)):
    """Return one complete context-handoff mailbox message."""
    def bullets(items):
        return "\n".join("- " + item for item in items)

    body = f"""{daemon.CONTEXT_HANDOFF_HEADING}

- **Ticket and cycle:** `{cycle}`
- **Base commit:** `{base}`
- **Current worktree HEAD:** `{head}`
- **Candidate created:** `{candidate}`

#### Completed
- focused test written

#### Known failures
- one assertion still fails

#### Rejected approaches
- do not weaken the assertion

#### Uncommitted changes
{bullets(uncommitted)}

#### Next exact action
- inspect the failing comparison

#### Do not revisit
- weakening the assertion
"""
    return ("MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
            + "\nMAILBOX-MODE: normal\n\n" + body)


class ContextHandoffTests(unittest.TestCase):
    """Keep replacement context small, exact, and tied to repository state."""

    def validate(self, message, *, head=HEAD, dirty=False):
        with mock.patch.object(daemon, "worktree_head", return_value=head), \
                mock.patch.object(
                    daemon, "_clean_worktree_status",
                    return_value=(b" M source.py\0" if dirty else b"")):
            return daemon.context_handoff_problem(
                message=message, expected_cycle=CYCLE,
                expected_mode="normal")

    def test_clean_record_is_valid(self):
        self.assertIsNone(self.validate(handoff()))
        record = daemon.parse_context_handoff(
            handoff().split("\n\n", 1)[1])
        self.assertEqual(record["Ticket and cycle"], CYCLE)
        self.assertEqual(record["sections"]["Do not revisit"],
                         ["weakening the assertion"])

    def test_dirty_record_must_admit_uncommitted_work(self):
        self.assertIsNone(self.validate(
            handoff(uncommitted=("M source.py",)), dirty=True))
        self.assertIn("uncommitted changes", self.validate(
            handoff(uncommitted=("none",)), dirty=True))
        self.assertIn("uncommitted changes", self.validate(
            handoff(uncommitted=("M source.py",)), dirty=False))

    def test_cycle_base_and_head_must_match(self):
        cases = (
            handoff(cycle="other-ticket@" + BASE),
            handoff(base="c" * 40),
            handoff(head="c" * 40),
        )
        for message in cases:
            with self.subTest(message=message[:90]):
                self.assertIsNotNone(self.validate(message))

    def test_candidate_claim_requires_a_clean_changed_commit(self):
        self.assertIsNone(self.validate(handoff(candidate="yes")))
        self.assertIn("clean changed commit", self.validate(
            handoff(candidate="yes", uncommitted=("M source.py",)),
            dirty=True))
        self.assertIn("clean changed commit", self.validate(
            handoff(candidate="yes", head=BASE), head=BASE))

    def test_missing_list_or_placeholder_is_refused(self):
        missing = handoff().replace(
            "#### Do not revisit\n- weakening the assertion\n", "")
        placeholder = handoff().replace(
            "- inspect the failing comparison", "- ...")
        for message in (missing, placeholder):
            body = message.split("\n\n", 1)[1]
            with self.assertRaises(daemon.TicketCycleStateError):
                daemon.parse_context_handoff(body)

    def test_new_record_is_found_and_replacement_reads_exact_path(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(daemon, "MAILBOX", folder), \
                mock.patch.object(daemon, "worktree_head", return_value=HEAD), \
                mock.patch.object(
                    daemon, "_clean_worktree_status", return_value=b""):
            before = Path(folder) / "0001-to-fable.md"
            before.write_text(handoff(), encoding="utf-8")
            snapshot = {daemon.regular_inode(str(before))}
            current = Path(folder) / "0002-to-fable.md"
            current.write_text(handoff(), encoding="utf-8")

            found, invalid, problem = daemon.matching_new_context_handoff(
                cycle_id=CYCLE, mode="normal", before_inodes=snapshot)
            self.assertEqual(found, str(current))
            self.assertEqual(invalid, [])
            self.assertIsNone(problem)

            done = Path(folder) / "done"
            done.mkdir()
            archived = done / current.name
            os.replace(current, archived)
            selected = daemon.latest_context_handoff_path(
                cycle_id=CYCLE, mode="normal")
            self.assertEqual(selected, str(archived))
            notice = daemon.replacement_context_notice(path=selected)
            self.assertIn(str(archived), notice)
            self.assertIn("not a daemon-written summary", notice)

    def test_context_checkpoint_never_receives_landing_instructions(self):
        preamble = daemon.agent_preamble(agent="fable", message=handoff())
        self.assertIn("IMPLEMENTER CONTEXT HANDOFF", preamble)
        self.assertNotIn(daemon.ARCHITECT_LANDING_PREAMBLE, preamble)

    def test_replacement_preserves_the_verified_worktree(self):
        """A replacement must not reset the work described by its record."""
        lock = object()
        active = {"active": {CYCLE: {"phase": "implementation"}}}
        with mock.patch.object(
                daemon, "acquire_ticket_cycle_lock", return_value=lock), \
                mock.patch.object(
                    daemon, "release_ticket_cycle_lock") as release, \
                mock.patch.object(
                    daemon, "read_ticket_cycle_state", return_value=active), \
                mock.patch.object(
                    daemon, "read_candidate_state", return_value={
                        "cycles": {}}), \
                mock.patch.object(
                    daemon, "candidate_record_locked", return_value=None), \
                mock.patch.object(
                    daemon, "worktree_head", return_value=HEAD), \
                mock.patch.object(daemon, "_run_git") as run_git:
            selected = daemon.prepare_implementer_cycle_checkout(
                cycle_id=CYCLE, preserve_current=True)

        self.assertEqual(selected, HEAD)
        run_git.assert_not_called()
        release.assert_called_once_with(lock_file=lock)


if __name__ == "__main__":
    unittest.main()
