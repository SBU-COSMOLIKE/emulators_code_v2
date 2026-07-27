"""CPU tests binding an Architect GO to the exact candidate it approved.

``matching_new_architect_go`` is the check that stops the daemon landing a
commit the Architect never accepted. The Architect audits one candidate commit
and writes a decision-only GO naming it; the daemon then lands that commit and
no other. If the GO's candidate were not compared, a GO written for one
candidate would authorize landing a different one.

Mutation testing found the function had no test of any kind: removing the
candidate comparison left the whole suite green.
"""

import os
import unittest

from ai.tests.tools_mailbox_daemon_ticket_cycle_repro import scratch_daemon


ANCHOR = "open-example-ticket"
CYCLE = ANCHOR + "@" + "1" * 40
CANDIDATE = "a" * 40
OTHER_CANDIDATE = "b" * 40
MODE = "normal"


class ArchitectGoCandidateBindingTests(unittest.TestCase):
  """Require a fresh GO to name this turn's exact cycle, candidate, mode."""

  def _write_go(self, daemon, mailbox, name, cycle, candidate, mode):
    """Write one decision-only Architect GO into the mailbox.

    Arguments:
      daemon    = the scratch daemon namespace.
      mailbox   = the scratch mailbox directory.
      name      = the message filename on the to-daemon route.
      cycle     = the cycle identifier the GO names.
      candidate = the candidate commit the GO names.
      mode      = the ticket mode the GO names.

    Returns:
      The written message path.
    """
    payload = daemon.architect_go_request_payload(
      cycle_id=cycle, candidate_commit=candidate, mode=mode)
    path = os.path.join(str(mailbox), name)
    with open(path, "w", encoding="utf-8") as handle:
      handle.write(payload)
    return path

  def test_a_go_naming_this_exact_candidate_is_accepted(self):
    """The baseline must pass, or the refusals below prove nothing."""
    with scratch_daemon() as (daemon, mailbox):
      written = self._write_go(
        daemon, mailbox, "0001-to-daemon.md", CYCLE, CANDIDATE, MODE)
      path, offending, problem = daemon.matching_new_architect_go(
        cycle_id=CYCLE, candidate_commit=CANDIDATE, mode=MODE,
        before_inodes=set())
      self.assertIsNone(problem)
      self.assertEqual(offending, [])
      self.assertEqual(path, written)

  def test_a_go_naming_a_different_candidate_is_refused(self):
    """A GO written for another commit must not authorize this landing."""
    with scratch_daemon() as (daemon, mailbox):
      written = self._write_go(
        daemon, mailbox, "0001-to-daemon.md", CYCLE, OTHER_CANDIDATE, MODE)
      path, offending, problem = daemon.matching_new_architect_go(
        cycle_id=CYCLE, candidate_commit=CANDIDATE, mode=MODE,
        before_inodes=set())
      self.assertIsNone(path)
      self.assertIn(written, offending)
      self.assertIn("exact cycle, candidate, and mode", problem)

  def test_a_go_naming_a_different_cycle_is_refused(self):
    """The same candidate under another cycle is still the wrong turn."""
    other_cycle = ANCHOR + "@" + "2" * 40
    with scratch_daemon() as (daemon, mailbox):
      self._write_go(
        daemon, mailbox, "0001-to-daemon.md", other_cycle, CANDIDATE, MODE)
      path, _offending, problem = daemon.matching_new_architect_go(
        cycle_id=CYCLE, candidate_commit=CANDIDATE, mode=MODE,
        before_inodes=set())
      self.assertIsNone(path)
      self.assertIn("exact cycle, candidate, and mode", problem)

  def test_two_fresh_decisions_are_refused_rather_than_chosen_between(self):
    """Two GO messages leave no single answer, so neither may be used."""
    with scratch_daemon() as (daemon, mailbox):
      self._write_go(
        daemon, mailbox, "0001-to-daemon.md", CYCLE, CANDIDATE, MODE)
      self._write_go(
        daemon, mailbox, "0002-to-daemon.md", CYCLE, CANDIDATE, MODE)
      path, offending, problem = daemon.matching_new_architect_go(
        cycle_id=CYCLE, candidate_commit=CANDIDATE, mode=MODE,
        before_inodes=set())
      self.assertIsNone(path)
      self.assertEqual(len(offending), 2)
      self.assertIn("at most one", problem)

  def test_a_message_already_present_before_the_turn_is_not_fresh(self):
    """Only decisions this turn produced may authorize this turn."""
    with scratch_daemon() as (daemon, mailbox):
      written = self._write_go(
        daemon, mailbox, "0001-to-daemon.md", CYCLE, CANDIDATE, MODE)
      before = {daemon.regular_inode(path=written)}
      path, offending, problem = daemon.matching_new_architect_go(
        cycle_id=CYCLE, candidate_commit=CANDIDATE, mode=MODE,
        before_inodes=before)
      self.assertIsNone(path)
      self.assertEqual(offending, [])
      self.assertIsNone(problem)


if __name__ == "__main__":
  unittest.main()
