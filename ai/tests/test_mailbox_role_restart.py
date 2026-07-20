"""Restart interrupted cheap roles from their saved Architect handoff."""

import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from ai.tools import mailbox_daemon as daemon


def git(repo, *arguments):
  """Run one Git command and return its stripped output."""
  result = subprocess.run(
    ["git", "-C", str(repo), *arguments],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
  )
  return result.stdout.strip()


@contextlib.contextmanager
def scratch_restart():
  """Provide one repository and a separate disposable mailbox."""
  with tempfile.TemporaryDirectory(prefix="mailbox-role-restart-") as tmp:
    root = Path(tmp)
    repo = root / "repo"
    mailbox = root / "mailbox"
    repo.mkdir()
    mailbox.mkdir()
    for name in ("inflight", "failed", "prelaunch", "done"):
      (mailbox / name).mkdir()
    git(repo, "init", "--quiet")
    git(repo, "config", "user.name", "Mailbox Test")
    git(repo, "config", "user.email", "mailbox@example.invalid")
    git(repo, "checkout", "-q", "-b", "main")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "base")
    base = git(repo, "rev-parse", "HEAD")
    worktrees = {"fable": str(repo), "opus": str(repo), "sol": str(repo)}
    with mock.patch.multiple(
        daemon,
        REPO_ROOT=str(repo),
        MAILBOX=str(mailbox),
        DONE=str(mailbox / "done"),
        RELAY_DIR=str(root / "relay"),
        AGENT_CWD=worktrees,
        IMPLEMENTER_BRANCH="refs/heads/main",
        SOL_BRANCH="refs/heads/main"):
      yield repo, mailbox, base


def active_implementation(cycle_id):
  """Write one active cycle that has not produced candidate C."""
  state = daemon.empty_ticket_cycle_state()
  state["active"][cycle_id] = {
    "phase": "implementation",
    "commit": None,
    "mode": "normal",
    "route": "primary",
  }
  daemon.write_ticket_cycle_state(state=state)


def architect_handoff(cycle_id):
  """Return the smallest valid saved Implementer plan."""
  return (
    "MAILBOX-FLOW: ticket\n"
    "MAILBOX-CYCLE: " + cycle_id + "\n"
    "MAILBOX-MODE: normal\n\n"
    "- **Directive:** [ai/notes/ticket.md, Implementation directive]\n"
  )


class MailboxRoleRestartTests(unittest.TestCase):
  """Protect plans while intentionally discarding interrupted role work."""

  def test_implementer_restart_discards_work_and_requeues_plan(self):
    """Tracked and untracked edits vanish, but the exact plan returns."""
    with scratch_restart() as (repo, mailbox, base):
      cycle_id = "restart-implementer@" + base
      active_implementation(cycle_id=cycle_id)
      handoff = mailbox / "inflight" / "0001-to-opus.md"
      payload = architect_handoff(cycle_id=cycle_id)
      handoff.write_text(payload, encoding="utf-8")
      (repo / "tracked.txt").write_text("partial\n", encoding="utf-8")
      (repo / "partial.py").write_text("unfinished = True\n", encoding="utf-8")

      output = io.StringIO()
      with contextlib.redirect_stdout(output):
        recovered = daemon.restart_implementer_from_architect_handoff()

      root_handoff = mailbox / "0001-to-opus.md"
      self.assertEqual(recovered, str(root_handoff))
      self.assertEqual(root_handoff.read_text(encoding="utf-8"), payload)
      self.assertFalse(handoff.exists())
      self.assertEqual(git(repo, "rev-parse", "HEAD"), base)
      self.assertEqual(git(repo, "status", "--porcelain"), "")
      self.assertIn("Architect handoff preserved", output.getvalue())
      self.assertIn("Implementer work discarded", output.getvalue())

  def test_implementer_restart_refuses_a_saved_candidate(self):
    """Candidate C belongs to Architect audit and is never discarded."""
    with scratch_restart() as (repo, mailbox, base):
      cycle_id = "candidate-exists@" + base
      active_implementation(cycle_id=cycle_id)
      handoff = mailbox / "0001-to-opus.md"
      handoff.write_text(
        architect_handoff(cycle_id=cycle_id), encoding="utf-8")
      (repo / "tracked.txt").write_text("candidate\n", encoding="utf-8")

      with mock.patch.object(
          daemon, "candidate_commit_for_cycle", return_value=base):
        with self.assertRaisesRegex(
            daemon.TicketCycleStateError, "already produced candidate C"):
          daemon.restart_implementer_from_architect_handoff()

      self.assertEqual(
        (repo / "tracked.txt").read_text(encoding="utf-8"), "candidate\n")
      self.assertTrue(handoff.exists())

  def test_implementer_restart_refuses_an_unprocessed_return(self):
    """A written Implementer handoff is precious even before registration."""
    with scratch_restart() as (repo, mailbox, base):
      cycle_id = "return-exists@" + base
      active_implementation(cycle_id=cycle_id)
      handoff = mailbox / "0001-to-opus.md"
      handoff.write_text(
        architect_handoff(cycle_id=cycle_id), encoding="utf-8")
      returned = mailbox / "0002-to-fable.md"
      returned.write_text(
        "MAILBOX-FLOW: ticket\n"
        "MAILBOX-CYCLE: " + cycle_id + "\n"
        "MAILBOX-MODE: normal\n\n"
        "### IMPLEMENTER_HANDOFF: READY FOR AUDIT\n",
        encoding="utf-8",
      )
      (repo / "tracked.txt").write_text("returned work\n", encoding="utf-8")

      with self.assertRaisesRegex(
          daemon.TicketCycleStateError, "already returned work"):
        daemon.restart_implementer_from_architect_handoff()

      self.assertEqual(
        (repo / "tracked.txt").read_text(encoding="utf-8"), "returned work\n")

  def test_valid_failed_return_is_recovered_without_rerunning(self):
    """A newly readable handoff preserves C and returns to Architect."""
    with scratch_restart() as (repo, mailbox, base):
      cycle_id = "recover-return@" + base
      active_implementation(cycle_id=cycle_id)
      request = mailbox / "failed" / "0001-to-opus.md"
      request.write_text(
        architect_handoff(cycle_id=cycle_id), encoding="utf-8")
      (repo / "tracked.txt").write_text("candidate\n", encoding="utf-8")
      git(repo, "add", "tracked.txt")
      git(repo, "commit", "-q", "-m", "candidate")
      candidate = git(repo, "rev-parse", "HEAD")
      returned = mailbox / "failed" / "0002-to-fable.md"
      returned.write_text(
        "MAILBOX-FLOW: ticket\n"
        "MAILBOX-CYCLE: " + cycle_id + "\n"
        "MAILBOX-MODE: normal\n\n"
        "### IMPLEMENTER_HANDOFF: READY FOR AUDIT\n"
        "- **Candidate commit:** `" + candidate + "`\n",
        encoding="utf-8",
      )
      contract = mock.Mock()
      contract.DirectiveError = RuntimeError
      contract.validate_implementer_handoff_subagent_evidence.return_value = {
        "completion_ready": True}
      evidence = {
        "contract": contract,
        "parallel_work_plan": {"mode": "subagents"},
      }

      with mock.patch.object(
          daemon, "prepare_implementer_evidence_contract",
          return_value=evidence):
        self.assertEqual(daemon.recover_failed_implementer_returns(), 1)
        self.assertEqual(daemon.recover_implementer_deliveries(), 1)

      self.assertEqual(
        daemon.candidate_commit_for_cycle(cycle_id=cycle_id), candidate)
      self.assertTrue((mailbox / "0002-to-fable.md").exists())
      self.assertTrue((mailbox / "done" / "0001-to-opus.md").exists())

  def test_redteam_restart_discards_work_and_requeues_request(self):
    """A failed Red Team request returns without preserving its edits."""
    with scratch_restart() as (repo, mailbox, base):
      request = mailbox / "failed" / "0002-to-sol.md"
      payload = daemon.sol_ticket_payload(
        ticket_kind="closure",
        review_cycle="redteam-restart@" + base,
        review_commit=base,
        text="Review this exact landing only.",
      )
      request.write_text(payload, encoding="utf-8")
      (repo / "tracked.txt").write_text("red-team draft\n", encoding="utf-8")
      (repo / "notes.tmp").write_text("discard me\n", encoding="utf-8")

      with mock.patch.object(
          daemon, "redteam_closure_problem", return_value=None), \
          mock.patch.object(
            daemon, "any_matching_redteam_receipt", return_value=False), \
          mock.patch.object(daemon, "discard_interrupted_audit_snapshot"):
        recovered = daemon.restart_redteam_from_architect_handoff()

      root_request = mailbox / "0002-to-sol.md"
      self.assertEqual(recovered, str(root_request))
      self.assertEqual(root_request.read_text(encoding="utf-8"), payload)
      self.assertEqual(git(repo, "rev-parse", "HEAD"), base)
      self.assertEqual(git(repo, "status", "--porcelain"), "")

  def test_baseline_sync_preserves_current_sealed_architect_backlog(self):
    """An active backlog note may stay dirty when Architect is at target."""
    target = "a" * 40
    old_base = "b" * 40
    paths = {
      "fable": "/roles/architect",
      "opus": "/roles/implementer",
      "sol": "/roles/redteam",
    }
    ticket_state = daemon.empty_ticket_cycle_state()
    ticket_state["active"]["active-ticket@" + old_base] = {
      "phase": "implementation",
      "commit": None,
      "mode": "normal",
      "route": "primary",
    }

    def head(worktree):
      return target if worktree != paths["opus"] else old_base

    def status(worktree):
      return b" M ai/notes/backlog.md\0" if worktree == paths["fable"] else b""

    with mock.patch.multiple(
        daemon,
        AGENT_CWD=paths,
        AGENT_BRANCH={"fable": "refs/heads/architect"},
        IMPLEMENTER_BRANCH="refs/heads/implementer",
        SOL_BRANCH="refs/heads/redteam"), \
        mock.patch.object(daemon, "read_candidate_state", return_value={
          "schema": daemon.CANDIDATE_STATE_SCHEMA, "cycles": {}}), \
        mock.patch.object(
          daemon, "read_ticket_cycle_state", return_value=ticket_state), \
        mock.patch.object(daemon, "_symbolic_worktree_branch"), \
        mock.patch.object(daemon, "worktree_head", side_effect=head), \
        mock.patch.object(daemon, "_clean_worktree_status", side_effect=status), \
        mock.patch.object(
          daemon, "_architect_only_sealed_backlog", return_value=b"sealed"), \
        mock.patch.object(daemon, "_require_ancestor_or_same"), \
        mock.patch.object(daemon, "git_commit_descends_from", return_value=True):
      plan = daemon._role_baseline_plan_locked(target=target)

    self.assertIn(
      (paths["fable"], "refs/heads/architect", "Architect", False), plan)
    self.assertIn(
      (paths["opus"], "refs/heads/implementer", "Implementer preserved work", False),
      plan,
    )


if __name__ == "__main__":
  unittest.main()
