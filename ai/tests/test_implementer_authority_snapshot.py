"""Check the small Git snapshot around every Implementer turn."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from ai.tools import mailbox_daemon as daemon


def git(root, *arguments):
  """Run one successful Git command in a temporary repository."""
  return subprocess.run(
      ["git", "-C", str(root), *arguments], check=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


class ImplementerAuthoritySnapshotTests(unittest.TestCase):
  """Detect ref or user-checkout movement without changing that state."""

  def setUp(self):
    self.directory = tempfile.TemporaryDirectory(prefix="authority-snapshot-")
    self.root = Path(self.directory.name)
    git(self.root, "init", "-b", "main")
    git(self.root, "config", "user.name", "Snapshot Test")
    git(self.root, "config", "user.email", "snapshot@example.invalid")
    (self.root / "value.txt").write_text("one\n", encoding="utf-8")
    git(self.root, "add", "value.txt")
    git(self.root, "commit", "-m", "initial")
    head = git(self.root, "rev-parse", "HEAD").decode().strip()
    git(self.root, "update-ref", "refs/remotes/origin/main", head)

  def tearDown(self):
    self.directory.cleanup()

  def test_unchanged_snapshot_has_no_problem(self):
    before = daemon.implementer_authority_snapshot(str(self.root))
    self.assertEqual(
        daemon.implementer_authority_changes(before, str(self.root)), [])

  def test_local_main_and_checkout_movement_are_named(self):
    before = daemon.implementer_authority_snapshot(str(self.root))
    (self.root / "value.txt").write_text("two\n", encoding="utf-8")
    git(self.root, "commit", "-am", "move main")

    changes = daemon.implementer_authority_changes(before, str(self.root))
    self.assertIn("local main", changes)
    self.assertIn("user checkout HEAD", changes)

  def test_origin_main_movement_is_named(self):
    before = daemon.implementer_authority_snapshot(str(self.root))
    git(self.root, "update-ref", "-d", "refs/remotes/origin/main")

    self.assertEqual(
        daemon.implementer_authority_changes(before, str(self.root)),
        ["origin/main"])

  def test_user_file_or_branch_change_is_named(self):
    before = daemon.implementer_authority_snapshot(str(self.root))
    (self.root / "value.txt").write_text("working change\n", encoding="utf-8")
    self.assertEqual(
        daemon.implementer_authority_changes(before, str(self.root)),
        ["user checkout status"])

    git(self.root, "reset", "--hard", "HEAD")
    before = daemon.implementer_authority_snapshot(str(self.root))
    git(self.root, "checkout", "--detach", "HEAD")
    self.assertEqual(
        daemon.implementer_authority_changes(before, str(self.root)),
        ["user checkout branch"])


if __name__ == "__main__":
  unittest.main()
