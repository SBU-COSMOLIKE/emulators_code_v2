"""Tests for initializing the tracked backlog in the Architect worktree."""

import hashlib
import unittest

from ai.tests.tools_mailbox_daemon_primary_worktree_repro import (
    default_primary,
    git,
    invoke,
    scratch_repository,
    seal_backlog,
    write_exact,
)


def _paths(worktree):
    """Return the backlog and its Architect-written SHA file."""
    notes = worktree / "ai" / "notes"
    return notes / "backlog.md", notes / ".backlog-guard.json"


class PrimaryBacklogBridgeTests(unittest.TestCase):
    """Preserve the Git-tracked backlog across worktree setup."""

    def test_first_live_action_initializes_the_tracked_backlog_guard(self):
        with scratch_repository() as root:
            rc, output, error = invoke(root, ["--once"])

            target_backlog, target_guard = _paths(default_primary(root))
            self.assertEqual(rc, 0, output + error)
            self.assertEqual(target_backlog.read_bytes(), b"")
            self.assertIn(
                hashlib.sha256(b"").hexdigest(),
                target_guard.read_text(encoding="utf-8"))

    def test_existing_primary_backlog_is_not_overwritten(self):
        with scratch_repository() as root:
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)
            primary = default_primary(root)
            target_backlog, target_guard = _paths(primary)
            write_exact(target_backlog, b"Architect primary backlog\n")
            seal_backlog(primary)
            expected = (target_backlog.read_bytes(), target_guard.read_bytes())

            source_backlog, _source_guard = _paths(root)
            write_exact(source_backlog, b"older user-checkout backlog\n")
            seal_backlog(root)
            rc, output, error = invoke(root, ["--once"])

            self.assertEqual(rc, 0, output + error)
            self.assertEqual(
                (target_backlog.read_bytes(), target_guard.read_bytes()),
                expected)

    def test_legacy_untracked_backlog_becomes_the_same_tracked_file(self):
        with scratch_repository() as root:
            git(root, "rm", "ai/notes/backlog.md")
            git(root, "commit", "-m", "legacy local backlog base")
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)
            primary = default_primary(root)
            target_backlog, _target_guard = _paths(primary)

            payload = b"# Execution backlog\n\nMigrated tickets.\n"
            write_exact(root / "ai/notes/backlog.md", payload)
            git(root, "add", "ai/notes/backlog.md")
            git(root, "commit", "-m", "track the backlog")
            write_exact(target_backlog, payload)
            seal_backlog(primary)

            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)
            self.assertEqual(target_backlog.read_bytes(), payload)
            self.assertEqual(
                git(primary, "status", "--porcelain").stdout, "")

    def test_missing_guard_is_recreated_only_for_committed_bytes(self):
        with scratch_repository() as root:
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)
            primary = default_primary(root)
            target_backlog, target_guard = _paths(primary)
            target_guard.unlink()

            rc, output, error = invoke(root, ["--once"])

            self.assertEqual(rc, 0, output + error)
            self.assertTrue(target_guard.is_file())

            target_guard.unlink()
            write_exact(target_backlog, b"different primary bytes\n")
            rc, output, error = invoke(root, ["--once"])
            self.assertNotEqual(rc, 0, output + error)
            self.assertFalse(target_guard.exists())
            self.assertIn("backlog and its Architect-sealed guard", output)

    def test_clean_committed_backlog_update_refreshes_the_local_guard(self):
        with scratch_repository() as root:
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)

            payload = b"# Execution backlog\n\nOne committed update.\n"
            source_backlog, _source_guard = _paths(root)
            write_exact(source_backlog, payload)
            git(root, "add", "ai/notes/backlog.md")
            git(root, "commit", "-m", "update tracked backlog")

            rc, output, error = invoke(root, ["--once"])

            primary = default_primary(root)
            target_backlog, target_guard = _paths(primary)
            self.assertEqual(rc, 0, output + error)
            self.assertEqual(target_backlog.read_bytes(), payload)
            self.assertIn(
                hashlib.sha256(payload).hexdigest(),
                target_guard.read_text(encoding="utf-8"))
            self.assertEqual(git(primary, "status", "--porcelain").stdout, "")


if __name__ == "__main__":
    unittest.main()
