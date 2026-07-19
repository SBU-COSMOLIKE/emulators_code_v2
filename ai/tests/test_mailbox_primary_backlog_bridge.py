"""Tests for moving a local backlog into the saved Architect worktree."""

import hashlib
import json
import unittest

from ai.tests.tools_mailbox_daemon_primary_worktree_repro import (
    default_primary,
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
    """Preserve one authoritative backlog across worktree setup."""

    def test_first_live_action_copies_the_exact_sealed_pair(self):
        with scratch_repository() as root:
            source_backlog, source_guard = _paths(root)
            write_exact(source_backlog, b"- OPEN **HIGH** sample ticket\n")
            seal_backlog(root)
            original = (source_backlog.read_bytes(), source_guard.read_bytes())

            rc, output, error = invoke(root, ["--once"])

            target_backlog, target_guard = _paths(default_primary(root))
            self.assertEqual(rc, 0, output + error)
            self.assertEqual(
                (target_backlog.read_bytes(), target_guard.read_bytes()),
                original)
            self.assertEqual(
                (source_backlog.read_bytes(), source_guard.read_bytes()),
                original)

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

    def test_interrupted_exact_copy_resumes_but_conflict_refuses(self):
        with scratch_repository() as root:
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)
            primary = default_primary(root)
            source_backlog, source_guard = _paths(root)
            target_backlog, target_guard = _paths(primary)
            write_exact(source_backlog, b"resume this backlog\n")
            seal_backlog(root)
            write_exact(target_backlog, source_backlog.read_bytes())

            rc, output, error = invoke(root, ["--once"])

            self.assertEqual(rc, 0, output + error)
            self.assertEqual(target_guard.read_bytes(), source_guard.read_bytes())

            target_guard.unlink()
            write_exact(target_backlog, b"different primary bytes\n")
            rc, output, error = invoke(root, ["--once"])
            self.assertNotEqual(rc, 0, output + error)
            self.assertFalse(target_guard.exists())
            self.assertIn("primary backlog conflicts", output)

    def test_bad_source_sha_is_refused_without_a_partial_target(self):
        with scratch_repository() as root:
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)
            source_backlog, source_guard = _paths(root)
            write_exact(source_backlog, b"unsealed edit\n")
            payload = {
                "backlog": "ai/notes/backlog.md",
                "sha256": hashlib.sha256(b"other bytes").hexdigest(),
                "version": 1,
            }
            write_exact(
                source_guard,
                (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))

            rc, output, error = invoke(root, ["--once"])

            target_backlog, target_guard = _paths(default_primary(root))
            self.assertNotEqual(rc, 0, output + error)
            self.assertFalse(target_backlog.exists())
            self.assertFalse(target_guard.exists())
            self.assertIn("backlog differs", output)


if __name__ == "__main__":
    unittest.main()
