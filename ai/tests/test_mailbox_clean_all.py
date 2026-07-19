#!/usr/bin/env python3
"""Regression tests for the explicit destructive mailbox cleanup command.

Every test uses a disposable Git repository.  The real repository's
worktrees, branches, mailbox, tags, remote-tracking refs, and stash are never
touched.
"""

import fcntl
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

from ai.tests.tools_mailbox_daemon_primary_worktree_repro import (
    DAEMON_PATH,
    git,
    invoke,
    managed_base,
    scratch_repository,
    write_exact,
)


AI_BRANCH_PREFIXES = (
    "refs/heads/claude/",
    "refs/heads/codex/",
    "refs/heads/worktree-agent-",
)
STATE_FILES = (
    ".mailbox-primary-worktree.json",
    ".mailbox-implementer-worktree.json",
    ".mailbox-sol-worktree.json",
)


def add_worktree(root, path, branch=None, detach=False):
    """Create one registered scratch worktree from ``main``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if detach:
        git(root, "worktree", "add", "--detach", str(path), "main")
    else:
        git(root, "worktree", "add", "-b", branch, str(path), "main")
    return path


def local_refs(root, prefix):
    """Return exact refs below one namespace in a scratch repository."""
    output = git(
        root, "for-each-ref", "--format=%(refname)", prefix).stdout
    return set(output.splitlines())


def worktree_paths(root):
    """Return registered worktree paths reported by Git."""
    output = git(root, "worktree", "list", "--porcelain").stdout
    return {
        str(Path(line[len("worktree "):]).resolve())
        for line in output.splitlines()
        if line.startswith("worktree ")
    }


def load_daemon():
    """Load an isolated daemon module for direct lock tests."""
    spec = importlib.util.spec_from_file_location(
        "mailbox_clean_all_test_daemon", DAEMON_PATH)
    daemon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daemon)
    return daemon


def prepare_ai_work(root):
    """Create dirty, unmerged, detached, and abandoned AI work."""
    managed = managed_base(root)
    managed.mkdir(parents=True, exist_ok=True)

    claude = add_worktree(
        root, managed / "dirty-claude", branch="claude/dirty-cleanup")
    with (claude / "ai" / "tools" / "mailbox_daemon.py").open(
            "a", encoding="utf-8") as stream:
        stream.write("\n# uncommitted Claude work that cleanup discards\n")

    codex = add_worktree(
        root, managed / "unmerged-codex", branch="codex/unmerged-cleanup")
    write_exact(codex / "unmerged.txt", b"not merged into main\n")
    git(codex, "add", "unmerged.txt")
    git(codex, "commit", "-m", "unmerged Sol work")

    legacy = add_worktree(
        root, managed / "legacy-agent",
        branch="worktree-agent-deadbeef")
    write_exact(legacy / "legacy-dirty.txt", b"discard this legacy work\n")

    add_worktree(
        root, managed / "mailbox-audit-detached", detach=True)
    git(root, "branch", "codex/stale-without-worktree", "main")
    write_exact(
        managed / "abandoned-unregistered" / "evidence.txt",
        b"not registered with Git\n")
    for name in STATE_FILES:
        write_exact(managed / name, b"stale local state\n")

    return {claude, codex, legacy}


class MailboxCleanAllTest(unittest.TestCase):
    """Pin what ``--clean-all`` destroys and what it must preserve."""

    def test_deletes_all_ai_work_but_preserves_user_git_state(self):
        """Dirty and unmerged AI work goes; user-owned Git state stays."""
        with scratch_repository() as root:
            ai_paths = prepare_ai_work(root)

            user_worktree = add_worktree(
                root, managed_base(root) / "student-kept-worktree",
                branch="student/keep")
            # These two user-owned refs deliberately contain AI-looking
            # words. Cleanup targets only local branch refs, not every ref
            # whose spelling happens to mention Claude or Codex.
            git(root, "tag", "claude/preserved-tag", "main")
            git(root, "update-ref",
                "refs/remotes/origin/codex/preserved-remote", "main")
            with (root / ".gitignore").open("a", encoding="utf-8") as stream:
                stream.write("\n# student's uncommitted setting\n")
            git(root, "stash", "push", "-m", "student scratch state")

            main_before = git(root, "rev-parse", "main").stdout.strip()
            user_before = git(
                root, "rev-parse", "student/keep").stdout.strip()
            tag_before = git(
                root, "rev-parse", "claude/preserved-tag").stdout.strip()
            remote_before = git(
                root, "rev-parse",
                "refs/remotes/origin/codex/preserved-remote").stdout.strip()
            stash_before = git(root, "rev-parse", "refs/stash").stdout.strip()

            rc, stdout, stderr = invoke(root, ["--clean-all"])
            self.assertEqual(rc, 0, stdout + stderr)

            refs = local_refs(root, "refs/heads")
            for prefix in AI_BRANCH_PREFIXES:
                self.assertFalse(
                    any(ref.startswith(prefix) for ref in refs), refs)
            self.assertIn("refs/heads/main", refs)
            self.assertIn("refs/heads/student/keep", refs)

            self.assertEqual(git(root, "rev-parse", "main").stdout.strip(),
                             main_before)
            self.assertEqual(
                git(root, "rev-parse", "student/keep").stdout.strip(),
                user_before)
            self.assertEqual(
                git(root, "rev-parse",
                    "claude/preserved-tag").stdout.strip(),
                tag_before)
            remote_after = git(
                root, "rev-parse",
                "refs/remotes/origin/codex/preserved-remote").stdout.strip()
            self.assertEqual(remote_after, remote_before)
            self.assertEqual(
                git(root, "rev-parse", "refs/stash").stdout.strip(),
                stash_before)

            remaining = worktree_paths(root)
            self.assertEqual(
                remaining, {str(root.resolve()), str(user_worktree.resolve())})
            for path in ai_paths:
                self.assertFalse(path.exists(), path)
            self.assertFalse(
                (managed_base(root) / "mailbox-audit-detached").exists())
            self.assertFalse(
                (managed_base(root) / "abandoned-unregistered").exists())
            for name in STATE_FILES:
                self.assertFalse((managed_base(root) / name).exists(), name)

            second_rc, second_stdout, second_stderr = invoke(
                root, ["--clean-all"])
            self.assertEqual(
                second_rc, 0, second_stdout + second_stderr)
            self.assertEqual(worktree_paths(root), remaining)

    def test_active_dispatch_lock_refuses_before_deleting_anything(self):
        """A running watcher wins the race and leaves AI work untouched."""
        with scratch_repository() as root:
            managed = managed_base(root)
            ai_worktree = add_worktree(
                root, managed / "live-claude", branch="claude/live")
            lock_path = root / "ai" / "notes" / "mailbox" / ".dispatch.lock"
            write_exact(lock_path, b'{"mode":"watch","pid":1}\n')
            with lock_path.open("r+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                rc, stdout, stderr = invoke(root, ["--clean-all"])
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

            report = (stdout + stderr).lower()
            self.assertNotEqual(rc, 0, report)
            self.assertNotIn("unrecognized arguments", report)
            self.assertTrue(
                "lock" in report or "running" in report
                or "watch" in report, report)
            self.assertTrue(ai_worktree.exists())
            self.assertIn("refs/heads/claude/live",
                          local_refs(root, "refs/heads"))

    def test_cleanup_holds_transport_locks_until_deletion_starts(self):
        """A sender cannot enter after cleanup's final idle check."""
        daemon = load_daemon()
        with scratch_repository() as root:
            managed = managed_base(root)
            ai_worktree = add_worktree(
                root, managed / "locked-claude", branch="claude/locked")
            mailbox = ai_worktree / "ai" / "notes" / "mailbox"
            mailbox.mkdir(parents=True, exist_ok=True)
            observed = []
            real_run_git = daemon._run_git

            def inspect_first_prune(repository_root, arguments, check=True,
                                    input_bytes=None):
                if arguments == ["worktree", "prune"] and not observed:
                    for name in (".dispatch.lock", ".sequence.lock"):
                        with (mailbox / name).open("r+") as probe:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(
                                    probe.fileno(),
                                    fcntl.LOCK_EX | fcntl.LOCK_NB)
                    observed.append(True)
                return real_run_git(
                    repository_root, arguments, check=check,
                    input_bytes=input_bytes)

            with mock.patch.object(
                    daemon, "_run_git", side_effect=inspect_first_prune):
                daemon.clean_all_ai_worktrees(
                    repository_root=str(root), current_worktree=str(root))

            self.assertEqual(observed, [True])
            self.assertFalse(ai_worktree.exists())

    def test_actions_refuse_if_cleanup_removed_the_saved_topology(self):
        """A lock waiter cannot publish into a worktree cleanup removed."""
        daemon = load_daemon()
        with scratch_repository() as root:
            mailbox = root / "action-mailbox"
            changing = [object(), daemon.PrimaryWorktreeError("removed")]
            with mock.patch.object(daemon, "MAILBOX", str(mailbox)), \
                    mock.patch.object(
                        daemon, "ACTIVE_TOPOLOGY", {"active": True}), \
                    mock.patch.object(
                        daemon, "validate_live_agent_dispatch_topology",
                        side_effect=changing), \
                    mock.patch.object(
                        daemon, "publish_message_locked") as publish:
                self.assertFalse(daemon.send(
                    agent="fable", text="request", dry_run=False))
                publish.assert_not_called()

            sequence = mailbox / ".sequence.lock"
            with sequence.open("r+") as probe:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)

            changing = [object(), daemon.PrimaryWorktreeError("removed")]
            with mock.patch.object(daemon, "MAILBOX", str(mailbox)), \
                    mock.patch.object(
                        daemon, "ACTIVE_TOPOLOGY", {"active": True}), \
                    mock.patch.object(
                        daemon, "validate_live_agent_dispatch_topology",
                        side_effect=changing):
                self.assertIsNone(daemon.acquire_dispatch_lock(mode="watch"))

            dispatch = mailbox / ".dispatch.lock"
            with dispatch.open("r+") as probe:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)

    def test_clean_all_conflicts_with_once_without_deleting(self):
        """A mixed cleanup/dispatch command is refused without side effects."""
        with scratch_repository() as root:
            ai_worktree = add_worktree(
                root, managed_base(root) / "conflict-codex",
                branch="codex/conflict")
            rc, stdout, stderr = invoke(root, ["--clean-all", "--once"])
            report = (stdout + stderr).lower()
            self.assertNotEqual(rc, 0, report)
            self.assertNotIn("unrecognized arguments", report)
            self.assertIn("clean-all", report)
            self.assertIn("once", report)
            self.assertTrue(ai_worktree.exists())
            self.assertIn("refs/heads/codex/conflict",
                          local_refs(root, "refs/heads"))

    def test_new_role_worktrees_use_role_specific_branch_namespaces(self):
        """Claude uses ``claude/`` and Sol uses ``codex/`` branches."""
        daemon = load_daemon()

        self.assertTrue(daemon.PRIMARY_BRANCH.startswith(
            "refs/heads/claude/"))
        self.assertTrue(daemon.IMPLEMENTER_BRANCH.startswith(
            "refs/heads/claude/"))
        self.assertTrue(daemon.SOL_BRANCH.startswith("refs/heads/codex/"))


if __name__ == "__main__":
    unittest.main()
