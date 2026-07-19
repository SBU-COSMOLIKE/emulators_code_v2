"""Regression tests for the Architect-owned local backlog SHA-256 guard."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from ai.tools import backlog_guard


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "ai" / "tools" / "backlog_guard.py"


@contextlib.contextmanager
def scratch_checkout():
    """Yield a minimal checkout-shaped folder with one local backlog."""
    with tempfile.TemporaryDirectory(prefix="backlog-guard-") as temporary:
        repo = Path(temporary).resolve() / "repo"
        notes = repo / "ai" / "notes"
        notes.mkdir(parents=True)
        backlog = notes / "backlog.md"
        backlog.write_bytes(b"# Execution backlog\n\n# Open tickets\n")
        yield repo, backlog, notes / backlog_guard.STATE_FILENAME


def run_guard(repo, *arguments, role=None):
    """Run the public CLI with a controlled mailbox role."""
    environment = os.environ.copy()
    if role is None:
        environment.pop(backlog_guard.MAILBOX_ROLE_ENVIRONMENT, None)
    else:
        environment[backlog_guard.MAILBOX_ROLE_ENVIRONMENT] = role
    return subprocess.run(
        [sys.executable, str(SOURCE), "--repo", str(repo)] + list(arguments),
        cwd=str(repo),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def initialize(repo):
    """Create state through the explicit manual Architect path."""
    result = run_guard(repo, "initialize", "--architect-ack")
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    marker = "accepted SHA-256: "
    if marker not in result.stdout:
        raise AssertionError("initialize did not print a SHA-256")
    digest = result.stdout.split(marker, 1)[1].splitlines()[0]
    if backlog_guard.SHA256_RE.fullmatch(digest) is None:
        raise AssertionError("initialize printed an invalid SHA-256")
    return digest


class BacklogGuardTests(unittest.TestCase):
    """Exercise initialization, checking, sealing, and fail-closed reads."""

    def test_initialize_writes_canonical_exact_byte_digest_and_check_passes(self):
        with scratch_checkout() as (repo, backlog, state):
            digest = initialize(repo)
            self.assertEqual(digest, hashlib.sha256(backlog.read_bytes()).hexdigest())
            expected = {
                "backlog": "ai/notes/backlog.md",
                "sha256": digest,
                "version": 1,
            }
            self.assertEqual(
                state.read_bytes(),
                (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"),
            )

            checked = run_guard(repo, "check", role="red-team")
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn(
                "BACKLOG-GUARD-CHECK PASS sha256=" + digest, checked.stdout)

    def test_manual_write_commands_need_ack_and_nonarchitect_role_cannot_bypass(self):
        with scratch_checkout() as (repo, _, _):
            no_ack = run_guard(repo, "initialize")
            self.assertEqual(no_ack.returncode, 2)
            self.assertIn("requires --architect-ack", no_ack.stderr)

            implementer = run_guard(
                repo, "initialize", "--architect-ack", role="implementer")
            self.assertEqual(implementer.returncode, 2)
            self.assertIn("only the Architect", implementer.stderr)

            architect = run_guard(repo, "initialize", role="architect")
            self.assertEqual(architect.returncode, 0, architect.stderr)

    def test_edit_refuses_until_architect_seals_the_checked_previous_digest(self):
        with scratch_checkout() as (repo, backlog, state):
            previous = initialize(repo)
            backlog.write_bytes(backlog.read_bytes() + b"\n## New ticket\n")

            mismatch = run_guard(repo, "check")
            self.assertEqual(mismatch.returncode, 2)
            self.assertIn("SHA-256 mismatch", mismatch.stderr)

            wrong = run_guard(
                repo,
                "seal",
                "--previous-sha256",
                "0" * 64,
                "--architect-ack",
            )
            self.assertEqual(wrong.returncode, 2)
            self.assertIn("does not match the saved state", wrong.stderr)
            self.assertEqual(json.loads(state.read_text())["sha256"], previous)

            sealed = run_guard(
                repo,
                "seal",
                "--previous-sha256",
                previous,
                "--architect-ack",
            )
            current = hashlib.sha256(backlog.read_bytes()).hexdigest()
            self.assertEqual(sealed.returncode, 0, sealed.stderr)
            self.assertIn(
                "BACKLOG-GUARD-SEAL PASS sha256=" + current, sealed.stdout)
            self.assertEqual(json.loads(state.read_text())["sha256"], current)
            self.assertEqual(run_guard(repo, "check").returncode, 0)

    def test_retry_accepts_a_complete_initialize_or_seal(self):
        with scratch_checkout() as (repo, backlog, _):
            real_write = backlog_guard._atomic_write_state

            def publish_then_stop(path, document):
                real_write(path, document)
                raise backlog_guard.GuardError("simulated stop after publish")

            with mock.patch.object(
                    backlog_guard, "_atomic_write_state",
                    side_effect=publish_then_stop):
                with self.assertRaisesRegex(
                        backlog_guard.GuardError, "simulated stop"):
                    backlog_guard.initialize(repo, acknowledged=True)

            expected = hashlib.sha256(backlog.read_bytes()).hexdigest()
            self.assertEqual(
                backlog_guard.initialize(repo, acknowledged=True), expected)

            previous = expected
            backlog.write_bytes(backlog.read_bytes() + b"\n## New ticket\n")
            with mock.patch.object(
                    backlog_guard, "_atomic_write_state",
                    side_effect=publish_then_stop):
                with self.assertRaisesRegex(
                        backlog_guard.GuardError, "simulated stop"):
                    backlog_guard.seal(
                        repo, previous, acknowledged=True)

            current = hashlib.sha256(backlog.read_bytes()).hexdigest()
            self.assertEqual(
                backlog_guard.seal(repo, previous, acknowledged=True), current)

    def test_seal_retry_refuses_if_backlog_changed_after_publish(self):
        with scratch_checkout() as (repo, backlog, _):
            previous = initialize(repo)
            backlog.write_bytes(backlog.read_bytes() + b"first edit\n")
            real_write = backlog_guard._atomic_write_state

            def publish_then_stop(path, document):
                real_write(path, document)
                raise backlog_guard.GuardError("simulated stop after publish")

            with mock.patch.object(
                    backlog_guard, "_atomic_write_state",
                    side_effect=publish_then_stop):
                with self.assertRaises(backlog_guard.GuardError):
                    backlog_guard.seal(repo, previous, acknowledged=True)

            backlog.write_bytes(backlog.read_bytes() + b"second edit\n")
            with self.assertRaisesRegex(
                    backlog_guard.GuardError, "does not match the saved state"):
                backlog_guard.seal(repo, previous, acknowledged=True)

    def test_seal_requires_a_canonical_previous_digest_and_saved_state(self):
        with scratch_checkout() as (repo, _, _):
            missing = run_guard(
                repo,
                "seal",
                "--previous-sha256",
                "a" * 64,
                "--architect-ack",
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("cannot inspect backlog guard state", missing.stderr)

            initialize(repo)
            bad_spelling = run_guard(
                repo,
                "seal",
                "--previous-sha256",
                "A" * 64,
                "--architect-ack",
            )
            self.assertEqual(bad_spelling.returncode, 2)
            self.assertIn("64 lowercase hex", bad_spelling.stderr)

    def test_missing_backlog_or_state_refuses_check(self):
        with scratch_checkout() as (repo, backlog, state):
            initialize(repo)
            state.unlink()
            missing_state = run_guard(repo, "check")
            self.assertEqual(missing_state.returncode, 2)
            self.assertIn("cannot inspect backlog guard state", missing_state.stderr)

            initialize(repo)
            backlog.unlink()
            missing_backlog = run_guard(repo, "check")
            self.assertEqual(missing_backlog.returncode, 2)
            self.assertIn("cannot inspect ai/notes/backlog.md", missing_backlog.stderr)

    def test_backlog_and_state_links_or_irregular_files_refuse(self):
        with scratch_checkout() as (repo, backlog, state):
            initialize(repo)
            target = repo / "outside-backlog.md"
            target.write_bytes(backlog.read_bytes())
            backlog.unlink()
            backlog.symlink_to(target)
            linked_backlog = run_guard(repo, "check")
            self.assertEqual(linked_backlog.returncode, 2)
            self.assertIn("backlog.md is not a regular file", linked_backlog.stderr)

        with scratch_checkout() as (repo, _, state):
            initialize(repo)
            state_target = repo / "outside-state.json"
            state_target.write_bytes(state.read_bytes())
            state.unlink()
            state.symlink_to(state_target)
            linked_state = run_guard(repo, "check")
            self.assertEqual(linked_state.returncode, 2)
            self.assertIn("state is not a regular file", linked_state.stderr)

        with scratch_checkout() as (repo, backlog, _):
            initialize(repo)
            backlog.unlink()
            backlog.mkdir()
            directory_backlog = run_guard(repo, "check")
            self.assertEqual(directory_backlog.returncode, 2)
            self.assertIn("backlog.md is not a regular file", directory_backlog.stderr)

        with scratch_checkout() as (repo, _, state):
            initialize(repo)
            state.unlink()
            state.mkdir()
            directory_state = run_guard(repo, "check")
            self.assertEqual(directory_state.returncode, 2)
            self.assertIn("state is not a regular file", directory_state.stderr)

    def test_redirected_notes_directory_and_hardlinked_files_refuse(self):
        with scratch_checkout() as (repo, _, _):
            original = repo / "ai" / "notes"
            moved = repo / "notes-target"
            original.rename(moved)
            original.symlink_to(moved, target_is_directory=True)
            redirected = run_guard(repo, "check")
            self.assertEqual(redirected.returncode, 2)
            self.assertIn("ai/notes is not a real directory", redirected.stderr)

        with scratch_checkout() as (repo, backlog, _):
            initialize(repo)
            os.link(backlog, repo / "second-backlog-name.md")
            hardlinked = run_guard(repo, "check")
            self.assertEqual(hardlinked.returncode, 2)
            self.assertIn("more than one filesystem name", hardlinked.stderr)

    def test_oversized_backlog_and_state_refuse(self):
        with scratch_checkout() as (repo, _, state):
            initialize(repo)
            with mock.patch.object(backlog_guard, "MAX_BACKLOG_BYTES", 4):
                with self.assertRaisesRegex(backlog_guard.GuardError, "size limit"):
                    backlog_guard.check(repo)

            state.write_bytes(b"x" * 32)
            with mock.patch.object(backlog_guard, "MAX_STATE_BYTES", 8):
                with self.assertRaisesRegex(backlog_guard.GuardError, "size limit"):
                    backlog_guard.check(repo)

    def test_malformed_or_noncanonical_state_refuses(self):
        with scratch_checkout() as (repo, _, state):
            digest = initialize(repo)
            state.write_text(
                '{"version":1,"backlog":"ai/notes/backlog.md","sha256":"'
                + digest + '"}\n',
                encoding="utf-8",
            )
            noncanonical = run_guard(repo, "check")
            self.assertEqual(noncanonical.returncode, 2)
            self.assertIn("not in canonical form", noncanonical.stderr)

            state.write_text('{"version": 1, "version": 1}\n', encoding="utf-8")
            duplicate = run_guard(repo, "check")
            self.assertEqual(duplicate.returncode, 2)
            self.assertIn("repeats the field version", duplicate.stderr)

            state.write_text(
                json.dumps({
                    "backlog": "ai/notes/backlog.md",
                    "sha256": digest,
                    "version": True,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            boolean_version = run_guard(repo, "check")
            self.assertEqual(boolean_version.returncode, 2)
            self.assertIn("unsupported version", boolean_version.stderr)

    def test_a_file_changed_during_its_read_refuses(self):
        with scratch_checkout() as (repo, backlog, state):
            initialize(repo)
            real_read = os.read
            changed = False

            def change_backlog_after_read(descriptor, size):
                nonlocal changed
                data = real_read(descriptor, size)
                if data and not changed:
                    changed = True
                    backlog.write_bytes(backlog.read_bytes() + b"changed\n")
                return data

            with mock.patch.object(
                    backlog_guard.os, "read", side_effect=change_backlog_after_read):
                with self.assertRaisesRegex(backlog_guard.GuardError, "changed"):
                    backlog_guard._read_regular_bytes(
                        backlog, "ai/notes/backlog.md", 1024 * 1024)

            state_bytes = state.read_bytes()
            changed = False

            def change_state_after_read(descriptor, size):
                nonlocal changed
                data = real_read(descriptor, size)
                if data and not changed:
                    changed = True
                    state.write_bytes(state_bytes + b" ")
                return data

            with mock.patch.object(
                    backlog_guard.os, "read", side_effect=change_state_after_read):
                with self.assertRaisesRegex(backlog_guard.GuardError, "changed"):
                    backlog_guard._read_regular_bytes(
                        state, "backlog guard state", 1024 * 1024)

    def test_stale_lock_file_is_reused_and_local_files_are_ignored(self):
        with scratch_checkout() as (repo, _, state):
            lock = state.with_name(backlog_guard.LOCK_FILENAME)
            lock.write_text("123\n", encoding="ascii")
            initialized = run_guard(repo, "initialize", "--architect-ack")
            self.assertEqual(initialized.returncode, 0, initialized.stderr)
            self.assertTrue(lock.is_file())
            self.assertEqual(run_guard(repo, "check").returncode, 0)

        ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/ai/notes/.backlog-guard.json\n", ignore)
        self.assertIn("/ai/notes/.backlog-guard.lock\n", ignore)
        self.assertIn("/ai/notes/.backlog-guard.json.tmp-*\n", ignore)

    def test_killed_lock_owner_releases_the_same_file(self):
        with scratch_checkout() as (repo, _, state):
            lock = state.with_name(backlog_guard.LOCK_FILENAME)
            helper = (
                "import fcntl, pathlib, sys\n"
                "path = pathlib.Path(sys.argv[1])\n"
                "with path.open('a+') as stream:\n"
                "    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\n"
                "    stream.seek(0); stream.truncate(); "
                "stream.write('999999\\n'); stream.flush()\n"
                "    print('ready', flush=True)\n"
                "    sys.stdin.read(1)\n")
            owner = subprocess.Popen(
                [sys.executable, "-c", helper, str(lock)],
                cwd=str(REPO_ROOT), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(owner.stdout.readline().strip(), "ready")
                refused = run_guard(repo, "initialize", "--architect-ack")
                self.assertEqual(refused.returncode, 2)
                self.assertIn("another backlog guard write is active",
                              refused.stderr)
                self.assertFalse(state.exists())
            finally:
                if owner.poll() is None:
                    owner.kill()
                owner.wait(timeout=5)
                owner.stdin.close()
                owner.stdout.close()
                owner.stderr.close()
            self.assertTrue(lock.is_file())
            recovered = run_guard(repo, "initialize", "--architect-ack")
            self.assertEqual(recovered.returncode, 0, recovered.stderr)
            self.assertTrue(lock.is_file())
            self.assertEqual(run_guard(repo, "check").returncode, 0)


if __name__ == "__main__":
    unittest.main()
