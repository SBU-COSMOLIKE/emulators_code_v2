"""Focused tests for the committed ticket character guard."""

from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from ai.tools import ticket_change_guard


@contextmanager
def mailbox_limit(value=None):
    """Temporarily set or remove the daemon's inherited ticket limit."""
    name = ticket_change_guard.MAX_CHARACTERS_ENVIRONMENT
    existed = name in os.environ
    previous = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if existed:
            os.environ[name] = previous
        else:
            os.environ.pop(name, None)


def git(repository, *arguments):
    """Run one required Git fixture command and return its text output."""
    result = subprocess.run(
        ["git", "-C", str(repository)] + list(arguments),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False)
    if result.returncode != 0:
        raise AssertionError(
            "fixture Git command failed: " + " ".join(arguments) + "\n"
            + result.stderr)
    return result.stdout.strip()


def write_bytes(repository, name, payload):
    """Write one fixture file, creating its parent directories."""
    path = Path(repository) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def commit_all(repository, message):
    """Commit the complete fixture tree and return the full commit name."""
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", message)
    return git(repository, "rev-parse", "HEAD")


@contextmanager
def repository(files=None):
    """Yield a temporary Git repository with one initial commit."""
    with tempfile.TemporaryDirectory(prefix="ticket-change-guard-") as tmp:
        root = Path(tmp)
        git(root, "init", "-q")
        git(root, "config", "user.email", "ticket@example.invalid")
        git(root, "config", "user.name", "Ticket Test")
        initial = {"seed.txt": b"seed\n"}
        if files is not None:
            initial.update(files)
        for name, payload in initial.items():
            write_bytes(repository=root, name=name, payload=payload)
        base = commit_all(repository=root, message="base")
        yield root, base


def run_guard(repository, base, maximum=None, environment_limit=None,
              candidate=None):
    """Run the guard in-process and capture its user-facing output."""
    arguments = ["--repo", str(repository), "--base", base]
    if candidate is not None:
        arguments.extend(("--architect-audit", "--candidate", candidate))
    if maximum is not None:
        arguments.extend(("--max", str(maximum)))
    output = io.StringIO()
    with mailbox_limit(environment_limit):
        with redirect_stdout(output):
            return_code = ticket_change_guard.main(argv=arguments)
    return return_code, output.getvalue()


class TicketChangeGuardTests(unittest.TestCase):
    """Pin size, repository-state, and text-decoding boundaries."""

    def test_default_zero_skips_dirty_binary_candidate(self):
        """Unlimited mode neither measures text nor requires a clean tree."""
        with repository() as (root, base):
            write_bytes(repository=root, name="image.bin",
                        payload=b"\x00\xff\x01")
            commit_all(repository=root, message="binary candidate")
            write_bytes(repository=root, name="seed.txt", payload=b"dirty\n")
            write_bytes(repository=root, name="untracked.txt",
                        payload=b"waiting\n")

            return_code, output = run_guard(
                repository=root, base=base)

        self.assertEqual(return_code, 0)
        self.assertIn("size limit disabled", output)
        self.assertIn("maximum changed characters: unlimited (0)", output)
        self.assertIn("measurement: skipped", output)
        self.assertNotIn("binary file cannot", output)

    def test_maximum_uses_strict_ascii_decimal_grammar(self):
        """Signs, whitespace, separators, and non-ASCII digits are refused."""
        invalid_values = ("-1", "+1", " 1", "1 ", "1_0", "١")
        with repository() as (root, base):
            for value in invalid_values:
                with self.subTest(value=value):
                    error = io.StringIO()
                    with redirect_stderr(error):
                        with self.assertRaises(SystemExit) as caught:
                            ticket_change_guard.main(argv=[
                                "--repo", str(root), "--base", base,
                                "--max", value])
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn("nonnegative ASCII decimal", error.getvalue())

    def test_omitted_maximum_uses_the_mailbox_environment(self):
        """A child cannot turn a dispatched positive limit into unlimited mode."""
        with repository() as (root, base):
            write_bytes(repository=root, name="candidate.txt", payload=b"four")
            commit_all(repository=root, message="candidate")

            return_code, output = run_guard(
                repository=root, base=base, environment_limit="3")

        self.assertEqual(return_code, 1)
        self.assertIn("maximum changed characters: 3", output)
        self.assertIn("changed characters: 4", output)

    def test_explicit_maximum_must_match_the_mailbox_environment(self):
        """An explicit zero or another value cannot bypass a dispatched limit."""
        with repository() as (root, base):
            for explicit in (0, 6):
                with self.subTest(explicit=explicit):
                    with mock.patch.object(
                            ticket_change_guard, "run_git") as run_git_mock:
                        return_code, output = run_guard(
                            repository=root, base=base, maximum=explicit,
                            environment_limit="5")
                    self.assertEqual(return_code, 2)
                    self.assertIn("--max does not match", output)
                    run_git_mock.assert_not_called()

            matching_code, matching_output = run_guard(
                repository=root, base=base, maximum=5,
                environment_limit="5")

        self.assertEqual(matching_code, 0)
        self.assertIn("maximum changed characters: 5", matching_output)

    def test_mailbox_environment_uses_strict_ascii_decimal_grammar(self):
        """Malformed inherited limits fail before the repository is inspected."""
        invalid_values = ("", "-1", "+1", " 1", "1 ", "1_0", "١")
        with repository() as (root, base):
            for value in invalid_values:
                with self.subTest(value=value):
                    with mock.patch.object(
                            ticket_change_guard, "run_git") as run_git_mock:
                        return_code, output = run_guard(
                            repository=root, base=base,
                            environment_limit=value)
                    self.assertEqual(return_code, 2)
                    self.assertIn("MAILBOX_MAX_CHARACTERS must be", output)
                    run_git_mock.assert_not_called()

    def test_dispatch_requires_its_authoritative_guard_copy(self):
        """A role cannot accidentally run the same tool from another worktree."""
        variable = ticket_change_guard.AUTHORITATIVE_GUARD_ENVIRONMENT
        with repository() as (root, base):
            wrong_path = str(root / "other" / "ticket_change_guard.py")
            with mock.patch.dict(os.environ, {variable: wrong_path}):
                with mock.patch.object(
                        ticket_change_guard, "run_git") as run_git_mock:
                    return_code, output = run_guard(
                        repository=root, base=base, maximum=10)

            self.assertEqual(return_code, 2)
            self.assertIn("not the authoritative dispatched copy", output)
            run_git_mock.assert_not_called()

            current_path = str(Path(ticket_change_guard.__file__).resolve())
            with mock.patch.dict(os.environ, {variable: current_path}):
                matching_code, matching_output = run_guard(
                    repository=root, base=base, maximum=10)

        self.assertEqual(matching_code, 0)
        self.assertIn("within limit", matching_output)

    def test_architect_audit_keeps_measuring_the_named_commit(self):
        """Later HEAD work cannot change the earlier audit measurement."""
        with repository() as (root, base):
            write_bytes(repository=root, name="ticket-a.txt", payload=b"four")
            ticket_a = commit_all(repository=root, message="ticket A")
            write_bytes(
                repository=root, name="ticket-b.txt", payload=b"later work")
            ticket_b = commit_all(repository=root, message="ticket B")

            audit_code, audit_output = run_guard(
                repository=root, base=base, maximum=4,
                candidate=ticket_a)
            default_code, default_output = run_guard(
                repository=root, base=base, maximum=4)

        self.assertEqual(audit_code, 0)
        self.assertIn("candidate commit: " + ticket_a, audit_output)
        self.assertIn(
            "changed characters: 4 (4 added + 0 deleted)", audit_output)
        self.assertEqual(default_code, 1)
        self.assertIn("candidate commit: " + ticket_b, default_output)
        self.assertIn(
            "changed characters: 14 (14 added + 0 deleted)", default_output)

    def test_architect_audit_reads_commit_objects_not_worktree_edits(self):
        """Uncommitted later work is irrelevant only to an immutable audit."""
        with repository() as (root, base):
            write_bytes(repository=root, name="ticket.txt", payload=b"ok\n")
            candidate = commit_all(repository=root, message="candidate")
            write_bytes(
                repository=root, name="waiting.txt", payload=b"next ticket\n")

            audit_code, audit_output = run_guard(
                repository=root, base=base, maximum=3,
                candidate=candidate)
            default_code, default_output = run_guard(
                repository=root, base=base, maximum=3)

        self.assertEqual(audit_code, 0)
        self.assertIn("candidate commit: " + candidate, audit_output)
        self.assertEqual(default_code, 2)
        self.assertIn("HEAD is not the exact candidate", default_output)

    def test_architect_audit_flags_and_full_candidate_are_required_together(
            self):
        """Partial, abbreviated, and symbolic audit requests fail at parsing."""
        with repository() as (root, base):
            candidate = git(root, "rev-parse", "HEAD")
            cases = (
                (["--architect-audit"], "requires --candidate"),
                (["--candidate", candidate],
                 "requires --architect-audit"),
                (["--architect-audit", "--candidate", candidate[:12]],
                 "one full 40-hex commit"),
                (["--architect-audit", "--candidate", "HEAD"],
                 "one full 40-hex commit"),
            )
            for extra, diagnostic in cases:
                with self.subTest(extra=extra):
                    error = io.StringIO()
                    with redirect_stderr(error):
                        with self.assertRaises(SystemExit) as caught:
                            ticket_change_guard.main(argv=[
                                "--repo", str(root), "--base", base,
                                "--max", "10"] + extra)
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn(diagnostic, error.getvalue())

    def test_architect_audit_candidate_must_exist_in_this_repository(self):
        """A full commit from another repository is not a local candidate."""
        with repository() as (root, base):
            with repository(files={"other.txt": b"different\n"}) as (
                    other_root, _other_base):
                write_bytes(
                    repository=other_root, name="foreign.txt",
                    payload=b"foreign commit\n")
                foreign = commit_all(
                    repository=other_root, message="foreign candidate")

            return_code, output = run_guard(
                repository=root, base=base, maximum=100,
                candidate=foreign)

        self.assertEqual(return_code, 2)
        self.assertIn("--candidate is not a commit", output)

    def test_architect_audit_candidate_must_descend_from_base(self):
        """An unrelated local commit cannot be presented as this ticket."""
        with repository() as (root, base):
            tree = git(root, "rev-parse", "HEAD^{tree}")
            unrelated = git(root, "commit-tree", tree, "-m", "unrelated")

            return_code, output = run_guard(
                repository=root, base=base, maximum=100,
                candidate=unrelated)

        self.assertEqual(return_code, 2)
        self.assertIn(
            "--base is not an ancestor of --candidate", output)

    def test_architect_audit_rechecks_candidate_identity_after_measurement(
            self):
        """An inconsistent Git answer cannot silently change the audit target."""
        with repository() as (root, base):
            write_bytes(repository=root, name="ticket.txt", payload=b"one\n")
            candidate = commit_all(repository=root, message="candidate")
            write_bytes(repository=root, name="later.txt", payload=b"two\n")
            later = commit_all(repository=root, message="later")
            real_resolve = ticket_change_guard.resolve_commit
            candidate_resolutions = 0

            def inconsistent_resolve(repository, revision, label):
                nonlocal candidate_resolutions
                resolved = real_resolve(repository, revision, label)
                if label == "--candidate":
                    candidate_resolutions += 1
                    if candidate_resolutions == 3:
                        return later
                return resolved

            with mock.patch.object(
                    ticket_change_guard, "resolve_commit",
                    side_effect=inconsistent_resolve):
                return_code, output = run_guard(
                    repository=root, base=base, maximum=100,
                    candidate=candidate)

        self.assertEqual(return_code, 2)
        self.assertIn(
            "--candidate changed while the ticket was being checked", output)

    def test_add_delete_replace_unicode_and_exact_boundary(self):
        """Spaces, newlines, and Unicode each count as one code point."""
        with repository(files={
                "delete.txt": "old \n".encode("utf-8"),
                "replace.txt": "café α\n".encode("utf-8")}) as (root, base):
            (root / "delete.txt").unlink()
            write_bytes(repository=root, name="replace.txt",
                        payload="café β\n".encode("utf-8"))
            write_bytes(repository=root, name="new.txt", payload=b" n\n")
            commit_all(repository=root, message="text candidate")

            boundary_code, boundary_output = run_guard(
                repository=root, base=base, maximum=10)
            over_code, over_output = run_guard(
                repository=root, base=base, maximum=9)

        self.assertEqual(boundary_code, 0)
        self.assertIn("within limit", boundary_output)
        self.assertIn("changed characters: 10 (4 added + 6 deleted)",
                      boundary_output)
        self.assertEqual(over_code, 1)
        self.assertIn("over limit", over_output)
        self.assertIn("changed characters: 10 (4 added + 6 deleted)",
                      over_output)

    def test_pure_rename_changes_zero_characters(self):
        """Git rename detection prevents a pure move from counting as a rewrite."""
        with repository(files={
                "old-name.txt": b"a readable file\nwith two lines\n"}) as (
                    root, base):
            git(root, "mv", "old-name.txt", "new-name.txt")
            commit_all(repository=root, message="rename candidate")

            return_code, output = run_guard(
                repository=root, base=base, maximum=1)

        self.assertEqual(return_code, 0)
        self.assertIn("changed characters: 0 (0 added + 0 deleted)", output)

    def test_unchanged_uncountable_renames_do_not_read_blobs(self):
        """A pure move is zero even when its unchanged blob is not text."""
        cases = (
            ("binary.dat", b"binary\x00payload"),
            ("invalid.txt", b"invalid\xffpayload"),
        )
        for old_name, payload in cases:
            with self.subTest(old_name=old_name):
                with repository(files={old_name: payload}) as (root, base):
                    new_name = "moved-" + old_name
                    git(root, "mv", old_name, new_name)
                    commit_all(repository=root, message="move uncountable blob")
                    with mock.patch.object(
                            ticket_change_guard, "blob_text",
                            wraps=ticket_change_guard.blob_text) as reader:
                        return_code, output = run_guard(
                            repository=root, base=base, maximum=1)
                self.assertEqual(return_code, 0)
                self.assertIn(
                    "changed characters: 0 (0 added + 0 deleted)", output)
                reader.assert_not_called()

    def test_content_changed_uncountable_renames_fail_closed(self):
        """The rename exception applies only while the blob is unchanged."""
        cases = (
            ("binary.dat", b"binary\x00old", b"binary\x00new", "binary"),
            ("invalid.txt", b"invalid\xffold", b"invalid\xffnew", "UTF-8"),
        )
        for old_name, old_payload, new_payload, diagnostic in cases:
            with self.subTest(old_name=old_name):
                with repository(files={old_name: old_payload}) as (root, base):
                    new_name = "moved-" + old_name
                    git(root, "mv", old_name, new_name)
                    write_bytes(
                        repository=root, name=new_name, payload=new_payload)
                    commit_all(repository=root, message="change renamed blob")
                    return_code, output = run_guard(
                        repository=root, base=base, maximum=100)
                self.assertEqual(return_code, 2)
                self.assertIn("cannot measure", output)
                self.assertIn(diagnostic, output)

    def test_positive_limit_requires_exact_clean_head(self):
        """Staged, unstaged, and nonignored untracked work all refuse."""
        with repository() as (root, base):
            write_bytes(repository=root, name="candidate.txt", payload=b"ok\n")
            commit_all(repository=root, message="clean candidate")
            write_bytes(repository=root, name="staged.txt", payload=b"staged\n")
            git(root, "add", "staged.txt")
            write_bytes(repository=root, name="seed.txt", payload=b"unstaged\n")
            write_bytes(repository=root, name="untracked.txt",
                        payload=b"untracked\n")

            return_code, output = run_guard(
                repository=root, base=base, maximum=100)

        self.assertEqual(return_code, 2)
        self.assertIn("cannot measure", output)
        self.assertIn("HEAD is not the exact candidate", output)
        self.assertIn("nonignored untracked", output)

    def test_positive_limit_refuses_index_flags_that_hide_edits(self):
        """Assume-unchanged and skip-worktree cannot conceal candidate work."""
        cases = (
            ("--assume-unchanged", "assume-unchanged"),
            ("--skip-worktree", "skip-worktree"),
        )
        for option, diagnostic in cases:
            with self.subTest(option=option):
                with repository() as (root, base):
                    git(root, "update-index", option, "seed.txt")
                    return_code, output = run_guard(
                        repository=root, base=base, maximum=10)
                self.assertEqual(return_code, 2)
                self.assertIn("cannot measure", output)
                self.assertIn(diagnostic, output)

    def test_guard_does_not_refresh_or_rewrite_the_git_index(self):
        """Even the cleanliness check leaves the index bytes and mtime alone."""
        with repository() as (root, base):
            seed = root / "seed.txt"
            seed_stat = seed.stat()
            future = seed_stat.st_mtime_ns + 2_000_000_000
            os.utime(seed, ns=(seed_stat.st_atime_ns, future))
            index_name = git(root, "rev-parse", "--git-path", "index")
            index = Path(index_name)
            if not index.is_absolute():
                index = root / index
            bytes_before = index.read_bytes()
            mtime_before = index.stat().st_mtime_ns

            return_code, output = run_guard(
                repository=root, base=base, maximum=10)

            bytes_after = index.read_bytes()
            mtime_after = index.stat().st_mtime_ns

        self.assertEqual(return_code, 0)
        self.assertIn("within limit", output)
        self.assertEqual(bytes_after, bytes_before)
        self.assertEqual(mtime_after, mtime_before)

    def test_git_timeout_fails_closed(self):
        """A stalled repository command cannot leave the result ambiguous."""
        with repository() as (root, base):
            timeout = subprocess.TimeoutExpired(
                cmd=["git", "status"],
                timeout=ticket_change_guard.GIT_COMMAND_TIMEOUT_SECONDS)
            with mock.patch.object(
                    ticket_change_guard.subprocess, "run",
                    side_effect=timeout):
                return_code, output = run_guard(
                    repository=root, base=base, maximum=10)

        self.assertEqual(return_code, 2)
        self.assertIn("cannot measure", output)
        self.assertIn("Git did not finish within 30 seconds", output)

    def test_positive_limit_refuses_binary_and_non_utf8_files(self):
        """A positive text limit fails closed for uncountable changed blobs."""
        cases = (
            ("binary.dat", b"\x00\x01", "binary file"),
            ("invalid.txt", b"\xff\xfe", "not valid UTF-8"),
        )
        for name, payload, diagnostic in cases:
            with self.subTest(name=name):
                with repository() as (root, base):
                    write_bytes(repository=root, name=name, payload=payload)
                    commit_all(repository=root, message="uncountable candidate")
                    return_code, output = run_guard(
                        repository=root, base=base, maximum=100)
                self.assertEqual(return_code, 2)
                self.assertIn("cannot measure", output)
                self.assertIn(diagnostic, output)

    def test_base_must_be_a_full_ancestor_commit(self):
        """An existing but unrelated commit cannot define this ticket."""
        with repository() as (root, _base):
            write_bytes(repository=root, name="candidate.txt", payload=b"ok\n")
            commit_all(repository=root, message="candidate")
            tree = git(root, "rev-parse", "HEAD^{tree}")
            unrelated = git(root, "commit-tree", tree, "-m", "unrelated")

            return_code, output = run_guard(
                repository=root, base=unrelated, maximum=100)

        self.assertEqual(return_code, 2)
        self.assertIn("--base is not an ancestor of HEAD", output)

    def test_measurement_spans_every_commit_after_base(self):
        """The guard compares endpoint trees, not only HEAD's last commit."""
        with repository() as (root, base):
            write_bytes(repository=root, name="series.txt", payload=b"ab")
            commit_all(repository=root, message="first ticket commit")
            write_bytes(repository=root, name="series.txt", payload=b"abcd")
            commit_all(repository=root, message="second ticket commit")

            return_code, output = run_guard(
                repository=root, base=base, maximum=3)

        self.assertEqual(return_code, 1)
        self.assertIn("changed characters: 4 (4 added + 0 deleted)", output)

    def test_character_comparison_is_exact_and_symmetric(self):
        """Repeated characters use the minimum insertion/deletion distance."""
        cases = (
            ("tide", "diet", 2, 2),
            ("aaaaab", "baaaaa", 1, 1),
            ("abcdef", "azced", 2, 3),
        )
        for old_text, new_text, added, deleted in cases:
            with self.subTest(old_text=old_text, new_text=new_text):
                forward = ticket_change_guard.character_delta(
                    old_text=old_text, new_text=new_text)
                reverse = ticket_change_guard.character_delta(
                    old_text=new_text, new_text=old_text)
                self.assertEqual((forward.added, forward.deleted),
                                 (added, deleted))
                self.assertEqual((reverse.added, reverse.deleted),
                                 (deleted, added))
                self.assertEqual(forward.total, reverse.total)

    def test_character_comparison_has_size_and_product_bounds(self):
        """An ambiguous replacement must fit both comparison limits."""
        with mock.patch.object(
                ticket_change_guard,
                "MAX_COMPARISON_MIDDLE_CODEPOINTS", 3):
            with self.assertRaisesRegex(
                    ticket_change_guard.GuardError, "comparison size limit"):
                ticket_change_guard.character_delta("ab", "cd")

        with mock.patch.object(
                ticket_change_guard, "MAX_LCS_CELLS_PER_FILE", 3):
            with self.assertRaisesRegex(
                    ticket_change_guard.GuardError, "per-file exact-match"):
                ticket_change_guard.character_delta("ab", "cd")

        insertion = ticket_change_guard.character_delta(
            old_text="", new_text="large insertion")
        self.assertEqual(insertion.total, 15)

    def test_blob_reads_have_per_file_and_aggregate_bounds(self):
        """The guard rejects oversized content before loading a changed blob."""
        with repository() as (root, base):
            write_bytes(repository=root, name="large.txt", payload=b"four")
            commit_all(repository=root, message="large blob")
            with mock.patch.object(ticket_change_guard, "MAX_BLOB_BYTES", 3):
                with mock.patch.object(
                        ticket_change_guard, "blob_text",
                        wraps=ticket_change_guard.blob_text) as reader:
                    with mock.patch.object(
                            ticket_change_guard, "binary_entry_keys",
                            wraps=ticket_change_guard.binary_entry_keys
                    ) as binary:
                        file_code, file_output = run_guard(
                            repository=root, base=base, maximum=100)
            reader.assert_not_called()
            binary.assert_not_called()

        self.assertEqual(file_code, 2)
        self.assertIn("per-blob read limit of 3 bytes", file_output)

        with repository() as (root, base):
            write_bytes(repository=root, name="one.txt", payload=b"abc")
            write_bytes(repository=root, name="two.txt", payload=b"def")
            commit_all(repository=root, message="aggregate blobs")
            with mock.patch.object(ticket_change_guard, "MAX_BLOB_BYTES", 10):
                with mock.patch.object(
                        ticket_change_guard,
                        "MAX_AGGREGATE_BLOB_BYTES", 5):
                    with mock.patch.object(
                            ticket_change_guard, "blob_text",
                            wraps=ticket_change_guard.blob_text) as reader:
                        with mock.patch.object(
                                ticket_change_guard, "binary_entry_keys",
                                wraps=ticket_change_guard.binary_entry_keys
                        ) as binary:
                            total_code, total_output = run_guard(
                                repository=root, base=base, maximum=100)
            reader.assert_not_called()
            binary.assert_not_called()

        self.assertEqual(total_code, 2)
        self.assertIn("aggregate blob-read limit of 5 bytes", total_output)

    def test_character_work_has_an_aggregate_bound(self):
        """Many individually small replacements cannot multiply total work."""
        with repository(files={
                "one.txt": b"ab", "two.txt": b"ef"}) as (root, base):
            write_bytes(repository=root, name="one.txt", payload=b"cd")
            write_bytes(repository=root, name="two.txt", payload=b"gh")
            commit_all(repository=root, message="two replacements")
            with mock.patch.object(
                    ticket_change_guard, "MAX_LCS_CELLS_PER_FILE", 10):
                with mock.patch.object(
                        ticket_change_guard, "MAX_TOTAL_LCS_CELLS", 7):
                    with mock.patch.object(
                            ticket_change_guard, "exact_lcs_length",
                            wraps=ticket_change_guard.exact_lcs_length) as lcs:
                        return_code, output = run_guard(
                            repository=root, base=base, maximum=100)

        self.assertEqual(return_code, 2)
        self.assertIn("aggregate exact-match work limit of 7", output)
        lcs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
