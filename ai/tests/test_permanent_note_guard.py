"""Regression tests for the Architect's permanent-note SHA-256 guard."""

import contextlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from ai.tools.backlog_bundle import PERMANENT_NOTES as BUNDLE_NOTES
from ai.tools.permanent_note_guard import PERMANENT_NOTES as GUARD_NOTES


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "ai" / "tools" / "permanent_note_guard.py"
PERMANENT_NOTES = (
    "ai/notes/MEMORY.md",
    "ai/notes/artifacts-inference-warmstart.md",
    "ai/notes/conventions-and-workflow.md",
    "ai/notes/data-generation-and-cuts.md",
    "ai/notes/families-background-mps.md",
    "ai/notes/families-scalar-cmb.md",
    "ai/notes/models-and-designs.md",
    "ai/notes/project-and-history.md",
    "ai/notes/readme-go-no-go.md",
    "ai/notes/training-stack.md",
    "ai/notes/user-didactics-and-python-voice.md",
)


def run_git(repo, *arguments):
    """Run one Git command in a disposable test repository."""
    result = subprocess.run(
        ["git", "-C", str(repo)] + list(arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def write(repo, path_text, text):
    """Write one UTF-8 fixture below a disposable repository."""
    path = repo.joinpath(*path_text.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@contextlib.contextmanager
def scratch_repository():
    """Yield a committed repository containing the production guard."""
    with tempfile.TemporaryDirectory(prefix="permanent-note-guard-") as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        run_git(repo, "init", "-q", "-b", "main")
        run_git(repo, "config", "user.name", "Guard Test")
        run_git(repo, "config", "user.email", "guard@example.invalid")

        guard = repo / "ai" / "tools" / "permanent_note_guard.py"
        guard.parent.mkdir(parents=True)
        shutil.copy2(SOURCE, guard)
        for index, path_text in enumerate(PERMANENT_NOTES):
            write(repo, path_text, "# Permanent note " + str(index) + "\n")
        run_git(repo, "add", "ai/tools/permanent_note_guard.py")
        run_git(repo, "add", *PERMANENT_NOTES)
        run_git(repo, "commit", "-q", "-m", "guard base")
        base = run_git(repo, "rev-parse", "HEAD")
        yield repo, guard, base


def run_guard(repo, guard, base):
    """Run the public guard command in one disposable repository."""
    return subprocess.run(
        [
            sys.executable,
            str(guard),
            "--repo",
            str(repo),
            "--base",
            base,
        ],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


class PermanentNoteGuardTests(unittest.TestCase):
    """Catch ordinary unstaged, staged, and committed note drift."""

    def test_clean_checkout_prints_every_sha_and_one_pass_marker(self):
        with scratch_repository() as (repo, guard, base):
            result = run_guard(repo, guard, base)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.count("PERMANENT-NOTE-GUARD PASS"), 1)
            for path in PERMANENT_NOTES:
                self.assertIn(path, result.stdout)
            self.assertIn("states: current HEAD, Git staging area, working tree",
                          result.stdout)

    def test_unstaged_note_edit_refuses(self):
        with scratch_repository() as (repo, guard, base):
            write(repo, PERMANENT_NOTES[-1], "# Accidental rewrite\n")
            result = run_guard(repo, guard, base)
            self.assertEqual(result.returncode, 2)
            self.assertIn("working tree", result.stderr)
            self.assertNotIn("PERMANENT-NOTE-GUARD PASS", result.stdout)

    def test_staged_note_edit_refuses_even_after_working_file_is_restored(self):
        with scratch_repository() as (repo, guard, base):
            path_text = PERMANENT_NOTES[0]
            original = (repo / path_text).read_text(encoding="utf-8")
            write(repo, path_text, "# Staged accidental rewrite\n")
            run_git(repo, "add", path_text)
            write(repo, path_text, original)
            result = run_guard(repo, guard, base)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Git staging area", result.stderr)
            self.assertNotIn("PERMANENT-NOTE-GUARD PASS", result.stdout)

    def test_committed_note_edit_refuses_even_after_working_file_is_restored(self):
        with scratch_repository() as (repo, guard, base):
            path_text = PERMANENT_NOTES[1]
            original = (repo / path_text).read_text(encoding="utf-8")
            write(repo, path_text, "# Committed accidental rewrite\n")
            run_git(repo, "add", path_text)
            run_git(repo, "commit", "-q", "-m", "accidental note edit")
            write(repo, path_text, original)
            run_git(repo, "add", path_text)
            result = run_guard(repo, guard, base)
            self.assertEqual(result.returncode, 2)
            self.assertIn("current HEAD", result.stderr)
            self.assertNotIn("PERMANENT-NOTE-GUARD PASS", result.stdout)

    def test_extra_tracked_note_refuses_but_untracked_ticket_is_allowed(self):
        with scratch_repository() as (repo, guard, base):
            write(repo, "ai/notes/local-ticket.md", "# Local ticket\n")
            clean = run_guard(repo, guard, base)
            self.assertEqual(clean.returncode, 0, clean.stderr)
            run_git(repo, "add", "ai/notes/local-ticket.md")
            result = run_guard(repo, guard, base)
            self.assertEqual(result.returncode, 2)
            self.assertIn("extra=ai/notes/local-ticket.md", result.stderr)

    def test_changed_guard_and_abbreviated_base_refuse(self):
        with scratch_repository() as (repo, guard, base):
            guard.write_text(
                guard.read_text(encoding="utf-8") + "\n# accidental edit\n",
                encoding="utf-8",
            )
            changed = run_guard(repo, guard, base)
            self.assertEqual(changed.returncode, 2)
            self.assertIn("permanent_note_guard.py differs", changed.stderr)
            abbreviated = run_guard(repo, guard, base[:12])
            self.assertEqual(abbreviated.returncode, 2)
            self.assertIn("full lowercase Git commit hash", abbreviated.stderr)

    def test_symlink_note_refuses(self):
        with scratch_repository() as (repo, guard, base):
            note = repo / PERMANENT_NOTES[2]
            target = write(repo, "scratch-target.txt", "# Same-looking bytes\n")
            note.unlink()
            note.symlink_to(target)
            result = run_guard(repo, guard, base)
            self.assertEqual(result.returncode, 2)
            self.assertIn("not a regular working file", result.stderr)

    def test_canonical_note_lists_agree(self):
        expected = set(PERMANENT_NOTES)
        self.assertEqual(set(GUARD_NOTES), expected)
        self.assertEqual(set(BUNDLE_NOTES), expected)

        ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        whitelisted = set()
        for match in re.finditer(r"^!/ai/notes/([^/]+\.md)$", ignore_text,
                                 flags=re.MULTILINE):
            whitelisted.add("ai/notes/" + match.group(1))
        self.assertEqual(whitelisted, expected)

        readme = (REPO_ROOT / "ai" / "README.md").read_text(encoding="utf-8")
        note_section = readme.split(
            "Exactly eleven Markdown notes are permanent repository knowledge:",
            1,
        )[1].split("The backlog, dated audits", 1)[0]
        readme_notes = set()
        for name in re.findall(r"^\d+\. `([^`]+\.md)`$", note_section,
                               flags=re.MULTILINE):
            readme_notes.add("ai/notes/" + name)
        self.assertEqual(readme_notes, expected)

        memory = (REPO_ROOT / "ai" / "notes" / "MEMORY.md").read_text(
            encoding="utf-8")
        memory_section = memory.split("## The permanent eleven", 1)[1].split(
            "## Local working records", 1)[0]
        memory_notes = set()
        for name in re.findall(r"\]\(([^)]+\.md)\)", memory_section):
            memory_notes.add("ai/notes/" + name)
        self.assertEqual(memory_notes, expected)

        contract = (REPO_ROOT / "ai" / "notes" /
                    "readme-go-no-go.md").read_text(encoding="utf-8")
        for path in expected:
            with self.subTest(contract_path=path):
                self.assertIn(path, contract)
        self.assertIn("ai/tools/permanent_note_guard.py", contract)

        tracked = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "ai/notes"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.split(b"\0")
        tracked_notes = set()
        for raw_path in tracked:
            if not raw_path:
                continue
            path_text = raw_path.decode("utf-8", errors="strict")
            path = Path(path_text)
            if path.parent == Path("ai/notes") and path.suffix == ".md":
                tracked_notes.add(path_text)
        self.assertEqual(tracked_notes, expected)


if __name__ == "__main__":
    unittest.main()
