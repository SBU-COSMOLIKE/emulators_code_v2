#!/usr/bin/env python3
"""Compare the permanent AI notes with one Architect-pinned Git commit.

This is a guardrail against accidental edits during any Implementer or Red
Team work.  It does not decide GO or NO-GO.  The Architect records the full
starting commit in the implementation directive and reruns this tool before
making that decision.

The expected SHA-256 values are calculated from Git.  They are not stored in
an editable checksum file.  The tool compares the base commit, current HEAD,
Git staging area, and working files so that staging or committing an accidental
note edit does not hide it.
"""

import argparse
import hashlib
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess
import sys
import unicodedata


GUARD_PATH = "ai/tools/permanent_note_guard.py"
NOTES_ROOT = PurePosixPath("ai/notes")
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

FULL_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
MAX_FILE_BYTES = 4 * 1024 * 1024


class GuardError(RuntimeError):
    """A protected file or Git state does not match the pinned base."""


def _clean_git_environment():
    """Return an environment without caller-selected Git state overrides."""
    environment = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        environment.pop(name, None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _git(repo, *arguments, check=True):
    """Run one read-only Git command and return raw output.

    Arguments:
        repo: Real path to the checkout being inspected.
        arguments: Git arguments passed without shell interpretation.
        check: Raise ``GuardError`` when Git exits nonzero.
    """
    result = subprocess.run(
        ["git", "-C", str(repo)] + list(arguments),
        env=_clean_git_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = result.stdout.decode("utf-8", errors="replace").strip()
        raise GuardError("git " + " ".join(arguments) + " failed: " + detail)
    return result


def repository_root(repo_argument=None):
    """Resolve and validate the checkout that the Architect named.

    Arguments:
        repo_argument: Optional checkout path.  When absent, use the checkout
            containing this installed tool.
    """
    if repo_argument is None:
        candidate = Path(__file__).resolve().parents[2]
    else:
        candidate = Path(repo_argument).expanduser().resolve()
    result = _git(candidate, "rev-parse", "--show-toplevel")
    try:
        reported = Path(result.stdout.decode("utf-8").strip()).resolve()
    except UnicodeDecodeError as error:
        raise GuardError("Git reported a non-UTF-8 checkout path") from error
    if reported != candidate:
        raise GuardError(
            "named checkout is not its Git top-level folder: " + str(candidate))
    return reported


def canonical_base(repo, base):
    """Require one full commit hash and return Git's canonical spelling."""
    if FULL_COMMIT_RE.fullmatch(base) is None:
        raise GuardError("--base must be a full lowercase Git commit hash")
    result = _git(repo, "rev-parse", "--verify", base + "^{commit}")
    canonical = result.stdout.decode("ascii", errors="strict").strip()
    if canonical != base:
        raise GuardError("--base does not name that exact commit object")
    ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        canonical,
        "HEAD",
        check=False,
    )
    if ancestor.returncode != 0:
        raise GuardError("the pinned base is not an ancestor of current HEAD")
    return canonical


def _decode_paths(raw, label):
    """Decode a NUL-delimited Git path list without losing odd filenames."""
    paths = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            path = item.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise GuardError(label + " contains a non-UTF-8 path") from error
        paths.append(path)
    return paths


def _top_level_markdown(paths):
    """Select tracked Markdown files directly below ``ai/notes``."""
    selected = []
    for text in paths:
        path = PurePosixPath(text)
        if path.parent == NOTES_ROOT and path.suffix.casefold() == ".md":
            selected.append(text)
    return selected


def _reject_name_collisions(paths, label):
    """Reject names that differ only by case or Unicode normalization."""
    seen = {}
    for path in paths:
        key = unicodedata.normalize("NFC", path).casefold()
        previous = seen.get(key)
        if previous is not None and previous != path:
            raise GuardError(
                label + " contains colliding note names: " + previous
                + " and " + path)
        seen[key] = path


def _require_exact_note_set(paths, label):
    """Require the eleven tracked top-level Markdown notes and no others."""
    notes = _top_level_markdown(paths)
    _reject_name_collisions(notes, label)
    expected = set(PERMANENT_NOTES)
    observed = set(notes)
    if observed == expected and len(notes) == len(expected):
        return
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    parts = [label + " does not contain exactly the eleven permanent notes"]
    if missing:
        parts.append("missing=" + ",".join(missing))
    if extra:
        parts.append("extra=" + ",".join(extra))
    raise GuardError("; ".join(parts))


def _base_paths(repo, base):
    result = _git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        base,
        "--",
        str(NOTES_ROOT),
    )
    return _decode_paths(result.stdout, "base commit")


def _head_paths(repo):
    result = _git(
        repo,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        "HEAD",
        "--",
        str(NOTES_ROOT),
    )
    return _decode_paths(result.stdout, "current HEAD")


def _index_paths(repo):
    result = _git(repo, "ls-files", "-z", "--", str(NOTES_ROOT))
    return _decode_paths(result.stdout, "Git staging area")


def _git_bytes(repo, object_name, path):
    """Read one exact blob from a commit or the Git staging area."""
    result = _git(repo, "show", object_name + ":" + path)
    if len(result.stdout) > MAX_FILE_BYTES:
        raise GuardError(path + " exceeds the protected-note size limit")
    return result.stdout


def _stat_signature(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _working_bytes(repo, path_text):
    """Read one regular working file once and reject links or replacement."""
    path = repo.joinpath(*PurePosixPath(path_text).parts)
    try:
        before = path.lstat()
    except OSError as error:
        raise GuardError("cannot inspect " + path_text + ": " + str(error))
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise GuardError(path_text + " is not a regular working file")
    if before.st_nlink != 1:
        raise GuardError(path_text + " has more than one filesystem name")
    if before.st_size > MAX_FILE_BYTES:
        raise GuardError(path_text + " exceeds the protected-note size limit")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GuardError("cannot open " + path_text + ": " + str(error))
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(before):
            raise GuardError(path_text + " changed while it was opened")
        chunks = []
        remaining = MAX_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_FILE_BYTES:
            raise GuardError(path_text + " exceeds the protected-note size limit")
        finished = os.fstat(descriptor)
        if _stat_signature(finished) != _stat_signature(opened):
            raise GuardError(path_text + " changed while it was read")
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise GuardError("cannot recheck " + path_text + ": " + str(error))
    if _stat_signature(after) != _stat_signature(before):
        raise GuardError(path_text + " changed during the comparison")
    return data


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _require_same(path, expected, observed, state):
    if observed == expected:
        return
    raise GuardError(
        path + " differs in " + state + "; base_sha256="
        + _sha256(expected) + " observed_sha256=" + _sha256(observed))


def _require_guard_unchanged(repo, base):
    """Catch an incidental edit to this guard before trusting its result."""
    expected = _git_bytes(repo, base, GUARD_PATH)
    head = _git_bytes(repo, "HEAD", GUARD_PATH)
    staged = _git_bytes(repo, "", GUARD_PATH)
    working = _working_bytes(repo, GUARD_PATH)
    _require_same(GUARD_PATH, expected, head, "current HEAD")
    _require_same(GUARD_PATH, expected, staged, "Git staging area")
    _require_same(GUARD_PATH, expected, working, "working tree")


def verify(repo, base):
    """Verify all protected states and return the base SHA-256 rows."""
    _require_guard_unchanged(repo=repo, base=base)
    _require_exact_note_set(_base_paths(repo, base), "base commit")
    _require_exact_note_set(_head_paths(repo), "current HEAD")
    _require_exact_note_set(_index_paths(repo), "Git staging area")

    rows = []
    for path in PERMANENT_NOTES:
        expected = _git_bytes(repo, base, path)
        head = _git_bytes(repo, "HEAD", path)
        staged = _git_bytes(repo, "", path)
        working = _working_bytes(repo, path)
        _require_same(path, expected, head, "current HEAD")
        _require_same(path, expected, staged, "Git staging area")
        _require_same(path, expected, working, "working tree")
        rows.append((_sha256(expected), path))
    return rows


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the eleven permanent notes with an Architect-pinned "
            "starting commit."))
    parser.add_argument(
        "--base",
        required=True,
        help="full starting commit recorded in the Architect directive",
    )
    parser.add_argument(
        "--repo",
        help=(
            "exact worktree from the Architect directive; default: the "
            "checkout containing this tool"),
    )
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        repo = repository_root(repo_argument=arguments.repo)
        base = canonical_base(repo=repo, base=arguments.base)
        rows = verify(repo=repo, base=base)
    except (GuardError, OSError, UnicodeError) as error:
        print("permanent-note guard refused: " + str(error), file=sys.stderr)
        return 2

    print("Permanent notes match the Architect-pinned starting commit.")
    print("base: " + base)
    print("worktree: " + str(repo))
    print("states: current HEAD, Git staging area, working tree")
    for digest, path in rows:
        print(digest + "  " + path)
    print(
        "PERMANENT-NOTE-GUARD PASS base=" + base
        + " notes=" + str(len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
