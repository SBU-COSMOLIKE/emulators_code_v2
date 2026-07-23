#!/usr/bin/env python3
"""Compare protected AI knowledge with one Architect-pinned Git commit.

This is a guardrail against accidental edits during any Implementer or Red
Team work.  It does not decide GO or NO-GO.  The Architect records the full
starting commit in the implementation directive and reruns this tool before
making that decision.

The eleven Markdown notes remain a fixed census. The structured role contract
and the small protected reference catalog receive the same byte-for-byte
protection.

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

try:
    from ai.tools.role_contract import ROLE_CONTRACT
except ImportError:  # Direct execution from ai/tools/.
    from role_contract import ROLE_CONTRACT

_BOOTSTRAP_GUARD_PATHS = (
    "ai/tools/permanent_note_guard.py",
    "ai/tools/role_contract.py",
)
_PATHS = ROLE_CONTRACT["protected_paths"]
GUARD_PATHS = tuple(_PATHS["guard_files"].values())
GUARD_PATH = _PATHS["guard_files"]["permanent_note_guard"]
ROLE_CONTRACT_PATH = _PATHS["contract"]
NOTES_ROOT = PurePosixPath(ROLE_CONTRACT_PATH).parent
PERMANENT_NOTES = tuple(_PATHS["permanent_notes"])
PROTECTED_REFERENCE_FILES = tuple(_PATHS["protected_reference_files"])
PROTECTED_POLICY_FILES = (
    PERMANENT_NOTES + PROTECTED_REFERENCE_FILES + (ROLE_CONTRACT_PATH,))
BACKLOG_PATH = ROLE_CONTRACT["backlog"]["path"]

FULL_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
MAX_FILE_BYTES = ROLE_CONTRACT["limits"]["protected_policy_file_bytes"]


class GuardError(RuntimeError):
    """A protected file or Git state does not match the pinned base.

    Any raise means the Architect must stop and inspect before handing
    work to another role or issuing the final decision; the guard
    itself never repairs anything.
    """


def _clean_git_environment():
    """Return an environment without caller-selected Git state overrides.

    Git honors environment variables that redirect where it looks:
    ``GIT_DIR`` (the repository database), ``GIT_WORK_TREE`` (the
    working folder), and ``GIT_INDEX_FILE`` (the staging area). A
    caller that set one of them could make every check below inspect
    some other repository while appearing to inspect this one, so all
    three are removed. ``GIT_NO_REPLACE_OBJECTS`` additionally
    disables Git's object-replacement feature, which can silently
    substitute one commit or file body for another during reads.

    Returns:
      A copy of the process environment with the overrides removed.
    """
    environment = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        environment.pop(name, None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _git(repo, *arguments, check=True):
    """Run one read-only Git command and return raw output.

    The command is passed as an argument list, never through a shell,
    so no text in ``arguments`` can be interpreted as shell syntax.
    Output is returned as raw bytes because file contents and paths
    are compared byte for byte; decoding is each caller's decision.

    Arguments:
      repo      = real path to the checkout being inspected.
      arguments = Git arguments passed without shell interpretation.
      check     = when True, raise on a nonzero Git exit; when False,
                  return the completed result for the caller to judge
                  (used where a nonzero exit is itself the answer).

    Returns:
      The ``subprocess.CompletedProcess`` with captured output.

    Raises:
      GuardError naming the Git command and its error text.
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

    ``git rev-parse --show-toplevel`` reports the top folder of the
    repository containing a path. The named folder must equal that
    report exactly: naming a subfolder, or a path that merely sits
    near a repository, is refused rather than silently widened to
    whatever Git found above it. The Architect's directive names one
    exact worktree, and this check holds the tool to it.

    Arguments:
      repo_argument = the ``--repo`` value, or None. None selects the
                      checkout containing this installed tool, found
                      two directory levels above ``ai/tools/``.

    Returns:
      The validated top-level folder as an absolute ``Path``.

    Raises:
      GuardError when Git refuses the path or reports a different
      top-level folder.
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
    """Require one full commit hash and return Git's canonical spelling.

    Three rules bind ``--base`` to one exact commit. The spelling must
    be a complete lowercase hash (40 hexadecimal characters, or 64 for
    a repository using the newer SHA-256 object format); an
    abbreviation is refused because a short prefix can become
    ambiguous as the repository grows. ``git rev-parse --verify`` with
    the ``^{commit}`` suffix must resolve it to a commit object whose
    canonical name equals the given spelling exactly, so a tag or
    other indirect name cannot stand in for the commit it points at.
    And ``git merge-base --is-ancestor`` must confirm the base lies in
    the history of the current ``HEAD``: comparing against a commit
    from an unrelated line of history would make every later
    difference report meaningless.

    Arguments:
      repo = the validated checkout from ``repository_root``.
      base = the ``--base`` value from the Architect directive.

    Returns:
      The canonical full hash.

    Raises:
      GuardError when the spelling, the resolution, or the ancestry
      check fails.
    """
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
    """Decode a NUL-delimited Git path list without losing odd filenames.

    The Git listing commands are run with ``-z``, which separates
    paths with NUL bytes instead of newlines. A filename may legally
    contain a newline, so newline-separated output would be ambiguous;
    a NUL byte can never appear inside a path, so this split is exact.
    Each entry must decode as strict UTF-8, the one text encoding this
    repository accepts.

    Arguments:
      raw   = the raw bytes printed by a ``-z`` Git listing.
      label = the state being listed, used in the error message.

    Returns:
      The list of decoded path strings, empty entries dropped.

    Raises:
      GuardError when an entry is not valid UTF-8.
    """
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
    """Select tracked Markdown files directly below ``ai/notes``.

    ``path.parent == NOTES_ROOT`` keeps only files immediately inside
    the notes folder, not in any subfolder, and the suffix comparison
    is case-insensitive so ``.MD`` cannot slip past the census. The
    tracked backlog is excluded: it is operational state with its own
    guard, not a permanent note.

    Arguments:
      paths = decoded path strings from one Git listing.

    Returns:
      The matching paths, in their original order.
    """
    selected = []
    for text in paths:
        path = PurePosixPath(text)
        if (path.parent == NOTES_ROOT and path.suffix.casefold() == ".md"
                and text != BACKLOG_PATH):
            selected.append(text)
    return selected


def _reject_name_collisions(paths, label):
    """Reject names that differ only by case or Unicode normalization.

    Two distinct byte spellings can name the same file on common
    filesystems: macOS folders are usually case-insensitive, and the
    same accented character can be stored composed or decomposed
    (Unicode offers both forms). Normalizing to the composed form
    (``NFC``) and lowering case (``casefold``) maps such lookalikes to
    one key; two different paths sharing a key would let one note
    silently shadow another, so the pair is refused by name.

    Arguments:
      paths = the note paths from one Git listing.
      label = the state being listed, used in the error message.

    Returns:
      None; a passing check does nothing.

    Raises:
      GuardError naming both colliding spellings.
    """
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
    """Require the eleven tracked top-level Markdown notes and no others.

    Both directions refuse: a missing note means protected knowledge
    was deleted or renamed, and an extra note means a twelfth file is
    posing as permanent policy without the Architect-only route that
    creates one. The error lists the missing and extra names so the
    reader repairs the exact difference.

    Arguments:
      paths = decoded path strings from one Git listing.
      label = the state being checked, used in the error message.

    Returns:
      None; a passing check does nothing.

    Raises:
      GuardError listing ``missing=`` and ``extra=`` names.
    """
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
    """List every tracked notes-root path in the base commit.

    ``git ls-tree -r`` walks one commit's saved tree recursively;
    ``--name-only -z`` prints just the paths, NUL-separated for
    ``_decode_paths``. Only the notes folder is listed.

    Arguments:
      repo = the validated checkout.
      base = the canonical pinned commit.

    Returns:
      The decoded path list for the base commit.
    """
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
    """List every tracked notes-root path at the current ``HEAD``.

    Same listing as ``_base_paths`` but against ``HEAD``, Git's name
    for the commit currently checked out in this worktree.

    Arguments:
      repo = the validated checkout.

    Returns:
      The decoded path list for the current ``HEAD``.
    """
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
    """List every notes-root path in the Git staging area.

    The staging area (also called the index) holds the file versions
    selected for the next commit. ``git ls-files`` lists it, so an
    edit that was staged but not yet committed is visible to the
    census too.

    Arguments:
      repo = the validated checkout.

    Returns:
      The decoded path list for the staging area.
    """
    result = _git(repo, "ls-files", "-z", "--", str(NOTES_ROOT))
    return _decode_paths(result.stdout, "Git staging area")


def _git_bytes(repo, object_name, path):
    """Read one exact blob from a commit or the Git staging area.

    ``git show name:path`` prints the saved bytes of one file (a blob)
    at one state. A small Git idiom carries the fourth state: an empty
    ``object_name`` produces the spelling ``:path``, which reads the
    staged copy from the index rather than from any commit.

    Arguments:
      repo        = the validated checkout.
      object_name = a commit name, or the empty string for the
                    staging area.
      path        = the repository-relative file to read.

    Returns:
      The exact saved bytes.

    Raises:
      GuardError when Git fails or the blob exceeds the protected-file
      size limit.
    """
    result = _git(repo, "show", object_name + ":" + path)
    if len(result.stdout) > MAX_FILE_BYTES:
        raise GuardError(path + " exceeds the protected-note size limit")
    return result.stdout


def _stat_signature(metadata):
    """Reduce one stat result to the identity fields compared for change.

    ``os.stat`` reports a file's metadata. Seven facts together
    identify one unchanged file: the device and inode numbers name the
    exact file object on disk (a replacement file gets a new inode
    even under the same path name), the mode records the file kind and
    permissions, the link count says how many directory names point at
    the object, and the size plus the nanosecond modification and
    change stamps expose an edit in place. Comparing signatures taken
    at different moments answers: is this still the same, untouched
    file?

    Arguments:
      metadata = one ``os.stat_result`` from ``lstat`` or ``fstat``.

    Returns:
      A tuple of the seven facts, ready for equality comparison.
    """
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
    """Read one regular working file once and reject links or replacement.

    This is the working-tree counterpart of ``_git_bytes``, and it
    inherits nothing from Git's own consistency, so it defends itself:
    the path must be an ordinary file, not a symbolic link (``lstat``
    does not follow links, and the ``O_NOFOLLOW`` open flag refuses
    one at open time); it must have exactly one filesystem name, so no
    second hard link can edit the bytes just approved; the size cap is
    checked from metadata and again on the bytes actually read (the
    loop reads one byte past the cap to detect growth); and the
    seven-fact ``_stat_signature`` must be identical before the open,
    after the open, after the read, and on a final fresh ``lstat``, so
    a file swapped or edited mid-read is refused rather than
    half-read.

    Arguments:
      repo      = the validated checkout.
      path_text = the repository-relative file, in POSIX spelling; it
                  is joined onto ``repo`` component by component so
                  the platform's own separator rules apply.

    Returns:
      The file's exact bytes.

    Raises:
      GuardError naming the path and the first failed guarantee.
    """
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
    """Return the SHA-256 hex digest of exact bytes.

    Arguments:
      data = the exact bytes to fingerprint.

    Returns:
      The 64-character lowercase hexadecimal fingerprint; changing one
      input byte produces an unrelated value.
    """
    return hashlib.sha256(data).hexdigest()


def _require_same(path, expected, observed, state):
    """Refuse with both digests when one protected file differs.

    The error names the file, the state where it differs, and both
    fingerprints, so the reader can see immediately which side moved
    and compare against other records without rerunning anything.

    Arguments:
      path     = the protected file being compared.
      expected = the bytes at the pinned base commit.
      observed = the bytes found in the named state.
      state    = the human name of that state, such as
                 ``Git staging area``.

    Returns:
      None when the bytes match.

    Raises:
      GuardError carrying ``base_sha256`` and ``observed_sha256``.
    """
    if observed == expected:
        return
    raise GuardError(
        path + " differs in " + state + "; base_sha256="
        + _sha256(expected) + " observed_sha256=" + _sha256(observed))


def _require_guard_unchanged(repo, base):
    """Catch an incidental edit to either trusted guard before using it.

    The obvious way to defeat a change detector is to change both the
    watched file and the detector. Before trusting its own path lists,
    the guard therefore compares the two files that define what
    ``protected`` means, this tool and the role-contract reader,
    across the same four states as everything else. Only after both
    prove unchanged does the real census run.

    Arguments:
      repo = the validated checkout.
      base = the canonical pinned commit.

    Returns:
      None; a passing check does nothing.

    Raises:
      GuardError from the four-state comparison.
    """
    for path in _BOOTSTRAP_GUARD_PATHS:
        expected = _git_bytes(repo, base, path)
        head = _git_bytes(repo, "HEAD", path)
        staged = _git_bytes(repo, "", path)
        working = _working_bytes(repo, path)
        _require_same(path, expected, head, "current HEAD")
        _require_same(path, expected, staged, "Git staging area")
        _require_same(path, expected, working, "working tree")


def _verify_snapshot(repo, base):
    """Read one complete protected-state snapshot.

    One snapshot runs, in order: record the current ``HEAD``; prove
    the two guard files unchanged (``_require_guard_unchanged``);
    require the role contract to agree with the compiled guard list;
    require the exact eleven-note census in the base commit, the
    current ``HEAD``, and the staging area; then compare every
    protected policy file across all four states, collecting one
    fingerprint row per file. A final ``HEAD`` read must equal the
    first, so a commit created mid-snapshot invalidates it rather than
    producing a half-old, half-new report.

    The four states cover the ways an accidental edit could hide: an
    edit in the working tree only, an edit already staged for the next
    commit, and an edit already committed on top of the base are each
    caught against the same pinned bytes.

    Arguments:
      repo = the validated checkout.
      base = the canonical pinned commit.

    Returns:
      The tuple ``(head_bytes, rows)`` where ``rows`` holds one
      ``(sha256, path)`` pair per protected file.

    Raises:
      GuardError on the first failed comparison or census.
    """
    head_before = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").stdout
    _require_guard_unchanged(repo=repo, base=base)
    if frozenset(GUARD_PATHS) != frozenset(_BOOTSTRAP_GUARD_PATHS):
        raise GuardError(
            "role contract disagrees with its two protected guards")
    _require_exact_note_set(_base_paths(repo, base), "base commit")
    _require_exact_note_set(_head_paths(repo), "current HEAD")
    _require_exact_note_set(_index_paths(repo), "Git staging area")

    rows = []
    for path in PROTECTED_POLICY_FILES:
        expected = _git_bytes(repo, base, path)
        head = _git_bytes(repo, "HEAD", path)
        staged = _git_bytes(repo, "", path)
        working = _working_bytes(repo, path)
        _require_same(path, expected, head, "current HEAD")
        _require_same(path, expected, staged, "Git staging area")
        _require_same(path, expected, working, "working tree")
        rows.append((_sha256(expected), path))
    head_after = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").stdout
    if head_after != head_before:
        raise GuardError("current HEAD changed during protected-state checks")
    return head_before, rows


def verify(repo, base):
    """Require two identical complete protected-state snapshots.

    A single pass reads many files one after another, so a concurrent
    edit could land between two reads and still produce an
    internally consistent-looking report. Taking the whole snapshot
    twice and requiring equality closes that window: any change to any
    protected file, census, or ``HEAD`` between the passes makes the
    snapshots differ, and the verdict becomes a refusal instead of a
    report about a state that never existed.

    Arguments:
      repo = the validated checkout.
      base = the canonical pinned commit.

    Returns:
      The fingerprint rows of the verified snapshot, one
      ``(sha256, path)`` pair per protected file, for ``main`` to
      print.

    Raises:
      GuardError from either snapshot, or when the two differ.
    """
    first = _verify_snapshot(repo=repo, base=base)
    second = _verify_snapshot(repo=repo, base=base)
    if second != first:
        raise GuardError("protected state changed while it was verified")
    return first[1]


def build_parser():
    """Build the ``--base`` / ``--repo`` command-line parser.

    Returns:
      The ``argparse`` parser: ``--base`` is required and carries the
      full starting commit from the Architect directive; ``--repo``
      optionally names the exact worktree. Kept separate from ``main``
      so tests can inspect the command surface without running a
      verification.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare the eleven permanent notes and protected role contract "
            "with an Architect-pinned starting commit."))
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
    """Compare the protected files with the pinned base and report.

    The comparison covers four states of every protected file: the base
    commit, the current ``HEAD``, the Git staging area, and the working
    files. A complete match prints one digest line per file and a final
    ``PERMANENT-NOTE-GUARD PASS`` line, then exits 0. Any difference or
    unreadable state prints one refusal line to standard error and exits
    2; the tool never modifies a file either way.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        repo = repository_root(repo_argument=arguments.repo)
        base = canonical_base(repo=repo, base=arguments.base)
        rows = verify(repo=repo, base=base)
    except (GuardError, OSError, UnicodeError) as error:
        print("permanent-note guard refused: " + str(error), file=sys.stderr)
        return 2

    print("Permanent notes and role contract match the Architect-pinned "
          "starting commit.")
    print("base: " + base)
    print("worktree: " + str(repo))
    print("states: current HEAD, Git staging area, working tree")
    for digest, path in rows:
        print(digest + "  " + path)
    print(
        "PERMANENT-NOTE-GUARD PASS base=" + base
        + " notes=" + str(len(PERMANENT_NOTES)) + " contract=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
