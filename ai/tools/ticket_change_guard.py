#!/usr/bin/env python3
"""Check one committed ticket against a changed-character limit.

By default, the guard compares the complete tree at ``--base`` with the clean
tree at ``HEAD``.  That is the Implementer's self-check.  An Architect can
instead use ``--architect-audit --candidate COMMIT`` to name an immutable,
full commit.  This audit mode keeps measuring that commit even when ``HEAD``
has advanced to later work.

Intermediate commits do not receive separate allowances.  A replacement uses
the exact minimum number of single-character insertions and deletions.
Unchanged characters count zero.  Python Unicode code points are counted, not
UTF-8 storage bytes.

Exit codes:

* 0: the candidate is within the limit, or the limit is zero (unlimited);
* 1: the candidate is measurable but exceeds the limit;
* 2: the command or repository state cannot be checked safely.
"""

import argparse
from dataclasses import dataclass
import os
import re
import subprocess
import sys


FULL_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
ASCII_DECIMAL_RE = re.compile(r"[0-9]+", flags=re.ASCII)
ZERO_OBJECT_ID = "0" * 40
MAX_BLOB_BYTES = 4 * 1024 * 1024
MAX_AGGREGATE_BLOB_BYTES = 16 * 1024 * 1024
MAX_COMPARISON_MIDDLE_CODEPOINTS = 200_000
MAX_LCS_CELLS_PER_FILE = 64_000_000
MAX_TOTAL_LCS_CELLS = 128_000_000
GIT_COMMAND_TIMEOUT_SECONDS = 30
MAX_CHARACTERS_ENVIRONMENT = "MAILBOX_MAX_CHARACTERS"
AUTHORITATIVE_GUARD_ENVIRONMENT = "MAILBOX_TICKET_CHANGE_GUARD"

# These limits keep a malformed or unexpectedly large candidate from making a
# maintenance gate consume unbounded memory or processor time.  The blob
# limits apply before any blob is read.  The comparison limits apply after
# equal starts and ends have been removed.  They are named constants so tests
# and user documentation can state the exact boundary without copying an
# implementation detail hidden inside an expression.


class GuardError(RuntimeError):
    """Report a repository state that cannot produce a safe decision."""


@dataclass(frozen=True)
class DiffEntry:
    """Describe the old and new Git blobs for one changed path.

    A blob is Git's stored copy of one file's contents, named by a
    40-hex object identifier.

    Arguments:
      old_mode   = six-digit file mode before the change; "000000"
                   means the file did not exist.
      new_mode   = file mode after the change; "000000" means the file
                   was deleted.
      old_object = blob identifier before the change; all zeros means
                   no old contents.
      new_object = blob identifier after the change; all zeros means
                   no new contents.
      status     = Git's change status letter, such as M for modified
                   or R for renamed, possibly followed by a similarity
                   score.
      old_path   = path before the change, as raw bytes.
      new_path   = path after the change; equals old_path except for
                   renames and copies.
    """

    old_mode: str
    new_mode: str
    old_object: str
    new_object: str
    status: str
    old_path: bytes
    new_path: bytes


@dataclass(frozen=True)
class CharacterCount:
    """Hold the added and deleted character totals.

    Arguments:
      added   = minimum single-character insertions.
      deleted = minimum single-character deletions.
    """

    added: int
    deleted: int

    @property
    def total(self):
        """Return additions plus deletions."""
        return self.added + self.deleted


@dataclass(frozen=True)
class PreparedDelta:
    """Describe one trimmed exact character comparison.

    Characters equal at the start and at the end of both texts are
    excluded before the expensive comparison; only the differing
    middles are compared.

    Arguments:
      old_text  = complete old text.
      new_text  = complete new text.
      old_start = index where the old middle begins.
      old_end   = index one past the old middle's last character.
      new_start = index where the new middle begins.
      new_end   = index one past the new middle's last character.
      cells     = old middle length times new middle length — the size
                  of the full comparison table, used to budget work.
    """

    old_text: str
    new_text: str
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    cells: int


def nonnegative_integer(value):
    """Parse one nonempty string of ASCII decimal digits.

    Arguments:
      value = the command-line text.

    Returns:
      The parsed nonnegative integer.

    Raises:
      argparse.ArgumentTypeError: for anything but plain ASCII digits,
        so a sign, spaces, or Unicode digit forms cannot slip in.
    """
    if not isinstance(value, str) or ASCII_DECIMAL_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative ASCII decimal integer")
    try:
        return int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative ASCII decimal integer") from exc


def full_base_commit(value):
    """Require the full 40-hex spelling of the ticket's base commit.

    Arguments:
      value = the command-line text.

    Returns:
      The commit identifier in lowercase.

    Raises:
      argparse.ArgumentTypeError: for an abbreviation or anything that
        is not exactly forty hexadecimal digits.
    """
    if FULL_COMMIT_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "--base must be one full 40-hex commit")
    return value.lower()


def full_candidate_commit(value):
    """Require the full 40-hex spelling of an audited candidate commit.

    Arguments:
      value = the command-line text.

    Returns:
      The commit identifier in lowercase.

    Raises:
      argparse.ArgumentTypeError: for an abbreviation or anything that
        is not exactly forty hexadecimal digits.
    """
    if FULL_COMMIT_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "--candidate must be one full 40-hex commit")
    return value.lower()


def parse_args(argv=None):
    """Parse the read-only guard command line.

    Arguments:
      argv = argument list, or ``None`` for the process arguments.

    Returns:
      The parsed options. ``--architect-audit`` and ``--candidate``
      must appear together or not at all.
    """
    parser = argparse.ArgumentParser(
        description="check a committed ticket's changed-character limit")
    parser.add_argument(
        "--base", required=True, type=full_base_commit,
        help="full 40-hex commit before this ticket began")
    parser.add_argument(
        "--architect-audit", action="store_true",
        help="audit the immutable commit named by --candidate instead of "
             "checking the Implementer's current HEAD")
    parser.add_argument(
        "--candidate", type=full_candidate_commit,
        help="full commit for --architect-audit")
    parser.add_argument(
        "--max", dest="maximum", type=nonnegative_integer, default=None,
        help="largest accepted Unicode-code-point additions plus deletions; "
             "when omitted, use MAILBOX_MAX_CHARACTERS or 0; 0 is unlimited")
    parser.add_argument(
        "--repo", default=os.getcwd(),
        help="repository or a directory inside it (default: current directory)")
    args = parser.parse_args(argv)
    if args.architect_audit and args.candidate is None:
        parser.error("--architect-audit requires --candidate")
    if args.candidate is not None and not args.architect_audit:
        parser.error("--candidate requires --architect-audit")
    return args


def run_git(repository, arguments):
    """Run Git without a shell and return its raw output.

    Arguments:
      repository = folder passed to ``git -C``.
      arguments  = Git subcommand and options.

    Returns:
      The completed-process object with raw output bytes; the exit
      code is left for the caller to interpret.

    Raises:
      GuardError: when Git cannot start or exceeds the timeout.
    """
    command = ["git", "-C", repository] + list(arguments)
    environment = os.environ.copy()
    # Read-only status normally refreshes and rewrites the index as an
    # optimization.  A guard must not modify the candidate it is inspecting.
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, env=environment,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise GuardError(
            "Git did not finish within "
            + str(GIT_COMMAND_TIMEOUT_SECONDS) + " seconds") from exc
    except OSError as exc:
        raise GuardError("could not start Git: " + str(exc)) from exc
    return result


def selected_maximum(explicit, environment=None):
    """Bind an explicit limit to the dispatch environment, when present.

    The daemon exports the ticket's limit in MAILBOX_MAX_CHARACTERS
    when it dispatches work. A command-line ``--max`` must then agree
    with it, so a hand-typed rerun cannot quietly measure against a
    different limit than the dispatched one.

    Arguments:
      explicit    = the ``--max`` value, or ``None`` when omitted.
      environment = mapping to read instead of the process
                    environment; tests substitute one here.

    Returns:
      The limit to enforce; 0 means unlimited.

    Raises:
      GuardError: for a malformed environment value or a disagreement
        between ``--max`` and the environment.
    """
    values = os.environ if environment is None else environment
    inherited = values.get(MAX_CHARACTERS_ENVIRONMENT)
    inherited_value = None
    if inherited is not None:
        if ASCII_DECIMAL_RE.fullmatch(inherited) is None:
            raise GuardError(
                MAX_CHARACTERS_ENVIRONMENT
                + " must be a nonnegative ASCII decimal integer")
        try:
            inherited_value = int(inherited, 10)
        except ValueError as exc:
            raise GuardError(
                MAX_CHARACTERS_ENVIRONMENT
                + " must be a nonnegative ASCII decimal integer") from exc

    if explicit is None:
        return 0 if inherited_value is None else inherited_value
    if inherited_value is not None and explicit != inherited_value:
        raise GuardError(
            "--max does not match " + MAX_CHARACTERS_ENVIRONMENT
            + " (" + str(inherited_value) + ")")
    return explicit


def require_authoritative_script(environment=None):
    """Refuse an accidental worktree copy when dispatch names another tool.

    The daemon exports the absolute path of the guard copy it trusts.
    When this running file is not that copy — for example a stale
    duplicate inside a worktree — the check refuses, because a
    divergent copy could measure with different rules.

    Arguments:
      environment = mapping to read instead of the process
                    environment; tests substitute one here.

    Raises:
      GuardError: for a relative path in the variable or a mismatch
        with this file's resolved location.
    """
    values = os.environ if environment is None else environment
    authoritative = values.get(AUTHORITATIVE_GUARD_ENVIRONMENT)
    if authoritative is None:
        return
    if not os.path.isabs(authoritative):
        raise GuardError(
            AUTHORITATIVE_GUARD_ENVIRONMENT + " must be an absolute path")
    expected = os.path.realpath(authoritative)
    current = os.path.realpath(os.path.abspath(__file__))
    if current != expected:
        raise GuardError(
            "this ticket guard is not the authoritative dispatched copy; "
            + AUTHORITATIVE_GUARD_ENVIRONMENT + " names " + expected)


def git_error(result):
    """Return one readable Git diagnostic.

    Arguments:
      result = completed Git process.

    Returns:
      The stripped standard-error text, or the exit code when Git
      printed nothing.
    """
    message = result.stderr.decode("utf-8", errors="replace").strip()
    return message if message else "Git exited " + str(result.returncode)


def repository_root(path):
    """Resolve a directory inside one Git working tree to its root.

    Arguments:
      path = the ``--repo`` argument: the repository or any folder
             inside it.

    Returns:
      Absolute path of the working-tree root.

    Raises:
      GuardError: when the path is not inside a Git working tree or
        the returned root is unusable.
    """
    candidate = os.path.abspath(path)
    result = run_git(
        repository=candidate, arguments=["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise GuardError("--repo is not a Git working tree: "
                         + git_error(result=result))
    try:
        root = result.stdout.rstrip(b"\n").decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GuardError("repository path is not valid UTF-8") from exc
    if not root:
        raise GuardError("Git returned an empty repository path")
    return os.path.abspath(root)


def resolve_commit(repository, revision, label):
    """Resolve one revision to an exact commit object.

    Arguments:
      repository = working-tree root.
      revision   = revision text such as ``HEAD`` or a full commit.
      label      = name used in error messages, such as ``--base``.

    Returns:
      The full lowercase 40-hex commit identifier.

    Raises:
      GuardError: when the revision does not name a commit or Git
        returns something unusable.
    """
    result = run_git(
        repository=repository,
        arguments=["rev-parse", "--verify", revision + "^{commit}"])
    if result.returncode != 0:
        raise GuardError(label + " is not a commit: "
                         + git_error(result=result))
    try:
        commit = result.stdout.strip().decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise GuardError("Git returned a non-ASCII " + label) from exc
    if FULL_COMMIT_RE.fullmatch(commit) is None:
        raise GuardError("Git returned an invalid " + label)
    return commit.lower()


def require_ancestor(repository, base, candidate, candidate_label="HEAD"):
    """Require the selected base to be in the candidate's history.

    The ticket is measured as everything from the base to the
    candidate. If the base is not an ancestor, that span would include
    unrelated work and the measurement would be meaningless.

    Arguments:
      repository      = working-tree root.
      base            = full base commit.
      candidate       = full candidate commit.
      candidate_label = name for the candidate in error messages.

    Raises:
      GuardError: when the base is not an ancestor or ancestry cannot
        be checked.
    """
    result = run_git(
        repository=repository,
        arguments=["merge-base", "--is-ancestor", base, candidate])
    if result.returncode == 0:
        return
    if result.returncode == 1:
        raise GuardError(
            "--base is not an ancestor of " + candidate_label)
    raise GuardError("could not check commit ancestry: "
                     + git_error(result=result))


def worktree_changes(repository):
    """Return staged, unstaged, and nonignored untracked status bytes.

    Arguments:
      repository = working-tree root.

    Returns:
      Raw ``git status --porcelain`` output; empty bytes mean a clean
      tree.

    Raises:
      GuardError: when the status command fails.
    """
    result = run_git(
        repository=repository,
        arguments=["status", "--porcelain=v1", "-z",
                   "--untracked-files=normal", "--ignore-submodules=none"])
    if result.returncode != 0:
        raise GuardError("could not check the working tree: "
                         + git_error(result=result))
    return result.stdout


def hidden_index_flags(repository):
    """Return tracked paths whose index flags can hide working-tree edits.

    Git lets a user mark a tracked file assume-unchanged or
    skip-worktree; either mark makes ``git status`` omit edits to that
    file, so a tree that looks clean could still carry uncommitted
    work.

    Arguments:
      repository = working-tree root.

    Returns:
      List of ``(tag, path)`` pairs for every flagged file; empty when
      no file carries such a mark.

    Raises:
      GuardError: when the flag listing fails or is malformed.
    """
    result = run_git(
        repository=repository,
        arguments=["ls-files", "-v", "-z", "--"])
    if result.returncode != 0:
        raise GuardError("could not check tracked-file flags: "
                         + git_error(result=result))

    flagged = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise GuardError("Git returned malformed tracked-file flags")
        tag = record[:1]
        path = record[2:]
        # `git ls-files -v` changes the normal tag to lower case for
        # assume-unchanged entries.  S is the skip-worktree tag; s has both
        # properties.  Either property can make `git status` omit an edit.
        if tag == b"S" or tag.islower():
            flagged.append((tag, path))
    return flagged


def require_visible_index(repository):
    """Refuse index flags that make a positive-limit cleanliness check lie.

    Arguments:
      repository = working-tree root.

    Raises:
      GuardError: naming the first flagged file and its property when
        any tracked file is assume-unchanged or skip-worktree.
    """
    flags = hidden_index_flags(repository=repository)
    if not flags:
        return
    tag, path = flags[0]
    if tag == b"S":
        property_name = "skip-worktree"
    elif tag == b"s":
        property_name = "assume-unchanged and skip-worktree"
    else:
        property_name = "assume-unchanged"
    raise GuardError(
        "tracked file uses " + property_name + ", which can hide edits: "
        + display_path(path=path))


def require_clean_candidate(repository, expected_head):
    """Require HEAD to stay fixed and all nonignored work to be committed.

    Arguments:
      repository    = working-tree root.
      expected_head = the commit HEAD resolved to when the check
                      began.

    Raises:
      GuardError: when HEAD moved, an index flag hides edits, or the
        working tree carries uncommitted changes.
    """
    current_head = resolve_commit(
        repository=repository, revision="HEAD", label="HEAD")
    if current_head != expected_head:
        raise GuardError("HEAD changed while the ticket was being checked")
    require_visible_index(repository=repository)
    if worktree_changes(repository=repository):
        raise GuardError(
            "HEAD is not the exact candidate: commit or remove staged, "
            "unstaged, and nonignored untracked changes")


def require_exact_named_commit(repository, revision, expected, label):
    """Require a full commit name to keep resolving to the audited object.

    Arguments:
      repository = working-tree root.
      revision   = the audited full commit name.
      expected   = the commit it resolved to earlier.
      label      = name used in the error message.

    Raises:
      GuardError: when the name now resolves elsewhere, which would
        mean the audited object changed mid-check.
    """
    current = resolve_commit(
        repository=repository, revision=revision, label=label)
    if current != expected:
        raise GuardError(label + " changed while the ticket was being checked")


def display_path(path):
    """Render a Git path without allowing invalid bytes to hide an error.

    Arguments:
      path = raw path bytes from Git.

    Returns:
      The path as text; bytes that are not valid UTF-8 appear as
      backslash escapes instead of being dropped.
    """
    return path.decode("utf-8", errors="backslashreplace")


def binary_entry_keys(repository, base, candidate):
    """Return changed path pairs that Git classifies as binary.

    ``git diff --numstat`` prints ``-`` in place of line counts for a
    binary file; those entries are collected so the measurement can
    refuse them by name.

    Arguments:
      repository = working-tree root.
      base       = full base commit.
      candidate  = full candidate commit.

    Returns:
      Set of ``(old_path, new_path)`` byte pairs Git considers binary.

    Raises:
      GuardError: when the diff fails or its records are malformed.
    """
    result = run_git(
        repository=repository,
        arguments=["diff", "--numstat", "-z", "--find-renames=50%",
                   "--no-ext-diff", "--no-textconv", base, candidate])
    if result.returncode != 0:
        raise GuardError("could not inspect changed files: "
                         + git_error(result=result))

    pieces = result.stdout.split(b"\0")
    if pieces and pieces[-1] == b"":
        pieces.pop()
    binaries = set()
    index = 0
    while index < len(pieces):
        record = pieces[index]
        index += 1
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            raise GuardError("Git returned malformed change statistics")
        added, deleted, path = fields
        if path == b"":
            if index + 1 >= len(pieces):
                raise GuardError("Git returned a malformed rename record")
            old_path = pieces[index]
            new_path = pieces[index + 1]
            index += 2
        else:
            old_path = path
            new_path = path
        if added == b"-" or deleted == b"-":
            binaries.add((old_path, new_path))
    return binaries


def changed_entries(repository, base, candidate):
    """Read the blob pairs in the complete base-to-candidate tree change.

    Arguments:
      repository = working-tree root.
      base       = full base commit.
      candidate  = full candidate commit.

    Returns:
      List of DiffEntry records, one per changed path, with renames
      and copies detected at fifty percent similarity.

    Raises:
      GuardError: when the diff fails or a record is malformed.
    """
    result = run_git(
        repository=repository,
        arguments=["diff-tree", "--no-commit-id", "--raw", "-z", "-r",
                   "--no-abbrev", "--find-renames=50%", "--no-ext-diff",
                   "--no-textconv", base, candidate])
    if result.returncode != 0:
        raise GuardError("could not read the committed change: "
                         + git_error(result=result))

    pieces = result.stdout.split(b"\0")
    if pieces and pieces[-1] == b"":
        pieces.pop()
    entries = []
    index = 0
    while index < len(pieces):
        metadata = pieces[index]
        index += 1
        if not metadata.startswith(b":"):
            raise GuardError("Git returned malformed raw change metadata")
        fields = metadata[1:].split(b" ")
        if len(fields) != 5:
            raise GuardError("Git returned incomplete raw change metadata")
        try:
            old_mode, new_mode, old_object, new_object, status = (
                field.decode("ascii", errors="strict") for field in fields)
        except UnicodeDecodeError as exc:
            raise GuardError("Git returned non-ASCII raw metadata") from exc
        if not status or status[0] not in "ACDMRTUXB":
            raise GuardError("Git returned an unknown change status")
        if status[0] in "RC":
            if index + 1 >= len(pieces):
                raise GuardError("Git returned a malformed rename record")
            old_path = pieces[index]
            new_path = pieces[index + 1]
            index += 2
        else:
            if index >= len(pieces):
                raise GuardError("Git returned a change without a path")
            old_path = pieces[index]
            new_path = old_path
            index += 1
        entries.append(DiffEntry(
            old_mode=old_mode, new_mode=new_mode,
            old_object=old_object.lower(), new_object=new_object.lower(),
            status=status, old_path=old_path, new_path=new_path))
    return entries


def blob_size(repository, object_id, path):
    """Return one blob's byte size without reading its contents.

    Arguments:
      repository = working-tree root.
      object_id  = the blob's 40-hex identifier.
      path       = path used in error messages.

    Returns:
      The size in bytes.

    Raises:
      GuardError: when the blob cannot be inspected or Git returns an
        invalid size.
    """
    result = run_git(
        repository=repository, arguments=["cat-file", "-s", object_id])
    if result.returncode != 0:
        raise GuardError("could not inspect changed file "
                         + display_path(path=path) + ": "
                         + git_error(result=result))
    value = result.stdout.strip()
    try:
        size_text = value.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise GuardError("Git returned a non-ASCII changed-file size") from exc
    if ASCII_DECIMAL_RE.fullmatch(size_text) is None:
        raise GuardError("Git returned an invalid changed-file size")
    return int(size_text, 10)


def requested_blobs(entries):
    """Return each unique blob needed for content-changing entries.

    An entry whose old and new objects are identical — a pure rename —
    changes no characters and requests nothing.

    Arguments:
      entries = DiffEntry records for the tree change.

    Returns:
      Mapping from blob identifier to one representative path.

    Raises:
      GuardError: for a changed Git submodule, whose contents cannot
        be counted as text.
    """
    requests = {}
    for entry in entries:
        if entry.old_object == entry.new_object:
            continue
        sides = (
            (entry.old_mode, entry.old_object, entry.old_path),
            (entry.new_mode, entry.new_object, entry.new_path),
        )
        for mode, object_id, path in sides:
            if object_id == ZERO_OBJECT_ID or mode == "000000":
                continue
            if mode == "160000":
                raise GuardError(
                    "changed Git submodule cannot be counted as text: "
                    + display_path(path=path))
            requests.setdefault(object_id, path)
    return requests


def preflight_blob_reads(repository, entries):
    """Refuse oversized blob reads before any changed content is loaded.

    Arguments:
      repository = working-tree root.
      entries    = DiffEntry records for the tree change.

    Returns:
      The requested-blob mapping, each blob now known to fit the
      per-blob and aggregate byte limits.

    Raises:
      GuardError: when one blob or the running total exceeds its
        limit.
    """
    requests = requested_blobs(entries=entries)
    aggregate = 0
    for object_id, path in requests.items():
        size = blob_size(
            repository=repository, object_id=object_id, path=path)
        if size > MAX_BLOB_BYTES:
            raise GuardError(
                "changed file exceeds the per-blob read limit of "
                + str(MAX_BLOB_BYTES) + " bytes: "
                + display_path(path=path))
        if size > MAX_AGGREGATE_BLOB_BYTES - aggregate:
            raise GuardError(
                "changed files exceed the aggregate blob-read limit of "
                + str(MAX_AGGREGATE_BLOB_BYTES) + " bytes")
        aggregate += size
    return requests


def blob_text(repository, mode, object_id, path, cache):
    """Read one preflighted changed blob as strict UTF-8 text.

    Arguments:
      repository = working-tree root.
      mode       = file mode; ``"000000"`` or an all-zero object means
                   no contents, returned as the empty string.
      object_id  = the blob's identifier.
      path       = path used in error messages.
      cache      = mapping that stores decoded text per blob, so a
                   blob appearing on several paths is read once.

    Returns:
      The blob's text.

    Raises:
      GuardError: for a submodule, a blob that grew past the read
        limit, a zero byte (binary content), or invalid UTF-8.
    """
    if object_id == ZERO_OBJECT_ID or mode == "000000":
        return ""
    if mode == "160000":
        raise GuardError("changed Git submodule cannot be counted as text: "
                         + display_path(path=path))
    if object_id in cache:
        return cache[object_id]
    result = run_git(
        repository=repository,
        arguments=["cat-file", "blob", object_id])
    if result.returncode != 0:
        raise GuardError("could not read changed file "
                         + display_path(path=path) + ": "
                         + git_error(result=result))
    payload = result.stdout
    if len(payload) > MAX_BLOB_BYTES:
        raise GuardError("changed file grew past the per-blob read limit: "
                         + display_path(path=path))
    if b"\x00" in payload:
        raise GuardError("changed binary file cannot be counted as text: "
                         + display_path(path=path))
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GuardError("changed file is not valid UTF-8: "
                         + display_path(path=path)) from exc
    cache[object_id] = text
    return text


def prepare_character_delta(old_text, new_text):
    """Trim equal ends and bound one exact longest-common-subsequence job.

    Characters shared at the start and at the end of both texts cannot
    be part of any minimal edit, so only the differing middles enter
    the expensive comparison. The middle sizes are bounded so one file
    cannot demand unbounded memory.

    Arguments:
      old_text = complete old text.
      new_text = complete new text.

    Returns:
      A PreparedDelta with the middle boundaries and the comparison
      table size.

    Raises:
      GuardError: when the combined middles exceed the comparison size
        limit.
    """
    prefix = 0
    shared_length = min(len(old_text), len(new_text))
    while (prefix < shared_length
           and old_text[prefix] == new_text[prefix]):
        prefix += 1

    old_end = len(old_text)
    new_end = len(new_text)
    while (old_end > prefix and new_end > prefix
           and old_text[old_end - 1] == new_text[new_end - 1]):
        old_end -= 1
        new_end -= 1

    old_length = old_end - prefix
    new_length = new_end - prefix
    cells = 0
    if old_length and new_length:
        if old_length + new_length > MAX_COMPARISON_MIDDLE_CODEPOINTS:
            raise GuardError(
                "changed text exceeds the comparison size limit of "
                + str(MAX_COMPARISON_MIDDLE_CODEPOINTS) + " code points")
        cells = old_length * new_length

    return PreparedDelta(
        old_text=old_text, new_text=new_text,
        old_start=prefix, old_end=old_end,
        new_start=prefix, new_end=new_end,
        cells=cells)


def exact_lcs_length(prepared):
    """Return an exact longest-common-subsequence length in bounded memory.

    The longest common subsequence (LCS) is the longest sequence of
    characters appearing in both texts in the same order, not
    necessarily adjacent; the minimum edit counts follow from its
    length. The classic table is computed one row at a time with the
    shorter middle across the columns, so memory stays proportional to
    the smaller text while the answer stays exact.

    Arguments:
      prepared = the trimmed comparison.

    Returns:
      The LCS length of the two middles.
    """
    old_length = prepared.old_end - prepared.old_start
    new_length = prepared.new_end - prepared.new_start
    if old_length == 0 or new_length == 0:
        return 0

    if old_length <= new_length:
        column_text = prepared.old_text
        column_start = prepared.old_start
        column_length = old_length
        row_text = prepared.new_text
        row_start = prepared.new_start
        row_length = new_length
    else:
        column_text = prepared.new_text
        column_start = prepared.new_start
        column_length = new_length
        row_text = prepared.old_text
        row_start = prepared.old_start
        row_length = old_length

    # `values` is one row of the usual longest-common-subsequence table.  Each
    # new row overwrites the preceding row.  `diagonal`, `above`, and `left`
    # preserve the three table cells needed for the next decision.
    values = [0] * (column_length + 1)
    for row_offset in range(row_length):
        diagonal = 0
        left = 0
        row_character = row_text[row_start + row_offset]
        for column_offset in range(1, column_length + 1):
            above = values[column_offset]
            if row_character == column_text[column_start + column_offset - 1]:
                current = diagonal + 1
            else:
                current = above if above >= left else left
            values[column_offset] = current
            diagonal = above
            left = current
    return values[-1]


def count_prepared_delta(prepared):
    """Count the exact minimum character insertions and deletions.

    Every character outside the longest common subsequence must be
    inserted or deleted: additions are the new middle's length minus
    the LCS length, deletions the old middle's length minus it.

    Arguments:
      prepared = the trimmed comparison.

    Returns:
      The CharacterCount for this file.
    """
    old_length = prepared.old_end - prepared.old_start
    new_length = prepared.new_end - prepared.new_start
    common = exact_lcs_length(prepared=prepared)
    return CharacterCount(
        added=new_length - common, deleted=old_length - common)


def count_low_change_delta(prepared, maximum_work, limit_name):
    """Count an exact low-change edit without building a full LCS table.

    This is the frontier form of the exact difference algorithm: it
    explores edits of growing total size along the diagonals of the
    comparison table and stops at the first size that reaches the end
    of both texts. For a small edit to a large file it finishes after
    far fewer steps than the full table, at the price of growing with
    the edit size instead.

    Arguments:
      prepared     = the trimmed comparison.
      maximum_work = comparison steps this job may spend.
      limit_name   = ``"per-file"`` or ``"aggregate"``, named in the
                     error message.

    Returns:
      ``(count, work)``: the exact CharacterCount and the steps spent.

    Raises:
      GuardError: when the work budget is exhausted before the edit is
        found.
    """
    old = prepared.old_text[prepared.old_start:prepared.old_end]
    new = prepared.new_text[prepared.new_start:prepared.new_end]
    frontier = {1: 0}
    work = 0
    displayed_limit = (MAX_TOTAL_LCS_CELLS
                       if limit_name == "aggregate" else maximum_work)

    for distance in range(len(old) + len(new) + 1):
        for diagonal in range(-distance, distance + 1, 2):
            work += 1
            if work > maximum_work:
                raise GuardError(
                    "changed text exceeds the " + limit_name
                    + " exact-match work limit of "
                    + str(displayed_limit) + " comparison steps")
            if (diagonal == -distance
                    or (diagonal != distance
                        and frontier.get(diagonal - 1, -1)
                        < frontier.get(diagonal + 1, -1))):
                old_index = frontier.get(diagonal + 1, 0)
            else:
                old_index = frontier.get(diagonal - 1, 0) + 1
            new_index = old_index - diagonal
            while (old_index < len(old) and new_index < len(new)
                   and old[old_index] == new[new_index]):
                work += 1
                if work > maximum_work:
                    raise GuardError(
                        "changed text exceeds the " + limit_name
                        + " exact-match work limit of "
                        + str(displayed_limit) + " comparison steps")
                old_index += 1
                new_index += 1
            frontier[diagonal] = old_index
            if old_index == len(old) and new_index == len(new):
                length_difference = len(new) - len(old)
                return (CharacterCount(
                    added=(distance + length_difference) // 2,
                    deleted=(distance - length_difference) // 2), work)
    raise GuardError("exact character comparison did not finish")


def count_bounded_delta(prepared, maximum_work, limit_name):
    """Use the cheaper exact method that fits the remaining work budget.

    Both methods return exact counts; only their costs differ. The
    full table costs old-middle times new-middle characters; the
    frontier method costs roughly the text length times the edit size.
    The table is used when it fits the budget, the frontier method
    otherwise.

    Arguments:
      prepared     = the trimmed comparison.
      maximum_work = comparison steps this file may spend.
      limit_name   = ``"per-file"`` or ``"aggregate"``, named in
                     errors.

    Returns:
      ``(count, work)``: the exact CharacterCount and the steps spent.

    Raises:
      GuardError: when neither method fits the budget.
    """
    if prepared.cells <= maximum_work:
        return count_prepared_delta(prepared=prepared), prepared.cells
    return count_low_change_delta(
        prepared=prepared, maximum_work=maximum_work, limit_name=limit_name)


def character_delta(old_text, new_text):
    """Count one exact, symmetric Unicode-character change.

    Symmetric means a replacement counts both sides: renaming a
    variable counts the deleted old spelling and the inserted new
    spelling. The count is exact — the minimum number of
    single-character insertions and deletions — so two honest
    measurements of the same texts always agree.

    Arguments:
      old_text = complete old text.
      new_text = complete new text.

    Returns:
      The CharacterCount for this pair of texts.

    Raises:
      GuardError: when the texts exceed the comparison size or work
        limits.
    """
    prepared = prepare_character_delta(
        old_text=old_text, new_text=new_text)
    count, _ = count_bounded_delta(
        prepared=prepared, maximum_work=MAX_LCS_CELLS_PER_FILE,
        limit_name="per-file")
    return count


def measure_characters(repository, base, candidate):
    """Measure the full committed text change between two trees.

    Arguments:
      repository = working-tree root.
      base       = full base commit.
      candidate  = full candidate commit.

    Returns:
      The CharacterCount summed over every changed file.

    Raises:
      GuardError: for binary or oversized changes, submodules, invalid
        UTF-8, or an exhausted comparison work budget.
    """
    entries = changed_entries(
        repository=repository, base=base, candidate=candidate)
    # Object sizes are cheap to inspect.  Apply the memory limit before asking
    # Git to classify content, because binary classification may scan blobs.
    preflight_blob_reads(repository=repository, entries=entries)
    binary_keys = binary_entry_keys(
        repository=repository, base=base, candidate=candidate)
    for entry in entries:
        if entry.old_object == entry.new_object:
            continue
        if (entry.old_path, entry.new_path) in binary_keys:
            raise GuardError(
                "changed binary file cannot be counted as text: "
                + display_path(path=entry.new_path))

    cache = {}
    prepared_deltas = []
    for entry in entries:
        # An identical object moved to a new path, including a binary or a
        # non-UTF-8 object, changes no characters and needs no blob read.
        if entry.old_object == entry.new_object:
            continue
        old_text = blob_text(
            repository=repository, mode=entry.old_mode,
            object_id=entry.old_object, path=entry.old_path, cache=cache)
        new_text = blob_text(
            repository=repository, mode=entry.new_mode,
            object_id=entry.new_object, path=entry.new_path, cache=cache)
        prepared = prepare_character_delta(
            old_text=old_text, new_text=new_text)
        prepared_deltas.append(prepared)

    if all(delta.cells <= MAX_LCS_CELLS_PER_FILE
           for delta in prepared_deltas):
        planned_work = sum(delta.cells for delta in prepared_deltas)
        if planned_work > MAX_TOTAL_LCS_CELLS:
            raise GuardError(
                "changed files exceed the aggregate exact-match work limit of "
                + str(MAX_TOTAL_LCS_CELLS) + " character pairs")

    total_cells = 0
    added = 0
    deleted = 0
    for prepared in prepared_deltas:
        remaining = MAX_TOTAL_LCS_CELLS - total_cells
        maximum_work = min(MAX_LCS_CELLS_PER_FILE, remaining)
        limit_name = ("aggregate" if remaining < MAX_LCS_CELLS_PER_FILE
                      else "per-file")
        count, work = count_bounded_delta(
            prepared=prepared, maximum_work=maximum_work,
            limit_name=limit_name)
        total_cells += work
        added += count.added
        deleted += count.deleted
    return CharacterCount(added=added, deleted=deleted)


def print_identity(base, candidate, maximum):
    """Print the exact commits and selected limit.

    Arguments:
      base      = full base commit.
      candidate = full candidate commit.
      maximum   = selected limit; zero prints as unlimited.
    """
    print("base commit: " + base)
    print("candidate commit: " + candidate)
    if maximum == 0:
        print("maximum changed characters: unlimited (0)")
    else:
        print("maximum changed characters: " + str(maximum))


def main(argv=None):
    """Run the ticket change check.

    The cleanliness or exact-commit requirement is checked again after
    measuring, so a tree that changed mid-measurement is refused
    rather than reported with a stale number.

    Arguments:
      argv = argument list, or ``None`` for the process arguments.

    Returns:
      The process exit code: 0 within the limit or unlimited, 1 over
      the limit, 2 when the repository state cannot be checked safely.
    """
    args = parse_args(argv=argv)
    try:
        require_authoritative_script()
        maximum = selected_maximum(explicit=args.maximum)
        repository = repository_root(path=args.repo)
        base = resolve_commit(
            repository=repository, revision=args.base, label="--base")
        if base != args.base:
            raise GuardError("--base does not resolve to the exact named commit")
        audit_mode = args.architect_audit
        candidate_revision = args.candidate if audit_mode else "HEAD"
        candidate_label = "--candidate" if audit_mode else "HEAD"
        candidate = resolve_commit(
            repository=repository, revision=candidate_revision,
            label=candidate_label)
        if audit_mode and candidate != args.candidate:
            raise GuardError(
                "--candidate does not resolve to the exact named commit")
        require_ancestor(
            repository=repository, base=base, candidate=candidate,
            candidate_label=candidate_label)

        if audit_mode:
            require_exact_named_commit(
                repository=repository, revision=args.candidate,
                expected=candidate, label="--candidate")
        else:
            require_clean_candidate(
                repository=repository, expected_head=candidate)

        if maximum == 0:
            print("ticket change guard: size limit disabled")
            print_identity(base=base, candidate=candidate,
                           maximum=maximum)
            print("changed-character measurement: skipped because --max 0 "
                  "is unlimited")
            return 0

        count = measure_characters(
            repository=repository, base=base, candidate=candidate)
        if audit_mode:
            require_exact_named_commit(
                repository=repository, revision=args.candidate,
                expected=candidate, label="--candidate")
        else:
            require_clean_candidate(
                repository=repository, expected_head=candidate)
    except GuardError as exc:
        print("ticket change guard: cannot measure: " + str(exc))
        return 2

    if count.total > maximum:
        print("ticket change guard: over limit")
        print_identity(base=base, candidate=candidate,
                       maximum=maximum)
        print("changed characters: " + str(count.total) + " ("
              + str(count.added) + " added + " + str(count.deleted)
              + " deleted)")
        print("reason: the committed ticket exceeds the selected maximum")
        return 1

    print("ticket change guard: within limit")
    print_identity(base=base, candidate=candidate, maximum=maximum)
    print("changed characters: " + str(count.total) + " ("
          + str(count.added) + " added + " + str(count.deleted)
          + " deleted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
