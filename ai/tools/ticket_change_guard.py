#!/usr/bin/env python3
"""Check one committed ticket against a changed-character limit.

The guard compares the complete tree at ``--base`` with the complete tree at
``HEAD``.  Intermediate commits do not receive separate allowances.  A
replacement uses the exact minimum number of single-character insertions and
deletions.  Unchanged characters count zero.  Python Unicode code points are
counted, not UTF-8 storage bytes.

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
MAX_LCS_CELLS_PER_FILE = 4_000_000
MAX_TOTAL_LCS_CELLS = 8_000_000
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
    """Describe the old and new Git blobs for one changed path."""

    old_mode: str
    new_mode: str
    old_object: str
    new_object: str
    status: str
    old_path: bytes
    new_path: bytes


@dataclass(frozen=True)
class CharacterCount:
    """Hold the added and deleted character totals."""

    added: int
    deleted: int

    @property
    def total(self):
        """Return additions plus deletions."""
        return self.added + self.deleted


@dataclass(frozen=True)
class PreparedDelta:
    """Describe one trimmed exact character comparison."""

    old_text: str
    new_text: str
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    cells: int


def nonnegative_integer(value):
    """Parse one nonempty string of ASCII decimal digits."""
    if not isinstance(value, str) or ASCII_DECIMAL_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative ASCII decimal integer")
    try:
        return int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative ASCII decimal integer") from exc


def full_commit(value):
    """Require the full 40-hex spelling used by this repository."""
    if FULL_COMMIT_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "--base must be one full 40-hex commit")
    return value.lower()


def parse_args(argv=None):
    """Parse the read-only guard command line."""
    parser = argparse.ArgumentParser(
        description="check a committed ticket's changed-character limit")
    parser.add_argument(
        "--base", required=True, type=full_commit,
        help="full 40-hex commit before this ticket began")
    parser.add_argument(
        "--max", dest="maximum", type=nonnegative_integer, default=None,
        help="largest accepted Unicode-code-point additions plus deletions; "
             "when omitted, use MAILBOX_MAX_CHARACTERS or 0; 0 is unlimited")
    parser.add_argument(
        "--repo", default=os.getcwd(),
        help="repository or a directory inside it (default: current directory)")
    return parser.parse_args(argv)


def run_git(repository, arguments):
    """Run Git without a shell and return its raw output."""
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
    """Bind an explicit limit to the dispatch environment, when present."""
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
    """Refuse an accidental worktree copy when dispatch names another tool."""
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
    """Return one readable Git diagnostic."""
    message = result.stderr.decode("utf-8", errors="replace").strip()
    return message if message else "Git exited " + str(result.returncode)


def repository_root(path):
    """Resolve a directory inside one Git working tree to its root."""
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
    """Resolve one revision to an exact commit object."""
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


def require_ancestor(repository, base, candidate):
    """Require the selected base to be in the candidate's history."""
    result = run_git(
        repository=repository,
        arguments=["merge-base", "--is-ancestor", base, candidate])
    if result.returncode == 0:
        return
    if result.returncode == 1:
        raise GuardError("--base is not an ancestor of HEAD")
    raise GuardError("could not check commit ancestry: "
                     + git_error(result=result))


def worktree_changes(repository):
    """Return staged, unstaged, and nonignored untracked status bytes."""
    result = run_git(
        repository=repository,
        arguments=["status", "--porcelain=v1", "-z",
                   "--untracked-files=normal", "--ignore-submodules=none"])
    if result.returncode != 0:
        raise GuardError("could not check the working tree: "
                         + git_error(result=result))
    return result.stdout


def hidden_index_flags(repository):
    """Return tracked paths whose index flags can hide working-tree edits."""
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
    """Refuse index flags that make a positive-limit cleanliness check lie."""
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
    """Require HEAD to stay fixed and all nonignored work to be committed."""
    current_head = resolve_commit(
        repository=repository, revision="HEAD", label="HEAD")
    if current_head != expected_head:
        raise GuardError("HEAD changed while the ticket was being checked")
    require_visible_index(repository=repository)
    if worktree_changes(repository=repository):
        raise GuardError(
            "HEAD is not the exact candidate: commit or remove staged, "
            "unstaged, and nonignored untracked changes")


def display_path(path):
    """Render a Git path without allowing invalid bytes to hide an error."""
    return path.decode("utf-8", errors="backslashreplace")


def binary_entry_keys(repository, base, candidate):
    """Return changed path pairs that Git classifies as binary."""
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
    """Read the blob pairs in the complete base-to-candidate tree change."""
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
    """Return one blob's byte size without reading its contents."""
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
    """Return each unique blob needed for content-changing entries."""
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
    """Refuse oversized blob reads before any changed content is loaded."""
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
    """Read one preflighted changed blob as strict UTF-8 text."""
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
    """Trim equal ends and bound one exact longest-common-subsequence job."""
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
        if old_length > MAX_LCS_CELLS_PER_FILE // new_length:
            raise GuardError(
                "changed text exceeds the per-file exact-match work limit of "
                + str(MAX_LCS_CELLS_PER_FILE) + " character pairs")
        cells = old_length * new_length

    return PreparedDelta(
        old_text=old_text, new_text=new_text,
        old_start=prefix, old_end=old_end,
        new_start=prefix, new_end=new_end,
        cells=cells)


def exact_lcs_length(prepared):
    """Return an exact longest-common-subsequence length in bounded memory."""
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
    """Count the exact minimum character insertions and deletions."""
    old_length = prepared.old_end - prepared.old_start
    new_length = prepared.new_end - prepared.new_start
    common = exact_lcs_length(prepared=prepared)
    return CharacterCount(
        added=new_length - common, deleted=old_length - common)


def character_delta(old_text, new_text):
    """Count one exact, symmetric Unicode-character change."""
    prepared = prepare_character_delta(
        old_text=old_text, new_text=new_text)
    return count_prepared_delta(prepared=prepared)


def measure_characters(repository, base, candidate):
    """Measure the full committed text change between two trees."""
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
    total_cells = 0
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
        if prepared.cells > MAX_TOTAL_LCS_CELLS - total_cells:
            raise GuardError(
                "changed files exceed the aggregate exact-match work limit of "
                + str(MAX_TOTAL_LCS_CELLS) + " character pairs")
        total_cells += prepared.cells
        prepared_deltas.append(prepared)

    added = 0
    deleted = 0
    for prepared in prepared_deltas:
        count = count_prepared_delta(prepared=prepared)
        added += count.added
        deleted += count.deleted
    return CharacterCount(added=added, deleted=deleted)


def print_identity(base, candidate, maximum):
    """Print the exact commits and selected limit."""
    print("base commit: " + base)
    print("candidate commit: " + candidate)
    if maximum == 0:
        print("maximum changed characters: unlimited (0)")
    else:
        print("maximum changed characters: " + str(maximum))


def main(argv=None):
    """Run the ticket change check."""
    args = parse_args(argv=argv)
    try:
        require_authoritative_script()
        maximum = selected_maximum(explicit=args.maximum)
        repository = repository_root(path=args.repo)
        base = resolve_commit(
            repository=repository, revision=args.base, label="--base")
        if base != args.base:
            raise GuardError("--base does not resolve to the exact named commit")
        candidate = resolve_commit(
            repository=repository, revision="HEAD", label="HEAD")
        require_ancestor(
            repository=repository, base=base, candidate=candidate)

        if maximum == 0:
            print("ticket change guard: size limit disabled")
            print_identity(base=base, candidate=candidate,
                           maximum=maximum)
            print("changed-character measurement: skipped because --max 0 "
                  "is unlimited")
            return 0

        require_clean_candidate(
            repository=repository, expected_head=candidate)
        count = measure_characters(
            repository=repository, base=base, candidate=candidate)
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
