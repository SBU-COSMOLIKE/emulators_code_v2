#!/usr/bin/env python3
"""Detect accidental edits to the Architect-owned tracked backlog.

Git preserves completed backlog updates. During an active ticket, the
Architect may need to edit the backlog before the ticket lands. This tool
keeps one small ignored state file beside it and records the exact bytes the
Architect accepted for that landing, as a SHA-256 fingerprint: a
64-hexadecimal-character value computed from the file's exact bytes, where
changing even one character produces an unrelated value.

Three commands manage the state. ``check`` compares the backlog with the
accepted fingerprint and changes nothing. ``initialize`` records the first
fingerprint. ``seal`` accepts one deliberate Architect edit, and it demands
the fingerprint that ``check`` printed before the edit, so every accepted
state names its predecessor and the only path to a new seal is
check, then edit, then seal, in that order.

This is an accidental-change guard, not an authorization system.  A malicious
program that can rewrite both local files can defeat it.  Its purpose is to
make an Implementer or Red Team hallucination visible before the Architect
continues working.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid

try:
    from ai.tools.role_contract import ROLE_CONTRACT
except ImportError:  # Direct execution from ai/tools/.
    from role_contract import ROLE_CONTRACT

BACKLOG_RELATIVE_PATH = Path(ROLE_CONTRACT["backlog"]["path"])
STATE_FILENAME = ".backlog-guard.json"
LOCK_FILENAME = ".backlog-guard.lock"
STATE_VERSION = 1
SEALED_STATE_VERSION = 2
MAX_BACKLOG_BYTES = 16 * 1024 * 1024
MAX_STATE_BYTES = 16 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAILBOX_ROLE_ENVIRONMENT = "MAILBOX_ROLE"


class GuardError(RuntimeError):
    """The backlog or its saved fingerprint cannot be trusted.

    Any raise means the guard could not prove the accepted state, so
    the caller must stop and inspect instead of continuing to edit or
    approve work against unverified backlog bytes.
    """


def repository_root(repo_argument=None):
    """Return the checkout named by the caller or containing this tool.

    Arguments:
      repo_argument = the ``--repo`` value, or None. None selects the
                      repository containing this file, found by walking
                      two directory levels up from ``ai/tools/``. A
                      given path is expanded (``~`` becomes the home
                      folder) and resolved to an absolute location.

    Returns:
      The repository folder as an absolute ``Path``.

    Raises:
      GuardError when the named folder does not exist.
    """
    if repo_argument is None:
        repo = Path(__file__).resolve().parents[2]
    else:
        repo = Path(repo_argument).expanduser().resolve()
    if not repo.is_dir():
        raise GuardError("the named repository folder does not exist")
    return repo


def _stat_signature(metadata):
    """Return filesystem facts that change when one file is replaced.

    ``os.stat`` reports a file's metadata. Seven facts together
    identify one unchanged file: the device and inode numbers name the
    exact file object on disk (a replacement file gets a new inode
    even under the same path name), the mode records the file kind and
    permissions, the link count says how many directory names point at
    the object, and the size plus the nanosecond modification and
    change stamps expose an edit in place. Comparing two signatures
    taken at different moments therefore answers: is this still the
    same, untouched file?

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


def _require_plain_directory(path, label):
    """Refuse a missing directory or one redirected through a link.

    ``lstat`` reports what the path itself is without following a
    symbolic link, a small file that redirects to another path. A
    symbolic link where a directory is expected would let the guard
    read a backlog somewhere else entirely, so only a real directory
    passes.

    Arguments:
      path  = the directory to inspect.
      label = the short name used in the error, such as ``ai/notes``.

    Returns:
      None; a passing check does nothing.

    Raises:
      GuardError when the path is missing, unreadable, a link, or not
      a directory.
    """
    try:
        metadata = path.lstat()
    except OSError as error:
        raise GuardError("cannot inspect " + label + ": " + str(error))
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise GuardError(label + " is not a real directory")


def guard_paths(repo):
    """Return validated paths for the backlog, state, and write lock.

    The three files live together in ``ai/notes/``: the tracked
    ``backlog.md``, the ignored fingerprint state, and the ignored
    persistent lock. Both parent directories are first proved to be
    real directories, not symbolic-link redirects.

    Arguments:
      repo = the repository folder from ``repository_root``.

    Returns:
      The tuple ``(backlog_path, state_path, lock_path)``.

    Raises:
      GuardError from ``_require_plain_directory``.
    """
    ai_directory = repo / "ai"
    notes_directory = ai_directory / "notes"
    _require_plain_directory(ai_directory, "ai")
    _require_plain_directory(notes_directory, "ai/notes")
    return (
        notes_directory / "backlog.md",
        notes_directory / STATE_FILENAME,
        notes_directory / LOCK_FILENAME,
    )


def _read_regular_bytes(path, label, maximum_bytes):
    """Read one stable regular file without following a final symlink.

    The guard's verdicts are only as good as its reads, so this reader
    refuses every ambiguous situation instead of tolerating it:

    - the path must be an ordinary file, not a symbolic link, checked
      with ``lstat`` (which does not follow links) and opened with the
      ``O_NOFOLLOW`` flag so the open itself also refuses a link;
    - the file must have exactly one filesystem name (link count 1),
      because a second hard link would let another path edit the same
      bytes the guard just approved;
    - the size must stay within ``maximum_bytes``, checked from
      metadata before opening and again on the bytes actually read
      (the loop reads one byte past the limit to detect growth);
    - the seven-fact ``_stat_signature`` must be identical before the
      open, after the open, after the read, and on a final fresh
      ``lstat``, so a file swapped or edited mid-read is refused
      rather than half-read.

    Arguments:
      path          = the file to read.
      label         = the short name used in every error message.
      maximum_bytes = the largest accepted file size in bytes.

    Returns:
      The file's exact bytes.

    Raises:
      GuardError naming the label and the first failed guarantee.
    """
    try:
        before = path.lstat()
    except OSError as error:
        raise GuardError("cannot inspect " + label + ": " + str(error))
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise GuardError(label + " is not a regular file")
    if before.st_nlink != 1:
        raise GuardError(label + " has more than one filesystem name")
    if before.st_size > maximum_bytes:
        raise GuardError(label + " exceeds its size limit")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GuardError("cannot open " + label + ": " + str(error))
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(before):
            raise GuardError(label + " changed while it was opened")
        chunks = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > maximum_bytes:
            raise GuardError(label + " exceeds its size limit")
        finished = os.fstat(descriptor)
        if _stat_signature(finished) != _stat_signature(opened):
            raise GuardError(label + " changed while it was read")
    finally:
        os.close(descriptor)

    try:
        after = path.lstat()
    except OSError as error:
        raise GuardError("cannot recheck " + label + ": " + str(error))
    if _stat_signature(after) != _stat_signature(before):
        raise GuardError(label + " changed during the read")
    return data


def _sha256(data):
    """Calculate the canonical lowercase SHA-256 spelling.

    Arguments:
      data = the exact bytes to fingerprint.

    Returns:
      The 64-character lowercase hexadecimal fingerprint. Lowercase is
      the one accepted spelling everywhere in this tool, so string
      comparison of two fingerprints is exact.
    """
    return hashlib.sha256(data).hexdigest()


def _state_document(digest, previous_digest=None):
    """Build the only accepted state representation.

    Two versions exist. Version 1 records an initialization: the
    backlog's path and its accepted fingerprint. Version 2 records a
    seal and adds ``previous_sha256``, the fingerprint the Architect
    checked before editing, which is what chains each accepted state
    to its predecessor and lets a crashed seal be recognized as a
    harmless retry (see ``seal``).

    Arguments:
      digest          = the accepted backlog fingerprint.
      previous_digest = the predecessor fingerprint for a seal, or
                        None for an initialization.

    Returns:
      The state mapping with exactly the fields of its version.
    """
    document = {
        "backlog": BACKLOG_RELATIVE_PATH.as_posix(),
        "sha256": digest,
        "version": (STATE_VERSION if previous_digest is None
                    else SEALED_STATE_VERSION),
    }
    if previous_digest is not None:
        document["previous_sha256"] = previous_digest
    return document


def _canonical_state_bytes(document):
    """Encode state deterministically so incidental state edits are visible.

    ``json.dumps`` with sorted keys and two-space indentation, plus one
    trailing newline, gives every state document exactly one byte
    representation. The parser later requires the stored bytes to
    equal this encoding, so even an edit that preserves JSON meaning,
    such as reordering keys or changing whitespace, changes the bytes
    and is refused.

    Arguments:
      document = the state mapping from ``_state_document``.

    Returns:
      The canonical UTF-8 bytes.
    """
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


def _reject_duplicate_json_pairs(pairs):
    """Build the state object while refusing any repeated JSON field.

    Used as the ``object_pairs_hook`` of ``json.loads``, which
    otherwise keeps only the last spelling of a repeated key. A state
    file that spelled ``sha256`` twice could show one value to a human
    and a different value to the program; refusing the repeat keeps
    one reading.

    Arguments:
      pairs = the ``(key, value)`` tuples of one JSON object, in file
              order, supplied by ``json.loads``.

    Returns:
      A plain ``dict`` in which each field appeared exactly once.

    Raises:
      GuardError naming the repeated field.
    """
    document = {}
    for key, value in pairs:
        if key in document:
            raise GuardError("backlog guard state repeats the field " + key)
        document[key] = value
    return document


def _parse_state(data):
    """Parse and validate the complete saved state file.

    Acceptance requires, in order: strict UTF-8 JSON with no repeated
    field; one JSON object; exactly the fields of its version (the
    sealed version 2 adds ``previous_sha256``, and nothing else may
    appear); a version that is exactly the integer 1 or 2; the
    backlog path this tool protects; well-formed lowercase SHA-256
    values; and, last, bytes equal to the canonical encoding of the
    parsed document, so a state file has exactly one accepted
    spelling.

    Arguments:
      data = the exact bytes of the state file.

    Returns:
      The validated state mapping.

    Raises:
      GuardError on the first violated rule.
    """
    try:
        text = data.decode("utf-8", errors="strict")
        document = json.loads(
            text, object_pairs_hook=_reject_duplicate_json_pairs)
    except GuardError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GuardError("backlog guard state is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise GuardError("backlog guard state must be one JSON object")
    version = document.get("version")
    expected_fields = {"backlog", "sha256", "version"}
    if version == SEALED_STATE_VERSION:
        expected_fields.add("previous_sha256")
    if set(document) != expected_fields:
        raise GuardError("backlog guard state has missing or unexpected fields")
    if type(version) is not int or version not in {
            STATE_VERSION, SEALED_STATE_VERSION}:
        raise GuardError("backlog guard state uses an unsupported version")
    if document["backlog"] != BACKLOG_RELATIVE_PATH.as_posix():
        raise GuardError("backlog guard state names a different backlog")
    digest = document["sha256"]
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise GuardError("backlog guard state has an invalid SHA-256 value")
    if version == SEALED_STATE_VERSION:
        previous = document["previous_sha256"]
        if (not isinstance(previous, str)
                or SHA256_RE.fullmatch(previous) is None):
            raise GuardError(
                "backlog guard state has an invalid previous SHA-256 value")
    if data != _canonical_state_bytes(document):
        raise GuardError("backlog guard state is not in canonical form")
    return document


def _read_state(state_path):
    """Read and validate the saved state; return its bytes and document.

    Arguments:
      state_path = the state file beside the backlog.

    Returns:
      The tuple ``(bytes, document)``: callers that must detect a
      mid-operation state change compare the raw bytes of two reads,
      not only the parsed values.

    Raises:
      GuardError from the stable read or from ``_parse_state``.
    """
    data = _read_regular_bytes(
        state_path, "backlog guard state", MAX_STATE_BYTES)
    return data, _parse_state(data)


def _require_architect(acknowledged):
    """Allow writes only from the mailbox Architect or explicit manual use.

    The mailbox daemon launches every role session with the
    environment variable ``MAILBOX_ROLE`` naming that session's role,
    so inside a role process the identity is not a matter of opinion.
    When the variable names a role, only ``architect`` may write; an
    Implementer or Red Team process is refused by name. When the
    variable is absent, the command is a person at a terminal, and the
    explicit ``--architect-ack`` flag is that person's acknowledgment
    of the Architect-only rule.

    Arguments:
      acknowledged = True when ``--architect-ack`` was given.

    Returns:
      None; a permitted write continues.

    Raises:
      GuardError naming the refused role, or asking for the flag.
    """
    role = os.environ.get(MAILBOX_ROLE_ENVIRONMENT, "").strip()
    if role:
        if role.casefold() != "architect":
            raise GuardError(
                "MAILBOX_ROLE identifies " + role
                + "; only the Architect may replace the saved backlog SHA-256")
        return
    if not acknowledged:
        raise GuardError(
            "manual initialization or sealing requires --architect-ack")


class _WriteLock:
    """Hold one crash-released lock on the persistent lock file.

    Two guard commands writing state at the same time could interleave
    their read-check-write sequences and publish a state neither of
    them verified. The lock serializes writers through ``flock``, an
    advisory kernel lock on an open file: advisory means only programs
    that ask for the lock are affected, and kernel-owned means the
    lock disappears automatically when the holding process exits or
    crashes, so a died writer can never leave the guard permanently
    locked. The lock is requested exclusively and without waiting
    (``LOCK_EX | LOCK_NB``), so a second writer fails immediately with
    an explanation instead of queueing invisibly. The holder's process
    number is written into the file purely as a human diagnostic.

    Entering also re-proves the lock file itself: it must be one
    ordinary file with one name, and the open descriptor must still
    name the same device and inode as the path, so a link or a swap
    cannot redirect the serialization elsewhere.
    """

    def __init__(self, path):
        """Remember the lock path; nothing is opened until ``with``."""
        self.path = path
        self.descriptor = None

    def __enter__(self):
        """Acquire the exclusive lock or raise; returns this holder."""
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as error:
            raise GuardError("cannot open backlog guard lock: " + str(error))
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise GuardError("backlog guard lock is not one regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise GuardError(
                    "another backlog guard write is active; inspect "
                    + str(self.path)) from error
            observed = self.path.lstat()
            if ((opened.st_dev, opened.st_ino)
                    != (observed.st_dev, observed.st_ino)):
                raise GuardError("backlog guard lock changed while opened")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            data = (str(os.getpid()) + "\n").encode("ascii")
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:
                    raise GuardError("cannot write backlog guard lock")
                offset += written
            os.fsync(descriptor)
            self.descriptor = descriptor
        except GuardError:
            os.close(descriptor)
            raise
        except OSError as error:
            os.close(descriptor)
            raise GuardError(
                "cannot acquire backlog guard lock: " + str(error)) from error
        except BaseException:
            os.close(descriptor)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the lock and close the descriptor; never swallows."""
        if self.descriptor is not None:
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self.descriptor)
                self.descriptor = None
        return False


def _atomic_write_state(state_path, document):
    """Save canonical state beside the backlog and leave no partial file.

    The write follows the standard crash-safe sequence. The bytes go
    to a uniquely named temporary file first, created with ``O_EXCL``
    so an existing name is never reused, and ``fsync`` forces them to
    disk. ``os.replace`` then swaps the temporary name onto the real
    name in one atomic step: every observer sees either the complete
    old state or the complete new state, never a half-written file. A
    final ``fsync`` on the directory makes the name change itself
    durable, because renaming edits the directory, not the file. The
    ``finally`` clause removes the temporary file on any failure, so a
    crashed write leaves evidence, not debris that a later run might
    mistake for state.

    Arguments:
      state_path = the real state-file path.
      document   = the state mapping from ``_state_document``.

    Returns:
      None.

    Raises:
      GuardError wrapping the first failed file operation.
    """
    data = _canonical_state_bytes(document)
    temporary = state_path.with_name(
        ".backlog-guard.json.tmp-" + str(os.getpid()) + "-" + uuid.uuid4().hex)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, state_path)
        directory_descriptor = os.open(state_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise GuardError("cannot save backlog guard state: " + str(error))
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def check(repo):
    """Return the accepted digest after comparing exact current bytes.

    Read-only, and safe for any role to run. The backlog and the state
    file are each read twice, and both reads must return identical
    bytes: without the second look, a file being edited during the
    check could produce a verdict about a state that never existed on
    disk. Only then is the backlog's fingerprint compared with the
    accepted one, and the mismatch error names both values so the
    reader can see immediately which side moved.

    Arguments:
      repo = the repository folder from ``repository_root``.

    Returns:
      The accepted SHA-256, for the Architect to copy into a later
      ``seal --previous-sha256``.

    Raises:
      GuardError on unreadable or unstable files, a malformed state,
      or a fingerprint mismatch.
    """
    backlog_path, state_path, _ = guard_paths(repo)
    backlog = _read_regular_bytes(
        backlog_path, "ai/notes/backlog.md", MAX_BACKLOG_BYTES)
    state_bytes, state = _read_state(state_path)
    repeated_backlog = _read_regular_bytes(
        backlog_path, "ai/notes/backlog.md", MAX_BACKLOG_BYTES)
    repeated_state_bytes, repeated_state = _read_state(state_path)
    if repeated_backlog != backlog:
        raise GuardError("ai/notes/backlog.md changed during the check")
    if repeated_state_bytes != state_bytes or repeated_state != state:
        raise GuardError("backlog guard state changed during the check")
    observed = _sha256(backlog)
    expected = state["sha256"]
    if observed != expected:
        raise GuardError(
            "backlog SHA-256 mismatch; accepted=" + expected
            + " observed=" + observed)
    return expected


def initialize(repo, acknowledged=False):
    """Create first state for an existing backlog.

    Architect-only (see ``_require_architect``). Under the write lock,
    a still-missing state file is created from the backlog's current
    fingerprint and immediately re-verified through ``check``. When a
    state file already exists, the command does not overwrite it:
    a previous ``initialize`` may have published the complete state
    and died before reporting success, so the retry simply runs
    ``check`` and accepts only that exact already-published result.
    Repeating the command is therefore always safe.

    Arguments:
      repo         = the repository folder from ``repository_root``.
      acknowledged = True when ``--architect-ack`` was given.

    Returns:
      The accepted SHA-256.

    Raises:
      GuardError when authority, the files, or the verification fail.
    """
    _require_architect(acknowledged)
    backlog_path, state_path, lock_path = guard_paths(repo)
    with _WriteLock(lock_path):
        try:
            existing = state_path.lstat()
        except FileNotFoundError:
            existing = None
        except OSError as error:
            raise GuardError("cannot inspect backlog guard state: " + str(error))
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(
                    existing.st_mode):
                raise GuardError("backlog guard state is not a regular file")
            # A previous initialize may have published the complete state and
            # died before reporting success.  Accept only that exact result.
            return check(repo)
        backlog = _read_regular_bytes(
            backlog_path, "ai/notes/backlog.md", MAX_BACKLOG_BYTES)
        digest = _sha256(backlog)
        _atomic_write_state(state_path, _state_document(digest))
        accepted = check(repo)
        if accepted != digest:
            raise GuardError("saved backlog state did not verify")
        return digest


def seal(repo, previous_digest, acknowledged=False):
    """Replace accepted state after proving which state was checked first.

    Architect-only. The caller must present the fingerprint that
    ``check`` printed before the edit, and it must equal the currently
    saved one; a stale or guessed value is refused with instructions
    to run ``check`` first. This is what makes the seal a chain: the
    new state records the presented value as ``previous_sha256``, so
    every accepted state names its predecessor, and skipping the check
    step is structurally impossible.

    One deliberate exception keeps a crash harmless. If an earlier
    ``seal`` wrote the new state and died before reporting success,
    the retry presents a value that no longer matches the saved
    ``sha256`` but does match the saved ``previous_sha256``. When the
    backlog also still verifies against that saved state, the retry is
    recognized as a no-op and returns the already-accepted value; any
    later edit still fails the check. Before the atomic replacement,
    both files are read a second time and must be byte-identical to
    the first read, so nothing changed while the command was deciding.

    Arguments:
      repo            = the repository folder from ``repository_root``.
      previous_digest = the accepted SHA-256 printed by ``check``
                        before the backlog edit.
      acknowledged    = True when ``--architect-ack`` was given.

    Returns:
      The newly accepted SHA-256 of the edited backlog.

    Raises:
      GuardError when authority, the chain proof, file stability, or
      the final verification fail.
    """
    _require_architect(acknowledged)
    if SHA256_RE.fullmatch(previous_digest or "") is None:
        raise GuardError("--previous-sha256 must be 64 lowercase hex characters")
    backlog_path, state_path, lock_path = guard_paths(repo)
    with _WriteLock(lock_path):
        original_state_bytes, original_state = _read_state(state_path)
        if previous_digest != original_state["sha256"]:
            # A previous seal may have published the new digest and died
            # before reporting success.  A matching backlog makes the retry a
            # harmless no-op; any later edit still fails this check.
            if (original_state["version"] == SEALED_STATE_VERSION
                    and original_state["previous_sha256"]
                    == previous_digest):
                try:
                    accepted = check(repo)
                except GuardError:
                    accepted = None
                if accepted == original_state["sha256"]:
                    return accepted
            raise GuardError(
                "--previous-sha256 does not match the saved state; run check "
                "before editing and copy its accepted SHA-256")
        backlog = _read_regular_bytes(
            backlog_path, "ai/notes/backlog.md", MAX_BACKLOG_BYTES)
        new_digest = _sha256(backlog)

        # A second stable read catches an incidental state edit between the
        # first comparison and the atomic replacement.
        repeated_state_bytes, repeated_state = _read_state(state_path)
        if (repeated_state_bytes != original_state_bytes
                or repeated_state != original_state):
            raise GuardError("backlog guard state changed while sealing")
        repeated_backlog = _read_regular_bytes(
            backlog_path, "ai/notes/backlog.md", MAX_BACKLOG_BYTES)
        if repeated_backlog != backlog:
            raise GuardError("ai/notes/backlog.md changed while sealing")

        _atomic_write_state(
            state_path, _state_document(new_digest, previous_digest))
        accepted = check(repo)
        if accepted != new_digest:
            raise GuardError("new backlog state did not verify")
        return new_digest


def build_parser():
    """Build the ``check`` / ``initialize`` / ``seal`` command parser.

    Returns:
      The ``argparse`` parser with the global ``--repo`` option and
      the three subcommands; ``initialize`` and ``seal`` carry the
      ``--architect-ack`` acknowledgment, and ``seal`` requires
      ``--previous-sha256``. Kept separate from ``main`` so tests can
      inspect the command surface without running a command.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Detect accidental changes to the Architect-owned backlog."))
    parser.add_argument(
        "--repo",
        help="checkout to inspect; default: the checkout containing this tool",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check",
        help="compare the backlog with the SHA-256 last accepted by Architect",
    )

    initialize_parser = commands.add_parser(
        "initialize",
        help="save the first accepted SHA-256 for the existing backlog",
    )
    initialize_parser.add_argument(
        "--architect-ack",
        action="store_true",
        help="acknowledge that this manual command is run by the Architect",
    )

    seal_parser = commands.add_parser(
        "seal",
        help="accept an Architect edit after naming the state checked before it",
    )
    seal_parser.add_argument(
        "--previous-sha256",
        required=True,
        help="accepted SHA-256 printed by check before the backlog edit",
    )
    seal_parser.add_argument(
        "--architect-ack",
        action="store_true",
        help="acknowledge that this manual command is run by the Architect",
    )
    return parser


def main(argv=None):
    """Run one guard command and print its verdict lines.

    ``check`` compares the backlog with the accepted fingerprint and
    changes nothing. ``initialize`` and ``seal`` write the ignored state
    file and are Architect-only. A completed command prints the accepted
    SHA-256 and a final ``BACKLOG-GUARD-... PASS`` line and exits 0; any
    refusal prints one reason to standard error and exits 2.
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        repo = repository_root(arguments.repo)
        if arguments.command == "check":
            digest = check(repo)
            action = "CHECK"
        elif arguments.command == "initialize":
            digest = initialize(repo, arguments.architect_ack)
            action = "INITIALIZE"
        else:
            digest = seal(
                repo,
                previous_digest=arguments.previous_sha256,
                acknowledged=arguments.architect_ack,
            )
            action = "SEAL"
    except (GuardError, OSError, UnicodeError) as error:
        print("backlog guard refused: " + str(error), file=sys.stderr)
        return 2

    print("Backlog guard " + action.casefold() + " passed.")
    print("backlog: " + str(repo / BACKLOG_RELATIVE_PATH))
    print("accepted SHA-256: " + digest)
    print("BACKLOG-GUARD-" + action + " PASS sha256=" + digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
