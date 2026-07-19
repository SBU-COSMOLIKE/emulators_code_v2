#!/usr/bin/env python3
"""Detect accidental edits to the Architect-owned local backlog.

``ai/notes/backlog.md`` is intentionally not committed to Git, so Git cannot
provide the fixed reference used by ``permanent_note_guard.py``.  This tool
instead keeps one small, ignored state file beside the backlog.  The state
records the SHA-256 fingerprint of the exact backlog bytes last accepted by
the Architect.

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
    """The backlog or its saved fingerprint cannot be trusted."""


def repository_root(repo_argument=None):
    """Return the checkout named by the caller or containing this tool."""
    if repo_argument is None:
        repo = Path(__file__).resolve().parents[2]
    else:
        repo = Path(repo_argument).expanduser().resolve()
    if not repo.is_dir():
        raise GuardError("the named repository folder does not exist")
    return repo


def _stat_signature(metadata):
    """Return filesystem facts that change when one file is replaced."""
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
    """Refuse a missing directory or one redirected through a link."""
    try:
        metadata = path.lstat()
    except OSError as error:
        raise GuardError("cannot inspect " + label + ": " + str(error))
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise GuardError(label + " is not a real directory")


def guard_paths(repo):
    """Return validated paths for the backlog, state, and write lock."""
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
    """Read one stable regular file without following a final symlink."""
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
    """Calculate the canonical lowercase SHA-256 spelling."""
    return hashlib.sha256(data).hexdigest()


def _state_document(digest, previous_digest=None):
    """Build the only accepted state representation."""
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
    """Encode state deterministically so incidental state edits are visible."""
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


def _reject_duplicate_json_pairs(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise GuardError("backlog guard state repeats the field " + key)
        document[key] = value
    return document


def _parse_state(data):
    """Parse and validate the complete saved state file."""
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
    data = _read_regular_bytes(
        state_path, "backlog guard state", MAX_STATE_BYTES)
    return data, _parse_state(data)


def _require_architect(acknowledged):
    """Allow writes only from the mailbox Architect or explicit manual use."""
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
    """Hold one crash-released lock on the persistent lock file."""

    def __init__(self, path):
        self.path = path
        self.descriptor = None

    def __enter__(self):
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
        if self.descriptor is not None:
            try:
                fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self.descriptor)
                self.descriptor = None
        return False


def _atomic_write_state(state_path, document):
    """Save canonical state beside the backlog and leave no partial file."""
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
    """Return the accepted digest after comparing exact current bytes."""
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
    """Create first state for an existing backlog."""
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
    """Replace accepted state after proving which state was checked first."""
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
    parser = argparse.ArgumentParser(
        description=(
            "Detect accidental changes to the Architect-owned local backlog."))
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
