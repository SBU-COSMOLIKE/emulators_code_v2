#!/usr/bin/env python3
"""Scratch-only witnesses for the daemon's persisted primary worktree.

The production repository's mailbox and Git registry are live infrastructure.
Every runtime arm below creates a disposable Git repository, commits a copy of
the production daemon, and invokes only that copy.  Claude and Codex are never
launched.  The witness is deliberately black-box at the CLI boundary for the
bootstrap/re-exec contract and uses imported helpers only for topology checks.
"""

import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time


AI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_ROOT.parent
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

PRIMARY_NAME = "mailbox-primary"
PRIMARY_BRANCH = "refs/heads/claude/mailbox-primary"
STATE_NAME = ".mailbox-primary-worktree.json"
LOCK_NAME = ".mailbox-primary-worktree.lock"
STATE_KEYS = {"schema", "repository", "name", "path", "branch"}
EXPECTED_MAX_ARCHIVE_FILE_BYTES = 16 * 1024 * 1024
EXPECTED_MAX_ARCHIVE_TOTAL_BYTES = 64 * 1024 * 1024


def run(command, cwd, check=True, timeout=20, env=None):
    """Run one scratch command and return its completed-process record."""
    return subprocess.run(
        list(command), cwd=str(cwd), check=check, timeout=timeout,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)


def git(root, *arguments, **kwargs):
    """Run Git against one disposable repository."""
    return run(["git"] + list(arguments), cwd=root, **kwargs)


def write_exact(path, data):
    """Write exact bytes, creating only parents inside a scratch repository."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(data)


def file_identity(path):
    """Return bytes plus stable identity metadata for preservation checks."""
    state = path.stat()
    return (path.read_bytes(), state.st_dev, state.st_ino, state.st_mode,
            state.st_size, state.st_mtime_ns)


def tree_snapshot(root):
    """Return a content/type snapshot without following scratch symlinks."""
    result = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(root))
        try:
            state = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(state.st_mode):
            result.append((relative, "symlink", os.readlink(str(path))))
        elif stat.S_ISREG(state.st_mode):
            result.append((relative, "file", path.read_bytes()))
        elif stat.S_ISDIR(state.st_mode):
            result.append((relative, "dir", b""))
        elif stat.S_ISFIFO(state.st_mode):
            result.append((relative, "fifo", b""))
        else:
            result.append((relative, "other", state.st_mode))
    return result


def source_with_repo_paths(source):
    """Return production source unchanged for a disposable committed copy."""
    # Kept as a named seam because mutation runs pass their source here too.
    # The daemon derives all repository paths from its scratch __file__.
    return source


@contextlib.contextmanager
def scratch_repository(source=None):
    """Yield a committed minimal repository containing the selected daemon."""
    if source is None:
        source = DAEMON_PATH.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="mailbox-primary-repro-") as tmp:
        root = Path(tmp).resolve()
        write_exact(
            root / "ai" / "tools" / "mailbox_daemon.py",
            source_with_repo_paths(source).encode("utf-8"))
        write_exact(root / "ai" / "notes" / "backlog.md", b"")
        # Production deliberately requires the repository's real .claude
        # directory to pre-exist; only its worktrees child may be bootstrapped.
        write_exact(root / ".claude" / ".keep", b"")
        # Worktrees and their two bootstrap sidecars are runtime state.  The
        # production ignore rule is asserted separately; the scratch rule
        # prevents Git status from recursively inspecting linked checkouts.
        ignore = GITIGNORE_PATH.read_text(encoding="utf-8")
        if ".claude/worktrees/" not in ignore:
            ignore = ignore + "\n.claude/worktrees/\n"
        write_exact(root / ".gitignore", ignore.encode("utf-8"))

        git(root, "init")
        git(root, "symbolic-ref", "HEAD", "refs/heads/main")
        git(root, "config", "user.name", "Primary Worktree Witness")
        git(root, "config", "user.email", "primary@example.invalid")
        git(root, "add", ".gitignore", ".claude/.keep",
            "ai/tools/mailbox_daemon.py", "ai/notes/backlog.md")
        git(root, "commit", "-m", "scratch daemon fixture")
        yield root


def daemon_environment(extra=None):
    """Return a deterministic child environment with bytecode writes off."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # A caller-owned value must never be mistaken for a validated re-exec.
    for name in list(env):
        if "MAILBOX" in name and "PRIMARY" in name and "REEXEC" in name:
            env.pop(name, None)
    if extra:
        env.update(extra)
    return env


def invoke(checkout, arguments, source_path=None, timeout=20, env=None):
    """Invoke a scratch daemon and return rc/stdout/stderr without raising."""
    if source_path is None:
        source_path = checkout / "ai" / "tools" / "mailbox_daemon.py"
    command = [sys.executable, "-B", str(source_path)] + list(arguments)
    try:
        completed = run(
            command, cwd=checkout, check=False, timeout=timeout,
            env=daemon_environment(extra=env))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return 124, stdout, stderr
    return completed.returncode, completed.stdout, completed.stderr


def managed_base(root):
    """Return the repository-shared managed worktree directory."""
    return root / ".claude" / "worktrees"


def state_path(root):
    """Return the exact v1 state path."""
    return managed_base(root) / STATE_NAME


def lock_path(root):
    """Return the exact v1 provisioning-lock path."""
    return managed_base(root) / LOCK_NAME


def default_primary(root):
    """Return the deterministic first-install worktree path."""
    return managed_base(root) / PRIMARY_NAME


def common_directory(root):
    """Return Git's canonical common directory for a scratch checkout."""
    raw = git(root, "rev-parse", "--git-common-dir").stdout.strip()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return str(candidate.resolve())


def load_state(root):
    """Read the exact persisted v1 state object."""
    with state_path(root).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_state_shape(root, expected_path=None, expected_branch=None):
    """Return whether persisted state and Git registry agree exactly."""
    if expected_path is None:
        expected_path = default_primary(root)
    if expected_branch is None:
        expected_branch = PRIMARY_BRANCH
    path = state_path(root)
    if not path.is_file() or path.is_symlink():
        return False
    state = load_state(root)
    if (set(state) != STATE_KEYS or state.get("schema") != 1
            or state.get("repository") != common_directory(root)
            or state.get("name") != expected_path.name
            or state.get("path") != str(expected_path.resolve())
            or state.get("branch") != expected_branch):
        return False
    top = git(expected_path, "rev-parse", "--show-toplevel").stdout.strip()
    branch = git(expected_path, "symbolic-ref", "HEAD").stdout.strip()
    common = common_directory(expected_path)
    return (str(Path(top).resolve()) == str(expected_path.resolve())
            and branch == expected_branch
            and common == common_directory(root))


def worktree_records(root):
    """Return parsed path/branch records from Git's porcelain registry."""
    output = git(root, "worktree", "list", "--porcelain").stdout
    records = []
    current = {}
    for line in output.splitlines() + [""]:
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    return records


def branch_exists(root, branch=PRIMARY_BRANCH):
    """Return whether an exact local branch ref exists."""
    result = git(root, "show-ref", "--verify", "--quiet", branch,
                 check=False)
    return result.returncode == 0


def pending_markdown(worktree):
    """Return root mailbox messages in deterministic order."""
    mailbox = worktree / "ai" / "notes" / "mailbox"
    if not mailbox.is_dir():
        return []
    return sorted(mailbox.glob("*-to-*.md"), key=lambda item: item.name)


def load_scratch_daemon(worktree):
    """Import the daemon copy from one validated scratch worktree."""
    path = worktree / "ai" / "tools" / "mailbox_daemon.py"
    spec = importlib.util.spec_from_file_location(
        "mailbox_daemon_primary_worktree_scratch", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arm_all_live_actions_bootstrap(source=None):
    """Every valid live action provisions, re-execs, and acts in primary."""
    cases = [
        ("once", ["--once"]),
        ("watch", ["--watch", "--cycle", "0"]),
        ("send", ["--send", "fable", "--unit",
                  "Coordinate the committed scratch unit."]),
        ("ping", ["--ping", "fable"]),
    ]
    results = []
    for label, arguments in cases:
        with scratch_repository(source=source) as root:
            rc, stdout, stderr = invoke(root, arguments)
            primary = default_primary(root)
            acted_in_primary = True
            if label in ("send", "ping"):
                acted_in_primary = (
                    len(pending_markdown(primary)) == 1
                    and pending_markdown(root) == [])
            records = worktree_records(root)
            passed = (
                rc == 0 and stderr == ""
                and validate_state_shape(root)
                and len([item for item in records
                         if item.get("worktree")
                         == str(primary.resolve())]) == 1
                and not (primary / ".claude" / "worktrees").exists()
                and acted_in_primary)
            results.append(passed)
            print(label + " bootstrap=" + str(passed)
                  + " rc=" + str(rc)
                  + " output=" + repr(stdout[-180:]))
    return all(results)


def arm_help_dry_run_and_invalid_are_zero_write(source=None):
    """Inspection and invalid invocations create no Git or runtime state."""
    invocations = [
        ("help", ["--help"], 0),
        ("preview", ["--dry-run"], 0),
        ("preview-send", ["--dry-run", "--send", "fable",
                          "--unit", "scratch preview"], 0),
        ("preview-ping", ["--dry-run", "--ping", "fable"], 0),
        ("two-actions", ["--watch", "--once"], 1),
        ("missing-unit", ["--send", "fable"], 1),
        ("unclassified-sol", ["--send", "sol", "--unit", "scratch"], 1),
        ("misplaced-two-role", ["--once", "--skip-redteam"], 1),
        ("bad-model", ["--once", "--architect-model", "bad model"], 1),
        ("bad-claude-context", ["--once", "--claude-context", "0"], 1),
        ("bad-sol-context", ["--once", "--sol-context", "-1"], 1),
    ]
    with scratch_repository(source=source) as root:
        baseline = tree_snapshot(root)
        baseline_registry = worktree_records(root)
        outcomes = []
        for label, arguments, expected_zero in invocations:
            rc, _stdout, _stderr = invoke(root, arguments)
            passed_rc = (rc == 0) if expected_zero == 0 else (rc != 0)
            unchanged = (tree_snapshot(root) == baseline
                         and worktree_records(root) == baseline_registry
                         and not managed_base(root).exists()
                         and not branch_exists(root))
            passed = passed_rc and unchanged
            outcomes.append(passed)
            print(label + " zero-write=" + str(passed)
                  + " rc=" + str(rc))
        return all(outcomes)


def arm_reuse_and_cross_checkout_converge(source=None):
    """A second checkout reuses state and publishes only in primary."""
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_state_shape(root):
            print("initial reuse fixture failed")
            return False
        primary = default_primary(root)
        dirty = primary / "scratch-dirty.txt"
        write_exact(dirty, b"keep this uncommitted byte-for-byte\n")
        dirty_before = file_identity(dirty)
        state_before = file_identity(state_path(root))
        primary_head = git(primary, "rev-parse", "HEAD").stdout.strip()

        other = root.parent / (root.name + "-other-coordinator")
        git(root, "worktree", "add", "-b", "claude/other-coordinator",
            str(other), "main")
        try:
            rc, stdout, stderr = invoke(
                other,
                ["--send", "fable", "--unit",
                 "Audit the committed scratch unit.",
                 "--architect-model", "opus",
                 "--implementer-model", "sonnet"])
            messages = pending_markdown(primary)
            passed = (
                rc == 0 and stderr == ""
                and len(messages) == 1
                and pending_markdown(root) == []
                and pending_markdown(other) == []
                and file_identity(dirty) == dirty_before
                and file_identity(state_path(root)) == state_before
                and git(primary, "rev-parse", "HEAD").stdout.strip()
                == primary_head
                and len(worktree_records(root)) == 3
                and str(primary / "ai" / "notes" / "mailbox")
                in stdout)
            print("cross-checkout convergence=" + str(passed)
                  + " rc=" + str(rc))
            return passed
        finally:
            git(root, "worktree", "remove", "--force", str(other),
                check=False)


def arm_existing_linked_coordinator_is_adopted(source=None):
    """First launch in an existing linked tree preserves its transport."""
    with scratch_repository(source=source) as root:
        existing = managed_base(root) / "existing-coordinator"
        git(root, "worktree", "add", "-b", "claude/existing-coordinator",
            str(existing), "main")
        archived = (existing / "ai" / "notes" / "mailbox" / "done"
                    / "0042-to-fable.md")
        relay = (existing / "ai" / "notes" / "relay"
                 / "20260714-adoption.log")
        write_exact(archived, b"historical mailbox bytes\r\n")
        write_exact(relay, b"historical relay bytes\n")
        archived_before = file_identity(archived)
        relay_before = file_identity(relay)

        rc, _stdout, _stderr = invoke(existing, ["--once"])
        expected_branch = "refs/heads/claude/existing-coordinator"
        adopted = (rc == 0 and validate_state_shape(
            root, expected_path=existing, expected_branch=expected_branch))
        rc_send, _stdout, _stderr = invoke(
            root, ["--send", "fable", "--unit", "Scratch after adoption."])
        queued = pending_markdown(existing)
        passed = (
            adopted and rc_send == 0
            and file_identity(archived) == archived_before
            and file_identity(relay) == relay_before
            and [path.name for path in queued] == ["0043-to-fable.md"]
            and not default_primary(root).exists()
            and len(worktree_records(root)) == 2)
        print("existing coordinator adoption=" + str(passed))
        return passed


def arm_unsafe_existing_coordinator_is_not_adopted(source=None):
    """Pre-ai or redirected stores cannot masquerade as safe adoption."""
    outcomes = []
    with scratch_repository(source=source) as root:
        existing = managed_base(root) / "pre-ai-coordinator"
        git(root, "worktree", "add", "-b", "claude/pre-ai-coordinator",
            str(existing), "main")
        pending, pending_before = transport_case(
            existing, "notes/mailbox/0160-to-fable.md",
            b"legacy active queue bytes\n")
        rc, stdout, _stderr = invoke(existing, ["--once"])
        refused = (
            rc != 0 and file_identity(pending) == pending_before
            and not state_path(root).exists()
            and not default_primary(root).exists() and not branch_exists(root)
            and "legacy pre-ai" in stdout.lower())
        outcomes.append(refused)
        print("pre-ai coordinator adoption refusal=" + str(refused))

    with scratch_repository(source=source) as root:
        existing = managed_base(root) / "redirected-coordinator"
        git(root, "worktree", "add", "-b", "claude/redirected-coordinator",
            str(existing), "main")
        target = existing / "redirect-target"
        target.mkdir()
        mailbox = existing / "ai" / "notes" / "mailbox"
        mailbox.parent.mkdir(parents=True, exist_ok=True)
        mailbox.symlink_to(target, target_is_directory=True)
        link_before = os.readlink(str(mailbox))
        rc, stdout, _stderr = invoke(existing, ["--once"])
        refused = (
            rc != 0 and mailbox.is_symlink()
            and os.readlink(str(mailbox)) == link_before
            and not state_path(root).exists()
            and not default_primary(root).exists() and not branch_exists(root)
            and ("redirected" in stdout.lower()
                 or "irregular" in stdout.lower()))
        outcomes.append(refused)
        print("redirected coordinator adoption refusal=" + str(refused))
    return all(outcomes)


def transport_case(root, relative, body):
    """Create one ignored transport artifact and return its exact identity."""
    path = root / relative
    write_exact(path, body)
    return path, file_identity(path)


def arm_legacy_transport_refuses_new_primary(source=None):
    """Active or ambiguous legacy queue state is never bridged by guessing."""
    cases = [
        ("pending", "ai/notes/mailbox/0007-to-opus.md"),
        ("inflight", "ai/notes/mailbox/inflight/0007-to-opus.md"),
        ("failed", "ai/notes/mailbox/failed/0007-to-opus.md"),
        ("pre-ai-pending", "notes/mailbox/0160-to-fable.md"),
        ("pre-ai-relay", "notes/relay/legacy-dispatch.log"),
    ]
    outcomes = []
    for label, relative in cases:
        with scratch_repository(source=source) as root:
            artifact, before = transport_case(
                root, relative, (label + " evidence\n").encode("utf-8"))
            rc, stdout, _stderr = invoke(root, ["--once"])
            passed = (
                rc != 0 and file_identity(artifact) == before
                and not state_path(root).exists()
                and not default_primary(root).exists()
                and not branch_exists(root)
                and len(worktree_records(root)) == 1
                and ("mailbox" in stdout.lower()
                     or "relay" in stdout.lower()
                     or "transport" in stdout.lower()))
            outcomes.append(passed)
            print(label + " transport refusal=" + str(passed)
                  + " rc=" + str(rc))
    with scratch_repository(source=source) as root:
        mailbox = root / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        sequence = mailbox / ".sequence.lock"
        handle = sequence.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            before = sequence.stat()
            rc, stdout, _stderr = invoke(root, ["--once"])
            after = sequence.stat()
            passed = (
                rc != 0 and not state_path(root).exists()
                and not default_primary(root).exists()
                and not branch_exists(root)
                and (before.st_dev, before.st_ino, before.st_size)
                == (after.st_dev, after.st_ino, after.st_size)
                and "live sender" in stdout.lower())
            outcomes.append(passed)
            print("held sequence transport refusal=" + str(passed)
                  + " rc=" + str(rc))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    return all(outcomes)


def arm_archived_main_transport_is_bridged(source=None):
    """A unique completed main archive is copied exactly before publication."""
    with scratch_repository(source=source) as root:
        archived, archived_before = transport_case(
            root, "ai/notes/mailbox/done/0042-to-fable.md",
            b"completed legacy mailbox bytes\r\n")
        relay, relay_before = transport_case(
            root, "ai/notes/relay/20260714-completed.log",
            b"completed legacy relay bytes\n")

        rc, stdout, stderr = invoke(root, ["--once"])
        primary = default_primary(root)
        copied_archive = (primary / "ai" / "notes" / "mailbox" / "done"
                          / archived.name)
        copied_relay = (primary / "ai" / "notes" / "relay" / relay.name)
        bridged = (
            rc == 0 and stderr == "" and validate_state_shape(root)
            and copied_archive.read_bytes() == archived.read_bytes()
            and copied_relay.read_bytes() == relay.read_bytes()
            and file_identity(archived) == archived_before
            and file_identity(relay) == relay_before
            and "bridged archived main-checkout" in stdout)

        rc_send, _stdout, _stderr = invoke(
            root, ["--send", "fable", "--unit",
                   "Audit the first post-bridge scratch unit."])
        queued = pending_markdown(primary)
        passed = (
            bridged and rc_send == 0
            and [path.name for path in queued] == ["0043-to-fable.md"]
            and pending_markdown(root) == [])
        print("archived main bridge=" + str(passed))
        return passed


def arm_interrupted_archive_bridge_is_exactly_resumable(source=None):
    """An exact partial copy resumes; any extra active transport refuses."""
    outcomes = []
    with scratch_repository(source=source) as root:
        archived, archived_before = transport_case(
            root, "ai/notes/mailbox/done/0042-to-fable.md",
            b"completed legacy mailbox bytes\r\n")
        relay, relay_before = transport_case(
            root, "ai/notes/relay/20260714-completed.log",
            b"completed legacy relay bytes\n")
        managed_base(root).mkdir(parents=True)
        primary = default_primary(root)
        git(root, "worktree", "add", "-b", "claude/mailbox-primary",
            str(primary), "main")
        partial = (primary / "ai" / "notes" / "mailbox" / "done"
                   / archived.name)
        write_exact(partial, archived.read_bytes())
        partial_before = file_identity(partial)

        rc, stdout, stderr = invoke(root, ["--once"])
        copied_relay = primary / "ai" / "notes" / "relay" / relay.name
        resumed = (
            rc == 0 and stderr == "" and validate_state_shape(root)
            and file_identity(partial) == partial_before
            and copied_relay.read_bytes() == relay.read_bytes()
            and file_identity(archived) == archived_before
            and file_identity(relay) == relay_before
            and "bridged archived main-checkout" in stdout)
        outcomes.append(resumed)
        print("exact partial archive resume=" + str(resumed))

    with scratch_repository(source=source) as root:
        archived, archived_before = transport_case(
            root, "ai/notes/mailbox/done/0042-to-fable.md",
            b"completed legacy mailbox bytes\n")
        managed_base(root).mkdir(parents=True)
        primary = default_primary(root)
        git(root, "worktree", "add", "-b", "claude/mailbox-primary",
            str(primary), "main")
        active, active_before = transport_case(
            primary, "ai/notes/mailbox/0042-to-opus.md",
            b"uncompleted primary bytes\n")

        rc, stdout, _stderr = invoke(root, ["--once"])
        refused = (
            rc != 0 and not state_path(root).exists()
            and file_identity(archived) == archived_before
            and file_identity(active) == active_before
            and primary.is_dir() and branch_exists(root)
            and len(worktree_records(root)) == 2
            and ("transport" in stdout.lower()
                 or "archive" in stdout.lower()
                 or "coordination" in stdout.lower()))
        outcomes.append(refused)
        print("active partial archive refusal=" + str(refused)
              + " rc=" + str(rc))
    return all(outcomes)


def arm_archive_bridge_bounds_and_sequences_refuse(source=None):
    """Oversized archives and duplicate numeric identities fail closed."""
    outcomes = []

    with scratch_repository(source=source) as root:
        oversized = root / "ai" / "notes" / "relay" / "oversized.log"
        oversized.parent.mkdir(parents=True, exist_ok=True)
        with oversized.open("wb") as stream:
            stream.truncate(EXPECTED_MAX_ARCHIVE_FILE_BYTES + 1)
        before = oversized.stat()
        rc, _stdout, _stderr = invoke(root, ["--once"])
        after = oversized.stat()
        refused = (
            rc != 0 and not state_path(root).exists()
            and not default_primary(root).exists() and not branch_exists(root)
            and (before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))
        outcomes.append(refused)
        print("per-file archive bound=" + str(refused))

    with scratch_repository(source=source) as root:
        relay = root / "ai" / "notes" / "relay"
        relay.mkdir(parents=True, exist_ok=True)
        each = EXPECTED_MAX_ARCHIVE_FILE_BYTES // 2
        count = EXPECTED_MAX_ARCHIVE_TOTAL_BYTES // each + 1
        identities = []
        for index in range(count):
            path = relay / ("aggregate-%02d.log" % index)
            with path.open("wb") as stream:
                stream.truncate(each)
            state = path.stat()
            identities.append((path, state.st_dev, state.st_ino,
                               state.st_size, state.st_mtime_ns))
        rc, _stdout, _stderr = invoke(root, ["--once"])
        preserved = all(
            (path.stat().st_dev, path.stat().st_ino, path.stat().st_size,
             path.stat().st_mtime_ns) == tuple(identity)
            for path, *identity in identities)
        refused = (
            rc != 0 and preserved and not state_path(root).exists()
            and not default_primary(root).exists() and not branch_exists(root))
        outcomes.append(refused)
        print("aggregate archive bound=" + str(refused))

    with scratch_repository(source=source) as root:
        first, first_before = transport_case(
            root, "ai/notes/mailbox/done/0042-to-fable.md",
            b"first completed route\n")
        second, second_before = transport_case(
            root, "ai/notes/mailbox/done/0042-to-opus.md",
            b"duplicate completed route\n")
        rc, _stdout, _stderr = invoke(root, ["--once"])
        refused = (
            rc != 0 and file_identity(first) == first_before
            and file_identity(second) == second_before
            and not state_path(root).exists()
            and not default_primary(root).exists() and not branch_exists(root))
        outcomes.append(refused)
        print("duplicate archive sequence=" + str(refused))

    with scratch_repository(source=source) as root:
        archived, archived_before = transport_case(
            root, "ai/notes/mailbox/done/0042-to-fable.md",
            b"completed route\n")
        unknown, unknown_before = transport_case(
            root, "ai/notes/mailbox/.unknown.lock",
            b"unrecognized transport sentinel\n")
        manifest_refused = (
            load_scratch_daemon(root)._archived_transport_manifest(
                str(root)) is None)
        rc, _stdout, _stderr = invoke(root, ["--once"])
        refused = (
            manifest_refused and rc != 0
            and file_identity(archived) == archived_before
            and file_identity(unknown) == unknown_before
            and not state_path(root).exists()
            and not default_primary(root).exists() and not branch_exists(root))
        outcomes.append(refused)
        print("unknown mailbox lock=" + str(refused))

    with scratch_repository(source=source) as root:
        unknown, unknown_before = transport_case(
            root, "ai/notes/mailbox/.unknown.lock",
            b"only unrecognized transport sentinel\n")
        rc, _stdout, _stderr = invoke(root, ["--once"])
        refused = (
            rc != 0 and file_identity(unknown) == unknown_before
            and not state_path(root).exists()
            and not default_primary(root).exists() and not branch_exists(root))
        outcomes.append(refused)
        print("unknown-only mailbox lock=" + str(refused))

    return all(outcomes)


def arm_concurrent_bootstrap_obeys_global_lock(source=None):
    """Two first launches wait on one lock and publish one primary state."""
    with scratch_repository(source=source) as root:
        managed_base(root).mkdir(parents=True)
        handle = lock_path(root).open("a+")
        first = None
        second = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            command = [sys.executable, "-B",
                       str(root / "ai" / "tools" / "mailbox_daemon.py"),
                       "--once"]
            env = daemon_environment()
            first = subprocess.Popen(
                command, cwd=str(root), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
            second = subprocess.Popen(
                command, cwd=str(root), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
            time.sleep(0.35)
            waited = (first.poll() is None and second.poll() is None
                      and not state_path(root).exists()
                      and not default_primary(root).exists())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            first_out, first_err = first.communicate(timeout=20)
            second_out, second_err = second.communicate(timeout=20)
        except BaseException:
            for process in (first, second):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
            raise
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()

        returncodes = [first.returncode, second.returncode]
        permitted_overlap_refusal = all(code in (0, 1) for code in returncodes)
        if 1 in returncodes:
            combined = first_out + first_err + second_out + second_err
            permitted_overlap_refusal = (
                permitted_overlap_refusal
                and "another dispatch loop" in combined.lower())
        records = worktree_records(root)
        passed = (
            waited and permitted_overlap_refusal and 0 in returncodes
            and validate_state_shape(root)
            and len([item for item in records
                     if item.get("worktree")
                     == str(default_primary(root).resolve())]) == 1)
        print("concurrent bootstrap=" + str(passed)
              + " returncodes=" + repr(returncodes))
        return passed


def arm_final_publication_fences_late_sender(source=None):
    """A sender admitted after the last scan cannot be stranded in main."""
    with scratch_repository(source=source) as root:
        daemon = load_scratch_daemon(root)
        original_evidence = daemon.coordination_transport_evidence
        root_scans = [0]
        holder = [None]

        def evidence_then_admit_sender(worktree):
            result = original_evidence(worktree)
            if Path(worktree).resolve() == root.resolve():
                root_scans[0] += 1
                if root_scans[0] == 2:
                    mailbox = root / "ai" / "notes" / "mailbox"
                    mailbox.mkdir(parents=True, exist_ok=True)
                    sequence = mailbox / ".sequence.lock"
                    program = (
                        "import fcntl,sys,time; "
                        "f=open(sys.argv[1],'a+'); "
                        "fcntl.flock(f.fileno(),fcntl.LOCK_EX); "
                        "print('ready',flush=True); time.sleep(30)")
                    holder[0] = subprocess.Popen(
                        [sys.executable, "-c", program, str(sequence)],
                        cwd=str(root), stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True)
                    if holder[0].stdout.readline().strip() != "ready":
                        raise RuntimeError("late-sender fixture did not lock")
            return result

        daemon.coordination_transport_evidence = evidence_then_admit_sender
        error = None
        try:
            daemon.provision_or_adopt_primary(
                repository_root=str(root), current_worktree=str(root))
        except BaseException as exc:
            error = exc
        finally:
            if holder[0] is not None:
                holder[0].terminate()
                holder[0].wait(timeout=5)
            daemon.coordination_transport_evidence = original_evidence
        passed = (
            isinstance(error, daemon.PrimaryWorktreeError)
            and "legacy transport is live" in str(error).lower()
            and root_scans[0] >= 2 and not state_path(root).exists()
            and default_primary(root).is_dir() and branch_exists(root))
        print("late sender publication fence=" + str(passed)
              + " error=" + repr(str(error)))
        return passed


def arm_concurrent_managed_root_winner_is_accepted(source=None):
    """A first-run loser accepts the ordinary directory a peer just made."""
    with scratch_repository(source=source) as root:
        daemon = load_scratch_daemon(root)
        target = managed_base(root)
        original_mkdir = daemon.os.mkdir

        def peer_wins_mkdir(path, mode=0o777):
            original_mkdir(path, mode)
            raise FileExistsError(path)

        daemon.os.mkdir = peer_wins_mkdir
        error = None
        try:
            daemon._plain_directory(
                path=str(target), label="managed worktree directory",
                create=True)
        except BaseException as exc:
            error = exc
        finally:
            daemon.os.mkdir = original_mkdir
        passed = error is None and target.is_dir() and not target.is_symlink()
        print("concurrent managed-root winner=" + str(passed))
        return passed


def arm_corrupt_and_redirected_state_fail_closed(source=None):
    """Invalid state is preserved and never falls back to caller mailbox."""
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_state_shape(root):
            return False
        primary = default_primary(root)
        original = state_path(root).read_bytes()
        state = json.loads(original.decode("utf-8"))
        variants = []
        variants.append(("invalid-json", b"{ definitely not json\n"))
        duplicate = (
            "{\"schema\":1,\"schema\":1,\"repository\":"
            + json.dumps(state["repository"])
            + ",\"name\":\"mailbox-primary\",\"path\":"
            + json.dumps(state["path"])
            + ",\"branch\":\"refs/heads/claude/mailbox-primary\"}\n")
        variants.append(("duplicate-key", duplicate.encode("utf-8")))
        unknown = dict(state)
        unknown["model"] = "opus"
        variants.append(("unknown-key",
                         (json.dumps(unknown) + "\n").encode("utf-8")))
        foreign = dict(state)
        foreign["repository"] = str(root / "foreign.git")
        variants.append(("foreign-repository",
                         (json.dumps(foreign) + "\n").encode("utf-8")))
        escaped = dict(state)
        escaped_path = root.parent / (root.name + "-escaped-primary")
        escaped["path"] = str(escaped_path)
        escaped["name"] = escaped_path.name
        variants.append(("escaped-path",
                         (json.dumps(escaped) + "\n").encode("utf-8")))
        # Whitespace after a complete JSON value remains valid JSON.  The
        # refusal therefore proves the byte bound, not merely parser failure.
        variants.append(("oversized", original.rstrip(b"\n")
                         + b" " * (2 * 1024 * 1024) + b"\n"))

        outcomes = []
        for label, payload in variants:
            write_exact(state_path(root), payload)
            before = file_identity(state_path(root))
            rc, _stdout, _stderr = invoke(
                root, ["--send", "fable", "--unit", "must not queue"])
            passed = (
                rc != 0 and file_identity(state_path(root)) == before
                and pending_markdown(root) == []
                and pending_markdown(primary) == [])
            outcomes.append(passed)
            print(label + " refused=" + str(passed))

        external = root / "external-state-sentinel"
        write_exact(external, original)
        state_path(root).unlink()
        os.symlink(str(external), str(state_path(root)))
        external_before = file_identity(external)
        rc, _stdout, _stderr = invoke(root, ["--once"])
        symlink_passed = (
            rc != 0 and state_path(root).is_symlink()
            and file_identity(external) == external_before)
        outcomes.append(symlink_passed)
        print("symlink state refused=" + str(symlink_passed))
        state_path(root).unlink()

        if hasattr(os, "mkfifo"):
            os.mkfifo(str(state_path(root)), 0o600)
            rc, _stdout, _stderr = invoke(root, ["--once"], timeout=4)
            fifo_passed = rc != 124 and rc != 0
            outcomes.append(fifo_passed)
            print("fifo state refused=" + str(fifo_passed))
            state_path(root).unlink()
        write_exact(state_path(root), original)
        return all(outcomes)


def arm_git_identity_and_collisions_fail_closed(source=None):
    """Wrong branch and first-run collisions are preserved, never repaired."""
    results = []
    with scratch_repository(source=source) as root:
        collision = default_primary(root)
        write_exact(collision / "sentinel", b"ordinary directory\n")
        sentinel = file_identity(collision / "sentinel")
        rc, _stdout, _stderr = invoke(root, ["--once"])
        passed = (
            rc != 0 and file_identity(collision / "sentinel") == sentinel
            and not state_path(root).exists() and not branch_exists(root)
            and len(worktree_records(root)) == 1)
        results.append(passed)
        print("path collision refused=" + str(passed))

    with scratch_repository(source=source) as root:
        git(root, "branch", "claude/mailbox-primary", "main")
        sha = git(root, "rev-parse", PRIMARY_BRANCH).stdout.strip()
        rc, _stdout, _stderr = invoke(root, ["--once"])
        passed = (
            rc != 0 and not state_path(root).exists()
            and not default_primary(root).exists()
            and git(root, "rev-parse", PRIMARY_BRANCH).stdout.strip() == sha
            and len(worktree_records(root)) == 1)
        results.append(passed)
        print("branch collision refused=" + str(passed))

    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0:
            return False
        state_before = file_identity(state_path(root))
        primary = default_primary(root)
        git(primary, "switch", "-c", "claude/wrong-primary")
        write_exact(primary / "wrong-branch-sentinel", b"preserve me\n")
        sentinel = file_identity(primary / "wrong-branch-sentinel")
        rc, _stdout, _stderr = invoke(root, ["--once"])
        passed = (
            rc != 0 and file_identity(state_path(root)) == state_before
            and file_identity(primary / "wrong-branch-sentinel") == sentinel
            and git(primary, "symbolic-ref", "HEAD").stdout.strip()
            == "refs/heads/claude/wrong-primary")
        results.append(passed)
        print("wrong branch refused=" + str(passed))
    return all(results)


def arm_route_topology_remains_role_based(source=None):
    """Claude shares primary while Sol remains the repository-root lane."""
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(
            root, ["--watch", "--cycle", "0", "--skip-redteam",
                   "--architect-model", "opus",
                   "--implementer-model", "sonnet"])
        if rc != 0 or not validate_state_shape(root):
            return False
        primary = default_primary(root)
        daemon = load_scratch_daemon(primary)
        command = daemon.AGENT_COMMANDS["sol"]
        cd_value = command[command.index("--cd") + 1]
        state = load_state(root)
        passed = (
            daemon.AGENT_CWD["fable"] == str(primary)
            and daemon.AGENT_CWD["opus"] == str(primary)
            and daemon.AGENT_CWD["sol"] == str(root)
            and cd_value == str(root)
            and state["name"] == PRIMARY_NAME
            and "opus" not in state.values()
            and "sonnet" not in state.values())
        print("role topology=" + str(passed)
              + " cwd=" + repr(daemon.AGENT_CWD))
        return passed


def replace_exact(source, old, new):
    """Return a one-site source mutant or None when its anchor drifted."""
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def mutation_cases(source):
    """Build exact source mutants for the binding primary-tree contracts."""
    cases = []

    def add(label, old, new, probe):
        mutant = replace_exact(source, old, new)
        cases.append((label, mutant, probe))

    # These anchors intentionally name semantic operations rather than line
    # numbers.  An unarmed mutation is a witness failure, not a silent skip.
    add("dry-run provisions",
        "ensure_primary_execution(\n"
        "                live_action=bool(primary_actions), "
        "dry_run=args.dry_run)",
        "ensure_primary_execution(\n"
        "                live_action=True, dry_run=False)",
        arm_help_dry_run_and_invalid_are_zero_write)
    add("Claude routes split",
        '    "opus": WORKTREE,',
        '    "opus": REPO_ROOT,',
        arm_route_topology_remains_role_based)
    add("Sol moved into primary",
        '    "sol": REPO_ROOT,',
        '    "sol": WORKTREE,',
        arm_route_topology_remains_role_based)
    add("duplicate keys accepted",
        "object_pairs_hook=_duplicate_key_refusal",
        "object_pairs_hook=dict",
        arm_corrupt_and_redirected_state_fail_closed)
    add("foreign repository accepted",
        '    if state["repository"] != repository:\n',
        "    if False:\n",
        arm_corrupt_and_redirected_state_fail_closed)
    add("transport evidence ignored",
        '        reasons = coordination_transport_evidence('
        'worktree=record["path"])',
        "        reasons = []",
        arm_legacy_transport_refuses_new_primary)
    add("live sender lock ignored",
        '        (".sequence.lock", "live sender or sequence lock is held"),\n',
        "",
        arm_legacy_transport_refuses_new_primary)
    add("unsafe current coordinator adopted",
        "                and set(evidence[0][1]).issubset(\n"
        "                    CURRENT_ADOPTION_SAFE_REASONS)\n",
        "",
        arm_unsafe_existing_coordinator_is_not_adopted)
    add("bootstrap lock dropped",
        "        fcntl.flock(descriptor, fcntl.LOCK_EX)\n",
        "        pass  # mutation: provisioning is no longer serialized\n",
        arm_concurrent_bootstrap_obeys_global_lock)
    add("concurrent managed-root winner refused",
        "        except FileExistsError:\n"
        "            # Two clean-clone first runs can both observe the absent "
        "managed\n"
        "            # root before either opens the shared bootstrap lock. "
        "The winner's\n"
        "            # ordinary directory is accepted by the lstat proof "
        "below.\n"
        "            pass\n",
        "        except FileExistsError:\n"
        "            raise\n",
        arm_concurrent_managed_root_winner_is_accepted)
    add("fresh publication fence dropped",
        "    return _publish_primary_record(\n"
        "        record=created, repository_root=repository_root,\n"
        "        bridge_main=bridge_main, fence_empty_main=not bridge_main)",
        "    return _publish_primary_record(\n"
        "        record=created, repository_root=repository_root,\n"
        "        bridge_main=bridge_main, fence_empty_main=False)",
        arm_final_publication_fences_late_sender)
    add("saved path containment dropped",
        "    if os.path.dirname(stored_path) != managed_root:\n",
        "    if False:\n",
        arm_corrupt_and_redirected_state_fail_closed)
    add("registered branch identity skipped",
        '    _validate_primary_record(record=record, branch=state["branch"],\n'
        "                             repository_root=repository_root)\n",
        "    pass  # mutation: registry identity is trusted without proof\n",
        arm_git_identity_and_collisions_fail_closed)
    add("state byte bound disabled",
        "MAX_PRIMARY_STATE_BYTES = 16384",
        "MAX_PRIMARY_STATE_BYTES = 4194304",
        arm_corrupt_and_redirected_state_fail_closed)
    add("archive file bound disabled",
        "MAX_PRIMARY_ARCHIVE_FILE_BYTES = 16 * 1024 * 1024",
        "MAX_PRIMARY_ARCHIVE_FILE_BYTES = 32 * 1024 * 1024",
        arm_archive_bridge_bounds_and_sequences_refuse)
    add("duplicate archive sequence accepted",
        "                        if sequence in mailbox_sequences:\n",
        "                        if False:\n",
        arm_archive_bridge_bounds_and_sequences_refuse)
    add("unknown mailbox content ignored",
        "                        found = (reason_prefix\n"
        '                                 + "unrecognized mailbox entry exists")\n',
        "                        continue  # mutation: unknown content lost\n",
        arm_archive_bridge_bounds_and_sequences_refuse)
    add("unknown archive locks wildcarded",
        "                        if (len(parts) == 1\n"
        "                                and name in "
        "PRIMARY_ARCHIVE_RUNTIME_LOCKS):\n",
        "                        if (len(parts) == 1 and name.startswith('.')\n"
        "                                and name.endswith('.lock')):\n",
        arm_archive_bridge_bounds_and_sequences_refuse)
    add("re-exec dropped",
        "        os.execv(sys.executable,\n"
        "                 [sys.executable, daemon] + list(sys.argv[1:]))",
        "        return state  # mutation: action continues in caller",
        arm_all_live_actions_bootstrap)
    return cases


def run_mutations(source):
    """Return whether every armed source mutation turns its witness red."""
    outcomes = []
    for label, mutant, probe in mutation_cases(source):
        if mutant is None:
            print("FAIL mutation not armed: " + label)
            outcomes.append(False)
            continue
        try:
            survived = probe(source=mutant)
        except BaseException as exc:
            survived = False
            print("  mutation raised " + type(exc).__name__ + ": " + str(exc))
        killed = not survived
        outcomes.append(killed)
        print(("PASS " if killed else "FAIL ") + "mutation killed: " + label)
    return all(outcomes)


def main():
    """Run every scratch runtime arm and targeted source mutation."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    runtime = [
        ("all live actions", arm_all_live_actions_bootstrap),
        ("zero-write inspection", arm_help_dry_run_and_invalid_are_zero_write),
        ("cross-checkout reuse", arm_reuse_and_cross_checkout_converge),
        ("existing coordinator adoption",
         arm_existing_linked_coordinator_is_adopted),
        ("unsafe coordinator refusal",
         arm_unsafe_existing_coordinator_is_not_adopted),
        ("legacy transport refusal", arm_legacy_transport_refuses_new_primary),
        ("archived main bridge", arm_archived_main_transport_is_bridged),
        ("interrupted archive bridge",
         arm_interrupted_archive_bridge_is_exactly_resumable),
        ("archive bridge bounds",
         arm_archive_bridge_bounds_and_sequences_refuse),
        ("concurrent bootstrap", arm_concurrent_bootstrap_obeys_global_lock),
        ("late sender publication fence",
         arm_final_publication_fences_late_sender),
        ("concurrent managed root",
         arm_concurrent_managed_root_winner_is_accepted),
        ("corrupt state refusal", arm_corrupt_and_redirected_state_fail_closed),
        ("Git identity refusal", arm_git_identity_and_collisions_fail_closed),
        ("role topology", arm_route_topology_remains_role_based),
    ]
    passed = 0
    for label, probe in runtime:
        try:
            outcome = probe(source=source)
        except BaseException as exc:
            outcome = False
            print(label + " raised " + type(exc).__name__ + ": " + str(exc))
        if outcome:
            passed += 1
        print(("PASS " if outcome else "FAIL ") + label)
    print("runtime: " + str(passed) + "/" + str(len(runtime)))

    mutations_passed = run_mutations(source=source)
    if passed != len(runtime) or not mutations_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
