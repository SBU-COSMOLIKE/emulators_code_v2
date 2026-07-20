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
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import tempfile
import time


AI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
HANDOFF_CONTRACT_PATH = AI_ROOT / "tools" / "handoff_contract.py"
PERMANENT_NOTE_GUARD_PATH = AI_ROOT / "tools" / "permanent_note_guard.py"
ROLE_CONTRACT_TOOL_PATH = AI_ROOT / "tools" / "role_contract.py"
ROLE_CONTRACT_PATH = AI_ROOT / "notes" / "role-contract.yaml"
FAILURE_MODES_PATH = AI_ROOT / "notes" / "implementer-failure-modes.yaml"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"
OTHER_TRUSTED_TOOLS = (
    "backlog_bundle.py",
    "backlog_guard.py",
    "handoff_router.py",
    "implementer_checkpoint_hook.py",
)
PERMANENT_NOTES = (
    "MEMORY.md",
    "project-and-history.md",
    "conventions-and-workflow.md",
    "python-changes-go-no-go.md",
    "models-and-designs.md",
    "training-stack.md",
    "artifacts-inference-warmstart.md",
    "data-generation-and-cuts.md",
    "families-background-mps.md",
    "families-scalar-cmb.md",
    "readme-go-no-go.md",
)

PRIMARY_NAME = "mailbox-primary"
PRIMARY_BRANCH = "refs/heads/claude/mailbox-primary"
STATE_NAME = ".mailbox-primary-worktree.json"
LOCK_NAME = ".mailbox-primary-worktree.lock"
PRIMARY_STATE_SCHEMA = 3
PRIMARY_TOPOLOGY = "separate-role-worktrees-v1"
PRIMARY_STATE_KEYS = {
    "schema", "repository", "name", "path", "branch", "topology"}
IMPLEMENTER_NAME = "mailbox-implementer"
IMPLEMENTER_BRANCH = "refs/heads/claude/mailbox-implementer"
IMPLEMENTER_STATE_NAME = ".mailbox-implementer-worktree.json"
IMPLEMENTER_STATE_SCHEMA = 1
IMPLEMENTER_STATE_KEYS = {"schema", "repository", "name", "path", "branch"}
SOL_NAME = "mailbox-sol"
SOL_BRANCH = "refs/heads/codex/mailbox-sol"
SOL_STATE_NAME = ".mailbox-sol-worktree.json"
SOL_STATE_SCHEMA = 1
SOL_STATE_KEYS = {"schema", "repository", "name", "path", "branch"}
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


def git_bytes(root, *arguments, input_bytes=None, check=True, timeout=20):
    """Run scratch Git without decoding its input or output bytes."""
    return subprocess.run(
        ["git"] + list(arguments), cwd=str(root), check=check,
        timeout=timeout, input=input_bytes, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)


def write_exact(path, data):
    """Write exact bytes, creating only parents inside a scratch repository."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(data)


def seal_backlog(primary):
    """Save the Architect-authorized digest for one scratch backlog."""
    backlog = primary / "ai" / "notes" / "backlog.md"
    state = primary / "ai" / "notes" / ".backlog-guard.json"
    payload = {
        "backlog": "ai/notes/backlog.md",
        "sha256": hashlib.sha256(backlog.read_bytes()).hexdigest(),
        "version": 1,
    }
    write_exact(
        state,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"))


def close_backlog_ticket(primary, anchor):
    """Move one scratch ticket below Closed and seal the result."""
    backlog = primary / "ai" / "notes" / "backlog.md"
    lines = backlog.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines
             if not (line.startswith("- OPEN ")
                     and line.endswith("](#" + anchor + ")"))]
    marker = '<a id="' + anchor + '"></a>'
    starts = [index for index, line in enumerate(lines) if line == marker]
    if len(starts) != 1:
        raise AssertionError("scratch backlog lacks one anchor: " + anchor)
    start = starts[0]
    end = next((index for index in range(start + 1, len(lines))
                if lines[index].startswith('<a id="')
                or lines[index] == "# Closed tickets"), len(lines))
    del lines[start:end]
    if "# Closed tickets" not in lines:
        lines.extend(["", "# Closed tickets"])
    lines.extend([
        "", marker, "## Closed scratch ticket", "",
        "### High-level summary", "", "The scratch repair is complete.",
        "",
        "### Current status", "", "**CLOSED.** Scratch repair accepted.",
        "", "### What is already fixed", "", "The repair is present.",
        "", "### What is missing", "", "Nothing for this ticket.",
    ])
    backlog.write_text(
        "\n".join(lines).strip() + "\n", encoding="utf-8", newline="")
    seal_backlog(primary=primary)


def create_empty_sealed_backlog(primary):
    """Create the valid empty local ledger required before Sol admission."""
    write_exact(primary / "ai" / "notes" / "backlog.md", b"")
    seal_backlog(primary=primary)


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
        # Production deliberately requires the repository's real .claude
        # directory to pre-exist; only its worktrees child may be bootstrapped.
        write_exact(root / ".claude" / ".keep", b"")
        # Sol receives these absolute, read-only instruction files from the
        # validated Claude primary. They must be committed regular files so
        # bootstrap proves the same authority boundary as production.
        write_exact(
            root / ".claude" / "FABLE_ROLE.md",
            b"# Scratch Architect role\n\nAudit the immutable candidate.\n")
        write_exact(
            root / ".claude" / "OPUS_ROLE.md",
            b"# Scratch Implementer role\n\nFollow the validated directive.\n")
        write_exact(
            root / ".codex" / "REDTEAM_ROLE.md",
            b"# Scratch Red Team role\n\nReview only the named change.\n")
        write_exact(
            root / "ai" / "tools" / "handoff_contract.py",
            HANDOFF_CONTRACT_PATH.read_bytes())
        write_exact(
            root / "ai" / "tools" / "ticket_change_guard.py",
            b"#!/usr/bin/env python3\n# Scratch ticket size guard.\n")
        write_exact(
            root / "ai" / "tools" / "permanent_note_guard.py",
            PERMANENT_NOTE_GUARD_PATH.read_bytes())
        write_exact(
            root / "ai" / "tools" / "role_contract.py",
            ROLE_CONTRACT_TOOL_PATH.read_bytes())
        for tool_name in OTHER_TRUSTED_TOOLS:
            write_exact(root / "ai" / "tools" / tool_name,
                        (AI_ROOT / "tools" / tool_name).read_bytes())
        write_exact(
            root / "ai" / "notes" / "role-contract.yaml",
            ROLE_CONTRACT_PATH.read_bytes())
        write_exact(
            root / "ai" / "notes" / "implementer-failure-modes.yaml",
            FAILURE_MODES_PATH.read_bytes())
        write_exact(root / "ai" / "notes" / "backlog.md", b"")
        # Worktrees and their two bootstrap sidecars are runtime state.  The
        # production ignore rule is asserted separately; the scratch rule
        # prevents Git status from recursively inspecting linked checkouts.
        ignore = GITIGNORE_PATH.read_text(encoding="utf-8")
        if ".claude/worktrees/" not in ignore:
            ignore = ignore + "\n.claude/worktrees/\n"
        write_exact(root / ".gitignore", ignore.encode("utf-8"))
        for note_name in PERMANENT_NOTES:
            write_exact(
                root / "ai" / "notes" / note_name,
                ("# Scratch permanent note\n\nArchitect-owned policy for "
                 + note_name + ".\n").encode("utf-8"))

        git(root, "init")
        git(root, "symbolic-ref", "HEAD", "refs/heads/main")
        git(root, "config", "user.name", "Primary Worktree Witness")
        git(root, "config", "user.email", "primary@example.invalid")
        # A clean clone contains the tracked backlog. The first live action
        # creates its ignored local fingerprint.
        git(root, "add", ".gitignore", ".claude/.keep",
            ".claude/FABLE_ROLE.md",
            ".claude/OPUS_ROLE.md", ".codex/REDTEAM_ROLE.md",
            "ai/tools/mailbox_daemon.py",
            "ai/tools/handoff_contract.py",
            "ai/tools/ticket_change_guard.py",
            "ai/tools/permanent_note_guard.py",
            "ai/tools/role_contract.py",
            *["ai/tools/" + name for name in OTHER_TRUSTED_TOOLS],
            "ai/notes/role-contract.yaml",
            "ai/notes/implementer-failure-modes.yaml",
            "ai/notes/backlog.md")
        git(root, "add", *[
            "ai/notes/" + note_name for note_name in PERMANENT_NOTES])
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
    """Return the exact schema-3 Architect-primary state path."""
    return managed_base(root) / STATE_NAME


def lock_path(root):
    """Return the exact v1 provisioning-lock path."""
    return managed_base(root) / LOCK_NAME


def default_primary(root):
    """Return the deterministic first-install worktree path."""
    return managed_base(root) / PRIMARY_NAME


def sol_state_path(root):
    """Return the exact schema-1 Sol-worktree state path."""
    return managed_base(root) / SOL_STATE_NAME


def implementer_state_path(root):
    """Return the exact schema-1 Implementer-worktree state path."""
    return managed_base(root) / IMPLEMENTER_STATE_NAME


def default_implementer(root):
    """Return the deterministic first-install Implementer worktree path."""
    return managed_base(root) / IMPLEMENTER_NAME


def default_sol(root):
    """Return the deterministic first-install Sol worktree path."""
    return managed_base(root) / SOL_NAME


def common_directory(root):
    """Return Git's canonical common directory for a scratch checkout."""
    raw = git(root, "rev-parse", "--git-common-dir").stdout.strip()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return str(candidate.resolve())


def load_state(root):
    """Read the exact persisted schema-3 Architect-primary state object."""
    with state_path(root).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_sol_state(root):
    """Read the exact persisted schema-1 Sol-worktree state object."""
    with sol_state_path(root).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_implementer_state(root):
    """Read the exact persisted schema-1 Implementer-worktree state."""
    with implementer_state_path(root).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def validate_state_shape(root, expected_path=None, expected_branch=None):
    """Return whether schema-3 primary state and Git agree exactly."""
    if expected_path is None:
        expected_path = default_primary(root)
    if expected_branch is None:
        expected_branch = PRIMARY_BRANCH
    path = state_path(root)
    if not path.is_file() or path.is_symlink():
        return False
    state = load_state(root)
    if (set(state) != PRIMARY_STATE_KEYS
            or state.get("schema") != PRIMARY_STATE_SCHEMA
            or state.get("topology") != PRIMARY_TOPOLOGY
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


def validate_sol_state_shape(root, expected_path=None):
    """Return whether schema-1 Sol state and Git agree exactly."""
    if expected_path is None:
        expected_path = default_sol(root)
    path = sol_state_path(root)
    if not path.is_file() or path.is_symlink():
        return False
    state = load_sol_state(root)
    if (set(state) != SOL_STATE_KEYS
            or state.get("schema") != SOL_STATE_SCHEMA
            or state.get("repository") != common_directory(root)
            or state.get("name") != expected_path.name
            or state.get("path") != str(expected_path.resolve())
            or state.get("branch") != SOL_BRANCH):
        return False
    top = git(expected_path, "rev-parse", "--show-toplevel").stdout.strip()
    branch = git(expected_path, "symbolic-ref", "HEAD").stdout.strip()
    common = common_directory(expected_path)
    return (str(Path(top).resolve()) == str(expected_path.resolve())
            and branch == SOL_BRANCH
            and common == common_directory(root)
            and expected_path.resolve() != root.resolve())


def validate_implementer_state_shape(root, expected_path=None):
    """Return whether saved Implementer state and Git agree exactly."""
    if expected_path is None:
        expected_path = default_implementer(root)
    path = implementer_state_path(root)
    if not path.is_file() or path.is_symlink():
        return False
    state = load_implementer_state(root)
    if (set(state) != IMPLEMENTER_STATE_KEYS
            or state.get("schema") != IMPLEMENTER_STATE_SCHEMA
            or state.get("repository") != common_directory(root)
            or state.get("name") != expected_path.name
            or state.get("path") != str(expected_path.resolve())
            or state.get("branch") != IMPLEMENTER_BRANCH):
        return False
    top = git(expected_path, "rev-parse", "--show-toplevel").stdout.strip()
    branch = git(expected_path, "symbolic-ref", "HEAD").stdout.strip()
    common = common_directory(expected_path)
    return (str(Path(top).resolve()) == str(expected_path.resolve())
            and branch == IMPLEMENTER_BRANCH
            and common == common_directory(root)
            and expected_path.resolve() != root.resolve())


def validate_topology(root, primary_path=None, primary_branch=None,
                      implementer_path=None, sol_path=None):
    """Prove the three saved role worktrees are valid and disjoint."""
    if primary_path is None:
        primary_path = default_primary(root)
    if implementer_path is None:
        implementer_path = default_implementer(root)
    if sol_path is None:
        sol_path = default_sol(root)
    paths = {
        root.resolve(), primary_path.resolve(), implementer_path.resolve(),
        sol_path.resolve()}
    return (
        validate_state_shape(
            root, expected_path=primary_path,
            expected_branch=primary_branch)
        and validate_implementer_state_shape(
            root, expected_path=implementer_path)
        and validate_sol_state_shape(root, expected_path=sol_path)
        and len(paths) == 4)


def root_checkout_identity(root):
    """Return the user checkout's branch, HEAD, and visible status."""
    return (
        git(root, "symbolic-ref", "HEAD").stdout.strip(),
        git(root, "rev-parse", "HEAD").stdout.strip(),
        git(root, "status", "--porcelain=v1", "--untracked-files=all")
        .stdout)


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
    """Every mailbox action provisions all role trees without touching root."""
    cases = [
        ("once", ["--once"]),
        ("watch", ["--watch", "--cycle", "0"]),
        ("send", ["--send", "architect", "--unit",
                  "Coordinate the committed scratch unit."]),
        ("severity", ["--send", "architect", "--severity", "high",
                      "--unit", "Coordinate the named review."]),
    ]
    results = []
    for label, arguments in cases:
        with scratch_repository(source=source) as root:
            root_before = root_checkout_identity(root)
            rc, stdout, stderr = invoke(root, arguments)
            primary = default_primary(root)
            implementer = default_implementer(root)
            sol = default_sol(root)
            acted_in_primary = True
            if label in ("send", "severity"):
                acted_in_primary = (
                    len(pending_markdown(primary)) == 1
                    and pending_markdown(root) == [])
                if label in ("send", "severity") and acted_in_primary:
                    expected = ("medium" if label == "send" else "high")
                    acted_in_primary = (
                        pending_markdown(primary)[0].read_text(
                            encoding="utf-8")
                        .startswith(
                            "MAILBOX-SEVERITY: " + expected + "\n"
                            "MAILBOX-SCOPE: bounded\n\n"))
            records = worktree_records(root)
            expected_rc = 0
            passed = (
                rc == expected_rc and stderr == ""
                and validate_topology(root)
                and root_checkout_identity(root) == root_before
                and len(records) == 4
                and len([item for item in records
                         if item.get("worktree")
                         == str(primary.resolve())]) == 1
                and len([item for item in records
                         if item.get("worktree")
                         == str(implementer.resolve())]) == 1
                and len([item for item in records
                         if item.get("worktree")
                         == str(sol.resolve())]) == 1
                and not (primary / ".claude" / "worktrees").exists()
                and not (implementer / ".claude" / "worktrees").exists()
                and not (sol / ".claude" / "worktrees").exists()
                and acted_in_primary)
            results.append(passed)
            print(label + " bootstrap=" + str(passed)
                  + " rc=" + str(rc)
                  + " output=" + repr(stdout[-180:]))
    return all(results)


def arm_stale_primary_protocol_refuses(source=None):
    """Refuse re-exec when the saved daemon lacks the current protocol."""
    marker = "MAILBOX_PROTOCOL_VERSION = 5"
    if source is None or source.count(marker) != 1:
        return False
    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr or not validate_topology(root):
            return False
        primary = default_primary(root)
        stale_source = source.replace(
            marker, "MAILBOX_PROTOCOL_VERSION = 4", 1)
        write_exact(
            primary / "ai" / "tools" / "mailbox_daemon.py",
            stale_source.encode("utf-8"))
        state_before = state_path(root).read_bytes()
        root_before = root_checkout_identity(root)
        rc, stdout, _stderr = invoke(
            root,
            ["--send", "architect", "--severity", "high", "--unit",
             "Please coordinate one named review."])
        passed = (
            rc != 0
            and "primary worktree error:" in stdout
            and "saved primary daemon does not enforce" in stdout
            and pending_markdown(root) == []
            and pending_markdown(primary) == []
            and state_path(root).read_bytes() == state_before
            and root_checkout_identity(root) == root_before)
        print("stale primary protocol refusal=" + str(passed)
              + " rc=" + str(rc))
        return passed


def arm_help_dry_run_and_invalid_are_zero_write(source=None):
    """Inspection and invalid invocations create no Git or runtime state."""
    invocations = [
        ("help", ["--help"], 0),
        ("preview", ["--dry-run"], 0),
        ("preview-send", ["--dry-run", "--send", "architect",
                          "--unit", "scratch preview"], 0),
        ("preview-ping", ["--dry-run", "--ping"], 0),
        ("two-actions", ["--watch", "--once"], 1),
        ("missing-unit", ["--send", "architect"], 1),
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
                         and not branch_exists(root)
                         and not branch_exists(
                             root, branch=IMPLEMENTER_BRANCH)
                         and not implementer_state_path(root).exists()
                         and not default_implementer(root).exists()
                         and not branch_exists(root, branch=SOL_BRANCH)
                         and not sol_state_path(root).exists()
                         and not default_sol(root).exists())
            passed = passed_rc and unchanged
            outcomes.append(passed)
            print(label + " zero-write=" + str(passed)
                  + " rc=" + str(rc))
        return all(outcomes)


def arm_reuse_and_cross_checkout_converge(source=None):
    """Another checkout reuses all role trees and primary transport."""
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            print("initial reuse fixture failed")
            return False
        primary = default_primary(root)
        dirty = primary / "scratch-dirty.txt"
        write_exact(dirty, b"keep this uncommitted byte-for-byte\n")
        dirty_before = file_identity(dirty)
        state_before = file_identity(state_path(root))
        implementer_state_before = file_identity(
            implementer_state_path(root))
        sol_state_before = file_identity(sol_state_path(root))
        primary_head = git(primary, "rev-parse", "HEAD").stdout.strip()
        root_before = root_checkout_identity(root)

        other = root.parent / (root.name + "-other-coordinator")
        git(root, "worktree", "add", "-b", "claude/other-coordinator",
            str(other), "main")
        try:
            rc, stdout, stderr = invoke(
                other,
                ["--send", "architect", "--unit",
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
                and file_identity(implementer_state_path(root))
                == implementer_state_before
                and file_identity(sol_state_path(root)) == sol_state_before
                and git(primary, "rev-parse", "HEAD").stdout.strip()
                == primary_head
                and root_checkout_identity(root) == root_before
                and len(worktree_records(root)) == 5
                and str(primary / "ai" / "notes" / "mailbox")
                in stdout)
            print("cross-checkout convergence=" + str(passed)
                  + " rc=" + str(rc))
            return passed
        finally:
            git(root, "worktree", "remove", "--force", str(other),
                check=False)


def arm_clean_user_main_advances_only_from_clean_checkout(source=None):
    """Accept clean user commits without discarding an Architect backlog.

    The first scratch repository reproduces the ordinary user workflow. The
    user commits through the checkout attached to ``main`` and then starts the
    daemon again. All three idle AI worktrees must advance to that exact
    commit without changing the user's checkout or the saved topology files.

    Further repositories prove that a sealed Architect backlog survives the
    advance and its recovery cut, that an active but unstarted ticket remains
    pinned, and that an Implementer-only ref move is refused.

    Arguments:
      source = optional mailbox-daemon source used by mutation witnesses.

    Returns:
      ``True`` only when the clean user commit is adopted and the ref-only
      Implementer commit is preserved but refused.
    """
    outcomes = []

    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        sol = default_sol(root)
        saved_states = {
            "primary": file_identity(state_path(root)),
            "implementer": file_identity(implementer_state_path(root)),
            "sol": file_identity(sol_state_path(root)),
        }

        marker = root / "user-main-update.txt"
        marker.write_text(
            "committed by the user checkout\n", encoding="utf-8", newline="")
        git(root, "add", marker.name)
        git(root, "commit", "-m", "user advances main")
        user_commit = git(root, "rev-parse", "HEAD").stdout.strip()
        user_checkout = root_checkout_identity(root)

        rc, _stdout, stderr = invoke(root, ["--once"])
        role_worktrees = (primary, implementer, sol)
        accepted = (
            rc == 0 and stderr == ""
            and user_checkout[0] == "refs/heads/main"
            and user_checkout[2] == ""
            and root_checkout_identity(root) == user_checkout
            and all(git(
                worktree, "rev-parse", "HEAD").stdout.strip() == user_commit
                    for worktree in role_worktrees)
            and all(git(
                worktree, "status", "--porcelain=v1",
                "--untracked-files=all").stdout == ""
                    for worktree in role_worktrees)
            and all((worktree / marker.name).read_bytes()
                    == b"committed by the user checkout\n"
                    for worktree in role_worktrees)
            and file_identity(state_path(root)) == saved_states["primary"]
            and file_identity(implementer_state_path(root))
            == saved_states["implementer"]
            and file_identity(sol_state_path(root)) == saved_states["sol"])
        outcomes.append(accepted)
        print("clean user-main commit advances idle roles=" + str(accepted))

    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        sol = default_sol(root)
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_bytes(backlog.read_bytes() + b"Architect record\n")
        seal_backlog(primary=primary)
        sealed = backlog.read_bytes()

        marker = root / "user-update-beside-architect-backlog.txt"
        marker.write_text("user update\n", encoding="utf-8", newline="")
        git(root, "add", marker.name)
        git(root, "commit", "-m", "user update beside Architect backlog")
        user_commit = git(root, "rev-parse", "HEAD").stdout.strip()

        rc, _stdout, stderr = invoke(root, ["--once"])
        status = git(
            primary, "status", "--porcelain=v1",
            "--untracked-files=all").stdout
        preserved = (
            rc == 0 and stderr == ""
            and git(primary, "rev-parse", "HEAD").stdout.strip()
            == user_commit
            and backlog.read_bytes() == sealed
            and status == " M ai/notes/backlog.md\n"
            and git(implementer, "rev-parse", "HEAD").stdout.strip()
            == user_commit
            and git(sol, "rev-parse", "HEAD").stdout.strip()
            == user_commit)
        outcomes.append(preserved)
        print("clean user-main commit preserves sealed backlog="
              + str(preserved))

    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_bytes(backlog.read_bytes() + b"Recover this record\n")
        seal_backlog(primary=primary)
        sealed = backlog.read_bytes()
        marker = root / "user-update-before-recovery-cut.txt"
        marker.write_text("user update\n", encoding="utf-8", newline="")
        git(root, "add", marker.name)
        git(root, "commit", "-m", "user update before recovery cut")
        target = git(root, "rev-parse", "HEAD").stdout.strip()

        daemon = load_scratch_daemon(primary)
        recovery = daemon._prepare_primary_backlog_overlay(
            primary_path=str(primary),
            primary_head=git(primary, "rev-parse", "HEAD").stdout.strip(),
            target=target)
        daemon._bridge_local_sealed_backlog(
            primary_worktree=str(primary))
        recovered = (
            recovery is not None and backlog.read_bytes() == sealed
            and not (primary / "ai" / "notes"
                     / daemon.BACKLOG_SYNC_RECOVERY_NAME).exists())
        outcomes.append(recovered)
        print("interrupted backlog preparation recovers=" + str(recovered))

    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        base = git(root, "rev-parse", "HEAD").stdout.strip()
        cycle = "candidate-before-user-update@" + base
        primary_daemon = load_scratch_daemon(primary)
        state = primary_daemon.read_ticket_cycle_state()
        state["active"][cycle] = {
            "phase": "implementation", "commit": None,
            "mode": "normal", "route": "primary",
            "ticket_class": "ordinary"}
        primary_daemon.write_ticket_cycle_state(state=state)
        reference = primary_daemon.cycle_candidate_ref(cycle_id=cycle)
        git(root, "update-ref", reference, base)
        candidates = primary_daemon.empty_candidate_state()
        candidates["cycles"][cycle] = {"ref": reference, "commit": base}
        primary_daemon.write_candidate_state(state=candidates)

        marker = root / "user-update-with-saved-candidate.txt"
        marker.write_text("user update\n", encoding="utf-8", newline="")
        git(root, "add", marker.name)
        git(root, "commit", "-m", "user update with saved candidate")
        target = git(root, "rev-parse", "HEAD").stdout.strip()
        root_daemon = load_scratch_daemon(root)
        advanced = root_daemon.bootstrap_sync_primary_from_main_authority(
            primary_path=str(primary), primary_branch=PRIMARY_BRANCH)
        candidate_preserved = (
            advanced
            and git(primary, "rev-parse", "HEAD").stdout.strip() == target
            and git(implementer, "rev-parse", "HEAD").stdout.strip() == base
            and git(root, "rev-parse", reference).stdout.strip() == base
            and primary_daemon.read_candidate_state() == candidates)
        outcomes.append(candidate_preserved)
        print("clean user-main commit preserves candidate="
              + str(candidate_preserved))

    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        sol = default_sol(root)
        base = git(root, "rev-parse", "HEAD").stdout.strip()
        daemon = load_scratch_daemon(primary)
        daemon.configure_agent_worktrees(
            primary_path=str(primary), implementer_path=str(implementer),
            sol_path=str(sol))
        cycle = "unstarted-user-main-update@" + base
        state = daemon.read_ticket_cycle_state()
        state["active"][cycle] = {
            "phase": "implementation", "commit": None,
            "mode": "normal", "route": "primary",
            "ticket_class": "ordinary"}
        daemon.write_ticket_cycle_state(state=state)

        marker = root / "user-main-during-unstarted-ticket.txt"
        marker.write_text(
            "committed before the Implementer starts\n",
            encoding="utf-8", newline="")
        git(root, "add", marker.name)
        git(root, "commit", "-m", "user advances main before child start")
        user_commit = git(root, "rev-parse", "HEAD").stdout.strip()

        rc, _stdout, stderr = invoke(root, ["--once"])
        state_after = daemon.read_ticket_cycle_state()
        accepted_unstarted = (
            rc == 0 and stderr == ""
            and git(primary, "rev-parse", "HEAD").stdout.strip()
            == user_commit
            and git(sol, "rev-parse", "HEAD").stdout.strip() == user_commit
            and git(implementer, "rev-parse", "HEAD").stdout.strip() == base
            and state_after["active"].get(cycle) == state["active"][cycle])
        outcomes.append(accepted_unstarted)
        print("clean user-main commit preserves unstarted ticket="
              + str(accepted_unstarted))

    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        sol = default_sol(root)
        base = git(root, "rev-parse", "HEAD").stdout.strip()
        saved_states = {
            "primary": file_identity(state_path(root)),
            "implementer": file_identity(implementer_state_path(root)),
            "sol": file_identity(sol_state_path(root)),
        }

        rogue = implementer / "unaudited-implementer-change.txt"
        rogue.write_text(
            "not authorized by the user checkout\n",
            encoding="utf-8", newline="")
        git(implementer, "add", rogue.name)
        git(implementer, "commit", "-m", "unaudited Implementer commit")
        rogue_commit = git(
            implementer, "rev-parse", "HEAD").stdout.strip()
        rogue_identity = file_identity(rogue)
        git(root, "update-ref", "refs/heads/main", rogue_commit, base)
        user_checkout = root_checkout_identity(root)

        rc, stdout, stderr = invoke(root, ["--once"])
        refused = (
            rc != 0 and stderr == ""
            and "primary worktree error:" in stdout
            and user_checkout[0] == "refs/heads/main"
            and user_checkout[1] == rogue_commit
            and user_checkout[2] != ""
            and root_checkout_identity(root) == user_checkout
            and git(primary, "rev-parse", "HEAD").stdout.strip() == base
            and git(sol, "rev-parse", "HEAD").stdout.strip() == base
            and git(implementer, "rev-parse", "HEAD").stdout.strip()
            == rogue_commit
            and file_identity(rogue) == rogue_identity
            and git(root, "rev-parse", "refs/heads/main").stdout.strip()
            == rogue_commit
            and file_identity(state_path(root)) == saved_states["primary"]
            and file_identity(implementer_state_path(root))
            == saved_states["implementer"]
            and file_identity(sol_state_path(root)) == saved_states["sol"])
        outcomes.append(refused)
        print("Implementer-only main ref move refused=" + str(refused))

    return outcomes == [True, True, True, True, True, True]


def arm_failed_ancestor_handoff_requeues_and_advances(source=None):
    """Recover after startup moved a clean Implementer past the ticket."""
    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        sol = default_sol(root)
        old_base = git(root, "rev-parse", "HEAD").stdout.strip()

        marker = root / "new-ticket-base.txt"
        marker.write_text("new base\n", encoding="utf-8", newline="")
        git(root, "add", marker.name)
        git(root, "commit", "-m", "new ticket base")
        ticket_base = git(root, "rev-parse", "HEAD").stdout.strip()
        git(primary, "merge", "--ff-only", ticket_base)
        git(sol, "merge", "--ff-only", ticket_base)

        repair = root / "later-daemon-repair.txt"
        repair.write_text("later repair\n", encoding="utf-8", newline="")
        git(root, "add", repair.name)
        git(root, "commit", "-m", "later daemon repair")
        current_main = git(root, "rev-parse", "HEAD").stdout.strip()
        git(primary, "merge", "--ff-only", current_main)
        git(sol, "merge", "--ff-only", current_main)

        daemon = load_scratch_daemon(primary)
        daemon.configure_agent_worktrees(
            primary_path=str(primary), implementer_path=str(implementer),
            sol_path=str(sol))
        cycle = "failed-ancestor-handoff@" + ticket_base
        state = daemon.read_ticket_cycle_state()
        state["active"][cycle] = {
            "phase": "implementation", "commit": None,
            "mode": "normal", "route": "primary"}
        daemon.write_ticket_cycle_state(state=state)
        # The strict state writer materializes compatibility defaults such as
        # the ordinary ticket class. Compare recovery with the exact saved
        # record rather than the caller's pre-normalization dictionary.
        expected_active = daemon.read_ticket_cycle_state()["active"][cycle]
        daemon.sync_all_clean_role_baselines(target=current_main)
        older_base_was_preserved = (
            git(implementer, "rev-parse", "HEAD").stdout.strip()
            == old_base)

        # Reproduce the live state made by the older startup order: the
        # managed checkout had already been advanced before recovery ran.
        git(implementer, "merge", "--ff-only", current_main)
        prelaunch = primary / "ai" / "notes" / "mailbox" / "prelaunch"
        prelaunch.mkdir(parents=True, exist_ok=True)
        message = prelaunch / "0017-to-opus.md"
        message.write_text(
            "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
            + "\nMAILBOX-MODE: normal\n\n"
            "- **Directive:** [ai/notes/ticket.md, exact "
            "Implementation directive section]\n",
            encoding="utf-8", newline="")

        recovered = daemon.recover_prelaunch_messages()
        root_message = message.parent.parent / message.name
        prepared = daemon.prepare_implementer_cycle_checkout(cycle_id=cycle)
        passed = (
            older_base_was_preserved and recovered == 1
            and root_message.is_file()
            and not message.exists()
            and prepared == ticket_base
            and git(implementer, "rev-parse", "HEAD").stdout.strip()
            == ticket_base
            and current_main != ticket_base
            and git(root, "merge-base", "--is-ancestor",
                    old_base, ticket_base, check=False).returncode == 0
            and daemon.read_ticket_cycle_state()["active"].get(cycle)
            == expected_active)
        print("failed handoff survives startup baseline sync=" + str(passed))
        return passed


def arm_interrupted_implementer_bootstrap_is_exactly_resumable(source=None):
    """A registered exact Implementer tree is recovered when state is absent."""
    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary_state_before = file_identity(state_path(root))
        sol_state_before = file_identity(sol_state_path(root))
        root_before = root_checkout_identity(root)
        implementer = default_implementer(root)
        implementer_head = git(
            implementer, "rev-parse", "HEAD").stdout.strip()
        sentinel = implementer / "interrupted-bootstrap-sentinel.txt"
        write_exact(sentinel, b"preserve exact Implementer bytes\n")
        sentinel_before = file_identity(sentinel)
        implementer_state_path(root).unlink()

        rc, stdout, stderr = invoke(root, ["--once"])
        passed = (
            rc == 0 and stderr == "" and validate_topology(root)
            and file_identity(state_path(root)) == primary_state_before
            and file_identity(sol_state_path(root)) == sol_state_before
            and file_identity(sentinel) == sentinel_before
            and git(implementer, "rev-parse", "HEAD").stdout.strip()
            == implementer_head
            and root_checkout_identity(root) == root_before
            and len(worktree_records(root)) == 4
            and "recovered exact interrupted implementer-worktree bootstrap"
            in stdout.lower())
        print("interrupted Implementer bootstrap recovery=" + str(passed))
        return passed


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

        dispatch_lock = archived.parents[1] / ".dispatch.lock"
        handle = dispatch_lock.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            rc_live, live_stdout, _stderr = invoke(existing, ["--once"])
            live_refused = (
                rc_live != 0 and "live watcher"
                in live_stdout.lower() and not state_path(root).exists())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

        rc, _stdout, _stderr = invoke(existing, ["--once"])
        expected_branch = "refs/heads/claude/existing-coordinator"
        adopted = (rc == 0 and validate_topology(
            root, primary_path=existing, primary_branch=expected_branch))

        marker = root / "after-adoption.txt"
        write_exact(marker, b"user main moved after adoption\n")
        git(root, "add", marker.name)
        git(root, "commit", "-m", "advance main after adoption")
        new_main = git(root, "rev-parse", "HEAD").stdout.strip()
        rc_update, _stdout, _stderr = invoke(root, ["--once"])
        state = json.loads(state_path(root).read_text(encoding="utf-8"))
        resynced = (
            rc_update == 0 and state["branch"] == expected_branch
            and all(git(path, "rev-parse", "HEAD").stdout.strip()
                    == new_main for path in (
                        existing, default_implementer(root),
                        default_sol(root))))

        daemon = load_scratch_daemon(existing)
        daemon.configure_agent_worktrees(
            primary_path=str(existing),
            implementer_path=str(default_implementer(root)),
            sol_path=str(default_sol(root)),
            primary_branch=expected_branch)
        observed = {}

        def capture_branch(*, worktree, expected_branch, label):
            observed["branch"] = expected_branch
            raise RuntimeError("branch captured")

        daemon._symbolic_worktree_branch = capture_branch
        try:
            daemon.require_architect_notes_commit("0" * 40, "1" * 40)
        except RuntimeError as exc:
            note_branch = (str(exc) == "branch captured"
                           and observed.get("branch") == expected_branch)
        else:
            note_branch = False
        rc_send, _stdout, _stderr = invoke(
            root, ["--send", "architect", "--unit",
                   "Scratch after adoption."])
        queued = pending_markdown(existing)
        passed = (
            live_refused and adopted and resynced and note_branch
            and rc_send == 0
            and file_identity(archived) == archived_before
            and file_identity(relay) == relay_before
            and [path.name for path in queued] == ["0043-to-fable.md"]
            and not default_primary(root).exists()
            and len(worktree_records(root)) == 4)
        print("existing coordinator adoption=" + str(passed))
        return passed


def arm_adoption_publication_fences_late_watcher(source=None):
    """A watcher starting after the first scan prevents state publication."""
    with scratch_repository(source=source) as root:
        existing = managed_base(root) / "late-watcher-coordinator"
        git(root, "worktree", "add", "-b", "claude/late-watcher",
            str(existing), "main")
        archived, archived_before = transport_case(
            existing, "ai/notes/mailbox/done/0042-to-fable.md",
            b"historical mailbox bytes\n")
        daemon = load_scratch_daemon(root)
        original_open = daemon._open_legacy_transport_lock
        holder = [None]

        def start_watcher_before_lock(path, nonblocking):
            if path.endswith("/.dispatch.lock") and holder[0] is None:
                program = (
                    "import fcntl,sys,time; "
                    "f=open(sys.argv[1],'a+'); "
                    "fcntl.flock(f.fileno(),fcntl.LOCK_EX); "
                    "print('ready',flush=True); time.sleep(30)")
                holder[0] = subprocess.Popen(
                    [sys.executable, "-c", program, path],
                    cwd=str(root), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True)
                readable, _writable, _errors = select.select(
                    [holder[0].stdout], [], [], 5.0)
                if (not readable
                        or holder[0].stdout.readline().strip() != "ready"):
                    raise RuntimeError("late-watcher fixture did not lock")
            return original_open(path=path, nonblocking=nonblocking)

        daemon._open_legacy_transport_lock = start_watcher_before_lock
        error = None
        try:
            daemon.provision_or_adopt_primary(
                repository_root=str(root), current_worktree=str(existing))
        except BaseException as exc:
            error = exc
        finally:
            if holder[0] is not None:
                holder[0].terminate()
                holder[0].wait(timeout=5)
            daemon._open_legacy_transport_lock = original_open
        passed = (
            isinstance(error, daemon.PrimaryWorktreeError)
            and "legacy transport is live" in str(error).lower()
            and file_identity(archived) == archived_before
            and not state_path(root).exists()
            and not default_primary(root).exists())
        print("late watcher adoption fence=" + str(passed))
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
            rc == 0 and stderr == "" and validate_topology(root)
            and copied_archive.read_bytes() == archived.read_bytes()
            and copied_relay.read_bytes() == relay.read_bytes()
            and file_identity(archived) == archived_before
            and file_identity(relay) == relay_before
            and "bridged archived main-checkout" in stdout)

        rc_send, _stdout, _stderr = invoke(
            root, ["--send", "architect", "--unit",
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
        interrupted_copy = partial.parent / ".primary-archive-stranded"
        write_exact(interrupted_copy, b"incomplete internal copy\n")

        rc, stdout, stderr = invoke(root, ["--once"])
        copied_relay = primary / "ai" / "notes" / "relay" / relay.name
        resumed = (
            rc == 0 and stderr == "" and validate_topology(root)
            and file_identity(partial) == partial_before
            and not interrupted_copy.exists()
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
            and validate_topology(root)
            and len(records) == 4
            and len([item for item in records
                     if item.get("worktree")
                     == str(default_primary(root).resolve())]) == 1
            and len([item for item in records
                     if item.get("worktree")
                     == str(default_implementer(root).resolve())]) == 1
            and len([item for item in records
                     if item.get("worktree")
                     == str(default_sol(root).resolve())]) == 1)
        print("concurrent bootstrap=" + str(passed)
              + " returncodes=" + repr(returncodes))
        return passed


def arm_final_publication_fences_late_sender(source=None):
    """A sender taking the sequence lock at publication cannot be stranded."""
    with scratch_repository(source=source) as root:
        daemon = load_scratch_daemon(root)
        original_open = daemon._open_legacy_transport_lock
        injected = [False]
        holder = [None]

        def admit_sender_between_publication_locks(path, nonblocking):
            lock = original_open(path=path, nonblocking=nonblocking)
            if path.endswith("/.dispatch.lock") and not injected[0]:
                injected[0] = True
                sequence = Path(path).with_name(".sequence.lock")
                program = (
                    "import fcntl,sys,time; "
                    "f=open(sys.argv[1],'a+'); "
                    "fcntl.flock(f.fileno(),fcntl.LOCK_EX); "
                    "print('ready',flush=True); time.sleep(30)")
                holder[0] = subprocess.Popen(
                    [sys.executable, "-c", program, str(sequence)],
                    cwd=str(root), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True)
                readable, _writable, _errors = select.select(
                    [holder[0].stdout], [], [], 5.0)
                if (not readable
                        or holder[0].stdout.readline().strip() != "ready"):
                    lock.close()
                    raise RuntimeError("late-sender fixture did not lock")
            return lock

        daemon._open_legacy_transport_lock = (
            admit_sender_between_publication_locks)
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
            daemon._open_legacy_transport_lock = original_open
        passed = (
            isinstance(error, daemon.PrimaryWorktreeError)
            and "legacy transport is live" in str(error).lower()
            and injected[0] and not state_path(root).exists()
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


def arm_legacy_v1_state_refuses_without_mutation(source=None):
    """A saved v1 topology is preserved and refuses every live dispatch."""
    with scratch_repository(source=source) as root:
        managed_base(root).mkdir(parents=True)
        primary = default_primary(root)
        git(root, "worktree", "add", "-b", "claude/mailbox-primary",
            str(primary), "main")
        legacy = {
            "schema": 1,
            "repository": common_directory(root),
            "name": PRIMARY_NAME,
            "path": str(primary.resolve()),
            "branch": PRIMARY_BRANCH,
        }
        write_exact(
            state_path(root),
            (json.dumps(legacy, sort_keys=True) + "\n").encode("utf-8"))
        state_before = file_identity(state_path(root))
        root_before = root_checkout_identity(root)
        primary_head = git(primary, "rev-parse", "HEAD").stdout.strip()

        preview_rc, preview, preview_err = invoke(
            root, ["--dry-run", "--once"])
        preview_preserved = (
            preview_rc == 0 and preview_err == ""
            and file_identity(state_path(root)) == state_before
            and not sol_state_path(root).exists()
            and not default_sol(root).exists()
            and "predates separate architect and implementer worktrees"
            in preview.lower()
            and "live action would refuse" in preview.lower())
        rc, stdout, stderr = invoke(root, ["--once"])
        explanation = stdout.lower()
        passed = (
            preview_preserved
            and rc != 0 and stderr == ""
            and file_identity(state_path(root)) == state_before
            and root_checkout_identity(root) == root_before
            and git(primary, "rev-parse", "HEAD").stdout.strip()
            == primary_head
            and len(worktree_records(root)) == 2
            and not sol_state_path(root).exists()
            and not default_sol(root).exists()
            and not branch_exists(root, branch=SOL_BRANCH)
            and "predates the separate implementer worktree" in explanation
            and "stop every old mailbox process" in explanation
            and "update" in explanation
            and "initialize" in explanation)
        print("legacy v1 topology refusal=" + str(passed)
              + " rc=" + str(rc))
        return passed


def arm_legacy_two_tree_state_refuses_without_mutation(source=None):
    """The former shared-Claude topology cannot resume under a new daemon."""
    with scratch_repository(source=source) as root:
        managed_base(root).mkdir(parents=True)
        primary = default_primary(root)
        git(root, "worktree", "add", "-b", "claude/mailbox-primary",
            str(primary), "main")
        legacy = {
            "schema": 2,
            "repository": common_directory(root),
            "name": PRIMARY_NAME,
            "path": str(primary.resolve()),
            "branch": PRIMARY_BRANCH,
            "topology": "dedicated-sol-worktree-v1",
        }
        write_exact(
            state_path(root),
            (json.dumps(legacy, sort_keys=True) + "\n").encode("utf-8"))
        state_before = file_identity(state_path(root))
        root_before = root_checkout_identity(root)
        primary_head = git(primary, "rev-parse", "HEAD").stdout.strip()

        preview_rc, preview, preview_err = invoke(
            root, ["--dry-run", "--once"])
        preview_preserved = (
            preview_rc == 0 and preview_err == ""
            and file_identity(state_path(root)) == state_before
            and not implementer_state_path(root).exists()
            and not default_implementer(root).exists()
            and not sol_state_path(root).exists()
            and not default_sol(root).exists()
            and "predates separate architect and implementer worktrees"
            in preview.lower())
        rc, stdout, stderr = invoke(root, ["--once"])
        explanation = stdout.lower()
        passed = (
            preview_preserved and rc != 0 and stderr == ""
            and file_identity(state_path(root)) == state_before
            and root_checkout_identity(root) == root_before
            and git(primary, "rev-parse", "HEAD").stdout.strip()
            == primary_head
            and len(worktree_records(root)) == 2
            and not implementer_state_path(root).exists()
            and not default_implementer(root).exists()
            and not branch_exists(root, branch=IMPLEMENTER_BRANCH)
            and not sol_state_path(root).exists()
            and not default_sol(root).exists()
            and "predates the separate implementer worktree" in explanation
            and "stop every old mailbox process" in explanation)
        print("legacy two-tree topology refusal=" + str(passed)
              + " rc=" + str(rc))
        return passed


def arm_sol_collisions_and_corrupt_state_fail_closed(source=None):
    """Sol path/branch collisions and corrupt state never fall back to root."""
    outcomes = []

    with scratch_repository(source=source) as root:
        sentinel = default_sol(root) / "sentinel"
        write_exact(sentinel, b"ordinary Sol-path collision\n")
        before = file_identity(sentinel)
        root_before = root_checkout_identity(root)
        rc, stdout, _stderr = invoke(root, ["--once"])
        refused = (
            rc != 0 and file_identity(sentinel) == before
            and validate_state_shape(root)
            and not sol_state_path(root).exists()
            and not branch_exists(root, branch=SOL_BRANCH)
            and root_checkout_identity(root) == root_before
            and len(worktree_records(root)) == 3
            and "not a registered worktree" in stdout)
        outcomes.append(refused)
        print("Sol path collision refused=" + str(refused))

    with scratch_repository(source=source) as root:
        git(root, "branch", "codex/mailbox-sol", "main")
        sol_sha = git(root, "rev-parse", SOL_BRANCH).stdout.strip()
        root_before = root_checkout_identity(root)
        rc, stdout, _stderr = invoke(root, ["--once"])
        refused = (
            rc != 0 and validate_state_shape(root)
            and not sol_state_path(root).exists()
            and not default_sol(root).exists()
            and git(root, "rev-parse", SOL_BRANCH).stdout.strip() == sol_sha
            and root_checkout_identity(root) == root_before
            and len(worktree_records(root)) == 3
            and "already exists" in stdout)
        outcomes.append(refused)
        print("Sol branch collision refused=" + str(refused))

    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        original = sol_state_path(root).read_bytes()
        state = json.loads(original.decode("utf-8"))
        primary_state_before = file_identity(state_path(root))
        root_before = root_checkout_identity(root)
        duplicate = (
            "{\"schema\":1,\"schema\":1,\"repository\":"
            + json.dumps(state["repository"])
            + ",\"name\":\"mailbox-sol\",\"path\":"
            + json.dumps(state["path"])
            + ",\"branch\":\"refs/heads/codex/mailbox-sol\"}\n")
        variants = [
            ("invalid-json", b"{ invalid Sol state\n"),
            ("duplicate-key", duplicate.encode("utf-8")),
        ]
        unknown = dict(state)
        unknown["topology"] = PRIMARY_TOPOLOGY
        variants.append((
            "unknown-key", (json.dumps(unknown) + "\n").encode("utf-8")))
        foreign = dict(state)
        foreign["repository"] = str(root / "foreign.git")
        variants.append((
            "foreign-repository",
            (json.dumps(foreign) + "\n").encode("utf-8")))
        in_root = dict(state)
        in_root["path"] = str(root)
        in_root["name"] = root.name
        variants.append((
            "user-root-fallback",
            (json.dumps(in_root) + "\n").encode("utf-8")))
        shared = dict(state)
        shared["path"] = str(primary)
        shared["name"] = primary.name
        variants.append((
            "Claude-Sol-collocation",
            (json.dumps(shared) + "\n").encode("utf-8")))
        wrong_branch = dict(state)
        wrong_branch["branch"] = PRIMARY_BRANCH
        variants.append((
            "wrong-branch",
            (json.dumps(wrong_branch) + "\n").encode("utf-8")))

        for label, payload in variants:
            write_exact(sol_state_path(root), payload)
            before = file_identity(sol_state_path(root))
            rc, _stdout, _stderr = invoke(
                root, ["--send", "architect", "--unit", "must not queue"])
            refused = (
                rc != 0 and file_identity(sol_state_path(root)) == before
                and file_identity(state_path(root)) == primary_state_before
                and root_checkout_identity(root) == root_before
                and pending_markdown(root) == []
                and pending_markdown(primary) == [])
            outcomes.append(refused)
            print("Sol " + label + " refused=" + str(refused))

        external = managed_base(root) / ".sol-state-sentinel"
        write_exact(external, original)
        sol_state_path(root).unlink()
        os.symlink(str(external), str(sol_state_path(root)))
        external_before = file_identity(external)
        rc, _stdout, _stderr = invoke(root, ["--once"])
        refused = (
            rc != 0 and sol_state_path(root).is_symlink()
            and file_identity(external) == external_before
            and file_identity(state_path(root)) == primary_state_before
            and root_checkout_identity(root) == root_before)
        outcomes.append(refused)
        print("Sol symlink state refused=" + str(refused))
        sol_state_path(root).unlink()

        if hasattr(os, "mkfifo"):
            os.mkfifo(str(sol_state_path(root)), 0o600)
            rc, _stdout, _stderr = invoke(root, ["--once"], timeout=4)
            refused = rc != 124 and rc != 0
            outcomes.append(refused)
            print("Sol fifo state refused=" + str(refused))
            sol_state_path(root).unlink()
        write_exact(sol_state_path(root), original)
    return all(outcomes)


def arm_sol_registered_move_and_reuse_are_preserved(source=None):
    """A Git-managed Sol move updates state; later runs reuse it exactly."""
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        sol = default_sol(root)
        moved = managed_base(root) / "mailbox-sol-moved"
        dirty = sol / "preserve-sol-dirty.txt"
        write_exact(dirty, b"preserve Sol local work byte-for-byte\n")
        primary_state_before = file_identity(state_path(root))
        root_before = root_checkout_identity(root)
        sol_head = git(sol, "rev-parse", "HEAD").stdout.strip()
        git(root, "worktree", "move", str(sol), str(moved))
        dirty_moved = moved / dirty.name
        dirty_before = file_identity(dirty_moved)

        rc, stdout, stderr = invoke(root, ["--once"])
        if (rc != 0 or stderr != ""
                or not validate_topology(root, sol_path=moved)):
            print("Sol move recovery setup failed rc=" + str(rc)
                  + " output=" + repr(stdout[-240:]))
            return False
        saved_before_reuse = file_identity(sol_state_path(root))
        rc_reuse, _reuse_out, reuse_err = invoke(root, ["--once"])
        daemon = load_scratch_daemon(primary)
        create_empty_sealed_backlog(primary=primary)
        queued = daemon.send(
            agent="sol", ticket_kind="discovery", severity="medium",
            scope="bounded", text="Review the saved moved Sol checkout.",
            dry_run=False)
        rc_queue = 0 if queued else 1
        queue_err = ""
        rc_preview, preview, preview_err = invoke(
            root, ["--dry-run", "--once"])
        checks = {
            "queue": rc_queue == 0 and queue_err == "",
            "preview": rc_preview == 0 and preview_err == "",
            "reuse": rc_reuse == 0 and reuse_err == "",
            "topology": validate_topology(root, sol_path=moved),
            "stable-sol-state": (file_identity(sol_state_path(root))
                                 == saved_before_reuse),
            "stable-primary-state": (file_identity(state_path(root))
                                     == primary_state_before),
            "dirty-preserved": file_identity(dirty_moved) == dirty_before,
            "head-preserved": (git(moved, "rev-parse", "HEAD")
                               .stdout.strip() == sol_head),
            "root-preserved": root_checkout_identity(root) == root_before,
            "preview-sol": str(moved) in preview,
            "preview-notes": (str(default_primary(root) / "ai" / "notes")
                              in preview),
            "move-reported": "worktree moved by git; saved" in stdout,
        }
        passed = all(checks.values())
        print("Sol registered move and reuse=" + str(passed))
        if not passed:
            print("Sol move checks=" + repr(checks)
                  + " preview-rc=" + str(rc_preview)
                  + " preview-error=" + repr(preview_err[-500:])
                  + " preview=" + repr(preview[-500:]))
        return passed


def arm_sol_launch_boundary_revalidates_branch_and_active_state(source=None):
    """A Sol branch race never becomes a successful child or archive."""

    class PopenProxy:
        """Override only this imported daemon's Popen attribute."""

        def __init__(self, module, replacement):
            self.module = module
            self.replacement = replacement

        def __getattr__(self, name):
            if name == "Popen":
                return self.replacement
            return getattr(self.module, name)

    class ObservedProcess:
        """Popen-shaped child recording whether topology refusal killed it."""

        def __init__(self):
            self.returncode = 0
            self.killed = False
            self.waited = False

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self):
            self.waited = True
            return self.returncode

    def prepare(root, unit, activate=True):
        """Bootstrap, queue one bounded Sol discovery, and import daemon."""
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return None
        primary = default_primary(root)
        daemon = load_scratch_daemon(primary)
        create_empty_sealed_backlog(primary=primary)
        queued = daemon.send(
            agent="sol", ticket_kind="discovery", severity="medium",
            scope="bounded", text=unit, dry_run=False)
        pending = pending_markdown(primary)
        if not queued or len(pending) != 1:
            return None
        if activate:
            daemon.ensure_primary_execution(live_action=True, dry_run=False)
            daemon.AGENT_COMMANDS = daemon.build_agent_commands(
                fable_effort=daemon.DEFAULT_FABLE_EFFORT,
                opus_effort=daemon.DEFAULT_OPUS_EFFORT,
                sol_effort=daemon.DEFAULT_SOL_EFFORT,
                sol_context_budget=daemon.DEFAULT_SOL_CONTEXT_BUDGET,
                sol_worktree=daemon.AGENT_CWD["sol"],
                shared_notes=daemon.ACTIVE_TOPOLOGY["shared_notes"])
        return daemon, primary, default_sol(root), pending[0]

    def transport_counts(primary):
        """Count each durable state for the one raced Sol message."""
        mailbox = primary / "ai" / "notes" / "mailbox"
        return {
            "pending": len(list(mailbox.glob("*-to-sol.md"))),
            "inflight": len(list((mailbox / "inflight").glob(
                "*-to-sol.md"))),
            "done": len(list((mailbox / "done").glob("*-to-sol.md"))),
            "failed": len(list((mailbox / "failed").glob(
                "*-to-sol.md"))),
        }

    outcomes = []

    # A regular leaf file is not authoritative when one of its primary-tree
    # parent directories redirects into a different tree. Refuse before the
    # pending message is claimed or a child is admitted.
    for relative_parent in (Path(".codex"), Path("ai") / "tools"):
        with scratch_repository(source=source) as root:
            prepared = prepare(
                root, "Refuse a redirected authoritative role parent.")
            if prepared is None:
                return False
            daemon, primary, _sol, message = prepared
            parent = primary / relative_parent
            relocated = primary / ("redirected-"
                                   + "-".join(relative_parent.parts))
            parent.rename(relocated)
            parent.symlink_to(relocated, target_is_directory=True)
            pending_before = file_identity(message)
            launches = []

            def redirected_parent_popen(command, stdout, stderr, cwd, env):
                del command, stdout, stderr, cwd, env
                launches.append("admitted")
                return ObservedProcess()

            original_subprocess = daemon.subprocess
            daemon.subprocess = PopenProxy(
                original_subprocess, redirected_parent_popen)
            try:
                result = daemon.dispatch(path=str(message), dry_run=False)
            finally:
                daemon.subprocess = original_subprocess
            redirected_parent_refused = (
                result is False and launches == [] and message.exists()
                and file_identity(message) == pending_before
                and transport_counts(primary)
                == {"pending": 1, "inflight": 0,
                    "done": 0, "failed": 0})
            outcomes.append(redirected_parent_refused)
            print("Sol redirected " + str(relative_parent)
                  + " parent refused=" + str(redirected_parent_refused))

    # No imported caller may dispatch Sol merely because its paths happen to
    # look plausible. ACTIVE_TOPOLOGY is the live bootstrap capability.
    with scratch_repository(source=source) as root:
        prepared = prepare(
            root, "Refuse this Sol turn without live topology authority.",
            activate=False)
        if prepared is None:
            return False
        daemon, primary, _sol, message = prepared
        pending_before = file_identity(message)
        launches = []

        def forbidden_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            launches.append("admitted")
            return ObservedProcess()

        original_subprocess = daemon.subprocess
        daemon.subprocess = PopenProxy(original_subprocess, forbidden_popen)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original_subprocess
        missing_active_refused = (
            result is False and launches == []
            and daemon.ACTIVE_TOPOLOGY is None
            and message.exists()
            and file_identity(message) == pending_before
            and transport_counts(primary)
            == {"pending": 1, "inflight": 0, "done": 0, "failed": 0})
        outcomes.append(missing_active_refused)
        print("Sol missing active topology refused="
              + str(missing_active_refused))

    # Drift after the initial valid proof and atomic claim, but before Popen.
    # The before-Popen full revalidation must refuse admission entirely.
    with scratch_repository(source=source) as root:
        prepared = prepare(
            root, "Race the Sol branch after claim and before child launch.")
        if prepared is None:
            return False
        daemon, primary, sol, message = prepared
        launches = []
        child = ObservedProcess()
        real_claim = daemon.claim_message
        original_identity = file_identity(message)

        def claim_then_switch(path):
            claimed = real_claim(path=path)
            if claimed is not None:
                git(sol, "switch", "--ignore-other-worktrees", "main")
            return claimed

        def observed_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            launches.append("admitted")
            return child

        original_subprocess = daemon.subprocess
        daemon.claim_message = claim_then_switch
        daemon.subprocess = PopenProxy(original_subprocess, observed_popen)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original_subprocess
        prelaunch_path = (primary / "ai" / "notes" / "mailbox"
                          / "prelaunch" / message.name)
        retained = (
            result is False and launches == []
            and not child.killed and not child.waited
            and git(sol, "symbolic-ref", "HEAD").stdout.strip()
            == "refs/heads/main"
            and transport_counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 0}
            and prelaunch_path.is_file()
            and file_identity(prelaunch_path) == original_identity)
        git(sol, "switch", "--ignore-other-worktrees",
            SOL_BRANCH.removeprefix("refs/heads/"))
        daemon.claim_message = real_claim
        requeued_path = primary / "ai" / "notes" / "mailbox" / message.name
        requeued = (
            daemon.recover_prelaunch_messages() == 1
            and file_identity(requeued_path) == original_identity
            and transport_counts(primary)
            == {"pending": 1, "inflight": 0, "done": 0, "failed": 0})
        retry_child = ObservedProcess()
        retry_child.returncode = 1
        retry_launches = []

        def retry_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            retry_launches.append("admitted")
            return retry_child

        daemon.subprocess = PopenProxy(original_subprocess, retry_popen)
        try:
            retry_result = daemon.dispatch(
                path=str(requeued_path), dry_run=False)
        finally:
            daemon.subprocess = original_subprocess
        retried = (
            retry_result is False and retry_launches == ["admitted"]
            and transport_counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1}
            and not prelaunch_path.exists())
        prelaunch_refused = retained and requeued and retried
    outcomes.append(prelaunch_refused)
    print("Sol pre-Popen branch race refused=" + str(prelaunch_refused))

    # Keep the exact role-file proof from the first admission check. Replacing
    # a role after the atomic claim but before Popen must not be accepted just
    # because the replacement still has the expected filename.
    with scratch_repository(source=source) as root:
        prepared = prepare(
            root, "Race an authoritative role after the Sol claim.")
        if prepared is None:
            return False
        daemon, primary, _sol, message = prepared
        launches = []
        child = ObservedProcess()
        role = primary / ".codex" / "REDTEAM_ROLE.md"
        role_before = role.read_bytes()
        real_claim = daemon.claim_message
        original_identity = file_identity(message)

        def claim_then_replace_role(path):
            claimed = real_claim(path=path)
            if claimed is not None:
                role.write_text(
                    "# Replaced after claim\n\nThis is not the admitted role.\n",
                    encoding="utf-8")
            return claimed

        def role_race_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            launches.append("admitted")
            return child

        original_subprocess = daemon.subprocess
        daemon.claim_message = claim_then_replace_role
        daemon.subprocess = PopenProxy(original_subprocess, role_race_popen)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original_subprocess
        prelaunch_path = (primary / "ai" / "notes" / "mailbox"
                          / "prelaunch" / message.name)
        retained = (
            result is False and launches == []
            and not child.killed and not child.waited
            and transport_counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 0}
            and prelaunch_path.is_file()
            and file_identity(prelaunch_path) == original_identity)
        role.write_bytes(role_before)
        daemon.claim_message = real_claim
        requeued_path = primary / "ai" / "notes" / "mailbox" / message.name
        requeued = (
            daemon.recover_prelaunch_messages() == 1
            and file_identity(requeued_path) == original_identity
            and transport_counts(primary)
            == {"pending": 1, "inflight": 0, "done": 0, "failed": 0})
        retry_child = ObservedProcess()
        retry_child.returncode = 1
        retry_launches = []

        def retry_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            retry_launches.append("admitted")
            return retry_child

        daemon.subprocess = PopenProxy(original_subprocess, retry_popen)
        try:
            retry_result = daemon.dispatch(
                path=str(requeued_path), dry_run=False)
        finally:
            daemon.subprocess = original_subprocess
        retried = (
            retry_result is False and retry_launches == ["admitted"]
            and transport_counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1}
            and not prelaunch_path.exists())
        role_race_refused = retained and requeued and retried
        outcomes.append(role_race_refused)
        print("Sol pre-Popen role-file race refused="
              + str(role_race_refused))

    # A switch can occur inside Popen after the last pre-launch check. The
    # immediate post-Popen full revalidation must kill and reap that child.
    with scratch_repository(source=source) as root:
        prepared = prepare(
            root, "Race the Sol branch while the child is being admitted.")
        if prepared is None:
            return False
        daemon, primary, sol, message = prepared
        launches = []
        child = ObservedProcess()

        def switching_popen(command, stdout, stderr, cwd, env):
            del command, stderr, cwd, env
            launches.append("admitted")
            git(sol, "switch", "--ignore-other-worktrees", "main")
            stdout.write("child returned across a branch race\n")
            stdout.flush()
            return child

        original_subprocess = daemon.subprocess
        daemon.subprocess = PopenProxy(original_subprocess, switching_popen)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original_subprocess
        postlaunch_killed = (
            result is False and launches == ["admitted"]
            and child.killed and child.waited and child.returncode == -9
            and git(sol, "symbolic-ref", "HEAD").stdout.strip()
            == "refs/heads/main"
            and transport_counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(postlaunch_killed)
        print("Sol around-Popen branch race killed="
              + str(postlaunch_killed))

    return all(outcomes)


def arm_implementer_launch_boundary_revalidates_branch_and_state(source=None):
    """Opus branch/state races cannot become successful child turns."""

    class PopenProxy:
        def __init__(self, module, replacement):
            self.module = module
            self.replacement = replacement

        def __getattr__(self, name):
            if name == "Popen":
                return self.replacement
            return getattr(self.module, name)

    class ObservedProcess:
        def __init__(self):
            self.returncode = 0
            self.killed = False
            self.waited = False

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self):
            self.waited = True
            return self.returncode

    def prepare(root, activate=True):
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return None
        primary = default_primary(root)
        implementer = default_implementer(root)
        base = git(implementer, "rev-parse", "HEAD").stdout.strip()
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — [Opus race](#opus-race)\n\n"
            "<a id=\"opus-race\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        message = mailbox / "0001-to-opus.md"
        message.write_text(
            "MAILBOX-FLOW: ticket\n"
            "MAILBOX-CYCLE: opus-race@" + base + "\n"
            "MAILBOX-MODE: normal\n\n"
            "Implement the bounded scratch race witness.\n",
            encoding="utf-8", newline="")
        daemon = load_scratch_daemon(primary)
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"],
            "opus": ["harmless-opus"],
            "sol": ["harmless-sol"],
        }
        if activate:
            daemon.ensure_primary_execution(live_action=True, dry_run=False)
            # This arm isolates launch-boundary Git races. The separate
            # evidence-contract arm owns directive/evidence parsing.
            daemon.prepare_implementer_evidence_contract = lambda message: {
                "contract": None, "parallel_work_plan": {},
                "note_path": "focused topology witness"}
        return daemon, primary, implementer, message

    def counts(primary):
        mailbox = primary / "ai" / "notes" / "mailbox"
        return {
            "pending": len(list(mailbox.glob("*-to-opus.md"))),
            "inflight": len(list((mailbox / "inflight").glob(
                "*-to-opus.md"))),
            "done": len(list((mailbox / "done").glob("*-to-opus.md"))),
            "failed": len(list((mailbox / "failed").glob(
                "*-to-opus.md"))),
        }

    outcomes = []

    with scratch_repository(source=source) as root:
        prepared = prepare(root=root, activate=False)
        if prepared is None:
            return False
        daemon, primary, _implementer, message = prepared
        before = file_identity(message)
        launches = []

        def forbidden_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            launches.append("admitted")
            return ObservedProcess()

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, forbidden_popen)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        refused = (
            result is False and launches == []
            and daemon.ACTIVE_TOPOLOGY is None
            and message.exists() and file_identity(message) == before
            and counts(primary)
            == {"pending": 1, "inflight": 0, "done": 0, "failed": 0})
        outcomes.append(refused)
        print("Implementer missing active topology refused=" + str(refused))

    with scratch_repository(source=source) as root:
        prepared = prepare(root=root)
        if prepared is None:
            return False
        daemon, primary, implementer, message = prepared
        launches = []
        child = ObservedProcess()
        real_claim = daemon.claim_message
        original_identity = file_identity(message)

        def claim_then_switch(path):
            claimed = real_claim(path=path)
            if claimed is not None:
                git(implementer, "switch", "--ignore-other-worktrees", "main")
            return claimed

        def observed_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            launches.append("admitted")
            return child

        original = daemon.subprocess
        daemon.claim_message = claim_then_switch
        daemon.subprocess = PopenProxy(original, observed_popen)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        prelaunch_path = (primary / "ai" / "notes" / "mailbox"
                          / "prelaunch" / message.name)
        retained = (
            result is False and launches == []
            and not child.killed and not child.waited
            and git(implementer, "symbolic-ref", "HEAD").stdout.strip()
            == "refs/heads/main"
            and counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 0}
            and prelaunch_path.is_file()
            and file_identity(prelaunch_path) == original_identity)
        git(implementer, "switch", "--ignore-other-worktrees",
            IMPLEMENTER_BRANCH.removeprefix("refs/heads/"))
        daemon.claim_message = real_claim
        requeued_path = primary / "ai" / "notes" / "mailbox" / message.name
        requeued = (
            daemon.recover_prelaunch_messages() == 1
            and file_identity(requeued_path) == original_identity
            and counts(primary)
            == {"pending": 1, "inflight": 0, "done": 0, "failed": 0})
        retry_child = ObservedProcess()
        retry_child.returncode = 1
        retry_launches = []

        def retry_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            retry_launches.append("admitted")
            return retry_child

        daemon.subprocess = PopenProxy(original, retry_popen)
        try:
            retry_result = daemon.dispatch(
                path=str(requeued_path), dry_run=False)
        finally:
            daemon.subprocess = original
        retried = (
            retry_result is False and retry_launches == ["admitted"]
            and counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1}
            and not prelaunch_path.exists())
        refused = retained and requeued and retried
        outcomes.append(refused)
        print("Implementer pre-Popen branch race refused=" + str(refused))

    with scratch_repository(source=source) as root:
        prepared = prepare(root=root)
        if prepared is None:
            return False
        daemon, primary, _implementer, message = prepared
        launches = []
        child = ObservedProcess()
        state = load_implementer_state(root)

        def corrupting_popen(command, stdout, stderr, cwd, env):
            del command, stderr, cwd, env
            launches.append("admitted")
            corrupt = dict(state)
            corrupt["branch"] = PRIMARY_BRANCH
            write_exact(
                implementer_state_path(root),
                (json.dumps(corrupt, sort_keys=True) + "\n").encode("utf-8"))
            stdout.write("child crossed an Implementer state race\n")
            stdout.flush()
            return child

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, corrupting_popen)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        killed = (
            result is False and launches == ["admitted"]
            and child.killed and child.waited and child.returncode == -9
            and counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(killed)
        print("Implementer around-Popen state race killed=" + str(killed))

    return all(outcomes)


def arm_architect_launch_boundary_revalidates_branch_and_role(source=None):
    """Fable may launch only from the proved saved primary and role file."""

    class PopenProxy:
        def __init__(self, module, replacement):
            self.module = module
            self.replacement = replacement

        def __getattr__(self, name):
            return (self.replacement if name == "Popen"
                    else getattr(self.module, name))

    class ObservedProcess:
        def __init__(self):
            self.returncode = 0
            self.killed = False
            self.waited = False

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self):
            self.waited = True
            return self.returncode

    def prepare(root, activate=True):
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return None
        primary = default_primary(root)
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        message = mailbox / "0001-to-fable.md"
        daemon = load_scratch_daemon(primary)
        message.write_text(
            daemon.architect_user_request_payload(
                text="Plan the bounded Architect topology witness.",
                discovery_severity="medium"),
            encoding="utf-8", newline="")
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"],
            "opus": ["harmless-opus"],
            "sol": ["harmless-sol"],
        }
        if activate:
            daemon.ensure_primary_execution(live_action=True, dry_run=False)
        return daemon, primary, message

    def counts(primary):
        mailbox = primary / "ai" / "notes" / "mailbox"
        return {
            state: len(list(((mailbox if state == "pending"
                              else mailbox / state)).glob("*-to-fable.md")))
            for state in ("pending", "inflight", "done", "failed")
        }

    outcomes = []
    with scratch_repository(source=source) as root:
        prepared = prepare(root=root, activate=False)
        if prepared is None:
            return False
        daemon, primary, message = prepared
        before = file_identity(message)
        launches = []
        original = daemon.subprocess
        daemon.subprocess = PopenProxy(
            original, lambda *args, **kwargs: launches.append("admitted"))
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        refused = (
            result is False and launches == []
            and message.exists() and file_identity(message) == before
            and counts(primary)
            == {"pending": 1, "inflight": 0, "done": 0, "failed": 0})
        outcomes.append(refused)
        print("Architect missing active topology refused=" + str(refused))

    with scratch_repository(source=source) as root:
        prepared = prepare(root=root)
        if prepared is None:
            return False
        daemon, primary, message = prepared
        launches = []
        child = ObservedProcess()
        role = primary / ".claude" / "FABLE_ROLE.md"
        real_claim = daemon.claim_message
        role_before = role.read_bytes()
        original_identity = file_identity(message)

        def claim_then_replace(path):
            claimed = real_claim(path=path)
            if claimed is not None:
                replacement = role.with_name("FABLE_ROLE.replacement")
                replacement.write_bytes(role.read_bytes())
                os.replace(replacement, role)
            return claimed

        def observed(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            launches.append("admitted")
            return child

        original = daemon.subprocess
        daemon.claim_message = claim_then_replace
        daemon.subprocess = PopenProxy(original, observed)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        prelaunch_path = (primary / "ai" / "notes" / "mailbox"
                          / "prelaunch" / message.name)
        retained = (
            result is False and launches == []
            and not child.killed and not child.waited
            and counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 0}
            and prelaunch_path.is_file()
            and file_identity(prelaunch_path) == original_identity)
        role.write_bytes(role_before)
        daemon.claim_message = real_claim
        requeued_path = primary / "ai" / "notes" / "mailbox" / message.name
        requeued = (
            daemon.recover_prelaunch_messages() == 1
            and file_identity(requeued_path) == original_identity
            and counts(primary)
            == {"pending": 1, "inflight": 0, "done": 0, "failed": 0})
        retry_child = ObservedProcess()
        retry_child.returncode = 1
        retry_launches = []

        def retry_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            retry_launches.append("admitted")
            return retry_child

        daemon.subprocess = PopenProxy(original, retry_popen)
        try:
            retry_result = daemon.dispatch(
                path=str(requeued_path), dry_run=False)
        finally:
            daemon.subprocess = original
        retried = (
            retry_result is False and retry_launches == ["admitted"]
            and counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1}
            and not prelaunch_path.exists()
            and git(primary, "status", "--porcelain=v1").stdout == "")
        refused = retained and requeued and retried
        outcomes.append(refused)
        print("Architect pre-Popen role race refused=" + str(refused))

    with scratch_repository(source=source) as root:
        prepared = prepare(root=root)
        if prepared is None:
            return False
        daemon, primary, message = prepared
        launches = []
        child = ObservedProcess()

        def switching(command, stdout, stderr, cwd, env):
            del command, stderr, cwd, env
            launches.append("admitted")
            git(primary, "switch", "--ignore-other-worktrees", "main")
            stdout.write("child crossed an Architect branch race\n")
            stdout.flush()
            return child

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, switching)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        killed = (
            result is False and launches == ["admitted"]
            and child.killed and child.waited and child.returncode == -9
            and counts(primary)
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(killed)
        print("Architect around-Popen branch race killed=" + str(killed))
    return all(outcomes)


def arm_candidate_snapshot_is_exact_and_immutable(source=None):
    """A later Implementer commit cannot move an earlier audit snapshot."""
    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        root_before = root_checkout_identity(root)

        base = git(implementer, "rev-parse", "HEAD").stdout.strip()
        first_cycle = "candidate-a@" + base
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — [Candidate A](#candidate-a)\n"
            "- OPEN **HIGH** **BUG FIX** — [Candidate B](#candidate-b)\n\n"
            "<a id=\"candidate-a\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n\n"
            "<a id=\"candidate-b\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)

        def flow(cycle_id):
            return (
                "MAILBOX-FLOW: ticket\n"
                "MAILBOX-CYCLE: " + cycle_id + "\n"
                "MAILBOX-MODE: normal\n\n"
                "Implement the exact scratch candidate.\n")

        daemon.register_ticket_cycle_message(
            agent="opus", message=flow(first_cycle))
        starting_a = daemon.prepare_implementer_cycle_checkout(
            cycle_id=first_cycle)
        marker = implementer / "candidate-snapshot-marker.txt"
        marker.write_text("candidate A\n", encoding="utf-8", newline="")
        git(implementer, "add", marker.name)
        git(implementer, "commit", "-m", "scratch candidate A")
        commit_a = daemon.record_implementer_candidate(
            cycle_id=first_cycle, starting_head=starting_a)
        if commit_a is None:
            return False

        fable_snapshot = Path(daemon.create_audit_snapshot(
            cycle_id=first_cycle, commit=commit_a, agent="fable"))
        sol_snapshot = Path(daemon.create_audit_snapshot(
            cycle_id=first_cycle, commit=commit_a, agent="sol"))
        first_ref = daemon.cycle_candidate_ref(cycle_id=first_cycle)

        second_cycle = "candidate-b@" + base
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow(second_cycle))
        starting_b = daemon.prepare_implementer_cycle_checkout(
            cycle_id=second_cycle)
        marker.write_text("candidate B\n", encoding="utf-8", newline="")
        git(implementer, "add", marker.name)
        git(implementer, "commit", "-m", "scratch candidate B")
        commit_b = daemon.record_implementer_candidate(
            cycle_id=second_cycle, starting_head=starting_b)
        second_ref = daemon.cycle_candidate_ref(cycle_id=second_cycle)
        candidate_state = daemon.read_candidate_state()

        checks = {
            "different-commits": commit_b is not None and commit_b != commit_a,
            "first-ref": (git(root, "rev-parse", first_ref).stdout.strip()
                          == commit_a),
            "second-ref": (commit_b is not None
                           and git(root, "rev-parse", second_ref).stdout.strip()
                           == commit_b),
            "different-refs": first_ref != second_ref,
            "state-a": candidate_state["cycles"][first_cycle]
            == {"ref": first_ref, "commit": commit_a},
            "state-b": candidate_state["cycles"][second_cycle]
            == {"ref": second_ref, "commit": commit_b},
            "fable-detached": git(
                fable_snapshot, "symbolic-ref", "-q", "HEAD",
                check=False).returncode == 1,
            "sol-detached": git(
                sol_snapshot, "symbolic-ref", "-q", "HEAD",
                check=False).returncode == 1,
            "fable-head-a": git(
                fable_snapshot, "rev-parse", "HEAD").stdout.strip()
            == commit_a,
            "sol-head-a": git(
                sol_snapshot, "rev-parse", "HEAD").stdout.strip()
            == commit_a,
            "fable-bytes-a": (
                fable_snapshot / marker.name).read_bytes() == b"candidate A\n",
            "sol-bytes-a": (
                sol_snapshot / marker.name).read_bytes() == b"candidate A\n",
            "implementer-b": marker.read_bytes() == b"candidate B\n",
            "root-preserved": root_checkout_identity(root) == root_before,
        }
        passed = all(checks.values())
        try:
            daemon.remove_audit_snapshot(
                cycle_id=first_cycle, commit=commit_a, agent="fable")
            daemon.remove_audit_snapshot(
                cycle_id=first_cycle, commit=commit_a, agent="sol")
        except BaseException:
            passed = False
        passed = passed and not fable_snapshot.exists() \
            and not sol_snapshot.exists()
        print("immutable candidate refs and audit snapshots=" + str(passed))
        if not passed:
            print("candidate snapshot checks=" + repr(checks))
        return passed


def arm_permanent_note_admin_is_exclusive_and_lands(source=None):
    """One admin turn excludes Opus, then lands P on every clean role."""

    class CompletedProcess:
        def __init__(self, callback):
            self.callback = callback
            self.returncode = None
            self.called = False

        def poll(self):
            if not self.called:
                self.called = True
                self.callback()
                self.returncode = 0
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self):
            return self.returncode

    class PopenProxy:
        def __init__(self, module, callback, launches):
            self.module = module
            self.callback = callback
            self.launches = launches

        def __getattr__(self, name):
            return (self.popen if name == "Popen"
                    else getattr(self.module, name))

        def popen(self, command, stdout, stderr, cwd, env):
            del command, stderr
            self.launches.append((Path(cwd), dict(env)))
            stdout.write("completed scratch note administration\n")
            stdout.flush()
            return CompletedProcess(
                callback=lambda: self.callback(dict(env)))

    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        sol = default_sol(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        admin = mailbox / "0001-to-fable.md"
        admin.write_text(
            daemon.architect_notes_admin_payload(
                "Record one durable scratch policy update."),
            encoding="utf-8", newline="")
        base = git(root, "rev-parse", "HEAD").stdout.strip()
        # This newer ticket is deliberately ready at the same instant.  The
        # admin boundary must leave it untouched rather than racing Opus.
        opus = mailbox / "0002-to-opus.md"
        opus.write_text(
            "MAILBOX-FLOW: ticket\n"
            "MAILBOX-CYCLE: later@" + base + "\n"
            "MAILBOX-MODE: normal\n\n"
            "Implement a later scratch ticket.\n",
            encoding="utf-8", newline="")
        launches = []
        created = {}

        def commit_note(environment):
            if environment.get("MAILBOX_NOTES_BASE") != base:
                return
            note = primary / "ai" / "notes" / "MEMORY.md"
            role = primary / ".claude" / "FABLE_ROLE.md"
            note.write_text(
                note.read_text(encoding="utf-8")
                + "\nAdmin-only scratch policy.\n",
                encoding="utf-8", newline="")
            role.write_text(
                role.read_text(encoding="utf-8")
                + "\nAdmin-only scratch role rule.\n",
                encoding="utf-8", newline="")
            git(primary, "add", "ai/notes/MEMORY.md",
                ".claude/FABLE_ROLE.md")
            git(primary, "commit", "-m", "scratch permanent note P")
            notes_commit = git(primary, "rev-parse", "HEAD").stdout.strip()
            created["P"] = notes_commit
            (mailbox / "0003-to-daemon.md").write_text(
                daemon.architect_notes_go_request_payload(
                    base_commit=base, notes_commit=notes_commit),
                encoding="utf-8", newline="")

        original_subprocess = daemon.subprocess
        daemon.subprocess = PopenProxy(
            original_subprocess, commit_note, launches)
        try:
            first = daemon.process_backlog(dry_run=False)
        finally:
            daemon.subprocess = original_subprocess
        p_commit = created.get("P")
        journal = Path(daemon.architect_notes_admin_journal_path(
            request_name=admin.name))
        journal_retained = False
        if journal.is_file():
            saved_admin = mailbox / "done" / admin.name
            saved_message = saved_admin.read_text(encoding="utf-8")
            journal_state = daemon.read_architect_notes_admin_journal(
                request_name=admin.name, request_message=saved_message)
            journal_retained = (
                journal_state["phase"] == "validated-commit"
                and journal_state["base"] == base
                and journal_state["notes_commit"] == p_commit)
        before_landing = (
            first is True and p_commit is not None
            and [path.name for path in pending_markdown(primary)]
            == ["0002-to-opus.md", "0003-to-daemon.md"]
            and len(launches) == 1
            and launches[0][0].resolve() == primary.resolve()
            and git(root, "rev-parse", "HEAD").stdout.strip() == base
            and journal_retained)
        pending_barrier, _pending_error = (
            daemon.acquire_positive_cycle_exit_barrier(
                backlog_outcome=True))
        pending_exit_blocked = pending_barrier is None
        if pending_barrier is not None:
            daemon.release_cycle_completion_barrier(
                lock_file=pending_barrier)
        daemon.push_exact_landing_or_record_debt = (
            lambda landing: (False, "scratch has no remote"))
        note_request = mailbox / "0003-to-daemon.md"
        landed = daemon.consume_daemon_message(
            path=str(note_request), dry_run=False)
        all_heads = {
            git(checkout, "rev-parse", "HEAD").stdout.strip()
            for checkout in (root, primary, implementer, sol)}
        after_landing = (
            landed is True and all_heads == {p_commit}
            and opus.is_file()
            and (mailbox / "done" / "0003-to-daemon.md").is_file()
            and not journal.exists())
        final_barrier, _final_error = (
            daemon.acquire_positive_cycle_exit_barrier(
                backlog_outcome=True))
        finite_exit_ready = final_barrier is not None
        if final_barrier is not None:
            daemon.release_cycle_completion_barrier(
                lock_file=final_barrier)
        passed = (before_landing and pending_exit_blocked
                  and after_landing and finite_exit_ready)
        print("exclusive permanent-note B-to-P landing=" + str(passed))
        return passed


def arm_permanent_note_journal_restart_is_exact(source=None):
    """Restart accepts validated B/P evidence and refuses missing/started."""

    def prepare(mode, admin_state="done"):
        context = scratch_repository(source=source)
        root = context.__enter__()
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            context.__exit__(None, None, None)
            return None
        primary = default_primary(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        mailbox = primary / "ai" / "notes" / "mailbox"
        state_dir = mailbox / admin_state
        state_dir.mkdir(parents=True, exist_ok=True)
        admin_name = "0001-to-fable.md"
        admin_message = daemon.architect_notes_admin_payload(
            "Record one restart-safe scratch policy update.")
        (state_dir / admin_name).write_text(
            admin_message, encoding="utf-8", newline="")
        base = git(root, "rev-parse", "HEAD").stdout.strip()
        note = primary / "ai" / "notes" / "MEMORY.md"
        note.write_text(
            note.read_text(encoding="utf-8")
            + "\nRestart-safe scratch policy.\n",
            encoding="utf-8", newline="")
        git(primary, "add", "ai/notes/MEMORY.md")
        git(primary, "commit", "-m", "scratch restart note P")
        notes_commit = git(primary, "rev-parse", "HEAD").stdout.strip()
        go = mailbox / "0002-to-daemon.md"
        go.write_text(
            daemon.architect_notes_go_request_payload(
                base_commit=base, notes_commit=notes_commit),
            encoding="utf-8", newline="")
        if mode == "validated-commit":
            daemon.write_architect_notes_admin_journal(
                request_name=admin_name, request_message=admin_message,
                base_commit=base, phase=mode,
                notes_commit=notes_commit,
                receipt_sha256=hashlib.sha256(go.read_bytes()).hexdigest())
        elif mode == "started":
            daemon.write_architect_notes_admin_journal(
                request_name=admin_name, request_message=admin_message,
                base_commit=base, phase=mode)
        return (context, root, primary, daemon, base, notes_commit,
                admin_name, go)

    positive = prepare(mode="validated-commit", admin_state="inflight")
    if positive is None:
        return False
    (context, root, primary, daemon, _base, notes_commit,
     admin_name, _go) = positive
    try:
        rc, stdout, stderr = invoke(root, ["--once"])
        journal = Path(daemon.architect_notes_admin_journal_path(
            request_name=admin_name))
        heads = {
            git(checkout, "rev-parse", "HEAD").stdout.strip()
            for checkout in (root, primary, default_implementer(root),
                             default_sol(root))}
        positive_ok = (
            rc == 0 and stderr == "" and heads == {notes_commit}
            and (primary / "ai" / "notes" / "mailbox" / "done"
                 / admin_name).is_file()
            and (primary / "ai" / "notes" / "mailbox" / "done"
                 / "0002-to-daemon.md").is_file()
            and not journal.exists()
            and "dispatching " + admin_name not in stdout)
        second_rc, second_stdout, second_stderr = invoke(root, ["--once"])
        positive_ok = (
            positive_ok and second_rc == 0 and second_stderr == ""
            and "dispatching " not in second_stdout
            and not journal.exists())
    finally:
        context.__exit__(None, None, None)

    negative_results = []
    for case in ("started", "missing", "no-go"):
        mode = "started" if case == "no-go" else case
        prepared = prepare(mode=mode, admin_state="inflight")
        if prepared is None:
            negative_results.append(False)
            continue
        (context, root, primary, daemon, base, notes_commit,
         admin_name, go) = prepared
        try:
            if case == "no-go":
                go.unlink()
            before = tree_snapshot(primary / "ai" / "notes")
            rc, stdout, _stderr = invoke(root, ["--once"])
            negative_results.append(
                rc == 1
                and git(root, "rev-parse", "HEAD").stdout.strip() == base
                and git(primary, "rev-parse", "HEAD").stdout.strip()
                == notes_commit
                and (go.is_file() if case != "no-go" else not go.exists())
                and (primary / "ai" / "notes" / "mailbox" / "inflight"
                     / admin_name).is_file()
                and tree_snapshot(primary / "ai" / "notes") == before
                and "dispatching " + admin_name not in stdout)
        finally:
            context.__exit__(None, None, None)

    noop_results = []
    for initial_state in ("inflight", "done"):
        noop_context = scratch_repository(source=source)
        noop_root = noop_context.__enter__()
        try:
            noop_rc, _noop_stdout, noop_stderr = invoke(
                noop_root, ["--once"])
            if (noop_rc != 0 or noop_stderr != ""
                    or not validate_topology(noop_root)):
                noop_results.append(False)
                continue
            noop_primary = default_primary(noop_root)
            noop_daemon = load_scratch_daemon(noop_primary)
            noop_daemon.ensure_primary_execution(
                live_action=True, dry_run=False)
            noop_mailbox = noop_primary / "ai" / "notes" / "mailbox"
            request_dir = noop_mailbox / initial_state
            request_dir.mkdir(parents=True, exist_ok=True)
            noop_name = "0001-to-fable.md"
            noop_message = noop_daemon.architect_notes_admin_payload(
                "Check the saved notes and make no change.")
            (request_dir / noop_name).write_text(
                noop_message, encoding="utf-8", newline="")
            noop_base = git(
                noop_root, "rev-parse", "HEAD").stdout.strip()
            noop_daemon.write_architect_notes_admin_journal(
                request_name=noop_name, request_message=noop_message,
                base_commit=noop_base, phase="validated-noop")
            noop_journal = Path(
                noop_daemon.architect_notes_admin_journal_path(
                    request_name=noop_name))
            marker = noop_root / ("later-main-" + initial_state + ".txt")
            marker.write_text(
                "ordinary later landing\n", encoding="utf-8", newline="")
            git(noop_root, "add", marker.name)
            git(noop_root, "commit", "-m", "later clean landing")
            later_main = git(
                noop_root, "rev-parse", "HEAD").stdout.strip()
            noop_rc, noop_stdout, noop_stderr = invoke(
                noop_root, ["--once"])
            noop_results.append(
                noop_rc == 0 and noop_stderr == ""
                and git(
                    noop_primary, "rev-parse", "HEAD").stdout.strip()
                == later_main
                and (noop_mailbox / "done" / noop_name).is_file()
                and not noop_journal.exists()
                and "dispatching " + noop_name not in noop_stdout)
        finally:
            noop_context.__exit__(None, None, None)
    noop_refusals = []
    for refusal in ("unrelated-base", "dirty-primary"):
        noop_context = scratch_repository(source=source)
        noop_root = noop_context.__enter__()
        try:
            noop_rc, _noop_stdout, noop_stderr = invoke(
                noop_root, ["--once"])
            if (noop_rc != 0 or noop_stderr != ""
                    or not validate_topology(noop_root)):
                noop_refusals.append(False)
                continue
            noop_primary = default_primary(noop_root)
            noop_daemon = load_scratch_daemon(noop_primary)
            noop_daemon.ensure_primary_execution(
                live_action=True, dry_run=False)
            noop_done = noop_primary / "ai" / "notes" / "mailbox" / "done"
            noop_done.mkdir(parents=True, exist_ok=True)
            noop_name = "0001-to-fable.md"
            noop_message = noop_daemon.architect_notes_admin_payload(
                "Check the saved notes and make no change.")
            noop_request = noop_done / noop_name
            noop_request.write_text(
                noop_message, encoding="utf-8", newline="")
            noop_base = git(
                noop_root, "rev-parse", "HEAD").stdout.strip()
            if refusal == "unrelated-base":
                tree = git(
                    noop_root, "rev-parse", "HEAD^{tree}").stdout.strip()
                noop_base = git(
                    noop_root, "commit-tree", tree, "-m",
                    "unrelated no-op base").stdout.strip()
            noop_daemon.write_architect_notes_admin_journal(
                request_name=noop_name, request_message=noop_message,
                base_commit=noop_base, phase="validated-noop")
            noop_journal = Path(
                noop_daemon.architect_notes_admin_journal_path(
                    request_name=noop_name))
            if refusal == "dirty-primary":
                (noop_primary / "untracked-admin-work.txt").write_text(
                    "preserve this work\n", encoding="utf-8", newline="")
            request_before = noop_request.read_bytes()
            journal_before = noop_journal.read_bytes()
            refused = False
            try:
                noop_daemon.reconcile_architect_notes_admin_journals()
            except noop_daemon.TicketCycleStateError:
                refused = True
            noop_refusals.append(
                refused and noop_request.read_bytes() == request_before
                and noop_journal.read_bytes() == journal_before)
        finally:
            noop_context.__exit__(None, None, None)
    noop_ok = all(noop_results) and all(noop_refusals)

    prelaunch_results = []
    for phase in ("missing", "started"):
        pre_context = scratch_repository(source=source)
        pre_root = pre_context.__enter__()
        try:
            pre_rc, _pre_stdout, pre_stderr = invoke(
                pre_root, ["--once"])
            if (pre_rc != 0 or pre_stderr != ""
                    or not validate_topology(pre_root)):
                prelaunch_results.append(False)
                continue
            pre_primary = default_primary(pre_root)
            pre_daemon = load_scratch_daemon(pre_primary)
            pre_daemon.ensure_primary_execution(
                live_action=True, dry_run=False)
            pre_mailbox = pre_primary / "ai" / "notes" / "mailbox"
            pre_inflight = pre_mailbox / "inflight"
            pre_inflight.mkdir(parents=True, exist_ok=True)
            pre_name = "0001-to-fable.md"
            pre_message = pre_daemon.architect_notes_admin_payload(
                "Exercise an interrupted prelaunch admin.")
            pre_path = pre_inflight / pre_name
            pre_path.write_text(
                pre_message, encoding="utf-8", newline="")
            pre_base = git(
                pre_root, "rev-parse", "HEAD").stdout.strip()
            if phase == "started":
                pre_daemon.write_architect_notes_admin_journal(
                    request_name=pre_name, request_message=pre_message,
                    base_commit=pre_base, phase="started")
            request_before = pre_path.read_bytes()
            journal_path = Path(
                pre_daemon.architect_notes_admin_journal_path(
                    request_name=pre_name))
            journal_before = (
                journal_path.read_bytes() if journal_path.is_file()
                else None)
            stopped_rc, stopped_stdout, _stopped_stderr = invoke(
                pre_root, ["--once"])
            prelaunch_results.append(
                stopped_rc == 1 and pre_path.is_file()
                and git(pre_root, "rev-parse", "HEAD").stdout.strip()
                == pre_base
                and git(pre_primary, "rev-parse", "HEAD").stdout.strip()
                == pre_base
                and pre_path.read_bytes() == request_before
                and ((journal_path.read_bytes()
                      if journal_path.is_file() else None)
                     == journal_before)
                and "dispatching " + pre_name not in stopped_stdout)
        finally:
            pre_context.__exit__(None, None, None)
    passed = (positive_ok and all(negative_results) and noop_ok
              and all(prelaunch_results))
    print("permanent-note journal restart=" + str(passed))
    if not passed:
        print("journal restart checks=" + repr({
            "positive": positive_ok,
            "ahead-negatives": negative_results,
            "validated-noop": noop_ok,
            "prelaunch-negatives": prelaunch_results,
        }))
    return passed


def arm_permanent_note_admin_publisher_is_architect_bound(source=None):
    """Only a primary-bound Architect can publish the raw admin envelope."""
    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        names = ("MAILBOX_ROLE", "MAILBOX_PRIMARY_WORKTREE",
                 "MAILBOX_SHARED_NOTES")
        saved = {name: os.environ.get(name) for name in names}
        try:
            os.environ["MAILBOX_ROLE"] = "implementer"
            os.environ["MAILBOX_PRIMARY_WORKTREE"] = str(primary)
            os.environ["MAILBOX_SHARED_NOTES"] = str(
                primary / "ai" / "notes")
            wrong_role = not daemon.send_architect_notes_admin(
                text="Wrong role must fail.")
            os.environ["MAILBOX_ROLE"] = "architect"
            os.environ["MAILBOX_PRIMARY_WORKTREE"] = str(root)
            wrong_runtime = not daemon.send_architect_notes_admin(
                text="Wrong checkout must fail.")
            os.environ["MAILBOX_PRIMARY_WORKTREE"] = str(primary)
            accepted = daemon.send_architect_notes_admin(
                text="Update the permanent scratch policy.")
            duplicate = not daemon.send_architect_notes_admin(
                text="A second update must wait.")
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        messages = pending_markdown(primary)
        exact = (len(messages) == 1
                 and messages[0].read_text(encoding="utf-8")
                 == daemon.architect_notes_admin_payload(
                     "Update the permanent scratch policy."))
        passed = (wrong_role and wrong_runtime and accepted and duplicate
                  and exact)
        print("Architect-bound permanent-note publisher=" + str(passed))
        return passed


def arm_persistent_roles_refuse_tracked_and_untracked_source_edits(
        source=None):
    """Architect and Red Team may not create persistent source files."""

    class PopenProxy:
        def __init__(self, module, callback):
            self.module = module
            self.callback = callback

        def __getattr__(self, name):
            return (self.popen if name == "Popen"
                    else getattr(self.module, name))

        def popen(self, command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            return MutatingProcess(callback=self.callback)

    class MutatingProcess:
        def __init__(self, callback):
            self.callback = callback
            self.returncode = None
            self.called = False

        def poll(self):
            if not self.called:
                self.called = True
                self.callback()
                self.returncode = 0
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self):
            return self.returncode

    def prepare_fable(root):
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return None
        primary = default_primary(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        daemon.AGENT_COMMANDS = daemon.build_agent_commands(
            fable_effort=daemon.DEFAULT_FABLE_EFFORT,
            opus_effort=daemon.DEFAULT_OPUS_EFFORT,
            sol_effort=daemon.DEFAULT_SOL_EFFORT,
            sol_context_budget=daemon.DEFAULT_SOL_CONTEXT_BUDGET,
            sol_worktree=daemon.AGENT_CWD["sol"],
            shared_notes=daemon.ACTIVE_TOPOLOGY["shared_notes"])
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        message = mailbox / "0001-to-fable.md"
        message.write_text(
            daemon.architect_user_request_payload(
                text="Coordinate the persistent role witness.",
                discovery_severity="medium"),
            encoding="utf-8", newline="")
        return daemon, primary, message

    def state(primary, agent):
        mailbox = primary / "ai" / "notes" / "mailbox"
        return {
            key: len(list(((mailbox if key == "pending" else mailbox / key)
                           ).glob("*-to-" + agent + ".md")))
            for key in ("pending", "inflight", "done", "failed")}

    outcomes = []
    with scratch_repository(source=source) as root:
        prepared = prepare_fable(root)
        if prepared is None:
            return False
        daemon, primary, message = prepared
        ordinary = primary / "ai" / "tools" / "mailbox_daemon.py"

        def edit_ordinary():
            ordinary.write_bytes(ordinary.read_bytes() + b"\n# role drift\n")

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, edit_ordinary)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        refused_preserved = (
            result is False and ordinary.read_bytes().endswith(b"role drift\n")
            and state(primary, "fable")
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(refused_preserved)
        print("Architect ordinary tracked edit preserved and refused="
              + str(refused_preserved))

    with scratch_repository(source=source) as root:
        prepared = prepare_fable(root)
        if prepared is None:
            return False
        daemon, primary, message = prepared
        created = primary / "emulator" / "architect_created.py"

        def create_architect_source():
            created.parent.mkdir(parents=True, exist_ok=True)
            created.write_text(
                "raise RuntimeError('Architect must not implement')\n",
                encoding="utf-8", newline="")

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, create_architect_source)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        refused_untracked_architect = (
            result is False and created.is_file()
            and state(primary, "fable")
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(refused_untracked_architect)
        print("Architect untracked source creation preserved and refused="
              + str(refused_untracked_architect))

    with scratch_repository(source=source) as root:
        prepared = prepare_fable(root)
        if prepared is None:
            return False
        daemon, primary, message = prepared
        note = primary / "ai" / "notes" / "MEMORY.md"

        def edit_note():
            note.write_bytes(note.read_bytes() + b"\nArchitect policy note.\n")

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, edit_note)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        refused = (
            result is False and note.read_bytes().endswith(
                b"Architect policy note.\n")
            and state(primary, "fable")
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(refused)
        print("uncommitted permanent-note edit preserved and refused="
              + str(refused))

    with scratch_repository(source=source) as root:
        prepared = prepare_fable(root)
        if prepared is None:
            return False
        daemon, primary, message = prepared
        message.write_text(
            daemon.architect_notes_admin_payload(
                "Exercise an uncommitted role edit."),
            encoding="utf-8", newline="")
        request_identity = file_identity(message)
        base = git(primary, "rev-parse", "HEAD").stdout.strip()
        role = primary / ".claude" / "FABLE_ROLE.md"
        dirty = role.read_bytes() + b"\nUncommitted scratch role rule.\n"

        def edit_role():
            role.write_bytes(dirty)

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, edit_role)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        failed = primary / "ai" / "notes" / "mailbox" / "failed" / message.name
        refused_role = (
            result is False
            and git(primary, "rev-parse", "HEAD").stdout.strip() == base
            and role.read_bytes() == dirty
            and file_identity(failed) == request_identity
            and state(primary, "fable")
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(refused_role)
        print("uncommitted protected-role edit preserved and refused="
              + str(refused_role))

    with scratch_repository(source=source) as root:
        prepared = prepare_fable(root)
        if prepared is None:
            return False
        daemon, primary, message = prepared
        ordinary = primary / "ai" / "tools" / "mailbox_daemon.py"
        ordinary.write_bytes(ordinary.read_bytes() + b"\n# existing work\n")
        before = ordinary.read_bytes()
        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, lambda: None)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        baseline_preserved = (
            result is True and ordinary.read_bytes() == before
            and state(primary, "fable")
            == {"pending": 0, "inflight": 0, "done": 1, "failed": 0})
        outcomes.append(baseline_preserved)
        print("Architect preexisting ordinary state preserved="
              + str(baseline_preserved))

    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        sol = default_sol(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        daemon.AGENT_COMMANDS = daemon.build_agent_commands(
            fable_effort=daemon.DEFAULT_FABLE_EFFORT,
            opus_effort=daemon.DEFAULT_OPUS_EFFORT,
            sol_effort=daemon.DEFAULT_SOL_EFFORT,
            sol_context_budget=daemon.DEFAULT_SOL_CONTEXT_BUDGET,
            sol_worktree=daemon.AGENT_CWD["sol"],
            shared_notes=daemon.ACTIVE_TOPOLOGY["shared_notes"])
        create_empty_sealed_backlog(primary=primary)
        if not daemon.send(
                agent="sol", ticket_kind="discovery", severity="medium",
                scope="bounded", text="Review persistent Sol authority.",
                dry_run=False):
            return False
        message = pending_markdown(primary)[0]
        ordinary = sol / "ai" / "tools" / "mailbox_daemon.py"

        def edit_sol():
            ordinary.write_bytes(ordinary.read_bytes() + b"\n# Sol drift\n")

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, edit_sol)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        refused_sol = (
            result is False and ordinary.read_bytes().endswith(b"Sol drift\n")
            and state(primary, "sol")
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(refused_sol)
        print("Red Team tracked edit preserved and refused="
              + str(refused_sol))

    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        sol = default_sol(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        daemon.AGENT_COMMANDS = daemon.build_agent_commands(
            fable_effort=daemon.DEFAULT_FABLE_EFFORT,
            opus_effort=daemon.DEFAULT_OPUS_EFFORT,
            sol_effort=daemon.DEFAULT_SOL_EFFORT,
            sol_context_budget=daemon.DEFAULT_SOL_CONTEXT_BUDGET,
            sol_worktree=daemon.AGENT_CWD["sol"],
            shared_notes=daemon.ACTIVE_TOPOLOGY["shared_notes"])
        create_empty_sealed_backlog(primary=primary)
        if not daemon.send(
                agent="sol", ticket_kind="discovery", severity="medium",
                scope="bounded", text="Try creating Red Team source.",
                dry_run=False):
            return False
        message = pending_markdown(primary)[0]
        created = sol / "emulator" / "redteam_created.py"

        def create_redteam_source():
            created.parent.mkdir(parents=True, exist_ok=True)
            created.write_text(
                "raise RuntimeError('Red Team is advisory')\n",
                encoding="utf-8", newline="")

        original = daemon.subprocess
        daemon.subprocess = PopenProxy(original, create_redteam_source)
        try:
            result = daemon.dispatch(path=str(message), dry_run=False)
        finally:
            daemon.subprocess = original
        refused_untracked_sol = (
            result is False and created.is_file()
            and state(primary, "sol")
            == {"pending": 0, "inflight": 0, "done": 0, "failed": 1})
        outcomes.append(refused_untracked_sol)
        print("Red Team untracked source creation preserved and refused="
              + str(refused_untracked_sol))
    return all(outcomes)


def arm_shared_protected_notes_require_architect_authority(source=None):
    """Opus/Sol may write temporary notes, not unsealed protected files."""
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        notes = primary / "ai" / "notes"
        backlog = notes / "backlog.md"
        guard_state = notes / ".backlog-guard.json"
        permanent = notes / "MEMORY.md"
        backlog.write_bytes(b"")
        seal_backlog(primary)

        temporary = notes / "ticket-finding.md"
        opus_temporary = daemon.capture_persistent_role_state(agent="opus")
        sol_temporary = daemon.capture_persistent_role_state(agent="sol")
        temporary.write_text(
            "Temporary finding evidence.\n", encoding="utf-8", newline="")
        temporary_allowed = True
        try:
            daemon.recheck_persistent_role_state(proof=opus_temporary)
            daemon.recheck_persistent_role_state(proof=sol_temporary)
        except daemon.PrimaryWorktreeError:
            temporary_allowed = False

        opus_extra = daemon.capture_persistent_role_state(agent="opus")
        sol_extra = daemon.capture_persistent_role_state(agent="sol")
        extra_note = notes / "accidental-twelfth.md"
        extra_note.write_text(
            "Accidentally staged note.\n", encoding="utf-8", newline="")
        git(primary, "add", "-f", "ai/notes/accidental-twelfth.md")
        extra_refused = []
        for proof in (opus_extra, sol_extra):
            refused = False
            try:
                daemon.recheck_persistent_role_state(proof=proof)
            except daemon.PrimaryWorktreeError:
                refused = True
            extra_refused.append(refused)
        git(primary, "reset", "HEAD", "--", "ai/notes/accidental-twelfth.md")
        extra_note.unlink()

        unsealed_refusals = []
        for agent in ("opus", "sol"):
            proof = daemon.capture_persistent_role_state(agent=agent)
            original = backlog.read_bytes()
            backlog.write_bytes(original + b"unsealed accidental edit\n")
            refused = False
            try:
                daemon.recheck_persistent_role_state(proof=proof)
            except daemon.PrimaryWorktreeError:
                refused = True
            backlog.write_bytes(original)
            seal_backlog(primary)
            unsealed_refusals.append(refused)

        permanent_refusals = []
        for agent in ("opus", "sol"):
            proof = daemon.capture_persistent_role_state(agent=agent)
            original = permanent.read_bytes()
            permanent.write_bytes(original + b"accidental policy edit\n")
            refused = False
            try:
                daemon.recheck_persistent_role_state(proof=proof)
            except daemon.PrimaryWorktreeError:
                refused = True
            permanent.write_bytes(original)
            permanent_refusals.append(refused)

        opus_concurrent = daemon.capture_persistent_role_state(agent="opus")
        sol_concurrent = daemon.capture_persistent_role_state(agent="sol")
        backlog.write_bytes(backlog.read_bytes() + b"Architect ticket edit\n")
        seal_backlog(primary)
        permanent.write_bytes(
            permanent.read_bytes() + b"Architect committed policy edit.\n")
        git(primary, "add", "ai/notes/MEMORY.md")
        git(primary, "commit", "-m", "Architect policy bookkeeping")
        concurrent_allowed = True
        try:
            daemon.recheck_persistent_role_state(proof=opus_concurrent)
            daemon.recheck_persistent_role_state(proof=sol_concurrent)
        except daemon.PrimaryWorktreeError:
            concurrent_allowed = False

        backlog.unlink()
        guard_state.unlink()
        absent_allowed = True
        try:
            for agent in ("opus", "sol"):
                proof = daemon.capture_persistent_role_state(agent=agent)
                daemon.recheck_persistent_role_state(proof=proof)
        except daemon.PrimaryWorktreeError:
            absent_allowed = False

        checks = {
            "temporary-note-allowed": temporary_allowed,
            "opus-extra-index-note-refused": extra_refused[0],
            "sol-extra-index-note-refused": extra_refused[1],
            "opus-unsealed-backlog-refused": unsealed_refusals[0],
            "sol-unsealed-backlog-refused": unsealed_refusals[1],
            "opus-permanent-note-refused": permanent_refusals[0],
            "sol-permanent-note-refused": permanent_refusals[1],
            "concurrent-architect-authority-allowed": concurrent_allowed,
            "both-absent-bootstrap-allowed": absent_allowed,
        }
        passed = all(checks.values())
        print("shared protected-note authority=" + str(passed))
        if not passed:
            print("shared protected-note checks=" + repr(checks))
        return passed


def arm_architect_receipt_binds_candidate_to_squash_landing(source=None):
    """Candidate C is audited; daemon GO creates and records exact L."""
    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)

        base = git(implementer, "rev-parse", "HEAD").stdout.strip()
        cycle_id = "squash-landing@" + base
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — "
            "[Squash landing](#squash-landing)\n\n"
            "<a id=\"squash-landing\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)
        flow = (
            "MAILBOX-FLOW: ticket\n"
            "MAILBOX-CYCLE: " + cycle_id + "\n"
            "MAILBOX-MODE: normal\n\n"
            "Implement the exact scratch landing candidate.\n")
        daemon.register_ticket_cycle_message(agent="opus", message=flow)
        starting = daemon.prepare_implementer_cycle_checkout(
            cycle_id=cycle_id)
        candidate_file = implementer / "candidate-change.txt"
        candidate_file.write_text(
            "candidate change\n", encoding="utf-8", newline="")
        git(implementer, "add", candidate_file.name)
        candidate_tree = git(
            implementer, "write-tree").stdout.strip()
        candidate_message_input = (
            "Add the accepted candidate file\n\n"
            "The landing reproduction needs one visible file change.\r\n"
            "The UTF-8 word café must remain unchanged.\r\n"
            "\r\n"
            "Evidence:\n\n"
            "- The reproduction checks the exact tree and raw message.\n\n\n"
        ).encode("utf-8")
        candidate = git_bytes(
            implementer, "commit-tree", candidate_tree, "-p", base,
            "-F", "-", input_bytes=candidate_message_input
        ).stdout.decode("ascii").strip()
        git(implementer, "update-ref", "HEAD", candidate, base)
        candidate = daemon.record_implementer_candidate(
            cycle_id=cycle_id, starting_head=starting)
        if candidate is None:
            return False
        mailbox = Path(daemon.MAILBOX)
        mailbox.mkdir(parents=True, exist_ok=True)
        wrong_mode_go = mailbox / "0001-to-daemon.md"
        wrong_mode_go.write_text(
            daemon.architect_go_request_payload(
                cycle_id=cycle_id, candidate_commit=candidate,
                mode="two-role"),
            encoding="utf-8", newline="")
        wrong_mode_refused = not daemon.consume_daemon_message(
            path=str(wrong_mode_go))
        open_go = mailbox / "0002-to-daemon.md"
        open_go.write_text(
            daemon.architect_go_request_payload(
                cycle_id=cycle_id, candidate_commit=candidate,
                mode="normal"),
            encoding="utf-8", newline="")
        open_go_deferred = daemon.consume_daemon_message(path=str(open_go))
        corrections = list(mailbox.glob("*-to-fable.md"))
        open_go_recovered = (
            open_go_deferred and len(corrections) == 1
            and daemon.BACKLOG_CLOSE_REQUIRED_HEADER
            in corrections[0].read_text(encoding="utf-8")
            and (mailbox / "done" / open_go.name).is_file())
        # This arm supplies the corrected fresh GO below. Remove the synthetic
        # Architect request so it cannot outlive the cycle under test.
        for correction in corrections:
            correction.unlink()
        open_candidate_preserved = (
            daemon.candidate_commit_for_cycle(cycle_id) == candidate
            and git(root, "rev-parse", "HEAD").stdout.strip() == base)
        close_backlog_ticket(primary=primary, anchor="squash-landing")
        candidate_object = git_bytes(
            implementer, "cat-file", "commit", candidate).stdout
        candidate_message_raw = candidate_object.partition(b"\n\n")[2]
        reserved_message = (
            b"Copy a prior landing message\n\n"
            b"mailbox-cycle : accidental-copy\n")
        reserved_candidate = git_bytes(
            implementer, "commit-tree", candidate_tree, "-p", base,
            "-F", "-", input_bytes=reserved_message
        ).stdout.decode("ascii").strip()
        reserved_message_refused = False
        try:
            daemon._landing_commit_message(
                cycle_id=cycle_id, candidate_commit=reserved_candidate)
        except daemon.TicketCycleStateError:
            reserved_message_refused = True
        architect_snapshot = Path(daemon.create_audit_snapshot(
            cycle_id=cycle_id, commit=candidate, agent="fable"))

        intervening_file = root / "intervening-main.txt"
        intervening_file.write_text(
            "main advanced independently\n", encoding="utf-8", newline="")
        git(root, "add", intervening_file.name)
        git(root, "commit", "-m", "intervening main change")
        landing_parent = git(root, "rev-parse", "HEAD").stdout.strip()
        expected_tree = git(
            root, "merge-tree", "--write-tree", landing_parent,
            candidate).stdout.strip()
        expected_tree = daemon._tree_with_backlog(
            expected_tree,
            daemon._validate_sealed_backlog(primary_worktree=str(primary)))
        git(root, "reset", "--hard", candidate)
        candidate_as_landing_refused = False
        try:
            daemon.record_architect_commit(
                cycle_id=cycle_id, accepted_commit=candidate, mode="normal")
        except daemon.TicketCycleStateError:
            candidate_as_landing_refused = True
        git(root, "reset", "--hard", landing_parent)
        wrong_file = root / "wrong-landing-tree.txt"
        wrong_file.write_text(
            "not candidate content\n", encoding="utf-8", newline="")
        git(root, "add", wrong_file.name)
        wrong_tree = git(root, "write-tree").stdout.strip()
        expected_landing_message = (
            candidate_message_raw
            + b"Mailbox-Cycle: " + cycle_id.encode("utf-8") + b"\n"
            + b"Mailbox-Candidate: " + candidate.encode("ascii") + b"\n")
        wrong_landing = git_bytes(
            root, "commit-tree", wrong_tree, "-p", landing_parent,
            "-F", "-", input_bytes=expected_landing_message
        ).stdout.decode("ascii").strip()
        git(root, "reset", "--hard", landing_parent)
        wrong_tree_refused = False
        try:
            daemon._verify_prepared_landing(
                cycle_id=cycle_id, candidate_commit=candidate,
                landing_commit=wrong_landing)
        except daemon.TicketCycleStateError:
            wrong_tree_refused = True
        wrong_human_message = (
            b"Replace the reviewed explanation\n\n"
            b"This body was not present in candidate C.\n\n")
        wrong_message_landing = git_bytes(
            root, "commit-tree", expected_tree, "-p", landing_parent,
            "-F", "-", input_bytes=(
                wrong_human_message
                + b"Mailbox-Cycle: " + cycle_id.encode("utf-8") + b"\n"
                + b"Mailbox-Candidate: " + candidate.encode("ascii")
                + b"\n")
        ).stdout.decode("ascii").strip()
        wrong_message_refused = False
        try:
            landing_ref = daemon.cycle_landing_ref(cycle_id=cycle_id)
            git(root, "update-ref", landing_ref, wrong_message_landing)
            daemon.prepare_exact_squash_landing(
                cycle_id=cycle_id, candidate_commit=candidate,
                mode="normal")
        except daemon.TicketCycleStateError:
            wrong_message_refused = True
        finally:
            git(root, "update-ref", "-d", landing_ref,
                wrong_message_landing, check=False)
        go_path = mailbox / "0003-to-daemon.md"
        go_path.write_text(
            daemon.architect_go_request_payload(
                cycle_id=cycle_id, candidate_commit=candidate,
                mode="normal"),
            encoding="utf-8", newline="")
        real_retire = daemon.retire_superseded_failed_architect_go
        try:
            daemon.retire_superseded_failed_architect_go = (
                lambda **_kwargs: None)
            consumed = daemon.consume_daemon_message(path=str(go_path))
        finally:
            daemon.retire_superseded_failed_architect_go = real_retire
        old_go_remained_archived = (
            mailbox / "done" / open_go.name).is_file()
        landing = git(root, "rev-parse", "HEAD").stdout.strip()
        landing_object = git_bytes(
            root, "cat-file", "commit", landing).stdout
        landing_message_raw = landing_object.partition(b"\n\n")[2]
        sol_snapshot = Path(daemon.create_audit_snapshot(
            cycle_id=cycle_id, commit=landing, agent="sol"))
        ticket_state = daemon.read_ticket_cycle_state()
        candidate_ref = daemon.cycle_candidate_ref(cycle_id=cycle_id)
        landing_ref = daemon.cycle_landing_ref(cycle_id=cycle_id)
        closures = []
        for path in mailbox.rglob("*-to-sol.md"):
            text = path.read_text(encoding="utf-8")
            if daemon.redteam_closure_ticket(text) == cycle_id:
                closures.append(path)
        recovery_one = daemon.reconcile_ticket_cycle_state()
        recovery_two = daemon.reconcile_ticket_cycle_state()
        closures_after_recovery = []
        for path in mailbox.rglob("*-to-sol.md"):
            text = path.read_text(encoding="utf-8")
            if daemon.redteam_closure_ticket(text) == cycle_id:
                closures_after_recovery.append(path)
        real_run = daemon.subprocess.run
        verification_timeout_became_debt = False
        try:
            def timeout_remote_verification(command, *args, **kwargs):
                if "push" in command:
                    return subprocess.CompletedProcess(
                        command, 0, stdout=b"push accepted\n", stderr=b"")
                if "ls-remote" in command:
                    raise subprocess.TimeoutExpired(command, 120)
                return real_run(command, *args, **kwargs)

            daemon.subprocess.run = timeout_remote_verification
            pushed, _detail = daemon.push_exact_landing_or_record_debt(
                landing=landing)
            verification_timeout_became_debt = (
                not pushed and Path(
                    daemon._push_debt_path(landing=landing)).is_file())
        finally:
            daemon.subprocess.run = real_run
        checks = {
            "open-go-recovery-queued": open_go_recovered,
            "wrong-mode-go-remains-explicit-debt": (
                wrong_mode_refused
                and (mailbox / "failed" / wrong_mode_go.name).is_file()),
            "old-go-archived-after-fresh-go": (
                old_go_remained_archived
                and not (mailbox / "failed" / open_go.name).exists()
                and (mailbox / "done" / open_go.name).is_file()),
            "open-candidate-preserved": open_candidate_preserved,
            "go-consumed": consumed,
            "candidate-is-not-landing": candidate_as_landing_refused,
            "wrong-tree-is-not-landing": wrong_tree_refused,
            "wrong-message-is-not-landing": wrong_message_refused,
            "reserved-message-is-refused": reserved_message_refused,
            "candidate-message-is-raw-input": (
                candidate_message_raw == candidate_message_input),
            "candidate-audited": git(
                architect_snapshot, "rev-parse", "HEAD").stdout.strip()
            == candidate,
            "candidate-detached": git(
                architect_snapshot, "symbolic-ref", "-q", "HEAD",
                check=False).returncode == 1,
            "landing-distinct": landing != candidate,
            "landing-parent": git(
                root, "rev-parse", landing + "^").stdout.strip()
            == landing_parent,
            "base-preserved": git(
                root, "merge-base", "--is-ancestor", base,
                landing_parent, check=False).returncode == 0,
            "exact-squash-tree": git(
                root, "rev-parse", landing + "^{tree}").stdout.strip()
            == expected_tree,
            "candidate-message-preserved": (
                landing_message_raw == expected_landing_message),
            "intervening-preserved": (
                root / intervening_file.name).read_bytes()
            == b"main advanced independently\n",
            "candidate-change-landed": (
                root / candidate_file.name).read_bytes()
            == b"candidate change\n",
            "normal-waits-redteam": cycle_id not in ticket_state["completed"],
            "landing-saved": ticket_state["active"][cycle_id]["commit"]
            == landing,
            "landing-phase": ticket_state["active"][cycle_id]["phase"]
            == "committed-awaiting-closure",
            "candidate-ref-retired": git(
                root, "rev-parse", "--verify", candidate_ref,
                check=False).returncode != 0,
            "landing-ref-retired": git(
                root, "rev-parse", "--verify", landing_ref,
                check=False).returncode != 0,
            "one-sol-closure": len(closures) == 1,
            "closure-binds-landing": (
                len(closures) == 1
                and daemon.redteam_closure_commit(
                    closures[0].read_text(encoding="utf-8")) == landing),
            "push-debt-visible": Path(
                daemon._push_debt_path(landing=landing)).is_file(),
            "two-restarts-idempotent": (
                recovery_one == 0 and recovery_two == 0),
            "go-stays-done": (
                (mailbox / "done" / go_path.name).is_file()
                and not (mailbox / "failed" / go_path.name).exists()),
            "closure-stays-unique": len(closures_after_recovery) == 1,
            "remote-verification-timeout-is-debt": (
                verification_timeout_became_debt),
            "sol-detached": git(
                sol_snapshot, "symbolic-ref", "-q", "HEAD",
                check=False).returncode == 1,
            "sol-sees-landing": git(
                sol_snapshot, "rev-parse", "HEAD").stdout.strip()
            == landing,
        }
        passed = all(checks.values())
        try:
            daemon.remove_audit_snapshot(
                cycle_id=cycle_id, commit=candidate, agent="fable")
            daemon.remove_audit_snapshot(
                cycle_id=cycle_id, commit=landing, agent="sol")
        except BaseException:
            passed = False
        print("candidate C and squash landing L are distinct=" + str(passed))
        if not passed:
            print("candidate/landing checks=" + repr(checks))
        return passed


def arm_landed_candidate_hands_off_without_clobbering_pipeline(source=None):
    """A retired C yields to L, while an already-saved next C is untouched."""
    outcomes = []
    for pipeline in (False, True):
        with scratch_repository(source=source) as root:
            rc, _stdout, stderr = invoke(root, ["--once"])
            if rc != 0 or stderr != "" or not validate_topology(root):
                outcomes.append(False)
                continue
            primary = default_primary(root)
            implementer = default_implementer(root)
            daemon = load_scratch_daemon(primary)
            daemon.ensure_primary_execution(live_action=True, dry_run=False)
            base = git(implementer, "rev-parse", "HEAD").stdout.strip()
            backlog = primary / "ai" / "notes" / "backlog.md"
            backlog.write_text(
                "- OPEN **HIGH** **BUG FIX** — [Handoff A](#handoff-a)\n"
                "- OPEN **HIGH** **BUG FIX** — [Handoff B](#handoff-b)\n\n"
                "<a id=\"handoff-a\"></a>\n"
                "**Red Team reopen count: 0.**\n"
                "**Red Team reopening: allowed.**\n\n"
                "<a id=\"handoff-b\"></a>\n"
                "**Red Team reopen count: 0.**\n"
                "**Red Team reopening: allowed.**\n",
                encoding="utf-8", newline="")
            seal_backlog(primary)

            def flow(cycle_id):
                return (
                    "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle_id
                    + "\nMAILBOX-MODE: two-role\n\n"
                    "Implement the exact scratch handoff ticket.\n")

            cycle_a = "handoff-a@" + base
            daemon.register_ticket_cycle_message(
                agent="opus", message=flow(cycle_a), skip_redteam=True)
            starting_a = daemon.prepare_implementer_cycle_checkout(
                cycle_id=cycle_a)
            file_a = implementer / "handoff-a.txt"
            file_a.write_text(
                "".join("candidate A line " + str(index) + "\n"
                        for index in range(451)),
                encoding="utf-8", newline="")
            git(implementer, "add", file_a.name)
            git(implementer, "commit", "-m", "scratch handoff A")
            candidate_a = daemon.record_implementer_candidate(
                cycle_id=cycle_a, starting_head=starting_a)
            before_landing_debt = daemon.landing_debt_snapshot()
            candidate_b = None
            state_b = None
            tree_b = None
            ref_b = None
            cycle_b = "handoff-b@" + base
            if pipeline:
                daemon.register_ticket_cycle_message(
                    agent="opus", message=flow(cycle_b), skip_redteam=True)
                starting_b = daemon.prepare_implementer_cycle_checkout(
                    cycle_id=cycle_b)
                file_b = implementer / "handoff-b.txt"
                file_b.write_text(
                    "candidate B\n", encoding="utf-8", newline="")
                git(implementer, "add", file_b.name)
                git(implementer, "commit", "-m", "scratch handoff B")
                candidate_b = daemon.record_implementer_candidate(
                    cycle_id=cycle_b, starting_head=starting_b)
                ref_b = daemon.cycle_candidate_ref(cycle_id=cycle_b)
                state_b = daemon.read_candidate_state()["cycles"][cycle_b]
                tree_b = git(
                    implementer, "rev-parse", "HEAD^{tree}").stdout.strip()

            mailbox = Path(daemon.MAILBOX)
            mailbox.mkdir(parents=True, exist_ok=True)
            close_backlog_ticket(primary=primary, anchor="handoff-a")
            go = mailbox / "0003-to-daemon.md"
            go.write_text(
                daemon.architect_go_request_payload(
                    cycle_id=cycle_a, candidate_commit=candidate_a,
                    mode="two-role"),
                encoding="utf-8", newline="")
            consumed = daemon.consume_daemon_message(path=str(go))
            landing_a = git(root, "rev-parse", "HEAD").stdout.strip()
            state_after = daemon.read_candidate_state()["cycles"]
            if not pipeline:
                # A planning-role branch can legitimately lag main. It is
                # not implementation debt once C/ref/state were retired.
                git(primary, "reset", "--hard", base)
            after_landing_debt = daemon.landing_debt_snapshot()
            with contextlib.redirect_stdout(io.StringIO()):
                daemon.report_landing_debt(snapshot=after_landing_debt)
            ref_a = daemon.cycle_candidate_ref(cycle_id=cycle_a)
            checks = {
                "consumed": consumed,
                "a-state-retired": cycle_a not in state_after,
                "a-ref-retired": git(
                    root, "rev-parse", "--verify", ref_a,
                    check=False).returncode != 0,
            }
            if pipeline:
                checks.update({
                    "b-head-preserved": git(
                        implementer, "rev-parse", "HEAD").stdout.strip()
                    == candidate_b,
                    "b-tree-preserved": git(
                        implementer, "rev-parse",
                        "HEAD^{tree}").stdout.strip() == tree_b,
                    "b-state-preserved": state_after.get(cycle_b) == state_b,
                    "b-ref-preserved": git(
                        root, "rev-parse", ref_b).stdout.strip()
                    == candidate_b,
                })
                go_b = mailbox / "0005-to-daemon.md"
                close_backlog_ticket(primary=primary, anchor="handoff-b")
                go_b.write_text(
                    daemon.architect_go_request_payload(
                        cycle_id=cycle_b, candidate_commit=candidate_b,
                        mode="two-role"),
                    encoding="utf-8", newline="")
                consumed_b = daemon.consume_daemon_message(path=str(go_b))
                landing_b = git(root, "rev-parse", "HEAD").stdout.strip()
                state_after_b = daemon.read_candidate_state()["cycles"]
                checks.update({
                    "b-consumed": consumed_b,
                    "b-landing-descends-a": git(
                        root, "merge-base", "--is-ancestor",
                        landing_a, landing_b, check=False).returncode == 0,
                    "both-changes-landed": (
                        (root / file_a.name).is_file()
                        and (root / "handoff-b.txt").read_bytes()
                        == b"candidate B\n"),
                    "b-state-retired": cycle_b not in state_after_b,
                    "b-ref-retired": git(
                        root, "rev-parse", "--verify", ref_b,
                        check=False).returncode != 0,
                    "implementer-at-b-L": git(
                        implementer, "rev-parse", "HEAD").stdout.strip()
                    == landing_b,
                    "implementer-clean-after-b": git(
                        implementer, "status", "--porcelain").stdout == "",
                })
            else:
                cycle_b = "handoff-b@" + landing_a
                daemon.register_ticket_cycle_message(
                    agent="opus", message=flow(cycle_b), skip_redteam=True)
                prepared_b = daemon.prepare_implementer_cycle_checkout(
                    cycle_id=cycle_b)
                checks.update({
                    "implementer-at-L": git(
                        implementer, "rev-parse", "HEAD").stdout.strip()
                    == landing_a,
                    "next-ticket-prepared": prepared_b == landing_a,
                    "implementer-clean": git(
                        implementer, "status", "--porcelain").stdout == "",
                    "large-candidate-was-visible": (
                        before_landing_debt["available"]
                        and before_landing_debt["changed_lines"] > 400),
                    "accepted-large-L-has-no-debt": (
                        after_landing_debt["available"]
                        and after_landing_debt["changed_lines"] == 0
                        and not list(mailbox.glob("*-to-fable.md"))),
                })
            passed = all(checks.values())
            if not passed:
                print("candidate handoff checks=" + repr(checks))
            outcomes.append(passed)
    passed = outcomes == [True, True]
    print("landed candidate handoff and pipeline preservation=" + str(passed))
    return passed


def arm_candidate_retirement_internal_crash_replays(source=None):
    """Reset-before-delete and ref-before-state cuts retire idempotently."""
    outcomes = []
    for cut in ("reset-complete", "ref-deleted"):
        with scratch_repository(source=source) as root:
            rc, _stdout, stderr = invoke(root, ["--once"])
            if rc != 0 or stderr != "" or not validate_topology(root):
                outcomes.append(False)
                continue
            primary = default_primary(root)
            implementer = default_implementer(root)
            daemon = load_scratch_daemon(primary)
            daemon.ensure_primary_execution(live_action=True, dry_run=False)
            base = git(implementer, "rev-parse", "HEAD").stdout.strip()
            anchor = "retirement-" + cut
            cycle = anchor + "@" + base
            backlog = primary / "ai" / "notes" / "backlog.md"
            backlog.write_text(
                "- OPEN **HIGH** **BUG FIX** — [Retirement](#" + anchor
                + ")\n\n<a id=\"" + anchor + "\"></a>\n"
                "**Red Team reopen count: 0.**\n"
                "**Red Team reopening: allowed.**\n",
                encoding="utf-8", newline="")
            seal_backlog(primary)
            flow = (
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                + "\nMAILBOX-MODE: two-role\n\n"
                "Implement the exact crash witness.\n")
            daemon.register_ticket_cycle_message(
                agent="opus", message=flow, skip_redteam=True)
            starting = daemon.prepare_implementer_cycle_checkout(
                cycle_id=cycle)
            changed = implementer / "retirement-cut.txt"
            changed.write_text(cut + "\n", encoding="utf-8", newline="")
            git(implementer, "add", changed.name)
            git(implementer, "commit", "-m", "scratch " + cut)
            candidate = daemon.record_implementer_candidate(
                cycle_id=cycle, starting_head=starting)
            landing, parent, _journal = daemon.prepare_exact_squash_landing(
                cycle_id=cycle, candidate_commit=candidate, mode="two-role")
            daemon.land_prepared_commit_in_clean_user_checkout(
                landing=landing, parent=parent)
            daemon.record_architect_commit(
                cycle_id=cycle, accepted_commit=landing, mode="two-role")
            git(implementer, "reset", "--hard", landing)
            reference = daemon.cycle_candidate_ref(cycle_id=cycle)
            if cut == "ref-deleted":
                git(root, "update-ref", "-d", reference, candidate)
            first = daemon.retire_cycle_candidate(
                cycle_id=cycle, candidate_commit=candidate,
                landing_commit=landing, mode="two-role")
            second = daemon.retire_cycle_candidate(
                cycle_id=cycle, candidate_commit=candidate,
                landing_commit=landing, mode="two-role")
            checks = {
                "first-retired": first,
                "second-idempotent": second,
                "state-retired": (
                    cycle not in daemon.read_candidate_state()["cycles"]),
                "ref-retired": git(
                    root, "rev-parse", "--verify", reference,
                    check=False).returncode != 0,
                "implementer-at-L": git(
                    implementer, "rev-parse", "HEAD").stdout.strip()
                == landing,
                "implementer-clean": git(
                    implementer, "status", "--porcelain").stdout == "",
            }
            passed = all(checks.values())
            if not passed:
                print("candidate retirement crash checks=" + repr(checks))
            outcomes.append(passed)
    passed = outcomes == [True, True]
    print("candidate retirement internal crash replay=" + str(passed))
    return passed


def arm_architect_go_crash_cuts_recover_once(source=None):
    """Prepared, landed, recorded, and archived GO cuts replay exactly."""
    results = []
    for cut in ("prepared", "fast-forwarded", "recorded", "closure",
                "archived"):
        with scratch_repository(source=source) as root:
            rc, _stdout, stderr = invoke(root, ["--once"])
            if rc != 0 or stderr != "" or not validate_topology(root):
                results.append(False)
                continue
            primary = default_primary(root)
            implementer = default_implementer(root)
            daemon = load_scratch_daemon(primary)
            daemon.ensure_primary_execution(live_action=True, dry_run=False)
            base = git(implementer, "rev-parse", "HEAD").stdout.strip()
            anchor = "crash-" + cut
            cycle_id = anchor + "@" + base
            backlog = primary / "ai" / "notes" / "backlog.md"
            backlog.write_text(
                "- OPEN **HIGH** **BUG FIX** — [Crash cut](#" + anchor
                + ")\n\n<a id=\"" + anchor + "\"></a>\n"
                "**Red Team reopen count: 0.**\n"
                "**Red Team reopening: allowed.**\n",
                encoding="utf-8", newline="")
            seal_backlog(primary)
            flow = (
                "MAILBOX-FLOW: ticket\n"
                "MAILBOX-CYCLE: " + cycle_id + "\n"
                "MAILBOX-MODE: normal\n\n"
                "Implement the crash-cut candidate.\n")
            daemon.register_ticket_cycle_message(agent="opus", message=flow)
            starting = daemon.prepare_implementer_cycle_checkout(
                cycle_id=cycle_id)
            changed = implementer / "crash-cut.txt"
            changed.write_text(cut + "\n", encoding="utf-8", newline="")
            git(implementer, "add", changed.name)
            git(implementer, "commit", "-m", "scratch " + cut)
            candidate = daemon.record_implementer_candidate(
                cycle_id=cycle_id, starting_head=starting)
            close_backlog_ticket(primary=primary, anchor=anchor)
            mailbox = Path(daemon.MAILBOX)
            inflight = mailbox / "inflight"
            inflight.mkdir(parents=True, exist_ok=True)
            go_path = inflight / "0003-to-daemon.md"
            go_path.write_text(
                daemon.architect_go_request_payload(
                    cycle_id=cycle_id, candidate_commit=candidate,
                    mode="normal"),
                encoding="utf-8", newline="")
            landing, parent, _reference = daemon.prepare_exact_squash_landing(
                cycle_id=cycle_id, candidate_commit=candidate,
                mode="normal")
            if cut != "prepared":
                daemon.land_prepared_commit_in_clean_user_checkout(
                    landing=landing, parent=parent)
            if cut in {"recorded", "closure", "archived"}:
                daemon.record_architect_commit(
                    cycle_id=cycle_id, accepted_commit=landing,
                    mode="normal")
            if cut in {"closure", "archived"}:
                daemon.publish_redteam_closure_request(
                    cycle_id=cycle_id, landing=landing)
            if cut == "archived":
                daemon.write_push_debt(
                    landing=landing, detail="simulated crash before push")
                if not daemon.archive_consumed_message(
                        dispatch_path=str(go_path)):
                    results.append(False)
                    continue
            try:
                first = daemon.reconcile_ticket_cycle_state()
                second = daemon.reconcile_ticket_cycle_state()
            except BaseException as exc:
                print("crash-cut " + cut + " failed: " + repr(exc))
                results.append(False)
                continue
            state = daemon.read_ticket_cycle_state()
            active = state["active"].get(cycle_id)
            closures = []
            for path in mailbox.rglob("*-to-sol.md"):
                text = path.read_text(encoding="utf-8")
                if daemon.redteam_closure_ticket(text) == cycle_id:
                    closures.append(path)
            checks = {
                "recovery-count": first == 0 and second == 0,
                "main-is-L": git(
                    root, "rev-parse", "HEAD").stdout.strip() == landing,
                "state-is-L": (
                    active is not None and active["commit"] == landing
                    and active["phase"] == "awaiting-redteam"),
                "go-done": (
                    (mailbox / "done" / go_path.name).is_file()
                    and not (mailbox / "failed" / go_path.name).exists()),
                "closure-unique": len(closures) == 1,
                "candidate-retired": git(
                    root, "rev-parse", "--verify",
                    daemon.cycle_candidate_ref(cycle_id=cycle_id),
                    check=False).returncode != 0,
                "landing-retired": git(
                    root, "rev-parse", "--verify",
                    daemon.cycle_landing_ref(cycle_id=cycle_id),
                    check=False).returncode != 0,
                "push-debt": Path(
                    daemon._push_debt_path(landing=landing)).is_file(),
            }
            passed = all(checks.values())
            if not passed:
                print("crash-cut " + cut + " checks=" + repr(checks))
            results.append(passed)
    passed = all(results) and len(results) == 5
    print("Architect GO crash cuts recover exactly once=" + str(passed))
    return passed


def arm_post_landing_error_preserves_go(source=None):
    """A verification error after main advances leaves one replayable GO."""
    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        base = git(implementer, "rev-parse", "HEAD").stdout.strip()
        anchor = "post-landing-error"
        cycle_id = anchor + "@" + base
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — [Recovery](#" + anchor
            + ")\n\n<a id=\"" + anchor + "\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)
        flow = (
            "MAILBOX-FLOW: ticket\n"
            "MAILBOX-CYCLE: " + cycle_id + "\n"
            "MAILBOX-MODE: normal\n\n"
            "Implement the recovery witness.\n")
        daemon.register_ticket_cycle_message(agent="opus", message=flow)
        starting = daemon.prepare_implementer_cycle_checkout(
            cycle_id=cycle_id)
        changed = implementer / "post-landing-error.txt"
        changed.write_text("candidate\n", encoding="utf-8", newline="")
        git(implementer, "add", changed.name)
        git(implementer, "commit", "-m", "post landing recovery")
        candidate = daemon.record_implementer_candidate(
            cycle_id=cycle_id, starting_head=starting)
        close_backlog_ticket(primary=primary, anchor=anchor)

        mailbox = Path(daemon.MAILBOX)
        inflight = mailbox / "inflight"
        inflight.mkdir(parents=True, exist_ok=True)
        go_path = inflight / "0003-to-daemon.md"
        go_path.write_text(
            daemon.architect_go_request_payload(
                cycle_id=cycle_id, candidate_commit=candidate,
                mode="normal"),
            encoding="utf-8", newline="")

        real_status = daemon._user_checkout_status
        status_calls = [0]

        def fail_after_landing():
            status_calls[0] += 1
            if status_calls[0] == 2:
                raise daemon.TicketCycleStateError(
                    "injected post-merge verification failure")
            return real_status()

        daemon._user_checkout_status = fail_after_landing
        stopped = False
        try:
            daemon.finish_claimed_architect_go(
                dispatch_path=str(go_path), cycle_id=cycle_id,
                candidate_commit=candidate, mode="normal")
        except daemon.FatalArchitectLandingError:
            stopped = True
        finally:
            daemon._user_checkout_status = real_status

        landing = daemon.git_ref_commit(
            reference=daemon.cycle_landing_ref(cycle_id=cycle_id))
        root_go = mailbox / go_path.name
        preserved = (
            stopped and landing is not None and root_go.is_file()
            and not (mailbox / "failed" / go_path.name).exists()
            and git(root, "rev-parse", "HEAD").stdout.strip() == landing)
        if not preserved:
            print("post-landing GO was not preserved")
            return False

        replayed = daemon.process_backlog(dry_run=False)
        second = daemon.reconcile_ticket_cycle_state()
        active = daemon.read_ticket_cycle_state()["active"].get(cycle_id)
        closures = [
            path for path in mailbox.rglob("*-to-sol.md")
            if daemon.redteam_closure_ticket(
                path.read_text(encoding="utf-8")) == cycle_id]
        checks = {
            "recovery-count": replayed is True and second == 0,
            "state": (active is not None
                      and active["phase"] == "awaiting-redteam"
                      and active["commit"] == landing),
            "go-done": (mailbox / "done" / go_path.name).is_file(),
            "closure-unique": len(closures) == 1,
            "candidate-retired": daemon.git_ref_commit(
                reference=daemon.cycle_candidate_ref(cycle_id)) is None,
            "landing-retired": daemon.git_ref_commit(
                reference=daemon.cycle_landing_ref(cycle_id)) is None,
        }
        passed = all(checks.values())
        if not passed:
            print("post-landing recovery checks=" + repr(checks))
        print("post-landing verification error replays once=" + str(passed))
        return passed


def arm_two_role_debt_failure_replays_past_cycle_limit(source=None):
    """A completed finite ticket still archives its requeued daemon GO."""
    with scratch_repository(source=source) as root:
        rc, _stdout, stderr = invoke(root, ["--once"])
        if rc != 0 or stderr != "" or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)

        base = git(implementer, "rev-parse", "HEAD").stdout.strip()
        anchor = "two-role-debt-restart"
        cycle_id = anchor + "@" + base
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — [Debt restart](#" + anchor
            + ")\n\n<a id=\"" + anchor + "\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)

        first_controller = daemon.SafeKillRendezvous(
            ticket_cycle_limit=1, ticket_cycle_topology="two-role")
        daemon._ACTIVE_WATCH_RENDEZVOUS = first_controller
        restored = daemon.prepare_finite_watch_progress(
            limit=1, topology="two-role")
        first_controller.restore_completed_ticket_cycles(count=restored)
        flow = (
            "MAILBOX-FLOW: ticket\n"
            "MAILBOX-CYCLE: " + cycle_id + "\n"
            "MAILBOX-MODE: two-role\n\n"
            "Implement the finite restart candidate.\n")
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow, skip_redteam=True)
        starting = daemon.prepare_implementer_cycle_checkout(
            cycle_id=cycle_id)
        changed = implementer / "two-role-debt-restart.txt"
        changed.write_text(
            "finite restart\n", encoding="utf-8", newline="")
        git(implementer, "add", changed.name)
        git(implementer, "commit", "-m", "scratch finite debt restart")
        candidate = daemon.record_implementer_candidate(
            cycle_id=cycle_id, starting_head=starting)
        if candidate is None:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
            return False
        close_backlog_ticket(primary=primary, anchor=anchor)
        mailbox = Path(daemon.MAILBOX)
        mailbox.mkdir(parents=True, exist_ok=True)
        go_path = mailbox / "0003-to-daemon.md"
        go_path.write_text(
            daemon.architect_go_request_payload(
                cycle_id=cycle_id, candidate_commit=candidate,
                mode="two-role"),
            encoding="utf-8", newline="")

        real_write_push_debt = daemon.write_push_debt
        debt_write_calls = 0

        def fail_first_debt_write(*, landing, detail):
            nonlocal debt_write_calls
            debt_write_calls = debt_write_calls + 1
            if debt_write_calls == 1:
                raise OSError("simulated relay write failure")
            return real_write_push_debt(landing=landing, detail=detail)

        daemon.write_push_debt = fail_first_debt_write
        fatal = False
        try:
            try:
                daemon.process_backlog(
                    dry_run=False, skip_redteam=True)
            except daemon.FatalArchitectLandingError:
                fatal = True
            state_after_failure = daemon.read_ticket_cycle_state()
            landing = state_after_failure["completed"].get(cycle_id)
            first_checks = {
                "fatal": fatal,
                "one-debt-write": debt_write_calls == 1,
                "go-requeued": go_path.is_file(),
                "go-not-done": not (
                    mailbox / "done" / go_path.name).exists(),
                "cycle-durable": (
                    isinstance(landing, str)
                    and state_after_failure["pending_cycle_returns"] == 1
                    and state_after_failure["finite_watch"] == {
                        "limit": 1, "completed": 0, "status": "active",
                        "topology": "two-role"}),
                "not-counted-in-memory": (
                    first_controller.completed_ticket_cycles() == 0),
                "no-debt-yet": (
                    isinstance(landing, str)
                    and not Path(
                        daemon._push_debt_path(landing=landing)).exists()),
            }
        finally:
            daemon.write_push_debt = real_write_push_debt
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        if not all(first_checks.values()):
            print("two-role first failure checks=" + repr(first_checks))
            return False

        replacement = daemon.SafeKillRendezvous(
            ticket_cycle_limit=1, ticket_cycle_topology="two-role")
        daemon._ACTIVE_WATCH_RENDEZVOUS = replacement
        try:
            restored_again = daemon.prepare_finite_watch_progress(
                limit=1, topology="two-role")
            replacement.restore_completed_ticket_cycles(
                count=restored_again)
            daemon.reconcile_ticket_cycle_state()
            delivered = daemon.deliver_pending_ticket_cycle_returns()
            limit_before_replay = replacement.ticket_cycle_limit_reached()
            restart_log = io.StringIO()
            with contextlib.redirect_stdout(restart_log):
                restart_outcome = daemon.process_backlog(
                    dry_run=False, skip_redteam=True)
            restart_text = restart_log.getvalue()
            second_outcome = daemon.process_backlog(
                dry_run=False, skip_redteam=True)
            final_state = daemon.read_ticket_cycle_state()
            final_count = replacement.completed_ticket_cycles()
            daemon.finish_finite_watch_progress(
                limit=1, completed=final_count, topology="two-role")
            finished_state = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None

        checks = {
            "restored-zero-before-pending": restored_again == 0,
            "delivered-once": delivered == 1,
            "limit-was-already-reached": limit_before_replay,
            "replay-consumed": restart_outcome is True,
            "replay-reports-completed-two-role": (
                "already complete at the exact local landing" in restart_text
                and "Red Team review is queued" not in restart_text),
            "second-pass-idempotent": second_outcome is None,
            "go-left-root": not go_path.exists(),
            "go-archived": (
                (mailbox / "done" / go_path.name).is_file()
                and not (mailbox / "failed" / go_path.name).exists()),
            "debt-durable": Path(
                daemon._push_debt_path(landing=landing)).is_file(),
            "cycle-not-recounted": (
                final_count == 1
                and final_state["completed"] == {cycle_id: landing}
                and final_state["pending_cycle_returns"] == 0
                and not final_state["active"]),
            "candidate-ref-retired": git(
                root, "rev-parse", "--verify",
                daemon.cycle_candidate_ref(cycle_id=cycle_id),
                check=False).returncode != 0,
            "landing-ref-retired": git(
                root, "rev-parse", "--verify",
                daemon.cycle_landing_ref(cycle_id=cycle_id),
                check=False).returncode != 0,
            "finite-exit-durable": finished_state["finite_watch"] == {
                "limit": 1, "completed": 1, "status": "complete",
                "topology": "two-role"},
        }
        passed = all(checks.values())
        print("finite two-role GO recovery bypasses admission limit="
              + str(passed))
        if not passed:
            print("finite two-role recovery checks=" + repr(checks))
        return passed


def arm_architect_go_user_checkout_stop_is_finite(source=None):
    """Dirty L and stale prepared L stop once while preserving all work."""
    results = []
    for case in ("dirty-landed", "moved-main"):
        with scratch_repository(source=source) as root:
            rc, _stdout, stderr = invoke(root, ["--once"])
            if rc != 0 or stderr != "" or not validate_topology(root):
                results.append(False)
                continue
            primary = default_primary(root)
            implementer = default_implementer(root)
            daemon = load_scratch_daemon(primary)
            daemon.ensure_primary_execution(live_action=True, dry_run=False)
            base = git(implementer, "rev-parse", "HEAD").stdout.strip()
            anchor = "landing-stop-" + case
            cycle_id = anchor + "@" + base
            backlog = primary / "ai" / "notes" / "backlog.md"
            backlog.write_text(
                "- OPEN **HIGH** **BUG FIX** — [Landing stop](#" + anchor
                + ")\n\n<a id=\"" + anchor + "\"></a>\n"
                "**Red Team reopen count: 0.**\n"
                "**Red Team reopening: allowed.**\n",
                encoding="utf-8", newline="")
            seal_backlog(primary)
            flow = (
                "MAILBOX-FLOW: ticket\n"
                "MAILBOX-CYCLE: " + cycle_id + "\n"
                "MAILBOX-MODE: normal\n\nPrepare stop witness.\n")
            daemon.register_ticket_cycle_message(agent="opus", message=flow)
            starting = daemon.prepare_implementer_cycle_checkout(
                cycle_id=cycle_id)
            changed = implementer / "stop-change.txt"
            changed.write_text(case + "\n", encoding="utf-8", newline="")
            git(implementer, "add", changed.name)
            git(implementer, "commit", "-m", "scratch " + case)
            candidate = daemon.record_implementer_candidate(
                cycle_id=cycle_id, starting_head=starting)
            landing, parent, _reference = daemon.prepare_exact_squash_landing(
                cycle_id=cycle_id, candidate_commit=candidate,
                mode="normal")
            close_backlog_ticket(primary=primary, anchor=anchor)
            if case == "dirty-landed":
                daemon.land_prepared_commit_in_clean_user_checkout(
                    landing=landing, parent=parent)
                user_file = root / "user-untracked.txt"
                user_file.write_text(
                    "preserve user work\n", encoding="utf-8", newline="")
                expected_main = landing
            else:
                user_file = root / "user-main-work.txt"
                user_file.write_text(
                    "committed user work\n", encoding="utf-8", newline="")
                git(root, "add", user_file.name)
                git(root, "commit", "-m", "user advances main")
                expected_main = git(
                    root, "rev-parse", "HEAD").stdout.strip()
            mailbox = Path(daemon.MAILBOX)
            mailbox.mkdir(parents=True, exist_ok=True)
            go_path = mailbox / "0003-to-daemon.md"
            go_path.write_text(
                daemon.architect_go_request_payload(
                    cycle_id=cycle_id, candidate_commit=candidate,
                    mode="normal"),
                encoding="utf-8", newline="")
            stop_rc, stop_stdout, stop_stderr = invoke(root, ["--once"])
            stop_text = stop_stdout + stop_stderr
            state = daemon.read_ticket_cycle_state()
            active = state["active"].get(cycle_id)
            checks = {
                "finite-nonzero": stop_rc == 1,
                "clear-stop": (
                    "Architect landing needs user action" in stop_text
                    or "primary worktree error:" in stop_text),
                "main-preserved": git(
                    root, "rev-parse", "HEAD").stdout.strip()
                == expected_main,
                "user-work-preserved": user_file.read_text(
                    encoding="utf-8").endswith("work\n"),
                "go-retryable": go_path.is_file(),
                "go-not-failed": not (
                    mailbox / "failed" / go_path.name).exists(),
                "state-not-advanced": (
                    active is not None
                    and active["phase"] == "implementation"
                    and active["commit"] is None),
                "candidate-preserved": git(
                    root, "rev-parse", "--verify",
                    daemon.cycle_candidate_ref(cycle_id=cycle_id),
                    check=False).returncode == 0,
                "landing-preserved": git(
                    root, "rev-parse", "--verify",
                    daemon.cycle_landing_ref(cycle_id=cycle_id),
                    check=False).returncode == 0,
            }
            if case == "dirty-landed":
                user_file.unlink()
                resumed = daemon.consume_daemon_message(path=str(go_path))
                checks["clean-restart-idempotent"] = resumed
            else:
                checks["honest-no-auto-recovery"] = (
                    "primary worktree error:" in stop_text
                    and daemon.STALE_INTEGRATION_REVALIDATION in stop_text
                    and all(
                        label + "=" + commit in stop_text
                        for label, commit in (
                            ("C", candidate), ("L", landing),
                            ("M0", parent), ("M1", expected_main))))
            passed = all(checks.values())
            if not passed:
                print("landing stop " + case + " checks=" + repr(checks))
            results.append(passed)
    passed = all(results) and len(results) == 2
    print("Architect GO user-checkout stops are finite=" + str(passed))
    return passed


def arm_timed_checkpoint_refusals(source=None):
    """A fired timer cannot become a candidate audit or landing GO."""
    from ai.tests.test_handoff_contract import packet

    evidence = (
        "#### Subagent return `failure-reproducer`\n"
        "- Returned artifact: The exact focused command and its complete "
        "pre-edit failing assertion output.\n"
        "- Acceptance: `pass`\n"
        "- Evidence: Command `python3 -m unittest "
        "ai.tests.test_example` exited one at the named assertion.\n"
        "#### Subagent return `regression-writer`\n"
        "- Returned artifact: The focused test-file diff and complete "
        "pre-production failing command output.\n"
        "- Acceptance: `pass`\n"
        "- Evidence: The diff changes only ExampleTests and the focused "
        "command output names the new assertion.")

    branch_checks = []

    # The marker represents the 90-minute hook. An ordinary COMPLETE return
    # after that marker must be parked before the candidate is frozen.
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        base = git(root, "rev-parse", "HEAD").stdout.strip()
        cycle = "timed-ordinary@" + base
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — "
            "[Timed ordinary return](#timed-ordinary)\n\n"
            "<a id=\"timed-ordinary\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)
        checkout = (
            "- Worktree: `" + str(implementer) + "`\n"
            "- Branch: `claude/mailbox-implementer`\n"
            "- Base: `" + base + "`")
        note = primary / "ai" / "notes" / "timed-spec.md"
        note.write_text(
            packet(role="architect", bodies={"Execution checkout": checkout}),
            encoding="utf-8", newline="")
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        inbound = mailbox / "0001-to-opus.md"
        inbound.write_text(
            "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
            "MAILBOX-MODE: normal\n\n"
            "- **Directive:** [ai/notes/timed-spec.md, section "
            "Implementation directive]\n",
            encoding="utf-8", newline="")
        child = primary / "ai" / "notes" / "relay" / \
            "timed-ordinary-child.py"
        child_source = (
            "import os\n"
            "from pathlib import Path\n"
            "import subprocess\n\n"
            "worktree = Path(os.environ['MAILBOX_EXECUTION_WORKTREE'])\n"
            "shared_notes = Path(os.environ['MAILBOX_SHARED_NOTES'])\n"
            "changed = worktree / 'timed-ordinary.txt'\n"
            "changed.write_text('candidate must not freeze\\n', "
            "encoding='utf-8', newline='')\n"
            "subprocess.run(['git', 'add', changed.name], cwd=worktree, "
            "check=True)\n"
            "subprocess.run(['git', 'commit', '-m', "
            "'scratch timed ordinary return'], cwd=worktree, check=True)\n"
            "candidate = subprocess.check_output("
            "['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True).strip()\n"
            "Path(os.environ['MAILBOX_IMPLEMENTER_CHECKPOINT_STATE'])."
            "write_text('triggered\\n', encoding='utf-8', newline='')\n"
            "body = " + repr(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
                "MAILBOX-MODE: normal\n\n"
                "### IMPLEMENTER_HANDOFF: COMPLETE\n\n"
                "- **Current state:** The bounded change is complete.\n"
                "- **Candidate commit:** `{candidate}`\n"
                "- **Subagent work:**\n" + evidence + "\n"
                "- **Blockers/findings:** No remaining blocker.\n"
                "- **Action required:** Architect audit of candidate.\n")
            + "\n"
            "message = body.format(candidate=candidate)\n"
            "target = shared_notes / 'mailbox' / '0002-to-fable.md'\n"
            "target.write_text(message, encoding='utf-8', newline='')\n")
        write_exact(child, child_source.encode("utf-8"))
        daemon.AGENT_COMMANDS = {
            "fable": ["unused-fable"],
            "opus": [sys.executable, str(child)],
            "sol": ["unused-sol"],
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = daemon.dispatch(path=str(inbound), dry_run=False)
        state = daemon.read_ticket_cycle_state()["active"].get(cycle)
        candidate_ref = daemon.cycle_candidate_ref(cycle_id=cycle)
        ordinary_checks = {
            "refused": result is False,
            "exact-reason": (
                "the 90-minute hook fired without its checkpoint handoff"
                in output.getvalue()),
            "request-failed": (
                (mailbox / "failed" / inbound.name).is_file()),
            "return-failed": (
                (mailbox / "failed" / "0002-to-fable.md").is_file()),
            "nothing-done": not list((mailbox / "done").glob("*.md")),
            "candidate-state-empty": (
                daemon.read_candidate_state()["cycles"].get(cycle) is None),
            "candidate-ref-absent": git(
                root, "rev-parse", "--verify", candidate_ref,
                check=False).returncode != 0,
            "ticket-still-implementation": (
                state is not None and state["phase"] == "implementation"
                and state["commit"] is None),
            "main-unchanged": (
                git(root, "rev-parse", "HEAD").stdout.strip() == base),
        }
        branch_checks.append(all(ordinary_checks.values()))
        if not branch_checks[-1]:
            print("timed ordinary-return checks=" + repr(ordinary_checks))

    # A checkpoint must preserve its partial work in a new clean commit.
    # Otherwise no immutable candidate exists for the Architect to inspect,
    # and the decision boundary would be skipped.
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        base = git(root, "rev-parse", "HEAD").stdout.strip()
        cycle = "timed-no-commit@" + base
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — "
            "[Timed no-commit checkpoint](#timed-no-commit)\n\n"
            "<a id=\"timed-no-commit\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)
        checkout = (
            "- Worktree: `" + str(implementer) + "`\n"
            "- Branch: `claude/mailbox-implementer`\n"
            "- Base: `" + base + "`")
        note = primary / "ai" / "notes" / "timed-spec.md"
        note.write_text(
            packet(role="architect", bodies={"Execution checkout": checkout}),
            encoding="utf-8", newline="")
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        inbound = mailbox / "0003-to-opus.md"
        inbound.write_text(
            "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
            "MAILBOX-MODE: normal\n\n"
            "- **Directive:** [ai/notes/timed-spec.md, section "
            "Implementation directive]\n",
            encoding="utf-8", newline="")
        child = primary / "ai" / "notes" / "relay" / \
            "timed-no-commit-child.py"
        child_source = (
            "import os\n"
            "from pathlib import Path\n"
            "import subprocess\n\n"
            "worktree = Path(os.environ['MAILBOX_EXECUTION_WORKTREE'])\n"
            "shared_notes = Path(os.environ['MAILBOX_SHARED_NOTES'])\n"
            "candidate = subprocess.check_output("
            "['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True).strip()\n"
            "Path(os.environ['MAILBOX_IMPLEMENTER_CHECKPOINT_STATE'])."
            "write_text('triggered\\n', encoding='utf-8', newline='')\n"
            "body = " + repr(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
                "MAILBOX-MODE: normal\n\n"
                "### IMPLEMENTER_HANDOFF: CHECKPOINT\n\n"
                "- **Current state:** 90 minutes reached; work is paused "
                "and may be stuck.\n"
                "- **Candidate commit:** `{candidate}`\n"
                "- **Subagent work:**\n" + evidence + "\n"
                "- **Blockers/findings:** Partial work needs review.\n"
                "- **Action required:** Architect checkpoint decision.\n")
            + "\n"
            "message = body.format(candidate=candidate)\n"
            "target = shared_notes / 'mailbox' / '0004-to-fable.md'\n"
            "target.write_text(message, encoding='utf-8', newline='')\n")
        write_exact(child, child_source.encode("utf-8"))
        daemon.AGENT_COMMANDS = {
            "fable": ["unused-fable"],
            "opus": [sys.executable, str(child)],
            "sol": ["unused-sol"],
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = daemon.dispatch(path=str(inbound), dry_run=False)
        state = daemon.read_ticket_cycle_state()["active"].get(cycle)
        candidate_ref = daemon.cycle_candidate_ref(cycle_id=cycle)
        no_commit_checks = {
            "refused": result is False,
            "exact-reason": (
                "checkpoint needs a new clean checkpoint commit"
                in output.getvalue()),
            "request-failed": (
                (mailbox / "failed" / inbound.name).is_file()),
            "return-failed": (
                (mailbox / "failed" / "0004-to-fable.md").is_file()),
            "candidate-state-empty": (
                daemon.read_candidate_state()["cycles"].get(cycle) is None),
            "candidate-ref-absent": git(
                root, "rev-parse", "--verify", candidate_ref,
                check=False).returncode != 0,
            "ticket-still-implementation": (
                state is not None and state["phase"] == "implementation"
                and state["commit"] is None),
            "implementer-head-unchanged": (
                git(implementer, "rev-parse", "HEAD").stdout.strip() == base),
            "main-unchanged": (
                git(root, "rev-parse", "HEAD").stdout.strip() == base),
        }
        branch_checks.append(all(no_commit_checks.values()))
        if not branch_checks[-1]:
            print("timed no-commit checks=" + repr(no_commit_checks))

    # A real checkpoint is advice to pause and reassess. Even if the
    # Architect child publishes a syntactically valid GO, no landing begins.
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        base = git(root, "rev-parse", "HEAD").stdout.strip()
        cycle = "timed-checkpoint@" + base
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — "
            "[Timed checkpoint](#timed-checkpoint)\n\n"
            "<a id=\"timed-checkpoint\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)
        flow = (
            "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
            "MAILBOX-MODE: normal\n\n")
        daemon.register_ticket_cycle_message(agent="opus", message=flow)
        starting = daemon.prepare_implementer_cycle_checkout(cycle_id=cycle)
        changed = implementer / "timed-checkpoint.txt"
        changed.write_text(
            "candidate remains pending\n", encoding="utf-8", newline="")
        git(implementer, "add", changed.name)
        git(implementer, "commit", "-m", "scratch timed checkpoint")
        candidate = daemon.record_implementer_candidate(
            cycle_id=cycle, starting_head=starting)
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        inbound = mailbox / "0003-to-fable.md"
        inbound.write_text(
            flow + daemon.IMPLEMENTER_CHECKPOINT_HEADING + "\n\n"
            + daemon.IMPLEMENTER_CHECKPOINT_CURRENT_STATE + "\n"
            "- **Candidate commit:** `" + candidate + "`\n"
            "- **Action required:** Architect complexity review.\n",
            encoding="utf-8", newline="")
        go_payload = daemon.architect_go_request_payload(
            cycle_id=cycle, candidate_commit=candidate, mode="normal")
        child = primary / "ai" / "notes" / "relay" / \
            "timed-checkpoint-child.py"
        child_source = (
            "import os\n"
            "from pathlib import Path\n\n"
            "mailbox = Path(os.environ['MAILBOX_SHARED_NOTES']) / 'mailbox'\n"
            "(mailbox / '0004-to-daemon.md').write_text("
            + repr(go_payload)
            + ", encoding='utf-8', newline='')\n"
            "(mailbox / '0005-to-opus.md').write_text("
            + repr(
                flow + "### ARCHITECT_HANDOFF: IMPLEMENTATION\n\n"
                "- **Checkpoint decision:** `GO`\n")
            + ", encoding='utf-8', newline='')\n")
        write_exact(child, child_source.encode("utf-8"))
        daemon.AGENT_COMMANDS = {
            "fable": [sys.executable, str(child)],
            "opus": ["unused-opus"],
            "sol": ["unused-sol"],
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = daemon.dispatch(path=str(inbound), dry_run=False)
        state = daemon.read_ticket_cycle_state()["active"].get(cycle)
        candidate_record = daemon.read_candidate_state()["cycles"].get(cycle)
        candidate_ref = daemon.cycle_candidate_ref(cycle_id=cycle)
        checkpoint_checks = {
            "refused": result is False,
            "exact-reason": (
                "a progress checkpoint cannot receive landing GO"
                in output.getvalue()),
            "request-failed": (
                (mailbox / "failed" / inbound.name).is_file()),
            "go-failed": (
                (mailbox / "failed" / "0004-to-daemon.md").is_file()),
            "handoff-failed": (
                (mailbox / "failed" / "0005-to-opus.md").is_file()),
            "nothing-done": not list((mailbox / "done").glob("*.md")),
            "ticket-still-implementation": (
                state is not None and state["phase"] == "implementation"
                and state["commit"] is None),
            "candidate-state-preserved": (
                candidate_record is not None
                and candidate_record["commit"] == candidate),
            "candidate-ref-preserved": git(
                root, "rev-parse", candidate_ref).stdout.strip()
            == candidate,
            "main-unchanged": (
                git(root, "rev-parse", "HEAD").stdout.strip() == base),
            "no-redteam-closure": not list(
                mailbox.rglob("*-to-sol.md")),
        }
        branch_checks.append(all(checkpoint_checks.values()))
        if not branch_checks[-1]:
            print("timed checkpoint GO checks=" + repr(checkpoint_checks))

        # A later Architect turn that sends only one explicit decision is
        # accepted, archived, and leaves the revised Implementer handoff
        # queued inside the same ticket.
        positive_inbound = mailbox / "0006-to-fable.md"
        positive_inbound.write_text(
            flow + daemon.IMPLEMENTER_CHECKPOINT_HEADING + "\n\n"
            + daemon.IMPLEMENTER_CHECKPOINT_CURRENT_STATE + "\n"
            "- **Candidate commit:** `" + candidate + "`\n"
            "- **Action required:** Architect complexity review.\n",
            encoding="utf-8", newline="")
        positive_child = primary / "ai" / "notes" / "relay" / \
            "timed-checkpoint-positive-child.py"
        positive_source = (
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "if 'MAILBOX-RETURN: architect-go' in sys.argv[-1]:\n"
            "    raise SystemExit('checkpoint prompt still grants landing')\n"
            "mailbox = Path(os.environ['MAILBOX_SHARED_NOTES']) / 'mailbox'\n"
            "(mailbox / '0007-to-opus.md').write_text("
            + repr(
                flow + "### ARCHITECT_HANDOFF: IMPLEMENTATION\n\n"
                "- **Checkpoint decision:** `NO-GO`\n")
            + ", encoding='utf-8', newline='')\n")
        write_exact(positive_child, positive_source.encode("utf-8"))
        daemon.AGENT_COMMANDS["fable"] = [
            sys.executable, str(positive_child)]
        positive_output = io.StringIO()
        with contextlib.redirect_stdout(positive_output):
            positive_result = daemon.dispatch(
                path=str(positive_inbound), dry_run=False)
        positive_state = daemon.read_ticket_cycle_state()["active"].get(cycle)
        positive_candidate = daemon.read_candidate_state()["cycles"].get(
            cycle)
        positive_checks = {
            "accepted": positive_result is True,
            "request-done": (
                (mailbox / "done" / positive_inbound.name).is_file()),
            "handoff-queued": (
                (mailbox / "0007-to-opus.md").is_file()),
            "handoff-not-failed": not (
                mailbox / "failed" / "0007-to-opus.md").exists(),
            "ticket-still-implementation": (
                positive_state is not None
                and positive_state["phase"] == "implementation"
                and positive_state["commit"] is None),
            "candidate-still-preserved": (
                positive_candidate is not None
                and positive_candidate["commit"] == candidate),
            "main-still-unchanged": (
                git(root, "rev-parse", "HEAD").stdout.strip() == base),
            "no-landing-go": not list(mailbox.glob("*-to-daemon.md")),
        }
        branch_checks.append(all(positive_checks.values()))
        if not branch_checks[-1]:
            print("timed checkpoint decision checks=" + repr(positive_checks))

        missing_inbound = mailbox / "0008-to-fable.md"
        missing_inbound.write_text(
            flow + daemon.IMPLEMENTER_CHECKPOINT_HEADING + "\n\n"
            + daemon.IMPLEMENTER_CHECKPOINT_CURRENT_STATE + "\n"
            "- **Candidate commit:** `" + candidate + "`\n"
            "- **Action required:** Architect complexity review.\n",
            encoding="utf-8", newline="")
        empty_child = primary / "ai" / "notes" / "relay" / \
            "timed-checkpoint-empty-child.py"
        write_exact(empty_child, b"pass\n")
        daemon.AGENT_COMMANDS["fable"] = [sys.executable, str(empty_child)]
        missing_output = io.StringIO()
        with contextlib.redirect_stdout(missing_output):
            missing_result = daemon.dispatch(
                path=str(missing_inbound), dry_run=False)
        missing_checks = {
            "refused": missing_result is False,
            "exact-reason": (
                "expected exactly one new Architect handoff to the "
                "Implementer; found 0"
                in missing_output.getvalue()),
            "request-failed": (
                (mailbox / "failed" / missing_inbound.name).is_file()),
            "prior-handoff-preserved": (
                (mailbox / "0007-to-opus.md").is_file()),
            "main-unchanged": (
                git(root, "rev-parse", "HEAD").stdout.strip() == base),
        }
        branch_checks.append(all(missing_checks.values()))
        if not branch_checks[-1]:
            print("timed missing-decision checks=" + repr(missing_checks))

        malformed_inbound = mailbox / "0009-to-fable.md"
        malformed_inbound.write_text(
            flow + daemon.IMPLEMENTER_CHECKPOINT_HEADING + "\n\n"
            + daemon.IMPLEMENTER_CHECKPOINT_CURRENT_STATE + "\n"
            "- **Current state:** progress exists.\n"
            "- **Candidate commit:** `" + candidate + "`\n",
            encoding="utf-8", newline="")
        launched = primary / "ai" / "notes" / "relay" / \
            "malformed-checkpoint-launched"
        malformed_child = primary / "ai" / "notes" / "relay" / \
            "malformed-checkpoint-child.py"
        write_exact(
            malformed_child,
            ("from pathlib import Path\nPath("
             + repr(str(launched))
             + ").write_text('launched\\n')\n").encode("utf-8"))
        daemon.AGENT_COMMANDS["fable"] = [
            sys.executable, str(malformed_child)]
        malformed_output = io.StringIO()
        with contextlib.redirect_stdout(malformed_output):
            malformed_result = daemon.dispatch(
                path=str(malformed_inbound), dry_run=False)
        malformed_checks = {
            "refused": malformed_result is False,
            "exact-reason": (
                "exact 90-minute Current state"
                in malformed_output.getvalue()),
            "request-failed": (
                (mailbox / "failed" / malformed_inbound.name).is_file()),
            "child-not-launched": not launched.exists(),
            "candidate-still-preserved": (
                daemon.read_candidate_state()["cycles"][cycle]["commit"]
                == candidate),
            "main-unchanged": (
                git(root, "rev-parse", "HEAD").stdout.strip() == base),
        }
        branch_checks.append(all(malformed_checks.values()))
        if not branch_checks[-1]:
            print("timed malformed-state checks=" + repr(malformed_checks))

    passed = branch_checks == [True, True, True, True, True, True]
    print("timed checkpoint refusal boundaries=" + str(passed))
    return passed


def arm_blocked_evidence_is_checkpoint_not_candidate(source=None):
    """Blocked evidence relays to Fable but cannot freeze or advance work."""
    from ai.tests.test_handoff_contract import NO_HELPER_EVIDENCE
    from ai.tests.test_handoff_contract import NO_HELPER_PLAN
    from ai.tests.test_handoff_contract import packet

    class PopenProxy:
        def __init__(self, module, replacement):
            self.module = module
            self.replacement = replacement

        def __getattr__(self, name):
            return (self.replacement if name == "Popen"
                    else getattr(self.module, name))

    class ReturningProcess:
        def __init__(self, callback):
            self.callback = callback
            self.returncode = None
            self.called = False

        def poll(self):
            if not self.called:
                self.called = True
                self.callback()
                self.returncode = 0
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self):
            return self.returncode

    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"], "opus": ["harmless-opus"],
            "sol": ["harmless-sol"]}
        base = git(implementer, "rev-parse", "HEAD").stdout.strip()
        cycle = "blocked-evidence@" + base
        backlog = primary / "ai" / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **HIGH** **BUG FIX** — "
            "[Blocked evidence](#blocked-evidence)\n\n"
            "<a id=\"blocked-evidence\"></a>\n"
            "**Red Team reopen count: 0.**\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8", newline="")
        seal_backlog(primary)
        checkout = (
            "- Worktree: `" + str(implementer) + "`\n"
            "- Branch: `claude/mailbox-implementer`\n"
            "- Base: `" + base + "`")
        note = primary / "ai" / "notes" / "blocked-spec.md"
        note.write_text(
            packet(role="architect", bodies={"Execution checkout": checkout}),
            encoding="utf-8", newline="")
        mailbox = primary / "ai" / "notes" / "mailbox"
        mailbox.mkdir(parents=True, exist_ok=True)
        inbound = mailbox / "0001-to-opus.md"
        inbound.write_text(
            "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
            "MAILBOX-MODE: normal\n\n"
            "- **Directive:** [ai/notes/blocked-spec.md, section "
            "Implementation directive]\n",
            encoding="utf-8", newline="")

        evidence = (
            "#### Subagent return `failure-reproducer`\n"
            "- Returned artifact: The launch attempt and complete runtime "
            "failure transcript.\n"
            "- Acceptance: `blocked`\n"
            "- Evidence: The exact launch operation failed before any "
            "implementation edit began.\n"
            "#### Subagent return `regression-writer`\n"
            "- Returned artifact: A no-edit checkpoint explaining that the "
            "second task never started.\n"
            "- Acceptance: `blocked`\n"
            "- Evidence: Repository status remained unchanged after the "
            "failed launch operation.")
        capability_plan = (
            "- Capability checked: `collaboration.spawn_agent`\n"
            "- Attempted operation: Launch the named reproducer subagent "
            "through collaboration.spawn_agent before implementation edits.\n"
            "- Raw failure: `Unknown collaboration.spawn_agent operation in "
            "the advertised runtime capability registry`")
        handoff_body = (
            "### IMPLEMENTER_HANDOFF: BLOCKED\n\n"
            "- **Current state:** Subagent launch is blocked.\n"
            "- **Candidate commit:** `" + base + "`\n"
            "- **Subagent work:**\n" + evidence + "\n" + capability_plan
            + "\n"
            "- **Blockers/findings:** Runtime lacks the launch operation.\n"
            "- **Action required:** Architect capability decision.\n")
        outbound = [mailbox / "0002-to-fable.md"]
        edit_mode = ["dirty"]

        def returned_body():
            candidate = git(
                implementer, "rev-parse", "HEAD").stdout.strip()
            if edit_mode[0] == "capability":
                return (
                    "### IMPLEMENTER_HANDOFF: COMPLETE\n\n"
                    "- **Current state:** The Architect-authorized no-helper "
                    "fallback completed the bounded implementation.\n"
                    "- **Candidate commit:** `" + candidate + "`\n"
                    "- **Subagent work:**\n" + capability_plan + "\n"
                    "- **Blockers/findings:** No remaining blocker.\n"
                    "- **Action required:** Architect audit of candidate.\n")
            body = handoff_body.replace(
                "- **Candidate commit:** `" + base + "`",
                "- **Candidate commit:** `" + candidate + "`")
            return body.replace(
                "Runtime lacks the launch operation.",
                "Runtime lacks the launch operation (" + edit_mode[0]
                + " attempt).")

        def publish_return():
            tracked = implementer / "ai" / "tools" / \
                "ticket_change_guard.py"
            if edit_mode[0] == "untracked":
                (implementer / "hallucinated_source.py").write_text(
                    "raise RuntimeError('not an accepted checkpoint')\n",
                    encoding="utf-8", newline="")
            if edit_mode[0] in {"dirty", "committed", "capability"}:
                tracked.write_text(
                    tracked.read_text(encoding="utf-8")
                    + "# forbidden blocked-attempt edit\n",
                    encoding="utf-8", newline="")
            if edit_mode[0] in {"committed", "capability"}:
                git(implementer, "add", str(tracked.relative_to(implementer)))
                git(implementer, "commit", "-m", "forbidden blocked edit")
            outbound[0].write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
                "MAILBOX-MODE: normal\n\n" + returned_body(),
                encoding="utf-8", newline="")

        def fake_popen(command, stdout, stderr, cwd, env):
            del command, stdout, stderr, cwd, env
            return ReturningProcess(callback=publish_return)

        calls = []
        real_record = daemon.record_implementer_candidate

        def record_probe(cycle_id, starting_head):
            calls.append((cycle_id, starting_head))
            return real_record(
                cycle_id=cycle_id, starting_head=starting_head)

        original = daemon.subprocess
        daemon.record_implementer_candidate = record_probe
        daemon.subprocess = PopenProxy(original, fake_popen)
        try:
            dirty_result = daemon.dispatch(
                path=str(inbound), dry_run=False)
            dirty_refused = (
                dirty_result is False and calls == []
                and not outbound[0].exists())
            git(implementer, "reset", "--hard", base)

            inbound = mailbox / "0003-to-opus.md"
            outbound[0] = mailbox / "0004-to-fable.md"
            inbound.write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
                "MAILBOX-MODE: normal\n\n"
                "- **Directive:** [ai/notes/blocked-spec.md, section "
                "Implementation directive]\n",
                encoding="utf-8", newline="")
            edit_mode[0] = "untracked"
            untracked_result = daemon.dispatch(
                path=str(inbound), dry_run=False)
            untracked_refused = (
                untracked_result is False and calls == []
                and not outbound[0].exists())
            git(implementer, "clean", "-fd")

            inbound = mailbox / "0005-to-opus.md"
            outbound[0] = mailbox / "0006-to-fable.md"
            inbound.write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
                "MAILBOX-MODE: normal\n\n"
                "- **Directive:** [ai/notes/blocked-spec.md, section "
                "Implementation directive]\n",
                encoding="utf-8", newline="")
            edit_mode[0] = "committed"
            committed_result = daemon.dispatch(
                path=str(inbound), dry_run=False)
            committed_refused = (
                committed_result is False and calls == []
                and not outbound[0].exists())
            git(implementer, "reset", "--hard", base)

            inbound = mailbox / "0007-to-opus.md"
            outbound[0] = mailbox / "0008-to-fable.md"
            inbound.write_text(
                "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
                "MAILBOX-MODE: normal\n\n"
                "- **Directive:** [ai/notes/blocked-spec.md, section "
                "Implementation directive]\n",
                encoding="utf-8", newline="")
            edit_mode[0] = "clean"
            result = daemon.dispatch(path=str(inbound), dry_run=False)
        finally:
            daemon.subprocess = original

        ticket_state = daemon.read_ticket_cycle_state()
        candidate_state = daemon.read_candidate_state()
        checkpoint_only = (
            dirty_refused and untracked_refused and committed_refused
            and result is True and calls == [] and outbound[0].exists()
            and candidate_state["cycles"] == {}
            and ticket_state["active"][cycle]["phase"] == "implementation"
            and git(root, "rev-parse", "--verify",
                    daemon.cycle_candidate_ref(cycle_id=cycle),
                    check=False).returncode != 0)

        blocked_body = returned_body()
        digest = daemon.hashlib.sha256(
            blocked_body.encode("utf-8")).hexdigest()

        def capability_note(source_cycle, source_digest,
                            failure_rows=capability_plan):
            checkpoint = (
                "### Prior Implementer subagent launch failure\n\n"
                "- Source cycle: `" + source_cycle + "`\n"
                "- Source handoff SHA-256: `" + source_digest + "`\n"
                "- Source: `prior same-cycle IMPLEMENTER_HANDOFF checkpoint`\n"
                + failure_rows)
            return packet(
                role="architect",
                bodies={"Execution checkout": checkout,
                        "Parallel work plan": failure_rows}).replace(
                            "No implementation evidence yet.", checkpoint)

        revised_message = (
            "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
            "MAILBOX-MODE: normal\n\n"
            "- **Directive:** [ai/notes/blocked-spec.md, section "
            "Implementation directive]\n")
        note.write_text(
            capability_note(cycle, digest), encoding="utf-8", newline="")
        correct_binding = True
        try:
            daemon.prepare_implementer_evidence_contract(
                message=revised_message)
        except daemon.TicketCycleStateError:
            correct_binding = False

        checkpoint_path = outbound[0]
        checkpoint_message = checkpoint_path.read_text(encoding="utf-8")
        checkpoint_prefix = (
            "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle + "\n"
            "MAILBOX-MODE: normal\n\n")
        checkpoint_bytes_match = (
            checkpoint_message == checkpoint_prefix + blocked_body)
        capability_rows = capability_plan.split("\n")
        missing_row_refusals = []
        for missing_row in capability_rows:
            missing_body = blocked_body.replace(missing_row, "", 1)
            missing_digest = daemon.hashlib.sha256(
                missing_body.encode("utf-8")).hexdigest()
            checkpoint_path.write_text(
                checkpoint_prefix + missing_body,
                encoding="utf-8", newline="")
            note.write_text(
                capability_note(cycle, missing_digest),
                encoding="utf-8", newline="")
            refused = False
            try:
                daemon.prepare_implementer_evidence_contract(
                    message=revised_message)
            except daemon.TicketCycleStateError:
                refused = True
            missing_row_refusals.append(refused)
        checkpoint_path.write_text(
            checkpoint_message, encoding="utf-8", newline="")

        changed_failure_rows = {
            "capability_checked": capability_plan.replace(
                "- Capability checked: `collaboration.spawn_agent`",
                "- Capability checked: `collaboration.spawn_agents`", 1),
            "attempted_operation": capability_plan.replace(
                "Launch the named reproducer subagent",
                "Launch the named failure reproducer subagent", 1),
            "raw_failure": capability_plan.replace(
                "Unknown collaboration.spawn_agent operation",
                "Rejected collaboration.spawn_agent operation", 1),
        }
        mismatch_refusals = {}
        for field, failure_rows in changed_failure_rows.items():
            note.write_text(
                capability_note(cycle, digest, failure_rows=failure_rows),
                encoding="utf-8", newline="")
            refused = False
            try:
                daemon.prepare_implementer_evidence_contract(
                    message=revised_message)
            except daemon.TicketCycleStateError as exc:
                refused = field in str(exc)
            mismatch_refusals[field] = refused

        stale_cycle = "stale-evidence@" + base
        note.write_text(
            capability_note(stale_cycle, digest),
            encoding="utf-8", newline="")
        stale_refused = False
        try:
            daemon.prepare_implementer_evidence_contract(
                message=revised_message)
        except daemon.TicketCycleStateError:
            stale_refused = True

        note.write_text(
            capability_note(cycle, "0" * 64),
            encoding="utf-8", newline="")
        wrong_sha_refused = False
        try:
            daemon.prepare_implementer_evidence_contract(
                message=revised_message)
        except daemon.TicketCycleStateError:
            wrong_sha_refused = True

        note.write_text(
            capability_note(cycle, digest), encoding="utf-8", newline="")
        inbound = mailbox / "0009-to-opus.md"
        outbound[0] = mailbox / "0010-to-fable.md"
        inbound.write_text(
            revised_message, encoding="utf-8", newline="")
        edit_mode[0] = "capability"
        daemon.subprocess = PopenProxy(original, fake_popen)
        try:
            fallback_result = daemon.dispatch(
                path=str(inbound), dry_run=False)
        finally:
            daemon.subprocess = original
        fallback_state = daemon.read_candidate_state()["cycles"].get(cycle)
        fallback_commit = git(
            implementer, "rev-parse", "HEAD").stdout.strip()
        fallback_completed = (
            fallback_result is True and len(calls) == 1
            and calls[0] == (cycle, base)
            and fallback_state is not None
            and fallback_state["commit"] == fallback_commit
            and fallback_commit != base
            and outbound[0].exists())

        note.write_text(
            packet(
                role="architect",
                bodies={"Execution checkout": checkout,
                        "Parallel work plan": NO_HELPER_PLAN}),
            encoding="utf-8", newline="")
        no_helper_contract = daemon.prepare_implementer_evidence_contract(
            message=revised_message)
        no_helper_handoff = (
            "### IMPLEMENTER_HANDOFF: COMPLETE\n\n"
            "- **Current state:** The indivisible edit is complete.\n"
            "- **Subagent work:**\n" + NO_HELPER_EVIDENCE + "\n"
            "- **Blockers/findings:** No remaining blocker.\n"
            "- **Action required:** Architect audit of candidate.\n")
        exact_no_helper = no_helper_contract["contract"].\
            validate_implementer_handoff_subagent_evidence(
                parallel_work_plan=no_helper_contract["parallel_work_plan"],
                handoff_text=no_helper_handoff)["completion_ready"]
        changed_no_helper_refused = False
        try:
            no_helper_contract["contract"].\
                validate_implementer_handoff_subagent_evidence(
                    parallel_work_plan=(
                        no_helper_contract["parallel_work_plan"]),
                    handoff_text=no_helper_handoff.replace(
                        "same inspection", "same source inspection"))
        except no_helper_contract["contract"].DirectiveError:
            changed_no_helper_refused = True

        passed = (
            checkpoint_only and correct_binding and checkpoint_bytes_match
            and all(missing_row_refusals)
            and all(mismatch_refusals.values())
            and stale_refused and wrong_sha_refused and fallback_completed
            and exact_no_helper and changed_no_helper_refused)
        print("blocked checkpoint cannot freeze candidate=" + str(passed))
        if not passed:
            print("blocked checkpoint checks=" + repr({
                "checkpoint-only": checkpoint_only,
                "correct-binding": correct_binding,
                "checkpoint-bytes-match": checkpoint_bytes_match,
                "missing-row-refusals": missing_row_refusals,
                "mismatch-refusals": mismatch_refusals,
                "stale-refused": stale_refused,
                "wrong-sha-refused": wrong_sha_refused,
                "dirty-refused": dirty_refused,
                "untracked-refused": untracked_refused,
                "committed-refused": committed_refused,
                "fallback-completed": fallback_completed,
                "exact-no-helper": exact_no_helper,
                "changed-no-helper-refused": changed_no_helper_refused,
                "record-calls": calls}))
        return passed


def arm_corrupt_and_redirected_state_fail_closed(source=None):
    """Invalid state is preserved and never falls back to caller mailbox."""
    with scratch_repository(source=source) as root:
        rc, _stdout, _stderr = invoke(root, ["--once"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        original = state_path(root).read_bytes()
        state = json.loads(original.decode("utf-8"))
        variants = []
        variants.append(("invalid-json", b"{ definitely not json\n"))
        duplicate = (
            "{\"schema\":3,\"schema\":3,\"repository\":"
            + json.dumps(state["repository"])
            + ",\"name\":\"mailbox-primary\",\"path\":"
            + json.dumps(state["path"])
            + ",\"branch\":\"refs/heads/claude/mailbox-primary\""
            + ",\"topology\":\"separate-role-worktrees-v1\"}\n")
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
                root, ["--send", "architect", "--unit", "must not queue"])
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
    """Every role uses its saved disjoint tree; root stays human-owned."""
    with scratch_repository(source=source) as root:
        root_before = root_checkout_identity(root)
        rc, _stdout, _stderr = invoke(
            root, ["--once",
                   "--architect-model", "opus",
                   "--implementer-model", "sonnet"])
        if rc != 0 or not validate_topology(root):
            return False
        primary = default_primary(root)
        implementer = default_implementer(root)
        sol = default_sol(root)
        daemon = load_scratch_daemon(primary)
        daemon.ensure_primary_execution(live_action=True, dry_run=False)
        create_empty_sealed_backlog(primary=primary)
        queued = daemon.send(
            agent="sol", ticket_kind="discovery", severity="medium",
            scope="bounded", text="Review the saved Sol worktree topology.",
            dry_run=False)
        rc_queue = 0 if queued else 1
        queue_err = ""
        rc_preview, preview, preview_err = invoke(
            root, ["--dry-run", "--once"])
        command = daemon.AGENT_COMMANDS["sol"]
        cd_value = command[command.index("--cd") + 1]
        notes_value = command[command.index("--add-dir") + 1]
        state = load_state(root)
        implementer_state = load_implementer_state(root)
        sol_state = load_sol_state(root)
        checks = {
            "queue": rc_queue == 0 and queue_err == "",
            "preview": rc_preview == 0 and preview_err == "",
            "preview-cd": ("--cd " + str(sol)) in preview,
            "preview-notes": (("--add-dir "
                               + str(primary / "ai" / "notes"))
                              in preview),
            "preview-cwd": ("(cwd " + str(sol) + ")") in preview,
            "fable-primary": daemon.AGENT_CWD["fable"] == str(primary),
            "opus-implementer": (
                daemon.AGENT_CWD["opus"] == str(implementer)),
            "sol-dedicated": daemon.AGENT_CWD["sol"] == str(sol),
            "command-cd": cd_value == str(sol),
            "command-notes": notes_value == str(primary / "ai" / "notes"),
            "four-trees": len({daemon.AGENT_CWD["fable"],
                               daemon.AGENT_CWD["opus"],
                               daemon.AGENT_CWD["sol"], str(root)}) == 4,
            "root-preserved": root_checkout_identity(root) == root_before,
            "primary-name": state["name"] == PRIMARY_NAME,
            "primary-schema": state["schema"] == PRIMARY_STATE_SCHEMA,
            "primary-topology": state["topology"] == PRIMARY_TOPOLOGY,
            "implementer-name": (
                implementer_state["name"] == IMPLEMENTER_NAME),
            "implementer-schema": (
                implementer_state["schema"] == IMPLEMENTER_STATE_SCHEMA),
            "sol-name": sol_state["name"] == SOL_NAME,
            "sol-schema": sol_state["schema"] == SOL_STATE_SCHEMA,
            "model-not-persisted": (
                "opus" not in state.values()
                and "sonnet" not in state.values()
                and "opus" not in implementer_state.values()
                and "sonnet" not in implementer_state.values()),
        }
        passed = all(checks.values())
        print("role topology=" + str(passed)
              + " cwd=" + repr(daemon.AGENT_CWD))
        if not passed:
            print("role topology checks=" + repr(checks)
                  + " preview-rc=" + str(rc_preview)
                  + " preview-error=" + repr(preview_err[-500:])
                  + " preview=" + repr(preview[-500:]))
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
    add("Implementer collocated with Architect",
        '    AGENT_CWD["opus"] = os.path.abspath(implementer_path)',
        '    AGENT_CWD["opus"] = os.path.abspath(primary_path)',
        arm_route_topology_remains_role_based)
    add("Implementer fell back to user root",
        '    AGENT_CWD["opus"] = os.path.abspath(implementer_path)',
        '    AGENT_CWD["opus"] = os.path.abspath(REPO_ROOT)',
        arm_route_topology_remains_role_based)
    add("Sol fell back to user root",
        '    AGENT_CWD["sol"] = os.path.abspath(sol_path)',
        '    AGENT_CWD["sol"] = os.path.abspath(REPO_ROOT)',
        arm_route_topology_remains_role_based)
    add("Sol collocated with Claude",
        '    AGENT_CWD["sol"] = os.path.abspath(sol_path)',
        '    AGENT_CWD["sol"] = os.path.abspath(primary_path)',
        arm_route_topology_remains_role_based)
    add("Codex cd fell back to user root",
        '                "--cd", sol_worktree,',
        '                "--cd", REPO_ROOT,',
        arm_route_topology_remains_role_based)
    add("Codex lost primary notes grant",
        '                "--add-dir", shared_notes],',
        '                "--add-dir", sol_worktree],',
        arm_route_topology_remains_role_based)
    add("candidate accepted as its own squash landing",
        "    if landing_commit == candidate_commit:\n",
        "    if False:\n",
        arm_architect_receipt_binds_candidate_to_squash_landing)
    add("unrelated landing tree accepted",
        "    if landing_tree != expected_tree:\n"
        "        raise TicketCycleStateError(\n"
        "            \"prepared landing tree is not the exact candidate "
        "squash\")\n",
        "    if False:\n"
        "        raise TicketCycleStateError(\n"
        "            \"prepared landing tree is not the exact candidate "
        "squash\")\n",
        arm_architect_receipt_binds_candidate_to_squash_landing)
    add("candidate ownership deleted without clean C-to-L handoff",
        '        _run_git(\n'
        '            repository_root=worktree,\n'
        '            arguments=["reset", "--hard", landing_commit])\n',
        '        pass  # mutation: C remains checked out after retirement\n',
        arm_landed_candidate_hands_off_without_clobbering_pipeline)
    add("prior retirement clobbers a saved pipeline candidate",
        '    if head == record["commit"]:\n',
        '    if True:  # mutation: reset even when another C owns HEAD\n',
        arm_landed_candidate_hands_off_without_clobbering_pipeline)
    add("primary role branch resurrected as automatic landing debt",
        '        diff_ranges = []\n',
        '        diff_ranges = [("main", "HEAD")]  # old branch debt\n',
        arm_landed_candidate_hands_off_without_clobbering_pipeline)
    add("reset-complete candidate retirement cannot replay",
        '    if head != landing_commit and head not in preserved_heads:\n',
        '    if True:  # mutation: exact prior reset cannot retire state\n',
        arm_candidate_retirement_internal_crash_replays)
    add("persistent role end-state recheck dropped",
        '            else:\n'
        '                revalidate_agent_dispatch_topology(\n'
        '                    proof=agent_topology_proof)\n'
        '            if not notes_admin_turn:\n'
        '                recheck_persistent_role_state(\n'
        '                    proof=persistent_role_state)\n'
        '        except (OSError, PrimaryWorktreeError) as exc:\n'
        '            persistent_role_error = exc',
        '            else:\n'
        '                revalidate_agent_dispatch_topology(\n'
        '                    proof=agent_topology_proof)\n'
        '            if False:  # mutation: persistent edits are ignored\n'
        '                recheck_persistent_role_state(\n'
        '                    proof=persistent_role_state)\n'
        '        except (OSError, PrimaryWorktreeError) as exc:\n'
        '            persistent_role_error = exc',
        arm_persistent_roles_refuse_tracked_and_untracked_source_edits)
    add("Sol untracked source files omitted from status proof",
        '         "--untracked-files=all", "--ignore-submodules=none"],',
        '         "--untracked-files=no", "--ignore-submodules=none"],',
        arm_persistent_roles_refuse_tracked_and_untracked_source_edits)
    add("Architect untracked source comparison neutralized",
        '            or current_untracked != proof["untracked_state"]):',
        '            or False):  # mutation: new Architect files ignored',
        arm_persistent_roles_refuse_tracked_and_untracked_source_edits)
    add("Opus shared protected-note recheck dropped",
        '    if agent in {"opus", "sol"}:\n'
        '        _recheck_shared_protected_state(proof=proof['
        '"shared_proof"])\n',
        '    if agent == "sol":\n'
        '        _recheck_shared_protected_state(proof=proof['
        '"shared_proof"])\n',
        arm_shared_protected_notes_require_architect_authority)
    add("Sol shared protected-note recheck dropped",
        '    if agent in {"opus", "sol"}:\n'
        '        _recheck_shared_protected_state(proof=proof['
        '"shared_proof"])\n',
        '    if agent == "opus":\n'
        '        _recheck_shared_protected_state(proof=proof['
        '"shared_proof"])\n',
        arm_shared_protected_notes_require_architect_authority)
    add("permanent MEMORY note omitted from protection",
        '    "ai/notes/MEMORY.md",\n',
        '',
        arm_shared_protected_notes_require_architect_authority)
    add("blocked Implementer evidence treated as final candidate",
        '            bool(evidence_results[0].get("completion_ready")))',
        '            True)  # mutation: blocked checkpoint advances',
        arm_blocked_evidence_is_checkpoint_not_candidate)
    add("fired timer accepts an ordinary Implementer return",
        '            if (evidence_problem is None\n'
        '                    and implementer_checkpoint_delivered(\n'
        '                        checkpoint_state_path)):\n',
        '            if False:  # mutation: timed return is not checked\n',
        arm_timed_checkpoint_refusals)
    add("checkpoint without a new commit accepted",
        '                if (evidence_problem is None\n'
        '                        and returned_candidate == '
        'implementer_starting_head):\n'
        '                    evidence_problem = (\n'
        '                        "the 90-minute checkpoint needs a new clean "\n'
        '                        "checkpoint commit")\n',
        '                if False:  # mutation: unchanged HEAD accepted\n'
        '                    evidence_problem = "unreachable"\n',
        arm_timed_checkpoint_refusals)
    add("contradictory checkpoint state reaches Architect",
        '        checkpoint_problem = (checkpoint_handoff_problem('
        'message=message)\n'
        '                              if checkpoint_request else None)\n',
        '        checkpoint_problem = None  # mutation: state not checked\n',
        arm_timed_checkpoint_refusals)
    add("progress checkpoint accepts landing GO",
        '            if go_path is not None:\n'
        '                invalid_go_paths.append(go_path)\n'
        '                go_problem = "a progress checkpoint cannot '
        'receive landing GO"\n',
        '            if False:  # mutation: checkpoint GO is accepted\n'
        '                invalid_go_paths.append(go_path)\n'
        '                go_problem = "a progress checkpoint cannot '
        'receive landing GO"\n',
        arm_timed_checkpoint_refusals)
    add("checkpoint decision handoff is optional",
        '            handoff_path, invalid_handoffs, handoff_problem = (\n'
        '                matching_new_checkpoint_handoff(\n'
        '                    cycle_id=audit_cycle_id, mode=flow_mode,\n'
        '                    before_inodes=architect_opus_before))\n',
        '            handoff_path, invalid_handoffs, handoff_problem = (\n'
        '                None, [], None)  # mutation: no decision required\n',
        arm_timed_checkpoint_refusals)
    add("checkpoint prompt restores ordinary landing instructions",
        '    common_preamble = common_preamble_for_dispatch(\n'
        '        checkpoint_audit=architect_checkpoint_audit)\n',
        '    common_preamble = PREAMBLE  # mutation: checkpoint may land\n',
        arm_timed_checkpoint_refusals)
    add("stale capability cycle accepted",
        '                or checkpoint.get("cycle") != cycle_id):',
        '                or False):',
        arm_blocked_evidence_is_checkpoint_not_candidate)
    add("missing active Sol topology accepted",
        '    if ACTIVE_TOPOLOGY is None:\n'
        '        raise PrimaryWorktreeError(\n'
        '            "live " + agent + " dispatch has no validated '
        'topology")',
        '    if ACTIVE_TOPOLOGY is None and agent != "sol":\n'
        '        raise PrimaryWorktreeError(\n'
        '            "mutation lets Sol dispatch without live topology")',
        arm_sol_launch_boundary_revalidates_branch_and_active_state)
    add("authoritative parent and role proof dropped",
        '        authoritative_files = validate_authoritative_role_files(\n'
        '            primary_path=primary["path"])',
        '        authoritative_files = {"directories": (), "files": ()}',
        arm_sol_launch_boundary_revalidates_branch_and_active_state)
    add("pre-Popen Architect topology revalidation dropped",
        '            if agent in {"fable", "opus", "sol"}:\n'
        '                revalidate_agent_dispatch_topology(\n'
        '                    proof=agent_topology_proof)\n'
        '            recheck_persistent_role_state('
        'proof=persistent_role_state)\n'
        '            proc = subprocess.Popen(command,',
        '            if agent in {"opus", "sol"}:\n'
        '                revalidate_agent_dispatch_topology(\n'
        '                    proof=agent_topology_proof)\n'
        '            recheck_persistent_role_state('
        'proof=persistent_role_state)\n'
        '            proc = subprocess.Popen(command,',
        arm_architect_launch_boundary_revalidates_branch_and_role)
    add("post-Popen Architect topology revalidation dropped",
        '                if agent in {"fable", "opus", "sol"}:\n'
        '                    if notes_admin_turn:\n'
        '                        revalidate_protected_policy_admin_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                    else:\n'
        '                        revalidate_agent_dispatch_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                if not notes_admin_turn:\n'
        '                    recheck_persistent_role_state(\n'
        '                        proof=persistent_role_state)\n'
        '            except (OSError, PrimaryWorktreeError):',
        '                if agent in {"opus", "sol"}:\n'
        '                    revalidate_agent_dispatch_topology(\n'
        '                        proof=agent_topology_proof)\n'
        '                if not notes_admin_turn:\n'
        '                    recheck_persistent_role_state(\n'
        '                        proof=persistent_role_state)\n'
        '            except (OSError, PrimaryWorktreeError):',
        arm_architect_launch_boundary_revalidates_branch_and_role)
    add("pre-Popen Sol topology revalidation dropped",
        '            if agent in {"fable", "opus", "sol"}:\n'
        '                revalidate_agent_dispatch_topology(\n'
        '                    proof=agent_topology_proof)\n'
        '            recheck_persistent_role_state('
        'proof=persistent_role_state)\n'
        '            proc = subprocess.Popen(command,',
        '            if agent in {"fable", "opus"}:\n'
        '                revalidate_agent_dispatch_topology(\n'
        '                    proof=agent_topology_proof)\n'
        '            recheck_persistent_role_state('
        'proof=persistent_role_state)\n'
        '            proc = subprocess.Popen(command,',
        arm_sol_launch_boundary_revalidates_branch_and_active_state)
    add("post-Popen Sol topology revalidation dropped",
        '                if agent in {"fable", "opus", "sol"}:\n'
        '                    if notes_admin_turn:\n'
        '                        revalidate_protected_policy_admin_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                    else:\n'
        '                        revalidate_agent_dispatch_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                if not notes_admin_turn:\n'
        '                    recheck_persistent_role_state(\n'
        '                        proof=persistent_role_state)\n'
        '            except (OSError, PrimaryWorktreeError):',
        '                if agent in {"fable", "opus"}:\n'
        '                    if notes_admin_turn:\n'
        '                        revalidate_protected_policy_admin_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                    else:\n'
        '                        revalidate_agent_dispatch_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                if not notes_admin_turn:\n'
        '                    recheck_persistent_role_state(\n'
        '                        proof=persistent_role_state)\n'
        '            except (OSError, PrimaryWorktreeError):',
        arm_sol_launch_boundary_revalidates_branch_and_active_state)
    add("pre-Popen Implementer topology revalidation dropped",
        '            if agent in {"fable", "opus", "sol"}:\n'
        '                revalidate_agent_dispatch_topology(\n'
        '                    proof=agent_topology_proof)\n'
        '            recheck_persistent_role_state('
        'proof=persistent_role_state)\n'
        '            proc = subprocess.Popen(command,',
        '            if agent in {"fable", "sol"}:\n'
        '                revalidate_agent_dispatch_topology(\n'
        '                    proof=agent_topology_proof)\n'
        '            recheck_persistent_role_state('
        'proof=persistent_role_state)\n'
        '            proc = subprocess.Popen(command,',
        arm_implementer_launch_boundary_revalidates_branch_and_state)
    add("post-Popen Implementer topology revalidation dropped",
        '                if agent in {"fable", "opus", "sol"}:\n'
        '                    if notes_admin_turn:\n'
        '                        revalidate_protected_policy_admin_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                    else:\n'
        '                        revalidate_agent_dispatch_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                if not notes_admin_turn:\n'
        '                    recheck_persistent_role_state(\n'
        '                        proof=persistent_role_state)\n'
        '            except (OSError, PrimaryWorktreeError):',
        '                if agent in {"fable", "sol"}:\n'
        '                    if notes_admin_turn:\n'
        '                        revalidate_protected_policy_admin_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                    else:\n'
        '                        revalidate_agent_dispatch_topology(\n'
        '                            proof=agent_topology_proof)\n'
        '                if not notes_admin_turn:\n'
        '                    recheck_persistent_role_state(\n'
        '                        proof=persistent_role_state)\n'
        '            except (OSError, PrimaryWorktreeError):',
        arm_implementer_launch_boundary_revalidates_branch_and_state)
    add("around-Popen Sol Git revalidation reduced to inode check",
        '    current = validate_live_agent_dispatch_topology('
        'agent=proof["agent"])\n'
        '    if current != proof:\n',
        '    current = proof  # mutation: branch and state not re-read\n'
        '    if current != proof:\n',
        arm_sol_launch_boundary_revalidates_branch_and_active_state)
    legacy_refusal = (
        '    raise PrimaryWorktreeError(\n'
        '        "the saved mailbox topology predates the separate '
        'Implementer "\n'
        '        "worktree and cannot be migrated safely while an "\n'
        '        "older daemon may already be admitted; stop every old '
        'mailbox "\n'
        '        "process, preserve the saved primary worktree and mailbox, '
        'update "\n'
        '        "that worktree to this daemon version, move the old local '
        'state "\n'
        '        "file aside for recovery, then run the current daemon from '
        'the "\n'
        '        "saved primary path to initialize the new topology")')
    add("legacy v1 dispatch accepted",
        legacy_refusal,
        "    return state  # mutation: legacy root-Sol runtime can resume",
        arm_legacy_v1_state_refuses_without_mutation)
    add("duplicate keys accepted",
        "        state = json.loads(text, "
        "object_pairs_hook=_duplicate_key_refusal)",
        "        state = json.loads(text, object_pairs_hook=dict)",
        arm_corrupt_and_redirected_state_fail_closed)
    add("completed daemon GO blocked by finite cycle limit",
        "    for daemon_path in daemon_paths:\n"
        "        # This GO belongs to a ticket already admitted against the "
        "finite\n"
        "        # limit. Always finish its durable landing/archive recovery. "
        "The\n"
        "        # positive limit gates new role work in drain_lane(), never "
        "this\n"
        "        # already-admitted daemon transition.\n",
        "    for daemon_path in daemon_paths:\n"
        "        controller = (_ACTIVE_WATCH_RENDEZVOUS\n"
        "                      if not dry_run else None)\n"
        "        if (controller is not None\n"
        "                and controller.ticket_cycle_limit_reached()):\n"
        "            break\n",
        arm_two_role_debt_failure_replays_past_cycle_limit)
    add("foreign repository accepted",
        '    if state["repository"] != repository:\n',
        "    if False:\n",
        arm_corrupt_and_redirected_state_fail_closed)
    add("transport evidence ignored",
        '    evidence = []\n'
        '    for record in records:\n'
        '        reasons = coordination_transport_evidence('
        'worktree=record["path"])',
        '    evidence = []\n'
        '    for record in records:\n'
        '        reasons = []',
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
    add("stale primary protocol accepted",
        "    if protocol_declarations != [\n"
        "            str(MAILBOX_PROTOCOL_VERSION).encode(\"ascii\")]:\n"
        "        raise PrimaryWorktreeError(\n"
        "            \"saved primary daemon does not enforce the current \"\n"
        "            \"Architect-only user entry point; \"\n"
        "            \"update that non-main worktree from main without discarding \"\n"
        "            \"its local work, then retry: \" + primary_path)",
        "    if False:\n"
        "        raise PrimaryWorktreeError(\n"
        "            \"mutation accepts a stale primary protocol\")",
        arm_stale_primary_protocol_refuses)
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
        ("stale primary protocol refusal",
         arm_stale_primary_protocol_refuses),
        ("zero-write inspection", arm_help_dry_run_and_invalid_are_zero_write),
        ("cross-checkout reuse", arm_reuse_and_cross_checkout_converge),
        ("clean user-main authority",
         arm_clean_user_main_advances_only_from_clean_checkout),
        ("interrupted Implementer bootstrap",
         arm_interrupted_implementer_bootstrap_is_exactly_resumable),
        ("existing coordinator adoption",
         arm_existing_linked_coordinator_is_adopted),
        ("late watcher adoption fence",
         arm_adoption_publication_fences_late_watcher),
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
        ("legacy v1 topology refusal",
         arm_legacy_v1_state_refuses_without_mutation),
        ("legacy two-tree topology refusal",
         arm_legacy_two_tree_state_refuses_without_mutation),
        ("Sol collision and corrupt-state refusal",
         arm_sol_collisions_and_corrupt_state_fail_closed),
        ("Sol move and reuse",
         arm_sol_registered_move_and_reuse_are_preserved),
        ("Sol launch-boundary topology race",
         arm_sol_launch_boundary_revalidates_branch_and_active_state),
        ("Architect launch-boundary topology race",
         arm_architect_launch_boundary_revalidates_branch_and_role),
        ("Implementer launch-boundary topology race",
         arm_implementer_launch_boundary_revalidates_branch_and_state),
        ("candidate snapshot isolation",
         arm_candidate_snapshot_is_exact_and_immutable),
        ("candidate and squash landing binding",
         arm_architect_receipt_binds_candidate_to_squash_landing),
        ("landed candidate checkout handoff",
         arm_landed_candidate_hands_off_without_clobbering_pipeline),
        ("candidate retirement internal crash replay",
         arm_candidate_retirement_internal_crash_replays),
        ("Architect GO crash recovery",
         arm_architect_go_crash_cuts_recover_once),
        ("post-landing GO preservation",
         arm_post_landing_error_preserves_go),
        ("finite two-role daemon GO recovery",
         arm_two_role_debt_failure_replays_past_cycle_limit),
        ("Architect GO finite user-action stop",
         arm_architect_go_user_checkout_stop_is_finite),
        ("exclusive permanent-note admin landing",
         arm_permanent_note_admin_is_exclusive_and_lands),
        ("permanent-note journal restart",
         arm_permanent_note_journal_restart_is_exact),
        ("Architect-bound permanent-note publisher",
         arm_permanent_note_admin_publisher_is_architect_bound),
        ("persistent role source authority",
         arm_persistent_roles_refuse_tracked_and_untracked_source_edits),
        ("shared protected-note authority",
         arm_shared_protected_notes_require_architect_authority),
        ("timed checkpoint refusal boundaries",
         arm_timed_checkpoint_refusals),
        ("blocked Implementer checkpoint binding",
         arm_blocked_evidence_is_checkpoint_not_candidate),
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
