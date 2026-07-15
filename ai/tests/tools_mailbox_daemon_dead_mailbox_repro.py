#!/usr/bin/env python3
"""Scratch-only witnesses for the mailbox daemon's dead-mailbox warning."""

import contextlib
import fcntl
import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import types


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
OWN_WARNING = "  !! warning: no active watch is polling this mailbox: "
OTHER_WARNING = (
    "  !! warning: another mailbox under this repository has a live watch: ")


def load_daemon(source=None):
    """Load a fresh daemon module for one isolated reproduction."""
    if source is not None:
        module = types.ModuleType("mailbox_daemon_dead_mailbox_mutant")
        module.__file__ = str(DAEMON_PATH)
        exec(compile(source, str(DAEMON_PATH), "exec"), module.__dict__)
        return module
    spec = importlib.util.spec_from_file_location(
        "mailbox_daemon_dead_mailbox_repro", DAEMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def scratch_daemon(create_mailbox=True, source=None):
    """Point a fresh daemon at a disposable repository and worktree."""
    with tempfile.TemporaryDirectory(prefix="mailbox-dead-repro-") as tmp:
        root = pathlib.Path(tmp)
        worktree = root / ".claude" / "worktrees" / "current"
        ai_root = worktree / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(worktree)
        daemon.AI_ROOT = str(ai_root)
        daemon.MAILBOX = str(mailbox)
        daemon.DONE = str(mailbox / "done")
        daemon.RELAY_DIR = str(ai_root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(ai_root / "notes" / "backlog.md")
        daemon.AGENT_CWD = {
            "fable": str(worktree),
            "opus": str(worktree),
            "sol": str(root),
        }
        daemon.report_demand = lambda backlog: None
        if create_mailbox:
            mailbox.mkdir(parents=True)
        yield daemon, root, mailbox


def warning_lines(output):
    """Return only dead-mailbox warning lines from captured output."""
    return [line for line in output.splitlines()
            if line.startswith(OWN_WARNING)
            or line.startswith(OTHER_WARNING)]


def captured_send(daemon, dry_run=True, text="scratch body"):
    """Run one scratch send and return its outcome and terminal output."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        outcome = daemon.send(agent="opus", text=text, dry_run=dry_run)
    return outcome, stream.getvalue()


@contextlib.contextmanager
def held_mode_lock(daemon, mailbox, mode):
    """Acquire a production-format dispatch lock in a chosen mailbox."""
    previous = daemon.MAILBOX
    daemon.MAILBOX = str(mailbox)
    try:
        handle = daemon.acquire_dispatch_lock(mode=mode)
    finally:
        daemon.MAILBOX = previous
    if handle is None:
        raise RuntimeError("scratch lock acquisition failed")
    try:
        yield handle
    finally:
        daemon.release_dispatch_lock(lock_file=handle)


@contextlib.contextmanager
def held_raw_lock(path, payload):
    """Hold an exclusive flock over caller-supplied raw metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    handle = open(path, "rb")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def arm_absent_lock_warns():
    """An absent own lock warns and names that exact mailbox."""
    with scratch_daemon() as (daemon, _, mailbox):
        outcome, output = captured_send(daemon)
        observed = warning_lines(output)
        expected = [OWN_WARNING + str(mailbox)]
        print("absent observed=" + repr(observed))
        return outcome and observed == expected


def arm_stale_lock_is_read_only():
    """An unlocked watch-tagged lock warns without changing its inode/data."""
    with scratch_daemon() as (daemon, _, mailbox):
        with held_mode_lock(daemon, mailbox, "watch"):
            pass
        lock_path = mailbox / ".dispatch.lock"
        before_stat = lock_path.stat()
        before_bytes = lock_path.read_bytes()
        outcome, output = captured_send(daemon)
        after_stat = lock_path.stat()
        preserved = (
            (before_stat.st_dev, before_stat.st_ino, before_stat.st_mode,
             before_stat.st_mtime_ns)
            == (after_stat.st_dev, after_stat.st_ino, after_stat.st_mode,
                after_stat.st_mtime_ns)
            and lock_path.read_bytes() == before_bytes)
        observed = warning_lines(output)
        print("stale preserved=" + str(preserved)
              + " observed=" + repr(observed))
        return (outcome and preserved
                and observed == [OWN_WARNING + str(mailbox)])


def arm_watch_suppresses_warning():
    """Only an exact held watch lock suppresses the warning."""
    with scratch_daemon() as (daemon, _, mailbox):
        with held_mode_lock(daemon, mailbox, "watch"):
            outcome, output = captured_send(daemon)
        observed = warning_lines(output)
        print("watch observed=" + repr(observed))
        return outcome and observed == []


def arm_once_legacy_unknown_warn():
    """Held once, legacy PID-only, and malformed locks are not watchers."""
    results = []
    for label in ("once", "unknown", "legacy", "malformed"):
        with scratch_daemon() as (daemon, _, mailbox):
            lock_path = mailbox / ".dispatch.lock"
            manager = (held_mode_lock(daemon, mailbox, label)
                       if label in ("once", "unknown") else
                       held_raw_lock(lock_path,
                                     b"31415" if label == "legacy"
                                     else b" watch pid 9 "))
            with manager:
                outcome, output = captured_send(daemon)
            observed = warning_lines(output)
            passed = outcome and observed == [OWN_WARNING + str(mailbox)]
            results.append(passed)
            print(label + " observed=" + repr(observed))
    return all(results)


def arm_other_watches_are_exact_and_sorted():
    """Own-dead diagnosis lists only the two live other watches, sorted."""
    with scratch_daemon() as (daemon, root, mailbox):
        main_mailbox = root / "ai" / "notes" / "mailbox"
        hidden_mailbox = (root / ".claude" / "worktrees" / ".hidden"
                          / "ai" / "notes" / "mailbox")
        stale_mailbox = (root / ".claude" / "worktrees" / "stale"
                         / "ai" / "notes" / "mailbox")
        once_mailbox = (root / ".claude" / "worktrees" / "once"
                        / "ai" / "notes" / "mailbox")
        with held_mode_lock(daemon, stale_mailbox, "watch"):
            pass
        with contextlib.ExitStack() as stack:
            stack.enter_context(held_mode_lock(
                daemon, main_mailbox, "watch"))
            stack.enter_context(held_mode_lock(
                daemon, hidden_mailbox, "watch"))
            stack.enter_context(held_mode_lock(
                daemon, once_mailbox, "once"))
            outcome, output = captured_send(daemon)
        others = sorted([str(main_mailbox), str(hidden_mailbox)])
        expected = [OWN_WARNING + str(mailbox)]
        expected.extend(OTHER_WARNING + path for path in others)
        observed = warning_lines(output)
        print("other-watches observed=" + repr(observed))
        return outcome and observed == expected


def arm_live_own_suppresses_alternatives():
    """A live own watcher suppresses both warning and alternative scan output."""
    with scratch_daemon() as (daemon, root, mailbox):
        other = root / "ai" / "notes" / "mailbox"
        with contextlib.ExitStack() as stack:
            stack.enter_context(held_mode_lock(daemon, other, "watch"))
            stack.enter_context(held_mode_lock(daemon, mailbox, "watch"))
            outcome, output = captured_send(daemon)
        observed = warning_lines(output)
        print("own-live observed=" + repr(observed))
        return outcome and observed == [] and str(other) not in output


def arm_nonexistent_dry_run_is_read_only():
    """Dry-run diagnoses a nonexistent mailbox without creating its tree."""
    with scratch_daemon(create_mailbox=False) as (daemon, root, mailbox):
        before = sorted(str(path.relative_to(root))
                        for path in root.rglob("*"))
        outcome, output = captured_send(daemon)
        after = sorted(str(path.relative_to(root))
                       for path in root.rglob("*"))
        observed = warning_lines(output)
        unchanged = before == after and not mailbox.exists()
        print("dry-run unchanged=" + str(unchanged)
              + " observed=" + repr(observed))
        return (outcome and unchanged
                and observed == [OWN_WARNING + str(mailbox)])


def arm_symlink_fifo_and_shared_probe_are_safe():
    """Hostile locks and redirected discovery paths stay dead and bounded."""
    results = []
    for label in ("symlink", "fifo", "shared"):
        with scratch_daemon() as (daemon, root, mailbox):
            lock_path = mailbox / ".dispatch.lock"
            if label == "symlink":
                target = root / "outside-lock"
                target.write_text("watch pid 7", encoding="ascii")
                os.symlink(target, lock_path)
                live = daemon.dispatch_lock_is_live_watch(str(mailbox))
            elif label == "fifo":
                os.mkfifo(lock_path)
                live = daemon.dispatch_lock_is_live_watch(str(mailbox))
            else:
                lock_path.write_text("watch pid 7", encoding="ascii")
                handle = open(lock_path, "rb")
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                try:
                    live = daemon.dispatch_lock_is_live_watch(str(mailbox))
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
            results.append(not live)
            print(label + " classified_live=" + str(live))
    for label, payload in (
            ("newline", b"watch pid 7\n"),
            ("non-ascii", b"\xffwatch pid 7"),
            ("oversized", b"x" * 129)):
        with scratch_daemon() as (daemon, _, mailbox):
            with held_raw_lock(mailbox / ".dispatch.lock", payload):
                live = daemon.dispatch_lock_is_live_watch(str(mailbox))
            results.append(not live)
            print(label + " classified_live=" + str(live))

    # A real worktree whose notes component redirects outside the repository
    # must not make that outside watcher look like an in-repository recovery
    # clue.
    with scratch_daemon() as (daemon, root, mailbox):
        redirected = root / ".claude" / "worktrees" / "redirected"
        (redirected / "ai").mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix="mailbox-outside-") as tmp:
            outside_notes = pathlib.Path(tmp)
            outside_mailbox = outside_notes / "mailbox"
            os.symlink(outside_notes, redirected / "ai" / "notes")
            with held_raw_lock(outside_mailbox / ".dispatch.lock",
                               b"watch pid 7"):
                outcome, output = captured_send(daemon)
        redirected_ok = (outcome
                         and warning_lines(output)
                         == [OWN_WARNING + str(mailbox)])
        results.append(redirected_ok)
        print("redirected-notes omitted=" + str(redirected_ok))

    # Nor may discovery walk an externally redirected worktrees base.
    with tempfile.TemporaryDirectory(prefix="mailbox-base-repro-") as tmp:
        with tempfile.TemporaryDirectory(
                prefix="mailbox-base-outside-") as outside_tmp:
            root = pathlib.Path(tmp)
            own = root / "own" / "ai" / "notes" / "mailbox"
            own.mkdir(parents=True)
            external = pathlib.Path(outside_tmp)
            external_mailbox = (external / "fake" / "ai" / "notes"
                                / "mailbox")
            daemon = load_daemon()
            daemon.REPO_ROOT = str(root)
            daemon.WORKTREE = str(root / "own")
            daemon.AI_ROOT = str(root / "own" / "ai")
            daemon.MAILBOX = str(own)
            daemon.BACKLOG_LEDGER = str(root / "missing-backlog.md")
            daemon.report_demand = lambda backlog: None
            (root / ".claude").mkdir()
            os.symlink(external, root / ".claude" / "worktrees")
            with held_raw_lock(external_mailbox / ".dispatch.lock",
                               b"watch pid 7"):
                outcome, output = captured_send(daemon)
            base_ok = (outcome
                       and warning_lines(output)
                       == [OWN_WARNING + str(own)])
            results.append(base_ok)
            print("redirected-worktrees omitted=" + str(base_ok))
    return all(results)


def arm_advisory_send_and_ping_succeed():
    """The warning never blocks atomic send publication or the ping CLI."""
    with scratch_daemon() as (daemon, _, mailbox):
        outcome, output = captured_send(
            daemon, dry_run=False, text="exact body")
        paths = daemon.pending_messages()
        send_ok = (outcome and len(paths) == 1
                   and pathlib.Path(paths[0]).read_bytes() == b"exact body\n"
                   and warning_lines(output) == [OWN_WARNING + str(mailbox)])
    with scratch_daemon() as (daemon, _, mailbox):
        previous_argv = sys.argv
        sys.argv = [str(DAEMON_PATH), "--ping", "opus"]
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream):
                rc = daemon.main()
        finally:
            sys.argv = previous_argv
        paths = daemon.pending_messages()
        ping_ok = (rc == 0 and len(paths) == 1
                   and pathlib.Path(paths[0]).read_text(
                       encoding="utf-8").startswith(
                           "RELAY CONFIRMATION PING for opus.")
                   and warning_lines(stream.getvalue())
                   == [OWN_WARNING + str(mailbox)])
    print("advisory send_ok=" + str(send_ok)
          + " ping_ok=" + str(ping_ok))
    return send_ok and ping_ok


def witness_once_warns(source):
    """Return whether a source variant still warns for a held once lock."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        with held_mode_lock(daemon, mailbox, "once"):
            outcome, output = captured_send(daemon)
        return (outcome and warning_lines(output)
                == [OWN_WARNING + str(mailbox)])


def witness_shared_probe_stays_dead(source):
    """Return whether a peer shared probe cannot masquerade as a watcher."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        lock_path = mailbox / ".dispatch.lock"
        lock_path.write_text("watch pid 7", encoding="ascii")
        handle = open(lock_path, "rb")
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            live = daemon.dispatch_lock_is_live_watch(str(mailbox))
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        return not live


def witness_dry_run_warns(source):
    """Return whether a source variant diagnoses a dry-run send."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        outcome, output = captured_send(daemon)
        return (outcome and warning_lines(output)
                == [OWN_WARNING + str(mailbox)])


def witness_other_watch_listed(source):
    """Return whether a source variant prints one live alternative."""
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        other = root / "ai" / "notes" / "mailbox"
        with held_mode_lock(daemon, other, "watch"):
            outcome, output = captured_send(daemon)
        return (outcome and warning_lines(output)
                == [OWN_WARNING + str(mailbox), OTHER_WARNING + str(other)])


def witness_owner_whitespace_rejected(source):
    """Return whether surrounding owner whitespace remains malformed."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        with held_raw_lock(mailbox / ".dispatch.lock", b" watch pid 9 "):
            outcome, output = captured_send(daemon)
        return (outcome and warning_lines(output)
                == [OWN_WARNING + str(mailbox)])


def witness_main_lock_mode(source, flag):
    """Return whether main passes the exact once/watch mode to acquisition."""
    daemon = load_daemon(source=source)
    modes = []

    class StopWatch(Exception):
        """End the otherwise infinite watch after its first loop begins."""

    daemon.acquire_dispatch_lock = (
        lambda mode="unknown": modes.append(mode) or object())
    daemon.release_dispatch_lock = lambda lock_file: None
    if flag == "once":
        daemon.process_backlog = lambda dry_run: None
    else:
        daemon.process_backlog = (
            lambda dry_run: (_ for _ in ()).throw(StopWatch()))
    previous_argv = sys.argv
    sys.argv = [str(DAEMON_PATH), "--" + flag]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                daemon.main()
            except StopWatch:
                pass
    finally:
        sys.argv = previous_argv
    return modes == [flag]


def run_source_mutations():
    """Kill load-bearing source mutations for each warning decision leg."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    cases = [
        (
            "any held lock accepted",
            'return WATCH_LOCK_OWNER_RE.fullmatch(owner) is not None',
            'return True',
            witness_once_warns,
        ),
        (
            "exclusive diagnostic probe",
            'fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)',
            'fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)',
            witness_shared_probe_stays_dead,
        ),
        (
            "dry-run diagnosis dropped",
            '        warn_if_mailbox_unwatched()\n        return True',
            '        return True',
            witness_dry_run_warns,
        ),
        (
            "other-mailbox listing dropped",
            '    for candidate in mailbox_candidates():',
            '    for candidate in []:',
            witness_other_watch_listed,
        ),
        (
            "owner whitespace stripped",
            'owner = owner_bytes.decode("ascii")',
            'owner = owner_bytes.decode("ascii").strip()',
            witness_owner_whitespace_rejected,
        ),
        (
            "once main mode dropped",
            'acquire_dispatch_lock(mode="once")',
            'acquire_dispatch_lock()',
            lambda candidate: witness_main_lock_mode(candidate, "once"),
        ),
        (
            "watch main mode dropped",
            'acquire_dispatch_lock(mode="watch")',
            'acquire_dispatch_lock()',
            lambda candidate: witness_main_lock_mode(candidate, "watch"),
        ),
    ]
    outcomes = []
    for label, anchor, replacement, witness in cases:
        mutant = source.replace(anchor, replacement, 1)
        armed = mutant != source
        baseline_green = witness(source)
        mutant_green = witness(mutant) if armed else True
        killed = armed and baseline_green and not mutant_green
        outcomes.append(killed)
        print(("PASS " if killed else "FAIL ") + "mutation " + label
              + " (armed=" + str(armed)
              + ", baseline=" + str(baseline_green)
              + ", mutant=" + str(mutant_green) + ")")
    print(str(sum(1 for passed in outcomes if passed)) + "/"
          + str(len(outcomes)) + " source mutations killed")
    return outcomes


def main():
    """Run the focused scratch-only dead-mailbox witness."""
    arms = [
        ("absent own lock warns", arm_absent_lock_warns),
        ("stale lock read-only", arm_stale_lock_is_read_only),
        ("held watch suppresses", arm_watch_suppresses_warning),
        ("once/legacy/malformed warn", arm_once_legacy_unknown_warn),
        ("other watches exact/sorted", arm_other_watches_are_exact_and_sorted),
        ("own live suppresses alternatives", arm_live_own_suppresses_alternatives),
        ("nonexistent dry-run read-only", arm_nonexistent_dry_run_is_read_only),
        ("symlink/FIFO/shared-probe safe", arm_symlink_fifo_and_shared_probe_are_safe),
        ("advisory send and ping", arm_advisory_send_and_ping_succeed),
    ]
    outcomes = []
    for label, arm in arms:
        try:
            passed = arm()
        except Exception as exc:
            print("  exception: " + type(exc).__name__ + ": " + str(exc))
            passed = False
        outcomes.append(passed)
        print(("PASS " if passed else "FAIL ") + label)
    passed_count = sum(1 for passed in outcomes if passed)
    print(str(passed_count) + "/" + str(len(arms))
          + " dead-mailbox runtime arms passed")
    mutations = run_source_mutations()
    return 0 if all(outcomes) and all(mutations) else 1


if __name__ == "__main__":
    sys.exit(main())
