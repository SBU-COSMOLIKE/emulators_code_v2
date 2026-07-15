#!/usr/bin/env python3
"""Reproduce mailbox-daemon failures without touching the live mailbox."""

import contextlib
import errno
import importlib.util
import os
import pathlib
import tempfile
import threading
import types


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"


def load_daemon():
    """Load a fresh daemon module for one isolated reproduction.

    Returns:
      The loaded module object.
    """
    spec = importlib.util.spec_from_file_location("mailbox_daemon_repro",
                                                  DAEMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def scratch_daemon():
    """Point a fresh daemon module at a temporary mailbox.

    Returns:
      A ``(daemon, scratch_root)`` pair for the ``with`` block.
    """
    with tempfile.TemporaryDirectory(prefix="mailbox-daemon-repro-") as tmp:
        root = pathlib.Path(tmp)
        ai_root = root / "ai"
        daemon = load_daemon()
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
        daemon.REPO_ROOT = str(root)
        daemon.MAILBOX = str(ai_root / "notes" / "mailbox")
        daemon.DONE = str(ai_root / "notes" / "mailbox" / "done")
        daemon.RELAY_DIR = str(ai_root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(ai_root / "notes" / "backlog.md")
        daemon.PREAMBLE = "SCRATCH MESSAGE\n"
        daemon.AGENT_COMMANDS = {
            "fable": ["/usr/bin/printf", "%s"],
            "opus": ["/usr/bin/printf", "%s"],
            "sol": ["/usr/bin/printf", "%s"],
        }
        daemon.AGENT_CWD = {
            "fable": str(root),
            "opus": str(root),
            "sol": str(root),
        }
        os.makedirs(daemon.MAILBOX, exist_ok=True)
        yield daemon, root


def write_message(daemon, name, body):
    """Write one scratch message as text or raw bytes.

    Arguments:
      daemon = the scratch daemon module.
      name   = the mailbox filename.
      body   = text or bytes to write.

    Returns:
      The written path as a string.
    """
    path = pathlib.Path(daemon.MAILBOX) / name
    if isinstance(body, bytes):
        path.write_bytes(body)
    else:
        path.write_text(body, encoding="utf-8")
    return str(path)


def clean_process(stream, output="stub ok"):
    """Return a harmless already-finished ``Popen``-shaped process.

    Arguments:
      stream = relay-log stream passed to ``subprocess.Popen``.
      output = text the harmless child appears to have written.

    Returns:
      A process-shaped namespace whose poll/wait report success.
    """
    stream.write(output)
    return types.SimpleNamespace(
        returncode=0,
        poll=lambda: 0,
        wait=lambda: 0,
        kill=lambda: None)


def arm_dry_run_is_read_only():
    """Check that refusing a placeholder during dry-run moves no file."""
    with scratch_daemon() as (daemon, _):
        path = write_message(daemon, "0001-to-opus.md", "<unit>\n")
        daemon.dispatch(path=path, dry_run=True)
        root_exists = os.path.isfile(path)
        failed_exists = os.path.exists(
            os.path.join(daemon.MAILBOX, "failed", os.path.basename(path)))
        print("dry-run root_exists=" + str(root_exists)
              + " failed_exists=" + str(failed_exists))
        return root_exists and not failed_exists


def arm_atomic_dispatch_claim():
    """Check that two dispatch callers invoke the harmless stub once."""
    with scratch_daemon() as (daemon, _):
        path = write_message(daemon, "0001-to-opus.md", "real unit\n")
        calls = []
        gate = threading.Barrier(2)

        def fake_popen(command, stdout, stderr, cwd, env):
            del stderr, cwd, env
            calls.append(command)
            try:
                gate.wait(timeout=0.25)
            except threading.BrokenBarrierError:
                pass
            return clean_process(stream=stdout)

        original_popen = daemon.subprocess.Popen
        daemon.subprocess.Popen = fake_popen
        threads = []
        try:
            for _ in range(2):
                worker = threading.Thread(target=daemon.dispatch,
                                          kwargs={"path": path,
                                                  "dry_run": False})
                worker.start()
                threads.append(worker)
            for worker in threads:
                worker.join()
        finally:
            daemon.subprocess.Popen = original_popen
        print("atomic-claim stub_calls=" + str(len(calls)))
        return len(calls) == 1


def arm_dispatch_loop_lock():
    """Check that a second dispatch loop cannot acquire the shared lock."""
    with scratch_daemon() as (daemon, _):
        first = daemon.acquire_dispatch_lock()
        second = daemon.acquire_dispatch_lock()
        blocked = first is not None and second is None
        print("dispatch-loop-lock second_blocked=" + str(blocked))
        if first is not None:
            daemon.release_dispatch_lock(lock_file=first)
        return blocked


def arm_atomic_send_publication():
    """Check that send does not expose its final path before body close."""
    with scratch_daemon() as (daemon, _):
        original_fdopen = daemon.os.fdopen
        entered = threading.Event()
        release = threading.Event()

        class BlockingWriter:
            """Pause the first write while retaining normal file behavior."""

            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __enter__(self):
                self.wrapped.__enter__()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return self.wrapped.__exit__(exc_type, exc_value, traceback)

            def write(self, value):
                entered.set()
                release.wait(timeout=2.0)
                return self.wrapped.write(value)

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

        def blocking_fdopen(handle, mode, encoding):
            wrapped = original_fdopen(handle, mode, encoding=encoding)
            return BlockingWriter(wrapped)

        daemon.os.fdopen = blocking_fdopen
        worker = threading.Thread(target=daemon.send,
                                  kwargs={"agent": "opus",
                                          "text": "complete body",
                                          "dry_run": False})
        worker.start()
        entered.wait(timeout=2.0)
        visible = daemon.pending_messages()
        print("atomic-publication visible_while_writing=" + str(len(visible)))
        release.set()
        worker.join()
        daemon.os.fdopen = original_fdopen
        return len(visible) == 0


def arm_unique_cross_recipient_sequence():
    """Check simultaneous sends cannot share one numeric sequence."""
    with scratch_daemon() as (daemon, _):
        original_glob = daemon.glob.glob
        gate = threading.Barrier(2)
        call_count = [0]
        call_lock = threading.Lock()

        def racing_glob(pattern, recursive=False):
            should_wait = False
            if recursive:
                with call_lock:
                    if call_count[0] < 2:
                        call_count[0] = call_count[0] + 1
                        should_wait = True
            if should_wait:
                try:
                    gate.wait(timeout=0.25)
                except threading.BrokenBarrierError:
                    pass
            return original_glob(pattern, recursive=recursive)

        daemon.glob.glob = racing_glob
        workers = []
        for agent in ["fable", "opus"]:
            worker = threading.Thread(target=daemon.send,
                                      kwargs={"agent": agent,
                                              "text": "real unit",
                                              "dry_run": False})
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()
        names = []
        for path in daemon.pending_messages():
            names.append(os.path.basename(path))
        sequences = []
        for name in names:
            sequences.append(name.split("-", maxsplit=1)[0])
        unique = len(sequences) == len(set(sequences))
        print("cross-recipient names=" + repr(names)
              + " unique_sequences=" + str(unique))
        return unique


def arm_five_digit_sequence_order():
    """Check that sequence 9999 is dispatched before sequence 10000."""
    with scratch_daemon() as (daemon, _):
        first = "9999-to-opus.md"
        second = "10000-to-opus.md"
        write_message(daemon, first, "first\n")
        write_message(daemon, second, "second\n")
        legacy = sorted([first, second])
        patched = []
        for path in daemon.pending_messages():
            patched.append(os.path.basename(path))
        print("five-digit legacy_order=" + repr(legacy)
              + " patched_order=" + repr(patched))
        return legacy != [first, second] and patched == [first, second]


def arm_hostile_bodies_are_parked():
    """Check malformed bodies are parked without running a command."""
    all_parked = True
    fixtures = [
        ("invalid-utf8", b"\xff\xfe\n"),
        ("nul", "unit\x00tail\n"),
        ("oversized-launch", "x" * 4096),
    ]
    for index, fixture in enumerate(fixtures, start=1):
        label, body = fixture
        with scratch_daemon() as (daemon, _):
            name = "%04d-to-opus.md" % index
            path = write_message(daemon, name, body)
            calls = []

            def hostile_stub(command, stdout, stderr, cwd, env):
                del stderr, cwd, env
                calls.append(command)
                prompt = command[-1]
                if "\x00" in prompt:
                    raise ValueError("embedded null byte")
                if len(prompt) > 1024:
                    raise OSError(errno.E2BIG, "argument list too long")
                return clean_process(stream=stdout)

            original_popen = daemon.subprocess.Popen
            daemon.subprocess.Popen = hostile_stub
            raised = "none"
            try:
                daemon.dispatch(path=path, dry_run=False)
            except Exception as exc:
                raised = type(exc).__name__
            finally:
                daemon.subprocess.Popen = original_popen
            failed = os.path.isfile(os.path.join(daemon.MAILBOX,
                                                 "failed", name))
            print("hostile-body " + label + " raised=" + raised
                  + " failed=" + str(failed)
                  + " stub_calls=" + str(len(calls)))
            if raised != "none" or not failed:
                all_parked = False
    return all_parked


def arm_literal_marker_is_not_a_placeholder():
    """Check that discussing a marker does not refuse a real review."""
    with scratch_daemon() as (daemon, _):
        body = ("MAILBOX-TICKET: closure\n\n"
                "Review why the literal <unit> marker was refused.\n")
        path = write_message(daemon, "0001-to-sol.md", body)
        calls = []

        def fake_popen(command, stdout, stderr, cwd, env):
            del stderr, cwd, env
            calls.append(command)
            return clean_process(stream=stdout)

        original_popen = daemon.subprocess.Popen
        daemon.subprocess.Popen = fake_popen
        try:
            result = daemon.dispatch(path=path, dry_run=False)
        finally:
            daemon.subprocess.Popen = original_popen
        print("literal-marker result=" + str(result)
              + " stub_calls=" + str(len(calls)))
        return result and len(calls) == 1


def main():
    """Run every scratch reproduction and return nonzero on a defect."""
    arms = [
        ("dry-run-read-only", arm_dry_run_is_read_only),
        ("atomic-dispatch-claim", arm_atomic_dispatch_claim),
        ("dispatch-loop-lock", arm_dispatch_loop_lock),
        ("atomic-send-publication", arm_atomic_send_publication),
        ("cross-recipient-sequence", arm_unique_cross_recipient_sequence),
        ("five-digit-sequence-order", arm_five_digit_sequence_order),
        ("hostile-bodies", arm_hostile_bodies_are_parked),
        ("literal-marker", arm_literal_marker_is_not_a_placeholder),
    ]
    failures = []
    for name, arm in arms:
        try:
            passed = arm()
        except Exception as exc:
            print(name + " uncaught=" + type(exc).__name__ + ": " + str(exc))
            passed = False
        verdict = "PASS" if passed else "FAIL"
        print("ARM " + name + " " + verdict)
        if not passed:
            failures.append(name)
    print("SUMMARY failures=" + repr(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
