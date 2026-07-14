#!/usr/bin/env python3
"""Scratch-only witness for the watch's global safe-kill rendezvous.

The live mailbox is infrastructure.  Every runtime arm loads a fresh daemon
and redirects every path into a temporary repository.  Most arms replace
dispatch with a bounded harmless worker; one deliberately calls production
dispatch with a fake Popen so its lifecycle hooks cannot disappear unnoticed.
"""

import contextlib
import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import threading
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = ROOT / "tools" / "mailbox_daemon.py"
COUNTDOWN_SECONDS = 20


class AttributeProxy:
    """Delegate module attributes except for explicit test overrides."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


class ImmediateProcess:
    """A harmless already-reaped Popen result for production integration."""

    def __init__(self):
        self.returncode = 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self):
        return self.returncode


def load_daemon(source=None):
    """Load a fresh production daemon, optionally from mutated source."""
    if source is not None:
        module = types.ModuleType("mailbox_daemon_rendezvous_mutant")
        module.__file__ = str(DAEMON_PATH)
        exec(compile(source, str(DAEMON_PATH), "exec"), module.__dict__)
        return module
    spec = importlib.util.spec_from_file_location(
        "mailbox_daemon_rendezvous_repro", DAEMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def scratch_daemon(source=None):
    """Redirect a fresh daemon into one disposable repository."""
    with tempfile.TemporaryDirectory(prefix="mailbox-rendezvous-") as tmp:
        root = pathlib.Path(tmp)
        mailbox = root / "notes" / "mailbox"
        mailbox.mkdir(parents=True)
        backlog = root / "notes" / "backlog.md"
        backlog.write_text("", encoding="utf-8")
        relay = root / "notes" / "relay"

        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(root)
        daemon.MAILBOX = str(mailbox)
        daemon.DONE = str(mailbox / "done")
        daemon.RELAY_DIR = str(relay)
        daemon.BACKLOG_LEDGER = str(backlog)
        daemon.PREAMBLE = "scratch preamble\n"
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"],
            "opus": ["harmless-opus"],
            "sol": ["harmless-sol"],
        }
        daemon.AGENT_CWD = {
            "fable": str(root / "shared-lane"),
            "opus": str(root / "shared-lane"),
            "sol": str(root / "sol-lane"),
        }
        daemon.warn_if_mailbox_unwatched = lambda: None
        daemon.report_demand = lambda backlog: None
        daemon.report_landing_debt = lambda: None
        yield daemon, root, mailbox


def write_pending(mailbox, name, body="close existing work\n"):
    """Create one exact root-pending message and return its path."""
    path = mailbox / name
    path.write_text(body, encoding="utf-8", newline="")
    return path


def tree_snapshot(root):
    """Return a byte-and-type snapshot of a scratch tree."""
    snapshot = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot.append((relative, "symlink", os.readlink(path)))
        elif path.is_file():
            snapshot.append((relative, "file", path.read_bytes()))
        elif path.is_dir():
            snapshot.append((relative, "dir", b""))
        else:
            snapshot.append((relative, "other", b""))
    return snapshot


def call_main(daemon, arguments):
    """Call main with isolated argv and capture BaseException for Ctrl-C."""
    previous_argv = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = None
    error = None
    sys.argv = ["mailbox_daemon.py"] + list(arguments)
    try:
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            try:
                result = daemon.main()
            except BaseException as exc:  # KeyboardInterrupt is evidence
                error = exc
    finally:
        sys.argv = previous_argv
    rc = 0 if result is None and error is None else result
    return rc, stdout.getvalue(), stderr.getvalue(), error


def start_main_thread(daemon, arguments):
    """Start a bounded watch invocation and return its result containers."""
    result = {}

    def target():
        rc, output, error_output, error = call_main(daemon, arguments)
        result.update({
            "rc": rc,
            "output": output,
            "error_output": error_output,
            "error": error,
        })

    worker = threading.Thread(
        target=target, name="rendezvous-watch-main", daemon=True)
    worker.start()
    return worker, result


def countdown_lines(output):
    """Return only the exact safe-window countdown lines."""
    return [line for line in output.splitlines()
            if line.startswith("all lanes idle; safe to Ctrl-C for ")
            and "s more;" in line]


def expected_countdown(waiting):
    """Return the binding 19..0 twenty-line countdown."""
    noun = "message" if waiting == 1 else "messages"
    return [
        "all lanes idle; safe to Ctrl-C for " + str(remaining)
        + "s more; " + str(waiting) + " " + noun + " waiting."
        for remaining in range(COUNTDOWN_SECONDS - 1, -1, -1)
    ]


@contextlib.contextmanager
def simulated_child_turn(daemon):
    """Make a fake dispatch exercise the production child-lifecycle hooks."""
    daemon._rendezvous_turn_started()
    try:
        yield
    finally:
        daemon._rendezvous_turn_finished()


class ControlledClock:
    """Monotonic fake clock with bounded poll and countdown sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.events = []
        self.abort = threading.Event()
        self.countdown_started = threading.Event()
        self.release_countdown = threading.Event()
        self.block_first_countdown = False
        self.raise_on_countdown = False
        self.raise_on_poll = False
        self.on_poll = None
        self.polls = 0

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        if self.abort.is_set():
            raise KeyboardInterrupt("bounded witness abort")
        self.sleeps.append(seconds)
        if seconds == 1:
            self.events.append("countdown")
            self.countdown_started.set()
            if self.raise_on_countdown:
                raise KeyboardInterrupt("Ctrl-C during safe window")
            if self.block_first_countdown and not self.release_countdown.is_set():
                if not self.release_countdown.wait(timeout=2.0):
                    raise TimeoutError("countdown witness was not released")
        elif seconds == 20:
            self.polls = self.polls + 1
            self.events.append("poll")
            if self.on_poll is not None:
                self.on_poll(self)
            if self.raise_on_poll:
                raise KeyboardInterrupt("fallback Ctrl-C during poll")
            if self.polls > 8:
                raise KeyboardInterrupt("watch failed to terminate")
        else:
            self.events.append("sleep:" + repr(seconds))
        self.now = self.now + seconds


def install_clock_and_stamp(daemon, clock, stop_when):
    """Install fake time and a source stamp that exits after the scenario."""
    real_os = daemon.os
    initial_stamp = 101.0

    def getmtime(path):
        del path
        if stop_when():
            return initial_stamp + 1.0
        return initial_stamp

    daemon.time = AttributeProxy(
        daemon.time, time=clock.time, monotonic=clock.monotonic,
        sleep=clock.sleep)
    daemon.os = AttributeProxy(
        real_os, path=AttributeProxy(real_os.path, getmtime=getmtime))


def arm_constants_are_named(source=None):
    """Both cadence knobs are named, positive, and countdown is exactly 20."""
    daemon = load_daemon(source=source)
    dispatches = getattr(daemon, "RENDEZVOUS_DISPATCH_INTERVAL", None)
    minutes = getattr(daemon, "RENDEZVOUS_MINUTE_INTERVAL", None)
    countdown = getattr(daemon, "SAFE_KILL_COUNTDOWN_SECONDS", None)
    passed = (
        isinstance(dispatches, int) and not isinstance(dispatches, bool)
        and dispatches > 0
        and isinstance(minutes, (int, float))
        and not isinstance(minutes, bool) and minutes > 0
        and countdown == COUNTDOWN_SECONDS)
    print("named rendezvous constants=" + str(passed))
    return passed


def arm_dispatch_cadence_global_window(source=None):
    """At K completions, all lanes drain and dynamic queue counts stay live."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 2
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        first = write_pending(mailbox, "0001-to-fable.md")
        second = write_pending(
            mailbox, "0002-to-sol.md",
            "MAILBOX-TICKET: closure\n\nclose existing work\n")
        third = write_pending(mailbox, "0003-to-fable.md")

        clock = ControlledClock()
        clock.block_first_countdown = True
        releases = [threading.Event(), threading.Event()]
        finished = [threading.Event(), threading.Event()]
        two_started = threading.Event()
        all_finished = threading.Event()
        launch_lock = threading.Lock()
        launches = []
        dispatch_errors = []
        fourth_holder = []

        def fake_dispatch(path, dry_run, fix_only=False):
            del dry_run, fix_only
            with simulated_child_turn(daemon):
                with launch_lock:
                    index = len(launches)
                    launches.append(pathlib.Path(path).name)
                    clock.events.append("launch:" + pathlib.Path(path).name)
                    if len(launches) >= 2:
                        two_started.set()
                if index < 2 and not releases[index].wait(timeout=2.0):
                    dispatch_errors.append("release timeout " + str(index))
                    return False
                target = pathlib.Path(path)
                if target.exists():
                    target.unlink()
                if index < 2:
                    finished[index].set()
                if fourth_holder and target == fourth_holder[0]:
                    all_finished.set()
                return True

        daemon.dispatch = fake_dispatch
        install_clock_and_stamp(
            daemon, clock=clock, stop_when=all_finished.is_set)
        worker, result = start_main_thread(daemon, ["--watch"])

        first_wave = two_started.wait(timeout=2.0)
        launches_at_cap = list(launches)
        if first_wave:
            releases[0].set()
        one_finished = finished[0].wait(timeout=1.0)
        still_draining = (
            one_finished and not clock.countdown_started.is_set()
            and len(launches) == 2)
        releases[1].set()
        window_started = clock.countdown_started.wait(timeout=2.0)
        held_in_window = (
            window_started and len(launches) == 2 and third.exists())
        fourth = write_pending(mailbox, "0004-to-fable.md")
        fourth_holder.append(fourth)
        dynamic_added = fourth.exists()
        clock.release_countdown.set()
        worker.join(timeout=5.0)
        if worker.is_alive():
            clock.abort.set()
            releases[0].set()
            releases[1].set()
            clock.release_countdown.set()
            worker.join(timeout=2.0)

        output = result.get("output", "")
        lines = countdown_lines(output)
        one_second_sleeps = [value for value in clock.sleeps if value == 1]
        status_truth = (
            "1 turn in flight; not safe to stop." in output
            and "2 turns in flight; not safe to stop." in output)
        passed = (
            first_wave and launches_at_cap == launches[:2]
            and len(launches_at_cap) == 2
            and still_draining and held_in_window
            and not worker.is_alive() and result.get("error") is None
            and result.get("rc") == 0 and dispatch_errors == []
            and lines == (expected_countdown(waiting=1)[:1]
                          + expected_countdown(waiting=2)[1:])
            and one_second_sleeps == [1] * COUNTDOWN_SECONDS
            and dynamic_added and status_truth and len(launches) == 4
            and set(launches) == {
                first.name, second.name, third.name, fourth.name}
            and not first.exists() and not second.exists()
            and not third.exists() and not fourth.exists())
        print("K-cadence global drain/window/resume=" + str(passed))
        return passed


def arm_minute_cadence_and_idle_status(source=None):
    """Elapsed M during a turn drains first; an empty poll is already safe."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 999
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1
        clock = ControlledClock()
        launches = []
        queued = []
        finished = threading.Event()

        def add_work_after_first_idle_window(fake_clock):
            if not queued:
                queued.append(write_pending(mailbox, "0001-to-fable.md"))
                queued.append(write_pending(mailbox, "0002-to-fable.md"))
                fake_clock.events.append("queued")

        clock.on_poll = add_work_after_first_idle_window

        def fake_dispatch(path, dry_run, fix_only=False):
            del dry_run, fix_only
            with simulated_child_turn(daemon):
                launches.append(pathlib.Path(path).name)
                clock.events.append("launch")
                if len(launches) == 1:
                    # The first turn crosses the minute deadline while live.
                    clock.now = 60.0
                pathlib.Path(path).unlink()
                if len(launches) == 2:
                    finished.set()
                return True

        daemon.dispatch = fake_dispatch
        install_clock_and_stamp(
            daemon, clock=clock, stop_when=finished.is_set)
        rc, output, error_output, error = call_main(daemon, ["--watch"])
        del error_output
        lines = countdown_lines(output)
        idle_lines = [line for line in output.splitlines()
                      if "all lanes idle" in line
                      and "safe to Ctrl-C" in line
                      and "s more;" not in line]
        try:
            queued_at = clock.events.index("queued")
            first_countdown = clock.events.index("countdown", queued_at)
            first_launch = clock.events.index("launch", queued_at)
            second_launch = clock.events.index("launch", first_launch + 1)
            ordered = (queued_at < first_launch < first_countdown
                       < second_launch)
        except ValueError:
            ordered = False
        passed = (
            error is None and rc == 0 and len(launches) == 2
            and lines == expected_countdown(waiting=1)
            and [value for value in clock.sleeps if value == 1]
            == [1] * COUNTDOWN_SECONDS
            and ordered and len(idle_lines) >= 1
            and any("no messages waiting" in line for line in idle_lines)
            and queued and not any(path.exists() for path in queued))
        print("M-cadence plus ordinary-idle status=" + str(passed))
        return passed


def arm_safe_line_expires_before_claim(source=None):
    """A flushed unsafe line precedes claim after an ordinary safe poll."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 999
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        clock = ControlledClock()
        queued = []
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        observed = {}

        def add_work_during_safe_poll(fake_clock):
            del fake_clock
            if not queued:
                queued.append(write_pending(
                    mailbox, "0001-to-fable.md", "exact queued bytes\n"))

        clock.on_poll = add_work_during_safe_poll

        def fake_dispatch(path, dry_run, fix_only=False):
            del dry_run, fix_only
            visible = sys.stdout.getvalue()
            target = pathlib.Path(path)
            observed["unsafe"] = (
                "dispatch preparation admitted; not safe to stop."
                in visible)
            observed["untouched"] = (
                target.exists() and target.read_bytes()
                == b"exact queued bytes\n")
            entered.set()
            if not release.wait(timeout=2.0):
                return False
            with simulated_child_turn(daemon):
                target.unlink()
            finished.set()
            return True

        daemon.dispatch = fake_dispatch
        install_clock_and_stamp(
            daemon, clock=clock, stop_when=finished.is_set)
        worker, result = start_main_thread(daemon, ["--watch"])
        blocked_before_child = entered.wait(timeout=2.0)
        proof_before_release = dict(observed)
        release.set()
        worker.join(timeout=4.0)
        if worker.is_alive():
            clock.abort.set()
            release.set()
            worker.join(timeout=2.0)

        output = result.get("output", "")
        try:
            safe_index = output.index(
                "all lanes idle; safe to Ctrl-C for this 20s poll; "
                "no messages waiting.")
            closed_index = output.index(
                "safe interval ended; not safe to stop.", safe_index + 1)
            unsafe_index = output.index(
                "dispatch preparation admitted; not safe to stop.",
                closed_index + 1)
            ordered = safe_index < closed_index < unsafe_index
        except ValueError:
            ordered = False
        passed = (
            blocked_before_child and proof_before_release.get("unsafe")
            and proof_before_release.get("untouched") and ordered
            and not worker.is_alive() and result.get("error") is None
            and result.get("rc") == 0 and queued
            and not queued[0].exists())
        print("ordinary safe line expires before claim=" + str(passed))
        return passed


def arm_ctrl_c_preserves_waiting_message(source=None):
    """Ctrl-C in the main-thread window leaves undispatched bytes intact."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 1
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        first = write_pending(mailbox, "0001-to-fable.md", "first body\n")
        second = write_pending(mailbox, "0002-to-fable.md", "second body\n")
        second_bytes = second.read_bytes()
        launches = []
        clock = ControlledClock()
        clock.raise_on_countdown = True
        clock.raise_on_poll = True

        def fake_dispatch(path, dry_run, fix_only=False):
            del dry_run, fix_only
            with simulated_child_turn(daemon):
                launches.append(pathlib.Path(path).name)
                pathlib.Path(path).unlink()
                return True

        daemon.dispatch = fake_dispatch
        install_clock_and_stamp(daemon, clock=clock, stop_when=lambda: False)
        rc, output, _, error = call_main(daemon, ["--watch"])
        terminated = (isinstance(error, KeyboardInterrupt)
                      or rc in (0, 130))
        retry_lock = daemon.acquire_dispatch_lock(mode="once")
        released = retry_lock is not None
        if retry_lock is not None:
            daemon.release_dispatch_lock(lock_file=retry_lock)
        passed = (
            terminated and launches == [first.name]
            and not first.exists() and second.exists()
            and second.read_bytes() == second_bytes
            and released
            and countdown_lines(output)
            == [expected_countdown(waiting=1)[0]])
        print("Ctrl-C window preserves queued message=" + str(passed))
        return passed


def arm_source_change_stops_mid_pass(source=None):
    """A source edit after one turn prevents the lane's next root claim."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 999
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        first = write_pending(mailbox, "0001-to-fable.md", "first body\n")
        second = write_pending(mailbox, "0002-to-fable.md", "second body\n")
        second_bytes = second.read_bytes()
        changed = threading.Event()
        launches = []
        clock = ControlledClock()

        def fake_dispatch(path, dry_run, fix_only=False):
            del dry_run, fix_only
            with simulated_child_turn(daemon):
                target = pathlib.Path(path)
                launches.append(target.name)
                target.unlink()
                changed.set()
                return True

        daemon.dispatch = fake_dispatch
        install_clock_and_stamp(
            daemon, clock=clock, stop_when=changed.is_set)
        rc, output, _, error = call_main(daemon, ["--watch"])
        passed = (
            error is None and rc == 0 and launches == [first.name]
            and not first.exists() and second.exists()
            and second.read_bytes() == second_bytes
            and "daemon source changed on disk" in output)
        print("source change stops mid-pass=" + str(passed))
        return passed


def arm_production_dispatch_lifecycle(source=None):
    """Production dispatch registers and reaps its fake Popen child."""
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 1
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        message = write_pending(mailbox, "0001-to-fable.md", "real body\n")
        for cwd in set(daemon.AGENT_CWD.values()):
            pathlib.Path(cwd).mkdir(parents=True, exist_ok=True)
        calls = []

        def fake_popen(command, stdout, stderr, cwd, env):
            calls.append((list(command), stderr, cwd, dict(env)))
            stdout.write("fake child completed\n")
            stdout.flush()
            return ImmediateProcess()

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=fake_popen)
        controller = daemon.SafeKillRendezvous()
        daemon._ACTIVE_WATCH_RENDEZVOUS = controller
        output = io.StringIO()
        error = None
        consumed = False
        try:
            with contextlib.redirect_stdout(output):
                consumed = daemon.drain_lane(
                    paths=[str(message)], dry_run=False)
        except BaseException as exc:
            error = exc
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None

        done = root / "notes" / "mailbox" / "done" / message.name
        visible = output.getvalue()
        passed = (
            error is None and consumed and len(calls) == 1
            and not message.exists() and done.is_file()
            and controller.window_ready()
            and "dispatch preparation admitted; not safe to stop." in visible
            and "1 turn in flight; not safe to stop." in visible)
        print("production Popen lifecycle hooks=" + str(passed))
        return passed


def finite_action_case(source, arguments, dry_run):
    """Run one finite action and return whether rendezvous stayed absent."""
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 1
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1
        first = write_pending(mailbox, "0001-to-fable.md")
        second = write_pending(
            mailbox, "0002-to-sol.md",
            "MAILBOX-TICKET: closure\n\nclose existing work\n")
        before = tree_snapshot(root)
        launches = []
        real_time = daemon.time

        def forbidden_sleep(seconds):
            raise AssertionError("finite action slept " + repr(seconds))

        def fake_dispatch(path, dry_run, fix_only=False):
            del fix_only
            launches.append(pathlib.Path(path).name)
            if not dry_run:
                pathlib.Path(path).unlink()
            return True

        daemon.time = AttributeProxy(real_time, sleep=forbidden_sleep)
        daemon.dispatch = fake_dispatch
        rc, output, _, error = call_main(daemon, arguments)
        if dry_run:
            filesystem_ok = tree_snapshot(root) == before
        else:
            filesystem_ok = not first.exists() and not second.exists()
        return (
            error is None and rc == 0 and len(launches) == 2
            and set(launches) == {first.name, second.name}
            and filesystem_ok and countdown_lines(output) == []
            and "turns in flight; not safe to stop." not in output)


def arm_once_and_dry_run_are_unaffected(source=None):
    """Finite modes neither rendezvous nor acquire watch-only behavior."""
    once = finite_action_case(
        source=source, arguments=["--once"], dry_run=False)
    dry = finite_action_case(
        source=source, arguments=["--dry-run"], dry_run=True)
    passed = once and dry
    print("once/dry-run unaffected=" + str(passed))
    return passed


def replace_exact(source, old, new):
    """Return a single-site source mutant, else None when not armed."""
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def arm_source_mutations():
    """Kill rendezvous boundary mutants once production exposes anchors."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    cases = [
        (
            "countdown shortened to 19",
            lambda text: replace_exact(
                text, "SAFE_KILL_COUNTDOWN_SECONDS = 20",
                "SAFE_KILL_COUNTDOWN_SECONDS = 19"),
            arm_dispatch_cadence_global_window,
        ),
        (
            "in-flight status lies safe",
            lambda text: replace_exact(
                text,
                '+ " in flight; not safe to stop.", flush=True)',
                '+ " in flight; safe to stop.", flush=True)'),
            arm_dispatch_cadence_global_window,
        ),
        (
            "pre-claim unsafe transition omitted",
            lambda text: replace_exact(
                text,
                "                report_admitted_status()\n",
                "                pass  # omitted unsafe transition\n"),
            arm_safe_line_expires_before_claim,
        ),
        (
            "production child-start hook omitted",
            lambda text: replace_exact(
                text, "            _rendezvous_turn_started()\n", ""),
            arm_production_dispatch_lifecycle,
        ),
        (
            "production child-finish hook omitted",
            lambda text: replace_exact(
                text, "                        _rendezvous_turn_finished()\n",
                "                        pass  # child finish omitted\n"),
            arm_production_dispatch_lifecycle,
        ),
        (
            "source change ignored at admission",
            lambda text: replace_exact(
                text,
                "                self._stop_for_source_change_locked()\n",
                "                pass  # source change ignored\n"),
            arm_source_change_stops_mid_pass,
        ),
        (
            "countdown queue count frozen",
            lambda text: replace_exact(
                text,
                "    for seconds_more in range(SAFE_KILL_COUNTDOWN_SECONDS - 1, -1, -1):\n"
                "        waiting = len(pending_messages())\n",
                "    for seconds_more in range(SAFE_KILL_COUNTDOWN_SECONDS - 1, -1, -1):\n"
                "        waiting = 1\n"),
            arm_dispatch_cadence_global_window,
        ),
    ]
    failures = []
    for label, mutator, probe in cases:
        mutant = mutator(source)
        armed = mutant is not None and mutant != source
        baseline = probe(source) if armed else False
        mutant_passed = probe(mutant) if armed and baseline else True
        killed = armed and baseline and not mutant_passed
        print("MUTATION " + label + " armed=" + str(armed)
              + " baseline=" + str(baseline)
              + " killed=" + str(killed))
        if not killed:
            failures.append(label)
    print("mutation-summary killed=" + str(len(cases) - len(failures))
          + "/" + str(len(cases)) + " failures=" + repr(failures))
    return not failures


def main():
    """Run every black-box rendezvous arm and planned source mutation."""
    arms = [
        ("named constants", arm_constants_are_named),
        ("K global window", arm_dispatch_cadence_global_window),
        ("M and idle status", arm_minute_cadence_and_idle_status),
        ("ordinary-safe expiry", arm_safe_line_expires_before_claim),
        ("Ctrl-C preservation", arm_ctrl_c_preserves_waiting_message),
        ("mid-pass source change", arm_source_change_stops_mid_pass),
        ("production dispatch hooks", arm_production_dispatch_lifecycle),
        ("finite modes", arm_once_and_dry_run_are_unaffected),
        ("source mutations", arm_source_mutations),
    ]
    failures = []
    for name, arm in arms:
        try:
            passed = arm()
        except BaseException as exc:
            print("ERROR " + name + ": " + type(exc).__name__
                  + ": " + str(exc))
            passed = False
        print(("PASS " if passed else "FAIL ") + name)
        if not passed:
            failures.append(name)
    print("runtime-summary passed=" + str(len(arms) - len(failures))
          + "/" + str(len(arms)) + " failures=" + repr(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
