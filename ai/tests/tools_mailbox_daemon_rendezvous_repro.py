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


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
if str(AI_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(AI_ROOT.parent))
try:
    from ai.tests import tools_mailbox_daemon_fix_only_repro as fix_only_repro
except ImportError:
    import tools_mailbox_daemon_fix_only_repro as fix_only_repro
daemon_source_files = fix_only_repro.daemon_source_files
COUNTDOWN_SECONDS = 20
BASE_COMMIT = "1" * 40
ACCEPTED_COMMIT = "2" * 40


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
    """Load a fresh production daemon, optionally from mutated source.

    Delegates to the shared loader, which accepts None, an entry-file
    mutant string, or a mapping of daemon file name to source text.
    """
    return fix_only_repro.load_daemon(source=source)


def install_test_role_state_proofs(daemon):
    """Install opaque topology and persistent-state proofs for each role."""
    agents = ("fable", "opus", "sol")
    topology_proofs = {agent: object() for agent in agents}
    persistent_proofs = {agent: object() for agent in agents}

    def validate_topology(agent):
        return topology_proofs[agent]

    def revalidate_topology(proof):
        if proof not in topology_proofs.values():
            raise AssertionError("scratch role topology proof changed")
        return proof

    def capture_persistent_state(agent):
        return persistent_proofs[agent]

    def recheck_persistent_state(proof):
        if proof not in persistent_proofs.values():
            raise AssertionError("scratch persistent role state changed")
        return proof

    daemon.validate_live_agent_dispatch_topology = validate_topology
    daemon.revalidate_agent_dispatch_topology = revalidate_topology
    daemon.capture_persistent_role_state = capture_persistent_state
    daemon.recheck_persistent_role_state = recheck_persistent_state


@contextlib.contextmanager
def scratch_daemon(source=None):
    """Redirect a fresh daemon into one disposable repository."""
    with tempfile.TemporaryDirectory(prefix="mailbox-rendezvous-") as tmp:
        # Resolved so no path crosses a symlink (macOS /var redirects);
        # the daemon's source-note checks refuse redirected paths.
        root = pathlib.Path(tmp).resolve()
        ai_root = root / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        mailbox.mkdir(parents=True)
        backlog = ai_root / "notes" / "backlog.md"
        backlog.write_text("", encoding="utf-8")
        # Every watch pass re-reads the role contract from the redirected
        # repository root, so the scratch tree carries the real one.
        (ai_root / "notes" / "role-contract.yaml").write_bytes(
            (AI_ROOT / "notes" / "role-contract.yaml").read_bytes())
        relay = ai_root / "notes" / "relay"

        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
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
            "fable": str(root / "architect-lane"),
            "opus": str(root / "implementer-lane"),
            "sol": str(root / "sol-lane"),
        }
        # Journaling an Architect outcome loads the authoritative handoff
        # contract from the Architect lane.
        architect_tools = root / "architect-lane" / "ai" / "tools"
        architect_tools.mkdir(parents=True)
        (architect_tools / "handoff_contract.py").write_bytes(
            (AI_ROOT / "tools" / "handoff_contract.py").read_bytes())
        daemon._production_warn_if_mailbox_unwatched = (
            daemon.warn_if_mailbox_unwatched)
        daemon.warn_if_mailbox_unwatched = lambda: None
        daemon.report_demand = lambda backlog, **kwargs: None
        daemon.report_landing_debt = lambda: None
        install_test_role_state_proofs(daemon=daemon)
        # Ticket-cycle registration is exercised only with synthetic commit
        # identities in this disposable repository.  Keep ancestry checks
        # deterministic and isolated from the caller's real checkout.
        daemon.git_commit_exists = (
            lambda commit: commit in {BASE_COMMIT, ACCEPTED_COMMIT})
        daemon.git_commit_descends_from = (
            lambda starting_commit, accepted_commit:
            starting_commit == BASE_COMMIT
            and accepted_commit == ACCEPTED_COMMIT)
        daemon._exact_git_object = (
            lambda arguments, label: BASE_COMMIT
            if arguments == ["rev-parse", "--verify",
                             "refs/heads/main^{commit}"]
            else ACCEPTED_COMMIT)
        # The scratch root is not a Git repository, so the Implementer
        # authority probes and private cycle refs are answered
        # synthetically.
        daemon.implementer_authority_snapshot = (
            lambda repository_root=None: {})
        daemon.implementer_authority_changes = (
            lambda before, repository_root=None: [])
        daemon.git_ref_commit = lambda reference: None
        # A Sol closure dispatch reads the reopen ledger before launch;
        # these rendezvous witnesses do not exercise that Git-facing proof.
        daemon.current_reopen_ticket = lambda cycle_id: None
        daemon._REOPEN_TRANSITION.redteam_brief = (
            lambda ticket, cycle, landing: "")
        # Scratch agent commands are bare stand-ins with no reasoning
        # effort option, so the routine-review effort swap cannot apply.
        daemon.routine_review_command = (
            lambda command_prefix, **_kwargs: (list(command_prefix), None))
        yield daemon, root, mailbox


def write_pending(mailbox, name, body="close existing work\n"):
    """Create one exact root-pending message and return its path."""
    path = mailbox / name
    path.write_text(body, encoding="utf-8", newline="")
    return path


def write_indexed_open_tickets(backlog, anchors=("cycle-witness",)):
    """Write classified Open tickets with exact cycle bookkeeping."""
    index = []
    details = []
    for anchor in anchors:
        title = anchor.replace("-", " ").title()
        index.append(
            "- OPEN **MEDIUM** **BUG FIX** — [" + title + "](#"
            + anchor + ")\n")
        details.extend([
            "\n<a id=\"" + anchor + "\"></a>\n",
            "**Red Team reopen count: 0.**\n",
            "**Red Team reopening: allowed.**\n",
        ])
    backlog.write_text("".join(index + details), encoding="utf-8")


def write_indexed_open_ticket(backlog):
    """Keep the original one-ticket helper used by cycle-zero arms."""
    write_indexed_open_tickets(backlog=backlog)


def ticket_flow_payload(
        cycle_id, mode="normal",
        text="Change the named fixture and run its focused regression."):
    """Return one exact Architect/Implementer cycle message."""
    return (
        "MAILBOX-FLOW: ticket\n"
        "MAILBOX-CYCLE: " + cycle_id + "\n"
        "MAILBOX-MODE: " + mode + "\n\n"
        + text + "\n")


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
            if line.startswith(
                "every enabled role is idle; safe to Ctrl-C for ")
            and "s more;" in line]


def expected_countdown(waiting):
    """Return the binding 19..0 twenty-line countdown."""
    if waiting == 0:
        count_text = "no messages waiting"
    else:
        noun = "message" if waiting == 1 else "messages"
        count_text = str(waiting) + " " + noun + " waiting"
    return [
        "every enabled role is idle; safe to Ctrl-C for " + str(remaining)
        + "s more; " + count_text + "."
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
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 2
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        first = write_pending(mailbox, "0001-to-fable.md")
        second = write_pending(mailbox, "0002-to-opus.md")
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

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
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

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
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
                      if "every enabled role is idle" in line
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

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
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
                "every enabled role is idle; safe to Ctrl-C for this "
                "20s poll; "
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

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
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

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
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

        def fake_popen(command, stdout, stderr, cwd, env,
                   start_new_session=False):
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

        done = root / "ai" / "notes" / "mailbox" / "done" / message.name
        visible = output.getvalue()
        passed = (
            error is None and consumed and len(calls) == 1
            and not message.exists() and done.is_file()
            and controller.window_ready()
            and "dispatch preparation admitted; not safe to stop." in visible
            and "1 turn in flight; not safe to stop." in visible)
        print("production Popen lifecycle hooks=" + str(passed))
        return passed


def arm_no_child_launch_is_restartable(source=None):
    """A Popen failure is recoverable without resembling a failed turn."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        request = write_pending(mailbox, "0001-to-fable.md", "real body\n")
        for cwd in set(daemon.AGENT_CWD.values()):
            pathlib.Path(cwd).mkdir(parents=True, exist_ok=True)

        def refused_popen(command, stdout, stderr, cwd, env,
                      start_new_session=False):
            del command, stdout, stderr, cwd, env
            raise OSError("provider process did not start")

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=refused_popen)
        consumed = daemon.dispatch(path=str(request), dry_run=False)
        held = mailbox / "prelaunch" / request.name
        retained = (consumed is False and held.is_file()
                    and not (mailbox / "failed" / request.name).exists())
        recovered = daemon.recover_prelaunch_messages()
        passed = (retained and recovered == 1 and request.is_file()
                  and not held.exists())
        print("no-child launch is restartable=" + str(passed))
        return passed


def arm_candidate_audit_requires_one_outcome(source=None):
    """Require a landing GO or one repair handoff after candidate audit."""
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        anchor = "silent-candidate-audit"
        cycle_id = anchor + "@" + BASE_COMMIT
        write_indexed_open_tickets(
            backlog=pathlib.Path(daemon.BACKLOG_LEDGER), anchors=(anchor,))
        flow = ticket_flow_payload(
            cycle_id=cycle_id, text="Audit the saved candidate.")
        daemon.register_ticket_cycle_message(agent="opus", message=flow)
        daemon.candidate_commit_for_cycle = lambda cycle_id: ACCEPTED_COMMIT
        daemon.candidate_record_locked = (
            lambda cycle_id, ticket_state, candidate_state:
            {"ref": "refs/mailbox/scratch/candidate",
             "commit": ACCEPTED_COMMIT})
        daemon.create_audit_snapshot = (
            lambda **kwargs: str(root / "candidate-audit"))
        daemon.remove_audit_snapshot = lambda **kwargs: None
        for cwd in set(daemon.AGENT_CWD.values()):
            pathlib.Path(cwd).mkdir(parents=True, exist_ok=True)
        daemon.capture_persistent_role_state = (
            lambda agent: {"agent": agent, "base": BASE_COMMIT}
            if agent == "fable" else object())
        daemon.recheck_persistent_role_state = lambda proof: None
        daemon.worktree_head = lambda worktree: BASE_COMMIT
        daemon._validate_current_protected_primary_state = (
            lambda primary_worktree: None)

        def silent_popen(command, stdout, stderr, cwd, env,
                     start_new_session=False):
            del command, stderr, cwd, env
            stdout.write("Architect exited without an outcome.\n")
            stdout.flush()
            return ImmediateProcess()

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=silent_popen)
        request = write_pending(mailbox, "0002-to-fable.md", flow)
        consumed = daemon.dispatch(path=str(request), dry_run=False)
        state = daemon.read_ticket_cycle_state()
        silent_refused = (
            consumed is False
            and not (mailbox / "done" / request.name).exists()
            and (mailbox / "failed" / request.name).is_file()
            and state["active"][cycle_id]["phase"] == "implementation")

        # The journaled repair handoff must cite a real directive note in
        # the Architect lane.
        from ai.tests.test_handoff_contract import packet
        architect_note = (pathlib.Path(daemon.AGENT_CWD["fable"])
                          / "ai" / "notes" / "ticket.md")
        architect_note.parent.mkdir(parents=True, exist_ok=True)
        architect_note.write_text(
            packet(role="architect"), encoding="utf-8", newline="")
        repair = ticket_flow_payload(
            cycle_id=cycle_id,
            text=("- **Directive:** [ai/notes/ticket.md, exact "
                  "Implementation directive section]"))

        def repair_popen(command, stdout, stderr, cwd, env,
                     start_new_session=False):
            del command, stderr, cwd, env
            write_pending(mailbox, "0004-to-opus.md", repair)
            stdout.write("Architect requested one repair.\n")
            stdout.flush()
            return ImmediateProcess()

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=repair_popen)
        retry = write_pending(mailbox, "0003-to-fable.md", flow)
        repaired = daemon.dispatch(path=str(retry), dry_run=False)
        passed = (
            silent_refused and repaired
            and (mailbox / "done" / retry.name).is_file()
            and (mailbox / "0004-to-opus.md").is_file())
        print("candidate audit requires one outcome=" + str(passed))
        return passed


def arm_three_role_pipeline_concurrency(source=None):
    """Three role lanes overlap while each role remains sequential."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        anchors = ("pipeline-first", "pipeline-second")
        write_indexed_open_tickets(
            backlog=pathlib.Path(daemon.BACKLOG_LEDGER), anchors=anchors)
        architect = write_pending(
            mailbox, "0001-to-fable.md", "Audit frozen candidate A.\n")
        implementer_first = write_pending(
            mailbox, "0002-to-opus.md",
            ticket_flow_payload(
                cycle_id=anchors[0] + "@" + BASE_COMMIT,
                text="Implement ticket B."))
        red_team = write_pending(
            mailbox, "0003-to-sol.md", "Review older accepted ticket.\n")
        implementer_second = write_pending(
            mailbox, "0004-to-opus.md",
            ticket_flow_payload(
                cycle_id=anchors[1] + "@" + BASE_COMMIT,
                text="Implement the next ticket."))

        releases = {
            architect.name: threading.Event(),
            implementer_first.name: threading.Event(),
            red_team.name: threading.Event(),
        }
        three_started = threading.Event()
        second_started = threading.Event()
        launch_lock = threading.Lock()
        launches = []
        first_names = {
            architect.name, implementer_first.name, red_team.name}

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
            target = pathlib.Path(path)
            agent = daemon.PENDING_MESSAGE_RE.match(target.name).group(1)
            with launch_lock:
                launches.append((target.name, daemon.AGENT_CWD[agent]))
                started_names = {name for name, _cwd in launches}
                if first_names.issubset(started_names):
                    three_started.set()
                if target.name == implementer_second.name:
                    second_started.set()
            release = releases.get(target.name)
            if release is not None and not release.wait(timeout=3.0):
                return False
            target.unlink()
            return True

        daemon.dispatch = fake_dispatch
        result = {}

        def run_pass():
            result["value"] = daemon.process_backlog(dry_run=False)

        worker = threading.Thread(target=run_pass)
        worker.start()
        first_wave = three_started.wait(timeout=2.0)
        second_held_initially = not second_started.is_set()
        releases[architect.name].set()
        releases[red_team.name].set()
        second_held_by_implementer = not second_started.wait(timeout=0.2)
        releases[implementer_first.name].set()
        second_released = second_started.wait(timeout=2.0)
        worker.join(timeout=4.0)
        if worker.is_alive():
            for release in releases.values():
                release.set()
            worker.join(timeout=1.0)

        first_wave_cwds = {
            cwd for name, cwd in launches if name in first_names}
        names = [name for name, _cwd in launches]
        passed = (
            first_wave and second_held_initially
            and second_held_by_implementer and second_released
            and not worker.is_alive() and result.get("value") is True
            and len(first_wave_cwds) == 3
            and names.count(architect.name) == 1
            and names.count(implementer_first.name) == 1
            and names.count(red_team.name) == 1
            and names.count(implementer_second.name) == 1
            and names.index(implementer_first.name)
            < names.index(implementer_second.name)
            and not any(path.exists() for path in (
                architect, implementer_first, red_team,
                implementer_second)))
        print("three role lanes overlap and each lane stays sequential="
              + str(passed))
        return passed


def finite_action_case(source, arguments, dry_run):
    """Run one finite action and return whether rendezvous stayed absent."""
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 1
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1
        first = write_pending(mailbox, "0001-to-fable.md")
        second = write_pending(mailbox, "0002-to-fable.md")
        before = tree_snapshot(root)
        launches = []
        real_time = daemon.time

        def forbidden_sleep(seconds):
            raise AssertionError("finite action slept " + repr(seconds))

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del fix_only, kwargs
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


def arm_once_recovers_prelaunch_request(source=None):
    """A finite pass sees work retained before a provider process started."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        held = mailbox / "prelaunch"
        held.mkdir()
        request = write_pending(held, "0001-to-fable.md", "real body\n")
        launches = []

        def fake_dispatch(path, dry_run, **kwargs):
            del dry_run, kwargs
            launches.append(pathlib.Path(path).name)
            pathlib.Path(path).unlink()
            return True

        daemon.dispatch = fake_dispatch
        rc, output, _, error = call_main(daemon, ["--once"])
        passed = (
            error is None and rc == 0
            and launches == [request.name]
            and not request.exists()
            and not (mailbox / request.name).exists()
            and "requeued pre-launch message" in output)
        print("once recovers pre-launch request=" + str(passed))
        return passed


def arm_watch_exits_on_token_exhaustion(source=None):
    """A finite watch reports exhaustion once and keeps progress unfinished."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        calls = []

        def exhausted_pass(dry_run, fix_only=False, skip_redteam=False):
            del dry_run, fix_only, skip_redteam
            calls.append("pass")
            raise daemon.RoleTokenExhaustionError(
                agent="opus",
                request_path=str(mailbox / "failed" / "0001-to-opus.md"))

        daemon.process_backlog = exhausted_pass
        rc, output, _, error = call_main(
            daemon, ["--watch", "--cycle", "1"])
        state = daemon.read_ticket_cycle_state()
        finite = state["finite_watch"]
        retry_lock = daemon.acquire_dispatch_lock(mode="once")
        lock_released = retry_lock is not None
        if retry_lock is not None:
            daemon.release_dispatch_lock(lock_file=retry_lock)
        passed = (
            error is None and rc == 1 and calls == ["pass"]
            and output.count("Error: Implementer is out of tokens") == 1
            and "safe to Ctrl-C" not in output
            and finite is not None and finite["status"] == "active"
            and finite["completed"] == 0 and lock_released)
        print("watch token exhaustion exits once=" + str(passed))
        return passed


def arm_once_exits_on_token_exhaustion(source=None):
    """A one-pass run reports exhaustion once and releases its lock."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        calls = []

        def exhausted_pass(dry_run, fix_only=False, skip_redteam=False):
            del dry_run, fix_only, skip_redteam
            calls.append("pass")
            raise daemon.RoleTokenExhaustionError(
                agent="fable",
                request_path=str(mailbox / "failed" / "0001-to-fable.md"))

        daemon.process_backlog = exhausted_pass
        rc, output, _, error = call_main(daemon, ["--once"])
        retry_lock = daemon.acquire_dispatch_lock(mode="once")
        lock_released = retry_lock is not None
        if retry_lock is not None:
            daemon.release_dispatch_lock(lock_file=retry_lock)
        passed = (
            error is None and rc == 1 and calls == ["pass"]
            and output.count("Error: Architect is out of tokens") == 1
            and lock_released)
        print("once token exhaustion exits once=" + str(passed))
        return passed


def arm_cycle_argument_contract(source=None):
    """Cycle is watch-only and works in both supported topologies."""
    with scratch_daemon(source=source) as (daemon, _, _):
        nonwatch_rc, nonwatch_output, _, nonwatch_error = call_main(
            daemon, ["--cycle", "1"])
        negative_rc, _, negative_stderr, negative_error = call_main(
            daemon, ["--watch", "--cycle", "-1"])
        invalid_rc, _, invalid_stderr, invalid_error = call_main(
            daemon, ["--watch", "--cycle", "two"])
        valid_rc, valid_output, _, valid_error = call_main(
            daemon, ["--watch", "--fix-only", "yes", "--cycle", "0"])
        two_role_rc, two_role_output, _, two_role_error = call_main(
            daemon, ["--watch", "--skip-redteam", "--cycle", "0"])
        retry_lock = daemon.acquire_dispatch_lock(mode="once")
        released = retry_lock is not None
        if retry_lock is not None:
            daemon.release_dispatch_lock(lock_file=retry_lock)
        passed = (
            nonwatch_error is None and nonwatch_rc == 1
            and "--cycle is valid only with --watch" in nonwatch_output
            and negative_rc is None
            and isinstance(negative_error, SystemExit)
            and negative_error.code == 2
            and "cycle count must be a nonnegative integer" in negative_stderr
            and invalid_rc is None
            and isinstance(invalid_error, SystemExit)
            and invalid_error.code == 2
            and "cycle count must be a nonnegative integer" in invalid_stderr
            and valid_error is None and valid_rc == 0 and released
            and not daemon.fix_only_watch_is_active()
            and "fix-only watch active" in valid_output
            and "cycle 0: wait until no role message is waiting or running "
            "and ai/notes/backlog.md has no '- OPEN' item, then exit"
            in valid_output
            and "cycle work complete after 0 cycles" in valid_output
            and two_role_error is None and two_role_rc == 0
            and "two-role watch: Red Team and the entire Sol route are "
            "disabled" in two_role_output
            and "implementation drain complete after 0 cycles"
            in two_role_output)
        print("cycle parser/action contract=" + str(passed))
        return passed


def arm_omitted_cycle_remains_unbounded(source=None):
    """Omitting the option preserves the ordinary indefinite watch."""
    with scratch_daemon(source=source) as (daemon, _, _):
        clock = ControlledClock()
        clock.raise_on_poll = True
        install_clock_and_stamp(daemon, clock=clock, stop_when=lambda: False)
        rc, output, _, error = call_main(daemon, ["--watch"])
        retry_lock = daemon.acquire_dispatch_lock(mode="once")
        released = retry_lock is not None
        if retry_lock is not None:
            daemon.release_dispatch_lock(lock_file=retry_lock)
        passed = (
            rc is None and isinstance(error, KeyboardInterrupt) and released
            and "every enabled role is idle; safe to Ctrl-C for this "
            "20s poll; "
            "no messages waiting." in output
            and "cycle 0:" not in output
            and "cycle limit:" not in output
            and "watcher stopped" not in output)
        print("omitted cycle remains unbounded=" + str(passed))
        return passed


def arm_positive_cycle_limit_preserves_queue(source=None):
    """Child completions and safe-stop windows never complete a ticket."""
    daemon = load_daemon(source=source)
    daemon.RENDEZVOUS_DISPATCH_INTERVAL = 8
    daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
    controller = daemon.SafeKillRendezvous(
        ticket_cycle_limit=2, ticket_cycle_topology="normal")
    every_child_admitted = True
    for _ in range(5):
        permit = controller.begin_attempt()
        if permit is None:
            every_child_admitted = False
            break
        controller.turn_started(permit=permit)
        controller.turn_finished(permit=permit)
        controller.finish_attempt(permit=permit)
    cadence_did_not_count = (
        controller.completed_ticket_cycles() == 0
        and not controller.ticket_cycle_limit_reached())
    controller.ticket_cycle_returned()
    first_return_not_enough = (
        controller.completed_ticket_cycles() == 1
        and not controller.ticket_cycle_limit_reached())
    controller.ticket_cycle_returned()
    passed = (
        every_child_admitted and cadence_did_not_count
        and first_return_not_enough
        and controller.completed_ticket_cycles() == 2
        and controller.ticket_cycle_limit_reached())
    print("safe-stop cadence is not ticket-cycle completion=" + str(passed))
    return passed


def arm_positive_cycle_waits_for_exact_boundary(source=None):
    """A positive limit closes admission only at its exact return count."""
    daemon = load_daemon(source=source)
    daemon.RENDEZVOUS_DISPATCH_INTERVAL = 8
    daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
    controller = daemon.SafeKillRendezvous(
        ticket_cycle_limit=2, ticket_cycle_topology="normal")
    controller.ticket_cycle_returned()
    first = controller.begin_attempt()
    admitted_after_one = first is not None
    if first is not None:
        controller.finish_attempt(permit=first)
    controller.ticket_cycle_returned()
    refused_after_two = controller.begin_attempt() is None
    passed = (
        admitted_after_one and refused_after_two
        and controller.completed_ticket_cycles() == 2)
    print("positive ticket-cycle boundary is exact=" + str(passed))
    return passed


def arm_finite_architect_admission_blocks_second_user_request(source=None):
    """One public request occupies the finite slot before Opus starts."""
    live_checks = False
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        anchor = "public-admission"
        write_indexed_open_tickets(
            backlog=pathlib.Path(daemon.BACKLOG_LEDGER), anchors=(anchor,))
        first = write_pending(
            mailbox, "0001-to-fable.md",
            daemon.architect_user_request_payload(
                "Fix the public admission witness."))
        second = write_pending(
            mailbox, "0002-to-fable.md",
            daemon.architect_user_request_payload(
                "Fix a later ticket only after the first cycle."))
        second_bytes = second.read_bytes()
        for cwd in set(daemon.AGENT_CWD.values()):
            pathlib.Path(cwd).mkdir(parents=True, exist_ok=True)
        daemon.capture_persistent_role_state = (
            lambda agent: {"agent": agent, "base": BASE_COMMIT}
            if agent == "fable" else object())
        daemon.recheck_persistent_role_state = lambda proof: None
        daemon.worktree_head = lambda worktree: BASE_COMMIT
        daemon._validate_current_protected_primary_state = (
            lambda primary_worktree: None)
        launches = []
        outbound = mailbox / "0003-to-opus.md"

        def fake_popen(command, stdout, stderr, cwd, env,
                   start_new_session=False):
            del command, stderr, cwd
            launches.append(first.name)
            token = env.get("MAILBOX_ARCHITECT_ADMISSION")
            if token is None:
                stdout.write("Architect launched without an admission.\n")
                stdout.flush()
                return ImmediateProcess()
            outbound.write_text(
                ticket_flow_payload(
                    cycle_id=anchor + "@" + BASE_COMMIT,
                    text=(daemon.MAILBOX_ADMISSION_HEADER + token
                          + "\n- **Directive:** [ai/notes/spec.md, section "
                          "Implementation directive]")),
                encoding="utf-8", newline="")
            stdout.write("Architect published one exact handoff.\n")
            stdout.flush()
            return ImmediateProcess()

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=fake_popen)
        controller = daemon.SafeKillRendezvous(
            ticket_cycle_limit=1, ticket_cycle_topology="normal")
        daemon._ACTIVE_WATCH_RENDEZVOUS = controller
        try:
            restored = daemon.prepare_finite_watch_progress(
                limit=1, topology="normal")
            controller.restore_completed_ticket_cycles(count=restored)
            consumed = daemon.drain_lane(
                paths=[str(first), str(second)], dry_run=False)
            after_first = daemon.read_ticket_cycle_state()
            charged_first = daemon.finite_cycle_capacity_used(
                state=after_first)

            daemon._ACTIVE_WATCH_RENDEZVOUS = None
            replacement = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")
            daemon._ACTIVE_WATCH_RENDEZVOUS = replacement
            restored_again = daemon.prepare_finite_watch_progress(
                limit=1, topology="normal")
            replacement.restore_completed_ticket_cycles(
                count=restored_again)
            launches_before_restart = list(launches)
            restarted = daemon.drain_lane(
                paths=[str(second)], dry_run=False)
            deferred, new_cycle = (
                daemon.reserve_implementer_ticket_before_claim(
                    path=str(outbound), skip_redteam=False))
            after_restart = daemon.read_ticket_cycle_state()
            charged_restart = daemon.finite_cycle_capacity_used(
                state=after_restart)
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        done = mailbox / "done" / first.name
        cycle = anchor + "@" + BASE_COMMIT
        live_checks = (
            consumed and restarted and launches == [first.name]
            and launches == launches_before_restart
            and done.is_file() and not first.exists()
            and second.is_file() and second.read_bytes() == second_bytes
            and not (mailbox / "done" / second.name).exists()
            and not (mailbox / "failed" / second.name).exists()
            and after_first["architect_admissions"] == {}
            and set(after_first["active"]) == {cycle}
            and charged_first == 1 and charged_restart == 1
            and restored_again == 0 and deferred is None
            and new_cycle is None
            and set(after_restart["active"]) == {cycle})

    recovery_checks = False
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        anchor = "public-recovery"
        write_indexed_open_tickets(
            backlog=pathlib.Path(daemon.BACKLOG_LEDGER), anchors=(anchor,))
        request = write_pending(
            mailbox, "0001-to-fable.md",
            daemon.architect_user_request_payload(
                "Recover only the exact public request."))
        controller = daemon.SafeKillRendezvous(
            ticket_cycle_limit=1, ticket_cycle_topology="normal")
        daemon._ACTIVE_WATCH_RENDEZVOUS = controller
        try:
            restored = daemon.prepare_finite_watch_progress(
                limit=1, topology="normal")
            controller.restore_completed_ticket_cycles(count=restored)
            deferred, token = daemon.reserve_architect_ticket_before_claim(
                path=str(request), skip_redteam=False)
            wrong_token = token[:-1] + ("0" if token[-1] != "0" else "1")
            wrong = write_pending(
                mailbox, "0002-to-opus.md",
                ticket_flow_payload(
                    cycle_id=anchor + "@" + BASE_COMMIT,
                    text=(daemon.MAILBOX_ADMISSION_HEADER + wrong_token
                          + "\nwrong binding")))
            wrong_refused = False
            try:
                daemon.register_ticket_cycle_message(
                    agent="opus", message=wrong.read_text(encoding="utf-8"),
                    implementer_request_name=wrong.name)
            except daemon.TicketCycleStateError:
                wrong_refused = True
            before_restart = daemon.read_ticket_cycle_state()
            wrong.unlink()
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
            replacement = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")
            daemon._ACTIVE_WATCH_RENDEZVOUS = replacement
            restored_again = daemon.prepare_finite_watch_progress(
                limit=1, topology="normal")
            replacement.restore_completed_ticket_cycles(
                count=restored_again)
            later = write_pending(
                mailbox, "0003-to-fable.md",
                daemon.architect_user_request_payload(
                    "This later request must stay outside the saved slot."))
            later_bytes = later.read_bytes()
            later_deferred, later_token = (
                daemon.reserve_architect_ticket_before_claim(
                    path=str(later), skip_redteam=False))
            correct = write_pending(
                mailbox, "0004-to-opus.md",
                ticket_flow_payload(
                    cycle_id=anchor + "@" + BASE_COMMIT,
                    text=(daemon.MAILBOX_ADMISSION_HEADER + token
                          + "\nexact crash-recovery binding")))
            correct_deferred, _ = (
                daemon.reserve_implementer_ticket_before_claim(
                    path=str(correct), skip_redteam=False))
            after_conversion = daemon.read_ticket_cycle_state()
            charged = daemon.finite_cycle_capacity_used(
                state=after_conversion)
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        recovery_checks = (
            deferred is None and token is not None and wrong_refused
            and set(before_restart["architect_admissions"]) == {request.name}
            and before_restart["active"] == {}
            and later_deferred is not None and later_token is None
            and later.is_file() and later.read_bytes() == later_bytes
            and correct_deferred is None
            and after_conversion["architect_admissions"] == {}
            and set(after_conversion["active"])
            == {anchor + "@" + BASE_COMMIT}
            and charged == 1)

    passed = live_checks and recovery_checks
    print("finite Architect admission blocks second public request="
          + str(passed))
    return passed


def arm_public_architect_non_ticket_outcomes_release_admission(source=None):
    """One Sol control or explicit receipt releases a provisional slot."""
    sol_pipeline = False
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        anchor = "after-sol-control"
        write_indexed_open_tickets(
            backlog=pathlib.Path(daemon.BACKLOG_LEDGER), anchors=(anchor,))
        control = write_pending(
            mailbox, "0001-to-fable.md",
            daemon.architect_user_request_payload(
                "Ask Sol to inspect this bounded concern."))
        first = write_pending(
            mailbox, "0003-to-fable.md",
            daemon.architect_user_request_payload(
                "Implement the first real ticket."))
        second = write_pending(
            mailbox, "0004-to-fable.md",
            daemon.architect_user_request_payload(
                "Wait until another finite watch."))
        second_bytes = second.read_bytes()
        for cwd in set(daemon.AGENT_CWD.values()):
            pathlib.Path(cwd).mkdir(parents=True, exist_ok=True)
        daemon.capture_persistent_role_state = (
            lambda agent: {"agent": agent, "base": BASE_COMMIT}
            if agent == "fable" else object())
        daemon.recheck_persistent_role_state = lambda proof: None
        daemon.worktree_head = lambda worktree: BASE_COMMIT
        daemon._validate_current_protected_primary_state = (
            lambda primary_worktree: None)
        launches = []
        issued_tokens = {}

        def fake_popen(command, stdout, stderr, cwd, env,
                   start_new_session=False):
            del command, stderr, cwd
            token = env["MAILBOX_ARCHITECT_ADMISSION"]
            request_name = token.split("@", 1)[0]
            launches.append(request_name)
            issued_tokens[request_name] = token
            if request_name == control.name:
                write_pending(
                    mailbox, "0002-to-sol.md",
                    daemon.sol_ticket_payload(
                        ticket_kind="discovery",
                        text=(daemon.MAILBOX_ADMISSION_HEADER + token
                              + "\nInspect only this public concern."),
                        discovery_severity="medium",
                        discovery_scope="bounded"))
            else:
                write_pending(
                    mailbox, "0005-to-opus.md",
                    ticket_flow_payload(
                        cycle_id=anchor + "@" + BASE_COMMIT,
                        text=(daemon.MAILBOX_ADMISSION_HEADER + token
                              + "\n- **Directive:** [ai/notes/spec.md, "
                              "section Implementation directive]")))
            stdout.write("Architect published one classified outcome.\n")
            stdout.flush()
            return ImmediateProcess()

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=fake_popen)
        controller = daemon.SafeKillRendezvous(
            ticket_cycle_limit=1, ticket_cycle_topology="normal")
        daemon._ACTIVE_WATCH_RENDEZVOUS = controller
        try:
            restored = daemon.prepare_finite_watch_progress(
                limit=1, topology="normal")
            controller.restore_completed_ticket_cycles(count=restored)
            control_consumed = daemon.drain_lane(
                paths=[str(control)], dry_run=False)
            after_control = daemon.read_ticket_cycle_state()
            control_charge = daemon.finite_cycle_capacity_used(
                state=after_control)
            ticket_pass = daemon.drain_lane(
                paths=[str(first), str(second)], dry_run=False)
            after_ticket = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        sol_output = mailbox / "0002-to-sol.md"
        sol_pipeline = (
            control_consumed and ticket_pass
            and launches == [control.name, first.name]
            and after_control["architect_admissions"] == {}
            and after_control["active"] == {} and control_charge == 0
            and (mailbox / "done" / control.name).is_file()
            and sol_output.is_file()
            and daemon.public_architect_sol_outcome_problem(
                message=sol_output.read_text(encoding="utf-8"),
                expected_token=issued_tokens[control.name]) is None
            and set(after_ticket["active"])
            == {anchor + "@" + BASE_COMMIT}
            and after_ticket["architect_admissions"] == {}
            and second.is_file() and second.read_bytes() == second_bytes
            and not (mailbox / "done" / second.name).exists())

    terminal_release = False
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        request = write_pending(
            mailbox, "0001-to-fable.md",
            daemon.architect_user_request_payload(
                "Explain why no tracked change is required."))
        for cwd in set(daemon.AGENT_CWD.values()):
            pathlib.Path(cwd).mkdir(parents=True, exist_ok=True)
        daemon.capture_persistent_role_state = (
            lambda agent: {"agent": agent, "base": BASE_COMMIT})
        daemon.recheck_persistent_role_state = lambda proof: None
        daemon.worktree_head = lambda worktree: BASE_COMMIT
        daemon._validate_current_protected_primary_state = (
            lambda primary_worktree: None)
        receipt = mailbox / "0002-to-user.md"

        def fake_terminal(command, stdout, stderr, cwd, env,
                      start_new_session=False):
            del command, stderr, cwd
            token = env["MAILBOX_ARCHITECT_ADMISSION"]
            receipt.write_text(
                "MAILBOX-RETURN: architect-no-ticket\n"
                "MAILBOX-ADMISSION: " + token + "\n"
                "MAILBOX-DECISION: NO TICKET\n\n"
                "No tracked change is required.\n",
                encoding="utf-8", newline="")
            stdout.write("Architect returned an explicit no-ticket result.\n")
            stdout.flush()
            return ImmediateProcess()

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=fake_terminal)
        controller = daemon.SafeKillRendezvous(
            ticket_cycle_limit=1, ticket_cycle_topology="normal")
        daemon._ACTIVE_WATCH_RENDEZVOUS = controller
        try:
            restored = daemon.prepare_finite_watch_progress(
                limit=1, topology="normal")
            controller.restore_completed_ticket_cycles(count=restored)
            consumed = daemon.drain_lane(
                paths=[str(request)], dry_run=False)
            state = daemon.read_ticket_cycle_state()
            charged = daemon.finite_cycle_capacity_used(state=state)
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        terminal_release = (
            consumed and state["architect_admissions"] == {}
            and state["active"] == {} and charged == 0
            and receipt.is_file()
            and (mailbox / "done" / request.name).is_file())

    passed = sol_pipeline and terminal_release
    print("public Architect Sol and no-ticket outcomes release admission="
          + str(passed))
    return passed


def arm_public_architect_ambiguous_outcomes_fail_closed(source=None):
    """Failed turns keep evidence but release their unused finite slot."""
    results = []
    for case in ("child-error", "silent", "malformed", "multiple", "mixed"):
        with scratch_daemon(source=source) as (daemon, _, mailbox):
            request = write_pending(
                mailbox, "0001-to-fable.md",
                daemon.architect_user_request_payload(
                    "Exercise the " + case + " outcome boundary."))
            for cwd in set(daemon.AGENT_CWD.values()):
                pathlib.Path(cwd).mkdir(parents=True, exist_ok=True)
            daemon.capture_persistent_role_state = (
                lambda agent: {"agent": agent, "base": BASE_COMMIT})
            daemon.recheck_persistent_role_state = lambda proof: None
            daemon.worktree_head = lambda worktree: BASE_COMMIT
            daemon._validate_current_protected_primary_state = (
                lambda primary_worktree: None)
            created = []

            def fake_invalid(command, stdout, stderr, cwd, env,
                         start_new_session=False):
                del command, stderr, cwd
                token = env["MAILBOX_ARCHITECT_ADMISSION"]
                receipt_text = (
                    "MAILBOX-RETURN: architect-no-ticket\n"
                    "MAILBOX-ADMISSION: " + token + "\n"
                    "MAILBOX-DECISION: NO TICKET\n")
                if case == "malformed":
                    path = mailbox / "0002-to-user.md"
                    wrong_token = token[:-1] + (
                        "0" if token[-1] != "0" else "1")
                    path.write_text(
                        receipt_text.replace(token, wrong_token),
                        encoding="utf-8", newline="")
                    created.append(path)
                elif case in {"multiple", "mixed"}:
                    first_output = mailbox / "0002-to-user.md"
                    first_output.write_text(
                        receipt_text, encoding="utf-8", newline="")
                    created.append(first_output)
                    if case == "multiple":
                        second_output = mailbox / "0003-to-user.md"
                        second_output.write_text(
                            receipt_text, encoding="utf-8", newline="")
                    else:
                        second_output = mailbox / "0003-to-sol.md"
                        second_output.write_text(
                            daemon.sol_ticket_payload(
                                ticket_kind="discovery",
                                text=(daemon.MAILBOX_ADMISSION_HEADER + token
                                      + "\nReview this exact concern."),
                                discovery_severity="medium",
                                discovery_scope="bounded"),
                            encoding="utf-8", newline="")
                    created.append(second_output)
                stdout.write("Architect invalid-outcome witness.\n")
                stdout.flush()
                process = ImmediateProcess()
                if case == "child-error":
                    process.returncode = 1
                return process

            daemon.subprocess = AttributeProxy(
                daemon.subprocess, Popen=fake_invalid)
            controller = daemon.SafeKillRendezvous(
                ticket_cycle_limit=1, ticket_cycle_topology="normal")
            daemon._ACTIVE_WATCH_RENDEZVOUS = controller
            try:
                restored = daemon.prepare_finite_watch_progress(
                    limit=1, topology="normal")
                controller.restore_completed_ticket_cycles(count=restored)
                consumed = daemon.drain_lane(
                    paths=[str(request)], dry_run=False)
                state = daemon.read_ticket_cycle_state()
                charged = daemon.finite_cycle_capacity_used(state=state)
                recovered_again = (
                    daemon.recover_failed_public_architect_admissions())
            finally:
                daemon._ACTIVE_WATCH_RENDEZVOUS = None
            results.append(
                not consumed
                and state["architect_admissions"] == {}
                and state["active"] == {} and charged == 0
                and recovered_again == 0
                and (mailbox / "failed" / request.name).is_file()
                and all((mailbox / "failed" / path.name).is_file()
                        for path in created)
                and all(not path.exists() for path in created))
    passed = all(results)
    print("failed public Architect outcomes release finite slot="
          + str(passed))
    return passed


def arm_cycle_one_never_overlaps_next_ticket(source=None):
    """A one-cycle watch reserves its only slot through Red Team review."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        first_anchor = "finite-one-first"
        second_anchor = "finite-one-second"
        backlog = pathlib.Path(daemon.BACKLOG_LEDGER)
        write_indexed_open_tickets(
            backlog=backlog, anchors=(first_anchor, second_anchor))
        first_cycle = first_anchor + "@" + BASE_COMMIT
        second_cycle = second_anchor + "@" + BASE_COMMIT
        first = write_pending(
            mailbox, "0001-to-opus.md",
            ticket_flow_payload(cycle_id=first_cycle))
        second = write_pending(
            mailbox, "0002-to-opus.md",
            ticket_flow_payload(cycle_id=second_cycle))
        second_bytes = second.read_bytes()
        launches = []

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
            target = pathlib.Path(path)
            launches.append(target.name)
            target.unlink()
            return True

        daemon.dispatch = fake_dispatch
        # This arm tests finite admission, not Git landing construction.
        # Supply the exact candidate/landing authority that the production
        # daemon would already have journaled before this state transition.
        daemon.require_architect_landing_locked = (
            lambda cycle_id, landing_commit, ticket_state: "3" * 40)
        controller = daemon.SafeKillRendezvous(
            ticket_cycle_limit=1, ticket_cycle_topology="normal")
        daemon._ACTIVE_WATCH_RENDEZVOUS = controller
        try:
            restored = daemon.prepare_finite_watch_progress(
                limit=1, topology="normal")
            controller.restore_completed_ticket_cycles(count=restored)
            first_pass = daemon.process_backlog(
                dry_run=False, fix_only=False)
            after_implementation = daemon.read_ticket_cycle_state()

            # The accepted commit does not complete normal three-role mode.
            # The slot remains occupied while the Red Team request waits and
            # after that request has been registered for review.
            completed_at_commit = daemon.record_architect_commit(
                cycle_id=first_cycle,
                accepted_commit=ACCEPTED_COMMIT,
                mode="normal")
            closure = daemon.sol_ticket_payload(
                ticket_kind="closure",
                text="Review only this accepted ticket.",
                review_cycle=first_cycle,
                review_commit=ACCEPTED_COMMIT)
            daemon.register_ticket_cycle_message(
                agent="sol", message=closure)
            second_pass = daemon.process_backlog(
                dry_run=False, fix_only=False)
            awaiting_review = daemon.read_ticket_cycle_state()
            charged_slots = daemon.finite_cycle_capacity_used(
                state=awaiting_review)
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None

        passed = (
            first_pass is True and second_pass is True
            and launches == [first.name]
            and not first.exists() and second.is_file()
            and second.read_bytes() == second_bytes
            and set(after_implementation["active"]) == {first_cycle}
            and after_implementation["active"][first_cycle]["phase"]
            == "implementation"
            and second_cycle not in after_implementation["active"]
            and completed_at_commit == 0
            and awaiting_review["active"][first_cycle]["phase"]
            == "awaiting-redteam"
            and second_cycle not in awaiting_review["active"]
            and controller.completed_ticket_cycles() == 0
            and charged_slots == 1)
        print("cycle one never starts a second ticket before review="
              + str(passed))
        return passed


def arm_two_role_limit_is_restart_safe(source=None):
    """A two-role finite watch keeps its reservations across a restart."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        anchors = tuple(
            "finite-primary-" + str(index) for index in range(1, 7))
        write_indexed_open_tickets(
            backlog=pathlib.Path(daemon.BACKLOG_LEDGER), anchors=anchors)

        paths = []
        for sequence, anchor in enumerate(anchors, start=1):
            paths.append(write_pending(
                mailbox,
                str(sequence).zfill(4) + "-to-opus.md",
                ticket_flow_payload(
                    cycle_id=anchor + "@" + BASE_COMMIT,
                    mode="two-role")))

        launches = []

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
            target = pathlib.Path(path)
            launches.append(target.name)
            target.unlink()
            return True

        daemon.dispatch = fake_dispatch
        controller = daemon.SafeKillRendezvous(
            ticket_cycle_limit=3, ticket_cycle_topology="two-role")
        daemon._ACTIVE_WATCH_RENDEZVOUS = controller
        try:
            restored = daemon.prepare_finite_watch_progress(
                limit=3, topology="two-role")
            controller.restore_completed_ticket_cycles(count=restored)
            first_pass = daemon.process_backlog(
                dry_run=False, fix_only=False, skip_redteam=True)
            after_first_pass = daemon.read_ticket_cycle_state()
            waiting_before_restart = {
                path.name: path.read_bytes() for path in paths if path.exists()}

            # Simulate a process replacement. Durable active reservations,
            # not a lane-local counter, must still occupy all three slots.
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
            replacement = daemon.SafeKillRendezvous(
                ticket_cycle_limit=3, ticket_cycle_topology="two-role")
            daemon._ACTIVE_WATCH_RENDEZVOUS = replacement
            restored_again = daemon.prepare_finite_watch_progress(
                limit=3, topology="two-role")
            replacement.restore_completed_ticket_cycles(
                count=restored_again)
            launches_before_restart_pass = list(launches)
            restart_pass = daemon.process_backlog(
                dry_run=False, fix_only=False, skip_redteam=True)
            after_restart = daemon.read_ticket_cycle_state()
            charged_after_restart = daemon.finite_cycle_capacity_used(
                state=after_restart, skip_redteam=True)
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None

        waiting_after_restart = {
            path.name: path.read_bytes() for path in paths if path.exists()}
        passed = (
            first_pass is True and restart_pass is True
            and len(launches) == 3
            and launches == launches_before_restart_pass
            and all(record["route"] == "primary"
                    for record in after_first_pass["active"].values())
            and len(after_first_pass["active"]) == 3
            and after_first_pass["finite_watch"] == {
                "limit": 3, "completed": 0, "status": "active",
                "topology": "two-role"}
            and controller.completed_ticket_cycles() == 0
            and restored_again == 0
            and replacement.completed_ticket_cycles() == 0
            and len(after_restart["active"]) == 3
            and charged_after_restart == 3
            and len(waiting_before_restart) == 3
            and waiting_after_restart == waiting_before_restart)
        print("two-role cycle three is total and restart-safe="
              + str(passed))
        return passed


def arm_reached_limit_raw_admin_fails_once(source=None):
    """A corrupt raw admin bypasses the ticket limit and becomes debt."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.require_no_ordinary_landing_transition = (
            lambda current_dispatch_path: None)
        state = daemon.read_ticket_cycle_state()
        state["finite_watch"] = {
            "limit": 1, "completed": 1, "status": "active",
            "topology": "normal"}
        daemon.write_ticket_cycle_state(state=state)
        message = mailbox / "0001-to-fable.md"
        message.write_bytes(
            daemon.MAILBOX_ADMIN_HEADER.encode("ascii")
            + b"permanent-notes\n\ninvalid\xfftail")
        launches = []
        real_subprocess = daemon.subprocess

        class NoChildProxy:
            def __getattr__(self, name):
                if name == "Popen":
                    def forbidden(*args, **kwargs):
                        launches.append((args, kwargs))
                        raise AssertionError(
                            "corrupt admin launched a child")
                    return forbidden
                return getattr(real_subprocess, name)

        daemon.subprocess = NoChildProxy()
        try:
            rc, output, stderr, error = call_main(
                daemon, ["--watch", "--cycle", "1"])
        finally:
            daemon.subprocess = real_subprocess
        failed = mailbox / "failed" / message.name
        passed = (
            error is None and rc == 1 and stderr == "" and not launches
            and not message.exists() and failed.is_file()
            and failed.read_bytes().endswith(b"invalid\xfftail")
            and ("permanent-note user action required: failed/"
                 + message.name) in output
            and "mailbox empty" not in output)
        print("reached-limit raw admin fails once=" + str(passed))
        return passed


def arm_zero_cycle_drains_ledger_and_preserves_cadence(source=None):
    """Zero drains recorded work; manual safe stops do not add cycles."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 2
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        backlog = pathlib.Path(daemon.BACKLOG_LEDGER)
        write_indexed_open_ticket(backlog=backlog)
        first = write_pending(mailbox, "0001-to-fable.md", "first body\n")
        queued = []
        launches = []
        clock = ControlledClock()

        def add_second_after_ordinary_poll(fake_clock):
            del fake_clock
            if not queued:
                queued.append(write_pending(
                    mailbox, "0002-to-fable.md", "second body\n"))

        clock.on_poll = add_second_after_ordinary_poll

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
            with simulated_child_turn(daemon):
                target = pathlib.Path(path)
                launches.append(target.name)
                target.unlink()
                if target.name == "0002-to-fable.md":
                    backlog.write_text(
                        "- CLOSED cycle witness\n", encoding="utf-8")
                return True

        daemon.dispatch = fake_dispatch
        install_clock_and_stamp(daemon, clock=clock, stop_when=lambda: False)
        rc, output, _, error = call_main(
            daemon, ["--watch", "--cycle", "0"])
        passed = (
            error is None and rc == 0
            and queued and launches == [first.name, queued[0].name]
            and not first.exists() and not queued[0].exists()
            and daemon.backlog_ledger_count() == 0
            and countdown_lines(output) == []
            and output.count(
                "every enabled role is idle; safe to Ctrl-C for this "
                "20s poll; "
                "no messages waiting.") == 1
            and output.count("safe interval ended; not safe to stop.") == 1
            and "cycle work complete after 0 cycles; no role message is "
            "waiting or running; ai/notes/backlog.md has no '- OPEN' item; "
            "watcher stopped." in output)
        print("zero cycle drains ledger and preserves cadence=" + str(passed))
        return passed


def arm_zero_cycle_waits_for_queue(source=None):
    """Zero does not call an idle rendezvous complete while a file waits."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 1
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        first = write_pending(mailbox, "0001-to-fable.md", "first body\n")
        second = write_pending(mailbox, "0002-to-fable.md", "second body\n")
        launches = []
        clock = ControlledClock()

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
            with simulated_child_turn(daemon):
                target = pathlib.Path(path)
                launches.append(target.name)
                target.unlink()
                return True

        daemon.dispatch = fake_dispatch
        install_clock_and_stamp(daemon, clock=clock, stop_when=lambda: False)
        rc, output, _, error = call_main(
            daemon, ["--watch", "--cycle", "0"])
        passed = (
            error is None and rc == 0
            and launches == [first.name, second.name]
            and not first.exists() and not second.exists()
            and countdown_lines(output) == expected_countdown(waiting=1)
            and "cycle work complete after 0 cycles; no role message is "
            "waiting or running; ai/notes/backlog.md has no '- OPEN' item; "
            "watcher stopped." in output)
        print("zero cycle waits for queue=" + str(passed))
        return passed


def arm_zero_cycle_send_before_cutoff_is_seen(source=None):
    """A daemon send before the completion cutoff prevents early exit."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 999
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        clock = ControlledClock()
        launches = []
        injected = []
        real_process_backlog = daemon.process_backlog

        def inject_after_empty_pass(dry_run, fix_only=False, **kwargs):
            outcome = real_process_backlog(
                dry_run=dry_run, fix_only=fix_only, **kwargs)
            if not injected and outcome is None:
                injected.append(daemon.send(
                    agent="opus", text="arrived before cutoff",
                    dry_run=False))
            return outcome

        def fake_dispatch(path, dry_run, fix_only=False, **kwargs):
            del dry_run, fix_only, kwargs
            with simulated_child_turn(daemon):
                target = pathlib.Path(path)
                launches.append(target.name)
                target.unlink()
                return True

        daemon.process_backlog = inject_after_empty_pass
        daemon.dispatch = fake_dispatch
        install_clock_and_stamp(
            daemon, clock=clock, stop_when=lambda: False)
        rc, output, _, error = call_main(
            daemon, ["--watch", "--cycle", "0"])
        passed = (
            error is None and rc == 0 and injected == [True]
            and launches == ["0001-to-opus.md"]
            and daemon.pending_messages() == []
            and not (mailbox / "0001-to-opus.md").exists()
            and "cycle work complete after 0 cycles; no role message is "
            "waiting or running; ai/notes/backlog.md has no '- OPEN' item; "
            "watcher stopped." in output)
        print("send before cycle-zero cutoff is observed=" + str(passed))
        return passed


def arm_zero_cycle_cutoff_serializes_sender(source=None):
    """A sender behind the cutoff publishes only after watch-lock release."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        sender_at_lock = threading.Event()
        sender_acquired_lock = threading.Event()
        sender_done = threading.Event()
        watch_released = threading.Event()
        sender_threads = []
        sender_result = {}
        observations = {}

        real_flock = daemon.fcntl.flock

        def observed_flock(descriptor, operation):
            if (threading.current_thread().name == "cycle-cutoff-sender"
                    and operation == daemon.fcntl.LOCK_EX):
                sender_at_lock.set()
            return real_flock(descriptor, operation)

        daemon.fcntl = AttributeProxy(daemon.fcntl, flock=observed_flock)
        real_next_seq = daemon.next_seq

        def observed_next_seq():
            if threading.current_thread().name == "cycle-cutoff-sender":
                sender_acquired_lock.set()
            return real_next_seq()

        daemon.next_seq = observed_next_seq

        def observed_warning():
            observations["live_at_warning"] = (
                daemon.dispatch_lock_is_live_watch(mailbox=str(mailbox)))
            daemon._production_warn_if_mailbox_unwatched()

        daemon.warn_if_mailbox_unwatched = observed_warning

        def run_sender():
            try:
                sender_result["queued"] = daemon.send(
                    agent="opus", text="arrived after cutoff",
                    dry_run=False)
            finally:
                sender_done.set()

        real_report = daemon.report_cycle_work_complete

        def report_with_waiting_sender(completed_cycles, skip_redteam=False):
            sender = threading.Thread(
                target=run_sender, name="cycle-cutoff-sender", daemon=True)
            sender_threads.append(sender)
            sender.start()
            observations["sender_reached_lock"] = sender_at_lock.wait(
                timeout=2.0)
            observations["sender_blocked"] = (
                not sender_acquired_lock.wait(timeout=0.1)
                and sender.is_alive() and daemon.pending_messages() == [])
            observations["watch_live_before_exit"] = (
                daemon.dispatch_lock_is_live_watch(mailbox=str(mailbox)))
            real_report(completed_cycles=completed_cycles,
                        skip_redteam=skip_redteam)

        daemon.report_cycle_work_complete = report_with_waiting_sender
        real_release_dispatch = daemon.release_dispatch_lock

        def observed_release_dispatch(lock_file):
            real_release_dispatch(lock_file=lock_file)
            watch_released.set()

        daemon.release_dispatch_lock = observed_release_dispatch
        real_release_barrier = daemon.release_cycle_completion_barrier

        def observed_release_barrier(lock_file):
            observations["watch_released_before_barrier"] = (
                watch_released.is_set())
            real_release_barrier(lock_file=lock_file)
            observations["sender_finished_after_barrier"] = (
                sender_done.wait(timeout=2.0))

        daemon.release_cycle_completion_barrier = observed_release_barrier
        rc, output, _, error = call_main(
            daemon, ["--watch", "--cycle", "0"])
        for sender in sender_threads:
            sender.join(timeout=2.0)
        pending = daemon.pending_messages()
        retry_lock = daemon.acquire_dispatch_lock(mode="once")
        released = retry_lock is not None
        if retry_lock is not None:
            daemon.release_dispatch_lock(lock_file=retry_lock)
        passed = (
            error is None and rc == 0 and released
            and observations.get("sender_reached_lock")
            and observations.get("sender_blocked")
            and observations.get("watch_live_before_exit")
            and observations.get("watch_released_before_barrier")
            and observations.get("sender_finished_after_barrier")
            and observations.get("live_at_warning") is False
            and sender_result.get("queued") is True
            and len(pending) == 1
            and pathlib.Path(pending[0]).read_bytes()
            == b"arrived after cutoff\n"
            and "cycle work complete after 0 cycles" in output
            and "warning: no active watch is polling this mailbox" in output)
        print("cycle-zero cutoff serializes sender=" + str(passed))
        return passed


def arm_cycle_ledger_fail_closed(source=None):
    """Missing and nonregular ledgers cannot satisfy the drain-all barrier."""
    results = []
    for case in ("missing", "nonregular"):
        with scratch_daemon(source=source) as (daemon, _, _):
            backlog = pathlib.Path(daemon.BACKLOG_LEDGER)
            backlog.unlink()
            if case == "nonregular":
                backlog.mkdir()
            barrier, error = daemon.acquire_cycle_completion_barrier(
                backlog_outcome=None)
            expected = (
                "cannot stat backlog ledger" if case == "missing"
                else "backlog ledger is not a regular file")
            results.append(barrier is None and expected in str(error))
            if barrier is not None:
                daemon.release_cycle_completion_barrier(lock_file=barrier)
    passed = results == [True, True]
    print("cycle-zero ledger failures stay active=" + str(passed))
    return passed


def arm_cycle_ledger_read_stability(source=None):
    """A ledger changed during the bounded read cannot prove completion."""
    results = []
    for case in ("replacement", "same-inode"):
        with scratch_daemon(source=source) as (daemon, _, _):
            backlog = pathlib.Path(daemon.BACKLOG_LEDGER)
            real_os = daemon.os
            real_read = real_os.read
            changed = []

            def racing_read(descriptor, size):
                chunk = real_read(descriptor, size)
                if chunk == b"" and not changed:
                    changed.append(True)
                    if case == "replacement":
                        replacement = backlog.with_suffix(".replacement")
                        replacement.write_text(
                            "- OPEN arrived during read\n", encoding="utf-8")
                        real_os.replace(replacement, backlog)
                    else:
                        backlog.write_text(
                            "- OPEN arrived same inode\n", encoding="utf-8")
                return chunk

            daemon.os = AttributeProxy(real_os, read=racing_read)
            count, error = daemon.strict_cycle_ledger_count()
            if case == "replacement":
                expected_error = "changed identity while reading"
            else:
                expected_error = "changed while verifying identity"
            results.append(
                changed == [True] and count is None and error is not None
                and expected_error in error
                and backlog.read_text(encoding="utf-8").startswith("- OPEN"))
    passed = results == [True, True]
    print("cycle ledger read stability=" + str(passed))
    return passed


def arm_cycle_ledger_preopen_fifo_is_bounded(source=None):
    """A regular-to-FIFO replacement before open cannot block zero mode."""
    with scratch_daemon(source=source) as (daemon, _, _):
        backlog = pathlib.Path(daemon.BACKLOG_LEDGER)
        real_os = daemon.os
        real_open = real_os.open
        replacements = []
        result = {}

        def replacing_open(path, flags, *args):
            if path == daemon.BACKLOG_LEDGER and not replacements:
                backlog.unlink()
                real_os.mkfifo(backlog)
                replacements.append(flags)
            return real_open(path, flags, *args)

        daemon.os = AttributeProxy(real_os, open=replacing_open)

        def run_reader():
            result["value"] = daemon.strict_cycle_ledger_count()

        reader = threading.Thread(
            target=run_reader, name="cycle-ledger-fifo-reader", daemon=True)
        reader.start()
        reader.join(timeout=0.2)
        bounded = not reader.is_alive()
        if reader.is_alive():
            # Best-effort cleanup for a mutant that omitted O_NONBLOCK. The
            # thread is daemonized, so a platform that cannot pair it here
            # still cannot wedge the witness process.
            try:
                writer = real_os.open(
                    backlog, real_os.O_WRONLY | real_os.O_NONBLOCK)
            except OSError:
                writer = None
            reader.join(timeout=0.2)
            if writer is not None:
                real_os.close(writer)
        count, error = result.get("value", (None, None))
        passed = (
            bounded and len(replacements) == 1
            and bool(replacements[0] & real_os.O_NONBLOCK)
            and count is None and error is not None
            and "changed identity while opening" in error)
        print("cycle ledger pre-open FIFO is bounded=" + str(passed))
        return passed


def replace_exact(source, old, new):
    """Return a single-site source mutant, else None when not armed.

    ``source`` is either one file's text or a mapping of daemon file name
    to text; a mapping requires the anchor to be unique across every file.
    """
    if isinstance(source, dict):
        matches = []
        for file_name in source:
            if old in source[file_name]:
                matches.append(file_name)
        if len(matches) != 1 or source[matches[0]].count(old) != 1:
            return None
        mutant = dict(source)
        mutant[matches[0]] = source[matches[0]].replace(old, new, 1)
        return mutant
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def arm_source_mutations():
    """Kill rendezvous boundary mutants once production exposes anchors."""
    source = daemon_source_files()
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
                "                daemon.report_admitted_status()\n",
                "                pass  # omitted unsafe transition\n"),
            arm_safe_line_expires_before_claim,
        ),
        (
            "production child-start hook omitted",
            lambda text: replace_exact(
                text, "            daemon._rendezvous_turn_started()\n", ""),
            arm_production_dispatch_lifecycle,
        ),
        (
            "production child-finish hook omitted",
            lambda text: replace_exact(
                text,
                "                        daemon._rendezvous_turn_finished()"
                "\n",
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
            "watch token exhaustion is swallowed",
            lambda text: replace_exact(
                text,
                "                except RoleTokenExhaustionError as exc:\n"
                "                    report_role_token_exhaustion(error=exc)\n"
                "                    return 1\n"
                "                except ImplementerAuthorityViolationError:"
                "\n",
                "                except ImplementerAuthorityViolationError:"
                "\n"),
            arm_watch_exits_on_token_exhaustion,
        ),
        (
            "once token exhaustion is swallowed",
            lambda text: replace_exact(
                text,
                "            except RoleTokenExhaustionError as exc:\n"
                "                report_role_token_exhaustion(error=exc)\n"
                "                return 1\n"
                "            except ImplementerAuthorityViolationError:\n",
                "            except ImplementerAuthorityViolationError:\n"),
            arm_once_exits_on_token_exhaustion,
        ),
        (
            "countdown queue count frozen",
            lambda text: replace_exact(
                text,
                "    for seconds_more in range("
                "daemon.SAFE_KILL_COUNTDOWN_SECONDS - 1, -1, -1):\n"
                "        waiting = len(daemon.pending_messages())\n",
                "    for seconds_more in range("
                "daemon.SAFE_KILL_COUNTDOWN_SECONDS - 1, -1, -1):\n"
                "        waiting = 1\n"),
            arm_dispatch_cadence_global_window,
        ),
        (
            "omitted cycle becomes zero mode",
            lambda text: replace_exact(
                text,
                "type=nonnegative_cycle_count, default=None",
                "type=nonnegative_cycle_count, default=0"),
            arm_omitted_cycle_remains_unbounded,
        ),
        (
            "a child turn falsely completes a ticket cycle",
            lambda text: replace_exact(
                text,
                "            self._completed = self._completed + 1\n"
                "            self._arm_if_due_locked()\n",
                "            self._completed = self._completed + 1\n"
                "            self._ticket_cycles_completed += 1\n"
                "            self._arm_if_due_locked()\n"),
            arm_positive_cycle_limit_preserves_queue,
        ),
        (
            "public Architect pre-claim admission omitted",
            lambda text: replace_exact(
                text,
                "                deferred, architect_admission = (\n"
                "                    daemon."
                "reserve_architect_ticket_before_claim(\n"
                "                        path=path, skip_redteam=skip_redteam))\n",
                "                deferred, architect_admission = (None, None)\n"),
            arm_finite_architect_admission_blocks_second_user_request,
        ),
        (
            "public Architect admissions omitted from finite capacity",
            lambda text: replace_exact(
                text,
                "            + len(daemon.active_cycle_records_for_topology(\n"
                "                state=state, skip_redteam=skip_redteam))\n"
                "            + len(daemon.architect_admissions_for_topology(\n"
                "                state=state, skip_redteam=skip_redteam)))\n",
                "            + len(daemon.active_cycle_records_for_topology(\n"
                "                state=state, skip_redteam=skip_redteam)))\n"),
            arm_finite_architect_admission_blocks_second_user_request,
        ),
        (
            "missing ledger is treated as empty",
            lambda text: replace_exact(
                text,
                '        return None, "cannot stat backlog ledger: " + str(exc)',
                "        return 0, None"),
            arm_cycle_ledger_fail_closed,
        ),
        (
            "nonregular ledger is treated as empty",
            lambda text: replace_exact(
                text,
                '        return None, "backlog ledger is not a regular file"',
                "        return 0, None"),
            arm_cycle_ledger_fail_closed,
        ),
        (
            "cycle completion publication lock omitted",
            lambda text: replace_exact(
                text,
                "def acquire_cycle_completion_barrier(backlog_outcome,\n"
                "                                     skip_redteam=False):\n"
                '    """Return a held send barrier only when cycle-zero work is verified done.\n'
                "\n"
                "    Daemon sends serialize publication through ``.sequence.lock``. Holding\n"
                "    that lock from the final queue/ledger scan until the watch lock is released\n"
                "    gives zero mode a real cutoff: a racing send either lands before the scan\n"
                "    and prevents exit, or lands after the watcher is no longer advertised.\n"
                '    """\n'
                "    failed_debt = daemon.architect_notes_failed_debt_error()"
                "\n"
                "    if failed_debt is not None:\n"
                "        return None, failed_debt\n"
                "    if backlog_outcome is False:\n"
                "        return None, None\n"
                "    lock_path = daemon.os.path.join("
                'daemon.MAILBOX, ".sequence.lock")\n'
                "    lock_file = None\n"
                "    try:\n"
                '        lock_file = open(lock_path, "a+", encoding="utf-8")\n'
                "        daemon.fcntl.flock(lock_file.fileno(), "
                "daemon.fcntl.LOCK_EX)\n",
                "def acquire_cycle_completion_barrier(backlog_outcome,\n"
                "                                     skip_redteam=False):\n"
                '    """Return a held send barrier only when cycle-zero work is verified done.\n'
                "\n"
                "    Daemon sends serialize publication through ``.sequence.lock``. Holding\n"
                "    that lock from the final queue/ledger scan until the watch lock is released\n"
                "    gives zero mode a real cutoff: a racing send either lands before the scan\n"
                "    and prevents exit, or lands after the watcher is no longer advertised.\n"
                '    """\n'
                "    failed_debt = daemon.architect_notes_failed_debt_error()"
                "\n"
                "    if failed_debt is not None:\n"
                "        return None, failed_debt\n"
                "    if backlog_outcome is False:\n"
                "        return None, None\n"
                "    lock_path = daemon.os.path.join("
                'daemon.MAILBOX, ".sequence.lock")\n'
                "    lock_file = None\n"
                "    try:\n"
                '        lock_file = open(lock_path, "a+", encoding="utf-8")\n'
                "        pass  # publication lock omitted\n"),
            arm_zero_cycle_cutoff_serializes_sender,
        ),
        (
            "cycle completion barrier released before watch lock",
            lambda text: replace_exact(
                text,
                "            release_dispatch_lock(lock_file=dispatch_lock)\n"
                "            if skip_redteam_lock is not None:\n"
                "                release_skip_redteam_lock("
                "lock_file=skip_redteam_lock)\n"
                "            if cycle_completion_barrier is not None:\n"
                "                release_cycle_completion_barrier(\n"
                "                    lock_file=cycle_completion_barrier)\n",
                "            if cycle_completion_barrier is not None:\n"
                "                release_cycle_completion_barrier(\n"
                "                    lock_file=cycle_completion_barrier)\n"
                "            release_dispatch_lock(lock_file=dispatch_lock)\n"
                "            if skip_redteam_lock is not None:\n"
                "                release_skip_redteam_lock("
                "lock_file=skip_redteam_lock)\n"),
            arm_zero_cycle_cutoff_serializes_sender,
        ),
        (
            "ledger replacement after read is ignored",
            lambda text: replace_exact(
                text,
                '            return None, "backlog ledger changed identity '
                'while reading"',
                "            pass  # ledger replacement ignored"),
            arm_cycle_ledger_read_stability,
        ),
        (
            "same-inode ledger mutation after read is ignored",
            lambda text: replace_exact(
                text,
                '            return None, "backlog ledger changed while '
                'verifying identity"',
                "            pass  # ledger metadata mutation ignored"),
            arm_cycle_ledger_read_stability,
        ),
        (
            "ledger open can block on FIFO replacement",
            lambda text: replace_exact(
                text,
                '    if hasattr(daemon.os, "O_NONBLOCK"):\n'
                "        flags = flags | daemon.os.O_NONBLOCK\n",
                ""),
            arm_cycle_ledger_preopen_fifo_is_bounded,
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
        ("no-child launch recovery", arm_no_child_launch_is_restartable),
        ("candidate audit outcome", arm_candidate_audit_requires_one_outcome),
        ("three-role pipeline", arm_three_role_pipeline_concurrency),
        ("finite modes", arm_once_and_dry_run_are_unaffected),
        ("once restart recovery", arm_once_recovers_prelaunch_request),
        ("watch token exhaustion", arm_watch_exits_on_token_exhaustion),
        ("once token exhaustion", arm_once_exits_on_token_exhaustion),
        ("cycle arguments", arm_cycle_argument_contract),
        ("cycle omitted", arm_omitted_cycle_remains_unbounded),
        ("cycle positive", arm_positive_cycle_limit_preserves_queue),
        ("cycle positive exact", arm_positive_cycle_waits_for_exact_boundary),
        ("public Architect finite admission",
         arm_finite_architect_admission_blocks_second_user_request),
        ("public Architect non-ticket outcomes",
         arm_public_architect_non_ticket_outcomes_release_admission),
        ("public Architect ambiguous outcomes",
         arm_public_architect_ambiguous_outcomes_fail_closed),
        ("cycle one finite reservation",
         arm_cycle_one_never_overlaps_next_ticket),
        ("two-role finite restart",
         arm_two_role_limit_is_restart_safe),
        ("reached-limit raw admin debt",
         arm_reached_limit_raw_admin_fails_once),
        ("cycle zero", arm_zero_cycle_drains_ledger_and_preserves_cadence),
        ("cycle zero queue", arm_zero_cycle_waits_for_queue),
        ("cycle zero pre-cutoff send",
         arm_zero_cycle_send_before_cutoff_is_seen),
        ("cycle zero cutoff sender",
         arm_zero_cycle_cutoff_serializes_sender),
        ("cycle zero ledger fail-closed", arm_cycle_ledger_fail_closed),
        ("cycle ledger read stability", arm_cycle_ledger_read_stability),
        ("cycle ledger pre-open FIFO",
         arm_cycle_ledger_preopen_fifo_is_bounded),
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
