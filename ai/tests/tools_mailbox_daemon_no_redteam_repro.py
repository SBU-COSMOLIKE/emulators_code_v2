#!/usr/bin/env python3
"""Scratch-only witnesses for the mailbox daemon's two-role watch mode.

Every arm redirects mailbox, ledger, logs, locks, and child commands into a
temporary repository.  No Claude or Codex executable is invoked.
"""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import types


AI_ROOT = Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"


class AttributeProxy:
    """Delegate attributes except for explicit scratch overrides."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


def load_daemon(source=None):
    """Execute one fresh daemon module, optionally from a source mutant."""
    if source is None:
        source = DAEMON_PATH.read_text(encoding="utf-8")
    module = types.ModuleType("mailbox_daemon_no_redteam_repro")
    module.__file__ = str(DAEMON_PATH)
    exec(compile(source, str(DAEMON_PATH), "exec"), module.__dict__)
    return module


def install_test_sol_topology_proof(daemon):
    """Install an explicit synthetic Sol topology proof in a scratch daemon.

    Arguments:
      daemon = the freshly loaded scratch daemon module.

    Returns:
      None.
    """
    expected_proof = object()

    def validate_test_topology():
        return expected_proof

    def revalidate_test_topology(proof):
        if proof is not expected_proof:
            raise AssertionError("scratch Sol topology proof changed")
        return expected_proof

    daemon.validate_live_sol_dispatch_topology = validate_test_topology
    daemon.revalidate_sol_dispatch_topology = revalidate_test_topology


@contextlib.contextmanager
def scratch_daemon(source=None):
    """Redirect a fresh daemon and every state path into a temporary tree."""
    with tempfile.TemporaryDirectory(prefix="mailbox-no-redteam-") as tmp:
        root = Path(tmp)
        ai_root = root / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        mailbox.mkdir(parents=True)
        backlog = ai_root / "notes" / "backlog.md"
        backlog.write_text("", encoding="utf-8")
        shared_lane = root / "claude-lane"
        sol_lane = root / "sol-lane"
        shared_lane.mkdir()
        sol_lane.mkdir()

        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
        daemon.MAILBOX = str(mailbox)
        daemon.DONE = str(mailbox / "done")
        daemon.RELAY_DIR = str(ai_root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(backlog)
        daemon.PREAMBLE = "scratch mailbox message\n"
        daemon.AGENT_COMMANDS = {
            "fable": ["fable-cli"],
            "opus": ["opus-cli"],
            "sol": ["sol-cli"],
        }
        daemon.AGENT_CWD = {
            "fable": str(shared_lane),
            "opus": str(shared_lane),
            "sol": str(sol_lane),
        }
        install_test_sol_topology_proof(daemon=daemon)
        daemon.report_landing_debt = lambda: None
        yield daemon, root, mailbox, backlog


def call_main(daemon, arguments):
    """Call ``main`` with isolated argv and captured terminal streams."""
    previous = sys.argv
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
            except BaseException as exc:
                error = exc
    finally:
        sys.argv = previous
    if isinstance(error, SystemExit):
        rc = error.code if isinstance(error.code, int) else 1
    elif error is None:
        rc = 0 if result is None else result
    else:
        rc = 1
    return rc, stdout.getvalue(), stderr.getvalue(), error


def captured_send(daemon, agent, text, dry_run, ticket_kind=None):
    """Call the production send path and capture its terminal output."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        outcome = daemon.send(
            agent=agent, text=text, dry_run=dry_run,
            ticket_kind=ticket_kind)
    return outcome, stream.getvalue()


@contextlib.contextmanager
def cleared_policy_environment(daemon):
    """Keep scratch sidecar witnesses independent of inherited watch state."""
    names = [daemon.FIX_ONLY_ENVIRONMENT,
             daemon.SKIP_REDTEAM_ENVIRONMENT]
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def all_watch_locks_reacquire(daemon):
    """Acquire and release all three public watch locks as a cleanup probe."""
    dispatch_lock = daemon.acquire_dispatch_lock(mode="watch")
    fix_only_lock = daemon.acquire_fix_only_lock()
    skip_redteam_lock = daemon.acquire_skip_redteam_lock()
    acquired = all(
        lock is not None
        for lock in (dispatch_lock, fix_only_lock, skip_redteam_lock))
    if skip_redteam_lock is not None:
        daemon.release_skip_redteam_lock(lock_file=skip_redteam_lock)
    if fix_only_lock is not None:
        daemon.release_fix_only_lock(lock_file=fix_only_lock)
    if dispatch_lock is not None:
        daemon.release_dispatch_lock(lock_file=dispatch_lock)
    return acquired


def write_message(mailbox, name, body):
    """Write one exact scratch root message and return its path."""
    path = mailbox / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(body)
    return path


def write_three_messages(mailbox):
    """Publish one valid message for each stable route."""
    return {
        "fable": write_message(
            mailbox, "0001-to-fable.md", "Architect scratch unit\n"),
        "opus": write_message(
            mailbox, "0002-to-opus.md", "Implementer scratch unit\n"),
        "sol": write_message(
            mailbox, "0003-to-sol.md",
            "MAILBOX-TICKET: closure\n\nSol scratch unit\n"),
    }


def install_harmless_children(daemon, captures):
    """Replace Popen with an immediate clean child that records its input."""
    class CleanProcess:
        returncode = 0

        def poll(self):
            return 0

        def wait(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def harmless_popen(command, stdout, stderr, cwd, env):
        del stderr
        command_agent = Path(command[0]).name.replace("-cli", "")
        if command_agent in {"fable", "opus", "sol"}:
            agent = command_agent
        elif "exec" in command:
            agent = "sol"
        else:
            model = command[command.index("--model") + 1]
            agent = ("fable" if model == daemon.DEFAULT_ARCHITECT_MODEL
                     else "opus")
        captures.append({
            "agent": agent,
            "prompt": command[-1],
            "cwd": cwd,
            "env": dict(env),
        })
        stdout.write("scratch child complete\n")
        stdout.flush()
        return CleanProcess()

    daemon.subprocess = AttributeProxy(
        daemon.subprocess, Popen=harmless_popen)


def file_identity(path):
    """Return the bytes and stable metadata used by the preservation arm."""
    state = path.stat()
    return (path.read_bytes(), state.st_dev, state.st_ino,
            state.st_size, state.st_mtime_ns)


def no_sol_state_artifacts(mailbox):
    """Return whether no state directory contains a claimed Sol message."""
    for directory in ["inflight", "done", "failed"]:
        if list((mailbox / directory).glob("*-to-sol.md")):
            return False
    return True


def arm_default_full_topology(source=None):
    """Omitting the option still launches and consumes all three routes."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        write_three_messages(mailbox=mailbox)
        captures = []
        install_harmless_children(daemon=daemon, captures=captures)
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 3
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        rc, output, errors, error = call_main(
            daemon, ["--watch", "--cycle", "0"])
        launched = sorted(item["agent"] for item in captures)
        done = sorted(path.name for path in (mailbox / "done").glob("*.md"))
        passed = (
            rc == 0 and error is None and errors == ""
            and launched == ["fable", "opus", "sol"]
            and done == ["0001-to-fable.md", "0002-to-opus.md",
                         "0003-to-sol.md"]
            and "two-role watch:" not in output
            and "red-team route disabled" not in output)
        print("default full topology=" + str(passed))
        return passed


def arm_two_role_watch_preserves_sol(source=None):
    """The option runs both Claude roles and leaves Sol byte-identical."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, ledger):
        messages = write_three_messages(mailbox=mailbox)
        ledger.write_text("".join(
            "- OPEN scratch item " + str(index) + "\n"
            for index in range(8)), encoding="utf-8")
        sol_before = file_identity(messages["sol"])
        captures = []
        install_harmless_children(daemon=daemon, captures=captures)
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 2
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        rc, output, errors, error = call_main(
            daemon, ["--watch", "--skip-redteam", "--cycle", "1"])

        launched = sorted(item["agent"] for item in captures)
        prompts_bound = all(
            item["prompt"].count("two-role watch:") == 1
            and "create no to-sol messages" in item["prompt"]
            and item["env"].get(daemon.SKIP_REDTEAM_ENVIRONMENT) == "1"
            for item in captures)
        sol_preserved = (
            messages["sol"].is_file()
            and file_identity(messages["sol"]) == sol_before
            and no_sol_state_artifacts(mailbox=mailbox))
        output_truth = (
            "two-role watch: Red Team and the entire Sol route are disabled"
            in output
            and "red-team route disabled; leaving 1 to-sol message queued "
            "and untouched." in output
            and "Give Sol separate implementation jobs" not in output)

        # A later normal dispatch must consume the exact deferred file.
        restart = daemon.process_backlog(dry_run=False)
        restart_agents = [item["agent"] for item in captures]
        sol_resumed = (
            restart is True and restart_agents.count("sol") == 1
            and not messages["sol"].exists()
            and (mailbox / "done" / messages["sol"].name).is_file())
        passed = (
            rc == 0 and error is None and errors == ""
            and launched == ["fable", "opus"]
            and prompts_bound and sol_preserved and output_truth
            and sol_resumed)
        print("two-role preserve/resume=" + str(passed))
        return passed


def arm_combined_fix_only_two_role_watch(source=None):
    """The composed watch keeps both policies live through dispatch."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, ledger):
        messages = write_three_messages(mailbox=mailbox)
        ledger.write_text("- OPEN scratch closure\n", encoding="utf-8")
        sol_before = file_identity(messages["sol"])
        captures = []
        install_harmless_children(daemon=daemon, captures=captures)
        live_lock_states = []
        harmless_popen = daemon.subprocess.Popen

        def observed_popen(*args, **kwargs):
            live_lock_states.append((
                daemon.fix_only_watch_is_active(),
                daemon.skip_redteam_watch_is_active(),
                daemon.dispatch_lock_is_live_watch(
                    mailbox=daemon.MAILBOX)))
            return harmless_popen(*args, **kwargs)

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=observed_popen)
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 2
        daemon.RENDEZVOUS_MINUTE_INTERVAL = 1000
        rc, output, errors, error = call_main(
            daemon,
            ["--watch", "--fix-only", "yes", "--skip-redteam",
             "--cycle", "1"])

        launched = sorted(item["agent"] for item in captures)
        child_policy = all(
            "fix-only watch: active" in item["prompt"]
            and "two-role watch:" in item["prompt"]
            and item["env"].get(daemon.FIX_ONLY_ENVIRONMENT) == "1"
            and item["env"].get(daemon.SKIP_REDTEAM_ENVIRONMENT) == "1"
            for item in captures)
        sol_preserved = (
            messages["sol"].is_file()
            and file_identity(messages["sol"]) == sol_before
            and no_sol_state_artifacts(mailbox=mailbox))
        cleanup_visible = (
            not daemon.fix_only_watch_is_active()
            and not daemon.skip_redteam_watch_is_active()
            and not daemon.dispatch_lock_is_live_watch(
                mailbox=daemon.MAILBOX))

        # Kernel-level reacquisition proves final cleanup, rather than merely
        # trusting the public probes' interpretation of owner text.
        dispatch_probe = daemon.acquire_dispatch_lock(mode="watch")
        fix_probe = daemon.acquire_fix_only_lock()
        skip_probe = daemon.acquire_skip_redteam_lock()
        reacquired = all(
            lock is not None
            for lock in (dispatch_probe, fix_probe, skip_probe))
        if skip_probe is not None:
            daemon.release_skip_redteam_lock(lock_file=skip_probe)
        if fix_probe is not None:
            daemon.release_fix_only_lock(lock_file=fix_probe)
        if dispatch_probe is not None:
            daemon.release_dispatch_lock(lock_file=dispatch_probe)

        passed = (
            rc == 0 and error is None and errors == ""
            and launched == ["fable", "opus"]
            and child_policy and sol_preserved
            and live_lock_states == [(True, True, True),
                                     (True, True, True)]
            and "fix-only watch active:" in output
            and "red-team route disabled" in output
            and cleanup_visible and reacquired)
        print("combined fix-only/two-role watch=" + str(passed))
        return passed


def arm_cycle_zero_defers_sol(source=None):
    """Cycle zero completes enabled work without claiming the mailbox empty."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        sol = write_message(
            mailbox, "0001-to-sol.md",
            "MAILBOX-TICKET: closure\n\nheld review\n")
        before = file_identity(sol)
        launches = []

        def forbidden_popen(*args, **kwargs):
            launches.append((args, kwargs))
            raise AssertionError("cycle zero launched deferred Sol")

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=forbidden_popen)
        rc, output, errors, error = call_main(
            daemon, ["--watch", "--no-red-team", "--cycle", "0"])
        passed = (
            rc == 0 and error is None and errors == "" and launches == []
            and sol.is_file() and file_identity(sol) == before
            and "Architect and Implementer lanes idle" in output
            and "enabled mailbox routes and ledger empty" in output
            and "1 Sol message deferred" in output
            and "all lanes idle; mailbox and ledger empty" not in output
            and not daemon.skip_redteam_watch_is_active())
        print("cycle zero deferred Sol=" + str(passed))
        return passed


def arm_completion_barrier_ignores_deferred_sol(source=None):
    """The cycle-zero cutoff treats deferred Sol as outside enabled work."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        sol = write_message(
            mailbox, "0001-to-sol.md",
            "MAILBOX-TICKET: closure\n\nheld review\n")
        before = file_identity(sol)
        barrier, error = daemon.acquire_cycle_completion_barrier(
            backlog_outcome=None, skip_redteam=True)
        passed = (
            barrier is not None and error is None
            and sol.is_file() and file_identity(sol) == before)
        if barrier is not None:
            daemon.release_cycle_completion_barrier(lock_file=barrier)
        print("completion barrier ignores deferred Sol=" + str(passed))
        return passed


def arm_cli_contract(source=None):
    """The two aliases are watch-only and compose with cycle/fix-only."""
    invalid = [
        ["--skip-redteam"],
        ["--once", "--skip-redteam"],
        ["--dry-run", "--skip-redteam"],
        ["--send", "opus", "--unit", "body", "--skip-redteam"],
        ["--ping", "sol", "--skip-redteam"],
    ]
    rejected = True
    for arguments in invalid:
        with scratch_daemon(source=source) as (daemon, _root, _mailbox,
                                                _ledger):
            rc, output, _errors, _error = call_main(daemon, arguments)
            rejected = rejected and rc == 1 and (
                "--skip-redteam is valid only with --watch" in output)

    accepted = True
    for arguments in [
            ["--watch", "--skip-redteam", "--cycle", "0"],
            ["--watch", "--no-red-team", "--cycle", "0"],
            ["--watch", "--skip-redteam", "--fix-only", "yes",
             "--cycle", "0"]]:
        with scratch_daemon(source=source) as (daemon, _root, _mailbox,
                                                _ledger):
            rc, _output, errors, error = call_main(daemon, arguments)
            accepted = accepted and (
                rc == 0 and errors == "" and error is None)

    daemon = load_daemon(source=source)
    rc, help_text, errors, error = call_main(daemon, ["--help"])
    normalized_help = " ".join(help_text.split())
    help_ok = (
        isinstance(error, SystemExit) and rc == 0 and errors == ""
        and "--skip-redteam, --no-red-team" in normalized_help
        and "leave existing to-sol messages" in normalized_help)
    passed = rejected and accepted and help_ok
    print("two-role CLI contract=" + str(passed))
    return passed


def arm_live_mode_refuses_sol_sends(source=None):
    """The held marker blocks Sol send/ping but permits Claude sends."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        daemon.warn_if_mailbox_unwatched = lambda: None
        mode_lock = daemon.acquire_skip_redteam_lock()
        if mode_lock is None:
            print("live mode Sol refusal=False")
            return False
        try:
            sol_send = daemon.send(
                agent="sol", text="held Sol work", dry_run=False,
                ticket_kind="closure")
            sol_ping = daemon.send(
                agent="sol", text=daemon.transport_ping_text(agent="sol"),
                dry_run=False, ticket_kind="transport")
            fable_send = daemon.send(
                agent="fable", text="Architect work", dry_run=False)
            held = daemon.skip_redteam_watch_is_active()
        finally:
            daemon.release_skip_redteam_lock(lock_file=mode_lock)
        after_release = daemon.send(
            agent="sol", text="later Sol work", dry_run=False,
            ticket_kind="closure")
        names = sorted(path.name for path in mailbox.glob("*.md"))

        old_environment = os.environ.get(daemon.SKIP_REDTEAM_ENVIRONMENT)
        os.environ[daemon.SKIP_REDTEAM_ENVIRONMENT] = "1"
        try:
            inherited_refused = not daemon.send(
                agent="sol", text="inherited", dry_run=True,
                ticket_kind="closure")
        finally:
            if old_environment is None:
                os.environ.pop(daemon.SKIP_REDTEAM_ENVIRONMENT, None)
            else:
                os.environ[daemon.SKIP_REDTEAM_ENVIRONMENT] = old_environment

        passed = (
            not sol_send and not sol_ping and fable_send and held
            and after_release and inherited_refused
            and names == ["0001-to-fable.md", "0002-to-sol.md"]
            and not daemon.skip_redteam_watch_is_active())
        print("live mode Sol refusal=" + str(passed))
        return passed


def arm_watch_owns_mode_marker(source=None):
    """A live two-role watch holds its policy marker until final cleanup."""
    with scratch_daemon(source=source) as (daemon, _root, _mailbox, _ledger):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        result = {}

        def paused_process_backlog(dry_run, fix_only=False,
                                   skip_redteam=False):
            calls.append((dry_run, fix_only, skip_redteam))
            entered.set()
            release.wait(timeout=2.0)
            return None

        daemon.process_backlog = paused_process_backlog

        def run_watch():
            result["call"] = call_main(
                daemon, ["--watch", "--skip-redteam", "--cycle", "0"])

        watcher = threading.Thread(target=run_watch)
        watcher.start()
        reached = entered.wait(timeout=2.0)
        active_during = daemon.skip_redteam_watch_is_active()
        refused_during = not daemon.send(
            agent="sol", text="concurrent review", dry_run=True,
            ticket_kind="closure")
        release.set()
        watcher.join(timeout=2.0)
        active_after = daemon.skip_redteam_watch_is_active()
        rc, _output, errors, error = result.get(
            "call", (None, "", "", RuntimeError("watch did not finish")))
        passed = (
            reached and not watcher.is_alive()
            and calls == [(False, False, True)]
            and active_during and refused_during and not active_after
            and rc == 0 and errors == "" and error is None)
        print("watch-owned two-role marker=" + str(passed))
        return passed


def arm_mode_lock_rejects_redirect(source=None):
    """A redirected mailbox path cannot publish a trusted mode marker."""
    with scratch_daemon(source=source) as (daemon, root, _mailbox, _ledger):
        outside = root / "outside-mailbox"
        outside.mkdir()
        redirected = root / "redirected-mailbox"
        redirected.symlink_to(outside, target_is_directory=True)
        daemon.MAILBOX = str(redirected)
        daemon.DONE = str(redirected / "done")
        lock = daemon.acquire_skip_redteam_lock()
        passed = lock is None and list(outside.iterdir()) == []
        if lock is not None:
            daemon.release_skip_redteam_lock(lock_file=lock)
        print("redirected mode lock refused=" + str(passed))
        return passed


def send_first_skip_activation_race(source=None, verbose=True):
    """A sender already inside sequence publication finishes first."""
    with scratch_daemon(source=source) as (daemon, _root, _mailbox, _ledger):
        with cleared_policy_environment(daemon):
            watch_lock = daemon.acquire_dispatch_lock(mode="watch")
            if watch_lock is None:
                return False

            sender_in_critical = threading.Event()
            activation_may_finish = threading.Event()
            order = []
            errors = []
            results = {}
            real_next_seq = daemon.next_seq
            real_activation = (
                daemon.acquire_skip_redteam_lock_while_sequence_locked)
            real_fcntl = daemon.fcntl
            real_os = daemon.os
            sender_thread = None
            activation_thread = None

            def paused_next_seq():
                if threading.current_thread() is sender_thread:
                    sender_in_critical.set()
                    if not activation_may_finish.wait(timeout=2.0):
                        raise TimeoutError(
                            "two-role activation never contended on sequence")
                return real_next_seq()

            def observed_link(source_path, destination_path):
                result = real_os.link(source_path, destination_path)
                if threading.current_thread() is sender_thread:
                    order.append("publish")
                return result

            def observed_flock(descriptor, operation):
                # Baseline activation first asks for plain LOCK_EX on the
                # sequence file. A mutant without that serialization reaches
                # the mode sidecar's LOCK_EX|LOCK_NB directly instead.
                if (threading.current_thread() is activation_thread
                        and operation == real_fcntl.LOCK_EX):
                    activation_may_finish.set()
                return real_fcntl.flock(descriptor, operation)

            def observed_activation():
                lock_file = real_activation()
                if lock_file is not None:
                    order.append("activate")
                activation_may_finish.set()
                return lock_file

            daemon.next_seq = paused_next_seq
            daemon.os = AttributeProxy(real_os, link=observed_link)
            daemon.fcntl = AttributeProxy(
                real_fcntl, flock=observed_flock)
            daemon.acquire_skip_redteam_lock_while_sequence_locked = (
                observed_activation)

            def send_target():
                try:
                    results["send"], results["send_output"] = captured_send(
                        daemon, agent="sol", text="close scratch review",
                        dry_run=False, ticket_kind="closure")
                except BaseException as exc:
                    errors.append(("send", repr(exc)))

            def activation_target():
                try:
                    results["mode_lock"] = (
                        daemon.acquire_skip_redteam_lock())
                except BaseException as exc:
                    errors.append(("activation", repr(exc)))
                    activation_may_finish.set()

            sender_thread = threading.Thread(
                target=send_target, name="skip-race-sender", daemon=True)
            activation_thread = threading.Thread(
                target=activation_target, name="skip-race-activator",
                daemon=True)
            sender_thread.start()
            reached = sender_in_critical.wait(timeout=2.0)
            if reached:
                activation_thread.start()
            else:
                activation_may_finish.set()
            sender_thread.join(timeout=4.0)
            if reached:
                activation_thread.join(timeout=4.0)

            mode_lock = results.get("mode_lock")
            mode_live = (mode_lock is not None
                         and daemon.skip_redteam_watch_is_active())
            passed = (
                reached and not sender_thread.is_alive()
                and (not reached or not activation_thread.is_alive())
                and errors == [] and results.get("send") is True
                and mode_live and order == ["publish", "activate"]
                and len(daemon.pending_messages()) == 1)
            if verbose:
                print("two-role send-first order=" + repr(order)
                      + " passed=" + str(passed))
            activation_may_finish.set()
            if mode_lock is not None:
                daemon.release_skip_redteam_lock(lock_file=mode_lock)
            daemon.release_dispatch_lock(lock_file=watch_lock)
            return passed


def activation_first_skip_sender_race(source=None, verbose=True):
    """Activation after a stale first probe is caught by the final probe."""
    with scratch_daemon(source=source) as (daemon, _root, _mailbox, _ledger):
        with cleared_policy_environment(daemon):
            watch_lock = daemon.acquire_dispatch_lock(mode="watch")
            if watch_lock is None:
                return False
            first_check_done = threading.Event()
            activation_done = threading.Event()
            errors = []
            results = {}
            original_active = daemon.skip_redteam_watch_is_active
            sender_thread = None
            sender_checks = 0

            def paused_active_probe(mailbox=None):
                nonlocal sender_checks
                result = original_active(mailbox=mailbox)
                if (threading.current_thread() is sender_thread
                        and sender_checks == 0):
                    sender_checks = sender_checks + 1
                    first_check_done.set()
                    if not activation_done.wait(timeout=2.0):
                        raise TimeoutError(
                            "two-role activation did not finish")
                    return result
                if threading.current_thread() is sender_thread:
                    sender_checks = sender_checks + 1
                return result

            daemon.skip_redteam_watch_is_active = paused_active_probe

            def send_target():
                try:
                    results["send"], results["send_output"] = captured_send(
                        daemon, agent="sol", text="close scratch review",
                        dry_run=False, ticket_kind="closure")
                except BaseException as exc:
                    errors.append(("send", repr(exc)))

            sender_thread = threading.Thread(
                target=send_target, name="skip-activation-first-sender",
                daemon=True)
            sender_thread.start()
            reached = first_check_done.wait(timeout=2.0)
            mode_lock = None
            try:
                if reached:
                    mode_lock = daemon.acquire_skip_redteam_lock()
            finally:
                activation_done.set()
            sender_thread.join(timeout=4.0)
            mode_live = (mode_lock is not None
                         and daemon.skip_redteam_watch_is_active())
            passed = (
                reached and not sender_thread.is_alive() and errors == []
                and mode_live and results.get("send") is False
                and sender_checks >= 2
                and "two-role" in results.get("send_output", "").lower()
                and daemon.pending_messages() == [])
            if verbose:
                print("two-role activation-first refusal=" + str(passed))
            if mode_lock is not None:
                daemon.release_skip_redteam_lock(lock_file=mode_lock)
            daemon.release_dispatch_lock(lock_file=watch_lock)
            return passed


def arm_skip_activation_publication_is_serialized(source=None):
    """Mode activation and Sol publication have a total atomic order."""
    return (
        send_first_skip_activation_race(source=source)
        and activation_first_skip_sender_race(source=source))


def skip_redteam_path_substitution_is_refused(source=None, verbose=True):
    """Replace the public sidecar after precheck and before mode flock."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        mode_path = mailbox / daemon.SKIP_REDTEAM_LOCK_NAME
        replacement = mailbox / ".replacement-skip-lock"
        opened_alias = mailbox / ".opened-skip-lock"
        displaced = mailbox / ".displaced-skip-lock"
        replacement.write_text("attacker replacement\n", encoding="utf-8")
        real_fcntl = daemon.fcntl
        real_os = daemon.os
        injected = False
        restored = False

        def swapping_flock(descriptor, operation):
            nonlocal injected
            if (not injected
                    and operation
                    == (real_fcntl.LOCK_EX | real_fcntl.LOCK_NB)):
                real_os.replace(mode_path, opened_alias)
                real_os.replace(replacement, mode_path)
                injected = True
            return real_fcntl.flock(descriptor, operation)

        def restoring_fsync(descriptor):
            nonlocal restored
            # A mutant missing the immediate post-flock identity check reaches
            # owner publication. Restore the opened inode so its later check
            # cannot accidentally hide that omission.
            if injected and not restored:
                real_os.replace(mode_path, displaced)
                real_os.replace(opened_alias, mode_path)
                restored = True
            return real_os.fsync(descriptor)

        daemon.fcntl = AttributeProxy(real_fcntl, flock=swapping_flock)
        daemon.os = AttributeProxy(real_os, fsync=restoring_fsync)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            mode_lock = daemon.acquire_skip_redteam_lock()
        active = daemon.skip_redteam_watch_is_active()
        passed = (
            injected and not restored and mode_lock is None and not active
            and mode_path.read_text(encoding="utf-8")
            == "attacker replacement\n"
            and "path changed" in stream.getvalue().lower())
        if verbose:
            print("two-role path substitution refused=" + str(passed))
        if mode_lock is not None:
            daemon.release_skip_redteam_lock(lock_file=mode_lock)
        return passed


def arm_skip_acquisition_failure_cleans_prior_locks(source=None):
    """A failed second policy activation releases earlier watch locks."""
    with scratch_daemon(source=source) as (daemon, _root, _mailbox, _ledger):
        with cleared_policy_environment(daemon):
            releases = []
            real_release_fix = daemon.release_fix_only_lock
            real_release_dispatch = daemon.release_dispatch_lock
            real_acquire_skip = daemon.acquire_skip_redteam_lock

            def observed_fix_release(lock_file):
                releases.append("fix")
                return real_release_fix(lock_file=lock_file)

            def observed_dispatch_release(lock_file):
                releases.append("dispatch")
                return real_release_dispatch(lock_file=lock_file)

            daemon.release_fix_only_lock = observed_fix_release
            daemon.release_dispatch_lock = observed_dispatch_release
            daemon.acquire_skip_redteam_lock = lambda: None
            rc, _output, errors, error = call_main(
                daemon,
                ["--watch", "--fix-only", "yes", "--skip-redteam",
                 "--cycle", "0"])
            recorded = list(releases)
            daemon.release_fix_only_lock = real_release_fix
            daemon.release_dispatch_lock = real_release_dispatch
            daemon.acquire_skip_redteam_lock = real_acquire_skip
            cleanup_visible = (
                not daemon.fix_only_watch_is_active()
                and not daemon.skip_redteam_watch_is_active()
                and not daemon.dispatch_lock_is_live_watch(
                    mailbox=daemon.MAILBOX))
            reacquired = all_watch_locks_reacquire(daemon=daemon)
            passed = (
                rc == 1 and error is None and errors == ""
                and recorded == ["fix", "dispatch"]
                and cleanup_visible and reacquired)
            print("failed skip activation cleanup=" + str(passed))
            return passed


def arm_combined_watch_exception_cleans_all_locks(source=None):
    """An exception after both policy locks releases every watch lock."""
    with scratch_daemon(source=source) as (daemon, _root, _mailbox, _ledger):
        with cleared_policy_environment(daemon):
            releases = []
            live_before_exception = []
            real_release_fix = daemon.release_fix_only_lock
            real_release_skip = daemon.release_skip_redteam_lock
            real_release_dispatch = daemon.release_dispatch_lock

            def observed_fix_release(lock_file):
                releases.append("fix")
                return real_release_fix(lock_file=lock_file)

            def observed_skip_release(lock_file):
                releases.append("skip")
                return real_release_skip(lock_file=lock_file)

            def observed_dispatch_release(lock_file):
                releases.append("dispatch")
                return real_release_dispatch(lock_file=lock_file)

            def failing_process_backlog(dry_run, fix_only=False,
                                        skip_redteam=False):
                live_before_exception.append((
                    dry_run, fix_only, skip_redteam,
                    daemon.fix_only_watch_is_active(),
                    daemon.skip_redteam_watch_is_active(),
                    daemon.dispatch_lock_is_live_watch(
                        mailbox=daemon.MAILBOX)))
                raise RuntimeError("scratch combined-watch failure")

            daemon.release_fix_only_lock = observed_fix_release
            daemon.release_skip_redteam_lock = observed_skip_release
            daemon.release_dispatch_lock = observed_dispatch_release
            daemon.process_backlog = failing_process_backlog
            rc, _output, errors, error = call_main(
                daemon,
                ["--watch", "--fix-only", "yes", "--skip-redteam",
                 "--cycle", "0"])
            recorded = list(releases)
            daemon.release_fix_only_lock = real_release_fix
            daemon.release_skip_redteam_lock = real_release_skip
            daemon.release_dispatch_lock = real_release_dispatch
            cleanup_visible = (
                not daemon.fix_only_watch_is_active()
                and not daemon.skip_redteam_watch_is_active()
                and not daemon.dispatch_lock_is_live_watch(
                    mailbox=daemon.MAILBOX))
            reacquired = all_watch_locks_reacquire(daemon=daemon)
            passed = (
                rc == 1 and isinstance(error, RuntimeError)
                and str(error) == "scratch combined-watch failure"
                and errors == ""
                and live_before_exception
                == [(False, True, True, True, True, True)]
                and sorted(recorded) == ["dispatch", "fix", "skip"]
                and len(recorded) == 3
                and cleanup_visible and reacquired)
            print("combined-watch exception cleanup=" + str(passed))
            return passed


def arm_sol_inflight_does_not_block_claude(source=None):
    """A historical Sol blocker is outside the enabled two-role lanes."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        inflight = mailbox / "inflight"
        inflight.mkdir()
        held = write_message(
            inflight, "0001-to-sol.md",
            "MAILBOX-TICKET: closure\n\nambiguous old Sol turn\n")
        held_before = file_identity(held)
        opus = write_message(
            mailbox, "0002-to-opus.md", "Implementer work\n")
        captures = []
        install_harmless_children(daemon=daemon, captures=captures)
        outcome = daemon.process_backlog(
            dry_run=False, skip_redteam=True)
        passed = (
            outcome is True
            and [item["agent"] for item in captures] == ["opus"]
            and held.is_file() and file_identity(held) == held_before
            and not opus.exists()
            and (mailbox / "done" / opus.name).is_file())
        print("Sol inflight outside enabled lanes=" + str(passed))
        return passed


def arm_collocated_sol_inflight_blocks_claude(source=None):
    """A Sol inflight blocker still binds a shared Claude working tree."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        daemon.AGENT_CWD["sol"] = daemon.AGENT_CWD["opus"]
        inflight = mailbox / "inflight"
        inflight.mkdir()
        held = write_message(
            inflight, "0001-to-sol.md",
            "MAILBOX-TICKET: closure\n\nshared-tree old Sol turn\n")
        held_before = file_identity(held)
        opus = write_message(
            mailbox, "0002-to-opus.md", "Implementer work\n")
        opus_before = file_identity(opus)
        captures = []
        install_harmless_children(daemon=daemon, captures=captures)
        outcome = daemon.process_backlog(
            dry_run=False, skip_redteam=True)
        passed = (
            outcome is False and captures == []
            and held.is_file() and file_identity(held) == held_before
            and opus.is_file() and file_identity(opus) == opus_before)
        print("collocated Sol inflight blocks Claude=" + str(passed))
        return passed


def arm_direct_sol_dispatch_is_defensive(source=None):
    """Even an erroneous direct call cannot claim Sol in two-role mode."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        sol = write_message(
            mailbox, "0001-to-sol.md",
            "MAILBOX-TICKET: closure\n\ndirect call\n")
        before = file_identity(sol)
        result = daemon.dispatch(
            path=str(sol), dry_run=False, skip_redteam=True)
        passed = (
            result is False and sol.is_file() and file_identity(sol) == before
            and no_sol_state_artifacts(mailbox=mailbox))
        print("direct Sol dispatch deferred=" + str(passed))
        return passed


def arm_process_filter_preserves_sol(source=None):
    """The process-level topology filter drains Claude and preserves Sol."""
    with scratch_daemon(source=source) as (daemon, _root, mailbox, _ledger):
        messages = write_three_messages(mailbox=mailbox)
        sol_before = file_identity(messages["sol"])
        captures = []
        install_harmless_children(daemon=daemon, captures=captures)
        outcome = daemon.process_backlog(
            dry_run=False, skip_redteam=True)
        launched = sorted(item["agent"] for item in captures)
        passed = (
            outcome is True and launched == ["fable", "opus"]
            and messages["sol"].is_file()
            and file_identity(messages["sol"]) == sol_before
            and not messages["fable"].exists()
            and not messages["opus"].exists()
            and no_sol_state_artifacts(mailbox=mailbox))
        print("process filter preserves Sol=" + str(passed))
        return passed


def replace_exact(source, old, new):
    """Replace one unique source anchor, or return ``None`` when unarmed."""
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def mutate_skip_activation_without_sequence_lock(source):
    """Publish the two-role sidecar without serializing Sol senders."""
    start = source.find("def acquire_skip_redteam_lock():")
    end = source.find("\ndef ", start + 1)
    if start < 0 or end < 0:
        return None
    function_source = source[start:end]
    body_start = function_source.find(
        "    os.makedirs(MAILBOX, exist_ok=True)\n")
    if body_start < 0:
        return None
    replacement = (
        "    os.makedirs(MAILBOX, exist_ok=True)\n"
        "    return acquire_skip_redteam_lock_while_sequence_locked()")
    mutant_function = function_source[:body_start] + replacement
    return source[:start] + mutant_function + source[end:]


def mutate_drop_skip_final_send_recheck(source):
    """Remove the Sol policy probe made inside sequence publication."""
    start = source.find("def send(agent, text, dry_run, ticket_kind=None):")
    end = source.find("\ndef ", start + 1)
    if start < 0 or end < 0:
        return None
    function_source = source[start:end]
    old = (
        "            reason = refusal_now()\n"
        "            if reason is not None:\n"
        "                print(\"refused --send sol: \" + reason + \".\")\n"
        "                return False\n"
        "            for _ in range(20):\n")
    new = "            for _ in range(20):\n"
    mutant_function = replace_exact(function_source, old, new)
    if mutant_function is None:
        return None
    return source[:start] + mutant_function + source[end:]


def mutate_drop_skip_post_flock_inode_check(source):
    """Remove the identity check immediately after the sidecar flock."""
    start = source.find(
        "def acquire_skip_redteam_lock_while_sequence_locked():")
    end = source.find("\ndef ", start + 1)
    if start < 0 or end < 0:
        return None
    function_source = source[start:end]
    old = (
        "    if not path_still_names_opened_inode(opened=opened):\n"
        "        print(\"cannot disable the red-team route: mode lock path "
        "changed \"\n"
        "              \"while its lock was acquired\")\n"
        "        release_skip_redteam_lock(lock_file=lock_file)\n"
        "        return None\n")
    mutant_function = replace_exact(function_source, old, "")
    if mutant_function is None:
        return None
    return source[:start] + mutant_function + source[end:]


def probe_skip_activation_serialization(source):
    """Activation cannot overtake a Sol sender inside publication."""
    return send_first_skip_activation_race(source=source, verbose=False)


def probe_skip_final_send_recheck(source):
    """A sender that saw stale policy must recheck after activation."""
    return activation_first_skip_sender_race(source=source, verbose=False)


def probe_skip_post_flock_inode_check(source):
    """A replaced two-role sidecar is rejected immediately after flock."""
    return skip_redteam_path_substitution_is_refused(
        source=source, verbose=False)


def arm_source_mutations():
    """Kill one source mutant for every binding two-role contract."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    cases = [
        (
            "flag polarity inverted",
            lambda text: replace_exact(
                text,
                'dest="skip_redteam", action="store_true"',
                'dest="skip_redteam", action="store_false"'),
            arm_default_full_topology,
        ),
        (
            "watch drops mode before processing",
            lambda text: replace_exact(
                text,
                "                        backlog_outcome = process_backlog(\n"
                "                            dry_run=False, skip_redteam=True)\n",
                "                        backlog_outcome = process_backlog(\n"
                "                            dry_run=False)\n"),
            arm_two_role_watch_preserves_sol,
        ),
        (
            "process filter admits only Sol",
            lambda text: replace_exact(
                text,
                "                       os.path.basename(path)).group(1) != \"sol\"]\n"
                "    else:\n"
                "        backlog = all_backlog\n",
                "                       os.path.basename(path)).group(1) == \"sol\"]\n"
                "    else:\n"
                "        backlog = all_backlog\n"),
            arm_process_filter_preserves_sol,
        ),
        (
            "cycle zero counts deferred Sol",
            lambda text: replace_exact(
                text,
                "        waiting_before = enabled_pending_messages(\n"
                "            skip_redteam=skip_redteam)\n"
                "        waiting_after = enabled_pending_messages(\n"
                "            skip_redteam=skip_redteam)\n",
                "        waiting_before = pending_messages()\n"
                "        waiting_after = pending_messages()\n"),
            arm_completion_barrier_ignores_deferred_sol,
        ),
        (
            "dynamic banner drops topology",
            lambda text: replace_exact(
                text,
                "        skip_redteam=skip_redteam)\n"
                "    # The dynamic banner precedes",
                "        skip_redteam=False)\n"
                "    # The dynamic banner precedes"),
            arm_two_role_watch_preserves_sol,
        ),
        (
            "child environment drops topology",
            lambda text: replace_exact(
                text,
                "        if skip_redteam:\n"
                "            env[SKIP_REDTEAM_ENVIRONMENT] = \"1\"\n"
                "        else:\n"
                "            env.pop(SKIP_REDTEAM_ENVIRONMENT, None)\n",
                "        env.pop(SKIP_REDTEAM_ENVIRONMENT, None)\n"),
            arm_two_role_watch_preserves_sol,
        ),
        (
            "disabled mode prints second-Implementer hint",
            lambda text: replace_exact(
                text,
                "    if total >= SECOND_IMPLEMENTER_THRESHOLD and not skip_redteam:\n",
                "    if total >= SECOND_IMPLEMENTER_THRESHOLD:\n"),
            arm_two_role_watch_preserves_sol,
        ),
        (
            "active mode permits Sol sends",
            lambda text: replace_exact(
                text,
                "        if skip_redteam_policy_active():\n"
                "            return (\"an active two-role watch has the Sol route disabled; \"\n"
                "                    \"wait for it to end or restart without --skip-redteam\")\n",
                "        if False:\n"
                "            return \"mode ignored\"\n"),
            arm_live_mode_refuses_sol_sends,
        ),
        (
            "combined watch drops two-role policy",
            lambda text: replace_exact(
                text,
                "                        backlog_outcome = process_backlog(\n"
                "                            dry_run=False, fix_only=True,\n"
                "                            skip_redteam=True)\n",
                "                        backlog_outcome = process_backlog(\n"
                "                            dry_run=False, fix_only=True,\n"
                "                            skip_redteam=False)\n"),
            arm_combined_fix_only_two_role_watch,
        ),
        (
            "two-role activation skips sequence serialization",
            mutate_skip_activation_without_sequence_lock,
            probe_skip_activation_serialization,
        ),
        (
            "Sol publication drops final two-role policy recheck",
            mutate_drop_skip_final_send_recheck,
            probe_skip_final_send_recheck,
        ),
        (
            "post-flock two-role inode check removed",
            mutate_drop_skip_post_flock_inode_check,
            probe_skip_post_flock_inode_check,
        ),
        (
            "watch omits two-role mode marker",
            lambda text: replace_exact(
                text,
                "        skip_redteam_lock = None\n"
                "        if skip_redteam:\n"
                "            skip_redteam_lock = acquire_skip_redteam_lock()\n"
                "            if skip_redteam_lock is None:\n"
                "                if fix_only_lock is not None:\n"
                "                    release_fix_only_lock(lock_file=fix_only_lock)\n"
                "                release_dispatch_lock(lock_file=dispatch_lock)\n"
                "                return 1\n",
                "        skip_redteam_lock = None\n"),
            arm_watch_owns_mode_marker,
        ),
        (
            "failed skip activation omits fix-lock cleanup",
            lambda text: replace_exact(
                text,
                "            if skip_redteam_lock is None:\n"
                "                if fix_only_lock is not None:\n"
                "                    release_fix_only_lock("
                "lock_file=fix_only_lock)\n"
                "                release_dispatch_lock("
                "lock_file=dispatch_lock)\n"
                "                return 1\n",
                "            if skip_redteam_lock is None:\n"
                "                release_dispatch_lock("
                "lock_file=dispatch_lock)\n"
                "                return 1\n"),
            arm_skip_acquisition_failure_cleans_prior_locks,
        ),
        (
            "failed skip activation omits dispatch cleanup",
            lambda text: replace_exact(
                text,
                "            if skip_redteam_lock is None:\n"
                "                if fix_only_lock is not None:\n"
                "                    release_fix_only_lock("
                "lock_file=fix_only_lock)\n"
                "                release_dispatch_lock("
                "lock_file=dispatch_lock)\n"
                "                return 1\n",
                "            if skip_redteam_lock is None:\n"
                "                if fix_only_lock is not None:\n"
                "                    release_fix_only_lock("
                "lock_file=fix_only_lock)\n"
                "                return 1\n"),
            arm_skip_acquisition_failure_cleans_prior_locks,
        ),
        (
            "combined exception omits fix-lock cleanup",
            lambda text: replace_exact(
                text,
                "            if fix_only_lock is not None:\n"
                "                release_fix_only_lock("
                "lock_file=fix_only_lock)\n"
                "            if skip_redteam_lock is None:\n",
                "            if skip_redteam_lock is None:\n"),
            arm_combined_watch_exception_cleans_all_locks,
        ),
        (
            "combined exception omits dispatch cleanup",
            lambda text: replace_exact(
                text,
                "            else:\n"
                "                release_dispatch_lock("
                "lock_file=dispatch_lock)\n"
                "                release_skip_redteam_lock("
                "lock_file=skip_redteam_lock)\n",
                "            else:\n"
                "                release_skip_redteam_lock("
                "lock_file=skip_redteam_lock)\n"),
            arm_combined_watch_exception_cleans_all_locks,
        ),
        (
            "combined exception omits skip-lock cleanup",
            lambda text: replace_exact(
                text,
                "            else:\n"
                "                release_dispatch_lock("
                "lock_file=dispatch_lock)\n"
                "                release_skip_redteam_lock("
                "lock_file=skip_redteam_lock)\n",
                "            else:\n"
                "                release_dispatch_lock("
                "lock_file=dispatch_lock)\n"),
            arm_combined_watch_exception_cleans_all_locks,
        ),
        (
            "Sol inflight blocks Claude lane result",
            lambda text: replace_exact(
                text,
                "        if (skip_redteam and agent == \"sol\"\n"
                "                and cwd not in enabled_claude_cwds):\n"
                "            continue\n",
                "        if False:\n"
                "            continue\n"),
            arm_sol_inflight_does_not_block_claude,
        ),
        (
            "collocated Sol blocker is discarded",
            lambda text: replace_exact(
                text,
                "        if (skip_redteam and agent == \"sol\"\n"
                "                and cwd not in enabled_claude_cwds):\n"
                "            continue\n",
                "        if skip_redteam and agent == \"sol\":\n"
                "            continue\n"),
            arm_collocated_sol_inflight_blocks_claude,
        ),
        (
            "watch-only validation removed",
            lambda text: replace_exact(
                text,
                "    if args.skip_redteam:\n"
                "        conflicting_action = (\n"
                "            not args.watch or args.once or args.send is not None\n"
                "            or args.ping is not None or args.dry_run)\n"
                "        if conflicting_action:\n"
                "            print(\"--skip-redteam is valid only with --watch\")\n"
                "            return 1\n",
                ""),
            arm_cli_contract,
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
    """Run every isolated runtime arm and source mutation."""
    arms = [
        ("default topology", arm_default_full_topology),
        ("two-role preservation", arm_two_role_watch_preserves_sol),
        ("combined fix-only/two-role",
         arm_combined_fix_only_two_role_watch),
        ("cycle zero", arm_cycle_zero_defers_sol),
        ("cycle completion barrier",
         arm_completion_barrier_ignores_deferred_sol),
        ("CLI contract", arm_cli_contract),
        ("live send refusal", arm_live_mode_refuses_sol_sends),
        ("watch marker lifetime", arm_watch_owns_mode_marker),
        ("mode path redirect", arm_mode_lock_rejects_redirect),
        ("activation/publication ordering",
         arm_skip_activation_publication_is_serialized),
        ("mode path substitution",
         skip_redteam_path_substitution_is_refused),
        ("activation failure cleanup",
         arm_skip_acquisition_failure_cleans_prior_locks),
        ("exception cleanup",
         arm_combined_watch_exception_cleans_all_locks),
        ("Sol inflight isolation", arm_sol_inflight_does_not_block_claude),
        ("collocated Sol inflight",
         arm_collocated_sol_inflight_blocks_claude),
        ("direct dispatch defense", arm_direct_sol_dispatch_is_defensive),
        ("process route filter", arm_process_filter_preserves_sol),
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
    raise SystemExit(main())
