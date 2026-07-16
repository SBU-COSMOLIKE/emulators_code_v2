#!/usr/bin/env python3
"""Scratch-only witnesses for Sol ticket deferral and fix-only watches.

The production mailbox is live infrastructure.  Every arm in this file loads
the daemon afresh and redirects all path globals to a temporary repository.
No arm reads, writes, or locks the real mailbox.
"""

import argparse
import contextlib
import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import threading
import types


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
README_PATH = AI_ROOT / "tools" / "README.md"
TICKET_HEADER = "MAILBOX-TICKET: "
SEVERITY_HEADER = "MAILBOX-SEVERITY: "
SCOPE_HEADER = "MAILBOX-SCOPE: "
FIX_ONLY_BANNER = (
    "fix-only watch: active; close existing ledger lines only; create no "
    "discovery tickets or new backlog lines.")
BASE_COMMIT = "1" * 40
ACCEPTED_COMMIT = "a" * 40
SCRATCH_HIGH_ANCHOR = "scratch-high-bug-fix-1"


class AttributeProxy:
    """Delegate module attributes except for explicit test overrides."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


def load_daemon(source=None):
    """Load one fresh production daemon, or a caller-supplied source mutant."""
    if source is not None:
        module = types.ModuleType("mailbox_daemon_fix_only_mutant")
        module.__file__ = str(DAEMON_PATH)
        exec(compile(source, str(DAEMON_PATH), "exec"), module.__dict__)
        return module
    spec = importlib.util.spec_from_file_location(
        "mailbox_daemon_fix_only_repro", DAEMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
def scratch_daemon(open_count=0, create_mailbox=True, source=None,
                   critical_count=0, medium_count=0, low_count=0,
                   high_feature_count=0, reopen_count=0):
    """Point a fresh daemon at a disposable classified backlog.

    ``open_count`` retains its historical name and now means open High bug
    fixes. The other arguments create exact ticket types needed by the two
    independent severity-count tests. ``reopen_count`` writes the canonical
    Red Team bookkeeping row in every generated ticket detail.
    """
    with tempfile.TemporaryDirectory(prefix="mailbox-fix-only-") as tmp:
        root = pathlib.Path(tmp)
        ai_root = root / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        backlog = ai_root / "notes" / "backlog.md"
        backlog.parent.mkdir(parents=True)
        lines = []
        detail_anchors = []
        serial = 0
        for label, count, ticket_type in (
                ("CRITICAL", critical_count, "BUG FIX"),
                ("HIGH", high_feature_count, "NEW FUNCTIONALITY"),
                ("HIGH", open_count, "BUG FIX"),
                ("MEDIUM", medium_count, "BUG FIX"),
                ("LOW", low_count, "BUG FIX")):
            for index in range(count):
                serial += 1
                anchor = ("scratch-" + label.lower() + "-"
                          + ticket_type.lower().replace(" ", "-") + "-"
                          + str(serial))
                title = "Scratch " + label.lower() + " ticket " + str(index)
                lines.append(
                    "- OPEN **" + label + "** **" + ticket_type
                    + "** — [" + title + "](#" + anchor + ")\n")
                detail_anchors.append(
                    '<a id="' + anchor + '"></a>\n## ' + title + "\n\n"
                    "**Red Team reopen count: " + str(reopen_count)
                    + ".**\n\n"
                    "**Red Team reopening: allowed.**\n")
        backlog.write_text(
            "".join(lines) + "\n" + "".join(detail_anchors),
            encoding="utf-8")
        if create_mailbox:
            mailbox.mkdir(parents=True)

        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
        daemon.MAILBOX = str(mailbox)
        daemon.DONE = str(mailbox / "done")
        daemon.RELAY_DIR = str(ai_root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(backlog)
        daemon.PREAMBLE = "scratch preamble\n"
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"],
            "opus": ["harmless-opus"],
            "sol": ["harmless-sol"],
        }
        daemon.AGENT_CWD = {
            "fable": str(root),
            "opus": str(root),
            "sol": str(root),
        }
        daemon.git_commit_exists = lambda commit: commit == BASE_COMMIT
        daemon.git_commit_descends_from = (
            lambda starting_commit, accepted_commit:
            starting_commit == BASE_COMMIT
            and accepted_commit == ACCEPTED_COMMIT)
        install_test_sol_topology_proof(daemon=daemon)
        # These are side effects of a successful publication, not the policy
        # under test.  Stubbing them also makes a refusal's zero-call property
        # explicit in the dedicated arm below.
        daemon.warn_if_mailbox_unwatched = lambda: None
        daemon.report_demand = lambda backlog: None
        yield daemon, root, mailbox, backlog


def tree_snapshot(root):
    """Return a content-and-type snapshot of a scratch repository tree."""
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


def read_text_exact(path):
    """Read text without newline translation on every supported Python."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def write_pending(daemon, name, body="counted pending unit\n"):
    """Write one root-level pending message in the scratch mailbox."""
    path = pathlib.Path(daemon.MAILBOX) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="")
    return path


def normal_review_exchange(daemon, text):
    """Prepare one accepted normal ticket and its exact Red Team exchange."""
    cycle_id = SCRATCH_HIGH_ANCHOR + "@" + BASE_COMMIT
    flow = (
        "MAILBOX-FLOW: ticket\n"
        "MAILBOX-CYCLE: " + cycle_id + "\n"
        "MAILBOX-MODE: normal\n\n")
    daemon.register_ticket_cycle_message(
        agent="opus", message=flow + "Implement the scratch fix.\n")
    daemon.register_ticket_cycle_message(
        agent="fable", message=flow + "Audit the scratch fix.\n")
    completed = daemon.record_architect_commit(
        cycle_id=cycle_id, accepted_commit=ACCEPTED_COMMIT, mode="normal")
    if completed != 0:
        raise AssertionError("a normal Architect commit completed a cycle")
    request = daemon.sol_ticket_payload(
        ticket_kind="closure", text=text,
        review_cycle=cycle_id, review_commit=ACCEPTED_COMMIT)
    receipt = daemon.redteam_review_receipt_payload(
        review_cycle=cycle_id, review_commit=ACCEPTED_COMMIT,
        result="NO CHANGE", text="No remaining bug in this scratch fix.")
    return request, receipt


def captured_send(daemon, agent, text, dry_run, ticket_kind=None,
                  severity=None, scope=None):
    """Call the production send path and capture its terminal output."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        outcome = daemon.send(agent=agent, text=text, dry_run=dry_run,
                              ticket_kind=ticket_kind, severity=severity,
                              scope=scope)
    return outcome, stream.getvalue()


def run_main(daemon, arguments):
    """Run main() with an isolated argv and capture stdout, stderr, and rc."""
    previous_argv = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    sys.argv = ["mailbox_daemon.py"] + list(arguments)
    try:
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            try:
                result = daemon.main()
                rc = 0 if result is None else result
            except SystemExit as exc:
                rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.argv = previous_argv
    return rc, stdout.getvalue(), stderr.getvalue()


def clean_process(stream, launches, command, cwd, env):
    """Return an already-finished harmless child and record its launch."""
    launches.append({
        "command": command,
        "cwd": cwd,
        "env": dict(env),
    })
    stream.write("harmless child output\n")
    stream.flush()
    return types.SimpleNamespace(
        returncode=0,
        poll=lambda: 0,
        wait=lambda: 0,
        kill=lambda: None)


def arm_threshold_edges_and_exact_header():
    """Nine non-Low tickets permit discovery; ten refuse only discovery."""
    threshold = load_daemon().DISCOVERY_ADMISSION_THRESHOLD
    checks = []

    with scratch_daemon(open_count=threshold - 1) as (
            daemon, _, _, _):
        outcome, _ = captured_send(
            daemon, agent="sol", text="quietly seek one new fact",
            dry_run=False, ticket_kind="discovery")
        pending = [pathlib.Path(path) for path in daemon.pending_messages()]
        exact = (len(pending) == 1
                 and read_text_exact(pending[0])
                 == (TICKET_HEADER + "discovery\n"
                     + SEVERITY_HEADER + "medium\n"
                     + SCOPE_HEADER + "bounded\n\n"
                     "quietly seek one new fact\n"))
        checks.append(outcome and exact)
        print("threshold-minus-one outcome=" + str(outcome)
              + " exact_header=" + str(exact))

    with scratch_daemon(open_count=threshold, create_mailbox=False) as (
            daemon, _, mailbox, _):
        outcome, output = captured_send(
            daemon, agent="sol", text="innocuous words only",
            dry_run=False, ticket_kind="discovery")
        refused = (not outcome and not mailbox.exists()
                   and "Critical, High, and Medium" in output
                   and "Low tickets do not count" in output
                   and "below" in output.lower())
        checks.append(refused)
        print("threshold-exact discovery_refused=" + str(refused))

    with scratch_daemon(open_count=threshold) as (daemon, _, _, _):
        outcome, _ = captured_send(
            daemon, agent="sol",
            text="Review, sweep, and probe while closing this OPEN line.",
            dry_run=False, ticket_kind="closure")
        pending = [pathlib.Path(path) for path in daemon.pending_messages()]
        exact = (len(pending) == 1
                 and read_text_exact(pending[0])
                 == (TICKET_HEADER + "closure\n\nReview, sweep, and probe "
                     "while closing this OPEN line.\n"))
        checks.append(outcome and exact)
        print("threshold-exact closure_allowed=" + str(outcome and exact))

    # Mailbox queue depth is informational and cannot saturate discovery.
    with scratch_daemon(open_count=threshold - 3) as (daemon, _, _, _):
        write_pending(daemon, "0001-to-fable.md")
        write_pending(daemon, "0002-to-opus.md")
        write_pending(daemon, "0003-to-sol.md")
        outcome, _ = captured_send(
            daemon, agent="sol", text="new finding", dry_run=False,
            ticket_kind="discovery")
        queue_excluded = outcome and len(daemon.pending_messages()) == 4
        checks.append(queue_excluded)
        print("mixed-lane queue excluded=" + str(queue_excluded))

    # Low tickets are also excluded, while Medium tickets participate.
    with scratch_daemon(open_count=threshold - 1, low_count=25) as (
            daemon, _, _, _):
        outcome, _ = captured_send(
            daemon, agent="sol", text="new finding", dry_run=False,
            ticket_kind="discovery")
        checks.append(outcome)
        print("low tickets excluded=" + str(outcome))
    with scratch_daemon(open_count=threshold - 1, medium_count=1,
                        create_mailbox=False) as (
            daemon, _, mailbox, _):
        outcome, _ = captured_send(
            daemon, agent="sol", text="new finding", dry_run=False,
            ticket_kind="discovery")
        medium_counts = not outcome and not mailbox.exists()
        checks.append(medium_counts)
        print("medium ticket counted=" + str(medium_counts))
    return all(checks)


def arm_refusal_is_zero_write():
    """Real and dry-run saturated refusals alter no path and call no reporter."""
    threshold = load_daemon().DISCOVERY_ADMISSION_THRESHOLD
    checks = []
    for dry_run in (False, True):
        with scratch_daemon(open_count=threshold,
                            create_mailbox=False) as (
                daemon, root, mailbox, _):
            calls = []
            daemon.warn_if_mailbox_unwatched = (
                lambda: calls.append("dead-mailbox-warning"))
            daemon.report_demand = (
                lambda backlog: calls.append(("demand", list(backlog))))
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol", text="discover a new problem",
                dry_run=dry_run, ticket_kind="discovery")
            after = tree_snapshot(root)
            passed = (
                not outcome
                and before == after
                and not mailbox.exists()
                and calls == []
                and "Critical, High, and Medium" in output
                and "Low tickets do not count" in output
                and "below" in output.lower()
                and "queued " not in output
                and "would queue" not in output)
            checks.append(passed)
            print("zero-write dry_run=" + str(dry_run)
                  + " passed=" + str(passed))
    return all(checks)


def arm_second_implementer_emergency_boundaries():
    """Only 2 Critical bugs or 11 High bugs unlock the extra Implementer."""
    checks = []
    cases = [
        ("one-critical", {"critical_count": 1}, False),
        ("two-critical", {"critical_count": 2}, True),
        ("ten-high-bugs", {"open_count": 10}, False),
        ("eleven-high-bugs", {"open_count": 11}, True),
        ("many-high-features", {"high_feature_count": 25}, False),
        ("low-and-medium", {"medium_count": 25, "low_count": 25}, False),
    ]
    assignment = (
        TICKET_HEADER + "closure\n\n### ARCHITECT_HANDOFF\n"
        "OpenAI Sol — this is a role as second Implementer for this unit.\n"
        "Implement the assigned emergency fix.\n")
    for label, arguments, expected in cases:
        with scratch_daemon(**arguments) as (daemon, _, _, _):
            counts = daemon.backlog_severity_counts()
            emergency = daemon.second_implementer_emergency(counts=counts)
            refusal = daemon.second_implementer_emergency_refusal(
                message=assignment, counts=counts)
            passed = (emergency is expected
                      and ((refusal is None) is expected))
            checks.append(passed)
            print(label + " emergency=" + str(emergency)
                  + " refusal=" + repr(refusal))
    return all(checks)


def arm_classification_is_explicit_and_fail_closed():
    """Missing/invalid kinds fail before writes; declared kind beats prose."""
    checks = []
    for ticket_kind in (None, "attack", "transport", "Closure", " closure "):
        with scratch_daemon(open_count=0, create_mailbox=False) as (
                daemon, root, mailbox, _):
            before = tree_snapshot(root)
            outcome, output = captured_send(
                daemon, agent="sol",
                text="Close an existing line without seeking findings.",
                dry_run=False, ticket_kind=ticket_kind)
            if ticket_kind == "transport":
                classified_refusal = (
                    "transport" in output and "--ping sol" in output)
            else:
                classified_refusal = (
                    "closure" in output and "discovery" in output)
            passed = (not outcome and before == tree_snapshot(root)
                      and not mailbox.exists()
                      and classified_refusal)
            checks.append(passed)
            print("invalid-kind " + repr(ticket_kind)
                  + " refused=" + str(passed))

    # The public CLI exposes no Sol address or ticket-classification option.
    # Internal callers above still own the strict classification contract.
    with scratch_daemon(open_count=0, create_mailbox=False) as (
            daemon, root, _, _):
        before = tree_snapshot(root)
        rc_sol, out_sol, err_sol = run_main(
            daemon, ["--send", "sol", "--unit", "close one line"])
        rc_kind, out_kind, err_kind = run_main(
            daemon, ["--send", "architect", "--unit", "close one line",
                     "--ticket-kind", "closure"])
        passed = (rc_sol != 0 and rc_kind != 0
                  and before == tree_snapshot(root)
                  and "invalid choice" in (out_sol + err_sol)
                  and "unrecognized arguments" in (out_kind + err_kind))
        checks.append(passed)
        print("CLI kind contract passed=" + str(passed))
    return all(checks)


def arm_inherited_fix_only_send():
    """A child send inherits closing-only mode through MAILBOX_FIX_ONLY=1."""
    previous = os.environ.get("MAILBOX_FIX_ONLY")
    try:
        os.environ["MAILBOX_FIX_ONLY"] = "1"
        with scratch_daemon(open_count=0, create_mailbox=False) as (
                daemon, root, mailbox, _):
            before = tree_snapshot(root)
            discovery, output = captured_send(
                daemon, agent="sol", text="seek one finding",
                dry_run=False, ticket_kind="discovery")
            discovery_blocked = (
                not discovery and before == tree_snapshot(root)
                and not mailbox.exists()
                and "fix-only" in output.lower())

        with scratch_daemon(open_count=0) as (daemon, _, _, _):
            closure, _ = captured_send(
                daemon, agent="sol", text="close an existing line",
                dry_run=False, ticket_kind="closure")
            pending = [pathlib.Path(path)
                       for path in daemon.pending_messages()]
            closure_allowed = (
                closure and len(pending) == 1
                and read_text_exact(pending[0])
                == (TICKET_HEADER + "closure\n\n"
                    "close an existing line\n"))

        # Transport is a narrow internal class, never a public target or
        # ticket-kind. Only the daemon's exact Sol body may use it.
        with scratch_daemon(open_count=0) as (daemon, _, _, _):
            ping_is_transport, _ = captured_send(
                daemon, agent="sol",
                text=daemon.transport_ping_text(agent="sol"),
                dry_run=False, ticket_kind="transport")
            pending = [pathlib.Path(path)
                       for path in daemon.pending_messages()]
            ping_is_transport = (ping_is_transport and len(pending) == 1
                and read_text_exact(pending[0])
                .startswith(TICKET_HEADER + "transport\n"))
    finally:
        if previous is None:
            os.environ.pop("MAILBOX_FIX_ONLY", None)
        else:
            os.environ["MAILBOX_FIX_ONLY"] = previous
    print("inherited discovery_blocked=" + str(discovery_blocked)
          + " closure_allowed=" + str(closure_allowed)
          + " ping_is_transport=" + str(ping_is_transport))
    return discovery_blocked and closure_allowed and ping_is_transport


def arm_live_watch_binds_external_send():
    """A held fix-only watch binds senders that inherit no environment."""
    previous = os.environ.pop("MAILBOX_FIX_ONLY", None)
    checks = []
    try:
        # Snapshot after both real locks are acquired.  Activation now
        # intentionally creates .sequence.lock; a refused sender may inspect
        # that state but must not change it or publish a message.
        with scratch_daemon(open_count=0) as (
                daemon, root, mailbox, _):
            watch_lock = daemon.acquire_dispatch_lock(mode="watch")
            if watch_lock is None:
                return False
            fix_only_lock = daemon.acquire_fix_only_lock()
            if fix_only_lock is None:
                daemon.release_dispatch_lock(lock_file=watch_lock)
                return False
            try:
                before = tree_snapshot(root)
                outcome, output = captured_send(
                    daemon, agent="sol", text="seek one finding",
                    dry_run=False, ticket_kind="discovery")
                refused_without_writes = (
                    not outcome and before == tree_snapshot(root)
                    and "fix-only" in output.lower())
                checks.append(refused_without_writes)
                print("live fix-only external discovery refused="
                      + str(refused_without_writes))
            finally:
                daemon.release_fix_only_lock(lock_file=fix_only_lock)
                daemon.release_dispatch_lock(lock_file=watch_lock)

        # Declared closure work remains legal while that same mode is live.
        with scratch_daemon(open_count=0) as (daemon, _, _, _):
            watch_lock = daemon.acquire_dispatch_lock(mode="watch")
            if watch_lock is None:
                return False
            fix_only_lock = daemon.acquire_fix_only_lock()
            if fix_only_lock is None:
                daemon.release_dispatch_lock(lock_file=watch_lock)
                return False
            try:
                outcome, _ = captured_send(
                    daemon, agent="sol", text="close one existing line",
                    dry_run=False, ticket_kind="closure")
                pending = [pathlib.Path(path)
                           for path in daemon.pending_messages()]
                closure_allowed = (
                    outcome and len(pending) == 1
                    and read_text_exact(pending[0]).startswith(
                            TICKET_HEADER + "closure\n"))
                checks.append(closure_allowed)
                print("live fix-only external closure allowed="
                      + str(closure_allowed))
            finally:
                daemon.release_fix_only_lock(lock_file=fix_only_lock)
                daemon.release_dispatch_lock(lock_file=watch_lock)

        # Persisted owner metadata is not authority after the kernel lock is
        # released.  A stopped fix-only watch must not poison later sends.
        with scratch_daemon(open_count=0) as (daemon, _, _, _):
            watch_lock = daemon.acquire_dispatch_lock(mode="watch")
            if watch_lock is None:
                return False
            fix_only_lock = daemon.acquire_fix_only_lock()
            if fix_only_lock is None:
                daemon.release_dispatch_lock(lock_file=watch_lock)
                return False
            daemon.release_fix_only_lock(lock_file=fix_only_lock)
            daemon.release_dispatch_lock(lock_file=watch_lock)
            outcome, _ = captured_send(
                daemon, agent="sol", text="seek one finding",
                dry_run=False, ticket_kind="discovery")
            stale_mode_ignored = outcome and len(daemon.pending_messages()) == 1
            checks.append(stale_mode_ignored)
            print("released fix-only watch ignored="
                  + str(stale_mode_ignored))

        # An ordinary watch proves liveness only.  It must not acquire the
        # closing-only policy merely because it holds the same loop lock.
        with scratch_daemon(open_count=0) as (daemon, _, _, _):
            watch_lock = daemon.acquire_dispatch_lock(mode="watch")
            if watch_lock is None:
                return False
            try:
                outcome, _ = captured_send(
                    daemon, agent="sol", text="seek one finding",
                    dry_run=False, ticket_kind="discovery")
                pending = daemon.pending_messages()
                ordinary_allows = outcome and len(pending) == 1
                checks.append(ordinary_allows)
                print("ordinary live watch discovery allowed="
                      + str(ordinary_allows))
            finally:
                daemon.release_dispatch_lock(lock_file=watch_lock)
    finally:
        if previous is not None:
            os.environ["MAILBOX_FIX_ONLY"] = previous
    return all(checks)


def send_first_activation_race(source=None, verbose=True):
    """Run the ordering where a sender already owns the sequence lock."""
    previous = os.environ.pop("MAILBOX_FIX_ONLY", None)
    try:
        with scratch_daemon(open_count=0, source=source) as (
                daemon, _, _, _):
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
                daemon.acquire_fix_only_lock_while_sequence_locked)
            real_fcntl = daemon.fcntl
            real_os = daemon.os
            sender_thread = None
            activation_thread = None

            def paused_next_seq():
                if threading.current_thread() is sender_thread:
                    sender_in_critical.set()
                    if not activation_may_finish.wait(timeout=2.0):
                        raise TimeoutError(
                            "activation never contended on sequence lock")
                return real_next_seq()

            def observed_link(source_path, destination_path):
                result = real_os.link(source_path, destination_path)
                if threading.current_thread() is sender_thread:
                    order.append("publish")
                return result

            def observed_flock(descriptor, operation):
                # The baseline activation first attempts plain LOCK_EX on the
                # sequence file.  A mutant that skips that serialization goes
                # directly to LOCK_EX|LOCK_NB on the mode sidecar, so its
                # sender stays paused until activation has actually finished.
                if (threading.current_thread() is activation_thread
                        and operation == real_fcntl.LOCK_EX):
                    activation_may_finish.set()
                return real_fcntl.flock(descriptor, operation)

            def observed_activation():
                lock_file = real_activation()
                if lock_file is not None:
                    order.append("activate")
                # This is the mutant path's gate.  In the baseline the plain
                # sequence-lock attempt above has already opened the gate.
                activation_may_finish.set()
                return lock_file

            daemon.next_seq = paused_next_seq
            daemon.os = AttributeProxy(real_os, link=observed_link)
            daemon.fcntl = AttributeProxy(
                real_fcntl, flock=observed_flock)
            daemon.acquire_fix_only_lock_while_sequence_locked = (
                observed_activation)

            def send_target():
                try:
                    results["send"], results["send_output"] = captured_send(
                        daemon, agent="sol", text="seek one finding",
                        dry_run=False, ticket_kind="discovery")
                except BaseException as exc:  # keep a failed arm bounded
                    errors.append(("send", repr(exc)))

            def activation_target():
                try:
                    results["mode_lock"] = daemon.acquire_fix_only_lock()
                except BaseException as exc:  # keep a failed arm bounded
                    errors.append(("activation", repr(exc)))
                    activation_may_finish.set()

            sender_thread = threading.Thread(
                target=send_target, name="fix-only-race-sender", daemon=True)
            activation_thread = threading.Thread(
                target=activation_target, name="fix-only-race-activator",
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
                         and daemon.fix_only_watch_is_active())
            passed = (
                reached and not sender_thread.is_alive()
                and (not reached or not activation_thread.is_alive())
                and errors == [] and results.get("send") is True
                and mode_live and order == ["publish", "activate"]
                and len(daemon.pending_messages()) == 1)
            if verbose:
                print("send-first activation order=" + repr(order)
                      + " passed=" + str(passed))
            if mode_lock is not None:
                daemon.release_fix_only_lock(lock_file=mode_lock)
            daemon.release_dispatch_lock(lock_file=watch_lock)
            return passed
    finally:
        if previous is not None:
            os.environ["MAILBOX_FIX_ONLY"] = previous


def activation_first_sender_race():
    """Run the ordering where activation wins after a stale first check."""
    previous = os.environ.pop("MAILBOX_FIX_ONLY", None)
    try:
        with scratch_daemon(open_count=0) as (daemon, _, _, _):
            watch_lock = daemon.acquire_dispatch_lock(mode="watch")
            if watch_lock is None:
                return False
            first_check_done = threading.Event()
            activation_done = threading.Event()
            errors = []
            results = {}
            original_active = daemon.fix_only_watch_is_active
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
                        raise TimeoutError("fix-only activation did not finish")
                    return result
                if threading.current_thread() is sender_thread:
                    sender_checks = sender_checks + 1
                return result

            daemon.fix_only_watch_is_active = paused_active_probe

            def send_target():
                try:
                    results["send"], results["send_output"] = captured_send(
                        daemon, agent="sol", text="seek one finding",
                        dry_run=False, ticket_kind="discovery")
                except BaseException as exc:  # keep a failed arm bounded
                    errors.append(("send", repr(exc)))

            sender_thread = threading.Thread(
                target=send_target, name="activation-first-sender",
                daemon=True)
            sender_thread.start()
            reached = first_check_done.wait(timeout=2.0)
            mode_lock = None
            try:
                if reached:
                    mode_lock = daemon.acquire_fix_only_lock()
            finally:
                activation_done.set()
            sender_thread.join(timeout=4.0)
            mode_live = (mode_lock is not None
                         and daemon.fix_only_watch_is_active())
            passed = (
                reached and not sender_thread.is_alive() and errors == []
                and mode_live and results.get("send") is False
                and sender_checks >= 2
                and "fix-only" in results.get("send_output", "").lower()
                and daemon.pending_messages() == [])
            print("activation-first final-check refusal=" + str(passed))
            if mode_lock is not None:
                daemon.release_fix_only_lock(lock_file=mode_lock)
            daemon.release_dispatch_lock(lock_file=watch_lock)
            return passed
    finally:
        if previous is not None:
            os.environ["MAILBOX_FIX_ONLY"] = previous


def arm_activation_publication_is_serialized():
    """The shared sequence lock makes activation/publication atomic."""
    return (send_first_activation_race()
            and activation_first_sender_race())


def fix_only_path_substitution_is_refused(source=None, verbose=True):
    """Replace the public sidecar after precheck and before mode flock."""
    with scratch_daemon(open_count=0, source=source) as (
            daemon, _, mailbox, _):
        mode_path = mailbox / daemon.FIX_ONLY_LOCK_NAME
        replacement = mailbox / ".replacement-mode-lock"
        opened_alias = mailbox / ".opened-mode-lock"
        displaced = mailbox / ".displaced-mode-lock"
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
            # Only a mutant that skips the immediate post-flock inode check
            # reaches owner publication.  Restore the original public inode
            # here so its later post-fsync check cannot mask that omission.
            if injected and not restored:
                real_os.replace(mode_path, displaced)
                real_os.replace(opened_alias, mode_path)
                restored = True
            return real_os.fsync(descriptor)

        daemon.fcntl = AttributeProxy(real_fcntl, flock=swapping_flock)
        daemon.os = AttributeProxy(real_os, fsync=restoring_fsync)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            mode_lock = daemon.acquire_fix_only_lock()
        active = daemon.fix_only_watch_is_active()
        passed = (
            injected and not restored and mode_lock is None and not active
            and mode_path.read_text(encoding="utf-8")
            == "attacker replacement\n"
            and "path changed" in stream.getvalue().lower())
        if verbose:
            print("fix-only path substitution refused=" + str(passed))
        if mode_lock is not None:
            daemon.release_fix_only_lock(lock_file=mode_lock)
        return passed


def arm_malformed_held_mode_fails_closed():
    """Held mode authority survives concurrent owner-text corruption."""
    previous = os.environ.pop("MAILBOX_FIX_ONLY", None)
    try:
        with scratch_daemon(open_count=0) as (daemon, root, _, _):
            mode_lock = daemon.acquire_fix_only_lock()
            if mode_lock is None:
                return False
            try:
                mode_lock.seek(0)
                mode_lock.truncate()
                mode_lock.write("malformed owner bytes\n")
                mode_lock.flush()
                daemon.os.fsync(mode_lock.fileno())
                before = tree_snapshot(root)
                active = daemon.fix_only_watch_is_active()
                outcome, output = captured_send(
                    daemon, agent="sol", text="seek one finding",
                    dry_run=False, ticket_kind="discovery")
                passed = (
                    active and not outcome and before == tree_snapshot(root)
                    and "fix-only" in output.lower()
                    and daemon.pending_messages() == [])
                print("malformed held mode remains active=" + str(passed))
                return passed
            finally:
                daemon.release_fix_only_lock(lock_file=mode_lock)
    finally:
        if previous is not None:
            os.environ["MAILBOX_FIX_ONLY"] = previous


def arm_admitted_discovery_does_not_count_itself():
    """A discovery admitted at nine non-Low tickets still launches."""
    threshold = load_daemon().DISCOVERY_ADMISSION_THRESHOLD
    checks = []
    previous = os.environ.pop("MAILBOX_FIX_ONLY", None)
    try:
        with scratch_daemon(open_count=threshold - 1) as (
                daemon, _, mailbox, _):
            queued, _ = captured_send(
                daemon, agent="sol", text="seek one new finding",
                dry_run=False, ticket_kind="discovery")
            pending = [pathlib.Path(path)
                       for path in daemon.pending_messages()]
            launches = []
            if queued and len(pending) == 1:
                dispatched, output = captured_dispatch(
                    daemon, pending[0], False, launches)
            else:
                dispatched, output = False, ""
            admitted_launches = (
                queued and dispatched and len(launches) == 1
                and (mailbox / "done" / pending[0].name).is_file()
                and "refused" not in output.lower())
            checks.append(admitted_launches)
            print("admission-nine discovery queued-and-launched="
                  + str(admitted_launches))

        with scratch_daemon(open_count=threshold,
                            create_mailbox=False) as (
                daemon, root, mailbox, _):
            before = tree_snapshot(root)
            queued, output = captured_send(
                daemon, agent="sol", text="seek one new finding",
                dry_run=False, ticket_kind="discovery")
            exact_threshold_refuses = (
                not queued and before == tree_snapshot(root)
                and not mailbox.exists()
                and "Critical, High, and Medium" in output
                and "below" in output.lower())
            checks.append(exact_threshold_refuses)
            print("admission-ten discovery refused="
                  + str(exact_threshold_refuses))
    finally:
        if previous is not None:
            os.environ["MAILBOX_FIX_ONLY"] = previous
    return all(checks)


def arm_concurrent_boundary_is_serialized():
    """Two sends at nine non-Low tickets both publish unique messages."""
    threshold = load_daemon().DISCOVERY_ADMISSION_THRESHOLD
    with scratch_daemon(open_count=threshold - 1) as (daemon, _, _, _):
        # Publication is serialized only to allocate unique sequence numbers.
        # A queued message is not an accepted backlog ticket, so the first
        # publication must not make the second request fail admission.
        outcomes = []
        outcome_lock = threading.Lock()

        def worker():
            outcome = daemon.send(
                agent="sol", text="new finding", dry_run=False,
                ticket_kind="discovery")
            with outcome_lock:
                outcomes.append(outcome)

        workers = [threading.Thread(target=worker) for _ in range(2)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(timeout=3.0)
        pending = [pathlib.Path(path) for path in daemon.pending_messages()]
        passed = (sorted(outcomes) == [True, True]
                  and len(pending) == 2
                  and all(read_text_exact(path).startswith(
                      TICKET_HEADER + "discovery\n") for path in pending)
                  and not any(thread.is_alive() for thread in workers))
        print("concurrent outcomes=" + repr(sorted(outcomes))
              + " pending=" + str(len(pending)))
        return passed


def one_watch_pass(daemon, value):
    """Run one fix-only watch pass without sleeping or touching a real lock."""
    calls = []
    sentinel = object()
    daemon.acquire_dispatch_lock = lambda mode: sentinel
    daemon.release_dispatch_lock = lambda lock_file: None

    def record_process(dry_run, fix_only=False):
        calls.append((dry_run, fix_only))
        return None

    daemon.process_backlog = record_process
    original_getmtime = daemon.os.path.getmtime
    stamps = iter([1.0, 2.0])
    daemon.os.path.getmtime = lambda path: next(stamps)
    try:
        rc, output, error = run_main(
            daemon, ["--watch", "--fix-only", value])
    finally:
        daemon.os.path.getmtime = original_getmtime
    return rc, output, error, calls


def arm_truthy_values_and_watch_scope():
    """Truthy parsing is forgiving, invalid values and non-watch uses are not."""
    checks = []
    for value in ("1", " TRUE ", "\tYeS\n"):
        with scratch_daemon() as (daemon, _, _, _):
            helper_value = daemon.truthy_fix_only(value)
            rc, output, error, calls = one_watch_pass(daemon, value)
            passed = (helper_value is True and rc == 0 and error == ""
                      and calls == [(False, True)]
                      and "fix-only watch active" in output
                      and "do not add tickets for newly found problems or "
                      "new backlog lines"
                      in output)
            checks.append(passed)
            print("truthy " + repr(value) + " passed=" + str(passed))

    for value in ("0", "false", "no", "on", "truee", ""):
        with scratch_daemon() as (daemon, _, _, _):
            try:
                daemon.truthy_fix_only(value)
                helper_rejected = False
            except argparse.ArgumentTypeError:
                helper_rejected = True
            rc, output, error = run_main(
                daemon, ["--watch", "--fix-only", value])
            passed = (helper_rejected and rc == 2
                      and "1, true, or yes" in (output + error).lower())
            checks.append(passed)
            print("invalid truthy " + repr(value)
                  + " rejected=" + str(passed))

    conflict_argv = [
        ["--fix-only", "true"],
        ["--once", "--fix-only", "true"],
        ["--dry-run", "--fix-only", "true"],
        ["--send", "architect", "--unit", "body", "--fix-only", "true"],
        ["--ping", "architect", "--fix-only", "true"],
        ["--watch", "--send", "architect", "--unit", "body",
         "--fix-only", "true"],
    ]
    for argv in conflict_argv:
        with scratch_daemon(create_mailbox=False) as (
                daemon, root, _, _):
            calls = []
            daemon.send = lambda *args, **kwargs: calls.append("send")
            daemon.process_backlog = (
                lambda *args, **kwargs: calls.append("process"))
            daemon.acquire_dispatch_lock = (
                lambda *args, **kwargs: calls.append("lock"))
            before = tree_snapshot(root)
            rc, output, error = run_main(daemon, argv)
            passed = (rc != 0 and calls == []
                      and before == tree_snapshot(root)
                      and "fix-only" in (output + error))
            checks.append(passed)
            print("scope " + repr(argv) + " rejected=" + str(passed))

    primary_conflicts = [
        ["--once", "--watch"],
        ["--watch", "--send", "architect", "--unit", "body"],
        ["--once", "--ping", "architect"],
        ["--send", "architect", "--unit", "body",
         "--ping", "architect"],
    ]
    for argv in primary_conflicts:
        with scratch_daemon(create_mailbox=False) as (
                daemon, root, _, _):
            calls = []
            daemon.send = lambda *args, **kwargs: calls.append("send")
            daemon.process_backlog = (
                lambda *args, **kwargs: calls.append("process"))
            daemon.acquire_dispatch_lock = (
                lambda *args, **kwargs: calls.append("lock"))
            before = tree_snapshot(root)
            rc, output, error = run_main(daemon, argv)
            passed = (rc != 0 and calls == []
                      and before == tree_snapshot(root)
                      and "one primary action" in (output + error))
            checks.append(passed)
            print("primary conflict " + repr(argv)
                  + " rejected=" + str(passed))
    return all(checks)


def captured_dispatch(daemon, path, fix_only, launches,
                      review_receipt=None):
    """Dispatch one scratch message with a harmless Popen replacement."""
    original_popen = daemon.subprocess.Popen

    def fake_popen(command, stdout, stderr, cwd, env):
        del stderr
        if review_receipt is not None:
            write_pending(
                daemon, "0999-to-fable.md", body=review_receipt)
        return clean_process(stdout, launches, command, cwd, env)

    daemon.subprocess.Popen = fake_popen
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            outcome = daemon.dispatch(path=str(path), dry_run=False,
                                      fix_only=fix_only)
    finally:
        daemon.subprocess.Popen = original_popen
    return outcome, stream.getvalue()


def arm_fix_only_dispatch_and_propagation():
    """Fix-only launches only declared Sol closures and binds every child."""
    checks = []

    with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
        body, receipt = normal_review_exchange(
            daemon=daemon, text="Review the accepted scratch fix.")
        path = write_pending(
            daemon, "0001-to-sol.md", body)
        launches = []
        outcome, _ = captured_dispatch(
            daemon, path, True, launches, review_receipt=receipt)
        prompt = launches[0]["command"][-1] if len(launches) == 1 else ""
        environment = launches[0]["env"] if len(launches) == 1 else {}
        passed = (outcome and len(launches) == 1
                  and FIX_ONLY_BANNER in prompt
                  and environment.get("MAILBOX_FIX_ONLY") == "1"
                  and (mailbox / "done" / path.name).is_file())
        checks.append(passed)
        print("fix-only closure launched=" + str(passed))

    # A Fable child is how new tickets would normally be authored.  Its
    # dynamic prompt and environment must carry the same binding mode.
    with scratch_daemon() as (daemon, _, mailbox, _):
        body = daemon.architect_user_request_payload(
            text="Close the existing scratch work only.")
        path = write_pending(daemon, "0002-to-fable.md", body)
        launches = []
        outcome, _ = captured_dispatch(daemon, path, True, launches)
        prompt = launches[0]["command"][-1] if len(launches) == 1 else ""
        environment = launches[0]["env"] if len(launches) == 1 else {}
        passed = (outcome and len(launches) == 1
                  and FIX_ONLY_BANNER in prompt
                  and environment.get("MAILBOX_FIX_ONLY") == "1"
                  and (mailbox / "done" / path.name).is_file())
        checks.append(passed)
        print("fix-only fable binding=" + str(passed))

    refused_bodies = [
        TICKET_HEADER + "discovery\n" + SEVERITY_HEADER + "medium\n"
        + SCOPE_HEADER + "bounded\n\nclose-sounding prose\n",
        "no ticket header\n",
        "body first\n" + TICKET_HEADER + "closure\n",
        TICKET_HEADER + "Closure\n",
        " MAILBOX-TICKET: closure\n",
        TICKET_HEADER + "transport\n\nnot the generated Sol ping\n",
    ]
    for index, body in enumerate(refused_bodies, start=3):
        with scratch_daemon() as (daemon, _, mailbox, _):
            path = write_pending(
                daemon, "%04d-to-sol.md" % index, body)
            launches = []
            outcome, output = captured_dispatch(
                daemon, path, True, launches)
            failed = mailbox / "failed" / path.name
            passed = (not outcome and launches == [] and failed.is_file()
                      and not path.exists()
                      and read_text_exact(failed)
                      == body
                      and "refused" in output.lower()
                      and ("fix-only" in output.lower()
                           or "mailbox-ticket: closure" in output.lower()
                           or "transport" in output.lower()))
            checks.append(passed)
            print("fix-only refused body " + str(index)
                  + " passed=" + str(passed))

    # The mandatory envelope must not make a placeholder look substantive.
    # Validation applies to the human body after the exact first line.
    with scratch_daemon(open_count=1) as (daemon, _, mailbox, _):
        body, _ = normal_review_exchange(daemon=daemon, text="<unit>")
        path = write_pending(daemon, "0009-to-sol.md", body)
        launches = []
        outcome, output = captured_dispatch(daemon, path, False, launches)
        failed = mailbox / "failed" / path.name
        passed = (not outcome and launches == [] and failed.is_file()
                  and read_text_exact(failed) == body
                  and "template placeholder '<unit>'" in output)
        checks.append(passed)
        print("Sol envelope placeholder refused=" + str(passed))

    # With the mode absent, declared discovery remains legitimate below the
    # saturation threshold and receives no active-mode environment or banner.
    with scratch_daemon() as (daemon, _, mailbox, _):
        path = write_pending(
            daemon, "0008-to-sol.md",
            TICKET_HEADER + "discovery\n" + SEVERITY_HEADER + "medium\n"
            + SCOPE_HEADER + "bounded\n\nseek a new finding\n")
        launches = []
        outcome, _ = captured_dispatch(daemon, path, False, launches)
        prompt = launches[0]["command"][-1] if len(launches) == 1 else ""
        environment = launches[0]["env"] if len(launches) == 1 else {}
        passed = (outcome and len(launches) == 1
                  and FIX_ONLY_BANNER not in prompt
                  and environment.get("MAILBOX_FIX_ONLY") != "1"
                  and (mailbox / "done" / path.name).is_file())
        checks.append(passed)
        print("ordinary discovery unchanged=" + str(passed))

    # Exercise the entire threaded path used by a watch.  Directly testing
    # dispatch() is insufficient if process_backlog() or drain_lane() drops
    # the mode on its way to the worker.
    checks.append(probe_pipeline_enforcement(source=None))
    return all(checks)


def probe_pipeline_enforcement(source):
    """Return whether fix-only survives process_backlog and its lane worker."""
    with scratch_daemon(source=source) as (daemon, _, mailbox, _):
        path = write_pending(
            daemon, "0090-to-sol.md",
            TICKET_HEADER + "discovery\n" + SEVERITY_HEADER + "medium\n"
            + SCOPE_HEADER + "bounded\n\nnew finding\n")
        launches = []
        original_popen = daemon.subprocess.Popen

        def fake_popen(command, stdout, stderr, cwd, env):
            del stderr
            return clean_process(stdout, launches, command, cwd, env)

        daemon.subprocess.Popen = fake_popen
        stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(stream):
                outcome = daemon.process_backlog(
                    dry_run=False, fix_only=True)
        finally:
            daemon.subprocess.Popen = original_popen
        passed = (outcome is False and launches == []
                  and (mailbox / "failed" / path.name).is_file()
                  and not path.exists())
        print("fix-only pipeline passed=" + str(passed))
        return passed


def probe_sol_envelope_placeholder(source):
    """The classification envelope must not hide an empty template body."""
    with scratch_daemon(open_count=1, source=source) as (
            daemon, _, mailbox, _):
        body, _ = normal_review_exchange(daemon=daemon, text="<unit>")
        path = write_pending(daemon, "0092-to-sol.md", body)
        launches = []
        outcome, output = captured_dispatch(daemon, path, False, launches)
        return (not outcome and launches == []
                and (mailbox / "failed" / path.name).is_file()
                and "template placeholder '<unit>'" in output)


def arm_help_and_readme_parity():
    """The checked-in options transcript is the real help, byte for byte."""
    daemon = load_daemon()
    rc, help_output, error = run_main(daemon, ["--help"])
    readme = README_PATH.read_text(encoding="utf-8")
    marker = "usage: mailbox_daemon.py"
    candidates = []
    position = 0
    while True:
        start = readme.find("```\n" + marker, position)
        if start < 0:
            break
        body_start = start + len("```\n")
        end = readme.find("```", body_start)
        if end < 0:
            break
        candidates.append(readme[body_start:end])
        position = end + 3
    passed = (rc == 0 and error == "" and candidates == [help_output]
              and "--send {architect}" in help_output
              and "--ping {architect}" in help_output
              and "--ticket-kind" not in help_output
              and "--severity {high,medium,low}" in help_output
              and "--fix-only" in help_output
              and all(word in help_output for word in ("1", "true", "yes")))
    print("help parity candidates=" + str(len(candidates))
          + " passed=" + str(passed))
    return passed


def replace_exact(source, old, new):
    """Return a one-site source mutant, or ``None`` when it is not armed."""
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def mutate_later_header(source):
    """Make the Sol classifier search the whole body instead of line one."""
    start = source.find("def sol_ticket_kind(")
    end = source.find("\ndef ", start + 1)
    if start < 0 or end < 0:
        return None
    function_source = source[start:end]
    old = ('    match = re.match(\n'
           '        r"\\A" + re.escape(SOL_TICKET_HEADER)')
    new = ('    match = re.search(\n'
           '        r"" + re.escape(SOL_TICKET_HEADER)')
    mutant_function = replace_exact(function_source, old, new)
    if mutant_function is None:
        return None
    return source[:start] + mutant_function + source[end:]


def mutate_header_trailing_space(source):
    """Make only the first-line classifier accept trailing spaces."""
    start = source.find("def sol_ticket_kind(")
    end = source.find("\ndef ", start + 1)
    if start < 0 or end < 0:
        return None
    function_source = source[start:end]
    old = '        + r")(?:\\r?\\n|\\Z)",'
    new = '        + r") *(?:\\r?\\n|\\Z)",'
    mutant_function = replace_exact(function_source, old, new)
    if mutant_function is None:
        return None
    return source[:start] + mutant_function + source[end:]


def mutate_activation_without_sequence_lock(source):
    """Make fix-only activation publish its sidecar without serialization."""
    start = source.find("def acquire_fix_only_lock():")
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
        "    return acquire_fix_only_lock_while_sequence_locked()")
    mutant_function = function_source[:body_start] + replacement
    return source[:start] + mutant_function + source[end:]


def mutate_drop_post_flock_inode_check(source):
    """Remove the sidecar identity check immediately after mode flock."""
    start = source.find("def acquire_fix_only_lock_while_sequence_locked():")
    end = source.find("\ndef ", start + 1)
    if start < 0 or end < 0:
        return None
    function_source = source[start:end]
    old = (
        "    if not path_still_names_opened_inode(opened=opened):\n"
        "        print(\"cannot activate fix-only mode: mode lock path "
        "changed while \"\n"
        "              \"its lock was acquired\")\n"
        "        release_fix_only_lock(lock_file=lock_file)\n"
        "        return None\n")
    mutant_function = replace_exact(function_source, old, "")
    if mutant_function is None:
        return None
    return source[:start] + mutant_function + source[end:]


def probe_threshold_comparator(source):
    """The exact threshold must produce a discovery refusal."""
    daemon = load_daemon(source=source)
    return daemon.sol_ticket_refusal(
        ticket_kind="discovery",
        admission_count=daemon.DISCOVERY_ADMISSION_THRESHOLD,
        fix_only=False, discovery_severity="medium",
        discovery_scope="bounded") is not None


def probe_critical_emergency_strict(source):
    """One Critical bug is insufficient and two are an emergency."""
    daemon = load_daemon(source=source)
    base = {
        "critical": 1,
        "high_bug_fix": 0,
    }
    two = dict(base, critical=2)
    return (not daemon.second_implementer_emergency(counts=base)
            and daemon.second_implementer_emergency(counts=two))


def probe_high_emergency_strict_and_typed(source):
    """Only eleven High bug fixes, not High features, are an emergency."""
    daemon = load_daemon(source=source)
    ten_bugs = {"critical": 0, "high_bug_fix": 10,
                "high_new_functionality": 0}
    eleven_bugs = dict(ten_bugs, high_bug_fix=11)
    features = dict(ten_bugs, high_bug_fix=0,
                    high_new_functionality=50)
    return (not daemon.second_implementer_emergency(counts=ten_bugs)
            and daemon.second_implementer_emergency(counts=eleven_bugs)
            and not daemon.second_implementer_emergency(counts=features))


def probe_truthy_whitespace(source):
    """Outer whitespace remains accepted."""
    daemon = load_daemon(source=source)
    try:
        return daemon.truthy_fix_only(" TRUE ") is True
    except argparse.ArgumentTypeError:
        return False


def probe_truthy_case(source):
    """Capitalization remains ignored."""
    daemon = load_daemon(source=source)
    try:
        return daemon.truthy_fix_only("YeS") is True
    except argparse.ArgumentTypeError:
        return False


def probe_truthy_vocabulary(source):
    """All three allowed tokens remain accepted."""
    daemon = load_daemon(source=source)
    try:
        return all(daemon.truthy_fix_only(value) is True
                   for value in ("1", "true", "yes"))
    except argparse.ArgumentTypeError:
        return False


def probe_exact_header_spacing(source):
    """Trailing whitespace on the classification line remains invalid."""
    daemon = load_daemon(source=source)
    return daemon.sol_ticket_kind(
        message=TICKET_HEADER + "closure \nbody\n") is None


def probe_first_line_only(source):
    """A valid-looking later line never classifies a Sol ticket."""
    daemon = load_daemon(source=source)
    return daemon.sol_ticket_kind(
        message="body first\n" + TICKET_HEADER + "closure\n") is None


def probe_missing_kind_refusal(source):
    """An unclassified Sol ticket remains fail-closed."""
    daemon = load_daemon(source=source)
    return daemon.sol_ticket_refusal(
        ticket_kind=None, admission_count=0, fix_only=False) is not None


def probe_fix_only_discovery_refusal(source):
    """Fix-only blocks discovery even below saturation."""
    daemon = load_daemon(source=source)
    return daemon.sol_ticket_refusal(
        ticket_kind="discovery", admission_count=0,
        fix_only=True, discovery_severity="medium",
        discovery_scope="bounded") is not None


def probe_early_zero_write(source):
    """The first policy check must run before mkdir and sequence locking."""
    threshold = load_daemon(source=source).DISCOVERY_ADMISSION_THRESHOLD
    with scratch_daemon(open_count=threshold, create_mailbox=False,
                        source=source) as (daemon, root, mailbox, _):
        previous = os.environ.pop("MAILBOX_FIX_ONLY", None)
        try:
            before = tree_snapshot(root)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                outcome = daemon.send(
                    agent="sol", text="new finding", dry_run=False,
                    ticket_kind="discovery")
            return (not outcome and before == tree_snapshot(root)
                    and not mailbox.exists())
        finally:
            if previous is not None:
                os.environ["MAILBOX_FIX_ONLY"] = previous


def probe_inherited_environment(source):
    """A child send must obey the active mode exported by its watch."""
    with scratch_daemon(open_count=0, create_mailbox=False,
                        source=source) as (daemon, root, mailbox, _):
        previous = os.environ.get("MAILBOX_FIX_ONLY")
        os.environ["MAILBOX_FIX_ONLY"] = "1"
        try:
            before = tree_snapshot(root)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                outcome = daemon.send(
                    agent="sol", text="new finding", dry_run=False,
                    ticket_kind="discovery")
            return (not outcome and before == tree_snapshot(root)
                    and not mailbox.exists())
        finally:
            if previous is None:
                os.environ.pop("MAILBOX_FIX_ONLY", None)
            else:
                os.environ["MAILBOX_FIX_ONLY"] = previous


def probe_dynamic_banner(source):
    """The binding closing-only sentence remains in the live prompt banner."""
    daemon = load_daemon(source=source)
    banner = daemon.dispatch_banner(
        store_max=1, newer_in_lane=0, previous_timeout_minutes=None,
        fix_only=True)
    return FIX_ONLY_BANNER in banner


def probe_child_environment(source):
    """A fix-only dispatch exports exactly MAILBOX_FIX_ONLY=1."""
    with scratch_daemon(source=source) as (daemon, _, _, _):
        path = write_pending(
            daemon, "0091-to-fable.md", "close existing work\n")
        launches = []
        outcome, _ = captured_dispatch(daemon, path, True, launches)
        return (outcome and len(launches) == 1
                and launches[0]["env"].get("MAILBOX_FIX_ONLY") == "1")


def probe_watch_propagation(source):
    """main() passes the parsed boolean into the watch backlog pass."""
    with scratch_daemon(source=source) as (daemon, _, _, _):
        rc, _, error, calls = one_watch_pass(daemon, "true")
        return rc == 0 and error == "" and calls == [(False, True)]


def probe_live_watch_binding(source):
    """An external sender reads the held watch's fix-only mode."""
    previous = os.environ.pop("MAILBOX_FIX_ONLY", None)
    try:
        with scratch_daemon(open_count=0, source=source) as (
                daemon, root, mailbox, _):
            try:
                watch_lock = daemon.acquire_dispatch_lock(mode="watch")
                fix_only_lock = daemon.acquire_fix_only_lock()
            except (AttributeError, TypeError):
                return False
            if watch_lock is None or fix_only_lock is None:
                if watch_lock is not None:
                    daemon.release_dispatch_lock(lock_file=watch_lock)
                return False
            try:
                before = tree_snapshot(root)
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    outcome = daemon.send(
                        agent="sol", text="new finding", dry_run=False,
                        ticket_kind="discovery")
                return (not outcome and before == tree_snapshot(root)
                        and "fix-only" in stream.getvalue().lower())
            finally:
                daemon.release_fix_only_lock(lock_file=fix_only_lock)
                daemon.release_dispatch_lock(lock_file=watch_lock)
    finally:
        if previous is not None:
            os.environ["MAILBOX_FIX_ONLY"] = previous


def probe_activation_serialization(source):
    """Activation cannot overtake a sender inside sequence publication."""
    return send_first_activation_race(source=source, verbose=False)


def probe_post_flock_inode_check(source):
    """A replaced public mode sidecar is rejected immediately after flock."""
    return fix_only_path_substitution_is_refused(
        source=source, verbose=False)


def arm_source_mutations():
    """Kill focused mutants for every policy boundary and propagation hop."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    cases = [
        (
            "threshold >= becomes >",
            lambda text: replace_exact(
                text,
                '    if (ticket_kind == "discovery"\n'
                '            and admission_count '
                '>= DISCOVERY_ADMISSION_THRESHOLD):',
                '    if (ticket_kind == "discovery"\n'
                '            and admission_count '
                '> DISCOVERY_ADMISSION_THRESHOLD):'),
            probe_threshold_comparator,
        ),
        (
            "Critical emergency > becomes >=",
            lambda text: replace_exact(
                text,
                '        > SECOND_IMPLEMENTER_CRITICAL_EMERGENCY_THRESHOLD',
                '        >= SECOND_IMPLEMENTER_CRITICAL_EMERGENCY_THRESHOLD'),
            probe_critical_emergency_strict,
        ),
        (
            "High emergency > becomes >=",
            lambda text: replace_exact(
                text,
                '        > SECOND_IMPLEMENTER_HIGH_EMERGENCY_THRESHOLD)',
                '        >= SECOND_IMPLEMENTER_HIGH_EMERGENCY_THRESHOLD)'),
            probe_high_emergency_strict_and_typed,
        ),
        (
            "truthy strip removed",
            lambda text: replace_exact(
                text, "normalized = str(value).strip().lower()",
                "normalized = str(value).lower()"),
            probe_truthy_whitespace,
        ),
        (
            "truthy lower removed",
            lambda text: replace_exact(
                text, "normalized = str(value).strip().lower()",
                "normalized = str(value).strip()"),
            probe_truthy_case,
        ),
        (
            "truthy vocabulary narrowed",
            lambda text: replace_exact(
                text, 'if normalized in {"1", "true", "yes"}:',
                'if normalized == "true":'),
            probe_truthy_vocabulary,
        ),
        (
            "header trailing space accepted",
            mutate_header_trailing_space,
            probe_exact_header_spacing,
        ),
        (
            "later header accepted",
            mutate_later_header,
            probe_first_line_only,
        ),
        (
            "missing kind allowed",
            lambda text: replace_exact(
                text, "if ticket_kind not in SOL_TICKET_KINDS:",
                "if ticket_kind is False:"),
            probe_missing_kind_refusal,
        ),
        (
            "fix-only discovery allowed",
            lambda text: replace_exact(
                text, 'if fix_only and ticket_kind != "closure":',
                'if False and ticket_kind != "closure":'),
            probe_fix_only_discovery_refusal,
        ),
        (
            "Sol envelope hides placeholder",
            lambda text: replace_exact(
                text,
                "    if agent == \"sol\":\n"
                "        placeholder_body = sol_ticket_body(message=message)\n",
                "    if agent == \"sol\":\n"
                "        placeholder_body = message\n"),
            probe_sol_envelope_placeholder,
        ),
        (
            "pre-mkdir refusal removed",
            lambda text: replace_exact(
                text,
                '    reason = refusal_now()\n'
                '    if reason is not None:\n'
                '        print("refused --send " + agent + ": " + reason + ".")',
                '    reason = refusal_now()\n'
                '    if False and reason is not None:\n'
                '        print("refused --send " + agent + ": " + reason + ".")'),
            probe_early_zero_write,
        ),
        (
            "inherited mode ignored",
            lambda text: replace_exact(
                text,
                "value = os.environ.get(FIX_ONLY_ENVIRONMENT)",
                "value = None"),
            probe_inherited_environment,
        ),
        (
            "active watch mode ignored",
            lambda text: replace_exact(
                text,
                "fix_only=(fix_only_environment_active()\n"
                "                      or fix_only_watch_is_active()),",
                "fix_only=fix_only_environment_active(),"),
            probe_live_watch_binding,
        ),
        (
            "activation skips sequence serialization",
            mutate_activation_without_sequence_lock,
            probe_activation_serialization,
        ),
        (
            "post-flock mode inode check removed",
            mutate_drop_post_flock_inode_check,
            probe_post_flock_inode_check,
        ),
        (
            "dynamic banner weakened",
            lambda text: replace_exact(
                text,
                '            "create no discovery tickets or new backlog '
                'lines.")',
                '            "create discovery tickets or new backlog '
                'lines.")'),
            probe_dynamic_banner,
        ),
        (
            "child environment disabled",
            lambda text: replace_exact(
                text, 'env[FIX_ONLY_ENVIRONMENT] = "1"',
                'env[FIX_ONLY_ENVIRONMENT] = "0"'),
            probe_child_environment,
        ),
        (
            "main drops fix-only",
            lambda text: replace_exact(
                text,
                "process_backlog(dry_run=False, fix_only=True)",
                "process_backlog(dry_run=False, fix_only=False)"),
            probe_watch_propagation,
        ),
        (
            "process drops fix-only",
            lambda text: replace_exact(
                text,
                "paths=paths, dry_run=dry_run, fix_only=fix_only)",
                "paths=paths, dry_run=dry_run, fix_only=False)"),
            probe_pipeline_enforcement,
        ),
        (
            "lane drops fix-only",
            lambda text: replace_exact(
                text,
                "dispatch(path=path, dry_run=dry_run, fix_only=fix_only)",
                "dispatch(path=path, dry_run=dry_run, fix_only=False)"),
            probe_pipeline_enforcement,
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
    """Run every isolated runtime, CLI, concurrency, and documentation arm."""
    arms = [
        ("threshold/header", arm_threshold_edges_and_exact_header),
        ("refusal zero-write", arm_refusal_is_zero_write),
        ("second-Implementer emergency",
         arm_second_implementer_emergency_boundaries),
        ("classification", arm_classification_is_explicit_and_fail_closed),
        ("inherited fix-only", arm_inherited_fix_only_send),
        ("live-watch binding", arm_live_watch_binds_external_send),
        ("activation race", arm_activation_publication_is_serialized),
        ("mode path substitution", fix_only_path_substitution_is_refused),
        ("malformed held mode", arm_malformed_held_mode_fails_closed),
        ("queued discovery admission",
         arm_admitted_discovery_does_not_count_itself),
        ("concurrent boundary", arm_concurrent_boundary_is_serialized),
        ("truthy/scope", arm_truthy_values_and_watch_scope),
        ("fix-only dispatch", arm_fix_only_dispatch_and_propagation),
        ("help/README", arm_help_and_readme_parity),
        ("source mutations", arm_source_mutations),
    ]
    failures = []
    for name, arm in arms:
        try:
            passed = arm()
        except Exception as exc:
            print(name + " uncaught=" + type(exc).__name__ + ": " + str(exc))
            passed = False
        print(("PASS " if passed else "FAIL ") + name)
        if not passed:
            failures.append(name)
    print("runtime-summary passed=" + str(len(arms) - len(failures))
          + "/" + str(len(arms)) + " failures=" + repr(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
