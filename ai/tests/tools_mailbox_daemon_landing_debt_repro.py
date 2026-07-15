#!/usr/bin/env python3
"""Scratch-only regression for automatic mailbox landing-debt correction.

Every runtime arm loads a fresh daemon and redirects its worktree, mailbox,
ledger, logs, and child commands into a temporary directory.  No arm reads or
writes the live mailbox, invokes an agent CLI, or changes a real Git branch.
"""

import contextlib
import io
import os
import pathlib
import sys
import tempfile
import threading
import types


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"


class FinishedProcess:
    """A harmless Popen-shaped child which has already succeeded."""

    def __init__(self):
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


class AttributeProxy:
    """Delegate module attributes except for explicit test overrides."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


def load_daemon(source=None):
    """Execute a fresh production module, optionally from mutated source."""
    if source is None:
        source = DAEMON_PATH.read_text(encoding="utf-8")
    module = types.ModuleType("mailbox_daemon_landing_debt_repro")
    module.__file__ = str(DAEMON_PATH)
    exec(compile(source, str(DAEMON_PATH), "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def scratch_daemon(source=None):
    """Yield a daemon whose complete mutable surface is disposable."""
    with tempfile.TemporaryDirectory(prefix="mailbox-landing-debt-") as tmp:
        root = pathlib.Path(tmp)
        ai_root = root / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        mailbox.mkdir(parents=True)
        backlog = ai_root / "notes" / "backlog.md"
        backlog.write_text("", encoding="utf-8")
        sol_lane = root / "sol-lane"
        sol_lane.mkdir()

        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
        daemon.MAILBOX = str(mailbox)
        daemon.DONE = str(mailbox / "done")
        daemon.RELAY_DIR = str(ai_root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(backlog)
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"],
            "opus": ["harmless-opus"],
            "sol": ["harmless-sol"],
        }
        daemon.AGENT_CWD = {
            "fable": str(root),
            "opus": str(root),
            "sol": str(sol_lane),
        }
        daemon.warn_if_mailbox_unwatched = lambda: None
        yield daemon, root, mailbox


def snapshot(lines, stat_text=None):
    """Return one available structured landing-debt measurement."""
    if stat_text is None:
        stat_text = "1 file changed, %d insertions(+)" % lines
    return {
        "available": True,
        "stat": stat_text,
        "changed_lines": lines,
        "returncode": 0,
    }


def unavailable_snapshot(returncode=128):
    """Return one unavailable structured landing-debt measurement."""
    return {
        "available": False,
        "stat": "",
        "changed_lines": 0,
        "returncode": returncode,
    }


def mailbox_messages(mailbox):
    """Return every numbered message below a scratch mailbox."""
    return sorted(
        (path for path in mailbox.rglob("*.md") if path.is_file()),
        key=lambda path: str(path.relative_to(mailbox)))


def root_messages(mailbox):
    """Return numbered messages still pending at the scratch root."""
    return sorted(path for path in mailbox.glob("*.md") if path.is_file())


def fake_popen(calls):
    """Capture the exact prompt and return an already-finished child."""
    def replacement(command, stdout, stderr, cwd, env):
        del stderr, cwd, env
        calls.append(list(command))
        stdout.write("bounded fake child output\n")
        stdout.flush()
        return FinishedProcess()

    return replacement


def arm_fable_grant_and_raw_suffix(source=None):
    """Only Fable receives same-turn landing authority and the hard STOP."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        body = (b"ARCHITECT_HANDOFF: audit the named unit.\r\n"
                b"Preserve this raw suffix.\r\n")
        path = mailbox / "0001-to-fable.md"
        path.write_bytes(body)
        calls = []
        daemon.subprocess.Popen = fake_popen(calls=calls)
        consumed = daemon.dispatch(path=str(path), dry_run=False)
        prompt = calls[0][-1] if len(calls) == 1 else ""
        grant = daemon.agent_preamble(agent="fable")
        other_roles_empty = (
            daemon.agent_preamble(agent="opus") == ""
            and daemon.agent_preamble(agent="sol") == "")
        normalized = " ".join(grant.split())
        return (
            consumed
            and other_roles_empty
            and "ARCHITECT STANDING LANDING GRANT" in grant
            and "records GO" in grant
            and "THIS SAME TURN" in grant
            and "git log main..<branch> --oneline" in normalized
            and "Architect GO is a STOP" in normalized
            and "abort the whole-branch squash" in normalized
            and prompt.endswith(body.decode("utf-8"))
            and prompt.encode("utf-8").endswith(body)
            and prompt.count("ARCHITECT STANDING LANDING GRANT") == 1)


def arm_strict_boundary_and_fable_queue(source=None):
    """Four hundred lines stay quiet; 401 queues one landing-only Fable turn."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(400))
        quiet_at_limit = mailbox_messages(mailbox=mailbox) == []
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        messages = mailbox_messages(mailbox=mailbox)
        if len(messages) != 1:
            return False
        message = messages[0]
        text = message.read_text(encoding="utf-8")
        state = daemon.read_landing_debt_state()
        return (
            quiet_at_limit
            and message.parent == mailbox
            and message.name.endswith("-to-fable.md")
            and "MAILBOX-AUTO: landing-debt-v1 generation=1" in text
            and "LANDING-ONLY ARCHITECT TURN" in text
            and "401 changed content lines" in text
            and "each GO unit as its own squash commit" in text
            and "foreign commit without an Architect GO is a STOP" in text
            and state == {"schema": daemon.LANDING_DEBT_STATE_SCHEMA,
                          "generation": 1, "active": True})


def arm_repeated_and_concurrent_dedup(source=None):
    """Repeated and simultaneous high passes publish one episode message."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        high = snapshot(401)
        for _ in range(4):
            daemon.reconcile_landing_debt_handoff(snapshot=high)
        if len(mailbox_messages(mailbox=mailbox)) != 1:
            return False

    with scratch_daemon(source=source) as (daemon, _, mailbox):
        high = snapshot(401)
        worker_count = 8
        start = threading.Barrier(worker_count)
        errors = []

        def worker():
            try:
                start.wait(timeout=3)
                daemon.reconcile_landing_debt_handoff(snapshot=high)
            except BaseException as exc:  # thread failure is named evidence
                errors.append(exc)

        workers = [threading.Thread(target=worker, daemon=True)
                   for _ in range(worker_count)]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join(timeout=5)
        return (
            not errors
            and all(not worker_thread.is_alive()
                    for worker_thread in workers)
            and len(mailbox_messages(mailbox=mailbox)) == 1)


def arm_done_marker_remains_deduplicated(source=None):
    """An archived marker prevents replay even if generation state is lost."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        high = snapshot(401)
        daemon.reconcile_landing_debt_handoff(snapshot=high)
        first = root_messages(mailbox=mailbox)[0]
        done = mailbox / "done"
        done.mkdir()
        archived = done / first.name
        os.replace(first, archived)
        os.unlink(daemon.landing_debt_state_path())

        daemon.reconcile_landing_debt_handoff(snapshot=high)
        state = daemon.read_landing_debt_state()
        return (
            root_messages(mailbox=mailbox) == []
            and mailbox_messages(mailbox=mailbox) == [archived]
            and state["generation"] == 1
            and state["active"] is True)


def arm_low_rearms_later_episode(source=None):
    """A low pass closes one episode and permits exactly one later generation."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        first = root_messages(mailbox=mailbox)[0]
        done = mailbox / "done"
        done.mkdir()
        os.replace(first, done / first.name)

        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(400))
        rearmed = daemon.read_landing_debt_state()
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(402))
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(999))
        messages = mailbox_messages(mailbox=mailbox)
        bodies = [path.read_text(encoding="utf-8") for path in messages]
        state = daemon.read_landing_debt_state()
        return (
            rearmed["generation"] == 2
            and rearmed["active"] is False
            and len(messages) == 2
            and sum("generation=1" in body for body in bodies) == 1
            and sum("generation=2" in body for body in bodies) == 1
            and state["generation"] == 2
            and state["active"] is True)


def arm_unavailable_demand_has_one_debt_line(source=None):
    """A failed Git probe still gives every demand report one truthful line."""
    with scratch_daemon(source=source) as (daemon, _, _):
        daemon.backlog_ledger_count = lambda: 0
        daemon.landing_debt_snapshot = lambda: unavailable_snapshot(128)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            daemon.report_demand(backlog=[])
        debt_lines = [line for line in output.getvalue().splitlines()
                      if line.startswith("landing debt:")]
        return (
            len(debt_lines) == 1
            and debt_lines[0] == (
                "landing debt: unavailable; git diff --shortstat "
                "main..branch exited 128"))


def hook_contract(source):
    """Prove the sole zero-argument reconciliation call is in the watch loop."""
    hook = "                reconcile_landing_debt_handoff()\n"
    if source.count(hook) != 1:
        return False
    watch_index = source.find("    if args.watch:\n")
    loop_index = source.find("            while True:\n", watch_index)
    hook_index = source.find(hook)
    process_index = source.find("backlog_outcome = process_backlog", hook_index)
    return (
        watch_index >= 0
        and loop_index > watch_index
        and hook_index > loop_index
        and process_index > hook_index
        and hook_index > source.find("    if args.once:\n")
        and hook_index > source.find("    if args.dry_run:\n"))


def arm_watch_only_hook(source=None):
    """The unconditional per-pass hook precedes backlog processing in watch."""
    if source is None:
        source = DAEMON_PATH.read_text(encoding="utf-8")
    return hook_contract(source=source)


def arm_busy_and_idle_pass_semantics(source=None):
    """High debt queues on both empty and already-busy mailbox snapshots."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        idle_ok = (
            len(root_messages(mailbox=mailbox)) == 1
            and root_messages(mailbox=mailbox)[0].name.endswith(
                "-to-fable.md"))

    with scratch_daemon(source=source) as (daemon, _, mailbox):
        ordinary = mailbox / "0007-to-opus.md"
        ordinary_bytes = b"ordinary implementer work stays byte-exact\n"
        ordinary.write_bytes(ordinary_bytes)
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        names = [path.name for path in root_messages(mailbox=mailbox)]
        busy_ok = (
            ordinary.read_bytes() == ordinary_bytes
            and names == ["0007-to-opus.md", "0008-to-fable.md"])
    return idle_ok and busy_ok


def dispatch_peak(daemon, mailbox, agents):
    """Run two wrapper dispatches and return bounded inner concurrency."""
    guard = threading.Lock()
    release = threading.Event()
    start = threading.Barrier(len(agents))
    active = [0]
    peak = [0]
    errors = []
    results = []

    def inner(path, dry_run, fix_only=False, skip_redteam=False):
        del dry_run, fix_only, skip_redteam
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            if active[0] > 1:
                release.set()
        try:
            # If the wrapper serialized both roles, the first turn waits only
            # this bounded interval and then releases the shared lock. If they
            # may overlap, the second entrant releases both immediately.
            release.wait(timeout=0.25)
            release.set()
            return pathlib.Path(path).name
        finally:
            with guard:
                active[0] -= 1

    daemon.dispatch_under_main_checkout_lock = inner

    def invoke(index, agent):
        try:
            start.wait(timeout=2)
            path = mailbox / ("%04d-to-%s.md" % (index + 1, agent))
            results.append(daemon.dispatch(path=str(path), dry_run=False))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=invoke, args=(index, agent), daemon=True)
               for index, agent in enumerate(agents)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
    return {
        "peak": peak[0],
        "results": results,
        "errors": errors,
        "alive": any(worker.is_alive() for worker in workers),
    }


def arm_main_checkout_role_mutex(source=None):
    """Fable and Sol serialize, while Opus remains parallel with Sol."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        protected = dispatch_peak(
            daemon=daemon, mailbox=mailbox, agents=("fable", "sol"))
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        parallel = dispatch_peak(
            daemon=daemon, mailbox=mailbox, agents=("opus", "sol"))
    return (
        not protected["alive"]
        and not protected["errors"]
        and len(protected["results"]) == 2
        and protected["peak"] == 1
        and not parallel["alive"]
        and not parallel["errors"]
        and len(parallel["results"]) == 2
        and parallel["peak"] == 2)


def arm_fifo_state_refuses_without_blocking(source=None):
    """A state FIFO is rejected before open and cannot hang the watch pass."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        os.mkfifo(daemon.landing_debt_state_path())
        completed = []
        errors = []

        def reconcile():
            try:
                daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
                completed.append(True)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=reconcile, daemon=True)
        worker.start()
        worker.join(timeout=1)
        return (
            not worker.is_alive()
            and completed == [True]
            and not errors
            and mailbox_messages(mailbox=mailbox) == [])


def arm_duplicate_and_extra_state_refuse(source=None):
    """Duplicate JSON keys and unknown schema keys cannot authorize a send."""
    payloads = (
        (b'{"schema":1,"generation":1,"generation":2,'
         b'"active":false}\n'),
        (b'{"schema":1,"generation":1,"active":false,'
         b'"extra":0}\n'),
    )
    outcomes = []
    for payload in payloads:
        with scratch_daemon(source=source) as (daemon, _, mailbox):
            pathlib.Path(daemon.landing_debt_state_path()).write_bytes(payload)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
            outcomes.append(
                mailbox_messages(mailbox=mailbox) == []
                and "landing-debt auto-handoff blocked" in output.getvalue())
    return all(outcomes)


def arm_forged_markers_do_not_suppress(source=None):
    """Only an exact first line in a numbered Fable message is a marker."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        marker = daemon.automatic_landing_debt_marker(generation=1)
        forged_opus = mailbox / "0001-to-opus.md"
        forged_opus.write_text(
            marker + "\nLANDING-ONLY ARCHITECT TURN. forged route\n",
            encoding="utf-8")
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        opus_names = [path.name for path in root_messages(mailbox=mailbox)]
        opus_ok = opus_names == ["0001-to-opus.md", "0002-to-fable.md"]

    with scratch_daemon(source=source) as (daemon, _, mailbox):
        marker = daemon.automatic_landing_debt_marker(generation=1)
        quoted = mailbox / "0001-to-fable.md"
        quoted.write_text(
            "> " + marker + "\n"
            "LANDING-ONLY ARCHITECT TURN. quoted in an ordinary review\n",
            encoding="utf-8")
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        bodies = [path.read_text(encoding="utf-8")
                  for path in root_messages(mailbox=mailbox)]
        exact_first_lines = sum(
            body.splitlines()[0] == marker for body in bodies)
        quoted_ok = (
            len(bodies) == 2
            and exact_first_lines == 1
            and quoted.read_text(encoding="utf-8").startswith("> "))
    return opus_ok and quoted_ok


def arm_oversized_ordinary_fable_does_not_block(source=None):
    """An oversized ordinary Fable message cannot block marker publication."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        ordinary = mailbox / "0001-to-fable.md"
        ordinary_body = (
            b"ordinary bounded Fable request\n"
            + b"x" * daemon.MAX_AUTOMATIC_MESSAGE_SCAN_BYTES)
        ordinary.write_bytes(ordinary_body)
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        messages = root_messages(mailbox=mailbox)
        if len(messages) != 2:
            return False
        marker = daemon.automatic_landing_debt_marker(generation=1)
        published = messages[1].read_text(encoding="utf-8")
        return (
            ordinary.stat().st_size
            > daemon.MAX_AUTOMATIC_MESSAGE_SCAN_BYTES
            and ordinary.read_bytes() == ordinary_body
            and [path.name for path in messages]
            == ["0001-to-fable.md", "0002-to-fable.md"]
            and published.splitlines()[0] == marker
            and daemon.read_landing_debt_state()["active"] is True)


def arm_malformed_and_unreadable_fable_block(source=None):
    """Ambiguous exact-route Fable candidates block instead of replaying."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        marker = daemon.automatic_landing_debt_marker(generation=1)
        malformed = mailbox / "0001-to-fable.md"
        malformed.write_text(
            marker + "\nordinary body under a reserved marker\n",
            encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        malformed_ok = (
            mailbox_messages(mailbox=mailbox) == [malformed]
            and "invalid body" in output.getvalue())

    with scratch_daemon(source=source) as (daemon, _, mailbox):
        unreadable = mailbox / "0001-to-fable.md"
        unreadable.write_text("ordinary Fable work\n", encoding="utf-8")
        real_os = daemon.os

        def denied_open(path, flags, *args):
            if os.path.abspath(path) == os.path.abspath(str(unreadable)):
                raise PermissionError("injected unreadable Fable candidate")
            return real_os.open(path, flags, *args)

        daemon.os = AttributeProxy(real_os, open=denied_open)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        unreadable_ok = (
            mailbox_messages(mailbox=mailbox) == [unreadable]
            and "blocked" in output.getvalue()
            and "cannot open Fable message" in output.getvalue())
    return malformed_ok and unreadable_ok


def arm_post_publish_state_failure_recovers(source=None):
    """A linked marker survives state failure and does not poison generation 2."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        real_write = daemon.write_landing_debt_state

        def fail_state_write(state):
            del state
            raise OSError("injected post-publish state failure")

        daemon.write_landing_debt_state = fail_state_write
        raised = None
        try:
            daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        except OSError as exc:
            raised = exc
        first_messages = mailbox_messages(mailbox=mailbox)
        failed_safely = (
            raised is not None
            and len(first_messages) == 1
            and first_messages[0].name.endswith("-to-fable.md")
            and not os.path.exists(daemon.landing_debt_state_path()))

        daemon.write_landing_debt_state = real_write
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(401))
        recovered = (
            len(mailbox_messages(mailbox=mailbox)) == 1
            and daemon.read_landing_debt_state()["active"] is True)
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(400))
        rearmed = daemon.read_landing_debt_state()
        daemon.reconcile_landing_debt_handoff(snapshot=snapshot(402))
        bodies = [path.read_text(encoding="utf-8")
                  for path in mailbox_messages(mailbox=mailbox)]
        later_episode = (
            rearmed["generation"] == 2
            and rearmed["active"] is False
            and len(bodies) == 2
            and sum("generation=1" in body for body in bodies) == 1
            and sum("generation=2" in body for body in bodies) == 1)
        return failed_safely and recovered and later_episode


def apply_replacements(source, replacements):
    """Apply exact source replacements, refusing ambiguous mutation anchors."""
    mutated = source
    for old, new, expected in replacements:
        count = mutated.count(old)
        if count != expected:
            raise ValueError(
                "mutation anchor count %d, expected %d for %r"
                % (count, expected, old))
        mutated = mutated.replace(old, new)
    return mutated


def mutation_killed(label, replacements, probe):
    """Return whether one named source mutation turns its witness red."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    try:
        mutant = apply_replacements(source=source, replacements=replacements)
    except ValueError as exc:
        print("MUTATION " + label + " INVALID (" + str(exc) + ")")
        return False
    try:
        compile(mutant, str(DAEMON_PATH), "exec")
    except SyntaxError as exc:
        print("MUTATION " + label + " INVALID (SyntaxError: "
              + str(exc) + ")")
        return False
    try:
        survived = bool(probe(source=mutant))
        detail = "probe returned " + str(survived)
    except BaseException as exc:
        survived = False
        detail = type(exc).__name__ + ": " + str(exc)
    killed = not survived
    print("MUTATION " + label + " " + ("RED" if killed else "SURVIVED")
          + " (" + detail + ")")
    return killed


def arm_source_mutations():
    """Kill each named production boundary independently."""
    dedup_block = (
        "        if state[\"active\"]:\n"
        "            return snapshot\n"
        "        try:\n"
        "            marker_exists = automatic_landing_debt_message_exists(\n"
        "                generation=state[\"generation\"])\n"
        "        except (OSError, ValueError) as exc:\n"
        "            print(\"landing-debt auto-handoff blocked: \" + str(exc)\n"
        "                  + \".\")\n"
        "            return snapshot\n"
        "        if marker_exists:\n"
        "            # Crash recovery can observe a linked marker from a publisher\n"
        "            # that died before its directory fsync. Make that marker\n"
        "            # durable before state is allowed to suppress a replay.\n"
        "            fsync_directory(directory=MAILBOX)\n"
        "            state[\"active\"] = True\n"
        "            write_landing_debt_state(state=state)\n"
        "            return snapshot\n")
    mutations = [
        (
            "strict-past-400-boundary",
            [("if snapshot[\"changed_lines\"] <= LANDING_DEBT_LINE_LIMIT:",
              "if snapshot[\"changed_lines\"] < LANDING_DEBT_LINE_LIMIT:",
              1)],
            arm_strict_boundary_and_fable_queue,
        ),
        (
            "automatic-recipient-is-fable",
            [("publish_message_locked(agent=\"fable\", payload=payload)",
              "publish_message_locked(agent=\"opus\", payload=payload)",
              1)],
            arm_strict_boundary_and_fable_queue,
        ),
        (
            "foreign-commit-stop",
            [("is a STOP: abort", "is advisory: continue", 1),
             ("is a STOP, so abort", "is advisory, so continue", 1)],
            arm_fable_grant_and_raw_suffix,
        ),
        (
            "demand-calls-debt-report",
            [("    report_landing_debt()\n\n\ndef landing_debt_snapshot():",
              "    pass\n\n\ndef landing_debt_snapshot():", 1)],
            arm_unavailable_demand_has_one_debt_line,
        ),
        (
            "watch-loop-hook",
            [("                reconcile_landing_debt_handoff()\n",
              "                pass\n", 1)],
            arm_watch_only_hook,
        ),
        (
            "episode-deduplication",
            [(dedup_block, "", 1)],
            arm_repeated_and_concurrent_dedup,
        ),
        (
            "fable-sol-main-checkout-mutex",
            [("if dry_run or agent == \"opus\":",
              "if dry_run or agent in (\"opus\", \"sol\"):", 1)],
            arm_main_checkout_role_mutex,
        ),
        (
            "recovery-recipient-exactness",
            [(r'r"\d+[a-z]?-to-fable\.md"',
              r'r"\d+[a-z]?-to-(?:fable|opus)\.md"', 1)],
            arm_forged_markers_do_not_suppress,
        ),
        (
            "recovery-first-line-exactness",
            [("if prefix not in (marker, marker + b\"\\n\"):",
              "if False:", 1),
             ("if lines and lines[0] == marker:",
              "if lines and marker in raw:", 1)],
            arm_forged_markers_do_not_suppress,
        ),
        (
            "recovery-prefix-read-remains-bounded",
            [("                complete=False)",
              "                complete=True)", 1)],
            arm_oversized_ordinary_fable_does_not_block,
        ),
    ]
    outcomes = [mutation_killed(label, replacements, probe)
                for label, replacements, probe in mutations]
    print("MUTATION SUMMARY killed=%d/%d" % (sum(outcomes), len(outcomes)))
    return all(outcomes)


def main():
    """Run all bounded runtime arms and the named source mutations."""
    arms = [
        ("fable-only-grant-and-raw-suffix",
         arm_fable_grant_and_raw_suffix),
        ("strict-boundary-and-fable-queue",
         arm_strict_boundary_and_fable_queue),
        ("repeated-and-concurrent-dedup",
         arm_repeated_and_concurrent_dedup),
        ("done-marker-dedup", arm_done_marker_remains_deduplicated),
        ("low-rearms-later-episode", arm_low_rearms_later_episode),
        ("unavailable-demand-one-line",
         arm_unavailable_demand_has_one_debt_line),
        ("watch-only-hook", arm_watch_only_hook),
        ("busy-and-idle-pass-semantics", arm_busy_and_idle_pass_semantics),
        ("fable-sol-main-checkout-mutex", arm_main_checkout_role_mutex),
        ("fifo-state-immediate-refusal",
         arm_fifo_state_refuses_without_blocking),
        ("duplicate-and-extra-state-refusal",
         arm_duplicate_and_extra_state_refuse),
        ("forged-markers-do-not-suppress",
         arm_forged_markers_do_not_suppress),
        ("oversized-ordinary-fable-does-not-block",
         arm_oversized_ordinary_fable_does_not_block),
        ("malformed-and-unreadable-fable-block",
         arm_malformed_and_unreadable_fable_block),
        ("post-publish-state-failure-recovers",
         arm_post_publish_state_failure_recovers),
    ]
    outcomes = []
    for label, arm in arms:
        try:
            passed = bool(arm())
            detail = ""
        except BaseException as exc:
            passed = False
            detail = " (" + type(exc).__name__ + ": " + str(exc) + ")"
        outcomes.append(passed)
        print("ARM " + label + " " + ("PASS" if passed else "FAIL")
              + detail)
    print("ARM SUMMARY passed=%d/%d" % (sum(outcomes), len(outcomes)))
    mutations_green = arm_source_mutations()
    all_green = all(outcomes) and mutations_green
    print("SUMMARY " + ("PASS" if all_green else "FAIL"))
    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
