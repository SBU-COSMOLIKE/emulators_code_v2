#!/usr/bin/env python3
"""Scratch-only witnesses for daemon currency, retry, and archive safety.

The exercises in this file never point at ``notes/mailbox`` in the checkout.
Each one imports a fresh daemon module and rebinds every state path to a
temporary directory before calling a daemon entry point.
"""

import contextlib
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = ROOT / "tools" / "mailbox_daemon.py"


def load_daemon(path=DAEMON_PATH):
    """Load a fresh daemon module for one isolated reproduction."""
    spec = importlib.util.spec_from_file_location(
        "mailbox_daemon_staleness_repro", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def scratch_daemon(daemon_path=DAEMON_PATH):
    """Yield a daemon whose worktree, mailbox, ledger, and logs are scratch."""
    with tempfile.TemporaryDirectory(
            prefix="mailbox-daemon-staleness-") as tmp:
        root = pathlib.Path(tmp)
        daemon = load_daemon(path=daemon_path)
        daemon.WORKTREE = str(root)
        daemon.REPO_ROOT = str(root / "sol-worktree")
        daemon.MAILBOX = str(root / "notes" / "mailbox")
        daemon.DONE = str(root / "notes" / "mailbox" / "done")
        daemon.RELAY_DIR = str(root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(root / "notes" / "backlog.md")
        daemon.PREAMBLE = "SCRATCH PREAMBLE\n--- MESSAGE ---\n"
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"],
            "opus": ["harmless-opus"],
            "sol": ["harmless-sol"],
        }
        shared = str(root / "shared-worktree")
        daemon.AGENT_CWD = {
            "fable": shared,
            "opus": shared,
            "sol": str(root / "sol-worktree"),
        }
        os.makedirs(daemon.MAILBOX, exist_ok=True)
        os.makedirs(shared, exist_ok=True)
        os.makedirs(daemon.AGENT_CWD["sol"], exist_ok=True)
        yield daemon, root


def write_message(daemon, name, body, directory=None):
    """Write one UTF-8 scratch mailbox artifact and return its path."""
    target_dir = pathlib.Path(directory or daemon.MAILBOX)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    path.write_text(body, encoding="utf-8")
    return str(path)


class FinishedProcess:
    """A tiny Popen-shaped process which has already exited."""

    def __init__(self, returncode=0):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


def success_popen(calls, inspect=None, output="live stub output\n"):
    """Return a harmless Popen replacement that captures the full call."""
    def fake_popen(command, stdout, stderr, cwd, env):
        call = {
            "command": list(command),
            "stdout_name": stdout.name,
            "stderr": stderr,
            "cwd": cwd,
            "env": dict(env),
        }
        calls.append(call)
        if inspect is not None:
            inspect(call)
        stdout.write(output)
        stdout.flush()
        return FinishedProcess(returncode=0)

    return fake_popen


def failed_popen(calls, returncode=1):
    """Return a harmless Popen replacement with an ordinary nonzero rc."""
    def fake_popen(command, stdout, stderr, cwd, env):
        del stderr, cwd, env
        calls.append(list(command))
        stdout.write("ordinary child failure\n")
        stdout.flush()
        return FinishedProcess(returncode=returncode)

    return fake_popen


def tree_snapshot(root):
    """Return content plus metadata for every path below ``root``."""
    root = pathlib.Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        relative = str(path.relative_to(root))
        entry = {
            "kind": "dir" if path.is_dir() else "file",
            "mode": stat.st_mode,
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
        if path.is_file():
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result[relative] = entry
    return result


def nested_has_number(value, expected):
    """Return whether decoded JSON contains ``expected`` as a scalar."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == expected
    if isinstance(value, str):
        return value == str(expected)
    if isinstance(value, list):
        return any(nested_has_number(item, expected) for item in value)
    if isinstance(value, dict):
        return any(nested_has_number(item, expected)
                   for item in value.values())
    return False


def arm_post_claim_currency_banner():
    """Prove currency is sampled after claim and remains advisory only."""
    with scratch_daemon() as (daemon, _):
        body = "RAW BODY TOKEN: inspect notes and do the work.\n"
        path = write_message(daemon, "0042-to-fable.md", body)
        original_claim = daemon.claim_message

        def claim_then_publish(path):
            claimed = original_claim(path=path)
            if claimed is None:
                return None
            # These artifacts appear only after the current message is no
            # longer pending. The recursive store maximum is the to-user
            # file in hold/. Only the two root-pending Fable/Opus messages
            # share the current dispatch's AGENT_CWD; Sol is another lane.
            write_message(daemon, "9999-to-sol.md", "old archive\n",
                          directory=daemon.DONE)
            write_message(daemon, "10000-to-user.md", "human note\n",
                          directory=os.path.join(daemon.MAILBOX, "hold"))
            write_message(daemon, "0043-to-opus.md", "newer shared one\n")
            write_message(daemon, "0044-to-fable.md", "newer shared two\n")
            write_message(daemon, "0045-to-sol.md", "newer other lane\n")
            return claimed

        calls = []
        daemon.claim_message = claim_then_publish
        daemon.subprocess.Popen = success_popen(calls=calls)
        result = daemon.dispatch(path=path, dry_run=False)
        prompt = calls[0]["command"][-1] if len(calls) == 1 else ""
        prefix = prompt[:-len(body)] if prompt.endswith(body) else ""
        lower = prefix.lower()
        maximum_visible = (
            "currency" in lower
            and "sequence" in lower
            and "10000" in prefix)
        newer_visible = (
            "newer" in lower
            and ("queued" in lower or "root-pending" in lower)
            and re.search(r"(?:newer[^\n]*\b2\b|\b2\b[^\n]*newer)",
                          lower) is not None)
        banner_index = lower.find("dispatch currency")
        preamble_index = prompt.find(daemon.PREAMBLE)
        delimiter_index = prompt.find("--- MESSAGE ---", preamble_index)
        body_index = len(prompt) - len(body)
        exact_envelope = (
            0 <= banner_index < preamble_index < delimiter_index < body_index
            and prompt.count(daemon.PREAMBLE) == 1
            and prompt.endswith(daemon.PREAMBLE + body)
            and prompt.endswith(body)
            and prompt.count(body) == 1)
        print("post-claim result=" + str(result)
              + " calls=" + str(len(calls))
              + " max10000=" + str(maximum_visible)
              + " newer2=" + str(newer_visible)
              + " raw_suffix=" + str(exact_envelope))
        return (result and len(calls) == 1 and maximum_visible
                and newer_visible and exact_envelope)


def arm_crlf_body_is_exact_suffix(daemon_path=DAEMON_PATH):
    """Preserve valid CRLF newlines in the launched prompt's body suffix."""
    with scratch_daemon(daemon_path=daemon_path) as (daemon, _):
        name = "0046-to-opus.md"
        raw_body = b"alpha\r\nbeta\r\n"
        path = pathlib.Path(daemon.MAILBOX) / name
        path.write_bytes(raw_body)
        calls = []
        daemon.subprocess.Popen = success_popen(calls=calls)
        result = daemon.dispatch(path=str(path), dry_run=False)
        prompt = calls[0]["command"][-1] if len(calls) == 1 else ""
        decoded_body = raw_body.decode("utf-8")
        exact = (result and len(calls) == 1
                 and prompt.endswith(daemon.PREAMBLE + decoded_body)
                 and prompt.encode("utf-8").endswith(raw_body))
        print("crlf-body result=" + str(result)
              + " calls=" + str(len(calls))
              + " exact_suffix=" + str(exact))
        return exact


def arm_timeout_history_and_retry_hint(daemon_path=DAEMON_PATH):
    """Prove a timeout is atomically recorded and described on retry."""
    with scratch_daemon(daemon_path=daemon_path) as (daemon, _):
        name = "0050-to-opus.md"
        body = "TIMEOUT RETRY BODY\n"
        path = write_message(daemon, name, body)
        daemon.DISPATCH_TIMEOUT_MINUTES = 7

        class HungProcess:
            """Stay alive until the daemon's timeout guard kills us."""

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def kill(self):
                self.returncode = -9

            def wait(self):
                return self.returncode

        class TimeoutClock:
            """Advance past the seven-minute deadline without sleeping."""

            def __init__(self):
                self.now = 0.0

            def time(self):
                return self.now

            def sleep(self, _seconds):
                self.now = 7.0 * 60.0

        def hung_popen(command, stdout, stderr, cwd, env):
            del command, stderr, cwd, env
            stdout.write("child appears hung\n")
            stdout.flush()
            return HungProcess()

        atomic_observations = []
        original_replace = daemon.os.replace

        def observing_replace(source, destination):
            destination_path = pathlib.Path(destination)
            if (destination_path.name == name + ".json"
                    and destination_path.parent.name == ".dispatch-history"):
                source_path = pathlib.Path(source)
                decoded = json.loads(source_path.read_text(encoding="utf-8"))
                atomic_observations.append(
                    source_path != destination_path
                    and source_path.parent == destination_path.parent
                    and nested_has_number(decoded, 7))
            return original_replace(source, destination)

        daemon.time = TimeoutClock()
        daemon.subprocess.Popen = hung_popen
        daemon.os.replace = observing_replace
        try:
            timed_out = daemon.dispatch(path=path, dry_run=False)
        finally:
            daemon.os.replace = original_replace

        history_path = (pathlib.Path(daemon.MAILBOX)
                        / ".dispatch-history" / (name + ".json"))
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            valid_history = nested_has_number(history, 7)
        except (OSError, ValueError):
            valid_history = False
        failed_path = pathlib.Path(daemon.MAILBOX) / "failed" / name
        timeout_parked = failed_path.is_file()

        # Requeue the same body and let the retry finish. The history is a
        # prompt hint, never a reason to suppress the launch.
        retry_path = pathlib.Path(daemon.MAILBOX) / name
        if timeout_parked:
            failed_path.replace(retry_path)
        calls = []
        daemon.time = __import__("time")
        daemon.subprocess.Popen = success_popen(calls=calls)
        retried = daemon.dispatch(path=str(retry_path), dry_run=False)
        prompt = calls[0]["command"][-1] if len(calls) == 1 else ""
        exact_hint = (
            "this dispatch previously ran for 7 minutes and was killed"
            in prompt)
        suffix_intact = prompt.endswith(body) and prompt.count(body) == 1
        print("timeout result=" + str(timed_out)
              + " atomic=" + str(atomic_observations)
              + " history=" + str(valid_history)
              + " parked=" + str(timeout_parked)
              + " retry_result=" + str(retried)
              + " retry_calls=" + str(len(calls))
              + " exact_hint=" + str(exact_hint)
              + " raw_suffix=" + str(suffix_intact))
        return (not timed_out and atomic_observations == [True]
                and valid_history and timeout_parked and retried
                and len(calls) == 1 and exact_hint and suffix_intact)


def arm_ordinary_failure_has_no_timeout_history():
    """Prove an ordinary rc=1 is parked without creating timeout state."""
    with scratch_daemon() as (daemon, _):
        name = "0051-to-opus.md"
        path = write_message(daemon, name, "ordinary failing body\n")
        calls = []
        daemon.subprocess.Popen = failed_popen(calls=calls, returncode=1)
        result = daemon.dispatch(path=path, dry_run=False)
        history_dir = pathlib.Path(daemon.MAILBOX) / ".dispatch-history"
        failed = pathlib.Path(daemon.MAILBOX) / "failed" / name
        print("ordinary-rc1 result=" + str(result)
              + " calls=" + str(len(calls))
              + " history_dir=" + str(history_dir.exists())
              + " parked=" + str(failed.is_file()))
        return (not result and len(calls) == 1
                and not history_dir.exists() and failed.is_file())


def arm_dry_run_is_strictly_read_only():
    """Prove dry-run changes no path, bytes, directory, or mtime."""
    with scratch_daemon() as (daemon, root):
        write_message(daemon, "0060-to-fable.md", "dry-run body\n")
        write_message(daemon, "10000-to-user.md", "held human body\n",
                      directory=os.path.join(daemon.MAILBOX, "hold"))
        history_dir = pathlib.Path(daemon.MAILBOX) / ".dispatch-history"
        history_dir.mkdir()
        (history_dir / "old.json").write_text(
            '{"timeout_minutes": 3}\n', encoding="utf-8")
        daemon.report_demand = lambda backlog: None
        before = tree_snapshot(root=root)
        result = daemon.process_backlog(dry_run=True)
        after = tree_snapshot(root=root)
        print("dry-run result=" + str(result)
              + " tree_equal=" + str(before == after)
              + " entries=" + str(len(before)))
        return result and before == after


def arm_archive_failure_propagates(daemon_path=DAEMON_PATH):
    """Prove dispatch and backlog processing fail when done/ cannot own it."""
    direct_ok = False
    process_ok = False

    with scratch_daemon(daemon_path=daemon_path) as (daemon, _):
        name = "0070-to-fable.md"
        path = write_message(daemon, name, "direct archive collision\n")
        write_message(daemon, name, "historical owner\n",
                      directory=daemon.DONE)
        calls = []
        daemon.subprocess.Popen = success_popen(calls=calls)
        result = daemon.dispatch(path=path, dry_run=False)
        inflight = pathlib.Path(daemon.MAILBOX) / "inflight" / name
        direct_ok = not result and len(calls) == 1 and inflight.is_file()
        print("archive-direct result=" + str(result)
              + " calls=" + str(len(calls))
              + " inflight=" + str(inflight.is_file()))

    with scratch_daemon(daemon_path=daemon_path) as (daemon, _):
        name = "0071-to-opus.md"
        write_message(daemon, name, "backlog archive collision\n")
        write_message(daemon, name, "historical owner\n",
                      directory=daemon.DONE)
        calls = []
        daemon.subprocess.Popen = success_popen(calls=calls)
        daemon.report_demand = lambda backlog: None
        result = daemon.process_backlog(dry_run=False)
        inflight = pathlib.Path(daemon.MAILBOX) / "inflight" / name
        process_ok = not result and len(calls) == 1 and inflight.is_file()
        print("archive-process result=" + str(result)
              + " calls=" + str(len(calls))
              + " inflight=" + str(inflight.is_file()))

    return direct_ok and process_ok


def arm_same_cwd_serializes_through_archive_and_logs():
    """Prove the shared Fable/Opus lane archives before its next launch."""
    with scratch_daemon() as (daemon, _):
        first_name = "0080-to-fable.md"
        second_name = "0081-to-opus.md"
        first_body = "FIRST SAME-LANE BODY\n"
        second_body = "SECOND SAME-LANE BODY\n"
        write_message(daemon, first_name, first_body)
        write_message(daemon, second_name, second_body)
        calls = []
        second_saw_first_done = []

        def inspect(call):
            prompt = call["command"][-1]
            if prompt.endswith(second_body):
                first_done = pathlib.Path(daemon.DONE) / first_name
                first_inflight = (pathlib.Path(daemon.MAILBOX)
                                  / "inflight" / first_name)
                second_saw_first_done.append(
                    first_done.is_file() and not first_inflight.exists())

        daemon.subprocess.Popen = success_popen(
            calls=calls, inspect=inspect, output="streamed live output\n")
        daemon.report_demand = lambda backlog: None
        result = daemon.process_backlog(dry_run=False)
        prompts = [call["command"][-1] for call in calls]
        order_ok = (
            len(prompts) == 2
            and prompts[0].endswith(first_body)
            and prompts[1].endswith(second_body))
        popen_shape_ok = all(
            call["stderr"] is subprocess.STDOUT
            and pathlib.Path(call["stdout_name"]).parent
            == pathlib.Path(daemon.RELAY_DIR)
            and call["cwd"] == daemon.AGENT_CWD[
                "fable" if index == 0 else "opus"]
            for index, call in enumerate(calls))
        logs = list(pathlib.Path(daemon.RELAY_DIR).glob("*.log"))
        logs_ok = (
            len(logs) == 2
            and all("streamed live output" in log.read_text(encoding="utf-8")
                    and "--- rc=0 ---" in log.read_text(encoding="utf-8")
                    for log in logs))
        both_done = all(
            (pathlib.Path(daemon.DONE) / name).is_file()
            for name in [first_name, second_name])
        print("same-cwd result=" + str(result)
              + " order=" + str(order_ok)
              + " archived_before_second=" + repr(second_saw_first_done)
              + " popen_shape=" + str(popen_shape_ok)
              + " logs=" + str(logs_ok)
              + " both_done=" + str(both_done))
        return (result and order_ok and second_saw_first_done == [True]
                and popen_shape_ok and logs_ok and both_done)


def currency_contract_probe(daemon_path=DAEMON_PATH):
    """Return whether one snapshot sees recursive max and CWD-lane count."""
    with scratch_daemon(daemon_path=daemon_path) as (daemon, _):
        inflight = pathlib.Path(daemon.MAILBOX) / "inflight"
        current = write_message(
            daemon, "0042-to-fable.md", "claimed\n", directory=inflight)
        write_message(daemon, "9999-to-sol.md", "archive\n",
                      directory=daemon.DONE)
        write_message(daemon, "10000-to-user.md", "human hold\n",
                      directory=os.path.join(daemon.MAILBOX, "hold"))
        write_message(daemon, "0043-to-opus.md", "same cwd\n")
        write_message(daemon, "0044-to-fable.md", "same cwd\n")
        write_message(daemon, "0045-to-sol.md", "other cwd\n")
        return daemon.dispatch_currency(
            dispatch_path=current, agent="fable") == (10000, 2)


def arm_archive_requires_source_absence(daemon_path=DAEMON_PATH):
    """Reject a claimed message that appears copied, not moved, to done/."""
    with scratch_daemon(daemon_path=daemon_path) as (daemon, _):
        inflight_dir = pathlib.Path(daemon.MAILBOX) / "inflight"
        source = write_message(
            daemon, "0085-to-opus.md", "still claimed\n",
            directory=inflight_dir)
        done = str(pathlib.Path(daemon.DONE) / "0085-to-opus.md")

        def linked_not_moved(path, directory):
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            os.link(path, done)
            return done

        daemon.move_without_overwrite = linked_not_moved
        result = daemon.archive_consumed_message(dispatch_path=source)
        source_still_exists = pathlib.Path(source).exists()
        print("archive-source-absence result=" + str(result)
              + " source_still_exists=" + str(source_still_exists))
        return not result and source_still_exists


def arm_lane_stops_after_unconsumed_head(daemon_path=DAEMON_PATH):
    """Prove an archive collision prevents a later shared-CWD launch."""
    with scratch_daemon(daemon_path=daemon_path) as (daemon, _):
        first = "0090-to-fable.md"
        second = "0091-to-opus.md"
        write_message(daemon, first, "unconsumed lane head\n")
        write_message(daemon, second, "must remain queued\n")
        write_message(daemon, first, "historical done owner\n",
                      directory=daemon.DONE)
        calls = []
        daemon.subprocess.Popen = success_popen(calls=calls)
        daemon.report_demand = lambda backlog: None
        result = daemon.process_backlog(dry_run=False)
        second_pending = pathlib.Path(daemon.MAILBOX) / second
        second_done = pathlib.Path(daemon.DONE) / second
        print("lane-head result=" + str(result)
              + " launches=" + str(len(calls))
              + " second_pending=" + str(second_pending.is_file())
              + " second_done=" + str(second_done.exists()))
        return (not result and len(calls) == 1
                and second_pending.is_file() and not second_done.exists())


def arm_cross_pass_inflight_blocks_only_its_cwd():
    """Let Sol drain while a prior Fable claim blocks Fable and Opus."""
    with scratch_daemon() as (daemon, _):
        inflight_name = "0100-to-fable.md"
        shared_name = "0101-to-opus.md"
        sol_name = "0102-to-sol.md"
        shared_body = "SHARED LANE MUST WAIT\n"
        sol_body = ("MAILBOX-TICKET: closure\n\n"
                    "INDEPENDENT SOL MAY RUN\n")
        write_message(daemon, inflight_name, "unresolved prior turn\n",
                      directory=os.path.join(daemon.MAILBOX, "inflight"))
        write_message(daemon, shared_name, shared_body)
        write_message(daemon, sol_name, sol_body)
        calls = []
        daemon.subprocess.Popen = success_popen(calls=calls)
        daemon.report_demand = lambda backlog: None
        result = daemon.process_backlog(dry_run=False)
        prompts = [call["command"][-1] for call in calls]
        shared_pending = pathlib.Path(daemon.MAILBOX) / shared_name
        sol_done = pathlib.Path(daemon.DONE) / sol_name
        print("cross-pass result=" + str(result)
              + " launches=" + str(len(calls))
              + " shared_pending=" + str(shared_pending.is_file())
              + " sol_done=" + str(sol_done.is_file()))
        return (not result and len(prompts) == 1
                and prompts[0].endswith(sol_body)
                and not prompts[0].endswith(shared_body)
                and shared_pending.is_file() and sol_done.is_file())


def arm_only_inflight_reports_failure():
    """Treat unresolved inflight work as failure even without root pending."""
    with scratch_daemon() as (daemon, _):
        name = "0103-to-opus.md"
        inflight = write_message(
            daemon, name, "only unresolved state\n",
            directory=os.path.join(daemon.MAILBOX, "inflight"))
        calls = []
        daemon.subprocess.Popen = success_popen(calls=calls)
        daemon.report_demand = lambda backlog: None
        result = daemon.process_backlog(dry_run=False)
        print("only-inflight result=" + str(result)
              + " launches=" + str(len(calls))
              + " inflight=" + str(pathlib.Path(inflight).is_file()))
        return (result is False and not calls
                and pathlib.Path(inflight).is_file())


def arm_timeout_value_validation():
    """Reject invalid CLI/direct timeouts after claim and before launch."""
    validator_daemon = load_daemon()
    cli_rejected = []
    for value in ["0", "-1",
                  str(validator_daemon.MAX_DISPATCH_TIMEOUT_MINUTES + 1)]:
        try:
            validator_daemon.positive_int(value)
            cli_rejected.append(False)
        except validator_daemon.argparse.ArgumentTypeError:
            cli_rejected.append(True)
    cli_positive = validator_daemon.positive_int("1") == 1

    direct_results = []
    invalid_direct = [
        0,
        -3,
        validator_daemon.MAX_DISPATCH_TIMEOUT_MINUTES + 1,
    ]
    for index, value in enumerate(invalid_direct, start=1):
        with scratch_daemon() as (daemon, _):
            daemon.DISPATCH_TIMEOUT_MINUTES = value
            name = "011%d-to-opus.md" % index
            path = write_message(daemon, name, "invalid direct timeout\n")
            calls = []
            daemon.subprocess.Popen = success_popen(calls=calls)
            result = daemon.dispatch(path=path, dry_run=False)
            inflight = (pathlib.Path(daemon.MAILBOX)
                        / "inflight" / name)
            history = (pathlib.Path(daemon.MAILBOX)
                       / ".dispatch-history" / (name + ".json"))
            direct_results.append(
                not result and not calls and inflight.is_file()
                and not history.exists())
    print("timeout-validation cli=" + repr(cli_rejected)
          + " positive=" + str(cli_positive)
          + " direct=" + repr(direct_results))
    return all(cli_rejected) and cli_positive and all(direct_results)


def arm_natural_completion_at_deadline():
    """Do not kill a child that reaches rc=0 during the deadline sleep."""
    with scratch_daemon() as (daemon, _):
        name = "0120-to-opus.md"
        body = "completes exactly at deadline\n"
        path = write_message(daemon, name, body)
        daemon.DISPATCH_TIMEOUT_MINUTES = 1

        class DeadlineClock:
            def __init__(self):
                self.now = 0.0

            def time(self):
                return self.now

            def sleep(self, _seconds):
                self.now = 60.0

        class DeadlineProcess:
            def __init__(self):
                self.returncode = None
                self.polls = 0
                self.kills = 0

            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    return None
                self.returncode = 0
                return self.returncode

            def kill(self):
                self.kills += 1
                self.returncode = -9

            def wait(self):
                return self.returncode

        child = DeadlineProcess()

        def deadline_popen(command, stdout, stderr, cwd, env):
            del command, stderr, cwd, env
            stdout.write("natural completion\n")
            stdout.flush()
            return child

        daemon.time = DeadlineClock()
        daemon.subprocess.Popen = deadline_popen
        result = daemon.dispatch(path=path, dry_run=False)
        history = (pathlib.Path(daemon.MAILBOX)
                   / ".dispatch-history" / (name + ".json"))
        done = pathlib.Path(daemon.DONE) / name
        print("deadline-natural result=" + str(result)
              + " polls=" + str(child.polls)
              + " kills=" + str(child.kills)
              + " history=" + str(history.exists())
              + " done=" + str(done.is_file()))
        return (result and child.polls >= 2 and child.kills == 0
                and not history.exists() and done.is_file())


def arm_killed_child_reporting_rc0_is_still_timeout():
    """Treat a child killed at deadline as timed out even if wait says rc0."""
    with scratch_daemon() as (daemon, _):
        name = "0121-to-opus.md"
        path = write_message(daemon, name, "kill then misleading rc0\n")
        daemon.DISPATCH_TIMEOUT_MINUTES = 1

        class DeadlineClock:
            def __init__(self):
                self.now = 0.0

            def time(self):
                return self.now

            def sleep(self, _seconds):
                self.now = 60.0

        class KillReportsSuccess:
            def __init__(self):
                self.returncode = None
                self.kills = 0

            def poll(self):
                return self.returncode

            def kill(self):
                self.kills += 1

            def wait(self):
                self.returncode = 0
                return self.returncode

        child = KillReportsSuccess()

        def misleading_popen(command, stdout, stderr, cwd, env):
            del command, stderr, cwd, env
            stdout.write("still live at deadline\n")
            stdout.flush()
            return child

        daemon.time = DeadlineClock()
        daemon.subprocess.Popen = misleading_popen
        result = daemon.dispatch(path=path, dry_run=False)
        history = (pathlib.Path(daemon.MAILBOX)
                   / ".dispatch-history" / (name + ".json"))
        failed = pathlib.Path(daemon.MAILBOX) / "failed" / name
        done = pathlib.Path(daemon.DONE) / name
        try:
            payload = json.loads(history.read_text(encoding="utf-8"))
            history_ok = nested_has_number(payload, 1)
        except (OSError, ValueError):
            history_ok = False
        print("kill-rc0 result=" + str(result)
              + " kills=" + str(child.kills)
              + " rc=" + str(child.returncode)
              + " history=" + str(history_ok)
              + " failed=" + str(failed.is_file())
              + " done=" + str(done.exists()))
        return (not result and child.kills == 1 and child.returncode == 0
                and history_ok and failed.is_file() and not done.exists())


def arm_timeout_history_replace_failure_is_conservative():
    """Keep the exact inflight source when atomic history replace fails."""
    with scratch_daemon() as (daemon, _):
        name = "0122-to-opus.md"
        path = write_message(daemon, name, "history replace failure\n")
        daemon.DISPATCH_TIMEOUT_MINUTES = 1

        class DeadlineClock:
            def __init__(self):
                self.now = 0.0

            def time(self):
                return self.now

            def sleep(self, _seconds):
                self.now = 60.0

        class HungProcess:
            def __init__(self):
                self.returncode = None
                self.kills = 0

            def poll(self):
                return self.returncode

            def kill(self):
                self.kills += 1
                self.returncode = -9

            def wait(self):
                return self.returncode

        child = HungProcess()

        def hung_popen(command, stdout, stderr, cwd, env):
            del command, stderr, cwd, env
            stdout.write("hung before atomic replace\n")
            stdout.flush()
            return child

        original_replace = daemon.os.replace

        def failing_replace(source, destination):
            del source, destination
            raise OSError("injected atomic replace failure")

        daemon.time = DeadlineClock()
        daemon.subprocess.Popen = hung_popen
        daemon.os.replace = failing_replace
        try:
            result = daemon.dispatch(path=path, dry_run=False)
        finally:
            daemon.os.replace = original_replace
        inflight = pathlib.Path(daemon.MAILBOX) / "inflight" / name
        history = (pathlib.Path(daemon.MAILBOX)
                   / ".dispatch-history" / (name + ".json"))
        history_dir = history.parent
        leftovers = list(history_dir.iterdir()) if history_dir.is_dir() else []
        failed = pathlib.Path(daemon.MAILBOX) / "failed" / name
        print("replace-failure result=" + str(result)
              + " kills=" + str(child.kills)
              + " inflight=" + str(inflight.is_file())
              + " history=" + str(history.exists())
              + " temp_leftovers=" + str(len(leftovers))
              + " failed=" + str(failed.exists()))
        return (not result and child.kills == 1 and inflight.is_file()
                and not history.exists() and not leftovers
                and not failed.exists())


def arm_state_move_rejects_substitution():
    """Reject symlink and copied-inode substitutions in done and failed."""
    outcomes = []
    for state in ["done", "failed"]:
        for variant in ["symlink", "copy"]:
            with scratch_daemon() as (daemon, root):
                name = "0130-to-opus.md"
                source = write_message(
                    daemon, name, "original claimed inode\n",
                    directory=os.path.join(daemon.MAILBOX, "inflight"))
                source_inode = daemon.regular_inode(path=source)
                directory = (daemon.DONE if state == "done" else
                             os.path.join(daemon.MAILBOX, "failed"))
                destination = pathlib.Path(directory) / name
                target = root / "substitution-target.txt"
                target.write_text("attacker-controlled\n", encoding="utf-8")

                def substitute(path, directory, variant=variant,
                               destination=destination, target=target):
                    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
                    if variant == "symlink":
                        pathlib.Path(path).unlink()
                        destination.symlink_to(target)
                    else:
                        shutil.copyfile(path, destination)
                        pathlib.Path(path).unlink()
                    return str(destination)

                daemon.move_without_overwrite = substitute
                if state == "done":
                    accepted = daemon.archive_consumed_message(
                        dispatch_path=source)
                else:
                    accepted = daemon.park_failed_message(
                        dispatch_path=source)
                guard = pathlib.Path(source + daemon.STATE_GUARD_SUFFIX)
                durable_blocker = (
                    daemon.regular_inode(path=source) == source_inode
                    or daemon.regular_inode(path=str(guard)) == source_inode)
                waiting_name = "0131-to-fable.md"
                waiting = write_message(
                    daemon, waiting_name, "must wait behind repair state\n")
                calls = []
                daemon.subprocess.Popen = success_popen(calls=calls)
                daemon.report_demand = lambda backlog: None
                second_pass = daemon.process_backlog(dry_run=False)
                refused = (not accepted and destination.exists()
                           and durable_blocker and second_pass is False
                           and not calls and pathlib.Path(waiting).is_file())
                outcomes.append(refused)
                print("state-substitution state=" + state
                      + " variant=" + variant
                      + " refused_restored_blocked=" + str(refused))
    return len(outcomes) == 4 and all(outcomes)


def arm_hostile_timeout_histories_are_controlled():
    """Refuse bounded hostile sidecars without launch or state consumption."""
    outcomes = []
    labels = ["huge", "deep", "oversized", "event-flood"]
    for index, label in enumerate(labels, start=1):
        with scratch_daemon() as (daemon, _):
            name = "014%d-to-opus.md" % index
            root_path = write_message(daemon, name, "history target body\n")
            history_path = pathlib.Path(
                daemon.timeout_history_path(name=name))
            history_path.parent.mkdir(parents=True, exist_ok=True)
            if label == "huge":
                payload = (
                    '{"schema":1,"message":"' + name
                    + '","timeouts":[{"killed_after_minutes":'
                    + ("9" * 4000) + '}]}' + "\n")
            elif label == "deep":
                payload = ("[" * 2000) + "0" + ("]" * 2000) + "\n"
            elif label == "oversized":
                payload = "x" * (daemon.MAX_TIMEOUT_HISTORY_BYTES + 1)
            else:
                payload = json.dumps({
                    "schema": 1,
                    "message": name,
                    "timeouts": [
                        {"killed_after_minutes": 1}
                        for _ in range(daemon.MAX_TIMEOUT_HISTORY_EVENTS + 1)
                    ],
                }) + "\n"
            history_path.write_text(payload, encoding="utf-8")
            before = hashlib.sha256(history_path.read_bytes()).hexdigest()
            calls = []
            daemon.subprocess.Popen = success_popen(calls=calls)
            raised = None
            try:
                result = daemon.dispatch(path=root_path, dry_run=False)
            except Exception as exc:
                raised = exc
                result = None
            inflight = (pathlib.Path(daemon.MAILBOX)
                        / "inflight" / name)
            after = hashlib.sha256(history_path.read_bytes()).hexdigest()
            controlled = (raised is None and result is False and not calls
                          and inflight.is_file()
                          and not pathlib.Path(root_path).exists()
                          and before == after)
            outcomes.append(controlled)
            print("hostile-history label=" + label
                  + " controlled=" + str(controlled)
                  + " raised="
                  + ("none" if raised is None else type(raised).__name__))
    return len(outcomes) == 4 and all(outcomes)


def mutation_killed(label, old, new, probe):
    """Write one source mutant to scratch and prove its probe goes red."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    occurrences = source.count(old)
    if occurrences != 1:
        print("MUTATION " + label + " INVALID anchor_count="
              + str(occurrences))
        return False
    mutant = source.replace(old, new, 1)
    try:
        compile(mutant, str(DAEMON_PATH), "exec")
    except SyntaxError as exc:
        print("MUTATION " + label + " INVALID syntax=" + str(exc))
        return False
    with tempfile.TemporaryDirectory(
            prefix="mailbox-daemon-source-mutant-") as tmp:
        mutant_path = pathlib.Path(tmp) / "mailbox_daemon.py"
        mutant_path.write_text(mutant, encoding="utf-8")
        try:
            survived = probe(mutant_path)
            detail = "probe_returned=" + str(survived)
        except Exception as exc:
            survived = False
            detail = "probe_raised=" + type(exc).__name__ + ":" + str(exc)
    killed = not survived
    print("MUTATION " + label + " " + ("RED" if killed else "SURVIVED")
          + " " + detail)
    return killed


def arm_source_mutations():
    """Kill load-bearing source mutants for each new safety property."""
    recursive_snapshot = (
        '    snapshot = glob.glob(os.path.join(MAILBOX, "**", "*.md"),\n'
        '                         recursive=True)')
    root_only_snapshot = (
        '    snapshot = glob.glob(os.path.join(MAILBOX, "*.md"))')
    cwd_lane = (
        '        if AGENT_CWD[queued_agent] == AGENT_CWD[agent]:')
    same_agent_only = '        if queued_agent == agent:'
    timeout_write = (
        '                        write_timeout_history(\n'
        '                            name=name,\n'
        '                            killed_after_minutes=killed_after_minutes,\n'
        '                            observed_elapsed_minutes=(\n'
        '                                observed_elapsed_minutes))')
    timeout_read = '            history = timeout_events(name=name)'
    archive_return = (
        '    return archive_consumed_message(dispatch_path=dispatch_path)')
    process_return = (
        '    return (not blockers\n'
        '            and len(lane_outcomes) == len(lanes)\n'
        '            and all(lane_outcomes.values()))')
    archive_verify = (
        '    verified = (destination_inode == source_inode\n'
        '                and not os.path.lexists(dispatch_path))')
    exact_newlines = (
        '        with open(dispatch_path, encoding="utf-8", newline="") as f:')
    lane_break = (
        '            # A false result can mean the head is still inflight '
        'because its\n'
        '            # archive or failed-state move was ambiguous. Do not '
        'release later\n'
        '            # work in the same lane past an unresolved head.\n'
        '            break')

    cases = [
        ("recursive-store-snapshot", recursive_snapshot,
         root_only_snapshot, currency_contract_probe),
        ("cwd-lane-not-recipient", cwd_lane, same_agent_only,
         currency_contract_probe),
        ("timeout-history-write", timeout_write, '                        pass',
         arm_timeout_history_and_retry_hint),
        ("timeout-history-read", timeout_read, '            history = []',
         arm_timeout_history_and_retry_hint),
        ("dispatch-archive-result", archive_return, '    return True',
         arm_archive_failure_propagates),
        ("process-backlog-result", process_return, '    return True',
         arm_archive_failure_propagates),
        ("archive-source-absence", archive_verify,
         '    verified = (destination_inode == source_inode)',
         arm_archive_requires_source_absence),
        ("exact-crlf-body", exact_newlines,
         '        with open(dispatch_path, encoding="utf-8") as f:',
         arm_crlf_body_is_exact_suffix),
        ("lane-break-on-unconsumed", lane_break,
         lane_break.rsplit('break', maxsplit=1)[0] + 'continue',
         arm_lane_stops_after_unconsumed_head),
    ]
    outcomes = []
    for label, old, new, probe in cases:
        outcomes.append(mutation_killed(
            label=label, old=old, new=new, probe=probe))
    print("mutation-score killed=" + str(sum(outcomes))
          + "/" + str(len(outcomes)))
    return outcomes and all(outcomes)


def main():
    """Run all scratch witnesses and return nonzero on any regression."""
    arms = [
        ("post-claim-currency", arm_post_claim_currency_banner),
        ("exact-crlf-body", arm_crlf_body_is_exact_suffix),
        ("timeout-history-retry", arm_timeout_history_and_retry_hint),
        ("ordinary-rc1-no-history",
         arm_ordinary_failure_has_no_timeout_history),
        ("dry-run-read-only", arm_dry_run_is_strictly_read_only),
        ("archive-failure-propagates", arm_archive_failure_propagates),
        ("same-cwd-archive-and-logs",
         arm_same_cwd_serializes_through_archive_and_logs),
        ("archive-source-absence", arm_archive_requires_source_absence),
        ("lane-stops-on-unconsumed-head",
         arm_lane_stops_after_unconsumed_head),
        ("cross-pass-inflight-lanes",
         arm_cross_pass_inflight_blocks_only_its_cwd),
        ("only-inflight-fails", arm_only_inflight_reports_failure),
        ("timeout-value-validation", arm_timeout_value_validation),
        ("natural-completion-at-deadline",
         arm_natural_completion_at_deadline),
        ("killed-child-rc0-is-timeout",
         arm_killed_child_reporting_rc0_is_still_timeout),
        ("timeout-history-replace-failure",
         arm_timeout_history_replace_failure_is_conservative),
        ("state-move-substitutions", arm_state_move_rejects_substitution),
        ("hostile-timeout-histories",
         arm_hostile_timeout_histories_are_controlled),
        ("source-mutations", arm_source_mutations),
    ]
    failures = []
    for name, arm in arms:
        try:
            passed = arm()
        except Exception as exc:
            print(name + " uncaught=" + type(exc).__name__ + ": " + str(exc))
            passed = False
        print("ARM " + name + " " + ("PASS" if passed else "FAIL"))
        if not passed:
            failures.append(name)
    print("SUMMARY failures=" + repr(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
