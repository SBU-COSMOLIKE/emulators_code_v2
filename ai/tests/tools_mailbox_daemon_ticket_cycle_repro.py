#!/usr/bin/env python3
"""Scratch-only checks for mailbox ticket-cycle accounting.

The tests in this file never launch Claude or Codex.  Each check loads a
fresh daemon module and redirects its mailbox, cycle state, and backlog into
one temporary directory.  The checks distinguish a completed ticket cycle
from both ordinary role conversation and the periodic safe-stop countdown.
"""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile


AI_ROOT = Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
BASE_A = "1" * 40
BASE_B = "2" * 40
BASE_C = "3" * 40
BASE_D = "4" * 40
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
COMMIT_C = "c" * 40
COMMIT_D = "d" * 40

TICKETS = (
    ("role-chatter", "HIGH"),
    ("normal-return", "HIGH"),
    ("receipt-correlation", "HIGH"),
    ("reopen-review", "HIGH"),
    # Two Critical bug fixes put the scratch ledger into the daemon's proved
    # emergency condition.  This is test state, not a severity example.
    ("emergency-primary", "CRITICAL"),
    ("emergency-second", "CRITICAL"),
    ("route-spoof", "HIGH"),
    ("same-ticket", "HIGH"),
    ("other-emergency", "HIGH"),
)


class AttributeProxy:
    """Delegate attributes except for explicit scratch replacements."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


def load_daemon():
    """Load an independent copy of the production daemon."""
    spec = importlib.util.spec_from_file_location(
        "mailbox_daemon_ticket_cycle_repro", DAEMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backlog_text(tickets=TICKETS, emergency=False):
    """Build a small but fully indexed backlog accepted by the daemon."""
    index = []
    details = []
    for anchor, severity in tickets:
        if severity == "CRITICAL" and not emergency:
            severity = "HIGH"
        title = anchor.replace("-", " ").title()
        index.append(
            "- OPEN **" + severity + "** **BUG FIX** — [" + title
            + "](#" + anchor + ")\n")
        details.extend([
            "\n<a id=\"" + anchor + "\"></a>\n",
            "### " + title + "\n\n",
            "**Red Team reopen count: 0.**\n\n",
            "**Red Team reopening: allowed.**\n\n",
            "Scratch-only ticket detail.\n",
        ])
    return "".join(index + details)


@contextlib.contextmanager
def scratch_daemon(emergency=False):
    """Redirect all state used by these checks into a temporary tree."""
    with tempfile.TemporaryDirectory(prefix="mailbox-ticket-cycle-") as tmp:
        root = Path(tmp)
        ai_root = root / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        mailbox.mkdir(parents=True)
        backlog = ai_root / "notes" / "backlog.md"
        backlog.write_text(
            backlog_text(emergency=emergency), encoding="utf-8")
        shared_lane = root / "claude-lane"
        sol_lane = root / "sol-lane"
        shared_lane.mkdir()
        sol_lane.mkdir()

        daemon = load_daemon()
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
        daemon.MAILBOX = str(mailbox)
        daemon.DONE = str(mailbox / "done")
        daemon.RELAY_DIR = str(ai_root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(backlog)
        daemon.PREAMBLE = "scratch ticket-cycle preamble\n"
        daemon.AGENT_CWD = {
            "fable": str(shared_lane),
            "opus": str(shared_lane),
            "sol": str(sol_lane),
        }
        daemon.report_demand = lambda backlog: None
        daemon.report_landing_debt = lambda: None
        daemon.reconcile_landing_debt_handoff = lambda: None
        daemon.warn_if_mailbox_unwatched = lambda: None
        daemon.git_commit_exists = lambda commit: commit in {
            BASE_A, BASE_B, BASE_C, BASE_D}
        daemon.git_commit_descends_from = (
            lambda starting_commit, accepted_commit:
            starting_commit in {BASE_A, BASE_B, BASE_C, BASE_D}
            and accepted_commit in {COMMIT_A, COMMIT_B, COMMIT_C, COMMIT_D}
            and starting_commit != accepted_commit)
        yield daemon, mailbox


def call_main(daemon, arguments):
    """Call main with isolated argv and return captured output."""
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


def flow_payload(cycle_id, text, mode="normal"):
    """Return one exact Architect/Implementer ticket-flow message."""
    return (
        "MAILBOX-FLOW: ticket\n"
        "MAILBOX-CYCLE: " + cycle_id + "\n"
        "MAILBOX-MODE: " + mode + "\n\n" + text + "\n")


def register_normal_commit(daemon, cycle_id, commit):
    """Register ordinary A/I work and the Architect's accepted commit."""
    daemon.register_ticket_cycle_message(
        agent="opus", message=flow_payload(cycle_id, "Implementer start"))
    daemon.register_ticket_cycle_message(
        agent="fable", message=flow_payload(cycle_id, "Architect return"))
    completed = daemon.record_architect_commit(
        cycle_id=cycle_id, accepted_commit=commit, mode="normal")
    if completed != 0:
        raise AssertionError("a normal Architect commit completed a cycle")


def register_closure(daemon, cycle_id, commit):
    """Register the exact Red Team request for one accepted commit."""
    closure = daemon.sol_ticket_payload(
        ticket_kind="closure",
        text="Review only this accepted change.",
        review_cycle=cycle_id,
        review_commit=commit)
    returned_cycle, returned_commit = daemon.register_ticket_cycle_message(
        agent="sol", message=closure)
    if (returned_cycle, returned_commit) != (cycle_id, commit):
        raise AssertionError("closure registration changed its identity")


def write_receipt(daemon, mailbox, name, cycle_id, commit, result):
    """Write one exact Red Team return to the scratch mailbox."""
    payload = daemon.redteam_review_receipt_payload(
        review_cycle=cycle_id,
        review_commit=commit,
        result=result,
        text="Evidence for the Architect.")
    path = mailbox / name
    path.write_text(payload, encoding="utf-8", newline="")
    return path


def arm_role_chatter_does_not_complete_cycle():
    """Repeated A/I exchanges register work but never complete a cycle."""
    with scratch_daemon() as (daemon, _mailbox):
        cycle_id = "role-chatter@" + BASE_A
        controller = daemon.SafeKillRendezvous()
        daemon._ACTIVE_WATCH_RENDEZVOUS = controller
        try:
            daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    cycle_id, "Implementer start"))
            daemon.register_ticket_cycle_message(
                agent="fable", message=flow_payload(
                    cycle_id, "Architect repair plan"))
            daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    cycle_id, "Implementer repair return"))
            state = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        passed = (
            state["generation"] == 0
            and state["active"][cycle_id]["phase"] == "implementation"
            and state["active"][cycle_id]["mode"] == "normal"
            and state["active"][cycle_id]["route"] == "primary"
            and state["active"][cycle_id]["epoch"] is None
            and controller.completed_ticket_cycles() == 0)
    print("A/I chatter is not a cycle=" + str(passed))
    return passed


def arm_normal_return_completes_cycle():
    """One exact review receipt completes one normal ticket cycle."""
    with scratch_daemon() as (daemon, mailbox):
        cycle_id = "normal-return@" + BASE_A
        register_normal_commit(daemon, cycle_id, COMMIT_A)
        register_closure(daemon, cycle_id, COMMIT_A)
        before = daemon.fable_message_inode_snapshot()
        receipt = write_receipt(
            daemon, mailbox, "0001-to-fable.md", cycle_id, COMMIT_A,
            "NO CHANGE")
        path, result, problem = daemon.matching_new_redteam_receipt(
            cycle_id=cycle_id,
            accepted_commit=COMMIT_A,
            before_inodes=before)
        completed = daemon.complete_ticket_cycle(cycle_id, COMMIT_A)
        state = daemon.read_ticket_cycle_state()
        passed = (
            problem is None and Path(path) == receipt
            and result == "NO CHANGE" and completed
            and state["generation"] == 1
            and state["completed"] == {cycle_id: COMMIT_A}
            and not state["active"])
    print("exact normal review completes one cycle=" + str(passed))
    return passed


def arm_wrong_or_missing_receipt_fails():
    """No receipt, or a receipt for another commit, cannot close the cycle."""
    with scratch_daemon() as (daemon, mailbox):
        cycle_id = "receipt-correlation@" + BASE_B
        register_normal_commit(daemon, cycle_id, COMMIT_B)
        register_closure(daemon, cycle_id, COMMIT_B)
        before = daemon.fable_message_inode_snapshot()
        no_path, no_result, missing_problem = (
            daemon.matching_new_redteam_receipt(
                cycle_id=cycle_id,
                accepted_commit=COMMIT_B,
                before_inodes=before))
        write_receipt(
            daemon, mailbox, "0002-to-fable.md", cycle_id, COMMIT_C,
            "NO CHANGE")
        wrong_path, wrong_result, wrong_problem = (
            daemon.matching_new_redteam_receipt(
                cycle_id=cycle_id,
                accepted_commit=COMMIT_B,
                before_inodes=before))
        state = daemon.read_ticket_cycle_state()
        passed = (
            no_path is None and no_result is None
            and "found 0" in missing_problem
            and wrong_path is None and wrong_result is None
            and "found 0" in wrong_problem
            and state["generation"] == 0
            and state["active"][cycle_id]["phase"] == "awaiting-redteam")
    print("wrong or missing review receipt fails=" + str(passed))
    return passed


def arm_reopen_return_completes_review():
    """REOPEN is advisory, but its correlated pass still completes a cycle."""
    with scratch_daemon() as (daemon, mailbox):
        cycle_id = "reopen-review@" + BASE_C
        register_normal_commit(daemon, cycle_id, COMMIT_C)
        register_closure(daemon, cycle_id, COMMIT_C)
        before = daemon.fable_message_inode_snapshot()
        write_receipt(
            daemon, mailbox, "0003-to-fable.md", cycle_id, COMMIT_C,
            "REOPEN")
        path, result, problem = daemon.matching_new_redteam_receipt(
            cycle_id=cycle_id,
            accepted_commit=COMMIT_C,
            before_inodes=before)
        completed = daemon.complete_ticket_cycle(cycle_id, COMMIT_C)
        state = daemon.read_ticket_cycle_state()
        passed = (
            path is not None and problem is None and result == "REOPEN"
            and completed and state["generation"] == 1
            and state["completed"] == {cycle_id: COMMIT_C})
    print("REOPEN return completes review pass=" + str(passed))
    return passed


def arm_emergency_pair_is_one_cycle():
    """A primary and second Implementer commit pair counts exactly once."""
    with scratch_daemon(emergency=True) as (daemon, mailbox):
        primary = "emergency-primary@" + BASE_A
        second = "emergency-second@" + BASE_D
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow_payload(
                primary, "Primary implementation",
                mode="emergency-primary"))
        first = daemon.record_architect_commit(
            cycle_id=primary,
            accepted_commit=COMMIT_A,
            mode="emergency-primary")
        after_one = daemon.read_ticket_cycle_state()
        second_assignment = (
            "MAILBOX-TICKET: closure\n"
            + flow_payload(
                second,
                daemon.SECOND_IMPLEMENTER_MODE_SENTENCE
                + "\n\nFollow the Architect's exact plan.",
                mode="emergency-second"))
        daemon.register_ticket_cycle_message(
            agent="sol", message=second_assignment)
        paired = daemon.record_architect_commit(
            cycle_id=second,
            accepted_commit=COMMIT_D,
            mode="emergency-second")
        after_pair = daemon.read_ticket_cycle_state()

        # Replaying one exact archived receipt is idempotent.  It must not be
        # mislabeled as a lone emergency ticket whose partner never arrived.
        duplicate_path = mailbox / "0089-to-daemon.md"
        duplicate_path.write_text(
            daemon.architect_commit_receipt_payload(
                cycle_id=primary, commit=COMMIT_A,
                mode="emergency-primary"),
            encoding="utf-8", newline="")
        duplicate_output = io.StringIO()
        with contextlib.redirect_stdout(duplicate_output):
            duplicate_consumed = daemon.consume_daemon_message(
                path=str(duplicate_path))
        passed = (
            first == 0 and after_one["generation"] == 0
            and after_one["active"][primary]["phase"]
            == "emergency-committed"
            and after_one["active"][primary]["route"] == "primary"
            and after_one["active"][primary]["epoch"] == 1
            and paired == 1 and after_pair["generation"] == 1
            and after_pair["completed"]
            == {primary: COMMIT_A, second: COMMIT_D}
            and not after_pair["active"]
            and not after_pair["emergency_commits"]
            and duplicate_consumed
            and "unpaired emergency" not in duplicate_output.getvalue())
    print("emergency commit pair is one cycle=" + str(passed))
    return passed


def arm_only_registered_emergency_work_is_grandfathered():
    """Emergency exit keeps admitted Sol work, not another queued ticket."""
    with scratch_daemon(emergency=True) as (daemon, mailbox):
        admitted_cycle = "emergency-second@" + BASE_D
        admitted_message = emergency_second_message(
            daemon, admitted_cycle)
        daemon.register_ticket_cycle_message(
            agent="sol", message=admitted_message)

        # Clearing both Critical classifications ends the emergency. The
        # already registered assignment may finish, while another file that
        # has only reached the root queue may not borrow that permission.
        Path(daemon.BACKLOG_LEDGER).write_text(
            backlog_text(emergency=False), encoding="utf-8")
        daemon.sync_ticket_cycle_emergency_condition()
        admitted_refusal = daemon.second_implementer_emergency_refusal(
            message=admitted_message)

        queued_cycle = "other-emergency@" + BASE_C
        queued_message = emergency_second_message(daemon, queued_cycle)
        (mailbox / "0090-to-sol.md").write_text(
            queued_message, encoding="utf-8", newline="")
        daemon.reconcile_ticket_cycle_state()
        queued_refusal = daemon.second_implementer_emergency_refusal(
            message=queued_message)
        state = daemon.read_ticket_cycle_state()
        passed = (
            admitted_refusal is None
            and queued_refusal is not None
            and "emergency-only" in queued_refusal
            and set(state["active"]) == {admitted_cycle}
            and not state["emergency_condition"])
    print("only registered emergency work is grandfathered=" + str(passed))
    return passed


def arm_unpaired_emergency_commit_does_not_block_drain():
    """A finished lone emergency ticket is not miscounted or left active."""
    with scratch_daemon(emergency=True) as (daemon, _mailbox):
        primary = "emergency-primary@" + BASE_A
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow_payload(
                primary, "Primary implementation",
                mode="emergency-primary"))
        Path(daemon.BACKLOG_LEDGER).write_text(
            backlog_text(emergency=False), encoding="utf-8")
        daemon.sync_ticket_cycle_emergency_condition()
        primary_result = daemon.record_architect_commit(
            cycle_id=primary, accepted_commit=COMMIT_A,
            mode="emergency-primary")
        primary_state = daemon.read_ticket_cycle_state()

    with scratch_daemon(emergency=True) as (daemon, _mailbox):
        second = "emergency-second@" + BASE_D
        daemon.register_ticket_cycle_message(
            agent="sol", message=emergency_second_message(daemon, second))
        second_result = daemon.record_architect_commit(
            cycle_id=second, accepted_commit=COMMIT_D,
            mode="emergency-second")
        committed_state = daemon.read_ticket_cycle_state()
        Path(daemon.BACKLOG_LEDGER).write_text(
            backlog_text(emergency=False), encoding="utf-8")
        daemon.sync_ticket_cycle_emergency_condition()
        second_state = daemon.read_ticket_cycle_state()

    passed = (
        primary_result == 0
        and primary_state["completed"] == {primary: COMMIT_A}
        and primary_state["generation"] == 0
        and not primary_state["active"]
        and not primary_state["emergency_commits"]
        and second_result == 0
        and committed_state["active"][second]["phase"]
        == "emergency-committed"
        and second_state["completed"] == {second: COMMIT_D}
        and second_state["generation"] == 0
        and not second_state["active"]
        and not second_state["emergency_commits"])
    print("unpaired emergency commit does not block drain=" + str(passed))
    return passed


def arm_bad_backlog_refuses_without_traceback():
    """A missing local backlog produces one plain fail-closed result."""
    with scratch_daemon() as (daemon, _mailbox):
        Path(daemon.BACKLOG_LEDGER).unlink()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = daemon.process_backlog(dry_run=False)
        text = output.getvalue()
        passed = (
            result is False
            and "refused mailbox pass" in text
            and "ai/notes/backlog.md is missing" in text
            and "no new role work was started" in text)
    print("bad backlog refuses without traceback=" + str(passed))
    return passed


def arm_rejected_daemon_receipts_are_recoverable():
    """State-rejected receipts stay out of done and can be fixed/requeued."""
    with scratch_daemon() as (daemon, mailbox):
        cycle_id = "normal-return@" + BASE_A
        payload = daemon.architect_commit_receipt_payload(
            cycle_id=cycle_id, commit=COMMIT_A, mode="normal")
        pending = mailbox / "0091-to-daemon.md"
        pending.write_text(payload, encoding="utf-8", newline="")
        first = daemon.consume_daemon_message(path=str(pending))
        failed = mailbox / "failed" / pending.name
        done = mailbox / "done" / pending.name
        stayed_out_of_done = not done.exists()

        # Reproduce a receipt archived by the older archive-before-state
        # ordering. Startup must move it to failed rather than fail forever.
        historical_cycle = "receipt-correlation@" + BASE_B
        historical_name = "0092-to-daemon.md"
        historical = mailbox / "done" / historical_name
        historical.parent.mkdir(parents=True, exist_ok=True)
        historical.write_text(
            daemon.architect_commit_receipt_payload(
                cycle_id=historical_cycle, commit=COMMIT_B, mode="normal"),
            encoding="utf-8", newline="")
        recovered_without_poison = daemon.reconcile_ticket_cycle_state()
        historical_failed = mailbox / "failed" / historical_name

        # Once the missing implementation identity exists, the first parked
        # receipt can be put back in the root and consumed normally.
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow_payload(
                cycle_id, "Implementer start"))
        requeued = mailbox / pending.name
        failed.rename(requeued)
        second = daemon.consume_daemon_message(path=str(requeued))
        state = daemon.read_ticket_cycle_state()
        passed = (
            first is False and stayed_out_of_done
            and historical_failed.is_file() and not historical.exists()
            and recovered_without_poison == 0
            and second is True and done.is_file()
            and state["active"][cycle_id]["phase"]
            == "committed-awaiting-closure")
    print("rejected daemon receipts are recoverable=" + str(passed))
    return passed


def arm_crash_replays_one_pending_cycle_return():
    """A crash between durable completion and the counter loses no cycle."""
    with scratch_daemon() as (daemon, _mailbox):
        cycle_id = "normal-return@" + BASE_A
        first_controller = daemon.SafeKillRendezvous(ticket_cycle_limit=1)
        daemon._ACTIVE_WATCH_RENDEZVOUS = first_controller
        try:
            register_normal_commit(daemon, cycle_id, COMMIT_A)
            register_closure(daemon, cycle_id, COMMIT_A)
            completed = daemon.complete_ticket_cycle(cycle_id, COMMIT_A)
            before_crash = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None

        replacement = daemon.SafeKillRendezvous(ticket_cycle_limit=1)
        daemon._ACTIVE_WATCH_RENDEZVOUS = replacement
        try:
            daemon.reconcile_ticket_cycle_state()
            delivered = daemon.deliver_pending_ticket_cycle_returns()
            delivered_again = daemon.deliver_pending_ticket_cycle_returns()
            after_restart = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        passed = (
            completed is True
            and first_controller.completed_ticket_cycles() == 0
            and before_crash["pending_cycle_returns"] == 1
            and delivered == 1 and delivered_again == 0
            and replacement.completed_ticket_cycles() == 1
            and replacement.ticket_cycle_limit_reached()
            and after_restart["pending_cycle_returns"] == 0)
    print("crash replays one pending cycle return=" + str(passed))
    return passed


def arm_completed_receipt_needs_no_git_object():
    """An idempotent archived receipt survives later Git object cleanup."""
    with scratch_daemon() as (daemon, mailbox):
        cycle_id = "normal-return@" + BASE_A
        register_normal_commit(daemon, cycle_id, COMMIT_A)
        register_closure(daemon, cycle_id, COMMIT_A)
        daemon.complete_ticket_cycle(cycle_id, COMMIT_A)
        done = mailbox / "done"
        done.mkdir(parents=True, exist_ok=True)
        (done / "0093-to-daemon.md").write_text(
            daemon.architect_commit_receipt_payload(
                cycle_id=cycle_id, commit=COMMIT_A, mode="normal"),
            encoding="utf-8", newline="")
        git_calls = []

        def unavailable_git(*args, **kwargs):
            git_calls.append((args, kwargs))
            return False

        daemon.git_commit_exists = unavailable_git
        daemon.git_commit_descends_from = unavailable_git
        direct = daemon.record_architect_commit(
            cycle_id=cycle_id, accepted_commit=COMMIT_A, mode="normal")
        recovered = daemon.reconcile_ticket_cycle_state()
        passed = (
            direct == 0 and recovered == 0 and not git_calls
            and (done / "0093-to-daemon.md").is_file()
            and daemon.read_ticket_cycle_state()["completed"]
            == {cycle_id: COMMIT_A})
    print("completed receipt needs no Git object=" + str(passed))
    return passed


def raises_cycle_error(daemon, action, phrase):
    """Return whether one action fails closed with the expected reason."""
    try:
        action()
    except daemon.TicketCycleStateError as exc:
        return phrase in str(exc)
    return False


def arm_mode_route_and_anchor_spoofing_fails():
    """A message cannot invent a ticket or change its saved route or mode."""
    with scratch_daemon() as (daemon, _mailbox):
        invented = "not-in-backlog@" + BASE_A
        invented_fails = raises_cycle_error(
            daemon,
            lambda: daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    invented, "Invented ticket")),
            "indexed Open backlog ticket")

        architect_first = "route-spoof@" + BASE_A
        architect_fails = raises_cycle_error(
            daemon,
            lambda: daemon.register_ticket_cycle_message(
                agent="fable", message=flow_payload(
                    architect_first, "Architect invents the cycle")),
            "Architect route cannot invent")

        cycle_id = "route-spoof@" + BASE_B
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow_payload(
                cycle_id, "Primary start", mode="normal"))
        changed_mode_fails = raises_cycle_error(
            daemon,
            lambda: daemon.register_ticket_cycle_message(
                agent="fable", message=flow_payload(
                    cycle_id, "Spoofed return", mode="two-role")),
            "changed its saved mode or Implementer route")
        primary_claims_second = raises_cycle_error(
            daemon,
            lambda: daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    "other-emergency@" + BASE_C,
                    "Wrong route", mode="emergency-second")),
            "primary Implementer cannot claim")
        sol_claims_primary_message = (
            "MAILBOX-TICKET: closure\n"
            + flow_payload(
                "other-emergency@" + BASE_D,
                daemon.SECOND_IMPLEMENTER_MODE_SENTENCE
                + "\n\nSpoofed primary route.",
                mode="emergency-primary"))
        sol_claims_primary = raises_cycle_error(
            daemon,
            lambda: daemon.register_ticket_cycle_message(
                agent="sol", message=sol_claims_primary_message),
            "must use MAILBOX-MODE: emergency-second")
        state = daemon.read_ticket_cycle_state()
        passed = (
            invented_fails and architect_fails and changed_mode_fails
            and primary_claims_second and sol_claims_primary
            and set(state["active"]) == {cycle_id}
            and state["active"][cycle_id]["mode"] == "normal")
    print("mode, route, and anchor spoofing fails=" + str(passed))
    return passed


def emergency_second_message(daemon, cycle_id):
    """Return an exact Sol second-Implementer assignment."""
    return (
        "MAILBOX-TICKET: closure\n"
        + flow_payload(
            cycle_id,
            daemon.SECOND_IMPLEMENTER_MODE_SENTENCE
            + "\n\nImplement only this ticket.",
            mode="emergency-second"))


def arm_emergency_pair_identity_fails_closed():
    """An emergency pair needs two ticket anchors and two accepted commits."""
    with scratch_daemon(emergency=True) as (daemon, _mailbox):
        first = "same-ticket@" + BASE_A
        second = "same-ticket@" + BASE_B
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow_payload(
                first, "Primary", mode="emergency-primary"))
        daemon.record_architect_commit(
            cycle_id=first, accepted_commit=COMMIT_A,
            mode="emergency-primary")
        daemon.register_ticket_cycle_message(
            agent="sol", message=emergency_second_message(daemon, second))
        same_ticket_fails = raises_cycle_error(
            daemon,
            lambda: daemon.record_architect_commit(
                cycle_id=second, accepted_commit=COMMIT_B,
                mode="emergency-second"),
            "two different tickets")

    with scratch_daemon(emergency=True) as (daemon, _mailbox):
        primary = "emergency-primary@" + BASE_C
        second = "emergency-second@" + BASE_D
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow_payload(
                primary, "Primary", mode="emergency-primary"))
        daemon.record_architect_commit(
            cycle_id=primary, accepted_commit=COMMIT_C,
            mode="emergency-primary")
        daemon.register_ticket_cycle_message(
            agent="sol", message=emergency_second_message(daemon, second))
        same_commit_fails = raises_cycle_error(
            daemon,
            lambda: daemon.record_architect_commit(
                cycle_id=second, accepted_commit=COMMIT_C,
                mode="emergency-second"),
            "two different accepted commits")
    passed = same_ticket_fails and same_commit_fails
    print("emergency pair identity fails closed=" + str(passed))
    return passed


def arm_accepted_commit_must_descend_from_base():
    """The Architect cannot close a cycle with its base or unrelated commit."""
    with scratch_daemon() as (daemon, _mailbox):
        cycle_id = "normal-return@" + BASE_A
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow_payload(
                cycle_id, "Implementer start"))
        daemon.git_commit_descends_from = (
            lambda starting_commit, accepted_commit: False)
        passed = raises_cycle_error(
            daemon,
            lambda: daemon.record_architect_commit(
                cycle_id=cycle_id, accepted_commit=COMMIT_A,
                mode="normal"),
            "not a new descendant")
    print("accepted commit must descend from base=" + str(passed))
    return passed


def arm_cycle_limit_closes_admission_race():
    """A returned cycle closes new admission while an older turn can finish."""
    with scratch_daemon() as (daemon, _mailbox):
        controller = daemon.SafeKillRendezvous(ticket_cycle_limit=1)
        admitted_before_return = controller.begin_attempt()
        controller.ticket_cycle_returned()
        refused_after_return = controller.begin_attempt()
        controller.finish_attempt(permit=admitted_before_return)
        passed = (
            admitted_before_return is not None
            and refused_after_return is None
            and controller.ticket_cycle_limit_reached()
            and controller.completed_ticket_cycles() == 1
            and controller.all_idle())
    print("cycle-limit admission race closes=" + str(passed))
    return passed


def arm_safe_stop_never_counts_cycle():
    """Child cadence and its 20-second manual window do not change cycles."""
    with scratch_daemon() as (daemon, _mailbox):
        daemon.RENDEZVOUS_DISPATCH_INTERVAL = 2
        controller = daemon.SafeKillRendezvous()
        for _ in range(2):
            permit = controller.begin_attempt()
            controller.turn_started(permit)
            controller.turn_finished(permit)
            controller.finish_attempt(permit)
        before = controller.completed_ticket_cycles()
        real_time = daemon.time
        daemon.time = AttributeProxy(real_time, sleep=lambda seconds: None)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            daemon.run_safe_kill_countdown(controller=controller)
        after = controller.completed_ticket_cycles()
        countdown = [
            line for line in output.getvalue().splitlines()
            if line.startswith("every enabled role is idle; safe to Ctrl-C")]
        passed = (
            before == 0 and after == 0
            and len(countdown) == daemon.SAFE_KILL_COUNTDOWN_SECONDS
            and controller.all_idle())
    print("safe-stop countdown never counts a cycle=" + str(passed))
    return passed


def arm_skip_redteam_rejects_positive_cycle():
    """A positive cycle limit cannot run without the required Red Team."""
    with scratch_daemon() as (daemon, _mailbox):
        rc, output, errors, error = call_main(
            daemon, ["--watch", "--skip-redteam", "--cycle", "1"])
        passed = (
            rc == 1 and error is None and errors == ""
            and "cannot use a positive --cycle" in output
            and "a normal ticket cycle requires a Red Team return" in output)
    print("skip-redteam rejects positive cycle=" + str(passed))
    return passed


def arm_cycle_zero_remains_drain():
    """Cycle zero drains recorded two-role work and applies no cycle count."""
    with scratch_daemon() as (daemon, _mailbox):
        Path(daemon.BACKLOG_LEDGER).write_text("", encoding="utf-8")
        rc, output, errors, error = call_main(
            daemon, ["--watch", "--skip-redteam", "--cycle", "0"])
        passed = (
            rc == 0 and error is None and errors == ""
            and "cycle 0: wait until no Architect or Implementer message"
            in output
            and "two-role drain complete; no ticket-cycle count applies"
            in output
            and "cycle limit reached" not in output)
    print("cycle zero remains a drain=" + str(passed))
    return passed


def main():
    """Run every focused check and return nonzero on the first regression."""
    checks = [
        ("role chatter", arm_role_chatter_does_not_complete_cycle),
        ("normal return", arm_normal_return_completes_cycle),
        ("receipt correlation", arm_wrong_or_missing_receipt_fails),
        ("reopen return", arm_reopen_return_completes_review),
        ("emergency pair", arm_emergency_pair_is_one_cycle),
        ("registered emergency grandfather",
         arm_only_registered_emergency_work_is_grandfathered),
        ("unpaired emergency completion",
         arm_unpaired_emergency_commit_does_not_block_drain),
        ("bad backlog plain refusal",
         arm_bad_backlog_refuses_without_traceback),
        ("recoverable daemon receipt",
         arm_rejected_daemon_receipts_are_recoverable),
        ("crash-safe cycle return",
         arm_crash_replays_one_pending_cycle_return),
        ("completed receipt without Git",
         arm_completed_receipt_needs_no_git_object),
        ("mode/route/anchor spoofing",
         arm_mode_route_and_anchor_spoofing_fails),
        ("emergency identity", arm_emergency_pair_identity_fails_closed),
        ("commit ancestry", arm_accepted_commit_must_descend_from_base),
        ("cycle admission race", arm_cycle_limit_closes_admission_race),
        ("safe-stop separation", arm_safe_stop_never_counts_cycle),
        ("positive two-role refusal",
         arm_skip_redteam_rejects_positive_cycle),
        ("cycle-zero drain", arm_cycle_zero_remains_drain),
    ]
    failures = []
    for name, check in checks:
        try:
            if not check():
                failures.append(name)
        except BaseException as exc:
            print(name + " raised " + type(exc).__name__ + ": " + str(exc))
            failures.append(name)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("all ticket-cycle regression checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
