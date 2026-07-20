#!/usr/bin/env python3
"""Scratch-only checks for mailbox ticket-cycle accounting.

The tests never launch Claude or Codex. Each check loads a fresh daemon and
redirects its mailbox, saved cycle state, and backlog into a temporary folder.

The rule under test is deliberately simple: one accepted ticket is one cycle.
In the normal three-role mode, that cycle finishes after the matched Red Team
return. In two-role mode, it finishes at the accepted Architect commit.
"""

import contextlib
import importlib.util
import io
import json
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
    ("two-role-return", "HIGH"),
    ("capacity-a", "HIGH"),
    ("capacity-b", "HIGH"),
    ("capacity-c", "HIGH"),
    ("pipeline-a", "HIGH"),
    ("pipeline-b", "HIGH"),
    ("pipeline-c", "HIGH"),
    ("repair-a", "HIGH"),
    ("repair-b", "HIGH"),
    ("repair-c", "HIGH"),
    ("route-spoof", "HIGH"),
    ("schema-primary", "HIGH"),
    ("schema-second", "HIGH"),
    ("schema-accepted", "HIGH"),
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


def backlog_text(tickets=TICKETS):
    """Build a small, fully indexed Open backlog accepted by the daemon."""
    index = []
    details = []
    for anchor, severity in tickets:
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
def scratch_daemon():
    """Redirect all state used by these checks into a temporary tree."""
    with tempfile.TemporaryDirectory(prefix="mailbox-ticket-cycle-") as tmp:
        root = Path(tmp)
        ai_root = root / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        mailbox.mkdir(parents=True)
        backlog = ai_root / "notes" / "backlog.md"
        backlog.write_text(backlog_text(), encoding="utf-8")
        architect_lane = root / "architect-lane"
        implementer_lane = root / "implementer-lane"
        sol_lane = root / "sol-lane"
        architect_lane.mkdir()
        implementer_lane.mkdir()
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
            "fable": str(architect_lane),
            "opus": str(implementer_lane),
            "sol": str(sol_lane),
        }
        daemon.report_demand = lambda backlog: None
        daemon.report_landing_debt = lambda: None
        daemon.warn_if_mailbox_unwatched = lambda: None
        daemon.git_commit_exists = lambda commit: commit in {
            BASE_A, BASE_B, BASE_C, BASE_D}
        descendants = {
            BASE_A: COMMIT_A,
            BASE_B: COMMIT_B,
            BASE_C: COMMIT_C,
            BASE_D: COMMIT_D,
        }
        daemon.git_commit_descends_from = (
            lambda starting_commit, accepted_commit:
            descendants.get(starting_commit) == accepted_commit)
        # This suite isolates ticket accounting from Git construction. Each
        # synthetic Opus admission makes its cycle suffix the exact current
        # main baseline, just as production admission requires.
        current_main = [BASE_A]
        production_register = daemon.register_ticket_cycle_message

        def register_with_synthetic_main(agent, message, **kwargs):
            if agent == "opus" and message.startswith(
                    daemon.MAILBOX_FLOW_HEADER):
                cycle_id, _mode, _body, problem = (
                    daemon._ticket_flow_envelope(message=message))
                if problem is None:
                    current_main[0] = cycle_id.rsplit("@", 1)[1]
            return production_register(
                agent=agent, message=message, **kwargs)

        daemon.register_ticket_cycle_message = register_with_synthetic_main
        daemon._exact_git_object = (
            lambda arguments, label: current_main[0])
        daemon.require_architect_landing_locked = (
            lambda cycle_id, landing_commit, ticket_state: landing_commit)
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


def raises_cycle_error(daemon, action, phrase, error_type=None):
    """Return whether one action fails closed with the expected reason."""
    expected = daemon.TicketCycleStateError if error_type is None else error_type
    try:
        action()
    except expected as exc:
        return phrase in str(exc)
    return False


def start_finite_controller(daemon, limit, topology="normal"):
    """Start or resume one durable finite watch in a scratch daemon."""
    controller = daemon.SafeKillRendezvous(
        ticket_cycle_limit=limit, ticket_cycle_topology=topology)
    daemon._ACTIVE_WATCH_RENDEZVOUS = controller
    restored = daemon.prepare_finite_watch_progress(
        limit=limit, topology=topology)
    controller.restore_completed_ticket_cycles(count=restored)
    return controller


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
            and state["active"][cycle_id] == {
                "phase": "implementation",
                "commit": None,
                "mode": "normal",
                "route": "primary",
                "ticket_class": "ordinary",
                "implementer_runtime": daemon.IMPLEMENTER_RUNTIME,
                "path_scope": None,
            }
            and controller.completed_ticket_cycles() == 0)
    print("A/I chatter is not a cycle=" + str(passed))
    return passed


def arm_normal_return_completes_one_cycle():
    """A matched Red Team return, not the commit, completes normal mode."""
    with scratch_daemon() as (daemon, mailbox):
        cycle_id = "normal-return@" + BASE_A
        controller = start_finite_controller(daemon, limit=1)
        try:
            register_normal_commit(daemon, cycle_id, COMMIT_A)
            after_commit = daemon.read_ticket_cycle_state()
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
            before_delivery = daemon.read_ticket_cycle_state()
            delivered = daemon.deliver_pending_ticket_cycle_returns()
            final = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        passed = (
            after_commit["generation"] == 0
            and after_commit["active"][cycle_id]["phase"]
            == "committed-awaiting-closure"
            and problem is None and Path(path) == receipt
            and result == "NO CHANGE" and completed
            and before_delivery["generation"] == 1
            and before_delivery["pending_cycle_returns"] == 1
            and delivered == 1
            and controller.completed_ticket_cycles() == 1
            and controller.ticket_cycle_limit_reached()
            and final["pending_cycle_returns"] == 0
            and final["completed"] == {cycle_id: COMMIT_A}
            and not final["active"])
    print("matched normal review completes one cycle=" + str(passed))
    return passed


def arm_two_role_commit_completes_one_cycle():
    """A primary two-role ticket completes at its accepted commit."""
    with scratch_daemon() as (daemon, _mailbox):
        cycle_id = "two-role-return@" + BASE_B
        daemon.register_ticket_cycle_message(
            agent="opus",
            message=flow_payload(
                cycle_id, "Primary Implementer start", mode="two-role"),
            skip_redteam=True)
        daemon.register_ticket_cycle_message(
            agent="fable",
            message=flow_payload(
                cycle_id, "Architect return", mode="two-role"),
            skip_redteam=True)
        completed = daemon.record_architect_commit(
            cycle_id=cycle_id, accepted_commit=COMMIT_B, mode="two-role")
        state = daemon.read_ticket_cycle_state()
        passed = (
            completed == 1 and state["generation"] == 1
            and state["pending_cycle_returns"] == 0
            and state["completed"] == {cycle_id: COMMIT_B}
            and not state["active"])
    print("two-role commit completes one cycle=" + str(passed))
    return passed


def arm_finite_capacity_reserves_before_completion():
    """A finite watch never admits more tickets than its remaining slots."""
    with scratch_daemon() as (daemon, _mailbox):
        first = "capacity-a@" + BASE_A
        later = "capacity-b@" + BASE_B
        later_message = flow_payload(later, "Later Implementer start")
        controller = start_finite_controller(daemon, limit=1)
        try:
            daemon.register_ticket_cycle_message(
                agent="opus",
                message=flow_payload(first, "First Implementer start"))
            deferred_while_active = raises_cycle_error(
                daemon,
                lambda: daemon.register_ticket_cycle_message(
                    agent="opus", message=later_message),
                "already reserved all 1 ticket cycle(s)",
                error_type=daemon.TicketCycleLimitDeferred)

            daemon.record_architect_commit(
                cycle_id=first, accepted_commit=COMMIT_A, mode="normal")
            register_closure(daemon, first, COMMIT_A)
            daemon.complete_ticket_cycle(first, COMMIT_A)
            deferred_while_pending = raises_cycle_error(
                daemon,
                lambda: daemon.register_ticket_cycle_message(
                    agent="opus", message=later_message),
                "already reserved all 1 ticket cycle(s)",
                error_type=daemon.TicketCycleLimitDeferred)
            pending = daemon.read_ticket_cycle_state()["pending_cycle_returns"]
            daemon.deliver_pending_ticket_cycle_returns()
            deferred_after_count = raises_cycle_error(
                daemon,
                lambda: daemon.register_ticket_cycle_message(
                    agent="opus", message=later_message),
                "already reserved all 1 ticket cycle(s)",
                error_type=daemon.TicketCycleLimitDeferred)

            # A restart before clean exit resumes the same finite limit.
            replacement = start_finite_controller(daemon, limit=1)
            still_deferred_after_restart = raises_cycle_error(
                daemon,
                lambda: daemon.register_ticket_cycle_message(
                    agent="opus", message=later_message),
                "already reserved all 1 ticket cycle(s)",
                error_type=daemon.TicketCycleLimitDeferred)
            daemon.finish_finite_watch_progress(
                limit=1, completed=1, topology="normal")

            # A later invocation after that proved exit starts a new limit.
            fresh = start_finite_controller(daemon, limit=1)
            daemon.register_ticket_cycle_message(
                agent="opus", message=later_message)
            final = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        passed = (
            deferred_while_active and deferred_while_pending
            and pending == 1 and deferred_after_count
            and controller.completed_ticket_cycles() == 1
            and replacement.completed_ticket_cycles() == 1
            and still_deferred_after_restart
            and fresh.completed_ticket_cycles() == 0
            and set(final["active"]) == {later}
            and final["active"][later]["phase"] == "implementation")
    print("finite limit reserves every admitted ticket=" + str(passed))
    return passed


def arm_finite_two_ticket_pipeline_counts_each_ticket_once():
    """A two-cycle watch may pipeline A and B, but must leave C untouched."""
    with scratch_daemon() as (daemon, _mailbox):
        first = "pipeline-a@" + BASE_A
        second = "pipeline-b@" + BASE_B
        deferred = "pipeline-c@" + BASE_C
        controller = start_finite_controller(daemon, limit=2)
        try:
            daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    first, "Implement candidate A"))
            daemon.register_ticket_cycle_message(
                agent="fable", message=flow_payload(
                    first, "Audit candidate A while B starts"))
            daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    second, "Implement candidate B"))
            after_two_admissions = daemon.read_ticket_cycle_state()
            c_blocked = raises_cycle_error(
                daemon,
                lambda: daemon.register_ticket_cycle_message(
                    agent="opus", message=flow_payload(
                        deferred, "Must wait for another watch")),
                "already reserved all 2 ticket cycle(s)",
                error_type=daemon.TicketCycleLimitDeferred)

            register_normal_commit(daemon, first, COMMIT_A)
            register_closure(daemon, first, COMMIT_A)
            first_completed = daemon.complete_ticket_cycle(first, COMMIT_A)
            first_delivery = daemon.deliver_pending_ticket_cycle_returns()
            after_first = daemon.read_ticket_cycle_state()
            c_still_blocked = raises_cycle_error(
                daemon,
                lambda: daemon.register_ticket_cycle_message(
                    agent="opus", message=flow_payload(
                        deferred, "Still belongs to another watch")),
                "already reserved all 2 ticket cycle(s)",
                error_type=daemon.TicketCycleLimitDeferred)

            register_normal_commit(daemon, second, COMMIT_B)
            register_closure(daemon, second, COMMIT_B)
            second_completed = daemon.complete_ticket_cycle(
                second, COMMIT_B)
            second_delivery = daemon.deliver_pending_ticket_cycle_returns()
            final = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        passed = (
            set(after_two_admissions["active"]) == {first, second}
            and after_two_admissions["generation"] == 0
            and c_blocked and first_completed and first_delivery == 1
            and controller.completed_ticket_cycles() == 2
            and after_first["generation"] == 1
            and after_first["completed"] == {first: COMMIT_A}
            and set(after_first["active"]) == {second}
            and c_still_blocked and second_completed
            and second_delivery == 1
            and final["generation"] == 2
            and final["completed"] == {
                first: COMMIT_A, second: COMMIT_B}
            and not final["active"]
            and controller.ticket_cycle_limit_reached())
    print("finite two-ticket pipeline counts A and B exactly once="
          + str(passed))
    return passed


def arm_no_go_preserves_later_ticket_for_replay():
    """Architect NO-GO leaves both A repair work and pipelined B intact."""
    with scratch_daemon() as (daemon, _mailbox):
        repair = "repair-a@" + BASE_A
        later = "repair-b@" + BASE_B
        deferred = "repair-c@" + BASE_C
        start_finite_controller(daemon, limit=2)
        try:
            daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    repair, "Implement candidate A"))
            daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    later, "Implement independent candidate B"))
            before_no_go = daemon.read_ticket_cycle_state()

            # GO/NO-GO is the Architect's decision inside its flow message.
            # Only a later exact architect-commit receipt changes cycle
            # state. Therefore NO-GO requests repair without completing A or
            # mutating B.
            daemon.register_ticket_cycle_message(
                agent="fable", message=flow_payload(
                    repair,
                    "NO-GO. Repair A from its saved candidate and base."))
            after_no_go = daemon.read_ticket_cycle_state()
            c_blocked = raises_cycle_error(
                daemon,
                lambda: daemon.register_ticket_cycle_message(
                    agent="opus", message=flow_payload(
                        deferred, "Must not pass the finite boundary")),
                "already reserved all 2 ticket cycle(s)",
                error_type=daemon.TicketCycleLimitDeferred)
            wrong_commit_fails = raises_cycle_error(
                daemon,
                lambda: daemon.record_architect_commit(
                    cycle_id=repair, accepted_commit=COMMIT_B,
                    mode="normal"),
                "not a new descendant")
            final = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        passed = (
            before_no_go == after_no_go == final
            and set(final["active"]) == {repair, later}
            and all(record["phase"] == "implementation"
                    for record in final["active"].values())
            and final["generation"] == 0
            and not final["completed"]
            and c_blocked and wrong_commit_fails)
    print("NO-GO preserves A repair and later ticket B=" + str(passed))
    return passed


def arm_retired_schema_three_refuses_without_rewrite():
    """Retired emergency state stops; the daemon never guesses a rewrite."""
    with scratch_daemon() as (daemon, _mailbox):
        primary = "schema-primary@" + BASE_A
        second = "schema-second@" + BASE_B
        accepted = "schema-accepted@" + BASE_C
        legacy = {
            "schema": 3,
            "generation": 4,
            "pending_cycle_returns": 1,
            "emergency_epoch": 8,
            "emergency_condition": True,
            "active": {
                primary: {
                    "phase": "implementation", "commit": None,
                    "mode": "emergency-primary", "route": "primary",
                    "epoch": 8,
                },
                second: {
                    "phase": "implementation", "commit": None,
                    "mode": "emergency-second", "route": "second",
                    "epoch": 8,
                },
                accepted: {
                    "phase": "emergency-committed", "commit": COMMIT_C,
                    "mode": "emergency-primary", "route": "primary",
                    "epoch": 7,
                },
            },
            "completed": {},
            "emergency_commits": [
                {"cycle": accepted, "commit": COMMIT_C,
                 "implementer": "primary", "epoch": 7},
            ],
        }
        state_path = Path(daemon.ticket_cycle_state_path())
        state_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
        before = state_path.read_bytes()
        passed = raises_cycle_error(
            daemon, daemon.read_ticket_cycle_state,
            "unsupported old schema")
        passed = passed and state_path.read_bytes() == before
    print("retired schema-three state refuses without rewrite=" + str(passed))
    return passed


def arm_corrupt_schema_three_fails_closed():
    """Malformed retired state receives the same fail-closed boundary."""
    with scratch_daemon() as (daemon, _mailbox):
        corrupt = {
            "schema": 3,
            "generation": 0,
            "pending_cycle_returns": 0,
            "emergency_epoch": 0,
            "emergency_condition": False,
            "active": {},
            "completed": {},
            "emergency_commits": [{"garbage": "accepted"}],
        }
        Path(daemon.ticket_cycle_state_path()).write_text(
            json.dumps(corrupt) + "\n", encoding="utf-8")
        passed = raises_cycle_error(
            daemon, daemon.read_ticket_cycle_state,
            "unsupported old schema")
    print("corrupt schema-three state fails closed=" + str(passed))
    return passed


def arm_topology_counts_only_enabled_active_tickets():
    """Normal and two-role watches count only their saved ticket mode."""
    with scratch_daemon() as (daemon, _mailbox):
        normal = "role-chatter@" + BASE_A
        primary = "two-role-return@" + BASE_B
        daemon.register_ticket_cycle_message(
            agent="opus", message=flow_payload(normal, "Normal"))
        daemon.register_ticket_cycle_message(
            agent="opus",
            message=flow_payload(primary, "Primary", mode="two-role"),
            skip_redteam=True)
        passed = (
            daemon.active_ticket_cycle_count() == 1
            and daemon.active_ticket_cycle_count(skip_redteam=True) == 1)
    print("active counts follow the selected topology=" + str(passed))
    return passed


def arm_incompatible_roots_remain_untouched():
    """A normal watch never claims saved no-Red-Team work."""
    with scratch_daemon() as (daemon, mailbox):
        cycle = "two-role-return@" + BASE_B
        payloads = {
            "0001-to-opus.md": flow_payload(
                cycle, "Saved primary work", mode="two-role"),
            "0003-to-daemon.md": daemon.architect_go_request_payload(
                cycle_id=cycle, candidate_commit=COMMIT_B, mode="two-role"),
        }
        before = {}
        for name, payload in payloads.items():
            path = mailbox / name
            path.write_text(payload, encoding="utf-8", newline="")
            before[name] = path.read_bytes()
        outcome = daemon.process_backlog(dry_run=False)
        passed = outcome is None and all(
            (mailbox / name).read_bytes() == content
            for name, content in before.items())
        passed = passed and not (mailbox / "inflight").exists()
    print("incompatible root messages stay untouched=" + str(passed))
    return passed


def arm_placeholder_never_reserves_a_slot():
    """A template handoff cannot poison a finite ticket reservation."""
    with scratch_daemon() as (daemon, mailbox):
        path = mailbox / "0001-to-opus.md"
        path.write_text(
            flow_payload("capacity-a@" + BASE_A, "<unit>"),
            encoding="utf-8", newline="")
        controller = start_finite_controller(daemon, limit=1)
        deferred, reservation = daemon.reserve_implementer_ticket_before_claim(
            path=str(path))
        state = daemon.read_ticket_cycle_state()
        passed = (
            deferred is None and reservation is None and not state["active"]
            and controller.completed_ticket_cycles() == 0
            and path.is_file())
    print("placeholder handoff reserves no cycle=" + str(passed))
    return passed


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
            "belongs to another watch role")

        legacy_mode_fails = raises_cycle_error(
            daemon,
            lambda: daemon.register_ticket_cycle_message(
                agent="opus", message=flow_payload(
                    "schema-second@" + BASE_C,
                    "Removed mode", mode="emergency-second")),
            "needs exact MAILBOX-FLOW")
        state = daemon.read_ticket_cycle_state()
        passed = (
            invented_fails and architect_fails and changed_mode_fails
            and legacy_mode_fails and set(state["active"]) == {cycle_id}
            and state["active"][cycle_id]["mode"] == "normal")
    print("mode, route, and anchor spoofing fails=" + str(passed))
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
                cycle_id=cycle_id,
                accepted_commit=COMMIT_A,
                mode="normal"),
            "not a new descendant")
    print("accepted commit must descend from base=" + str(passed))
    return passed


def arm_crash_replays_one_pending_cycle_return():
    """A crash between durable completion and the counter loses no cycle."""
    with scratch_daemon() as (daemon, _mailbox):
        cycle_id = "normal-return@" + BASE_A
        first_controller = start_finite_controller(daemon, limit=1)
        try:
            register_normal_commit(daemon, cycle_id, COMMIT_A)
            register_closure(daemon, cycle_id, COMMIT_A)
            completed = daemon.complete_ticket_cycle(cycle_id, COMMIT_A)
            before_crash = daemon.read_ticket_cycle_state()
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None

        replacement = start_finite_controller(daemon, limit=1)
        try:
            daemon.reconcile_ticket_cycle_state()
            delivered = daemon.deliver_pending_ticket_cycle_returns()
            delivered_again = daemon.deliver_pending_ticket_cycle_returns()
            after_restart = daemon.read_ticket_cycle_state()
            # Crash again after RAM delivery but before the clean exit.
            second_replacement = start_finite_controller(daemon, limit=1)
            extra = "capacity-b@" + BASE_B
            blocked_after_second_crash = raises_cycle_error(
                daemon,
                lambda: daemon.register_ticket_cycle_message(
                    agent="opus",
                    message=flow_payload(extra, "Must remain deferred")),
                "already reserved all 1 ticket cycle(s)",
                error_type=daemon.TicketCycleLimitDeferred)
        finally:
            daemon._ACTIVE_WATCH_RENDEZVOUS = None
        passed = (
            completed is True
            and first_controller.completed_ticket_cycles() == 0
            and before_crash["pending_cycle_returns"] == 1
            and delivered == 1 and delivered_again == 0
            and replacement.completed_ticket_cycles() == 1
            and replacement.ticket_cycle_limit_reached()
            and after_restart["pending_cycle_returns"] == 0
            and after_restart["finite_watch"] == {
                "limit": 1, "completed": 1, "status": "active",
                "topology": "normal"}
            and second_replacement.completed_ticket_cycles() == 1
            and blocked_after_second_crash)
    print("crash replays one pending cycle return=" + str(passed))
    return passed


def arm_safe_stop_never_counts_cycle():
    """Child cadence and its manual window do not change ticket cycles."""
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


def arm_removed_sol_flag_is_rejected():
    """The removed Sol implementation option is not part of the CLI."""
    with scratch_daemon() as (daemon, _mailbox):
        rc_plain, _plain, errors_plain, error_plain = call_main(
            daemon, ["--sol_as_implementer"])
        passed = (
            rc_plain == 2 and isinstance(error_plain, SystemExit)
            and "unrecognized arguments: --sol_as_implementer"
            in errors_plain)
    print("removed Sol implementation flag is rejected=" + str(passed))
    return passed


def main():
    """Run every focused check and return nonzero on any regression."""
    checks = [
        ("role chatter", arm_role_chatter_does_not_complete_cycle),
        ("normal return", arm_normal_return_completes_one_cycle),
        ("two-role return", arm_two_role_commit_completes_one_cycle),
        ("finite capacity", arm_finite_capacity_reserves_before_completion),
        ("finite two-ticket pipeline",
         arm_finite_two_ticket_pipeline_counts_each_ticket_once),
        ("NO-GO pipeline preservation",
         arm_no_go_preserves_later_ticket_for_replay),
        ("retired schema-three refusal",
         arm_retired_schema_three_refuses_without_rewrite),
        ("schema-three corruption", arm_corrupt_schema_three_fails_closed),
        ("topology-aware active count",
         arm_topology_counts_only_enabled_active_tickets),
        ("topology root deferral", arm_incompatible_roots_remain_untouched),
        ("placeholder reservation", arm_placeholder_never_reserves_a_slot),
        ("mode/route/anchor spoofing",
         arm_mode_route_and_anchor_spoofing_fails),
        ("commit ancestry", arm_accepted_commit_must_descend_from_base),
        ("crash-safe cycle return",
         arm_crash_replays_one_pending_cycle_return),
        ("safe-stop separation", arm_safe_stop_never_counts_cycle),
        ("removed Sol CLI", arm_removed_sol_flag_is_rejected),
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
