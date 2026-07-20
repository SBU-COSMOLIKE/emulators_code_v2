"""Check the small rules that distinguish protected tickets from ordinary ones."""

import copy
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from ai.tools import mailbox_daemon as daemon


CYCLE = "protect-router@" + "1" * 40
OTHER_CYCLE = "protect-router@" + "2" * 40
CANDIDATE = "3" * 40
OTHER_CANDIDATE = "4" * 40


def load_isolated_daemon(root):
  """Load the live daemon code with every state path under ``root``."""
  root = Path(root)
  tools = root / "ai/tools"
  notes = root / "ai/notes"
  tools.mkdir(parents=True, exist_ok=True)
  notes.mkdir(parents=True, exist_ok=True)
  source_tools = Path(daemon.__file__).resolve().parent
  shutil.copy2(source_tools / "mailbox_daemon.py", tools)
  shutil.copy2(source_tools / "role_contract.py", tools)
  for name in (
      "candidate_admission.py", "provider_health.py", "reopen_transition.py",
      "review_dispatch.py"):
    shutil.copy2(source_tools / name, tools)
  shutil.copy2(
      source_tools.parent / "notes/role-contract.yaml",
      notes / "role-contract.yaml")
  spec = importlib.util.spec_from_file_location(
      "isolated_protected_ticket_daemon", tools / "mailbox_daemon.py")
  isolated = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(isolated)
  return isolated


class ProtectedPathTests(unittest.TestCase):
  """Keep ordinary work away from the trusted control-plane files."""

  def test_ordinary_ticket_cannot_change_a_trusted_tool(self):
    """A ticket-local allow-list cannot override global protection."""
    path = "ai/tools/mailbox_daemon.py"

    self.assertEqual(
        daemon.classify_candidate_scope(
            changed_paths={path}, path_scope={path}, ticket_class="ordinary"),
        ("PROTECTED_PATH_VIOLATION", {path}))

  def test_retired_protected_ticket_cannot_change_a_trusted_tool(self):
    """A ticket label cannot reopen external-maintainer-only tools."""
    path = "ai/tools/mailbox_daemon.py"

    self.assertEqual(
        daemon.classify_candidate_scope(
            changed_paths={path}, path_scope={path},
            ticket_class="protected-control-plane"),
        ("PROTECTED_PATH_VIOLATION", {path}))

  def test_tool_violation_wins_over_an_ordinary_scope_expansion(self):
    """The tool refusal is stricter than an undeclared ordinary path."""
    declared = "ai/tools/mailbox_daemon.py"
    extra = "emulator/model.py"

    self.assertEqual(
        daemon.classify_candidate_scope(
            changed_paths={declared, extra}, path_scope={declared},
            ticket_class="protected-control-plane"),
        ("PROTECTED_PATH_VIOLATION", {declared}))

  def test_protected_class_does_not_open_always_forbidden_files(self):
    """The backlog remains Architect-owned even during protected work."""
    path = "ai/notes/backlog.md"

    self.assertEqual(
        daemon.classify_candidate_scope(
            changed_paths={path}, path_scope={path},
            ticket_class="protected-control-plane"),
        ("PROTECTED_PATH_VIOLATION", {path}))
    self.assertNotIn(path, daemon.ARCHITECT_PROTECTED_POLICY_PATHS)

  def test_protected_class_keeps_policy_files_architect_only(self):
    """A protected candidate cannot edit permanent memory or authority."""
    for path in (
        "ai/notes/MEMORY.md",
        "ai/notes/role-contract.yaml",
        "ai/notes/implementer-failure-modes.yaml",
        ".claude/FABLE_ROLE.md",
    ):
      with self.subTest(path=path):
        self.assertEqual(
            daemon.classify_candidate_scope(
                changed_paths={path}, path_scope={path},
                ticket_class="protected-control-plane"),
            ("PROTECTED_PATH_VIOLATION", {path}))

  def test_unknown_ticket_class_fails_closed(self):
    """A typo or invented class cannot gain protected-file authority."""
    path = "ai/tools/mailbox_daemon.py"

    with self.assertRaisesRegex(
        daemon.TicketCycleStateError, "invalid ticket class"):
      daemon.classify_candidate_scope(
          changed_paths={path}, path_scope={path}, ticket_class="invented")

  def test_protected_class_is_blocked_with_or_without_redteam(self):
    self.assertIsNone(daemon.ticket_class_configuration_problem(
        ticket_class="ordinary", skip_redteam=True))
    for skip_redteam in (False, True):
      with self.subTest(skip_redteam=skip_redteam):
        problem = daemon.ticket_class_configuration_problem(
            ticket_class="protected-control-plane",
            skip_redteam=skip_redteam)
        self.assertIn("retired", problem)
        self.assertIn("keep the ticket Open", problem)

  def test_landing_rechecks_ai_tools_even_for_old_saved_candidates(self):
    with mock.patch.object(
        daemon, "candidate_changed_paths",
        return_value={"ai/tools/mailbox_daemon.py"}):
      with self.assertRaisesRegex(
          daemon.TicketCycleStateError, "cannot land"):
        daemon.prepare_exact_squash_landing(
            cycle_id=CYCLE, candidate_commit=CANDIDATE, mode="normal")


class ControlPlaneMessageTests(unittest.TestCase):
  """Bind each mandatory Red Team message to one cycle and candidate."""

  def test_review_request_round_trip_preserves_exact_identity(self):
    request = daemon.sol_ticket_payload(
        ticket_kind="control-plane", text="Review this candidate.",
        review_cycle=CYCLE, review_commit=CANDIDATE)

    self.assertEqual(daemon.sol_ticket_kind(request), "control-plane")
    self.assertEqual(
        daemon._redteam_control_plane_envelope(request),
        (CYCLE, CANDIDATE, "Review this candidate.\n", None))

  def test_review_request_refuses_incomplete_identity(self):
    for cycle, candidate in (
        (None, CANDIDATE), (CYCLE, None), ("not-a-cycle", CANDIDATE),
        (CYCLE, "short")):
      with self.subTest(cycle=cycle, candidate=candidate):
        with self.assertRaises(ValueError):
          daemon.sol_ticket_payload(
              ticket_kind="control-plane", text="Review.",
              review_cycle=cycle, review_commit=candidate)

  def test_accept_and_reject_receipts_round_trip(self):
    for result in daemon.CONTROL_PLANE_REVIEW_RESULTS:
      with self.subTest(result=result):
        receipt = daemon.control_plane_review_receipt_payload(
            review_cycle=CYCLE, candidate=CANDIDATE, result=result,
            text="Bounded evidence.")
        self.assertEqual(
            daemon._control_plane_review_receipt(receipt),
            (CYCLE, CANDIDATE, result, "Bounded evidence.\n", None))

  def test_receipt_rejects_an_invalid_cycle_or_candidate(self):
    cases = (
        ("not-a-cycle", CANDIDATE),
        (CYCLE, "not-a-full-commit"),
    )
    for cycle, candidate in cases:
      with self.subTest(cycle=cycle, candidate=candidate):
        with self.assertRaises(ValueError):
          daemon.control_plane_review_receipt_payload(
              review_cycle=cycle, candidate=candidate,
              result="ACCEPT-CONTROL-PLANE", text="Evidence.")

  def test_receipt_parser_exposes_a_wrong_but_well_formed_identity(self):
    """The state transition can compare full values instead of trusting prose."""
    receipt = daemon.control_plane_review_receipt_payload(
        review_cycle=OTHER_CYCLE, candidate=OTHER_CANDIDATE,
        result="ACCEPT-CONTROL-PLANE", text="Evidence.")

    cycle, candidate, result, _body, problem = (
        daemon._control_plane_review_receipt(receipt))
    self.assertIsNone(problem)
    self.assertNotEqual((cycle, candidate), (CYCLE, CANDIDATE))
    self.assertEqual(result, "ACCEPT-CONTROL-PLANE")

  def test_duplicate_identity_header_is_not_treated_as_evidence(self):
    receipt = daemon.control_plane_review_receipt_payload(
        review_cycle=CYCLE, candidate=CANDIDATE,
        result="ACCEPT-CONTROL-PLANE",
        text="MAILBOX-CANDIDATE: " + OTHER_CANDIDATE)

    self.assertEqual(
        daemon._control_plane_review_receipt(receipt)[4],
        "duplicate control-plane receipt")


class ProtectedStateTests(unittest.TestCase):
  """Keep protected decisions durable without breaking older schema-6 work."""

  def test_candidate_cannot_write_the_architect_key_without_d0_receipt(self):
    """D0 refuses the first key unless its own Architect-turn proof exists."""
    with tempfile.TemporaryDirectory(prefix="protected-ticket-d0-") as tmp:
      isolated = load_isolated_daemon(tmp)
      state = isolated.empty_ticket_cycle_state()
      state["active"][CYCLE] = {
          "phase": "implementation",
          "commit": None,
          "mode": "normal",
          "route": "primary",
          "ticket_class": "protected-control-plane",
          "control_plane": isolated.empty_control_plane_state(),
      }
      isolated.write_ticket_cycle_state(state)
      candidates = isolated.empty_candidate_state()
      candidates["cycles"][CYCLE] = {
          "ref": isolated.cycle_candidate_ref(CYCLE),
          "commit": CANDIDATE,
      }
      isolated.write_candidate_state(candidates)

      with self.assertRaisesRegex(
          isolated.TicketCycleStateError, "D0-validated Architect-turn"):
        isolated.record_control_plane_architect_go(CYCLE, CANDIDATE)
      saved = isolated.read_ticket_cycle_state()["active"][CYCLE]
      self.assertIsNone(
          saved["control_plane"]["architect_candidate"])

  def test_landing_requires_two_accepting_keys_for_the_same_candidate(self):
    """Neither one key nor two keys naming different commits can make L."""
    cases = (
        (None, None, None, False),
        (CANDIDATE, None, None, False),
        (None, CANDIDATE, "ACCEPT-CONTROL-PLANE", False),
        (CANDIDATE, CANDIDATE, "REJECT-CONTROL-PLANE", False),
        (CANDIDATE, OTHER_CANDIDATE, "ACCEPT-CONTROL-PLANE", False),
        (CANDIDATE, CANDIDATE, "ACCEPT-CONTROL-PLANE", True),
    )
    for architect, redteam, result, expected in cases:
      with self.subTest(
          architect=architect, redteam=redteam, result=result):
        control = daemon.empty_control_plane_state()
        control["architect_candidate"] = architect
        control["redteam_candidate"] = redteam
        control["redteam_result"] = result
        self.assertIs(
            daemon.control_plane_keys_ready(
                control=control, candidate_commit=CANDIDATE),
            expected)

  def test_acceptance_for_one_candidate_cannot_authorize_another(self):
    control = daemon.empty_control_plane_state()
    control["architect_candidate"] = CANDIDATE
    control["redteam_candidate"] = CANDIDATE
    control["redteam_result"] = "ACCEPT-CONTROL-PLANE"

    self.assertFalse(daemon.control_plane_keys_ready(
        control=control, candidate_commit=OTHER_CANDIDATE))

  def test_raw_receipt_cannot_create_the_redteam_key(self):
    """Only the D0-observed Sol turn may persist the second key."""
    control = daemon.empty_control_plane_state()
    self.assertFalse(daemon.control_plane_redteam_key_matches(
        control=control, candidate_commit=CANDIDATE,
        decision="ACCEPT-CONTROL-PLANE"))

    control["redteam_candidate"] = CANDIDATE
    control["redteam_result"] = "ACCEPT-CONTROL-PLANE"
    self.assertTrue(daemon.control_plane_redteam_key_matches(
        control=control, candidate_commit=CANDIDATE,
        decision="ACCEPT-CONTROL-PLANE"))

  def test_state_update_refuses_a_wrong_cycle_or_candidate(self):
    """Neither a valid-looking foreign cycle nor foreign C enters the state."""
    with tempfile.TemporaryDirectory(prefix="protected-ticket-state-") as tmp:
      isolated = load_isolated_daemon(tmp)
      state = isolated.empty_ticket_cycle_state()
      control = isolated.empty_control_plane_state()
      control["architect_candidate"] = CANDIDATE
      state["active"][CYCLE] = {
          "phase": "implementation",
          "commit": None,
          "mode": "normal",
          "route": "primary",
          "ticket_class": "protected-control-plane",
          "control_plane": control,
      }
      isolated.write_ticket_cycle_state(state)
      candidates = isolated.empty_candidate_state()
      candidates["cycles"][CYCLE] = {
          "ref": isolated.cycle_candidate_ref(CYCLE),
          "commit": CANDIDATE,
      }
      isolated.write_candidate_state(candidates)

      for cycle, candidate in (
          (OTHER_CYCLE, CANDIDATE), (CYCLE, OTHER_CANDIDATE)):
        with self.subTest(cycle=cycle, candidate=candidate):
          with self.assertRaisesRegex(
              isolated.TicketCycleStateError, "does not name active C"):
            isolated.record_control_plane_redteam_decision(
                cycle_id=cycle, candidate_commit=candidate,
                decision="ACCEPT-CONTROL-PLANE")

      isolated.record_control_plane_redteam_decision(
          cycle_id=CYCLE, candidate_commit=CANDIDATE,
          decision="ACCEPT-CONTROL-PLANE")
      saved = isolated.read_ticket_cycle_state()["active"][CYCLE]
      self.assertEqual(
          saved["control_plane"]["redteam_candidate"], CANDIDATE)

  def test_old_schema_six_active_ticket_defaults_to_ordinary(self):
    """A ticket saved before this feature continues under ordinary rules."""
    state = daemon.empty_ticket_cycle_state()
    del state["control_plane_history"]
    state["active"][CYCLE] = {
        "phase": "implementation",
        "commit": None,
        "mode": "normal",
        "route": "primary",
    }

    normalized = daemon.validate_ticket_cycle_state(state)

    self.assertEqual(
        normalized["active"][CYCLE]["ticket_class"], "ordinary")
    self.assertNotIn("control_plane", normalized["active"][CYCLE])
    self.assertEqual(normalized["control_plane_history"], {})

  def test_restart_preserves_approval_and_shadow_evidence(self):
    """Reloading D0 preserves the exact approvals behind a shadow pass."""
    with tempfile.TemporaryDirectory(prefix="protected-ticket-restart-") as tmp:
      isolated = load_isolated_daemon(tmp)
      control = isolated.empty_control_plane_state()
      control["architect_candidate"] = CANDIDATE
      control["redteam_candidate"] = CANDIDATE
      control["redteam_result"] = "ACCEPT-CONTROL-PLANE"
      control["shadow_status"] = "PASSED"
      control["shadow_evidence"] = "relay/shadow.log"
      state = isolated.empty_ticket_cycle_state()
      state["active"][CYCLE] = {
          "phase": "implementation",
          "commit": None,
          "mode": "normal",
          "route": "primary",
          "ticket_class": "protected-control-plane",
          "control_plane": control,
      }
      isolated.write_ticket_cycle_state(state)

      restarted = load_isolated_daemon(tmp)
      saved = restarted.read_ticket_cycle_state()["active"][CYCLE]
      self.assertEqual(
          saved["control_plane"]["architect_candidate"], CANDIDATE)
      self.assertEqual(saved["control_plane"]["shadow_status"], "PASSED")
      self.assertEqual(
          saved["control_plane"]["redteam_result"],
          "ACCEPT-CONTROL-PLANE")
      self.assertTrue(restarted.control_plane_keys_ready(
          control=saved["control_plane"], candidate_commit=CANDIDATE))

  def test_shadow_and_health_checks_require_the_saved_candidate(self):
    """Neither check may attach evidence to a different candidate commit."""
    with tempfile.TemporaryDirectory(prefix="protected-ticket-check-") as tmp:
      isolated = load_isolated_daemon(tmp)
      control = isolated.empty_control_plane_state()
      control["architect_candidate"] = CANDIDATE
      control["redteam_candidate"] = CANDIDATE
      control["redteam_result"] = "ACCEPT-CONTROL-PLANE"
      state = isolated.empty_ticket_cycle_state()
      state["active"][CYCLE] = {
          "phase": "implementation",
          "commit": None,
          "mode": "normal",
          "route": "primary",
          "ticket_class": "protected-control-plane",
          "control_plane": control,
      }
      isolated.write_ticket_cycle_state(state)
      candidates = isolated.empty_candidate_state()
      candidates["cycles"][CYCLE] = {
          "ref": isolated.cycle_candidate_ref(CYCLE),
          "commit": CANDIDATE,
      }
      isolated.write_candidate_state(candidates)

      for kind in ("shadow", "health"):
        with self.subTest(kind=kind):
          with self.assertRaisesRegex(
              isolated.TicketCycleStateError,
              "exact saved candidate C"):
            isolated.record_control_plane_check(
                cycle_id=CYCLE, candidate_commit=OTHER_CANDIDATE,
                kind=kind, ok=True, evidence="relay/check.log")

      saved = isolated.read_ticket_cycle_state()["active"][CYCLE]
      self.assertIsNone(saved["control_plane"]["shadow_status"])
      self.assertIsNone(saved["control_plane"]["health_status"])

  def test_active_state_refuses_incoherent_protected_results(self):
    """Active protected state must bind each result to evidence and exact C."""
    base = daemon.empty_ticket_cycle_state()
    control = daemon.empty_control_plane_state()
    control["architect_candidate"] = CANDIDATE
    control["redteam_candidate"] = CANDIDATE
    control["redteam_result"] = "ACCEPT-CONTROL-PLANE"
    base["active"][CYCLE] = {
        "phase": "implementation",
        "commit": None,
        "mode": "normal",
        "route": "primary",
        "ticket_class": "protected-control-plane",
        "control_plane": control,
    }
    daemon.validate_ticket_cycle_state(base)

    cases = []
    wrong_candidate = copy.deepcopy(base)
    wrong_candidate["active"][CYCLE]["control_plane"][
        "redteam_candidate"] = OTHER_CANDIDATE
    cases.append((wrong_candidate, "same exact C"))

    shadow_without_evidence = copy.deepcopy(base)
    shadow_without_evidence["active"][CYCLE]["control_plane"][
        "shadow_status"] = "FAILED"
    cases.append((shadow_without_evidence, "shadow result and evidence"))

    health_without_evidence = copy.deepcopy(base)
    health_control = health_without_evidence["active"][CYCLE]["control_plane"]
    health_control["shadow_status"] = "PASSED"
    health_control["shadow_evidence"] = "relay/shadow.log"
    health_control["health_status"] = "CONTROL_PLANE_HEALTH_FAILED"
    cases.append((health_without_evidence, "health result and evidence"))

    revalidated_without_evidence = copy.deepcopy(base)
    integration = revalidated_without_evidence["active"][CYCLE][
        "control_plane"]
    integration["integration_status"] = "REVALIDATED"
    integration["integration_main"] = "5" * 40
    integration["stale_landing"] = "6" * 40
    integration["stale_parent"] = "7" * 40
    cases.append((revalidated_without_evidence,
                  "revalidation lacks evidence"))

    for malformed, message in cases:
      with self.subTest(message=message):
        with self.assertRaisesRegex(
            daemon.TicketCycleStateError, message):
          daemon.validate_ticket_cycle_state(malformed)

  def test_completed_history_requires_exact_accepted_healthy_candidate(self):
    """Completed history cannot preserve partial or contradictory evidence."""
    landing = "5" * 40
    control = daemon.empty_control_plane_state()
    control["architect_candidate"] = CANDIDATE
    control["redteam_candidate"] = CANDIDATE
    control["redteam_result"] = "ACCEPT-CONTROL-PLANE"
    control["shadow_status"] = "PASSED"
    control["shadow_evidence"] = "relay/shadow.log"
    control["health_status"] = "HEALTHY"
    control["health_evidence"] = "relay/health.log"
    base = daemon.empty_ticket_cycle_state()
    base["completed"][CYCLE] = landing
    base["control_plane_history"][CYCLE] = {
        "candidate": CANDIDATE,
        "landing": landing,
        "control_plane": control,
    }
    daemon.validate_ticket_cycle_state(base)

    cases = []
    wrong_candidate = copy.deepcopy(base)
    wrong_candidate["control_plane_history"][CYCLE]["control_plane"][
        "architect_candidate"] = OTHER_CANDIDATE
    cases.append((wrong_candidate, "exact C"))

    failed_shadow = copy.deepcopy(base)
    failed_shadow["control_plane_history"][CYCLE]["control_plane"][
        "shadow_status"] = "FAILED"
    cases.append((failed_shadow, "PASSED shadow evidence"))

    failed_health = copy.deepcopy(base)
    failed_health["control_plane_history"][CYCLE]["control_plane"][
        "health_status"] = "CONTROL_PLANE_HEALTH_FAILED"
    cases.append((failed_health, "HEALTHY evidence"))

    missing_health_evidence = copy.deepcopy(base)
    missing_health_evidence["control_plane_history"][CYCLE]["control_plane"][
        "health_evidence"] = None
    cases.append((missing_health_evidence, "health result and evidence"))

    stale_integration = copy.deepcopy(base)
    stale_control = stale_integration["control_plane_history"][CYCLE][
        "control_plane"]
    stale_control["integration_status"] = "STALE"
    stale_control["integration_main"] = "6" * 40
    stale_control["stale_landing"] = "7" * 40
    stale_control["stale_parent"] = "8" * 40
    cases.append((stale_integration, "stale"))

    for malformed, message in cases:
      with self.subTest(message=message):
        with self.assertRaisesRegex(
            daemon.TicketCycleStateError, message):
          daemon.validate_ticket_cycle_state(malformed)

  def test_protected_ticket_requires_the_complete_two_key_record(self):
    state = daemon.empty_ticket_cycle_state()
    state["active"][CYCLE] = {
        "phase": "implementation",
        "commit": None,
        "mode": "normal",
        "route": "primary",
        "ticket_class": "protected-control-plane",
        "control_plane": {},
    }

    with self.assertRaisesRegex(
        daemon.TicketCycleStateError, "exact two-key state"):
      daemon.validate_ticket_cycle_state(state)

  def test_ordinary_ticket_cannot_carry_protected_decisions(self):
    state = daemon.empty_ticket_cycle_state()
    state["active"][CYCLE] = {
        "phase": "implementation",
        "commit": None,
        "mode": "normal",
        "route": "primary",
        "ticket_class": "ordinary",
        "control_plane": daemon.empty_control_plane_state(),
    }

    with self.assertRaisesRegex(
        daemon.TicketCycleStateError, "unexpectedly has protected state"):
      daemon.validate_ticket_cycle_state(state)

  def test_health_failure_is_a_durable_state_not_invalid_json(self):
    control = daemon.empty_control_plane_state()
    control["architect_candidate"] = CANDIDATE
    control["redteam_candidate"] = CANDIDATE
    control["redteam_result"] = "ACCEPT-CONTROL-PLANE"
    control["shadow_status"] = "PASSED"
    control["shadow_evidence"] = "relay/shadow.log"
    control["health_status"] = "CONTROL_PLANE_HEALTH_FAILED"
    control["health_evidence"] = "relay/health.log"
    state = daemon.empty_ticket_cycle_state()
    state["active"][CYCLE] = {
        "phase": "committed-awaiting-closure",
        "commit": "5" * 40,
        "mode": "normal",
        "route": "primary",
        "ticket_class": "protected-control-plane",
        "control_plane": control,
    }

    normalized = daemon.validate_ticket_cycle_state(state)

    saved = normalized["active"][CYCLE]["control_plane"]
    self.assertEqual(
        saved["health_status"], "CONTROL_PLANE_HEALTH_FAILED")
    self.assertEqual(saved["health_evidence"], "relay/health.log")

  def test_skip_redteam_does_not_requeue_blocked_protected_work(self):
    """A same-topology restart leaves the durable blocked queue alone."""
    self.assertEqual(
        daemon.recover_blocked_redteam_messages(skip_redteam=True), 0)

  def test_restart_never_requeues_old_protected_work(self):
    """Historical bytes stay parked for external maintenance."""
    with tempfile.TemporaryDirectory(prefix="protected-ticket-blocked-") as tmp:
      isolated = load_isolated_daemon(tmp)
      blocked = Path(isolated.blocked_redteam_directory())
      blocked.mkdir(parents=True)
      saved = blocked / "0001-to-opus.md"
      payload = b"exact protected request\n"
      saved.write_bytes(payload)

      self.assertEqual(
          isolated.recover_blocked_redteam_messages(skip_redteam=True), 0)
      self.assertEqual(saved.read_bytes(), payload)
      with redirect_stdout(io.StringIO()):
        recovered = isolated.recover_blocked_redteam_messages(
            skip_redteam=False)
      self.assertEqual(recovered, 0)
      self.assertEqual(saved.read_bytes(), payload)
      self.assertFalse((Path(isolated.MAILBOX) / saved.name).exists())


if __name__ == "__main__":
  unittest.main()
