"""Exercise protected stale-main recovery without patching live functions."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from ai.tools import mailbox_daemon as daemon


def git(root, *arguments):
  """Run one local Git command and return its stripped text output."""
  result = subprocess.run(
      ["git", "-C", str(root), *arguments], check=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  return result.stdout.strip()


def isolated_daemon(root):
  """Load the current daemon with its mailbox and Git state under ``root``."""
  root = Path(root)
  git(root.parent, "init", "--quiet", str(root))
  git(root, "config", "user.name", "Control Plane Test")
  git(root, "config", "user.email", "control-plane@example.invalid")
  (root / ".claude/worktrees").mkdir(parents=True)
  tools = root / "ai/tools"
  notes = root / "ai/notes"
  tools.mkdir(parents=True)
  notes.mkdir(parents=True)
  source = Path(daemon.__file__).resolve().parent
  for name in (
      "candidate_admission.py", "control_plane_handoff.py",
      "mailbox_daemon.py", "provider_health.py", "reopen_transition.py",
      "review_dispatch.py", "role_contract.py"):
    shutil.copy2(source / name, tools / name)
  shutil.copy2(
      source.parent / "notes/role-contract.yaml",
      notes / "role-contract.yaml")
  (root / "content.txt").write_text("M0\n", encoding="utf-8")
  git(root, "add", ".")
  git(root, "commit", "--quiet", "-m", "M0")
  git(root, "branch", "-M", "main")
  spec = importlib.util.spec_from_file_location(
      "isolated_stale_control_plane", tools / "mailbox_daemon.py")
  loaded = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(loaded)
  loaded.AGENT_CWD["fable"] = str(root)
  return loaded


class ProtectedStaleIntegrationTests(unittest.TestCase):
  """Keep C and both decisions while a newer main receives its own audit."""

  def setUp(self):
    self.cycle = "protected-stale@" + "b" * 40
    self.candidate = "c" * 40
    self.landing = "1" * 40
    self.old_main = "0" * 40
    self.new_main = "2" * 40

  def accepted_control(self, module=daemon):
    control = module.empty_control_plane_state()
    control.update({
        "architect_candidate": self.candidate,
        "redteam_candidate": self.candidate,
        "redteam_result": "ACCEPT-CONTROL-PLANE",
        "shadow_status": "PASSED",
        "shadow_evidence": "first-shadow.log",
    })
    return control

  def save_active(self, module, control):
    state = module.empty_ticket_cycle_state()
    state["active"][self.cycle] = {
        "phase": "implementation", "commit": None,
        "mode": "normal", "route": "primary",
        "ticket_class": "protected-control-plane",
        "path_scope": ["ai/tools/mailbox_daemon.py"],
        "control_plane": control,
    }
    module.write_ticket_cycle_state(state)
    candidates = module.empty_candidate_state()
    candidates["cycles"][self.cycle] = {
        "ref": module.cycle_candidate_ref(self.cycle),
        "commit": self.candidate,
    }
    module.write_candidate_state(candidates)

  def test_request_binds_c_l_m0_and_m1(self):
    payload = daemon.control_plane_integration_request_payload(
        cycle_id=self.cycle, candidate=self.candidate,
        stale_landing=self.landing, old_main=self.old_main,
        new_main=self.new_main, mode="normal")

    self.assertEqual(daemon.control_plane_integration_request(payload), {
        "cycle_id": self.cycle, "mode": "normal",
        "candidate": self.candidate, "stale_landing": self.landing,
        "old_main": self.old_main, "new_main": self.new_main,
    })

  def test_stale_diagnosis_has_four_exact_git_identities(self):
    text = (
        daemon.STALE_INTEGRATION_REVALIDATION + ": C=" + self.candidate
        + " L=" + self.landing + " M0=" + self.old_main
        + " M1=" + self.new_main + "; bounded explanation")

    self.assertEqual(daemon.stale_integration_details(text), {
        "candidate": self.candidate, "stale_landing": self.landing,
        "old_main": self.old_main, "new_main": self.new_main,
    })

  def test_stale_record_preserves_both_exact_candidate_keys(self):
    with tempfile.TemporaryDirectory(prefix="protected-stale-") as tmp:
      isolated = isolated_daemon(Path(tmp) / "repo")
      self.save_active(isolated, self.accepted_control(isolated))

      isolated.record_control_plane_integration_stale(
          cycle_id=self.cycle, candidate_commit=self.candidate,
          stale_landing=self.landing, old_main=self.old_main,
          new_main=self.new_main)

      saved = isolated.read_ticket_cycle_state()["active"][self.cycle]
      control = saved["control_plane"]
      self.assertTrue(isolated.control_plane_keys_ready(
          control=control, candidate_commit=self.candidate))
      self.assertEqual(control["integration_status"], "STALE")
      self.assertIsNone(control["shadow_status"])

  def test_fresh_go_is_bound_to_the_still_current_m1(self):
    with tempfile.TemporaryDirectory(prefix="protected-revalidate-") as tmp:
      root = Path(tmp) / "repo"
      isolated = isolated_daemon(root)
      m1 = git(root, "rev-parse", "HEAD")
      control = self.accepted_control(isolated)
      control.update({
          "integration_status": "STALE", "integration_main": m1,
          "stale_landing": self.landing, "stale_parent": self.old_main,
          "shadow_status": None, "shadow_evidence": None,
      })
      self.save_active(isolated, control)

      isolated.record_control_plane_integration_go(
          cycle_id=self.cycle, candidate_commit=self.candidate,
          new_main=m1, evidence="validated-delivery-receipt")

      saved = isolated.read_ticket_cycle_state()["active"][self.cycle]
      self.assertEqual(
          saved["control_plane"]["integration_status"], "REVALIDATED")
      self.assertEqual(
          saved["control_plane"]["integration_evidence"],
          "validated-delivery-receipt")

  def test_another_main_advance_refuses_the_old_revalidation(self):
    with tempfile.TemporaryDirectory(prefix="protected-moved-main-") as tmp:
      root = Path(tmp) / "repo"
      isolated = isolated_daemon(root)
      m1 = git(root, "rev-parse", "HEAD")
      control = self.accepted_control(isolated)
      control.update({
          "integration_status": "STALE", "integration_main": m1,
          "stale_landing": self.landing, "stale_parent": self.old_main,
          "shadow_status": None, "shadow_evidence": None,
      })
      self.save_active(isolated, control)
      (root / "content.txt").write_text("M1 moved\n", encoding="utf-8")
      git(root, "add", "content.txt")
      git(root, "commit", "--quiet", "-m", "newer main")

      with self.assertRaisesRegex(
          isolated.TicketCycleStateError, "main advanced again"):
        isolated.record_control_plane_integration_go(
            cycle_id=self.cycle, candidate_commit=self.candidate,
            new_main=m1, evidence="validated-delivery-receipt")

  def test_revalidated_state_requires_evidence(self):
    control = self.accepted_control()
    control.update({
        "integration_status": "REVALIDATED",
        "integration_main": self.new_main,
        "stale_landing": self.landing,
        "stale_parent": self.old_main,
        "integration_evidence": None,
    })
    state = daemon.empty_ticket_cycle_state()
    state["active"][self.cycle] = {
        "phase": "implementation", "commit": None,
        "mode": "normal", "route": "primary",
        "ticket_class": "protected-control-plane",
        "control_plane": control,
    }

    with self.assertRaisesRegex(
        daemon.TicketCycleStateError, "lacks evidence"):
      daemon.validate_ticket_cycle_state(state)


if __name__ == "__main__":
  unittest.main()
