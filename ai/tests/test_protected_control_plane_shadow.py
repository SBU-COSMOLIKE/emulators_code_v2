"""Run D0's protected-ticket harness against real disposable Git commits."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_FILES = (
    "ai/tools/candidate_admission.py",
    "ai/tools/control_plane_handoff.py",
    "ai/tools/mailbox_daemon.py",
    "ai/tools/handoff_contract.py",
    "ai/tools/provider_health.py",
    "ai/tools/review_dispatch.py",
    "ai/tools/reopen_transition.py",
    "ai/tools/role_contract.py",
    "ai/notes/role-contract.yaml",
    "ai/notes/implementer-failure-modes.yaml",
    "ai/notes/backlog.md",
)


def git(repository, *arguments):
  """Run one Git command in the disposable controller repository."""
  result = subprocess.run(
      ["git", "-C", str(repository), *arguments], check=True,
      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  return result.stdout.strip()


def make_controller_repository(parent):
  """Commit the current controller files without touching the live checkout."""
  repository = Path(parent) / "controller"
  git(parent, "init", "--quiet", str(repository))
  git(repository, "config", "user.name", "D0 shadow test")
  git(repository, "config", "user.email", "shadow@example.invalid")
  for relative in CONTROLLER_FILES:
    destination = repository / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_ROOT / relative, destination)
  git(repository, "add", *CONTROLLER_FILES)
  git(repository, "commit", "--quiet", "-m", "trusted D0")
  git(repository, "branch", "-M", "main")
  return repository, git(repository, "rev-parse", "HEAD")


def load_controller(repository):
  """Load D0 from its own committed path before a proposed D1 is written."""
  path = repository / "ai/tools/mailbox_daemon.py"
  name = "trusted_shadow_driver_" + os.urandom(8).hex()
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def seed_d0_state(d0, repository, base_commit):
  """Give D0 active, completed, recovery, topology, and ref state."""
  active_cycle = "handoff-active@" + base_commit
  completed_cycle = "handoff-completed@" + base_commit
  state = d0.empty_ticket_cycle_state()
  state["generation"] = 2
  state["active"][active_cycle] = {
      "phase": "implementation",
      "commit": None,
      "mode": "normal",
      "route": "primary",
      "ticket_class": "ordinary",
      "path_scope": ["emulator/model.py"],
  }
  state["completed"][completed_cycle] = base_commit
  d0.write_ticket_cycle_state(state)

  candidate_ref = d0.cycle_candidate_ref(active_cycle)
  git(repository, "update-ref", candidate_ref, base_commit)
  d0.write_candidate_state({
      "schema": d0.CANDIDATE_STATE_SCHEMA,
      "cycles": {
          active_cycle: {"ref": candidate_ref, "commit": base_commit},
      },
  })
  d0.write_push_debt(base_commit, "seeded D0 push debt")
  notes = Path(d0.WORKTREE) / "ai/notes"
  (notes / d0.BACKLOG_SYNC_RECOVERY_NAME).write_bytes(
      (notes / "backlog.md").read_bytes())

  topology_dir = Path(repository) / "saved-topology"
  topology_dir.mkdir()
  records = {
      "primary_state": {
          "schema": d0.PRIMARY_STATE_SCHEMA,
          "repository": str(Path(repository).resolve()),
          "name": "mailbox-primary",
          "path": str((topology_dir / "mailbox-primary").resolve()),
          "branch": d0.PRIMARY_BRANCH,
          "topology": d0.PRIMARY_TOPOLOGY_MARKER,
      },
      "implementer_state": {
          "schema": d0.IMPLEMENTER_STATE_SCHEMA,
          "repository": str(Path(repository).resolve()),
          "name": "mailbox-implementer",
          "path": str((topology_dir / "mailbox-implementer").resolve()),
          "branch": d0.IMPLEMENTER_BRANCH,
      },
      "sol_state": {
          "schema": d0.SOL_STATE_SCHEMA,
          "repository": str(Path(repository).resolve()),
          "name": "mailbox-sol",
          "path": str((topology_dir / "mailbox-sol").resolve()),
          "branch": d0.SOL_BRANCH,
      },
  }
  active_topology = {}
  for key, payload in records.items():
    path = topology_dir / (key + ".json")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    active_topology[key] = str(path)
  d0.ACTIVE_TOPOLOGY = active_topology
  return active_cycle


def state_schemas(d0, ticket_cycle=None):
  """Name every durable schema covered by the D0 to D1 handoff."""
  return {
      "ticket_cycle": (d0.TICKET_CYCLE_STATE_SCHEMA
                       if ticket_cycle is None else ticket_cycle),
      "candidate": d0.CANDIDATE_STATE_SCHEMA,
      "primary_worktree": d0.PRIMARY_STATE_SCHEMA,
      "implementer_worktree": d0.IMPLEMENTER_STATE_SCHEMA,
      "red_team_worktree": d0.SOL_STATE_SCHEMA,
      "notes_admin": d0.ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA,
  }


class ProtectedControlPlaneShadowTests(unittest.TestCase):
  """Prove the trusted harness drives D1 instead of trusting its tests."""

  def test_current_controller_passes_real_state_and_landing_scenarios(self):
    """The D0 probe reaches restart, one-parent L, and stale-main checks."""
    with tempfile.TemporaryDirectory(prefix="protected-shadow-pass-") as tmp:
      repository, candidate = make_controller_repository(tmp)
      d0 = load_controller(repository)
      active_cycle = seed_d0_state(d0, repository, candidate)
      refs_before = git(
          repository, "for-each-ref", "--format=%(refname) %(objectname)",
          d0.CANDIDATE_REF_ROOT, "refs/heads/main")

      passed, log_path = d0.trusted_control_plane_check(
          commit=candidate, label="focused-pass")

      self.assertTrue(passed, Path(log_path).read_text(encoding="utf-8"))
      log = Path(log_path).read_text(encoding="utf-8")
      self.assertIn("D0_TO_D1_STATE_HANDOFF_PASSED", log)
      self.assertIn("D0_SHADOW_SCENARIOS_PASSED", log)
      self.assertEqual(
          git(repository, "for-each-ref",
              "--format=%(refname) %(objectname)",
              d0.CANDIDATE_REF_ROOT, "refs/heads/main"),
          refs_before)
      self.assertIn(active_cycle, d0.read_ticket_cycle_state()["active"])
      self.assertIn(active_cycle, d0.read_candidate_state()["cycles"])

  def test_d0_refuses_new_state_schema_without_migration_declaration(self):
    """A D1-only reader cannot take over D0's nonempty saved state."""
    with tempfile.TemporaryDirectory(prefix="protected-shadow-schema-") as tmp:
      repository, base = make_controller_repository(tmp)
      d0 = load_controller(repository)
      seed_d0_state(d0, repository, base)
      daemon_path = repository / "ai/tools/mailbox_daemon.py"
      source = daemon_path.read_text(encoding="utf-8")
      boundary = "TICKET_CYCLE_STATE_SCHEMA = 6\n"
      self.assertEqual(source.count(boundary), 1)
      daemon_path.write_text(
          source.replace(boundary, "TICKET_CYCLE_STATE_SCHEMA = 7\n"),
          encoding="utf-8")
      git(repository, "add", "ai/tools/mailbox_daemon.py")
      git(repository, "commit", "--quiet", "-m", "new state without migration")
      candidate = git(repository, "rev-parse", "HEAD")

      passed, log_path = d0.trusted_control_plane_check(
          commit=candidate, label="focused-schema-refusal")

      self.assertFalse(passed)
      self.assertIn(
          "state schema changed without an explicit migration declaration",
          Path(log_path).read_text(encoding="utf-8"))

  def test_d0_refuses_wrong_migration_versions(self):
    """A declaration cannot rename the D0 or D1 schema involved."""
    with tempfile.TemporaryDirectory(prefix="protected-shadow-version-") as tmp:
      repository, base = make_controller_repository(tmp)
      d0 = load_controller(repository)
      seed_d0_state(d0, repository, base)
      daemon_path = repository / "ai/tools/mailbox_daemon.py"
      source = daemon_path.read_text(encoding="utf-8").replace(
          "TICKET_CYCLE_STATE_SCHEMA = 6\n",
          "TICKET_CYCLE_STATE_SCHEMA = 7\n", 1)
      daemon_path.write_text(source, encoding="utf-8")
      declaration = {
          "state_migration": {
              "from_schema": state_schemas(d0, ticket_cycle=5),
              "to_schema": state_schemas(d0, ticket_cycle=7),
              "preserved_invariants": list(
                  d0.CONTROL_PLANE_PRESERVED_INVARIANTS),
              "function": "unused_migration",
          },
      }
      migration = repository / d0.CONTROL_PLANE_MIGRATION_PATH
      migration.write_text(json.dumps(declaration) + "\n", encoding="utf-8")
      git(repository, "add", "ai/tools/mailbox_daemon.py", str(migration))
      git(repository, "commit", "--quiet", "-m", "wrong migration versions")

      passed, log_path = d0.trusted_control_plane_check(
          commit=git(repository, "rev-parse", "HEAD"),
          label="focused-version-refusal")

      self.assertFalse(passed)
      self.assertIn(
          "state migration declaration is not exact",
          Path(log_path).read_text(encoding="utf-8"))

  def test_d0_refuses_migration_that_drops_saved_ticket(self):
    """An explicit migration still must preserve D0's exact identities."""
    with tempfile.TemporaryDirectory(prefix="protected-shadow-drop-") as tmp:
      repository, base = make_controller_repository(tmp)
      d0 = load_controller(repository)
      seed_d0_state(d0, repository, base)
      daemon_path = repository / "ai/tools/mailbox_daemon.py"
      source = daemon_path.read_text(encoding="utf-8")
      source = source.replace(
          "TICKET_CYCLE_STATE_SCHEMA = 6\n",
          "TICKET_CYCLE_STATE_SCHEMA = 7\n", 1)
      source += """

def discard_saved_ticket_state(repository, ticket_path, candidate_path):
    write_ticket_cycle_state(empty_ticket_cycle_state())
"""
      daemon_path.write_text(source, encoding="utf-8")
      declaration = {
          "state_migration": {
              "from_schema": state_schemas(d0),
              "to_schema": state_schemas(d0, ticket_cycle=7),
              "preserved_invariants": list(
                  d0.CONTROL_PLANE_PRESERVED_INVARIANTS),
              "function": "discard_saved_ticket_state",
          },
      }
      migration = repository / d0.CONTROL_PLANE_MIGRATION_PATH
      migration.write_text(
          json.dumps(declaration, sort_keys=True, indent=2) + "\n",
          encoding="utf-8")
      git(repository, "add", "ai/tools/mailbox_daemon.py", str(migration))
      git(repository, "commit", "--quiet", "-m", "migration drops state")
      candidate = git(repository, "rev-parse", "HEAD")

      passed, log_path = d0.trusted_control_plane_check(
          commit=candidate, label="focused-drop-refusal")

      self.assertFalse(passed)
      self.assertIn("AssertionError", Path(log_path).read_text(encoding="utf-8"))

  def test_d0_accepts_declared_migration_that_preserves_saved_state(self):
    """D1 may change a schema only by migrating D0's copied bytes."""
    with tempfile.TemporaryDirectory(prefix="protected-shadow-migrate-") as tmp:
      repository, base = make_controller_repository(tmp)
      d0 = load_controller(repository)
      active_cycle = seed_d0_state(d0, repository, base)
      daemon_path = repository / "ai/tools/mailbox_daemon.py"
      source = daemon_path.read_text(encoding="utf-8")
      source = source.replace(
          "TICKET_CYCLE_STATE_SCHEMA = 6\n",
          "TICKET_CYCLE_STATE_SCHEMA = 7\n", 1)
      source += """

def migrate_ticket_state_schema(repository, ticket_path, candidate_path):
    with open(ticket_path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != 6:
        raise TicketCycleStateError("migration expected ticket schema 6")
    payload["schema"] = 7
    with open(ticket_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\\n")
"""
      daemon_path.write_text(source, encoding="utf-8")
      declaration = {
          "state_migration": {
              "from_schema": state_schemas(d0),
              "to_schema": state_schemas(d0, ticket_cycle=7),
              "preserved_invariants": list(
                  d0.CONTROL_PLANE_PRESERVED_INVARIANTS),
              "function": "migrate_ticket_state_schema",
          },
      }
      migration = repository / d0.CONTROL_PLANE_MIGRATION_PATH
      migration.write_text(
          json.dumps(declaration, sort_keys=True, indent=2) + "\n",
          encoding="utf-8")
      git(repository, "add", "ai/tools/mailbox_daemon.py", str(migration))
      git(repository, "commit", "--quiet", "-m", "preserve state migration")
      candidate = git(repository, "rev-parse", "HEAD")

      passed, log_path = d0.trusted_control_plane_check(
          commit=candidate, label="focused-migration-pass")

      self.assertTrue(passed, Path(log_path).read_text(encoding="utf-8"))
      log = Path(log_path).read_text(encoding="utf-8")
      self.assertIn("D0_TO_D1_STATE_HANDOFF_PASSED", log)
      self.assertIn(active_cycle, d0.read_ticket_cycle_state()["active"])

  def test_d0_detects_a_candidate_that_weakens_stale_main_refusal(self):
    """A D1 landing shortcut fails the D0-owned real-path scenario."""
    with tempfile.TemporaryDirectory(prefix="protected-shadow-stale-") as tmp:
      repository, _base = make_controller_repository(tmp)
      d0 = load_controller(repository)
      daemon_path = repository / "ai/tools/mailbox_daemon.py"
      source = daemon_path.read_text(encoding="utf-8")
      boundary = "    if current_main in {parent_commit, landing_commit}:\n"
      self.assertEqual(source.count(boundary), 1)
      daemon_path.write_text(
          source.replace(boundary, "    return None\n" + boundary),
          encoding="utf-8")
      git(repository, "add", "ai/tools/mailbox_daemon.py")
      git(repository, "commit", "--quiet", "-m", "weaken stale guard")
      candidate = git(repository, "rev-parse", "HEAD")

      passed, log_path = d0.trusted_control_plane_check(
          commit=candidate, label="focused-stale-refusal")

      self.assertFalse(passed)
      self.assertIn(
          "changed main reused a stale protected landing",
          Path(log_path).read_text(encoding="utf-8"))

  def test_candidate_test_code_is_not_part_of_the_trusted_harness(self):
    """A candidate test cannot replace the scenarios generated by D0."""
    with tempfile.TemporaryDirectory(prefix="protected-shadow-tests-") as tmp:
      repository, _base = make_controller_repository(tmp)
      d0 = load_controller(repository)
      candidate_test = repository / "ai/tests/test_untrusted_shadow.py"
      candidate_test.parent.mkdir(parents=True, exist_ok=True)
      candidate_test.write_text(
          "raise RuntimeError('candidate test must not be imported')\n",
          encoding="utf-8")
      git(repository, "add", "ai/tests/test_untrusted_shadow.py")
      git(repository, "commit", "--quiet", "-m", "untrusted candidate test")
      candidate = git(repository, "rev-parse", "HEAD")

      passed, log_path = d0.trusted_control_plane_check(
          commit=candidate, label="focused-untrusted-test")

      self.assertTrue(passed, Path(log_path).read_text(encoding="utf-8"))
      log = Path(log_path).read_text(encoding="utf-8")
      self.assertIn("D0_SHADOW_SCENARIOS_PASSED", log)
      self.assertNotIn("candidate test must not be imported", log)


if __name__ == "__main__":
  unittest.main()
