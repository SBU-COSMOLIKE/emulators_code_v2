"""Exercise role boundaries through code rather than explanatory wording."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from ai.tools import mailbox_daemon


class RoleWorkflowBehaviorTests(unittest.TestCase):
    """Check serialized decisions, scope enforcement, and safe Git behavior."""

    def test_architect_go_names_one_exact_candidate(self):
        cycle = "cycle-a@" + "1" * 40
        candidate = "2" * 40

        self.assertEqual(
            mailbox_daemon.architect_go_request_payload(
                cycle_id=cycle, candidate_commit=candidate, mode="normal"),
            "MAILBOX-RETURN: architect-go\n"
            "MAILBOX-CYCLE: " + cycle + "\n"
            "MAILBOX-CANDIDATE: " + candidate + "\n"
            "MAILBOX-MODE: normal\n"
            "MAILBOX-DECISION: GO\n")

    def test_rejected_push_never_retries_with_force(self):
        """One rejected ordinary push leaves debt without rewriting history."""
        landing = "a" * 40
        calls = []
        original_run = mailbox_daemon.subprocess.run
        original_relay = mailbox_daemon.RELAY_DIR

        with tempfile.TemporaryDirectory(prefix="push-rejection-") as relay:
            def reject_non_fast_forward(command, *args, **kwargs):
                calls.append(list(command))
                return mailbox_daemon.subprocess.CompletedProcess(
                    command, 1, stdout=b"",
                    stderr=b"! [rejected] non-fast-forward\n")

            try:
                mailbox_daemon.RELAY_DIR = relay
                mailbox_daemon.subprocess.run = reject_non_fast_forward
                pushed, detail = (
                    mailbox_daemon.push_exact_landing_or_record_debt(
                        landing=landing))
            finally:
                mailbox_daemon.subprocess.run = original_run
                mailbox_daemon.RELAY_DIR = original_relay

            expected = [
                "git", "-C", mailbox_daemon.AGENT_CWD["fable"], "push",
                "--porcelain", "origin", landing + ":refs/heads/main"]
            self.assertFalse(pushed)
            self.assertIn("non-fast-forward", detail)
            self.assertEqual(calls, [expected])
            self.assertFalse(any(
                option in calls[0]
                for option in ("--force", "-f", "--force-with-lease",
                               "--force-if-includes")))
            self.assertFalse(calls[0][-1].startswith(("+", ":")))

            debt = Path(relay, "pending-main-push-" + landing + ".txt")
            debt_text = debt.read_text(encoding="utf-8")
            self.assertIn(
                "Push is still required: git push origin " + landing
                + ":refs/heads/main\n",
                debt_text)
            self.assertIn(
                "Last push result: ! [rejected] non-fast-forward",
                debt_text)

    def test_github_no_skips_the_push_and_preserves_earlier_debt(self):
        """--github no contacts no remote and erases no earlier debt file."""
        landing = "c" * 40
        earlier = "d" * 40
        earlier_payload = ("Local main contains verified landing " + earlier
                           + ".\nPush is still required: git push origin "
                           + earlier + ":refs/heads/main\n")
        original_relay = mailbox_daemon.RELAY_DIR
        original_choice = mailbox_daemon.GITHUB_PUSH_ENABLED
        original_cwd = mailbox_daemon.AGENT_CWD

        with tempfile.TemporaryDirectory(prefix="github-choice-") as tmp:
            relay = str(Path(tmp, "relay"))
            Path(relay).mkdir()
            earlier_debt = Path(relay, "pending-main-push-" + earlier + ".txt")
            earlier_debt.write_text(earlier_payload, encoding="utf-8")
            # Any accidental Git command would run inside this missing
            # folder, fail, and write a new debt record that the assertions
            # below would catch — so no subprocess replacement is needed.
            missing_repo = str(Path(tmp, "missing-repo"))
            captured = io.StringIO()
            try:
                mailbox_daemon.RELAY_DIR = relay
                mailbox_daemon.GITHUB_PUSH_ENABLED = False
                mailbox_daemon.AGENT_CWD = {"fable": missing_repo,
                                            "opus": missing_repo,
                                            "sol": missing_repo}
                with contextlib.redirect_stdout(captured):
                    pushed, detail = (
                        mailbox_daemon.push_exact_landing_or_record_debt(
                            landing=landing))
            finally:
                mailbox_daemon.RELAY_DIR = original_relay
                mailbox_daemon.GITHUB_PUSH_ENABLED = original_choice
                mailbox_daemon.AGENT_CWD = original_cwd

            self.assertIsNone(pushed)
            self.assertEqual(detail, "")
            printed = captured.getvalue()
            self.assertIn("skipped by user choice (--github no)", printed)
            self.assertIn(landing, printed)
            new_debt = Path(relay, "pending-main-push-" + landing + ".txt")
            self.assertFalse(new_debt.exists())
            self.assertEqual(
                earlier_debt.read_text(encoding="utf-8"), earlier_payload)

    def test_github_push_stays_on_by_default(self):
        """Existing commands keep pushing unless the user says otherwise."""
        self.assertEqual(mailbox_daemon.DEFAULT_GITHUB_PUSH, "yes")
        self.assertTrue(mailbox_daemon.GITHUB_PUSH_ENABLED)

    def test_github_value_outside_yes_no_fails_before_any_work(self):
        """The command line refuses an unknown --github value at parse time."""
        daemon_path = Path(mailbox_daemon.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(daemon_path), "--github", "maybe"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--github", result.stderr)
        self.assertIn("invalid choice", result.stderr)

    def test_candidate_scope_has_three_mechanical_results(self):
        allowed = {"emulator/training.py", "ai/tests/test_training.py",
                   ".claude/FABLE_ROLE.md"}
        cases = (
            ({"emulator/training.py"}, ("IN_SCOPE", set())),
            ({"emulator/training.py", "emulator/model.py"},
             ("SCOPE_EXCEEDED", {"emulator/model.py"})),
            ({"emulator/training.py", ".claude/FABLE_ROLE.md"},
             ("PROTECTED_PATH_VIOLATION", {".claude/FABLE_ROLE.md"})),
        )
        for changed, expected in cases:
            with self.subTest(result=expected[0]):
                self.assertEqual(
                    mailbox_daemon.classify_candidate_scope(
                        changed_paths=changed, path_scope=allowed),
                    expected)

    def test_trusted_tool_is_never_a_mailbox_candidate(self):
        trusted_tool = "ai/tools/mailbox_daemon.py"

        self.assertEqual(
            mailbox_daemon.classify_candidate_scope(
                changed_paths={trusted_tool}, path_scope={trusted_tool},
                ticket_class="ordinary"),
            ("PROTECTED_PATH_VIOLATION", {trusted_tool}))
        self.assertEqual(
            mailbox_daemon.classify_candidate_scope(
                changed_paths={trusted_tool}, path_scope={trusted_tool},
                ticket_class="protected-control-plane"),
            ("PROTECTED_PATH_VIOLATION", {trusted_tool}))
        self.assertEqual(
            mailbox_daemon.classify_candidate_scope(
                changed_paths={trusted_tool}, path_scope={"ai/tools/other.py"},
                ticket_class="protected-control-plane"),
            ("PROTECTED_PATH_VIOLATION", {trusted_tool}))

    def test_protected_landing_needs_both_keys_for_the_same_candidate(self):
        candidate = "c" * 40
        cases = (
            ({"architect_candidate": candidate,
              "redteam_candidate": candidate,
              "redteam_result": "ACCEPT-CONTROL-PLANE"}, True),
            ({"architect_candidate": None,
              "redteam_candidate": candidate,
              "redteam_result": "ACCEPT-CONTROL-PLANE"}, False),
            ({"architect_candidate": candidate,
              "redteam_candidate": "d" * 40,
              "redteam_result": "ACCEPT-CONTROL-PLANE"}, False),
            ({"architect_candidate": candidate,
              "redteam_candidate": candidate,
              "redteam_result": "REJECT-CONTROL-PLANE"}, False),
        )
        for saved, expected in cases:
            with self.subTest(saved=saved), mock.patch.object(
                    mailbox_daemon, "control_plane_ticket_state",
                    return_value=saved):
                self.assertIs(
                    mailbox_daemon.protected_landing_ready(
                        cycle_id="protected-ticket@" + "b" * 40,
                        candidate_commit=candidate),
                    expected)

    def test_protected_state_and_completed_history_round_trip(self):
        active_cycle = "protected-active@" + "1" * 40
        completed_cycle = "protected-complete@" + "2" * 40
        active_candidate = "3" * 40
        completed_candidate = "4" * 40
        landing = "5" * 40
        active_control = mailbox_daemon.empty_control_plane_state()
        active_control.update({
            "architect_candidate": active_candidate,
            "redteam_candidate": active_candidate,
            "redteam_result": "ACCEPT-CONTROL-PLANE",
            "shadow_status": "PASSED",
            "shadow_evidence": "relay/shadow.log",
        })
        completed_control = dict(active_control)
        completed_control.update({
            "architect_candidate": completed_candidate,
            "redteam_candidate": completed_candidate,
            "health_status": "HEALTHY",
            "health_evidence": "relay/health.log",
        })
        state = mailbox_daemon.empty_ticket_cycle_state()
        state["active"][active_cycle] = {
            "phase": "implementation",
            "commit": None,
            "mode": "normal",
            "route": "primary",
            "ticket_class": "protected-control-plane",
            "path_scope": ["ai/tools/mailbox_daemon.py"],
            "control_plane": active_control,
        }
        state["completed"][completed_cycle] = landing
        state["control_plane_history"][completed_cycle] = {
            "candidate": completed_candidate,
            "landing": landing,
            "control_plane": completed_control,
        }

        restored = mailbox_daemon.validate_ticket_cycle_state(
            json.loads(json.dumps(state)))

        self.assertEqual(
            restored["active"][active_cycle]["control_plane"],
            active_control)
        self.assertEqual(
            restored["control_plane_history"][completed_cycle],
            state["control_plane_history"][completed_cycle])

    def test_stale_landing_names_the_narrow_revalidation_evidence(self):
        candidate = "c" * 40
        landing = "1" * 40
        old_main = "0" * 40
        new_main = "2" * 40
        with mock.patch.object(
                mailbox_daemon, "_commit_is_ancestor",
                side_effect=(False, True)):
            problem = mailbox_daemon._prepared_landing_main_problem(
                candidate_commit=candidate, landing_commit=landing,
                parent_commit=old_main, current_main=new_main)

        self.assertTrue(problem.startswith(
            mailbox_daemon.STALE_INTEGRATION_REVALIDATION + ":"))
        for label, commit in (("C", candidate), ("L", landing),
                              ("M0", old_main), ("M1", new_main)):
            self.assertIn(label + "=" + commit, problem)
        self.assertIn("inspect M0-to-M1", problem)
        self.assertIn("provisional combined result on M1", problem)
        self.assertIn("complete candidate audit only if", problem)

    def test_stale_landing_is_not_confused_with_recovery_or_divergence(self):
        commits = dict(candidate_commit="c" * 40,
                       landing_commit="1" * 40,
                       parent_commit="0" * 40,
                       current_main="2" * 40)
        with mock.patch.object(
                mailbox_daemon, "_commit_is_ancestor", return_value=True):
            recovery = mailbox_daemon._prepared_landing_main_problem(
                **commits)
        self.assertIn("durable-state recovery", recovery)
        self.assertNotIn(mailbox_daemon.STALE_INTEGRATION_REVALIDATION,
                         recovery)

        with mock.patch.object(
                mailbox_daemon, "_commit_is_ancestor",
                side_effect=(False, False)):
            divergence = mailbox_daemon._prepared_landing_main_problem(
                **commits)
        self.assertIn("user reconciliation", divergence)
        self.assertNotIn(mailbox_daemon.STALE_INTEGRATION_REVALIDATION,
                         divergence)

    def test_scope_exceeded_banner_carries_data_not_untrusted_lines(self):
        unsafe_path = "emulator/extra\n\x1b[31m.py"
        banner = mailbox_daemon.dispatch_banner(
            store_max=4, newer_in_lane=0,
            previous_timeout_minutes=None,
            candidate_scope={
                "result": "SCOPE_EXCEEDED",
                "paths": [unsafe_path],
            })

        self.assertIn("result: SCOPE_EXCEEDED", banner)
        self.assertIn("Architect GO explicitly accepts this expansion", banner)
        self.assertIn("a repair handoff rejects it", banner)
        self.assertNotIn(unsafe_path, banner)
        self.assertNotIn("\x1b", banner)
        self.assertIn(repr(unsafe_path), banner)

    def test_routine_review_banner_omits_discovery_lecture(self):
        banner = mailbox_daemon.dispatch_banner(
            store_max=4, newer_in_lane=0,
            previous_timeout_minutes=None,
            routine_review="Red Team closure")

        self.assertIn("kind: Red Team closure", banner)
        self.assertIn("named ticket and commit only", banner)
        self.assertNotIn("DISCOVERY SEVERITY", banner)
        self.assertNotIn("DISCOVERY SCOPE", banner)
        self.assertIn("ticket character limit:", banner)


if __name__ == "__main__":
    unittest.main()
