"""Exercise role boundaries through code rather than explanatory wording."""

from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
