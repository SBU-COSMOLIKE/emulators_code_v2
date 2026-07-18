"""Check the single Red Team pass used for protected policy edits."""

import unittest

from ai.tools import mailbox_daemon as daemon


class ProtectedPolicyReviewTests(unittest.TestCase):

    def test_policy_request_is_cycle_free_and_allowed_during_repairs(self):
        request = daemon.sol_ticket_payload(
            ticket_kind="policy", text="Review this exact draft.")
        self.assertEqual(daemon.sol_ticket_kind(request), "policy")
        self.assertEqual(
            daemon.register_ticket_cycle_message("sol", request),
            (None, None))
        self.assertIsNone(daemon.sol_ticket_refusal(
            ticket_kind="policy", admission_count=3, fix_only=True))

    def test_role_files_share_the_protected_admin_route(self):
        self.assertEqual(
            set(daemon.ARCHITECT_PROTECTED_POLICY_PATHS)
            - set(daemon.ARCHITECT_PERMANENT_NOTE_PATHS),
            {".claude/FABLE_ROLE.md", ".codex/REDTEAM_ROLE.md",
             "ai/notes/role-contract.yaml"})
        self.assertTrue(
            set(daemon.ARCHITECT_PROTECTED_POLICY_PATHS).issubset(
                daemon.ARCHITECT_PROTECTED_TRACKED_PATHS))


if __name__ == "__main__":
    unittest.main()
