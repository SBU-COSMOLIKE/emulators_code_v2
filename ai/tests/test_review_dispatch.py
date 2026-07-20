"""Check the lower-cost command used only for routine ticket reviews."""

import unittest

from ai.tools import review_dispatch


class ReviewDispatchTests(unittest.TestCase):
    """Keep planning and implementation effort separate from later reviews."""

    def test_only_later_architect_work_is_a_routine_review(self):
        """A first plan is full effort; candidate and reopening checks are not."""
        self.assertIsNone(review_dispatch.review_kind(agent="fable"))
        self.assertEqual(
            review_dispatch.review_kind(
                agent="fable", candidate_audit=True),
            "Architect candidate audit")
        self.assertEqual(
            review_dispatch.review_kind(agent="fable", reopening=True),
            "Architect reopening decision")

    def test_implementer_and_discovery_keep_their_selected_effort(self):
        """The cheaper path never applies to coding or a new Red Team search."""
        self.assertIsNone(
            review_dispatch.review_kind(agent="opus", candidate_audit=True))
        self.assertIsNone(
            review_dispatch.review_kind(agent="sol", ticket_kind="discovery"))

    def test_closure_and_control_plane_are_routine_redteam_reviews(self):
        """Both bounded Red Team checks may use the review effort."""
        self.assertEqual(
            review_dispatch.review_kind(
                agent="sol", ticket_kind="closure"),
            "Red Team closure")
        self.assertEqual(
            review_dispatch.review_kind(
                agent="sol", ticket_kind="control-plane"),
            "Red Team control-plane review")

    def test_claude_review_replaces_only_the_effort_value(self):
        """Model, permissions, and the caller's original command remain intact."""
        original = [
            "claude", "-p", "--model", "opus", "--effort", "xhigh",
            "--permission-mode", "acceptEdits"]
        changed = review_dispatch.command_with_effort(
            original, agent="fable", effort="medium")
        self.assertEqual(changed[5], "medium")
        self.assertEqual(original[5], "xhigh")
        self.assertEqual(changed[:5] + changed[6:], original[:5] + original[6:])

    def test_codex_review_replaces_only_the_reasoning_setting(self):
        """The bounded Red Team keeps its model, service tier, and sandbox."""
        original = [
            "codex", "exec", "--model", "sol",
            "model_reasoning_effort=xhigh", "service_tier=standard"]
        changed = review_dispatch.command_with_effort(
            original, agent="sol", effort="low")
        self.assertEqual(changed[4], "model_reasoning_effort=low")
        self.assertEqual(original[4], "model_reasoning_effort=xhigh")
        self.assertEqual(changed[:4] + changed[5:], original[:4] + original[5:])

    def test_missing_or_unsupported_effort_refuses(self):
        """A malformed provider command cannot silently stay expensive."""
        with self.assertRaisesRegex(ValueError, "effort"):
            review_dispatch.command_with_effort(
                ["claude", "-p"], agent="fable", effort="medium")
        with self.assertRaisesRegex(ValueError, "review effort"):
            review_dispatch.command_with_effort(
                ["claude", "--effort", "max"],
                agent="fable", effort="max")

if __name__ == "__main__":
    unittest.main()
