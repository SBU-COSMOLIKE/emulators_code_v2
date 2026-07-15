"""Pin the thinking-role and execution-role responsibility boundary."""

from pathlib import Path
import unittest

from ai.tools import mailbox_daemon
from ai.tools.handoff_contract import REQUIRED_SECTIONS


REPO_ROOT = Path(__file__).resolve().parents[2]


def read(path):
    """Read one repository contract surface as UTF-8."""
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class RoleDirectiveContractTests(unittest.TestCase):
    """Refuse a return to goal-only handoffs or Implementer design work."""

    @classmethod
    def setUpClass(cls):
        cls.architect = read(".claude/FABLE_ROLE.md")
        cls.implementer = read(".claude/OPUS_ROLE.md")
        cls.redteam = read(".codex/REDTEAM_ROLE.md")
        cls.conventions = read("ai/notes/conventions-and-workflow.md")
        cls.ai_readme = read("ai/README.md")
        cls.router = read("ai/tools/handoff_router.py")

    def test_old_goal_only_rule_is_absent_from_binding_surfaces(self):
        forbidden = ("Goals over steps", "blueprints state goals", "never steps",
                     "over-prescription degrades")
        for name, source in (("Architect", self.architect),
                             ("workflow", self.conventions)):
            for phrase in forbidden:
                with self.subTest(name=name, phrase=phrase):
                    self.assertNotIn(phrase, source)

    def test_architect_packet_has_every_required_section_and_validator(self):
        for heading in REQUIRED_SECTIONS["architect"]:
            with self.subTest(heading=heading):
                self.assertIn("### " + heading, self.architect)
        self.assertIn(
            "python3 ai/tools/handoff_contract.py architect", self.architect)
        self.assertIn("decision-complete implementation", self.architect)
        self.assertIn("lower-capability Implementer", self.architect)
        self.assertIn("complete, self-contained repair packet", self.architect)
        self.assertIn("## Implementation evidence / resume state",
                      self.architect)
        self.assertIn("- `repo/path::symbol-or-section`:", self.architect)
        self.assertIn("- `repo/path::test-name`:", self.architect)

    def test_redteam_finding_has_candidate_repair_not_execution_authority(self):
        for heading in REQUIRED_SECTIONS["redteam"]:
            with self.subTest(heading=heading):
                self.assertIn("### " + heading, self.redteam)
        self.assertIn(
            "python3 ai/tools/handoff_contract.py redteam", self.redteam)
        self.assertIn("candidate input only", self.redteam)
        self.assertIn("next numbered\n`ai/notes/mailbox/NNN-to-fable.md`",
                      self.redteam)
        self.assertNotIn("NNN-to-<fable|opus>.md", self.redteam)
        self.assertIn("Architect GO/NO-GO is required", self.redteam)
        self.assertIn("- `repo/path::symbol-or-section`:", self.redteam)
        self.assertIn("- `repo/path::test-name`:", self.redteam)

    def test_implementer_preflights_and_stops_instead_of_designing(self):
        self.assertIn(
            "python3 ai/tools/handoff_contract.py architect", self.implementer)
        self.assertIn("Do not infer a design", self.implementer)
        self.assertIn("missing or conflicting\n   decisions", self.implementer)
        self.assertIn("Repair directive` is advisory input", self.implementer)
        self.assertIn("`Execution checkout`", self.implementer)
        self.assertIn("## Implementation evidence / resume\n   state",
                      self.implementer)

    def test_daemon_names_each_role_file_and_repeats_the_stop_boundary(self):
        architect = mailbox_daemon.agent_preamble(agent="fable")
        implementer = mailbox_daemon.agent_preamble(agent="opus")
        redteam = mailbox_daemon.agent_preamble(agent="sol")
        self.assertIn(".claude/FABLE_ROLE.md", architect)
        self.assertIn("Implementation\ndirective", architect)
        self.assertIn(".claude/OPUS_ROLE.md", implementer)
        self.assertIn("return a blocker", implementer)
        self.assertIn(".codex/REDTEAM_ROLE.md", redteam)
        self.assertIn("candidate input", redteam)
        assignment = (
            "MAILBOX-TICKET: closure\n\n"
            + mailbox_daemon.SECOND_IMPLEMENTER_MODE_SENTENCE + "\n")
        second = mailbox_daemon.agent_preamble(
            agent="sol", message=assignment)
        self.assertIn(".claude/OPUS_ROLE.md", second)
        self.assertIn("second Implementer", second)
        headed_assignment = (
            "MAILBOX-TICKET: closure\n\n"
            "### ARCHITECT_HANDOFF (relay)\n\n"
            + mailbox_daemon.SECOND_IMPLEMENTER_MODE_SENTENCE + "\n")
        self.assertTrue(mailbox_daemon.sol_second_implementer_assignment(
            message=headed_assignment))
        self.assertIn(
            "second Implementer",
            mailbox_daemon.agent_preamble(
                agent="sol", message=headed_assignment))
        quoted = (
            "MAILBOX-TICKET: closure\n\nReview the named delta.\n"
            + mailbox_daemon.SECOND_IMPLEMENTER_MODE_SENTENCE + "\n")
        self.assertIn(
            "bounded Red Team",
            mailbox_daemon.agent_preamble(agent="sol", message=quoted))
        headed_then_prose = (
            "MAILBOX-TICKET: closure\n\n"
            "### ARCHITECT_HANDOFF: READY FOR EXECUTION\n\n"
            "Review this quotation before acting.\n"
            + mailbox_daemon.SECOND_IMPLEMENTER_MODE_SENTENCE + "\n")
        self.assertFalse(mailbox_daemon.sol_second_implementer_assignment(
            message=headed_then_prose))
        with self.assertRaisesRegex(ValueError, "unknown mailbox agent"):
            mailbox_daemon.agent_preamble(agent="user")

    def test_manual_router_preserves_normal_and_second_implementer_roles(self):
        self.assertIn(".claude/OPUS_ROLE.md", self.router)
        self.assertIn(".codex/REDTEAM_ROLE.md", self.router)
        self.assertEqual(
            self.router.count('header="### IMPLEMENTER_HANDOFF:"'), 1)
        self.assertIn("INSTEAD OF Opus", self.router)
        self.assertIn('header="### ARCHITECT_REDTEAM_HANDOFF:"', self.router)
        self.assertIn("never directly to the Implementer", self.router)
        self.assertIn("--section may name only the validated", self.router)

    def test_reader_facing_guide_explains_why_the_plan_is_detailed(self):
        self.assertIn("The thinking roles must finish the plan", self.ai_readme)
        self.assertIn("Haiku, an open-source model", self.ai_readme)
        self.assertIn("Use your best judgment", self.ai_readme)
        self.assertIn("candidate, never a self-executing ruling",
                      self.ai_readme)


if __name__ == "__main__":
    unittest.main()
