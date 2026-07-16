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
        cls.claude_entry = read("CLAUDE.md")
        cls.architect_command = read(".claude/commands/architect.md")
        cls.implementer_command = read(".claude/commands/implementer.md")
        cls.readme_contract = read("ai/notes/readme-go-no-go.md")
        cls.python_contract = read("ai/notes/python-changes-go-no-go.md")
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
            'python3 "$MAILBOX_HANDOFF_CONTRACT" architect', self.architect)
        self.assertIn("decision-complete implementation", self.architect)
        self.assertIn("lower-capability Implementer", self.architect)
        self.assertIn("complete, self-contained repair packet", self.architect)
        self.assertIn("## Implementation evidence / resume state",
                      self.architect)
        self.assertIn("- `repo/path::symbol-or-section`:", self.architect)
        self.assertIn("- `repo/path::test-name`:", self.architect)
        self.assertIn("--max RUNTIME_N", self.architect)

    def test_user_contacts_only_architect_and_courier_cannot_reauthor(self):
        self.assertIn("## Sole user contact", self.architect)
        self.assertIn(
            "The user never addresses the Implementer or Red Team directly",
            self.architect)
        self.assertIn("Please instruct the Red Team to do a widespread",
                      self.architect)
        self.assertIn("copy an unchanged handoff", self.architect)
        self.assertIn("## User-contact boundary", self.implementer)
        self.assertIn("only to the Architect", self.implementer)
        self.assertIn("do not act on it", self.implementer)
        self.assertIn("## User-contact boundary", self.redteam)
        self.assertIn("A direct user request does\nnot start Red Team work",
                      self.redteam)
        self.assertIn("unchanged Architect handoff", self.redteam)
        self.assertIn("USER CONTACT RULE", mailbox_daemon.PREAMBLE)
        self.assertIn("Only the Architect\nturn may interpret or answer",
                      mailbox_daemon.PREAMBLE)
        self.assertIn(
            "The user gives every ticket request and correction only to\n"
            "the Architect", self.claude_entry)
        self.assertIn("user-authored or\nedited imitation is not valid",
                      self.claude_entry)
        self.assertIn("only to the Architect", self.implementer_command)
        self.assertIn("unchanged Architect-authored handoff",
                      self.implementer_command)
        self.assertIn("return a blocker to the Architect",
                      self.implementer_command)
        self.assertIn("every downstream Implementer or Red Team handoff",
                      self.architect_command)

    def test_redteam_finding_has_candidate_repair_not_execution_authority(self):
        for heading in REQUIRED_SECTIONS["redteam"]:
            with self.subTest(heading=heading):
                self.assertIn("### " + heading, self.redteam)
        self.assertIn(
            'python3 "$MAILBOX_HANDOFF_CONTRACT" redteam', self.redteam)
        self.assertIn("candidate input only", self.redteam)
        self.assertIn("next numbered\n`ai/notes/mailbox/NNN-to-fable.md`",
                      self.redteam)
        self.assertNotIn("NNN-to-<fable|opus>.md", self.redteam)
        self.assertIn("Architect GO/NO-GO is required", self.redteam)
        self.assertIn("- `repo/path::symbol-or-section`:", self.redteam)
        self.assertIn("- `repo/path::test-name`:", self.redteam)
        self.assertIn("--max RUNTIME_N", self.redteam)

    def test_discovery_severity_keeps_user_redteam_and_architect_roles(self):
        for phrase in (
                "User severity setting",
                "Red Team severity",
                "Likelihood: probable|improbable",
                "Likelihood evidence",
                "Meets user setting: yes|no"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.redteam)
        self.assertIn("severely impacts core functionality", self.redteam)
        self.assertIn("causes data loss", self.redteam)
        self.assertIn("halts system operations", self.redteam)
        self.assertIn("makes the science wrong", self.redteam)
        self.assertIn("merely theoretical or improbable edge case",
                      self.redteam)
        self.assertIn("accepts, upgrades, or downgrades", self.redteam)
        self.assertIn("Architect severity decision: accept|upgrade|downgrade",
                      self.architect)
        self.assertIn("Ticket decision: GO|NO-GO", self.architect)
        self.assertIn("The Red Team never", self.architect)
        self.assertEqual(mailbox_daemon.DEFAULT_DISCOVERY_SEVERITY, "medium")
        self.assertEqual(
            mailbox_daemon.DISCOVERY_SEVERITIES,
            ("high", "medium", "low"))
        banner = mailbox_daemon.dispatch_banner(
            store_max=1, newer_in_lane=0, previous_timeout_minutes=None,
            discovery_severity="high", saved_discovery=True)
        self.assertIn(
            "user's saved minimum severity for this discovery: high",
            banner)
        self.assertIn("The Architect accepts, upgrades, or downgrades",
                      banner)
        self.assertNotIn("MAILBOX-SEVERITY: high|medium|low",
                         mailbox_daemon.PREAMBLE)

    def test_implementer_preflights_and_stops_instead_of_designing(self):
        self.assertIn(
            'python3 "$MAILBOX_HANDOFF_CONTRACT" architect', self.implementer)
        self.assertIn("Do not infer a design", self.implementer)
        self.assertIn("missing or conflicting\n   decisions", self.implementer)
        self.assertIn("Repair directive` is advisory input", self.implementer)
        self.assertIn("`Execution checkout`", self.implementer)
        self.assertIn("## Implementation evidence / resume\n   state",
                      self.implementer)
        self.assertIn("--max RUNTIME_N", self.implementer)

    def test_character_limit_never_licenses_obfuscated_or_partial_work(self):
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        redteam = " ".join(self.redteam.split())
        for phrase in (
                "minification", "shortened names", "packed statements",
                "collapsed control flow", "dense expressions or metaprogramming",
                "removed comments or docstrings", "removed tests or type information",
                "stripped whitespace", "omitted errors or documentation",
                "partial fix"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, architect)
                self.assertIn(phrase, implementer)
                self.assertIn(phrase, redteam)
        self.assertIn("smallest complete readable tested unit", architect)
        self.assertIn("ask the user", architect)
        self.assertIn("C programmer and a physics undergraduate",
                      architect)
        self.assertIn("C programmer and a physics undergraduate",
                      implementer)
        self.assertIn("C programmer and a physics undergraduate", redteam)
        self.assertIn("ticket_change_guard.py", architect)
        self.assertIn("ticket_change_guard.py", implementer)
        self.assertIn("ticket_change_guard.py", redteam)
        self.assertIn("added, deleted, total, and limit", architect)
        self.assertIn("added, deleted, total, and limit", implementer)
        self.assertIn("added, deleted, total, and limit", redteam)
        self.assertIn("`0` removes the size cap only", architect)
        self.assertIn("`0` means no size cap", implementer)
        self.assertIn("`0` removes only the size cap", redteam)
        self.assertIn("Character-change result", implementer)
        self.assertIn("Character-change result", redteam)
        self.assertIn("Character-change budget", self.architect_command)
        self.assertIn("MAILBOX_MAX_CHARACTERS", architect)
        self.assertIn("MAILBOX_MAX_CHARACTERS", implementer)
        self.assertIn("MAILBOX_MAX_CHARACTERS", redteam)
        self.assertIn("same turn that can issue `GO` and land", architect)
        self.assertIn("guard's printed `candidate commit` is still `HEAD`",
                      architect)
        for name, source in (("Architect", architect),
                             ("Implementer", implementer),
                             ("Red Team", redteam)):
            with self.subTest(role=name):
                self.assertIn("MAILBOX_HANDOFF_CONTRACT", source)
                self.assertIn("MAILBOX_TICKET_CHANGE_GUARD", source)
                self.assertIn("MAILBOX_SHARED_NOTES", source)
                self.assertIn(
                    "size limit disabled (0); measurement skipped", source)
                self.assertIn("never invent", source)
                self.assertIn("relative `ai/tools/`", source)
                self.assertIn("`ai/notes/`", source)

    def test_manual_router_binds_budget_to_validator_and_every_prompt(self):
        router = " ".join(self.router.split())
        self.assertIn("--max", self.router)
        self.assertIn("expected_max=expected_max", self.router)
        self.assertIn('budget = directive["character_change_budget"]',
                      self.router)
        self.assertEqual(self.router.count("+ budget_prompt"), 4)
        self.assertIn("Zero removes the", router)
        self.assertIn("size cap only", router)
        self.assertIn("--severity", router)
        self.assertIn("+ severity_prompt", router)
        self.assertIn(
            '+ (severity_prompt if redteam_block else "")', router)
        self.assertIn("User severity setting for any new Red Team ticket",
                      router)
        self.assertIn("accepts, upgrades, or downgrades", router)

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

    def test_readme_changes_require_the_architect_gate_twice(self):
        contract_path = "ai/notes/readme-go-no-go.md"
        self.assertIn(contract_path, self.architect_command)
        self.assertGreaterEqual(self.architect.count(contract_path), 4)
        self.assertIn("before writing the directive", self.architect)
        self.assertIn("again before final `GO`", self.architect)
        self.assertIn(contract_path, self.implementer)
        self.assertIn(contract_path, self.redteam)
        self.assertIn(contract_path, self.conventions)
        self.assertIn("physics undergraduate", self.readme_contract)
        self.assertIn("The Architect reads this file twice",
                      self.readme_contract)
        self.assertIn("comments, docstrings, command help",
                      self.readme_contract)
        self.assertIn("GO` for the directive", self.readme_contract)
        self.assertIn("NO-GO", self.readme_contract)
        self.assertIn("Hard-zero words", self.readme_contract)
        self.assertIn("Do not use an AI detector", self.readme_contract)

    def test_every_python_change_uses_the_mandatory_style_gate(self):
        contract_path = "ai/notes/python-changes-go-no-go.md"
        self.assertIn(contract_path, self.architect_command)
        self.assertGreaterEqual(self.architect.count(contract_path), 4)
        self.assertIn(contract_path, self.implementer)
        self.assertIn(contract_path, self.implementer_command)
        self.assertIn(contract_path, self.redteam)
        self.assertIn("before writing the directive", self.architect)
        self.assertIn("Python-change review-time gate", self.architect)
        self.assertIn("hot or cold", self.architect)
        self.assertIn("Python style evidence", self.implementer)
        self.assertIn("full changed symbols", self.redteam)
        self.assertIn("style is a release condition", self.python_contract)
        self.assertIn("Architect gate before dispatch", self.python_contract)
        self.assertIn("Architect gate before final verdict",
                      self.python_contract)
        self.assertIn("Hard NO-GO conditions", self.python_contract)
        self.assertIn("character budget", self.python_contract)
        self.assertIn("The Implementer and Red Team never edit",
                      self.python_contract)

    def test_every_execution_role_keeps_permanent_notes_off_limits(self):
        guard = "ai/tools/permanent_note_guard.py"
        self.assertGreaterEqual(self.architect.count(guard), 3)
        self.assertIn("for any ticket type", self.architect)
        self.assertIn("all twelve exact paths", self.implementer)
        self.assertIn("regardless of ticket type", self.implementer)
        self.assertIn("regardless of ticket type", self.redteam)
        self.assertIn(guard, self.redteam)
        self.assertIn("PERMANENT-NOTE-GUARD PASS", self.architect)


if __name__ == "__main__":
    unittest.main()
