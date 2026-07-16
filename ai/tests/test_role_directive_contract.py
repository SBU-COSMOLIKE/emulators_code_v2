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
        cls.gates_readme = read("ai/gates/README.md")
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

    def test_redteam_finding_note_is_detailed_persuasive_and_advisory(self):
        headings = (
            "High-level summary",
            "Affected behavior and code path",
            "Reproduction and evidence",
            "Impact and proposed severity",
            "Review scope and exclusions",
            "Proposed acceptance evidence",
            "Uncertainty and counterevidence",
            "Repair directive",
        )
        for name, source in (("Architect", self.architect),
                             ("Red Team", self.redteam),
                             ("workflow", self.conventions)):
            normalized = " ".join(source.split())
            for heading in headings:
                with self.subTest(name=name, heading=heading):
                    self.assertIn(heading, normalized)
            self.assertIn(
                "ai/notes/<plain-ticket-slug>-red-team-finding.md",
                normalized)
            self.assertIn("See further instructions at", normalized)

        redteam = " ".join(self.redteam.split())
        architect = " ".join(self.architect.split())
        conventions = " ".join(self.conventions.split())
        self.assertIn("Even if the Red Team is the most capable model",
                      redteam)
        self.assertIn("approve a commit, or veto", redteam)
        self.assertIn("Architect books `NEW TICKET` or `REOPEN` immediately",
                      redteam)
        self.assertIn("Admission is bookkeeping", redteam)
        self.assertIn("conserves Architect tokens", redteam)
        self.assertIn("Fabricated evidence is a failed review", redteam)
        self.assertIn("do not reproduce or substantively analyze", architect)
        self.assertIn("Only when priority later brings that ticket forward",
                      conventions)
        ai_readme = " ".join(self.ai_readme.split())
        self.assertIn("How the three bots work now", ai_readme)
        self.assertIn("Red Team is deliberately outside that approval path",
                      ai_readme)
        self.assertIn("This is why the Red Team is optional", ai_readme)
        self.assertIn("saves Architect tokens", ai_readme)
        for prohibited_failure in (
                "thin assertion", "rhetorical pressure", "inflated severity",
                "diary", "fabricated"):
            with self.subTest(prohibited_failure=prohibited_failure):
                self.assertIn(prohibited_failure, redteam.lower())
                self.assertIn(prohibited_failure, conventions.lower())

    def test_backlog_guard_is_architect_owned_and_documented(self):
        conventions = " ".join(self.conventions.split())
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        redteam = " ".join(self.redteam.split())

        self.assertIn("Protect the Architect-owned backlog", conventions)
        for command in (
                "backlog_guard.py initialize",
                "backlog_guard.py check",
                "backlog_guard.py seal"):
            with self.subTest(command=command):
                self.assertIn(command, conventions)
                self.assertIn(command, architect)
        self.assertIn("A mismatch is `NO-GO`", conventions)
        self.assertIn("records byte identity, not ticket truth", architect)
        for name, source in (("Implementer", implementer),
                             ("Red Team", redteam)):
            with self.subTest(role=name):
                self.assertIn("may read `ai/notes/backlog.md`", source)
                self.assertIn("never edit", source)
                self.assertIn("`initialize` or `seal`", source)
                self.assertIn("ai/tools/backlog_guard.py", source)

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
        architect = " ".join(self.architect.split())
        self.assertIn(
            "Architect severity decision: accept|upgrade|downgrade",
            architect)
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
        self.assertEqual(self.router.count("+ budget_prompt"), 3)
        self.assertIn("Zero removes the", router)
        self.assertIn("size cap only", router)
        self.assertIn("--severity", router)
        self.assertIn(
            '+ (severity_prompt if role_plan["uses_red_team"] else "")',
            router)
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

    def test_manual_router_audits_before_any_later_redteam_review(self):
        router = " ".join(self.router.split())
        self.assertIn(".claude/OPUS_ROLE.md", self.router)
        self.assertIn(".codex/REDTEAM_ROLE.md", self.router)
        self.assertEqual(
            self.router.count('header="### IMPLEMENTER_HANDOFF:"'), 1)
        self.assertIn("INSTEAD OF Opus", self.router)
        self.assertNotIn(
            'header="### ARCHITECT_REDTEAM_HANDOFF:"', self.router)
        self.assertIn(
            "this router never inserts Red Team between implementation and "
            "the Architect's audit", router)
        self.assertIn("Post-acceptance Red Team plan", router)
        self.assertIn("Only afterward, create a", router)
        self.assertIn(
            "separate Architect-authored Red Team handoff", router)
        self.assertIn(
            "Do not wait for Red Team before this audit", router)
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
        self.assertIn("Abstraction examples", self.readme_contract)
        self.assertIn("Use real examples to explain abstractions",
                      self.readme_contract)
        self.assertIn("a concrete input, filename, setting, command",
                      self.readme_contract)
        self.assertIn("the result the user can observe",
                      self.readme_contract)
        self.assertIn("A specialist folder README opens",
                      self.readme_contract)
        self.assertIn("Neighbor distinction", self.readme_contract)
        self.assertIn("An operational paragraph has one job",
                      self.readme_contract)
        self.assertIn("`test (gate in the code)` receives `NO-GO`",
                      self.readme_contract)
        self.assertIn("`test_parameter_table.py` gives the loader a table",
                      self.ai_readme)
        self.assertIn("the dataset-publication gate runs",
                      self.ai_readme)
        self.assertIn("**validation board** is the saved list of all gates",
                      self.ai_readme)

    def test_backlog_ticket_contract_is_human_first_and_count_safe(self):
        normalized = " ".join(self.conventions.split())
        self.assertIn("### Backlog ticket GO / NO-GO", self.conventions)
        for required_part in (
                "**High-level summary**",
                "**Current status**",
                "**What is already fixed**",
                "**What is missing**",
                "**Technical record for development tools**"):
            with self.subTest(required_part=required_part):
                self.assertIn(required_part, self.conventions)
        self.assertIn("at least three complete sentences in ordinary",
                      self.conventions)
        self.assertIn("normal purpose and one concrete example",
                      self.conventions)
        self.assertIn("exactly one linked line for each unfinished",
                      self.conventions)
        self.assertIn("every index link resolves to exactly one detailed open",
                      self.conventions)
        self.assertIn("A workstation-only check stays open",
                      self.conventions)
        self.assertIn("Uses only `unit 8`", self.conventions)
        self.assertIn("same turn as every state change", self.conventions)
        self.assertIn(
            "Architect acceptance then closes and commits it without waiting",
            normalized)
        self.assertIn("grouped in priority order: Critical first",
                      self.conventions)
        self.assertIn("A user-designated High feature comes before High bugs",
                      self.conventions)
        self.assertIn("Keep the five human-first parts", self.architect)
        self.assertIn("Classify before ordering", self.architect)
        self.assertIn("Architect GO closes without Red Team", self.architect)
        self.assertNotIn("A GO retires the line immediately", self.architect)
        self.assertNotIn("ledger stays countable one-liners", self.architect)

    def test_local_backlog_recreation_contract_is_exact_and_fail_closed(self):
        start = self.conventions.index(
            "#### Recreate the local backlog consistently")
        end = self.conventions.index(
            "The Architect updates the ticket in the same turn", start)
        contract = self.conventions[start:end]

        exact_skeleton = (
            "# Execution backlog\n\n"
            "This file is local to this clone and is not committed to "
            "GitHub. The Architect\n"
            "recreates it from this contract and updates it whenever a "
            "ticket changes.\n\n"
            "## Contents\n\n"
            "- [Open tickets](#open-tickets)\n"
            "- [Closed tickets](#closed-tickets)\n\n"
            "## How to read this backlog\n")
        self.assertIn(exact_skeleton, contract)
        self.assertIn(
            "Each line beginning `- OPEN` represents one unfinished ticket",
            contract)
        self.assertIn(
            "New discovery stops when ten or more Critical, High, and Medium",
            contract)
        self.assertIn(
            "more than one Critical Bug fix or more than ten High\n"
            "Bug fix tickets", contract)
        for priority in ("Critical", "High", "Medium", "Low"):
            with self.subTest(empty_priority=priority):
                self.assertIn("No open " + priority + " tickets.", contract)
        self.assertIn("No closed tickets.", contract)
        self.assertIn(
            "empty sentence and a ticket line never appear together", contract)
        for earlier, later in (
                ("# Open tickets", "## Open ticket index"),
                ("## Open ticket index", "### Critical"),
                ("### Critical", "### High"),
                ("### High", "### Medium"),
                ("### Medium", "### Low"),
                ("### Low", "# Closed tickets")):
            with self.subTest(earlier=earlier, later=later):
                self.assertLess(contract.index(earlier), contract.index(later))

        self.assertIn(
            "- OPEN **PRIORITY** **TYPE** — "
            "[Plain human title](#unique-anchor)", contract)
        self.assertIn(
            "`PRIORITY` is exactly `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`",
            contract)
        self.assertIn(
            "`TYPE` is\nexactly `BUG FIX` or `NEW FUNCTIONALITY`", contract)
        self.assertIn(
            "`CRITICAL` with `NEW FUNCTIONALITY`\nis invalid", contract)
        self.assertIn("does not copy an imported backlog\nblindly", contract)
        self.assertIn("validates\nand normalizes it to this contract", contract)
        self.assertIn(
            "lowercase ASCII letters, digits, and hyphens", contract)
        self.assertIn("must match byte for byte", contract)
        self.assertIn("**Ticket type: BUG FIX.**", contract)
        self.assertIn("**Ticket type: NEW FUNCTIONALITY.**", contract)
        self.assertIn("**Severity: PRIORITY.**", contract)
        self.assertIn("**Priority: PRIORITY.**", contract)

        template_start = contract.index(
            "Each detailed open ticket uses this exact heading order")
        template_end = contract.index(
            "The Architect writes a feature's user-supplied priority",
            template_start)
        template = contract[template_start:template_end]
        for earlier, later in (
                ('<a id="unique-anchor"></a>', "## Plain human title"),
                ("## Plain human title", "### High-level summary"),
                ("### High-level summary", "### Current status"),
                ("### Current status", "### What is already fixed"),
                ("### What is already fixed", "### What is missing"),
                ("### What is missing", "<details>"),
                ("<details>",
                 "<summary>Technical record for development tools</summary>")):
            with self.subTest(template_earlier=earlier,
                              template_later=later):
                self.assertLess(template.index(earlier), template.index(later))

        self.assertIn("Three or more short, complete sentences", contract)
        self.assertIn("normal\npurpose with a concrete example", contract)
        self.assertIn("current failure", contract)
        self.assertIn("user or scientific consequence", contract)
        self.assertIn(
            "Saved CMB progress can lose its multipole labels", contract)
        self.assertIn("example,\nnot an admitted ticket", contract)

        self.assertIn(
            "To close a ticket, the Architect removes its one "
            "index", contract)
        self.assertIn("changes `**OPEN.**` to\n`**CLOSED.**`", contract)
        self.assertIn("**Red Team reopen count: 0.**", contract)
        self.assertIn("commits the accepted Implementer fix without waiting "
                      "for Red Team", contract)
        self.assertIn(
            "At the end of each normal cycle, Red Team reviews", contract)
        self.assertIn("Backlog action: REOPEN", contract)
        self.assertIn("Nothing\nfor this ticket.", contract)
        self.assertIn("Malformed backlog state always fails closed", contract)
        self.assertIn("A malformed\nline is never ignored, guessed, or rewritten",
                      contract)

    def test_redteam_is_advisory_and_reopen_count_is_binding(self):
        normalized = " ".join(self.conventions.split())
        architect = " ".join(self.architect.split())
        redteam = " ".join(self.redteam.split())

        self.assertIn("Every ticket also keeps an integer named **Red Team "
                      "reopen count**", normalized)
        self.assertIn("It starts at `0` and never resets", normalized)
        self.assertIn("quick bookkeeping", normalized)
        self.assertIn("leave the deeper evidence review for a later "
                      "Architect turn", normalized)
        self.assertIn("The Architect still has the final word after that "
                      "immediate bookkeeping", normalized)
        self.assertIn("When the count becomes `6`", normalized)
        self.assertIn("automatically Low", normalized)
        self.assertIn("Red Team is always advisory", normalized)
        self.assertIn("Architect never waits for Red Team before committing",
                      normalized)
        self.assertIn(
            "Architect may start the next ticket while that advisory review "
            "is pending", normalized)
        self.assertIn(
            "finite watcher, however, does not count or exit that cycle "
            "until the correlated Red Team return exists", normalized)
        self.assertIn("Backlog action: NEW TICKET", normalized)
        self.assertIn("provisional priority", normalized)

        self.assertIn("Count every formal Red Team reopening request",
                      architect)
        self.assertIn("immediately increment the integer", architect)
        self.assertIn(
            "Do this bookkeeping without reproducing or substantively "
            "analyzing the finding", architect)
        self.assertIn("A value greater than five automatically makes the "
                      "ticket Low", architect)
        self.assertIn("Exercise final authority after the quick reopening",
                      architect)
        self.assertIn("Red Team is advisory and never supplies a required "
                      "GO", architect)
        self.assertIn("close and commit that ticket immediately", architect)
        self.assertIn("Backlog action: NEW TICKET", architect)

        self.assertIn("## Advisory review after the Architect closes a ticket",
                      self.redteam)
        self.assertIn("Backlog action: REOPEN", redteam)
        self.assertIn("Backlog action: NO CHANGE", redteam)
        self.assertIn("restore the open ticket, increment the counter",
                      redteam)
        self.assertIn("never supplies a required GO", redteam)
        self.assertIn("never blocks the Architect", redteam)
        self.assertIn("Backlog action: NEW TICKET", redteam)
        self.assertIn("marks the severity as provisional", redteam)
        self.assertIn("Red Team does not edit the backlog", redteam)

    def test_backlog_counts_do_not_mix_admission_and_emergency(self):
        self.assertIn(
            "Count open Critical, High, and Medium backlog\n   tickets",
            self.conventions)
        self.assertIn(
            "Open Low tickets and waiting mailbox files do not count",
            self.conventions)
        self.assertIn(
            "more than one\n   open bug is Critical or more than ten open "
            "bugs are High",
            self.conventions)
        self.assertIn(
            "High features,\n   Medium work, Low work, and waiting mailbox "
            "files do not contribute",
            self.conventions)
        self.assertIn(
            "An unclassified\n   open line fails closed", self.conventions)

    def test_high_severity_requires_unusual_concrete_harm(self):
        normalized = " ".join(self.conventions.split())
        architect = " ".join(self.architect.split())
        redteam = " ".join(self.redteam.split())
        self.assertIn("High is deliberately difficult to assign", normalized)
        self.assertIn("why Medium is not enough", normalized)
        self.assertIn("Urgency, a missing test, unfinished cleanup", normalized)
        self.assertIn("why Medium is insufficient", architect)
        self.assertIn("keep the system in emergency mode during ordinary "
                      "maintenance", architect)
        self.assertIn("why Medium is insufficient", redteam)
        self.assertIn("Inflating High would keep the system in emergency "
                      "mode", redteam)

    def test_permanent_workflow_records_runtime_governance(self):
        conventions = " ".join(self.conventions.split())
        architect = " ".join(self.architect.split())
        self.assertIn("there is no independent Red Team model option",
                      conventions)
        self.assertNotIn(
            "different model to Architect, Implementer, or Red Team",
            conventions)
        for phrase in (
                "### Discovery severity",
                "Medium is the default",
                "Harm and likelihood are separate judgments",
                "### Ticket character limit",
                "Unicode code points",
                "An exact-boundary result is accepted",
                "with no `--cycle` option, the watcher continues watching",
                "20-second Ctrl-C countdown",
                "Fix-only mode permits work that closes an existing ticket",
                "### Discovery demand and a second Implementer",
                "cannot audit the same ticket",
                "In normal advisory Red Team work, Sol does not edit tracked "
                "files or create a commit",
                "only for an explicit second-Implementer unit or another "
                "separately authorized tracked unit",
                "The configured CoCoA environment uses NumPy 1.x",
                "### Tests, gates, and the validation board",
                "shifts one saved multipole coordinate"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, conventions)
        self.assertIn("emergency count does not change Sol's role automatically",
                      architect)
        self.assertIn("leaving Sol idle is not a dispatch failure",
                      architect)
        self.assertNotIn("an idle [S] lane", architect)

    def test_project_note_records_role_cost_and_repository_ownership(self):
        project = read("ai/notes/project-and-history.md")
        normalized = " ".join(project.split())
        self.assertIn("The role system is optional", normalized)
        self.assertIn("token-heavy reading, editing, and test work", normalized)
        self.assertIn("Authority belongs to the role, not to a model name",
                      normalized)
        self.assertIn("This repository owns the Python emulators", normalized)
        self.assertIn("must not turn into a Fortran CAMB port", normalized)

    def test_gates_guide_keeps_tests_gates_and_board_concrete(self):
        self.assertIn("It is not limited to the cosmic-shear emulator",
                      self.gates_readme)
        self.assertIn("supplies saved CMB progress\nfiles without `dv_ell.npy`",
                      self.gates_readme)
        self.assertIn("python3 -m unittest", self.gates_readme)
        self.assertIn("python3 ai/gates/run_board.py --gate "
                      "dataset-publication", self.gates_readme)
        self.assertIn("[harness] GATE dataset-publication: PASS",
                      self.gates_readme)
        self.assertIn("`pre-manifest`", self.gates_readme)
        self.assertIn("`UNAVAILABLE` is additional information inside a PASS",
                      self.gates_readme)
        self.assertIn("dataset-publication.20260716-143012-123456.log",
                      self.gates_readme)
        self.assertIn("`.inprogress` added at the end",
                      self.gates_readme)
        self.assertNotIn("\npython ", self.gates_readme)

    def test_every_python_change_uses_the_mandatory_style_gate(self):
        contract_path = "ai/notes/python-changes-go-no-go.md"
        self.assertIn(contract_path, self.architect_command)
        self.assertGreaterEqual(self.architect.count(contract_path), 4)
        self.assertIn(contract_path, self.implementer)
        self.assertIn(contract_path, self.implementer_command)
        self.assertIn(contract_path, self.redteam)
        self.assertIn("before writing the directive", self.architect)
        self.assertIn("Python-change review-time check", self.architect)
        self.assertIn("hot or cold", self.architect)
        self.assertIn("Python style evidence", self.implementer)
        self.assertIn("full changed symbols", self.redteam)
        self.assertIn("style is a release condition", self.python_contract)
        self.assertIn("Architect review before dispatch", self.python_contract)
        self.assertIn("Architect review before final verdict",
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
