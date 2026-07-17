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
        cls.tools_readme = read("ai/tools/README.md")
        cls.tests_readme = read("ai/tests/README.md")
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
        self.assertIn("complete, self-contained repair packet",
                      " ".join(self.architect.split()))
        self.assertIn("## Implementation evidence / resume state",
                      self.architect)
        self.assertIn("- `repo/path::symbol-or-section`:", self.architect)
        self.assertIn("- `repo/path::test-name`:", self.architect)
        self.assertIn("--max RUNTIME_N", self.architect)

    def test_public_admission_has_three_exact_architect_outcomes(self):
        """The real Architect is taught every daemon-accepted outcome."""
        token = "0001-to-fable.md@" + "a" * 64
        required_receipt = (
            "MAILBOX-RETURN: architect-no-ticket\n"
            "MAILBOX-ADMISSION: EXACT-TOKEN\n"
            "MAILBOX-DECISION: NO TICKET")
        self.assertIn(required_receipt, self.architect)
        self.assertIn("exactly one of these outcomes", self.architect)
        self.assertIn("one `to-opus` ticket handoff", self.architect)
        self.assertIn("one `to-sol` discovery request", self.architect)
        self.assertIn("never remain silent", self.architect)
        self.assertIn("MAILBOX_ARCHITECT_ADMISSION",
                      self.architect_command)
        self.assertIn("three-outcome rule", self.architect_command)

        terminal = (
            "MAILBOX-RETURN: architect-no-ticket\n"
            "MAILBOX-ADMISSION: " + token + "\n"
            "MAILBOX-DECISION: NO TICKET\n\n"
            "No tracked change is needed.\n")
        self.assertIsNone(
            mailbox_daemon.public_architect_no_ticket_problem(
                message=terminal, expected_token=token))
        self.assertIsNotNone(
            mailbox_daemon.public_architect_no_ticket_problem(
                message=terminal,
                expected_token="0002-to-fable.md@" + "b" * 64))

        sol = mailbox_daemon.sol_ticket_payload(
            ticket_kind="discovery",
            text=("MAILBOX-ADMISSION: " + token
                  + "\nReview the named change only."),
            discovery_severity="medium", discovery_scope="bounded")
        self.assertIsNone(
            mailbox_daemon.public_architect_sol_outcome_problem(
                message=sol, expected_token=token))

    def test_implementer_always_attempts_the_planned_subagents(self):
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        architect_command = " ".join(self.architect_command.split())
        implementer_command = " ".join(self.implementer_command.split())

        self.assertIn("Plan bounded Implementer subagents", architect)
        self.assertIn("reproducing the failure and collecting evidence",
                      architect)
        self.assertIn("non-overlapping file ownership", architect)
        self.assertIn("personally run the final combined validation commands",
                      architect)
        self.assertIn("launch every planned helper before making any "
                      "Integrator-owned implementation edit", architect)
        self.assertIn("Independent helpers with non-overlapping ownership run "
                      "concurrently", architect)
        self.assertIn("only then must personally run the final combined "
                      "validation commands", architect)
        self.assertIn("Only then may a runtime with no subagent support",
                      architect)
        self.assertIn("The first directive always contains named subagent jobs",
                      architect)
        self.assertIn("Prior Implementer subagent launch failure", architect)
        self.assertIn("full source cycle and SHA-256", architect)
        self.assertIn("an unresolved blocked return is always `NO-GO`",
                      architect)
        self.assertIn("fabricated delegation", architect)

        self.assertIn("Use the directive's bounded subagent plan", implementer)
        self.assertIn("You remain the Integrator", implementer)
        self.assertIn("must launch every helper named in `Parallel work plan` before "
                      "making any Integrator-owned implementation edit",
                      implementer)
        self.assertIn("run independent helper jobs concurrently", implementer)
        self.assertIn("Only after integration do you personally run the final "
                      "combined validation commands", implementer)
        self.assertIn("personally run the final combined validation commands",
                      implementer)
        self.assertIn("Never claim delegation that did not happen", implementer)
        self.assertIn("make no implementation edit", implementer)
        self.assertIn("Wait for the Architect to revise and revalidate",
                      implementer)
        self.assertIn("cannot support final `GO`", implementer)
        self.assertIn("Subagent work:", implementer)
        self.assertIn("#### Subagent return `exact-planned-name`", implementer)
        self.assertIn("An unplanned, missing, duplicate, or renamed return is "
                      "`NO-GO`", architect)

        self.assertIn("Every implementation directive must give the "
                      "Implementer a bounded subagent plan", architect_command)
        self.assertIn("Require the Implementer to launch that plan",
                      architect_command)
        self.assertIn("Never predeclare the runtime incapable",
                      architect_command)
        self.assertIn("mandatory even for a small edit", implementer_command)
        self.assertIn("Every valid directive contains a bounded `Parallel "
                      "work plan`", implementer_command)
        self.assertIn("You must attempt every exact named Subagent block",
                      implementer_command)
        self.assertIn("- **Subagent work:**", implementer_command)
        self.assertIn("- Launch: `required before implementation edits`",
                      self.conventions)
        self.assertIn("#### Subagent `failure-reproducer`", self.conventions)
        self.assertIn("- Capability checked:", self.conventions)
        self.assertIn("- Source cycle:", self.conventions)
        self.assertIn("- Source handoff SHA-256:", self.conventions)
        self.assertIn("One editing helper owns the whole file",
                      self.conventions)
        self.assertIn("prior same-cycle IMPLEMENTER_HANDOFF checkpoint",
                      self.conventions)
        self.assertIn("The ticket is small", self.conventions)

    def test_first_failed_subagent_launch_is_exact_and_sha_bound(self):
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        architect_command = " ".join(self.architect_command.split())
        ai_readme = " ".join(self.ai_readme.split())
        tools_readme = " ".join(self.tools_readme.split())
        tests_readme = " ".join(self.tests_readme.split())

        for name, source in (
                ("Implementer role", implementer),
                ("tools README", tools_readme),
                ("tests README", tests_readme)):
            with self.subTest(source=name):
                self.assertIn("first", source.lower())
                self.assertIn("before", source.lower())
                self.assertIn("`IMPLEMENTER_HANDOFF`", source)
                self.assertIn("`Subagent work`", source)
                self.assertIn("`Capability checked`", source)
                self.assertIn("`Attempted operation`", source)
                self.assertIn("`Raw failure`", source)

        self.assertIn("If the first helper cannot start", ai_readme)
        self.assertIn("stops before editing", ai_readme)
        self.assertIn("exact helper-failure record", ai_readme)

        first_failure_rows = (
            "- Capability checked: `the exact launch capability`\n"
            "- Attempted operation: The concrete first subagent launch "
            "attempted before editing.\n"
            "- Raw failure: `the unchanged first runtime failure`")
        first_failure_rows_compact = " ".join(first_failure_rows.split())
        self.assertIn(first_failure_rows_compact, implementer)
        self.assertIn(first_failure_rows_compact, architect)

        for name, source in (("Architect role", architect),
                             ("Architect command", architect_command)):
            with self.subTest(source=name):
                self.assertIn("SHA-256", source)
                self.assertIn("character-for-character", source)
                self.assertIn("Prior Implementer subagent launch failure",
                              source)
                self.assertIn("Parallel work plan", source)
                self.assertIn("do not invent", source.lower())

        handoff_start = self.implementer.index(
            "### IMPLEMENTER_HANDOFF: REQUESTING REVIEW")
        evidence_start = self.implementer.index(
            "- **Subagent work:**", handoff_start)
        failure_start = self.implementer.index(
            "- Capability checked: `the exact launch capability`",
            evidence_start)
        evidence_end = self.implementer.index(
            "- **Blockers/findings:**", failure_start)
        self.assertLess(evidence_start, failure_start)
        self.assertLess(failure_start, evidence_end)

    def test_pipeline_uses_isolated_immutable_git_snapshots(self):
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        redteam = " ".join(self.redteam.split())
        architect_command = " ".join(self.architect_command.split())
        implementer_command = " ".join(self.implementer_command.split())

        self.assertIn("Only the Implementer edits tracked source", architect)
        self.assertIn("MAILBOX_CANDIDATE_COMMIT", architect)
        self.assertIn("MAILBOX_AUDIT_WORKTREE", architect)
        self.assertIn("--architect-audit", architect)
        self.assertIn('--candidate "$MAILBOX_CANDIDATE_COMMIT"', architect)
        self.assertIn("Never audit the Implementer's moving `HEAD`", architect)
        self.assertIn(
            "MAILBOX-RETURN: architect-go\n"
            "MAILBOX-CYCLE: THE-SAME-CYCLE\n"
            "MAILBOX-CANDIDATE: MAILBOX_CANDIDATE_COMMIT\n"
            "MAILBOX-MODE: normal\n"
            "MAILBOX-DECISION: GO",
            self.architect)
        self.assertNotIn("MAILBOX-RETURN: architect-" + "commit", architect)
        self.assertNotIn("git merge --" + "squash", architect)
        self.assertIn("You do not create or name the landing commit", architect)
        self.assertIn("parent daemon prepares a squash landing L", architect)
        self.assertIn("commit identity differs from C", architect)
        self.assertIn("attached to `main`, clean, and unchanged", architect)
        self.assertIn("bounded non-force push attempt", architect)
        self.assertIn("does not reopen the ticket", architect)
        self.assertLess(
            architect.index("queues one bounded Red Team closure request"),
            architect.index("makes one bounded non-force push attempt"))
        self.assertIn("The daemon restores that cycle's execution lane from "
                      "its saved candidate", architect)
        self.assertIn("Other active candidate refs", architect)

        self.assertIn("MAILBOX_EXECUTION_WORKTREE", implementer)
        self.assertIn("MAILBOX_IMPLEMENTER_WORKTREE", implementer)
        self.assertIn("Do not run `git reset`, `git switch`, or `git checkout`",
                      implementer)
        self.assertIn("Commit only the named ticket's tracked changes",
                      implementer)
        self.assertIn("Other cycles keep separate candidate refs", implementer)
        self.assertIn("The daemon restores this cycle's execution lane from its "
                      "saved candidate", implementer)

        self.assertIn("Only the Implementer edits tracked source", redteam)
        self.assertIn("exact landing commit L", redteam)
        self.assertIn("dispatch-provided isolated audit snapshot", redteam)
        self.assertIn("never edits, commits, amends, merges, resets, or switches",
                      redteam)

        self.assertIn("MAILBOX_CANDIDATE_COMMIT", architect_command)
        self.assertIn("MAILBOX_AUDIT_WORKTREE", architect_command)
        self.assertIn("Do not merge, commit, update a ref, push, or touch the "
                      "user's checkout", architect_command)
        self.assertIn(
            "records the local landing, safely advances every clean idle role "
            "baseline to L, queues optional Sol review, and "
            "attempts one bounded non-force push", architect_command)
        self.assertIn("MAILBOX_EXECUTION_WORKTREE", implementer_command)
        self.assertIn("MAILBOX_IMPLEMENTER_WORKTREE", implementer_command)

        self.assertIn("ticket B", architect)
        self.assertIn("ticket A", architect)
        self.assertIn("earlier daemon-recorded landing", architect)

        for name, source in (("Architect role", architect),
                             ("Architect command", architect_command),
                             ("Implementer role", implementer),
                             ("Red Team role", redteam)):
            with self.subTest(source=name):
                self.assertNotIn("MAILBOX-RETURN: architect-" + "commit", source)

        expected_go = (
            "MAILBOX-RETURN: architect-go\n"
            "MAILBOX-CYCLE: cycle-a@" + "1" * 40 + "\n"
            "MAILBOX-CANDIDATE: " + "2" * 40 + "\n"
            "MAILBOX-MODE: normal\n"
            "MAILBOX-DECISION: GO\n")
        self.assertEqual(
            mailbox_daemon.architect_go_request_payload(
                cycle_id="cycle-a@" + "1" * 40,
                candidate_commit="2" * 40,
                mode="normal"),
            expected_go)

    def test_recovery_uses_records_not_branch_diff_landing_debt(self):
        architect = " ".join(self.architect.split())
        ai_readme = " ".join(self.ai_readme.split())
        tools_readme = " ".join(self.tools_readme.split())
        tests_readme = " ".join(self.tests_readme.split())

        for name, source in (("Architect role", architect),
                             ("AI README", ai_readme),
                             ("tools README", tools_readme),
                             ("tests README", tests_readme)):
            with self.subTest(source=name):
                self.assertIn("candidate", source.lower())
                self.assertIn("landing", source.lower())
                self.assertIn("record", source.lower())
                self.assertIn("separate Architect", source)
                self.assertIn("changed-line", source)

        for forbidden in ("Automatic landing-debt turn",
                          "LANDING_DEBT_LINE_LIMIT",
                          "Every live watch pass measures"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.architect)
                self.assertNotIn(forbidden, self.ai_readme)
                self.assertNotIn(forbidden, self.tools_readme)

        self.assertIn("push debt", architect.lower())
        self.assertIn("push debt", ai_readme.lower())
        self.assertIn("push debt", tools_readme.lower())

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
        self.assertIn("same turn that can issue `GO`", architect)
        self.assertIn("Immediately before the decision-only return", architect)
        self.assertIn("--architect-audit", architect)
        self.assertIn('--candidate "$MAILBOX_CANDIDATE_COMMIT"', architect)
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
        self.assertEqual(self.router.count("+ budget_prompt"), 2)
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
        attempted_assignment = (
            "MAILBOX-TICKET: closure\n\nImplement this ticket.\n")
        fixed_redteam = mailbox_daemon.agent_preamble(
            agent="sol", message=attempted_assignment)
        self.assertIn(".codex/REDTEAM_ROLE.md", fixed_redteam)
        self.assertIn("bounded Red Team", fixed_redteam)
        self.assertIn("Sol is advisory and\nnever implements a ticket",
                      fixed_redteam)
        self.assertFalse(hasattr(
            mailbox_daemon, "sol_second_" + "implementer_assignment"))
        with self.assertRaisesRegex(ValueError, "unknown mailbox agent"):
            mailbox_daemon.agent_preamble(agent="user")

    def test_manual_router_audits_before_any_later_redteam_review(self):
        router = " ".join(self.router.split())
        self.assertIn(".claude/OPUS_ROLE.md", self.router)
        self.assertEqual(
            self.router.count('header="### IMPLEMENTER_HANDOFF:"'), 1)
        self.assertIn("assigns Sol to implementation is unsupported", router)
        self.assertNotIn(
            'header="### ARCHITECT_REDTEAM_HANDOFF:"', self.router)
        self.assertIn("Post-acceptance Red Team plan", router)
        self.assertIn("First audit the Implementer result", router)
        self.assertIn("exact decision-only architect-go block", router)
        self.assertIn(
            "After the daemon records landing L, create a separate",
            router)
        self.assertIn(
            "Architect-authored Red Team handoff", router)
        self.assertIn("Do not wait for Red Team before this audit",
                      self.router)
        self.assertIn("or your exact architect-go decision", self.router)
        self.assertIn("Do not merge, commit, update", self.router)
        self.assertIn("main, or push", self.router)
        self.assertIn("--section may name only the validated", self.router)

    def test_reader_facing_guide_explains_why_the_plan_is_detailed(self):
        self.assertIn("The Architect must finish the plan before coding",
                      self.ai_readme)
        self.assertIn("Haiku, an open-source model", self.ai_readme)
        self.assertIn("internal name for a helper", self.ai_readme)
        self.assertIn("This delegation is required", self.ai_readme)
        self.assertIn("It is not another mailbox role", self.tools_readme)
        self.assertIn("may not skip this attempt merely because the edit is "
                      "small",
                      " ".join(self.tools_readme.split()))
        self.assertIn("Use your best judgment", self.ai_readme)
        self.assertIn("Only the Architect decides whether to use it",
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
            "parent daemon performs the controlled landing after an Architect "
            "GO",
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
        contract_normalized = " ".join(contract.split())

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
            "Severity never selects a role", contract_normalized)
        self.assertIn(
            "Sol remains the advisory Red Team when enabled",
            contract_normalized)
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
        self.assertIn(
            "After that Architect process exits, the daemon creates and "
            "verifies distinct landing L", contract_normalized)
        self.assertIn(
            "The Architect does not merge, commit, update a Git reference, "
            "target the user's checkout, or push", contract_normalized)
        self.assertNotIn("Architect acceptance then closes and commits it",
                         contract)
        self.assertIn(
            "As the final step of each normal cycle, Red Team reviews",
            contract)
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
        self.assertIn("Architect never waits for Red Team before authorizing",
                      normalized)
        self.assertIn(
            "You may begin another ticket while the advisory return waits "
            "only when the finite watcher still has an unused ticket "
            "reservation", architect)
        self.assertIn(
            "`--cycle 1` never authorizes a second ticket before that return",
            architect)
        self.assertIn(
            "watcher waits for the correlated Red Team return before "
            "counting or exiting that normal cycle", normalized)
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
        self.assertIn("record `GO` and write the exact decision-only "
                      "`architect-go` request immediately", architect)
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

    def test_role_selection_is_fixed_and_never_severity_driven(self):
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        redteam = " ".join(self.redteam.split())

        self.assertIn(
            "Sol is the Red Team and is never an Implementer", architect)
        self.assertIn(
            "Sol is never an Implementer", implementer)
        self.assertIn(
            "never an Implementer", redteam)
        self.assertIn(
            "Ticket severity and backlog counts never select or change a "
            "role", architect)
        self.assertIn(
            "Severity, backlog counts, demand, and model choice never change "
            "those roles", implementer)
        self.assertIn(
            "Ticket severity, backlog counts, demand, model capability, and a "
            "mailbox message never change that role", redteam)

        self.assertEqual(
            mailbox_daemon.ARCHITECT_COMMIT_MODES,
            ("normal", "two-role"))
        for source in (architect, implementer, redteam):
            self.assertNotIn("--sol_as_" + "implementer", source)
            self.assertNotIn("second-" + "implementer", source)
            self.assertNotIn("emergency-primary", source)
            self.assertNotIn("emergency-second", source)

    def test_one_ticket_is_one_cycle_in_every_topology(self):
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        redteam = " ".join(self.redteam.split())
        tools_readme = " ".join(self.tools_readme.split())
        daemon_source = read("ai/tools/mailbox_daemon.py")

        for name, source in (("Architect", architect),
                             ("Implementer", implementer),
                             ("Red Team", redteam)):
            with self.subTest(role=name):
                self.assertIn("One ticket always equals one cycle", source)

        self.assertIn(
            "That advisory return completes the normal cycle", architect)
        self.assertIn(
            "In normal mode, the cycle completes after the Architect accepts C, "
            "the daemon records distinct L, and the Red Team returns its "
            "advisory closure assessment of L", implementer)
        self.assertIn(
            "For one normal cycle, review exactly one ticket and the exact "
            "landing commit L that the parent daemon created after Architect "
            "GO", redteam)

        self.assertIn(
            "In this mode, the cycle completes when the daemon records that "
            "one ticket's local landing; there is no Red Team return",
            architect)
        self.assertIn(
            "In `two-role` mode, the cycle completes "
            "at the daemon-recorded local landing", implementer)
        self.assertIn(
            "A watch started with `--skip-redteam` has no Sol work and "
            "completes each cycle at the daemon's recorded local landing",
            redteam)
        self.assertIn(
            "without Red Team it ends when the daemon records local landing "
            "L", tools_readme)
        self.assertIn(
            "In a mode without Red Team it ends when the daemon records "
            "local landing L", daemon_source)
        self.assertIn('"records local landing L; 0 "', daemon_source)

    def test_finite_cycle_limit_is_topology_bound_across_restarts(self):
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        redteam = " ".join(self.redteam.split())

        self.assertIn(
            "A positive cycle limit is valid in both topologies", architect)
        self.assertIn("remains binding across a watcher restart", implementer)
        self.assertIn("remain binding across watcher restarts", redteam)
        for name, source in (("Architect", architect),
                             ("Implementer", implementer),
                             ("Red Team", redteam)):
            with self.subTest(role=name):
                lowered = source.lower()
                self.assertIn("active ticket reservations", lowered)
                self.assertIn("completed cycles", lowered)

    def test_backlog_counts_control_discovery_admission_only(self):
        self.assertIn(
            "Count open Critical, High, and Medium backlog\n   tickets",
            self.conventions)
        self.assertIn(
            "Open Low tickets and waiting mailbox files do not count",
            self.conventions)
        self.assertIn(
            "An unclassified\n   open line fails closed", self.conventions)
        normalized = " ".join(self.conventions.split())
        self.assertIn("Sol is never reassigned as an Implementer", normalized)

    def test_high_severity_requires_unusual_concrete_harm(self):
        normalized = " ".join(self.conventions.split())
        architect = " ".join(self.architect.split())
        redteam = " ".join(self.redteam.split())
        self.assertIn("High is deliberately difficult to assign", normalized)
        self.assertIn("why Medium is not enough", normalized)
        self.assertIn("Urgency, a missing test, unfinished cleanup", normalized)
        self.assertIn("why Medium is insufficient", architect)
        self.assertIn("Ticket severity, backlog counts, demand, model "
                      "capability", architect)
        self.assertIn("why Medium is insufficient", redteam)
        self.assertIn("Ticket severity, backlog counts, demand, model "
                      "capability", redteam)

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
                "Sol is never reassigned as an Implementer",
                "The Red Team reviews an immutable snapshot of `L`",
                "does not edit those artifacts",
                "The configured CoCoA environment uses NumPy 1.x",
                "### Tests, gates, and the validation board",
                "shifts one saved multipole coordinate"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, conventions)
        self.assertIn("Sol is the Red Team and is never an Implementer",
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
        architect = " ".join(self.architect.split())
        self.assertGreaterEqual(self.architect.count(guard), 3)
        self.assertIn("for any ticket type", self.architect)
        self.assertIn("eleven permanent notes are a separate Architect-owned "
                      "policy surface", architect)
        self.assertIn("edit and commit those notes in the Architect "
                      "coordination branch as a distinct policy change",
                      architect)
        self.assertIn("all twelve exact paths", self.implementer)
        self.assertIn("regardless of ticket type", self.implementer)
        self.assertIn("regardless of ticket type", self.redteam)
        self.assertIn(guard, self.redteam)
        self.assertIn("PERMANENT-NOTE-GUARD PASS", self.architect)

    def test_architect_only_note_landing_is_narrow_and_cycle_free(self):
        architect = " ".join(self.architect.split())
        architect_command = " ".join(self.architect_command.split())
        implementer = " ".join(self.implementer.split())
        implementer_command = " ".join(self.implementer_command.split())
        redteam = " ".join(self.redteam.split())
        ai_readme = " ".join(self.ai_readme.split())
        tools_readme = " ".join(self.tools_readme.split())

        for name, source in (
                ("Architect role", architect),
                ("Architect command", architect_command),
                ("AI README", ai_readme),
                ("tools README", tools_readme)):
            with self.subTest(source=name):
                self.assertIn("Only the Architect may edit", source)
                self.assertIn("eleven permanent notes", source)
                self.assertIn("one parent", source)
                self.assertIn("push debt", source.lower())
                self.assertIn("clean", source.lower())
                self.assertIn("idle", source.lower())
                self.assertIn("reset", source.lower())

        self.assertIn("B is the unchanged local `main` commit", architect)
        self.assertIn("P is the clean Architect coordination `HEAD`", architect)
        self.assertIn("no ordinary ticket is active", architect)
        self.assertIn("parent daemon", architect.lower())
        self.assertIn("does not reserve, advance, or complete a ticket cycle",
                      architect)
        self.assertIn("does not queue a Sol review", architect)

        self.assertIn("parent is the exact unchanged local-main base B",
                      architect_command)
        self.assertIn("no ordinary ticket reservation", architect_command)
        self.assertIn("consumes no ticket cycle", architect_command)
        self.assertIn("receives no Sol review", architect_command)

        self.assertIn("B is the exact local `main` version", ai_readme)
        self.assertIn("P is the Architect's clean saved update", ai_readme)
        self.assertIn("parent watcher", ai_readme.lower())
        self.assertIn("does not use or complete a ticket cycle", ai_readme)
        self.assertIn("does not ask Sol for a review", ai_readme)

        self.assertIn("B is the local `main` commit", tools_readme)
        self.assertIn("P is the clean Architect coordination `HEAD`",
                      tools_readme)
        self.assertIn("parent watcher", tools_readme.lower())
        self.assertIn("consumes no cycle slot", tools_readme)
        self.assertIn("queues no Red Team request", tools_readme)
        self.assertIn("not an Implementer", implementer)
        self.assertIn("not an Implementer ticket", implementer_command)
        self.assertIn("do not edit, commit", implementer)
        self.assertIn("not a Red Team review target", redteam)
        self.assertIn("candidate-to-landing recovery", tools_readme)
        self.assertIn("saved Architect GO", tools_readme)
        self.assertIn("leave `HEAD` at B and write no daemon or Implementer "
                      "output", architect)
        self.assertIn("leaves `HEAD` at B and writes no daemon or Implementer "
                      "output", architect_command)
        self.assertIn("leaves the saved version at B and sends neither a "
                      "daemon request nor an Implementer request", ai_readme)
        self.assertIn("leaves `HEAD` at B and creates neither a daemon "
                      "request nor an Implementer request", tools_readme)

        request_template = (
            "MAILBOX-RETURN: architect-notes-go\n"
            "MAILBOX-BASE: FULL-B-FROM-MAILBOX_NOTES_BASE\n"
            "MAILBOX-NOTES-COMMIT: FULL-P\n"
            "MAILBOX-DECISION: GO")
        publisher_command = (
            "python3 \"$MAILBOX_PRIMARY_WORKTREE/ai/tools/"
            "handoff_router.py\" \\\n"
            "  --architect-notes-admin \"PLAIN-LANGUAGE SUMMARY\"")
        for name, source in (
                ("Architect role", self.architect),
                ("Architect command", self.architect_command),
                ("AI README", self.ai_readme),
                ("tools README", self.tools_readme)):
            with self.subTest(request=name):
                self.assertIn("MAILBOX-ADMIN: permanent-notes", source)
                self.assertIn("MAILBOX_NOTES_BASE", source)
                self.assertIn(request_template, source)
                self.assertIn(publisher_command, source)

        base = "a" * 40
        notes = "b" * 40
        self.assertEqual(
            mailbox_daemon.architect_notes_admin_payload("Explain update"),
            "MAILBOX-ADMIN: permanent-notes\n\nExplain update\n")
        self.assertEqual(
            mailbox_daemon.architect_notes_go_request_payload(base, notes),
            "MAILBOX-RETURN: architect-notes-go\n"
            "MAILBOX-BASE: " + base + "\n"
            "MAILBOX-NOTES-COMMIT: " + notes + "\n"
            "MAILBOX-DECISION: GO\n")
        self.assertTrue(callable(mailbox_daemon.preflight_role_baseline_sync))
        self.assertTrue(callable(mailbox_daemon.sync_all_clean_role_baselines))
        self.assertTrue(callable(mailbox_daemon.send_architect_notes_admin))
        self.assertIn("--architect-notes-admin", self.router)
        self.assertIn("exact `MAILBOX_ROLE=architect` binding",
                      self.implementer)
        self.assertIn("must refuse the Implementer", self.implementer_command)
        self.assertIn("exact `MAILBOX_ROLE=architect` binding", self.redteam)

        self.assertIn("each persistent role baseline", architect)
        self.assertIn("every clean idle Architect, Implementer, and Red Team "
                      "baseline to L", architect)
        self.assertIn("all three persistent role baselines", architect)
        self.assertIn("every clean idle role baseline to L", architect_command)
        self.assertIn("updates a role folder only when no AI job is using it "
                      "and the folder has no edits that Git has not saved",
                      ai_readme)
        self.assertIn("authoritative daemon and role files do not stay behind",
                      tools_readme)
        self.assertIn("refusal rather than a reset", tools_readme)
        self.assertIn("never resets or overwrites an unsafe lane", architect)

    def test_subagent_fanout_has_no_small_or_convenience_exception(self):
        architect = " ".join(self.architect.split())
        implementer = " ".join(self.implementer.split())
        implementer_command = " ".join(self.implementer_command.split())

        self.assertIn("Every implementation directive", architect)
        self.assertIn("launch every planned helper before making any "
                      "Integrator-owned implementation edit", architect)
        self.assertIn("Even a small source edit", architect)
        self.assertIn("serial execution merely because it was convenient",
                      architect)
        self.assertIn("Every ticket must attempt the plan", implementer)
        self.assertIn("delegation is mandatory, not a suggestion", implementer)
        self.assertIn("The ticket is small", implementer_command)
        self.assertIn("serial work is convenient", implementer_command)
        self.assertIn("real failed launch", implementer_command)


if __name__ == "__main__":
    unittest.main()
