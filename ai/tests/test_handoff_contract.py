"""Focused tests for decision-complete handoff directive packets."""

from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from ai.tools.handoff_contract import DirectiveError
from ai.tools.handoff_contract import MAX_NOTE_BYTES
from ai.tools.handoff_contract import REQUIRED_SECTIONS
from ai.tools.handoff_contract import \
    extract_blocked_implementer_capability_evidence
from ai.tools.handoff_contract import main
from ai.tools.handoff_contract import resolve_character_limit
from ai.tools.handoff_contract import resolve_discovery_severity
from ai.tools.handoff_contract import validate_directive_file
from ai.tools.handoff_contract import validate_directive_text
from ai.tools.handoff_contract import \
    validate_implementer_handoff_subagent_evidence
from ai.tools.handoff_contract import validate_implementer_subagent_evidence


BASE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
WORKTREE = "/repo/.claude/worktrees/mailbox-primary"
CHECKPOINT_CYCLE = "scratch-ticket@" + BASE_COMMIT
CHECKPOINT_SHA256 = "a" * 64
NO_HELPER_REASON = (
    "The complete edit and its assertion share one small parser branch; a "
    "separate helper would repeat the same inspection without producing "
    "independent evidence.")
NO_HELPER_PLAN = (
    "#### Subagents not required\n- Reason: " + NO_HELPER_REASON)
NO_HELPER_EVIDENCE = NO_HELPER_PLAN


ARCHITECT_BODIES = {
    "Outcome": "Add one bounded read-only directive validation command.",
    "Starting point": (
        "At commit 0123456, ai/tools/example.py accepts unchecked notes."),
    "Execution checkout": (
        "- Worktree: `" + WORKTREE + "`\n"
        "- Branch: `claude/mailbox-primary`\n"
        "- Base: `" + BASE_COMMIT + "`"),
    "Character-change budget": (
        "- Limit: `0`\n"
        "- Planned maximum: `900`\n"
        "- Readability plan: Keep descriptive names and explicit control "
        "flow throughout the complete tested change."),
    "Role plan": (
        "- Roles: `Architect + Implementer + Red Team`\n"
        "- Discovery severity: `medium`\n"
        "- Review scope: `bounded`"),
    "Files and symbols": (
        "- `ai/tools/example.py::validate`: modify the validator.\n"
        "- `ai/tests/test_example.py::ExampleTests`: add the validator "
        "regression cases."),
    "Ordered implementation steps": (
        "1. Add the exact validator before publication.\n"
        "2. Return its diagnostic without changing the source note."),
    "Interfaces and exact behavior": (
        "Keep `validate(path: str) -> None`; accept UTF-8 Markdown only."),
    "Failure behavior and edge cases": (
        "Refuse missing, duplicate, empty, and reordered sections before use."),
    "Tests to write": (
        "- `ai/tests/test_example.py::test_missing_section_refuses`: add "
        "one exact ValueError match."),
    "Validation commands": (
        "Run the focused suite and require exit zero.\n"
        "```bash\npython3 -m unittest ai.tests.test_example\n```"),
    "Acceptance checklist": (
        "- [ ] Valid notes pass and every malformed fixture refuses."),
    "Do not change": (
        "Do not edit mailbox state, scientific code, or existing thresholds."),
    "Stop and ask if": (
        "Stop if the named symbol is absent or another writer owns the file."),
    "Parallel work plan": (
        "#### Subagents required\n"
        "- Launch: `required before implementation edits`\n"
        "#### Subagent `failure-reproducer`\n"
        "- Mode: `read-only`\n"
        "- Ownership: `none (read-only)`\n"
        "- Task: Run `python3 -m unittest ai.tests.test_example` and isolate "
        "its exact pre-edit failure without modifying repository files.\n"
        "- Return: Return the exact command, exit code, and complete failing "
        "assertion output to the Integrator.\n"
        "- Acceptance: The returned output shows one deterministic failure "
        "at the missing-section assertion before editing begins.\n"
        "- Stop: Stop and report if the focused test cannot start with the "
        "standard-library runtime.\n"
        "#### Subagent `regression-writer`\n"
        "- Mode: `edit`\n"
        "- Ownership: `ai/tests/test_example.py::ExampleTests`\n"
        "- Task: Add the named missing-section regression case with the "
        "directive's exact diagnostic and no production edits.\n"
        "- Return: Return the focused test-file diff and the exact failing "
        "test command output before production integration.\n"
        "- Acceptance: The diff contains only the named test class and the "
        "focused command fails at its new assertion.\n"
        "- Stop: Stop and report if the named test class is absent from the "
        "owned file.\n"
        "#### Integrator\n"
        "- Integration: Review every subagent return, integrate the "
        "non-overlapping test edit, and make the production validator change.\n"
        "- Final validation: Run `python3 -m unittest "
        "ai.tests.test_example` and require exit zero after integration."),
}


def bounded_bodies(limit, planned=None, guard_tool="ai/tools/ticket_change_guard.py"):
    """Return the exact packet additions required by a positive limit."""
    if planned is None:
        planned = limit
    return {
        "Character-change budget": (
            "- Limit: `" + str(limit) + "`\n"
            "- Planned maximum: `" + str(planned) + "`\n"
            "- Readability plan: Keep descriptive names, explicit branches, "
            "complete tests, and explanatory prose."),
        "Validation commands": (
            "Run the focused suite, then measure the exact candidate.\n"
            "```bash\n"
            "python3 -m unittest ai.tests.test_example\n"
            "python3 " + guard_tool + " --repo " + WORKTREE
            + " --base " + BASE_COMMIT + " --max " + str(limit) + "\n"
            "```"),
        "Acceptance checklist": (
            "- [ ] Valid notes pass and every malformed fixture refuses.\n"
            "- [ ] `ai/tools/ticket_change_guard.py` reports `within limit` "
            "for the exact clean candidate."),
    }

REDTEAM_BODIES = {
    "Finding and evidence": (
        "Commit 0123456 accepts a missing directive; raw test exits zero.\n"
        "- User severity setting: `medium`\n"
        "- Red Team severity: `medium`\n"
        "- Likelihood: `probable`\n"
        "- Likelihood evidence: A normal missing-section input reaches the "
        "unchecked dispatch path.\n"
        "- Meets user setting: `yes`"),
    "Root cause": (
        "The dispatch path checks transport bytes but not the cited packet."),
    "Required outcome": (
        "Refuse an incomplete packet before an Implementer begins editing."),
    "Character-change budget": (
        "- Limit: `0`\n"
        "- Planned maximum: `600`\n"
        "- Readability plan: Preserve descriptive names and explicit repair "
        "steps for the lower-capability Implementer."),
    "Files and symbols": (
        "- `ai/tools/example.py::validate`: repair the validator.\n"
        "- `ai/tests/test_example.py::ExampleTests`: add the validator "
        "regression cases."),
    "Ordered repair steps": (
        "1. Validate the packet before use.\n"
        "2. Add the missing-section regression witness."),
    "Exact invariants": (
        "The check is read-only, preserves note bytes, and exits nonzero."),
    "Regression test": (
        "- `ai/tests/test_example.py::test_missing_section_refuses`: it "
        "must red without the check."),
    "Validation commands": (
        "Run the focused suite and require exit zero.\n"
        "```bash\npython3 -m unittest ai.tests.test_example\n```"),
    "Acceptance checklist": (
        "- [ ] The malformed note refuses before any implementation begins."),
    "Do not change": (
        "Do not widen review scope or weaken an existing gate surface."),
    "Stop and ask if": (
        "Stop if repair needs an unreviewed file or a new architecture choice."),
    "Architect adjudication required": (
        "This is candidate input only; the Architect must adopt it first."),
}


def packet(role, bodies=None, sections=None):
    """Render one test directive using the production heading contract."""
    title = ("Implementation directive" if role == "architect"
             else "Repair directive")
    values = dict(ARCHITECT_BODIES if role == "architect"
                  else REDTEAM_BODIES)
    if bodies is not None:
        values.update(bodies)
    ordered = REQUIRED_SECTIONS[role] if sections is None else sections
    lines = ["# Scratch ticket", "", "## " + title, ""]
    for heading in ordered:
        lines.extend(("### " + heading, values[heading], ""))
    evidence_heading = ("Implementation evidence / resume state"
                        if role == "architect" else "Red Team evidence")
    lines.extend(("## " + evidence_heading,
                  "No implementation evidence yet.", ""))
    return "\n".join(lines)


class HandoffContractTests(unittest.TestCase):
    """Pin complete packets and reject common weak-handoff failures."""

    def test_valid_architect_and_redteam_packets(self):
        for role in ("architect", "redteam"):
            with self.subTest(role=role):
                result = validate_directive_text(
                    role=role, text=packet(role=role))
                self.assertEqual(
                    result["character_change_budget"]["limit"], 0)

    def test_architect_role_plan_selects_roles_and_severity(self):
        cases = (
            (
                "- Roles: `Architect + Implementer + Red Team`\n"
                "- Discovery severity: `high`\n"
                "- Review scope: `bounded`",
                {
                    "route": "three-role",
                    "uses_red_team": True,
                    "roles": "Architect + Implementer + Red Team",
                    "discovery_severity": "high",
                    "review_scope": "bounded",
                },
            ),
            (
                "- Roles: `Architect + Implementer`\n"
                "- Discovery severity: `not-used`\n"
                "- Review scope: `not-used`",
                {
                    "route": "two-role",
                    "uses_red_team": False,
                    "roles": "Architect + Implementer",
                    "discovery_severity": "not-used",
                    "review_scope": "not-used",
                },
            ),
        )
        for role_plan, expected in cases:
            with self.subTest(role_plan=role_plan):
                result = validate_directive_text(
                    role="architect",
                    text=packet(
                        role="architect", bodies={"Role plan": role_plan}))
                self.assertEqual(result["role_plan"], expected)

    def test_architect_role_plan_refuses_runner_choices_in_the_note(self):
        invalid_plans = (
            (
                "- Roles: `Architect + Implementer + Red Team`\n"
                "- Discovery severity: `not-used`\n"
                "- Review scope: `bounded`",
                "must name high, medium, or low",
            ),
            (
                "- Roles: `Architect + Implementer`\n"
                "- Discovery severity: `medium`\n"
                "- Review scope: `not-used`",
                "must use discovery severity `not-used`",
            ),
            (
                "- Roles: `Architect + Implementer`\n"
                "- Discovery severity: `not-used`\n"
                "- Review scope: `not-used`\n"
                "- Runner override: `Red Team`",
                "requires exactly these rows",
            ),
        )
        for role_plan, message in invalid_plans:
            with self.subTest(role_plan=role_plan):
                with self.assertRaisesRegex(DirectiveError, message):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Role plan": role_plan}))

    def test_architect_parses_a_decision_complete_subagent_plan(self):
        result = validate_directive_text(
            role="architect", text=packet(role="architect"))
        plan = result["parallel_work_plan"]
        self.assertEqual(plan["mode"], "subagents")
        self.assertEqual(
            [row["name"] for row in plan["subagents"]],
            ["failure-reproducer", "regression-writer"])
        self.assertEqual(plan["subagents"][0]["mode"], "read-only")
        self.assertEqual(plan["subagents"][0]["ownership"], [])
        self.assertEqual(plan["subagents"][1]["mode"], "edit")
        self.assertEqual(
            plan["subagents"][1]["ownership"],
            ["ai/tests/test_example.py::ExampleTests"])
        self.assertIn(
            "every subagent return", plan["integrator"]["integration"])

    def test_architect_refuses_vague_or_optional_parallel_work(self):
        invalid_plans = (
            "Work in parallel where useful and report the final results.",
            "This ticket is genuinely indivisible because it changes one "
            "line and needs no delegation.",
            "- Subagent reproducer: Run the focused test and report output.\n"
            "- Integrator: Combine the result and finish the work.",
            "The runtime has no subagent support because no launch tool is "
            "listed in the prompt.",
            ARCHITECT_BODIES["Parallel work plan"].replace(
                "- Launch: `required before implementation edits`\n", ""),
            ARCHITECT_BODIES["Parallel work plan"].replace(
                "#### Integrator", "The Integrator combines all returns"),
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan[:80]):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Parallel work plan": plan}))

    def test_architect_may_explain_why_subagents_add_no_independent_value(self):
        plan = validate_directive_text(
            role="architect",
            text=packet(
                role="architect",
                bodies={"Parallel work plan": NO_HELPER_PLAN}))[
                    "parallel_work_plan"]
        self.assertEqual(plan, {
            "mode": "not-required",
            "reason": NO_HELPER_REASON,
            "subagents": [],
        })

        returned = validate_implementer_subagent_evidence(
            plan, NO_HELPER_EVIDENCE)
        self.assertTrue(returned["completion_ready"])
        self.assertEqual(returned["reason"], NO_HELPER_REASON)

        invalid_plans = (
            "#### Subagents not required",
            "#### Subagents not required\n- Reason: The ticket is small.",
            "#### Subagents not required\n- Reason: "
            "No helper is needed because this is convenient serial work.",
            "#### Subagents not required\n- Reason: The source edit and an "
            "independent test are separate useful tasks that produce distinct "
            "evidence and should be assigned to different helpers.",
            NO_HELPER_PLAN + "\n#### Subagent `unplanned-reviewer`",
        )
        for invalid in invalid_plans:
            with self.subTest(invalid=invalid):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Parallel work plan": invalid}))

        with self.assertRaisesRegex(DirectiveError, "repeat.*Reason exactly"):
            validate_implementer_subagent_evidence(
                plan,
                NO_HELPER_EVIDENCE.replace(
                    "same inspection", "same source inspection"))
        with self.assertRaises(DirectiveError):
            validate_implementer_subagent_evidence(
                plan,
                "#### Subagent return `unplanned-reviewer`\n"
                "- Returned artifact: A separate focused review report.\n"
                "- Acceptance: `pass`\n"
                "- Evidence: The named command exited zero with full output.")

    def test_subagent_blocks_require_exact_fields_and_real_ownership(self):
        valid = ARCHITECT_BODIES["Parallel work plan"]
        invalid_plans = (
            valid.replace("- Mode: `read-only`", "- Mode: `review`", 1),
            valid.replace(
                "- Ownership: `none (read-only)`",
                "- Ownership: none", 1),
            valid.replace(
                "- Mode: `read-only`", "- Mode: `edit`", 1),
            valid.replace(
                "- Ownership: `ai/tests/test_example.py::ExampleTests`",
                "- Ownership: `path/to/file::symbol`", 1),
            valid.replace(
                "- Ownership: `ai/tests/test_example.py::ExampleTests`",
                "- Ownership: `ai/tests/test_example.py::ExampleTests::"
                "test_case`", 1),
            valid.replace("- Task:", "- Action:", 1),
            valid.replace("- Return:", "- Acceptance:", 1),
            valid.replace(
                "Run `python3 -m unittest ai.tests.test_example` and isolate "
                "its exact pre-edit failure without modifying repository "
                "files.",
                "Work on the task as needed and report results.", 1),
            valid.replace(
                "Run `python3 -m unittest ai.tests.test_example` and isolate "
                "its exact pre-edit failure without modifying repository "
                "files.",
                "Investigate the malformed-note behavior comprehensively "
                "and prepare exact evidence for the Integrator.", 1),
            valid.replace(
                "Run `python3 -m unittest ai.tests.test_example` and isolate "
                "its exact pre-edit failure without modifying repository "
                "files.",
                "Run the focused malformed-note test and isolate its exact "
                "pre-edit failure without modifying repository files.", 1),
            valid.replace(
                "Return the exact command, exit code, and complete failing "
                "assertion output to the Integrator.",
                "Return a careful narrative about the completed activity.",
                1),
            valid.replace(
                "The returned output shows one deterministic failure at the "
                "missing-section assertion before editing begins.",
                "The work is careful, comprehensive, and ready for review.",
                1),
            valid.replace(
                "Stop and report if the focused test cannot start with the "
                "standard-library runtime.",
                "Report any general concerns to the Integrator promptly.",
                1),
            valid.replace(
                "#### Subagent `regression-writer`",
                "#### Subagent `failure-reproducer`", 1),
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan[:100]):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Parallel work plan": plan}))

    def test_edit_subagent_may_own_multiple_exact_locator_entries(self):
        plan = ARCHITECT_BODIES["Parallel work plan"].replace(
            "- Ownership: `ai/tests/test_example.py::ExampleTests`",
            "- Ownership: `ai/tests/test_example.py::ExampleTests`, "
            "`ai/tests/test_contract_extra.py::ContractExtraTests`", 1)
        parsed = validate_directive_text(
            role="architect",
            text=packet(
                role="architect", bodies={"Parallel work plan": plan}))[
                    "parallel_work_plan"]
        self.assertEqual(
            parsed["subagents"][1]["ownership"],
            ["ai/tests/test_example.py::ExampleTests",
             "ai/tests/test_contract_extra.py::ContractExtraTests"])

    def test_parallel_plan_refuses_duplicate_edit_ownership(self):
        valid = ARCHITECT_BODIES["Parallel work plan"]
        duplicate = (
            "#### Subagent `duplicate-writer`\n"
            "- Mode: `edit`\n"
            "- Ownership: `ai/tests/test_example.py::ExampleTests`\n"
            "- Task: Add the exact duplicate-section refusal case without "
            "editing any production implementation files.\n"
            "- Return: Return the focused test diff and exact failing "
            "assertion output before production integration.\n"
            "- Acceptance: The returned diff contains the named test and its "
            "output fails at the new assertion.\n"
            "- Stop: Stop and report if the owned test class already has an "
            "equivalent regression case.\n")
        plan = valid.replace("#### Integrator", duplicate + "#### Integrator")
        with self.assertRaisesRegex(DirectiveError, "duplicated"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect", bodies={"Parallel work plan": plan}))

    def test_parallel_plan_refuses_same_file_with_different_symbols(self):
        valid = ARCHITECT_BODIES["Parallel work plan"]
        same_file = (
            "#### Subagent `second-section-writer`\n"
            "- Mode: `edit`\n"
            "- Ownership: `ai/tests/test_example.py::OtherExampleTests`\n"
            "- Task: Add the exact second-section refusal case without "
            "editing any production implementation files.\n"
            "- Return: Return the focused test diff and exact failing "
            "assertion output before production integration.\n"
            "- Acceptance: The returned diff contains the named test and its "
            "output fails at the new assertion.\n"
            "- Stop: Stop and report if the owned test class already has an "
            "equivalent regression case.\n")
        plan = valid.replace("#### Integrator",
                             same_file + "#### Integrator")
        with self.assertRaisesRegex(
                DirectiveError,
                r"Ownership file `ai/tests/test_example\.py` is duplicated"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect", bodies={"Parallel work plan": plan}))

    def test_parallel_plan_refuses_noncanonical_ownership_alias(self):
        """A leading ``./`` cannot hide a second writer for one file."""
        valid = ARCHITECT_BODIES["Parallel work plan"]
        aliased = (
            "#### Subagent `alias-writer`\n"
            "- Mode: `edit`\n"
            "- Ownership: `./ai/tests/test_example.py::OtherTests`\n"
            "- Task: Add the exact alias refusal test without editing any "
            "production file.\n"
            "- Return: Return the focused diff and failing assertion output "
            "to the Integrator.\n"
            "- Acceptance: The diff contains only the named test and the "
            "command output fails at its assertion.\n"
            "- Stop: Stop and report if the named test file is absent.\n")
        plan = valid.replace("#### Integrator", aliased + "#### Integrator")
        with self.assertRaisesRegex(DirectiveError, "malformed Ownership"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect", bodies={"Parallel work plan": plan}))

    def test_integrator_requires_exact_integration_and_validation_fields(self):
        valid = ARCHITECT_BODIES["Parallel work plan"]
        invalid_plans = (
            valid.replace("- Integration:", "- Integrator:", 1),
            valid.replace("- Final validation:", "- Validation:", 1),
            valid.replace(
                "Review every subagent return, integrate the non-overlapping "
                "test edit, and make the production validator change.",
                "Apply the final production change with careful readable "
                "control flow and complete regression coverage.", 1),
            valid.replace(
                "Run `python3 -m unittest ai.tests.test_example` and require "
                "exit zero after integration.",
                "Run the complete focused validation after integrating all "
                "accepted subagent changes.", 1),
            valid + "\nAdditional work may be performed after validation.",
        )
        for plan in invalid_plans:
            with self.subTest(plan=plan[-120:]):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Parallel work plan": plan}))

    def test_integrator_validation_must_repeat_directive_command(self):
        plan = ARCHITECT_BODIES["Parallel work plan"].replace(
            "- Final validation: Run `python3 -m unittest "
            "ai.tests.test_example`",
            "- Final validation: Run `true`", 1)
        with self.assertRaisesRegex(
                DirectiveError, "Validation commands section"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect", bodies={"Parallel work plan": plan}))

    def test_capability_exception_requires_three_exact_raw_fields(self):
        unsupported = (
            "- Capability checked: `collaboration.spawn_agent`\n"
            "- Attempted operation: Launch the named reproducer subagent "
            "through the advertised collaboration operation before "
            "implementation edits.\n"
            "- Raw failure: `Unknown tool collaboration.spawn_agent in the "
            "advertised runtime capability registry`")
        checkpoint = (
            "### Prior Implementer subagent launch failure\n\n"
            "- Source cycle: `" + CHECKPOINT_CYCLE + "`\n"
            "- Source handoff SHA-256: `" + CHECKPOINT_SHA256 + "`\n"
            "- Source: `prior same-cycle IMPLEMENTER_HANDOFF checkpoint`\n"
            "- Capability checked: `collaboration.spawn_agent`\n"
            "- Attempted operation: Launch the named reproducer subagent "
            "through the advertised collaboration operation before "
            "implementation edits.\n"
            "- Raw failure: `Unknown tool collaboration.spawn_agent in the "
            "advertised runtime capability registry`")
        revised = packet(
            role="architect",
            bodies={"Parallel work plan": unsupported}).replace(
                "No implementation evidence yet.", checkpoint)
        directive = validate_directive_text(
            role="architect", text=revised)
        parsed = directive["parallel_work_plan"]
        self.assertEqual(parsed["mode"], "capability-unavailable")
        self.assertEqual(parsed["subagents"], [])
        self.assertEqual(
            directive["capability_checkpoint"],
            {"cycle": CHECKPOINT_CYCLE,
             "handoff_sha256": CHECKPOINT_SHA256})

        with self.assertRaisesRegex(
                DirectiveError, "Prior Implementer subagent launch failure"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect",
                    bodies={"Parallel work plan": unsupported}))

        invalid = (
            unsupported.replace(
                "- Capability checked: `collaboration.spawn_agent`\n", ""),
            unsupported.replace(
                "Launch the named reproducer subagent through the advertised "
                "collaboration operation before implementation edits.",
                "Tried it."),
            unsupported.replace(
                "before implementation edits",
                "after all implementation edits"),
            unsupported.replace(
                "`Unknown tool collaboration.spawn_agent in the advertised "
                "runtime capability registry`", "`failed`"),
            unsupported.replace("collaboration.spawn_agent", "[tool]", 1),
            unsupported + "\n- Reason: The ticket is genuinely indivisible.",
        )
        for plan in invalid:
            with self.subTest(plan=plan):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Parallel work plan": plan}).replace(
                                "No implementation evidence yet.",
                                checkpoint))

        mismatched_checkpoint = revised.replace(
            "Unknown tool collaboration.spawn_agent in the advertised "
            "runtime capability registry`",
            "Runtime returned a different launch failure after retry`",
            1)
        with self.assertRaisesRegex(DirectiveError, "must repeat the exact"):
            validate_directive_text(
                role="architect", text=mismatched_checkpoint)

        for stale_binding in (
                revised.replace(CHECKPOINT_CYCLE, "not-a-full-cycle", 1),
                revised.replace(CHECKPOINT_SHA256, "abc123", 1)):
            with self.subTest(stale_binding=stale_binding[-300:]):
                with self.assertRaisesRegex(
                        DirectiveError, "Source cycle|Source handoff"):
                    validate_directive_text(
                        role="architect", text=stale_binding)

    def test_implementer_subagent_evidence_matches_every_planned_name(self):
        plan = validate_directive_text(
            role="architect", text=packet(role="architect"))[
                "parallel_work_plan"]
        evidence = (
            "#### Subagent return `failure-reproducer`\n"
            "- Returned artifact: The exact focused command and its complete "
            "pre-edit failing assertion output.\n"
            "- Acceptance: `pass`\n"
            "- Evidence: Command `python3 -m unittest "
            "ai.tests.test_example` exited one at the named assertion.\n"
            "#### Subagent return `regression-writer`\n"
            "- Returned artifact: The focused test-file diff and complete "
            "pre-production failing command output.\n"
            "- Acceptance: `pass`\n"
            "- Evidence: The diff changes only ExampleTests and the focused "
            "command output names the new assertion.")
        parsed = validate_implementer_subagent_evidence(plan, evidence)
        self.assertEqual(
            [row["name"] for row in parsed["returns"]],
            ["failure-reproducer", "regression-writer"])
        self.assertTrue(parsed["completion_ready"])

        blocked = validate_implementer_subagent_evidence(
            plan, evidence.replace(
                "- Acceptance: `pass`", "- Acceptance: `blocked`", 1))
        self.assertFalse(blocked["completion_ready"])

        invalid = (
            evidence.replace(
                "#### Subagent return `regression-writer`\n"
                "- Returned artifact: The focused test-file diff and complete "
                "pre-production failing command output.\n"
                "- Acceptance: `pass`\n"
                "- Evidence: The diff changes only ExampleTests and the "
                "focused command output names the new assertion.", ""),
            evidence.replace("failure-reproducer", "unplanned-reviewer", 1),
            evidence.replace("- Acceptance: `pass`", "- Acceptance: `yes`", 1),
            evidence.replace("- Returned artifact:", "- Artifact:", 1),
            evidence + "\n#### Subagent return `extra-reviewer`\n"
            "- Returned artifact: The extra reviewer returned a detailed "
            "command transcript and focused output.\n"
            "- Acceptance: `pass`\n"
            "- Evidence: The extra command exited zero with complete output.",
        )
        for malformed in invalid:
            with self.subTest(malformed=malformed[-100:]):
                with self.assertRaises(DirectiveError):
                    validate_implementer_subagent_evidence(plan, malformed)

    def test_blocked_full_handoff_requires_exact_capability_failure_rows(self):
        plan = validate_directive_text(
            role="architect", text=packet(role="architect"))[
                "parallel_work_plan"]
        returns = (
            "#### Subagent return `failure-reproducer`\n"
            "- Returned artifact: The exact launch attempt and complete "
            "runtime failure transcript.\n"
            "- Acceptance: `blocked`\n"
            "- Evidence: The launch operation failed before any source edit "
            "began.\n"
            "#### Subagent return `regression-writer`\n"
            "- Returned artifact: The no-edit checkpoint and repository "
            "status transcript.\n"
            "- Acceptance: `blocked`\n"
            "- Evidence: The second task did not start after the launch "
            "capability failed.")
        failure = (
            "- Capability checked: `collaboration.spawn_agent`\n"
            "- Attempted operation: Launch the named reproducer subagent "
            "through collaboration.spawn_agent before implementation edits.\n"
            "- Raw failure: `Unknown tool collaboration.spawn_agent in the "
            "advertised runtime capability registry`")

        def handoff(fragment):
            return (
                "### IMPLEMENTER_HANDOFF: BLOCKED\n\n"
                "- **Current state:** The required subagent launch failed.\n"
                "- **Candidate commit:** `" + BASE_COMMIT + "`\n"
                "- **Subagent work:**\n" + fragment + "\n"
                "- **Blockers/findings:** The runtime rejected the exact "
                "launch operation.\n"
                "- **Action required:** Architect capability decision.\n")

        complete = handoff(returns + "\n" + failure)
        parsed = extract_blocked_implementer_capability_evidence(complete)
        self.assertEqual(
            [record["name"] for record in parsed["returns"]],
            ["failure-reproducer", "regression-writer"])
        self.assertEqual(
            parsed["capability_checked"], "collaboration.spawn_agent")
        validated = validate_implementer_handoff_subagent_evidence(
            parallel_work_plan=plan, handoff_text=complete)
        self.assertFalse(validated["completion_ready"])
        self.assertEqual(
            validated["capability_failure"]["raw_failure"],
            "Unknown tool collaboration.spawn_agent in the advertised "
            "runtime capability registry")

        rows = failure.split("\n")
        for missing_index in range(3):
            malformed = handoff(
                returns + "\n" + "\n".join(
                    row for index, row in enumerate(rows)
                    if index != missing_index))
            with self.subTest(missing_row=rows[missing_index]):
                with self.assertRaisesRegex(
                        DirectiveError, "blocked IMPLEMENTER_HANDOFF"):
                    validate_implementer_handoff_subagent_evidence(
                        parallel_work_plan=plan, handoff_text=malformed)

    def test_capability_exception_evidence_must_repeat_the_raw_failure(self):
        unsupported = (
            "- Capability checked: `collaboration.spawn_agent`\n"
            "- Attempted operation: Launch the named reproducer subagent "
            "through the advertised collaboration operation before "
            "implementation edits.\n"
            "- Raw failure: `Unknown tool collaboration.spawn_agent in the "
            "advertised runtime capability registry`")
        checkpoint = (
            "### Prior Implementer subagent launch failure\n\n"
            "- Source cycle: `" + CHECKPOINT_CYCLE + "`\n"
            "- Source handoff SHA-256: `" + CHECKPOINT_SHA256 + "`\n"
            "- Source: `prior same-cycle IMPLEMENTER_HANDOFF checkpoint`\n"
            + unsupported)
        plan = validate_directive_text(
            role="architect",
            text=packet(
                role="architect",
                bodies={"Parallel work plan": unsupported}).replace(
                    "No implementation evidence yet.", checkpoint))[
                    "parallel_work_plan"]
        parsed = validate_implementer_subagent_evidence(plan, unsupported)
        self.assertEqual(parsed["mode"], "capability-unavailable")
        self.assertTrue(parsed["completion_ready"])
        changed = unsupported.replace("Unknown tool", "Operation rejected")
        with self.assertRaisesRegex(DirectiveError, "does not match"):
            validate_implementer_subagent_evidence(plan, changed)

    def test_architect_role_plan_binds_review_scope_without_prose_inference(self):
        widespread = (
            "- Roles: `Architect + Implementer + Red Team`\n"
            "- Discovery severity: `low`\n"
            "- Review scope: `widespread`")
        parsed = validate_directive_text(
            role="architect",
            text=packet(role="architect", bodies={"Role plan": widespread}))
        self.assertEqual(parsed["role_plan"]["review_scope"], "widespread")
        self.assertEqual(parsed["role_plan"]["discovery_severity"], "low")

        prose_only = packet(
            role="architect",
            bodies={
                "Outcome": (
                    "Do a widespread search is quoted here only as text; "
                    "the structured field still binds a bounded review."),
            })
        parsed = validate_directive_text(role="architect", text=prose_only)
        self.assertEqual(parsed["role_plan"]["review_scope"], "bounded")

    def test_architect_role_plan_refuses_malformed_or_inconsistent_scope(self):
        invalid_plans = (
            (
                "- Roles: `Architect + Implementer + Red Team`\n"
                "- Discovery severity: `medium`",
                "exactly these rows",
            ),
            (
                "- Roles: `Architect + Implementer + Red Team`\n"
                "- Review scope: `bounded`\n"
                "- Discovery severity: `medium`",
                "followed by one Discovery severity value",
            ),
            (
                "- Roles: `Architect + Implementer + Red Team`\n"
                "- Discovery severity: `medium`\n"
                "- Review scope: `wide`",
                "one Review scope value",
            ),
            (
                "- Roles: `Architect + Implementer + Red Team`\n"
                "- Discovery severity: `medium`\n"
                "- Review scope: `not-used`",
                "must use review scope `bounded` or `widespread`",
            ),
            (
                "- Roles: `Architect + Implementer`\n"
                "- Discovery severity: `not-used`\n"
                "- Review scope: `bounded`",
                "must use review scope `not-used`",
            ),
            (
                "- Roles: `Architect + Sol as Implementer`\n"
                "- Discovery severity: `not-used`\n"
                "- Review scope: `widespread`",
                "one supported Roles value",
            ),
            (
                "- Roles: `Architect + Implementer + Red Team`\n"
                "- Discovery severity: `medium`\n"
                "- Review scope: `widespread`",
                "widespread review scope requires discovery severity `low`",
            ),
            (
                "- Roles: `Architect + Implementer + Red Team`\n"
                "- Discovery severity: `low`\n"
                "- Review scope: `bounded`\n"
                "- Review scope: `widespread`",
                "exactly these rows",
            ),
        )
        for role_plan, message in invalid_plans:
            with self.subTest(role_plan=role_plan):
                with self.assertRaisesRegex(DirectiveError, message):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Role plan": role_plan}))

    def test_redteam_severity_assessment_is_ordered_and_consistent(self):
        result = validate_directive_text(
            role="redteam", text=packet(role="redteam"))
        self.assertEqual(
            result["discovery_severity_assessment"],
            {
                "user_setting": "medium",
                "redteam_severity": "medium",
                "likelihood": "probable",
                "likelihood_evidence": (
                    "A normal missing-section input reaches the unchecked "
                    "dispatch path."),
                "meets_user_setting": "yes",
            })

        base = REDTEAM_BODIES["Finding and evidence"]
        invalid = {
            "missing": base.replace(
                "- Red Team severity: `medium`\n", ""),
            "duplicate": base + "\n- Likelihood: `probable`",
            "reordered": base.replace(
                "- Red Team severity: `medium`\n"
                "- Likelihood: `probable`\n",
                "- Likelihood: `probable`\n"
                "- Red Team severity: `medium`\n"),
            "wrong value": base.replace(
                "- Red Team severity: `medium`",
                "- Red Team severity: `critical`"),
            "weak evidence": base.replace(
                "A normal missing-section input reaches the unchecked "
                "dispatch path.", "Rare."),
            "contradiction": base.replace(
                "- User severity setting: `medium`",
                "- User severity setting: `high`"),
            "malformed duplicate": (
                base + "\n- User severity setting: high"),
            "case-variant duplicate": (
                base + "\n- user severity setting: `medium`"),
        }
        for name, finding in invalid.items():
            with self.subTest(name=name):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="redteam",
                        text=packet(
                            role="redteam",
                            bodies={"Finding and evidence": finding}))

        matrix = (
            ("high", "high", "probable", "yes"),
            ("high", "medium", "probable", "no"),
            ("medium", "high", "improbable", "yes"),
            ("medium", "medium", "probable", "yes"),
            ("medium", "medium", "improbable", "no"),
            ("medium", "low", "probable", "no"),
            ("low", "low", "improbable", "yes"),
        )
        for user_setting, redteam_severity, likelihood, meets in matrix:
            with self.subTest(
                    user=user_setting, redteam=redteam_severity,
                    likelihood=likelihood):
                finding = base.replace(
                    "- User severity setting: `medium`",
                    "- User severity setting: `" + user_setting + "`")
                finding = finding.replace(
                    "- Red Team severity: `medium`",
                    "- Red Team severity: `" + redteam_severity + "`")
                finding = finding.replace(
                    "- Likelihood: `probable`",
                    "- Likelihood: `" + likelihood + "`")
                finding = finding.replace(
                    "- Meets user setting: `yes`",
                    "- Meets user setting: `" + meets + "`")
                parsed = validate_directive_text(
                    role="redteam",
                    text=packet(
                        role="redteam",
                        bodies={"Finding and evidence": finding}),
                    expected_severity=user_setting)
                self.assertEqual(
                    parsed["discovery_severity_assessment"]
                    ["meets_user_setting"], meets)

    def test_character_change_budget_is_exact_and_policy_matched(self):
        bounded = bounded_bodies(limit=1200, planned=1100)
        for role in ("architect", "redteam"):
            with self.subTest(role=role):
                result = validate_directive_text(
                    role=role,
                    text=packet(role=role, bodies=bounded),
                    expected_max=1200)
                self.assertEqual(
                    result["character_change_budget"],
                    {
                        "limit": 1200,
                        "planned_maximum": 1100,
                        "readability_plan": (
                            "Keep descriptive names, explicit branches, "
                            "complete tests, and explanatory prose."),
                    })

        mismatch = packet(
            role="architect",
            bodies=bounded)
        with self.assertRaisesRegex(DirectiveError, "does not match.*--max"):
            validate_directive_text(
                role="architect", text=mismatch, expected_max=1199)

        unlimited = packet(
            role="architect",
            bodies={"Character-change budget": (
                "- Limit: `0`\n"
                "- Planned maximum: `999999`\n"
                "- Readability plan: Keep the complete implementation "
                "didactic even though no size cap applies.")})
        result = validate_directive_text(
            role="architect", text=unlimited, expected_max=0)
        self.assertEqual(
            result["character_change_budget"]["planned_maximum"], 999999)

    def test_character_change_budget_malformed_or_over_limit_refuses(self):
        malformed = (
            ("- Planned maximum: `4`\n"
             "- Readability plan: Preserve descriptive names and complete "
             "tests for every changed behavior."),
            ("- Limit: `4`\n"
             "- Limit: `4`\n"
             "- Planned maximum: `4`\n"
             "- Readability plan: Preserve descriptive names and complete "
             "tests for every changed behavior."),
            ("- Limit: `-1`\n"
             "- Planned maximum: `0`\n"
             "- Readability plan: Preserve descriptive names and complete "
             "tests for every changed behavior."),
            ("- Limit: `4`\n"
             "- Planned maximum: `4`\n"
             "- Readability plan: Preserve descriptive names and complete "
             "tests for every changed behavior.\n"
             "- Extra: `4`"),
            ("- Planned maximum: `4`\n"
             "- Limit: `4`\n"
             "- Readability plan: Preserve descriptive names and complete "
             "tests for every changed behavior."),
        )
        for body in malformed:
            with self.subTest(body=body):
                with self.assertRaisesRegex(DirectiveError, "exactly these"):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Character-change budget": body}),
                        expected_max=4)

        over = packet(
            role="architect",
            bodies={"Character-change budget": (
                "- Limit: `4`\n"
                "- Planned maximum: `5`\n"
                "- Readability plan: Preserve descriptive names and complete "
                "tests for every changed behavior.")})
        with self.assertRaisesRegex(DirectiveError, "exceeds limit"):
            validate_directive_text(
                role="architect", text=over, expected_max=4)

        weak_plan = packet(
            role="architect",
            bodies={"Character-change budget": (
                "- Limit: `4`\n"
                "- Planned maximum: `4`\n"
                "- Readability plan: Keep clear.")})
        with self.assertRaisesRegex(DirectiveError, "substantive visible"):
            validate_directive_text(
                role="architect", text=weak_plan, expected_max=4)

        leading_zero = packet(
            role="architect",
            bodies={"Character-change budget": (
                "- Limit: `04`\n"
                "- Planned maximum: `04`\n"
                "- Readability plan: Preserve descriptive names and complete "
                "tests for every changed behavior.")})
        with self.assertRaisesRegex(DirectiveError, "without leading zeros"):
            validate_directive_text(
                role="architect", text=leading_zero, expected_max=4)

        for invalid in (-1, True, "4"):
            with self.subTest(expected_max=invalid):
                with self.assertRaisesRegex(
                        DirectiveError, "nonnegative integer"):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect"),
                        expected_max=invalid)

    def test_positive_limit_requires_exact_guard_command_and_condition(self):
        bodies = bounded_bodies(limit=37, planned=30)
        for role in ("architect", "redteam"):
            with self.subTest(role=role):
                result = validate_directive_text(
                    role=role,
                    text=packet(role=role, bodies=bodies),
                    expected_max=37)
                self.assertEqual(
                    result["ticket_change_guard"],
                    {
                        "tool": "ai/tools/ticket_change_guard.py",
                        "repo": WORKTREE,
                        "base": BASE_COMMIT,
                        "max": 37,
                    })

        missing_command = dict(bodies)
        missing_command["Validation commands"] = (
            "Run the focused suite and require exit zero.\n"
            "```bash\npython3 -m unittest ai.tests.test_example\n```")
        with self.assertRaisesRegex(DirectiveError, "direct literal command"):
            validate_directive_text(
                role="architect",
                text=packet(role="architect", bodies=missing_command),
                expected_max=37)

        missing_condition = dict(bodies)
        missing_condition["Acceptance checklist"] = (
            "- [ ] Every focused regression test passes.")
        with self.assertRaisesRegex(DirectiveError, "within limit"):
            validate_directive_text(
                role="architect",
                text=packet(role="architect", bodies=missing_condition),
                expected_max=37)

        negated_condition = dict(bodies)
        negated_condition["Acceptance checklist"] = (
            "- [ ] `ai/tools/ticket_change_guard.py` does not report "
            "`within limit` for the exact candidate.")
        with self.assertRaisesRegex(DirectiveError, "within limit"):
            validate_directive_text(
                role="architect",
                text=packet(role="architect", bodies=negated_condition),
                expected_max=37)

    def test_guard_command_cannot_be_prose_variable_echo_or_wrong_binding(self):
        valid = bounded_bodies(limit=37, planned=30)
        valid_command = (
            "python3 ai/tools/ticket_change_guard.py --repo " + WORKTREE
            + " --base " + BASE_COMMIT + " --max 37")
        replacements = (
            (valid_command,
             "# " + valid_command,
             "direct literal command"),
            (valid_command,
             "echo '" + valid_command + "'",
             "direct literal command"),
            (valid_command,
             "if false; then\n" + valid_command + "\nfi",
             "direct literal command"),
            ("--base " + BASE_COMMIT, "--base $BASE", "direct literal"),
            ("--max 37", "--max $MAX", "direct literal"),
            ("--max 37", "--max 38", "does not match.*--max"),
            ("--max 37", "--max 037", "does not match.*--max"),
            ("--repo " + WORKTREE,
             "--repo /repo/.claude/worktrees/other",
             "does not match.*Worktree"),
            ("--base " + BASE_COMMIT,
             "--base 1123456789abcdef0123456789abcdef01234567",
             "does not match.*Base"),
            ("--repo " + WORKTREE,
             "--repo /repo/$USER/worktree",
             "direct literal command"),
            ("python3 ai/tools/ticket_change_guard.py",
             "ai/tools/ticket_change_guard.py",
             "direct literal command"),
        )
        for old, new, message in replacements:
            with self.subTest(replacement=new):
                malformed = dict(valid)
                malformed["Validation commands"] = malformed[
                    "Validation commands"].replace(old, new)
                with self.assertRaisesRegex(DirectiveError, message):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect", bodies=malformed),
                        expected_max=37)

        duplicate = dict(valid)
        duplicate["Validation commands"] = valid[
            "Validation commands"].replace(valid_command,
                                            valid_command + "\n" + valid_command)
        with self.assertRaisesRegex(DirectiveError, "one direct literal"):
            validate_directive_text(
                role="architect",
                text=packet(role="architect", bodies=duplicate),
                expected_max=37)

    def test_positive_guard_accepts_clear_backslash_wrapping(self):
        bodies = bounded_bodies(limit=37, planned=30)
        one_line = (
            "python3 ai/tools/ticket_change_guard.py --repo " + WORKTREE
            + " --base " + BASE_COMMIT + " --max 37")
        wrapped = (
            "python3 ai/tools/ticket_change_guard.py \\\n"
            "  --repo " + WORKTREE + " \\\n"
            "  --base " + BASE_COMMIT + " \\\n"
            "  --max 37")
        bodies["Validation commands"] = bodies[
            "Validation commands"].replace(one_line, wrapped)
        result = validate_directive_text(
            role="architect",
            text=packet(role="architect", bodies=bodies),
            expected_max=37)
        self.assertEqual(result["ticket_change_guard"]["max"], 37)

    def test_mailbox_guard_path_rejects_relative_tool_and_accepts_authority(self):
        authoritative = (
            "/primary/ai/tools/ticket_change_guard.py")
        relative = bounded_bodies(limit=37, planned=30)
        with mock.patch.dict(
                os.environ,
                {"MAILBOX_TICKET_CHANGE_GUARD": authoritative},
                clear=False):
            with self.assertRaisesRegex(
                    DirectiveError, "direct literal command"):
                validate_directive_text(
                    role="redteam",
                    text=packet(role="redteam", bodies=relative),
                    expected_max=37)

            absolute = bounded_bodies(
                limit=37, planned=30, guard_tool=authoritative)
            result = validate_directive_text(
                role="redteam",
                text=packet(role="redteam", bodies=absolute),
                expected_max=37)
            self.assertEqual(
                result["ticket_change_guard"]["tool"], authoritative)

        with mock.patch.dict(
                os.environ,
                {"MAILBOX_TICKET_CHANGE_GUARD": "relative/tool.py"},
                clear=False):
            with self.assertRaisesRegex(
                    DirectiveError, "authoritative absolute"):
                validate_directive_text(
                    role="redteam",
                    text=packet(role="redteam", bodies=relative),
                    expected_max=37)

    def test_missing_reordered_duplicate_and_unknown_sections_refuse(self):
        required = REQUIRED_SECTIONS["architect"]
        malformed = (
            required[:-1],
            (required[1], required[0]) + required[2:],
            required + (required[-1],),
        )
        for sections in malformed:
            with self.subTest(sections=sections):
                with self.assertRaisesRegex(DirectiveError, "exact order"):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect", sections=sections))

        text = packet(role="architect").replace(
            "### Starting point", "### Unruled design choice", 1)
        with self.assertRaisesRegex(DirectiveError, "exact order"):
            validate_directive_text(role="architect", text=text)

    def test_empty_placeholder_and_delegated_choices_refuse(self):
        for body, message in (("", "too short"),
                              ("[write details here]", "placeholder"),
                              ("Use your best judgment for the algorithm.",
                               "unresolved design choice")):
            with self.subTest(body=body):
                text = packet(
                    role="architect",
                    bodies={"Interfaces and exact behavior": body})
                with self.assertRaisesRegex(DirectiveError, message):
                    validate_directive_text(role="architect", text=text)

        embedded = (
            "Keep the API stable; choose either JSON or YAML.",
            "TODO decide schema later.",
            "The Implementer must choose the storage layout.",
            "Use JSON or YAML for the serialized state.",
            "The Implementer selects the storage layout.",
            "Select a suitable algorithm for the new parser.",
            "Either JSON or YAML is acceptable for the serialized state.",
            "Choose JSON or YAML during implementation.",
            "Pick JSON or YAML after inspecting the file.",
            "Select whichever format seems best during implementation.",
            "Either JSON or YAML will satisfy the interface.",
            "The format may be JSON or YAML for this state.",
            "JSON versus YAML remains an open decision.",
            "The Implementer determines the storage layout.",
            "Defer the serialization decision to implementation time.",
            "One of JSON and YAML should be used for state.",
            "The storage format is left open for implementation.",
        )
        for body in embedded:
            with self.subTest(embedded=body):
                text = packet(
                    role="architect",
                    bodies={"Interfaces and exact behavior": body})
                with self.assertRaises(DirectiveError):
                    validate_directive_text(role="architect", text=text)

        inline_code_choice = packet(
            role="architect",
            bodies={"Interfaces and exact behavior":
                    "Use `JSON or YAML` for serialized state."})
        with self.assertRaisesRegex(DirectiveError, "unresolved design"):
            validate_directive_text(
                role="architect", text=inline_code_choice)

        resolved_alternatives = (
            "Support CPU or CUDA, selected by the existing device "
            "configuration.",
            "Accept JSON or YAML and normalize both formats.",
            "Select float32 or float64 according to the explicit dtype "
            "argument.",
            "Use JSON, alternatively YAML, selected by the explicit file "
            "extension.",
        )
        for body in resolved_alternatives:
            with self.subTest(resolved=body):
                validate_directive_text(
                    role="architect",
                    text=packet(
                        role="architect",
                        bodies={"Interfaces and exact behavior": body}))

        deterministic_or_rules = (
            "Return zero on success or one on validation failure.",
            "Raise FileNotFoundError when the note is missing or unreadable.",
            "Treat CRLF or CR line endings as LF.",
            "The command exits zero when valid or two when invalid.",
            "Return the cached model on a cache hit or train it on a cache "
            "miss.",
            "Use float32 or float64 based on the explicit model dtype.",
            "Run the agent locally or remotely based on the explicit "
            "deployment configuration.",
            "Return zero for success or one for failure.",
            "Map true to one or false to zero.",
        )
        for body in deterministic_or_rules:
            with self.subTest(deterministic_or_rule=body):
                validate_directive_text(
                    role="architect",
                    text=packet(
                        role="architect",
                        bodies={"Interfaces and exact behavior": body}))

        fake_resolvers = (
            "Use JSON or YAML based on the Implementer's judgment.",
            "Use JSON or YAML according to whichever seems best.",
            "Select float32 or float64 depending on what the Implementer "
            "prefers.",
            "Use JSON or YAML based on an open decision.",
            "Use JSON or YAML based on the developer's preference.",
            "Use JSON or YAML based on coder preference.",
            "Use JSON or YAML depending on convenience.",
            "Use JSON or YAML selected by whoever implements it.",
            "Use JSON or YAML determined by the person writing the code.",
            "Use JSON or YAML based on the Implementer.",
            "Use JSON or YAML selected by the developer.",
            "Use JSON or YAML based on discretion.",
            "Use JSON or YAML according to what seems easiest.",
            "Use JSON or YAML based on the Architect's opinion.",
            "Use JSON or YAML selected by the existing developer.",
            "Use JSON or YAML based on the named coder.",
            "Use JSON or YAML determined by the configured Implementer.",
            "The format is JSON or YAML.",
            "Output JSON or YAML.",
            "Store data as JSON or YAML.",
        )
        for body in fake_resolvers:
            with self.subTest(fake_resolver=body):
                with self.assertRaisesRegex(
                        DirectiveError, "unresolved design"):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Interfaces and exact behavior": body}))

        unrelated_negative_clauses = (
            "Do not change the API; use JSON or YAML.",
            "Refuse missing input; store state as JSON or YAML.",
            "Never weaken validation; the format is JSON or YAML.",
            "Stop if the file is absent; output JSON or YAML.",
            "Reject invalid packets, then choose JSON or YAML.",
            "Do not change the API, and use JSON or YAML.",
            "Reject invalid packets but store valid state as JSON or YAML.",
            "Never weaken validation and output JSON or YAML.",
            "Refuse missing input while using JSON or YAML.",
            "Use JSON or YAML while returning one if missing or two if "
            "malformed.",
            "Refuse missing input because the serializer must use JSON or "
            "YAML.",
            "Do not change the API before choosing JSON or YAML.",
            "Reject the packet after selecting JSON or YAML.",
            "Do not change the API although the state format is JSON or "
            "YAML.",
            "Use JSON or YAML while supporting CSV or TSV based on the "
            "explicit format field.",
            "Use JSON or YAML before returning one if missing or two if "
            "malformed.",
            "Use JSON or YAML because it returns one if missing or two if "
            "malformed.",
            "Use JSON/YAML for serialized state.",
            "Choose JSON/YAML during implementation.",
            "Use JSON vs. YAML for serialized state.",
            "Use JSON, alternatively YAML, for serialized state.",
            "Choose from JSON and YAML during implementation.",
            "Select from JSON and YAML during implementation.",
            "Use JSON/YAML, but do not weaken validation.",
            "Use JSON, alternatively YAML, but do not change the API.",
            "Choose from JSON and YAML and never weaken validation.",
        )
        for body in unrelated_negative_clauses:
            with self.subTest(unrelated_negative_clause=body):
                with self.assertRaisesRegex(
                        DirectiveError, "unresolved design"):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Interfaces and exact behavior": body}))

        deterministic_negative_rules = (
            "Do not delete the file unless it is empty or stale.",
            "Do not return until the job succeeds or times out.",
            "Do not accept values before one or after ten.",
            "Never write state while the lock is absent or stale.",
            "Reject the packet if it is missing or malformed.",
        )
        for body in deterministic_negative_rules:
            with self.subTest(deterministic_negative_rule=body):
                validate_directive_text(
                    role="architect",
                    text=packet(
                        role="architect",
                        bodies={"Interfaces and exact behavior": body}))

        undecided_step = packet(
            role="architect",
            bodies={"Ordered implementation steps":
                    "1. Decide which algorithm is best, then implement it."})
        with self.assertRaisesRegex(DirectiveError, "unresolved design"):
            validate_directive_text(role="architect", text=undecided_step)

        undecided_format = packet(
            role="architect",
            bodies={"Ordered implementation steps":
                    "1. Decide on the serialization format, then implement it."})
        with self.assertRaisesRegex(DirectiveError, "unresolved design"):
            validate_directive_text(role="architect", text=undecided_format)

    def test_concrete_file_test_and_checkout_locators_are_required(self):
        cases = (
            ("Files and symbols", "Modify the relevant source files."),
            ("Tests to write", "Add a suitable regression test."),
            ("Execution checkout", "Use the usual development checkout."),
        )
        for heading, body in cases:
            with self.subTest(heading=heading):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect", bodies={heading: body}))

        untouched_checkout = (
            "- Worktree: `<exact linked-worktree path>`\n"
            "- Branch: `<exact non-main branch>`\n"
            "- Base: `<full base commit>`")
        with self.assertRaises(DirectiveError):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect",
                    bodies={"Execution checkout": untouched_checkout}))

        invalid_checkouts = (
            ("- Worktree: `not-a-path`\n"
             "- Branch: `claude/work`\n"
             "- Base: `0123456789abcdef0123456789abcdef01234567`"),
            ("- Worktree: `/repo/.claude/worktrees/worker`\n"
             "- Branch: `main`\n"
             "- Base: `0123456789abcdef0123456789abcdef01234567`"),
            ("- Worktree: `/repo/.claude/worktrees/worker`\n"
             "- Branch: `claude/worker`\n"
             "- Base: `short`"),
            ("- Worktree: `/repo/.claude/worktrees/one`\n"
             "- Worktree: `/repo/.claude/worktrees/two`\n"
             "- Branch: `claude/worker`\n"
             "- Base: `0123456789abcdef0123456789abcdef01234567`"),
            ("- Worktree: `/repo/$USER/worktree`\n"
             "- Branch: `claude/worker`\n"
             "- Base: `0123456789abcdef0123456789abcdef01234567`"),
            ("- Worktree: `/repo/$(id)/worktree`\n"
             "- Branch: `claude/worker`\n"
             "- Base: `0123456789abcdef0123456789abcdef01234567`"),
            ("- Worktree: `/repo/*/worktree`\n"
             "- Branch: `claude/worker`\n"
             "- Base: `0123456789abcdef0123456789abcdef01234567`"),
        )
        for checkout in invalid_checkouts:
            with self.subTest(checkout=checkout):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Execution checkout": checkout}))

        redteam = packet(
            role="redteam",
            bodies={"Regression test": "Add a suitable regression test."})
        with self.assertRaisesRegex(DirectiveError, "visible bullet locator"):
            validate_directive_text(role="redteam", text=redteam)

        weak_locators = (
            "- `repo/path::symbol-or-section`: exact edit",
            "- `repo/path::test-name`: exact test",
            "- `ai/tools/example.py::validate`: x",
            "- `ai/tools/example.py::validate`: !!!!!!!!!!!!",
            "- `ai/tools/example.py::validate`: fix",
            "- `ai/tools/example.py::validate`: edit",
            "- `ai/tools/example.py::validate`: add",
            "- `ai/tools/example.py::validate`: test",
            "- `ai/tools/example.py::validate`: change it",
            "- `ai/tools/example.py::validate`: edit code",
            "- `ai/tools/example.py::validate`: update code",
            "- `ai/tests/test_example.py::test_validator`: add coverage",
            "- `path/to/source::function-name`: change validator",
            "- `some/path::some_symbol`: update code",
            "- `your/file.py::your_symbol`: modify behavior",
            "- `ai/tools/*.py::all_functions`: replace each parser with "
            "exact validated behavior",
            "- `ai/tools/example.py::*`: replace the parser with exact "
            "validated behavior",
            "- `some_file.py::some_function`: replace the parser with "
            "exact validated behavior",
            "- `your_file.py::your_function`: replace the parser with "
            "exact validated behavior",
            "- `example.py::whatever`: replace the parser with exact "
            "validated behavior",
            "- `ai/tools/example.py::anything`: replace the parser with "
            "exact validated behavior",
            "- `ai/tools/example.py::relevant function`: replace the "
            "parser with exact validated behavior",
            "- `http://example.py::validate`: replace the parser with exact "
            "validated behavior",
        )
        for locator in weak_locators:
            with self.subTest(locator=locator):
                with self.assertRaisesRegex(
                        DirectiveError, "visible bullet locator"):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Files and symbols": locator}))

        mixed_locator = ARCHITECT_BODIES["Files and symbols"]
        mixed_locator += "\n- `ai/tools/other.py::run`: edit code"
        with self.assertRaisesRegex(DirectiveError, "every locator"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect",
                    bodies={"Files and symbols": mixed_locator}))

        extra_scope_locators = (
            ARCHITECT_BODIES["Files and symbols"]
            + "\nAlso edit `../../outside.py::whatever` to keep behavior "
            "aligned.",
            "- `ai/tools/example.py::validate`: replace the parser and edit "
            "`../../outside.py::whatever` too.",
        )
        for locator_body in extra_scope_locators:
            with self.subTest(extra_scope=locator_body):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Files and symbols": locator_body}))

        contradictory_checkout = ARCHITECT_BODIES["Execution checkout"]
        contradictory_checkout += (
            "\nIgnore those fields and use the primary checkout instead.")
        with self.assertRaisesRegex(DirectiveError, "extra prose"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect",
                    bodies={"Execution checkout": contradictory_checkout}))

        indented_checkout = "\n".join(
            "    " + line
            for line in ARCHITECT_BODIES["Execution checkout"].split("\n"))
        with self.assertRaises(DirectiveError):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect",
                    bodies={"Execution checkout": indented_checkout}))

    def test_numbered_steps_checkbox_and_command_fence_are_required(self):
        cases = (
            ({"Ordered implementation steps":
              "Change the validator without a numbered procedure."},
             "numbered procedure"),
            ({"Acceptance checklist":
              "The validator should probably work after this change."},
             "Markdown checkbox"),
            ({"Validation commands":
              "Run python3 -m unittest and expect exit zero."},
             "closed bash/sh/shell/zsh fence"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(DirectiveError, message):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect", bodies=changes))

        decorative = (
            "Inline markers ```do not form a block``` and the real block "
            "has no command.\n```bash\n# TBD\n```")
        with self.assertRaisesRegex(DirectiveError, "placeholder"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect",
                    bodies={"Validation commands": decorative}))

        comment_only = (
            "Inline markers ```still are not a command block```.\n"
            "```bash\n# explain the future command\n```")
        with self.assertRaisesRegex(DirectiveError, "non-comment"):
            validate_directive_text(
                role="architect",
                text=packet(
                    role="architect",
                    bodies={"Validation commands": comment_only}))

        non_commands = (
            "This is only an example, not a command.",
            "if then else",
            "./this-command-does-not-exist",
            "/definitely/missing/command",
        )
        for command in non_commands:
            with self.subTest(non_command=command):
                body = "Run the exact validation command.\n```bash\n"
                body += command + "\n```"
                with self.assertRaisesRegex(
                        DirectiveError, "syntax-valid, resolvable"):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Validation commands": body}))

        invisible_commands = ("\u200b", "\u2060", "&nbsp;", "------------")
        for command in invisible_commands:
            with self.subTest(command=repr(command)):
                body = "Run the exact validation command.\n```bash\n"
                body += command + "\n```"
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Validation commands": body}))

    def test_code_examples_cannot_supply_binding_packet_structure(self):
        fenced = (
            ("Files and symbols",
             "The exact locator is unspecified.\n"
             "```text\n`ai/tools/example.py::validate`\n```",
             "visible bullet locator"),
            ("Execution checkout",
             "The exact checkout is unspecified.\n```text\n"
             "- Worktree: `/repo/.claude/worktrees/worker`\n"
             "- Branch: `claude/worker`\n"
             "- Base: `0123456789abcdef0123456789abcdef01234567`\n```",
             "Execution checkout"),
            ("Ordered implementation steps",
             "The ordered procedure is unspecified.\n"
             "```text\n1. This is only a quoted example.\n```",
             "numbered procedure"),
            ("Acceptance checklist",
             "The acceptance conditions are unspecified.\n"
             "```text\n- [ ] This is only a quoted example.\n```",
             "Markdown checkbox"),
        )
        for heading, body, message in fenced:
            with self.subTest(kind="fenced", heading=heading):
                with self.assertRaisesRegex(DirectiveError, message):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect",
                                    bodies={heading: body}))

        hidden_reference = packet(
            role="architect",
            bodies={"Outcome":
                    "[hidden]: /unused \"Add one bounded validator.\""})
        with self.assertRaisesRegex(DirectiveError, "too short"):
            validate_directive_text(
                role="architect", text=hidden_reference)

        reference_locator = packet(
            role="architect",
            bodies={"Files and symbols":
                    "No exact binding source locator is visible.\n"
                    "[fake]: `ai/tools/example.py::validate`"})
        with self.assertRaisesRegex(
                DirectiveError, "visible bullet locator"):
            validate_directive_text(
                role="architect", text=reference_locator)

        indented = (
            ("Files and symbols",
             "The exact locator is unspecified.\n"
             "    `ai/tools/example.py::validate`"),
            ("Files and symbols",
             "The exact locator is unspecified.\n"
             " \t`ai/tools/example.py::validate`"),
            ("Execution checkout",
             "The exact checkout is unspecified.\n"
             "    - Worktree: `/repo/.claude/worktrees/worker`\n"
             "    - Branch: `claude/worker`\n"
             "    - Base: `0123456789abcdef0123456789abcdef01234567`"),
            ("Ordered implementation steps",
             "The ordered procedure is unspecified.\n"
             "    1. This is only a quoted example."),
            ("Acceptance checklist",
             "The acceptance conditions are unspecified.\n"
             "    - [ ] This is only a quoted example."),
            ("Files and symbols",
             "The exact locator is unspecified.\n"
             "> `ai/tools/example.py::quoted_example`"),
            ("Files and symbols",
             "> This quoted paragraph names only an example.\n"
             "`ai/tools/example.py::lazy_quoted_locator`"),
        )
        for heading, body in indented:
            with self.subTest(kind="indented", heading=heading):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect",
                                    bodies={heading: body}))

    def test_hidden_markdown_metadata_cannot_supply_binding_content(self):
        hidden_outcomes = (
            "![One bounded validator checks directive notes safely.]"
            "(ai/notes/assets/role-model-agent-loop.svg)",
            "[x](unused \"One bounded validator checks notes safely.\")",
            "[x](one-bounded-validator-checks-notes-safely)",
            "[a\\]b]: /unused \"One bounded validator checks notes safely.\"",
            "&#x200B;&#x200B;&#x200B;&#x200B;",
            "&nbsp;&nbsp;&nbsp;&nbsp;",
            "\u200b" * 12,
            "\u2060" * 12,
            "------------",
            "************",
            "_ _ _ _ _ _ _",
        )
        for outcome in hidden_outcomes:
            with self.subTest(outcome=repr(outcome)):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect", bodies={"Outcome": outcome}))

        hidden_locators = (
            "- ![`ai/tools/example.py::validate`]"
            "(ai/notes/assets/role-model-agent-loop.svg)",
            "- [No visible locator](unused "
            "\"`ai/tools/example.py::validate`\")",
            "- [No visible locator](`ai/tools/example.py::validate`)",
            "- The locator appears later: `ai/tools/example.py::validate`",
        )
        for locator in hidden_locators:
            with self.subTest(locator=locator):
                with self.assertRaises(DirectiveError):
                    validate_directive_text(
                        role="architect",
                        text=packet(
                            role="architect",
                            bodies={"Files and symbols": locator}))

    def test_required_structured_rows_have_substantive_payloads(self):
        weak_rows = (
            ("Ordered implementation steps",
             "Procedure marker follows below.\n1. !!!!!!!!!!!!",
             "numbered procedure"),
            ("Acceptance checklist",
             "Acceptance marker follows below.\n- [ ] !!!!!!!!!!!!",
             "Markdown checkbox"),
            ("Ordered implementation steps",
             "1. Add the exact validator before publication.\n"
             "2. !!!!!!!!!!!!",
             "numbered procedure"),
            ("Acceptance checklist",
             "- [ ] The complete valid packet passes validation.\n"
             "- [ ] !!!!!!!!!!!!",
             "Markdown checkbox"),
        )
        for heading, body, diagnostic in weak_rows:
            with self.subTest(heading=heading):
                with self.assertRaisesRegex(DirectiveError, diagnostic):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect", bodies={heading: body}))

    def test_container_hidden_markers_cannot_supply_structure(self):
        nested_cases = (
            ("Files and symbols",
             "- `ai/tools/example.py::validate`: fake edit only."),
            ("Ordered implementation steps",
             "1. Fake plan inside code."),
            ("Acceptance checklist",
             "- [ ] Fake acceptance inside code."),
            ("Execution checkout",
             "- Worktree: `/repo/.claude/worktrees/worker`\n"
             "  - Branch: `claude/worker`\n"
             "  - Base: `0123456789abcdef0123456789abcdef01234567`"),
        )
        for heading, hidden in nested_cases:
            body = (
                "The value below is only an example.\n- ~~~text\n  "
                + hidden.replace("\n", "\n  ")
                + "\n  ~~~\n~~~~\n~~~\n~~~~")
            with self.subTest(kind="list-fence", heading=heading):
                with self.assertRaisesRegex(DirectiveError, "nested"):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect", bodies={heading: body}))

        math_cases = (
            ("Files and symbols",
             "- `ai/tools/example.py::validate`: fake edit only."),
            ("Ordered implementation steps", "1. Fake plan inside math."),
            ("Acceptance checklist", "- [ ] Fake check inside math."),
        )
        for heading, hidden in math_cases:
            body = "The row is mathematical source only.\n$$\n"
            body += hidden + "\n$$"
            with self.subTest(kind="display-math", heading=heading):
                with self.assertRaisesRegex(DirectiveError, "display-math"):
                    validate_directive_text(
                        role="architect",
                        text=packet(role="architect", bodies={heading: body}))

    def test_leading_frontmatter_cannot_hide_a_packet(self):
        visible = "---\ntitle: Scratch ticket\n---\n" + packet(
            role="architect")
        validate_directive_text(role="architect", text=visible)
        validate_directive_text(
            role="architect", text="\ufeff" + visible)

        hidden_packets = (
            "---\n" + packet(role="architect") + "\n---\n",
            "\ufeff---\n" + packet(role="architect") + "\n---\n",
        )
        for hidden in hidden_packets:
            with self.subTest(bom=hidden.startswith("\ufeff")):
                with self.assertRaisesRegex(DirectiveError, "found 0"):
                    validate_directive_text(role="architect", text=hidden)

        for whitespace in ("   ", "\t"):
            visible = (
                "---" + whitespace + "\ntitle: Scratch ticket\n---"
                + whitespace + "\n" + packet(role="architect"))
            validate_directive_text(role="architect", text=visible)
            hidden = (
                "---" + whitespace + "\n" + packet(role="architect")
                + "\n---" + whitespace + "\n")
            with self.subTest(delimiter=repr(whitespace)):
                with self.assertRaisesRegex(DirectiveError, "found 0"):
                    validate_directive_text(role="architect", text=hidden)

    def test_only_markdown_line_endings_define_packet_rows(self):
        source = packet(role="architect")
        validate_directive_text(role="architect", text=source)
        validate_directive_text(
            role="architect", text=source.replace("\n", "\r\n"))
        validate_directive_text(
            role="architect", text=source.replace("\n", "\r"))

        non_markdown_breaks = (
            "\u000b", "\u000c", "\u001c", "\u001d", "\u001e",
            "\u0085", "\u2028", "\u2029",
        )
        for separator in non_markdown_breaks:
            with self.subTest(separator=repr(separator)):
                malformed = source.replace("\n", separator)
                with self.assertRaisesRegex(DirectiveError, "non-Markdown"):
                    validate_directive_text(
                        role="architect", text=malformed)

    def test_noncanonical_heading_boundaries_refuse(self):
        source = packet(role="architect")
        evidence = "## Implementation evidence / resume state"
        setext_boundaries = (
            "Different top-level ticket\n==========================\n\n",
            "Different peer ticket\n---------------------\n\n",
        )
        for boundary in setext_boundaries:
            with self.subTest(setext=boundary.split("\n")[1]):
                malformed = source.replace(
                    evidence, boundary + evidence, 1)
                with self.assertRaisesRegex(DirectiveError, "Setext"):
                    validate_directive_text(
                        role="architect", text=malformed)

        midpacket = source.replace(
            "### Files and symbols",
            "Different packet boundary\n-------------------------\n\n"
            "### Files and symbols", 1)
        with self.assertRaisesRegex(DirectiveError, "Setext"):
            validate_directive_text(role="architect", text=midpacket)

        atx_boundaries = (
            " # Different boundary\n\n",
            "   ## Different boundary\n\n",
            "#\n\n",
            "##\n\n",
            "  ##\n\n",
        )
        for boundary in atx_boundaries:
            with self.subTest(atx=repr(boundary)):
                malformed = source.replace(
                    evidence, boundary + evidence, 1)
                with self.assertRaisesRegex(
                        DirectiveError, "followed immediately"):
                    validate_directive_text(
                        role="architect", text=malformed)

    def test_code_fence_headings_do_not_change_packet_structure(self):
        commands = (
            "The command emits a Markdown-looking line.\n"
            "```bash\nprintf '### not a packet heading\\n'\n```")
        plan = ARCHITECT_BODIES["Parallel work plan"].replace(
            "- Final validation: Run `python3 -m unittest "
            "ai.tests.test_example` and require exit zero after integration.",
            "- Final validation: Run `printf '### not a packet heading\\n'` "
            "and require exit zero after integration.")
        validate_directive_text(
            role="architect",
            text=packet(role="architect",
                        bodies={"Validation commands": commands,
                                "Parallel work plan": plan}))

    def test_commented_and_invalidly_closed_packets_do_not_validate(self):
        commented = "<!--\n" + packet(role="architect") + "\n-->\n"
        with self.assertRaisesRegex(DirectiveError, "found 0"):
            validate_directive_text(role="architect", text=commented)

        hidden = "```bash\n# held open\n```still-code\n" + packet(
            role="architect")
        with self.assertRaisesRegex(DirectiveError, "found 0"):
            validate_directive_text(role="architect", text=hidden)

        hidden_body = packet(
            role="architect",
            bodies={"Outcome":
                    "<!--\nThis complete result is hidden from readers.\n-->"})
        with self.assertRaisesRegex(DirectiveError, "too short"):
            validate_directive_text(role="architect", text=hidden_body)

        manufactured = packet(role="architect")
        manufactured = manufactured.replace(
            "## Implementation directive",
            "#<!-- hidden --># Implementation directive", 1)
        with self.assertRaisesRegex(DirectiveError, "found 0"):
            validate_directive_text(role="architect", text=manufactured)

        suffixed = packet(role="architect").replace(
            "## Implementation directive",
            "## Implementation directive###", 1)
        with self.assertRaisesRegex(DirectiveError, "found 0"):
            validate_directive_text(role="architect", text=suffixed)

    def test_architect_packet_requires_one_immediate_evidence_destination(self):
        valid = packet(role="architect")
        missing = valid.replace(
            "## Implementation evidence / resume state",
            "## Evidence log", 1)
        with self.assertRaisesRegex(DirectiveError, "followed immediately"):
            validate_directive_text(role="architect", text=missing)

        duplicate = valid + (
            "\n## Implementation evidence / resume state\nDuplicate.\n")
        with self.assertRaisesRegex(DirectiveError, "exactly one sibling"):
            validate_directive_text(role="architect", text=duplicate)

        wrong_parent = valid.replace(
            "## Implementation evidence / resume state",
            "# Different top-level ticket\n\n"
            "## Implementation evidence / resume state", 1)
        with self.assertRaisesRegex(DirectiveError, "followed immediately"):
            validate_directive_text(role="architect", text=wrong_parent)

    def test_raw_html_blocks_cannot_hide_packet_structure(self):
        wrappers = (
            ("<pre>", "</pre>"),
            ("<script>", "</script>"),
            ("<style>", "</style>"),
            ("<textarea>", "</textarea>"),
            ("<?hide", "?>"),
            ("<![CDATA[", "]]>") ,
            ("<!HIDE", ">"),
            ("<div>", "</div>"),
            ("<x-hide>", "</x-hide>"),
        )
        for opening, closing in wrappers:
            with self.subTest(opening=opening):
                hidden = opening + "\n" + packet(role="architect")
                hidden += "\n" + closing + "\n"
                with self.assertRaisesRegex(DirectiveError, "raw HTML"):
                    validate_directive_text(role="architect", text=hidden)

        inline_wrappers = (
            ("prefix <pre>", "suffix </pre>"),
            ("prefix <details>", "suffix </details>"),
            ("prefix <div style=\"display:none\">", "suffix </div>"),
        )
        for opening, closing in inline_wrappers:
            with self.subTest(opening=opening):
                hidden = opening + "\n" + packet(role="architect")
                hidden += "\n" + closing + "\n"
                with self.assertRaisesRegex(DirectiveError, "raw HTML"):
                    validate_directive_text(role="architect", text=hidden)

        for opening in ("<pre", "<script", "<style", "<textarea"):
            with self.subTest(opening=opening):
                hidden = opening + "\n" + packet(role="architect")
                with self.assertRaisesRegex(DirectiveError, "raw HTML"):
                    validate_directive_text(role="architect", text=hidden)

    def test_unknown_role_nonstring_and_nul_refuse(self):
        with self.assertRaisesRegex(DirectiveError, "unknown directive role"):
            validate_directive_text(role="implementer", text="note")
        with self.assertRaisesRegex(DirectiveError, "native string"):
            validate_directive_text(role="architect", text=b"note")
        with self.assertRaisesRegex(DirectiveError, "NUL"):
            validate_directive_text(role="architect", text="note\x00")

    def test_file_validation_is_read_only_bounded_and_utf8_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            valid = root / "valid.md"
            valid.write_text(packet(role="architect"), encoding="utf-8")
            before = valid.read_bytes()
            validate_directive_file(role="architect", path=valid)
            self.assertEqual(valid.read_bytes(), before)

            bad_utf8 = root / "bad.md"
            bad_utf8.write_bytes(b"\xff")
            with self.assertRaisesRegex(DirectiveError, "UTF-8"):
                validate_directive_file(role="architect", path=bad_utf8)

            oversized = root / "oversized.md"
            with oversized.open("wb") as stream:
                stream.truncate(MAX_NOTE_BYTES + 1)
            with self.assertRaisesRegex(DirectiveError, "exceeds"):
                validate_directive_file(role="architect", path=oversized)

            if hasattr(os, "symlink") and hasattr(os, "O_NOFOLLOW"):
                link = root / "link.md"
                link.symlink_to(valid)
                with self.assertRaisesRegex(DirectiveError, "open.*safely"):
                    validate_directive_file(role="architect", path=link)

            before_stat = os.stat(valid)
            after_stat = SimpleNamespace(
                st_dev=before_stat.st_dev,
                st_ino=before_stat.st_ino,
                st_size=before_stat.st_size,
                st_mtime_ns=before_stat.st_mtime_ns + 1,
                st_ctime_ns=before_stat.st_ctime_ns,
                st_mode=before_stat.st_mode)
            with mock.patch(
                    "ai.tools.handoff_contract.os.fstat",
                    side_effect=(before_stat, after_stat)):
                with self.assertRaisesRegex(DirectiveError, "changed"):
                    validate_directive_file(role="architect", path=valid)

            ctime_changed = SimpleNamespace(
                st_dev=before_stat.st_dev,
                st_ino=before_stat.st_ino,
                st_size=before_stat.st_size,
                st_mtime_ns=before_stat.st_mtime_ns,
                st_ctime_ns=before_stat.st_ctime_ns + 1,
                st_mode=before_stat.st_mode)
            with mock.patch(
                    "ai.tools.handoff_contract.os.fstat",
                    side_effect=(before_stat, ctime_changed)):
                with self.assertRaisesRegex(DirectiveError, "changed"):
                    validate_directive_file(role="architect", path=valid)

    def test_mailbox_shared_notes_rejects_relative_local_and_redirected_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            shared = root / "primary" / "ai" / "notes"
            sol_local = root / "sol" / "ai" / "notes"
            shared.mkdir(parents=True)
            sol_local.mkdir(parents=True)
            authoritative = shared / "ticket.md"
            local = sol_local / "ticket.md"
            authoritative.write_text(
                packet(role="architect"), encoding="utf-8")
            local.write_text(packet(role="architect"), encoding="utf-8")

            with mock.patch.dict(
                    os.environ, {"MAILBOX_SHARED_NOTES": str(shared)},
                    clear=False):
                result = validate_directive_file(
                    role="architect", path=authoritative)
                self.assertEqual(result["role"], "architect")

                for path, message in (
                        (Path("ai/notes/ticket.md"), "absolute path"),
                        (local, "outside MAILBOX_SHARED_NOTES")):
                    with self.subTest(path=str(path)):
                        with self.assertRaisesRegex(DirectiveError, message):
                            validate_directive_file(
                                role="architect", path=path)

                linked_parent = root / "linked-notes"
                try:
                    linked_parent.symlink_to(shared, target_is_directory=True)
                except (OSError, NotImplementedError):
                    return
                with self.assertRaisesRegex(
                        DirectiveError, "outside MAILBOX_SHARED_NOTES"):
                    validate_directive_file(
                        role="architect", path=linked_parent / "ticket.md")

                linked_note = shared / "linked-ticket.md"
                linked_note.symlink_to(authoritative)
                with self.assertRaisesRegex(DirectiveError, "redirected path"):
                    validate_directive_file(
                        role="architect", path=linked_note)

    def test_mailbox_contract_rejects_a_non_authoritative_validator_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "ticket.md"
            note.write_text(packet(role="architect"), encoding="utf-8")
            authoritative = str(
                (Path(__file__).resolve().parents[1]
                 / "tools" / "handoff_contract.py").resolve())
            with mock.patch.dict(
                    os.environ,
                    {"MAILBOX_HANDOFF_CONTRACT": authoritative},
                    clear=False):
                result = validate_directive_file(
                    role="architect", path=note)
                self.assertEqual(result["role"], "architect")

            with mock.patch.dict(
                    os.environ,
                    {"MAILBOX_HANDOFF_CONTRACT":
                     str(Path(directory) / "handoff_contract.py")},
                    clear=False):
                with self.assertRaisesRegex(
                        DirectiveError, "not the authoritative absolute"):
                    validate_directive_file(role="architect", path=note)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO unavailable")
    def test_fifo_refuses_promptly_in_a_bounded_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "directive.fifo"
            os.mkfifo(fifo)
            tool = Path(__file__).resolve().parents[1] / "tools" / "handoff_contract.py"
            result = subprocess.run(
                [sys.executable, str(tool), "architect", str(fifo)],
                capture_output=True,
                text=True,
                timeout=2)
            self.assertEqual(result.returncode, 1)
            self.assertIn("directive: INVALID", result.stdout)

    def test_cli_reports_structure_without_issuing_role_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            architect = Path(directory) / "architect.md"
            redteam = Path(directory) / "redteam.md"
            architect.write_text(packet(role="architect"), encoding="utf-8")
            redteam.write_text(packet(role="redteam"), encoding="utf-8")

            for role, note, expected in (
                    ("architect", architect, "architect directive: VALID:"),
                    ("redteam", redteam, "redteam directive: VALID:")):
                with self.subTest(role=role):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        result = main(argv=[role, str(note)])
                    self.assertEqual(result, 0)
                    self.assertIn(expected, output.getvalue())

            redteam.write_text("# incomplete\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(argv=["redteam", str(redteam)])
            self.assertEqual(result, 1)
            self.assertIn("redteam directive: INVALID:", output.getvalue())

    def test_cli_character_limit_defaults_to_zero_and_matches_note(self):
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "architect.md"
            note.write_text(packet(role="architect"), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(argv=["architect", str(note)])
            self.assertEqual(result, 0)
            self.assertIn("architect directive: VALID:", output.getvalue())

            bounded = packet(
                role="architect",
                bodies=bounded_bodies(limit=25, planned=20))
            note.write_text(bounded, encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(
                    argv=["architect", str(note), "--max", "25"])
            self.assertEqual(result, 0)

            output = io.StringIO()
            with redirect_stdout(output):
                result = main(argv=["architect", str(note)])
            self.assertEqual(result, 1)
            self.assertIn("does not match", output.getvalue())

            for invalid in ("-1", "one"):
                with self.subTest(invalid=invalid):
                    with redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as raised:
                            main(argv=[
                                "architect", str(note), "--max", invalid])
                    self.assertEqual(raised.exception.code, 2)

    def test_mailbox_environment_binds_omitted_and_explicit_cli_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "architect.md"
            note.write_text(
                packet(role="architect", bodies=bounded_bodies(25, 20)),
                encoding="utf-8")

            with mock.patch.dict(
                    os.environ, {"MAILBOX_MAX_CHARACTERS": "25"},
                    clear=False):
                for argv in (
                        ["architect", str(note)],
                        ["architect", str(note), "--max", "25"]):
                    with self.subTest(argv=argv):
                        output = io.StringIO()
                        with redirect_stdout(output):
                            result = main(argv=argv)
                        self.assertEqual(result, 0)
                        self.assertIn("directive: VALID", output.getvalue())

                output = io.StringIO()
                with redirect_stdout(output):
                    result = main(
                        argv=["architect", str(note), "--max", "24"])
                self.assertEqual(result, 1)
                self.assertIn(
                    "does not match MAILBOX_MAX_CHARACTERS",
                    output.getvalue())

    def test_mailbox_environment_limit_is_strict_ascii_decimal(self):
        for invalid in ("", " 25", "+25", "2_5", "٢٥", "１２"):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(
                        DirectiveError, "ASCII decimal digits"):
                    resolve_character_limit(
                        cli_value=None, environment_value=invalid)

        self.assertEqual(
            resolve_character_limit(cli_value=None, environment_value="0"),
            0)
        self.assertEqual(
            resolve_character_limit(cli_value=None, environment_value="25"),
            25)
        self.assertEqual(
            resolve_character_limit(cli_value=25, environment_value="25"),
            25)
        with self.assertRaisesRegex(
                DirectiveError, "does not match MAILBOX_MAX_CHARACTERS"):
            resolve_character_limit(cli_value=24, environment_value="25")

    def test_mailbox_environment_binds_redteam_discovery_severity(self):
        high_finding = REDTEAM_BODIES["Finding and evidence"].replace(
            "User severity setting: `medium`",
            "User severity setting: `high`").replace(
                "Red Team severity: `medium`",
                "Red Team severity: `high`")
        high_packet = packet(
            role="redteam",
            bodies={"Finding and evidence": high_finding})
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "redteam.md"
            note.write_text(high_packet, encoding="utf-8")
            with mock.patch.dict(
                    os.environ,
                    {"MAILBOX_DISCOVERY_SEVERITY": "high"}, clear=False):
                result = validate_directive_file(
                    role="redteam", path=note)
                self.assertEqual(
                    result["discovery_severity_assessment"]["user_setting"],
                    "high")
                with self.assertRaisesRegex(
                        DirectiveError, "does not match the run-time"):
                    validate_directive_text(
                        role="redteam", text=packet(role="redteam"),
                        expected_severity="high")
                output = io.StringIO()
                with redirect_stdout(output):
                    rc = main(argv=[
                        "redteam", str(note), "--severity", "low"])
                self.assertEqual(rc, 1)
                self.assertIn("does not match", output.getvalue())

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MAILBOX_DISCOVERY_SEVERITY", None)
                output = io.StringIO()
                with redirect_stdout(output):
                    rc = main(argv=[
                        "redteam", str(note), "--severity", "high"])
                self.assertEqual(rc, 0)
                output = io.StringIO()
                with redirect_stdout(output):
                    rc = main(argv=[
                        "architect", str(note), "--severity", "high"])
                self.assertEqual(rc, 1)
                self.assertIn("only for a Red Team", output.getvalue())

        self.assertEqual(
            resolve_discovery_severity(
                cli_value=None, environment_value=None), "medium")
        for invalid in ("", " HIGH ", "High", "critical"):
            with self.subTest(invalid=repr(invalid)), self.assertRaisesRegex(
                    DirectiveError, "must be exactly"):
                resolve_discovery_severity(
                    cli_value=None, environment_value=invalid)


if __name__ == "__main__":
    unittest.main()
