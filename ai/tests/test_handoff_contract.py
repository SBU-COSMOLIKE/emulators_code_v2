"""Focused tests for decision-complete handoff directive packets."""

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
from ai.tools.handoff_contract import main
from ai.tools.handoff_contract import validate_directive_file
from ai.tools.handoff_contract import validate_directive_text


ARCHITECT_BODIES = {
    "Outcome": "Add one bounded read-only directive validation command.",
    "Starting point": (
        "At commit 0123456, ai/tools/example.py accepts unchecked notes."),
    "Execution checkout": (
        "- Worktree: `/repo/.claude/worktrees/mailbox-primary`\n"
        "- Branch: `claude/mailbox-primary`\n"
        "- Base: `0123456789abcdef0123456789abcdef01234567`"),
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
        "Keep this unit serial because code and test share one tiny contract."),
}

REDTEAM_BODIES = {
    "Finding and evidence": (
        "Commit 0123456 accepts a missing directive; raw test exits zero."),
    "Root cause": (
        "The dispatch path checks transport bytes but not the cited packet."),
    "Required outcome": (
        "Refuse an incomplete packet before an Implementer begins editing."),
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
                validate_directive_text(role=role, text=packet(role=role))

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
        validate_directive_text(
            role="architect",
            text=packet(role="architect",
                        bodies={"Validation commands": commands}))

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
            root = Path(directory)
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


if __name__ == "__main__":
    unittest.main()
