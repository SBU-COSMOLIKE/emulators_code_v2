"""Keep permanent AI notes neutral, durable, and free of ticket history."""

from collections import Counter
from pathlib import Path
import re
import subprocess
import unittest

from ai.tools.permanent_note_guard import PERMANENT_NOTES


REPO_ROOT = Path(__file__).resolve().parents[2]

CALENDAR_RE = re.compile(
    r"(?:\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b|"
    r"(?<![~:])\b(?:19\d{2}|20\d{2})\b|"
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}\b)",
    re.IGNORECASE,
)
PERSONAL_RE = re.compile(
    r"\b(?:Vivian|she|her|hers|herself|he|him|his|himself)\b",
    re.IGNORECASE,
)
DIARY_PATTERNS = (
    re.compile(r"Architect[- ](?:VERIFIED|CONFIRMED|REPRODUCED)", re.IGNORECASE),
    re.compile(r"awaiting Architect (?:audit|adjudication)", re.IGNORECASE),
    re.compile(r"\b(?:20M|25M|45M|BLOAT|DIDACTICS)-[A-Za-z0-9-]+\b"),
    re.compile(r"\bRT-(?:19|20)\d{2}-\d{2}-\d{2}-\d+\b"),
    re.compile(r"\b(?:UNIT|Unit)\s+\d+\b"),
    re.compile(r"\bqueue\s+\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
               r"ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|"
               r"fifteenth|sixteenth|seventeenth)\s+wave\b",
               re.IGNORECASE),
    re.compile(r"\bboard run\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bthe user asleep\b", re.IGNORECASE),
    re.compile(r"\bgit archaeology\b", re.IGNORECASE),
    re.compile(r"\b(?:today|tonight|yesterday|tomorrow|same evening)\b",
               re.IGNORECASE),
    re.compile(r"\b(?:authorizing ruling|implementation readback|"
               r"audit readback|current owner)\b", re.IGNORECASE),
    re.compile(r"\b(?:full|untruncated) grep\b", re.IGNORECASE),
    re.compile(r"\b(?:all verified|both confirmed|second false report)\b",
               re.IGNORECASE),
    re.compile(r"\*\*Required implementation\.\*\*", re.IGNORECASE),
)
ANCHOR_RE = re.compile(r'<a\s+id="([^"]+)"\s*></a>')

# Each entry is (permanent-note path, exact date-bearing text, lasting reason).
# Keep the list narrow: the current permanent notes need no date exception.
PERMANENT_NOTE_CALENDAR_ALLOWLIST = ()


def read(relative):
    """Read one UTF-8 repository file."""
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


class PermanentNoteStyleContractTests(unittest.TestCase):
    """Enforce the Architect-owned permanent-note writing contract."""

    def test_python_contract_replaces_the_personal_voice_note(self):
        retired_name = "user-didactics-" + "and-python-voice.md"
        self.assertIn(
            "ai/notes/python-changes-go-no-go.md",
            PERMANENT_NOTES,
        )
        self.assertNotIn(
            "ai/notes/" + retired_name,
            PERMANENT_NOTES,
        )
        contract = read("ai/notes/python-changes-go-no-go.md")
        self.assertTrue(contract.startswith(
            "# GO/NO-GO contract for the style of Python changes\n"
        ))
        self.assertIn("style is a release condition", contract)
        self.assertIn("Architect review before dispatch", contract)
        self.assertIn("Architect review before final verdict", contract)

    def test_memory_starts_the_permanent_note_change_contract(self):
        memory = read("ai/notes/MEMORY.md")
        normalized = " ".join(memory.split())
        contract_at = memory.index(
            "## GO/NO-GO contract for changing a permanent note"
        )
        list_at = memory.index("## The permanent eleven")
        self.assertLess(contract_at, list_at)
        self.assertIn("The permanent notes are not a development diary", memory)
        self.assertIn("Only the Architect may edit a permanent note", memory)
        self.assertIn("Rule:", memory)
        self.assertIn("Acceptance evidence:", memory)
        self.assertIn("future development model and a physics undergraduate",
                      " ".join(memory.split()))
        self.assertIn("creates or updates a tracked backlog ticket in the same turn",
                      memory)
        self.assertIn("anti-AI requirements", memory)
        self.assertIn(
            "The text contains no development date, policy-provenance "
            "timestamp",
            normalized,
        )
        self.assertIn(
            "A date is allowed only when it is part of scientific, release, "
            "citation, input, or public-interface subject matter",
            normalized,
        )
        self.assertIn(
            "the review records why removing it would make the statement "
            "incomplete or false",
            normalized,
        )
        self.assertIn(
            "numbered review-run history used as a development diary",
            normalized,
        )

    def test_readme_contract_covers_repeated_reader_failures(self):
        contract = read("ai/notes/readme-go-no-go.md")
        self.assertIn("neutral-audience", contract)
        self.assertIn("flows from top to bottom", contract)
        self.assertIn("first diagram is a small mental model", contract)
        self.assertIn("GitHub-supported `$$ ... $$` blocks render", contract)
        self.assertIn("Plain `[ ... ]` is used as pseudo-math", contract)
        self.assertIn("section **README and teaching contract**", contract)
        self.assertNotIn("section **README / didactics**", contract)
        for number in range(1, 7):
            self.assertIn("## Review " + str(number) + ":", contract)
            self.assertNotIn("## Gate " + str(number) + ":", contract)

    def test_prose_contract_requires_one_current_account_not_policy_history(self):
        """Keep the durable rule while discarding its development diary."""
        contract = read("ai/notes/readme-go-no-go.md")
        normalized = " ".join(contract.split())

        self.assertIn("## Describe one coherent current system", contract)
        self.assertIn(
            "README files, long-form documentation, permanent notes, commit "
            "explanations, and explanatory Python prose describe how the "
            "library works now.",
            normalized,
        )
        self.assertIn("rewrite the owning explanation in place", normalized)
        self.assertIn("Do not append a dated correction", normalized)
        for rejected in (
                "`hard user rule`",
                "ticket numbers",
                "audit waves",
                "review rounds",
                "model names",
                "chronological addendum",
                "policy-patch paragraph"):
            with self.subTest(rejected=rejected):
                self.assertIn(rejected, contract)
        self.assertIn(
            "one consistent current explanation after the edit", normalized)

        # Calendar and order words remain available when they describe the
        # subject itself rather than the history of a policy decision.
        for allowed in (
                "scientific data release named by year",
                "publication citation",
                "user input that is a date",
                "algorithm whose ordered phases are current behavior",
                "`history` and `phase` also have valid technical meanings"):
            with self.subTest(allowed=allowed):
                self.assertIn(allowed, normalized)

    def test_python_contract_replaces_policy_patches_with_current_explanation(self):
        """Require the concrete comment example used to teach this boundary."""
        contract = read("ai/notes/python-changes-go-no-go.md")
        normalized = " ".join(contract.split())

        self.assertIn(
            "### Explain current code, not the policy patches that produced it",
            contract,
        )
        self.assertIn("replace the old explanation in place", normalized)
        for rejected in (
                "dated correction",
                "`hard user rule`",
                "ticket number",
                "audit wave",
                "review round",
                "model name",
                "development chronology"):
            with self.subTest(rejected=rejected):
                self.assertIn(rejected, contract)

        bad = (
            "# Hard user rule from the latest review: now reject a dirty "
            "worktree."
        )
        good = (
            "# Refuse a dirty worktree so uncommitted user files cannot enter "
            "the landing."
        )
        self.assertIn("NO-GO:\n\n```python\n" + bad, contract)
        self.assertIn("GO:\n\n```python\n" + good, contract)
        self.assertLess(contract.index(bad), contract.index(good))

        for allowed in (
                "program reads or calculates that date",
                "scientific dataset or publication is identified by year",
                "public compatibility interface contains the date",
                "real runtime data or algorithmic order"):
            with self.subTest(allowed=allowed):
                self.assertIn(allowed, normalized)
        self.assertIn(
            "one compatible current explanation, not an old comment followed "
            "by a later exception",
            normalized,
        )
        self.assertIn(
            "explanatory text is personal, development-dated, vague, "
            "undefined, narrates policy or review history, or uses chronology "
            "without a scientific, runtime, algorithmic, or compatibility "
            "need;",
            normalized,
        )
        self.assertNotIn(
            "explanatory text is personal, dated, historical, vague, or "
            "undefined",
            normalized,
        )

    def test_feature_documentation_searches_for_one_existing_owner_first(self):
        """Keep deep guides bounded, discoverable, and nonduplicative."""
        conventions = read("ai/notes/conventions-and-workflow.md")
        readme_contract = read("ai/notes/readme-go-no-go.md")
        memory = read("ai/notes/MEMORY.md")
        normalized_conventions = " ".join(conventions.split())
        normalized_contract = " ".join(readme_contract.split())

        self.assertIn("### Feature-specific long-form documentation",
                      conventions)
        self.assertIn(
            "Before planning a new file, the Architect searches "
            "`documentation/README.md`, tracked files under "
            "`documentation/`, relevant README headings, and likely source "
            "names, symbols, commands, and synonyms.",
            normalized_conventions,
        )
        self.assertIn(
            "If one document already answers the same reader question, the "
            "plan updates that owner or improves the link to it.",
            normalized_conventions,
        )
        self.assertIn(
            "A second guide for the same question is `NO-GO`.",
            normalized_conventions,
        )
        self.assertIn(
            "Before creating a long-form document, the Architect searches "
            "`documentation/README.md`",
            normalized_contract,
        )
        self.assertIn(
            "Creating a second guide for the same question is `NO-GO`.",
            normalized_contract,
        )
        self.assertIn("search-first planning for feature-specific long-form "
                      "documentation", memory)

        focused = "documentation/candidate_to_landing.tex"
        whole_library = "documentation/emulator_code_guide.tex"
        for source in (conventions, readme_contract):
            with self.subTest(source=source[:40], document=focused):
                self.assertIn(focused, source)
            with self.subTest(source=source[:40], document=whole_library):
                self.assertIn(whole_library, source)

    def test_feature_documentation_has_fixed_priority_and_role_ownership(self):
        """Pin the Low default and the single explicit urgent exception."""
        conventions = read("ai/notes/conventions-and-workflow.md")
        normalized = " ".join(conventions.split())

        self.assertIn(
            "Feature-specific documentation is a **Low new-functionality "
            "ticket** by default.",
            normalized,
        )
        self.assertIn(
            "It becomes **High** only when the user explicitly requests High "
            "priority because understanding that feature is urgent.",
            normalized,
        )
        self.assertIn("Importance alone does not promote it.", normalized)
        self.assertIn(
            "Incorrect existing documentation that can damage normal use is "
            "a bug",
            normalized,
        )
        self.assertIn(
            "The Architect owns scope, duplicate prevention, the complete "
            "directive, factual review, and final `GO` or `NO-GO`.",
            normalized,
        )
        self.assertIn(
            "The Implementer writes the tracked source and compiled artifact.",
            normalized,
        )
        self.assertIn(
            "The Red Team may report a documentation defect and review the "
            "rendered result, but it never edits tracked documentation.",
            normalized,
        )

    def test_known_temporary_status_phrases_stay_out_of_permanent_notes(self):
        combined = "\n".join(read(relative) for relative in PERMANENT_NOTES)
        for phrase in (
                "That implementation does not yet satisfy this rule",
                "A future explicit dense-CMB mode",
                "must be the new `emulator/experiment.py::"
                "validate_active_model_values`"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, combined)

    def test_notes_record_the_cross_family_identity_and_capability_rules(self):
        artifact = " ".join(read(
            "ai/notes/artifacts-inference-warmstart.md").split())
        generation = " ".join(read(
            "ai/notes/data-generation-and-cuts.md").split())
        cmb = " ".join(read("ai/notes/families-scalar-cmb.md").split())
        models = " ".join(read("ai/notes/models-and-designs.md").split())

        for text in (artifact, generation):
            for identity in (
                    "Staged-selection identity",
                    "Artifact identity"):
                with self.subTest(note=text[:40], identity=identity):
                    self.assertIn(identity, text)
        self.assertIn("<data-vector-root>_ell.npy", cmb)
        self.assertIn("_load_axis_checkpoint", cmb)
        self.assertIn("Dense covariance training is unsupported", cmb)
        self.assertIn(
            "moves each requested endpoint one representable floating-point "
            "value toward the interval interior",
            generation)
        self.assertIn(
            "derived from the seed together with the existing row count",
            generation)
        self.assertIn(
            "never silently becomes fresh generation",
            generation)
        self.assertIn("output decoder and loss composition", artifact)
        self.assertIn("When `TCMB` is a sampled input", cmb)
        self.assertIn("both fixed-temperature and sampled-temperature", cmb)
        self.assertIn("validated ten-template dataset", models)
        self.assertIn("registry construction alone is not a claim", models)

    def test_notes_have_no_calendar_or_person_specific_language(self):
        self.assertIsNotNone(
            CALENDAR_RE.search("Hard user rule, 2040: add another patch."))
        allowed_by_path = {}
        seen_path_literals = set()
        for path, literal, reason in PERMANENT_NOTE_CALENDAR_ALLOWLIST:
            key = (path, literal)
            with self.subTest(note=path, literal=literal, kind="allowlist"):
                self.assertIn(path, PERMANENT_NOTES)
                self.assertTrue(literal.strip())
                self.assertEqual(len(list(CALENDAR_RE.finditer(literal))), 1)
                self.assertTrue(reason.strip())
                self.assertGreaterEqual(len(reason.split()), 4)
                self.assertNotIn(key, seen_path_literals)
            seen_path_literals.add(key)
            allowed_by_path.setdefault(path, []).append(literal)

        for relative in PERMANENT_NOTES:
            text = read(relative)
            calendar_text = text
            for literal in allowed_by_path.get(relative, []):
                with self.subTest(note=relative, literal=literal,
                                  kind="allowlist-occurrence"):
                    self.assertEqual(calendar_text.count(literal), 1)
                calendar_text = calendar_text.replace(literal, "", 1)
            with self.subTest(note=relative, kind="calendar"):
                self.assertIsNone(CALENDAR_RE.search(calendar_text))
            with self.subTest(note=relative, kind="personal"):
                self.assertIsNone(PERSONAL_RE.search(text))

    def test_notes_have_no_ticket_diary_markers(self):
        for relative in PERMANENT_NOTES:
            text = read(relative)
            for pattern in DIARY_PATTERNS:
                with self.subTest(note=relative, pattern=pattern.pattern):
                    self.assertIsNone(pattern.search(text))

    def test_stable_note_anchors_are_unique(self):
        for relative in PERMANENT_NOTES:
            anchors = ANCHOR_RE.findall(read(relative))
            duplicates = sorted(
                anchor for anchor, count in Counter(anchors).items()
                if count > 1
            )
            with self.subTest(note=relative):
                self.assertEqual(duplicates, [])

    def test_training_partition_anchor_has_a_durable_name(self):
        """Accept the deliberate migration away from a dated audit label."""
        training = read("ai/notes/training-stack.md")
        durable = "eval-batch-invariance-real-partitions"
        retired = "didactics-59-red-team-return-" + "2026-07-14"
        self.assertIn(f'<a id="{durable}"></a>', training)
        self.assertNotIn(retired, training)

        offenders = []
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for raw in result.stdout.split(b"\0"):
            if not raw:
                continue
            relative = raw.decode("utf-8", errors="strict")
            path = REPO_ROOT / relative
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if retired in text:
                offenders.append(relative)
        self.assertEqual(offenders, [])

    def test_permanent_notes_do_not_depend_on_the_local_audit_board(self):
        for relative in PERMANENT_NOTES:
            with self.subTest(note=relative):
                self.assertNotIn("gates-and-board.md", read(relative))

    def test_tracked_files_do_not_name_the_retired_note(self):
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        old_name = "user-didactics-" + "and-python-voice.md"
        offenders = []
        for raw in result.stdout.split(b"\0"):
            if not raw:
                continue
            relative = raw.decode("utf-8", errors="strict")
            path = REPO_ROOT / relative
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if old_name in text:
                offenders.append(relative)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
