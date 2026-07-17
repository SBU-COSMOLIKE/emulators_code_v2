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
    r"(?<![~:])\b(?:19\d{2}|20[0-3]\d)\b|"
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
        self.assertIn("creates or updates a local backlog ticket in the same turn",
                      memory)
        self.assertIn("anti-AI requirements", memory)

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
                    "Request identity",
                    "Generation identity",
                    "Staged-selection identity",
                    "Artifact identity"):
                with self.subTest(note=text[:40], identity=identity):
                    self.assertIn(identity, text)
        self.assertIn("<data-vector-root>_ell.npy", cmb)
        self.assertIn("ai/tests/test_cmb_checkpoint_axis.py", cmb)
        self.assertIn("Dense covariance training is unsupported", cmb)
        self.assertIn(
            "target-producing physics implementations belong to request "
            "identity",
            generation)
        self.assertIn(
            "current sealed generation does not save enough changing "
            "random-engine, sampler, walker, log-probability, and "
            "unique-row-selection state to continue the sequence exactly",
            generation)
        self.assertIn(
            "Append therefore authenticates the active generation and then "
            "refuses without drawing or publishing a row",
            generation)
        self.assertIn("output decoder and loss composition", artifact)
        self.assertIn("When `TCMB` is a sampled input", cmb)
        self.assertIn("both fixed-temperature and sampled-temperature", cmb)
        self.assertIn("validated ten-template dataset", models)
        self.assertIn("registry construction alone is not a claim", models)

    def test_notes_have_no_calendar_or_person_specific_language(self):
        for relative in PERMANENT_NOTES:
            text = read(relative)
            with self.subTest(note=relative, kind="calendar"):
                self.assertIsNone(CALENDAR_RE.search(text))
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
