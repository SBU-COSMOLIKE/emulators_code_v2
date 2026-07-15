#!/usr/bin/env python3
"""Regression tests for terminal mailbox prompts and their wording surfaces."""

import importlib.util
import pathlib
import unittest


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = AI_ROOT.parent
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"


def load_daemon():
    """Load the mailbox daemon without starting its command-line entry point.

    Returns:
      The imported mailbox-daemon module.
    """
    spec = importlib.util.spec_from_file_location(
        "mailbox_daemon_conditional_preamble", DAEMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConditionalPreambleTest(unittest.TestCase):
    """Pin ordinary, terminal, and documentation prompt contracts."""

    @classmethod
    def setUpClass(cls):
        """Load one read-only daemon module for the test class."""
        cls.daemon = load_daemon()

    def test_ordinary_prompt_still_requires_outbound(self):
        """An ordinary handoff retains the notes-first outbound rule."""
        message = "ARCHITECT_HANDOFF: implement the cited unit."
        prompt = self.daemon.PREAMBLE + message
        self.assertIn("Ordinary rule: end\nyour turn by", prompt)
        self.assertIn("writing your outbound handoff block", prompt)
        self.assertIn("Ambiguity follows the ordinary\nrule", prompt)
        self.assertNotIn("TERMINAL and no reply is owed\n" + message, prompt)

    def test_terminal_prompt_explicitly_requires_no_outbound(self):
        """A binding terminal/no-reply handoff is not made to echo."""
        message = (
            "BINDING: this thread is TERMINAL and no reply is owed.\n"
            "Do not acknowledge this receipt.")
        prompt = self.daemon.PREAMBLE + message
        self.assertIn("if and only if the inbound's binding instruction",
                      prompt)
        self.assertIn(
            "explicitly says the thread is TERMINAL and no reply is owed",
            prompt.replace("\n", " "))
        self.assertIn("write no\noutbound merely to satisfy this wrapper",
                      prompt)
        self.assertTrue(prompt.endswith(message))

    def test_full_prompt_has_no_second_unconditional_instruction(self):
        """The complete dispatch prompt contains one outbound imperative."""
        message = "ordinary body with no wrapper instructions"
        prompt = self.daemon.PREAMBLE + message
        self.assertEqual(prompt.count("writing your outbound handoff block"),
                         1)
        self.assertEqual(prompt.count("Ordinary rule:"), 1)
        self.assertEqual(prompt.count("Narrow exception:"), 1)
        self.assertEqual(prompt.count("Ambiguity follows the ordinary"), 1)

    def test_all_ruled_word_surfaces_are_conditional(self):
        """Every ruled wording surface states or cites the narrow exception."""
        daemon_source = DAEMON_PATH.read_text(encoding="utf-8")
        opus_role = (REPO_ROOT / ".claude" / "OPUS_ROLE.md").read_text(
            encoding="utf-8")
        memory = (AI_ROOT / "notes" / "MEMORY.md").read_text(
            encoding="utf-8")
        conventions = (AI_ROOT / "notes" /
                       "conventions-and-workflow.md").read_text(
                           encoding="utf-8")
        redteam_role = (REPO_ROOT / ".codex" / "REDTEAM_ROLE.md").read_text(
            encoding="utf-8")
        architect_role = (REPO_ROOT / ".claude" / "FABLE_ROLE.md").read_text(
            encoding="utf-8")

        self.assertIn("explicitly says TERMINAL and no reply is\nowed",
                      daemon_source)
        self.assertIn(
            "binding instruction explicitly says the thread is TERMINAL",
            opus_role.replace("\n", " "))
        self.assertIn("explicit binding TERMINAL/no-reply exception", memory)
        self.assertIn("only\noutbound exception", conventions)
        self.assertIn("normal Red Team turn that has a result",
                      " ".join(redteam_role.split()))
        self.assertIn("audited GO or NO-GO + delta", architect_role)
        self.assertIn(
            "When asked to review a commit or change, attack that named "
            "change", redteam_role.replace("\n", " "))
        self.assertIn('"Do a widespread search for ..."', redteam_role)


if __name__ == "__main__":
    unittest.main()
