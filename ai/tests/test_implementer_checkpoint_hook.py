"""Focused tests for the Implementer's 90-minute pause hook."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ai.tools import implementer_checkpoint_hook as checkpoint
from ai.tools import mailbox_daemon as daemon


HOOK_PATH = Path(checkpoint.__file__).resolve()


class ImplementerCheckpointHookTests(unittest.TestCase):
    """Check the deadline and one-shot result without replacing live code."""

    def test_work_before_the_deadline_continues_silently(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "checkpoint.state"
            result = checkpoint.checkpoint_result(
                event="PostToolBatch", now=89.0, deadline=90.0,
                state_path=state)

            self.assertEqual(result, (0, "", ""))
            self.assertFalse(state.exists())

    def test_post_tool_batch_requests_one_handoff_at_the_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "checkpoint.state"
            first = checkpoint.checkpoint_result(
                event="PostToolBatch", now=90.0, deadline=90.0,
                state_path=state)
            second = checkpoint.checkpoint_result(
                event="Stop", now=91.0, deadline=90.0,
                state_path=state)

            document = json.loads(first[1])
            specific = document["hookSpecificOutput"]
            self.assertEqual(first[0], 0)
            self.assertEqual(first[2], "")
            self.assertEqual(specific["hookEventName"], "PostToolBatch")
            self.assertEqual(
                specific["additionalContext"],
                checkpoint.CHECKPOINT_INSTRUCTION)
            self.assertEqual(second, (0, "", ""))
            self.assertEqual(state.read_bytes(), b"triggered\n")

    def test_stop_is_the_one_shot_fallback_when_no_batch_fires(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "checkpoint.state"
            first = checkpoint.checkpoint_result(
                event="Stop", now=100.0, deadline=90.0,
                state_path=state)
            second = checkpoint.checkpoint_result(
                event="Stop", now=101.0, deadline=90.0,
                state_path=state)

            document = json.loads(first[1])
            self.assertEqual(first[0], 0)
            self.assertEqual(first[2], "")
            self.assertEqual(document["decision"], "block")
            self.assertEqual(document["reason"],
                             checkpoint.CHECKPOINT_INSTRUCTION)
            self.assertEqual(second, (0, "", ""))
            self.assertEqual(state.read_bytes(), b"triggered\n")

    def test_a_revised_implementer_turn_gets_a_fresh_period(self):
        with tempfile.TemporaryDirectory() as folder:
            first_state = Path(folder) / "first.state"
            second_state = Path(folder) / "second.state"

            first = checkpoint.checkpoint_result(
                event="PostToolBatch", now=90.0, deadline=90.0,
                state_path=first_state)
            revised_early = checkpoint.checkpoint_result(
                event="PostToolBatch", now=100.0, deadline=190.0,
                state_path=second_state)
            revised_late = checkpoint.checkpoint_result(
                event="PostToolBatch", now=190.0, deadline=190.0,
                state_path=second_state)

            self.assertTrue(first[1])
            self.assertEqual(revised_early, (0, "", ""))
            self.assertTrue(revised_late[1])

    def test_atomic_state_allows_only_one_concurrent_instruction(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "checkpoint.state"
            environment = os.environ.copy()
            environment[checkpoint.DEADLINE_ENVIRONMENT] = "0"
            environment[checkpoint.STATE_ENVIRONMENT] = str(state)
            command = [sys.executable, str(HOOK_PATH)]
            processes = [
                subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, env=environment)
                for _ in range(4)
            ]
            payload = json.dumps({"hook_event_name": "PostToolBatch"})
            results = [process.communicate(payload) for process in processes]

            self.assertTrue(all(process.returncode == 0
                                for process in processes))
            outputs = [output for output, _ in results if output]
            errors = [error for _, error in results if error]
            self.assertEqual(len(outputs), 1)
            self.assertEqual(errors, [])
            self.assertEqual(
                json.loads(outputs[0])["hookSpecificOutput"]
                ["additionalContext"], checkpoint.CHECKPOINT_INSTRUCTION)
            self.assertEqual(state.read_bytes(), b"triggered\n")

    def test_subagent_event_cannot_claim_the_main_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "checkpoint.state"
            environment = os.environ.copy()
            environment[checkpoint.DEADLINE_ENVIRONMENT] = "0"
            environment[checkpoint.STATE_ENVIRONMENT] = str(state)
            payload = json.dumps({
                "hook_event_name": "PostToolBatch",
                "agent_id": "helper-1",
            })

            result = subprocess.run(
                [sys.executable, str(HOOK_PATH)], input=payload,
                capture_output=True, text=True, env=environment,
                check=False)

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")
            self.assertFalse(state.exists())

    def test_checkpoint_audit_omits_the_landing_instruction(self):
        cycle = "open-example@" + "a" * 40
        envelope = (
            "MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
            + "\nMAILBOX-MODE: normal\n\n")
        checkpoint_message = (
            envelope + daemon.IMPLEMENTER_CHECKPOINT_HEADING + "\n\n"
            + "- **Current state:** 90 minutes reached; work is paused and "
            + "may be stuck.\n")
        ordinary_message = envelope + "### IMPLEMENTER_HANDOFF: REVIEW\n"

        checkpoint_preamble = daemon.agent_preamble(
            agent="fable", message=checkpoint_message)
        ordinary_preamble = daemon.agent_preamble(
            agent="fable", message=ordinary_message)

        self.assertIn("90-MINUTE IMPLEMENTER CHECKPOINT",
                      checkpoint_preamble)
        self.assertNotIn(daemon.ARCHITECT_LANDING_PREAMBLE,
                         checkpoint_preamble)
        self.assertIn(daemon.ARCHITECT_LANDING_PREAMBLE, ordinary_preamble)
        self.assertIsNone(daemon.checkpoint_handoff_problem(
            checkpoint_message))
        self.assertIn("90-minute hook", daemon.checkpoint_handoff_problem(
            ordinary_message))

    def test_daemon_installs_only_the_two_checkpoint_hooks(self):
        settings = daemon.implementer_checkpoint_settings(
            python="/usr/bin/python3", hook_path="/repo/hook.py")

        self.assertEqual(set(settings["hooks"]), {"PostToolBatch", "Stop"})
        for event in settings["hooks"].values():
            [hook] = event[0]["hooks"]
            self.assertEqual(hook["command"], "/usr/bin/python3")
            self.assertEqual(hook["args"], ["/repo/hook.py"])

        self.assertGreater(daemon.DISPATCH_TIMEOUT_MINUTES,
                           daemon.IMPLEMENTER_REVIEW_MINUTES)


if __name__ == "__main__":
    unittest.main()
