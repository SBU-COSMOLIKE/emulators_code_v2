"""Keep provider choice separate from the stable Implementer role."""

import unittest
from unittest import mock

from ai.tests.tools_mailbox_daemon_fix_only_repro import run_main
from ai.tests.tools_mailbox_daemon_fix_only_repro import scratch_daemon
from ai.tests.tools_mailbox_daemon_ticket_cycle_repro import BASE_B
from ai.tests.tools_mailbox_daemon_ticket_cycle_repro import flow_payload
from ai.tests.tools_mailbox_daemon_ticket_cycle_repro import scratch_daemon as cycle_daemon


class OllamaImplementerRuntimeTests(unittest.TestCase):
    """Prove that Ollama changes execution, not Implementer authority."""

    def test_ollama_model_is_explicit_before_live_setup(self):
        with scratch_daemon() as (daemon, _root, _mailbox, _relay):
            rc, output, error = run_main(
                daemon, ["--watch", "--implementer-provider", "ollama"])

        self.assertEqual(rc, 1, output + error)
        self.assertIn("requires an explicit --implementer-model", output)

    def test_runtime_keeps_the_opus_address(self):
        with scratch_daemon() as (daemon, _root, _mailbox, _relay):
            runtime = daemon.implementer_runtime_record(
                provider="ollama", model="qwen3.5",
                context_limit=65536, compaction_limit=64000)

        self.assertEqual(runtime["role_address"], "opus")
        self.assertEqual(runtime["provider"], "ollama")
        self.assertEqual(runtime["model"], "qwen3.5")

    def test_live_once_runs_ollama_preflight_before_dispatch(self):
        with scratch_daemon() as (daemon, _root, _mailbox, _relay):
            runtime = daemon.implementer_runtime_record(
                provider="ollama", model="qwen3.5",
                context_limit=65536, compaction_limit=64000)
            check = mock.Mock(return_value=runtime)
            daemon.verified_implementer_runtime = check

            rc, output, error = run_main(daemon, [
                "--once", "--implementer-provider", "ollama",
                "--implementer-model", "qwen3.5",
                "--claude-context", "64000"])

        self.assertEqual(rc, 0, output + error)
        check.assert_called_once_with(
            provider="ollama", model="qwen3.5", compaction_limit=64000,
            dry_run=False)

    def test_cycle_saves_runtime_and_refuses_silent_replacement(self):
        with cycle_daemon() as (daemon, _mailbox):
            ollama = daemon.implementer_runtime_record(
                provider="ollama", model="qwen3.5",
                context_limit=65536, compaction_limit=64000)
            daemon.IMPLEMENTER_RUNTIME = ollama
            cycle = "route-spoof@" + BASE_B
            message = flow_payload(cycle, "Implementer start")
            daemon.register_ticket_cycle_message("opus", message)

            saved = daemon.read_ticket_cycle_state()["active"][cycle]
            self.assertEqual(saved["implementer_runtime"], ollama)

            daemon.IMPLEMENTER_RUNTIME = daemon.implementer_runtime_record(
                provider="claude", model="claude-opus-4-8",
                context_limit=500000, compaction_limit=500000)
            with self.assertRaisesRegex(
                    daemon.TicketCycleStateError, "active ticket is bound"):
                daemon.register_ticket_cycle_message("fable", message)

    def test_ollama_failure_guidance_never_suggests_claude_login(self):
        with scratch_daemon() as (daemon, _root, _mailbox, _relay):
            daemon.IMPLEMENTER_RUNTIME = daemon.implementer_runtime_record(
                provider="ollama", model="glm-5.2:cloud",
                context_limit=131072, compaction_limit=64000)
            output = daemon.provider_failure_guidance(
                "opus", ["Error: failed to pull model manifest"])

        self.assertIn("ollama pull glm-5.2:cloud", output)
        self.assertNotIn("type /login", output)

    def test_context_limit_and_compaction_are_independent(self):
        with scratch_daemon() as (daemon, _root, _mailbox, _relay):
            with self.assertRaisesRegex(ValueError, "compaction"):
                daemon.implementer_runtime_record(
                    provider="ollama", model="qwen3.5",
                    context_limit=32768, compaction_limit=64000)

    def test_both_providers_keep_the_same_route_and_checkpoint(self):
        with scratch_daemon() as (daemon, _root, _mailbox, _relay):
            claude = daemon.build_agent_commands(
                "high", "max", "high", 64000,
                implementer_provider="claude",
                implementer_model="sonnet")
            ollama = daemon.build_agent_commands(
                "high", "max", "high", 64000,
                implementer_provider="ollama",
                implementer_model="qwen3.5")
            checkpoint = daemon.implementer_checkpoint_settings(
                python="python3", hook_path="/trusted/checkpoint.py")

        self.assertEqual(set(claude), {"fable", "opus", "sol"})
        self.assertEqual(set(ollama), {"fable", "opus", "sol"})
        self.assertIn("--no-session-persistence", claude["opus"])
        self.assertIn("--no-session-persistence", ollama["opus"])
        self.assertEqual(
            checkpoint["hooks"]["PreCompact"][0]["hooks"][0]["args"],
            ["/trusted/checkpoint.py"])


if __name__ == "__main__":
    unittest.main()
