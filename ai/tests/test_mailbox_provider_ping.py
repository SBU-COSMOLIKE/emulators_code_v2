"""CPU tests for the direct Claude, Ollama, and Sol connection check.

The real command spends one very small model request per provider.  These
tests replace both command-line programs with controlled subprocess results,
so the test suite never contacts either service or consumes account credits.
"""

import contextlib
import io
import pathlib
import re
import subprocess
import unittest
from unittest import mock

from ai.tests.tools_mailbox_daemon_fix_only_repro import run_main
from ai.tests.tools_mailbox_daemon_fix_only_repro import scratch_daemon
from ai.tests.tools_mailbox_daemon_fix_only_repro import tree_snapshot


FIXED_NONCE = "0123456789abcdef"


def _command_marker(command):
    """Read the exact nonce-bearing answer requested by a fake command."""
    prompt = command[-1]
    matches = re.findall(
        r"[A-Za-z0-9_.:-]*" + re.escape(FIXED_NONCE)
        + r"[A-Za-z0-9_.:-]*",
        prompt)
    if not matches:
        raise AssertionError("provider prompt does not contain its nonce")
    return max(matches, key=len)


def _sol_answer_path(command):
    """Return the temporary final-answer path supplied to the Codex CLI."""
    index = command.index("--output-last-message")
    return pathlib.Path(command[index + 1])


def _successful_run(daemon, calls):
    """Return a subprocess replacement that answers both provider pings."""
    def run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        if command[0] == "git":
            return subprocess.CompletedProcess(
                command, 0, stdout=b"", stderr=b"")
        if (command[0] == daemon.OLLAMA_EXECUTABLE
                and command[1:2] == ["show"]):
            return subprocess.CompletedProcess(
                command, 0, stdout=b"context length 600000\n", stderr=b"")
        marker = _command_marker(command)
        if command[0] == daemon.CLAUDE_EXECUTABLE:
            return subprocess.CompletedProcess(
                command, 0, stdout=(marker + "\n").encode("utf-8"),
                stderr=b"")
        if command[0] == daemon.OLLAMA_EXECUTABLE:
            return subprocess.CompletedProcess(
                command, 0, stdout=(marker + "\n").encode("utf-8"),
                stderr=b"")
        if command[0] == daemon.CODEX_EXECUTABLE:
            _sol_answer_path(command).write_text(
                marker + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(
                command, 0, stdout=b"Sol completed.\n", stderr=b"")
        raise AssertionError("unexpected provider executable " + command[0])

    return run


def _call_ping(daemon, run, *, include_sol, implementer_provider="claude"):
    """Run the production helper with fixed randomness and captured output."""
    output = io.StringIO()
    with mock.patch.object(
            daemon.secrets, "token_hex", return_value=FIXED_NONCE), \
            mock.patch.object(daemon.subprocess, "run", side_effect=run), \
            mock.patch.object(
                daemon._PROVIDER_HEALTH.shutil, "which",
                return_value="/test/bin/ollama"), \
            contextlib.redirect_stdout(output):
        outcome = daemon.check_provider_connectivity(
            architect_model="claude-test-model",
            implementer_provider=implementer_provider,
            implementer_model="qwen-test-model",
            include_sol=include_sol)
    return outcome, output.getvalue()


class MailboxProviderPingTests(unittest.TestCase):
    """Prove that ping checks providers directly without creating work."""

    def test_both_providers_must_return_the_exact_nonce(self):
        with scratch_daemon() as (daemon, root, _, _):
            calls = []
            before = tree_snapshot(root)
            outcome, output = _call_ping(
                daemon, _successful_run(daemon, calls), include_sol=True)

            self.assertTrue(outcome, output)
            self.assertEqual(tree_snapshot(root), before)
            self.assertEqual(len(calls), 3)
            claude, implementer, sol = calls
            self.assertEqual(claude[0][0], daemon.CLAUDE_EXECUTABLE)
            self.assertEqual(implementer[0][0], daemon.CLAUDE_EXECUTABLE)
            self.assertIn("qwen-test-model", implementer[0])
            self.assertEqual(sol[0][0], daemon.CODEX_EXECUTABLE)
            self.assertIn("--output-format", claude[0])
            self.assertIn("text", claude[0])
            self.assertIn("--output-last-message", sol[0])
            self.assertIn(daemon.SOL_MODEL, sol[0])
            self.assertEqual(
                claude[1]["timeout"], daemon.PROVIDER_PING_TIMEOUT_SECONDS)
            self.assertEqual(
                sol[1]["timeout"], daemon.PROVIDER_PING_TIMEOUT_SECONDS)

    def test_skip_redteam_never_starts_sol(self):
        with scratch_daemon() as (daemon, _, _, _):
            calls = []
            outcome, output = _call_ping(
                daemon, _successful_run(daemon, calls), include_sol=False)

            self.assertTrue(outcome, output)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0][0], daemon.CLAUDE_EXECUTABLE)
            self.assertEqual(calls[1][0][0], daemon.CLAUDE_EXECUTABLE)
            self.assertTrue(all(
                daemon.CODEX_EXECUTABLE not in call[0] for call in calls))

    def test_identical_claude_model_pair_is_checked_once(self):
        with scratch_daemon() as (daemon, _, _, _):
            calls = []
            output = io.StringIO()
            with mock.patch.object(
                    daemon.secrets, "token_hex", return_value=FIXED_NONCE), \
                    mock.patch.object(
                        daemon.subprocess, "run",
                        side_effect=_successful_run(daemon, calls)), \
                    contextlib.redirect_stdout(output):
                outcome = daemon.check_provider_connectivity(
                    architect_model="same-model",
                    implementer_provider="claude",
                    implementer_model="same-model", include_sol=False)

            self.assertTrue(outcome, output.getvalue())
            self.assertEqual(len(calls), 1)

    def test_ollama_implementer_is_checked_independently(self):
        with scratch_daemon() as (daemon, _, _, _):
            calls = []
            outcome, output = _call_ping(
                daemon, _successful_run(daemon, calls),
                include_sol=False, implementer_provider="ollama")

            self.assertTrue(outcome, output)
            self.assertEqual(
                [call[0][0] for call in calls],
                [daemon.CLAUDE_EXECUTABLE, "git",
                 daemon.OLLAMA_EXECUTABLE, daemon.OLLAMA_EXECUTABLE])
            self.assertEqual(
                calls[2][0],
                [daemon.OLLAMA_EXECUTABLE, "show", "qwen-test-model",
                 "--verbose"])
            self.assertEqual(
                calls[3][0][:5],
                [daemon.OLLAMA_EXECUTABLE, "launch", "claude", "--model",
                 "qwen-test-model"])
            self.assertIn("--no-session-persistence", calls[3][0])
            self.assertIn("IMPLEMENTER ROLE", calls[3][0][-1])
            self.assertNotEqual(calls[3][1]["cwd"], str(pathlib.Path.cwd()))
            self.assertEqual(calls[1][1]["cwd"], calls[3][1]["cwd"])

    def test_small_ollama_context_stops_before_integration_launch(self):
        with scratch_daemon() as (daemon, _, _, _):
            calls = []
            success = _successful_run(daemon, calls)

            def small_context(command, **kwargs):
                if (command[0] == daemon.OLLAMA_EXECUTABLE
                        and command[1:2] == ["show"]):
                    calls.append((list(command), dict(kwargs)))
                    return subprocess.CompletedProcess(
                        command, 0, stdout=b"context length 16000\n",
                        stderr=b"")
                return success(command, **kwargs)

            outcome, output = _call_ping(
                daemon, small_context, include_sol=False,
                implementer_provider="ollama")

            self.assertFalse(outcome)
            self.assertIn("below the required minimum", output)
            self.assertFalse(any(
                call[0][0] == daemon.OLLAMA_EXECUTABLE
                and call[0][1:3] == ["launch", "claude"]
                for call in calls))

    def test_ollama_build_keeps_the_architect_on_claude(self):
        with scratch_daemon() as (daemon, _, _, _):
            commands = daemon.build_agent_commands(
                "high", "max", "xhigh", 64000,
                architect_model="opus",
                implementer_model="qwen3.5",
                implementer_provider="ollama")

            self.assertEqual(commands["fable"][0], daemon.CLAUDE_EXECUTABLE)
            self.assertEqual(
                commands["opus"][:7],
                [daemon.OLLAMA_EXECUTABLE, "launch", "claude", "--model",
                 "qwen3.5", "--yes", "--"])
            self.assertIn("-p", commands["opus"])
            self.assertNotIn("--effort", commands["opus"])
            self.assertEqual(commands["sol"][0], daemon.CODEX_EXECUTABLE)

    def test_unknown_implementer_provider_is_refused(self):
        with scratch_daemon() as (daemon, _, _, _):
            with self.assertRaisesRegex(ValueError, "provider"):
                daemon.build_agent_commands(
                    "high", "max", "xhigh", 64000,
                    implementer_provider="unknown")

    def test_claude_failure_does_not_hide_sol_status(self):
        with scratch_daemon() as (daemon, _, _, _):
            calls = []
            success = _successful_run(daemon, calls)

            def claude_fails(command, **kwargs):
                if command[0] == daemon.CLAUDE_EXECUTABLE:
                    calls.append((list(command), dict(kwargs)))
                    return subprocess.CompletedProcess(
                        command, 1, stdout=b"", stderr=b"login required")
                return success(command, **kwargs)

            outcome, output = _call_ping(
                daemon, claude_fails, include_sol=True)

            self.assertFalse(outcome)
            self.assertEqual(
                [call[0][0] for call in calls],
                [daemon.CLAUDE_EXECUTABLE, daemon.CLAUDE_EXECUTABLE,
                 daemon.CODEX_EXECUTABLE])
            self.assertIn("Claude", output)
            self.assertIn("Sol", output)

    def test_timeout_and_missing_executable_are_clean_failures(self):
        cases = ("timeout", "missing")
        for failure in cases:
            with self.subTest(failure=failure), \
                    scratch_daemon() as (daemon, _, _, _):
                calls = []
                success = _successful_run(daemon, calls)

                def one_provider_fails(command, **kwargs):
                    if (failure == "timeout"
                            and command[0] == daemon.CLAUDE_EXECUTABLE):
                        calls.append((list(command), dict(kwargs)))
                        raise subprocess.TimeoutExpired(
                            command, kwargs["timeout"])
                    if (failure == "missing"
                            and command[0] == daemon.CODEX_EXECUTABLE):
                        calls.append((list(command), dict(kwargs)))
                        raise FileNotFoundError(
                            2, "No such file or directory", command[0])
                    return success(command, **kwargs)

                outcome, _ = _call_ping(
                    daemon, one_provider_fails, include_sol=True)

                self.assertFalse(outcome)
                self.assertEqual(len(calls), 3)

    def test_echoed_prompt_cannot_masquerade_as_a_model_answer(self):
        for provider in ("claude", "sol"):
            with self.subTest(provider=provider), \
                    scratch_daemon() as (daemon, _, _, _):
                calls = []
                success = _successful_run(daemon, calls)

                def spoofed(command, **kwargs):
                    if (provider == "claude"
                            and command[0] == daemon.CLAUDE_EXECUTABLE):
                        calls.append((list(command), dict(kwargs)))
                        return subprocess.CompletedProcess(
                            command, 0, stdout=command[-1].encode("utf-8"),
                            stderr=b"")
                    if (provider == "sol"
                            and command[0] == daemon.CODEX_EXECUTABLE):
                        calls.append((list(command), dict(kwargs)))
                        # The nonce appears in stdout only.  A valid Sol ping
                        # must read the separate final-answer file.
                        return subprocess.CompletedProcess(
                            command, 0,
                            stdout=command[-1].encode("utf-8"), stderr=b"")
                    return success(command, **kwargs)

                outcome, _ = _call_ping(
                    daemon, spoofed, include_sol=True)

                self.assertFalse(outcome)
                self.assertEqual(len(calls), 3)

    def test_cli_ping_and_skip_redteam_select_the_expected_providers(self):
        for arguments, include_sol in (
                (["--ping"], True),
                (["--ping", "--skip-redteam"], False),
                (["--ping", "--no-red-team"], False)):
            with self.subTest(arguments=arguments), \
                    scratch_daemon() as (daemon, root, _, _):
                before = tree_snapshot(root)
                check = mock.Mock(return_value=True)
                daemon.check_provider_connectivity = check

                rc, output, error = run_main(daemon, arguments)

                self.assertEqual(rc, 0, output + error)
                self.assertEqual(tree_snapshot(root), before)
                self.assertEqual(check.call_count, 1)
                self.assertEqual(
                    check.call_args.kwargs["architect_model"],
                    daemon.DEFAULT_ARCHITECT_MODEL)
                self.assertEqual(
                    check.call_args.kwargs["implementer_provider"],
                    daemon.DEFAULT_IMPLEMENTER_PROVIDER)
                self.assertEqual(
                    check.call_args.kwargs["implementer_model"],
                    daemon.DEFAULT_IMPLEMENTER_MODEL)
                self.assertEqual(
                    check.call_args.kwargs["include_sol"], include_sol)
                self.assertEqual(
                    check.call_args.kwargs["implementer_compaction_limit"],
                    daemon.DEFAULT_CLAUDE_CONTEXT_BUDGET)

    def test_ping_conflicts_refuse_before_any_provider_starts(self):
        conflicts = (
            ["--ping", "--once"],
            ["--ping", "--watch"],
            ["--ping", "--clean-all"],
            ["--ping", "--send", "architect", "--unit", "one request"],
            ["--ping", "--max", "1"],
            ["--ping", "--severity", "high"],
            ["--ping", "--fix-only", "true"],
            ["--ping", "--cycle", "1"],
            ["--ping", "architect"],
        )
        for arguments in conflicts:
            with self.subTest(arguments=arguments), \
                    scratch_daemon() as (daemon, root, _, _):
                before = tree_snapshot(root)
                check = mock.Mock(return_value=True)
                daemon.check_provider_connectivity = check

                rc, _, _ = run_main(daemon, arguments)

                self.assertNotEqual(rc, 0)
                self.assertEqual(check.call_count, 0)
                self.assertEqual(tree_snapshot(root), before)


if __name__ == "__main__":
    unittest.main()
