#!/usr/bin/env python3
"""Check whether the AI services selected for a watch can answer.

This module knows how to make one small connection request to Claude,
Ollama, and Codex.  It does not read the mailbox, change a worktree, or make
workflow decisions.  ``mailbox_daemon.py`` supplies the selected programs and
models and uses the Boolean result.
"""

import os
import subprocess
import tempfile


def _prompt(marker):
    """Ask one provider for an exact reply without assigning work."""
    return ("This is a connection test. Do not use tools, read files, or "
            "explain. Reply with exactly this one line:\n" + marker)


def _answered(command, marker, directory, timeout, run, response_path=None):
    """Return whether one provider produced the requested exact reply.

    ``command`` is the complete provider command. ``run`` starts it inside
    the empty temporary ``directory`` and must finish within ``timeout``.
    Claude and Ollama answer on standard output. Codex instead writes the
    final model message to ``response_path``.
    """
    try:
        result = run(
            command, cwd=directory, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=timeout)
        if result.returncode != 0:
            return False
        if response_path is None:
            answer = result.stdout.decode("utf-8").strip()
        else:
            with open(response_path, encoding="utf-8") as stream:
                answer = stream.read().strip()
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        return False
    return answer == marker


def check_connectivity(
        *, architect_model, implementer_provider, implementer_model,
        include_sol, dry_run, nonce, claude_executable, ollama_executable,
        codex_executable, sol_model, timeout, run, output=print):
    """Check the distinct AI services selected for one mailbox watch.

    Returns ``True`` only when every selected service returns its private
    marker.  All program names and model choices come from the daemon, which
    keeps this helper independent of repository policy.

    The ``architect_model`` and Implementer pair name the requested services.
    ``include_sol`` adds the Red Team check. ``dry_run`` prints the selection
    without starting a provider. The daemon supplies the executables, Sol
    model, timeout, private ``nonce``, subprocess function, and output
    function so tests can exercise the same code without using AI credits.
    """
    if implementer_provider not in {"claude", "ollama"}:
        raise ValueError("Implementer provider must be claude or ollama")
    if dry_run:
        output("[dry-run] would check Claude Architect model "
               + architect_model + ".")
        if implementer_provider == "ollama":
            output("[dry-run] would check Ollama Implementer model "
                   + implementer_model + ".")
        if include_sol:
            output("[dry-run] would check Sol model " + sol_model + ".")
        else:
            output("Sol: skipped by --skip-redteam.")
        return True

    with tempfile.TemporaryDirectory(prefix="cocoa-flow-ping-") as directory:
        claude_marker = "COCOA-FLOW-PONG-CLAUDE-" + nonce
        claude_command = [
            claude_executable, "-p", "--model", architect_model,
            "--effort", "low", "--permission-mode", "plan",
            "--tools", "", "--safe-mode", "--no-session-persistence",
            "--output-format", "text", _prompt(claude_marker),
        ]
        claude_ok = _answered(
            claude_command, claude_marker, directory, timeout, run)
        output("Claude Architect: "
               + ("online and answered the connection test."
                  if claude_ok else "unavailable."))

        ollama_ok = True
        if implementer_provider == "ollama":
            ollama_marker = "COCOA-FLOW-PONG-OLLAMA-" + nonce
            ollama_command = [
                ollama_executable, "run", implementer_model,
                "--hidethinking", _prompt(ollama_marker)]
            ollama_ok = _answered(
                ollama_command, ollama_marker, directory, timeout, run)
            output("Ollama Implementer: "
                   + ("online and answered the connection test."
                      if ollama_ok else "unavailable."))

        sol_ok = True
        if include_sol:
            sol_marker = "COCOA-FLOW-PONG-SOL-" + nonce
            sol_output = os.path.join(directory, "sol-response.txt")
            sol_command = [
                codex_executable, "exec", "--model", sol_model,
                "-c", "model_reasoning_effort=none",
                "-c", "service_tier=standard", "--sandbox", "read-only",
                "--cd", directory, "--skip-git-repo-check", "--ephemeral",
                "--ignore-rules", "--ignore-user-config",
                "--output-last-message", sol_output,
                _prompt(sol_marker),
            ]
            sol_ok = _answered(
                sol_command, sol_marker, directory, timeout, run, sol_output)
            output("Sol: " + ("online and answered the connection test."
                              if sol_ok else "unavailable."))
        else:
            output("Sol: skipped by --skip-redteam.")

    successful = claude_ok and ollama_ok and (not include_sol or sol_ok)
    if successful:
        services = ["Claude"]
        if implementer_provider == "ollama":
            services.append("Ollama")
        if include_sol:
            services.append("Sol")
        if len(services) == 1:
            checked = services[0]
        elif len(services) == 2:
            checked = " and ".join(services)
        else:
            checked = ", ".join(services[:-1]) + ", and " + services[-1]
        output("connection check passed: " + checked + " responded.")
        return True

    output("connection check failed; check login and service availability.")
    if not claude_ok:
        output("Claude login: " + claude_executable + " auth status")
    if not ollama_ok:
        output("Ollama: start the Ollama service and run `ollama pull "
               + implementer_model + "`.")
    if include_sol and not sol_ok:
        output("Sol login: " + codex_executable + " login status")
    return False
