#!/usr/bin/env python3
"""Check whether the AI services selected for a watch can answer.

This module knows how to make one small connection request to Claude,
Ollama, and Codex.  It does not read the mailbox, change a worktree, or make
workflow decisions.  ``mailbox_daemon.py`` supplies the selected programs and
models and uses the Boolean result.
"""

import os
import re
import shutil
import subprocess
import tempfile


CONTEXT_LINE = re.compile(
    r"\bcontext(?:\s+|_)length\b\s*:?\s*([0-9][0-9,]*)",
    re.IGNORECASE)


def _prompt(marker):
    """Ask one provider for an exact reply without assigning work."""
    return ("This is a connection test. Do not use tools, read files, or "
            "explain. Reply with exactly this one line:\n" + marker)


def _answered(command, marker, directory, timeout, run, response_path=None):
    """Return whether one provider produced the requested exact reply.

    ``command`` is the complete provider command. ``run`` starts it inside
    the disposable ``directory`` and must finish within ``timeout``.
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


def _run_result(command, directory, timeout, run):
    """Run one local provider inspection command without raising."""
    try:
        return run(
            command, cwd=directory, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _ollama_context(
        *, model, directory, timeout, run, ollama_executable):
    """Return ``(context, problem)`` from one local Ollama inspection."""
    if shutil.which(ollama_executable) is None:
        return None, "executable-missing"
    result = _run_result(
        [ollama_executable, "show", model, "--verbose"],
        directory, timeout, run)
    if result is None:
        return None, "temporary-provider"
    try:
        text = (result.stdout + b"\n" + result.stderr).decode(
            "utf-8", errors="strict")
    except UnicodeError:
        return None, "context-unverified"
    except AttributeError:
        return None, "context-unverified"
    if result.returncode == 0:
        match = CONTEXT_LINE.search(text)
        if match is None:
            return None, "context-unverified"
        return int(match.group(1).replace(",", "")), None
    folded = text.casefold()
    if any(mark in folded for mark in (
            "connection refused", "could not connect", "dial tcp",
            "ollama is not running")):
        return None, "service-unavailable"
    if any(mark in folded for mark in (
            "pull model manifest", "model not found", "does not exist")):
        return None, "model-unavailable"
    return None, "temporary-provider"


def _temporary_git_repository(directory, timeout, run):
    """Create the disposable Git checkout required by Claude Code."""
    result = _run_result(
        ["git", "init", "--quiet", directory], directory, timeout, run)
    return result is not None and result.returncode == 0


def check_ollama_implementer(
        *, model, compaction_limit, minimum_context, preamble, nonce,
        ollama_executable, timeout, run, output=print):
    """Exercise the real Ollama-backed Implementer launch and return context.

    Returns ``(verified_context, None)`` on success or ``(None, reason)`` on
    failure.  The disposable repository proves the integration without
    allowing the connection test to inspect or edit the user's checkout.
    """
    with tempfile.TemporaryDirectory(
            prefix="cocoa-flow-ollama-preflight-") as directory:
        if not _temporary_git_repository(directory, timeout, run):
            output("Ollama Implementer: temporary Git preflight failed.")
            return None, "temporary-git"
        context_limit, problem = _ollama_context(
            model=model, directory=directory, timeout=timeout, run=run,
            ollama_executable=ollama_executable)
        if context_limit is None:
            if problem == "executable-missing":
                output("Ollama Implementer: the `ollama` executable was not "
                       "found.")
            elif problem == "service-unavailable":
                output("Ollama Implementer: the Ollama service is "
                       "unavailable.")
            elif problem == "model-unavailable":
                output("Ollama Implementer: model " + model
                       + " is unavailable.")
            else:
                output("Ollama Implementer: model context could not be "
                       "verified with `ollama show " + model
                       + " --verbose`.")
            return None, problem
        if context_limit < minimum_context:
            output("Ollama Implementer: model context "
                   + str(context_limit) + " is below the required minimum "
                   + str(minimum_context) + ".")
            return None, "context-too-small"
        if compaction_limit > context_limit:
            output("Ollama Implementer: Claude Code compaction threshold "
                   + str(compaction_limit) + " exceeds the model context "
                   + str(context_limit) + ".")
            return None, "compaction-too-high"

        marker = "COCOA-FLOW-PONG-OLLAMA-" + nonce
        command = [
            ollama_executable, "launch", "claude", "--model", model,
            "--yes", "--", "-p", "--no-session-persistence",
            "--permission-mode", "acceptEdits", "--",
            preamble + "\n\n" + _prompt(marker)]
        if not _answered(command, marker, directory, timeout, run):
            output("Ollama Implementer: `ollama launch claude` failed.")
            return None, "integration-launch"
        output("Ollama Implementer: online; Claude Code integration answered "
               "the connection test; model context is "
               + str(context_limit) + ".")
        return context_limit, None


def check_connectivity(
        *, architect_model, implementer_provider, implementer_model,
        include_sol, dry_run, nonce, claude_executable, ollama_executable,
        codex_executable, sol_model, timeout, run,
        implementer_compaction_limit, ollama_minimum_context,
        implementer_preamble, output=print):
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
        if (implementer_provider == "claude"
                and implementer_model != architect_model):
            output("[dry-run] would separately check Claude Implementer "
                   "model " + implementer_model + ".")
        elif implementer_provider == "ollama":
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

        claude_implementer_ok = True
        if (implementer_provider == "claude"
                and implementer_model != architect_model):
            implementer_marker = (
                "COCOA-FLOW-PONG-CLAUDE-IMPLEMENTER-" + nonce)
            implementer_command = [
                claude_executable, "-p", "--model", implementer_model,
                "--effort", "low", "--permission-mode", "plan",
                "--tools", "", "--safe-mode", "--no-session-persistence",
                "--output-format", "text", _prompt(implementer_marker),
            ]
            claude_implementer_ok = _answered(
                implementer_command, implementer_marker, directory,
                timeout, run)
            output("Claude Implementer: "
                   + ("online and answered the connection test."
                      if claude_implementer_ok else "unavailable."))

        ollama_ok = True
        if implementer_provider == "ollama":
            context_limit, ollama_problem = check_ollama_implementer(
                model=implementer_model,
                compaction_limit=implementer_compaction_limit,
                minimum_context=ollama_minimum_context,
                preamble=implementer_preamble, nonce=nonce,
                ollama_executable=ollama_executable, timeout=timeout,
                run=run, output=output)
            ollama_ok = context_limit is not None
        else:
            ollama_problem = None

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

    successful = (claude_ok and claude_implementer_ok and ollama_ok
                  and (not include_sol or sol_ok))
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
    if not claude_implementer_ok:
        output("Claude Implementer model is unavailable: "
               + implementer_model + ".")
    if not ollama_ok:
        if ollama_problem in {"context-unverified", "temporary-provider"}:
            output("Ollama: start the service and confirm the model with "
                   "`ollama show " + implementer_model + " --verbose`.")
        elif ollama_problem == "executable-missing":
            output("Ollama: install the `ollama` command, then retry.")
        elif ollama_problem == "service-unavailable":
            output("Ollama: start the service, then retry.")
        elif ollama_problem == "model-unavailable":
            output("Ollama: run `ollama pull " + implementer_model
                   + "`, then retry.")
        elif ollama_problem in {"context-too-small", "compaction-too-high"}:
            output("Ollama: choose a model or compaction threshold that "
                   "satisfies the reported context requirement.")
        elif ollama_problem == "integration-launch":
            output("Ollama: verify `ollama launch claude --model "
                   + implementer_model + "`.")
        else:
            output("Ollama: start the service and run `ollama pull "
                   + implementer_model + "`.")
    if include_sol and not sol_ok:
        output("Sol login: " + codex_executable + " login status")
    return False
