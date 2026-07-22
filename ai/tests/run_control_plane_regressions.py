#!/usr/bin/env python3
"""Run every required AI control-plane regression with one command.

This folder holds two kinds of workflow checks: ordinary ``test_*.py``
modules that ``unittest`` can discover, and stand-alone ``*_repro.py``
programs that rebuild mailbox, Git, archive, and process failures. The
discover command never runs the second kind, so "the AI tests passed"
could silently omit them. This program runs both kinds from one explicit
manifest and returns one honest verdict.

Before anything runs, the manifest itself is checked: every listed file
must exist exactly once and have a row in this folder's README table, and
every ``*_repro.py`` file on disk must be listed. A manifest problem stops
the run with exit code 2 before any check starts. There is no skip
mechanism: a check that cannot run is a failure, never a silent omission.

Each command is printed before it runs and executes in its own child
process from the repository root, so one reproduction's module state and
temporary Git repositories cannot leak into the next. The terminal shows
one verdict line per command; complete child output goes to one log file
whose path is printed first. ``--debug`` additionally streams every
child's complete output to the terminal.

Exit codes:

* 0: every required check ran and passed (the complete pass);
* 1: at least one required check failed;
* 2: the manifest or the run setup is unsafe, and no verdict exists.
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
README_PATH = os.path.join(TESTS_DIR, "README.md")
CHILD_TIMEOUT_SECONDS = 1800

# Discoverable modules that exercise the AI workflow controller: the
# daemon parts, the guards, the contracts, and their recovery paths.
# Scientific emulator tests stay out so this command needs only a CPU
# and the Git program.
CONTROL_PLANE_TEST_MODULES = (
    "test_backlog_guard",
    "test_context_handoff",
    "test_handoff_contract",
    "test_implementer_authority_snapshot",
    "test_implementer_checkpoint_hook",
    "test_mailbox_candidate_delivery_recovery",
    "test_mailbox_candidate_state_recovery",
    "test_mailbox_clean_all",
    "test_mailbox_conditional_preamble",
    "test_mailbox_daemon_architect_entrypoint",
    "test_mailbox_daemon_interrupts",
    "test_mailbox_daemon_severity",
    "test_mailbox_primary_backlog_bridge",
    "test_mailbox_provider_ping",
    "test_mailbox_role_restart",
    "test_ollama_implementer_runtime",
    "test_permanent_note_guard",
    "test_permanent_note_style_contract",
    "test_protected_control_plane_shadow",
    "test_protected_control_plane_stale_integration",
    "test_protected_control_plane_ticket",
    "test_protected_policy_review",
    "test_relay_log_reservation",
    "test_reopen_transition",
    "test_review_dispatch",
    "test_role_contract",
    "test_role_directive_contract",
    "test_role_workflow_behavior",
    "test_ticket_change_guard",
    "test_tracked_backlog_landing",
)

# Stand-alone reproduction programs. Every *_repro.py file in this folder
# must appear here; the manifest check below enforces that completeness.
CONTROL_PLANE_REPRODUCTIONS = (
    "finite_contract_cuda_wording_repro.py",
    "tools_backlog_bundle_repro.py",
    "tools_handoff_router_repro.py",
    "tools_mailbox_daemon_dead_mailbox_repro.py",
    "tools_mailbox_daemon_fix_only_repro.py",
    "tools_mailbox_daemon_landing_debt_repro.py",
    "tools_mailbox_daemon_max_repro.py",
    "tools_mailbox_daemon_no_redteam_repro.py",
    "tools_mailbox_daemon_output_style_repro.py",
    "tools_mailbox_daemon_primary_worktree_repro.py",
    "tools_mailbox_daemon_redteam_repro.py",
    "tools_mailbox_daemon_rendezvous_repro.py",
    "tools_mailbox_daemon_role_models_repro.py",
    "tools_mailbox_daemon_staleness_repro.py",
    "tools_mailbox_daemon_ticket_cycle_repro.py",
)


def manifest_problems():
    """Check the manifest against the folder before any child runs.

    Four properties are required: no manifest entry appears twice, every
    listed file exists in this folder, every listed file has a row in the
    README inventory table, and every ``*_repro.py`` file on disk is
    listed. The last property is the omission guard: adding a new
    reproduction without registering it here makes this command fail
    instead of quietly narrowing the regression surface.

    Returns:
      a list of plain-language problem sentences; empty when the manifest
      is safe to run.
    """
    problems = []
    listed_files = []
    for module in CONTROL_PLANE_TEST_MODULES:
        listed_files.append(module + ".py")
    for name in CONTROL_PLANE_REPRODUCTIONS:
        listed_files.append(name)
    seen = set()
    for name in listed_files:
        if name in seen:
            problems.append("manifest lists " + name + " more than once")
        seen.add(name)
    try:
        with open(README_PATH, "r", encoding="utf-8") as stream:
            readme_text = stream.read()
    except OSError as failure:
        problems.append("cannot read the README inventory: " + str(failure))
        readme_text = ""
    for name in listed_files:
        path = os.path.join(TESTS_DIR, name)
        if not os.path.isfile(path):
            problems.append("manifest entry is missing on disk: " + name)
        if readme_text and ("| `" + name + "` |") not in readme_text:
            problems.append("README inventory has no row for: " + name)
    on_disk = glob.glob(os.path.join(TESTS_DIR, "*_repro.py"))
    for path in sorted(on_disk):
        name = os.path.basename(path)
        if name not in CONTROL_PLANE_REPRODUCTIONS:
            problems.append("reproduction on disk is not in the manifest: "
                            + name)
    return problems


def run_check(label, command, log_stream, debug):
    """Run one required check in a child process and record its verdict.

    Arguments:
      label = short name printed beside the PASS or FAIL verdict.
      command = complete argument list for the child process.
      log_stream = open text stream receiving the child's complete output.
      debug = when True, also print the complete child output to the
              terminal instead of only the verdict line.

    Returns:
      True when the child returned exit code 0 within the time limit.
    """
    print("$ " + " ".join(command))
    try:
        completed = subprocess.run(command,
                                   cwd=REPO_ROOT,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   timeout=CHILD_TIMEOUT_SECONDS)
        output = completed.stdout
        return_code = completed.returncode
    except (OSError, subprocess.TimeoutExpired) as failure:
        output = str(failure) + "\n"
        return_code = -1
    log_stream.write("$ " + " ".join(command) + "\n")
    log_stream.write(output)
    log_stream.write("exit code: " + str(return_code) + "\n\n")
    log_stream.flush()
    if debug:
        print(output, end="")
    if return_code == 0:
        print("  PASS " + label)
        return True
    print("  FAIL " + label + " (exit code " + str(return_code) + ")")
    if not debug:
        tail_lines = output.splitlines()[-20:]
        for line in tail_lines:
            print("    " + line)
    return False


def main(argv=None):
    """Run the complete manifest and report one honest verdict.

    Arguments:
      argv = optional argument list for tests; None reads the command line.

    Returns:
      the process exit code described in the module docstring.
    """
    parser = argparse.ArgumentParser(
        description="Run every required AI control-plane regression: the "
                    "listed unittest modules plus every stand-alone "
                    "reproduction program, each in its own child process.")
    parser.add_argument("--debug", action="store_true",
                        help="also print every child's complete output to "
                             "the terminal; the log file is written either "
                             "way")
    args = parser.parse_args(argv)

    problems = manifest_problems()
    if problems:
        for problem in problems:
            print("manifest problem: " + problem)
        print("CONTROL-PLANE-REGRESSIONS UNSAFE: no check was run.")
        return 2

    descriptor, log_path = tempfile.mkstemp(
        prefix="control-plane-regressions-", suffix=".log")
    print("full output log: " + log_path)
    failed_labels = []
    total = 0
    with os.fdopen(descriptor, "w", encoding="utf-8") as log_stream:
        unittest_command = [sys.executable, "-m", "unittest"]
        for module in CONTROL_PLANE_TEST_MODULES:
            unittest_command.append("ai.tests." + module)
        total += 1
        if not run_check(label="control-plane unit tests",
                         command=unittest_command,
                         log_stream=log_stream,
                         debug=args.debug):
            failed_labels.append("control-plane unit tests")
        for name in CONTROL_PLANE_REPRODUCTIONS:
            command = [sys.executable, os.path.join("ai", "tests", name)]
            total += 1
            if not run_check(label=name,
                             command=command,
                             log_stream=log_stream,
                             debug=args.debug):
                failed_labels.append(name)

    if failed_labels:
        print("CONTROL-PLANE-REGRESSIONS FAIL: "
              + str(len(failed_labels)) + " of " + str(total)
              + " commands failed: " + ", ".join(failed_labels))
        print("full output: " + log_path)
        return 1
    print("CONTROL-PLANE-REGRESSIONS PASS: all " + str(total)
          + " required commands ran and passed (complete run, no skips).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
