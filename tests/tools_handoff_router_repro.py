#!/usr/bin/env python3
"""Reproduce and verify the handoff router's relay defects in scratch.

This script never reads or writes the live relay directory and never invokes
an agent CLI. Each arm copies the router into a temporary fake repository.
"""

import concurrent.futures
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "handoff_router.py"


def load_scratch_router(root, name):
    """Copy and import the router from one temporary fake repository.

    Arguments:
      root = temporary directory that will contain the fake repository.
      name = unique import name for this scratch module.

    Returns:
      ``(module, repo)`` for the isolated router and fake repository root.
    """
    repo = root / "repo"
    tools_dir = repo / "tools"
    relay_dir = repo / "notes" / "relay"
    tools_dir.mkdir(parents=True)
    relay_dir.mkdir(parents=True)
    target = tools_dir / "handoff_router.py"
    shutil.copy2(SOURCE, target)
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (module, repo)


def run_git(repo, *args):
    """Run one harmless git command inside a temporary repository.

    Arguments:
      repo = temporary repository root.
      args = git arguments.

    Returns:
      the completed subprocess.
    """
    return subprocess.run(
      ["git"] + list(args),
      cwd=repo,
      check=True,
      capture_output=True,
      text=True,
    )


def arm_cwd():
    """Show that repo-relative I/O no longer follows an unusual shell cwd."""
    with tempfile.TemporaryDirectory(prefix="router-cwd-") as tmp:
        root = Path(tmp)
        module, repo = load_scratch_router(root, "scratch_router_cwd")
        note = repo / "notes" / "spec.md"
        note.write_text("scratch specification\n", encoding="utf-8")
        run_git(repo, "init", "-q", "-b", "main")

        outside = root / "outside"
        outside.mkdir()
        old_cwd = Path.cwd()
        os.chdir(outside)
        try:
            legacy = subprocess.run(
              "pwd -P",
              shell=True,
              check=True,
              capture_output=True,
              text=True,
            ).stdout.strip()
            seq = module.reserve_run_sequence(stamp="20000101-000000")
            log_path, all_green = module.run_gates(commands=["pwd -P"], seq=seq)
            log = (repo / log_path).read_text(encoding="utf-8")
            note_path, note_display = module.resolve_note_path("notes/spec.md")
            git_root = module._git(["rev-parse", "--show-toplevel"])
        finally:
            os.chdir(old_cwd)

        patched_gate_root = str(repo.resolve()) in log
        resolved_note = note_path == str(note)
        patched_git_root = Path(git_root).resolve() == repo.resolve()
        legacy_followed_caller = Path(legacy).resolve() == outside.resolve()
        print("ARM cwd")
        print("  legacy gate cwd followed caller:", legacy_followed_caller)
        print("  patched gate cwd is fake repo:", patched_gate_root)
        print("  patched relative note:", note_display, resolved_note)
        print("  patched git cwd is fake repo:", patched_git_root)
        assert all_green
        assert legacy_followed_caller
        assert patched_gate_root
        assert resolved_note
        assert patched_git_root


def arm_sequence_collision():
    """Show the former overwrite and the atomic sequence reservations."""
    with tempfile.TemporaryDirectory(prefix="router-seq-") as tmp:
        root = Path(tmp)
        module, repo = load_scratch_router(root, "scratch_router_seq")
        relay = repo / "notes" / "relay"
        stamp = "20000101-000000"

        legacy_path = relay / (stamp + "-implementer.md")
        legacy_path.write_text("first\n", encoding="utf-8")
        legacy_path.write_text("second\n", encoding="utf-8")
        legacy_lost_first = legacy_path.read_text(encoding="utf-8") == "second\n"

        def reserve(_index):
            return module.reserve_run_sequence(stamp=stamp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            sequences = list(pool.map(reserve, range(8)))
        for index, seq in enumerate(sequences):
            module.archive(seq, "implementer", "payload " + str(index))
        payloads = []
        for seq in sequences:
            path = relay / (seq + "-implementer.md")
            payloads.append(path.read_text(encoding="utf-8"))

        print("ARM sequence collision")
        print("  legacy first payload lost:", legacy_lost_first)
        print("  patched unique reservations:", len(set(sequences)), "of 8")
        print("  patched payload files preserved:", len(payloads), "of 8")
        assert legacy_lost_first
        assert len(set(sequences)) == 8
        assert len(payloads) == 8


def arm_clipboard_lock():
    """Show concurrent flows collide and a stale lock file is harmless."""
    with tempfile.TemporaryDirectory(prefix="router-lock-") as tmp:
        root = Path(tmp)
        module, _repo = load_scratch_router(root, "scratch_router_lock")
        module.ROUTER_LOCK_PATH = str(root / "router.lock")

        response = "### IMPLEMENTER_HANDOFF: REQUESTING REVIEW\n"
        baselines = ["prompt A", "prompt B"]
        legacy_claims = 0
        for baseline in baselines:
            if response != baseline and "IMPLEMENTER_HANDOFF" in response:
                legacy_claims += 1

        first_lock = module.acquire_router_lock()
        refused = False
        try:
            try:
                module.acquire_router_lock()
            except RuntimeError:
                refused = True
        finally:
            module.release_router_lock(first_lock)

        stale_file_exists = Path(module.ROUTER_LOCK_PATH).is_file()
        next_lock = module.acquire_router_lock()
        module.release_router_lock(next_lock)

        print("ARM clipboard lock")
        print("  legacy flows claiming one response:", legacy_claims)
        print("  patched concurrent start refused:", refused)
        print("  persistent lock file reacquired:", stale_file_exists)
        assert legacy_claims == 2
        assert refused
        assert stale_file_exists


def arm_handoff_header():
    """Show token prose is rejected while a real handoff heading is read."""
    with tempfile.TemporaryDirectory(prefix="router-header-") as tmp:
        root = Path(tmp)
        module, _repo = load_scratch_router(root, "scratch_router_header")
        header = "### IMPLEMENTER_HANDOFF:"
        prose = "I cannot produce IMPLEMENTER_HANDOFF yet."
        valid = "### IMPLEMENTER_HANDOFF: REQUESTING REVIEW\n\n- done\n"
        legacy_accepted_prose = "IMPLEMENTER_HANDOFF" in prose

        clipboard_values = iter([prose, valid])

        def scratch_read_clipboard():
            return next(clipboard_values)

        def no_sleep(_seconds):
            return None

        module.read_clipboard = scratch_read_clipboard
        module.time.sleep = no_sleep
        captured = module.wait_for_block(header=header, last_copied="prompt")

        print("ARM handoff header")
        print("  legacy accepted token prose:", legacy_accepted_prose)
        prose_accepted = module.has_handoff_header(prose, header)
        print("  patched token prose accepted:", prose_accepted)
        print("  patched captured real heading:", captured == valid)
        assert legacy_accepted_prose
        assert not prose_accepted
        assert captured == valid


def arm_clipboard_failure():
    """Show a failed clipboard read is an error instead of an endless wait."""
    with tempfile.TemporaryDirectory(prefix="router-paste-") as tmp:
        root = Path(tmp)
        module, _repo = load_scratch_router(root, "scratch_router_paste")

        class FailedPaste:
            returncode = 1
            stdout = b""

        def failed_run(*_args, **_kwargs):
            return FailedPaste()

        legacy_text = FailedPaste.stdout.decode("utf-8", errors="replace")
        original_platform = module.sys.platform
        original_run = module.subprocess.run
        module.sys.platform = "darwin"
        module.subprocess.run = failed_run
        raised = False
        try:
            try:
                module.read_clipboard()
            except RuntimeError:
                raised = True
        finally:
            module.sys.platform = original_platform
            module.subprocess.run = original_run

        print("ARM clipboard failure")
        print("  legacy failed read looked like empty clipboard:",
              legacy_text == "")
        print("  patched failed read raises:", raised)
        assert legacy_text == ""
        assert raised


def arm_integrated_status():
    """Show a Codex branch merged to main is integrated without Claude."""
    with tempfile.TemporaryDirectory(prefix="router-status-") as tmp:
        root = Path(tmp)
        module, repo = load_scratch_router(root, "scratch_router_status")
        run_git(repo, "init", "-q", "-b", "main")
        run_git(repo, "config", "user.email", "scratch@example.invalid")
        run_git(repo, "config", "user.name", "Scratch Probe")
        tracked = repo / "tracked.txt"
        tracked.write_text("base\n", encoding="utf-8")
        run_git(repo, "add", "tracked.txt")
        run_git(repo, "commit", "-q", "-m", "base")
        run_git(repo, "branch", "claude/working")
        run_git(repo, "checkout", "-q", "-b", "codex/done")
        tracked.write_text("base\ncodex\n", encoding="utf-8")
        run_git(repo, "commit", "-q", "-am", "codex")
        run_git(repo, "checkout", "-q", "main")
        run_git(repo, "merge", "-q", "--no-ff", "codex/done", "-m", "merge")

        legacy = subprocess.run(
          ["git", "merge-base", "--is-ancestor", "codex/done",
           "claude/working"],
          cwd=repo,
          capture_output=True,
        )
        patched = subprocess.run(
          ["git", "merge-base", "--is-ancestor", "codex/done", "main"],
          cwd=repo,
          capture_output=True,
        )
        module.reserve_run_sequence(stamp="20000101-000000")
        status_stream = io.StringIO()
        with contextlib.redirect_stdout(status_stream):
            module.status_report()
        status_text = status_stream.getvalue()
        status_is_integrated = "[integrated]" in status_text
        reservation_hidden = ".router-runs" not in status_text

        print("ARM integrated status")
        print("  legacy reports merged Codex branch open:",
              legacy.returncode != 0)
        print("  patched main ancestry says integrated:",
              patched.returncode == 0)
        print("  patched status reports integrated:", status_is_integrated)
        print("  reservation metadata hidden from status:", reservation_hidden)
        assert legacy.returncode != 0
        assert patched.returncode == 0
        assert status_is_integrated
        assert reservation_hidden


def arm_second_implementer_mode():
    """Drive the ruled CLI value and inspect the routed Sol declaration."""
    with tempfile.TemporaryDirectory(prefix="router-second-implementer-") as tmp:
        root = Path(tmp)
        module, repo = load_scratch_router(
          root, "scratch_router_second_implementer")
        note = repo / "notes" / "spec.md"
        note.write_text("scratch specification\n", encoding="utf-8")
        module.ROUTER_LOCK_PATH = str(root / "router.lock")

        copied = []
        returns = iter([
          "### IMPLEMENTER_HANDOFF: DONE\n",
          "### ARCHITECT_REDTEAM_HANDOFF: DONE\n",
        ])

        def capture_copy(text):
            copied.append(text)

        def next_handoff(**_kwargs):
            return next(returns)

        module.copy_to_clipboard = capture_copy
        module.wait_for_block = next_handoff
        module.run_gates = lambda commands, seq: (
          "notes/relay/scratch-gates.md", True)

        original_argv = module.sys.argv
        module.sys.argv = [
          "handoff_router.py",
          "--note", "notes/spec.md",
          "--mode", "second-implementer",
        ]
        routed_stream = io.StringIO()
        try:
            with contextlib.redirect_stdout(routed_stream):
                routed_rc = module.main()
        finally:
            module.sys.argv = original_argv

        sol_prompts = [
          text for text in copied
          if text.startswith("### ARCHITECT_REDTEAM_HANDOFF")
        ]
        expected = (
          "OpenAI Sol — this is a role as second Implementer for this unit.")
        declaration_exact = (
          module.SECOND_IMPLEMENTER_MODE_SENTENCE == expected)
        prompt_exact = (
          len(sol_prompts) == 1
          and sol_prompts[0].count(expected) == 1
          and "\n\n" + expected + "\n\n" in sol_prompts[0])

        module.sys.argv = ["handoff_router.py", "--mode", "backup"]
        invalid_stream = io.StringIO()
        backup_rejected = False
        try:
            with contextlib.redirect_stderr(invalid_stream):
                module.main()
        except SystemExit as error:
            backup_rejected = error.code == 2
        finally:
            module.sys.argv = original_argv

        print("ARM second-Implementer mode")
        print("  ruled CLI value completes routing:", routed_rc == 0)
        print("  exact declaration routed once:",
              declaration_exact and prompt_exact)
        print("  retired backup value rejected:", backup_rejected)
        assert routed_rc == 0
        assert declaration_exact
        assert prompt_exact
        assert backup_rejected


def main():
    """Run every isolated reproduction arm."""
    arm_cwd()
    arm_sequence_collision()
    arm_clipboard_lock()
    arm_handoff_header()
    arm_clipboard_failure()
    arm_integrated_status()
    arm_second_implementer_mode()
    print("ALL SCRATCH ROUTER REPRODUCTIONS PASS")


if __name__ == "__main__":
    main()
