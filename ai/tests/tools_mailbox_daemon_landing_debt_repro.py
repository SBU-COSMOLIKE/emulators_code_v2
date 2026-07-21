#!/usr/bin/env python3
"""Scratch regression for the read-only saved-candidate debt meter.

The meter is deliberately not a mailbox publisher.  Each runtime arm loads a
fresh daemon into a disposable directory and, when Git history is needed,
creates a private repository there.  No arm reads or writes the live mailbox,
changes a real branch, or invokes an agent command.
"""

import contextlib
import io
import pathlib
import subprocess
import sys
import tempfile
import types

try:
    from ai.tests import tools_mailbox_daemon_fix_only_repro as fix_only_repro
except ImportError:
    import tools_mailbox_daemon_fix_only_repro as fix_only_repro


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"


def load_daemon(source=None):
    """Execute one fresh production module, optionally from mutated source.

    ``source`` may be None (production), the entry file's text, or a
    mapping of daemon file name to text so a mutation may live in any of
    the daemon's source files.
    """
    if isinstance(source, dict):
        return fix_only_repro.load_daemon(source=source)
    if source is None:
        source = DAEMON_PATH.read_text(encoding="utf-8")
    module = types.ModuleType("mailbox_daemon_landing_debt_repro")
    module.__file__ = str(DAEMON_PATH)
    exec(compile(source, str(DAEMON_PATH), "exec"), module.__dict__)
    return module


def run_git(repository, *arguments, input_text=None):
    """Run one bounded scratch Git command and return stripped stdout."""
    process = subprocess.run(
        ["git", "-C", str(repository)] + list(arguments),
        input=input_text, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=10)
    if process.returncode != 0:
        raise RuntimeError(
            "scratch git " + " ".join(arguments) + " failed: "
            + process.stderr.strip())
    return process.stdout.strip()


@contextlib.contextmanager
def scratch_daemon(source=None):
    """Yield a daemon and a complete disposable Git/mailbox surface."""
    with tempfile.TemporaryDirectory(prefix="mailbox-candidate-meter-") as tmp:
        root = pathlib.Path(tmp)
        repository = root / "repository"
        repository.mkdir()
        run_git(repository, "init", "-b", "main")
        run_git(repository, "config", "user.name", "Candidate Meter Test")
        run_git(repository, "config", "user.email", "meter@example.invalid")
        (repository / "science.txt").write_text(
            "alpha\nbeta\ngamma\n", encoding="utf-8")
        run_git(repository, "add", "science.txt")
        run_git(repository, "commit", "-m", "base")

        mailbox = root / "mailbox"
        mailbox.mkdir()
        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(repository)
        daemon.WORKTREE = str(repository)
        daemon.MAILBOX = str(mailbox)
        daemon.AGENT_CWD = dict(daemon.AGENT_CWD)
        daemon.AGENT_CWD["fable"] = str(repository)
        yield daemon, repository, mailbox


def exact_snapshot(changed_lines, active_candidates=1):
    """Return the expected available meter payload."""
    noun = "candidate" if active_candidates == 1 else "candidates"
    return {
        "available": True,
        "stat": (str(active_candidates) + " active " + noun + ", "
                 + str(changed_lines) + " changed lines"),
        "changed_lines": changed_lines,
        "returncode": 0,
    }


def install_candidate(daemon, repository, phase="implementation"):
    """Create C, its private ref, and exact durable daemon records.

    The candidate changes three lines by insertion and one by deletion, so
    the contract's addition-plus-deletion size is exactly four.
    """
    base = run_git(repository, "rev-parse", "HEAD")
    cycle_id = "meter@" + base
    (repository / "science.txt").write_text(
        "alpha\nBETA\ngamma\ndelta\nepsilon\n", encoding="utf-8")
    run_git(repository, "add", "science.txt")
    run_git(repository, "commit", "-m", "candidate")
    candidate = run_git(repository, "rev-parse", "HEAD")
    candidate_tree = run_git(repository, "rev-parse", candidate + "^{tree}")
    landing = run_git(
        repository, "commit-tree", candidate_tree, "-p", base,
        input_text="distinct squash landing\n")
    run_git(repository, "reset", "--hard", base)

    reference = daemon.cycle_candidate_ref(cycle_id=cycle_id)
    run_git(repository, "update-ref", reference, candidate)
    candidate_state = daemon.empty_candidate_state()
    candidate_state["cycles"][cycle_id] = {
        "ref": reference, "commit": candidate}
    daemon.write_candidate_state(state=candidate_state)

    ticket_state = daemon.empty_ticket_cycle_state()
    ticket_state["generation"] = 1
    if phase == "implementation":
        ticket_state["active"][cycle_id] = {
            "phase": "implementation", "commit": None,
            "mode": "normal", "route": "primary"}
    elif phase == "landed":
        ticket_state["active"][cycle_id] = {
            "phase": "awaiting-redteam", "commit": landing,
            "mode": "normal", "route": "primary"}
    elif phase == "completed":
        ticket_state["completed"][cycle_id] = landing
    elif phase != "unowned":
        raise ValueError("unknown scratch ticket phase: " + phase)
    daemon.write_ticket_cycle_state(state=ticket_state)
    return {
        "base": base, "cycle": cycle_id, "candidate": candidate,
        "landing": landing, "ref": reference,
    }


def filesystem_manifest(root):
    """Return byte content and write timestamps for every ordinary file."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = str(path.relative_to(root))
            status = path.stat()
            result[relative] = (path.read_bytes(), status.st_mtime_ns)
    return result


def arm_no_automatic_publication_surface(source=None):
    """No retired auto-Fable publisher, watch hook, state, or marker remains."""
    if source is None:
        files = fix_only_repro.daemon_source_files()
        source = "\n".join(files[name] for name in sorted(files))
    forbidden = (
        "def reconcile_landing_debt_handoff(",
        "reconcile_landing_debt_handoff()",
        "def automatic_landing_debt_marker(",
        "def automatic_landing_debt_message_exists(",
        "MAILBOX-AUTO: landing-debt",
        "LANDING_DEBT_STATE_SCHEMA",
        "LANDING_DEBT_STATE_NAME",
        "def landing_debt_state_path(",
        "def read_landing_debt_state(",
        "def write_landing_debt_state(",
    )
    return (
        all(item not in source for item in forbidden)
        and source.count("def landing_debt_snapshot():") == 1
        and source.count("def report_landing_debt(snapshot=None):") == 1)


def arm_missing_state_is_zero(source=None):
    """A clean repository with no saved records reports zero, not branch debt."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        snapshot = daemon.landing_debt_snapshot()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            returned = daemon.report_landing_debt(snapshot=snapshot)
        return (
            snapshot == {
                "available": True, "stat": "", "changed_lines": 0,
                "returncode": 0}
            and returned == snapshot
            and output.getvalue().splitlines() == [
                "landing debt: none; no saved active candidate is waiting"]
            and not pathlib.Path(daemon.ticket_cycle_state_path()).exists()
            and not pathlib.Path(daemon.candidate_state_path()).exists()
            and not list(mailbox.glob("*-to-fable.md")))


def arm_active_candidate_exact_size(source=None):
    """One owned active C reports additions plus deletions for base..C."""
    with scratch_daemon(source=source) as (daemon, repository, _):
        record = install_candidate(
            daemon=daemon, repository=repository, phase="implementation")
        snapshot = daemon.landing_debt_snapshot()
        ref_commit = run_git(repository, "rev-parse", record["ref"])
        head = run_git(repository, "rev-parse", "HEAD")
        return (
            snapshot == exact_snapshot(changed_lines=4)
            and ref_commit == record["candidate"]
            and head == record["base"])


def arm_landed_recovery_candidate_is_ignored(source=None):
    """Retained C authority is not new debt after landing or completion."""
    outcomes = []
    for phase in ("landed", "completed"):
        with scratch_daemon(source=source) as (daemon, repository, _):
            record = install_candidate(
                daemon=daemon, repository=repository, phase=phase)
            snapshot = daemon.landing_debt_snapshot()
            outcomes.append(
                snapshot == {
                    "available": True, "stat": "", "changed_lines": 0,
                    "returncode": 0}
                and run_git(repository, "rev-parse", record["ref"])
                == record["candidate"])
    return all(outcomes)


def arm_malformed_state_is_unavailable(source=None):
    """Malformed candidate JSON fails closed instead of reporting zero."""
    with scratch_daemon(source=source) as (daemon, _, _):
        pathlib.Path(daemon.candidate_state_path()).write_text(
            '{"schema":1,"cycles":[],"extra":true}\n',
            encoding="utf-8")
        return daemon.landing_debt_snapshot() == {
            "available": False, "stat": "", "changed_lines": 0,
            "returncode": 1}


def arm_unowned_candidate_is_unavailable(source=None):
    """A saved C without an active/completed ticket cannot be counted."""
    with scratch_daemon(source=source) as (daemon, repository, _):
        install_candidate(
            daemon=daemon, repository=repository, phase="unowned")
        return daemon.landing_debt_snapshot() == {
            "available": False, "stat": "", "changed_lines": 0,
            "returncode": 1}


def arm_ref_mismatch_is_unavailable(source=None):
    """Candidate state and its private Git ref must name the same C."""
    with scratch_daemon(source=source) as (daemon, repository, _):
        record = install_candidate(
            daemon=daemon, repository=repository, phase="implementation")
        run_git(repository, "update-ref", record["ref"], record["base"])
        return daemon.landing_debt_snapshot() == {
            "available": False, "stat": "", "changed_lines": 0,
            "returncode": 1}


def arm_report_is_truthful_and_read_only(source=None):
    """An injected measurement prints one line and changes no file bytes."""
    with scratch_daemon(source=source) as (daemon, repository, mailbox):
        install_candidate(
            daemon=daemon, repository=repository, phase="implementation")
        injected = exact_snapshot(changed_lines=4)
        before = filesystem_manifest(root=mailbox)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            returned = daemon.report_landing_debt(snapshot=injected)
        after = filesystem_manifest(root=mailbox)
        return (
            returned is injected
            and output.getvalue().splitlines()
            == ["landing debt: 1 active candidate, 4 changed lines"]
            and before == after
            and not list(mailbox.rglob("*-to-fable.md")))


def old_main_head_mutant(source):
    """Replace the candidate meter with the retired branch-tip comparison.

    ``source`` is the text of ``mailbox_recovery.py``, the daemon part
    file that owns the landing-debt meter.
    """
    start = source.find("def landing_debt_snapshot():\n")
    end = source.find("\ndef report_landing_debt(snapshot=None):\n", start)
    if start < 0 or end < 0:
        raise ValueError("landing-debt function boundary is not unique")
    replacement = '''def landing_debt_snapshot():
    """Incorrect historical branch-tip meter used only as a mutant."""
    process = daemon.subprocess.run(
        ["git", "diff", "--shortstat", "main..HEAD"],
        stdout=daemon.subprocess.PIPE, stderr=daemon.subprocess.PIPE,
        text=True, cwd=daemon.AGENT_CWD["fable"], check=False)
    if process.returncode != 0:
        return {"available": False, "stat": "", "changed_lines": 0,
                "returncode": process.returncode}
    changed_lines = sum(int(value) for value, _kind in daemon.re.findall(
        r"(\\d+) (insertion|deletion)", process.stdout))
    return {"available": True, "stat": process.stdout.strip(),
            "changed_lines": changed_lines, "returncode": 0}

'''
    return source[:start] + replacement + source[end + 1:]


def mutation_old_main_head_is_killed():
    """The active-C witness must reject a return to main..HEAD sizing."""
    sources = fix_only_repro.daemon_source_files()
    part = sources["mailbox_recovery.py"]
    mutant_part = old_main_head_mutant(source=part)
    compile(mutant_part, "mailbox_recovery.py", "exec")
    mutated = dict(sources)
    mutated["mailbox_recovery.py"] = mutant_part
    return not arm_active_candidate_exact_size(source=mutated)


def main():
    """Run every bounded runtime arm and the one named source mutation."""
    arms = [
        ("no-automatic-publication-surface",
         arm_no_automatic_publication_surface),
        ("missing-state-is-zero", arm_missing_state_is_zero),
        ("active-candidate-exact-size", arm_active_candidate_exact_size),
        ("landed-recovery-candidate-is-ignored",
         arm_landed_recovery_candidate_is_ignored),
        ("malformed-state-is-unavailable", arm_malformed_state_is_unavailable),
        ("unowned-candidate-is-unavailable",
         arm_unowned_candidate_is_unavailable),
        ("ref-mismatch-is-unavailable", arm_ref_mismatch_is_unavailable),
        ("report-is-truthful-and-read-only",
         arm_report_is_truthful_and_read_only),
    ]
    outcomes = []
    for label, arm in arms:
        try:
            passed = bool(arm())
            detail = ""
        except BaseException as exc:
            passed = False
            detail = " (" + type(exc).__name__ + ": " + str(exc) + ")"
        outcomes.append(passed)
        print("ARM " + label + " " + ("PASS" if passed else "FAIL")
              + detail)
    print("ARM SUMMARY passed=%d/%d" % (sum(outcomes), len(outcomes)))

    try:
        mutation_green = mutation_old_main_head_is_killed()
        detail = ""
    except BaseException as exc:
        mutation_green = False
        detail = " (" + type(exc).__name__ + ": " + str(exc) + ")"
    print("MUTATION old-main-dotdot-HEAD "
          + ("RED" if mutation_green else "SURVIVED") + detail)
    print("MUTATION SUMMARY killed=%d/1" % int(mutation_green))

    all_green = all(outcomes) and mutation_green
    print("SUMMARY " + ("PASS" if all_green else "FAIL"))
    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
