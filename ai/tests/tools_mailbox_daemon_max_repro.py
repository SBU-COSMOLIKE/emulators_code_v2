#!/usr/bin/env python3
"""Scratch-only regression for the mailbox ticket character budget.

Every live-looking arm redirects the daemon into a temporary folder and
replaces agent commands with harmless finished children. No arm creates a
real worktree, writes the live mailbox, or launches an agent program.
"""

import contextlib
import io
import os
import pathlib
import sys
import tempfile
import types


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
BASE_COMMIT = "1" * 40
CYCLE_ID = "character-budget@" + BASE_COMMIT


class FinishedProcess:
    """A harmless Popen-shaped child which has already succeeded."""

    def __init__(self):
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


def load_daemon(source=None):
    """Load a fresh production daemon, optionally from mutated source."""
    if source is None:
        source = DAEMON_PATH.read_text(encoding="utf-8")
    module = types.ModuleType("mailbox_daemon_max_repro")
    module.__file__ = str(DAEMON_PATH)
    exec(compile(source, str(DAEMON_PATH), "exec"), module.__dict__)
    return module


def install_test_agent_topology_proof(daemon):
    """Install opaque topology and persistent-state proofs for each role."""
    agents = ("fable", "opus", "sol")
    topology_proofs = {agent: object() for agent in agents}
    persistent_proofs = {agent: object() for agent in agents}

    def validate_topology(agent):
        return topology_proofs[agent]

    def revalidate_topology(proof):
        if proof not in topology_proofs.values():
            raise AssertionError("scratch role topology proof changed")
        return proof

    def capture_persistent_state(agent):
        return persistent_proofs[agent]

    def recheck_persistent_state(proof):
        if proof not in persistent_proofs.values():
            raise AssertionError("scratch persistent role state changed")

    daemon.validate_live_agent_dispatch_topology = validate_topology
    daemon.revalidate_agent_dispatch_topology = revalidate_topology
    daemon.capture_persistent_role_state = capture_persistent_state
    daemon.recheck_persistent_role_state = recheck_persistent_state


def tree_snapshot(root):
    """Return the byte-and-type state below a disposable folder."""
    result = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            result.append((relative, "symlink", os.readlink(path)))
        elif path.is_file():
            result.append((relative, "file", path.read_bytes()))
        elif path.is_dir():
            result.append((relative, "dir", b""))
        else:
            result.append((relative, "other", b""))
    return result


@contextlib.contextmanager
def scratch_daemon(source=None):
    """Yield one daemon whose complete mutable surface is disposable."""
    with tempfile.TemporaryDirectory(prefix="mailbox-max-characters-") as tmp:
        root = pathlib.Path(tmp)
        ai_root = root / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        relay = ai_root / "notes" / "relay"
        mailbox.mkdir(parents=True)
        backlog = ai_root / "notes" / "backlog.md"
        backlog.write_text(
            "- OPEN **MEDIUM** **BUG FIX** — "
            "[Character budget](#character-budget)\n\n"
            '<a id="character-budget"></a>\n'
            "## Character budget\n\n"
            "**Red Team reopen count: 0.**\n\n"
            "**Red Team reopening: allowed.**\n",
            encoding="utf-8")
        architect_worktree = root / "architect-worktree"
        implementer_worktree = root / "implementer-worktree"
        sol_worktree = root / "sol-worktree"
        architect_worktree.mkdir()
        implementer_worktree.mkdir()
        sol_worktree.mkdir()

        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
        daemon.MAILBOX = str(mailbox)
        daemon.DONE = str(mailbox / "done")
        daemon.RELAY_DIR = str(relay)
        daemon.BACKLOG_LEDGER = str(backlog)
        daemon.PREAMBLE = "scratch common preamble\n--- MESSAGE ---\n"
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"],
            "opus": ["harmless-opus"],
            "sol": ["harmless-sol"],
        }
        daemon.AGENT_CWD = {
            "fable": str(architect_worktree),
            "opus": str(implementer_worktree),
            "sol": str(sol_worktree),
        }
        install_test_agent_topology_proof(daemon=daemon)
        daemon.git_commit_exists = lambda commit: commit == BASE_COMMIT
        daemon.warn_if_mailbox_unwatched = lambda: None
        daemon.report_demand = lambda backlog, skip_redteam=False: None
        daemon.report_landing_debt = lambda: None
        yield daemon, root, mailbox


def call_main(daemon, arguments):
    """Call main with isolated argv and capture output plus SystemExit."""
    previous_argv = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = None
    error = None
    sys.argv = ["mailbox_daemon.py"] + list(arguments)
    try:
        with contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            try:
                result = daemon.main()
            except BaseException as exc:
                error = exc
    finally:
        sys.argv = previous_argv
    if isinstance(error, SystemExit):
        returncode = error.code if isinstance(error.code, int) else 1
    elif error is None:
        returncode = 0 if result is None else result
    else:
        returncode = 1
    return returncode, stdout.getvalue(), stderr.getvalue(), error


def prepare_finite_once(daemon):
    """Replace one live once pass with a read-only empty pass."""
    lock = object()
    daemon.acquire_dispatch_lock = lambda mode: lock
    daemon.release_dispatch_lock = lambda lock_file: None
    daemon.process_backlog = lambda dry_run, **kwargs: None


def arm_default_and_explicit_zero(source=None):
    """Omitted and explicit zero both mean an unlimited once pass."""
    observations = []
    for arguments in (["--once"], ["--once", "--max", "0"]):
        with scratch_daemon(source=source) as (daemon, root, _):
            prepare_finite_once(daemon=daemon)
            before = tree_snapshot(root)
            returncode, output, error_output, error = call_main(
                daemon=daemon, arguments=arguments)
            observations.append(
                returncode == 0 and error is None and error_output == ""
                and daemon.MAX_CHARACTERS == 0
                and output.count(
                    "ticket character limit: none (--max 0)") == 1
                and tree_snapshot(root) == before)
    passed = all(observations)
    print("default and explicit zero unlimited=" + str(passed))
    return passed


def arm_positive_once(source=None):
    """A positive once value reaches the module policy and startup line."""
    with scratch_daemon(source=source) as (daemon, root, _):
        prepare_finite_once(daemon=daemon)
        before = tree_snapshot(root)
        returncode, output, error_output, error = call_main(
            daemon=daemon, arguments=["--once", "--max", "37"])
        passed = (
            returncode == 0 and error is None and error_output == ""
            and daemon.MAX_CHARACTERS == 37
            and output.count(
                "ticket character limit: 37 added plus deleted characters "
                "per ticket") == 1
            and tree_snapshot(root) == before)
    print("positive once propagation=" + str(passed))
    return passed


def arm_dry_run_once(source=None):
    """A one-pass preview accepts and reports the selected ticket limit."""
    with scratch_daemon(source=source) as (daemon, root, _):
        prepare_finite_once(daemon=daemon)
        before = tree_snapshot(root)
        returncode, output, error_output, error = call_main(
            daemon=daemon,
            arguments=["--dry-run", "--once", "--max", "29"])
        passed = (
            returncode == 0 and error is None and error_output == ""
            and daemon.MAX_CHARACTERS == 29
            and "ticket character limit: 29 added plus deleted characters "
            "per ticket" in output
            and tree_snapshot(root) == before)
    print("dry-run once limit preview=" + str(passed))
    return passed


class ExitAfterOnePass:
    """A watch controller that requests a clean source-change exit."""

    def __init__(self, source_path, source_stamp, ticket_cycle_limit=None,
                 ticket_cycle_topology=None):
        del source_path, source_stamp, ticket_cycle_limit
        del ticket_cycle_topology

    def source_changed(self):
        return True


def arm_positive_watch(source=None):
    """A watch accepts max independently of cycle and exits without launch."""
    with scratch_daemon(source=source) as (daemon, root, _):
        lock = object()
        daemon.acquire_dispatch_lock = lambda mode: lock
        daemon.release_dispatch_lock = lambda lock_file: None
        daemon.process_backlog = lambda dry_run, **kwargs: None
        daemon.SafeKillRendezvous = ExitAfterOnePass
        before = tree_snapshot(root)
        returncode, output, error_output, error = call_main(
            daemon=daemon, arguments=["--watch", "--max", "41"])
        passed = (
            returncode == 0 and error is None and error_output == ""
            and daemon.MAX_CHARACTERS == 41
            and "ticket character limit: 41 added plus deleted characters "
            "per ticket" in output
            and "cycle mode:" not in output
            and tree_snapshot(root) == before)
    print("positive watch independent of cycle=" + str(passed))
    return passed


def arm_invalid_values_and_actions(source=None):
    """Malformed or misplaced max values refuse before any filesystem write."""
    cases = [
        (["--once", "--max", "-1"], 2,
         "max characters must use only decimal digits 0 through 9"),
        (["--once", "--max", "many"], 2,
         "max characters must use only decimal digits 0 through 9"),
        (["--once", "--max", "1.5"], 2,
         "max characters must use only decimal digits 0 through 9"),
        (["--once", "--max", "+1"], 2,
         "max characters must use only decimal digits 0 through 9"),
        (["--once", "--max", " 1"], 2,
         "max characters must use only decimal digits 0 through 9"),
        (["--once", "--max", "1 "], 2,
         "max characters must use only decimal digits 0 through 9"),
        (["--once", "--max", "1_0"], 2,
         "max characters must use only decimal digits 0 through 9"),
        (["--once", "--max", "١"], 2,
         "max characters must use only decimal digits 0 through 9"),
        (["--max", "7"], 1,
         "--max is valid only with --watch or --once"),
        (["--dry-run", "--max", "0"], 1,
         "--max is valid only with --watch or --once"),
        (["--send", "architect", "--unit", "scratch", "--max", "7"], 1,
         "--max is valid only with --watch or --once"),
        (["--ping", "--max", "7"], 1,
         "--max is valid only with --watch or --once"),
    ]
    observations = []
    for arguments, expected_returncode, expected_text in cases:
        with scratch_daemon(source=source) as (daemon, root, _):
            primary_attempts = []
            daemon.__name__ = "__main__"
            daemon.ensure_primary_execution = (
                lambda live_action, dry_run:
                primary_attempts.append((live_action, dry_run)))
            before = tree_snapshot(root)
            returncode, output, error_output, _error = call_main(
                daemon=daemon, arguments=arguments)
            combined = output + error_output
            observations.append(
                returncode == expected_returncode
                and expected_text in combined
                and primary_attempts == []
                and tree_snapshot(root) == before)
    passed = all(observations)
    print("invalid max values/actions refuse without writes=" + str(passed))
    return passed


def fake_popen(calls):
    """Capture command, prompt, and environment for one harmless child."""
    def replacement(command, stdout, stderr, cwd, env):
        del stderr
        calls.append({
            "command": list(command),
            "cwd": cwd,
            "env": dict(env),
        })
        stdout.write("bounded fake child output\n")
        stdout.flush()
        return FinishedProcess()

    return replacement


def pending_body(daemon, agent):
    """Return one valid byte-exact mailbox body for a route."""
    if agent == "fable":
        return daemon.architect_user_request_payload(
            text="Work the scratch Architect unit.").encode("utf-8")
    if agent == "opus":
        return (
            "MAILBOX-FLOW: ticket\n"
            "MAILBOX-CYCLE: " + CYCLE_ID + "\n"
            "MAILBOX-MODE: normal\n\n"
            "Work the scratch Implementer unit.\n").encode("utf-8")
    if agent == "sol":
        return daemon.sol_ticket_payload(
            ticket_kind="discovery", text="Review one bounded scratch unit.",
            discovery_severity="medium",
            discovery_scope="bounded").encode("utf-8")
    raise ValueError("unknown scratch route: " + repr(agent))


def arm_banner_environment_and_suffix(source=None):
    """Every role receives one binding budget, environment, and raw suffix."""
    with scratch_daemon(source=source) as (daemon, _, mailbox):
        daemon.MAX_CHARACTERS = 53
        daemon.agent_preamble = (
            lambda agent, message: "scratch role preamble " + agent + "\n")

        def ignore_cycle_registration(agent, message,
                                      return_reservation=False, **kwargs):
            """Keep this character-budget witness outside cycle Git state."""
            del agent, message, kwargs
            if return_reservation:
                return None, None, False
            return None, None

        daemon.register_ticket_cycle_message = ignore_cycle_registration
        calls = []
        daemon.subprocess.Popen = fake_popen(calls=calls)
        consumed = []
        bodies = {}
        for index, agent in enumerate(("fable", "opus", "sol"), start=1):
            body = pending_body(daemon=daemon, agent=agent)
            bodies[agent] = body
            path = mailbox / ("%04d-to-%s.md" % (index, agent))
            path.write_bytes(body)
            consumed.append(daemon.dispatch_under_main_checkout_lock(
                path=str(path), dry_run=False))

        observations = []
        for agent, call in zip(("fable", "opus", "sol"), calls):
            prompt = call["command"][-1]
            primary = daemon.AGENT_CWD["fable"]
            implementer = daemon.AGENT_CWD["opus"]
            observations.append(
                call["env"].get("MAILBOX_MAX_CHARACTERS") == "53"
                and call["env"].get("MAILBOX_PRIMARY_WORKTREE") == primary
                and call["env"].get("MAILBOX_IMPLEMENTER_WORKTREE")
                == implementer
                and call["env"].get("MAILBOX_EXECUTION_WORKTREE")
                == implementer
                and call["env"].get("MAILBOX_SHARED_NOTES")
                == str(pathlib.Path(primary) / "ai" / "notes")
                and call["env"].get("MAILBOX_HANDOFF_CONTRACT")
                == str(pathlib.Path(primary) / "ai" / "tools"
                       / "handoff_contract.py")
                and call["env"].get("MAILBOX_TICKET_CHANGE_GUARD")
                == str(pathlib.Path(primary) / "ai" / "tools"
                       / "ticket_change_guard.py")
                and prompt.count(
                    "--- TICKET CHARACTER BUDGET (binding) ---") == 1
                and "at most 53 characters added plus deleted" in prompt
                and "ticket_change_guard.py --repo EXECUTION_WORKTREE "
                "--base BASE --max 53" in prompt
                and "handoff_contract.py architect NOTE_ABSOLUTE_PATH "
                "--max 53" in prompt
                and "Over-limit, unmeasurable, or obfuscated work is "
                "NO-GO" in prompt
                and prompt.index("--- TICKET CHARACTER BUDGET")
                < prompt.index("scratch role preamble " + agent)
                < prompt.index("scratch common preamble")
                < prompt.index(bodies[agent].decode("utf-8"))
                and prompt.endswith(bodies[agent].decode("utf-8"))
                and prompt.encode("utf-8").endswith(bodies[agent]))
        passed = (
            consumed == [True, True, True]
            and len(calls) == 3 and all(observations))
    print("all roles receive budget/env/raw suffix=" + str(passed))
    return passed


def arm_sol_uses_primary_ticket_tools(source=None):
    """Sol is told to use primary tools while measuring the named checkout."""
    with scratch_daemon(source=source) as (daemon, _, _):
        prompt = daemon.agent_preamble(
            agent="sol", message=daemon.sol_ticket_payload(
                ticket_kind="discovery", text="Review one scratch fact.",
                discovery_severity="medium", discovery_scope="bounded"))
        primary = pathlib.Path(daemon.AGENT_CWD["fable"])
        sol = str(pathlib.Path(daemon.AGENT_CWD["sol"]))
        expected_tools = (
            str(primary / "ai" / "tools" / "handoff_contract.py"),
            str(primary / "ai" / "tools" / "ticket_change_guard.py"),
        )
        passed = (
            all(tool in prompt for tool in expected_tools)
            and "not relative copies in the Sol checkout" in prompt
            and "pass `--repo`" in prompt
            and "Never measure the Sol checkout" in prompt
            and all(not tool.startswith(sol) for tool in expected_tools))
    print("Sol uses primary ticket tools=" + str(passed))
    return passed


def arm_zero_banner(source=None):
    """Zero ships one no-limit block while retaining readability rules."""
    daemon = load_daemon(source=source)
    daemon.MAX_CHARACTERS = 0
    banner = daemon.dispatch_banner(
        store_max=1, newer_in_lane=0, previous_timeout_minutes=None)
    passed = (
        banner.count("--- TICKET CHARACTER BUDGET (binding) ---") == 1
        and "ticket limit: none (--max 0)" in banner
        and "readability, complete behavior" in banner
        and "obfuscated work is NO-GO" in banner
        and "handoff_contract.py architect NOTE_ABSOLUTE_PATH --max 0"
        in banner
        and "ticket_change_guard.py" not in banner)
    print("zero banner remains readability-bound=" + str(passed))
    return passed


def replace_exact(source, old, new):
    """Replace one production anchor, or return None when it drifted."""
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def arm_source_mutations():
    """Kill ignored CLI, banner, and child-environment propagation."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    cases = [
        (
            "numeric grammar accepts non-ASCII digits",
            lambda text: replace_exact(
                text,
                're.fullmatch(r"[0-9]+", value)',
                're.fullmatch(r"\\d+", value)'),
            arm_invalid_values_and_actions,
        ),
        (
            "CLI value ignored",
            lambda text: replace_exact(
                text,
                "    MAX_CHARACTERS = (DEFAULT_MAX_CHARACTERS\n"
                "                      if args.max_characters is None\n"
                "                      else args.max_characters)\n",
                "    MAX_CHARACTERS = DEFAULT_MAX_CHARACTERS\n"),
            arm_positive_once,
        ),
        (
            "banner reads default",
            lambda text: replace_exact(
                text,
                "    if MAX_CHARACTERS == 0:\n"
                "        lines.append(\n",
                "    if DEFAULT_MAX_CHARACTERS == 0:\n"
                "        lines.append(\n"),
            arm_banner_environment_and_suffix,
        ),
        (
            "child environment omitted",
            lambda text: replace_exact(
                text,
                "        env[MAX_CHARACTERS_ENVIRONMENT] = "
                "str(MAX_CHARACTERS)\n",
                ""),
            arm_banner_environment_and_suffix,
        ),
    ]
    failures = []
    for label, mutate, probe in cases:
        mutant = mutate(source)
        armed = mutant is not None and mutant != source
        baseline = probe(source) if armed else False
        mutant_passed = probe(mutant) if armed and baseline else True
        killed = armed and baseline and not mutant_passed
        print("MUTATION " + label + " armed=" + str(armed)
              + " baseline=" + str(baseline)
              + " killed=" + str(killed))
        if not killed:
            failures.append(label)
    print("mutation-summary killed=" + str(len(cases) - len(failures))
          + "/" + str(len(cases)) + " failures=" + repr(failures))
    return not failures


def main():
    """Run every isolated character-budget arm and source mutation."""
    arms = [
        ("default/zero", arm_default_and_explicit_zero),
        ("positive once", arm_positive_once),
        ("dry-run once", arm_dry_run_once),
        ("positive watch", arm_positive_watch),
        ("invalid values/actions", arm_invalid_values_and_actions),
        ("banner/environment/suffix", arm_banner_environment_and_suffix),
        ("Sol authoritative tools", arm_sol_uses_primary_ticket_tools),
        ("zero banner", arm_zero_banner),
        ("source mutations", arm_source_mutations),
    ]
    failures = []
    for name, arm in arms:
        try:
            passed = arm()
        except BaseException as exc:
            print("ERROR " + name + ": " + type(exc).__name__
                  + ": " + str(exc))
            passed = False
        print(("PASS " if passed else "FAIL ") + name)
        if not passed:
            failures.append(name)
    print("runtime-summary passed=" + str(len(arms) - len(failures))
          + "/" + str(len(arms)) + " failures=" + repr(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
