#!/usr/bin/env python3
"""Scratch-only witnesses for independent role models and providers.

The fable and opus filenames are stable legacy route addresses, not model
identities.  Every runtime arm loads a fresh daemon and redirects its paths to
a disposable repository.  Dry runs are additionally guarded by a Popen stub,
so this witness cannot launch Claude, Codex, or touch the live mailbox.
"""

import argparse
import contextlib
import io
import os
import pathlib
import re
import sys
import tempfile
import types


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
CLAUDE_BINARY = "/Users/vivianmiranda/.local/bin/claude"
OLLAMA_BINARY = "ollama"
ARCHITECT_DEFAULT = "claude-fable-5"
IMPLEMENTER_DEFAULT = "claude-opus-4-8"
CUSTOM_ARCHITECT = "opus"
CUSTOM_IMPLEMENTER = "sonnet"
SCRATCH_BASE = "1" * 40


class AttributeProxy:
    """Delegate module attributes except for explicit test overrides."""

    def __init__(self, base, **overrides):
        self._base = base
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._base, name)


def load_daemon(source=None):
    """Execute one fresh production module, optionally from a source mutant."""
    if source is None:
        source = DAEMON_PATH.read_text(encoding="utf-8")
    module = types.ModuleType("mailbox_daemon_role_models_repro")
    module.__file__ = str(DAEMON_PATH)
    exec(compile(source, str(DAEMON_PATH), "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def scratch_daemon(source=None):
    """Redirect a fresh daemon and every mailbox path into a temporary tree."""
    with tempfile.TemporaryDirectory(prefix="mailbox-role-models-") as tmp:
        root = pathlib.Path(tmp)
        ai_root = root / "ai"
        mailbox = ai_root / "notes" / "mailbox"
        mailbox.mkdir(parents=True)
        backlog = ai_root / "notes" / "backlog.md"
        backlog.write_text("", encoding="utf-8")

        daemon = load_daemon(source=source)
        daemon.REPO_ROOT = str(root)
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
        daemon.MAILBOX = str(mailbox)
        daemon.DONE = str(mailbox / "done")
        daemon.RELAY_DIR = str(ai_root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(backlog)
        daemon.PREAMBLE = "scratch role-model preamble\n"
        daemon.AGENT_CWD = {
            "fable": str(root / "shared-claude-lane"),
            "opus": str(root / "shared-claude-lane"),
            "sol": str(root / "sol-lane"),
        }
        daemon.warn_if_mailbox_unwatched = lambda: None
        daemon.report_demand = lambda backlog: None
        daemon.report_landing_debt = lambda: None
        yield daemon, root, mailbox


def tree_snapshot(root):
    """Return a byte-and-type snapshot of a disposable repository."""
    snapshot = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot.append((relative, "symlink", os.readlink(path)))
        elif path.is_file():
            snapshot.append((relative, "file", path.read_bytes()))
        elif path.is_dir():
            snapshot.append((relative, "dir", b""))
        else:
            snapshot.append((relative, "other", b""))
    return snapshot


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
        rc = error.code if isinstance(error.code, int) else 1
    elif error is None:
        rc = 0 if result is None else result
    else:
        rc = 1
    return rc, stdout.getvalue(), stderr.getvalue(), error


def model_in(command):
    """Return the argv value immediately following the exact --model flag."""
    index = command.index("--model")
    return command[index + 1]


def effort_in(command):
    """Return the argv value immediately following the exact --effort flag."""
    index = command.index("--effort")
    return command[index + 1]


def arm_each_dispatch_starts_fresh(source=None):
    """Keep old provider conversations out of every new mailbox turn."""
    daemon = load_daemon(source=source)
    commands = build_commands(daemon)
    claude_routes_are_fresh = all(
        "--no-session-persistence" in commands[route]
        and "--continue" not in commands[route]
        and "--resume" not in commands[route]
        for route in ("fable", "opus"))
    sol_is_fresh = (
        commands["sol"][:3]
        == [daemon.CODEX_EXECUTABLE, "exec", "--ephemeral"]
        and "resume" not in commands["sol"])
    passed = claude_routes_are_fresh and sol_is_fresh
    print("every dispatch starts fresh=" + str(passed))
    return passed


def build_commands(daemon, architect=CUSTOM_ARCHITECT,
                   implementer=CUSTOM_IMPLEMENTER):
    """Build one stable custom command set through the public API."""
    return daemon.build_agent_commands(
        "high", "max", "xhigh", 345678, architect, implementer)


def arm_defaults_and_validation(source=None):
    """Defaults stay compatible and malformed model arguments fail closed."""
    daemon = load_daemon(source=source)
    commands = daemon.build_agent_commands("low", "medium", "high", 123456)
    invalid = ["", " ", "\t", "claude opus", "bad\x00model", None]
    rejected = []
    for value in invalid:
        try:
            daemon.validate_model_name(value=value)
        except argparse.ArgumentTypeError:
            rejected.append(value)
    passed = (
        daemon.DEFAULT_ARCHITECT_MODEL == ARCHITECT_DEFAULT
        and daemon.DEFAULT_IMPLEMENTER_MODEL == IMPLEMENTER_DEFAULT
        and model_in(commands["fable"]) == ARCHITECT_DEFAULT
        and model_in(commands["opus"]) == IMPLEMENTER_DEFAULT
        and model_in(daemon.AGENT_COMMANDS["fable"]) == ARCHITECT_DEFAULT
        and model_in(daemon.AGENT_COMMANDS["opus"]) == IMPLEMENTER_DEFAULT
        and daemon.validate_model_name("opus") == "opus"
        and daemon.validate_model_name("claude-sonnet-4-6")
        == "claude-sonnet-4-6"
        and rejected == invalid)
    print("defaults and validation=" + str(passed))
    return passed


def arm_swapped_aliases_and_sol_stability(source=None):
    """Role aliases drive legacy routes while effort and Sol remain stable."""
    daemon = load_daemon(source=source)
    custom = build_commands(daemon)
    baseline = daemon.build_agent_commands(
        "high", "max", "xhigh", 345678)
    passed = (
        model_in(custom["fable"]) == CUSTOM_ARCHITECT
        and effort_in(custom["fable"]) == "high"
        and model_in(custom["opus"]) == CUSTOM_IMPLEMENTER
        and effort_in(custom["opus"]) == "max"
        and custom["sol"] == baseline["sol"]
        and set(custom) == {"fable", "opus", "sol"})
    print("role aliases and Sol stability=" + str(passed))
    return passed


def write_routing_messages(mailbox):
    """Publish one harmless dry-run message to each legacy route."""
    messages = {
        "fable": mailbox / "0001-to-fable.md",
        "opus": mailbox / "0002-to-opus.md",
        "sol": mailbox / "0003-to-sol.md",
    }
    messages["fable"].write_text(
        "audit the named scratch delta\n", encoding="utf-8", newline="")
    messages["opus"].write_text(
        "MAILBOX-FLOW: ticket\n"
        "MAILBOX-CYCLE: scratch-model-route@" + SCRATCH_BASE + "\n"
        "MAILBOX-MODE: normal\n\n"
        "implement the named scratch delta\n",
        encoding="utf-8", newline="")
    messages["sol"].write_text(
        "MAILBOX-TICKET: discovery\n"
        "MAILBOX-SEVERITY: medium\n"
        "MAILBOX-SCOPE: bounded\n\n"
        "review the scratch model routes\n",
        encoding="utf-8", newline="")
    return messages


def expected_claude_line(name, model, effort, cwd):
    """Return the daemon's exact dry-run line for one Claude route."""
    command = [CLAUDE_BINARY, "-p", "--no-session-persistence",
               "--model", model,
               "--effort", effort, "--permission-mode", "acceptEdits"]
    return ("[dry-run] would dispatch " + name + " -> "
            + " ".join(command) + "  (cwd " + cwd + ")")


def expected_ollama_line(name, model, cwd):
    """Return the daemon's dry-run line for an Ollama Implementer."""
    command = [
        OLLAMA_BINARY, "launch", "claude", "--model", model, "--yes", "--",
        "-p", "--no-session-persistence", "--permission-mode", "acceptEdits"]
    return ("[dry-run] would dispatch " + name + " -> "
            + " ".join(command) + "  (cwd " + cwd + ")")


def arm_ollama_implementer_cli(source=None):
    """A CLI provider choice changes only the Implementer command."""
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        messages = write_routing_messages(mailbox=mailbox)
        before = tree_snapshot(root)
        launches = []

        def forbidden_popen(*args, **kwargs):
            launches.append((args, kwargs))
            raise AssertionError("dry-run attempted to launch a child")

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=forbidden_popen)
        rc, output, error_output, error = call_main(daemon, [
            "--dry-run", "--implementer-provider", "ollama",
            "--implementer-model", "qwen3.5"])
        lines = output.splitlines()
        passed = (
            error is None and rc == 0 and error_output == ""
            and launches == [] and tree_snapshot(root) == before
            and expected_ollama_line(
                messages["opus"].name, "qwen3.5",
                daemon.AGENT_CWD["opus"]) in lines
            and daemon.AGENT_COMMANDS["fable"][0]
            == daemon.CLAUDE_EXECUTABLE
            and daemon.AGENT_COMMANDS["opus"][0]
            == daemon.OLLAMA_EXECUTABLE
            and "--effort" not in daemon.AGENT_COMMANDS["opus"])
        print("Ollama Implementer CLI=" + str(passed))
        return passed


def arm_cli_plumbing_and_legacy_routes(source=None):
    """Custom CLI flags reach exact legacy routes without launch or writes."""
    with scratch_daemon(source=source) as (daemon, root, mailbox):
        messages = write_routing_messages(mailbox=mailbox)
        before = tree_snapshot(root)
        launches = []

        def forbidden_popen(*args, **kwargs):
            launches.append((args, kwargs))
            raise AssertionError("dry-run attempted to launch a child")

        daemon.subprocess = AttributeProxy(
            daemon.subprocess, Popen=forbidden_popen)
        arguments = [
            "--dry-run",
            "--architect-model", CUSTOM_ARCHITECT,
            "--implementer-model", CUSTOM_IMPLEMENTER,
            "--fable-effort", "high",
            "--opus-effort", "max",
            "--sol-effort", "xhigh",
            "--sol-context", "345678",
        ]
        rc, output, error_output, error = call_main(daemon, arguments)
        fable_line = expected_claude_line(
            messages["fable"].name, CUSTOM_ARCHITECT, "high",
            daemon.AGENT_CWD["fable"])
        opus_line = expected_claude_line(
            messages["opus"].name, CUSTOM_IMPLEMENTER, "max",
            daemon.AGENT_CWD["opus"])
        fable_match = daemon.PENDING_MESSAGE_RE.match(
            messages["fable"].name)
        opus_match = daemon.PENDING_MESSAGE_RE.match(
            messages["opus"].name)
        sol_line = ("[dry-run] would dispatch " + messages["sol"].name
                    + " -> " + " ".join(daemon.AGENT_COMMANDS["sol"])
                    + "  (cwd " + daemon.AGENT_CWD["sol"] + ")")
        passed = (
            error is None and rc == 0 and error_output == ""
            and launches == [] and tree_snapshot(root) == before
            and fable_line in output.splitlines()
            and opus_line in output.splitlines()
            and sol_line in output.splitlines()
            and model_in(daemon.AGENT_COMMANDS["fable"])
            == CUSTOM_ARCHITECT
            and model_in(daemon.AGENT_COMMANDS["opus"])
            == CUSTOM_IMPLEMENTER
            and model_in(daemon.AGENT_COMMANDS["sol"]) == "gpt-5.6-sol"
            and fable_match is not None and fable_match.group(1) == "fable"
            and opus_match is not None and opus_match.group(1) == "opus")
        print("CLI plumbing and legacy routes=" + str(passed))
        return passed


def arm_invalid_cli_is_pre_dispatch(source=None):
    """Invalid role-model flags stop before backlog inspection or mutation."""
    invalid_cases = [
        ["--dry-run", "--architect-model", " "],
        ["--dry-run", "--architect-model", "bad\x00model"],
        ["--dry-run", "--implementer-model", "\t"],
        ["--dry-run", "--implementer-model", "bad model"],
    ]
    checks = []
    for arguments in invalid_cases:
        with scratch_daemon(source=source) as (daemon, root, mailbox):
            write_routing_messages(mailbox=mailbox)
            before = tree_snapshot(root)
            backlog_calls = []

            def forbidden_process(*args, **kwargs):
                backlog_calls.append((args, kwargs))
                raise AssertionError("invalid model reached backlog dispatch")

            daemon.process_backlog = forbidden_process
            rc, _, error_output, error = call_main(daemon, arguments)
            checks.append(
                isinstance(error, SystemExit) and rc == 2
                and "model must" in error_output
                and backlog_calls == [] and tree_snapshot(root) == before)
    passed = all(checks)
    print("invalid CLI pre-dispatch=" + str(passed))
    return passed


def replace_regex_once(source, pattern, replacement):
    """Replace one regex site or return None when the mutation is unarmed."""
    mutated, count = re.subn(pattern, replacement, source, count=1)
    if count != 1:
        return None
    return mutated


def mutate_route_model(source, route, old, new):
    """Replace one model expression inside one named command route."""
    pattern = (r'("' + re.escape(route)
               + r'"\s*:\s*\[[^\]]*?"--model"\s*,\s*)'
               + re.escape(old))
    return replace_regex_once(source, pattern, r"\1" + new)


def mutate_swapped_routes(source):
    """Swap Architect and Implementer variables across legacy routes."""
    first = mutate_route_model(
        source, "fable", "architect_model", "implementer_model")
    if first is None:
        return None
    return mutate_route_model(
        first, "opus", "implementer_model", "architect_model")


def arm_source_mutations():
    """Kill model re-hardcoding and ignored model/provider flags."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    cases = [
        (
            "Architect route re-hardcoded",
            lambda text: mutate_route_model(
                text, "fable", "architect_model",
                "DEFAULT_ARCHITECT_MODEL"),
            arm_swapped_aliases_and_sol_stability,
        ),
        (
            "Implementer route re-hardcoded",
            lambda text: replace_regex_once(
                text,
                r'("--model",\s*)implementer_model(,\s*"--effort")',
                r'\1DEFAULT_IMPLEMENTER_MODEL\2'),
            arm_swapped_aliases_and_sol_stability,
        ),
        (
            "Ollama provider ignored",
            lambda text: text.replace(
                'if implementer_provider == "claude":',
                'if True:', 1),
            arm_ollama_implementer_cli,
        ),
        (
            "Architect CLI flag ignored",
            lambda text: replace_regex_once(
                text,
                (r"architect_model=args\.architect_model,\n"
                 r"\s*implementer_model="),
                ("architect_model=DEFAULT_ARCHITECT_MODEL,\n"
                 "        implementer_model=")),
            arm_cli_plumbing_and_legacy_routes,
        ),
        (
            "Implementer CLI flag ignored",
            lambda text: replace_regex_once(
                text,
                (r"(# Rebuild the dispatch commands[\s\S]*?"
                 r"AGENT_COMMANDS = build_agent_commands\([\s\S]*?"
                 r"implementer_model=)args\.implementer_model"),
                r"\1DEFAULT_IMPLEMENTER_MODEL"),
            arm_cli_plumbing_and_legacy_routes,
        ),
        (
            "Implementer provider CLI flag ignored",
            lambda text: replace_regex_once(
                text,
                (r"(# Rebuild the dispatch commands[\s\S]*?"
                 r"AGENT_COMMANDS = build_agent_commands\([\s\S]*?"
                 r"implementer_provider=)args\.implementer_provider"),
                r"\1DEFAULT_IMPLEMENTER_PROVIDER"),
            arm_ollama_implementer_cli,
        ),
    ]
    failures = []
    for label, mutator, probe in cases:
        mutant = mutator(source)
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
    """Run all isolated role-model and mutation witnesses."""
    arms = [
        ("defaults/validation", arm_defaults_and_validation),
        ("aliases/Sol stability", arm_swapped_aliases_and_sol_stability),
        ("fresh provider contexts", arm_each_dispatch_starts_fresh),
        ("CLI/routes", arm_cli_plumbing_and_legacy_routes),
        ("Ollama Implementer CLI", arm_ollama_implementer_cli),
        ("invalid pre-dispatch", arm_invalid_cli_is_pre_dispatch),
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
