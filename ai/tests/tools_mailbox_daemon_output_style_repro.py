#!/usr/bin/env python3
"""Keep daemon terminal prose and its README examples register-compliant."""

import ast
import contextlib
import importlib.util
import io
import os
import pathlib
import re
import tempfile
import types


AI_ROOT = pathlib.Path(__file__).resolve().parents[1]
DAEMON_PATH = AI_ROOT / "tools" / "mailbox_daemon.py"
README_PATH = AI_ROOT / "README.md"
ALLOWED_TERMINAL_ACRONYMS = {
    "AGENT", "CLI", "MINUTES", "NUL", "TOKENS", "UTF",
}


def load_daemon():
    """Load a fresh daemon module for an isolated output exercise."""
    spec = importlib.util.spec_from_file_location(
        "mailbox_daemon_output_style_repro", DAEMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def terminal_literal_violations(source):
    """Return register violations in static terminal-output strings.

    This scans prints plus argparse description/help strings, the two shipped
    terminal-prose surfaces. The dispatch PREAMBLE is agent-facing prompt text
    and was explicitly excluded from this ruling.
    """
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_print = (isinstance(node.func, ast.Name)
                    and node.func.id == "print")
        is_argparse_prose = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"ArgumentParser", "add_argument"})
        if not (is_print or is_argparse_prose):
            continue
        fragments = []
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) \
                    and isinstance(child.value, str):
                fragments.append(child.value)
        literal = "".join(fragments)
        if " -- " in literal:
            violations.append((node.lineno, "separator", literal))

        # The role filename is data, not emphasis. The remaining uppercase
        # words in shipped print literals must be genuine acronyms.
        prose = literal.replace(".claude/FABLE_ROLE.md", "")
        for word in re.findall(r"(?<![A-Za-z0-9])([A-Z]{2,})"
                               r"(?![A-Za-z0-9])", prose):
            if word not in ALLOWED_TERMINAL_ACRONYMS:
                violations.append((node.lineno, "emphasis:" + word,
                                   literal))
    return violations


@contextlib.contextmanager
def scratch_daemon():
    """Point a fresh daemon module at a disposable mailbox."""
    with tempfile.TemporaryDirectory(
            prefix="mailbox-daemon-output-style-") as tmp:
        root = pathlib.Path(tmp)
        ai_root = root / "ai"
        daemon = load_daemon()
        daemon.WORKTREE = str(root)
        daemon.AI_ROOT = str(ai_root)
        daemon.REPO_ROOT = str(root)
        daemon.MAILBOX = str(ai_root / "notes" / "mailbox")
        daemon.DONE = str(ai_root / "notes" / "mailbox" / "done")
        daemon.RELAY_DIR = str(ai_root / "notes" / "relay")
        daemon.BACKLOG_LEDGER = str(ai_root / "notes" / "backlog.md")
        daemon.PREAMBLE = "scratch message\n"
        daemon.AGENT_COMMANDS = {
            "fable": ["harmless-fable"],
            "opus": ["harmless-opus"],
            "sol": ["harmless-sol"],
        }
        daemon.AGENT_CWD = {
            "fable": str(root),
            "opus": str(root),
            "sol": str(root),
        }
        os.makedirs(daemon.MAILBOX, exist_ok=True)
        yield daemon, root


def runtime_hint_line():
    """Exercise the threshold hint and return the line that shipped."""
    daemon = load_daemon()
    daemon.backlog_ledger_count = lambda: daemon.SECOND_IMPLEMENTER_THRESHOLD
    daemon.report_landing_debt = lambda: None
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        daemon.report_demand(backlog=[])
    lines = [line for line in stream.getvalue().splitlines()
             if line.startswith("  hint: 10 or more items are waiting")]
    return lines[0] if len(lines) == 1 else ""


def runtime_refusal_line():
    """Exercise a real scratch refusal without touching the live mailbox."""
    with scratch_daemon() as (daemon, _):
        path = pathlib.Path(daemon.MAILBOX) / "0001-to-opus.md"
        path.write_text("<unit>\n", encoding="utf-8")
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            daemon.dispatch(path=str(path), dry_run=False)
        lines = [line for line in stream.getvalue().splitlines()
                 if line.startswith("refused ")]
        return lines[0] if len(lines) == 1 else ""


def runtime_heartbeat_line():
    """Drive one harmless dispatch far enough to print its heartbeat."""
    with scratch_daemon() as (daemon, root):
        path = pathlib.Path(daemon.MAILBOX) / "0046-to-opus.md"
        path.write_text("real scratch unit\n", encoding="utf-8")

        class FrozenDateTime:
            """Supply the timestamp used by the README's example."""

            @classmethod
            def now(cls):
                return types.SimpleNamespace(
                    strftime=lambda _format: "20260714-031840")

        class HarmlessProcess:
            """Look alive for one poll and then finish successfully."""

            def __init__(self):
                self.returncode = 0
                self.polls = 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 0

            def kill(self):
                self.returncode = -9

            def wait(self):
                return self.returncode

        times = iter([0.0, 180.0, 180.0, 180.0])

        def fake_popen(command, stdout, stderr, cwd, env):
            del command, stderr, cwd, env
            # Fill the relay log to exactly 12.4 kB after the daemon's header.
            # Newline-delimited bytes keep the reply-tail print compact.
            remaining = 12698 - stdout.tell()
            stdout.write("x\n" * (remaining // 2))
            if remaining % 2:
                stdout.write("x")
            stdout.flush()
            return HarmlessProcess()

        daemon.datetime = types.SimpleNamespace(datetime=FrozenDateTime)
        daemon.time = types.SimpleNamespace(
            time=lambda: next(times), sleep=lambda _seconds: None)
        daemon.subprocess.Popen = fake_popen
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            daemon.dispatch(path=str(path), dry_run=False)
        lines = [line for line in stream.getvalue().splitlines()
                 if " still running " in line]
        if len(lines) != 1:
            return ""
        return lines[0].replace(str(root), "...")


def main():
    """Run static, runtime, documentation, and mutation checks."""
    source = DAEMON_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")
    violations = terminal_literal_violations(source=source)

    hint = runtime_hint_line()
    expected_hint = (
        "  hint: 10 or more items are waiting. Give Sol separate "
        "implementation jobs as a second Implementer, but only a message "
        "with the required declaration changes Sol's role; otherwise Sol "
        "remains the Red Team.")
    refusal = runtime_refusal_line()
    expected_refusal = (
        "refused 0001-to-opus.md: the whole body is the template placeholder "
        "'<unit>'; parked in failed/; fill in the real text and requeue.")
    heartbeat = runtime_heartbeat_line()
    expected_heartbeat = (
        "  ... 0046-to-opus.md still running (3 min elapsed, log 12.4 kB; "
        "tail -f .../ai/notes/relay/20260714-031840-dispatch-opus.log)")

    separator_anchor = '"; parked in failed/."'
    separator_mutant = source.replace(
        separator_anchor, '" -- parked in failed/."', 1)
    separator_mutation_red = (
        separator_mutant != source
        and any(kind == "separator" for _, kind, _ in
                terminal_literal_violations(source=separator_mutant)))

    emphasis_anchor = 'print("refused " + name'
    emphasis_mutant = source.replace(
        emphasis_anchor, 'print("REFUSED " + name', 1)
    emphasis_mutation_red = (
        emphasis_mutant != source
        and any(kind == "emphasis:REFUSED" for _, kind, _ in
                terminal_literal_violations(source=emphasis_mutant)))

    checks = {
        "shipped terminal literals": not violations,
        "runtime refusal": refusal == expected_refusal,
        "runtime threshold hint": hint == expected_hint,
        "runtime heartbeat": heartbeat == expected_heartbeat,
        "README hint parity": readme.count(hint + "\n") == 1,
        "README heartbeat parity": readme.count(heartbeat + "\n") == 1,
        "separator mutation reds": separator_mutation_red,
        "all-caps mutation reds": emphasis_mutation_red,
    }
    for label, passed in checks.items():
        print(("PASS " if passed else "FAIL ") + label)
    if violations:
        for line, kind, literal in violations:
            print("  line " + str(line) + " " + kind + ": " + repr(literal))
    if not all(checks.values()):
        print("  observed refusal: " + repr(refusal))
        print("  observed hint: " + repr(hint))
        print("  observed heartbeat: " + repr(heartbeat))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
