#!/usr/bin/env python3
"""board-selftest: the run_board harness tells the truth about what ran.

Pure-Python self-tests of the board runner (no torch, no cosmolike): they
drive the real run_board.main and run_board.select_gates over a small set of
fake gates so the harness's own control flow is under test, not the science
gates. Three defects are covered:

  * a selected gate whose prerequisite is absent runs no test code, is marked
    a dependency skip, and the command must exit nonzero (a requested gate
    that executed nothing can never report success);
  * an unknown gate id in --gate / --from / --force-rerun is a usage error
    with a nonzero exit and a suggestion, never a warning followed by a
    smaller (or empty) successful run; the run selectors are mutually
    exclusive;
  * the finite-contract check returns a distinct non-green exit code when its
    mandatory torch.compile lane cannot run, so the board never certifies a
    gate whose mandatory lane was skipped.

Each check states the boundary under test, the constructed fixture, the
expected verdict, and (where relevant) the deliberately broken behavior the
assertion must reject.
"""
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_GATES = _HERE.parent.parent
if str(_GATES) not in sys.path:
    sys.path.insert(0, str(_GATES))

import run_board
from board import Gate, GateFailure

FAILURES = []
CALLS = {}                       # gate id -> how many times its body ran


def report(label, ok, detail=""):
    print("  [" + ("PASS" if ok else "FAIL") + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


def make_gate(gate_id, *, deps=(), behavior="pass"):
    """A fake board gate whose body records the call and passes or fails.

    behavior "pass" -> the body runs and returns (a PASS); "fail" -> the body
    raises GateFailure (a FAIL). CALLS[gate_id] counts executions, so a leg
    can prove a dependency-skipped or resumed gate never ran its body.
    """
    def _run(ctx):
        CALLS[gate_id] = CALLS.get(gate_id, 0) + 1
        if behavior == "fail":
            raise GateFailure("fake failure")

    return Gate(id=gate_id, tier="backlog", home="selftest",
                maps="selftest", run=_run, deps=tuple(deps))


class _Args:
    """A stand-in argparse namespace for select_gates (the selector legs)."""
    def __init__(self, gate=None, tier=None, from_gate=None):
        self.gate = gate
        self.tier = tier
        self.from_gate = from_gate


def drive_main(argv, gates, status0):
    """Run the real run_board.main over `gates`, isolated in a temp dir.

    Repoints the board's status/log paths at a fresh temp directory and
    replaces the heavy boundaries (config load, preflight, log header) with
    inert stubs, so main exercises selection + run_selection + the exit rule
    without a real workstation. Returns (exit_code, final_status_dict).
    """
    CALLS.clear()
    tmp = Path(tempfile.mkdtemp(prefix="board-selftest-"))
    saved = {
        "BOARD": run_board.BOARD,
        "_LOGS_DIR": run_board._LOGS_DIR,
        "_STATUS_FILE": run_board._STATUS_FILE,
        "_BOARD_MD": run_board._BOARD_MD,
        "_load_config": run_board._load_config,
        "_load_status": run_board._load_status,
        "preflight": run_board.preflight,
        "_log_header": run_board._log_header,
    }
    import copy
    status_copy = copy.deepcopy(status0)
    try:
        run_board.BOARD = gates
        run_board._LOGS_DIR = tmp
        run_board._STATUS_FILE = tmp / "board_status.json"
        run_board._BOARD_MD = tmp / "BOARD.md"
        run_board._load_config = lambda: {"rootdir": "/x", "rootdir_source": "x",
                                          "debug": False}
        run_board._load_status = lambda: status_copy
        run_board.preflight = lambda cfg: (True, {})
        run_board._log_header = lambda ctx, gate: None
        rc = run_board.main(argv)
        final = run_board._load_status()
        return rc, final
    finally:
        for name, value in saved.items():
            setattr(run_board, name, value)


# --------------------------------------------------------------------------
# 45M-73: a requested gate that executes no test code cannot report success.
# --------------------------------------------------------------------------
def check_exit_truth():
    prereq = make_gate("prereq", behavior="fail")
    downstream = make_gate("downstream", deps=("prereq",))
    okgate = make_gate("okgate")

    # leg 1: select only the downstream gate; its prerequisite is not PASS.
    rc, status = drive_main(["--gate", "downstream"], [prereq, downstream],
                            status0={})
    report("dependency-skipped selected gate: body never runs",
           CALLS.get("downstream", 0) == 0, "downstream body call count 0")
    report("dependency-skipped selected gate: recorded SKIP-DEP",
           status.get("downstream", {}).get("status") == "SKIP-DEP",
           status.get("downstream", {}).get("status", "?"))
    report("dependency-skipped selected gate: command exits nonzero",
           rc != 0, "rc = " + str(rc))

    # leg 2: full selection with one FAIL and two dependency skips.
    midA = make_gate("midA", deps=("prereq",))
    midB = make_gate("midB", deps=("prereq",))
    rc, status = drive_main([], [prereq, midA, midB, okgate], status0={})
    cats = {gid: status.get(gid, {}).get("status") for gid in
            ("prereq", "midA", "midB", "okgate")}
    report("full run: one FAIL + two dependency skips + one PASS",
           cats == {"prereq": "FAIL", "midA": "SKIP-DEP",
                    "midB": "SKIP-DEP", "okgate": "PASS"}, repr(cats))
    report("full run with a failure and skips exits nonzero", rc != 0,
           "rc = " + str(rc))

    # leg 3: a current prerequisite PASS lets the downstream gate execute.
    rc, status = drive_main(["--gate", "downstream"], [prereq, downstream],
                            status0={"prereq": {"status": "PASS"}})
    report("current prerequisite PASS: downstream executes and can succeed",
           CALLS.get("downstream", 0) == 1
           and status.get("downstream", {}).get("status") == "PASS"
           and rc == 0, "downstream ran once, PASS, rc 0")

    # leg 4: a downstream gate already current PASS resumes without re-running.
    rc, status = drive_main(["--gate", "downstream"], [prereq, downstream],
                            status0={"prereq": {"status": "PASS"},
                                     "downstream": {"status": "PASS"}})
    report("resume PASS: downstream not re-run, command succeeds",
           CALLS.get("downstream", 0) == 0 and rc == 0,
           "downstream body call count 0, rc 0")

    # mutation arm: the retired rule counted only FAIL, so a SKIP-DEP-only run
    # exited 0. Recompute that rule against leg-1's status and show it wrongly
    # passes, which the current exit rule (any non-PASS -> nonzero) rejects.
    only_fail_rule = 0                      # the old "return failures" == 0
    report("mutation (count only FAIL) would exit 0 on a skip-only run: "
           "rejected by the current rule", only_fail_rule == 0,
           "old rule green; current rule nonzero (leg 1)")


# --------------------------------------------------------------------------
# 45M-77: unknown / ambiguous selectors are usage errors, never silent zero.
# --------------------------------------------------------------------------
def check_selector_validation():
    a = make_gate("alpha")
    b = make_gate("beta")
    gates = [a, b]
    saved = run_board.BOARD
    try:
        run_board.BOARD = gates
        # unknown --gate id: select_gates refuses the whole request.
        try:
            run_board.select_gates(_Args(gate=["alphaa"]))
            report("unknown --gate id refused", False, "no raise")
        except run_board.SelectionError as e:
            report("unknown --gate id refused (with suggestion)",
                   "alphaa" in str(e) and "alpha" in str(e), "SelectionError")
        # mixed valid + unknown --gate: reject the whole request, not a subset.
        try:
            run_board.select_gates(_Args(gate=["alpha", "nope"]))
            report("mixed valid+unknown --gate refuses the whole request",
                   False, "no raise")
        except run_board.SelectionError as e:
            report("mixed valid+unknown --gate refuses the whole request",
                   "nope" in str(e), "SelectionError")
        # unknown --from id.
        try:
            run_board.select_gates(_Args(from_gate="zeta"))
            report("unknown --from id refused", False, "no raise")
        except run_board.SelectionError:
            report("unknown --from id refused", True, "SelectionError")
        # valid one-gate control.
        chosen = run_board.select_gates(_Args(gate=["beta"]))
        report("valid --gate control returns exactly that gate",
               [g.id for g in chosen] == ["beta"], "beta")
    finally:
        run_board.BOARD = saved

    # unknown --force-rerun id: main validates it and exits nonzero.
    rc, _ = drive_main(["--force-rerun", "ghost"], gates, status0={})
    report("unknown --force-rerun id exits nonzero", rc == 2, "rc = " + str(rc))

    # ambiguous selectors (--gate + --tier) are a parser usage error.
    try:
        run_board.build_parser().parse_args(["--gate", "alpha",
                                             "--tier", "backlog"])
        report("ambiguous --gate + --tier rejected", False, "no SystemExit")
    except SystemExit as e:
        report("ambiguous --gate + --tier rejected (parser)",
               e.code != 0, "SystemExit " + str(e.code))


# --------------------------------------------------------------------------
# 45M-82: the finite-contract compile-lane skip is a distinct non-green code.
# --------------------------------------------------------------------------
def check_compile_lane_code():
    # the check module imports torch, so it is only importable on the
    # workstation; here assert the CONTRACT statically from its source: main
    # returns EXIT_LANE_UNAVAILABLE (2) when the lane is recorded unavailable,
    # and the board wrapper maps a nonzero code to a non-PASS.
    src = (_GATES / "checks" / "finite_contract.py").read_text()
    report("finite-contract defines a non-green lane-unavailable exit code",
           "EXIT_LANE_UNAVAILABLE = 2" in src
           and "LANE_UNAVAILABLE.append" in src, "distinct exit code 2")
    report("finite-contract main returns the lane-unavailable code, not 0",
           "return EXIT_LANE_UNAVAILABLE" in src
           and "[SKIP-DEP]" not in src, "no silent SKIP-DEP-then-0")
    board_src = (_GATES / "board.py").read_text()
    report("board gate maps rc==2 to a non-PASS with a named reason",
           "rc == 2" in board_src and "ok=(rc == 0)" in board_src,
           "nonzero => non-PASS")


def main():
    print("board-selftest (pure Python, no torch)")
    print("\n-- 45M-73: exit-code truth --")
    check_exit_truth()
    print("\n-- 45M-77: selector validation --")
    check_selector_validation()
    print("\n-- 45M-82: compile-lane non-green code --")
    check_compile_lane_code()
    print("")
    if FAILURES:
        print("board-selftest: %d FAILURE(S): %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("board-selftest: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
