#!/usr/bin/env python3
"""board-selftest: the run_board harness tells the truth about what ran.

Pure-Python self-tests of the board runner (no torch, no cosmolike): they
drive the real run_board.main / run_selection / select_gates over a small set
of fake gates so the harness's own control flow is under test, not the science
gates. Grouped by the defect each proves:

  exit-code truth -- a selected gate that runs no test code (a dependency
    skip) cannot report success; the command exits nonzero unless every
    selected gate is a current PASS.

  selector validation -- an unknown gate id in --gate / --from / --force-rerun
    is a usage error with a suggestion, never a warning-then-run; the run
    selectors are mutually exclusive.

  compile-lane code -- the finite-contract check exposes a distinct non-green
    exit code when its mandatory torch.compile lane cannot run.

  resume identity -- a stored PASS is trusted only when BOTH the gate's
    executable-surface digest and its input digest are unchanged; a
    configuration change, a mutated referenced YAML, or an interrupted attempt
    reruns the gate and never satisfies a dependency.

  evidence atomicity -- a RUNNING record is persisted before any gate code, so
    an interruption leaves an interrupted attempt (never the prior PASS); each
    attempt writes its own immutable log, so a rerun never truncates the
    evidence a prior PASS still cites; a successful run publishes a fresh log
    whose stored digest matches its bytes.

Each check states the boundary under test, the fixture, the expected verdict,
and (where relevant) the deliberately broken behavior it must reject.
"""
import copy
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

# the fixed fake configuration drive_main runs under; the digest of a PASS
# record must be computed against THIS cfg to read as current.
FAKE_CFG = {"rootdir": "/x", "rootdir_source": "x", "debug": False}


def report(label, ok, detail=""):
    print("  [" + ("PASS" if ok else "FAIL") + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


class _Interrupt(BaseException):
    """A stand-in for a KeyboardInterrupt / process kill: not an Exception, so
    run_selection does not catch it -- it propagates, leaving RUNNING."""


def make_gate(gate_id, *, deps=(), behavior="pass"):
    """A fake board gate whose body records the call and passes / fails / dies.

    behavior "pass" -> runs and returns (PASS); "fail" -> raises GateFailure;
    "interrupt" -> raises _Interrupt (an uncaught interruption). CALLS[gate_id]
    counts executions, so a leg can prove a skipped or resumed gate never ran.
    """
    def _run(ctx):
        CALLS[gate_id] = CALLS.get(gate_id, 0) + 1
        if behavior == "fail":
            raise GateFailure("fake failure")
        if behavior == "interrupt":
            raise _Interrupt("interrupted mid-gate")

    return Gate(id=gate_id, tier="backlog", home="selftest",
                maps="selftest", run=_run, deps=tuple(deps))


def pass_record(gate, cfg=None):
    """A current-PASS status record for `gate` under cfg (both digests set)."""
    use = FAKE_CFG if cfg is None else cfg
    return {"status": "PASS",
            "code_digest": run_board._gate_code_digest(gate),
            "input_digest": run_board._gate_input_digest(gate, use)}


class _Args:
    """A stand-in argparse namespace for select_gates (the selector legs)."""
    def __init__(self, gate=None, tier=None, from_gate=None):
        self.gate = gate
        self.tier = tier
        self.from_gate = from_gate


def drive_main(argv, gates, status0, cfg=None):
    """Run the real run_board.main over `gates`, isolated in a temp dir.

    Repoints the board's status/log paths at a fresh temp directory and
    replaces the heavy boundaries (config load, preflight, log header) with
    inert stubs, so main exercises selection + run_selection + the exit rule
    and the resume/atomicity machinery without a real workstation. Returns
    (exit_code, final_status_dict, temp_dir).
    """
    CALLS.clear()
    tmp = Path(tempfile.mkdtemp(prefix="board-selftest-"))
    use_cfg = FAKE_CFG if cfg is None else cfg
    saved = {name: getattr(run_board, name) for name in
             ("BOARD", "_LOGS_DIR", "_STATUS_FILE", "_BOARD_MD",
              "_load_config", "_load_status", "preflight", "_log_header")}
    status_copy = copy.deepcopy(status0)
    try:
        run_board.BOARD = gates
        run_board._LOGS_DIR = tmp
        run_board._STATUS_FILE = tmp / "board_status.json"
        run_board._BOARD_MD = tmp / "BOARD.md"
        run_board._load_config = lambda: dict(use_cfg)
        run_board._load_status = lambda: status_copy
        run_board.preflight = lambda cfg: (True, {})
        run_board._log_header = lambda ctx, gate: None
        try:
            rc = run_board.main(argv)
        except _Interrupt:
            rc = None                       # interrupted before returning
        final = run_board._load_status()
        return rc, final, tmp
    finally:
        for name, value in saved.items():
            setattr(run_board, name, value)


# --------------------------------------------------------------------------
# exit-code truth (a requested gate that runs no code cannot report success)
# --------------------------------------------------------------------------
def check_exit_truth():
    prereq = make_gate("prereq", behavior="fail")
    downstream = make_gate("downstream", deps=("prereq",))
    okgate = make_gate("okgate")

    rc, status, _ = drive_main(["--gate", "downstream"], [prereq, downstream], {})
    report("dependency-skipped selected gate: body never runs",
           CALLS.get("downstream", 0) == 0, "call count 0")
    report("dependency-skipped selected gate: recorded SKIP-DEP",
           status.get("downstream", {}).get("status") == "SKIP-DEP",
           status.get("downstream", {}).get("status", "?"))
    report("dependency-skipped selected gate: command exits nonzero",
           rc != 0, "rc = " + str(rc))

    midA = make_gate("midA", deps=("prereq",))
    midB = make_gate("midB", deps=("prereq",))
    rc, status, _ = drive_main([], [prereq, midA, midB, okgate], {})
    cats = {gid: status.get(gid, {}).get("status") for gid in
            ("prereq", "midA", "midB", "okgate")}
    report("full run: one FAIL + two dependency skips + one PASS",
           cats == {"prereq": "FAIL", "midA": "SKIP-DEP",
                    "midB": "SKIP-DEP", "okgate": "PASS"}, repr(cats))
    report("full run with a failure and skips exits nonzero", rc != 0,
           "rc = " + str(rc))

    okp = make_gate("prereq")                      # a passing prerequisite
    rc, status, _ = drive_main(["--gate", "downstream"], [okp, downstream],
                               {"prereq": pass_record(okp)})
    report("current prerequisite PASS: downstream executes and can succeed",
           CALLS.get("downstream", 0) == 1
           and status.get("downstream", {}).get("status") == "PASS"
           and rc == 0, "downstream ran once, PASS, rc 0")

    rc, status, _ = drive_main(["--gate", "downstream"], [okp, downstream],
                               {"prereq": pass_record(okp),
                                "downstream": pass_record(downstream)})
    report("resume PASS: downstream not re-run, command succeeds",
           CALLS.get("downstream", 0) == 0 and rc == 0, "call count 0, rc 0")


# --------------------------------------------------------------------------
# selector validation
# --------------------------------------------------------------------------
def check_selector_validation():
    a = make_gate("alpha")
    b = make_gate("beta")
    gates = [a, b]
    saved = run_board.BOARD
    try:
        run_board.BOARD = gates
        try:
            run_board.select_gates(_Args(gate=["alphaa"]))
            report("unknown --gate id refused", False, "no raise")
        except run_board.SelectionError as e:
            report("unknown --gate id refused (with suggestion)",
                   "alphaa" in str(e) and "alpha" in str(e), "SelectionError")
        try:
            run_board.select_gates(_Args(gate=["alpha", "nope"]))
            report("mixed valid+unknown --gate refuses the whole request",
                   False, "no raise")
        except run_board.SelectionError as e:
            report("mixed valid+unknown --gate refuses the whole request",
                   "nope" in str(e), "SelectionError")
        try:
            run_board.select_gates(_Args(from_gate="zeta"))
            report("unknown --from id refused", False, "no raise")
        except run_board.SelectionError:
            report("unknown --from id refused", True, "SelectionError")
        chosen = run_board.select_gates(_Args(gate=["beta"]))
        report("valid --gate control returns exactly that gate",
               [g.id for g in chosen] == ["beta"], "beta")
    finally:
        run_board.BOARD = saved

    rc, _, _ = drive_main(["--force-rerun", "ghost"], gates, {})
    report("unknown --force-rerun id exits nonzero", rc == 2, "rc = " + str(rc))
    try:
        run_board.build_parser().parse_args(["--gate", "alpha",
                                             "--tier", "backlog"])
        report("ambiguous --gate + --tier rejected", False, "no SystemExit")
    except SystemExit as e:
        report("ambiguous --gate + --tier rejected (parser)",
               e.code != 0, "SystemExit " + str(e.code))


# --------------------------------------------------------------------------
# compile-lane non-green code (static contract; the check imports torch)
# --------------------------------------------------------------------------
def check_compile_lane_code():
    src = (_GATES / "checks" / "finite_contract.py").read_text()
    report("finite-contract defines a non-green lane-unavailable exit code",
           "EXIT_LANE_UNAVAILABLE = 2" in src
           and "LANE_UNAVAILABLE.append" in src, "distinct exit code 2")
    report("finite-contract main returns the lane-unavailable code, not 0",
           "return EXIT_LANE_UNAVAILABLE" in src, "no silent skip-then-0")
    board_src = (_GATES / "board.py").read_text()
    report("board gate maps rc==2 to a non-PASS with a named reason",
           "rc == 2" in board_src and "ok=(rc == 0)" in board_src,
           "nonzero => non-PASS")


# --------------------------------------------------------------------------
# resume identity (both digests) + evidence atomicity
# --------------------------------------------------------------------------
def check_resume_identity():
    g = make_gate("g")

    # config A -> config B: a PASS current under config A reruns under config B.
    cfg_a = dict(FAKE_CFG, driver_fileroot="rootA")
    cfg_b = dict(FAKE_CFG, driver_fileroot="rootB")
    status0 = {"g": pass_record(g, cfg_a)}
    rc, status, _ = drive_main(["--gate", "g"], [g], status0, cfg=cfg_b)
    report("config change (A -> B) reruns a PASS current only under A",
           CALLS.get("g", 0) == 1 and status["g"]["status"] == "PASS"
           and rc == 0, "gate re-executed under config B")

    # a change to an explicitly excluded logging control (debug) reuses.
    cfg_dbg = dict(FAKE_CFG, debug=True)
    status0 = {"g": pass_record(g, FAKE_CFG)}
    rc, status, _ = drive_main(["--gate", "g"], [g], status0, cfg=cfg_dbg)
    report("debug-only change reuses the PASS (logging control excluded)",
           CALLS.get("g", 0) == 0 and rc == 0, "gate not re-run")

    # a mutated referenced YAML (same path, changed contents) reruns.
    ydir = Path(tempfile.mkdtemp(prefix="board-yaml-"))
    (ydir / "cfg.yaml").write_text("value: 1\n")
    cfg_y = dict(FAKE_CFG, yaml_dir=str(ydir))
    status0 = {"g": pass_record(g, cfg_y)}
    (ydir / "cfg.yaml").write_text("value: 2\n")       # same path, new contents
    rc, status, _ = drive_main(["--gate", "g"], [g], status0, cfg=cfg_y)
    report("mutated referenced YAML (same path) reruns the gate",
           CALLS.get("g", 0) == 1 and rc == 0, "content change re-executed")

    # a stored PASS whose code digest no longer matches is stale-code.
    stale = dict(pass_record(g), code_digest="deadbeef")
    report("a PASS with a mismatched code digest reads stale-code",
           run_board._resume_state({"g": stale}, g, FAKE_CFG) == "stale-code",
           "not current")


def check_evidence_atomicity():
    g = make_gate("g", behavior="interrupt")
    okg = make_gate("g")                              # same id, passing body

    # force-rerun a prior PASS, interrupt mid-gate: the stored state is a
    # non-PASS interrupted RUNNING, NOT the prior PASS.
    prior = pass_record(okg)
    prior["log"] = "g.OLD.log"
    status0 = {"g": prior}
    rc, status, tmp = drive_main(["--gate", "g", "--force-rerun", "g"],
                                 [g], status0)
    report("interrupt during a forced rerun leaves a non-PASS state",
           status["g"]["status"] == "RUNNING", status["g"]["status"])
    report("interrupted attempt writes its OWN log, not the prior PASS log",
           status["g"]["log"] != "g.OLD.log"
           and status["g"]["log"].endswith(".log"), status["g"]["log"])
    report("interrupted attempt is reported interrupted, never current PASS",
           run_board._resume_state(status, g, FAKE_CFG) == "interrupted",
           "interrupted")

    # a dependent gate refuses an interrupted prerequisite.
    down = make_gate("down", deps=("g",))
    report("a dependent gate refuses an interrupted prerequisite",
           not run_board._dep_current_pass(status, "g", FAKE_CFG),
           "prereq not current")

    # a successful (re)run publishes a fresh immutable log whose stored digest
    # matches its bytes, and a PASS record carrying both digests.
    rc, status, tmp = drive_main(["--gate", "g"], [okg], {})
    rec = status["g"]
    log_path = tmp / rec["log"]
    import hashlib
    matches = (log_path.is_file()
               and hashlib.sha256(log_path.read_bytes()).hexdigest()
               == rec.get("log_digest"))
    report("successful run: fresh immutable log, stored digest matches bytes",
           rec["status"] == "PASS" and matches
           and "code_digest" in rec and "input_digest" in rec,
           "PASS with both digests + matching log digest")

    # the status + board files are published atomically (temp + os.replace):
    # after a run the temp files are gone and the canonical files parse.
    leftovers = [p.name for p in tmp.iterdir()
                 if p.name.startswith(".board_status")
                 or p.name.endswith(".inprogress")]
    report("no temp status / in-progress log files leak after a run",
           leftovers == [], "clean temp dir")


def main():
    print("board-selftest (pure Python, no torch)")
    print("\n-- exit-code truth --")
    check_exit_truth()
    print("\n-- selector validation --")
    check_selector_validation()
    print("\n-- compile-lane non-green code --")
    check_compile_lane_code()
    print("\n-- resume identity (both digests) --")
    check_resume_identity()
    print("\n-- evidence atomicity (RUNNING + immutable logs) --")
    check_evidence_atomicity()
    print("")
    if FAILURES:
        print("board-selftest: %d FAILURE(S): %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("board-selftest: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
