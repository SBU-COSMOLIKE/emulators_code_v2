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

  raw-log trust -- a stored PASS is trusted only while the raw log it cites
    still verifies. Deleting, truncating, or editing that log, or a record
    that never stored a log digest, reads as stale-log through the same resume
    decision the runner and the board display consume, so the gate reruns
    rather than skipping on unverifiable evidence.

  structured evidence map -- validate_evidence resolves every gate's Assertion
    anchors against the notes/ markers and enforces board-wide id uniqueness,
    so a gate's maps= claim cannot drift from the note it cites. The shipped
    board validates (a live control), and a bad anchor, a missing note, a
    duplicate id, and a malformed anchor are each rejected.

Each check states the boundary under test, the fixture, the expected verdict,
and (where relevant) the deliberately broken behavior it must reject.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
_GATES = _HERE.parent.parent
if str(_GATES) not in sys.path:
    sys.path.insert(0, str(_GATES))

import run_board
import board
from board import BOARD, Assertion, Gate, GateFailure, Manifest, TIER_BACKLOG

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


def make_gate(gate_id, *, deps=(), behavior="pass", optional=False):
    """A fake board gate whose body records the call and passes / fails / dies.

    behavior "pass" -> runs and returns (PASS); "fail" -> raises GateFailure;
    "interrupt" -> raises _Interrupt (an uncaught interruption). CALLS[gate_id]
    counts executions, so a leg can prove a skipped or resumed gate never ran.
    optional -> the Gate.optional flag (off the default sweep unless named).
    """
    def _run(ctx):
        CALLS[gate_id] = CALLS.get(gate_id, 0) + 1
        if behavior == "fail":
            raise GateFailure("fake failure")
        if behavior == "interrupt":
            raise _Interrupt("interrupted mid-gate")

    return Gate(id=gate_id, tier="backlog", home="selftest",
                maps="selftest", run=_run, deps=tuple(deps), optional=optional)


# the byte content of the seed log a seeded current-PASS record cites. The
# runner now verifies a stored PASS against its raw log's digest, so a seeded
# PASS must name a log whose bytes exist and match; drive_main materializes
# this content into its temp log dir for every seeded PASS.
SEED_LOG_BYTES = b"seeded current-pass log\n"
# the attempt id every seeded PASS carries, so a seeded child's lineage snapshot
# (25M-26) matches its seeded dependency's attempt and the child reads current.
SEED_ATTEMPT = "seed-attempt"


def pass_record(gate, cfg=None):
    """A current-PASS status record for `gate` under cfg (all three digests set).

    Includes the raw-log evidence the runner now verifies: a seed log name and
    its digest over SEED_LOG_BYTES. drive_main writes that content into its temp
    log dir, so the record reads as a genuine current PASS (not one flagged
    stale-log for a missing or unverifiable log). A gate WITH dependencies also
    carries a lineage snapshot (25M-26) whose stored attempt matches its seeded
    dependency's SEED_ATTEMPT, so the child reads current rather than
    stale-dependency.
    """
    use = FAKE_CFG if cfg is None else cfg
    log_digest = hashlib.sha256(SEED_LOG_BYTES).hexdigest()
    record = {"status": "PASS",
              "code_digest": run_board._gate_code_digest(gate),
              "input_digest": run_board._gate_input_digest(gate, use),
              "log": gate.id + ".seed.log",
              "log_digest": log_digest,
              "attempt": SEED_ATTEMPT}
    if gate.deps:
        record["deps"] = {dep: {"attempt": SEED_ATTEMPT,
                                "log_digest": log_digest}
                          for dep in gate.deps}
    return record


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
        # materialize each seeded current-PASS record's cited log into the temp
        # log dir, so the runner's raw-log digest check sees a genuine current
        # PASS. A seeded record that deliberately cites a different log (e.g. an
        # old-log fixture) is left alone.
        for rec in status_copy.values():
            if (rec.get("status") == "PASS"
                    and rec.get("log", "").endswith(".seed.log")):
                (tmp / rec["log"]).write_bytes(SEED_LOG_BYTES)
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

    # 25M-24: action modes are STANDALONE and validate their run controls.
    # These drive the REAL main() so main()'s action-mode ORDERING is under
    # test -- the ordering the select_gates-only legs above never exercised
    # (why board-selftest stayed green against this defect).
    rc, _, _ = drive_main(["--list", "--gate", "nope"], gates, {})
    report("--list with an unknown --gate id exits nonzero (25M-24)",
           rc == 2, "rc = " + str(rc))
    rc, _, _ = drive_main(["--list", "--from", "nope"], gates, {})
    report("--list with an unknown --from id exits nonzero",
           rc == 2, "rc = " + str(rc))
    rc, _, _ = drive_main(["--list", "--force-rerun", "nope"], gates, {})
    report("--list with an unknown --force-rerun id exits nonzero",
           rc == 2, "rc = " + str(rc))
    rc, _, _ = drive_main(["--list", "--check"], gates, {})
    report("--list --check (incompatible actions) exits nonzero",
           rc == 2, "rc = " + str(rc))
    rc, _, _ = drive_main(["--list"], gates, {})
    report("a clean --list still exits 0 (regression guard)",
           rc == 0, "rc = " + str(rc))
    rc, _, _ = drive_main(["--check"], gates, {})
    report("a clean --check still exits 0 (regression guard)",
           rc == 0, "rc = " + str(rc))

    # item-7 completion (25M-24): an action mode ignores run controls, and a
    # VALID ignored control fails just as an unknown one does (the pre-merge
    # audit's gap 1). Real main(), red-capable against the un-fixed code (which
    # returned 0); the clean --list / --check controls above pin the contrast.
    rc, _, _ = drive_main(["--list", "--force-rerun", "beta"], gates, {})
    report("--list with a VALID but ignored --force-rerun exits 2 (item 7)",
           rc == 2, "rc = " + str(rc))
    rc, _, _ = drive_main(["--check", "--gate", "beta"], gates, {})
    report("--check with an ignored --gate selection exits 2 (item 7)",
           rc == 2, "rc = " + str(rc))
    rc, _, _ = drive_main(["--list", "--dry-run"], gates, {})
    report("--list with an ignored --dry-run exits 2 (item 7)",
           rc == 2, "rc = " + str(rc))

    # item-7 completion (25M-24 rider, bcf4ce2): an explicit --force-rerun id
    # OUTSIDE the selected surface is a usage error, not a silent discard.
    rc, _, _ = drive_main(["--gate", "alpha", "--force-rerun", "beta"], gates, {})
    report("--force-rerun of a gate outside the selection exits 2 (item 7)",
           rc == 2, "rc = " + str(rc))
    # control: the SAME force-rerun id INSIDE the selection is accepted -- proof
    # the rc-2 is caused by being outside, not by --force-rerun itself.
    rc, _, _ = drive_main(["--gate", "beta", "--force-rerun", "beta",
                           "--dry-run"], gates, {})
    report("--force-rerun of a gate inside the selection is accepted (control)",
           rc == 0, "rc = " + str(rc))

    # 25M-25: --from an OPTIONAL start includes it FIRST; a later optional
    # gate stays excluded. Pin the exact id list.
    from_gates = [make_gate("head"), make_gate("opt-start", optional=True),
                  make_gate("mid"), make_gate("opt-late", optional=True),
                  make_gate("tail")]
    saved = run_board.BOARD
    try:
        run_board.BOARD = from_gates
        chosen = run_board.select_gates(_Args(from_gate="opt-start"))
        report("--from an optional start: it is included and first (25M-25)",
               [g.id for g in chosen] == ["opt-start", "mid", "tail"],
               ", ".join(g.id for g in chosen))
        chosen = run_board.select_gates(_Args(from_gate="mid"))
        report("--from a non-optional start: unchanged tail (regression)",
               [g.id for g in chosen] == ["mid", "tail"],
               ", ".join(g.id for g in chosen))
    finally:
        run_board.BOARD = saved
    rc, _, _ = drive_main(["--from", "opt-start", "--dry-run"], from_gates, {})
    report("--from an optional start drives main() to a clean dry-run",
           rc == 0, "rc = " + str(rc))


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
def check_dependency_currency():
    """25M-20 (the unit-4 reopen): resume must not bypass dependency currency.

    run_selection's resume skip runs before the dependency loop, so a gate whose
    OWN stored PASS is current used to resume-skip even when a prerequisite was
    stale / failed / interrupted -- a green child hanging off a non-current
    parent. The reusable-PASS predicate now also requires every dependency to be
    a current PASS that was not itself rerun this run. Each prerequisite state
    must make the current-PASS child non-green.
    """
    prereq = make_gate("prereq")
    child = make_gate("child", deps=("prereq",))

    # baseline: both current PASS, prerequisite not rerun -> child resumes.
    both = {"prereq": pass_record(prereq), "child": pass_record(child)}
    rc, status, _ = drive_main([], [prereq, child], both)
    report("dep-currency: both current PASS -> child resumes, no body runs",
           CALLS.get("prereq", 0) == 0 and CALLS.get("child", 0) == 0
           and status["child"]["status"] == "PASS" and rc == 0,
           "prereq/child calls " + str(CALLS.get("prereq", 0)) + "/"
           + str(CALLS.get("child", 0)) + ", rc " + str(rc))

    # THE resume-before-deps mutation: a stale-code prerequisite + a current-PASS
    # child. Before the fix the child resume-skipped (body never ran); now the
    # reran prerequisite reruns the child so it reads the fresh output.
    stale = {"prereq": dict(pass_record(prereq), code_digest="deadbeef"),
             "child": pass_record(child)}
    rc, status, _ = drive_main([], [prereq, child], stale)
    report("dep-currency: a stale prerequisite reruns its current-PASS child",
           CALLS.get("prereq", 0) >= 1 and CALLS.get("child", 0) >= 1
           and status["child"]["status"] == "PASS",
           "prereq/child calls " + str(CALLS.get("prereq", 0)) + "/"
           + str(CALLS.get("child", 0)))

    # an interrupted (RUNNING) prerequisite is also not current -> it reruns, and
    # its current-PASS child reruns with it.
    running = {"prereq": dict(pass_record(prereq), status="RUNNING"),
               "child": pass_record(child)}
    rc, status, _ = drive_main([], [prereq, child], running)
    report("dep-currency: an interrupted (RUNNING) prerequisite reruns its child",
           CALLS.get("child", 0) >= 1 and status["child"]["status"] == "PASS",
           "child calls " + str(CALLS.get("child", 0)))

    # a FAILED prerequisite is not current -> the current-PASS child is a
    # dependency skip, never a resumed pass, and the run exits nonzero.
    failing = make_gate("prereq", behavior="fail")
    rc, status, _ = drive_main([], [failing, child], {"child": pass_record(child)})
    report("dep-currency: a FAILED prerequisite skip-deps its current-PASS child",
           status.get("child", {}).get("status") == "SKIP-DEP"
           and CALLS.get("child", 0) == 0 and rc != 0,
           "child status " + str(status.get("child", {}).get("status"))
           + ", calls " + str(CALLS.get("child", 0)) + ", rc " + str(rc))


def check_config_readers():
    """25M-22: every non-documentation board_config key has a Python reader.

    A documented control no code reads is a standing lie -- saved_emulator_root
    was one (removed). This census re-derives the public key set from the
    shipped board_config.json and refuses any key that no repo .py source
    mentions, so no future dead key can accumulate.
    """
    cfg = json.loads(run_board._CONFIG_FILE.read_text())
    public = [k for k in cfg if not k.startswith("_")]
    blob = []
    for py in run_board._REPO.rglob("*.py"):
        if "__pycache__" in str(py) or "/.git/" in str(py):
            continue
        try:
            blob.append(py.read_text(errors="replace"))
        except OSError:
            pass
    text = "\n".join(blob)
    dead = [k for k in public if k not in text]
    report("config census: every public board_config key has a Python reader",
           not dead, "keys no .py mentions: " + repr(dead))


def check_dependency_topology():
    """25M-20 rider: BOARD lists every gate's dependencies BEFORE the gate.

    The resume dependency-currency fix relies on prerequisites being processed
    before their children (a reran prerequisite is in the reran set by the time
    the child is reached). Authoring order is a correctness invariant of the
    resume machinery, so the board asserts it.
    """
    order = {g.id: i for i, g in enumerate(BOARD)}
    bad = []
    for g in BOARD:
        for dep in g.deps:
            if dep not in order or order[dep] >= order[g.id]:
                bad.append(g.id + " <- " + dep)
    report("topology: every gate's dependencies precede it in BOARD",
           not bad, "out-of-order or missing: " + repr(bad))


def check_digest_projection():
    """25M-21: a _help (documentation) edit leaves the input digest fixed; a
    value edit to an execution-relevant key stales it.
    """
    g = _mf_gate("proj", _mf_body_trivial, code=())
    cfg = json.loads(run_board._CONFIG_FILE.read_text())
    base = run_board._gate_input_digest(g, cfg)
    doc = copy.deepcopy(cfg)
    doc["_help"]["driver_root"] = "EDITED DOCUMENTATION PROSE"
    val = copy.deepcopy(cfg)
    val["driver_root"] = "edited-value"
    report("digest projection: a _help prose edit leaves the input digest fixed",
           run_board._gate_input_digest(g, doc) == base,
           "documentation (_help) is excluded from the execution projection")
    report("digest projection: an execution-value edit stales the input digest",
           run_board._gate_input_digest(g, val) != base,
           "a driver_root value change reruns the consuming gates")


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


def check_log_trust():
    """A stored PASS is trusted only while its cited raw log verifies.

    Drives the REAL resume decision (run_board._resume_state) -- the same
    function the runner's skip and the board display both consume -- over a
    genuine current PASS and then over a deleted, truncated, edited, and
    digest-missing log. Only the untouched control reads PASS; every altered or
    absent log reads stale-log, and a stale-log record is not a current
    dependency. This closes the reopen where deleting or editing a raw log left
    the status a current PASS and a rerun skipped it.
    """
    g = make_gate("g")
    tmp = Path(tempfile.mkdtemp(prefix="board-logtrust-"))
    saved = run_board._LOGS_DIR
    try:
        run_board._LOGS_DIR = tmp
        rec = pass_record(g)                       # cites g.seed.log + its digest
        log_path = tmp / rec["log"]

        # valid control: the cited log exists and its bytes match the digest.
        log_path.write_bytes(SEED_LOG_BYTES)
        state = run_board._resume_state({"g": rec}, g, FAKE_CFG)
        report("valid unchanged log reads current PASS", state == "PASS", state)

        # truncation: fewer bytes than the digest was taken over.
        log_path.write_bytes(SEED_LOG_BYTES[:5])
        state = run_board._resume_state({"g": rec}, g, FAKE_CFG)
        report("a truncated raw log reads stale-log", state == "stale-log", state)

        # a stale-log PASS is not a current dependency (same decision).
        dep_current = run_board._dep_current_pass({"g": rec}, "g", FAKE_CFG)
        report("a stale-log prerequisite is not current", not dep_current,
               "not current")

        # deletion: the cited log is gone.
        log_path.unlink()
        state = run_board._resume_state({"g": rec}, g, FAKE_CFG)
        report("a deleted raw log reads stale-log", state == "stale-log", state)

        # missing stored digest: a PASS that never recorded a log digest, even
        # with the file present, cannot be verified and is stale.
        rec_nodig = dict(pass_record(g))
        rec_nodig.pop("log_digest")
        (tmp / rec_nodig["log"]).write_bytes(SEED_LOG_BYTES)
        state = run_board._resume_state({"g": rec_nodig}, g, FAKE_CFG)
        report("a PASS with no stored log digest reads stale-log",
               state == "stale-log", state)

        # mutation arm: the digest is load-bearing. A valid log reads PASS; the
        # same record after a byte edit reads stale-log. Retaining the old
        # code (digest ignored on the skip path) would keep both PASS.
        log_path.write_bytes(SEED_LOG_BYTES)
        good = run_board._resume_state({"g": rec}, g, FAKE_CFG)
        log_path.write_bytes(b"tampered\n")
        bad = run_board._resume_state({"g": rec}, g, FAKE_CFG)
        report("the log digest is load-bearing (tamper flips PASS -> stale-log)",
               good == "PASS" and bad == "stale-log", good + " -> " + bad)
    finally:
        run_board._LOGS_DIR = saved


def _evidence_gate(gate_id, aid, anchor):
    """A fake gate carrying one structured evidence assertion (no body run)."""
    def _run(ctx):
        pass

    return Gate(id=gate_id, tier="backlog", home="selftest", maps="selftest",
                run=_run, evidence=(Assertion(aid, anchor),))


def check_evidence_map():
    """validate_evidence resolves anchors + enforces id uniqueness, and the
    real board validates (every migrated anchor already resolves)."""
    # control: the shipped board's structured evidence map validates -- every
    # migrated gate's anchor resolves to a real <a id> marker in notes/, and
    # no two assertions share an id. This is the leg that would go red if a
    # future note rewording orphaned a live anchor.
    ok, errs = run_board.validate_evidence(BOARD)
    report("the shipped board's evidence map validates", ok,
           "all anchors resolve, ids unique" if ok else "; ".join(errs))

    # a migrated gate really does carry structured evidence (the foundation is
    # not vacuous): at least the seven red-team gates have a populated map.
    n_evid = sum(1 for gate in BOARD if gate.evidence)
    report("the board carries migrated structured evidence", n_evid >= 7,
           "%d gate(s) with evidence" % n_evid)

    # mutation 1: an anchor whose marker is not declared in the note is caught.
    # The aid->anchor transform is satisfied (bad.leg -> bad-leg), so the
    # unresolved marker is the only defect this leg isolates.
    bad = _evidence_gate("bad", "bad.leg", "gates-and-board.md#bad-leg")
    ok, errs = run_board.validate_evidence([bad])
    report("an unresolved anchor marker is rejected", not ok,
           errs[0] if errs else "")

    # mutation 2: an anchor naming a note that does not exist is caught. The
    # transform is satisfied (miss.leg -> miss-leg), so the missing note is the
    # only defect.
    miss = _evidence_gate("miss", "miss.leg", "no-such-note.md#miss-leg")
    ok, errs = run_board.validate_evidence([miss])
    report("an anchor citing a missing note is rejected", not ok,
           errs[0] if errs else "")

    # mutation 3: two gates sharing an assertion id is caught (board-wide
    # uniqueness, so a leg's id names exactly one leg). The shared aid's anchor
    # resolves and satisfies the transform, so the duplicate is the only defect.
    real_anchor = "gates-and-board.md#board-selftest-exit-truth"
    dup_a = _evidence_gate("dupA", "board-selftest.exit-truth", real_anchor)
    dup_b = _evidence_gate("dupB", "board-selftest.exit-truth", real_anchor)
    ok, errs = run_board.validate_evidence([dup_a, dup_b])
    report("a duplicate assertion id across gates is rejected", not ok,
           errs[0] if errs else "")

    # mutation 4: an anchor missing the '#<marker>' shape is caught.
    mal = _evidence_gate("mal", "mal.leg", "gates-and-board.md")
    ok, errs = run_board.validate_evidence([mal])
    report("a malformed anchor (no #marker) is rejected", not ok,
           errs[0] if errs else "")

    # mutation 5 (invariant 3): an anchor that resolves to a REAL marker but is
    # not the aid with '.'->'-' is caught -- the marker exists, so this isolates
    # the aid<->anchor transform violation (the second naming convention this
    # rollout kills).
    xform = _evidence_gate("xform", "xform.leg",
                          "gates-and-board.md#board-selftest-exit-truth")
    ok, errs = run_board.validate_evidence([xform])
    has_transform_err = False
    for e in errs:
        if "not the aid" in e:
            has_transform_err = True
    report("a non-transform anchor (marker != aid) is rejected",
           (not ok) and has_transform_err, errs[0] if errs else "")

    # the CLI refuses to LIST a board whose evidence violates the transform:
    # validate_evidence runs on every invocation, so the REAL main --list exits
    # 2 before any gate runs (the whole CLI path, not just the predicate).
    rc, _, _ = drive_main(["--list"], [xform], {})
    report("a non-transform anchor makes --list exit 2", rc == 2,
           "rc=" + str(rc))

    # control: a gate with no evidence is never itself a failure (the
    # migration is rolling, not a flag day).
    ok, errs = run_board.validate_evidence([make_gate("plain")])
    report("a gate with no evidence is not a failure", ok and errs == [],
           "empty evidence tolerated")


def _reconcile_gate(gate_id, aids):
    """A fake gate declaring `aids` as its evidence (bodies are irrelevant to
    _reconcile_evidence, which reads only gate.evidence and the executed list)."""
    def _run(ctx):
        pass
    ev = []
    for aid in aids:
        ev.append(Assertion(aid, "n.md#" + aid.replace(".", "-")))
    return Gate(id=gate_id, tier="backlog", home="selftest", maps="s",
                run=_run, evidence=tuple(ev))


def check_evidence_reconciliation():
    """run_board._reconcile_evidence enforces binding ruling 6 + fork D1-ii.

    Drives the REAL reconcile predicate: every declared leg emits exactly one
    terminal (PASS / UNAVAILABLE; a FAIL raised and never reaches here), the
    gate passes on its available legs, and a gate that proved nothing (every
    declared leg UNAVAILABLE) may not pass -- the dead-network rule turned on
    the harness itself.
    """
    # control: declared == executed, every leg a real PASS -> green, and the
    # pinned summary reports 2/2 with zero unavailable.
    g = _reconcile_gate("ctrl", ["ctrl.a", "ctrl.b"])
    ok, ev, line = run_board._reconcile_evidence(
        g, [("ctrl.a", "PASS", ""), ("ctrl.b", "PASS", "")])
    report("clean declared==executed reconciles green", ok and ev is not None,
           line)
    report("the pinned summary counts executed/declared",
           ev == {"executed": 2, "declared": 2, "unavailable": []}, repr(ev))

    # a DECLARED leg the body never emitted (silently dropped) reds the gate.
    g = _reconcile_gate("drop", ["drop.a", "drop.b"])
    ok, ev, line = run_board._reconcile_evidence(g, [("drop.a", "PASS", "")])
    report("a declared-not-executed leg reds the gate",
           (not ok) and "declared-not-executed: drop.b" in line, line)

    # the same id emitted twice (two terminals for one leg) reds the gate.
    g = _reconcile_gate("dupe", ["dupe.a"])
    ok, ev, line = run_board._reconcile_evidence(
        g, [("dupe.a", "PASS", ""), ("dupe.a", "PASS", "")])
    report("a leg emitted twice reds the gate",
           (not ok) and "emitted-twice: dupe.a" in line, line)

    # an emitted id the gate never declared reds the gate.
    g = _reconcile_gate("unk", ["unk.a"])
    ok, ev, line = run_board._reconcile_evidence(
        g, [("unk.a", "PASS", ""), ("unk.b", "PASS", "")])
    report("an executed-not-declared id reds the gate",
           (not ok) and "executed-not-declared: unk.b" in line, line)

    # fork D1-ii: a gate that PROVED one leg and marked another UNAVAILABLE
    # passes on its available legs, and the summary names the owed leg.
    g = _reconcile_gate("mix", ["mix.a", "mix.b"])
    ok, ev, line = run_board._reconcile_evidence(
        g, [("mix.a", "PASS", ""), ("mix.b", "UNAVAILABLE", "owed-workstation")])
    report("a mixed PASS + UNAVAILABLE gate passes on its available legs",
           ok and ev == {"executed": 1, "declared": 2,
                         "unavailable": [["mix.b", "owed-workstation"]]},
           repr(ev))

    # zero-executed guard: a gate whose every declared leg is UNAVAILABLE proved
    # nothing and MAY NOT pass.
    g = _reconcile_gate("void", ["void.a", "void.b"])
    ok, ev, line = run_board._reconcile_evidence(
        g, [("void.a", "UNAVAILABLE", "owed"), ("void.b", "UNAVAILABLE", "owed")])
    report("an all-UNAVAILABLE gate may not PASS (zero-executed guard)",
           (not ok) and "zero executed legs" in line, line)


def check_evidence_gate_verdict():
    """The run_selection HOOK applies reconciliation to the REAL verdict.

    Drives run_board.run_selection through drive_main (validate_evidence patched
    off, so fabricated anchors need not resolve -- anchor resolution is covered
    by check_evidence_map). A mixed gate ends PASS with the pinned evidence block
    persisted; an all-UNAVAILABLE gate and a silently-dropped-leg gate each end
    FAIL, never a bare green.
    """
    saved_ve = run_board.validate_evidence
    run_board.validate_evidence = lambda gates: (True, [])
    try:
        # a gate that emits one PASS + one UNAVAILABLE leg: ends PASS, and the
        # persisted record carries the pinned executed/UNAVAILABLE block.
        def _mix_run(ctx):
            ctx.expect(aid="mixg.a", label="a", ok=True, detail="")
            ctx.unavailable(aid="mixg.b", label="b", reason="owed-workstation")
        mixg = Gate(id="mixg", tier="backlog", home="selftest", maps="s",
                    run=_mix_run,
                    evidence=(Assertion("mixg.a", "n.md#mixg-a"),
                              Assertion("mixg.b", "n.md#mixg-b")))
        rc, final, tmp = drive_main(["--gate", "mixg"], [mixg], {})
        rec = final.get("mixg", {})
        report("a mixed PASS+UNAVAILABLE gate ends PASS via the real runner",
               rec.get("status") == "PASS", repr(rec.get("status")))
        report("the passing gate persists the pinned evidence block",
               rec.get("evidence") == {"executed": 1, "declared": 2,
                                       "unavailable": [["mixg.b",
                                                        "owed-workstation"]]},
               repr(rec.get("evidence")))

        # a gate that marks BOTH declared legs UNAVAILABLE proved nothing: the
        # real runner records FAIL, never a green (the zero-executed guard).
        def _void_run(ctx):
            ctx.unavailable(aid="voidg.a", label="a", reason="owed")
            ctx.unavailable(aid="voidg.b", label="b", reason="owed")
        voidg = Gate(id="voidg", tier="backlog", home="selftest", maps="s",
                     run=_void_run,
                     evidence=(Assertion("voidg.a", "n.md#voidg-a"),
                               Assertion("voidg.b", "n.md#voidg-b")))
        rc, final, tmp = drive_main(["--gate", "voidg"], [voidg], {})
        rec = final.get("voidg", {})
        report("an all-UNAVAILABLE gate ends FAIL via the real runner (not PASS)",
               rec.get("status") == "FAIL", repr(rec.get("status")))

        # a gate that silently drops a declared leg reds through the real runner
        # (the body passes its one emitted leg, but the second is never emitted).
        def _drop_run(ctx):
            ctx.expect(aid="dropg.a", label="a", ok=True, detail="")
        dropg = Gate(id="dropg", tier="backlog", home="selftest", maps="s",
                     run=_drop_run,
                     evidence=(Assertion("dropg.a", "n.md#dropg-a"),
                               Assertion("dropg.b", "n.md#dropg-b")))
        rc, final, tmp = drive_main(["--gate", "dropg"], [dropg], {})
        rec = final.get("dropg", {})
        report("a silently-dropped declared leg reds via the real runner",
               rec.get("status") == "FAIL", repr(rec.get("status")))
    finally:
        run_board.validate_evidence = saved_ve


def check_aid_manifest():
    """run_check folds a check script's ##AID per-leg manifest into the executed
    set (binding ruling 6 + the one-verdict constraint).

    Drives the REAL parser, the REAL run_check subprocess, and the REAL
    run_selection over tiny temp check scripts: a script's legs fold and reconcile
    green; a script that drops a declared leg, crashes before its manifest, or
    prints an unparseable line each red the gate; and a leg counted by BOTH the
    script and the gate body reds (no second parallel verdict).
    """
    # unit: the parser extracts (aid, result, reason), keeping the reason for an
    # UNAVAILABLE leg and ignoring non-manifest output.
    recs, mal = run_board._parse_aid_manifest(
        "noise\n##AID g.a PASS\n##AID g.b UNAVAILABLE owed to box\nmore\n")
    report("the manifest parser extracts (aid, result, reason)",
           recs == [("g.a", "PASS", ""), ("g.b", "UNAVAILABLE", "owed to box")]
           and mal == [], repr(recs))
    recs, mal = run_board._parse_aid_manifest("##AID g.a MAYBE\n##AID g.b\n")
    report("the parser flags malformed manifest lines", recs == [] and len(mal) == 2,
           repr(mal))

    saved_ve = run_board.validate_evidence
    run_board.validate_evidence = lambda gates: (True, [])
    with tempfile.TemporaryDirectory(prefix="board-selftest-aid-") as sdir:
        sd = Path(sdir)
        try:
            # (a) a script emitting both declared legs -> folded -> gate PASS.
            ok_s = sd / "ok.py"
            ok_s.write_text("print('##AID okg.a PASS')\nprint('##AID okg.b PASS')\n")

            def _ok_run(ctx):
                ctx.run_check(str(ok_s))
            okg = Gate(id="okg", tier="backlog", home="selftest", maps="s",
                       run=_ok_run,
                       evidence=(Assertion("okg.a", "n.md#okg-a"),
                                 Assertion("okg.b", "n.md#okg-b")))
            rc, final, tmp = drive_main(["--gate", "okg"], [okg], {})
            report("a script's ##AID legs fold into the executed set (gate PASS)",
                   final.get("okg", {}).get("status") == "PASS",
                   repr(final.get("okg", {}).get("status")))

            # (b) a script that DROPS a declared leg -> reconciliation reds.
            drop_s = sd / "drop.py"
            drop_s.write_text("print('##AID dpg.a PASS')\n")

            def _dp_run(ctx):
                ctx.run_check(str(drop_s))
            dpg = Gate(id="dpg", tier="backlog", home="selftest", maps="s",
                       run=_dp_run,
                       evidence=(Assertion("dpg.a", "n.md#dpg-a"),
                                 Assertion("dpg.b", "n.md#dpg-b")))
            rc, final, tmp = drive_main(["--gate", "dpg"], [dpg], {})
            report("a script that drops a declared leg reds the gate",
                   final.get("dpg", {}).get("status") == "FAIL",
                   repr(final.get("dpg", {}).get("status")))

            # (c) a script that CRASHES before its manifest -> the declared leg is
            # never emitted -> reconciliation reds (no headline rc-check needed).
            crash_s = sd / "crash.py"
            crash_s.write_text("import sys\nsys.exit(3)\n")

            def _cr_run(ctx):
                ctx.run_check(str(crash_s))
            crg = Gate(id="crg", tier="backlog", home="selftest", maps="s",
                       run=_cr_run,
                       evidence=(Assertion("crg.a", "n.md#crg-a"),))
            rc, final, tmp = drive_main(["--gate", "crg"], [crg], {})
            report("a script crash before its manifest reds the gate",
                   final.get("crg", {}).get("status") == "FAIL",
                   repr(final.get("crg", {}).get("status")))

            # (d) an unparseable ##AID line -> run_check raises -> gate FAIL.
            mal_s = sd / "mal.py"
            mal_s.write_text("print('##AID malg.a PROBABLY')\n")

            def _ml_run(ctx):
                ctx.run_check(str(mal_s))
            mlg = Gate(id="mlg", tier="backlog", home="selftest", maps="s",
                       run=_ml_run,
                       evidence=(Assertion("malg.a", "n.md#malg-a"),))
            rc, final, tmp = drive_main(["--gate", "mlg"], [mlg], {})
            report("an unparseable ##AID manifest line reds the gate",
                   final.get("mlg", {}).get("status") == "FAIL",
                   repr(final.get("mlg", {}).get("status")))

            # (e) one-verdict: a script emits the leg AND the body also expects the
            # same aid -> emitted twice -> reds (no second parallel verdict).
            dv_s = sd / "dv.py"
            dv_s.write_text("print('##AID dvg.a PASS')\n")

            def _dv_run(ctx):
                ctx.run_check(str(dv_s))
                ctx.expect(aid="dvg.a", label="dup", ok=True, detail="")
            dvg = Gate(id="dvg", tier="backlog", home="selftest", maps="s",
                       run=_dv_run,
                       evidence=(Assertion("dvg.a", "n.md#dvg-a"),))
            rc, final, tmp = drive_main(["--gate", "dvg"], [dvg], {})
            report("a leg counted by both the script and the body reds (one-verdict)",
                   final.get("dvg", {}).get("status") == "FAIL",
                   repr(final.get("dvg", {}).get("status")))

            # (f) the composition hole (increment-2 audit ab07a2e): a script that
            # prints '##AID <aid> FAIL' but EXITS 0 passes the wrapper's rc==0
            # expect, so the passing path sees a FOLDED FAIL -- it must red, not
            # be relabeled UNAVAILABLE.
            fail0_s = sd / "fail0.py"
            fail0_s.write_text("print('##AID failg.a PASS')\n"
                               "print('##AID failg.b FAIL')\n")

            def _fl_run(ctx):
                rc, out = ctx.run_check(str(fail0_s))
                ctx.expect(label="ran (rc==0)", ok=(rc == 0), detail="rc=" + str(rc))
            failg = Gate(id="failg", tier="backlog", home="selftest", maps="s",
                         run=_fl_run,
                         evidence=(Assertion("failg.a", "n.md#failg-a"),
                                   Assertion("failg.b", "n.md#failg-b")))
            rc, final, tmp = drive_main(["--gate", "failg"], [failg], {})
            report("a folded FAIL while the wrapper rc==0 expect passes reds the gate",
                   final.get("failg", {}).get("status") == "FAIL",
                   repr(final.get("failg", {}).get("status")))

            # (g) control: both legs PASS under the same wrapper rc==0 expect -> green.
            pass0_s = sd / "pass0.py"
            pass0_s.write_text("print('##AID passg.a PASS')\n"
                               "print('##AID passg.b PASS')\n")

            def _ps_run(ctx):
                rc, out = ctx.run_check(str(pass0_s))
                ctx.expect(label="ran (rc==0)", ok=(rc == 0), detail="rc=" + str(rc))
            passg = Gate(id="passg", tier="backlog", home="selftest", maps="s",
                         run=_ps_run,
                         evidence=(Assertion("passg.a", "n.md#passg-a"),
                                   Assertion("passg.b", "n.md#passg-b")))
            rc, final, tmp = drive_main(["--gate", "passg"], [passg], {})
            report("an all-PASS manifest under the wrapper expect stays green (control)",
                   final.get("passg", {}).get("status") == "PASS",
                   repr(final.get("passg", {}).get("status")))
        finally:
            run_board.validate_evidence = saved_ve


class _GoldenCtx:
    """A stub ctx that drives the REAL board._golden_leg over controlled child
    (rc, output) pairs, so the empty-selection / crashed-child mutations red
    through the production leg itself (DIDACTICS-46 + the rc addendum).

    _golden_leg calls run_driver twice: first the current tree, then the pinned
    worktree. This returns _cur on the first call and _pre on the second, and
    captures the leg's single ctx.expect verdict in .result = (ok, detail).
    """

    def __init__(self, *, pre, cur, pre_rc=0, cur_rc=0, base="deadbeef"):
        self._pre = (pre_rc, pre)
        self._cur = (cur_rc, cur)
        self._base = base
        self.dry = False
        self.result = None
        self._n = 0

    def golden_base(self, gate_id):
        return self._base                # a configured base, or None (null-base path)

    def config_yaml_name(self, name):
        return "c.yaml"

    def require_config(self, key):
        return "c.yaml"

    def log(self, msg):
        pass

    @contextlib.contextmanager
    def staged_golden(self, *, gate_id, source):
        yield "bare.yaml"

    @contextlib.contextmanager
    def worktree(self, *, commit):
        yield "/tmp/wt"

    def run_driver(self, *, yaml_path, cwd=None):
        self._n = self._n + 1
        return self._cur if self._n == 1 else self._pre

    def expect(self, *, label, ok, detail="", aid=None):
        self.result = (ok, detail)

    def unavailable(self, *, aid, label, reason):
        self.result = ("UNAVAILABLE", reason)


def check_golden_leg():
    """board._golden_leg reds on a crashed child or an empty selection, not only
    on a diff (DIDACTICS-46 + the rc addendum). Drives the REAL leg via _GoldenCtx.

    The pre-46 leg discarded both child rcs and compared whatever the pattern
    selected, so a nonzero-rc child after its matching lines, or a pattern that
    matched nothing on both sides, passed vacuously. These arms hold the fix.
    """
    def _drive(**kw):
        c = _GoldenCtx(**kw)
        board._golden_leg(c, "g", "^epoch", yaml_name="c.yaml")
        return c.result

    same = "epoch 1 loss 0.5\nepoch 2 loss 0.3\n"

    # control: rc0/rc0 + identical non-empty selection -> green.
    ok, detail = _drive(pre=same, cur=same)
    report("golden control: clean rcs + identical non-empty selection passes",
           ok, detail)

    # the original equality check still bites: a diverging line reds.
    ok, detail = _drive(pre=same, cur="epoch 1 loss 0.5\nepoch 2 loss 0.9\n")
    report("golden: a diverging selection reds", not ok, detail)

    # the 46 defect: neither side has an 'epoch' line -> empty selection must
    # red, not pass vacuously.
    ok, detail = _drive(pre="hello\nworld\n", cur="hello\nworld\n")
    report("golden: an empty selection reds (not a vacuous pass)", not ok, detail)

    # the rc addendum: both children exit 1 after identical matching lines.
    ok, detail = _drive(pre=same, cur=same, pre_rc=1, cur_rc=1)
    report("golden: both children rc 1 reds despite matching lines", not ok,
           detail)

    # the rc addendum: a tip-only nonzero child rc.
    ok, detail = _drive(pre=same, cur=same, cur_rc=1)
    report("golden: a tip-only nonzero child rc reds", not ok, detail)

    # queue-2 migration (loss-schema-equivalence): a NULL golden base with a
    # DECLARED aid emits an explicit UNAVAILABLE (fork D1-ii) -- not a crash and
    # not a silent skip that would red reconciliation as declared-not-executed.
    c = _GoldenCtx(pre=same, cur=same, base=None)
    board._golden_leg(c, "g", "^epoch", yaml_name="c.yaml", aid="g.golden")
    report("golden: a null base with a declared aid emits UNAVAILABLE",
           c.result is not None and c.result[0] == "UNAVAILABLE", repr(c.result))


def check_dirty_watch():
    """1c-bis: the clean-tree watch parses porcelain per line, immune to the
    global strip, and the pathspec + exclusion + surface text share one owner.

    Drives the REAL run_board._dirty_lines / _git / _watched_paths. A porcelain
    line is 'XY <path>' -- two status columns, a space, then the path -- so the
    path is line[3:] only when the transport kept the leading column. A global
    strip drops the FIRST line's leading space and shifts its path by one, which
    is exactly how the portable config escaped its exclusion when it was the
    head (only or alphabetically first) dirty entry.
    """
    exclude = run_board._WATCH_EXCLUDE
    cfg_line = " M " + exclude               # raw porcelain, config as head line
    nbr_line = " M gates/board.py"           # a watched neighbor

    # (1) config-only, config as the head line: dropped -> clean. The reopened
    #     head-line case, on the raw porcelain the fixed transport now delivers.
    report("config-only edit (head line) stays clean",
           run_board._dirty_lines(cfg_line) == [],
           "head-line case: " + repr(cfg_line))
    # (2) config + neighbor: only the neighbor reds; the config is still dropped
    #     even though it is the alphabetically first / head line.
    both = run_board._dirty_lines(cfg_line + "\n" + nbr_line)
    report("config + neighbor edit reds ONLY the neighbor",
           both == [nbr_line], "offenders = " + repr(both))
    # (3) neighbor-only: reds.
    report("neighbor-only edit reds",
           run_board._dirty_lines(nbr_line) == [nbr_line], "offender kept")
    # (4) clean-tree control.
    report("empty porcelain is clean",
           run_board._dirty_lines("") == [], "no offenders")
    # (5) mutation arm: restore the head-line misparse by stripping the leading
    #     column (what the retired global-strip transport did to the first
    #     line). The config then escapes exclusion and false-reds -- the failure
    #     the fix removes, so the mutant must produce exactly that offender.
    mutated = cfg_line.strip()               # "M gates/board_config.json"
    misparsed = run_board._dirty_lines(mutated)
    report("mutation (stripped head line) false-reds the config -> caught",
           misparsed == [mutated],
           "line[3:]=" + repr(mutated[3:]) + " != exclude, so it leaks")
    # transport: strip=False leaves git's bytes untouched (the property the
    # per-line parse relies on); strip=True trims, for the single-value callers.
    _rc, raw = run_board._git(["rev-parse", "HEAD"], strip=False)
    _rc2, trimmed = run_board._git(["rev-parse", "HEAD"])
    report("_git(strip=False) preserves the raw transport",
           raw.endswith("\n") and trimmed == raw.strip() and trimmed != raw,
           "raw keeps the trailing newline; strip=True trims it")
    # one owner: the pathspec covers the executable surface + a root driver, and
    # the excluded config lives inside it (so _dirty_lines, not the pathspec,
    # drops it), and the exclusion string is the same constant every consumer
    # reads.
    watched = run_board._watched_paths()
    owner_ok = (all(d in watched for d in run_board._EXECUTABLE_DIRS)
                and any(p.endswith(".py") for p in watched)
                and exclude.split("/")[0] in run_board._EXECUTABLE_DIRS)
    report("one owner: pathspec covers the surface + drivers; config lives in it",
           owner_ok, "watched dirs + root *.py; exclude under the surface")


# Fabricated gate bodies for the manifest census. These are never CALLED --
# validate_manifests reads their SOURCE (inspect.getsource), so an undefined
# name inside is fine; the source is the fixture.
def _mf_body_undeclared_target(ctx):
    """Launches a subprocess .py the manifest does not declare (a REAL repo
    module, so declaring it is a valid root that clears the census; it is
    dynamic-import clean, so declaring it adds no other requirement)."""
    ctx.run_check("gates/checks/board_selftest.py")   # a real check, auto-covered
    _spawn("emulator/data_staging.py")                # a real .py, undeclared -> reds


def _mf_body_launches_driver(ctx):
    """Launches the default training driver (run_driver -> _DRIVER)."""
    ctx.run_driver(yaml_path="x")


def _mf_body_trivial(ctx):
    """No subprocess targets; used to isolate the dynamic-import census."""
    return None


def _mf_body_phantom_py_tokens(ctx):
    """A ".py" appears only inside longer tokens: the ctx.python attribute and a
    .pyc name. The word-boundary census must lift neither as a target."""
    ctx.sh(cmd=[ctx.python, "-c", "print(1)"])   # the interpreter, not a file
    _cache = "board_selftest.pyc"                 # a compiled-cache name


def _mf_body_sentence_final_py(ctx):
    """Names a real, undeclared repo module at a sentence end. The census still
    catches it: emulator/data_staging.py. ends the path at the period."""
    ctx.run_check("gates/checks/board_selftest.py")   # auto-covered check


def _mf_gate(gid, body, code, inputs=()):
    return Gate(id=gid, tier=TIER_BACKLOG, home="gates-and-board", maps="",
                run=body, manifest=Manifest(code=tuple(code), inputs=tuple(inputs)))


# a minimal resolved board_config for the manifest legs: validate_manifests
# resolves declared input keys against it. The reconciliation fixtures declare
# no inputs, so only the root/closure checks run there.
_MF_CFG = {"rootdir": None, "yaml_dir": None}


def check_manifest_reconciliation():
    """1b phase 1: validate_manifests catches the two under-declarations the
    scan is blind to -- an uncovered subprocess target and an unwaived (or
    uncovered) dynamic import -- and clears a fully-covered declaration.

    Drives the REAL run_board.validate_manifests over fabricated gates whose
    bodies are real source (so inspect.getsource reads them) and whose declared
    roots are real repo modules (so the closure and the dynamic-import census
    run against real files). A gate with no manifest is skipped; the populated
    live BOARD instead reconciles against the real board_config.
    """
    # live board: the populated manifests must all reconcile against the REAL
    # board_config (declared input keys resolve there); a manifest-less gate is
    # still skipped. _MF_CFG below drives only the fabricated fixtures, which
    # declare no inputs.
    live_cfg = run_board._load_config()
    declared = sum(1 for g in BOARD if g.manifest is not None)
    ok, errs = run_board.validate_manifests(BOARD, live_cfg)
    report("live BOARD manifests validate against board_config", ok,
           str(declared) + " gate(s) declared; errors=" + str(errs[:3]))

    # (a) literal-path census: an undeclared subprocess target reds.
    g = _mf_gate("mf-target", _mf_body_undeclared_target, code=())
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("uncovered subprocess target reds",
           (not ok) and any("uncovered subprocess target 'emulator/data_staging"
                            in e for e in errs),
           "the undeclared .py the import graph never sees")
    # declaring it clears the census (the real check script stays auto-covered;
    # data_staging.py exists and is dynamic-clean, so the declaration is valid).
    g = _mf_gate("mf-target-ok", _mf_body_undeclared_target,
                 code=("emulator/data_staging.py",))
    report("declaring the target clears the census",
           run_board.validate_manifests([g], _MF_CFG)[0], "target now a root")
    # the run_driver default driver is a target too.
    g = _mf_gate("mf-driver", _mf_body_launches_driver, code=())
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("run_driver's default driver is an uncovered target when undeclared",
           (not ok) and any(run_board._DRIVER in e for e in errs),
           "run_driver -> " + run_board._DRIVER)
    g = _mf_gate("mf-driver-ok", _mf_body_launches_driver,
                 code=(run_board._DRIVER,))
    _ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("declaring the driver clears the literal census",
           not any("uncovered subprocess target" in e for e in errs),
           "no literal error once the driver is a declared root (its closure's "
           "model-recipe imports still want a designs root, correctly)")

    # (a') word-boundary: a ".py" inside a longer token is not a phantom target
    # (ctx.python must not read as ctx.py, and .pyc/.pyx never match), while a
    # genuine sentence-final mention is still caught.
    g = _mf_gate("mf-phantom", _mf_body_phantom_py_tokens, code=())
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("word-boundary census: ctx.python / a .pyc are not phantom targets",
           ok,
           "no uncovered-target error from ctx.python or a .pyc name; errors="
           + str(errs))
    g = _mf_gate("mf-sentence-final", _mf_body_sentence_final_py, code=())
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("word-boundary census: a real sentence-final .py mention still reds",
           (not ok) and any("emulator/data_staging.py" in e for e in errs),
           "the period ends the path, so the undeclared module is still an "
           "uncovered target")

    # (b) dynamic-import census over the derived closure. results.py is a
    # waived file (the model-recipe pattern); its covering roots are the
    # design / loss trees.
    g = _mf_gate("mf-dyn", _mf_body_trivial, code=("emulator/results.py",))
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("waived dynamic import with NO covering root declared reds",
           (not ok) and any("covering root" in e for e in errs),
           "declares results.py but neither designs nor losses")

    # (25M-18) coverage is ALL-quantified over the required covers, and a root
    # covers a cover only by being that cover or an ANCESTOR of it. results.py's
    # waiver requires BOTH emulator/designs AND emulator/losses.
    # child-as-cover (the pre-25M-18 blessing fixture, now flipped to must-red):
    # a file INSIDE the tree does not satisfy the tree waiver.
    g = _mf_gate("mf-dyn-child", _mf_body_trivial,
                 code=("emulator/results.py", "emulator/designs/blocks.py",
                       "emulator/losses"))
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("25M-18 child-as-cover reds: a file inside the tree is not the tree",
           (not ok) and any("emulator/designs" in e and "covering root" in e
                            for e in errs),
           "designs/blocks.py does not cover the emulator/designs waiver")
    # strip-one-of-two: one satisfied cover does not clear a multi-cover waiver.
    g = _mf_gate("mf-dyn-strip", _mf_body_trivial,
                 code=("emulator/results.py", "emulator/designs"))
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("25M-18 strip-one-of-two reds: the second cover is still required",
           (not ok) and any("emulator/losses" in e for e in errs),
           "designs declared, losses missing -> losses uncovered")
    # the green control: declaring BOTH full trees clears the census.
    g = _mf_gate("mf-dyn-ok", _mf_body_trivial,
                 code=("emulator/results.py", "emulator/designs",
                       "emulator/losses"))
    report("declaring every covering root (full trees) clears the census",
           run_board.validate_manifests([g], _MF_CFG)[0],
           "designs AND losses declared")

    # any-one-of-eight (25M-18): cli-strict's cli_strict.py waiver lists eight
    # driver entry points; dropping any ONE leaves that driver uncovered. Uses
    # the LIVE gate + config (the real eight-cover waiver), the 40/40 audit probe.
    live_probe_cfg = run_board._load_config()
    cli = [g for g in BOARD if g.id == "cli-strict"][0]
    kept = tuple(r for r in cli.manifest.code
                 if r != "scalar_train_emulator.py")
    probe = Gate(id="cli-strict-probe", tier=cli.tier, home=cli.home,
                 maps=cli.maps, run=cli.run,
                 manifest=Manifest(code=kept, inputs=cli.manifest.inputs))
    ok, errs = run_board.validate_manifests([probe], live_probe_cfg)
    report("25M-18 any-one-of-eight reds: dropping one waiver entry-point",
           (not ok) and any("scalar_train_emulator.py" in e for e in errs),
           "the cli_strict waiver's eight covers are all-required")

    # mutation arm: empty the reviewed waiver table -> the same dynamic site is
    # now unreviewed and must red (a NEW dynamic import cannot slip in unwaived).
    saved = run_board._DYNAMIC_IMPORT_WAIVERS
    try:
        run_board._DYNAMIC_IMPORT_WAIVERS = {}
        ok, errs = run_board.validate_manifests([g], _MF_CFG)
        report("mutation (empty waiver table) reds the now-unwaived dynamic site",
               (not ok) and any("unwaived dynamic-import" in e for e in errs),
               "an unreviewed importlib site fails")
    finally:
        run_board._DYNAMIC_IMPORT_WAIVERS = saved

    # determinism (delta 3): the derived closure is a set, independent of
    # traversal order; the same seeds give the same members.
    a = run_board._derive_closure({"emulator/experiment.py"})
    b = run_board._derive_closure({"emulator/experiment.py"})
    report("derived closure is deterministic (order-independent)",
           a == b and len(a) > 1, str(len(a)) + " members, stable")


def check_runtime_loader_census():
    """1b hardening (25M-16): the runtime-loader census (c). A gate that loads an
    adapter by FILE PATH (importlib spec_from_file_location) or a Cobaya
    python_path component must DECLARE the loaded .py, so an edit to the adapter
    reruns the gate rather than escaping the digest. Drives the REAL
    validate_manifests over live gates, a reviewed-table mutation, and the
    bare-sibling resolver.
    """
    live_cfg = run_board._load_config()
    si = [g for g in BOARD if g.id == "scalar-identity"][0]
    # positive strip: scalar-identity minus its cobaya_theory/emul_scalars root
    # still reaches the spec_from_file_location site but no longer covers it.
    stripped = Gate(id=si.id, tier=si.tier, home=si.home, maps=si.maps,
                    run=si.run,
                    manifest=Manifest(code=("emulator/designs", "emulator/losses"),
                                      inputs=si.manifest.inputs))
    ok, errs = run_board.validate_manifests([stripped], live_cfg)
    report("25M-16 positive: an identity gate missing its adapter root reds",
           (not ok) and any("emul_scalars" in e for e in errs),
           "the loaded adapter must be a declared root")
    report("25M-16 the populated identity gate clears census (c)",
           run_board.validate_manifests([si], live_cfg)[0],
           "emul_scalars declared -> covered")

    # negative mutation: drop cmb_identity from the reviewed table -> its
    # spec_from_file_location site is now unreviewed and reds (a NEW adapter
    # loader in an unlisted file cannot slip in undeclared).
    saved = run_board._RUNTIME_LOADER_COVERS
    try:
        table = dict(saved)
        table.pop("gates/checks/cmb_identity.py")
        run_board._RUNTIME_LOADER_COVERS = table
        ci = [g for g in BOARD if g.id == "cmb-identity"][0]
        ok, errs = run_board.validate_manifests([ci], live_cfg)
        report("25M-16 negative: an UNLISTED runtime-loader site reds",
               (not ok) and any("unreviewed runtime-loader" in e for e in errs),
               "the spec_from_file_location site has no reviewed cover")
    finally:
        run_board._RUNTIME_LOADER_COVERS = saved

    # the bare-sibling resolver (25M-16): `from gsv_bitwise_drift import ...` in
    # gates/checks/gct_parity.py resolves against the importer's OWN directory,
    # so the sibling enters the closure and is digested; a real third-party
    # top-level still resolves to nothing (environment drift is preflight's job).
    sib = run_board._module_to_repo_paths(
        "gsv_bitwise_drift", 0, "gates/checks/gct_parity.py")
    report("25M-16 bare-sibling import resolves against the importer's dir",
           sib == ["gates/checks/gsv_bitwise_drift.py"], str(sib))
    third = run_board._module_to_repo_paths(
        "torch", 0, "gates/checks/gct_parity.py")
    report("25M-16 a third-party top-level still resolves to nothing",
           third == [], str(third))
    # and the live cobaya-adapter really carries the sibling in its closure.
    ca = [g for g in BOARD if g.id == "cobaya-adapter"][0]
    ca_closure = run_board._derive_closure(run_board._manifest_seeds(ca))
    report("25M-16 cobaya-adapter's closure includes the resolved sibling",
           "gates/checks/gsv_bitwise_drift.py" in ca_closure,
           "gct_parity's sibling import is digested")


def check_data_read_census():
    """1b hardening (25M-16 data-read half): a check that OPENS .py source AS
    DATA hashes it as a leaf (never closure-seeded); a whole-scope reader hashes
    the shared repo enumeration; an unreviewed data-read site reds. Drives the
    REAL _gate_code_manifest / _gate_code_digest / validate_manifests.
    """
    live_cfg = run_board._load_config()
    by_id = {g.id: g for g in BOARD}

    # geo-paths whole-scope SET EQUALITY: the gate's enumerated scan set
    # (repo_py_files) equals its manifest members -- one shared function, so the
    # scanned surface and the hashed surface can never disagree.
    geo = by_id["geo-paths"]
    members = set(m["path"] for m in run_board._gate_code_manifest(geo))
    scan = set(run_board.repo_py_files())
    report("25M-16 geo-paths whole-scope set-equality (scan set == manifest members)",
           members == scan and len(scan) > 1,
           str(len(scan)) + " repo .py; members == scan")

    # byte-edit control: changing ANY repo .py moves geo-paths' code digest
    # (the whole-scope reader stales on any repo .py change -- correct, cheap).
    base = run_board._gate_code_digest(geo)
    real_sha = run_board._file_sha256
    victim = "emulator/results.py"
    run_board._file_sha256 = (lambda rel: ("dead" + real_sha(rel))
                              if rel == victim else real_sha(rel))
    try:
        edited = run_board._gate_code_digest(geo)
    finally:
        run_board._file_sha256 = real_sha
    report("25M-16 byte-edit any repo .py stales the whole-scope gate",
           base != edited, "a one-file sha change moves geo-paths' digest")

    # the five reviewed data-readers each carry their cover in their members.
    def members_of(gid):
        return set(m["path"] for m in run_board._gate_code_manifest(by_id[gid]))
    checks = [
        ("board-selftest", "gates/run_board.py"),          # whole-repo -> harness in
        ("artifact-readback", "scalar_train_emulator.py"), # a driver read as data
        ("family-first", "cosmic_shear_sweep_ntrain_emulator.py"),  # an UNdeclared driver
        ("generator-seed", "compute_data_vectors/generator_core.py"),
    ]
    for gid, cover in checks:
        report("25M-16 " + gid + " hashes its data-read cover " + cover,
               cover in members_of(gid), "member present")
    # family-first's data cover really CLOSES a hole: the three sweep/tune drivers
    # it reads as data were not code roots, so without the data cover they escaped.
    ff = members_of("family-first")
    report("25M-16 family-first data-read closes the undeclared-driver hole",
           "cosmic_shear_tune_emulator.py" in ff
           and "cosmic_shear_sweep_hyperparam_emulator.py" in ff,
           "the drivers read as data are now hashed")

    # negative catch + restoration mutation: dropping artifact-readback from the
    # reviewed table leaves its open(...results.py) read unreviewed -> validation
    # reds (red-capable). (geo-paths now scans via the shared enumerator, so its
    # raw os.walk is gone -- the scanner's negative catch is for a NEW reader that
    # still uses a raw idiom, so the mutation lands on one that does.)
    ar = by_id["artifact-readback"]
    saved = run_board._DATA_READ_COVERS
    try:
        table = dict(saved)
        table.pop("gates/checks/artifact_readback.py")
        run_board._DATA_READ_COVERS = table
        ok, errs = run_board.validate_manifests([ar], live_cfg)
        report("25M-16 negative catch: an UNREVIEWED data-read site reds",
               (not ok) and any("unreviewed data-read" in e for e in errs),
               "a new source-as-data reader must be reviewed")
    finally:
        run_board._DATA_READ_COVERS = saved

    # (25M-19 run-time clause) a declared input that does not resolve/hash at RUN
    # time refuses BEFORE the gate body -- a None sha is a validation-time
    # allowance only. Drive the real run_selection via main().
    def _rt_run(ctx):
        CALLS["rt"] = CALLS.get("rt", 0) + 1
    rt_gate = Gate(id="rt", tier="backlog", home="selftest", maps="selftest",
                   run=_rt_run,
                   manifest=Manifest(code=(), inputs=("gate_configs.nope",)))
    rt_cfg = dict(FAKE_CFG)
    rt_cfg["yaml_dir"] = "/nonexistent-yaml-dir"
    rt_cfg["gate_configs"] = {"nope": "nope.yaml"}
    CALLS.clear()
    rc, st, _ = drive_main(["--gate", "rt"], [rt_gate], {}, cfg=rt_cfg)
    report("25M-19 run-time refusal: an unresolvable input refuses before the body",
           CALLS.get("rt", 0) == 0
           and st.get("rt", {}).get("status") == "FAIL" and rc != 0,
           "None sha at run time -> FAIL, body never ran, rc " + str(rc))


def check_manifest_riders():
    """1b phase-2 riders (kill the audit's P1/P2/P3 validation holes): root
    schema totality, directory-root expansion, and input-key resolution.

    Drives the REAL validate_manifests / _expand_root / _derive_closure.
    """
    # r1: a misspelled / non-existent code root is a validation error (P3).
    g = _mf_gate("mf-typo", _mf_body_trivial, code=("emulator/desings/typo.py",))
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("r1: a non-existent code root reds (kills the typo hole)",
           (not ok) and any("not a repo .py file or a directory" in e for e in errs),
           "a misspelled root cannot pass while seeding nothing")

    # r2: a directory root expands recursively into the closure, so declaring
    # emulator/designs really covers AND pulls in the design classes (P1/P2).
    dir_seeds = run_board._expand_root("emulator/designs")
    report("r2: a directory root expands to its .py members",
           len(dir_seeds) >= 2 and all(p.endswith(".py") for p in dir_seeds),
           str(len(dir_seeds)) + " .py members under emulator/designs")
    # both waiver covers are declared (results.py needs designs AND losses under
    # 25M-18); the leg's point is that the designs DIRECTORY root expands into
    # the closure, not the coverage rule itself.
    g = _mf_gate("mf-dir", _mf_body_trivial,
                 code=("emulator/results.py", "emulator/designs",
                       "emulator/losses"))
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    closure = run_board._derive_closure(run_board._manifest_seeds(g))
    report("r2: the directory root covers the dynamic import AND enters the closure",
           ok and ("emulator/designs/blocks.py" in closure),
           "designs tree is really hashed, not blessed empty")

    # r2b: a directory that expands to zero .py files is a validation error.
    import tempfile, os
    empty = tempfile.mkdtemp(dir=str(run_board._REPO / "gates"))
    try:
        rel = os.path.relpath(empty, str(run_board._REPO))
        g = _mf_gate("mf-empty", _mf_body_trivial, code=(rel,))
        ok, errs = run_board.validate_manifests([g], _MF_CFG)
        report("r2: an empty directory root reds (expands to no .py files)",
               (not ok) and any("expands to no .py files" in e for e in errs),
               rel + " has no .py members")
    finally:
        os.rmdir(empty)

    # r3: an input key that does not resolve against board_config is an error;
    # a resolving key clears.
    g = _mf_gate("mf-in-bad", _mf_body_trivial, code=(),
                 inputs=("gate_configs.nope",))
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("r3: an unresolvable input key reds",
           (not ok) and any("does not resolve against board_config" in e for e in errs),
           "gate_configs.nope is absent from the config")
    real = str(run_board._REPO / "gates" / "board.py")
    cfg = dict(_MF_CFG); cfg["probe_input"] = real
    g = _mf_gate("mf-in-ok", _mf_body_trivial, code=(), inputs=("probe_input",))
    report("r3: a resolving input key clears",
           run_board.validate_manifests([g], cfg)[0], "the key names a real file")


def check_input_owner_resolution():
    """1b hardening (25M-19): one owner-specific resolver per input namespace,
    NO process-CWD candidate, shared by the manifest writer and the gate
    consumer -- so the hashed path is the executed path from any cwd, and a
    repo-owned input that fails to resolve reds instead of hashing None.

    Drives the REAL _resolve_config_path / RunContext.evaluate_yaml /
    validate_manifests over the live board_config.
    """
    import os
    import tempfile
    cfg = run_board._load_config()

    # owner dispatch: each namespace resolves under its reviewed owner.
    report("25M-19 owner dispatch (evaluate_yaml=repo, gate_configs=yaml_dir, "
           "deploy_data=machine)",
           run_board._input_owner("evaluate_yaml") == "repo"
           and run_board._input_owner("gate_configs.ema-smoke-config") == "yaml_dir"
           and run_board._input_owner("deploy_data.lsst_y1_ggl_dataset") == "machine",
           "one owner per namespace")

    # two-cwd identity: resolution is a function of the owner, not the shell cwd.
    here = os.getcwd()
    p1 = run_board._resolve_config_path("evaluate_yaml", cfg)
    tmp = tempfile.mkdtemp(prefix="owner-cwd-")
    try:
        os.chdir(tmp)
        p2 = run_board._resolve_config_path("evaluate_yaml", cfg)
    finally:
        os.chdir(here)
    report("25M-19 two-cwd identity: same resolved path from any cwd",
           p1 == p2 and p1 is not None, str(p1))

    # collision-ignored: a decoy of the same relative name in the cwd is NOT
    # picked up -- proof there is no process-CWD candidate (the CWD-first
    # mutation the old resolver carried is gone).
    decoy_root = Path(tempfile.mkdtemp(prefix="owner-decoy-"))
    (decoy_root / "gates" / "configs").mkdir(parents=True)
    (decoy_root / "gates" / "configs"
     / "cobaya-adapter-evaluate.yaml").write_text("DECOY\n")
    try:
        os.chdir(decoy_root)
        p3 = run_board._resolve_config_path("evaluate_yaml", cfg)
    finally:
        os.chdir(here)
    report("25M-19 collision-ignored: a cwd decoy is not chosen (no CWD candidate)",
           p3 == p1 and "DECOY" not in Path(p3).read_text(),
           "the owner base wins over a same-named file in the cwd")

    # executed == hashed: the RunContext consumer resolves the SAME path the
    # manifest writer hashes (the executed path is the hashed path).
    ctx = run_board.RunContext(cfg=cfg, dry=False, log_fh=None, env={},
                               debug=False)
    report("25M-19 executed == hashed: consumer path == manifest-writer path",
           ctx.evaluate_yaml() == run_board._resolve_config_path("evaluate_yaml",
                                                                 cfg),
           str(ctx.evaluate_yaml()))

    # repo-owned refuse-None-sha: a repo input pointing at an absent repo file
    # reds (a repo file must resolve, never hash None); the mutation restores an
    # absent path and must red.
    bad = dict(cfg)
    bad["evaluate_yaml"] = "gates/configs/does-not-exist.yaml"
    ca = [g for g in BOARD if g.id == "cobaya-adapter"][0]
    ok, errs = run_board.validate_manifests([ca], bad)
    report("25M-19 repo-owned input that fails to resolve reds (refuse None sha)",
           (not ok) and any("does not resolve to a repo file" in e for e in errs),
           "an absent repo file is a resolution bug, not a dev-box gap")
    # control: the real evaluate_yaml resolves and the gate validates.
    report("25M-19 the real repo-owned input resolves and clears (control)",
           run_board.validate_manifests([ca], cfg)[0],
           "evaluate_yaml resolves under _REPO on any machine")


def check_cross_invocation_lineage():
    """1b hardening (25M-26): cross-invocation dependency lineage. A child PASS
    records the attempt id of each dependency result it consumed; a LATER,
    SEPARATE invocation whose dependency has since been rerun reads
    stale-dependency and reruns the child, instead of resuming it against a
    superseded result. This is the cross-PROCESS half of 25M-20 the in-process
    reran set cannot see, so the legs share ONE status file + log dir across
    several real main() invocations.
    """
    import tempfile
    prereq = make_gate("prereq")
    child = make_gate("child", deps=("prereq",))
    gates = [prereq, child]

    tmp = Path(tempfile.mkdtemp(prefix="lineage-"))
    saved = {name: getattr(run_board, name) for name in
             ("BOARD", "_LOGS_DIR", "_STATUS_FILE", "_BOARD_MD",
              "_load_config", "_load_status", "preflight", "_log_header")}
    try:
        run_board.BOARD = gates
        run_board._LOGS_DIR = tmp
        run_board._STATUS_FILE = tmp / "board_status.json"
        run_board._BOARD_MD = tmp / "BOARD.md"
        run_board._load_config = lambda: dict(FAKE_CFG)
        run_board._load_status = lambda: (
            json.loads(run_board._STATUS_FILE.read_text())
            if run_board._STATUS_FILE.exists() else {})
        run_board.preflight = lambda cfg: (True, {})
        run_board._log_header = lambda ctx, gate: None

        # invocation 1: run both -> child PASSes, snapshotting prereq's attempt.
        CALLS.clear()
        rc1 = run_board.main([])
        st1 = run_board._load_status()
        report("25M-26 inv1: both run; the child snapshots its dependency lineage",
               rc1 == 0 and isinstance(st1["child"].get("deps"), dict),
               "child.deps snapshot persisted")

        # invocation 2 (separate process): force-rerun ONLY prereq -> new attempt.
        a1 = st1["prereq"]["attempt"]
        CALLS.clear()
        rc2 = run_board.main(["--gate", "prereq", "--force-rerun", "prereq"])
        st2 = run_board._load_status()
        report("25M-26 inv2: force-rerunning the prerequisite advances its attempt",
               rc2 == 0 and st2["prereq"]["attempt"] != a1, "new prereq attempt")

        # the pure lineage predicate (no log entanglement): st1 matches, st2 stales.
        report("25M-26 lineage predicate: a matching snapshot is current",
               run_board._dependency_lineage_state(st1, child) is None,
               "inv1 child snapshot matches inv1 prereq attempt")
        report("25M-26 lineage predicate: a since-rerun dependency is stale-dependency",
               run_board._dependency_lineage_state(st2, child) == "stale-dependency",
               "child snapshot points at the superseded prereq attempt")

        # invocation 3: the child's snapshot no longer matches -> it RERUNS (the
        # exact two-invocation witness; without the fix it would resume, exit-0,
        # zero bodies).
        CALLS.clear()
        rc3 = run_board.main(["--gate", "child"])
        report("25M-26 two-invocation witness: a since-rerun dependency reruns the child",
               CALLS.get("child", 0) == 1 and rc3 == 0,
               "child ran once, not resumed against the superseded prereq")

        # snapshot-refresh control: the re-passed child snapshots the new attempt,
        # so a further invocation RESUMES it (no body).
        CALLS.clear()
        rc4 = run_board.main(["--gate", "child"])
        report("25M-26 snapshot-refresh: the re-passed child then resumes (no body)",
               CALLS.get("child", 0) == 0 and rc4 == 0,
               "the refreshed snapshot matches -> resume")

        # legacy / mutation: strip the child's lineage snapshot from the persisted
        # record (a pre-25M-26 PASS). The snapshot-free record is non-green and
        # reruns -- never retroactively blessed; without the lineage check it
        # would resume (exit-0, zero bodies), so the mutation is red-capable.
        rec = json.loads(run_board._STATUS_FILE.read_text())
        del rec["child"]["deps"]
        run_board._STATUS_FILE.write_text(json.dumps(rec))
        CALLS.clear()
        rc5 = run_board.main(["--gate", "child"])
        report("25M-26 legacy/mutation: a snapshot-free dependent PASS reruns",
               CALLS.get("child", 0) == 1,
               "no lineage snapshot -> non-green, exit " + str(rc5))
    finally:
        for name, value in saved.items():
            setattr(run_board, name, value)


def check_watch_tracked_drivers():
    """1b machinery follow-up (25M-27): the clean-tree watch derives its root
    drivers from git-TRACKED identity (union current files), so a DELETED tracked
    driver stays in the pathspec and reds -- it cannot be certified clean by a
    glob built from the already-damaged filesystem; and a nonzero git status is a
    failure, not an empty clean result. Pure-git legs in a temp repo with the
    real _watched_paths / _git / _dirty_lines.
    """
    tmp = Path(tempfile.mkdtemp(prefix="watch-"))

    def git(*a):
        subprocess.run(["git"] + list(a), cwd=tmp,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    git("init")
    git("config", "user.email", "x@x")
    git("config", "user.name", "x")
    (tmp / "driver.py").write_text("print(1)\n")
    (tmp / "emulator").mkdir()
    (tmp / "emulator" / "m.py").write_text("x = 1\n")
    (tmp / "gates").mkdir()
    (tmp / "gates" / "board_config.json").write_text("{}\n")
    (tmp / "README.md").write_text("hi\n")
    git("add", "-A")
    git("commit", "-m", "init")

    def offenders():
        w = run_board._watched_paths()
        rc, out = run_board._git(["status", "--porcelain", "--"] + w, strip=False)
        return w, rc, run_board._dirty_lines(out)

    saved = {name: getattr(run_board, name)
             for name in ("_REPO", "_EXECUTABLE_DIRS", "_WATCH_EXCLUDE")}
    try:
        run_board._REPO = tmp
        run_board._EXECUTABLE_DIRS = ("emulator", "gates")
        run_board._WATCH_EXCLUDE = "gates/board_config.json"

        w, rc, off = offenders()
        report("25M-27 clean control: a committed tree has no offenders",
               rc == 0 and off == [], repr(off))
        report("25M-27 the tracked root driver is watched",
               "driver.py" in w, "union includes tracked roots")

        (tmp / "driver.py").write_text("print(2)\n")
        w, rc, off = offenders()
        report("25M-27 a modified tracked root driver reds",
               any("driver.py" in o for o in off), repr(off))
        git("checkout", "--", "driver.py")

        os.remove(tmp / "driver.py")
        w, rc, off = offenders()
        report("25M-27 a DELETED tracked root driver reds and is named",
               "driver.py" in w and any("driver.py" in o for o in off),
               repr(off))
        # mutation: the OLD existence-glob (current files only) misses the delete.
        old_watch = list(run_board._EXECUTABLE_DIRS) + [
            e.name for e in run_board._REPO.glob("*.py")]
        _rc, old_out = run_board._git(["status", "--porcelain", "--"] + old_watch,
                                      strip=False)
        report("25M-27 mutation (existence glob) misses the deleted driver",
               run_board._dirty_lines(old_out) == [],
               "the glob-only watch certifies the damaged tree clean")
        git("checkout", "--", "driver.py")

        (tmp / "newdriver.py").write_text("print(3)\n")
        w, rc, off = offenders()
        report("25M-27 a NEWLY ADDED untracked root driver reds",
               "newdriver.py" in w and any("newdriver.py" in o for o in off),
               repr(off))
        os.remove(tmp / "newdriver.py")

        (tmp / "README.md").write_text("changed\n")
        w, rc, off = offenders()
        report("25M-27 an unrelated root text file stays outside the surface",
               all("README.md" not in o for o in off), repr(off))
        git("checkout", "--", "README.md")

        (tmp / "gates" / "board_config.json").write_text('{"x": 1}\n')
        w, rc, off = offenders()
        report("25M-27 a config-only change stays clean (excluded)",
               off == [], repr(off))
    finally:
        for name, value in saved.items():
            setattr(run_board, name, value)


def check_stale_member_surface():
    """1b machinery follow-up (25M-28): --list and BOARD.md name the SAME first
    stale member via one shared formatter, and an input is compared by its FULL
    identity (key, path, sha256) so a byte-identical RELOCATION names the changed
    path. Drives the real _stale_member / _state_detail / cmd_list.
    """
    import io
    import contextlib

    def _r(ctx):
        pass
    g = Gate(id="x", tier="backlog", home="h", maps="m", run=_r,
             manifest=Manifest(code=(), inputs=("evaluate_yaml",)))
    rec = {"status": "PASS",
           "manifest": {"code": [],
                        "inputs": [{"key": "evaluate_yaml",
                                    "path": "/old/a.yaml", "sha256": "abc"}]}}
    orig = run_board._gate_input_manifest
    try:
        # byte-identical relocation: same sha, new path -> names key + path change.
        run_board._gate_input_manifest = lambda gg, c: [
            {"key": "evaluate_yaml", "path": "/new/b.yaml", "sha256": "abc"}]
        m = run_board._stale_member(rec, g, {})
        report("25M-28 a byte-identical input relocation names key + old->new path",
               "evaluate_yaml" in m and "->" in m and "/new/b.yaml" in m, repr(m))
        # mutation: the OLD hash-only compare ({key: sha}) misses the relocation.
        old_fresh = {mm["key"]: mm["sha256"]
                     for mm in run_board._gate_input_manifest(g, {})}
        old_named = any(old_fresh.get(mm.get("key")) != mm.get("sha256")
                        for mm in rec["manifest"]["inputs"])
        report("25M-28 mutation (hash-only compare) misses the relocation",
               not old_named, "same sha -> old logic calls it unchanged")
        # content-only change: different sha -> names just the key.
        run_board._gate_input_manifest = lambda gg, c: [
            {"key": "evaluate_yaml", "path": "/old/a.yaml", "sha256": "zzz"}]
        report("25M-28 a content-only input change names the key",
               run_board._stale_member(rec, g, {}) == "input:evaluate_yaml",
               "sha differs")
    finally:
        run_board._gate_input_manifest = orig

    # cmd_list uses the SAME shared _state_detail as BOARD.md, so --list names a
    # stale code member (the operator surface the ruling requires can inspect it).
    def _r2(ctx):
        pass
    cg = Gate(id="cg", tier="backlog", home="h", maps="m", run=_r2,
              manifest=Manifest(code=("emulator/results.py",), inputs=()))
    stale_rec = {"status": "PASS",
                 "manifest": {"code": [{"path": "emulator/results.py",
                                        "sha256": "STALE"}],
                              "inputs": []}}
    detail = run_board._state_detail("stale-code", stale_rec, cg, FAKE_CFG)
    report("25M-28 the shared formatter names the stale code member",
           "stale member" in detail and "emulator/results.py" in detail, detail)
    saved_board = run_board.BOARD
    try:
        run_board.BOARD = [cg]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_board.cmd_list({"cg": stale_rec}, FAKE_CFG)
        out = buf.getvalue()
        report("25M-28 --list names the stale member (same formatter as BOARD.md)",
               "emulator/results.py" in out and "stale member" in out,
               "cmd_list carries the detail")
    finally:
        run_board.BOARD = saved_board


def check_manifest_persistence():
    """1b phase 2: a declared gate persists its resolved manifest members, its
    digest IS the member digest, a changed member reads stale-code (named), a
    pre-1b PASS record reads pre-manifest, and an undeclared gate is untouched.

    Drives the REAL _gate_manifest_block / _gate_code_digest / _resume_state /
    _stale_member -- no runner, no torch.
    """
    cfg = {"rootdir": None, "yaml_dir": None}
    g = _mf_gate("mf-persist", _mf_body_trivial, code=("emulator/data_staging.py",))

    block = run_board._gate_manifest_block(g, cfg)
    code_paths = [m["path"] for m in block["code"]]
    report("declared gate persists sorted resolved code members",
           code_paths == sorted(code_paths) and len(code_paths) > 2
           and all("sha256" in m and len(m["sha256"]) == 64 for m in block["code"]),
           str(len(code_paths)) + " members, sorted, each a path+sha256")
    report("overall code_digest is the resolved members' digest",
           run_board._gate_code_digest(g) == run_board._members_digest(block["code"]),
           "digest binds the persisted membership")

    # input side: a declared key resolves to a specific file, whose sha is its
    # member (the whole-yaml_dir hash retires for a declared gate).
    real = str(run_board._REPO / "gates" / "board.py")
    cfg_in = dict(cfg); cfg_in["probe_input"] = real
    gi = Gate(id="mf-input", tier=TIER_BACKLOG, home="x", maps="",
              run=_mf_body_trivial, manifest=Manifest(code=(), inputs=("probe_input",)))
    inmembers = run_board._gate_input_manifest(gi, cfg_in)
    report("declared input key resolves to its specific file + sha",
           len(inmembers) == 1 and inmembers[0]["key"] == "probe_input"
           and inmembers[0]["sha256"] is not None,
           "the named file, not the whole yaml_dir")

    # a pre-1b PASS record (no manifest block) on a now-declared gate reads
    # pre-manifest -> reruns.
    rec_pre = {"status": "PASS",
               "code_digest": run_board._gate_code_digest(g),
               "input_digest": run_board._gate_input_digest(g, cfg)}
    report("declared gate with a pre-manifest PASS record reads pre-manifest",
           run_board._resume_state({g.id: rec_pre}, g, cfg) == "pre-manifest",
           "digestless-is-stale; it reruns and republishes members")

    # a persisted record whose first code member's sha no longer matches reads
    # stale-code, and _stale_member names that member.
    rec_stale = {"status": "PASS", "manifest": copy.deepcopy(block),
                 "code_digest": "bogus-does-not-match", "input_digest": "bogus"}
    rec_stale["manifest"]["code"][0]["sha256"] = "0" * 64
    st = run_board._resume_state({g.id: rec_stale}, g, cfg)
    named = run_board._stale_member(rec_stale, g, cfg)
    report("a changed code member reads stale-code and is named",
           st == "stale-code" and named.startswith("code:"),
           "state=" + st + " member=" + named)

    # an undeclared gate keeps the legacy fallback: no manifest persisted, and
    # its resume never reads pre-manifest.
    gu = Gate(id="mf-undeclared", tier=TIER_BACKLOG, home="x", maps="",
              run=_mf_body_trivial)                 # manifest defaults to None
    rec_u = {"status": "PASS",
             "code_digest": run_board._gate_code_digest(gu),
             "input_digest": run_board._gate_input_digest(gu, cfg)}
    report("undeclared gate: no manifest persisted, never pre-manifest",
           run_board._gate_manifest_block(gu, cfg) is None
           and run_board._resume_state({gu.id: rec_u}, gu, cfg) != "pre-manifest",
           "legacy dual-digest fallback intact")


def check_child_env():
    """1d: sh() is the one owner of the child environment -- every child a gate
    launches observes ROOTDIR = the board's resolved rootdir, never the
    inherited shell value, and an unresolved rootdir refuses before launch.

    Drives the REAL RunContext.sh over a tiny child that echoes its ROOTDIR.
    """
    A = "/tmp/mf_shell_root_A"          # the value the launching shell carries
    B = "/tmp/mf_board_root_B"          # the board's resolved (certified) rootdir
    echo = [sys.executable, "-c",
            "import os; print(os.environ.get('ROOTDIR', ''))"]
    ctx = run_board.RunContext(cfg={"rootdir": B}, dry=False,
                               log_fh=io.StringIO(), env={}, debug=False)
    saved = os.environ.get("ROOTDIR")
    try:
        os.environ["ROOTDIR"] = A
        _rc, out = ctx.sh(cmd=echo)
        report("child observes the board rootdir B, not the inherited shell A",
               out.strip() == B, "injected " + B + " over inherited " + A)
        del os.environ["ROOTDIR"]
        _rc, out = ctx.sh(cmd=echo)
        report("child still observes B when $ROOTDIR is absent from the shell",
               out.strip() == B, "injection does not depend on inheritance")
        # mutation arm: the retired inherit-only environment (no injection) ->
        # the child sees the shell A, not the board B -> the contract fails.
        os.environ["ROOTDIR"] = A
        proc = subprocess.run(echo, env=dict(os.environ),
                              stdout=subprocess.PIPE, text=True)
        report("mutation (inherit-only env) makes the child see A, not B",
               proc.stdout.strip() == A and A != B,
               "an uninjected child executes against the wrong root")
        # refusal: an unresolved board rootdir refuses before any launch.
        ctx_none = run_board.RunContext(cfg={"rootdir": None}, dry=False,
                                        log_fh=io.StringIO(), env={}, debug=False)
        raised = False
        try:
            ctx_none.sh(cmd=echo)
        except run_board.GateFailure:
            raised = True
        report("unresolved board rootdir refuses before launch",
               raised, "no child runs against an uncertified root")
        # census: every gate child launch routes through sh()'s single Popen
        # (run_check / run_driver / the golden-run git ops all call self.sh);
        # _git and _probe_import use subprocess.run and are harness-internal,
        # not gate children.
        rb_src = Path(run_board.__file__).read_text()
        report("one owner: exactly one subprocess.Popen (inside sh) in the runner",
               rb_src.count("subprocess.Popen") == 1,
               "all gate children route through sh()")
    finally:
        if saved is None:
            os.environ.pop("ROOTDIR", None)
        else:
            os.environ["ROOTDIR"] = saved


class _GhaFakeCtx:
    """A fake ctx that drives the REAL gate_gha_f warning leg: run_driver
    returns a chosen (rc, text) for the --activation flag run and a failing
    frozen-trunk result for the license run, and captures ctx.expect verdicts.
    """
    def __init__(self, warn_rc):
        self.dry = False
        self._warn_rc = warn_rc
        self.expects = {}

    def require_caps(self, *a):
        pass

    def require_config(self, key):
        return "y.yaml"

    def run_driver(self, *, yaml_path, extra=(), allow_fail=False, **kw):
        if extra:                        # the pin run WITH the --activation flag
            return (self._warn_rc,
                    "the head keeps its model.trf.activation pin (gated_power)")
        return (1, "frozen")             # the invalid-license run (rc_l != 0)

    def expect(self, *, label, ok, detail="", aid=None):
        # aid= accepted so this stub matches the real ctx.expect signature
        # (gate_gha_f now names a queue-2 assertion id on each warning leg);
        # the RT-04 checks still key on label.
        self.expects[label] = ok


def check_gha_f_warning():
    """RT-04: head-activation-pin's flag-vs-pin warning leg must require the
    flag run to SUCCEED (rc_w == 0), not merely print the warning.

    Drives the REAL board.gate_gha_f with a fake ctx; the golden and smoke
    helpers are stubbed (they run on the box), so only the warning-leg logic
    under test executes.
    """
    saved_g, saved_s = board._golden_leg, board._smoke_driver
    try:
        board._golden_leg = lambda **k: None
        board._smoke_driver = lambda **k: None
        label = "head-activation-pin flag-vs-pin warning"
        # a warning printed but the flag run exited nonzero -> the leg must FAIL.
        fc = _GhaFakeCtx(warn_rc=1)
        board.gate_gha_f(fc)
        report("RT-04: a warning on a FAILED flag run fails the warning leg",
               fc.expects.get(label) is False,
               "rc_w != 0 must not pass even with the warning present")
        # warning printed AND the flag run succeeded -> the leg passes (control).
        fc2 = _GhaFakeCtx(warn_rc=0)
        board.gate_gha_f(fc2)
        report("RT-04: a warning on a SUCCESSFUL flag run passes (control)",
               fc2.expects.get(label) is True, "rc_w == 0 and the warning")
    finally:
        board._golden_leg, board._smoke_driver = saved_g, saved_s


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
    check_dependency_currency()
    check_config_readers()
    check_dependency_topology()
    check_digest_projection()
    print("\n-- evidence atomicity (RUNNING + immutable logs) --")
    check_evidence_atomicity()
    print("\n-- raw-log trust (a PASS is only as good as its cited log) --")
    check_log_trust()
    print("\n-- structured evidence map (anchors resolve, ids unique, transform) --")
    check_evidence_map()
    print("\n-- evidence reconciliation (declared vs executed, ruling 6 + D1-ii) --")
    check_evidence_reconciliation()
    print("\n-- evidence gate verdict (the runner hook flips + persists the block) --")
    check_evidence_gate_verdict()
    print("\n-- ##AID manifest (check-script per-leg fold, crash/malformed/one-verdict) --")
    check_aid_manifest()
    print("\n-- golden leg (both child rcs + non-empty selection, not just a diff) --")
    check_golden_leg()
    print("\n-- clean-tree watch (per-line porcelain, one owner) --")
    check_dirty_watch()
    print("\n-- manifest reconciliation (subprocess + dynamic-import censuses) --")
    check_manifest_reconciliation()
    print("\n-- runtime-loader census (adapters loaded by path / python_path) --")
    check_runtime_loader_census()
    print("\n-- data-read census (source opened as data hashes as a leaf) --")
    check_data_read_census()
    print("\n-- manifest riders (root schema, dir expansion, input keys) --")
    check_manifest_riders()
    print("\n-- input owner resolution (owner base, no cwd, executed==hashed) --")
    check_input_owner_resolution()
    print("\n-- cross-invocation lineage (a since-rerun dependency reruns its child) --")
    check_cross_invocation_lineage()
    print("\n-- clean-tree watch on tracked drivers (a deleted driver still reds) --")
    check_watch_tracked_drivers()
    print("\n-- stale-member surface (--list + BOARD.md name the same member) --")
    check_stale_member_surface()
    print("\n-- manifest persistence (resolved members, digest, pre-manifest) --")
    check_manifest_persistence()
    print("\n-- child environment (sh injects the certified ROOTDIR) --")
    check_child_env()
    print("\n-- head-activation-pin warning leg (RT-04: rc_w == 0 required) --")
    check_gha_f_warning()
    print("")
    if FAILURES:
        print("board-selftest: %d FAILURE(S): %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("board-selftest: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
