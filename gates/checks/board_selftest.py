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


def pass_record(gate, cfg=None):
    """A current-PASS status record for `gate` under cfg (all three digests set).

    Includes the raw-log evidence the runner now verifies: a seed log name and
    its digest over SEED_LOG_BYTES. drive_main writes that content into its temp
    log dir, so the record reads as a genuine current PASS (not one flagged
    stale-log for a missing or unverifiable log).
    """
    use = FAKE_CFG if cfg is None else cfg
    return {"status": "PASS",
            "code_digest": run_board._gate_code_digest(gate),
            "input_digest": run_board._gate_input_digest(gate, use),
            "log": gate.id + ".seed.log",
            "log_digest": hashlib.sha256(SEED_LOG_BYTES).hexdigest()}


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
    bad = _evidence_gate("bad", "bad.leg",
                         "gates-and-board.md#no-such-marker-xyz")
    ok, errs = run_board.validate_evidence([bad])
    report("an unresolved anchor marker is rejected", not ok,
           errs[0] if errs else "")

    # mutation 2: an anchor naming a note that does not exist is caught.
    miss = _evidence_gate("miss", "miss.leg", "no-such-note.md#m")
    ok, errs = run_board.validate_evidence([miss])
    report("an anchor citing a missing note is rejected", not ok,
           errs[0] if errs else "")

    # mutation 3: two gates sharing an assertion id is caught (board-wide
    # uniqueness, so a leg's id names exactly one leg).
    real_anchor = "gates-and-board.md#brd-a-board-truth"
    dup_a = _evidence_gate("dupA", "dup.id", real_anchor)
    dup_b = _evidence_gate("dupB", "dup.id", real_anchor)
    ok, errs = run_board.validate_evidence([dup_a, dup_b])
    report("a duplicate assertion id across gates is rejected", not ok,
           errs[0] if errs else "")

    # mutation 4: an anchor missing the '#<marker>' shape is caught.
    mal = _evidence_gate("mal", "mal.leg", "gates-and-board.md")
    ok, errs = run_board.validate_evidence([mal])
    report("a malformed anchor (no #marker) is rejected", not ok,
           errs[0] if errs else "")

    # control: a gate with no evidence is never itself a failure (the
    # migration is rolling, not a flag day).
    ok, errs = run_board.validate_evidence([make_gate("plain")])
    report("a gate with no evidence is not a failure", ok and errs == [],
           "empty evidence tolerated")


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

    def expect(self, *, label, ok, detail=""):
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
    print("\n-- structured evidence map (anchors resolve, ids unique) --")
    check_evidence_map()
    print("\n-- clean-tree watch (per-line porcelain, one owner) --")
    check_dirty_watch()
    print("\n-- manifest reconciliation (subprocess + dynamic-import censuses) --")
    check_manifest_reconciliation()
    print("\n-- runtime-loader census (adapters loaded by path / python_path) --")
    check_runtime_loader_census()
    print("\n-- manifest riders (root schema, dir expansion, input keys) --")
    check_manifest_riders()
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
