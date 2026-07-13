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

    # (b) dynamic-import census over the derived closure. results.py is a
    # waived file (the model-recipe pattern); its covering roots are the
    # design / loss trees.
    g = _mf_gate("mf-dyn", _mf_body_trivial, code=("emulator/results.py",))
    ok, errs = run_board.validate_manifests([g], _MF_CFG)
    report("waived dynamic import with NO covering root declared reds",
           (not ok) and any("covering roots" in e for e in errs),
           "declares results.py but not emulator/designs")
    g = _mf_gate("mf-dyn-ok", _mf_body_trivial,
                 code=("emulator/results.py", "emulator/designs/blocks.py"))
    report("declaring a covering root clears the dynamic-import census",
           run_board.validate_manifests([g], _MF_CFG)[0], "designs root declared")

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
    g = _mf_gate("mf-dir", _mf_body_trivial,
                 code=("emulator/results.py", "emulator/designs"))
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
