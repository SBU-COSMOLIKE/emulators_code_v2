#!/usr/bin/env python3
"""The program you run: ``python gates/run_board.py``.

It parses the command line and runs each test in board.py's order.
A small class, RunContext, hands every test the few things it needs:
run a command, write to its log, check a value. Preflight blocks the
run before any GPU time if the git tip is stale, the tree is dirty, a
cocoa import fails, a data path is missing, $ROOTDIR is unset (so rootdir
cannot resolve), or the driver fileroot is still the shipped placeholder.
Each test gets one raw
log (the full streamed output, never a summary); a rerun skips tests
already PASS; a failed save-rebuild-drift skips cobaya-adapter (which
needs its saved file); any other failure never stops the rest.
BOARD.md is the final pass/fail table. Terms: the glossary at the top
of board.py.

Typical use on the workstation, from the Cocoa root ($ROOTDIR, cocoa
env active; the harness finds its own files, so any directory works):

    G=external_modules/code/emulators_code_v2/gates
    git -C $G/.. pull                  # tip = the harness commit
    python $G/run_board.py --check     # preflight only
    python $G/run_board.py --dry-run   # print the plan
    python $G/run_board.py             # the whole board, in order
    git -C $G/.. add -f gates/logs
    git -C $G/.. commit -m "workstation board run: logs"

The full REGRESSION pass (rerun everything, greens included — the
no-regressions proof after a batch of library changes):

    python $G/run_board.py --force-rerun-all

It composes with --gate / --tier / --from (forces only the selected
gates) and never deletes the resume map, so an interrupted regression
pass re-run WITHOUT the flag resumes from whatever it already
re-proved.
"""

import argparse
import contextlib
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import board
from board import BOARD, Gate, GateFailure


# The commit that carries the board + harness spec (the base-notes
# constant). Preflight requires it an ANCESTOR of HEAD, never an
# exact-tip equality: committing the harness and later the logs moves
# the tip forward, and a commit cannot name its own hash.
_BASE_NOTES_COMMIT = "3b39824159ba7df177fb070658d22b7dd81162a4"

# The full cocoa stack preflight requires (harness rule 4c).
_REQUIRED_IMPORTS = (
  ("torch", "import torch"),
  ("cosmolike", "import cosmolike_lsst_y1_interface"),
  ("cobaya", "import cobaya"))

_GATES_DIR = Path(__file__).resolve().parent
_REPO = _GATES_DIR.parent
_LOGS_DIR = _GATES_DIR / "logs"
_STATUS_FILE = _LOGS_DIR / "board_status.json"
_BOARD_MD = _LOGS_DIR / "BOARD.md"
_CONFIG_FILE = _GATES_DIR / "board_config.json"

# The training driver every run-shaped gate invokes.
_DRIVER = "cosmic_shear_train_emulator.py"

# Pre-rename driver filenames (verb-first), for the pinned golden legs:
# a golden worktree built at a commit BEFORE the family-first rename
# carries these names, so run_driver resolves by existence instead of
# assuming today's name (the run-10 ema-off-identity catch: the pinned
# pre-EMA build had no cosmic_shear_train_emulator.py and the golden
# leg failed on a missing file, not on the byte identity it exists to
# check).
_LEGACY_DRIVERS = {
  "cosmic_shear_train_emulator.py":
    "train_single_emulator_cosmic_shear.py",
  "cosmic_shear_tune_emulator.py":
    "tune_single_emulator_cosmic_shear.py",
  "cosmic_shear_sweep_ntrain_emulator.py":
    "sweep_ntrain_emulator_cosmic_shear.py",
  "cosmic_shear_sweep_hyperparam_emulator.py":
    "sweep_hyperparam_emulator_cosmic_shear.py",
  "cosmic_shear_bakeoff_activation_emulator.py":
    "bakeoff_activation_emulator_cosmic_shear.py",
}


# --------------------------------------------------------------------------
# The per-test helper each test function receives.
# --------------------------------------------------------------------------

class RunContext:
  """The services a gate body uses: shell, logging, config, worktree.

  One helper is built per test (or one shared dry helper for
  --dry-run). It tees every command into the gate's raw log, writes
  each acceptance verdict, resolves config paths, and manages the
  temporary worktree. Gate bodies never touch subprocess, files, or
  git directly; they go through these methods so every command lands
  in the log and every path comes from board_config.json.

  Arguments:
    cfg     = the parsed board_config.json (deployment paths).
    dry     = when True, commands are printed, not run, and acceptance
              is skipped (--dry-run prints the plan).
    log_fh  = the open gate-log file handle to tee into, or None for a
              dry run (stdout only).
    env     = the capability map {"torch": bool, ...} preflight built;
              require_caps reads it (empty in dry mode).
    debug   = when True, mirror the full command output + the config dump
              to the terminal too; when False (the quiet default) the
              terminal shows only the gate header, CHECK / GATE verdicts,
              and the log-only streams go to the gate log alone.
  """

  def __init__(self, *, cfg, dry, log_fh, env, debug=False):
    self.cfg = cfg
    self.dry = dry
    self._log_fh = log_fh
    self._env = env
    self.debug = debug
    self.repo = _REPO
    # The interpreter running the harness runs the gates too, so
    # a check / driver never depends on a bare "python" being on PATH
    # (the dev Mac has only python3). cobaya-run stays a PATH lookup (it
    # is a console script, not an interpreter).
    self.python = sys.executable

  # ---- logging -----------------------------------------------------------

  def _emit(self, text, *, log_only=False):
    """Write a line to the gate log, and to the terminal unless quieted.

    Every line always lands in the gate log (the full raw record). The
    terminal also gets it unless log_only is set, so the terminal carries
    the gate header, the CHECK / GATE verdicts, and the harness notes,
    while the tee'd command output and the config dump go to the log
    alone. debug True overrides log_only, mirroring everything.

    Arguments:
      text     = the text to write.
      log_only = when True, keep this off the terminal (unless debug).
    """
    if self._log_fh is not None:
      self._log_fh.write(text)
      self._log_fh.flush()
    if self.debug or not log_only:
      sys.stdout.write(text)
      sys.stdout.flush()

  def log(self, msg):
    """Write a harness annotation line (a note, not command output).

    Arguments:
      msg = the annotation text; prefixed so it reads apart from the
            tee'd command output in the raw log.
    """
    self._emit("[harness] " + msg + "\n")

  def expect(self, *, label, ok, detail=""):
    """Write an acceptance verdict, raising GateFailure on failure.

    Arguments:
      label  = what is being checked (appears as CHECK <label>).
      ok      = the boolean verdict the harness computed.
      detail = the acceptance value(s) behind the verdict, always
               logged so the raw log carries the evidence, not a bare
               PASS/FAIL.

    Returns:
      None. Raises GateFailure when ok is False.
    """
    verdict = "PASS" if ok else "FAIL"
    self._emit("[harness] CHECK " + label + ": " + verdict
               + ("" if detail == "" else "  (" + detail + ")") + "\n")
    if not ok:
      raise GateFailure(label + ("" if detail == "" else ": " + detail))

  # ---- shell -------------------------------------------------------------

  def sh(self, *, cmd, cwd=None, allow_fail=False, env=None):
    """Run a command, teeing its output into the gate log.

    Arguments:
      cmd        = the command as an argv list (no shell).
      cwd        = the working directory (default the repo root).
      allow_fail = when False a nonzero exit raises GateFailure; when
                   True the caller inspects the returned code itself.
      env        = extra environment variables merged over the current
                   environment for the child (e.g. PYTHONPATH for a
                   check script); None runs in the inherited env.

    Returns:
      (returncode, output) where output is the combined stdout+stderr
      captured while it streamed. In dry mode the command is printed
      and (0, "") returned.
    """
    where = self.repo if cwd is None else Path(cwd)
    printable = " ".join(cmd)
    env_note = ""
    if env is not None:
      env_note = "  [env: " + ", ".join(sorted(env)) + "]"
    if self.dry:
      self._emit("[dry-run] would run (in " + str(where) + "): "
                 + printable + env_note + "\n")
      return (0, "")

    child_env = None
    if env is not None:
      child_env = dict(os.environ)
      child_env.update(env)
    # the command echo and its streamed output are log-only: the full
    # driver / check stream belongs in the gate log, not on the terminal
    # (debug mirrors it). The gate's CHECK / GATE verdicts stay visible.
    self._emit("[harness] $ " + printable + env_note + "\n", log_only=True)
    proc = subprocess.Popen(cmd,
                            cwd=str(where),
                            env=child_env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1)
    captured = []
    for line in proc.stdout:
      self._emit(line, log_only=True)
      captured.append(line)
    proc.wait()
    out = "".join(captured)
    if proc.returncode != 0 and not allow_fail:
      raise GateFailure("command failed (rc " + str(proc.returncode)
                        + "): " + printable)
    return (proc.returncode, out)

  def run_check(self, script, *, extra=(), allow_fail=True):
    """Run a harness check script with the repo root on PYTHONPATH.

    A check script is invoked as ``python gates/checks/<name>.py`` from
    the repo root, so Python puts the script's own directory
    (gates/checks) on sys.path, NOT the repo, and ``import emulator``
    would raise ModuleNotFoundError. This injects PYTHONPATH=<repo> so
    the check imports the package it tests. The verbatim scripts stay
    untouched; the path fix lives entirely in the runner.

    Arguments:
      script     = the check script path relative to the repo (or
                   absolute); passed straight to python.
      extra      = extra script arguments.
      allow_fail = passed through to sh (a gate reads the rc itself).

    Returns:
      (returncode, output) from the check script.
    """
    cmd = [self.python, str(script)]
    for flag in extra:
      cmd.append(flag)
    existing = os.environ.get("PYTHONPATH", "")
    if existing == "":
      pythonpath = str(self.repo)
    else:
      pythonpath = str(self.repo) + os.pathsep + existing
    return self.sh(cmd=cmd,
                   env={"PYTHONPATH": pythonpath},
                   allow_fail=allow_fail)

  # ---- environment -------------------------------------------------------

  def require_caps(self, *caps):
    """Fail the gate loudly if a required capability is absent.

    Preflight already verified the full stack, so on a real workstation
    run this always passes; it gives a clear per-gate message if a gate
    is somehow run without its capability.

    Arguments:
      caps = the capability names the gate needs ("torch", "cosmolike",
             "cobaya", "gpu").
    """
    if self.dry:
      return
    missing = []
    for cap in caps:
      if not self._env.get(cap, False):
        missing.append(cap)
    if len(missing) > 0:
      raise GateFailure("environment missing: " + ", ".join(missing)
                        + " (run --check to see the remedy)")

  # ---- config paths ------------------------------------------------------

  def _yaml_dir(self):
    """The absolute directory holding the driver YAMLs, or None if unset.

    A rootdir-relative yaml_dir is resolved against rootdir (the same
    rule preflight (d) uses), so the --yaml handed to every driver is an
    absolute path (an already-absolute yaml_dir is used unchanged). The
    driver then reads that path from any launch directory, and the pinned
    golden leg (run in a temporary worktree) reads the same file.
    """
    value = self.cfg.get("yaml_dir")
    if value is None:
      return None
    base = Path(value)
    rootdir_value = self.cfg.get("rootdir")
    if not base.is_absolute() and rootdir_value is not None:
      base = Path(rootdir_value) / value
    return base

  def config_yaml_name(self, yaml_name):
    """Resolve a literal YAML filename against the configured yaml_dir.

    Arguments:
      yaml_name = the shipped config filename (e.g.
                  cosmic_shear_train_emulator.yaml).

    Returns:
      the absolute Path. In dry mode an unset yaml_dir yields a visible
      placeholder rather than aborting the plan.
    """
    base = self._yaml_dir()
    if base is None:
      if self.dry:
        return Path("<UNSET:yaml_dir>/" + yaml_name)
      raise GateFailure("board_config.json yaml_dir is unset; set it to "
                        "the driver YAML directory")
    return base / yaml_name

  def require_config(self, config_key):
    """Resolve a gate's bespoke smoke YAML from gate_configs.

    Arguments:
      config_key = the board_config.json gate_configs key.

    Returns:
      the absolute Path to the YAML. An unset key or a missing file
      raises GateFailure naming the file to author (in dry mode a
      placeholder is returned so the plan still prints).
    """
    value = self.cfg.get("gate_configs", {}).get(config_key)
    if value is None:
      if self.dry:
        return Path("<UNSET:gate_configs['" + config_key + "']>")
      raise GateFailure("board_config.json gate_configs['" + config_key
                        + "'] is unset; author the smoke YAML and set "
                        "its path")
    path = self.config_yaml_name(value)
    if not self.dry and not path.exists():
      raise GateFailure("gate config not found: " + str(path)
                        + " (gate_configs['" + config_key + "'])")
    return path

  def evaluate_yaml(self):
    """The cobaya evaluate YAML for cobaya-adapter (repo-relative or absolute)."""
    value = self.cfg.get("evaluate_yaml")
    if value is None:
      if self.dry:
        return Path("<UNSET:evaluate_yaml>")
      raise GateFailure("board_config.json evaluate_yaml is unset")
    path = Path(value)
    return path if path.is_absolute() else self.repo / path

  def rootdir(self):
    """The cocoa $ROOTDIR (cwd for cobaya-run), or a dry placeholder.

    Reads the effective rootdir resolved at config load (the file's explicit
    override, else the $ROOTDIR environment variable). Preflight already
    failed loudly if it was unresolved, so on a real run this is set.
    """
    value = self.cfg.get("rootdir")
    if value is None:
      if self.dry:
        return Path("<UNSET:rootdir>")
      raise GateFailure("rootdir is unresolved: board_config.json rootdir "
                        "is null and $ROOTDIR is not set")
    return Path(value)

  def golden_base(self, gate_id):
    """The pre-feature commit for a gate's golden leg, or None."""
    return self.cfg.get("golden_bases", {}).get(gate_id)

  # ---- the training driver ----------------------------------------------

  def run_driver(self, *, yaml_path, cwd=None, extra=(), allow_fail=False,
                 driver=_DRIVER):
    """Run a driver on a YAML with the configured deploy paths.

    Arguments:
      yaml_path  = the resolved config path (from require_config or
                   config_yaml_name).
      cwd        = the tree to run in (default the repo; a worktree path
                   for the pinned golden leg, so its own driver and
                   emulator package are used).
      extra      = extra driver flags (e.g. ("--diagnostic",)).
      driver     = the driver filename; defaults to the
                   single-train driver, overridden e.g. by the npce-training
                   sweep leg with cosmic_shear_sweep_ntrain_emulator.py.
      allow_fail = passed through to sh (a gate that asserts on a
                   nonzero exit sets it True).

    Returns:
      (returncode, output) from the driver run.
    """
    root = self.cfg.get("driver_root")
    fileroot = self.cfg.get("driver_fileroot")
    if not self.dry and (root is None or fileroot is None):
      raise GateFailure("board_config.json driver_root / driver_fileroot "
                        "are unset; set the deploy --root and --fileroot")
    where = self.repo if cwd is None else Path(cwd)
    driver_path = where / driver
    # a pinned golden worktree may predate the family-first driver
    # rename, so resolve the filename by existence: today's name first,
    # then the legacy verb-first name; neither existing is a loud error
    # naming both candidates (the run-10 ema-off-identity catch — the
    # golden leg must fail on the byte identity, never on a filename).
    if not self.dry and not driver_path.exists():
      legacy = _LEGACY_DRIVERS.get(driver)
      if legacy is not None and (where / legacy).exists():
        driver_path = where / legacy
      else:
        raise GateFailure(
          "driver " + driver + " not found in " + str(where)
          + ("" if legacy is None
             else " (legacy candidate " + legacy + " also missing)")
          + "; a pinned golden build must carry one of the known driver "
          "names")
    cmd = [self.python, str(driver_path),
           "--root=" + ("<UNSET:driver_root>" if root is None else root),
           "--fileroot=" + ("<UNSET:driver_fileroot>" if fileroot is None
                            else fileroot),
           "--yaml=" + str(yaml_path)]
    for flag in extra:
      cmd.append(flag)
    return self.sh(cmd=cmd, cwd=where, allow_fail=allow_fail)

  # ---- the temporary worktree -------------------------------------------

  @contextlib.contextmanager
  def worktree(self, *, commit):
    """Yield a throwaway git worktree pinned at a commit, always removed.

    The golden run: a pre-feature build runs in this
    worktree so the pinned code is exercised without a checkout in the
    user's tree. The worktree is removed in a finally, so a failed gate
    still leaves the tree clean.

    Arguments:
      commit = the commit-ish to pin the worktree at.

    Yields:
      the Path to the worktree root (a dry placeholder in dry mode).
    """
    if self.dry:
      self._emit("[dry-run] would run (in " + str(self.repo)
                 + "): git worktree add --detach <tmp> " + commit + "\n")
      yield Path("<dry-worktree:" + commit + ">")
      self._emit("[dry-run] would run: git worktree remove --force <tmp>\n")
      return

    holder = Path(tempfile.mkdtemp(prefix="gates-wt-"))
    wt = holder / ("build-" + commit.replace("/", "_"))
    try:
      self.sh(cmd=["git", "worktree", "add", "--detach", str(wt), commit])
      yield wt
    finally:
      self.sh(cmd=["git", "worktree", "remove", "--force", str(wt)],
              allow_fail=True)
      shutil.rmtree(holder, ignore_errors=True)

  # ---- golden config staging --------------------------------------------

  @contextlib.contextmanager
  def staged_golden(self, *, gate_id, source):
    """Copy a golden config into the driver fileroot; yield its bare name.

    The pinned golden leg runs a worktree checkout of a pre-fix commit,
    whose driver re-prefixes an absolute --yaml under its own fileroot
    (the absolute-path passthrough postdates that commit). Copying the
    golden config into <rootdir>/<driver_root>/<driver_fileroot>/ under a
    gates-golden-<gate>.yaml name and passing the BARE filename to both
    legs makes the fileroot convention work on every commit, old or new.
    The staged copy is removed in a finally (this assumes the deploy's
    rootdir equals $ROOTDIR, so the path the driver builds from its
    --root / --fileroot flags is the one staged into).

    Arguments:
      gate_id = the gate id (names the staged file gates-golden-<id>.yaml).
      source  = the resolved absolute path of the golden config to copy.

    Yields:
      the bare filename to pass as --yaml to both legs.
    """
    name = "gates-golden-" + gate_id + ".yaml"
    root = self.cfg.get("rootdir")
    driver_root = self.cfg.get("driver_root")
    driver_fileroot = self.cfg.get("driver_fileroot")
    fileroot = Path(str(root)) / str(driver_root) / str(driver_fileroot)
    dest = fileroot / name
    if self.dry:
      self._emit("[dry-run] would stage " + str(source) + " -> "
                 + str(dest) + "\n")
      yield name
      self._emit("[dry-run] would remove " + str(dest) + "\n")
      return

    try:
      fileroot.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(str(source), str(dest))
      self._emit("[harness] staged golden config -> " + str(dest) + "\n")
      yield name
    finally:
      if dest.exists():
        dest.unlink()


# --------------------------------------------------------------------------
# Preflight (harness rule 4).
# --------------------------------------------------------------------------

def _git(args):
  """Run a git command from the repo, returning (rc, stdout stripped)."""
  proc = subprocess.run(["git"] + args,
                        cwd=str(_REPO),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True)
  return (proc.returncode, proc.stdout.strip())


def _probe_import(statement):
  """Whether a python import statement succeeds in the active env."""
  proc = subprocess.run([sys.executable, "-c", statement],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
  return proc.returncode == 0


def _dirty_lines(porcelain_out):
  """The clean-tree offenders, excluding gates/board_config.json.

  board_config.json is machine-portable and normally runs unedited, but a
  user may override a value for a non-standard deploy, so a modified config
  must not fail the clean-tree check; it is excluded here and its effective
  values are dumped into every gate-log header instead, so reproducibility
  is still kept.

  Arguments:
    porcelain_out = the ``git status --porcelain`` output over the
                    watched paths.

  Returns:
    a list of the offending status lines, board_config.json removed.
  """
  offenders = []
  for line in porcelain_out.splitlines():
    if line.strip() == "":
      continue
    path = line[3:].strip()
    if path == "gates/board_config.json":
      continue
    offenders.append(line)
  return offenders


def preflight(cfg):
  """Run every pre-GPU check, printing remedies; return (ok, env).

  Checks, in order: (a) the base-notes commit is an ancestor of HEAD;
  (b) the working tree is clean in emulator/, gates/, and the drivers;
  (c) torch (with CUDA), cosmolike, and cobaya import; (d) the effective
  rootdir resolves (from $ROOTDIR or an explicit override, its source
  printed) and the data paths board_config.json names exist; (e) the
  debug key is a bool; (f) driver_fileroot is a real value, not the
  shipped placeholder. Any failure prints a remedy and the whole function
  returns ok False, so the caller exits nonzero before spending GPU time.

  Arguments:
    cfg = the parsed board_config.json.

  Returns:
    (ok, env) where env maps each capability name to a bool (torch /
    cosmolike / cobaya / gpu), consumed by RunContext.require_caps.
  """
  ok = True
  env = {"torch": False,
         "cosmolike": False,
         "cobaya": False,
         "gpu": False}

  print("== preflight ==")

  # (a) base-notes commit ancestor of HEAD (never exact-tip equality).
  rc_anc, _ = _git(["merge-base", "--is-ancestor",
                    _BASE_NOTES_COMMIT, "HEAD"])
  if rc_anc == 0:
    print("  [ok] base-notes commit " + _BASE_NOTES_COMMIT[:9]
          + " is an ancestor of HEAD")
  else:
    ok = False
    print("  [FAIL] base-notes commit " + _BASE_NOTES_COMMIT[:9]
          + " is NOT an ancestor of HEAD")
    print("         remedy: git pull -- the tip must be the harness "
          "commit (a descendant of the board + spec commit)")

  # (b) clean tree in emulator/, gates/, and the root drivers, but NOT
  # gates/board_config.json (portable; excluded so a local deploy override
  # does not fail the clean-tree check).
  watched = ["emulator", "gates"]
  for entry in sorted(_REPO.glob("*.py")):
    watched.append(entry.name)
  rc_st, out_st = _git(["status", "--porcelain", "--"] + watched)
  offenders = _dirty_lines(out_st)
  if len(offenders) == 0:
    print("  [ok] working tree clean in emulator/ + gates/ + drivers "
          "(board_config.json excluded)")
  else:
    ok = False
    print("  [FAIL] dirty working tree (a run must be reproducible):")
    for line in offenders:
      print("         " + line)
    print("         remedy: commit or stash your changes, then rerun")

  # (c) the cocoa stack imports.
  for name, statement in _REQUIRED_IMPORTS:
    if _probe_import(statement):
      env[name] = True
      print("  [ok] import " + name)
    else:
      ok = False
      print("  [FAIL] cannot " + statement)
      print("         remedy: activate the cocoa env (source start_cocoa)")
  if env["torch"] and _probe_import("import torch; "
                                    "assert torch.cuda.is_available()"):
    env["gpu"] = True
    print("  [ok] CUDA visible to torch")
  else:
    ok = False
    print("  [FAIL] CUDA not visible to torch")
    print("         remedy: run on the workstation GPUs, not the dev Mac")

  # (d) the effective rootdir (resolved at load) exists, with its source
  # named; then driver_root and yaml_dir exist, resolving against that
  # rootdir when relative.
  rootdir_value = cfg.get("rootdir")
  rootdir_source = cfg.get("rootdir_source", "unset")
  if rootdir_value is None:
    ok = False
    print("  [FAIL] rootdir is unresolved: board_config.json rootdir is "
          "null and $ROOTDIR is not set")
    print("         remedy: export ROOTDIR to the cocoa clone root (source "
          "start_cocoa), or set an absolute rootdir in board_config.json")
  elif not Path(rootdir_value).exists():
    ok = False
    print("  [FAIL] rootdir does not exist: " + rootdir_value
          + "  (source: " + rootdir_source + ")")
  else:
    print("  [ok] rootdir = " + rootdir_value
          + "  (source: " + rootdir_source + ")")

  for key in ("driver_root", "yaml_dir"):
    value = cfg.get(key)
    if value is None:
      ok = False
      print("  [FAIL] board_config.json " + key + " is unset")
      print("         remedy: edit gates/board_config.json with the "
            "deploy path for " + key)
      continue
    resolved = Path(value)
    if not resolved.is_absolute() and rootdir_value is not None:
      resolved = Path(rootdir_value) / value
    if not resolved.exists():
      ok = False
      print("  [FAIL] board_config.json " + key + " does not exist: "
            + str(resolved))
    else:
      print("  [ok] path " + key + " -> " + str(resolved))

  # (e) the quiet-mode switch is a required key (a bool); a config that
  # predates the terminal-output rule is missing it, which must fail as
  # loudly as a missing path.
  if "debug" not in cfg:
    ok = False
    print("  [FAIL] board_config.json is missing the required 'debug' key")
    print("         remedy: add \"debug\": false (or true for full output)")
  elif not isinstance(cfg["debug"], bool):
    ok = False
    print("  [FAIL] board_config.json 'debug' must be true or false, got "
          + repr(cfg["debug"]))
  else:
    print("  [ok] debug = " + str(cfg["debug"]))

  # (f) driver_fileroot is a real value, not the shipped placeholder. The
  # committed config carries "gates_board"; a fileroot that is missing,
  # empty, or still wrapped in < > (the placeholder signature) fails here so
  # a run never trains under a literal "<...>" stem.
  fileroot = cfg.get("driver_fileroot")
  if fileroot is None or str(fileroot).strip() == "":
    ok = False
    print("  [FAIL] board_config.json driver_fileroot is unset or empty")
    print("         remedy: set driver_fileroot to the run's file stem "
          "(see the _help 'driver_fileroot' entry); the shipped default "
          "is \"gates_board\"")
  elif "<" in str(fileroot) or ">" in str(fileroot):
    ok = False
    print("  [FAIL] board_config.json driver_fileroot is still the "
          "placeholder: " + repr(fileroot))
    print("         remedy: replace it with the run's file stem (see the "
          "_help 'driver_fileroot' entry); the shipped default is "
          "\"gates_board\"")
  else:
    print("  [ok] driver_fileroot = " + repr(fileroot))

  print("== preflight " + ("PASSED" if ok else "FAILED") + " ==")
  return (ok, env)


# --------------------------------------------------------------------------
# Selection, status, and the BOARD.md / board_status.json writers.
# --------------------------------------------------------------------------

def _resolve_rootdir(cfg):
  """Resolve the effective rootdir and where it came from, at load time.

  Precedence: a non-null rootdir in board_config.json wins (an operator
  override, for a machine whose $ROOTDIR does not point at the clone under
  test); otherwise the ROOTDIR environment variable (the normal case, so the
  committed config carries no machine path); otherwise it stays unresolved
  and the preflight rootdir check fails loudly naming the variable. The file
  is never rewritten; the value lives only in the in-memory config.

  Arguments:
    cfg = the parsed board_config.json.

  Returns:
    (value, source): value is the rootdir string, or None when unresolved;
    source is "board_config.json", "$ROOTDIR", or "unset".
  """
  explicit = cfg.get("rootdir")
  if explicit is not None:
    return str(explicit), "board_config.json"
  env_value = os.environ.get("ROOTDIR")
  if env_value:
    return env_value, "$ROOTDIR"
  return None, "unset"


def _load_config():
  """Parse board_config.json and resolve this run's rootdir in memory.

  Reads the file (a clear error if it is malformed), then resolves rootdir
  from the file's own explicit value or the $ROOTDIR environment variable
  (see _resolve_rootdir), so a machine-portable config needs no edit. The
  resolved value replaces cfg["rootdir"] and its origin is stashed under
  cfg["rootdir_source"] for the preflight line and the per-gate config dump;
  the file on disk is left untouched.
  """
  with open(_CONFIG_FILE, "r") as handle:
    cfg = json.load(handle)
  value, source = _resolve_rootdir(cfg)
  cfg["rootdir"] = value
  cfg["rootdir_source"] = source
  return cfg


def _load_status():
  """Load board_status.json, or an empty map on the first run."""
  if not _STATUS_FILE.exists():
    return {}
  with open(_STATUS_FILE, "r") as handle:
    return json.load(handle)


def _save_status(status):
  """Persist board_status.json (the resume + BOARD.md source)."""
  _LOGS_DIR.mkdir(exist_ok=True)
  with open(_STATUS_FILE, "w") as handle:
    json.dump(status, handle, indent=2, sort_keys=True)


def _write_board_md(status):
  """Write the human-readable BOARD.md pass/fail table."""
  _LOGS_DIR.mkdir(exist_ok=True)
  lines = []
  lines.append("# Workstation board run")
  lines.append("")
  lines.append("Base-notes commit: `" + _BASE_NOTES_COMMIT + "`")
  lines.append("")
  lines.append("| Gate | Tier | Status | Detail | Log |")
  lines.append("|------|------|--------|--------|-----|")
  for gate in BOARD:
    record = status.get(gate.id, {})
    state = record.get("status", "not run")
    detail = record.get("detail", "")
    log_cell = gate.id + ".log" if state in ("PASS", "FAIL") else ""
    lines.append("| " + gate.id + " | " + gate.tier + " | " + state
                 + " | " + detail.replace("|", "/") + " | " + log_cell + " |")
  lines.append("")
  with open(_BOARD_MD, "w") as handle:
    handle.write("\n".join(lines) + "\n")


def select_gates(args):
  """Build the ordered gate selection from the CLI selectors.

  Default is the whole board minus the optional gates. --gate names an
  explicit set (optional gates allowed). --tier picks one tier. --from
  starts the board at a gate. The result stays in board order.

  Arguments:
    args = the parsed argparse namespace.

  Returns:
    a list of Gate in board order.
  """
  if args.gate:
    wanted = set(args.gate)
    chosen = []
    for gate in BOARD:
      if gate.id in wanted:
        chosen.append(gate)
    seen = set()
    for gate in chosen:
      seen.add(gate.id)
    for name in args.gate:
      if name not in seen:
        print("warning: unknown gate id '" + name + "' (see --list)")
    return chosen

  if args.tier:
    chosen = []
    for gate in BOARD:
      if gate.tier == args.tier and not gate.optional:
        chosen.append(gate)
    return chosen

  if getattr(args, "from_gate", None):
    ids = []
    for gate in BOARD:
      ids.append(gate.id)
    if args.from_gate not in ids:
      print("warning: unknown --from gate '" + args.from_gate + "'")
      return []
    start = ids.index(args.from_gate)
    chosen = []
    index = 0
    for gate in BOARD:
      if index >= start and not gate.optional:
        chosen.append(gate)
      index = index + 1
    return chosen

  chosen = []
  for gate in BOARD:
    if not gate.optional:
      chosen.append(gate)
  return chosen


def _passed(status, gate_id):
  """Whether a gate is marked PASS in the status map."""
  return status.get(gate_id, {}).get("status") == "PASS"


# --------------------------------------------------------------------------
# The runner.
# --------------------------------------------------------------------------

def _now():
  """An ISO-ish timestamp for the log header and status entries."""
  return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_header(ctx, gate):
  """Write the gate log's header + the effective config (log-only).

  The full multi-line header block and the config dump are log-only: they
  land in the gate log for reproducibility and reach the terminal only in
  debug mode. The runner prints the one-line terminal gate header
  separately. Byte-compatible with the pre-quiet-mode log format (the same
  text lands in the log, just no longer echoed to the terminal by default).

  Arguments:
    ctx  = the gate's RunContext (routes to its log; quiet-aware).
    gate = the gate being run (id, tier, home, maps, worktree pin).
  """
  _, head = _git(["rev-parse", "HEAD"])
  lines = []
  lines.append("=" * 72)
  lines.append("GATE " + gate.id + "  [" + gate.tier + "]")
  lines.append("spec code: " + gate.spec_code)
  lines.append("home note: notes/" + gate.home + ".md")
  lines.append("maps (assertion -> home-note line): " + gate.maps)
  lines.append("needs: " + ", ".join(gate.needs))
  if gate.worktree_commit is not None:
    lines.append("worktree pin: " + gate.worktree_commit)
  lines.append("base-notes commit: " + _BASE_NOTES_COMMIT)
  lines.append("HEAD at run: " + head)
  lines.append("started: " + _now())
  lines.append("=" * 72)
  text = "\n".join(lines) + "\n"
  cfg_dump = ("effective board_config.json (excluded from the clean-tree "
              "check; recorded here for reproducibility):\n"
              + json.dumps(ctx.cfg, indent=2) + "\n"
              + "=" * 72 + "\n")
  ctx._emit(text, log_only=True)
  ctx._emit(cfg_dump, log_only=True)


def run_selection(*, selection, cfg, env, status, force_rerun, dry,
                  debug=False):
  """Execute the selected gates in order, logging and marking each.

  Arguments:
    selection   = the ordered gates to run (from select_gates).
    cfg         = board_config.json.
    env         = the capability map from preflight (empty in dry mode).
    status      = the resume map, updated in place and persisted after
                  each gate.
    force_rerun = gate ids to run even if already PASS.
    dry         = when True, print the plan without executing.
    debug       = when True, each gate's RunContext mirrors the full
                  command output + config dump to the terminal too.

  Returns:
    the count of gates that FAILED (0 means the selection is green).
  """
  failures = 0
  for gate in selection:
    # dry mode prints every selected gate's plan, BEFORE (and bypassing)
    # the resume + dependency checks, so --dry-run --gate cobaya-adapter shows the
    # plan with a deps annotation, not a skip line.
    if dry:
      print("\n--- plan: " + gate.id + " ---")
      if gate.deps:
        print("[dry-run] (deps: " + ", ".join(gate.deps)
              + " -- at run time this gate SKIPs unless they PASS)")
      ctx = RunContext(cfg=cfg, dry=True, log_fh=None, env={}, debug=debug)
      try:
        gate.run(ctx)
      except GateFailure as failure:
        print("[dry-run] (gate would need: " + str(failure) + ")")
      continue

    # resume: an already-passed gate is skipped unless forced.
    if _passed(status, gate.id) and gate.id not in force_rerun:
      print("[skip] " + gate.id + ": already PASS (resume); "
            "--force-rerun " + gate.id + " to rerun")
      continue

    # dependency skip: an unmet prerequisite marks SKIPPED, no abort.
    unmet = []
    for dep in gate.deps:
      if not _passed(status, dep):
        unmet.append(dep)
    if len(unmet) > 0:
      detail = "dependency not passed: " + ", ".join(unmet)
      print("[skip] " + gate.id + ": " + detail)
      status[gate.id] = {"status": "SKIP-DEP",
                         "detail": detail,
                         "ts": _now()}
      _save_status(status)
      _write_board_md(status)
      continue

    log_path = _LOGS_DIR / (gate.id + ".log")
    _LOGS_DIR.mkdir(exist_ok=True)
    # one-line terminal header; the full header block + config dump are
    # log-only content (debug mirrors them to the terminal).
    print("GATE " + gate.id + " [" + gate.tier + "] started "
          + datetime.datetime.now().strftime("%H:%M:%S"))
    with open(log_path, "w") as log_fh:
      ctx = RunContext(cfg=cfg, dry=False, log_fh=log_fh, env=env, debug=debug)
      _log_header(ctx, gate)
      outcome = "PASS"
      detail = ""
      try:
        gate.run(ctx)
      except GateFailure as failure:
        outcome = "FAIL"
        detail = str(failure)
      except Exception as unexpected:  # a crash is a gate failure, not the board's
        outcome = "FAIL"
        detail = "unexpected: " + repr(unexpected)
      # the final verdict stays on the terminal (log_only default False).
      footer = ("[harness] GATE " + gate.id + ": " + outcome
                + ("" if detail == "" else "  -- " + detail) + "\n")
      ctx._emit(footer)

    if outcome == "FAIL":
      failures = failures + 1
    status[gate.id] = {"status": outcome,
                       "detail": detail,
                       "ts": _now()}
    _save_status(status)
    _write_board_md(status)

  return failures


# --------------------------------------------------------------------------
# The CLI.
# --------------------------------------------------------------------------

def cmd_list(status):
  """Print the board with each gate's current status (the --list view)."""
  print("Workstation board (base-notes " + _BASE_NOTES_COMMIT[:9] + "):")
  current_tier = None
  for gate in BOARD:
    if gate.tier != current_tier:
      current_tier = gate.tier
      print("\n[" + current_tier + "]")
    state = status.get(gate.id, {}).get("status", "not run")
    flags = []
    if gate.optional:
      flags.append("optional")
    if gate.deps:
      flags.append("deps: " + ",".join(gate.deps))
    if gate.worktree_commit is not None:
      flags.append("worktree@" + gate.worktree_commit)
    tail = "" if len(flags) == 0 else "  (" + "; ".join(flags) + ")"
    print("  " + gate.id.ljust(26) + state.ljust(10)
          + "home: " + gate.home + tail)


def build_parser():
  """Assemble the argparse CLI (also the printed --help usage)."""
  parser = argparse.ArgumentParser(
    prog="run_board.py",
    description="Drive the workstation gates board (see the header "
                "docstring for the full user flow).")
  parser.add_argument("--check",
                      action="store_true",
                      help="run preflight only, then exit")
  parser.add_argument("--list",
                      action="store_true",
                      help="print the board with each gate's status")
  parser.add_argument("--dry-run",
                      action="store_true",
                      help="print every command a selection would run")
  parser.add_argument("--gate",
                      nargs="+",
                      metavar="ID",
                      help="run only these gate ids (optional gates allowed)")
  parser.add_argument("--tier",
                      choices=(board.TIER_BACKLOG,
                               board.TIER_NEW_FEATURES,
                               board.TIER_SAVE_AND_SAMPLE),
                      help="run only this tier")
  parser.add_argument("--from",
                      dest="from_gate",
                      metavar="ID",
                      help="run the board from this gate onward")
  parser.add_argument("--force-rerun",
                      nargs="+",
                      default=[],
                      metavar="ID",
                      help="rerun these gates even if already PASS")
  parser.add_argument("--force-rerun-all",
                      action="store_true",
                      help="rerun EVERY selected gate even if already "
                           "PASS (the full regression pass: ignores the "
                           "resume map without deleting it, so an "
                           "interrupted run still resumes within itself)")
  parser.add_argument("--debug",
                      action="store_true",
                      help="mirror the full command output + config dump to "
                           "the terminal (forces board_config.json debug "
                           "true; the logs always get everything)")
  return parser


def main(argv=None):
  """Parse the CLI and run the requested action.

  Arguments:
    argv = the argument list (defaults to sys.argv[1:]); accepted for
           the harness self-tests.

  Returns:
    the process exit code (0 on success / green, nonzero otherwise).
  """
  args = build_parser().parse_args(argv)
  cfg = _load_config()
  status = _load_status()
  # effective quiet/debug: the committed board_config.json debug key, or
  # the --debug flag forcing it true. (preflight enforces the key's
  # presence for real runs; dry-run / --list do not need it.)
  debug = bool(cfg.get("debug", False)) or args.debug

  if args.list:
    cmd_list(status)
    return 0

  if args.check:
    ok, _ = preflight(cfg)
    return 0 if ok else 1

  selection = select_gates(args)
  if len(selection) == 0:
    print("no gates selected")
    return 0

  # --force-rerun-all = force every SELECTED gate (composes with
  # --gate / --tier / --from). Built here as the forced-id set so
  # run_selection needs no second code path; the resume map itself is
  # untouched, so interrupting the regression pass and re-running
  # WITHOUT the flag resumes from whatever it re-proved.
  forced = set(args.force_rerun)
  if args.force_rerun_all:
    for gate in selection:
      forced.add(gate.id)

  if args.dry_run:
    print("dry-run plan (" + str(len(selection)) + " gates, in order):")
    run_selection(selection=selection, cfg=cfg, env={}, status=status,
                  force_rerun=forced, dry=True, debug=debug)
    return 0

  ok, env = preflight(cfg)
  if not ok:
    print("preflight failed; not running any gate")
    return 1

  failures = run_selection(selection=selection, cfg=cfg, env=env,
                           status=status, force_rerun=forced,
                           dry=False, debug=debug)
  print("\nboard run complete: " + str(failures) + " gate(s) FAILED; "
        "see gates/logs/BOARD.md")
  return 1 if failures > 0 else 0


if __name__ == "__main__":
  sys.exit(main())
