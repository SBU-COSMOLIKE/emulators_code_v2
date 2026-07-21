#!/usr/bin/env python3
"""This driver runs an Optuna search over one xi emulator's hyperparameters.

PS: memmap = a NumPy array backed by the on-disk dump file, read in slices
so an array larger than RAM is never loaded whole.
"""

#-------------------------------------------------------------------------------
# How to run this program
#-------------------------------------------------------------------------------
# This driver is the tuning twin of cosmic_shear_train_emulator.py: it
# reuses the same single cosmic-shear (xi) emulator setup (resmlp | rescnn |
# restrf, optionally with a factored ia design, per the YAML), but runs an
# Optuna study minimizing validation f(delta-chi2 > 0.2) rather than one run.
#
# python .../emultrfv2/cosmic_shear_tune_emulator.py \
#   --root projects/lsst_y1/ \
#   --fileroot emulators/training_scripts/ \
#   --yaml cosmic_shear_train_emulator.yaml \
#   --n-trials 50 --timeout 4200
#
#- The searched hyperparameters come from the YAML train_args block. Each leaf is
#  a fixed scalar or a range: a 4-item list [default, min, max, kind], kind int /
#  float / log (a whitespace string "default min max kind" also works). A range
#  may sit at any nesting depth: inside the model sub-blocks (model.mlp.width,
#  model.cnn.kernel_size), the schedules (trim.start), or the per-phase
#  trunk / head override blocks (head.lr.lr_base):
#
#      lr:
#        lr_base: [0.0025, 1.0e-5, 1.0e-1, log]   # searched, log scale
#        bs_base: 64.0                            # fixed
#      model:
#        mlp:
#          width:    [128, 64, 256, int]          # searched, integer
#          n_blocks: 4                            # fixed
#
#  The first value is the default: the training driver uses it and this search
#  warm-starts trial 0 from it, so one YAML serves both drivers.
#
#- Multiple GPUs (one node): trials are independent trainings, so the study
#  parallelizes, one worker process per GPU (spawn), all sharing a single study
#  through an Optuna journal-file storage under --fileroot. Each worker builds
#  its own experiment on its GPU, asks the shared study for suggestions
#  (sampler seeded per worker), and reports results back; --n-trials is the
#  total across workers. The journal file persists: rerunning with the same
#  --journal resumes only when its scientific manifest matches exactly.
#  Pass a fresh --journal name to start a scientifically different study. One GPU
#  (or the Apple-MPS dev machine) keeps the original serial in-memory study,
#  no file written.
#
#- `--root` (required): project folder under $ROOTDIR (data resolves under it);
#  `--fileroot` (required): subfolder holding the YAML (e.g.
#  emulators/training_scripts). Cocoa layout, as in the training driver.
#- `--yaml` (default test.yaml): config under --fileroot, or an absolute path
#  used as-is, `data` + `train_args` blocks (training driver schema; train_args
#  may now carry ranges). The `data` block lists bare filenames, resolved under
#  --root/chains.
#- `--n-trials` (default 50) and `--timeout` (seconds, optional) bound the study
#  (the timeout applies per worker in the parallel path).
#- `--n-gpus` (default: all visible CUDA devices): GPUs to spread trials
#  across. 1, or no CUDA, is the serial path.
#- `--journal` (default tune_journal.log): the shared study's journal file
#  under --fileroot (parallel path only). The same name resumes only an
#  identical scientific manifest; a new name starts a different study.
#- `--rescale` / `--activation` set the analytic-R mode and ResBlock activation,
#  fixed across the study, not searched (see the training driver).
#- `--quiet` suppresses all stdout (per-trial lines and final summary).
#
#- The fixed single-emulator choices (probe = xi, AdamW, ReduceLROnPlateau,
#  use_amp = False, report thresholds, the (name, ia) MODELS registry) are
#  EmulatorExperiment defaults (emulator/experiment.py), shared with the training
#  driver. The model is the YAML's (train_args.model.name = resmlp | rescnn |
#  restrf, plus model.ia = omit | nla | tatt), also fixed; only hyperparameters
#  vary.
#
#- Output: stdout, a per-trial line (frac>0.2, running best, params) and a
#  final summary of the best frac>0.2 and params, plus, in the parallel
#  path, the persistent journal file.
#-------------------------------------------------------------------------------

import argparse
from pathlib import Path

import optuna
import torch

# This script sits beside the emulator/ package (same .../emultrfv2/ folder),
# so launching it by path makes its own directory sys.path[0] and
# `import emulator` resolves with no path manipulation. Run it from $ROOTDIR;
# emulator.cocoa reads $ROOTDIR to resolve the data paths.

from emulator.cocoa import (
  add_cocoa_path_args, resolve_cocoa_config, cocoa_output)
from emulator.training import suggest_train_args, search_defaults
from emulator.experiment import EmulatorExperiment, validate_sweep_paths


def journal_storage(path):
  """
  Open (or create) an Optuna journal-file storage.

  The journal backend is Optuna's recommended storage for several
  processes sharing one study through the filesystem: every worker
  appends its trial records to the file under a file lock, so no
  database server is needed and a crashed run leaves a readable
  log. The import shim covers the backend's rename
  (JournalFileStorage in optuna 3.x -> JournalFileBackend in 4.x).

  Arguments:
    path = the journal file's path (created if absent; appending to
           an existing file resumes its studies).

  Returns:
    an optuna.storages.JournalStorage.
  """
  from optuna.storages import JournalStorage
  try:
    from optuna.storages.journal import JournalFileBackend
  except ImportError:                     # optuna < 4.0
    from optuna.storages import JournalFileStorage as JournalFileBackend
  return JournalStorage(JournalFileBackend(path))


def open_journal_study(study_name, journal_path):
  """Open (creating if absent) one named study inside a journal file.

  Arguments:
    study_name   = stable family-owned study name (resolve_study_name).
    journal_path = journal file shared by all workers.

  Returns:
    the optuna.Study. Reusing a journal resumes its recorded trials; the
    user owns not resuming a journal with a changed YAML or dataset (a
    new --journal name starts a clean study).
  """
  return optuna.create_study(
    study_name=study_name,
    storage=journal_storage(journal_path),
    direction="minimize",
    load_if_exists=True)


def resolve_study_name(family):
  """The stable Optuna study name owned by one emulator family.

  The command name is deliberately not an input: a wrapper may be
  renamed without changing the study an existing journal resumes.

  Arguments:
    family = the emulator family key ("cosmolike", "outputs", "cmb",
             "grid", or "grid2d").

  Returns:
    the study-name string for that family.

  Raises:
    ValueError for an unknown family key.
  """
  names = {
    "cosmolike": "cosmic_shear_tune",
    "outputs":   "scalar_tune_emulator",
    "cmb":       "cmb_tune_emulator",
    "grid":      "baosn_tune_emulator",
    "grid2d":    "mps_tune_emulator",
  }
  if family not in names:
    raise ValueError(
      "unknown emulator family " + repr(family) + "; expected one of "
      + repr(sorted(names)))
  return names[family]


def enqueue_default_control(study, default_trial):
  """Enqueue the YAML-default control trial once, despite failures.

  The first trial of a fresh study evaluates the YAML's own default
  values (each range's first entry), so the search always contains the
  known-good starting configuration. Returns true only when this call
  enqueues the control. Its marker belongs to the queued trial itself,
  so an unrelated failed or abandoned trial does not suppress the
  control and a second invocation cannot duplicate it.
  """
  if not default_trial:
    return False
  for trial in study.trials:
    if trial.user_attrs.get("cocoa_default_control", False):
      return False
  study.enqueue_trial(
    default_trial,
    user_attrs={"cocoa_default_control": True})
  return True


def refuse_failed_workers(processes):
  """Refuse a parallel result when any joined worker exited nonzero."""
  failed = []
  for process in processes:
    if process.exitcode != 0:
      failed.append(process.exitcode)
  if failed:
    raise RuntimeError(
      "tuning worker failure; refusing to report a winner from an "
      "incomplete run. Exit codes: " + repr(failed))


def best_complete_trial(study):
  """Select the winner using the objective rule recorded in the manifest.

  Arguments:
    study = completed Optuna study whose trials carry objective values and
            the per-trial median tie break.

  Returns:
    the COMPLETE trial with lowest objective value, then lowest median.
  """
  complete = []
  for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
      complete.append(trial)
  if not complete:
    raise RuntimeError("no trial completed; refusing to select a winner")

  def rank(trial):
    median = trial.user_attrs.get("median", float("inf"))
    return (trial.value, median)

  return min(complete, key=rank)


def _tune_worker(gpu_id, n_trials, cfg, rescale, activation,
                 journal_path, study_name, timeout, quiet):
  """
  One GPU's share of the study; runs in its own spawned process.

  Loads the shared journal, stages its own copy of the data, and runs
  its trial share.
  Each process has its own cosmolike global state and runs its share of trials:
  Optuna serializes suggestions/reports through the storage, so
  workers cooperate on one search history. The sampler is seeded
  per worker (seed = gpu_id) because identical seeds would propose
  identical points in parallel.

  Arguments:
    gpu_id       = CUDA device index this worker owns.
    n_trials     = this worker's trial share (the total splits
                   across workers).
    cfg          = parsed YAML config; host ram_frac already divided
                   across workers by the parent.
    rescale      = analytic-R mode, forwarded to the experiment.
    activation   = ResBlock activation name, forwarded likewise.
    journal_path = the shared study's journal file.
    study_name   = the stable family-owned name inside the journal.
    timeout      = per-worker optimize() timeout in seconds (None =
                   unbounded).
    quiet        = suppress this worker's per-trial lines.
  """
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  study = optuna.load_study(
    study_name=study_name,
    storage=journal_storage(journal_path),
    sampler=optuna.samplers.TPESampler(seed=gpu_id))

  torch.cuda.set_device(gpu_id)
  device = torch.device(f"cuda:{gpu_id}")
  exp = EmulatorExperiment.from_config(cfg,
                                       device=device,
                                       rescale=rescale,
                                       activation=activation,
                                       quiet=True)
  raw_ta = exp.raw_train_args
  exp.stage_train()
  exp.stage_val()
  exp.build_geometry()

  def objective(trial):
    """
    Score one Optuna trial: resolve its train_args, train, return frac>0.2.

    Arguments:
      trial = the Optuna trial (its suggestions resolve the searched
              train_args leaves).

    Returns:
      the best epoch's frac>0.2 (minimized); the median is stashed as
      a trial user attribute for the tiebreak/report.
    """
    # suggest_train_args (training.py): draw each searched leaf's value
    # from the trial and fold it into a concrete train_args mapping.
    ta = suggest_train_args(trial=trial, train_args=raw_ta)
    (_m, _tl, medians,
     _mn, fracs) = exp.train(train_args=ta, silent=True)
    def epoch_rank(i):
      return (fracs[i][0].item(), medians[i])
    best = min(range(len(fracs)), key=epoch_rank)
    trial.set_user_attr("median", float(medians[best]))
    return fracs[best][0].item()

  def log_trial(study_, trial):
    """
    Optuna callback: print this worker's per-trial result line.

    Arguments:
      study_ = the shared study (read for the running best value).
      trial  = the trial that just finished (number, value, params).

    Returns:
      None (prints a line unless --quiet).
    """
    if not quiet:
      print(f"[gpu {gpu_id}] trial {trial.number:3d}  "
            f"frac>0.2 {trial.value:.4f}  "
            f"best {study_.best_value:.4f}  {trial.params}")

  study.optimize(objective,
                 n_trials=n_trials,
                 timeout=timeout,
                 callbacks=[log_trial])


def main(prog="cosmic_shear_tune_emulator", family="cosmolike"):
  parser = argparse.ArgumentParser(prog=prog)
  # --root / --fileroot / --yaml: the cocoa project layout (data under
  # --root, YAML under --fileroot; train_args may carry [default, min,
  # max, kind] ranges).
  add_cocoa_path_args(parser)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across "
                           "the study: 'none' (default), 'rescaled' "
                           "(v1), or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "study: 'H', 'power', 'multigate', or "
                           "'gated_power'. Overrides the YAML "
                           "train_args.model.activation; default: "
                           "the YAML's choice, else 'H'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default=None)
  parser.add_argument("--n-trials",
                      dest="n_trials",
                      help="number of Optuna trials (default 50)",
                      type=int,
                      default=50)
  parser.add_argument("--timeout",
                      dest="timeout",
                      help="stop the study after this many seconds "
                           "(optional; default no limit; applies "
                           "per worker in the parallel path)",
                      type=int,
                      default=None)
  parser.add_argument("--n-gpus",
                      dest="n_gpus",
                      help="number of GPUs to spread trials across "
                           "(default: all visible CUDA devices). "
                           "1, or no CUDA, is the serial path.",
                      type=int,
                      default=None)
  parser.add_argument("--journal",
                      dest="journal",
                      help="journal file (under --fileroot) the "
                           "parallel workers share the study "
                           "through (default tune_journal.log). "
                           "Persistent: the same name resumes the "
                           "recorded trials; a new name starts a "
                           "different study",
                      type=str,
                      default="tune_journal.log")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (per-trial lines "
                           "and the final summary)",
                      action="store_true")
  # strict parse: a misspelled flag (--sav, --activaton, --diagnostc) is a
  # usage error naming the token and exiting nonzero, never silently ignored
  # and then run at a default (which could publish to the wrong --save root).
  args = parser.parse_args()

  # resolve_cocoa_config (cocoa.py): resolve the cocoa layout (data under
  # $ROOTDIR/<root>, YAML under <fileroot>), load the YAML, and make its data
  # paths absolute. The fileroot also hosts the parallel path's journal file.
  cfg, fileroot, _ = resolve_cocoa_config(args)
  # reject a wrong-family YAML at startup (family is always a real identity:
  # a per-family wrapper passes its key, a direct run owns "cosmolike").
  from cosmic_shear_train_emulator import require_family_block
  require_family_block(data=cfg["data"], family=family, prog=prog)
  # per-family studies own their name (one journal file can never
  # mix families); the cosmic-shear study keeps its historic name.
  study_name = resolve_study_name(family=family)

  # Setup (config parse, model resolution, device, data staging, geometry,
  # chi2, per-run spec assembly) lives in EmulatorExperiment, shared with the
  # training driver. Geometry / chi2 / activation are fixed, so build them once
  # here; only the searched train_args vary per trial. Single-emulator choices are
  # EmulatorExperiment defaults; model is the YAML's (train_args.model.name).
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=args.activation,
                                       quiet=args.quiet)
  # the experiment's quiet-gated logger
  log = exp.log
  # print_design (experiment.py): the startup banner (the resolved
  # model block, run knobs, guards, every train_args sub-block, and the
  # physical cuts; search ranges already collapsed to their defaults, the
  # study varies them per trial from there). A stale YAML here would
  # waste a whole study, not one training. Shared with the train / sweep
  # drivers.
  exp.print_design()

  # raw_ta keeps the ranges (exp.train_args collapsed them to defaults), while
  # ranges is the manifest-owned warm-start control.
  raw_ta = exp.raw_train_args
  ranges = search_defaults(train_args=raw_ta)
  # validate_sweep_paths (experiment.py): reject phase-schedule search
  # axes (head. / trunk_epochs / trunk.) on a single-phase model before
  # any worker spawns, so a dead or disguised search dimension fails at
  # startup instead of being sampled every trial. ranges is keyed by the
  # searched dotted paths, so its keys are exactly the leaves to check.
  validate_sweep_paths(
    paths=list(ranges),
    two_phase=hasattr(exp.model_cls, "set_train_phase"))
  if not ranges:
    log("WARNING: no [default, min, max, kind] search ranges in "
        "train_args: every trial is identical.")

  # quiet Optuna's per-trial INFO spam (we print our own line)
  optuna.logging.set_verbosity(optuna.logging.WARNING)

  # how many GPU workers: capped by what is visible, by --n-gpus, and
  # by the trial count. 1 (or the MPS dev machine) keeps the original
  # serial in-memory study.
  n_cuda    = torch.cuda.device_count()
  n_request = n_cuda if args.n_gpus is None else min(args.n_gpus,
                                                     n_cuda)
  n_workers = min(n_request, args.n_trials)

  if n_workers <= 1:
    log("serial study (1 worker, in-memory)")
    log("loading sources:")
    exp.stage_train()
    exp.stage_val()
    exp.build_geometry()

    def objective(trial):
      """
      Score one serial-path trial: resolve train_args, train, return frac>0.2.

      Arguments:
        trial = the Optuna trial (each searched range becomes one
                suggestion).

      Returns:
        the best epoch's frac>0.2 (minimized); the median is stashed
        as a trial user attribute for the tiebreak/report.
      """
      # suggest_train_args (training.py): fold this trial's suggestions
      # into a concrete train_args mapping. exp.train then builds the
      # per-run specs (model / optimizer / scheduler + activation +
      # ResCNN geom) on the fixed data + geometry; silent=True keeps
      # each trial quiet even when the study isn't.
      ta = suggest_train_args(trial=trial, train_args=raw_ta)
      (_m, _tl, medians,
       _mn, fracs) = exp.train(train_args=ta, silent=True)
      # the run restored its best-frac>0.2 epoch (median tiebreaker); minimize
      def epoch_rank(i):
        return (fracs[i][0].item(), medians[i])
      best = min(range(len(fracs)), key=epoch_rank)
      trial.set_user_attr("median", float(medians[best]))
      return fracs[best][0].item()

    def log_trial(study_, trial):
      """
      Optuna callback: print the serial study's per-trial result line.

      Arguments:
        study_ = the study (read for the running best value).
        trial  = the trial that just finished (number, value, params).

      Returns:
        None (prints one line through the quiet-gated logger).
      """
      log(f"trial {trial.number:3d}  frac>0.2 {trial.value:.4f}"
          f"  best {study_.best_value:.4f}  {trial.params}")

    # minimize frac>0.2; TPE seed fixed for reproducibility
    study = optuna.create_study(
      direction="minimize",
      sampler=optuna.samplers.TPESampler(seed=0))
    # warm-start trial 0 from the YAML defaults (range first values),
    # beginning at the known-good config
    enqueue_default_control(
      study=study,
      default_trial=ranges)
    study.optimize(objective,
                   n_trials=args.n_trials,
                   timeout=args.timeout,
                   callbacks=[log_trial])
  else:
    # Parallel study: one spawned process per GPU, all cooperating on
    # a single study through the journal file (Optuna's multi-process
    # storage; see journal_storage). The parent creates the study,
    # warm-starts it once, splits the trial budget, and reads the
    # result back after the workers join.
    import torch.multiprocessing as mp

    # cocoa_output (cocoa.py): join the fileroot to the journal name.
    journal_path = cocoa_output(fileroot, args.journal)
    study = open_journal_study(
      study_name=study_name,
      journal_path=journal_path)
    done = len(study.trials)
    if done:
      log(f"resuming study in {journal_path}: {done} recorded "
          f"trial(s); adding {args.n_trials} more")
    enqueue_default_control(
      study=study,
      default_trial=ranges)

    # split the total trial budget across workers (first workers take
    # the remainder), and divide the host-RAM staging budget so the
    # per-worker private copies of the same subset cannot together
    # overflow RAM (stage_source falls back to the shared memmap when
    # its share is too small).
    worker_cfg = dict(cfg)
    worker_cfg["data"] = dict(cfg["data"])
    worker_cfg["data"]["ram_frac"] = (
      float(cfg["data"].get("ram_frac", 0.7)) / n_workers)
    shares = []
    for k in range(n_workers):
      shares.append(args.n_trials // n_workers
                    + (1 if k < args.n_trials % n_workers else 0))
    log(f"parallel study across {n_workers} GPUs "
        f"(trials {shares}, journal {journal_path})")

    # spawn (not fork): each child needs its own CUDA context. The
    # workers log their own "[gpu k] trial ..." lines.
    ctx = mp.get_context("spawn")
    procs = []
    for k in range(n_workers):
      # args positional order matches _tune_worker's signature:
      # (gpu_id, n_trials, cfg, rescale, activation, journal_path,
      #  study_name, timeout, quiet). Process forwards them
      # positionally, so the tuple order is load-bearing.
      p = ctx.Process(target=_tune_worker,
                      args=(k,
                            shares[k],
                            worker_cfg,
                            args.rescale,
                            args.activation,
                            journal_path,
                            study_name,
                            args.timeout,
                            args.quiet))
      p.start()
      procs.append(p)
    for p in procs:
      p.join()
    refuse_failed_workers(processes=procs)

    # re-read the study so the summary reflects every worker's trials.
    study = optuna.load_study(study_name=study_name,
                              storage=journal_storage(journal_path))
    finished = False
    for t in study.trials:
      if t.state == optuna.trial.TrialState.COMPLETE:
        finished = True
    if not finished:
      raise RuntimeError(
        "no trial completed; check the worker stderr above "
        f"(journal: {journal_path})")

  winner = best_complete_trial(study=study)
  log("\n--- search complete ---")
  log(f"study name: {study_name}")
  log(f"best frac>0.2: {winner.value:.4f}  "
      f"(median {winner.user_attrs.get('median', float('nan')):.4f})")
  log("best params:")
  for k, v in winner.params.items():
    log(f"  {k}: {v}")


if __name__ == "__main__":
  main()
