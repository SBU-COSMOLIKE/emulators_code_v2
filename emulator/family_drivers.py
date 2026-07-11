"""Shared serial machinery for the per-family sweep / tune drivers.

The program-wide driver namespace is `<family>_<verb>_emulator.py`
(scalar / cmb / baosn / mps x sweep_ntrain_ / sweep_hyperparam_ /
tune_). Every family trains through the same
EmulatorExperiment.from_config dispatch — the family is picked by the
config (data.outputs -> scalar, data.cmb -> CMB, ...) — so the
per-family drivers are thin: they own their CLI prog name, defaults,
and header docs, and call the run functions here. The loops are the
cosmic-shear drivers' SERIAL paths, single-sourced (the multi-GPU
pool / gpu-pack machinery stays the cosmic-shear drivers' tool: those
trainings are the expensive ones). The sweep-block helpers
(read_sweep_block, set_by_path) live here too and the cosmic-shear
hyperparameter driver imports them — one definition for every family.

PS: N_train sweep = train one fresh model per training-set size and
record the validation f(delta-chi2 > threshold) at each, the learning
curve saying whether the error floor is data-limited (still falling)
or capacity-limited (flat); a hyperparameter sweep = one training per
value of ONE YAML-chosen train_args leaf at fixed N_train; a tune
study = an Optuna search over the train_args leaves marked with
[default, min, max, kind] ranges in the YAML, minimizing the same
fraction.
"""

import copy
import time

import numpy as np

from .cocoa import resolve_cocoa_config, cocoa_output
from .experiment import EmulatorExperiment, validate_sweep_paths
from .results import save_learning_curves, save_sweep_table
from .training import suggest_train_args, search_defaults

# The train_args keys a hyperparameter sweep may enter through (first
# dotted segment). Guards against a typo'd path silently no-opping:
# exp.train reads train_args by .get, so an unknown top-level key
# would be ignored, and the sweep would train the same config N
# times. model.* keys are further validated by build_specs
# (MODEL_BLOCK_KEYS) at train time, loudly.
SWEEPABLE_TOP_KEYS = ("nepochs", "bs", "loss", "trunk_epochs",
                      "freeze_trunk", "clip", "rewind", "trunk", "head",
                      "model", "optimizer", "lr", "scheduler", "trim",
                      "focus", "ema")

# dotted paths that sweep the activation family: these are resolved
# by from_config into exp.activation (build_specs deliberately does
# not re-read the YAML block, so a train_args copy would be
# ignored); the run sets exp.activation per value instead.
ACTIVATION_PATHS = ("model.activation", "model.activation.type")


def set_by_path(train_args, path, value):
  """
  A deep copy of train_args with one dotted-path leaf replaced.

  Walks the nested mapping along `path` ("lr.lr_base" -> ["lr",
  "lr_base"]), creating intermediate mappings that do not exist yet
  (so `head.lr.lr_base` sweeps even when the YAML has no head: block;
  a head. / trunk_epochs / trunk. sweep on a single-phase model is
  rejected up front by validate_sweep_paths, since resolve_phase_args
  would demote it away), and sets the final key. The input is never
  mutated; each sweep point gets its own copy.

  Arguments:
    train_args = the resolved train_args mapping to copy.
    path       = dotted path of the leaf to set.
    value      = the value this sweep point tries.

  Returns:
    the modified deep copy.
  """
  out  = copy.deepcopy(train_args)
  node = out
  keys = path.split(".")
  for k in keys[:-1]:
    nxt = node.get(k)
    if not isinstance(nxt, dict):
      nxt = {}
      node[k] = nxt
    node = nxt
  node[keys[-1]] = value
  return out


def read_sweep_block(cfg):
  """
  Validate and unpack the YAML `sweep` block.

  Arguments:
    cfg = the resolved config mapping (data + train_args + sweep).

  Returns:
    (param, values, act_mode): the dotted path, the value list, and
    whether this is the activation-family special case.
  """
  if "sweep" not in cfg:
    raise KeyError(
      "the YAML needs a `sweep` block:\n"
      "  sweep:\n"
      "    parameter: lr.lr_base\n"
      "    values:\n"
      "      - 0.001\n"
      "      - 0.0025")
  blk    = cfg["sweep"]
  param  = str(blk.get("parameter", "")).strip()
  values = blk.get("values")
  if not param or not isinstance(values, list) or len(values) == 0:
    raise ValueError(
      "sweep block needs `parameter` (a dotted train_args path) "
      "and a non-empty `values` list")
  if param in ("model.name", "model.ia"):
    raise ValueError(
      f"cannot sweep {param}: it changes the model class: run "
      "one sweep per architecture (or the activation bake-off "
      "driver) and overlay the saved tables")
  act_mode = param in ACTIVATION_PATHS
  if not act_mode and param.split(".")[0] not in SWEEPABLE_TOP_KEYS:
    raise ValueError(
      f"sweep parameter {param!r} does not enter train_args "
      f"(first segment must be one of: "
      f"{' / '.join(SWEEPABLE_TOP_KEYS)})")
  return param, values, act_mode


def add_sweep_args(parser):
  """Attach the shared N_train-sweep flags to a driver's parser.

  Arguments:
    parser = the driver's argparse parser (the cocoa path args are the
             driver's own add_cocoa_path_args call).
  """
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "sweep: 'H', 'power', 'multigate', or "
                           "'gated_power'. Overrides the YAML "
                           "train_args.model.activation",
                      type=str,
                      choices=["H", "power", "multigate", "gated_power"],
                      default=None)
  parser.add_argument("--n-min",
                      dest="n_min",
                      help="smallest N_train in the grid (default 2000)",
                      type=int,
                      default=2000)
  parser.add_argument("--n-max",
                      dest="n_max",
                      help="largest N_train in the grid (default and "
                           "ceiling: the staged training pool)",
                      type=int,
                      default=None)
  parser.add_argument("--n-points",
                      dest="n_points",
                      help="grid points, geometric spacing (default 5)",
                      type=int,
                      default=5)
  parser.add_argument("--threshold",
                      dest="threshold",
                      help="delta-chi2 cutoff the fraction counts "
                           "(default 0.2)",
                      type=float,
                      default=0.2)
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress stdout (txt and pdf still written)",
                      action="store_true")


def add_tune_args(parser):
  """Attach the shared Optuna-study flags to a driver's parser."""
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "study: 'H', 'power', 'multigate', or "
                           "'gated_power'. Overrides the YAML "
                           "train_args.model.activation",
                      type=str,
                      choices=["H", "power", "multigate", "gated_power"],
                      default=None)
  parser.add_argument("--n-trials",
                      dest="n_trials",
                      help="number of Optuna trials (default 50)",
                      type=int,
                      default=50)
  parser.add_argument("--timeout",
                      dest="timeout",
                      help="stop the study after this many seconds "
                           "(optional; default no limit)",
                      type=int,
                      default=None)
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress stdout (per-trial lines and the "
                           "final summary)",
                      action="store_true")


def run_ntrain_sweep(args, family, out_default):
  """The serial N_train sweep, shared by the per-family drivers.

  Per grid point: stage the nested training subset, rebuild the
  geometry from its means, train a fresh model quietly, and score
  f(delta-chi2 > --threshold) on the fixed validation set. Writes
  <out>.txt (np.loadtxt-loadable curve + config) and <out>.pdf under
  the fileroot.

  Arguments:
    args        = the parsed CLI namespace (cocoa paths + the
                  add_sweep_args flags + an optional args.out).
    family      = the family label written into the curve metadata and
                  the figure legend ("scalar" / "cmb" / "baosn" /
                  "mps").
    out_default = the output name root when --out is absent.
  """
  cfg, fileroot, _ = resolve_cocoa_config(args)
  out_name = getattr(args, "out", None) or out_default

  # no rescale flag on the family drivers: an analytic-R rescaling is a
  # cosmic-shear data-vector concept, and each family's validator
  # rejects it anyway; from_config's default is "none".
  exp = EmulatorExperiment.from_config(cfg,
                                       activation=args.activation,
                                       quiet=args.quiet)
  log = exp.log
  exp.print_design()

  pool = exp.pool_size()
  n_max = pool if args.n_max is None else min(args.n_max, pool)
  if args.n_min >= n_max:
    raise ValueError(
      f"--n-min {args.n_min} must be below n_max {n_max} (pool {pool})")
  sizes = np.unique(
    np.geomspace(args.n_min, n_max, num=args.n_points).astype(int))
  log(f"pool {pool}  |  N_train grid: {sizes.tolist()}  |  serial")

  log("loading validation source:")
  exp.stage_val()
  fracs = []
  for N in sizes:
    t0 = time.time()
    exp.stage_train(n_train=int(N))
    exp.build_geometry()
    exp.train(silent=True)
    f = exp.frac_above(threshold=args.threshold)
    fracs.append(f)
    log(f"  N_train {int(N):8d}  f(>{args.threshold:g}) {f:.4f}  "
        f"({time.time() - t0:.0f}s)")

  model_name = exp.model_cls.__name__
  out_txt = cocoa_output(fileroot, out_name + ".txt")
  out_pdf = cocoa_output(fileroot, out_name + ".pdf")
  save_learning_curves(
    path=out_txt,
    sizes=sizes,
    curves={"frac": fracs},
    meta={"model": model_name,
          "family": family,
          "activation": args.activation,
          "threshold": args.threshold,
          "pool": pool})
  log(f"saved curve data -> {out_txt}")
  from .plotting import plot_learning_curves
  plot_learning_curves(
    curves={f"{model_name} ({family})": (sizes, fracs)},
    threshold=args.threshold,
    savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


def run_tune(args, family):
  """The serial Optuna study, shared by the per-family drivers.

  Stages the data and geometry once (they are fixed; only the searched
  train_args vary), then minimizes the best epoch's frac>0.2 over the
  YAML's [default, min, max, kind] ranges — trial 0 warm-starts from
  the range defaults, TPE seeded for reproducibility. Serial and
  in-memory (the multi-GPU journal study stays the cosmic-shear tune
  driver's tool).

  Arguments:
    args   = the parsed CLI namespace (cocoa paths + the add_tune_args
             flags).
    family = the family label for the log lines.
  """
  import optuna

  cfg, fileroot, _ = resolve_cocoa_config(args)
  exp = EmulatorExperiment.from_config(cfg,
                                       activation=args.activation,
                                       quiet=args.quiet)
  log = exp.log
  exp.print_design()

  raw_ta = exp.raw_train_args
  ranges = search_defaults(train_args=raw_ta)
  validate_sweep_paths(
    paths=list(ranges),
    two_phase=hasattr(exp.model_cls, "set_train_phase"))
  if not ranges:
    log("WARNING: no [default, min, max, kind] search ranges in "
        "train_args: every trial is identical.")

  optuna.logging.set_verbosity(optuna.logging.WARNING)
  log(f"serial {family} study (1 worker, in-memory)")
  log("loading sources:")
  exp.stage_train()
  exp.stage_val()
  exp.build_geometry()

  def objective(trial):
    """Score one trial: resolve train_args, train, return frac>0.2.

    Arguments:
      trial = the Optuna trial (each searched range becomes one
              suggestion).

    Returns:
      the best epoch's frac>0.2 (minimized); the median rides along as
      a trial user attribute for the report.
    """
    ta = suggest_train_args(trial=trial, train_args=raw_ta)
    (_m, _tl, medians,
     _mn, fracs) = exp.train(train_args=ta, silent=True)
    def epoch_rank(i):
      return (fracs[i][0].item(), medians[i])
    best = min(range(len(fracs)), key=epoch_rank)
    trial.set_user_attr("median", float(medians[best]))
    return fracs[best][0].item()

  def log_trial(study_, trial):
    """Optuna callback: one line per finished trial.

    Arguments:
      study_ = the study (read for the running best value).
      trial  = the trial that just finished (number, value, params).
    """
    log(f"trial {trial.number:3d}  frac>0.2 {trial.value:.4f}"
        f"  best {study_.best_value:.4f}  {trial.params}")

  study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=0))
  if ranges:
    study.enqueue_trial(ranges)
  study.optimize(objective,
                 n_trials=args.n_trials,
                 timeout=args.timeout,
                 callbacks=[log_trial])

  best = study.best_trial
  log(f"best trial {best.number}: frac>0.2 {best.value:.4f}  "
      f"median {best.user_attrs.get('median', float('nan')):.4f}")
  log(f"best params: {best.params}")


def add_hyperparam_args(parser):
  """Attach the shared one-knob-sweep flags to a driver's parser.

  Arguments:
    parser = the driver's argparse parser (the cocoa path args are the
             driver's own add_cocoa_path_args call).
  """
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "sweep: 'H', 'power', 'multigate', or "
                           "'gated_power'. Overrides the YAML "
                           "train_args.model.activation; forbidden "
                           "when the sweep parameter IS the "
                           "activation",
                      type=str,
                      choices=["H", "power", "multigate", "gated_power"],
                      default=None)
  parser.add_argument("--threshold",
                      dest="threshold",
                      help="delta-chi2 cutoff the fraction counts "
                           "(default 0.2)",
                      type=float,
                      default=0.2)
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress stdout (txt and pdf still written)",
                      action="store_true")


def run_hyperparam_sweep(args, family, out_default):
  """The serial one-knob sweep, shared by the per-family drivers.

  The YAML's `sweep` block names ONE dotted train_args leaf and a
  value list; every point trains a fresh model on the SAME staged
  data + geometry with only that leaf changed (the activation special
  case sets exp.activation instead — build_specs reads the family off
  the experiment, not the YAML). Scores f(delta-chi2 > --threshold)
  on the fixed validation set; writes <out>.txt (the value/frac
  table) and <out>.pdf under the fileroot. The cosmic-shear sibling
  (cosmic_shear_sweep_hyperparam_emulator.py) adds the multi-GPU
  pool; this loop is its serial path, single-sourced for the
  families.

  Arguments:
    args        = the parsed CLI namespace (cocoa paths + the
                  add_hyperparam_args flags + an optional args.out).
    family      = the family label written into the table metadata
                  ("scalar" / "cmb" / "baosn" / "mps").
    out_default = the output name root when --out is absent.
  """
  cfg, fileroot, _ = resolve_cocoa_config(args)
  out_name = getattr(args, "out", None) or out_default

  param, values, act_mode = read_sweep_block(cfg)
  if act_mode and args.activation is not None:
    raise ValueError(
      "the sweep parameter is the activation family; drop the "
      "--activation flag (it would pin what the sweep varies)")

  # no rescale flag on the family drivers: an analytic-R rescaling is a
  # cosmic-shear data-vector concept, and each family's validator
  # rejects it anyway; from_config's default is "none".
  exp = EmulatorExperiment.from_config(cfg,
                                       activation=args.activation,
                                       quiet=args.quiet)
  # validate_sweep_paths (experiment.py): fail before any training when
  # the axis would be silently demoted away on a single-phase model
  # (every family model is single-phase resmlp today, so head./trunk
  # axes are caught here). act_mode sweeps the experiment, not a path.
  if not act_mode:
    validate_sweep_paths(
      paths=[param],
      two_phase=hasattr(exp.model_cls, "set_train_phase"))
  log = exp.log
  exp.print_design()
  log(f"sweep: {param}  ->  {values}"
      + ("  (activation family)" if act_mode else "") + "  |  serial")

  log("loading sources:")
  exp.stage_train()
  exp.stage_val()
  exp.build_geometry()

  fracs = []
  for value in values:
    t0 = time.time()
    if act_mode:
      # the activation family lives on the experiment (see
      # ACTIVATION_PATHS above); train_args stay untouched.
      exp.activation = str(value)
      ta = None
    else:
      ta = set_by_path(exp.train_args, param, value)
    exp.train(train_args=ta, silent=True)
    f = float(exp.frac_above(threshold=args.threshold))
    fracs.append(f)
    log(f"  {param} = {value!r:>12}  "
        f"f(>{args.threshold:g}) {f:.4f}  ({time.time() - t0:.0f}s)")

  model_name = exp.model_cls.__name__
  out_txt = cocoa_output(fileroot, out_name + ".txt")
  out_pdf = cocoa_output(fileroot, out_name + ".pdf")
  # save_sweep_table (results.py): the value/frac table, np.loadtxt-
  # loadable with a "#" metadata header (overlay several <out>.txt
  # yourself to compare configs).
  save_sweep_table(
    path=out_txt,
    param=param,
    values=values,
    fracs=fracs,
    meta={"model": model_name,
          "family": family,
          "activation": ("swept" if act_mode else args.activation),
          "threshold": args.threshold,
          "n_train": cfg["data"]["n_train"]})
  log(f"saved sweep table -> {out_txt}")
  # plot_sweep_curve (plotting.py): render the value/frac figure.
  from .plotting import plot_sweep_curve
  plot_sweep_curve(param=param,
                   values=values,
                   fracs=fracs,
                   threshold=args.threshold,
                   savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")
