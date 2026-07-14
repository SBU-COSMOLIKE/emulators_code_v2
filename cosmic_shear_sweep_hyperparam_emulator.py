#!/usr/bin/env python3
"""Sweep one YAML-chosen hyperparameter at fixed N_train.

PS: a loader is a closure load(rows) -> a ready-to-train batch on the
device, hiding where the data lives; dump = the full on-disk array from the
data-generation run (the dv dump is the .npy, the param dump the .txt);
memmap = a NumPy array backed by that file, read in slices so it is never
loaded whole.
"""

#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
# This driver sweeps a single hyperparameter (chosen in the YAML, one value per
# training run at a fixed N_train) and records validation f(delta-chi2 >
# threshold) per value, with the same table + figure outputs as the N_train
# sweep. cosmic_shear_sweep_ntrain_emulator.py stays the driver for the N_train
# axis (its points have unequal cost and its staging differs per point); this
# one covers every other knob: batch size, learning rate, kernel size, film
# on/off, the activation family, ...
#
# python .../emultrfv2/cosmic_shear_sweep_hyperparam_emulator.py \
#   --root projects/lsst_y1/ \
#   --fileroot emulators/training_scripts/ \
#   --yaml cosmic_shear_train_emulator.yaml \
#   --out lrsweep_rescnn
#
#- The swept knob lives in a `sweep` block of the same training YAML (one YAML
#  serves every driver). `parameter` is the dotted path of exactly one
#  train_args leaf; `values` the list to try:
#
#      sweep:
#        parameter: lr.lr_base
#        values:
#          - 0.0010
#          - 0.0025
#          - 0.0063
#
#  Any train_args leaf works by its dotted path: top-level knobs (bs, clip,
#  trunk_epochs), sub-block knobs (scheduler.patience, trim.start,
#  model.cnn.kernel_size, model.cnn.film, model.mlp.width), and per-phase
#  overrides (head.lr.lr_base, whose intermediate blocks are created if the
#  YAML omits them; a phase axis on a single-phase model is rejected at
#  startup by validate_sweep_paths). Values may be numbers, strings,
#  or booleans (film: [false, true]).
#
#  Two special cases:
#  - model.activation (or model.activation.type) sweeps the activation family:
#    the driver sets the experiment's resolved activation per value (the
#    --activation flag must then be left unset); values are the family names
#    (H, power, multigate, gated_power). model.activation.n_gates sweeps as an
#    ordinary leaf.
#  - model.name / model.ia are refused: they change the model class, so
#    comparing architectures is one sweep per architecture (or the bake-off
#    driver), not one sweep across them.
#
#- Per value: a fresh model is trained on the same staged data and geometry
#  (staged once per worker, since the data does not vary across values, unlike
#  the N_train sweep) and scored as f(delta-chi2 > --threshold) on the fixed
#  val set.
#
#- Multiple GPUs (one node): values are near-equal-cost jobs, so they split
#  round-robin, one process per GPU (spawn; each worker stages its own copy
#  and streams from the shared dump memmap). One GPU (or the Apple-MPS dev
#  machine) falls back to a serial loop. `--gpu-pack` (off by default)
#  additionally co-locates up to 4 trainings per GPU when each is estimated
#  at <= 20% of the card (<= 40% -> 2), worth it on an H200, not on an
#  RTX 3060; see sweep_ntrain's header for the full rationale and caveats.
#
#- `--root` / `--fileroot` / `--yaml`: the cocoa layout, as in the training
#  driver (data under --root/chains, YAML + outputs under --fileroot).
#- `--rescale` / `--activation`: fixed across the sweep, as in the training
#  driver (--activation conflicts with sweeping model.activation).
#- `--n-gpus` (default: all visible CUDA devices): GPUs to spread across.
#- `--gpu-pack`: co-locate small trainings on one GPU (see above; off by
#  default).
#- `--threshold` (default 0.2): delta-chi2 cutoff the fraction counts.
#- `--out` (default hyperparam_sweep): writes <out>.txt (save_sweep_table:
#  numeric values as a value/frac table, categorical as index/frac with a
#  label map line) and <out>.pdf (plot_sweep_curve), under --fileroot.
#- `--quiet`: suppress stdout (txt and pdf still written).
#
#- Trains one full model per value, so run it on a machine with a working
#  Cocoa installation (cosmolike).
#-------------------------------------------------------------------------------

import argparse
import os
import time

import numpy as np
import torch

# This script sits beside the emulator/ package (same .../emultrfv2/ folder),
# so launching it by path makes its own directory sys.path[0] and
# `import emulator` resolves with no path manipulation. Run it from $ROOTDIR;
# emulator.cocoa reads $ROOTDIR to resolve the data paths.

from emulator.cocoa import (
  add_cocoa_path_args, resolve_cocoa_config, cocoa_output)
from emulator.experiment import (
  EmulatorExperiment, validate_sweep_paths, _pinned_head_warning)
from emulator.results import save_sweep_table
# single-sourced sweep helpers (family_drivers.py): the same
# constants and block parser the per-family sweep drivers use.
from emulator.family_drivers import (
  SWEEPABLE_TOP_KEYS, ACTIVATION_PATHS, set_by_path,
  read_sweep_block, resolved_sweep_record, sweep_record_value,
  sweep_design_label)
from emulator.scheduling import (
  even_assign, run_gpu_pool, GPU_TOKENS,
  estimate_train_vram_fraction, vram_tokens)


def _hyper_setup(gpu_id, extra):
  """
  Per-worker setup for run_gpu_pool: experiment + data, staged once.

  Unlike the N_train sweep, the data and geometry are fixed across
  the sweep points, so each worker stages train + val and builds the
  geometry a single time; jobs then only train.

  Arguments:
    gpu_id = CUDA device index this lane owns.
    extra  = the parent's payload dict: cfg, rescale, activation,
             param, act_mode, threshold.

  Returns:
    the fully staged EmulatorExperiment.
  """
  device = torch.device(f"cuda:{gpu_id}")
  exp = EmulatorExperiment.from_config(extra["cfg"],
                                       device=device,
                                       rescale=extra["rescale"],
                                       activation=extra["activation"],
                                       quiet=True)
  exp.stage_train()
  exp.stage_val()
  exp.build_geometry()
  return exp


def _hyper_job(gpu_id, exp, payload, extra):
  """
  One sweep point for run_gpu_pool: set the knob, train, score.

  Ordinary knobs go through set_by_path into a per-point train_args
  copy; the activation special case sets exp.activation instead (the
  train_args stay untouched, since build_specs reads the family off
  the experiment). Total by design: a failed point returns frac = nan.

  Arguments:
    gpu_id  = CUDA device index (bookkeeping; also used by the
              serial path with gpu_id 0).
    exp     = this lane's staged experiment from _hyper_setup.
    payload = (index, value): the point's position in the value list
              (results realign by it, so values need not be unique)
              and the value to try.
    extra   = the parent's payload dict (param, act_mode, threshold).

  Returns:
    (index, frac, gpu_id, seconds).
  """
  idx, value = payload
  t0 = time.time()
  try:
    if extra["act_mode"]:
      exp.activation = str(value)
      ta = None                     # train_args unchanged
    else:
      ta = set_by_path(exp.train_args, extra["param"], value)
    exp.train(train_args=ta, silent=True)
    f = float(exp.frac_above(threshold=extra["threshold"]))
  except Exception as err:          # keep the sweep alive
    f = float("nan")
    print(f"[gpu {gpu_id}] {extra['param']} = {value!r} "
          f"failed: {err}")
  # free this point's model before the next (possibly co-located)
  # point sizes its loaders; the staged data + geometry stay.
  exp.model = None
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  return (idx, f, gpu_id, time.time() - t0)


def main(prog="cosmic_shear_sweep_hyperparam_emulator", family="cosmolike",
         out_default="hyperparam_sweep"):
  parser = argparse.ArgumentParser(prog=prog)
  # --root / --fileroot / --yaml: the cocoa project layout (data under
  # --root, YAML + sweep outputs under --fileroot). Same schema as the
  # training driver, plus the sweep: block.
  add_cocoa_path_args(parser)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across "
                           "the sweep: 'none' (default), 'rescaled' "
                           "(v1), or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "sweep (leave unset when sweeping "
                           "model.activation): 'H', 'power', "
                           "'multigate', or 'gated_power'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default=None)
  parser.add_argument("--n-gpus",
                      dest="n_gpus",
                      help="number of GPUs to spread the sweep "
                           "across (default: all visible CUDA "
                           "devices). 1, or no CUDA, is serial.",
                      type=int,
                      default=None)
  parser.add_argument("--gpu-pack",
                      dest="gpu_pack",
                      help="co-locate small trainings on one GPU "
                           "(<=20%% of the card -> up to 4 share, "
                           "<=40%% -> up to 2, else exclusive). Off "
                           "by default; meant for large cards "
                           "(H200), not a 12 GB RTX 3060",
                      action="store_true")
  parser.add_argument("--threshold",
                      dest="threshold",
                      help="delta-chi2 cutoff the fraction counts "
                           "(default 0.2, the emulator goal)",
                      type=float,
                      default=0.2)
  parser.add_argument("--out",
                      dest="out",
                      help="output base path -> <out>.txt + "
                           "<out>.pdf (default: the driver's own "
                           "name, e.g. hyperparam_sweep)",
                      type=str,
                      default=None)
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (txt / pdf still "
                           "written)",
                      action="store_true")
  # strict parse: a misspelled flag (--sav, --activaton, --diagnostc) is a
  # usage error naming the token and exiting nonzero, never silently ignored
  # and then run at a default (which could publish to the wrong --save root).
  args = parser.parse_args()
  # --out absent -> the driver's own default (the family
  # wrappers pass their per-family name through out_default).
  if args.out is None:
    args.out = out_default

  # headless figure output, set before pyplot loads and before any
  # worker spawns (children inherit it).
  os.environ.setdefault("MPLBACKEND", "Agg")

  # resolve_cocoa_config (cocoa.py): load the YAML and make its data paths
  # absolute under $ROOTDIR/<root>; fileroot receives the sweep outputs.
  cfg, fileroot, _ = resolve_cocoa_config(args)
  # reject a wrong-family YAML at startup (family is always a real identity:
  # a per-family wrapper passes its key, a direct run owns "cosmolike").
  from cosmic_shear_train_emulator import require_family_block
  require_family_block(data=cfg["data"], family=family, prog=prog)
  param, values, act_mode = read_sweep_block(cfg)
  if act_mode and args.activation is not None:
    raise ValueError(
      "--activation conflicts with sweeping model.activation: "
      "drop the flag and let the sweep set the family per value")

  # the parent's experiment: banner + device + pool size; the serial
  # path reuses it.
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=args.activation,
                                       quiet=args.quiet)
  # validate_sweep_paths (experiment.py): fail before any dispatch (the
  # serial loop below and the gpu pool) if this sweep axis would be
  # silently demoted away on a single-phase model. act_mode sweeps the
  # activation family, not a train_args path, so skip the check there.
  if not act_mode:
    validate_sweep_paths(
      paths=[param],
      two_phase=hasattr(exp.model_cls, "set_train_phase"))
  if exp.device.type == "cpu":
    raise RuntimeError(
      "no GPU found (need CUDA, or Apple MPS on the dev machine): "
      "this sweep trains one model per value and is not meant for "
      "CPU")
  log = exp.log
  # print_design (experiment.py): the startup banner; the sweep then
  # overrides one leaf of what it shows, per point.
  exp.print_design()
  log(f"sweep: {param}  ->  {values}"
      + ("  (activation family)" if act_mode else ""))
  # ruling (a): sweeping the shared activation family leaves a per-head pin
  # fixed across every point; flag it once, quiet-gated through log.
  if act_mode:
    _pin_warn = _pinned_head_warning(
      exp.train_args, exp.model_cls.head_block,
      "the sweep varies the shared family, not this pin")
    if _pin_warn is not None:
      log(_pin_warn)

  n_cuda    = torch.cuda.device_count()
  n_request = n_cuda if args.n_gpus is None else min(args.n_gpus,
                                                     n_cuda)
  n_workers = min(n_request, len(values))

  # One immutable record supplies worker setup and every saved product. The
  # experiment has already resolved command-line-over-YAML precedence.
  activation_values = values if act_mode else None
  run_record = resolved_sweep_record(
    exp=exp,
    family=family,
    threshold=args.threshold,
    n_gpus=n_workers,
    n_train=cfg["data"]["n_train"],
    activation_values=activation_values)
  design_label = sweep_design_label(record=run_record)
  log("sweep design: " + design_label)

  # payloads carry the position so results realign afterward.
  payloads = []
  for i, v in enumerate(values):
    payloads.append((i, v))

  # --gpu-pack engages the pool even on a single CUDA card (its whole
  # point there: up to 4 small trainings co-located on one big GPU,
  # e.g. a lone H200 allocation).
  use_pool = (n_workers > 1
              or (args.gpu_pack and n_cuda >= 1 and len(values) > 1))

  if not use_pool:
    # serial on this one device (single GPU or the MPS dev machine),
    # reusing the parent experiment; same setup-once, train-per-value
    # shape as the workers.
    log("serial (1 worker)")
    log("loading sources:")
    exp.stage_train()
    exp.stage_val()
    exp.build_geometry()
    extra = {"param":     param,
             "act_mode":  act_mode,
             "threshold": args.threshold}
    fracs = [None] * len(values)
    for payload in payloads:
      idx, f, gpu, secs = _hyper_job(0, exp, payload, extra)
      fracs[idx] = f
      log(f"  {param} = {values[idx]!r:>12}  "
          f"f(>{args.threshold:g}) {f:.4f}  ({secs:.0f}s)")
  else:
    # equal-cost jobs -> round-robin; ram_frac 0 keeps every worker
    # streaming from the one shared dump memmap.
    n_pool = max(1, n_workers)
    log(f"parallel sweep across {n_pool} GPU(s)"
        + (" (gpu-pack)" if args.gpu_pack else "") + ":")
    worker_cfg = dict(cfg)
    worker_cfg["data"] = dict(cfg["data"])
    worker_cfg["data"]["ram_frac"] = 0.0
    # even_assign (scheduling.py): split the payloads round-robin into
    # one bucket per GPU (near-equal counts, since the jobs cost alike).
    buckets = even_assign(jobs=payloads, n_workers=n_pool)
    for k, b in enumerate(buckets):
      vals = []
      for i, v in b:
        vals.append(v)
      log(f"  gpu {k}: {len(b)} points  ->  {vals}")

    # --gpu-pack: every point stages the same N_train, so one token
    # count covers all jobs. n_train is the exact staged row count
    # (post-cut, enforced by load_source), so the estimate is exact;
    # the memmap (when one exists) stays open only for its width.
    lanes = 1
    job_tokens = None
    if args.gpu_pack:
      if "train_dv" in cfg["data"]:
        dv_width = np.load(cfg["data"]["train_dv"],
                           mmap_mode="r").shape[1]
      else:
        # a scalar run has no dv dump; its targets are the named
        # output columns, len(outputs) wide (tiny).
        dv_width = len(cfg["data"]["outputs"])
      n_est = int(cfg["data"]["n_train"])
      # get_device_properties(0): the positional 0 is the CUDA device
      # index (GPU 0; a homogeneous-GPU assumption for the estimate).
      total = torch.cuda.get_device_properties(0).total_memory
      # estimate_train_vram_fraction / vram_tokens (scheduling.py): turn
      # the point's row/width estimate into a fraction of a card, then
      # into capacity tokens the pool packs against.
      frac  = estimate_train_vram_fraction(n_rows=n_est,
                                           dv_width=dv_width,
                                           total_bytes=total)
      tokens = vram_tokens(fraction=frac)
      def job_tokens(payload):
        """
        run_gpu_pool token callback: this point's VRAM token count.

        Arguments:
          payload = the (index, value) sweep point (unused; every
                    point stages the same N_train, so the token count
                    is constant).

        Returns:
          the shared per-point token count (out of GPU_TOKENS).
        """
        return tokens
      lanes = GPU_TOKENS
      log(f"  gpu-pack on: est. {frac:.2f} of a GPU per point "
          f"-> {tokens} token(s)/4")

    worker_activation = sweep_record_value(
      record=run_record,
      key="activation")
    if worker_activation == "swept":
      ordered_activations = sweep_record_value(
        record=run_record,
        key="activation_values")
      worker_activation = ordered_activations[0]
    extra = {"cfg":        worker_cfg,
             "rescale":    sweep_record_value(
               record=run_record,
               key="rescale"),
             "activation": worker_activation,
             "param":      param,
             "act_mode":   act_mode,
             "threshold":  args.threshold}

    def on_result(r):
      """
      run_gpu_pool result callback: log one sweep point as it lands.

      Arguments:
        r = one result tuple (index, frac, gpu_id, seconds) from a
            finished _hyper_job.

      Returns:
        None (prints one line through the quiet-gated logger).
      """
      idx, f, gpu, secs = r
      log(f"  {param} = {values[idx]!r:>12}  "
          f"f(>{args.threshold:g}) {f:.4f}  (gpu {gpu}, {secs:.0f}s)")

    results = run_gpu_pool(setup_fn=_hyper_setup,
                           job_fn=_hyper_job,
                           buckets=buckets,
                           extra=extra,
                           lanes_per_gpu=lanes,
                           job_tokens=job_tokens,
                           on_result=on_result)
    fracs = [None] * len(values)
    for idx, f, gpu, secs in results:
      fracs[idx] = f

  # table + figure, like the N_train sweep (overlay several <out>.txt
  # yourself to compare configs). cocoa_output (cocoa.py) joins the
  # fileroot to each output name; save_sweep_table (results.py) writes
  # the value/frac table.
  out_txt = cocoa_output(fileroot, args.out + ".txt")
  out_pdf = cocoa_output(fileroot, args.out + ".pdf")
  save_sweep_table(
    path=out_txt,
    param=param,
    values=values,
    fracs=fracs,
    meta=dict(run_record))
  log(f"saved sweep table -> {out_txt}")

  # plot_sweep_curve (plotting.py): render the value/frac sweep figure.
  from emulator.plotting import plot_sweep_curve
  plot_sweep_curve(param=param,
                   values=values,
                   fracs=fracs,
                   threshold=args.threshold,
                   design_label=design_label,
                   savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


if __name__ == "__main__":
  main()
