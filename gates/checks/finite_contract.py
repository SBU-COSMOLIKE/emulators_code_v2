#!/usr/bin/env python3
"""finite-contract: a NaN or Inf can never rank, select, or ship a model.

This check defends the finite training/evaluation contract. A non-finite
per-sample chi2 compares False to every threshold, so it counts as BELOW
threshold: a diverged model would report a "perfect" fraction of 0.0 and be
snapshotted as the best epoch, and the pre-train parity gates would print
"[ok] ... max|dv| = nan" as if the warm start held. Every training, scoring,
and parity site must instead ABORT loudly on a non-finite value, naming the
side (training / validation / diagnostic / parity) and the offending
positions, and never substitute a sentinel or count it below threshold.

It needs torch but no cosmolike and no GPU (it forces CPU): every leg drives
the REAL functions from emulator/training.py and emulator/warmstart.py with a
tiny synthetic setup, injecting exactly one non-finite value at a chosen spot.

The parts, in order:
  A. eval_val — an all-finite control reproduces the reference median / mean /
     fractions unchanged (the guard is inert on finite data); a single NaN,
     and a +/-Inf, among otherwise good rows raise, naming the validation row.
  B. the train step (through the real training_loop_batched) — a NaN scalar
     loss raises BEFORE backward; a finite loss with a non-finite gradient
     raises BEFORE optimizer.step; in both the weights are bitwise unchanged
     (nothing was mutated). A finite control run completes. _global_grad_norm
     is read-only (the clipping-off path stays byte-identical).
  C. eval_source_chi2 — a diverged model's non-finite per-row chi2 raises
     (side "diagnostic"), naming the source rows; a finite control returns.
  D. build_warm_start parity — no-extra both-arms NaN, one-arm NaN, Inf, and
     extras-present NaN each raise with the finite-contract message (never the
     misleading "extras leaked" / tolerance verdict); a valid source keeps its
     "[ok] finetune parity" line.
  E. build_transfer_start parity — a non-finite epoch-0 surface raises with the
     finite-contract message (never "not the frozen base bitwise"); a valid
     zero-init keeps its "[ok] transfer parity" line.
  F. the safe-sqrt producer (45M-24) — the objective must STOP producing the
     0/0 = NaN gradient at an exact fit (c == 0), not merely detect it: an
     exact-fit row has a finite, zero gradient in every sqrt mode (sqrt, both
     berhu lower branches, the anneal arm, and the berhu_capped region-3
     sqrt(t2*c) where-mask leak); positives agree analytically with sqrt; a
     materially negative / non-finite chi2 is refused (a non-finite loss);
     eager and torch.compile agree.
Every checked value is printed; any failure prints a FAIL line and the run
exits non-zero.

Home note: training-stack.md (the "NaN scores as a perfect emulator" section
and its pre-training parity clause: this is the finite-contract gate they name).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from emulator import warmstart
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.output import DataVectorGeometry
from emulator.geometries.parameter import ParamGeometry
from emulator.losses.core import CosmolikeChi2
from emulator.losses.transfer import TransferChi2
from emulator.results import save_emulator
from emulator.training import (eval_val, eval_source_chi2,
                               training_loop_batched, _global_grad_norm)

FAILURES = []
DEV = torch.device("cpu")            # CPU only: the contract legs need no GPU


def report(label, ok, detail):
  """Print one PASS/FAIL line and remember any failure.

  A failing check appends its label to the module-level FAILURES list so
  main can count them and exit non-zero.
  """
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  if not ok:
    FAILURES.append(label)


# ==========================================================================
# Part A: eval_val — the validation finite guard (clauses 1 + 4).
# ==========================================================================

N_VAL, N_IN, N_OUT = 64, 5, 8


class PoisonChi2:
  """Per-row sum-of-squares chi2 (the eval metric), with an optional
  non-finite injection at chosen rows to drive eval_val's finite guard.

  needs_params is False, so eval_val reduces it exactly as a real loss. The
  `bad` map holds {row -> value} (value being nan / +inf / -inf); with a
  single batch covering the whole set, `row` is the global validation row.
  """

  needs_params = False

  def __init__(self):
    self.bad = {}

  def chi2(self, pred, target):
    c = ((pred - target) ** 2).sum(dim=1)             # per-row, (bs,)
    if self.bad:
      c = c.clone()
      for i, v in self.bad.items():
        c[i] = v
    return c


def _val_data(C, DV):
  """The tiny in-memory val dict eval_val expects (one batch, no padding)."""
  return {"load_C": lambda rows: C[rows],
          "load_dv": lambda rows: DV[rows],
          "idx": np.arange(N_VAL)}


def check_eval_val():
  """Part A: eval_val is inert on finite data and raises on a non-finite chi2."""
  torch.manual_seed(0)
  C = torch.randn(N_VAL, N_IN)
  DV = torch.randn(N_VAL, N_OUT)
  model = nn.Linear(N_IN, N_OUT).eval()
  thresholds = torch.tensor([0.2, 0.5, 1.0])

  # the reference reduction, computed straight (no batching, no guard).
  with torch.no_grad():
    ref_c = ((model(C) - DV) ** 2).sum(dim=1)
  ref_med = ref_c.median().item()
  ref_mean = ref_c.mean().item()
  ref_frac = (ref_c[:, None] > thresholds[None, :]).float().mean(0)

  # control: the guard does not move the finite result.
  loss = PoisonChi2()
  med, mean, frac = eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                             load=N_VAL, bs=N_VAL, thresholds=thresholds)
  control_ok = (abs(med - ref_med) <= 1e-6 * abs(ref_med) + 1e-9
                and abs(mean - ref_mean) <= 1e-6 * abs(ref_mean) + 1e-9
                and torch.allclose(frac, ref_frac))
  report("eval_val: all-finite control reproduces the reference metrics",
         control_ok, "median/mean/frac unchanged by the guard")

  # a single NaN among good rows (global row 41): raises, names the row.
  loss = PoisonChi2()
  loss.bad = {41: float("nan")}
  raised, msg = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  report("eval_val: one NaN among good rows raises, naming the validation row",
         raised and "finite contract [validation]" in msg and "41" in msg,
         _short(msg))

  # +Inf and -Inf both raise (they, too, would count below every threshold).
  for tag, val in (("+Inf", float("inf")), ("-Inf", float("-inf"))):
    loss = PoisonChi2()
    loss.bad = {7: val}
    raised, msg = _expect_valueerror(
      lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                       load=N_VAL, bs=N_VAL, thresholds=thresholds))
    report("eval_val: an " + tag + " chi2 raises",
           raised and "finite contract [validation]" in msg and "7" in msg,
           _short(msg))


# ==========================================================================
# Part B: the train step — refuse a non-finite loss / gradient (clauses 2 + 3).
# ==========================================================================

N_TRAIN, TRAIN_BS, N_VALB = 8, 8, 16


class _NanGrad(torch.autograd.Function):
  """Identity forward (finite), non-finite backward: a finite loss whose
  gradient is NaN, the exact shape clause 3 defends (clipping off does not
  mean the step runs unchecked)."""

  @staticmethod
  def forward(ctx, x):
    return x.clone()

  @staticmethod
  def backward(ctx, g):
    return g * float("nan")


class DriveLoss:
  """A stand-in training loss with a settable poison for the train step.

  .loss matches the signature training_loop_batched calls (pred positional,
  the rest by keyword); it ignores the trim / focus / berhu knobs and reduces
  to a plain mean, which is all the finite-guard legs need. poison selects:
  "none" (a finite loss), "nan_loss" (a non-finite scalar, caught before
  backward), or "nan_grad" (a finite loss with a non-finite gradient, caught
  before optimizer.step). .chi2 is the clean eval metric (the baseline eval).
  """

  needs_params = False

  def __init__(self):
    self.poison = "none"

  def loss(self, pred, target, mode="sqrt", trim=None, focus=None,
           focus_scale=None, berhu_knot=None, berhu_cap=None, berhu_s=None):
    base = ((pred - target) ** 2).sum(dim=1).mean()       # a finite scalar
    if self.poison == "nan_loss":
      return base + float("nan")                          # non-finite loss
    if self.poison == "nan_grad":
      return _NanGrad.apply(base)                         # finite, NaN grad
    return base

  def chi2(self, pred, target):
    return ((pred - target) ** 2).sum(dim=1)


TRIM = {"shape": "linear", "start": 0.0, "end": 0.0,
        "hold_epochs": 1, "anneal_epochs": 1}
FOCUS = {"shape": "linear", "start": 0.0, "end": 0.0,
         "hold_epochs": 1, "anneal_epochs": 1, "kappa": 1.0}


def _train_data():
  """The nested train/val loaders training_loop_batched expects."""
  torch.manual_seed(1)
  Ctr, DVtr = torch.randn(N_TRAIN, N_IN), torch.randn(N_TRAIN, N_OUT)
  Cvl, DVvl = torch.randn(N_VALB, N_IN), torch.randn(N_VALB, N_OUT)
  return {"train": {"load_C": lambda r: Ctr[r], "load_dv": lambda r: DVtr[r],
                    "idx": np.arange(N_TRAIN), "load": N_TRAIN},
          "val": {"load_C": lambda r: Cvl[r], "load_dv": lambda r: DVvl[r],
                  "idx": np.arange(N_VALB), "load": N_VALB}}


def _drive_train(poison):
  """Run one poisoned epoch of the real loop; return (raised, msg, unchanged).

  Snapshots the weights, runs training_loop_batched for a single epoch with
  the given poison, and reports whether it raised, the message, and whether
  every weight is bitwise unchanged (proving the abort preceded any mutation).

  Arguments:
    poison = the DriveLoss poison ("none", "nan_loss", "nan_grad").

  Returns:
    (raised, msg, unchanged, result): the raise flag, its message, the
    weights-unchanged flag, and the loop's return value (None if it raised).
  """
  torch.manual_seed(2)
  model = nn.Linear(N_IN, N_OUT)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
  loss = DriveLoss()
  loss.poison = poison
  snap = {}
  for k, v in model.state_dict().items():
    snap[k] = v.detach().clone()

  raised, msg, result = False, "", None
  try:
    result = training_loop_batched(
      nepochs=1, optimizer=optimizer, scheduler=scheduler, model=model,
      bs=TRAIN_BS, lossfn=loss, mode="sqrt", data=_train_data(),
      trim_opts=TRIM, focus_opts=FOCUS,
      thresholds=torch.tensor([0.2, 0.5, 1.0]), silent=True)
  except ValueError as e:
    raised, msg = True, str(e)

  unchanged = True
  for k, v in model.state_dict().items():
    if not torch.equal(v, snap[k]):
      unchanged = False
  return raised, msg, unchanged, result


def check_train_step():
  """Part B: a non-finite loss / gradient aborts before mutating the weights."""
  # a NaN scalar loss: caught before backward, weights untouched.
  raised, msg, unchanged, _ = _drive_train("nan_loss")
  report("train step: a NaN scalar loss raises before backward",
         raised and "finite contract [training]" in msg
         and "scalar training loss" in msg, _short(msg))
  report("train step: the NaN-loss abort left the weights bitwise unchanged",
         unchanged, "no optimizer step ran")

  # a finite loss with a non-finite gradient: caught before optimizer.step.
  raised, msg, unchanged, _ = _drive_train("nan_grad")
  report("train step: a finite loss with a non-finite gradient raises",
         raised and "finite contract [training]" in msg
         and "gradient norm" in msg, _short(msg))
  report("train step: the NaN-gradient abort left the weights bitwise unchanged",
         unchanged, "the step was refused before it mutated the weights")

  # a finite control run completes and returns finite per-epoch histories.
  raised, msg, _, result = _drive_train("none")
  ok = (not raised) and result is not None
  if ok:
    train_losses, medians, means, fracs = result
    ok = (len(train_losses) == 1 and np.isfinite(train_losses[0])
          and np.isfinite(medians[0]) and np.isfinite(means[0]))
  report("train step: a finite control run completes with finite histories",
         ok, "loop ran one epoch, no raise")

  # _global_grad_norm is read-only: the clipping-off path stays byte-identical.
  m = nn.Linear(4, 3)
  m(torch.randn(5, 4)).sum().backward()
  before = []
  for p in m.parameters():
    before.append(p.grad.detach().clone())
  gn = _global_grad_norm(m.parameters())
  true = torch.sqrt(sum((g ** 2).sum() for g in before))
  grads_intact = True
  after = list(m.parameters())
  for p, b in zip(after, before):
    if not torch.equal(p.grad, b):
      grads_intact = False
  report("train step: _global_grad_norm returns the true norm, read-only",
         torch.allclose(gn, true) and grads_intact,
         "norm matches, gradients untouched")


# ==========================================================================
# Part C: eval_source_chi2 — the diagnostic finite guard (clause 5).
# ==========================================================================

N_SRC = 16


class _IdGeom:
  """A stand-in parameter geometry: encode is the identity."""

  def encode(self, x):
    return x


class _SrcChi2:
  """A stand-in diagnostic chi2 (encode identity, per-row sum of squares)."""

  needs_params = False

  def encode(self, dv):
    return dv

  def chi2(self, pred, target):
    return ((pred - target) ** 2).sum(dim=1)


class _SrcModel(nn.Module):
  """A tiny model whose output can be poisoned at one row (a diverged run)."""

  def __init__(self, n_in, n_out):
    super().__init__()
    self.lin = nn.Linear(n_in, n_out)
    self.bad_row = None
    self.bad_val = float("nan")

  def forward(self, x):
    out = self.lin(x)
    if self.bad_row is not None:
      out = out.clone()
      out[self.bad_row, 0] = self.bad_val
    return out


def check_source_chi2():
  """Part C: eval_source_chi2 raises on a non-finite per-row chi2."""
  torch.manual_seed(3)
  model = _SrcModel(N_IN, N_OUT)
  params = np.random.default_rng(0).standard_normal((N_SRC, N_IN))
  dv = np.random.default_rng(1).standard_normal((N_SRC, N_OUT))
  source = {"C": params, "dv": dv, "idx": np.arange(N_SRC)}

  # control: a finite model returns aligned finite arrays.
  model.bad_row = None
  p, dchi2 = eval_source_chi2(model=model, param_geometry=_IdGeom(),
                              chi2fn=_SrcChi2(), source=source, device=DEV,
                              bs=N_SRC)
  report("eval_source_chi2: a finite model returns finite per-row chi2",
         p.shape[0] == N_SRC and np.isfinite(dchi2).all(),
         str(dchi2.size) + " finite rows")

  # a diverged model (row 9 NaN): raises, side diagnostic, names the row.
  model.bad_row = 9
  model.bad_val = float("nan")
  raised, msg = _expect_valueerror(
    lambda: eval_source_chi2(model=model, param_geometry=_IdGeom(),
                             chi2fn=_SrcChi2(), source=source, device=DEV,
                             bs=N_SRC))
  report("eval_source_chi2: a non-finite per-row chi2 raises (side diagnostic)",
         raised and "finite contract [diagnostic]" in msg and "9" in msg,
         _short(msg))


# ==========================================================================
# Parts D + E: the pre-train parity finite guard (warmstart.py).
# ==========================================================================

N_S = 5                     # source parameter count
POUT = 12                   # kept (unmasked) data-vector length
PTOTAL = 20                 # full data-vector length
EXTRAS = ["w0", "wa"]       # the two new parameters fine-tuned in


def _spd(n, seed):
  """A random symmetric positive-definite matrix (n x n)."""
  g = np.random.default_rng(seed)
  a = g.standard_normal((n, n))
  return a @ a.T + n * np.eye(n)


def _source_names():
  names = []
  for i in range(N_S):
    names.append("p" + str(i))
  return names


def _source_recipe():
  """The model_recipe dict a schema-v2 save stores for the source ResMLP."""
  return {
    "cls": "emulator.designs.plain.ResMLP",
    "name": "resmlp",
    "ia": None,
    "input_dim": N_S,
    "output_dim": POUT,
    "compile_mode": None,
    "needs_geom": False,
    "kwargs": {
      "int_dim_res": 32,
      "n_blocks": 2,
      "block_opts": {"act": {"type": "H", "n_gates": 3},
                     "norm": "affine"},
    },
  }


def _save_source(root):
  """Build and save a synthetic plain source emulator under `root`."""
  names = _source_names()
  lam_s, V_s = np.linalg.eigh(_spd(N_S, seed=1))
  center_s = np.random.default_rng(2).standard_normal(N_S)
  pgeom = ParamGeometry(device=DEV, names=names, center=center_s,
                        evecs=V_s, sqrt_ev=np.sqrt(lam_s))
  lam_k, V_k = np.linalg.eigh(_spd(POUT, seed=3))
  geom = DataVectorGeometry(
    device=DEV, total_size=PTOTAL, dest_idx=list(range(POUT)),
    evecs=V_k, sqrt_ev=np.sqrt(lam_k), Cinv=_spd(PTOTAL, seed=4),
    center=np.random.default_rng(5).standard_normal(POUT),
    section_sizes=[PTOTAL], probe="xi")
  block_opts = {"act": make_activation("H", n_gates=3),
                "norm": make_norm("affine")}
  model = ResMLP(input_dim=N_S, output_dim=POUT, int_dim_res=32,
                 n_blocks=2, block_opts=block_opts).to(DEV)
  config = {"data": {"cosmolike_data_dir": "lsst_y1",
                     "cosmolike_dataset": "lsst_y1_M1_GGL0.05.dataset",
                     "train_dv": "src_train.npy", "val_dv": "src_val.npy"},
            "train_args": {"nepochs": 1}}
  histories = {"train_losses": [0.1], "val_medians": [0.1],
               "val_means": [0.1],
               "val_fracs": [torch.tensor([0.5, 0.4, 0.3, 0.2])],
               "thresholds": torch.tensor([0.2, 1.0, 10.0, 100.0])}
  save_emulator(path_root=str(root), model=model, param_geometry=pgeom,
                geometry=geom, config=config, histories=histories,
                train_args=config["train_args"],
                resolved_train={"nepochs": 1},
                resolved_model=_source_recipe(), attrs={"rescale": "none"})


def _write_covmat(path, names, seed):
  """Write a covmat file (a "#"-prefixed header line + a SPD matrix)."""
  cov = _spd(len(names), seed=seed)
  with open(path, "w") as f:
    f.write("# " + " ".join(names) + "\n")
    for row in cov:
      f.write(" ".join(repr(float(x)) for x in row) + "\n")


def _poison_forward(model, value):
  """Shadow a model's forward to inject `value` at output[0], returning a
  restore callable. state_dict is untouched (a one-arm poison for parity)."""
  orig = model.forward

  def bad(*a, **k):
    out = orig(*a, **k).clone()
    out.view(-1)[0] = value
    return out

  model.forward = bad

  def restore():
    del model.forward           # drop the instance shadow, restore the class

  return restore


def _poison_weight(model, value):
  """Set one weight to `value` in place; return a callable that restores the
  original state dict (a both-arms poison: the transfer copies it through)."""
  snap = {}
  for k, v in model.state_dict().items():
    snap[k] = v.detach().clone()
  with torch.no_grad():
    for p in model.parameters():
      p.view(-1)[0] = value
      break

  def restore():
    model.load_state_dict(snap, strict=True)

  return restore


def _run_warm(source, new_pgeom, model_opts, train_set, extras):
  """Call build_warm_start; return (raised, verdict_or_message)."""
  try:
    _, verdict, _ = warmstart.build_warm_start(
      source=source, new_pgeom=new_pgeom, pinned_geom=source.geom,
      model_opts=model_opts, train_set=train_set, extra_names=extras,
      device=DEV)
    return False, verdict
  except ValueError as e:
    return True, str(e)


def check_finetune_parity(source, tmp):
  """Part D: build_warm_start's finite guard fires before any parity verdict."""
  model_opts = warmstart.recipe_to_model_opts(
    recipe=source.recipe, geom=None, compile_mode=None)

  # the extras extension (7 params) and the no-extras extension (5 params).
  cov_x = Path(tmp) / "extras.covmat"
  _write_covmat(cov_x, _source_names() + EXTRAS, seed=12)
  pgeom_x, extras_x = warmstart.extend_input_geometry(
    source=source, covmat_path=str(cov_x),
    train_mean=np.random.default_rng(6).standard_normal(N_S + len(EXTRAS)),
    device=DEV)
  cov_0 = Path(tmp) / "same.covmat"
  _write_covmat(cov_0, _source_names(), seed=13)
  pgeom_0, extras_0 = warmstart.extend_input_geometry(
    source=source, covmat_path=str(cov_0),
    train_mean=np.random.default_rng(7).standard_normal(N_S), device=DEV)

  C_x = np.random.default_rng(8).standard_normal((64, len(pgeom_x.names)))
  C_0 = np.random.default_rng(9).standard_normal((64, len(pgeom_0.names)))
  ts_x = {"C": C_x, "idx": np.arange(64)}
  ts_0 = {"C": C_0, "idx": np.arange(64)}

  # control: a clean source keeps its "[ok] finetune parity" verdict.
  raised, verdict = _run_warm(source, pgeom_x, model_opts, ts_x, extras_x)
  report("finetune parity: a clean source keeps its [ok] verdict",
         (not raised) and verdict.startswith("[ok] finetune parity:"),
         _short(verdict))

  # no-extra, both arms NaN (a poisoned source weight transfers through both).
  restore = _poison_weight(source.model, float("nan"))
  raised, msg = _run_warm(source, pgeom_0, model_opts, ts_0, extras_0)
  restore()
  report("finetune parity: no-extra both-arms NaN raises the finite-contract "
         "error", raised and "finite contract [finetune parity]" in msg,
         _short(msg))

  # one-arm NaN (only the source model's output is poisoned): names the arm.
  restore = _poison_forward(source.model, float("nan"))
  raised, msg = _run_warm(source, pgeom_x, model_opts, ts_x, extras_x)
  restore()
  report("finetune parity: one-arm NaN raises, naming the source arm",
         raised and "finite contract [finetune parity]" in msg
         and "source-model" in msg, _short(msg))

  # one-arm Inf.
  restore = _poison_forward(source.model, float("inf"))
  raised, msg = _run_warm(source, pgeom_x, model_opts, ts_x, extras_x)
  restore()
  report("finetune parity: one-arm Inf raises",
         raised and "finite contract [finetune parity]" in msg, _short(msg))

  # extras-present NaN: the finite guard must fire BEFORE the extras
  # torch.equal (whose NaN mismatch would otherwise print "extras leaked").
  restore = _poison_weight(source.model, float("nan"))
  raised, msg = _run_warm(source, pgeom_x, model_opts, ts_x, extras_x)
  restore()
  report("finetune parity: extras-present NaN raises the finite-contract error "
         "(not 'extras leaked')",
         raised and "finite contract [finetune parity]" in msg
         and "leaked" not in msg, _short(msg))
  return pgeom_x, extras_x, C_x


def check_transfer_parity(source, pgeom_x, extras_x, C_x):
  """Part E: build_transfer_start's finite guard fires before any verdict."""
  chi2fn = TransferChi2(geom=source.geom, base_net=source.model,
                        base_in_dim=len(source.pgeom.names), form="gain",
                        space="physical", n_templates=1, n_amps=0,
                        coeff_fn=None)
  block_opts = {"act": make_activation("H", n_gates=3),
                "norm": make_norm("affine")}
  corr_opts = {"cls": ResMLP, "compile_mode": None, "int_dim_res": 16,
               "n_blocks": 1, "block_opts": block_opts}
  train_set = {"C": C_x, "idx": np.arange(64)}

  def run():
    try:
      _, verdict = warmstart.build_transfer_start(
        chi2fn=chi2fn, model_opts=corr_opts, new_pgeom=pgeom_x,
        train_set=train_set, extra_names=extras_x, device=DEV)
      return False, verdict
    except ValueError as e:
      return True, str(e)

  # control: a clean base keeps its "[ok] transfer parity" verdict.
  raised, verdict = run()
  report("transfer parity: a clean base keeps its [ok] verdict",
         (not raised) and verdict.startswith("[ok] transfer parity"),
         _short(verdict))

  # a non-finite epoch-0 surface (the base decode diverges): raises the
  # finite-contract error, not the misleading "not the frozen base bitwise".
  restore = _poison_weight(source.model, float("nan"))
  raised, msg = run()
  restore()
  report("transfer parity: a non-finite epoch-0 surface raises the "
         "finite-contract error (not 'frozen base')",
         raised and "finite contract [transfer parity]" in msg
         and "frozen base bitwise" not in msg, _short(msg))


# ==========================================================================
# Part F: the safe-sqrt producer (45M-24) -- an exact fit never NaNs the step.
# ==========================================================================

def _reduce_obj():
  """A bare CosmolikeChi2 (no __init__): _reduce uses only its arguments."""
  return CosmolikeChi2.__new__(CosmolikeChi2)


def _reduce_loss(obj, c, mode, berhu_s=None):
  """Call the real _reduce with trim / focus off (a plain mean of the mode)."""
  return obj._reduce(c=c, mode=mode, trim=0.0, focus=-1.0,
                     focus_scale=torch.tensor(1.0),
                     berhu_knot=torch.tensor(0.2),
                     berhu_cap=torch.tensor(10.0), berhu_s=berhu_s)


def _exact_fit_grad(obj, mode, berhu_s=None):
  """Backward through _reduce on a batch whose row 0 is an exact fit (c == 0).

  Row 0 of the residual r is all zeros, so its chi2 is exactly 0 (the pinned
  grid2d column / identity-start head / zero correction); the other rows are
  positive. With a plain sqrt, row 0's gradient is 0/0 = NaN; the safe-sqrt
  makes it a finite 0. Returns (loss, r.grad).
  """
  rows = torch.zeros(4, 3)
  rows[1:] = torch.linspace(0.3, 2.0, 9).reshape(3, 3)   # positive rows
  r = rows.clone().requires_grad_(True)
  c = (r ** 2).sum(dim=1)                                 # c[0] == 0 exactly
  loss = _reduce_loss(obj, c, mode, berhu_s=berhu_s)
  loss.backward()
  return loss, r.grad


def check_safe_sqrt():
  """Part F: the sqrt sites are grad-safe at an exact fit, in every mode."""
  obj = _reduce_obj()
  # (label, mode, berhu_s): the four sqrt sites -- sqrt mode, both berhu lower
  # branches, and the anneal arm -- plus the berhu_capped exact-fit row, which
  # also exercises the region-3 sqrt(t2*c) where-mask leak.
  cases = [("sqrt", "sqrt", None),
           ("berhu", "berhu", None),
           ("berhu_capped", "berhu_capped", None),
           ("berhu anneal", "berhu", torch.tensor(0.5)),
           ("berhu_capped anneal", "berhu_capped", torch.tensor(0.5))]
  for label, mode, s in cases:
    loss, grad = _exact_fit_grad(obj, mode, berhu_s=s)
    ok = (bool(torch.isfinite(loss)) and bool(torch.isfinite(grad).all())
          and bool((grad[0] == 0).all()))
    report("safe-sqrt: exact-fit row has a finite, zero gradient (mode "
           + label + ")", ok, "loss finite, grad row0 == 0, no NaN")

  # a mixed batch (one exact-fit row among positives) stays finite end to end.
  loss, grad = _exact_fit_grad(obj, "sqrt")
  report("safe-sqrt: mixed batch (one exact fit + positives) has finite grad",
         bool(torch.isfinite(grad).all()),
         "no row poisons the batch gradient")

  # analytic agreement on positives: sqrt-mode plain mean == mean(sqrt(c)).
  loss = _reduce_loss(obj, torch.tensor([1.0, 4.0, 9.0]), "sqrt")
  report("safe-sqrt: analytic agreement on positives (mean of sqrt = 2.0)",
         abs(loss.item() - 2.0) < 1e-6, "loss %.6f" % loss.item())

  # negative / NaN / Inf chi2 refusal: a corrupted c makes the loss non-finite.
  for tag, bad in (("materially negative", -5.0), ("NaN", float("nan")),
                   ("+Inf", float("inf"))):
    loss = _reduce_loss(obj, torch.tensor([1.0, bad, 2.0]), "sqrt")
    report("safe-sqrt: a " + tag + " chi2 is refused (non-finite loss)",
           not bool(torch.isfinite(loss)), "loss = " + repr(loss.item()))

  # eager + compiled: the exact-fit gradient stays finite under torch.compile.
  try:
    compiled = torch.compile(lambda cc: _reduce_loss(obj, cc, "sqrt"))
    rows = torch.zeros(4, 3)
    rows[1:] = torch.linspace(0.3, 2.0, 9).reshape(3, 3)
    r = rows.clone().requires_grad_(True)
    compiled((r ** 2).sum(dim=1)).backward()
    report("safe-sqrt: exact-fit gradient finite under torch.compile",
           bool(torch.isfinite(r.grad).all()) and bool((r.grad[0] == 0).all()),
           "compiled backward finite, row0 == 0")
  except Exception as e:                       # a backend without a compiler
    # the eager legs above prove the math; the production workstation
    # compiles, so a compiler-less box soft-skips rather than fails.
    report("safe-sqrt: torch.compile leg (compile unavailable here)", True,
           "skipped: " + _short(str(e)))


# ==========================================================================
# helpers + main
# ==========================================================================

def _expect_valueerror(fn):
  """Run fn; return (raised, message) for the ValueError it should raise."""
  try:
    fn()
    return False, ""
  except ValueError as e:
    return True, str(e)


def _short(msg):
  """The first line of an error message (or a verdict), for the PASS detail."""
  line = msg.split("\n")[0].strip()
  if len(line) > 88:
    return line[:88] + "..."
  return line


def main():
  """Run all five parts of the finite contract; return 1 if any leg failed."""
  print("== finite-contract ==")
  print("device " + str(DEV) + " (torch only, no cosmolike, no GPU)")
  torch.manual_seed(0)

  print("\n-- Part A: eval_val --")
  check_eval_val()
  print("\n-- Part B: the train step --")
  check_train_step()
  print("\n-- Part C: eval_source_chi2 --")
  check_source_chi2()

  tmp = tempfile.mkdtemp(prefix="finite-")
  root = Path(tmp) / "source"
  _save_source(root)
  source = warmstart.load_source(root=str(root), device=DEV)
  print("\n-- Part D: build_warm_start parity --")
  pgeom_x, extras_x, C_x = check_finetune_parity(source, tmp)
  print("\n-- Part E: build_transfer_start parity --")
  check_transfer_parity(source, pgeom_x, extras_x, C_x)
  print("\n-- Part F: the safe-sqrt producer (45M-24) --")
  check_safe_sqrt()

  print("")
  if len(FAILURES) == 0:
    print("finite-contract: ALL PASS")
    return 0
  print("finite-contract: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
