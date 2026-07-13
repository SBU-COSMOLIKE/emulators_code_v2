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
  F. the safe-sqrt producer — the objective must STOP producing the
     0/0 = NaN gradient at an exact fit (c == 0), not merely detect it: an
     exact-fit row has a finite, zero gradient in every sqrt mode (sqrt, both
     berhu lower branches, the anneal arm, and the berhu_capped region-3
     sqrt(t2*c) where-mask leak); positives agree analytically with sqrt; a
     materially negative / non-finite chi2 is refused (a non-finite loss);
     eager and torch.compile agree.
  G. epoch-reduction finite truth — a finite per-batch loss near the
     float32 max yields a finite epoch mean (accumulated on the host in
     float64), where the old device-float32 loss*bs product would overflow to
     Inf and publish it.
  H. the chi2-domain boundary — eval_val and eval_source_chi2
     RAISE on a finite negative chi2 (which training folds), so a corrupted row
     can never rank as "perfect": the finite-only check would crown it; an
     exact zero is accepted; the scale-aware band tolerates roundoff and
     refuses corruption on both edges; the same fold is compile-safe. The band
     scales with the per-row reduction DEPTH = kept width w, not w^2:
     a production-width (>= 780) leg through a REAL CosmolikeChi2 subclass
     refuses -2 / -4 and both sides of the actual float32 band; restoring the
     retired w^2 rule (a mutation arm) lets -2 through; ScalarChi2 declares
     n_out; a mechanical subclass census proves no family overrides the width
     rule; and an ill-conditioned SPD control shows genuine roundoff near zero
     falls inside the band. The band derives from the dtype the chi2 was
     COMPUTED in, never a storage upcast (the): _reduce,
     eval_val, and eval_source_chi2 give ONE verdict on a value between the
     1e-6 floor and the float32 band, a restored .double() upcast would split
     them, and a genuinely float64-computed loss still gets the tight float64
     band. The Part F compile arm is now capability-gated (mandatory-red on a
     compile-capable box, an explicit SKIP-DEP otherwise, plus a
     raising-callable control).
Every checked value is printed; any failure prints a FAIL line and the run
exits non-zero.

Home note: training-stack.md (the "NaN scores as a perfect emulator" section
and its pre-training parity clause: this is the finite-contract gate they name).
"""

import sys
import tempfile
import types
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
from emulator.losses.core import CosmolikeChi2, _chi2_neg_band
from emulator.losses.scalar import ScalarChi2
from emulator.losses.transfer import TransferChi2
from emulator.results import save_emulator
from emulator.training import (eval_val, eval_source_chi2,
                               training_loop_batched, _global_grad_norm,
                               ordinary_median, _validate_optimizer_opts)

FAILURES = []
# set True when a required check lane could not run for a missing capability
# (the torch.compile backward lane). It is NOT a pass and NOT a failure of a
# tested assertion; main turns it into a distinct exit code so the board can
# report a non-green result instead of certifying a gate whose mandatory lane
# never executed.
LANE_UNAVAILABLE = []
DEV = torch.device("cpu")            # CPU only: the contract legs need no GPU

# exit codes main returns, read by the board's gate wrapper:
#   0 = every leg ran and passed;
#   1 = a tested assertion failed;
#   2 = every leg that ran passed, but a mandatory lane could not run for a
#       missing capability (a non-green result, never a silent PASS).
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_LANE_UNAVAILABLE = 2


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
  # the reference reduction matches eval_val's published reductions (unit 60
  # ordinary median + unit 14(f) float64 mean), so the finite control agrees.
  ref_med = ordinary_median(ref_c)
  ref_mean = ref_c.to(torch.float64).mean().item()
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
# Part F: the safe-sqrt producer -- an exact fit never NaNs the step.
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


def _can_compile():
  """True iff this box can build AND run a torch.compile'd backward.

  The capability probe: compile a trivial function and run its
  backward. A compiler-less dev box fails here, so the mandatory compile legs
  are gated on a real capability -- never on a broad except that would green a
  genuine Inductor / backward regression.
  """
  try:
    f = torch.compile(lambda t: (t * t).sum())
    x = torch.ones(3, requires_grad=True)
    f(x).backward()
    return bool(torch.isfinite(x.grad).all())
  except Exception:
    return False


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
  # capability detection FIRST. A broad except that greened
  # on ANY exception turned a real Inductor / backward regression into a green
  # skip -- the exact failure class the leg exists to catch. On a
  # compile-capable box the leg is MANDATORY (an exception is RED, with the
  # traceback); a genuinely compiler-less box emits an explicit non-green
  # SKIP-DEP that can never count toward closure.
  if _can_compile():
    import traceback
    try:
      compiled = torch.compile(lambda cc: _reduce_loss(obj, cc, "sqrt"))
      rows = torch.zeros(4, 3)
      rows[1:] = torch.linspace(0.3, 2.0, 9).reshape(3, 3)
      r = rows.clone().requires_grad_(True)
      compiled((r ** 2).sum(dim=1)).backward()
      report("safe-sqrt: exact-fit gradient finite under torch.compile "
             "(MANDATORY on the compile lane)",
             bool(torch.isfinite(r.grad).all())
             and bool((r.grad[0] == 0).all()),
             "compiled backward finite, row0 == 0")
    except Exception:
      report("safe-sqrt: exact-fit gradient finite under torch.compile "
             "(MANDATORY on the compile lane)", False,
             "compiled leg raised: " + traceback.format_exc().splitlines()[-1])
    # mutation control: a compiled callable that RAISES must be detected, so
    # the arm can never green on a real compile / backward failure.
    raised = False
    try:
      broken = torch.compile(lambda x: x + _does_not_exist_ce)   # NameError
      broken(torch.ones(2))
    except Exception:
      raised = True
    report("safe-sqrt: a raising compiled callable is detected (the arm "
           "cannot green on a compile failure)", raised,
           "compile failure surfaces as an exception, not a green skip")
  else:
    # the compile lane is mandatory; when the box cannot run a compiled
    # backward, record the lane as unavailable so main returns a non-green
    # exit code. A printed skip line alone would let the board read the
    # process's success and certify a gate whose mandatory lane never ran.
    LANE_UNAVAILABLE.append("safe-sqrt torch.compile backward")
    print("  [LANE UNAVAILABLE] safe-sqrt: torch.compile could not run a "
          "backward on this box; the compile lane is mandatory, so this gate "
          "reports a non-green result (run on a compile-capable workstation).")


# ==========================================================================
# Part G: epoch-reduction finite truth -- a finite per-batch loss
#         must not publish an Inf epoch mean.
# ==========================================================================

class _BigLoss:
  """A finite training loss pinned near the float32 max (finite gradient).

  Its per-batch scalar is 1e38 (finite in float32) with a real gradient, so
  the per-step finite guard passes -- but the OLD device-float32 epoch
  product loss * bs would overflow to Inf before the accumulator. .chi2 is
  the clean eval metric.
  """

  needs_params = False

  def loss(self, pred, target, mode="sqrt", trim=None, focus=None,
           focus_scale=None, berhu_knot=None, berhu_cap=None, berhu_s=None):
    base = ((pred - target) ** 2).sum(dim=1).mean()
    return base + (1.0e38 - base.detach())        # value 1e38, gradient d base

  def chi2(self, pred, target):
    return ((pred - target) ** 2).sum(dim=1)


def check_epoch_reduction():
  """Part G: the epoch mean accumulates on the host, finite despite a
  near-overflow per-batch loss; the old float32 product would overflow."""
  torch.manual_seed(5)
  model = nn.Linear(N_IN, N_OUT)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
  n_train, bs = 16, 8                             # two full batches per epoch
  Ctr, DVtr = torch.randn(n_train, N_IN), torch.randn(n_train, N_OUT)
  Cvl, DVvl = torch.randn(N_VALB, N_IN), torch.randn(N_VALB, N_OUT)
  data = {"train": {"load_C": lambda r: Ctr[r], "load_dv": lambda r: DVtr[r],
                    "idx": np.arange(n_train), "load": n_train},
          "val": {"load_C": lambda r: Cvl[r], "load_dv": lambda r: DVvl[r],
                  "idx": np.arange(N_VALB), "load": N_VALB}}
  result = training_loop_batched(
    nepochs=1, optimizer=optimizer, scheduler=scheduler, model=model,
    bs=bs, lossfn=_BigLoss(), mode="sqrt", data=data,
    trim_opts=TRIM, focus_opts=FOCUS,
    thresholds=torch.tensor([0.2, 0.5, 1.0]), silent=True)
  epoch_loss = result[0][0]
  report("epoch reduction: a finite 1e38 per-batch loss yields a finite "
         "epoch mean (host float64)",
         np.isfinite(epoch_loss) and 5e37 < epoch_loss < 5e38,
         "epoch loss = %.3e" % epoch_loss)

  # mutation arm: the pre-fix device float32 product loss * bs overflowed.
  old = np.float32(1.0e38) * np.float32(bs)
  report("epoch reduction: the old float32 loss*bs product overflows to Inf",
         not np.isfinite(old),
         "np.float32(1e38) * " + str(bs) + " = " + repr(float(old)))


# ==========================================================================
# Part H: the chi2-domain boundary -- eval / diagnostic reject a
#         finite NEGATIVE chi2 that training folds; no false "perfect" row.
# ==========================================================================

class _NegSrcChi2:
  """A diagnostic chi2 (encode identity) with a settable negative injection."""

  needs_params = False

  def __init__(self):
    self.neg_row = None

  def encode(self, dv):
    return dv

  def chi2(self, pred, target):
    c = ((pred - target) ** 2).sum(dim=1)
    if self.neg_row is not None:
      c = c.clone()
      c[self.neg_row] = -4.0
    return c


def check_chi2_domain():
  """Part H: eval_val and eval_source_chi2 raise on a finite NEGATIVE chi2."""
  torch.manual_seed(7)
  C = torch.randn(N_VAL, N_IN)
  DV = torch.randn(N_VAL, N_OUT)
  model = nn.Linear(N_IN, N_OUT).eval()
  thresholds = torch.tensor([0.2, 0.5, 1.0])

  # eval_val: a real one-output r^T[-1]r = -4 (row 17) must RAISE before the
  # fractions -- a negative counts False against every positive threshold and
  # would crown the corrupted row as perfect.
  loss = PoisonChi2()
  loss.bad = {17: -4.0}
  raised, msg = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  report("chi2 domain: eval_val raises on a finite negative chi2",
         raised and "chi2 domain contract [validation]" in msg and "17" in msg,
         _short(msg))

  # an exact-zero chi2 is accepted (within-band -> 0), not raised.
  loss = PoisonChi2()
  loss.bad = {5: 0.0}
  raised, _ = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  report("chi2 domain: an exact-zero chi2 is accepted (not raised)",
         not raised, "0.0 kept as a valid perfect row")

  # positive control: the domain check is inert on all-positive scores.
  with torch.no_grad():
    ref_c = ((model(C) - DV) ** 2).sum(dim=1)
  ref_med = ordinary_median(ref_c)
  med, _, _ = eval_val(model=model, lossfn=PoisonChi2(),
                       data=_val_data(C, DV), load=N_VAL, bs=N_VAL,
                       thresholds=thresholds)
  report("chi2 domain: all-positive control byte-identical (guard inert)",
         abs(med - ref_med) <= 1e-6 * abs(ref_med) + 1e-9, "median unchanged")

  # eval_source_chi2: a negative per-row chi2 raises (side diagnostic).
  params = np.random.default_rng(0).standard_normal((N_SRC, N_IN))
  dv = np.random.default_rng(1).standard_normal((N_SRC, N_OUT))
  source = {"C": params, "dv": dv, "idx": np.arange(N_SRC)}
  neg = _NegSrcChi2()
  neg.neg_row = 9
  raised, msg = _expect_valueerror(
    lambda: eval_source_chi2(model=nn.Linear(N_IN, N_OUT),
                             param_geometry=_IdGeom(), chi2fn=neg,
                             source=source, device=DEV, bs=N_SRC))
  report("chi2 domain: eval_source_chi2 raises on a negative per-row chi2",
         raised and "chi2 domain contract [diagnostic]" in msg and "9" in msg,
         _short(msg))

  # finite-only mutation (the false crowning): the retired finite-only check
  # ranks the -4 row BELOW every threshold, lowering frac>0.2 so the corrupted
  # epoch would win selection. Shown as a known answer.
  poisoned = ref_c.clone()
  poisoned[17] = -4.0
  f_honest = (ref_c[:, None] > thresholds[None, :]).float().mean(0)[0]
  f_poison = (poisoned[:, None] > thresholds[None, :]).float().mean(0)[0]
  report("chi2 domain: the finite-only check would crown the negative row "
         "(mutation)", float(f_poison) < float(f_honest),
         "frac>0.2 %.4f -> %.4f (a lower, 'better' score)"
         % (float(f_honest), float(f_poison)))

  # band-edge: a within-band roundoff negative is accepted, a just-outside one
  # raises (PoisonChi2 has no _chi2_n_terms, so eval_val uses the ~1e-6 floor).
  loss = PoisonChi2()
  loss.bad = {3: -5.0e-7}
  raised_in, _ = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  loss = PoisonChi2()
  loss.bad = {3: -1.0e-2}
  raised_out, _ = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  report("chi2 domain: band-edge -- within-band accepted, outside raises",
         (not raised_in) and raised_out,
         "roundoff tolerated, corruption refused")

  # a negative through the COMPILED training reduce folds to NaN (the same
  # predicate, compile-safe), so the per-step guard refuses it.
  if _can_compile():
    obj = _reduce_obj()
    compiled = torch.compile(lambda cc: _reduce_loss(obj, cc, "sqrt"))
    lo = compiled(torch.tensor([1.0, -4.0, 2.0]))
    report("chi2 domain: a negative chi2 through the COMPILED reduce -> "
           "non-finite loss", not bool(torch.isfinite(lo)),
           "compiled fold to NaN")


# ==========================================================================
# Part H (production band): the band scales with the kept WIDTH,
#         not w^2 -- a realistic dense width refuses a chi2 = -2 the retired
#         w^2 rule crowned as perfect (band 34.3 at w = 3000).
# ==========================================================================

class _WidthChi2(CosmolikeChi2):
  """A REAL CosmolikeChi2 (inheriting the production _chi2_n_terms) with a
  declared dense width and a settable per-row value, for the production-band
  legs.

  Only chi2 is stubbed, to inject a chosen per-row value; _chi2_n_terms, the
  band, and _reduce are the SHIPPED code -- so the leg exercises the width the
  fix installs (dest_idx.numel()), not a hand-set n_terms. A SimpleNamespace
  geom supplies the dest_idx (the realistic dense width) the base reads.
  needs_params is False, so eval_val reduces it exactly as a real loss.
  """

  needs_params = False

  def __init__(self, width):
    self.geom = types.SimpleNamespace(dest_idx=torch.arange(int(width)))
    self.bad = {}

  def chi2(self, pred, target, full=False):
    c = ((pred - target) ** 2).sum(dim=1)              # per-row, (bs,)
    if self.bad:
      c = c.clone()
      for row, value in self.bad.items():
        c[row] = float(value)
    return c


class _WidthChi2Wsq(_WidthChi2):
  """MUTATION control: restore the retired w^2 rule. At width 780 the
  float32 band balloons to 2.32 and SWALLOWS a chi2 = -2, so eval_val no longer
  raises -- proving the width rule is load-bearing (a w^2 regression reopens the
  false-crowning hole). Named so the subclass census can exclude it."""

  def _chi2_n_terms(self):
    w = int(self.dest_idx.numel())
    return w * w


class _WidthSrcChi2(CosmolikeChi2):
  """A diagnostic (encode-identity) CosmolikeChi2 with a declared dense width, a
  settable per-row value, and a chosen COMPUTE dtype -- for the (g) second-
  addendum band-dtype-provenance legs.

  chi2 emits its per-row values in _dtype, so eval_source_chi2 derives the band
  from the dtype the chi2 was COMPUTED in; _chi2_n_terms (inherited) is the
  width. needs_params is False, so eval_source reduces it as a real diagnostic.
  """

  needs_params = False

  def __init__(self, width, dtype=torch.float32):
    self.geom = types.SimpleNamespace(dest_idx=torch.arange(int(width)))
    self._dtype = dtype
    self.bad = {}

  def encode(self, dv):
    return dv

  def chi2(self, pred, target, full=False):
    c = ((pred - target) ** 2).sum(dim=1).to(self._dtype)     # per-row, _dtype
    if self.bad:
      c = c.clone()
      for row, value in self.bad.items():
        c[row] = float(value)
    return c


def _all_subclasses(cls):
  """Every subclass of cls, transitively (the loss-family census)."""
  found = []
  for sub in cls.__subclasses__():
    found.append(sub)
    for deeper in _all_subclasses(sub):
      found.append(deeper)
  return found


def _spd_roundoff_min(width, seed=0):
  """The most-negative chi2 a genuine ill-conditioned SPD contraction yields.

  Builds a width x width SPD precision M = Q^T diag(s^2) Q with s spanning
  [1e-4, 1] (M's condition ~1e8), and a batch of residuals aimed at M's
  smallest-eigenvalue direction, so the true quadratic form r^T M r sits just
  above 0. Evaluated in float32 (eval_val's compute dtype), catastrophic
  cancellation can push a near-zero row slightly negative -- a GENUINE roundoff
  negative, the kind the width band must tolerate. Returns the batch minimum (a
  float); the caller asserts it is >= -band (roundoff lands inside the band)
  while a corrupt -2 does not.

  Arguments:
    width = the contraction width (matches the production band under test).
    seed  = RNG seed for the orthogonal basis (reproducible).

  Returns:
    the minimum float32 chi2 over the batch, a Python float.
  """
  gen = torch.Generator().manual_seed(seed)
  w = int(width)
  # Q orthogonal (rows orthonormal); A = diag(s) @ Q so M = A^T A =
  # Q^T diag(s^2) Q is a full dense SPD with eigenvalues s^2 and eigenvectors
  # the rows of Q. s ascending -> s[0]^2 is the smallest eigenvalue, its
  # eigenvector row Q[0] the near-null direction.
  q, _ = torch.linalg.qr(torch.randn(w, w, generator=gen))
  s = torch.logspace(-4.0, 0.0, w)
  a = s.unsqueeze(1) * q                               # row i scaled by s[i]
  m = (a.t() @ a).float()                              # dense SPD, float32
  v_min = q[0, :]                                       # smallest-eigenvalue dir
  scales = torch.linspace(0.05, 0.5, 32).unsqueeze(1)  # tiny norms
  r = (scales * v_min.unsqueeze(0)).float()            # (32, w) near-null resid
  chi2 = torch.einsum("bi,ij,bj->b", r, m, r)          # r^T M r, float32
  return float(chi2.min())


def check_chi2_band_production():
  """Part H: the chi2-domain band at realistic dense WIDTHS.

  The shipped negative / band-edge legs above use PoisonChi2 (no
  _chi2_n_terms), so eval_val falls back to n_terms = 1 and only the 1e-6 floor
  is exercised. These legs use a REAL CosmolikeChi2 subclass whose declared
  width is a realistic dense 780, so the band is the scale-aware production
  value the width rule installs -- and the retired w^2 rule is caught by a
  mutation arm. All legs route through eval_val, whose band is computed in the
  model's float32; eval_source_chi2 casts to float64, where both width and w^2
  floor to 1e-6 at 780 and could not discriminate.
  """
  torch.manual_seed(11)
  C = torch.randn(N_VAL, N_IN)
  DV = torch.randn(N_VAL, N_OUT)
  model = nn.Linear(N_IN, N_OUT).eval()
  thresholds = torch.tensor([0.2, 0.5, 1.0])

  width = 780
  band = _chi2_neg_band(torch.float32, width)          # the ACTUAL prod band

  # -2 and -4 at a realistic dense width RAISE (before ranking / normalizing):
  # |value| >> band, so the corrupted row is refused, not crowned perfect.
  for value in (-2.0, -4.0):
    loss = _WidthChi2(width)
    loss.bad = {17: value}
    raised, msg = _expect_valueerror(
      lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                       load=N_VAL, bs=N_VAL, thresholds=thresholds))
    report("chi2 band: a %.0f chi2 at width %d raises (band %.6g)"
           % (value, width, band),
           raised and "chi2 domain contract [validation]" in msg
           and "17" in msg, _short(msg))

  # both sides of the ACTUAL band (the scaling term, far above the 1e-6 floor):
  # a half-band roundoff negative is accepted (normalized to 0), a double-band
  # one raises.
  loss = _WidthChi2(width)
  loss.bad = {3: -0.5 * band}
  raised_in, _ = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  loss = _WidthChi2(width)
  loss.bad = {3: -2.0 * band}
  raised_out, _ = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  report("chi2 band: width-%d band %.6g -- half-band accepted, double-band "
         "raises" % (width, band), (not raised_in) and raised_out,
         "roundoff tolerated at the production width, corruption refused")

  # MUTATION arm: restore the retired w^2 rule. band(w^2) = 2.32 at width 780
  # SWALLOWS -2, so eval_val does NOT raise -- the width rule is load-bearing.
  band_wsq = _chi2_neg_band(torch.float32, width * width)
  loss = _WidthChi2Wsq(width)
  loss.bad = {17: -2.0}
  raised_mut, _ = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=loss, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  report("chi2 band: restoring w^2 (band %.4f) accepts -2 at width %d "
         "(mutation -- width rule load-bearing)" % (band_wsq, width),
         not raised_mut, "w^2 reopens the false-crowning hole")

  # scalar-width leg: ScalarChi2 declares n_terms = n_out, not its square.
  n_out_scalar = 96
  scalar_loss = ScalarChi2(types.SimpleNamespace(
    dest_idx=torch.arange(n_out_scalar)))
  report("chi2 band: ScalarChi2 declares n_terms = n_out (%d), not its square"
         % n_out_scalar, scalar_loss._chi2_n_terms() == n_out_scalar,
         "diagonal width inherited from the base, no override")

  # mechanical subclass census: every CosmolikeChi2 subclass returns the base
  # width (no family redefines _chi2_n_terms), so a future diagonal family can
  # not silently inherit a wrong rule. Import the family modules so their
  # subclasses register; the gate's own mutation double is excluded by name.
  import emulator.losses.cmb
  import emulator.losses.scalar
  import emulator.losses.ia
  import emulator.losses.pce
  import emulator.losses.transfer
  overrides = []
  for cls in _all_subclasses(CosmolikeChi2):
    if cls.__name__ == "_WidthChi2Wsq":
      continue                                          # deliberate mutation
    if "_chi2_n_terms" in cls.__dict__:
      overrides.append(cls.__name__)
  report("chi2 band: no CosmolikeChi2 subclass overrides _chi2_n_terms "
         "(one width rule)", overrides == [],
         "census clean" if not overrides
         else "OVERRIDES: " + ", ".join(overrides))

  # ill-conditioned SPD valid control: the genuine roundoff of a near-null
  # dense r^T M r (condition ~1e8) sits well inside the width band
  # [-band, band] -- whichever sign it lands, |roundoff| << band << |a corrupt
  # -2|, so the band admits real roundoff while refusing corruption. This is
  # the growth clause's evidence base: the band may only ever widen on measured
  # valid negatives like these, never for convenience.
  min_chi2 = _spd_roundoff_min(width)
  report("chi2 band: ill-conditioned SPD roundoff %.3e sits inside the width "
         "band [-%.3e, .] (corruption -2 outside)" % (min_chi2, band),
         (min_chi2 >= -band) and (band < 0.5),
         "|genuine roundoff| << band << |corruption|")


def check_chi2_band_dtype_provenance():
  """Part H: the band derives from the COMPUTE dtype.

  eval_source_chi2 formerly upcast the per-row chi2 to float64 before deriving
  the band, flooring it to 1e-6 -- so it REFUSED a roundoff negative that
  training's _reduce and eval_val (the float32 band, ~3e-3 at w = 780) normalize
  to exact 0: one score, two verdicts. These legs pin the three boundaries to
  ONE verdict on a value chosen between the 1e-6 floor and the float32 band,
  show that restoring the .double() upcast would split them, and confirm a
  genuinely float64-computed loss still receives the (tight) float64 band.
  """
  torch.manual_seed(13)
  width = 780
  band32 = _chi2_neg_band(torch.float32, width)          # ~0.002975
  band64 = _chi2_neg_band(torch.float64, width)          # the 1e-6 floor
  # V sits between the two bands: the compute-dtype (float32) band accepts it as
  # roundoff; a float64 band (the upcast bug) would refuse it as corruption.
  value = -5.0e-4

  C = torch.randn(N_VAL, N_IN)
  DV = torch.randn(N_VAL, N_OUT)
  model = nn.Linear(N_IN, N_OUT).eval()
  thresholds = torch.tensor([0.2, 0.5, 1.0])
  params = np.random.default_rng(0).standard_normal((N_SRC, N_IN))
  dv = np.random.default_rng(1).standard_normal((N_SRC, N_OUT))
  source = {"C": params, "dv": dv, "idx": np.arange(N_SRC)}

  # three boundaries, ONE verdict on `value` (accept + normalize to exact 0):
  # (a) training _reduce -- within-band -> normalized, a FINITE loss (not NaN).
  reduce_loss = _reduce_loss(_WidthChi2(width),
                             torch.tensor([1.0, value, 2.0], dtype=torch.float32),
                             "chi2")
  reduce_ok = bool(torch.isfinite(reduce_loss))
  # (b) eval_val -- accepts value (no raise).
  lv = _WidthChi2(width)
  lv.bad = {17: value}
  raised_val, _ = _expect_valueerror(
    lambda: eval_val(model=model, lossfn=lv, data=_val_data(C, DV),
                     load=N_VAL, bs=N_VAL, thresholds=thresholds))
  # (c) eval_source_chi2 -- accepts value (no raise): THE FIX. Pre-fix the
  # float64 upcast floored the band to 1e-6 and this raised.
  ls = _WidthSrcChi2(width, torch.float32)
  ls.bad = {9: value}
  raised_src, _ = _expect_valueerror(
    lambda: eval_source_chi2(model=nn.Linear(N_IN, N_OUT),
                             param_geometry=_IdGeom(), chi2fn=ls,
                             source=source, device=DEV, bs=N_SRC))
  report("chi2 band dtype: _reduce / eval_val / eval_source agree on %.0e at "
         "width %d (one verdict: accept, band %.6g)" % (value, width, band32),
         reduce_ok and (not raised_val) and (not raised_src),
         "float32 compute-dtype band; all three normalize to 0")

  # mutation: restoring the .double() upcast derives the band from float64 (the
  # 1e-6 floor) and marks `value` bad -- eval_source would then raise on a value
  # the compute-dtype band accepts, splitting the verdict.
  _, bad64 = _chi2_domain(torch.tensor([value], dtype=torch.float64), band64)
  report("chi2 band dtype: the .double() upcast (band %.1e) would refuse %.0e "
         "that the float32 band (%.6g) accepts (mutation)"
         % (band64, value, band32),
         bool(bad64.any()) and (band64 < abs(value) < band32),
         "upcast splits the verdict; the compute-dtype band keeps it")

  # a genuinely float64-computed loss still receives the (tight) float64 band:
  # `value` is material there (refused), a within-float64-band -5e-7 accepted.
  lf = _WidthSrcChi2(width, torch.float64)
  lf.bad = {9: value}
  raised_f64_material, _ = _expect_valueerror(
    lambda: eval_source_chi2(model=nn.Linear(N_IN, N_OUT),
                             param_geometry=_IdGeom(), chi2fn=lf,
                             source=source, device=DEV, bs=N_SRC))
  lf = _WidthSrcChi2(width, torch.float64)
  lf.bad = {9: -5.0e-7}
  raised_f64_round, _ = _expect_valueerror(
    lambda: eval_source_chi2(model=nn.Linear(N_IN, N_OUT),
                             param_geometry=_IdGeom(), chi2fn=lf,
                             source=source, device=DEV, bs=N_SRC))
  report("chi2 band dtype: a float64-computed loss gets the float64 band "
         "(-5e-4 refused, -5e-7 accepted)",
         raised_f64_material and (not raised_f64_round),
         "the band tracks the actual compute dtype")


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


def check_optimizer_schema():
  """Part J: the optimizer spec validates its protocol-bearing numeric kwargs.

  The finite objective supports an exact-fit row with a finite, zero gradient;
  a freshly initialized Adam state at zero gradient then forms 0 / (sqrt(0) +
  eps) = 0 / 0 when eps is 0, a non-finite update the pre-step loss and gradient
  guards do not catch. The optimizer spec now rejects a zero (or non-finite) eps
  before the optimizer is built, along with a non-finite / negative weight
  decay, a non-positive / non-finite learning rate, and a beta outside [0, 1).
  Each control is validated by type, not made valid by coercion: a bool (a
  subclass of int) and a numeric string would both pass a float()-then-range
  check, so both are refused at the boundary. (Validating the parameters an
  optimizer step actually produces is the workstation companion to this schema
  check.)
  """
  base = {"cls": torch.optim.AdamW, "weight_decay": 0.0}
  # valid controls pass, including a genuine int learning rate.
  try:
    _validate_optimizer_opts(base, 1e-3)
    _validate_optimizer_opts(dict(base, eps=1e-8, betas=(0.9, 0.999)), 1e-3)
    _validate_optimizer_opts(dict(base), 1)
    report("optimizer schema: valid AdamW opts pass (incl int lr)", True,
           "accepted")
  except ValueError as exc:
    report("optimizer schema: valid AdamW opts pass (incl int lr)", False,
           str(exc))
  # each out-of-range OR wrong-type kwarg is refused. The bool and string legs
  # are the coercion arms: float(True) is 1.0 and float("0.1") is 0.1, so a
  # coercing check would silently accept them.
  cases = [("eps = 0 (the 0/0 zero-gradient trap)", dict(base, eps=0.0), 1e-3),
           ("non-finite eps", dict(base, eps=float("inf")), 1e-3),
           ("negative weight_decay", dict(base, weight_decay=-1.0), 1e-3),
           ("non-positive lr", dict(base), 0.0),
           ("non-finite lr", dict(base), float("nan")),
           ("beta2 == 1.0", dict(base, betas=(0.9, 1.0)), 1e-3),
           ("single beta", dict(base, betas=(0.9,)), 1e-3),
           ("boolean lr (True)", dict(base), True),
           ("boolean lr (False)", dict(base), False),
           ("boolean eps (True)", dict(base, eps=True), 1e-3),
           ("boolean weight_decay (False)", dict(base, weight_decay=False),
            1e-3),
           ("boolean beta (True)", dict(base, betas=(True, 0.999)), 1e-3),
           ("string lr ('0.1')", dict(base), "0.1"),
           ("string eps ('1e-8')", dict(base, eps="1e-8"), 1e-3),
           ("string weight_decay ('0')", dict(base, weight_decay="0"), 1e-3),
           ("string betas ('0.9')", dict(base, betas="0.9"), 1e-3)]
  for label, opts, lr in cases:
    refused = False
    try:
      _validate_optimizer_opts(opts, lr)
    except ValueError:
      refused = True
    report("optimizer schema: " + label + " is refused", refused, "ValueError")


def main():
  """Run every part of the finite contract; return 1 if any leg failed."""
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
  print("\n-- Part F: the safe-sqrt producer --")
  check_safe_sqrt()
  print("\n-- Part G: epoch-reduction finite truth --")
  check_epoch_reduction()
  print("\n-- Part H: the chi2-domain boundary --")
  check_chi2_domain()
  print("\n-- Part H: the chi2-domain band at production widths --")
  check_chi2_band_production()
  print("\n-- Part H: the chi2-domain band's compute-dtype provenance "
        " --")
  check_chi2_band_dtype_provenance()

  print("\n-- Part J: the optimizer-kwarg schema (zero-eps guard) --")
  check_optimizer_schema()

  print("")
  if len(FAILURES) > 0:
    print("finite-contract: " + str(len(FAILURES)) + " FAILURE(S)")
    return EXIT_FAIL
  if len(LANE_UNAVAILABLE) > 0:
    print("finite-contract: NON-GREEN -- a mandatory lane could not run: "
          + ", ".join(LANE_UNAVAILABLE)
          + " (run on a compile-capable box)")
    return EXIT_LANE_UNAVAILABLE
  print("finite-contract: ALL PASS")
  return EXIT_PASS


if __name__ == "__main__":
  sys.exit(main())
