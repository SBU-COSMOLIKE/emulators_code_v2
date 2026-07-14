"""Device selection, construction factories, evaluation, and training.

This module is the run layer that ties the package together. pick_device
and make_logger are setup helpers. make_model, make_optimizer, and
make_scheduler each build one component from a {cls, **kwargs} spec
dict; build_run_specs assembles the six
spec dicts from a config (with the default / suggest / search resolvers for the
[default, min, max, kind] hyperparameter ranges). eval_val and eval_source_chi2
score the model, training_loop_batched is the per-epoch loop (trim / focus /
berhu-blend / EMA annealing, best-epoch tracking on the EMA average when ema:
is set), and run_emulator orchestrates: builds everything, trains, and returns
the model plus the per-epoch histories.

In front of the loop sits a pure configuration layer: validate_phase_block,
validate_loss, validate_berhu, and validate_ema check and canonicalize the
train_args blocks (the eight-key per-phase whitelist, the nested loss {mode,
berhu, roughness} block, the berhu {knot, cap, anneal} schedule, the ema
{horizon_epochs, anneal} block) before anything runs; derive_eval_bs and
derive_ema_beta turn
run-global targets into the evaluation batch size and the per-epoch EMA decay.
The loss modes the loop can apply are chi2 / sqrt / sqrt_dchi2 / berhu /
berhu_capped (losses/core.py).

PS: a loader is a closure ``load(rows) -> tensor``. The row numbers address
the active source array: local coordinates for a compact RAM copy and original
dump-row coordinates for a disk-backed source. It returns a ready-to-train
batch on the compute device and hides whether data are resident on the GPU,
streamed from RAM, or read from a disk memmap. build_loaders
(batching.py) makes one loader for the whitened parameters (load_C) and one for
the encoded targets (load_dv) per source; the loop here just asks for the rows
it wants. whitened = rotated into a covariance eigenbasis and scaled to unit
variance, decorrelating the components (the form the model sees, input and
target). encoded = a dv put through the geometry's encode (kept entries,
centered, whitened). dump = the full on-disk array from the data-generation
run, one row per cosmology (the dv dump is the .npy, the param dump the .txt).
memmap = a NumPy array backed by that file, read in slices so it is never
loaded whole.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from .batching import build_loaders
from .designs.plain import ResMLP
from .designs.blocks import Affine, BinLinear
from .losses.core import anneal_value, screen_chi2


def pick_device(name=None):
  """
  Choose the compute device: an explicit name, else auto-detect.

  Arguments:
    name = force a device by string ("cuda" / "mps" / "cpu"); None
           (default) auto-detects CUDA, else Apple MPS, else CPU.

  Returns:
    a torch.device.
  """
  if name is not None:
    return torch.device(name)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def make_logger(quiet=False):
  """
  Build a print function gated by a quiet flag.

  Returns a `log(*args, **kwargs)` callable that forwards to the
  builtin print when `quiet` is False and is a no-op when True,
  the standard "--quiet" stdout gate a CLI driver wraps its prints
  in. `quiet` is captured once, at build time.

  Arguments:
    quiet = if True, the returned logger swallows every call (prints
            nothing); if False (default), it forwards to print.

  Returns:
    log = a function with print's signature (*args, **kwargs) that
          prints unless quiet.
  """
  def log(*args, **kwargs):
    if not quiet:
      print(*args, **kwargs)
  return log


# the default torch.compile mode on CUDA; the one source of truth make_model
# applies and the save recipe records, so a persisted compile_mode can never
# diverge from what the model was actually built with.
DEFAULT_COMPILE_MODE = "reduce-overhead"


def make_model(model_opts, input_dim, output_dim, device,
               init_state=None):
  """
  Build the network from a spec dict.

  Mirrors make_optimizer: the model class is a value in the spec
  dict, its settings the other keys, so swapping architectures is
  a one-dict change.

  Arguments:
    model_opts = model spec dict. "cls" is the model class (e.g.
                 ResMLP), stored as a value (same factory trick as
                 the optimizer). "compile_mode" (optional) sets the
                 CUDA torch.compile mode (see below). Every other
                 key forwards to the constructor (int_dim_res,
                 n_blocks, block_opts, ...).
    input_dim  = number of input features (the cosmological
                 parameter count); injected, not in the dict.
    output_dim = number of outputs (unmasked dv length); injected.
    device     = device to build on. On CUDA the model is
                 torch.compile'd per compile_mode; eager on MPS/CPU.
    init_state = optional warm-start state dict (fine-tune path,
                 emulator/warmstart.py). When given, it is loaded
                 strict into the eager module before any
                 torch.compile, so its keys match the plain
                 architecture and no OptimizedModule "_orig_mod."
                 prefix enters. None (default) = the ordinary
                 random-init build, byte-identical.

  compile_mode (CUDA only; default "reduce-overhead"):
    "reduce-overhead" = inductor + CUDA graphs; fastest but fragile
      large constant buffers or a skip-add of the trunk output
      can trip CUDA-graph-trees bookkeeping (internal
      AssertionError during warmup).
    "default" = inductor kernel fusion, no CUDA graphs (robust).
    None      = plain eager (no compile).
  Returns:
    the model on `device`, compiled per compile_mode on CUDA.
  """
  cls = model_opts["cls"]
  compile_mode = model_opts.get(
    "compile_mode", DEFAULT_COMPILE_MODE)
  # forward every key except cls / compile_mode to the constructor.
  extra = {}
  for k, v in model_opts.items():
    if k not in ("cls", "compile_mode"):
      extra[k] = v
  model = cls(input_dim=input_dim,
              output_dim=output_dim, **extra).to(device)
  # fine-tune warm start: load the transferred weights into the eager
  # module before torch.compile, so the state dict carries the plain
  # architecture's keys (a compiled OptimizedModule prefixes them).
  if init_state is not None:
    model.load_state_dict(init_state, strict=True)
  if device.type == "cuda" and compile_mode is not None:
    model = torch.compile(model, mode=compile_mode)
    # record the mode as a plain attribute (reads reach through
    # the wrapper): training_loop_batched compiles its combined
    # forward+loss step the same way, so the loss math joins the
    # model in one compiled graph instead of launching eagerly
    # kernel by kernel (the training loop is launch-bound: a tiny
    # model's epoch time is mostly the CPU dispatching kernels).
    model.emul_compile_mode = compile_mode
  return model


def _is_finite_real(value):
  """True only for a finite, non-boolean real number (a plain int or float).

  A public numeric control is validated by type, never made valid by
  coercion. Two values would slip through a float()-then-range check: bool is
  a subclass of int (float(True) is 1.0), and a numeric string converts
  (float("0.1") is 0.1). This admits only a genuine int or float that is not a
  bool and whose value is finite, so True / False / "0.1" / NaN / inf are all
  rejected at the boundary rather than silently accepted.

  Arguments:
    value = the candidate configuration value.

  Returns:
    True when value is a finite int or float and not a bool.
  """
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return False
  return math.isfinite(value)


def _validate_optimizer_opts(opt_opts, lr):
  """Validate an optimizer spec's protocol-bearing numeric kwargs.

  The optimizer spec forwards arbitrary kwargs into the constructor, so a
  config can set an Adam-family epsilon of 0. The finite objective deliberately
  supports an exact-fit row with a finite, zero gradient; a freshly initialized
  Adam state at zero gradient then forms 0 / (sqrt(0) + eps) = 0 / 0 when eps is
  0, a non-finite update the pre-step guards (the scalar loss and the gradient
  norm) do not catch because both are finite. This validates the numeric kwargs
  before the optimizer is built: the learning rate finite and positive, the
  weight decay finite and nonnegative, an Adam-family eps finite and strictly
  positive, and each beta finite in the half-open range 0 to 1. Validating the
  parameters and optimizer state produced by each step is the workstation
  companion to this schema check.

  Arguments:
    opt_opts = the optimizer spec dict (the class, the weight decay, and any
               forwarded constructor kwargs such as eps / betas).
    lr       = the resolved base learning rate.

  Raises:
    ValueError naming the offending kwarg and the range it must fall in.
  """
  if not (_is_finite_real(lr) and lr > 0.0):
    raise ValueError(
      "optimizer lr must be a finite positive real (not a bool or string); "
      "got " + repr(lr))
  weight_decay = opt_opts.get("weight_decay", 0.0)
  if not (_is_finite_real(weight_decay) and weight_decay >= 0.0):
    raise ValueError(
      "optimizer weight_decay must be a finite nonnegative real (not a bool "
      "or string); got " + repr(weight_decay))
  if "eps" in opt_opts:
    eps = opt_opts["eps"]
    if not (_is_finite_real(eps) and eps > 0.0):
      raise ValueError(
        "optimizer eps must be a finite strictly positive real (not a bool or "
        "string); an Adam-family eps of 0 forms 0 / (sqrt(0) + 0) at a "
        "zero-gradient exact-fit row, a non-finite update the loss and "
        "gradient guards do not catch. Got " + repr(eps))
  if "betas" in opt_opts:
    # a string ("0.9") tuples into single characters, so a non-pair (including
    # a string) is rejected here before the per-element real check.
    try:
      pair = tuple(opt_opts["betas"])
    except TypeError:
      pair = ()
    if len(pair) != 2 or isinstance(opt_opts["betas"], str):
      raise ValueError(
        "optimizer betas must be a pair (beta1, beta2); got "
        + repr(opt_opts["betas"]))
    for index, beta in enumerate(pair):
      if not (_is_finite_real(beta) and 0.0 <= beta < 1.0):
        raise ValueError(
          "optimizer beta" + str(index + 1) + " must be a finite real in "
          "[0, 1) (not a bool or string); got " + repr(beta))


def make_optimizer(model, opt_opts, lr, device):
  """
  Build the optimizer from a spec dict.

  Mirrors make_model / make_scheduler: the optimizer class is a
  value in the dict, its settings the other keys. Parameters split
  into two groups so weight decay falls only on true weight matrices,
  chosen by module role, not tensor shape: the .weight of every
  nn.Linear / nn.Conv1d / BinLinear. Everything else is undecayed —
  biases, Affine / FeatureAffine gains, and every activation
  parameter of any shape (e.g. multigate's (K, dim) w / beta / mu) —
  since decaying a bias or an activation shape parameter has no
  principled meaning and would drag the activation toward degenerate
  forms. A module left off the allowlist defaults to undecayed (the
  safe failure direction).

  Arguments:
    model    = network whose parameters are optimized;
               named_parameters() splits into the decayed weight
               matrices (the .weight of nn.Linear / nn.Conv1d /
               BinLinear) and everything else undecayed (all biases,
               Affine / FeatureAffine gain/bias, every activation
               parameter, whatever its shape).
    opt_opts = optimizer spec dict. "cls" is the optimizer
               class (e.g. optim.AdamW), stored as a value;
               "weight_decay" (optional, default 0.0) decays
               the weight-matrix group only; other keys forward
               to the constructor (betas, eps, ...).
    lr       = learning rate; injected (the sqrt-batch-scaled
               value), not in the dict.
    device   = device the model lives on; on CUDA the fused
               optimizer kernel is enabled.

  Returns:
    the optimizer, with two param groups: weight matrices decayed
    by opt_opts["weight_decay"], the rest at 0. fused on CUDA.
  """
  # Decay only true weight matrices: the .weight of nn.Linear /
  # nn.Conv1d / BinLinear. Collect those weights by id from a module
  # walk, then split every parameter on membership — so biases,
  # Affine / FeatureAffine gains, and every activation parameter (any
  # shape, e.g. multigate's (K, dim) w / beta / mu, and BinLinear's
  # (G, out) biases) stay undecayed, decided by module role not tensor
  # shape. A future module left off the list defaults to undecayed
  # (the safe direction). named_parameters() yields each shared
  # parameter once, so the id split needs no extra dedupe.
  decay_ids = set()
  for m in model.modules():
    if isinstance(m, (nn.Linear, nn.Conv1d, BinLinear)):
      w = getattr(m, "weight", None)
      if w is not None:
        decay_ids.add(id(w))
  decay, no_decay = [], []
  for _, p in model.named_parameters():
    if id(p) in decay_ids:
      decay.append(p)
    else:
      no_decay.append(p)
  _validate_optimizer_opts(opt_opts, lr)
  wd    = opt_opts.get("weight_decay", 0.0)
  cls   = opt_opts["cls"]
  # forward every key except cls / weight_decay to the constructor.
  extra = {}
  for k, v in opt_opts.items():
    if k not in ("cls", "weight_decay"):
      extra[k] = v
  groups = [
    {"params": decay,    "weight_decay": wd},
    {"params": no_decay, "weight_decay": 0.0},
  ]
  # fused is a CUDA-only Adam/SGD-family speedup.
  if device.type == "cuda":
    extra["fused"] = True
  return cls(groups, lr=lr, **extra)


class Anchor:
  """A decoupled L2-SP anchor: a post-step pull toward a reference W_0.

  The shared anchor facility for both the finetune warm start (anchor to the
  transferred weights) and the transfer refine stage (anchor the unfrozen base
  to its pretrained weights). After each optimizer.step, under no_grad, it
  applies in place

      W <- W - lr * lambda * mask * (W - W_0)

  where lr is read from the parameter's own optimizer group (so a
  discriminative per-group learning rate carries through). This is the
  AdamW-decoupling argument applied to the anchor: kept OUT of the loss, it is
  not rescaled by Adam's adaptive second moments (a loss-term
  lambda*||W - W_0||^2 would be), so lambda means the same thing across
  parameters. It mirrors the EMA post-step in-place update already in
  training_loop_batched (the same cheap _foreach-style pass).

  mask (per parameter, optional) zeroes the pull on columns that must stay
  free: the finetune warm start excludes the padded extra input columns (exact
  zeros by design, the carriers of the new-physics dependence, so anchoring
  them to zero would fight the warm start). None = pull every element.

  Note on weight_decay: the optimizer's weight decay pulls toward 0, i.e. AWAY
  from W_0, so an anchored run should normally set weight_decay 0.0 (a
  recommendation surfaced as a quiet notice by the caller, never an error).

  PS: W_0 = the reference (anchor) weights the run is pulled back toward;
  decoupled = applied after the optimizer step, not inside the loss, so it
  bypasses the optimizer's adaptive moment rescaling.
  """

  def __init__(self, entries, lam):
    """Store the anchored parameters and the strength.

    Arguments:
      entries = list of (param, reference, mask, group_index): the live
                parameter, its frozen reference W_0 (same shape / device), an
                optional element mask (None = all ones), and the index of the
                optimizer param group the parameter belongs to (its lr).
      lam     = the anchor strength lambda (0.0 = a no-op, free training).
    """
    self.entries = entries
    self.lam     = float(lam)

  def apply(self, optimizer):
    """Do one decoupled anchor step in place, after optimizer.step().

    Arguments:
      optimizer = the optimizer just stepped; its param_groups[gi]["lr"] gives
                  each anchored parameter's current (warmed / scheduled) lr.
    """
    if self.lam == 0.0:
      return
    with torch.no_grad():
      for p, ref, mask, gi in self.entries:
        coef  = optimizer.param_groups[gi]["lr"] * self.lam
        delta = p - ref
        if mask is not None:
          delta = delta * mask
        # W <- W - coef * mask * (W - W_0), decoupled from the loss.
        p.add_(delta, alpha=-coef)


def build_anchor(model, optimizer, reference_state, lam, masks=None):
  """Assemble an Anchor over a model's parameters against a reference state.

  Only parameters present in reference_state AND owned by an optimizer group
  (a frozen parameter has none) are anchored, so a stage that freezes part of
  the model anchors only what it trains.

  Arguments:
    model           = the network whose parameters are anchored.
    optimizer       = the built optimizer (its groups give each param its lr).
    reference_state = name -> W_0 tensor (e.g. the transferred init_state, or
                      the pretrained base weights).
    lam             = the anchor strength lambda.
    masks           = optional name -> element mask (same shape as the param;
                      None entry or absent = pull every element).

  Returns:
    an Anchor (its apply() is called after each optimizer.step).
  """
  # param id -> its optimizer group index (for the per-group lr).
  id_to_group = {}
  for gi, group in enumerate(optimizer.param_groups):
    for p in group["params"]:
      id_to_group[id(p)] = gi
  entries = []
  for name, p in model.named_parameters():
    if name not in reference_state:
      continue
    gi = id_to_group.get(id(p))
    if gi is None:
      continue                         # a frozen parameter: nothing to anchor
    ref  = reference_state[name].to(device=p.device, dtype=p.dtype)
    mask = None if masks is None else masks.get(name)
    entries.append((p, ref, mask, gi))
  return Anchor(entries=entries, lam=lam)


class TransferComposite(nn.Module):
  """Wraps the correction net and the unfrozen base for the transfer refine
  stage. The forward returns the correction's output (the base is
  evaluated by the loss, TransferChi2 in live mode); holding both as submodules
  lets one optimizer, one gradient clip, and one train/eval toggle cover both,
  and lets make_refine_optimizer split them into discriminative-lr groups. The
  base object is shared with the loss (chi2fn.base_net), so training it here is
  what the live composition sees."""

  def __init__(self, correction, base):
    """Store the two submodules.

    Arguments:
      correction = the stage-1 correction network (its output is the forward).
      base       = the now-trainable base network (evaluated by the live loss).
    """
    super().__init__()
    self.correction = correction
    self.base        = base

  def forward(self, x):
    return self.correction(x)


def _decay_weight_ids(module):
    """ids of the .weight of every nn.Linear / nn.Conv1d / BinLinear in a module
    (the only tensors weight decay touches; see make_optimizer)."""
    ids = set()
    for m in module.modules():
      if isinstance(m, (nn.Linear, nn.Conv1d, BinLinear)):
        w = getattr(m, "weight", None)
        if w is not None:
          ids.add(id(w))
    return ids


def make_refine_optimizer(correction, base, opt_opts, lr, base_lr_scale,
                          device):
  """Optimizer with discriminative per-module learning rates (the refine stage).

  Four parameter groups: the base's decayed / undecayed weights at
  lr*base_lr_scale, the correction's at the full lr. The decay split is
  make_optimizer's (only true weight matrices decay, by module role); the
  per-group lr is preserved through the loop's warmup and plateau scheduler
  (both scale every group from its own captured base lr, so the ratio holds).

  Arguments:
    correction    = the correction network (the full-lr group).
    base          = the base network (the scaled-lr group; unfrozen for refine).
    opt_opts      = the optimizer spec dict ("cls" + "weight_decay" + extras).
    lr            = the run's resolved (sqrt-batch) learning rate.
    base_lr_scale = the base group's lr as a multiple of lr.
    device        = the device (fused Adam on CUDA).

  Returns:
    the optimizer with the four groups.
  """
  _validate_optimizer_opts(opt_opts, lr)
  wd  = opt_opts.get("weight_decay", 0.0)
  cls = opt_opts["cls"]
  extra = {}
  for k, v in opt_opts.items():
    if k not in ("cls", "weight_decay"):
      extra[k] = v
  if device.type == "cuda":
    extra["fused"] = True
  base_lr = lr * float(base_lr_scale)
  groups  = []
  for module, mlr in ((base, base_lr), (correction, lr)):
    dids = _decay_weight_ids(module)
    dec, nod = [], []
    for _, p in module.named_parameters():
      if id(p) in dids:
        dec.append(p)
      else:
        nod.append(p)
    groups.append({"params": dec,
                   "weight_decay": wd,
                   "lr": mlr})
    groups.append({"params": nod,
                   "weight_decay": 0.0,
                   "lr": mlr})
  return cls(groups, lr=lr, **extra)


def make_scheduler(optimizer, sched_opts):
  """
  Build the LR scheduler from a spec dict.

  Mirrors make_model / make_optimizer: the scheduler class is a
  value in the dict, its settings the other keys, so swapping
  schedulers is a one-dict change.

  Arguments:
    optimizer  = the optimizer whose learning rate this
                 schedules; injected, not in the dict.
    sched_opts = scheduler spec dict. "cls" is the scheduler
                 class (e.g. lr_scheduler.ReduceLROnPlateau),
                 stored as a value; every other key forwards to
                 its constructor (mode, patience, factor, ...).

  Returns:
    the constructed scheduler.
  """
  cls   = sched_opts["cls"]
  # forward every key except cls to the constructor.
  extra = {}
  for k, v in sched_opts.items():
    if k != "cls":
      extra[k] = v
  return cls(optimizer, **extra)


# the eight keys a per-phase override block (train_args.trunk / .head) may
# carry, mirroring the top-level train_args schema key-for-key: lr is an
# overlay onto the top-level lr block, scheduler a full replacement of the
# scheduler kwargs, the other six (loss / trim / focus / clip / rewind / ema)
# replace their same-named top-level value (loss is the nested loss block
# {mode, berhu}, ema the weight-average block {horizon_epochs, anneal}; both
# full replacement like trim/focus and validated by validate_loss /
# validate_ema (a phase ema: null disables an inherited top-level ema for
# that pass). validate_phase_block is the one whitelist, imported by
# experiment.py (training must not import experiment, so the validator
# lives here).
_PHASE_BLOCK_KEYS = ("lr", "scheduler", "loss", "trim", "focus",
                     "clip", "rewind", "ema")


def _phase_lr_migration_message(block, which):
  """Build the paste-ready nested lr: block for the flat lr_base migration.

  Arguments:
    block = the phase block holding the offending flat lr_base.
    which = "trunk" or "head", named in the message.

  Returns:
    a multi-line message whose body is a valid, paste-ready lr: sub-block
    carrying the old lr_base value over.
  """
  lines = [
    f"train_args.{which}.lr_base is gone: the phase blocks now mirror the",
    "top-level lr: schema, so the base learning rate nests under an lr:",
    f"sub-block. Replace it under {which}: with:",
    "",
    "  lr:",
    f"    lr_base: {block['lr_base']}",
  ]
  return "\n".join(lines)


def validate_phase_block(block, which):
  """
  Validate one per-phase override block (train_args.trunk or .head).

  A standalone pure function (no torch), so experiment.py can import it for
  the single-phase demotion path and it is unit-testable in isolation. The
  phase blocks mirror the top-level train_args schema: lr is a nested
  sub-block (overlay), scheduler a nested kwargs block (full replacement),
  the other six keys scalars / their own blocks. Absent (None) validates
  trivially and the run is unchanged. The head block alone also accepts a
  ninth key, activation: the head-only alias for the per-head activation
  pin (model.<head>.activation), a construction knob consumed by
  build_specs, not a training key (the per-pass resolution never reads it);
  trunk: activation: is a config error (the trunk is the same modules in
  both phases).

  Arguments:
    block = the trunk: or head: mapping, or None when the block is absent.
    which = "trunk" or "head", named in every error message.

  Returns:
    the block unchanged (or None).

  Raises:
    TypeError if the block (or its lr / scheduler sub-block) is present but
    not a mapping (a scalar `trunk: sqrt` is a config error, not a silent
    no-op). ValueError on a bare lr_base or a flat loss_mode / berhu (each
    a migration error printing the paste-ready nested block), an unknown
    key (the eight-key whitelist), a bs_base inside the phase lr (the
    sqrt-rule batch anchor is run-global), or a cls inside the phase
    scheduler (the scheduler class is the run's; a phase overrides only its
    kwargs).
  """
  if block is None:
    return None
  if not isinstance(block, dict):
    raise TypeError(
      f"train_args.{which} must be a mapping of per-phase overrides, got "
      f"{type(block).__name__}")
  # the old flat lr_base -> a migration error printing the nested lr: block
  # to paste in its place (the phase blocks now mirror the top-level lr:).
  if "lr_base" in block:
    raise ValueError(_phase_lr_migration_message(block, which))
  # the flat loss_mode / berhu keys are gone: loss options nest under one
  # loss: block now. Reject either with the paste-ready loss: block, not a
  # bare unknown-key error.
  if "loss_mode" in block or "berhu" in block:
    raise ValueError(_loss_migration_message(
      block.get("loss_mode"), block.get("berhu"), which))
  # head: activation: is the head-only alias for the per-head activation
  # pin (model.<head>.activation): a construction knob consumed by
  # build_specs, not a training key, so it is accepted here for the head
  # block only (the per-pass resolution never reads it). trunk: activation:
  # is a config error, the trunk being the same modules in both phases.
  allowed = set(_PHASE_BLOCK_KEYS)
  if "activation" in block:
    if which != "head":
      raise ValueError(
        f"train_args.{which}.activation: the trunk is the same modules "
        f"in both phases, so it cannot have a phase-local activation — "
        f"set model.activation (the run's trunk + default family). "
        f"head: activation: is accepted (the head only trains in phase "
        f"2); its canonical spelling is model.cnn.activation / "
        f"model.trf.activation.")
    allowed = allowed | {"activation"}
  unknown = set(block) - allowed
  if unknown:
    raise ValueError(
      f"unknown train_args.{which} key(s): {sorted(unknown)}; a phase "
      f"block overrides only {list(_PHASE_BLOCK_KEYS)}"
      + (" (plus activation, the head-only pin alias)"
         if which == "head" else ""))
  # lr is an overlay of {lr_base, warmup_epochs}; bs_base is run-global.
  lr = block.get("lr")
  if lr is not None:
    if not isinstance(lr, dict):
      raise TypeError(
        f"train_args.{which}.lr must be a mapping {{lr_base, "
        f"warmup_epochs}}, got {type(lr).__name__}")
    if "bs_base" in lr:
      raise ValueError(
        f"train_args.{which}.lr must not set bs_base: the sqrt-rule batch "
        f"anchor is run-global (set it once in the top-level lr: block)")
    lr_unknown = set(lr) - {"lr_base", "warmup_epochs"}
    if lr_unknown:
      raise ValueError(
        f"unknown train_args.{which}.lr key(s): {sorted(lr_unknown)}; a "
        f"phase lr overlays only lr_base / warmup_epochs")
  # scheduler is a full replacement of the kwargs; the class stays the run's.
  sched = block.get("scheduler")
  if sched is not None:
    if not isinstance(sched, dict):
      raise TypeError(
        f"train_args.{which}.scheduler must be a mapping of scheduler "
        f"kwargs, got {type(sched).__name__}")
    if "cls" in sched:
      raise ValueError(
        f"train_args.{which}.scheduler must not set cls: the scheduler "
        f"class is the run's; a phase overrides only its kwargs "
        f"(mode / patience / factor / ...)")
  # loss / ema are the nested blocks a phase may carry; validate them here so
  # a malformed phase block fails identically on the two-phase path
  # (run_emulator) and the single-phase demotion (resolve_phase_args), and
  # even when the block is later dropped (a head: on a single-phase model).
  # A phase ema: null (key present, value None) validates trivially; it is
  # the opt-out (disable an inherited top-level ema for that pass).
  validate_loss(block.get("loss"), which)
  validate_ema(block.get("ema"), which)
  return block


def build_run_specs(train_args, model_cls, opt_cls, sched_cls):
  """
  Assemble the six run_emulator spec dicts from a config mapping.

  Each constructible component is a {"cls": <class>, **kwargs}
  spec, the first-class-class trick make_model / make_optimizer
  / make_scheduler consume. The caller picks the class (a driver
  fixes ResMLP / AdamW / ReduceLROnPlateau, or swaps any), its
  settings come from the matching sub-block of train_args, spread
  in with **; the classless schedules (lr, trim, focus) are copied
  verbatim. Keyed by the exact run_emulator argument names, so a
  caller can splat it: run_emulator(..., **build_run_specs(...)).

  Spreading **train_args["model"] (etc.) means this never has to
  know a class's kwargs: whatever serializable settings the YAML
  lists flow through (int_dim_res / n_blocks for ResMLP,
  kernel_size / n_blocks_cnn for ResCNN, ...). Each {...} / dict(...)
  builds a new dict, never mutating the input mapping.

  Arguments:
    train_args = mapping (e.g. a YAML "train_args" block) holding
                 the sub-mappings "model", "optimizer", "lr",
                 "scheduler", "trim", "focus". Each carries only
                 serializable settings; injected/runtime args
                 (device, in/out dims, lr value, a geometry) are
                 added later by make_X or the driver.
    model_cls  = model class for model_opts["cls"] (ResMLP, ...).
    opt_cls    = optimizer class for opt_opts["cls"] (AdamW, ...).
    sched_cls  = scheduler class for sched_opts["cls"]
                 (ReduceLROnPlateau, ...).

  Returns:
    dict with keys model_opts, opt_opts, lr_opts, sched_opts,
    trim_opts, focus_opts, the six spec dicts run_emulator takes.
  """
  return {
    "model_opts": {"cls": model_cls, **train_args["model"]},
    "opt_opts":   {"cls": opt_cls,   **train_args["optimizer"]},
    "lr_opts":    dict(train_args["lr"]),
    "sched_opts": {"cls": sched_cls, **train_args["scheduler"]},
    "trim_opts":  dict(train_args["trim"]),
    "focus_opts": dict(train_args["focus"]),
  }


# --- hyperparameter search ranges inside train_args ---
# A train_args leaf is either a fixed scalar or a search range
# [default, min, max, kind] with kind one of "int" / "float" /
# "log" (a whitespace string "default min max kind" also works).
# The first value is the default: the plain driver uses it, a
# search warm-starts trial 0 from it. The three resolvers below
# share one walk over the (nested) mapping.
_SEARCH_KINDS = ("int", "float", "log")


def _as_search_range(value):
  """Return [default, min, max, kind] if value marks a search range.

  Accepts a YAML 4-list or a whitespace string "d min max kind";
  returns None for a fixed scalar (or any non-range value), leaving
  it untouched in the walk.
  """
  if isinstance(value, str):
    parts = value.split()
    value = parts if len(parts) == 4 else value
  if (isinstance(value, (list, tuple)) and len(value) == 4
      and str(value[3]) in _SEARCH_KINDS):
    return [value[0], value[1], value[2], str(value[3])]
  return None


def _range_default(rng):
  """A range's default (first) value, typed by its kind."""
  d, kind = rng[0], rng[3]
  return int(d) if kind == "int" else float(d)


def _suggest_range(trial, name, rng):
  """One Optuna suggestion for a search range.

  kind selects the suggestion: "int" -> suggest_int, "float" ->
  suggest_float (linear), "log" -> suggest_float(log=True). min/max
  are cast, so a YAML 1e-5 that parsed as a string still works.
  """
  _, lo, hi, kind = rng
  if kind == "int":
    return trial.suggest_int(name, int(lo), int(hi))
  if kind == "log":
    return trial.suggest_float(name, float(lo), float(hi), log=True)
  return trial.suggest_float(name, float(lo), float(hi))


def _walk_train_args(train_args, path, on_leaf):
  """Recurse train_args, applying on_leaf(path, value) to each leaf.

  Returns a new mapping with the same nesting (an explicit loop
  rebuilds each level, so the input is never mutated); on_leaf
  decides each leaf.
  """
  if isinstance(train_args, dict):
    out = {}
    for k, v in train_args.items():
      child_path = f"{path}.{k}" if path else k
      out[k] = _walk_train_args(v, child_path, on_leaf)
    return out
  return on_leaf(path, train_args)


def default_train_args(train_args):
  """
  Resolve every search range to its default, a fixed config.

  Walks train_args (any nesting) and replaces each
  [default, min, max, kind] range with its default value, leaving
  scalars untouched. This lets the plain training driver consume a
  YAML that also carries search ranges, it uses each range's
  first value, so one YAML serves both the plain and search drivers.

  Arguments:
    train_args = a YAML "train_args" mapping (may hold ranges).

  Returns:
    the same mapping with every range collapsed to its default.
  """
  def leaf(path, v):
    rng = _as_search_range(v)
    return _range_default(rng) if rng else v
  return _walk_train_args(train_args, "", leaf)


def suggest_train_args(trial, train_args):
  """
  Resolve train_args for one Optuna trial.

  Walks train_args; each [default, min, max, kind] range becomes an
  Optuna suggestion named by its dotted path (e.g. "lr.lr_base"),
  each scalar kept. Returns a fully-resolved train_args (same nested
  shape) ready for build_run_specs / run_emulator. Never imports
  optuna, it only calls the passed trial's suggest_* .

  Arguments:
    trial      = an optuna Trial (the source of the suggestions).
    train_args = a YAML "train_args" mapping (may hold ranges).

  Returns:
    train_args with every range replaced by this trial's sample.
  """
  def leaf(path, v):
    rng = _as_search_range(v)
    return _suggest_range(trial, path, rng) if rng else v
  return _walk_train_args(train_args, "", leaf)


def search_defaults(train_args):
  """
  The {dotted-path: default} of every search range in train_args.

  The warm-start point for a study (enqueue it as trial 0), keyed
  to match the names suggest_train_args registers. Empty when no
  range is present.

  Arguments:
    train_args = a YAML "train_args" mapping (may hold ranges).

  Returns:
    a dict {path: default} over the ranges, typed by their kind.
  """
  out = {}

  def leaf(path, v):
    rng = _as_search_range(v)
    if rng:
      out[path] = _range_default(rng)
    return v
  _walk_train_args(train_args, "", leaf)
  return out


def audit_devices(model, lossfn, device):
  """
  Name every tensor that should live on `device` but does not.

  A stray off-device tensor inside the compiled forward+loss
  disables CUDA-graph replay silently: inductor only warns
  "skipping cudagraphs due to cpu device (primals_N)", and
  primals_N is a traced-input position, useless for finding the
  owner. This walk names owners directly. Checked: every model
  parameter and buffer (named_* reach through a torch.compile
  wrapper), and every tensor attribute of lossfn and of
  lossfn.geom, the loss's geometry carries the whitening /
  covariance tensors the chi2 contracts with, so they enter the
  traced graph too.

  A mismatch is a different device type (cpu vs cuda), or a
  different index when both sides pin one (cuda:0 vs cuda:1 on a
  two-GPU box breaks replay just as silently).

  Arguments:
    model  = the network (possibly torch.compile'd).
    lossfn = the chi2/loss object; its .geom is walked when
             present.
    device = the device everything should live on.

  Returns:
    a list of "owner: actual_device" strings; empty when clean.
  """
  def mismatch(t):
    if t.device.type != device.type:
      return True
    if device.index is not None and t.device.index is not None:
      return t.device.index != device.index
    return False

  bad = []
  for name, p in model.named_parameters():
    if mismatch(p):
      bad.append(f"model parameter {name}: {p.device}")
  for name, b in model.named_buffers():
    if mismatch(b):
      bad.append(f"model buffer {name}: {b.device}")
  owners = [("lossfn", lossfn)]
  geom = getattr(lossfn, "geom", None)
  if geom is not None:
    owners.append(("lossfn.geom", geom))
  for label, obj in owners:
    # vars() = the instance's plain attributes (the geometry
    # classes store their tensors that way); non-tensors skipped.
    for name, v in vars(obj).items():
      if isinstance(v, torch.Tensor) and mismatch(v):
        bad.append(f"{label}.{name}: {v.device}")
  return bad


# the batch-size scale the validation pass aims for. Not a YAML key: the
# eval batch is a performance choice with a computable optimum (see
# derive_eval_bs), so it is derived, not user-selected. Only sets the
# scale, since the forward + chi2 matmuls saturate the GPU by a few
# hundred rows.
_EVAL_BS_TARGET = 1024

# the default knot / cap of the berhu loss family (mode "berhu" /
# "berhu_capped"), used when a berhu mode runs with no berhu: sub-block in
# its loss block. knot 0.2 is the frac>0.2 goal threshold; cap 10.0 the
# catastrophic band above which berhu_capped stops escalating the vote.
_BERHU_DEFAULTS = {"knot": 0.2, "cap": 10.0}

# the keys an anneal: sub-block accepts (trim's argument names) and the
# schedule shapes it may name (the anneal_value helper's set). Shared by the
# two anneal features: loss.berhu.anneal ramps the berhu blend, ema.anneal
# ramps the average's horizon; both feed {start: 0, end: 1, **this block} to
# anneal_value. See _validate_anneal_block + the s_t / beta(e) threading in
# training_loop_batched.
_ANNEAL_KEYS   = ("hold_epochs", "anneal_epochs", "shape")
_ANNEAL_SHAPES = ("const", "linear", "cosine", "step")


def _validate_anneal_block(anneal, which):
  """
  Validate an optional anneal: sub-block (the trim-style 0 -> 1 schedule).

  Shared by loss.berhu.anneal (ramps the sqrt -> berhu blend) and ema.anneal
  (ramps the average's horizon); the anneal is feature-agnostic in substance
  (trim's argument names, fed to the existing anneal_value helper with
  start 0 / end 1, no new schedule code). Standalone / pure (no torch).

  Arguments:
    anneal = the raw anneal sub-block.
    which  = the owning block's path below train_args (e.g. "loss.berhu",
             "trunk.loss.berhu", "ema", "trunk.ema"); the block is
             train_args.{which}.anneal, named in error messages.

  Returns:
    the validated anneal mapping (a copy).

  Raises:
    TypeError if anneal is not a mapping. ValueError on an unknown key, a
    missing key, a non-integer / negative hold_epochs, an anneal_epochs
    below 1 (bool rejected in both), or an unknown shape.
  """
  q = f"train_args.{which}.anneal"
  if not isinstance(anneal, dict):
    raise TypeError(
      f"{q} must be a mapping {{hold_epochs, anneal_epochs, shape}}, got "
      f"{type(anneal).__name__}")
  unknown = set(anneal) - set(_ANNEAL_KEYS)
  if unknown:
    raise ValueError(
      f"unknown {q} key(s): {sorted(unknown)}; allowed: "
      f"{sorted(_ANNEAL_KEYS)}")
  for key in _ANNEAL_KEYS:
    if key not in anneal:
      raise ValueError(
        f"{q} needs '{key}' (the sub-block is {{{', '.join(_ANNEAL_KEYS)}}})")
  # hold_epochs >= 0, anneal_epochs >= 1, both integers (bool is an int
  # subclass but never an epoch count).
  for key, lo in (("hold_epochs", 0), ("anneal_epochs", 1)):
    val = anneal[key]
    if isinstance(val, bool) or not isinstance(val, int):
      raise ValueError(
        f"{q}.{key} must be an integer >= {lo}, got {val!r}")
    if val < lo:
      raise ValueError(f"{q}.{key} must be >= {lo}, got {val}")
  shape = anneal["shape"]
  if shape not in _ANNEAL_SHAPES:
    raise ValueError(
      f"unknown {q}.shape {shape!r}; one of {list(_ANNEAL_SHAPES)}")
  return dict(anneal)


def validate_berhu(berhu, loss_mode, which):
  """
  Validate a berhu knot sub-block and resolve it to {knot, cap, anneal}.

  A standalone pure function (no torch), called by validate_loss to check
  the berhu: sub-block of a loss block. The berhu family's two knots are
  YAML parameters, plus an optional anneal: sub-block (the sqrt -> berhu
  ramp); this fills the defaults, enforces the shape, and rejects a berhu
  sub-block paired with a non-berhu mode (a silent no-op is a config error,
  the trunk-without-trunk_epochs precedent). The enclosing loss block always
  passes a concrete mode, so the check is unconditional.

  Arguments:
    berhu     = the raw berhu sub-block (loss["berhu"]), or None when
                absent.
    loss_mode = the resolved mode of the enclosing loss block.
    which     = "loss" / "trunk.loss" / "head.loss", named in error
                messages (the sub-block is train_args.{which}.berhu).

  Returns:
    the resolved {"knot", "cap", "anneal"} mapping (defaults filled; anneal
    the validated sub-block or None when absent).

  Raises:
    ValueError on: a berhu sub-block with a non-berhu mode; an unknown
    key; a non-positive / non-numeric (bool rejected) knot or cap;
    knot >= cap; or a malformed anneal sub-block (see
    _validate_anneal_block). TypeError if berhu (or its anneal) is present
    but not a mapping.
  """
  is_berhu = loss_mode in ("berhu", "berhu_capped")
  if berhu is None:
    # no sub-block: the defaults (harmless for a non-berhu mode; the knots
    # are built but the specialization never reads them). No anneal.
    out = dict(_BERHU_DEFAULTS)
    out["anneal"] = None
    return out
  if not is_berhu:
    raise ValueError(
      f"train_args.{which} has a berhu: sub-block but mode is "
      f"{loss_mode!r}, not a berhu mode; drop the sub-block or set mode "
      f"to berhu / berhu_capped (a berhu sub-block on a non-berhu mode is "
      f"a silent no-op)")
  if not isinstance(berhu, dict):
    raise TypeError(
      f"train_args.{which}.berhu must be a mapping {{knot, cap}}, got "
      f"{type(berhu).__name__}")
  unknown = set(berhu) - {"knot", "cap", "anneal"}
  if unknown:
    raise ValueError(
      f"unknown train_args.{which}.berhu key(s): {sorted(unknown)}; "
      f"allowed: ['anneal', 'cap', 'knot']")
  out = dict(_BERHU_DEFAULTS)
  for key in ("knot", "cap"):
    if key not in berhu:
      continue
    val = berhu[key]
    # bool is an int subclass, but True/False is never a chi2 knot.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
      raise ValueError(
        f"train_args.{which}.berhu.{key} must be a positive number, got "
        f"{val!r}")
    if val <= 0:
      raise ValueError(
        f"train_args.{which}.berhu.{key} must be > 0, got {val}")
    out[key] = val
  if out["knot"] >= out["cap"]:
    raise ValueError(
      f"train_args.{which}.berhu needs knot < cap, got knot "
      f"{out['knot']}, cap {out['cap']}")
  # the optional anneal: sub-block (presence = on; the sqrt -> berhu ramp).
  # the shared validator names the block train_args.{which}.anneal, so pass
  # the berhu path so it reads train_args.loss.berhu.anneal.
  if "anneal" in berhu:
    out["anneal"] = _validate_anneal_block(berhu["anneal"], f"{which}.berhu")
  else:
    out["anneal"] = None
  return out


# the five loss transforms CosmolikeChi2._reduce implements; validate_loss
# whitelists them at config time, so a typo'd mode fails at startup instead
# of surviving to the first compiled loss call. Order matches the _reduce
# ladder.
_LOSS_MODES = ("chi2", "sqrt", "sqrt_dchi2", "berhu", "berhu_capped")

# the keys a train_args.loss block accepts: the mode string, the berhu
# knot sub-block (valid only beside a berhu mode; see validate_loss), and
# the CMB residual-roughness sub-block (top-level loss only).
_LOSS_KEYS = ("mode", "berhu", "roughness")


def validate_loss(loss, which):
  """
  Validate a train_args.loss block and resolve it to {mode, berhu}.

  A standalone pure function (no torch), beside validate_ema /
  validate_berhu. The loss block localizes every loss option: the mode
  string and, for the berhu family, its {knot, cap} sub-block. Absent (or
  no mode key) resolves to plain "sqrt", the run default, byte-identical to
  a run with no loss block. The berhu sub-block routes through
  validate_berhu against this block's own mode, so "a berhu sub-block on a
  non-berhu mode" is a purely local error (no cross-pass logic). The knot
  sub-block is accepted under the family name berhu: or the exact active
  mode string (a berhu_capped: block under mode berhu_capped); either
  canonicalizes to a single berhu: key. berhu: is the sweep-safe spelling
  (valid across a loss.mode sweep); a mode-named block mismatches once the
  sweep leaves that mode.

  Arguments:
    loss  = the raw loss block (train_args["loss"] or a phase's), or None
            when absent.
    which = "train_args" / "trunk" / "head", naming the block's container
            in error messages (the block itself is {which}.loss).

  Returns:
    a {"mode", "berhu"} mapping: mode one of _LOSS_MODES; berhu the resolved
    {knot, cap} for a berhu mode, or None for a non-berhu mode (which
    carries no knots, the training loop uses the unused defaults).

  Raises:
    TypeError if loss is present but not a mapping. ValueError on an unknown
    key, an unknown mode (naming the five), a mode/knot-block mismatch or
    both spellings of the knot block present, or (via validate_berhu) a
    berhu sub-block paired with a non-berhu mode / a malformed knot pair.
  """
  qual = ("train_args.loss" if which == "train_args"
          else f"train_args.{which}.loss")
  if loss is None:
    return {"mode": "sqrt",
            "berhu": None,
            "roughness": None}
  if not isinstance(loss, dict):
    raise TypeError(
      f"{qual} must be a mapping {{mode, berhu}}, got "
      f"{type(loss).__name__}")
  mode = loss.get("mode", "sqrt")
  # canonicalize the knot-block spelling before the whitelist: it is
  # accepted under the family name berhu: or the exact active mode string,
  # and collapses to a single berhu: key (no input mutation) so the
  # whitelist and validate_berhu see one form. Only "berhu" is a valid
  # spelling under either berhu mode; "berhu_capped" is the mode-string
  # spelling, valid only under mode berhu_capped.
  if mode in ("berhu", "berhu_capped"):
    # a berhu-family mode string that is not the active mode is a wrong-mode
    # block (only "berhu_capped" under mode berhu; "berhu" is the family
    # name, valid under either mode).
    other = "berhu_capped" if mode == "berhu" else None
    if other is not None and other in loss:
      raise ValueError(
        f"{qual} has a {other}: block but mode is {mode!r}; name the knot "
        f"sub-block berhu: (works for every mode; sweep-safe) or set "
        f"mode: {other}")
    if mode != "berhu" and mode in loss:
      # the mode-string spelling (berhu_capped:) -> the family key.
      if "berhu" in loss:
        raise ValueError(
          f"{qual} has both a berhu: and a {mode}: block naming the same "
          f"knot sub-block; keep one (berhu: is the sweep-safe spelling)")
      loss = dict(loss)
      loss["berhu"] = loss.pop(mode)
  unknown = set(loss) - set(_LOSS_KEYS)
  if unknown:
    raise ValueError(
      f"unknown {qual} key(s): {sorted(unknown)}; allowed: "
      f"{list(_LOSS_KEYS)}")
  if mode not in _LOSS_MODES:
    raise ValueError(
      f"unknown {qual}.mode {mode!r}; one of {list(_LOSS_MODES)}")
  # the berhu sub-block is validated against this block's mode (the local
  # check that replaces the deleted cross-pass machinery). validate_berhu
  # names it train_args.{which_berhu}.berhu, so pass "loss" / "trunk.loss".
  which_berhu = "loss" if which == "train_args" else f"{which}.loss"
  resolved = validate_berhu(loss.get("berhu"), mode, which_berhu)
  # a non-berhu mode carries no knots (None -> the training loop's unused
  # defaults); a berhu mode keeps the resolved pair for its knot tensors.
  berhu = resolved if mode in ("berhu", "berhu_capped") else None
  # the CMB residual-roughness sub-block: a per-sample penalty on
  # short-period residual oscillations, added to the per-sample chi2
  # before the shared reduction. Absent = the term does not exist (the
  # off-identity rule; lam 0 is rejected — delete the block instead).
  # Both keys are required when present (never a silent default), and the
  # block rides the TOP-LEVEL loss only (the trunk/head phases are a
  # cosmic-shear head feature; run_emulator applies the term once, to the
  # run's chi2fn).
  rough = loss.get("roughness")
  if rough is not None:
    if which != "train_args":
      raise ValueError(
        f"{qual}.roughness: the roughness term rides the top-level "
        "train_args.loss block only (it configures the run's loss object "
        "once); remove it from the phase block")
    if not isinstance(rough, dict):
      raise TypeError(
        f"{qual}.roughness must be a mapping {{lam, period_cut}}, got "
        f"{type(rough).__name__}")
    unknown_r = set(rough) - {"lam", "period_cut"}
    if unknown_r:
      raise ValueError(
        f"unknown {qual}.roughness key(s): {sorted(unknown_r)}; allowed: "
        "['lam', 'period_cut']")
    for key in ("lam", "period_cut"):
      if key not in rough:
        raise ValueError(
          f"{qual}.roughness needs the {key!r} key (lam = the penalty "
          "weight; period_cut = the penalized-band edge in multipoles); "
          "it is missing (both are stated explicitly, never defaulted)")
      val = rough[key]
      if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(
          f"{qual}.roughness.{key} must be a number, got {val!r}")
    if float(rough["lam"]) <= 0.0:
      raise ValueError(
        f"{qual}.roughness.lam must be > 0; an absent roughness block "
        "states OFF (never lam 0, which would be a silent no-op block)")
    if float(rough["period_cut"]) < 5.0:
      raise ValueError(
        f"{qual}.roughness.period_cut must be >= 5 multipoles, got "
        f"{rough['period_cut']!r}")
    rough = {"lam": float(rough["lam"]),
             "period_cut": float(rough["period_cut"])}
  return {"mode": mode,
          "berhu": berhu,
          "roughness": rough}


def _loss_migration_message(loss_mode, berhu, which):
  """
  Build the paste-ready nested loss: block for the flat-key migration.

  The flat train_args.loss_mode string and the top-level train_args.berhu
  block are gone (the no-alias rule); both nest under one loss: block. This
  renders the offending flat values as a valid loss: sub-block to paste in
  their place, the param_cuts / phase-lr migration precedent.

  Arguments:
    loss_mode = the offending flat loss_mode value, or None when absent.
    berhu     = the offending flat berhu block, or None when absent.
    which     = "train_args" / "trunk" / "head"; the flat keys sit directly
                under it, and the message names it.

  Returns:
    a multi-line message whose body is a paste-ready loss: block carrying
    the old flat values over.
  """
  prefix = ("train_args" if which == "train_args"
            else f"train_args.{which}")
  gone = []
  if loss_mode is not None:
    gone.append(f"{prefix}.loss_mode")
  if berhu is not None:
    gone.append(f"{prefix}.berhu")
  verb  = "is" if len(gone) == 1 else "are"
  it    = "it" if len(gone) == 1 else "them"
  under = "" if which == "train_args" else f" under {which}:"
  lines = [
    f"{' and '.join(gone)} {verb} gone: loss options now nest under a "
    f"single loss: block. Replace {it}{under} with:",
    "",
    "  loss:",
    f"    mode: {loss_mode if loss_mode is not None else 'berhu_capped'}",
  ]
  if isinstance(berhu, dict):
    lines.append("    berhu:")
    for key in ("knot", "cap"):
      if key in berhu:
        lines.append(f"      {key}: {berhu[key]}")
  return "\n".join(lines)


def derive_eval_bs(n_val, target, load):
  """
  The validation batch size, derived from n_val (no user knob).

  The eval pass is pure inference over a fixed n_val, so its batch is a
  performance choice with a computable optimum, not something to expose.
  The free integer is the batch count, not the size: pick
  k = ceil(n_val / target), then equalize the size bs = ceil(n_val / k).
  The fixed-shape tail padding is then k*bs - n_val < k rows (fewer padded
  rows than batches; 0 at n_val = 5000, where bs = 1000 beats both 1024
  and 2048). bs is clamped to `load` on the memmap-streaming path: a batch
  never spans a chunk, so a bs above the chunk size would pad every chunk.

  Arguments:
    n_val  = number of validation rows (>= 1); exact at config time via
             the absolute n_val count.
    target = the batch-size scale to aim for (_EVAL_BS_TARGET); it only
             sets the scale, since the forward + chi2 matmuls saturate
             the GPU by a few hundred rows.
    load   = rows per streamed chunk; the ceiling for bs (a batch stays
             within one chunk).

  Returns:
    the eval batch size (a positive int, <= min(n_val, load)).

  Raises:
    ValueError if n_val < 1 (nothing to evaluate).
  """
  if n_val < 1:
    raise ValueError(
      f"derive_eval_bs needs n_val >= 1 (validation rows), got {n_val}")
  # the free integer is the batch count; equalize the size across it.
  # (a + b - 1) // b is ceil(a / b), the integer-ceil idiom used for the
  # chunk count below.
  k  = (n_val + target - 1) // target    # batches at the ~target scale
  bs = (n_val + k - 1) // k              # equalized batch size
  # a batch never spans a streamed chunk, so cap bs at the chunk size.
  return min(bs, load)


# the keys the optional train_args.ema block accepts: horizon_epochs (the
# weight-averaging window) and an optional anneal: sub-block ramping the
# horizon from 0 to the target (the berhu-anneal twins). The whitelist is
# the typo guard.
_EMA_KEYS = {"horizon_epochs", "anneal"}


def validate_ema(ema, which="train_args"):
  """
  Validate a train_args.ema block and resolve it to {horizon_epochs, anneal}.

  A standalone pure function (no torch), so it is unit-testable in
  isolation. None (the block absent) is the disabled sentinel and the loop
  stays byte-identical; otherwise the block must be a mapping with a positive
  horizon_epochs, plus an optional anneal: sub-block (the same schedule the
  berhu anneal uses, via the shared _validate_anneal_block) that ramps the
  averaging window from 0.

  Arguments:
    ema   = the train_args["ema"] value (or a phase's), or None when absent.
    which = "train_args" / "trunk" / "head", naming the block's container in
            error messages (the block itself is {which}.ema).

  Returns:
    a {"horizon_epochs", "anneal"} mapping (anneal the validated sub-block or
    None), or None when ema was absent (disabled).

  Raises:
    ValueError on an unknown ema key, a missing horizon_epochs, a
    non-positive / non-numeric horizon_epochs (bool rejected), or a malformed
    anneal sub-block. TypeError if ema (or its anneal) is present but not a
    mapping.
  """
  qual = "train_args.ema" if which == "train_args" else f"train_args.{which}.ema"
  if ema is None:
    return None
  if not isinstance(ema, dict):
    raise TypeError(
      f"{qual} must be a mapping ({{horizon_epochs: ...}}), got "
      f"{type(ema).__name__}")
  unknown = set(ema) - _EMA_KEYS
  if unknown:
    raise ValueError(
      f"unknown {qual} key(s): {sorted(unknown)}; allowed: "
      f"{sorted(_EMA_KEYS)}")
  if "horizon_epochs" not in ema:
    raise ValueError(
      f"{qual} needs 'horizon_epochs' (the averaging window in epochs, a "
      f"positive number)")
  h = ema["horizon_epochs"]
  # bool is an int subclass, but True/False is never an epoch count.
  if isinstance(h, bool) or not isinstance(h, (int, float)):
    raise ValueError(
      f"{qual}.horizon_epochs must be a positive number of epochs, got "
      f"{h!r}")
  if h <= 0:
    raise ValueError(
      f"{qual}.horizon_epochs must be > 0, got {h}")
  # the optional anneal: sub-block (presence = on; the horizon ramp). The
  # shared validator names it train_args.{path}.anneal, so pass the ema path.
  which_anneal = "ema" if which == "train_args" else f"{which}.ema"
  anneal = None
  if "anneal" in ema:
    anneal = _validate_anneal_block(ema["anneal"], which_anneal)
  return {"horizon_epochs": h, "anneal": anneal}


def derive_ema_beta(horizon_epochs, steps_per_epoch):
  """
  The per-step ema decay beta from a horizon given in epochs.

  Averaging over `horizon_epochs` epochs means a per-step decay
  beta = 1 - 1/(horizon_epochs * steps_per_epoch): the count of steps in
  the window, not the batch size, sets beta, so the effective window is
  batch-size-invariant (the same reasoning as derive_eval_bs). A tiny run
  whose window holds under one step gets beta 0 (theta_bar tracks theta
  exactly, a harmless no-op), never a negative beta.

  Arguments:
    horizon_epochs  = the averaging window in epochs (> 0).
    steps_per_epoch = optimizer steps per epoch (the loop's real
                      full-batch count).

  Returns:
    beta in [0, 1): the per-step decay for theta_bar.
  """
  denom = horizon_epochs * steps_per_epoch
  if denom < 1:
    return 0.0
  return 1.0 - 1.0 / denom


# --- the finite training/evaluation contract ---
# A NaN or Inf score must NEVER rank, select, or report as a valid
# result. A non-finite per-sample chi2 compares False to every threshold,
# so it counts as BELOW threshold — a diverged model would report a
# perfect frac 0.0 and be snapshotted as the best epoch, defeating the
# dead-network gate discipline itself. So every training/scoring site
# aborts LOUDLY on a non-finite value: never a sentinel, never counted
# below threshold. These helpers give one uniform error across the sites.
def _report_nonfinite(side, quantity, n_bad, n_total, positions):
  """Raise the finite-contract error, one message shape for every site.

  Arguments:
    side      = "training" or "validation" (the pipeline side at fault).
    quantity  = what went non-finite (e.g. "per-sample chi2").
    n_bad     = how many entries are non-finite.
    n_total   = the total checked.
    positions = the first few offending positions (validation row
                indices, or a batch identity), already a Python list.

  Raises:
    ValueError, always — the caller only calls this once it has found a
    non-finite value.
  """
  raise ValueError(
    "finite contract [" + side + "]: " + str(int(n_bad)) + " of "
    + str(int(n_total)) + " " + quantity + " are non-finite (NaN/Inf) "
    "— a diverged run. First offending positions: " + str(positions)
    + ". A non-finite score must never rank or select a model; fix the "
    "run, never score it (no sentinel, never counted below threshold).")


def _global_grad_norm(params):
  """The L2 norm of all parameter gradients, READ-ONLY (None grads
  skipped, the frozen trunk in a head phase).

  The finite contract's gradient check when clipping is off: clip is 0
  so clip_grad_norm_ is not called, but the step must still be refused on
  a NaN/Inf gradient. This computes the same global norm clip_grad_norm_
  would, WITHOUT scaling the gradients, so a clipping-off run stays
  byte-identical to the pre-contract path (only the finite check is
  added).

  Arguments:
    params = the model parameters (an iterator; each may carry .grad).

  Returns:
    a 0-dim tensor: the global gradient L2 norm (0 when no grads exist).
  """
  parts = []
  for p in params:
    if p.grad is not None:
      parts.append(torch.linalg.vector_norm(p.grad.detach()))
  if not parts:
    return torch.zeros(())
  return torch.linalg.vector_norm(torch.stack(parts))


def ordinary_median(values):
  """The ordinary 50th-percentile median (unit 60).

  The center value for odd N, the arithmetic MEAN of the two center values
  for even N. It is the standard estimator every prose, plot, history, and gate
  surface already names "median". torch.median returns the LOWER of the two
  central ordered values for even N (a lower-median), which biases the
  plateau-scheduler feed, the equal-fraction best-epoch tie-break, and the
  persisted / plotted history low; torch.quantile(., 0.5) is the ordinary
  median on both parities and is byte-identical to torch.median for odd N.
  Computed in float64 so an extreme-scale even-N midpoint cannot overflow
  (the reduction unit 14(f) hardens); the values already live on the CPU.

  This is the ONE shared median reduction: eval_val, the scheduler feed, the
  tie-break, the saved histories, and the five gate reference sites all use
  it, so repaired production code and the gates cannot disagree.

  torch.quantile caps its input at about 2^24 elements; every n_val here
  (board runs 200, the shipped placeholder 5000) sits far below that, and a
  larger validation set raises from torch.quantile loudly rather than
  rounding silently.

  Arguments:
    values = a 1-D tensor of per-sample values (the validation chi2).

  Returns:
    a Python float: the ordinary median.
  """
  v = values.reshape(-1).to(torch.float64)
  return float(torch.quantile(v, 0.5))


def _validate_published_reductions(mean, median, frac):
  """Refuse a non-finite PUBLISHED reduction (unit 14(f),, clause 4).

  eval_val's row guard validates the per-sample chi2, but the reductions it
  publishes (the mean, the median, the threshold fractions) are appended to
  histories, plotted, persisted, and stepped into the plateau scheduler. A
  reduction that is non-finite (a mean that overflowed, a NaN that slipped in)
  must be a REFUSED evaluation, not a sentinel or a big number that silently
  ranks or reschedules. The mean is the vulnerable reduction (float32 overflow
  before the float64 fix); the median (order statistic) and the fractions
  (bounded by 1) cannot overflow, but the one-line check covers all three.

  Arguments:
    mean   = the published mean (a Python float).
    median = the published median (a Python float).
    frac   = the published per-threshold fractions (a tensor).

  Raises:
    ValueError naming the reduction, the side ("validation"), and the value
    when any published reduction is non-finite.
  """
  for name, value in (("mean", mean), ("median", median)):
    if not math.isfinite(value):
      raise ValueError(
        "published reduction [validation]: the " + name + " is "
        + repr(value) + ", not finite. A reduction that overflowed or went "
        "NaN must not rank, reschedule, or ship a model. Fix the run "
        "(the per-sample rows passed the domain guard; the reduction "
        "itself is out of range).")
  if not bool(torch.isfinite(frac).all()):
    raise ValueError(
      "published reduction [validation]: a threshold fraction is not "
      "finite (" + repr(frac.tolist()) + "); a fraction is bounded in "
      "[0, 1], so a non-finite value signals a corrupted reduction.")


def eval_val(model, lossfn, data, load, bs, thresholds,
             fwd_chi2=None):
  """
  Evaluate the model on the validation set.

  Streams the val rows in chunks of `load`, runs the model in
  fixed bs-sized batches, and reduces every batch to its
  per-sample chi2 immediately (consume, don't stash):

    xb, yb  (bs, ...)         one padded val batch
       │  fwd_chi2            model forward + per-sample chi2,
       ▼                      one compiled graph (or eager)
    c_b (bs,)                 tiny; pad rows sliced off, cloned
       │  ... all batches, all chunks ...
       ▼
    c  (n_val,)               concatenated on the compute device
       │  .cpu()              moves to CPU only when c is on an accelerator
       ▼
    median / mean / frac      (median and the threshold fractions
                              need the full distribution at once,
                              so they reduce here, not per chunk)

  (legend: bs = model batch size; xb / yb = one padded batch's
  inputs / targets; c_b = the batch's per-sample chi2, shape (bs,);
  n_val = number of validation points; c = per-sample chi2 over the
  whole val set, shape (n_val,); frac = fraction over each
  threshold.)

  The previous form stashed every batch's full prediction (with a
  per-batch clone of (bs, out_dim); reduce-overhead reuses its
  output buffer) and concatenated them all before one big chi2:
  for the factored heads that meant ~1 GB of VRAM churn per epoch.
  Reducing per batch keeps only (bs,) scalars alive (the clone
  survives but copies bs floats, not bs x out_dim) and the
  accelerator-to-CPU traffic stays at one small transfer per eval,
  which matters when the GPU hangs off a bandwidth- and
  latency-limited link (an eGPU over Thunderbolt).

  The model runs in batches of exactly `bs` (the final partial
  batch padded up to bs with copies of row 0, pad chi2 sliced off
  after), so a torch.compile'd graph sees one static input shape
  and never recompiles. Padding, not dropping, because evaluation
  must score every val point; the pad rows' chi2 values are real
  (a duplicated valid row) and discarded.

  Arguments:
    model      = the network, in eval mode.
    lossfn     = CosmolikeChi2; .chi2 gives the per-sample chi2 of
                 a prediction against its target.
    data       = dict with load_C, load_dv (the loaders) and idx.
                 idx contains coordinates into the active validation
                 source and is read into the local vidx.
    load       = rows per streamed chunk.
    bs         = model batch size (the derived eval batch, see
                 derive_eval_bs; independent of the training bs), so
                 the compiled graph sees one fixed input shape.
    thresholds = 1D tensor of delta-chi2 cutoffs; the returned
                 fraction counts val points above each.
    fwd_chi2   = optional (xb, yb) -> (bs,) per-sample-chi2
                 callable; training_loop_batched passes its
                 compiled twin (one compile per phase, shared by
                 the baseline and every epoch). None -> the same
                 math built here, eager (direct callers).

  Returns:
    median = median per-sample chi2 over the val set.
    mean   = mean per-sample chi2 over the val set.
    frac   = 1D tensor (len = #thresholds): fraction of val
             points with chi2 above each threshold.
  """
  load_C = data["load_C"]
  load_dv = data["load_dv"]
  vidx = data["idx"]

  if fwd_chi2 is None:
    # eager fallback: the same math the compiled twin traces.
    needs_p = getattr(lossfn, "needs_params", False)
    def fwd_chi2(xb, yb):
      pred = model(xb)
      if needs_p:
        return lossfn.chi2(pred=pred, target=yb,
                           params_whitened=xb)
      return lossfn.chi2(pred=pred, target=yb)

  chi2s = []
  # Active validation-source coordinates in c's order. They are local to a
  # compact RAM source and original dump rows for a disk-backed source.
  order_rows = []
  with torch.no_grad():
    for cs in range(0, len(vidx), load):
      rows = np.sort(vidx[cs:cs+load])
      order_rows.append(rows)
      Cc  = load_C(rows)             # (m, Ncosmo)
      dvc = load_dv(rows)            # (m, out_dim)
      m   = Cc.shape[0]
      for s in range(0, m, bs):
        xb = Cc[s:s+bs]
        yb = dvc[s:s+bs]
        n  = xb.shape[0]             # real rows this batch
        if n < bs:
          # pad the final short batch up to bs, keeping the
          # compiled graph on one fixed shape. [:1] is the chunk's
          # first row; .expand(bs-n, -1) stretches its size-1 row
          # axis to bs-n copies (-1 = keep the column axis) as a
          # stride-0 view, no copy. Both inputs need the padding
          # now (the chi2 runs per batch); the pad chi2 is sliced
          # off ([:n]) below.
          xb = torch.cat([xb, Cc[:1].expand(bs - n, -1)], dim=0)
          yb = torch.cat([yb, dvc[:1].expand(bs - n, -1)], dim=0)
        # clone: under reduce-overhead (CUDA graphs) the compiled
        # graph reuses a static output buffer per call, so the
        # next call overwrites this result before the cat, but
        # the stash is now (bs,) floats, not (bs, out_dim).
        chi2s.append(fwd_chi2(xb, yb)[:n].clone())

  # torch.cat allocates the concatenated score tensor on its current device.
  # cpu() then transfers it only when that tensor is on an accelerator. For
  # an already-CPU tensor, cpu() normally returns CPU storage without a move.
  c = torch.cat(chi2s).cpu()
  # chi2-domain contract (the single validation chokepoint: the baseline,
  # the raw eval, and the ema eval all pass through here, so best-epoch
  # selection only ever compares valid scores). The shared score-domain
  # boundary raises on a non-finite OR materially negative per-sample chi2
  # (a negative compares False to every positive threshold and would rank a
  # corrupted model as PERFECT), naming the offending validation
  # rows; within-band roundoff negatives normalize to exact 0, the same rule
  # the training reduction folds to NaN. c is already the compute-dtype tensor
  # (no .double() before this), so the band matches the accumulated roundoff.
  c = screen_chi2(c, loss=lossfn, label="validation",
                  positions=np.concatenate(order_rows))
  # published reductions in float64 (unit 14(f)): a float32 mean of
  # rows near the float32 max overflows to Inf AFTER the row guard passed
  # (the sum exceeds float32 range in any order), so the mean is formed in
  # float64. The median is the ordinary 50th-percentile estimator (unit 60,
  #). torch.median's lower-middle sample for even n_val biased the
  # plateau-scheduler feed, the equal-fraction tie-break, and the persisted
  # history; ordinary_median (float64, torch.quantile 0.5) fixes all four at
  # once because they all consume this returned median.
  mean   = c.to(torch.float64).mean().item()
  median = ordinary_median(c)

  # c[:, None] (Nval, 1) and thresholds[None, :] (1, T) broadcast
  # into a (Nval, T) boolean grid: entry [i, j] = "is point i's
  # chi2 above threshold j?". mean(0) over samples -> the fraction
  # past each threshold. ([:, None] is the numpy/torch unsqueeze.)
  frac = (c[:, None] > thresholds[None, :]).float().mean(0)
  # clause 4 (unit 14(f)): every PUBLISHED reduction must be finite before it
  # is returned, appended to histories, plotted, and stepped into the
  # scheduler. An infinite mean is a REFUSED evaluation, never a sentinel.
  _validate_published_reductions(mean=mean, median=median, frac=frac)
  return median, mean, frac


def eval_source_chi2(model,
                     param_geometry,
                     chi2fn,
                     source,
                     device,
                     bs):
  """
  Per-cosmology delta-chi2 of the emulator over one source.

  Scores every row named by ``source["idx"]`` in the active source
  coordinate system. It encodes the parameters into model inputs, predicts
  the whitened data vector,
  and evaluates the full masked chi2 against the encoded truth.
  Returns plain numpy arrays aligned row-for-row, ready for a
  parameter-space plot.

  Works for plain CosmolikeChi2 and RescaledChi2: the rescaled
  geometry needs the params (to build R), which are the whitened
  inputs X, passed to encode and chi2 when chi2fn rescales.

  Arguments:
    model          = trained network; set to eval mode here.
    param_geometry = ParamGeometry; .encode whitens the raw
                     parameters into the model inputs.
    chi2fn         = CosmolikeChi2 or RescaledChi2.
    source         = source dict with "C", "dv", "idx".
    device         = device the model lives on.
    bs             = rows per forward batch (bounds memory).

  Returns:
    params = (N, n_param) float64 raw parameters of the rows.
    dchi2  = (N,) float64 per-row delta-chi2, same row order.
  """
  model.eval()
  rows   = np.sort(source["idx"])
  params = np.asarray(source["C"][rows], dtype="float64")

  with torch.no_grad():
    # whitened model inputs for these rows
    X = param_geometry.encode(
      torch.from_numpy(params).float().to(device))
    dv = torch.from_numpy(
      source["dv"][rows]).float().to(device)
    # rescaled geometry needs the params to build R; the plain
    # one does not.
    if getattr(chi2fn, "needs_params", False):
      T = chi2fn.encode(dv=dv, params_whitened=X)
    else:
      T = chi2fn.encode(dv)

    chunks = []
    for s in range(0, X.shape[0], bs):
      pred = model(X[s:s + bs])
      if getattr(chi2fn, "needs_params", False):
        c = chi2fn.chi2(pred=pred, target=T[s:s + bs], params_whitened=X[s:s + bs])
      else:
        c = chi2fn.chi2(pred=pred, target=T[s:s + bs])
      chunks.append(c.cpu())

  c_compute = torch.cat(chunks)          # per-row chi2 in the COMPUTE dtype
  # chi2-domain contract: a diverged model's
  # non-finite OR materially negative per-row chi2 must not be published as a
  # diagnostic metric (a silent NaN, or a negative delta-chi2, in the
  # parameter-space plots). The shared score-domain boundary raises here (it
  # folds to NaN in training), naming the offending source rows; within-band
  # roundoff negatives normalize to exact 0. The band derives from the dtype
  # the chi2 was COMPUTED in (normally float32), NEVER a storage upcast: an
  # earlier .double() here relabelled the dtype to float64 and floored the
  # band to 1e-6, so this diagnostic scorer REFUSED a roundoff negative that
  # _reduce / eval_val (the float32 band, ~3e-3 at w = 780) normalize to exact
  # 0: one score, two verdicts. So screen in the compute dtype; the ACCEPTED
  # result is cast to float64 for reporting only.
  c_norm = screen_chi2(c_compute, loss=chi2fn, label="diagnostic",
                       positions=rows)
  dchi2 = c_norm.double().numpy()        # accepted result -> float64 to report
  return params, dchi2


def training_loop_batched(nepochs,
                          optimizer,
                          scheduler,
                          model,
                          bs,
                          lossfn,
                          mode,
                          data,
                          trim_opts,
                          focus_opts,
                          thresholds,
                          warmup_epochs=0,
                          silent=False,
                          use_amp=False,
                          clip=0.0,
                          rewind=False,
                          ema=None,
                          berhu=None,
                          anchor=None,
                          amp_dtype=torch.bfloat16,
                          scaler_policy="unscaled"):
  """
  Train the emulator, with a validation pass per epoch.

  Each epoch reshuffles the training rows, streams them in
  chunks through the loaders, and steps the optimizer one
  minibatch at a time. After each epoch it evaluates on the val
  set and steps the scheduler on the val median (after an optional
  linear lr warmup over the first warmup_epochs epochs). Data
  placement (resident or streamed) is hidden behind the loaders,
  so this loop is identical in every regime.

  Arguments:
    nepochs    = number of passes over the training set.
    optimizer  = the optimizer (e.g. Adam).
    scheduler  = LR scheduler stepped on the val median each
                 epoch (e.g. ReduceLROnPlateau).
    model      = the network (possibly torch.compile'd).
    bs         = minibatch size.
    lossfn     = CosmolikeChi2; .loss(pred, target, mode) is
                 the training loss, .chi2 the eval metric.
    mode       = loss mode passed to .loss ("sqrt", "chi2", ...).
    data       = nested source dict: data["train"] and data["val"],
                 each with load_C, load_dv, idx, and load. idx contains
                 active source coordinates, which are compact-RAM local
                 rows or disk-backed original rows. load is the number of
                 rows per streamed chunk. The val sub-dict is handed to
                 eval_val each epoch.
    trim_opts  = trim schedule (see anneal_value): "start"/"end"
                 trim fractions, "hold_epochs"/"anneal_epochs",
                 "shape". None -> hold 5% then cosine-anneal to 0.
    focus_opts = focal-weight schedule (see anneal_value): the
                 per-epoch focus exponent gamma (0 = uniform,
                 higher = harder points weighted more), via
                 "start"/"end", "hold_epochs"/"anneal_epochs",
                 "shape", plus "kappa" (the chi2 scale where the
                 focal weight turns on, fixed over the run, read as
                 focus_scale; default 1.0). None -> no focal
                 weighting (gamma = 0).
    thresholds = delta-chi2 cutoffs for the val fractions.
    amp_dtype  = resolved autocast dtype selected by run_emulator.
    scaler_policy = resolved gradient-scaling policy selected by
                    run_emulator. "unscaled" preserves the current
                    plain backward and optimizer step.
    warmup_epochs = epochs of linear lr ramp before the plateau
                    scheduler takes over (0 = none).
    silent     = if True, suppress all per-epoch and summary
                 prints; metrics and returns are unchanged.
    use_amp    = if True, run the forward in bfloat16 autocast;
                 the loss stays in float32/64.
    clip       = gradient-norm ceiling per optimizer step (0 =
                 off, the default). Each step, the norm of the
                 full gradient vector (all trainable parameters
                 together) is measured; if it exceeds clip, every
                 gradient is rescaled by clip/norm (same
                 direction, bounded size). Kills the single-batch
                 kick a monster-outlier batch produces under a
                 quadratic loss, regardless of loss mode.
    rewind     = if True, whenever the plateau scheduler cuts the
                 lr, reload the best-so-far weights and the
                 optimizer state snapshotted with them, then keep
                 the new (reduced) lr. An excursion into a bad
                 basin then costs at most `patience` epochs: the
                 median stalls, the scheduler fires, and the run
                 resumes from its best point at a lower lr,
                 instead of decaying the lr inside the wreckage.
                 Applies only to ReduceLROnPlateau (an epoch
                 scheduler like CosineAnnealingLR changes the lr
                 every epoch; rewinding on each change would pin
                 the run to its best forever).
    ema        = optional validated ema block ({horizon_epochs, anneal})
                 or None (off = a byte-identical loop). When set, a
                 Polyak weight average theta_bar is kept from the live
                 point: updated after every optimizer.step at
                 beta = derive_ema_beta(horizon_epochs,
                 steps_per_epoch), evaluated per epoch by swapping it
                 into the model in place, and coupled to the best
                 snapshot as one unit {theta, optimizer, theta_bar}
                 (saved on best, restored on rewind), so anything the
                 rewind un-lives leaves the average too. Selection and
                 the printed metrics then use the average; the plateau
                 scheduler stays on the raw median (dynamics
                 unchanged). The returned model carries the best
                 average. With an anneal sub-block {hold_epochs,
                 anneal_epochs, shape} the horizon ramps from 0:
                 h(e) = horizon_epochs * s(e), beta(e) an eager per-epoch
                 float (the lerp is uncompiled), and the live point moves
                 to max(warmup end, first s > 0); absent = the constant
                 beta from warmup end (byte-identical).
    berhu      = the resolved {knot, cap, anneal} for the berhu /
                 berhu_capped modes (run_emulator resolved it from the
                 pass's loss block via validate_loss); None -> the defaults
                 (a non-berhu mode carries none). knot / cap feed the 0-dim
                 tensors the loss reads (non-berhu modes ignore them). A
                 present anneal sub-block {hold_epochs, anneal_epochs,
                 shape} enables the sqrt -> berhu blend: a 0-dim s_t tensor
                 filled in place per epoch from anneal_value (start 0,
                 end 1) and passed as berhu_s; absent -> no blend (the
                 branch is byte-identical to before this feature).

  Returns:
    train_losses, medians, means, fracs = per-epoch lists
      (fracs holds one fraction tensor per epoch).
  """
  # loaders: active source coordinates -> ready-to-train parameter inputs
  # and targets on the device. The regime hides where the source lives.
  load_C  = data["train"]["load_C"]
  load_dv = data["train"]["load_dv"]
  tidx    = data["train"]["idx"]   # active training-source coordinates
  # device the model lives on; place new tensors here too.
  # model.parameters() is an iterator, so next(...), not [0].
  device  = next(model.parameters()).device
  ntrain  = len(tidx)              # training rows per epoch
  load    = data["train"]["load"]  # rows per streamed chunk
  # chunks per epoch = ceil(ntrain / load); + load - 1 rounds the
  # integer division up.
  nchunks = (ntrain + load - 1) // load

  if scaler_policy != "unscaled":
    raise ValueError(
      "training_loop_batched supports scaler_policy 'unscaled'; got "
      + repr(scaler_policy))

  if not silent:
    print(f"{load} rows/chunk, {nchunks} chunks/epoch, "
          f"amp={use_amp}, loss mode = {mode}")

  # ema (weight averaging) state, all gated on an ema block being set:
  # when ema is None every ema branch below is skipped and the loop is
  # byte-identical to before the feature. validate_ema already ran in
  # run_emulator, so a non-None block is well-formed here.
  ema_on         = ema is not None
  theta_bar      = None    # the running average; allocated at the live point
  ema_tmp        = None    # param-sized scratch for the in-place eval swap
  best_theta_bar = None    # the average at the best epoch (the shipped one)
  best_is_ema    = False   # does the current best snapshot carry theta_bar?
  ema_s_opts     = None    # the horizon-anneal schedule (None = no anneal)
  if ema_on:
    # steps_per_epoch = the loop's real full-batch count (whole batches
    # per chunk, summed over chunks), so the horizon is counted in steps
    # and beta comes out batch-size-invariant.
    steps_per_epoch = 0
    for cs in range(0, ntrain, load):
      chunk = min(load, ntrain - cs)
      steps_per_epoch += chunk // bs
    horizon = ema["horizon_epochs"]
    # the target beta (the full-horizon value, s = 1): the no-anneal run's
    # constant beta and the anneal ramp's endpoint.
    target_beta = derive_ema_beta(horizon_epochs=horizon,
                                  steps_per_epoch=steps_per_epoch)
    # optional horizon anneal (ema.anneal): schedule the horizon, not a
    # blend. h(e) = horizon * s(e), beta(e) = derive_ema_beta(h(e), ...);
    # s = 0 gives beta = 0 (the existing denom<1 clamp -> theta_bar tracks
    # theta, no memory of the terrible early era), growing to the target as s
    # ramps. ema_s_opts feeds the shared anneal_value helper (start 0/end 1).
    ema_anneal = ema.get("anneal")
    if ema_anneal is not None:
      ema_s_opts = {"shape":         ema_anneal["shape"],
                    "start":         0.0,
                    "end":           1.0,
                    "hold_epochs":   ema_anneal["hold_epochs"],
                    "anneal_epochs": ema_anneal["anneal_epochs"]}
    # beta this epoch. Note: beta is a per-epoch Python float, deliberately;
    # the EMA lerp below is eager (theta_bar is a private buffer, never
    # graph-captured), unlike trim_t / focus_t / s_t which feed the compiled
    # step. Do not turn beta into a device tensor, and do not copy this float
    # pattern into any compiled-side schedule. With no anneal it is the target
    # every epoch (the one-time constant, byte-identical); with anneal it is
    # recomputed per epoch in the loop below.
    beta = target_beta
    # the live parameter tensors (reached through any torch.compile
    # wrapper). The in-place foreach ops below mutate their storage, the
    # storage the compiled graph captured, so replay stays valid; never
    # rebind a parameter's .data or the list entry.
    ema_params = list(model.parameters())
    if not silent:
      # the no-anneal banner is byte-identical; the anneal banner marks the
      # target as the ramp endpoint and names the schedule.
      if ema_anneal is None:
        beta_note = f"beta {target_beta:.6f}"
      else:
        beta_note = (f"beta -> {target_beta:.6f}; anneal: hold "
                     f"{ema_anneal['hold_epochs']} + "
                     f"{ema_anneal['anneal_epochs']} {ema_anneal['shape']}")
      print(f"ema: horizon {horizon} epochs ({beta_note}; selection + "
            f"metrics on the average, scheduler on the raw median)")

  train_losses, medians, means, fracs = [], [], [], []

  # target lr per param group, captured before warmup ramps it;
  # warmup scales each group up to its own base.
  base_lrs = []
  for g in optimizer.param_groups:
    base_lrs.append(g["lr"])

  # kappa = chi2 scale where the focal weight turns on (fixed over
  # the run, unlike the annealed gamma); from focus_opts, default
  # 1.0. Feeds the loss as focus_scale.
  kappa = focus_opts.get("kappa", 1.0)

  # ---- the combined forward+loss step (the launch-bound fix) ----
  # A tiny model's step is dozens of micro-kernels, each launched
  # by the CPU; the GPU spends the step waiting on those launches
  # (and on any CPU contention). Compiling the model alone (as
  # make_model does) collapses its launches but leaves the loss,
  # a dozen kernels forward, more backward, plus its per-step
  # Python, eager. Tracing model + loss together collapses the
  # whole step to a few graph replays:
  #
  #    xb, yb  (contiguous batch views, see the pre-shuffle below)
  #       │  model(xb)          under autocast, as before
  #       ▼
  #    pred
  #       │  lossfn.loss(...)   static-shape reduction (_reduce)
  #       ▼
  #    scalar loss              one compiled graph for all of it
  #
  # Off-CUDA (or compile disabled) fwd_loss is this same function,
  # simply not compiled, one code path, two execution modes.
  needs_p = getattr(lossfn, "needs_params", False)

  # kappa as a 0-dim device tensor, like trim_t / focus_t below: a
  # Python float in the traced closure is torch-version-dependent
  #, some versions specialize it to a constant, others lift it as
  # an unspecialized float backed by a 0-dim CPU tensor input,
  # which silently disables CUDA-graph replay ("skipping cudagraphs
  # due to cpu device (primals_N)"). A device tensor is graph-safe
  # everywhere; kappa is fixed per pass, so it is created once, not
  # filled per epoch.
  kappa_t = torch.as_tensor(float(kappa), device=device)
  # the berhu family knots (mode "berhu" / "berhu_capped"), resolved from
  # the pass's loss block (run_emulator passed the validate_loss'd
  # {knot, cap}; None = the defaults). Built once per pass as 0-dim device
  # tensors, the same graph-safe discipline as kappa_t (a Python float in
  # the traced closure can surface as a CPU input and kill CUDA-graph
  # replay); the non-berhu modes never read them, so inductor prunes them.
  berhu_r = _BERHU_DEFAULTS if berhu is None else berhu
  knot_t = torch.as_tensor(float(berhu_r["knot"]), device=device)
  cap_t  = torch.as_tensor(float(berhu_r["cap"]), device=device)
  # the berhu anneal schedule (loss.berhu.anneal), if present: blend the
  # loss from plain sqrt (s = 0) into the full berhu form (s = 1) over a
  # trim-style schedule, so the escalated (knot, cap) window votes arrive
  # late (few points still pass down through the window by then). s_t is a
  # 0-dim device tensor filled in place per epoch (the trim_t / focus_t
  # pattern), passed to _fwd_loss as berhu_s; None when the anneal block is
  # absent, so the blend ops never enter the compiled graph (static
  # specialization -> byte-identical to a run without the block). start 0 /
  # end 1 feed the existing anneal_value helper (no new schedule code).
  berhu_anneal = berhu_r.get("anneal")
  s_t    = None
  s_opts = None
  if berhu_anneal is not None:
    s_t = torch.zeros((), device=device)
    s_opts = {"shape":         berhu_anneal["shape"],
              "start":         0.0,
              "end":           1.0,
              "hold_epochs":   berhu_anneal["hold_epochs"],
              "anneal_epochs": berhu_anneal["anneal_epochs"]}

  def _fwd_loss(xb, yb, trim, focus, berhu_s):
    # the model forward under autocast (unchanged semantics); the
    # loss math stays outside it, in full precision.
    with torch.autocast(device.type,
                        dtype=amp_dtype,
                        enabled=use_amp):
      pred = model(xb)
    if needs_p:
      return lossfn.loss(pred=pred, target=yb,
                         params_whitened=xb, mode=mode,
                         trim=trim, focus=focus,
                         focus_scale=kappa_t, berhu_knot=knot_t,
                         berhu_cap=cap_t, berhu_s=berhu_s)
    return lossfn.loss(pred, target=yb, mode=mode, trim=trim,
                       focus=focus, focus_scale=kappa_t,
                       berhu_knot=knot_t, berhu_cap=cap_t,
                       berhu_s=berhu_s)

  # the eval twin: model forward + per-sample chi2 in one compiled
  # graph, handed to eval_val (same launch-bound argument, and it
  # lets eval reduce each batch immediately instead of stashing
  # every full prediction, see eval_val's docstring).
  def _fwd_chi2(xb, yb):
    pred = model(xb)
    if needs_p:
      return lossfn.chi2(pred=pred, target=yb,
                         params_whitened=xb)
    return lossfn.chi2(pred=pred, target=yb)

  # compile both with the same mode make_model used for the model
  # (the attribute is absent off-CUDA or with compile disabled).
  cmode = getattr(model, "emul_compile_mode", None)
  if cmode is not None:
    fwd_loss = torch.compile(_fwd_loss, mode=cmode)
    fwd_chi2 = torch.compile(_fwd_chi2, mode=cmode)
  else:
    fwd_loss = _fwd_loss
    fwd_chi2 = _fwd_chi2

  # the annealed per-epoch loss scalars, as 0-dim device tensors:
  # a compiled function guards on a Python float by value, so an
  # annealing trim / focus passed as floats would recompile every
  # epoch; a tensor updates in place (fill_ below) with no
  # recompile, and keeps the compiled graph's shapes static (the
  # loss's _reduce is built for tensor scalars).
  trim_t  = torch.zeros((), device=device)
  focus_t = torch.zeros((), device=device)

  # track the best epoch by the inference metric, the fraction
  # of val points with chi2 > the first threshold (0.2), to keep
  # the best model, not the last. Seeded by a baseline eval of the
  # incoming weights (epoch 0, before any training), so a pass can
  # never end worse than it started: ordinarily the baseline is a
  # random init and is overtaken immediately, but at the two-phase
  # handoff the incoming model is phase 1's best (the zero-init
  # head makes them identical), and this seed guarantees phase 2
  # returns at least that even if its first epochs wander.
  # derive_eval_bs (above): run the validation pass at a batch derived
  # from n_val (~_EVAL_BS_TARGET rows), decoupled from the training bs, so
  # a small training bs does not multiply the eval's launch count.
  # Computed once here (n_val is exact) and reused by the baseline and
  # every epoch, so the compiled fwd_chi2 twin keeps one static shape.
  eval_bs = derive_eval_bs(n_val=len(data["val"]["idx"]),
                           target=_EVAL_BS_TARGET,
                           load=load)
  # stash the derived eval batch so resolved_train (save schema v2) records
  # the real value, not a re-derivation (data is run_emulator's loaders dict).
  data["eval_bs"] = eval_bs
  model.eval()
  b_median, b_mean, b_frac = eval_val(model=model,
                                      lossfn=lossfn,
                                      data=data["val"],
                                      load=load,
                                      bs=eval_bs,
                                      thresholds=thresholds,
                                      fwd_chi2=fwd_chi2)
  best_frac   = b_frac[0].item()
  best_median = b_median
  best_epoch  = 0
  # snapshot the incoming weights (clone: state_dict returns live
  # references that training would overwrite).
  best_state = {}
  for k, v in model.state_dict().items():
    best_state[k] = v.detach().clone()
  # rewind needs the optimizer state that belongs to the best
  # weights (Adam's moments track a trajectory; moments from a bad
  # basin would kick the restored weights right back out). deepcopy:
  # state_dict() returns live tensor references. At this baseline
  # the optimizer is fresh, so the snapshot is the empty state,
  # restoring it simply resets the moments.
  best_opt_state = None
  if rewind:
    best_opt_state = copy.deepcopy(optimizer.state_dict())
  if not silent:
    print(f"epoch   0  baseline (no training yet): "
          f"val {b_mean:.4f}  med {b_median:.4f}"
          f"  frac>0.2 {best_frac:.4f}")

  # wall-clock timing for GPU comparison. eval_val's .item() below
  # syncs the GPU before each epoch's log line, so perf_counter around
  # the epoch measures real elapsed time. epoch 1 carries the one-time
  # torch.compile warmup, reported apart from the steady-state rate.
  t_run   = time.perf_counter()
  t_first = 0.0

  for epoch in range(1, nepochs + 1):
    t_epoch = time.perf_counter()
    # warmup before this epoch trains: epoch e (of W) runs at
    # base*e/W, so epoch 1 uses base/W, protecting exactly the
    # steps warmup exists for. (It used to be applied after the
    # epoch: epoch 1 of every pass then trained at the full base lr
    # while printing the ramped value it had just set for epoch 2,
    # at a two-phase handoff that full-strength first epoch could
    # wreck the identity start.)
    if epoch <= warmup_epochs:
      scale = epoch / warmup_epochs
      for grp, base in zip(optimizer.param_groups, base_lrs):
        grp["lr"] = base * scale
    # initialize the weight average at the live point: the later of warmup
    # end (the high-lr ramp is not worth averaging) and the first epoch the
    # horizon anneal leaves the hold (s > 0, so the average never starts in
    # the dead-memory hold). No anneal -> the warmup-end floor alone,
    # unchanged. The first post-warmup live epoch clones theta into
    # theta_bar + the scratch buffers, then the per-step update starts;
    # before this, ema_on runs the loop as usual.
    if (ema_on and theta_bar is None and epoch > warmup_epochs
        and (ema_s_opts is None
             or anneal_value(epoch=epoch, opts=ema_s_opts) > 0.0)):
      theta_bar      = []
      ema_tmp        = []
      best_theta_bar = []
      for p in ema_params:
        theta_bar.append(p.detach().clone())
        ema_tmp.append(p.detach().clone())
        best_theta_bar.append(p.detach().clone())
    model.train()
    perm = tidx[torch.randperm(ntrain).numpy()]

    # this epoch's annealed trim fraction (large early, 0 late).
    # One value per epoch, shared by all its batches.
    rob = anneal_value(epoch=epoch, opts=trim_opts)

    # this epoch's focal weight exponent gamma (annealed): 0 early
    # -> uniform weighting (a plain mean, stable while the bulk is
    # still being learned), rising to focus_opts["end"] late ->
    # up-weight the hard points so the optimizer keeps chasing the
    # tail, not out-voted by the solved bulk. One per epoch.
    focus = anneal_value(epoch=epoch, opts=focus_opts)

    # write this epoch's annealed values into the 0-dim tensors the
    # compiled fwd_loss reads (in-place: no recompile, no realloc).
    trim_t.fill_(rob)
    focus_t.fill_(focus)
    # the berhu anneal blend factor s (0 -> plain sqrt, 1 -> full berhu),
    # on the same in-place per-epoch schedule; only when the anneal block is
    # present (s_t stays None otherwise, so the blend is absent from the
    # compiled graph). The schedule restarts at this pass's own epoch 1.
    if s_t is not None:
      s_t.fill_(anneal_value(epoch=epoch, opts=s_opts))
    # the ema horizon anneal: recompute beta from this epoch's scheduled
    # horizon h(e) = horizon * s(e) (an eager Python float; see the ema setup
    # note above). Only when ema.anneal is present; without it beta stays the
    # one-time target (byte-identical). The lerp reads this beta below.
    if ema_s_opts is not None:
      s_ema = anneal_value(epoch=epoch, opts=ema_s_opts)
      beta = derive_ema_beta(horizon_epochs=horizon * s_ema,
                             steps_per_epoch=steps_per_epoch)

    # epoch training loss, accumulated on-device
    # accumulate the epoch loss on the HOST as a python float
    # (float64 on every backend, MPS included), not a device float32 sum.
    # A finite per-batch loss near the float32 max, times bs, overflows a
    # float32 product to Inf before it reaches the accumulator (and on MPS
    # the accumulator itself is float32). The finite contract already syncs
    # every step at the isfinite(loss) check below, so float(loss) adds no
    # new stall; the accumulator is diagnostic-only (selection reads the val
    # metrics), so the host read does not touch the training path.
    run_sum = 0.0
    run_n   = 0
    for cs in range(0, ntrain, load):
      rows = np.sort(perm[cs:cs+load])
      Cc  = load_C(rows)
      dvc = load_dv(rows)
      # pre-shuffle once per chunk (the rows arrive sorted, for
      # host-side read locality): applying the batch permutation
      # here makes every step's batch a contiguous slice, a free
      # view, replacing the per-step gather kernels (the factored
      # path gathered Cc twice and dvc once per step). Costs one
      # transient chunk-sized copy on the GPU.
      bp = torch.randperm(Cc.shape[0], device=device)
      Cc  = Cc[bp]
      dvc = dvc[bp]
      # Drop the ragged last batch so every batch is one size.
      # This matters under torch.compile: it specializes per input
      # shape, and reduce-overhead (CUDA graphs) needs it fixed. bp
      # reshuffles each epoch, so dropped tail rows rotate, no
      # data is permanently lost.
      n_full = (Cc.shape[0] // bs) * bs   # whole batches only
      for s in range(0, n_full, bs):
        loss = fwd_loss(Cc[s:s+bs], dvc[s:s+bs], trim_t, focus_t, s_t)
        # finite contract: refuse to backward a non-finite loss — it
        # would produce NaN gradients, corrupt the weights, and the
        # corrupted model would then score frac 0.0 (a NaN val chi2
        # counts below every threshold) and be selected best. One host
        # sync per step, the deliberate price of catching divergence at
        # its source rather than mis-selecting the run.
        if not bool(torch.isfinite(loss)):
          _report_nonfinite(
            side="training", quantity="scalar training loss",
            n_bad=1, n_total=1,
            positions=["epoch " + str(epoch) + " chunk@" + str(cs)
                       + " batch@" + str(s)])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # gradient-norm clipping (0 = off): rescale the full gradient
        # vector to norm <= clip before the step, so one monster-outlier
        # batch cannot kick the weights (direction kept, size bounded).
        # clip_grad_norm_ skips parameters whose grad is None, the frozen
        # trunk in a head phase, and returns the pre-clip norm.
        if clip > 0.0:
          grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=clip)
        else:
          # clipping off: still compute the norm (read-only, byte-
          # identical to the old no-clip path) so the finite contract can
          # refuse a NaN/Inf gradient before optimizer.step mutates the
          # weights — clipping disabled is not the same as unchecked.
          grad_norm = _global_grad_norm(model.parameters())
        if not bool(torch.isfinite(grad_norm)):
          _report_nonfinite(
            side="training", quantity="gradient norm",
            n_bad=1, n_total=1,
            positions=["epoch " + str(epoch) + " chunk@" + str(cs)
                       + " batch@" + str(s) + " grad_norm="
                       + repr(float(grad_norm))])
        optimizer.step()
        # weight-average update (once ema is live): theta_bar <-
        # beta*theta_bar + (1-beta)*theta, a handful of fused foreach
        # launches (~tens of us vs the ~2 ms step). In-place on
        # theta_bar (a private buffer, not graph-captured), reading the
        # just-stepped params; no_grad, no autograd trail.
        if theta_bar is not None:
          with torch.no_grad():
            torch._foreach_lerp_(theta_bar, ema_params, 1.0 - beta)
        # decoupled L2-SP anchor (finetune.anchor / transfer.refine): a
        # post-step pull toward the reference weights, W <- W - lr*lambda*
        # mask*(W - W_0), read the per-group lr; None = no anchor, byte-
        # identical. After the ema update so the average sees the anchored
        # weights (the shipped model).
        if anchor is not None:
          anchor.apply(optimizer)
        # host float64 accumulation: read the scalar loss to the
        # host and multiply by bs there, so the product cannot overflow a
        # float32 before the sum. loss is already finite (the guard above).
        run_sum += float(loss.detach()) * bs
        run_n   += bs
    train_loss = run_sum / run_n
    # a reduction's result must be checked. Finite per-batch
    # operands do not prove a finite epoch mean. Refuse to publish (append,
    # print, persist) a non-finite epoch loss; name the epoch.
    if not np.isfinite(train_loss):
      _report_nonfinite(side="training", quantity="epoch mean loss",
                        n_bad=1, n_total=1,
                        positions=["epoch " + str(epoch)])
    model.eval()
    # the raw eval drives the plateau scheduler (its dynamics are
    # unchanged); when ema is off this is also the selected / printed
    # metric, so the loop stays byte-identical.
    raw_median, raw_mean, raw_frac = eval_val(model=model,
                                              lossfn=lossfn,
                                              data=data["val"],
                                              load=load,
                                              bs=eval_bs,
                                              thresholds=thresholds,
                                              fwd_chi2=fwd_chi2)
    if theta_bar is not None:
      # swap the average into the model in place (foreach copy_, never
      # rebind a parameter: the compiled graph holds the storage
      # pointers), eval, then restore theta. selection + the printed
      # metrics are the average's, the model that would actually ship.
      with torch.no_grad():
        torch._foreach_copy_(ema_tmp, ema_params)      # stash theta
        torch._foreach_copy_(ema_params, theta_bar)    # theta <- average
      median, mean, frac = eval_val(model=model,
                                    lossfn=lossfn,
                                    data=data["val"],
                                    load=load,
                                    bs=eval_bs,
                                    thresholds=thresholds,
                                    fwd_chi2=fwd_chi2)
      with torch.no_grad():
        torch._foreach_copy_(ema_params, ema_tmp)      # restore theta
    else:
      median, mean, frac = raw_median, raw_mean, raw_frac
    # the scheduler always steps on the raw median (== median when ema is
    # off), so its plateau dynamics never change.
    sched_median = raw_median
    train_losses.append(train_loss)
    medians.append(median)
    means.append(mean)
    fracs.append(frac)

    # f0 = this epoch's fraction of val points with chi2 >
    # thresholds[0] (0.2), the inference goal we minimize.
    # frac[0] is a 0-dim tensor; .item() pulls it to a Python
    # float (one host sync, once per epoch).
    f0 = frac[0].item()
    # Make this epoch the new best when it strictly lowers frac>0.2,
    # or ties the best fraction at a lower median. frac>0.2 is
    # coarse (k/Nval, a step function), so many epochs share a
    # value; the median tiebreaker then keeps the tightest bulk
    # among the equally-good fractions.
    if (f0 < best_frac or
        (f0 == best_frac and median < best_median)):
      best_frac   = f0
      best_median = median
      best_epoch  = epoch
      # snapshot the weights. state_dict() returns references to
      # the live parameters, which keep changing as training
      # continues, so clone to freeze them here. detach drops grad
      # tracking (a stored snapshot needs no autograd). One copy,
      # replaced whenever the best improves.
      best_state = {}
      for k, v in model.state_dict().items():
        best_state[k] = v.detach().clone()
      # keep the optimizer moments matching the best weights, for
      # a possible rewind (see the baseline seed above).
      if rewind:
        best_opt_state = copy.deepcopy(optimizer.state_dict())
      # couple the average into this snapshot: the best now carries its
      # theta_bar as one unit, so a later rewind restores the average
      # too and the shipped model is this epoch's average. Only once ema
      # is live (theta_bar set); a pre-warmup best stays raw-only.
      if theta_bar is not None:
        best_is_ema = True
        with torch.no_grad():
          torch._foreach_copy_(best_theta_bar, theta_bar)

    # the scheduler takes over once the warmup ramp (applied at the
    # top of the epoch) is done, not stepped during warmup, since
    # the plateau scheduler's no-improvement counter must not run
    # while the lr rises. Steps once per epoch, right for
    # ReduceLROnPlateau and epoch schedulers (StepLR,
    # CosineAnnealingLR); a per-batch scheduler (OneCycleLR) would
    # step inside the batch loop instead.
    if epoch > warmup_epochs:
      if isinstance(scheduler,
                    lr_scheduler.ReduceLROnPlateau):
        lrs_before = []
        for grp in optimizer.param_groups:
          lrs_before.append(grp["lr"])
        scheduler.step(sched_median)
        # rewind-to-best: a plateau lr cut means `patience` epochs
        # brought no median improvement, either a true plateau
        # (rewind to best is a no-op, best ~= current) or the run
        # wandered into a bad basin (rewind is the rescue: without
        # it the scheduler keeps decaying the lr inside the
        # wreckage and freezes the run there). Restore the best
        # weights and their optimizer snapshot, then reapply the
        # new (reduced) lrs, load_state_dict would otherwise
        # bring back the snapshot's old lr.
        cut = False
        for grp, lr_old in zip(optimizer.param_groups, lrs_before):
          if grp["lr"] < lr_old:
            cut = True
        if rewind and cut:
          lrs_new = []
          for grp in optimizer.param_groups:
            lrs_new.append(grp["lr"])
          model.load_state_dict(best_state)
          optimizer.load_state_dict(best_opt_state)
          # the invariant: whatever the rewind un-lives must also leave
          # the average. Restore theta_bar as part of the same snapshot
          # when the best carried one; if the best predates ema, re-base
          # the average onto the restored (best) weights so it holds no
          # rejected trajectory.
          if theta_bar is not None:
            with torch.no_grad():
              if best_is_ema:
                torch._foreach_copy_(theta_bar, best_theta_bar)
              else:
                torch._foreach_copy_(theta_bar, ema_params)
          for grp, lr_new in zip(optimizer.param_groups, lrs_new):
            grp["lr"] = lr_new
          if not silent:
            print(f"           lr cut -> rewound to best epoch "
                  f"{best_epoch} (frac>0.2 {best_frac:.4f}), "
                  f"resuming at lr {lrs_new[0]:.2e}")
      else:
        scheduler.step()

    # per-epoch wall time; the GPU is already synced by eval_val
    # above, so this is real elapsed time (train + val).
    dt = time.perf_counter() - t_epoch
    if epoch == 1:
      t_first = dt

    if not silent:
      lr_now = optimizer.param_groups[0]["lr"]
      pairs = []
      for t, f in zip(thresholds.tolist(), frac.tolist()):
        pairs.append(f"{t:g}:{f:.3f}")
      fr = ", ".join(pairs)
      print(f"epoch {epoch:3d}  lr {lr_now:.2e}"
            f"  train {train_loss:.4f}"
            f"  val {mean:.4f}  med {median:.4f}"
            f"  frac>[{fr}]  {dt:5.1f}s")

  # restore the best epoch's weights (possibly the epoch-0 baseline:
  # the pass then returns exactly what it was handed). load_state_dict
  # brings back the raw weights + the constant buffers; when the best
  # carried an average, overwrite the parameters in place with it, so
  # the shipped model is the best average (buffers unchanged).
  model.load_state_dict(best_state)
  if best_is_ema:
    with torch.no_grad():
      torch._foreach_copy_(ema_params, best_theta_bar)
  if not silent:
    print(f"best epoch {best_epoch}: "
          f"frac>0.2 {best_frac:.4f}")

  if not silent:
    # total wall time and the steady-state per-epoch rate (epochs
    # 2..N, dropping epoch 1's compile warmup), the numbers to
    # compare GPUs by.
    total = time.perf_counter() - t_run
    if nepochs > 1:
      steady = (total - t_first) / (nepochs - 1)
      print(f"training done: {nepochs} epochs in {total:.1f}s "
            f"(steady {steady:.2f}s/epoch; epoch 1 {t_first:.1f}s "
            f"incl. compile)")
    else:
      print(f"training done: 1 epoch in {total:.1f}s")
  return train_losses, medians, means, fracs


def run_emulator(train_set,
                 val_set,
                 chi2fn,
                 param_geometry,
                 device,
                 bs=128,
                 nepochs=300,
                 loss=None,
                 model_opts=None,
                 opt_opts=None,
                 lr_opts=None,
                 sched_opts=None,
                 trim_opts=None,
                 focus_opts=None,
                 thresholds=None,
                 gpu_mem_gb=16,
                 use_amp=False,
                 silent=False,
                 seed=0,
                 clip=0.0,
                 rewind=False,
                 trunk_epochs=0,
                 freeze_trunk=None,
                 trunk_opts=None,
                 head_opts=None,
                 ema=None,
                 init_state=None,
                 anchor=None,
                 refine=None):
  """
  One training run; model, optimizer, schedule auto-built.

  Builds the model, optimizer, scheduler, and the regime
  loaders, then trains. Three spec dicts (model_opts, opt_opts,
  lr_opts) group the related knobs, as block_opts groups a
  ResBlock's.

  The two-phase schedule (trunk_epochs > 0, factored head models):

    phase "trunk"  (epochs 1 .. trunk_epochs)
       │  head bypassed: trains as the pure trunk, at trunk cost
       │  (own optimizer / lr warmup / scheduler / trim / focus)
       ▼
    best trunk weights restored  (best frac>0.2 epoch, never the
       │                          last one)
       │  set_train_phase("head" if freeze_trunk else "joint"): the
       │  trunk freezes and the head trains alone (the default), or
       │  trunk + head train together (freeze_trunk false, a joint
       │  fine-tune warm-started by phase 1)
       ▼
    phase "head" / "joint"   (the remaining nepochs - trunk_epochs
       │  epochs) head + gates from the zero-init identity start, so
       │  the loss is continuous at the handoff (fresh optimizer /
       │  warmup / scheduler; the trunk: / head: blocks override
       │  lr / scheduler / loss / trim / focus / clip / rewind / ema
       │  per phase)
       ▼
    model restored to the pass's best frac>0.2 epoch

  (legend: trunk_epochs = epochs of phase 1, the pure trunk; nepochs
  = total epochs, so phase 2 runs nepochs - trunk_epochs; frac>0.2 =
  fraction of val points with delta-chi2 > 0.2, the best-epoch
  selection metric.)

  Arguments:
    train_set    = training source dict: "C" full param dump,
                   "dv" full dv dump, "idx" rows to train on.
    val_set      = validation source dict, same three keys.
    chi2fn         = CosmolikeChi2 (output geometry + loss).
    param_geometry = ParamGeometry (input whitening).
    device         = torch.device the model, geometry, and batches
                     live on (from pick_device); required, no
                     default. make_model and make_optimizer branch
                     on device.type, so a string here (not a
                     torch.device) would fail the .type checks.
    bs           = minibatch size.
    nepochs      = number of passes over the training set.
    loss         = optional nested loss block {mode, berhu} (None or no
                   mode key -> {mode: "sqrt"}, byte-identical to the old
                   default). mode one of "sqrt" / "chi2" / "sqrt_dchi2" /
                   "berhu" (sqrt below the berhu knot, chi2-like above) /
                   "berhu_capped" (the tail vote plateaus above the cap,
                   monster-robust; C1 at the knots, see
                   CosmolikeChi2._reduce). berhu = the {knot, cap} knot
                   sub-block (defaults 0.2 / 10.0), valid only beside a
                   berhu mode, with an optional anneal: sub-block
                   {hold_epochs, anneal_epochs, shape} that ramps the loss
                   from plain sqrt into the full berhu shape (presence =
                   on). Resolved per pass (a phase loss: block
                   full-replaces it); validate_loss enforces the schema.
    model_opts   = model spec dict (see make_model): "cls" the
                   model class + constructor settings. None ->
                   ResMLP, int_dim_res 128, n_blocks 4, block_opts
                   {}.
    opt_opts     = optimizer spec dict (see make_optimizer):
                   "cls" + "weight_decay" + extra kwargs.
                   None -> AdamW, weight_decay 1e-4.
    lr_opts      = learning-rate dict:
                     "lr_base"/"bs_base" -> sqrt-batch rule
                       (lr = lr_base * sqrt(bs / bs_base))
                     "warmup_epochs"     -> linear lr warmup
                   None -> a sensible default. Each phase of a
                   two-phase run restarts at its base lr with a
                   fresh warmup + scheduler, never at the other
                   phase's decayed lr.
    sched_opts   = scheduler spec dict (see make_scheduler):
                   "cls" + its kwargs (mode, patience, factor,
                   ...). None -> ReduceLROnPlateau, mode "min",
                   patience 15, factor 0.75.
    trim_opts    = trim schedule (see anneal_value): "start"/"end"
                   trim fractions, "hold_epochs"/"anneal_epochs",
                   "shape". None -> hold 5% then cosine-anneal to 0.
    focus_opts   = focal-weight schedule (see anneal_value): the
                   per-epoch focus exponent gamma (0 = uniform,
                   higher = harder points weighted more), via
                   "start"/"end", "hold_epochs"/"anneal_epochs",
                   "shape", plus "kappa" (the chi2 scale where the
                   focal weight turns on, fixed over the run, read as
                   focus_scale; default 1.0). None -> no focal
                   weighting (gamma = 0).
    thresholds   = delta-chi2 cutoffs for the val fractions
                   (None -> [0.2, 1, 10, 100]).
    gpu_mem_gb   = emulated budget in GB (non-CUDA only; on CUDA
                   the real free VRAM is used).
    use_amp      = run the forward in low-precision autocast.
    silent       = suppress all printing if True.
    seed         = manual seed for init + per-epoch shuffles.
    clip         = gradient-norm ceiling per step (0 = off); see
                   training_loop_batched. A guard against
                   single-batch gradient kicks from monster
                   outliers under a quadratic loss.
    rewind       = reload the best weights + optimizer snapshot
                   whenever the plateau scheduler cuts the lr; see
                   training_loop_batched. Bounds any excursion
                   into a bad basin to `patience` epochs.
    trunk_epochs = if > 0, two-phase training (the model must
                   define set_train_phase — every design with a
                   correction head does: plain ResCNN / ResTRF on
                   any family, and the factored-IA templates):
                   the first trunk_epochs epochs train the trunk
                   alone with the head bypassed (pure-trunk cost),
                   then the loop restores that phase's best
                   weights and trains phase 2 for the remaining
                   nepochs - trunk_epochs epochs (fresh optimizer,
                   scheduler, and warmup; the zero-init head starts
                   as an exact identity, so the handoff is
                   loss-continuous). Phase 2 freezes the trunk and
                   trains the head alone by default, or trains trunk
                   + head together when freeze_trunk is false. 0
                   (default) = ordinary joint training from epoch 1.
    freeze_trunk = phase-2 mode (None / absent = the frozen default,
                   byte-identical). True: freeze the trunk at the
                   handoff and train the head alone. False: train
                   trunk + head together (a joint fine-tune
                   warm-started by phase 1) via set_train_phase
                   ("joint") — costlier per epoch (the trunk backward
                   returns). Needs trunk_epochs > 0 (else a config
                   error); a non-bool is rejected.
    trunk_opts   = optional trunk-phase (phase 1) overrides;
    head_opts    = optional head-phase (phase 2) overrides.
                   Two symmetric blocks (two-phase runs only; need
                   trunk_epochs > 0) that mirror the top-level
                   train_args schema key-for-key; each phase's block
                   overrides the run defaults for its own pass.
                   (Typical use: by the handoff the trunk has absorbed
                   most outliers, so the head phase wants a different
                   objective, or a lower scheduler patience.) Validated
                   by validate_phase_block; the eight keys, each absent
                   -> the run default is reused:
                     "lr"        -> an overlay {lr_base (the pass's base
                       lr, same sqrt-batch rule), warmup_epochs (the
                       pass's own warmup)}; bs_base stays run-global;
                     "scheduler" -> a full replacement of the scheduler
                       kwargs (mode / patience / factor / ...), keeping
                       the run's scheduler class (a lower head patience
                       lives here);
                     "loss"      -> the pass's nested loss block
                       {mode, berhu}, a full replacement (a berhu head
                       restates its whole loss block, knots included);
                     "trim"      -> the pass's trim schedule, a
                       full replacement block (its hold/anneal
                       count from the pass's own epoch 1);
                     "focus"     -> the pass's focus schedule,
                       ditto (include kappa, no merge with the
                       main block);
                     "clip"      -> the pass's gradient-norm
                       ceiling (0 = off);
                     "rewind"    -> the pass's rewind-on-lr-cut
                       switch (true / false);
                     "ema"       -> the pass's weight-average block
                       {horizon_epochs, anneal}, a full replacement; a
                       null block (key present, value None) disables an
                       inherited top-level ema for that pass.
                   The head block alone may also carry activation: the
                   head-only alias for the per-head activation pin
                   (model.<head>.activation), consumed upstream by
                   build_specs, not here (run_emulator never reads it, so a
                   direct caller gets no alias).
                   EmulatorExperiment.train resolves these away for
                   single-phase models (it merges trunk: into the top
                   level and drops head: / trunk_epochs); a direct
                   caller passes clean args and the guards below stay
                   strict.
    ema          = optional weight-averaging block ({horizon_epochs,
                   anneal}) or None (off). When set, a Polyak average of
                   the weights is kept, coupled to the best snapshot and
                   the rewind as one unit (see training_loop_batched):
                   selection and the reported metrics use the average,
                   the plateau scheduler stays on the raw median, and
                   the returned model carries the best average. An anneal
                   sub-block ramps the horizon from 0 (defers the average
                   past the terrible early era). Off = a byte-identical
                   run. Resolved + re-initialized per phase (a phase ema:
                   full-replaces it, or ema: null disables it there).
    init_state   = optional warm-start state dict (the fine-tune path;
                   emulator/warmstart.py builds it by transferring a
                   source emulator's weights). Forwarded to make_model,
                   which loads it strict into the eager module before any
                   torch.compile. Everything else stays fresh (optimizer,
                   scheduler, warmup, ema, trim / focus schedules): a warm
                   start is not a resume, and the loop's incoming-weights
                   snapshot makes the loaded weights the epoch-0 best
                   baseline for selection and rewind. None (default) = the
                   ordinary random-init run, byte-identical.

  Returns:
    model        = trained network, restored to the best frac>0.2
                   epoch (the best average when ema is set).
    train_losses = per-epoch training loss (list).
    medians      = per-epoch val median chi2 (list).
    means        = per-epoch val mean chi2 (list).
    fracs        = per-epoch list of frac-over-threshold tensors.
  """
  # a bad two-phase config should fail before any setup work.
  trunk_epochs = int(trunk_epochs)
  if trunk_epochs > 0 and trunk_epochs >= nepochs:
    raise ValueError(
      f"trunk_epochs ({trunk_epochs}) must be < nepochs "
      f"({nepochs}): the head needs the remaining epochs")
  if (trunk_opts or head_opts) and trunk_epochs == 0:
    raise ValueError(
      "per-phase overrides (the train_args trunk: / head: blocks) "
      "need trunk_epochs > 0: without the two-phase schedule "
      "they would silently do nothing")
  # freeze_trunk (None = absent = the frozen-trunk default, byte-identical)
  # only means something with a two-phase schedule: a non-bool is a typo,
  # and an explicit value without trunk_epochs > 0 would silently do nothing
  # (the trunk: / head: precedent).
  if freeze_trunk is not None:
    if not isinstance(freeze_trunk, bool):
      raise TypeError(
        f"train_args.freeze_trunk must be a bool (true / false), got "
        f"{type(freeze_trunk).__name__}")
    if trunk_epochs == 0:
      raise ValueError(
        "train_args.freeze_trunk needs trunk_epochs > 0: without the "
        "two-phase schedule it would silently do nothing")
  # validate each present phase block up front (None = absent = no-op):
  # the eight-key whitelist, the flat lr_base migration, the bs_base / cls
  # rejections. Same errors the demotion path raises (both call this), so
  # a typo fails identically whether the model is one- or two-phase.
  validate_phase_block(trunk_opts, "trunk")
  validate_phase_block(head_opts, "head")
  # validate the optional weight-averaging block once, up front (None =
  # off = a byte-identical run); training_loop_batched derives beta from
  # it per phase. Fails before any setup work, like the guards above.
  ema = validate_ema(ema)

  # validate the optional loss block once, up front (None or no mode key ->
  # {mode: sqrt}, byte-identical to the old default): the mode whitelist and
  # the berhu sub-block's local mode check, before any setup work. Each pass
  # re-resolves it below (a phase loss: full-replaces the top-level one).
  loss_top = validate_loss(loss, "train_args")

  # Resolve the numerical AMP policy once, beside the artifact record that
  # persists it. Every training pass consumes these same resolved values.
  amp_dtype = (torch.float16 if device.type == "mps"
               else torch.bfloat16)
  scaler_policy = "unscaled"

  # the CMB residual-roughness term: configured once on the run's
  # loss object (the term is per-sample state on the chi2fn, not a per-pass
  # knob; validate_loss already restricted the block to the top level). A
  # loss object without configure_roughness is not a CMB loss — loud, so a
  # cosmolike / scalar YAML carrying the block never trains silently
  # without it.
  if loss_top["roughness"] is not None:
    if not hasattr(chi2fn, "configure_roughness"):
      raise ValueError(
        "train_args.loss.roughness is the CMB residual-roughness term; "
        "this run's loss (" + type(chi2fn).__name__ + ") does "
        "not support it — remove the block (it applies to data.cmb runs "
        "only)")
    chi2fn.configure_roughness(lam=loss_top["roughness"]["lam"],
                               period_cut=loss_top["roughness"]
                                                  ["period_cut"])

  if model_opts is None:
    model_opts = {"cls": ResMLP,
                  "int_dim_res": 128,
                  "n_blocks": 4,
                  "block_opts": {}}
  if opt_opts is None:
    opt_opts = {"cls": optim.AdamW,
                "weight_decay": 1e-4}
  if lr_opts is None:
    lr_opts = {"lr_base": 5e-3,
               "bs_base": 64.0,
               "warmup_epochs": 5}
  if sched_opts is None:
    sched_opts = {"cls": lr_scheduler.ReduceLROnPlateau,
                  "mode": "min",
                  "patience": 15,
                  "factor": 0.75}
  if thresholds is None:
    # delta-chi2 cutoffs for the reported val fractions (fraction
    # of val points with chi2 above each). The first, 0.2, is the
    # emulator goal and the best-model selection metric (frac >
    # thresholds[0]); the rest are diagnostic bands up the cascade.
    thresholds = torch.tensor([0.2, 1.0, 10.0, 100.0])
  if trim_opts is None:
    # hold a 5% trim, then cosine-anneal it to 0 over the run:
    # drop the worst points while they are junk, re-admit once the
    # model can fit them.
    trim_opts = {"start": 0.05,
                 "end": 0.0,
                 "hold_epochs": 50,
                 "anneal_epochs": max(1, nepochs - 100),
                 "shape": "cosine"}
  if focus_opts is None:
    # default: no focal weighting (the opt-in baseline). shape
    # "const" holds start every epoch, and a gamma (start) of
    # -1 is <= 0, so loss() takes the plain-mean path, runs
    # match no-focus unless a real focus_opts is passed.
    focus_opts = {"shape": "const",
                  "start": -1.0}

  out_dim = chi2fn.dest_idx.numel()

  # sqrt-batch-size rule: lr ~ sqrt(bs).
  learning_rate = (lr_opts["lr_base"]
                   * (bs / lr_opts["bs_base"]) ** 0.5)

  torch.manual_seed(seed)

  # input width = the encoded width, read off the geometry when it
  # advertises one (encoded_dim); the raw parameter count otherwise.
  # The geometry owns its output width, the model should size itself
  # by that statement, not by re-deriving it from the dump.
  in_dim = getattr(param_geometry, "encoded_dim",
                   train_set["C"].shape[1])
  model = make_model(model_opts=model_opts,
                     input_dim=in_dim,
                     output_dim=out_dim,
                     device=device,
                     init_state=init_state)

  # trainable-parameter counts, for comparing model capacity across runs.
  # .parameters() reaches through a torch.compile wrapper (it delegates to
  # the wrapped module), so this is the real count either way; requires_grad
  # filters out the frozen basis buffers (registered as buffers, not
  # parameters, so they never appear here anyway).
  #
  # The second number excludes the pure linear transformations: a Linear
  # or Affine sitting directly in a Sequential composition (the input
  # projection, the output projection, the final Affine) is an affine map
  # with no nonlinearity of its own, it adds width, not shape, and the
  # output projection alone scales with 3*n_keep, dominating the total.
  # The Linears inside ResBlock and the head convs stay
  # counted: interleaved with activations, they are the nonlinear map.
  # A separable head's depthwise+pointwise pair also stays counted,
  # the pair is itself linear (no activation between), but it is a
  # cheap factorization of the plain conv it replaces, and the block
  # activation follows it just the same; its Sequential holds only
  # Conv1d children, which the walk below correctly ignores.
  if not silent:
    n_total  = 0
    n_linear = 0
    for p in model.parameters():
      if p.requires_grad:
        n_total += p.numel()
    for mod in model.modules():
      if isinstance(mod, nn.Sequential):
        # a Sequential iterates over its direct children in order.
        for child in mod:
          if isinstance(child, (nn.Linear, Affine)):
            for p in child.parameters():
              n_linear += p.numel()
    print(f"trainable parameters: {n_total:,} "
          f"({n_total - n_linear:,} excluding pure linear "
          f"transformations)")

    # trunk vs head split, both excluding the pure linear
    # transformations (same convention as above). The trunk is the
    # ResMLP layer stack: at .mlp on the correction models (ResCNN
    # / ResTRF) and at .model on ResMLP and the factored trunks
    # (TemplateMLP / TemplateResCNN / TemplateResTRF); getattr
    # reaches through a torch.compile wrapper. Everything outside
    # it (convs, TRF blocks, gates) is the head; printed only
    # when a head exists (a pure trunk has nothing to split).
    trunk = getattr(model, "mlp", None)
    if trunk is None:
      trunk = getattr(model, "model", None)
    if trunk is not None:
      t_total  = 0
      t_linear = 0
      for p in trunk.parameters():
        if p.requires_grad:
          t_total += p.numel()
      for mod in trunk.modules():
        if isinstance(mod, nn.Sequential):
          for child in mod:
            if isinstance(child, (nn.Linear, Affine)):
              for p in child.parameters():
                t_linear += p.numel()
      if n_total > t_total:
        # head = the complement of the trunk, in both counts.
        head_ex = (n_total - t_total) - (n_linear - t_linear)
        print(f"  trunk {t_total - t_linear:,} vs head "
              f"{head_ex:,} (excluding pure linear "
              f"transformations)")

  # two-phase capability check, now that the model exists.
  # set_train_phase is a duck-typed model capability — every design
  # with a correction head defines it (plain ResCNN / ResTRF on any
  # family, and the factored-IA templates); hasattr reaches through a
  # torch.compile wrapper, which forwards attribute lookups to the
  # wrapped module.
  if trunk_epochs > 0 and not hasattr(model, "set_train_phase"):
    # name the real class, not the torch.compile wrapper: a compiled model
    # is an OptimizedModule forwarding attribute lookups to model._orig_mod,
    # so type(model).__name__ would unhelpfully report "OptimizedModule".
    real_cls = type(getattr(model, "_orig_mod", model)).__name__
    raise ValueError(
      "trunk_epochs needs a two-phase model (one defining "
      "set_train_phase — any design with a correction head, name: "
      "rescnn or restrf); this model is "
      f"{real_cls}")

  if device.type == "cuda":
    budget = torch.cuda.mem_get_info()[0]   # NVIDIA only
  else:
    budget = gpu_mem_gb * 1024**3           # GB -> bytes

  data  = build_loaders(device=device,
                        train_set=train_set,
                        val_set=val_set,
                        param_geometry=param_geometry,
                        chi2fn=chi2fn,
                        model=model,
                        bs=bs,
                        budget=budget)

  # device audit: a stray off-device tensor inside the compiled
  # forward+loss disables CUDA-graph replay silently (inductor's
  # "skipping cudagraphs due to cpu device (primals_N)" names a
  # position, not an owner). Name the owners loudly instead; the
  # run still proceeds, this costs performance, not correctness.
  if not silent:
    for msg in audit_devices(model=model, lossfn=chi2fn,
                             device=device):
      print(f"device audit: {msg}: expected {device}; an "
            "off-device tensor in the compiled step disables "
            "CUDA-graph replay")

  wmupe = lr_opts["warmup_epochs"]

  # phases to run: one (nepochs, phase-name) pair for ordinary
  # training, two for the trunk-then-head schedule. Each pass gets a
  # fresh optimizer + scheduler + warmup: make_optimizer collects
  # every parameter, but frozen ones (requires_grad False) never
  # receive a gradient and AdamW skips grad-None params entirely,
  # no step, no state, no weight decay, so rebuilding per phase
  # both resets the lr schedule for the new phase and leaves the
  # frozen group untouched. training_loop_batched restores its own
  # best-frac>0.2 weights at the end of each pass, so phase 2
  # starts from phase 1's best trunk (not its last epoch), with the
  # zero-init head making the handoff loss-continuous. With
  # freeze_trunk false phase 2 runs "joint" instead: the trunk is not
  # frozen, so its backward returns (a costlier fine-tune).
  freeze = freeze_trunk is not False
  if trunk_epochs > 0:
    plan = [(trunk_epochs, "trunk"),
            (nepochs - trunk_epochs, "head")]
  else:
    plan = [(nepochs, None)]

  train_losses, medians, means, fracs = [], [], [], []
  for n_pass, phase in plan:
    # phase 2 runs "joint" (trunk + head together) when freeze_trunk is
    # false; the set_train_phase name is then "joint", but the pass role
    # stays "head" — it still selects head_opts, drives the best-epoch
    # restore, and labels the override tail. freeze_trunk None/absent/true
    # keeps the frozen default, byte-identical.
    model_phase = phase
    if phase == "head":
      model_phase = "head" if freeze else "joint"
    if model_phase is not None:
      model.set_train_phase(model_phase)

    # per-pass knob resolution: each pass restarts the lr at its base
    # (never the other phase's decayed floor) and falls back to the run
    # defaults; the symmetric trunk: / head: blocks override them for
    # their own pass. The eight keys mirror the top-level schema: lr is
    # an overlay {lr_base (same sqrt-batch rule), warmup_epochs}, scheduler
    # a full replacement of the kwargs (the run's class stays), and
    # loss / trim / focus / clip / rewind / ema replace their value (loss
    # is the nested {mode, berhu} block and ema the {horizon_epochs,
    # anneal} block, both resolved just below; trim / focus each restart
    # at the pass's own epoch 1, like the main ones).
    phase_opts = None
    if phase == "trunk":
      phase_opts = trunk_opts
    elif phase == "head":
      phase_opts = head_opts
    lr_pass     = learning_rate
    wmupe_pass  = wmupe
    sched_pass  = sched_opts
    trim_pass   = trim_opts
    focus_pass  = focus_opts
    clip_pass   = clip
    rewind_pass = rewind
    if phase_opts:
      # lr: overlay lr_base and/or warmup_epochs; bs_base stays run-global.
      phase_lr = phase_opts.get("lr")
      if phase_lr:
        if "lr_base" in phase_lr:
          lr_pass = (phase_lr["lr_base"]
                     * (bs / lr_opts["bs_base"]) ** 0.5)
        wmupe_pass = phase_lr.get("warmup_epochs", wmupe)
      # scheduler: full replacement of the kwargs, keeping the run's class.
      if "scheduler" in phase_opts:
        sched_pass = {"cls": sched_opts["cls"], **phase_opts["scheduler"]}
      trim_pass   = phase_opts.get("trim", trim_opts)
      focus_pass  = phase_opts.get("focus", focus_opts)
      clip_pass   = phase_opts.get("clip", clip)
      rewind_pass = phase_opts.get("rewind", rewind)
    # resolve the effective loss block for this pass: a phase loss:
    # full-replaces the top-level one (the scheduler/trim/focus full
    # replacement), then validate_loss resolves {mode, berhu}. The berhu
    # sub-block's mode check is local to its loss block, so no cross-pass
    # logic is needed (a berhu head simply restates its own loss block).
    loss_raw = loss
    if phase_opts is not None and "loss" in phase_opts:
      loss_raw = phase_opts["loss"]
    which_l = phase if phase is not None else "train_args"
    # the roughness sub-block is run-level state (configured on the chi2fn
    # once, before the passes); when a phase INHERITS the top-level block,
    # drop the key from the inherited copy so re-resolving it under the
    # phase name does not trip the phase rejection (a phase's OWN
    # roughness block still rejects loudly inside validate_loss).
    if (phase is not None and loss_raw is loss
        and isinstance(loss_raw, dict) and "roughness" in loss_raw):
      pruned = {}
      for loss_key, loss_val in loss_raw.items():
        if loss_key != "roughness":
          pruned[loss_key] = loss_val
      loss_raw = pruned
    loss_pass  = validate_loss(loss_raw, which_l)
    mode_pass  = loss_pass["mode"]
    berhu_pass = loss_pass["berhu"]
    # resolve the effective ema block for this pass (the loss/trim/focus
    # full-replacement semantics): a phase ema: replaces the top-level one
    # when the key is present (including ema: null, which disables it for
    # this pass, validate_ema(None) -> off); key-absent inherits. theta_bar
    # already re-initializes per pass, so per-phase ema is independent.
    if phase_opts is not None and "ema" in phase_opts:
      ema_pass = validate_ema(phase_opts["ema"], which_l)
    else:
      ema_pass = ema
    if phase is not None and not silent:
      noted = []
      if wmupe_pass != wmupe:
        noted.append(f"warmup {wmupe_pass}")
      if sched_pass is not sched_opts:
        noted.append("scheduler")
      if trim_pass is not trim_opts:
        noted.append("trim")
      if focus_pass is not focus_opts:
        noted.append("focus")
      if clip_pass != clip:
        noted.append(f"clip {clip_pass:g}")
      if rewind_pass != rewind:
        noted.append(f"rewind {rewind_pass}")
      # ema: note when the phase sets its own (a null phase block disables
      # an inherited one); the full horizon / beta / anneal line then prints
      # from training_loop_batched's own ema banner for this pass.
      if ema_pass is not ema:
        noted.append("ema off" if ema_pass is None
                     else f"ema horizon {ema_pass['horizon_epochs']:g}")
      tail = (f"  [{phase} overrides: {', '.join(noted)}]"
              if noted else "")
      # berhu prints its resolved knot(s) (the cap too for the capped
      # variant) and, when the anneal: schedule is set, its hold + ramp;
      # other modes are unchanged.
      if mode_pass in ("berhu", "berhu_capped"):
        if mode_pass == "berhu":
          knot_note = f" (knot {berhu_pass['knot']:g}"
        else:
          knot_note = (f" (knot {berhu_pass['knot']:g}, "
                       f"cap {berhu_pass['cap']:g}")
        an = berhu_pass.get("anneal")
        if an is not None:
          knot_note += (f"; anneal: hold {an['hold_epochs']} + "
                        f"{an['anneal_epochs']} {an['shape']}")
        knot_note += ")"
      else:
        knot_note = ""
      print(f"phase '{model_phase}': {n_pass} epochs, lr restarts "
            f"at {lr_pass:.2e} (+ {wmupe_pass}-epoch warmup), "
            f"loss_mode {mode_pass}{knot_note}{tail}")

    opt   = make_optimizer(model=model,
                           opt_opts=opt_opts,
                           lr=lr_pass,
                           device=device)
    sched = make_scheduler(optimizer=opt, sched_opts=sched_pass)

    # decoupled L2-SP anchor (finetune.anchor): pull the trained weights back
    # toward the reference (the transferred init_state), with the padded extra
    # input columns masked out. Built here so it reads this pass's optimizer
    # groups (their lr); None on an ordinary run (byte-identical). The
    # transfer refine stage's base-group anchor is a separate pass with its own
    # spec.
    anchor_obj = None
    if anchor is not None:
      anchor_obj = build_anchor(model=model,
                                optimizer=opt,
                                reference_state=anchor["reference"],
                                lam=anchor["lam"],
                                masks=anchor.get("masks"))

    (tl, md, mn,
     fr) = training_loop_batched(nepochs=n_pass,
                                 optimizer=opt,
                                 scheduler=sched,
                                 model=model,
                                 bs=bs,
                                 lossfn=chi2fn,
                                 mode=mode_pass,
                                 data=data,
                                 thresholds=thresholds,
                                 warmup_epochs=wmupe_pass,
                                 trim_opts=trim_pass,
                                 focus_opts=focus_pass,
                                 use_amp=use_amp,
                                 amp_dtype=amp_dtype,
                                 scaler_policy=scaler_policy,
                                 silent=silent,
                                 clip=clip_pass,
                                 rewind=rewind_pass,
                                 ema=ema_pass,
                                 berhu=berhu_pass,
                                 anchor=anchor_obj)
    # histories concatenate across phases: one continuous per-epoch
    # record, as a single-pass run produces. ema re-initializes inside
    # each pass (theta_bar at the pass's own warmup end), so the average
    # never spans the trunk -> head regime change.
    train_losses += tl
    medians      += md
    means        += mn
    fracs        += fr

  # transfer refine stage: after the correction is trained (base
  # frozen, above), optionally unfreeze the base ONCE and train jointly for
  # refine["epochs"] more, with the base at a scaled lr and pulled back toward
  # its pretrained weights by the decoupled anchor. The loss switches to live
  # mode (the base re-enters the graph; the truth half of the staged target is
  # reused, no re-staging). None (every non-refine run) = skipped, byte-
  # identical. The resolved per-pass knobs (mode / trim / focus / berhu /
  # scheduler / warmup) carry over from the correction pass above.
  if refine is not None:
    base_net = chi2fn.base_net
    # capture the PRETRAINED base weights before any drift (the anchor
    # reference W_0); the caller keeps a separate clone for the artifact.
    pretrained = {}
    for name, p in base_net.named_parameters():
      pretrained[name] = p.detach().clone()
    for p in base_net.parameters():
      p.requires_grad_(True)
    base_net.train()
    chi2fn.set_live(True)
    composite = TransferComposite(correction=model, base=base_net)
    if not silent:
      n_base = 0
      for p in base_net.parameters():
        n_base += p.numel()
      print(f"transfer refine: {refine['epochs']} epochs, base unfrozen "
            f"(lr x{refine['base_lr_scale']:g}, anchor lambda "
            f"{refine['anchor']:g}); {n_base:,} base params join the "
            f"correction (weight_decay {opt_opts.get('weight_decay', 0.0):g})")
    opt_r   = make_refine_optimizer(correction=model,
                                    base=base_net,
                                    opt_opts=opt_opts,
                                    lr=learning_rate,
                                    base_lr_scale=refine["base_lr_scale"],
                                    device=device)
    sched_r = make_scheduler(optimizer=opt_r, sched_opts=sched_pass)
    # the anchor pulls only the base group back toward its pretrained weights
    # (build_anchor keys by the base's parameter names, all in opt_r's groups).
    anchor_r = build_anchor(model=base_net,
                            optimizer=opt_r,
                            reference_state=pretrained,
                            lam=refine["anchor"])
    (tl, md, mn,
     fr) = training_loop_batched(nepochs=refine["epochs"],
                                 optimizer=opt_r,
                                 scheduler=sched_r,
                                 model=composite,
                                 bs=bs,
                                 lossfn=chi2fn,
                                 mode=mode_pass,
                                 data=data,
                                 thresholds=thresholds,
                                 warmup_epochs=wmupe_pass,
                                 trim_opts=trim_pass,
                                 focus_opts=focus_pass,
                                 use_amp=use_amp,
                                 amp_dtype=amp_dtype,
                                 scaler_policy=scaler_policy,
                                 silent=silent,
                                 clip=clip_pass,
                                 rewind=rewind_pass,
                                 ema=ema_pass,
                                 berhu=berhu_pass,
                                 anchor=anchor_r)
    train_losses += tl
    medians      += md
    means        += mn
    fracs        += fr

  # resolved_train (save schema v2): the consumed training config, defaults
  # materialized, for config_resolved_yaml. Assembled from the values this
  # run actually used (never a re-derivation): the resolved *_opts, the
  # computed lr, the derived eval batch (off the loaders), the per-phase
  # override blocks as validated. Class objects serialize by qualname (the
  # recipe, not the object). Provenance only: the model reconstructs from
  # resolved_model + the geometry states, not from this.
  def _qual(c):
    return c.__module__ + "." + c.__qualname__
  # the optimizer/scheduler *_opts minus their bookkeeping keys: what is
  # left is exactly the constructor extras the run passed through.
  opt_extras = {}
  for opt_key, opt_val in opt_opts.items():
    if opt_key not in ("cls", "weight_decay"):
      opt_extras[opt_key] = opt_val
  sched_kwargs = {}
  for sched_key, sched_val in sched_opts.items():
    if sched_key != "cls":
      sched_kwargs[sched_key] = sched_val
  resolved_train = {
    "bs": bs,
    "nepochs": nepochs,
    "seed": seed,
    "thresholds": [float(t) for t in thresholds],
    "use_amp": bool(use_amp),
    "amp_dtype": str(amp_dtype),
    "scaler_policy": scaler_policy,
    "clip": clip,
    "rewind": bool(rewind),
    "trunk_epochs": trunk_epochs,
    "freeze_trunk": (freeze_trunk is not False),
    "loss": loss,
    "ema": ema,
    "lr": {"lr_base": lr_opts["lr_base"],
           "bs_base": lr_opts["bs_base"],
           "warmup_epochs": lr_opts.get("warmup_epochs"),
           "lr": learning_rate},
    "optimizer": {"cls": _qual(opt_opts["cls"]),
                  "weight_decay": opt_opts.get("weight_decay", 0.0),
                  "extras": opt_extras},
    "scheduler": {"cls": _qual(sched_opts["cls"]),
                  "kwargs": sched_kwargs},
    "trim": trim_opts,
    "focus": focus_opts,
    "trunk": trunk_opts,
    "head": head_opts,
    "eval_bs": (data.get("eval_bs") if isinstance(data, dict) else None),
    "device": str(device),
  }

  return (model, train_losses, medians, means, fracs, resolved_train)
