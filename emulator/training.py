"""Device selection, construction factories, evaluation, and training.

This module is the run layer that ties the package together. pick_device
and make_logger are setup helpers. make_model, make_optimizer, and
make_scheduler each build one component from a {cls, **kwargs} spec
dict; build_run_specs assembles the six
spec dicts from a config (with the default / suggest / search resolvers for the
[default, min, max, kind] hyperparameter ranges). eval_val and eval_source_chi2
score the model, training_loop_batched is the per-epoch loop (trim / focus
annealing, best-epoch tracking), and run_emulator orchestrates: builds
everything, trains, and returns the model plus the per-epoch histories.

PS: a loader is a closure load(rows) -> tensor mapping global row indices to a
ready-to-train batch already on the compute device, hiding where the data lives
(resident on the GPU, streamed from RAM, or a disk memmap). build_loaders
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
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from .batching import build_loaders
from .emulator_designs import ResMLP
from .emulator_designs_building_blocks import Affine
from .loss_functions import anneal_value


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


def make_model(model_opts, input_dim, output_dim, device):
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
    "compile_mode", "reduce-overhead")
  # forward every key except cls / compile_mode to the constructor.
  extra = {}
  for k, v in model_opts.items():
    if k not in ("cls", "compile_mode"):
      extra[k] = v
  model = cls(input_dim=input_dim,
              output_dim=output_dim, **extra).to(device)
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


def make_optimizer(model, opt_opts, lr, device):
  """
  Build the optimizer from a spec dict.

  Mirrors make_model / make_scheduler: the optimizer class is a
  value in the dict, its settings the other keys. Parameters split
  into two groups so weight decay falls only on the weight matrices
  (ndim>=2), never on the 1D params below, decaying those would
  pull a unit-init gain toward 0 and attenuate signal.

  Arguments:
    model    = network whose parameters are optimized;
               named_parameters() splits into weight matrices
               (ndim>=2, decayed) and 1D params (biases, Affine
               gain/bias, activation gamma/beta) not decayed.
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
  # decay weight matrices (ndim>=2); leave the 1D params undecayed.
  decay, no_decay = [], []
  for _, p in model.named_parameters():
    if p.ndim >= 2:
      decay.append(p)
    else:
      no_decay.append(p)
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


# the seven keys a per-phase override block (train_args.trunk / .head) may
# carry, mirroring the top-level train_args schema key-for-key: lr is an
# overlay onto the top-level lr block, scheduler a full replacement of the
# scheduler kwargs, the other five (loss / trim / focus / clip / rewind)
# replace their same-named top-level value (loss is the nested loss block
# {mode, berhu}, full replacement like trim/focus and validated by
# validate_loss). validate_phase_block is the one whitelist, imported by
# experiment.py (training must not import experiment, so the validator
# lives here).
_PHASE_BLOCK_KEYS = ("lr", "scheduler", "loss", "trim", "focus",
                     "clip", "rewind")


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
  the other five keys scalars / their own blocks. Absent (None) validates
  trivially and the run is unchanged.

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
    key (the seven-key whitelist), a bs_base inside the phase lr (the
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
  unknown = set(block) - set(_PHASE_BLOCK_KEYS)
  if unknown:
    raise ValueError(
      f"unknown train_args.{which} key(s): {sorted(unknown)}; a phase "
      f"block overrides only {list(_PHASE_BLOCK_KEYS)}")
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
  # loss is the nested loss block {mode, berhu}; validate it here so a
  # malformed phase loss block fails identically on the two-phase path
  # (run_emulator) and the single-phase demotion (resolve_phase_args), and
  # even when the block is later dropped (a head: on a single-phase model).
  validate_loss(block.get("loss"), which)
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

# the keys the optional berhu anneal: sub-block accepts (trim's argument
# names) and the schedule shapes it may name (the anneal_value helper's
# set). Present -> the loss ramps from plain sqrt into the full berhu shape
# over the schedule; see _validate_berhu_anneal and the s_t threading in
# training_loop_batched.
_ANNEAL_KEYS   = ("hold_epochs", "anneal_epochs", "shape")
_ANNEAL_SHAPES = ("const", "linear", "cosine", "step")


def _validate_berhu_anneal(anneal, which):
  """
  Validate the optional anneal: sub-block of a loss.berhu block.

  The anneal schedule (trim's argument names) ramps a blend factor s from 0
  to 1, so a berhu run starts as plain sqrt and eases into the full berhu
  shape; the training loop feeds {start: 0, end: 1, **this block} to the
  existing anneal_value helper (no new schedule code). Standalone / pure
  (no torch).

  Arguments:
    anneal = the raw anneal sub-block (loss["berhu"]["anneal"]).
    which  = "loss" / "trunk.loss" / "head.loss"; the block is
             train_args.{which}.berhu.anneal, named in error messages.

  Returns:
    the validated anneal mapping (a copy).

  Raises:
    TypeError if anneal is not a mapping. ValueError on an unknown key, a
    missing key, a non-integer / negative hold_epochs, an anneal_epochs
    below 1 (bool rejected in both), or an unknown shape.
  """
  q = f"train_args.{which}.berhu.anneal"
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
    _validate_berhu_anneal). TypeError if berhu (or its anneal) is present
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
  if "anneal" in berhu:
    out["anneal"] = _validate_berhu_anneal(berhu["anneal"], which)
  else:
    out["anneal"] = None
  return out


# the five loss transforms CosmolikeChi2._reduce implements; validate_loss
# whitelists them at config time, so a typo'd mode fails at startup instead
# of surviving to the first compiled loss call. Order matches the _reduce
# ladder.
_LOSS_MODES = ("chi2", "sqrt", "sqrt_dchi2", "berhu", "berhu_capped")

# the keys a train_args.loss block accepts: the mode string and the berhu
# knot sub-block (valid only beside a berhu mode; see validate_loss).
_LOSS_KEYS = ("mode", "berhu")


def validate_loss(loss, which):
  """
  Validate a train_args.loss block and resolve it to {mode, berhu}.

  A standalone pure function (no torch), beside validate_ema /
  validate_berhu. The loss block localizes every loss option: the mode
  string and, for the berhu family, its {knot, cap} sub-block. Absent (or
  no mode key) resolves to plain "sqrt", the run default, byte-identical to
  a run with no loss block. The berhu sub-block routes through
  validate_berhu against this block's own mode, so "a berhu sub-block on a
  non-berhu mode" is a purely local error (no cross-pass logic).

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
    key, an unknown mode (naming the five), or (via validate_berhu) a berhu
    sub-block paired with a non-berhu mode / a malformed knot pair.
  """
  qual = ("train_args.loss" if which == "train_args"
          else f"train_args.{which}.loss")
  if loss is None:
    return {"mode": "sqrt", "berhu": None}
  if not isinstance(loss, dict):
    raise TypeError(
      f"{qual} must be a mapping {{mode, berhu}}, got "
      f"{type(loss).__name__}")
  unknown = set(loss) - set(_LOSS_KEYS)
  if unknown:
    raise ValueError(
      f"unknown {qual} key(s): {sorted(unknown)}; allowed: "
      f"{list(_LOSS_KEYS)}")
  mode = loss.get("mode", "sqrt")
  if mode not in _LOSS_MODES:
    raise ValueError(
      f"unknown {qual}.mode {mode!r}; one of {list(_LOSS_MODES)}")
  # the berhu sub-block is validated against this block's mode (the local
  # check that replaces the deleted cross-pass D-B1 machinery). validate_berhu
  # names it train_args.{which_berhu}.berhu, so pass "loss" / "trunk.loss".
  which_berhu = "loss" if which == "train_args" else f"{which}.loss"
  resolved = validate_berhu(loss.get("berhu"), mode, which_berhu)
  # a non-berhu mode carries no knots (None -> the training loop's unused
  # defaults); a berhu mode keeps the resolved pair for its knot tensors.
  berhu = resolved if mode in ("berhu", "berhu_capped") else None
  return {"mode": mode, "berhu": berhu}


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


# the keys the optional train_args.ema block accepts (the weight-averaging
# window). horizon_epochs is the only one; the whitelist is the typo guard.
_EMA_KEYS = {"horizon_epochs"}


def validate_ema(ema):
  """
  Validate the optional train_args.ema block (weight ema / Polyak).

  A standalone pure function (no torch), so it is unit-testable in
  isolation. None (the block absent) is the disabled sentinel and the
  loop stays byte-identical; otherwise the block must be a mapping with a
  positive horizon_epochs and no other key.

  Arguments:
    ema = the train_args["ema"] value, or None when the block is absent.

  Returns:
    the validated ema mapping, or None when it was absent (disabled).

  Raises:
    ValueError on an unknown ema key, a missing horizon_epochs, or a
    non-positive / non-numeric horizon_epochs (bool rejected).
    TypeError if ema is present but not a mapping.
  """
  if ema is None:
    return None
  if not isinstance(ema, dict):
    raise TypeError(
      f"train_args.ema must be a mapping ({{horizon_epochs: ...}}), got "
      f"{type(ema).__name__}")
  unknown = set(ema) - _EMA_KEYS
  if unknown:
    raise ValueError(
      f"unknown train_args.ema key(s): {sorted(unknown)}; allowed: "
      f"{sorted(_EMA_KEYS)}")
  if "horizon_epochs" not in ema:
    raise ValueError(
      "train_args.ema needs 'horizon_epochs' (the averaging window in "
      "epochs, a positive number)")
  h = ema["horizon_epochs"]
  # bool is an int subclass, but True/False is never an epoch count.
  if isinstance(h, bool) or not isinstance(h, (int, float)):
    raise ValueError(
      f"train_args.ema.horizon_epochs must be a positive number of "
      f"epochs, got {h!r}")
  if h <= 0:
    raise ValueError(
      f"train_args.ema.horizon_epochs must be > 0, got {h}")
  return ema


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
    c  (n_val,)               on device, 4 bytes per val point
       │  one .cpu()          the eval's only device-to-host copy
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
  device-to-host traffic stays at one small transfer per eval,
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
    data       = dict with load_C, load_dv (the loaders) and idx
                 (global validation row indices; read into the
                 local vidx).
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
  with torch.no_grad():
    for cs in range(0, len(vidx), load):
      rows = np.sort(vidx[cs:cs+load])
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

  c = torch.cat(chi2s).cpu() # per-sample chi2; the one D2H copy
  mean   = c.mean().item()
  median = c.median().item()

  # c[:, None] (Nval, 1) and thresholds[None, :] (1, T) broadcast
  # into a (Nval, T) boolean grid: entry [i, j] = "is point i's
  # chi2 above threshold j?". mean(0) over samples -> the fraction
  # past each threshold. ([:, None] is the numpy/torch unsqueeze.)
  frac = (c[:, None] > thresholds[None, :]).float().mean(0)
  return median, mean, frac


def eval_source_chi2(model,
                     param_geometry,
                     chi2fn,
                     source,
                     device,
                     bs):
  """
  Per-cosmology delta-chi2 of the emulator over one source.

  Scores every row of `source` in source["idx"]: encodes its
  parameters into model inputs, predicts the whitened data vector,
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

  dchi2 = torch.cat(chunks).double().numpy()
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
                          berhu=None):
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
                 each with load_C, load_dv (the loaders), idx
                 (global rows into the full dump), and load (rows
                 per streamed chunk). The val sub-dict is handed to
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
    ema        = optional validated ema block ({horizon_epochs}) or
                 None (off = a byte-identical loop). When set, a
                 Polyak weight average theta_bar is kept from the end
                 of warmup: updated after every optimizer.step at
                 beta = derive_ema_beta(horizon_epochs,
                 steps_per_epoch), evaluated per epoch by swapping it
                 into the model in place, and coupled to the best
                 snapshot as one unit {theta, optimizer, theta_bar}
                 (saved on best, restored on rewind), so anything the
                 rewind un-lives leaves the average too. Selection and
                 the printed metrics then use the average; the plateau
                 scheduler stays on the raw median (dynamics
                 unchanged). The returned model carries the best
                 average.
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
  # loaders: global row indices -> ready-to-train param inputs
  # / dv targets on the GPU (the regime hides where they live).
  load_C  = data["train"]["load_C"]
  load_dv = data["train"]["load_dv"]
  tidx    = data["train"]["idx"]   # global training rows into the dump
  # device the model lives on; place new tensors here too.
  # model.parameters() is an iterator, so next(...), not [0].
  device  = next(model.parameters()).device
  ntrain  = len(tidx)              # training rows per epoch
  load    = data["train"]["load"]  # rows per streamed chunk
  # chunks per epoch = ceil(ntrain / load); + load - 1 rounds the
  # integer division up.
  nchunks = (ntrain + load - 1) // load

  if not silent:
    print(f"{load} rows/chunk, {nchunks} chunks/epoch, "
          f"amp={use_amp}, loss mode = {mode}")

  # ema (weight averaging) state, all gated on an ema block being set:
  # when ema is None every ema branch below is skipped and the loop is
  # byte-identical to before the feature. validate_ema already ran in
  # run_emulator, so a non-None block is well-formed here.
  ema_on         = ema is not None
  theta_bar      = None    # the running average; allocated at warmup end
  ema_tmp        = None    # param-sized scratch for the in-place eval swap
  best_theta_bar = None    # the average at the best epoch (the shipped one)
  best_is_ema    = False   # does the current best snapshot carry theta_bar?
  if ema_on:
    # steps_per_epoch = the loop's real full-batch count (whole batches
    # per chunk, summed over chunks), so the horizon is counted in steps
    # and beta comes out batch-size-invariant.
    steps_per_epoch = 0
    for cs in range(0, ntrain, load):
      chunk = min(load, ntrain - cs)
      steps_per_epoch += chunk // bs
    beta = derive_ema_beta(horizon_epochs=ema["horizon_epochs"],
                           steps_per_epoch=steps_per_epoch)
    # the live parameter tensors (reached through any torch.compile
    # wrapper). The in-place foreach ops below mutate their storage, the
    # storage the compiled graph captured, so replay stays valid; never
    # rebind a parameter's .data or the list entry.
    ema_params = list(model.parameters())
    if not silent:
      print(f"ema: horizon {ema['horizon_epochs']} epochs "
            f"(beta {beta:.6f}; selection + metrics on the average, "
            f"scheduler on the raw median)")

  train_losses, medians, means, fracs = [], [], [], []

  # MPS (Apple Silicon) has no float64. Accumulate the loss in
  # float64 where supported (CUDA/CPU), float32 on MPS: only
  # the epoch-mean train loss, so the fallback is harmless.
  acc_dtype = (torch.float32 if device.type == "mps"
                             else torch.float64)

  amp_dtype = (torch.float16 if device.type == "mps"
               else torch.bfloat16)

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
    # initialize the weight average at the end of warmup (the high-lr
    # ramp is not worth averaging): the first post-warmup epoch clones
    # theta into theta_bar + the scratch buffers, then the per-step
    # update below starts. Before this, ema_on runs the loop as usual.
    if ema_on and theta_bar is None and epoch > warmup_epochs:
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

    # epoch training loss, accumulated on-device
    run_sum = torch.zeros((),
                          device=device,
                          dtype=acc_dtype)

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
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # gradient-norm clipping (0 = off): rescale the full
        # gradient vector to norm <= clip before the step, so one
        # monster-outlier batch cannot kick the weights (direction
        # kept, size bounded). clip_grad_norm_ skips parameters
        # whose grad is None, the frozen trunk in a head phase.
        if clip > 0.0:
          nn.utils.clip_grad_norm_(model.parameters(),
                                   max_norm=clip)
        optimizer.step()
        # weight-average update (once ema is live): theta_bar <-
        # beta*theta_bar + (1-beta)*theta, a handful of fused foreach
        # launches (~tens of us vs the ~2 ms step). In-place on
        # theta_bar (a private buffer, not graph-captured), reading the
        # just-stepped params; no_grad, no autograd trail.
        if theta_bar is not None:
          with torch.no_grad():
            torch._foreach_lerp_(theta_bar, ema_params, 1.0 - beta)
        run_sum += loss.detach() * bs
        run_n   += bs
    train_loss = (run_sum / run_n).item()
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
                 trunk_opts=None,
                 head_opts=None,
                 ema=None):
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
       │  set_train_phase("head"): trunk frozen, run under no_grad
       ▼
    phase "head"   (the remaining nepochs - trunk_epochs epochs)
       │  head + gates only, from the zero-init identity start, so
       │  the loss is continuous at the handoff (fresh optimizer /
       │  warmup / scheduler; the trunk: / head: blocks override
       │  lr / loss / trim / focus / clip / rewind per
       │  phase)
       ▼
    model restored to the head pass's best frac>0.2 epoch

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
                   define set_train_phase, e.g. TemplateResCNN):
                   the first trunk_epochs epochs train the trunk
                   alone with the head bypassed (pure-trunk cost),
                   then the loop restores that phase's best
                   weights, freezes the trunk, and trains the head
                   only for the remaining nepochs - trunk_epochs
                   epochs (fresh optimizer, scheduler, and warmup;
                   the zero-init head starts as an exact identity,
                   so the handoff is loss-continuous). 0 (default)
                   = ordinary joint training.
    trunk_opts   = optional trunk-phase (phase 1) overrides;
    head_opts    = optional head-phase (phase 2) overrides.
                   Two symmetric blocks (two-phase runs only; need
                   trunk_epochs > 0) that mirror the top-level
                   train_args schema key-for-key; each phase's block
                   overrides the run defaults for its own pass.
                   (Typical use: by the handoff the trunk has absorbed
                   most outliers, so the head phase wants a different
                   objective, or a lower scheduler patience.) Validated
                   by validate_phase_block; the seven keys, each absent
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
                       switch (true / false).
                   EmulatorExperiment.train resolves these away for
                   single-phase models (it merges trunk: into the top
                   level and drops head: / trunk_epochs); a direct
                   caller passes clean args and the guards below stay
                   strict.
    ema          = optional weight-averaging block ({horizon_epochs})
                   or None (off). When set, a Polyak average of the
                   weights is kept, coupled to the best snapshot and
                   the rewind as one unit (see training_loop_batched):
                   selection and the reported metrics use the average,
                   the plateau scheduler stays on the raw median, and
                   the returned model carries the best average. Off =
                   a byte-identical run. Re-initialized per phase.

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
  # validate each present phase block up front (None = absent = no-op):
  # the seven-key whitelist, the flat lr_base migration, the bs_base / cls
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
  validate_loss(loss, "train_args")

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
                     device=device)

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
  # set_train_phase is a duck-typed model capability (TemplateResCNN);
  # hasattr reaches through a torch.compile wrapper, which forwards
  # attribute lookups to the wrapped module.
  if trunk_epochs > 0 and not hasattr(model, "set_train_phase"):
    # name the real class, not the torch.compile wrapper: a compiled model
    # is an OptimizedModule forwarding attribute lookups to model._orig_mod,
    # so type(model).__name__ would unhelpfully report "OptimizedModule".
    real_cls = type(getattr(model, "_orig_mod", model)).__name__
    raise ValueError(
      "trunk_epochs needs a two-phase model (one defining "
      "set_train_phase, e.g. name: rescnn + ia: nla); this model is "
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
  # zero-init head making the handoff loss-continuous.
  if trunk_epochs > 0:
    plan = [(trunk_epochs, "trunk"),
            (nepochs - trunk_epochs, "head")]
  else:
    plan = [(nepochs, None)]

  train_losses, medians, means, fracs = [], [], [], []
  for n_pass, phase in plan:
    if phase is not None:
      model.set_train_phase(phase)

    # per-pass knob resolution: each pass restarts the lr at its base
    # (never the other phase's decayed floor) and falls back to the run
    # defaults; the symmetric trunk: / head: blocks override them for
    # their own pass. The seven keys mirror the top-level schema: lr is
    # an overlay {lr_base (same sqrt-batch rule), warmup_epochs}, scheduler
    # a full replacement of the kwargs (the run's class stays), and
    # loss / trim / focus / clip / rewind replace their value (loss is the
    # nested {mode, berhu} block, resolved just below; trim / focus each
    # restart at the pass's own epoch 1, like the main ones).
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
    loss_pass  = validate_loss(loss_raw, which_l)
    mode_pass  = loss_pass["mode"]
    berhu_pass = loss_pass["berhu"]
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
      print(f"phase '{phase}': {n_pass} epochs, lr restarts "
            f"at {lr_pass:.2e} (+ {wmupe_pass}-epoch warmup), "
            f"loss_mode {mode_pass}{knot_note}{tail}")

    opt   = make_optimizer(model=model,
                           opt_opts=opt_opts,
                           lr=lr_pass,
                           device=device)
    sched = make_scheduler(optimizer=opt, sched_opts=sched_pass)

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
                                 silent=silent,
                                 clip=clip_pass,
                                 rewind=rewind_pass,
                                 ema=ema,
                                 berhu=berhu_pass)
    # histories concatenate across phases: one continuous per-epoch
    # record, as a single-pass run produces. ema re-initializes inside
    # each pass (theta_bar at the pass's own warmup end), so the average
    # never spans the trunk -> head regime change.
    train_losses += tl
    medians      += md
    means        += mn
    fracs        += fr

  return model, train_losses, medians, means, fracs
