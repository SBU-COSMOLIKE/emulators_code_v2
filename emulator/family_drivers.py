"""Shared sweep-block helpers for the hyperparameter-sweep drivers.

The `sweep:` YAML block (one dotted train_args leaf + a value list) is
parsed and applied through exactly one definition, used by
cosmic_shear_sweep_hyperparam_emulator.py and, through it, by every
per-family wrapper. The per-family drivers themselves are thin
wrappers over the cosmic-shear drivers' main(prog, family) surface —
the same code path, so the multi-GPU pool, --gpu-pack, and the Optuna
journal study carry over to every family (see any
<family>_<verb>_emulator.py header).

PS: a hyperparameter sweep = one training per value of ONE YAML-chosen
train_args leaf at fixed N_train; act_mode = the activation special
case, where the family is resolved onto the experiment rather than
through train_args.
"""

import copy



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
