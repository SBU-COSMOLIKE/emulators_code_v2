"""Shared helpers for the sweep drivers, which compare training settings.

A sweep is a controlled comparison.  The same emulator is trained
several times, once per candidate value of a single setting, with
everything else held fixed, and every training is scored the same way,
so comparing the scores shows which value of that setting works best —
the way a lab sweeps a dial through a range of positions and records
the reading at each stop.  One training at one candidate value is
called a sweep point.  Two kinds of sweep share these helpers:

- A hyperparameter sweep varies a training choice that is fixed before
  learning starts — the learning rate, the batch size, the number of
  epochs — at a fixed number of training rows.  Such pre-fixed choices
  are the hyperparameters; the weights training itself adjusts are
  not.
- An N-train sweep varies the number of training rows itself, at one
  fixed configuration, so the scores trace how accuracy improves as
  the training set grows.

For a hyperparameter sweep, the varied choice is named in the YAML
``sweep:`` block by a dotted path into the ``train_args`` mapping plus
a list of values to try; in a dotted path such as ``lr.lr_base``, each
dot descends one nesting level, and the final key is called the leaf.
That block is parsed and applied through exactly one definition, in
this module.  Both sweep kinds run from the cosmic-shear drivers
(cosmic_shear_sweep_hyperparam_emulator.py and
cosmic_shear_sweep_ntrain_emulator.py); the per-family drivers are
thin wrappers over those drivers' main(prog, family) surface — the
same code path, so the multi-GPU worker pool, --gpu-pack, and the
Optuna journal study (Optuna is the hyperparameter-search library the
tune drivers use; its journal file lets several processes share one
search) carry over to every family (see any <family>_<verb>_emulator.py
header).

One special case, flagged as act_mode by read_sweep_block: sweeping
the activation — the nonlinear function applied between network
layers — cannot go through train_args, because the experiment resolves
its activation family at construction; the sweep sets it directly on
the experiment object per value instead.
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


def resolved_sweep_record(exp,
                          family,
                          threshold,
                          n_gpus,
                          pool=None,
                          n_train=None,
                          activation_values=None):
  """Record the settings every training of one sweep holds fixed.

  A sweep trains the same emulator once per candidate value of a
  single setting (see the module docstring).  This record collects the
  facts that stay constant across all of those trainings, so every
  saved table and figure can answer "which exact configuration did
  this sweep run?".  It is built AFTER the experiment has applied its
  command-line-over-YAML precedence, so the values here are the ones
  that will actually execute, not the ones any one file requested.
  The record is a tuple of (key, value) pairs rather than a dict: a
  tuple cannot be edited after creation, and it survives pickling —
  pickle is Python's object serializer, and worker processes receive
  their arguments through it — without any dict-ordering doubt.

  Two of the recorded facts deserve definition.  The activation is the
  nonlinear function between network layers, and n_gates counts the
  learned parameters inside the adaptive activation variants.  A head
  is a model's final family-specific output stage; when the
  configuration pins the head's activation separately from the trunk's,
  that pin and its gate count are recorded too, and None records that
  no separate pin exists.

  Arguments:
    exp               = the EmulatorExperiment, with every default and
                        command-line override already applied.
    family            = the output-family identity.
    threshold         = the delta-chi2 cutoff the sweep scores against
                        (chi2 is the covariance-weighted squared
                        difference between the emulated and the exact
                        data vector).
    n_gpus            = the resolved number of worker GPUs.
    pool              = for an N-train sweep — the kind that varies the
                        number of training rows — the total row count
                        those training sets are drawn from; None for a
                        hyperparameter sweep.
    n_train           = for a hyperparameter sweep, the training-row
                        count every sweep point shares; None for an
                        N-train sweep.
    activation_values = the ordered activation-family values when the
                        activation itself is the swept setting, or None
                        for a fixed activation.

  Returns:
    a tuple of (key, value) pairs. ``dict(record)`` is ready for table I/O.

  Raises:
    ValueError for a fine-tune configuration (no train_args.model
    block): the sweep drivers do not support sweeping a fine-tune run.
  """
  # a fine-tune run has NO train_args.model block by contract
  # (validate_finetune_config requires it deleted: the architecture
  # lives in the source artifact), so the model-block reads below
  # would crash on one. No sweep of a fine-tune run has ever been
  # possible — this record is built before any training starts — so
  # the honest behavior is a named refusal here, the one choke point
  # both sweep drivers pass through.
  if "model" not in exp.train_args:
    raise ValueError(
      "this configuration has no train_args.model block, so it is a "
      "fine-tune run (the architecture lives in the source artifact). "
      "Sweeping a fine-tune run is not supported: sweep the source "
      "training instead, then fine-tune the winning artifact once.")
  activation_block = exp.train_args["model"].get("activation")
  activation_n_gates = 3
  if isinstance(activation_block, dict):
    activation_n_gates = int(activation_block.get("n_gates", 3))

  head_activation = None
  head_activation_n_gates = None
  head_block = exp.model_cls.head_block
  if head_block is not None:
    head_pin = exp.train_args["model"].get(head_block, {}).get("activation")
    if head_pin is None and isinstance(exp.train_args.get("head"), dict):
      head_pin = exp.train_args["head"].get("activation")
    if isinstance(head_pin, dict):
      head_activation = str(head_pin["type"])
      head_activation_n_gates = int(head_pin.get("n_gates", 3))
    elif head_pin is not None:
      head_activation = str(head_pin)
      head_activation_n_gates = 3

  activation = exp.activation
  if activation_values is not None:
    activation = "swept"

  pairs = [("model", exp.model_name),
           ("family", family or "cosmic_shear"),
           ("rescale", exp.rescale),
           ("activation", activation),
           ("activation_n_gates", activation_n_gates),
           ("head_activation", head_activation),
           ("head_activation_n_gates", head_activation_n_gates),
           ("threshold", threshold)]
  if activation_values is not None:
    pairs.append(("activation_values", tuple(activation_values)))
  if pool is not None:
    pairs.append(("pool", int(pool)))
  if n_train is not None:
    pairs.append(("n_train", int(n_train)))
  pairs.append(("n_gpus", int(n_gpus)))
  return tuple(pairs)


def sweep_record_value(record, key):
  """Look up one named field in a record resolved_sweep_record built.

  The record is a tuple of (key, value) pairs, not a dict, so this
  helper does the lookup by walking the pairs.  A missing field is an
  error rather than a silent None: the record carries a known, fixed
  set of fields, so asking for anything else is a programming mistake
  worth stopping on.

  Arguments:
    record = tuple returned by resolved_sweep_record.
    key    = field name to read.

  Returns:
    the field value.

  Raises:
    KeyError when the record does not carry the requested field.
  """
  for field, value in record:
    if field == key:
      return value
  raise KeyError("the resolved sweep record has no field " + repr(key))


def sweep_design_label(record):
  """Format the resolved model and activation facts for a figure legend.

  The label states the facts every training of the plotted sweep
  shared: the model class, the analytic-R rescaling mode, the
  activation and its gate count, the swept value list when the
  activation itself is the swept setting, and the head's separately
  pinned activation when one exists.

  Arguments:
    record = tuple returned by resolved_sweep_record.

  Returns:
    one compact label naming model, rescale, activation, and any head pin.
  """
  model = sweep_record_value(record=record, key="model")
  rescale = sweep_record_value(record=record, key="rescale")
  activation = sweep_record_value(record=record, key="activation")
  n_gates = sweep_record_value(record=record, key="activation_n_gates")
  label = (f"{model} ({rescale}; activation {activation}, "
           f"n_gates {n_gates}")
  if activation == "swept":
    values = sweep_record_value(record=record, key="activation_values")
    value_text = []
    for value in values:
      value_text.append(str(value))
    label += "; values " + ", ".join(value_text)
  head_activation = sweep_record_value(
    record=record,
    key="head_activation")
  if head_activation is not None:
    head_n_gates = sweep_record_value(
      record=record,
      key="head_activation_n_gates")
    label += (f"; head {head_activation}, "
              f"n_gates {head_n_gates}")
  return label + ")"


def set_by_path(train_args, path, value):
  """Build a deep copy of train_args with one dotted-path leaf replaced.

  ``copy.deepcopy`` duplicates the whole nested mapping, new inner
  dicts included, so editing the copy can never change the original —
  each sweep point (one candidate value's training) runs from its own
  configuration and the shared baseline stays pristine.  The walk then
  splits the dotted path ("lr.lr_base" -> ["lr", "lr_base"]), descends
  one mapping per segment, creating intermediate mappings that do not
  exist yet (so ``head.lr.lr_base`` sweeps even when the YAML has no
  head: block; a
  head. / trunk_epochs / trunk. sweep on a single-phase model is
  rejected up front by validate_sweep_paths, since resolve_phase_args
  would demote it away), and sets the final key.

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
  """Validate and unpack the YAML `sweep` block.

  The block names the one setting a hyperparameter sweep varies and
  the list of values to try (the module docstring defines the sweep
  itself).  Three refusals guard the sweep before any training starts.  A
  missing block is refused with a paste-ready example (an empty
  `sweep:` line — every child commented out parses to nothing —
  counts as missing).  The two keys
  that select the model class (model.name, model.ia) are refused
  because changing the class mid-sweep would compare architectures,
  not values of one choice; that comparison is a separate sweep per
  architecture with the saved tables overlaid.  Finally the path's
  first segment must be one of SWEEPABLE_TOP_KEYS (defined with its
  reason at the top of this module): training reads train_args
  permissively, so a typo'd path would otherwise be ignored and the
  sweep would silently train the identical configuration once per
  value.

  Arguments:
    cfg = the resolved config mapping (data + train_args + sweep).

  Returns:
    (param, values, act_mode): the dotted path, the value list, and
    whether this is the activation special case that bypasses
    train_args (see the module docstring).

  Raises:
    KeyError when the sweep block is absent or empty; ValueError for a
    missing parameter or value list, a model-class key, or a first
    segment outside SWEEPABLE_TOP_KEYS.
  """
  # cfg.get, not `in`: a bare `sweep:` line with every child commented
  # out parses to None, and None.get below would be an unexplained
  # AttributeError instead of this teaching refusal.
  if cfg.get("sweep") is None:
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
