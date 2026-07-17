"""Small complete artifact recipes shared by CPU checks.

The production writer refuses incomplete training records.  Tests that are
about another artifact rule still need an honest one-pass record, so they use
this helper instead of copying a large dictionary or falling back to the old
``{"nepochs": 1}`` placeholder.
"""

import copy


def one_pass_training_recipe(
    *, thresholds=(1.0,), epochs=1, compile_mode=None):
  """Return a complete inert record for a deterministic fixture run.

  The helper describes what the fixture already did; it does not run training
  or invent production defaults.  Callers must pass the same thresholds and
  epoch count that their saved history arrays contain.
  """
  thresholds = [float(value) for value in thresholds]
  epochs = int(epochs)
  loss = {"mode": "sqrt", "berhu": None, "roughness": None}
  trim = {
    "shape": "const", "start": 0.0, "end": 0.0,
    "hold_epochs": 0, "anneal_epochs": 1,
  }
  focus = {
    "shape": "const", "start": -1.0, "end": -1.0,
    "hold_epochs": 0, "anneal_epochs": 1, "kappa": 1.0,
  }
  optimizer = {
    "cls": "fixture.NoOptimizer",
    "constructor": {"lr": 1.0e-3},
    "groups": [{
      "learning_rate": 1.0e-3,
      "weight_decay": 0.0,
      "parameter_count": 1,
    }],
  }
  scheduler = {"cls": "fixture.NoScheduler", "kwargs": {}}
  return {
    "recipe_schema": 1,
    "bs": 1,
    "nepochs": epochs,
    "seed": 0,
    "thresholds": thresholds,
    "use_amp": False,
    "amp_dtype": "torch.bfloat16",
    "scaler_policy": "unscaled",
    "clip": 0.0,
    "rewind": False,
    "trunk_epochs": 0,
    "freeze_trunk": True,
    "loss": copy.deepcopy(loss),
    "ema": None,
    "lr": {
      "lr_base": 1.0e-3, "bs_base": 1.0,
      "warmup_epochs": 0, "lr": 1.0e-3,
    },
    "optimizer": {
      "cls": "fixture.NoOptimizer",
      "weight_decay": 0.0,
      "extras": {},
      "constructor": {"lr": 1.0e-3},
    },
    "scheduler": copy.deepcopy(scheduler),
    "trim": copy.deepcopy(trim),
    "focus": copy.deepcopy(focus),
    "trunk": None,
    "head": None,
    "passes": [{
      "role": "single",
      "model_phase": "single",
      "epochs": epochs,
      "history_start": 0,
      "history_stop": epochs,
      "learning_rate": 1.0e-3,
      "warmup_epochs": 0,
      "optimizer": copy.deepcopy(optimizer),
      "scheduler": copy.deepcopy(scheduler),
      "loss": copy.deepcopy(loss),
      "trim": copy.deepcopy(trim),
      "focus": copy.deepcopy(focus),
      "clip": 0.0,
      "rewind": False,
      "ema": None,
      "train_chunk_rows": 1,
      "steps_per_epoch": 1,
      "step_compile_mode": compile_mode,
      "anchor": None,
    }],
    "total_epochs": epochs,
    "execution": {
      "configured_compile_mode": compile_mode,
      "applied_compile_mode": compile_mode,
    },
    "eval_bs": 1,
    "device": "cpu",
  }


def transfer_refined_training_recipe(
    *, thresholds=(1.0,), ordinary_epochs=1, refine_epochs=1,
    compile_mode=None):
  """Return a complete two-pass record for a refined transfer fixture.

  The first pass trains the correction model.  The second jointly refines the
  frozen base and correction, so its history follows the ordinary history
  without resetting the row counter.
  """
  recipe = one_pass_training_recipe(
    thresholds=thresholds, epochs=ordinary_epochs,
    compile_mode=compile_mode)
  start = recipe["total_epochs"]
  refine = dict(recipe["passes"][0])
  refine.update({
    "role": "transfer_refine",
    "model_phase": "joint",
    "epochs": int(refine_epochs),
    "history_start": start,
    "history_stop": start + int(refine_epochs),
    "step_compile_mode": None,
    "anchor": {
      "kind": "transfer_base_l2sp",
      "lambda": 0.1,
      "masked": False,
      "base_lr_scale": 0.1,
      "base_learning_rate": 1.0e-4,
    },
  })
  recipe["passes"].append(refine)
  recipe["total_epochs"] = refine["history_stop"]
  return recipe
