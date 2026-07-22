"""Check that a saved training recipe describes every executed pass.

The training loop can run once, split into trunk and head work, or append a
transfer-refinement pass.  These CPU tests replace the numerical loop with a
small deterministic stand-in and inspect the resolved recipe returned by the
real orchestration code.
"""

import unittest
from unittest import mock

import numpy as np
import torch

from emulator import training


class _Model(torch.nn.Module):
  """Small module exposing the phase and compile facts run_emulator reads."""

  def __init__(self, compile_mode="default"):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.tensor([1.0]))
    self.emul_compile_mode = compile_mode
    self.phases = []

  def forward(self, value):
    return value * self.weight

  def set_train_phase(self, phase):
    self.phases.append(phase)


class _Loss:
  """Only the destination width and optional transfer base are needed."""

  def __init__(self, base_net=None):
    self.dest_idx = torch.tensor([0])
    self.base_net = base_net
    self.live = False

  def set_live(self, value):
    self.live = bool(value)


def _sources():
  """Return tiny arrays whose row count exercises chunk-step accounting."""
  return ({"C": np.zeros((10, 1)), "dv": np.zeros((10, 1)),
           "idx": np.arange(10)},
          {"C": np.zeros((4, 1)), "dv": np.zeros((4, 1)),
           "idx": np.arange(4)})


def _loaders():
  """The inert loader record used by the mocked numerical loop."""
  def load(rows):
    return torch.zeros((len(rows), 1))

  return {
    "train": {"load_C": load, "load_dv": load,
              "idx": np.arange(10), "load": 8},
    "val": {"load_C": load, "load_dv": load,
            "idx": np.arange(4), "load": 4},
    "eval_bs": 4,
  }


def _loop_result(**kwargs):
  """Return one finite history row for each requested epoch."""
  count = int(kwargs["nepochs"])
  return ([1.0] * count,
          [2.0] * count,
          [3.0] * count,
          [torch.tensor([0.1, 0.2])] * count)


def _common_specs():
  """Explicit construction specs keep the test independent of defaults."""
  return {
    "model_opts": {"cls": _Model, "compile_mode": "default"},
    "opt_opts": {"cls": torch.optim.AdamW, "weight_decay": 0.01},
    "lr_opts": {"lr_base": 0.004, "bs_base": 4.0,
                "warmup_epochs": 2},
    "sched_opts": {"cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
                   "mode": "min", "patience": 3, "factor": 0.75},
    "trim_opts": {"shape": "const", "start": 0.0},
    "focus_opts": {"shape": "const", "start": -1.0},
  }


class TrainingPassRecipeTests(unittest.TestCase):
  """Exercise pass resolution without allocating a real model or dataset."""

  def _run(self, *, loss, nepochs, anchor=None, refine=None, device=None,
           spec_overrides=None, loop_effect=None, **phase):
    train_set, val_set = _sources()
    model = _Model(compile_mode="default")
    specs = _common_specs()
    if spec_overrides:
      specs.update(spec_overrides)
    if device is None:
      device = torch.device("cpu")
    if loop_effect is None:
      loop_effect = _loop_result
    with mock.patch.object(training, "make_model", return_value=model), \
         mock.patch.object(training, "build_loaders", return_value=_loaders()), \
         mock.patch.object(torch.cuda, "mem_get_info",
                           return_value=(8 * 1024**3, 8 * 1024**3)), \
         mock.patch.object(training, "build_anchor", return_value=object()), \
         mock.patch.object(training, "training_loop_batched",
                           side_effect=loop_effect):
      return training.run_emulator(
        train_set=train_set,
        val_set=val_set,
        chi2fn=_Loss(base_net=_Model(compile_mode=None)),
        param_geometry=object(),
        device=device,
        bs=4,
        nepochs=nepochs,
        loss=loss,
        thresholds=torch.tensor([0.2, 1.0]),
        use_amp=False,
        silent=True,
        seed=7,
        anchor=anchor,
        refine=refine,
        **phase,
        **specs)

  def test_two_phase_recipe_materializes_overrides_and_boundaries(self):
    anchor = {
      "reference": {"weight": torch.tensor([1.0])},
      "masks": {"weight": torch.tensor([True])},
      "lam": 0.3,
    }
    result = self._run(
      loss=None,
      nepochs=5,
      anchor=anchor,
      trunk_epochs=2,
      freeze_trunk=False,
      ema={"horizon_epochs": 2},
      trunk_opts={
        "lr": {"lr_base": 0.008, "warmup_epochs": 1},
        "clip": 1.25,
      },
      head_opts={
        "loss": {"mode": "chi2"},
        "scheduler": {"mode": "min", "patience": 2, "factor": 0.5},
        "rewind": True,
        "ema": None,
      })
    recipe = result[-1]

    self.assertEqual(recipe["recipe_schema"], 1)
    self.assertEqual(recipe["total_epochs"], 5)
    self.assertEqual(recipe["loss"], {
      "mode": "sqrt", "berhu": None, "roughness": None})
    self.assertEqual(
      recipe["execution"],
      {"configured_compile_mode": "default",
       "applied_compile_mode": "default"})

    trunk, head = recipe["passes"]
    self.assertEqual(
      (trunk["role"], trunk["model_phase"], trunk["epochs"],
       trunk["history_start"], trunk["history_stop"]),
      ("trunk", "trunk", 2, 0, 2))
    self.assertEqual(trunk["learning_rate"], 0.008)
    self.assertEqual(trunk["warmup_epochs"], 1)
    self.assertEqual(trunk["clip"], 1.25)
    self.assertEqual(trunk["loss"]["mode"], "sqrt")
    self.assertEqual(trunk["ema"]["horizon_epochs"], 2)
    self.assertEqual(
      trunk["anchor"],
      {"kind": "finetune_l2sp", "lambda": 0.3, "masked": True})

    self.assertEqual(
      (head["role"], head["model_phase"], head["epochs"],
       head["history_start"], head["history_stop"]),
      ("head", "joint", 3, 2, 5))
    self.assertEqual(head["learning_rate"], 0.004)
    self.assertEqual(head["loss"]["mode"], "chi2")
    self.assertTrue(head["rewind"])
    self.assertIsNone(head["ema"])
    self.assertEqual(head["scheduler"]["kwargs"]["patience"], 2)
    for executed_pass in recipe["passes"]:
      self.assertEqual(executed_pass["train_chunk_rows"], 8)
      self.assertEqual(executed_pass["steps_per_epoch"], 2)
      self.assertEqual(executed_pass["step_compile_mode"], "default")
      self.assertEqual(
        executed_pass["optimizer"]["constructor"]["betas"], [0.9, 0.999])
      self.assertEqual(
        executed_pass["optimizer"]["constructor"]["eps"], 1.0e-8)
      self.assertTrue(executed_pass["optimizer"]["groups"])
      self.assertEqual(
        executed_pass["scheduler"]["kwargs"]["threshold_mode"], "rel")
      self.assertEqual(executed_pass["trim"], {
        "shape": "const", "start": 0.0, "end": 0.0,
        "hold_epochs": 0, "anneal_epochs": 1})
      self.assertEqual(executed_pass["focus"], {
        "shape": "const", "start": -1.0, "end": -1.0,
        "hold_epochs": 0, "anneal_epochs": 1, "kappa": 1.0})

  def test_transfer_refine_is_a_third_kind_of_materialized_pass(self):
    result = self._run(
      loss={"mode": "sqrt"},
      nepochs=2,
      refine={"epochs": 1, "base_lr_scale": 0.1, "anchor": 0.4})
    recipe = result[-1]

    self.assertEqual(recipe["total_epochs"], 3)
    self.assertEqual(len(recipe["passes"]), 2)
    ordinary, refine = recipe["passes"]
    self.assertEqual(
      (ordinary["role"], ordinary["model_phase"], ordinary["history_start"],
       ordinary["history_stop"]),
      ("single", "single", 0, 2))
    self.assertEqual(
      (refine["role"], refine["model_phase"], refine["history_start"],
       refine["history_stop"]),
      ("transfer_refine", "joint", 2, 3))
    self.assertIsNone(refine["step_compile_mode"])
    self.assertEqual(
      refine["anchor"],
      {"kind": "transfer_base_l2sp", "lambda": 0.4, "masked": False,
       "base_lr_scale": 0.1, "base_learning_rate": 0.0004})

  def test_fractional_warmup_is_refused_before_model_construction(self):
    specs = _common_specs()
    specs["lr_opts"] = dict(specs["lr_opts"], warmup_epochs=2.5)
    train_set, val_set = _sources()
    with mock.patch.object(
        training, "make_model",
        side_effect=AssertionError("model construction must not begin")) as make:
      with self.assertRaisesRegex(ValueError, "warmup_epochs.*native integer"):
        training.run_emulator(
          train_set=train_set, val_set=val_set, chi2fn=_Loss(),
          param_geometry=object(), device=torch.device("cpu"), bs=4,
          nepochs=2, loss=None, thresholds=torch.tensor([0.2, 1.0]),
          use_amp=False, silent=True, seed=7, **specs)
    make.assert_not_called()

  def test_cuda_recipe_records_the_forced_fused_optimizer(self):
    result = self._run(
      loss=None, nepochs=1, device=torch.device("cuda"),
      spec_overrides={
        "opt_opts": {
          "cls": torch.optim.AdamW, "weight_decay": 0.01, "fused": False}})
    recipe = result[-1]
    self.assertIs(recipe["optimizer"]["extras"]["fused"], True)
    self.assertIs(recipe["optimizer"]["constructor"]["fused"], True)
    self.assertIs(
      recipe["passes"][0]["optimizer"]["constructor"]["fused"], True)

  def test_pass_records_initial_group_rates_before_scheduler_mutation(self):
    def decayed_loop(**kwargs):
      for group in kwargs["optimizer"].param_groups:
        group["lr"] *= 0.5
      return _loop_result(**kwargs)

    recipe = self._run(
      loss=None, nepochs=1, loop_effect=decayed_loop)[-1]
    group_rates = {
      group["learning_rate"]
      for group in recipe["passes"][0]["optimizer"]["groups"]}
    self.assertEqual(group_rates, {0.004})


class OptimizerSchedulerProtocolTests(unittest.TestCase):
  """Capability refusals that stop before an expensive run starts.

  Each case would otherwise fail after setup (a constructor TypeError
  deep in a driver, an LBFGS step without its closure, a per-batch
  schedule silently stretched to per-epoch, or float16 gradients
  underflowing on MPS), so the refusal must happen at resolution time
  with a message naming the corrective action.
  """

  def test_cuda_fused_is_forced_only_on_a_fused_capable_class(self):
    """A class without a fused argument keeps its ordinary kernels."""
    fused_capable = training._effective_optimizer_extras(
      {"cls": torch.optim.AdamW, "weight_decay": 0.0},
      torch.device("cuda"))
    self.assertIs(fused_capable["fused"], True)
    no_fused_argument = training._effective_optimizer_extras(
      {"cls": torch.optim.Rprop, "weight_decay": 0.0},
      torch.device("cuda"))
    self.assertNotIn("fused", no_fused_argument)

  def test_explicit_fused_on_an_unsupported_class_is_refused(self):
    """A configured fused key on a fused-less class stops by name."""
    with self.assertRaisesRegex(ValueError, "Rprop.*fused"):
      training._effective_optimizer_extras(
        {"cls": torch.optim.Rprop, "weight_decay": 0.0, "fused": True},
        torch.device("cpu"))

  def test_lbfgs_is_refused_before_construction(self):
    """The loop steps without a closure, so LBFGS cannot run here."""
    model = torch.nn.Linear(2, 2)
    with self.assertRaisesRegex(ValueError, "closure"):
      training.make_optimizer(
        model=model,
        opt_opts={"cls": torch.optim.LBFGS, "weight_decay": 0.0},
        lr=1.0e-3,
        device=torch.device("cpu"))

  def test_per_batch_schedulers_are_refused(self):
    """A per-batch schedule cannot run under the per-epoch cadence."""
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    for scheduler_class in (torch.optim.lr_scheduler.OneCycleLR,
                            torch.optim.lr_scheduler.CyclicLR):
      with self.subTest(cls=scheduler_class.__name__):
        with self.assertRaisesRegex(ValueError, "per batch"):
          training.make_scheduler(
            optimizer, {"cls": scheduler_class})

  def test_reduced_precision_on_mps_is_refused(self):
    """MPS float16 autocast without gradient scaling cannot start."""
    with self.assertRaisesRegex(ValueError, "use_amp"):
      training.resolve_amp_policy(use_amp=True,
                                  device=torch.device("mps"))
    amp_dtype, policy = training.resolve_amp_policy(
      use_amp=False, device=torch.device("mps"))
    self.assertIs(amp_dtype, torch.float16)
    self.assertEqual(policy, "unscaled")
    amp_dtype, policy = training.resolve_amp_policy(
      use_amp=True, device=torch.device("cpu"))
    self.assertIs(amp_dtype, torch.bfloat16)
    self.assertEqual(policy, "unscaled")


if __name__ == "__main__":
  unittest.main()
