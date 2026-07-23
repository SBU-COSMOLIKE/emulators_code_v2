"""Witnesses for the training loop's selection record.

The loop compares several candidate weight sets — the incoming weights
(the epoch-0 baseline evaluation, which never enters the history lists)
and every trained epoch — and restores the winner. The record it returns
must therefore state exactly which candidate the caller received, with
that candidate's own statistics, because a scan over the histories names
a trained epoch even when the incoming weights won.

Four witnesses: the baseline-wins case, the trained-epoch case, the
threshold validation (shape, finiteness, strict order), and the YAML
round trip that proves the record is plain data ready for the saved
recipe.
"""

import unittest

import numpy as np
import torch
import yaml

from emulator.training import (
  eval_val, training_loop_batched, validate_thresholds,
)


class _SquaredErrorLoss:
  """Squared-error stand-in exposing the .loss / .chi2 surface the loop calls.

  .loss ignores the trim / focus / berhu knobs (this witness needs no
  annealing) and reduces to a plain mean; .chi2 is the per-row score the
  validation pass thresholds.
  """

  needs_params = False

  def loss(self, pred, target, mode="sqrt", trim=None, focus=None,
           focus_scale=None, berhu_knot=None, berhu_cap=None, berhu_s=None):
    return ((pred - target) ** 2).sum(dim=1).mean()

  def chi2(self, pred, target):
    return ((pred - target) ** 2).sum(dim=1)


# schedules that hold trim and focus off for every epoch, so the witness
# loss is the plain squared error throughout.
_TRIM = {"shape": "const", "start": 0.0, "end": 0.0,
         "hold_epochs": 1, "anneal_epochs": 1}
_FOCUS = {"shape": "const", "start": -1.0, "end": -1.0,
          "hold_epochs": 1, "anneal_epochs": 1, "kappa": 1.0}
_THRESHOLDS = torch.tensor([0.2, 1.0])


def _linear_sources(n_train, n_val, slope, seed):
  """Deterministic 1-D linear-regression loaders: target = slope * input.

  Arguments:
    n_train = training rows.
    n_val   = validation rows.
    slope   = the true linear coefficient the model can learn.
    seed    = generator seed for the input draws.

  Returns:
    the nested train/val loaders mapping training_loop_batched expects.
  """
  generator = torch.Generator().manual_seed(seed)
  train_inputs = torch.randn(n_train, 1, generator=generator)
  val_inputs = torch.randn(n_val, 1, generator=generator)
  train_targets = slope * train_inputs
  val_targets = slope * val_inputs
  return {
    "train": {"load_C": lambda rows: train_inputs[rows],
              "load_dv": lambda rows: train_targets[rows],
              "idx": np.arange(n_train),
              "load": n_train},
    "val": {"load_C": lambda rows: val_inputs[rows],
            "load_dv": lambda rows: val_targets[rows],
            "idx": np.arange(n_val),
            "load": n_val},
  }


def _run_loop(model, learning_rate, nepochs, data):
  """Run the real loop on the witness problem; return its full result."""
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
  return training_loop_batched(
    nepochs=nepochs,
    optimizer=optimizer,
    scheduler=scheduler,
    model=model,
    bs=8,
    lossfn=_SquaredErrorLoss(),
    mode="sqrt",
    data=data,
    trim_opts=_TRIM,
    focus_opts=_FOCUS,
    thresholds=_THRESHOLDS,
    silent=True)


class BaselineSelectedTests(unittest.TestCase):
  """A run no epoch improves must publish the incoming weights as such."""

  def test_zero_learning_rate_publishes_the_baseline(self):
    # learning rate 0 keeps the weights bitwise fixed, so every epoch
    # evaluates identically to the baseline and the strict-improvement
    # rule keeps epoch 0. A history scan cannot represent this outcome:
    # the smallest history index it can name is the first trained epoch.
    torch.manual_seed(5)
    model = torch.nn.Linear(1, 1)
    data = _linear_sources(n_train=16, n_val=8, slope=3.0, seed=11)

    # the baseline statistics, from the same evaluation path the loop
    # seeds its search with.
    model.eval()
    baseline_median, baseline_mean, baseline_frac = eval_val(
      model=model,
      lossfn=_SquaredErrorLoss(),
      data=data["val"],
      load=data["val"]["load"],
      bs=8,
      thresholds=_THRESHOLDS)
    snapshot = {}
    for key, value in model.state_dict().items():
      snapshot[key] = value.detach().clone()

    _, medians, _, _, selection = _run_loop(
      model=model, learning_rate=0.0, nepochs=3, data=data)

    self.assertEqual(selection["candidate"], "baseline")
    self.assertEqual(selection["epoch"], 0)
    self.assertEqual(selection["weights"], "raw")
    self.assertEqual(selection["median"], float(baseline_median))
    self.assertEqual(selection["mean"], float(baseline_mean))
    self.assertEqual(selection["frac"], baseline_frac.tolist())
    self.assertEqual(len(medians), 3)
    # the published weights are the incoming weights, bitwise.
    for key, value in model.state_dict().items():
      self.assertTrue(torch.equal(value, snapshot[key]))


class TrainedEpochSelectedTests(unittest.TestCase):
  """A learnable run must publish a trained epoch with that epoch's stats."""

  def _trained_run(self):
    # start from deliberately wrong weights (zero map) on an exactly
    # linear target, so training is guaranteed to improve the baseline.
    torch.manual_seed(6)
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
      model.weight.fill_(0.0)
      model.bias.fill_(0.0)
    data = _linear_sources(n_train=32, n_val=16, slope=3.0, seed=12)
    return _run_loop(model=model, learning_rate=0.2, nepochs=25, data=data)

  def test_selection_names_a_trained_epoch_with_its_own_statistics(self):
    _, medians, means, fracs, selection = self._trained_run()

    self.assertEqual(selection["candidate"], "trained_epoch")
    self.assertEqual(selection["weights"], "raw")
    epoch = selection["epoch"]
    self.assertGreaterEqual(epoch, 1)
    self.assertLessEqual(epoch, len(medians))
    # the record's statistics are the named epoch's history entries.
    self.assertEqual(selection["median"], float(medians[epoch - 1]))
    self.assertEqual(selection["mean"], float(means[epoch - 1]))
    self.assertEqual(selection["frac"], fracs[epoch - 1].tolist())
    # and the named epoch is the loop's winner: no epoch has a smaller
    # (goal fraction, median) pair, and the first minimizer is kept.
    ranks = []
    for index in range(len(fracs)):
      ranks.append((fracs[index][0].item(), medians[index]))
    self.assertEqual(min(ranks), ranks[epoch - 1])
    self.assertEqual(ranks.index(min(ranks)), epoch - 1)

  def test_record_is_plain_data_and_survives_the_yaml_round_trip(self):
    # save_emulator serializes the resolved recipe with yaml.safe_dump,
    # so every leaf must be a native value (a leaked tensor would either
    # fail the dump or come back changed).
    _, _, _, _, selection = self._trained_run()
    round_trip = yaml.safe_load(yaml.safe_dump(selection))
    self.assertEqual(round_trip, selection)


class ThresholdValidationTests(unittest.TestCase):
  """The one-time threshold check: shape, finiteness, strict order."""

  def test_default_and_valid_vectors_pass(self):
    default = validate_thresholds(None)
    self.assertTrue(
      torch.equal(default, torch.tensor([0.2, 1.0, 10.0, 100.0])))
    vector = torch.tensor([0.2, 0.5, 1.0])
    self.assertIs(validate_thresholds(vector), vector)
    # a plain sequence normalizes to a tensor once, at validation.
    from_list = validate_thresholds([0.5, 2.0])
    self.assertTrue(torch.is_tensor(from_list))
    self.assertTrue(torch.equal(from_list, torch.tensor([0.5, 2.0])))

  def test_malformed_vectors_are_refused(self):
    cases = [
      ([], "at least one value"),
      (torch.zeros((2, 2)), "must be one number"),
      ([True, 1.0], "not a boolean"),
      ([0.2, float("nan")], "finite nonnegative"),
      ([-1.0, 1.0], "finite nonnegative"),
      ([0.2, 0.2], "strictly increasing"),
      ([1.0, 0.2], "strictly increasing"),
    ]
    for vector, expected_message in cases:
      with self.assertRaisesRegex(ValueError, expected_message):
        validate_thresholds(vector)


if __name__ == "__main__":
  unittest.main()
