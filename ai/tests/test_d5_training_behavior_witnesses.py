"""CPU tests for the DIDACTICS-62 numerical gate witnesses."""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from ai.gates.checks import d5_training_behaviors as d5
from ai.gates.checks import logscan
from emulator.training import (_ema_first_live_line, _parameter_digest,
                               _phase2_digest_line, _training_trunk,
                               make_scheduler)


class ScheduleBehaviorTest(unittest.TestCase):

  def setUp(self):
    self.opts = {
      "shape": "cosine",
      "start": 0.0,
      "end": 1.0,
      "hold_epochs": 5,
      "anneal_epochs": 10,
    }

  def test_production_schedule_has_all_known_values_and_continuity(self):
    ok, detail = d5.schedule_behavior(self.opts)
    self.assertTrue(ok, detail)
    self.assertIn("4/5/6/10/15/16", detail)
    self.assertIn("both joins continuous", detail)

  def test_constant_schedule_mutation_is_rejected(self):
    with mock.patch.object(d5, "anneal_value", return_value=0.0):
      ok, detail = d5.schedule_behavior(self.opts)
    self.assertFalse(ok)
    self.assertIn("epoch 6", detail)

  def test_early_start_mutation_is_rejected(self):
    production = d5.anneal_value

    def one_epoch_early(epoch, opts):
      return production(epoch=epoch + 1, opts=opts)

    with mock.patch.object(d5, "anneal_value", side_effect=one_epoch_early):
      ok, detail = d5.schedule_behavior(self.opts)
    self.assertFalse(ok)
    self.assertIn("epoch 5", detail)

  def test_wrong_configured_boundary_is_rejected(self):
    changed = dict(self.opts)
    changed["hold_epochs"] = 4
    ok, detail = d5.schedule_behavior(changed)
    self.assertFalse(ok)
    self.assertIn("hold 5", detail)


class ActivationLearningTest(unittest.TestCase):

  def test_relu_per_feature_has_exact_values_and_learns(self):
    ok, detail = d5.activation_learning("relu", "per_feature", 1701)
    self.assertTrue(ok, detail)
    self.assertIn("relu exact values", detail)

  def test_tanh_affine_has_exact_values_and_learns(self):
    ok, detail = d5.activation_learning("tanh", "affine", 1702)
    self.assertTrue(ok, detail)
    self.assertIn("tanh exact values", detail)

  def test_dead_network_is_rejected(self):
    ok, detail = d5.judge_learning(
      initial_loss=2.0, final_loss=2.0, mean_only_loss=1.0)
    self.assertFalse(ok)
    self.assertIn("did not descend", detail)

  def test_mean_only_predictor_is_rejected(self):
    ok, detail = d5.judge_learning(
      initial_loss=2.0, final_loss=1.0, mean_only_loss=1.0)
    self.assertFalse(ok)
    self.assertIn("mean-only", detail)

  def test_nonfinite_loss_is_rejected(self):
    ok, detail = d5.judge_learning(
      initial_loss=2.0, final_loss=float("nan"), mean_only_loss=1.0)
    self.assertFalse(ok)
    self.assertIn("finite", detail)


def _head_log(*, cut_epoch=20, second_cut_epoch=None, missing_epoch=None,
              phase_name="head"):
  """Build a complete, phase-tagged 30-epoch LR record."""
  lines = ["phase 'trunk': 30 epochs, lr restarts at 5.00e-03",
           "epoch   1  lr 6.25e-04 train 1 val 1 med 1",
           "phase '" + phase_name
           + "': 30 epochs, lr restarts at 2.00e-03"]
  for epoch in range(1, 31):
    if epoch == missing_epoch:
      continue
    if epoch <= 8:
      lr = 0.002 * epoch / 8.0
    elif epoch < cut_epoch:
      lr = 0.002
    else:
      lr = 0.0016
    if second_cut_epoch is not None and epoch >= second_cut_epoch:
      lr = 0.00128
    lines.append("epoch " + str(epoch) + "  lr " + format(lr, ".2e")
                 + " train 1 val 1 med 1")
  return "\n".join(lines) + "\n"


class HeadCadenceLogTest(unittest.TestCase):

  def _judge(self, text):
    return logscan.head_lr_cadence(
      text=text, phase="head", phase_epochs=30, warmup_epochs=8,
      patience=10, factor=0.8)

  def test_complete_head_phase_has_one_cut_at_epoch_20(self):
    ok, detail = self._judge(_head_log())
    self.assertTrue(ok, detail)
    self.assertIn("one cut across all 30 epochs", detail)

  def test_shipped_forced_plateau_cuts_once_at_epoch_20(self):
    cfg = d5._load_yaml(
      "ai/gates/configs/head-scheduler-override-config.yaml")
    scheduler_cfg = dict(cfg["train_args"]["head"]["scheduler"])
    scheduler_cfg["cls"] = torch.optim.lr_scheduler.ReduceLROnPlateau
    parameter = nn.Parameter(torch.tensor([0.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.002)
    scheduler = make_scheduler(optimizer, scheduler_cfg)
    observed = []
    for epoch in range(1, 31):
      if epoch > 8:
        scheduler.step(1.0)
      observed.append(optimizer.param_groups[0]["lr"])
    transitions = []
    for index in range(8, len(observed)):
      if observed[index] != observed[index - 1]:
        transitions.append(index + 1)
    self.assertEqual(transitions, [20])
    self.assertAlmostEqual(observed[19], 0.002 * 0.8)

  def test_ignored_override_with_no_cut_is_rejected(self):
    ok, detail = self._judge(_head_log(cut_epoch=31))
    self.assertFalse(ok)
    self.assertIn("expected first cut", detail)

  def test_early_cut_is_rejected(self):
    ok, detail = self._judge(_head_log(cut_epoch=19))
    self.assertFalse(ok)
    self.assertIn("before the expected cut", detail)

  def test_later_second_cut_is_rejected(self):
    ok, detail = self._judge(_head_log(second_cut_epoch=29))
    self.assertFalse(ok)
    self.assertIn("changed lr again", detail)

  def test_missing_epoch_is_rejected(self):
    ok, detail = self._judge(_head_log(missing_epoch=27))
    self.assertFalse(ok)
    self.assertIn("missing [27]", detail)

  def test_wrong_phase_label_is_rejected(self):
    ok, detail = self._judge(_head_log(phase_name="joint"))
    self.assertFalse(ok)
    self.assertIn("missing phase", detail)


def _ema_line(*, epoch=6, schedule=0.024471741852423234, beta=None,
              steps=100, raw_median=0.4, averaged_median=0.35,
              raw_mean=0.5, averaged_mean=0.45):
  if beta is None:
    denom = 3.0 * schedule * steps
    beta = 0.0 if denom < 1.0 else 1.0 - 1.0 / denom
  return _ema_first_live_line(
    epoch=epoch, schedule=schedule, beta=beta, steps_per_epoch=steps,
    raw_median=raw_median, averaged_median=averaged_median,
    raw_mean=raw_mean, averaged_mean=averaged_mean)


class EmaFirstLiveLogTest(unittest.TestCase):

  def _judge(self, text):
    return logscan.ema_first_live(
      text=text, expected_epoch=6,
      expected_schedule=0.024471741852423234,
      horizon_epochs=3.0)

  def test_exact_record_is_accepted_and_beta_is_recomputed(self):
    ok, detail = self._judge(_ema_line())
    self.assertTrue(ok, detail)
    self.assertIn("raw/average medians", detail)

  def test_early_live_record_is_rejected(self):
    ok, detail = self._judge(_ema_line(epoch=5))
    self.assertFalse(ok)
    self.assertIn("expected 6", detail)

  def test_wrong_beta_is_rejected(self):
    ok, detail = self._judge(_ema_line(beta=0.25))
    self.assertFalse(ok)
    self.assertIn("independently expected", detail)

  def test_nonfinite_metric_is_rejected(self):
    ok, detail = self._judge(_ema_line(raw_median="nan"))
    self.assertFalse(ok)
    self.assertIn("finite", detail)

  def test_duplicate_record_is_rejected(self):
    line = _ema_line()
    ok, detail = self._judge(line + "\n" + line)
    self.assertFalse(ok)
    self.assertIn("exactly one", detail)

  def test_claim_without_metrics_is_rejected(self):
    ok, detail = self._judge(
      "ema first-live: epoch=6 schedule=0.02 beta=0.3")
    self.assertFalse(ok)
    self.assertIn("malformed", detail)


class _ToyTwoPhaseModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.mlp = nn.Sequential(nn.Linear(2, 3), nn.Tanh())
    self.head = nn.Linear(3, 1)


def _digest_line(phase, before, after, count=9, extra=""):
  return _phase2_digest_line(
    phase=phase, before=before, after=after,
    parameter_count=count) + extra


class TrunkDigestTest(unittest.TestCase):

  def test_digest_selects_trunk_and_ignores_head_only_change(self):
    torch.manual_seed(42)
    model = _ToyTwoPhaseModel()
    self.assertIs(_training_trunk(model), model.mlp)
    before, count_before = _parameter_digest(_training_trunk(model))
    with torch.no_grad():
      model.head.weight.add_(1.0)
    after_head, count_after = _parameter_digest(_training_trunk(model))
    self.assertEqual(before, after_head)
    self.assertEqual(count_before, count_after)
    with torch.no_grad():
      model.mlp[0].weight.add_(1.0)
    after_trunk, _ = _parameter_digest(_training_trunk(model))
    self.assertNotEqual(before, after_trunk)

  def test_nonfinite_or_empty_trunk_is_refused(self):
    model = _ToyTwoPhaseModel()
    with torch.no_grad():
      model.mlp[0].weight[0, 0] = float("inf")
    with self.assertRaisesRegex(ValueError, "NaN/Inf"):
      _parameter_digest(model.mlp)
    with self.assertRaisesRegex(ValueError, "no parameters"):
      _parameter_digest(nn.Identity())

  def test_joint_change_and_frozen_identity_records_are_accepted(self):
    before = "1" * 64
    after = "2" * 64
    ok, detail = logscan.phase2_trunk_digest(
      text=_digest_line("joint", before, after),
      expected_phase="joint", should_change=True)
    self.assertTrue(ok, detail)
    ok, detail = logscan.phase2_trunk_digest(
      text=_digest_line("head", before, before),
      expected_phase="head", should_change=False)
    self.assertTrue(ok, detail)

  def test_equal_joint_and_changed_frozen_mutations_are_rejected(self):
    before = "a" * 64
    changed = "b" * 64
    ok, detail = logscan.phase2_trunk_digest(
      text=_digest_line("joint", before, before),
      expected_phase="joint", should_change=True)
    self.assertFalse(ok)
    self.assertIn("must be different", detail)
    ok, detail = logscan.phase2_trunk_digest(
      text=_digest_line("head", before, changed),
      expected_phase="head", should_change=False)
    self.assertFalse(ok)
    self.assertIn("must be identical", detail)

  def test_printed_changed_flag_is_not_trusted(self):
    line = _digest_line("joint", "a" * 64, "a" * 64,
                        extra=" changed=true")
    ok, detail = logscan.phase2_trunk_digest(
      text=line, expected_phase="joint", should_change=True)
    self.assertFalse(ok)
    self.assertIn("malformed", detail)

  def test_duplicate_or_wrong_phase_record_is_rejected(self):
    line = _digest_line("joint", "a" * 64, "b" * 64)
    ok, detail = logscan.phase2_trunk_digest(
      text=line + "\n" + line,
      expected_phase="joint", should_change=True)
    self.assertFalse(ok)
    self.assertIn("exactly one", detail)
    ok, detail = logscan.phase2_trunk_digest(
      text=line, expected_phase="head", should_change=True)
    self.assertFalse(ok)
    self.assertIn("expected 'head'", detail)


if __name__ == "__main__":
  unittest.main()
