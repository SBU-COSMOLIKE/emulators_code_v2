#!/usr/bin/env python3
"""CPU numerical witnesses for the DIDACTICS-62 training gates.

The workstation smokes prove that full scientific configurations run on the
configured GPU.  This companion proves three deterministic numerical rules
without pretending that a CPU fixture ran those scientific jobs:

* the production annealing function reaches known values and is continuous;
* the shipped ReLU/Tanh and norm factories learn a small nonlinear table;
* a constant or prematurely started schedule, dead network, or mean-only
  predictor does not satisfy the same judge.

Each command selects one gate and prints only that gate's ``##AID`` records.
The board runner collects those records when it later runs the full gate.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.losses.core import anneal_value


REPO = Path(__file__).resolve().parents[3]

BEHAVIOR_AIDS = {
  "berhu-anneal": ("berhu-anneal.schedule-behavior",),
  "ema-anneal": ("ema-anneal.schedule-behavior",),
  "relu-tanh-norm": ("relu-tanh-norm.relu-finite-descent",
                     "relu-tanh-norm.tanh-finite-descent"),
}


def _load_yaml(relative_path):
  """Read one shipped gate YAML as a mapping."""
  path = REPO / relative_path
  with path.open(encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
  if not isinstance(value, dict):
    raise ValueError(str(relative_path) + " must contain a YAML mapping")
  return value


def _unit_schedule(block):
  """Add the fixed zero-to-one endpoints used by BerHu and EMA."""
  return {
    "shape": block["shape"],
    "start": 0.0,
    "end": 1.0,
    "hold_epochs": block["hold_epochs"],
    "anneal_epochs": block["anneal_epochs"],
  }


def schedule_behavior(opts):
  """Check the production cosine schedule against independent known values.

  The two affected configurations deliberately use hold=5 and ramp=10.  The
  fixed values below therefore do not repeat ``anneal_value``'s formula.  The
  small probes on either side of both joins catch a discontinuous branch even
  when all integer epochs happen to look plausible.

  Returns:
    ``(ok, detail)``.
  """
  expected_shape = (opts.get("shape") == "cosine"
                    and opts.get("start") == 0.0
                    and opts.get("end") == 1.0
                    and opts.get("hold_epochs") == 5
                    and opts.get("anneal_epochs") == 10)
  if not expected_shape:
    return (False, "expected cosine 0->1, hold 5, ramp 10; got " + repr(opts))

  known = {
    4: 0.0,                         # before the hold boundary
    5: 0.0,                         # boundary
    6: 0.024471741852423234,        # strict interior
    10: 0.5,                        # midpoint
    15: 1.0,                        # endpoint
    16: 1.0,                        # after the ramp
  }
  observed = {}
  for epoch, expected in known.items():
    value = anneal_value(epoch=epoch, opts=opts)
    observed[epoch] = value
    if (not math.isfinite(value)
        or not math.isclose(value, expected,
                            rel_tol=1.0e-12, abs_tol=1.0e-14)):
      return (False, "epoch " + str(epoch) + " expected " + repr(expected)
              + ", observed " + repr(value))

  epsilon = 1.0e-6
  joins = ((5.0 - epsilon, 5.0 + epsilon, 0.0),
           (15.0 - epsilon, 15.0 + epsilon, 1.0))
  for left_epoch, right_epoch, expected in joins:
    left = anneal_value(epoch=left_epoch, opts=opts)
    right = anneal_value(epoch=right_epoch, opts=opts)
    if (not math.isfinite(left) or not math.isfinite(right)
        or abs(left - expected) > 1.0e-11
        or abs(right - expected) > 1.0e-11
        or abs(left - right) > 1.0e-11):
      return (False, "schedule is not continuous around "
              + repr((left_epoch + right_epoch) / 2.0)
              + ": left=" + repr(left) + ", right=" + repr(right))

  detail = ("epochs 4/5/6/10/15/16="
            + "/".join(format(observed[epoch], ".12g")
                       for epoch in (4, 5, 6, 10, 15, 16))
            + "; both joins continuous")
  return (True, detail)


class _TinyActivationRegressor(nn.Module):
  """One hidden layer built from the same activation and norm factories."""

  def __init__(self, activation_name, norm_name, width=12):
    super().__init__()
    activation = make_activation(activation_name)
    norm = make_norm(norm_name)
    self.first = nn.Linear(1, width)
    self.norm = norm(width)
    self.activation = activation(width)
    self.last = nn.Linear(width, 1)

  def forward(self, value):
    value = self.first(value)
    value = self.norm(value)
    value = self.activation(value)
    return self.last(value)


def judge_learning(initial_loss, final_loss, mean_only_loss):
  """Require finite descent and a result clearly better than a constant.

  The factor of one half is a broad separation, not a fitted golden number.
  Both real witnesses finish far below it.  A dead network fails the strict
  descent check; a predictor that returns only the target mean fails the
  independent baseline check.
  """
  values = (initial_loss, final_loss, mean_only_loss)
  if not all(math.isfinite(value) for value in values):
    return (False, "loss values must be finite: " + repr(values))
  if not final_loss < initial_loss:
    return (False, "loss did not descend: initial " + repr(initial_loss)
            + ", final " + repr(final_loss))
  if not final_loss < 0.5 * mean_only_loss:
    return (False, "final loss " + repr(final_loss)
            + " did not beat half the mean-only loss "
            + repr(mean_only_loss))
  return (True, "initial=" + format(initial_loss, ".8g")
          + " final=" + format(final_loss, ".8g")
          + " mean-only=" + format(mean_only_loss, ".8g"))


def _factory_probe(activation_name, norm_name):
  """Check exact parameter-free activation values and identity norm starts."""
  activation = make_activation(activation_name)(3)
  if activation_name == "relu":
    source = torch.tensor([-2.0, 0.0, 3.0])
    expected = torch.tensor([0.0, 0.0, 3.0])
  elif activation_name == "tanh":
    source = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.tensor([-0.7615941762924194, 0.0,
                             0.7615941762924194])
  else:
    return (False, "unsupported activation probe " + repr(activation_name))
  actual = activation(source)
  if not torch.allclose(actual, expected, rtol=0.0, atol=1.0e-7):
    return (False, activation_name + " probe expected " + repr(expected)
            + ", observed " + repr(actual))

  norm = make_norm(norm_name)(3)
  norm_source = torch.tensor([[-2.0, 0.0, 3.0], [4.0, -5.0, 6.0]])
  norm_actual = norm(norm_source)
  if not torch.equal(norm_actual, norm_source):
    return (False, norm_name + " norm must start as the identity")
  return (True, activation_name + " exact values and " + norm_name
          + " identity initialization")


def activation_learning(activation_name, norm_name, seed):
  """Train the production factories on one fixed, nonlinear CPU table."""
  torch.manual_seed(seed)
  model = _TinyActivationRegressor(activation_name, norm_name).double()
  x = torch.linspace(-2.0, 2.0, 129, dtype=torch.float64).reshape(-1, 1)
  target = 0.35 * x.square() + 1.1 * x - 0.25
  mean_only_loss = float(torch.mean((target - target.mean()).square()))

  optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
  with torch.no_grad():
    initial_loss = float(torch.mean((model(x) - target).square()))
  for _ in range(240):
    prediction = model(x)
    loss = torch.mean((prediction - target).square())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
  with torch.no_grad():
    final_loss = float(torch.mean((model(x) - target).square()))

  ok, detail = judge_learning(initial_loss, final_loss, mean_only_loss)
  probe_ok, probe_detail = _factory_probe(activation_name, norm_name)
  return (ok and probe_ok, detail + "; " + probe_detail)


def _emit(aid, ok, detail):
  """Print one board-readable result line."""
  print("##AID " + aid + " " + ("PASS" if ok else "FAIL") + " " + detail)
  return ok


def _run_schedule(gate_name):
  if gate_name == "berhu-anneal":
    cfg = _load_yaml("ai/gates/configs/berhu-anneal-config.yaml")
    block = cfg["train_args"]["head"]["loss"]["berhu"]["anneal"]
  else:
    cfg = _load_yaml("ai/gates/configs/ema-anneal-config.yaml")
    block = cfg["train_args"]["ema"]["anneal"]
  ok, detail = schedule_behavior(_unit_schedule(block))
  return _emit(BEHAVIOR_AIDS[gate_name][0], ok, detail)


def _run_activations():
  relu_cfg = _load_yaml(
    "ai/gates/configs/relu-tanh-norm-per-feature.yaml")
  tanh_cfg = _load_yaml("ai/gates/configs/relu-tanh-norm-affine.yaml")
  cases = ((relu_cfg, "relu", "per_feature", 1701,
            BEHAVIOR_AIDS["relu-tanh-norm"][0]),
           (tanh_cfg, "tanh", "affine", 1702,
            BEHAVIOR_AIDS["relu-tanh-norm"][1]))
  all_ok = True
  for cfg, expected_activation, expected_norm, seed, aid in cases:
    model_cfg = cfg["train_args"]["model"]
    observed_activation = model_cfg["activation"]["type"]
    observed_norm = model_cfg["norm"]
    if (observed_activation != expected_activation
        or observed_norm != expected_norm):
      ok = False
      detail = ("config expected " + expected_activation + "+" + expected_norm
                + ", observed " + str(observed_activation) + "+"
                + str(observed_norm))
    else:
      ok, detail = activation_learning(expected_activation, expected_norm, seed)
    all_ok = _emit(aid, ok, detail) and all_ok
  return all_ok


def main(argv=None):
  parser = argparse.ArgumentParser(
    description="Run one CPU numerical witness used by a training gate.")
  parser.add_argument("--gate", required=True, choices=tuple(BEHAVIOR_AIDS))
  args = parser.parse_args(argv)
  if args.gate in ("berhu-anneal", "ema-anneal"):
    ok = _run_schedule(args.gate)
  else:
    ok = _run_activations()
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
