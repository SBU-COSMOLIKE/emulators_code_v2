"""CPU checks for strict polynomial-emulator fit acceptance.

The polynomial base is allowed into a saved emulator only when at least one
output mode has finite leave-one-out error below the configured limit. These
tests prove that a rejected fit writes no artifact and that the greedy term
search stops when no unused polynomial column remains.
"""

import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from emulator import fixed_facts
from emulator.activations import make_activation
from emulator.designs.blocks import make_norm
from emulator.designs import pce as pce_module
from emulator.designs.pce import PCEEmulator, select_lars_loo
from emulator.designs.plain import ResMLP
from emulator.experiment import validate_pce
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry
from emulator.results import rebuild_emulator, save_emulator
from ai.gates.checks import gsv_bitwise_drift, logscan


CPU = torch.device("cpu")


def _smooth_training_data():
  """Return two exactly low-degree output patterns on one input axis."""
  x = np.linspace(-1.0, 1.0, 31)
  inputs = torch.tensor(x[:, None], dtype=torch.float32)
  targets = torch.tensor(np.stack((
    1.0 + 2.0 * x,
    -0.5 + 0.7 * x + 0.3 * x ** 2,
  ), axis=1), dtype=torch.float32)
  return x, inputs, targets


def _fit(inputs, targets, **overrides):
  """Fit one small deterministic PCE with caller-selected changes."""
  options = {
    "p_max": 2,
    "r_max": 1,
    "q": 1.0,
    "k_max": 2,
    "loo_max": 0.05,
    "max_terms": 20,
    "max_fail": 2,
    "silent": True,
  }
  options.update(overrides)
  return PCEEmulator.from_training(CPU, inputs, targets, **options)


def _artifact_training_data():
  """Return two low-degree surfaces on a five-by-five input grid."""
  axis = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
  first, second = np.meshgrid(axis, axis, indexing="ij")
  inputs = np.column_stack((first.ravel(), second.ravel())).astype(np.float32)
  targets = np.column_stack((
    1.0 + 0.75 * inputs[:, 0] - 0.25 * inputs[:, 1],
    -0.4 + 0.5 * inputs[:, 0] ** 2 + 0.2 * inputs[:, 1],
  )).astype(np.float32)
  return torch.from_numpy(inputs), torch.from_numpy(targets)


def _saved_three_parameter_design(seed):
  """Build the exact float32 design used by a saved three-input PCE."""
  random = np.random.default_rng(seed)
  input_values = random.uniform(-1.0, 1.0, (40, 3)).astype(np.float32)
  inputs = torch.from_numpy(input_values)
  input64 = input_values.astype(np.float64)
  low = input64.min(axis=0)
  high = input64.max(axis=0)
  midpoint = 0.5 * (low + high)
  half_width = 0.5 * (high - low) * 1.05 + 1.0e-12
  saved_low = torch.tensor(midpoint - half_width, dtype=torch.float32)
  saved_high = torch.tensor(midpoint + half_width, dtype=torch.float32)
  mapped = 2.0 * (inputs - saved_low) / (saved_high - saved_low) - 1.0
  multi_index = pce_module.pce_multi_index(
    n_dim=3, p_max=3, r_max=2, q=1.0)
  design = pce_module.pce_design(
    mapped.clamp(-1.0, 1.0),
    torch.tensor(multi_index, dtype=torch.long),
  ).to(torch.float64).numpy()
  return random, inputs, design


def _save_pce_artifact(path_root, pce, resolved_pce):
  """Save one tiny current-format polynomial-base artifact on the CPU."""
  input_count = int(pce.lo.numel())
  output_count = int(pce.Ybar.numel())
  input_names = [f"p{index}" for index in range(input_count)]
  output_names = [f"d{index}" for index in range(output_count)]
  param_geometry = ParamGeometry(
    device=CPU,
    names=input_names,
    center=np.zeros(input_count),
    evecs=np.eye(input_count),
    sqrt_ev=np.ones(input_count),
  )
  geometry = ScalarGeometry(
    device=CPU,
    names=output_names,
    center=np.zeros(output_count),
    scale=np.ones(output_count),
  )
  block_options = {
    "act": make_activation("H", n_gates=3),
    "norm": make_norm("affine"),
  }
  torch.manual_seed(177)
  model = ResMLP(
    input_dim=input_count,
    output_dim=output_count,
    int_dim_res=4,
    n_blocks=1,
    block_opts=block_options,
  ).to(CPU)
  model_recipe = {
    "cls": "emulator.designs.plain.ResMLP",
    "name": "resmlp",
    "ia": None,
    "input_dim": input_count,
    "output_dim": output_count,
    "compile_mode": None,
    "needs_geom": False,
    "kwargs": {
      "int_dim_res": 4,
      "n_blocks": 1,
      "block_opts": {
        "act": {"type": "H", "n_gates": 3},
        "norm": "affine",
      },
    },
  }
  histories = {
    "train_losses": [0.1],
    "val_medians": [0.1],
    "val_means": [0.1],
    "val_fracs": [torch.tensor([0.5])],
    "thresholds": torch.tensor([1.0]),
  }
  config = {
    "data": {},
    "pce": dict(resolved_pce),
    "train_args": {"nepochs": 1},
  }
  return save_emulator(
    path_root=str(path_root),
    model=model,
    param_geometry=param_geometry,
    geometry=geometry,
    config=config,
    histories=histories,
    train_args=config["train_args"],
    attrs={"rescale": "none"},
    pce=pce,
    pce_form="residual",
    resolved_train={"nepochs": 1},
    resolved_model=model_recipe,
    composition_mode="npce",
    transfer_refined=False,
    resolved_pce=dict(resolved_pce),
    resolved_transfer=None,
    facts_yaml=fixed_facts.synthetic_sidecar(
      names=input_names,
      label="pce-strict-control",
      family="scalar",
      support=None,
    ),
  )


class StrictPCESelectionTests(unittest.TestCase):
  """Check fit refusal, unique support, finite evidence, and rebuild."""

  def test_smooth_control_passes_and_state_rebuild_is_identical(self):
    """Two accepted polynomial modes survive the ordinary state round trip."""
    _, inputs, targets = _smooth_training_data()
    fitted = _fit(inputs, targets)
    prediction = fitted(inputs)

    self.assertEqual(fitted.C.shape[1], 2)
    self.assertTrue(torch.isfinite(prediction).all())
    self.assertLess(torch.max(torch.abs(prediction - targets)).item(), 1.0e-5)

    saved = {
      name: value.detach().cpu().numpy()
      for name, value in fitted.state().items()
    }
    rebuilt = PCEEmulator.from_state(saved, CPU)
    rebuilt_prediction = rebuilt(inputs)
    self.assertTrue(torch.equal(prediction, rebuilt_prediction))
    self.assertEqual(set(rebuilt.state()),
                     {"lo", "hi", "multi_index", "C", "Vk", "Ybar"})

  def test_valid_pce_survives_real_artifact_save_and_rebuild(self):
    """A passing polynomial base survives the complete artifact round trip."""
    inputs, targets = _artifact_training_data()
    resolved_pce = {
      "form": "residual",
      "p_max": 2,
      "r_max": 1,
      "q": 1.0,
      "k_max": 2,
      "loo_max": 1.0e-8,
      "max_terms": 20,
      "max_fail": 2,
    }
    fit_options = {
      key: value for key, value in resolved_pce.items() if key != "form"
    }
    fitted = PCEEmulator.from_training(
      CPU, inputs, targets, silent=True, **fit_options)
    self.assertEqual(tuple(fitted.multi_index.shape), (5, 2))
    self.assertEqual(tuple(fitted.C.shape), (5, 2))
    self.assertEqual(tuple(fitted.Vk.shape), (2, 2))
    self.assertLess(
      torch.max(torch.abs(fitted(inputs) - targets)).item(), 2.0e-7)

    with tempfile.TemporaryDirectory(prefix="pce-artifact-") as tmp:
      root = Path(tmp) / "accepted-pce"
      emul_path, h5_path = _save_pce_artifact(
        root, fitted, resolved_pce)
      self.assertTrue(Path(emul_path).is_file())
      self.assertTrue(Path(h5_path).is_file())
      _, _, _, info = rebuild_emulator(
        str(root), CPU, compile_model=False)
      rebuilt = info["pce_base"]

    self.assertEqual(info["composition_mode"], "npce")
    self.assertEqual(info["pce_form"], "residual")
    self.assertIsNotNone(rebuilt)
    for name, live_value in fitted.state().items():
      self.assertTrue(
        torch.equal(live_value, rebuilt.state()[name]), msg=name)
    self.assertTrue(torch.equal(fitted(inputs), rebuilt(inputs)))

  def test_no_mode_below_loo_max_refuses_and_writes_no_artifact(self):
    """A human-checkable rejected fit names its evidence and writes nothing."""
    x = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    inputs = torch.from_numpy(x[:, None])
    targets = torch.from_numpy((x * x)[:, None])
    resolved_pce = {
      "form": "residual",
      "p_max": 1,
      "r_max": 1,
      "q": 1.0,
      "k_max": 1,
      "loo_max": 1.0e-6,
      "max_terms": 8,
      "max_fail": 1,
    }
    fit_options = {
      key: value for key, value in resolved_pce.items() if key != "form"
    }
    with tempfile.TemporaryDirectory(prefix="pce-refusal-") as tmp:
      root = Path(tmp) / "rejected-pce"
      with self.assertRaisesRegex(
          ValueError,
          r"no mode passed.*loo_max=1e-06.*best attempted LOO=1\.19008.*"
          r"mode 0"):
        fitted = PCEEmulator.from_training(
          CPU, inputs, targets, silent=True, **fit_options)
        _save_pce_artifact(root, fitted, resolved_pce)
      self.assertEqual(list(Path(tmp).iterdir()), [])

  def test_selection_stops_when_every_candidate_is_active(self):
    """Exact one- and two-column fits never reuse an exhausted candidate."""
    one_design = np.ones((6, 1), dtype=np.float64)
    one_target = np.array(
      [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float64)
    support, coefficients, loo = select_lars_loo(
      one_design, one_target, max_terms=8, patience=10)
    np.testing.assert_array_equal(support, [0])
    np.testing.assert_allclose(coefficients, [0.0], atol=1.0e-12)
    self.assertAlmostEqual(loo, 36.0 / 25.0, places=10)

    x = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
    two_design = np.column_stack((np.ones(4), x))
    two_target = np.array([-2.0, -2.0, -1.0, -1.0])
    support, coefficients, loo = select_lars_loo(
      two_design, two_target, max_terms=8, patience=10)
    np.testing.assert_array_equal(support, [0, 1])
    np.testing.assert_allclose(coefficients, [-1.5, 0.6], atol=1.0e-8)
    self.assertAlmostEqual(loo, 0.589569161, places=8)

    saturated_design = np.array([[1.0, -1.0], [1.0, 1.0]])
    saturated_target = np.array([-1.0, 1.0])
    support, _, loo = select_lars_loo(
      saturated_design, saturated_target, max_terms=2, patience=10)
    np.testing.assert_array_equal(support, [0])
    self.assertAlmostEqual(loo, 4.0, places=8)

    zero_column_design = np.column_stack((
      np.ones(6), np.zeros(6), np.linspace(-1.0, 1.0, 6)))
    support, _, _ = select_lars_loo(
      zero_column_design, np.linspace(-1.0, 1.0, 6),
      max_terms=8, patience=10)
    np.testing.assert_array_equal(support, [0, 2])

  def test_selection_rejects_invalid_arrays_and_limits(self):
    """Malformed or nonfinite selection inputs fail before matrix fitting."""
    x = np.linspace(-1.0, 1.0, 7)
    design = np.column_stack((np.ones_like(x), x))
    target = np.sin(x)
    cases = (
      ("nonfinite design", np.where(design == 1.0, np.nan, design),
       target, 3, 2, "finite"),
      ("nonfinite target", design, np.where(x == x[0], np.inf, target),
       3, 2, "finite"),
      ("complex design", design.astype(complex), target,
       3, 2, "real numerical"),
      ("flat design", design[:, 0], target, 3, 2, "2-dimensional"),
      ("column target", design, target[:, None], 3, 2, "1-dimensional"),
      ("row mismatch", design, target[:-1], 3, 2, "row count"),
      ("one row", design[:1], target[:1], 3, 2, "at least two"),
      ("no candidates", np.empty((x.size, 0)), target,
       3, 2, "one candidate"),
      ("zero max terms", design, target, 0, 2, "positive integer"),
      ("Boolean max terms", design, target, True, 2, "positive integer"),
      ("zero patience", design, target, 3, 0, "positive integer"),
    )
    for label, bad_design, bad_target, max_terms, patience, message in cases:
      with self.subTest(case=label), self.assertRaisesRegex(ValueError, message):
        select_lars_loo(
          bad_design, bad_target, max_terms=max_terms, patience=patience)

    huge_target = np.array([1.0e308, -1.0e308] * 4)
    with self.assertRaisesRegex(ValueError, "target variance"):
      select_lars_loo(
        np.column_stack((np.ones(8), np.linspace(-1.0, 1.0, 8))),
        huge_target)

  def test_training_rejects_invalid_shapes_values_and_fit_limits(self):
    """The complete fit validates data and direct-call options before SVD."""
    _, inputs, targets = _smooth_training_data()
    cases = [
      ("flat inputs", inputs[:, 0], targets, {}, "2-dimensional"),
      ("row mismatch", inputs[:-1], targets, {}, "row counts must match"),
      ("one row", inputs[:1], targets[:1], {}, "at least two rows"),
      ("no input columns", inputs[:, :0], targets, {}, "one input column"),
      ("no target columns", inputs, targets[:, :0], {}, "one target column"),
      ("nonfinite loo limit", inputs, targets,
       {"loo_max": float("nan")}, "finite and strictly positive"),
      ("infinite loo limit", inputs, targets,
       {"loo_max": float("inf")}, "finite and strictly positive"),
      ("zero term limit", inputs, targets,
       {"max_terms": 0}, "positive integer"),
    ]
    for value in (float("nan"), float("inf"), -float("inf")):
      bad_inputs = inputs.clone()
      bad_inputs[0, 0] = value
      cases.append((
        f"input value {value!r}", bad_inputs, targets, {}, "finite"))
      bad_targets = targets.clone()
      bad_targets[0, 0] = value
      cases.append((
        f"target value {value!r}", inputs, bad_targets, {}, "finite"))
    for label, bad_inputs, bad_targets, options, message in cases:
      with self.subTest(case=label), self.assertRaisesRegex(ValueError, message):
        _fit(bad_inputs, bad_targets, **options)

    large_inputs = torch.tensor(
      [[1.0e10], [1.0e10 + 1.0], [1.0e10 + 2.0]],
      dtype=torch.float64)
    large_targets = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    with self.assertRaisesRegex(ValueError, "distinct after conversion"):
      _fit(large_inputs, large_targets, p_max=1, k_max=1)

    overflow_targets = torch.tensor(
      [[-1.0e39], [0.0], [1.0e39]], dtype=torch.float64)
    with self.assertRaisesRegex(ValueError, "nonfinite in float32"):
      _fit(torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64),
           overflow_targets, p_max=1, k_max=1, loo_max=1.0)

  def test_nonfinite_or_duplicate_selection_result_refuses(self):
    """The fit boundary does not trust an invalid selector return value."""
    _, inputs, targets = _smooth_training_data()
    returns = (
      ("nonfinite loo", (np.array([0]), np.array([1.0]), np.nan),
       "finite nonnegative"),
      ("positive infinite loo", (
        np.array([0]), np.array([1.0]), np.inf), "finite nonnegative"),
      ("negative infinite loo", (
        np.array([0]), np.array([1.0]), -np.inf), "finite nonnegative"),
      ("nonfinite coefficient", (
        np.array([0]), np.array([np.inf]), 0.0), "finite coefficient"),
      ("missing coefficient", (
        np.array([0]), None, 0.0), "finite coefficient"),
      ("floating support", (
        np.array([0.0]), np.array([1.0]), 0.0), "unique in-range"),
      ("duplicate support", (
        np.array([0, 0]), np.array([1.0, 2.0]), 0.0), "unique in-range"),
      ("out-of-range support", (
        np.array([100]), np.array([1.0]), 0.0), "unique in-range"),
      ("mismatched support", (
        np.array([0, 1]), np.array([1.0]), 0.0), "finite coefficient"),
    )
    for label, selector_return, message in returns:
      with self.subTest(case=label), mock.patch.object(
          pce_module, "select_lars_loo", return_value=selector_return), \
          self.assertRaisesRegex(ValueError, message):
        _fit(inputs, targets)

  def test_accuracy_limit_is_strict_and_uses_saved_precision(self):
    """Equality fails; saved coefficients and bounds govern acceptance."""
    _, inputs, targets = _smooth_training_data()
    with mock.patch.object(
        pce_module, "select_lars_loo",
        return_value=(np.array([0]), np.array([0.0]), 0.01)), \
        mock.patch.object(pce_module, "_fixed_fit_loo", return_value=0.05), \
        self.assertRaisesRegex(ValueError, "no mode passed"):
      _fit(inputs, targets[:, :1], k_max=1, loo_max=0.05, max_fail=1)

    # Large coefficients can cancel in float64 yet leave a visible residual
    # in the float32 matrix multiplication performed by the saved base. The
    # threshold lies between those two results, so a promoted dot product
    # would accept and the real saved-format calculation must refuse.
    random, inputs, design = _saved_three_parameter_design(seed=0)
    raw_coefficients = random.normal(0.0, 1.0e6, design.shape[1])
    target = (design @ raw_coefficients).astype(np.float32)
    targets = torch.from_numpy(target[:, None])
    centered = target.astype(np.float64) - target.astype(np.float64).mean()
    support, beta, _ = select_lars_loo(
      design, centered, max_terms=20, patience=10)
    self.assertEqual(support.size, design.shape[1])
    saved_beta = beta.astype(np.float32).astype(np.float64)
    saved_loo = pce_module._fixed_fit_loo(
      design, centered, support, saved_beta)
    active = design[:, support]
    normal = active.T @ active + 1.0e-10 * np.eye(support.size)
    inverse = np.linalg.inv(normal)
    leverage = np.einsum("ni,ij,nj->n", active, inverse, active)
    full_beta = np.zeros(design.shape[1], dtype=np.float64)
    full_beta[support] = saved_beta
    promoted_residual = centered - design @ full_beta
    promoted_loo = np.mean(
      (promoted_residual / (1.0 - leverage)) ** 2
    ) / np.var(centered)
    accuracy_limit = 0.5 * (promoted_loo + saved_loo)
    self.assertLess(promoted_loo, accuracy_limit)
    self.assertGreater(saved_loo, accuracy_limit)
    with self.assertRaisesRegex(ValueError, "no mode passed"):
      _fit(inputs, targets, p_max=3, r_max=2, q=1.0, k_max=1,
           loo_max=accuracy_limit, max_terms=20, max_fail=1)

    # At this large offset, the padded float64 bounds round to different
    # float32 values. Fitting against the pre-round bounds used to pass while
    # the saved polynomial missed its own limit. The current fit uses the
    # stored bounds from the start, so its saved prediction and PRESS agree.
    offset_inputs = torch.tensor(
      [[1.0e8 + 8.0 * index] for index in range(6)], dtype=torch.float32)
    offset_targets = torch.arange(6, dtype=torch.float32)[:, None]
    offset_fit = _fit(
      offset_inputs, offset_targets, p_max=1, k_max=1, max_terms=2,
      loo_max=1.0e-4, max_fail=1)
    offset_prediction = offset_fit(offset_inputs)
    self.assertLess(
      torch.max(torch.abs(offset_prediction - offset_targets)).item(),
      1.0e-5)
    mapped = 2.0 * (offset_inputs - offset_fit.lo) / (
      offset_fit.hi - offset_fit.lo) - 1.0
    actual_design = pce_module.pce_design(
      mapped.clamp(-1.0, 1.0), offset_fit.multi_index,
    ).to(torch.float64).numpy()
    support = np.flatnonzero(offset_fit.C[:, 0].numpy())
    mode_target = (
      (offset_targets - offset_fit.Ybar) @ offset_fit.Vk[:, :1]
    )[:, 0].to(torch.float64).numpy()
    actual_loo = pce_module._fixed_fit_loo(
      actual_design, mode_target, support,
      offset_fit.C[support, 0].to(torch.float64).numpy())
    self.assertLess(actual_loo, 1.0e-4)

  def test_all_saved_modes_are_checked_in_one_matrix_multiplication(self):
    """Two columns cannot pass separately and fail when saved together."""
    random, inputs, design = _saved_three_parameter_design(seed=1)
    raw_coefficients = random.normal(
      0.0, 1.0e6, (design.shape[1], 2))
    target = (design @ raw_coefficients).astype(np.float32)
    targets = torch.from_numpy(target)
    target64 = target.astype(np.float64)
    centered = target64 - target64.mean(axis=0)
    _, _, output_modes = np.linalg.svd(centered, full_matrices=False)

    columns = []
    supports = []
    betas = []
    mode_targets = []
    separate_loos = []
    for mode in range(2):
      mode_target = centered @ output_modes[mode]
      support, beta, _ = select_lars_loo(
        design, mode_target, max_terms=20, patience=10)
      saved_beta = beta.astype(np.float32).astype(np.float64)
      column = np.zeros(design.shape[1], dtype=np.float64)
      column[support] = saved_beta
      columns.append(column)
      supports.append(support)
      betas.append(saved_beta)
      mode_targets.append(mode_target)
      separate_loos.append(pce_module._fixed_fit_loo(
        design, mode_target, support, saved_beta))

    coefficient_matrix = np.stack(columns, axis=1).astype(np.float32)
    with torch.no_grad():
      joint_prediction = (
        torch.from_numpy(design.astype(np.float32))
        @ torch.from_numpy(coefficient_matrix)
      ).to(torch.float64).numpy()
    joint_loos = [
      pce_module._fixed_fit_loo(
        design, mode_targets[mode], supports[mode], betas[mode],
        saved_prediction=joint_prediction[:, mode])
      for mode in range(2)
    ]
    accuracy_limit = 0.5 * (max(separate_loos) + max(joint_loos))
    self.assertLess(max(separate_loos), accuracy_limit)
    self.assertGreater(max(joint_loos), accuracy_limit)

    fitted = _fit(
      inputs, targets, p_max=3, r_max=2, q=1.0, k_max=2,
      loo_max=accuracy_limit, max_terms=20, max_fail=2)
    self.assertEqual(fitted.C.shape[1], 1)
    np.testing.assert_allclose(
      fitted.Vk[:, 0].numpy(), output_modes[0], rtol=1.0e-6, atol=1.0e-6)
    final_prediction = (
      torch.from_numpy(design.astype(np.float32)) @ fitted.C.cpu()
    )[:, 0].to(torch.float64).numpy()
    final_support = np.flatnonzero(fitted.C[:, 0].cpu().numpy())
    final_loo = pce_module._fixed_fit_loo(
      design, mode_targets[0], final_support,
      fitted.C[final_support, 0].to(torch.float64).cpu().numpy(),
      saved_prediction=final_prediction)
    self.assertLess(final_loo, accuracy_limit)

  def test_configuration_rejects_nonfinite_loo_max(self):
    """YAML validation rejects NaN and infinity instead of accepting both."""
    for value in (float("nan"), float("inf"), -float("inf")):
      with self.subTest(value=value), self.assertRaisesRegex(
          ValueError, "loo_max must be finite and > 0"):
        validate_pce(
          {"form": "residual", "loo_max": value}, diagonal=True)

  def test_npce_sweep_gate_requires_exact_finite_results(self):
    """Two result-shaped lines cannot hide failed or duplicated sweep points."""
    valid = (
      "pce: form residual\n"
      "  N_train     2000  f(>0.2) 0.1250  (gpu 1, 4s)\n"
      "  N_train     1000  f(>0.2) 0.7500  (gpu 0, 3s)\n"
    )
    ok, detail = logscan.finite_sweep_points(
      valid, expected_sizes=(1000, 2000), expected_threshold=0.2)
    self.assertTrue(ok, detail)
    mutations = (
      ("not a number", valid.replace("0.7500", "nan"), "invalid fraction"),
      ("infinity", valid.replace("0.7500", "inf"), "invalid fraction"),
      ("duplicate size", valid.replace("2000", "1000"), "duplicate"),
      ("missing size", valid.splitlines()[0] + "\n" + valid.splitlines()[1],
       "missing"),
      ("out of range", valid.replace("0.7500", "1.2500"),
       "invalid fraction"),
      ("wrong threshold", valid.replace("f(>0.2)", "f(>0.3)", 1),
       "threshold"),
      ("worker failure", valid + "[gpu 0] N_train 1000 failed: fit refused\n",
       "failed sweep point"),
      ("malformed result", valid + "N_train 1000 f(>0.2)\n", "malformed"),
    )
    for label, text, message in mutations:
      with self.subTest(case=label):
        ok, detail = logscan.finite_sweep_points(
          text, expected_sizes=(1000, 2000), expected_threshold=0.2)
        self.assertFalse(ok)
        self.assertIn(message, detail)

  def test_save_rebuild_gate_pins_a_serialization_only_loo_limit(self):
    """The artifact gate does not inherit the scientific fitting default."""
    config = gsv_bitwise_drift.tiny_config(Path("/tmp"), pce=True)
    self.assertEqual(config["pce"], {"form": "residual", "loo_max": 1.1})
    design = np.ones((200, 1), dtype=np.float64)
    target = np.linspace(-1.0, 1.0, 200)
    _, _, loo = select_lars_loo(design, target)
    self.assertLess(loo, 1.1)


if __name__ == "__main__":
  unittest.main()
