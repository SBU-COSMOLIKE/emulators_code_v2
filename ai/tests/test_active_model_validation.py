"""CPU checks for model settings that must refuse before model building.

The selected model block is allowed to reach a constructor only after its
Booleans, counts, correction gate, activation, and output layout have an
unambiguous meaning.  These checks use tiny plain CNN and transformer models
to show both sides of that boundary: valid settings build the requested head,
while invalid settings fail without relying on Python ``assert`` statements.
"""

import copy
import os
from pathlib import Path
import re
import sys
import textwrap
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from emulator.activations import (
  GatedPowerActivation,
  PowerGatedActivation,
  make_activation,
)
from emulator import data_staging
from emulator import experiment as experiment_module
from emulator.designs.ia import TemplateResCNN
from emulator.designs.plain import ResCNN, ResMLP, ResTRF
from emulator.experiment import validate_active_model_values


def _train_args(name, *, activation="H", head_values=None,
                inactive_values=None):
  """Return one small parsed model configuration for a selected design."""
  values = {
    "model": {
      "name": name,
      "mlp": {"width": 6, "n_blocks": 1},
      "activation": {"type": activation, "n_gates": 3},
    },
  }
  if name == "rescnn":
    values["model"]["cnn"] = dict(head_values or {})
    if inactive_values is not None:
      values["model"]["trf"] = dict(inactive_values)
  elif name == "restrf":
    values["model"]["trf"] = dict(head_values or {})
    if inactive_values is not None:
      values["model"]["cnn"] = dict(inactive_values)
  return values


def _cnn_geometry():
  """Return two length-two bins with a valid xi-plus/xi-minus split."""
  return types.SimpleNamespace(
    bin_sizes=[2, 2],
    pm_kept=[0, 0, 1, 1],
  )


def _public_grid_config():
  """Return a complete background-run config that needs no real data yet."""
  return {
    "data": {
      "grid": {
        "quantity": "Hubble",
        "units": "km/s/Mpc",
        "law": "none",
        "z_file": "background_z.npy",
      },
      "train_dv": "train.npy",
      "val_dv": "val.npy",
      "train_params": "train.1.txt",
      "val_params": "val.1.txt",
      "train_covmat": "train.covmat",
      "n_train": 1,
      "n_val": 1,
      "split_seed": 7,
    },
    "train_args": _train_args(
      "rescnn",
      head_values={
        "kernel_size": 3,
        "rescale_kernel": False,
        "groups": 1,
        "separable": False,
        "film": False,
        "n_blocks": 1,
        "gate_init": 0.1,
      },
    ),
  }


def _trf_geometry():
  """Return two physical tokens, each with width four."""
  return types.SimpleNamespace(bin_sizes=[4, 4])


def _assert_head_step_moves(test_case, model, watched_parameters, inputs):
  """Require one frozen-trunk step to move the zero-initialized layer."""
  model.set_train_phase("head")
  selected = list(watched_parameters())
  before = [parameter.detach().clone() for parameter in selected]
  optimizer = torch.optim.SGD(
    [parameter for parameter in model.parameters()
     if parameter.requires_grad],
    lr=0.05,
  )

  initial = model(inputs).detach()
  offset = torch.linspace(
    0.2, 0.8, initial.numel(), dtype=initial.dtype,
  ).reshape_as(initial)
  target = initial + offset
  loss = torch.mean((model(inputs) - target) ** 2)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()

  finite_nonzero_gradient = False
  for parameter in selected:
    if parameter.grad is not None \
        and torch.isfinite(parameter.grad).all() \
        and torch.count_nonzero(parameter.grad).item() > 0:
      finite_nonzero_gradient = True
  test_case.assertTrue(
    finite_nonzero_gradient,
    "the H-activated correction head received no usable first-step gradient",
  )

  optimizer.step()
  test_case.assertTrue(
    any(not torch.equal(old, new.detach())
        for old, new in zip(before, selected)),
    "the first optimizer step did not move a correction-head parameter",
  )


class ActiveModelValueTests(unittest.TestCase):
  """Check exact parsed values before a selected model is constructed."""

  def test_only_the_selected_head_block_is_validated(self):
    """An unused alternative may stay in a shared YAML without affecting it."""
    inactive = {
      "n_heads": "not an integer",
      "shared_mlp": "false",
      "a_future_setting": "ignored because this head is inactive",
    }
    train_args = _train_args(
      "rescnn",
      head_values={
        "kernel_size": 3,
        "rescale_kernel": False,
        "groups": 1,
        "separable": False,
        "film": False,
        "n_blocks": 2,
        "gate_init": 0.1,
      },
      inactive_values=inactive,
    )
    original = copy.deepcopy(train_args)

    checked = validate_active_model_values(
      train_args=train_args,
      model_cls=ResCNN,
      geom=_cnn_geometry(),
    )

    self.assertEqual(train_args, original)
    self.assertEqual(checked["trunk_activation"],
                     {"type": "H", "n_gates": 3})
    self.assertIsNone(checked["head_pin"])

  def test_norm_and_compile_mode_accept_only_the_documented_names(self):
    """Shared model settings refuse wrong types and unknown spellings."""
    invalid_values = (
      ("norm", True, TypeError),
      ("norm", "batch", ValueError),
      ("compile_mode", False, TypeError),
      ("compile_mode", "fastest", ValueError),
    )
    for setting, value, error_type in invalid_values:
      with self.subTest(setting=setting, value=value):
        train_args = _train_args("resmlp")
        train_args["model"][setting] = value
        with self.assertRaisesRegex(
            error_type, re.escape("model." + setting)):
          validate_active_model_values(
            train_args=train_args,
            model_cls=ResMLP,
          )

    for norm_name in ("affine", "per_feature", "none"):
      with self.subTest(valid_norm=norm_name):
        train_args = _train_args("resmlp")
        train_args["model"]["norm"] = norm_name
        validate_active_model_values(
          train_args=train_args,
          model_cls=ResMLP,
        )
    for compile_mode in (None, "default", "reduce-overhead"):
      with self.subTest(valid_compile_mode=compile_mode):
        train_args = _train_args("resmlp")
        train_args["model"]["compile_mode"] = compile_mode
        validate_active_model_values(
          train_args=train_args,
          model_cls=ResMLP,
        )

  def test_exact_false_values_pass_but_quoted_false_values_refuse(self):
    """A YAML Boolean stays false; text that merely spells false is invalid."""
    cases = (
      (ResCNN, "rescnn", "rescale_kernel"),
      (ResCNN, "rescnn", "separable"),
      (ResCNN, "rescnn", "film"),
      (ResTRF, "restrf", "shared_mlp"),
      (ResTRF, "restrf", "film"),
    )
    for model_cls, name, key in cases:
      with self.subTest(model=name, setting=key, value=False):
        valid = _train_args(name, head_values={key: False})
        validate_active_model_values(
          train_args=valid,
          model_cls=model_cls,
        )
        self.assertIs(valid["model"][model_cls.head_block][key], False)

      with self.subTest(model=name, setting=key, value="false"):
        invalid = _train_args(name, head_values={key: "false"})
        full_path = "model." + model_cls.head_block + "." + key
        with self.assertRaisesRegex(TypeError, re.escape(full_path)):
          validate_active_model_values(
            train_args=invalid,
            model_cls=model_cls,
          )

  def test_public_config_refuses_active_head_before_files_or_device(self):
    """The public loader reports a quoted Boolean before expensive setup."""
    config = _public_grid_config()
    config["train_args"]["model"]["cnn"]["rescale_kernel"] = "false"
    with mock.patch.object(
        experiment_module, "validated_facts_sidecar",
        side_effect=AssertionError("facts files must not be opened")) \
        as facts, mock.patch.object(
          experiment_module, "pick_device",
          side_effect=AssertionError("a device must not be selected")) \
        as device, mock.patch.object(
          experiment_module.warmstart, "load_source",
          side_effect=AssertionError("a saved source must not be opened")) \
        as source, mock.patch.object(
          data_staging.np, "load",
          side_effect=AssertionError("training arrays must not be opened")) \
        as arrays, mock.patch.object(
          experiment_module.ResCNN, "__init__",
          side_effect=AssertionError("a model must not be constructed")) \
        as model:
      with self.assertRaisesRegex(
          TypeError, r"model\.cnn\.rescale_kernel"):
        experiment_module.EmulatorExperiment.from_config(config)

    facts.assert_not_called()
    device.assert_not_called()
    source.assert_not_called()
    arrays.assert_not_called()
    model.assert_not_called()

  def test_width_depth_and_gate_counts_are_exact_integers(self):
    """Booleans, strings, fractions, and nonpositive required counts refuse."""
    cases = (
      (ResMLP, _train_args("resmlp"), ("model", "mlp", "width"), 0),
      (ResMLP, _train_args("resmlp"), ("model", "mlp", "width"), True),
      (ResMLP, _train_args("resmlp"), ("model", "mlp", "width"), "6"),
      (ResMLP, _train_args("resmlp"),
       ("model", "activation", "n_gates"), 0),
      (ResMLP, _train_args("resmlp"),
       ("model", "activation", "n_gates"), False),
      (ResMLP, _train_args("resmlp"),
       ("model", "activation", "n_gates"), "3"),
      (ResMLP, _train_args("resmlp"),
       ("model", "activation", "n_gates"), 2.5),
      (ResCNN, _train_args("rescnn"),
       ("model", "cnn", "kernel_size"), 0),
      (ResCNN, _train_args("rescnn"), ("model", "cnn", "n_blocks"), 0),
      (ResTRF, _train_args("restrf"), ("model", "trf", "n_heads"), 0),
      (ResTRF, _train_args("restrf"),
       ("model", "trf", "n_blocks"), False),
      (ResTRF, _train_args("restrf"),
       ("model", "trf", "n_mlp_blocks"), "2"),
      (ResTRF, _train_args("restrf"),
       ("model", "trf", "n_tokens"), 2.5),
      (ResTRF, _train_args("restrf"),
       ("model", "trf", "n_tokens"), False),
      (ResTRF, _train_args("restrf"),
       ("model", "trf", "n_tokens"), "3"),
      (ResTRF, _train_args("restrf"),
       ("model", "trf", "n_tokens"), 0),
    )
    for model_cls, base, path, invalid_value in cases:
      with self.subTest(path=".".join(path), value=invalid_value):
        candidate = copy.deepcopy(base)
        destination = candidate
        for key in path[:-1]:
          destination = destination[key]
        destination[path[-1]] = invalid_value
        with self.assertRaisesRegex(
            (TypeError, ValueError), re.escape(".".join(path))):
          validate_active_model_values(
            train_args=candidate,
            model_cls=model_cls,
          )

    linear_trunk = _train_args("resmlp")
    linear_trunk["model"]["mlp"]["n_blocks"] = 0
    validate_active_model_values(
      train_args=linear_trunk,
      model_cls=ResMLP,
    )

  def test_gate_must_survive_storage_as_a_finite_nonzero_float32(self):
    """A correction gate must not become zero or nonfinite in the model."""
    invalid_values = (0, -0.0, 1.0e-50, float("nan"), float("inf"),
                      True, "0.1")
    for model_cls, name in ((ResCNN, "rescnn"), (ResTRF, "restrf")):
      for value in invalid_values:
        with self.subTest(model=name, gate_init=value):
          train_args = _train_args(
            name, head_values={"gate_init": value})
          with self.assertRaises((TypeError, ValueError)):
            validate_active_model_values(
              train_args=train_args,
              model_cls=model_cls,
            )

      valid = _train_args(name, head_values={"gate_init": -0.1})
      validate_active_model_values(
        train_args=valid,
        model_cls=model_cls,
      )

  def test_gated_activation_factories_require_an_exact_positive_gate_count(self):
    """The two K-gate families reject values that Python could coerce."""
    for family in ("multigate", "gated_power"):
      for value in (0, False, "3", 1.5):
        with self.subTest(family=family, n_gates=value):
          with self.assertRaisesRegex(
              (TypeError, ValueError), r"activation\.n_gates"):
            make_activation(family, n_gates=value)

      control = make_activation(family, n_gates=2)(4)
      self.assertEqual(control.w.shape, (2, 4))

  def test_relu_is_a_valid_trunk_activation_but_not_an_implicit_head(self):
    """ReLU may shape a trunk, but its zero derivative cannot wake a head."""
    trunk_only = _train_args("resmlp", activation="relu")
    validate_active_model_values(
      train_args=trunk_only,
      model_cls=ResMLP,
    )

    unsafe = _train_args("rescnn", activation="relu")
    with self.assertRaisesRegex(ValueError, "zero-initialized"):
      validate_active_model_values(
        train_args=unsafe,
        model_cls=ResCNN,
      )

    safe = _train_args(
      "rescnn",
      activation="relu",
      head_values={"activation": {"type": "H", "n_gates": 3}},
    )
    safe["trunk_epochs"] = 1
    safe["freeze_trunk"] = True
    checked = validate_active_model_values(
      train_args=safe,
      model_cls=ResCNN,
    )
    self.assertEqual(checked["trunk_activation"]["type"], "relu")
    self.assertEqual(checked["head_activation"]["type"], "H")

  def test_geometry_checks_attention_width_and_physical_cnn_groups(self):
    """The known output layout licenses only compatible heads and grouping."""
    transformer = _train_args(
      "restrf", head_values={"n_heads": 2})
    validate_active_model_values(
      train_args=transformer,
      model_cls=ResTRF,
      geom=_trf_geometry(),
    )
    incompatible = _train_args(
      "restrf", head_values={"n_heads": 3})
    with self.assertRaisesRegex(ValueError, "divide"):
      validate_active_model_values(
        train_args=incompatible,
        model_cls=ResTRF,
        geom=_trf_geometry(),
      )

    grouped = _train_args("rescnn", head_values={"groups": 2})
    validate_active_model_values(
      train_args=grouped,
      model_cls=ResCNN,
      geom=_cnn_geometry(),
    )
    wrong_branch_order = types.SimpleNamespace(
      bin_sizes=[2, 2], pm_kept=[0, 0, 0, 0])
    with self.assertRaisesRegex(ValueError, "xi\\+"):
      validate_active_model_values(
        train_args=grouped,
        model_cls=ResCNN,
        geom=wrong_branch_order,
      )

  def test_sweep_value_is_rechecked_with_the_built_output_layout(self):
    """A later search value cannot bypass attention-width divisibility."""
    experiment = experiment_module.EmulatorExperiment.__new__(
      experiment_module.EmulatorExperiment)
    experiment._finetune = None
    experiment.model_cls = ResTRF
    experiment.activation = "H"
    experiment.ia = None
    experiment.geom = _trf_geometry()
    swept_train_args = _train_args(
      "restrf", head_values={"n_heads": 3})

    with mock.patch.object(
        experiment_module, "build_run_specs",
        side_effect=AssertionError(
          "an incompatible sweep value reached spec translation")) \
        as translate:
      with self.assertRaisesRegex(
          ValueError, r"model\.trf\.n_heads.*divide"):
        experiment.build_specs(train_args=swept_train_args)
    translate.assert_not_called()

  def test_single_axis_token_count_is_checked_against_the_layout(self):
    """Window tokenization must leave a usable padded width of at least two."""
    accepted = _train_args(
      "restrf", head_values={"n_tokens": 3, "n_heads": 2})
    validate_active_model_values(
      train_args=accepted,
      model_cls=ResTRF,
      geom=types.SimpleNamespace(bin_sizes=[5]),
    )

    scalar_width = _train_args(
      "restrf", head_values={"n_tokens": 5, "n_heads": 1})
    with self.assertRaisesRegex(ValueError, "at least 2"):
      validate_active_model_values(
        train_args=scalar_width,
        model_cls=ResTRF,
        geom=types.SimpleNamespace(bin_sizes=[5]),
      )

    physical_bins = _train_args(
      "restrf", head_values={"n_tokens": 2, "n_heads": 1})
    with self.assertRaisesRegex(ValueError, "physical bins"):
      validate_active_model_values(
        train_args=physical_bins,
        model_cls=ResTRF,
        geom=types.SimpleNamespace(bin_sizes=[3, 3]),
      )

  def test_factored_cnn_groups_follow_the_template_count(self):
    """A factored NLA head accepts one group per template, not plain groups."""
    accepted = _train_args("rescnn", head_values={"groups": 3})
    validate_active_model_values(
      train_args=accepted,
      model_cls=TemplateResCNN,
      ia="nla",
    )
    refused = _train_args("rescnn", head_values={"groups": 2})
    with self.assertRaisesRegex(ValueError, "groups"):
      validate_active_model_values(
        train_args=refused,
        model_cls=TemplateResCNN,
        ia="nla",
      )


class ActiveModelConstructionTests(unittest.TestCase):
  """Check defensive constructors and the actual requested head modules."""

  def test_requested_cnn_and_transformer_blocks_are_present(self):
    """Positive head depths create that many correction blocks, not a trunk."""
    head_activation = make_activation("H", n_gates=3)
    cnn = ResCNN(
      input_dim=3,
      output_dim=4,
      int_dim_res=5,
      geom=_cnn_geometry(),
      kernel_size=3,
      n_blocks=1,
      n_blocks_cnn=2,
      gate_init=0.1,
      head_act=head_activation,
    )
    self.assertEqual(len(cnn.convs), 2)
    self.assertEqual(len(cnn.acts), 2)
    self.assertTrue(all(isinstance(block, nn.Conv1d)
                        for block in cnn.convs))

    transformer = ResTRF(
      input_dim=3,
      output_dim=8,
      int_dim_res=5,
      geom=_trf_geometry(),
      n_heads=2,
      n_blocks=1,
      n_blocks_trf=2,
      n_mlp_blocks=1,
      gate_init=0.1,
      head_act=head_activation,
    )
    self.assertEqual(len(transformer.trf), 2)
    self.assertTrue(all(block.n_heads == 2 for block in transformer.trf))

  def test_known_bad_constructor_values_refuse_before_any_linear_layer(self):
    """Defensive constructor checks run before learned layers are allocated."""
    cases = (
      ("even CNN kernel", ValueError, "kernel_size", lambda: ResCNN(
        input_dim=3, output_dim=4, int_dim_res=5, geom=_cnn_geometry(),
        kernel_size=2)),
      ("unknown CNN grouping", ValueError, "groups", lambda: ResCNN(
        input_dim=3, output_dim=4, int_dim_res=5, geom=_cnn_geometry(),
        kernel_size=3, groups=3)),
      ("zero CNN depth", ValueError, "n_blocks_cnn", lambda: ResCNN(
        input_dim=3, output_dim=4, int_dim_res=5, geom=_cnn_geometry(),
        kernel_size=3, n_blocks_cnn=0)),
      ("quoted CNN Boolean", TypeError, "separable", lambda: ResCNN(
        input_dim=3, output_dim=4, int_dim_res=5, geom=_cnn_geometry(),
        kernel_size=3, separable="false")),
      ("incompatible attention heads", ValueError, "n_heads", lambda: ResTRF(
        input_dim=3, output_dim=8, int_dim_res=5, geom=_trf_geometry(),
        n_heads=3)),
      ("zero transformer depth", ValueError, "n_blocks_trf", lambda: ResTRF(
        input_dim=3, output_dim=8, int_dim_res=5, geom=_trf_geometry(),
        n_blocks_trf=0)),
    )
    for label, error_type, message, construct in cases:
      with self.subTest(case=label):
        with mock.patch.object(
            nn, "Linear",
            side_effect=AssertionError(
              "an invalid setting reached a learned linear layer")):
          with self.assertRaisesRegex(error_type, message):
            construct()

  def test_direct_constructor_refuses_relu_for_the_head_only(self):
    """A ReLU trunk remains possible when the correction uses a live family."""
    with self.assertRaisesRegex(ValueError, "zero-initialized"):
      ResCNN(
        input_dim=3,
        output_dim=4,
        int_dim_res=5,
        geom=_cnn_geometry(),
        kernel_size=3,
        block_opts={"act": nn.ReLU},
      )

    model = ResCNN(
      input_dim=3,
      output_dim=4,
      int_dim_res=5,
      geom=_cnn_geometry(),
      kernel_size=3,
      block_opts={"act": nn.ReLU},
      head_act=make_activation("H", n_gates=3),
    )
    self.assertIsInstance(model.mlp[1].acts[0], nn.ReLU)

  def test_direct_constructors_refuse_power_families_in_zeroed_heads(self):
    """Known zero-gradient power factories cannot bypass YAML validation."""
    cases = (
      ("CNN power", "power", lambda: ResCNN(
        input_dim=3,
        output_dim=4,
        int_dim_res=5,
        geom=_cnn_geometry(),
        kernel_size=3,
        head_act=PowerGatedActivation,
      )),
      ("transformer gated power", "gated_power", lambda: ResTRF(
        input_dim=3,
        output_dim=8,
        int_dim_res=5,
        geom=_trf_geometry(),
        n_heads=2,
        head_act=GatedPowerActivation,
      )),
    )
    for label, family, construct in cases:
      with self.subTest(case=label), mock.patch.object(
          nn, "Linear",
          side_effect=AssertionError(
            "an unsafe head activation reached a learned layer")):
        with self.assertRaisesRegex(ValueError, family):
          construct()

  def test_invalid_bin_sizes_refuse_before_any_linear_layer(self):
    """Empty, zero-width, and fractional output bins have no model layout."""
    cases = (
      ("plain CNN empty", ValueError, lambda: ResCNN(
        input_dim=3,
        output_dim=4,
        int_dim_res=5,
        geom=types.SimpleNamespace(bin_sizes=[]),
        kernel_size=3,
      )),
      ("plain transformer zero", ValueError, lambda: ResTRF(
        input_dim=3,
        output_dim=4,
        int_dim_res=5,
        geom=types.SimpleNamespace(bin_sizes=[0]),
        n_heads=1,
      )),
      ("factored CNN fraction", TypeError, lambda: TemplateResCNN(
        input_dim=3,
        output_dim=4,
        n_amps=1,
        n_templates=3,
        int_dim_res=5,
        geom=types.SimpleNamespace(bin_sizes=[1.5]),
        kernel_size=3,
      )),
    )
    for label, error_type, construct in cases:
      with self.subTest(case=label), mock.patch.object(
          nn, "Linear",
          side_effect=AssertionError(
            "an invalid output layout reached a learned layer")):
        with self.assertRaisesRegex(error_type, r"bin_sizes"):
          construct()

  def test_one_h_activated_head_step_moves_cnn_and_transformer_weights(self):
    """A frozen trunk still gives both correction designs a first update."""
    torch.manual_seed(824)
    cnn = ResCNN(
      input_dim=3,
      output_dim=4,
      int_dim_res=5,
      geom=_cnn_geometry(),
      kernel_size=3,
      n_blocks=1,
      n_blocks_cnn=1,
      gate_init=0.1,
      head_act=make_activation("H", n_gates=3),
    )
    cnn_inputs = torch.tensor((
      (0.2, -0.4, 0.8),
      (-0.7, 0.5, 0.1),
      (1.0, -0.2, -0.6),
    ), dtype=torch.float32)
    _assert_head_step_moves(
      self,
      cnn,
      lambda: (cnn.convs[-1].weight, cnn.convs[-1].bias),
      cnn_inputs,
    )

    torch.manual_seed(825)
    transformer = ResTRF(
      input_dim=3,
      output_dim=8,
      int_dim_res=5,
      geom=_trf_geometry(),
      n_heads=2,
      n_blocks=1,
      n_blocks_trf=1,
      n_mlp_blocks=1,
      gate_init=0.1,
      head_act=make_activation("H", n_gates=3),
    )
    trf_inputs = torch.tensor((
      (0.4, -0.1, 0.9),
      (-0.2, 0.8, -0.5),
      (0.7, 0.3, -0.6),
    ), dtype=torch.float32)
    _assert_head_step_moves(
      self,
      transformer,
      lambda: tuple(
        transformer.trf[-1].mlp_lins[-1].parameters()),
      trf_inputs,
    )

  def test_optimized_python_keeps_kernel_group_and_head_refusals(self):
    """The public constructor checks survive ``python -O`` unchanged."""
    script = textwrap.dedent(
      """
      import types
      from emulator.designs.plain import ResCNN, ResTRF

      cnn_geom = types.SimpleNamespace(
          bin_sizes=[2, 2], pm_kept=[0, 0, 1, 1])
      trf_geom = types.SimpleNamespace(bin_sizes=[4, 4])
      cases = (
          ("kernel", lambda: ResCNN(
              input_dim=3, output_dim=4, int_dim_res=5,
              geom=cnn_geom, kernel_size=2)),
          ("groups", lambda: ResCNN(
              input_dim=3, output_dim=4, int_dim_res=5,
              geom=cnn_geom, kernel_size=3, groups=3)),
          ("n_heads", lambda: ResTRF(
              input_dim=3, output_dim=8, int_dim_res=5,
              geom=trf_geom, n_heads=3)),
      )
      for label, build in cases:
          try:
              build()
          except (TypeError, ValueError):
              continue
          raise SystemExit(label + " was accepted under optimized Python")
      control = ResTRF(
          input_dim=3, output_dim=8, int_dim_res=5,
          geom=trf_geom, n_heads=2, n_blocks=0,
          n_blocks_trf=1, n_mlp_blocks=1)
      if len(control.trf) != 1 or control.trf[0].n_heads != 2:
          raise SystemExit("valid optimized control lost its requested head")
      print("optimized-refusals: PASS")
      """
    )
    repo_root = str(Path(__file__).resolve().parents[2])
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
      repo_root if not old_pythonpath
      else repo_root + os.pathsep + old_pythonpath)
    environment.update({
      "KMP_AFFINITY": "disabled",
      "KMP_USE_SHM": "0",
      "MKL_NUM_THREADS": "1",
      "OMP_NUM_THREADS": "1",
    })

    # posix_spawn starts the optimized interpreter without running libomp's
    # unsafe fork hook after this parent has imported Torch. A normal nested
    # subprocess can abort with OpenMP error 179 on macOS before the witness
    # reaches the constructors, which is an environment failure rather than
    # model evidence.
    read_fd, write_fd = os.pipe()
    try:
      file_actions = (
        (os.POSIX_SPAWN_DUP2, write_fd, 1),
        (os.POSIX_SPAWN_DUP2, write_fd, 2),
        (os.POSIX_SPAWN_CLOSE, read_fd),
        (os.POSIX_SPAWN_CLOSE, write_fd),
      )
      pid = os.posix_spawn(
        sys.executable,
        (sys.executable, "-O", "-c", script),
        environment,
        file_actions=file_actions,
      )
    finally:
      os.close(write_fd)
    output_parts = []
    while True:
      part = os.read(read_fd, 65536)
      if not part:
        break
      output_parts.append(part)
    os.close(read_fd)
    _, wait_status = os.waitpid(pid, 0)
    returncode = os.waitstatus_to_exitcode(wait_status)
    output = b"".join(output_parts).decode("utf-8", errors="replace")
    self.assertEqual(
      returncode,
      0,
      msg="optimized subprocess failed:\n" + output,
    )
    self.assertIn("optimized-refusals: PASS", output)


if __name__ == "__main__":
  unittest.main()
