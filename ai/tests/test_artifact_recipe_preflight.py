"""CPU checks that saved-artifact metadata refuses before executable work.

A saved emulator names Python classes and also carries tensor bytes.  Damaged
metadata must be rejected while it is still plain YAML and NumPy data.  These
tests place sentinels at model-state access, dynamic import, and checkpoint
loading so a failure in the required ordering is visible immediately.
"""

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch
import yaml

from ai.gates.checks.artifact_fixtures import one_pass_training_recipe
from ai.gates.checks.compile_recipe import model_recipe, save_fixture
from emulator import fixed_facts, results
from emulator.activations import GatedPowerActivation, make_activation
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.output import DataVectorGeometry
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry


class _StateDictMustNotRun:
  """Model sentinel proving a save refusal happened before weight access."""

  def state_dict(self):
    raise AssertionError("model.state_dict must not run during preflight")


def _geometries():
  """Return registered one-input and one-output geometries for save checks."""
  device = torch.device("cpu")
  return (
    ParamGeometry(
      device=device, names=["p0"], center=[0.0], evecs=[[1.0]],
      sqrt_ev=[1.0]),
    ScalarGeometry(
      device=device, names=["derived"], center=[0.0], scale=[1.0]),
  )


def _scalar_recipe():
  """Return a complete one-input ResMLP recipe."""
  recipe = model_recipe(None)
  recipe["input_dim"] = 1
  recipe["output_dim"] = 1
  return recipe


def _histories():
  """Return one epoch of finite values at one validation threshold."""
  return {
    "train_losses": [0.1],
    "val_medians": [0.1],
    "val_means": [0.1],
    "val_fracs": [torch.tensor([0.5])],
    "thresholds": torch.tensor([1.0]),
  }


def _tiny_model(activation="H", output_dim=1):
  """Build the one-input network described by ``_scalar_recipe``."""
  return ResMLP(
    input_dim=1, output_dim=output_dim, int_dim_res=4, n_blocks=1,
    block_opts={
      "act": make_activation(activation, n_gates=3),
      "norm": make_norm("affine"),
    }).to(torch.device("cpu"))


def _data_vector_geometry(width=4):
  """Return a small unmasked data-vector geometry of a requested width."""
  return DataVectorGeometry(
    device=torch.device("cpu"), total_size=width,
    dest_idx=np.arange(width, dtype=np.int64),
    evecs=np.eye(width, dtype=np.float32),
    sqrt_ev=np.ones(width, dtype=np.float32),
    Cinv=np.eye(width, dtype=np.float32),
    center=np.zeros(width, dtype=np.float32))


def _save_attempt(
    root, *, recipe=None, histories=None, transfer_base=None, model=None,
    param_geometry=None, geometry=None, resolved_train=None):
  """Call the production writer with a weight-access sentinel."""
  if param_geometry is None or geometry is None:
    default_param_geometry, default_geometry = _geometries()
    if param_geometry is None:
      param_geometry = default_param_geometry
    if geometry is None:
      geometry = default_geometry
  if recipe is None:
    recipe = _scalar_recipe()
  if histories is None:
    histories = _histories()
  if model is None:
    model = _StateDictMustNotRun()
  if resolved_train is None:
    resolved_train = one_pass_training_recipe(thresholds=(1.0,))
  transfer = transfer_base is not None
  return results.save_emulator(
    path_root=str(root),
    model=model,
    param_geometry=param_geometry,
    geometry=geometry,
    config={"data": {}, "train_args": {"nepochs": 1}},
    histories=histories,
    resolved_train=resolved_train,
    resolved_model=recipe,
    transfer_base=transfer_base,
    facts_yaml=fixed_facts.synthetic_sidecar(
      names=["p0"], label="artifact-preflight", family="scalar",
      support=None),
    composition_mode="transfer" if transfer else "plain",
    transfer_refined=False,
    resolved_pce=None,
    resolved_transfer=(
      {"form": "gain", "space": "physical", "refine": None}
      if transfer else None),
    attrs={"rescale": "none"},
    resolved_rescale="none",
  )


def _rewrite_yaml(group, name, value):
  """Replace one scalar YAML dataset while preserving its HDF5 location."""
  del group[name]
  group.create_dataset(
    name, data=yaml.safe_dump(value, sort_keys=False),
    dtype=h5py.string_dtype(encoding="utf-8"))


def _read_yaml(group, name):
  """Read one fixture YAML dataset as a plain mapping."""
  raw = group[name][()]
  if isinstance(raw, bytes):
    raw = raw.decode("utf-8")
  return yaml.safe_load(raw)


class ArtifactRecipePreflightTests(unittest.TestCase):
  """Require all recipe and training refusals before executable surfaces."""

  def test_save_refuses_incomplete_main_recipe_before_weights_or_staging(self):
    recipe = _scalar_recipe()
    del recipe["kwargs"]["n_blocks"]
    with tempfile.TemporaryDirectory(prefix="recipe-save-main-") as temp:
      root = Path(temp) / "artifact"
      with mock.patch.object(
          results.torch, "save",
          side_effect=AssertionError("staging must not begin")) as staging:
        with self.assertRaisesRegex(ValueError, "missing.*n_blocks"):
          _save_attempt(root, recipe=recipe)
      staging.assert_not_called()

  def test_save_refuses_incomplete_transfer_recipe_before_weights(self):
    param_geometry, geometry = _geometries()
    base_model = _tiny_model()
    base_recipe = _scalar_recipe()
    del base_recipe["kwargs"]["block_opts"]
    transfer_base = {
      "recipe": base_recipe,
      "model": base_model,
      "state": base_model.state_dict(),
      "param_geometry": param_geometry,
      "dv_geometry": geometry,
      "form": "gain",
      "space": "physical",
    }
    with tempfile.TemporaryDirectory(prefix="recipe-save-transfer-") as temp:
      with self.assertRaisesRegex(ValueError, "missing.*block_opts"):
        _save_attempt(Path(temp) / "artifact", transfer_base=transfer_base)

  def test_save_refuses_incompatible_history_curves_before_weights(self):
    histories = _histories()
    histories["val_means"] = []
    with tempfile.TemporaryDirectory(prefix="recipe-save-history-") as temp:
      with self.assertRaisesRegex(ValueError, "common nonzero length"):
        _save_attempt(Path(temp) / "artifact", histories=histories)

  def test_save_refuses_nonfinite_history_before_weights(self):
    histories = _histories()
    histories["val_fracs"] = [torch.tensor([float("nan")])]
    with tempfile.TemporaryDirectory(prefix="recipe-save-history-finite-") as temp:
      with self.assertRaisesRegex(ValueError, "val_fracs.*finite numbers"):
        _save_attempt(Path(temp) / "artifact", histories=histories)

  def test_save_refuses_tanh_model_described_as_relu_before_staging(self):
    recipe = _scalar_recipe()
    recipe["kwargs"]["block_opts"]["act"]["type"] = "relu"
    with tempfile.TemporaryDirectory(prefix="recipe-live-main-") as temp:
      root = Path(temp) / "artifact"
      with mock.patch.object(
          results.torch, "save",
          side_effect=AssertionError("staging must not begin")) as staging:
        with self.assertRaisesRegex(
            ValueError, "constructor recipe does not exactly match"):
          _save_attempt(root, recipe=recipe, model=_tiny_model("tanh"))
      staging.assert_not_called()

  def test_save_refuses_direct_relu_class_as_registered_relu(self):
    """Direct nn.ReLU consumes the block width as its inplace flag."""
    model = ResMLP(
      input_dim=1, output_dim=1, int_dim_res=4, n_blocks=1,
      block_opts={"act": torch.nn.ReLU, "norm": make_norm("affine")})
    recipe = _scalar_recipe()
    recipe["kwargs"]["block_opts"]["act"] = {
      "type": "relu", "n_gates": 3}
    with tempfile.TemporaryDirectory(prefix="recipe-direct-relu-") as temp:
      root = Path(temp) / "artifact"
      with mock.patch.object(
          results.torch, "save",
          side_effect=AssertionError("staging must not begin")) as staging:
        with self.assertRaisesRegex(
            ValueError, "constructor recipe does not exactly match"):
          _save_attempt(root, recipe=recipe, model=model)
      staging.assert_not_called()

  def test_direct_gated_power_roundtrip_preserves_one_gate_shape(self):
    """A direct gated-power class records the default it really executes."""
    model = ResMLP(
      input_dim=1, output_dim=1, int_dim_res=4, n_blocks=1,
      block_opts={
        "act": GatedPowerActivation, "norm": make_norm("affine")})
    recipe = _scalar_recipe()
    recipe["kwargs"]["block_opts"]["act"] = {
      "type": "gated_power", "n_gates": 1}
    param_geometry, geometry = _geometries()
    with tempfile.TemporaryDirectory(prefix="recipe-direct-gated-power-") as temp:
      root = Path(temp) / "artifact"
      results.save_emulator(
        path_root=str(root), model=model,
        param_geometry=param_geometry, geometry=geometry,
        config={"data": {}, "train_args": {"nepochs": 1}},
        histories=_histories(),
        resolved_train=one_pass_training_recipe(thresholds=(1.0,)),
        resolved_model=recipe,
        facts_yaml=fixed_facts.synthetic_sidecar(
          names=["p0"], label="artifact-direct-gated-power",
          family="scalar", support=None),
        composition_mode="plain", transfer_refined=False,
        resolved_pce=None, resolved_transfer=None,
        attrs={"rescale": "none"}, resolved_rescale="none")
      rebuilt, _, _, _ = results.rebuild_emulator(
        str(root), device=torch.device("cpu"), compile_model=False)
      direct = [
        module for module in rebuilt.modules()
        if isinstance(module, GatedPowerActivation)]
      self.assertTrue(direct)
      self.assertTrue(all(tuple(module.w.shape)[0] == 1 for module in direct))
      sample = torch.tensor([[0.25]], dtype=torch.float32)
      torch.testing.assert_close(rebuilt(sample), model(sample), rtol=0, atol=0)

  def test_save_refuses_transfer_recipe_that_misnames_live_activation(self):
    param_geometry, geometry = _geometries()
    base_model = _tiny_model("tanh")
    base_recipe = _scalar_recipe()
    base_recipe["kwargs"]["block_opts"]["act"]["type"] = "relu"
    transfer_base = {
      "recipe": base_recipe,
      "model": base_model,
      "state": base_model.state_dict(),
      "param_geometry": param_geometry,
      "dv_geometry": geometry,
      "form": "gain",
      "space": "physical",
    }
    with tempfile.TemporaryDirectory(prefix="recipe-live-transfer-") as temp:
      with self.assertRaisesRegex(
          ValueError, "live transfer base.*constructor recipe"):
        _save_attempt(
          Path(temp) / "artifact", model=_tiny_model(),
          transfer_base=transfer_base)

  def test_save_refuses_recipe_input_width_disagreeing_with_geometry(self):
    param_geometry = ParamGeometry(
      device=torch.device("cpu"), names=["p0", "p1"],
      center=[0.0, 0.0], evecs=[[1.0, 0.0], [0.0, 1.0]],
      sqrt_ev=[1.0, 1.0])
    with tempfile.TemporaryDirectory(prefix="recipe-input-width-") as temp:
      with self.assertRaisesRegex(ValueError, "encoded width 2"):
        _save_attempt(
          Path(temp) / "artifact", model=_tiny_model(),
          param_geometry=param_geometry)

  def test_save_refuses_output_width_that_would_broadcast_on_decode(self):
    geometry = ScalarGeometry(
      device=torch.device("cpu"), names=["a", "b"],
      center=[0.0, 0.0], scale=[1.0, 1.0])
    with tempfile.TemporaryDirectory(prefix="recipe-output-width-") as temp:
      with self.assertRaisesRegex(ValueError, "destination width 2"):
        _save_attempt(
          Path(temp) / "artifact", model=_tiny_model(), geometry=geometry)

  def _saved_fixture(self, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    root = Path(directory) / "artifact"
    save_fixture(
      path_root=root, compile_mode="default", case_label="preflight")
    return root

  def _saved_transfer_fixture(self, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    root = Path(directory) / "transfer-artifact"
    param_geometry, geometry = _geometries()
    base_model = _tiny_model()
    results.save_emulator(
      path_root=str(root), model=_tiny_model(),
      param_geometry=param_geometry, geometry=geometry,
      config={"data": {}, "train_args": {"nepochs": 1}},
      histories=_histories(),
      resolved_train=one_pass_training_recipe(thresholds=(1.0,)),
      resolved_model=_scalar_recipe(),
      transfer_base={
        "recipe": _scalar_recipe(),
        "model": base_model,
        "state": base_model.state_dict(),
        "param_geometry": param_geometry,
        "dv_geometry": geometry,
        "form": "gain",
        "space": "physical",
      },
      facts_yaml=fixed_facts.synthetic_sidecar(
        names=["p0"], label="artifact-transfer-preflight",
        family="scalar", support=None),
      composition_mode="transfer", transfer_refined=False,
      resolved_pce=None,
      attrs={"rescale": "none"},
      resolved_transfer={
        "form": "gain", "space": "physical", "refine": None,
      },
    )
    return root

  def _saved_data_vector_fixture(self, directory, width=4):
    """Save a plain data-vector artifact for inert width mutations."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    root = Path(directory) / "data-vector-artifact"
    param_geometry, _ = _geometries()
    geometry = _data_vector_geometry(width)
    recipe = _scalar_recipe()
    recipe["output_dim"] = width
    model = _tiny_model(output_dim=width)
    results.save_emulator(
      path_root=str(root), model=model,
      param_geometry=param_geometry, geometry=geometry,
      config={"data": {}, "train_args": {"nepochs": 1}},
      histories=_histories(),
      resolved_train=one_pass_training_recipe(thresholds=(1.0,)),
      resolved_model=recipe,
      facts_yaml=fixed_facts.synthetic_sidecar(
        names=["p0"], label="artifact-data-vector-preflight",
        family="cosmic_shear", support=None),
      composition_mode="plain", transfer_refined=False,
      resolved_pce=None, resolved_transfer=None,
      attrs={"rescale": "none"}, resolved_rescale="none")
    return root

  def _assert_rebuild_refuses_without_execution(self, root, message):
    with mock.patch.object(
        importlib, "import_module",
        side_effect=AssertionError("dynamic import must not run")) as imports, \
        mock.patch.object(
          results, "_load_tensor_state_dict",
          side_effect=AssertionError("checkpoint load must not run")) as load:
      with self.assertRaisesRegex((TypeError, ValueError, KeyError), message):
        results.rebuild_emulator(
          path_root=str(root), device=torch.device("cpu"),
          compile_model=False)
    imports.assert_not_called()
    load.assert_not_called()

  def test_current_recipe_only_artifacts_rebuild_without_manifests(self):
    """Plain and transfer saves need no duplicate compatibility record."""
    with tempfile.TemporaryDirectory(prefix="recipe-only-artifacts-") as temp:
      roots = (
        self._saved_fixture(Path(temp) / "plain"),
        self._saved_transfer_fixture(Path(temp) / "transfer"),
      )
      for root in roots:
        with self.subTest(root=root.name):
          with h5py.File(str(root) + ".h5", "r") as artifact:
            self.assertNotIn("compatibility_manifest", artifact)
            if "transfer_base" in artifact:
              self.assertNotIn(
                "compatibility_manifest", artifact["transfer_base"])
          model, _, _, _ = results.rebuild_emulator(
            path_root=str(root), device=torch.device("cpu"),
            compile_model=False)
          self.assertIsInstance(model, torch.nn.Module)

  def test_rebuild_refuses_recipe_before_dynamic_import_or_tensor_load(self):
    with tempfile.TemporaryDirectory(prefix="recipe-read-main-") as temp:
      root = self._saved_fixture(temp)
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        recipe = _read_yaml(artifact, "model_recipe")
        del recipe["kwargs"]["n_blocks"]
        _rewrite_yaml(artifact, "model_recipe", recipe)
      self._assert_rebuild_refuses_without_execution(
        root, "model_recipe.*missing.*n_blocks")

  def test_rebuild_refuses_input_width_before_import_or_tensor_load(self):
    with tempfile.TemporaryDirectory(prefix="recipe-read-input-width-") as temp:
      root = self._saved_fixture(temp)
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        recipe = _read_yaml(artifact, "model_recipe")
        recipe["input_dim"] = 3
        _rewrite_yaml(artifact, "model_recipe", recipe)
      self._assert_rebuild_refuses_without_execution(
        root, "parameter geometry 'names'.*shape \\(3,\\).*got \\(2,\\)")

  def test_rebuild_refuses_parameter_names_width_before_execution(self):
    """Input width follows the saved name order as well as its tensors."""
    with tempfile.TemporaryDirectory(prefix="recipe-read-param-names-") as temp:
      root = self._saved_fixture(temp)
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        geometry = artifact["param_geometry"]
        del geometry["names"]
        geometry.create_dataset(
          "names", data=np.asarray(["p0"], dtype=object),
          dtype=h5py.string_dtype(encoding="utf-8"))
      self._assert_rebuild_refuses_without_execution(
        root, "parameter geometry 'names'.*shape \\(2,\\).*got \\(1,\\)")

  def test_inert_width_check_binds_factored_parameter_geometry(self):
    """Factored input width includes the nested basis and amplitude columns."""
    with tempfile.TemporaryDirectory(prefix="recipe-factored-input-") as temp:
      path = Path(temp) / "factored.h5"
      with h5py.File(path, "w") as artifact:
        parameter = artifact.create_group("param_geometry")
        parameter.attrs["cls"] = (
          "emulator.geometries.parameter.AmplitudeFactorGeometry")
        parameter.attrs["n_param"] = 4
        parameter.create_dataset(
          "names", data=np.asarray(["p0", "a1", "p2", "p3"], dtype=object),
          dtype=h5py.string_dtype(encoding="utf-8"))
        parameter.create_dataset("amp_idx", data=np.asarray([1], dtype=np.int64))
        kept = parameter.create_group("pg_keep")
        kept.create_dataset(
          "names", data=np.asarray(["p0", "p2", "p3"], dtype=object),
          dtype=h5py.string_dtype(encoding="utf-8"))
        kept.create_dataset("center", data=np.zeros(3))
        kept.create_dataset("sqrt_ev", data=np.ones(3))
        kept.create_dataset("evecs", data=np.eye(3))
        output = artifact.create_group("dv_geometry")
        output.attrs["cls"] = "emulator.geometries.scalar.ScalarGeometry"
        output.create_dataset(
          "names", data=np.asarray(["derived"], dtype=object),
          dtype=h5py.string_dtype(encoding="utf-8"))
        output.create_dataset("center", data=np.zeros(1))
        output.create_dataset("scale", data=np.ones(1))
        recipe = {
          "cls": "emulator.designs.ia.TemplateMLP",
          "name": "resmlp", "ia": "nla",
          "input_dim": 4, "output_dim": 1,
          "compile_mode": None, "needs_geom": False,
          "kwargs": {
            "n_amps": 1, "n_templates": 3,
            "int_dim_res": 4, "n_blocks": 1,
            "block_opts": {
              "n_layers": 2,
              "act": {"type": "H", "n_gates": 3},
              "norm": "affine"},
          },
        }
        results._validate_saved_recipe_geometry_widths(
          recipe, parameter, output, "factored fixture")
        del kept["center"]
        kept.create_dataset("center", data=np.zeros(2))
        with self.assertRaisesRegex(
            ValueError, "pg_keep 'center'.*shape \\(3,\\).*got \\(2,\\)"):
          results._validate_saved_recipe_geometry_widths(
            recipe, parameter, output, "factored fixture")

  def test_inert_width_check_binds_log_parameter_mask(self):
    """The saved log mask has one entry for every encoded input column."""
    with tempfile.TemporaryDirectory(prefix="recipe-log-input-") as temp:
      path = Path(temp) / "log.h5"
      with h5py.File(path, "w") as artifact:
        parameter = artifact.create_group("param_geometry")
        parameter.attrs["cls"] = (
          "emulator.geometries.parameter.LogParamGeometry")
        parameter.create_dataset(
          "names", data=np.asarray(["p0", "p1"], dtype=object),
          dtype=h5py.string_dtype(encoding="utf-8"))
        parameter.create_dataset("center", data=np.zeros(2))
        parameter.create_dataset("sqrt_ev", data=np.ones(2))
        parameter.create_dataset("evecs", data=np.eye(2))
        parameter.create_dataset("log_mask", data=np.ones(1, dtype=np.uint8))
        output = artifact.create_group("dv_geometry")
        output.attrs["cls"] = "emulator.geometries.scalar.ScalarGeometry"
        output.create_dataset(
          "names", data=np.asarray(["derived"], dtype=object),
          dtype=h5py.string_dtype(encoding="utf-8"))
        output.create_dataset("center", data=np.zeros(1))
        output.create_dataset("scale", data=np.ones(1))
        recipe = _scalar_recipe()
        recipe["input_dim"] = 2
        with self.assertRaisesRegex(
            ValueError, "log_mask.*shape \\(2,\\).*got \\(1,\\)"):
          results._validate_saved_recipe_geometry_widths(
            recipe, parameter, output, "log fixture")

  def test_rebuild_refuses_output_width_before_import_or_tensor_load(self):
    with tempfile.TemporaryDirectory(prefix="recipe-read-output-width-") as temp:
      root = self._saved_fixture(temp)
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        recipe = _read_yaml(artifact, "model_recipe")
        recipe["output_dim"] = 2
        _rewrite_yaml(artifact, "model_recipe", recipe)
      self._assert_rebuild_refuses_without_execution(
        root, "output geometry 'names'.*shape \\(2,\\).*got \\(1,\\)")

  def test_rebuild_refuses_scalar_names_width_before_execution(self):
    """Scalar width comes from names, not merely center or scale."""
    with tempfile.TemporaryDirectory(prefix="recipe-read-scalar-names-") as temp:
      root = self._saved_fixture(temp)
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        geometry = artifact["dv_geometry"]
        del geometry["names"]
        geometry.create_dataset(
          "names", data=np.asarray(["derived", "extra"], dtype=object),
          dtype=h5py.string_dtype(encoding="utf-8"))
      self._assert_rebuild_refuses_without_execution(
        root, "output geometry 'names'.*shape \\(1,\\).*got \\(2,\\)")

  def test_rebuild_refuses_data_vector_dest_idx_width_before_execution(self):
    """Masked data-vector width comes from kept positions, not center alone."""
    with tempfile.TemporaryDirectory(prefix="recipe-read-dest-idx-") as temp:
      root = self._saved_data_vector_fixture(temp, width=4)
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        geometry = artifact["dv_geometry"]
        del geometry["dest_idx"]
        geometry.create_dataset(
          "dest_idx", data=np.asarray([0, 1, 2], dtype=np.int64))
      self._assert_rebuild_refuses_without_execution(
        root, "output geometry 'dest_idx'.*shape \\(4,\\).*got \\(3,\\)")

  def test_inert_width_check_covers_cmb_and_grid_arrays(self):
    """Every derived-width family binds all arrays used by its decoder."""
    cases = (
      (
        "cmb", "emulator.geometries.cmb.CmbDiagonalGeometry", 3,
        {"ell": (3,), "center": (3,), "sigma": (3,),
         "fiducial_cl": (2,)},
        "fiducial_cl.*shape \\(3,\\).*got \\(2,\\)",
      ),
      (
        "grid", "emulator.geometries.grid.GridGeometry", 3,
        {"z": (3,), "center": (3,), "scale": (2,)},
        "scale.*shape \\(3,\\).*got \\(2,\\)",
      ),
      (
        "grid2d", "emulator.geometries.grid2d.Grid2DGeometry", 6,
        {"z": (2,), "k": (3,), "center": (6,), "scale": (6,),
         "const_mask": (5,)},
        "const_mask.*shape \\(6,\\).*got \\(5,\\)",
      ),
    )
    with tempfile.TemporaryDirectory(prefix="recipe-inert-width-families-") as temp:
      for label, class_path, width, datasets, message in cases:
        with self.subTest(label=label):
          path = Path(temp) / (label + ".h5")
          with h5py.File(path, "w") as artifact:
            parameter = artifact.create_group("param_geometry")
            parameter.attrs["cls"] = (
              "emulator.geometries.parameter.ParamGeometry")
            parameter.create_dataset(
              "names", data=np.asarray(["p0"], dtype=object),
              dtype=h5py.string_dtype(encoding="utf-8"))
            parameter.create_dataset("center", data=np.zeros(1))
            parameter.create_dataset("sqrt_ev", data=np.ones(1))
            parameter.create_dataset("evecs", data=np.eye(1))
            output = artifact.create_group("dv_geometry")
            output.attrs["cls"] = class_path
            for name, shape in datasets.items():
              output.create_dataset(name, data=np.zeros(shape))
            recipe = _scalar_recipe()
            recipe["output_dim"] = width
            with self.assertRaisesRegex(ValueError, message):
              results._validate_saved_recipe_geometry_widths(
                recipe, parameter, output, label + " fixture")

  def test_rebuild_refuses_transfer_recipe_before_execution(self):
    with tempfile.TemporaryDirectory(prefix="recipe-read-transfer-") as temp:
      root = self._saved_transfer_fixture(temp)
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        transfer = artifact["transfer_base"]
        recipe = _read_yaml(transfer, "model_recipe")
        del recipe["kwargs"]["n_blocks"]
        _rewrite_yaml(transfer, "model_recipe", recipe)
      self._assert_rebuild_refuses_without_execution(
        root, "transfer_base model_recipe.*missing.*n_blocks")

  def test_rebuild_does_not_read_training_history(self):
    """Deleting provenance curves cannot change model reconstruction."""
    with tempfile.TemporaryDirectory(prefix="recipe-read-no-history-") as temp:
      root = self._saved_fixture(temp)
      with h5py.File(str(root) + ".h5", "r+") as artifact:
        del artifact["history"]
      model, _, _, _ = results.rebuild_emulator(
        path_root=str(root), device=torch.device("cpu"), compile_model=False)
      self.assertIsInstance(model, torch.nn.Module)

  def test_rebuild_keeps_unknown_training_fields_as_provenance(self):
    """Future training descriptions do not become reconstruction grammar."""
    training_record = {"future_training_policy": {"label": "opaque"}}
    with tempfile.TemporaryDirectory(prefix="recipe-read-opaque-training-") as temp:
      root = Path(temp) / "artifact"
      _save_attempt(
        root, model=_tiny_model(), resolved_train=training_record)
      _, _, _, info = results.rebuild_emulator(
        path_root=str(root), device=torch.device("cpu"), compile_model=False)
      self.assertEqual(info["config_resolved"]["train_args"], training_record)


if __name__ == "__main__":
  unittest.main()
