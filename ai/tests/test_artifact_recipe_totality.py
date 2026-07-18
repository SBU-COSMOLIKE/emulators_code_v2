"""Focused checks for complete, import-free artifact model recipes."""

import ast
import copy
import importlib
import inspect
from pathlib import Path
import unittest

from emulator import artifact_recipe


def _recipe(class_path="emulator.designs.plain.ResCNN", ia=None):
  """Return one complete small recipe for any supported model class."""
  specs = {
    "emulator.designs.plain.ResMLP": {
      "name": "resmlp", "needs_geom": False,
      "kwargs": {"int_dim_res": 5, "n_blocks": 1}},
    "emulator.designs.plain.ResCNN": {
      "name": "rescnn", "needs_geom": True,
      "kwargs": {
        "int_dim_res": 5, "kernel_size": 3, "rescale_kernel": False,
        "groups": 1, "separable": False, "film": False, "n_blocks": 1,
        "n_blocks_cnn": 2, "gate_init": 0.25, "head_act": None}},
    "emulator.designs.plain.ResTRF": {
      "name": "restrf", "needs_geom": True,
      "kwargs": {
        "int_dim_res": 5, "n_heads": 1, "n_blocks": 1,
        "n_blocks_trf": 2, "n_mlp_blocks": 2, "n_tokens": None,
        "gate_init": 0.25, "shared_mlp": False, "film": False,
        "head_act": None}},
    "emulator.designs.ia.TemplateMLP": {
      "name": "resmlp", "needs_geom": False,
      "kwargs": {
        "n_amps": 1, "n_templates": 3, "int_dim_res": 5,
        "n_blocks": 1}},
    "emulator.designs.ia.TemplateResCNN": {
      "name": "rescnn", "needs_geom": True,
      "kwargs": {
        "n_amps": 1, "n_templates": 3, "int_dim_res": 5,
        "kernel_size": 3, "rescale_kernel": False, "groups": 1,
        "separable": False, "film": False, "n_blocks": 1,
        "n_blocks_cnn": 2, "gate_init": 0.25, "head_act": None}},
    "emulator.designs.ia.TemplateResTRF": {
      "name": "restrf", "needs_geom": True,
      "kwargs": {
        "n_amps": 1, "n_templates": 3, "int_dim_res": 5,
        "n_heads": 1, "n_blocks": 1, "n_blocks_trf": 2,
        "n_mlp_blocks": 2, "gate_init": 0.25, "shared_mlp": False,
        "film": False, "head_act": None}},
  }
  selected = copy.deepcopy(specs[class_path])
  selected["kwargs"]["block_opts"] = {
    "n_layers": 2,
    "act": {"type": "H", "n_gates": 3},
    "norm": "affine",
  }
  if class_path.startswith("emulator.designs.ia."):
    if ia is None:
      ia = "nla"
    if ia == "tatt":
      selected["kwargs"]["n_amps"] = 3
      selected["kwargs"]["n_templates"] = 10
  return {
    "cls": class_path,
    "name": selected["name"],
    "ia": ia,
    "input_dim": 4,
    "output_dim": 6,
    "compile_mode": None,
    "needs_geom": selected["needs_geom"],
    "kwargs": selected["kwargs"],
  }


def _live_model(recipe):
  """Construct one of the six registered models from a complete recipe."""
  import torch
  from emulator.activations import make_activation
  from emulator.designs.blocks import make_norm
  from emulator.geometries.output import DataVectorGeometry

  module_name, _, class_name = recipe["cls"].rpartition(".")
  model_class = getattr(importlib.import_module(module_name), class_name)
  kwargs = copy.deepcopy(recipe["kwargs"])
  block = kwargs["block_opts"]
  activation = block["act"]
  kwargs["block_opts"] = {
    "n_layers": block["n_layers"],
    "act": make_activation(
      activation["type"], n_gates=activation["n_gates"]),
    "norm": make_norm(block["norm"]),
  }
  if kwargs.get("head_act") is not None:
    head = kwargs["head_act"]
    kwargs["head_act"] = make_activation(
      head["type"], n_gates=head["n_gates"])
  if recipe["needs_geom"]:
    width = recipe["output_dim"]
    kwargs["geom"] = DataVectorGeometry(
      device=torch.device("cpu"), total_size=width,
      dest_idx=torch.arange(width), evecs=torch.eye(width),
      sqrt_ev=torch.ones(width), Cinv=torch.eye(width),
      center=torch.zeros(width), bin_sizes=[width],
      head_pad_idx=torch.arange(width),
      head_valid_mask=torch.ones((1, width), dtype=torch.bool))
  return model_class(
    input_dim=recipe["input_dim"], output_dim=recipe["output_dim"],
    **kwargs)


class ArtifactRecipeTotalityTests(unittest.TestCase):
  """Require every supported reconstruction choice before imports."""

  def test_all_six_complete_recipes_pass(self):
    for class_path in artifact_recipe.MODEL_RECIPE_CLASSES:
      with self.subTest(class_path=class_path):
        recipe = _recipe(class_path)
        self.assertIs(
          artifact_recipe.validate_model_recipe(recipe), recipe)

  def test_all_six_live_constructors_attach_the_exact_recipe(self):
    for class_path in artifact_recipe.MODEL_RECIPE_CLASSES:
      recipe = _recipe(class_path)
      recipe["kwargs"]["block_opts"]["n_layers"] = 3
      with self.subTest(class_path=class_path):
        model = _live_model(recipe)
        self.assertEqual(model.emul_runtime_recipe, recipe)
        blocks = [
          module for module in model.modules()
          if type(module).__name__ == "ResBlock"]
        self.assertTrue(blocks)
        self.assertTrue(all(len(block.layers) == 3 for block in blocks))

  def test_every_top_level_key_is_required_and_unknown_keys_refuse(self):
    original = _recipe()
    for key in artifact_recipe.MODEL_RECIPE_TOP_LEVEL_KEYS:
      changed = copy.deepcopy(original)
      del changed[key]
      with self.subTest(missing=key), self.assertRaisesRegex(
          ValueError, "missing.*" + key):
        artifact_recipe.validate_model_recipe(changed)
    changed = copy.deepcopy(original)
    changed["future_default"] = True
    with self.assertRaisesRegex(ValueError, "unknown.*future_default"):
      artifact_recipe.validate_model_recipe(changed)

  def test_every_constructor_kwarg_is_required_for_every_class(self):
    for class_path in artifact_recipe.MODEL_RECIPE_CLASSES:
      original = _recipe(class_path)
      for key in tuple(original["kwargs"]):
        changed = copy.deepcopy(original)
        del changed["kwargs"][key]
        with self.subTest(class_path=class_path, missing=key), \
            self.assertRaisesRegex(ValueError, "missing.*" + key):
          artifact_recipe.validate_model_recipe(changed)

  def test_explicit_null_head_activation_differs_from_absence(self):
    recipe = _recipe()
    self.assertIsNone(recipe["kwargs"]["head_act"])
    artifact_recipe.validate_model_recipe(recipe)
    del recipe["kwargs"]["head_act"]
    with self.assertRaisesRegex(ValueError, "missing.*head_act"):
      artifact_recipe.validate_model_recipe(recipe)

  def test_unknown_constructor_and_nested_factory_keys_refuse(self):
    mutations = []
    extra_kwarg = _recipe()
    extra_kwarg["kwargs"]["future"] = 1
    mutations.append((extra_kwarg, "future"))
    extra_block = _recipe()
    extra_block["kwargs"]["block_opts"]["future"] = 1
    mutations.append((extra_block, "future"))
    extra_activation = _recipe()
    extra_activation["kwargs"]["block_opts"]["act"]["future"] = 1
    mutations.append((extra_activation, "future"))
    for changed, key in mutations:
      with self.subTest(location=key), self.assertRaisesRegex(
          ValueError, "unknown.*" + key):
        artifact_recipe.validate_model_recipe(changed)

  def test_factory_specs_require_native_complete_values(self):
    mutations = []
    missing_gate = _recipe()
    del missing_gate["kwargs"]["block_opts"]["act"]["n_gates"]
    mutations.append((missing_gate, ValueError, "n_gates"))
    bool_gate = _recipe()
    bool_gate["kwargs"]["block_opts"]["act"]["n_gates"] = True
    mutations.append((bool_gate, TypeError, "native integer"))
    unknown_activation = _recipe()
    unknown_activation["kwargs"]["block_opts"]["act"]["type"] = "mystery"
    mutations.append((unknown_activation, ValueError, "one of"))
    unknown_norm = _recipe()
    unknown_norm["kwargs"]["block_opts"]["norm"] = "batch"
    mutations.append((unknown_norm, ValueError, "one of"))
    missing_depth = _recipe()
    del missing_depth["kwargs"]["block_opts"]["n_layers"]
    mutations.append((missing_depth, ValueError, "n_layers"))
    bool_depth = _recipe()
    bool_depth["kwargs"]["block_opts"]["n_layers"] = True
    mutations.append((bool_depth, TypeError, "native integer"))
    for changed, error_type, message in mutations:
      with self.subTest(message=message), self.assertRaisesRegex(
          error_type, message):
        artifact_recipe.validate_model_recipe(changed)

  def test_class_identity_fields_must_agree(self):
    wrong_name = _recipe()
    wrong_name["name"] = "resmlp"
    wrong_geometry = _recipe()
    wrong_geometry["needs_geom"] = False
    wrong_ia = _recipe()
    wrong_ia["ia"] = "nla"
    unknown_class = _recipe()
    unknown_class["cls"] = "other.Model"
    for changed, message in (
        (wrong_name, "name"),
        (wrong_geometry, "needs_geom"),
        (wrong_ia, "ia"),
        (unknown_class, "not a supported")):
      with self.subTest(message=message), self.assertRaisesRegex(
          ValueError, message):
        artifact_recipe.validate_model_recipe(changed)

  def test_factored_shapes_are_bound_to_the_named_design(self):
    recipe = _recipe("emulator.designs.ia.TemplateMLP", ia="nla")
    recipe["kwargs"]["n_templates"] = 10
    with self.assertRaisesRegex(ValueError, "factored shape"):
      artifact_recipe.validate_model_recipe(recipe)
    artifact_recipe.validate_model_recipe(
      _recipe("emulator.designs.ia.TemplateMLP", ia="tatt"))

  def test_recipe_validation_does_not_mutate_the_input(self):
    recipe = _recipe()
    before = copy.deepcopy(recipe)
    artifact_recipe.validate_model_recipe(recipe)
    self.assertEqual(recipe, before)

  def test_production_validator_imports_no_model_or_geometry_modules(self):
    source_path = Path(artifact_recipe.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
      elif isinstance(node, ast.ImportFrom):
        imports.append(node.module)
    self.assertEqual(imports, ["math"])

  def test_registry_census_matches_all_six_constructor_signatures(self):
    for class_path in artifact_recipe.MODEL_RECIPE_CLASSES:
      module_name, _, class_name = class_path.rpartition(".")
      model_class = getattr(importlib.import_module(module_name), class_name)
      observed = tuple(
        name for name in inspect.signature(model_class.__init__).parameters
        if name != "self")
      with self.subTest(class_path=class_path):
        self.assertEqual(
          observed,
          artifact_recipe.expected_constructor_parameters(class_path))

  def test_resblock_constructor_census_includes_persisted_depth(self):
    from emulator.designs.blocks import ResBlock
    observed = tuple(inspect.signature(ResBlock.__init__).parameters)
    self.assertEqual(observed, ("self", "size", "n_layers", "norm", "act"))

  def test_direct_activation_factory_uses_one_canonical_gate_spelling(self):
    from emulator.activations import (
      GatedActivation, GatedPowerActivation, activation_factory_recipe,
      activation_fcn, make_activation)
    self.assertEqual(
      activation_factory_recipe(activation_fcn),
      {"type": "H", "n_gates": 3})
    self.assertEqual(
      activation_factory_recipe(make_activation("H", n_gates=7)),
      {"type": "H", "n_gates": 7})
    self.assertEqual(
      activation_factory_recipe(GatedActivation),
      {"type": "multigate", "n_gates": 1})
    self.assertEqual(
      activation_factory_recipe(GatedPowerActivation),
      {"type": "gated_power", "n_gates": 1})

if __name__ == "__main__":
  unittest.main()
