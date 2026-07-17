"""CPU tests for the generator's validated route and checkpoint census.

The generator module imports MPI, Cobaya, GetDist, emcee, and NumPy. These
tests compile the real selected ``GeneratorCore`` and driver methods from their
syntax trees. No configured scientific run or checkpoint file is needed.
"""

import ast
import copy
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from compute_data_vectors import dataset_manifest
from compute_data_vectors.dataset_manifest import (
  DATASET_PROBE_FAMILIES,
  DATASET_PROBE_GENERATORS,
  UNIFORM_BOUNDARY_INTERIOR_POLICY,
  build_dataset_member_census,
  build_dataset_member_map,
  build_dataset_request_identity,
)
from emulator import fixed_facts


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "compute_data_vectors" / "generator_core.py"
MPS_GENERATOR = ROOT / "compute_data_vectors" / "dataset_generator_mps.py"
CASES = (
  {
    "label": "cosmolike",
    "family": "cosmolike",
    "variant": "standard",
    "probe": "cs",
    "generator": "dataset_generator_lensing",
    "source": ROOT / "compute_data_vectors" / "dataset_generator_lensing.py",
  },
  {
    "label": "cmb",
    "family": "cmb",
    "variant": "standard",
    "probe": "cmblensed",
    "generator": "dataset_generator_cmb",
    "source": ROOT / "compute_data_vectors" / "dataset_generator_cmb.py",
  },
  {
    "label": "background",
    "family": "grid",
    "variant": "standard",
    "probe": "background",
    "generator": "dataset_generator_background",
    "source": (
      ROOT / "compute_data_vectors" / "dataset_generator_background.py"),
  },
  {
    "label": "mps-native",
    "family": "grid2d",
    "variant": "native",
    "probe": "mps",
    "generator": "dataset_generator_mps",
    "source": MPS_GENERATOR,
  },
  {
    "label": "mps-syren-base",
    "family": "grid2d",
    "variant": "syren-base",
    "probe": "mps",
    "generator": "dataset_generator_mps",
    "source": MPS_GENERATOR,
  },
)
CORE_METHODS = {
  "_generator_program_name",
  "_family_variant",
  "_bind_dataset_member_census",
  "_checkpoint_member_paths",
  "_dv_chk_files",
  "_resolve_fixed_facts",
}


def _identity(case, dataset_mode):
  """Build one complete identity for comparison with the early census."""
  return build_dataset_request_identity(
    dataset_mode=dataset_mode,
    family=case["family"],
    family_variant=case["variant"],
    generator=case["generator"],
    probe=case["probe"],
    sampling_mode="uniform",
    temperature=64,
    boundary_factor=1.0,
    max_correlation=None,
    sampling_algorithm="uniform-box-v1",
    seed=17,
    rng_bit_generator="PCG64",
    rng_emcee_random=None,
    rng_policy="persist-complete-state-v1",
    boundary_interior_policy=UNIFORM_BOUNDARY_INTERIOR_POLICY,
    ordered_names=["H0", "omegabh2"],
    configuration_sha256="1" * 64,
    scientific_contract_sha256="a" * 64)


def _census(case, dataset_mode, params="params", dvs="dvs", fail="fail"):
  """Build one early member census with the case's canonical route."""
  return build_dataset_member_census(
    dataset_mode=dataset_mode,
    family=case["family"],
    family_variant=case["variant"],
    generator=case["generator"],
    probe=case["probe"],
    params_stem=params,
    dvs_stem=dvs,
    fail_stem=fail)


def _class_node(source, class_name):
  """Return one copied class node from source text."""
  tree = ast.parse(source)
  classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == class_name
  ]
  if len(classes) != 1:
    raise AssertionError(
      "expected exactly one class " + class_name + ", found "
      + str(len(classes)))
  return copy.deepcopy(classes[0])


def _method_node(source, class_name, method_name):
  """Return one copied method from one named class."""
  production = _class_node(source, class_name)
  methods = [
    node for node in production.body
    if isinstance(node, ast.FunctionDef) and node.name == method_name
  ]
  if len(methods) != 1:
    raise AssertionError(
      "expected exactly one method " + method_name + ", found "
      + str(len(methods)))
  return copy.deepcopy(methods[0])


def _is_self_call(node, method_name):
  """Say whether one syntax node calls the named method on self."""
  return (
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and isinstance(node.func.value, ast.Name)
    and node.func.value.id == "self"
    and node.func.attr == method_name)


def _assert_setup_member_binding_order(source):
  """Require setup to bind once after settings and scoped output stems."""
  method = _method_node(source, "GeneratorCore", "__setup_flags")
  all_bind_calls = [
    node for node in ast.walk(method)
    if _is_self_call(node, "_bind_dataset_member_census")
  ]
  if len(all_bind_calls) != 1:
    raise AssertionError("setup must contain exactly one census-binding call")

  bind_positions = [
    index for index, statement in enumerate(method.body)
    if isinstance(statement, ast.Expr)
    and statement.value is all_bind_calls[0]
  ]
  if len(bind_positions) != 1:
    raise AssertionError("the census-binding call must be a direct setup step")

  read_positions = []
  for index, statement in enumerate(method.body):
    if not isinstance(statement, ast.Expr):
      continue
    call = statement.value
    if not _is_self_call(call, "_read_train_args"):
      continue
    if (len(call.args) != 1
        or not isinstance(call.args[0], ast.Name)
        or call.args[0].id != "train_args"
        or call.keywords):
      raise AssertionError("driver settings must receive train_args directly")
    read_positions.append(index)
  if len(read_positions) != 1:
    raise AssertionError("setup must read driver settings exactly once")

  scope_positions = {name: [] for name in ("dvsf", "paramsf", "failf")}
  for index, statement in enumerate(method.body):
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
      continue
    target = statement.targets[0]
    value = statement.value
    if (not isinstance(target, ast.Attribute)
        or not isinstance(target.value, ast.Name)
        or target.value.id != "self"
        or target.attr not in scope_positions
        or not isinstance(value, ast.Call)
        or not isinstance(value.func, ast.Name)
        or value.func.id != "scope_dataset_stem"):
      continue
    scope_positions[target.attr].append(index)
  if any(len(positions) != 1 for positions in scope_positions.values()):
    raise AssertionError("setup must scope each output stem exactly once")

  bind_position = bind_positions[0]
  required_positions = [read_positions[0]]
  required_positions.extend(
    positions[0] for positions in scope_positions.values())
  if bind_position <= max(required_positions):
    raise AssertionError(
      "census binding must follow driver settings and all scoped stems")


def _assert_mps_write_base_binding(source):
  """Require MPS settings to use the native-boolean validator."""
  method = _method_node(source, "dataset", "_read_train_args")
  assignments = []
  for statement in ast.walk(method):
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
      continue
    target = statement.targets[0]
    if (isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr == "write_base"):
      assignments.append(statement)
  if len(assignments) != 1 or assignments[0] not in method.body:
    raise AssertionError("MPS settings must assign write_base once directly")

  value = assignments[0].value
  if (not _is_self_call(value, "_read_write_base")
      or len(value.args) != 1
      or not isinstance(value.args[0], ast.Name)
      or value.args[0].id != "train_args"
      or value.keywords):
    raise AssertionError(
      "MPS write_base must come from _read_write_base(train_args)")
  validator_calls = [
    node for node in ast.walk(method)
    if _is_self_call(node, "_read_write_base")
  ]
  if validator_calls != [value]:
    raise AssertionError("MPS settings must use the base validator once")


def _compile_core(source=None):
  """Compile the selected real GeneratorCore methods without heavy imports."""
  if source is None:
    source = GENERATOR.read_text(encoding="utf-8")
  tree = ast.parse(source, filename=str(GENERATOR))
  helpers = [
    copy.deepcopy(node) for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_dark_energy_publication_facts"
  ]
  if len(helpers) != 1:
    raise AssertionError(
      "expected one generator dark-energy publication helper, found "
      + str(len(helpers)))
  production = _class_node(source, "GeneratorCore")
  methods = [
    copy.deepcopy(node) for node in production.body
    if isinstance(node, ast.FunctionDef) and node.name in CORE_METHODS
  ]
  observed_names = {method.name for method in methods}
  if observed_names != CORE_METHODS:
    raise AssertionError(
      "selected GeneratorCore methods differ: " + repr(observed_names))
  fixture = ast.ClassDef(
    name="BindingCore",
    bases=[],
    keywords=[],
    body=methods,
    decorator_list=[])
  module = ast.Module(body=helpers + [fixture], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "__name__": "_generator_member_core_fixture",
    "math": math,
    "np": np,
    "Path": Path,
    "os": os,
    "sys": sys,
    "fixed_facts": fixed_facts,
    "CMB_CL_UNITS": "muK2",
    "NEUTRINO_CONVENTION_KEY": "neutrino_hierarchy",
    "DATASET_PROBE_FAMILIES": DATASET_PROBE_FAMILIES,
    "DATASET_PROBE_GENERATORS": DATASET_PROBE_GENERATORS,
    "build_dataset_member_census": build_dataset_member_census,
  }
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["BindingCore"]


def _mps_variant_method(source=None):
  """Return the real MPS variant method, optionally from mutant source."""
  if source is None:
    source = MPS_GENERATOR.read_text(encoding="utf-8")
  production = _class_node(source, "dataset")
  methods = [
    copy.deepcopy(node) for node in production.body
    if isinstance(node, ast.FunctionDef) and node.name == "_family_variant"
  ]
  if len(methods) != 1:
    raise AssertionError("expected one MPS _family_variant method")
  fixture = ast.ClassDef(
    name="MpsVariant",
    bases=[],
    keywords=[],
    body=methods,
    decorator_list=[])
  module = ast.Module(body=[fixture], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {"__name__": "_generator_member_mps_fixture"}
  exec(compile(module, str(MPS_GENERATOR), "exec"), namespace)
  return namespace["MpsVariant"].__dict__["_family_variant"]


def _mps_write_base_method(source=None):
  """Return the real MPS YAML boolean validator."""
  if source is None:
    source = MPS_GENERATOR.read_text(encoding="utf-8")
  production = _class_node(source, "dataset")
  methods = [
    copy.deepcopy(node) for node in production.body
    if isinstance(node, ast.FunctionDef) and node.name == "_read_write_base"
  ]
  if len(methods) != 1:
    raise AssertionError("expected one MPS _read_write_base method")
  fixture = ast.ClassDef(
    name="MpsWriteBase",
    bases=[],
    keywords=[],
    body=methods,
    decorator_list=[])
  module = ast.Module(body=[fixture], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {"__name__": "_generator_member_mps_bool_fixture"}
  exec(compile(module, str(MPS_GENERATOR), "exec"), namespace)
  return namespace["MpsWriteBase"].__dict__["_read_write_base"]


def _new_driver(core_class, defining_file, mps_variant=None):
  """Create one lightweight concrete driver with a registered source file."""
  module_name = "_generator_member_driver_" + str(id(defining_file))
  module = types.ModuleType(module_name)
  if defining_file is not None:
    module.__file__ = str(defining_file)
  sys.modules[module_name] = module
  attributes = {"__module__": module_name}
  if mps_variant is not None:
    attributes["_family_variant"] = mps_variant
  driver_class = type("dataset", (core_class,), attributes)
  return object.__new__(driver_class), module_name


def _bind_instance(core_class, case, directory, dataset_mode="full",
                   defining_file=None, params_name="params", dvs_name="dvs",
                   fail_name="fail", variant_method=None):
  """Bind one production-method fixture and return its module for cleanup."""
  if defining_file is None:
    defining_file = case["source"]
  if variant_method is None and case["family"] == "grid2d":
    variant_method = _mps_variant_method()
  instance, module_name = _new_driver(
    core_class=core_class,
    defining_file=defining_file,
    mps_variant=variant_method)
  root = Path(directory)
  instance.run_control = types.SimpleNamespace(dataset_mode=dataset_mode)
  instance.probe = case["probe"]
  instance.paramsf = str(root / params_name)
  instance.dvsf = str(root / dvs_name)
  instance.failf = str(root / fail_name)
  instance.write_base = case["variant"] == "syren-base"
  instance._bind_dataset_member_census()
  return instance, module_name


def _family_checkpoint_paths(core_class, case, directory, source=None):
  """Run the actual driver's inherited or overridden checkpoint method."""
  if source is None:
    source = case["source"].read_text(encoding="utf-8")
  source_tree = ast.parse(source)
  production = _class_node(source, "dataset")
  wanted = {"_dv_chk_files"}
  if case["family"] == "grid2d":
    wanted.add("_quantities")
  methods = [
    copy.deepcopy(node) for node in production.body
    if isinstance(node, ast.FunctionDef) and node.name in wanted
  ]
  constants = []
  for node in source_tree.body:
    if not isinstance(node, ast.Assign):
      continue
    names = [target.id for target in node.targets
             if isinstance(target, ast.Name)]
    if not set(names).intersection({"SPECTRA", "QUANTITIES"}):
      continue
    try:
      ast.literal_eval(node.value)
    except (TypeError, ValueError) as exc:
      raise AssertionError(
        "checkpoint member constants must remain literal") from exc
    constants.append(copy.deepcopy(node))
  fixture = ast.ClassDef(
    name="FamilyCheckpoint",
    bases=[ast.Name(id="BindingCore", ctx=ast.Load())],
    keywords=[],
    body=methods if methods else [ast.Pass()],
    decorator_list=[])
  module = ast.Module(body=constants + [fixture], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {"BindingCore": core_class}
  exec(compile(module, str(case["source"]), "exec"), namespace)
  family_class = namespace["FamilyCheckpoint"]
  instance = object.__new__(family_class)
  instance.dvsf = str(Path(directory) / "dvs")
  instance.write_base = case["variant"] == "syren-base"
  return tuple(Path(path) for path in instance._dv_chk_files())


class GeneratorMemberBindingTests(unittest.TestCase):
  """Bind one canonical member census before any checkpoint inspection."""

  def test_setup_binds_once_after_driver_settings_and_scoped_stems(self):
    source = GENERATOR.read_text(encoding="utf-8")
    _assert_setup_member_binding_order(source)

  def test_mps_settings_use_the_native_boolean_validator(self):
    source = MPS_GENERATOR.read_text(encoding="utf-8")
    _assert_mps_write_base_binding(source)

  def test_census_matches_complete_identity_for_both_modes_and_all_cases(self):
    for dataset_mode in ("full", "chain-only"):
      for case in CASES:
        with self.subTest(mode=dataset_mode, case=case["label"]):
          census = _census(case, dataset_mode)
          identity_members = build_dataset_member_map(
            _identity(case, dataset_mode),
            params_stem="params",
            dvs_stem="dvs",
            fail_stem="fail")
          self.assertEqual(dict(census.members), identity_members)
          self.assertEqual(dict(census.route), {
            "dataset_mode": dataset_mode,
            "family": case["family"],
            "family_variant": case["variant"],
            "generator": case["generator"],
            "probe": case["probe"],
          })
          with self.assertRaises(TypeError):
            census.route["probe"] = "other"
          with self.assertRaises(TypeError):
            census.members["new-role"] = "new-file"

  def test_real_core_methods_return_every_bound_absolute_path(self):
    core_class = _compile_core()
    for dataset_mode in ("full", "chain-only"):
      for case in CASES:
        with self.subTest(mode=dataset_mode, case=case["label"]):
          with tempfile.TemporaryDirectory() as directory:
            instance, module_name = _bind_instance(
              core_class, case, directory, dataset_mode=dataset_mode)
            try:
              expected = tuple(
                Path(directory) / relative
                for relative in instance.dataset_members.values())
              observed = tuple(instance._checkpoint_member_paths())
              self.assertEqual(observed, expected)
              self.assertTrue(all(path.is_absolute() for path in observed))
              self.assertEqual(
                instance.dataset_member_directory, Path(directory))
            finally:
              sys.modules.pop(module_name, None)

  def test_full_census_matches_each_driver_checkpoint_method(self):
    core_class = _compile_core()
    for case in CASES:
      with self.subTest(case=case["label"]):
        with tempfile.TemporaryDirectory() as directory:
          census = _census(case, "full")
          selected_roles = (
            "payload.",
            "axis.",
            "base.",
          )
          census_paths = {
            Path(directory) / relative
            for role, relative in census.members.items()
            if role.startswith(selected_roles)
          }
          driver_paths = set(_family_checkpoint_paths(
            core_class, case, directory))
          self.assertEqual(driver_paths, census_paths)

  def test_lensing_fixture_observes_a_driver_checkpoint_override(self):
    source = CASES[0]["source"].read_text(encoding="utf-8")
    marker = "  EXTRA_TRAIN_KEYS = ()\n"
    self.assertEqual(source.count(marker), 1)
    mutated = source.replace(
      marker,
      marker
      + "\n"
      + "  def _dv_chk_files(self):\n"
      + "    return [f'{self.dvsf}_driver_override.npy']\n",
      1)
    core_class = _compile_core()
    with tempfile.TemporaryDirectory() as directory:
      observed = _family_checkpoint_paths(
        core_class, CASES[0], directory, source=mutated)
      self.assertEqual(
        observed,
        (Path(directory) / "dvs_driver_override.npy",))

  def test_chain_only_never_calls_the_old_data_vector_census(self):
    core_class = _compile_core()
    case = CASES[0]
    with tempfile.TemporaryDirectory() as directory:
      instance, module_name = _bind_instance(
        core_class, case, directory, dataset_mode="chain-only")
      calls = []

      def reject_old_census():
        calls.append("called")
        raise AssertionError("chain-only called _dv_chk_files")

      instance._dv_chk_files = reject_old_census
      try:
        paths = instance._checkpoint_member_paths()
        self.assertEqual(len(paths), 5)
        self.assertEqual(calls, [])
        self.assertFalse(any("dvs" in path.name for path in paths))
        self.assertFalse(any("fail" in path.name for path in paths))
      finally:
        sys.modules.pop(module_name, None)

  def test_mps_variant_switch_adds_or_removes_exactly_both_base_roles(self):
    core_class = _compile_core()
    mps_method = _mps_variant_method()
    native_case = CASES[3]
    based_case = CASES[4]
    with tempfile.TemporaryDirectory() as directory:
      native, native_module = _bind_instance(
        core_class, native_case, directory, variant_method=mps_method)
      based, based_module = _bind_instance(
        core_class, based_case, directory, variant_method=mps_method)
      try:
        self.assertEqual(native._family_variant(), "native")
        self.assertEqual(based._family_variant(), "syren-base")
        native_base = {
          role for role in native.dataset_members if role.startswith("base.")
        }
        based_base = {
          role for role in based.dataset_members if role.startswith("base.")
        }
        self.assertEqual(native_base, set())
        self.assertEqual(based_base, {
          "base.grid2d.pklin",
          "base.grid2d.boost",
        })
        common_native = {
          role: value for role, value in native.dataset_members.items()
          if not role.startswith("base.")
        }
        common_based = {
          role: value for role, value in based.dataset_members.items()
          if not role.startswith("base.")
        }
        self.assertEqual(common_native, common_based)
      finally:
        sys.modules.pop(native_module, None)
        sys.modules.pop(based_module, None)

  def test_mps_base_switch_requires_a_yaml_boolean(self):
    method = _mps_write_base_method()
    instance = object()
    self.assertIs(method(instance, {"write_syren_base": False}), False)
    self.assertIs(method(instance, {"write_syren_base": True}), True)
    for wrong in (0, 1, "false", "true", None, [], {}):
      with self.subTest(value=wrong):
        with self.assertRaisesRegex(ValueError, "YAML boolean"):
          method(instance, {"write_syren_base": wrong})

  def test_binding_normalizes_paths_without_reading_them(self):
    core_class = _compile_core()
    case = CASES[0]
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      nested = root / "unused" / ".."
      instance, module_name = _new_driver(
        core_class=core_class,
        defining_file=case["source"])
      instance.run_control = types.SimpleNamespace(dataset_mode="full")
      instance.probe = case["probe"]
      instance.paramsf = str(nested / "params")
      instance.dvsf = str(nested / "dvs")
      instance.failf = str(nested / "fail")
      try:
        with mock.patch("builtins.open", side_effect=AssertionError(
            "path normalization read a file")):
          instance._bind_dataset_member_census()
        self.assertEqual(instance.dataset_member_directory, root)
        self.assertEqual(
          tuple(instance._checkpoint_member_paths()),
          tuple(root / value for value in instance.dataset_members.values()))
      finally:
        sys.modules.pop(module_name, None)

  def test_invalid_route_or_stems_refuse_before_any_file_probe(self):
    core_class = _compile_core()
    case = CASES[0]
    file_probes = []

    def reject_file_probe(*args, **kwargs):
      file_probes.append((args, kwargs))
      raise AssertionError("validation opened a checkpoint file")

    with mock.patch("builtins.open", side_effect=reject_file_probe):
      with self.assertRaisesRegex(ValueError, "belongs to family"):
        build_dataset_member_census(
          dataset_mode="full",
          family="cmb",
          family_variant="standard",
          generator="dataset_generator_lensing",
          probe="cs",
          params_stem="params",
          dvs_stem="dvs",
          fail_stem="fail")
      with self.assertRaisesRegex(ValueError, "grid2d family variant"):
        build_dataset_member_census(
          dataset_mode="full",
          family="grid2d",
          family_variant="standard",
          generator="dataset_generator_mps",
          probe="mps",
          params_stem="params",
          dvs_stem="dvs",
          fail_stem="fail")

      refusal_cases = (
        ("wrong driver", {"defining_file": CASES[1]["source"]},
         "requires generator"),
        ("different folders", {"dvs_name": "other/dvs"},
         "share one folder"),
        ("case collision", {"params_name": "DATA", "dvs_name": "data"},
         "case-insensitive"),
        ("missing defining file", {"defining_file": False},
         "cannot be proved"),
      )
      for label, options, message in refusal_cases:
        with self.subTest(case=label):
          defining_file = options.pop("defining_file", None)
          if defining_file is False:
            defining_file = None
            instance, module_name = _new_driver(
              core_class, defining_file=None)
            root = Path(tempfile.gettempdir()) / "member-binding-no-file"
            instance.run_control = types.SimpleNamespace(dataset_mode="full")
            instance.probe = case["probe"]
            instance.paramsf = str(root / "params")
            instance.dvsf = str(root / "dvs")
            instance.failf = str(root / "fail")
            try:
              with self.assertRaisesRegex(ValueError, message):
                instance._bind_dataset_member_census()
            finally:
              sys.modules.pop(module_name, None)
          else:
            with tempfile.TemporaryDirectory() as directory:
              module_name = None
              try:
                with self.assertRaisesRegex(ValueError, message):
                  _, module_name = _bind_instance(
                    core_class,
                    case,
                    directory,
                    defining_file=defining_file,
                    **options)
              finally:
                if module_name is not None:
                  sys.modules.pop(module_name, None)
    self.assertEqual(file_probes, [])

  def test_fixed_facts_reuse_the_route_bound_during_setup(self):
    core_class = _compile_core()
    case = CASES[4]
    with tempfile.TemporaryDirectory() as directory:
      instance, module_name = _bind_instance(core_class, case, directory)
      try:
        instance.sampled_params = ()
        instance.model = types.SimpleNamespace(
          parameterization=types.SimpleNamespace(
            input_params=lambda: {},
            constant_params=lambda: {},
            sampled_params=lambda: {},
            input_dependencies={}))
        instance._resolved_constants = lambda: {}
        instance._syren_base_identity = lambda: "bound-syren-base"
        paths_before = tuple(instance._checkpoint_member_paths())
        instance.paramsf = str(Path(directory) / "other-params")
        instance.dvsf = str(Path(directory) / "other-dvs")
        instance.failf = str(Path(directory) / "other-fail")
        instance.write_base = False
        instance.probe = "cmblensed"
        sys.modules[module_name].__file__ = str(CASES[1]["source"])
        facts = instance._resolve_fixed_facts()
        self.assertEqual(facts["family"], "grid2d")
        self.assertEqual(facts["generator"], "dataset_generator_mps")
        self.assertEqual(facts["cl_units"], fixed_facts.NOT_APPLICABLE)
        self.assertEqual(facts["base_identity"], "bound-syren-base")
        self.assertEqual(tuple(instance._checkpoint_member_paths()),
                         paths_before)
      finally:
        sys.modules.pop(module_name, None)


if __name__ == "__main__":
  unittest.main()
