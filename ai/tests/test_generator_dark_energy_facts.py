"""Check the dark-energy law published by the dataset generator.

Cobaya can sample one set of coordinates and calculate the values used by a
theory.  These CPU tests compile only the relevant production helper and
publication method, so they exercise the real decision without importing
Cobaya, MPI, GetDist, or a Boltzmann solver.
"""

import ast
import copy
from contextlib import contextmanager
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from emulator import fixed_facts
from compute_data_vectors.generator_ingress import (
  finite_number,
  native_boolean,
  native_integer,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "compute_data_vectors" / "generator_core.py"
MPS_GENERATOR = ROOT / "compute_data_vectors" / "dataset_generator_mps.py"


def _compile_publication_boundary():
  """Compile the production dark-energy helper and sidecar decision."""
  tree = ast.parse(GENERATOR.read_text(encoding="utf-8"), filename=str(GENERATOR))
  helpers = [
    copy.deepcopy(node) for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_dark_energy_publication_facts"
  ]
  classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "GeneratorCore"
  ]
  if len(helpers) != 1 or len(classes) != 1:
    raise AssertionError("generator dark-energy boundary has changed shape")
  methods = [
    copy.deepcopy(node) for node in classes[0].body
    if isinstance(node, ast.FunctionDef) and node.name == "_resolve_fixed_facts"
  ]
  if len(methods) != 1:
    raise AssertionError("generator has no unique _resolve_fixed_facts method")
  fixture = ast.ClassDef(
    name="PublicationCore", bases=[], keywords=[], body=methods,
    decorator_list=[])
  module = ast.Module(body=helpers + [fixture], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "np": np,
    "math": math,
    "fixed_facts": fixed_facts,
    "CMB_CL_UNITS": "muK2",
    "NEUTRINO_CONVENTION_KEY": "neutrino_hierarchy",
  }
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  core_class = namespace["PublicationCore"]
  core_class.dark_energy_helper = staticmethod(
    namespace["_dark_energy_publication_facts"])
  return core_class


@contextmanager
def _captured_output():
  """Give the MPS method an empty native-output buffer."""
  yield io.StringIO()


def _compile_mps_boundary(publication_helper):
  """Compile MPS setup and sample evaluation without its script imports."""
  tree = ast.parse(
    MPS_GENERATOR.read_text(encoding="utf-8"), filename=str(MPS_GENERATOR))
  classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "dataset"
  ]
  wanted = {
    "_read_train_args", "_read_write_base", "_compute_dvs_from_sample",
  }
  if len(classes) != 1:
    raise AssertionError("MPS generator has no unique dataset class")
  methods = [
    copy.deepcopy(node) for node in classes[0].body
    if isinstance(node, ast.FunctionDef) and node.name in wanted
  ]
  if {method.name for method in methods} != wanted:
    raise AssertionError("MPS dark-energy boundary has changed shape")
  fixture = ast.ClassDef(
    name="MPSCore", bases=[], keywords=[], body=methods, decorator_list=[])
  module = ast.Module(body=[fixture], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "np": np,
    "math": math,
    "capture_native_output": _captured_output,
    "finite_number": finite_number,
    "native_boolean": native_boolean,
    "native_integer": native_integer,
    "_dark_energy_publication_facts": publication_helper,
  }
  exec(compile(module, str(MPS_GENERATOR), "exec"), namespace)
  return namespace["MPSCore"]


class FakeParameterization:
  """Small copy of the four public Cobaya surfaces the producer reads."""

  def __init__(self, *, inputs, constants, sampled, dependencies):
    self._inputs = dict(inputs)
    self._constants = dict(constants)
    self._sampled = dict(sampled)
    self.input_dependencies = {
      name: set(required) for name, required in dependencies.items()
    }
    self.calls = []

  def input_params(self):
    self.calls.append("input_params")
    return dict(self._inputs)

  def constant_params(self):
    self.calls.append("constant_params")
    return dict(self._constants)

  def sampled_params(self):
    self.calls.append("sampled_params")
    return dict(self._sampled)


class NameListParameterization(FakeParameterization):
  """Cobaya-compatible variant returning names without placeholder values."""

  def input_params(self):
    self.calls.append("input_params")
    return list(self._inputs)

  def sampled_params(self):
    self.calls.append("sampled_params")
    return list(self._sampled)


def _facts(core_class, parameterization, *, sampled, pinned=None,
           family="mps"):
  """Run the real publication method on one inert standard-family model."""
  core = object.__new__(core_class)
  core.model = SimpleNamespace(parameterization=parameterization)
  core.sampled_params = list(sampled)
  core.dataset_route = {
    "family": family,
    "family_variant": "standard",
    "generator": ("dataset_generator_background"
                  if family == "grid" else "dataset_generator_mps"),
  }
  resolved = dict(pinned or {})
  core._resolved_constants = lambda: dict(resolved)
  return core._resolve_fixed_facts()


class GeneratorDarkEnergyFactsTests(unittest.TestCase):
  """Require canonical physical facts for direct and transformed coordinates."""

  @classmethod
  def setUpClass(cls):
    cls.core_class = _compile_publication_boundary()
    cls.mps_class = _compile_mps_boundary(cls.core_class.dark_energy_helper)

  def test_transformed_sampled_coordinates_publish_cpl(self):
    """The shipped w0pwa/w sampling route makes functional wa vary."""
    parameterization = FakeParameterization(
      inputs={"w0pwa": np.nan, "w": np.nan, "wa": np.nan},
      constants={},
      sampled={"w0pwa": np.nan, "w": np.nan},
      dependencies={"wa": {"w0pwa", "w"}})

    facts = _facts(
      self.core_class, parameterization, sampled=("w0pwa", "w"))

    self.assertEqual(facts["dark_energy_law"], "w0wa-cpl")
    self.assertEqual(facts["dark_energy_inputs"], ["w", "wa"])
    self.assertNotIn("w", facts["cosmology_fixed"])
    self.assertNotIn("wa", facts["cosmology_fixed"])
    self.assertEqual(
      parameterization.calls,
      ["input_params", "constant_params", "sampled_params"])

  def test_canonical_sampled_coordinates_publish_the_same_cpl(self):
    """Sampling w0 and wa directly has the same physical law and names."""
    parameterization = FakeParameterization(
      inputs={"w0": np.nan, "wa": np.nan},
      constants={},
      sampled={"w0": np.nan, "wa": np.nan},
      dependencies={})

    facts = _facts(
      self.core_class, parameterization, sampled=("w0", "wa"))

    self.assertEqual(facts["dark_energy_law"], "w0wa-cpl")
    self.assertEqual(facts["dark_energy_inputs"], ["w", "wa"])
    self.assertNotIn("w", facts["cosmology_fixed"])
    self.assertNotIn("wa", facts["cosmology_fixed"])

  def test_name_list_public_surfaces_keep_the_same_classification(self):
    """Public surfaces that return ordered names need no invented values."""
    parameterization = NameListParameterization(
      inputs={"w0pwa": np.nan, "w": np.nan, "wa": np.nan},
      constants={},
      sampled={"w0pwa": np.nan, "w": np.nan},
      dependencies={"wa": {"w0pwa", "w"}})

    facts = _facts(
      self.core_class, parameterization, sampled=("w0pwa", "w"))

    self.assertEqual(facts["dark_energy_law"], "w0wa-cpl")
    self.assertEqual(facts["dark_energy_inputs"], ["w", "wa"])

  def test_fixed_w0_is_published_under_canonical_w(self):
    """An alias spelling cannot produce a second fixed-fact vocabulary."""
    parameterization = FakeParameterization(
      inputs={"w0": -0.9},
      constants={"w0": -0.9},
      sampled={"H0": np.nan},
      dependencies={})

    facts = _facts(
      self.core_class, parameterization, sampled=("H0",),
      pinned={"w0": -0.9})

    self.assertEqual(facts["dark_energy_law"], "constant-w")
    self.assertEqual(facts["dark_energy_inputs"], ["w"])
    self.assertEqual(facts["cosmology_fixed"]["w"], -0.9)
    self.assertNotIn("w0", facts["cosmology_fixed"])

  def test_fixed_sum_and_w0_derive_canonical_wa(self):
    """A fixed transformed pair publishes the separate physical values."""
    parameterization = FakeParameterization(
      inputs={"w0pwa": -0.7, "w0": -0.9, "wa": 0.2},
      constants={"w0pwa": -0.7, "w0": -0.9},
      sampled={"H0": np.nan},
      dependencies={"wa": {"w0pwa", "w0"}})

    facts = _facts(
      self.core_class, parameterization, sampled=("H0",),
      pinned={"w0pwa": -0.7, "w0": -0.9})

    self.assertEqual(facts["dark_energy_law"], "w0wa-cpl")
    self.assertEqual(facts["dark_energy_inputs"], ["w", "wa"])
    self.assertAlmostEqual(facts["cosmology_fixed"]["w"], -0.9)
    self.assertAlmostEqual(facts["cosmology_fixed"]["wa"], 0.2)

  def test_fixed_canonical_lcdm_stays_canonical(self):
    """The direct (-1, 0) representation records the LCDM law."""
    parameterization = FakeParameterization(
      inputs={"w": -1.0, "wa": 0.0},
      constants={"w": -1.0, "wa": 0.0},
      sampled={"H0": np.nan},
      dependencies={})

    facts = _facts(
      self.core_class, parameterization, sampled=("H0",),
      pinned={"w": -1.0, "wa": 0.0})

    self.assertEqual(facts["dark_energy_law"], "cosmological-constant")
    self.assertEqual(facts["dark_energy_inputs"], [])
    self.assertEqual(facts["cosmology_fixed"]["w"], -1.0)
    self.assertEqual(facts["cosmology_fixed"]["wa"], 0.0)

  def test_missing_coordinates_publish_lcdm_defaults(self):
    """An implicit LCDM model records both physical fixed values."""
    parameterization = FakeParameterization(
      inputs={}, constants={}, sampled={"H0": np.nan}, dependencies={})

    facts = _facts(
      self.core_class, parameterization, sampled=("H0",), pinned={})

    self.assertEqual(facts["dark_energy_law"], "cosmological-constant")
    self.assertEqual(
      facts["cosmology_fixed"],
      {"w": -1.0, "wa": 0.0,
       "mnu": "n/a", "omk": "n/a", "TCMB": "n/a", "nnu": "n/a"})

  def test_fixed_w_publishes_constant_w_with_zero_wa(self):
    """A fixed present value records the implicit constant-w evolution."""
    parameterization = FakeParameterization(
      inputs={"w": -0.9}, constants={"w": -0.9},
      sampled={"H0": np.nan}, dependencies={})

    facts = _facts(
      self.core_class, parameterization, sampled=("H0",),
      pinned={"w": -0.9})

    self.assertEqual(facts["dark_energy_law"], "constant-w")
    self.assertEqual(facts["cosmology_fixed"]["w"], -0.9)
    self.assertEqual(facts["cosmology_fixed"]["wa"], 0.0)

  def test_background_generation_refuses_sampled_or_fixed_curvature(self):
    """The flat distance family cannot publish a curved-universe dataset."""
    cases = (
      (
        "sampled",
        FakeParameterization(
          inputs={"omk": np.nan}, constants={},
          sampled={"H0": np.nan, "omk": np.nan}, dependencies={}),
        ("H0", "omk"),
        {},
      ),
      (
        "fixed",
        FakeParameterization(
          inputs={"omk": 0.01}, constants={"omk": 0.01},
          sampled={"H0": np.nan}, dependencies={}),
        ("H0",),
        {"omk": 0.01},
      ),
    )
    for name, parameterization, sampled, pinned in cases:
      with self.subTest(curvature=name):
        with self.assertRaisesRegex(ValueError, "[Ff]lat"):
          _facts(
            self.core_class,
            parameterization,
            sampled=sampled,
            pinned=pinned,
            family="grid",
          )

  def test_background_generation_accepts_fixed_zero_curvature(self):
    """A directly named omk=0 remains an honest flat background run."""
    parameterization = FakeParameterization(
      inputs={"omk": 0.0}, constants={"omk": 0.0},
      sampled={"H0": np.nan}, dependencies={})
    facts = _facts(
      self.core_class,
      parameterization,
      sampled=("H0",),
      pinned={"omk": 0.0},
      family="grid",
    )
    self.assertIs(facts["flat_only"], True)
    self.assertEqual(facts["cosmology_fixed"]["omk"], 0.0)

  def test_inconsistent_fixed_transformation_refuses(self):
    """A sidecar cannot silently choose one of two incompatible values."""
    parameterization = FakeParameterization(
      inputs={"w0pwa": -0.7, "w": -0.9, "wa": 0.1},
      constants={"w0pwa": -0.7, "w": -0.9, "wa": 0.1},
      sampled={"H0": np.nan},
      dependencies={})

    with self.assertRaisesRegex(ValueError, "inconsistent fixed dark-energy"):
      _facts(
        self.core_class, parameterization, sampled=("H0",),
        pinned={"w0pwa": -0.7, "w": -0.9, "wa": 0.1})

  def test_inconsistent_w_and_w0_aliases_refuse(self):
    """Two names for the present-day value must not disagree."""
    parameterization = FakeParameterization(
      inputs={"w": -0.9, "w0": -0.8},
      constants={"w": -0.9, "w0": -0.8},
      sampled={"H0": np.nan},
      dependencies={})

    with self.assertRaisesRegex(ValueError, "inconsistent fixed aliases"):
      _facts(
        self.core_class, parameterization, sampled=("H0",),
        pinned={"w": -0.9, "w0": -0.8})

  def test_mps_setup_caches_transformed_cpl_before_route_binding(self):
    """MPS setup resolves its law without calling route-dependent facts."""
    parameterization = FakeParameterization(
      inputs={"w0pwa": np.nan, "w": np.nan, "wa": np.nan},
      constants={},
      sampled={"w0pwa": np.nan, "w": np.nan},
      dependencies={"wa": {"w0pwa", "w"}})
    model = SimpleNamespace(parameterization=parameterization)
    requirements = []
    model.add_requirements = requirements.append
    core = object.__new__(self.mps_class)
    core.model = model
    core._resolved_constants = lambda: {}
    core._resolve_fixed_facts = mock.Mock(
      side_effect=AssertionError("the dataset route is not bound yet"))
    train_args = {
      "z_segments": [[0.0, 1.0, 4, True]],
      "k_log10": [-4.0, -2.0, 8],
      "extrap_kmax": 1.0,
      "write_syren_base": True,
    }

    core._read_train_args(train_args)

    self.assertEqual(core.dark_energy_law, "w0wa-cpl")
    core._resolve_fixed_facts.assert_not_called()
    self.assertEqual(len(requirements), 1)

  def test_mps_sample_passes_explicit_constant_and_lcdm_laws(self):
    """Incomplete valid points are completed only by the cached model law."""
    from emulator import syren_base

    cases = (
      ("constant-w", {"w": -0.9}, (-0.9, 0.0)),
      ("cosmological-constant", {}, (-1.0, 0.0)),
    )
    for law, dark_coordinates, expected in cases:
      with self.subTest(law=law):
        resolved_calls = []
        params = {
          "As_1e9": 2.1,
          "ns": 0.965,
          "H0": 70.0,
          "omegab": 0.05,
          "omegam": 0.3,
          **dark_coordinates,
        }

        class Parameterization:
          @staticmethod
          def to_input(sampled_params_values):
            del sampled_params_values
            return dict(params)

        class Interpolator:
          @staticmethod
          def P(z, k):
            return np.ones((len(z), len(k)), dtype=np.float64)

        provider = SimpleNamespace(
          set_current_input_params=lambda value: None,
          get_Pk_interpolator=lambda *args, **kwargs: Interpolator())
        model = SimpleNamespace(
          parameterization=Parameterization(),
          provider=provider,
          prior=SimpleNamespace(logp=lambda value: 0.0),
          _component_order={},
          _params_of_dependencies=[],
        )
        core = object.__new__(self.mps_class)
        core.model = model
        core.names = ["sample"]
        core.sampled_params = ["sample"]
        core.derived = False
        core.reorder_idx_from_ord_to_yaml = lambda: np.asarray([0])
        core.extrap_kmax = 1.0
        core.z_mps = np.asarray([0.0, 1.0])
        core.k_mps = np.asarray([0.1, 0.2])
        core.dtype = np.float32
        core.write_base = True
        core.dark_energy_law = law
        production_resolver = syren_base.syren_params_from

        def resolve_spy(point, *, dark_energy_law=None):
          values = production_resolver(
            point, dark_energy_law=dark_energy_law)
          resolved_calls.append((dark_energy_law, values))
          return values

        with mock.patch.object(
            syren_base, "syren_params_from",
            side_effect=resolve_spy), mock.patch.object(
              syren_base, "base_pklin",
              return_value=np.ones((2, 2))), mock.patch.object(
                syren_base, "base_boost",
                return_value=np.ones((2, 2))):
          result = core._compute_dvs_from_sample(np.asarray([0.0]))

        self.assertEqual(len(resolved_calls), 1)
        used_law, resolved = resolved_calls[0]
        self.assertEqual(used_law, law)
        self.assertEqual(resolved[-2:], expected)
        self.assertEqual(result["pklin_base"].shape, (4,))
        self.assertEqual(result["boost_base"].shape, (4,))

  def test_conflicting_amplitudes_refuse(self):
    """Both As_1e9 and As present must be rejected as contradictory."""
    class Parameterization:
      @staticmethod
      def to_input(sampled_params_values):
        del sampled_params_values
        return {
          "As_1e9": 2.1,
          "As": 9e-9,
          "ns": 0.965,
          "H0": 67.0,
          "omegab": 0.05,
          "omegam": 0.3,
          "w": -1.0,
          "wa": 0.0,
        }

    class Interpolator:
      @staticmethod
      def P(z, k):
        return np.ones((len(z), len(k)), dtype=np.float64)

    provider = SimpleNamespace(
      set_current_input_params=lambda value: None,
      get_Pk_interpolator=lambda *args, **kwargs: Interpolator())
    model = SimpleNamespace(
      parameterization=Parameterization(),
      provider=provider,
      prior=SimpleNamespace(logp=lambda value: 0.0),
      _component_order={},
      _params_of_dependencies=[],
    )
    core = object.__new__(self.mps_class)
    core.model = model
    core.names = ["sample"]
    core.sampled_params = ["sample"]
    core.derived = False
    core.reorder_idx_from_ord_to_yaml = lambda: np.asarray([0])
    core.extrap_kmax = 1.0
    core.z_mps = np.asarray([0.0, 1.0])
    core.k_mps = np.asarray([0.1, 0.2])
    core.dtype = np.float32
    core.write_base = True
    core.dark_energy_law = "cosmological-constant"

    with self.assertRaisesRegex(ValueError, "conflicting primordial amplitudes"):
      core._compute_dvs_from_sample(np.asarray([0.0]))


if __name__ == "__main__":
  unittest.main()
