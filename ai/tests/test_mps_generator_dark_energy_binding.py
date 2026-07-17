"""Check how the matter-power generator binds dark-energy coordinates.

The matter-power generator writes a Syren analytic base beside each CAMB
training row.  The base must use the same physical dark-energy law recorded
for the dataset.  These small CPU tests compile the two relevant production
methods from their syntax tree, then replace CAMB and the Syren formulas with
in-memory witnesses.  No MPI process, model training, or scientific data is
needed.
"""

import ast
import contextlib
import copy
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from compute_data_vectors.generator_ingress import finite_number
from compute_data_vectors.generator_ingress import native_boolean
from compute_data_vectors.generator_ingress import native_integer
from emulator import syren_base


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "compute_data_vectors" / "dataset_generator_mps.py"


def _compile_generator_boundary(publication_facts):
  """Compile the real setup and row methods without importing MPI or Cobaya."""
  tree = ast.parse(GENERATOR.read_text(encoding="utf-8"), filename=str(GENERATOR))
  classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "dataset"
  ]
  if len(classes) != 1:
    raise AssertionError("matter-power generator has no unique dataset class")
  selected = {"_read_train_args", "_read_write_base", "_compute_dvs_from_sample"}
  methods = [
    copy.deepcopy(node) for node in classes[0].body
    if isinstance(node, ast.FunctionDef) and node.name in selected
  ]
  if {method.name for method in methods} != selected or len(methods) != 3:
    raise AssertionError(
      "matter-power dark-energy methods changed shape; update this focused "
      "production-method test")
  fixture = ast.ClassDef(
    name="MpsGeneratorBoundary", bases=[], keywords=[], body=methods,
    decorator_list=[])
  module = ast.Module(body=[fixture], type_ignores=[])
  ast.fix_missing_locations(module)

  @contextlib.contextmanager
  def capture_native_output():
    output = io.StringIO()
    yield output

  namespace = {
    "np": np,
    "math": math,
    "finite_number": finite_number,
    "native_boolean": native_boolean,
    "native_integer": native_integer,
    "_dark_energy_publication_facts": publication_facts,
    "capture_native_output": capture_native_output,
  }
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["MpsGeneratorBoundary"]


class _Parameterization:
  """Return one already-resolved input point, as Cobaya does per sample."""

  def __init__(self, point, law):
    self.point = dict(point)
    self.law = law

  def to_input(self, *, sampled_params_values):
    del sampled_params_values
    return dict(self.point)


class _Interpolator:
  """Return one positive two-by-two power surface."""

  def __init__(self, value):
    self.value = value

  def P(self, z, k):
    return np.full((len(z), len(k)), self.value, dtype=np.float64)


class _Provider:
  """Record the input point and serve linear/nonlinear power surfaces."""

  def __init__(self):
    self.current = None

  def set_current_input_params(self, point):
    self.current = dict(point)

  def get_Pk_interpolator(self, pair, *, nonlinear, extrap_kmax):
    del pair, extrap_kmax
    return _Interpolator(4.0 if nonlinear else 2.0)


def _scientific_point(**dark_energy):
  """Return the non-dark-energy values required by the Syren base."""
  point = {
    "As": 2.1e-9,
    "ns": 0.966,
    "H0": 67.3,
    "omegab": 0.049,
    "omegam": 0.31,
  }
  point.update(dark_energy)
  return point


class MpsGeneratorDarkEnergyBindingTests(unittest.TestCase):
  """Require one explicit physical law from setup through every base row."""

  def _production_class(self, calls):
    def publication_facts(parameterization, pinned):
      calls.append((parameterization, dict(pinned)))
      return ({}, set(), parameterization.law, [])

    return _compile_generator_boundary(publication_facts)

  def test_setup_caches_each_explicit_law_once(self):
    """Setup asks the publication boundary once, before any row is generated."""
    for law in ("w0wa-cpl", "constant-w", "cosmological-constant"):
      with self.subTest(law=law):
        calls = []
        generator_class = self._production_class(calls)
        generator = generator_class()
        parameterization = _Parameterization({}, law)
        requirements = []
        generator.model = SimpleNamespace(
          parameterization=parameterization,
          add_requirements=requirements.append)
        generator._resolved_constants = lambda: {"H0": 67.3}
        generator._read_train_args({
          "z_segments": [[0.0, 1.0, 4, True]],
          "k_log10": [-2.0, 0.0, 8],
          "extrap_kmax": 2.0,
          "write_syren_base": False,
        })

        self.assertEqual(generator.dark_energy_law, law)
        self.assertEqual(calls, [(parameterization, {"H0": 67.3})])
        self.assertEqual(len(requirements), 1)

  def _generate_base(self, point, law):
    calls = []
    generator_class = self._production_class([])
    generator = generator_class()
    provider = _Provider()
    generator.model = SimpleNamespace(
      parameterization=_Parameterization(point, law),
      provider=provider,
      prior=SimpleNamespace(logp=lambda sample: 0.0),
      _component_order={},
      _params_of_dependencies=())
    generator.reorder_idx_from_ord_to_yaml = lambda: np.asarray([0])
    generator.names = ("sample",)
    generator.sampled_params = ("sample",)
    generator.derived = False
    generator.extrap_kmax = 2.0
    generator.z_mps = np.asarray([0.0, 1.0])
    generator.k_mps = np.asarray([0.1, 1.0])
    generator.dtype = np.float32
    generator.write_base = True
    generator.dark_energy_law = law

    production_resolver = syren_base.syren_params_from

    def resolver_spy(params, *, dark_energy_law=None):
      result = production_resolver(
        params, dark_energy_law=dark_energy_law)
      calls.append((dict(params), dark_energy_law, result[-2:]))
      return result

    base_coordinates = []

    def base_pklin(*, k_mpc, z, w0, wa, **params):
      del params
      base_coordinates.append(("pklin", w0, wa))
      return np.ones((len(z), len(k_mpc)), dtype=np.float64)

    def base_boost(*, k_mpc, z, w0, wa, **params):
      del params
      base_coordinates.append(("boost", w0, wa))
      return np.ones((len(z), len(k_mpc)), dtype=np.float64)

    with mock.patch.object(
        syren_base, "syren_params_from", side_effect=resolver_spy), \
        mock.patch.object(syren_base, "base_pklin", side_effect=base_pklin), \
        mock.patch.object(syren_base, "base_boost", side_effect=base_boost):
      result = generator._compute_dvs_from_sample(np.asarray([0.0]))
    return result, calls, base_coordinates, provider

  def test_transformed_nonzero_wa_reaches_both_base_formulas(self):
    """The sampled ``w0pwa, w`` form derives a nonzero ``wa`` without loss."""
    point = _scientific_point(w=-0.9, w0pwa=-0.7)
    result, calls, coordinates, provider = self._generate_base(
      point, "w0wa-cpl")

    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0][:2], (point, "w0wa-cpl"))
    np.testing.assert_allclose(calls[0][2], (-0.9, 0.2), rtol=0.0, atol=1e-15)
    self.assertEqual([entry[0] for entry in coordinates], ["pklin", "boost"])
    np.testing.assert_allclose(
      [entry[1:] for entry in coordinates],
      [(-0.9, 0.2), (-0.9, 0.2)], rtol=0.0, atol=1e-15)
    self.assertEqual(provider.current, point)
    self.assertEqual(
      set(result), {"pklin", "boost", "pklin_base", "boost_base"})

  def test_constant_w_generation_passes_constant_law(self):
    """A missing ``wa`` becomes zero only under the explicit constant-w law."""
    point = _scientific_point(w=-0.9)
    _, calls, coordinates, _ = self._generate_base(point, "constant-w")

    self.assertEqual(calls, [(point, "constant-w", (-0.9, 0.0))])
    self.assertEqual(
      coordinates, [("pklin", -0.9, 0.0), ("boost", -0.9, 0.0)])

  def test_cosmological_constant_generation_passes_lcdm_law(self):
    """LCDM defaults are available only under the explicit persisted law."""
    point = _scientific_point()
    _, calls, coordinates, _ = self._generate_base(
      point, "cosmological-constant")

    self.assertEqual(
      calls, [(point, "cosmological-constant", (-1.0, 0.0))])
    self.assertEqual(
      coordinates, [("pklin", -1.0, 0.0), ("boost", -1.0, 0.0)])

  def test_cpl_present_day_value_alone_cannot_silently_supply_wa(self):
    """A CPL row with neither ``wa`` nor ``w0pwa`` stops before base math."""
    with self.assertRaisesRegex(ValueError, "cannot resolve wa"):
      self._generate_base(_scientific_point(w=-0.9), "w0wa-cpl")


if __name__ == "__main__":
  unittest.main()
