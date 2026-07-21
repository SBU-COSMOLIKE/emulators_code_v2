"""CPU checks for numerical validation at public inference boundaries.

The tests use tiny stand-ins for networks and geometries.  They do not train a
model.  Each stand-in places one bad value at one named stage, which proves the
public predictor refuses that stage before a later transform can hide or cache
the value.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import scipy.interpolate  # Load once before temporary Cobaya module stubs.
import torch

from emulator import fixed_facts, results
from emulator.inference import EmulatorPredictor
from emulator.losses.cmb import make_cmb_chi2
from emulator.losses.core import RescaledChi2


ROOT = Path(__file__).resolve().parents[2]


class _ParameterGeometry:
  """Return a caller-selected encoded row."""

  names = ["p0", "p1"]

  def __init__(self, encoded):
    self.encoded = encoded

  def encode(self, row):
    del row
    return self.encoded


class _Model:
  """Return a caller-selected model result."""

  def __init__(self, output):
    self.output = output

  def __call__(self, encoded):
    del encoded
    return self.output


class _VectorGeometry:
  """Scatter two kept values into a three-element public vector."""

  PROBE_BLOCKS = {"xi": (0,)}

  def __init__(self, scattered=None):
    self.dest_idx = torch.arange(2)
    self.total_size = 3
    self.scattered = scattered

  def unsqueeze(self, kept):
    if self.scattered is not None:
      return self.scattered
    return torch.cat(
      [kept, torch.zeros((kept.shape[0], 1), dtype=kept.dtype)], dim=1)


def _predictor(*, encoded=None, model_output=None, decoded=None,
               scattered=None):
  """Build one plain data-vector predictor without reading artifact files."""
  if encoded is None:
    encoded = torch.tensor([[0.1, 0.2]])
  if model_output is None:
    model_output = torch.tensor([[0.3, 0.4]])
  if decoded is None:
    decoded = torch.tensor([[3.0, 4.0]])
  predictor = EmulatorPredictor.__new__(EmulatorPredictor)
  predictor.names = ["p0", "p1"]
  predictor._where = "test-artifact"
  predictor._support = object()
  predictor._dtype = torch.float32
  predictor.device = torch.device("cpu")
  predictor._input_dim = 2
  predictor._decoded_dim = 2
  predictor._model_output_shape = (1, 2)
  predictor.pgeom = _ParameterGeometry(encoded)
  predictor.model = _Model(model_output)
  predictor._decode = lambda pred, x_enc: decoded
  predictor._scalar = False
  predictor._grid = False
  predictor._grid2d = False
  predictor._cmb = False
  predictor.geom = _VectorGeometry(scattered=scattered)
  predictor.total_size = 3
  predictor.dest_idx = predictor.geom.dest_idx
  predictor.dv_return = "3x2pt"
  predictor.section_sizes = [3]
  predictor.probe = "xi"
  return predictor


class _CmbParameterGeometry:
  """Identity map for the physical (A_s, tau) pair."""

  names = ["As", "tau"]

  def decode(self, encoded):
    return encoded


class _CmbGeometry:
  """Two-multipole identity geometry for amplitude-law tests."""

  def __init__(self):
    self.dest_idx = torch.arange(2)
    self.center = torch.zeros(2)

  def squeeze(self, value):
    return value

  def whiten(self, value):
    return value

  def unwhiten(self, value):
    return value


class _NativeAttrs(dict):
  """Small HDF5-attribute stand-in with caller-selected storage dtype."""

  def __init__(self, values, dtype=None):
    super().__init__(values)
    self.dtype = dtype or np.dtype("O", metadata={"vlen": str})

  def get_id(self, key):
    del key
    return types.SimpleNamespace(dtype=self.dtype)


def _load_adapter(filename, module_name):
  """Load a Cobaya adapter with small dependency stubs for CPU tests."""
  cobaya_module = types.ModuleType("cobaya")
  theory_module = types.ModuleType("cobaya.theory")
  log_module = types.ModuleType("cobaya.log")

  class _Theory:
    renames = {}
    extra_args = {}

    def initialize(self):
      return None

  theory_module.Theory = _Theory
  log_module.LoggedError = RuntimeError
  log_module.get_logger = lambda name: name
  replacements = {
    "cobaya": cobaya_module,
    "cobaya.theory": theory_module,
    "cobaya.log": log_module,
  }
  spec = importlib.util.spec_from_file_location(
    module_name, ROOT / "cobaya_theory" / filename)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, replacements):
    spec.loader.exec_module(module)
  return module


class _Log:
  """Accept adapter debug messages without printing them."""

  def debug(self, message):
    del message


class PublicPredictionValidationTests(unittest.TestCase):
  """Exercise the raw, encoded, model, decoder, and return boundaries."""

  def _predict(self, predictor, values=None):
    if values is None:
      values = {"p0": 0.1, "p1": 0.2}
    with mock.patch.object(fixed_facts, "check_support"):
      return predictor.predict(values)

  def test_raw_values_are_finite_real_scalars_before_support_check(self):
    invalid = (True, np.bool_(False), "0.1", [0.1], np.nan, np.inf, -np.inf,
               10 ** 1000)
    for value in invalid:
      forms = ({"p0": value, "p1": 0.2},
               (["p0", "p1"], [value, 0.2]))
      for form in forms:
        with self.subTest(value=repr(value), form=type(form).__name__):
          predictor = _predictor()
          with mock.patch.object(fixed_facts, "check_support") as support:
            with self.assertRaisesRegex(
                (TypeError, ValueError), r"parameter 'p0'.*(finite|scalar)"):
              predictor.predict(form)
          support.assert_not_called()

  def test_numpy_real_scalars_keep_the_finite_control_unchanged(self):
    result = self._predict(
      _predictor(), {"p0": np.float32(0.1), "p1": np.int64(1)})
    np.testing.assert_array_equal(result, np.array([3.0, 4.0, 0.0]))

  def test_named_pair_value_count_is_exact(self):
    predictor = _predictor()
    for values in ([0.1], [0.1, 0.2, 0.3]):
      with self.subTest(values=values):
        with self.assertRaisesRegex(ValueError, r"got .* values"):
          self._predict(predictor, (["p0", "p1"], values))

  def test_each_tensor_stage_names_its_own_failure(self):
    cases = (
      ("parameter encoding", {"encoded": torch.tensor([[np.nan, 0.2]])}),
      ("parameter encoding", {"encoded": torch.tensor([[0.1]])}),
      ("parameter encoding", {"encoded": torch.tensor([[1, 2]])}),
      ("model evaluation", {"model_output": torch.tensor([[np.inf, 0.4]])}),
      ("model evaluation", {"model_output": torch.tensor([0.3, 0.4])}),
      ("model evaluation", {"model_output": torch.tensor([[1, 2]])}),
      ("physical decoding", {"decoded": torch.tensor([[3.0]])}),
      ("physical decoding", {"decoded": torch.tensor([[3.0, np.nan]])}),
      ("physical decoding", {"decoded": torch.tensor([[3, 4]])}),
      ("data-vector scattering",
       {"scattered": torch.tensor([[3.0, 4.0, np.inf]])}),
    )
    for stage, overrides in cases:
      with self.subTest(stage=stage, overrides=repr(overrides)):
        with self.assertRaisesRegex((TypeError, ValueError), stage):
          self._predict(_predictor(**overrides))

  def test_raw_tensor_and_returned_section_have_their_own_boundaries(self):
    predictor = _predictor()
    predictor._as_row_trusted = lambda values: torch.tensor([[1, 2]])
    with self.assertRaisesRegex(TypeError, "raw parameter row"):
      self._predict(predictor)

    for returned in (torch.tensor([3.0, 4.0]),
                     torch.tensor([3.0, 4.0, float("inf")])):
      with self.subTest(returned=repr(returned)):
        predictor = _predictor()
        predictor.dv_return = "section"
        predictor._section = lambda scattered, value=returned: value
        with self.assertRaisesRegex(
            (TypeError, ValueError), "returned data vector"):
          self._predict(predictor)

  def test_factored_model_shape_is_checked_before_decoding(self):
    predictor = _predictor()
    predictor._model_output_shape = (1, 3, 2)
    predictor.model = _Model(torch.ones((1, 3, 2)))
    result = self._predict(predictor)
    np.testing.assert_array_equal(result, np.array([3.0, 4.0, 0.0]))
    predictor.model = _Model(torch.ones((1, 2)))
    with self.assertRaisesRegex(ValueError, r"model evaluation.*shape"):
      self._predict(predictor)

  def test_every_family_returns_the_validated_decoded_shape(self):
    scalar = _predictor()
    scalar._scalar = True
    scalar.output_names = ["a", "b"]
    self.assertEqual(self._predict(scalar), {"a": 3.0, "b": 4.0})

    grid = _predictor()
    grid._grid = True
    grid.z = torch.tensor([0.0, 1.0])
    grid.quantity = "Hubble"
    np.testing.assert_array_equal(self._predict(grid)["Hubble"], [3.0, 4.0])

    grid2d = _predictor()
    grid2d._grid2d = True
    grid2d.z = torch.tensor([0.0, 1.0])
    grid2d.k = torch.tensor([0.1])
    grid2d.quantity = "pklin"
    self.assertEqual(self._predict(grid2d)["pklin"].shape, (2, 1))

    cmb = _predictor()
    cmb._cmb = True
    np.testing.assert_array_equal(self._predict(cmb), [3.0, 4.0])

  def test_cmb_factor_requires_positive_amplitude_and_finite_factor(self):
    chi2 = make_cmb_chi2(
      geom=_CmbGeometry(), law="as_exp2tau_ref",
      param_geometry=_CmbParameterGeometry(),
      as_name="As", tau_name="tau", as_ref=2.0, tau_ref=0.1)
    factor = chi2._factor(torch.tensor([[2.0, 0.1]]))
    torch.testing.assert_close(factor, torch.ones((1, 1)), rtol=0, atol=0)
    for values, message in (
        ([0.0, 0.1], "strictly positive"),
        ([-2.0, 0.1], "strictly positive"),
        ([float("nan"), 0.1], "NaN or infinity"),
        ([2.0, float("inf")], "NaN or infinity"),
        ([2.0, 1000.0], "amplitude factor"),
        ([2.0, -1000.0], "strictly positive")):
      with self.subTest(values=values):
        with self.assertRaisesRegex(ValueError, message):
          chi2._factor(torch.tensor([values]))

    chi2.param_geometry = types.SimpleNamespace(
      names=["As", "tau"],
      decode=lambda encoded: torch.tensor([[float("nan"), 0.1]]))
    with self.assertRaisesRegex(ValueError, "decoded physical parameters"):
      chi2._factor(torch.tensor([[2.0, 0.1]]))

  def test_cmb_scaling_checks_exact_shapes(self):
    chi2 = make_cmb_chi2(
      geom=_CmbGeometry(), law="as_exp2tau_ref",
      param_geometry=_CmbParameterGeometry(),
      as_name="As", tau_name="tau", as_ref=2.0, tau_ref=0.1)
    params = torch.tensor([[2.0, 0.1]])
    with self.assertRaisesRegex(ValueError, r"spectrum.*shape"):
      chi2.decode(torch.ones((1, 1)), params)
    with self.assertRaisesRegex(ValueError, r"encoded parameters.*shape"):
      chi2._factor(torch.ones((1, 1)))
    with self.assertRaisesRegex(ValueError, "physical spectrum before scaling"):
      chi2.encode(torch.tensor([[1.0, float("nan")]]), params)
    with self.assertRaisesRegex(ValueError, "physical spectrum after scaling"):
      chi2.encode(torch.full((1, 2), 3.0e38),
                  torch.tensor([[0.2, 0.1]]))
    with self.assertRaisesRegex(
        ValueError, "physical spectrum after inverse scaling"):
      chi2.decode(torch.full((1, 2), 3.0e38),
                  torch.tensor([[20.0, 0.1]]))

  def test_public_rescale_reader_accepts_only_native_none(self):
    self.assertEqual(
      results._read_public_rescale(
        _NativeAttrs({"rescale": "none"}), where="fixture.h5"),
      "none")
    with self.assertRaisesRegex(KeyError, r"rescale.*inverse transform"):
      results._read_public_rescale(_NativeAttrs({}), where="fixture.h5")
    invalid = (
      (_NativeAttrs({"rescale": "rescaled"}), "rescaled"),
      (_NativeAttrs({"rescale": "residual"}), "residual"),
      (_NativeAttrs({"rescale": "unknown"}), "unknown"),
      (_NativeAttrs({"rescale": True}), "True"),
      (_NativeAttrs({"rescale": b"none"}, dtype=np.dtype("S4")), "none"),
    )
    for attrs, label in invalid:
      with self.subTest(value=label):
        with self.assertRaisesRegex(ValueError, r"rescale.*inverse transform"):
          results._read_public_rescale(attrs, where="fixture.h5")

  def test_ignoring_a_saved_rescale_can_return_a_finite_wrong_vector(self):
    class _IdentityGeometry:
      center = torch.zeros(1)

      def unwhiten(self, value):
        return value

    transformed = RescaledChi2(_IdentityGeometry())
    transformed._R = lambda params: torch.tensor([[2.0]])
    network_value = torch.tensor([[56.472]])
    correct = transformed.decode(network_value, torch.zeros((1, 1)))
    plain = network_value
    self.assertTrue(torch.isfinite(correct).all())
    self.assertTrue(torch.isfinite(plain).all())
    self.assertAlmostEqual(
      float(torch.max(torch.abs(plain - correct))), 28.236, places=3)


class AdapterPublicationTests(unittest.TestCase):
  """Prove adapter arithmetic cannot leave a partial sampled-point state."""

  def test_every_adapter_propagates_the_shared_rescale_refusal(self):
    cases = (("emul_cosmic_shear.py", "emul_cosmic_shear", ["bad"]),
             ("emul_cmb.py", "emul_cmb", ["bad"]),
             ("emul_scalars.py", "emul_scalars", ["bad"]),
             ("emul_baosn.py", "emul_baosn", ["bad-h", "bad-dm"]),
             ("emul_mps.py", "emul_mps", ["bad-lin", "bad-boost"]))
    for index, (filename, class_name, roots) in enumerate(cases):
      with self.subTest(adapter=class_name):
        module = _load_adapter(filename, "rescale_adapter_" + str(index))
        adapter = getattr(module, class_name)()
        adapter.extra_args = {
          "emulators": [str(ROOT / root) for root in roots],
        }
        with mock.patch.object(
            module,
            "EmulatorPredictor",
            side_effect=ValueError(
              "rescale='rescaled' has no public inverse transform")) \
            as build_predictor:
          with self.assertRaisesRegex(ValueError, "rescale.*inverse transform"):
            adapter.initialize()
        build_predictor.assert_called_once()

  def test_scalar_adapter_stages_all_predictors_before_publication(self):
    module = _load_adapter("emul_scalars.py", "finite_scalar_adapter")

    class _Good:
      def predict(self, params):
        del params
        return {"first": 1.0}

    class _Bad:
      def predict(self, params):
        del params
        raise ValueError("second predictor refused")

    adapter = module.emul_scalars()
    adapter.predictors = [_Good(), _Bad()]
    state = {"derived": {}}
    with self.assertRaisesRegex(ValueError, "second predictor refused"):
      adapter.calculate(state, p0=0.1)
    self.assertEqual(state, {"derived": {}})

  def test_background_zero_hubble_refuses_before_cache(self):
    module = _load_adapter("emul_baosn.py", "finite_background_adapter")

    class _Hubble:
      def predict(self, params):
        del params
        return {"z": np.linspace(0.0, 1.0, 4), "Hubble": np.zeros(4)}

    class _Distance:
      def predict(self, params):
        del params
        raise AssertionError("distance predictor must not run")

    adapter = module.emul_baosn()
    adapter.p_h = _Hubble()
    adapter.p_dm = _Distance()
    state = {}
    with self.assertRaisesRegex(ValueError, "strictly positive"):
      adapter.calculate(state, p0=0.1)
    self.assertEqual(state, {})

  def test_background_derived_and_recombination_values_are_checked(self):
    module = _load_adapter("emul_baosn.py", "derived_background_adapter")

    class _Predictor:
      def __init__(self, quantity, value):
        self.quantity = quantity
        self.value = value

      def predict(self, params):
        del params
        return {"z": np.linspace(0.0, 1.0, 4),
                self.quantity: np.asarray(self.value)}

    class _Interpolation:
      def __init__(self, value):
        self.x = np.linspace(0.0, 1.0, 4)
        self.value = value

      def __call__(self, axis):
        return np.full(np.asarray(axis).shape, self.value)

    adapter = module.emul_baosn()
    adapter.p_h = _Predictor("Hubble", np.ones(4))
    adapter.p_dm = _Predictor("D_M", np.ones(4))
    bad_interpolators = {
      "H": _Interpolation(1.0), "chi": _Interpolation(float("inf")),
      "da": _Interpolation(1.0), "dl": _Interpolation(1.0),
      "z_max": 1.0,
    }
    state = {}
    with mock.patch.object(
        module, "distance_interpolators", return_value=bad_interpolators):
      with self.assertRaisesRegex(ValueError, "derived chi grid"):
        adapter.calculate(state, p0=0.1)
    self.assertEqual(state, {})

    for distance in (np.array([1.0, np.nan, 3.0, 4.0]), np.ones(3)):
      with self.subTest(distance=repr(distance)):
        adapter = module.emul_baosn()
        adapter.p_h = _Predictor("Hubble", np.full(4, 70.0))
        adapter.p_dm = _Predictor("D_M", distance)
        state = {}
        with self.assertRaisesRegex(ValueError, "D_M"):
          adapter.calculate(state, p0=0.1)
        self.assertEqual(state, {})

  def _mps_adapter(self, module, linear, boost, *, output_params=()):
    class _Predictor:
      def __init__(self, law, quantity, value):
        self.law = law
        self.quantity = quantity
        self.value = value

      def predict(self, params):
        del params
        return {self.quantity: np.asarray(self.value)}

    adapter = module.emul_mps()
    adapter._z = np.array([0.0, 1.0])
    adapter._k = np.array([0.1, 0.2])
    adapter.p_lin = _Predictor("none", "pklin", linear)
    adapter.p_boost = _Predictor("none", "boost", boost)
    adapter._dark_energy_law = "cosmological-constant"
    adapter._fixed_dark_energy = {}
    adapter._dark_energy_needed = True
    adapter.output_params = list(output_params)
    adapter.log = _Log()
    return adapter

  def test_mps_overflow_and_bad_sigma8_leave_no_partial_state(self):
    module = _load_adapter("emul_mps.py", "finite_mps_adapter")
    huge = np.full((2, 2), 1.0e308)
    adapter = self._mps_adapter(module, huge, huge)
    state = {}
    self.assertFalse(adapter.calculate(state, want_derived=False))
    self.assertEqual(state, {})

    adapter = self._mps_adapter(
      module, np.ones((2, 2)), np.ones((2, 2)),
      output_params=("sigma8",))
    adapter._compute_sigma8 = lambda *args, **kwargs: float("nan")
    state = {}
    self.assertFalse(adapter.calculate(
      state, want_derived=True, H0=67.0))
    self.assertEqual(state, {})


if __name__ == "__main__":
  unittest.main()
