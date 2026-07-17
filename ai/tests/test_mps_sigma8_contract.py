"""CPU checks for the conventional sigma-eight calculation.

Sigma-eight measures the linear matter fluctuations inside a spherical
top-hat of radius 8 Mpc/h.  These tests use a power spectrum with an analytic
answer, so they can distinguish the required 8/h-Mpc radius from the old
literal 8-Mpc radius without using the production calculation as its own
reference.  Separate cases prove that a nearby redshift, an incomplete
wavenumber range, or a poorly sampled grid cannot return a plausible number.
"""

import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import scipy.interpolate  # Load SciPy before the temporary Cobaya stand-ins.
import torch  # Load PyTorch before the temporary Cobaya stand-ins.


ROOT = Path(__file__).resolve().parents[2]


def _load_mps_adapter():
  """Load the adapter with the small part of Cobaya used by these tests."""
  cobaya = types.ModuleType("cobaya")
  theory_module = types.ModuleType("cobaya.theory")
  log_module = types.ModuleType("cobaya.log")

  class _Theory:
    extra_args = {}
    output_params = []

    def initialize(self):
      return None

    def must_provide(self, **requirements):
      del requirements
      return None

  class _LoggedError(Exception):
    pass

  class _Logger:
    def debug(self, *args, **kwargs):
      del args, kwargs

  theory_module.Theory = _Theory
  log_module.LoggedError = _LoggedError
  log_module.get_logger = lambda name: _Logger()
  path = ROOT / "cobaya_theory" / "emul_mps.py"
  spec = importlib.util.spec_from_file_location("sigma8_adapter", path)
  module = importlib.util.module_from_spec(spec)
  stand_ins = {
    "cobaya": cobaya,
    "cobaya.theory": theory_module,
    "cobaya.log": log_module,
  }
  with mock.patch.dict(sys.modules, stand_ins):
    spec.loader.exec_module(module)
  return module


def _analytic_surface(*, points=4001, x_min=1.0e-4, x_max=400.0,
                      h=0.64, dtype=np.float64):
  """Return a z=0 surface whose infinite-domain sigma-eight equals h.

  For ``P(k) = C/k`` with ``C = 512 pi^2 / 9``, direct integration gives
  ``sigma_R = 8/R``.  Conventional ``R = 8/h`` therefore gives
  ``sigma8 = h``.  The finite x range used by the main known-answer test has
  the independently integrated result 0.6399980037465730.
  """
  radius = 8.0 / h
  k = np.geomspace(x_min / radius, x_max / radius, points).astype(dtype)
  constant = 512.0 * np.pi ** 2 / 9.0
  p_at_zero = (constant / k.astype(np.float64)).astype(dtype)
  z = np.array([0.0, 0.009, 0.5, 1.0], dtype=dtype)
  surface = np.stack((p_at_zero, 4.0 * p_at_zero,
                      0.5 * p_at_zero, 0.25 * p_at_zero))
  return h, k, z, surface


class _Predictor:
  """Return one fixed law-free matter-power surface."""

  law = "none"

  def __init__(self, quantity, value):
    self.quantity = quantity
    self.value = value

  def predict(self, params):
    del params
    return {self.quantity: self.value}


class SigmaEightContractTests(unittest.TestCase):
  """Check units, exact redshift, input validity, and k completeness."""

  @classmethod
  def setUpClass(cls):
    cls.module = _load_mps_adapter()

  def test_analytic_known_answer_uses_eight_over_h_mpc(self):
    """A checked analytic spectrum returns h, not the old value one."""
    expected_finite_grid = 0.6399980037465730
    for dtype in (np.float64, np.float32):
      with self.subTest(dtype=dtype.__name__):
        h, k, z, surface = _analytic_surface(dtype=dtype)
        got = self.module.emul_mps()._compute_sigma8(
          surface, k, z, h=h)
        self.assertAlmostEqual(got, expected_finite_grid, delta=2.0e-8)

    # A wider grid supports both radii.  Passing h=1 deliberately asks for an
    # 8-Mpc radius and returns approximately one; ignoring the supplied h
    # would therefore make the h=0.64 assertion above fail by about 0.36.
    h, k, z, surface = _analytic_surface(
      points=8001, x_min=1.0e-5, x_max=1000.0)
    conventional = self.module.emul_mps()._compute_sigma8(
      surface, k, z, h=h)
    literal_eight_mpc = self.module.emul_mps()._compute_sigma8(
      surface, k, z, h=1.0)
    self.assertLess(abs(conventional - h), 2.0e-6)
    self.assertGreater(literal_eight_mpc - conventional, 0.3)

  def test_sigma8_requires_an_exact_zero_redshift_row(self):
    """A z=0.009 row is never relabelled or interpolated as z=0."""
    h, k, z, surface = _analytic_surface()
    got = self.module.emul_mps()._compute_sigma8(surface, k, z, h=h)
    self.assertAlmostEqual(got, 0.6399980037465730, delta=2.0e-8)

    for missing_zero in (
        np.array([0.009, 0.1, 0.5, 1.0]),
        np.array([0.011, 0.1, 0.5, 1.0])):
      with self.subTest(z=missing_zero[0]):
        with self.assertRaisesRegex(ValueError, "exact z=0"):
          self.module.emul_mps()._compute_sigma8(
            surface, k, missing_zero, h=h)

  def test_incomplete_or_under_resolved_k_grids_refuse(self):
    """Short tails and sparse wide grids stop instead of returning a bias."""
    cases = (
      ("low-only", 1.0e-6, 0.1, 4001, "incomplete high-k tail"),
      ("high-only", 10.0, 1.0e6, 4001, "incomplete low-k tail"),
      ("one-to-ten", 12.5, 125.0, 4001, "more than one decade"),
      ("wide-eight-points", 1.0e-4, 400.0, 8, "too coarse"),
    )
    for label, x_min, x_max, points, message in cases:
      with self.subTest(case=label):
        h, k, z, surface = _analytic_surface(
          points=points, x_min=x_min, x_max=x_max)
        with self.assertRaisesRegex(ValueError, message):
          self.module.emul_mps()._compute_sigma8(
            surface, k, z, h=h)

  def test_each_completeness_limit_has_an_independent_boundary_case(self):
    """Tail, coarsening, and panel limits each stop a distinct near miss."""
    def check(log_k, contribution):
      variance = float(np.trapz(contribution, log_k))
      self.module.emul_mps._check_sigma8_completeness(
        log_k, contribution, variance)

    # The first tail has an estimated missing fraction about 7.9e-6 and
    # passes.  Shortening both ends gives about 1.3e-5 and must fail the 1e-5
    # limit while its coarsening and largest-panel checks still pass.
    log_k = np.linspace(-11.75, 11.75, 4001)
    check(log_k, np.exp(-np.abs(log_k)))
    log_k = np.linspace(-11.25, 11.25, 4001)
    with self.assertRaisesRegex(ValueError, "estimated variance fraction"):
      check(log_k, np.exp(-np.abs(log_k)))

    # Alternating adjacent values isolate the interlaced recalculation.  A
    # relative modulation of 8e-4 passes; 1.2e-3 must fail the 1e-3 limit.
    log_k = np.linspace(-5.0, 5.0, 4001)
    signs = np.where(np.arange(log_k.size) % 2 == 0, 1.0, -1.0)
    envelope = np.exp(-0.5 * log_k ** 2)
    check(log_k, envelope * (1.0 + 8.0e-4 * signs))
    with self.assertRaisesRegex(ValueError, "interlaced every-other-point"):
      check(log_k, envelope * (1.0 + 1.2e-3 * signs))

    # A single wider central gap isolates the local panel limit.  A gap of
    # .36 makes the largest panel carry about 9% and passes; .44 makes it
    # carry about 11% and must fail the 10% limit.
    def panel_grid(gap):
      log_k = np.concatenate((
        np.linspace(-13.0, -gap / 2.0, 2001),
        np.linspace(gap / 2.0, 13.0, 2001)))
      contribution = np.exp(-np.maximum(np.abs(log_k) - 1.0, 0.0))
      return log_k, contribution

    log_k, contribution = panel_grid(0.36)
    check(log_k, contribution)
    log_k, contribution = panel_grid(0.44)
    with self.assertRaisesRegex(ValueError, "one trapezoid"):
      check(log_k, contribution)

  def test_invalid_h_axes_and_surface_refuse_before_integration(self):
    """Every direct helper input has a named finite physical boundary."""
    h, k, z, surface = _analytic_surface()
    for bad_h in (True, "0.64", 0.0, -0.64, np.nan, np.inf):
      with self.subTest(h=repr(bad_h)):
        with self.assertRaisesRegex(ValueError, "h=H0/100"):
          self.module.emul_mps()._compute_sigma8(
            surface, k, z, h=bad_h)

    bad_inputs = (
      ("k axis", surface, np.array([0.2, 0.1]), z, "k axis"),
      ("z axis", surface, k, np.array([0.0, 0.5, 0.4, 1.0]), "z axis"),
      ("shape", surface[:, :-1], k, z, "exact shape"),
      ("nonfinite surface", np.where(
        np.indices(surface.shape)[1] == 0, np.nan, surface),
       k, z, "finite, strictly positive"),
      ("zero surface", np.where(
        np.indices(surface.shape)[1] == 0, 0.0, surface),
       k, z, "finite, strictly positive"),
    )
    for label, bad_surface, bad_k, bad_z, message in bad_inputs:
      with self.subTest(case=label):
        with self.assertRaisesRegex(ValueError, message):
          self.module.emul_mps()._compute_sigma8(
            bad_surface, bad_k, bad_z, h=h)

  def _law_free_adapter(self):
    """Build the public calculate path without reading artifact files."""
    h, k, z, surface = _analytic_surface(points=2001)
    adapter = self.module.emul_mps()
    adapter._k = k
    adapter._z = z
    adapter.p_lin = _Predictor("pklin", surface)
    adapter.p_boost = _Predictor("boost", 3.0 * np.ones_like(surface))
    adapter._dark_energy_law = "cosmological-constant"
    adapter._fixed_dark_energy = {}
    adapter._dark_energy_needed = True
    adapter.output_params = ["sigma8"]
    adapter.log = types.SimpleNamespace(debug=lambda *args, **kwargs: None)
    return h, adapter

  def test_calculate_passes_h0_to_law_free_sigma8_atomically(self):
    """Even law-free artifacts receive H0 and publish only after sigma8."""
    h, adapter = self._law_free_adapter()
    seen = {}

    def sigma8_spy(power, k, z, *, h):
      seen["power"] = np.array(power, copy=True)
      seen["shape"] = power.shape
      seen["h"] = h
      seen["k"] = k
      seen["z"] = z
      return 0.8

    adapter._compute_sigma8 = sigma8_spy
    state = {}
    self.assertTrue(adapter.calculate(
      state, want_derived=True, H0=100.0 * h))
    self.assertEqual(seen["shape"], adapter.p_lin.value.shape)
    np.testing.assert_array_equal(seen["power"], adapter.p_lin.value)
    self.assertAlmostEqual(seen["h"], h)
    self.assertEqual(state["derived"], {"sigma8": 0.8})

    for params, message in (({}, "requires H0"),
                            ({"H0": 0.0}, "strictly positive")):
      with self.subTest(params=params):
        state = {}
        with self.assertRaisesRegex(ValueError, message):
          adapter.calculate(state, want_derived=True, **params)
        self.assertEqual(state, {})

  def test_cobaya_declares_sigma8_as_output_and_conditionally_needs_h0(self):
    """Cobaya sees a derived result, not three invented input parameters."""
    adapter = self.module.emul_mps()
    self.assertEqual(adapter.get_can_support_params(), [])
    self.assertEqual(adapter.get_can_provide_params(), ["sigma8"])
    self.assertEqual(adapter.must_provide(sigma8=None), {"H0": None})
    self.assertIsNone(adapter.must_provide(Pk_grid={}))

  def test_real_cobaya_routes_h0_only_when_sigma8_is_requested(self):
    """The live dependency resolver preserves P(k) and both sigma8 routes."""
    try:
      from cobaya.likelihood import Likelihood
      from cobaya.log import LoggedError
      from cobaya.model import get_model
      from cobaya.theory import Theory
      from cobaya_theory.emul_mps import emul_mps as LiveEmulMPS
    except ImportError as error:
      self.skipTest("real Cobaya is unavailable: " + str(error))

    class RoutingMPS(LiveEmulMPS):
      """Exercise inherited routing without loading saved artifacts."""

      seen = []

      def initialize(self):
        self._req = {"x": None}
        self._sigma8_requested = False

      def initialize_with_provider(self, provider):
        Theory.initialize_with_provider(self, provider)

      def calculate(self, state, want_derived=True, **params):
        type(self).seen.append(dict(params))
        k = np.geomspace(1.0e-3, 1.0, 4)
        z = np.arange(4.0)
        linear = np.ones((4, 4))
        state[("Pk_grid", False, "delta_tot", "delta_tot")] = (
          k, z, linear)
        state[("Pk_grid", True, "delta_tot", "delta_tot")] = (
          k, z, 2.0 * linear)
        if want_derived and (
            "sigma8" in self.output_params or self._sigma8_requested):
          if "H0" in params:
            h0 = params["H0"]
          else:
            h0 = self.provider.get_param("H0")
          state["derived"]["sigma8"] = h0 / 100.0
        return True

    class PkLike(Likelihood):
      """Read the two matter-power products but never request sigma8."""

      def get_requirements(self):
        return {"Pk_grid": None, "Pk_interpolator": None}

      def calculate(self, state, want_derived=True, **params):
        del want_derived, params
        k, z, power = self.provider.get_Pk_grid(nonlinear=False)
        interpolator = self.provider.get_Pk_interpolator(nonlinear=False)
        assert power.shape == (z.size, k.size)
        assert np.isfinite(interpolator.P(1.0, 0.1))
        state["logp"] = 0.0

    class SigmaLike(Likelihood):
      """Request sigma8 through its public getter rather than a YAML output."""

      def get_requirements(self):
        return {"sigma8": None}

      def calculate(self, state, want_derived=True, **params):
        del want_derived, params
        assert self.provider.get_sigma8() == 0.67
        state["logp"] = 0.0

    def model_for(params, likelihood):
      return get_model({
        "params": params,
        "theory": {"mps": RoutingMPS},
        "likelihood": {"like": likelihood},
      })

    RoutingMPS.seen = []
    pk_model = model_for({"x": 1}, PkLike)
    pk_theory = pk_model.theory["mps"]
    self.assertEqual(pk_theory.input_params, ["x"])
    self.assertEqual(pk_theory.output_params, [])
    pk_model.logposterior({})
    self.assertEqual(RoutingMPS.seen[-1], {"x": 1.0})
    self.assertTrue(
      {"Pk_grid", "Pk_interpolator", "sigma8"}
      <= set(pk_theory.get_can_provide_methods()))

    RoutingMPS.seen = []
    derived_model = model_for(
      {"x": 1, "H0": 67, "sigma8": {"derived": True}}, PkLike)
    derived_theory = derived_model.theory["mps"]
    self.assertEqual(derived_theory.input_params, ["x"])
    self.assertEqual(derived_theory.output_params, ["sigma8"])
    result = derived_model.logposterior({})
    self.assertEqual(RoutingMPS.seen[-1], {"x": 1.0})
    self.assertEqual(result.derived, [0.67])

    RoutingMPS.seen = []
    method_model = model_for({"x": 1, "H0": 67}, SigmaLike)
    method_theory = method_model.theory["mps"]
    self.assertEqual(method_theory.output_params, [])
    self.assertTrue(method_theory._sigma8_requested)
    method_model.logposterior({})
    self.assertEqual(RoutingMPS.seen[-1], {"x": 1.0})

    with self.assertRaisesRegex(LoggedError, "Requirement H0"):
      model_for({"x": 1}, SigmaLike)


if __name__ == "__main__":
  unittest.main()
