"""Matter-power adapter checks, run in a child process by the contract test.

Sigma-eight measures the linear matter fluctuations inside a spherical
top-hat of radius 8 Mpc/h.  The numeric checks here use a power spectrum
with an analytic answer, so they can distinguish the required 8/h-Mpc
radius from the old literal 8-Mpc radius without using the production
calculation as its own reference.  Separate cases prove that a nearby
redshift or an incomplete wavenumber range cannot return a plausible
number.  A second group checks the adapter's early request contract:
``must_provide`` refuses an unsupported density pair, redshift, or
wavenumber request while Cobaya setup can still stop cleanly.

This file is not collected by the test discovery pattern.  The public
entry is ai/tests/test_mps_sigma8_contract.py, which launches this file
in a child process whose PYTHONPATH names ai/tests/cobaya_minimal_stub
ahead of any installed package.  The adapter's cobaya imports therefore
resolve to that on-disk stand-in, selected before Python started:
nothing here edits any process's import table.  The parent also launches
one negative control (the MPS_SIGMA8_EXPECTED_OVERRIDE variable below)
that must make the known-answer test fail, proving a wrong sigma-eight
still reaches the parent as a nonzero exit code.
"""

import os
import types
import unittest

import numpy as np

import cobaya

if not getattr(cobaya, "COBAYA_MINIMAL_STUB", False):
  raise SystemExit(
    "mps_sigma8_child_checks: the imported cobaya package is not the "
    "ai/tests/cobaya_minimal_stub stand-in. Run these checks through "
    "ai/tests/test_mps_sigma8_contract.py, which launches this file with "
    "the stand-in first on PYTHONPATH.")

from cobaya_theory import emul_mps as adapter_module


# The independently integrated sigma-eight of the finite analytic grid
# below (h = 0.64, 4001 points over x in [1e-4, 400]).
EXPECTED_FINITE_GRID = 0.6399980037465730


def _expected_known_answer():
  """The finite-grid known answer, unless the launcher overrides it.

  The parent contract test launches this file once with
  MPS_SIGMA8_EXPECTED_OVERRIDE set to a wrong value and requires that
  run to FAIL.  A harness that swallowed child failures, or a
  known-answer assertion that stopped biting, would pass that control
  and be caught.

  Returns:
    the expected sigma-eight for the known-answer test as a float.
  """
  text = os.environ.get("MPS_SIGMA8_EXPECTED_OVERRIDE")
  if text is None:
    return EXPECTED_FINITE_GRID
  return float(text)


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


class SigmaRecordingAdapter(adapter_module.emul_mps):
  """An adapter whose sigma-eight helper records its inputs.

  The subclass is defined before any instance exists, so no live method
  is ever replaced: ``calculate`` runs unchanged and hands its power
  surface to this recording helper instead of the real integral.  The
  recorded arguments land in the instance's ``seen`` mapping.
  """

  def __init__(self):
    self.seen = {}

  def _compute_sigma8(self, Pk_2d, k_array, z_array, *, h):
    """Record the exact arguments calculate passed; return a marker."""
    self.seen["power"] = np.array(Pk_2d, copy=True)
    self.seen["shape"] = Pk_2d.shape
    self.seen["h"] = h
    self.seen["k"] = k_array
    self.seen["z"] = z_array
    return 0.8


class SigmaEightContractTests(unittest.TestCase):
  """Check units, exact redshift, input validity, and k completeness."""

  def test_analytic_known_answer_uses_eight_over_h_mpc(self):
    """A checked analytic spectrum returns h, not the old value one."""
    expected_finite_grid = _expected_known_answer()
    for dtype in (np.float64, np.float32):
      with self.subTest(dtype=dtype.__name__):
        h, k, z, surface = _analytic_surface(dtype=dtype)
        got = adapter_module.emul_mps()._compute_sigma8(
          surface, k, z, h=h)
        self.assertAlmostEqual(got, expected_finite_grid, delta=2.0e-8)

    # A wider grid supports both radii.  Passing h=1 deliberately asks for an
    # 8-Mpc radius and returns approximately one; ignoring the supplied h
    # would therefore make the h=0.64 assertion above fail by about 0.36.
    h, k, z, surface = _analytic_surface(
      points=8001, x_min=1.0e-5, x_max=1000.0)
    conventional = adapter_module.emul_mps()._compute_sigma8(
      surface, k, z, h=h)
    literal_eight_mpc = adapter_module.emul_mps()._compute_sigma8(
      surface, k, z, h=1.0)
    self.assertLess(abs(conventional - h), 2.0e-6)
    self.assertGreater(literal_eight_mpc - conventional, 0.3)

  def test_sigma8_requires_an_exact_zero_redshift_row(self):
    """A z=0.009 row is never relabelled or interpolated as z=0."""
    h, k, z, surface = _analytic_surface()
    got = adapter_module.emul_mps()._compute_sigma8(surface, k, z, h=h)
    self.assertAlmostEqual(got, EXPECTED_FINITE_GRID, delta=2.0e-8)

    for missing_zero in (
        np.array([0.009, 0.1, 0.5, 1.0]),
        np.array([0.011, 0.1, 0.5, 1.0])):
      with self.subTest(z=missing_zero[0]):
        with self.assertRaisesRegex(ValueError, "exact z=0"):
          adapter_module.emul_mps()._compute_sigma8(
            surface, k, missing_zero, h=h)

  def test_truncated_k_grids_refuse(self):
    """A k range that cuts the integrand stops instead of returning a bias."""
    cases = (
      ("low-only", 1.0e-6, 0.1, 4001),
      ("high-only", 10.0, 1.0e6, 4001),
      ("narrow-window", 12.5, 125.0, 4001),
    )
    for label, x_min, x_max, points in cases:
      with self.subTest(case=label):
        h, k, z, surface = _analytic_surface(
          points=points, x_min=x_min, x_max=x_max)
        with self.assertRaisesRegex(ValueError, "truncates the sigma8"):
          adapter_module.emul_mps()._compute_sigma8(
            surface, k, z, h=h)

  def _law_free_adapter(self, adapter_class):
    """Build the public calculate path without reading artifact files.

    Arguments:
      adapter_class = the adapter class to instantiate (the production
                      class, or the recording subclass above).

    Returns:
      (h, adapter): the surface's h and the hand-assembled adapter.
    """
    h, k, z, surface = _analytic_surface(points=2001)
    adapter = adapter_class()
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
    h, adapter = self._law_free_adapter(SigmaRecordingAdapter)
    state = {}
    self.assertTrue(adapter.calculate(
      state, want_derived=True, H0=100.0 * h))
    self.assertEqual(adapter.seen["shape"], adapter.p_lin.value.shape)
    np.testing.assert_array_equal(adapter.seen["power"],
                                  adapter.p_lin.value)
    self.assertAlmostEqual(adapter.seen["h"], h)
    self.assertEqual(state["derived"], {"sigma8": 0.8})

    for params, message in (({}, "requires H0"),
                            ({"H0": 0.0}, "finite positive")):
      with self.subTest(params=params):
        state = {}
        with self.assertRaisesRegex(ValueError, message):
          adapter.calculate(state, want_derived=True, **params)
        self.assertEqual(state, {})

  def test_cobaya_declares_sigma8_as_output_and_conditionally_needs_h0(self):
    """Cobaya sees a derived result, not three invented input parameters."""
    adapter = adapter_module.emul_mps()
    self.assertEqual(adapter.get_can_support_params(), [])
    self.assertEqual(adapter.get_can_provide_params(), ["sigma8"])
    self.assertEqual(adapter.must_provide(sigma8=None), {"H0": None})
    self.assertIsNone(adapter.must_provide(Pk_grid={}))


class MatterPowerRequestContractTests(unittest.TestCase):
  """must_provide refuses unservable requests while setup can stop.

  Cobaya forwards each likelihood's matter-power requirement options
  before sampling starts.  The adapter checks them at that moment, so a
  wrong density pair, an out-of-range redshift, or a wavenumber the
  stored grid cannot serve stops with a startup error the user can fix
  in the YAML, instead of failing deep inside a running chain.
  """

  def _adapter(self, *, allow_extrapolation=True):
    """Return a bare adapter carrying one stored (z, k) grid."""
    adapter = adapter_module.emul_mps()
    adapter._z = np.array([0.0, 0.5, 1.0, 2.0])
    adapter._k = np.geomspace(1.0e-4, 50.0, 16)
    adapter._allow_k_extrapolation = allow_extrapolation
    adapter._sigma8_requested = False
    adapter.log = types.SimpleNamespace(debug=lambda *args, **kwargs: None)
    return adapter

  def test_supported_requests_pass_and_sigma8_still_adds_h0(self):
    """In-range products pass; a sigma8 request still adds only H0."""
    adapter = self._adapter()
    supported = {
      "z": [0.0, 1.0, 2.0],
      "k_max": 10.0,
      "nonlinear": (True, False),
      "vars_pairs": [("delta_tot", "delta_tot")],
    }
    self.assertIsNone(adapter.must_provide(
      Pk_grid=dict(supported), Pk_interpolator=dict(supported)))
    self.assertEqual(
      adapter.must_provide(Pk_grid=dict(supported), sigma8=None),
      {"H0": None})

  def test_unsupported_density_pair_is_refused(self):
    """A pair other than delta_tot stops at setup, naming the rule."""
    adapter = self._adapter()
    with self.assertRaisesRegex(adapter_module.LoggedError, "delta_tot"):
      adapter.must_provide(
        Pk_grid={"vars_pairs": [("delta_nonu", "delta_nonu")]})

  def test_redshift_outside_the_stored_range_is_refused(self):
    """z is never extrapolated, above or below the stored grid."""
    adapter = self._adapter()
    for bad_z in ([0.0, 5.0], [-0.5, 1.0]):
      with self.subTest(z=bad_z):
        with self.assertRaisesRegex(adapter_module.LoggedError,
                                    "stored redshift range"):
          adapter.must_provide(Pk_interpolator={"z": bad_z})

  def test_grid_k_max_above_the_stored_edge_is_refused(self):
    """The fixed stored grid cannot extend for a raw-grid consumer."""
    adapter = self._adapter()
    with self.assertRaisesRegex(adapter_module.LoggedError,
                                "cannot extend"):
      adapter.must_provide(Pk_grid={"k_max": 100.0})

  def test_interpolator_k_max_follows_the_extrapolation_choice(self):
    """Power-law tails serve a wider k_max only when they are enabled."""
    tail_request = {"k_max": 100.0}
    self.assertIsNone(
      self._adapter(allow_extrapolation=True).must_provide(
        Pk_interpolator=dict(tail_request)))
    with self.assertRaisesRegex(adapter_module.LoggedError,
                                "allow_k_extrapolation"):
      self._adapter(allow_extrapolation=False).must_provide(
        Pk_interpolator=dict(tail_request))

  def test_malformed_options_and_nonlinear_values_are_refused(self):
    """Non-mapping options and non-boolean nonlinear choices stop."""
    adapter = self._adapter()
    with self.assertRaisesRegex(adapter_module.LoggedError, "mapping"):
      adapter.must_provide(Pk_grid=[("delta_tot", "delta_tot")])
    with self.assertRaisesRegex(adapter_module.LoggedError, "boolean"):
      adapter.must_provide(Pk_grid={"nonlinear": "yes"})
    self.assertIsNone(adapter.must_provide(Pk_grid=None))


if __name__ == "__main__":
  unittest.main()
