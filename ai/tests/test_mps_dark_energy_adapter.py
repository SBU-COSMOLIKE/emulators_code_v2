"""CPU checks for dark-energy coordinates in the matter-power adapter.

The shipped EMUL2 example samples ``w0pwa = w0 + wa`` but marks that
sampling coordinate as dropped.  Cobaya can still calculate ``wa`` from it;
the Theory adapter must request the calculated physical coordinates and then
reconstruct the names stored by the emulator.  These tests prove that the
nonzero value reaches both learned predictors and the Syren base before any
matter-power surface is assembled.
"""

import types
import unittest
from unittest import mock

import numpy as np

from ai.tests.test_mps_sigma8_contract import _load_mps_adapter


class _Artifact:
  """Small saved-artifact stand-in for requirement and calculation checks."""

  def __init__(self, *, law, quantity, output, fixed_facts=None):
    self.law = law
    self.quantity = quantity
    self.output = np.asarray(output)
    self.fixed_facts = fixed_facts
    self.seen = []

  def predict(self, params):
    self.seen.append(dict(params))
    return {self.quantity: np.array(self.output, copy=True)}


def _facts(law, inputs, *, w="n/a", wa="n/a"):
  """Return the dark-energy part of one schema-3 scientific record."""
  return {
    "dark_energy_law": law,
    "dark_energy_inputs": list(inputs),
    "cosmology_fixed": {"w": w, "wa": wa},
  }


class MPSDarkEnergyAdapterTests(unittest.TestCase):
  """Check startup requirements, coordinate reconstruction, and ordering."""

  @classmethod
  def setUpClass(cls):
    cls.module = _load_mps_adapter()

  def test_dropped_sum_is_replaced_by_physical_requirements(self):
    """Cobaya is asked for w and calculated wa, never dropped w0pwa."""
    artifact = types.SimpleNamespace(
      fixed_facts=_facts("w0wa-cpl", ["w", "wa"]))
    requirements = {"w": None, "w0pwa": None, "As": None}

    law, fixed, needed = self.module._dark_energy_contract(
      artifact, {"w", "w0pwa"}, requirements, need_base=True)

    self.assertEqual(law, "w0wa-cpl")
    self.assertEqual(fixed, {})
    self.assertTrue(needed)
    self.assertEqual(requirements, {"w": None, "wa": None, "As": None})

  def test_old_transformed_artifact_with_constant_w_record_refuses(self):
    """A record made by the silent-wa bug cannot be served as constant w."""
    artifact = types.SimpleNamespace(
      fixed_facts=_facts("constant-w", ["w"]))
    with self.assertRaisesRegex(
        ValueError, "sampled w0pwa.*constant-w.*Re-generate"):
      self.module._dark_energy_contract(
        artifact, {"w", "w0pwa"}, {"w": None, "w0pwa": None},
        need_base=True)

  def test_reconstruction_supplies_every_saved_coordinate_name(self):
    """A nonzero-wa point recreates w, w0, wa, and their saved sum."""
    point = self.module._resolved_dark_energy_point(
      {"w": -0.9, "wa": 0.2}, law="w0wa-cpl", fixed={}, needed=True)
    self.assertEqual(point["w"], -0.9)
    self.assertEqual(point["w0"], -0.9)
    self.assertEqual(point["wa"], 0.2)
    self.assertAlmostEqual(point["w0pwa"], -0.7)

  def _adapter(self):
    """Build the production calculate path around two observable spies."""
    adapter = self.module.emul_mps()
    adapter._z = np.array([0.0, 1.0])
    adapter._k = np.array([0.1, 0.2])
    adapter.p_lin = _Artifact(
      law="syren_linear", quantity="pklin", output=np.zeros((2, 2)))
    adapter.p_boost = _Artifact(
      law="none", quantity="boost", output=np.ones((2, 2)))
    adapter._dark_energy_law = "w0wa-cpl"
    adapter._fixed_dark_energy = {}
    adapter._dark_energy_needed = True
    adapter.output_params = []
    adapter.log = types.SimpleNamespace(debug=lambda *args, **kwargs: None)
    return adapter

  def test_nonzero_wa_reaches_predictors_and_syren_base(self):
    """The adapter resolves coordinates before either learned or Syren work."""
    adapter = self._adapter()
    base_calls = []

    def base_spy(*, k_mpc, z, As_1e9, ns, H0, Ob, Om, w0, wa):
      del As_1e9, ns, H0, Ob, Om
      base_calls.append((w0, wa))
      return np.ones((len(z), len(k_mpc)))

    params = {
      "As_1e9": 2.1,
      "ns": 0.965,
      "H0": 67.0,
      "omegab": 0.049,
      "omegam": 0.31,
      "w": -0.9,
      "wa": 0.2,
    }
    state = {}
    with mock.patch.object(self.module.syren_base, "base_pklin", base_spy):
      self.assertTrue(adapter.calculate(state, want_derived=False, **params))

    self.assertEqual(base_calls, [(-0.9, 0.2)])
    for seen in (adapter.p_lin.seen[-1], adapter.p_boost.seen[-1]):
      self.assertEqual(seen["w"], -0.9)
      self.assertEqual(seen["w0"], -0.9)
      self.assertEqual(seen["wa"], 0.2)
      self.assertAlmostEqual(seen["w0pwa"], -0.7)
    np.testing.assert_array_equal(
      state[("Pk_grid", False, "delta_tot", "delta_tot")][2],
      np.ones((2, 2)))

  def test_inconsistent_sum_refuses_before_any_prediction(self):
    """Conflicting redundant input stops before either model can run."""
    adapter = self._adapter()
    params = {
      "As_1e9": 2.1,
      "ns": 0.965,
      "H0": 67.0,
      "omegab": 0.049,
      "omegam": 0.31,
      "w": -0.9,
      "wa": 0.2,
      "w0pwa": -0.9,
    }
    with self.assertRaisesRegex(ValueError, r"w0pwa.*w0 \+ wa"):
      adapter.calculate({}, want_derived=False, **params)
    self.assertEqual(adapter.p_lin.seen, [])
    self.assertEqual(adapter.p_boost.seen, [])


if __name__ == "__main__":
  unittest.main()
