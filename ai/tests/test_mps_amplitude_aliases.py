"""Amplitude-alias contract for the matter-power adapter.

A Cobaya params block may expose the primordial scalar amplitude as either
``As_1e9`` (the Syren convention, A_s in units of 1e-9) or as ``As`` (the
raw dimensionless value).  The two names are aliases for one physical number,
so the adapter must treat them as a single input and must refuse a point where
they disagree.
"""

import types
import unittest

import numpy as np

from ai.tests.test_mps_dark_energy_real_cobaya import (
  _load_adapter_without_neural_network_runtime,
)


class _LinearPredictor:
  """Stand-in pklin artifact that returns a finite law-space surface."""

  def __init__(self):
    self.law = "syren_linear"
    self.seen = []

  def predict(self, params):
    self.seen.append(dict(params))
    return {"pklin": np.ones((2, 2))}


class _BoostPredictor:
  """Stand-in boost artifact that returns a finite dimensionless surface."""

  def __init__(self):
    self.law = "none"
    self.seen = []

  def predict(self, params):
    self.seen.append(dict(params))
    return {"boost": np.ones((2, 2))}


class MPSAmplitudeAliasTests(unittest.TestCase):
  """Check that ``As_1e9`` and ``As`` are routed consistently or rejected."""

  @classmethod
  def setUpClass(cls):
    cls.module = _load_adapter_without_neural_network_runtime()

  def _routed_point(self):
    """Return a resolved input mapping that carries both amplitude spellings."""
    return {
      "w": -0.9,
      "w0": -0.9,
      "wa": 0.2,
      "w0pwa": -0.7,
      "As_1e9": 2.1,
      "As": 2.1e-9,
      "ns": 0.965,
      "H0": 67.0,
      "omegab": 0.05,
      "omegam": 0.3,
    }

  def test_adapter_routed_point_accepts_consistent_amplitudes(self):
    """A point with matching ``As_1e9`` and ``As`` resolves to ``As_1e9``."""
    routed = self._routed_point()
    resolved_params = self.module._resolved_dark_energy_point(
      routed,
      law="w0wa-cpl",
      fixed={},
      needed=True,
    )
    as_1e9 = self.module.syren_base.syren_params_from(
      resolved_params, dark_energy_law="w0wa-cpl")[0]
    self.assertEqual(as_1e9, 2.1)

  def test_adapter_calculate_refuses_conflicting_amplitudes(self):
    """Conflicting ``As_1e9`` and ``As`` stop before any spectrum is stored."""
    EmulMPS = self.module.emul_mps
    adapter = object.__new__(EmulMPS)
    adapter._z = np.asarray([0.0, 1.0])
    adapter._k = np.asarray([0.1, 0.2])
    adapter._dark_energy_law = "w0wa-cpl"
    adapter._fixed_dark_energy = {}
    adapter._dark_energy_needed = True
    adapter._sigma8_requested = False
    adapter.output_params = []
    adapter.log = types.SimpleNamespace(
      debug=lambda *args, **kwargs: None)
    adapter.p_lin = _LinearPredictor()
    adapter.p_boost = _BoostPredictor()

    conflicting_params = dict(self._routed_point())
    conflicting_params["As"] = 9e-9
    state = {}
    with self.assertRaisesRegex(
        ValueError, "conflicting primordial amplitudes"):
      adapter.calculate(
        state, want_derived=False, **conflicting_params)
    self.assertNotIn("Pk_grid", state)


if __name__ == "__main__":
  unittest.main()
