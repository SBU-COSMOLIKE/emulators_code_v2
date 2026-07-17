"""Real-Cobaya check for the transformed matter-power coordinates.

The shipped EMUL2 setup samples two numbers: the present-day value ``w`` and
the sum ``w0pwa = w + wa``.  It marks the sum as dropped and asks Cobaya to
calculate ``wa``.  A Theory component must therefore receive ``w`` and the
calculated, possibly nonzero ``wa``; it must not ask Cobaya for the dropped
sampling coordinate.

This file uses the installed Cobaya parameter engine rather than the small
stand-ins used by the fast adapter tests.  The light Cobaya-only environment
does not contain PyTorch, so two imports that are irrelevant to parameter
routing are replaced while this test module loads: the saved neural-network
reader and the device-option helpers.  The production matter-power adapter,
its dark-energy reconstruction helper, and the shared Syren coordinate
resolver remain the real repository code.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load_adapter_without_neural_network_runtime():
  """Load the production adapter with real Cobaya but no PyTorch dependency.

  Artifact loading and device selection happen only during the production
  ``initialize`` method.  The focused Theory subclass below replaces that
  method because this test is about Cobaya's parameter routing.  Small import
  placeholders therefore keep a Cobaya-only environment honest without
  pretending to load or evaluate a saved emulator.
  """
  inference = types.ModuleType("emulator.inference")
  inference.EmulatorPredictor = object
  inference.check_artifacts_belong_to = lambda **kwargs: None
  inference.check_artifacts_pair_up = lambda **kwargs: None

  contract = types.ModuleType("cobaya_theory._adapter_contract")
  contract.exact_bool = lambda *args, **kwargs: False
  contract.pick_device = lambda *args, **kwargs: None
  contract.resolve_emulator_roots = lambda *args, **kwargs: []
  contract.validate_extra_args = lambda *args, **kwargs: None

  module_name = "real_cobaya_dark_energy_mps_adapter"
  path = ROOT / "cobaya_theory" / "emul_mps.py"
  spec = importlib.util.spec_from_file_location(module_name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  replacements = {
    "emulator.inference": inference,
    "cobaya_theory._adapter_contract": contract,
  }
  try:
    with mock.patch.dict(sys.modules, replacements):
      spec.loader.exec_module(module)
  except Exception:
    sys.modules.pop(module_name, None)
    raise
  return module


class RealCobayaDarkEnergyRouteTests(unittest.TestCase):
  """Follow one nonzero-wa point through real Cobaya and the adapter helper."""

  def test_dropped_sum_becomes_nonzero_wa_before_adapter_reconstruction(self):
    """Cobaya sends ``w=-0.9, wa=0.2`` and the adapter rebuilds the sum."""
    try:
      from cobaya.likelihood import Likelihood
      from cobaya.model import get_model
      from cobaya.theory import Theory
    except ImportError as error:
      self.skipTest("real Cobaya is unavailable: " + str(error))

    adapter_module = _load_adapter_without_neural_network_runtime()
    self.addCleanup(sys.modules.pop, adapter_module.__name__, None)

    class RoutingMPS(adapter_module.emul_mps):
      """Use the production coordinate route without loading an artifact."""

      seen = []

      def initialize(self):
        artifact = types.SimpleNamespace(fixed_facts={
          "dark_energy_law": "w0wa-cpl",
          "dark_energy_inputs": ["w", "wa"],
          "cosmology_fixed": {"w": "n/a", "wa": "n/a"},
        })
        requirements = {"w": None, "w0pwa": None}
        (self._dark_energy_law,
         self._fixed_dark_energy,
         self._dark_energy_needed) = adapter_module._dark_energy_contract(
           artifact,
           {"w", "w0pwa"},
           requirements,
           need_base=True,
         )
        self._req = requirements
        self._sigma8_requested = False

      def initialize_with_provider(self, provider):
        # The production override verifies saved artifacts.  This focused
        # witness has none, so retain only Cobaya's provider registration.
        Theory.initialize_with_provider(self, provider)

      def get_can_provide_params(self):
        return ["dark_energy_witness"]

      def calculate(self, state, want_derived=True, **params):
        resolved = adapter_module._resolved_dark_energy_point(
          params,
          law=self._dark_energy_law,
          fixed=self._fixed_dark_energy,
          needed=self._dark_energy_needed,
        )
        type(self).seen.append((dict(params), resolved))
        if want_derived:
          state["derived"]["dark_energy_witness"] = resolved["wa"]
        return True

    class WitnessLikelihood(Likelihood):
      """Request the derived number so Cobaya must evaluate the Theory."""

      seen = []

      def get_requirements(self):
        return {"dark_energy_witness": None}

      def calculate(self, state, want_derived=True, **params):
        del want_derived, params
        value = self.provider.get_param("dark_energy_witness")
        type(self).seen.append(value)
        state["logp"] = 0.0

    # Cobaya reads a component's Python module while finding defaults.  These
    # focused classes are made visible in the already loaded adapter module so
    # the same discovery path used by an ordinary external Theory succeeds.
    RoutingMPS.__module__ = adapter_module.__name__
    WitnessLikelihood.__module__ = adapter_module.__name__
    adapter_module.RoutingMPS = RoutingMPS
    adapter_module.WitnessLikelihood = WitnessLikelihood

    model = get_model({
      "params": {
        "w0pwa": {
          "prior": {"min": -5.0, "max": -0.01},
          "drop": True,
        },
        "w": {"prior": {"min": -3.0, "max": -0.01}},
        "wa": {
          "value": "lambda w0pwa, w: w0pwa - w",
          "derived": False,
        },
        "dark_energy_witness": {"derived": True},
      },
      "theory": {"mps": RoutingMPS},
      "likelihood": {"witness": WitnessLikelihood},
    })

    theory = model.theory["mps"]
    self.assertEqual(theory.input_params, ["w", "wa"])
    self.assertNotIn("w0pwa", theory.input_params)
    self.assertEqual(theory.output_params, ["dark_energy_witness"])

    # Compile the production publication decision without importing the MPI
    # generator runtime.  This gives the producer and consumer the same real
    # Cobaya Parameterization object, which is the identity this test exists
    # to protect.
    from ai.tests.test_generator_dark_energy_facts import (
      _compile_publication_boundary,
    )
    publication = _compile_publication_boundary()
    fixed, varying, law, inputs = publication.dark_energy_helper(
      model.parameterization, {})
    self.assertEqual(fixed, {})
    self.assertEqual(varying, {"w", "wa"})
    self.assertEqual(law, "w0wa-cpl")
    self.assertEqual(inputs, ["w", "wa"])

    producer_point = dict(model.parameterization.to_input({
      "w0pwa": -0.7,
      "w": -0.9,
    }))
    base_inputs = {
      "As_1e9": 2.1,
      "ns": 0.965,
      "H0": 67.0,
      "omegab": 0.049,
      "omegam": 0.31,
    }
    producer_coordinates = adapter_module.syren_base.syren_params_from(
      {**base_inputs, **producer_point}, dark_energy_law=law)[-2:]

    result = model.logposterior({"w0pwa": -0.7, "w": -0.9})
    incoming, resolved = RoutingMPS.seen[-1]
    self.assertEqual(set(incoming), {"w", "wa"})
    self.assertEqual(incoming["w"], -0.9)
    self.assertAlmostEqual(incoming["wa"], 0.2)
    self.assertEqual(resolved["w"], -0.9)
    self.assertEqual(resolved["w0"], -0.9)
    self.assertAlmostEqual(resolved["wa"], 0.2)
    self.assertAlmostEqual(resolved["w0pwa"], -0.7)
    consumer_coordinates = adapter_module.syren_base.syren_params_from(
      {**base_inputs, **resolved}, dark_energy_law=law)[-2:]
    self.assertEqual(producer_coordinates, consumer_coordinates)
    self.assertAlmostEqual(WitnessLikelihood.seen[-1], 0.2)
    self.assertAlmostEqual(result.derived[0], 0.2)


if __name__ == "__main__":
  unittest.main()
