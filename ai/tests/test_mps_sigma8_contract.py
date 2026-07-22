"""CPU checks for the conventional sigma-eight calculation.

The numeric checks live in ai/tests/mps_sigma8_child_checks.py and run
in a child process.  That child imports the matter-power adapter through
the on-disk stand-in package ai/tests/cobaya_minimal_stub, placed ahead
of every installed package on the child's PYTHONPATH before the child
starts.  Loading the adapter that way keeps this process's import table
and the shared ``emulator`` package untouched, so no other test can
observe a half-restored module.  This file launches the child, requires
it to pass, and launches one negative control that must fail — proving
the child's known-answer assertion still bites and that a child failure
reaches this process as a nonzero exit code.

The live-Cobaya routing test stays in this process because it needs the
real installed cobaya, not the stand-in.
"""

import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHILD_CHECKS = ROOT / "ai" / "tests" / "mps_sigma8_child_checks.py"
COBAYA_STUB_DIR = ROOT / "ai" / "tests" / "cobaya_minimal_stub"


def run_isolated_child_checks(child_path, extra_env=None):
  """Run one child-process check file against the cobaya stand-in.

  The child's PYTHONPATH is written before launch: the cobaya stand-in
  directory first, then the repository root (for ``cobaya_theory`` and
  ``emulator``).  Any inherited PYTHONPATH is replaced so the child's
  imports do not depend on this process's environment.

  Arguments:
    child_path = the check file to run (a Path; the file ends in
                 ``unittest.main()``).
    extra_env  = optional mapping merged into the child's environment;
                 the negative control passes the wrong-expectation key.

  Returns:
    the finished subprocess.CompletedProcess with captured text output.
  """
  env = dict(os.environ)
  env["PYTHONPATH"] = str(COBAYA_STUB_DIR) + os.pathsep + str(ROOT)
  if extra_env is not None:
    env.update(extra_env)
  return subprocess.run(
    [sys.executable, str(child_path), "-v"],
    env=env,
    capture_output=True,
    text=True,
    check=False)


class SigmaEightChildProcessTests(unittest.TestCase):
  """Launch the isolated numeric checks and their negative control."""

  def test_numeric_checks_pass_in_child_process(self):
    """The child suite passes without touching this process's modules."""
    modules_before = set(sys.modules)
    completed = run_isolated_child_checks(CHILD_CHECKS)
    self.assertEqual(
      completed.returncode, 0,
      "child sigma-eight checks failed:\n" + completed.stdout
      + completed.stderr)
    # All five numeric checks must actually have run; a child that
    # collected nothing would exit 0 without testing anything.
    self.assertIn("Ran 5 tests", completed.stderr)
    self.assertEqual(modules_before, set(sys.modules))

  def test_negative_control_wrong_expectation_fails_in_child(self):
    """A wrong known answer must fail the child, visibly, at the parent."""
    completed = run_isolated_child_checks(
      CHILD_CHECKS,
      extra_env={"MPS_SIGMA8_EXPECTED_OVERRIDE": "0.9"})
    self.assertNotEqual(
      completed.returncode, 0,
      "the child accepted a wrong sigma-eight known answer")
    self.assertIn("test_analytic_known_answer_uses_eight_over_h_mpc",
                  completed.stderr)


class RealCobayaRoutingTests(unittest.TestCase):
  """Exercise the live dependency resolver when cobaya is installed."""

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
