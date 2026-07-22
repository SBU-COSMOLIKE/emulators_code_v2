"""CPU checks for dark-energy coordinates in the matter-power adapter.

The checks themselves live in ai/tests/mps_dark_energy_child_checks.py
and run in a child process, launched exactly as
ai/tests/test_mps_sigma8_contract.py launches its own numeric child:
the on-disk stand-in package ai/tests/cobaya_minimal_stub is placed
ahead of every installed package on the child's PYTHONPATH before the
child starts, so this process's import table and the shared
``emulator`` package are never touched.
"""

from pathlib import Path
import sys
import unittest

from ai.tests.test_mps_sigma8_contract import run_isolated_child_checks


ROOT = Path(__file__).resolve().parents[2]
CHILD_CHECKS = ROOT / "ai" / "tests" / "mps_dark_energy_child_checks.py"


class DarkEnergyChildProcessTests(unittest.TestCase):
  """Launch the isolated dark-energy adapter checks."""

  def test_dark_energy_checks_pass_in_child_process(self):
    """The child suite passes without touching this process's modules."""
    modules_before = set(sys.modules)
    completed = run_isolated_child_checks(CHILD_CHECKS)
    self.assertEqual(
      completed.returncode, 0,
      "child dark-energy adapter checks failed:\n" + completed.stdout
      + completed.stderr)
    # All five checks must actually have run; a child that collected
    # nothing would exit 0 without testing anything.
    self.assertIn("Ran 5 tests", completed.stderr)
    self.assertEqual(modules_before, set(sys.modules))


if __name__ == "__main__":
  unittest.main()
