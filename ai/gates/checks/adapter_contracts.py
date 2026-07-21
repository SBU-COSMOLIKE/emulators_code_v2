#!/usr/bin/env python3
"""Run the CPU checks shared by all five Cobaya emulator adapters.

The first group checks the values read from YAML and the order used when
several cosmic-shear sections form one result.  For example, a misspelled
extra_args key is refused instead of being ignored, and two sections are
placed in physical data-vector order rather than YAML order.

The second group checks what the adapters publish to Cobaya.  For example, a
scalar result is written under Cobaya's ``derived`` entry, a saved ``As_1e9``
input does not create a second amplitude requirement named ``As``, and a
caller cannot corrupt cached arrays by changing a returned value.

Each group runs named classes from ``ai/tests/test_cobaya_adapter_contracts``.
This keeps a red gate useful: its transcript identifies which public boundary
failed without training a model or reading a scientific dataset.
"""

import io
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ai.tests import test_cobaya_adapter_contracts


LEGS = (
  (
    "adapter-contracts.strict-inputs-and-composition",
    (
      test_cobaya_adapter_contracts.SharedOptionAndPathTests,
      test_cobaya_adapter_contracts.CosmicSectionCompositionTests,
    ),
  ),
  (
    "adapter-contracts.publication-and-owned-results",
    (
      test_cobaya_adapter_contracts.ScalarAndCmbContractTests,
      test_cobaya_adapter_contracts.LiveCobayaScalarRoutingTests,
      test_cobaya_adapter_contracts.ReturnedOwnershipTests,
      test_cobaya_adapter_contracts.MatterPowerServingDomainTests,
      test_cobaya_adapter_contracts.MatterPowerDependencyTests,
      test_cobaya_adapter_contracts.BackgroundDependencyTests,
    ),
  ),
)


def _suite_from_classes(test_classes):
  """Collect every test method in the named classes."""
  suite = unittest.TestSuite()
  for test_class in test_classes:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_class))
  return suite


def _run_leg(aid, test_classes):
  """Run one related group and print its transcript plus an AID verdict."""
  suite = _suite_from_classes(test_classes)
  expected = suite.countTestCases()
  stream = io.StringIO()
  result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  transcript = stream.getvalue().rstrip()
  if transcript:
    print(transcript)
  complete = (expected > 0 and result.testsRun == expected)
  if not complete or not result.wasSuccessful():
    mark = "FAIL"
    reason = ""
  elif result.skipped:
    mark = "UNAVAILABLE"
    reasons = sorted({str(item[1]).replace("\n", " ")
                      for item in result.skipped})
    reason = "required check skipped: " + "; ".join(reasons)
  else:
    mark = "PASS"
    reason = ""
  print("  [" + mark + "] " + aid + " ("
        + str(result.testsRun) + "/" + str(expected) + " checks)")
  terminal = "##AID " + aid + " " + mark
  if reason:
    terminal += " " + reason
  print(terminal)
  return mark == "PASS"


def main():
  """Return success only when both adapter-boundary groups pass."""
  passed = [_run_leg(aid, test_classes)
            for aid, test_classes in LEGS]
  if not all(passed):
    print("adapter-contracts: FAIL")
    return 1
  print("adapter-contracts: ALL PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
