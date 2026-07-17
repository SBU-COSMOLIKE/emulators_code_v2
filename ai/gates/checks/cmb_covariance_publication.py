#!/usr/bin/env python3
"""Run the CPU checks for safe CMB covariance publication.

The checks use tiny NumPy archives. They do not run CAMB or train an emulator.
They cover a failed private write, a late competing file, cleanup, and the
command's early existing-file refusal.
"""

import io
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ai.tests import test_cmb_covariance_publication


def main():
  """Return success only when every publication boundary passes."""
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(
    test_cmb_covariance_publication.CmbCovariancePublicationTests)
  expected = suite.countTestCases()
  stream = io.StringIO()
  result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  transcript = stream.getvalue().rstrip()
  if transcript:
    print(transcript)
  complete = expected > 0 and result.testsRun == expected
  if complete and result.wasSuccessful() and not result.skipped:
    print("  [PASS] cmb-covariance-publication.transactional-output ("
          + str(result.testsRun) + "/" + str(expected) + " checks)")
    print("##AID cmb-covariance-publication.transactional-output PASS")
    print("cmb-covariance-publication: ALL PASS")
    return 0
  print("  [FAIL] cmb-covariance-publication.transactional-output ("
        + str(result.testsRun) + "/" + str(expected) + " checks)")
  print("##AID cmb-covariance-publication.transactional-output FAIL")
  print("cmb-covariance-publication: FAIL")
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
