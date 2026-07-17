#!/usr/bin/env python3
"""Run the CPU checks that protect dataset-generator input handling.

The checks use small dictionaries, text covariance files, and NumPy arrays.
They do not start a sampler or create a scientific dataset.  Together they
show that ambiguous settings refuse before output begins, an MCMC row
shortfall cannot publish a smaller table, and an optional display label has a
safe fallback.
"""

import io
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ai.tests import test_generator_ingress


AID = "generator-ingress.valid-before-output"


def main():
  """Return success only when every generator-ingress test passes."""
  suite = unittest.defaultTestLoader.loadTestsFromModule(
    test_generator_ingress)
  expected = suite.countTestCases()
  stream = io.StringIO()
  result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  transcript = stream.getvalue().rstrip()
  if transcript:
    print(transcript)

  complete = expected > 0 and result.testsRun == expected
  passed = complete and result.wasSuccessful() and not result.skipped
  status = "PASS" if passed else "FAIL"
  print("  [" + status + "] " + AID + " (" + str(result.testsRun)
        + "/" + str(expected) + " checks)")
  print("##AID " + AID + " " + status)
  print("generator-ingress: " + ("ALL PASS" if passed else "FAIL"))
  return 0 if passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
