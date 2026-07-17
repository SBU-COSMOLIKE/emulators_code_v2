#!/usr/bin/env python3
"""Run the CPU evidence for artifact naming and occupied-root refusal.

The identity leg checks that scientific products, resolved recipes, exact
dataset selections, and authenticated reusable sources receive distinct
names.  It also checks that dictionary order and local path spelling do not
rename unchanged inputs.

The publication leg checks the saved pair boundary.  An existing complete,
partial, symbolic-link, or interrupted destination must remain byte-for-byte
unchanged, and the saver must refuse before it creates a temporary file.
"""

import io
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ai.tests import test_artifact_output_identity
from ai.tests import test_results_artifact_pair


LEGS = (
  ("artifact-output-identity.scientific-identity",
   test_artifact_output_identity),
  ("artifact-output-identity.existing-root-refusal",
   test_results_artifact_pair),
)


def _run_leg(aid, module):
  """Run every test in one module and print one machine-readable verdict."""
  suite = unittest.defaultTestLoader.loadTestsFromModule(module)
  expected = suite.countTestCases()
  stream = io.StringIO()
  result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  transcript = stream.getvalue().rstrip()
  if transcript:
    print(transcript)
  passed = (expected > 0 and result.wasSuccessful()
            and result.testsRun == expected)
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + aid + " ("
        + str(result.testsRun) + "/" + str(expected) + " witnesses)")
  print("##AID " + aid + " " + mark)
  return passed


def main():
  """Return success only when naming and occupied-root checks both pass."""
  passed = [_run_leg(aid, module) for aid, module in LEGS]
  if not all(passed):
    print("artifact-output-identity: FAIL")
    return 1
  print("artifact-output-identity: ALL PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
