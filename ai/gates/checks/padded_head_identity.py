#!/usr/bin/env python3
"""Run the CPU evidence for physical coordinates in padded model heads.

The first leg exercises live CNN and Transformer heads.  It checks that a
geometry's saved coordinate map, rather than the rank of a surviving value,
decides where each value enters the rectangular head.  Artificial positions
must remain zero after convolutions, attention, FiLM, and repeated blocks.

The second leg saves and rebuilds a structured-head artifact.  A valid saved
geometry and checkpoint must reproduce the same result. A live mismatch must
stop before staging, and a missing or disagreeing saved model buffer must stop
before state loading.
"""

import io
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ai.tests import test_padded_head_artifact
from ai.tests import test_padded_head_identity


LEGS = (
  ("padded-head-identity.layout", test_padded_head_identity),
  ("padded-head-identity.artifact", test_padded_head_artifact),
)


def _run_leg(aid, module):
  """Run every test in one focused module and emit its single verdict."""
  suite = unittest.defaultTestLoader.loadTestsFromModule(module)
  expected = suite.countTestCases()
  stream = io.StringIO()
  result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  transcript = stream.getvalue().rstrip()
  if transcript:
    print(transcript)
  passed = expected > 0 and result.wasSuccessful() \
    and result.testsRun == expected
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + aid + " ("
        + str(result.testsRun) + "/" + str(expected) + " witnesses)")
  print("##AID " + aid + " " + mark)
  return passed


def main():
  passed = []
  for aid, module in LEGS:
    passed.append(_run_leg(aid, module))
  if not all(passed):
    print("padded-head-identity: FAIL")
    return 1
  print("padded-head-identity: ALL PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
