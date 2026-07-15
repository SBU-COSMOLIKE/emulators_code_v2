#!/usr/bin/env python3
"""Catch stale machine guidance in finite-contract's non-green summary."""

from pathlib import Path


AI_ROOT = Path(__file__).resolve().parents[1]
SOURCE = AI_ROOT / "gates" / "checks" / "finite_contract.py"
EXPECTED_SUMMARY = (
  '    print("finite-contract: NON-GREEN -- a mandatory lane could not run: "\n'
  '          + ", ".join(LANE_UNAVAILABLE)\n'
  '          + " (run on a compile-capable CUDA box)")')


def main():
  """Require the summary to name the CUDA capability its lane needs."""
  source = SOURCE.read_text(encoding="utf-8")
  if EXPECTED_SUMMARY not in source:
    print("FAIL: finite-contract non-green summary does not direct the user "
          "to a compile-capable CUDA box")
    return 1
  print("PASS: finite-contract non-green summary names a compile-capable "
        "CUDA box")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
