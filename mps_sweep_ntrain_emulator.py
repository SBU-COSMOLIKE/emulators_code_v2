#!/usr/bin/env python3
"""Trace f(delta-chi2 > thr) vs N_train for an MPS grid2d emulator (D-MP5).

The matter-power-spectrum sibling of cosmic_shear_sweep_ntrain_emulator.py:
same YAML schema as the training driver with a data.grid2d block (the
linear P(k, z) or the nonlinear boost — one artifact per run), one fresh
training per grid point, serial. Writes <out>.txt (the
np.loadtxt-loadable curve) and <out>.pdf under --fileroot.

Example:
  python .../mps_sweep_ntrain_emulator.py \\
    --root projects/mps/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml mps_boost_emulator.yaml \\
    --n-min 2000 --n-points 6 --out ntrain_mps_boost
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import add_sweep_args, run_ntrain_sweep


def main():
  """Parse the shared sweep CLI and run the serial MPS sweep."""
  parser = argparse.ArgumentParser(prog="mps_sweep_ntrain_emulator")
  add_cocoa_path_args(parser)
  add_sweep_args(parser)
  parser.add_argument("--out",
                      dest="out",
                      help="output name root under --fileroot "
                           "(default ntrain_mps)",
                      type=str,
                      default="ntrain_mps")
  args, _ = parser.parse_known_args()
  run_ntrain_sweep(args, family="mps", out_default="ntrain_mps")


if __name__ == "__main__":
  main()
