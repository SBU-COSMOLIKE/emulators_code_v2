#!/usr/bin/env python3
"""Trace f(delta-chi2 > thr) vs N_train for a SCALAR emulator (D-MP5).

The scalar sibling of sweep_ntrain_emulator_cosmic_shear.py: same YAML
schema as train_scalar_emulator.py (data.outputs marks the family), one
fresh training per grid point, serial (scalar trainings are cheap; the
multi-GPU pool stays the cosmic-shear driver's tool). Writes <out>.txt
(the np.loadtxt-loadable curve) and <out>.pdf under --fileroot.

Example:
  python .../sweep_ntrain_scalar_emulator.py \\
    --root projects/lsst_y1/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml scalar_emulator.yaml \\
    --n-min 2000 --n-points 6 --out ntrain_scalar
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import add_sweep_args, run_ntrain_sweep


def main():
  """Parse the shared sweep CLI and run the serial scalar sweep."""
  parser = argparse.ArgumentParser(prog="sweep_ntrain_scalar_emulator")
  add_cocoa_path_args(parser)
  add_sweep_args(parser)
  parser.add_argument("--out",
                      dest="out",
                      help="output name root under --fileroot "
                           "(default ntrain_scalar)",
                      type=str,
                      default="ntrain_scalar")
  args, _ = parser.parse_known_args()
  run_ntrain_sweep(args, family="scalar", out_default="ntrain_scalar")


if __name__ == "__main__":
  main()
