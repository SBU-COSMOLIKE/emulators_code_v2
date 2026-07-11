#!/usr/bin/env python3
"""Trace f(delta-chi2 > thr) vs N_train for a BAOSN grid emulator (D-MP5).

The background-family sibling of cosmic_shear_sweep_ntrain_emulator.py:
same YAML schema as the training driver with a data.grid block (H(z) on
the SN range, or D_M on the recombination window — one artifact per
run), one fresh training per grid point, serial. Writes <out>.txt (the
np.loadtxt-loadable curve) and <out>.pdf under --fileroot.

Example:
  python .../baosn_sweep_ntrain_emulator.py \\
    --root projects/baosn/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml baosn_hubble_emulator.yaml \\
    --n-min 2000 --n-points 6 --out ntrain_baosn_h
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import add_sweep_args, run_ntrain_sweep


def main():
  """Parse the shared sweep CLI and run the serial BAOSN sweep."""
  parser = argparse.ArgumentParser(prog="baosn_sweep_ntrain_emulator")
  add_cocoa_path_args(parser)
  add_sweep_args(parser)
  parser.add_argument("--out",
                      dest="out",
                      help="output name root under --fileroot "
                           "(default ntrain_baosn)",
                      type=str,
                      default="ntrain_baosn")
  args, _ = parser.parse_known_args()
  run_ntrain_sweep(args, family="baosn", out_default="ntrain_baosn")


if __name__ == "__main__":
  main()
