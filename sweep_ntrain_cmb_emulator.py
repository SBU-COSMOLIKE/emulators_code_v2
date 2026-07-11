#!/usr/bin/env python3
"""Trace f(delta-chi2 > thr) vs N_train for a CMB emulator (D-MP5).

The CMB sibling of sweep_ntrain_emulator_cosmic_shear.py: same YAML
schema as the training driver with a data.cmb block (see
example_yamls/cmb_emulator.yaml), one fresh training per grid point,
serial on the run's device. Writes <out>.txt (the np.loadtxt-loadable
curve) and <out>.pdf under --fileroot.

Example:
  python .../sweep_ntrain_cmb_emulator.py \\
    --root projects/cmb/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml cmb_emulator.yaml \\
    --n-min 2000 --n-points 6 --out ntrain_cmb_tt
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import add_sweep_args, run_ntrain_sweep


def main():
  """Parse the shared sweep CLI and run the serial CMB sweep."""
  parser = argparse.ArgumentParser(prog="sweep_ntrain_cmb_emulator")
  add_cocoa_path_args(parser)
  add_sweep_args(parser)
  parser.add_argument("--out",
                      dest="out",
                      help="output name root under --fileroot "
                           "(default ntrain_cmb)",
                      type=str,
                      default="ntrain_cmb")
  args, _ = parser.parse_known_args()
  run_ntrain_sweep(args, family="cmb", out_default="ntrain_cmb")


if __name__ == "__main__":
  main()
