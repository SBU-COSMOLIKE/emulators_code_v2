#!/usr/bin/env python3
"""Sweep ONE YAML-chosen hyperparameter for a scalar (data.outputs) emulator.

The scalar sibling of cosmic_shear_sweep_hyperparam_emulator.py: the
YAML's `sweep` block names one dotted train_args leaf (or the
activation family) and a value list; one fresh training per value at
fixed N_train, serial (the multi-GPU pool stays the cosmic-shear
driver's tool). Writes <out>.txt (the value/frac table) and <out>.pdf
under --fileroot.

Example:
  python .../scalar_sweep_hyperparam_emulator.py \\
    --root projects/lsst_y1/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml scalar_emulator.yaml \\
    --out lrsweep_scalar

  with, in the YAML:
    sweep:
      parameter: lr.lr_base
      values:
        - 0.001
        - 0.0025
        - 0.005
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import (
  add_hyperparam_args, run_hyperparam_sweep)


def main():
  """Parse the shared sweep CLI and run the serial scalar sweep."""
  parser = argparse.ArgumentParser(prog="scalar_sweep_hyperparam_emulator")
  add_cocoa_path_args(parser)
  add_hyperparam_args(parser)
  parser.add_argument("--out",
                      dest="out",
                      help="output name root under --fileroot "
                           "(default hyperparam_scalar)",
                      type=str,
                      default="hyperparam_scalar")
  args, _ = parser.parse_known_args()
  run_hyperparam_sweep(args, family="scalar",
                       out_default="hyperparam_scalar")


if __name__ == "__main__":
  main()
