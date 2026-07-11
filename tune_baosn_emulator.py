#!/usr/bin/env python3
"""Optuna-tune a BAOSN grid emulator's train_args (D-MP5).

The background-family sibling of tune_single_emulator_cosmic_shear.py:
same YAML schema as the training driver with a data.grid block, with
[default, min, max, kind] search ranges on any train_args leaf; the
study minimizes the best epoch's frac(delta-chi2 > 0.2). Serial and
in-memory.

Example:
  python .../tune_baosn_emulator.py \\
    --root projects/baosn/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml baosn_hubble_emulator.yaml \\
    --n-trials 50
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import add_tune_args, run_tune


def main():
  """Parse the shared tune CLI and run the serial BAOSN study."""
  parser = argparse.ArgumentParser(prog="tune_baosn_emulator")
  add_cocoa_path_args(parser)
  add_tune_args(parser)
  args, _ = parser.parse_known_args()
  run_tune(args, family="baosn")


if __name__ == "__main__":
  main()
