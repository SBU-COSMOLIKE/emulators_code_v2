#!/usr/bin/env python3
"""Optuna-tune a SCALAR emulator's train_args (D-MP5).

The scalar sibling of tune_single_emulator_cosmic_shear.py: same YAML
schema as train_scalar_emulator.py (data.outputs marks the family), with
[default, min, max, kind] search ranges on any train_args leaf; the
study minimizes the best epoch's frac(delta-chi2 > 0.2). Serial and
in-memory (the multi-GPU journal study stays the cosmic-shear tune
driver's tool).

Example:
  python .../tune_scalar_emulator.py \\
    --root projects/lsst_y1/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml scalar_emulator.yaml \\
    --n-trials 50
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import add_tune_args, run_tune


def main():
  """Parse the shared tune CLI and run the serial scalar study."""
  parser = argparse.ArgumentParser(prog="tune_scalar_emulator")
  add_cocoa_path_args(parser)
  add_tune_args(parser)
  args, _ = parser.parse_known_args()
  run_tune(args, family="scalar")


if __name__ == "__main__":
  main()
