#!/usr/bin/env python3
"""Optuna-tune an MPS grid2d emulator's train_args (D-MP5).

The matter-power-spectrum sibling of cosmic_shear_tune_emulator.py:
same YAML schema as the training driver with a data.grid2d block, with
[default, min, max, kind] search ranges on any train_args leaf; the study
minimizes the best epoch's frac(delta-chi2 > 0.2). Serial and in-memory.

Example:
  python .../mps_tune_emulator.py \\
    --root projects/mps/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml mps_boost_emulator.yaml \\
    --n-trials 50
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import add_tune_args, run_tune


def main():
  """Parse the shared tune CLI and run the serial MPS study."""
  parser = argparse.ArgumentParser(prog="mps_tune_emulator")
  add_cocoa_path_args(parser)
  add_tune_args(parser)
  args, _ = parser.parse_known_args()
  run_tune(args, family="mps")


if __name__ == "__main__":
  main()
