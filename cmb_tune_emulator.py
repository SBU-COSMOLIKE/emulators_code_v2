#!/usr/bin/env python3
"""Optuna-tune a CMB emulator's train_args (D-MP5).

The CMB sibling of cosmic_shear_tune_emulator.py: same YAML
schema as the training driver with a data.cmb block (see
example_yamls/cmb_emulator.yaml), with [default, min, max, kind] search
ranges on any train_args leaf (the D-CM8 roughness lam / period_cut are
sweepable leaves too); the study minimizes the best epoch's
frac(delta-chi2 > 0.2). Serial and in-memory.

Example:
  python .../cmb_tune_emulator.py \\
    --root projects/cmb/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml cmb_emulator.yaml \\
    --n-trials 50
"""

import argparse

from emulator.cocoa import add_cocoa_path_args
from emulator.family_drivers import add_tune_args, run_tune


def main():
  """Parse the shared tune CLI and run the serial CMB study."""
  parser = argparse.ArgumentParser(prog="cmb_tune_emulator")
  add_cocoa_path_args(parser)
  add_tune_args(parser)
  args, _ = parser.parse_known_args()
  run_tune(args, family="cmb")


if __name__ == "__main__":
  main()
