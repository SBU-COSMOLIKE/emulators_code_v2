#!/usr/bin/env python3
"""Sweep ONE YAML-chosen hyperparameter for a scalar (derived-parameter) emulator.

The scalar sibling of cosmic_shear_sweep_hyperparam_emulator.py, and a
thin wrapper over that driver's main(): the SAME code path, so every
capability carries over — the multi-GPU pool (--n-gpus), --gpu-pack
co-location, and the serial path on one GPU / Apple MPS. The YAML's
`sweep` block names one dotted train_args leaf (or the activation
family) and a value list; one fresh training per value at fixed
N_train; writes <out>.txt (the value/frac table) and <out>.pdf under
--fileroot.

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

# main (cosmic_shear_sweep_hyperparam_emulator.py): the whole one-knob
# sweep driver — the sweep-block parsing (read_sweep_block,
# family_drivers.py), the CLI (--threshold/--activation/--n-gpus/
# --gpu-pack/--out), the serial and multi-GPU paths, and the outputs.
# The wrapper only pins the family: family="outputs" makes a YAML
# without a data.outputs block fail at startup NAMING the right driver
# (require_family_block), and out_default keeps this family's own
# output name when --out is absent.
from cosmic_shear_sweep_hyperparam_emulator import main

if __name__ == "__main__":
  main(prog="scalar_sweep_hyperparam_emulator", family="outputs",
       out_default="hyperparam_scalar")
