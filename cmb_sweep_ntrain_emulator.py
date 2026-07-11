#!/usr/bin/env python3
"""Trace f(delta-chi2 > thr) vs N_train for a CMB-spectrum emulator.

The cmb sibling of cosmic_shear_sweep_ntrain_emulator.py, and a thin
wrapper over that driver's main(): the SAME code path, so every
capability carries over — the multi-GPU pool (--n-gpus), --gpu-pack
co-location, the LPT balance, and the serial path on one GPU / Apple
MPS. The YAML is the training driver's, with a data.cmb block marking
the family; one fresh training per grid point; writes <out>.txt (the
np.loadtxt-loadable curve) and <out>.pdf under --fileroot.

Example:
  python .../cmb_sweep_ntrain_emulator.py \\
    --root projects/lsst_y1/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml cmb_emulator.yaml \\
    --n-min 2000 --n-points 6 --out ntrain_cmb
"""

# main (cosmic_shear_sweep_ntrain_emulator.py): the whole sweep driver —
# the CLI (--n-min/--n-max/--n-points/--threshold/--n-gpus/--gpu-pack/
# --out), the serial and multi-GPU paths, and the outputs. The wrapper
# only pins the family: family="cmb" makes a YAML without a
# data.cmb block fail at startup NAMING the right driver
# (require_family_block), and out_default keeps this family's own
# output name when --out is absent.
from cosmic_shear_sweep_ntrain_emulator import main

if __name__ == "__main__":
  main(prog="cmb_sweep_ntrain_emulator", family="cmb",
       out_default="ntrain_cmb")
