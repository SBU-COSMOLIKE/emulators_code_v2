#!/usr/bin/env python3
"""Train one background (BAO/SN) emulator (a data.grid YAML) from a YAML.

A background emulator maps cosmological parameters to one background
quantity on a stored redshift grid: H(z) on the supernova window
(z in [0, ~3], trained under the log_offset law) or the comoving
distance D_M on the recombination window (z in [1000, 1200], trained
raw). Every distance an MCMC needs is computed FROM the served H(z) by
emulator/background.py (cumulative Simpson, flat conversions) — only
H(z) and D_M are networks; two artifacts serve the whole background.

This is a thin wrapper over train_single_emulator_cosmic_shear.py's
main() — the training stack is identical for every family; the wrapper
pins the family, so a YAML without a data.grid block fails here with
the right driver's name instead of deep inside config validation.

PS: law = a known analytic transform divided out of the training target
and undone at decode (log_offset: the network learns log(H + offset));
grid = the ascending redshift grid persisted into the artifact from the
generator's _z.npy sidecar.
"""

#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
# python .../emulators_code_v2/train_baosn_emulator.py \
#   --root projects/lsst_y1/ \
#   --fileroot emulators/training_scripts/ \
#   --yaml baosn_hubble_emulator.yaml \
#   --diagnostic diagnostic
#
#- The YAML must carry a data.grid block (quantity / units / law /
#  offset / z_file); the full commented example is
#  example_yamls/baosn_hubble_emulator.yaml, and the README's
#  expansion-history section walks every key. All flags are the shared
#  train-driver flags; --diagnostic appends the per-redshift residual
#  bands and, for a Hubble artifact, the derived-distance page computed
#  through the real integration pipeline.

from train_single_emulator_cosmic_shear import main

if __name__ == "__main__":
  main(prog="train_baosn_emulator", family="grid")
