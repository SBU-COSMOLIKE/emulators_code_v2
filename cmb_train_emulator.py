#!/usr/bin/env python3
"""Train one CMB-spectrum emulator (a data.cmb YAML) from a YAML.

A CMB emulator maps cosmological parameters to one spectrum's C_ell
values (TT, TE, EE, or phi-phi) on a fixed multipole grid, whitened per
multipole by the error bar sigma_ell from the covariance script's .npz;
an optional amplitude law (as_exp2tau) divides the known A_s e^(-2 tau)
scaling out of the target and multiplies it back at inference. One
emulator learns ONE spectrum.

This is a thin wrapper over cosmic_shear_train_emulator.py's
main() — the training stack is identical for every family; the wrapper
pins the family, so a YAML without a data.cmb block fails here with the
right driver's name instead of deep inside config validation.

PS: whitened = each multipole divided by its own error bar, so the
network sees unit-variance targets; sigma_ell = the per-multipole
Gaussian error bar (Motloch & Hu eq 3) computed once by
compute_data_vectors/compute_cmb_covariance.py.
"""

#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
# python .../emulators_code_v2/cmb_train_emulator.py \
#   --root projects/lsst_y1/ \
#   --fileroot emulators/training_scripts/ \
#   --yaml cmb_emulator.yaml \
#   --diagnostic diagnostic
#
#- The YAML must carry a data.cmb block (spectrum / covariance /
#  amplitude_law + names); the full commented example is
#  example_yamls/cmb_emulator.yaml, and the README's CMB section walks
#  every key. All flags (--save, --diagnostic, --quiet, ...) are the
#  shared train-driver flags; --diagnostic appends the two CMB pages
#  (per-multipole residual bands + the short-period wiggle content) to
#  the standard PDF.

# main (cosmic_shear_train_emulator.py): the whole train driver —
# the CLI (--save/--diagnostic/--quiet/...), training, saving, and
# the per-family diagnostics pages. The wrapper only pins the
# family: family="cmb" makes a wrong-family YAML fail at
# startup NAMING the right driver (require_family_block).
from cosmic_shear_train_emulator import main

if __name__ == "__main__":
  main(prog="cmb_train_emulator", family="cmb")
