#!/usr/bin/env python3
"""Train one matter-power emulator (a data.grid2d YAML) from a YAML.

A matter-power emulator maps cosmological parameters to a surface on a
stored (z, k) grid — the linear P(k, z) or the nonlinear boost
B = P_nl / P_lin. Under a syren law the network learns only the
CORRECTION log(P / P_base) to the vendored analytic formulas (syren/),
with the base divided out once at staging from the generator's *_base
dumps and multiplied back at inference by the emul_mps adapter.

This is a thin wrapper over cosmic_shear_train_emulator.py's
main() — the training stack is identical for every family; the wrapper
pins the family, so a YAML without a data.grid2d block fails here with
the right driver's name instead of deep inside config validation.

PS: boost = P_nonlinear / P_linear, the ratio the second MPS artifact
emulates; base = the syren analytic surface the network corrects, read
from the generator's *_base files (never recomputed at training time);
k_stride = the k-grid thinning (top edge always kept) the served
interpolator fills back in.
"""

#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
# python .../emulators_code_v2/mps_train_emulator.py \
#   --root projects/lsst_y1/ \
#   --fileroot emulators/training_scripts/ \
#   --yaml mps_boost_emulator.yaml \
#   --diagnostic diagnostic
#
#- The YAML must carry a data.grid2d block (quantity / units / law /
#  train_base + val_base under a syren law / z_file / k_file /
#  k_stride); the full commented example is
#  example_yamls/mps_boost_emulator.yaml, and the README's
#  matter-power section walks every key. All flags are the shared
#  train-driver flags; --diagnostic appends the two matter-power pages
#  (the (z, k) |residual| surfaces, median + worst cosmology, and per-k
#  bands at three redshifts — under a syren law the residual reads as
#  ln(P_pred / P_truth), the base cancels in law space).

# main (cosmic_shear_train_emulator.py): the whole train driver —
# the CLI (--save/--diagnostic/--quiet/...), training, saving, and
# the per-family diagnostics pages. The wrapper only pins the
# family: family="grid2d" makes a wrong-family YAML fail at
# startup NAMING the right driver (require_family_block).
from cosmic_shear_train_emulator import main

if __name__ == "__main__":
  main(prog="mps_train_emulator", family="grid2d")
