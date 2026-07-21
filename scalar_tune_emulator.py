#!/usr/bin/env python3
"""Run an Optuna hyperparameter search for a scalar (derived-parameter) emulator.

The scalar sibling of cosmic_shear_tune_emulator.py, and a thin wrapper
over that driver's main(): the SAME code path, so every capability
carries over — serial on one GPU / Apple MPS, or a multi-GPU study
(--n-gpus) cooperating through a shared journal file. Reusing --journal
resumes only when its scientific manifest matches exactly. The YAML is the training
driver's, with a data.outputs block marking the family and
[default, min, max, kind] search ranges on any train_args leaf; the
study minimizes the best epoch's frac(delta-chi2 > 0.2).

Example:
  python .../scalar_tune_emulator.py \\
    --root projects/lsst_y1/ \\
    --fileroot emulators/training_scripts/ \\
    --yaml scalar_emulator.yaml \\
    --n-trials 40
"""

# main (cosmic_shear_tune_emulator.py): the whole Optuna driver — the
# CLI (--n-trials/--timeout/--n-gpus/--journal), trial suggestion off
# the YAML's [default, min, max, kind] ranges, the serial in-memory
# study and the multi-GPU journal study. The wrapper only pins the
# family: family="outputs" makes a YAML without a data.outputs block fail
# at startup NAMING the right driver (require_family_block), and the
# stable family name comes from resolve_study_name in the shared tune
# driver. Renaming this program
# label therefore does not fork or silently rename the Optuna study.
from cosmic_shear_tune_emulator import main

if __name__ == "__main__":
  main(prog="scalar_tune_emulator", family="outputs")
