# README draft: the BAOSN background-emulators section (BSN)

**Date:** 2026-07-11. **Status:** DRAFT — paste into the MAIN checkout's
README beside the CMB section ([[readme-cmb-section-draft]]), same
renumbering ritual. Body below is README-ready.

---

## Emulating the expansion history (H(z), BAO and SN distances)

Only H(z) is a network; every distance is known physics computed from
it. The BAOSN family serves the background — the Hubble rate and the
comoving / angular-diameter / luminosity distances — to BAO and
supernova likelihoods from TWO small artifacts:

    the "Hubble" artifact                the "D_M" artifact
    H(z) on the SN range, z in [0, 3]    the comoving distance, trained
       │                                 directly on the recombination
       │  emulator/background.py:        window z in [1000, 1200] (the
       │  chi(z) = integral of c/H       CMB-distance anchor)
       ▼  (cumulative Simpson)
    D_C = chi, D_A = chi/(1+z),
    D_L = chi*(1+z)   (flat)

Nothing is emulated between the two windows — no likelihood queries
that desert — and a query there is a loud error, never a silent bridge.
The training target for H(z) is log(H + offset) (the `log_offset` law,
persisted in the artifact); D_M trains raw (`none`). Dumps come from
`compute_data_vectors/dataset_generator_background.py`: one
background-only CAMB evaluation per sampled cosmology yields BOTH
quantities and the grids ride beside the dumps as `_z.npy` sidecars.

```yaml
data:
  grid:
    quantity: Hubble        # or D_M for the recombination artifact
    units:    km/s/Mpc      # Mpc for D_M
    law:      log_offset    # none for D_M
    offset:   0.0
    z_file:   dvs_train_background_unifs_h_z.npy
```

Serving in an MCMC pairs the two artifacts in one theory block (rdrag
comes separately, from the scalar-emulator family):

```yaml
theory:
  emul_baosn:
    python_path: ./cobaya_theory/
    extra_args:
      device: cuda
      emulators:
        - chains/emulator_hubble_resmlp_ntrain50000
        - chains/emulator_dm_resmlp_ntrain50000
```

get_Hubble (km/s/Mpc or 1/Mpc), get_comoving_radial_distance,
get_angular_diameter_distance (+ the two-redshift variant), and
get_luminosity_distance are served piecewise by query redshift. V1 is
flat-only (a sampled omk is a loud error; the legacy curvature formula
was dimensionally wrong and is not reproduced). Fine-tuning works per
artifact (same quantity, grid, units, and law; the model: block is
inherited). `--diagnostic` adds the per-redshift residual bands and,
for the Hubble artifact, the derived-distance page computed through the
real integration pipeline. Gates: bsn-identity / bsn-smoke — the smoke
checks the served values against CAMB's own background, so it is the
strongest end-to-end test in the board.

---

**Also update:** the code-map rows (emulator/background.py,
emulator/geometries_grid.py, dataset_generator_background.py,
cobaya_theory/emul_baosn.py, the family driver pairs), Contents anchors,
appendix 20's direct-scripting pattern gains the BSN example
(EmulatorPredictor -> {"z", "Hubble"} -> background.distance_interpolators).
