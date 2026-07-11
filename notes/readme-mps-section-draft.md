# README draft: the MPS section + the Drivers table (MPS / D-MP5)

**Date:** 2026-07-11. **Status:** DRAFT — paste into the MAIN checkout's
README beside the CMB/BSN sections ([[readme-cmb-section-draft]],
[[readme-baosn-section-draft]]), same renumbering ritual. TWO pieces:
the MPS section and the program-wide "Drivers" subsection table the
user commissioned.

---

## Emulating the matter power spectrum (hybrid inference, EMUL2)

The MPS emulators CORRECT an approximate formula. The syren
(symbolic_pofk) expressions give an analytic P(k, z); the network
learns only the residual:

    target = log( P(k, z) / P_syren(k, z; params) )

so the amplitude and shape it must capture are gentle, and the exact
formula is multiplied back at inference. Two artifacts serve
everything:

    the "pklin" artifact               the "boost" artifact
    corrects the syren linear          corrects syren-halofit's
    formula -> P_lin(k, z) [Mpc^3]     boost -> B = P_nl / P_lin
                        \\                /
                         P_nl = B * P_lin

Dumps come from `compute_data_vectors/dataset_generator_mps.py`: one
CAMB call per sampled cosmology writes the raw surfaces AND the syren
base beside them (the training divides the base out once, at staging —
it is never recomputed under a possibly-updated package). The (z, k)
grids ride as `_z.npy` / `_k.npy` sidecars and persist into the
artifact; V1 trains on a thinned k grid (`k_stride`, top edge always
kept) and the served interpolator fills between kept points.

```yaml
data:
  grid2d:
    quantity: boost           # or pklin for the linear artifact
    units:    dimensionless   # Mpc3 for pklin
    law:      syren_halofit   # syren_linear for pklin; none = raw
    train_base: dvs_train_mps_unifs_boost_base.npy
    val_base:   dvs_val_mps_unifs_boost_base.npy
    z_file:     dvs_train_mps_unifs_z.npy
    k_file:     dvs_train_mps_unifs_k.npy
    k_stride:   10
```

**Hybrid inference (EMUL2):** cosmolike's `use_emulator: 2` mode
consumes EMULATED CAMB PRODUCTS instead of a full data-vector emulator:
P(k, z) from `emul_mps`, distances and H(z) from `emul_baosn`, r_drag
from the scalar-emulator family — three theories in one sampling YAML.
`emul_mps` serves `get_Pk_grid` / `get_Pk_interpolator` (linear and
nonlinear) through the CAMB-compatible interpolator (adapted from CAMB
by Antony Lewis), so a likelihood written against CAMB's provider needs
no change:

```yaml
theory:
  emul_mps:
    python_path: ./cobaya_theory/
    extra_args:
      device: cuda
      emulators:
        - chains/emulator_pklin_resmlp_ntrain50000
        - chains/emulator_boost_resmlp_ntrain50000
```

The acceptance experiment for the unit is the full EMUL2 evaluate run
(the EXAMPLE_EMUL2_EVALUATE1.yaml pattern) with all three theories.
Fine-tuning works per artifact (same quantity, law, and grids);
transfer learning is permanently out for this family (it is exclusive
to the cosmolike and CMB data-vector families). Gates: mps-identity /
mps-smoke.

---

## The Drivers table (a new README subsection, D-MP5)

The driver namespace is `<verb>_<family>_emulator.py`. Every family
trains through the same config dispatch (`data.outputs` -> scalar,
`data.cmb` -> CMB, `data.grid` -> background, `data.grid2d` -> MPS; a
cosmolike block -> cosmic shear), so the per-family drivers differ
only in their prog names and defaults.

| Driver | Family | What it does |
|---|---|---|
| train_single_emulator_cosmic_shear.py | cosmic shear + cmb + baosn + mps | train one emulator from a YAML (the family comes from the data block); --diagnostic writes the multipage PDF |
| train_scalar_emulator.py | scalar | train one derived-parameter emulator; --diagnostic |
| sweep_ntrain_emulator_cosmic_shear.py | cosmic shear | f(delta-chi2>thr) vs N_train, multi-GPU pool + gpu-pack |
| sweep_ntrain_scalar_emulator.py | scalar | the same learning curve, serial |
| sweep_ntrain_cmb_emulator.py | cmb | the same, serial |
| sweep_ntrain_baosn_emulator.py | baosn | the same, serial |
| sweep_ntrain_mps_emulator.py | mps | the same, serial |
| tune_single_emulator_cosmic_shear.py | cosmic shear | Optuna study, multi-GPU journal |
| tune_scalar_emulator.py | scalar | Optuna study, serial in-memory |
| tune_cmb_emulator.py | cmb | the same |
| tune_baosn_emulator.py | baosn | the same |
| tune_mps_emulator.py | mps | the same |
| bakeoff_activation_emulator_cosmic_shear.py | cosmic shear | activation bake-off learning curves |
| sweep_hyperparam_emulator_cosmic_shear.py | cosmic shear | one-axis hyperparameter sweeps |

(Renaming the existing cosmic-shear drivers into the namespace is the
recorded POL-1 item — board configs and README references move with
it.)

---

**Also update:** code-map rows (emulator/syren_base.py,
emulator/geometries_grid2d.py, dataset_generator_mps.py,
cobaya_theory/emul_mps.py), Contents anchors, and a short EMUL2
paragraph in the inference section pointing at the three-theory YAML.
