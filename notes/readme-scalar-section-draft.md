# README scalar-section draft (ready-to-apply block)

Drafted against `origin/main`'s rewritten README (the branch diverged on
README.md only, so this is applied AFTER `git merge origin/main` into the
branch, per the integration order). It is a new numbered section, not an
appendix, because a scalar emulator is a training capability.

## How to apply

1. Insert the section below immediately after section 13 (right before the
   `---` and `## 14. Appendix: the pipeline` line, at ~README.md:1633 on
   origin/main).
2. Renumber the appendices that follow: `14 -> 15` (the pipeline), `15 -> 16`
   (chi2), `16 -> 17` (activation functions), `17 -> 18` (precedence),
   `18 -> 19` (generating the training set), `19 -> 20` (AI-Usage, stays
   last). Bump every in-text cross-reference to those numbers.
3. Add a Contents row for the new section 14 (and shift the appendix rows to
   15-20).

The section itself is byte-ready; the renumbering is the only manual part.

---

## 14. Scalar (derived-parameter) emulators

The emulator above maps cosmological parameters to a cosmic-shear DATA
VECTOR. A scalar emulator instead maps them to a small set of NAMED derived
parameters — H0, omegam, rdrag — one number each. The classic use lets a
sampler walk a fast variable while the slow map runs as an emulator: sample
the acoustic scale thetastar and emulate (omegabh2, omegach2, thetastar) ->
(H0, omegam), so cosmolike, which needs H0, keeps running. A separate driver,
`train_scalar_emulator.py`, trains one; there is no data vector, no mask, and
no cosmolike anywhere on this path.

```
sampled parameters (rescaled, section 2)
     │
     ▼  resmlp trunk            the same architecture as section 10,
     │                          just narrower — a scalar map is easy
standardized outputs
     │
     ▼  undo the standardization
H0, omegam, ...                 one physical number each, served by name
```

**Inputs and outputs are both columns of one parameter file.** The inputs
are the covmat-header names, rescaled into the network's training units
exactly as on a data-vector run (section 2). The outputs are the columns you
name in `data.outputs`, each standardized — shifted to zero mean and scaled
to unit variance. The outputs live on wildly different scales, H0 near 70
and omegam near 0.3, and standardizing puts each on the same footing before
the network sees it. The `.txt` needs its getdist `.paramnames` sidecar
beside it, since the outputs are usually derived columns located by name.

```yaml
data:
  train_params: chain_thetastar_lcdm.1.txt    # needs a .paramnames sidecar
  train_covmat: chain_thetastar_lcdm.covmat    # header = the input names
  val_params:   chain_thetastar_lcdm_val.1.txt
  outputs:                                     # the derived columns to emulate
    - H0
    - omegam
  n_train:    100000
  n_val:      20000
  split_seed: 0
```

The model is a plain trunk (`name: resmlp`); the conv and transformer heads
correct along an angular axis a scalar output does not have, so `rescnn` /
`restrf` are a loud error here. Everything else — the loss ladder, trimming,
focus, EMA, the L2-SP anchor — works unchanged, since they act on a
per-sample error. A scalar map is cheap, so small widths and a few hundred
epochs are plenty. The physical-window `param_cuts` are optional on this
path, because a parameter chain is already the target distribution.

**In an MCMC, the theory block reads what it provides from the file.** One
generic class, `emul_scalars`, serves any scalar emulator: it lists each
saved emulator's path root and nothing else — the required inputs and the
provided outputs both come from the `.h5`, never a hand-typed list. Point it
at several roots and it provides their union. Three misconfigurations are
loud errors at startup:

| the error | why |
|---|---|
| two emulators provide the same output name | each derived parameter must come from exactly one emulator |
| one emulator's output is another emulator's input | chaining scalar emulators is out of scope |
| a data-vector emulator in the list | `emul_scalars` serves scalar artifacts only — a data-vector emulator belongs in `emul_cosmic_shear`'s list |

```yaml
theory:
  emul_scalars:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    extra_args:
      device: cuda
      emulators:
        - projects/lsst_y1/emulators/thetaH0/emul_v2
        - projects/lsst_y1/emulators/rdrag/emul_v2
```

Full example: `example_yamls/scalar_emulator.yaml`. Design record:
`notes/scalar-parameter-emulators.md`.
