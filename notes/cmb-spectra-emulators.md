# CMB spectra emulators (spec)

**Date:** 2026-07-10. **Status:** SPEC (Architect, Fable). **Spec code:**
CME. **Home note** for the gates `cmb-identity` / `cmb-smoke`.
Companion unit to [[scalar-parameter-emulators]] (SPE); both ship in one
implementation pass (user away from the computer; sequencing below).

## The request (user design goal)

Bring CMB spectra emulation (TT / TE / EE / phiphi, the legacy emulcmb)
into this library: the training-set GENERATION (CAMB data vectors), the
training, and a cobaya theory block — replacing the legacy pattern
(per-spectrum file/extra/ord/extrapar lists + a manual eval mask + one
hand-written theory class). "The CMB emulator can have 4 outputs" = four
per-spectrum artifacts served by ONE generic theory block.

**What the legacy training script fixes as the physics conventions
(emultraincmb.py, read 2026-07-10):**

- loss = sqrt of a cosmic-variance-weighted chi2: covinv =
  diag(2/(2l+1) / Cl_fid^2), l = 2..ellmax — OUR existing sqrt loss over
  a DIAGONAL covariance built analytically from a fiducial Cl (no
  cosmolike, no data file);
- the primary amplitude scaling is DIVIDED OUT of the target:
  target' = Cl * exp(2*tau) / As — the network learns the shape, the
  As*exp(-2tau) law is imposed, not learned (the factored-IA philosophy);
- per-spectrum networks; TT/TE/EE are CNN-over-bins (our rescnn), phiphi
  was ResMLP + a PCA output projection.

## Design rules

### D-CM1 — the output geometry: diagonal, from a fiducial Cl

A constructor (new subclass or classmethod beside DiagonalGeometry) that
builds the standard DataVectorGeometry state from analytic pieces: ell
range 2..ellmax, kept = all, Cinv = the cosmic-variance diagonal from a
STORED fiducial Cl, center = the training-mean (of the amplitude-rescaled
target), whitening = the per-ell diagonal scale. Everything downstream
(CosmolikeChi2, the loop, save/rebuild via the cls marker, FTW) reuses
unchanged — the geometry is data, not new machinery. The artifact stores:
spectrum name ("tt"/"te"/"ee"/"pp"), ellmax, the fiducial Cl, the units
convention (raw Cl, muK^2 for T/E; dimensionless for pp), and the
amplitude law (D-CM2).

### D-CM2 — the imposed amplitude law

A small registry, persisted by name in the artifact (never a code
default): `as_exp2tau` (target' = Cl * e^{2 tau} / As; TT/TE/EE) and
`none` (phiphi V1, or any spectrum the user opts out of). The law reads
As/tau from NAMED input columns — reuse the AmplitudeFactorGeometry
pattern (the named columns ride raw, the law is closed-form) or a thin
chi2 wrapper mirroring TemplateFactoredChi2 with one template and a
closed-form coefficient; Implementer proposes the smaller diff, the
identity gate (exact target round-trip through the law) rules either way.
The decode path multiplies the law back, so get_Cl returns physical Cl.

### D-CM3 — training-set generation, in the library

New `compute_data_vectors/compute_cmb_dvs.py`: samples parameters (the
existing generator conventions — covmat + temperature sampling, the
README appendix pattern), evaluates Cl through cobaya's CAMB requirements
(the legacy dummy-likelihood trick, done properly: one cobaya model, Cl
to ellmax, loop the sample), and writes THE SAME dump format the whole
training stack already stages: params .txt (+ covmat header = the input
names) + dv .npy (rows = samples, columns = l=2..ellmax). One dump per
spectrum family run; TT/TE/EE/pp columns written as separate dv files
from one CAMB pass (CAMB gives all four at once — never re-run the
Boltzmann code per spectrum).

### D-CM4 — training: the existing drivers, a cmb data block

train_single (and the sweeps) gain a `data.cmb:` alternative to the
cosmolike keys: `{spectrum: tt, ellmax: 5000, fiducial_cl: <file>,
amplitude_law: as_exp2tau}` — mutually exclusive with cosmolike_data_dir/
dataset, loud both-present error. build_geometry branches to the D-CM1
constructor (no cosmolike import on this path). Architectures: resmlp /
rescnn / restrf as-is (rescnn IS the legacy CNNMLP's role); the phiphi
PCA output projection is OUT OF SCOPE V1 — train phiphi as a plain
design first; if quality demands compression, that is a recorded V2
(NPCE or a PCA head) with the evidence in hand.

### D-CM5 — the cobaya theory block: spectra derived from the files

`cobaya_theory/emul_cmb.py`, the emul_scalars pattern applied to Cl:

```yaml
theory:
  emul_cmb:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    extra_args:
      device: 'cuda'
      emulators:
        - projects/cmb/emulators/tt/emul_v2
        - projects/cmb/emulators/te/emul_v2
        - projects/cmb/emulators/ee/emul_v2
        - projects/cmb/emulators/pp/emul_v2
```

- WHICH spectra + each ellmax: read from the artifacts. The legacy
  `eval:` mask, `ord`, `file`, `extra`, `extrapar` all die.
- `get_can_support_params` / requirements = the union of stored input
  names; two artifacts with the same spectrum = loud error; a likelihood
  requesting a spectrum no artifact provides = loud error at
  must_provide, naming the loaded spectra.
- get_Cl assembles the cobaya Cl dict (ell array + per-spectrum arrays,
  zero-padded l=0,1; units per the artifact convention) from batch-1
  decodes, amplitude law multiplied back.

### D-CM6 — gates

- `cmb-identity` (CME-A; Mac+board, torch only): synthetic fiducial Cl ->
  geometry build + state round-trip byte-identical; the amplitude law
  exact both ways (encode(decode) == identity bitwise on the law's
  closed form); save -> rebuild -> predict bitwise (same-path); get_Cl
  assembly from two synthetic artifacts; the duplicate-spectrum and
  missing-spectrum errors.
- `cmb-smoke` (CME-B; board; needs camb+cobaya): END-TO-END — the D-CM3
  generator makes a TINY dump (e.g. 300 rows, ellmax 512) through real
  CAMB, train_single trains 2 epochs on it (tt, as_exp2tau), a cobaya
  evaluate run through emul_cmb returns Cl at a test point (the
  cobaya-adapter gate pattern). This smoke also gates the generator —
  the piece the user asked to "include in the library".

### D-CM7 — out of scope (recorded)

The phiphi PCA/TMAT output compression (V2, evidence-gated per D-CM4);
CosmoRec/recombination variants (the legacy comment "not trained with
CosmoRec" is a training-data property, not code); the legacy emulbaosn
(H(z) integrator) — its own future unit; TPE transfer over CMB emulators
(possible later — the geometry is standard — but not V1).

## Sequencing (both units, one implementation pass)

SPE first (smaller, establishes the artifact-derived provides pattern and
the EmulatorPredictor dispatch), then CME (reuses both). Each unit gets
its own commit and its own gates; the board grows by four (scalar-identity,
scalar-smoke, cmb-identity, cmb-smoke).

## Links

[[scalar-parameter-emulators]], [[finetune-warm-start]],
[[gates-harness-user-run]], [[py-module-style-conventions]],
[[docs-plain-language-define-or-drop]].

## Resume state (Implementer appends below)

(none yet)
