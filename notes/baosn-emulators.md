# BAOSN background emulators (spec)

**Date:** 2026-07-10. **Status:** SPEC (Architect, Fable) — QUEUED as
unit 3 of the emulator-output program, after CME closes. **Spec code:**
BSN. **Home note** for the gates `bsn-identity` / `bsn-smoke`.

## The request (user design goal)

Replace the legacy emulbaosn (Downloads/emulators_code-main/emulbaosn,
v1 theory + v2 direct class, both read 2026-07-10): a background
emulator serving H(z) and the distances (comoving, angular-diameter,
luminosity) to BAO and SN likelihoods — "distance and H(z) emulator as
function of redshift". The legacy design's key insight KEEPS: only
H(z) is a network; every distance is imposed physics computed from it.

**What the legacy fixes as conventions (verbatim unless noted):**

- H(z): ResMLP -> PCA (TMAT) -> destandardize -> `exp(y) - offset`
  on a stored z grid (ZLIN .npy sidecar); inputs `ord` (LCDM profile:
  omegam, H0); offset from extrapar.
- Distances (the "INT" MLA — no network): c/H cubic-interpolated;
  comoving chi by CUMULATIVE SIMPSON on the doubled grid
  `zstep = linspace(0, zmax, 2*NZ+1)` (odd point count enforced);
  HIGH-Z EXTENSION from the grid's zmax to z = 1200 (NZEXT 4501) with
  the matter+radiation approximation
  `H_ext = H0*sqrt(omegam*(1+z)^3 + omegar*(1+z)^4)`,
  `omegar = 3.612711417813115e-05/h^2` (verbatim constant), chained
  `chi_ext + chi[-1]`; flat conversions `dl = chi*(1+z)`,
  `da = dl/(1+z)^2`, `dc = da*(1+z)`.
- Cobaya surface: calculate caches INTERPOLATORS on the state
  (H_interp, da_interp); getters get_Hubble(z, units km/s/Mpc |
  1/Mpc), get_angular_diameter_distance, get_comoving_radial_distance,
  get_luminosity_distance, get_angular_diameter_distance_2 (zpair).

**Legacy finding (do NOT port verbatim):** the curvature branch is
dimensionally suspect — it computes `sinh(chi*K_abs)/K_abs` with
`K_abs = |omk|*(H0/c)^2` (units 1/length^2), where the correct form is
`sinh(sqrt(K_abs)*chi)/sqrt(K_abs)`. It agrees with flat only because
sinh(x) ~ x at small argument. Recorded; V1 is flat-only (D-BSN3).

**Legacy quirk (drop):** the v1 theory's get_requirements added
`rdrag: None` — plumbing convenience, not a physics need of this
theory. In v2 rdrag comes from the SPE rdrag emulator (emul_scalars);
a sampling YAML pairs the two theories.

## Design rules

### D-BSN1 — the output geometry: a function on a stored z grid

H(z) is a vector over a persisted z grid — a `GridGeometry` beside
ScalarGeometry: per-component `center`/`scale` from the training
targets (the ScalarGeometry math at width NZ), PLUS the stored grid
`z`, the quantity name ("Hubble"), the units ("km/s/Mpc"), and the
TARGET LAW by name from a small registry (the D-CM2 pattern):
`log_offset` (target' = log(H + offset), decode = exp(y) - offset;
offset persisted — the legacy convention) and `none`. state()/
from_state + the cls marker; save/rebuild generalize as SPE proved.
No ZLIN/TMAT sidecars: the grid and the law live IN the artifact
(never-trust-defaults). PCA/TMAT output compression is OUT (V2,
evidence-gated — the phiphi ruling applied here; NZ ~ 600 outputs is
plain-ResMLP territory). Loss: ScalarChi2 reused unchanged (it reads
only encode/decode/dest_idx off the geometry).

### D-BSN2 — training set: the generator core's third driver

`compute_data_vectors/dataset_generator_background.py`, the third thin
driver on the D-CM3-A generator_core: per sample, a background-only
CAMB evaluation (cheap — no perturbations) writes H(z_grid) as one 2D
dv file; params/.paramnames/.covmat/.ranges come from the core (the
sidecar conventions by construction). The z grid is a train_args key,
persisted into the artifact at training.

### D-BSN3 — distances are imposed physics, single-sourced

One module, `emulator/background.py`, owns the H(z) -> distances
pipeline, used by BOTH the cobaya adapter and direct scripting (the
single-source rule that kept SPE's predictor honest):

- verbatim numerics: the cumulative-Simpson rule (odd-point guard),
  the doubled zstep grid, the z->1200 extension with the H_ext
  approximation and its omegar constant, the chained chi, the flat
  dl/da/dc conversions;
- the extension reads H0 and omegam from the sampled point, so those
  two names must be inputs of the H(z) artifact — a loud initialize
  check, not an assumption;
- FLAT ONLY in V1: an `omk` among the inputs (or requested) is a loud
  error citing this rule; the CORRECTED curvature branch
  (sqrt(K) in both places) is a recorded future item that lands with
  a CAMB-comparison gate leg at omk != 0, never the legacy formula.

### D-BSN4 — the cobaya adapter: emul_baosn on the emul_scalars template

`cobaya_theory/emul_baosn.py`: the whitelist / _pick_device /
ROOTDIR-relative roots / artifact-derived requirements pattern;
ONE H(z) artifact root (V1: exactly one; a list is future-proofing the
YAML shape only). calculate caches the H interpolator + the distance
pipeline's outputs on the state; the getter surface ports from the
legacy: get_Hubble (both unit conventions), get_angular_diameter_
distance, get_comoving_radial_distance, get_luminosity_distance,
get_angular_diameter_distance_2. Wrong-kind guards BOTH ways (the
D-SPE2-4 lesson): emul_baosn rejects a non-GridGeometry artifact
loudly, and the ScalarGeometry/dv adapters already reject a
GridGeometry one by dispatch. No rdrag requirement (see the quirk
above).

### D-BSN5 — direct scripting

EmulatorPredictor grows the grid branch by geometry dispatch:
`predict(dict) -> {"z": grid, "H": array}`; the distance functions
come from `emulator/background.py` applied to that output (one
pipeline, two doors — README appendix 20's pattern extends with a BSN
example when the unit lands).

### D-BSN6 — gates

- `bsn-identity` (BSN-A; torch only): GridGeometry + law round-trip
  bitwise (encode(decode) exact on log_offset's closed form);
  save -> rebuild -> predict bitwise; the Simpson integrator against
  an analytic antiderivative (pure numpy — Mac-checkable); the
  distance pipeline against the closed-form LCDM integral at
  tolerance; the error legs (wrong-kind both ways, omk loud, missing
  H0/omegam inputs loud); the stubbed-adapter provides/requirements
  legs (the SPE stub pattern).
- `bsn-smoke` (BSN-B; board, torch + CAMB + cobaya): the generator
  makes a tiny background dump through real CAMB; 2-epoch train; a
  cobaya evaluate through emul_baosn returns H(z) and D_A(z) at an
  off-center override point, checked against CAMB's OWN background at
  that point (truth is available here — the strongest smoke of the
  program); bars below the dead-network baseline (the D-SPE2-5 rule);
  the evaluate YAML in the proven priors+override shape; readback from
  stdout/chain header; the diag from day one (the full SPE lesson
  bank).

### D-BSN7 — out of scope (recorded)

PCA/TMAT output compression (V2, evidence-gated); curvature (V1
flat-only; corrected-formula future item with its CAMB leg); multiple
H(z) artifacts / model mixing; TPE transfer over background emulators;
the legacy .pt/.npy artifacts (replaced-not-ported, the D-SP6 trade —
retrain via the new generator + driver).

## Sequencing

Unit 3 of the pass: AFTER CME closes. BSN stacks on CME's foundations
(the D-CM3-A generator core, the target-law registry, the from-grid
geometry pattern) — building it earlier would duplicate in-flight
work. Board grows 27 -> 29 when BSN lands.

## Links

[[cmb-spectra-emulators]] (the law registry + generator core),
[[scalar-parameter-emulators]] (the adapter template + lesson bank),
[[gates-harness-user-run]], [[py-module-style-conventions]].

## Resume state (Implementer appends below)

(not started — queued behind CME)
