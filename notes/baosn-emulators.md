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
TWO artifact roots in V1 (D-BSN3-A): the SN-range H(z) emulator and
the recombination-window D_M emulator, each self-describing (grid +
quantity + units + law from its h5; the adapter checks the quantity
tags and window coverage at initialize, loud on a mismatch).
calculate caches the H interpolator + the distance pipeline's outputs
on the state; the getter surface ports from the legacy: get_Hubble
(both unit conventions), get_angular_diameter_distance,
get_comoving_radial_distance, get_luminosity_distance,
get_angular_diameter_distance_2 — each served PIECEWISE per
D-BSN3-A(4). Wrong-kind guards BOTH ways (the
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

### D-BSN3-A — the two-regime domain: the SN range, then recombination
(2026-07-10 amendment, user directive)

**The directive:** "the distance emulator will always need to emulate
around the distance to recombination — it is a discontinuity — you
emulate the SN range and then the distance around recombination."
The emulation DOMAIN is two disjoint windows: the SN/BAO range
(z <~ 3, where the data lives) and the window around recombination
(z* ~ 1090, the CMB-distance anchor). No likelihood queries the
desert between them, so one function across [0, 1200] wastes capacity
and forces the legacy workaround.

**RULING — two trained pieces, one adapter:**
1. The H(z) grid emulator covers the SN range [0, z_max ~ 3]
   (D-BSN1 unchanged); distances INSIDE that window come from the
   Simpson integration of c/H (D-BSN3 unchanged), valid strictly
   within the grid.
2. A SECOND grid emulator covers the comoving distance D_M(z) (or
   chi(z) — one convention, persisted) directly on a small grid
   around recombination, z in ~[1000, 1200], trained from the same
   background CAMB dumps — the network learns the full integral as a
   smooth function of the parameters, so NO bridging integration
   through the desert exists anywhere. The adapter interpolates this
   window to serve D_M(z*(params)) and friends.
3. **The legacy z->1200 analytic extension DIES** (H_ext =
   H0*sqrt(om(1+z)^3 + omegar(1+z)^4), self-labeled "this is an
   approximation" in the legacy source): it existed only because the
   legacy had no recombination emulator. It is not ported.
4. The adapter serves getters PIECEWISE by query redshift: the SN
   window -> integrate-from-H; the recombination window -> the D_M
   emulator; a query in the desert -> a LOUD error naming both
   covered windows (never a silent bridge).
5. The generator writes BOTH dv files from the one background pass
   per sample (H on the SN grid, D_M on the recombination grid) —
   the CME one-pass rule applied here.
6. Gates: bsn-identity gains the desert-query loud-error leg;
   bsn-smoke checks BOTH windows against CAMB's own background
   (H and D_A in the SN range; D_M in the recombination window) at
   the off-center point.

### D-BSN8 — diagnostics pages (2026-07-10, with D-CM9's factoring)

BSN lands its pages on the family dispatch D-CM9 builds: the shared
chi2 pages (history, coverage, floor, hard directions, shaded
triangle) come free; the BSN-specific pages are
- fractional H(z) residual vs z — median + 68/95 bands over the
  validation set, the SN-range panel;
- fractional D_M residual vs z in the recombination window, same
  band style — the two panels TOGETHER on one page with the desert
  marked, so the two-regime coverage is visible at a glance;
- derived-distance validation: D_A and D_L fractional error vs z in
  the SN range, computed through the REAL pipeline
  (emulator/background.py) against CAMB truth from the validation
  dump — this page tests the integration path, not just the network;
- worst-cosmology overlay (highest-chi2 val point, both windows).
Colorblind-safe, never red+green. The smoke gains the cheap
PDF-builds + page-count leg (the D-CM9 pattern).

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

## D-BSN9 — fine-tune support; transfer permanently out
(2026-07-10, user directive)

Fine-tuning must work for BOTH BSN artifacts (the SN-range H(z)
emulator and the recombination-window D_M emulator), each
independently: the SPE-FT pattern ([[scalar-parameter-emulators]])
applied to GridGeometry — source must be a grid artifact of the SAME
quantity, grid, units, and law (loud diff otherwise); the source
geometry pinned through build_warm_start's pinned_geom slot; D-FT3
input extension + the padded-keys anchor mask unchanged; bsn-identity
carries the epoch-0 parity and loud-error legs. Transfer
(`transfer:` / `transfer.refine`) is PERMANENTLY out for this family —
the user's scope ruling: transfer is exclusive to the cosmolike and
CMB data-vector families. (This upgrades D-BSN7's transfer line from
deferral to permanent.)

## Resume state (Implementer appends below)

(not started — queued behind CME)
