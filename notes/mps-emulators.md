# Matter power spectrum emulators + hybrid inference (spec)

**Date:** 2026-07-11. **Status:** SPEC (Architect) — QUEUED as a major
unit after BSN (it consumes BSN's GridGeometry pattern + the D-CM3-A
generator core). **Spec code:** MPS. Home note for the gates
`mps-identity` / `mps-smoke`. Assigned by the user "all that is you to
implement" (Architect implements, the overnight-mode precedent).

## The request (user, 2026-07-11)

Port the legacy MPS emulator (Downloads/emulators_code-main/emulmps/,
580-line cobaya Theory over keras models; structure read):
1. **It corrects an approximate formula** — the legacy base is the
   symbolic_pofk syren-halofit family; the true emulator learns the
   CORRECTION to that formula (the base+refiner philosophy with an
   ANALYTIC physics base).
2. **TensorFlow -> PyTorch**: rewrite in the v2 stack; the .keras +
   metadata.joblib artifacts stay on the legacy classes
   (replaced-not-ported, D-SP6 — RETRAIN via the new generator).
3. **The grids are fixed** (the legacy generator, read): z = concat of
   linspace(0,2,100,endpoint=False) + linspace(2,10,10,endpoint=False)
   + linspace(10,50,12) = 122 redshifts; k = logspace(-4, 2, 2000).
   TWO quantities: linear P(k,z), and the nonlinear BOOST
   B(k,z) = P_nl/P_lin (the ratio is the emulated target).
4. **dataset_generator_mps.py** is primitive (ad-hoc MPI, LHS+uniform,
   no checkpoints) — rewrite it to the D-CM3-A generator-core level as
   the FOURTH thin driver (no duplication).
5. **EMUL2 = hybrid inference** (EXAMPLE_EMUL2_EVALUATE1.yaml, saved
   by the user): the cosmolike likelihood runs with use_emulator: 2,
   consuming EMULATED CAMB PRODUCTS — Pk from emul_mps, distances/H
   from emul_baosn, rdrag from the SPE emulator — instead of a full
   data-vector emulator. The v2 emul_mps must provide the
   Pk_interpolator products that mode consumes.

## Design rules

- **D-MP1 — outputs = two grid functions.** A 2D GridGeometry over
  (z, k) — the BSN GridGeometry generalized to two stored axes,
  flattened row-major for the network; grids/units/law persisted
  in-artifact (never the legacy sidecar bundles). Two artifacts:
  "pklin" and "boost". Output widths are large (122 x 2000 raw) —
  V1 trains on a DOWNSAMPLED k grid (a knob, persisted; the adapter
  interpolates, exactly as CAMB's own interpolator does); PCA output
  compression stays the recorded evidence-gated V2 (the phiphi
  ruling).
- **D-MP2 — the imposed formula base (the headline).** The target is
  the correction to the analytic base: target = log(P / P_base(k,z,θ))
  for pklin (base = the syren linear formula) and log(B / B_base) for
  the boost (base = syren-halofit), through the LAW-REGISTRY pattern
  (D-CM2/BSN): law names "syren_linear" / "syren_halofit" / "none",
  persisted by name; decode multiplies the base back. The base
  formulas come from the symbolic_pofk package (a dependency the
  legacy already carries); READ legacy emulmps.py lines 27-66 before
  implementing (the import fallbacks + run_halofit conventions are
  binding).
- **D-MP3 — the generator.** dataset_generator_mps.py, the fourth
  thin driver on generator_core: per sample one CAMB-in-cobaya call
  through the Pk_interpolator requirements (BINDING legacy quirk,
  keep the comment: a Cl requirement must ride along — "DONT REMOVE
  THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL"); writes TWO dv
  files from one pass (pklin rows + boost rows on the persisted
  grids); params/.paramnames/.covmat from the core. extrap_kmax=200
  and the mead2020 halofit choice ride the training YAML, persisted.
- **D-MP4 — the cobaya adapter (hybrid mode).** cobaya_theory/
  emul_mps.py on the emul_scalars template: artifact-derived
  requirements; provides Pk_interpolator (linear + nonlinear via
  P_nl = P_lin * B) through a CAMB-compatible RectBivariateSpline
  interpolator — PORT the legacy PowerSpectrumInterpolator class
  (lines 68-198, itself adapted from CAMB/Antony Lewis; keep the
  attribution). Must satisfy the EMUL2 YAML's consumer set: the
  cosmolike likelihood at use_emulator: 2 + emul_baosn + the rdrag
  emulator, all in one run. The wrong-kind guards both ways.
- **D-MP5 — drivers + THE NAMESPACE RULING (applies program-wide).**
  New drivers follow `<verb>_<family>_emulator.py`
  (train_/tune_/sweep_ntrain_ x cmb/baosn/mps/scalar); commissioned
  NOW by the user: chi2-vs-N_train sweep drivers AND optuna tune
  drivers for SN/BAO (baosn), CMB, and MPS — each landing WITH or
  right after its family unit, built as thin wrappers over the
  existing sweep_ntrain/tune machinery (family = the from_config
  branch, so the wrappers are mostly config plumbing). The README
  gains a "Drivers" subsection with a table: file, family, verb, what
  it produces (the user: "some prefix and suffix so it is clear this
  is about CMB or MPS"). Renaming the EXISTING cosmic-shear drivers
  = a POL-1 item (board configs + README references move with it),
  recorded there.
- **D-MP6 — gates.** mps-identity (grid geometry 2D round-trip +
  law-exactness on synthetic formulas + save/rebuild/predict bitwise
  + wrong-kind/error legs, torch-only via the stub pattern);
  mps-smoke (tiny generated dump through real CAMB, 2-epoch train of
  the boost, a cobaya evaluate through emul_mps in a MINIMAL hybrid
  YAML — the full EMUL2 cosmolike run is the acceptance experiment,
  not the smoke). The full SPE lesson bank binds (subscript census,
  dead-network bars off-fiducial, priors+override evaluate, stdout
  readback, diag from day one).
- **D-MP7 — out of scope (recorded).** PCA output heads (V2,
  evidence-gated); transfer (PERMANENT forbid — the scope ruling:
  transfer = cosmolike + CMB dv only); the legacy .keras weights
  (retrain); syren-formula refits. Fine-tuning IS in scope (the
  universal rule) via the grid-geometry pin (the SPE-FT/BSN pattern).

## Sequencing (the full queue after this commission)

CME (increments 3-5 remain) -> SPE-FT (small) -> BSN -> MPS (this) ->
GEO -> POL. The per-family tune/sweep drivers (D-MP5) land with their
families: CME's pair right after CME closes, BSN's and MPS's in-unit;
the scalar pair rides SPE-FT. Hybrid-inference acceptance (the EMUL2
YAML end to end) closes the MPS unit. Board: +2 gates per family.

## Links

[[cmb-spectra-emulators]] (law registry, D-CM3-A core, D-CM11),
[[baosn-emulators]] (GridGeometry), [[scalar-parameter-emulators]]
(adapter template + lesson bank), [[post-program-polish]] (driver
renames), [[geometry-family-folder]].

## Resume state (appended below by whoever implements)

(not started — queued after BSN)
