# The background and matter-power output families (BSN + MPS)

Consolidated 2026-07-11 from baosn-emulators.md and mps-emulators.md
(retired; full delta ledgers in git history). User-facing stories are
README sections 16 and 17; the EMUL2 acceptance config is
cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml.

Both families carry the FULL training surface (shared loop: loss
ladder + anneals, trim/focus, EMA, clip/rewind, finetune) and — since
the D-CM13 lift (2026-07-11) — the conv/TRF correction heads: grid =
one bin over z (model.trf.n_tokens re-segments it for attention);
grid2d = one bin per z slice (conv channels / TRF tokens = z slices;
n_tokens rejected, n_heads must divide the post-stride nk). The
identity-basis details live in families-scalar-cmb.md (D-CM13) and
models-and-designs.md.

NPCE rides both since the 2026-07-12 family-wide ruling (residual
only; the base fits the law-space rows — for the boost that is
exactly the user's arXiv 2404.12344 configuration, and
EuclidEmulator2 is a PCE of pklin). A D-MP9 constant pin composes in
decode (the geometry pins the COMBINED base + net prediction). Legs:
check_npce in bsn-identity and mps-identity. Design facts:
models-and-designs.md (the NPCE FAMILY-WIDE bullet).

## BSN — the expansion history. ACCEPTED END TO END (board run 6, 2026-07-11); gates bsn-identity/bsn-smoke.

- THE STALE-BACKGROUND SAGA (board runs 1 + 3, 2026-07-11; the full
  hypothesis-falsified arc, recorded so nobody re-walks it): with
  background-only requirements the legacy hand-rolled
  check_cache_and_compute(cached=True) component loop returned the
  SAME background for every sample (run 1: bitwise-constant H(z)
  columns). Hypothesis 1 — the MPS wants-Cl quirk ("Cl": {tt: 0})
  forces the transfers component to own the cosmology params —
  FALSIFIED by run 3: the quirk was added and bsn-smoke's
  dump-variance tripwire still measured spread exactly 0.0. The fix
  that stands: dataset_generator_background._compute_dvs_from_sample
  evaluates through the STANDARD model.logposterior(point,
  cached=False) lifecycle (cobaya's own parameter routing, every
  component recomputed — the SPE "prefer the programmatic lifecycle"
  lesson), and the Cl requirement is gone again, so the generator
  stays background-only fast (no perturbations). The tripwire
  remains the sentinel: a regression fails AT THE DUMP. NB: the MPS
  generator KEEPS its own wants-Cl quirk verbatim (its requirement
  set is different, its dumps proven varying — mps pklin trained);
  do not harmonize the two evaluation idioms without gate evidence.
- Headline: only H(z) is a network; every distance is IMPOSED physics
  computed from it. The two-regime ruling (D-BSN3-A, the user's call:
  the recombination distance is the discontinuity): (1) an H(z)
  artifact on the SN window z in [0, ~3] — distances inside by
  cumulative Simpson of c/H; (2) a SECOND artifact of D_M trained
  DIRECTLY on the recombination window z in ~[1000, 1200]; (3) the
  legacy analytic z->1200 extension is DEAD (it existed only for want
  of a recombination emulator); (4) getters serve PIECEWISE by query
  redshift and a desert query is a LOUD error naming both windows —
  never a silent bridge.
- GridGeometry (geometries/grid.py): ScalarGeometry math at width NZ
  + a persisted z grid, quantity, units, and a TARGET LAW by name —
  TARGET_LAWS {none, log_offset(offset)}, the law INSIDE
  encode/decode (encode = standardize(log(y + offset))); from_targets
  applies the law FIRST. H trains under log_offset; D_M raw.
- emulator/background.py (single-sourced for the adapter, the
  diagnostics, and scripting): cumulative_simpson VERBATIM from the
  legacy (the recorded FINDING: exact at the even doubled-grid points
  — every served point — but a half-chunk O(dz^3) approximation at
  odd points, ~6.5e-4 on the probe cubic; kept bug-for-bug, only
  shades interpolation); doubled grid linspace(0, z_max, 2NZ+1); flat
  conversions dl = chi(1+z), da = dl/(1+z)^2; C_KMS = 2.99792458e5.
  FLAT ONLY V1: omk is loud. The legacy CURVATURE branch was
  dimensionally WRONG (sinh(chi*K_abs)/K_abs with K_abs =
  |omk|(H0/c)^2; correct is sinh(sqrt(K) chi)/sqrt(K)) and is
  deliberately not reproduced — a corrected branch is a future spec
  with a CAMB-comparison gate leg at omk != 0.
- Generation: dataset_generator_background.py — one background-only
  CAMB pass yields BOTH quantities; z_sn/z_rec specs with the
  no-overlap desert rule; grids written ONCE as _z.npy sidecars
  (training consumes the grid from a FILE).
- emul_baosn: exactly two self-declaring roots (Hubble km/s/Mpc +
  D_M Mpc); window layout persisted, desert-loud at startup; getters
  get_Hubble (km/s/Mpc or 1/Mpc), get_comoving_radial_distance,
  get_angular_diameter_distance (+ the two-redshift variant
  (chi2-chi1)/(1+z2)), get_luminosity_distance. rdrag comes from the
  SCALAR family — a sampling YAML pairs the theories.
- Diagnostics: per-window residual bands with the desert marked + the
  derived-distance page computed through the REAL pipeline
  (pipeline(pred H) vs pipeline(true H)).
- bsn-smoke checks the served values against CAMB'S OWN background at
  an off-center point (2%) — truth available, the strongest
  end-to-end test on the board.
- Never re-propose: the analytic extension; the legacy curvature
  formula; PCA/TMAT compression in V1; porting legacy .pt artifacts
  (retrain); a silent desert bridge. (Transfer over BSN was on this
  list as PERMANENT until the user overturned it, 2026-07-12: both
  BSN and MPS now carry frozen-base transfer — TransferDiagChi2,
  whitened space, sum recommended; details in
  artifacts-inference-warmstart.md.)

## MPS — the matter power spectrum. ACCEPTED END TO END (board run 9, 2026-07-12: rel 0.93% vs CAMB against the 5% bar); gates mps-identity/mps-smoke.

- Headline (D-MP2): the network learns the CORRECTION to an analytic
  formula — target = log(P / P_base); two artifacts: pklin (base =
  the syren linear formula, Mpc^3) and boost (base = syren-halofit,
  B = P_nl/P_lin); P_nl = B * P_lin at serving. Law registry
  TARGET_LAWS_2D {none, syren_linear, syren_halofit} by name;
  law-quantity pairing enforced.
- D-MP2-A (where the cosmology-dependent base lives — the ruling that
  shaped the unit): never inside geometry encode/decode (a per-batch
  GPU sync, or a torch-port drift channel against the very formula
  being corrected). Instead: (1) the GENERATOR writes the base beside
  the raw dump (*_base.npy files) — the consumed base is ON DISK;
  (2) STAGING forms log(raw/base) once, cold (_grid2d_law_rows,
  row-aligned through load_source's dump_rows key; k_stride thinning,
  top edge always kept); (3) ONE base module emulator/syren_base.py
  (base_pklin: k/h conversion, plin_emulated at z=0, approximate
  growth (Dz/D0)^2 (Rz/R0) at kref=1e-4, mnu=0.06, /h^3; base_boost:
  As_to_sigma8 -> run_halofit_vec(return_boost=True, Plin_in=P*h^3));
  syren_params_from is the ONE parameter-mapping rule (As or As_1e9;
  an absent equation of state means LCDM — a model fact) shared by
  generator and adapter; (4) the predictor's grid2d branch returns
  the LAW-SPACE surface — the base multiply-back is emul_mps's job.
- D-MP8 — syren VENDORED: `syren/` holds the two symbolic_pofk
  modules copied from the LEGACY emulmps bundle (the user's own
  edits included — never PyPI); function bodies AST-byte-verbatim;
  only import-line deviations (two dead imports dropped, one
  retargeted), recorded in syren/README.md; numpy-only, so
  syren_base imports UNCONDITIONALLY and the base math runs on the
  Mac. Vendored == original proven byte-identical over 5,433 values.
  Re-vendoring is a deliberate act + RETRAIN.
- Grid2DGeometry (geometries/grid2d.py): two stored axes (z, k), rows
  flattened z-outer; standardize-only in law space; the k downsample
  persists as the stored grid itself. Legacy grids recorded: z = 122
  points (three linspace segments 0-2-10-50), k = logspace(-4, 2,
  2000), extrap_kmax 200.
- D-MP9 (board run 1; AMENDED LAW-AGNOSTIC run 7, 2026-07-11):
  law-space columns constant across the training cosmologies are
  PHYSICS, not a bug — the boost is 1 below the nonlinear scale for
  every cosmology, so its low-k columns are constant under ANY law
  (syren: log(B/B_base) = 0 identically, the base exact; none: the
  raw 1 itself). The first ruling pinned syren laws only and kept a
  law-none error — run 7 falsified that split when the gate's
  deliberate law-none boost training hit the same physics.
  from_targets now PINS constant columns for every law: scale 1,
  decode returns the training constant exactly (= the base under a
  syren law, doubly consistent with emul_mps's k_t blend; = the
  physical value under none), const_mask persisted schema-additively
  (no pins = byte-identical state), one quiet-gated report line at
  build. The dead-dump protection is the WHOLLY-constant-surface
  guard, loud for every law (the bsn stale-generator signature).
  Gate legs in mps-identity (both laws + the dead-dump refusal).
- Generation: dataset_generator_mps.py — the verbatim CAMB quirk
  `"Cl": {"tt": 0}` ("DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB
  WITHOUT WANTS_CL") kept; write_syren_base fails AT SETUP when the
  base cannot be formed. The Pk requirement's k_max is DERIVED
  (2026-07-11): max(2 x k grid top, 20) — the legacy verbatim 200
  IS this formula on the production grid (k top 100), byte-identical
  there, while a small-grid smoke stops paying for k = 200 transfers
  it never reads (the first mps-smoke run burned ~1 h on exactly
  that; the gate's camb_truth mirrors the derived value +
  AccuracyBoost so truth and training data share one convention).
- emul_mps (the EMUL2 provider): PowerSpectrumInterpolator ported
  VERBATIM from the legacy (adapted from CAMB by Antony Lewis,
  attribution kept; proven line-for-line); P_lin = exp(net)*base; the
  boost base is fed the EMULATED P_lin (the legacy flow); the
  verbatim low-k blend k_t = 0.005 1/Mpc, n = 2: boost = 1 +
  (boost-1)(1 - exp(-(k/k_t)^n)); get_Pk_grid / get_Pk_interpolator /
  sigma8. The legacy w0/wa param-padding block is DEAD — the
  artifact's stored input names ARE the contract.
- EMUL2 hybrid inference: cosmolike `use_emulator: 2` consumes
  emulated CAMB products — emul_mps + emul_baosn + the scalar rdrag,
  three theories in one YAML. The unit's ACCEPTANCE experiment is the
  full EMUL2 evaluate run, user-run on the workstation; the shipped
  config is cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml (the legacy
  yaml with the three legacy theory blocks replaced; likelihood/
  params/override kept verbatim so both runs evaluate the same
  point; placeholder artifact roots to fill).
- MPS-DIAG (closed 2026-07-11): grid2d diagnostics pages. THE key
  fact: in law space the residual pred - truth = ln(P_pred/P_truth)
  — THE BASE CANCELS — so the pages read directly as fractional
  error of the served spectrum (law none = plain fractional;
  res_kind names which). grid2d_residual_diagnostic returns the
  (nz, nk) median-|res| surface, per-k bands at first/middle/last z
  (deduplicated for small nz), and the worst-chi2 surface;
  _grid2d_pages = median + worst heatmaps on a SHARED viridis scale
  + stacked per-k band panels with the worst overlay; mps-smoke's
  boost training carries the leg.
- mps-smoke runs the law-NONE path end to end vs CAMB's own P(k,z)
  (5%); the syren assembly is gated by mps-identity's closed-form
  STUB bases — deliberate, so a formula update can never mask an
  assembly bug.
- First-run risks — RESOLVED by the board saga: the Pk_grid
  requirement resolved onto emul_mps cleanly (run 9 green), and the
  Pk_interpolator constraint materialized exactly as recorded —
  cobaya demands >= 4 redshifts for its 2D spline (run 8; the gate's
  truth request now carries an 11-node support containing the
  probes). Both are known quantities for the EMUL2 acceptance.
- Never re-propose: the base inside encode/decode; a torch port of
  the syren formulas; recomputing the base at staging or serving; the
  w0/wa padding block; porting .keras weights (retrain); removing
  the wants-Cl quirk. (Transfer over MPS sat on this list as
  PERMANENT until the 2026-07-12 symmetry ruling overturned it —
  the same overturn recorded in the BSN list above; MPS now carries
  frozen-base transfer via TransferDiagChi2.)
