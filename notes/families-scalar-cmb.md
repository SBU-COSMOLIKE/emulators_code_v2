# The scalar and CMB output families (SPE + CME)

Consolidated 2026-07-11 from scalar-parameter-emulators.md and
cmb-spectra-emulators.md (retired; the full delta ledgers and run
trails are in git history). The family-pattern recipe both units
instantiate lives in project-and-history.md; user-facing stories are
README sections 14 and 15.

## SPE — scalar (derived-parameter) emulators. CLOSED, board 25/25.

- Design: ScalarGeometry (geometries/scalar.py) — names/center/scale
  per-output standardization from training targets; from_targets =
  mean + population std with the RELATIVE zero-variance guard
  (8 * float32-eps * |center|; an absolute guard missed real constants
  whose std is mean-rounding noise ~5e-17). Loss ScalarChi2 overrides
  ONLY chi2 (diagonal unit-variance Mahalanobis). Inputs = covmat
  header names; outputs = named columns of the SAME params .txt via
  data.outputs; no dv files, no cosmolike on the path.
- Staging by NAME through the .paramnames sidecar (REQUIRED): col =
  2 + line_index over all sidecar lines with `*` stripped; name
  uniqueness asserted; check_paramnames pins the sampled block to
  covmat order; sidecar resolution is CHAIN-ROOT-AWARE (X.1.txt pairs
  with X.paramnames — the one REAL library bug a board run caught).
- emul_scalars: get_can_provide_params = union of the artifacts'
  stored output names; generic get_param from a per-point cache;
  provides: in the YAML is a subset-CHECK only, never a source;
  input/provide overlap (chaining) and duplicate outputs are loud;
  wrong-kind guards both ways.
- Driver scalar_train_emulator.py: own run_tag <model>_ntrain<N>, own
  attrs — the forward-walk proved the cs driver could not be reused.
- SPE-FT (fine-tuning): source must rebuild scalar with the outputs
  list equal exactly (names AND order); source geometry pinned
  (epoch-0 bitwise parity); legs ride scalar-identity.
- The SPE lesson bank (each lesson cost one board run; binding on
  every later family): (1) the required-subscript census — every gate
  cfg and example YAML materializes ALL six train_args blocks; fixes
  in configs, never code defaults. (2) generators write .paramnames;
  loaders resolve chain roots. (3) smoke bars must fail a dead
  network: assert OFF the mean at an explicit point, bar BELOW the
  computed mean-predictor. (4) evaluate YAMLs = priors + the evaluate
  sampler's override, never value:-fixed params; force: True.
  (5) evaluate runs write no .paramnames — read the "Derived params:"
  stdout block or the chain header; prefer the programmatic
  get_model + add_requirements lifecycle; ship the self-diagnosis day
  one. (6) honest counts (enumerate the Gate registry) and honest
  margins (predict passed at 4.65% of a 5% bar — recorded, not
  loosened).
- Never re-propose: chaining scalar emulators; provides: as a source;
  the absolute zero-variance guard; porting joblib/GP or .pt legacy
  artifacts (replaced-not-ported: retrain); transfer over scalars
  (PERMANENT — transfer is exclusive to the cosmolike + CMB
  data-vector families; fine-tuning is universal).

## CME — CMB spectra emulators. CODE COMPLETE; gates cmb-identity/cmb-smoke.

- One emulator learns ONE spectrum (tt/te/ee/pp) on l = 2..lmax
  (l = 0,1 are zero-variance whitening poison). CmbDiagonalGeometry:
  standalone per-l vectors {spectrum, ell, center, sigma, fiducial_cl,
  units, law, as_name, tau_name}; deliberately NOT a dense
  DiagonalGeometry (O(n_ell^2) at lmax ~5000).
- THE covinv ruling (authority: Motloch & Hu 1709.03599 eqs 1-7):
  covinv_l = (2l+1) / (2 (Cl_fid + N_l)^2); sigma_l = C_fid *
  sqrt(2/(2l+1)); N^XY_l = Delta^2_XY exp(l(l+1) theta_FWHM^2 /
  (8 ln 2)). Two DEAD forms, never re-propose: the spec's
  2/((2l+1) Cl^2) (Architect mis-transcription) and the legacy
  2/(2l+1) * cl_fid^2 (the VARIANCE misnamed covinv). Whitening by
  the Gaussian sigma makes plain sum-of-squares the Gaussian chi2.
- The amplitude law: AMPLITUDE_LAWS {none, as_exp2tau} persisted BY
  NAME; target = C_ell * exp(2 tau)/A_s; CmbFactoredChi2
  (needs_params) reads a RAW linear As column — tau/logA stay IN the
  whitened input (the input-side factor-geometry pattern was
  rejected for this).
- compute_cmb_covariance.py (D-CM11): Gaussian eq-3 always (all seven
  blocks; noise from Delta muK-arcmin + beam + fsky); the eq-6
  non-Gaussian N^(phi) behind cov_args.nongaussian.enabled (band-
  perturbed re-lensing, ONE Boltzmann solve, 5-point stencil with a
  convergence harness — non-convergence loud). Output .npz: ell,
  sigma_<s> always, cov_<s> dense when NG, cl_<s>, provenance json
  with the exact camb extra_args (the user's verbatim high-accuracy
  block). LCDM-only validation.
- Generation: dataset_generator_cmb.py on the shared core — ONE CAMB
  pass writes four spectra files (never re-run Boltzmann per
  spectrum); phiphi FILLED (legacy zeroed it); get_Cl(ell_factor=
  False, units="muK2") — the same call as the covariance script.
- D-CM8 roughness (loss.roughness {lam, period_cut}, both required;
  absent = byte-identical OFF): a double-boxcar high-pass on the
  whitened RESIDUAL — residual never prediction (a prediction-
  smoothness prior would mimic lensing peak smoothing, the A_L-shaped
  science risk); band-explicit (period_cut ~50 vs the acoustic
  ~200-300; separation >= 4); c_total = c_chi2 + lam*c_rough per
  sample BEFORE the one shared reduction. Phase blocks reject the
  key. A bare second-difference penalty is inadmissible (no band
  edge). Calibrating lam = science thread.
- emul_cmb: spectra/lmax/units are artifact facts; must_provide
  validates every request (never truncate/pad); serves the artifact
  convention only (raw C_ell muK^2, pp dimensionless, zero-padded
  l<2). The predictor's CMB decoder law-dispatches through
  make_cmb_chi2 (single-sourced with training).
- Diagnostics (D-CM9, the family dispatch): the chi2 pages are
  family-generic; CMB adds two pages (per-multipole residual bands
  fractional AND in error-bar units — TE crosses zero, read the
  sigma panel — + the high-pass wiggle content with the acoustic
  band marked).
- Fine-tuning: four loud pin checks (spectrum, law + columns, ell
  grid, covariance file). Transfer for CMB is DEFERRED (D-CM7), not
  permanent — the one family besides cosmolike allowed to get it.
- First-run risks (recorded for the board): the get_model +
  add_requirements path; the generator's first CAMB-only run; serial
  ~400 CAMB calls (lower LMAX/NROWS if slow);
  get_lensed_cls_with_spectrum's call signature.

## SPECS AWAITING AUDIT (written 2026-07-11, deliberately NOT implemented)

Sequencing: both AFTER the first full 32-gate green + the EMUL2
acceptance; D-CM12 first (science value), D-CM13 an optimization
experiment.

**D-CM12 — dense-Cinv training from the non-Gaussian covariance.**
The producing side is DONE (the npz already carries cov_tt/te/ee when
NG is on); training reads only sigma today. Design: `data.cmb.dense:
true` (default false = byte-identical); the validator requires
cov_<spectrum> loudly; build_geometry whitens by the dense block's
eigen-decomposition — law FIRST, then rotation, persisted like the dv
eigenbasis; the LOSS is unchanged (whitened sum of squares IS
r^T Cinv r) — the change lives in the geometry. OPEN RULING for the
user: roughness under a rotated basis (compute it in the PRE-rotation
law basis, or forbid roughness+dense in V1, loudly). Deltas:
D-CM12-1 validator+geometry, D-CM12-2 the roughness ruling, D-CM12-3
gate legs (dense round-trip byte-parity + diagonal-vs-dense
OFF-identity). Risk: NG-block eigenvalue conditioning — clip loudly.

**D-CM13 — conv/TRF heads on the CMB path.** The guard exists because
the heads consume the dv geometry's theta machinery. Design: a
geometry `head_coords()` interface {order permutation, bin coordinate
array, bin sizes} — DataVectorGeometry = the existing theta machinery
byte-identical; CmbDiagonalGeometry = identity permutation,
coordinate = ell, one bin; the guard lifts only when the geometry
implements the interface (errors name geometries, not families).
Deltas D-CM13-1/2/3. Risks: rescale_kernel_size must see n_ell; the
head's zero-init gate must preserve epoch-0 finetune parity. The
physics bet: C_ell is smooth with acoustic structure — the conv
locality prior plausibly fits better than for xi.

Never re-propose (CME): the two dead covinv forms; per-spectrum
Boltzmann re-runs; prediction-side smoothness; bare second-difference
roughness; heads-as-is without D-CM13; the legacy ord/file/extra/
extrapar pattern.
