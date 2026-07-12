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
  (D-SP8 — the one family transfer does NOT ride after the 2026-07-12
  symmetry lift; a recorded ruling the user may overturn, not a
  structural bar; fine-tuning is universal).

## CME — CMB spectra emulators. ACCEPTED END TO END (board run 4, 2026-07-11); gates cmb-identity/cmb-smoke.

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
- D-CM11 EXTENDED 2026-07-12 (user overnight ask "the nondiagonal
  terms from Wayne and Pavel"): eq 6 now assembles EVERY spectrum
  pair — the cross blocks cov_tt_te / cov_tt_ee / cov_te_ee join the
  per-spectrum three (assemble_lensing_blocks, a pure function the
  Mac probe checks against D_a^T diag(S) D_b), each carrying its
  eq-3 Gaussian l-diagonal; together the six tile the full joint
  TT/TE/EE covariance a D-CM12 dense whitening or a joint likelihood
  consumes. The capability was already in the script behind the flag
  (the Gaussian-first directive) — what was missing was the cross
  pairs, gate execution, and visibility. cmb-smoke gained leg 2b
  (check_cov_nondiagonal): the NG path runs END TO END at smoke
  scale (16 re-lensings) and must produce symmetric, PSD, off-
  diagonal-alive blocks with the step study in the provenance —
  the first real execution of eq 6 anywhere. That first execution
  (board run 11) was RED: the hand-built clpp array stopped at
  lens_lmax while CAMB demands Params.max_l length — fixed by taking
  the fiducial array whole from get_lens_potential_cls (which also
  stopped a silent delensing above lens_lmax); rerun pending
  (gates-and-board.md run 11). The red-team static audit then caught
  a normalization defect in the contraction itself — D-CM11-A below,
  the next Implementer unit.
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
  grid, covariance file). Transfer for CMB is IMPLEMENTED since
  2026-07-12 (D-CM7's deferral closed by the symmetry ruling):
  TransferDiagChi2, whitened space, amplitude_law "none" both sides,
  the same four whitening pins as the finetune path; details in
  artifacts-inference-warmstart.md, legs in transfer-identity.
- First-run risks — ALL RESOLVED by the board saga (run 4 green):
  the get_model + add_requirements path and the generator's CAMB run
  worked as shipped; the ~400 serial CAMB calls cost ~10 min at
  AccuracyBoost 0.7. The two failures the first runs DID hit were
  gate-fixture conventions of the covariance script (plain-number
  params, run 1; the script's OWN omegabh2/omegach2 names, run 3) —
  the fixture now mirrors example_yamls/cmb_covariance_lcdm.yaml
  exactly, and the lesson is recorded in conventions-and-workflow.md.

## D-CM11-A — the eq-6 contraction is mis-normalized (red-team catch, 2026-07-12; THE NEXT IMPLEMENTER UNIT)

- The defect (independently re-derived by the Architect from eq 6 and
  the code — the red team is right): nongaussian_blocks perturbs a
  band by a dimensionless FRACTIONAL amplitude (clpp *= 1 + eps), so
  the stencil returns, at band_width 1,
  D_lL = dC_l/dA_L = C^pp_L * (dC_l/dC^pp_L). Substituting into eq 6
  cancels the C_L^2 inside Cov^pp_LL = 2 C_L^2/((2L+1) fsky):

      N_ll' = sum_L  D_lL * [2/((2L+1) fsky)] * D_l'L

  i.e. the correct weight is the Gaussian variance of the FRACTIONAL
  amplitude, Var(A_L) = Var(C_L)/C_L^2 = 2/((2L+1) fsky). The shipped
  code contracts with S_b = sum_L 2 C_L^2/((2L+1) fsky) — an extra
  C_L^2 factor: wrong dimensions, wrong scale (tens of orders low at
  CMB C^pp values). Convention-invariant: scaling the
  [L(L+1)]^2/2pi-scaled array by (1+eps) scales raw C^pp by the same
  factor, so A_L and the fix are the same in either convention.
- Why every existing check missed it: the smoke leg proves symmetry /
  PSD / off-diagonal liveness / stencil convergence — ALL invariant
  under any positive diagonal reweighting — and the Mac probe
  validated assemble_lensing_blocks against the Architect's own
  D^T diag(S) D spec, i.e. against the same wrong algebra. Only an
  oracle INDEPENDENT of the spec author's contraction can catch a
  normalization error (lesson also in conventions-and-workflow.md).
- Containment: the eq-6 path has NEVER completed a run (board run 11
  crashed on the clpp length before writing output). No .npz with
  cov_* blocks exists anywhere; training reads only sigma_<s>. Zero
  science impact — a pre-first-light fix.
- RULING (wide bands): band_width stays as the cost knob, as a
  DOCUMENTED approximation with the projected weight

      w_b = [sum_{L in b} 2 C_L^2/((2L+1) fsky)] / [sum_{L in b} C_L]^2

  valid when dC_l/dC^pp_L is close to constant across the band (the
  smooth-response assumption; at band_width 1 this degenerates to the
  exact 2/((2L+1) fsky) — eq 6 verbatim). A band with
  sum_{L in b} C_L = 0 contributes nothing (its fractional derivative
  is identically zero): w_b = 0 with a comment, never a division.
- Implementer contract: (1) contract deriv with w_b as above;
  (2) the Gaussian outputs stay BITWISE (the fix touches only the NG
  weights); (3) provenance gains the derivative coordinate
  ("fractional_band_amplitude"), the band policy ("exact eq 6" at
  width 1, "smooth-response band projection" wider), and the per-band
  weights, persisted; (4) nothing produced by the old normalization
  is ever labeled an eq-6 covariance (none exists; the provenance
  keys are the forward guard).
- The oracle gate (new legs in cmb-identity — torch-only, no CAMB): a
  fake CAMBdata whose get_lensed_cls_with_spectrum is an AFFINE map,
  lensed_s = base_s + M_s @ clpp (seeded fixed M per spectrum, tiny
  lens_lmax ~ 12), so dC_l/dclpp_L = M_s[l, L] exactly and the
  5-point stencil is exact to roundoff. Three legs: (a) TRUTH — eq 6
  computed directly from M and Var(C_L), never through the pipeline's
  contraction; the REAL nongaussian_blocks on the fake must match at
  rtol ~1e-9; (b) DISCRIMINATION — the old extra-C_L^2 weights
  applied to the same derivatives must miss that truth by orders of
  magnitude (an oracle the old code passes is a defective oracle);
  (c) BAND — band_width 3 with M built constant across each band must
  match truth exactly (w_b proven on its domain of validity).
  cmb-smoke leg 2b keeps the structural checks (symmetry, PSD,
  liveness, convergence, the provenance study) on real CAMB, plus
  asserts the new provenance keys.

## D-CM12 — SPEC AWAITING AUDIT (written 2026-07-11, NOT implemented; the PRODUCING side is BLOCKED ON D-CM11-A)

Sequencing: AFTER the first full 32-gate green + the EMUL2 acceptance.

**D-CM12 — dense-Cinv training from the non-Gaussian covariance.**
The producing side is DONE (the npz carries cov_tt/te/ee AND, since
the 2026-07-12 extension, the cross blocks cov_tt_te/tt_ee/te_ee when
NG is on — gate-executed by cmb-smoke leg 2b); training reads only
sigma today. Design: `data.cmb.dense:
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
NB: a D-CM12 dense CMB geometry would carry an eigenbasis — the heads
would then need the REAL basis change, exactly the cosmolike path;
revisit the D-CM13 identity shortcut when auditing this.

## D-CM13 — IMPLEMENTED 2026-07-11 (user order, generalized past CMB)

The user ordered the capability symmetry the same evening the spec was
written ("I want that for CMB and MPS minimum — I prefer that they all
have"), citing arXiv 2505.22574 (attention-based CMB-spectrum
emulators, Part III of the multi-probe series: dot-product attention
cuts the outlier count vs plain MLPs), which made D-CM13
science-motivated rather than an optimization experiment. This
supersedes the "after board + EMUL2" sequencing for this one item.

What shipped (simpler than the spec — the identity insight):
- The spec's head_coords() interface collapsed: the diagonal family
  geometries whiten per element IN physical order, so the trunk
  already predicts in the head's local basis — no permutation, no
  basis change. ResCNN / ResTRF keep W_fd / W_df as None when the
  geometry has no eigenbasis (hasattr evecs) and skip both matmuls
  (never build n_keep x n_keep identities). Cosmic shear byte-safe:
  its geometry has evecs, the old path is untouched.
- The split attach is `attach_head_coords()` on the geometry (pure,
  idempotent, no files): cmb = one bin, coordinate ell; grid = one
  bin, coordinate z; grid2d = one bin PER Z SLICE of length nk
  (z-outer flattening: conv channels / TRF tokens = z slices — the
  physically right mapping). Called in build_geometry (fresh AND
  finetune-pin paths) and in results._rebuild_model (rebuild works
  from the files alone; the split is derived, never persisted).
- `model.trf.n_tokens` (MODEL_BLOCK_KEYS + ResTRF kwarg, recipe-
  recorded): re-segments a SINGLE-bin geometry into contiguous
  near-equal windows so attention has tokens (the paper's
  tokenization, minus embeddings); loud errors on multi-bin
  geometries (physical bins ARE the tokens) and out-of-range T.
  n_heads must divide ceil(n / n_tokens) (TRFBlock's assert).
- The from_config guards lifted for cmb / grid / grid2d with the
  cs-style head-pin notice resolution; SCALAR stays trunk-only
  (named outputs have no coordinate axis) with the reworded error.
- Two-phase (SUPERSEDED 2026-07-12, user ruling "any trunk-head
  design could benefit"): plain ResCNN/ResTRF now define
  set_train_phase, mirroring the IA-template contract exactly
  (joint/trunk/head requires_grad groups; the trunk phase bypasses
  the zero-init head at pure-ResMLP cost; the head phase runs the
  frozen trunk under no_grad) — trunk_epochs / freeze_trunk / the
  trunk:/head: phase blocks now work on every family the heads
  ride, and the per-head activation pin (model.cnn/.trf.activation,
  licensed by a frozen-trunk head phase) is reachable everywhere.
  Phase-discipline legs ride the cmb/mps-identity head checks.
- Gate legs (no board-count change): cmb-identity check_head (ResTRF
  + n_tokens: attach, identity basis, epoch-0 identity, range error,
  save->rebuild->predict bitwise) and mps-identity check_head
  (ResCNN on z-slice channels + the n_tokens-on-real-bins rejection
  + the bitwise round-trip). The round-trip legs specifically prove
  the rebuild-side attach.
- NPCE rides both families since the 2026-07-12 family-wide ruling
  (scalar included — the PCE trunk needs no coordinate axis, so the
  heads-on-scalar exclusion does NOT extend to it): residual-only,
  and on cmb only with amplitude_law "none" (the imposed law and the
  base each replace the target construction — validate_cmb is loud).
  Roughness composes on a cmb NPCE run (the penalty sees the full
  whitened residual). Legs: check_npce in scalar-identity and
  cmb-identity (algebra bitwise + save->rebuild->predict composing
  base + net + the exclusivity raises). Design facts:
  models-and-designs.md (the NPCE FAMILY-WIDE bullet).
- DISCOVERED IN PASSING and FIXED the same evening (the follow-up
  commit): the COSMIC-SHEAR head artifacts could not rebuild
  (build_shear_angle_map is never called on the rebuild path, and
  DataVectorGeometry.state() did not persist bin_sizes). Fix =
  schema-additive persistence, the section_sizes/probe pattern:
  state() writes bin_sizes (+ pm_kept) when the attach ran; __init__
  gained the optional kwargs, attribute-UNSET when None so the
  hasattr guards survive; results._rebuild_model refuses a
  pre-persistence head file loudly ("bin-split persistence"), never
  re-derives (that would need ROOTDIR data files at inference).
  Gate: save-rebuild-drift gained a rescnn head variant (real
  training path, bitwise round-trip) + a deleted-split refusal leg
  — it was GREEN on the 25/25 board, so it needs --force-rerun.

Never re-propose (CME): the two dead covinv forms; per-spectrum
Boltzmann re-runs; prediction-side smoothness; bare second-difference
roughness; the legacy ord/file/extra/extrapar pattern; heads on the
SCALAR family (no coordinate axis).
