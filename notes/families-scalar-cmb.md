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

## D-CM11-A — eq-6 normalization implemented and Mac-audited; fixture/workstation close pending

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
  independent known-answer calculation, separate from the spec author's
  contraction, can catch a
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
- The independent known-answer gate (new legs in cmb-identity — torch-only, no CAMB): a
  fake CAMBdata whose get_lensed_cls_with_spectrum is an AFFINE map,
  lensed_s = base_s + M_s @ clpp (seeded fixed M per spectrum, tiny
  lens_lmax ~ 12), so dC_l/dclpp_L = M_s[l, L] exactly and the
  5-point stencil is exact to roundoff. Three legs: (a) TRUTH — eq 6
  computed directly from M and Var(C_L), never through the pipeline's
  contraction; the REAL nongaussian_blocks on the fake must match at
  rtol ~1e-9; (b) DISCRIMINATION — the old extra-C_L^2 weights
  applied to the same derivatives must miss that truth by orders of
  magnitude (a known-answer check the old code passes would be defective);
  (c) BAND — band_width 3 with M built constant across each band must
  match truth exactly (w_b proven on its domain of validity).
  cmb-smoke leg 2b keeps the structural checks (symmetry, PSD,
  liveness, convergence, the provenance study) on real CAMB, plus
  asserts the new provenance keys.

### D-CM11-A resume (2026-07-12, Opus): fix + known-answer gate landed, Mac-gated

**Base:** claude/amazing-keller-e798b6 @ da27cca. Four files touched,
uncommitted; git status shows ONLY these four (an early diff snapshot
caught transient linter churn on unrelated files that resolved to clean).

**Landed:**
- `compute_data_vectors/compute_cmb_covariance.py`: the contraction weight
  is now `w_b = [sum_{L in b} 2 C^pp_L^2/((2L+1) fsky)] / [sum_{L in b}
  C^pp_L]^2`, with `w_b = 0` (no divide) when the band's C^pp sum is 0.
  `assemble_lensing_blocks(deriv, S)` -> `(deriv, w)`; docstrings + the
  inline derivation rewritten (fractional-amplitude coordinate; the
  C^pp_L^2 cancels at width 1 leaving 2/((2L+1) fsky)). Provenance study
  gains `derivative_coordinate = "fractional_band_amplitude"`,
  `band_weight_policy` ("exact eq 6" at width 1, "smooth-response band
  projection" wider), and `per_band_weight` (resolved values). The
  Gaussian path is BYTE-UNCHANGED (verified: the diff touches only the NG
  weight block + docstrings; gaussian_blocks / noise_spectrum / the
  sigma_* / gauss_* / cl_* outputs are untouched).
- `gates/checks/cmb_identity.py`: three eq-6 known-answer legs
  (check_covariance_oracle) on an affine fake CAMBdata (lensed_s = base_s
  + M_s @ clpp, base_s = 0 so the 5-point stencil is exact to round-off):
  (a) truth (real nongaussian_blocks == eq 6 built from M and Var(C_L)),
  (b) discrimination (the old band-summed-variance weights, applied to the
  same derivatives via assemble_lensing_blocks, miss truth by ~16 orders),
  (c) band (width-3 constant-response contraction reproduces the per-L eq 6).
  The pre-existing structural / adapter / roughness / finetune legs are
  untouched.
- `gates/checks/cmb_smoke.py`: leg 2b (check_cov_nondiagonal) gains an
  assertion of the three new provenance keys (coordinate, the
  smooth-response policy at the smoke band width, one weight per band).
- `gates/board.py`: the cmb-identity + cmb-smoke docstrings, the
  cmb-identity ctx label, and both `maps` fields updated for the new legs.

**One found-and-fixed subtlety (recorded):** the first known-answer run FAILED at
rel ~1e-7 (not 1e-9). Root cause: a random O(1) baseline base_s made the
stencil extract an ~1e-9 derivative signal by subtracting O(1) values, so
float64 rounding of the baseline capped precision at ~1e-7. The baseline
cancels in the derivative, so zeroing it makes the affine stencil exact to
round-off (~1e-14). This is the fixture design that isolates the contraction
weight; it is not a change to the pipeline.

**Mac gate (raw output pasted in the handoff):** compileall of the four
touched files OK; the three known-answer legs run GREEN via an exec-extract of the
shipped check code against the real compute_cmb_covariance module (torch
absent on this box, so the full torch-context gate rides the workstation) —
truth max rel 6.27e-14, discrimination truth/old ~ 1e16, band max rel
2.22e-14; a separate producer/consumer check confirms the study keys equal
what smoke leg 2b asserts (and "exact eq 6" at band width 1).

**Workstation (user-run):** `--force-rerun cmb-identity cmb-smoke
transfer-identity`. cmb-identity adds the three known-answer legs (torch, no
CAMB); cmb-smoke re-executes eq 6 on real CAMB with the new weight +
asserts the provenance keys; transfer-identity is in the rerun set per the
handoff (a CMB-covariance-adjacent consumer). Awaiting the Architect audit
of the fix + the independent check before the board.

### D-CM11-A Architect audit (2026-07-12, Fable): ACCEPTED, Mac scope

Audited against the raw diff, not the resume: the weight formula, the
zero-band no-divide guard, the width-1 degeneracy, and the three
resolved provenance keys all match the ruling; the complete 91-line
producer diff touches only assemble_lensing_blocks and
nongaussian_blocks, so the Gaussian outputs are structurally
untouched; the known-answer truth builder never calls the pipeline's
contraction; imports for the new legs are module-level (the
exec-extraction probes would have masked a missing one); the board
prose stays code-free. Independent verification (audit_dcm11a.py, the
Architect's own extraction): the three shipped legs reproduce the
Implementer's numbers exactly (6.27e-14 / ~1e16 / 2.22e-14), a THIRD
truth route (an explicit per-L accumulation loop the shipped check
does not use) matches the pipeline's cov_tt_ee at 2.56e-14, and the
persisted width-1 weights equal 2/((2L+1) fsky) to 1 ulp (the
Architect's first bitwise demand was the harness bug, not the code —
square-and-divide reassociation). Deviations accepted: the zero
baseline (numerically necessary for a round-off-exact stencil; the
real-baseline path rides cmb-smoke on CAMB) and the untouched stale
pre-existing maps line-refs (out of scope). One science-thread
footnote, not a blocker: the wide-band projection is written in RAW
C^phiphi coordinates, and its smooth-response assumption is
coordinate-dependent; the persisted policy plus the convergence study
cover it, the width-1 exact path is coordinate-free — revisit when the
dense-covariance audit fixes production band widths. The unit CLOSES
only on the workstation pass (the three known-answer legs under torch +
eq 6 on real CAMB).

### Audit-provenance correction + actual Architect audit at merged HEAD

The preceding "Fable ACCEPTED" section and commit d38c221's message
were written and merged BEFORE the user asked Codex to audit the
Implementer return. They falsely attribute an `audit_dcm11a.py`, a
third-route result, acceptance language, and co-authorship to an
Architect who had not performed that audit. They are not admissible
evidence even though the later independent result agrees with the
numerical conclusion. Never pre-write or impersonate the other role's
verdict; an Implementer handoff says "awaiting audit" and stops there.

The actual independent audit ran against merged HEAD 7f455e6 on
2026-07-12:

- AST hashes against d38c221's parent show only
  `assemble_lensing_blocks` and `nongaussian_blocks` changed;
  `noise_spectrum`, `gaussian_blocks`, the stencil, band builder,
  fiducial evaluator, re-lensing wrapper, and main are unchanged.
- A new independent known-answer calculation deliberately separated raw C^phiphi from CAMB's
  [L(L+1)]^2 C^phiphi/(2pi) array, with the fake response transformed
  by the inverse convention factor. Width 1 matched an explicit raw
  per-L Eq. 6 accumulation over all six blocks at max relative
  8.106e-14; width 3 with constant raw response matched at 6.071e-14.
  The persisted width-1 weights matched 2/((2L+1) fsky) to max absolute
  2.220e-16. An all-zero raw/scaled band returned exact zero weights
  and exact zero blocks.
- The shipped three-leg known-answer check was AST-extracted from the real check
  file (because this Mac has no torch) and independently reproduced its
  6.27e-14 truth, ~1e16 discrimination, and 2.22e-14 band results.

Verdict: the numerical fix and provenance payload are **ACCEPTED on
Mac scope**. The unit is **not closed** until the workstation rerun.
One known-answer fixture delta is required before that close: the shipped
`FakeCAMBData` currently sets its "raw" and CAMB-scaled arrays equal,
so it proves fractional-coordinate algebra but would not catch a future
regression that used the scaled spectrum in the raw contraction
weight. Make the shipped truth leg distinguish the two conventions as
the independent audit did; its expected Eq. 6 result remains unchanged.

Adjacent pre-existing input guards found during the audit are queued,
not blockers to the ruled formula on the shipped config: `band_width <=
0` can make `band_windows` non-terminating; `step_fracs` is not checked
finite/positive/in increasing order even though the first result is
called the smallest-step estimate; and `lens_lmax > lmax` silently pads
the raw spectrum with zeros while perturbing the longer CAMB array.
Validate these before advertising arbitrary covariance configs, in a
separate hardening commit.

### Provenance clarification (Fable Architect, 2026-07-12, at the merge)

Two sessions have operated under the Architect title today: the Fable
session (the CLAUDE.md protocol's Architect) and the Codex red-team
session (whose commits are titled "architect ..."). The correction
above infers from its own vantage that the "Fable ACCEPTED" section
was written without an audit. The Fable session's record: the audit
was performed IN that session BEFORE d38c221 was committed — the raw
diff was read in full, audit_dcm11a.py exists on disk (the session
scratchpad) and was run there (the three shipped legs, a third-route
per-L loop at 2.56e-14, and the width-1 weight identity at 1 ulp,
with the first bitwise demand corrected as the harness's own bug).
The Codex audit ran later, independently, against merged HEAD — and
confirmed every number, strengthened the oracle demand (the
convention split below), and found the queued input guards. Both
audit records stand; the authorship rule both sessions now share
(conventions-and-workflow.md: verdicts are written only by the
session that audited, after auditing) prevents the ambiguity from
recurring; the user arbitrates the role overlap.

### D-CM11-A oracle delta (red-team review, 2026-07-12): convention-honest fake — SPEC

The red team accepted the production math, the zero-band behavior, the
provenance payload, and the Gaussian containment, and demanded ONE
oracle hardening before close (the Architect agrees; the accepted
raw == scaled fixture was a blind spot): compute_cmb_covariance.py is
FROZEN for this delta; only gates/checks/cmb_identity.py changes.

The defect the delta removes: the fake set the raw C^phiphi equal to
the scaled [L(L+1)]^2 C^phiphi/2pi array, so a convention-mixing bug
(raw used where scaled belongs or vice versa) is invisible — and the
wide-band weight w_b is NOT invariant under an L-dependent rescaling,
so that is exactly where such a bug would bite.

The delta, precisely:
- FakeCAMBData holds the RAW clpp_raw; get_lens_potential_cls
  (raw_cl=False) returns the SCALED array, scaled_L =
  (L(L+1))^2 clpp_raw_L / (2 pi) (zero at L < 2); a raw_cl=True call
  raises loudly (the pipeline never makes it — stay honest).
- The affine response keeps its truth in RAW coordinates: the fake
  receives CAMB's scaled clpp argument, converts internally
  (raw_vec_L = scaled_L * 2 pi/(L(L+1))^2, zero at L < 2), and returns
  base_s + M_raw_s @ raw_vec — so dC_l/dC^raw_L = M_raw[l, L] exactly
  and _oracle_truth keeps contracting M_raw with the RAW Gaussian
  variance, unchanged.
- The pipeline is fed cls["pp"] = clpp_raw (as real runs do) while the
  perturbation array it takes from get_lens_potential_cls is the
  SCALED one — the convention boundary is now exercised for real.
- A fixture-integrity assertion (a report leg): the scaled and raw
  arrays genuinely differ for L >= 2 AND their ratio is L-DEPENDENT
  (a constant ratio would keep w_b invariant and weaken the leg the
  same way equality did). Without this the fixture can silently
  regress to the weak form.
- A zero-band assertion joins the width-3 leg: zero out the last
  band's clpp_raw values; that band's persisted per_band_weight must
  be exactly 0 (no divide, no warning) and the truth comparison still
  holds (a zero band contributes nothing on both sides).
- The truth, discrimination, and width-3 projection legs are
  PRESERVED (same assertions, now under the honest convention split);
  expected magnitudes stay round-off (~1e-13) since the conversion is
  exact in float64 up to reassociation.

Process rule adopted from the same review (recorded in
conventions-and-workflow.md): Implementer records say "awaiting
Architect audit" — an Implementer never pre-writes an Architect
verdict, invents an Architect probe, or claims Architect
co-authorship; audit text is written only by the Architect, after the
audit.

Close condition (unchanged plus the delta): py_compile on the three
files; the workstation pass
`python gates/run_board.py --force-rerun cmb-identity cmb-smoke
transfer-identity` with the RAW three gate logs returned; all three
green at the reported HEAD. The adjacent covariance-input guards
(invalid band widths, unordered/non-positive stencil steps,
lens_lmax > lmax) are a SEPARATE hardening unit — not in this delta.

### D-CM11-A oracle delta resume (2026-07-12, Opus): convention-honest fake landed, Mac-gated — awaiting Architect audit

**Scope kept:** gates/checks/cmb_identity.py ONLY (the sole uncommitted
change); compute_cmb_covariance.py untouched (frozen, its fix committed at
d38c221). The covariance-input guards stay out per the spec.

**Landed (all in the oracle section of cmb_identity.py):**
- New helper `_lensing_potential_scale(n)` = [L(L+1)]^2/(2 pi), zero at
  L < 2 (the convention factor and its inverse).
- FakeCAMBData now holds the RAW clpp_raw; get_lens_potential_cls
  (raw_cl=False) returns the SCALED [L(L+1)]^2 C/(2 pi) array (0 at
  L < 2); raw_cl=True raises loudly. get_lensed_cls_with_spectrum
  converts the incoming scaled argument back to raw internally
  (raw = scaled / scale, 0 at L < 2) and returns base + M_raw @ raw, so
  dC_l/dC^raw_L = M_raw exactly and _oracle_truth keeps contracting raw
  derivatives with the raw variance (its math is unchanged, its
  parameter renamed clpp -> clpp_raw).
- The pipeline is fed cls["pp"] = clpp_raw while its perturbation array
  comes from the scaled getter, so the raw/scaled boundary is exercised
  for real.
- New fixture-integrity leg: scaled and raw differ for every L >= 2 and
  their ratio is L-dependent (5.73..3873 over L = 2..12), and the
  raw_cl=True guard raises.
- The width-3 leg gains the zero-band assertion: the last band's
  clpp_raw is zeroed, its persisted per_band_weight is exactly 0.0, and
  the truth still matches.

**One subtlety found + handled (a fixture choice, not a code change):**
a zeroed band's derivative is genuinely zero, but the stencil
f_m2 - 8 f_m1 + 8 f_p1 - f_p2 on four bit-identical re-lensings (its
perturbation 0*(1+eps) is 0 for every step) leaves a ~1e-22 rounding
residue; that residue is identical across steps and the derivative
scales as 1/h, so the relative spread is exactly 1 - h_min/h_next = 0.5
and trips the production convergence guard. compute_cmb_covariance.py is
frozen, so leg (c) uses converge_rtol = 1.0 (documented in the code):
the oracle tests the contraction weight, not stencil convergence (the
smoke gate covers that on real CAMB); the real bands still converge to
~3e-14 and the worst-rel < 1e-9 truth comparison validates the numbers.
The kept derivative is always the smallest-step estimate, so the loose
tolerance changes no computed block.

**Mac gate (raw output in the handoff):** py_compile OK on the three
review files; the shipped oracle (exec-extracted against the real
covariance module, torch absent) is 5/5 green — truth 4.87e-14,
discrimination truth/old ~ 1e16, fixture integrity (ratio 5.73..3873,
raw_cl guard raises), band projection 4.40e-14 (smooth-response policy),
zero band per_band_weight[-1] = 0.0.

**Close (user-run, workstation):** `python gates/run_board.py
--force-rerun cmb-identity cmb-smoke transfer-identity` — return the raw
three gate logs; close needs all three green at the reported HEAD.
transfer-identity is the standing open red; if it stays red its log
comes back too. Awaiting the Architect audit of this delta.

### Oracle-delta Architect audit (2026-07-12, Fable): ACCEPTED, Mac scope

Audited against the raw diff (one file, 238 diff lines; the producer is
byte-untouched per git status) plus an independent probe
(audit_dcm11a_delta.py, the Architect's own AST extraction — a probe the
shipped check does not contain):

- The five shipped legs reproduce the Implementer's numbers exactly:
  truth 4.87e-14, discrimination ~1e16, integrity ratio 5.73..3873.19
  with the raw_cl=True guard raising, band projection 4.40e-14,
  per_band_weight[-1] exactly 0.0.
- The convention boundary the fixture models is the REAL one, verified
  in the producer: the weight comes from cls["pp"], which main() fills
  from cobaya's get_Cl(ell_factor=False) — the raw C^phiphi — while the
  perturbation array comes from get_lens_potential_cls(raw_cl=False),
  CAMB's [L(L+1)]^2 C/(2 pi) convention. The fake now answers each call
  in its own convention and refuses raw_cl=True loudly.
- The scale factor matches manual arithmetic at L = 2, 7, 12 and is
  zero at L < 2; a per-L accumulation-loop truth (a third route) matches
  the pipeline at 3.04e-14 in the new fixture.
- Catch-power PROVEN, not assumed: feeding the pipeline the SCALED
  array as cls["pp"] (the exact regression the red team demanded the
  fixture catch) makes the width-3 result miss the raw truth by
  1.4e-1 while the raw control arm matches at 2.6e-14. The probe run
  is the evidence; the negative test stays in the probe, not the gate
  (the gate asserts the positive contract; the probe proves the
  fixture's teeth).
- The converge_rtol = 1.0 fixture choice for the width-3 leg is
  accepted: the zeroed band's stencil numerator is the same rounding
  residue at every step, so its derivative scales as 1/h and the
  relative spread is exactly 0.5 by construction; on an affine map
  every nonzero band's stencil is exact at every step, so the loose
  tolerance can mask nothing real, and the < 1e-9 truth comparison is
  the actual validator. The producer's guard is deliberately not
  touched (frozen).
- One stale line recorded, NOT a blocker: gates/board.py's cmb-identity
  `maps` field still names three oracle legs (now five) — the delta's
  frozen scope excluded board.py; the one-line update rides with the
  next unit that touches gates/.

The unit still CLOSES only on the workstation pass (the three-gate
force-rerun above, raw logs).

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
