# The scalar and CMB output families (SPE + CME)

Consolidated 2026-07-11 from scalar-parameter-emulators.md and
cmb-spectra-emulators.md (retired; the full delta ledgers and run
trails are in git history). The family-pattern recipe both units
instantiate lives in project-and-history.md; user-facing stories are
README sections 14 and 15.

## SPE — scalar (derived-parameter) emulators. CLOSED, board 25/25.

<a id="scalar-identity"></a>
**scalar-identity — synthetic scalar artifacts exercise the scalar geometry,
saved predictor, adapter, NPCE residual model, and fine-tune boundary.**

- files: temporary covariance and parameter-name sidecars plus synthetic
  `.h5`/`.emul` artifact pairs; every output is owned by the child check's
  temporary directory.
- subprocess: `gates/checks/scalar_identity.py`; no nested subprocess (the
  scalar Cobaya adapter is loaded with a minimal `Theory` stub).
- metric: per-leg — exact value or tensor equality for round trips and
  compositions, set/order equality for declared names, explicit tolerances
  only for the tiny-scale standardization and live-PCE controls, and typed
  refusal for invalid inputs.
- legs: 5, named `scalar-identity.artifact-round-trip`,
  `scalar-identity.geometry-and-schema-guards`,
  `scalar-identity.scalar-adapter-contract`,
  `scalar-identity.npce-composition`, and
  `scalar-identity.finetune-parity`.
- evidence: the assertions below exist in the child, but the child currently
  imports `DataVectorGeometry` before its first report. That module eagerly
  imports both `cosmolike_lsst_y1_interface` and GetDist's `IniFile`, so a
  CPU-PyTorch environment missing either dependency exits before any declared
  leg executes even though the board and child both advertise a Torch-only
  gate. The board wrapper currently reduces child assertions to the exit code;
  no logged-only claim is promoted.
- owed: repair both eager output-geometry dependencies at the boundary
  described by `25M-37` and rerun the complete child. Until then there is no
  whole-gate PASS on the declared Torch-only environment. Missing the compiled
  interface or GetDist is a current pre-leg red failure, while missing the
  registry's PyTorch capability skips the gate as `UNAVAILABLE`. The gate
  claims neither GPU nor a real-Cobaya lifecycle.

<a id="scalar-identity-artifact-round-trip"></a>
`scalar-identity.artifact-round-trip` asserts exact named predictions before
and after save/rebuild, exact scalar geometry-state tensors, and the rebuilt
scalar-family flags.

<a id="scalar-identity-geometry-and-schema-guards"></a>
`scalar-identity.geometry-and-schema-guards` rejects a constant output,
duplicate sidecar names, and a correction-head architecture while accepting
a genuinely varying tiny-magnitude output whose standardized spread is
within 0.05 of one.

<a id="scalar-identity-scalar-adapter-contract"></a>
`scalar-identity.scalar-adapter-contract` checks the stored output-name union
and input-name set, accepts an explicit output subset, and rejects duplicate
outputs, input/output chaining, an unavailable superset, and a non-scalar
artifact.

<a id="scalar-identity-npce-composition"></a>
`scalar-identity.npce-composition` requires a nonzero fitted polynomial base,
then checks the residual encode/decode algebra and saved base-plus-network
prediction by exact tensor or scalar equality.

<a id="scalar-identity-finetune-parity"></a>
`scalar-identity.finetune-parity` runs the epoch-zero warm-start parity check,
checks that the anchor mask excludes exactly an appended input column, and
rejects output-name and source-family mismatches before staging.

<a id="scalar-smoke"></a>
**scalar-smoke — a trained scalar emulator is checked against the analytic
relation `omegamh2 = omegam*(H0/100)^2` and through a real Cobaya evaluate
run.**

- files: temporary train/validation parameter tables, covariance and
  parameter-name sidecars, one saved `.h5`/`.emul` pair, a diagnostics PDF,
  and the temporary YAML and chain output written by Cobaya evaluate.
- subprocess: `gates/checks/scalar_smoke.py`; that child additionally runs
  `python -m cobaya run` for the evaluate leg, while training, prediction,
  and diagnostics execute in-process.
- metric: per-leg — validation median below `0.3`, off-center analytic
  relative error below `0.05`, diagnostics page/file-shape assertions, and
  Cobaya-derived-value relative error below `0.05` after a zero exit code.
- legs: 4, named `scalar-smoke.training-collapse`,
  `scalar-smoke.analytic-prediction`, `scalar-smoke.diagnostics-output`, and
  `scalar-smoke.cobaya-evaluate`.
- evidence: each claim is asserted inside the child; the board wrapper
  currently exposes only the aggregate child exit code, and diagnostic text
  printed after a failed readback is troubleshooting output rather than a
  green assertion.
- owed: the registry capability lane requires CPU PyTorch and Cobaya,
  including its CLI. Matplotlib and GetDist are additional diagnostics
  imports; if either is absent, that assertion fails rather than all four legs
  becoming capability-`UNAVAILABLE`. No GPU, CosmoLike, or CAMB result is
  claimed.

<a id="scalar-smoke-training-collapse"></a>
`scalar-smoke.training-collapse` asserts only that the best validation median
after two training epochs is below `0.3`. The child comment motivates that bar
with the theoretical chi-square-one median; it neither computes an initial
median nor evaluates the fixture's mean predictor for this leg.

<a id="scalar-smoke-analytic-prediction"></a>
`scalar-smoke.analytic-prediction` compares an off-center saved prediction
with the analytic `omegamh2` value and requires relative error below `0.05`.

<a id="scalar-smoke-diagnostics-output"></a>
`scalar-smoke.diagnostics-output` asserts three scalar diagnostic pages and
a nonempty diagnostics PDF larger than 10,000 bytes; it does not certify the
scientific content of the plots.

<a id="scalar-smoke-cobaya-evaluate"></a>
`scalar-smoke.cobaya-evaluate` requires the Cobaya subprocess to exit zero
and its derived `omegamh2` readback to agree with the analytic value within
relative error `0.05`; subprocess exit alone is not the leg's evidence.

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

## Original unit 5(a): the scalar driver cannot require a data-vector path

The first red-team queue found a contradiction in the advertised generic
driver contract. The cosmic-shear driver reads
`cfg["data"]["train_dv"]` while building its run tag and root attributes, but
the scalar schema correctly forbids `train_dv`: its training source is a
parameter-chain table and its targets are named derived columns. Therefore
the cosmic-shear driver could not be the generic scalar/rdrag route the prose
claimed.

A dedicated scalar driver already existed before that audit: commit
`7d024553` introduced it as `train_scalar_emulator.py`, and commit `3b6724c`
later renamed it `scalar_train_emulator.py`. The audit repaired the routing
and documentation claim; it did not create that driver. The later family-first
unit (45M-80, commit `e9943bc`) made the wrong-driver refusal executable
rather than merely documented. The current scalar driver's run tag uses the
resolved architecture and staged row count, and its artifact attributes
record `train_params`, `val_params`, and named outputs. It never reads scalar
`train_dv` or `val_dv`.

The enduring contract is family-owned metadata, not merely a special-case
`if` statement:

1. Every driver reads only keys legal in its validated family schema.
2. The scalar run tag and artifact fields use only scalar-legal facts; they
   are never inferred from a nonexistent data-vector filename.
3. Scalar artifact provenance names both parameter tables, ordered input and
   output names, and row counts. It does not emit placeholder data-vector
   fields. Collision-resistant scientific identity and complete resolved-pass
   provenance remain owned by the separate artifact-integrity campaign; this
   historical closure does not claim those later units are complete.
4. A scalar config containing `train_dv` remains a schema error; making the
   driver tolerate the forbidden key would hide a wrong-family file.
5. The ordinary rdrag example must run through config validation, staging,
   geometry construction, training, save, rebuild, and named prediction. A
   mutation that routes it through the cosmic-shear run-tag/attribute builder
   must fail before training.

This closes only the stale generic-driver/routing claim in unit 5(a). The
original unit-5 bundle's other independent findings have their own durable
owners: NPCE selection is global unit 19; covariance/geometry validation is
global unit 11 plus MPS unit 16 where applicable; the BAOSN pair getter is
unit 15; optimized-mode assertions are unit 12; artifact loading belongs to
artifact integrity; adapter device values belong to the typed adapter
contract; and plotting defaults belong to the plotting/documentation
campaign.

## CME — CMB spectra emulators. ACCEPTED END TO END (board run 4, 2026-07-11); gates cmb-identity/cmb-smoke.

<a id="cmb-identity"></a>
**cmb-identity — synthetic CMB artifacts exercise the diagonal geometry,
amplitude-dependent score, saved predictor and adapter, model variants,
fine-tuning, and the non-Gaussian covariance contraction.**

- files: temporary covariance files and synthetic `.h5`/`.emul` artifact
  pairs; the covariance known-answer arrays are constructed in memory and no
  persistent science product is written.
- subprocess: `gates/checks/cmb_identity.py`; no nested subprocess, real
  CAMB, or real Cobaya process (the adapter uses a minimal `Theory` stub).
- metric: per-leg — exact value, tensor, state, or mapping comparisons where
  promised, explicitly bounded floating-point comparisons for transforms, score and
  roughness, typed refusal checks, and relative error below `1e-9` for the
  direct covariance known answer.
- legs: 7, named `cmb-identity.geometry-and-reference-schema`,
  `cmb-identity.amplitude-law-and-score`,
  `cmb-identity.artifact-and-adapter-round-trip`,
  `cmb-identity.roughness-contract`,
  `cmb-identity.model-variant-composition`,
  `cmb-identity.finetune-parity`, and
  `cmb-identity.covariance-known-answer`.
- evidence: every listed claim is asserted inside the child check; the board
  wrapper currently sees only its exit code, so future per-leg reconciliation
  must preserve these narrower claims; no printed banner or detail line is
  counted independently.
- owed: none once CPU PyTorch is available; without it all seven declared
  legs are environment-UNAVAILABLE even though the covariance known-answer
  cluster itself is NumPy. No GPU, real-CAMB, or real-Cobaya evidence is
  claimed.

<a id="cmb-identity-geometry-and-reference-schema"></a>
`cmb-identity.geometry-and-reference-schema` checks the declared Gaussian
per-multipole scale, exact persistence of the fiducial amplitude references,
the geometry state round trip, the endpoint comparison `sigma[0] >
sigma[-1]` (not monotonicity over the axis), refusal of a nonpositive
fiducial value naming multipole 50, and typed rejection of nonfinite, Boolean,
string, or nonpositive reference values where the domain forbids them.

<a id="cmb-identity-amplitude-law-and-score"></a>
`cmb-identity.amplitude-law-and-score` checks the order-one reference law,
transform round trip, parameter-aware physical score, factor-corrected
roughness residual, stale-parameter isolation, and the corresponding missing
or invalid-law refusals, including mutation controls for the retired raw
factor.

<a id="cmb-identity-artifact-and-adapter-round-trip"></a>
`cmb-identity.artifact-and-adapter-round-trip` checks exact saved predictions
for both supported amplitude laws and checks the stubbed adapter's shared
axis, low-multipole padding, requirements, convention guards, spectrum
uniqueness, and request-range refusals.

<a id="cmb-identity-roughness-contract"></a>
`cmb-identity.roughness-contract` checks frequency discrimination, exact zero
for a zero residual, bitwise identity when roughness is off, one-reduction
score composition, and the bounded lensing-period contribution on its
synthetic residuals.

<a id="cmb-identity-model-variant-composition"></a>
`cmb-identity.model-variant-composition` checks correction-head attachment,
epoch-zero identity, phase/refusal behavior and save/rebuild equality, plus
exact NPCE residual algebra, roughness composition, saved prediction, and
the PCE/amplitude-law exclusivity guard.

<a id="cmb-identity-finetune-parity"></a>
`cmb-identity.finetune-parity` accepts a CMB source, runs its epoch-zero
warm-start parity check, accepts the CMB fine-tune config shape, and rejects
using that source through the CosmoLike-only pin.

<a id="cmb-identity-covariance-known-answer"></a>
`cmb-identity.covariance-known-answer` compares all six non-Gaussian blocks
with a direct sensitivity-matrix contraction at relative error below `1e-9`,
proves the retired weights miss by more than six orders of magnitude, keeps
raw and scaled lensing-potential fixtures distinct, checks a width-three
constant-response projection, and requires an exactly zero weight for a
zeroed band.

<a id="cmb-smoke"></a>
**cmb-smoke — the generator, Gaussian and non-Gaussian covariance builders,
training loop, Cobaya provider, and diagnostics execute on a small real-CAMB
fixture.**

- files: temporary train/validation parameter tables and sidecars, four CMB
  spectrum dumps per split, Gaussian and non-Gaussian covariance `.npz`
  files, one saved `.h5`/`.emul` pair, and a diagnostics PDF; the `$ROOTDIR`
  work directory is removed in `finally`.
- subprocess: `gates/checks/cmb_smoke.py`, which launches
  `dataset_generator_cmb.py` twice and `compute_cmb_covariance.py` once for
  each of the Gaussian and non-Gaussian configurations; training, the Cobaya
  model lifecycle, and diagnostics then execute in-process.
- metric: per-leg — process/file/schema assertions for generation and
  covariance, structural and tolerance checks for the dense covariance,
  validation median below half the staged mean-predictor median, provider
  relative equality at `rtol=1e-6`, and diagnostics page/file assertions.
- legs: 6, named `cmb-smoke.generated-spectrum-dumps`,
  `cmb-smoke.gaussian-covariance`,
  `cmb-smoke.nondiagonal-covariance-structure`,
  `cmb-smoke.training-collapse`, `cmb-smoke.cobaya-serving`, and
  `cmb-smoke.diagnostics-output`.
- evidence: all six claims are asserted in the child; child-process exit
  codes are combined with output checks and are never promoted by
  themselves; stdout/stderr tails are failure diagnostics only.
- owed: the registry capability lane requires CPU PyTorch and Cobaya. An
  unresolved `$ROOTDIR` aborts preflight; a missing compiled CAMB makes the
  generator/covariance assertions fail and leaves later aids unreported;
  missing plotting imports make the diagnostics assertion fail. Those are
  current red/missing evidence outcomes, not capability-`UNAVAILABLE` legs. A
  banner or old log cannot substitute for execution.

<a id="cmb-smoke-generated-spectrum-dumps"></a>
`cmb-smoke.generated-spectrum-dumps` requires both generator subprocesses to
exit zero, all parameter-table sidecars and four spectrum dumps to exist,
each TT dump to have the expected shape, and each lensing-potential dump to
contain a nonzero value.

<a id="cmb-smoke-gaussian-covariance"></a>
`cmb-smoke.gaussian-covariance` requires the Gaussian covariance subprocess
to exit zero and writes an `.npz` whose multipole axis is exactly `2..LMAX`
and whose TT standard deviations are positive.

<a id="cmb-smoke-nondiagonal-covariance-structure"></a>
`cmb-smoke.nondiagonal-covariance-structure` requires six dense blocks with
the expected shapes, then checks TT symmetry, diagonal growth, a nonzero
off-diagonal, a tolerance-bounded nonnegative spectrum, and the step-study
and fractional-amplitude keys in provenance. “Diagonal growth” here means no
TT diagonal falls below its Gaussian counterpart beyond the `1e-10` relative
allowance; it does not require a strict increase. This is a structural
real-CAMB check, not the independent numerical known answer supplied by
cmb-identity.

<a id="cmb-smoke-training-collapse"></a>
`cmb-smoke.training-collapse` requires the best validation median to fall
below half the median score of the staged mean predictor, using the same
batch parameters for the amplitude-dependent score.

<a id="cmb-smoke-cobaya-serving"></a>
`cmb-smoke.cobaya-serving` runs the real in-process Cobaya lifecycle and
requires its padded TT array to equal the saved predictor on multipoles
`2..LMAX` within `rtol=1e-6`, with finite values and exact zeros below two.

<a id="cmb-smoke-diagnostics-output"></a>
`cmb-smoke.diagnostics-output` asserts two CMB diagnostic pages and a
nonempty diagnostics PDF larger than 10,000 bytes; it does not certify the
scientific interpretation of the plotted curves.

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

### Workstation run 12 (2026-07-12 12:53, HEAD 7f455e6): the eq-6 MATH is workstation-proven

The user ran the three-gate force-rerun BEFORE the oracle delta merged,
so the executed tree carried the three-leg oracle:

- cmb-identity GREEN under torch: truth 5.89e-14, discrimination
  ~1e16, band projection 3.02e-14 (the Mac numbers reproduced on CUDA
  to the same order).
- cmb-smoke GREEN — the FIRST full eq-6 execution on real CAMB: six
  dense blocks symmetric/PSD/off-diagonal-alive, and leg 2b asserted
  the fractional-amplitude provenance keys on the real .npz
  (derivative_coordinate fractional_band_amplitude, policy
  smooth-response band projection, per-band weights present). The
  run-11 clpp-length fix is proven by the same execution.
- transfer-identity stayed red on ONE leg — a gate-fixture defect,
  root-caused (artifacts-inference-warmstart.md, "transfer-identity
  cross-family leg"), unrelated to the covariance path.

Status: the D-CM11-A NUMERICAL fix is closed on workstation evidence.
The five-leg oracle delta (86db0b4, Architect-accepted) still needs
one cmb-identity force-rerun after it merges — that rerun rides with
the transfer-identity fixture fix:
`python gates/run_board.py --force-rerun cmb-identity transfer-identity`
(cmb-smoke needs no rerun: the delta touches only the gate file, and
smoke is green at the unchanged producer).

## Covariance-input validation unit (red-team finding 2026-07-12, Architect-VERIFIED; queued after grid2d staging)

The covariance script trusts cov_args without a schema/range boundary.
Every red-team failure path re-derived against the code:

- band_windows (compute_cmb_covariance.py ~279-285): band_width = 0
  never advances `start` (stop = start - 1, start = stop + 1 = start);
  negative walks BACKWARD — both non-terminating, before any CAMB cost.
- The kept derivative is `stack[0]` (~580), i.e. the FIRST-listed step,
  while the comment (~545) and docstring promise "the smallest step's
  estimate": reordering step_fracs silently changes the science answer
  on any nonlinear (i.e. real) response. Fix by VALIDATION, not
  behavior change: require step_fracs strictly increasing, so element
  zero really is the smallest; the shipped configs already comply and
  stay byte-identical.
- Silent zero-pad (~529-531): pp_raw is zeros(lens_lmax+1) filled only
  to len(cls["pp"]); a lens_lmax beyond the supplied raw power gets
  zero bands, and the D-CM11-A zero-band no-divide guard — correct for
  PHYSICAL zeros — then silently deletes those eq-6 contributions
  (verified arithmetic: live derivatives at L=2,3,4 with power only
  through L=2 give weights [0.4, 0, 0] where [0.4, 2/7, 2/9] belong).
  Data absence must be loud; only physical zeros may be quiet. main()
  also requests pp only to lmax (~415), so lens_lmax > lmax ALWAYS
  under-fills.
- fsky = float(cov.get("fsky", 1.0)) (~686): unvalidated — 0 gives
  infinite variance, negative gives negative variance then NaN under
  the square root; ALSO a silent code default on a science-critical
  value.
- No key whitelist at any of the three levels (cov_args / noise /
  nongaussian) — mistyped keys are ignored (contrast: the script's own
  params whitelist, and emul_mps._check_extra_args); and
  bool(ng_cfg.get("enabled", False)) makes the quoted YAML string
  "false" ENABLE the non-Gaussian path.

**The contract (Implementer; producer + pure gate legs, no unrelated
refactor, band policy unchanged):**

1. One pure `validate_cov_args` boundary, called in main() before the
   CAMB solve (and unit-callable without CAMB). Unknown keys at all
   three levels raise naming the key and the allowed set.
2. Required keys and ranges — and, per the house never-trust-defaults
   doctrine, ALL keys explicit in the YAML, no code defaults (the
   Architect sharpening; delta_te / fsky / band_width / converge_rtol
   currently default silently): lmax non-bool int >= 2; fsky finite,
   0 < fsky <= 1; delta_tt / delta_ee / delta_te finite >= 0;
   beam_fwhm finite > 0; enabled a REAL bool (isinstance, rejecting
   strings); when enabled: lens_lmax non-bool int >= 2, band_width
   non-bool int >= 1, step_fracs >= 2 finite, strictly positive,
   strictly increasing values, converge_rtol finite > 0.
3. Range completeness: main() requests pp through
   max(lmax, lens_lmax); both the raw cls["pp"] and the scaled
   re-lensing array must cover every requested L — a short array
   raises naming the needed and available maxima; the zero-pad is
   REMOVED.
4. The shipped example YAML + any gate config gain the newly-required
   keys (config fix, never a code default); every YAML change is shown
   as a paste-ready block in the resume.
5. Gate legs (pure numpy, no CAMB, riding cmb-identity): zero/negative
   band_width raise before band_windows; unordered / duplicate / zero
   / NaN / infinite step_fracs raise; zero / negative / nonfinite
   fsky, negative / nonfinite noise, nonpositive beam raise;
   unknown-key and quoted-"false" cases raise; short raw/scaled
   lensing arrays raise instead of zero weights; and a valid fixture
   reproduces the currently-accepted numbers (the five known-answer
   legs stay green unchanged).
6. README prose in plain language; ledger codes stay in notes/.

### 25M-07 RETRACTED by the Red Team (2026-07-13): the proposed step-size ceiling assumed a domain restriction that was not established

The executed observation remains accurate: if the numerical step fraction is
called `s_step`, the `-2 s_step` stencil arm uses factor `1 - 2 s_step`.
For `s_step > 0.5`, that factor is negative. The filed conclusion did not
follow. Vivian's statement that the cosmological reduced Hubble parameter is
near `h = 0.7` exposed the dangerously ambiguous symbol; it was not evidence
that the numerical step fraction should be near 0.7. Independently of that
naming error, a centered finite-difference calculation may deliberately
evaluate a formal signed extension even though that arm is not a standalone
physical spectrum. The Red Team had not proved from CAMB's executed contract
that signed arms are forbidden, and the affine fake proved only the
arithmetic, not physical invalidity.

Therefore the proposed `0 < s_step < 0.5` validation, refusal legs, and
unit-13 amendment are withdrawn. No code or gate change is owed from 25M-07.
The identifier is retired and may not be reused. A future restriction needs
an independent CAMB known-answer showing that the signed arm produces a wrong
derivative, not the category assumption that every stencil evaluation must be
a realizable cosmology. This correction was made immediately after the user
identified the notation collision and before Architect adjudication of the
second batch.

### 25M-08 amendment (Red Team CONFIRMED, awaiting Architect adjudication): positive steps can round to no perturbation and certify zero non-Gaussian covariance

The same validator owns no lower representability boundary. Factors are
formed as `1.0 + eps` in float64 and convergence compares response stacks only
with one another (`compute_cmb_covariance.py:558-585`). For
`step_fracs=[1e-20,2e-20]`, every factor rounds to exactly float64 `1.0`.

Executed through the real `nongaussian_blocks` body with a finite linear fake:
all 16 relensing inputs were byte-identical to the fiducial (one unique byte
string), every derivative and covariance block was exactly zero, and every
spread was `0.0`. The study therefore labels a no-op perturbation perfectly
converged and silently deletes the non-Gaussian term. This representability
failure stands independently of the retracted 25M-07 domain claim.

Required contract: before CAMB, each `s_step` has ordered representable
float64 factors
`1-2*s_step < 1-s_step < 1 < 1+s_step < 1+2*s_step`. After multiplication, every nonzero
band must actually change on both signs; genuinely zero physical bands retain
their existing zero-band policy. Persist the factors and changed-value counts.
Derive the boundary from representation (`nextafter`) rather than a magic
decimal floor.

Pure gate legs: the `[1e-20,2e-20]` false-green fixture refuses before
relensing; cases bracketing `nextafter(1, +/-inf)` prove the exact boundary;
shipped steps remain unchanged; nonzero-band payloads differ on both signs;
the physical-zero control stays legal; and a mutation retaining only
positive/increasing checks returns zero blocks and must red.

### User ruling for covariance reasonableness (2026-07-13)

Scientific reasonableness is anchored to the repository's Planck-LCDM
fiducial, not to arbitrary extreme synthetic cosmologies. The current
reference is the explicit `example_yamls/cmb_covariance_lcdm.yaml` mapping:
`H0=67.36` (therefore cosmological `h=0.6736`), `As=2.1e-9`,
`ns=0.9660`, `omegabh2=0.02237`, `omegach2=0.1200`, `tau=0.0544`, and
`mnu=0.06`, together with its declared experiment and numerical controls.
Those values must remain a byte-identical known-answer control whenever the
covariance validator changes.

An extreme fake can still demonstrate that a schema accepts an undefined or
unrepresentable input, or prove that a gate catches a mutation. It cannot by
itself establish that the code gives a scientifically wrong covariance for a
reasonable cosmology. Findings 25M-08, 25M-11, and 25M-12 are therefore
schema-totality and catch-power claims; they are not claims that the shipped
Planck-LCDM calculation is numerically wrong. Any future science-result claim
must execute the Planck-LCDM control or a clearly justified neighboring
cosmology and compare against an independent known answer.

### 25M-11 amendment (Red Team CONFIRMED, awaiting Architect adjudication): individually nonnegative T/E noise amplitudes can define an indefinite joint covariance

The queued schema requires `delta_tt`, `delta_ee`, and `delta_te` to be finite
and nonnegative but never checks whether they are one physical 2-by-2 T/E
noise covariance. `noise_spectrum` squares each amplitude independently and
`gaussian_blocks` assembles the joint TT/TE/EE covariance without a PSD check
(`compute_cmb_covariance.py:169-243`). The required per-ell condition is
`N_te^2 <= N_tt * N_ee`; because all three use the same beam formula, the
input-amplitude condition is `delta_te^2 <= delta_tt * delta_ee`.

The real production functions were executed at ell 2 with zero signal,
`delta_tt=delta_ee=1`, `delta_te=10`, beam 1, and `fsky=1`. Every published
per-spectrum sigma was finite, but the joint covariance eigenvalues were
`[-2.86365772e-11, 1.43097071e-11, 2.86537506e-11]`. Thus the accepted public
configuration publishes a strongly indefinite matrix while all scalar
finiteness checks green. Output-geometry SPD work is defense in depth; the
covariance producer itself must not label an unphysical matrix as science.

Required contract: the pre-CAMB config boundary enforces the 2-by-2 noise PSD
inequality in the storage arithmetic with a representation-derived rounding
band. After fiducial spectra are available, verify
`(Cte+Nte)^2 <= (Ctt+Ntt)(Cee+Nee)` at every ell; before publication verify the
assembled symmetric joint Gaussian/dense covariance is PSD within one owned
numerical tolerance. No clipping, diagonal loading, or absolute value repairs
an invalid input silently.

Pure red legs: the `1/1/10` witness refuses before CAMB; equality passes;
just-over-boundary refuses under the declared band; shipped `delta_te=0`
remains byte-identical; one realistic signal+noise control passes both the
per-ell and tiled-matrix checks; a constructed post-signal violation refuses;
and a mutation checking only individual nonnegativity publishes the negative
eigenvalue above and must red.

### 25M-12 amendment (Red Team CONFIRMED, awaiting Architect adjudication): finite positive inputs can overflow derived noise and still publish

The queued schema validates the finiteness/positivity of beam, noise
amplitudes, `fsky`, and `lmax` separately. The derived arithmetic has no
representability check. `noise_spectrum` exponentiates the beam factor
directly (`compute_cmb_covariance.py:185-188`), `gaussian_blocks` squares and
multiplies it (`:229-243`), and `main` takes square roots and writes every
array without a postcompute finite check (`:700-769`).

The production function was executed at ell 5000, delta 1 muK-arcmin. A
finite positive 60-arcmin beam returned `inf` with an overflow warning. At 32
arcmin the noise itself is finite (about `4.11e162`) while covariance squaring
overflows. All input values satisfy the recorded future schema. A one-degree
beam is not malformed by type; it is incompatible with the requested
multipole support and must be refused rather than published in an apparently
complete `.npz`.

Required contract: before CAMB, derive the largest beam exponent from the
resolved `lmax` in float64 and prove both the full noise expression and its
covariance products are representable using the named formulas, not a guessed
beam cap. After computation, require every fiducial spectrum, sigma, Gaussian
cross term, and enabled dense block to be finite before any output mutation;
name the first key/ell/value on failure. No clipping or infinite sentinel.

Pure red legs: the exact 60-arcmin/ell-5000 witness preflight-refuses; a value
just inside the derived representability boundary remains finite; the
32-arcmin case whose noise is finite but square overflows refuses; shipped
1-arcmin output is byte-identical; and a mutation checking input finiteness
only reaches `np.savez` with an infinite member and must red.

### 45M-01 amendment (2026-07-12, red-team; Architect-VERIFIED): the fiducial params block — schema, resolution, provenance

(The red-team rounds are now labeled 45M-XY by user convention; the
Implementer handoff for this unit carries 45M-01.)

The unit above validates cov_args; the params block has the same
disease one level up. validate_lcdm_params
(compute_cmb_covariance.py:334-378) checks only entries that happen
to be PRESENT — no required-key schema, no
mutually-exclusive-alternatives rule — and its per-value checks have
two type holes: `isinstance(value, (int, float))` admits bool (a
subclass of int), and the LCDM_FIXED_ONLY pin uses
`abs(float(value) - pin) > 1e-12`, which NaN answers False (every
NaN comparison is false), so `omk: .nan` PASSES the "omk must be 0"
check. Probe-confirmed on the shipped body (exec-extracted via AST,
7/7 accepted): empty `params: {}`; `As: true`; `As: .nan`;
`omk: .nan`; no amplitude parameter at all; both As and logA; both
H0 and thetastar. The accepted mapping is handed UNCHANGED to cobaya
(fiducial_spectra: `model_info["params"] = info["params"]`, :405)
and provenance persists that same unresolved mapping
(`"fiducial_params": info["params"]`, :751) — omitted parameters are
therefore external Cobaya/CAMB defaults, and the covariance file
cannot reconstruct the fiducial cosmology that generated its own
spectra. That breaks the never-trust-defaults doctrine on both
surfaces at once: a silent input default AND an unresolved persisted
value.

Contract (folds into the SAME Implementer unit; same discipline —
producer + pure gate legs, shipped numbers preserved):

1. One exact fiducial-parameter schema (in validate_lcdm_params or a
   sibling it calls), validated BEFORE any Cobaya/CAMB construction.
2. Every value a finite, non-bool real: reject
   `isinstance(value, bool)` explicitly, reject non-finite via
   math.isfinite. This kills `As: true`, `As: .nan`, and the
   `omk: .nan` pin defeat upstream of the pin comparison.
3. Exactly ONE amplitude parameter from {As, logA} and exactly ONE
   expansion parameter from {H0, thetastar, cosmomc_theta}; zero or
   two-plus from either set raises a corrective error naming the set
   and what was found.
4. Required singletons: ns, omegabh2, omegach2, tau, mnu — each
   present or a corrective error naming the missing key.
5. omk, w, wa REQUIRED EXPLICIT in the YAML at their LCDM values,
   where the existing pin check applies (Architect ruling, consistent
   with the cov_args sharpening "ALL keys explicit in the YAML, no
   code defaults"). omk 0 / w -1 / wa 0 ARE the CAMB defaults, so the
   shipped solve is numerically identical. Contingency: if the
   workstation's real-CAMB check shows the explicit keys perturb the
   solve (they should not), fall back to the red team's alternative —
   materialize the three into the resolved mapping inside this
   repository before model construction — and record which branch
   shipped in the resume.
6. The mapping that passed the schema IS the resolved mapping: the
   same object goes to cobaya and into provenance
   ("fiducial_params") — written and consumed cosmologies identical
   by construction.
7. The shipped example_yamls/cmb_covariance_lcdm.yaml params block
   gains omk/w/wa (config fix, never a code default; paste-ready
   block in the resume); every other number untouched.
8. Gate legs (pure, no CAMB/Torch/GPU, riding cmb-identity beside the
   cov_args legs): empty mapping; bool value; NaN value; Inf value;
   missing amplitude; double amplitude (As + logA); missing
   expansion; double expansion (H0 + thetastar); `omk: .nan` (the
   defeated pin — must now die on finiteness); missing omk/w/wa;
   non-flat omk (regression — already caught, keep it caught); and a
   shipped-config control: the amended params block validates clean
   and the persisted fiducial mapping equals the consumed one and is
   complete.

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

## REOPENED: the CMB amplitude law — the metric carries f^2 and the factor is 1e9-scale (red-team 45M-21 + 45M-22, 2026-07-12, Architect-VERIFIED; queue 42 + 43, CRITICAL — sequenced right after the BAOSN quadrature unit)

The CMB amplitude-law acceptance is REOPENED on two coupled defects.

### Queue 42 (45M-21): the reported chi2 is f^2 times the physical chi2

CmbFactoredChi2 (losses/cmb.py:238) encodes
t = whiten(squeeze(C_ell) * f - center) with the per-row factor
f = exp(2 tau) / A_s (:334), decodes by dividing f back out — but
chi2 (:371-380) IGNORES params_whitened and delegates to
CmbDiagonalChi2.chi2 on the whitened residual. Its docstring claims
the factor "cancels in the residual": FALSE — pred and target of one
row share the same f, so the residual is f * (C_pred - C_truth) /
sigma and the reported chi2 is f^2 * chi2_physical. Consequences,
adopted as stated: delta-chi2 and every threshold fraction depend on
A_s and tau at fixed physical error; best-epoch selection is biased
toward small-f cosmologies; the 0.2 acceptance threshold stops
representing the covariance metric; the roughness "law-neutral" claim
is false (it acts on pred - target, so it carries f^2 too); the
existing identity gate proves factor arithmetic and the encode/decode
round-trip only — it never compares factored chi2 against a direct
physical known answer.

Contract: chi2 REQUIRES params_whitened, computes f, divides the
whitened residual by f, then sums the square; loss stashes/passes the
parameters instead of documenting them as unused; roughness (when
enabled) is defined on the factor-corrected whitened residual if the
intended penalty is law-neutral; encode/decode preserved (the defect
is the metric, not the invertible transform); every "cancels" claim
corrected; cmb-identity AND cmb-smoke rerun after the fix.
Workstation legs: factored chi2 vs a direct physical-spectrum
reference; fixed physical residual under varying (A_s, tau) leaves
physical chi2 invariant; the uncorrected form misses by exactly f^2
(catch-power); round-trip stays green; roughness invariant under the
factor; nonfinite/nonpositive-factor legs ride the finite contract.

### Queue 43 (45M-22): raw A_s makes the encoded target ~1e9-scale

The shipped configuration uses Cobaya's physical A_s ~ 2.1e-9, so
f = exp(2 tau) / A_s ~ 5e8 at fiducial: the network target carries an
arbitrary 1e9-scale normalization, and the float32 training center
subtracts nearly equal values at that inflated scale. This reads as a
unit-porting defect (a legacy 1e9*A_s ~ 2.1 variable replaced by raw
A_s without the reference normalization). The existing gate only
round-trips the huge factor; it never tests conditioning or the
encoded scale.

Contract: a dimensionless order-one factor
f = (A_s_ref / A_s) * exp(2 (tau - tau_ref)) with f = 1 at the
persisted fiducial; as_ref/tau_ref persisted as RESOLVED
geometry/artifact facts sourced from the covariance fiducial or an
explicit validated configuration (never a code default — the house
doctrine); the corrected convention gets a NEW semantic law version —
an old as_exp2tau artifact is never silently reinterpreted (unit 37's
implementation manifest distinguishes them); affected CMB artifacts
RETRAIN (the constant changes the learned target and weights even
though decode inverts it); the 45M-21 residual division applies
independently — an order-one factor does not excuse the metric fix;
staging reports the target-center and encoded-target scale so a
future unit mismatch is visible before training. Workstation legs:
fiducial (A_s, tau) gives exactly f = 1; primary-amplitude scaling
leaves the law-space spectrum invariant; encoded targets have a
finite documented scale; physical known answers stay correct; an old
unnormalized-law artifact is REFUSED under the new implementation; a
mutation using raw exp(2 tau)/A_s fails the fiducial-unity and
target-scale legs.

#### CMB amplitude-law resume (2026-07-12, Opus) — queue 42 (metric) LANDED; queue 43 PROPOSED, owed

Queue 42 (45M-21, the f^2 metric) is implemented and self-committed on the
branch (batch grant, pending Architect audit). Queue 43 (45M-22, the
dimensionless order-one factor + new law version) is design-sensitive and
requires a retrain, so its layout is PROPOSED below for audit before I
finalize it, and it lands as a second increment.

QUEUE 42 (landed):
- `CmbFactoredChi2.chi2` now DIVIDES the per-row factor out of the whitened
  residual before summing: `r = (pred - target) / f`, `chi2 = sum(r^2)`. It
  REQUIRES params_whitened (explicit from eval, or the value `loss` stashed);
  a plain sum reported `f^2 * chi2_physical` (the old "cancels" claim was
  false -- pred and target of one row share f).
- Roughness law-neutrality: a shared hook `_penalty_residual` (pred - target)
  on CmbDiagonalChi2, overridden on CmbFactoredChi2 to `(pred - target)/f`,
  so the roughness penalty measures the physical residual. `loss` stashes the
  params for both the chi2 and the penalty.
- encode/decode untouched (the defect is the metric, not the invertible
  transform); every "cancels / does not enter the metric" docstring
  corrected (class, chi2, loss).
- cmb-identity `check_law` gains the metric legs: physical-chi2 invariance
  under (A_s, tau) at a fixed physical residual; the uncorrected form's f^2
  catch-power; the factor-corrected roughness residual; chi2-without-params
  raises. Module docstring + board maps updated. (Nonfinite / nonpositive-
  factor legs ride the finite-contract unit.)
- Mac gate: py_compile OK (cmb.py, cmb_identity.py, board.py);
  probe_cmb_factored.py 4/4 on the REAL chi2 + _factor -- corrected chi2 ==
  physical exactly (max|d| 0), old == f^2 * physical, old / corrected == f^2
  exactly, chi2 without params raises. The torch save->rebuild->predict
  round-trip + roughness-in-loss ride the workstation cmb-identity + cmb-smoke
  reruns.
- USER-VISIBLE: delta-chi2, threshold fractions, and best-epoch selection
  change for as_exp2tau CMB runs (the metric is now physical); cmb-identity
  and cmb-smoke RERUN on the workstation.

QUEUE 43 PROPOSAL (45M-22, owed -- Architect confirm before I finalize):
The dimensionless factor f = (A_s_ref / A_s) * exp(2 (tau - tau_ref)), f = 1
at the persisted fiducial. Open design points I want confirmed:
1. NEW law name/version -- a registry entry distinct from "as_exp2tau" (a
   new key and/or a stored law-version string) so an old artifact is never
   silently reinterpreted, with unit 37's manifest carrying the version;
   loading an old as_exp2tau artifact under the new code is REFUSED with a
   named error (the retrain instruction).
2. as_ref / tau_ref PERSISTENCE -- resolved artifact facts, sourced from the
   covariance fiducial or an explicit validated config, never a code default
   (house doctrine). configure_law gains as_ref / tau_ref (required for the
   new law); the geometry persists them (state()/rebuild); _factor reads
   them. CONFIRM the source (covariance fiducial file vs a config block) and
   the exact persistence keys.
3. STAGING report -- the target-center and encoded-target scale printed at
   staging, so a future unit mismatch is visible before training.
4. RETRAIN -- affected CMB artifacts retrain (the constant changes the
   learned target/weights though decode inverts it); workstation, user-run.
5. The queue-42 metric division (landed) applies unchanged.
Red legs (workstation): fiducial (A_s, tau) -> f == 1; amplitude scaling
leaves the law-space spectrum invariant; encoded targets have a finite
documented scale; physical known answers stay correct; an old
unnormalized-law artifact is REFUSED; a mutation using raw exp(2 tau)/A_s
fails the fiducial-unity and target-scale legs.

Files (queue 42): emulator/losses/cmb.py, gates/checks/cmb_identity.py,
gates/board.py, notes/families-scalar-cmb.md.

QUEUE 43 RULINGS (Architect, 2026-07-12 — the proposal above is
CONFIRMED with these bindings; 43 is GO):

1. Version key: a NEW registry law NAME — `as_exp2tau_ref` — and NO
   parallel law-version field. One axis of truth: the law string is
   already persisted, validated (unknown law raises), byte-round-
   tripped, and exposed by rebuild info; a second "version" key could
   contradict the name (the never-trust-defaults failure mode). Unit
   37's implementation manifest carries the new name naturally.
   Loading an artifact whose law is the old "as_exp2tau" under the
   new implementation is REFUSED with a named error carrying the
   retrain instruction (consistent with the adjudicated 45M-22
   contract and the project's retrain-over-compat stance).
2. Fiducial source + persistence keys: the source of record is an
   EXPLICIT validated config pair in the data.cmb block — `as_ref`
   and `tau_ref` — REQUIRED when the law is as_exp2tau_ref, no code
   default, validated (finite; as_ref > 0). The covariance fiducial
   is the documented RECOMMENDED value for the pair; at
   implementation, IF the covariance sidecar already records its
   fiducial (A_s, tau), staging cross-checks and a mismatch is LOUD;
   if it does not, the config is authoritative — do NOT invent a new
   sidecar field for this. Persistence: geometry state keys `as_ref`
   / `tau_ref` (the config spelling), stored as RESOLVED float64 in
   state()/rebuild, byte-round-tripped; _factor READS the persisted
   values with no fallback, and a rebuilt as_exp2tau_ref geometry
   missing either key REFUSES. The artifact records the NUMBERS,
   never a pointer or path.
3. Staging report: confirmed as proposed — one staging verdict line
   with the target-center, the encoded-target scale, and the
   f(fiducial) == 1 check (terminal essential-only discipline
   holds; the full values go to the log).
4. Retrain: confirmed — affected CMB artifacts retrain; workstation,
   user-run, declared USER-VISIBLE.
5. The queue-42 metric division applies unchanged.

Red legs: as proposed, PLUS a refusal leg for a rebuilt state
missing as_ref / tau_ref (the no-fallback proof).

QUEUE 43 COMPLETE resume (2026-07-12, Opus) — committed 4a19a17, cmb-identity green on real torch

Queue 43 self-committed on the branch as 4a19a17 (batch grant, pending
Architect audit). The imposed CMB amplitude law is now the dimensionless
order-one factor f = (A_s_ref / A_s) * exp(2 (tau - tau_ref)), exactly 1
at the persisted fiducial, replacing the retired raw exp(2 tau)/A_s. The
loss-side (registry rename + configure_law refs + reject_retired) was the
pre-(g) WIP reapplied via scratchpad/queue43_cmb.patch; this resume covers
the whole surface the RULINGS scoped.

Delivered exactly to the RULINGS:
- Law name as_exp2tau_ref, NO version field; reject_retired_amplitude_law
  (promoted to public in losses/cmb.py) is the ONE message source, called
  at config build (validate_cmb), h5 rebuild (CmbDiagonalGeometry.from_state),
  and loss build (make_cmb_chi2) — an old as_exp2tau name is refused with
  the retrain instruction at every boundary.
- as_ref / tau_ref: a REQUIRED validated data.cmb pair (finite; as_ref > 0),
  persisted as RESOLVED float64 geometry state (only for the order-one law;
  the none law records none), byte-round-tripped; from_state refuses a
  rebuilt as_exp2tau_ref state missing either (the no-fallback proof); the
  loss reads them via make_cmb_chi2. The artifact records the NUMBERS.
- Staging: one verdict line (_report_cmb_staging) with target center,
  encoded fiducial scale, factor range + f-at-fiducial; the covariance
  cross-check (_cross_check_cmb_fiducial) reads the provenance fiducial IF
  present and flags a drift LOUD (config stays authoritative — no invented
  sidecar field).
- Threading: both training call sites (fresh + finetune pin, the pin now
  matching as_ref/tau_ref too), inference _build_cmb_decoder, and the
  results.py info keys all carry the pair.
- Retrain: USER-VISIBLE, declared. queue-42 metric division unchanged.

Files: emulator/losses/cmb.py, emulator/geometries/cmb.py,
emulator/experiment.py, emulator/inference.py, emulator/results.py,
emulator/losses/pce.py (comment), plus docs (README.md, emulator/README.md,
example_yamls/cmb_emulator.yaml, cmb_train_emulator.py) and gates
(cmb_identity.py, cmb_smoke.py, transfer_identity.py, board.py).

Gate: cmb_identity.py factored-law legs test as_exp2tau_ref — the order-one
_factor bitwise, f == 1 at the fiducial, order-one over the box, byte-exact
reference persistence, the retired-law + missing-reference refusals, and a
raw-factor mutation arm that FAILS both the fiducial-unity (raw f_fid ~
5.3e8) and order-one legs. The WHOLE cmb-identity gate is GREEN on the
Cocoa torch (CPU), including the eq-6 covariance oracle legs. Standalone
probes: probe_q43_geom 15/15, probe_q43_loss_config 18/18 (loss + factor +
validate_cmb), probe_q43_staging 6/6.

Workstation owed (user-run): cmb-smoke rerun (real CAMB + the retrained
target) and any as_exp2tau CMB artifacts retrain under the new law.

Queue order after 43: 14(h) 45M-61 (the diagnostic score boundary, propose
the shared screen_chi2 helper first), then 50(+60+14f) -> 52 -> 55 ->
22(+20) -> 13(+01).


### 45M-27 amendment: lmax validation does not prove multipole coverage (2026-07-12, Architect-VERIFIED; extends unit 26's axis-identity contract to the CMB read/rebuild boundary)

emul_cmb.must_provide (:195-209) validates a requested lmax only
against the artifact's stored MAXIMUM ("an lmax beyond the artifact's
stored range" — its own docstring), and calculate zero-fills a full
array and scatters predictions at the stored ell values only
(:247-249, row[self._ell_arrays[spec]] = predict). Nothing on the
read side proves the stored ell axis is the complete integer sequence
2..lmax: an artifact with ell = [2, 4, 6] passes a request to 6 and
serves zeros at 3 and 5, indistinguishable from predictions; an axis
starting at 10 silently serves zeros for 2-9. The covariance producer
writes the complete grid, so this is guard-only for valid shipped
artifacts — but artifacts are authoritative and weight shape cannot
authenticate coordinate completeness, so the guard is required at
both the training/read boundaries.

Contract: CMB geometry construction AND h5 rebuild require a 1-D
nonempty exact-integer ell axis equal to np.arange(2, ell[-1] + 1) —
no duplicates, gaps, reordering, fractional values, or alternate
start; center/sigma/fiducial_cl/model-output width equal that axis
length; the adapter validates the invariant before building
_ell_arrays so must_provide's lmax check becomes a real coverage
proof; only multipoles 0 and 1 are assembly-zero-filled (the
different-spectra-maxima rule stays as enforced); error text names
the first offending multipole and the artifact root. Red legs: gapped
[2,4]; start-at-3; duplicate; descending; fractional ell in a
hand-built state; axis/data-width mismatch; valid 2..L control; two
spectra with different complete maxima and valid independent
requests; the mutation leg — a same-shaped h5 edit that strict weight
loading accepts but the coordinate guard catches (board-listed,
torch/h5, workstation).

## UNIT 70 (20M-02, 2026-07-13, HIGH): needs_params diagnostics pass the batch's own parameters everywhere — the training stash is private

Finding (red team, CONFIRMED): cmb_residual_diagnostic branches on
needs_params for encode/decode with the validation batch's x_enc
(diagnostics.py:530-535) and then drops it at the chi2 call (:537);
CmbFactoredChi2.chi2 falls back to the mutable self._params
(losses/cmb.py:486) that loss() stashed from the LAST TRAINING BATCH
(:529). Under the shipped as_exp2tau_ref example the diagnostic
scores validation rows with stale training amplitude factors: finite,
nonnegative, and belonging to the wrong cosmologies — worst-row
selection and overlays can invert (analytic two-row control: correct
[3,3], shipped [12,0.75]); mismatched batch lengths crash on shape
instead. cmb_smoke's mean-predictor bar (cmb_smoke.py:394) has the
same parameterless call, so its collapse bar can be calibrated
against stale factors; diagnostics-domain constructs only law="none"
and cannot expose any of this.

Contract (the red team's clauses, ratified):

1. Every needs_params diagnostic branch passes that same batch's
   x_enc to encode, decode, AND chi2.
2. The mutable _params stash is private to the immediate
   inherited-loss reduction (and _penalty_residual inside loss); no
   public diagnostic or gate obtains scientific meaning from it.
3. The cmb-smoke mean-predictor bar uses explicit validation
   parameters.
4. The caller rule applies uniformly across the CMB / grid / grid2d
   diagnostic family, including where today's implementation accepts
   and ignores the argument — capability dispatch means the same
   thing at every call site. A call-site census over
   emulator/diagnostics.py + gates/checks/ is part of the landing.

Legs (board-listed; small-tensor CPU, Mac-validatable): the real
CmbFactoredChi2 two-row analytic control expecting [3,3]; _params
preloaded with different same-shaped factors -> diagnostic invariant;
_params preloaded with a different batch length -> no shape failure;
the same validation rows split across different diagnostic bs values
-> identical per-row scores and worst row; a mutation arm omitting
params_whitened must reproduce [12,0.75] (or the shape crash) and
red; the cmb-smoke bar gains the same stale-cache mutation arm; a
law="none" control proves unchanged numerics. Sequencing: lands with
UNIT 69, parallel to phase-3 population, before queue 2.

### UNIT 70 DONE (Opus, 2026-07-13): the caller rule is uniform, the stash is private, the [3,3]/[12,0.75] control is green

Caller fix (clause 1 + 4): all THREE needs_params residual diagnostics --
cmb_residual_diagnostic, grid_residual_diagnostic, grid2d_residual_diagnostic
(emulator/diagnostics.py:541/764/901) -- now pass THIS batch's x_enc to chi2 in
their needs_p branch, so the physical metric divides by the right amplitude
factor. The bare chi2(pred, target) survives only in each else (non-factored)
branch.

Stash privatized (clause 2): CmbFactoredChi2.loss clears self._params in a
finally, so it is live ONLY during the immediate reduction (the inherited
chi2 + _penalty_residual run inside loss, before the clear). A public caller
that omits params after a loss now hits the loud ValueError instead of the
last batch's stale factor. The docstring says so.

cmb-smoke bar (clause 3): the mean-predictor collapse bar
(gates/checks/cmb_smoke.py) passes params_whitened in the needs_params branch
(workstation-verified with CAMB; compileall clean here). The grid/grid2d smoke
bars (mps_smoke:203, bsn_smoke:193) call encode(dv) paramless -> their chi2fn is
needs_params=False, so the bare chi2 is correct; not the defect.

Legs: cmb_identity gains the analytic control -- two rows with factors f=[2,0.5]
(tau at ref, A_s = as_ref/2 and 2*as_ref), three multipoles, zero prediction,
whitened target = f: params-passing chi2 == [3,3] (physical), the omitted path
reading a stale [1,1]-factor stash == the shipped [12,0.75] defect (the mutation
arm), stash-invariance (byte-identical regardless of the stash), and a
wrong-length stash survives (no shape crash). diagnostics-domain gains an
executable call-site census (>= 3 params-passing residual chi2 calls). The
existing "factored chi2 without params raises" leg still passes.

Verification (Mac, cocoa-torch): cmb-identity ALL green (control factors exactly
[2.0000, 0.5000]; [3.0000, 3.0000] vs [12.0000, 0.7500]); diagnostics-domain ALL
green (census 3/3); compileall clean. Workstation-owed: cmb-smoke's bar rerun +
its stale-cache mutation arm, and one real-Cobaya provider-routed diagnostic
call (queue 5). Paired with UNIT 69 (disjoint files: 69 = emul_mps getters).

## UNIT 71 (20M-03, 2026-07-13, HIGH): emul_cmb serves Cobaya's documented get_Cl protocol — conversions from persisted facts, one startup/runtime verdict

Finding (red team, CONFIRMED; probe: BoltzmannBase.get_Cl defaults
units='FIRASmuK2', so even a default-argument call fails): the
adapter advertises generic "Cl" and must_provide validates only
spectra + l_max, but get_Cl refuses ell_factor=True and any units
but the stored "muK2" (emul_cmb.py:253-282) — bundled real consumers
(Planck low-l EE sroll2: ell_factor=True; ACT DR6 lensing:
units="FIRASmuK2") pass startup and deterministically fail at
evaluation.

RULING: honor the generic contract (option 1). Contract (the red
team's clauses, ratified, with deltas):

1. The generator persists the temperature/unit convention of the
   dumps; the adapter reads it back as an artifact fact (part of the
   shared fixed-facts block, see UNIT 74) and derives unit
   conversions from it — "muK2" is never assumed equal to
   "FIRASmuK2" by default-temperature coincidence.
2. Cobaya's documented unit choices are supported without
   truthiness/coercion; a DEFAULT-argument get_Cl() call succeeds.
3. Spectrum-specific ell factors: l(l+1)/2pi for TT/TE/EE,
   [l(l+1)]^2/2pi for pp; l = 0, 1 behavior explicit.
4. must_provide and get_Cl produce the same capability verdict — no
   startup-green/runtime-red combination.
5. Raw "muK2" output stays byte-identical; a legacy artifact with no
   persisted convention is refused with a migration instruction.

Gates (ratified): raw/muK2 control byte-identical; TT/TE/EE
ell-factor known-answers at several l; pp squared-factor
known-answer; FIRAS conversion known-answer from a persisted
temperature fact; real Planck-low-l consumer lifecycle; real ACT-DR6
getter lifecycle; forged/missing convention refuses before
calculation; mutation arms restoring each current refusal must red.
Conversion legs CPU/NumPy; lifecycle legs board-listed with the CMB
adapter gate (workstation if the rebuild needs Torch). Sequencing:
wave-4 CMB adapter visit; EMUL2-blocking.

## UNIT 72 (20M-04, 2026-07-13, HIGH): scalar outputs publish into Cobaya's derived namespace — never top-level state keys

Finding (red team, CONFIRMED): emul_scalars.calculate writes
state[name] = value for every artifact-defined output name
(emul_scalars.py:222) and validate_scalar accepts any nonempty unique
list (experiment.py:648-651, no string or reserved-name check) — an
output named "derived" crashes on the next line (TypeError), and
"params" / "dependency_params" silently replace Cobaya's cache and
dependency bookkeeping.

Contract (ratified): (1) artifact outputs never become arbitrary
top-level Cobaya state keys; (2) publication goes into
state["derived"] and get_param reads that namespace per the
supported Cobaya lifecycle; (3) state["params"] and
state["dependency_params"] are preserved across calculate; (4)
multi-artifact assembly is atomic — validate and compute into a
local result before touching state, so a failing second predictor
cannot leave the first partially published; (5) at the training
boundary, output names are native nonempty strings checked against
the complete supported-Cobaya reserved-name set (defense in depth;
the namespace is the primary mechanism); (6) error text names the
offending output and why Cobaya owns it.

Gates (ratified): validate_scalar refusal legs for "derived" /
"params" / "dependency_params"; a REAL Cobaya Theory lifecycle per
collision (no hand-built dicts); the ordinary rdrag control;
two-predictor atomicity with an invalid second result; a cache-reuse
control proving the sampled/dependency dicts unchanged; get_param
reads the same mapping Cobaya exposes; a mutation arm restoring
state[name] = value must red. CPU legs; lifecycle board-listed
(workstation if the rebuild needs Torch). Sequencing: wave-4 scalar
adapter visit; EMUL2-blocking.

## UNIT 81 (20M-14, 2026-07-13): the amplitude law's two roles resolve to two columns — validated at config, at readback, and through one shared mapping

Finding (red team, CONFIRMED; witnesses reproduced through the real
classes): as_name / tau_name accept any present column including the
SAME one — validate_cmb requires presence only (:806), configure_law
resolves each with an independent names.index (:392-393), staging
repeats the parallel mapping (:3899-3900), the persisted geometry
never validates the relationship, and rebuild reproduces a same-role
artifact. Aliased roles give finite factors (0.8976275921;
3.8889e-8) that both sides consume consistently, and the
covariance-fiducial cross-check is warning-only.

Contract (ratified): (1) for as_exp2tau_ref, as_name and tau_name
are each nonempty native strings AND different strings; (2) after
resolution their column INDICES must differ (protects future alias
machinery where distinct spellings resolve to one column); (3)
enforced at public config validation before staging; (4) repeated at
artifact readback before the factor is constructed — a saved
same-role artifact is REFUSED, never reproduced; (5) staging and
loss construction consume ONE shared resolved-role helper or one
persisted validated mapping — the parallel names.index definitions
are retired; (6) legitimately distinct aliases resolving to distinct
columns stay allowed; (7) the factor-at-reference invariant is
evaluated through the RESOLVED roles — never the tautology
(as_ref/as_ref)*exp(2*(tau_ref-tau_ref)) — closing the
identity-computed staging banner as well.

Legs (ratified; CPU, board-listed): same-name config refuses in
validate_cmb; same-name direct configure_law refuses; distinct
spellings resolving to one column refuse; a same-role persisted
artifact refuses on rebuild; the correct mapping returns exactly
unity at the fiducial and keeps its non-fiducial known answers; a
valid distinct-column alias control passes; a mutation restoring
membership-only validation reproduces 0.8976275921; a staging/loss
parity leg proves both consume the one resolved mapping.
Sequencing: rides the wave-4 CMB adapter visit with unit 71; the
readback-refusal clause is binding before any CMB production
training.

## UNIT 81 AMENDED (20M-14 addendum, 2026-07-13): roles are semantic, not structural — the fiducial-unity check through resolved roles is the executable proof

The red team's addendum is ACCEPTED: nonempty + native + distinct
names still admit SWAPPED roles or any arbitrary existing pair
(witness at the real fiducial: as_name="tau"/tau_name="As" gives
f = 3.4907e-8; Architect re-derivation exact). Amended contract:
(1) explicit persisted column-role semantics — a canonical role
registry in which as_name must denote a raw linear A_s coordinate
and tau_name an optical depth; aliases legal ONLY when registered to
the scientific role; (2) presence/native/distinct remain necessary,
never sufficient; (3) swapped roles and unrelated pairs (ns /
omegabh2) refuse; (4) the executable semantic check: f evaluated at
the RECORDED fiducial through the RESOLVED roles equals the law's
identity value within its declared numerical contract — this refuses
swaps without the registry having to enumerate every spelling.
Added legs: swapped As/tau; two unrelated existing columns; a valid
registered-alias control; canonical pair at the fiducial gives
exactly f = 1; a mutation validating only presence + index
distinctness must fail.

## UNIT 21 AMENDED (20M-22, 2026-07-13): the published CMB spectra are physically possible — auto spectra nonnegative, the TT/TE/EE triplet positive semidefinite

Finding (red team, CONFIRMED through the real Cobaya lifecycle):
emul_cmb publishes decoded rows with no family-validity check
(:244-250) — finite negative TT/EE/pp and a non-PSD TT/TE/EE triplet
(det = -3) pass the shape/finite boundary and reach every consumer.

Amendment (binding): (1) after decode/shape/finite and BEFORE any
NumPy/state publication, every stored-ell value of TT, EE, and pp is
physically nonnegative — the exact-zero policy for the trained
ell >= 2 ranges is decided and documented; values are NEVER clipped
or absolute-valued; (2) TE remains signed (matching unit 56's
generator-side "signed TE" semantics); (3) where TT, TE, and EE are
jointly present at a multipole, (TE)^2 <= TT * EE within ONE
representation-derived rounding band — the tolerance covers storage
arithmetic only and never blesses a materially impossible
prediction; (4) a failure names the spectrum or triplet, the
multipole, the offending values, and the bound, and leaves NO
partial state["Cl"]; (5) the proof is board-listed in cmb-identity
(Torch/predictor reconstruction). Legs (ratified): one negative arm
per auto spectrum; the finite signed-TE control; joint
equality/below-bound valid controls; the TT=EE=1, TE=2 rejection; a
near-bound rounding control; the partial-state-absence leg; a
mutation reducing the rule to isfinite must red.
