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

<a id="bsn-identity-evidence"></a>
**bsn-identity — synthetic background artifacts exercise the integration
rule, distance construction, grid geometry, saved predictor, adapter, NPCE
residual model, and fine-tune boundary.**

- files: temporary covariance files and synthetic `.h5`/`.emul` artifact
  pairs for Hubble and recombination-distance quantities; no persistent
  science product is written.
- subprocess: `gates/checks/bsn_identity.py`; no nested subprocess, real
  CAMB, or real Cobaya process (the adapter uses a minimal `Theory` stub).
- metric: per-leg — analytic polynomial tolerances for integration,
  tolerance-bounded agreement with a dense path using the same integrator,
  exact saved-state comparisons and typed refusal checks.
- legs: 6, named `bsn-identity.simpson-polynomial-nodes`,
  `bsn-identity.distance-pipeline-consistency`,
  `bsn-identity.geometry-and-artifact-round-trip`,
  `bsn-identity.adapter-piecewise-contract`,
  `bsn-identity.npce-composition`, and `bsn-identity.finetune-parity`.
- evidence: every listed claim is asserted inside the child; the board
  wrapper currently exposes only its aggregate exit code, and the
  distance-pipeline leg is deliberately named as consistency rather than an
  independent numerical reference; no logged-only line is promoted.
- owed: the board registry models CPU PyTorch only. SciPy is an eager child
  import; without it the child is red before reporting any leg, not six
  capability-`UNAVAILABLE` legs. No GPU, real-CAMB, or real-Cobaya evidence is
  claimed.

<a id="bsn-identity-simpson-polynomial-nodes"></a>
`bsn-identity.simpson-polynomial-nodes` compares every node with analytic
constant, linear, quadratic, and cubic integrals, requires the tighter exact
bars where the rule supports them, proves the retired odd-node formula misses
the linear and quadratic answers, and checks the odd-point-count guard.

<a id="bsn-identity-distance-pipeline-consistency"></a>
`bsn-identity.distance-pipeline-consistency` compares the interpolation
pipeline with a very fine grid evaluated by the same cumulative integrator
and requires relative agreement within `1e-6`; it does not claim an
algorithmically independent cosmology calculation.

<a id="bsn-identity-geometry-and-artifact-round-trip"></a>
`bsn-identity.geometry-and-artifact-round-trip` checks the log-offset
transform to float32 tolerance, exact grid-state values, explicit law and
domain refusals, and exact saved predictions and family metadata under both
the log-offset and no-law configurations.

<a id="bsn-identity-adapter-piecewise-contract"></a>
`bsn-identity.adapter-piecewise-contract` checks the two-window artifact
layout, the code-declared exact or tolerance-bounded comparisons with the
same constructed pipeline and interpolator, derived-distance and unit
relations, and loud desert, missing-pair, duplicate-quantity, and
out-of-window refusals using the stubbed adapter.

<a id="bsn-identity-npce-composition"></a>
`bsn-identity.npce-composition` requires a nonzero fitted polynomial base and
checks exact residual encode/decode algebra through the log law plus exact
base-plus-network prediction after save/rebuild.

<a id="bsn-identity-finetune-parity"></a>
`bsn-identity.finetune-parity` accepts a grid source, runs its epoch-zero
warm-start parity check, and rejects metadata and quantity mismatches before
staging.

<a id="bsn-smoke-evidence"></a>
**bsn-smoke — a real-CAMB background fixture is generated, two grid
emulators are trained, the Cobaya provider is compared with CAMB, and the
grid diagnostics path writes its output.**

- files: temporary train/validation parameter tables and covariance files,
  Hubble and recombination-distance dumps with redshift sidecars, two saved
  `.h5`/`.emul` pairs, and a diagnostics PDF; the `$ROOTDIR` work directory
  is removed in `finally`.
- subprocess: `gates/checks/bsn_smoke.py`, which launches
  `dataset_generator_background.py` twice; training, the real Cobaya/CAMB
  model lifecycles, comparison, and diagnostics execute in-process.
- metric: per-leg — process/file checks plus a relative-spread tripwire,
  validation medians below half their staged mean-predictor medians, maximum
  relative CAMB error below `0.02` with a typed desert refusal, and
  diagnostics page/file assertions.
- legs: 4, named `bsn-smoke.generated-background-dumps`,
  `bsn-smoke.training-collapse`, `bsn-smoke.cobaya-vs-camb`, and
  `bsn-smoke.diagnostics-output`.
- evidence: all four claims are asserted in the child; generator exit codes
  are combined with output checks rather than treated as sufficient, and
  failure-stream tails are troubleshooting text only.
- owed: the registry capability lane requires CPU PyTorch and Cobaya. An
  unresolved `$ROOTDIR` aborts preflight; a missing compiled CAMB makes the
  generator assertion fail and leaves later aids unreported; missing plotting
  imports make the diagnostics assertion fail. Those are current red/missing
  evidence outcomes, not capability-`UNAVAILABLE` legs. No prior log
  substitutes for execution.

<a id="bsn-smoke-generated-background-dumps"></a>
`bsn-smoke.generated-background-dumps` requires both generator subprocesses
to exit zero and write both quantities and both redshift sidecars, then
requires every Hubble column to have relative spread greater than `1e-5`
across generated cosmologies.

<a id="bsn-smoke-training-collapse"></a>
`bsn-smoke.training-collapse` requires the Hubble and recombination-distance
trainings each to reduce their best validation median below half the median
score of their own staged mean predictor.

<a id="bsn-smoke-cobaya-vs-camb"></a>
`bsn-smoke.cobaya-vs-camb` runs the real provider lifecycle and requires the
served Hubble, angular-diameter-distance, and recombination-distance arrays
to agree with CAMB's background within maximum relative error `0.02`, while
a query in the unemulated redshift desert must raise.

<a id="bsn-smoke-diagnostics-output"></a>
`bsn-smoke.diagnostics-output` asserts two grid diagnostic pages and a
nonempty diagnostics PDF larger than 10,000 bytes; it does not certify the
scientific interpretation of the plotted curves.

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

## BAOSN physical-domain + pair-shape guards (red-team 2026-07-12 fourth wave, Architect-VERIFIED, open; land before the EMUL2 acceptance)

The inference boundary validates redshift WINDOWS but not physical
values or pair geometry. Verified: emulator/background.py carries no
finite/positivity/monotonicity guard anywhere (distance_interpolators
accepts H = 0 -> all-NaN chi, all-negative H -> negative distances);
emul_baosn.get_angular_diameter_distance_2 (~356-368) documents
z1 <= z2 but enforces nothing — atleast_2d + column reads silently
accept a reversed pair (negative D_A served), ignore a third column,
and die on a one-column row with a bare IndexError.

Contract (Implementer unit; the red-team block of record adopted
whole; guard-only, the ruled cubic/Simpson algorithm untouched):
validate z_grid/h_grid before interpolation (1-D, equal length,
enough points for cubic, finite, strictly increasing nonnegative z,
finite strictly positive H); after the doubled-grid integration
require finite strictly positive integrand and finite nondecreasing
chi(z) with chi(0) = 0; emul_baosn.calculate validates every
predictor row before caching (Hubble row matches its grid + finite
strictly positive; recombination D_M matches its grid + finite
nonnegative increasing; derived products finite); must_provide and
get_angular_diameter_distance_2 require an exact (N, 2) finite pair
array with z1 <= z2 (equal endpoints return zero; never flatten
malformed input before validation); query-point values finite and
physically nonnegative; the desert/window refusals stay. Gate legs:
valid control byte-identical; zero/negative/NaN/Inf Hubble raise
before division; malformed shapes + nonmonotonic grids raise;
nonfinite/decreasing D_M raises; reversed/1-col/3-col pairs raise
naming the (N, 2), z1 <= z2 rule; a same-redshift pair returns
exactly zero. No interpolation-law redesign.

## MPS query/composition totality (red-team 2026-07-12 fourth wave, Architect-VERIFIED, open; land before the EMUL2 acceptance)

PowerSpectrumInterpolator.check_ranges (cobaya_theory/emul_mps.py
~144-164) guards only by </> comparisons: NaN z or k compares False
everywhere and returns NaN instead of raising; an empty query reaches
builtin min() ("min() iterable argument is empty", a non-contract
error). The constructor (~94-114) stores extrap bounds unvalidated:
`if extrap_kmax and extrap_kmax > input_kmax` on a NaN is skipped but
the NaN boundary is KEPT, so a later range check against NaN accepts
any k (verified mechanism: k = 1e6 on a grid ending at 1 is
spline-extrapolated). calculate() checks pk_lin and boost separately
but never their product — two finite positive factors can overflow to
Inf and be cached as the served nonlinear spectrum.

Contract (Implementer unit; the red-team block of record adopted
whole): construction requires nonempty 1-D finite z/k, strictly
positive unique k, a surface of exact (len(z), len(k)) finite values;
extrap limits validated before any log (finite, strictly positive,
kmin strictly below the input minimum, kmax strictly above the
maximum, NaN/Inf/contradictory bounds loud); check_ranges rejects
empty/nonfinite/nonpositive-k queries before min/max/log; pk_nl
requires exact shape + finite strictly positive values before caching
(keep the point-rejection semantics; never cache a partial state);
the logged surface finite before the spline. Gate legs: valid
in-range + valid finite extrapolation controls byte-identical;
NaN/Inf/empty queries raise the PUBLIC range error; zero/negative k
raises before np.log; bad extrap bounds raise at construction;
malformed/duplicate axes + wrong surface shape raise; an overflowing
finite product is rejected leaving NO Pk state keys. Independent of
the queued sigma8 and Cobaya-registration units.

## MPS — the matter power spectrum. ACCEPTED END TO END (board run 9, 2026-07-12: rel 0.93% vs CAMB against the 5% bar); gates mps-identity/mps-smoke.

<a id="mps-identity-evidence"></a>
**mps-identity — synthetic matter-power artifacts exercise two-dimensional
geometry, bounded staging and its temporary-file lifecycle, saved model
variants, adapter assembly, config validation, and fine-tuning.**

- files: temporary raw/base dumps and axis sidecars, temporary staged
  memmaps, covariance files, and synthetic `.h5`/`.emul` artifact pairs;
  lifecycle legs assert removal of the files they supersede or release.
- subprocess: `gates/checks/mps_identity.py`; no nested subprocess, real
  CAMB, real Syren formula evaluation, or real Cobaya process (the adapter
  uses stubs, including synthetic base functions).
- metric: per-leg — exact transform, state, and composition comparisons,
  NumPy-relative floating-point comparisons for streamed
  statistics, bounded read and filesystem-state assertions, installed-API
  signature equality, and typed refusal checks.
- legs: 7, named `mps-identity.geometry-laws-and-pins`,
  `mps-identity.bounded-staging-values`,
  `mps-identity.stable-streamed-moments`,
  `mps-identity.staging-file-lifecycle`,
  `mps-identity.saved-model-variants`,
  `mps-identity.adapter-assembly-and-defaults`, and
  `mps-identity.config-and-finetune`.
- evidence: the child forms its independent law rows in float64, converts the
  rows to the stored float32 representation, then accumulates the reference
  mean in float64. The producer's center is array-equal to that reference.
  The production-width fixture also proves that the former mean-before-cast
  order differs by `5.960464478e-08`. A direct Cocoa Torch 2.6.0 CPU run
  reaches `PASS: mps-identity all checks green`. The board wrapper currently
  exposes only the aggregate child exit code, and the synthetic assembly leg
  is not evidence for the real Syren formulas. Diagnostic print lines inside
  a failed config check remain logged-only.
- owed: Architect audit of the Red Team repair and queue-2 wiring of the
  existing `mps-identity.bounded-staging-values` logical aid. The new
  mean-before-cast mutation stays inside that same leg. The registry models
  only CPU PyTorch: missing SciPy can crash the dynamically loaded adapter,
  while a missing installed Cobaya base API reaches explicit protocol
  failures. Those are red or missing evidence rather than seven
  capability-`UNAVAILABLE` legs. Real Syren/CAMB and GPU behavior remain
  outside this gate.

<a id="mps-identity-geometry-laws-and-pins"></a>
`mps-identity.geometry-laws-and-pins` checks float32 transform round trips,
exact grid-state values, width/law guards, exact constant-column pins under
both laws, and refusal of a wholly constant surface.

<a id="mps-identity-bounded-staging-values"></a>
`mps-identity.bounded-staging-values` compares stored row values and their
float64-accumulated mean with an independent float32-payload reference. It
also checks positivity refusal, bounded reads, disk/RAM selection, the
whole-selection mutation and a mean-before-cast mutation that must disagree.

<a id="mps-identity-stable-streamed-moments"></a>
`mps-identity.stable-streamed-moments` compares streamed centers and
population standard deviations with NumPy over uneven chunkings and large
offsets, checks the relative constant-pin boundary, and checks that
from-stats encoding matches materialized-law encoding.

<a id="mps-identity-staging-file-lifecycle"></a>
`mps-identity.staging-file-lifecycle` checks that restaging supersedes the old
temporary file, a three-point sweep keeps at most one live file and releases
it, failures unlink partial files, and the resident-memory control creates no
temporary file while preserving values exactly.

<a id="mps-identity-saved-model-variants"></a>
`mps-identity.saved-model-variants` checks exact saved predictions for Syren-
linear and no-law artifacts, correction-head attachment/phase behavior and
save/rebuild equality, plus exact NPCE residual algebra, state, saved
prediction, and rejection of the unsupported ratio form.

<a id="mps-identity-adapter-assembly-and-defaults"></a>
`mps-identity.adapter-assembly-and-defaults` checks linear/nonlinear
composition against synthetic base stubs, low-wavenumber blending, grid and
interpolator readback, the omitted nonlinear argument against Cobaya's
installed default signature, requirements containing the Syren-base names,
point rejection, and pair/quantity/grid refusals; it does not execute the real
Syren formula implementation.

<a id="mps-identity-config-and-finetune"></a>
`mps-identity.config-and-finetune` checks the current grid2d pairing and
transfer config cases, accepts a grid2d fine-tune source and its epoch-zero
parity check, and rejects a metadata mismatch before staging.

<a id="mps-smoke-evidence"></a>
**mps-smoke — a real-CAMB law-none fixture is generated, linear and boost
emulators are trained, the Cobaya provider is compared with CAMB, and the
grid2d diagnostics path writes its output.**

- files: temporary train/validation parameter tables and covariance files,
  linear-power and boost dumps with redshift/wavenumber sidecars, two saved
  `.h5`/`.emul` pairs, and a diagnostics PDF; the `$ROOTDIR` work directory
  is removed in `finally`.
- subprocess: `gates/checks/mps_smoke.py`, which launches
  `dataset_generator_mps.py` twice; training, the real Cobaya/CAMB model
  lifecycles, comparison, and diagnostics execute in-process.
- metric: per-leg — process/file/shape/positivity checks, validation medians
  below half their staged mean-predictor medians, maximum relative CAMB error
  below `0.05` plus a range refusal, and diagnostics shape/page/file checks.
- legs: 4, named `mps-smoke.generated-power-dumps`,
  `mps-smoke.training-collapse`, `mps-smoke.cobaya-vs-camb`, and
  `mps-smoke.diagnostics-output`.
- evidence: all four claims are asserted in the child; generator exit codes
  are combined with output checks rather than treated as sufficient, and
  failure-stream tails are troubleshooting text only.
- owed: the registry capability lane requires CPU PyTorch and Cobaya. An
  unresolved `$ROOTDIR` aborts preflight; a missing compiled CAMB makes the
  generator assertion fail and leaves later aids unreported; missing plotting
  imports make the diagnostics assertion fail. Those are current red/missing
  evidence outcomes, not capability-`UNAVAILABLE` legs. The real Syren-law
  hybrid is not executed by this law-none smoke gate.

<a id="mps-smoke-generated-power-dumps"></a>
`mps-smoke.generated-power-dumps` requires both generator subprocesses to
exit zero and write the linear-power and boost arrays plus both axis
sidecars, then checks the expected linear-dump shape and positivity of every
nonfailed row.

<a id="mps-smoke-training-collapse"></a>
`mps-smoke.training-collapse` requires the linear-power and boost trainings
each to reduce their best validation median below half the median score of
their own staged mean predictor.

<a id="mps-smoke-cobaya-vs-camb"></a>
`mps-smoke.cobaya-vs-camb` runs the real provider lifecycle and requires
served linear and nonlinear power to agree with CAMB within maximum relative
error `0.05`; an out-of-range interpolator query must raise, but the exception
type is not constrained by the current child.

<a id="mps-smoke-diagnostics-output"></a>
`mps-smoke.diagnostics-output` asserts the expected grid2d diagnostic array
shapes, one to three redshift slices, two pages, and a nonempty PDF larger
than 10,000 bytes; it does not certify the scientific interpretation of the
plots.

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

## REOPENED: the odd-node "cumulative Simpson" is mathematically wrong (red-team 45M-12, 2026-07-12, Architect-VERIFIED by direct arithmetic; queue 36 — CRITICAL, first code unit after the finite contract)

The recorded legacy acceptance above ("kept bug-for-bug, only shades
interpolation", with an O(dz^3) error claim) is REOPENED and its error
claim is FALSIFIED. emulator/background.py::cumulative_simpson
computes each odd node as
`C[i] = C[i-1] + dz/6 * (y[i-1] + 4*y[i] + y[i+1])` (:85) — that is
HALF the two-interval Simpson total, not the one-interval integral.
Architect reproduction on the exec-extracted shipped function: for
y = z the odd-node error is EXACTLY h^2/2 at every h tested
(5.000e-3, 1.250e-3, 3.125e-4 at h = 0.1, 0.05, 0.025) — first
order, not O(h^3); for y = z^2 the shipped value is 4h^3/3 vs truth
h^3/3. The correct one-interval integral through the same three
samples, h/12 * (5*y[i-1] + 8*y[i] - y[i+1]), reproduces both
exactly. Even nodes are exact on cubics (composite Simpson, correct).
The damage is served: distance_interpolators fits a cubic through the
WHOLE doubled grid including the wrong odd values, so arbitrary-z
queries interpolate a contaminated distance curve. The prior
acceptance rested on the false error-order label and on a gate
(bsn_identity.py:87-88, e_odd < 1e-3) that encodes the bug as
tolerance; "legacy" is provenance, not a scientific justification —
the reopen is adopted whole.

Contract (Implementer):
1. Replace the odd-node increment with h/12 * (5*y[i-1] + 8*y[i]
   - y[i+1]) (one authoritative implementation, convention pinned in
   the docstring).
2. Even-node values preserved (already composite Simpson).
3. Module docstring, READMEs, gate description, and this note's
   recorded-finding paragraph corrected: the O(dz^3) claim removed,
   the reopening recorded, history kept (do not erase the original
   acceptance — supersede it).
4. Known-answer legs at EVERY node (constant, linear, quadratic,
   cubic): even nodes exact, odd nodes near machine precision on
   constants/linears/quadratics.
5. bsn-identity revised: the current dz/6 (1,4,1) odd form must FAIL
   the linear and quadratic legs by a wide margin (mutation
   catch-power); the (5,8,-1)/12 form passes near machine precision;
   the e_odd < 1e-3 acceptance is retired.
6. The BAOSN served-value comparison is RERUN on the workstation —
   this deliberately changes valid predictions between the original
   grid nodes. USER-VISIBLE: served BAOSN numbers change; flagged in
   the landing report.

#### Odd-node Simpson resume (2026-07-12, Opus) — landed, self-committed (batch grant)

Unit 36 implemented and self-committed on the branch (batch grant, pending
Architect audit).

- `emulator/background.py` `cumulative_simpson`: the odd node is now the
  correct one-interval integral `dz/12 * (5*y[i-1] + 8*y[i] - y[i+1])`;
  even nodes unchanged (composite Simpson). The docstring is rewritten —
  the O(dz^3) / bug-for-bug acceptance superseded, the reopen recorded,
  the original mislabel kept in quotes as history.
- Docs corrected (the "verbatim legacy / half-chunk approximation" labels
  replaced with the one-interval-exact-on-quadratics description): the
  module flow diagram, the bsn-identity gate docstring (both `board.py` and
  `bsn_identity.py`), and `emulator/README.md` (two rows).
- bsn-identity `check_simpson` revised: known-answer integrals at EVERY node
  for constant / linear / quadratic (even AND odd machine precision) and
  cubic (even exact, odd bounded); a mutation control (`_old_odd_simpson`,
  the retired (1,4,1)/6 form) that must be wide of machine precision on the
  linear and quadratic legs; the `e_odd < 1e-3` acceptance retired; the
  even-point-count guard kept.
- Mac gate (the REAL function, numpy): py_compile OK (background.py,
  board.py, bsn_identity.py); probe_simpson.py 9/9 — the new form is
  machine-precision on constant/linear/quadratic (~1e-14) and the cubic odd
  node bounded (3.1e-10); the old (1,4,1)/6 form is wide on linear
  (1.25e-5) and quadratic (7.5e-5); the guard raises; the spec known-answers
  reproduce (new C[1] = 5e-3 exact; old error exactly h^2/2).
- USER-VISIBLE (item 6): served BAOSN distance numbers change between the
  original grid nodes (the contaminated odd values are corrected); the
  BAOSN served-value comparison must be RERUN on the workstation.

Files (unit 36): emulator/background.py, gates/checks/bsn_identity.py,
gates/board.py, emulator/README.md, notes/families-background-mps.md.

## Syren silently reflects out-of-domain physics (red-team 45M-14, 2026-07-12, Architect-VERIFIED by live reproduction; queue 38)

The vendored production path contains local "prevent NaN" edits that
change the fitted mathematics instead of validating its domain:
linear.py:31 log(abs(arg) + 1e-10) (and :186/:226/:232/:373-375),
syrenhalofit.py:317 sqrt(abs(radicand)). abs is not a guard — it maps
the invalid side of a fit onto the valid side, and the epsilon turns
an exact domain boundary into a finite number instead of a refusal.
Architect reproduction on the REAL vendored code (Mac, numpy): at
sigma8 1.2, Om 0.1, Ob 0.05, h 0.5, ns 0.8, a 0.1 the vectorized
C_emulated_vec radicand is -0.04754964, reflected to a plausible
finite output 0.11591803 — while the scalar sibling C_emulated hits
"invalid value in sqrt" (NaN) at the same point. The two
implementations disagree exactly where the domain boundary matters.

Contract (Implementer): (1) document the calibrated domain of every
active syren fit argument (cosmology ranges + required signs of
log/sqrt expressions); (2) validate BEFORE evaluating — a nonpositive
log argument or negative radicand fails naming the expression,
cosmology, redshift, and allowed range; (3) remove silent abs/epsilon
continuation from the production path unless a primary formula source
defines it — an extended formula, if wanted, gets its own persisted
base-version name; (4) scalar and vectorized paths agree throughout
the accepted domain; (5) generator setup validates its whole
configured prior + z grid inside the domain BEFORE MPI launch;
inference validates each requested point; (6) the chosen base variant
is recorded in the 45M-13 implementation manifest (queue 37).
Acceptance legs (numpy-pure gate; the full MPS confirmation stays on
the workstation board): scalar/vector agreement on a valid-domain
grid; each log boundary and the -0.04755 radicand example fails
before returning a spectrum; a mutation restoring the abs form fails
the negative-domain leg; no accepted base value NaN/Inf; the shipped
production prior/grid proven wholly inside the domain or rejected
with the exact offending corner.

### 45M-26 amendment to unit 15 (BAOSN physical-domain + pair-shape guards): startup accepts a Hubble request the getter refuses (2026-07-12, Architect-VERIFIED)

must_provide (emul_baosn.py:214-236) sends EVERY product — including
Hubble — through the uniform _check_windows union rule (SN window OR
recombination window), while get_Hubble serves the SN grid only and
loudly refuses beyond it (its docstring says so; the getter enforces
it). So must_provide(Hubble={"z": [1090]}) succeeds at Cobaya startup
and get_Hubble([1090]) later raises — a deterministic late failure
the identity gate cannot see (it tests valid low-z Hubble at startup
and high-z refusal only through the getter, never crossing the arms).
Amendment to unit 15's contract: ONE product-specific domain helper
used by BOTH must_provide and the getters — Hubble: finite query
wholly inside the SN/H grid; the three distance products: each query
inside the union of the two windows; angular_diameter_distance_2
keeps the already-queued exact (N,2)/finite/ordered contract with
both endpoints individually serviceable; startup and runtime return
the SAME verdict and the SAME domain explanation for an identical
request — no accepted-now-refused-later path; the loud desert rule
and current valid numerics preserved. Red legs: Hubble at
recombination refused by BOTH arms; Hubble in the desert and above
both windows; each distance product valid in each window; desert
refusal; mixed valid/invalid; exact boundary equality at z = _sn_max;
the malformed-pair cases from the existing unit.

### 45M-09 amendment to unit 16 (MPS query/composition totality): the spline boundary is under-validated and alien-Python; teach the power-law extension (2026-07-12, Architect-VERIFIED; one implementation, one gate suite — no duplicate unit)

cobaya_theory/emul_mps.py::PowerSpectrumInterpolator (vendored from
cobaya's boltzmannbase, Antony Lewis attribution — KEEP it, and note
the divergence) opens with the lazy-generator tuple unpack
`z, k = (np.atleast_1d(x) for x in [z, k])` and packs validation,
sorting, permutation, log conversion, two-sided extrapolation, and
the SciPy call into one body. Verified gaps: len(z) >= 4 is checked
but len(k) is NOT (the bicubic RectBivariateSpline default needs
four points on BOTH axes — a short k dies inside FITPACK);
P_or_logP's (n_z, n_k) shape is never required before the chained
advanced indexes P[i_z, :][:, i_k]; axes are silently sorted but
duplicate / nonfinite / nonpositive k are never rejected (duplicates
reach FITPACK, zero/negative reach np.log); z/k/surface never
required finite; logP/logsign carry no value schema; extrap_kmin is
missing from the class parameter docs though it changes the served
domain; the truthiness extrapolation guards are the NaN hole this
unit already records; the 0.1/0.9 pair (:99/:112 — node placement
log(edge)*0.1 + log(extrap)*0.9 AND the tail value delta*0.9) is an
unexplained algorithmic control; super().__init__ leans on SciPy's
hidden default degrees; grid=True (Cartesian product) vs grid=False
(paired coordinates) is never stated though it changes output shape.

Amendment (rides unit 16 whole — validation + didactics in ONE
implementation): (1) all numerical checks join the totality
contract — nonempty 1-D finite axes, >= 4 UNIQUE points on BOTH
axes, finite surface exactly (n_z, n_k), strictly positive k,
duplicate rejection, typed logP, valid logsign, fully validated
finite extrapolation bounds before any log or SciPy call;
(2) unsorted valid axes stay supported — explicit named sort
indexes, surface permuted along the matching axes, stated to return
copies never mutating caller arrays; (3) the generator expression
becomes named eager steps (z_values = np.atleast_1d(z), ...);
(4) the two tail extensions are extracted/separated with every
quantity named: the two edge samples estimating the log-space
slope, the requested endpoint, the artificial interior node, the
fraction locating it, the extended surface;
(5) EXTRAPOLATION_INTERIOR_FRACTION = 0.9 named once, the 0.1
DERIVED as its complement; (6) the algorithm taught correctly: a
straight-line continuation of log P vs log k — a power law in
ordinary P and k — never "adding two points"; (7) grid semantics,
logP/logsign, input_kmin/max vs served kmin/max, and why self.k
stays the original grid while the internal log-k axis extends, all
stated; (8) the SciPy API and its default cubic-degree requirement
named before the call; (9) valid-input numerics PRESERVED — a
didactic and validation rewrite, never a replacement of the
inherited algorithm. CPU/SciPy gate legs (no torch), riding unit
16's suite: valid sorted grid reproduces the current interpolator at
nodes and interior queries; valid unsorted grid matches after paired
permutation with caller arrays untouched; four-point boundary
controls on both axes with three-z and three-k raising the PUBLIC
error before FITPACK; wrong rank/shape, duplicates, NaN/Inf,
zero/negative k raise diagnostically; both extrapolation fixtures
match an independent analytic log-linear continuation at both
artificial nodes and the endpoint; grid=True Cartesian shape vs
grid=False paired results with scalar behavior named and tested;
invalid/NaN limits and empty/nonfinite queries exercise the adopted
totality contract; a leftover scan proves the generator expression,
the magic 0.1/0.9 pair, and the incomplete parameter docs are gone.

## Pointer: syren alias-consistency amendment (45M-45)

The syren_params_from precedence documented above (As_1e9 over As, w
over w0) is AMENDED by unit 7's 45M-45 alias-consistency boundary:
dual names must agree (As_1e9 == 1e9 * As; w == w0) or the reader
raises naming both — silent preference is retired as a correctness
policy. Spec + red legs: artifacts-inference-warmstart.md ("UNIT 7
AMENDED (45M-45)").

## UNIT 16 AMENDED (45M-54, ninth batch): the mps-smoke range leg must certify the refusal contract, not "any raise"

The final range leg of the board-listed mps-smoke
(gates/checks/mps_smoke.py:394-398) is

    try:
        lin.P(0.25, 50.0)
        report("interpolator range guard", False, "no raise")
    except Exception:
        report("interpolator range guard", True, "raised")

— ANY exception is accepted as proof of a correct refusal, and none
of the other three boundaries (low k, low z, high z) is exercised at
all. The leg has zero catch power of its own: an interpolator whose
refusal path raises the wrong class (a scipy internal error, a
corrupt-spline RuntimeError, an attribute error from a refactor)
greens it.

The contract it should certify is vendored in this repo and precise:
PowerSpectrumInterpolator.check_ranges
(cobaya_theory/emul_mps.py:144-165) raises cobaya's LoggedError with
"Not possible to extrapolate to k=... 1/Mpc (maximum k possible is
... 1/Mpc)." (and the z / minimum analogues) — the coordinate, the
requested value, and the stored limit are all named — and both P
(:167) and logP (:176) call it before evaluating. np.allclose slack
applies at the boundary, so refusal probes must sit clearly outside
it.

Architect precision ruling on the mutation arm: a P that raises for
EVERY input already reds the whole gate today — the lifecycle leg
computes got_lin = lin.P(z_probe, k_probe) inside the same try and
reports "cobaya lifecycle through emul_mps" FAIL — so the
always-raising mutant is non-discriminating at gate level (the red
team's LEG-level false green stands as stated). The discriminating
mutant is an interpolator healthy in range whose refusal raises the
WRONG class: today's leg PASSES that mutant; the amended leg must
RED it.

The amendment (all inside mps-smoke's check_cobaya; no production
code change; workstation-owed like every torch/cobaya leg):

1. In-range scalar control FIRST: lin.P at an interior (z, k)
   returns a finite value (the earlier grid-path lifecycle call is
   not a substitute for the scalar path the refusal legs use).
2. Four independent refusal legs — k below kmin, k above kmax, z
   below zmin, z above zmax — probe values derived from the object's
   own stored bounds (lin.z[0], lin.z[-1], lin.kmin, lin.kmax),
   placed a factor ~1.05 beyond the limit (outside allclose slack).
3. Catch LoggedError ONLY (from cobaya.log import LoggedError);
   assert the message carries the offending coordinate token ("k="
   or "z="), the requested value, and the stored limit (the two
   formatted numbers must appear in the text).
4. Any other exception class = the leg goes RED reporting the
   exception type and message.
5. Mutation arm: the wrong-class mutant above must red the amended
   legs; the always-raising variant is recorded as covered by the
   lifecycle leg (both documented in the check).
6. This stays inside the existing board-listed mps-smoke; no new
   gate, no new production code.

## UNIT 58 (45M-55, ninth batch): BAOSN distances integrate through an untrained redshift interval — the SN grid must start at z = 0

CONFIRMED (Fable, 2026-07-12, live reproduction with the REAL
module). The generator schema and the distance pipeline contradict
each other by construction:

- dataset_generator_background.py:102-104 REQUIRES 0 < zmin (a
  zero-starting SN grid is refused today); its docstring example is
  z_sn: [0.001, 3.0, 600] (:42).
- emulator/background.py comoving_distance_grid builds the integral
  from exactly z = 0: interp1d(z_grid, c/H, kind='cubic',
  fill_value="extrapolate") (:121-124) evaluated on
  zstep = linspace(0.0, z_grid[-1], 2*NZ + 1) (:125) — every point
  of [0, zmin) is cubic EXTRAPOLATION through an interval no H value
  was ever generated or trained on. The docstring blesses it ("the
  grid need not start at 0 — the interpolation extends c/H to 0
  exactly as the legacy did"): a legacy-verbatim convention hereby
  adjudicated WRONG (the 45M-12 cumulative-Simpson precedent — a
  faithful port of an error is still an error).
- distance_interpolators wraps H ITSELF in the same extrapolating
  interp1d over z_grid (:156-166), so get_Hubble serves untrained
  H(z < zmin) directly — a second serving surface.
- The adapter hardcodes the lower window bound: _check_windows uses
  z >= 0.0 (emul_baosn.py:249) and get_Hubble refuses only z < 0.0
  (:325), while _sn_max is read from the persisted grid (:155). The
  advertised window [0, z_max] is never proven against the persisted
  first node.

Reproduction (the real emulator/background.py imported by file path
under the cocoa clone's python, scipy 1.12.0; truth via scipy quad;
analytic flat constant-w model
H(z) = 70*sqrt(0.1(1+z)^3 + 0.9(1+z)^(3(1+w))), w = -1.5):

    z_sn = [1, 3, 600]:   served chi(1) = 4614.90 Mpc vs truth
                          4521.66 -> +2.06%; served H(0.5) = 63.5065
                          vs 63.6730 -> -0.26%, finite and positive
    z_sn = [2, 3, 600]:   served chi(2) = 8626.36 Mpc vs truth
                          7753.45 -> +11.26%; served H(1.0) -1.14%
    z_sn = [0.001, 3, 600] and the board fixture [0.01, 3, 120]:
                          bias < 1e-3% — shipped-scale zmin is NOT
                          materially wrong; the bite is the OPEN
                          SCHEMA CLASS, reachable at the validator
                          boundary today

(the red team quoted +3.02% for zmin = 1 on an unstated grid spec;
the zmin = 2 figures agree to 0.1% and the direction and magnitude
class agree everywhere — the contract does not hang on the exact
figure). The extrapolated H stays finite, positive, and monotonic,
so unit 15's physical-totality guards are BLIND to this by design:
15 certifies values, 58 certifies that the integration/declaration
DOMAIN is actually trained. Distinct units, same home note, land
together.

Contract (six clauses):

1. Generator schema: z_sn must start at exactly zero — require
   z_sn[0] == 0.0, replacing the current 0 < zmin rule (z_rec keeps
   its existing rule; the desert check is unchanged).
2. comoving_distance_grid REQUIRES z_grid[0] == 0.0 and drops
   fill_value="extrapolate" (bounds_error=True): it never
   establishes the lower integration boundary by extrapolating
   through an untrained interval. distance_interpolators' H wrapper
   gets the same boundary honesty. Docstrings corrected (the "need
   not start at 0" sentence is retired WITH this record of why).
3. The adapter refuses legacy Hubble artifacts whose persisted grid
   starts above zero, at LOAD time (where z_sn is read from the
   artifact, before distance_interpolators at :279 and before any
   getter), with a migration message naming the artifact, the
   persisted z[0], and the regeneration/retraining requirement.
   emulator/diagnostics.py:642 inherits the same protection through
   the artifact refusal.
4. The declared window comes from the persisted grid: [0, _sn_max]
   may be advertised only after proving the first node is exactly
   zero.
5. Migration (dataset/artifact break, declared): the generator
   docstring example [0.001, 3.0, 600] -> [0.0, 3.0, 600]; the
   board's bsn-smoke config z_sn: [0.01, 3.0, 120]
   (bsn_smoke.py:97) -> [0.0, 3.0, 120]; the SIX 0.001-starting
   bsn_identity fixture grids (:145, :164, :226, :308, :414, :515)
   -> zero-starting; existing artifacts must be regenerated and
   retrained; served-number change at shipped-scale zmin is below
   1e-5 relative, declared here.
6. The cubic interpolation and the corrected (45M-12) cumulative
   Simpson are UNTOUCHED on valid zero-starting grids — this finding
   authorizes no quadrature redesign.

Gate legs (inside the existing board-listed bsn-identity +
bsn-smoke):

- zero-starting analytic H controls retain the expected distances
  (known-answer values recomputed on the migrated fixtures);
- grids starting at 0.001, 0.5, and 1.0 are REFUSED by
  comoving_distance_grid (and the adapter-load path for a persisted
  z[0] > 0), never extrapolated;
- mutation arm: the old extrapolating construction on the
  z_sn = [1, 3, 600] fixture must reproduce the wrong finite
  distance (+2.06% is the recorded known answer);
- artifact reload with z[0] > 0 fails BEFORE serving (constructor,
  not first query);
- the real-CAMB smoke regenerates its fixture with z_sn[0] = 0 and
  keeps comparing off-center distances.

Placement: fourth wave, lands WITH unit 15 (same surfaces, same
gates), BEFORE the EMUL2 acceptance. USER-VISIBLE: schema break (old
z_sn configs refused loudly) + artifact migration (old artifacts
refused loudly); served numbers at shipped-scale zmin change
imperceptibly (< 1e-5 relative).

## UNIT 62 (45M-62, fourteenth batch, 2026-07-12): background metadata schema — producer/consumer units unification

Finding (red team, CONFIRMED live): validate_grid (experiment.py
~:826-905) requires the data.grid.units KEY but never checks its
VALUE — reproduced through the REAL validator (cocoa python,
2026-07-12): units = 'bananas', True, None, and 3.14 are ALL
accepted and returned in the grid dict, while validate_grid2d in
the SAME FILE refuses a wrong units string by name (:967-972,
Mpc3/dimensionless per quantity). GridGeometry.__init__ then
str()-coerces and persists (grid.py:95, True -> "True"), and
from_state (:130-138) rebuilds with no check either. The ONLY
value check in the program is the public consumer's:
emul_baosn.initialize demands the exact pair Hubble -> "km/s/Mpc",
D_M -> "Mpc" (emul_baosn.py ~:128-142) and refuses the artifact.
Concrete wrong outcome: an expensive, otherwise-successful
training run saves a schema-v2 pair the intended adapter can never
serve — the producer and consumer implement different metadata
schemas. Transfer/finetune equality checks (:2277, :3793) enforce
only SAMENESS against a base, so a bananas base propagates.

Contract (red team's six clauses adopted, one narrowing):

1. ONE shared background-quantity registry beside TARGET_LAWS in
   emulator/geometries/grid.py: quantity -> its units string
   (Hubble -> km/s/Mpc, D_M -> Mpc). validate_grid AND
   emul_baosn.initialize read the SAME registry (the adapter
   already imports emulator.inference / emulator.background at
   emul_baosn.py:46-47 — a direct import, no vendoring problem).
   NARROWED (Architect): the registry does NOT constrain laws per
   quantity — log_offset is legitimate for either quantity (the
   shipped docstring recommends none for D_M without forbidding
   log_offset) and no evidence shows any law pairing wrong; the
   defect is units.
2. validate_grid requires the exact registry pair BEFORE staging
   any data or constructing torch objects — the mirror of its
   grid2d sibling in the same file.
3. No str() coercion anywhere on the units surface: the value must
   be a plain str equal to the registry string; bool/int/float/
   None/any other type is a schema error naming the received type
   (the probe proved every one of them currently rides through to
   persistence).
4. GridGeometry.__init__ and from_state validate (quantity, units)
   against the registry too, so a forged or corrupted artifact
   cannot bypass the YAML validator; refusal at load, never at
   first query.
5. Shipped configs byte-identical (baosn_hubble_emulator.yaml is
   already Hubble / km/s/Mpc; the gate D_M fixtures already Mpc);
   NO numeric change anywhere — a metadata-only unit.
6. The error names the received (quantity, units, law) tuple and
   the allowed tuple(s), and says WHY: units is an artifact fact
   the public consumer dispatches on.

Red legs (inside the existing board-listed bsn-identity; the
artifact/rebuild legs are torch-backed):

- valid Hubble and D_M controls (registry pairs accepted
  end-to-end);
- Hubble/Mpc, D_M/km/s/Mpc, and Hubble/bananas refused at config
  validation naming the allowed pair;
- non-string units (True, None, 3.14) refused as type errors,
  never stringified;
- forged geometry state with a mismatched pair refused on
  from_state;
- saved valid artifact rebuilds and is ACCEPTED by the real
  emul_baosn initialize (the producer/consumer-agreement leg);
- mutation arm: restore the units-blind validator while the
  adapter stays strict — the gate must fail BECAUSE producer
  acceptance and consumer acceptance differ (the agreement leg is
  what catches it).

Adjacency: distinct from units 15/58 (they certify the numerical
redshift domain; this unit certifies the metadata is consumable at
all). Placement: fourth wave, lands WITH units 15+58 (one
background visit, same gate homes), BEFORE the EMUL2 acceptance.
USER-VISIBLE: configs with wrong or non-string units are now
refused loudly (previously they trained to an unservable
artifact); valid configs and all numerics unchanged.

## UNIT 63 (45M-63, fourteenth batch, 2026-07-12): the grid2d constant pin must prove its science, not just zero variance

Finding (red team, CONFIRMED live): D-MP9's carve-out was argued
on boost physics (B = 1 below the nonlinear scale, so the low-k
law-space columns are constant under any law — gates-and-board.md
runs 7-8) but implemented value-, quantity-, and coordinate-BLIND.
Grid2DGeometry.from_stats (grid2d.py ~:196-226) pins EVERY
sufficiently-constant column (scale <= 8*eps32*|center|), sets its
scale to 1, and decode (:340-341) permanently serves the training
constant there. Reproduced through the REAL geometry (cocoa
python, 2026-07-12):

- pklin/none with ONE stale column -> pinned; decode serves
  12345.6 at that (z, k) for EVERY input — a cosmology-independent
  linear power, which cannot exist in the sampled A_s/shape space.
  A partial generator failure served as physics; reachable at the
  validator boundary (validate_grid2d allows law "none" for either
  quantity).
- the mps_identity pin leg (:175-197) itself pins an arbitrary
  boost constant of 7.0 under BOTH syren_halofit and none —
  neither the raw identity B = 1 nor the residual identity 0 — so
  the current gate BLESSES the corruption class.
- a FORGED const_mask on a varying pklin column is accepted by
  from_state (the constructor checks only numel, :119-127) and
  decode then serves the training mean regardless of input.

Distinct from unit 11 / 45M-49 (representability decides whether a
column IS constant in storage; this unit decides whether that
constant is PERMITTED by the science schema) and from the
whole-surface dead-dump guard (which stays).

Contract (red team's adopted, one precision ruling):

1. A pin is legal ONLY when all three gates pass:
   (i) quantity == "boost" — any constant pklin coordinate is a
   loud partial-dead-dump error naming (z, k), quantity, law, and
   the stored value;
   (ii) the stored center matches the law's identity within a
   documented stored-dtype tolerance — raw none boost: 1; syren
   residual space: 0. The tolerance is float32-eps-derived with
   its derivation recorded, and (the band precedent) may only
   WIDEN on measured valid evidence;
   (iii) the pinned set is a LOW-K region: per z-row the pinned
   columns form a contiguous prefix from the lowest k (the physics
   that justified D-MP9 — B = 1 BELOW the nonlinear scale); a
   pinned column with an unpinned lower-k column in the same z-row
   is refused naming both coordinates. A constant boost outside
   the allowed region, or at the wrong value (7, -3), is a loud
   error. If the REAL production/smoke dumps violate strict
   prefix-ness at the nonlinear boundary, the Implementer reports
   the measured pattern BEFORE weakening the region rule — never
   silently.
2. PRECISION RULING (Architect) on the persist-the-policy clause:
   NO new artifact field and NO policy/version key — the queue-43
   no-version precedent applies. Legality is recomputable from
   facts the artifact ALREADY persists (quantity, law, z, k,
   center, const_mask), so readback VALIDATES const_mask against
   them in the constructor — the single home both from_stats and
   from_state flow through — and a forged mask cannot pin
   arbitrary science. A second trusted axis (a version integer
   beside the mask) would itself be forgeable and adds nothing the
   persisted facts do not carry.
3. The whole-surface dead-dump refusal stays, for every law.
4. Pre-pin artifacts (const_mask None) stay legal; the VALID
   existing boost pin and every nonconstant valid numeric are
   byte-identical.
5. The mps_identity 7.0 legs are REPLACED with the replacement
   recorded in the check (the run-10 precedent: delete the stale
   expectation and say why): the valid-pin legs use the identity
   values inside a low-k region; the 7.0 shape moves to the
   refusal legs.

### 25M-17 amendment (Red Team CONFIRMED, awaiting Architect adjudication): deleting the mask dataset silently turns off a valid low-k pin

Unit 63 validates the science of a `const_mask` only when the dataset is
present. `Grid2DGeometry.state()` omits the key when no pins exist
(`grid2d.py:270-283`), the generic HDF5 writer/reader persists only keys that
happen to exist (`results.py:306-319,563-578`), and the constructor treats a
missing key as `const_mask=None` (`grid2d.py:108-130`). Decode clamps the
low-k identity only when the mask is non-None (`grid2d.py:340-342`). Thus
schema v2 still infers this scientific behavior from dataset presence.

Delete only `dv_geometry/const_mask` from a valid pinned artifact. The class
marker, model recipe, weights, center/scale, quantity/law, and axes remain
valid; strict model loading succeeds; from_state selects the unpinned branch.
For a `boost/none` point with persisted center 1, stored scale 1, and a
deterministic network value 0.25, the intact artifact serves exactly 1 while
the deleted-key artifact serves 1.25. This `none`-law control is the exact
public catch-power witness. A Syren law also changes before its low-k blend,
but the final public ratio is blend-weighted and is deliberately not quoted as
`exp(0.25)`. This is a finite wrong science result, not merely a malformed-file
exception.

This reopens one precision clause of unit 63: “pre-pin artifacts with
`const_mask=None` stay legal” makes intentional absence indistinguishable
from erasure. Keep the no-policy-version ruling—the mask itself is the
scientific fact, not a version integer—but current saves must always persist
it, using an all-false mask for an explicitly unpinned geometry. The current
schema declares the exact required geometry-state member set and validates it
before `from_state`; anonymous legacy absence refuses with a re-save/migration
instruction rather than guessing. Unit 63 still owns whether a present mask
is scientifically legal; unit 96/general artifact-state authenticity owns
whether the declared fact was erased or spuriously added.

Board-listed MPS artifact legs (Torch/HDF5, workstation): valid masked pin;
valid explicit all-false unmasked state; delete the mask from declared masked;
add/toggle it against declared unmasked; legacy anonymous absence refuses;
unit-63 forged-mask science checks still red. The discriminating mutation
restores today's omit-when-None/presence inference and must rebuild the
`1 -> 1.25` wrong-result witness.

Red legs (inside the existing board-listed mps-identity;
torch-backed geometry/readback legs):

- pklin/none with one constant column raises naming
  (z, k)/quantity/law/value;
- pklin/syren_linear with one zero-residual column raises;
- low-k boost/none == 1 and low-k boost/syren_halofit == 0 pin AND
  round-trip (the valid D-MP9 case preserved bit-for-bit);
- low-k boost constants 7 and -3 raise;
- high-k boost constant at the otherwise-correct identity value
  raises (the region gate);
- forged artifact state with a mask on pklin or on high-k boost
  raises on rebuild;
- whole-surface constant still raises;
- mutation arm: restore the current quantity-blind
  scale <= tiny pin — the gate must fail.

Note: the red team cited decode at :366; the torch.where site
verifies at :340-341 at HEAD — line drift only, mechanism
identical. Placement: fourth wave, lands with the MPS visit
(beside unit 16's mps-smoke amendment; this unit's legs live in
mps-identity), BEFORE the EMUL2 acceptance. USER-VISIBLE: dumps
with stale constant columns now refuse loudly at geometry build
(previously they trained "green" and served corrupted science);
existing valid boost artifacts rebuild unchanged.

### Unit 63 reopen implementation readback (Red Team, 2026-07-13)

The Wave-5 implementation claim is bounded to the grid2d geometry, its one
pin-count banner in `emulator/experiment.py`, plus a pure CPU test file.
`state()` now always writes `const_mask` as uint8 zeros and ones. The unpinned
state is an all-false array of length `nz*nk`. `from_state()` requires the
member and gives an explicit re-save instruction when it is absent. Direct
construction may still pass `None`; the constructor converts that value
immediately into the same all-false in-memory mask, so any subsequent save is
current-schema. Omitting the constructor argument is a `TypeError`, so absence
cannot select the unpinned policy through either construction path.
The banner counts true entries and stays silent for an all-false mask, which
preserves the former unpinned output.

The reopened state boundary checks the representation before decode can
consume it. The accepted storage types are boolean and uint8. The mask must be
one-dimensional, have exactly `nz*nk` entries and contain only zeros and ones.
The original unit-63 quantity, identity-value and low-k-prefix clauses remain
outside this reopen and are unchanged. No policy or version field was added.

The command
`PYTHONPATH=. ../cocoa/Cocoa/.local/bin/python -m unittest -v
tests.test_grid2d_const_mask` runs five CPU tests. They prove a required direct
constructor argument, an explicit all-false unpinned round-trip, an existing
two-redshift low-k pin round-trip, shape/type/binary-value refusal and the
key-deletion mutation. In
the mutation witness, the intact `boost/none` pin serves exactly `1.0`; the
retired presence-inferred branch serves `1.25`. Current `from_state` refuses
the deleted key before that wrong value can be served.

The existing `mps-identity` child is intentionally not edited in this branch
because its file was owned by the concurrent `25M-36` repair when this work
started. The owning gate must absorb the three persistence legs and update its
seven-key preamble after that file becomes quiet. The package README now states
the current always-present mask behavior. Its wider scheduled teaching visit
remains separate. This is implementation evidence for the Architect's audit,
not Red Team certification.

After the `25M-36` repair landed on current main, its unmodified complete CPU
child was rerun against this branch's geometry code. It reports eight state
keys and ends `PASS: mps-identity all checks green`. That pass covers the
existing checks only. It does not contain the three new persistence and
missing-key legs, so it cannot close this reopen.

The three board-listed persistence and missing-key legs are now unblocked on
the quiet `mps-identity` file and remain the final integration step. The
add-or-toggle-against-declared-unmasked case remains the unit-96
artifact-authenticity interlock. This branch does not edit
`emulator/results.py` or claim that wider artifact proof.

## UNIT 62 EXTENDED (45M-66, sixteenth batch, 2026-07-12): log_offset totality — a +Inf offset passes both guards and builds an all-NaN background target

Finding (red team, CONFIRMED live): validate_grid requires
data.grid.offset under log_offset but checks neither type nor
finiteness (experiment.py:857-864) — the probe accepted
offset = +Inf, NaN, "inf", and True through the REAL validator
(NaN is one beyond the red team's list). from_targets then
coerces float(offset) (grid.py:177) and the poisoning chain
clears BOTH guards on ordinary finite Hubble rows: shifted =
targets + Inf = Inf; the positivity guard passes (Inf > 0);
log(shifted) = Inf; center = Inf; the population std is NaN; the
degeneracy guard is scale <= tiny and NaN <= Inf is False, so
zero bad columns are reported. Reproduced through the REAL
from_targets: center [inf, inf], scale [nan, nan], encode of the
training rows themselves all-NaN — and the poisoned state
ROUND-TRIPS through from_state (the constructor validates
nothing it loads). A setup-boundary defect: unit 14's
finite-training contract catching the poisoned loss later is
defense in depth, not acceptance.

Contract (the red team's six clauses adopted; folded here per
their own no-second-mechanism clause):

1. data.grid.offset must be an explicit finite non-bool real; no
   string coercion (a quoted "inf" is a schema error naming the
   received type — the unit-62 units-clause pattern).
2. from_targets independently validates the resolved offset, the
   law-space rows, and the computed center/scale as FINITE before
   any comparative guard runs (comparative guards are undefined
   on NaN — the exact evasion this finding proves).
3. The law-domain check reports SEPARATELY: invalid offset;
   nonpositive target+offset (naming the grid coordinate);
   nonfinite law transform; unrepresentable/zero stored scale.
4. Accepted finite offsets preserve numerics byte-for-byte.
5. Persisted/rebuilt geometry applies the same finite contract —
   a forged nonfinite offset/center/scale refuses at load, before
   prediction (the probe proved today's from_state accepts all
   three).
6. NO second generic finite mechanism: the finite checks use the
   shared stored-integrity mechanism unit 11 establishes; this
   extension owns only the background law-domain semantics and
   messages.

Red legs (CPU validator/NumPy discrimination legs; the geometry
round-trip/rebuild legs board-listed under bsn-identity):

- +Inf, -Inf, NaN, bool, and quoted numeric offsets fail at
  validation;
- a bypassed +Inf offset fails inside from_targets before any
  geometry is returned;
- a finite offset with a nonpositive shifted target fails naming
  the offending grid coordinate (existing behavior, now a pinned
  leg);
- a finite-offset control round-trips decode(encode(y));
- forged artifact state with a nonfinite offset, center, or scale
  fails on rebuild;
- mutation arm: retain ONLY the current positivity and
  scale <= tiny comparisons — must fail (NaN evades both).

Placement: rides unit 62 (the background value-schema unit) in
the wave-4 background visit with 15+58; same gate home.
USER-VISIBLE: poisoned offsets refuse at config/build (today they
train an all-NaN target); valid runs byte-identical.

## UNIT 2 EXTENDED (45M-67, sixteenth batch, 2026-07-12): sigma8 domain totality — no z relabeling, no partial top-hat integrals, no unconditional advertisement

(The sigma8 half of ledger entry 2 — dataset readiness + MPS
sigma8 — is specified here with the MPS family; the R = 8 Mpc vs
8 Mpc/h USER RULING recorded in entry 2 was RESOLVED on
2026-07-13 — R = 8 Mpc/h, the conventional definition; see
"USER RULING (2026-07-13)" below. This extension supplies the
domain-totality contract, not that resolution.)

Finding (red team, CONFIRMED live, numbers reproduced
digit-for-digit): emul_mps advertises sigma8 unconditionally
(get_can_support_params) and _compute_sigma8 has two independent
domain holes. (1) z relabeling: the nearest stored redshift
within 0.01 of z_eval is used UNCHANGED (:487-493) — P(k, 0.009)
is served labeled z = 0; under the toy growth law P ~ e^{-2z}
that is a deterministic 0.896% bias (e^{-0.009} = 0.99104); a
grid starting above 0.01 instead hits SciPy's out-of-domain
interp1d error. The generator validator admits either grid
(z_segments requires only ascending, >= 4 points). (2) k-domain
truncation: the integral runs over whatever stored k interval
exists (:495-505) while k_log10 validates ONLY lo < hi and
nk >= 8 (dataset_generator_mps.py:128-130). With the EXACT
shipped integration expression on the smooth positive toy
spectrum P(k) = k/(1+(k/0.2)^4): the reference grid 1e-4..100
gives 0.0049304012; the validator-admitted 8-point grid 1..10
gives 0.0000884680 — reported/reference = 0.01794, a 98.2%
SILENT underestimate. Not a tolerance issue, and independent of
the open radius/unit ruling.

Contract (the red team's seven clauses adopted):

1. Folded HERE — no second sigma8 implementation.
2. NEVER substitute a nearby redshift. Serving sigma8 at z = 0
   requires an exact stored z = 0 row; otherwise the product is
   unavailable/refused naming the stored range (the physical grid
   is nonnegative, so interpolation cannot manufacture z = 0
   either).
3. sigma8 is advertised/registered ONLY when the loaded
   artifact's axis contract supports it (the Implementer proposes
   the cobaya-correct hook — initialize/must_provide — at build).
4. z/k axes, P shape, finiteness, positivity, and exact z
   ownership are validated BEFORE integration.
5. k-domain completeness is certified by a documented
   convergence/omitted-tail criterion tied to the top-hat
   integrand (never nk or guessed endpoint constants); the
   certification FACTS persist with the generator/file-set
   manifest (never-trust-defaults: resolved facts, not a
   boolean); the criterion's derivation is recorded,
   propose-first.
6. Integration in float64; the final radicand/result validated
   (the ledger already records a negative-radicand case at
   extreme parameters).
7. The shipped wide-grid result is preserved subject ONLY to the
   separately queued physical-radius correction — which still
   awaits the USER RULING.

Red legs (NumPy/CPU for the numerical-domain legs; the cobaya
registration + provider-comparison legs board-listed in
mps-identity / mps-smoke for the workstation):

- a grid containing exact z = 0 returns the direct float64
  reference;
- an otherwise-identical grid beginning at z = 0.009 is REFUSED,
  not relabeled;
- a grid beginning above 0.01 receives the same early domain
  refusal, never a SciPy bounds error;
- narrow low-k-only, high-k-only, and 1..10 grids fail the
  completeness proof despite finite positive spectra;
- a converged wide grid passes; extending it further moves the
  result only within the recorded tail tolerance;
- mutation arm: restore the nearest-within-0.01 branch — fails;
- mutation arm: completeness reduced to nk >= 8 — fails on the
  98.2%-wrong example;
- the corrected sigma8 is compared with the underlying Boltzmann
  provider at a known cosmology on the workstation (mps-smoke).

Placement: unit 2 (dataset readiness + MPS sigma8), where queued;
the sigma8 code work naturally batches with the MPS family visit
but the unit's number and order do not change. USER-VISIBLE:
sigma8 refusals on unsupporting grids (today: silently biased or
98%-wrong values); the radius ruling remains the user's.

## UNIT 65 (BLOAT-03, RT-2026-07-13-01, 2026-07-13): one adapter mechanics owner — device, compile, options, and root expansion shared across the five Cobaya adapters

Finding (red team, CONFIRMED): the five adapters duplicate their
mechanics. _pick_device is verbatim-identical at least in
emul_cmb.py (:167-185) and emul_scalars.py (:171-189), with variants
in the others; torch.compile coercion, option handling, and
ROOTDIR-relative root expansion repeat per file. A device- or
compile-policy fix currently needs five edits.

Contract:

1. One shared mechanics module under cobaya_theory/ (importable by
   all five adapters) owning: device resolution (the cuda/mps/cpu
   fallback ladder), torch.compile coercion and policy, unknown /
   mistyped option refusal, and ROOTDIR-relative path expansion.
2. The five Theory classes REMAIN separate, explicit owners of their
   family lifecycle (initialize / must_provide / calculate / getters).
   NO parameterized adapter superclass (ratified from the handoff):
   shared helpers, not shared inheritance.
3. Sequencing: lands WITH the typed adapter contract — after the
   wave-4 adapter visits (units 15 + 58 + 62 background; 16 + 63
   MPS) establish what the typed boundary validates, so the shared
   module is written once against the final contract, not twice.
4. Acceptance: adapter behavior byte-identical on valid configs;
   refusal messages may only gain precision; the board-listed adapter
   identity gates rerun green.

## USER RULING (2026-07-13): sigma8 radius = 8 Mpc/h — the conventional definition

Asked and answered 2026-07-13 (during the Architect texnotes-gap
triage): the derived sigma8 the MPS adapter serves uses the
CONVENTIONAL top-hat radius R = 8 Mpc/h, not the legacy R = 8.0 with
k in 1/Mpc that _compute_sigma8 ships today. This changes
legacy-served values ON PURPOSE — the BSN-curvature precedent applies
(dimensionally or conventionally wrong legacy math is not
reproduced). No renamed legacy product and no dual serving: one
product, one convention, named sigma8. Implementation rides the
wave-4 MPS visit (units 16 + 63 + the entry-2 sigma8 half, under
UNIT 2 EXTENDED's domain-totality contract); the guide's Current-gap
paragraph (texnotes/emulator_code_guide.tex ~:3885) is then
RED-TEAM-owed — only the red team edits the guide (USER RULE
2026-07-13; custody rule in conventions-and-workflow.md).

## UNIT 67 (RT-2026-07-13-05, 2026-07-13, CRITICAL): flat-only is one enforced fact — curvature refused at generation AND consumption, from the global model

Finding (red team, CONFIRMED; math independently verified): the
flat-only refusal inspects only the emulator input names
(emul_baosn.py:161 — "omk" in req), so a fixed or
separately-consumed GLOBAL Cobaya omk bypasses it: the red team's
real-Cobaya composition returned bit-identical adapter distances for
omk = 0 vs 0.1 while CAMB's distances changed. The producer
compounds it: dataset_generator_background.py:346-347 requests
get_comoving_radial_distance (chi) and stores it as D_M — equal only
in a flat model — with the assumption recorded in a comment only,
and no curvature guard exists anywhere in compute_data_vectors (the
only omk enforcement in the tree is compute_cmb_covariance's
LCDM_FIXED_ONLY). The counterexample is exact: at z = 1100,
Omega_k = 0.1, chi = 13296.826 Mpc vs D_M = 15538.408 Mpc — 16.858%;
the Architect's independent check confirms D_M = R sinh(chi/R) with
R = c/(H0 sqrt(Omega_k)) reproduces their D_M at H0 = 70.0 exactly.

Contract:

1. ONE explicit flat-only fact governs both ends. Generation: the
   background generator REFUSES a sampled OR fixed nonzero curvature
   read from the GLOBAL Cobaya model info — not from emulator input
   names — before any sampling starts. Consumption: emul_baosn
   refuses nonzero curvature from the global model composition the
   same way; the existing names-based refusal may stay as a cheap
   early layer but is no longer THE guard.
2. The artifact records the flat-only fact it was generated under;
   the consumer reads it from the artifact (never-trust-defaults),
   so a future curvature-capable artifact is distinguishable from a
   flat-only one at load time.
3. Legs: real-Cobaya fixed omk != 0 refusal; sampled-omk refusal;
   the omk = 0 flat control byte-identical to today; the independent
   chi-vs-D_M curved counterexample (analytic sinh mapping,
   Mac-runnable; the CAMB cross-check rides the workstation board);
   mutation arm restoring the names-only guard must FAIL (a global
   fixed omk = 0.1 composition slips through it).
4. Producer physics for the flat case unchanged — byte-identical
   dumps on flat configs.
5. Sequencing: rides the wave-4 background visit (units 15 + 58 +
   62 + 67 — same files, one visit); the CRITICAL flag means the
   visit cannot close without it.

## UNIT 69 (20M-01, 2026-07-13, HIGH): the MPS getters serve Cobaya's public default — nonlinear=True when the argument is omitted

Finding (red team, CONFIRMED; Architect probe on the installed
cobaya 3.6.2): BoltzmannBase.get_Pk_grid and get_Pk_interpolator
default nonlinear=True; emul_mps declares nonlinear=False on both
(emul_mps.py:411-412, :426-428) while promising CAMB/Cobaya-
compatible getters. A likelihood that legally omits the argument
receives the LINEAR spectrum labeled as its default product. Both
stored branches are correct when the argument is spelled; every
existing gate leg spells it, so the board is green around the defect.

Contract (the red team's clauses, ratified):

1. Both getters adopt Cobaya's nonlinear=True default.
2. Explicit nonlinear=False and nonlinear=True stay byte-identical
   to today's linear and nonlinear branches.
3. A protocol-guard gate leg pins both adapter signatures (parameter
   names + defaults) against the INSTALLED Cobaya base signatures,
   so an upstream default drift reds for review instead of being
   copied silently.
4. The adapter docstrings name the omitted-call behavior: nonlinear
   is the public default; callers request the linear spectrum
   explicitly.

Legs (board-listed in the existing MPS gates): omitted
get_Pk_grid() == explicit True and != explicit False on a real
calculated state; the same three-arm comparison through
get_Pk_interpolator at stored nodes and one interior point; a
sentinel state with deliberately separated linear/nonlinear values
proves catch power; a mutation arm restoring nonlinear=False must
red; one real-Cobaya provider-routed call proves the protocol, not
just direct method calls. Sequencing: lands with UNIT 70 as one
increment, parallel to phase-3 population, before queue 2.

## UNIT 73 (20M-05, 2026-07-13, HIGH): emul_mps implements must_provide — one capability verdict at startup and runtime

Finding (red team, CONFIRMED): emul_mps subclasses Theory with NO
must_provide (class at :193; only get_requirements at :330), so it
inherits the accept-everything base: unservable P(k,z) requests
(z outside the artifact domain, ("Weyl","Weyl") pairs, off-node grid
redshifts, excessive k_max) pass startup and fail — or mis-serve
under Cobaya's include-the-requested-nodes grid contract — at
evaluation. ACT DR6's optional Limber path is a real bundled
consumer of the Weyl pair.

Contract (ratified): explicit must_provide on the supported Cobaya
schema, without mutating the caller's mapping; native finite
types only (exact supported variable pair, native booleans, finite
1-d z requests, finite positive k_max); every interpolator z inside
the validated domain; every grid z present on the returned axis or
explicitly resampled — in-range coverage alone is not the grid
contract; ONE explicit k_max policy (stored support or a documented
validated extrapolation capability); repeated requirements
accumulate as the union of z / pairs / branches and the max k_max;
startup and runtime share one capability helper with one
verdict/reason (the BAOSN one-verdict law, now program-wide); the
base lifecycle is called so changed requirements invalidate stale
states.

Gates (ratified): real-Cobaya linear-only / nonlinear-only /
combined lifecycles; Weyl refused during must_provide with zero
predictor calls; z below/above domain refused at startup; off-node
grid request exercises the include-or-refuse policy; excessive k_max
exercises the declared policy; repeated requests merge; the caller
dict is unchanged; deleting the override must red; the ACT-DR6
Limber requirement is refused at startup with a teaching error or
supported end to end — never accepted then failed. Schema legs CPU;
provider lifecycle board-listed in the MPS gate. Sequencing: wave-4
MPS adapter visit; EMUL2-blocking.

## UNIT 74 (20M-06, 2026-07-13, CRITICAL): fixed cosmology facts are artifact identity — persisted at generation, compared against the global model before serving

Finding (red team, CONFIRMED): the shipped EMUL2 YAML fixes
mnu: 0.06; a fixed parameter is absent from predictor.names, the
adapter requests only the sampled names + the five syren params
(emul_mps.py:253-292), syren_params_from returns seven values
(syren_base.py:39-90), and the base functions keep their hidden
mnu=0.06 default (:94). A global change to mnu=0.12 therefore serves
an EXACTLY unchanged spectrum (red team live repro: 0.0 relative
difference) while the real base is 6.76% sensitive on the same grid
— finite/positivity guards cannot see the substitution. The general
class: any network trained at a fixed mnu / w0 / wa / curvature
carries a correction specific to that cosmology while the artifact
records only sampled columns.

Contract (ratified, with the unifying delta): (1) generation
persists every non-sampled cosmology fact that changes the target or
the analytic base — at minimum fixed mnu, w0/w, wa, curvature,
noncanonical radiation/temperature facts, and the neutrino
convention — in ONE shared "fixed scientific facts" artifact block
(the same block carries UNIT 71's temperature convention and UNIT
67's flat-only fact; producer side coordinates with units 37 + 62);
(2) the MPS artifact pair agrees on those facts and on base
implementation identity before either predictor serves; (3) the
adapter obtains the GLOBAL resolved values and compares them against
the artifact before evaluation — sampled-name equality is not a
substitute; (4) a mismatch raises at startup naming artifact value,
requested value, and the remediation; (5) a sampled fact that is an
artifact input validates through the ordinary domain contract and is
NOT also pinned; (6) the mnu=0.06 syren baseline stays explicit —
generalizing it is a new producer/consumer law identity, not an
inference keyword; (7) legacy artifacts without fixed-fact identity
are refused with a migration instruction.

Gates (ratified): fixed-0.06 control byte-identical; real-Cobaya
fixed mnu=0.12 refused before predictor/base execution; a fabricated
fixed-0.12 manifest accepted only at 0.12; the sampled-mnu path
accepted only with mnu a stored input and matching law identity;
fixed non-LCDM w0 and nonzero wa mismatch legs; artifact-pair
disagreement on any fixed fact; a mutation checking only
predictor.names must red; an independent base/CAMB comparison proves
the chosen fact scientifically active. Manifest/schema legs CPU; the
end-to-end comparison board-listed in mps-smoke (workstation if
Torch/CAMB is needed). Sequencing: wave-4 MPS adapter visit;
EMUL2-blocking; the producer clause lands with the generator-side
schema work, coordinated so the consumer never reads a fact the
producer does not yet write.

## UNIT 75 (20M-08, 2026-07-13, HIGH): the MPS pair proves one scientific domain — axis equality is not provenance

Finding (red team, CONFIRMED): emul_mps.initialize accepts any two
grid2d artifacts labeled pklin + boost with equal stored z,k arrays
(:215-280), unions their predictor.names into one requirement
mapping (:231-232), and multiplies their outputs (:357-394); the
in-code comment that matched axes prove "they come from one
generator run" (:267, :279) is false. Probe: an LCDM pklin (no w)
composed with a w-carrying identity boost initializes, publishes w
to Cobaya, and serves the w = -1 linear surface at w = -0.5 — 74.5%
maximum relative deviation from the real vendored base, finite and
positive throughout.

Contract (ratified; extends the artifact-pair integrity campaign —
no new publication mechanism):

1. Both MPS products persist a canonical generator / dataset /
   scientific-domain binding (carried in the SAME fixed-facts block
   as units 71/74), and equality is REQUIRED before Cobaya
   requirements are published.
2. Compatible input schemas are proven, not manufactured: same named
   parameter coordinates under the canonical mapping, same domains,
   same fixed cosmology facts. The requirement union NEVER creates
   compatibility a pair does not have.
3. Mismatches refuse before either predictor serves.
4. Valid same-run pair numerics stay byte-identical; swapped
   artifact list order is valid (quantity labels identify the pair,
   not list position).

Legs (ratified): same-run pklin+boost passes; equal axes with a
different dataset digest refuses; w present in only one artifact
refuses; differing parameter order proves canonical equivalence or
refuses; differing fixed facts refuse; differing generator manifest
refuses; swapped root order passes; an axes-only mutation (the
current check) must FAIL the gate; a real valid-pair calculate
control pins today's output. Schema legs CPU; the real-artifact
lifecycle leg board-listed (workstation if Torch is needed).
Sequencing: wave-4 MPS adapter visit with units 73 + 74;
EMUL2-blocking.

### 25M-13 amendment (Red Team CONFIRMED, awaiting Architect adjudication): the BAOSN adapter also manufactures pair compatibility by unioning incompatible inputs

`emul_baosn.initialize` loads one Hubble and one D_M artifact, unions every
predictor input name into one Cobaya requirement mapping, and checks only
quantity/units plus redshift-window layout (`cobaya_theory/emul_baosn.py:120-168`).
`calculate` then evaluates the two artifacts independently (`:277-285`). It
never proves that the pair belongs to one generator/dataset/scientific domain
or that their sampled/fixed coordinate contracts are compatible.

Public real-artifact CPU reproduction on schema v2: the Hubble artifact used
inputs `[omegam,H0,w]`; the D_M artifact used `[omegam,H0]`. Initialization
accepted and published the union `{omegam,H0,w}`. Changing `w` from `-1` to
`-0.5` changed served H(z=1), while served D_M(z=1050) remained bit-identical
at `13999.394531250002`. An independent flat-wCDM integral changed from
`13672.42999` to `12646.12951`, a `7.506%` discrepancy. The adapter therefore
returns a finite stitched background whose low- and high-redshift pieces
describe different cosmologies.

This is the BAOSN instance of unit 75's ratified rule that requirement union
never creates compatibility. It shares unit 74/67's fixed-facts and flat-only
schema; it does not create another manifest mechanism.

Required contract: before publishing requirements, the Hubble and D_M
artifacts must have equal canonical generator/dataset/scientific-domain
binding, compatible named sampled coordinates and domains, and identical
fixed facts. Compatible ordering is canonicalized explicitly; a parameter
sampled by only one half is a refusal, never added to a union that the other
half ignores. Root list order remains irrelevant because quantity labels own
the pair.

Red legs: a same-run Hubble+D_M pair passes with current numerics pinned;
sampled `w` on only one half refuses; equal windows but different dataset or
fixed `w` refuses; differing fixed facts refuse; differing parameter order is
proved canonically equivalent or refused; swapped roots pass; and a mutation
restoring union-only compatibility reproduces the bit-identical D_M under
changed `w` and must red. Schema legs are CPU; one real-pair calculate control
is board-listed for the workstation if Torch is required.

## UNIT 85 (20M-18, 2026-07-13, HIGH): one canonical dark-energy resolver — sampled w0pwa never silently degrades to wa = 0

Finding (red team, CONFIRMED): syren_params_from supplies wa = 0.0
when absent (syren_base.py:54, :88) and understands no w0pwa; the
adapter requests only artifact names + the five syren params and
never derives wa — while the GENERATOR computes through Cobaya's
resolved input mapping (w0pwa=-0.5, w=-1 -> wa=0.5). The stored
correction therefore belongs to wa = 0.5 while the served analytic
base runs at wa = 0: the shipped-adapter probe with a perfect
zero-residual pklin + identity boost initializes, returns True,
serves finite positive spectra, and misses the real vendored base by
up to 12.987%. The sampled-coordinate sibling of unit 74's fixed
mnu lane in the same function; unit 7's alias rule does not apply
(w0pwa is a different COORDINATE requiring arithmetic, not a
spelling).

Contract (ratified): (1) ONE canonical dark-energy resolver shared
by generator and adapter; (2) it accepts explicit (w or w0, wa) or
sampled (w0pwa, w or w0) and computes wa = w0pwa - w0; (3) when all
three are present, w0pwa == w0 + wa must hold under one documented
representation tolerance before canonicalization; (4) the w/w0
alias equality (unit 7) applies BEFORE the coordinate relation; (5)
the adapter's requirements guarantee enough inputs to resolve the
artifact's exact parameterization — wa is NEVER defaulted to 0 when
w0pwa is present; (6) the absent-all LCDM case stays legal only as
an explicit persisted fact under unit 74; (7) the resolved
dark-energy parameterization/role identity is PERSISTED (a member of
the shared fixed-facts block) so generator and consumer cannot
choose different coordinate laws; (8) a mismatch or underdetermined
mapping refuses before predictor or base runs, leaving no partial
Pk state; (9) explicit wa = 0 behavior byte-identical.

Legs (ratified; CPU, board-listed): the real Cobaya to_input control
(w0pwa=-0.5, w=-1 -> wa=0.5); the shipped-adapter zero-residual
known answer against the vendored base at nonzero wa; explicit
(w, wa) and transformed (w0pwa, w) forms produce one base; the
wa = 0 control byte-identical; consistent and inconsistent
all-three mappings; missing information refuses before prediction;
pair artifacts declare one coordinate law; a mutation restoring
"absent wa means zero even when w0pwa is present" reproduces the
12.987% miss and reds. Placement: the shared Syren
parameter-resolution contract, coordinated with units 74-75 in the
wave-4 MPS visit; EMUL2-blocking.

## UNIT 85 ADDENDUM (20M-18 addendum, 2026-07-13): publicly reachable via the shipped EMUL2 YAML; served-error magnitudes; the spy-gate leg

The shipped cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml already
carries the defect's trigger: sampled w0pwa (dropped), the dynamic
bridge wa = w0pwa - w with derived: false, and a single evaluation
point at w0pwa == w (wa = 0) that MASKS the omission — any point
with w0pwa != w inside the published priors (e.g. w = -0.99,
w0pwa = -0.89 -> wa = 0.1) exercises it. Independent full-grid
magnitudes on the real vendored base at wa = 0 vs the generator's
wa = 0.1: 1.7774% (P_lin), 3.0239% (combined pklin x boost within
z<=2, k<=10, where the low-k blend weight is numerically one — the
FINAL served nonlinear error under a perfect network), 6.8977%
(z<=10, k<=100), 121.5% (full z<=50, k<=100 grid).

Contract refinements folded in: wa-under-w0pwa is a DYNAMIC
per-point fact resolved by unit 74's consumer-side mechanism — it is
never pinned as a fixed fact; unit 7 remains the shared
alias/coordinate resolver applied first. Added leg (binding): a
real-Cobaya gate on the PUBLIC w0pwa YAML at a nonzero-wa point with
a perfect-residual predictor SPIES the canonical Syren tuple
received at generation and at serving and requires equality;
removing the wa requirement must reproduce the miss. The unit-7
record sentence ("full calculate(**params) mapping") is corrected in
artifacts-inference-warmstart.md — Cobaya routes only
required/supported inputs.

## UNIT 85 REACHABILITY CORRECTION (2026-07-13): the silent lane is the non-drop configuration — the shipped drop YAML is startup-red, a separate defect in scope

Superseding this note's earlier ADDENDUM claim of shipped-YAML
reachability (the red team retracted it after forward-walking real
Cobaya routing): with w0pwa: {drop: true}, a component requiring
w0pwa FAILS at model construction — the shipped EMUL2 example is
startup-red for any artifact that stores w0pwa and is NOT a working
example today. The silent-wrong-result lane is the valid NON-drop
configuration, proven at full standard through a complete real
Cobaya model (real adapter, vendored Syren, deterministic artifacts,
real likelihood via the provider): theory assigned exactly the seven
sampled names, logposterior served a finite positive spectrum at the
wa = 0 base, 0.1298745470923877 max relative error vs the requested
wa = 0.5. What stands unchanged: the producer/consumer fork (the
generator's full to_input mapping carries wa; calculate receives
only declared requirements), the Syren magnitudes for the reachable
configuration, and the one-shared-resolver contract. ADDED SCOPE:
the shipped drop-YAML startup incompatibility is unit 85's to close
— repair the example to valid routing OR refuse it with a correct
migration message. The gate carries BOTH branches: the non-drop
real-Cobaya configuration that reaches and catches the silent wrong
result, and the drop configuration proving repair-or-refusal.

## UNIT 86 (20M-19, 2026-07-13): the training metric honors the grid2d pin — one effective-residual owner applies const_mask before every diagonal reduction

Finding (red team, CONFIRMED; the untruncated grep shows const_mask
lives only in geometries/grid2d.py, consumed by NO loss): the pin is
enforced only in Grid2DGeometry.decode, so for legal pinned rows two
standardized predictions ([0,0] vs [100,0]) decode bit-identically
to the same served answer yet score chi2 = 0 vs 10000 — training,
validation, best-epoch selection, and diagnostics distinguish
physically identical predictors and backpropagate through a
coordinate the public path discards. The program currently optimizes
and selects a different function from the one it serves. Distinct
from unit 63 (which rules whether a pin is LEGAL): permission to pin
does not make the metric honor the pin.

Contract (ratified): (1) ONE effective-residual owner applies the
persisted const_mask before every diagonal chi-squared, loss,
validation, and diagnostic reduction; (2) a pinned coordinate
contributes exactly ZERO residual and ZERO gradient (decode
guarantees its physical prediction equals the stored constant); (3)
the rule covers plain ScalarChi2, diagonal NPCE residual
composition, and TransferDiagChi2, including fine-tune and
rebuilt-artifact evaluation; (4) the mask applies AFTER the
effective base/correction composition is formed, so packed targets
and internal base terms cannot bypass it; (5) unit 63 remains the
sole authority on mask legality — the metric owner consumes only a
validated mask; (6) const_mask = None and every unpinned coordinate
are byte-identical; (7) finite screening stays defense in depth and
may not convert a discarded pinned-coordinate NaN into a usable
value — the owner defines the refusal of that corruption before
masking.

Legs (ratified; CPU/Torch, board-listed): the executed [0,0] vs
[100,0] witness (bit-identical decode, equal ZERO chi2/loss);
perturb only the unpinned coordinate and recover its direct squared
residual; validation metrics + best-epoch ranking invariant under a
pinned-only output mutation; a legal unit-63 low-k boost pin under
none and syren_halofit; plain, NPCE-residual, and diagonal-
transfer/fine-tune paths; save/rebuild retains the verdict; the
no-pin control byte-identical; a mutation restoring the
all-coordinate sum reports 10000 and reds. Placement: the
grid2d/loss contract beside unit 63 (never folded into its legality
validator); a science-metric defect ahead of any grid2d production
training.

### UNIT 85 — the coupling clause (20M-18 rejection notice, 2026-07-13)

Adopted from the red team's crossing notice: repairing the shipped
EMUL2 example by merely REMOVING drop: true is insufficient — that
alone converts the loud startup failure into the silent wrong-physics
lane. The example repair and the canonical dark-energy resolver land
IN THE SAME UNIT: either the routing delivers a correct wa to the
adapter (resolver landed) or the example refuses with migration
text. The magnitude ladder and the spy-tuple leg run on the non-drop
branch; the shipped drop branch is the startup/migration leg.

## 25M-36 (Architect-confirmed Red Team finding; repair awaiting audit): the bounded-staging mean leg used the forbidden pre-cast reference and made `mps-identity` red

The queue-2 evidence walk executed the real
`gates/checks/mps_identity.py` child with Cocoa Torch 2.6.0.  The child
completed every other printed check but failed
`bounded staging: streamed mean equals the known answer`.  This is a gate-
reference defect, not evidence of a producer defect.

The independent law calculation first forms float64
`log(raw / base)` as `want`.  The producer then stores those law rows as
float32 and streams its moments over that stored payload.  The child instead
computes `want.mean(axis=0)` before the float32 conversion and only converts
the final mean.  Those operations do not commute.  On the seeded production-
width fixture, the two means differ by as much as
`5.960464477539063e-08`; at column 4799 the default `numpy.allclose` ratio is
`1.183065878154562`, so the check is deterministically false.  In the same
probe, the producer center is exactly array-equal to
`want.astype(float32).astype(float64).mean(axis=0).astype(float32)`.  The
stored float32 rows also remain exactly equal to `want.astype(float32)`.

This reopens an already-issued acceptance clause: the grid2d moments contract
states that an analytic pre-cast log result alone is not the reference for a
float32 stored payload.  The present gate repeats precisely that forbidden
reference.  It therefore cannot certify the correctly implemented producer,
and the queue-2 evidence block above records the logical
`mps-identity.bounded-staging-values` leg as current-red rather than claiming a
whole-gate pass.

Required repair: keep the producer frozen.  In the child, derive the
independent stored-payload reference by converting the independently computed
law rows to float32 first, then accumulate their mean in float64 and convert
the final reference to the producer's persisted dtype.  The stored values and
mean are two assertions: the former remains exact, while the latter compares
the same represented quantity the geometry receives.  A mutation arm that
restores the pre-cast-then-convert ordering must fail on the seeded fixture,
so a future tolerance widening cannot bless the wrong reference.  The full
`mps-identity` child must then exit zero.  This is a `gates/checks/` repair and
was transferred to the Red Team after the collision window closed.

### Red Team implementation record for Architect audit

The repair changes only the check and its records. A small helper now names
the representation boundary explicitly: independently calculated float64 law
rows become float32 stored rows, those stored rows are promoted to float64 for
the column sum and the final mean is converted to the producer's persisted
float32 dtype. Both the small staging fixture and the production-width bounded
fixture use this order. The producer remains unchanged.

The bounded fixture retains two separate assertions. Its stored row values are
array-equal to the independent float32 payload, and its streamed center is
array-equal to the mean of that payload. A mutation control calculates the old
mean-before-cast result and requires disagreement. The seeded fixture reports
a maximum difference of `5.960464478e-08`, so the control distinguishes the
two orders without a widened tolerance.

The direct Cocoa Torch 2.6.0 CPU child changed from one deterministic failure
before the repair to `PASS: mps-identity all checks green` after it. The
existing logical aid `mps-identity.bounded-staging-values` owns the added
mutation. No board or runner file was changed. This is implementation evidence
for the Architect and does not certify the landing.
