# Models, designs, and the science doctrine

Consolidated 2026-07-11 from nla-as-design-spec.md,
head-activation-per-component.md, activation-families-norm-knob.md,
npce-yaml-wiring.md, npce-and-ia-template-factoring.md,
resmlp-cnn-perbin-architecture.md, per-bin-parallel-resmlp-plan.md,
trf-mlp-width-knob.md, film-conditioning.md,
emulator-high-d-and-tatt-templates.md, activation-function-
generalizations.md, geometry-loss-composition.md,
designs-losses-family-folders.md, analytic-scaling-preprocessing.md,
emulator-sample-efficiency-is-the-goal.md,
emulator-floor-is-data-coverage.md (retired; full texts + build logs
in git history). Code homes: emulator/designs/, emulator/losses/,
emulator/activations.py.

## The architecture family

- Layout (the GRF family folders): designs/{blocks, plain, ia, pce}.py
  + losses/{core, ia, pce, scalar, cmb, transfer}.py; activations.py
  stays FLAT (shared beyond the family; the drift proof monkeypatches
  it by path). Verbatim moves only; artifact-safe (state_dict .emul,
  cls markers, hardcoded pce import).
- MODELS registry keyed by (name, ia); name in {resmlp, rescnn,
  restrf}, ia in {None, nla, tatt}; IA_DESIGNS centralizes
  {amp_names, coeff_fn, n_templates} — TATT was one new entry, zero
  new code paths. Capability flags replace isinstance everywhere:
  factored, needs_geom, needs_bins, needs_params, head_block
  (enforced at class definition by DesignSpec.__init_subclass__).
- The correction-head philosophy: a shared ResMLP trunk does the
  params->targets MAPPING; a small axis-aware head (conv or TRF)
  corrects the theta-structured residual the trunk leaves. A dense
  trunk is OUTPUT-PERMUTATION-INVARIANT — output-axis structure is
  invisible to it; weight sharing along theta is an effective-DOF
  reduction on the OUTPUT side (published CMB precedent 0.2 -> 0.06).
  Heads act in theta order via FIXED W_fd/W_df buffers (live geometry
  calls in forward() break CUDA graphs).
- D-CM13 family lift (2026-07-11, user order + arXiv 2505.22574): the
  heads ride cmb / grid / grid2d too. The diagonal family geometries
  whiten IN physical order (ell / z / z-slices x k), so the basis
  change is the IDENTITY — W_fd/W_df stay None (hasattr(geom,
  "evecs") branches; never build n_keep^2 identity buffers) and the
  split comes from geometry.attach_head_coords() (cmb/grid: one bin;
  grid2d: one bin per z slice = conv channels / TRF tokens). New knob
  model.trf.n_tokens re-segments a single-bin spectrum into
  contiguous attention windows (rejected on multi-bin geometries).
  Scalar stays trunk-only: named outputs have no coordinate axis.
  Rebuild-side attach lives in results._rebuild_model; the COSMIC-
  SHEAR head-artifact rebuild gap was fixed the same evening:
  bin_sizes (+ pm_kept) now PERSIST in DataVectorGeometry.state()
  (schema-additive; attribute-unset when None so the hasattr guards
  survive), and a pre-persistence head file is refused loudly at
  rebuild.
- Identity-at-init discipline: the LAST layer of every head branch is
  zero-initialized so model == trunk EXACTLY at init and at the
  two-phase handoff; every activation family maps 0 -> 0 (which
  licenses per-head activations). gate_init 0.1 = soft-start brake,
  1.0 for short head phases, 0 = deadlock.
- Two-phase is a HEAD capability, not an IA privilege (2026-07-12
  user ruling): plain ResCNN/ResTRF define set_train_phase with the
  template contract (joint/trunk/head; trunk phase = head bypassed
  at pure-trunk cost; head phase = frozen trunk under no_grad), so
  trunk-then-head scheduling + the trunk:/head: blocks + the head
  activation pin work on cosmic shear AND cmb/grid/grid2d alike.
  Single-phase = resmlp only.
- Conv head (bins-as-channels): pad_idx scatter into (n_bins, max_bin)
  -> Conv1d(n_bins->n_bins, k) blocks -> gather -> gate; knobs
  kernel_size, n_blocks_cnn, groups (probe-block cuts, validated),
  separable (parameter economy, measured SLOWER), rescale_kernel
  (odd-up receptive-field ladder). Two stacked LINEAR convs collapse
  without a mid activation — the essential act_mid.
- TRF head: tokens = raw padded bin segments at natural width max_bin
  (26 on LSST-Y1; n_heads must divide it); NO adapters (the published
  design's embed/project layers deliberately removed); the per-token
  MLP width is PINNED to the token width (mlp_width SHELVED by user
  ruling — do not re-propose unprompted); shared_mlp = permutation-
  equivariant caveat.
- FiLM (model.cnn.film / model.trf.film): per-channel gamma(z)*h +
  beta(z) from the cosmological inputs, identity-init, per block —
  makes the correction parameter-dependent; factored runs MUST
  condition on x[:, :n_in] only (amplitude-blind or the closed-form
  exactness dies). First evidence strong (0.28 vs 0.37 stall on an
  immature trunk).

## Activations and norms

- Six families in make_activation: H (the paper's learnable
  identity<->Swish interpolation; non-saturating linear tails),
  power, multigate(K), gated_power, relu, tanh. Learnable shape
  params are per-feature; weight decay must never touch them (the
  module-role allowlist in training-stack.md).
- model.norm: affine (default — the paper's per-layer gx+b) /
  per_feature (FeatureAffine, the tanh saturation guard) / none.
  batchnorm DELIBERATELY not offered: batch coupling confounds
  bs/EMA experiments, train/eval stats split risks compiled-twin
  mode-baking, BN buffers sit outside the EMA average.
- Per-head activation pin (model.cnn/.trf.activation): construction
  knob, run-global (never in phase blocks — a mid-run family swap
  re-inits learnable params under trained weights); licensed by
  trunk_epochs > 0 AND freeze_trunk true; `head: activation:` is a
  legal alias, both spellings = error; `trunk: activation:` errors
  with a teaching message.

## Factored IA (what "factored" means)

Parameters entering the dv as polynomial COEFFICIENTS (NLA A1 -> 3
templates [1, A1, A1^2]; TATT a1/a2/b_TA -> 10 templates) are excluded
from the net input, appended raw past n_in by AmplitudeFactorGeometry,
and combined closed-form in the loss — exact, prior-width-independent
generalization, implemented as ARCHITECTURE on the existing scattered
samples (the loss reads each sample's own amplitudes; no re-simulation,
no N/3). Z-evolution eta POWERS do not factor and stay emulated. The
benefit scales with amplitude prior width: neutral on narrow NLA
(a validation success), the whole point for TATT's wide coupled
amplitudes. NEVER emulate a parameter dependence you can write down.

## NPCE (the pce: block)

- Top-level `pce:` (NOT inside train_args — structurally unsweepable:
  sweep_hyperparam stages once, a pce knob there would sweep without
  refitting the base). {form: residual|ratio, p_max, r_max, q, k_max,
  loo_max, max_terms, max_fail}; exclusive with rescale/ia.
- The science verdict stands: a PCE base adds CAPACITY only (the
  shape modes are not low-degree polynomial); NPCE is infrastructure
  by user directive ("past failures on low T do not discourage me"),
  not a proven floor-lever. Fit lessons baked into the defaults: low
  degree (Runge), keep only well-predicted modes (loo_max ~0.05),
  early-stop, CPU-numpy LARS with closed-form LOO.
- FAMILY-WIDE since 2026-07-12 (user ruling: "nothing should prevent
  users from using PCE as the trunk and MLP as the head, on all
  cases" — the user's arXiv 2404.12344 runs an NPCE on the boost;
  EuclidEmulator2 is a PCE of pklin). scalar / cmb / grid / grid2d
  wrap losses/pce.py::PCEResidualDiagChi2 (subclasses CmbDiagonalChi2
  — the diagonal metric IS these families' chi2; roughness composes
  because pred - target is the full whitened residual). Fit hook =
  experiment._fit_diag_pce, shared by the four build_geometry
  branches; predictor side = inference._build_diag_decoder on every
  family branch (a bare geom.decode would be silently wrong with a
  base in the file). Two deliberate boundaries: residual-only off
  cosmolike (ratio is a dense-covariance concept — validate_pce
  diagonal=True), and on CMB only amplitude_law "none" (the law loss
  owns the target construction — validate_cmb). Note the cosmic-shear
  verdict above does NOT transfer: on MPS the PCE fits the law-space
  boost, exactly the 2404.12344 regime where it worked.

### NPCE LOO gate must be absolute (red-team 2026-07-12 fifth wave, Architect-VERIFIED, CRITICAL, open; the full contract for the wave-1 pce-fallback finding)

Verified at designs/pce.py: line ~414 `if not cols:  # always keep
mode 0` unconditionally refits and keeps mode 0 when NO mode passed
loo < loo_max — the persisted base can carry a mode ~1e30 above the
requested ceiling while the startup report prints "kept 1 (loo<T)"
as if the predicate held. This defeats the "a wiggly base must not
poison the refiner" rule: a requested NPCE run can be WORSE than a
plain network. Second edge, also verified (~209-211): in
select_lars_loo, once every candidate column is active the score
vector is all -1 and argmax picks column 0 again — max_terms above
the candidate count appends DUPLICATE support indices instead of
stopping.

Contract (Implementer; the red-team block of record adopted whole):
delete the fallback; when no mode passes, FAIL the fit loudly naming
the best attempted LOO, the threshold, and the modes tried (never a
mean-only or failed-mode base — the "NPCE base is alive" rule is
preserved by refusing, not by faking); every recorded/kept LOO
finite; X_white/Y_white finite, 2-D, row-aligned, nonzero widths,
enough rows; select_lars_loo stops when all candidates are active,
never duplicates a support index, caps terms at the candidate count;
best_beta/support must exist and be finite before returning;
the fit report derives from actual kept-mode predicates (printing
"kept K (loo<T)" with a violating persisted mode must be
impossible); math.isfinite on pce.loo_max (NaN passes the <= 0 check
today). Gates: predictable control keeps a real mode with every LOO
below threshold; the strict-threshold fixture raises "no mode
passed" and writes NO artifact; NaN/Inf input/target/LOO/loo_max
raise; max_terms > n_candidates terminates with unique support; a
one/two-column candidate set cannot duplicate index 0; valid PCE
save/rebuild unchanged. Land BEFORE any NPCE production training.

## Composition spine

CosmolikeChi2 HOLDS a geometry (composition, never inheritance);
build the geometry once, wrap in any loss; losses forward dest_idx/
total_size/encode/decode. needs_params = "encode/decode/chi2/loss
take the whitened params" — every diagnostic MUST branch on it (a
hardcoded geom.decode is silently wrong for param-aware losses).

## Model-block value schema (red-team 2026-07-12 fourteenth wave, Architect-VERIFIED, open; joins the train_args-totality cluster)

The nested model: schema validates KEY NAMES only (experiment.py:225
"an unknown key raises, listing what is allowed") and copies active
block values straight into the design constructors. There is no value
contract, and the headline failure is a SILENT ARCHITECTURE DEMOTION,
not a crash:

- `model.trf.n_blocks: 0` builds an empty block list; the ResTRF
  forward (ia.py:933-947) leaves t == t0 so corr = t - t0 is
  identically zero FOREVER. The identity-start doctrine ("corr = 0 at
  init", the two-phase enabler) is exactly what makes this silent:
  the trunk trains normally, aggregate collapse bars can pass, and
  the requested transformer head never exists scientifically. This is
  the architecture-level analogue of the dead-network rule — a gate
  must prove the head CANNOT silently reduce to the trunk.
- Quoted "false" is truthy: rescale_kernel / separable / film /
  shared_mlp flow untyped into constructors and flip designs on.
- Zero-block crashes with unrelated messages: `model.cnn.n_blocks: 0`
  hits `self.convs[-1]` (ia.py:505, IndexError);
  `model.trf.n_mlp_blocks: 0` hits `self.mlp_lins[-1]`
  (blocks.py:639).
- `n_heads: 0` divides/modulos by zero (blocks.py:602 — the assert
  ITSELF evaluates `dim % 0`); incompatible n_heads relies on that
  assert and vanishes under python -O. kernel_size (ia.py:353),
  groups (ia.py:416), and the geometry assumptions (ia.py:423, :448)
  likewise rely on assertions on public config paths.
- `gate_init` passes through float() (ia.py:490): NaN/Inf accepted,
  and since corr starts 0, `out = y + gate * corr` makes Inf * 0 =
  NaN immediately.
- int() coercions: n_gates on BOTH the trunk activation path
  (experiment.py:292) and the head-pin path (:4108, :4258, :4267);
  n_tokens at plain.py:866. Bools become 0/1, floats truncate,
  numeric strings pass, zero reaches a zero-length gate tensor.

Adopted contract (theirs, whole): ONE pure active-model value
validator before geometry/model construction — the standing ruling
that INACTIVE architecture blocks may stay configured-but-unused is
preserved. Boolean fields require actual YAML booleans (no truthiness,
no coercion). Integral fields reject bools, strings, and fractional
values: positive width / gate count / head depth / MLP depth / head
count; CNN and TRF correction-block counts at least one; n_tokens
None or an exact integer, then its geometry-dependent bounds check.
kernel_size positive odd; groups an exact allowed value for the
selected design; n_heads positive and dividing the resolved token
width. gate_init a finite real non-bool. Normalized values persist in
the resolved recipe; validation never changes accepted-run values.
Constructor assertions on public config paths become explicit typed
exceptions so -O behaves identically.

Interlock with the asserts-under--O unit (queue 12): the ia.py /
blocks.py constructor asserts in its census are SATISFIED BY THIS
UNIT's typed-exception clause — cross-reference, no double work; unit
12 keeps the non-model surfaces.

Red legs (adopted): each quoted-false field raises at its full dotted
path while genuine false keeps today's recipe and parameter census;
zero CNN / TRF / TRF-MLP blocks each raise before construction;
zero/incompatible heads, even/non-positive kernel, invalid groups,
coerced n_tokens raise diagnostically; NaN/Inf/bool gate_init and
zero/negative/fractional/bool/string n_gates raise; valid boundary
controls build under ordinary Python AND python -O; and the
demotion-proof leg — a model requested as ResCNN/ResTRF must contain
at least one corresponding head block.

## The science doctrine

- The objective is SAMPLE EFFICIENCY: the position of the
  f(dchi2>0.2) vs N_train learning curve; N_target = smallest N with
  f < 0.10. The real target is high temperature + w0wa + TATT where
  N_train is the binding constraint. The floor at T=16 is
  DATA/COVERAGE-limited (10k -> 0.219, 46k -> 0.100) — "it is
  capacity" was called once and refuted; size claims to evidence,
  run the curve before writing a law.
- Effective (nonlinear) dimension sets the cost, not nominal
  parameter count (photo-z shifts and IA amplitudes are ~free). The
  two levers for a data floor: physics structure (factoring,
  features) and point placement (importance sampling);
  REPRESENTATION beats sampling when failures are diffuse.
- Hardness is H0-led "amount of small-scale structure" (ln omegab
  NEGATIVELY correlated — more baryons = easier); the omega_b h^2
  story was the right CUT variable but the wrong hardness gradient.
- Certification for f < 0.1: binomial noise ~±0.015 at Nval~400 and
  best-epoch selection biases LOW — certify with margin (~0.085),
  seeds, a larger val set.
- Scoreboard (T=256, 250k): resmlp 0.1558, nla 0.1472 (winner),
  rescnn+nla two-phase 0.1105.

## CLOSED experiments (never re-propose)

nla_as (As-scaled factoring — code DELETED; errors became
As-directional); target rescaling by analytic R (lost the
sample-efficiency test; the R machinery in analytics.py survives as
optional preprocessing); per-bin dense-MLP split (discarded the
shared map; MSE != chi2 under block whitening — keep the full Cinv
contraction); global CNN head at T=16 (neutral — the win lives at
high T); conv-as-matmul (CPU-only pathology); ParallelResMLP /
template_mix / GLU mixing / max-pooling heads; batchnorm; trf
mlp_width (shelved); smoothness priors (the chi2 is a HIGH-PASS
filter; its blind spot is the smooth common-mode); loss-shaping as a
floor lever; log-whitened inputs; space-filling sampling (fights the
deliberate tempered-Gaussian design); capacity beyond width 256
(saturates); the local-linear floor as an instrument. The width sweep
(128 slightly starved; 256 saturated at 0.212) means width-128 "wins"
were capacity — use 256 baselines.

## Recurring gotchas

Mid activation between stacked convs; fixed basis buffers not live
geometry calls in forward(); 0-dim device tensors for compiled-loop
scalars; warmup lr at the TOP of the epoch; epoch-0 baseline eval
seeds best-tracking; diagnostics branch on needs_params; benchmark
conclusions do not transfer across devices.

## The power activations have a zero derivative at exactly zero (red-team 45M-16, 2026-07-12, Architect-VERIFIED; queue 40)

Both PowerGatedActivation and GatedPowerActivation implement the
signed power as `psi = torch.sign(x) * ((1.0 + ax) ** p - 1.0) / p`
(activations.py:147, :220) and document slope 1 at the origin with
p = 1 recovering psi = x (:115, :162) — the claimed identity /
H-recovery start. The VALUES agree, but the Jacobian does not:
torch's sign has zero derivative, |x| has zero derivative at the
origin, and the inner magnitude is zero there, so autograd returns
d psi / dx = 0 at exactly x = 0 (documented claim: 1; full default
activation: 0 instead of H's 0.5). This is a gradient-absorbing
point exactly where the identity-start doctrine deliberately places
zeros (corrections, padding, fresh channels), and no forward
identity check can see it — the defect is Jacobian-only.

Contract (Implementer): (1) reimplement as x times an even magnitude
ratio, psi_p(x) = x * ((1+|x|)^p - 1) / (p |x|), with the analytic
limit 1 at zero built in; (2) the near-zero ratio computed stably
(log1p/expm1 or a justified series), no unguarded 0/0 branch;
(3) constructor requires finite positive p_min < p_max (the /p
denominator can never approach zero through a malformed direct
call); (4) forward values preserved away from the near-zero
neighborhood; (5) documentation corrected only AFTER the derivative
is proven. Gate legs (torch, workstation, riding an activation/model
identity gate): H vs power vs gated_power values AND input gradients
at x = [-eps, 0, +eps] at default init; p = 1 equals x with
derivative 1 including exactly zero; a zero preactivation inside a
small residual block transmits a nonzero gradient; float64 gradcheck
over several learned p; a mutation restoring sign(x) * f(|x|) fails
specifically at the zero-Jacobian assertion.

## NPCE maps arbitrarily out-of-domain cosmologies to the same boundary (red-team 45M-28, 2026-07-12, Architect-VERIFIED; queue 46 — joins the inference-boundary campaign, unit 21, with explicit NPCE legs)

PCEEmulator.forward maps whitened inputs to the fitted Legendre box
and applies an unconditional clamp (designs/pce.py:505-506,
Xm = 2(X - lo)/(hi - lo) - 1; Xm.clamp(-1, 1)); the comment (:503-504)
says a point "just outside" stays in range, but there is no
definition of "just": one rounding unit outside and an arbitrarily
distant cosmology collapse to the identical boundary coordinate, the
output stays finite and plausible, and the finite guard cannot see
that the base evaluated a DIFFERENT cosmology. Both residual NPCE
forms are affected — the refiner sees the real X but was trained
around a base whose hidden saturation is part of its target; nothing
guarantees it repairs an arbitrarily clipped base outside the
calibration box. lo/hi are already persisted: the missing piece is a
policy, not data.

Contract: (1) a NAMED, persisted PCE domain policy — scientific
serving defaults to refusal outside the calibrated whitened box with
only a documented floating-point tolerance; (2) if exact-boundary
clipping is kept for roundoff, a scale-aware tolerance with rejection
beyond it — an unconditional clamp is never the validator; (3) errors
name the stored parameter coordinate, whitened value, allowed
[lo, hi], and overshoot, mapped back to the input-geometry record
where available; (4) training/validation evaluation and inference use
the IDENTICAL policy (a validation set outside the fitted box cannot
be silently scored as inside); (5) persisted lo/hi validated: finite,
1-D, aligned with the PCE input dimension, strictly lo < hi;
(6) boundary-hit / near-tolerance counts recorded in the resolved
fit/evaluation record. Distinctness ruling accepted as argued: this
is NOT the LOO-selection unit (LOO judges the polynomial inside its
domain; this defines whether a query belongs to the domain at all).
Red legs (torch, board-listed): two far-out same-side inputs that
currently collide must refuse; below-low and above-high per
dimension; NaN/Inf bounds; lo == hi; shape mismatch; exact endpoints;
one-ULP/tolerance control; training and rebuilt-artifact inference
agree; residual NPCE on a diagonal family AND a dense-covariance
family.
