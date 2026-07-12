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

## Composition spine

CosmolikeChi2 HOLDS a geometry (composition, never inheritance);
build the geometry once, wrap in any loss; losses forward dest_idx/
total_size/encode/decode. needs_params = "encode/decode/chi2/loss
take the whitened params" — every diagnostic MUST branch on it (a
hardcoded geom.decode is silently wrong for param-aware losses).

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
