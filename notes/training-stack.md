# The training stack: losses, schedules, phases, EMA, sizing

Consolidated 2026-07-11 from loss-mode-berhu.md, loss-block-nesting.md,
resolve-phase-args-single-phase.md, phase-blocks-nested-lr-scheduler.md,
eval-bs-decoupling.md, weight-ema-snapshot-coupled.md,
ema-anneal-schedule.md, berhu-anneal-schedule.md,
weight-decay-only-weight-matrices.md (+ its older twin),
freeze-trunk-joint-phase2.md, driver-audit-phase-sweep-guards.md,
n-train-n-val-absolute-counts.md, shared-budget-across-sequential-calls.md,
banner-prints-consumed-view.md (retired; full texts in git history).
The code home is emulator/training.py + emulator/experiment.py; the
user-facing story is the main README sections 4-11.

## The loss family (losses/core.py `_reduce`, ONE mode ladder tree-wide)

- Modes (`_LOSS_MODES`): sqrt (default), chi2, sqrt_dchi2, berhu,
  berhu_capped. The transform is TRAINING-only; eval/metrics/selection/
  EMA always run on the raw per-sample chi2.
- berhu (knot k, default 0.2): L = sqrt(c) for c <= k;
  L = (c + k)/(2 sqrt k) above. berhu_capped adds a third regime above
  cap K (default 10): L = (2 sqrt(K c) + k - K)/(2 sqrt k). C1 at both
  knots by construction. Vote profile (dL/dc * sqrt c): 1/2 in the
  bulk, rising ~x7 across the (knot, cap) window, plateau
  sqrt(K/k)/2 ~ 3.54 above the cap — bounded monster votes.
- Naming caveats (recorded twice): the knot is in delta-chi2 units
  (literature BerHu delta = sqrt(knot)); the shape applies to the
  per-SAMPLE total misfit, not elementwise — matching the frac>0.2
  sample-counting goal metric.
- Config schema (the CURRENT one): `train_args.loss` block, whitelist
  {mode, berhu, roughness}; absent = {mode: sqrt}. `berhu` sub-block
  {knot, cap, anneal}, valid only beside a berhu mode; `cap` is
  accepted-but-unused under plain berhu (one shared block survives a
  mode sweep). D-L1v3 ruling: the knot block is also accepted under
  the EXACT ACTIVE mode string (`berhu_capped:` beside mode
  berhu_capped canonicalizes to `berhu` on a copy); mismatches error
  naming both fixes. `roughness` {lam, period_cut} is CMB-only
  (families-scalar-cmb.md); non-CMB configs reject it loudly.
- berhu anneal: L_s = (1-s) sqrt + s L_mode; s(e) ramps 0->1 after
  hold. Advice (recorded, unenforced): hold_epochs >= trim's hold.

## The shared anneal-schedule family

One validator (`_validate_anneal_block`) + one evaluator
(`anneal_value`, shapes {const, linear, cosine, step}) serve every
0->1 schedule: trim, focus, loss.berhu.anneal (the blend s), and
ema.anneal (the horizon: h(e) = horizon_epochs * s(e)). Shared keys
{hold_epochs >=0 int, anneal_epochs >=1 int, shape}, all required,
bools rejected. Universal rules: activation by block PRESENCE (never
a bool key); per-phase schedules restart at that pass's epoch 1;
schedules never rewind (functions of the epoch counter); error texts
name their own path.

## Phase blocks and single-phase demotion

- Phase blocks `trunk:` / `head:` mirror the top level; final
  whitelist `_PHASE_BLOCK_KEYS` = {lr, scheduler, loss, trim, focus,
  clip, rewind, ema} (ONE tuple in training.py; experiment imports
  it, never duplicates it).
- Semantics: `lr` = OVERLAY (bs_base inside a phase lr is rejected —
  the sqrt-batch anchor is run-global); everything else = FULL
  REPLACEMENT (a phase scheduler replaces kwargs, keeps the class —
  `cls` rejected; `ema: null` = key-present-None disables an
  inherited ema, key-absent inherits).
- Capability probe everywhere: `hasattr(model_cls, "set_train_phase")`
  on the CLASS, never the model name (every resmlp incl. ia: is
  single-phase; two-phase = rescnn/restrf with an ia).
- Single-phase demotion (`resolve_phase_args`, pure, the single choke
  point at the top of experiment.train()): drops head/trunk_epochs/
  freeze_trunk, merges trunk.X to the top by pure prefix-strip, on a
  COPY (sweep drivers reuse train_args), with a notice naming only
  what happened. The head: block is validated BEFORE being dropped.
- `validate_sweep_paths(paths, two_phase)`: on single-phase, sweeping
  head.*/trunk_epochs/freeze_trunk raises ("every sweep point would be
  identical"); trunk.X names the concrete top-level key to sweep
  instead. Both sweep and tune drivers call it before spawning.
- `freeze_trunk: true|false` (default true): false = phase 2 trains
  trunk AND head jointly (the pass ROLE stays "head"; only the
  set_train_phase string becomes "joint"). Joint epochs cost more —
  the sanity signal. License rule: a per-head activation pin requires
  trunk_epochs > 0 AND freeze_trunk true.

## The consumed-view principle (banner contract, five rules)

1. Every printed configuration surface shows the RESOLVED view — what
   the run executes, never the raw YAML. 2. Displays reuse the SAME
resolution functions the execution path uses (drift structurally
impossible). 3. Tolerant consumption, truthful display (irrelevant
blocks ignored; displaying only-what-is-consumed IS the notice).
4. Every new config feature's spec must state its banner rendering.
5. Every audit checks display surfaces beside execution wiring (the
D-P2 lesson: a banner printed `2000 trunk + -1000 head` from raw keys
on a single-phase run). Idiom: the class prints itself —
`head_block` class attribute + `describe_spec`; a new architecture
missing head_block fails at import.

## EMA (weight averaging) and its coupling invariant

- `train_args.ema` {horizon_epochs, anneal}: horizon in EPOCHS, beta
  derived per step (1 - 1/(h*steps_per_epoch)) — batch-size-invariant
  by construction. Absent block = the loop is BYTE-IDENTICAL.
- THE invariant: anything the rewind un-lives must disappear from the
  average — the best snapshot is one unit {theta, optimizer state,
  theta_bar}, saved and restored together.
- Eval by in-place weight swap (foreach copy_), no second compiled
  twin: raw eval drives the plateau scheduler; the EMA eval drives
  selection + printed metrics; shipped model = best EMA weights.
  CRITICAL: in-place copy_ only — compiled/CUDA-graph closures hold
  parameter STORAGE pointers; rebinding .data breaks replay silently.
- ema.anneal schedules the HORIZON (not a blend); beta = 0 while
  h(e)*steps < 1 gives exact tracking early, continuity for free.
- Open science margin (bs+EMA thread): at smoke scale the averaged
  model's best val lands EARLY (epoch 7/20) with val rising after —
  EMA-selection superiority is unproven.

## Compiled-scalar discipline (one deliberate exception)

Every scalar feeding compiled loss code is a 0-dim device tensor
built per pass or filled in place per epoch (kappa_t, knot_t, cap_t,
trim_t, focus_t, s_t) — closure Python floats silently kill CUDA-graph
replay. The exception: EMA beta(e) is an eager per-epoch float because
the lerp lives outside every graph; a comment forbids both "fixing" it
and copying the float pattern into compiled-side schedules.

## Sizing: absolute counts, derived eval batch, weight decay

- `data.n_train` / `data.n_val`: REQUIRED absolute row counts enforced
  AFTER param_cuts (the old divisors used the PRE-cut dump size — the
  silent-shrink leak). Under one split_seed the n=50 selection is a
  prefix of the n=100 selection (the learning-curve promise; SET
  inclusion on the sorted in-RAM path). n_keep >= 1 guarded (D-1: a
  negative n_keep silently staged phys[:-5]).
- The eval batch is DERIVED, deliberately NO YAML knob (a pure-perf
  parameter with a computable optimum is not user-selectable):
  k = ceil(n_val/1024), bs = ceil(n_val/k), clamped to the memmap
  chunk; `_EVAL_BS_TARGET = 1024` module constant. eval_val pads the
  final batch so the compiled twin keeps ONE static shape — do not
  touch the padding. Metrics are per-row = partition-invariant.
- Weight decay: module-aware ALLOWLIST — decay exactly the `.weight`
  of nn.Linear / nn.Conv1d / BinLinear; everything else (all biases,
  Affine gains, every activation parameter of any shape) undecayed.
  Never a shape heuristic: ndim>=2 misclassified (K,dim) activation
  params and (G,out) BinLinear biases — the superseded bug. Allowlists
  fail SAFE (unlisted future module = undecayed). Default decay 0.0.
- Shared-budget rule: sequential allocators against one budget must
  each see budget minus what earlier calls made resident (the
  build_loaders train-then-val VRAM fix). Finish resource accounting
  in BOTH directions — "it's conservative" must not close an analysis.

## The loud no-alias migration pattern

Every schema break raises a ValueError whose body IS a paste-ready
YAML block carrying the offending values (divisors -> n_train/n_val
with an explicit semantics-changed warning; flat lr_base -> nested
lr:; flat loss_mode/berhu -> the loss: block; flat cut keys -> the
param_cuts block), identical on every reach path, checked BEFORE
generic whitelists. ValueError, never KeyError (KeyError reprs the
message and escapes the newlines). Companions: a block no pass can
consume is a loud config error; when a "natural mistake" is
self-consistent, accept it by canonicalization (D-L1v3) — a trap with
a good apology is still a trap.

## The driver surface (family-first names, thin wrappers, one code path)

Every driver is `<family>_<verb>_emulator.py` (user ruling: "what you
are emulating comes first always"). The cosmic-shear drivers are the
ENGINES — their mains take (prog, family[, out_default]) — and every
per-family driver (train / tune / sweep_ntrain / sweep_hyperparam,
for scalar / cmb / baosn / mps) is a thin wrapper that pins the
family: a wrong-family YAML fails at startup NAMING the right driver
(require_family_block), the Optuna study name becomes the wrapper's
prog (per-family studies never mix in a shared journal), and EVERY
capability rides through — the multi-GPU pool, --gpu-pack (with the
scalar dv-less VRAM fallback), LPT balancing, the journal study. The
sweep-block helpers (read_sweep_block, set_by_path,
SWEEPABLE_TOP_KEYS, ACTIVATION_PATHS) live ONCE in
emulator/family_drivers.py; the earlier serial per-family loops were
deleted when parity landed (2026-07-11, commit 2fcd367). Each wrapper
carries provenance comments naming where its main lives and what the
wrapper pins.

## Where the deltas live (IDs preserved for git archaeology)

D-B1 (deleted by the loss-block nesting — the structural fix beat the
behavioral patch), D-L1 v1-v3, D-P1/P2/P2v2/P3 (capability probe +
consumed-view banners), D-E1/E2, D-M1 (EMA an allowlisted acronym),
D-1 (n_keep guard), GMP (ema joins the phase whitelist). Gate homes:
the training-stack gates on the board (ema-off-identity, ema-smoke,
berhu-loss, loss-schema-equivalence, berhu-anneal, ema-anneal,
single-phase-demotion, head-scheduler-override, eval-batch-invariance,
joint-training, weight-decay-census, production-diagnostic) all
pointed here; their verdicts (all PASS, boards of 2026-07-08..10) are
in git history under the retired per-topic notes.
