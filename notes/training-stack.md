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
  on the CLASS, never the model name. Two-phase = EVERY design with a
  correction head: plain rescnn/restrf on every family they ride
  (the 2026-07-12 user ruling — "any trunk-head design could benefit
  from the two-phase training"; the plain heads mirror the template
  contract exactly: joint/trunk/head requires_grad groups, the trunk
  phase bypasses the zero-init head at pure-ResMLP cost, the head
  phase runs the frozen trunk under no_grad) and the factored-IA
  templates. Single-phase = resmlp, incl. its ia variants.
- NPCE family-wide (the 2026-07-12 ruling): the top-level pce: block
  fits the closed-form base on EVERY family (residual-only off
  cosmolike; on cmb only with amplitude_law none); the refiner is
  whatever model: names — heads, two-phase, and the whole loss
  surface included. Design facts: models-and-designs.md (the NPCE
  section's FAMILY-WIDE bullet).
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

### Red-team resource and process gaps (verified 2026-07-12, open)

- The loader planner still calls `build_loaders` without the live
  output width, leaving `dv_len=3000`. The same number estimates a
  dense float64 Cinv and per-batch chi2 scratch even for diagonal
  families that own no dense Cinv. This is neither a safe upper bound
  nor the right family model: a legal grid2d run can be 122 x 2,000
  outputs before thinning. Resource accounting must use the actual
  geometry/loss buffers, output width, target width (including PCE /
  transfer packed targets), and dtypes. Gates cover one cosmolike
  dense-Cinv loss and one wide diagonal grid2d loss and require the
  estimate not to understate measured peak allocation beyond a stated
  tolerance.
- Parallel Optuna joins workers but never reads `Process.exitcode`.
  After reloading a persistent journal it asks only whether any trial
  in the entire study is COMPLETE. Therefore a journal with one old
  success can report "search complete" and the old best when every
  worker in the current invocation crashes; partial worker loss also
  silently reduces the requested budget. Record the before/after trial
  set, inspect every worker exit, and distinguish a deliberately
  timeout-limited run from a failed current budget. The red gate seeds
  one old COMPLETE trial, crashes every current worker, and requires a
  nonzero parent exit.
- `run_gpu_pool` has no enclosing cleanup on its early error paths. A
  setup-failure marker can make the parent raise while other spawned
  GPU processes keep running. Token counts, lane counts, and required
  callbacks are not validated before spawning; a token count above
  `GPU_TOKENS` can leave a live worker blocked forever, defeating the
  liveness check. Validate the complete plan first; terminate/join all
  children and close queues in `finally`; inspect exit codes after the
  drain; and add an overall progress watchdog rather than treating
  "one process is alive" as proof of progress.
- The activation bake-off has a separate, older multiprocessing loop and
  does not use `run_gpu_pool`. Its worker catches only failures inside the
  individual `exp.train` call. A failure during device selection,
  `EmulatorExperiment.from_config`, validation/train staging, or geometry
  construction exits the worker without putting the fixed number of result
  tuples the parent expects. The parent performs blocking `result_q.get()`
  calls with no timeout or liveness check before it joins or inspects a
  child, so one setup failure hangs the command permanently. Give every
  worker a top-level failure envelope, drain with bounded waits plus child
  liveness/exit-code checks, and terminate/join siblings in `finally`. The
  red gate raises in `stage_val` in one worker and requires a prompt nonzero
  parent exit with every child reaped and no result file published.

The table writers have a smaller truth leak: `save_sweep_table` uses
`zip(values, fracs)`, so unequal inputs silently truncate; the
categorical branch can also emit more fraction rows than labels.
`save_learning_curves` fails incidentally by indexing only when a curve
is short. Both public writers should validate all column lengths before
opening the destination, so an existing result file is not replaced by
a partial/mislabeled table.

### The validation loader computes a safe chunk and then ignores it

`build_loaders` independently sizes train and validation and stores
`data["val"]["load"]`, correctly using the reduced budget after resident
training tensors. `training_loop_batched` reads only
`data["train"]["load"]` and passes that number to every `eval_val` call,
including the epoch-0 baseline; the validation chunk is otherwise dead.
When train fits resident but validation falls to RAM/disk streaming, eval
can therefore request more rows than the validation sizing proved safe
and OOM despite the memory ladder. `derive_eval_bs` is also capped against
the wrong chunk. The gate must force different train/val regimes and safe
chunk sizes, record loader request sizes, and prove validation never
exceeds its own `load` while all rows are still scored.

The same boundary needs basic totality guards. A configured batch size
larger than the staged training set makes every ragged chunk drop all its
rows, leaves `run_n == 0`, and divides by zero after doing no optimizer
step. Reject `bs > n_train` (and non-positive `bs`/`nepochs`) before model
or loader setup, with a sweep leg at the smallest N.

### The generic diagnostics are not wide-output safe

The optional diagnostic runs the local-linear floor before the grid2d
family pages whenever the loss is not parameter-aware. That calculation
materializes all train/val targets on the accelerator and then constructs
`Yn = Ttr[nbr]`, shape approximately
`N_val x k_nn x output_width`. At the production thinned MPS width
(122 x 201 = 24,522) and the default 40 neighbours, this is tens of
gigabytes even before the batched least-squares solution. A normal
Syren-law MPS loss is not parameter-aware, so `mps_train_emulator.py
--diagnostic ...` takes this path and cannot deliver the promised PDF at
production scale.

`eval_source_chi2` likewise moves the complete validation target to the
accelerator, and `grid2d_residual_diagnostic` retains full float64 truth,
prediction, and residual matrices on the host. These are cold paths but
still part of the public production command. Make the chi2 scoring
streaming; define a bounded wide-output floor (coordinate chunks, a fixed
validation subsample, or an explicit family-specific skip with truthful
PDF text); and compute grid2d bands without three simultaneous full-width
float64 copies. Acceptance sets a deliberately small memory ceiling on a
wide synthetic shape and requires the diagnostic artifact to finish.

The same diagnostic pair is not total on small datasets. `coverage_diagnostic`
accepts any `k_nn` even when it exceeds the number of distinct training rows
(the tree returns infinite sentinel neighbours), while `local_linear_floor`
uses those sentinel indices to index a tensor and fails. Validate positive
row counts and require enough distinct anchors for the requested fit
(`k_nn > n_param + 1` and `k_nn <= n_train`), naming the effective counts;
the small-data gate covers one row below each boundary.

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

## NaN scores as a perfect emulator (red-team 2026-07-12 fourth wave, Architect-VERIFIED; the queue-jumping unit)

The board's primary selection metric silently rewards numerical
failure. Verified against the code:

- eval_val (training.py ~1427-1433): `median = c.median()` propagates
  NaN; `frac = (c > thresholds).float().mean(0)` — a NaN chi2 compares
  False to every threshold, so NaN rows count as BELOW threshold. A
  [NaN, 0, NaN, 0] validation set returns frac 0.0: "perfect".
- The best-epoch rule (~2047): `f0 < best_frac` — a NaN-poisoned
  f0 = 0.0 beats every honest epoch and SNAPSHOTS the corrupted
  weights as best (the median tiebreak is NaN-safe by accident; the
  primary comparison is the hole). The driver then saves and reports
  that model.
- The train step (~1971-1980): loss.backward() with no finite check;
  clip_grad_norm_ (default error_if_nonfinite=False) scales NaN grads
  by NaN; optimizer.step() unconditional.
- eval_source_chi2 / diagnostic scoring: same pattern — zero
  isfinite/isnan calls exist in training.py.
- SEVERITY SHARPENING (Architect): this defeats the dead-network gate
  discipline itself — every smoke gate's collapse bar asserts
  best < bar, and a NaN run reports best frac 0.0, so the gates built
  to fail a dead network PASS a NaN one.

**Contract (Implementer unit; the red-team block of record adopted
whole):** (1) eval_val requires every per-sample chi2 finite before
mean/median/fractions — raise with the nonfinite count and the first
few validation row positions; (2) the scalar training loss must be
finite before backward; (3) gradients (or the computed norm) must be
finite before optimizer.step — clipping disabled does not mean
unchecked; (4) mean/median/fractions must be finite before the
best-epoch comparison/snapshot; (5) the same finite-output rule on
eval_source_chi2 / diagnostic scoring so no saved diagnostic carries a
silent NaN metric; (6) never replace NaN/Inf with a sentinel, never
count it below threshold — abort loudly, the error naming train vs
validation and the affected batch/row positions.
Gate legs (self-diagnosing): all-finite control byte-identical; one
NaN among good rows raises; +/-Inf raises; a finite forward loss with
a nonfinite gradient raises BEFORE the optimizer mutates weights; a
NaN scalar loss raises before backward; the error text names the side
and positions. Independent of the bs > n_train / run_n == 0 unit.

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
