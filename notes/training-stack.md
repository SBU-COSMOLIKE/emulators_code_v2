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

#### Thirteenth-wave extension (Architect-VERIFIED, folded into this unit): finite inputs publish undefined diagnostic statistics

Red-team finding, every claim re-derived. The diagnostics CREATE
non-finite values from entirely finite, valid inputs and publish them
as computed results — which is why this lives here and NOT in the
finite-training unit: finite model outputs are the control condition.

- `coverage_diagnostic` (diagnostics.py:123-126): `bad = dchi2 > 0.2`,
  then unconditional medians of both classes — an all-good run makes
  `knn_dist[bad]` empty and `median_bad` NaN; an all-bad run makes
  `median_good` NaN. Line 129's verdict
  `cov = (median_bad > median_good) and (rho > 0.1)` evaluates a NaN
  comparison as False, and the driver
  (cosmic_shear_train_emulator.py:483, :487-488) then formats the NaN
  medians into the log and prints "not clearly coverage: failures not
  sparser" — conflating "comparison impossible" with a negative
  scientific finding. A PERFECT run is reported as ambiguous-coverage.
- `hard_direction_regression`: line 293 floors every dchi2 <= 1e-4, so
  a sufficiently accurate emulator makes y constant and line 315's
  `r2 = 1 - var(y - Z@coef)/var(y)` is 0/0 = NaN; line 310's
  `np.corrcoef` NaNs on a constant feature or constant y; line 302
  guards the feature std (`+ 1e-30`) but line 326's omega-baryon
  direction divides by `g.std()` UNGUARDED — the asymmetry is the
  tell. Lines 320-331 already use numeric NaN as a deliberate
  documented sentinel for r2_omega — the exact pattern this contract
  removes from published surfaces.
- `cmb_residual_diagnostic`: line 420 `frac = (pred - truth) / truth`
  with no validity mask; a legal zero-crossing spectrum (TE) yields
  NaN/Inf in that column and line 424's percentiles make ALL FIVE
  fractional bands NaN at that multipole. The docstring (:354-355)
  acknowledges "spiky where te crosses zero" and points at the
  error-bar panel, but ships no mask and no definedness record.
- No gate executes the real functions on adversarial outcomes: grep
  for coverage_diagnostic / hard_direction over gates/ is empty, and
  all four family smokes hand-build finite coverage dicts
  (bsn_smoke.py:341, cmb_smoke.py:484, scalar_smoke.py:203,
  mps_smoke.py:265) for the plotting layer only.
- The artifact pair is saved before diagnostics run — deliberate
  (driver :358 "Persist the trained emulator first, before any
  diagnostics can fail"), and it STAYS: losing a trained model to a
  plotting crash is worse. The consequence is the contract below —
  the published diagnostic products must be truthful about
  unavailability, because they accompany an already-published
  artifact.

Adopted contract (theirs, whole): undefined statistics are represented
explicitly with a status/reason and counts — never numeric NaN
sentinels. Coverage records n_good and n_bad; all-good reports "no
failures", all-bad reports "comparison unavailable"; neither may
masquerade as "not coverage-limited". Hard-direction analysis detects
zero response variance, zero feature variance, invalid log domains,
and insufficient rows BEFORE regressing; unavailable coefficients/R^2
are omitted with a reason. Fractional residuals get a spectrum-aware
validity mask — for TE either omit the fractional panel or mask
denominators by a documented physical threshold and carry
per-multipole valid counts; the error-bar panel remains authoritative.
Logs and PDFs label unavailable analyses plainly: no formatted nan
statistics, no unexplained blank panels.

Red legs (adopted): all-good, all-bad, all-hardness-floored, constant
feature, constant omega-baryon direction, and an exact/near-zero TE
crossing — each executing the REAL diagnostic functions and verifying
the resulting PDF and log semantics. These join this unit's existing
small-data and memory legs; one unit, one audit cycle.

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

**Pre-training parity clause (eighth wave, Architect-VERIFIED, folded
in per the red team's sequencing):** build_warm_start's parity verdict
has the same NaN hole — warmstart.py ~863 computes
`max_dv = (out_new - out_src).abs().max().item()` and raises only on
`max_dv > _PARITY_TOL`; NaN compares False, so NaN outputs print
"[ok] finetune parity: max|dv| = nan". The extras torch.equal leg
(~867) happens to catch NaN incidentally but is SKIPPED when
n_extra == 0, so the verdict depends on the parameter-extension
shape. Addition to the contract: the finite guard covers pre-training
parity — finite encoded inputs, BOTH model outputs, their difference,
and the scalar max before any tolerance comparison; the [ok] verdict
must be impossible unless the compared tensors are finite; transfer
parity gets the same explicit guard (never rely on torch.equal's
incidental NaN behavior). Red legs: no-extra both-arms NaN, one-arm
NaN, Inf, extras-present NaN, transfer NaN — all fail with the
finite-contract message; valid exact/tolerance fixtures keep their
verdicts.

#### Finite-contract resume (2026-07-12, Opus) — CHECKPOINT, training.py core landed; gate + two scope calls open

training.py guards in place and Mac-verified (guard logic + message via
exec-extraction; the torch wiring rides the workstation gate):

- Two module-level helpers before eval_val: `_report_nonfinite(side,
  quantity, n_bad, n_total, positions)` — the one uniform ValueError
  ("finite contract [side]: N of M ... non-finite ... First offending
  positions: [...] ... no sentinel, never counted below threshold"); and
  `_global_grad_norm(params)` — a read-only global grad L2 norm for the
  clipping-off finite check (byte-identical to the old no-clip path).
- eval_val (clause 1 + 4): after `c = torch.cat(chi2s).cpu()`, before
  mean/median/frac, `torch.isfinite(c).all()` or abort naming the
  validation rows (an `order_rows` list now tracks the global val row per
  c position). It is the SINGLE validation chokepoint — baseline, raw,
  and ema evals all pass through it, so the best-epoch compare only ever
  sees finite scores (clause 4 needs no separate guard; documented).
- The train step (clauses 2 + 3): `torch.isfinite(loss)` before
  backward; `grad_norm` from clip_grad_norm_ (clip>0) or
  `_global_grad_norm` (clip==0) checked finite before optimizer.step,
  naming the diverged batch (epoch / chunk / batch). One host sync per
  step — the deliberate price of catching divergence at its source.
- eval_source_chi2 (clause 5): dchi2 finite before return, naming the
  source rows; side "diagnostic".

Two scope calls flagged for the Architect (not guessed):
1. GATE HOME. The self-diagnosing legs need torch. The natural existing
   training gate is GE-C (eval-batch-invariance, home=training-stack) —
   but it is `needs=("torch","gpu")` and thematically "eval-batch
   invariance". The finite legs are CPU-runnable and CRITICAL (deserve
   broad coverage). My default: extend GE-C; alternative: a dedicated
   CPU-only `finite-contract` gate (`needs=("torch",)`). Confirm which.
2. PARITY CLAUSE. The pre-training parity clause is warmstart.py's
   build_warm_start, not training.py; the handoff scoped this unit to
   "training.py's own finite contract". I did NOT touch warmstart.py.
   Confirm whether the parity guard rides this unit or a separate rider.

Hot-path note: clauses 2/3 add one host sync per training step (the
contract's "before backward / before optimizer.step" is per-step).
eval_val (clause 1) is the per-epoch backstop; clip==0 stays byte-
identical. Flagged against the "hot paths never slowed" rule — the
CRITICAL correctness contract wins, but the Architect may prefer a knob.

Mac gate: `probe_finite_contract.py` 5/5 (all-finite control no-raise;
NaN/Inf raise naming side + count + rows; diagnostic side; training-batch
message). `py_compile emulator/training.py` clean. Held.

#### Architect rulings on the two scope calls (2026-07-12, Fable)

1. GATE HOME: a DEDICATED CPU-only `finite-contract` gate
   (`needs=("torch",)`, home=training-stack) — not an extension of the
   eval-batch gate. Three reasons: the legs are CRITICAL and
   CPU-runnable, so they must not hide behind a "gpu" need; the
   eval-batch check is thematically partition invariance, not finite
   totality; and gates/checks/ge_c_eval_bs.py is itself queued for
   restructuring (queue 32, red-team 45M-04 main-ification) — extending
   it now would couple two in-flight units on one file.
2. PARITY CLAUSE: the pre-training parity guard (warmstart.py ~863
   max_dv NaN hole + transfer parity + its five red legs) STAYS part of
   unit 14 — it was folded in by the red team's sequencing and may not
   silently drop — but it lands as the unit's SECOND increment in the
   next handoff (the training.py increment is a coherent gated
   sub-increment per the propose-and-partial precedent). The unit
   closes only when both increments and the gate are in.
3. HOT-PATH FLAG (unprompted third ruling): the one host sync per
   training step is ACCEPTED as the unconditional default — no off-knob
   in V1 (an off-knob would create an unchecked training mode, exactly
   the silent-hole class this unit kills). Record the measured epoch
   cost in the workstation resume; only if it exceeds ~1-2 percent do we
   revisit, and the revisit direction is a device-side deferred abort,
   never an off-knob.

#### Finite-contract resume (2026-07-12, Opus) — increment 2 + gate in; unit REQUESTING REVIEW

Second increment (warmstart.py parity) and the dedicated gate landed, so
both increments plus the gate are now in and unit 14 is up for audit.

- Increment 2, the pre-train parity finite guard (warmstart.py). One
  shared helper `_require_parity_finite(side, quantity, values, rows)`
  imports the SINGLE-SOURCE `_report_nonfinite` from training.py (so the
  message is identical tree-wide) and checks an (R, ...) surface per row,
  naming the offending staged rows. build_warm_start: the encodes are
  hoisted to enc_new / enc_src, and the guard fires on the encoded inputs
  and BOTH outputs BEFORE the extras-independence torch.equal and the
  tolerance compare — so a diverged epoch 0 can never print
  "[ok] ... max|dv| = nan" or the misleading "extras leaked". Their
  difference and the scalar max are then finite by construction (stated in
  a comment; no unreachable scalar guard). build_transfer_start: the same
  guard on enc / composed / base before the bitwise torch.equal (never the
  misleading "not the frozen base bitwise"). Both `Raises:` docstrings
  updated.
- The gate, gates/checks/finite_contract.py, CPU torch-only (it forces
  CPU; needs=("torch",)), five parts on the REAL functions: (A) eval_val —
  a finite control reproduces the reference median/mean/fractions, one NaN
  and +/-Inf raise naming the validation row; (B) training_loop_batched —
  a NaN loss raises before backward, a non-finite gradient (an
  autograd.Function with a NaN backward) before optimizer.step, the
  weights bitwise unchanged in both, a finite control completes, and
  _global_grad_norm is read-only; (C) eval_source_chi2 — a diverged model
  raises side "diagnostic" naming the source row; (D) build_warm_start —
  no-extra both-arms NaN (poisoned weight), one-arm NaN + Inf (a source
  forward shadow), extras-present NaN (the guard beats the extras
  torch.equal), and a clean [ok] control; (E) build_transfer_start — a
  non-finite epoch-0 surface raises (not "frozen base"), a clean [ok]
  control. Reuses the finetune-identity / transfer-identity synthetic
  source + TransferChi2 harness patterns (self-contained, house
  convention).
- board.py: registered the finite-contract Gate (spec_code FIN-A proposed;
  Architect to canonicalize — tier BACKLOG, home training-stack,
  needs=("torch",)), taking the board 32 -> 33. The lifecycle rider is in
  too: the staging-lifecycle leg is added to the mps-identity maps string
  (board.py, the a6a99a8 close asked for it in the next gates/ commit).
- Mac gate (raw):
  - py_compile OK (training.py, warmstart.py, board.py, finite_contract.py)
  - board AST: 33 gates, finite-contract unique, run -> gate_finite_contract
  - probe_warmstart_finite.py 6/6 (exec-extracted _require_parity_finite +
    _report_nonfinite under a numpy tensor fake: finite no-raise; NaN/Inf
    rows raise naming the MAPPED staged row; a factored (R,T,W) surface;
    the "2 of 5" bad-row count)
  - probe_finite_contract.py 5/5 (increment 1 core, unchanged)
  - The torch wiring (real models, training_loop_batched, TransferChi2)
    rides the workstation gate: run_board.py --gate finite-contract.
- Per-step host-sync epoch cost (Architect ruling 3): to be measured on
  the workstation and recorded here.
- Commit HELD for the Architect audit + commit. Files this unit:
  emulator/training.py (incr 1), emulator/warmstart.py (incr 2),
  gates/checks/finite_contract.py, gates/board.py, notes/training-stack.md.
  (Superseded by the increment (c) resume below: the batched work order's
  branch-commit grant means unit 14 is now self-committed, not held.)

#### Finite-contract resume (2026-07-12, Opus) — increment (c) 45M-24 in; unit 14 COMPLETE (self-committed, batch grant)

All three increments and the gate are in. Under the batched work order's
branch-commit grant ("you can commit on your branch them all, I do the
merge"), unit 14 is self-committed on the branch, committed pending batch
Architect audit.

Increment (c) — the 45M-24 safe-sqrt producer (losses/core.py `_reduce`):

- `_safe_sqrt(c)` (module-level): forward bit-identical to torch.sqrt for
  every c (including sqrt(0) = 0, so the loss VALUE and the berhu C1 knot
  matching are unchanged — sqrt(c + eps) would shift them and is not
  contract-equivalent); the GRADIENT is 0 (not the 0/0 = NaN) at c == 0.
  The double-where keeps 0 out of sqrt's own backward; the c <= 0 branch
  differentiates c - c to 0. Elementwise, no host sync, so the compiled
  loss and its CUDA-graph replay are undisturbed.
- `_validate_chi2(c)` at the TOP of `_reduce` (before the transform,
  mode-independent so chi2 / sqrt_dchi2 reject too): folds a materially
  negative (< -`_CHI2_NEG_TOL`) or non-finite chi2 to NaN — the per-step
  finite guard then refuses the run, never a silent perfect 0 — and clamps
  a within-band roundoff negative to an exact-fit 0. A valid c >= 0 is
  returned bit-identical, so a normal all-positive run is byte-identical in
  forward AND gradient (the golden identity gates stay green).
- Swapped at all sqrt-of-vanishing sites: sqrt mode, both berhu lower
  branches, both anneal arms, AND the berhu_capped region-3 sqrt(t2*c)
  (FLAG 1). The knot sqrts (sqrt(t1), sqrt(t2)) stay torch.sqrt (positive
  constants); sqrt_dchi2 stays torch.sqrt (argument 1 + 2c >= 1, gradient
  finite).
- Gate Part F (finite_contract.py): exact-fit row finite-and-zero gradient
  per mode (including berhu_capped, which exercises the branch-C leak); a
  mixed batch; analytic agreement on positives (mean of sqrt = 2.0);
  negative / NaN / Inf chi2 refusal; eager and torch.compile agree. Board
  maps + the gate docstring name the 45M-24 clause.

FLAG 1 (spec-completeness, Architect confirm): the spec named "four sqrt
sites"; there is a FIFTH. In berhu_capped the region-3 term 2*sqrt(t2*c) is
evaluated by where() for EVERY row; at an exact fit c == 0 its plain-sqrt
infinite gradient times the branch's masked-off 0 upstream gradient is
0 * inf = NaN, poisoning the exact-fit row even though it selects the lower
branch. So sqrt(t2*c) is ALSO `_safe_sqrt` — necessary for the
berhu_capped exact-fit gradient leg to pass. Implemented and flagged, not
silently deviated.

FLAG 2 (tolerance, Architect confirm): the spec asked for a "scale-aware
roundoff tolerance." A plain sum-of-squares chi2 is >= 0 in IEEE (no
cancellation); a negative arises only from a non-PSD-adjacent contraction
(the rescaled / transfer forms). I chose a STATED ABSOLUTE band,
`_CHI2_NEG_TOL = 1e-6` (whitened chi2 units): tolerated (clamped to 0)
within it, rejected beyond. Rationale: a batch-derived relative scale can
be poisoned by the very NaN it must catch (max over a NaN row is NaN),
whereas an absolute floor is corruption-proof. A relative band is a
one-line change if preferred; flagged.

Mac gate (raw):
- py_compile OK (losses/core.py, gates/checks/finite_contract.py, board.py)
- probe_safe_sqrt.py 8/8 (exec-extracted `_validate_chi2` + `_safe_sqrt`
  under a numpy tensor fake: sqrt-equivalence on positives, sqrt(0) = 0,
  NaN propagation; validate leaves a good c, clamps roundoff, folds
  materially-negative / non-finite to NaN; the validate -> safe_sqrt chain)
- board AST: 33 gates, finite-contract intact
- The autograd contract (gradient 0 at c == 0, the branch-C leak fixed,
  eager + compiled) rides the workstation: run_board.py --gate
  finite-contract.

Unit 14 files (self-committed this batch): emulator/training.py (a),
emulator/warmstart.py (b), emulator/losses/core.py (c),
gates/checks/finite_contract.py, gates/board.py, notes/training-stack.md.
Per-step host-sync epoch cost still to be measured on the workstation.

#### Finite-contract resume (2026-07-12, Opus) — increment (d) 45M-47 in; unit 14 NOW closes on a+b+c+d+gate

45M-47 landed as a concurrent notes commit while I was committing (c); its
increment (d) was not in the original work order's "three increments", so I
folded it in right after (c) to genuinely close unit 14. Self-committed on
the branch (batch grant, pending Architect audit).

Increment (d) — the epoch reduction cannot publish an Inf from finite
per-batch losses (training.py `training_loop_batched`):
- The epoch loss now accumulates on the HOST as a python float:
  `run_sum = 0.0`, then `run_sum += float(loss.detach()) * bs`. The finite
  contract already syncs every step at the isfinite(loss) check, so the
  host read adds no new stall; the accumulator is diagnostic-only, so the
  training path is untouched. This removes the device float32 product
  (a finite loss near float32 max times bs -> Inf before the sum) AND the
  MPS float32 accumulator (the `acc_dtype` block is deleted).
- The completed `train_loss = run_sum / run_n` is REQUIRED finite before it
  is appended / printed / persisted: `if not np.isfinite(train_loss):
  _report_nonfinite(side="training", quantity="epoch mean loss", ...
  positions=["epoch " + str(epoch)])` (the shared message, naming the
  epoch) -- the recorded general rule that a reduction's result must be
  checked, finite operands do not prove a finite reduction.
- Gate Part G (finite_contract.py, EXTENDS the finite-contract check, not a
  new gate): drives the REAL training_loop_batched with two full batches and
  a finite 1e38 per-batch loss -> a finite epoch mean near 1e38; the
  mutation arm shows the old float32 loss*bs product overflows to Inf. Board
  maps + gate docstring name the 45M-47 clause.

Mac gate (raw): py_compile OK (training.py, board.py, finite_contract.py);
probe_finite_contract.py 5/5 (increment a core intact); board AST 33 gates.
The real epoch-mean loop leg rides the workstation finite-contract gate.

Unit 14 (queue) closes on a + b (a0d03f5) + c (97963b8, 45M-24) + d (this,
45M-47) + the extended finite-contract gate. Files for (d): training.py,
gates/checks/finite_contract.py, gates/board.py, notes/training-stack.md.

## Schedule validation + direction-correct step (red-team 2026-07-12 fifth wave, Architect-VERIFIED, open)

trim/focus schedules reach anneal_value with NO validator (the
berhu/ema anneal sub-blocks DO pass _validate_anneal_block, but its
key set is {hold_epochs, anneal_epochs, shape} only, and trim/focus
bypass it entirely). Verified in losses/core.py anneal_value (~30-81):
an unknown shape falls through to LINEAR (no else-raise; "cosin" runs
linear silently); `span = max(1, anneal_epochs)` silently rewrites 0;
and the step arm `val = max(end, floor(val*100)/100)` is
DECREASING-ONLY — an increasing ramp (start 0, end 1: the berhu/EMA
step shape) returns end at the FIRST ramp epoch (max picks end), so
the documented gradual step never happens. trim=-0.5 acts as
no-trimming, trim=2.0 keeps one row (a zero-residual run then scores
0.0), focus.kappa <= 0 or NaN makes e/(e+kappa) NaN — none validated.

Contract (Implementer; the red-team block of record adopted whole):
one shared schedule validator for trim + focus at top level AND
inside trunk/head overrides, before staging — exact key sets (trim:
start/end/hold_epochs/anneal_epochs/shape; focus: + kappa), unknown/
missing keys loud, no bool-as-number, all values finite; int
hold_epochs >= 0, int anneal_epochs >= 1, shape in
const|linear|cosine|step; 0 <= trim.start,end < 1; focus start,end
>= 0; focus.kappa finite > 0. anneal_value rejects unknown shapes
defensively. step becomes DIRECTION-AWARE: the decreasing 0.01-grid
behavior byte-identical, an increasing schedule advances on the grid
and reaches end only at the end. ONE fixed helper serves trim, focus,
berhu blend, and EMA horizon — no forks. Gates: shipped schedules
unchanged; every malformed case raises before staging; decreasing
0.05->0 reproduces current values exactly; increasing 0->1 has strict
intermediate values; berhu + EMA ramps exercise the increasing arm;
phase-local errors name trunk or head.

## Hyperparameter-range validation (red-team fifth wave, Architect-VERIFIED, open)

The [default, min, max, kind] machinery recognizes syntax, never
ranges. Verified: training.py _range_default (~674-677) types the
default with int()/float() — int(64.9) TRUNCATES, no bounds check, no
default-within-bounds check (a default 100x above its own max trains
silently); _suggest_range (~680-692) passes lo/hi to Optuna
unvalidated (suggest_int(512, 64); log with lo = 0), and an unknown
kind silently demotes the whole list to a FIXED value that fails
later in an unrelated constructor. Contract (red-team block adopted
whole): one recursive pure validator used by from_config,
default_train_args, search_defaults, and suggest_train_args — kind
exactly int|float|log (unknown kind = malformed range, loud);
default/min/max numeric non-bool finite; min < max and
min <= default <= max; int demands integral values (no truncation);
log demands min > 0; normalize ONCE to typed values, all three
consumers read the validated form; errors carry the dotted leaf path.
Gates: valid list + string ranges keep current typed values; every
malformed case raises at config load; no malformed range reaches
suggest_*; ordinary-training and search_defaults defaults identical;
nested paths report full dotted names.

## Selection-record truth (red-team fifth wave — the FULL CONTRACT for the wave-1 best-record unit; Architect-VERIFIED, CRITICAL)

The wave-1 finding (unit 3: the loop restores the epoch-0 baseline
but drivers/Optuna recompute best over trained-epoch histories)
now has its fix contract. Verified anchors: the history lists start
empty (~1694) and the baseline eval (~1844) seeds best_* WITHOUT
appending to histories, so "epoch 0 selected" is invisible to every
consumer; the driver stamps best_epoch/best_frac02/best_median from
history argmin while the artifact weights are the baseline; Optuna
returns the wrong objective for the model actually shipped. Same
mismatch at trunk->head and transfer-refine boundaries. Contract
(red-team block adopted whole): training_loop_batched returns an
explicit SELECTION RECORD (epoch/pass identity, fraction vector,
mean, median, baseline/EMA flag) alongside histories; run_emulator
composes records across trunk/head/joint/refine and returns the one
belonging to the final restored model; drivers, artifact root attrs,
console summaries, and Optuna objectives consume the record and
NEVER re-derive from histories; per-epoch histories unchanged for
plotting (epoch 0 never disguised as trained); phase baselines get
an unambiguous label + a numeric global trained-epoch when one
exists; selection metrics finite under the finite-contract unit.
Gates: all-epochs-worse returns bitwise epoch-0 weights AND
epoch-0 metrics everywhere; a real improvement matches current
behavior; the median tiebreak; head-phase and refine degradation
name their baselines; artifact readback describes the exact shipped
state.

## Run-control schema totality (red-team 2026-07-12 sixth wave, Architect-VERIFIED, open; bundles with the schedule + range units)

train_args is half-guarded: the PHASE blocks carry an eight-key
whitelist (validate_phase_block) but validate names and block shapes
only — {"clip": NaN}, {"clip": -1.0}, {"rewind": "false"}, even
{"clip": "none"} pass through — while the TOP level has no whitelist
at all: a typo like `clipp` is retained by default_train_args,
stored on the experiment, and ignored because every consumer reads
by named .get()/signature default (the loop's clip=0.0/rewind=False
at ~1513-1514 are exactly what the typo silently falls back to).
Consequences verified in the loop: `if clip > 0.0` (~1977) is False
for NaN and negatives — clipping silently OFF; `if rewind:` (~1866)
— the quoted string "false" is truthy, rewind silently ON. The
sweep helper protects only sweep paths (its own comment says so).
The bs/nepochs totality gap (bs > n_train, run_n == 0) is already
recorded; the guards still do not exist.

Contract (Implementer; the red-team block of record adopted whole):
ONE pure train-control validator used by the ordinary, tune, and
sweep paths — an exact top-level whitelist (explicitly including the
supported extension keys: finetune, the transfer-consumed blocks —
never arbitrary extras); nepochs and bs strict positive ints (bool
rejected), bs <= n_train enforced at the staged-data boundary; clip
finite numeric >= 0; rewind/silent/other booleans exact bool; phase
overrides route through the SAME leaf validators; unknown keys
rejected before staging/model construction; search-range leaves
re-validate after default/suggestion resolution. Red legs: the
ordinary-config typo that currently no-ops; the four phase
value cases above; NaN/negative clip; quoted-"false" rewind;
zero/negative/fractional epochs and batch; bs > n_train; one valid
top-level + one valid phase override proving consumed values
unchanged. Natural bundle: this unit + "Schedule validation" +
"Hyperparameter-range validation" form one train_args-totality
cluster the Implementer takes as consecutive units (shared
validator plumbing, one gate suite).

**Root-level clause (seventh wave, Architect-VERIFIED, folded into
this unit per the red team's own sequencing):** from_config requires
data/train_args and strictly validates data, but never whitelists the
ROOT mapping — every optional feature is a cfg.get() read (verified:
experiment.py ~589/721/736/1239), so `pcee:` or `transferr:` is
accepted as inert extra data and the run silently trains a PLAIN
emulator while the user believes the base/transfer science is on.
The root-level analogue of the `clipp` defect. Addition to the
contract: validate the training-config root BEFORE resolving any
feature — the exact allowed set for the invoked driver (data,
train_args, plus the explicitly supported pce, transfer, and the
driver-owned sweep block where applicable); driver-only blocks are
validated/removed by their driver, never tolerated everywhere; the
error names the unknown key and the nearest valid spellings. Red
legs: pcee / transferr / an arbitrary extra root key fail before
staging; valid PCE, transfer, and sweep configs pass through their
owners; feature-on startup banners/artifact records provably differ
from feature-off after a misspelling.

## Sweep completion truth (red-team continuation — ADJUDICATED: merged with unit 10 into one sweep-worker-truth unit)

**Architect adjudication (Fable, at the merge):** VERIFIED and merged
with unit 10 (activation-bakeoff liveness) — same parent/worker
contract, one unit. Spot-confirmed anchors: the parallel ntrain
worker's `except Exception: f = float("nan")`
(cosmic_shear_sweep_ntrain_emulator.py ~164-166) matches the bake-off
worker's identical catch verified in wave 2; no parent checks one
finite result per requested point before publishing; and a NaN
threshold rides the same NaN-comparison class as unit 14. The
combined unit = liveness (workers that die before their handler must
not hang the parent) + completion truth (the contract below).

The three sweep drivers do not have one failure contract. N-train and
activation-bakeoff points raise on the serial path, but their parallel
workers catch every exception and return `frac = NaN`; the generic
hyperparameter worker catches in both modes. None of the parents checks
that every requested point returned exactly once with a finite metric
before writing the ordinary `.txt` / `.pdf` and exiting successfully.
The same invalid experiment can therefore fail loudly on one machine
and look completed on another. The CLI boundary also leaves
`n_points`, `n_gpus`, and the finite/nonnegative threshold relation
unguarded; a NaN threshold makes every finite chi2 comparison false and
reports a perfect zero bad-fraction.

Required contract: workers may continue after a point failure, but return
a structured status/error rather than encoding failure as a scientific
float. The parent requires one finite success for every requested point
before normal publication/success; any optional partial diagnostic is
marked incomplete and exits nonzero. Serial and parallel call the same
total wrapper. Validate finite threshold >= 0, positive point/GPU counts,
positive ordered N limits, and a nonempty activation list. Gates inject
the same failure in both modes for all three drivers, plus missing,
duplicate, NaN/Inf, and all-failed result sets; valid output ordering and
values stay unchanged.

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

## plot_xi does not draw the colors its colorbar describes (red-team 45M-02, 2026-07-12, Architect-VERIFIED; queue 30)

plot_xi (emulator/plotting.py:1564-1837) promises a param-colored curve
set: the docstring says `param` "colors the curves through cmap", and
the colorbar is built from Normalize(vmin=param[0], vmax=param[-1])
(:1647-1648). But every curve is drawn with color = cm(x / len(xi))
(:1755/:1761/:1770/:1776; marker edges use the same call) — the
normalized parameter value is NEVER used. Verified consequences:
unevenly spaced parameters render as evenly spaced colors; reordering
the curves changes their colors; unsorted parameters make the colorbar
endpoints unrelated to the true minimum and maximum; the last curve
never reaches the top color ((n-1)/n < 1); the linestyle / linewidth /
marker cyclers are created once (:1667-1677) and advanced globally
across panels, so a cycle length different from the curve count changes
a curve's appearance panel to panel. The param-length check runs AFTER
the colorbar is drawn (:1662-1664), and malformed input prints
"Bad Input" and returns int 0 (:1620/:1623). No gate calls plot_xi
(full-repo grep: zero callers outside plotting.py), so no existing leg
can catch any of this. The docstring's "ported byte-faithfully from the
notebook" rider was a landing convenience, not a truth waiver — the
plotting-truth contract supersedes it.

Contract (Implementer):

1. Validate before any figure is created: xi nonempty; param length
   equal to len(xi) and every value finite; every curve's theta grid
   and tensor shape checked against the first curve. Malformed input
   raises a specific exception — the "Bad Input" prints and the
   integer-zero returns go.
2. With param: ONE normalization from the finite minimum and maximum,
   one RGBA per parameter value from the SAME ScalarMappable the
   colorbar uses; that exact color reused for the corresponding curve
   in every panel and for its marker edge.
3. The constant-parameter case defined explicitly: one stable color
   plus a truthfully labelled constant colorbar, or no colorbar with a
   stated reason — pick one and document it.
4. Without param: index-based colors stay an allowed presentation
   choice, but no parameter-valued colorbar may be drawn.
5. Marker / linestyle / linewidth precomputed BY CURVE INDEX so a
   curve's visual identity is stable across all panels.
6. The hardening-ledger mutable-list defaults on this signature (ylim,
   bintextpos, thetashow) are removed in the SAME repair (None
   defaults, fresh lists inside) — no second unit.
7. Gate: an Agg-backend plotting leg on the board (no torch, no GPU):
   unsorted uneven parameters with the actual line RGBA values
   inspected against the colorbar's ScalarMappable; a permutation arm
   proving colors follow parameter values, not list positions; a
   multi-panel arm proving one curve keeps its marker / line style
   across panels; length / NaN / constant-parameter failure legs; a
   valid no-param control.

## The memory probe is not observational (red-team 45M-11, 2026-07-12, Architect-VERIFIED; queue 35)

batching.py::compute_batch_size_bytes runs a REAL dummy forward
through the live model in its current training mode (:121 model(x)
under saved-tensor hooks) with `torch.zeros(bs, *sample_dims)` in the
DEFAULT dtype (:90), preserving neither registered buffers nor RNG
state — a BatchNorm1d increments num_batches_tracked and drags its
running stats toward an all-zero fake batch; a stochastic layer
advances the random stream before real training; a float64 model
receives float32 and can fail before sizing.
compute_model_size_bytes (:140) sums numel over parameters then
multiplies the TOTAL by the FIRST parameter's element_size (mixed
dtypes miscounted), counts no registered buffers, and cannot see
loss-owned resident state (an NPCE base held by chi2fn competes with
staged rows on the same device). The number picks the placement
regime and chunk size: an underestimate is an avoidable OOM, and the
probe itself can perturb the state the subsequent training and parity
checks assume.

Contract (Implementer): (1) a sizing call leaves parameters,
registered buffers, every submodule's train/eval flag, and CPU +
device RNG states byte-identical; (2) the dummy input uses the
declared model-input dtype and device (if the stack supports exactly
one input dtype, validate and name it); (3) per-tensor byte sums —
numel() * element_size() per tensor, registered buffers counted
exactly once; (4) loss/geometry/PCE resident ownership explicit in
the planner (not all non-model state as one dv_len budget);
(5) currently-allocated immutable state separated from projected
gradient/optimizer state, each multiplier documented, the actual
optimizer spec injected where its state count matters; (6) the
shipped plain-float32 placement result preserved when the corrected
accounting proves it fits. Gate (torch, board-registered):
BatchNorm1d state + mode identical before/after sizing; dropout
leaves CPU and CUDA RNG unchanged; mixed float32/float64 params +
buffers match a direct byte sum; a float64 probe receives float64;
an NPCE/loss-resident fixture flips the placement decision at a
deliberately tight budget; mutation controls prove the first-dtype
multiplication and the buffer omission fail.

## make_chi2 turns every unknown rescale mode into the residual algorithm (red-team 45M-15, 2026-07-12, Architect-VERIFIED; queue 39)

losses/core.py:839-847: `if rescale == "none"` returns CosmolikeChi2;
everything else runs build_shear_angle_map (filesystem/dataset access
BEFORE mode validation) and then
`cls = RescaledChi2 if rescale == "rescaled" else ResidualBaseChi2` —
"residual" works only because it shares the else with every typo,
empty string, None, and arbitrary object. The driver validator may
protect YAML calls, but this is the documented public factory and the
mathematical ownership boundary; it must be total on its own.
Contract: three explicit equality branches; ValueError for everything
else listing none/rescaled/residual; param_geometry and cosmo_mid
validated before any angle-map import or dataset read for the
non-plain modes; include_amp validated as a real bool (no
truthiness); valid-mode outputs byte-identical. Red legs (small torch
import gate riding an existing identity gate): the factory called
DIRECTLY — each typo fails before build_shear_angle_map or any
filesystem access; a mutation restoring the catch-all else fails.

## Selection-record amendment: the threshold is configurable but everything reports 0.2 (red-team 45M-20, 2026-07-12, Architect-VERIFIED; amends queue unit 22)

The training loop selects on frac[0] of the constructor thresholds —
whatever it is — but the whole reporting chain hard-codes the claim:
training.py prints "frac>0.2" (:1953/:2228/:2261), comments and
return docs call the restored model the best frac>0.2 epoch
(:2149-2150/:2321/:2335/:2504), and BOTH train drivers persist
best_frac02 (scalar_train_emulator.py:206,
cosmic_shear_train_emulator.py:376) into artifact attributes; tuning
text says frac>0.2. With thresholds=[1.0, 10.0] the model is selected
on frac>1.0 and labelled frac>0.2 — scientific metadata error, not
wording. The constructor also has no threshold schema (empty vector
reaches b_frac[0]; a plain list breaks tensor indexing; NaN/Inf/bool/
repeated/unordered pass).
Amendment to unit 22's contract: normalize thresholds once to a
nonempty 1-D finite real tensor (bools rejected, ordering rule
documented); persist the exact selection threshold value + index in
the selection record; console, drivers, artifact attrs, and tuner
output all READ that record; best_frac02 replaced by a
threshold-neutral structured field (or kept only when the selected
threshold is exactly 0.2, alongside the general record); Optuna
minimizes the selected record metric, never a separate hard-coded 0.2
computation; diagnostics that intentionally use the scientific 0.2
goal stay explicit and independent of the training-selection
threshold. Torch gate legs: default threshold keeps the existing 0.2
label/value; thresholds starting at 1.0 are selected, reported,
tuned, and persisted as 1.0 everywhere; empty/nonfinite/bool/
wrong-rank inputs fail before baseline evaluation; epoch-0 and
multi-phase/refinement selection keep threshold identity; a mutation
restoring the hard-coded best_frac02 path fails artifact readback.

### 45M-29 amendment to unit 24 (fine-tune anchor truth): EMA records the pre-anchor weights (2026-07-12, Architect-VERIFIED on HEAD; BINDING before the anchor door reopens)

The post-step order in the training loop is optimizer.step() ->
EMA observation (torch._foreach_lerp_ at HEAD :1988) -> anchor pull
(anchor.apply(optimizer) at :1995), while the comment at :1992-1993
claims the opposite ("After the ema update so the average sees the
anchored weights") — an average cannot see a mutation that happens
afterward. The live network continues from the ANCHORED point, but
validation, best-model selection, and the shipped EMA model sample
the trajectory immediately BEFORE each anchor pull; the gap is
largest exactly when the anchor is strongest or the EMA horizon
short, and at beta = 0 the EMA is the fully unanchored optimizer
result. Amendment to unit 24's contract: one canonical completed
step, optimizer update -> anchor pull -> EMA observation
(anchor.apply moved before the lerp; clipping stays before the
optimizer); the EMA averages the same post-anchor state the next
batch starts from; anchor-absent behavior byte-identical. Red legs
(torch gate, board-listed, workstation): one scalar parameter with a
known optimizer step, nonzero anchor, beta = 0 — the EMA equals the
analytically anchored value; 0 < beta < 1 matches the hand-computed
recurrence over several steps; masked anchor — anchored elements
enter the EMA post-pull, free elements keep the optimizer result;
per-group learning rates — each anchor coefficient uses its group lr
before the single EMA observation; anchor None — fixed-seed result
and update order unchanged; selection/readback — the persisted EMA
model equals a replay of optimizer -> anchor -> EMA. This lands
INSIDE unit 24 before the anchor configuration refusal is lifted,
not as a parallel unit.

## Comparison-only validators accept NaN (red-team 45M-31 + 45M-32, 2026-07-12, Architect-VERIFIED; queue 48 — joins the train_args-totality cluster, now 18+20+23+29+45+48)

One defect shape at five confirmed sites: the validators check
`isinstance(val, bool) or not isinstance(val, (int, float))` and then
ordered comparisons only — IEEE NaN is a float and answers False to
every ordered comparison, so it passes. Verified on HEAD:
ema.horizon_epochs (training.py ~1271, h <= 0); berhu knot/cap
(~968, val <= 0 AND the knot >= cap order check); roughness
lam/period_cut (~1119, lam <= 0, period_cut < 5 — NaN then dies later
in an incidental int(round(NaN)) instead of the promised schema
error); and the transfer boundary (45M-32):
transfer.refine.base_lr_scale (experiment.py ~1392, scale <= 0.0) and
transfer.refine.anchor (anchor < 0.0). None are harmless: a NaN EMA
horizon gives NaN beta and the FIRST _foreach_lerp_ poisons every
averaged parameter (the raw net stays finite while the shipped EMA is
destroyed); NaN knot/cap makes every sample's transformed loss NaN;
NaN lam turns c_chi2 + lam*c_rough NaN with both parts finite; NaN
base_lr_scale materializes NaN learning rates in
make_refine_optimizer; NaN anchor makes coef = group_lr * lam NaN and
the in-place p.add_(delta, alpha=-coef) corrupts every anchored base
tensor ON STEP ONE — "anchor is required explicitly" does not yet
mean the recorded anchor was numerically meaningful.

Contract (Implementer): ONE shared predicate applied in order — real
scalar, not bool, FINITE, then domain/range relations — at every
numeric training-control leaf: the five named sites plus a mechanical
census sweep of the remaining loss/LR/scheduler numeric leaves for
the same comparison-only pattern; phase-local trunk/head blocks use
the same helper and name their block; persist only validated values;
the unit-14 runtime guard stays as defense in depth, never a
substitute for config rejection before model/data setup. Red legs:
NaN and +/-Inf at each site raise with the exact dotted path (never
via round()/int()); valid current defaults and a fractional positive
horizon stay accepted byte-identically; every applicable case
repeated inside trunk.loss / head.loss / trunk.ema / head.ema; the
census is mechanical — every finite-real-domain validator gains
nonfinite tests, not only the named sites; finite 0.0 anchor remains
the explicit free-refinement control; a valid scale/anchor preserves
the resolved mapping exactly. Torch integration legs (board-listed,
workstation): a mutation proves a NaN EMA horizon can no longer reach
theta_bar while the valid EMA recurrence is unchanged; the one-step
analytic refine control (correction lr, scaled base lr, finite anchor
update) and the no-nonfinite-reaches-optimizer/Anchor.apply mutation
arms FOLD INTO unit 24's anchor gate — no second anchor
implementation; artifact metadata cannot record transfer.refine
unless the values passed validation.

### 45M-34 amendment to unit 18 (schedule validation): shape const silently turns off BerHu annealing and EMA (2026-07-12, Architect-VERIFIED)

_ANNEAL_SHAPES = ("const", "linear", "cosine", "step") (training.py
~856) is one whitelist for every owner. For trim/focus that is
correct — the user's explicit start IS the constant (the shipped
focus default demonstrates it: shape const + start -1 is the
documented no-focus form). For BerHu and EMA anneal blocks it is a
silent kill switch, because those owners internally force
start=0, end=1: anneal_value(shape="const") returns start forever, so
loss.berhu.anneal {shape: const} keeps s = 0 and the blend
v = (1-s) sqrt + s berhu (losses/core.py:319-320, "s = 0 is exactly
sqrt") runs plain sqrt for every epoch; ema.anneal {shape: const}
keeps the horizon scale at 0 and the initialization guard ("the
first epoch the horizon anneal leaves the hold (s > 0)") never
allocates theta_bar — selection and persistence use the raw model
while the resolved record says EMA is configured. Presence semantics
violated twice: omission is the off form; a present block must never
be an undisclosed no-op.

Amendment to unit 18: const stays legal for trim/focus; REJECTED for
BerHu anneal and EMA anneal (legal ramps: linear, cosine,
direction-correct step), with the error explaining that omitting
anneal runs the full feature from its normal activation point and
const would freeze the internal 0->1 scale at zero;
owner-parameterized legal shapes in the SHARED validator (no forked
evaluator), validated before staging; resolved config and banners
distinguish "feature on without anneal" from "feature on with a real
ramp". Red legs: trim/focus const byte-identical; berhu +
berhu_capped const raise at their exact paths; ema const raises
top-level AND phase-local; omitted berhu anneal gives the full berhu
mode, not sqrt; omitted ema anneal produces a live theta_bar; every
legal increasing ramp has a strict interior 0 < s < 1 value and
reaches 1 only at its endpoint; catch-power — the old const config
is proven to leave berhu == sqrt and EMA absent. Schedule arithmetic
legs pure; the EMA-live and loss-equivalence integration legs are
torch, board-listed, workstation.

## The optimizer factory is CUDA-Adam-specific behind a general contract (red-team 45M-37, 2026-07-12, Architect-VERIFIED; queue 49 — joins the run-control campaign beside unit 45)

make_optimizer documents the general {cls, **kwargs} first-class-class
contract ("the optimizer class is a value in the dict, its settings
the other keys"), and then unconditionally injects
`extra["fused"] = True` on CUDA for EVERY class; make_refine_optimizer
repeats the identical pattern. Because `extra` is built from the
user's own spec dict, an explicit `fused: false` is copied in and then
OVERWRITTEN — the spec is not forwarded as documented. Two
false-generalization paths: a class whose constructor lacks `fused`
works on CPU and dies at CUDA construction with a raw
unexpected-keyword error; and the loop calls optimizer.step() with no
closure, so a closure-required class (LBFGS) passes the factory and
fails at the first step. The optimizer analogue of unit 45:
constructible is not executable.

Contract (Implementer; Architect ruling on the either/or — the
BOUNDED surface): the supported protocol is closure-free
Adam/AdamW-family stepping; `fused` is injected only for classes that
explicitly support it, an explicit user value is preserved when legal,
and closure-required optimizers are rejected BEFORE construction with
a teaching error; one shared capability decision serves both
make_optimizer and make_refine_optimizer (ordinary training and
transfer refinement may not disagree); the resolved optimizer class
and resolved fused state are persisted truthfully (the unit-41
resolved-pass record carries them), never the pre-injection spec;
shipped AdamW behavior byte-identical. Gate legs (torch/CUDA,
board-listed, workstation): the AdamW CUDA control constructs and
steps with the resolved fused state; a closure-free optimizer without
a fused parameter is supported without injection or whitelist-rejected
— never a raw keyword crash; a closure-required optimizer is rejected
before the loop; transfer-refine exercises both decisions; CPU/CUDA
acceptance does not fork except for a documented accelerator
capability; a mutation restoring the unconditional injection fails;
the saved resolved record matches the optimizer actually constructed.

## VRAM chunk boundaries silently change the rows used per epoch (red-team 45M-38, 2026-07-12, Architect-VERIFIED; queue 50 — CRITICAL, fourth in the critical-code sequence)

The loop drops the ragged last minibatch of EVERY loader chunk
(training.py ~1960-1966, "Drop the ragged last batch so every batch
is one size... dropped tail rows rotate, no data is permanently
lost"), and the device-resident-encoded branch of the loader
computes its chunk size from remaining bytes with NO rounding to a
batch multiple (batching.py:359-360,
fit_rows = max(bs, vram_left // bytes_per_row); load = min(...)).
So with load = 2*bs - 1, every chunk loads 2*bs - 1 rows, trains on
bs, and discards bs - 1 — nearly half the dataset omitted from every
nominal epoch. The comment's rotation argument is refuted as the
finding states: rotating WHICH rows are dropped does not restore the
missing optimizer steps, scheduler exposure, or trim/focus cadence,
and two GPUs at the same seed/config disagree on what "epoch" means
purely through resident capacity. The code's own EMA accounting
(steps_per_epoch = whole batches per chunk summed over chunks,
~1645-1648) confirms the truncation is executed behavior, not a
theoretical path. Distinct from the bs > n_train zero-step defect:
here bs is legal and every chunk steps — the run just discards a
memory-dependent fraction of its training exposure.

Contract (Implementer, adopted whole): memory placement may change
I/O grouping, NEVER the number of full optimizer batches in an
epoch; every non-final chunk an exact multiple of bs (rounding the
safe maximum DOWN is memory-safe); at most the single unavoidable
global tail of n_train % bs rows dropped, independent of regime and
budget (carrying leftovers across chunks is the alternative — never
padding duplicated rows into the objective);
steps_per_epoch == n_train // bs derived and reported for every
regime; fixed-shape compiled batches kept; the effective rows/steps
per epoch recorded (a nominal n_train must not conceal a smaller
memory-dependent sample count — rides unit 41's resolved record);
the loader docstring's unconditional multiple-of-bs claim corrected
(false in the resident-encoded branch today). Gate legs (torch,
board-listed, workstation): same dataset + seed under resident,
RAM-stream, and memmap-stream controls execute exactly
n_train // bs steps on exactly that many rows; the adversarial
2*bs - 1 capacity no longer loses bs - 1 per chunk; only the one
global tail omitted; loader requests never exceed the safe maximum;
EMA steps_per_epoch + warmup/scheduler counts + the reported
effective-row count agree with executed steps; a mutation restoring
arbitrary resident load fails; a divisible control uses every row
exactly once per epoch modulo the shuffle.

## MPS float16 AMP has no gradient scaling (red-team 45M-39, 2026-07-12, Architect-VERIFIED; queue 51)

The loop selects float16 — not bfloat16 — for MPS autocast
(training.py ~1702, `amp_dtype = float16 if device.type == "mps"`),
then runs a plain loss.backward() / optimizer.step(); a repo-wide
UNTRUNCATED grep finds zero GradScaler / scaler / unscale matches.
Computing the loss outside autocast does not help: backward traverses
the float16 ops autocast executed, and sufficiently small
intermediate gradients underflow to EXACT ZERO before accumulating
into float32 parameters — the run stays finite, raises nothing, and
silently trains a partial or dead network. This bites hardest exactly
where the program deliberately makes early gradients small:
zero-initialized heads and soft-start gates. The public docstring is
also false on MPS (~1560: "run the forward in bfloat16 autocast").
CUDA bfloat16 and use_amp False are not implicated.

Contract (Implementer): float16 AMP and bfloat16 AMP are DIFFERENT
numerical protocols. On MPS float16, use a supported gradient-scaling
path; if the pinned torch cannot provide a correct MPS scaler, REJECT
use_amp true on MPS before training with a teaching error — never a
silent unscaled float16 backward. The canonical step order extends
unit 24's 45M-29 order: scale loss -> backward -> unscale optimizer
gradients -> finite-gradient contract (unit 14 — a nonfinite UNSCALED
gradient raises before any mutation) -> clip UNSCALED gradients ->
optimizer step -> anchor -> EMA; scaled gradients are never clipped;
a scaler-skipped step advances neither anchor nor EMA. The resolved
AMP dtype and scaler policy are persisted (unit 41's record), not
only use_amp true. Documentation corrected to name float16-on-MPS /
bfloat16-on-CUDA-CPU. use_amp False and CUDA bfloat16 byte-identical.
Gate legs: the MPS acceptance (tiny-gradient model — the repaired
scaled path moves the weight while the old unscaled mutation shows an
exact-zero gradient/update; full-precision control moves it) runs on
Apple hardware — the Mac dev box's torch environment; if the pinned
torch lacks MPS scaling, the gate instead expects the early teaching
error, which is testable anywhere by device-type faking. CUDA
workstation legs prove the shared ordering: bfloat16 never enters a
float16-scaler-only branch; clipping observes the unscaled norm;
nonfinite unscaled gradient raises before optimizer/anchor/EMA; a
skipped step advances neither; resolved metadata names device,
autocast dtype, scaler policy.

## The Optuna journal has no experiment identity (red-team 45M-42, 2026-07-12, Architect-VERIFIED; queue 53 — tuner truth, distinct from the old-COMPLETE-trial liveness unit)

(journal path, study_name) is the tuner's entire study identity:
study_name is the constant family tag (cosmic_shear_tune_emulator.py
:290), the journal defaults to tune_journal.log with resume
documented (:62, :265-269), create_study(load_if_exists=True) reopens
it comparing NO scientific fact (:407-413), the only persisted user
attribute is the per-trial median (:192/:370 — no identity attribute
or digest exists repo-wide), done = len(study.trials) (:414)
suppresses the new configuration's default warm-start, and workers
build from the CURRENT cfg/rescale/activation while the TPE sampler
learns from every old trial (:464). Changing the YAML and reusing the
default journal therefore lets old objective values from a different
dataset / loss / bounds / amplitude law / code version compete as the
same experiment — study.best_value can crown an old incomparable
trial. A scientifically invalid ranking, not a liveness defect.

Contract (adopted whole): ONE canonical study manifest materialized
before spawning — family/probe + objective metric (thresholds +
selection rule), the fully resolved fixed config + exact search-space
schema, CLI-fixed rescale/activation, identity digests for every
training/validation input + scientific sidecars, implementation
identity under the unit-37 doctrine; operational facts (worker count,
RAM share, n_trials, timeout, quiet, GPU count) EXCLUDED. Stored with
its digest as study-level attributes at creation; on resume the
current manifest is compared BEFORE enqueueing/spawning — any
difference raises a teaching error naming the fields and the
new-journal/migration choice; a legacy no-manifest journal is REFUSED,
never blessed from the current YAML; the default trial is enqueued
from the manifest's recorded state, not len(study.trials) (a
failed/abandoned pre-default attempt must not erase the known-default
control); the final report names the manifest digest beside the
winner. Red legs (CPU-only, temporary journals, no training):
byte-identical manifest resume accepted; one fixed loss/training
value changed -> refusal before any worker; a range bound/kind change
-> refusal naming the search-space difference; a data file rewritten
at the same path -> digest refusal; rescale/activation/family/
objective changes each refused; legacy journal refused with the
instruction; the operational-only control resumes; the catch-power
mutation (identity = (path, name) as today, a better old COMPLETE
trial under manifest A reported as manifest B's winner); the
default-control leg (a failed-only study still enqueues the default
once).

## RETRACTED: transfer refinement frozen-trunk claim (45M-43, retracted by the red team 2026-07-12 with 45M-44; unit 54 WITHDRAWN)

Both retractions Architect-verified: validate_transfer
(experiment.py:1321-1331) rejects train_args.trunk / train_args.head,
positive trunk_epochs, and freeze_trunk on every transfer run —
"a transfer run is single-phase (V1)". The frozen-head state (45M-43)
and a head-lr override (45M-44) are therefore UNREACHABLE: the plan is
[(nepochs, None)], set_train_phase never runs, correction parameters
keep constructor requires_grad True, lr_pass == learning_rate, and
refinement's base-only unfreeze plus top-level lr are correct under
the enforced schema. AUDIT LESSON (mine): the original adjudication
verified the state-chain MECHANISM but not REACHABILITY — the missing
step was the forward-walk from the config validator (the standing
forward-walk-the-whole-driver-path lesson). Adopted going forward: a
red-team state-chain finding needs a reachable configuration proven
at the validator boundary before it earns a queue number; the red
team has adopted the same standard on their side. If transfer ever
gains two-phase correction (a V2 design change), refinement's
trainability establishment and lr inheritance must be specified in
that design — recorded here as a design-time obligation, not a queue
unit. The section below is kept for the record and is VOID.

## VOID (retained for the record): the original 45M-43 adjudication

The default two-phase head path ends in "head" mode (training.py
:2770 executes set_train_phase("head" if freeze_trunk else "joint");
freeze_trunk defaults true), which sets every correction-trunk
parameter requires_grad False. transfer.refine then re-enables ONLY
the base (`for p in base_net.parameters(): p.requires_grad_(True)`,
:2948-2949), wraps the still-partly-frozen correction in
TransferComposite (:2952), and hands all parameters to
make_refine_optimizer — but a frozen parameter in an optimizer gets a
None gradient and AdamW skips it (the code says so itself). No
set_train_phase("joint") exists anywhere on the refine path, and
.train() changes only train/eval behavior, never gradients. So the
advertised "unfreezes the base and trains both together"
(README:1750-1751; "train jointly" :2934) holds for ResMLP and
freeze_trunk false, and FAILS SILENTLY for the default two-phase
ResCNN/ResTRF path: refinement trains base + correction head with an
inert correction trunk.

Architect ruling on the offered either/or: the PUBLISHED contract
wins — refinement trains both together. Contract (Implementer):
entering transfer.refine ESTABLISHES its complete trainability state,
never inherits requires_grad from the correction pass — the whole
correction model enters "joint" when it exposes set_train_phase, and
every correction + intended base parameter is trainable BEFORE
optimizer construction; the base keeps base_lr_scale, the correction
its full refine lr; the refinement banner AND the resolved-pass
record (unit 41) state trainable parameter counts SEPARATELY for
correction trunk, correction head, and base — counts, not optimizer
membership, are the truth; byte-identical when refine is absent;
ResMLP and an already-joint freeze_trunk:false correction numerically
unchanged apart from the assertions/record. Red legs (gate under
gates/checks/, board-listed; state-transition legs CPU, the real
compiled/head path on the workstation): ResCNN control proves the
trunk frozen immediately before refinement; one analytic refine step
moves ALL THREE sets; the ResTRF counterpart (separate ownership
code); the MUTATION arm (no joint transition: base + head move, every
trunk gradient None, trunk tensors bitwise unchanged — must fail the
current implementation); freeze_trunk:false and ResMLP controls
unchanged; resolved-record readback equals the tensors that received
gradients; anchor interaction — a nonzero base anchor touches only
the intended base keys, and enabling the correction trunk must not
accidentally anchor it.

## UNIT 14 AMENDED (45M-47): a finite per-batch loss can publish an Inf epoch loss — increment (d)

Adjudicated + reproduced (Fable, 2026-07-12). The per-step finite
contract (training.py:2058) accepts a finite float32 loss, but the
epoch reduction overflows independently: the loop accumulates
`run_sum += loss.detach() * bs` (:2103) — the product is computed in
FLOAT32 before it reaches the accumulator, so a finite loss of 1e38
with bs=8 becomes Inf (float32 max 3.4028e38) even though run_sum is
float64 on CPU/CUDA (acc_dtype, :1781). On MPS the accumulator itself
is float32 (:1781). train_loss = (run_sum / run_n).item() (:2105)
publishes it: appended unguarded (:2139), printed (~:2246), persisted
in train_losses by the save path. Reproduced: np.float32(1e38)*8 ->
Inf; the float64-first mean is 9.999999680285692e37 (representable).

Contract (increment d of the finite unit):
1. The epoch mean must not overflow a float32 weighted sum. RULING:
   accumulate on the HOST in a python float (float64 on every
   backend): run_sum += float(loss.detach()) * bs. The finite
   contract already pays one host sync per step at :2058, so the host
   read adds no new sync — and this fixes MPS, whose device
   accumulator cannot be float64.
2. The completed epoch train_loss is REQUIRED finite before it is
   appended, reported, or persisted (_report_nonfinite, the shared
   message shape, naming the epoch).
3. General rule, recorded: a reduction's result must be checked;
   finite operands do not prove a finite reduction.
4. Ordinary finite-run numerics: the accumulator is diagnostic-only
   (selection reads the val metrics), so the float32-vs-float64
   product difference is within the existing tolerance; the training
   path itself is untouched.

Gate: EXTENDS the board-listed finite-contract check (not a new
gate). Drive the REAL training_loop_batched with >=2 full batches
whose differentiable scalar loss is finite 1e38 (finite gradients,
ordinary validation); the repaired loop returns a finite epoch loss
near 1e38. Mutation arm: restore the `loss.detach() * bs` ordering —
it must produce Inf and fail the leg. Workstation board run.

Unit 14 now closes on increments a+b (landed, a0d03f5) + c
(safe-sqrt, 45M-24, owed) + d (this amendment) + the extended gate.

## UNIT 55 (45M-46): repeated-training state isolation — transfer-refine sweeps are order- and worker-dependent

Adjudicated + chain-verified on HEAD (Fable, 2026-07-12),
reachability FIRST per the standard: validate_transfer SUPPORTS
transfer.refine on the cosmic-shear family (experiment.py:1368-1410;
only the cmb/grid/grid2d families refuse it, :1371-1375) — a
validated V1 feature. Nothing like the retracted 45M-43: no forbidden
key is needed to reach this state. The mutation chain, every link
confirmed on HEAD:
- from_config loads ONE transfer source into exp._transfer_base
  (:2232 / :2352 / :2475 / :2576); every exp.train() hands the SAME
  object to run_emulator.
- The refine stage (training.py:2941-2996) takes base_net =
  chi2fn.base_net (the same module), unfreezes it IN PLACE, sets
  chi2fn.set_live(True) (:2951 — the only set_live call in the file;
  never reset to False), and trains it jointly. No restore of the
  weights, the requires_grad flags, or the live mode afterward —
  neither run_emulator nor exp.train restores anything.
- Each exp.train() snapshots the base AS IT CURRENTLY STANDS
  (experiment.py:4461-4463) into _transfer_pretrained_base — after
  point 1 drifts the base, point 2's "pretrained" anchor/artifact
  reference is W_1, not the source artifact's W_0. The in-stage
  anchor clone (training.py:2945) drifts identically.
- All four repeated-training drivers reuse one EmulatorExperiment
  across points and never restore the base: tune (one staged exp
  closed over by objective — serial :364-366, and each parallel
  worker likewise), hyperparameter sweep (:131-171 worker,
  :271-327 serial), activation bakeoff (:138-156, :356-412), N-train
  sweep (:127-162, :411-464).

Consequences (verified plausible on the confirmed chain): sweep
results depend on point ORDER and on n_gpus / lane packing; a failed
point can leave a half-refined base for the next; Optuna trials are
history-dependent while the unit-53 manifest matches perfectly; an
N-train learning curve no longer compares sizes against one common
pretrained emulator. Every value stays finite and plausible — no
existing check can see it. In fixed-geometry loops chi2fn.live also
REMAINS True, so the next point's nominal frozen-correction stage is
not even in stage-1 mode; the N-train sweep rebuilds the loss (live
resets) but wraps the already-drifted base.

Contract:
1. Every repeated training point/trial starts from one immutable
   source state W_0 — independent of execution order, worker count,
   and prior failures/successes.
2. Capture pristine W_0 (parameters AND buffers) once per
   experiment/lane immediately after artifact load. Restore IN PLACE
   before every point so every existing chi2fn.base_net reference
   still points at the restored object — never rebuild a detached
   base without rewiring the loss (in-place restoration or complete
   experiment reconstruction are the only safe forms).
3. Restore the complete stage-1 runtime state at point entry:
   chi2fn.live False, base eval mode, cleared gradients, the original
   requires_grad flags; reset _transfer_pretrained_base so the point
   records/clones W_0, never a predecessor's drift.
4. N-train / activation-size loops restore before build_geometry;
   fixed-geometry hyperparameter/tune loops restore before loader
   construction and training.
5. A point failure passes through the same reset discipline in
   finally; the next point cannot inherit a partially updated base.
6. A point's refinement anchor reference must hash identical to the
   pristine source state; post-point drift belongs to that point
   only.
7. Persist the common source-state digest in the study/sweep identity
   record (reuse the artifact-manifest digest machinery — interlocks
   units 37 + 53); per-point/trial diagnostics can assert entry
   digest == the common digest.
8. Runs without transfer.refine stay byte-identical; frozen-only
   transfer, ordinary, NPCE, and finetune runs pay no semantic
   change.

Distinct from sweep-worker truth (unit 10) and study identity (unit
53): those prove the intended jobs completed under the intended
configuration; THIS unit proves each job began from the intended
model state.

Red legs: (1) two-point deterministic refine sweep — point-entry base
digest is W_0 at both points; current code must show W_0 then W_1
(the mutation witness); (2) reverse order [A,B] vs [B,A] with fixed
seeds — identical per-point final weights/metrics; (3) one lane vs
two lanes — identical per-point entry digests and results; (4) every
point's anchor reference == W_0, never the predecessor's final base;
(5) every point enters its correction stage with chi2fn.live False;
only refinement flips it; (6) failure leg — point A performs at least
one base update then raises; point B still begins at W_0; (7) one
fixed-geometry (hyperparam/tune-style) AND one rebuild-geometry
(N-train-style) leg; (8) mutation arm — omit the reset between two
points; the gate must observe point 2 entering with point 1's drifted
digest; (9) frozen-only / no-transfer controls unchanged.

Torch gate under gates/checks/, LISTED on the board, driving the real
repeated-driver paths (not a standalone reset helper); Vivian runs
the workstation leg. Placement: beside sweep-worker truth and the
unit-53 manifest; MUST land before any transfer-refine tune,
hyperparameter sweep, activation bakeoff, or N-train science curve is
trusted. Pipeline slot: after unit 52, before 22(+20).
